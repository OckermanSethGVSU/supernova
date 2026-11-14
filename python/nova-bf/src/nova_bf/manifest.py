"""Run manifest: what a brute-force run was, what it ran on, and how it went.

A result parquet's schema metadata (see `results.provenance`) answers "what is
this ground truth?" — corpus, queries, metric, k, precision, tie-break. It
deliberately says nothing about the RUN: how many ranks, which corpus files
this worker took, how long it took, what it ran on, what it cost. Those are
properties of an execution, not of the rows, and they are what someone asks
months later when re-running, attributing cost, or explaining why one GT took
2 hours and another 12.

So each phase also writes one JSON manifest next to its output, in the same
shape nova-embed's `_manifest.json` uses (source / destination / created_at /
compute / settings / counts / timing / output_files) plus the nova-bf-specific
blocks — `searches`, `sharding`, `tiebreak` — that make a GT run reproducible.

Writing it is best-effort: a manifest is a record OF the outputs, so failing to
write one must never fail a run whose outputs already landed (a multi-hour
sharded GT run is not worth losing to an S3 hiccup on a 4 KB JSON PUT).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import socket
import sys
import subprocess

from datetime import datetime, timezone

from nova_bf.config import BruteForceConfig, SearchSpec
from nova_bf.io import ParquetFile, Store
from nova_bf.results import queries_stem

logger = logging.getLogger(__name__)

# Bump when a field's MEANING changes (not when one is added) — consumers that
# parse manifests across runs need to tell a rename from a new key.
# The liveness fields under `kernel_usage.prune` are OPTIONAL within v1: they
# were added (and renamed once) during development, and no released manifest
# carries the earlier names, so their arrival is an addition, not a rename.
MANIFEST_VERSION = 1


def manifest_name(
    cfg: BruteForceConfig,
    phase: str,
    job_rank: int | None = None,
    num_jobs: int | None = None,
    search: str | None = None,
) -> str:
    """Where this run's manifest goes, under `cfg.output.path`.

    Keyed by the queries stem for the same reason `result_name`/`partial_dir`
    are: several configs can share one output prefix, and two runs' manifests
    overwriting each other would be worse than having none. A sharded compute
    run writes one manifest PER RANK (each describes only its own slice, files,
    and timings), collected in a directory rather than scattered across the
    output root next to the result parquets.
    """
    stem = f"_bf_manifest_{queries_stem(cfg.queries.path)}_{phase}"
    if search is not None:
        return f"{stem}/{search}.json"
    if job_rank is None or num_jobs is None:
        return f"{stem}.json"
    width = max(3, len(str(num_jobs - 1)))
    return f"{stem}/rank{job_rank:0{width}d}.json"


def detect_compute(device: str | None = None) -> dict:
    """Best-effort hardware fingerprint: instance type, region, GPU, torch.

    So a run's output is self-describing for cost attribution and
    reproducibility — "31 ranks, 2h" means nothing without knowing they were
    g5.8xlarge. Mirrors nova-embed's `_detect_compute` (AWS IMDSv2 + torch), so
    the two toolsets' manifests describe hardware the same way.

    Returns `{}` off-cloud / without torch and never raises: short timeouts,
    every exception swallowed.
    """
    info: dict = {}
    try:
        import urllib.request

        tok = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(tok, timeout=0.5).read().decode()

        def _imds(path: str) -> str:
            req = urllib.request.Request(
                f"http://169.254.169.254/latest/meta-data/{path}",
                headers={"X-aws-ec2-metadata-token": token},
            )
            return urllib.request.urlopen(req, timeout=0.5).read().decode()

        info["instance_type"] = _imds("instance-type")
        info["region"] = _imds("placement/region")
        info["availability_zone"] = _imds("placement/availability-zone")
    except Exception:
        pass
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
            # The denominator for this run's peak usage (see `gpu_peak`): a
            # peak is only actionable next to the card it had to fit in.
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_bytes"] = props.total_memory
            # WHICH card (nvidia-smi's `GPU-...` form, so it joins an
            # nvidia-smi log of the node) and which driver it ran under.
            uuid = getattr(props, "uuid", None)
            if uuid is not None:
                info["gpu_uuid"] = f"GPU-{uuid}"
            driver = _nvidia_driver()
            if driver:
                info["driver_version"] = driver
    except Exception:
        pass
    if device is not None:
        # What scoring ACTUALLY ran on. Not redundant with `gpu`: a CUDA box
        # whose driver failed at import time still lists a GPU here while
        # having scored on the CPU, and that gap explains an odd wall time.
        info["device"] = device
    return info


def _nvidia_driver(proc: str = "/proc/driver/nvidia/version") -> str:
    """The NVIDIA driver version ("Kernel Module  580.82.07" / "Open Kernel
    Module for x86_64  580.82.07" in /proc, else nvidia-smi); "" if neither
    says. Duplicated from the streaming harness's run.py (nvidia_driver)."""
    try:
        with open(proc) as f:
            m = re.search(r"Kernel Module(?: for \S+)?\s+(\d[\d.]*)", f.read())
        if m:
            return m.group(1)
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        lines = out.stdout.strip().splitlines() if out.returncode == 0 else []
        return lines[0].strip() if lines else ""
    except Exception:  # noqa: BLE001 - no nvidia-smi: no driver to report
        return ""


def _workspace_version(start: str) -> str | None:
    """The supernova workspace version from the root `Cargo.toml`, if this is a
    checkout rather than an installed wheel.

    nova-bf's own `[project] version` is a static `0.0.1` nobody bumps, so the
    workspace version is the only release-shaped number the toolset has — and
    it is what the git tags carry (`v0.0.12`), so it lines up with
    `git_describe` below. Walks up from the package directory; `None` when
    there is no Cargo.toml above it.
    """
    import tomllib

    path = os.path.abspath(start)
    for _ in range(8):  # src/nova_bf → … → repo root is 4 up; 8 is slack
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
        candidate = os.path.join(path, "Cargo.toml")
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "rb") as fh:
                ws = tomllib.load(fh).get("workspace", {})
        except Exception:  # noqa: BLE001 - an unreadable Cargo.toml is not fatal
            continue
        # `[workspace] version` today; `[workspace.package] version` is the
        # more common spelling, so accept either rather than break on a move.
        # A crate's own Cargo.toml carries neither: keep walking up rather than
        # concluding there is no workspace above it.
        version = ws.get("version") or ws.get("package", {}).get("version")
        if version:
            return version
    return None


def kernel_usage(
    prune_launches: int,
    live_stats: dict | None = None,
    two_pass_policy: str = "auto",
) -> dict:
    """Report GPU fast paths that actually ran.

    Each entry records whether the path was permitted, how many times it ran,
    and why it was unavailable. Prune effectiveness is included only when
    measured.
    """
    from nova_bf import merge_triton, topk_triton

    prune = {
        "permitted": not os.environ.get("NOVA_BF_NO_PRUNE"),
        # Prune has no kernel and cannot decline at runtime; this counts the
        # (slice, search) pairs where a threshold was applied -- per search,
        # so it equals the sum of `by_search[*].slices`, not the slice count.
        "launches": prune_launches,
        "unavailable": None,
    }
    if live_stats:
        # These are (query, slice) decisions, not corpus rows.
        total_live = sum(v["live"] for v in live_stats.values())
        total_rows = sum(v["rows"] for v in live_stats.values())
        prune["live_query_slice_rows"] = total_live
        prune["query_slice_rows"] = total_rows

        # Compare live_fraction only across runs with the same batch sizing.
        prune["live_fraction"] = (
            round(total_live / total_rows, 6) if total_rows else None
        )
        prune["by_search"] = {
            name: {
                "live_query_slice_rows": v["live"],
                "query_slice_rows": v["rows"],
                "slices": v["slices"],
                "live_fraction": (
                    round(v["live"] / v["rows"], 6) if v["rows"] else None
                ),
            }
            for name, v in live_stats.items()
        }

    from nova_bf import twopass

    tp = twopass.stats()
    # `rows_live / rows_full` is what the two-pass actually skipped; the gap
    # between it and `rows_padded / rows_full` is what the GEMM-height padding
    # cost to keep cuBLAS on one kernel (see
    # docs/brute-force/claude_only/two-pass-bound.md). Tiles-mode slices are
    # not padded (they book `rows_padded = rows_live`), so in a tiles run
    # `padded_fraction` reflects only the slice-mode slices' padding.
    tp["live_fraction"] = (
        round(tp["rows_live"] / tp["rows_full"], 6) if tp["rows_full"] else None
    )
    tp["padded_fraction"] = (
        round(tp["rows_padded"] / tp["rows_full"], 6) if tp["rows_full"] else None
    )
    # Tiles mode (`rescore == "tiles"`): the (query, tile) pairs re-scored out
    # of those spanned by the slices that were actually tile-rescored
    # (including all-pruned ones). NOT the engage gate's input: slices that
    # bailed past the tile crossover, fell back to slice mode (kernel
    # declined, no pass one) or ran out of memory are excluded here but feed
    # the gate's hint, so this reads lower than what the gate saw.
    tp["tile_live_fraction"] = (
        round(tp["tile_pairs_live"] / tp["tile_pairs_full"], 6)
        if tp["tile_pairs_full"] else None
    )
    tp["tile_rescore_kernel"] = _tile_rescore_usage()
    # THE COUNTER IDENTITY, checked on the real run rather than only in tests.
    # Recorded, never raised: a manifest that fails to write costs a rank its
    # provenance, which is worse than an inconsistent counter.
    ran = tp["slices_fused"] + tp["slices_unfused"]
    booked = tp["slices_twopass"] + tp["slices_discarded"]
    tp["counters_consistent"] = ran == booked
    if ran != booked:
        logger.warning(
            "two-pass counters are inconsistent: %d pass-one runs "
            "(fused %d + unfused %d) against %d booked (twopass %d + "
            "discarded %d). One of them is miscounting; the run's RESULTS are "
            "unaffected, but the manifest's accounting of two-pass work "
            "cannot be trusted.",
            ran, tp["slices_fused"], tp["slices_unfused"], booked,
            tp["slices_twopass"], tp["slices_discarded"],
        )
    # `off` is an explicit reproducible run policy from YAML. The environment
    # variable remains a diagnostic/temporary kill switch and wins over auto.
    tp["policy"] = two_pass_policy
    tp["permitted"] = (
        two_pass_policy == "auto" and not os.environ.get("NOVA_BF_NO_TWOPASS")
    )
    # `launches`: two-pass slices that still had a pass two to run (one per
    # slice, slice- or tiles-mode). All-pruned slices are in `slices_twopass`
    # but not here, so `slices_twopass - launches` is the slices pass one
    # pruned completely.
    tp["launches"] = tp.pop("gemms", 0)


    return {
        "prune": prune,
        "fold_kernel": merge_triton.usage(),
        "topk_kernel": topk_triton.usage(),
        "twopass": tp,
        "fused_rowmax_kernel": twopass.fuse_usage(),
    }


def _tile_rescore_usage() -> dict:
    """The tiles-mode pass-two kernel: launches and why it is unavailable."""
    from nova_bf import tile_rescore

    return {"permitted": not os.environ.get("NOVA_BF_NO_TILE_KERNEL"),
            **tile_rescore.usage()}


def code_versions() -> dict:
    """WHICH CODE produced this ground truth: the supernova version and git
    commit, plus the versions of the libraries whose numerics it depends on.

    The scoring and tie-break kernels are the ground truth — a change to the
    top-K merge or the tie-break packing changes which of two equal-scoring
    docs wins, and nothing in a result file records the revision that decided
    it. The package version alone cannot: it is a static `0.0.1` that nobody
    bumps per commit, so the commit sha is the only real answer.

    `git_dirty` is scoped to the nova-bf package directory, not the whole
    repo — an unrelated edit elsewhere in the monorepo says nothing about the
    code that ran, while an uncommitted edit HERE means the sha above is a
    lie by itself.

    Everything is best-effort and returns partial: a wheel install is not a git
    checkout, and git may not exist on the box at all.
    """
    info: dict = {}
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        supernova = _workspace_version(pkg_dir)
        if supernova:
            info["supernova_version"] = supernova
    except Exception:  # noqa: BLE001
        pass
    try:
        from importlib.metadata import version

        info["nova_bf_version"] = version("nova-bf")
    except Exception:
        pass
    info["python_version"] = platform.python_version()
    for mod in ("numpy", "pyarrow"):
        try:
            info[f"{mod}_version"] = __import__(mod).__version__
        except Exception:
            pass
    try:
        def _git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", pkg_dir, *args],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()

        # Only trust git if the repository it answers about actually CONTAINS
        # this package.
        toplevel = os.path.realpath(_git("rev-parse", "--show-toplevel"))
        if os.path.commonpath([toplevel, os.path.realpath(pkg_dir)]) != toplevel:
            raise RuntimeError("git toplevel does not contain the nova-bf package")

        info["git_commit"] = _git("rev-parse", "HEAD")
        # e.g. "v0.0.12-9-g659c42c-dirty" — the release-relative form of the
        # sha above, and the line a human reads first.
        info["git_describe"] = _git("describe", "--tags", "--always", "--dirty")
        # Fleet jobs deploy a branch (`@master`), so the branch is what someone
        # would have to check out to reproduce this.
        info["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        info["git_dirty"] = bool(_git("status", "--porcelain", "-uno", "--", pkg_dir))
    except Exception:
        pass  # not a checkout, or no git — the versions above still stand
    return info


def job_identity() -> dict:
    """Which launch and which host produced this manifest.

    A 64-rank run leaves 64 manifests, and the reason to open one is that
    something about that rank looks wrong — at which point you need its log,
    which lives under the SkyPilot task/cluster id. Without these the manifest
    is an orphan: it says a rank did something odd, and nothing says where to
    go read about it.
    """
    info: dict = {}
    for var in (
        "SKYPILOT_TASK_ID",
        "SKYPILOT_CLUSTER_NAME",
        "SKYPILOT_JOB_ID",
        "SKYPILOT_JOB_RANK",
        "SKYPILOT_NODE_RANK",
        "SKYPILOT_NUM_NODES",
    ):
        value = os.environ.get(var)
        if value:  # absent off-SkyPilot; a missing key beats a null
            info[var.lower()] = value
    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        pass
    info["pid"] = os.getpid()
    return info


def corpus_fingerprint(files: list[ParquetFile]) -> dict:
    """A hash of the corpus file list, IN THE ORDER the run indexed it.

    This is an id-scheme fingerprint, not a data checksum. With no
    `corpus.id_column`, hit ids are `make_point_id(file_key, row)` and the
    global file index also drives the ordinal tie-break — so the ORDER of this
    list is load-bearing. Add, remove or rename one corpus file and every later
    file's index shifts: the ground truth silently stops joining to a
    collection loaded from the other list, and recall reads near zero with
    nothing in either artifact to explain it. Comparing this hash is the cheap
    way to prove two runs saw the same corpus, and it catches
    `corpus.include`/`exclude` drift between ranks for free.

    Hashed over the loader `key` (the s3 object key / absolute local path),
    because that is exactly what `make_point_id` consumes — not `read_path`,
    which carries the bucket on s3 and would differ for the same logical
    corpus. `first`/`last` are carried in the clear so a human can eyeball
    which corpus this was without resolving a hash.
    """
    keys = [f.key for f in files]
    return {
        "files": len(keys),
        "sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "first": keys[0] if keys else None,
        "last": keys[-1] if keys else None,
    }


def search_entry(spec: SearchSpec) -> dict:
    """The nova-bf-specific core of a manifest: what this search actually was.

    `filter` is dumped in full (not just a `filtered: true` flag, which is all
    the parquet metadata can afford) because the filter IS the search for a
    filtered GT — recall numbers from two runs are comparable only if their
    predicates were identical, and a YAML edit between runs is otherwise
    invisible in the artifacts.
    """
    return {
        "name": spec.name,
        "vector_type": spec.vector_type,
        "metric": spec.metric,
        "k": spec.k,
        "filter": (
            None if spec.filter is None
            else spec.filter.model_dump(mode="json", exclude_defaults=True)
        ),
        "rows": (
            None if spec.rows is None
            else spec.rows.model_dump(mode="json")
        ),
    }


def source_block(cfg: BruteForceConfig) -> dict:
    """Corpus + queries: the inputs, and the columns actually read from them."""
    return {
        "corpus": {
            "path": cfg.corpus.path,
            "include": cfg.corpus.include,
            "exclude": cfg.corpus.exclude,
            "id_column": cfg.corpus.id_column,
            "dense_column": cfg.corpus.dense_column,
            "sparse_column": cfg.corpus.sparse_column,
            "multivector_column": cfg.corpus.multivector_column,
        },
        "queries": {
            "path": cfg.queries.path,
            "id_column": cfg.queries.id_column,
            "payload_fields": list(cfg.queries.payload_fields),
            "dense_column": cfg.queries.dense_column,
            "sparse_column": cfg.queries.sparse_column,
            "multivector_column": cfg.queries.multivector_column,
        },
    }


def base_manifest(cfg: BruteForceConfig, phase: str, device: str | None = None) -> dict:
    """The fields every phase's manifest carries, in a stable key order."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source_block(cfg),
        "destination": cfg.output.path,
        "compute": detect_compute(device),
        "code": code_versions(),
        "job": job_identity(),
        # Run-level knobs, resolved. CLI overrides (`--io-workers` etc.) are
        # merged in by the caller, so this is what RAN, not what the YAML said.
        "params": cfg.params.model_dump(mode="json"),
        # Which tie-break rule decided exact ties — the same value stamped into
        # the parquet metadata, repeated here so a manifest stands alone.
        "tiebreak": cfg.params.tiebreak,
    }


def gpu_peak(device: str | None) -> dict:
    """How much GPU memory this run actually needed, at its high-water mark.

    Sizing a fleet is the recurring question here — `dense_batch_size`,
    `multivector_budget_mib` and the query matrix all trade against one
    card's memory, and the failure mode is an OOM hours into a run on an
    instance one size too small. The logs say what was configured; only this
    says what it cost. `allocated` is what nova-bf's tensors held, `reserved`
    is what the caching allocator took from the driver — reserved is the one
    to compare against `gpu_total_bytes`, since that is what actually has to
    fit.

    `{}` on CPU or without torch; never raises.
    """
    if device != "cuda":
        return {}
    try:
        import torch

        return {
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:  # noqa: BLE001
        return {}


# The CUDA library families the paper stack pins (benchmarks/streaming/
# paper_results/env/versions.env), each with its own version call (None: no
# version API). Duplicated, not shared, in the streaming harness's cudalibs.py.
#   *GetProperty(MAJOR_VERSION=0 / MINOR_VERSION=1 / PATCH_LEVEL=2, int*) -> "M.m.p"
#   cudaRuntimeGetVersion(int*) -> "M.m"; nvrtcVersion / nvJitLinkVersion(M*, m*) -> "M.m"
_CUDA_FAMILIES = {
    "libcublas": "cublasGetProperty",
    "libcublasLt": "cublasLtGetProperty",
    "libcusparse": "cusparseGetProperty",
    "libcusolver": "cusolverGetProperty",
    "libcufft": "cufftGetProperty",
    "libcurand": "curandGetProperty",
    "libcudart": "cudaRuntimeGetVersion",
    "libnvrtc": "nvrtcVersion",
    "libnvrtc-builtins": None,
    "libnvJitLink": "nvJitLinkVersion",
}
# libcublasLt.so.13 / libcusparse.so.12.7.10.12 / libnvrtc-builtins.so.13.2;
# not libcusolverMg / libcufftw / libnvrtc.alt (other libraries)
_CUDA_LIB = re.compile(r"/(lib(?:cublasLt|cublas|cusparse|cusolver|cufft|curand|cudart|"
                       r"nvrtc-builtins|nvrtc|nvJitLink|nvjitlink))\.so(?:\.[0-9][^/\s]*)?$")


def _mapped_cuda_libraries(maps_text: str | None = None) -> dict[str, list[str]]:
    """family -> distinct real paths of every copy mapped into this process
    (/proc/self/maps, or `maps_text`), first-seen order. The maps are
    address-ordered: with two copies, the order says nothing about which was
    loaded first. Raises OSError without /proc."""
    if maps_text is None:
        with open("/proc/self/maps") as f:
            maps_text = f.read()
    fam: dict[str, list[str]] = {}
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        m = _CUDA_LIB.search(parts[5].strip())
        if m:
            key = m.group(1)
            key = "libnvJitLink" if key.lower() == "libnvjitlink" else key
            path = os.path.realpath(parts[5].strip())
            seen = fam.setdefault(key, [])
            if path not in seen:
                seen.append(path)
    return {k: fam[k] for k in sorted(fam)}


def _cuda_library_version(fam: str, path: str) -> str:
    """The version the mapped copy at `path` reports about itself, through a
    handle on that very file: dlopen(RTLD_NOLOAD) returns the already-mapped
    object and never maps a new copy. No GPU needed. "" when it is not
    mapped, has no version call, or the call fails."""
    fn_name = _CUDA_FAMILIES.get(fam)
    if fn_name is None:
        return ""
    try:
        import ctypes

        lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
        fn = getattr(lib, fn_name)
        if fn_name.endswith("GetProperty"):
            vals = []
            for prop in (0, 1, 2):    # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                v = ctypes.c_int()
                if fn(prop, ctypes.byref(v)) != 0:
                    return ""
                vals.append(v.value)
            return ".".join(map(str, vals))
        if fn_name == "cudaRuntimeGetVersion":
            v = ctypes.c_int()
            if fn(ctypes.byref(v)) != 0:
                return ""
            return f"{v.value // 1000}.{v.value % 1000 // 10}"
        a, b = ctypes.c_int(), ctypes.c_int()
        if fn(ctypes.byref(a), ctypes.byref(b)) != 0:
            return ""
        return f"{a.value}.{b.value}"
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        return ""


def loaded_cuda_libraries(device: str, maps_text: str | None = None) -> dict:
    """Report CUDA library paths and versions from the files actually loaded
    by the process, not just installed libraries.

    Report duplicate copies separately, since they violate the pinned CUDA
    stack. Include the cuBLAS version only when exactly one copy is loaded.
    Return empty results for CPU runs or unreadable process maps. Never raises.
    """
    if str(device).split(":", 1)[0] != "cuda":
        return {}
    out: dict = {"cuda_libraries": {}, "cuda_library_duplicates": {}, "cublas_version": ""}
    try:
        libs = _mapped_cuda_libraries(maps_text)
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        return out
    for fam, paths in libs.items():
        copies = [{"path": p, "version": _cuda_library_version(fam, p)} for p in paths]
        if len(copies) == 1:
            out["cuda_libraries"][fam] = copies[0]
        else:
            out["cuda_library_duplicates"][fam] = copies
    out["cublas_version"] = out["cuda_libraries"].get("libcublas", {}).get("version", "")
    return out


ALLOW_DUPLICATE_CUDA_LIBS_ENV = "NOVA_BF_ALLOW_DUPLICATE_CUDA_LIBS"


def refuse_duplicate_cuda_libraries(device: str, maps_text: str | None = None) -> None:
    """Reject CUDA runs that load multiple copies of the same CUDA library,
    Allow overrides for debugging via NOVA_BF_ALLOW_DUPLICATE_CUDA_LIBS=1. 
    Libraries loaded later are checked in the manifest. Skip CPU runs."""
    if os.environ.get(ALLOW_DUPLICATE_CUDA_LIBS_ENV):
        return
    dups = loaded_cuda_libraries(device, maps_text).get("cuda_library_duplicates") or {}
    if dups:
        lines = "\n".join(f"  {fam}: " + ", ".join(c["path"] for c in copies)
                          for fam, copies in sorted(dups.items()))
        raise RuntimeError(
            "two copies of a CUDA library are mapped into this process, so the run "
            f"need not use the copy it records:\n{lines}\nImport torch first and keep "
            "other copies off LD_LIBRARY_PATH (the paper env's activate.sh does); "
            f"{ALLOW_DUPLICATE_CUDA_LIBS_ENV}=1 runs anyway, for diagnosis."
        )


def host_peak() -> dict:
    """Peak resident set size for this process, in bytes.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB here, macOS reports bytes. Normalize so the field
        # is comparable with `peak_gpu_allocated_bytes` and cannot be misread
        # by a factor of 1024.
        scale = 1 if sys.platform == "darwin" else 1024
        return {"peak_host_rss_bytes": int(peak) * scale}
    except Exception:  # noqa: BLE001
        return {}


def write(store: Store, filename: str, payload: dict) -> str | None:
    """Write `payload` as JSON to `store/filename`. Never raises.

    Returns the path written, or None if writing failed (logged as a warning,
    not an error: the run's real outputs are already on disk by the time this
    is called, and losing the manifest must not fail them).
    """
    try:
        # default=str so an unexpected value (a date in a filter dump, say)
        # degrades to its string form instead of sinking the whole manifest.
        data = json.dumps(payload, indent=2, default=str).encode()
        path = store.write_bytes(filename, data)
        logger.info("wrote run manifest %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - a manifest must never fail a run
        logger.warning("could not write run manifest %s: %s", filename, exc)
        return None
