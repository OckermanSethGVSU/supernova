"""Record and validate the environment used for each benchmark run.

Tracks the paper's pinned software stack, environment settings, hardware,
driver, and library versions. Checks for conflicting CUDA libraries,
unpinned dependencies, and incorrect bench or nova-bf import paths.

Provides shared environment checks for the harness and env/check.py,
and records the configuration used by each benchmark result.
"""

import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import re
import subprocess
import sys

from bench.record import cudalibs

# Import FAISS and related dependencies only when needed. CUDA libraries must
# be loaded through cudalibs.torch_first() before FAISS is imported.

# Paper stack: env/paper_run.sh validates the pinned environment and records
# the software, code, and environment hashes and the run directory.
# The harness checks these values for CUDA runs and includes them in each result.

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # clean_benchmarks/
VERSIONS_ENV = os.path.join(ROOT, "env", "versions.env")
STACK_ENV, CODE_ENV, PREFIX_ENV, RUN_DIR_ENV = (
    "PAPER_STACK_SHA", "PAPER_CODE_SHA", "PAPER_ENV_PREFIX", "PAPER_RUN_DIR")
NO_IMDS_ENV = "HARNESS_NO_IMDS"   # set (tests): no instance metadata lookup at all
IMDS = "http://169.254.169.254"
# the stamp env/build.sh writes into the env: `sha256sum` of each cuVS patch applied, in order
CUVS_PATCH_STAMP = ".cuvs_patches.sha256"
# where the cuVS source patches live (env/build.sh's CUVS_PATCH_DIR default)
LIBRARY_PATCHES_DIRS = (os.path.join(ROOT, "baselines", "cuvs", "patches"),)


def library_patches_dir():
    """The first of LIBRARY_PATCHES_DIRS that exists (the checkout's when neither does)."""
    return next((d for d in LIBRARY_PATCHES_DIRS if os.path.isdir(d)), LIBRARY_PATCHES_DIRS[-1])

# environment that changes how the CPU / GPU side runs (threads, affinity,
# MKL dispatch; CUDA launch / allocator / determinism switches)
THREAD_ENV_PREFIXES = ("OMP_", "KMP_", "MKL_", "GOMP_", "OPENBLAS_")
CUDA_ENV_PREFIXES = ("CUDA_", "PYTORCH_", "TORCH_", "CUBLAS_", "CUBLASLT_", "NVIDIA_TF32")

# OpenMP runtimes and BLAS / threading libraries (conda FAISS links MKL + LLVM libomp, pip torch
# libgomp: both may be mapped in one process). Matches e.g. libgomp-a34b3233.so.1, libomp.so,
# libiomp5.so, libmkl_core.so.2, libscipy_openblas64_-ff651d7f.so, libtbb.so.12
_THREAD_LIB = re.compile(r"/(lib(?:gomp|iomp5|omp|mkl_[a-z0-9_]+|scipy_openblas[0-9]*|"
                         r"openblas[a-z0-9]*|tbb[a-z0-9]*))(?:-[0-9a-f]{6,}|_-[0-9a-f]{6,})?"
                         r"\.so(?:\.[0-9][^/\s]*)?$")
OPENMP_RUNTIMES = ("libgomp", "libiomp5", "libomp")
_LIBCUVS = re.compile(r"/(libcuvs(?:_c)?)\.so(?:\.[0-9][^/\s]*)?$")


def versions_sha(path=VERSIONS_ENV):
    """versions.env's stack hash, computed exactly as env/activate.sh and env/build.sh do:

        grep -v '^STACK_FROZEN_ON=' versions.env | sed 's/[[:space:]]*#.*$//' \\
            | grep -v '^[[:space:]]*$' | sha256sum

    Comments and the freeze date are left out (filling in STACK_FROZEN_ON is not a new
    stack). None when the file cannot be read."""
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "surrogateescape")
    except OSError:
        return None
    out = []
    for line in text.split("\n"):
        if line.startswith("STACK_FROZEN_ON="):
            continue
        line = re.sub(r"[ \t\r\f\v]*#.*$", "", line, count=1)
        if re.fullmatch(r"[ \t\r\f\v]*", line):
            continue
        out.append(line + "\n")
    return hashlib.sha256("".join(out).encode("utf-8", "surrogateescape")).hexdigest()


def stack_problem(versions_path=VERSIONS_ENV, environ=None, prefix=None):
    """Why this process is not on the paper stack, or None when it is.

    On the stack means: PAPER_STACK_SHA is set (the launcher ran and env/check.py passed),
    it equals this tree's versions.env hash (not a stale value from another stack), and this
    interpreter is the launcher's env (PAPER_ENV_PREFIX)."""
    env = os.environ if environ is None else environ
    got = env.get(STACK_ENV, "")
    launch = ("launch the job through env/paper_run.sh (it activates the "
              "pinned env, runs env/check.py and exports PAPER_STACK_SHA)")
    if not got:
        return f"PAPER_STACK_SHA is not set: this is not a paper-stack run; {launch}"
    if env.get("PAPER_LAUNCH_STUBBED"):
        return ("PAPER_LAUNCH_STUBBED is set: the launcher ran with a stub activate.sh / "
                f"env/check.py (tests only), so the stack was never checked; {launch}")
    want = versions_sha(versions_path)
    if want is not None and got != want:
        return (f"PAPER_STACK_SHA={got} but this harness's versions.env ({versions_path}) "
                f"hashes to {want}: the job was launched for another stack; {launch}")
    want_prefix = env.get(PREFIX_ENV, "")
    here = os.path.realpath(prefix or sys.prefix)
    if not want_prefix:
        return f"PAPER_ENV_PREFIX is not set; {launch}"
    if os.path.realpath(want_prefix) != here:
        return (f"this interpreter's prefix {here} is not the paper env "
                f"(PAPER_ENV_PREFIX={want_prefix}): every engine must run from that env")
    return None


def run_fields(environ=None):
    """The launcher's values, as the run sees them ("" when not launched by it)."""
    env = os.environ if environ is None else environ
    return {"paper_stack_sha": env.get(STACK_ENV, ""), "paper_code_sha": env.get(CODE_ENV, ""),
            "paper_run_dir": env.get(RUN_DIR_ENV, "")}


def env_str(prefixes, environ=None):
    """``K=V;K=V`` of every environment variable starting with one of ``prefixes``, sorted."""
    env = os.environ if environ is None else environ
    return ";".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith(prefixes))


def thread_libs(maps_text=None):
    """name -> real paths of every OpenMP / MKL / OpenBLAS / TBB library mapped into this
    process (from /proc/self/maps, or ``maps_text``), in first-seen order. {} without /proc."""
    if maps_text is None:
        try:
            with open("/proc/self/maps") as f:
                maps_text = f.read()
        except OSError:
            return {}
    out = {}
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        m = _THREAD_LIB.search(parts[5].strip())
        if m:
            path = os.path.realpath(parts[5].strip())
            seen = out.setdefault(m.group(1), [])
            if path not in seen:
                seen.append(path)
    return {k: out[k] for k in sorted(out)}


def thread_libs_fields(libs=None):
    """{"thread_libs": "name=path;..." (every copy), "openmp_runtimes": the OpenMP runtimes
    mapped, comma-separated}. More than one runtime means two thread pools in one process."""
    libs = thread_libs() if libs is None else libs
    return {"thread_libs": ";".join(f"{k}={p}" for k, ps in libs.items() for p in ps),
            "openmp_runtimes": ",".join(k for k in libs if k in OPENMP_RUNTIMES)}


def _imds_open():
    import urllib.request
    # IMDS is link-local: never through an http(s)_proxy from the environment
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


_EC2_KEYS = (("instance_id", "instance-id"), ("instance_type", "instance-type"),
             ("availability_zone", "placement/availability-zone"))
_ec2_cache = None


def ec2_identity(opener=None, timeout=0.3, cache=True):
    """{"instance_id", "instance_type", "availability_zone"} from EC2 IMDSv2 (short timeouts).

    All "" off EC2, on any error, or with $HARNESS_NO_IMDS set. Looked up once per process
    (``cache``)."""
    global _ec2_cache
    blank = {k: "" for k, _ in _EC2_KEYS}
    if os.environ.get(NO_IMDS_ENV):
        return blank
    if cache and _ec2_cache is not None:
        return dict(_ec2_cache)
    out = dict(blank)
    try:
        import urllib.request
        open_ = opener or _imds_open()
        tok = urllib.request.Request(f"{IMDS}/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = open_(tok, timeout=timeout).read().decode().strip()
        for key, path in _EC2_KEYS:
            req = urllib.request.Request(f"{IMDS}/latest/meta-data/{path}",
                                         headers={"X-aws-ec2-metadata-token": token})
            out[key] = open_(req, timeout=timeout).read().decode().strip()
    except Exception:  # noqa: BLE001 - off EC2, or IMDS refused: blank, never fatal
        out = blank
    if cache:
        _ec2_cache = dict(out)
    return out


def conda_packages(names, prefix=None):
    """``name-version-build`` of each conda package in ``prefix`` (default: this
    interpreter's) named in ``names``, sorted; [] when it is not a conda env."""
    meta = os.path.join(prefix or sys.prefix, "conda-meta")
    try:
        files = os.listdir(meta)
    except OSError:
        return []
    out = []
    for f in files:
        if f.endswith(".json"):
            name = f[:-len(".json")].rsplit("-", 2)[0]
            if name in names:
                out.append(f[:-len(".json")])
    return sorted(out)


FAISS_CONDA = ("faiss-gpu", "faiss-gpu-cuvs", "faiss-cpu", "faiss", "libfaiss", "libfaiss-avx2",
               "libfaiss-avx512")
FAISS_PIP = ("faiss-gpu-cu12", "faiss-gpu-cu13", "faiss-gpu", "faiss-cpu", "faiss")


def faiss_build(faiss, prefix=None):
    """{"faiss_build", "faiss_compile_options"} of the installed FAISS.

    faiss_build is the package(s) with build string, e.g.
    ``conda:faiss-gpu-1.16.0-py3.12_hc84be16_0_cuda13.2,libfaiss-...`` or
    ``pip:faiss-gpu-cu12==1.14.1``; the options say SIMD level, GPU and cuVS routing."""
    conda = conda_packages(FAISS_CONDA, prefix)
    build = "conda:" + ",".join(conda) if conda else ""
    if not build:
        for dist in FAISS_PIP:
            try:
                build = f"pip:{dist}=={md.version(dist)}"
                break
            except md.PackageNotFoundError:
                continue
            except Exception:  # noqa: BLE001
                break
    try:
        opts = faiss.get_compile_options().strip()
    except Exception:  # noqa: BLE001 - an old FAISS without it
        opts = ""
    return {"faiss_build": build, "faiss_compile_options": opts}


def check_env_sha256(run_dir=None):
    """{path: sha256} from the launcher's check_env.json (the libcuvs / libfaiss / shim
    hashes env/check.py took at job start), {} without one."""
    run_dir = run_dir if run_dir is not None else os.environ.get(RUN_DIR_ENV, "")
    if not run_dir:
        return {}
    try:
        with open(os.path.join(run_dir, "check_env.json"), encoding="utf-8") as f:
            return dict(json.load(f).get("sha256") or {})
    except Exception:  # noqa: BLE001
        return {}


def cuvs_patches(prefix=None):
    """``name=sha256;...`` of the patches the env's cuVS was built with (from the
    CUVS_PATCH_STAMP file env/build.sh writes), "" without a stamp."""
    try:
        with open(os.path.join(prefix or sys.prefix, CUVS_PATCH_STAMP), encoding="utf-8") as f:
            rows = [l.split() for l in f if l.strip()]
    except OSError:
        return ""
    return ";".join(f"{os.path.basename(r[-1])}={r[0]}" for r in rows if len(r) >= 2)


def cuvs_fields(maps_text=None, run_dir=None, prefix=None):
    """{"cuvs_version", "libcuvs", "libcuvs_sha256", "cuvs_patches"}, blank unless cuvs is
    already imported in this process.

    This never imports cuvs: recording a FAISS row must not map libcuvs. libcuvs is the real
    path of each mapped libcuvs / libcuvs_c. Its sha256 comes from env/check.py at job start
    (no 1 GB hash per run), "" when check.py did not hash that file."""
    out = dict.fromkeys(("cuvs_version", "libcuvs", "libcuvs_sha256", "cuvs_patches"), "")
    mod = sys.modules.get("cuvs")
    if mod is None:
        return out
    out["cuvs_version"] = str(getattr(mod, "__version__", ""))
    if maps_text is None:
        try:
            with open("/proc/self/maps") as f:
                maps_text = f.read()
        except OSError:
            maps_text = ""
    paths = []
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and _LIBCUVS.search(parts[5].strip()):
            p = os.path.realpath(parts[5].strip())
            if p not in paths:
                paths.append(p)
    hashes = check_env_sha256(run_dir)
    out["libcuvs"] = ";".join(paths)
    out["libcuvs_sha256"] = ";".join(hashes.get(p, "") for p in paths) if any(
        p in hashes for p in paths) else ""
    out["cuvs_patches"] = cuvs_patches(prefix)
    return out


# =============================================================================================
# 2. The per-row record
# =============================================================================================

def _duckdb_version():
    import duckdb
    return duckdb.__version__


def nvidia_smi_lines(field, dev=None):
    """``nvidia-smi --query-gpu=<field>`` for GPU ``dev`` (nvidia-smi index or UUID; None =
    every GPU): one line per GPU, [] when nvidia-smi is missing or fails. Call it before the
    passes only."""
    cmd = ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"]
    try:
        out = subprocess.run(cmd + (["-i", dev] if dev else []), capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.stdout.strip().splitlines()] if out.returncode == 0 else []


def nvidia_smi(field, dev=None):
    """One ``nvidia-smi --query-gpu`` field for GPU ``dev`` (nvidia-smi index or UUID), ""
    when nvidia-smi is missing or fails.

    The default ``dev`` is the first ``CUDA_VISIBLE_DEVICES`` entry read as an nvidia-smi
    index: fine for fields every GPU shares (the driver). To find the GPU CUDA uses, see
    ``smi_device``."""
    dev = dev or (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
    lines = nvidia_smi_lines(field, dev)
    return lines[0] if lines else ""


def smi_device():
    """``(dev, warning)``: the nvidia-smi id (index or UUID) of CUDA's device 0 in this
    process, for a run that cannot ask CUDA itself (no torch).

    A ``CUDA_VISIBLE_DEVICES`` entry is a CUDA ordinal, not an nvidia-smi index: CUDA orders
    devices fastest first (identical GPUs in no promised order) unless
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, which is nvidia-smi's order. So a UUID entry is used as
    is; an ordinal (default 0) is trusted only with ``PCI_BUS_ID`` or when nvidia-smi lists at
    most one GPU. Otherwise ``dev`` is None and ``warning`` says why and what to set."""
    first = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if first.startswith(("GPU-", "MIG-")):
        return first, ""
    ordinal = first or "0"
    if os.environ.get("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID":
        return ordinal, ""
    gpus = nvidia_smi_lines("name")
    if len(gpus) <= 1:
        return ordinal, ""
    return None, (f"WARNING: cannot tell which GPU CUDA device {ordinal} is: nvidia-smi lists "
                  f"{len(gpus)} GPUs ({', '.join(sorted(set(gpus)))}) and CUDA's default order "
                  "only puts the fastest first (identical GPUs in no promised order); "
                  "gpu_name / gpu_uuid / gpu_index left blank (so "
                  "gpu_busy.py cannot pick this run's GPU). Set CUDA_DEVICE_ORDER=PCI_BUS_ID "
                  "or put GPU UUIDs in CUDA_VISIBLE_DEVICES.")


def nvidia_driver(path=None):
    """The NVIDIA driver version from /proc/driver/nvidia/version ("Kernel Module  580.82.07"
    or "Open Kernel Module for x86_64  580.82.07"), else from nvidia-smi; "" if neither says."""
    from bench.record import schema
    try:
        with open(path or schema.NVIDIA_PROC) as f:
            m = re.search(r"Kernel Module(?: for \S+)?\s+(\d[\d.]*)", f.read())
        if m:
            return m.group(1)
    except OSError:
        pass
    return nvidia_smi("driver_version")


def provenance(args, backend):
    """What a row was measured with, beyond the engine.

    The corpus / query source; the library versions on the shared code path (torch for the
    resident H2D / widen, pyarrow for reads, numpy, and the faiss whose ResultHeap merges every
    engine's results, which in a cuVS venv is not the engine); the GPU and driver. torch
    columns are n/a without torch; GPU columns are blank on the CPU backend.

    The GPU is named by ``gpu_name``, ``gpu_uuid`` (``GPU-...``) and ``gpu_index``
    (nvidia-smi's index, which need not be CUDA's), so gpu_busy.py can pick this GPU's lines
    out of a whole node's nvidia-smi log. With torch, the UUID is torch's for the run's device
    and the index is looked up by it. Without torch, all three come from nvidia-smi for
    ``smi_device()``'s GPU, and are blank (with a warning) when that GPU cannot be told."""
    import faiss
    import numpy as np
    import pyarrow as pa
    torch = getattr(backend, "torch", None)
    gpu = backend.name in ("gpu", "cuvs", "spare", "pylate")
    gpu_name = gpu_uuid = gpu_index = ""
    if gpu:
        if torch is not None and getattr(backend, "_cuda", False):
            gpu_name = torch.cuda.get_device_name(backend.device)
            uuid = getattr(torch.cuda.get_device_properties(backend.device), "uuid", None)
            gpu_uuid = f"GPU-{uuid}" if uuid is not None else ""
        else:
            dev, warning = smi_device()
            if dev is None:
                print(warning, file=sys.stderr)
            else:
                gpu_name, gpu_uuid = nvidia_smi("name", dev), nvidia_smi("uuid", dev)
        gpu_index = nvidia_smi("index", gpu_uuid) if gpu_uuid else ""
    return {
        "column": args.column,
        "query_file": args.query_file or "",
        "query_column": args.query_column if args.query_file else "",
        "seed": "" if args.query_file else args.seed,  # only generated queries use it
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_index": gpu_index,
        "driver": nvidia_driver() if gpu else "",
        "torch_version": torch.__version__ if torch is not None else "n/a",
        "torch_cuda": (torch.version.cuda or "") if torch is not None else "n/a",
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
        "merge_faiss_version": faiss.__version__,
    }


GPU_BACKENDS = ("gpu", "cuvs", "spare", "pylate")
CUDA_LIB_FIELDS = ("cublas_version", "cuda_libraries", "cuda_lib_versions", "cuda_lib_duplicates")


def backend_uses_cuda(backend):
    """Whether the engine runs on CUDA: a GPU backend on a CUDA device (a torch backend on a CPU
    device, as tests build it, does not)."""
    if backend.name not in GPU_BACKENDS:
        return False
    if getattr(backend, "torch", None) is not None:
        return bool(getattr(backend, "_cuda", False))
    return True


def cuda_libs_fields(used_cuda, libs=None):
    """The CUDA libraries the engine ran on (the families the paper stack pins,
    cudalibs.FAMILIES), read from /proc/self/maps (or ``libs``) when a row is written, so
    after the pass that loaded them. Never raises.

      cuda_libraries       family=real path of the one mapped copy, ``;``-separated
      cuda_lib_versions    family=the version that copy reports (cudalibs.version;
                           families without one left out)
      cublas_version       cuBLAS's M.m.p, "" unless exactly one copy
      cuda_lib_duplicates  family=path@version for every copy of a family mapped from two
                           files (those families are then left out above); only seen with
                           --allow-duplicate-cuda-libs, since main refuses such a run

    All blank when the run did not use CUDA, whatever is mapped (``import torch`` maps the
    libraries in a CPU run too). nova-bf's manifest records the same (compute.cuda_libraries
    / cuda_library_duplicates / cublas_version)."""
    out = dict.fromkeys(CUDA_LIB_FIELDS, "")
    if not used_cuda:
        return out
    rec = cudalibs.record(libs)
    single, dups = rec["libraries"], rec["duplicates"]
    out["cuda_libraries"] = ";".join(f"{f}={c['path']}" for f, c in single.items())
    out["cuda_lib_versions"] = ";".join(f"{f}={c['version']}" for f, c in single.items()
                                        if c["version"])
    out["cublas_version"] = single.get("libcublas", {}).get("version", "")
    out["cuda_lib_duplicates"] = ";".join(f"{f}={c['path']}@{c['version']}"
                                          for f, cs in dups.items() for c in cs)
    return out


def refuse_duplicate_cuda_libs(args, backend, libs=None):
    """SystemExit when a CUDA run maps two copies of one CUDA library (cudalibs.duplicates).

    The paper stack requires every engine to run on the same cuBLAS and friends. Called once
    the backend is set up (before any pass) and again for every row (a pass may load a library
    lazily). --allow-duplicate-cuda-libs only records them."""
    if not backend_uses_cuda(backend) or getattr(args, "allow_duplicate_cuda_libs", False):
        return
    dups = cudalibs.duplicates(cudalibs.mapped() if libs is None else libs)
    if dups:
        raise SystemExit(cudalibs.duplicate_message(dups) +
                         "\n(--allow-duplicate-cuda-libs runs anyway and records both copies)")


ALLOW_UNPINNED_ENV = "HARNESS_ALLOW_UNPINNED_STACK"


def refuse_unpinned_stack(args, backend):
    """SystemExit when a CUDA run is not on the paper stack set up by env/paper_run.sh
    (see stack_problem). Checked once the backend is built, before any pass.

    CPU runs (tests, the dev box) are never refused. --allow-unpinned-stack (or
    $HARNESS_ALLOW_UNPINNED_STACK, which the test suite sets so its GPU tests run on any node)
    runs anyway; the row's paper_* columns then show what was set."""
    if not backend_uses_cuda(backend) or getattr(args, "allow_unpinned_stack", False) \
            or os.environ.get(ALLOW_UNPINNED_ENV):
        return
    problem = stack_problem()
    if problem:
        raise SystemExit(f"refusing a GPU run off the paper stack: {problem}\n"
                         "(--allow-unpinned-stack runs anyway, for diagnosis: such rows are "
                         "not paper numbers)")


# the GPU's state at the run's start (gpu_state); the launcher's nvidia-smi log has clocks /
# power / temperature / throttle reasons during the passes
GPU_STATE_FIELDS = ("persistence_mode", "ecc.mode.current", "compute_mode",
                    "clocks.applications.graphics", "clocks.applications.memory", "clocks.max.sm",
                    "clocks.max.mem", "power.limit", "pcie.link.gen.current",
                    "pcie.link.gen.max", "pcie.link.width.current")


def gpu_state(dev):
    """``field=value;...`` (GPU_STATE_FIELDS) for nvidia-smi GPU ``dev`` (index or UUID), ""
    without one or without nvidia-smi. Call it before the passes only.

    One query, or field by field when the driver rejects a name (one unknown name fails the
    whole query)."""
    if not dev:
        return ""
    lines = nvidia_smi_lines(",".join(GPU_STATE_FIELDS), dev)
    vals = [v.strip() for v in lines[0].split(",")] if lines else []
    if len(vals) == len(GPU_STATE_FIELDS):
        pairs = zip(GPU_STATE_FIELDS, vals)
    else:
        pairs = ((f, nvidia_smi(f, dev)) for f in GPU_STATE_FIELDS)
    return ";".join(f"{f}={v}" for f, v in pairs if v)


def stack_fields(args, backend, gpu_dev=""):
    """The per-run stack / code / host columns (CSV_FIELDS, "which stack"). Never raises.

      paper_stack_sha / paper_code_sha / paper_run_dir   the launcher's PAPER_STACK_SHA
          (versions.env's hash), PAPER_CODE_SHA (the job's code tree, env/code_sha.py) and
          PAPER_RUN_DIR (check_env.json, code.json, GPU state, nvidia-smi log); blank when
          not launched through env/paper_run.sh
      instance_id / instance_type / availability_zone    EC2 IMDSv2, blank off EC2
      faiss_build / faiss_compile_options                see faiss_build()
      cuvs_version / libcuvs / libcuvs_sha256 / cuvs_patches   --backend cuvs only; see
          cuvs_fields()
      gpu_state                GPU state at the start (GPU runs; gpu_state())
      thread_env / cuda_env    the THREAD_ENV_PREFIXES / CUDA_ENV_PREFIXES variables set
    thread_libs / openmp_runtimes are per row (thread_libs_fields)."""
    import faiss
    out = dict(run_fields())
    out.update(ec2_identity())
    out.update(faiss_build(faiss))
    out.update(cuvs_fields() if args.backend == "cuvs" else
               dict.fromkeys(("cuvs_version", "libcuvs", "libcuvs_sha256", "cuvs_patches"), ""))
    out["gpu_state"] = gpu_state(gpu_dev) if backend.name in GPU_BACKENDS else ""
    out["thread_env"] = env_str(THREAD_ENV_PREFIXES)
    out["cuda_env"] = env_str(CUDA_ENV_PREFIXES)
    return out


# =============================================================================================
# 3. Import origins: where `bench` and `nova_bf` come from
# =============================================================================================
#
# A leg's engine process runs harness_cli.py or novabf.py by path, so the `bench` it imports is
# whatever the interpreter resolves (maybe an install). Both packages must be this tree's own:
# `bench` = <cb>/bench and `nova_bf` = <root>/src/nova_bf of the nova-bf tree the code hash
# covers (a repo checkout's ../python/nova-bf, a stage's nova-bf/). <cb> is the running script's
# location, never sys.path, so putting a tree first on sys.path cannot fake a pass; env/check.py
# (which does that) asks a fresh `python -P` instead (probe).

def novabf_root(cb: str) -> str:
    """The nova-bf tree paper_run.sh hashes for this clean_benchmarks tree: a repo checkout's
    ../python/nova-bf, else (a stage) its nova-bf/ copy."""
    repo = os.path.join(os.path.dirname(os.path.abspath(cb)), "python", "nova-bf")
    return repo if os.path.isdir(repo) else os.path.join(os.path.abspath(cb), "nova-bf")


def _under(path: str | None, root: str) -> bool:
    if not path:
        return False
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def _origin(name: str) -> str | None:
    """The module's file: the loaded one if it is imported, else where an import would find it
    (find_spec on a top-level package: its __init__ is not executed)."""
    m = sys.modules.get(name)
    if m is not None and getattr(m, "__file__", None):
        return os.path.abspath(m.__file__)
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return os.path.abspath(spec.origin) if spec is not None and spec.origin else None


def judge(cb: str, got: dict, need_novabf: bool) -> dict:
    """{bench, nova_bf, expected, ok, problems} for origins ``got`` ({bench, nova_bf} files)."""
    # nova_bf must be <root>/src/nova_bf (what an editable install of the hashed tree
    # imports), not a copy elsewhere under the root (.venv, build/, tests/).
    want = {"bench": os.path.join(os.path.abspath(cb), "bench"),
            "nova_bf": os.path.join(novabf_root(cb), "src", "nova_bf")}
    problems = []
    if not _under(got.get("bench"), want["bench"]):
        problems.append(f"bench is imported from {got.get('bench')}, not this tree's {want['bench']}")
    nb = got.get("nova_bf")
    if nb is None:
        if need_novabf:
            problems.append("nova_bf is not importable")
    elif not _under(nb, want["nova_bf"]):
        problems.append(f"nova_bf is imported from {nb}, not under {want['nova_bf']} (the "
                        "package in the nova-bf tree the code hash covers)")
    return {"bench": got.get("bench"), "nova_bf": nb, "expected": want, "ok": not problems,
            "problems": problems}


def check(cb: str, need_novabf: bool = True) -> dict:
    """This process's origins (the loaded modules, or where they would load from)."""
    return judge(cb, {"bench": _origin("bench"), "nova_bf": _origin("nova_bf")}, need_novabf)


PROBE = ("import importlib.util, json, os\n"
         "def o(n):\n"
         "    try:\n"
         "        s = importlib.util.find_spec(n)\n"
         "    except (ImportError, ValueError):\n"
         "        return None\n"
         "    return os.path.abspath(s.origin) if s is not None and s.origin else None\n"
         "print(json.dumps({'bench': o('bench'), 'nova_bf': o('nova_bf')}))\n")


def probe(cb: str, python: str | None = None, need_novabf: bool = True) -> dict:
    """What a by-path engine process of this interpreter resolves: a fresh `python -P` (no script
    dir or cwd on sys.path; PYTHONPATH and the installs as for a leg), run from a neutral cwd."""
    try:
        p = subprocess.run([python or sys.executable, "-P", "-c", PROBE], capture_output=True,
                           text=True, timeout=60, cwd=os.path.sep)
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "problems": [f"probe failed: {type(exc).__name__}: {exc}"],
                "bench": None, "nova_bf": None, "expected": None}
    return judge(cb, got, need_novabf)


def refuse_unless_ok(rec: dict, who: str) -> None:
    if not rec["ok"]:
        print(f"{who}: REFUSED: " + "; ".join(rec["problems"]), file=sys.stderr, flush=True)
        raise SystemExit(4)
