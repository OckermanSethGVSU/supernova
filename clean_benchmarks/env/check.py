#!/usr/bin/env python3
"""Verify that the environment matches the pinned benchmark stack (versions.env).

Checks installed package versions, commits, FAISS build, cuVS patches, GPU
model, instance type, and executable paths.

Loads engines in the harness's import order and verifies that each pinned
CUDA library is loaded exactly once, from PyTorch's pip installation rather
than conda. Conda's CUDA libraries must be installed but not loaded.

Records OpenMP runtimes, math libraries, and relevant environment variables
without treating differences as failures.

Usage:
    python check.py --versions versions.env
    python check.py --versions versions.env --json [OUT]
    python check.py --versions versions.env --report-only

By default, mismatches cause failure. --json prints or saves the check results;
--report-only reports mismatches without failing.

Run after activate.sh in each benchmark environment.
"""

import argparse
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.dirname(HERE)                            # clean_benchmarks
CUVS_PATCH_DIR = os.path.join(CB, "baselines", "cuvs", "patches")
CUVS_SPARSE_PY = os.path.join(CB, "baselines", "cuvs", "sparse_binding", "cuvs_sparse.py")
# import the harness's own cudalibs, so this check loads the engines in the harness's order
sys.path.insert(0, CB)
from bench.record import cudalibs  # noqa: E402
from bench.record import provenance  # noqa: E402 - versions.env's hash, IMDS, OpenMP libs, import origins
# the commands that must resolve inside the paper env
ENV_COMMANDS = ("python", "nova-bf", "bench")

# every CUDA / RAPIDS / engine library, for the record and the hashes (the CUDA families are
# checked through cudalibs, below)
LIBS = re.compile(r"/(lib(?:cublas|cublasLt|cudart|cusparse|cusolver|curand|cufft|nvJitLink|nvjitlink|"
                  r"nvrtc|cudnn|rmm|cuvs|cuvs_c|raft|faiss|faiss_gpu|nova_cuvs_sparse)"
                  r"[^/\s]*\.so[^/\s]*)$")

# (versions.env key, conda package, pip wheel, loaded families)
CUDA_LIBS = (
    ("CUBLAS_VERSION", "libcublas", "nvidia-cublas", ("libcublas", "libcublasLt")),
    ("CUSPARSE_VERSION", "libcusparse", "nvidia-cusparse", ("libcusparse",)),
    ("CUSOLVER_VERSION", "libcusolver", "nvidia-cusolver", ("libcusolver",)),
    ("CUFFT_VERSION", "libcufft", "nvidia-cufft", ("libcufft",)),
    ("CURAND_VERSION", "libcurand", "nvidia-curand", ("libcurand",)),
    ("CUDART_VERSION", "cuda-cudart", "nvidia-cuda-runtime", ("libcudart",)),
    ("NVRTC_VERSION", "cuda-nvrtc", "nvidia-cuda-nvrtc", ("libnvrtc", "libnvrtc-builtins")),
    ("NVJITLINK_VERSION", "libnvjitlink", "nvidia-nvjitlink", ("libnvJitLink",)),
)
# the RAPIDS packages the cuVS build links: (conda package prefix, required)
RAPIDS = (("librmm", True), ("libraft", True), ("rmm", False), ("pylibraft", False))


def read_versions(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v.startswith('"'):
            v = v[1:v.index('"', 1)]
        else:
            v = v.split("#", 1)[0].strip()
        out[k.strip()] = v
    return out


def cuvs_patch_checks(want, stamp, patch_dir):
    """The cuVS patch rows, as [(name, got, expect, ok)].

    The pin is versions.env's CUVS_PATCH_SHA256 (``name=sha256`` per entry of CUVS_PATCHES,
    in order). The patch files are git-ignored, so the pin, not the directory, says which
    patch content the paper stack means. Two rows:
    - the env's build stamp (build.sh: sha256 of each patch it applied, in order) against
      the pin: was this cuVS built with exactly the pinned patches;
    - this tree's patch files (``patch_dir``/<name>) against the pin: a missing file (not
      staged or copied over) or a changed one fails.
    A patch named in CUVS_PATCHES without a pin fails both."""
    names = want.get("CUVS_PATCHES", "").split()
    pins = dict(e.split("=", 1) for e in want.get("CUVS_PATCH_SHA256", "").split() if "=" in e)
    expect = ";".join(f"{p}={pins.get(p, '(not pinned in CUVS_PATCH_SHA256)')}" for p in names)
    files = []
    for p in names:
        f = os.path.join(patch_dir, p)
        files.append(f"{p}={sha256(f)}" if os.path.isfile(f) else
                     f"{p}=(missing: {f})")
    got_files = ";".join(files)
    stamp = stamp or f"(no {provenance.CUVS_PATCH_STAMP} in the env: not built by build.sh)"
    return [("cuvs patches applied (build stamp vs CUVS_PATCH_SHA256)", stamp, expect,
             stamp == expect),
            ("cuvs patch files (vs CUVS_PATCH_SHA256)", got_files, expect, got_files == expect)]


def path_checks(names=ENV_COMMANDS, prefix=None, path=None):
    """[(name, got, expect, ok)]: each command in ``names`` must resolve through PATH to
    <prefix>/bin/<name> (default prefix: sys.prefix). A command missing from PATH fails. The
    path is compared unresolved (an env's bin/python is often a symlink to the base
    interpreter)."""
    env_bin = os.path.join(prefix or sys.prefix, "bin")
    out = []
    for n in names:
        got = shutil.which(n, path=path)
        got = os.path.abspath(got) if got else "(not on PATH)"
        want_p = os.path.join(env_bin, n)
        out.append((f"PATH: {n} resolves inside the paper env", got, want_p, got == want_p))
    return out


def try_import(name):
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001 - reported, not raised
        return None, f"{type(e).__name__}: {e}"


def pinned_commit(dist):
    """The VCS commit a pip-installed distribution was built from (direct_url.json)."""
    try:
        raw = md.distribution(dist).read_text("direct_url.json")
        return json.loads(raw).get("vcs_info", {}).get("commit_id", "") if raw else ""
    except Exception:  # noqa: BLE001
        return ""


def sh(*cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loaded_libs():
    """soname family -> sorted real paths mapped into this process, for every library LIBS
    matches (the record and the hashes)."""
    fam = {}
    try:
        maps = open("/proc/self/maps").read().splitlines()
    except OSError:
        return fam
    for line in maps:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        m = LIBS.search(parts[5])
        if m:
            key = re.sub(r"\.so.*$", "", m.group(1))
            fam.setdefault(key, set()).add(os.path.realpath(parts[5]))
    return {k: sorted(v) for k, v in sorted(fam.items())}


def conda_versions(meta, pkg):
    """Installed versions of conda package ``pkg`` (conda-meta/<pkg>-<ver>-<build>.json)."""
    out = []
    for f in os.listdir(meta):
        if f.endswith(".json") and f.startswith(pkg + "-"):
            rest = f[len(pkg) + 1:-len(".json")]
            ver, _, _build = rest.rpartition("-")
            if ver and ver[0].isdigit():   # not libcusparse-dev-... when pkg is libcusparse
                out.append(ver)
    return sorted(out)


def conda_packages(meta, prefix):
    """{name: version} of every installed conda package named ``prefix`` or ``prefix-*``."""
    out = {}
    for f in os.listdir(meta):
        if not f.endswith(".json"):
            continue
        name, ver, _build = (f[:-len(".json")].rsplit("-", 2) + ["", ""])[:3]
        if name == prefix or name.startswith(prefix + "-"):
            out[name] = ver
    return out


def runtime_version(fam, pin):
    """What the library reports about itself for a pin: the first VERSION_PARTS components."""
    if pin in (None, "", "TO-FILL"):
        return pin
    return ".".join(str(pin).split(".")[:cudalibs.VERSION_PARTS[fam]])


def under(path, root):
    return root is not None and os.path.commonpath([path, root]) == root


def gpu_smoke(torch, faiss, cuvs):
    """Run each engine once on the GPU, so every library a real pass loads lazily is mapped
    before the maps are read. [(name, error or None)]."""
    out = []

    def run(name, fn):
        try:
            fn()
            out.append((name, None))
        except Exception as e:  # noqa: BLE001 - reported as a failed check
            out.append((name, f"{type(e).__name__}: {e}"))

    def t():
        a = torch.ones(64, 64, device="cuda")
        (a @ a).sum().item()   # cuBLAS
        torch.linalg.qr(a)     # cuSOLVER
        torch.fft.rfft(a)      # cuFFT
        torch.randn(8, device="cuda")   # cuRAND / torch's own generator
        torch.cuda.synchronize()
    run("torch GPU ops (cuBLAS / cuSOLVER / cuFFT)", t)
    if faiss is not None and hasattr(faiss, "StandardGpuResources"):
        import numpy as np

        def f():
            x = np.random.default_rng(0).random((256, 16), dtype=np.float32)
            faiss.knn_gpu(faiss.StandardGpuResources(), x[:4], x, 4)
        run("faiss.knn_gpu", f)
    if cuvs is not None:
        def c():
            from cuvs.neighbors import brute_force
            x = torch.rand(256, 16, device="cuda")
            brute_force.search(brute_force.build(x, metric="inner_product"), x[:4], 4)
            torch.cuda.synchronize()
        run("cuvs brute_force", c)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--versions", required=True)
    ap.add_argument("--json", nargs="?", const="-", default=None, metavar="OUT",
                    help="the result as JSON: on stdout (no OUT, instead of the table), or "
                         "written to OUT (the table still printed)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--engines", nargs="+", default=["faiss", "cuvs", "nova_bf", "spare", "pylate"],
                    help="what this environment must provide (the fallback layout has one env per "
                         "engine: name only that engine there)")
    a = ap.parse_args(argv)
    want = read_versions(a.versions)
    rows, bad = [], []

    def check(name, got, expect, how="eq"):
        ok = (got == expect if how == "eq" else str(got).startswith(str(expect)) if how == "prefix"
              else bool(got))
        if expect in ("TO-FILL", "", None):
            ok, expect = True, f"(not pinned yet: {expect})"
        rows.append((name, str(got), str(expect), ok))
        if not ok:
            bad.append(name)

    def fail(name, got, expect):
        rows.append((name, str(got), str(expect), False))
        bad.append(name)

    # 1. the harness's import order, before anything else that could map a CUDA library
    try:
        torch, err = cudalibs.torch_first(), "ImportError: torch is not installed"
    except Exception as e:  # noqa: BLE001 - a broken torch is a failed check
        torch, err = None, f"{type(e).__name__}: {e}"
    faiss, ferr = try_import("faiss")          # every env needs it: the top-k merge is FAISS's
    cuvs = cerr = None
    if "cuvs" in a.engines:
        cuvs, cerr = try_import("cuvs")
        if cuvs is not None:
            try_import("cuvs.neighbors.brute_force")

    check("python", ".".join(map(str, sys.version_info[:2])), want.get("PYTHON_VERSION"))
    for name, got, expect, _ in path_checks():
        check(name, got, expect)
    # what a leg's by-path engine process imports. This script put CB first on sys.path,
    # which would hide the answer, so a fresh `python -P` (no script dir / cwd) is asked
    orig = provenance.probe(CB, need_novabf="nova_bf" in a.engines)
    if orig["ok"]:
        rows.append(("import origins (bench / nova_bf, python -P)",
                     f"bench {orig['bench']}, nova_bf {orig['nova_bf']}",
                     f"bench under {orig['expected']['bench']}, nova_bf under "
                     f"{orig['expected']['nova_bf']}", True))
    else:
        fail("import origins (bench / nova_bf, python -P)", "; ".join(orig["problems"]),
             "bench = this tree's, nova_bf under the hashed nova-bf root")
    if os.environ.get("CONDA_PREFIX") and os.path.realpath(os.environ["CONDA_PREFIX"]) \
            != os.path.realpath(sys.prefix):
        fail("check.py runs in the activated env", sys.prefix, os.environ["CONDA_PREFIX"])
    check("torch", getattr(torch, "__version__", err), want.get("TORCH_VERSION"))
    gpu = False
    if torch is not None:
        check("torch CUDA", torch.version.cuda, want.get("CUDA_VERSION"))
        gpu = torch.cuda.is_available()
        check("CUDA device available", gpu, True)
        if gpu:
            check("GPU model", torch.cuda.get_device_name(0), want.get("GPU_NAME"))
    # the instance (IMDS): blank off EC2 or with HARNESS_NO_IMDS, which fails the check
    ec2 = provenance.ec2_identity()
    check("EC2 instance type", ec2["instance_type"] or "(no instance metadata: not on EC2?)",
          want.get("INSTANCE_TYPE"))
    rows.append(("EC2 instance", f"{ec2['instance_id'] or '-'} {ec2['availability_zone'] or '-'}",
                 "(recorded)", True))
    for mod, key in (("numpy", "NUMPY_VERSION"), ("pyarrow", "PYARROW_VERSION"),
                     ("duckdb", "DUCKDB_VERSION")):
        m, err2 = try_import(mod)
        check(mod, getattr(m, "__version__", err2), want.get(key))

    check("faiss (merge + baseline)", getattr(faiss, "__version__", ferr), want.get("FAISS_VERSION"))
    meta = os.path.join(sys.prefix, "conda-meta")
    has_meta = os.path.isdir(meta)
    if faiss is not None:
        opts = faiss.get_compile_options()
        rows.append(("faiss compile options", opts, "(recorded)", True))
        if "faiss" in a.engines:
            check("faiss GPU build", hasattr(faiss, "StandardGpuResources"), True)
            # FAISS rows must time FAISS's own kernels: a cuVS-enabled build can route bfKnn to cuVS
            builds = sorted(f for f in (os.listdir(meta) if has_meta else [])
                            if f.startswith(("libfaiss-", "faiss-gpu")))
            rows.append(("faiss conda packages", ", ".join(builds) or "(none: not a conda install)",
                         "(recorded)", True))
            check("faiss is NOT a cuVS-enabled build",
                  not (any("cuvs" in b for b in builds) or "CUVS" in opts.upper()), True)
            # the exact build versions.env pins, channel::name=version=build (the channel also
            # carries *_nightly builds of the same version and other CUDA variants)
            m = re.fullmatch(r"(?:([^:]+)::)?([^=]+)=([^=]+)=(.+)", want.get("FAISS_CONDA_SPEC", ""))
            if m:
                chan, name, ver, build = m.groups()
                got = provenance.conda_packages((name,), sys.prefix)
                check("faiss conda build (exact)", ", ".join(got) or "(not a conda install)",
                      f"{name}-{ver}-{build}")
                if chan and got == [f"{name}-{ver}-{build}"]:
                    try:
                        with open(os.path.join(meta, got[0] + ".json"), encoding="utf-8") as f:
                            got_chan = json.load(f).get("channel", "")
                    except Exception as e:  # noqa: BLE001
                        got_chan = f"(unreadable: {e})"
                    # conda-meta records "pytorch", ".../pytorch" or (conda 25+)
                    # "https://conda.anaconda.org/pytorch/linux-64": the channel name is the last
                    # path component once a platform subdir is dropped
                    parts = got_chan.rstrip("/").split("/")
                    if len(parts) > 1 and re.fullmatch(r"noarch|(linux|osx|win)-[a-z0-9_]+",
                                                       parts[-1]):
                        parts = parts[:-1]
                    check("faiss conda channel", got_chan,
                          got_chan if parts[-1] == chan else chan)
    if "cuvs" in a.engines:
        check("cuvs", getattr(cuvs, "__version__", cerr), want.get("CUVS_VERSION"))
        rmm, err2 = try_import("rmm")
        check("rmm", getattr(rmm, "__version__", err2), want.get("RAPIDS_VERSION"), "prefix")
        if cuvs is not None:
            rows.append(("cuvs module", os.path.dirname(cuvs.__file__), "(recorded)", True))
        # the env's build stamp and this tree's patch files, both against CUVS_PATCH_SHA256
        for name, got, expect, _ in cuvs_patch_checks(want, provenance.cuvs_patches(sys.prefix),
                                                      CUVS_PATCH_DIR):
            check(name, got, expect)
        # the sparse shim, loaded as the harness loads it (after libcuvs)
        try:
            spec = importlib.util.spec_from_file_location(
                "cuvs_sparse", CUVS_SPARSE_PY)
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            shim.load()
            rows.append(("cuvs sparse shim", shim.lib_path(), "(recorded)", True))
        except Exception as e:  # noqa: BLE001
            fail("cuvs sparse shim loads", f"{type(e).__name__}: {e}", "loads")
    if gpu:
        for name, e in gpu_smoke(torch, faiss if "faiss" in a.engines else None, cuvs):
            check(name, e or "ok", "ok")
    if "spare" in a.engines:
        check("spare commit", pinned_commit("spare"), want.get("SPARE_COMMIT"), "prefix")
        # the module the harness imports, not just the package: spare.backend_torch needs
        # psutil, which SPARe does not declare (`import spare` alone can pass while runs fail)
        sb, err2 = try_import("spare.backend_torch")
        check("spare.backend_torch importable", sb is not None or err2, True, "truthy" if sb else "eq")
    if "pylate" in a.engines:
        pl, err2 = try_import("pylate")
        check("pylate", getattr(pl, "__version__", err2), want.get("PYLATE_VERSION"))
        for mod in ("pylate.rank", "pylate.scores"):   # what the harness imports
            m, err3 = try_import(mod)
            check(f"{mod} importable", m is not None or err3, True, "truthy" if m else "eq")
        check("pylate commit", pinned_commit("pylate"), want.get("PYLATE_COMMIT"), "prefix")
    if "nova_bf" in a.engines:
        nb, err2 = try_import("nova_bf")
        rf, err3 = try_import("rust_fetch")
        check("nova_bf importable", nb is not None or err2, True, "truthy" if nb else "eq")
        check("rust_fetch importable", rf is not None or err3, True, "truthy" if rf else "eq")
        if nb is not None:
            root = os.path.dirname(os.path.abspath(nb.__file__))
            head = sh("git", "rev-parse", "HEAD", cwd=root)
            # not part of the frozen stack: recorded for the run, never a failure
            dirty = sh("git", "status", "--porcelain", "--", "python/nova-bf",
                       cwd=sh("git", "rev-parse", "--show-toplevel", cwd=root) or root)
            rows.append(("nova-bf commit", head or "(not a git checkout)", "(recorded)", True))
            rows.append(("nova-bf tree", "clean" if not dirty else "uncommitted changes",
                         "(recorded)", True))

    # every engine is loaded by now: record (never fail on) the OpenMP / BLAS / threading
    # libraries this process maps, and the environment that steers them
    tlibs = provenance.thread_libs()
    omp = [k for k in tlibs if k in provenance.OPENMP_RUNTIMES]
    rows.append(("OpenMP runtimes mapped", ", ".join(omp) or "(none)",
                 "(recorded; more than one: one thread pool each)", True))
    for k, paths in tlibs.items():
        rows.append((f"mapped {k}", ", ".join(paths), "(recorded)", True))
    thread_env = provenance.env_str(provenance.THREAD_ENV_PREFIXES)
    cuda_env = provenance.env_str(provenance.CUDA_ENV_PREFIXES)
    rows.append(("OMP / KMP / MKL environment", thread_env or "(none set)", "(recorded)", True))
    rows.append(("CUDA / torch environment", cuda_env or "(none set)", "(recorded)", True))

    drv = sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()
    check("NVIDIA driver", drv[0] if drv else "(nvidia-smi unavailable)", want.get("NVIDIA_DRIVER"))

    # 2. the CUDA libraries. The pip wheels beside torch are the one copy every engine runs on,
    # at the versions.env pins; the conda env's copies (what libfaiss / libcuvs were built
    # against) must be installed, -dev included, and never loaded: recorded, not pinned.
    conda_side = has_meta and any(e in a.engines for e in ("faiss", "cuvs"))
    for key, cpkg, ppkg, _fams in CUDA_LIBS:
        if conda_side:
            lib, dev = conda_versions(meta, cpkg), conda_versions(meta, cpkg + "-dev")
            name = f"conda {cpkg} + -dev (installed, never loaded)"
            if not lib or not dev:
                fail(name, f"lib {lib or 'MISSING'}, -dev {dev or 'MISSING'}", "both installed")
            elif len(lib) > 1 or dev != lib:
                fail(name, f"lib {lib}, -dev {dev}", "one version, -dev == lib")
            else:
                rows.append((name, lib[0], "(recorded)", True))
        try:
            pv = md.version(ppkg)
        except md.PackageNotFoundError:
            pv = "(not installed)"
        check(f"pip {ppkg}", pv, want.get(key))
    if conda_side:   # the RAPIDS packages the cuVS build links
        for prefix, required in RAPIDS:
            found = conda_packages(meta, prefix)
            if not found and required:
                fail(f"conda {prefix}*", "(not installed)", want.get("RAPIDS_VERSION"))
            for name, ver in sorted(found.items()):
                check(f"conda {name}", ver, want.get("RAPIDS_VERSION"), "prefix")

    pip_root = cudalibs.pip_nvidia_root()
    check("torch's pip nvidia/ dir", pip_root or "(none: no CUDA torch with pip CUDA wheels)",
          pip_root or "a directory", "truthy" if pip_root else "eq")
    cuda = cudalibs.mapped()
    for key, _cpkg, _ppkg, fams in CUDA_LIBS:
        for fam in fams:
            paths = cuda.get(fam, [])
            if not paths:   # torch_first maps every one of them: missing = not this stack
                fail(f"loaded {fam}", "(not loaded)", "one copy, under the pip nvidia/ dir")
                continue
            if len(paths) > 1:
                fail(f"loaded {fam}: one copy", f"{len(paths)} copies: {paths}", "one copy")
            for p in paths:
                check(f"loaded {fam} is the pip copy", p if under(p, pip_root) else f"NOT pip: {p}",
                      p if under(p, pip_root) else f"under {pip_root}")
            if cudalibs.FAMILIES[fam] is not None:
                got = sorted({cudalibs.version(fam, p) or "(version call failed)" for p in paths})
                check(f"loaded {fam} version", ", ".join(got), runtime_version(fam, want.get(key)))
    libs = loaded_libs()
    dup = {k: v for k, v in libs.items() if len(v) > 1}
    check("each CUDA / RAPIDS library loaded from ONE file", not dup or f"duplicates: {dup}", True,
          "eq" if dup else "truthy")
    hashes = {p: sha256(p) for v in libs.values() for p in v
              if re.search(r"lib(cuvs|faiss|nova_cuvs_sparse)", p)}

    gpu_rec = {}
    if gpu:
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        gpu_rec = {"name": torch.cuda.get_device_name(0),
                   "uuid": f"GPU-{uuid}" if uuid is not None else ""}
    result = {"stack": want.get("STACK_NAME"), "stack_sha": provenance.versions_sha(a.versions),
              "ok": not bad, "mismatches": bad,
              "checks": [dict(name=n, got=g, expected=e, ok=o) for n, g, e, o in rows],
              "cuda_libraries": cudalibs.record(cuda), "loaded_libraries": libs,
              "thread_libraries": tlibs, "openmp_runtimes": omp,
              "thread_env": dict(kv.split("=", 1) for kv in thread_env.split(";") if kv),
              "cuda_env": dict(kv.split("=", 1) for kv in cuda_env.split(";") if kv),
              "ec2": ec2, "gpu": gpu_rec,
              "sha256": hashes, "prefix": sys.prefix,
              "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "")}
    if a.json and a.json != "-":
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    if a.json == "-":
        print(json.dumps(result, indent=1))
    else:
        w = max(len(r[0]) for r in rows)
        for n, g, e, o in rows:
            print(f"{'ok  ' if o else 'FAIL'} {n:<{w}}  {g}" + ("" if o or e == "True" else f"   (pinned: {e})"))
        print("\nloaded CUDA / RAPIDS libraries:")
        for k, v in libs.items():
            for p in v:
                print(f"  {k:<18} {p}")
        if not libs:
            print("  (none mapped)")
        print(f"\n{'PINNED STACK OK' if not bad else 'NOT THE PINNED STACK: ' + ', '.join(bad)}")
    return 0 if (not bad or a.report_only) else 1


if __name__ == "__main__":
    sys.exit(main())
