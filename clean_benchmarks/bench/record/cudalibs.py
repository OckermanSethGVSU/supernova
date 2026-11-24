"""Ensure all GPU engines use the same CUDA libraries.

Call ``torch_first()`` before importing FAISS, cuVS, or CuPy. It loads
PyTorch's pip-installed CUDA libraries first, preventing the dynamic
loader from using conflicting copies from the conda environment.

``mapped``, ``duplicates``, and ``record`` inspect the libraries
actually loaded by the process, while ``version`` reports their versions.

This ensures consistent CUDA dependencies across nova-bf, FAISS, cuVS,
SPARe, and PyLate for fair benchmarking.
"""

import ctypes
import glob
import importlib.util
import os
import re

# family -> the library's own version call (None: no version API)
#   *GetProperty(MAJOR_VERSION=0 / MINOR_VERSION=1 / PATCH_LEVEL=2, int*) -> "M.m.p"
#   cudaRuntimeGetVersion(int*) -> "M.m"; nvrtcVersion / nvJitLinkVersion(M*, m*) -> "M.m"
FAMILIES = {
    "libcublas": "cublasGetProperty",
    "libcublasLt": "cublasLtGetProperty",
    "libcusparse": "cusparseGetProperty",
    "libcusolver": "cusolverGetProperty",
    "libcufft": "cufftGetProperty",
    "libcurand": "curandGetProperty",
    "libcudart": "cudaRuntimeGetVersion",
    "libnvrtc": "nvrtcVersion",
    "libnvrtc-builtins": None,   # loaded by NVRTC itself; checked for copies only
    "libnvJitLink": "nvJitLinkVersion",
}
# how many version components each family reports about itself
VERSION_PARTS = {"libcublas": 3, "libcublasLt": 3, "libcusparse": 3, "libcusolver": 3,
                 "libcufft": 3, "libcurand": 3, "libcudart": 2, "libnvrtc": 2,
                 "libnvJitLink": 2}
# libcublasLt.so.13 / libcusparse.so.12.7.10.12 / libnvrtc-builtins.so.13.2, not
# libcusolverMg / libcufftw / libnvrtc.alt (other libraries)
_LIB = re.compile(r"/(lib(?:cublasLt|cublas|cusparse|cusolver|cufft|curand|cudart|"
                  r"nvrtc-builtins|nvrtc|nvJitLink|nvjitlink))\.so(?:\.[0-9][^/\s]*)?$")
# torch_first's load order: each library after the ones it links (readelf -d of the cu13 wheels:
# cuBLAS needs cuBLASLt, cuSPARSE needs nvJitLink, cuSOLVER needs cuBLAS, cuBLASLt, cuSPARSE and
# nvJitLink), so no DT_NEEDED is resolved by search path.
# Entries are (family, cu12-layout wheel folder); cu13 wheels all share nvidia/cu13/lib.
PRELOAD = (("libnvJitLink", "nvjitlink"), ("libcudart", "cuda_runtime"),
           ("libcublasLt", "cublas"), ("libcublas", "cublas"), ("libcusparse", "cusparse"),
           ("libcusolver", "cusolver"), ("libcufft", "cufft"), ("libcurand", "curand"),
           ("libnvrtc", "cuda_nvrtc"), ("libnvrtc-builtins", "cuda_nvrtc"))


def _torch_site():
    """(site-packages dir holding torch, torch's CUDA version or None), found without
    importing torch; (None, None) when torch is not installed."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None, None
    if spec is None or not spec.origin:
        return None, None
    pkg = os.path.dirname(spec.origin)
    cuda = None
    try:
        with open(os.path.join(pkg, "version.py"), encoding="utf-8") as f:
            m = re.search(r"^cuda\s*(?::[^=]*)?=\s*['\"]([0-9.]+)['\"]", f.read(), re.M)
        cuda = m.group(1) if m else None
    except OSError:
        pass
    return os.path.dirname(pkg), cuda


def pip_nvidia_root():
    """Real path of the pip ``nvidia`` dir next to torch (the one copy of every CUDA library
    that may run); None without a CUDA torch."""
    site, cuda = _torch_site()
    if site is None or cuda is None:
        return None
    d = os.path.join(site, "nvidia")
    return os.path.realpath(d) if os.path.isdir(d) else None


def pip_lib_paths():
    """family -> the pip copy torch would load, for the families present. Uses torch's own
    lookup order: nvidia/<wheel>/lib, then nvidia/cu<major>/lib."""
    site, cuda = _torch_site()
    if site is None or cuda is None:
        return {}
    major = cuda.split(".")[0]
    out = {}
    for fam, folder in PRELOAD:
        name = f"{fam}.so.*[0-9]"
        for d in (os.path.join(site, "nvidia", folder, "lib"),
                  os.path.join(site, "nvidia", f"cu{major}", "lib")):
            hits = sorted(glob.glob(os.path.join(d, name)))
            if hits:
                out[fam] = hits[0]
                break
    return out


def preload_pip_cuda():
    """Load torch's pip CUDA libraries by absolute path, in PRELOAD order; return {family:
    path} of what loaded ({} without a CUDA torch). A library that fails to load is skipped:
    torch's import, or the run's duplicate check, then says what is wrong."""
    done = {}
    for fam, path in pip_lib_paths().items():
        try:
            ctypes.CDLL(path)
            done[fam] = path
        except OSError:
            continue
    return done


def torch_first():
    """Make the pip CUDA libraries the process's copies, then import torch.

    Call before importing faiss, cuvs or cupy (anything that links a CUDA
    library). Returns the torch module, or None when torch is not installed;
    any other import error is raised (a broken torch is not "no torch")."""
    preload_pip_cuda()
    try:
        import torch
    except ImportError:
        return None
    return torch


def mapped(maps_text=None):
    """family -> real paths of every copy mapped into this process (from /proc/self/maps, or
    ``maps_text``), in first-seen order. {} without /proc.

    The maps are ordered by address, so with two copies the order says nothing about which
    was loaded first."""
    if maps_text is None:
        try:
            with open("/proc/self/maps") as f:
                maps_text = f.read()
        except OSError:
            return {}
    fam = {}
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        m = _LIB.search(parts[5].strip())
        if m:
            key = m.group(1)
            key = "libnvJitLink" if key.lower() == "libnvjitlink" else key
            path = os.path.realpath(parts[5].strip())
            seen = fam.setdefault(key, [])
            if path not in seen:
                seen.append(path)
    return {k: fam[k] for k in sorted(fam)}


def duplicates(libs):
    """The families of ``mapped()`` mapped from more than one file."""
    return {k: v for k, v in libs.items() if len(v) > 1}


def version(fam, path):
    """The version the mapped copy at ``path`` reports about itself (its FAMILIES call).

    dlopen(RTLD_NOLOAD) returns the already-mapped object and never maps a new one. No GPU
    needed. "" when the file is not mapped, has no version call, or the call fails."""
    fn_name = FAMILIES.get(fam)
    if fn_name is None:
        return ""
    try:
        lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
        fn = getattr(lib, fn_name)
        if fn_name.endswith("GetProperty"):
            vals = []
            for prop in (0, 1, 2):   # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
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
        a, b = ctypes.c_int(), ctypes.c_int()   # nvrtcVersion(int*, int*); nvJitLink: unsigned*
        if fn(ctypes.byref(a), ctypes.byref(b)) != 0:
            return ""
        return f"{a.value}.{b.value}"
    except Exception:  # noqa: BLE001 - provenance never raises
        return ""


def record(libs=None):
    """{"libraries": {family: {"path", "version"}}, "duplicates": {family: [{"path",
    "version"}, ...]}}. A family mapped from one file goes in "libraries"; one mapped from
    more than one file has no single copy to name, so it goes only in "duplicates"."""
    libs = mapped() if libs is None else libs
    out = {"libraries": {}, "duplicates": {}}
    for fam, paths in libs.items():
        copies = [{"path": p, "version": version(fam, p)} for p in paths]
        if len(copies) == 1:
            out["libraries"][fam] = copies[0]
        else:
            out["duplicates"][fam] = copies
    return out


def duplicate_message(dups):
    """The error a run refuses with when ``dups`` (duplicates()) is not empty."""
    lines = [f"  {fam}: " + ", ".join(paths) for fam, paths in sorted(dups.items())]
    return ("two copies of a CUDA library are mapped into this process, so engines "
            "may not all run on the same one:\n" + "\n".join(lines) + "\n"
            "Every engine must run on torch's pip copies, so all share one CUDA stack: "
            "import torch through cudalibs.torch_first() before "
            "faiss / cuvs, and keep other copies off LD_LIBRARY_PATH "
            "(env/activate.sh).")
