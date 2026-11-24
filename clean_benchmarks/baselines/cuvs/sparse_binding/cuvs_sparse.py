"""ctypes binding for cuVS's sparse exact brute-force kNN (libnova_cuvs_sparse.so,
built from cuvs_sparse_shim.cpp by build.sh).

cuVS 26.10 has sparse brute force only in C++ (``cuvs::neighbors::
brute_force::build`` on a ``raft::device_csr_matrix_view<const float, int,
int, int>`` + ``search`` with ``sparse_search_params``); its C API and Python
package expose dense brute force only. The shim is a thin ``extern "C"``
wrapper; this module loads it (after ``libcuvs.load_library()`` so the
wheel's libcuvs.so and its dependencies are already in the process) and
calls it with device pointers taken from ``__cuda_array_interface__`` (torch
CUDA tensors and CuPy arrays both have it), so neither torch nor CuPy is
imported here. ctypes releases the GIL for the call.

    lib = load()                       # or load("/path/libnova_cuvs_sparse.so")
    knn(res_handle, (indptr, indices, values, n_rows, nnz), (q_indptr, ...),
        n_cols, k, metric, batch_size_index, batch_size_query, neighbors, distances)

``res_handle``: a cuvsResources_t (``cuvs.common.Resources().get_c_obj()``),
its stream bound with ``cuvsStreamSet``. Bind ``CUDA_STREAM_PER_THREAD``
(0x2, what a default ``Resources()`` already has) or the legacy default
stream: cuVS 26.10's sparse path issues ``cudaMemset`` / ``cudaMemsetAsync``
with no stream argument (sparse_knn.cuh ``batch_dists``; l2_distance.cuh
``compute_l2``'s norm buffers, cosine), and libcuvs is built with
``-DCUDA_API_PER_THREAD_DEFAULT_STREAM`` (cpp/cmake/modules/
ConfigureCUDA.cmake), so those land on the calling thread's per-thread
default stream; on any other non-default stream they are unordered with the
kernels that accumulate into the same buffers.

``neighbors`` int32 [nq, k], ``distances`` float32 [nq, k], C-contiguous,
written in place. metric 0 = inner product (largest first), 1 = cosine
(cuVS returns 1 - cos, smallest first).
"""

import ctypes
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = "libnova_cuvs_sparse.so"
LIB_ENV = "NOVA_CUVS_SPARSE_LIB"
ABI = 1
CUDA_STREAM_PER_THREAD = 0x2  # cudaStreamPerThread (driver_types.h)
METRICS = {"ip": 0, "inner_product": 0, "cosine": 1}

_lib = None


def lib_path(path=None):
    """The shim to load: ``path``, else $NOVA_CUVS_SPARSE_LIB, else the one
    build.sh writes next to this file."""
    return path or os.environ.get(LIB_ENV) or os.path.join(HERE, LIB_NAME)


def load(path=None):
    """Load (once) and return the shim with its signatures set. Raises
    RuntimeError with what to do when it is missing or stale."""
    global _lib
    if _lib is not None:
        return _lib
    p = lib_path(path)
    if not os.path.exists(p):
        raise RuntimeError(f"{p} not found: build it with "
                           f"clean_benchmarks/baselines/cuvs/sparse_binding/build.sh (PYTHON=<the cuvs "
                           f"venv's python>), or point ${LIB_ENV} at it")
    try:  # the wheel's libcuvs.so (+ librmm, rapids_logger, ...) first
        import libcuvs
        libcuvs.load_library()
    except ImportError:  # conda: the loader finds libcuvs.so itself
        pass
    lib = ctypes.CDLL(p)
    lib.nova_cuvs_sparse_abi.restype = ctypes.c_int
    if lib.nova_cuvs_sparse_abi() != ABI:
        raise RuntimeError(f"{p}: ABI {lib.nova_cuvs_sparse_abi()} != {ABI}; rebuild it")
    i32, vp = ctypes.c_int32, ctypes.c_void_p
    lib.nova_cuvs_sparse_knn.argtypes = (
        [ctypes.c_size_t, vp, vp, vp, i32, i32, vp, vp, vp, i32, i32,
         i32, i32, i32, i32, i32, vp, vp, ctypes.c_char_p, ctypes.c_size_t])
    lib.nova_cuvs_sparse_knn.restype = ctypes.c_int
    lib.nova_cuvs_sparse_default_batches.argtypes = [ctypes.POINTER(i32), ctypes.POINTER(i32)]
    lib.nova_cuvs_sparse_default_batches.restype = None
    _lib = lib
    return lib


def default_batches(lib=None):
    """cuVS's own ``sparse_search_params`` defaults ``(batch_size_index,
    batch_size_query)`` (2 << 14 each in 26.10), from the header the shim was
    built against."""
    lib = lib or load()
    bi, bq = ctypes.c_int32(), ctypes.c_int32()
    lib.nova_cuvs_sparse_default_batches(ctypes.byref(bi), ctypes.byref(bq))
    return bi.value, bq.value


_TYPESTR = {"int32": "<i4", "float32": "<f4"}


def _ptr(a, dtype, n, what):
    """Device pointer of ``a`` (``__cuda_array_interface__``), checked: dtype,
    ``n`` elements, C-contiguous (``n`` None: any length)."""
    cai = getattr(a, "__cuda_array_interface__", None)
    if cai is None:
        raise TypeError(f"{what}: not a CUDA array (no __cuda_array_interface__)")
    if cai["typestr"] != _TYPESTR[dtype]:
        raise TypeError(f"{what}: {cai['typestr']}, need {dtype}")
    shape = tuple(cai["shape"])
    size = 1
    for s in shape:
        size *= s
    if n is not None and size != n:
        raise ValueError(f"{what}: {size} elements, need {n}")
    strides = cai.get("strides")
    if strides is not None:
        itemsize = 4
        want, acc = [], itemsize
        for s in reversed(shape):
            want.append(acc)
            acc *= s
        if tuple(reversed(want)) != tuple(strides) and size > 1:
            raise ValueError(f"{what}: not C-contiguous (strides {strides})")
    return cai["data"][0] or None


def knn(res_handle, index, queries, n_cols, k, metric, batch_size_index,
        batch_size_query, neighbors, distances, lib=None):
    """One cuVS sparse brute-force build + search. ``index`` / ``queries``:
    ``(indptr, indices, values, n_rows, nnz)`` with CUDA int32 / int32 /
    float32 arrays. Writes ``neighbors`` / ``distances`` [n_query_rows, k];
    raises RuntimeError with cuVS's message on failure."""
    lib = lib or load()
    m = METRICS[metric] if isinstance(metric, str) else int(metric)
    ip, ix, iv, n_rows, nnz = index
    qp, qx, qv, nq, qnnz = queries
    args = [ctypes.c_size_t(res_handle),
            _ptr(ip, "int32", n_rows + 1, "index indptr"),
            _ptr(ix, "int32", nnz, "index indices"),
            _ptr(iv, "float32", nnz, "index values"), n_rows, nnz,
            _ptr(qp, "int32", nq + 1, "query indptr"),
            _ptr(qx, "int32", qnnz, "query indices"),
            _ptr(qv, "float32", qnnz, "query values"), nq, qnnz,
            n_cols, k, m, batch_size_index, batch_size_query,
            _ptr(neighbors, "int32", nq * k, "neighbors"),
            _ptr(distances, "float32", nq * k, "distances")]
    err = ctypes.create_string_buffer(2048)
    rc = lib.nova_cuvs_sparse_knn(*args, err, len(err))
    if rc != 0:
        raise RuntimeError(f"cuVS sparse brute force failed: {err.value.decode(errors='replace')}")
