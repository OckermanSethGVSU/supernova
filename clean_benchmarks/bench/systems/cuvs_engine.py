"""The cuVS engines for a harness leg: dense brute force (CuvsBackend) and sparse brute force
(SparseCuvsBackend, via the ctypes binding in baselines/cuvs/sparse_binding/), plus what they
share: the C API, the RMM pool, stream binding, cuVS's own top-k merge recipe (CalcTruthMerge)
and the --cuvs-sparse-batches grammar.
"""

import argparse
import contextlib
import os
import re

import numpy as np

from bench.record import schema
from bench.timing import now
from bench.systems import streaming as base
from bench.data.formats import sparse


def cuvs_c_api():
    """libcuvs_c through ctypes, with the two stream calls' signatures set.
    ``import cuvs`` already loaded it; libcuvs.load_library() just returns
    the handle already in the process."""
    import ctypes
    lib = None
    try:
        import libcuvs
        lib = next((l for l in libcuvs.load_library()
                    if os.path.basename(l._name) == "libcuvs_c.so"), None)
    except ImportError:  # conda: no libcuvs module, the loader finds the .so
        pass
    if lib is None:
        lib = ctypes.CDLL("libcuvs_c.so")
    lib.cuvsStreamSet.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cuvsStreamGet.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.cuvsStreamSet.restype = lib.cuvsStreamGet.restype = ctypes.c_int
    return lib


CUVS_SUCCESS = 1  # cuvsError_t: CUVS_ERROR = 0, CUVS_SUCCESS = 1 (c_api.h)


# --cuvs-rmm-pool: the default initial pool size, and the values that turn the
# pool off (RMM's default resource: a cudaMalloc / cudaFree per allocation).
CUVS_RMM_POOL_DEFAULT = "1GiB"
CUVS_RMM_POOL_OFF = ("off", "none", "default", "0")


def cuvs_pool_spec(args):
    """The RMM pool spec a run uses, or None for RMM's default resource.

    On by default for ``--backend cuvs``, dense (filtered included) and
    sparse. The sparse search's workspace (sparse_knn.cuh's device_uvector
    tiles) comes from RMM's current device resource inside libcuvs.so, so
    the pool reaches it too (sparse_binding/validate_gpu.py --rmm-pool)."""
    if args.backend != "cuvs" or getattr(args, "modality", "dense") not in ("dense", "sparse"):
        return None
    spec = getattr(args, "cuvs_rmm_pool", CUVS_RMM_POOL_DEFAULT)
    spec = CUVS_RMM_POOL_DEFAULT if spec is None else str(spec).strip()
    return None if spec.lower() in CUVS_RMM_POOL_OFF else spec


def cuvs_set_rmm_pool(spec):
    """Make an RMM pool the current device resource before the backend is
    built, so cuVS's temporary allocations come from it instead of a
    cudaMalloc / cudaFree per tile (RMM's default cuda_memory_resource).

    On by default (``--cuvs-rmm-pool off`` restores RMM's default). This is
    a fairness rule: the unpooled run is not what cuVS's authors run or
    recommend, and it handicaps cuVS against the other systems:

    - cuVS's own streaming ground-truth tool sets this exact pool (1 GiB
      initial, growable) before the same build + search loop
      (cuvs_bench/generate_groundtruth/__main__.py), its C++ benchmark runner
      installs a pool globally, and the brute_force.search docstring
      recommends one;
    - torch (nova-bf) and FAISS's StandardGpuResources both pool device
      memory by default, so without a pool cuVS alone pays cudaMalloc /
      cudaFree inside every call;
    - a profile of one Coyo-VE file showed 345 cudaMalloc / cudaFree pairs
      per call, cudaFree blocking the host for 268 ms of a ~323 ms call.
    It sets the library's allocator from the user side; cuVS is unchanged.
    ``spec``: the initial pool size, e.g. "8GiB" or "50%" (of the device's
    free memory); the pool may grow. Returns the size in bytes."""
    import rmm
    import rmm.mr
    spec = str(spec).strip()
    if spec.endswith("%"):
        import torch
        free, _ = torch.cuda.mem_get_info()
        size = int(free * float(spec[:-1]) / 100) // 256 * 256
    else:
        units = {"gib": 1 << 30, "mib": 1 << 20, "gb": 10 ** 9, "mb": 10 ** 6}
        low = spec.lower()
        unit = next((u for u in units if low.endswith(u)), None)
        size = int(float(low[:-len(unit)]) * units[unit]) if unit else int(low)
        size = size // 256 * 256
    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=size))
    return size


def cuvs_bind_stream(res, handle):
    """Bind cuvs ``Resources`` ``res`` to CUDA stream ``handle`` (an int, torch's
    ``Stream.cuda_stream``) with the C API ``cuvsStreamSet``, then read it
    back with ``cuvsStreamGet`` so a silent mis-bind cannot happen.

    Not ``Resources(stream=handle)``: in cuvs 26.10 its Cython cast of the
    Python int to a pointer passes the object's address, not its value, so
    cuVS enqueues on a garbage stream (``Resources.sync()`` fails with
    cudaErrorInvalidResourceHandle, ``brute_force.search`` segfaults)."""
    import ctypes
    lib = cuvs_c_api()
    c_res = ctypes.c_void_p(res.get_c_obj())
    rc = lib.cuvsStreamSet(c_res, ctypes.c_void_p(handle))
    if rc != CUVS_SUCCESS:
        raise RuntimeError(f"cuvsStreamSet failed (cuvsError_t {rc})")
    got = ctypes.c_void_p()
    rc = lib.cuvsStreamGet(c_res, ctypes.byref(got))
    if rc != CUVS_SUCCESS or got.value != handle:
        raise RuntimeError(f"cuvs stream not bound: wanted {handle:#x}, cuvsStreamGet "
                           f"gave {got.value or 0:#x} (cuvsError_t {rc})")


# ------------------------------------------------- cuVS's own exact-search recipe
# NVIDIA cuvs python/cuvs_bench/cuvs_bench/generate_groundtruth/__main__.py, calc_truth
# (lines 195-235; benchmarks/external/cuvs): per corpus batch (here each file as the reader
# hands it, the unit every system streams), build + search + resources.sync(), the batch's
# ids offset by its start, then the running top-k kept on the GPU with CuPy
# (concatenate, argsort, take_along_axis), copied to the host once at the end.
# --cuvs-merge: device = that recipe (the default); host = each file's results merged on the
# host with faiss.ResultHeap (the `per_file_host` comparison variant).
CUVS_MERGES = ("device", "host")
# --cuvs-sync (device merge only): per-file = calc_truth's resources.sync() after each file
# (the default); none = no per-file sync: the merge is ordered by the stream alone, and the
# host waits for a block's work only before its buffers are reused (one block in flight)
CUVS_SYNCS = ("per-file", "none")


def neighbor_index_accumulator_dtype(n_base):
    """cuvs_bench generate_groundtruth/utils.py:71-80, copied: int64 ids for a base of more
    than int32-max rows, else int32. ``n_base`` None (not known to the caller) = int64."""
    if n_base is None or n_base > np.iinfo(np.int32).max:
        return np.int64
    return np.int32


def offset_neighbor_indices(indices, batch_offset, n_base):
    """cuvs_bench generate_groundtruth/utils.py:90-93, copied: shift a batch's local ids by
    its start row (no -1 slots in their tool: k <= batch rows, no filter)."""
    dtype = neighbor_index_accumulator_dtype(n_base)
    return indices.astype(dtype) + batch_offset


def cupy_module():
    """CuPy, imported on first use: only the device-merge path needs it."""
    try:
        import cupy
    except ImportError as e:
        raise RuntimeError("--cuvs-merge device needs cupy (cuVS's own ground-truth tool "
                           "merges with it; the paper env installs it); "
                           f"use --cuvs-merge host for the per-file host merge (import failed: {e})"
                           ) from e
    return cupy


class CalcTruthMerge:
    """The running top-k of NVIDIA's calc_truth (generate_groundtruth/__main__.py:195-235),
    step for step, on ``xp`` (cupy on the GPU; a numpy stand-in in the tests).

    ``metric`` is cuVS's name: "inner_product" (similarities, sorted on negated keys) or
    "cosine" (cuVS's native 1 - cos, a distance, sorted ascending like their sqeuclidean).
    Ties: their default ``xp.argsort`` (no id tie-break), so exact ties at the k-th place may
    keep another id than the per-file host merge; the checks are tie-aware.

    One addition their tool has no need for: a filtered search's empty slots (worst score, id
    -1) are already global (offset 0 is passed), so -1 stays -1 and sorts last."""

    def __init__(self, xp, k, metric, n_base=None):
        if metric not in ("inner_product", "cosine"):
            raise ValueError(metric)
        self.xp, self.k, self.metric, self.n_base = xp, k, metric, n_base
        self.distances = self.indices = None
        self.batches = 0

    def add(self, D, Ind, offset):
        xp = self.xp
        D, Ind = xp.asarray(D), xp.asarray(Ind)                        # __main__.py:219
        Ind = offset_neighbor_indices(Ind, offset, self.n_base)        # __main__.py:220
        self.batches += 1
        if self.distances is None:                                     # __main__.py:222-224
            self.distances, self.indices = D, Ind
            return
        distances = xp.concatenate([self.distances, D], axis=1)        # __main__.py:226
        indices = xp.concatenate([self.indices, Ind], axis=1)          # __main__.py:227
        sort_keys = -distances if self.metric == "inner_product" else distances  # :228
        idx = xp.argsort(sort_keys, axis=1)[:, :self.k]               # __main__.py:229
        self.distances = xp.take_along_axis(distances, idx, axis=1)   # __main__.py:230
        self.indices = xp.take_along_axis(indices, idx, axis=1)       # __main__.py:231

    def result(self, nq):
        """(D float32, I int64) on the host: the one copy back (their write path's ``.get()``,
        utils.py:96-101; cp.asnumpy), padded to k columns as every engine pads a short result (worst
        score, -1) when the corpus has fewer than k rows (or none)."""
        xp = self.xp
        worst = np.finfo(np.float32).max if self.metric == "cosine" else np.finfo(np.float32).min
        if self.distances is None:
            return (np.full((nq, self.k), worst, np.float32), np.full((nq, self.k), -1, np.int64))
        D = np.asarray(xp.asnumpy(self.distances), dtype=np.float32)
        I = np.asarray(xp.asnumpy(self.indices)).astype(np.int64)
        # an empty slot (id -1: a filtered query with fewer than k allowed rows, or none in a
        # file it was not searched in, padded +-inf by the harness's expansion) carries the
        # worst float32, the score every engine's empty slots carry after the host merge
        D[I < 0] = worst
        short = self.k - D.shape[1]
        if short > 0:
            D = np.pad(D, ((0, 0), (0, short)), constant_values=worst)
            I = np.pad(I, ((0, 0), (0, short)), constant_values=-1)
        return D, I


class CalcTruthMixin:
    """cuVS's own recipe (CalcTruthMerge) wired into ResidentBackend's per-block flow.

    begin_pass() (run_pass, inside the LegClock) starts a pass; each block, as the reader
    hands it, is one calc_truth batch: its search goes into the device merge instead of a host
    heap (``device_merge``: True); pass_results() is the one copy back."""

    device_merge = False
    cuvs_sync = "per-file"
    n_base = None   # the corpus rows of the pass (harness_cli, from the footers): the id dtype
    _merge = None
    _xp = None

    def configure_merge(self, merge="device", xp=None):
        if merge not in CUVS_MERGES:
            raise ValueError(merge)
        self.device_merge = merge == "device"
        if self.device_merge:
            self._xp = xp if xp is not None else cupy_module()

    def configure_sync(self, sync="per-file"):
        """--cuvs-sync: per-file (calc_truth) or none; none needs the device merge."""
        if sync not in CUVS_SYNCS:
            raise ValueError(sync)
        if sync == "none" and not self.device_merge:
            raise ValueError("--cuvs-sync none needs --cuvs-merge device")
        self.cuvs_sync = sync

    @property
    def defer_wait(self):
        """True: no per-file sync; run_pass keeps one block in flight and waits for a
        block's end only before its buffers can be reused (see loop.run_pass)."""
        return self.device_merge and self.cuvs_sync == "none"

    def sync_str(self):
        """The cuvs_sync column: per-file / none for the device merge, n/a for the host merge."""
        return self.cuvs_sync if self.device_merge else "n/a"

    def merge_str(self):
        """The cuvs_merge column: device:file (each file as handed, merged on the GPU) or host."""
        return "device:file" if self.device_merge else "host"

    def _metric_name(self):
        return "inner_product" if self.metric == "ip" else "cosine"

    def _xp_stream(self):
        """CuPy's ops on this backend's stream (an ExternalStream over its handle), so they
        are ordered with the torch / cuVS work around them; off CUDA a no-op."""
        if self.stream is None or not self._cuda:
            return contextlib.nullcontext()
        return self._xp.cuda.ExternalStream(self.stream.cuda_stream)

    def begin_pass(self):
        self._merge = CalcTruthMerge(self._xp, self.k, self._metric_name(), self.n_base)

    # the warm-up merge's shape: 2 queries x 2 columns per block, two blocks
    WARM_ROWS = WARM_K = 2

    def warm_merge(self):
        """Run one tiny two-block CalcTruthMerge before any timed pass, so every CuPy kernel
        the merge uses is compiled (or loaded from CuPy's cache) here, not inside the first
        timed merge. Same dtypes, metric and stream as a pass; the result is dropped and no
        stats are recorded. Returns its wall seconds (added to setup_s). No-op for the host
        merge."""
        if not self.device_merge:
            return 0.0
        t0 = now()
        torch = self.torch
        r, k = self.WARM_ROWS, self.WARM_K
        with self._on_stream():
            D = torch.arange(r * k, dtype=torch.float32, device=self.device).reshape(r, k)
            I = torch.arange(r * k, dtype=torch.int64, device=self.device).reshape(r, k)
            m = CalcTruthMerge(self._xp, k, self._metric_name(), self.n_base)
            with self._xp_stream():
                m.add(D, I, 0)        # the first block: assigned (asarray x2, the id offset)
                m.add(D, I, r * k)    # the second: the merge itself
                m.result(r)           # the copy back (asnumpy x2)
            del D, I, m
        if self._cuda and self.stream is not None:
            self.stream.synchronize()
        return now() - t0

    def _sync_res(self):
        """__main__.py:215 ``resources.sync()`` after each batch's search (device merge only;
        the per-file host path is unchanged; skipped with --cuvs-sync none)."""
        if self.device_merge and self.cuvs_sync == "per-file":
            self.res.sync()

    def _merge_add(self, D, I, offset):
        with self._xp_stream():
            self._merge.add(D, I, offset)

    def pass_results(self, nq):
        """The one copy back of the merged top-k (``d2h``), inside the LegClock."""
        ev = self.events
        with self._on_stream():
            a = self._mark() if ev else None
            with self._xp_stream():
                D, I = self._merge.result(nq)
            b = self._mark() if ev else None
        if ev:
            if self._cuda:
                b.synchronize()
            self.stats.add("d2h", self._elapsed(a, b))
        self._merge = None
        return D, I


# how cuVS's filtered search marks slots past a query's allowed rows: "id"
# (an out-of-range / negative id) or "distance" (a max / inf distance too);
# set from a GPU probe of cuVS 26.10
CUVS_FILTER_FILL = "distance"
# raft select_k uses warpsort up to this k (radix above): the k range where the
# SDDMM short-row bug can occur (cuvs_filter_small_k_calls)
CUVS_WARPSORT_MAX_K = 256


class CuvsBackend(CalcTruthMixin, base.ResidentBackend):
    """cuvs.neighbors.brute_force on one GPU (engine "cuvs"), resident only
    (its Python API takes device arrays only).

    Allocator: make_backend installs an RMM pool before this is built (see
    cuvs_set_rmm_pool for why); with ``--cuvs-rmm-pool off`` the workspace
    is a cudaMalloc / cudaFree per call, inside engine_call.

    Stream: one explicit non-default torch stream, bound to ``Resources()``
    with cuvs_bind_stream, and every op of setup() and search() runs on it.
    So the events bracket cuVS's kernels as they bracket bfKnn for FAISS,
    cuVS adds no sync of its own, and torch's allocator reuses a freed
    block only behind cuVS's reads on that same stream.

    Per block (_engine_search): xq and xb must have the same dtype (cuVS
    checks only the dtype code, so fp16 index + fp32 queries would be a
    silent type pun) and be contiguous; k_eff = min(k, rows); outputs are
    pre-allocated (int64 ids, float32 distances); build (timed as host wall
    ``build``; for IP just a view, no copy) then search; the index, a view
    of xb_t, is dropped before xb_t.

    cosine: build(metric="cosine"), no normalisation by the harness. cuVS
    computes the block norms in build and the query norms in every search,
    and returns 1 - cos ascending; run_pass turns it back into cos in the
    merge bucket. k_eff keeps every returned slot a real row, so the host's
    -1 padding is never converted.
    """

    name = "cuvs"
    engine = "cuvs"
    CUVS_METRIC = {"ip": "inner_product", "cosine": "cosine"}

    def __init__(self, k, dtype=np.float32, device="cuda", events=True, metric="ip"):
        try:
            # torch is already loaded (cudalibs.torch_first in harness_cli), so
            # libcuvs's CUDA sonames resolve to torch's pip copies; here it
            # only turns a missing torch into an ImportError
            import torch  # noqa: F401
            import cuvs
            from cuvs.common import Resources
            from cuvs.neighbors import brute_force
        except ImportError as e:
            raise RuntimeError("--backend cuvs needs the cuvs package (the patched source "
                               "build from env/build.sh) "
                               f"(import failed: {e})") from e
        if metric not in schema.METRICS:
            raise ValueError(metric)
        super().__init__(k, dtype, device, events)
        self.queries_mode = "resident"
        self.metric = metric
        self.returns_distance = metric == "cosine"  # 1 - cos, ascending
        self.engine_version = getattr(cuvs, "__version__", "unknown")
        self._bf = brute_force
        self._init_torch()
        self.stream = self._make_stream()
        # never Resources(stream=...): its Cython cast passes the int's
        # PyObject address, not its value (see cuvs_bind_stream)
        self.res = Resources()
        if self.stream is not None:  # None: torch's CPU device (tests), nothing to share
            handle = self.stream.cuda_stream
            if not handle:
                raise RuntimeError("cuvs needs a non-default stream: handle 0 is the "
                                   "legacy default stream, not torch's explicit one")
            cuvs_bind_stream(self.res, handle)

    # ------------------------------------------------------ --filter-mode bitmap
    _flt = None   # (FilterBlock, queries, host block) during search_filtered

    def search_filtered(self, xb, fblock, queries=None, ahead=None, ahead_fblock=None,
                        offset=0, post=None, ahead_keep=None):
        """Search one block with cuVS's own prefilter (bitmap mode); =
        complete(submit_filtered(...)).

        A static filter goes in as a bitset (``filters.from_bitset``: one row
        mask for all queries), a per-query filter as a bitmap
        (``filters.from_bitmap``: nq x rows bits, row-major and contiguous,
        LSB first in uint32 words). ``queries``: the subset of the setup()
        queries to search (dead ones dropped). Which internal path runs is
        cuVS's own stock choice.

        Layout: the reader pads each mask to whole bytes (padding bits 0), so
        a bitmap over rows8 = ceil(rows / 8) * 8 rows is exactly the masks laid
        end to end. A per-query block is therefore uploaded into a rows8-row
        device tensor with up to 7 zero rows (never allowed): no bit
        re-packing. Only the distinct masks are uploaded, with the block's
        vectors (so inside the h2d span); the per-query bitmap is expanded on
        the device inside the search span (filter_bits).
        ``ahead`` / ``ahead_fblock`` (--h2d-overlap): block i+1 and its filter,
        uploaded (padded the same way) before this search."""
        return self.complete(self.submit_filtered(xb, fblock, queries, ahead, ahead_fblock,
                                                  offset, post, ahead_keep))

    def submit_filtered(self, xb, fblock, queries=None, ahead=None, ahead_fblock=None,
                        offset=0, post=None, ahead_keep=None):
        if fblock.kind == "static" and queries is not None:
            raise ValueError("a static filter searches every query")
        self._flt = (fblock, queries, xb)
        # rows8 per host array uploaded in this call (the block unless it went
        # ahead, and block i+1 under --h2d-overlap), keyed by identity
        self._pad_rows = {}
        for x, f in ((xb, fblock), (ahead, ahead_fblock)):
            if x is not None and f is not None:
                # every filtered block's masks ride with its vectors (_to_dev)
                self._pad_rows[id(x)] = (x, None if f.kind == "static"
                                         else f.nbytes_per_mask * 8, f)
        try:
            tok = self.submit(xb, ahead=ahead, offset=offset, post=post,
                              ahead_keep=ahead_keep)
        finally:
            self._flt = None
            self._pad_rows = None
        tok["fblock"] = fblock  # its host masks live until complete(), as xb does
        return tok

    _pad_rows = None
    # id(host block) -> (host block, device masks): uploaded with the block's
    # vectors in _to_dev (block i+1's on the copy stream under --h2d-overlap,
    # during block i's search), taken by _prefilter when that block is searched
    _masks_dev = None

    def _to_dev(self, x):
        got = None if self._pad_rows is None else self._pad_rows.pop(id(x), None)
        if got is None or got[0] is not x:
            return super()._to_dev(x)
        rows8, fb = got[1], got[2]
        torch = self.torch
        if rows8 is None:  # static: the block as is
            dst = super()._to_dev(x)
        else:
            if x.ndim != 2 or x.shape[0] > rows8:
                raise ValueError(f"cuvs bitmap: block of {x.shape[0]} rows, masks cover {rows8}")
            src = torch.from_numpy(x)
            dst = torch.empty((rows8, x.shape[1]), dtype=src.dtype, device=self.device)
            dst[x.shape[0]:].zero_()
            dst[:x.shape[0]].copy_(src, non_blocking=self._nb)
        # the masks right behind the vectors, on the same stream (so inside the
        # same h2d span: c0..c1 on the copy stream, e0..e1 without overlap)
        if self._masks_dev is None:
            self._masks_dev = {}
        self._masks_dev[id(x)] = (x, self._upload_host(fb.masks))
        return dst

    def drain(self):
        super().drain()
        self._masks_dev = None  # an upload ahead that will never be searched

    def _u32(self, u8):
        """uint8 device bytes -> a uint32 view (zero-padded to 4 bytes) for cuVS."""
        torch = self.torch
        u8 = u8.reshape(-1)
        pad = (-u8.numel()) % 4
        if pad:
            u8 = torch.cat([u8, torch.zeros(pad, dtype=torch.uint8, device=u8.device)])
        return u8.contiguous().view(torch.uint32)

    def _block_masks(self, xb, fb):
        """The device masks uploaded with block ``xb`` (_to_dev), handed to
        this (compute) stream; an upload on demand on it when there is none
        (a backend whose block took another path)."""
        got = None if self._masks_dev is None else self._masks_dev.pop(id(xb), None)
        if got is None or got[0] is not xb:
            return self._upload_small(fb.masks)
        masks_t = got[1]
        if self._cuda and self.copy_stream is not None:
            # uploaded on the copy stream: this stream already waits on its c1
            # (_await, before the engine call); keep the memory until this
            # stream is done with it
            masks_t.record_stream(self.stream)
        return masks_t

    def _prefilter(self, xq_t):
        """(queries, prefilter, real rows) for the pending filter. The masks
        came up with the block (_block_masks); the small query / mask index
        arrays are non-blocking from pinned staging (_upload_small)."""
        from cuvs.neighbors import filters
        fb, queries, xb = self._flt
        masks_t = self._block_masks(xb, fb)
        if fb.kind == "static":
            bits = self._u32(masks_t[0])
            return xq_t, filters.from_bitset(bits), bits, fb.n_rows
        mq = fb.mask_of_query if queries is None else fb.mask_of_query[queries]
        if queries is not None:
            xq_t = xq_t.index_select(0, self._upload_small(np.asarray(queries, np.int64)))
        bm = masks_t.index_select(0, self._upload_small(np.asarray(mq, np.int64)))
        bits = self._u32(bm)
        return xq_t, filters.from_bitmap(bits), bits, fb.n_rows

    def _engine_search(self, xq_t, xb_t):
        torch = self.torch
        if xq_t.dtype != xb_t.dtype:
            raise TypeError(f"cuvs: queries are {xq_t.dtype}, block is {xb_t.dtype}; "
                            "cuVS would reinterpret the index as the queries' type")
        if not (xq_t.is_contiguous() and xb_t.is_contiguous()):
            raise ValueError("cuvs: queries and block must be C-contiguous")
        pf = bits = None
        n_real = xb_t.shape[0]
        if self._flt is not None:
            # filter_bits: the per-query bitmap's expansion on the device (and
            # the small query / mask index uploads), timed with device events as
            # a sub-span of search (like SPARe's stage events). The masks
            # themselves came up with the block's vectors, in its h2d span
            b0 = self._mark() if self.events else None
            xq_t, pf, bits, n_real = self._prefilter(xq_t)
            if b0 is not None:
                self._add_stage("filter_bits", b0, self._mark())
        return self._bf_call(xq_t, xb_t, pf, bits, n_real)

    def _bf_call(self, xq_t, xb_t, pf, bits, n_real):
        """One cuVS build + search of ``xb_t`` (one file's block): the
        calc_truth loop body's build / search / resources.sync()
        (generate_groundtruth/__main__.py:213-215; the sync on the device-merge path only)."""
        torch = self.torch
        nq, k_eff = xq_t.shape[0], min(self.k, n_real)
        if pf is None:
            I_t = torch.empty((nq, k_eff), dtype=torch.int64, device=xb_t.device)
            D_t = torch.empty((nq, k_eff), dtype=torch.float32, device=xb_t.device)
        else:
            # filtered: pre-filled with (worst, -1), so a slot cuVS might leave
            # unwritten past a query's allowed rows can never surface a stale id
            f32 = torch.finfo(torch.float32)
            I_t = torch.full((nq, k_eff), -1, dtype=torch.int64, device=xb_t.device)
            D_t = torch.full((nq, k_eff), f32.max if self.returns_distance else f32.min,
                             dtype=torch.float32, device=xb_t.device)
        lib = self._lib()
        lib.__enter__()  # build + search: one library span (closed after search)
        t = now()
        index = self._bf.build(xb_t, metric=self.CUVS_METRIC[self.metric],
                               resources=self.res)                     # __main__.py:213
        self.last_build = now() - t
        self.stats.add("build", self.last_build)
        kw = {} if pf is None else {"prefilter": pf}
        if pf is not None and k_eff <= CUVS_WARPSORT_MAX_K:
            # diagnostic only: raft's CSR select_k (cuVS's SDDMM path, selectivity
            # < 3%) is wrong for rows with fewer than k entries when it runs its
            # multi-block warpsort (k <= 256, small query batch); radix (k > 256)
            # is exact. A run whose count is 0 never met that case. Counted per cuVS call.
            self.stats.add("cuvs_filter_small_k_calls", 1)
        try:
            self._bf.search(index, xq_t, k_eff, neighbors=I_t, distances=D_t,
                            resources=self.res, **kw)                  # __main__.py:214
            self._sync_res()                                           # __main__.py:215
        finally:
            lib.__exit__(None, None, None)
        del index  # a non-owning view of xb_t: dropped before xb_t
        if pf is not None:
            del pf, bits   # the bits live (on this stream) until the search is enqueued
            D_t, I_t = self._filtered_fill(D_t, I_t, n_real)
        return D_t, I_t

    def _filtered_fill(self, D_t, I_t, n_real):
        """Slots past a query's allowed rows -> (worst score, -1), the padding
        every engine's short results carry (worst = the highest float32 for a
        distance engine, cosine; the lowest for inner product). A slot is
        empty when its id is out of range, or (CUVS_FILTER_FILL "distance")
        its score is non-finite or the worst float32 (how cuVS 26.10 marks
        them was established by a GPU probe)."""
        torch = self.torch
        f32 = torch.finfo(torch.float32)
        bad = (I_t < 0) | (I_t >= n_real)
        if CUVS_FILTER_FILL == "distance":
            worst = D_t >= f32.max if self.returns_distance else D_t <= f32.min
            bad |= ~torch.isfinite(D_t) | worst
        fill = f32.max if self.returns_distance else f32.min
        I_t = torch.where(bad, torch.full_like(I_t, -1), I_t)
        D_t = torch.where(bad, torch.full_like(D_t, fill), D_t)
        return D_t, I_t


def cuvs_sparse_binding():
    """The ctypes binding module (baselines/cuvs/sparse_binding/cuvs_sparse.py),
    imported on first use."""
    return base.sibling_module("cuvs_sparse", "knn", os.path.join("cuvs", "sparse_binding"))


class SparseCuvsBackend(CalcTruthMixin, base.ResidentBackend):
    """cuVS's sparse exact brute force on one GPU (engine "cuvs"): C++
    ``brute_force::build`` on a CSR view + ``search``, through the shim in
    baselines/cuvs/sparse_binding (cuVS 26.10 has no C or Python API for it).

    It uses the dense engines' per-block flow unchanged (ResidentBackend
    submit / complete, every --h2d / --d2h mode); only the block step
    (_block_search) is overridden: e0 [one H2D of the packed bytes] e1
    [fp16-stored values widened to float32] e2 [cuVS build + search, RMM
    workspace included; int32 ids widened to int64 and globalised on the
    device] e3 [D / I to the host] e4. The CSR parts are device views of the
    uploaded bytes (sparse.py's 256-byte-aligned layout). Queries are
    uploaded once in setup().

    Stream: cudaStreamPerThread (0x2), as a torch ExternalStream, bound with
    cuvs_bind_stream. It is what a default Resources() uses and the only
    stream on which cuVS 26.10's sparse path is ordered: it issues cudaMemset
    without a stream argument, which libcuvs's per-thread-default-stream
    build puts on this stream; on another stream those memsets would race
    the kernels that accumulate into the buffers. Every torch op of the call
    runs on it too.

    Parameters: cuVS's stock ``sparse_search_params`` by default
    (batch_size_index = batch_size_query = 2 << 14, read from the build);
    ``batches`` (--cuvs-sparse-batches) overrides either side with a number,
    ``stock`` keeps that side's default, ``file`` (index side) is one batch
    per corpus block. These are parameters cuVS exposes. Correct results need
    the patched cuVS (baselines/cuvs/patches/0001): stock 26.10 merges index
    batches keeping the smallest values (wrong inner-product top-k once a
    block exceeds batch_size_index rows) and mis-strides an index batch of
    fewer than k rows (wrong for every metric). setup() runs ``self_test``,
    which trips both, and refuses an unfixed build before any pass.
    Memory: each (query batch x index batch) tile is a dense float32
    distance matrix plus an int32 id matrix, 8 B a cell: 8 GiB at the stock
    32768 x 32768 tile.
    """


    # ids globalised on the device by ResidentBackend.submit, as for the dense
    # engines; required by --d2h-overlap
    applies_offset = True
    name = "cuvs"
    engine = "cuvs"
    METRIC = {"ip": 0, "cosine": 1}

    def __init__(self, k, device="cuda", events=True, metric="ip", knn=None, binding=None,
                 batches=None):
        try:
            import torch  # noqa: F401 - already in (torch_first), as CuvsBackend
            import cuvs
            from cuvs.common import Resources
        except ImportError as e:
            raise RuntimeError("--backend cuvs needs the cuvs package (the patched source "
                               "build from env/build.sh) "
                               f"(import failed: {e})") from e
        if metric not in schema.METRICS:
            raise ValueError(metric)
        super().__init__(k, np.float32, device, events)
        self.queries_mode = "resident"
        self.metric = metric
        self.returns_distance = metric == "cosine"  # 1 - cos, ascending
        self.engine_version = getattr(cuvs, "__version__", "unknown")
        self._init_torch()
        binding = binding if binding is not None else (
            None if knn is not None else cuvs_sparse_binding())
        self._knn = knn if knn is not None else binding.knn
        self.default_batches = binding.default_batches() if binding is not None \
            else (2 << 14, 2 << 14)
        # --cuvs-sparse-batches: (index, query), each an int or "stock", the
        # index side also "file"; None = stock on both sides
        self.requested_batches = tuple(batches) if batches is not None else ("stock", "stock")
        self.res = Resources()
        self.stream = None
        if self._cuda:
            handle = 0x2  # cudaStreamPerThread
            self.stream = self.torch.cuda.ExternalStream(handle, device=self.device)
            cuvs_bind_stream(self.res, handle)
        self.batches = []  # (batch_size_index, batch_size_query) per call (tests)

    def setup(self, q):
        t0 = now()
        self.self_test()  # refuse an unfixed cuVS before anything else (ms)
        self.q = q
        with self._on_stream():
            self.q_t = tuple(self._to_dev(a) for a in (q.indptr, q.indices, q.values))
        self._sync()
        return now() - t0

    # self_test's corpus: 5 rows over 4 columns (negative scores included), 2
    # queries, k = 3, searched with batch_size_index = 2: three index batches
    # (two merges) and a last batch of 1 row < k. Scores are distinct integers
    # (no ties), so the exact top-3 is unique; unfixed cuVS returns other ids.
    _ST_X = np.array([[1, 1, 3, 5], [-3, -2, 4, 5], [-1, -1, 4, 0], [-1, 4, -1, 0],
                      [2, 1, -3, -3]], np.float32)
    _ST_Q = np.array([[3, 2, 3, 1], [2, -1, 0, 2]], np.float32)

    def self_test(self):
        """Check this cuVS build against numpy on _ST_X / _ST_Q.

        batch_size_index 2 forces a multi-batch merge and a short last batch,
        the two bugs the patched cuVS fixes. Raises RuntimeError on a
        mismatch: this build would return wrong top-k at stock batch sizes."""
        torch = self.torch
        X, Q, k = self._ST_X, self._ST_Q, 3

        def csr(m):
            nz = m != 0
            return (np.concatenate([[0], np.cumsum(nz.sum(1))]).astype(np.int32),
                    np.nonzero(nz)[1].astype(np.int32), m[nz].astype(np.float32))
        S = Q.astype(np.float64) @ X.T.astype(np.float64)
        want_i = np.argsort(-S, axis=1, kind="stable")[:, :k]
        want_d = np.take_along_axis(S, want_i, 1)
        (xp, xi, xv), (qp, qi, qv) = csr(X), csr(Q)
        with self._on_stream():
            dev = [self._to_dev(a) for a in (xp, xi, xv, qp, qi, qv)]
            I_t = torch.empty((Q.shape[0], k), dtype=torch.int32, device=dev[0].device)
            D_t = torch.empty((Q.shape[0], k), dtype=torch.float32, device=dev[0].device)
            self._knn(self.res.get_c_obj(), (dev[0], dev[1], dev[2], X.shape[0], len(xi)),
                      (dev[3], dev[4], dev[5], Q.shape[0], len(qi)), X.shape[1], k,
                      self.METRIC["ip"], 2, 2, I_t, D_t)
            got_i, got_d = I_t.cpu().numpy().astype(np.int64), D_t.cpu().numpy()
        self._sync()
        if not (np.array_equal(got_i, want_i) and np.allclose(got_d, want_d, atol=1e-5)):
            raise RuntimeError(
                "cuVS sparse self-test failed: inner-product top-k over several index "
                f"batches (and a batch of fewer than k rows) is wrong (got ids {got_i.tolist()}, "
                f"expected {want_i.tolist()}). This cuVS build lacks the sparse_knn.cuh fix: "
                "it would return "
                "wrong results at cuVS's stock batch sizes. Use the patched build (env/build.sh).")

    def params(self, rows, nq):
        """(batch_size_index, batch_size_query) for a block of ``rows`` rows."""
        return cuvs_sparse_effective_batches(self.requested_batches, self.default_batches,
                                             rows)

    def batches_str(self):
        """The sparse_batches column (see sparse_batches_str)."""
        return sparse_batches_str(self.requested_batches, self.default_batches)

    def _views(self, buf_t, c, o):
        return base.csr_views(self.torch, buf_t, c, o)

    def _engine_search(self, ip, ix, iv, c):
        torch = self.torch
        q = self.q
        nq, k_eff = q.n_rows, min(self.k, c.n_rows)
        n_cols = max(q.n_cols, c.n_cols, 1)
        bi, bq = self.params(c.n_rows, nq)
        self.batches.append((bi, bq))
        I_t = torch.empty((nq, k_eff), dtype=torch.int32, device=ip.device)
        D_t = torch.empty((nq, k_eff), dtype=torch.float32, device=ip.device)
        with self._lib():
            # build + search (one C++ call through the binding), then calc_truth's
            # resources.sync() (generate_groundtruth/__main__.py:213-215; device merge only)
            self._knn(self.res.get_c_obj(), (ip, ix, iv, c.n_rows, c.nnz),
                      (*self.q_t, nq, q.nnz), n_cols, k_eff, self.METRIC[self.metric], bi, bq,
                      I_t, D_t)
            self._sync_res()
        return D_t, I_t

    def _block_search(self, xb, xb_h, e1):
        """ResidentBackend's per-block hook for a packed sparse block.

        The block's bytes are already on the device (one H2D); its CSR parts
        are views of them, fp16 values are widened to float32 (timed as
        widen), then cuVS build + search. The int32 ids are widened to int64
        on the device so submit() can globalise them there, as for dense."""
        torch = self.torch
        xb_t = xb_h.pop()
        c = sparse.unpack(xb)  # header only: sizes and offsets (host)
        if c.nnz == 0 and sparse.is_projected(xb):
            # cuVS's sparse brute force crashes on a block without nonzeros;
            # --sparse-project makes one likelier, so refuse it clearly
            raise ValueError("--sparse-project query-vocab: a block is "
                             "empty after query-vocab projection: it has no "
                             f"nonzeros left ({c.n_rows} rows, every column outside the "
                             "queries' vocabulary); cuVS's sparse brute force cannot search "
                             "an all-empty block (it crashes): run without the projection "
                             "or with --backend spare / cpu")
        o = [int(v) for v in sparse._header(xb)[5:8]]
        ip, ix, iv = self._views(xb_t, c, o)
        e2 = e1
        if iv.dtype != torch.float32:
            iv = iv.to(torch.float32)
            if self.events:
                e2 = self._mark()
        D, I = self._engine_search(ip, ix, iv, c)
        return D, I.to(torch.int64), (e1, e2), (e2, e2), e2


# cuVS's stock sparse_search_params (batch_size_index, batch_size_query)
SPARSE_STOCK_BATCHES = (2 << 14, 2 << 14)
# --cuvs-sparse-batches words: stock = that side's build default; file (index
# side only) = one index batch per corpus block
SPARSE_BATCH_STOCK = "stock"
SPARSE_BATCH_FILE = "file"
_SPARSE_BATCH_INT = re.compile(r"[1-9][0-9]*")


def parse_cuvs_sparse_batches(text):
    """--cuvs-sparse-batches INDEXxQUERY -> (index, query). Each side is a
    positive integer (no sign, no leading zero), ``stock``, or, on the index
    side only, ``file``; the two are joined by ONE lowercase ``x``. Raises
    argparse.ArgumentTypeError on anything else."""
    parts = text.split("x")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected INDEXxQUERY, e.g. 8192x32768, 8192xstock, filexstock")
    out = []
    for side, v, words in (("index", parts[0], (SPARSE_BATCH_STOCK, SPARSE_BATCH_FILE)),
                           ("query", parts[1], (SPARSE_BATCH_STOCK,))):
        if v in words:
            out.append(v)
        elif _SPARSE_BATCH_INT.fullmatch(v) and int(v) <= 2**31 - 1:
            out.append(int(v))
        else:
            raise argparse.ArgumentTypeError(
                f"{text!r}: the {side} side {v!r} must be a positive integer (< 2^31) or "
                + " / ".join(words))
    return tuple(out)


def cuvs_sparse_effective_batches(requested, stock, rows):
    """The (batch_size_index, batch_size_query) cuVS gets for a block of
    ``rows`` rows: a number as given, ``stock`` as the build's default,
    ``file`` (index side) as the block's row count (at least 1)."""
    (ri, rq), (si, sq) = requested, stock
    bi = si if ri == SPARSE_BATCH_STOCK else max(1, int(rows)) if ri == SPARSE_BATCH_FILE \
        else int(ri)
    bq = sq if rq == SPARSE_BATCH_STOCK else int(rq)
    return bi, bq


def sparse_batches_str(requested, stock):
    """The sparse_batches column: INDEXxQUERY, a ``stock`` side as the
    build's number (e.g. 32768x32768), ``file`` as the word (it varies per file)."""
    (ri, rq), (si, sq) = requested, stock
    i = si if ri == SPARSE_BATCH_STOCK else ri
    q = sq if rq == SPARSE_BATCH_STOCK else rq
    return f"{i}x{q}"
# knn_merge_parts (cuvs detail/knn_merge_parts.cuh) THROWs "Unimplemented for k" above it
SPARSE_CUVS_MAX_K = 1024
