"""CuvsBackend plumbing with a fake ``cuvs`` package (no GPU, no cuVS here).

The fake ``cuvs.common.Resources`` reproduces cuvs 26.10's real
``Resources(stream=h)`` defect: its Cython ``<cudaStream_t>stream`` cast binds
``id(h)`` (the int's PyObject address), not ``h``; on the node that made
``search`` segfault. The backend must instead bind through the C API
(``cuvs_engine.cuvs_c_api``, faked here as ``FakeCApi``: ``cuvsStreamSet`` /
``cuvsStreamGet`` on ``Resources.get_c_obj()``), and build/search assert the
stream the Resources carries IS the backend's explicit stream's handle;
``cuvs.neighbors.brute_force.build/search`` compute with faiss.knn on the CPU
and assert what the real C layer needs or what the backend promises: torch
tensors, matching dtypes, C-contiguous inputs, ``metric="inner_product"``
(or ``"cosine"`` when the test says so: then, as the cuVS source does, the
distances are 1 - cos, ascending, from the raw vectors' own norms),
``resources`` passed (never None, which would make cuVS create and sync its
own), pre-allocated int64 neighbors / float32 distances of shape (nq, k_eff)
written in place, k_eff <= rows. A fake ``rmm`` records the pool that
make_backend sets by default (``fake_cuvs.rmm_calls``); with
``--cuvs-rmm-pool off`` a tripwire shows rmm is never touched. torch's CPU
device stands in for CUDA, as in test_gpu_backend.py; tests that need CUDA
behaviour (events, the explicit stream, the allocator counters) flip
``_cuda`` and fake the torch.cuda calls, as in test_timing_detail.py.

What these mocks CANNOT prove (the on-node smoke script must):
- pylibraft's ``wrap_array`` accepting torch's CUDA array interface (torch
  omits ``strides`` for contiguous tensors), contiguity through ``dlpack_c``,
  and the C layer's device-memory-only checks;
- that the torch stream is really shared (events bracketing cuVS's kernels,
  ``search_s`` non-zero), and that ``resources=`` adds no sync;
- that cuVS writes into the supplied outputs (``auto_convert_output``
  identity);
- raft ``select_k`` with k > rows (the backend avoids it with k_eff), and
  score order / sign on real hardware, TF32 or accumulation-order
  differences (``--verify-atol``);
- the cost of RMM's default allocator (cudaMalloc / cudaFree per call)
  inside engine_call, and gpu_alloc_peak_bytes' real values.
"""

import csv
import sys
import time
import types

import faiss
import numpy as np
import pytest

from bench.systems import harness_cli as run
from bench.systems import loop
from bench.record import results_io, schema
from bench.verify import oracle as verify
from bench.systems import streaming as base, cpu_reference, cuvs_engine, faiss_engine
from conftest import ROWS
from bench.data.sources import InMemorySource, ParquetSource

torch = pytest.importorskip("torch")
K = 10
FAKE_VERSION = "26.10.00-fake"


class FakeIndex:
    def __init__(self, dataset, metric="inner_product"):
        self.dataset, self.metric = dataset, metric


CUVS_SUCCESS, CUVS_ERROR = 1, 0


class FakeResources:
    """cuvs.common.Resources: a C handle (``get_c_obj``) whose stream lives on
    the C side (``FakeCApi.streams``); default stream 0x2 (per-thread, what
    cuvsResourcesCreate gives on the node)."""

    _next = 0x1000

    def __init__(self, capi, stream=None):
        FakeResources._next += 0x10
        self.c_obj, self.capi = FakeResources._next, capi
        capi.streams[self.c_obj] = 0x2
        if stream:  # the real defect: <cudaStream_t>stream on a Python int
            capi.streams[self.c_obj] = id(stream)

    def get_c_obj(self):
        return self.c_obj

    def sync(self):
        """cuVS's ``Resources.sync()`` (calc_truth's per-batch sync): counted."""
        self.capi.syncs = getattr(self.capi, "syncs", 0) + 1

    @property
    def stream(self):
        return self.capi.streams[self.c_obj]


class FakeCApi:
    """libcuvs_c through ctypes: the two stream calls cuvs_engine.cuvs_bind_stream
    makes, with ctypes arguments as the real ones get them."""

    def __init__(self):
        self.streams = {}
        self.set_rc = CUVS_SUCCESS
        self.drop_set = False  # cuvsStreamSet "succeeds" but binds nothing

    def cuvsStreamSet(self, res, stream):
        if self.set_rc != CUVS_SUCCESS:
            return self.set_rc
        if not self.drop_set:
            self.streams[res.value] = stream.value
        return CUVS_SUCCESS

    def cuvsStreamGet(self, res, out):
        out._obj.value = self.streams[res.value]
        return CUVS_SUCCESS


class FakeCuvs:
    """State of the fake cuvs package: every call is logged; ``on_stream``
    (when set) is asked whether the explicit stream is active;
    ``expect_stream`` (when set) is the handle build/search must find bound
    to their ``resources``."""

    def __init__(self):
        self.log, self.resources, self.ks = [], [], []
        self.prefilters = []
        self.delay = 0.0
        self.on_stream = None
        self.expect_stream = None
        self.clock = None  # a test_timing_detail.Clock: search advances it
        self.capi = FakeCApi()
        self.metric = "inner_product"  # the metric build must be called with

    def _check_stream(self, what, resources):
        if self.on_stream is not None:
            assert self.on_stream(), f"{what} ran off the explicit stream"
        if self.expect_stream is not None:
            assert resources.stream == self.expect_stream, \
                f"{what}: cuVS is on stream {resources.stream:#x}, not the explicit one"

    def Resources(self, stream=None):
        res = FakeResources(self.capi, stream)
        self.resources.append(res)
        return res

    def build(self, dataset, metric="sqeuclidean", resources=None):
        self._check_stream("build", resources)
        assert isinstance(dataset, torch.Tensor) and dataset.is_contiguous()
        assert metric == self.metric
        assert resources is not None and resources is self.resources[-1]
        time.sleep(1e-4)  # so build_s is visibly > 0
        self.log.append(("build", dataset.dtype, dataset.shape[0]))
        return FakeIndex(dataset, metric)

    def search(self, index, queries, k, neighbors=None, distances=None,
               resources=None, prefilter=None):
        self._check_stream("search", resources)
        xb = index.dataset
        assert isinstance(queries, torch.Tensor) and queries.is_contiguous()
        assert queries.dtype == xb.dtype, "the backend must refuse a dtype pun"
        assert resources is not None and resources is self.resources[-1]
        assert 1 <= k <= xb.shape[0]
        nq = queries.shape[0]
        assert neighbors.dtype == torch.int64 and tuple(neighbors.shape) == (nq, k)
        assert distances.dtype == torch.float32 and tuple(distances.shape) == (nq, k)
        self.ks.append(k)
        self.log.append(("search", queries.dtype, xb.dtype))
        if self.clock is not None:
            self.clock.op("knn", 4)
        if prefilter is not None:
            self.prefilters.append(prefilter.kind)
            time.sleep(self.delay)  # a slow engine: the readers get ahead
            Dn, In = filtered_knn(queries.numpy(), xb.numpy(), k, index.metric, prefilter)
        elif index.metric == "cosine":
            Dn, In = cosine_distance_knn(queries.numpy(), xb.numpy(), k)
        else:
            Dn, In = faiss.knn(queries.numpy().astype(np.float32),
                               xb.numpy().astype(np.float32), k,
                               metric=faiss.METRIC_INNER_PRODUCT)
        distances.copy_(torch.from_numpy(Dn))  # written in place
        neighbors.copy_(torch.from_numpy(In))
        return distances, neighbors


class FakePrefilter:
    def __init__(self, kind, bits):
        # what pylibraft's wrap_array + _check_input_array demand: uint32
        assert isinstance(bits, torch.Tensor) and bits.dtype == torch.uint32
        assert bits.is_contiguous()
        self.kind, self.bits = kind, bits


def _fake_filters():
    m = types.ModuleType("cuvs.neighbors.filters")
    m.from_bitmap = lambda b: FakePrefilter("bitmap", b)
    m.from_bitset = lambda b: FakePrefilter("bitset", b)
    return m


def filtered_knn(q, x, k, metric, pf):
    """cuVS's filtered brute force as its docs state it: bit (q * rows + r) of
    the bitmap (LSB first in uint32 words), or bit r of the bitset, allows
    the pair. Slots past a query's allowed rows: the worst float32 score and
    a VALID-looking id 0 (worst case for the backend's fill detection)."""
    nq, n = q.shape[0], x.shape[0]
    bits = np.unpackbits(pf.bits.view(torch.uint8).numpy(), bitorder="little")
    if pf.kind == "bitmap":
        allowed = bits[:nq * n].reshape(nq, n).astype(bool)
    else:
        allowed = np.broadcast_to(bits[:n].astype(bool), (nq, n))
    q, x = q.astype(np.float32), x.astype(np.float32)
    if metric == "cosine":
        qn = np.linalg.norm(q, axis=1).astype(np.float32)
        xn = np.linalg.norm(x, axis=1).astype(np.float32)
        with np.errstate(all="ignore"):
            s = np.float32(1) - (q @ x.T) / (qn[:, None] * xn[None, :])
        s = np.where(allowed, s, np.inf)
        I = np.argsort(s, axis=1, kind="stable")[:, :k]
    else:
        s = np.where(allowed, q @ x.T, -np.inf)
        I = np.argsort(-s, axis=1, kind="stable")[:, :k]
    D = np.take_along_axis(s, I, axis=1)
    empty = ~np.take_along_axis(allowed, I, axis=1)
    worst = np.finfo(np.float32).max if metric == "cosine" else np.finfo(np.float32).min
    D = np.where(empty, worst, D).astype(np.float32)
    return D, np.where(empty, 0, I).astype(np.int64)


def cosine_distance_knn(q, x, k):
    """What cuVS's brute force returns for metric="cosine" (tiled_brute_force_knn
    in cpp/src/neighbors/detail/knn_brute_force.cuh): 1 - q.x / (|q| |x|), the
    k smallest, ascending (is_min_close), fp32 throughout."""
    q, x = q.astype(np.float32), x.astype(np.float32)
    qn = np.linalg.norm(q, axis=1).astype(np.float32)
    xn = np.linalg.norm(x, axis=1).astype(np.float32)
    dist = np.float32(1) - (q @ x.T) / (qn[:, None] * xn[None, :])
    I = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(dist, I, axis=1).astype(np.float32), I.astype(np.int64)


class RmmTripwire(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"rmm.{name} touched: --cuvs-rmm-pool off must leave RMM alone")


class FakeMR(types.ModuleType):
    """rmm.mr stand-in: records the pool cuvs_set_rmm_pool builds and installs."""

    def __init__(self, calls):
        super().__init__("rmm.mr")
        self.calls = calls

    def CudaMemoryResource(self):
        return "cuda"

    def PoolMemoryResource(self, upstream, initial_pool_size=None):
        self.calls.append(("pool", upstream, initial_pool_size))
        return "pool"

    def set_current_device_resource(self, mr):
        self.calls.append(("set", mr))


def install_fake_rmm(monkeypatch, calls):
    rmm = types.ModuleType("rmm")
    rmm.mr = FakeMR(calls)
    monkeypatch.setitem(sys.modules, "rmm", rmm)
    monkeypatch.setitem(sys.modules, "rmm.mr", rmm.mr)


@pytest.fixture
def fake_cuvs(monkeypatch):
    st = FakeCuvs()
    cuvs = types.ModuleType("cuvs")
    cuvs.__version__ = FAKE_VERSION
    common = types.ModuleType("cuvs.common")
    common.Resources = st.Resources
    neighbors = types.ModuleType("cuvs.neighbors")
    bf = types.ModuleType("cuvs.neighbors.brute_force")
    bf.build, bf.search = st.build, st.search
    flt = _fake_filters()
    cuvs.common, cuvs.neighbors, neighbors.brute_force = common, neighbors, bf
    neighbors.filters = flt
    for name, mod in (("cuvs", cuvs), ("cuvs.common", common),
                      ("cuvs.neighbors", neighbors), ("cuvs.neighbors.brute_force", bf),
                      ("cuvs.neighbors.filters", flt)):
        monkeypatch.setitem(sys.modules, name, mod)
    st.rmm_calls = []
    install_fake_rmm(monkeypatch, st.rmm_calls)
    monkeypatch.setattr(cuvs_engine, "cuvs_c_api", lambda: st.capi)
    import fake_cupy
    st.cupy = fake_cupy.install(monkeypatch)   # cuVS's own recipe merges with CuPy
    return st


def _unit(q, d):
    q = np.ascontiguousarray(q[:, :d])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ------------------------------------------------------------ construction

def test_missing_cuvs_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cuvs", None)  # import raises ImportError
    with pytest.raises(RuntimeError, match="needs the cuvs package"):
        cuvs_engine.CuvsBackend(K, device="cpu")


def test_construction(fake_cuvs):
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    assert (b.name, b.engine, b.engine_version, b.queries_mode) == \
        ("cuvs", "cuvs", FAKE_VERSION, "resident")
    assert fake_cuvs.resources == [b.res]  # one Resources for the lifetime
    assert b.stream is None  # torch's CPU device: no stream to share


def test_filter_uploads_are_pinned_and_non_blocking_on_cuda(fake_cuvs, monkeypatch):
    """--filter-mode bitmap on CUDA: every mask / index upload is staged in
    pinned host memory and copied non-blocking (the host does not wait for
    it); on a CPU device a plain copy. Exercised with the CPU torch by
    recording the calls."""
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    torch = b.torch
    calls = []
    real_empty = torch.empty

    def empty(*a, pin_memory=False, **kw):
        calls.append(("empty", pin_memory))
        return real_empty(*a, **kw)

    real_to = torch.Tensor.to

    def to(self, *a, **kw):
        calls.append(("to", kw.get("non_blocking", False)))
        return real_to(self, *a, **{k: v for k, v in kw.items() if k != "non_blocking"})

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    a = np.arange(10, dtype=np.uint8)
    b._cuda = True
    out = b._upload_small(a)
    assert np.array_equal(out.numpy(), a)
    assert calls == [("empty", True), ("to", True)]   # pinned staging, non-blocking copy
    calls.clear()
    b._cuda = False
    out = b._upload_small(a[::2])                      # non-contiguous input is fine
    assert np.array_equal(out.numpy(), a[::2]) and calls == [("to", False)]


def test_masks_upload_straight_from_pinned_memory(fake_cuvs, monkeypatch):
    """--h2d pinned on CUDA: masks already in page-locked memory (the reader's
    registered ring slot) are copied non-blocking STRAIGHT from it, with no
    pinned staging copy; masks elsewhere (not pinned, or --h2d pageable) are
    staged as before. Exercised with the CPU torch by recording the calls."""
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    torch = b.torch
    calls = []
    real_empty, real_to = torch.empty, torch.Tensor.to

    def empty(*a, pin_memory=False, **kw):
        calls.append(("empty", pin_memory))
        return real_empty(*a, **kw)

    def to(self, *a, **kw):
        calls.append(("to", kw.get("non_blocking", False)))
        return real_to(self, *a, **{k: v for k, v in kw.items() if k != "non_blocking"})

    pinned = {"v": True}
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: pinned["v"])
    masks = np.arange(24, dtype=np.uint8).reshape(3, 8)
    b._cuda, b._nb = True, True
    out = b._upload_host(masks)
    assert np.array_equal(out.numpy(), masks)
    assert calls == [("to", True)]                     # no staging copy at all
    for v, nb in ((False, True), (True, False)):       # not pinned / --h2d pageable
        calls.clear()
        pinned["v"], b._nb = v, nb
        out = b._upload_host(masks)
        assert np.array_equal(out.numpy(), masks)
        assert calls == [("empty", True), ("to", True)]   # pinned staging, as before


def test_masks_ride_with_the_block_upload(fake_cuvs):
    """Each filtered block's masks are uploaded in _to_dev, with its vectors
    (so under --h2d-overlap block i+1's go up during block i's search), and
    the search of that block takes exactly those device masks; a block whose
    masks were never uploaded that way is uploaded on demand."""
    from bench.data.filters import reader as filtered
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    rows, nb = 13, 2

    def fblock(seed):
        rng = np.random.default_rng(seed)
        m = rng.integers(0, 256, size=(2, nb), dtype=np.uint8)
        return filtered.FilterBlock("per_query", rows, m, np.array([1, 1]),
                                    np.array([0, 1, 1, 0], dtype=np.int64))
    x0 = np.ones((rows, 4), np.float32)
    x1 = np.full((rows, 4), 2.0, np.float32)
    f0, f1 = fblock(0), fblock(1)
    b._pad_rows = {id(x0): (x0, nb * 8, f0), id(x1): (x1, nb * 8, f1)}
    t0 = b._to_dev(x0)
    b._to_dev(x1)                                      # "ahead": uploaded now
    assert t0.shape[0] == nb * 8                       # padded to rows8
    up = b._masks_dev[id(x1)]
    assert up[0] is x1 and np.array_equal(up[1].numpy(), f1.masks)
    got = b._block_masks(x1, f1)
    assert got is up[1] and id(x1) not in b._masks_dev  # consumed exactly once
    assert np.array_equal(b._block_masks(x1, f1).numpy(), f1.masks)  # on demand
    b.drain()
    assert b._masks_dev is None                        # x0's entry dropped with it


class FakeStream:
    def __init__(self, handle):
        self.cuda_stream = handle


def test_fake_resources_reproduces_the_stream_cast_defect(fake_cuvs):
    """Guard on the fake itself: Resources(stream=h) binds id(h), as the real
    cuvs 26.10 does, so a backend passing stream= fails the tests below."""
    h = 0xbeef
    assert fake_cuvs.Resources(stream=h).stream == id(h) != h


def test_explicit_stream_bound_through_the_c_api(fake_cuvs, monkeypatch):
    monkeypatch.setattr(cuvs_engine.CuvsBackend, "_make_stream", lambda self: FakeStream(0xbeef))
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    assert b.res.stream == 0xbeef


def test_bind_stream_failure_is_loud(fake_cuvs):
    res = fake_cuvs.Resources()
    fake_cuvs.capi.set_rc = CUVS_ERROR
    with pytest.raises(RuntimeError, match="cuvsStreamSet failed"):
        cuvs_engine.cuvs_bind_stream(res, 0xbeef)
    fake_cuvs.capi.set_rc, fake_cuvs.capi.drop_set = CUVS_SUCCESS, True
    with pytest.raises(RuntimeError, match="stream not bound"):
        cuvs_engine.cuvs_bind_stream(res, 0xbeef)


def test_default_stream_handle_refused(fake_cuvs, monkeypatch):
    """Handle 0 is the legacy default stream (and cuVS's own ``if stream:``
    would ignore it): never what the backend means to share."""
    monkeypatch.setattr(cuvs_engine.CuvsBackend, "_make_stream", lambda self: FakeStream(0))
    with pytest.raises(RuntimeError, match="non-default stream"):
        cuvs_engine.CuvsBackend(K, device="cpu")


def test_every_op_runs_on_the_explicit_stream(fake_cuvs, monkeypatch, corpus, queries):
    """setup()'s upload and each block's H2D / cast / build / search / events
    happen inside torch.cuda.stream(backend.stream), one context per call."""
    from test_timing_detail import Clock, fake_alloc_counters, fake_event_cls
    stream = FakeStream(0xbeef)
    active, entered = [], []

    class StreamCtx:
        def __init__(self, s):
            assert s is stream
            entered.append(s)

        def __enter__(self):
            active.append(1)

        def __exit__(self, *exc):
            active.pop()

    monkeypatch.setattr(cuvs_engine.CuvsBackend, "_make_stream", lambda self: stream)
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    monkeypatch.setattr(torch.cuda, "stream", StreamCtx)
    fake_cuvs.on_stream = lambda: bool(active)
    fake_cuvs.expect_stream = 0xbeef  # build/search: cuVS really on that stream
    real_to_dev = b._to_dev

    def to_dev(x):
        assert active, "H2D off the explicit stream"
        return real_to_dev(x)

    monkeypatch.setattr(b, "_to_dev", to_dev)
    b.setup(queries)
    clock = Clock()
    b._cuda = True  # events (faked) from here on
    fake_alloc_counters(monkeypatch, torch, clock)
    ev_cls = fake_event_cls(clock)

    class CheckedEvent(ev_cls):
        def record(self):
            assert active, "event recorded off the explicit stream"
            super().record()

    monkeypatch.setattr(torch.cuda, "Event", CheckedEvent)
    src = InMemorySource.load(corpus[0])
    D, I, _ = loop.run_pass(src, b, len(queries), K)
    assert len(entered) == 1 + len(src.blocks) and not active
    assert [op for op, *_ in fake_cuvs.log] == ["build", "search"] * len(src.blocks)
    D_ref, I_ref = faiss.knn(queries, corpus[1], K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []


@pytest.mark.parametrize("engine", ["faiss", "cuvs"])
def test_both_engines_resident_use_a_non_default_stream(fake_cuvs, monkeypatch, engine):
    """On CUDA both resident engines get their stream from the shared
    ResidentBackend._make_stream: an explicit torch.cuda.Stream (non-default,
    on the backend's device), not torch's default stream."""
    made = []

    class Stream(FakeStream):
        def __init__(self, device=None):
            super().__init__(0xbeef)
            made.append(device)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    if engine == "faiss":
        b = faiss_engine.GpuBackend(K, device="cuda")
    else:
        b = cuvs_engine.CuvsBackend(K, device="cuda")
        assert b.res.stream == 0xbeef
    assert type(b)._make_stream is base.ResidentBackend._make_stream
    assert isinstance(b.stream, Stream) and b.stream.cuda_stream and made == ["cuda"]


# -------------------------------------------------------------- search path

@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_resident_path_matches_reference(fake_cuvs, corpus, queries, regime):
    path, xb = corpus
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    assert b.setup(queries) > 0
    src = InMemorySource.load(path) if regime == "memory" else \
        ParquetSource(path, prefetch=2, workers=2)
    D, I, info = loop.run_pass(src, b, len(queries), K)
    D_ref, I_ref = faiss.knn(queries, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []
    assert I.min() >= 0 and I.max() < len(xb)
    # one build per block, k_eff = min(k, rows) (ROWS has 3- and 1-row files)
    assert [n for op, _, n in fake_cuvs.log if op == "build"] == ROWS
    assert fake_cuvs.ks == [min(K, n) for n in ROWS]
    assert info["build_s"] > len(ROWS) * 1e-4
    assert info["build_s"] < info["engine_call_s"]
    _, files = info["trace"]
    assert sum(f[9] for f in files) == pytest.approx(info["build_s"])


def test_short_block_padded_like_faiss(fake_cuvs, queries):
    """k > rows: cuVS is asked for k_eff = rows, the host pads to k with
    FAISS's sentinels (lowest float32, -1), bit for bit what faiss.knn does."""
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    b.setup(queries)
    xb = np.ascontiguousarray(queries[:3])
    D, I = b.search(xb)
    assert fake_cuvs.ks == [3] and D.shape == I.shape == (len(queries), K)
    D_f, I_f = faiss.knn(queries, xb, K, metric=faiss.METRIC_INNER_PRODUCT)
    np.testing.assert_array_equal(D[:, 3:], D_f[:, 3:])
    np.testing.assert_array_equal(I[:, 3:], I_f[:, 3:])
    assert (I[:, 3:] == -1).all() and (D[:, 3:] == np.finfo(np.float32).min).all()
    assert D.dtype == np.float32 and I.dtype == np.int64


def test_dtype_pun_and_layout_refused(fake_cuvs, queries):
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    b.setup(queries)
    xq = b.xq
    with pytest.raises(TypeError, match="reinterpret"):
        b._engine_search(xq, torch.zeros(20, xq.shape[1], dtype=torch.float16))
    with pytest.raises(ValueError, match="contiguous"):
        b._engine_search(xq, torch.zeros(xq.shape[1], 20).t())
    assert fake_cuvs.log == []  # refused before cuVS is called


@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_device_widen_uploads_stored_dtype_then_casts(fake_cuvs, monkeypatch, tmp_path,
                                                      queries, regime):
    """--widen device: fp16-stored blocks cross H2D as fp16, are cast on the
    device, and cuVS sees fp32 index and fp32 queries; results equal the
    fp32-stored copy (widening is exact)."""
    from test_run import _fp16_pair
    f16, f32, xb = _fp16_pair(tmp_path)
    q = _unit(queries, 24)
    out = {}
    for name, path in (("f16", f16), ("f32", f32)):
        fake_cuvs.log.clear()
        b = cuvs_engine.CuvsBackend(K, device="cpu")
        b.setup(q)
        h2d = []
        real = b._to_dev
        monkeypatch.setattr(b, "_to_dev", lambda x: h2d.append(x.dtype) or real(x))
        src = InMemorySource.load(path) if regime == "memory" else \
            ParquetSource(path, prefetch=2, workers=2)
        out[name] = loop.run_pass(src, b, len(q), K)
        stored = np.float16 if name == "f16" else np.float32
        assert h2d == [np.dtype(stored)] * 3
        assert [e[1:] for e in fake_cuvs.log if e[0] == "search"] == \
            [(torch.float32, torch.float32)] * 3
        assert (out[name][2]["widen_s"] > 0) == (name == "f16")
    np.testing.assert_array_equal(out["f16"][0], out["f32"][0])
    np.testing.assert_array_equal(out["f16"][1], out["f32"][1])
    D_ref, I_ref = faiss.knn(q, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(out["f16"][0], out["f16"][1], D_ref, I_ref) == []


# ---------------------------------------------- shared instrumentation

def _clocked(b, clock, monkeypatch):
    """The ops on the fake clock: H2D +1 ms, cast +2 ms (engine +4 ms is the
    engine's own mock)."""
    real_to_dev, real_cast = b._to_dev, b._cast

    def to_dev(x):
        clock.op("h2d", 1)
        return real_to_dev(x)

    def cast(x):
        clock.op("cast", 2)
        return real_cast(x)

    monkeypatch.setattr(b, "_to_dev", to_dev)
    monkeypatch.setattr(b, "_cast", cast)


def test_both_engines_run_the_same_event_code(fake_cuvs, monkeypatch, tmp_path, queries):
    """One code path: the search body is ResidentBackend's for both engines,
    and with the same fake device clock both produce the same op / event /
    guard sequence and the same h2d / widen / search / d2h split."""
    from test_run import _fp16_pair
    from test_timing_detail import Clock, fake_alloc_counters, fake_event_cls
    assert cuvs_engine.CuvsBackend.search is base.ResidentBackend.search
    assert "search" not in vars(cuvs_engine.CuvsBackend)
    f16, _, _ = _fp16_pair(tmp_path)
    q = _unit(queries, 24)
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    runs = {}
    for engine in ("faiss", "cuvs"):
        clock = Clock()
        if engine == "faiss":
            b = faiss_engine.GpuBackend(K, device="cpu")

            def knn_gpu(res, xq, xb_t, k, metric=None, clock=clock, **kw):
                clock.op("knn", 4)
                Dn, In = faiss.knn(xq.numpy(), xb_t.numpy(), k,
                                   metric=faiss.METRIC_INNER_PRODUCT)
                return torch.from_numpy(Dn), torch.from_numpy(In)

            monkeypatch.setattr(faiss, "knn_gpu", knn_gpu)  # after construction
        else:
            b = cuvs_engine.CuvsBackend(K, device="cpu")
            fake_cuvs.clock = clock
        _clocked(b, clock, monkeypatch)
        b.setup(q)
        b._cuda = True
        monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
        monkeypatch.setattr(torch.cuda, "synchronize",
                            lambda *a: pytest.fail("no device-wide sync in search"))
        fake_alloc_counters(monkeypatch, torch, clock)
        clock.log.clear()
        D, I, info = loop.run_pass(InMemorySource.load(f16), b, len(q), K)
        runs[engine] = (clock.log, info, D, I)
    (log_f, info_f, D_f, I_f), (log_c, info_c, D_c, I_c) = runs["faiss"], runs["cuvs"]
    assert log_f == log_c == ["ev", "h2d", "ev", "cast", "ev", "ev", "knn", "ev", "ev", "ev", "ev",
                              "query"] * 3
    for c in ("h2d_s", "widen_s", "search_s", "d2h_s", "sync_wait_s", "sync_waits"):
        assert info_f[c] == pytest.approx(info_c[c]), c
    assert info_c["search_s"] == pytest.approx(3 * 4e-3)
    assert info_f["build_s"] == 0 and info_c["build_s"] > 0
    assert info_f["gpu_alloc_peak_bytes"] == info_c["gpu_alloc_peak_bytes"] == 12345
    np.testing.assert_array_equal(D_f, D_c)
    np.testing.assert_array_equal(I_f, I_c)


def test_no_gpu_events_creates_no_events_for_cuvs(fake_cuvs, monkeypatch, corpus, queries):
    """--no-gpu-events means the same for cuVS: no Event object, one
    device-wide synchronize per block, event columns empty; build_s is host
    wall, not an event column, so it is still measured."""
    from test_timing_detail import Clock, fake_alloc_counters
    b = cuvs_engine.CuvsBackend(K, device="cpu", events=False)
    b.setup(queries)
    b._cuda = True
    clock = Clock()
    fake_alloc_counters(monkeypatch, torch, clock)

    def no_event(*a, **kw):
        raise AssertionError("a CUDA event was created")

    syncs = []
    monkeypatch.setattr(torch.cuda, "Event", no_event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: syncs.append(1))
    src = InMemorySource.load(corpus[0])
    D, I, info = loop.run_pass(src, b, len(queries), K)
    assert len(syncs) == len(src.blocks)
    for c in schema.EVENT_FIELDS:
        assert info[c] == "", c
    assert info["build_s"] > 0 and info["engine_call_s"] > info["build_s"]
    assert info["gpu_alloc_peak_bytes"] == 12345 and len(clock.resets) == 1
    assert b.last_wait == (None, None)
    D_ref, I_ref = faiss.knn(queries, corpus[1], K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []


def test_alloc_peak_counter_populated_or_blank(fake_cuvs, monkeypatch, corpus, queries):
    """gpu_alloc_peak_bytes: torch's peak, reset once per pass outside the
    timed span, on CUDA; "" (not measured) on torch's CPU device; "n/a"
    where torch is not involved (CPU backend)."""
    from test_timing_detail import Clock, fake_alloc_counters, fake_event_cls
    src = InMemorySource.load(corpus[0])
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    b.setup(queries)
    assert loop.run_pass(src, b, len(queries), K)[2]["gpu_alloc_peak_bytes"] == ""
    clock = Clock()
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    fake_alloc_counters(monkeypatch, torch, clock, peak=777)
    for _ in range(2):
        assert loop.run_pass(src, b, len(queries), K)[2]["gpu_alloc_peak_bytes"] == 777
    assert clock.resets == [("cpu",), ("cpu",)]  # once per pass
    c = cpu_reference.CpuBackend(K)
    c.setup(queries)
    assert loop.run_pass(src, c, len(queries), K)[2]["gpu_alloc_peak_bytes"] == "n/a"



# ------------------------------------------------------------------- CLI

BASE = ["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "cuvs"]


@pytest.mark.parametrize("extra", [
    ["--temp-mem", "0"],
    ["--temp-mem", "1024"],
])
def test_cuvs_arg_guards(extra):
    with pytest.raises(SystemExit):
        run.parse_args(BASE + extra)


def test_cuvs_arg_defaults():
    a = run.parse_args(BASE)
    assert a.widen == "device"
    assert run.parse_args(BASE + ["--k", "5000"]).k == 5000  # no FAISS k cap


def _cuvs_factory(monkeypatch):
    """run.make_backend, with --backend cuvs on torch's CPU device."""
    real = run.make_backend

    def make(args, dtype):
        if args.backend != "cuvs":
            return real(args, dtype)
        return cuvs_engine.CuvsBackend(args.k, dtype, device="cpu",
                               events=not args.no_gpu_events, metric=args.metric)
    monkeypatch.setattr(run, "make_backend", make)


@pytest.mark.slow
def test_main_cuvs_end_to_end(fake_cuvs, monkeypatch, tmp_path, capsys):
    """run.main --backend cuvs on torch's CPU device: CSV and trace record
    the engine, the stock-allocator config and build_s; --verify passes
    against the CPU faiss.knn oracle; events-off rows keep build_s; the plot
    keeps the engines in separate bars."""
    from test_run import _fp16_pair
    f16, _, _ = _fp16_pair(tmp_path, payload_bytes=300)
    _cuvs_factory(monkeypatch)
    out = tmp_path / "r.csv"
    common = ["--corpus", f16, "--nq", "16", "--warmup", "1", "--repeats", "2",
              "--csv", str(out), "--verify", "--backend", "cuvs"]
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--prefetch", pf] + common)
    run.main(["--regime", "stream", "--prefetch", "0", "--no-gpu-events"] + common)
    assert capsys.readouterr().out.count("verify OK") == 3
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 9 and list(rows[0]) == schema.CSV_FIELDS
    for r in rows:
        assert (r["backend"], r["engine"], r["engine_version"]) == \
            ("cuvs", "cuvs", FAKE_VERSION)
        assert (r["queries_mode"], r["temp_mem"], r["rmm_pool"]) == \
            ("resident", "n/a", "pool:1GiB")
        assert r["widen_where"] == "device" and r["stored_dtype"] == "f16"
        assert float(r["build_s"]) > 0
        assert r["gpu_alloc_peak_bytes"] == ""  # torch's CPU device: not measured
    on, off = rows[:6], rows[6:]
    assert {r["gpu_events"] for r in on} == {"on"} and {r["gpu_events"] for r in off} == {"off"}
    assert all(float(r["search_s"]) > 0 for r in on)
    assert all(r[c] == "" for r in off for c in schema.EVENT_FIELDS)
    trows = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert {r["engine"] for r in trows} == {"cuvs"}
    for p in rows:
        mine = [t for t in trows if t["t_start_unix"] == p["t_start_unix"]]
        assert sum(float(t["build_s"]) for t in mine) == pytest.approx(float(p["build_s"]))
    # a FAISS (cpu) run of the same config next to it: never one bar
    run.main(["--regime", "stream", "--prefetch", "0", "--corpus", f16, "--nq", "16",
              "--warmup", "0", "--repeats", "1", "--csv", str(out), "--backend", "cpu"])
    from bench.report import plots as plot
    med = plot.median_rows(plot.read_csvs([str(out)]))
    assert sorted(set(med["engine"])) == ["cuvs", "faiss"]
    assert len(med) == 4  # cuvs prefetch 0 / 2 / no-events + faiss
    png = tmp_path / "a.png"
    assert plot.attribution([str(out)], str(png)) == 0
    assert png.stat().st_size > 10_000 and "cuvs" in capsys.readouterr().out


# -------------------------------------------- old CSVs

# the real pre-rename header (FAISS-only harness), pinned literally rather than
# derived from CSV_FIELDS, so a later change to CSV_FIELDS cannot move it
OLD_FIELDS = [
    "regime", "backend", "queries_mode", "nq", "k", "d", "n_files", "max_files",
    "rows", "rows_per_file", "prefetch", "workers", "vectors_mem_limit", "temp_mem",
    "omp_threads", "pa_cpu_threads", "pa_io_threads", "read_mode",
    "read_mode_effective", "column_share", "range_concurrency", "range_scope",
    "stored_dtype", "dtype", "widen_where", "gpu_events", "warmup", "repeat",
    "total_s",
    "source_stall_s", "faiss_call_s", "merge_s", "other_s", "h2d_s",
    "widen_s", "search_s", "d2h_s", "faiss_gap_s", "sync_wait_s", "sync_waits",
    "fetch_s", "decode_s",
    "read_table_s", "convert_s", "bytes", "net_rx_bytes", "net_rx_MBps",
    "row_groups", "row_groups_per_file", "setup_s", "preload_s", "cpu_user_s",
    "cpu_sys_s", "t_start_unix", "t_end_unix", "corpus_fs", "corpus", "host",
]


def test_old_csv_refused_by_run_and_plot(corpus, tmp_path):
    old = tmp_path / "old.csv"
    with open(old, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OLD_FIELDS)
        w.writeheader()
        w.writerow({c: "0" for c in OLD_FIELDS})
    with pytest.raises(SystemExit, match="existing header differs"):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--warmup", "0", "--repeats", "1", "--csv", str(old)])
    new = tmp_path / "new.csv"
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "0", "--repeats", "1", "--csv", str(new)])
    from bench.report import plots
    with pytest.raises(SystemExit) as e:
        plots.attribution([str(new), str(old)], str(tmp_path / "a.png"))
    assert "old.csv: missing columns" in str(e.value.code) and "engine_call_s" in str(e.value.code)


def test_d2h_cuvs_distance_merge_identical(fake_cuvs, corpus, queries):
    """cuVS cosine (merged as distances): pinned D2H changes nothing either."""
    path, _ = corpus
    fake_cuvs.metric = "cosine"
    out = {}
    for name in ("pageable", "pinned"):
        b = cuvs_engine.CuvsBackend(K, device="cpu", metric="cosine")
        if name == "pinned":
            b.configure_d2h("pinned")
        b.setup(queries)
        with ParquetSource(path, prefetch=2, workers=2).start() as src:
            out[name] = loop.run_pass(src, b, len(queries), K)
    np.testing.assert_array_equal(out["pinned"][0], out["pageable"][0])
    np.testing.assert_array_equal(out["pinned"][1], out["pageable"][1])


# ------------------------------------------------- --filter-mode bitmap (prefilter)

@pytest.mark.slow
@pytest.mark.parametrize("name,extra", [
    ("static", []), ("per_query", []), ("per_query", ["--filter-skip", "none"]),
    ("per_query", ["--filter-compact", "union"]), ("per_query", ["--metric", "cosine"]),
    ("static", ["--metric", "cosine"]), ("per_query", ["--k", "200"]),
    # overlap modes (prepare / submit / finish pipelined in run_pass)
    ("per_query", ["--h2d", "pinned", "--h2d-overlap"]),
    ("per_query", ["--d2h", "pinned", "--d2h-overlap"]),
    ("per_query", ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap",
                   "--metric", "cosine"]),
    ("static", ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap"]),
    ("static", ["--h2d", "pinned", "--h2d-overlap"]),
    ("per_query", ["--filter-compact", "union", "--d2h", "pinned", "--d2h-overlap"]),
    # union blocks are compacted by the reader IN the (pinned) ring slot
    ("per_query", ["--filter-compact", "union", "--h2d", "pinned"]),
    ("per_query", ["--filter-compact", "union", "--h2d", "pinned", "--h2d-overlap"]),
    ("per_query", ["--filter-compact", "union", "--filter-skip", "none", "--h2d", "pinned",
                   "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap", "--metric", "cosine"])])
def test_cuvs_prefilter_matches_masked(fake_cuvs, monkeypatch, tmp_path, name, extra):
    """--backend cuvs --filter-mode bitmap (bitset for static, contiguous
    bitmap over rows padded to a multiple of 8 for per-query) returns what the
    masked CPU reference returns, short lists included; the files' row counts
    (test_filtered_run.ROWS) are not multiples of 8."""
    import test_filtered_run as T
    from test_duckdb_filter import make_corpus, make_queries
    import pyarrow as pa
    import pyarrow.parquet as pq
    _cuvs_factory(monkeypatch)
    from test_h2d import _pinner
    monkeypatch.setattr(base, "host_pinner", lambda backend: _pinner()[0])
    if "cosine" in extra:
        fake_cuvs.metric = "cosine"
    fake_cuvs.delay = 0.3 if "--h2d-overlap" in extra else 0.0
    if "--d2h-overlap" in extra:
        # a completion that takes time (on a GPU: the wait for the copy back),
        # so a skipped file that double-charged block i-1's finish() would
        # show as a clearly negative other_s
        real_complete = cuvs_engine.CuvsBackend.complete

        def slow_complete(self, tok):
            time.sleep(0.1)
            return real_complete(self, tok)
        monkeypatch.setattr(cuvs_engine.CuvsBackend, "complete", slow_complete)
    root = tmp_path / "d"
    (root / "corpus").mkdir(parents=True)
    for i, n in enumerate(T.ROWS):
        p = make_corpus(n, seed=200 + i)
        x = np.random.default_rng(i).standard_normal((n, T.D)).astype(np.float32)
        pq.write_table(p.append_column("embedding", T._emb(x)),
                       str(root / "corpus" / f"part-{i:05d}.parquet"))
    q = make_queries(T.NQ, seed=11)
    xq = np.random.default_rng(99).standard_normal((T.NQ, T.D)).astype(np.float32)
    pq.write_table(q.append_column("embedding", T._emb(xq)), str(root / "q.parquet"))
    fpath = T.write_filter(root, name)
    out = {}
    for backend in ("masked", "cuvs"):
        res = tmp_path / f"{backend}.npz"
        k = extra[extra.index("--k") + 1] if "--k" in extra else str(T.K)
        ex = [e for e in extra if e not in ("--k", k)]
        if backend == "masked":   # a host engine: no transfer modes
            ex = [e for e in ex if e not in ("--h2d", "pinned", "--h2d-overlap", "--d2h",
                                             "--d2h-overlap")]
        run.main(["--regime", "stream", "--corpus", str(root / "corpus"), "--column",
                  "embedding", "--backend", backend, "--query-file", str(root / "q.parquet"),
                  "--query-column", "embedding", "--nq", str(T.NQ), "--k", k,
                  "--filter-config", fpath, "--filter-search", "f", "--filter-mode", "bitmap", "--csv", str(tmp_path / "o.csv"),
                  "--save-results", str(res), "--verify", "--warmup", "0", "--repeats", "1",
                  "--prefetch", "3", "--workers", "2", *ex])
        out[backend] = np.load(res)
    assert set(fake_cuvs.prefilters) == {"bitset" if name == "static" else "bitmap"}
    rows = list(csv.DictReader(open(tmp_path / "o.csv")))
    assert rows[-1]["backend"] == "cuvs" and rows[-1]["verified"] == "ok"
    # k = 10 or 200 and files <= 512 rows: EVERY prefilter call has k_eff <= 256,
    # so the diagnostic counts each of them (one timed pass, no warmup)
    assert int(rows[-1]["cuvs_filter_small_k_calls"]) == len(fake_cuvs.prefilters) > 0
    # the per-query expansion: a sub-span of search_s (blank without events)
    cu = rows[-1]
    # buckets partition the pass: a skipped file under --d2h-overlap once
    # double-charged block i-1's finish() (other_s ~ -0.3 s at this delay)
    assert float(cu["other_s"]) > -0.01
    assert float(cu["engine_host_s"]) <= float(cu["engine_call_s"]) + 1e-6
    if cu["search_s"] != "":
        assert 0 <= float(cu["filter_bits_s"]) <= float(cu["search_s"])
        # the library spans (build + search) and the harness's rest of search_s:
        # the bitmap expansion and the id step lie outside the library spans
        lib, harn = float(cu["lib_s"]), float(cu["search_harness_s"])
        assert 0 < lib <= float(cu["search_s"])
        assert harn >= float(cu["filter_bits_s"]) + float(cu["ids_s"]) - 1e-3
    else:
        assert cu["filter_bits_s"] == ""
    if "--h2d-overlap" in extra:
        assert rows[-1]["h2d_mode"] == "pinned-overlap"
        # some block i+1 really was prepared and uploaded during block i
        # (static: the 7- and 1-row files pass nothing and are skipped, so no
        # two searched files are adjacent there -- nothing to upload ahead)
        if name == "per_query":
            assert int(rows[-1]["h2d_overlapped"]) > 0, rows[-1]["h2d_overlapped"]
        else:
            assert int(rows[-1]["filter_skipped_files"]) == 2
    Im, Ic = out["masked"]["I"], out["cuvs"]["I"]
    assert np.array_equal(Im, Ic)
    ok = Im >= 0
    assert ok.any() and (~ok).any() == ("200" in extra or name == "per_query")
    assert np.allclose(out["masked"]["D"][ok], out["cuvs"]["D"][ok], atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("extra", [[], ["--h2d", "pinned"], ["--h2d", "pinned", "--h2d-overlap"],
                                   ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned",
                                    "--d2h-overlap"]])
def test_cuvs_gather_transfer_modes_match_masked(fake_cuvs, monkeypatch, tmp_path, extra):
    """--filter-mode gather on cuVS: the reader-compacted block (in its
    pinned ring slot, uploaded ahead, copied back overlapped) gives the masked
    gather's ids and scores; the compaction is reader time."""
    import test_filtered_run as T
    from test_duckdb_filter import make_corpus, make_queries
    import pyarrow.parquet as pq
    _cuvs_factory(monkeypatch)
    from test_h2d import _pinner
    monkeypatch.setattr(base, "host_pinner", lambda backend: _pinner()[0])
    if "--h2d-overlap" in extra:
        # a slow engine (gather calls cuVS unfiltered, where the fake has no
        # delay): the readers get ahead, so block i+1 is uploaded during i
        real = cuvs_engine.CuvsBackend._engine_search

        def slow(self, xq_t, xb_t):
            time.sleep(0.3)
            return real(self, xq_t, xb_t)
        monkeypatch.setattr(cuvs_engine.CuvsBackend, "_engine_search", slow)
    root = tmp_path / "d"
    (root / "corpus").mkdir(parents=True)
    rows = [301, 7, 512, 1, 200, 330]   # 512 / 200 / 330: adjacent searched files
    for i, n in enumerate(rows):
        p = make_corpus(n, seed=200 + i)
        x = np.random.default_rng(i).standard_normal((n, T.D)).astype(np.float32)
        pq.write_table(p.append_column("embedding", T._emb(x)),
                       str(root / "corpus" / f"part-{i:05d}.parquet"))
    q = make_queries(T.NQ, seed=11)
    xq = np.random.default_rng(99).standard_normal((T.NQ, T.D)).astype(np.float32)
    pq.write_table(q.append_column("embedding", T._emb(xq)), str(root / "q.parquet"))
    fpath = T.write_filter(root, "static")
    out = {}
    for backend in ("masked", "cuvs"):
        res = tmp_path / f"{backend}.npz"
        run.main(["--regime", "stream", "--corpus", str(root / "corpus"), "--column",
                  "embedding", "--backend", backend, "--query-file", str(root / "q.parquet"),
                  "--query-column", "embedding", "--nq", str(T.NQ), "--k", str(T.K),
                  "--filter-config", fpath, "--filter-search", "f", "--filter-mode", "gather", "--csv", str(tmp_path / "o.csv"),
                  "--save-results", str(res), "--verify", "--warmup", "0", "--repeats", "1",
                  "--prefetch", "3", "--workers", "2", *(extra if backend == "cuvs" else [])])
        out[backend] = np.load(res)
    r = list(csv.DictReader(open(tmp_path / "o.csv")))[-1]
    assert r["backend"] == "cuvs" and r["verified"] == "ok" and not fake_cuvs.prefilters
    assert float(r["filter_compact_s"]) > 0
    assert float(r["filter_s"]) >= float(r["filter_compact_s"]) + float(r["filter_eval_s"])
    if "--h2d-overlap" in extra:
        assert int(r["h2d_overlapped"]) > 0
    assert np.array_equal(out["masked"]["I"], out["cuvs"]["I"])
    ok = out["masked"]["I"] >= 0
    assert np.allclose(out["masked"]["D"][ok], out["cuvs"]["D"][ok], atol=1e-5)


def test_cuvs_small_k_counter_zero_at_large_k(fake_cuvs, monkeypatch):
    """k_eff > 256 (raft's radix select, exact on short rows) is not counted;
    k_eff <= 256 is, whatever k was asked for (k_eff = min(k, rows))."""
    from bench.data.filters.reader import FilterBlock
    from bench.data.filters import duckdb_filter as dfm
    rng = np.random.default_rng(0)
    for k, rows, want in ((300, 400, 0), (1000, 300, 0), (1000, 257, 0), (1000, 256, 1),
                          (1000, 200, 1), (10, 5000, 1)):
        b = cuvs_engine.CuvsBackend(k, device="cpu")
        b.setup(rng.standard_normal((4, 8)).astype(np.float32))
        mask = np.packbits(rng.random(rows) < 0.5, bitorder="little")[None]
        fb = FilterBlock("static", rows, mask, np.unpackbits(mask, axis=1).sum(1))
        b.search_filtered(rng.standard_normal((rows, 8)).astype(np.float32), fb)
        assert int(b.stats.as_dict().get("cuvs_filter_small_k_calls", 0)) == want, (k, rows)


def test_rmm_pool_flag(fake_cuvs, monkeypatch, tmp_path):
    """Dense cuVS gets an RMM pool by default (1 GiB initial, what cuVS's own
    ground-truth tool sets), set BEFORE the backend exists and recorded;
    --cuvs-rmm-pool SIZE changes the size; off leaves rmm untouched."""
    calls = fake_cuvs.rmm_calls
    assert cuvs_engine.cuvs_set_rmm_pool("1GiB") == 1 << 30
    assert calls == [("pool", "cuda", 1 << 30), ("set", "pool")]
    base = ["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "cuvs"]
    a = run.parse_args(base)
    assert a.cuvs_rmm_pool == cuvs_engine.CUVS_RMM_POOL_DEFAULT == "1GiB"
    assert cuvs_engine.cuvs_pool_spec(a) == "1GiB"
    assert cuvs_engine.cuvs_pool_spec(run.parse_args(base + ["--cuvs-rmm-pool", "2GiB"])) == "2GiB"
    for off in cuvs_engine.CUVS_RMM_POOL_OFF + ("OFF",):
        assert cuvs_engine.cuvs_pool_spec(run.parse_args(base + ["--cuvs-rmm-pool", off])) is None
    # cuVS only, dense and sparse (same allocator policy for both); FAISS and
    # the other sparse / multi-vector engines stay as they were
    assert cuvs_engine.cuvs_pool_spec(types.SimpleNamespace(**{**vars(a), "modality": "sparse"})) == "1GiB"
    for change in ({"backend": "gpu"}, {"backend": "spare", "modality": "sparse"},
                   {"modality": "multivector"}):
        assert cuvs_engine.cuvs_pool_spec(types.SimpleNamespace(**{**vars(a), **change})) is None

    # sparse make_backend: the pool is installed before the backend, off leaves rmm alone
    sp = types.SimpleNamespace(**{**vars(a), "modality": "sparse"})
    del calls[:]
    seen = []
    monkeypatch.setattr(cuvs_engine, "SparseCuvsBackend", lambda *a, **k: seen.append(list(calls)))
    run.make_backend(sp, np.float32)
    assert calls == [("pool", "cuda", 1 << 30), ("set", "pool")] and seen == [calls]
    del calls[:]
    sp.cuvs_rmm_pool = "off"
    run.make_backend(sp, np.float32)
    assert calls == [] and len(seen) == 2

    # make_backend: the pool is installed before the backend (its Resources) exists
    del calls[:]
    seen = []
    monkeypatch.setattr(cuvs_engine, "CuvsBackend", lambda *a, **k: seen.append(list(calls)))
    run.make_backend(a, np.float32)
    assert calls == [("pool", "cuda", 1 << 30), ("set", "pool")] and seen == [calls]

    # off: rmm is never imported or touched
    monkeypatch.setitem(sys.modules, "rmm", RmmTripwire("rmm"))
    monkeypatch.delitem(sys.modules, "rmm.mr")
    del calls[:]
    run.make_backend(run.parse_args(base + ["--cuvs-rmm-pool", "off"]), np.float32)
    assert calls == []


@pytest.mark.parametrize("dist", [False, True])
@pytest.mark.parametrize("keep", [None, "some"])
@pytest.mark.parametrize("queries", [None, "subset"])
@pytest.mark.parametrize("k_eff", [K, K - 3])
def test_device_post_matches_host_finish(fake_cuvs, dist, keep, queries, k_eff):
    """ResidentBackend._device_post (the on-device shaping of a filtered
    block's results) is bit for bit FilterAdapter.finish on the host after
    complete()'s _pad: kept rows -> file rows, + offset, expansion of a
    searched subset to all queries with (+-inf, -1), -1 kept everywhere."""
    from bench.data.filters import search as fs
    torch = __import__("torch")
    nq, rows, off = 12, 40, 1000
    rng = np.random.default_rng(7)
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    b.returns_distance = dist
    q = np.sort(rng.choice(nq, 7, replace=False)) if queries else None
    nr = nq if q is None else len(q)
    kp = np.sort(rng.choice(5000, rows, replace=False)).astype(np.int64) if keep else None
    D = rng.standard_normal((nr, k_eff)).astype(np.float32)
    I = rng.integers(0, rows, size=(nr, k_eff)).astype(np.int64)
    I[rng.random(I.shape) < 0.2] = -1   # empty slots
    xb = np.zeros((rows, 4), np.float32)
    # host: complete()'s _pad, then finish (offset added there: on_device False)
    ad = fs.FilterAdapter("bitmap", nq, K, returns_distance=dist)
    Dh, Ih = b._pad(D.copy(), I.copy())
    Dh, Ih = ad.finish(fs.Prep(xb, None, q, kp, off), Dh, Ih)
    # device: the shaping in submit, then complete()'s _pad (a no-op once expanded)
    Dd, Id = b._device_post(xb, torch.from_numpy(D.copy()), torch.from_numpy(I.copy()), off,
                            fs.DevicePost(kp, q, nq))
    Dd, Id = b._pad(Dd.numpy(), Id.numpy())
    assert Dd.dtype == Dh.dtype and Id.dtype == Ih.dtype
    assert np.array_equal(Id, Ih)
    assert np.array_equal(Dd, Dh)   # inf / finfo pads included, bit for bit
