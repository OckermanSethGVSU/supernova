"""--metric ip|cosine and --gt-ids-column.

cosine: every engine's merged scores are cos(q, x), largest first. FAISS has
no cosine metric, so its backends normalise (queries once in setup, each
block inside engine_call, measured as normalize_s: on the device with its own
CUDA event pair on the resident path, on the host with perf_counter on the
CPU backend); cuVS runs metric="cosine" natively
(fake cuvs here: 1 - cos, ascending, as the cuVS source computes it) and
run_pass turns its distances into similarities in the merge bucket, leaving
the -1 padding alone. Everything is checked against an independent numpy
float64 cosine oracle on NON-unit vectors (unit vectors would make cosine
and inner product the same and prove nothing). The GPU paths are mocked as
in test_gpu_backend.py / test_timing_detail.py (torch's CPU device, fake
events on a fake clock); what only a GPU node can show is listed in the
docs/engines.md ("Metric").
"""

import csv

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import make_dummy
from bench.systems import harness_cli as run
from bench.systems import guards, loop
from bench.record import results_io, schema
from bench.verify import oracle as verify
from bench.systems import streaming as base, cpu_reference, cuvs_engine, faiss_engine
from bench.data.sources import InMemorySource, ParquetSource
from test_cuvs_backend import _cuvs_factory, cosine_distance_knn, fake_cuvs  # noqa: F401
from test_timing_detail import Clock, Out, fake_alloc_counters, fake_event_cls

torch = pytest.importorskip("torch")
K = 10
DIM = 24
NROWS = [600, 3, 400, 1, 900]  # one file smaller than k, one of 1 row: -1 padding


def _scaled(rng, n, d):
    """Rows of random direction and norms spread over 0.2 .. 5."""
    x = rng.standard_normal((n, d)).astype(np.float32)
    return (x * rng.uniform(0.2, 5.0, (n, 1))).astype(np.float32)


@pytest.fixture(scope="module")
def nonunit(tmp_path_factory):
    """A corpus and queries whose norms vary a lot: cosine and inner product
    rank them differently. Returns (corpus dir, concatenated xb, xq)."""
    out = tmp_path_factory.mktemp("nonunit")
    xs = []
    for i, n in enumerate(NROWS):
        x = _scaled(np.random.default_rng([21, i]), n, DIM)
        make_dummy.write_vectors(str(out / f"part-{i:05d}.parquet"), x)
        xs.append(x)
    xq = _scaled(np.random.default_rng(22), 40, DIM)
    return str(out), np.concatenate(xs), xq


def unit64(x):
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cos_oracle(xq, xb, k):
    """Independent cosine top-k in float64 (numpy, not faiss): (S, I, xq_n,
    xb_n), the normalised copies for compare_results' rescoring."""
    qn, xn = unit64(xq), unit64(xb)
    S = qn @ xn.T
    I = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(S, I, axis=1), I, qn, xn


def check_cosine(D, I, xq, xb):
    S, Iref, qn, xn = cos_oracle(xq, xb, K + 1)
    assert verify.compare_results(D, I, S, Iref, xq=qn, xb=xn) == []
    assert I.min() >= 0 and D.dtype == np.float32


def _src(path, regime):
    return InMemorySource.load(path) if regime == "memory" else \
        ParquetSource(path, prefetch=2, workers=2)


# ------------------------------------------------------------ helpers

def test_l2_normalized_is_a_unit_copy_and_keeps_zero_rows():
    x = np.array([[3, 4], [0, 0], [-1, 0]], np.float32)
    x.flags.writeable = False  # a read-only Arrow-like block: copied, not touched
    y = base.l2_normalized(x)
    np.testing.assert_allclose(y, [[0.6, 0.8], [0, 0], [-1, 0]], rtol=1e-6)
    assert y.dtype == np.float32 and y.flags.c_contiguous and not np.shares_memory(x, y)
    q = base.normalized_queries(np.array([[0, 2]], np.float16))
    assert q.dtype == np.float16 and q.tolist() == [[0, 1]]


def test_device_normalize_keeps_zero_rows_and_the_source():
    b = base.ResidentBackend(K, np.float32, "cpu", True)
    b._init_torch()
    src = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = b._normalize(src, False)
    assert out.tolist() == [[pytest.approx(0.6), pytest.approx(0.8)], [0.0, 0.0]]
    assert src.tolist() == [[3.0, 4.0], [0.0, 0.0]]  # out of place: a new tensor
    own = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    assert b._normalize(own, True).data_ptr() == own.data_ptr()  # in place
    assert own.tolist() == out.tolist()
    half = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float16)
    for inplace in (False, True):
        h = b._normalize(half.clone(), inplace)
        assert h.dtype == torch.float16
        assert h.tolist() == [[float(np.float16(0.6)), float(np.float16(0.8))], [0, 0]]


def test_owned_only_refuses_a_view_of_the_host_block():
    """In place only on the call's own tensor: a CPU-device .to() of the host
    block is a zero-copy view (never written); a copy or a cast is owned."""
    b = base.ResidentBackend(K, np.float32, "cpu", True)
    b._init_torch()
    xb = np.ones((5, 3), np.float32)
    view = b._to_dev(xb)
    assert view.data_ptr() == xb.ctypes.data and not b._owned(view, xb)
    assert not b._owned(view[2:], xb)  # any overlap
    assert b._owned(view.clone(), xb) and b._owned(view.half(), xb)


def test_oracle_discriminates(nonunit):
    """The non-unit corpus really ranks differently under ip and cosine, so
    an ip result fails the cosine check (and the harness's cosine oracle
    agrees with the numpy one)."""
    path, xb, xq = nonunit
    D_ip, I_ip = faiss.knn(xq, xb, K, metric=faiss.METRIC_INNER_PRODUCT)
    S, Iref, qn, xn = cos_oracle(xq, xb, K + 1)
    assert verify.compare_results(D_ip, I_ip, S, Iref, xq=qn, xb=xn)
    D, I, xq_n, xb_n = verify.oneshot_reference(xq, InMemorySource.load(path).blocks, K + 1,
                                             "cosine")
    assert verify.compare_results(D[:, :K], I[:, :K], S, Iref, xq=qn, xb=xn) == []
    # the float64 oracle's own unit queries / rows (normalised in float64)
    np.testing.assert_allclose(np.linalg.norm(xq_n, axis=1), 1, rtol=1e-14)
    np.testing.assert_allclose([np.linalg.norm(xb_n[i]) for i in range(len(xb_n))], 1,
                               rtol=1e-14)


# ------------------------------------------------------------- FAISS cpu

@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_cpu_cosine_matches_oracle(nonunit, regime):
    path, xb, xq = nonunit
    b = cpu_reference.CpuBackend(K, metric="cosine")
    assert b.setup(xq) > 0  # the one-time query normalise
    np.testing.assert_allclose(np.linalg.norm(b.xq, axis=1), 1, rtol=1e-6)
    src = _src(path, regime)
    blocks_before = [blk.vectors.copy() for blk in src.blocks] if regime == "memory" else []
    D, I, info = loop.run_pass(src, b, len(xq), K)
    check_cosine(D, I, xq, xb)
    for blk, before in zip(getattr(src, "blocks", []), blocks_before):
        np.testing.assert_array_equal(blk.vectors, before)  # never normalised in place
    assert 0 < info["normalize_s"] < info["engine_call_s"]  # inside the call
    assert info["total_s"] == pytest.approx(info["source_stall_s"] + info["engine_call_s"]
                                            + info["merge_s"] + info["other_s"], abs=1e-12)
    assert info["h2d_s"] == "" and info["search_s"] == ""  # no events, as with ip


def test_cpu_ip_unchanged(nonunit):
    path, xb, xq = nonunit
    b = cpu_reference.CpuBackend(K)
    assert b.metric == "ip" and b.setup(xq) == 0.0 and b.xq is xq
    D, I, info = loop.run_pass(InMemorySource.load(path), b, len(xq), K)
    D_ref, I_ref = faiss.knn(xq, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref, xq=xq, xb=xb) == []
    assert info["normalize_s"] == ""


# ------------------------------------------------- FAISS GPU (mocked)

def _faiss_resident(monkeypatch, metric="cosine", events=True, clock=None, seen=None):
    """A resident GpuBackend on torch's CPU device; knn_gpu (mocked after
    construction, see test_gpu_backend.py) checks that it only ever sees
    unit rows under cosine and computes inner product, as the real one."""
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    b = faiss_engine.GpuBackend(K, device="cpu", events=events,
                       metric=metric)

    def knn_gpu(res, xq, xb_t, k, metric=None, **kw):
        assert metric == faiss.METRIC_INNER_PRODUCT  # FAISS has no cosine
        if seen is not None:
            seen.append((xq.norm(dim=1), xb_t.norm(dim=1)))
        if clock is not None:
            clock.op("knn", 4)
        Dn, In = faiss.knn(xq.numpy(), xb_t.numpy(), k, metric=faiss.METRIC_INNER_PRODUCT)
        Dt, It = torch.from_numpy(Dn), torch.from_numpy(In)
        return (Out(Dt, clock), Out(It, clock)) if clock is not None else (Dt, It)

    monkeypatch.setattr(faiss, "knn_gpu", knn_gpu)
    if clock is not None:
        from test_timing_detail import globalize_out
        monkeypatch.setattr(b, "_globalize", globalize_out(b._globalize, clock))
    return b


@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_faiss_resident_cosine_matches_oracle(nonunit, monkeypatch, regime):
    path, xb, xq = nonunit
    seen = []
    b = _faiss_resident(monkeypatch, seen=seen)
    assert b.normalize_blocks and not b.returns_distance
    b.setup(xq)
    src = _src(path, regime)
    blocks_before = [blk.vectors.copy() for blk in src.blocks] if regime == "memory" else []
    D, I, info = loop.run_pass(src, b, len(xq), K)
    check_cosine(D, I, xq, xb)
    # torch's CPU device: the "H2D" is a view of the host block, so the
    # normalise falls back to out of place and the block is never touched
    for blk, before in zip(getattr(src, "blocks", []), blocks_before):
        np.testing.assert_array_equal(blk.vectors, before)
    assert len(seen) == len(NROWS)
    for qn, xn in seen:  # unit queries (setup) and unit blocks (in the call)
        assert torch.allclose(qn, torch.ones_like(qn), atol=1e-5)
        assert torch.allclose(xn, torch.ones_like(xn), atol=1e-5)
    assert info["normalize_s"] > 0  # host stamps on torch's CPU device


@pytest.mark.parametrize("stored", ["f16", "f32"])
def test_faiss_resident_event_chain_has_a_normalize_pair(monkeypatch, tmp_path, stored):
    """cosine: e0 [H2D] e1 [cast] e2 [normalize] e2n [knn] e3 [.cpu()] e4;
    each column is its own pair's span, normalize_s inside engine_call and
    engine_gap_s = engine_call_s minus all five event parts."""
    xs = [_scaled(np.random.default_rng([31, i]), n, DIM) for i, n in enumerate([50, 7, 80])]
    for i, x in enumerate(xs):
        vt = np.float16 if stored == "f16" else np.float32
        make_dummy.write_vectors(str(tmp_path / f"part-{i:05d}.parquet"),
                                 x.astype(np.float16).astype(vt), value_dtype=vt)
    xb = np.concatenate(xs).astype(np.float16).astype(np.float32)
    xq = _scaled(np.random.default_rng(32), 16, DIM)
    clock = Clock()
    b = _faiss_resident(monkeypatch, clock=clock)
    real_to_dev, real_cast, real_norm = b._to_dev, b._cast, b._normalize
    monkeypatch.setattr(b, "_to_dev", lambda x: clock.op("h2d", 1) or real_to_dev(x))
    monkeypatch.setattr(b, "_cast", lambda x: clock.op("cast", 2) or real_cast(x))
    monkeypatch.setattr(b, "_normalize", lambda x, *a: clock.op("norm", 3) or real_norm(x, *a))
    b.setup(xq)
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a: pytest.fail("no device-wide sync in search"))
    fake_alloc_counters(monkeypatch, torch, clock)
    clock.log.clear()
    D, I, info = loop.run_pass(InMemorySource.load(str(tmp_path)), b, len(xq), K)
    cast = stored == "f16"
    per_block = ["ev", "h2d", "ev"] + (["cast", "ev"] if cast else []) + \
        ["norm", "ev", "ev", "knn", "ev", "ev", "ev", "d2h", "d2h", "ev", "query"]
    assert clock.log == per_block * 3
    assert info["h2d_s"] == pytest.approx(3e-3)
    assert info["widen_s"] == pytest.approx(6e-3 if cast else 0.0)
    assert info["normalize_s"] == pytest.approx(9e-3)
    assert info["search_s"] == pytest.approx(12e-3)  # the knn alone
    assert info["d2h_s"] == pytest.approx(48e-3)
    parts = sum(info[c] for c in ("h2d_s", "widen_s", "normalize_s", "search_s", "d2h_s"))
    assert info["engine_gap_s"] == pytest.approx(info["engine_call_s"] - parts)
    check_cosine(D, I, xq, xb)


@pytest.mark.parametrize("stored", ["f32", "f16"])
def test_faiss_resident_cuda_normalises_in_place(monkeypatch, tmp_path, stored):
    """On CUDA (mocked: fake events, a fresh tensor from the "H2D" as a real
    H2D gives) the block is normalised IN PLACE: the tensor the engine
    searches is the H2D's own (fp32) or the widen's own (fp16 stored), not a
    second block-sized copy, and the results still match the oracle."""
    xs = [_scaled(np.random.default_rng([61, i]), n, DIM) for i, n in enumerate([50, 7, 80])]
    vt = np.float16 if stored == "f16" else np.float32
    for i, x in enumerate(xs):
        make_dummy.write_vectors(str(tmp_path / f"part-{i:05d}.parquet"), x.astype(vt),
                                 value_dtype=vt)
    xb = np.concatenate(xs).astype(vt).astype(np.float32)
    xq = _scaled(np.random.default_rng(62), 16, DIM)
    clock = Clock()
    b = _faiss_resident(monkeypatch, clock=clock)
    b.setup(xq)  # the queries' upload is not a block
    made, searched = [], []
    real_to_dev, real_cast = b._to_dev, b._cast
    monkeypatch.setattr(b, "_to_dev",
                        lambda x: made.append(real_to_dev(x).clone()) or made[-1])
    monkeypatch.setattr(b, "_cast", lambda x: made.append(real_cast(x)) or made[-1])
    real_knn = faiss.knn_gpu
    monkeypatch.setattr(faiss, "knn_gpu",
                        lambda res, q, x, k, **kw: searched.append(x.data_ptr())
                        or real_knn(res, q, x, k, **kw))
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    fake_alloc_counters(monkeypatch, torch, clock)
    src = InMemorySource.load(str(tmp_path))
    before = [blk.vectors.copy() for blk in src.blocks]
    D, I, info = loop.run_pass(src, b, len(xq), K)
    check_cosine(D, I, xq, xb)
    per_block = 2 if stored == "f16" else 1  # H2D (+ widen)
    assert len(made) == per_block * len(xs) and len(searched) == len(xs)
    assert searched == [t.data_ptr() for t in made[per_block - 1::per_block]]
    for t in made[per_block - 1::per_block]:  # the searched tensors are unit rows
        assert torch.allclose(t.float().norm(dim=1), torch.ones(t.shape[0]), atol=1e-5)
    for blk, prev in zip(src.blocks, before):
        np.testing.assert_array_equal(blk.vectors, prev)


def test_faiss_resident_ip_chain_unchanged(monkeypatch):
    b = _faiss_resident(monkeypatch, metric="ip")
    assert not b.normalize_blocks
    called = []
    monkeypatch.setattr(b, "_normalize", lambda x, *a: called.append(1) or x)
    b.setup(np.ones((4, DIM), np.float32) * 3)
    assert torch.equal(b.xq, torch.full((4, DIM), 3.0))  # ip: queries untouched
    D, I = b.search(np.ones((20, DIM), np.float32))
    assert not called


def test_faiss_resident_cosine_events_off_runs_unmeasured(nonunit, monkeypatch):
    path, xb, xq = nonunit
    b = _faiss_resident(monkeypatch, events=False)
    calls = []
    real = b._normalize
    monkeypatch.setattr(b, "_normalize", lambda x, *a: calls.append(1) or real(x, *a))
    b.setup(xq)
    D, I, info = loop.run_pass(InMemorySource.load(path), b, len(xq), K)
    check_cosine(D, I, xq, xb)
    assert len(calls) == len(NROWS) and info["normalize_s"] == ""
    assert all(info[c] == "" for c in schema.EVENT_FIELDS)


def _assert_scratch_reused(b, normalize):
    """Same-size and smaller blocks reuse the backend's one scratch buffer
    (no reallocation); a larger one grows it once; the source block (read
    only, like an Arrow buffer) is never written."""
    rng = np.random.default_rng(71)
    blocks = [_scaled(rng, n, DIM) for n in (300, 300, 120, 500)]
    for blk in blocks:
        blk.flags.writeable = False
    copies = [blk.copy() for blk in blocks]
    bufs = []
    for blk in blocks:
        out = normalize(blk)
        bufs.append((b.scratch.buf.ctypes.data, b.scratch.buf.size, b.scratch.grows))
        np.testing.assert_allclose(np.linalg.norm(out.astype(np.float64), axis=1), 1,
                                   rtol=2e-3 if out.dtype == np.float16 else 1e-5)
    assert bufs[0][2] == 1 and bufs[1] == bufs[0] and bufs[2] == bufs[0]
    assert bufs[3][2] == 2 and bufs[3][1] == 500 * DIM
    for blk, c in zip(blocks, copies):
        np.testing.assert_array_equal(blk, c)


def test_cpu_cosine_scratch_is_reused(nonunit):
    _, _, xq = nonunit
    b = cpu_reference.CpuBackend(K, metric="cosine")
    b.setup(xq)
    seen = []
    real = b.scratch.normalized
    b.scratch.normalized = lambda x, *a: seen.append(1) or real(x, *a)
    _assert_scratch_reused(b, lambda blk: (b.search(blk), real(blk))[1])
    assert len(seen) == 4 and cpu_reference.CpuBackend(K).scratch is None  # ip: no scratch


def test_main_presizes_the_scratch(nonunit, monkeypatch, tmp_path):
    """When the largest block is known before the passes (every footer under
    --verify) the scratch is sized and faulted then, so no pass grows it;
    without --verify the first block does."""
    regime = "stream"
    path, _, _ = nonunit
    made = []

    def make_backend(args, dtype):
        made.append(cpu_reference.CpuBackend(args.k, metric=args.metric))
        return made[-1]

    monkeypatch.setattr(run, "make_backend", make_backend)
    for verify in ([], ["--verify"]):
        run.main(["--regime", regime, "--corpus", path, "--backend", "cpu", "--nq", "8",
                  "--metric", "cosine", "--warmup", "0", "--repeats", "1",
                  "--csv", str(tmp_path / "r.csv")] + verify)
        sc = made[-1].scratch
        # stream without --verify: grown by the first block, then by the
        # first larger one (600 rows, then 900)
        assert sc.grows == (2 if regime == "stream" and not verify else 1)
        assert sc.buf.size == max(NROWS) * DIM
    # the search itself never grew it when it was pre-sized
    b = cpu_reference.CpuBackend(K, metric="cosine")
    b.setup(np.ones((2, DIM), np.float32))
    assert b.reserve(max(NROWS) * DIM) >= 0 and b.scratch.grows == 1
    b.search(np.ones((max(NROWS), DIM), np.float32))
    assert b.scratch.grows == 1


# -------------------------------------------------------- cuVS (fake)

@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_cuvs_native_cosine_matches_oracle(fake_cuvs, nonunit, regime):
    path, xb, xq = nonunit
    fake_cuvs.metric = "cosine"  # build must be asked for cuVS's own cosine
    b = cuvs_engine.CuvsBackend(K, device="cpu", metric="cosine")
    assert b.returns_distance and not b.normalize_blocks
    b.setup(xq)
    assert torch.equal(b.xq, torch.from_numpy(xq))  # no harness normalisation
    D, I, info = loop.run_pass(_src(path, regime), b, len(xq), K)
    check_cosine(D, I, xq, xb)
    assert fake_cuvs.ks == [min(K, n) for n in NROWS]
    assert info["normalize_s"] == "" and info["build_s"] > 0


def test_cuvs_distance_conversion_keeps_padding(fake_cuvs, tmp_path):
    """A 3-row block with k=10: cuVS returns 3 distances (1 - cos,
    ascending), the backend pads to k with (HIGHEST, -1) -- the worst
    distance, since a distance engine is merged in a min-heap -- and
    run_pass turns the final top-k into cos once: real columns 1 - d, the
    -1 columns FAISS's lowest float (as every engine's empty slots)."""
    fake_cuvs.metric = "cosine"
    x = _scaled(np.random.default_rng(41), 3, DIM)
    xq = _scaled(np.random.default_rng(42), 8, DIM)
    make_dummy.write_vectors(str(tmp_path / "part-00000.parquet"), x)
    b = cuvs_engine.CuvsBackend(K, device="cpu", metric="cosine")
    b.setup(xq)
    Draw, Iraw = b.search(x)
    Dd, Id = cosine_distance_knn(xq, x, 3)
    np.testing.assert_array_equal(Draw[:, :3], Dd)
    np.testing.assert_array_equal(Iraw[:, :3], Id)
    assert (np.diff(Draw[:, :3], axis=1) >= 0).all()  # ascending distances
    assert (Draw[:, 3:] == np.finfo(np.float32).max).all() and (Iraw[:, 3:] == -1).all()
    D, I, _ = loop.run_pass(InMemorySource.load(str(tmp_path)), b, len(xq), K)
    np.testing.assert_allclose(D[:, :3], 1 - Dd, rtol=0, atol=0)
    S, Iref, _, _ = cos_oracle(xq, x, 3)
    np.testing.assert_array_equal(I[:, :3], Iref)
    np.testing.assert_allclose(D[:, :3], S, atol=1e-6)
    assert (D[:, 3:] == np.finfo(np.float32).min).all() and (I[:, 3:] == -1).all()


@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_cuvs_cosine_merged_as_distances_converted_once(fake_cuvs, nonunit, monkeypatch,
                                                        regime):
    """cuVS cosine: the per-block results go into the heap AS distances (a
    min-heap, keep_max=False), never converted per block, and the final top-k
    is converted once: the same ids and the same float32 1 - d scores the
    old per-block conversion gave (no exact ties in this corpus)."""
    path, xb, xq = nonunit
    fake_cuvs.metric = "cosine"
    made, added = [], []
    real = faiss.ResultHeap

    class Spy(real):
        def __init__(self, nq, k, keep_max=False):
            super().__init__(nq, k, keep_max=keep_max)
            made.append(keep_max)

        def add_result(self, D, I):
            added.append(D.copy())
            return super().add_result(D, I)

    monkeypatch.setattr(faiss, "ResultHeap", Spy)
    b = cuvs_engine.CuvsBackend(K, device="cpu", metric="cosine")
    b.setup(xq)
    D, I, _ = loop.run_pass(_src(path, regime), b, len(xq), K)
    assert made == [False]
    # what reached the heap were the engine's raw distances (cuVS's own
    # ascending 1 - cos), never similarities
    raws = []
    for blk in InMemorySource.load(path).blocks:
        Dr, _ = b.search(blk.vectors)
        raws.append(Dr)
    assert len(added) == len(raws)
    for got, raw in zip(added, raws):
        np.testing.assert_array_equal(got, raw)
    # the old flow (convert each block, merge similarities in a max-heap)
    old = real(len(xq), K, keep_max=True)
    for blk in InMemorySource.load(path).blocks:
        Dr, Ir = b.search(blk.vectors)
        # the old flow's padding was the lowest float (now the backend pads a
        # distance engine with the highest; either way -1 ids)
        old.add_result(np.where(Ir >= 0, 1 - Dr, np.finfo(np.float32).min),
                       np.where(Ir >= 0, Ir + blk.offset, -1))
    old.finalize()
    np.testing.assert_array_equal(I, old.I)
    np.testing.assert_array_equal(D, old.D)
    check_cosine(D, I, xq, xb)


def test_cuvs_ip_unconverted(fake_cuvs, corpus, queries):
    b = cuvs_engine.CuvsBackend(K, device="cpu")
    assert b.metric == "ip" and not b.returns_distance
    b.setup(queries)
    D, I, info = loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)
    D_ref, I_ref = faiss.knn(queries, corpus[1], K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == [] and info["normalize_s"] == ""


# ----------------------------------------------------- run.main / CSV

def _rows(p):
    return list(csv.DictReader(open(p)))


def test_main_cpu_cosine_verify(nonunit, tmp_path, capsys):
    path, _, _ = nonunit
    out = tmp_path / "r.csv"
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--prefetch", pf, "--corpus", path, "--backend", "cpu", "--nq", "16",
                  "--metric", "cosine", "--warmup", "0", "--repeats", "2", "--verify",
                  "--csv", str(out)])
    assert capsys.readouterr().out.count("verify OK") == 2
    rows = _rows(out)
    assert {r["metric"] for r in rows} == {"cosine"} and {r["verified"] for r in rows} == {"ok"}
    assert all(0 < float(r["normalize_s"]) < float(r["engine_call_s"]) for r in rows)
    assert all(r["gt_recall"] == "" for r in rows)
    assert {r["metric"] for r in _rows(results_io.trace_path(str(out)))} == {"cosine"}


def test_main_cosine_verify_catches_an_ip_engine(nonunit, monkeypatch, tmp_path):
    """The cosine oracle is really cosine: an engine computing raw inner
    product on this non-unit corpus fails --verify."""
    path, _, _ = nonunit
    monkeypatch.setattr(run, "make_backend", lambda args, dtype: cpu_reference.CpuBackend(args.k))
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu", "--nq", "16",
                  "--metric", "cosine", "--warmup", "0", "--repeats", "1", "--verify",
                  "--csv", str(tmp_path / "r.csv")])


def test_main_faiss_resident_cosine_verify(nonunit, monkeypatch, tmp_path, capsys):
    path, _, _ = nonunit

    def make_backend(args, dtype):
        return _faiss_resident(monkeypatch, metric=args.metric)

    monkeypatch.setattr(run, "make_backend", make_backend)
    out = tmp_path / "r.csv"
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--prefetch", pf, "--corpus", path, "--nq", "16", "--metric", "cosine",
                  "--warmup", "0", "--repeats", "1", "--verify", "--csv", str(out)])
    assert capsys.readouterr().out.count("verify OK") == 2
    for r in _rows(out):
        assert (r["backend"], r["metric"], r["gpu_events"]) == ("gpu", "cosine", "on")
        assert float(r["normalize_s"]) > 0
        parts = sum(float(r[c]) for c in ("h2d_s", "widen_s", "normalize_s", "search_s",
                                           "d2h_s"))
        assert float(r["engine_gap_s"]) == pytest.approx(float(r["engine_call_s"]) - parts)


def test_main_cuvs_cosine_verify(fake_cuvs, nonunit, monkeypatch, tmp_path, capsys):
    path, _, _ = nonunit
    fake_cuvs.metric = "cosine"
    _cuvs_factory(monkeypatch)
    out = tmp_path / "r.csv"
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--prefetch", pf, "--corpus", path, "--backend", "cuvs", "--nq", "16",
                  "--metric", "cosine", "--warmup", "0", "--repeats", "1", "--verify",
                  "--csv", str(out)])
    assert capsys.readouterr().out.count("verify OK") == 2
    for r in _rows(out):
        assert (r["engine"], r["metric"], r["normalize_s"]) == ("cuvs", "cosine", "")


# the engine side's fp32 query copies (cosine: + a FAISS backend's normalised
# one); the float64 oracle's own queries are counted in its phase
@pytest.mark.parametrize("metric,copies", [("ip", 1), ("cosine", 2)])
def test_memory_guard_counts_the_normalised_query_copies(nonunit, monkeypatch, tmp_path,
                                                         metric, copies):
    path, _, _ = nonunit
    seen = []
    real = guards.verify_memory_need
    monkeypatch.setattr(guards, "verify_memory_need",
                        lambda *a, **kw: seen.append(kw["query_copies"]) or real(*a, **kw))
    run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu", "--nq", "8",
              "--metric", metric, "--warmup", "0", "--repeats", "1", "--verify",
              "--csv", str(tmp_path / "r.csv")])
    assert seen == [copies]
    nq, d = 1000, 64
    assert real(10, d, nq, 5, 1, 4, query_copies=2) - real(10, d, nq, 5, 1, 4) == \
        nq * d * 4


# ------------------------------------------------------- --gt-ids-column

def test_gt_recall_counts():
    I = np.array([[1, 2, 3], [4, 5, -1], [7, 8, 9]])
    gt = np.array([[3, 2, 9, 7], [4, -1, 8, 8], [-1, -1, -1, 0]])
    # q0: {2, 3} of {3, 2, 9} = 2/3; q1: {4} of {4, 8} = 1/2; q2: no valid gt, skipped
    assert results_io.gt_recall(I, gt) == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert results_io.gt_recall(np.array([[0, 1]]), np.array([[1, 0, 5]])) == 1.0


def test_gt_id_matrix_list_types():
    fixed = pa.FixedSizeListArray.from_arrays(pa.array(np.arange(12), pa.int64()), 4)
    np.testing.assert_array_equal(results_io.gt_id_matrix(fixed[1:], "x"),
                                  np.arange(4, 12).reshape(2, 4))
    var = pa.array([[1, 2, 3], [4, 5], [6, 7, 8, 9]], pa.list_(pa.int32()))
    np.testing.assert_array_equal(results_io.gt_id_matrix(var[1:], "x"), [[4, 5], [6, 7]])
    with pytest.raises(ValueError, match="not a list of integer ids"):
        results_io.gt_id_matrix(pa.array([[1.0]], pa.list_(pa.float32())), "x")
    with pytest.raises(ValueError, match="null"):
        results_io.gt_id_matrix(pa.array([[1], None], pa.list_(pa.int64())), "x")


@pytest.fixture
def gt_queries(nonunit, tmp_path):
    """A query file with exact cosine ground truth (best first) beside the
    embeddings, aligned row for row: ``ids`` fixed_size_list<int64>[15],
    ``half`` the same with ranks 5..9 swapped for ranks 10..14 (recall@10
    exactly 0.5), ``var`` a variable-length list<int64> (12..15 ids)."""
    _, xb, xq = nonunit
    _, gt, _, _ = cos_oracle(xq, xb, 15)
    half = gt.copy()
    half[:, 5:10] = gt[:, 10:15]
    var = [list(r[:12 + i % 4]) for i, r in enumerate(gt.tolist())]
    t = pa.table({
        "embedding": pa.array(xq.tolist(), pa.list_(pa.float32())),
        "ids": pa.FixedSizeListArray.from_arrays(pa.array(gt.ravel(), pa.int64()), 15),
        "half": pa.FixedSizeListArray.from_arrays(pa.array(half.ravel(), pa.int64()), 15),
        "var": pa.array(var, pa.list_(pa.int64())),
    })
    p = tmp_path / "q.parquet"
    pq.write_table(t, p, row_group_size=7)  # several row groups
    return str(p)


@pytest.mark.parametrize("column,want", [("ids", 1.0), ("half", 0.5), ("var", 1.0)])
def test_main_gt_recall(nonunit, gt_queries, tmp_path, capsys, column, want):
    path, _, _ = nonunit
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu", "--metric",
              "cosine", "--query-file", gt_queries, "--nq", "30", "--gt-ids-column", column,
              "--warmup", "1", "--repeats", "1", "--csv", str(out)])
    rows = _rows(out)
    assert [float(r["gt_recall"]) for r in rows] == [pytest.approx(want)] * 2
    assert f"gt_recall@{K} (cosine, last pass, 30 queries vs {column}): {want:.4f}" in \
        capsys.readouterr().out


def test_main_gt_recall_ip_reads_lower(nonunit, gt_queries, tmp_path):
    """Sanity of the sanity check: the cosine ground truth against an ip run
    on non-unit vectors is visibly below 1."""
    path, _, _ = nonunit
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu", "--query-file",
              gt_queries, "--nq", "30", "--gt-ids-column", "ids", "--warmup", "0",
              "--repeats", "1", "--csv", str(out)])
    assert float(_rows(out)[0]["gt_recall"]) < 0.9


def test_gt_guards(nonunit, gt_queries, tmp_path):
    path, _, _ = nonunit
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", path, "--csv", "x",
                        "--gt-ids-column", "ids"])  # no --query-file
    with pytest.raises(SystemExit, match="fewer than --k 16"):
        run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu", "--query-file",
                  gt_queries, "--gt-ids-column", "ids", "--k", "16", "--csv",
                  str(tmp_path / "r.csv")])
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", path, "--csv", "x",
                        "--metric", "l2"])

