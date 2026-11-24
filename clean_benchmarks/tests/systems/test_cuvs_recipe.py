"""cuVS's own exact-search recipe (cuvs_engine.CalcTruthMerge / CalcTruthMixin: NVIDIA's
generate_groundtruth calc_truth) against the per-file host ResultHeap path (per_file_host), on
the CPU with the fake cuvs package (test_cuvs_backend.fake_cuvs) and the numpy-backed CuPy
stand-in (tests/fake_cupy.py).

Covered: the merge itself against a one-shot top-k (numpy); dense / sparse / filtered top-k of
device:file (each file as handed is one batch, k > rows in a file) equal to the host path's
(byte-identical: the fixtures have no exact ties); every corpus row searched exactly once;
calc_truth's op order (build, search, sync, asarray, concatenate, argsort, take); the untimed
merge-kernel warm-up; the h2d overlap issued before a block's sync; --d2h flags a
recorded no-op; the CLI (no regroup knob: each file is one block) and the GPU memory guard's
arithmetic.

What the mocks cannot show (checked on a GPU): CuPy on the torch stream through ExternalStream,
the rmm_cupy_allocator pool, real cuVS tie order and timings.
"""

import csv
import types

import faiss
import numpy as np
import pytest

from bench.systems import harness_cli as run
from bench.systems import loop, cuvs_engine, guards
from bench.systems import base as systems
from bench.verify import oracle as verify
from bench.data.sources import InMemorySource, ParquetSource
from test_cuvs_backend import _cuvs_factory, fake_cuvs  # noqa: F401 (fixture)
from test_sparse import FakeRes, fake_knn, sparse_corpus  # noqa: F401 (fixture)
import fake_cupy

torch = pytest.importorskip("torch")
K = 10


# ------------------------------------------------------------- the merge (numpy)

@pytest.mark.parametrize("metric", ["inner_product", "cosine"])
@pytest.mark.parametrize("sizes", [[50, 50, 50], [3, 40, 1, 7], [200], [5, 5]])
def test_merge_equals_one_shot_topk(metric, sizes):
    """CalcTruthMerge over batches (local ids + their offsets, k_eff = min(k, rows)) gives the
    one-shot top-k of the whole corpus; distinct scores, so ids and scores are exact."""
    rng = np.random.default_rng(len(sizes))
    nq, n = 6, sum(sizes)
    S = rng.permutation(nq * n).reshape(nq, n).astype(np.float32) / 7
    cp = fake_cupy.make()
    m = cuvs_engine.CalcTruthMerge(cp, K, metric, n_base=n)
    off = 0
    for s in sizes:
        blk = S[:, off:off + s]
        o = np.argsort(-blk if metric == "inner_product" else blk, axis=1)[:, :min(K, s)]
        m.add(np.take_along_axis(blk, o, 1), o.astype(np.int64), off)
        off += s
    D, I = m.result(nq)
    want = np.argsort(-S if metric == "inner_product" else S, axis=1)[:, :K]
    kk = min(K, n)
    assert np.array_equal(I[:, :kk], want[:, :kk])
    assert np.array_equal(D[:, :kk], np.take_along_axis(S, want, 1)[:, :kk])
    assert I.dtype == np.int64 and D.dtype == np.float32 and D.shape == (nq, K)
    if n < K:   # padded like every engine (worst, -1)
        assert (I[:, n:] == -1).all()
    # calc_truth's ops: asarray x2 (the first batch is assigned), then per later batch
    # asarray x2, concatenate x2, argsort, take_along_axis x2; one asnumpy pair at the end
    ops = [o[0] for o in cp.log]
    later = ["asarray", "asarray", "concatenate", "concatenate", "argsort",
             "take_along_axis", "take_along_axis"]
    assert ops == ["asarray", "asarray"] + later * (len(sizes) - 1) + ["asnumpy", "asnumpy"]


def test_merge_offset_and_id_dtype_match_their_utils():
    """offset_neighbor_indices / neighbor_index_accumulator_dtype as cuvs_bench's utils.py:
    int32 up to int32-max rows, int64 above (and when the base is not known)."""
    a = np.array([[0, 3]], np.int64)
    assert cuvs_engine.offset_neighbor_indices(a, 10, 100).dtype == np.int32
    assert cuvs_engine.offset_neighbor_indices(a, 10, 2**31).dtype == np.int64
    assert cuvs_engine.offset_neighbor_indices(a, 10, None).dtype == np.int64
    assert cuvs_engine.offset_neighbor_indices(a, 10, 100).tolist() == [[10, 13]]


def test_merge_ties_follow_argsort_not_an_id_rule():
    """Exact ties keep whatever xp.argsort keeps (their default), not lowest-id-first; the
    tie-aware checks accept either."""
    cp = fake_cupy.make()
    m = cuvs_engine.CalcTruthMerge(cp, 2, "inner_product", 10)
    m.add(np.array([[1.0, 0.5]], np.float32), np.array([[5, 6]]), 0)
    m.add(np.array([[1.0, 0.1]], np.float32), np.array([[0, 1]]), 0)
    D, I = m.result(1)
    assert D.tolist() == [[1.0, 1.0]] and sorted(I[0].tolist()) == [0, 5]


def test_merge_empty_slots_carry_the_worst_float():
    cp = fake_cupy.make()
    m = cuvs_engine.CalcTruthMerge(cp, 3, "inner_product", 10)
    m.add(np.array([[2.0, -np.inf, -np.inf]], np.float32), np.array([[4, -1, -1]]), 0)
    D, I = m.result(1)
    assert I.tolist() == [[4, -1, -1]] and D[0, 1] == D[0, 2] == np.finfo(np.float32).min
    D, I = cuvs_engine.CalcTruthMerge(cp, 2, "cosine").result(2)   # nothing searched
    assert (I == -1).all() and (D == np.finfo(np.float32).max).all()


# ------------------------------------------------------------- dense, run_pass

def _backend(fake, metric="ip", merge="device", **kw):
    b = cuvs_engine.CuvsBackend(K, device="cpu", metric=metric, **kw)
    b.configure_merge(merge)
    return b


def _searched_rows(fake):
    return [n for op, _, n in fake.log if op == "build"]


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_dense_device_merge_equals_per_file_host(fake_cuvs, corpus, queries, metric):
    """device:file (ROWS = 1200, 3, 700, 2000, 1, 950: files of fewer than k rows) gives the
    per-file host path's top-k byte for byte, and every file is built exactly once."""
    path, xb = corpus
    fake_cuvs.metric = "inner_product" if metric == "ip" else "cosine"
    out = {}
    for merge in ("host", "device"):
        fake_cuvs.log.clear()
        b = _backend(fake_cuvs, metric, merge)
        b.setup(queries)
        D, I, info = loop.run_pass(ParquetSource(path, prefetch=2, workers=2), b,
                                   len(queries), K)
        out[merge] = (D, I, _searched_rows(fake_cuvs), list(fake_cuvs.log))
    Dh, Ih, _, _ = out["host"]
    Dd, Id, built, log = out["device"]
    assert np.array_equal(Ih, Id)
    assert np.array_equal(Dh, Dd)   # the same cuVS calls: bit for bit
    from conftest import ROWS
    assert sum(built) == len(xb) and built == ROWS
    # build, search, then their resources.sync(): one per batch
    assert fake_cuvs.capi.syncs >= len(built)


def test_every_row_searched_once_in_order(fake_cuvs, corpus, queries, monkeypatch):
    """The searched blocks, laid end to end, are the corpus rows in file order, each once."""
    path, xb = corpus
    seen = []
    real = fake_cuvs.build

    def build(dataset, **kw):
        seen.append(dataset.numpy().copy())
        return real(dataset, **kw)
    monkeypatch.setattr(fake_cuvs, "build", build)
    import sys
    sys.modules["cuvs.neighbors.brute_force"].build = build
    b = _backend(fake_cuvs)
    b.setup(queries)
    seen.clear()
    loop.run_pass(InMemorySource.load(path), b, len(queries), K)
    assert np.array_equal(np.concatenate(seen), xb)


def test_dense_device_merge_matches_reference(fake_cuvs, corpus, queries):
    path, xb = corpus
    b = _backend(fake_cuvs)
    b.setup(queries)
    D, I, info = loop.run_pass(ParquetSource(path, prefetch=2, workers=2), b, len(queries), K)
    D_ref, I_ref = faiss.knn(queries, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []
    # torch's CPU device: host stamps stand in for events; the merge lies inside search_s
    assert info["merge_s"] >= 0 and 0 <= info["device_merge_s"] <= info["search_s"]


def test_device_merge_events_attribute_the_merge(fake_cuvs, monkeypatch, corpus, queries):
    """With (faked) CUDA events: device_merge_s is its own span inside search_s, ids_s ends
    where it starts, d2h_s is the one copy back; no host heap is built."""
    from test_timing_detail import Clock, fake_alloc_counters, fake_event_cls
    path, _ = corpus
    calls = []
    monkeypatch.setattr(faiss, "ResultHeap", lambda *a, **kw: calls.append(a))
    for _ in range(2):
        b = _backend(fake_cuvs)
        b.setup(queries)
        clock = Clock()
        b._cuda = True
        monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
        fake_alloc_counters(monkeypatch, torch, clock)
        D, I, info = loop.run_pass(InMemorySource.load(path), b, len(queries), K)
        assert isinstance(info["device_merge_s"], float)
        assert 0 <= info["device_merge_s"] <= info["search_s"]
        assert info["merge_s"] >= 0
        b._cuda = False
    assert calls == []   # no faiss.ResultHeap on the device-merge path


def test_h2d_overlap_upload_issued_before_the_sync(fake_cuvs, monkeypatch, tmp_path):
    """--h2d-overlap with cuVS's recipe: block i+1's upload (the copy stream) is issued before
    block i's resources.sync(), and h2d_overlapped > 0 (the harness's shared pipelining; their
    tool uploads in-line)."""
    from test_run import _fp16_pair
    from test_h2d import _pinner
    import time
    f16, _, _ = _fp16_pair(tmp_path)
    order = []
    real_upload = cuvs_engine.CuvsBackend._upload
    monkeypatch.setattr(cuvs_engine.CuvsBackend, "_upload",
                        lambda self, x: order.append("upload") or real_upload(self, x))
    real_search = fake_cuvs.search

    def slow_search(*a, **kw):
        time.sleep(0.2)   # a slow engine: the readers get ahead, so i+1 is ready
        return real_search(*a, **kw)
    import sys
    sys.modules["cuvs.neighbors.brute_force"].search = slow_search
    real_sync = type(fake_cuvs.Resources()).sync
    monkeypatch.setattr(type(fake_cuvs.Resources()), "sync",
                        lambda self: order.append("sync") or real_sync(self))
    _cuvs_factory(monkeypatch)
    monkeypatch.setattr(cuvs_engine.base, "host_pinner", lambda backend: _pinner()[0])
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", f16, "--nq", "8", "--warmup", "0",
              "--repeats", "1", "--csv", str(out), "--backend", "cuvs", "--prefetch", "3",
              "--workers", "2", "--h2d", "pinned", "--h2d-overlap"])
    r = list(csv.DictReader(open(out)))[-1]
    assert r["h2d_mode"] == "pinned-overlap" and int(r["h2d_overlapped"]) > 0
    # an upload issued ahead (two uploads before a sync) at least once
    assert any(order[i:i + 3] == ["upload", "upload", "sync"] for i in range(len(order)))


def test_d2h_flags_are_a_recorded_noop(fake_cuvs, monkeypatch, tmp_path):
    """cuVS's recipe copies back once at the end: --d2h pinned / --d2h-overlap change nothing,
    d2h_mode records final-once and cuvs_merge the recipe; the top-k is the same."""
    from test_run import _fp16_pair
    f16, _, _ = _fp16_pair(tmp_path)
    _cuvs_factory(monkeypatch)
    out = tmp_path / "r.csv"
    res = {}
    for name, extra in (("plain", []), ("d2h", ["--h2d", "pinned", "--d2h", "pinned",
                                                "--d2h-overlap"])):
        from test_h2d import _pinner
        monkeypatch.setattr(cuvs_engine.base, "host_pinner", lambda backend: _pinner()[0])
        run.main(["--regime", "stream", "--corpus", f16, "--nq", "8", "--warmup", "0",
                  "--repeats", "1", "--csv", str(out), "--backend", "cuvs", "--prefetch", "2",
                  "--save-results", str(tmp_path / f"{name}.npz"), *extra])
        res[name] = np.load(tmp_path / f"{name}.npz")
    rows = list(csv.DictReader(open(out)))
    assert [r["d2h_mode"] for r in rows] == ["final-once", "final-once"]
    assert {r["cuvs_merge"] for r in rows} == {"device:file"}
    assert np.array_equal(res["plain"]["I"], res["d2h"]["I"])


# ------------------------------------------------------------- the untimed warm-up

WARM_OPS = ["asarray", "asarray", "asarray", "asarray", "concatenate", "concatenate", "argsort",
            "take_along_axis", "take_along_axis", "asnumpy", "asnumpy"]


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_warm_merge_issues_every_merge_op_once_untimed(fake_cuvs, queries, metric):
    """warm_merge: one two-block CalcTruthMerge (every op the timed merge uses, float32 D /
    int64 I in, the pass's id dtype), no stats recorded, the pass's own merge untouched; a
    no-op for the host merge."""
    b = _backend(fake_cuvs, metric)
    b.setup(queries)
    b.n_base = 4000
    log = fake_cuvs.cupy.log
    log.clear()
    before = b.stats.as_dict()
    assert b.warm_merge() >= 0
    assert [o[0] for o in log] == WARM_OPS
    assert [o[2] for o in log if o[0] == "asarray"] == ["float32", "int64"] * 2
    assert b._merge is None   # begin_pass makes the pass's own merge
    assert b.stats.as_dict() == before
    h = _backend(fake_cuvs, metric, merge="host")
    h.setup(queries)
    log.clear()
    assert h.warm_merge() == 0.0 and log == []


def test_sparse_warm_merge(monkeypatch, sparse_corpus):
    from bench.data.formats import sparse
    _cdir, qp, _parts, _x = sparse_corpus
    cuvs = types.ModuleType("cuvs")
    cuvs.__version__ = "26.10.00-fake"
    common = types.ModuleType("cuvs.common")
    common.Resources = FakeRes
    cuvs.common = common
    import sys
    monkeypatch.setitem(sys.modules, "cuvs", cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.common", common)
    cp = fake_cupy.install(monkeypatch)
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    b = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=fake_knn([]), metric="ip")
    b.configure_merge("device")
    b.setup(q)
    cp.log.clear()
    b.warm_merge()
    assert [o[0] for o in cp.log] == WARM_OPS


@pytest.mark.parametrize("extra", [[], ["--metric", "cosine"], ["--cuvs-merge", "host"]])
def test_timed_first_merge_is_not_the_first_use(fake_cuvs, monkeypatch, tmp_path, extra):
    """harness_cli runs the warm-up after the guard and before the pass: every CuPy op of the
    timed pass's merge (and its copy back) was already issued, with the same input dtypes,
    before the first cuVS build; --cuvs-merge host issues none."""
    from test_run import _fp16_pair
    _f16, f32, _ = _fp16_pair(tmp_path)
    _cuvs_factory(monkeypatch)
    if "cosine" in extra:
        fake_cuvs.metric = "cosine"
    log = fake_cuvs.cupy.log
    real_build = fake_cuvs.build

    def build(dataset, **kw):
        log.append(("BUILD",))
        return real_build(dataset, **kw)
    import sys
    sys.modules["cuvs.neighbors.brute_force"].build = build
    log.clear()
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", f32, "--nq", "8", "--warmup", "0",
              "--repeats", "1", "--csv", str(out), "--backend", "cuvs", "--prefetch", "2",
              "--clock-json", str(tmp_path / "clock.json"), *extra])
    first = next(i for i, o in enumerate(log) if o[0] == "BUILD")
    warm, timed = log[:first], [o for o in log[first:] if o[0] != "BUILD"]
    if "host" in extra:
        assert warm == [] and timed == []
        return
    assert [o[0] for o in warm] == WARM_OPS
    assert {o[0] for o in timed} <= {o[0] for o in warm}
    assert {"concatenate", "argsort", "take_along_axis", "asnumpy"} <= {o[0] for o in timed}
    dt = lambda ops: {o[2] for o in ops if o[0] == "asarray"}   # noqa: E731
    assert dt(timed) == dt(warm) == {"float32", "int64"}
    r = list(csv.DictReader(open(out)))[-1]
    assert float(r["setup_s"]) >= 0


# ------------------------------------------------------------- filtered

FILTER_CASES = [
    ("static", ["--filter-mode", "bitmap"]),
    ("per_query", ["--filter-mode", "bitmap"]),
    ("per_query", ["--filter-mode", "bitmap", "--filter-skip", "none"]),
    ("per_query", ["--filter-mode", "bitmap", "--filter-skip", "files"]),
    ("per_query", ["--filter-mode", "bitmap", "--filter-compact", "union"]),
    ("per_query", ["--filter-mode", "bitmap", "--metric", "cosine"]),
    ("static", ["--filter-mode", "gather"]),
    ("static", ["--filter-mode", "gather", "--metric", "cosine"]),
]


@pytest.fixture(scope="module")
def filt_corpus(tmp_path_factory):
    import test_filtered_run as T
    from test_duckdb_filter import make_corpus, make_queries
    import pyarrow.parquet as pq
    root = tmp_path_factory.mktemp("cuvs_recipe_filt")
    (root / "corpus").mkdir()
    for i, n in enumerate(T.ROWS):
        p = make_corpus(n, seed=200 + i)
        x = np.random.default_rng(i).standard_normal((n, T.D)).astype(np.float32)
        pq.write_table(p.append_column("embedding", T._emb(x)),
                       str(root / "corpus" / f"part-{i:05d}.parquet"))
    q = make_queries(T.NQ, seed=11)
    xq = np.random.default_rng(99).standard_normal((T.NQ, T.D)).astype(np.float32)
    pq.write_table(q.append_column("embedding", T._emb(xq)), str(root / "q.parquet"))
    return root, {name: T.write_filter(root, name) for name in ("static", "per_query")}


@pytest.mark.slow
@pytest.mark.parametrize("name,extra", FILTER_CASES)
def test_filtered_device_merge_equals_host(fake_cuvs, monkeypatch, tmp_path, filt_corpus,
                                           name, extra):
    """Every filtered mode (bitmap static / per-query, union, files, gather) under cuVS's
    recipe (each file as handed, ROWS 301, 7, 512, 1, 200) gives the masked CPU reference's and
    the per-file host path's top-k; --verify passes; the small-k counter still counts every
    prefilter call (k = 10 <= 256)."""
    import test_filtered_run as T
    root, fpaths = filt_corpus
    _cuvs_factory(monkeypatch)
    if "cosine" in extra:
        fake_cuvs.metric = "cosine"
    out = {}
    calls = {}
    for variant in ("masked", "host", "device"):
        res = tmp_path / f"{variant}.npz"
        csvp = tmp_path / f"{variant}.csv"
        backend = "masked" if variant == "masked" else "cuvs"
        ex = list(extra)
        if variant == "masked":
            ex = [e for e in ex if e not in ("--filter-mode",)]
            i = ex.index("gather") if "gather" in ex else ex.index("bitmap")
            ex.pop(i)
            ex = ["--filter-mode", "gather" if "gather" in extra else "bitmap"] + ex
        if variant == "host":
            ex += ["--cuvs-merge", "host"]
        n0 = len(fake_cuvs.prefilters)
        run.main(["--regime", "stream", "--corpus", str(root / "corpus"), "--column",
                  "embedding", "--backend", backend, "--query-file", str(root / "q.parquet"),
                  "--query-column", "embedding", "--nq", str(T.NQ), "--k", str(T.K),
                  "--filter-config", fpaths[name], "--filter-search", "f",
                  "--csv", str(csvp), "--save-results", str(res), "--verify", "--warmup", "0",
                  "--repeats", "1", "--prefetch", "3", "--workers", "2", *ex])
        out[variant] = np.load(res)
        r = list(csv.DictReader(open(csvp)))[-1]
        assert r["verified"] == "ok"
        calls[variant] = (len(fake_cuvs.prefilters) - n0, r)
    for v in ("host", "device"):
        assert np.array_equal(out["masked"]["I"], out[v]["I"]), v
        ok = out["masked"]["I"] >= 0
        assert np.allclose(out["masked"]["D"][ok], out[v]["D"][ok], atol=1e-5)
    assert np.array_equal(out["host"]["D"], out["device"]["D"])
    n_pf, r = calls["device"]
    if "bitmap" in extra:
        assert int(r["cuvs_filter_small_k_calls"]) == n_pf > 0
    assert r["cuvs_merge"] == "device:file"


# ------------------------------------------------------------- sparse

@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_sparse_device_merge_equals_host(monkeypatch, metric, sparse_corpus):
    """Sparse cuVS: each file as handed, one build + search per file, the device merge: the
    host path's top-k, every row searched once."""
    from bench.data.formats import sparse
    cdir, qp, parts, x = sparse_corpus
    cuvs = types.ModuleType("cuvs")
    cuvs.__version__ = "26.10.00-fake"
    common = types.ModuleType("cuvs.common")
    common.Resources = FakeRes
    cuvs.common = common
    import sys
    monkeypatch.setitem(sys.modules, "cuvs", cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.common", common)
    fake_cupy.install(monkeypatch)
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    out = {}
    for merge in ("host", "device"):
        calls = []
        b = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=fake_knn(calls), metric=metric)
        b.configure_merge(merge)
        b.setup(q)
        del calls[:]
        src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
        D, I, _ = loop.run_pass(src, b, q.n_rows, K)
        out[merge] = (D, I, [c[0] for c in calls])
    (Dh, Ih, _), (Dd, Id, built) = out["host"], out["device"]
    np.testing.assert_array_equal(Ih, Id)
    np.testing.assert_array_equal(Dh, Dd)
    assert sum(built) == len(x) and len(built) == len(parts)


# ------------------------------------------------------------- CLI, presets, guard

BASE = ["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "cuvs"]


def test_cli_defaults_and_refusals():
    """--cuvs-merge device is the default for --backend cuvs; there is no 500k regroup knob
    (each file is one block): --cuvs-batch-rows is not an option."""
    a = run.parse_args(BASE)
    assert a.cuvs_merge == "device" and not hasattr(a, "cuvs_batch_rows")
    assert run.parse_args(BASE + ["--cuvs-merge", "host"]).cuvs_merge == "host"
    f = run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "gpu"])
    assert f.cuvs_merge is None
    for bad in (["--cuvs-batch-rows", "500000"], ["--cuvs-batch-rows", "file"],
                ["--cuvs-merge", "both"]):
        with pytest.raises(SystemExit):
            run.parse_args(BASE + bad)
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "gpu",
                        "--cuvs-merge", "device"])


def test_presets_and_knobs():
    spec = systems.SYSTEMS["cuvs"]
    assert spec.presets["per_file_host"] == {"cuvs_merge": "host"}
    assert "cuvs_batch_rows" not in spec.knobs
    assert not hasattr(systems, "CUVS_BATCH_ROWS_DEFAULT")
    assert not hasattr(systems, "RowsOrFile")
    with pytest.raises(systems.SystemError_):
        systems.resolve("x", {"system": "cuvs", "cuvs_batch_rows": 500000})
    for p, kn in spec.presets.items():   # every other preset runs the recipe by default
        if p != "per_file_host":
            assert "cuvs_merge" not in kn
    for gone in ("parse_cuvs_batch_rows", "_bool_to_bits", "_bits_to_bool", "_Batch"):
        assert not hasattr(cuvs_engine, gone), gone
    for gone in ("_batch_ids", "_batch_take", "flush_pass", "regroup", "batch_rows"):
        assert not hasattr(cuvs_engine.CalcTruthMixin, gone), gone


# ------------------------------------------------------------- the memory guard

def _guard_args(**kw):
    a = dict(k=K, modality="dense", filter_config=None, h2d_overlap=False, widen="device",
             backend="cuvs", cuvs_rmm_pool="off")
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_device_merge_bytes_per_cell_arithmetic():
    """The GPU memory guard: the itemised per (query x 2k) cell bytes add up to 118, rounded up to 140."""
    items = [12 / 2, 12 / 2, 16 / 2, 12, 4, 4, 16, 48, 6 + 8]
    assert sum(items) == 118
    assert guards.DEVICE_MERGE_BYTES_PER_2K_CELL == 140 >= sum(items)


def test_device_merge_need_merge_and_widen(fake_cuvs, corpus, tmp_path):
    """merge = nq x 2k x 140; an fp32 corpus has no widen copy; an fp16-stored corpus widened
    on the device adds the largest file's rows x d x 4; --widen host none."""
    from test_run import _fp16_pair
    path, _ = corpus
    src = ParquetSource(path, prefetch=0, workers=1)
    feet = src.read_footers()
    need, parts = guards.cuvs_device_merge_need(_guard_args(), src, feet, 40, np.float32)
    assert parts == {"merge": 40 * 2 * K * 140} and need == 40 * 2 * K * 140
    f16, _, _ = _fp16_pair(tmp_path)
    s16 = ParquetSource(f16, prefetch=0, workers=1)
    f = s16.read_footers()
    big = max(f.values(), key=lambda x: x.num_rows)
    assert big.value_bits == 16
    need, parts = guards.cuvs_device_merge_need(_guard_args(), s16, f, 8, np.float32)
    assert parts["widen"] == big.num_rows * big.dim * 4
    assert need == 8 * 2 * K * 140 + big.num_rows * big.dim * 4
    _, parts = guards.cuvs_device_merge_need(_guard_args(widen="host"), s16, f, 8, np.float32)
    assert "widen" not in parts
    # an fp16 compute dtype is not wider than an fp16-stored block: no widen
    _, parts = guards.cuvs_device_merge_need(_guard_args(), s16, f, 8, np.float16)
    assert "widen" not in parts


def test_device_merge_guard_counts_the_rmm_pool_as_headroom(fake_cuvs, corpus, monkeypatch):
    """The merge (CuPy, rmm_cupy_allocator) may use the reserved RMM pool; the block(s) and
    the widen (torch) must fit in the free memory. Exact at the boundary."""
    path, _ = corpus
    b = _backend(fake_cuvs)
    b._cuda = True
    src = ParquetSource(path, prefetch=0, workers=1)
    nq, merge = 4096, 4096 * 2 * K * 140
    block = src.max_block_bytes()
    torch_need = block + guards.OVERLAP_GPU_MARGIN
    free = {"v": 0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (free["v"], 24 << 30))
    off = _guard_args()
    free["v"] = torch_need + merge
    guards.cuvs_device_merge_guard(off, b, src, nq, np.float32)          # fits exactly
    free["v"] -= 1
    with pytest.raises(SystemExit, match="--cuvs-merge host"):
        guards.cuvs_device_merge_guard(off, b, src, nq, np.float32)
    # pool on: the pool covers the merge, only the torch side must be free
    on = _guard_args(cuvs_rmm_pool="1GiB")
    b.rmm_pool_bytes = merge
    free["v"] = torch_need
    guards.cuvs_device_merge_guard(on, b, src, nq, np.float32)
    free["v"] = torch_need - 1
    with pytest.raises(SystemExit, match="RMM pool"):
        guards.cuvs_device_merge_guard(on, b, src, nq, np.float32)
    # a pool smaller than the merge: the rest of the merge comes from the free memory
    b.rmm_pool_bytes = merge // 2
    free["v"] = torch_need + merge - merge // 2
    guards.cuvs_device_merge_guard(on, b, src, nq, np.float32)
    free["v"] -= 1
    with pytest.raises(SystemExit):
        guards.cuvs_device_merge_guard(on, b, src, nq, np.float32)
    # the pool is not headroom when it is off, whatever the backend records
    b.rmm_pool_bytes = merge
    free["v"] = torch_need
    with pytest.raises(SystemExit):
        guards.cuvs_device_merge_guard(off, b, src, nq, np.float32)
    # two blocks with --h2d-overlap
    free["v"] = torch_need + merge
    with pytest.raises(SystemExit, match="2 block"):
        guards.cuvs_device_merge_guard(_guard_args(h2d_overlap=True), b, src, nq, np.float32)


# ------------------------------------------------------------- attribution

def test_attribution_device_merge_part_and_views_add_up():
    """A cuVS row run with its own recipe: the GPU view has the device merge as its own part
    (out of device_other), both views still sum to the LegClock time, the host merge bucket is
    labelled as the one copy back; FAISS and the per_file_host row have no device merge."""
    from bench import leg as L
    row = {"total_s": 100.0, "source_stall_s": 60.0, "engine_call_s": 38.5, "merge_s": 0.4,
           "other_s": 1.1, "h2d_s": 0.2, "h2d_exposed_s": 0.1, "widen_s": 0.0,
           "normalize_s": None, "search_s": 38.2, "lib_s": 37.0, "ids_s": 0.05,
           "device_merge_s": 1.0, "d2h_s": 0.3, "engine_gap_s": 0.05, "gpu_idle_s": 61.8,
           "build_s": 0.0}
    a = L.attribution_harness(row, 100.0, 0.0, "cuvs", "on", {"cuvs_merge": "device:file"})
    h, g = a["host_view"], a["gpu_view"]
    assert h["additive"] and h["labels"]["merge_s"] == L.DEVICE_MERGE_HOST_LABEL
    assert g["parts"]["device_merge_s"] == 1.0
    assert g["parts"]["device_other_s"] == pytest.approx(38.2 - 37.0 - 0.05 - 1.0)
    assert g["sum_s"] == pytest.approx(100.0) and g["additive"]
    assert g["labels"]["device_merge_s"] == "device merge (cuVS recipe)"
    assert a["busy"]["device_merge_s"] == 1.0
    for modes in ({"cuvs_merge": "host"}, {"cuvs_merge": "n/a"}):
        b = L.attribution_harness(dict(row, device_merge_s=None), 100.0, 0.0, "cuvs", "on", modes)
        assert b["gpu_view"]["parts"]["device_merge_s"] is None
        assert b["host_view"]["labels"]["merge_s"] == L.HOST_LABELS["merge_s"]


def test_run_pass_buckets_tile_the_pass(fake_cuvs, corpus, queries):
    """run_pass's host buckets still sum to total_s with the device merge (the one copy back
    in merge); the trace's per-file engine_call sums to it."""
    path, _ = corpus
    b = _backend(fake_cuvs)
    b.setup(queries)
    D, I, info = loop.run_pass(InMemorySource.load(path), b, len(queries), K)
    parts = info["source_stall_s"] + info["engine_call_s"] + info["merge_s"] + info["other_s"]
    assert parts == pytest.approx(info["total_s"], abs=1e-9) and info["other_s"] > -1e-3
    _, files = info["trace"]
    assert sum(f[3] for f in files) == pytest.approx(info["engine_call_s"])
    assert all(f[4] < 1e-3 for f in files)   # no per-file host merge (an empty call)


# ------------------------------------------------------------- reference rules


