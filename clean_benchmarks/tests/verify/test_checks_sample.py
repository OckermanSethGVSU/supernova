"""The exhaustive check's query sample (checks.queries) and its torch / GPU reference scan
(BENCH_CHECK_DEVICE), bench/verify/check.py + exact.py.

Synthetic tests are fast; the ones on the `bench run --dev` fixture runs (helpers.check_runs)
are slow. CUDA tests skip without a GPU (torch.cuda.is_available, as test_sparse_gpu.py)."""

import os

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from bench.verify import check as V
from bench.verify import exact as X
from bench.verify import oracle as O
from helpers.check_runs import runs  # noqa: F401 (fixture)


# ---------------------------------------------------------------------------------------------
# the sample
# ---------------------------------------------------------------------------------------------
def test_the_sample_is_seeded_sorted_and_depends_only_on_n_and_the_setting():
    a = V.sample_queries(300, 50)
    assert np.array_equal(a, sorted(np.random.default_rng(0).choice(300, 50, replace=False)))
    assert np.array_equal(a, V.sample_queries(300, 50)) and len(set(a.tolist())) == 50
    assert V.sample_queries(300, "all") is None
    assert V.sample_queries(300, 300) is None and V.sample_queries(300, 1000) is None  # every query


# ---------------------------------------------------------------------------------------------
# the torch scan (BENCH_CHECK_DEVICE=torch-cpu here; cuda on a GPU node)
# ---------------------------------------------------------------------------------------------
def _files(tmp_path, xs, dtype=np.float32):
    paths = []
    for i, x in enumerate(xs):
        p = str(tmp_path / f"f{i}.parquet")
        pq.write_table(pa.table({"v": pa.array(list(np.asarray(x, dtype)), pa.list_(pa.from_numpy_dtype(dtype)))}),
                       p, row_group_size=13)
        paths.append(p)
    off = np.concatenate([[0], np.cumsum([len(x) for x in xs])])
    return X.Corpus(pafs.LocalFileSystem(), paths, off)


def _ref(monkeypatch, device, scorer, corpus, K, flt=None):
    monkeypatch.setenv(X.DEVICE_ENV, device)
    r = X.build_reference(scorer, corpus, K, flt)
    assert r.pop("device") == device
    return r


def _same(a, b, exact=False):
    for k in ("I", "PI"):
        assert np.array_equal(a[k], b[k]), k
    for k in ("D", "P", "N"):
        if exact:
            assert np.array_equal(a[k], b[k], equal_nan=True), k
        else:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12, atol=1e-12, err_msg=k)


def _close(a, b, scorer, corpus, tol=1e-12):
    """The torch reference ``b`` against the numpy one ``a``: every value within tol (absolute,
    times the largest finite |value|, at least 1), the same empty slots, and the ids equal
    except where the numpy float64 values of the differing ids are within tol of each other:
    rows tied to float64 precision, which the two GEMMs may order differently. The numpy values
    of the differing ids come from score_pairs (exact score, and score - bound for P). The
    torch path computes the norms on its device too (its own sum of squares), so N is within
    tol (relative) wherever the ids are equal, and NaN in the same places."""
    for v, i in (("D", "I"), ("P", "PI")):
        A, B = a[v], b[v]
        assert np.array_equal(a[i] < 0, b[i] < 0), i
        f = np.isfinite(A)
        assert np.array_equal(f, np.isfinite(B)), v
        t = tol * max(1.0, float(np.abs(A[f]).max(initial=0.0)))
        assert np.abs(A[f] - B[f]).max(initial=0.0) <= t, v
        q, j = np.nonzero(a[i] != b[i])

        def value(ids):
            e, xn = X.score_pairs(scorer, corpus, q, ids)
            return e if v == "D" else e - scorer.bound(q, xn)
        if len(q):
            assert np.all(np.abs(value(a[i][q, j]) - value(b[i][q, j])) <= t), (v, q, j)
    same = a["I"] == b["I"]
    np.testing.assert_allclose(a["N"][same], b["N"][same], rtol=tol, atol=0, equal_nan=True)


@pytest.fixture
def small_tiles(monkeypatch):
    """Row tiles of 7 rows and torch tiles of 30 scores: many tiles, query chunks and merges."""
    monkeypatch.setattr(O, "R_TILE", 7)
    monkeypatch.setattr(X, "TORCH_TILE_CELLS", 30)
    monkeypatch.setattr(X, "STREAM_BATCH_BYTES", 200)


@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_the_torch_scan_equals_the_numpy_scan(tmp_path, monkeypatch, small_tiles, metric, dtype):
    """float32 and float16 stored rows (the torch path copies the stored dtype to the device and
    widens, takes norms and normalises there): within 1e-12 of numpy."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    xs = [rng.standard_normal((n, 16)) * rng.uniform(0.5, 2, (n, 1)) for n in (40, 1, 0, 33)]
    xs[0][5] = 0.0                    # a zero row
    xs[3][7, 2] = np.nan              # a non-finite row: never a candidate
    xs[3][9, 4] = np.inf              # an inf row: never a candidate either
    corpus = _files(tmp_path, xs, dtype)
    sc = X.Dense(rng.standard_normal((9, 16)).astype(np.float32), metric, "v")
    a, b = _ref(monkeypatch, "cpu", sc, corpus, 12), _ref(monkeypatch, "torch-cpu", sc, corpus, 12)
    _close(a, b, sc, corpus)
    assert not np.isin([40 + 1 + 7, 40 + 1 + 9], a["I"]).any() and (a["I"] >= 0).all()
    assert not np.isin([40 + 1 + 7, 40 + 1 + 9], b["I"]).any() and (b["I"] >= 0).all()


def test_ties_at_the_cutoff_go_to_the_lowest_id_on_torch(tmp_path, monkeypatch, small_tiles):
    """Small integer values: every float64 score is exact on any device, so ids and scores must
    agree exactly. Exact duplicate rows straddle the cutoff, within and across tiles and files."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    base = rng.integers(-3, 4, (6, 8))
    xs = [base[rng.integers(0, 6, 25)], base[rng.integers(0, 6, 1)], base[rng.integers(0, 6, 30)]]
    corpus = _files(tmp_path, xs)
    sc = X.Dense(rng.integers(-2, 3, (7, 8)).astype(np.float32), "ip", "v")
    a, b = _ref(monkeypatch, "cpu", sc, corpus, 10), _ref(monkeypatch, "torch-cpu", sc, corpus, 10)
    _same(a, b, exact=True)
    # every query has ties at the cutoff here (6 distinct rows, 56 rows); among equal scores the
    # ids rise, and none left out beats or ties-with-a-lower-id a kept one
    allx = np.concatenate(xs).astype(np.float64)
    s = sc.q @ allx.T
    for q in range(sc.n):
        want = np.lexsort((np.arange(len(allx)), -s[q]))[:10]
        assert np.array_equal(b["I"][q], want)


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_duplicate_float_rows_give_a_deterministic_torch_reference(tmp_path, monkeypatch, small_tiles,
                                                                   metric):
    """Random float rows, each repeated at scattered positions (tie groups at the cutoff, within
    and across tiles and files): the torch reference is the same twice, within 1e-12 of numpy
    (ids equal up to float64-precision ties), and in each list equal values come lowest id first."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    base = rng.standard_normal((8, 24)).astype(np.float32)
    rows = base[rng.integers(0, 8, 70)]
    corpus = _files(tmp_path, [rows[:33], rows[33:]])
    sc = X.Dense(rng.standard_normal((6, 24)).astype(np.float32), metric, "v")
    a = _ref(monkeypatch, "cpu", sc, corpus, 15)
    b, b2 = (_ref(monkeypatch, "torch-cpu", sc, corpus, 15) for _ in range(2))
    _same(b, b2, exact=True)
    _close(a, b, sc, corpus)
    for v, i in (("D", "I"), ("P", "PI")):
        eq = (b[v][:, 1:] == b[v][:, :-1]) & np.isfinite(b[v][:, 1:])
        assert (b[i][:, 1:][eq] > b[i][:, :-1][eq]).all()
        assert eq.any()         # the test has ties to break


def test_the_torch_scan_applies_the_filter_like_numpy(tmp_path, monkeypatch, small_tiles):
    """allowed(q0, q1, r0, r1) masks (query, row) pairs out on both paths alike."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    xs = [rng.standard_normal((31, 8)), rng.standard_normal((18, 8))]
    corpus = _files(tmp_path, xs)
    sc = X.Dense(rng.standard_normal((5, 8)).astype(np.float32), "cosine", "v")
    keep = rng.random((5, 49)) < 0.4

    class F:   # a per-query filter over global rows
        def tile(self, f):
            o = int(corpus.offsets[f])
            return lambda q0, q1, r0, r1: keep[q0:q1, o + r0:o + r1]
    a, b = _ref(monkeypatch, "cpu", sc, corpus, 8, F()), _ref(monkeypatch, "torch-cpu", sc, corpus, 8, F())
    _close(a, b, sc, corpus)
    q, j = np.nonzero(a["I"] >= 0)
    assert keep[q, a["I"][q, j]].all()


def test_the_device_merge_is_topk_rows_with_ties_to_the_lowest_id():
    """_device_merge of a tile into running lists = O._merge_topk of the running lists and
    O._topk_rows of the tile (best first, ties to the lowest id, NaN / -inf never a row, empty
    slots (-inf, -1)), for a K below, at and above the tile's width and over several tiles."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    for K in (1, 4, 10, 40, 50):
        rv, ri = np.full((6, K), -np.inf), np.full((6, K), -1, np.int64)
        tv, ti = torch.from_numpy(rv.copy()), torch.from_numpy(ri.copy())
        for r0 in (0, 40, 80):
            v = rng.integers(0, 5, (6, 40)).astype(np.float64)    # many exact ties
            v[0] = -np.inf
            v[1, :35] = -np.inf
            v[2, 3] = np.nan
            dd, ii = O._topk_rows(v, K)
            rv, ri = O._merge_topk(rv, ri, dd, np.where(ii >= 0, ii + r0, -1), K)
            tv, ti = X._device_merge(torch, tv, ti, torch.from_numpy(v), r0)
            assert np.array_equal(rv, tv.numpy()) and np.array_equal(ri, ti.numpy()), (K, r0)


def test_the_device_setting(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(X.DEVICE_ENV, "cpu")
    assert X.scan_device() == ("cpu", None)
    monkeypatch.setenv(X.DEVICE_ENV, "torch-cpu")
    assert X.scan_device()[0] == "torch-cpu"
    monkeypatch.setenv(X.DEVICE_ENV, "gpu")
    with pytest.raises(ValueError, match="auto | cpu"):
        X.scan_device()
    monkeypatch.setenv(X.DEVICE_ENV, "auto")
    assert X.scan_device()[0] == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        monkeypatch.setenv(X.DEVICE_ENV, "cuda")
        with pytest.raises(ValueError, match="no CUDA device"):
            X.scan_device()


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_the_cuda_scan_equals_the_numpy_scan(tmp_path, monkeypatch, metric):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(11)
    xs = [rng.standard_normal((n, 96)) for n in (5000, 3, 2600)]
    corpus = _files(tmp_path, xs)
    sc = X.Dense(rng.standard_normal((700, 96)).astype(np.float32), metric, "v")
    a, b = _ref(monkeypatch, "cpu", sc, corpus, 42), _ref(monkeypatch, "cuda", sc, corpus, 42)
    _close(a, b, sc, corpus)
    _same(b, _ref(monkeypatch, "cuda", sc, corpus, 42), exact=True)     # deterministic
    base = rng.integers(-3, 4, (6, 16))           # exact ties on the GPU too
    os.makedirs(tmp_path / "t")
    corpus = _files(tmp_path / "t", [base[rng.integers(0, 6, 900)], base[rng.integers(0, 6, 400)]])
    sc = X.Dense(rng.integers(-2, 3, (50, 16)).astype(np.float32), "ip", "v")
    _same(_ref(monkeypatch, "cpu", sc, corpus, 42), _ref(monkeypatch, "cuda", sc, corpus, 42), exact=True)


# ---------------------------------------------------------------------------------------------
# on the fixture runs (slow)
# ---------------------------------------------------------------------------------------------
def _ctx(d, search):
    from test_checks import ctx    # skips without torch / nova_bf, as test_checks does
    return ctx(d, search)


@pytest.mark.slow
@pytest.mark.parametrize("shape,search", [("dense", "dense"), ("filtered", "filtered_text"),
                                          ("filtered", "static_struct"), ("sparse", None),
                                          ("multivector", None)])
def test_a_subset_scorer_gives_the_full_references_rows(runs, shape, search, monkeypatch):
    """build_reference over the sampled queries = the full reference's rows of those queries
    (scorer.subset, Filter.subset); and the torch path equals numpy on the fixtures."""
    import json
    d = runs[shape]
    exp = json.load(open(os.path.join(d, "experiment.json")))
    search = search or exp["searches"][0]["name"]
    exp, s, a, inputs, scorer, corpus, flt = _ctx(d, search)
    K = int(exp["search"]["k"]) + X.SPARE
    qs = V.sample_queries(scorer.n, 7)
    monkeypatch.setenv(X.DEVICE_ENV, "cpu")
    full = X.build_reference(scorer, corpus, K, flt)
    sub = X.build_reference(scorer.subset(qs), corpus, K, flt and flt.subset(qs))
    _same({k: full[k][qs] for k in ("D", "I", "N", "P", "PI")}, sub)
    if shape in ("dense", "filtered"):
        monkeypatch.setenv(X.DEVICE_ENV, "torch-cpu")
        tf = X.build_reference(scorer, corpus, K, flt)
        assert tf.pop("device") == "torch-cpu"
        _close(full, tf, scorer, corpus)


def _verdicts(doc):
    """What a check run decides: per search and system, ok, the failed parts and the failures."""
    return doc["ok"], {s: {nm: (v["ok"], v.get("failed_parts"), v.get("fails"))
                           for nm, v in x["systems"].items()} for s, x in doc["searches"].items()}


@pytest.mark.slow
def test_every_verdict_is_the_same_with_the_torch_reference(runs, tmp_path, monkeypatch):
    """check.run (exhaustive, every query) on the dense and filtered fixture runs, clean and with
    a clearly worse row planted: BENCH_CHECK_DEVICE=cpu and torch-cpu decide alike."""
    from helpers.check_runs import _npz_rewrite, copy_run
    from test_checks import reference, verdict
    d = copy_run(runs["dense"], tmp_path / "worse")
    k, D, I, scorer, *_ = reference(d, "dense")
    q = next(q for q in range(len(D)) if D[q, k - 1] - D[q, -1] > 4 * scorer.bound(0, 1.0))

    def fn(II, DD):
        II[q, 0], DD[q, 0] = I[q, -1], np.float32(D[q, -1])
        return II, DD
    _npz_rewrite(d, "dense", "cpu", fn)
    for name, run in (("dense", runs["dense"]), ("filtered", runs["filtered"]), ("worse", d)):
        got = {}
        for dev in ("cpu", "torch-cpu"):
            monkeypatch.setenv(X.DEVICE_ENV, dev)
            doc = verdict(run, True, tmp_path / f"c-{name}-{dev}", queries="all")
            assert {x["reference"]["device"] for x in doc["searches"].values()} == {dev}
            got[dev] = _verdicts(doc)
        assert got["cpu"] == got["torch-cpu"], name
        assert got["cpu"][0] == (name != "worse"), name


def _plant_common_miss(d, q, best, worse, s_worse):
    """Both systems (cpu and nova-bf) return ``worse`` in place of ``best`` for query q."""
    from helpers.check_runs import _nb_rewrite, _npz_rewrite
    from test_checks import nb_uuid

    def fn(I, D):
        j = list(I[q]).index(best)
        I[q, j], D[q, j] = worse, np.float32(s_worse)
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    bu, wu = nb_uuid(d, best), nb_uuid(d, worse)

    def nb(ids, sc):
        j = ids[q].index(bu)
        ids[q][j], sc[q][j] = wu, s_worse
        return ids, sc
    _nb_rewrite(d, "dense", nb)


def _clear_queries(d, n_sample):
    """Two queries with a row clearly below the k-th: one in the sample of n_sample, one not."""
    from test_checks import reference
    k, D, I, scorer, *_ = reference(d, "dense")
    qs = set(V.sample_queries(len(D), n_sample).tolist())
    ok = [q for q in range(len(D)) if D[q, k - 1] - D[q, -1] > 4 * scorer.bound(0, 1.0)]
    q_in = next(q for q in ok if q in qs)
    q_out = next(q for q in ok if q not in qs)
    row = lambda q: (q, int(I[q, 0]), int(I[q, -1]), float(D[q, -1]))  # noqa: E731
    return row(q_in), row(q_out)


@pytest.mark.slow
def test_a_common_miss_is_seen_on_a_sampled_query_only(runs, tmp_path):
    from helpers.check_runs import copy_run
    from test_checks import sysok, verdict
    d0 = runs["dense"]
    q_in, q_out = _clear_queries(d0, 30)
    d = copy_run(d0, tmp_path / "in")
    _plant_common_miss(d, *q_in)
    doc = verdict(d, True, tmp_path / "c", queries=30)
    s = doc["searches"]["dense"]
    assert s["sample"] == {"queries": 30, "of": 300, "seed": 0, "setting": 30}
    for nm in ("cpu", "novabf"):
        v = s["systems"][nm]
        assert not v["ok"] and v["failed_parts"] == ["exhaustive"], v["fails"]
        assert all(f.startswith("exhaustive (30 of 300 queries): ") for f in v["fails"])
        assert any(x["query"] == q_in[0] for x in v["examples"])    # the search's query number
    d = copy_run(d0, tmp_path / "out")
    _plant_common_miss(d, *q_out)
    doc = verdict(d, True, tmp_path / "c", queries=30)              # the documented blind spot
    assert sysok(doc, "dense", "cpu") and sysok(doc, "dense", "novabf")
    assert "outside the 30-query sample" in doc["searches"]["dense"]["blind_spot"]
    assert doc["searches"]["dense"]["reference"]["cached"] is True  # same sample, other results


@pytest.mark.slow
def test_a_disagreement_on_a_query_outside_the_sample_is_caught_by_the_disputed_rule(runs, tmp_path):
    from helpers.check_runs import _npz_rewrite, copy_run
    from test_checks import sysok, verdict
    d = copy_run(runs["dense"], tmp_path / "run")
    _, (q, best, worse, s_worse) = _clear_queries(d, 30)

    def fn(I, D):
        I[q, 0], D[q, 0] = worse, np.float32(s_worse)
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    doc = verdict(d, True, tmp_path / "c", queries=30)
    v = doc["searches"]["dense"]["systems"]["cpu"]
    assert not v["ok"] and v["failed_parts"] == ["disputed"], v["fails"]
    assert v["disputed"]["fails"] and v["fails"][0].startswith("disputed rows (every query): ")
    assert sysok(doc, "dense", "novabf")


@pytest.mark.slow
def test_all_is_the_unsampled_check_and_the_cache_is_reused(runs, tmp_path):
    """"all" and a size >= n are one sample (every query, one cache key); the reference is
    today's full build bit for bit; a second run with other results reuses it."""
    from helpers.check_runs import _npz_rewrite, copy_run
    from test_checks import verdict
    d0 = runs["dense"]
    a = verdict(d0, True, tmp_path / "c", queries="all")
    b = verdict(d0, True, tmp_path / "c", queries=1000)
    ra, rb = a["searches"]["dense"]["reference"], b["searches"]["dense"]["reference"]
    assert ra["key"] == rb["key"] and not ra["cached"] and rb["cached"]
    assert a["searches"]["dense"]["blind_spot"] is None
    assert a["searches"]["dense"]["sample"]["queries"] == 300
    exp, s, aa, inputs, scorer, corpus, flt = _ctx(d0, "dense")
    z = np.load(os.path.join(tmp_path / "c", ra["key"] + ".npz"))
    full = X.build_reference(scorer, corpus, int(exp["search"]["k"]) + X.SPARE, flt)
    _same({k: z[k] for k in ("D", "I", "N", "P", "PI")}, full, exact=True)
    d = copy_run(d0, tmp_path / "run")
    _npz_rewrite(d, "dense", "cpu", lambda I, D: (I[:, ::-1].copy(), D[:, ::-1].copy()))
    c = verdict(d, True, tmp_path / "c", queries=1000)
    assert c["searches"]["dense"]["reference"]["cached"] and not c["searches"]["dense"]["systems"]["cpu"]["ok"]


@pytest.mark.slow
def test_expect_inexact_takes_fnmatch_patterns(runs, tmp_path):
    from helpers.check_runs import _npz_rewrite, copy_run
    from test_checks import verdict
    d = copy_run(runs["dense"], tmp_path / "run")
    _, (q, best, worse, s_worse) = _clear_queries(d, 30)

    def fn(I, D):
        I[q, 0], D[q, 0] = worse, np.float32(s_worse)
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    for pat, ok in (("c*", True), ("*p?", True), ("cpu-*", False), ("novabf", False)):
        doc = verdict(d, False, tmp_path / "c", expect_inexact=[pat])
        v = doc["searches"]["dense"]["systems"]["cpu"]
        assert not v["ok"] and bool(v.get("expect_inexact")) is ok and (not doc["problems"]) is ok, pat
    # the PyLate sweep's pattern: the zero-padded rerank mode only, never ownpad (exact)
    pat = {"checks": {"expect_inexact": ["*-pylate_mode-rerank-released-pylate_rerank_results-*"]}}
    assert V.expect_inexact(pat, "pylate_scores-batched-pylate_mode-rerank-released-pylate_rerank_results-host")
    assert not V.expect_inexact(pat, "pylate_scores-batched-pylate_mode-rerank-released-ownpad-pylate_rerank_results-host")
    assert not V.expect_inexact(pat, "pylate_scores-batched-pylate_mode-scores-batched")
    assert not V.expect_inexact({"checks": {}}, "x")
