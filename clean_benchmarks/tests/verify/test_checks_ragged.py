"""bench/verify/exact.py's Sparse and MultiVector scorers and the rules of check.py on them.

Three kinds of test: the streamed float64 reference and pair scores against an independent brute
force (math.fsum per pair) on scripted files; honest float32 engines (shuffled summation order,
float32 normalisation, random tie breaks, near-ties planted) that must pass both checks for every
seed, the soundness test of the float32 bounds; and planted mistakes in copies of the sparse and
multivector `bench run --dev` fixture runs (helpers.fixture_runs), each failing the system that
made it.
"""

import json
import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from bench.data.formats import multivector as MV
from bench.data.formats import sparse as SP
from bench.verify import check as V
from bench.verify import exact as X
from bench.verify import load as L
from bench.verify import oracle as O


# ---------------------------------------------------------------------------------------------
# scripted files
# ---------------------------------------------------------------------------------------------
def _sp_array(rows):
    """rows: [(indices, values)] -> a sparse struct array (uint32 indices, float32 values)."""
    return pa.StructArray.from_arrays(
        [pa.array([list(map(int, i)) for i, _ in rows], pa.list_(pa.uint32())),
         pa.array([[float(v) for v in x] for _, x in rows], pa.list_(pa.float32()))],
        ["indices", "values"])


def _mv_array(docs):
    """docs: [(n_tok, d) array, [] or None] -> list<list<float32>> (None: a null doc)."""
    return pa.array([None if x is None else [list(map(float, t)) for t in x] for x in docs],
                    pa.list_(pa.list_(pa.float32())))


def _corpus(tmp_path, arrays, rg=7):
    paths, off = [], [0]
    (tmp_path / "c").mkdir(exist_ok=True)
    for i, a in enumerate(arrays):
        p = str(tmp_path / "c" / f"p{i}.parquet")
        pq.write_table(pa.table({"v": a}), p, row_group_size=rg)
        paths.append(p)
        off.append(off[-1] + len(a))
    return X.Corpus(pafs.LocalFileSystem(), paths, off)


def _queries(tmp_path, arr, mod):
    p = str(tmp_path / "q.parquet")
    pq.write_table(pa.table({"v": arr}), p)
    return mod.load_queries(p, "v")[0]


def _split(rows, sizes):
    out, a = [], 0
    for s in sizes:
        out.append(rows[a:a + s])
        a += s
    return out


def _sparse_rows(rng, n, vocab, nnz_max, dup=True):
    rows = []
    for _ in range(n):
        m = int(rng.integers(0, nnz_max + 1))
        idx = rng.choice(vocab, m, replace=False)
        if dup and m and rng.random() < 0.3:                 # a duplicate index
            idx = np.concatenate([idx, idx[:1]])
        rows.append((idx, rng.random(len(idx)).astype(np.float32) + 0.01))
    return rows


def _mv_docs(rng, n, d, tok_max, scale=1.0):
    docs = []
    for _ in range(n):
        t = int(rng.integers(0, tok_max + 1))
        docs.append(None if t == 0 and rng.random() < 0.5 else
                    (rng.standard_normal((t, d)) * scale).astype(np.float32))
    return docs


@pytest.fixture
def small_tiles(monkeypatch):
    """Tiny batches, tiles and query chunks: never aligned with files or row groups."""
    monkeypatch.setattr(X, "STREAM_BATCH_BYTES", 150)
    monkeypatch.setattr(X, "SCAN_CELLS", 37)
    monkeypatch.setattr(X, "MV_TILE_CELLS", 50)
    monkeypatch.setattr(X, "MV_PAIR_TOKENS", 5)
    monkeypatch.setattr(X, "SPARSE_PAIRS", 11)
    monkeypatch.setattr(O, "Q_CHUNK", 4)


# ---------------------------------------------------------------------------------------------
# the independent brute force (python per pair, math.fsum: no scipy, no shared code)
# ---------------------------------------------------------------------------------------------
def _sp_dict(idx, val):
    d = {}
    for i, v in zip(idx, val):
        d[int(i)] = d.get(int(i), 0.0) + float(v)
    return d


def bf_sparse(qrows, rows, metric):
    """(nq, rows) float64 scores; a row with a non-finite value -inf. A query's duplicate
    indices are summed and rounded to float32 (the harness's canonical queries); a row's stay
    float64."""
    Q = [{i: float(np.float32(v)) for i, v in _sp_dict(i, v).items()} for i, v in qrows]
    R = [_sp_dict(i, v) for i, v in rows]
    nrm = lambda d: math.sqrt(math.fsum(v * v for v in d.values()))  # noqa: E731
    S = np.zeros((len(Q), len(R)))
    for a, q in enumerate(Q):
        for b, x in enumerate(R):
            if not all(np.isfinite(v) for v in x.values()):
                S[a, b] = -np.inf
                continue
            s = math.fsum(q[i] * x[i] for i in q if i in x)
            if metric == "cosine":
                s = s / nrm(q) if nrm(q) > 0 else s
                s = s / nrm(x) if nrm(x) > 0 else s
            S[a, b] = s
    return S


def _unit(t):
    t = np.asarray(t, np.float64)
    n = np.sqrt((t * t).sum(axis=1, keepdims=True))
    return np.where(n > 0, t / np.where(n > 0, n, 1), 0.0)


def bf_mv(qtoks, docs, metric):
    S = np.full((len(qtoks), len(docs)), -np.inf)
    for a, q in enumerate(qtoks):
        for b, x in enumerate(docs):
            if x is None or not len(x) or not len(q) or not np.isfinite(x).all():
                continue
            qq, xx = (_unit(q), _unit(x)) if metric == "cosine" else \
                (np.asarray(q, np.float64), np.asarray(x, np.float64))
            S[a, b] = math.fsum(max(math.fsum(float(u) * float(w) for u, w in zip(qt, xt)) for xt in xx)
                                for qt in qq)
    return S


def bf_topk(S, K):
    D = np.full((len(S), K), -np.inf)
    I = np.full((len(S), K), -1, np.int64)
    for q in range(len(S)):
        o = np.lexsort((np.arange(S.shape[1]), -S[q]))[:K]
        o = o[np.isfinite(S[q, o])]
        D[q, :len(o)], I[q, :len(o)] = S[q, o], o
    return D, I


# ---------------------------------------------------------------------------------------------
# scripted shapes: (scorer, corpus, brute-force scores)
# ---------------------------------------------------------------------------------------------
def sparse_case(tmp_path, metric, seed=0, n=150, sizes=(40, 0, 61, 49), vocab=40, nq=13):
    rng = np.random.default_rng(seed)
    rows = _sparse_rows(rng, n, vocab, 9)
    rows[5] = (np.array([], np.int64), np.array([], np.float32))       # an empty row
    rows[70] = (np.array([3, 7]), np.array([np.nan, 0.5], np.float32))  # a NaN row
    qrows = _sparse_rows(rng, nq, vocab, 6)
    qrows[2] = (np.array([], np.int64), np.array([], np.float32))      # an empty query
    qrows[3] = (np.array([vocab + 5]), np.array([1.0], np.float32))    # shares no term with anything
    corpus = _corpus(tmp_path, [_sp_array(r) for r in _split(rows, sizes)])
    q = _queries(tmp_path, _sp_array(qrows), SP)
    return X.Sparse(q, metric, "v"), corpus, bf_sparse(qrows, rows, metric), (qrows, rows)


def mv_case(tmp_path, metric, seed=0, n=90, sizes=(30, 0, 33, 27), d=5, nq=9):
    rng = np.random.default_rng(seed)
    docs = _mv_docs(rng, n, d, 5)
    docs[4], docs[11] = None, np.zeros((0, d), np.float32)           # zero-token docs
    docs[20] = np.array([[np.nan] * d], np.float32)                  # a NaN doc
    docs[30] = (docs[30] if docs[30] is not None and len(docs[30]) else np.ones((2, d), np.float32)) * 40
    qt = [rng.standard_normal((int(rng.integers(1, 5)), d)).astype(np.float32) for _ in range(nq)]
    qt[1] = np.zeros((0, d), np.float32)                             # a zero-token query
    corpus = _corpus(tmp_path, [_mv_array(r) for r in _split(docs, sizes)])
    q = _queries(tmp_path, _mv_array(qt), MV)
    return X.MultiVector(q, metric, "v"), corpus, bf_mv(qt, docs, metric), (qt, docs)


CASES = {"sparse": sparse_case, "multivector": mv_case}


@pytest.mark.parametrize("shape", ["sparse", "multivector"])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_the_streamed_reference_equals_a_brute_force(tmp_path, small_tiles, shape, metric):
    sc, corpus, S, _ = CASES[shape](tmp_path, metric)
    K = 12
    r = X.build_reference(sc, corpus, K)
    D, I = bf_topk(S, K)
    assert np.array_equal(r["I"], I)
    np.testing.assert_allclose(r["D"], D, rtol=1e-12, atol=1e-12)
    if shape == "sparse":   # floor 0: rows sharing no term are candidates (the empty query too)
        assert (r["D"][2] == 0).all() and (r["I"][2] == np.arange(K)).all()
        assert (r["D"][3] == 0).all()
    else:                   # a zero-token query has no candidate
        assert (r["I"][1] == -1).all()
        assert not np.isin(r["I"], [4, 11, 20]).any()


@pytest.mark.parametrize("shape", ["sparse", "multivector"])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_score_pairs_equals_a_brute_force(tmp_path, small_tiles, shape, metric):
    sc, corpus, S, _ = CASES[shape](tmp_path, metric)
    rng = np.random.default_rng(5)
    qi = rng.integers(0, S.shape[0], 500)
    gi = rng.integers(0, S.shape[1], 500)
    gi[:6] = [0, corpus.rows - 1, 5, 70 if shape == "sparse" else 20, 4, 11]
    e, aux = X.score_pairs(sc, corpus, qi, gi)
    want = S[qi, gi]
    assert np.array_equal(np.isfinite(e), np.isfinite(want))
    np.testing.assert_allclose(e[np.isfinite(e)], want[np.isfinite(want)], rtol=1e-12, atol=1e-12)
    assert np.isfinite(aux).all()
    # the reference's aux (N) is the same per-row value
    r = X.build_reference(sc, corpus, 12)
    sel = r["I"] >= 0
    _, a2 = X.score_pairs(sc, corpus, np.nonzero(sel)[0], r["I"][sel])
    np.testing.assert_array_equal(a2, r["N"][sel])


def test_the_bounds_are_the_documented_formulas(tmp_path):
    g = X.gamma
    sp_ip = sparse_case(tmp_path, "ip")[0]
    nq, qn = sp_ip.nq[0], sp_ip.qnorm[0]
    assert sp_ip.bound(0, 2.0) == pytest.approx(g(nq + 8) * qn * 2.0) and sp_ip.floor == 0.0
    sp_cos = X.Sparse.__new__(X.Sparse)
    sp_cos.__dict__.update(sp_ip.__dict__, unit=True)
    assert sp_cos.bound(0, 7) == pytest.approx(g(2 * nq + 7 + 8))
    (tmp_path / "m").mkdir()
    mv = mv_case(tmp_path / "m", "ip")[0]
    T, s = mv.T[0], mv.qsum[0]
    assert mv.bound(0, 3.0) == pytest.approx((g(2 * mv.d + 8) + g(T)) * s * 3.0) and mv.floor == -np.inf
    mv.unit = True
    assert mv.bound(0, 123.0) == pytest.approx((g(2 * mv.d + 8) + g(T)) * T)


# ---------------------------------------------------------------------------------------------
# honest float32 engines must pass (soundness of the bounds)
# ---------------------------------------------------------------------------------------------
def _f32_sum(vals, rng):
    """Sequential float32 sum in a random order (the worst-case growth of a plain loop)."""
    v = np.asarray(vals, np.float32)[rng.permutation(len(vals))]
    return np.cumsum(v, dtype=np.float32)[-1] if len(v) else np.float32(0)


def f32_sparse(qrows, rows, metric, rng):
    def dedup(idx, val):
        d = {}
        for i in rng.permutation(len(idx)):
            d[int(idx[i])] = np.float32(d.get(int(idx[i]), np.float32(0)) + np.float32(val[i]))
        return d
    Q = [dedup(*r) for r in qrows]
    R = [dedup(*r) for r in rows]
    nrm = lambda d: np.sqrt(_f32_sum([v * v for v in d.values()], rng))  # noqa: E731
    S = np.zeros((len(Q), len(R)), np.float32)
    for a, q in enumerate(Q):
        for b, x in enumerate(R):
            s = _f32_sum([np.float32(q[i] * x[i]) for i in q if i in x], rng)
            if metric == "cosine":
                nq_, nx_ = nrm(q), nrm(x)
                s = np.float32(s / nq_) if nq_ > 0 else s
                s = np.float32(s / nx_) if nx_ > 0 else s
            S[a, b] = s
    return S


def f32_mv(qtoks, docs, metric, rng):
    def unit32(t):
        t = np.asarray(t, np.float32)
        n = np.sqrt(np.array([_f32_sum(r * r, rng) for r in t], np.float32))
        return np.where(n[:, None] > 0, t / np.where(n > 0, n, 1)[:, None], 0).astype(np.float32)
    S = np.full((len(qtoks), len(docs)), -np.inf, np.float32)
    for a, q in enumerate(qtoks):
        for b, x in enumerate(docs):
            if x is None or not len(x) or not len(q) or not np.isfinite(x).all():
                continue
            qq, xx = (unit32(q), unit32(x)) if metric == "cosine" else (q, x)
            S[a, b] = _f32_sum((qq @ xx.T).max(axis=1), rng)
    return S


def honest_lists(S, k, rng, pad_zero=False):
    """Top-k by float32 score, ties broken at random; -inf rows (and with ``pad_zero`` rows
    scoring 0, as nova-bf's sparse candidate rule) are pads."""
    ids, sc = [], []
    for q in range(len(S)):
        o = np.lexsort((rng.random(S.shape[1]), -S[q].astype(np.float64)))[:k]
        keep = np.isfinite(S[q, o]) & ((S[q, o] != 0) if pad_zero else True)
        o = o[keep]
        ids.append(np.concatenate([o, -np.ones(k - len(o), np.int64)]))
        sc.append(np.concatenate([S[q, o], np.full(k - len(o), -np.inf, np.float32)]))
    return ids, sc


def judge(sc, corpus, k, systems):
    """{system: (exhaustive passed, default passed)} of the rules on scripted lists."""
    ref = X.build_reference(sc, corpus, k + X.SPARE)
    names = list(systems)
    mats = {nm: V.matrices(i, s, k, corpus.rows) for nm, (i, s) in systems.items()}
    ex = V.exhaustive_check(names, mats, k, sc, corpus, None, ref)
    dis, _, _ = V.disputed_check(names, mats, sc, corpus, None)
    r1 = {nm: V.defects(systems[nm][0], k, False)["failed"] for nm in names}
    return {nm: (not ex[nm]["fails"] and not r1[nm], not dis[nm]["fails"] and not r1[nm]) for nm in names}, \
        ex, dis


def _near_ties_sparse(rows, rng):
    """Duplicated rows, rows nudged by one float32 ULP, and large-norm rows."""
    out = list(rows)
    for j in rng.choice(len(rows), 12, replace=False):
        i, v = rows[j]
        if not len(i):
            continue
        out.append((i, v.copy()))
        w = v.copy()
        p = int(rng.integers(len(w)))
        w[p] = np.nextafter(w[p], np.float32(2))
        out.append((i, w))
        out.append((i, (v * 50).astype(np.float32)))
    return out


def _near_ties_mv(docs, rng):
    out = list(docs)
    for j in rng.choice(len(docs), 10, replace=False):
        x = docs[j]
        if x is None or not len(x):
            continue
        out.append(x.copy())
        out.append(x[::-1].copy())                         # same MaxSim exactly
        y = x.copy()
        y[0, 0] = np.nextafter(y[0, 0], np.float32(10))
        out.append(y)
        out.append((x * 30).astype(np.float32))
    return out


@pytest.mark.parametrize("shape", ["sparse", "multivector"])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_honest_float32_engines_pass_both_checks(tmp_path, shape, metric):
    """Many seeds: float32 arithmetic in a shuffled order, random tie breaks, near-ties planted.
    Never a failure; and the float32 lists must differ from the float64 reference somewhere
    (else the test would prove nothing)."""
    k, differ = 8, 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        sub = tmp_path / f"s{seed}"
        sub.mkdir()
        if shape == "sparse":
            rows = _near_ties_sparse(_sparse_rows(rng, 120, 60, 7), rng)
            qrows = _sparse_rows(rng, 10, 60, 5, dup=False)
            qrows[0] = (np.array([1]), np.array([0.7], np.float32))          # few positive rows
            corpus = _corpus(sub, [_sp_array(r) for r in _split(rows, (50, len(rows) - 50))])
            sc = X.Sparse(_queries(sub, _sp_array(qrows), SP), metric, "v")
            S = f32_sparse(qrows, rows, metric, rng)
        else:
            docs = _near_ties_mv(_mv_docs(rng, 70, 6, 5), rng)
            qt = [rng.standard_normal((int(rng.integers(0, 5)), 6)).astype(np.float32) for _ in range(8)]
            corpus = _corpus(sub, [_mv_array(r) for r in _split(docs, (30, len(docs) - 30))])
            sc = X.MultiVector(_queries(sub, _mv_array(qt), MV), metric, "v")
            S = f32_mv(qt, docs, metric, rng)
        systems = {"a": honest_lists(S, k, rng), "b": honest_lists(S, k, rng)}
        if shape == "sparse":
            systems["pads"] = honest_lists(S, k, rng, pad_zero=True)
        got, ex, dis = judge(sc, corpus, k, systems)
        assert all(a and b for a, b in got.values()), (seed, got, ex, dis)
        ref = X.build_reference(sc, corpus, k)
        differ += sum(not np.array_equal(np.asarray(systems["a"][0][q]), ref["I"][q])
                      for q in range(len(S)))
    assert differ > 0


# ---------------------------------------------------------------------------------------------
# scripted legitimate cases and the floor
# ---------------------------------------------------------------------------------------------
def test_a_sparse_list_padded_where_the_true_scores_are_zero_passes(tmp_path):
    """Query 0 shares terms with rows 3, 7, 9 only; k 6. Zero rows or pads after them pass; a list
    that drops one positive row (pads or zero rows in its place) fails both checks."""
    rows = [(np.array([5]), np.array([1.0], np.float32))] * 12
    rows = list(rows)
    for j, v in ((3, 0.9), (7, 0.5), (9, 0.2)):
        rows[j] = (np.array([1, 5]), np.array([v, 1.0], np.float32))
    corpus = _corpus(tmp_path, [_sp_array(rows[:6]), _sp_array(rows[6:])])
    sc = X.Sparse(_queries(tmp_path, _sp_array([(np.array([1]), np.array([1.0], np.float32))]), SP),
                  "ip", "v")
    s = lambda *v: [np.array(v, np.float32)]  # noqa: E731
    i = lambda *v: [np.array(v, np.int64)]     # noqa: E731
    systems = {"zeros": (i(3, 7, 9, 0, 1, 2), s(.9, .5, .2, 0, 0, 0)),
               "zeros_other": (i(3, 7, 9, 11, 4, 8), s(.9, .5, .2, 0, 0, 0)),
               "pads": (i(3, 7, 9, -1, -1, -1), s(.9, .5, .2, -np.inf, -np.inf, -np.inf)),
               "short": (i(3, 7, -1, -1, -1, -1), s(.9, .5, *[-np.inf] * 4)),
               "zero_for_positive": (i(3, 7, 0, 1, 2, 4), s(.9, .5, 0, 0, 0, 0))}
    got, ex, dis = judge(sc, corpus, 6, systems)
    assert got["zeros"] == got["zeros_other"] == got["pads"] == (True, True), (got, ex)
    assert got["short"] == got["zero_for_positive"] == (False, False), got


@pytest.mark.parametrize("shape", ["sparse", "multivector"])
def test_a_tie_at_the_cutoff_broken_the_other_way_passes(tmp_path, shape):
    """Rows 2 and 5 are identical and tie at rank k; one system holds each."""
    if shape == "sparse":
        rows = [(np.array([1]), np.array([v], np.float32)) for v in (0.9, 0.8, 0.5, 0.1, 0.05, 0.5)]
        arr, q, mod = _sp_array, [(np.array([1]), np.array([1.0], np.float32))], SP
        cls = X.Sparse
    else:
        rows = [np.array([[v, 0.0]], np.float32) for v in (0.9, 0.8, 0.5, 0.1, 0.05, 0.5)]
        arr, q, mod = _mv_array, [np.array([[1.0, 0.0]], np.float32)], MV
        cls = X.MultiVector
    corpus = _corpus(tmp_path, [arr(rows[:3]), arr(rows[3:])])
    sc = cls(_queries(tmp_path, arr(q), mod), "ip", "v")
    systems = {"a": ([np.array([0, 1, 2])], [np.array([.9, .8, .5], np.float32)]),
               "b": ([np.array([0, 1, 5])], [np.array([.9, .8, .5], np.float32)])}
    got, ex, dis = judge(sc, corpus, 3, systems)
    assert got == {"a": (True, True), "b": (True, True)}, (ex, dis)


def test_a_zero_token_doc_is_never_a_candidate(tmp_path):
    docs = [np.array([[0.9, 0.0]], np.float32), None, np.zeros((0, 2), np.float32),
            np.array([[0.5, 0.0]], np.float32)]
    corpus = _corpus(tmp_path, [_mv_array(docs)])
    sc = X.MultiVector(_queries(tmp_path, _mv_array([np.array([[1.0, 0.0]], np.float32)]), MV), "ip", "v")
    systems = {"ok": ([np.array([0, 3, -1])], [np.array([.9, .5, -np.inf], np.float32)]),
               "r2": ([np.array([0, 3, 1])], [np.array([.9, .5, 0.0], np.float32)])}
    got, ex, dis = judge(sc, corpus, 3, systems)
    assert got["ok"] == (True, True) and got["r2"] == (False, False), (ex, dis)
    assert any(f.startswith("R2") for f in ex["r2"]["fails"])


@pytest.mark.parametrize("where", ["none", "queries", "first_file", "second_file"])
def test_negative_sparse_weights_are_a_run_problem(tmp_path, where):
    ok = [(np.array([1, 2]), np.array([1.0, 2.0], np.float32))] * 2
    bad = [(np.array([1, 2]), np.array([1.0, -2.0], np.float32))] * 2
    corpus = _corpus(tmp_path, [_sp_array(bad if where == "first_file" else ok),
                                _sp_array(bad if where == "second_file" else ok)])
    sc = X.Sparse(_queries(tmp_path, _sp_array(bad if where == "queries" else ok), SP), "ip", "v")
    got = sc.run_problems(corpus)
    if where in ("none", "second_file"):   # the corpus check reads the FIRST file only (documented)
        assert got == []
    else:
        assert len(got) == 1 and "negative" in got[0] and "non-negative" in got[0]


# ---------------------------------------------------------------------------------------------
# the fixture runs (sparse / multivector: systems novabf and cpu, k 10)
# ---------------------------------------------------------------------------------------------
from helpers import fixture_runs  # noqa: E402
from helpers.check_runs import _npz_rewrite, copy_run  # noqa: E402


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{shape: run dir}: this module's own copy of the shared, cached fixture runs."""
    pytest.importorskip("torch")
    pytest.importorskip("nova_bf")
    return fixture_runs.ok_runs(tmp_path_factory, "check_ragged_results", ("sparse", "multivector"))


def verdict(d, exact_scan, cache, exp=None):
    exp = exp or json.load(open(os.path.join(d, "experiment.json")))
    return V.run(d, exp, exact_scan=exact_scan, cache_dir=str(cache))


def sysok(doc, shape, nm):
    return doc["searches"][shape]["systems"][nm]["ok"]


def reference(d, shape, extra=30):
    exp = json.load(open(os.path.join(d, "experiment.json")))
    s = exp["searches"][0]
    a, inputs, sc, corpus, flt, _ = V.load_search(exp, d, s, L.records(d), [])
    k = int(exp["search"]["k"])
    return k, X.build_reference(sc, corpus, k + extra), sc, corpus


def _plant(runs, tmp_path, shape, fn):
    d = copy_run(runs[shape], tmp_path / "run")
    _npz_rewrite(d, shape, "cpu", fn)
    return d


def _clear_q(r, sc, k):
    """A query whose rank k-1 is clearly (10 bounds) above the last reference row."""
    D, N = r["D"], np.nan_to_num(r["N"])
    gap = D[:, k - 1] - D[:, -1]
    b = sc.bound(np.arange(len(D)), N[:, k - 1]) + sc.bound(np.arange(len(D)), N[:, -1])
    return int(np.flatnonzero(np.isfinite(gap) & (gap > 10 * b))[0])


SHAPES = ["sparse", "multivector"]


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("exact_scan", [False, True])
def test_the_ragged_fixture_runs_pass_both_checks(runs, shape, exact_scan, tmp_path):
    doc = verdict(runs[shape], exact_scan, tmp_path / "c")
    assert doc["ok"], doc["problems"]
    v = doc["searches"][shape]
    assert v["check"] == ("exhaustive" if exact_scan else "disputed") and v["run_problems"] == []
    assert all(x["ok"] for x in v["systems"].values())


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mistake", ["worse_row", "shifted", "duplicate", "short"])
def test_a_planted_mistake_fails_the_system_that_made_it(runs, tmp_path, shape, mistake):
    k, r, sc, _ = reference(runs[shape], shape)
    D, I = r["D"], r["I"]
    q = _clear_q(r, sc, k)
    if mistake == "shifted":
        q = next(int(q) for q in range(len(D)) if D[q, 0] - D[q, k] >
                 10 * (sc.bound(q, np.nan_to_num(r["N"][q, 0])) + sc.bound(q, np.nan_to_num(r["N"][q, k]))))

    def fn(II, DD):
        if mistake == "worse_row":     # a clearly worse row kept at rank 0
            II[q, 0], DD[q, 0] = I[q, -1], np.float32(D[q, -1])
        elif mistake == "shifted":     # the top row dropped, row k+1 taken
            II[q, :k], DD[q, :k] = I[q, 1:k + 1], D[q, 1:k + 1].astype(np.float32)
        elif mistake == "duplicate":
            II[q, 1] = II[q, 0]
        else:                          # short: another system found a positive / finite row there
            assert np.isfinite(D[q, k - 1]) and (shape == "multivector" or D[q, k - 1] > 0)
            II[q, -1], DD[q, -1] = -1, np.finfo(np.float32).min
        return II, DD
    d = _plant(runs, tmp_path, shape, fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        assert not sysok(doc, shape, "cpu") and sysok(doc, shape, "novabf"), (ex, doc["problems"])


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
def test_swapped_ids_fail_the_exhaustive_check(runs, tmp_path, shape):
    """Ranks 0 and k-1 swapped (the scores left in place): the same set as nova-bf's, so the
    default check does not see it (its stated blind spot: order, reported scores of common rows);
    the exhaustive check fails it (R3 order / R4 reported score)."""
    k, r, sc, _ = reference(runs[shape], shape)
    D = r["D"]
    q = next(int(q) for q in range(len(D)) if D[q, 0] - D[q, k - 1] > 10 * sc.bound(q, np.nan_to_num(r["N"][q, 0])))

    def fn(II, DD):
        II[q, [0, k - 1]] = II[q, [k - 1, 0]]
        return II, DD
    d = _plant(runs, tmp_path, shape, fn)
    assert verdict(d, False, tmp_path / "c")["ok"]
    doc = verdict(d, True, tmp_path / "c")
    fails = doc["searches"][shape]["systems"]["cpu"]["fails"]
    assert not sysok(doc, shape, "cpu") and sysok(doc, shape, "novabf")
    assert any(f.startswith(("R3", "R4")) for f in fails), fails


@pytest.mark.slow
def test_a_zero_token_doc_returned_fails_on_the_fixture_run(runs, tmp_path):
    k, r, sc, corpus = reference(runs["multivector"], "multivector")
    # a null (zero-token) document of the fixture corpus
    col = pq.read_table(corpus.paths[0], columns=[sc.vcol]).column(0).combine_chunks()
    g = int(np.flatnonzero(np.asarray(col.is_null()))[0])

    def fn(II, DD):
        II[0, k - 1], DD[0, k - 1] = g, np.float32(0.0)
        return II, DD
    d = _plant(runs, tmp_path, "multivector", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        assert not sysok(doc, "multivector", "cpu") and sysok(doc, "multivector", "novabf"), doc["problems"]


@pytest.mark.slow
def test_a_negative_query_weight_is_a_problem_of_the_run(runs, tmp_path):
    d = copy_run(runs["sparse"], tmp_path / "run")
    exp = json.load(open(os.path.join(d, "experiment.json")))
    t = pq.read_table(L.abs_path(exp["queries"]["path"]))
    c = t.column(exp["queries"]["column"]).combine_chunks()
    rows = [(i, np.array(v, np.float32)) for i, v in zip(c.field("indices").to_pylist(), c.field("values").to_pylist())]
    j = next(n for n, (i, _) in enumerate(rows) if len(i))
    rows[j][1][0] = -rows[j][1][0]
    p = str(tmp_path / "neg_queries.parquet")
    pq.write_table(t.set_column(t.schema.get_field_index(exp["queries"]["column"]), exp["queries"]["column"],
                                _sp_array(rows)), p)
    exp["queries"]["path"] = p
    doc = verdict(d, False, tmp_path / "c", exp)
    assert not doc["ok"] and any("negative query weight" in x for x in doc["problems"]), doc["problems"]
