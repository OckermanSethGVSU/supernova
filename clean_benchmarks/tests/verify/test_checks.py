"""bench/verify/check.py + exact.py (the redesigned checks, CHECKS_REDESIGN.md).

The inputs are real `bench run --dev` runs of the dense and filtered fixture experiments (shared,
cached: helpers.fixture_runs); sparse and multivector are in test_checks_ragged.py. Each test plants one mistake in a copy of a run and asserts the
verdict of both checks: the default disputed-rows check and the exhaustive one.
"""

import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
pytest.importorskip("nova_bf")

from bench.verify import check as V  # noqa: E402
from bench.verify import exact as X  # noqa: E402
from bench.verify import load as L  # noqa: E402
from bench.verify import oracle as O  # noqa: E402
from helpers.check_runs import _nb_rewrite, _npz_rewrite, _uuid, copy_run  # noqa: E402
from helpers.check_runs import runs  # noqa: F401,E402 (fixture)

pytestmark = pytest.mark.slow


def verdict(d, exact_scan, cache, **exp_over):
    exp = json.load(open(os.path.join(d, "experiment.json")))
    exp.setdefault("checks", {}).update(exp_over)
    return V.run(d, exp, exact_scan=exact_scan, cache_dir=str(cache))


def sysok(doc, search, nm):
    return doc["searches"][search]["systems"][nm]["ok"]


def ctx(d, search):
    exp = json.load(open(os.path.join(d, "experiment.json")))
    s = next(x for x in exp["searches"] if x["name"] == search)
    return (exp, s) + V.load_search(exp, d, s, L.records(d), [])[:5]


def reference(d, search, extra=30):
    exp, s, a, inputs, scorer, corpus, flt = ctx(d, search)
    k = int(exp["search"]["k"])
    r = X.build_reference(scorer, corpus, k + extra, flt)
    D, I, N = r["D"], r["I"], r["N"]
    return k, D, I, scorer, corpus, flt, inputs


def nb_uuid(d, g):
    exp = json.load(open(os.path.join(d, "experiment.json")))
    keys, rows, offsets, _ = L.corpus_index(L.abs_path(exp["corpus"]["path"]), need_ids=False)
    f = int(np.searchsorted(offsets, g, side="right") - 1)
    return _uuid(keys[f], int(g - offsets[f]))


# ---------------------------------------------------------------------------------------------
# clean runs pass
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["dense", "filtered", "sparse", "multivector"])
@pytest.mark.parametrize("exact_scan", [False, True])
def test_the_fixture_runs_pass_both_checks(runs, shape, exact_scan, tmp_path):
    doc = verdict(runs[shape], exact_scan, tmp_path / "cache")
    assert doc["ok"], doc["problems"]
    for s, v in doc["searches"].items():
        assert v["check"] == ("exhaustive" if exact_scan else "disputed")
        assert all(x["ok"] for x in v["systems"].values())
    if shape == "dense":
        assert doc["e2e"] == "IDENTICAL"


def test_the_streamed_reference_equals_a_one_shot_reference(runs):
    k, D, I, scorer, corpus, flt, _ = reference(runs["dense"], "dense", extra=0)
    xs = [np.asarray(pq.read_table(p, columns=[scorer.vcol]).column(0).to_pylist(), np.float32)
          for p in corpus.paths]
    D1, I1 = O.exact_topk_f64(scorer.q, [np.concatenate(xs)], k, scorer.unit)
    assert np.array_equal(I, I1) and np.array_equal(D, D1)


def test_score_pairs_is_the_float64_score_of_each_pair(runs):
    k, D, I, scorer, corpus, flt, _ = reference(runs["dense"], "dense")
    qs = np.repeat(np.arange(len(D)), 3)
    gs = I[:, [0, 5, 20]].ravel()
    e, _ = X.score_pairs(scorer, corpus, qs, gs)
    # float64 both ways; only the summation order differs (matmul vs pairwise dot)
    np.testing.assert_allclose(e, D[:, [0, 5, 20]].ravel(), rtol=1e-12, atol=1e-15)


def test_the_reference_is_cached_and_reused(runs, tmp_path):
    a = verdict(runs["dense"], True, tmp_path / "c")
    b = verdict(runs["dense"], True, tmp_path / "c")
    ra, rb = a["searches"]["dense"]["reference"], b["searches"]["dense"]["reference"]
    assert ra["cached"] is False and rb["cached"] is True and ra["key"] == rb["key"]
    files = sorted(os.listdir(tmp_path / "c"))
    assert [f for f in files if f.endswith(".npz")] == [ra["key"] + ".npz"]
    # beside it, the corpus's nova-bf uuid table (load.cached_uuid_table), and nothing else
    assert len(files) == 2 and [f for f in files if f.startswith("uuids-") and f.endswith(".arrow")]


# ---------------------------------------------------------------------------------------------
# planted mistakes (dense: systems novabf and cpu, k 10)
# ---------------------------------------------------------------------------------------------
def _worse_row(d, search="dense"):
    """(query, true rank-0 row, a row outside the top-k clearly below the k-th)."""
    k, D, I, scorer, corpus, flt, _ = reference(d, search)
    bnd = scorer.bound(0, 1.0) * 4
    q = next(q for q in range(len(D)) if D[q, k - 1] - D[q, -1] > bnd)
    return q, int(I[q, 0]), int(I[q, -1]), float(D[q, -1]), k


def test_a_clearly_worse_row_fails_the_system_that_kept_it(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    q, best, worse, s_worse, k = _worse_row(d)

    def fn(I, D):
        I[q, 0], D[q, 0] = worse, np.float32(s_worse)
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        assert not doc["ok"] and not sysok(doc, "dense", "cpu") and sysok(doc, "dense", "novabf"), doc["problems"]


@pytest.mark.parametrize("tie", ["exact", "within_bound"])
def test_a_tie_at_the_cutoff_broken_the_other_way_passes(runs, tmp_path, tie):
    """cpu holds one of the two rows tied (exactly, or within the float32 bound) at rank k / k+1;
    giving it the other one instead must pass both checks."""
    d = copy_run(runs["dense"], tmp_path / "run")
    k, D, I, scorer, *_ = reference(d, "dense")
    z = np.load(os.path.join(d, "legs", "dense__cpu__r1.topk.npz"))
    gap = D[:, k - 1] - D[:, k]
    want = (gap == 0) if tie == "exact" else ((gap > 0) & (gap <= scorer.bound(0, 1.0)))
    q = next((int(q) for q in np.flatnonzero(want)
              if len({int(I[q, k - 1]), int(I[q, k])} & set(z["I"][q].tolist())) == 1), None)
    assert q is not None, f"the fixture has a {tie} tie at the cutoff"
    have = int(I[q, k - 1]) if int(I[q, k - 1]) in z["I"][q].tolist() else int(I[q, k])
    other = int(I[q, k]) if have == int(I[q, k - 1]) else int(I[q, k - 1])
    j = z["I"][q].tolist().index(have)

    def fn(II, DD):
        II[q, j] = other
        DD[q, j] = np.float32(D[q, list(I[q]).index(other)])
        return II, DD
    _npz_rewrite(d, "dense", "cpu", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        assert doc["ok"], doc["problems"]


def test_wrong_order_fails_only_the_exhaustive_check(runs, tmp_path):
    """Order inside a list is not part of the default check (accepted in the design)."""
    d = copy_run(runs["dense"], tmp_path / "run")
    k, D, I, scorer, *_ = reference(d, "dense")
    q = next(q for q in range(len(D)) if D[q, 0] - D[q, k - 1] > 4 * scorer.bound(0, 1.0))

    def fn(II, DD):
        II[q, [0, k - 1]] = II[q, [k - 1, 0]]
        DD[q, [0, k - 1]] = DD[q, [k - 1, 0]]
        return II, DD
    _npz_rewrite(d, "dense", "cpu", fn)
    assert verdict(d, False, tmp_path / "c")["ok"]
    doc = verdict(d, True, tmp_path / "c")
    assert not sysok(doc, "dense", "cpu") and sysok(doc, "dense", "novabf")
    assert any(f.startswith("R3") for f in doc["searches"]["dense"]["systems"]["cpu"]["fails"])


@pytest.mark.parametrize("delta,fails", [(1e-3, True), (None, False)])
def test_a_reported_score_is_judged_against_the_float32_bound(runs, tmp_path, delta, fails):
    d = copy_run(runs["dense"], tmp_path / "run")
    k, D, I, scorer, *_ = reference(d, "dense")
    z = np.load(os.path.join(d, "legs", "dense__cpu__r1.topk.npz"))
    step = delta if delta is not None else 0.4 * scorer.bound(0, 1.0)

    def fn(II, DD):
        DD[3, 4] = np.float32(D[3, list(I[3]).index(int(II[3, 4]))] + step)
        return II, DD
    assert int(z["I"][3, 4]) in set(I[3].tolist())
    _npz_rewrite(d, "dense", "cpu", fn)
    doc = verdict(d, True, tmp_path / "c")
    assert sysok(doc, "dense", "cpu") is (not fails), doc["problems"]
    if fails:
        assert any(f.startswith("R4") for f in doc["searches"]["dense"]["systems"]["cpu"]["fails"])


def test_a_nan_reported_score_fails(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")

    def fn(I, D):
        D[2, 1] = np.nan
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    assert not sysok(verdict(d, True, tmp_path / "c"), "dense", "cpu")


def test_a_duplicate_id_is_a_list_defect(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")

    def fn(I, D):
        I[5, 1] = I[5, 0]
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        v = doc["searches"]["dense"]["systems"]["cpu"]
        assert not v["ok"] and v["defects"]["duplicate_ids"] == 1


def test_a_short_list_fails_when_another_system_found_the_row(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")

    def fn(I, D):
        I[7, -1], D[7, -1] = -1, np.finfo(np.float32).min
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        assert not sysok(doc, "dense", "cpu") and sysok(doc, "dense", "novabf"), (ex, doc["problems"])


def test_a_row_every_system_missed_is_seen_only_by_the_exhaustive_check(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    q, best, worse, s_worse, k = _worse_row(d)

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
    dd = verdict(d, False, tmp_path / "c")    # the stated blind spot (e2e differs: nova-bf r1 was edited)
    assert sysok(dd, "dense", "cpu") and sysok(dd, "dense", "novabf"), dd["problems"]
    doc = verdict(d, True, tmp_path / "c")
    assert not sysok(doc, "dense", "cpu") and not sysok(doc, "dense", "novabf")


def test_expect_inexact_reports_but_does_not_fail_the_run(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    q, best, worse, s_worse, k = _worse_row(d)

    def fn(I, D):
        I[q, 0], D[q, 0] = worse, np.float32(s_worse)
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    doc = verdict(d, True, tmp_path / "c", expect_inexact=["cpu"])
    v = doc["searches"]["dense"]["systems"]["cpu"]
    assert doc["ok"] and not v["ok"] and v["expect_inexact"] and doc["warnings"]


# ---------------------------------------------------------------------------------------------
# filtered (static_struct: novabf, masked_bitmap; k 20)
# ---------------------------------------------------------------------------------------------
def test_a_row_the_filter_excludes_fails_the_system_that_returned_it(runs, tmp_path):
    d = copy_run(runs["filtered"], tmp_path / "run")
    search = "static_struct"
    exp, s, a, inputs, scorer, corpus, flt = ctx(d, search)
    q = 0
    allg = np.arange(corpus.rows)
    ok = flt.allowed(np.zeros(corpus.rows, np.int64), allg)
    assert (~ok).any(), "the fixture filter excludes some rows"
    e, _ = X.score_pairs(scorer, corpus, np.zeros((~ok).sum(), np.int64), allg[~ok])
    masked = int(allg[~ok][np.argmax(e)])

    def fn(I, D):
        I[q, 1:] = I[q, :-1].copy()
        D[q, 1:] = D[q, :-1].copy()
        I[q, 0], D[q, 0] = masked, np.float32(e.max())
        return I, D
    _npz_rewrite(d, search, "masked_bitmap", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        v = doc["searches"][search]["systems"]["masked_bitmap"]
        assert not v["ok"] and v["filter_violations"] == 1, (ex, v["fails"])
        assert sysok(doc, search, "novabf"), doc["problems"]


# ---------------------------------------------------------------------------------------------
# the exhaustive rules on scripted inputs (no corpus read: every id is in the reference)
# ---------------------------------------------------------------------------------------------
class _Scorer:
    floor = -np.inf

    def __init__(self, g, qnorm=1.0, unit=True):
        self.g, self.qnorm, self.unit = g, qnorm, unit

    def bound(self, qi, xn):
        if self.unit:
            return np.full(np.broadcast(np.asarray(qi), np.asarray(xn)).shape, self.g)
        return self.g * self.qnorm * np.asarray(xn, np.float64) + 0 * np.asarray(qi)


class _Corpus:
    rows = 10 ** 6


def _ref(rows, K, scorer):
    """A reference over a scripted corpus ``rows`` = [(id, exact score, norm)]: D / I the top-K by
    score, P / PI the top-K by score minus bound (as build_reference)."""
    ids = np.array([r[0] for r in rows], np.int64)
    sc = np.array([r[1] for r in rows], np.float64)
    nr = np.array([r[2] for r in rows], np.float64)
    p = sc - scorer.bound(np.zeros(len(rows), np.int64), nr)
    out = {}
    for (kd, ki), v in ((("D", "I"), sc), (("P", "PI"), p)):
        o = np.lexsort((ids, -v))[:K]
        A = np.full((1, K), -np.inf)
        AI = np.full((1, K), -1, np.int64)
        A[0, :len(o)], AI[0, :len(o)] = v[o], ids[o]
        out[kd], out[ki] = A, AI
    o = np.lexsort((ids, -sc))[:K]
    out["N"] = np.full((1, K), np.nan)
    out["N"][0, :len(o)] = nr[o]
    return out


def test_a_list_that_dropped_a_top_row_fails_even_when_scores_are_densely_spaced():
    """Dropping row 0 shifts every later rank by one; each rank's gap (2e-5) is under the bound,
    but the dropped row beats the kept tail by 1e-3 (the review's finding)."""
    k, K = 50, 82
    sc = 0.9 - np.arange(K) * 2e-5
    g = X.gamma(2 * 768 + 8)
    assert 2e-5 < 2 * g < sc[0] - sc[k]
    ref = _ref([(i, sc[i], 1.0) for i in range(K)], K, _Scorer(g))
    M = np.arange(1, k + 1)[None, :]
    out = V.exhaustive_check(["s"], {"s": (M, sc[1:k + 1][None, :].copy())}, k, _Scorer(g), _Corpus(), None, ref)
    assert any("dropped a row clearly better" in f for f in out["s"]["fails"]), out["s"]
    ok = V.exhaustive_check(["s"], {"s": (np.arange(k)[None, :], sc[:k][None, :].copy())}, k, _Scorer(g),
                            _Corpus(), None, ref)
    assert not ok["s"]["fails"]


def test_an_honest_ip_list_with_one_huge_norm_row_passes():
    """ip, one row of norm 1e6 (its float32 error is large): an honest float32 order x1, x2, y
    differs from the exact order y, x1, x2 at every rank, yet is within the bound pairwise."""
    g = X.gamma(2 * 768 + 8)
    sc = _Scorer(g, unit=False)
    ref = _ref([(7, 0.99848, 1e6), (1, 0.99669, 1.0), (2, 0.99629, 1.0)], 3, sc)
    M = np.array([[1, 2, 7]])
    S = np.array([[0.99669, 0.99629, 0.99414]])
    out = V.exhaustive_check(["s"], {"s": (M, S)}, 3, sc, _Corpus(), None, ref)
    assert not out["s"]["fails"], out["s"]


def test_ip_a_dropped_row_outside_the_top_k_by_score_is_still_caught():
    """ip, k 1, K 2: two rows score 1.0 with norm 1e4 (large bound); a third scores 0.99 with norm
    ~1 (tiny bound) and is outside the top-K by score; the system kept 0.95 (norm 0.95). The 0.99
    row beats it by far more than their bounds: the pessimistic list holds it (review round 2)."""
    g = X.gamma(2 * 64 + 8)
    sc = _Scorer(g, unit=False)
    ref = _ref([(0, 1.0, 1e4), (1, 1.0, 1e4), (9, 0.99, 0.99), (5, 0.95, 0.95)], 2, sc)
    assert 9 not in ref["I"] and 9 in ref["PI"]
    orig = X.score_pairs
    X.score_pairs = lambda scorer, corpus, qs, gs: (np.array([0.95]), np.array([0.95]))
    try:
        bad = V.exhaustive_check(["s"], {"s": (np.array([[5]]), np.array([[0.95]]))}, 1, sc, _Corpus(), None, ref)
    finally:
        X.score_pairs = orig
    assert any("dropped a row clearly better" in f for f in bad["s"]["fails"]), bad["s"]


def test_ip_an_honest_list_behind_inverted_large_norm_rows_passes():
    """ip, k 1, K 2 (review round 3): rows with large norms whose float32 scores round low sit just
    above x in exact score; an honest float32 engine returns x, a legitimate tie with each of them."""
    g = X.gamma(2 * 2 + 8)
    sc = _Scorer(g, unit=False)
    rows = [(i, 10.0000001, 1.5e5) for i in range(3)] + [(9, 9.9995, 99.995)]
    ref = _ref(rows, 2, sc)
    orig = X.score_pairs
    X.score_pairs = lambda scorer, corpus, qs, gs: (np.array([9.9995]), np.array([99.995]))
    try:
        out = V.exhaustive_check(["s"], {"s": (np.array([[9]]), np.array([[9.999512]]))}, 1, sc, _Corpus(), None, ref)
    finally:
        X.score_pairs = orig
    assert not any("dropped" in f for f in out["s"]["fails"]), out["s"]


def test_a_list_out_of_order_beyond_the_bound_fails_the_order_rule():
    g = 1e-6
    sc = np.array([0.9, 0.8, 0.7, 0.6])
    ref = _ref([(i, sc[i], 1.0) for i in range(4)], 5, _Scorer(g))
    M = np.array([[0, 2, 1, 3]])          # right set, ranks 1 and 2 swapped (gap 0.1 >> bound)
    out = V.exhaustive_check(["s"], {"s": (M, sc[[0, 2, 1, 3]][None, :])}, 4, _Scorer(g), _Corpus(), None, ref)
    assert any("out of order" in f for f in out["s"]["fails"])


def test_a_shifted_list_fails_on_a_real_run(runs, tmp_path):
    """cpu drops its true top row and takes the next row past the cutoff (a sorted, full list)."""
    d = copy_run(runs["dense"], tmp_path / "run")
    k, D, I, scorer, *_ = reference(d, "dense")
    q = next(q for q in range(len(D)) if D[q, 0] - D[q, k] > 4 * scorer.bound(0, 1.0))

    def fn(II, DD):
        II[q, :k] = I[q, 1:k + 1]
        DD[q, :k] = D[q, 1:k + 1].astype(np.float32)
        return II, DD
    _npz_rewrite(d, "dense", "cpu", fn)
    for ex in (False, True):
        doc = verdict(d, ex, tmp_path / "c")
        assert not sysok(doc, "dense", "cpu") and sysok(doc, "dense", "novabf"), (ex, doc["problems"])


def test_a_tie_choice_outside_the_reference_is_rescored_from_the_corpus(runs, tmp_path, monkeypatch):
    """No spare rows: cpu taking the row tied at rank k+1 is outside the reference, so it is scored
    from the corpus (the deep-tie path) and still passes."""
    monkeypatch.setattr(X, "SPARE", 0)
    d = copy_run(runs["dense"], tmp_path / "run")
    k, D, I, *_ = reference(d, "dense")
    z = np.load(os.path.join(d, "legs", "dense__cpu__r1.topk.npz"))
    pair = lambda q: {int(I[q, k - 1]), int(I[q, k])}  # noqa: E731
    q = next(int(q) for q in np.flatnonzero(D[:, k - 1] == D[:, k])
             if len(pair(q) & set(z["I"][q].tolist())) == 1)
    have = (pair(q) & set(z["I"][q].tolist())).pop()
    j = z["I"][q].tolist().index(have)

    def fn(II, DD):
        II[q, j] = int(I[q, k])          # the tied row just outside the reference (K = k)
        return II, DD
    _npz_rewrite(d, "dense", "cpu", fn)
    doc = verdict(d, True, tmp_path / "c")
    v = doc["searches"]["dense"]["systems"]["cpu"]
    assert v["ok"] and v["rescored_outside_reference"] >= 1, v


def test_a_corrupt_cache_file_is_rebuilt(runs, tmp_path):
    a = verdict(runs["dense"], True, tmp_path / "c")
    key = a["searches"]["dense"]["reference"]["key"]
    open(tmp_path / "c" / f"{key}.npz", "wb").write(b"not an npz")
    b = verdict(runs["dense"], True, tmp_path / "c")
    assert b["ok"] and b["searches"]["dense"]["reference"]["cached"] is False
    assert verdict(runs["dense"], True, tmp_path / "c")["searches"]["dense"]["reference"]["cached"] is True


def test_an_empty_corpus_file_is_scanned(tmp_path):
    import pyarrow as pa
    import pyarrow.fs as pafs
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 8)).astype(np.float32)
    t = lambda a: pa.table({"v": pa.array(list(a), pa.list_(pa.float32()))})  # noqa: E731
    paths = [str(tmp_path / f"p{i}.parquet") for i in range(3)]
    pq.write_table(t(x[:30]), paths[0])
    pq.write_table(t(x[:0]), paths[1])
    pq.write_table(t(x[30:]), paths[2])
    corpus = X.Corpus(pafs.LocalFileSystem(), paths, [0, 30, 30, 50])
    q = rng.standard_normal((4, 8))
    sc = X.Dense(q, "cosine", "v")
    I = X.build_reference(sc, corpus, 5)["I"]
    D1, I1 = O.exact_topk_f64(sc.q, [x], 5, True)
    assert np.array_equal(I, I1)


def test_a_missing_e2e_leg_fails(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    for f in os.listdir(os.path.join(d, "legs")):
        if f.startswith("e2e__") and f.endswith(".json") and not f.endswith("clock.json"):
            os.remove(os.path.join(d, "legs", f))
    doc = verdict(d, False, tmp_path / "c")
    assert not doc["ok"] and doc["e2e"] == "ERROR"


def _files(tmp_path, blocks):
    import pyarrow as pa
    import pyarrow.fs as pafs
    paths, off = [], [0]
    for i, b in enumerate(blocks):
        p = str(tmp_path / f"p{i}.parquet")
        pq.write_table(pa.table({"v": pa.array(list(b), pa.list_(pa.float32()))}), p)
        paths.append(p)
        off.append(off[-1] + len(b))
    return X.Corpus(pafs.LocalFileSystem(), paths, off)


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_a_row_with_a_non_finite_value_is_never_in_the_reference(tmp_path, bad):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 8)).astype(np.float32)
    x[3, 0] = bad
    q = np.abs(rng.standard_normal((5, 8)))          # q0 > 0: row 3 would score +inf
    corpus = _files(tmp_path, [x[:20], x[20:]])
    r = X.build_reference(X.Dense(q, "ip", "v"), corpus, 10)
    D, I = r["D"], r["I"]
    assert np.isfinite(D).all() and not (I == 3).any() and (I >= 0).all()


def test_a_same_size_rewrite_with_the_mtime_kept_changes_the_cache_key(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 8)).astype(np.float32)
    qf = tmp_path / "queries.bin"
    qf.write_bytes(b"queries")                        # unchanged: only the corpus can move the key
    corpus = _files(tmp_path, [x])
    st = os.stat(corpus.paths[0])
    key = lambda c: X.reference_key(c, str(qf), None, "v", "v", "ip", "dense", None, 5)  # noqa: E731
    k1 = key(corpus)
    corpus2 = _files(tmp_path, [-x])                  # same shape, same size
    os.utime(corpus.paths[0], ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(corpus.paths[0]).st_size == st.st_size
    assert corpus.identity()[0][:3] == [corpus.paths[0], st.st_size, corpus.identity()[0][2]]
    assert k1 != key(corpus2)



def test_a_copied_run_dir_is_checked_on_its_own_files(runs, tmp_path):
    import shutil
    src = copy_run(runs["dense"], tmp_path / "orig")
    d = str(tmp_path / "copy")
    shutil.copytree(src, d)                  # leg records still name the original's paths
    q, best, worse, s_worse, k = _worse_row(d)
    bu, wu = nb_uuid(d, best), nb_uuid(d, worse)

    def nb(ids, sc):
        j = ids[q].index(bu)
        ids[q][j], sc[q][j] = wu, s_worse
        return ids, sc
    for p in sorted(os.listdir(os.path.join(d, "novabf"))):
        for f in os.listdir(os.path.join(d, "novabf", p)):
            if f.endswith(".parquet") and "__r1" in p:
                t = pq.read_table(os.path.join(d, "novabf", p, f))
                ids, sc = nb(t.column("hit_ids").to_pylist(), t.column("hit_scores").to_pylist())
                import pyarrow as pa
                t = t.set_column(t.schema.get_field_index("hit_ids"), t.schema.field("hit_ids"),
                                 pa.array(ids, t.schema.field("hit_ids").type))
                t = t.set_column(t.schema.get_field_index("hit_scores"), t.schema.field("hit_scores"),
                                 pa.array(sc, t.schema.field("hit_scores").type))
                pq.write_table(t, os.path.join(d, "novabf", p, f))
    doc = verdict(d, False, tmp_path / "c")
    assert not sysok(doc, "dense", "novabf"), "the copy's own (edited) nova-bf result was checked"
    assert verdict(src, False, tmp_path / "c")["ok"]




def test_a_copy_without_its_configs_fails_loudly(runs, tmp_path):
    import shutil
    d = copy_run(runs["filtered"], tmp_path / "run")
    shutil.rmtree(os.path.join(d, "configs"))
    doc = verdict(d, False, tmp_path / "c")
    assert not doc["ok"] and all("could not check" in p for p in doc["problems"] if "input" not in p)



def test_a_synced_dense_run_uses_the_config_text_the_manifest_recorded(runs, tmp_path):
    """Dense legs name the experiment's repo config; on a run synced from a node that path does
    not exist here, and the repo file may have changed: the manifest's copy decides."""
    d = copy_run(runs["dense"], tmp_path / "run")
    for f in os.listdir(os.path.join(d, "legs")):
        if f.endswith(".json") and "novabf" in f and not f.endswith("clock.json"):
            p = os.path.join(d, "legs", f)
            txt = open(p).read().replace(L.CB, "/home/sky_workdir/clean_benchmarks")
            open(p, "w").write(txt)
    doc = verdict(d, False, tmp_path / "c")
    assert sysok(doc, "dense", "novabf") and sysok(doc, "dense", "cpu"), doc["problems"]


def test_a_bare_file_name_in_the_manifest_never_claims_a_config(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    json.dump({"files": {"x.yaml": "corpus: {id_column: id}\n", "experiments/b/x.yaml": "queries: {}\n"}},
              open(run / "manifest.json", "w"))
    with pytest.raises(V.CheckError):
        V.novabf_cfg(str(run), "/n/cb/experiments/c/x.yaml", "s")
    cfg, warn = V.novabf_cfg(str(run), "/home/sky_workdir/clean_benchmarks/experiments/b/x.yaml", "s")
    assert cfg == {"queries": {}} and warn is None



def test_a_stale_checks_json_is_found(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    exp = json.load(open(os.path.join(d, "experiment.json")))
    V.write(d, V.run(d, exp, cache_dir=str(tmp_path / "c")))
    doc = json.load(open(os.path.join(d, "checks.json")))
    assert V.input_problems(d, doc) == []
    p = os.path.join(d, "legs", "dense__cpu__r1.json")
    open(p, "a").write("\n")
    assert any("legs/dense__cpu__r1.json: changed" in x for x in V.input_problems(d, doc))
    open(os.path.join(d, "legs", "extra.json"), "w").write("{}")
    assert any("legs/extra.json: not read by the checks" in x for x in V.input_problems(d, doc))
    os.remove(os.path.join(d, "legs", "dense__cpu__r2.csv"))
    assert any("legs/dense__cpu__r2.csv: missing" in x for x in V.input_problems(d, doc))
    assert "records no inputs" in V.input_problems(d, {"ok": True})[0]


def test_a_search_that_cannot_be_checked_fails_and_could_not_run_records_inputs(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    exp = dict(json.load(open(os.path.join(d, "experiment.json"))), modality="nonsense")
    doc = V.run(d, exp, cache_dir=str(tmp_path / "c"))
    assert doc["ok"] is False and any("could not check" in p and "no scorer" in p for p in doc["problems"])
    doc = V.could_not_run(d, V.CheckError("experiment.json unreadable"))
    assert doc["ok"] is False and doc["problems"] == ["checks could not run: experiment.json unreadable"]
    V.write(d, doc)
    assert V.input_problems(d, json.load(open(os.path.join(d, "checks.json")))) == []


def test_compare_lists_judges_exactly_the_given_legs(runs, tmp_path):
    """s8/compare_s8.py's tie verdict: the default rule on two legs' files wherever they are."""
    d = copy_run(runs["dense"], tmp_path / "run")
    exp = json.load(open(os.path.join(d, "experiment.json")))
    q, best, worse, s_worse, k = _worse_row(d)
    good = os.path.join(d, "legs", "dense__cpu__r1.topk.npz")
    z = dict(np.load(good))
    z["I"], z["D"] = z["I"].copy(), z["D"].copy()
    z["I"][q, 0], z["D"][q, 0] = worse, np.float32(s_worse)
    bad = str(tmp_path / "elsewhere.npz")      # outside the run dir
    np.savez(bad, **z)
    same = V.compare_lists(exp, d, "dense", [("new", {"_topk": good}, "harness"),
                                             ("old", {"_topk": good}, "harness")])
    assert same == {"tie_equivalent": True, "fails": []}
    out = V.compare_lists(exp, d, "dense", [("new", {"_topk": good}, "harness"),
                                            ("old", {"_topk": bad}, "harness")])
    assert out["tie_equivalent"] is False and all(f.startswith(("old: ", "new vs old: ")) for f in out["fails"]), out
    # a nova-bf result, mapped by the config its record names
    rec = L.records(d)["dense__novabf__r1"]
    s = exp["searches"][0]
    nb = {"_result": L.novabf_result(rec, s), "_config": L.novabf_config(rec)}
    out = V.compare_lists(exp, d, "dense", [("nb", nb, "novabf"), ("old", {"_topk": bad}, "harness")])
    assert out["tie_equivalent"] is False and all(f.startswith(("old: ", "nb vs old: ")) for f in out["fails"]), out


def test_compare_lists_sees_order_and_reported_scores(runs, tmp_path):
    """s8's tie verdict covers order and scores, not only sets: a reversed list or scaled scores
    are not tie-equivalent; ties broken the other way still are."""
    d = copy_run(runs["dense"], tmp_path / "run")
    exp = json.load(open(os.path.join(d, "experiment.json")))
    good = os.path.join(d, "legs", "dense__cpu__r1.topk.npz")
    z = dict(np.load(good))

    def variant(name, fn):
        zz = {k: v.copy() for k, v in z.items()}
        zz["I"], zz["D"] = fn(zz["I"], zz["D"])
        p = str(tmp_path / f"{name}.npz")
        np.savez(p, **zz)
        return V.compare_lists(exp, d, "dense", [("new", {"_topk": good}, "harness"),
                                                 ("old", {"_topk": p}, "harness")])
    k = int(exp["search"]["k"])
    rev = variant("rev", lambda I, D: (I[:, :k][:, ::-1].copy(), D[:, :k][:, ::-1].copy()))
    assert rev["tie_equivalent"] is False and any("ranks hold rows" in f for f in rev["fails"]), rev
    scaled = variant("scaled", lambda I, D: (I, (D * 1.5).astype(D.dtype)))
    assert scaled["tie_equivalent"] is False and any("reported scores" in f for f in scaled["fails"]), scaled
    k_, Dref, Iref, *_ = reference(d, "dense")
    q = next(int(q) for q in range(len(Dref)) if Dref[q, 0] == Dref[q, 1])  # an exact tie at the top

    def swap(I, D):
        j0, j1 = list(I[q]).index(int(Iref[q, 0])), list(I[q]).index(int(Iref[q, 1]))
        I[q, [j0, j1]] = I[q, [j1, j0]]
        D[q, [j0, j1]] = D[q, [j1, j0]]
        return I, D
    try:
        tie = variant("tie", swap)
    except (StopIteration, ValueError):
        pytest.skip("no exact tie at rank 0 held by cpu")
    assert tie["tie_equivalent"] is True, tie


def test_bench_verify_resolves_paths_and_refreshes_the_reports(runs, tmp_path, monkeypatch):
    """bench verify resolves the experiment's ${READ}-style section (as bench run does) and
    rewrites summary.md from the new checks.json, so no report keeps the old verdict."""
    from bench import paths
    d = copy_run(runs["dense"], tmp_path / "run")
    used = []
    real = paths.use
    monkeypatch.setattr(paths, "use", lambda e: used.append(e.get("name")) or real(e))
    q, best, worse, s_worse, k = _worse_row(d)

    def fn(I, D):
        I[q, 0], D[q, 0] = worse, np.float32(s_worse)
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    assert "checks ok: True" in open(os.path.join(d, "summary.md")).read()
    assert V.main([d, "--cache", str(tmp_path / "c")]) == 1
    assert used, "paths.use was not called"
    assert "checks ok: False" in open(os.path.join(d, "summary.md")).read()


def test_run_inputs_keys_a_nested_config_by_its_own_path(runs, tmp_path):
    d = copy_run(runs["filtered"], tmp_path / "run")
    os.makedirs(os.path.join(d, "configs", "sub"))
    open(os.path.join(d, "configs", "sub", "x.yaml"), "w").write("a: 1\n")
    keys = set(V.run_inputs(d))
    assert "configs/sub/x.yaml" in keys and "configs/x.yaml" not in keys


def test_a_synced_runs_base_experiment_is_found_in_this_checkout(tmp_path):
    from bench import paths
    exp = {"name": "child", "_base": ["/home/ubuntu/sky_workdir/experiments/dense/pes2o.yaml"]}
    assert paths.use(exp) == ["child", "dense_pes2o"]


def test_a_crashed_bench_verify_still_rewrites_the_reports(runs, tmp_path, monkeypatch):
    d = copy_run(runs["dense"], tmp_path / "run")
    assert "checks ok: True" in open(os.path.join(d, "summary.md")).read()

    def boom(*a, **k):
        raise RuntimeError("leg file unreadable")
    monkeypatch.setattr(V, "run", boom)
    assert V.main([d, "--cache", str(tmp_path / "c")]) == 2
    doc = json.load(open(os.path.join(d, "checks.json")))
    assert doc["ok"] is False and "leg file unreadable" in doc["problems"][0]
    assert "checks ok: False" in open(os.path.join(d, "summary.md")).read()


def test_any_error_while_checking_is_a_failed_check_in_bench_run_too(runs, tmp_path, monkeypatch):
    """bench run / sweep / verify share run_or_fail: an error that is not a CheckError (expired S3
    credentials, an unreadable file) still ends with a failing checks.json saying why."""
    d = copy_run(runs["dense"], tmp_path / "run")
    exp = json.load(open(os.path.join(d, "experiment.json")))

    def boom(*a, **k):
        raise OSError("AWS Error ACCESS_DENIED during HeadObject")
    monkeypatch.setattr(V, "run", boom)
    doc, crashed = V.run_or_fail(d, exp)
    assert crashed and doc["ok"] is False and "ACCESS_DENIED" in doc["problems"][0]
