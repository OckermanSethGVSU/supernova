"""bench/verify/compare.py: list defects, tied fixtures, the exact scan and the shared-id
score rule."""

import types

import numpy as np
import pytest

from bench.verify import agree
from bench.verify import compare as C

from helpers.scan_fixtures import _args, _inputs, _mv_fixture, _oneshot, _sparse_fixture


def test_the_fixtures_really_tie():
    """The sparse fixture's integer weights give exact ties at the k-th place (so the tie rule
    above is exercised, not vacuous)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        nf, rows = _sparse_fixture(Path(d))
        a = _args(Path(d), "sparse", "sp")
        inputs = _inputs(nf, rows)
        sparse, _ = C.harness_modules()
        q, _ = sparse.load_queries(a.queries, a.qcol)
        need = [set(range(nf * rows))] * 5
        fs, paths = C.corpus_paths(a.corpus, nf)
        ex = C.exact_scores_sparse(need, inputs["offsets"], paths, fs, "sp", q, "ip", threads=1)
        scores = sorted((ex[(0, g)] for g in range(nf * rows)), reverse=True)
        assert len(set(scores[:6])) < 6


@pytest.mark.parametrize("modality", ["sparse", "multivector"])
def test_a_row_every_system_missed_is_caught_with_the_exact_scan(tmp_path, monkeypatch, modality):
    """Both systems return the same list without the true best row: the union oracle cannot see
    it (both pass), the streamed exact scan can (both fail)."""
    nf, rows = (_sparse_fixture if modality == "sparse" else _mv_fixture)(tmp_path)
    a = _args(tmp_path, modality, "sp" if modality == "sparse" else "mv", "ip", k=3)
    a.tol, a.tie_ulps, a.rescore_threads = 1e-4, 128, 1
    inputs = _inputs(nf, rows, n=5)
    full = _oneshot(a, inputs, list(range(5)), nf)
    off = inputs["offsets"]
    need = [set(range(off[nf]))] * 5
    fs, paths = C.corpus_paths(a.corpus, nf)
    sparse, mv = C.harness_modules()
    mod = sparse if modality == "sparse" else mv
    q, _ = mod.load_queries(a.queries, a.qcol)
    fn = C.exact_scores_sparse if modality == "sparse" else C.exact_scores_mv
    ex = fn(need, off, paths, fs, a.vcol, q, "ip", threads=1)
    ids, sc = [], []
    for qq in range(5):
        order = sorted((g for g in range(off[nf]) if np.isfinite(ex[(qq, g)])),
                       key=lambda g: (-ex[(qq, g)], g))
        best = ex[(qq, order[0])] if order else -np.inf
        # drop every row tied with the best: a list that is wrong beyond the tie window
        rest = [g for g in order if ex[(qq, g)] < best - 0.5][:3]
        ids.append(np.asarray(rest + [-1] * (3 - len(rest)), np.int64))
        sc.append(np.asarray([ex[(qq, g)] for g in rest] + [-np.inf] * (3 - len(rest)), np.float32))
    inputs["systems"] = {"s1": (ids, sc), "s2": (list(ids), list(sc))}
    inputs["mapped"] = {}
    assert full
    on = agree.oracle_check(a, inputs, 5, exact_scan=True, short_guard=False)
    off_ = agree.oracle_check(a, inputs, 5, exact_scan=False, short_guard=False)
    assert all(v["ok"] for v in off_["systems"].values())          # the union oracle is blind
    assert not any(v["ok"] for v in on["systems"].values()), on["systems"]


def test_an_out_of_range_harness_id_is_a_list_defect(tmp_path, monkeypatch):
    p = tmp_path / "x.topk.npz"
    np.savez(p, I=np.asarray([[2, 1000, 1], [3, -1, -1], [-7, 0, 1]], np.int64),
             D=np.asarray([[0.9, 0.8, 0.5], [0.4, -np.inf, -np.inf], [0.9, 0.8, 0.1]], np.float32))
    ids, sc = C.load_npz(str(p), 5)
    assert ids[0].tolist() == [2, C.OUT_OF_RANGE, 1] and ids[1].tolist() == [3, -1, -1]
    assert ids[2].tolist() == [C.OUT_OF_RANGE, 0, 1]
    d = C.list_defects(ids, 3, False)
    assert d["failed"] and d["queries_with_out_of_range_ids"] == 2 and d["out_of_range_entries"] == 2
    assert d["queries_with_unmapped_entries"] == 0
    # a clean list's document is compare.py's, unchanged (no new keys)
    assert set(C.list_defects([np.asarray([1, 2, 3])], 3, False)) == {
        "queries_with_duplicate_ids", "duplicate_entries", "queries_with_unmapped_entries",
        "unmapped_entries", "failed", "examples"}
    # the oracle check runs (no IndexError) and blames the list
    ex = {0: 0.1, 1: 0.5, 2: 0.9, 3: 0.4, 4: 0.0}
    monkeypatch.setattr(agree, "_exact_for",
                        lambda a, need, inputs: {(q, g): ex[g] for q in range(3) for g in need[q]})
    a = types.SimpleNamespace(k=3, tol=1e-4, tie_ulps=128, modality="dense")
    inputs = {"systems": {"s": (ids, sc)}, "n": 3, "mapped": {}, "offsets": [0, 5]}
    o = agree.oracle_check(a, inputs, 3, short_guard=False)
    assert not o["systems"]["s"]["ok"]
    assert any("out-of-range" in f for f in o["systems"]["s"]["fails"])


def test_shared_score_rule_unit():
    from bench.verify import compare as C
    assert C.SHARED_SCORE_ATOL == agree.SCORE_ATOL and C.SHARED_SCORE_RTOL == agree.SCORE_RTOL
    assert not C.shared_score_bad(0.5, 0.5, 1e-5, 1e-5)
    assert not C.shared_score_bad(-np.inf, -np.inf, 1e-5, 1e-5)        # exact tie
    s = float(np.float32(0.7))
    assert not C.shared_score_bad(s, float(np.nextafter(np.float32(s), np.float32(1))), 1e-5, 1e-5)
    assert not C.shared_score_bad(1.0, 1.0 + 2 * (1e-5 + 1e-5) * 0.999, 1e-5, 1e-5)
    assert C.shared_score_bad(1.0, 1.0 + 2 * (1e-5 + 1e-5) * 1.01, 1e-5, 1e-5)
    assert C.shared_score_bad(float("nan"), float("nan"), 1e-5, 1e-5)
    assert C.shared_score_bad(-np.inf, 0.3, 1e-5, 1e-5)
