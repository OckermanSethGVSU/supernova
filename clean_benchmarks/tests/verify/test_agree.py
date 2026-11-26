"""bench/verify/agree.py (the checks step).

`bench run --dev` of the four fixture experiments (real nova-bf parquet + real harness npz, every
leg a real process) gives the inputs: the real checks.json of every shape, then mutated copies
of the runs, each of which must get the expected verdict from agree's pairs, oracle and rounds
checks. Then the oracle's and the scans' rules on scripted inputs.
"""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")
pytest.importorskip("nova_bf")

from bench.verify import agree  # noqa: E402
from bench.verify import compare as C  # noqa: E402
from helpers.agree_runs import _env, _exact_ties, _filtered_rows, _nb_results, _nb_rewrite, _npz_rewrite, _only_run, _uuid, bench, checks, copy_run, SHAPES  # noqa: E402
from helpers.agree_runs import runs  # noqa: F401,E402 (fixture)
from helpers.scan_fixtures import _args, _inputs, _mv_fixture, _oneshot, _sparse_fixture, _tiny_corpus  # noqa: E402

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


# ---------------------------------------------------------------------------------------------
# the real checks.json of bench run
# ---------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
def test_bench_run_writes_real_checks(runs, shape):
    d = runs[shape]
    ch = json.load(open(os.path.join(d, "checks.json")))
    assert ch["implemented"] is True and ch["ok"] is True, ch["problems"]
    exp = json.load(open(os.path.join(d, "experiment.json")))
    for s in exp["searches"]:
        for r in ("r1", "r2"):
            x = ch["searches"][s["name"]][r]
            assert x["ok"] and x["pairs"]["ok"] and x["oracle"]["ok"]
            assert set(x["systems"]) == set(s["systems"])
            assert set(x["oracle"]["systems"]) == set(s["systems"])
            assert x["oracle"]["sample"]["queries"] > 0
            assert os.path.isfile(os.path.join(d, x["log"]))
        assert ch["rounds"][s["name"]] == {sy: {"r2": "IDENTICAL"} for sy in s["systems"]}
    assert ch["e2e"] == ("IDENTICAL" if shape == "dense" else None)
    assert not ch["skipped"]


# ---------------------------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------------------------
def _agree_verdict(d, search="dense", r="r1"):
    ch = checks(d)
    return ch, ch["searches"][search][r]


@pytest.mark.slow
def test_swapped_ids_are_caught(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")

    def fn(I, D):
        I[0, 0], I[0, 9] = I[0, 9], I[0, 0]    # ids swapped, scores left: ids carry wrong scores
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    ch, x = _agree_verdict(d)
    assert not x["pairs"]["ok"] and not ch["ok"]
    assert x["pairs"]["compare"]["pairs"]["novabf_vs_cpu"]["f64_order_real_disagreements"] >= 1
    assert not x["oracle"]["systems"]["cpu"]["ok"] and x["oracle"]["systems"]["novabf"]["ok"]


@pytest.mark.slow
def test_a_wrong_score_beyond_tolerance_is_caught_by_the_oracle(runs, tmp_path):
    """agree's pairs check every shared id's score on every query, so the pairs fail it, and the
    oracle's --verify score bar fails it."""
    d = copy_run(runs["dense"], tmp_path / "run")

    def fn(I, D):
        D[3, 4] -= 1e-3
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    ch, x = _agree_verdict(d)
    assert not x["pairs"]["ok"]                 # the shared-id score check
    assert any("shared-id score mismatches" in f for f in x["pairs"]["fails"])
    o = x["oracle"]["systems"]["cpu"]
    assert not o["ok"] and o["scores_out_of_tolerance"] == 1 and o["score_examples"][0]["query"] == 3
    assert x["oracle"]["systems"]["novabf"]["ok"] and not ch["ok"]


@pytest.mark.slow
def test_a_score_inside_the_tolerance_passes(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")

    def fn(I, D):
        D[3, 4] -= 2e-6     # < 1e-5 + 1e-5 |s|, and not enough to break the order
        return I, D
    _npz_rewrite(d, "dense", "cpu", fn)
    ch, x = _agree_verdict(d)
    assert x["ok"] and ch["ok"], ch["problems"]


@pytest.mark.slow
def test_a_missing_query_is_caught(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    _npz_rewrite(d, "dense", "cpu", lambda I, D: (I[:-1], D[:-1]))
    ch, x = _agree_verdict(d)
    assert not x["ok"] and "not the same queries" in " ".join(x["fails"]) and not ch["ok"]


@pytest.mark.slow
def test_a_missing_novabf_query_is_caught(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    for p in _nb_results(d, "dense"):
        t = pq.read_table(p)
        pq.write_table(t.slice(0, t.num_rows - 1), p)
    ch, x = _agree_verdict(d)
    assert not x["ok"] and not ch["ok"]


@pytest.mark.slow
def test_a_tie_broken_differently_passes_and_a_real_swap_fails(runs, tmp_path):
    ties, ex = _exact_ties(runs["dense"], "dense", "cpu", 10)
    assert ties, "the fixture has exactly tied rows"
    q, r = ties[0]

    def swap(qq, a, b):
        def fn(I, D):
            I[qq, a], I[qq, b] = I[qq, b], I[qq, a]
            D[qq, a], D[qq, b] = D[qq, b], D[qq, a]
            return I, D
        return fn
    d = copy_run(runs["dense"], tmp_path / "tie")
    _npz_rewrite(d, "dense", "cpu", swap(q, r, r + 1))    # the other order of an exact tie
    ch, x = _agree_verdict(d)
    assert x["ok"] and ch["ok"], ch["problems"]
    # a swap of two clearly different scores (rank 0 and rank 9, scores moved with the ids)
    z = np.load(os.path.join(runs["dense"], "legs", "dense__cpu__r1.topk.npz"))
    q2 = next(i for i in range(len(z["I"])) if ex[(i, int(z["I"][i, 0]))] - ex[(i, int(z["I"][i, 9]))] > 1e-3)
    d = copy_run(runs["dense"], tmp_path / "real")
    _npz_rewrite(d, "dense", "cpu", swap(q2, 0, 9))
    ch, x = _agree_verdict(d)
    assert not x["pairs"]["ok"] and not x["oracle"]["systems"]["cpu"]["ok"] and not ch["ok"]


@pytest.mark.slow
def test_novabf_tie_at_the_cutoff_broken_the_other_way_passes(runs, tmp_path):
    """Where nova-bf and the cpu reference keep different, exactly tied ids at the cutoff,
    nova-bf keeping the cpu's choice instead is just as right; a clearly worse id is not."""
    d0 = runs["dense"]
    ch0 = json.load(open(os.path.join(d0, "checks.json")))
    assert ch0["searches"]["dense"]["r1"]["pairs"]["compare"]["pairs"]["novabf_vs_cpu"]["f64_tie_zone_differences"] > 0
    z = np.load(os.path.join(d0, "legs", "dense__cpu__r1.topk.npz"))
    nb = pq.read_table(_nb_results(d0, "dense")[0])
    exp = json.load(open(os.path.join(d0, "experiment.json")))
    keys, rows, offsets, _ = C.corpus_index(agree._abs(exp["corpus"]["path"]), need_ids=False)
    tab = C.uuid_table(keys, rows, offsets)
    u2g = {u: g for g, u in enumerate(tab.to_pylist())}   # uuid -> global row (test lookups)
    g2u = {g: u for u, g in u2g.items()}
    _, nb_ids, _ = C.load_novabf(_nb_results(d0, "dense")[0], tab)
    q = next(i for i in range(len(nb_ids)) if set(nb_ids[i]) != set(z["I"][i]))
    a = (set(nb_ids[q].tolist()) - set(z["I"][q].tolist())).pop()
    b = (set(z["I"][q].tolist()) - set(nb_ids[q].tolist())).pop()
    sc_b = float(z["D"][q][list(z["I"][q]).index(b)])

    def take_cpu_choice(ids, sc):
        j = ids[q].index(g2u[a])
        ids[q][j], sc[q][j] = g2u[b], sc_b
        return ids, sc
    d = copy_run(d0, tmp_path / "tie")
    _nb_rewrite(d, "dense", take_cpu_choice)
    ch, x = _agree_verdict(d)
    # the rounds were changed alike; the cold e2e leg (untouched) now differs from round 1:
    # every pair / oracle check passes, the e2e DIFFERENT is a FAILURE (was a warning)
    assert x["ok"] and ch["problems"] == ["e2e vs dense r1 result: DIFFERENT"] and not ch["ok"]
    assert ch["e2e"] == "DIFFERENT" and not ch["warnings"]
    assert all(rd["ok"] for per in ch["searches"].values() for rd in per.values())
    # the k-th id replaced by a low-scoring row nobody else holds
    worst = int(z["I"][q][-1]) + 1
    while worst in set(z["I"][q].tolist()) | set(nb_ids[q].tolist()):
        worst += 1

    def take_bad(ids, sc):
        ids[q][-1] = g2u[worst]
        return ids, sc
    d = copy_run(d0, tmp_path / "bad")
    _nb_rewrite(d, "dense", take_bad)
    ch, x = _agree_verdict(d)
    assert not x["pairs"]["ok"] and not x["oracle"]["systems"]["novabf"]["ok"] and not ch["ok"]


@pytest.mark.slow
@pytest.mark.parametrize("how", ["shifted_row", "unknown"])
def test_a_wrong_novabf_id_mapping_is_caught(runs, tmp_path, how):
    d0 = runs["dense"]
    exp = json.load(open(os.path.join(d0, "experiment.json")))
    keys, rows, offsets, _ = C.corpus_index(agree._abs(exp["corpus"]["path"]), need_ids=False)
    tab = C.uuid_table(keys, rows, offsets)
    u2g = {u: g for g, u in enumerate(tab.to_pylist())}   # uuid -> global row (test lookups)
    f_of = {g: (keys[f], g - int(offsets[f])) for f in range(len(keys)) for g in range(offsets[f], offsets[f + 1])}

    def fn(ids, sc):
        out = []
        for row in ids:
            if how == "shifted_row":      # md5 of key:row+1 (an off-by-one row mapping)
                out.append([_uuid(f_of[u2g[u]][0], f_of[u2g[u]][1] + 1) for u in row])
            else:                         # an id from another corpus root
                out.append([_uuid("/elsewhere/" + os.path.basename(f_of[u2g[u]][0]), f_of[u2g[u]][1]) for u in row])
        return out, sc
    d = copy_run(d0, tmp_path / "run")
    _nb_rewrite(d, "dense", fn)
    ch, x = _agree_verdict(d)
    assert not x["pairs"]["ok"] and not ch["ok"]
    if how == "unknown":
        assert x["pairs"]["compare"]["system_defects"]["novabf"]["failed"]


@pytest.mark.slow
def test_a_wrong_point_id_mapping_is_caught(runs, tmp_path):
    """filtered: nova-bf's hits are its point ids (no corpus id_column,
    --novabf-ids uuid): each query given the NEXT query's point ids (real ids, wrong rows) fails."""
    d = copy_run(runs["filtered"], tmp_path / "run")
    _nb_rewrite(d, "static_struct", lambda ids, sc: (ids[1:] + ids[:1], sc))
    ch, x = _agree_verdict(d, search="static_struct")
    assert not x["pairs"]["ok"] and not ch["ok"]


@pytest.mark.slow
def test_a_filtered_result_holding_a_masked_row_is_caught(runs, tmp_path):
    """A harness list with a row its filter excludes. compare.py judges a
    filtered search as dense: the masked row beats nova-bf's k-th, so the pair FAILS (compare
    blames the list that lacks the better row: filter compliance itself was the harness
    --verify's job, `bench check-verify`)."""
    d0 = runs["filtered"]
    exp = json.load(open(os.path.join(d0, "experiment.json")))
    search = "static_struct"
    z = np.load(os.path.join(d0, "legs", f"{search}__masked_bitmap__r1.topk.npz"))
    rows = _filtered_rows(exp, search)
    qt = pq.read_table(agree._abs(exp["queries"]["path"]))
    sel = [i for i, v in enumerate(qt.column(rows["column"]).to_pylist()) if str(v) in set(map(str, rows["isin"]))]
    Q = np.asarray(qt.column(exp["queries"]["column"]).to_pylist(), np.float64)[sel]
    files = sorted(os.path.join(agree._abs(exp["corpus"]["path"]), f)
                   for f in os.listdir(agree._abs(exp["corpus"]["path"])) if f.endswith(".parquet"))
    X = np.concatenate([np.asarray(pq.read_table(f, columns=[exp["corpus"]["column"]]).column(0).to_pylist(), np.float64)
                        for f in files])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    q = 0
    s = Xn @ (Q[q] / np.linalg.norm(Q[q]))
    kept = set(int(i) for i in z["I"][q] if i >= 0)
    masked = next(int(g) for g in np.argsort(-s, kind="stable") if int(g) not in kept)
    assert s[masked] > s[int(z["I"][q][len(kept) - 1])]   # it would be in an unfiltered top-k

    def fn(I, D):
        I[q, 1:] = I[q, :-1].copy()
        D[q, 1:] = D[q, :-1].copy()
        I[q, 0], D[q, 0] = masked, np.float32(s[masked])
        return I, D
    d = copy_run(d0, tmp_path / "run")
    _npz_rewrite(d, search, "masked_bitmap", fn)
    ch, x = _agree_verdict(d, search=search)
    assert not x["pairs"]["ok"] and not ch["ok"]
    # the ORACLE drops the masked candidate (the harness's own filter
    # evaluation), so the system that returned it is the one blamed, never nova-bf
    o = x["oracle"]
    assert o["filter_masked_candidates"] >= 1
    assert o["systems"]["masked_bitmap"]["ok"] is False
    assert o["systems"]["masked_bitmap"]["compare"]["f64_wrong_by_system"]["masked_bitmap"] >= 1
    assert o["systems"]["novabf"]["ok"] is True, o["systems"]["novabf"]["fails"]


@pytest.mark.slow
@pytest.mark.parametrize("shape,search", [("sparse", "sparse"), ("multivector", "multivector")])
def test_swapped_ids_are_caught_in_the_ragged_modalities(runs, tmp_path, shape, search):
    d = copy_run(runs[shape], tmp_path / "run")
    z = np.load(os.path.join(d, "legs", f"{search}__cpu__r1.topk.npz"))
    q = next(i for i in range(len(z["I"])) if z["D"][i, 0] - z["D"][i, 9] > 1e-2 and z["I"][i, 9] >= 0)

    def fn(I, D):
        I[q, 0], I[q, 9] = I[q, 9], I[q, 0]
        return I, D
    _npz_rewrite(d, search, "cpu", fn)
    ch, x = _agree_verdict(d, search=search)
    assert not x["pairs"]["ok"] and not ch["ok"]


@pytest.mark.slow
def test_round_differences_are_warnings(runs, tmp_path):
    """a round-to-round difference is recorded as a warning; ok stays true."""
    d = copy_run(runs["sparse"], tmp_path / "run")
    p = os.path.join(d, "legs", "sparse__cpu__r2.topk.npz")
    z = dict(np.load(p))
    z["D"] = z["D"].copy()
    z["D"][0, 0] = np.nextafter(z["D"][0, 0], np.float32(-np.inf))   # one ULP: still a valid result
    np.savez(p, **z)
    ch = checks(d)
    assert ch["rounds"]["sparse"]["cpu"]["r2"] == "DIFFERENT" and ch["ok"], ch["problems"]
    assert ch["warnings"] == ["rounds sparse cpu: r1 vs r2 DIFFERENT"] and not ch["problems"]
    assert ch["searches"]["sparse"]["r2"]["ok"]       # the round itself still checks out
    from bench.report import summary as SUM
    with open(os.path.join(d, "checks.json"), "w") as f:
        json.dump(ch, f)
    rows, problems, excluded, e2e = SUM.summarize(d)
    assert not problems
    assert "rounds sparse cpu: r1 vs r2 DIFFERENT" in SUM.write(d, rows, problems, excluded, e2e)


@pytest.mark.slow
def test_round_comparison_that_cannot_run_fails(runs, tmp_path):
    d = copy_run(runs["sparse"], tmp_path / "run")
    os.remove(os.path.join(d, "legs", "sparse__cpu__r2.topk.npz"))
    ch = checks(d)
    assert ch["rounds"]["sparse"]["cpu"]["r2"] == "ERROR" and not ch["ok"]
    assert any("could not be compared" in p for p in ch["problems"])


@pytest.mark.slow
def test_a_failed_leg_is_skipped_not_checked(runs, tmp_path):
    d = copy_run(runs["dense"], tmp_path / "run")
    p = os.path.join(d, "legs", "dense__cpu__r2.json")
    rec = json.load(open(p))
    rec["ok"] = False
    json.dump(rec, open(p, "w"))
    ch = checks(d)
    assert ch["skipped"] == [{"leg": "dense__cpu__r2", "why": "leg failed (summary lists it)"}]
    assert ch["searches"]["dense"]["r2"]["systems"] == ["novabf"] and ch["ok"]


def test_oracle_sample_rule():
    """precision.py: default_rng(0).choice(n, min(N, n), replace=False) | the first 500 queries
    whose sets differ."""
    n = 50
    I = [np.arange(q, q + 5) for q in range(n)]
    J = [x.copy() for x in I]
    for q in (3, 40):
        J[q][-1] = 999
    systems = {"a": (I, None), "b": (J, None), "gt": ([np.zeros(5, int)] * n, None)}
    s, info = agree.oracle_sample(systems, n, 5, 10)
    want = sorted(set(np.random.default_rng(0).choice(n, size=10, replace=False).tolist()) | {3, 40})
    assert s == want and info["differing"] == 2 and info["random"] == 10
    s2, _ = agree.oracle_sample(systems, n, 5, 500)
    assert s2 == list(range(n))
    # differing queries chosen OUTSIDE the random sample are still in (not by luck)
    rnd = set(np.random.default_rng(0).choice(n, size=10, replace=False).tolist())
    outside = [q for q in range(n) if q not in rnd][:3]
    J = [x.copy() for x in I]
    for q in outside:
        J[q][0] = 999
    s3, info3 = agree.oracle_sample({"a": (I, None), "b": (J, None)}, n, 5, 10)
    assert set(outside) <= set(s3) and set(s3) == rnd | set(outside) and len(s3) == 13
    assert info3["differing"] == 3
    # nq > oracle_queries: a strict subset, and only the first ORACLE_MAX_DIFF differing ones
    n = 2 * agree.ORACLE_MAX_DIFF + 300   # 650 differing (every odd query)
    I = [np.arange(q, q + 5) for q in range(n)]
    J = [x.copy() for x in I]
    for q in range(1, n, 2):
        J[q][0] = -7 - q
    diff = list(range(1, n, 2))
    s4, info4 = agree.oracle_sample({"a": (I, None), "b": (J, None)}, n, 5, 20)
    rnd = set(np.random.default_rng(0).choice(n, size=20, replace=False).tolist())
    assert set(s4) == rnd | set(diff[:agree.ORACLE_MAX_DIFF]) and len(s4) < n
    assert info4 == {"random": 20, "differing": len(diff), "differing_included": agree.ORACLE_MAX_DIFF,
                     "seed": 0}
    assert diff[agree.ORACLE_MAX_DIFF] not in s4 or diff[agree.ORACLE_MAX_DIFF] in rnd


SITE = r"""
import os, sys
if any(a.endswith("harness_cli.py") for a in sys.argv[:1]) and "dense__cpu__r1" in " ".join(sys.argv):
    import numpy as np
    _savez = np.savez
    def savez(path, *a, **kw):
        if "I" in kw and str(path).endswith(".topk.npz"):
            I = kw["I"].copy()
            I[0, 0], I[0, 9] = I[0, 9], I[0, 0]
            kw["I"] = I
        return _savez(path, *a, **kw)
    np.savez = savez
"""


@pytest.mark.slow
def test_a_mutated_leg_fails_bench_run(tmp_path):
    """bench run --dev with one harness leg whose saved top-k is corrupted inside the leg process
    (a sitecustomize on PYTHONPATH swaps two ids as the harness saves them): the run fails on the
    checks, and only on them."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITE)
    env = _env()
    env["PYTHONPATH"] = str(site) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = str(tmp_path / "res")
    p = bench("run", os.path.join(FIX, "dense.yaml"), "--dev", "--rounds", "1", "--results", res, env=env)
    assert p.returncode == 1, p.stdout[-3000:]
    d = _only_run(res, "s4_dense")
    ch = json.load(open(os.path.join(d, "checks.json")))
    assert ch["ok"] is False and any(x.startswith("dense r1:") for x in ch["problems"])
    assert all(json.load(open(os.path.join(d, "legs", f)))["ok"]
               for f in os.listdir(os.path.join(d, "legs")) if f.endswith("r1.json"))
    assert "checks failed" in p.stdout


# ---------------------------------------------------------------------------------------------
# the oracle's own rules on scripted inputs (no run needed)
# ---------------------------------------------------------------------------------------------
def _oracle(monkeypatch, lists, exact, k=2):
    """oracle_check on one query; `lists` = {system: (ids, scores)}; `exact` = the float64
    rescoring (agree._exact_for replaced)."""
    import types
    monkeypatch.setattr(agree, "_exact_for", lambda a, need, inputs: {(0, g): exact[g] for g in need[0]})
    a = types.SimpleNamespace(k=k, tol=1e-4, tie_ulps=128, modality="dense")
    inputs = {"systems": {nm: ([np.asarray(i, np.int64)], [np.asarray(sc, np.float32)])
                          for nm, (i, sc) in lists.items()}, "n": 1, "mapped": {}}
    return agree.oracle_check(a, inputs, 1)


def test_the_oracle_breaks_exact_ties_to_the_lowest_id(monkeypatch):
    """rows 7 and 9 are exact duplicates at the cutoff: the oracle keeps 7, the lowest id;
    the system holding 9 instead is inside the tie window and passes too."""
    exact = {5: 0.9, 7: 0.5, 9: 0.5}
    o = _oracle(monkeypatch, {"s1": ([5, 9], [0.9, 0.5]), "s2": ([5, 7], [0.9, 0.5])}, exact)
    assert o["oracle_lists_head"]["0"] == [5, 7]
    assert o["systems"]["s1"]["ok"] and o["systems"]["s2"]["ok"]


def test_a_wrong_id_at_the_cutoff_is_blamed_on_the_system(monkeypatch):
    """s1 differs from the oracle only at the k-th place, by far more than the tie
    window: f64_wrong_by_system names s1, not s2."""
    exact = {5: 0.9, 7: 0.5, 11: 0.3}
    o = _oracle(monkeypatch, {"s1": ([5, 11], [0.9, 0.3]), "s2": ([5, 7], [0.9, 0.5])}, exact)
    assert o["systems"]["s1"]["compare"]["f64_wrong_by_system"]["s1"] >= 1
    assert o["systems"]["s2"]["compare"]["f64_wrong_by_system"]["s2"] == 0
    assert not o["systems"]["s1"]["ok"] and o["systems"]["s2"]["ok"]


def test_a_list_out_of_order_fails_the_oracle_order_check(monkeypatch):
    """the right ids with their right scores, listed in the wrong order."""
    exact = {5: 0.9, 7: 0.5}
    o = _oracle(monkeypatch, {"s1": ([7, 5], [0.5, 0.9])}, exact)
    assert any("ORDER" in f for f in o["systems"]["s1"]["fails"]), o["systems"]["s1"]["fails"]


def test_a_duplicate_id_is_a_list_defect():
    """a duplicate id fails the oracle check as a list defect."""
    d = C.list_defects([np.asarray([5, 5, 7])], 3, False)
    assert d["failed"] and d["queries_with_duplicate_ids"] == 1
    assert not C.list_defects([np.asarray([5, 6, 7])], 3, False)["failed"]


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["dense", "sparse", "multivector", "filtered"])
def test_exact_scan_passes_on_every_shape(runs, shape, tmp_path):
    """`checks.exact_scan: true` re-checks a correct run against a full float64
    scan of its files: every round still passes, and the scan really ran."""
    d = runs[shape]
    exp = json.load(open(os.path.join(d, "experiment.json")))
    exp["checks"] = {**(exp.get("checks") or {}), "exact_scan": True}
    ch = agree.run_checks(d, exp)
    assert ch["ok"], ch["problems"]
    assert ch["settings"]["exact_scan"] is True
    for per in ch["searches"].values():
        for rd in per.values():
            es = rd["oracle"]["exact_scan"]
            assert es["on"] and es["rows"] > 0 and es["candidates"] > 0
            assert rd["oracle"]["short_guard"]["on"] is (exp["modality"] in ("dense", "multivector"))


@pytest.mark.slow
def test_filtered_novabf_runs_on_position_ids_and_the_shared_io(runs):
    """the filtered nova-bf config has no corpus id_column, so its hit ids
    are point ids mapped to rows as for dense (`--novabf-ids uuid`), and every check passes; its
    I/O comes from the experiment's `io:` like every other experiment's."""
    d = runs["filtered"]
    ch = checks(d)
    assert ch["ok"], ch["problems"]
    for per in ch["searches"].values():
        for rd in per.values():
            argv = rd["compare_argv"]
            assert argv[argv.index("--novabf-ids") + 1] == "uuid"
            assert rd["pairs"]["ok"] and rd["oracle"]["ok"]
    rec = json.load(open(os.path.join(d, "legs", "filtered_text__novabf__r1.json")))
    assert rec["env_added"]["NOVABF_IO_WORKERS"] == "2"
    assert rec["env_added"]["NOVABF_RUST_FETCH_CONC"] == "64"
    assert rec["env_added"]["NOVABF_RUST_FETCH_PART_MIB"] == "16"


@pytest.mark.parametrize("cells", [1 << 23, 2])   # one chunk per file / a chunk per row or two
@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("modality", ["sparse", "multivector"])
def test_the_streamed_scan_equals_the_one_shot_reference(tmp_path, monkeypatch, modality, metric,
                                                         cells):
    nf, rows = (_sparse_fixture if modality == "sparse" else _mv_fixture)(tmp_path)
    a = _args(tmp_path, modality, "sp" if modality == "sparse" else "mv", metric)
    inputs = _inputs(nf, rows)
    monkeypatch.setattr(agree, "EXACT_SCAN_CELLS", cells)
    reads = []
    orig = pq.read_table

    def spy(p, *aa, **kw):
        reads.append(str(p))
        return orig(p, *aa, **kw)
    monkeypatch.setattr(pq, "read_table", spy)
    for files in (nf, 2):
        reads.clear()
        got = agree.exact_scan_candidates(a, inputs, [0, 2, 4], files)
        assert [os.path.basename(r) for r in reads if "/c/" in r] == [f"f{i}.parquet" for i in range(files)]
        monkeypatch.setattr(pq, "read_table", orig)
        assert got == _oneshot(a, inputs, [0, 2, 4], files), (files, got)
        monkeypatch.setattr(pq, "read_table", spy)


@pytest.mark.parametrize("static", [False, True])
def test_the_filtered_scan_applies_the_mask_before_the_top_k(tmp_path, static):
    from helpers.scan_fixtures import _tiny_corpus
    X, Q = _tiny_corpus(tmp_path, n_files=3, rows=7)
    a = _args(tmp_path, "dense", "emb", "cosine")
    inputs = _inputs(3, 7, n=6)
    rng = np.random.default_rng(3)
    allowed = rng.random((6, 21)) < 0.4
    if static:
        allowed[:] = allowed[0]
    calls = []

    def mask_fn(f, sample):
        calls.append(f)
        m = allowed[np.asarray(sample)][:, 7 * f:7 * f + 7]
        return m[:1] if static else m
    sample = [0, 3, 5]
    got = agree.exact_scan_candidates(a, inputs, sample, None, filtered=True, mask_fn=mask_fn)
    assert calls == [0, 1, 2]
    want = _oneshot(a, inputs, sample, 3, allowed=allowed)
    assert got == want
    unmasked = _oneshot(a, inputs, sample, 3)
    assert got != unmasked                      # the mask changed the answer
    for q in sample:
        assert all(allowed[q][g] for g in got[q])
    with pytest.raises(agree.CheckError, match="filter mask"):
        agree.exact_scan_candidates(a, inputs, sample, None, filtered=True)


def test_there_is_no_pair_refusal_any_more():
    assert not hasattr(agree, "EXACT_SCAN_MAX_PAIRS")


@pytest.mark.parametrize("where", ["none", "queries", "first_file", "second_file"])
def test_negative_sparse_weights_are_a_check_failure(tmp_path, where):
    agree._CACHE.clear()
    (tmp_path / "c").mkdir()

    def write(path, vals):
        arr = pa.StructArray.from_arrays(
            [pa.array([[1, 2]] * len(vals), pa.list_(pa.uint32())),
             pa.array(vals, pa.list_(pa.float32()))], ["indices", "values"])
        pq.write_table(pa.table({"sp": arr}), path)
    ok = [[1.0, 2.0], [0.5, 0.0]]
    bad = [[1.0, -2.0], [0.5, 0.0]]
    write(tmp_path / "c" / "f0.parquet", bad if where == "first_file" else ok)
    write(tmp_path / "c" / "f1.parquet", bad if where == "second_file" else ok)
    write(tmp_path / "q.parquet", bad if where == "queries" else ok)
    a = _args(tmp_path, "sparse", "sp")
    got = agree.sparse_weight_problems(a, {"keys": ["k0", "k1"]})
    agree._CACHE.clear()
    if where in ("none", "second_file"):   # the corpus check reads the FIRST file only (documented)
        assert got == []
    else:
        assert len(got) == 1 and "negative" in got[0] and "non-negative" in got[0]


EXACT = {1: 0.9, 2: 0.8, 3: 0.5, 4: 0.1}      # corpus rows 1..4 (+ row 0 at 0.0); top-2 = [1, 2]
CASES = {
    "A": {"s": ([2, 3], [0.8, 0.5])},
    "B": {"s": ([2, -1], [0.8, -np.inf])},
    "C": {"s1": ([2, 3], [0.8, 0.5]), "s2": ([2, 3], [0.8, 0.5])},
    "D": {"s1": ([2, 3], [0.8, 0.5]), "s2": ([1, 2], [0.9, 0.8])},
    "G": {"s1": ([-1, -1], [-np.inf, -np.inf]), "s2": ([-1, -1], [-np.inf, -np.inf])},
}


def _probe(monkeypatch, lists, exact_scan):
    import types
    ex = {0: 0.0, **EXACT}
    monkeypatch.setattr(agree, "_exact_for", lambda a, need, inputs: {(0, g): ex[g] for g in need[0]})
    # the full scan of the 5-row corpus: every row is a candidate (what the scan's top-k adds)
    monkeypatch.setattr(agree, "exact_scan_candidates",
                        lambda a, inputs, sample, nf, filtered=False: {q: {1, 2} for q in sample})
    a = types.SimpleNamespace(k=2, tol=1e-4, tie_ulps=128, modality="dense")
    inputs = {"systems": {nm: ([np.asarray(i, np.int64)], [np.asarray(sc, np.float32)])
                          for nm, (i, sc) in lists.items()}, "n": 1, "mapped": {},
              "offsets": [0, 5]}
    o = agree.oracle_check(a, inputs, 1, exact_scan=exact_scan)
    return {nm: v["ok"] for nm, v in o["systems"].items()}


@pytest.mark.parametrize("case", ["A", "B", "C", "G"])
def test_exact_scan_catches_what_the_union_oracle_cannot(monkeypatch, case):
    assert not any(_probe(monkeypatch, CASES[case], True).values())


def test_exact_scan_off_catches_only_the_short_lists(monkeypatch):
    got = {c: _probe(monkeypatch, CASES[c], False) for c in CASES}
    assert got["A"] == {"s": True} and got["C"] == {"s1": True, "s2": True}   # union oracle: blind
    assert got["B"] == {"s": False}                                          # the short-list guard
    assert got["G"] == {"s1": False, "s2": False}


@pytest.mark.parametrize("exact_scan", [False, True])
def test_the_control_case_works_either_way(monkeypatch, exact_scan):
    assert _probe(monkeypatch, CASES["D"], exact_scan) == {"s1": False, "s2": True}


def test_short_guard_is_dense_only_and_uses_min_k_rows():
    systems = {"a": ([np.asarray([3, -1, -1])], None)}
    assert agree.short_queries(systems, ["a"], 1, 3, 1) == (1, [])       # a 1-row corpus: 1 hit is full
    assert agree.short_queries(systems, ["a"], 1, 3, 10) == (3, [0])


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_exact_scan_candidates_are_the_true_top_k_of_the_subset(tmp_path, metric):
    import types
    X, Q = _tiny_corpus(tmp_path)
    a = types.SimpleNamespace(k=3, modality="dense", metric=metric, corpus=str(tmp_path / "c"),
                              vcol="emb", queries=str(tmp_path / "q.parquet"), qcol="emb",
                              query_rows=None)
    inputs = {"keys": ["k0", "k1", "k2"], "offsets": [0, 7, 14, 21], "n": 6,
              "qn": [str(i) for i in range(6)]}
    X64, Q64 = X.astype(np.float64), Q.astype(np.float64)
    if metric == "cosine":
        X64 /= np.linalg.norm(X64, axis=1, keepdims=True)
        Q64 /= np.linalg.norm(Q64, axis=1, keepdims=True)
    for nf in (None, 2):   # the whole corpus, a 2-file subset
        rows = 21 if nf is None else 14
        got = agree.exact_scan_candidates(a, inputs, [0, 4, 5], nf)
        for q in (0, 4, 5):
            want = set(np.argsort(-(X64[:rows] @ Q64[q]), kind="stable")[:3].tolist())
            assert got[q] == want


def test_the_oracle_fails_a_list_with_a_duplicate_id(monkeypatch):
    """a duplicate id is a list defect in the ORACLE check too (not only in pairs)."""
    import types
    monkeypatch.setattr(agree, "_exact_for",
                        lambda a, need, inputs: {(0, g): {5: 0.9, 7: 0.5}[g] for g in need[0]})
    a = types.SimpleNamespace(k=2, tol=1e-4, tie_ulps=128, modality="sparse")
    inputs = {"systems": {"s1": ([np.asarray([5, 5], np.int64)], [np.asarray([0.9, 0.9], np.float32)]),
                          "s2": ([np.asarray([5, 7], np.int64)], [np.asarray([0.9, 0.5], np.float32)])},
              "n": 1, "mapped": {}}
    o = agree.oracle_check(a, inputs, 1)
    assert "list defects (duplicate or unmapped ids)" in o["systems"]["s1"]["fails"]
    assert o["systems"]["s2"]["ok"]


def _mv_corpus(tmp_path, tokens_per_doc, q_tokens=(2, 2)):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(3)
    t = pa.list_(pa.list_(pa.float32()))
    (tmp_path / "c").mkdir()
    for f, tpd in enumerate(tokens_per_doc):
        docs = [[rng.standard_normal(4).astype(np.float32).tolist() for _ in range(n)] for n in tpd]
        pq.write_table(pa.table({"mv": pa.array(docs, t)}), tmp_path / "c" / f"f{f}.parquet")
    qs = [[rng.standard_normal(4).astype(np.float32).tolist() for _ in range(n)] for n in q_tokens]
    pq.write_table(pa.table({"mv": pa.array(qs, t)}), tmp_path / "q.parquet")


def _mv_args(tmp_path, k=5):
    import types
    return types.SimpleNamespace(k=k, modality="multivector", corpus=str(tmp_path / "c"),
                                 vcol="mv", queries=str(tmp_path / "q.parquet"), qcol="mv")


def _short_systems(n_hits, n_q=2, k=5):
    ids = [np.array(list(range(n_hits)) + [-1] * (k - n_hits)) for _ in range(n_q)]
    sc = [np.array([1.0] * n_hits + [-np.inf] * (k - n_hits)) for _ in range(n_q)]
    return {"a": (ids, sc), "b": (ids, sc)}


def test_multivector_both_systems_short_with_finite_documents_present_fails(tmp_path):
    from bench.verify import agree
    _mv_corpus(tmp_path, [[2, 3, 1, 2, 2], [1, 1, 2, 3, 1]])     # 10 documents, all with tokens
    inputs = {"keys": ["k0", "k1"], "offsets": [0, 5, 10], "n": 2}
    want, short = agree.mv_short_queries(_mv_args(tmp_path), inputs, _short_systems(3),
                                         ["a", "b"], 2, 5, None)
    assert want == 5 and short == [0, 1]
    # full lists: no corpus read needed, nothing short
    assert agree.mv_short_queries(_mv_args(tmp_path), inputs, _short_systems(5),
                                  ["a", "b"], 2, 5, None) == (5, [])


def test_multivector_guard_counts_only_documents_with_tokens_and_finite_hits(tmp_path):
    from bench.verify import agree
    _mv_corpus(tmp_path, [[2, 0, 0, 1, 0], [0, 0, 3, 0, 0]], q_tokens=(2, 0))  # 3 docs with tokens
    inputs = {"keys": ["k0", "k1"], "offsets": [0, 5, 10], "n": 2}
    a = _mv_args(tmp_path)
    assert agree.mv_short_queries(a, inputs, _short_systems(3), ["a", "b"], 2, 5, None) == (3, [])
    want, short = agree.mv_short_queries(a, inputs, _short_systems(2), ["a", "b"], 2, 5, None)
    assert want == 3 and short == [0]        # query 1 has no tokens: nothing can score
    # an id with a -inf score is not a hit (a zero-token filler)
    s = _short_systems(3)
    s["a"][1][0][:] = -np.inf
    s["b"][1][0][:] = -np.inf
    assert agree.mv_short_queries(a, inputs, s, ["a", "b"], 2, 5, None)[1] == [0]


def test_the_oracle_check_turns_the_guard_on_for_multivector():
    import inspect
    from bench.verify import agree
    src = inspect.getsource(agree.oracle_check) + inspect.getsource(agree.check_round)
    assert 'a.modality in ("dense", "multivector")' in src
    assert 'exp["modality"] in ("dense", "multivector")' in src


def test_the_checks_use_the_experiments_threads_not_omp_num_threads(monkeypatch):
    """SkyPilot sets OMP_NUM_THREADS=1 in every job: the checks take the experiment's `threads`
    (0 = every CPU) for Arrow and BLAS, for their own duration only."""
    import pyarrow as pa
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert agree.check_threads({"threads": 7}) == 7
    from bench.data.sources import usable_cpu_count
    assert agree.check_threads({"threads": 0}) == usable_cpu_count()
    before = pa.cpu_count()
    with agree._threads(3):
        assert pa.cpu_count() == 3 and pa.io_thread_count() >= 3
        from threadpoolctl import threadpool_info
        assert all(p["num_threads"] == 3 for p in threadpool_info())
    assert pa.cpu_count() == before


def test_the_checks_leave_the_process_environment_as_it_was(monkeypatch):
    """threadpoolctl sets KMP_DUPLICATE_LIB_OK on import: legs started after the checks from the
    same process must not inherit it."""
    import sys
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delitem(sys.modules, "threadpoolctl", raising=False)
    before = dict(os.environ)
    with agree._threads(2):
        pass
    assert dict(os.environ) == before


def _without_timing(x):
    """A checks document with its timing and the score-sharing / pair-mode bookkeeping taken out
    (what may differ between two runs of the same checks)."""
    import re
    if isinstance(x, dict):
        return {k: _without_timing(v) for k, v in x.items()
                if k not in ("log", "pair_mode") and not re.search(r"seconds|_s$|time", k)}
    if isinstance(x, list):
        return [_without_timing(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"\(\d+ s\)", "(N s)", x)
    return x


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
def test_the_faster_checks_give_the_old_checks_document(runs, tmp_path, shape):
    """numpy shared scores, row-group reads and shared scores (one pass per round, kept across
    rounds) change no verdict and no number: the checks document equals the old code's (the
    three switches off), every round and every system."""
    d = copy_run(runs[shape], tmp_path / "run")
    exp = json.load(open(os.path.join(d, "experiment.json")))
    docs = {}
    for new in (False, True):
        C.VECTOR_SHARED, C.ROWGROUP_READ, agree.SHARE_SCORES = new, new, new
        try:
            docs[new] = json.loads(json.dumps(agree.run_checks(d, exp), default=str))
        finally:
            C.VECTOR_SHARED = C.ROWGROUP_READ = agree.SHARE_SCORES = True
    assert _without_timing(docs[False]) == _without_timing(docs[True])
    assert docs[True]["ok"] == docs[False]["ok"]


def test_needed_rows_reads_only_the_holding_row_groups_with_the_same_values(tmp_path):
    """a file of 7 row groups: the selected rows read through their row groups equal the rows
    taken from the whole column, in the asked order; a row past the end is refused."""
    vec = pa.array([[float(i), float(-i)] for i in range(70)], pa.list_(pa.float32()))
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"v": vec}), p, row_group_size=10)
    import pyarrow.fs as pafs
    fs = pafs.LocalFileSystem()
    local = np.array([0, 9, 10, 33, 34, 69], np.int64)
    C.ROWGROUP_READ = True
    got = C._needed_rows(str(p), fs, "v", local).to_pylist()
    C.ROWGROUP_READ = False
    try:
        ref = C._needed_rows(str(p), fs, "v", local).to_pylist()
    finally:
        C.ROWGROUP_READ = True
    assert got == ref == [[float(i), float(-i)] for i in local]
    with pytest.raises(SystemExit, match="beyond"):
        C._needed_rows(str(p), fs, "v", np.array([70], np.int64))


def test_the_score_cache_computes_each_pair_once_and_returns_the_old_order(tmp_path, monkeypatch):
    """with C.SCORE_CACHE, a second call computes only the pairs it has no score for, and
    every call returns exactly its own pairs in the order computing them all would give."""
    import pyarrow.fs as pafs
    vec = pa.array([[float(i % 7), 1.0 + i] for i in range(40)], pa.list_(pa.float32()))
    for f in range(2):
        pq.write_table(pa.table({"v": vec.slice(20 * f, 20)}), tmp_path / f"p{f}.parquet", row_group_size=5)
    paths = [str(tmp_path / f"p{f}.parquet") for f in range(2)]
    offsets = np.array([0, 20, 40])
    xq = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    need1 = [{1, 25}, {3}, set()]
    need2 = [{1, 30}, {3, 21}, {39}]
    fs = pafs.LocalFileSystem()
    ref1 = C.exact_scores(need1, None, "v", offsets, paths, fs, xq, "cosine")
    ref2 = C.exact_scores(need2, None, "v", offsets, paths, fs, xq, "cosine")
    calls = []
    real = C._exact_scores
    monkeypatch.setattr(C, "_exact_scores", lambda need, *a: calls.append(sum(len(s) for s in need)) or real(need, *a))
    C.SCORE_CACHE = {}
    try:
        got1 = C.exact_scores(need1, None, "v", offsets, paths, fs, xq, "cosine", cache_key="k")
        got2 = C.exact_scores(need2, None, "v", offsets, paths, fs, xq, "cosine", cache_key="k")
    finally:
        C.SCORE_CACHE = None
    assert list(got1.items()) == list(ref1.items()) and list(got2.items()) == list(ref2.items())
    assert calls == [3, 3]          # the second call scores only (0,30), (1,21), (2,39)


def test_a_flat_listing_takes_only_the_roots_own_files_as_the_harness_does(tmp_path):
    """a corpus with files at the root and in a sub-folder: the flat listing (corpus.recursive
    false) equals the harness's list_parquet without recursion, the recursive one with it, and
    corpus_paths agrees with corpus_index either way."""
    from bench.data import sources
    for rel in ("a.parquet", "c.parquet", "sub/b.parquet", "sub/deeper/d.parquet"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": pa.array([rel], pa.string())}), p)
    (tmp_path / "notes.txt").write_text("x")
    for recursive in (False, True):
        keys, rows, _, _ = C.corpus_index(str(tmp_path), need_ids=False, recursive=recursive)
        _, infos = sources.list_parquet(str(tmp_path), recursive=recursive)
        harness = [i.path for i in infos]
        assert keys == harness, (recursive, keys, harness)
        _, paths = C.corpus_paths(str(tmp_path), len(keys), recursive=recursive)
        assert paths == keys
    assert len(C.corpus_index(str(tmp_path), need_ids=False, recursive=False)[0]) == 2


def test_the_memory_estimate_counts_only_the_row_groups_read(tmp_path):
    """with row-group sizes the estimate's decoded column is the row groups holding a needed
    row (what _needed_rows reads), never more than the whole-file estimate; without them, with
    ROWGROUP_READ off, or a footer that disagrees with the index, it is the whole file."""
    import pyarrow.fs as pafs
    for f in range(2):
        vec = pa.array([[float(i), 1.0] for i in range(100)], pa.list_(pa.float32()))
        pq.write_table(pa.table({"v": vec}), tmp_path / f"f{f}.parquet", row_group_size=10)
    paths = [str(tmp_path / f"f{f}.parquet") for f in range(2)]
    fs = pafs.LocalFileSystem()
    offsets = np.array([0, 100, 200])
    need = [{3, 5, 150}, {7}]                       # file 0: group 0; file 1: group 5
    files = C.needed_files(need, offsets)
    assert files == [0, 1]
    groups = C.row_group_rows(paths, fs, files)
    assert groups == {0: [10] * 10, 1: [10] * 10}
    assert C.rescore_bytes(need, offsets, 2, 4, 8, {0: None, 1: [10] * 10})[0] \
        == C.rescore_bytes(need, offsets, 2, 4, 8, {0: [100], 1: [10] * 10})[0]   # unreadable
    assert C.rescore_bytes(need, offsets, 2, 4, 8, groups, held_pairs=10)[0] \
        == C.rescore_bytes(need, offsets, 2, 4, 8, groups)[0] + 10 * C.PAIR_BYTES
    whole, _, _ = C.rescore_bytes(need, offsets, 2, 4, 8)
    part, n_g, n_f = C.rescore_bytes(need, offsets, 2, 4, 8, groups)
    assert (n_g, n_f) == (4, 2)
    col = (2 * 4 + 8) * C.COLUMN_COPIES            # one decoded row's bytes, every copy
    assert whole - part == (200 - 20) * col         # 180 fewer decoded rows
    assert C.rescore_bytes(need, offsets, 2, 4, 8, {0: [50, 50], 1: [10] * 10})[0] \
        == whole - (200 - 50 - 10) * col            # file 0: its first group of 50; file 1: 10
    assert C.rescore_bytes(need, offsets, 2, 4, 8, {0: [10] * 9, 1: [10] * 10})[0] \
        == whole - 90 * col                         # file 0's footer sums 90, not 100: whole
    C.ROWGROUP_READ = False
    try:
        assert C.row_group_rows(paths, fs, files) is None
        assert C.rescore_bytes(need, offsets, 2, 4, 8, groups)[0] == whole
    finally:
        C.ROWGROUP_READ = True


def test_a_flat_corpus_with_nested_files_is_refused_when_novabf_reads_it(tmp_path):
    """nova-bf lists every level: a flat experiment whose corpus has *.parquet files in
    sub-folders would have nova-bf read files the harness legs and the checks do not."""
    for rel in ("a.parquet", "sub/b.parquet"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": pa.array([rel], pa.string())}), p)
    with pytest.raises(SystemExit, match="1 \\*.parquet files sit in sub-folders"):
        C.refuse_flat_with_nested(str(tmp_path))
    (tmp_path / "sub" / "b.parquet").unlink()
    C.refuse_flat_with_nested(str(tmp_path))   # flat now: nothing to refuse


def test_unreadable_footers_fall_back_and_footers_are_read_once(tmp_path, monkeypatch):
    """row_group_rows reads each path's footer once per process; a footer that cannot be read
    maps to None (the estimate counts that file whole) instead of failing the check."""
    import pyarrow.fs as pafs
    p = tmp_path / "f.parquet"
    pq.write_table(pa.table({"v": pa.array([[1.0]] * 30, pa.list_(pa.float32()))}), p,
                   row_group_size=10)
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    monkeypatch.setattr(C, "_ROW_GROUPS", {})
    calls = []
    real = pq.ParquetFile
    monkeypatch.setattr(pq, "ParquetFile", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    paths, fs = [str(p), str(bad)], pafs.LocalFileSystem()
    assert C.row_group_rows(paths, fs, [0, 1]) == {0: [10, 10, 10], 1: None}
    assert C.row_group_rows(paths, fs, [0]) == {0: [10, 10, 10]}
    assert calls.count(str(p)) == 1


def test_the_rescoring_reads_fewer_files_at_once_to_fit_the_limit(tmp_path, monkeypatch):
    """dense_rescore_plan lowers the files read at once until the estimate fits the limit,
    estimates only the pairs the score cache does not hold (plus the cache and the table it
    returns), and returns the one-thread estimate when nothing fits (the caller refuses)."""
    import types
    import pyarrow.fs as pafs
    for f in range(4):
        vec = pa.array([[float(i), 1.0] for i in range(100)],
                       pa.list_(pa.float32()))
        pq.write_table(pa.table({"v": vec}), tmp_path / f"f{f}.parquet", row_group_size=100)
    paths = [str(tmp_path / f"f{f}.parquet") for f in range(4)]
    fs, offsets = pafs.LocalFileSystem(), np.array([0, 100, 200, 300, 400])
    need = [{5, 105, 205, 305}]
    a = types.SimpleNamespace(rescore_threads=4, rescore_max_gib=1.0, vcol="v", corpus="c",
                              metric="ip", queries="q", qcol="e", query_rows=None, max_files=None)
    monkeypatch.setattr(C, "SCORE_CACHE", None)
    est = {t: C.rescore_bytes(need, offsets, 2, 4, t, {f: [100] for f in range(4)})[0]
           for t in (1, 2, 3, 4)}
    assert C.dense_rescore_plan(a, need, offsets, paths, fs, 2)[:2] == (4, est[4])
    a.rescore_max_gib = (est[2] + 1) / 2**30            # 3 and 4 at once do not fit
    assert C.dense_rescore_plan(a, need, offsets, paths, fs, 2)[:2] == (2, est[2])
    a.rescore_max_gib = (est[1] - 1) / 2**30            # nothing fits: refused by the caller
    assert C.dense_rescore_plan(a, need, offsets, paths, fs, 2)[:2] == (1, est[1])
    a.rescore_max_gib = 1.0
    store = {(0, 5): 1.0, (0, 105): 1.0, (0, 205): 1.0, (0, 305): 1.0}
    monkeypatch.setattr(C, "SCORE_CACHE", {C.score_cache_key(a): store})
    t, e, n_g, n_f = C.dense_rescore_plan(a, need, offsets, paths, fs, 2)
    assert (n_g, n_f) == (0, 0) and e == (4 + 4) * C.PAIR_BYTES   # nothing left to read
    assert C.column_copies(2) == C.COLUMN_COPIES_F16 > C.column_copies(4) == C.COLUMN_COPIES


def test_the_ragged_estimates_count_the_measured_column_copies_and_fit_the_limit(tmp_path,
                                                                               monkeypatch):
    """Sparse / multivector: a decoded column costs ragged_column_copies of its stored bytes
    (measured up to 5.6, the old estimate assumed 2) and the plan lowers the files read at once
    until the estimate fits; agree's oracle rescoring refuses above the limit as judge does."""
    import types
    stats = {0: (100, 10_000), 1: (100, 10_000)}
    offsets = np.array([0, 100, 200])
    need = [{5, 105}]
    one = C.rescore_bytes_sparse(need, offsets, stats, 4, 4, 0, 1, 1)[0]
    assert one >= 10_000 * 8 * C.RAGGED_COLUMN_COPIES
    two = C.rescore_bytes_sparse(need, offsets, stats, 4, 4, 0, 1, 2)[0]
    assert two - one >= 10_000 * 8 * C.RAGGED_COLUMN_COPIES + C.READ_FIXED_BYTES
    mv = C.rescore_bytes_mv(need, offsets, stats, 2, 0, 4, 1, 16)[0]
    assert mv >= 10_000 * 2 * C.RAGGED_COLUMN_COPIES_F16
    assert C.ragged_column_copies(2) == C.RAGGED_COLUMN_COPIES_F16 > \
        C.ragged_column_copies(4) == C.RAGGED_COLUMN_COPIES
    # the plan: a real sparse file pair
    import pyarrow.fs as pafs
    for f in range(2):
        col = pa.array([{"indices": [i, i + 1], "values": [1.0, 2.0]} for i in range(100)],
                       pa.struct([("indices", pa.list_(pa.uint32())),
                                  ("values", pa.list_(pa.float32()))]))
        pq.write_table(pa.table({"sp": col}), tmp_path / f"f{f}.parquet")
    paths = [str(tmp_path / f"f{f}.parquet") for f in range(2)]
    fs = pafs.LocalFileSystem()
    q = types.SimpleNamespace(nnz=2, n_rows=1)
    a = types.SimpleNamespace(rescore_threads=2, rescore_max_gib=1.0, vcol="sp",
                              modality="sparse")
    t, e2, n_g, n_f, _ = C.ragged_rescore_plan(a, need, offsets, paths, fs, q)
    assert (t, n_g, n_f) == (2, 2, 2)
    a.rescore_max_gib = (e2 - 1) / 2**30                 # two at once do not fit
    t, e1, _, _, _ = C.ragged_rescore_plan(a, need, offsets, paths, fs, q)
    assert t == 1 and e1 < e2
    t, e1b, _, _, _ = C.ragged_rescore_plan(a, need, offsets, paths, fs, q, live=10)
    assert e1b == e1 + 10 * C.PAIR_BYTES
    a.rescore_max_gib = (e1 - 1) / 2**30                 # nothing fits: the caller refuses
    assert C.ragged_rescore_plan(a, need, offsets, paths, fs, q)[:2] == (1, e1)


@pytest.mark.parametrize("modality", ["sparse", "multivector"])
def test_the_oracles_ragged_rescoring_is_gated_and_reports_a_missing_column(tmp_path, modality):
    """agree's oracle rescoring of a sparse / multivector round goes through the same plan as
    judge's: within the limit it scores, above it a CheckError; a --vcol the files lack is a
    CheckError too (not a KeyError)."""
    nf, rows = (_sparse_fixture if modality == "sparse" else _mv_fixture)(tmp_path)
    a = _args(tmp_path, modality, "sp" if modality == "sparse" else "mv")
    inputs = _inputs(nf, rows)
    need = [{0, rows}] + [set() for _ in range(4)]
    assert set(agree._exact_for(a, need, inputs)) == {(0, 0), (0, rows)}
    a.rescore_max_gib = 1e-6
    with pytest.raises(agree.CheckError, match="oracle rescoring refused"):
        agree._exact_for(a, need, inputs)
    a.rescore_max_gib, a.vcol = 32.0, "nope"
    with pytest.raises(agree.CheckError, match="no such column"):
        agree._exact_for(a, need, inputs)


def test_the_pool_is_released_only_when_the_estimate_no_longer_fits(monkeypatch):
    """release_pool: with a budget (limit, estimated peak) the release is skipped while resident
    memory + the whole estimate fits under the limit, done when it does not; without a budget
    or a readable RSS it always releases."""
    calls = []

    class Pool:
        def release_unused(self):
            calls.append(1)
    monkeypatch.setattr(C.pa, "default_memory_pool", lambda: Pool())
    monkeypatch.setattr(C, "_rss_bytes", lambda: 6 << 30)
    C.release_pool((10 << 30, 4 << 30))          # 6 + 4 <= 10: room, skipped
    assert calls == []
    C.release_pool((10 << 30, (4 << 30) + 1))    # one byte over: released
    C.release_pool()                              # no budget: always
    monkeypatch.setattr(C, "_rss_bytes", lambda: None)
    C.release_pool((10 << 30, 1))                 # RSS unknown: always
    assert len(calls) == 3
    monkeypatch.undo()
    assert C._rss_bytes() > 0
