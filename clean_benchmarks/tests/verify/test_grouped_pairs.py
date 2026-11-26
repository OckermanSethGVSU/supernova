"""compare.judge's grouped pairs (a sweep, --group): every pair's document is computed per
query by identity (GroupedPairs: a member holding its representative's lists exactly on a query
takes the representative's outcome there, and is compared for real elsewhere), so the judge
document, the FAILs and runner.pairs_blame equal the all-pairs ones EXACTLY. On small synthetic
rounds the grouped document is compared with the all-pairs one byte for byte; a normal run (no
--group) is unchanged; the float64 rescoring covers the same (query, id) pairs as all pairs. A
tiny local dense corpus (cosine), no GPU, no network."""
import contextlib
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench import runner
from bench.verify import agree
from bench.verify import compare as C

K = 4
N_FILES, ROWS, D, NQ = 3, 7, 4, 6


@pytest.fixture
def corpus(tmp_path):
    """3 files x 7 rows; row 1 = row 0 (an exact tie) and query 0 = row 0, so rows 0 and 1 are
    query 0's top two with equal scores. -> (tmp_path, float64 exact cosine [query, row])."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((N_FILES * ROWS, D)).astype(np.float32)
    x[1] = x[0]
    q = rng.standard_normal((NQ, D)).astype(np.float32)
    q[0] = x[0]
    (tmp_path / "c").mkdir()
    for f in range(N_FILES):
        pq.write_table(pa.table({"emb": pa.array(list(x[f * ROWS:(f + 1) * ROWS]), pa.list_(pa.float32(), D))}),
                       tmp_path / "c" / f"f{f}.parquet")
    pq.write_table(pa.table({"emb": pa.array(list(q), pa.list_(pa.float32(), D))}), tmp_path / "q.parquet")
    x64, q64 = x.astype(np.float64), q.astype(np.float64)
    x64 /= np.linalg.norm(x64, axis=1, keepdims=True)
    q64 /= np.linalg.norm(q64, axis=1, keepdims=True)
    return tmp_path, q64 @ x64.T


def _f32_up(s):
    return float(np.nextafter(np.float32(s), np.float32(np.inf)))


def _lists(ex, kind):
    """One system's (ids, scores): the true top-K (ties to the lower id), or a variant: bad1
    (query 2's K-th id swapped for the worst row), bad2 (query 4's ranks 1 and 3 swapped: out
    of order), bad3 (query 3's K-th id swapped for the second-worst row), tie (query 0's exactly tied rows 0 and 1 swapped), ulp_some (queries 1 and 3's
    scores one float32 ULP up: correct, but not the same lists), ulp_all (every query's)."""
    ids, sc = [], []
    for qi in range(NQ):
        order = sorted(range(ex.shape[1]), key=lambda g: (-ex[qi, g], g))
        top = order[:K]
        if kind == "bad1" and qi == 2:
            top[-1] = order[-1]
        if kind == "bad3" and qi == 3:
            top[-1] = order[-2]
        if kind == "bad2" and qi == 4:
            top[0], top[2] = top[2], top[0]
        if kind == "tie" and qi == 0:
            assert top[:2] == [0, 1]
            top[0], top[1] = 1, 0
        s = [float(np.float32(ex[qi, g])) for g in top]
        if kind == "ulp_all" or (kind == "ulp_some" and qi in (1, 3)):
            s = [_f32_up(v) for v in s]
        ids.append(np.array(top, np.int64))
        sc.append(np.array(s, np.float64))
    return ids, sc


def _round(tmp_path, ex, spec, grouped, order=True):
    """judge on systems spec [(name, kind, group or None)] (round order; `gt` is the dataset
    GT). -> (judge document, fails, a, inputs)."""
    argv = ["--corpus", str(tmp_path / "c"), "--novabf", "-", "--rescore", "--vcol", "emb",
            "--queries", str(tmp_path / "q.parquet"), "--qcol", "emb", "--metric", "cosine",
            "--k", str(K), "--rescore-threads", "1"] + ([] if order else ["--no-order-check"])
    if grouped:
        argv += [z for nm, _, g in spec if g for z in ("--group", f"{nm}={g}")]
    a = C.parse_args(argv)
    systems = {nm: _lists(ex, kind) for nm, kind, _ in spec}
    inputs = {"keys": [f"k{i}" for i in range(N_FILES)],
              "offsets": np.array([ROWS * i for i in range(N_FILES + 1)]), "n_ids": N_FILES * ROWS,
              "dup": 0, "qn": [str(i) for i in range(NQ)], "systems": systems, "n": NQ,
              "unmapped": 0, "gt_missing": 0 if "gt" in systems else None, "mapped": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        out, fails = C.judge(a, inputs, 0.0)
    return out, fails, a, inputs


def _blamed(out, fails, spec):
    listed = [nm for nm, _, _ in spec if nm != "gt"]
    if not fails:
        return set()
    return set(runner.pairs_blame({"ok": False, "fails": fails, "compare": out}, listed))


GROUPED_KEYS = ("pair_mode", "group_of", "group_differing_queries", "grouped_compares")
FULL_KEYS = ("f64_order_full_queries", "f64_order_full_contradictions_by_system",
             "f64_order_full_unjudged_by_system", "f64_order_short_queries_by_system")


def _same_document(full, grp):
    """the grouped judge document is the all-pairs one, byte for byte (key order included),
    but its grouped-mode bookkeeping."""
    assert not set(GROUPED_KEYS) & set(full) and set(GROUPED_KEYS) <= set(grp)
    g = {k: v for k, v in grp.items() if k not in GROUPED_KEYS}
    assert json.dumps(full, default=str) == json.dumps(g, default=str)


CASES = {
    # (systems, blamed with the order check (the checks' default), blamed without it). A bad1
    # list also differs at rank K from a correct list by more than ties, a float64 ORDER
    # failure; the bad1 list keeps the worst row while lacking a better id, so its own list
    # contradicts the float64 scores (f64_order_contradictions_by_system), and the correct side
    # passes judge's full-rank order check (f64_order_full_*): runner.order_blame blames the
    # bad1 list alone, also when two systems are wrong (before, its common-system rule blamed
    # every side of those pairs, correct ones included).
    "all_identical": ([("x", "ok", "x"), ("x-1", "ok", "x"), ("x-2", "ok", "x"),
                       ("y", "ok", "y"), ("y-1", "ok", "y")], set(), set()),
    "all_correct": ([("x", "ok", "x"), ("x-1", "ok", "x"), ("x-2", "tie", "x"),
                     ("y", "ok", "y"), ("y-1", "ulp_some", "y")], set(), set()),
    "one_wrong": ([("x", "ok", "x"), ("x-1", "bad1", "x"), ("x-2", "ok", "x"),
                   ("y", "ok", "y")], {"x-1"}, {"x-1"}),
    "two_wrong_differently": ([("x", "ok", "x"), ("x-1", "bad1", "x"), ("x-2", "bad2", "x"),
                               ("x-3", "tie", "x"), ("y", "ok", "y")], {"x-1", "x-2"}, {"x-1"}),
    "rep_and_member_wrong_alike": ([("x", "bad1", "x"), ("x-1", "bad1", "x"), ("x-2", "ok", "x"),
                                    ("y", "ok", "y"), ("y-1", "tie", "y")],
                                   {"x", "x-1"}, {"x", "x-1"}),
    "rep_and_member_wrong_one_group_correct": ([("x", "bad1", "x"), ("x-1", "bad1", "x"),
                                                ("y", "ok", "y"), ("y-1", "ok", "y")],
                                               {"x", "x-1"}, {"x", "x-1"}),
    "member_ties_only": ([("x", "ok", "x"), ("x-1", "tie", "x"), ("y", "ok", "y")], set(), set()),
    "member_differs_everywhere": ([("x", "ok", "x"), ("x-1", "ulp_all", "x"), ("x-2", "bad1", "x"),
                                   ("y", "ok", "y")], {"x-2"}, {"x-2"}),
    "rep_differs_everywhere_from_members": ([("x", "ulp_all", "x"), ("x-1", "ok", "x"),
                                             ("x-2", "bad2", "x"), ("y", "ulp_some", "y")],
                                            {"x-2"}, set()),
    "gt_correct_one_wrong": ([("x", "ok", "x"), ("gt", "ok", None), ("x-1", "bad1", "x"),
                              ("y", "tie", "y")], {"x-1"}, {"x-1"}),
    "gt_wrong": ([("x", "ok", "x"), ("gt", "bad1", None), ("x-1", "tie", "x"),
                  ("y", "ok", "y")], set(), set()),
    "gt_wrong_alike_rep": ([("x", "bad1", "x"), ("gt", "bad1", None), ("x-1", "bad1", "x"),
                            ("y", "ok", "y")], {"x", "x-1"}, {"x", "x-1"}),
}


@pytest.mark.parametrize("order", [True, False], ids=["order", "no_order"])
@pytest.mark.parametrize("case", list(CASES))
def test_grouped_verdicts_equal_the_all_pairs_verdicts(corpus, case, order):
    """with --group (a sweep) the judge document, the FAILs and every system's blame equal the
    all-pairs ones exactly (and are the expected ones); a member compares for real only the
    queries where its lists are not its representative's."""
    tmp_path, ex = corpus
    spec, want_order, want_no_order = CASES[case]
    want = want_order if order else want_no_order
    full, full_fails, _, inputs = _round(tmp_path, ex, spec, grouped=False, order=order)
    grp, grp_fails, _, _ = _round(tmp_path, ex, spec, grouped=True, order=order)
    _same_document(full, grp)
    assert full_fails == grp_fails
    assert _blamed(full, full_fails, spec) == _blamed(grp, grp_fails, spec) == want
    st = grp["grouped_compares"]
    n_pairs = len(spec) * (len(spec) - 1) // 2
    assert st["pairs"] == n_pairs
    # the representatives' pairs (incl. a representative with itself) run once each, shared
    reps = {r for r in grp["group_of"].values()}
    assert st["full_compares"] <= len(reps) ** 2 + n_pairs
    for m, c in grp["group_differing_queries"].items():
        r = grp["group_of"][m]
        same = C.same_lists(inputs["systems"][m], inputs["systems"][r], NQ, K)
        assert c == int((~same).sum())
    if case == "all_identical":
        assert grp["group_differing_queries"] == {"x-1": 0, "x-2": 0, "y-1": 0}
        assert st["as_representatives"] == n_pairs and st["query_compares"] == 0
        assert st["full_compares"] == 3   # (x, x), (x, y), (y, y): shared by every pair
    if case == "member_differs_everywhere":
        assert grp["group_differing_queries"]["x-1"] == NQ
    if case in ("one_wrong", "all_correct", "two_wrong_differently"):
        assert st["query_compares"] > 0   # the per-query path (fold_queries) ran


# order_blame on real judge documents: (systems, order_blame's blame, pairs_blame's). Under
# the old rule (no per-side contradictions: the common-system rule) the order blames were:
# one_wrong {z}; two_wrong_differently {x, y, z}; two_wrong_alike {x}; ties_only none.
ORDER_BLAME = {
    "one_wrong": ([("x", "ok", "x"), ("y", "ok", "y"), ("z", "bad1", "z")], {"z"}, {"z"}),
    "two_wrong_differently": ([("x", "ok", "x"), ("y", "bad1", "y"), ("z", "bad3", "z")],
                              {"y", "z"}, {"y", "z"}),
    "two_wrong_alike": ([("x", "ok", "x"), ("y", "bad1", "y"), ("z", "bad1", "y")],
                        {"y", "z"}, {"y", "z"}),
    "ties_only": ([("x", "ok", "x"), ("y", "tie", "x"), ("z", "ulp_some", "z")], set(), set()),
}


def _order_blamed(out, spec):
    listed = [nm for nm, _, _ in spec if nm != "gt"]
    return set(runner.order_blame(out, listed))


@pytest.mark.parametrize("case", list(ORDER_BLAME))
def test_order_blame_names_only_the_wrong_sides(corpus, case):
    """each order-failed pair is blamed on the side whose own list contradicts the float64
    scores (f64_order_contradictions_by_system), so correct systems compared with two wrong
    ones are not blamed; grouped (a sweep) gives the all-pairs blame."""
    tmp_path, ex = corpus
    spec, want_order, want_pairs = ORDER_BLAME[case]
    full, full_fails, _, _ = _round(tmp_path, ex, spec, grouped=False)
    grp, grp_fails, _, _ = _round(tmp_path, ex, spec, grouped=True)
    _same_document(full, grp)
    assert _order_blamed(full, spec) == _order_blamed(grp, spec) == want_order
    assert _blamed(full, full_fails, spec) == _blamed(grp, grp_fails, spec) == want_pairs
    kind = {nm: k for nm, k, _ in spec}
    for pk, r in full["pairs"].items():
        con = r["f64_order_contradictions_by_system"]
        # a wrong side contradicts; two lists wrong alike do not differ, so neither does
        wrong = {nm for nm in con if kind[nm].startswith("bad")
                 and len({kind[s] for s in con}) == 2}
        assert {nm for nm, c in con.items() if c} == wrong, pk
        if case == "ties_only":
            assert r["f64_order_real_disagreements"] == 0
    if case != "ties_only":
        assert full["f64_order_failed_pairs"]


def test_a_real_disagreement_no_side_contradicts_keeps_the_old_blame():
    """the tie-window chain at k=2 (rows 1.5, 1.5-60u, 1.5-120u; window 100u): W = [0, 2] and
    B = [1, 0] differ at rank 1 by 120u (a real order disagreement) but neither list
    contradicts the float64 scores (row 1 is within the window of both others), so
    order_blame falls back to the common-system rule: both sides."""
    ex = {(0, g): float(CHAIN_VALS[g]) for g in range(3)}
    W = ([np.array([0, 2], np.int64)], [np.array([CHAIN_VALS[0], CHAIN_VALS[2]], np.float64)])
    B = ([np.array([1, 0], np.int64)], [np.array([CHAIN_VALS[1], CHAIN_VALS[0]], np.float64)])
    r = C.compare(W, B, 1, 1e-4, 2, exact=ex, names=("w", "b"), tie_ulps=100, order=True)
    assert r["f64_order_real_disagreements"] == 1
    assert r["f64_order_inversions_by_system"] == {"w": 0, "b": 0}
    assert r["f64_order_contradictions_by_system"] == {"w": 0, "b": 0}
    res = {"f64_order_failed_pairs": {"w_vs_b": 1}, "pairs": {"w_vs_b": r}}
    assert set(runner.order_blame(res, ["w", "b"])) == {"w", "b"}
    # B ranked [0, 2] (W's ids) against [0, 1]: 2 is not beyond the window of 1, still a tie;
    # against [0, 3] (row 3 = 0.1) W's row 2 beats row 3 beyond the window: B contradicts
    ex[(0, 3)] = float(CHAIN_VALS[3])
    B3 = ([np.array([0, 3], np.int64)], [np.array([CHAIN_VALS[0], CHAIN_VALS[3]], np.float64)])
    r = C.compare(W, B3, 1, 1e-4, 2, exact=ex, names=("w", "b"), tie_ulps=100, order=True)
    assert r["f64_order_contradictions_by_system"] == {"w": 0, "b": 1}


def _ip_round(root, vals, lists, k, groups=None, tie=128, scores=None, rescore=True,
              listed=None, exact_hook=None):
    """judge on a 1-d inner-product corpus ``vals`` (query 1.0, so a row's score is its value)
    with systems ``lists`` {name: [ids per query]}; ``scores`` {name: [scores per query]}
    replaces a system's reported scores (default the float32 row values). ``listed``: the
    systems pairs_blame is told of (default every one). -> (document, fails, pairs_blame)."""
    vals = np.asarray(vals, np.float32)
    (root / "c").mkdir(exist_ok=True)
    pq.write_table(pa.table({"emb": pa.array([[v] for v in vals], pa.list_(pa.float32(), 1))}),
                   root / "c" / "f0.parquet")
    nq = len(next(iter(lists.values())))
    pq.write_table(pa.table({"emb": pa.array([[1.0]] * nq, pa.list_(pa.float32(), 1))}),
                   root / "q.parquet")
    argv = ["--corpus", str(root / "c"), "--novabf", "-"] + (["--rescore"] if rescore else []) + [
            "--vcol", "emb",
            "--queries", str(root / "q.parquet"), "--qcol", "emb", "--metric", "ip", "--k",
            str(k), "--rescore-threads", "1", "--tie-ulps", str(tie)]
    argv += [z for nm, g in (groups or {}).items() for z in ("--group", f"{nm}={g}")]
    a = C.parse_args(argv)
    sy = {nm: ([np.array(r, np.int64) for r in rows],
               [np.array([np.float64(vals[g]) for g in r]) for r in rows])
          for nm, rows in lists.items()}
    for nm, rows in (scores or {}).items():
        sy[nm] = (sy[nm][0], [np.array(r, np.float64) for r in rows])
    inputs = {"keys": ["k0"], "offsets": np.array([0, len(vals)]), "n_ids": len(vals), "dup": 0,
              "qn": [str(i) for i in range(nq)], "systems": sy, "n": nq, "unmapped": 0,
              "gt_missing": None, "mapped": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        out, fails = C.judge(a, inputs, 0.0, exact_hook=exact_hook)
    blamed = runner.pairs_blame({"ok": not fails, "fails": fails, "compare": out},
                                listed or list(lists)) if fails else {}
    return out, fails, blamed


U50 = 50 * float(np.spacing(np.float32(0.75)))
# (rows in 50-ULP steps above 0.75, k, systems, wanted pairs_blame set, the old rule's set)
IP_CASES = {
    # second reviewer's repro: a, z correct, b wrong (holds row 2 for row 4); the pair a_vs_b
    # is resolved by b's count, b_vs_z is not: the common system over every failed pair is b
    "resolved_pair_keeps_common": ([6, 5, 3, 4, 6, 0], 3,
                                   {"a": [[0, 4, 1]], "b": [[0, 2, 1]], "z": [[3, 0, 1]]},
                                   {"b"}, {"b"}),
    # x correct, y and z wrong differently (each holds a low row for a high one): old all three
    "two_wrong_differently": ([9, 8, 7, 6, 0, 1], 3,
                              {"x": [[0, 1, 2]], "y": [[0, 1, 4]], "z": [[0, 5, 2]]},
                              {"y", "z"}, {"x", "y", "z"}),
    # y, z wrong alike: the old order rule blamed the correct x alone (y, z by their float64
    # set disagreements)
    "two_wrong_alike": ([9, 8, 7, 6, 0, 1], 3,
                        {"x": [[0, 1, 2]], "y": [[0, 1, 4]], "z": [[0, 1, 4]]},
                        {"y", "z"}, {"x", "y", "z"}),
    # a matched-rank inversion (reviewer finding 3): A = [0, 1] holds 0 above 1 that beats it,
    # B = [0, 2]; only rank 1 differs. A is wrong, B too: neither cleared
    "matched_rank_inversion": ([4, 8, 0], 2, {"A": [[0, 1]], "B": [[0, 2]]}, {"A", "B"},
                               {"A", "B"}),
    # tie-only differences (rows 0 and 1 tied): nobody
    "ties_only": ([5, 5, 0], 2, {"x": [[0, 1]], "y": [[1, 0]]}, set(), set()),
}


@pytest.mark.parametrize("case", list(IP_CASES))
def test_order_blame_end_to_end(tmp_path, case):
    """judge (with the full-rank order check) -> pairs_blame, all pairs and grouped alike;
    a system whose order blame and partners are the old rule's keeps its old text, word for
    word; any other order clause states the new verdict."""
    steps, k, lists, want, want_old = IP_CASES[case]
    vals = [0.75 + st * U50 for st in steps]
    out, fails, blamed = _ip_round(tmp_path, vals, lists, k)
    names = list(lists)
    gout, gfails, gblamed = _ip_round(tmp_path, vals, lists, k, {nm: "g" for nm in names})
    _same_document(out, gout)
    assert fails == gfails and blamed == gblamed
    assert set(blamed) == want
    res = {"ok": False, "fails": fails, "compare": out}
    old_order = runner._order_blames(out, names)[1] if fails else {}
    old_why = {}
    for nm in names:   # the old pairs_blame: f64_wrong_total + the old order rule
        w = ([f"{out['f64_wrong_total'][nm]} float64-checked disagreements"]
             if out.get("f64_wrong_total", {}).get(nm) else []) + old_order.get(nm, [])
        if w:
            old_why[nm] = "; ".join(w)
    assert set(old_why) == want_old
    if case == "two_wrong_alike":
        assert set(old_order) == {"x"} and set(runner.order_blame(out, names)) == {"y", "z"}
    v = runner._order_verdict(out, names) if fails else None
    for nm in set(blamed) & set(old_why):
        if v is None:
            assert blamed[nm] == old_why[nm], nm
            continue
        was = [pk for pk, d in v["old_in"].items() if nm in d]
        now = [pk for pk in v["final"] if nm in v["final"][pk]]
        if sorted(was) == sorted(now) and all(v["final"][pk] == set(v["old_in"][pk])
                                              for pk in now):
            assert blamed[nm] == old_why[nm], nm   # order blame and partners unchanged
        else:   # changed: every order clause states the new verdict
            for w in blamed[nm].split("; "):
                if "order disagreements" in w:
                    pk = w.split(" in ")[1].split(" ")[0]
                    # an old reason stays only while still true: an inversion, the common
                    # system, or "ambiguous" with both partners still blamed
                    assert w in v["old_in"].get(pk, {}).values() and (
                        v["final"][pk] == set(v["old_in"][pk])
                        or not w.endswith("(ambiguous: blamed on both)")) or \
                        "contradict the float64 scores" in w or "is consistent with" in w, w
                    assert pk in now
    if case == "matched_rank_inversion":
        r = out["pairs"]["A_vs_B"]
        assert r["f64_order_contradictions_by_system"] == {"A": 0, "B": 1}
        assert r["f64_order_full_contradictions_by_system"]["A"] > 0
    assert res


def test_the_full_check_runs_only_where_clearing_is_in_question():
    """(review round 2) one system wrong with proofs (rank swaps: inversions) against three
    correct ones: every failure is explained, nothing to clear, so no second rescoring pass;
    two ambiguous systems: the pass runs, its memory estimate told the first table it holds."""
    n, k = 4, 3
    ex = {(q, g): float(10 - g) for q in range(n) for g in range(8)}
    ok = ([np.arange(k) for _ in range(n)], [np.array([10.0, 9.0, 8.0]) for _ in range(n)])
    bad = ([np.array([1, 0, 2]) for _ in range(n)], [np.array([9.0, 10.0, 8.0]) for _ in range(n)])
    calls = []

    def rescore(pairs, have=None, live=0):
        calls.append((len(pairs), live))
        return {key: ex[key] for key in ex if key not in have}
    a = type("A", (), {"k": k, "tie_ulps": 128, "modality": "dense"})()
    systems = {"a": ok, "b": ok, "c": ok, "w": bad}
    exact = {key: v for key, v in ex.items() if key[1] < 3}
    docs = {f"{x}_vs_{y}": C.compare(systems[x], systems[y], n, 0.0, k, exact=exact,
                                     names=(x, y), order=True) for x, y in C.system_pairs(list(systems))}
    bad_pairs = {pk: r["f64_order_real_disagreements"] for pk, r in docs.items()
                 if r["f64_order_real_disagreements"]}
    assert set(bad_pairs) == {"a_vs_w", "b_vs_w", "c_vs_w"}
    C.order_full_checks(a, systems, n, docs, bad_pairs, exact, rescore)
    assert calls == [] and not any("f64_order_full_queries" in r for r in docs.values())
    # two systems, one inverted at a MATCHED rank only (no mismatched-rank proof): in question
    x = ([np.array([1, 0, 5]) for _ in range(n)], [np.array([9.0, 10.0, 5.0]) for _ in range(n)])
    y = ([np.array([1, 0, 6]) for _ in range(n)], [np.array([9.0, 10.0, 4.0]) for _ in range(n)])
    systems = {"x": x, "y": y}
    exact = {(q, g): ex[(q, g)] for q in range(n) for g in (5, 6)}
    docs = {"x_vs_y": C.compare(x, y, n, 0.0, k, exact=exact, names=("x", "y"), order=True)}
    C.order_full_checks(a, systems, n, docs, {"x_vs_y": docs["x_vs_y"][
        "f64_order_real_disagreements"]}, exact, rescore)
    assert calls == [(1, len(exact))]
    assert docs["x_vs_y"]["f64_order_full_contradictions_by_system"]["x"] > 0


def test_the_binade_edge_does_not_hide_a_contradiction():
    """(reviewer finding 2) at a binade edge the tie window is not monotone in the score, so
    testing only the best v after u missed B's contradiction (and, with A's count, cleared B);
    every v is tested now."""
    E0 = {0: 0.5 + 1e-6, 1: 0.5 - 1e-6, 2: 0.5 - 5e-6, 3: 0.5 + 1.5e-6}
    E1 = {0: 0.9, 1: 0.8, 2: 0.1, 3: 0.05}
    exact = {(q, g): e for q, E in enumerate((E0, E1)) for g, e in E.items()}
    L = lambda rows: ([np.array(r) for r in rows],
                      [np.array([exact[(q, g)] for g in r], np.float32) for q, r in enumerate(rows)])
    A, B = L([[0, 1], [1, 2]]), L([[3, 2], [0, 1]])
    r = C.compare(A, B, 2, 1e9, 2, exact=exact, names=("A", "B"), tie_ulps=128, order=True)
    assert r["f64_order_contradictions_by_system"]["B"] > 0
    res = {"f64_order_failed_pairs": {"A_vs_B": r["f64_order_real_disagreements"]},
           "pairs": {"A_vs_B": r}}
    assert set(runner.order_blame(res, ["A", "B"])) == {"A", "B"}


def test_a_normal_run_is_unchanged(corpus):
    """without --group: every pair in round order, each one compare() on the float64 table of
    every pair's needed ids (the code before groups), and no new keys."""
    tmp_path, ex = corpus
    spec = CASES["two_wrong_differently"][0] + [("gt", "ok", None)]
    out, fails, a, inputs = _round(tmp_path, ex, spec, grouped=False)
    names = [nm for nm, _, _ in spec]
    assert list(out["pairs"]) == [f"{x}_vs_{y}" for i, x in enumerate(names) for y in names[i + 1:]]
    assert not set(GROUPED_KEYS) & set(out)
    systems = inputs["systems"]
    need = C.needed_ids(systems, NQ, K, order=True)
    fs, paths = C.corpus_paths(a.corpus, N_FILES)
    xq = pq.read_table(a.queries).column("emb").to_numpy(zero_copy_only=False)
    xq = np.stack(xq)
    exact = C.exact_scores(need, a.corpus, "emb", inputs["offsets"], paths, fs, xq, "cosine", threads=1)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ref = C.compare(systems[x], systems[y], NQ, a.tol, K, exact=exact, names=(x, y),
                            tie_ulps=a.tie_ulps, modality="dense", order=True,
                            score_tol=(a.score_atol, a.score_rtol))
            got = dict(out["pairs"][f"{x}_vs_{y}"])
            # an order-failed pair also holds the full-rank order check (judge's second pass)
            full = {k_: got.pop(k_) for k_ in FULL_KEYS if k_ in got}
            assert not full or f"{x}_vs_{y}" in out["f64_order_failed_pairs"], (x, y)
            assert got == ref, (x, y)


@pytest.mark.parametrize("case", ["all_identical", "all_correct", "two_wrong_differently",
                                  "gt_correct_one_wrong", "member_differs_everywhere"])
def test_the_rescoring_covers_exactly_the_all_pairs_ids(corpus, monkeypatch, case):
    """judge's grouped rescoring is one pass over exactly every pair's needed (query, id)
    pairs (all pairs' float64 table: what a dense filtered round's non-candidate rule looks
    up), and pair_plan (prefetch_scores' list) gives the same ids. A failed order check adds
    one more pass (the full-rank order check): only ids some list holds on the failed
    queries that the first pass did not score, the same in all pairs."""
    tmp_path, ex = corpus
    spec = CASES[case][0]
    got = []
    real = C.exact_scores

    def spy(need, *a, **kw):
        got.append({(q, g) for q, s in enumerate(need) for g in s})
        return real(need, *a, **kw)
    monkeypatch.setattr(C, "exact_scores", spy)
    out, _, a, inputs = _round(tmp_path, ex, spec, grouped=True)
    want = {(q, g) for q, s in enumerate(C.needed_ids(inputs["systems"], NQ, K, order=True))
            for g in s}
    assert got[0] == want
    if len(got) > 1:
        assert len(got) == 2 and out["f64_order_failed_pairs"] and not got[1] & want
        held = {(q, int(g)) for sy in inputs["systems"].values() for q, L in enumerate(sy[0])
                for g in L[:K] if g >= 0}
        assert got[1] <= held
        got_all = []
        monkeypatch.setattr(C, "exact_scores",
                            lambda need, *a_, **kw: got_all.append(
                                {(q, g) for q, s in enumerate(need) for g in s})
                            or real(need, *a_, **kw))
        _round(tmp_path, ex, spec, grouped=False)
        assert got_all == got
    plan = C.needed_ids(inputs["systems"], NQ, K, order=True,
                        pairs=C.pair_plan(a, inputs["systems"], NQ))
    assert {(q, g) for q, s in enumerate(plan) for g in s} == want


def test_prefetch_scores_covers_the_grouped_pairs(corpus, monkeypatch):
    """after prefetch_scores (its pair list = pair_plan), judge's grouped rescoring computes
    nothing more (all from the score cache), with a member differing on some queries."""
    tmp_path, ex = corpus
    spec = CASES["all_correct"][0] + [("x-3", "bad1", "x")]
    calls = []
    real = C._exact_scores
    monkeypatch.setattr(C, "_exact_scores",
                        lambda need, *a: calls.append(sum(len(s) for s in need)) or real(need, *a))
    argv = ["--corpus", str(tmp_path / "c"), "--novabf", "-", "--rescore", "--vcol", "emb",
            "--queries", str(tmp_path / "q.parquet"), "--qcol", "emb", "--metric", "cosine",
            "--k", str(K), "--rescore-threads", "1"]
    argv += [z for nm, _, g in spec if g for z in ("--group", f"{nm}={g}")]
    a = C.parse_args(argv)
    systems = {nm: _lists(ex, kind) for nm, kind, _ in spec}
    inputs = {"keys": [f"k{i}" for i in range(N_FILES)],
              "offsets": np.array([ROWS * i for i in range(N_FILES + 1)]), "n_ids": N_FILES * ROWS,
              "dup": 0, "qn": [str(i) for i in range(NQ)], "systems": systems, "n": NQ,
              "unmapped": 0, "gt_missing": None, "mapped": {}}
    C.SCORE_CACHE = {}
    try:
        assert agree.prefetch_scores(a, inputs, 2) > 0
        n_pre = len(calls)
        with contextlib.redirect_stdout(io.StringIO()):
            out, fails = C.judge(a, inputs, 0.0)
    finally:
        C.SCORE_CACHE = None
    assert fails and out["group_differing_queries"]["x-3"] == 1
    assert calls[n_pre:] == []   # judge found every score in the cache


def test_system_groups_and_grouped_plan():
    import types
    a = types.SimpleNamespace(group=["a=A", "b=A", "c=A", "d=D"], k=2)
    names = ["a", "gt", "b", "d", "c"]
    g = C.system_groups(a, names)
    assert list(g.values()) == [["a", "b", "c"], ["gt"], ["d"]]
    assert C.group_reps(g) == {"a": "a", "b": "a", "c": "a", "gt": "gt", "d": "d"}
    L = lambda *rows: ([np.array(r, np.int64) for r in rows], [np.array([2.0, 1.0])] * len(rows))
    systems = {"a": L([1, 2], [3, 4]), "gt": L([1, 2], [3, 4]), "b": L([1, 2], [3, 5]),
               "d": L([1, 3], [3, 4]), "c": L([1, 2], [3, 4])}
    rep_of, diff, items = C.grouped_plan(systems, 2, 2, g)
    assert diff == {"a": [], "gt": [], "b": [1], "d": [], "c": []}
    # every pair of representatives, then the pairs with b on its differing query only
    assert items == [("a", "gt"), ("a", "d"), ("gt", "d"), ("a", "b", [1]), ("gt", "b", [1]),
                     ("b", "d", [1]), ("b", "c", [1])]
    assert C.pair_plan(a, systems, 2) == items
    assert C.pair_plan(types.SimpleNamespace(group=[], k=2), systems, 2) == C.system_pairs(names)
    assert C.system_groups(types.SimpleNamespace(group=[]), names) is None
    # a system named like a group is still its own group
    g = C.system_groups(types.SimpleNamespace(group=["v=x"]), ["x", "v"])
    assert list(g.values()) == [["x"], ["v"]]
    # same_lists: ids, scores bit for bit, and the raw length (a padded -1 counts)
    x = ([np.array([1, 2]), np.array([1, 2]), np.array([1, 2, -1])],
         [np.array([1.0, 0.5]), np.array([1.0, 0.5]), np.array([1.0, 0.5, 0.0])])
    y = ([np.array([1, 2]), np.array([1, 2]), np.array([1, 2])],
         [np.array([1.0, 0.5]), np.array([1.0, np.nextafter(0.5, 1)]), np.array([1.0, 0.5])])
    assert C.same_lists(x, y, 3, 3).tolist() == [True, False, False]


# ---- fold_queries: compare() over n queries from its n one-query documents ----

def _random_pair(rng, n, k, rows, special):
    base = [rng.choice(rows, k, replace=False).astype(np.int64) for _ in range(n)]
    sc = [np.sort(rng.random(k))[::-1].astype(np.float64) for _ in range(n)]
    ids_b, sc_b = [], []
    for b, s in zip(base, sc):
        a = b.copy()
        m = rng.random(k) < 0.15
        a[m] = rng.integers(0, rows, int(m.sum()))
        if rng.random() < 0.3:
            a[[0, 1]] = a[[1, 0]]
        t = s + rng.normal(0, 1e-6, k) * (rng.random(k) < 0.3)
        if special:
            if rng.random() < 0.3:
                t[rng.integers(0, k)] = np.inf
            if rng.random() < 0.3:
                t[rng.integers(0, k)] = np.nan
            if rng.random() < 0.2:
                a[1] = a[0]                       # a repeated id
            if rng.random() < 0.2:
                a[rng.integers(0, k)] = -1        # a padded slot
            if rng.random() < 0.15:
                a, t = a[:k - 2], t[:k - 2]       # a short list
        ids_b.append(a)
        sc_b.append(t)
    return (base, sc), (ids_b, sc_b)


@pytest.mark.parametrize("special", [False, True], ids=["plain", "special"])
@pytest.mark.parametrize("modality", ["dense", "sparse"])
@pytest.mark.parametrize("mode", ["no_exact", "exact", "exact_order"])
def test_fold_queries_is_compare_over_all_queries(special, modality, mode):
    """fold_queries of compare()'s one-query documents (exact table keyed (0, gid)) is
    compare() over all the queries, byte for byte (key order, examples, medians, sums)."""
    rng = np.random.default_rng(5 + special)
    n, k = 40, 12
    A, B = _random_pair(rng, n, k, 60, special)
    exact = None
    if mode != "no_exact":
        need = C.needed_ids({"a": A, "b": B}, n, k, order=True)
        exact = {}
        for q, s in enumerate(need):
            for g in s:
                r = rng.random()
                exact[(q, g)] = (-np.inf if r < 0.05 else np.nan if r < 0.1 and special
                                 else float(rng.random()))
            for g in A[0][q][:3]:   # a non-candidate among shared ids (dense short-list rule)
                if rng.random() < 0.05:
                    exact[(q, int(g))] = -np.inf
    kw = dict(tol=1e-4, kmax=k, tie_ulps=128, modality=modality, order=mode == "exact_order",
              score_tol=(1e-5, 1e-5))
    full = C.compare(A, B, n, exact=exact, names=C.SIDES, **kw)
    recs = []
    for q in range(n):
        exq = None if exact is None else {(0, g): v for (qq, g), v in exact.items() if qq == q}
        recs.append(C.compare(([A[0][q]], [A[1][q]]), ([B[0][q]], [B[1][q]]), 1, exact=exq,
                              names=C.SIDES, **kw))
    assert json.dumps(C.fold_queries(recs, k), default=str) == json.dumps(full, default=str)
    # and naming the sides afterwards is compare() with the names
    assert json.dumps(C.rename_sides(full, "x", "y"), default=str) == json.dumps(
        C.compare(A, B, n, exact=exact, names=("x", "y"), **kw), default=str)


# ---- the tie-window chain (a review's counterexample to "a member within one window of its
# representative shares its verdict"): k=1, 1-d inner product, rows 0, 1, 2 at 1.5, 1.5-60u,
# 1.5-120u (u = one float32 ULP at 1.5), --tie-ulps 100. Row 1 is within the window of both
# rows 0 and 2, rows 0 and 2 are not. ----

CHAIN_U = float(np.spacing(np.float32(1.5)))
CHAIN_VALS = np.array([1.5, 1.5 - 60 * CHAIN_U, 1.5 - 120 * CHAIN_U, 0.1, 0.2, 0.3], np.float32)


@pytest.fixture
def chain(tmp_path):
    (tmp_path / "c").mkdir()
    pq.write_table(pa.table({"emb": pa.array([[v] for v in CHAIN_VALS], pa.list_(pa.float32(), 1))}),
                   tmp_path / "c" / "f0.parquet")
    pq.write_table(pa.table({"emb": pa.array([[1.0]] * 3, pa.list_(pa.float32(), 1))}),
                   tmp_path / "q.parquet")
    return tmp_path


def _chain_round(root, spec, grouped, order):
    """spec [(name, [row per query], group)]: each system's k=1 list per query."""
    argv = ["--corpus", str(root / "c"), "--novabf", "-", "--rescore", "--vcol", "emb",
            "--queries", str(root / "q.parquet"), "--qcol", "emb", "--metric", "ip", "--k", "1",
            "--rescore-threads", "1", "--tie-ulps", "100"] + ([] if order else ["--no-order-check"])
    if grouped:
        argv += [z for nm, _, g in spec for z in ("--group", f"{nm}={g}")]
    a = C.parse_args(argv)
    sy = {nm: ([np.array([r], np.int64) for r in rows],
               [np.array([np.float64(CHAIN_VALS[r])]) for r in rows]) for nm, rows, _ in spec}
    inputs = {"keys": ["k0"], "offsets": np.array([0, len(CHAIN_VALS)]), "n_ids": len(CHAIN_VALS),
              "dup": 0, "qn": ["0", "1", "2"], "systems": sy, "n": 3, "unmapped": 0,
              "gt_missing": None, "mapped": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        out, fails = C.judge(a, inputs, 0.0)
    listed = [nm for nm, _, _ in spec]
    blamed = sorted(runner.pairs_blame({"ok": not fails, "fails": fails, "compare": out},
                                       listed)) if fails else []
    return out, fails, blamed


CHAIN = {
    # z within the window of its representative x (row 1 vs 2), beyond it from w (row 0): the
    # tie-window stay rule passed z; all pairs (and now grouped) blame it
    "false_pass": ([("x", [1, 1, 1], "X"), ("z", [2, 2, 2], "X"), ("w", [0, 0, 0], "W")],
                   ["z"], ["z"]),
    # the reverse: x wrong against w, z (row 1) within the window of w and of x: the stay
    # rule failed z with x; all pairs blame x alone
    "false_fail": ([("x", [2, 2, 2], "X"), ("z", [1, 1, 1], "X"), ("w", [0, 0, 0], "W")],
                   ["x"], ["x"]),
    # z equals x on queries 1 and 2, the chain on query 0 only (the per-query path)
    "false_pass_one_query": ([("x", [1, 0, 0], "X"), ("z", [2, 0, 0], "X"), ("w", [0, 0, 0], "W")],
                             ["z"], ["z"]),
    "false_fail_one_query": ([("x", [2, 0, 0], "X"), ("z", [1, 0, 0], "X"), ("w", [0, 0, 0], "W")],
                             ["x"], ["x"]),
    # members of two groups, each within one window of its representative (both row 1, equal),
    # two windows apart from each other (rows 2 and 0): the row-2 member is wrong (it keeps row
    # 2 while lacking row 0, beyond the window: blamed alone), both directions
    "member_to_member": ([("x", [1, 1, 0], "X"), ("x-1", [2, 2, 0], "X"), ("w", [1, 1, 0], "W"),
                          ("w-1", [0, 0, 0], "W")], ["x-1"], ["x-1"]),
    "member_to_member_reverse": ([("x", [1, 0, 1], "X"), ("x-1", [0, 0, 0], "X"),
                                  ("w", [1, 0, 1], "W"), ("w-1", [2, 0, 2], "W")],
                                 ["w-1"], ["w-1"]),
}


@pytest.mark.parametrize("order", [True, False], ids=["order", "no_order"])
@pytest.mark.parametrize("case", list(CHAIN))
def test_the_tie_window_chain_gives_the_all_pairs_verdicts(chain, case, order):
    """tie windows do not compose (a member within one window of its representative can be two
    beyond another system): the grouped document, FAILs and blame equal all pairs' anyway."""
    spec, want_order, want_no_order = CHAIN[case]
    full, full_fails, full_blamed = _chain_round(chain, spec, False, order)
    grp, grp_fails, grp_blamed = _chain_round(chain, spec, True, order)
    _same_document(full, grp)
    assert full_fails == grp_fails
    assert full_blamed == grp_blamed == (want_order if order else want_no_order)


SHARED_KEYS = ("shared_score_checked", "shared_score_wrong_by_system",
               "shared_score_unjudged_by_system")
F = [float(np.float32(v)) for v in (0.9, 0.8, 0.7, 0.6)]   # the rows' float32 scores
W75, W85 = [F[0], 0.75, F[2]], [F[0], 0.85, F[2]]          # row 1's score wrong (0.8 really)
# (systems' reported scores on [[0, 1, 2]] (None: correct), wanted pairs_blame {system:
# why}, the old rule's blamed set)
SHARED_CASES = {
    # the toy: A and B share a wrong score, C is right. A_vs_B agrees, so the rule blamed C
    "two_wrong_alike": ({"A": W75, "B": W75, "C": None},
                        {"A": "1 shared-id score mismatches in A_vs_C (its score is off the "
                              "float64 score on 1 ids)",
                         "B": "1 shared-id score mismatches in B_vs_C (its score is off the "
                              "float64 score on 1 ids)"}, {"C"}),
    # wrong differently: every pair fails, the rule blamed all three; C agrees with float64
    "two_wrong_differently": ({"A": W75, "B": W85, "C": None},
                              {"A": "1 shared-id score mismatches in A_vs_B; 1 shared-id score "
                                    "mismatches in A_vs_C (its score is off the float64 score "
                                    "on 1 ids)",
                               "B": "1 shared-id score mismatches in A_vs_B; 1 shared-id score "
                                    "mismatches in B_vs_C (its score is off the float64 score "
                                    "on 1 ids)"}, {"A", "B", "C"}),
    # only A wrong: the rule's blame, word for word
    "one_wrong": ({"A": W75, "B": None, "C": None},
                  {"A": "1 shared-id score mismatches in A_vs_B; 1 shared-id score mismatches "
                        "in A_vs_C"}, {"A"}),
    # two systems: the rule blamed both, the float64 scores name A
    "two_systems": ({"A": W75, "B": None},
                    {"A": "1 shared-id score mismatches in A_vs_B (its score is off the "
                          "float64 score on 1 ids)"}, {"A", "B"}),
    # two systems both wrong: both, the rule's text
    "two_systems_both_wrong": ({"A": W75, "B": W85},
                               {"A": "1 shared-id score mismatches in A_vs_B",
                                "B": "1 shared-id score mismatches in A_vs_B"}, {"A", "B"}),
}


def _shared_round(root, case, rescore=True, groups=False, exact_hook=None):
    rep, want, old = SHARED_CASES[case]
    lists = {nm: [[0, 1, 2]] for nm in rep}
    scores = {nm: [s] for nm, s in rep.items() if s is not None}
    return _ip_round(root, [0.9, 0.8, 0.7, 0.6], lists, 3, scores=scores, rescore=rescore,
                     groups={nm: "g" for nm in rep} if groups else None,
                     exact_hook=exact_hook)


@pytest.mark.parametrize("case", list(SHARED_CASES))
def test_shared_score_blame_end_to_end(tmp_path, case):
    """judge's float64 check of the mismatched shared ids -> shared_score_blame names the
    systems whose scores are off the float64 scores, not a correct one the common-system rule
    picked; grouped (a sweep) gives the all-pairs document and blame; without --rescore the
    rule's blame and text, and no new keys."""
    rep, want, old = SHARED_CASES[case]
    out, fails, blamed = _shared_round(tmp_path, case)
    gout, gfails, gblamed = _shared_round(tmp_path, case, groups=True)
    _same_document(out, gout)
    assert fails == gfails and blamed == gblamed == want
    failed = out["shared_score_failed_pairs"]
    assert set(runner._shared_score_common(out, list(rep), failed, {})) == old
    for pk, r in out["pairs"].items():
        assert (r["shared_score_checked"] == r["shared_score_mismatches"]) if pk in failed \
            else not set(SHARED_KEYS) & set(r)
        if pk in failed:
            assert r["shared_score_unjudged_by_system"] == dict.fromkeys(pk.split("_vs_"), 0)
            assert r["shared_score_wrong_by_system"] == {
                nm: int(rep[nm] is not None) for nm in pk.split("_vs_")}
        assert not set(SHARED_KEYS) & set(C.legacy_view(out)["pairs"][pk])
    # without --rescore: the rule's blame, its text word for word, no new keys
    out, fails, blamed = _shared_round(tmp_path, case, rescore=False)
    assert out["shared_score_failed_pairs"] == failed
    assert not any(set(SHARED_KEYS) & set(r) for r in out["pairs"].values())
    rule = runner._shared_score_common(out, list(rep), failed, {})
    assert runner.shared_score_blame(out, list(rep)) == rule
    assert blamed == {nm: "; ".join(w) for nm, w in rule.items()}


def test_shared_score_check_pass_counts_its_masked_rows(tmp_path):
    """the shared-score pass rescores only the mismatched ids (one here) after the first pass,
    and filter_masked_pairs includes the rows exact_hook saw in it."""
    seen = []

    def hook(ex):
        seen.append(len(ex))
        return len(ex)
    out, _, _ = _shared_round(tmp_path, "two_wrong_alike", exact_hook=hook)
    assert seen == [0, 1] and out["filter_masked_pairs"] == 1


def test_shared_score_gt_pairs_stay_a_note(tmp_path):
    """a pair with the dataset GT is a note, not a FAIL: no float64 shared-score check for
    it; the systems under test are judged as before (A wrong alone: the rule's text)."""
    lists = {nm: [[0, 1, 2]] for nm in ("A", "gt", "C", "D")}
    out, fails, blamed = _ip_round(tmp_path, [0.9, 0.8, 0.7, 0.6], lists, 3,
                                   scores={"A": [W75], "gt": [W75]}, listed=["A", "C", "D"])
    assert set(out["shared_score_failed_pairs"]) == {"A_vs_C", "A_vs_D"}
    assert not any(set(SHARED_KEYS) & set(r) for pk, r in out["pairs"].items() if "gt" in pk)
    assert out["pairs"]["A_vs_gt"]["shared_score_mismatches"] == 0
    assert out["pairs"]["gt_vs_C"]["shared_score_mismatches"] == 1
    assert blamed == {"A": "1 shared-id score mismatches in A_vs_C; 1 shared-id score "
                           "mismatches in A_vs_D"}


def test_needed_ids_shared_item_form():
    """(x, y, queries, "shared", tol): exactly the shared ids whose two scores are
    shared_score_bad on those queries (rescore_ragged takes the same items via needed_ids)."""
    x = ([np.array([0, 1, 2]), np.array([3, 4, 5])], [np.array([0.9, 0.8, 0.7]),
                                                     np.array([0.6, 0.5, 0.4])])
    y = ([np.array([0, 2, 1]), np.array([3, 4, 6])], [np.array([0.9, 0.7, 0.75]),
                                                     np.array([0.6, 0.6, 0.3])])
    sy = {"x": x, "y": y}
    tol = (1e-5, 1e-5)
    need = C.needed_ids(sy, 2, 3, pairs=[("x", "y", [0, 1], "shared", tol)])
    assert need == [{1}, {4}]
    assert C.needed_ids(sy, 2, 3, order=True, pairs=[("x", "y", [1], "shared", tol)]) == \
        [set(), {4}]
    assert C.shared_score_ids(x, y, 0, 3, tol) == [(1, 0.8, 0.75)]
    r = C.shared_score_check(sy, ("x", "y"), [0, 1], 3, {(0, 1): 0.8}, tol)
    assert r == {"shared_score_checked": 2, "shared_score_wrong_by_system": {"x": 0, "y": 1},
                 "shared_score_unjudged_by_system": {"x": 1, "y": 1}}


def test_score_off_rule():
    assert not C.score_off(0.8, 0.8, 1e-5, 1e-5)
    assert not C.score_off(-np.inf, -np.inf, 1e-5, 1e-5)
    assert C.score_off(np.inf, -np.inf, 1e-5, 1e-5)
    assert C.score_off(0.3, -np.inf, 1e-5, 1e-5) and C.score_off(-np.inf, 0.3, 1e-5, 1e-5)
    assert C.score_off(float("nan"), 0.3, 1e-5, 1e-5)
    assert not C.score_off(1.0 + 0.99 * 2e-5, 1.0, 1e-5, 1e-5)
    assert C.score_off(1.0 + 1.01 * 2e-5, 1.0, 1e-5, 1e-5)


def test_a_capped_shared_score_check_proves_but_never_clears(tmp_path, monkeypatch):
    """past SHARED_CHECK_MAX_IDS mismatched ids a pair is checked on a sample: the wrong sides
    are still proven (A and B), but the correct one the rule blamed (C) is not cleared, since
    not every mismatched id was checked; grouped gives the same."""
    monkeypatch.setattr(C, "SHARED_CHECK_MAX_IDS", 1)
    bad = [F[0], 0.75, 0.65]                     # rows 1 and 2 wrong, alike in A and B
    lists = {nm: [[0, 1, 2]] for nm in "ABC"}
    scores = {"A": [bad], "B": [bad]}
    out, fails, blamed = _ip_round(tmp_path, [0.9, 0.8, 0.7, 0.6], lists, 3, scores=scores)
    gout, gfails, gblamed = _ip_round(tmp_path, [0.9, 0.8, 0.7, 0.6], lists, 3, scores=scores,
                                      groups={nm: "g" for nm in "ABC"})
    _same_document(out, gout)
    assert fails == gfails and blamed == gblamed
    for pk in ("A_vs_C", "B_vs_C"):
        r = out["pairs"][pk]
        assert r["shared_score_mismatches"] == 2 and r["shared_score_checked"] == 1
    assert set(blamed) == {"A", "B", "C"}
