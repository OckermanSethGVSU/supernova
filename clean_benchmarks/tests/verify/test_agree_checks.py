"""bench/verify/agree.py on real `bench run --dev` runs of the fixture experiments.

  stale verdicts      checks.json hashes its inputs; summarize refuses a changed / missing one;
                      `bench leg` removes checks.json when it re-runs a leg
  shared-id scores    every query, not only the oracle sample
  fp32 evidence       per-system max ULP / abs error, the 2x-tie-window warning in
                      checks.json warnings and summary.md, optional checks.max_score_ulps
  filtered blame      the pairs' exact table masks filter-excluded pairs (-inf)
  coverage            each timed pass's n_files = the corpus listing / subset
"""

import contextlib
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("nova_bf")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from bench.report import summary as SUM  # noqa: E402
from bench.verify import agree  # noqa: E402
from helpers import fixture_runs  # noqa: E402

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NOVA_BF_", "NOVABF_", "PAPER_"))}
    env.update(HARNESS_NO_IMDS="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    return env


def bench(*args):
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=CB, env=_env(),
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{shape: run dir}: this module's own copy of the shared, cached runs (fixture_runs)."""
    return fixture_runs.ok_runs(tmp_path_factory, "cf_results", ("dense", "filtered"))


# --- the probes' helpers (mut.py) -------------------------------------------------------------
def fresh(src, dst):
    """A copy of a finished run dir, its leg records' paths moved to the copy, checks re-run."""
    shutil.copytree(src, dst)
    for f in glob.glob(os.path.join(dst, "legs", "*.json")):
        t = open(f).read().replace(src, str(dst))
        open(f, "w").write(t)
    return str(dst)


def checks(d, exp=None):
    exp = exp or json.load(open(os.path.join(d, "experiment.json")))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        c = agree.run_checks(d, exp)
    json.dump(c, open(os.path.join(d, "checks.json"), "w"), indent=1, default=str)
    return c


def npz_edit(d, tag, fn):
    p = os.path.join(d, "legs", tag + ".topk.npz")
    z = dict(np.load(p))
    fn(z)
    np.savez(p, **z)


def nb_path(d, tag):
    return list(json.load(open(os.path.join(d, "legs", tag + ".json")))["counters"]["results"].values())[0]


def nb_edit(d, tag, fn):
    p = nb_path(d, tag)
    t = pq.read_table(p)
    ids, sc = t.column("hit_ids").to_pylist(), t.column("hit_scores").to_pylist()
    fn(ids, sc)
    for name, v in (("hit_ids", ids), ("hit_scores", sc)):
        t = t.set_column(t.schema.get_field_index(name), name, pa.array(v, t.schema.field(name).type))
    pq.write_table(t, p)


def corpus_files(exp):
    c = exp["corpus"]["path"]
    c = c if os.path.isabs(c) else os.path.join(CB, c)
    return sorted(glob.glob(os.path.join(c, "**", "*.parquet"), recursive=True))


def uid(key, row):
    h = hashlib.md5(f"{key}:{row}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def gid2uid(exp):
    out = []
    for f in corpus_files(exp):
        out += [uid(f, r) for r in range(pq.read_metadata(f).num_rows)]
    return out


def setrec(d, tag, **kv):
    p = os.path.join(d, "legs", tag + ".json")
    r = json.load(open(p))
    r.update(kv)
    json.dump(r, open(p, "w"))


def summ(d, exp=None, **kw):
    return SUM.summarize(d, exp, **kw)[1]


# --- 1. stale verdicts ------------------------------------------------------------------------
@pytest.mark.slow
def test_stale_checks_json_is_refused(runs, tmp_path):
    """mut_time.py's stale-verdict case: a top-k corrupted (or a leg's time changed) after the
    checks; the old checks.json still says ok, `bench summarize` must exit 1."""
    d = fresh(runs["dense"], tmp_path / "stale")
    c = checks(d)
    assert c["ok"] and c["inputs"], c["problems"]
    assert "legs/dense__cpu__r1.topk.npz" in c["inputs"]
    assert any(k.endswith(".parquet") for k in c["inputs"])          # nova-bf's results
    assert summ(d) == []
    npz_edit(d, "dense__cpu__r1", lambda z: z["I"].__setitem__((5, slice(None)), z["I"][6]))
    p = bench("summarize", d)
    assert p.returncode == 1, p.stdout[-2000:]
    assert "stale checks.json" in p.stdout and "dense__cpu__r1.topk.npz: changed" in p.stdout
    checks(d)                                    # re-run: the corruption is now caught
    assert any("stale" in x or "checks failed" in x for x in summ(d))
    d2 = fresh(runs["dense"], tmp_path / "stale_time")
    checks(d2)
    setrec(d2, "dense__cpu__r1", timed_seconds=123.0)   # a re-run leg's new time
    assert any("stale checks.json" in x and "dense__cpu__r1.json: changed" in x for x in summ(d2))


@pytest.mark.slow
def test_stale_checks_json_missing_or_added_input(runs, tmp_path):
    d = fresh(runs["dense"], tmp_path / "miss")
    checks(d)
    os.remove(nb_path(d, "dense__novabf__r2"))
    assert any("stale" in x and "missing" in x for x in summ(d))
    d2 = fresh(runs["dense"], tmp_path / "added")
    checks(d2)
    shutil.copy(os.path.join(d2, "legs", "dense__cpu__r1.json"),
                os.path.join(d2, "legs", "old_dense__cpu__r1.json"))
    assert any("stale" in x and "added after them" in x for x in summ(d2))
    d3 = fresh(runs["dense"], tmp_path / "noinputs")
    json.dump({"ok": True}, open(os.path.join(d3, "checks.json"), "w"))   # hand-written
    assert any("records no inputs" in x for x in summ(d3))


@pytest.mark.slow
def test_bench_leg_rerun_removes_checks_json(runs, tmp_path):
    d = fresh(runs["dense"], tmp_path / "rerun")
    checks(d)
    assert os.path.isfile(os.path.join(d, "checks.json"))
    p = bench("leg", d, "--system", "cpu", "--search", "dense", "--round", "1", "--dev")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert not os.path.exists(os.path.join(d, "checks.json"))
    assert any("no readable checks.json" in x for x in summ(d))


# --- 2. shared-id scores on every query -------------------------------------------------------
def _unsampled(d, n_random):
    n = len(np.load(os.path.join(d, "legs", "dense__cpu__r1.topk.npz"))["I"])
    samp = set(np.random.default_rng(agree.ORACLE_SEED).choice(n, size=n_random, replace=False).tolist())
    return next(i for i in range(n) if i not in samp)


@pytest.mark.slow
@pytest.mark.parametrize("how", ["one", "halved"])
def test_score_corruption_at_a_non_sampled_query_fails(runs, tmp_path, how):
    """mut_dense.py: a score-only corruption at a query outside the oracle sample (ids right)."""
    d = fresh(runs["dense"], tmp_path / how)
    exp = json.load(open(os.path.join(d, "experiment.json")))
    exp.setdefault("checks", {})["oracle_queries"] = 10
    json.dump(exp, open(os.path.join(d, "experiment.json"), "w"))
    qq = _unsampled(d, 10)
    if how == "one":
        npz_edit(d, "dense__cpu__r1", lambda z: z["D"].__setitem__((qq, 2), z["D"][qq, 2] + 0.05))
    else:
        npz_edit(d, "dense__cpu__r1", lambda z: z["D"].__setitem__((qq, slice(None)), z["D"][qq] * 0.5))
    c = checks(d, exp)
    assert not c["ok"]
    r1 = c["searches"]["dense"]["r1"]
    assert r1["oracle"]["ok"], "the sample does not hold the query: only the pairs can see it"
    assert any("shared-id score mismatches" in f for f in r1["pairs"]["fails"]), r1["pairs"]["fails"]
    ex = r1["pairs"]["compare"]["pairs"]["novabf_vs_cpu"]["shared_score_examples"][0]
    assert ex["query"] == qq


@pytest.mark.slow
@pytest.mark.parametrize("ulps", [1, -1])
def test_one_ulp_and_exact_ties_still_pass(runs, tmp_path, ulps):
    d = fresh(runs["dense"], tmp_path / f"ulp{ulps}")
    to = np.float32(np.inf if ulps > 0 else -np.inf)
    npz_edit(d, "dense__cpu__r1", lambda z: z["D"].__setitem__(slice(None), np.nextafter(z["D"], to)))
    c = checks(d)
    assert c["ok"], c["problems"]
    pr = c["searches"]["dense"]["r1"]["pairs"]["compare"]["pairs"]["novabf_vs_cpu"]
    assert pr["shared_score_mismatches"] == 0


# --- 3. fp32-exactness evidence ---------------------------------------------------------------
def _plus_ulps(n):
    def f(z):
        z["D"][:] = (z["D"].astype(np.float64)
                     + n * np.spacing(np.abs(z["D"]).astype(np.float32)).astype(np.float64)).astype(np.float32)
    return f


@pytest.mark.slow
def test_score_error_recorded_and_warned(runs, tmp_path):
    """mut_ulp.py: every cpu score +n float32 ULPs. The per-system max error is in checks.json;
    twice it above the 128-ULP tie window is a checks.json warning and shows in summary.md."""
    d = fresh(runs["dense"], tmp_path / "base")
    c = checks(d)
    se = c["score_error"]["dense"]
    assert set(se) == {"cpu", "novabf"} and all(v["max_ulps"] < 64 for v in se.values()), se
    assert all(v["max_abs"] < 1e-5 for v in se.values())
    for n, warn in ((60, False), (150, True)):
        d = fresh(runs["dense"], tmp_path / f"u{n}")
        npz_edit(d, "dense__cpu__r1", _plus_ulps(n))
        c = checks(d)
        assert c["ok"], c["problems"]               # within the score bar: not a failure
        assert c["score_error"]["dense"]["cpu"]["max_ulps"] >= n - 1
        assert any("ULP" in w and "cpu" in w for w in c["warnings"]) is warn, c["warnings"]
        rows, problems, excluded, e2e = SUM.summarize(d)
        text = SUM.write(d, rows, problems, excluded, e2e)
        assert "Max own-score error vs float64" in text and "- dense: cpu " in text
        assert ("2 > the 128-ULP tie window" in text) is warn


@pytest.mark.slow
def test_max_score_ulps_bound(runs, tmp_path):
    from bench import experiment as E
    d = fresh(runs["dense"], tmp_path / "bound")
    npz_edit(d, "dense__cpu__r1", _plus_ulps(60))
    exp = json.load(open(os.path.join(d, "experiment.json")))
    exp["checks"]["max_score_ulps"] = 40
    c = checks(d, exp)
    assert not c["ok"] and any("checks.max_score_ulps 40" in p and "cpu" in p for p in c["problems"])
    assert not any("novabf" in p for p in c["problems"])
    raw = E.load_raw(os.path.join(FIX, "dense.yaml"))
    raw["checks"] = {"max_score_ulps": 64}
    assert E.finish(raw, os.path.join(FIX, "dense.yaml"))["checks"]["max_score_ulps"] == 64
    raw["checks"] = {"max_score_ulps": "x"}
    with pytest.raises(E.ExperimentError):
        E.finish(raw, os.path.join(FIX, "dense.yaml"))


# --- 4. filtered pairs: the system that returned an excluded row is blamed ---------------------
@pytest.mark.slow
def test_filtered_pairs_blame_the_system_returning_an_excluded_row(runs, tmp_path):
    """mut_other.py's masked-row case: masked_bitmap returns a filter-excluded row with its
    true score (sorted). Only masked_bitmap may be blamed, in the pairs as in the oracle."""
    from bench.verify import agree as A
    src = runs["filtered"]
    c0 = json.load(open(os.path.join(src, "checks.json")))
    # near-zero scores show many ULPs at ~1e-7 absolute error: recorded, not warned
    assert c0["ok"] and not any("score error" in w for w in c0["warnings"]), c0["warnings"]
    assert all(e["max_abs"] < 1e-6 for by in c0["score_error"].values() for e in by.values())
    exp = json.load(open(os.path.join(src, "experiment.json")))
    k = exp["search"]["k"]
    files = corpus_files(exp)
    X = np.concatenate([np.asarray(pq.read_table(f).column(exp["corpus"]["column"]).to_pylist(),
                                   np.float64) for f in files])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    U = gid2uid(exp)
    S = "filtered_text"
    H1, N1 = f"{S}__masked_bitmap__r1", f"{S}__novabf__r1"
    z0 = np.load(os.path.join(src, "legs", H1 + ".topk.npz"))
    a = A.C.parse_args(A.compare_args(exp, next(s for s in exp["searches"] if s["name"] == S),
                                      nb_path(src, N1), "uuid"))
    qn = pq.read_table(nb_path(src, N1)).column("query_id").to_pylist()
    xq = A._dense_queries(a, len(qn), qn).astype(np.float64)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    q = next(i for i in range(len(qn)) if (z0["I"][i] >= 0).sum() == k)
    sc = xq[q] @ X.T
    kth = sc[z0["I"][q, k - 1]]
    bad = next(int(g) for g in np.argsort(-sc) if g not in set(z0["I"][q].tolist()))
    assert sc[bad] > kth, "the best row not returned is a filter-excluded row above the k-th"

    def put(z):
        z["I"][q, k - 1] = bad
        z["D"][q, k - 1] = np.float32(sc[bad])
        o = np.argsort(-z["D"][q], kind="stable")
        z["I"][q] = z["I"][q][o]
        z["D"][q] = z["D"][q][o]
    d = fresh(src, tmp_path / "masked_row")
    npz_edit(d, H1, put)
    c = checks(d)
    r1 = c["searches"][S]["r1"]
    blamed = r1["pairs"]["compare"]["f64_wrong_total"]
    assert blamed.get("masked_bitmap", 0) >= 1, blamed
    assert blamed.get("novabf", 0) == 0 and blamed.get("masked_files", 0) == 0, blamed
    assert r1["pairs"]["compare"]["filter_masked_pairs"] >= 1
    assert not r1["oracle"]["systems"]["novabf"]["fails"]
    assert r1["oracle"]["systems"]["masked_bitmap"]["fails"]

    def nbput(ids, s_):            # both return it: the pairs cannot see it, the oracle blames both
        ids[q][k - 1] = U[bad]
        s_[q][k - 1] = float(sc[bad])
        o = np.argsort(-np.asarray(s_[q]), kind="stable")
        ids[q] = [ids[q][i] for i in o]
        s_[q] = [s_[q][i] for i in o]
    d2 = fresh(src, tmp_path / "both_row")
    npz_edit(d2, H1, put)
    nb_edit(d2, N1, nbput)
    c2 = checks(d2)
    assert not c2["ok"]
    o = c2["searches"][S]["r1"]["oracle"]["systems"]
    assert o["novabf"]["fails"] and o["masked_bitmap"]["fails"]


# --- 5. coverage ------------------------------------------------------------------------------
@pytest.mark.slow
def test_coverage_every_timed_pass_covers_the_listing(runs, tmp_path):
    """mut_time.py's n_files cases: each must be a problem (before: two passed)."""
    d = fresh(runs["dense"], tmp_path / "cov")
    checks(d)
    m = json.load(open(os.path.join(d, "manifest.json")))
    nf = m["data"]["corpus"]["n_files"]
    assert nf == 4 and summ(d) == []

    def case(name, **legs):
        dd = fresh(runs["dense"], tmp_path / name)
        checks(dd)
        for tg, v in legs.items():
            setrec(dd, tg, n_files=v)
        return [p for p in summ(dd) if "stale" not in p]
    p = case("nb3", dense__novabf__r1=3)
    assert any("dense novabf r1: excluded" in x and "expected 4" in x for x in p), p
    p = case("hnone", dense__cpu__r1=None, dense__cpu__r2=None)
    assert any("dense cpu r1: excluded" in x and "no n_files" in x for x in p), p
    p = case("all3", dense__cpu__r1=3, dense__cpu__r2=3, dense__novabf__r1=3, dense__novabf__r2=3)
    assert sum("expected 4 (the run's corpus listing" in x for x in p) == 4, p
    # the subset is the count when there is one
    exp = json.load(open(os.path.join(d, "experiment.json")))
    assert SUM.expected_files(d, {**exp, "subset": {"files": 2}}) == (2, "the experiment's subset.files")
    assert SUM.expected_files(d, exp, 7)[0] == 7
