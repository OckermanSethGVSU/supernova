"""End-to-end check on CPU: `bench run --dev` of every experiment shape, failure injection, a sweep.

Shapes (tests/fixtures/experiments, on the nova-bf reference fixtures): dense, sparse, multivector
(nova-bf + the harness cpu reference) and filtered (nova-bf + the masked reference, two searches
so the per-search rotation runs). Every leg is a real fresh process through the real leg runner.

Injected failures (each must fail the leg / the run, never give a time):
  crashed leg         a corrupt corpus file (both kinds)
  missing round       a leg record removed -> summarize fails
  warmup changed      harness --warmup-files 2 / --warmup 0 / --repeats 2; nova-bf
                      NOVABF_WARMUP_FILES=2 / unset
  clock started late  nova-bf's corpus_start hook deferred to the first pause
  write outside pause an output write while the clock runs before any pause
  and the write allowance: a multi-search nova-bf leg (writes submitted in pauses) passes.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("nova_bf")

from bench import experiment as E  # noqa: E402
from bench import leg as L  # noqa: E402
from bench import runner  # noqa: E402
from bench.report import summary as SUM  # noqa: E402
from helpers import fixture_runs  # noqa: E402

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(CB, "tests", "fixtures", "experiments")
NBFIX = os.path.join(CB, "tests", "fixtures", "novabf")


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NOVA_BF_", "NOVABF_", "PAPER_"))}
    env.update(HARNESS_NO_IMDS="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    return env


def bench(*args, cwd=CB):
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=cwd, env=_env(),
                          capture_output=True, text=True)


def _only_run(results, name):
    runs = sorted(os.listdir(os.path.join(results, name)))
    assert len(runs) == 1, runs
    return os.path.join(results, name, runs[0])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{shape: {"proc", "dir"}}: this module's own copy of the shared, cached `bench run --dev`
    of every shape (tests/helpers/fixture_runs.py)."""
    return fixture_runs.runs(tmp_path_factory, "s4_results")


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["dense", "sparse", "multivector", "filtered"])
def test_every_shape_runs_end_to_end(runs, shape):
    """Every leg of every shape runs and every shape's checks pass."""
    r = runs[shape]
    assert r["proc"].returncode == 0, r["proc"].stdout[-3000:] + r["proc"].stderr[-3000:]
    d = r["dir"]
    for f in ("manifest.json", "experiment.json", "checks.json", "summary.csv", "summary.md"):
        assert os.path.isfile(os.path.join(d, f)), f
    exp = json.load(open(os.path.join(d, "experiment.json")))
    rows, problems, excluded, _ = SUM.summarize(d, require_checks=True)
    assert not problems and not excluded
    n_legs = sum(len(s["systems"]) for s in exp["searches"])
    assert len(rows) == n_legs and all(row["n"] == 2 for row in rows)
    for row in rows:
        assert row["median_s"] > 0
        assert row["time_source"] == "LegClock"      # every system
    checks = json.load(open(os.path.join(d, "checks.json")))
    assert checks["ok"] is True and set(checks["searches"]) == {s["name"] for s in exp["searches"]}, \
        checks["problems"]


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["dense", "sparse", "multivector", "filtered"])
def test_leg_records(runs, shape):
    d = runs[shape]["dir"]
    exp = json.load(open(os.path.join(d, "experiment.json")))
    for r, s, sy in runner.timing.schedule(exp):
        rec = json.load(open(os.path.join(d, "legs", f"{s}__{sy}__r{r}.json")))
        assert rec["ok"] and rec["failures"] == [] and rec["exit_code"] == 0
        assert rec["wrapper"] == ["/usr/bin/time", "-v"] and rec["peak_rss_bytes"] > 0
        assert rec["protocol"] == {"warmup_files": 1, "repeats": 1, "fresh_process": True, "caches_cleared": []}
        assert rec["cwd"] == CB and rec["n_files"] == 4
        assert os.path.isfile(os.path.join(d, rec["topk"]))
        att = rec["attribution"]                                          # every leg
        assert att["kind"] == rec["kind"] and att["total_s"] == rec["timed_seconds"] and att["formula"]
        if rec["kind"] == "harness":   # the host view sums to the LegClock time
            hv = att["host_view"]
            assert att["measurable"] and hv["parts"]["engine_call_s"] > 0
            assert hv["additive"] and sum(hv["parts"].values()) == pytest.approx(rec["timed_seconds"], abs=1e-3)
            assert att["gpu_view"]["measurable"] is False   # CPU / masked backends: no device
            if shape == "filtered":   # the reader part is the read + filter eval
                assert hv["labels"]["io_wait_s"] == "reader (read + filter eval)"
                assert att["filter"]["filter_s"] is not None and att["filter"]["filter_handoff_s"] is not None
        else:
            assert att["host_view"] is None and "engine_pct" not in att
        if rec["kind"] == "novabf":
            assert rec["env_added"]["NOVA_BF_DEVICE"] == "cpu" and rec["env_added"]["NOVABF_WARMUP_FILES"] == "1"
            assert rec["clock"]["state"] == "stopped" and rec["clock_agreement"]["ok"]
            assert rec["boundary"]["violations"] == []
            assert rec["timed_seconds"] == rec["clock"]["seconds"]
            assert rec["footer_read_s"] is None
            assert rec["stage_dirs_importable"] == []                    # run by path
            assert rec["native_textscan"]["status"]                      # recorded
            bb = rec["internal"]["bf_bench"]                             # unrounded split
            assert bb["files"] == 4 and bb["io_wait_s"] is not None and bb["gpu_s"] is not None
        else:
            assert rec["clock"]["events"] == ["start", "stop"] and rec["clock_agreement"]["ok"]
            assert rec["timed_seconds"] == rec["clock"]["seconds"]
            assert abs(rec["timed_seconds"] - rec["internal"]["total_s"]) <= L.HARNESS_CLOCK_TOL_S
            assert rec["clock_agreement"]["harness_total_s"] == rec["internal"]["total_s"]
            assert rec["counters"]["warmup_files"] == "1"
            # the footer pre-read recorded beside the headline, never in it
            assert rec["footer_read_s"] > 0
            assert "timed_seconds_with_footers" not in rec   # never summed
            assert rec["boundary"]["violations"] == [] and rec["boundary"]["rows"] == 4


@pytest.mark.slow
def test_e2e_leg_runs_first_without_a_warmup(runs):
    d = runs["dense"]["dir"]
    rec = json.load(open(os.path.join(d, "legs", "e2e__novabf.json")))
    assert rec["ok"], rec["failures"]
    assert rec["e2e"] and rec["protocol"]["warmup_files"] == 0 and rec["protocol"]["caches_cleared"] == []
    assert "NOVABF_WARMUP_FILES" not in rec["env_added"]
    assert rec["clock"]["state"] == "stopped" and rec["timed_seconds"] > 0
    assert rec["e2e_manifest_timing"]["elapsed_seconds"] > 0
    first = json.load(open(os.path.join(d, "legs", "dense__novabf__r1.json")))
    assert rec["ended_unix"] <= first["started_unix"]
    assert "e2e (cold" in open(os.path.join(d, "summary.md")).read()


def test_e2e_cache_clearing(tmp_path, monkeypatch):
    from bench import timing
    monkeypatch.setenv("HOME", str(tmp_path))
    for p in timing.E2E_CACHES:
        os.makedirs(os.path.expanduser(p) + "/x")
    assert sorted(timing.clear_e2e_caches()) == sorted(os.path.expanduser(p) for p in timing.E2E_CACHES)
    assert not any(os.path.exists(os.path.expanduser(p)) for p in timing.E2E_CACHES)


@pytest.mark.slow
def test_stale_leg_files_are_removed_before_a_leg(tmp_path, monkeypatch):
    exp, d = _prep(tmp_path)
    spec = L.plan(exp, "dense", "cpu", 1, d, dev=True)
    csv_p = spec["outputs"]["csv"]
    os.makedirs(os.path.dirname(csv_p), exist_ok=True)
    for p in (csv_p, csv_p[:-4] + ".trace.csv", spec["outputs"]["clock"]):
        open(p, "w").write("stale\n")
    rec = _leg(exp, d, "cpu", spec=spec, monkeypatch=monkeypatch)
    assert rec["ok"], rec["failures"]
    assert "stale" not in open(csv_p).read() and "stale" not in open(csv_p[:-4] + ".trace.csv").read()


@pytest.mark.slow
def test_legs_ran_in_the_rotated_order(runs):
    d = runs["filtered"]["dir"]
    exp = json.load(open(os.path.join(d, "experiment.json")))
    started = []
    for r, s, sy in runner.timing.schedule(exp):
        started.append(json.load(open(os.path.join(d, "legs", f"{s}__{sy}__r{r}.json")))["started_unix"])
    assert started == sorted(started)
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert man["schedule"][:5] == [[1, "filtered_text", "novabf"], [1, "filtered_text", "masked_bitmap"],
                                   [1, "filtered_text", "masked_files"], [1, "static_struct", "novabf"],
                                   [1, "static_struct", "masked_bitmap"]]
    assert man["schedule"][5:8] == [[2, "filtered_text", "masked_bitmap"], [2, "filtered_text", "masked_files"],
                                    [2, "filtered_text", "novabf"]]


@pytest.mark.slow
def test_novabf_and_the_cpu_reference_agree_on_dense(runs):
    """A sanity check that both systems searched the same data (verify/check.py does the real checks)."""
    d = runs["dense"]["dir"]
    a = np.load(os.path.join(d, "legs", "dense__novabf__r1.topk.npz"))
    b = np.load(os.path.join(d, "legs", "dense__cpu__r1.topk.npz"))
    np.testing.assert_allclose(a["D"], b["D"], rtol=0, atol=2e-5)


@pytest.mark.slow
def test_filtered_novabf_runs_one_search_per_config(runs):
    import yaml
    d = runs["filtered"]["dir"]
    c = yaml.safe_load(open(os.path.join(d, "configs", "static_struct__novabf__r1.yaml")))
    assert [s["name"] for s in c["searches"]] == ["static_struct"]
    assert c["corpus"]["path"] == "tests/fixtures/novabf/filtered/corpus"
    f = yaml.safe_load(open(os.path.join(d, "configs", "filter.yaml")))
    assert f["corpus"]["path"] == c["corpus"]["path"] and len(f["searches"]) == 4


# ---------------------------------------------------------------------------------------------
# failure injection
# ---------------------------------------------------------------------------------------------
def _prep(tmp_path, fixture="dense.yaml", sets=()):
    """(experiment, run dir) as bench run prepares them, without running the legs."""
    raw = runner._set(E.load_raw(os.path.join(FIX, fixture)), list(sets))
    exp = E.finish(raw, os.path.join(FIX, fixture))
    d = runner.new_run_dir(str(tmp_path / "res"))
    with open(os.path.join(d, "experiment.json"), "w") as f:
        f.write(E.dumps(exp))
    return json.load(open(os.path.join(d, "experiment.json"))), d


def _leg(exp, d, system, spec=None, search=None, monkeypatch=None):
    if monkeypatch is not None:
        for k, v in _env().items():
            monkeypatch.setenv(k, v)
        for k in [k for k in os.environ if k.startswith(("NOVA_BF_", "NOVABF_"))]:
            monkeypatch.delenv(k)
    return L.run_leg(exp, search or exp["searches"][0]["name"], system, 1, d, dev=True, spec=spec)


@pytest.mark.slow
def test_crashed_legs_are_excluded_and_fail_the_run(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(os.path.join(NBFIX, "dense", "corpus"), corpus)
    files = sorted(os.listdir(corpus))
    with open(corpus / files[1], "r+b") as f:   # file 0 (the warmup's) stays intact
        f.truncate(64)
    p = bench("run", os.path.join(FIX, "dense.yaml"), "--dev", "--rounds", "1",
              "--results", str(tmp_path / "res"), "--set", f"corpus.path={corpus}")
    assert p.returncode == 1, p.stdout[-2000:]
    d = _only_run(str(tmp_path / "res"), "s4_dense")
    for sy in ("novabf", "cpu"):
        rec = json.load(open(os.path.join(d, "legs", f"dense__{sy}__r1.json")))
        assert not rec["ok"] and rec["timed_seconds"] is None
        assert any("exit status" in f for f in rec["failures"]), rec["failures"]
    md = open(os.path.join(d, "summary.md")).read()
    assert "Excluded legs" in md and "dense novabf r1" in md and "dense cpu r1" in md
    rows = {r["system"]: r for r in SUM.summarize(d)[0]}
    assert rows["novabf"]["n"] == 0 and rows["novabf"]["median_s"] is None


@pytest.mark.slow
def test_missing_round_fails_summarize(runs, tmp_path):
    d = str(tmp_path / "run")
    shutil.copytree(runs["dense"]["dir"], d)
    os.remove(os.path.join(d, "legs", "dense__cpu__r2.json"))
    p = bench("summarize", d)
    assert p.returncode == 1
    assert "dense cpu: no usable round(s) [2]" in p.stdout
    os.remove(os.path.join(d, "legs", "dense__novabf__r1.json"))
    p = bench("summarize", d)
    assert p.returncode == 1 and "dense novabf: no usable round(s) [1]" in p.stdout


@pytest.mark.slow
def test_unreadable_record_and_failed_checks_fail_summarize(runs, tmp_path):
    d = str(tmp_path / "run")
    shutil.copytree(runs["dense"]["dir"], d)
    assert bench("summarize", d).returncode == 0
    json.dump({"ok": False}, open(os.path.join(d, "checks.json"), "w"))
    assert bench("summarize", d).returncode == 1
    json.dump({"ok": None}, open(os.path.join(d, "checks.json"), "w"))
    open(os.path.join(d, "legs", "dense__cpu__r1.json"), "w").write("{trunc")
    p = bench("summarize", d)
    assert p.returncode == 1 and "unreadable leg record" in p.stdout


@pytest.mark.slow
@pytest.mark.parametrize("old,new,msg", [
    (["--warmup-files", "1"], ["--warmup-files", "2"], "warmup pass over warmup_files=2"),
    (["--warmup", "1"], ["--warmup", "0"], "exit status 2"),   # the harness refuses --warmup-files without a warmup
    (["--repeats", "1"], ["--repeats", "2"], "2 timed pass(es)"),
    (["--warmup", "1"], ["--warmup", "2"], "2 warmup pass(es) in the CSV"),
])
def test_harness_warmup_or_repeats_changed_is_caught(tmp_path, monkeypatch, old, new, msg):
    exp, d = _prep(tmp_path)
    spec = L.plan(exp, "dense", "cpu", 1, d, dev=True)
    a = spec["argv"]
    i = next(i for i in range(len(a) - 1) if a[i:i + 2] == old)
    spec["argv"] = a[:i] + new + a[i + 2:]
    rec = _leg(exp, d, "cpu", spec=spec, monkeypatch=monkeypatch)
    assert not rec["ok"] and rec["timed_seconds"] is None
    assert any(msg in f for f in rec["failures"]), rec["failures"]


@pytest.mark.slow
def test_harness_leg_without_its_clock_is_caught(tmp_path, monkeypatch):
    exp, d = _prep(tmp_path)
    spec = L.plan(exp, "dense", "cpu", 1, d, dev=True)
    i = spec["argv"].index("--clock-json")
    spec["argv"] = spec["argv"][:i] + spec["argv"][i + 2:]
    rec = _leg(exp, d, "cpu", spec=spec, monkeypatch=monkeypatch)
    assert not rec["ok"] and rec["timed_seconds"] is None
    assert any("0 LegClock record(s)" in f for f in rec["failures"]), rec["failures"]


HARNESS_BOUNDARY = r"""
import json, runpy, sys
import faiss
import bench.timing as C
import bench.data.sources as S
ev, cur = [], {"clock": None}
def st():
    c = cur["clock"]
    return None if c is None else c.state
_start, _stop = C.LegClock.start, C.LegClock.stop
def start(self):
    cur["clock"] = self; _start(self); ev.append(["clock.start", None])
def stop(self):
    _stop(self); ev.append(["clock.stop", None])
C.LegClock.start, C.LegClock.stop = start, stop
for name in dir(S):
    cls = getattr(S, name)
    if isinstance(cls, type) and hasattr(cls, "stream") and "__iter__" in vars(cls):
        def wrap(f, label):
            def w(self, *a, **k):
                ev.append([label, st()])
                return f(self, *a, **k)
            return w
        cls.__iter__ = wrap(cls.__iter__, "source.iter")
        cls.stream = wrap(cls.stream, "source.stream")
_fin = faiss.ResultHeap.finalize
def fin(self, *a):
    ev.append(["heap.finalize", st()])
    return _fin(self, *a)
faiss.ResultHeap.finalize = fin
out, argv = sys.argv[1], sys.argv[2:]
sys.argv = [argv[0]] + argv[1:]
try:
    runpy.run_path(argv[0], run_name="__main__")
except SystemExit as e:
    assert not e.code, e.code
json.dump(ev, open(out, "w"))
"""


@pytest.mark.slow
def test_harness_clock_boundary(tmp_path, monkeypatch):
    """The harness's LegClock: the warmup pass runs without a clock; the timed
    pass's source is iterated (readers started, the first corpus byte requested) only after
    the clock started, its final finalize() runs while the clock runs, and the clock stops
    after it. The real harness, the real leg argv, a recorder around the clock / sources /
    ResultHeap.finalize."""
    exp, d = _prep(tmp_path)
    spec = L.plan(exp, "dense", "cpu", 1, d, dev=True)
    out = tmp_path / "events.json"
    p = subprocess.run([sys.executable, "-c", HARNESS_BOUNDARY, str(out), *spec["argv"][1:]],
                       cwd=CB, env={**_env(), **spec["env_add"]}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    ev = json.load(open(out))
    i0 = ev.index(["clock.start", None])
    warm, timed = ev[:i0], ev[i0:]
    assert [e[0] for e in warm if e[0].startswith("source")] and all(e[1] is None for e in warm)
    assert [e[0] for e in warm].count("heap.finalize") == 1
    reads = [e for e in timed if e[0].startswith("source")]
    assert reads and all(e[1] == "running" for e in reads), timed
    assert [e for e in timed if e[0] == "heap.finalize"] == [["heap.finalize", "running"]]
    assert timed[-1] == ["clock.stop", None]
    assert ev.count(["clock.start", None]) == 1


@pytest.mark.slow
@pytest.mark.parametrize("value,msg", [("2", "warmup_files=2 (want 1)"), (None, "warmup_files=0 (want 1)")])
def test_novabf_warmup_changed_is_caught(tmp_path, monkeypatch, value, msg):
    exp, d = _prep(tmp_path)
    spec = L.plan(exp, "dense", "novabf", 1, d, dev=True)
    if value is None:
        del spec["env_add"]["NOVABF_WARMUP_FILES"]
    else:
        spec["env_add"]["NOVABF_WARMUP_FILES"] = value
    rec = _leg(exp, d, "novabf", spec=spec, monkeypatch=monkeypatch)
    assert not rec["ok"] and rec["timed_seconds"] is None
    assert any(msg in f for f in rec["failures"]), rec["failures"]


# the nova-bf child with ONE hook changed (python -c: patch bench.timing.ClockHooks, then the
# adapter's own main with the leg's own arguments)
LATE_START = """
import bench.timing as c
c.ClockHooks.corpus_start = lambda self: None            # the clock does not start here...
_p = c.ClockHooks.pause
def pause(self):
    if self.clock.state == "new":
        self.clock.start()                                # ...but only at the first pause
    _p(self)
c.ClockHooks.pause = pause
"""
EARLY_WRITE = """
import bench.timing as c, tempfile, pyarrow as pa
_s = c.ClockHooks.corpus_start
def corpus_start(self):
    _s(self)
    from nova_bf.io import Store                          # an output write while timed, no pause
    Store(tempfile.mkdtemp()).write("stray.parquet", pa.table({"a": [1]}))
c.ClockHooks.corpus_start = corpus_start
"""


def _patched(spec, patch):
    a = spec["argv"]
    assert a[1] == L.NOVABF_ADAPTER    # by path
    spec["argv"] = [a[0], "-c", patch + f"\nimport sys\nfrom bench.systems import novabf\n"
                    f"sys.exit(novabf.main({a[2:]!r}))\n"]
    return spec


@pytest.mark.slow
@pytest.mark.parametrize("patch,msg", [(LATE_START, "corpus data read before corpus_start"),
                                       (EARLY_WRITE, "output write while timed with no pause")])
def test_novabf_boundary_violations_are_caught(tmp_path, monkeypatch, patch, msg):
    exp, d = _prep(tmp_path)
    spec = _patched(L.plan(exp, "dense", "novabf", 1, d, dev=True), patch)
    rec = _leg(exp, d, "novabf", spec=spec, monkeypatch=monkeypatch)
    assert rec["exit_code"] == 0                       # nova-bf itself finished...
    assert not rec["ok"] and rec["timed_seconds"] is None   # ...but the leg has no time
    assert any(msg in f for f in rec["failures"]), rec["failures"]


@pytest.mark.slow
def test_unpatched_child_command_passes(tmp_path, monkeypatch):
    """The control for the two injections above: the same python -c route, no patch."""
    exp, d = _prep(tmp_path)
    spec = _patched(L.plan(exp, "dense", "novabf", 1, d, dev=True), "")
    rec = _leg(exp, d, "novabf", spec=spec, monkeypatch=monkeypatch)
    assert rec["ok"], rec["failures"]


@pytest.mark.slow
def test_multi_search_novabf_writes_inside_pauses_pass(tmp_path, monkeypatch):
    """several searches, each write submitted inside its pause (it may run while a
    later sort is timed): allowed."""
    import yaml
    c = yaml.safe_load(open(os.path.join(FIX, "nb_dense.yaml")))
    c["searches"] = [dict(c["searches"][0], name=f"d_{m}{k}", metric=m, k=k)
                     for m, k in (("cosine", 10), ("dot", 10), ("cosine", 5))]
    cfg = tmp_path / "multi.yaml"
    cfg.write_text(yaml.safe_dump(c))
    exp, d = _prep(tmp_path, sets=[f"systems.novabf.config={cfg}"])
    rec = _leg(exp, d, "novabf", monkeypatch=monkeypatch)
    assert rec["ok"], rec["failures"]
    assert rec["clock"]["pauses"] == 3
    writes = rec["boundary"]["writes"]
    assert sum(w["method"] == "write" for w in writes) == 3


def test_boundary_guard_rules():
    """The guard's accounting on a scripted clock: writes inside pauses, later while running
    (allowed while writes <= pauses), one too many, a read before start, a read after done."""
    from bench.timing import LegClock
    from bench.systems.novabf import BoundaryGuard

    class Store:
        def read_columns(self, path, cols=None):
            return None

        def read_schema(self, path):
            return None

        def write(self, name, table):
            return name

    clock = LegClock()
    g = BoundaryGuard(clock, "/c")
    rd = g._read("read_columns", Store.read_columns)
    sch = g._read("read_schema", Store.read_schema)
    wr = g._write("write", Store.write)
    s = Store()
    sch(s, "/c/f0")                 # footer before start: allowed
    rd(s, "/q/queries.parquet")     # not the corpus: not counted
    assert g.violations == []
    rd(s, "/c/f0")
    assert "read before corpus_start" in g.violations[-1]
    g.violations.clear()
    clock.start()
    rd(s, "/c/f1")
    clock.pause()
    wr(s, "a", None)                # inside the pause
    clock.resume()
    wr(s, "b", None)                # running, 2 writes vs 1 pause: not accounted for
    assert len(g.violations) == 1 and "no pause to account" in g.violations[0]
    g.violations.clear()
    clock.pause()
    clock.resume()
    wr(s, "c", None)                # 3 writes, 2 pauses: one too many again
    assert len(g.violations) == 1
    g.violations.clear()
    clock.pause()                   # 3 pauses
    clock.resume()
    clock.stop()
    wr(s, "manifest", None)         # after done: fine
    rd(s, "/c/f2")
    assert g.violations == ["corpus data read after done: read_columns(/c/f2)"]


# ---------------------------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_cpu_sweep_writes_its_table_and_pick_chooses_a_checked_variant(tmp_path):
    """A sparse sweep runs and checks every variant; --pick prints the winner's YAML change (and
    never edits a file)."""
    res = str(tmp_path / "res")
    p = bench("sweep", os.path.join(FIX, "sweep_sparse.yaml"), "--dev", "--results", res)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    d = _only_run(os.path.join(res, "sweeps"), "sweep_sparse")
    for f in ("sweep.json", "sweep_table.csv", "sweep_table.md", "summary.csv", "manifest.json"):
        assert os.path.isfile(os.path.join(d, f)), f
    import csv
    rows = list(csv.DictReader(open(os.path.join(d, "sweep_table.csv"))))
    assert [r["variant"] for r in rows] == ["novabf", "cpu-sparse_project-off", "cpu-sparse_project-query-vocab"]
    assert all(r["complete"] == "True" and float(r["median_s"]) > 0 for r in rows)
    assert all(r["checks_ok"] == "True" for r in rows), rows
    for r in json.load(open(os.path.join(d, "experiment.json")))["searches"][0]["systems"]:
        rec = json.load(open(os.path.join(d, "legs", f"sparse__{r}__r1.json")))
        assert rec["n_files"] == 2     # the subset: harness --max-files 2 / NOVABF_MAX_FILES=2
    q = bench("sweep", "--pick", os.path.join(FIX, "sweep_sparse.yaml"), "--results", res)
    assert "cpu: winner cpu-sparse_project-" in q.stdout and "REFUSED" not in q.stdout, q.stdout + q.stderr
    # --pick never edits a file
    assert open(os.path.join(FIX, "sparse.yaml")).read().count("chosen_from") == 0


@pytest.mark.slow
def test_placeholders_and_dev_refusals(tmp_path):
    p = bench("run", os.path.join(CB, "tests", "fixtures", "experiments", "sparse", "fineweb.yaml"), "--dev",
              "--results", str(tmp_path))
    assert p.returncode == 2 and "placeholder" in p.stderr
    p = bench("run", os.path.join(CB, "tests", "fixtures", "experiments", "dense", "coyo.yaml"), "--dev",
              "--results", str(tmp_path))
    assert p.returncode == 2 and "does not run on a CPU dev box" in p.stderr
    assert not os.listdir(tmp_path)   # nothing started


@pytest.mark.slow
def test_strict_env_check_refuses_on_a_dev_box(tmp_path):
    """Without --dev the env check is strict: on this box (not the paper env) nothing runs."""
    p = bench("run", os.path.join(FIX, "dense.yaml"), "--results", str(tmp_path))
    assert p.returncode == 3, p.stdout[-2000:] + p.stderr[-2000:]
    d = _only_run(str(tmp_path), "s4_dense")
    assert os.listdir(os.path.join(d, "legs")) == []


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["dense", "filtered"])
def test_summary_and_manifest_carry_the_s7_fields(runs, shape):
    """summary.csv has both time medians and the attribution for EVERY system;
    the manifest records prep, the nova-bf text scanner, the dirty paths."""
    import csv as _csv
    d = runs[shape]["dir"]
    rows = list(_csv.DictReader(open(os.path.join(d, "summary.csv"))))
    for r in rows:
        assert r["attribution_kind"] in ("harness", "novabf")
        assert (float(r["engine_call_s"]) > 0) if r["attribution_kind"] == "harness" else r["engine_call_s"] == ""
        assert "median_with_footers_s" not in r
        if r["system"] == "novabf":
            assert r["median_footer_read_s"] == ""
        else:
            assert float(r["median_footer_read_s"]) > 0
    md = open(os.path.join(d, "summary.md")).read()
    assert "median footer read s" in md and "Where the time goes" in md and "never added to median_s" in md
    m = json.load(open(os.path.join(d, "manifest.json")))
    assert set(m["prep"]) == {"stage", "queries", "queries_src"}
    assert m["novabf_native_textscan"] and "dirty_paths" in m
