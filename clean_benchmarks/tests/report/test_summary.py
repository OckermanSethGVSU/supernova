"""bench/report/summary.py: which leg records count, the median of the rounds, and the
refusals."""

import json
import math
import os
import subprocess
import sys

import pytest

from bench.report import summary as SUM
from bench.systems import base as SB

from helpers.leg_records import _leg_rec, _sum_exp
from helpers.leg_records import rd  # noqa: F401 (fixture)


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stamp(d, **doc):
    """checks.json as the checks write it: with the recorded inputs (size and sha256 of each file read), here ok."""
    from bench.verify import check
    json.dump({"ok": True, **doc, "inputs": check.run_inputs(d)}, open(os.path.join(d, "checks.json"), "w"))


@pytest.fixture
def sdir(tmp_path):
    os.makedirs(tmp_path / "legs")
    json.dump({"ok": True}, open(tmp_path / "checks.json", "w"))
    return str(tmp_path)


def test_summary_median_of_three_rounds_with_an_outlier(sdir):
    for r, t in ((1, 1.0), (2, 1.1), (3, 9.0)):
        _leg_rec(sdir, "s", "a", r, t)
        _leg_rec(sdir, "s", "b", r, 2.0)
    _stamp(sdir)
    rows, problems, _, _ = SUM.summarize(sdir, _sum_exp())
    a = next(x for x in rows if x["system"] == "a")
    assert problems == [] and a["median_s"] == 1.1          # the median, not the mean (3.7)


def test_summary_excludes_a_failed_leg_even_with_a_time(sdir):
    for r in (1, 2, 3):
        _leg_rec(sdir, "s", "a", r, 1.0, ok=(r != 2))
        _leg_rec(sdir, "s", "b", r, 2.0)
    rows, problems, excluded, _ = SUM.summarize(sdir, _sum_exp())
    a = next(x for x in rows if x["system"] == "a")
    assert a["n"] == 2 and any(e[1] == 2 for e in excluded)
    assert any("no usable round(s) [2]" in p for p in problems)


def test_summary_excludes_novabf_covering_other_files(sdir):
    for r in (1, 2, 3):
        _leg_rec(sdir, "s", "a", r, 1.0, kind="novabf", n_files=3)
        _leg_rec(sdir, "s", "b", r, 2.0, n_files=4)
    rows, problems, excluded, _ = SUM.summarize(sdir, _sum_exp())
    assert any("covers files=3, expected 4" in p for p in problems) and len(excluded) == 3


def test_summary_counts_only_this_runs_rounds_and_systems(sdir):
    """a leftover round-4 record and a record of a system the experiment does not run
    are listed as problems and never enter a median."""
    for r in (1, 2, 3):
        _leg_rec(sdir, "s", "a", r, 1.0)
        _leg_rec(sdir, "s", "b", r, 2.0)
    _leg_rec(sdir, "s", "a", 4, 0.001)
    _leg_rec(sdir, "s", "zz", 1, 0.001)
    rows, problems, _, _ = SUM.summarize(sdir, _sum_exp())
    a = next(x for x in rows if x["system"] == "a")
    assert a["n"] == 3 and a["median_s"] == 1.0
    assert not any(x["system"] == "zz" for x in rows)
    assert any("s__a__r4" in p and "not a leg of this run" in p for p in problems)
    assert any("s__zz__r1" in p for p in problems)


def test_summary_requires_passing_checks(sdir):
    """no checks.json, or ok not true, is a problem; --no-checks waives it."""
    for r in (1, 2, 3):
        _leg_rec(sdir, "s", "a", r, 1.0)
        _leg_rec(sdir, "s", "b", r, 2.0)
    _stamp(sdir)
    assert SUM.summarize(sdir, _sum_exp())[1] == []
    os.remove(os.path.join(sdir, "checks.json"))
    assert any("no readable checks.json" in p for p in SUM.summarize(sdir, _sum_exp())[1])
    assert SUM.summarize(sdir, _sum_exp(), require_checks=False)[1] == []
    json.dump({"ok": None}, open(os.path.join(sdir, "checks.json"), "w"))
    assert any("checks failed (ok=None" in p for p in SUM.summarize(sdir, _sum_exp())[1])
    json.dump(_sum_exp(), open(os.path.join(sdir, "experiment.json"), "w"))
    p = subprocess.run([sys.executable, "-m", "bench", "summarize", sdir, "--no-checks"], cwd=CB,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_a_missing_e2e_record_is_a_problem(sdir):
    """the experiment has an e2e leg, there is no record of it."""
    for r in (1, 2, 3):
        _leg_rec(sdir, "s", "a", r, 1.0)
        _leg_rec(sdir, "s", "b", r, 2.0)
    problems = SUM.summarize(sdir, _sum_exp(e2e={"system": "a"}))[1]
    assert any("no e2e leg record" in p for p in problems)


def test_summary_md_shows_the_label(tmp_path):
    os.makedirs(tmp_path / "legs")
    json.dump({"ok": True}, open(tmp_path / "checks.json", "w"))
    exp = {"protocol": {"rounds": 1}, "e2e": None,
           "searches": [{"name": "s", "systems": ["cuvs_p", "spare_b"]}],
           "systems": {"cuvs_p": SB.resolve("cuvs_p", {"system": "cuvs", "sparse_project": "query-vocab"}),
                       "spare_b": SB.resolve("spare_b", {"system": "spare", "preset": "batched_dot"})}}
    for sy in ("cuvs_p", "spare_b"):
        json.dump({"tag": f"s__{sy}__r1", "search": "s", "system": sy, "round": 1, "e2e": False,
                   "kind": "harness", "ok": True, "failures": [], "timed_seconds": 1.0,
                   "time_source": "LegClock", "n_files": 2, "peak_rss_bytes": 1, "bytes_read": 0},
                  open(tmp_path / "legs" / f"s__{sy}__r1.json", "w"))
    rows, problems, excluded, e2e = SUM.summarize(str(tmp_path), exp)
    md = SUM.write(str(tmp_path), rows, problems, excluded, e2e)
    assert "| LegClock | stock +proj |" in md
    assert "| LegClock | batched_dot (ours: SPARe batched) |" in md
    import csv
    got = {r["system"]: r["label"] for r in csv.DictReader(open(tmp_path / "summary.csv"))}
    assert got == {"cuvs_p": "stock +proj", "spare_b": "batched_dot (ours: SPARe batched)"}


def test_the_summary_labels_the_e2e_numbers(tmp_path):
    os.makedirs(tmp_path / "legs")
    md = SUM.write(str(tmp_path), [], [], [], [{
        "system": "novabf", "ok": True, "timed_seconds": 12.5, "process_wall_s": 20.25,
        "e2e_manifest_timing": {"elapsed_seconds": 15.0}}])
    line = next(x for x in md.splitlines() if x.startswith("- novabf"))
    assert "LegClock (nova-bf's timed scan, the same clock as every leg) 12.5000 s" in line
    assert "process wall (/usr/bin/time: the whole leg process" in line and "20.2500 s" in line
    assert "nova-bf manifest elapsed 15.0000 s" in line


def _leg(d, search, system, rnd, secs, ok=True, kind="harness", n_files=4, fname=None):
    rec = {"tag": f"{search}__{system}__r{rnd}", "search": search, "system": system, "round": rnd,
           "e2e": False, "kind": kind, "ok": ok, "failures": [] if ok else ["boom"],
           "timed_seconds": secs, "time_source": "LegClock", "n_files": n_files,
           "peak_rss_bytes": 1, "bytes_read": 0}
    json.dump(rec, open(os.path.join(d, "legs", (fname or rec["tag"]) + ".json"), "w"))


def _sexp(systems=("a", "b"), rounds=3):
    return {"protocol": {"rounds": rounds}, "searches": [{"name": "s", "systems": list(systems)}],
            "systems": {x: {"preset": "stock", "knobs": {}} for x in systems}, "e2e": None}


def test_summary_flags_harness_legs_of_one_search_covering_different_file_counts(rd):
    for r in (1, 2, 3):
        _leg(rd, "s", "a", r, 1.0, n_files=4)
        _leg(rd, "s", "b", r, 0.5, n_files=3)
    _, problems, _, _ = SUM.summarize(rd, _sexp())
    assert any("different file counts [3, 4]" in p for p in problems)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_summary_requires_a_finite_positive_time(rd, bad):
    for r in (1, 2, 3):
        _leg(rd, "s", "a", r, bad if r == 2 else 1.0)
        _leg(rd, "s", "b", r, 2.0)
    rows, problems, excluded, _ = SUM.summarize(rd, _sexp())
    a = next(x for x in rows if x["system"] == "a")
    assert a["n"] == 2 and math.isfinite(a["median_s"]) and a["median_s"] == 1.0
    assert any("s a r2: excluded: no usable time" in p for p in problems)


def test_summary_refuses_a_renamed_copy_and_a_second_record_of_a_round(rd):
    for r in (1, 2, 3):
        _leg(rd, "s", "a", r, 5.0, ok=(r != 2))
        _leg(rd, "s", "b", r, 2.0)
    _leg(rd, "s", "a", 2, 1.0, fname="zz_copy_of_s__a__r2")     # tag != file stem
    rows, problems, _, _ = SUM.summarize(rd, _sexp())
    assert next(x for x in rows if x["system"] == "a")["rounds_s"] == "5.0000 - 5.0000"
    assert any("zz_copy_of_s__a__r2.json: its tag" in p for p in problems)
    # a record whose fields name another leg than its tag (a hand-made duplicate of r1)
    rec = json.load(open(os.path.join(rd, "legs", "s__b__r1.json")))
    rec.update(tag="s__b__r9", round=1)
    json.dump(rec, open(os.path.join(rd, "legs", "s__b__r9.json"), "w"))
    rows, problems, _, _ = SUM.summarize(rd, _sexp())
    assert any("s__b__r9: the record says" in p for p in problems)


def test_summary_counts_a_round_only_once_when_two_records_claim_it(rd, monkeypatch):
    for r in (1, 2, 3):
        _leg(rd, "s", "a", r, 1.0)
        _leg(rd, "s", "b", r, 2.0)
    real = SUM.load_legs

    def twice(run_dir):   # two readable records of s a r2 (e.g. a second results tree merged in)
        legs = real(run_dir)
        return legs + [dict(next(x for x in legs if x["tag"] == "s__a__r2"), _stem="s__a__r2x",
                            tag="s__a__r2x")]
    monkeypatch.setattr(SUM, "load_legs", twice)
    _, problems, _, _ = SUM.summarize(rd, _sexp())
    assert any("not the file name" in p or "the record says" in p for p in problems)


def test_summary_carries_the_run_identity(rd):
    import csv
    import hashlib
    for r in (1, 2, 3):
        _leg(rd, "s", "a", r, 1.0)
        _leg(rd, "s", "b", r, 2.0)
    json.dump({"commit": "abc123", "dirty": True, "code_sha": "c0de", "stack_sha": "57ac"},
              open(os.path.join(rd, "manifest.json"), "w"))
    open(os.path.join(rd, "experiment.json"), "w").write('{"name": "x"}')
    rows, problems, excluded, e2e = SUM.summarize(rd, _sexp())
    md = SUM.write(rd, rows, problems, excluded, e2e)
    sha = hashlib.sha256(b'{"name": "x"}').hexdigest()
    head = md.splitlines()[:2]
    assert "commit abc123" in head[0] and "dirty True" in head[0] and "code_sha c0de" in head[0]
    assert "stack_sha 57ac" in head[0] and sha in head[1] and "checks ok: True" in head[1]
    out = list(csv.DictReader(open(os.path.join(rd, "summary.csv"))))
    assert {r["run_commit"] for r in out} == {"abc123"} and {r["experiment_sha256"] for r in out} == {sha}
    assert {r["checks_ok"] for r in out} == {"True"} and {r["run_dirty"] for r in out} == {"True"}
