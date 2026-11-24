"""bench check-copies / check-verify (bench/verify/steps.py) and bench plot.

The planned steps (which runs, which mode-only knobs, files / repeats precedence, the check
json), the refusals, and the real commands on the fixture experiments (CPU).
"""

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from bench import experiment as E
from bench import runner
from bench.systems import base, pylate_engine, spare_engine
from bench.verify import steps
from helpers.step_plans import _load

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(CB, "experiments")
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NOVA_BF_", "NOVABF_", "PAPER_"))}
    env.update(HARNESS_NO_IMDS="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    return env


def test_spare_mode_only_knobs(tmp_path):
    exp, d = _load("sparse_fineweb", tmp_path)
    exp["systems"]["spare"]["knobs"].update(spare_max_mib=4096, no_spare_events=True)
    by = {r["name"]: r["argv_tail"] for r in steps.plan_verify(exp, d)}
    for m in spare_engine.SPARE_MODES:
        assert ("--spare-max-mib" in by[f"spare_{m}"]) == (not spare_engine.spare_released(m)), m
        assert ("--no-spare-events" in by[f"spare_{m}"]) == (m in spare_engine.SPARE_STAGED_MODES), m


def test_pylate_mode_only_knobs(tmp_path):
    exp, d = _load("multivector_pubmed", tmp_path)
    exp["systems"]["pylate_scores-batched"]["knobs"].update(pylate_rerank_results="host",
                                                            pylate_query_buckets=2)
    by = {r["name"]: r["argv_tail"] for r in steps.plan_verify(exp, d)}
    for m in pylate_engine.PYLATE_MODES:
        a = by[f"pylate_{m}"]
        assert ("--pylate-rerank-results" in a) == pylate_engine.pylate_released(m), m
        assert ("--pylate-query-buckets" in a) == (not pylate_engine.pylate_released(m)), m


def test_mode_lists_are_the_engines():
    assert base.SYSTEMS["spare"].knobs["spare_mode"].choices is spare_engine.SPARE_MODES
    assert base.SYSTEMS["pylate"].knobs["pylate_mode"].choices is pylate_engine.PYLATE_MODES


def test_a_new_engine_mode_is_verified_without_editing_steps(tmp_path, monkeypatch):
    spec = base.SYSTEMS["spare"]
    kn = dict(spec.knobs, spare_mode=dataclasses.replace(
        spec.knobs["spare_mode"], choices=spec.knobs["spare_mode"].choices + ("fake-released",)))
    monkeypatch.setitem(base.SYSTEMS, "spare", dataclasses.replace(spec, knobs=kn))
    exp, d = _load("sparse_fineweb", tmp_path)
    exp["systems"]["spare"]["knobs"]["spare_max_mib"] = 4096
    by = {r["name"]: r["argv_tail"] for r in steps.plan_verify(exp, d)}
    a = by["spare_fake-released"]
    assert a[a.index("--spare-mode") + 1] == "fake-released" and "--spare-max-mib" not in a


def test_every_modality_gets_both_checks(tmp_path):
    """Derived from the systems, not a modality list: dense verifies its harness systems,
    filtered copy-checks its GPU ones."""
    exp, d = _load("dense_coyo", tmp_path)
    assert [r["name"] for r in steps.plan_verify(exp, d)] == ["faiss", "cuvs"]
    exp, d = _load("filtered_fineweb_smoke", tmp_path)
    names = {r["system"] for r in steps.plan_copies(exp, d)}
    assert names == {n for n, sy in exp["systems"].items() if sy["kind"] == "harness"}


def test_copies_refused_without_a_gpu_harness_system(tmp_path):
    exp = E.finish(E.load_raw(os.path.join(FIX, "dense.yaml")), os.path.join(FIX, "dense.yaml"))
    with pytest.raises(steps.StepError, match="no GPU harness system"):
        steps.plan_copies(exp, str(tmp_path))
    assert [r["name"] for r in steps.plan_verify(exp, str(tmp_path))] == ["cpu"]


def _files(r):
    a = r["argv_tail"]
    assert a.count("--max-files") == 1
    return int(a[a.index("--max-files") + 1]), int(a[a.index("--repeats") + 1])


def test_files_and_repeats_precedence(tmp_path):
    """flag > experiment checks: > subset: > default"""
    exp, d = _load("dense_coyo", tmp_path)
    first = lambda kind, **kw: _files((steps.plan_copies if kind == "copies" else steps.plan_verify)(
        exp, d, steps.settings(exp, kind, **kw))[0])
    assert first("copies") == (4, 1)
    exp["subset"] = {"files": 3}
    assert first("copies") == (3, 1)
    exp["checks"].update(copies_files=7, verify_files=5, check_repeats=2)
    assert first("copies") == (7, 2) and first("verify") == (5, 2)
    assert first("copies", files=9, repeats=3) == (9, 3)


@pytest.mark.parametrize("checks", [{"copies_files": 0}, {"check_repeats": True},
                                    {"verify_files": "2"}, {"check_files": 2}])
def test_checks_keys_are_validated(checks, tmp_path):
    raw = runner._set(E.load_raw(os.path.join(FIX, "dense.yaml")), [])
    raw["checks"] = checks
    with pytest.raises(E.ExperimentError):
        E.finish(raw, os.path.join(FIX, "dense.yaml"))


@pytest.mark.parametrize("kind", ["copies", "verify"])
def test_check_json_records_the_settings(kind, tmp_path, monkeypatch):
    exp, d = _load("sparse_fineweb", tmp_path)

    def spawn(spec):
        os.makedirs(os.path.dirname(spec["log"]), exist_ok=True)
        open(spec["log"], "w").write("verify OK\n")
        return {"exit_code": 0}
    monkeypatch.setattr(steps.leg_mod, "spawn", spawn)
    monkeypatch.setattr(steps.agree, "same_topk", lambda a, b: True)
    cfg = steps.settings(exp, kind, files=3, warmup=1, warmup_files=1)
    doc = steps.execute(kind, exp, d, False, cfg)
    assert doc["ok"] and doc["settings"] == {"files": 3, "repeats": 1, "warmup": 1, "warmup_files": 1}
    assert [v["name"] for v in doc["variants"]] == list(doc["runs"])
    if kind == "copies":
        assert doc["copies"] == {"serial": steps.SERIAL, "overlap": steps.OVERLAP}
        assert set(doc["pairs"]) == {"cuvs", "spare", "cuvs_proj", "spare_proj"}
    else:
        assert "copies" not in doc
        assert [v["mode"] for v in doc["variants"] if v["system"] == "spare"] == list(spare_engine.SPARE_MODES)
    for r in doc["runs"].values():
        assert r["argv"][r["argv"].index("--max-files") + 1] == "3" and "--warmup-files" in r["argv"]


def bench(*args):
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=CB, env=_env(),
                          capture_output=True, text=True)


@pytest.mark.slow
def test_check_verify_runs_on_cpu(tmp_path):
    """The masked / cpu references on the filtered fixture, through the real command."""
    pytest.importorskip("torch")
    p = bench("check-verify", os.path.join(FIX, "filtered.yaml"), "--dev", "--results", str(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    assert "bench check-verify: PASS" in p.stdout
    d = os.path.join(tmp_path, "s4_filtered", "check-verify")
    doc = json.load(open(os.path.join(d, os.listdir(d)[0], "check.json")))
    assert set(doc["runs"]) == {"filtered_text__masked_bitmap", "filtered_text__masked_files",
                                "static_struct__masked_bitmap"}
    for r in doc["runs"].values():
        assert r["ok"] and r["verify_lines"][-1].startswith("verify OK")
        assert r["argv"][r["argv"].index("--max-files") + 1] == "4"


@pytest.mark.slow
def test_check_verify_fails_on_a_broken_variant(tmp_path):
    pytest.importorskip("torch")
    import shutil
    corpus = tmp_path / "corpus"
    shutil.copytree(os.path.join(CB, "tests", "fixtures", "novabf", "sparse", "corpus"), corpus)
    f = sorted(os.listdir(corpus))[1]
    with open(corpus / f, "r+b") as fh:
        fh.truncate(64)
    # the old sparse verify step runs the experiment's harness systems: here the cpu reference
    p = bench("check-verify", os.path.join(FIX, "sparse.yaml"), "--dev", "--results", str(tmp_path / "r"),
              "--set", f"corpus.path={corpus}")
    assert p.returncode == 1 and "bench check-verify: FAIL" in p.stdout, p.stdout[-2000:]


def test_check_copies_refuses_gpu_systems_on_the_dev_box(tmp_path):
    p = bench("check-copies", os.path.join(CB, E.path_of("dense_coyo")), "--dev", "--results", str(tmp_path))
    assert p.returncode == 2 and "does not run on a CPU" in p.stderr


def test_check_copies_plan_command(tmp_path):
    p = bench("check-copies", os.path.join(CB, E.path_of("dense_coyo")), "--plan", "--results", str(tmp_path))
    assert p.returncode == 0, p.stderr
    lines = p.stdout.strip().splitlines()
    assert [ln.split()[0] for ln in lines] == ["faiss_serial", "faiss_overlap", "cuvs_serial", "cuvs_overlap"]
    assert all("bench/systems/harness_cli.py" in ln and "--max-files 4" in ln for ln in lines)
    p = bench("check-copies", os.path.join(CB, E.path_of("dense_coyo")), "--plan", "--results", str(tmp_path),
              "--files", "16", "--repeats", "2", "--warmup", "1", "--warmup-files", "1")
    assert all("--max-files 16" in ln and "--warmup 1 --repeats 2 --warmup-files 1" in ln
               for ln in p.stdout.strip().splitlines())
    assert "--files 16 --warmup 1" in bench("check-copies", "--help").stdout


@pytest.mark.slow
def test_bench_plot_draws_every_search_from_saved_results(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("torch")
    res = tmp_path / "res"
    p = bench("run", os.path.join(FIX, "filtered.yaml"), "--dev", "--rounds", "1", "--results", str(res))
    assert p.returncode == 0, p.stdout[-2000:]
    d = os.path.join(res, "s4_filtered", os.listdir(os.path.join(res, "s4_filtered"))[0])
    p = bench("plot", d)
    assert p.returncode == 0, p.stdout + p.stderr
    for s in ("filtered_text", "static_struct"):
        assert os.path.getsize(os.path.join(d, "figures", f"{s}_attribution.png")) > 0
        assert "masked" in open(os.path.join(d, "figures", f"{s}.txt")).read()
    assert bench("plot", str(tmp_path)).returncode == 2
