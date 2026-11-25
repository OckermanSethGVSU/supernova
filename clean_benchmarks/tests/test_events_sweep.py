"""The GPU-events check (an events sweep over the gpu_events knob, report only), the knob
itself, and the one-run default for sweeps."""

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys

import pytest

from bench import experiment as E
from bench import leg as L
from bench import runner as R
from bench.systems import base as SB

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


def bench(*args, timeout=900):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "HARNESS_NO_IMDS": "1"}
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=CB, capture_output=True,
                          text=True, timeout=timeout, env=env)


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------------------------
# the gpu_events knob
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("system", ["faiss", "cuvs", "spare", "pylate", "cpu", "masked"])
def test_gpu_events_knob_default_on_and_off_is_the_flag(system):
    spec = SB.SYSTEMS[system]
    assert "gpu_events" in spec.knobs and spec.knobs["gpu_events"].choices == ("on", "off")
    assert all("gpu_events" not in p for p in spec.presets.values())   # nothing sets it
    assert SB.harness_knob_args(spec, {}, ["gpu_events"]) == []
    assert SB.harness_knob_args(spec, {"gpu_events": "on"}, ["gpu_events"]) == []
    assert SB.harness_knob_args(spec, {"gpu_events": "off"}, ["gpu_events"]) == ["--no-gpu-events"]
    with pytest.raises(SB.SystemError_):
        SB.resolve("x", {"system": system, "gpu_events": "maybe"})


def test_gpu_events_off_reaches_the_harness_command(tmp_path):
    e = E.finish(E.load_raw(os.path.join(FIX, "dense.yaml")), os.path.join(FIX, "dense.yaml"))
    for mode, flag in (("on", False), ("off", True)):
        e["systems"]["cpu"] = SB.resolve("cpu", {"system": "cpu", "gpu_events": mode})
        a = L.harness_args(e, e["searches"][0], e["systems"]["cpu"], str(tmp_path), "t")
        assert ("--no-gpu-events" in a) is flag


@pytest.mark.parametrize("backend,modality,cls", [
    ("gpu", "dense", ("faiss_engine", "GpuBackend")), ("cuvs", "dense", ("cuvs_engine", "CuvsBackend")),
    ("cuvs", "sparse", ("cuvs_engine", "SparseCuvsBackend")),
    ("spare", "sparse", ("spare_engine", "SparseSpareBackend")),
    ("pylate", "multivector", ("pylate_engine", "MvPylateBackend"))])
def test_the_gpu_engines_get_events_off_from_the_flag(monkeypatch, backend, modality, cls):
    """The GPU path, mocked: the leg's --no-gpu-events makes every GPU engine run events=False."""
    from bench.systems import harness_cli as H
    mod = getattr(H, cls[0])
    seen = {}

    def fake(*a, **kw):
        seen["events"] = kw.get("events")
        return object()
    monkeypatch.setattr(mod, cls[1], fake)
    monkeypatch.setattr(H.cuvs_engine, "cuvs_pool_spec", lambda args: None)
    for extra, want in (([], True), (["--no-gpu-events"], False)):
        argv = ["--regime", "stream", "--corpus", "/x", "--column", "e", "--query-file", "/q",
                "--query-column", "e", "--k", "10", "--backend", backend, "--csv", "/nonexistent.csv", *extra]
        if modality != "dense":
            argv += ["--modality", modality]
        argv += {"spare": ["--spare-mode", "dot"], "pylate": ["--pylate-mode", "scores-batched"]}.get(backend, [])
        args = H.parse_args(argv)
        H.make_backend(args, "float32")
        assert seen["events"] is want, (backend, extra)


# ---------------------------------------------------------------------------------------------
# the events sweeps (report only)
# ---------------------------------------------------------------------------------------------
EVENT_SWEEPS = sorted(set(glob.glob(os.path.join(EXP, "sweeps", "*_events.yaml")))
                      - {os.path.join(EXP, "sweeps", "sparse_events.yaml")})   # SPARe's own study


def test_every_experiment_has_an_events_sweep():
    # sweeps are local tuning files (not committed); the smoke test is never reported
    if not EVENT_SWEEPS:
        pytest.skip("no local sweep files")
    names = {os.path.basename(p)[:-len("_events.yaml")] for p in EVENT_SWEEPS} - {"s8_pes2o_subset"}
    # the three new experiments still have TODO placeholders and no sweeps yet
    assert names == {E.load_raw(p)["name"] for p in E.experiment_files()} \
        - {"dense_pubmed", "filtered_structured", "filtered_text"}


def test_the_filtered_smoke_test_has_no_sweeps():
    for p in glob.glob(os.path.join(EXP, "sweeps", "*.yaml")):
        assert "filtered_fineweb" not in open(p).read(), p


@pytest.mark.parametrize("path", EVENT_SWEEPS + [os.path.join(FIX, "sweep_events.yaml")])
def test_an_events_sweep_covers_every_harness_system_alternating(path):
    """Every harness system the base schedules, in the base's order, swept over gpu_events
    alone; the on / off order alternates per system; each runs in the base's searches for it."""
    import yaml
    raw = yaml.safe_load(open(path))
    base = E.load_raw(os.path.normpath(os.path.join(os.path.dirname(path), raw["base"])))
    searches = base.get("searches") or [{"name": None, "systems": list(base["systems"])}]
    used = {n for s in searches for n in s["systems"]}
    want = [n for n, e in base["systems"].items() if n in used
            and SB.SYSTEMS[SB.resolve(n, e)["system"]].kind == "harness"]
    ents = raw["sweep"]
    assert [x["system"] for x in ents] == want
    for i, x in enumerate(ents):
        assert x["grid"] == {"gpu_events": [["on", "off"], ["off", "on"]][i % 2]}
        inn = [s["name"] for s in searches if x["system"] in s["systems"]]
        assert x.get("searches", inn) == inn
    assert raw["subset"] == {"files": 3}
    e, variants, _ = E.load_sweep(path)
    assert set(R.events_pairs(variants)) == set(want)
    assert e["protocol"]["rounds"] == 1


@pytest.mark.slow
def test_the_events_sweep_on_cpu_reports_na(tmp_path):
    f = os.path.join(FIX, "sweep_events.yaml")
    before = _md5(f)
    p = bench("sweep", f, "--set", "subset.files=2", "--dev", "--results", str(tmp_path))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
    assert _md5(f) == before
    run, = glob.glob(str(tmp_path / "sweeps" / "sweep_events" / "*"))
    rows = json.load(open(os.path.join(run, "events_table.json")))
    assert [(r["search"], r["system"], r["na"], r["worth_a_look"]) for r in rows] == [("dense", "cpu", True, False)]
    assert rows[0]["on_n"] == rows[0]["off_n"] == 1 and rows[0]["spread"] is None
    assert "n/a (no CUDA events on this path)" in p.stdout and "YAML change" not in p.stdout
    legs = {os.path.basename(x): json.load(open(x)) for x in glob.glob(os.path.join(run, "legs", "*__r1.json"))}
    on, off = legs["dense__cpu-gpu_events-on__r1.json"], legs["dense__cpu-gpu_events-off__r1.json"]
    assert "--no-gpu-events" in off["argv"] and "--no-gpu-events" not in on["argv"]
    assert not any("novabf" in k for k in legs)                 # harness engines only
    assert on["n_files"] == off["n_files"] == 2
    m = json.load(open(os.path.join(run, "manifest.json")))
    assert m["protocol"]["rounds"] == 1
    # --pick: report only, no winner and no YAML change
    p = bench("sweep", "--pick", run, "--results", str(tmp_path))
    assert p.returncode == 0 and "report only" in p.stdout and "YAML change" not in p.stdout


def _fake_run(tmp_path, times, modes):
    """A run dir with leg records + CSVs: times[(variant, round)] = seconds."""
    d = tmp_path / "run"
    (d / "legs").mkdir(parents=True)
    for (v, r), sec in times.items():
        tg = f"s__{v}__r{r}"
        json.dump({"ok": True, "timed_seconds": sec}, open(d / "legs" / f"{tg}.json", "w"))
        with open(d / "legs" / f"{tg}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["warmup", "gpu_events"])
            w.writeheader()
            w.writerow({"warmup": 0, "gpu_events": modes[v.rsplit("-", 1)[1]]})
    return str(d)


VARS = [{"variant": "cuvs-gpu_events-on", "base_system": "cuvs", "setting": {"gpu_events": "on"}},
        {"variant": "cuvs-gpu_events-off", "base_system": "cuvs", "setting": {"gpu_events": "off"}}]


@pytest.mark.parametrize("times,rounds,look,spread", [
    # one run per side: no spread, never marked (1 % or 0.3 %): "not decidable"
    ({("cuvs-gpu_events-on", 1): 1.010, ("cuvs-gpu_events-off", 1): 1.000}, 1, False, None),
    ({("cuvs-gpu_events-on", 1): 1.003, ("cuvs-gpu_events-off", 1): 1.000}, 1, False, None),
    # 2 % with a 0.5 % spread: worth a look
    ({("cuvs-gpu_events-on", 1): 1.020, ("cuvs-gpu_events-on", 2): 1.025,
      ("cuvs-gpu_events-off", 1): 1.000, ("cuvs-gpu_events-off", 2): 1.000}, 2, True, 0.005 / 1.0225),
    # 1 % but the rounds spread 3 %: within noise
    ({("cuvs-gpu_events-on", 1): 1.00, ("cuvs-gpu_events-on", 2): 1.03, ("cuvs-gpu_events-on", 3): 1.01,
      ("cuvs-gpu_events-off", 1): 1.00, ("cuvs-gpu_events-off", 2): 1.00, ("cuvs-gpu_events-off", 3): 1.00},
     3, False, 0.03 / 1.01)])
def test_events_table_rule_on_a_mocked_gpu_run(tmp_path, times, rounds, look, spread):
    d = _fake_run(tmp_path, times, {"on": "on", "off": "off"})
    exp = {"searches": [{"name": "s", "systems": [v["variant"] for v in VARS]}],
           "protocol": {"rounds": rounds}}
    rows = R.events_table(d, exp, VARS)
    r = rows[0]
    assert r["worth_a_look"] is look and not r["na"]
    assert (r["spread"] is None) if spread is None else abs(r["spread"] - spread) < 1e-9
    txt = R.format_events(rows, rounds)
    assert ("| worth a look |" in txt.splitlines()[3]) is look
    if spread is None:
        assert "|  |" in txt.splitlines()[3]                    # the spread column is blank
        assert "| no spread: not decidable |" in txt.splitlines()[3] and not r["decidable"]


# ---------------------------------------------------------------------------------------------
# the io sweeps
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# sweeps: one warm run per variant by default
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(EXP, "sweeps", "*.yaml"))))
def test_paper_sweeps_default_to_one_round(path):
    e, _, _ = E.load_sweep(path)
    assert e["protocol"]["rounds"] == 1
    base = E.load(e["_base"][-1])
    assert base["protocol"]["rounds"] == (1 if base["name"] == "s8_pes2o_subset" else 3)


def test_a_sweep_file_that_sets_rounds_keeps_them(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text(f"name: s\nbase: {os.path.join(FIX, 'sparse.yaml')}\nsweep:\n  - system: cpu\n"
                 "    grid: {sparse_project: ['off', query-vocab]}\nsubset: {files: 2}\nprotocol: {rounds: 2}\n")
    e, _, _ = E.load_sweep(str(f))
    assert e["protocol"]["rounds"] == 2
