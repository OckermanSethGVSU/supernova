"""bench/runner.py: --set, sweeps (table, pick, variant checks and blame), events sweeps,
the manifest and the data identity."""

import json
import os
import subprocess
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench import experiment as E
from bench import runner

from helpers import leg_records as S1
from helpers.leg_records import _sum_exp, _sweep_dir
from helpers.leg_records import rd  # noqa: F401 (fixture)


CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


def test_a_bench_leg_that_exits_non_zero_is_a_problem(tmp_path):
    """run_legs reports every `bench leg` with a non-zero exit."""
    exp = _sum_exp(rounds=1)
    exp["e2e"] = {"system": "a"}
    rcs = iter([0, 1, 0])
    probs = runner.run_legs(exp, str(tmp_path), False, call=lambda *a, **k: next(rcs))
    assert probs == ["bench leg s a r1: exit 1"]


@pytest.mark.slow
@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_rounds_must_be_a_positive_integer(tmp_path, value):
    """--rounds 0 is refused (not read as 'not given')."""
    p = subprocess.run([sys.executable, "-m", "bench", "run", os.path.join(FIX, "dense.yaml"), "--dev",
                        "--rounds", value, "--results", str(tmp_path)], cwd=CB, capture_output=True, text=True)
    assert p.returncode == 2 and "positive integer" in p.stderr
    assert not os.listdir(tmp_path)


def test_set_creates_missing_mappings_and_expands_a_preset_short_form():
    """--set into a mapping the file does not have."""
    raw = runner._set({"a": 1, "systems": {"x": "stock"}}, ["b.c.d=2", "systems.x.temp_mem=5"])
    assert raw["b"] == {"c": {"d": 2}} and raw["systems"]["x"] == {"preset": "stock", "temp_mem": 5}
    with pytest.raises(E.ExperimentError):
        runner._set({"a": 1}, ["a.b=2"])          # a scalar is still not a mapping


def test_set_protocol_rounds_is_honoured_for_sweeps():
    path = os.path.join(EXP, "sweeps", "sparse_project.yaml")   # its own file sets no rounds

    class A:
        set = ["protocol.rounds=2"]
    exp, _, _ = runner._prepare(path, A, sweep=True)
    assert exp["protocol"]["rounds"] == 2
    A.set = []
    exp, _, _ = runner._prepare(path, A, sweep=True)
    assert exp["protocol"]["rounds"] == E.SWEEP_ROUNDS_DEFAULT


def test_events_sweep_alternates_the_on_off_order_per_engine():
    """engine 1 runs on, off; engine 2 off, on (then the rotation shifts them)."""
    from bench import runner
    exp, variants, _ = E.load_sweep(os.path.join(EXP, "sweeps", "dense_coyo_events.yaml"))
    order = exp["searches"][0]["systems"]
    assert order == ["faiss-gpu_events-on", "faiss-gpu_events-off",
                     "cuvs-gpu_events-off", "cuvs-gpu_events-on"]
    assert "alternates the on / off legs per engine" in runner.format_events([], 1)


def test_sweep_table_has_a_checks_verdict_and_pick_refuses_a_failed_variant(tmp_path, capsys):
    res, d, exp, variants = _sweep_dir(tmp_path, bad="c-x-1")    # the fastest one is wrong
    t = runner.sweep_table(d, exp, variants)
    v = {r["variant"]: r for r in t}
    assert v["c-x-1"]["checks_ok"] is False and "float64" in v["c-x-1"]["checks_why"]
    assert v["c-x-2"]["checks_ok"] is True
    assert "FAILED" in open(os.path.join(d, "sweep_table.md")).read()
    capsys.readouterr()
    assert runner.pick(d, res) == 0
    out = capsys.readouterr().out
    assert "c-x-1: REFUSED, its checks failed" in out and "c: winner c-x-2" in out


def test_pick_chosen_from_is_relative_to_the_results_root_for_a_relative_dir(tmp_path, capsys, monkeypatch):
    res, d, exp, variants = _sweep_dir(tmp_path)
    runner.sweep_table(d, exp, variants)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    assert runner.pick(os.path.relpath(d, tmp_path), "results") == 0
    out = capsys.readouterr().out
    assert "chosen_from: sweeps/sw/20990101T000000Z_abc}" in out, out


def test_local_dirty_status_covers_novabf_and_textscan(monkeypatch):
    """the manifest's dirty check asks git about python/nova-bf (and textscan for
    filtered), not only clean_benchmarks."""
    seen = []
    monkeypatch.setattr(runner, "_git", lambda *a: seen.append(a) or "")
    runner.dirty_paths({"modality": "filtered"})
    spec = seen[-1]
    assert os.path.join(runner.REPO, "python", "nova-bf") in spec
    assert os.path.join(runner.REPO, "crates", "nova-textscan") in spec and runner.CB in spec
    runner.dirty_paths({"modality": "dense"})
    assert os.path.join(runner.REPO, "crates", "nova-textscan") not in seen[-1]


def test_a_multi_search_sweep_table_labels_each_search(tmp_path):
    from bench import runner
    os.makedirs(tmp_path / "legs")
    json.dump({"ok": True, "searches": {}}, open(tmp_path / "checks.json", "w"))
    exp = {"protocol": {"rounds": 1}, "e2e": None,
           "searches": [{"name": "s1", "systems": ["v"]}, {"name": "s2", "systems": ["v"]}],
           "systems": {"v": {"preset": "stock", "knobs": {}}}}
    for s, t in (("s1", 1.5), ("s2", 2.5)):
        json.dump({"tag": f"{s}__v__r1", "search": s, "system": "v", "round": 1, "e2e": False,
                   "kind": "harness", "ok": True, "failures": [], "timed_seconds": t,
                   "time_source": "LegClock", "n_files": 2, "peak_rss_bytes": 1, "bytes_read": 0},
                  open(tmp_path / "legs" / f"{s}__v__r1.json", "w"))
    out = runner.sweep_table(str(tmp_path), exp, [{"variant": "v", "base_system": "v",
                                                   "setting": {"a": "x|y"}}])
    assert out[0]["rounds"] == "s1: median 1.5000, rounds 1.5000; s2: median 2.5000, rounds 2.5000"
    row = open(tmp_path / "sweep_table.md").read().splitlines()[2]
    import re
    assert len(re.split(r"(?<!\\)\|", row)) == 9      # 7 cells between 8 unescaped bars
    assert "x\\|y" in row


def test_the_data_block_hashes_corpus_gt_and_stage_source(tmp_path, monkeypatch):
    import hashlib
    from bench import runner
    (tmp_path / "c").mkdir()
    for i in range(2):
        pq.write_table(pa.table({"x": [i, i + 1]}), tmp_path / "c" / f"f{i}.parquet")
    pq.write_table(pa.table({"x": [7]}), tmp_path / "q.parquet")
    (tmp_path / "gt.npz").write_bytes(b"gt-bytes")
    (tmp_path / "src.parquet").write_bytes(b"source-bytes")
    monkeypatch.setattr(runner, "staged", lambda: None)
    exp = {"corpus": {"path": str(tmp_path / "c")}, "queries": {"path": str(tmp_path / "q.parquet")},
           "checks": {"gt": str(tmp_path / "gt.npz")},
           "prep": {"stage": [{"key": "K", "path": "data/x.parquet", "source": str(tmp_path / "src.parquet"),
                               "how": "copy"}]}}
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    d = runner.data_identity(exp)
    assert [f["sha256"] for f in d["corpus"]["files"]] == [sha(tmp_path / "c" / f"f{i}.parquet") for i in range(2)]
    assert d["corpus"]["hashed"] == "sha256 of every file"
    assert d["gt"]["sha256"] == sha(tmp_path / "gt.npz")
    assert d["prep_sources"] == [{"key": "K", "path": "data/x.parquet", "source": str(tmp_path / "src.parquet"),
                                  "sha256": sha(tmp_path / "src.parquet"), "size": 12, "from": "source file"}]
    monkeypatch.setattr(runner, "LOCAL_CORPUS_SHA_MAX_BYTES", 1)
    d = runner.data_identity(exp)
    assert all("sha256" not in f for f in d["corpus"]["files"])
    assert d["corpus"]["hashed"].startswith("not hashed:") and "size + mtime only" in d["corpus"]["hashed"]
    # a node: the source is not there, STAGED carries its hash (bench launch's stage_data line)
    monkeypatch.setattr(runner, "staged", lambda: {"text": "data data/x.parquet: copy of /gone md5 abc "
                                                   "sha256 " + "e" * 64 + "\n"})
    exp["prep"]["stage"][0]["source"] = "/gone/src.parquet"
    assert runner.data_identity(exp)["prep_sources"][0]["sha256"] == "e" * 64


def _fixture_copy_of_filtered_io(tmp_path):
    """filtered_structured_io.yaml's sweep shape on the CPU filtered fixture: the same entries and grid keys,
    nova-bf on the fixture's config, cuVS / FAISS as the masked CPU reference (FAISS's entries
    keep their static-search restriction), the first two values of each grid, 1 file."""
    import yaml
    f = os.path.join(CB, "tests", "fixtures", "experiments", "sweeps", "filtered_structured_io.yaml")
    real = yaml.safe_load(open(f))
    fx = os.path.join(CB, "tests", "fixtures", "experiments")
    out = {"name": "fio_fixture", "base": os.path.join(fx, "filtered.yaml"),
           "systems": {"novabf": {"system": "novabf", "config": os.path.join(fx, "nb_filtered.yaml")},
                       "cuvs_bitmap": {"system": "masked", "filter_mode": "bitmap"},
                       "faiss_gather": {"system": "masked", "filter_mode": "bitmap"}},
           "sweep": [], "subset": {"files": 2}, "protocol": {"rounds": 1}}
    for ent in real["sweep"]:
        e = {"system": ent["system"],
             "grid": {k: (v[:2] if k.startswith("io.") else [1]) for k, v in ent["grid"].items()}}
        if "searches" in ent:
            e["searches"] = ["static_struct"]
        out["sweep"].append(e)
    p = tmp_path / "fio.yaml"
    p.write_text(yaml.safe_dump(out))
    return str(p)


@pytest.mark.slow
def test_filtered_io_sweep_runs_on_the_cpu_filtered_fixture(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("nova_bf")
    import csv
    import subprocess
    import sys
    p = _fixture_copy_of_filtered_io(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NOVA_BF_", "NOVABF_", "PAPER_"))}
    env.update(HARNESS_NO_IMDS="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    res = tmp_path / "res"
    r = subprocess.run([sys.executable, "-m", "bench", "sweep", p, "--dev", "--results", str(res)],
                       cwd=CB, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = os.path.join(res, "sweeps", "fio")
    d = os.path.join(d, sorted(os.listdir(d))[0])
    rows = list(csv.DictReader(open(os.path.join(d, "sweep_table.csv"))))
    assert len(rows) == 2 + 2 + 2
    assert all(x["complete"] == "True" and x["checks_ok"] == "True" for x in rows), rows
    exp = json.load(open(os.path.join(d, "experiment.json")))
    by = {s["name"]: s["systems"] for s in exp["searches"]}
    assert not any(x.startswith("faiss") for x in by["filtered_text"])


def _vc(tmp_path, rd):
    json.dump({"ok": False, "searches": {"s": {"r1": rd}}, "problems": []},
              open(tmp_path / "checks.json", "w"))
    return runner.variant_checks(str(tmp_path), ["cuvs-b-1", "cuvs-b-2"])


def test_variant_checks_fail_every_listed_system_when_the_oracle_could_not_run(tmp_path):
    out = _vc(tmp_path, {"ok": False, "systems": ["cuvs-b-1", "cuvs-b-2"],
                         "pairs": {"ok": True, "fails": []},
                         "oracle": {"ok": False, "fails": ["oracle could not run: refused"]}})
    assert out["cuvs-b-1"][0] is False and out["cuvs-b-2"][0] is False
    assert "oracle could not run" in out["cuvs-b-1"][1]


def test_variant_checks_fail_every_listed_system_when_pairs_failed(tmp_path):
    ok_sys = {"ok": True, "fails": []}
    out = _vc(tmp_path, {"ok": False, "systems": ["cuvs-b-1", "cuvs-b-2"],
                         "pairs": {"ok": False, "fails": ["compare could not run: x"]},
                         "oracle": {"ok": True, "systems": {"cuvs-b-1": ok_sys, "cuvs-b-2": ok_sys}}})
    assert not out["cuvs-b-1"][0] and not out["cuvs-b-2"][0] and "pairs" in out["cuvs-b-2"][1]
    out = _vc(tmp_path, {"ok": True, "systems": ["cuvs-b-1", "cuvs-b-2"],
                         "pairs": {"ok": True, "fails": []},
                         "oracle": {"ok": True, "systems": {"cuvs-b-1": ok_sys, "cuvs-b-2": ok_sys}}})
    assert out == {"cuvs-b-1": (True, ""), "cuvs-b-2": (True, "")}


def test_pick_refuses_a_variant_named_only_by_a_checks_problem(tmp_path, capsys):
    """every oracle verdict passes; the ONLY failure is a problem string naming c-x-1 (a
    round-to-round comparison that could not run)."""
    res, d, exp, variants = S1._sweep_dir(tmp_path)
    c = json.load(open(os.path.join(d, "checks.json")))
    c.update(ok=False, problems=["rounds s c-x-1: r1 vs r2 could not be compared (boom)"])
    for rd in c["searches"]["s"].values():
        rd["pairs"] = {"ok": True, "fails": []}
    json.dump(c, open(os.path.join(d, "checks.json"), "w"))
    t = {r["variant"]: r for r in runner.sweep_table(d, exp, variants)}
    assert t["c-x-1"]["checks_ok"] is False and "could not be compared" in t["c-x-1"]["checks_why"]
    assert t["c-x-2"]["checks_ok"] is True
    capsys.readouterr()
    runner.pick(d, res)
    out = capsys.readouterr().out
    assert "c-x-1: REFUSED" in out and "c: winner c-x-2" in out


@pytest.mark.parametrize("name", ["sparse_fineweb", "sparse_pubmed", "multivector_pubmed",
                                  "filtered_structured", "dense_coyo"])
def test_an_io_placeholder_points_at_the_io_sweeps(name):
    from bench import runner
    e = E.load(os.path.join(CB, E.path_of(name)))
    hint = runner.io_hint(e, ["io.files_in_flight = 'TODO'"])
    assert "an io sweep over io.*" in hint and "--pick" in hint
    assert runner.io_hint(e, ["queries.nq = 'TODO'"]) == ""


@pytest.mark.slow
def test_a_cpu_io_sweep_runs_and_pick_prints_the_io_change(tmp_path):
    f = tmp_path / "io_cpu.yaml"
    f.write_text(f"name: io_cpu\nbase: {os.path.join(FIX, 'sparse.yaml')}\nsweep:\n"
                 "  - system: cpu\n    grid: {io.files_in_flight: [1, 2]}\nsubset: {files: 2}\n")
    res = tmp_path / "res"
    p = subprocess.run([sys.executable, "-m", "bench", "sweep", str(f), "--dev", "--results", str(res)],
                       cwd=CB, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    (run,) = [os.path.join(res, "sweeps", "io_cpu", x) for x in os.listdir(res / "sweeps" / "io_cpu")]
    for fif in (1, 2):
        rec = json.load(open(os.path.join(run, "legs", f"sparse__cpu-io.files_in_flight-{fif}__r1.json")))
        assert rec["ok"] and rec["argv"][rec["argv"].index("--prefetch") + 1] == str(fif)
    p = subprocess.run([sys.executable, "-m", "bench", "sweep", "--pick", run, "--results", str(res)],
                       cwd=CB, capture_output=True, text=True)
    line = [x for x in p.stdout.splitlines() if x.startswith("    cpu: {")]   # the system's own io
    assert p.returncode == 0 and line and "io: {files_in_flight: " in line[0], p.stdout + p.stderr
    assert not any(x.startswith("  io: ") for x in p.stdout.splitlines()), p.stdout


V3 = ["cuvs-b-1", "cuvs-b-2", "cuvs-b-3"]


def _vc3(tmp_path, pairs, systems=V3, names=V3):
    from bench import runner
    orc = {"ok": True, "systems": {n: {"ok": True, "fails": []} for n in systems}}
    json.dump({"ok": False, "problems": [],
               "searches": {"s": {"r1": {"systems": systems, "pairs": pairs, "oracle": orc}}}},
              open(tmp_path / "checks.json", "w"))
    return runner.variant_checks(str(tmp_path), names)


def _judge_doc(wrong=None, defects=(), order=None):
    return {"system_defects": {n: {"failed": n in defects} for n in V3},
            "f64_wrong_total": wrong or {}, "f64_order_failed_pairs": order or {}}


def test_one_wrong_variant_of_three_fails_only_that_one(tmp_path):
    out = _vc3(tmp_path, {"ok": False, "fails": ["float64 disagreements: cuvs-b-2 4"],
                          "compare": _judge_doc(wrong={"cuvs-b-1": 0, "cuvs-b-2": 4, "cuvs-b-3": 0})})
    assert out["cuvs-b-1"] == (True, "") and out["cuvs-b-3"] == (True, "")
    assert out["cuvs-b-2"][0] is False and "4 float64-checked disagreements" in out["cuvs-b-2"][1]


def test_list_defects_and_order_pairs_are_blamed_on_their_systems(tmp_path):
    out = _vc3(tmp_path, {"ok": False, "fails": ["x"], "compare": _judge_doc(defects=("cuvs-b-3",))})
    assert [out[n][0] for n in V3] == [True, True, False]
    out = _vc3(tmp_path, {"ok": False, "fails": ["x"],
                          "compare": _judge_doc(order={"cuvs-b-1_vs_cuvs-b-2": 3})})
    assert [out[n][0] for n in V3] == [False, False, True]


def test_a_compare_that_could_not_run_fails_every_variant(tmp_path):
    out = _vc3(tmp_path, {"ok": False, "fails": ["compare could not run: boom"], "compare": None})
    assert all(not out[n][0] and "could not run" in out[n][1] for n in V3)


def test_a_variant_with_no_record_in_any_round_was_never_checked(tmp_path):
    # cuvs-b-3 is in no round's systems list and has no oracle verdict anywhere
    out = _vc3(tmp_path, {"ok": True, "fails": [], "compare": _judge_doc()},
               systems=V3[:2], names=V3)
    assert out["cuvs-b-1"] == (True, "") and out["cuvs-b-2"] == (True, "")
    assert out["cuvs-b-3"][0] is False and "never checked" in out["cuvs-b-3"][1]


@pytest.mark.parametrize("args,want", [(["--set", "protocol.rounds=3"], 3), ([], 1),
                                       (["--set", "protocol.rounds=3", "--rounds", "2"], 2)])
def test_the_events_sweep_takes_the_users_protocol_rounds(tmp_path, monkeypatch, args, want):
    from bench import runner
    got = {}

    def fake_execute(exp, run_dir, *a, **k):
        got["rounds"] = exp["protocol"]["rounds"]
        return 0
    monkeypatch.setattr(runner, "execute", fake_execute)
    monkeypatch.setattr(runner, "sweep_table", lambda *a: [{}])
    monkeypatch.setattr(runner, "events_table", lambda *a: got.setdefault("events", []))
    open(tmp_path / "sweep_table.md", "w").close()
    monkeypatch.setattr(runner, "new_run_dir", lambda root: str(tmp_path))
    rc = runner.cmd_sweep([os.path.join(CB, "tests", "fixtures", "experiments", "sweep_events.yaml"), "--dev",
                           "--set", "subset.files=2", "--results", str(tmp_path), *args])
    assert rc == 0 and got["rounds"] == want and got["events"] == []
    assert "REPORT ONLY" in open(tmp_path / "events_table.md").read()
    sw = json.load(open(tmp_path / "sweep.json"))
    assert sw["subset"] == {"files": 2}


def test_the_events_table_promises_no_one_run_marker():
    from bench import runner
    txt = runner.format_events([], 1)
    assert "with one run: above 0.5 %" not in txt
    assert "no spread: not decidable" in txt and "--rounds 3" in txt


def test_run_dirs_on_a_stage_use_the_staged_commit(tmp_path, monkeypatch):
    from bench import runner
    monkeypatch.setattr(runner, "_git", lambda *a: "")
    monkeypatch.setattr(runner, "CB", str(tmp_path))
    (tmp_path / "STAGED").write_text("staged 20261120T000000Z commit=0123456789abcdef (HEAD) from git\n"
                                     "code sha256: ff\ndirty: no\n")
    d = runner.new_run_dir(str(tmp_path / "res"))
    assert os.path.basename(d).endswith("_0123456") and runner.short_sha() == "0123456"
    (tmp_path / "STAGED").unlink()
    assert runner.short_sha() == "nogit"


def test_the_manifest_records_the_corpus_listing_and_the_queries_sha256(tmp_path):
    import hashlib
    from bench import experiment as E, runner
    exp = E.finish(E.load_raw(os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml")),
                   os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml"))
    d = runner.data_identity(exp)
    cdir = os.path.join(CB, "tests", "fixtures", "novabf", "dense", "corpus")
    want = sorted(f for f in os.listdir(cdir) if f.endswith(".parquet"))
    assert [os.path.basename(f["key"]) for f in d["corpus"]["files"]] == want
    for f in d["corpus"]["files"]:
        assert f["size"] == os.path.getsize(f["key"]) and f["mtime"]
    assert d["corpus"]["n_files"] == len(want)
    q = d["queries"]["files"]
    qp = os.path.join(CB, "tests", "fixtures", "novabf", "dense", "queries.parquet")
    assert len(q) == 1 and q[0]["sha256"] == hashlib.sha256(open(qp, "rb").read()).hexdigest()
    assert q[0]["size"] == os.path.getsize(qp)
    # an unreadable corpus is recorded, never a crash
    exp["corpus"] = {**exp["corpus"], "path": str(tmp_path / "missing")}
    assert "error" in runner.data_identity(exp)["corpus"]


def test_the_data_identity_is_gathered_before_the_first_leg(tmp_path, monkeypatch):
    from bench import experiment as E, runner
    exp = E.finish(E.load_raw(os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml")),
                   os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml"))
    order = []
    real = runner.data_identity
    monkeypatch.setattr(runner, "data_identity", lambda e: order.append("data") or real(e))
    monkeypatch.setattr(runner, "run_legs", lambda *a, **k: order.append("legs") or ["stop"])
    monkeypatch.setattr(runner, "finish_manifest", lambda *a: None)
    monkeypatch.setattr(runner.agree, "run_checks", lambda *a: {"ok": False})
    run_dir = runner.new_run_dir(str(tmp_path))
    runner.execute(exp, run_dir, ["bench"], [], True, "skip")
    assert order[:2] == ["data", "legs"]
    m = json.load(open(os.path.join(run_dir, "manifest.json")))
    assert m["data"]["corpus"]["n_files"] > 0 and m["data"]["queries"]["files"][0]["sha256"]


def test_an_s3_listing_is_never_sent_with_placeholder_or_absent_credentials(monkeypatch):
    from bench import runner
    import boto3
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("an S3 request was made"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "<secret>")        # the env fingerprint's fake node
    with pytest.raises(RuntimeError, match="not an AWS key id"):
        runner._s3_listing("s3://b/p/", "us-east-1", False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.setenv("HARNESS_NO_IMDS", "1")
    with pytest.raises(RuntimeError, match="HARNESS_NO_IMDS"):
        runner._s3_listing("s3://b/p/", None, False)
    exp = {"corpus": {"path": "s3://b/p/"}, "queries": {"path": "s3://b/q.parquet"}}
    d = runner.data_identity(exp)
    assert "skipped" in d["corpus"]["error"] and "skipped" in d["queries"]["error"]


LID = "20261120T100000Z_bbbbbbb"


def test_bench_run_and_sweep_record_the_launch_id(tmp_path, monkeypatch):
    from bench import runner
    got = {}
    monkeypatch.setattr(runner, "execute", lambda *a: got.setdefault("run", a[-1]) and 0)
    monkeypatch.setattr(runner, "new_run_dir", lambda p: str(tmp_path))
    fx = os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml")
    runner.cmd_run([fx, "--dev", "--launch-id", LID])
    assert got["run"] == LID
    with pytest.raises(SystemExit):
        runner.cmd_run([fx, "--dev", "--launch-id", "not-an-id"])
    # bench sweep passes it to execute too
    monkeypatch.setattr(runner, "execute", lambda *a: got.setdefault("sweep", a[-1]) and 0)
    monkeypatch.setattr(runner, "sweep_table", lambda *a: [{}])
    monkeypatch.setattr(runner, "events_table", lambda *a: [])
    open(tmp_path / "sweep_table.md", "w").close()
    runner.cmd_sweep([os.path.join(CB, "tests", "fixtures", "experiments", "sweep_events.yaml"), "--dev",
                      "--launch-id", LID, "--results", str(tmp_path)])
    assert got["sweep"] == LID


def test_the_manifest_records_the_launch_id(tmp_path):
    from bench import experiment as E, runner
    fx = os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml")
    exp = E.finish(E.load_raw(fx), fx)
    m = runner.manifest(exp, str(tmp_path), ["bench"], [], True, {}, LID)
    assert m["launch_id"] == LID
    assert runner.manifest(exp, str(tmp_path), ["bench"], [], True, {})["launch_id"] is None


def _checks(d, c):
    json.dump(c, open(os.path.join(d, "checks.json"), "w"))


OKR = {"systems": ["faiss-io.gets-1", "faiss-io.gets-2"],
       "pairs": {"ok": True, "fails": [], "compare": {}},
       "oracle": {"ok": True, "systems": {"faiss-io.gets-1": {"ok": True, "fails": []},
                                          "faiss-io.gets-2": {"ok": True, "fails": []}}}}
V2 = ["faiss-io.gets-1", "faiss-io.gets-2"]


@pytest.mark.parametrize("fails", [["CheckError: B__faiss-io.gets-2__r1: no saved top-k"],
                                   ["ValueError: shapes (3,) (4,)"]])
def test_p_vc_a_crashed_round_fails_every_scheduled_variant(tmp_path, fails):
    from bench import runner
    _checks(str(tmp_path), {"ok": False, "searches": {"A": {"r1": OKR}, "B": {"r1": {"ok": False, "fails": fails}}},
                            "problems": ["B r1: " + fails[0]]})
    out = runner.variant_checks(str(tmp_path), V2)
    for n in V2:
        assert out[n][0] is False and "B r1: the round's checks crashed" in out[n][1], out


def test_a_crashed_round_fails_only_the_variants_scheduled_in_its_search(tmp_path):
    from bench import runner
    json.dump({"searches": [{"name": "A", "systems": V2}, {"name": "B", "systems": ["faiss-io.gets-1"]}]},
              open(tmp_path / "experiment.json", "w"))
    _checks(str(tmp_path), {"ok": False, "searches": {
        "A": {"r1": OKR}, "B": {"r1": {"ok": False, "fails": ["ValueError: x"]}}}, "problems": []})
    out = runner.variant_checks(str(tmp_path), V2)
    assert out["faiss-io.gets-1"][0] is False and out["faiss-io.gets-2"] == (True, "")


def test_a_variant_named_inside_a_leg_tag_is_matched(tmp_path):
    from bench import runner
    _checks(str(tmp_path), {"ok": False, "searches": {"A": {"r1": OKR}},
                            "problems": ["A r1: leg A__faiss-io.gets-2__r1 had no record"]})
    out = runner.variant_checks(str(tmp_path), V2 + ["faiss-io.gets-20"])
    assert out["faiss-io.gets-2"][0] is False and out["faiss-io.gets-1"][0] is True
    assert "faiss-io.gets-2__r1" not in out["faiss-io.gets-20"][1]


def _order_round(names, failed, inv=None):
    pairs = {pk: {"f64_order_inversions_by_system": v} for pk, v in (inv or {}).items()}
    return {"systems": names,
            "pairs": {"ok": False, "fails": ["float64 order disagreements"], "compare": {
                "system_defects": {}, "f64_wrong_total": {}, "f64_order_failed_pairs": failed,
                "pairs": pairs}},
            "oracle": {"ok": True, "systems": {n: {"ok": True, "fails": []} for n in names}}}


def test_p_vc_one_out_of_order_variant_of_three_fails_alone(tmp_path):
    from bench import runner
    n3 = ["v-x-1", "v-x-2", "v-x-3"]
    _checks(str(tmp_path), {"ok": False, "problems": [], "searches": {"A": {"r1": _order_round(
        n3, {"v-x-1_vs_v-x-2": 5, "v-x-1_vs_v-x-3": 5})}}})
    out = runner.variant_checks(str(tmp_path), n3)
    assert out["v-x-1"][0] is False and out["v-x-2"] == (True, "") and out["v-x-3"] == (True, "")


def test_a_list_provably_out_of_order_takes_the_blame():
    from bench import runner
    res = _order_round(["a", "b"], {"a_vs_b": 3}, {"a_vs_b": {"a": 0, "b": 2}})["pairs"]["compare"]
    assert set(runner.order_blame(res, ["a", "b"])) == {"b"}


def test_ambiguous_order_failures_blame_both():
    from bench import runner
    # two systems, no own-list inversion: ambiguous
    res = _order_round(["a", "b"], {"a_vs_b": 3})["pairs"]["compare"]
    assert set(runner.order_blame(res, ["a", "b"])) == {"a", "b"}
    # three systems, every pair failed: no single common system
    res = _order_round(["a", "b", "c"], {"a_vs_b": 1, "a_vs_c": 1, "b_vs_c": 1})["pairs"]["compare"]
    assert set(runner.order_blame(res, ["a", "b", "c"])) == {"a", "b", "c"}
    # three systems, one failed pair, the others passed: both of that pair
    res = _order_round(["a", "b", "c"], {"a_vs_b": 1})["pairs"]["compare"]
    assert set(runner.order_blame(res, ["a", "b", "c"])) == {"a", "b"}


BF_GPU = os.path.join(CB, "tests", "fixtures", "experiments", "novabf", "fineweb_filtered_k1000.yaml")


def _sweep(tmp_path, body):
    p = tmp_path / "s.yaml"
    p.write_text(f"name: s\nbase: {CB}/tests/fixtures/experiments/filtered/structured.yaml\n" + body)
    return str(p)


@pytest.mark.parametrize("body,match", [
    # a nova-bf line whose params.io_workers 16 shadows NOVABF_IO_WORKERS
    ("systems:\n  novabf_iw16: {system: novabf, config: " + BF_GPU + ", params: {io_workers: 16}}\n"
     "sweep:\n  - system: novabf_iw16\n    grid: {io.files_in_flight: [2, 8]}\n",
     "params.io_workers=16"),
    ("systems:\n  nb: {system: novabf, config: " + BF_GPU + ", params: {rust_fetch_concurrency: 64}}\n"
     "sweep:\n  - system: nb\n    grid: {io.gets: [256, 512]}\n", "params.rust_fetch_concurrency=64"),
    ("systems:\n  cv: {system: cuvs, filter_mode: bitmap, prefetch: 4}\n"
     "sweep:\n  - system: cv\n    grid: {io.files_in_flight: [2, 8]}\n", "knob prefetch=4"),
    ("systems:\n  cv: {system: cuvs, filter_mode: bitmap, read_mode: native}\n"
     "sweep:\n  - system: cv\n    grid: {io.read_mode: [native, whole]}\n", "knob read_mode='native'"),
    ("systems:\n  cuvs_stock: {system: cuvs, filter_mode: bitmap, filter_threads: 8}\n"
     "sweep:\n  - system: cuvs_stock\n    grid: {io.files_in_flight: [2, 8], workers: [2, 4]}\n",
     "the grid's own knob 'workers'"),
])
def test_shadowed_io_keys_are_refused(tmp_path, body, match):
    from bench import experiment as E, runner
    a = types.SimpleNamespace(set=["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"])
    with pytest.raises(E.ExperimentError, match="shadowed by"):
        try:
            runner._prepare(_sweep(tmp_path, body), a, sweep=True)
        except E.ExperimentError as exc:
            assert match in str(exc), str(exc)
            raise


def test_an_unshadowed_io_sweep_still_prepares(tmp_path):
    from bench import runner
    a = types.SimpleNamespace(set=["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"])
    body = ("systems:\n  nb: {system: novabf, config: " + BF_GPU + "}\n"
            "  cs: {system: cuvs, filter_mode: bitmap, filter_threads: 8}\n"
            "sweep:\n  - system: nb\n    grid: {io.files_in_flight: [2, 8], io.gets: [256]}\n"
            "  - system: cs\n    grid: {io.files_in_flight: [2, 8]}\n")
    _, variants, _ = runner._prepare(_sweep(tmp_path, body), a, sweep=True)
    assert len(variants) == 4


def test_sweep_blame_for_shared_score_mismatches():
    """A sweep variant is blamed for a shared-id score mismatch only when it is in every failed
    pair (and some pair passed); with two systems, both."""
    from bench import runner
    listed = ["a", "b", "c"]
    res = {"shared_score_failed_pairs": {"a_vs_b": 2, "a_vs_c": 1}}
    out = runner.pairs_blame({"ok": False, "fails": ["x"], "compare": res}, listed)
    assert set(out) == {"a"} and "shared-id score" in out["a"]
    out = runner.pairs_blame({"ok": False, "fails": ["x"],
                              "compare": {"shared_score_failed_pairs": {"a_vs_b": 1}}}, ["a", "b"])
    assert set(out) == {"a", "b"}
