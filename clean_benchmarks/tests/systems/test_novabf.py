"""bench/systems/novabf.py: nova-bf in-process with the 1-file warmup, timed by LegClock.

Each run is a FRESH process (`python -m bench.systems.novabf compute CFG`), as a leg is:
the launcher settings are module globals of nova_bf. Fixtures and configs are in
tests/fixtures/novabf (configs/ beside the data), so these shapes are covered: dense,
dense with slices, several dense searches, sparse, multivector, filtered (all searches
in one pass).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("nova_bf")

from bench.timing import ClockError, LegClock  # noqa: E402
from bench.systems import novabf  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "novabf")
FIX = REF
CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_SEARCHES = {"dense": 1, "dense_slices": 1, "dense_multi": 4, "sparse": 1, "multivector": 1,
              "filtered_all": 4}


def _env(out, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NOVA_BF_", "NOVABF_"))}
    env.update(NOVA_BF_DEVICE="cpu", NOVABF_FIX=FIX, NOVABF_OUT=str(out), PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def _leg(config, out, clock_json, hooks="clock", warm="1", **extra):
    argv = [sys.executable, os.path.join(CB, "bench", "systems", "novabf.py"), "compute", config, "--hooks", hooks,
            "--clock-json", str(clock_json)]
    env = _env(out, **({"NOVABF_WARMUP_FILES": warm} if warm else {}), **extra)
    p = subprocess.run(argv, env=env, cwd=CB, capture_output=True, text=True)
    rec = json.load(open(clock_json)) if os.path.exists(clock_json) else None
    return p, rec


def _shas(out):
    return {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest()
            for f in sorted(os.listdir(out)) if f.endswith(".parquet")}


@pytest.fixture(scope="module")
def legs(tmp_path_factory):
    """Every config once with hooks off, then once with the clock, into the SAME output
    dir (so raw result bytes are comparable: md5 ids hash absolute paths)."""
    base = tmp_path_factory.mktemp("novabf_legs")
    got = {}
    for name in N_SEARCHES:
        out = base / name / "out"
        cfg = os.path.join(REF, "configs", name + ".yaml")
        p_off, rec_off = _leg(cfg, out, base / f"{name}.off.json", hooks="off")
        assert p_off.returncode == 0, p_off.stderr[-3000:]
        sha_off = _shas(out)
        shutil.rmtree(out)
        shutil.rmtree(str(out) + "_warmup")
        p_on, rec_on = _leg(cfg, out, base / f"{name}.on.json", hooks="clock")
        assert p_on.returncode == 0, p_on.stderr[-3000:]
        got[name] = {"off": rec_off, "on": rec_on, "sha_off": sha_off, "sha_on": _shas(out),
                     "log_on": p_on.stderr, "log_off": p_off.stderr}
    return got


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(N_SEARCHES))
def test_clock_sequence_once_per_run_and_warmup_untouched(legs, name):
    rec = legs[name]["on"]
    assert rec["ok"] and rec["warmup_files"] == 1
    c = rec["clock"]
    # exactly one start/stop although the warmup ran run_compute too; a
    # multivector run has one extra pause/resume: its timed token sample
    extra = 1 if name == "multivector" else 0
    assert c["events"] == ["start"] + ["pause", "resume"] * (N_SEARCHES[name] + extra) + ["stop"]
    assert (rec["mv_sample_seconds"] is not None) == bool(extra)
    if extra:
        assert rec["mv_sample_seconds"] == pytest.approx(c["segments"][0])
    assert c["state"] == "stopped" and c["seconds"] > 0
    assert "TIMED_RUN_START warmup_files=1" in legs[name]["log_on"]


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(N_SEARCHES))
def test_hooks_on_and_off_write_identical_results(legs, name):
    assert legs[name]["sha_off"] and legs[name]["sha_on"] == legs[name]["sha_off"]
    assert "clock" not in legs[name]["off"]


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(N_SEARCHES))
def test_clock_agrees_with_novabf_search_seconds(legs, name):
    rec = legs[name]["on"]
    t = rec["novabf_timing"]
    assert t["search_seconds"] == pytest.approx(t["scan_seconds"] + t["result_sort_seconds"])
    agree = rec["clock_agreement"]
    assert agree["ok"], agree
    # nova-bf's intervals are nested in the clock's running time: never negative
    assert rec["clock"]["seconds"] - (rec["mv_sample_seconds"] or 0) >= t["search_seconds"] - 1e-6


@pytest.mark.slow
def test_stock_nova_bf_settings_no_launcher_overrides(legs):
    """no launcher override is applied; the settings lines report nova-bf's own
    defaults (ranged GET x 24, fetch pool 512, two-pass MIN_QUERY_ROWS 8192)."""
    lines = [ln for ln in legs["dense"]["log_on"].splitlines() if ln.startswith("novabf_launch:")]
    assert [ln.split(" ")[1] for ln in lines] == ["stock", "two-pass", "warmup", "warmup", "TIMED_RUN_START"]
    assert "x 24 per file (min file 256 MiB), fetch pool 512 (default)" in lines[0]
    assert lines[1] == "novabf_launch: two-pass MIN_QUERY_ROWS = 8192 (stock)"


@pytest.mark.slow
def test_launcher_override_vars_are_ignored(tmp_path):
    """The old NOVABF_RANGED_CONC / NOVABF_TWOPASS_MIN_ROWS change nothing; they are named."""
    out = tmp_path / "out"
    p, rec = _leg(os.path.join(REF, "configs", "dense.yaml"), out, tmp_path / "c.json",
                  NOVABF_RANGED_CONC="64", NOVABF_TWOPASS_MIN_ROWS="0")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "x 24 per file" in p.stderr and "MIN_QUERY_ROWS = 8192 (stock)" in p.stderr
    assert "NOVABF_RANGED_CONC=64 IGNORED" in p.stderr and "NOVABF_TWOPASS_MIN_ROWS=0 IGNORED" in p.stderr


@pytest.mark.slow
def test_crashed_leg_is_flagged_with_no_time(tmp_path):
    """A corpus file that fails to decode mid-scan: non-zero exit, the clock FAILED with
    the reason, and no seconds anywhere in the record."""
    corpus = tmp_path / "corpus"
    shutil.copytree(os.path.join(FIX, "dense", "corpus"), corpus)
    files = sorted(os.listdir(corpus))
    with open(corpus / files[1], "r+b") as f:   # keep the warmup's file 0 intact
        f.truncate(64)
    src = open(os.path.join(REF, "configs", "dense.yaml")).read()
    cfg = tmp_path / "dense.yaml"
    cfg.write_text(src.replace("${NOVABF_FIX}/dense/corpus", str(corpus)))
    p, rec = _leg(str(cfg), tmp_path / "out", tmp_path / "clock.json")
    assert p.returncode != 0
    assert rec["ok"] is False and "error" in rec
    c = rec["clock"]
    assert c["state"] == "failed" and "run_compute raised" in c["failure"]
    assert c["events"] == ["start"] and "seconds" not in c and "clock_agreement" not in rec


def test_run_without_done_fails_the_clock(monkeypatch, tmp_path):
    """A run_compute that returns without calling done() (a misplaced hook) is caught."""
    import nova_bf.cli
    import nova_bf.compute

    monkeypatch.setattr(nova_bf.cli, "_setup_logging", lambda: None)

    monkeypatch.delenv("NOVABF_WARMUP_FILES", raising=False)
    monkeypatch.setenv("NOVABF_FIX", FIX)
    monkeypatch.setenv("NOVABF_OUT", str(tmp_path / "out"))

    def no_done(cfg, timing_hooks=None, **kw):
        timing_hooks.corpus_start()
        return {}

    monkeypatch.setattr(nova_bf.compute, "run_compute", no_done)
    clock = LegClock()
    with pytest.raises(ClockError, match="no done"):
        novabf.run(os.path.join(REF, "configs", "dense.yaml"), clock, err=open(os.devnull, "w"))
    assert clock.state == "failed"


def test_agreement_window():
    t = {"search_seconds": 100.0}
    assert novabf.clock_agreement(100.0, t)["ok"]
    assert novabf.clock_agreement(100.9, t)["ok"]          # 0.5 + 0.5 % of 100
    assert not novabf.clock_agreement(101.1, t)["ok"]
    assert not novabf.clock_agreement(99.99, t)["ok"]      # below search_seconds
    assert not novabf.clock_agreement(1.0, None)["ok"]


def test_the_novabf_log_tap_keeps_the_unrounded_bf_bench_args():
    import logging
    from bench.systems.novabf import _SearchSecondsTap
    tap = _SearchSecondsTap()
    lg = logging.getLogger("s7_tap_test")
    lg.setLevel(logging.INFO)
    lg.addHandler(tap)
    try:
        lg.info("bf-bench io_workers=%d cpu_threads=%d files=%d rows=%d gb=%.3f wall_s=%.1f "
                "wall_mbps=%.1f stream_mbps=%.1f io_wait_s=%.1f gpu_s=%.1f filter_s=%.1f "
                "read_wall_s=%.1f filter_wall_s=%.1f",
                4, 8, 3, 100, 0.123456, 1.23456, 9.0, 9.0, 0.123456, 1.04321, 0.0, 0.5, 0.0)
        lg.info("bf-bench-search search_s=%.2f scan_s=%.2f result_sort_s=%.3f", 1.23456, 1.2, 0.03456)
    finally:
        lg.removeHandler(tap)
    assert tap.bench["io_wait_s"] == 0.123456 and tap.bench["gpu_s"] == 1.04321   # not 0.1 / 1.0
    assert tap.values["search_seconds"] == 1.23456


def test_stage_dirs_importable_sees_the_cwd_when_it_is_on_sys_path():
    """The control for the by-path leg-record check (tests/test_s4_run.py test_leg_records:
    `stage_dirs_importable == []` in every nova-bf leg): from the stage root with the cwd on
    sys.path (`-c`, as `-m` did), tests/ and env/ ARE importable, and the probe says so."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    probe = "import json; from bench.systems import novabf as n; print(json.dumps(n.stage_dirs_importable()))"
    m = subprocess.run([sys.executable, "-c", probe], cwd=CB, env=env, capture_output=True, text=True)
    got = json.loads(m.stdout)
    assert "tests" in got and "env" in got, m.stdout + m.stderr


def _guard_case(mv, sample):
    from bench.timing import LegClock
    from bench.systems.novabf import BoundaryGuard
    c = LegClock()
    g = BoundaryGuard(c, "/corpus", sample_pauses=sample)
    w = g._write("write", lambda store, *a, **k: None)
    c.start()
    if mv:            # the multivector token sample's own pause / resume
        c.pause()
        c.resume()
    w(None, "/out/search0.parquet")   # a write while timed, before any SEARCH pause
    if mv:            # the search's own pause accounts for one write
        c.pause()
        w(None, "/out/search0.parquet")
        c.resume()
    return g.violations


def test_the_multivector_sample_pause_accounts_for_no_write():
    assert _guard_case(False, 0)                      # dense: flagged
    v = _guard_case(True, 1)                          # mv: the sample's pause does not cover it
    assert len(v) == 1 and "0 pause(s) so far" in v[0]


def test_sample_pauses_comes_from_the_config():
    import types
    from bench.systems.novabf import sample_pauses
    cfg = types.SimpleNamespace(searches=[types.SimpleNamespace(vector_type="dense"),
                                          types.SimpleNamespace(vector_type="multivector")])
    assert sample_pauses(cfg) == 1
    cfg.searches.pop()
    assert sample_pauses(cfg) == 0


@pytest.mark.slow
def test_novabf_store_accepts_the_region_query(tmp_path):
    pytest.importorskip("nova_bf")
    import pyarrow.fs as pafs
    from nova_bf.io import Store
    fs, root = pafs.FileSystem.from_uri("s3://bucket/out/RUN/t?region=us-east-1")
    assert root == "bucket/out/RUN/t" and fs.region == "us-east-1"   # no network: region given
    st = Store("s3://bucket/out/RUN/t?region=us-east-1")
    assert st.root == "bucket/out/RUN/t" and st._native_region == "us-east-1"


def test_the_warmup_output_keeps_the_region_query_at_the_end():
    from bench.systems.novabf import warmup_output
    assert warmup_output("s3://b/o/RUN/t?region=us-east-1") == "s3://b/o/RUN/t_warmup?region=us-east-1"
    assert warmup_output("s3://b/o/RUN/t/") == "s3://b/o/RUN/t_warmup"          # a trailing slash
    assert warmup_output("/local/out/t") == "/local/out/t_warmup"


def test_a_corpus_data_read_while_paused_is_flagged_and_a_footer_read_is_not():
    from bench.systems.novabf import BoundaryGuard
    c = LegClock()
    g = BoundaryGuard(c, "/corpus")
    rd = g._read("read_columns", lambda store, *a, **k: None)
    sch = g._read("read_schema", lambda store, *a, **k: None)
    c.start()
    rd(None, "/corpus/f0.parquet")          # running: fine
    c.pause()
    sch(None, "/corpus/f1.parquet")         # a footer / schema read while paused: allowed
    assert g.violations == []
    rd(None, "/corpus/f1.parquet")          # corpus DATA while paused: off the clock
    rd(None, "/queries/q.parquet")          # not the corpus: not counted
    c.resume()
    c.stop()
    assert len(g.violations) == 1 and "while the clock is paused" in g.violations[0]
    assert g.record()["corpus_reads"]["read_columns@paused"] == 1


def test_the_boundary_guard_counts_s3_reads_given_as_bucket_and_key():
    """nova-bf's Store reads an S3 corpus as `bucket/key` (pyarrow's form, no scheme): those
    are corpus reads, not relative local paths. A local corpus still matches absolute paths."""
    from bench.timing import LegClock
    from bench.systems.novabf import BoundaryGuard
    c = LegClock()
    g = BoundaryGuard(c, "s3://bkt/corpus/")
    r = g._read("read_dense_versioned", lambda store, *a, **k: None)
    c.start()
    r(None, "bkt/corpus/part-0.parquet")
    r(None, "bkt/other/part-0.parquet")            # not the corpus
    assert g.record()["corpus_reads"] == {"read_dense_versioned@running": 1}
    g = BoundaryGuard(c, "tests/corpus")           # local and relative: made absolute
    r = g._read("read_dense", lambda store, *a, **k: None)
    r(None, os.path.abspath("tests/corpus/f.parquet"))
    assert g.record()["corpus_reads"] == {"read_dense@running": 1}


def test_result_paths_on_s3_are_recorded_as_uris():
    """nova-bf names an S3 result `bucket/key`; the leg records `s3://bucket/key` so the checks,
    the input hashes and the manifest lookup read it from S3, not as a relative local path."""
    from bench.systems.novabf import result_uri
    assert result_uri("bkt/run/tag/bf_q.parquet", "s3://bkt/run/tag") == "s3://bkt/run/tag/bf_q.parquet"
    assert result_uri("s3://bkt/x.parquet", "s3://bkt/") == "s3://bkt/x.parquet"
    assert result_uri("/res/out/x.parquet", "/res/out") == "/res/out/x.parquet"
