"""bench/leg.py: leg records, the two attribution views, the clock checks and the planned
argv / env of a leg."""

import json
import os
import queue
import threading
import time

import numpy as np
import pytest

from bench import experiment as E
from bench import leg as L
from bench.data.sources import Block, NOT_READY
from bench.systems import base as SB
from bench.systems import streaming as _streaming
from bench.timing import LegClock, now, Stages

from helpers import leg_records as S1
from helpers.attribution_rows import _row, CLOCK, P_ATTR_ROW, ROW
from helpers.experiments import _exp
from helpers.leg_records import _harness_leg_files


CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


def test_harness_attribution_engine_is_lib_s_and_both_views_sum_to_the_clock():
    # two views summing to the LegClock
    internal = {"total_s": 10.0, "lib_s": 9.0, "search_s": 9.4, "ids_s": 0.1, "merge_s": 0.2,
                "source_stall_s": 0.05, "d2h_s": 0.1, "gpu_idle_s": 0.6, "other_s": 0.05,
                "engine_call_s": 9.7, "h2d_exposed_s": 0.0, "engine_gap_s": 0.2}
    a = L.attribution_harness(internal, 10.0, 0.3, backend="gpu", gpu_events="on")
    h, g = a["host_view"]["parts"], a["gpu_view"]["parts"]
    assert a["engine_s"] == g["engine_s"] == 9.0 and h["merge_s"] == 0.2 and h["io_wait_s"] == 0.05
    assert h["engine_call_s"] == 9.7 and h["harness_other_s"] == 0.05 and g["batching_s"] == 0.0
    assert a["host_view"]["additive"] and a["gpu_view"]["additive"] and g["gpu_idle_s"] == 0.6
    assert a["footer_read_s"] == 0.3 and "CUDA events" in a["clocks"]


def test_harness_attribution_without_events_says_the_gpu_view_is_not_measurable():
    a = L.attribution_harness({"total_s": 2.0, "lib_s": None, "search_s": None, "source_stall_s": 1.0,
                               "engine_call_s": 0.9, "merge_s": 0.05, "other_s": 0.05}, 2.0, 0.0,
                              backend="gpu", gpu_events="off")
    assert a["engine_s"] is None and a["gpu_view"]["measurable"] is False
    assert "no device spans" in a["gpu_view"]["why"]
    assert a["measurable"] and a["host_view"]["additive"]     # the host view needs no events


def test_novabf_attribution_is_its_total_only():
    a = L.attribution_novabf(8.4)    # no engine / framework split, no percentages
    assert a["total_s"] == 8.4 and a["host_view"] is None and a["gpu_view"] is None
    assert "engine_pct" not in a


def test_footer_read_is_recorded_beside_the_headline_not_in_it(tmp_path):
    spec = _harness_leg_files(tmp_path, total=1.0, clock=1.0005, footer=0.25)
    fails = []
    out = L.check_harness(spec, {"subset": None}, "", fails)
    assert fails == []
    assert out["timed_seconds"] == 1.0005                     # the LegClock pass only
    assert out["footer_read_s"] == 0.25
    assert "timed_seconds_with_footers" not in out            # never summed
    assert out["internal"]["lib_s"] == 0.8      # no gpu_idle_s in this CSV: no GPU view
    assert out["attribution"]["gpu_view"]["measurable"] is False


def test_a_clock_record_without_the_footer_time_fails_the_leg(tmp_path):
    spec = _harness_leg_files(tmp_path, with_footer=False)
    fails = []
    L.check_harness(spec, {"subset": None}, "", fails)
    assert any("footer_read_s" in f for f in fails)


def test_the_harness_clock_must_agree_with_total_s(tmp_path):
    spec = _harness_leg_files(tmp_path, total=1.0, clock=1.0 + 2 * L.HARNESS_CLOCK_TOL_S)
    fails = []
    out = L.check_harness(spec, {"subset": None}, "", fails)
    assert any("disagrees with the harness total_s" in f for f in fails)
    assert out["clock_agreement"]["ok"] is False


@pytest.mark.parametrize("row,msg", [
    (dict(warmup=0, file_index=1, w_start_s=-0.001, w_end_s=0.6, t_done_s=0.9), "w_start_s=-0.001 < 0"),
    (dict(warmup=0, file_index=1, w_start_s=0.2, w_end_s=0.6, t_done_s=1.01), "t_done_s=1.01 after"),
    (dict(warmup=0, file_index=1, w_start_s=0.2, w_end_s=1.2, t_done_s=0.9), "w_end_s=1.2 after"),
])
def test_trace_rows_outside_the_timed_pass_fail_the_leg(tmp_path, row, msg):
    ok = dict(warmup=0, file_index=0, w_start_s=0.01, w_end_s=0.3, t_done_s=0.5)
    spec = _harness_leg_files(tmp_path, trace=[ok, row])
    fails = []
    out = L.check_harness(spec, {"subset": None}, "", fails)
    assert any(f.startswith("boundary:") and msg in f for f in fails), fails
    assert out["boundary"]["violations"]


def test_trace_rows_inside_the_pass_pass(tmp_path):
    spec = _harness_leg_files(tmp_path)
    fails = []
    out = L.check_harness(spec, {"subset": None}, "", fails)
    assert fails == [] and out["boundary"]["rows"] == 2 and out["boundary"]["min_start_s"] == 0.01


def test_faiss_temp_mem_is_a_sweepable_knob_and_unset_changes_nothing(tmp_path):
    spec = SB.SYSTEMS["faiss"]
    assert spec.knobs["temp_mem"].flag == "--temp-mem" and spec.knobs["temp_mem"].sweepable
    e0 = _exp("dense_coyo")
    a0 = L.harness_args(e0, e0["searches"][0], e0["systems"]["faiss"], str(tmp_path), "t")
    assert "--temp-mem" not in a0
    e1 = _exp("dense_coyo", ["systems.faiss.temp_mem=536870912"])
    a1 = L.harness_args(e1, e1["searches"][0], e1["systems"]["faiss"], str(tmp_path), "t")
    i = a1.index("--temp-mem")
    assert a1[i + 1] == "536870912" and a1[:i] + a1[i + 2:] == a0


def test_the_novabf_leg_runs_the_adapter_by_path():
    """not `-m` (which puts the cwd, the stage root, first on sys.path)."""
    exp = json.loads(E.dumps(E.finish(E.load_raw(os.path.join(FIX, "dense.yaml")),
                                      os.path.join(FIX, "dense.yaml"))))
    spec = L.plan(exp, "dense", "novabf", 1, "/tmp/x", dev=True)
    assert spec["argv"][1] == os.path.join(CB, "bench", "systems", "novabf.py")
    assert "-m" not in spec["argv"]


def test_a_failed_leg_names_the_engines_last_error():
    from bench import leg as L
    log = ("Traceback (most recent call last):\n  File x\nValueError: --sparse-project query-vocab: "
           "a block is empty after query-vocab projection: it has no nonzeros left\n"
           "\tCommand being timed: \"python ...\"\n\tExit status: 1\n")
    assert L.last_error_line(log).startswith("ValueError: --sparse-project query-vocab: a block is "
                                             "empty after query-vocab projection")
    assert L.last_error_line("all fine\n") is None


def test_process_wall_is_parsed_from_usr_bin_time():
    from bench import leg as L
    assert L.process_wall_s("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n") == 62.5
    assert L.process_wall_s("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:00:01\n") == 3601.0
    assert L.process_wall_s("no time here") is None


@pytest.mark.parametrize("var, val, refused", [
    ("TRITON_CACHE_DIR", "/tmp/elsewhere", True),
    ("TRITON_CACHE_DIR", "~/.triton/cache", False),
    ("CUDA_CACHE_PATH", "/scratch/nv", True),
    ("CUDA_CACHE_PATH", "~/.nv/ComputeCache", False),
    ("TRITON_HOME", "/opt/th", True),
    ("TRITON_HOME", "~", False),
    ("OTHER", "/x", False),
])
def test_the_cold_e2e_leg_refuses_a_moved_cache(tmp_path, monkeypatch, var, val, refused):
    from bench import leg as L
    from bench import timing as P
    spawned, cleared = [], []
    monkeypatch.setattr(L, "spawn", lambda spec, env=None: spawned.append(1) or {
        "exit_code": 0, "wrapper": [], "started_unix": 0, "ended_unix": 0, "children_maxrss_bytes": 1})
    monkeypatch.setattr(P, "clear_e2e_caches", lambda: cleared.append(1) or [])
    monkeypatch.setattr(L, "check_novabf", lambda spec, exp, txt, failures: {})
    monkeypatch.setattr(L, "gpu_foreign_procs", lambda spec, dev: ([], None))
    os.makedirs(tmp_path / "legs")
    spec = {"tag": "e2e__nb", "search": None, "system": "nb", "round": None, "e2e": True,
            "kind": "novabf", "system_kind": "novabf", "preset": "stock", "knobs": {},
            "chosen_from": None, "argv": ["x"], "env_add": {}, "cwd": str(tmp_path),
            "log": str(tmp_path / "legs" / "e2e__nb.log"),
            "outputs": {"topk": str(tmp_path / "legs" / "e2e__nb.topk.npz")}}
    env = {k: v for k, v in os.environ.items() if k not in P.E2E_CACHE_VARS}
    env[var] = val
    rec = L.run_leg({"name": "x", "e2e": {"system": "nb", "keep_cache": False}}, None, "nb", None,
                    str(tmp_path), e2e=True, base_env=env, spec=spec)
    if refused:
        assert not rec["ok"] and not spawned and not cleared
        assert rec["failures"][0].startswith(f"e2e refused: {var}={val}")
    else:
        assert spawned and cleared and not any("refused" in f for f in rec["failures"])


def _gpu_spec(tmp_path, system_kind="cuvs"):
    os.makedirs(tmp_path / "legs", exist_ok=True)
    return {"tag": "s__x__r1", "search": "s", "system": "x", "round": 1, "e2e": False,
            "kind": "harness", "system_kind": system_kind, "preset": "stock", "knobs": {},
            "chosen_from": None, "argv": ["x"], "env_add": {}, "cwd": str(tmp_path),
            "log": str(tmp_path / "legs" / "s__x__r1.log"),
            "outputs": {"topk": str(tmp_path / "legs" / "s__x__r1.topk.npz")}}


def _fake_smi(tmp_path, monkeypatch, out=None, rc=0):
    """PATH = one dir holding a fake nvidia-smi that prints `out` (None: no nvidia-smi at all)."""
    d = tmp_path / "bin"
    d.mkdir()
    if out is not None:
        p = d / "nvidia-smi"
        p.write_text("#!/bin/sh\n"
                     f"[ \"$1 $2\" = \"{' '.join(L.GPU_APPS_QUERY)}\" ] || exit 9\n"
                     f"printf '%s' '{out}'\nexit {rc}\n")
        p.chmod(0o755)
    monkeypatch.setenv("PATH", str(d))


def test_gpu_check_no_other_process(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, out="")
    assert L.gpu_foreign_procs(_gpu_spec(tmp_path), dev=False) == ([], None)


def test_gpu_check_one_other_process_refuses(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, out="4242, /usr/bin/python3\n")
    procs, why = L.gpu_foreign_procs(_gpu_spec(tmp_path), dev=False)
    assert procs == [{"pid": 4242, "name": "/usr/bin/python3"}]
    assert why == "refused: the GPU is in use by another process: pid 4242 (/usr/bin/python3)"


def test_gpu_check_without_nvidia_smi(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, out=None)
    assert L.gpu_foreign_procs(_gpu_spec(tmp_path), dev=False) == ("not checked", None)
    # --dev and the CPU backends never check (and so never need nvidia-smi)
    assert L.gpu_foreign_procs(_gpu_spec(tmp_path), dev=True) == ("n/a", None)
    assert L.gpu_foreign_procs(_gpu_spec(tmp_path, "masked"), dev=False) == ("n/a", None)


def test_gpu_check_failing_nvidia_smi_refuses(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, out="NVML: driver mismatch", rc=3)
    v, why = L.gpu_foreign_procs(_gpu_spec(tmp_path), dev=False)
    assert v == "check failed" and "exited 3" in why and "driver mismatch" in why


@pytest.mark.parametrize("out, runs, recorded", [
    ("", True, []),
    ("17, cuvs_leftover\n", False, [{"pid": 17, "name": "cuvs_leftover"}]),
    (None, True, "not checked"),
])
def test_run_leg_checks_the_gpu_before_the_leg_process_starts(tmp_path, monkeypatch, out, runs,
                                                              recorded):
    spawned = []
    monkeypatch.setattr(L, "spawn", lambda spec, env=None: spawned.append(1) or {
        "exit_code": 0, "wrapper": [], "started_unix": 0, "ended_unix": 0, "children_maxrss_bytes": 1})
    monkeypatch.setattr(L, "check_harness", lambda spec, exp, txt, failures: {"timed_seconds": 1.0})
    _fake_smi(tmp_path, monkeypatch, out=out)
    rec = L.run_leg({"name": "x"}, "s", "x", 1, str(tmp_path), spec=_gpu_spec(tmp_path))
    assert rec["gpu_foreign_procs"] == recorded
    assert bool(spawned) == runs and rec["ok"] == runs
    if not runs:
        assert rec["timed_seconds"] is None and rec["exit_code"] is None
        assert rec["failures"] == ["refused: the GPU is in use by another process: pid 17 (cuvs_leftover)"]
    assert json.load(open(tmp_path / "legs" / "s__x__r1.json"))["gpu_foreign_procs"] == recorded


def test_harness_views_have_one_column_each_and_sum_to_the_legclock_time():
    # p_attr.py's shape: 100 s LegClock, 60 s reader wait, 38 s library, 1.5 s merge
    a = L.attribution_harness(_row(), 100.0, 0.3, "cuvs", "on")
    h, g = a["host_view"]["parts"], a["gpu_view"]["parts"]
    assert list(h) == list(L.HOST_PARTS) and list(g) == list(L.GPU_PARTS)
    assert h == {"io_wait_s": 60.0, "engine_call_s": 38.5, "merge_s": 1.5, "harness_other_s": 0.0}
    assert g["engine_s"] == 38.0 and g["batching_s"] == 0.0 and g["ids_s"] == 0.05
    assert g["device_other_s"] == pytest.approx(0.15) and g["gpu_idle_s"] == 62.0
    assert a["host_view"]["sum_s"] == pytest.approx(100.0)
    assert a["gpu_view"]["sum_s"] == pytest.approx(100.2)          # this synthetic row's busy
    assert a["busy"]["d2h_s"] == 0.15 and a["busy"]["h2d_exposed_s"] == 0.1   # information
    assert "engine_pct" not in a and "harness_s" not in a        # no engine-vs-harness split


def test_merge_is_its_own_column_not_folded_into_harness_or_engine():
    a = L.attribution_harness(_row(merge_s=7.0), 100.0, 0.0, "faiss", "on")
    assert a["host_view"]["parts"]["merge_s"] == 7.0 and a["gpu_view"]["parts"]["engine_s"] == 38.0
    assert a["busy"]["merge_s"] == 7.0


def test_spare_batched_wrapper_is_its_own_part_inside_lib_s():
    g = _row(lib_s=30.0, search_s=30.1, build_s=2.0, spare_score_s=20.0, spare_topk_s=5.0,
             spare_plan_s=1.5)
    a = L.attribution_harness(g, 100.0, 0.0, "spare", "on",
                              {"spare_mode": "dot", "spare_events": "on", "pylate_mode": ""})
    p = a["gpu_view"]["parts"]
    assert p["batching_s"] == pytest.approx(3.0)       # lib - build - score - topk
    assert p["engine_s"] == pytest.approx(27.0)        # build + score + topk
    assert "SPARe batched" in a["batching"]["how"]
    # a released mode is SPARe's own loop: no wrapper part
    a = L.attribution_harness(g, 100.0, 0.0, "spare", "on",
                              {"spare_mode": "dot-released", "spare_events": "n/a"})
    assert a["gpu_view"]["parts"]["batching_s"] == 0.0 and a["gpu_view"]["parts"]["engine_s"] == 30.0


def test_pylate_scores_batched_wrapper_is_its_own_part_outside_lib_s():
    g = _row(lib_s=20.0, search_s=26.0, ids_s=1.0)
    a = L.attribution_harness(g, 100.0, 0.0, "pylate", "on", {"pylate_mode": "scores-batched"})
    p = a["gpu_view"]["parts"]
    assert p["batching_s"] == pytest.approx(5.0)       # search - lib - ids
    assert p["engine_s"] == 20.0 and p["ids_s"] == 1.0 and p["device_other_s"] == pytest.approx(0.0)


def test_overlap_mode_is_flagged_and_the_host_view_still_sums():
    a = L.attribution_harness(_row(engine_gap_s=None, merge_s=30.0, source_stall_s=30.0, other_s=1.5), 100.0,
                              0.0, "faiss", "on")
    assert a["overlap_mode"] is True and a["host_view"]["additive"]
    assert a["host_view"]["sum_s"] == pytest.approx(100.0)


def test_novabf_has_no_breakdown_and_no_percentages():
    a = L.attribution_novabf(100.0)
    assert a == {**a, "kind": "novabf", "total_s": 100.0, "host_view": None, "gpu_view": None}
    assert not any("pct" in k for k in a) and "engine_s" not in a


def test_the_harness_clock_tolerance_is_pinned_at_10_ms(tmp_path):
    """the tolerance itself (not only relative to itself) is 10 ms."""
    assert L.HARNESS_CLOCK_TOL_S == 0.01
    spec = S1._harness_leg_files(tmp_path, total=1.0, clock=1.02)
    fails = []
    L.check_harness(spec, {"subset": None}, "", fails)
    assert any("disagrees with the harness total_s" in f for f in fails)


def test_a_nan_clock_difference_fails(tmp_path):
    """`not abs(diff) <= TOL`: a NaN clock (or total) is a disagreement."""
    spec = S1._harness_leg_files(tmp_path, total=1.0, clock=float("nan"))
    fails = []
    L.check_harness(spec, {"subset": None}, "", fails)
    assert any("disagrees with the harness total_s" in f for f in fails)


def test_a_harness_pass_covering_other_than_the_subset_fails(tmp_path):
    """the timed row's n_files (2 in the helper) vs the subset."""
    spec = S1._harness_leg_files(tmp_path)
    fails = []
    L.check_harness(spec, {"subset": {"files": 2}}, "", fails)
    assert not any("subset" in f for f in fails)
    L.check_harness(spec, {"subset": {"files": 3}}, "", fails)
    assert any("timed pass covered n_files=2, the subset is 3" in f for f in fails)


HARNESS_OF = {"dense": "cpu", "sparse": "cpu", "multivector": "cpu", "filtered": "masked_bitmap"}


def _io_sweep(tmp_path, shape, extra=""):
    f = tmp_path / f"io_{shape}.yaml"
    f.write_text(f"name: io_{shape}\nbase: {os.path.join(FIX, shape + '.yaml')}\nsweep:\n"
                 f"  - system: novabf\n    grid: {{io.files_in_flight: [1, 3], io.gets: [32]}}\n"
                 f"  - system: {HARNESS_OF[shape]}\n    grid: {{io.files_in_flight: [1, 3], io.part_mib: [8]}}\n"
                 f"subset: {{files: 2}}\n{extra}")
    return str(f)


@pytest.mark.parametrize("shape", ["dense", "sparse", "multivector", "filtered"])
def test_an_io_grid_reaches_every_leg_through_the_leg_runner(tmp_path, shape):
    e, variants, _ = E.load_sweep(_io_sweep(tmp_path, shape))
    assert len(variants) == 4
    s = e["searches"][0]["name"]
    for v in variants:
        sysr = e["systems"][v["variant"]]
        fif = v["setting"]["io.files_in_flight"]
        spec = L.plan(e, s, v["variant"], 1, str(tmp_path / "run"), dev=True)
        if sysr["kind"] == "novabf":
            assert spec["env_add"]["NOVABF_IO_WORKERS"] == str(fif)
            assert spec["env_add"]["NOVABF_RUST_FETCH_CONC"] == "32"
            assert spec["env_add"]["NOVABF_RUST_FETCH_PART_MIB"] == str(e["io"]["part_mib"])
        else:
            a = spec["argv"]
            assert a[a.index("--prefetch") + 1] == a[a.index("--workers") + 1] == str(fif)
            assert a[a.index("--range-mib") + 1] == "8"
            assert a[a.index("--range-concurrency") + 1] == str(e["io"]["gets"])
    assert e["io"]["files_in_flight"] == 2     # the shared io itself is unchanged


@pytest.mark.parametrize("base,region,want", [
    ("s3://b/out", "us-east-1", "s3://b/out/RUN/t?region=us-east-1"),
    ("s3://b/out/", "us-east-1", "s3://b/out/RUN/t?region=us-east-1"),
    ("s3://b/out?region=eu-west-1", "us-east-1", "s3://b/out/RUN/t?region=eu-west-1"),
    ("s3://b/out?scheme=http", "us-east-1", "s3://b/out/RUN/t?scheme=http&region=us-east-1"),
    ("s3://b/out", None, "s3://b/out/RUN/t"),
    ("/local/out", "us-east-1", "/local/out/RUN/t"),
])
def test_novabf_s3_output_gets_the_corpus_region(base, region, want):
    exp = {"novabf_output": base, "corpus": {"region": region}}
    assert L.novabf_output(exp, "/r/RUN", "t") == want


def _novabf_clock(tmp_path, agree_ok):
    p = tmp_path / "x.clock.json"
    json.dump({"ok": True, "clock": {"state": "stopped", "events": ["start", "pause", "resume", "stop"],
                                     "seconds": 2.0},
               "boundary": {"violations": [], "corpus_reads": {"read_columns@running": 4}},
               "clock_agreement": {"ok": agree_ok, "diff_seconds": 0.0 if agree_ok else 0.5},
               "results": {}}, open(p, "w"))
    return L.LegSpec(outputs={"clock": str(p)}, e2e=False)


def test_a_novabf_clock_that_disagrees_with_its_search_seconds_fails(tmp_path):
    """the LegClock vs nova-bf search_seconds cross-check fails the leg."""
    txt = ("novabf_launch: TIMED_RUN_START warmup_files=1\n"
           "bf-bench io_workers=4 files=4 rows=1 gb=0.001 wall_s=1.0 x io_wait_s=0.1 gpu_s=0.5\n"
           "bf-bench-search search_s=1.9 scan_s=1.8 result_sort_s=0.1\n")
    fails = []
    L.check_novabf(_novabf_clock(tmp_path, True), {"subset": None}, txt, fails)
    assert not any("disagrees" in f for f in fails), fails
    fails = []
    out = L.check_novabf(_novabf_clock(tmp_path, False), {"subset": None}, txt, fails)
    assert any("LegClock disagrees with nova-bf's search_seconds" in f for f in fails)
    assert "timed_seconds" not in out


def test_the_e2e_leg_sets_no_warmup(tmp_path):
    """the cold e2e leg has NO NOVABF_WARMUP_FILES (a timed leg has it, last)."""
    from bench import experiment as E
    e = E.finish(E.load_raw(os.path.join(CB, "tests", "fixtures", "experiments", "dense", "coyo.yaml")),
                 os.path.join(CB, "tests", "fixtures", "experiments", "dense", "coyo.yaml"))
    s = e["searches"][0]["name"]
    timed = L.plan(e, s, "novabf", 1, str(tmp_path / "r"))
    cold = L.plan(e, None, "novabf", None, str(tmp_path / "r"), e2e=True)
    assert list(timed["env_add"])[-1] == "NOVABF_WARMUP_FILES"
    assert "NOVABF_WARMUP_FILES" not in cold["env_add"]


class _Mark:
    def __init__(self):
        self.t = None
        self.done = threading.Event()


class FakeStream:
    """One FIFO worker thread = one stream. enqueue(seconds, wait=marks) runs after every earlier
    op of this stream and after `wait` (marks of other streams: a stream wait_event); mark()
    is an event recorded on this stream."""

    def __init__(self):
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            op = self.q.get()
            if op is None:
                return
            kind, arg, m = op
            if kind == "wait":
                for w in arg:
                    w.done.wait()
            elif kind == "sleep":
                time.sleep(arg)
            if m is not None:
                m.t = now()
                m.done.set()

    def run(self, seconds):
        self.q.put(("sleep", seconds, None))

    def wait(self, *marks):
        self.q.put(("wait", marks, None))

    def mark(self):
        m = _Mark()
        self.q.put(("mark", None, m))
        return m

    def close(self):
        self.q.put(None)


def _el(a, b):
    a.done.wait()
    b.done.wait()
    return b.t - a.t


class FakeDevice(_streaming.ResidentBackend):
    """The resident event chain of streaming.ResidentBackend on fake streams (run_pass sees a
    ResidentBackend with events: it computes gpu_idle_s from its busy spans). Durations in s."""

    DUR = {"h2d": 0.004, "widen": 0.001, "normalize": 0.001, "lib": 0.008, "pad": 0.001,
           "ids": 0.001, "d2h": 0.003}

    def __init__(self, k, d2h_overlap=False, h2d_overlap=False):
        self.k, self.events, self.stats = k, True, Stages()
        self.d2h_overlap, self.overlap = d2h_overlap, h2d_overlap
        self.normalize_blocks = True
        self.last_wait, self.last_build, self.last_h2d = (0.0, False), 0.0, (None, None, None)
        self.S, self.C, self.D = FakeStream(), FakeStream(), FakeStream()
        self._ahead = None
        self.xq = None

    def alloc_peak_reset(self):
        pass

    def alloc_peak(self):
        return ""

    def _upload(self, x):
        c0 = self.C.mark()
        self.C.run(self.DUR["h2d"])
        return (x, c0, self.C.mark())

    def submit(self, xb, ahead=None, offset=0, **kw):
        d = self.DUR
        up, issued = None, False
        if self.overlap:
            up, self._ahead = self._ahead, None
            issued = up is not None and up[0] is xb
            if not issued:
                up = self._upload(xb)
            if ahead is not None:
                self._ahead = self._upload(ahead)
        th = now()
        e0 = self.S.mark()
        if up is not None:
            self.S.wait(up[2])            # the compute stream waits for the copy (exposed part)
        else:
            self.S.run(d["h2d"])
        e1 = self.S.mark()
        self.S.run(d["widen"])
        e2 = self.S.mark()
        self.S.run(d["normalize"])
        e2n = self.S.mark()
        la = self.S.mark()
        self.S.run(d["lib"])               # the engine library
        lb = self.S.mark()
        self.S.run(d["pad"])               # our other device work in the search span
        eg = self.S.mark()
        self.S.run(d["ids"])               # the id step
        e3 = self.S.mark()
        if self.d2h_overlap:
            self.D.wait(e3)
            self.D.run(d["d2h"])
            e4 = self.D.mark()
        else:
            self.S.run(d["d2h"])
            e4 = self.S.mark()
        self.stats.add("engine_host", now() - th)
        s = self.xq @ xb.T                 # the real answer (the merge reads it)
        kk = min(self.k, s.shape[1])
        I = np.argsort(-s, axis=1, kind="stable")[:, :kk]
        D = np.take_along_axis(s, I, 1).astype(np.float32)
        I = (I + offset).astype(np.int64)
        return {"e": (e0, e1, e2, e2n, e3, e4), "lib": (la, lb), "eg": eg, "up": up,
                "issued": issued, "D": D, "I": I}

    def complete(self, tok):
        e0, e1, e2, e2n, e3, e4 = tok["e"]
        e4.done.wait()                     # the host waits for this block's copy back only
        up = tok["up"]
        h2d = _el(up[1], up[2]) if up is not None else _el(e0, e1)
        self.stats.add("h2d", h2d)
        self.stats.add("h2d_exposed", _el(e0, e1))
        self.stats.add("h2d_overlapped", int(tok["issued"]))
        self.last_h2d = (h2d, _el(e0, e1), int(tok["issued"]))
        self.stats.add("widen", _el(e1, e2))
        self.stats.add("normalize", _el(e2, e2n))
        self.stats.add("search", _el(e2n, e3))
        self.stats.add("lib", _el(*tok["lib"]))
        self.stats.add("ids", _el(tok["eg"], e3))
        self.stats.add("d2h", _el(e3, e4))
        self.stats.add("sync_wait", 0.0)
        return self._pad(tok["D"], tok["I"])

    def search(self, xb, ahead=None, offset=0, **kw):
        return self.complete(self.submit(xb, ahead, offset))

    def drain(self):
        self._ahead = None

    def close(self):
        for s in (self.S, self.C, self.D):
            s.close()


class _Pollable:
    def __init__(self, q, n):
        self.q, self.n, self.got = q, n, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.got == self.n:
            raise StopIteration
        self.got += 1
        return self.q.get()

    def poll(self):
        if self.got == self.n:
            return None
        try:
            b = self.q.get_nowait()
        except queue.Empty:
            return NOT_READY
        self.got += 1
        return b

    def close(self):
        pass


class PrefetchSource:
    """Blocks read on a reader thread (READ seconds each), at most `depth` ahead."""

    READ = 0.006
    reader = "thread"

    def __init__(self, blocks, depth=2):
        self.blocks, self.depth, self.stats = blocks, depth, Stages()

    def stream(self, n_files=None, pollable=False):
        todo = self.blocks[:n_files] if n_files else self.blocks
        q = queue.Queue(maxsize=self.depth)

        def read():
            for b in todo:
                time.sleep(self.READ)
                self.stats.add("fetch", self.READ)
                q.put(b)
        threading.Thread(target=read, daemon=True).start()
        return _Pollable(q, len(todo))

    def __iter__(self):
        return self.stream()


def _fake_pass(mode):
    from bench.systems import loop
    rng = np.random.default_rng(0)
    nq, d, k, rows = 8, 16, 5, 40
    blocks, off = [], 0
    for i in range(8):
        x = rng.standard_normal((rows, d)).astype(np.float32)
        blocks.append(Block(off, x, f"f{i}"))
        off += rows
    dev = FakeDevice(k, d2h_overlap=mode in ("d2h", "both"), h2d_overlap=mode in ("h2d", "both"))
    dev.xq = rng.standard_normal((nq, d)).astype(np.float32)
    clock = LegClock()
    try:
        D, I, info = loop.run_pass(PrefetchSource(blocks), dev, nq, k, d=d, clock=clock)
    finally:
        dev.close()
    info.pop("trace")
    internal = {c: L._f(v) for c, v in info.items() if c.endswith("_s") or c in L.HARNESS_EVENT_COLS}
    a = L.attribution_harness(internal, clock.seconds, 0.0, "faiss", "on", {"widen_where": "device"})
    # the answer is the exact top-k (the fake device computes it for real)
    xb = np.concatenate([b.vectors for b in blocks])
    ref = np.argsort(-(dev.xq @ xb.T), axis=1, kind="stable")[:, :k]
    assert (np.sort(I, 1) == np.sort(ref, 1)).all()
    return a, info, clock.seconds


@pytest.mark.parametrize("mode", ["serial", "d2h", "h2d", "both"])
def test_both_views_add_up_to_the_legclock_time_on_an_async_device(mode):
    """the host view and the GPU view, from the real run_pass's columns, each sum
    to the LegClock time (residual within ADDITIVE_TOL_S), with every part >= 0 (no part counted
    twice), in serial and in --d2h-overlap / --h2d-overlap modes."""
    a, info, clock_s = _fake_pass(mode)
    hv, gv = a["host_view"], a["gpu_view"]
    assert gv["measurable"] and hv["measurable"]
    for v in (hv, gv):
        assert v["additive"] is True, (mode, v)
        assert abs(v["residual_s"]) <= 1e-3 and v["sum_s"] == pytest.approx(clock_s, abs=1e-3)
        assert all(x is None or x >= -1e-4 for x in v["parts"].values()), (mode, v["parts"])
    gp = gv["parts"]
    D = FakeDevice.DUR
    n = 8   # the fake's own durations come back as the right parts (to scheduling noise)
    assert gp["engine_s"] == pytest.approx(n * D["lib"], rel=0.5)
    assert gp["ids_s"] == pytest.approx(n * D["ids"], rel=0.5)
    assert gp["device_other_s"] == pytest.approx(n * D["pad"], rel=1.0)
    assert gp["gpu_idle_s"] > 0
    if mode in ("d2h", "both"):
        # the overlapped busy time is information only: d2h + merge would exceed the host's
        # own merge-and-wait time if it were stacked; it is never part of either view
        assert info["engine_gap_s"] == "" and a["busy"]["d2h_s"] > 0
        assert "d2h_s" not in hv["parts"] and "d2h_s" not in gp
    if mode in ("h2d", "both"):
        assert int(info["h2d_overlapped"]) > 0
        assert a["busy"]["h2d_s"] >= a["busy"]["h2d_exposed_s"]


def test_host_view_parts_are_exactly_their_columns():
    a = L.attribution_harness(dict(ROW), CLOCK, 0.0, "faiss", "on", {"widen_where": "device"})
    h = a["host_view"]["parts"]
    assert h["io_wait_s"] == 11.0          # source_stall_s
    assert h["engine_call_s"] == 23.0      # engine_call_s (no filter handoff)
    assert h["merge_s"] == 3.0             # merge_s
    assert h["harness_other_s"] == 0.7     # other_s
    assert a["host_view"]["sum_s"] == pytest.approx(CLOCK) and a["host_view"]["additive"]


def test_gpu_view_parts_are_exactly_their_formulas():
    a = L.attribution_harness(dict(ROW), CLOCK, 0.0, "faiss", "on", {"widen_where": "device"})
    g = a["gpu_view"]["parts"]
    assert g["engine_s"] == 17.0                                   # lib_s
    assert g["batching_s"] == 0.0 and g["engine_and_batching_s"] is None
    assert g["ids_s"] == 0.29                                      # ids_s
    assert g["widen_s"] == 0.13                                    # widen_s (device widen)
    assert g["normalize_s"] == 0.17                                # normalize_s
    assert g["device_other_s"] == pytest.approx(19.0 - 17.0 - 0.29)   # search - lib - ids
    assert g["gpu_idle_s"] == ROW["gpu_idle_s"]
    assert a["gpu_view"]["sum_s"] == pytest.approx(CLOCK) and a["gpu_view"]["additive"]
    # a host widen is not on the stream: 0 in the GPU view, so the view does not add up
    b = L.attribution_harness(dict(ROW), CLOCK, 0.0, "faiss", "on", {"widen_where": "host"})
    assert b["gpu_view"]["parts"]["widen_s"] == 0.0 and not b["gpu_view"]["additive"]


def test_busy_times_are_information_not_parts():
    a = L.attribution_harness(dict(ROW), CLOCK, 0.0, "faiss", "on", {"widen_where": "device"})
    assert a["busy"]["h2d_s"] == 0.9 and a["busy"]["h2d_exposed_s"] == 0.3
    assert a["busy"]["d2h_s"] == 0.41 and a["busy"]["merge_s"] == 3.0
    for v in (a["host_view"], a["gpu_view"]):
        assert not {"h2d_s", "d2h_s", "h2d_exposed_s"} & set(v["parts"])


def test_a_view_with_time_moved_out_of_it_is_not_additive():
    # the views are checked against the clock, not made to add up by a remainder (20 ms: beyond
    # the tolerance, shared with the LegClock check)
    a = L.attribution_harness(dict(ROW), CLOCK + 0.02, 0.0, "faiss", "on", {"widen_where": "device"})
    assert a["host_view"]["residual_s"] == pytest.approx(0.02)
    assert not a["host_view"]["additive"] and not a["gpu_view"]["additive"]


def test_filtered_reader_label_and_the_handoff_moves_to_harness_other():
    row = dict(ROW, filter_s=9.0, filter_eval_s=8.5, filter_handoff_s=0.6)
    a = L.attribution_harness(row, CLOCK, 0.0, "faiss", "on", {"widen_where": "device"})
    hv = a["host_view"]
    assert hv["labels"]["io_wait_s"] == "reader (read + filter eval)"
    assert hv["parts"]["engine_call_s"] == pytest.approx(23.0 - 0.6)
    assert hv["parts"]["harness_other_s"] == pytest.approx(0.7 + 0.6)
    assert hv["additive"]
    assert a["filter"] == {"reader_label": "reader (read + filter eval)", "filter_s": 9.0,
                           "filter_eval_s": 8.5, "filter_handoff_s": 0.6}
    from bench.report import summary as SUM
    v = SUM.attr_values(a)
    assert v["filter_s"] == 9.0 and v["filter_eval_s"] == 8.5
    # not filtered: no label change, no filter block
    b = L.attribution_harness(dict(ROW), CLOCK, 0.0, "faiss", "on")
    assert b["filter"] is None and b["host_view"]["labels"]["io_wait_s"] == "io wait (reader)"


def test_additivity_shares_the_legclock_tolerance():
    assert L.ADDITIVE_TOL_S == L.HARNESS_CLOCK_TOL_S
    # p_attr.py: 3 ms LegClock vs total_s passes the clock check, so the view is additive too
    a = L.attribution_harness(dict(P_ATTR_ROW), 1.003, 0.01, "gpu", "off", {})
    assert a["host_view"]["additive"] is True
    b = L.attribution_harness(dict(P_ATTR_ROW), 1.02, 0.01, "gpu", "off", {})
    assert b["host_view"]["additive"] is False


def test_overlap_mode_comes_from_the_recorded_copy_modes():
    assert {"h2d_mode", "d2h_mode"} <= set(L.HARNESS_MODE_COLS)
    row = dict(P_ATTR_ROW, engine_gap_s=None)   # p_attr.py: --d2h-overlap, events off
    m = {"spare_mode": None, "pylate_mode": None, "widen_where": "device"}
    f = lambda **k: L.attribution_harness(dict(row), 1.0, 0.0, "gpu", "off", {**m, **k})["overlap_mode"]
    assert f(d2h_mode="pinned-overlap", h2d_mode="pageable") is True
    assert f(d2h_mode="pageable", h2d_mode="pinned-overlap") is True
    assert f(d2h_mode="pinned", h2d_mode="pinned") is False
    assert f(d2h_mode="n/a", h2d_mode="n/a") is False


def test_leg_check_refuses_a_harness_row_without_n_files(tmp_path):
    from bench import leg as L
    from bench import timing
    legs = tmp_path / "legs"
    legs.mkdir()
    json.dump({"data": {"corpus": {"n_files": 4}}}, open(tmp_path / "manifest.json", "w"))
    cols = ["warmup", "warmup_files", "n_files", "total_s"]

    def run(nf):
        p = legs / "x.csv"
        with open(p, "w") as f:
            f.write(",".join(cols) + "\n")
            f.write(f"1,{timing.WARMUP_FILES},{timing.WARMUP_FILES},0.1\n")
            f.write(f"0,0,{nf},0.1\n")
        spec = {"outputs": {"csv": str(p), "clock": str(legs / "x.clock.json"),
                            "topk": str(legs / "x.topk.npz")}}
        failures = []
        try:
            L.check_harness(spec, {"subset": None}, "", failures)
        except Exception:  # noqa: BLE001  later checks need a full spec; only coverage matters
            pass
        return failures
    assert any("no n_files" in f for f in run(""))
    assert any("corpus listing has 4 files" in f for f in run(3))
    assert not any("n_files" in f for f in run(4))


def test_a_filtered_novabf_leg_reads_the_experiments_s3_corpus_with_its_region(tmp_path):
    """filtered_structured: the generated nova-bf config's corpus is the experiment's S3 prefix
    (placeholder filled in) with ?region=<corpus.region>; nova-bf lists it recursively itself."""
    import yaml
    from bench import paths
    from helpers.experiments import _exp
    e = _exp("filtered_structured", ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"])
    spec = L.plan(e, "structured_per_query", "novabf", 1, str(tmp_path))
    c = yaml.safe_load(open(spec["outputs"]["config"]))
    want = paths.expand(e["corpus"]["path"]) + "?region=us-east-1"
    assert want.startswith("s3://") and c["corpus"]["path"] == want
    assert [s["name"] for s in c["searches"]] == ["structured_per_query"]
    assert "NOVABF_CORPUS" not in spec["env_add"]     # the filter config names no ${NOVABF_CORPUS}
    # a local corpus (realfile) or one already naming its region is left as it is
    assert L.novabf_corpus_url({"corpus": {"path": "data/real", "region": "us-east-1"}}) == "data/real"
    assert L.novabf_corpus_url({"corpus": {"path": "s3://b/p/?region=eu-west-1",
                                           "region": "us-east-1"}}) == "s3://b/p/?region=eu-west-1"
