"""Leg files and leg records written by hand (harness csv / trace / clock, summary records,
a sweep dir)."""

import csv
import json
import os

import pytest

from bench import leg as L


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(CB, "tests", "fixtures", "experiments")


TRACE_COLS = ["warmup", "file_index", "w_start_s", "w_end_s", "t_done_s", "w_publish_end_s",
              "w_filter_start_s", "w_filter_end_s"]
CSV_COLS = ["warmup", "warmup_files", "n_files", "total_s", "bytes", "verified", "backend",
            "gpu_events", "lib_s", "search_s", "engine_call_s"]


def _harness_leg_files(tmp_path, total=1.0, clock=1.0, footer=0.25, trace=None, with_footer=True):
    legs = tmp_path / "legs"
    legs.mkdir(exist_ok=True)
    with open(legs / "x.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerow(dict(warmup=1, warmup_files=1, n_files=1, total_s=0.1, bytes=1, verified="not_run",
                        backend="gpu", gpu_events="on", lib_s=0.05, search_s=0.06, engine_call_s=0.07))
        w.writerow(dict(warmup=0, warmup_files=1, n_files=2, total_s=total, bytes=10, verified="not_run",
                        backend="gpu", gpu_events="on", lib_s=0.8, search_s=0.85, engine_call_s=0.9))
    rows = trace or [dict(warmup=0, file_index=0, w_start_s=0.01, w_end_s=0.3, t_done_s=0.5),
                     dict(warmup=0, file_index=1, w_start_s=0.2, w_end_s=0.6, t_done_s=0.95)]
    with open(legs / "x.trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLS)
        w.writeheader()
        w.writerow(dict(warmup=1, file_index=0, w_start_s=-5.0, w_end_s=99.0, t_done_s=99.0))  # warmup: ignored
        w.writerows(rows)
    rec = {"clocks": [{"state": "stopped", "events": ["start", "stop"], "seconds": clock}],
           "import_origins": {"ok": True, "problems": []}}
    if with_footer:
        rec["footer_read_s"] = footer
    json.dump(rec, open(legs / "x.clock.json", "w"))
    (legs / "x.topk.npz").write_bytes(b"x")
    return L.LegSpec(outputs={"csv": str(legs / "x.csv"), "clock": str(legs / "x.clock.json"),
                              "topk": str(legs / "x.topk.npz")}, e2e=False)


def _leg_rec(d, search, system, rnd, secs, ok=True, kind="harness", n_files=4, **kw):
    rec = {"tag": f"{search}__{system}__r{rnd}", "search": search, "system": system, "round": rnd,
           "e2e": False, "kind": kind, "ok": ok, "failures": [] if ok else ["boom"],
           "timed_seconds": secs, "time_source": "LegClock", "n_files": n_files,
           "peak_rss_bytes": 1, "bytes_read": 0, **kw}
    json.dump(rec, open(os.path.join(d, "legs", f"{rec['tag']}.json"), "w"))


def _sum_exp(rounds=3, systems=("a", "b"), e2e=None):
    return {"protocol": {"rounds": rounds}, "searches": [{"name": "s", "systems": list(systems)}],
            "systems": {x: {"preset": "stock", "knobs": {}} for x in systems}, "e2e": e2e}


def _sweep_dir(tmp_path, bad=None):
    res = tmp_path / "results"
    d = res / "sweeps" / "sw" / "20990101T000000Z_abc"
    os.makedirs(d / "legs")
    exp = {"protocol": {"rounds": 1}, "searches": [{"name": "s", "systems": ["c-x-1", "c-x-2"]}],
           "systems": {v: {"system": "cpu", "preset": "stock", "knobs": {"sparse_project": v[-1]}}
                       for v in ("c-x-1", "c-x-2")}, "e2e": None}
    json.dump(exp, open(d / "experiment.json", "w"))
    variants = [{"variant": "c-x-1", "base_system": "c", "setting": {"x": 1}},
                {"variant": "c-x-2", "base_system": "c", "setting": {"x": 2}}]
    json.dump({"base": [os.path.join(FIX, "sparse.yaml")], "variants": variants}, open(d / "sweep.json", "w"))
    _leg_rec(str(d), "s", "c-x-1", 1, 1.0)
    _leg_rec(str(d), "s", "c-x-2", 1, 2.0)
    why = "3 queries kept a row clearly worse than one it dropped (beyond the float32 bound)"
    systems = {v: {"ok": v != bad, "fails": [why] if v == bad else []} for v in ("c-x-1", "c-x-2")}
    json.dump({"ok": bad is None, "problems": [f"s {bad}: {why}"] if bad else [],
               "searches": {"s": {"check": "disputed", "systems": systems}}},
              open(d / "checks.json", "w"))
    return str(res), str(d), exp, variants


@pytest.fixture
def rd(tmp_path):
    os.makedirs(tmp_path / "legs")
    json.dump({"ok": True}, open(tmp_path / "checks.json", "w"))
    return str(tmp_path)
