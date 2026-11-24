"""bench/leg.py: a harness leg's must-be-zero counters."""

import json

import numpy as np
import pytest


def _harness_leg_files(tmp_path, extra):
    import csv
    from bench import leg as L
    d = tmp_path
    cols = ["warmup", "warmup_files", "n_files", "total_s", "verified", "bytes",
            "cuvs_filter_small_k_calls", "filter_mask_overflows"]
    rows = [{"warmup": 1, "warmup_files": 1, "n_files": 1, "total_s": 0.1, "verified": "",
             "bytes": 0, "cuvs_filter_small_k_calls": 0, "filter_mask_overflows": 0},
            {"warmup": 0, "warmup_files": 1, "n_files": 2, "total_s": 1.0, "verified": "",
             "bytes": 0, "cuvs_filter_small_k_calls": 0, "filter_mask_overflows": 0, **extra}]
    with open(d / "t.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(d / "t.trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["warmup", "file_index", "w_start_s", "t_done_s"])
        w.writeheader()
        w.writerow({"warmup": 0, "file_index": 0, "w_start_s": 0.0, "t_done_s": 0.5})
    json.dump({"clocks": [{"state": "stopped", "events": ["start", "stop"], "seconds": 1.0}],
               "footer_read_s": 0.01, "import_origins": {"ok": True}}, open(d / "t.clock.json", "w"))
    np.savez(d / "t.topk.npz", I=np.zeros((1, 1), np.int64), D=np.zeros((1, 1), np.float32))
    spec = {"outputs": {"csv": str(d / "t.csv"), "clock": str(d / "t.clock.json"),
                        "topk": str(d / "t.topk.npz")}}
    failures = []
    L.check_harness(spec, {"subset": None}, "", failures)
    return failures


@pytest.mark.parametrize("col", ["cuvs_filter_small_k_calls", "filter_mask_overflows"])
def test_a_nonzero_must_be_zero_counter_fails_the_leg(tmp_path, col):
    assert _harness_leg_files(tmp_path, {}) == []
    f = _harness_leg_files(tmp_path, {col: 3})
    assert f == [f"{col} = 3 in the timed pass (must be 0: {__import__('bench.leg').leg.MUST_BE_ZERO[col]})"]


def test_a_blank_must_be_zero_counter_is_not_applicable(tmp_path):
    assert _harness_leg_files(tmp_path, {"cuvs_filter_small_k_calls": ""}) == []
