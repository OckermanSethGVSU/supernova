"""bench/report/summary.py: the attribution columns and warnings of summary.csv / summary.md."""

import json
import os

import pytest

from bench import leg as L
from bench.report import summary as SUM

from helpers.attribution_rows import CLOCK, P_ATTR_ROW, ROW


def test_summary_has_the_attribution_and_the_footer_column():
    for c in ("median_footer_read_s", "gpu_engine_s", "merge_s", "io_wait_s", "gpu_idle_s"):
        assert c in SUM.COLS
    assert "median_with_footers_s" not in SUM.COLS and "unattributed_s" not in SUM.COLS
    assert any("nova-bf" in n and "schema" in n for n in SUM.COL_NOTES)   # the footer-read doc note


def test_summary_columns_are_the_views():
    for c in SUM.HOST_COLS + SUM.GPU_COLS + SUM.INFO_COLS:
        assert c in SUM.COLS
    assert not any("pct" in c for c in SUM.COLS) and "harness_s" not in SUM.COLS


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_spare_batched_without_stage_events_does_not_put_all_of_lib_in_engine():
    row = dict(ROW, spare_score_s=None, spare_topk_s=None)
    a = L.attribution_harness(row, CLOCK, 0.0, "spare", "on",
                              {"spare_mode": "dot", "spare_events": "off", "widen_where": "device"})
    g = a["gpu_view"]
    assert g["parts"]["engine_s"] is None and g["parts"]["batching_s"] is None
    assert g["parts"]["engine_and_batching_s"] == 17.0
    assert g["split_measurable"] is False and "no_spare_events" in g["why"]
    assert a["engine_s"] is None and g["additive"]
    from bench.report import summary as SUM
    v = SUM.attr_values(a)
    assert v["gpu_engine_s"] is None and v["gpu_engine_and_batching_s"] == 17.0


def test_no_with_footers_column_anywhere():
    from bench.report import summary as SUM
    assert "median_with_footers_s" not in SUM.COLS and "median_footer_read_s" in SUM.COLS
    for root, _, files in os.walk(os.path.join(CB, "bench")):
        for f in files:
            if f.endswith(".py"):
                assert "with_footers" not in open(os.path.join(root, f)).read(), f
    assert "with_footers" not in open(os.path.join(CB, "README.md")).read()


def _leg(rd, tag, att, ok=True):
    os.makedirs(os.path.join(rd, "legs"), exist_ok=True)
    s, sy, r = tag.split("__")
    json.dump({"tag": tag, "search": s, "system": sy, "round": int(r[1:]), "ok": ok,
               "kind": "harness", "timed_seconds": 1.0, "attribution": att},
              open(os.path.join(rd, "legs", f"{tag}.json"), "w"))


def test_a_non_additive_measurable_view_is_a_summary_warning(tmp_path):
    from bench.report import summary as SUM
    rd = str(tmp_path)
    _leg(rd, "s__faiss__r1", L.attribution_harness(dict(P_ATTR_ROW), 1.05, 0.0, "gpu", "off", {}))
    _leg(rd, "s__cuvs__r1", L.attribution_harness(dict(P_ATTR_ROW), 1.0, 0.0, "gpu", "off", {}))
    w = SUM.attribution_warnings(rd)
    assert len(w) == 1 and w[0].startswith("s__faiss__r1: the attribution host_view does not add up")
    # the GPU view is not measurable here (events off): never a warning
    assert not any("gpu_view" in x for x in w)
    md = SUM.write(rd, [], [], [], [])
    assert "Warnings (1, not failures):" in md and "s__faiss__r1: the attribution host_view" in md
    assert "Warnings" in open(os.path.join(rd, "summary.md")).read()


def test_summary_gpu_view_has_cuvs_device_merge_column():
    """The GPU view's device_merge_s (cuVS's own recipe step) is its own summary column, kept out
    of gpu_ours, so a cuVS row's GPU columns sum to its GPU view."""
    parts = {"engine_s": 1.0, "batching_s": 0.0, "device_merge_s": 0.25, "ids_s": 0.1,
             "widen_s": 0.05, "normalize_s": 0.0, "device_other_s": 0.1, "gpu_idle_s": 0.5}
    a = {"kind": "harness", "gpu_view": {"measurable": True, "parts": parts}}
    v = SUM.attr_values(a)
    assert "gpu_device_merge_s" in SUM.GPU_COLS and v["gpu_device_merge_s"] == 0.25
    assert v["gpu_ours_s"] == pytest.approx(0.25)     # ids + widen + normalize + other only
    assert sum(v[c] for c in SUM.GPU_COLS if v[c] is not None) == pytest.approx(sum(parts.values()))
    parts.pop("device_merge_s")                       # a non-cuVS leg: blank, not 0
    assert SUM.attr_values(a)["gpu_device_merge_s"] is None
