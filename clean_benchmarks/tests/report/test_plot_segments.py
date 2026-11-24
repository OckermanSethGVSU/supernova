import pandas as pd
import pytest

from bench.report import plots as plot

EVENTS = ["h2d_s", "widen_s", "search_s", "d2h_s", "engine_gap_s", "sync_wait_s",
          "sync_waits"]


def row(regime="stream", backend="gpu", gpu_events="on", widen_where="device",
        queries_mode="resident", h2d=0.10, widen=0.05, search=0.60, d2h=0.02,
        call=0.80, stall=0.30, merge=0.07, other=0.01, prefetch=2):
    """One CSV row as the harness writes it (strings; blank = not measured)."""
    r = {c: "x" for c in plot.CONFIG} | {
        "regime": regime, "backend": backend, "queries_mode": queries_mode,
        "gpu_events": gpu_events, "widen_where": widen_where, "nq": 16, "k": 10,
        "d": 32, "rows_per_file": 100, "row_groups_per_file": 1, "prefetch": prefetch,
        "workers": 2, "read_mode_effective": "arrow", "stored_dtype": "f16",
        "dtype": "f32", "warmup": 0, "repeat": 0, "pa_cpu_threads": 4,
        "verified": "not_run"}
    r |= {"source_stall_s": stall, "engine_call_s": call, "merge_s": merge,
          "other_s": other, "total_s": stall + call + merge + other}
    r |= {c: "" for c in plot.ANNOTATIONS}
    r |= {"fetch_s": 0.1, "decode_s": 0.2, "read_table_s": 0.15, "convert_s": 0.05}
    if gpu_events == "on":
        r |= {"h2d_s": h2d, "widen_s": widen, "search_s": search, "d2h_s": d2h,
              "sync_wait_s": 0.0, "sync_waits": 0}
        dev = h2d + search + d2h + (widen if widen_where == "device" else 0)
        r["engine_gap_s"] = call - dev if search else 0.0
    elif widen_where == "host" and regime == "stream":
        r["widen_s"] = widen  # worker cast still timed with --no-gpu-events
    return {k: str(v) for k, v in r.items()}


def med_of(rows, tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    med = plot.median_rows(pd.read_csv(p, keep_default_na=False))
    return plot.numeric_annotations(med)


MIXED = [
    row(),                                          # 0 events on, device widen
    row(gpu_events="off", prefetch=3),              # 1 --no-gpu-events: blanks
    row(widen_where="host", widen=0.5, prefetch=4),  # 2 host widen (big)
    row(backend="cpu", gpu_events="n/a", h2d=0, widen=0, search=0, d2h=0,
        prefetch=5),                                # 3 CPU backend
    row(search=0.70, h2d=0.08, widen=0.03, d2h=0.02, prefetch=6),  # 4 gap < 0
    row(gpu_events="n/a", prefetch=7),              # 5 events n/a: blanks, unsplit
]


def test_segments_sum_to_total_and_fallback(tmp_path):
    med = med_of(MIXED, tmp_path)
    seg = plot.segments(med).set_index(med["prefetch"].astype(int))
    tot = pd.Series(med["total_s"].to_numpy(float), index=seg.index)
    assert (seg[plot.SEG_KEYS].sum(axis=1) - tot).abs().max() < 1e-12
    assert list(seg["split"]) == [True, False, True, False, True, False]
    call = pd.Series(med["engine_call_s"].to_numpy(float), index=seg.index)
    for pf in (3, 5, 7):  # fallback: one combined call segment
        assert seg.loc[pf, "call"] == pytest.approx(call[pf])
        assert (seg.loc[pf, plot.SPLIT] == 0).all()
    for pf in (2, 4, 6):  # split: the GPU parts are the whole call
        assert seg.loc[pf, "call"] == 0
        assert seg.loc[pf, plot.SPLIT].sum() == pytest.approx(call[pf])
    # merge is always its own segment, never folded into the call
    merge = pd.Series(med["merge_s"].to_numpy(float), index=seg.index)
    assert (seg["merge"] == merge).all()
    assert seg.loc[2, "search"] == pytest.approx(0.60)
    assert seg.loc[2, "widen"] == pytest.approx(0.05)
    assert seg.loc[2, "gpu_gap"] == pytest.approx(0.80 - 0.77)


def test_host_widen_is_not_stacked(tmp_path):
    seg = plot.segments(med_of([row(widen_where="host", widen=5.0)], tmp_path))
    assert seg.loc[0, "widen"] == 0  # worker-side, outside engine_call
    assert seg.loc[0, "gpu_gap"] == pytest.approx(0.80 - 0.72)
    assert (seg[plot.SEG_KEYS].to_numpy() >= 0).all()
    assert seg[plot.SEG_KEYS].sum(axis=1)[0] == pytest.approx(1.18)


def test_negative_gap_is_drawn_as_zero_but_printed_raw(tmp_path):
    med = med_of([MIXED[4]], tmp_path)
    assert med.loc[0, "engine_gap_s"] == pytest.approx(-0.03)
    seg = plot.segments(med)
    assert seg.loc[0, "gpu_gap"] == 0 and seg.loc[0, "search"] == pytest.approx(0.67)
    assert seg[plot.SEG_KEYS].sum(axis=1)[0] == pytest.approx(med.loc[0, "total_s"])
    assert "-0.0300" in plot.table(med, ["engine_gap_s"])


def test_segment_table_covers_both_modes():
    keys = {sg[0] for sg in plot.SEGMENTS}
    assert keys == set(plot.SEG_KEYS) and set(plot.SPLIT) | {"call"} < keys
    assert plot.SEG_KEYS[:2] == ["gpu_idle", "source_stall"]
    assert plot.SEG_KEYS[-2:] == ["merge", "other"]
    assert set(plot.TIMELINE) < keys
    colours = {sg[0]: sg[2] for sg in plot.SEGMENTS}
    assert (colours["source_stall"], colours["call"], colours["merge"],
            colours["other"]) == ("#2a78d6", "#eb6834", "#1baf7a", "#9a9994")
    assert len(set(colours.values())) == len(colours)


def test_plot_renders_mixed_bars(tmp_path, capsys):
    p = tmp_path / "r.csv"
    pd.DataFrame(MIXED).to_csv(p, index=False)
    png = tmp_path / "a.png"
    assert plot.attribution([str(p)], str(png)) == 0
    assert png.stat().st_size > 10_000 and "-0.0300" in capsys.readouterr().out


def pipe_row(idle=0.05, lib=None, **kw):
    """A --d2h-overlap row: engine_call is host-blocked time, smaller than
    the device search (here 0.30 s of call against a 0.60 s search)."""
    r = row(call=0.30, **kw)
    r["d2h_mode"] = "pinned-overlap"
    r["engine_gap_s"] = ""  # the harness blanks it in pipe mode
    if idle is not None:
        busy = float(r["widen_s"] or 0) + float(r["search_s"] or 0)
        r["gpu_idle_s"] = str(float(r["total_s"]) - busy)
    if lib is not None:
        r["lib_s"] = str(lib)
        r["search_harness_s"] = str(float(r["search_s"]) - lib)
    return r


def test_pipe_row_is_a_gpu_timeline_bar(tmp_path):
    med = med_of([pipe_row(lib=0.55)], tmp_path)
    seg = plot.segments(med)
    assert seg.loc[0, "timeline"] and seg.loc[0, "split"]
    # the whole measured search is drawn (library + harness), never cut down
    # to fit the host-blocked engine_call, and the hidden copies are not drawn
    assert seg.loc[0, "lib"] + seg.loc[0, "search"] == pytest.approx(0.60)
    assert seg.loc[0, "lib"] == pytest.approx(0.55)
    assert seg.loc[0, "search"] == pytest.approx(0.05)
    for k in ("h2d", "d2h", "gpu_gap", "call", "source_stall", "merge", "other"):
        assert seg.loc[0, k] == 0
    assert seg.loc[0, "gpu_idle"] == pytest.approx(med.loc[0, "gpu_idle_s"])
    assert seg[plot.SEG_KEYS].sum(axis=1)[0] == pytest.approx(med.loc[0, "total_s"])


def test_pipe_row_without_gpu_idle_is_unsplit(tmp_path):
    med = med_of([pipe_row(idle=None)], tmp_path)
    seg = plot.segments(med)
    assert not seg.loc[0, "timeline"] and not seg.loc[0, "split"]
    assert seg.loc[0, "call"] == pytest.approx(0.30)
    assert (seg.loc[0, plot.SPLIT] == 0).all() and seg.loc[0, "gpu_idle"] == 0
    assert seg[plot.SEG_KEYS].sum(axis=1)[0] == pytest.approx(med.loc[0, "total_s"])


def test_serial_row_splits_search_into_library_and_harness(tmp_path):
    r = row()
    r["lib_s"], r["search_harness_s"] = "0.45", "0.15"
    med = med_of([r], tmp_path)
    seg = plot.segments(med)
    assert not seg.loc[0, "timeline"]
    assert seg.loc[0, "lib"] == pytest.approx(0.45)
    assert seg.loc[0, "search"] == pytest.approx(0.15)
    assert seg.loc[0, "gpu_gap"] == pytest.approx(0.80 - 0.77)
    assert seg[plot.SEG_KEYS].sum(axis=1)[0] == pytest.approx(med.loc[0, "total_s"])


def test_plot_renders_pipe_and_serial_bars(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame(MIXED + [pipe_row(lib=0.55, prefetch=8), pipe_row(idle=None, prefetch=9)]
                 ).to_csv(p, index=False)
    png = tmp_path / "a.png"
    assert plot.attribution([str(p)], str(png)) == 0
    assert png.stat().st_size > 10_000
