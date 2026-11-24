"""Generate timing breakdown figures from saved benchmark results.

Produces two figures per search:

- attribution: Detailed timing breakdown for each configuration, using
  the run closest to the median total time.
- time_parts: Host and GPU timing breakdowns from leg records, using
  median values across rounds.

Timing segments distinguish I/O stalls, data transfers, search, and
merging. Overlapping work is reported separately to avoid double-counting.

For asynchronous GPU execution, attribution uses the GPU timeline rather
than host call times. Failed or unverified legs are excluded where required.
"""

import argparse
import contextlib
import glob
import io
import json
import os
import statistics
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main(argv=None) -> int:
    """`bench plot`: both figures for every search of the run dirs."""
    ap = argparse.ArgumentParser(prog="bench plot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("-o", "--out", default=None, help="output dir (default RUN_DIR/figures)")
    a = ap.parse_args(argv)
    for d in a.run_dir:
        if not os.path.isdir(os.path.join(d, "legs")):
            print(f"bench plot: {d} is not a results dir (no legs/)", file=sys.stderr)
            return 2
    out = a.out or os.path.join(a.run_dir[0], "figures")
    os.makedirs(out, exist_ok=True)
    rc = 0
    err = io.StringIO()   # matplotlib's warnings; shown only when the figure fails
    try:
        with contextlib.redirect_stderr(err):
            time_parts(out, a.run_dir)
    except Exception:  # noqa: BLE001  reported; the attribution figures still draw
        print(f"bench plot: time-parts figure FAILED: {err.getvalue()[-500:]}"
              f"{traceback.format_exc()[-1500:]}", file=sys.stderr)
        rc = 1
    by = leg_csvs(a.run_dir)
    if not by:
        print("bench plot: no harness leg CSVs (nova-bf legs write none)", file=sys.stderr)
        return 1
    for search, csvs in sorted(by.items()):
        fig = os.path.join(out, f"{search}_attribution.png")
        # the table, warnings and any error go to <search>.txt
        with open(os.path.join(out, f"{search}.txt"), "w") as f, \
                contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                frc = attribution(csvs, fig)
            except SystemExit as e:   # read_csvs / attribution refuse with a message
                if e.code not in (None, 0):
                    print(e.code if isinstance(e.code, str) else f"exit {e.code}", file=sys.stderr)
                frc = 0 if e.code in (None, 0) else e.code if isinstance(e.code, int) else 1
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                frc = 1
        print(f"{search}: {len(csvs)} CSV(s) -> {fig}" + ("" if frc == 0 else
                                                          f"  FAILED (exit {frc})"))
        rc = rc or frc
    return 1 if rc else 0


def leg_csvs(run_dirs: list[str]) -> dict[str, list[str]]:
    """{search: [its harness leg CSVs]} of the run dirs (trace CSVs excluded)."""
    by = {}
    for d in run_dirs:
        for p in sorted(glob.glob(os.path.join(d, "legs", "*.csv"))):
            if p.endswith(".trace.csv"):
                continue
            search = os.path.basename(p).split("__", 1)[0]
            by.setdefault(search, []).append(p)
    return by


# =============================================================================================
# The attribution figure: stacked segments from the harness CSVs
# =============================================================================================
# the backend call bucket (renamed in one place when the backend changes)
CALL, CALL_NAME = "engine_call_s", "engine_call"
# the four contiguous buckets the run records (printed table)
BUCKETS = ["source_stall_s", CALL, "merge_s", "other_s"]
# drawn segments, bottom to top: key, legend label, colour, hatch, text colour.
# The GPU keys split CALL (gpu_gap = CALL minus the event parts); "call" is the unsplit
# fallback, drawn instead of them for a bar without usable events.
GPU_PARTS = ["h2d", "widen", "search", "d2h"]  # <part>_s columns
SEGMENTS = [
    # GPU-timeline bars only (pipe mode, see segments): total_s minus the
    # compute stream's busy spans, i.e. what the host side cost the GPU
    ("gpu_idle", "gpu_idle (GPU-timeline bar: stall + merge + exposed copies + loop)",
     "#c9d6e8", "\\\\", "#0b0b0b"),
    ("source_stall", "source_stall", "#2a78d6", None, "white"),
    ("h2d", "h2d", "#f5b48f", None, "#0b0b0b"),
    ("widen", "widen (device)", "#fbd8c3", None, "#0b0b0b"),
    ("normalize", "normalize (FAISS cosine)", "#c99a2e", None, "#0b0b0b"),
    ("lib", "search: library calls (lib_s)", "#b8461a", None, "white"),
    ("search", "search (GEMM + k-select; with lib_s: the harness's part)", "#d4581f",
     None, "white"),
    ("d2h", "d2h", "#8f3a12", None, "white"),
    ("gpu_gap", "gpu_gap (launch / host)", "#f4f1ec", "////", "#0b0b0b"),
    ("call", f"{CALL_NAME} (not split)", "#eb6834", None, "white"),
    ("merge", "merge", "#1baf7a", None, "white"),
    ("other", "other", "#9a9994", None, "white"),
]
SEG_KEYS = [sg[0] for sg in SEGMENTS]
SPLIT = GPU_PARTS + ["gpu_gap", "lib"]
# a GPU-timeline bar's segments (pipe mode): they add up to total_s
TIMELINE = ["gpu_idle", "widen", "normalize", "lib", "search"]
# software / hardware a row was measured with: part of the config (one bar never spans
# versions)
VERSIONS = ["torch_version", "torch_cuda", "pyarrow_version", "numpy_version",
            "merge_faiss_version", "gpu_name", "driver"]
CONFIG = ["regime", "backend", "engine", "engine_version", "queries_mode", "nq", "k",
          "metric", "d", "n_files", "max_files", "rows", "rows_per_file", "prefetch", "workers",
          "readers", "reader", "reader_pool", "reader_pa_cpu_threads", "reader_pa_io_threads",
          "vectors_mem_limit",
          "temp_mem", "rmm_pool", "omp_threads", "pa_cpu_threads",
          "pa_io_threads", "read_mode", "read_mode_effective", "range_concurrency",
          "range_scope", "range_mib", "row_groups_per_file", "stored_dtype", "dtype",
          "widen_where", "gpu_events", "h2d_mode", "d2h_mode", "corpus", "column",
          "query_file",
          "query_column", "seed", "host",
          "filter", "filter_mode", "filter_kind", "filter_skip", "filter_compact",
          "filter_threads", "query_rows", "filter_text_impl", "duckdb_version"] + VERSIONS + ["modality", "sparse_batches",
                                                                   "spare_mode", "spare_events",
                                                                   "sparse_project",
                                                                   "pylate_mode",
                                                                   "pylate_batches",
                                                                   "pylate_rerank_results",
                                                                   "cuvs_merge", "cuvs_sync"]
# annotation columns that may be empty (not measured): numeric, blank = NaN
ANNOTATIONS = ["h2d_s", "widen_s", "normalize_s", "search_s", "d2h_s", "ids_s", "build_s", "engine_host_s", "device_merge_s", "spare_plan_s",
               "spare_score_s", "spare_topk_s", "engine_gap_s", "gpu_idle_s",
               "lib_s", "search_harness_s",
               "sync_wait_s", "sync_waits", "gpu_alloc_peak_bytes", "fetch_s", "decode_s",
               "read_table_s", "convert_s", "publish_s", "handoff_s", "net_rx_MBps",
               "gt_recall"]
# integer-valued columns: printed without decimals (blank / n/a stays blank)
INT_COLUMNS = ["sync_waits", "gpu_alloc_peak_bytes"]
INK, INK2, GRID, HATCH = "#0b0b0b", "#52514e", "#e4e3df", "#b5b1a8"


def attribution(csvs, out) -> int:
    """The attribution figure of harness CSVs into `out` (a .png); prints its table."""
    med = median_rows(read_csvs(csvs))
    if med.empty:
        raise SystemExit("no non-warmup rows")
    med = numeric_annotations(med)
    print(table(med, table_columns(med)))
    plot(med, out)
    print(f"wrote {out}")
    return 0


def read_csvs(paths):
    """The run CSVs, concatenated; refuses one that lacks a column the plot needs.

    CONFIG columns are read as strings: pandas infers each file's types on its own (an
    all-number temp_mem column is int in one CSV and object next to "default" in another),
    and 1536 != "1536" would split one config into two bars. Bucket / annotation columns stay
    numeric.

    Rows of a pass whose --verify failed (verified == failed) are dropped with a warning:
    their timings belong to a pass with wrong results. Rows whose check raised (verified ==
    error: MemoryError, I/O, a signal) are kept with a note: their results were never judged."""
    need = CONFIG + BUCKETS + ANNOTATIONS + ["warmup", "total_s", "verified"]
    dfs = []
    for p in paths:
        df = pd.read_csv(p, keep_default_na=False, dtype={c: str for c in CONFIG})
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"{p}: missing columns {missing} (a CSV from an older "
                             "harness?); re-run it with `bench run`")
        bad = df["verified"].astype(str) == "failed"
        if bad.any():
            print(f"WARNING: {p}: dropping {int(bad.sum())} row(s) of runs whose "
                  "--verify FAILED (verified=failed)", file=sys.stderr)
            df = df[~bad]
        err = df["verified"].astype(str) == "error"
        if err.any():
            print(f"NOTE: {p}: keeping {int(err.sum())} row(s) whose --verify check "
                  "raised (verified=error): their results were never judged", file=sys.stderr)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def median_rows(df):
    """Per config, the non-warmup repeat with the (lower) median total_s, and n_repeats."""
    df = df[df["warmup"] == 0]
    out = []
    for _, g in df.groupby(CONFIG, sort=False, dropna=False):
        g = g.sort_values("total_s")
        row = g.iloc[(len(g) - 1) // 2].copy()
        row["n_repeats"] = len(g)
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def numeric_annotations(med):
    """Annotation columns as numbers; an empty cell (not measured) becomes NaN, which the
    table prints blank and nothing reads as 0."""
    med = med.copy()
    for c in ANNOTATIONS:
        med[c] = pd.to_numeric(med[c], errors="coerce")
    return med


def table(med, cols):
    """The printed table: floats with 4 decimals, INT_COLUMNS without, an empty (not
    measured) cell blank."""
    fmts = {c: (lambda v: "" if pd.isna(v) else f"{v:.0f}") for c in INT_COLUMNS
            if c in cols}
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        return med[cols].to_string(index=False, na_rep="", formatters=fmts,
                                   float_format=lambda v: f"{v:.4f}")


# columns label() shows for every bar (d and k only when they differ, see dk_title);
# labelled() adds the ones shown only for some bars
LABELLED = {"engine", "backend", "regime", "nq", "rows_per_file", "d", "k",
            "prefetch", "readers", "reader", "row_groups_per_file", "read_mode_effective"}
GPU_LABELLED = {"queries_mode", "stored_dtype", "dtype", "widen_where",
                "gpu_events"}  # gpu / cuvs bars only
PATHS = {"corpus", "query_file"}  # shown by their last path component (full_paths)


def labelled(r):
    """The config columns label() itself shows for bar ``r`` (prefetch 0's "serial" stands
    for its readers / reader)."""
    cols = set(LABELLED)
    if r["backend"] in ("gpu", "cuvs", "spare", "pylate"):
        cols |= GPU_LABELLED
    return cols


def varying(med, cols=CONFIG):
    """The ``cols`` whose value differs between the bars (compared as strings)."""
    return [c for c in cols if c in med and med[c].astype(str).nunique() > 1]


def compact(c, v, full=False):
    """``key=value`` for a label; a path by its last component, or whole with ``full`` (for
    paths whose last components collide)."""
    v = str(v)
    if c in PATHS and v and not full:
        v = os.path.basename(v.split("?")[0].rstrip("/")) or v
    return f"{c}={v}"


def full_paths(med):
    """The PATHS columns whose distinct values share a last component among the bars
    (/data/fineweb vs s3://bkt/fineweb/), so they are shown in full."""
    out = set()
    for c in PATHS:
        if c in med:
            vals = set(med[c].astype(str))
            if len({compact(c, v) for v in vals}) < len(vals):
                out.add(c)
    return out


def label(r, dk=False, extra=(), full=()):
    """Bar label. ``dk`` adds d and k (when the chart mixes them); ``extra`` lists more
    config columns to show as key=value where the label does not already show them
    (label_extras); ``full`` names path columns shown in full (full_paths)."""
    engine = r["engine"] if r["backend"] == r["engine"] else f"{r['engine']}/{r['backend']}"
    s = f"{engine} {r['regime']}\nnq={r['nq']} rpf={r['rows_per_file']}"
    if dk:
        s += f"\nd={r['d']} k={r['k']}"
    s += "\npf=0 serial" if str(r["prefetch"]) == "0" else \
        f"\npf={r['prefetch']} readers={r['readers']} {r['reader']}"
    s += f"\nrg={r['row_groups_per_file']} {r['read_mode_effective']}"
    if r["backend"] in ("gpu", "cuvs", "spare", "pylate"):
        s += f"\n{r['queries_mode']} {r['stored_dtype']}->{r['dtype']}"
        s += f" ({r['widen_where']})"
        if r["gpu_events"] == "off":
            s += "\nno events"
    shown = labelled(r)
    for c in extra:
        if c not in shown:
            s += "\n" + compact(c, r[c], c in full)
    return s


def label_extras(med):
    """Config columns that differ between bars whose label does not show them. (A bar that
    shows a column already differs from one that does not by backend, on the label.)"""
    rows = [r for _, r in med.iterrows()]
    out = []
    for c in varying(med):
        hidden = {str(r[c]) for r in rows if c not in labelled(r)}
        if len(hidden) > 1:
            out.append(c)
    return out


def dk_title(med):
    """``(title text, on_bars)``: d and k for the title when every bar shares them; else the
    title says they are mixed and each bar label carries its own (``on_bars``)."""
    dks = list(dict.fromkeys((str(r["d"]), str(r["k"])) for _, r in med.iterrows()))
    if len(dks) == 1:
        return f"d={dks[0][0]}, k={dks[0][1]}", False
    return "mixed d / k (see bars)", True


def segments(med):
    """Seconds per drawn segment, one row per bar: columns SEG_KEYS, plus `split` (CALL is
    split into GPU sub-segments) and `timeline` (a GPU-timeline bar).

    Split bars: h2d / widen / search / d2h come from the event columns (widen only when
    widen_where == device; a host widen is outside CALL) and gpu_gap = CALL minus those
    parts, so they add up to CALL exactly. A negative gap (event granularity) is drawn as 0
    and taken out of search, keeping the sum. Unsplit bars (events off / blank, CPU backend,
    or event parts that are all 0) put all of CALL into "call". normalize (normalize_s,
    FAISS cosine) is one more event part of a split bar; on an unsplit bar a measured
    normalize_s (host perf_counter, a contiguous part of CALL) is its own segment and "call"
    holds the rest; blank (ip, cuVS, events off) counts as 0. Every row sums to total_s
    (other_s is total minus the recorded buckets; a negative other, which harness_cli.py
    warns about, is drawn as 0).

    Pipe-mode rows (--d2h-overlap) with gpu_idle_s become GPU-timeline bars (TIMELINE
    segments only, see the module docstring); without it they are unsplit. On split and
    timeline bars with lib_s, the library's part of search is the "lib" segment and
    "search" keeps the harness's part (search_harness_s)."""
    seg = pd.DataFrame(0.0, index=med.index, columns=SEG_KEYS)
    seg["source_stall"] = med["source_stall_s"].astype(float)
    seg["merge"] = med["merge_s"].astype(float)
    seg["other"] = med["other_s"].astype(float)
    call = med[CALL].astype(float)
    parts = pd.DataFrame({p: pd.to_numeric(med[f"{p}_s"], errors="coerce")
                          for p in GPU_PARTS})
    if "h2d_exposed_s" in med:
        # the h2d segment is the copy time inside engine_call: all of h2d_s, except with
        # --h2d-overlap, where only the exposed part is (the rest ran on the copy stream
        # during the previous search). Without this column h2d_s is that time.
        exp = pd.to_numeric(med["h2d_exposed_s"], errors="coerce")
        parts["h2d"] = exp.where(exp.notna(), parts["h2d"])
    parts.loc[med["widen_where"].astype(str) != "device", "widen"] = 0.0
    split = ((med["gpu_events"].astype(str) == "on") & parts.notna().all(axis=1)
             & (parts.sum(axis=1) > 0))
    norm = pd.to_numeric(med["normalize_s"], errors="coerce").fillna(0.0)
    seg["normalize"] = norm
    gap = call - parts.sum(axis=1) - norm
    for p in GPU_PARTS:
        seg.loc[split, p] = parts.loc[split, p]
    seg.loc[split, "gpu_gap"] = gap[split]
    neg = split & (gap < 0)
    seg.loc[neg, "search"] += gap[neg]
    seg.loc[neg, "gpu_gap"] = 0.0
    seg.loc[~split, "call"] = call[~split] - norm[~split]

    # pipe mode (--d2h-overlap): engine_call is only the time the host was blocked in the
    # call, not the engine's time (see the module docstring). Such a bar is drawn on the GPU
    # timeline instead: gpu_idle_s (total_s minus the compute stream's busy spans) + device
    # widen + normalize + search, which add up to total_s by construction. Without
    # gpu_idle_s (events off, or not recorded) it stays one unsplit engine_call (host wall).
    def col(c):  # a column a hand-built frame may lack: not measured
        return med[c] if c in med else pd.Series("", index=med.index)

    pipe = col("d2h_mode").astype(str) == "pinned-overlap"
    idle = pd.to_numeric(col("gpu_idle_s"), errors="coerce")
    timeline = pipe & split & idle.notna()
    seg.loc[pipe & ~timeline, list(SPLIT)] = 0.0
    seg.loc[pipe & ~timeline, "call"] = call[pipe & ~timeline] - norm[pipe & ~timeline]
    if timeline.any():
        busy = parts["widen"] + norm + parts["search"]
        seg.loc[timeline, :] = 0.0
        seg.loc[timeline, "widen"] = parts.loc[timeline, "widen"]
        seg.loc[timeline, "normalize"] = norm[timeline]
        seg.loc[timeline, "search"] = parts.loc[timeline, "search"]
        # gpu_idle_s is total_s - busy as harness_cli.py measured it; recompute it from the
        # drawn parts so the stack sums exactly (they agree to rounding)
        seg.loc[timeline, "gpu_idle"] = (med["total_s"].astype(float) - busy)[timeline]

    # split the search span into library and harness on every split bar that recorded
    # library spans: lib = the library's own calls, search = the rest (ids, padding, CSR
    # views, ...: search_harness_s)
    lib = pd.to_numeric(col("lib_s"), errors="coerce")
    has_lib = (split | timeline) & lib.notna()
    take = lib.where(has_lib, 0.0).clip(upper=seg["search"].clip(lower=0))
    seg.loc[has_lib, "lib"] = take[has_lib]
    seg.loc[has_lib, "search"] -= take[has_lib]
    seg["split"] = split & ~(pipe & ~timeline)
    seg["timeline"] = timeline
    return seg


def plot(med, out):
    """Draw the two-panel stacked bars of the median rows into `out`."""
    n = len(med)
    x = np.arange(n)
    fig, (ax_pct, ax_abs) = plt.subplots(
        2, 1, figsize=(max(6.0, 1.5 * n + 2), 8), sharex=True,
        gridspec_kw={"hspace": 0.12})
    w = 0.6
    tot = med["total_s"].to_numpy(float)
    seg = segments(med)
    # legend lists each segment once, and only the modes some bar uses
    used = [k for k in SEG_KEYS if (seg[k] > 0).any()]
    for ax, norm in ((ax_pct, True), (ax_abs, False)):
        bottom = np.zeros(n)
        for key, name, color, hatch, ink in SEGMENTS:
            if key not in used:
                continue
            v = seg[key].to_numpy(float).clip(min=0)
            h = 100 * v / tot if norm else v
            # the hatch takes the edge colour, so a hatched segment gets its white
            # separator from a second, unfilled bar
            ax.bar(x, h, w, bottom=bottom, color=color, label=name, hatch=hatch,
                   edgecolor=HATCH if hatch else "white",
                   linewidth=0 if hatch else 1.0)
            if hatch:
                ax.bar(x, h, w, bottom=bottom, color="none", edgecolor="white",
                       linewidth=1.0)
            if norm:
                for i in range(n):
                    if h[i] >= 7:
                        ax.text(x[i], bottom[i] + h[i] / 2, f"{h[i]:.0f}%",
                                ha="center", va="center", fontsize=8, color=ink)
            bottom += h
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
    ax_pct.set_ylim(0, 100)
    ax_pct.set_ylabel("share of wall time (%)", color=INK)
    ax_abs.set_ylabel("seconds (median repeat)", color=INK)
    off = w / 2 + 0.08
    sc_dec = ax_abs.scatter(x + off, med["decode_s"], marker="D", s=40,
                            color=INK, label="worker decode_s", zorder=3)
    sc_fet = ax_abs.scatter(x + off, med["fetch_s"], marker="o", s=48,
                            facecolors="none", edgecolors=INK, linewidths=1.5,
                            label="worker fetch_s", zorder=3)
    for i in range(n):
        ax_abs.text(x[i], tot[i], f"{tot[i]:.3g}s", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    dk, dk_on_bars = dk_title(med)
    ax_abs.set_xticks(x)
    extra, full = label_extras(med), full_paths(med)
    ax_abs.set_xticklabels([label(r, dk=dk_on_bars, extra=extra, full=full)
                            + ("\n[GPU timeline]" if seg.loc[i, "timeline"] else "")
                            for i, r in med.iterrows()], fontsize=7.5)
    h1, l1 = ax_pct.get_legend_handles_labels()
    fig.legend(h1 + [sc_dec, sc_fet], l1 + ["worker decode_s", "worker fetch_s"],
               loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.9))
    engines = " vs ".join(dict.fromkeys(f"{r['engine']} ({r['backend']})"
                                        for _, r in med.iterrows()))
    metrics = list(dict.fromkeys(str(m) for m in med["metric"]))
    metric = metrics[0] if len(metrics) == 1 else "mixed metric (see bars)"
    fig.suptitle(f"Where the wall time goes - {engines}, {dk}, {metric}", y=1.0,
                 color=INK, fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table_columns(med):
    """The printed table's columns: the usual ones plus every other config column that
    differs between the bars."""
    cols = ["engine", "backend", "regime", "nq", "rows_per_file", "row_groups_per_file",
            "prefetch", "workers", "readers", "reader", "read_mode_effective",
            "widen_where", "gpu_events", "pa_cpu_threads"]
    cols += [c for c in varying(med) if c not in cols]
    return cols + ["n_repeats", "total_s"] + BUCKETS + ANNOTATIONS


# =============================================================================================
# The time-parts figure: where each harness system's time goes
# =============================================================================================
HOST = (("io_wait_s", "io wait / reader"), ("engine_call_s", "engine call"),
        ("merge_s", "merge"), ("harness_other_s", "harness other"))
GPU = (("gpu_engine_s", "engine"), ("gpu_batching_s", "batching (ours)"),
       ("gpu_engine_and_batching_s", "engine + batching (unsplit)"),
       ("gpu_ours_s", "our device work"), ("gpu_idle_s", "gpu idle"))
INFO = ("h2d_busy_s", "d2h_busy_s", "filter_s", "filter_eval_s")


def collect(run_dirs):
    """{search: {system: [attribution blocks]}} of the usable harness leg records."""
    by = {}
    for d in run_dirs:
        for p in sorted(glob.glob(os.path.join(d, "legs", "*.json"))):
            if p.endswith(".clock.json"):
                continue
            try:
                r = json.load(open(p))
            except ValueError:
                continue
            a = r.get("attribution") or {}
            if r.get("e2e") or not r.get("ok") or r.get("timed_seconds") is None:
                continue
            if a.get("kind") != "harness":   # nova-bf: its total only
                continue
            by.setdefault(r["search"], {}).setdefault(r["system"], []).append(a)
    return by


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def time_parts_rows(by_sys):
    """One row per system: the median of every part over its rounds."""
    from bench.report.summary import attr_values
    rows = []
    for sy, atts in by_sys.items():
        vals = [attr_values(a) for a in atts]
        row = {"system": sy, "n": len(atts), "total_s": _med([a.get("total_s") for a in atts])}
        for k in [k for k, _ in HOST + GPU] + list(INFO):
            row[k] = _med([v.get(k) for v in vals])
        rows.append(row)
    return rows


def _stack(ax, rows, parts, title):
    x = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for k, label_ in parts:
        vals = [r[k] or 0.0 for r in rows]
        if not any(vals):
            continue
        ax.bar(x, vals, bottom=bottom, label=label_)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x, [r["system"] for r in rows], rotation=30, ha="right")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def time_parts(out, run_dirs) -> int:
    """The time-parts figure and table of every search into `out`."""
    os.makedirs(out, exist_ok=True)
    by = collect(run_dirs)
    if not by:
        print("no usable harness leg records with an attribution: no time-parts figure")
        return 0
    for search, by_sys in sorted(by.items()):
        rows = time_parts_rows(by_sys)
        f = (lambda v, fmt=".4f": "-" if v is None else format(v, fmt))
        cols = ("total_s",) + tuple(k for k, _ in HOST) + tuple(k for k, _ in GPU) + INFO
        lines = ["| system | n | " + " | ".join(c[:-2] for c in cols) + " |",
                 "|---|---|" + "---|" * len(cols)]
        lines += [f"| {r['system']} | {r['n']} | " + " | ".join(f(r[c]) for c in cols) + " |"
                  for r in rows]
        lines.append("host view (io_wait | engine_call | merge | harness_other) and GPU view "
                     "(gpu_engine | gpu_batching | gpu_engine_and_batching | gpu_ours | "
                     "gpu_idle) each sum to a leg's LegClock time; h2d_busy / d2h_busy / filter "
                     "/ filter_eval overlap them (information, never stacked) "
                     "(README.md#attribution). nova-bf: total only, not drawn.")
        open(os.path.join(out, f"{search}_time_parts.txt"), "w").write("\n".join(lines) + "\n")
        w = max(4, 1.2 * len(rows) + 2)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(2 * w, 4), sharey=True)
        _stack(a1, rows, HOST, "host view (the bar): host thread wall")
        _stack(a2, rows, GPU, "GPU view: compute stream")
        a1.set_ylabel("seconds (median over rounds)")
        fig.suptitle(f"{search}: where the time goes (harness systems)")
        fig.text(0.01, 0.01, "host view: host perf_counter; GPU view: CUDA events", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{search}_time_parts.png"), dpi=120)
        plt.close(fig)
        print(f"{search}: {len(rows)} harness system(s) -> {search}_time_parts.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
