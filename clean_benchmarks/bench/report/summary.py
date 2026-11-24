"""Summarize benchmark results across rounds.

Computes median execution times from successful legs and reports failed or
missing rounds. Checks that each leg covers the expected corpus files and
that verification results are present and match the current run.

Generates summary.csv and summary.md with timing results, errors, and
reproducibility information. Exits with an error if any checks fail.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import statistics
import sys


def load_legs(run_dir: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "legs", "*.json"))):
        if p.endswith(".clock.json"):
            continue
        try:
            rec = json.load(open(p))
            if isinstance(rec, dict):
                rec["_stem"] = os.path.basename(p)[:-5]
            out.append(rec)
        except (OSError, ValueError) as exc:
            out.append({"tag": os.path.basename(p)[:-5], "ok": False, "unreadable": True,
                        "failures": [f"unreadable leg record: {exc}"]})
    return out


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def usable_time(v) -> bool:
    """True when a leg's time is a finite number > 0 (NaN, inf, 0 or negative is no time)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


# the harness legs' two attribution views (bench/leg.py attribution_harness): the host view
# (the stacked bar) and the GPU view, then information columns that overlap them
HOST_COLS = ["io_wait_s", "engine_call_s", "merge_s", "harness_other_s"]
GPU_COLS = ["gpu_engine_s", "gpu_batching_s", "gpu_engine_and_batching_s",
            "gpu_device_merge_s", "gpu_ours_s", "gpu_idle_s"]
INFO_COLS = ["h2d_busy_s", "d2h_busy_s", "filter_s", "filter_eval_s"]
ATTR_COLS = HOST_COLS + GPU_COLS + INFO_COLS


def attr_values(a: dict) -> dict:
    """One leg's attribution block -> the summary's attribution columns (None = not measured)."""
    hv = (a.get("host_view") or {}).get("parts") or {}
    gv = a.get("gpu_view") or {}
    gp = gv.get("parts") or {} if gv.get("measurable") else {}
    ours = [gp.get(k) for k in ("ids_s", "widen_s", "normalize_s", "device_other_s")]
    busy, flt = a.get("busy") or {}, a.get("filter") or {}
    return {**{c: hv.get(c) for c in HOST_COLS},
            "gpu_engine_s": gp.get("engine_s"), "gpu_batching_s": gp.get("batching_s"),
            "gpu_engine_and_batching_s": gp.get("engine_and_batching_s"),
            # cuVS's own recipe step (the GPU merge of the running top-k): cuVS's work, never ours
            "gpu_device_merge_s": gp.get("device_merge_s"),
            "gpu_ours_s": sum(ours) if gp and all(x is not None for x in ours) else None,
            "gpu_idle_s": gp.get("gpu_idle_s"),
            "h2d_busy_s": busy.get("h2d_s"), "d2h_busy_s": busy.get("d2h_s"),
            "filter_s": flt.get("filter_s"), "filter_eval_s": flt.get("filter_eval_s")}
# the run's identity, on every row of summary.csv and at the top of summary.md
META_COLS = ["run_commit", "run_dirty", "code_sha", "stack_sha", "experiment_sha256", "checks_ok"]


def summary_meta(run_dir: str) -> dict:
    """The run's identity (META_COLS): commit / dirty / code_sha / stack_sha from
    manifest.json, the sha256 of experiment.json as `bench run` wrote it, and checks.json's
    `ok`. A missing file gives blanks."""
    out = dict.fromkeys(META_COLS, "")
    try:
        m = json.load(open(os.path.join(run_dir, "manifest.json")))
        out.update(run_commit=m.get("commit") or "", run_dirty=m.get("dirty"),
                   code_sha=m.get("code_sha") or "", stack_sha=m.get("stack_sha") or "")
    except (OSError, ValueError):
        pass
    try:
        out["experiment_sha256"] = hashlib.sha256(
            open(os.path.join(run_dir, "experiment.json"), "rb").read()).hexdigest()
    except OSError:
        pass
    try:
        out["checks_ok"] = json.load(open(os.path.join(run_dir, "checks.json"))).get("ok")
    except (OSError, ValueError):
        out["checks_ok"] = "missing"
    return out


def summarize(run_dir: str, exp: dict | None = None, rounds: int | None = None,
              expect_files: int | None = None, require_checks: bool = True) -> tuple:
    """(rows, problems, excluded, e2e legs), by the rules in the module docstring.

    rows has one entry per (search, system) with the median and the per-round values. A
    missing checks.json, or one whose `ok` is not true, is a problem unless
    require_checks=False (`--no-checks`)."""
    if exp is None:
        exp = json.load(open(os.path.join(run_dir, "experiment.json")))
    R = rounds or int(exp["protocol"]["rounds"])
    legs = load_legs(run_dir)
    problems = []
    by_key: dict = {}
    excluded = []
    wanted = {(s["name"], sy) for s in exp["searches"] for sy in s["systems"]}
    e2e_sys = (exp.get("e2e") or {}).get("system")
    seen: dict = {}
    for L in legs:
        if L.get("unreadable"):
            problems.append(f"{L['tag']}: {L['failures'][0]}")
            continue
        if L.get("tag") != L.get("_stem"):   # a copy under another name is not a leg
            problems.append(f"legs/{L.get('_stem')}.json: its tag is {L.get('tag')!r}, not the file "
                            "name: not counted")
            continue
        if L.get("e2e"):
            if L.get("system") != e2e_sys:
                problems.append(f"{L.get('tag')}: an e2e leg of {L.get('system')!r}, not this "
                                "experiment's e2e system: not counted")
            continue
        key = (L.get("search"), L.get("system"))
        rnd = L.get("round")
        if key not in wanted or not isinstance(rnd, int) or not 1 <= rnd <= R:
            problems.append(f"{L.get('tag')}: not a leg of this run (search {key[0]!r}, system "
                            f"{key[1]!r}, round {rnd!r}; the experiment has rounds 1..{R}): not counted")
            continue
        if L["tag"] != f"{key[0]}__{key[1]}__r{rnd}":   # the tag names this leg
            problems.append(f"{L['tag']}: the record says search {key[0]!r} system {key[1]!r} "
                            f"round {rnd!r}, not its tag: not counted")
            continue
        if seen.setdefault((key, rnd), L["_stem"]) != L["_stem"]:
            problems.append(f"{L['_stem']}: a second record of {key[0]} {key[1]} r{rnd} (also "
                            f"{seen[(key, rnd)]}): round not counted")
            by_key.setdefault(key, {}).pop(rnd, None)   # the first record of it, kept below
            continue
        if not L.get("ok") or not usable_time(L.get("timed_seconds")):
            reasons = "; ".join(L.get("failures") or []) or \
                f"no usable time (timed_seconds={L.get('timed_seconds')!r})"
            excluded.append((key, L["round"], reasons))
            problems.append(f"{L['search']} {L['system']} r{L['round']}: excluded: {reasons}")
            by_key.setdefault(key, {})
            continue
        # only the first record of a (search, system, round) gets here; a later one is a
        # problem above, which also drops the stored first one
        by_key.setdefault(key, {})[int(L["round"])] = L
    # every scheduled (search, system) must be there, every round
    for s in exp["searches"]:
        for sy in s["systems"]:
            by_key.setdefault((s["name"], sy), {})
    # coverage: every timed pass covers the run's files (module docstring)
    files_by_search: dict = {}
    for (s, sy), by_r in by_key.items():
        for L in by_r.values():
            if L["kind"] == "harness" and L.get("n_files") is not None:
                files_by_search.setdefault(s, set()).add(L["n_files"])
    for s, fs in sorted(files_by_search.items()):   # one corpus coverage per search
        if len(fs) > 1:
            problems.append(f"{s}: its harness legs cover different file counts {sorted(fs)}")
    run_files, run_why = expected_files(run_dir, exp, expect_files)
    for (s, sy), by_r in list(by_key.items()):
        want, why_src = run_files, run_why
        if want is None and len(files_by_search.get(s, ())) == 1:
            want, why_src = next(iter(files_by_search[s])), "the search's harness legs"
        for r, L in list(by_r.items()):
            nf = L.get("n_files")
            if L["kind"] == "harness" and nf is None:
                why = "the timed pass records no n_files (its coverage is unknown)"
            elif want is not None and nf != want:
                why = f"the timed run covers files={nf}, expected {want} ({why_src})"
            else:
                continue
            problems.append(f"{s} {sy} r{r}: excluded: {why}")
            excluded.append(((s, sy), r, why))
            del by_r[r]
    rows = []
    for (s, sy), by_r in sorted(by_key.items()):
        missing = [r for r in range(1, R + 1) if r not in by_r]
        if missing:
            problems.append(f"{s} {sy}: no usable round(s) {missing}")
        vals = [by_r[r]["timed_seconds"] for r in sorted(by_r)]
        any_leg = next(iter(by_r.values()), None)
        legs_r = [by_r[r] for r in sorted(by_r)]
        att = [L.get("attribution") or {} for L in legs_r]
        rows.append({
            "search": s, "system": sy, "n": len(vals),
            "median_s": statistics.median(vals) if vals else None,
            # the headline is median_s (the timed pass); the harness's up-front footer read is
            # its own column, never added to it (blank for nova-bf, whose schema reads are
            # inside its timed scan)
            "median_footer_read_s": _med([L.get("footer_read_s") for L in legs_r]),
            # where a harness leg's time goes, one median per column over the usable rounds;
            # blank for nova-bf (its total only)
            "attribution_kind": att[0].get("kind", "") if att else "",
            **{c: _med([attr_values(a).get(c) for a in att]) for c in ATTR_COLS},
            "rounds_s": " ".join(f"{by_r[r]['timed_seconds']:.4f}" if r in by_r else "-"
                                 for r in range(1, R + 1)),
            "missing_rounds": " ".join(map(str, missing)),
            "time_source": any_leg["time_source"] if any_leg else "",
            "preset": exp["systems"][sy]["preset"] if sy in exp["systems"] else "",
            # the preset + `+proj` + the "(ours: ...)" tag (systems.base)
            "label": _label(exp, sy),
            "knobs": json.dumps(exp["systems"][sy]["knobs"], sort_keys=True) if sy in exp["systems"] else "",
            "median_peak_rss_GiB": statistics.median(by_r[r]["peak_rss_bytes"] / 2**30 for r in by_r)
            if by_r and all(by_r[r].get("peak_rss_bytes") for r in by_r) else None,
            "GB_read": statistics.median((by_r[r].get("bytes_read") or 0) / 1e9 for r in by_r) if by_r else None,
        })
    e2e = [L for L in legs if L.get("e2e")]
    if e2e_sys and not any(L.get("system") == e2e_sys for L in e2e):
        problems.append(f"e2e {e2e_sys}: no e2e leg record (the experiment has an e2e leg)")
    for L in e2e:
        if not L.get("ok"):
            problems.append(f"e2e {L['system']}: excluded: {'; '.join(L.get('failures') or [])}")
    checks = os.path.join(run_dir, "checks.json")
    if require_checks:
        try:
            c = json.load(open(checks))
        except (OSError, ValueError) as exc:
            problems.append(f"no readable checks.json ({checks}: {type(exc).__name__}); "
                            "`bench summarize --no-checks` to summarize without checks")
        else:
            if c.get("ok") is not True:
                problems.append(f"checks failed (ok={c.get('ok')!r}, {checks})")
            from bench.verify import agree
            stale = agree.input_problems(run_dir, c if isinstance(c, dict) else {})
            if stale:
                problems.append(f"stale checks.json ({len(stale)} input(s) differ from what the "
                                f"checks read; re-run the checks): " + "; ".join(stale[:5]))
    return rows, problems, excluded, e2e


def expected_files(run_dir: str, exp: dict, expect_files: int | None = None) -> tuple:
    """(file count every timed pass must cover, where it comes from), or (None, None):
    --expect-files, else subset.files, else manifest.json's corpus listing n_files."""
    if expect_files is not None:
        return int(expect_files), "--expect-files"
    if (exp.get("subset") or {}).get("files") is not None:
        return int(exp["subset"]["files"]), "the experiment's subset.files"
    try:
        c = (json.load(open(os.path.join(run_dir, "manifest.json"))).get("data") or {}).get("corpus") or {}
    except (OSError, ValueError, AttributeError):
        return None, None
    if isinstance(c.get("n_files"), int) and not c.get("error"):
        return c["n_files"], "the run's corpus listing in manifest.json"
    return None, None


def score_error_lines(run_dir: str) -> list[str]:
    """summary.md: one line per search with each system's largest own-score error against
    float64 (checks.json score_error), the evidence behind "fp32-exact outputs"."""
    try:
        se = json.load(open(os.path.join(run_dir, "checks.json"))).get("score_error") or {}
    except (OSError, ValueError, AttributeError):
        return []
    out = []
    for s, by in sorted(se.items()):
        if by:
            out.append(f"- {s}: " + ", ".join(
                f"{nm} {e.get('max_ulps', 0):.1f} ULPs (abs {e.get('max_abs', 0):.2e}; "
                f"adjudicated ids {e.get('max_ulps_adjudicated', 0):.1f})"
                for nm, e in sorted(by.items())))
    return out


def _label(exp: dict, sy: str) -> str:
    from bench.systems import base
    e = exp["systems"].get(sy)
    return base.display_label(e["preset"], e.get("knobs") or {}) if e else ""


COLS = ["search", "system", "n", "median_s", "median_footer_read_s",
        "rounds_s", "missing_rounds", "time_source", "preset", "label",
        "knobs", "median_peak_rss_GiB", "GB_read",
        "attribution_kind", *ATTR_COLS, *META_COLS]
COL_NOTES = [
    "median_s = the timed pass on LegClock (the headline). median_footer_read_s is the "
    "harness's up-front footer read (source.start() before the warmup), measured separately "
    "and never added to median_s; blank for nova-bf, which reads each file's "
    "schema inside its timed scan.",
    "attribution (README.md#attribution), harness systems only, two views that each add up to "
    "a leg's LegClock time: the HOST view (the stacked bar: io_wait | engine_call | merge | "
    "harness_other, the host thread's wall; a filtered leg's io_wait is the reader, read + "
    "filter eval) and the GPU view (gpu_engine, gpu_batching = our SPARe-batched / PyLate "
    "scores-batched wrapper, gpu_engine_and_batching when that split is not measurable, "
    "gpu_device_merge = device merge (cuVS recipe), cuVS's own running top-k merge on the GPU, "
    "blank for other systems; gpu_ours = ids + widen + normalize + other device work, gpu_idle). h2d_busy / d2h_busy / "
    "filter / filter_eval overlap the views: information, never stacked. Each column is a "
    "median over rounds, so the medians need not add up to median_s; each leg's own views do. "
    "nova-bf: its timed total only (its own timers are in the leg record's `internal`).",
]


def attribution_warnings(run_dir: str) -> list:
    """One warning per measurable attribution view of a usable harness leg that does not add
    up to its LegClock time (|residual| > leg.ADDITIVE_TOL_S, the tolerance of the LegClock vs
    total_s check). Reported, not hidden."""
    out = []
    for L in load_legs(run_dir):
        a = L.get("attribution") or {}
        if L.get("unreadable") or not L.get("ok") or a.get("kind") != "harness":
            continue
        for name in ("host_view", "gpu_view"):
            v = a.get(name) or {}
            if v.get("measurable") and v.get("additive") is not True:
                out.append(f"{L.get('tag')}: the attribution {name} does not add up to the "
                           f"LegClock time (sum {v.get('sum_s')} s, LegClock {v.get('clock_s')} s, "
                           f"residual {v.get('residual_s')} s; README.md#attribution)")
    return out


def write(run_dir: str, rows, problems, excluded, e2e) -> str:
    from bench import paths
    rows, problems, excluded, e2e = paths.mask((rows, problems, excluded, e2e))
    meta = summary_meta(run_dir)
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows({**r, **meta} for r in rows)
    def f(v, fmt=".4f"):
        return format(v, fmt) if v is not None else "-"
    lines = [f"run: commit {meta['run_commit'] or '-'}, dirty {meta['run_dirty']}, "
             f"code_sha {meta['code_sha'] or '-'}, stack_sha {meta['stack_sha'] or '-'}",
             f"experiment.json sha256 {meta['experiment_sha256'] or '-'}; checks ok: {meta['checks_ok']}",
             "",
             "| search | system | n | median s (timed pass) | median footer read s | per round | "
             "time source | preset (+proj: query-vocab projection; ours: our code) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['search']} | {r['system']} | {r['n']} | {f(r['median_s'])} | "
                     f"{f(r.get('median_footer_read_s'))} | {r['rounds_s']} | "
                     f"{r['time_source']} | {r.get('label') or r['preset']} |")
    att = [r for r in rows if r.get("attribution_kind") == "harness"]
    if att:
        lines += ["", "Where the time goes (harness systems; median seconds; host view | GPU "
                  "view | information):", "",
                  "| search | system | " + " | ".join(c[:-2] for c in ATTR_COLS) + " |",
                  "|---|---|" + "---|" * len(ATTR_COLS)]
        for r in att:
            lines.append(f"| {r['search']} | {r['system']} | "
                         + " | ".join(f(r.get(c)) for c in ATTR_COLS) + " |")
    se = score_error_lines(run_dir)
    if se:
        lines += ["", "Max own-score error vs float64 (float32 ULPs of the exact score, every "
                  "round's oracle sample + pairs adjudication; abs = absolute; adjudicated ids = "
                  "the ids the tie window judges; checks.json score_error):"] + se
    lines += [""] + [f"Note: {n}" for n in COL_NOTES]
    if e2e:
        lines += ["", "e2e (cold: one fresh process, no warmup, Triton + CUDA JIT caches cleared):"]
        for L in e2e:   # which number is which
            mt = L.get("e2e_manifest_timing") or {}
            el = mt.get("elapsed_seconds") if isinstance(mt, dict) else None
            lines.append(f"- {L['system']}: {'ok' if L.get('ok') else 'FAILED'}; "
                         f"LegClock (nova-bf's timed scan, the same clock as every leg) "
                         f"{f(L.get('timed_seconds'))} s; process wall (/usr/bin/time: the whole "
                         f"leg process, imports + JIT + result write) {f(L.get('process_wall_s'))} s; "
                         f"nova-bf manifest elapsed {f(el)} s (manifest timing {mt or '-'})")
    if excluded:
        lines += ["", "Excluded legs (in no median):"]
        lines += [f"- {s} {sy} r{r}: {why}" for (s, sy), r, why in excluded]
    if problems:
        lines += ["", f"PROBLEMS ({len(problems)}):"] + [f"- {p}" for p in problems]
    warnings = []
    try:   # round-to-round differences are listed, never a problem
        warnings = json.load(open(os.path.join(run_dir, "checks.json"))).get("warnings") or []
    except (OSError, ValueError):
        pass
    warnings = list(warnings) + attribution_warnings(run_dir)
    if warnings:
        lines += ["", f"Warnings ({len(warnings)}, not failures):"] + [f"- {w}" for w in warnings]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bench summarize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--rounds", type=int, default=None, help="default: the experiment's")
    ap.add_argument("--expect-files", type=int, default=None)
    ap.add_argument("--no-checks", action="store_true",
                    help="summarize without checks.json (otherwise a missing / failed one is a problem)")
    a = ap.parse_args(argv)
    if a.rounds is not None and a.rounds < 1:
        ap.error("--rounds: a positive integer")
    rows, problems, excluded, e2e = summarize(a.run_dir, rounds=a.rounds, expect_files=a.expect_files,
                                              require_checks=not a.no_checks)
    print(write(a.run_dir, rows, problems, excluded, e2e), end="")
    if problems:
        print(f"bench summarize: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    return 0
