"""Run standalone correctness checks for benchmark engines.

check-copies runs every GPU harness system of the experiment with serial and with
overlapped host<->GPU copies, requiring identical top-k results.

check-verify runs every harness system (every mode of SPARe / PyLate) against the
harness's float64 oracle, including filter compliance for filtered search.

Both checks run untimed in fresh processes, independently of bench run,
and fail if any required check fails. Files and passes come from the flags, else the
experiment's `checks:` (copies_files / verify_files / check_repeats), else its
`subset:`, else the defaults.
"""


from __future__ import annotations

import copy
import json
import os
import re
import sys

from bench import experiment as experiment_mod
from bench import leg as leg_mod
from bench.systems import base as systems_base
from bench.verify import agree

DEFAULT_FILES = 4
DEFAULT_REPEATS = 1
# This is the check: the same pinned copies run serial (no overlap either way) and overlapped
# (both ways) must give the identical top-k. Every GPU harness engine takes both overlaps.
SERIAL = {"h2d": "pinned", "h2d_overlap": False, "d2h": "pinned", "d2h_overlap": False}
OVERLAP = {**SERIAL, "h2d_overlap": True, "d2h_overlap": True}
VERIFY_OK = re.compile(r"^verify OK", re.M)

EPILOG = """\
typical settings:
  check-copies  dense: --files 16 --warmup 1 --warmup-files 1 --repeats 2
                sparse: --files 16 --warmup 1 --warmup-files 1; multivector: --files 8 (same warmup)
  check-verify  sparse, multivector: --files 2; filtered: --files 4 --warmup 1 --repeats 2
"""


class StepError(Exception):
    pass


def settings(exp: dict, kind: str, files=None, repeats=None, warmup=0, warmup_files=0) -> dict:
    """Files and passes of a check: flag > experiment checks: > subset: (files) > default."""
    ch = exp.get("checks") or {}
    sub = (exp.get("subset") or {}).get("files")
    files = next(v for v in (files, ch.get(f"{kind}_files"), sub, DEFAULT_FILES) if v is not None)
    if repeats is None:
        repeats = ch.get("check_repeats", DEFAULT_REPEATS)
    return {"files": files, "repeats": repeats, "warmup": warmup, "warmup_files": warmup_files}


def _proto(cfg):
    out = ["--warmup", str(cfg["warmup"]), "--repeats", str(cfg["repeats"])]
    return out + (["--warmup-files", str(cfg["warmup_files"])] if cfg["warmup_files"] else [])


def _strip_tail(argv: list[str]) -> list[str]:
    """harness_args' leg tail (--save-results / --clock-json / --csv / protocol) removed."""
    return argv[: argv.index("--save-results")]


def _ordered(exp):
    """(name, system entry) in the order the searches list them (round-1 leg order);
    experiment.json itself keeps systems sorted."""
    names = list(dict.fromkeys(sy for s in exp["searches"] for sy in s["systems"]))
    return [(n, exp["systems"][n]) for n in names]


def _search_of(exp, sysname):
    for s in exp["searches"]:
        if sysname in s["systems"]:
            return s
    return exp["searches"][0]


def _gpu_harness(sy):
    spec = systems_base.SYSTEMS[sy["system"]]
    return (spec.kind == "harness" and spec.backend not in systems_base.HOST_BACKENDS
            and set(systems_base.COPY_KNOBS) <= set(spec.knobs))


def _run(exp, run_dir, cfg, s, name, knobs, tag, **extra):
    """One check run: the leg's harness args for SYSTEM with KNOBS, on cfg's files and passes."""
    e = dict(exp, subset={"files": cfg["files"]})
    sysr = dict(copy.deepcopy(exp["systems"][name]), knobs=knobs)
    tail = _strip_tail(leg_mod.harness_args(e, s, sysr, run_dir, tag)) + _proto(cfg)
    return {"name": tag, "system": name, "search": s["name"], "argv_tail": tail, **extra}


def plan_copies(exp: dict, run_dir: str, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or settings(exp, "copies")
    runs = []
    for name, sy in _ordered(exp):
        if not _gpu_harness(sy):
            continue
        for mode, kn in (("serial", SERIAL), ("overlap", OVERLAP)):
            tag = f"{name}_{mode}"
            npz = os.path.join(run_dir, "legs", f"{tag}.npz")
            r = _run(exp, run_dir, cfg, _search_of(exp, name), name, {**sy["knobs"], **kn}, tag,
                     mode=mode, topk=npz)
            r["argv_tail"] += ["--save-results", npz, "--csv", os.path.join(run_dir, "legs", f"{tag}.csv")]
            runs.append(r)
    if not runs:
        raise StepError("no GPU harness system in the experiment")
    return runs


def plan_verify(exp: dict, run_dir: str, cfg: dict | None = None) -> list[dict]:
    """Every harness system once, every mode of a system with modes once per engine; per
    search for filtered (each has its own filter), else on the first search."""
    cfg = cfg or settings(exp, "verify")
    filtered = exp["modality"] == "filtered"
    runs, done = [], set()
    for s in (exp["searches"] if filtered else exp["searches"][:1]):
        names = s["systems"] if filtered else [n for n, _ in _ordered(exp)]
        for name in names:
            sy = exp["systems"][name]
            if sy["kind"] == "novabf":
                continue
            for mode, kn in systems_base.mode_variants(systems_base.SYSTEMS[sy["system"]], sy["knobs"]):
                if mode is not None and (s["name"], sy["system"], mode) in done:
                    continue
                done.add((s["name"], sy["system"], mode))
                tag = name if mode is None else f"{sy['system']}_{mode}"
                tag = f"{s['name']}__{tag}" if filtered else tag
                r = _run(exp, run_dir, cfg, s, name, kn, tag, mode=mode)
                r["argv_tail"] += ["--verify", "--csv", os.path.join(run_dir, "legs", f"{tag}.csv")]
                runs.append(r)
    if not runs:
        raise StepError("no harness system in the experiment")
    return runs


def _spec(exp, run_dir, r, dev):
    """A LegSpec for one check run: the leg runner's own command and env for that system."""
    base = leg_mod.plan(exp, r["search"], r["system"], 1, run_dir, dev=dev)
    argv = base["argv"][:2] + r["argv_tail"]
    return leg_mod.LegSpec(base, argv=argv, log=os.path.join(run_dir, "logs", f"{r['name']}.log"))


def _plan(kind, exp, run_dir, cfg):
    return (plan_copies if kind == "copies" else plan_verify)(exp, run_dir, cfg)


def execute(kind: str, exp: dict, run_dir: str, dev: bool, cfg: dict | None = None) -> dict:
    cfg = cfg or settings(exp, kind)
    runs = _plan(kind, exp, run_dir, cfg)
    doc = {"kind": kind, "experiment": exp["name"], "dev": dev, "settings": cfg,
           "variants": [{k: r[k] for k in ("name", "system", "mode", "search")} for r in runs],
           "runs": {}, "pairs": {}, "ok": True}
    if kind == "copies":
        doc["copies"] = {"serial": SERIAL, "overlap": OVERLAP}
    for r in runs:
        spec = _spec(exp, run_dir, r, dev)
        res = leg_mod.spawn(spec)
        txt = open(spec["log"], errors="replace").read()
        ok = res["exit_code"] == 0
        entry = {"system": r["system"], "search": r["search"], "mode": r["mode"], "argv": spec["argv"],
                 "exit_code": res["exit_code"], "log": os.path.relpath(spec["log"], run_dir)}
        if kind == "verify":
            entry["verify_lines"] = [ln for ln in txt.splitlines() if re.search(r"verify (OK|FAIL)", ln)
                                     or ln.startswith("VERIFY FAILED")]
            ok = ok and bool(VERIFY_OK.search(txt))
        entry["ok"] = ok
        doc["runs"][r["name"]] = entry
        doc["ok"] = doc["ok"] and ok
    if kind == "copies":
        for name in dict.fromkeys(r["system"] for r in runs):
            a, b = (os.path.join(run_dir, "legs", f"{name}_{m}.npz") for m in ("serial", "overlap"))
            try:
                same = agree.same_topk(a, b)
            except (OSError, KeyError, ValueError):
                same = False
            doc["pairs"][name] = "IDENTICAL" if same else "DIFFERENT"
            doc["ok"] = doc["ok"] and same
    return doc


def _count(lo):
    def parse(v):
        n = int(v)
        if n < lo:
            raise ValueError(v)
        return n
    parse.__name__ = f"integer >= {lo}"
    return parse


def main(kind: str, argv=None) -> int:
    import argparse
    from bench import runner
    ap = argparse.ArgumentParser(prog=f"bench check-{kind}", description=__doc__, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--dev", action="store_true", help="CPU dev box")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--results", default=runner.RESULTS)
    ap.add_argument("--files", type=_count(1), default=None,
                    help=f"corpus files (default: checks.{kind}_files, subset.files, {DEFAULT_FILES})")
    ap.add_argument("--repeats", type=_count(1), default=None,
                    help=f"timed passes (default: checks.check_repeats, {DEFAULT_REPEATS})")
    ap.add_argument("--warmup", type=_count(0), default=0, help="warmup passes (default 0)")
    ap.add_argument("--warmup-files", type=_count(0), default=0,
                    help="files a warmup pass reads (default 0 = all)")
    ap.add_argument("--plan", action="store_true", help="print the planned harness argv, run nothing")
    a = ap.parse_args(argv)
    if a.warmup_files and not a.warmup:
        ap.error("--warmup-files needs --warmup >= 1")
    try:
        raw = runner._set(experiment_mod.load_raw(a.experiment), a.set)
        exp = experiment_mod.finish(raw, a.experiment)
    except experiment_mod.ExperimentError as exc:
        print(f"bench check-{kind}: {exc}", file=sys.stderr)
        return 2
    ph = experiment_mod.placeholders(exp)
    if ph:
        print(f"bench check-{kind}: placeholder value(s) (TODO): " + "; ".join(ph), file=sys.stderr)
        return 2
    if a.dev:
        bad = runner.dev_refusals(exp)
        if bad:
            print(f"bench check-{kind} --dev: " + "; ".join(bad), file=sys.stderr)
            return 2
    cfg = settings(exp, kind, a.files, a.repeats, a.warmup, a.warmup_files)
    run_dir = runner.new_run_dir(os.path.join(a.results, exp["name"], f"check-{kind}"))
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        f.write(experiment_mod.dumps(exp))
    exp = json.load(open(os.path.join(run_dir, "experiment.json")))
    try:
        if a.plan:
            for r in _plan(kind, exp, run_dir, cfg):
                print(r["name"], " ".join(_spec(exp, run_dir, r, a.dev)["argv"]))
            return 0
        doc = execute(kind, exp, run_dir, a.dev, cfg)
    except StepError as exc:
        print(f"bench check-{kind}: {exc}", file=sys.stderr)
        return 2
    with open(os.path.join(run_dir, "check.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    for n, r in doc["runs"].items():
        print(f"{n}: exit {r['exit_code']} {'OK' if r['ok'] else 'FAIL'}"
              + (f"  {r['verify_lines'][-1][:160]}" if r.get("verify_lines") else ""))
    for n, v in doc["pairs"].items():
        print(f"{n}: serial vs overlap {v} top-k")
    print(f"bench check-{kind}: {'PASS' if doc['ok'] else 'FAIL'} ({run_dir})")
    return 0 if doc["ok"] else 1
