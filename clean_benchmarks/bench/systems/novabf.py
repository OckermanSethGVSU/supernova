"""Run nova-bf as a benchmarked system.

Uses the same execution path and defaults as `nova-bf compute`, with timing
hooks connected to LegClock.

Runs an untimed warmup before the measured search. Supports limiting the
number of corpus files processed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

if __name__ == "__main__":
    # Run by path, sys.path[0] is this file's directory: drop it, so neither the stage's
    # top-level dirs nor bench/systems/*.py can shadow top-level modules in the nova-bf
    # process. bench itself is imported from its install.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

from bench.timing import ClockError, ClockHooks, LegClock  # noqa: E402

CB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stage_dirs_importable(root: str = CB_ROOT) -> list[str]:
    """The stage root's top-level dirs (other than bench) importable as top-level modules;
    must be [] in a leg. Uses find_spec only, so nothing is imported."""
    import importlib.util
    out = []
    for d in sorted(os.listdir(root)):
        if d == "bench" or not os.path.isdir(os.path.join(root, d)) or not d.isidentifier():
            continue
        try:
            spec = importlib.util.find_spec(d)
        except (ImportError, ValueError):
            continue
        locs = list(getattr(spec, "submodule_search_locations", None) or []) + [getattr(spec, "origin", None) or ""]
        if spec is not None and any(str(x).startswith(os.path.join(root, d)) for x in locs):
            out.append(d)
    return out


# LegClock vs nova-bf's own `search_seconds`. nova-bf's intervals lie inside the clock's
# running intervals, so clock - search_seconds is never negative (it is start-up, the gap
# before the first sort, and loop overhead). A multivector search's token sample (timed
# search work nova-bf does not count) is subtracted first. Allowed: [-1 ms, 0.5 s + 0.5 %].
CLOCK_AGREE_MIN_S = -0.001
CLOCK_AGREE_MAX_S = 0.5
CLOCK_AGREE_MAX_REL = 0.005


def clock_agreement(clock_seconds: float, novabf_timing: dict | None) -> dict:
    """{ok, diff_seconds, allowed: [lo, hi]}; ok=False also when nova-bf reported nothing."""
    if not novabf_timing:
        return {"ok": False, "diff_seconds": None, "reason": "no bf-bench-search record"}
    s = novabf_timing["search_seconds"]
    lo, hi = CLOCK_AGREE_MIN_S, CLOCK_AGREE_MAX_S + CLOCK_AGREE_MAX_REL * s
    d = clock_seconds - s
    return {"ok": lo <= d <= hi, "diff_seconds": d, "allowed": [lo, hi]}


# Override variables that are never applied (nova-bf runs at its defaults); reported as
# ignored if set.
IGNORED_LAUNCHER_VARS = ("NOVABF_RANGED_CONC", "NOVABF_TWOPASS_MIN_ROWS")


def report_stock_settings(err=None) -> None:
    """Print (never change) nova-bf's stock I/O and two-pass settings, read from nova-bf
    itself (imports nova_bf.io, then nova_bf.twopass)."""
    err = err or sys.stderr
    import nova_bf.io as _io

    pool = os.environ.get("NOVA_BF_FETCH_POOL_SIZE") or f"{_io._FETCH_POOL_SIZE_DEFAULT} (default)"
    print(f"novabf_launch: stock nova-bf, no overrides: ranged GET {_io._RANGED_GET_BYTES >> 20} MiB x "
          f"{_io._RANGED_GET_CONCURRENCY} per file (min file {_io._RANGED_GET_MIN_BYTES >> 20} MiB), "
          f"fetch pool {pool}", file=err, flush=True)

    import nova_bf.twopass as _tp

    print(f"novabf_launch: two-pass MIN_QUERY_ROWS = {_tp.MIN_QUERY_ROWS} (stock)", file=err, flush=True)
    for var in IGNORED_LAUNCHER_VARS:
        if var in os.environ:
            print(f"novabf_launch: {var}={os.environ[var]} IGNORED (stock settings)",
                  file=err, flush=True)


class _SearchSecondsTap(logging.Handler):
    """Capture nova-bf's own `bf-bench-search` numbers (unrounded) from the timed run's log,
    to cross-check LegClock. Observe only: other log handlers are unaffected."""

    # the timed run's `bf-bench io_workers=...` record args, unrounded (the printed line rounds
    # to 0.1 s). Kept in the leg record's `internal` as information only.
    BENCH_ARGS = ("io_workers", "cpu_threads", "files", "rows", "gb", "wall_s", "wall_mbps",
                  "stream_mbps", "io_wait_s", "gpu_s", "filter_s", "read_wall_s", "filter_wall_s")

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.values: dict | None = None
        self.bench: dict | None = None

    def emit(self, record):
        if not isinstance(record.msg, str) or not isinstance(record.args, tuple):
            return
        if record.msg.startswith("bf-bench-search ") and len(record.args) == 3:
            s, scan, sort = record.args
            self.values = {"search_seconds": float(s), "scan_seconds": float(scan),
                           "result_sort_seconds": float(sort)}
        elif record.msg.startswith("bf-bench io_workers=") and len(record.args) == len(self.BENCH_ARGS):
            self.bench = {k: float(v) for k, v in zip(self.BENCH_ARGS, record.args)}


def run(config_path: str, clock: LegClock | None = None, err=None, guard: dict | None = None,
        hooks_factory=ClockHooks) -> dict:
    """Warmup + timed run.

    `clock=None`: no timing hooks. With a clock, the timed run drives it
    through ClockHooks; an exception marks it failed and is re-raised, and a run that
    returns without calling done() fails it (ClockError).

    Returns {"results": {search: path}, "warmup_files": N, "max_files": N|None,
             "novabf_timing": {search_seconds, scan_seconds, result_sort_seconds} | None}."""
    err = err or sys.stderr
    report_stock_settings(err)

    warm = int(os.environ.get("NOVABF_WARMUP_FILES", "0"))
    warmed = 0
    if warm > 0:
        from nova_bf.cli import _setup_logging
        from nova_bf.compute import run_compute
        from nova_bf.config import load_config

        _setup_logging()
        cfg = load_config(config_path)
        wout = warmup_output(cfg.output.path)
        cfg = cfg.model_copy(update={"output": cfg.output.model_copy(update={"path": wout})})
        print(f"novabf_launch: warmup on {warm} file(s) into {wout} (untimed, discarded)",
              file=err, flush=True)
        t0 = time.perf_counter()
        run_compute(cfg, max_files=warm)   # no timing_hooks: the warmup never touches the clock
        warmed = warm
        print(f"novabf_launch: warmup done in {time.perf_counter() - t0:.1f} s; timed run next",
              file=err, flush=True)

    from nova_bf.cli import _setup_logging  # noqa: F811 - imported here even without a warmup

    max_files = int(os.environ.get("NOVABF_MAX_FILES", "0"))
    if max_files > 0:
        print(f"novabf_launch: timed run limited to the first {max_files} file(s)",
              file=err, flush=True)
    print(f"novabf_launch: TIMED_RUN_START warmup_files={warmed}", file=err, flush=True)

    # --- the timed run: what `nova-bf compute CONFIG [--max-files N]` does ------------
    _setup_logging()
    from nova_bf.compute import run_compute
    from nova_bf.config import load_config

    tap = _SearchSecondsTap()
    logging.getLogger("nova_bf.compute").addHandler(tap)
    cfg = load_config(config_path)
    bguard = None
    if guard is not None and clock is not None:   # the leg's boundary observer
        bguard = guard["guard"] = BoundaryGuard(
            clock, cfg.corpus.path,
            sample_pauses=sample_pauses(cfg),
        ).install()
    try:
        results = run_compute(
            cfg,
            num_jobs=None, job_rank=None, io_workers=None, io_thread_count=None,
            cpu_thread_count=None, max_files=max_files if max_files > 0 else None,
            timing_hooks=hooks_factory(clock) if clock is not None else None,
        )
    except BaseException as exc:
        if clock is not None:
            clock.fail(f"run_compute raised {type(exc).__name__}: {exc}")
        raise
    finally:
        logging.getLogger("nova_bf.compute").removeHandler(tap)
        if bguard is not None:
            bguard.uninstall()
    sample_s = None
    if clock is not None:
        if not clock.stopped:
            clock.fail(f"run_compute returned with the clock {clock.state} (no done())")
            raise ClockError(clock.failure)
        sample_s = multivector_sample_seconds(clock, cfg)
    return {"results": results, "warmup_files": warmed,  # results: {search: output path}
            "max_files": max_files if max_files > 0 else None, "novabf_timing": tap.values,
            "novabf_bench": tap.bench, "mv_sample_seconds": sample_s}


def native_textscan_status() -> dict:
    """nova-bf's native text scanner status, as nova-bf reports it (None reason = usable).

    Called only after the timed run: the first call runs the scanner's self-check, which a
    text-filter run must do itself inside its scan. `probed_by_run`: whether the run already
    probed it."""
    try:
        from nova_bf import nativetok
    except Exception as exc:  # noqa: BLE001  provenance never fails a leg
        return {"status": f"nova_bf.nativetok not importable: {type(exc).__name__}: {exc}",
                "probed_by_run": None}
    probed = bool(getattr(nativetok, "_PROBED", False))
    try:
        why = nativetok.unavailable_reason()
    except Exception as exc:  # noqa: BLE001
        why = f"unavailable_reason() raised {type(exc).__name__}: {exc}"
    return {"status": "IN USE" if why is None else why, "probed_by_run": probed}


class BoundaryGuard:
    """The boundary invariants, observed in the leg itself.

    Wraps nova-bf's six `Store` read entry points and its two write entry points, and tags
    each call with the LegClock's state when it starts. Observe only, and cheap: these are
    per-file calls, never per row.

    Violations:
      - a corpus read before corpus_start other than a footer / schema read (`read_schema`:
        file listing and footers are allowed);
      - a corpus data read after done, or while the clock is paused (a pause is the
        output write's; corpus data read then would be search work off the clock);
      - an output write before corpus_start, or while the clock runs with more writes started
        than search pauses so far. nova-bf submits each search's write inside its pause, and
        it may finish while a later search is timed, so `writes <= pauses` must hold. The
        multivector token sample's pause (`sample_pauses=1`) submits no write and is not
        counted.
    Corpus reads are told apart from query / output I/O by the config's corpus path prefix."""

    READS = ("read_columns", "read_columns_fetched", "read_dense_versioned", "read_dense",
             "read_schema", "_ranged_download")
    FOOTER_READS = ("read_schema",)
    WRITES = ("write", "write_bytes")

    def __init__(self, clock: LegClock, corpus_path: str, sample_pauses: int = 0):
        self.clock = clock
        self.sample_pauses = sample_pauses   # pauses that are not a search's (mv sample)
        self.corpus = self._norm(corpus_path).rstrip("/") + "/"
        self.reads: dict = {}        # (method, phase) -> count, corpus only
        self.writes: list = []       # (method, state, pauses at start)
        self.violations: list[str] = []
        self._orig: dict = {}

    @staticmethod
    def _norm(p) -> str:
        p = str(p)
        return p[5:] if p.startswith("s3://") else os.path.abspath(p) if not p.startswith("/") and "://" not in p else p

    def _read(self, name, orig):
        guard = self

        def w(store, *a, **kw):
            path = a[0] if a else kw.get("read_path")
            if path is not None and guard._norm(path).startswith(guard.corpus):
                st = guard.clock.state
                key = (name, st)
                guard.reads[key] = guard.reads.get(key, 0) + 1
                if st == "new" and name not in guard.FOOTER_READS:
                    guard.violations.append(f"corpus data read before corpus_start: {name}({path})")
                elif st == "stopped" and name not in guard.FOOTER_READS:
                    guard.violations.append(f"corpus data read after done: {name}({path})")
                elif st == "paused" and name not in guard.FOOTER_READS:
                    guard.violations.append(f"corpus data read while the clock is paused: "
                                            f"{name}({path})")
            return orig(store, *a, **kw)
        return w

    def _write(self, name, orig):
        guard = self

        def w(store, *a, **kw):
            st = guard.clock.state
            pauses = max(0, guard.clock.pauses - guard.sample_pauses)   # search pauses only
            guard.writes.append((name, st, pauses))
            n_running = sum(1 for _, s, _ in guard.writes if s != "stopped")
            if st == "new":
                guard.violations.append(f"output write before corpus_start: {name}({a[:1]})")
            elif st == "running" and n_running > pauses:
                guard.violations.append(
                    f"output write while timed with no pause to account for it: {name}({a[:1]}) "
                    f"(write #{n_running}, {pauses} pause(s) so far)")
            return orig(store, *a, **kw)
        return w

    def install(self):
        from nova_bf.io import Store
        for name in self.READS + self.WRITES:
            if hasattr(Store, name):
                orig = getattr(Store, name)
                self._orig[name] = orig
                setattr(Store, name, (self._read if name in self.READS else self._write)(name, orig))
        return self

    def uninstall(self):
        from nova_bf.io import Store
        for name, orig in self._orig.items():
            setattr(Store, name, orig)
        self._orig = {}

    def record(self) -> dict:
        return {"violations": list(self.violations),
                "corpus_reads": {f"{m}@{s}": n for (m, s), n in sorted(self.reads.items())},
                "writes": [{"method": m, "state": s, "pauses": p} for m, s, p in self.writes]}


def warmup_output(path: str) -> str:
    """`<output.path>_warmup`; an S3 `?region=...` suffix stays last."""
    p, sep, query = path.partition("?")
    return p.rstrip("/") + "_warmup" + (sep + query if sep else "")


def sample_pauses(cfg) -> int:
    """Clock pauses that are not a search's: 1 with a multivector search (the token sample), else 0."""
    return 1 if any(s.vector_type == "multivector" for s in cfg.searches) else 0


def multivector_sample_seconds(clock: LegClock, cfg) -> float | None:
    """Seconds of the multivector token sample, the clock's first segment (None without one).

    It is search work (it reads corpus data) that nova-bf's search_seconds leaves out, so the
    cross-check subtracts it. Refuses a clock without exactly S+1 pauses (S without a sample)."""
    n = len(cfg.searches)
    has_mv = any(s.vector_type == "multivector" for s in cfg.searches)
    want = n + 1 if has_mv else n
    if clock.pauses != want:
        clock_state = clock.record()["events"]
        raise ClockError(f"expected {want} pauses for {n} search(es) (multivector={has_mv}), "
                         f"got {clock_state}")
    return clock.segments()[0] if has_mv else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench.systems.novabf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["compute"])
    ap.add_argument("config")
    ap.add_argument("--hooks", choices=["off", "clock"], default="clock")
    ap.add_argument("--clock-json", default=None,
                    help="write the clock record + nova-bf's own timing here (JSON)")
    ap.add_argument("--boundary", action="store_true",
                    help="observe the boundary invariants (BoundaryGuard; bench leg sets it)")
    a = ap.parse_args(argv)
    # bench must come from this file's tree and nova_bf from the hashed nova-bf root
    # (checked against this file's path, not sys.path); refuse before anything runs
    from bench.record import provenance
    orig = provenance.check(CB_ROOT, need_novabf=True)
    clock = LegClock() if a.hooks == "clock" else None
    rec: dict = {"hooks": a.hooks, "argv0": sys.argv[0], "stage_dirs_importable": stage_dirs_importable(),
                 "import_origins": orig}
    if not orig["ok"]:
        rec.update(ok=False, error="REFUSED: " + "; ".join(orig["problems"]))
        if a.clock_json:
            with open(a.clock_json, "w") as f:
                json.dump(rec, f, indent=1, sort_keys=True)
        provenance.refuse_unless_ok(orig, "bench.systems.novabf")
    holder: dict | None = {} if a.boundary else None
    try:
        out = run(a.config, clock, guard=holder)
        rec.update(ok=True, novabf_timing=out["novabf_timing"], novabf_bench=out["novabf_bench"],
                   warmup_files=out["warmup_files"],
                   mv_sample_seconds=out["mv_sample_seconds"], max_files=out["max_files"],
                   results={k: str(v) for k, v in (out["results"] or {}).items()})
    except BaseException as exc:
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        rec["native_textscan"] = native_textscan_status()   # after the run: observe only
        if holder and "guard" in holder:
            rec["boundary"] = holder["guard"].record()
        if clock is not None:
            rec["clock"] = clock.record()
            if clock.stopped and rec.get("ok"):
                rec["clock_agreement"] = clock_agreement(
                    clock.seconds - (rec.get("mv_sample_seconds") or 0.0), rec.get("novabf_timing"))
        if a.clock_json:
            with open(a.clock_json, "w") as f:
                json.dump(rec, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
