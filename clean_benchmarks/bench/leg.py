"""Execute a single system and search for one benchmark round.

Each leg runs in a fresh engine process with a warmup followed by a timed
pass. The supervisor handles execution and validation but does not time
the search itself.

Failed legs record their errors without reporting a valid time.
"""

from __future__ import annotations

import csv
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

from bench import experiment, paths, timing
from bench.systems import base as systems_base

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # clean_benchmarks
HARNESS_CLI = os.path.join(CB, "bench", "systems", "harness_cli.py")
NOVABF_ADAPTER = os.path.join(CB, "bench", "systems", "novabf.py")
RECORD_SCHEMA = 1

# nova-bf config env variables the leg sets, and where each value comes from. Each is set only
# when the config references it (`${NAME`), so a leg's env holds only the variables its config uses.
NOVABF_CONFIG_ENV = {
    "NOVABF_CORPUS": lambda e, k, io: e["corpus"]["path"],
    "NOVABF_QUERIES": lambda e, k, io: e["queries"]["path"],
    "NOVABF_VCOL": lambda e, k, io: e["corpus"]["column"],
    "NOVABF_QCOL": lambda e, k, io: e["queries"]["column"],
    "NOVABF_IO_WORKERS": lambda e, k, io: io and io["files_in_flight"],
    "NOVABF_RUST_FETCH_CONC": lambda e, k, io: io and io["gets"],
    "NOVABF_RUST_FETCH_PART_MIB": lambda e, k, io: io and io["part_mib"],
    "NOVABF_MV_BUDGET_MIB": lambda e, k, io: k.get("mv_budget_mib"),
}


def leg_io(exp: dict, sysr: dict) -> dict | None:
    """The `io:` a leg uses: the experiment's, overlaid with a sweep variant's own `io` keys;
    None when the experiment has none."""
    if not exp["io"]:
        return None
    return {**exp["io"], **(sysr.get("io") or {})}


def _x(p):
    """A path string as the leg uses it: private-path placeholders filled in, ~ expanded;
    anything else is returned as is."""
    return os.path.expanduser(paths.expand(p)) if isinstance(p, str) else p


class LegSpec(dict):
    """What a leg runs: tag, kind, argv (engine, without the time wrapper), env_add, cwd,
    log path, expected outputs."""


# ---------------------------------------------------------------------------------------------
# 1. the engine command
# ---------------------------------------------------------------------------------------------
def harness_args(exp: dict, search: dict, sysr: dict, run_dir: str, tg: str) -> list[str]:
    """The harness argv after the script (bench/systems/harness_cli.py), in order: the shared
    args, the subset, the copy modes, the system's args, then the leg's outputs and the protocol
    flags last."""
    spec = systems_base.SYSTEMS[sysr["system"]]
    kn = sysr["knobs"]
    c, q, s = exp["corpus"], exp["queries"], exp["search"]
    url = _x(c["path"]) + (f"?region={c['region']}" if c.get("region") else "")
    a = ["--regime", "stream", "--corpus", url]
    if c.get("recursive"):
        a.append("--corpus-recursive")
    if exp["modality"] in ("sparse", "multivector"):
        a += ["--modality", exp["modality"]]
    a += ["--column", c["column"], "--query-file", _x(q["path"]), "--query-column", q["column"]]
    if q.get("nq") is not None:
        a += ["--nq", str(q["nq"])]
    a += ["--k", str(s["k"]), "--metric", s["metric"]]
    if exp["modality"] == "filtered":
        a += ["--filter-config", filter_config_path(exp, run_dir),
              "--filter-search", search["filter_search"]]
    io = leg_io(exp, sysr)
    if io:
        a += ["--read-mode", kn.get("read_mode", io["read_mode"]),
              "--range-concurrency", str(io["gets"]), "--range-scope", io["range_scope"],
              "--range-mib", str(io["part_mib"]),
              "--prefetch", str(kn.get("prefetch", io["files_in_flight"])),
              "--workers", str(kn.get("workers", io["files_in_flight"]))]
    else:
        for k in ("read_mode", "prefetch", "workers"):
            a += systems_base.harness_knob_args(spec, kn, [k])
    a += ["--omp-threads", str(timing.omp_threads(exp))]
    if exp.get("subset"):
        a += ["--max-files", str(exp["subset"]["files"])]
    copies = experiment.leg_copies(exp.get("copies", {}), sysr)
    a += systems_base.harness_knob_args(spec, copies, list(systems_base.COPY_KNOBS))
    a += ["--backend", spec.backend]
    own = [k for k in spec.knobs if k not in systems_base.COPY_KNOBS and k not in systems_base.IO_KNOBS]
    a += systems_base.harness_knob_args(spec, kn, own)
    for flag in a:
        if flag in timing.HARNESS_RESERVED and flag != "--max-files":
            raise ValueError(f"{tg}: {flag} is fixed by the timing protocol")
    a += ["--save-results", os.path.join(run_dir, "legs", f"{tg}.topk.npz"),
          "--clock-json", os.path.join(run_dir, "legs", f"{tg}.clock.json"),
          "--csv", os.path.join(run_dir, "legs", f"{tg}.csv"), *timing.HARNESS_PROTOCOL_ARGS]
    return a


def filter_config_path(exp: dict, run_dir: str) -> str:
    """The harness's --filter-config: the experiment's filter config with corpus.path set to
    the experiment's corpus, written once per run."""
    out = os.path.join(run_dir, "configs", "filter.yaml")
    if not os.path.exists(out):
        import yaml
        c = yaml.safe_load(open(exp["filter"]["config"]))
        c["corpus"]["path"] = paths.expand(exp["corpus"]["path"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(c, f)
    return out


def novabf_output(exp: dict, run_dir: str, tg: str) -> str:
    """Where a nova-bf leg writes: <novabf_output>/<run id>/<tag> (default <run>/novabf/<tag>).
    An S3 output that names no region gets `?region=<corpus.region>`, like the corpus URL the
    harness reads: pyarrow and nova-bf's Store strip the query before joining paths, and their
    native reader takes the region from it."""
    base = paths.expand(exp.get("novabf_output"))
    if not base:
        return os.path.join(run_dir, "novabf", tg)
    path, _, query = base.partition("?")
    out = f"{path.rstrip('/')}/{os.path.basename(run_dir)}/{tg}"
    region = (exp.get("corpus") or {}).get("region")
    if path.startswith("s3://") and region and "region=" not in query:
        query = f"{query}&region={region}" if query else f"region={region}"
    return f"{out}?{query}" if query else out


def novabf_corpus_url(exp: dict) -> str:
    """corpus.path as a generated nova-bf config takes it: placeholders filled in, and an S3
    path that names no region gets `?region=<corpus.region>` (pyarrow and nova-bf's native
    reader take the region from it). nova-bf always lists a corpus directory recursively."""
    c = exp["corpus"]
    path = paths.expand(c["path"])
    base, _, query = path.partition("?")
    if base.startswith("s3://") and c.get("region") and "region=" not in query:
        query = f"{query}&region={c['region']}" if query else f"region={c['region']}"
    return f"{base}?{query}" if query else base


def novabf_config(exp: dict, search: dict | None, sysr: dict, run_dir: str, tg: str, out: str,
                  e2e: bool = False) -> str:
    """The config the leg runs: the experiment's config file as is (its ${NOVABF_*} filled from
    the env), unless the leg must change it. A filtered experiment runs one search per config,
    with corpus and output paths set; a `params` knob is merged into params. Generated configs
    go to <run>/configs/<tag>.yaml."""
    src = sysr["knobs"]["config"]
    params = sysr["knobs"].get("params")
    # the cold e2e leg may order ties by a corpus field instead of corpus position
    tb = (exp.get("e2e") or {}).get("tiebreak") or {} if e2e else {}
    by_id = tb.get("mode") == "id"
    if exp["modality"] != "filtered" and not params and not by_id:
        return src
    import yaml
    c = yaml.safe_load(open(src))
    if exp["modality"] == "filtered" and search is not None:
        c["searches"] = [x for x in c["searches"] if x["name"] == search["filter_search"]]
        if len(c["searches"]) != 1:
            raise ValueError(f"{src}: no single search named {search['filter_search']!r}")
        c["corpus"]["path"] = novabf_corpus_url(exp)
        c["output"]["path"] = out
    if params:
        c["params"] = {**(c.get("params") or {}), **params}
    if by_id:
        c["params"] = {**(c.get("params") or {}), "tiebreak": "id"}
        c["corpus"]["id_column"] = tb["field"]
    p = os.path.join(run_dir, "configs", f"{tg}.yaml")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(c, f)
    return p


def plan(exp: dict, search_name: str | None, system: str, rnd: int | None, run_dir: str,
         dev: bool = False, e2e: bool = False, engine_python: str | None = None) -> LegSpec:
    """The engine command of one leg. Its only side effect is writing generated configs."""
    sysr = exp["systems"][system]
    search = next((s for s in exp["searches"] if s["name"] == search_name), None)
    if search is None and not e2e:
        raise ValueError(f"no search {search_name!r}")
    tg = timing.e2e_tag(system) if e2e else timing.tag(search_name, system, rnd)
    py = engine_python or os.environ.get("BENCH_ENGINE_PYTHON") or sys.executable
    env_add = timing.leg_env(exp)
    legs = os.path.join(run_dir, "legs")
    if sysr["kind"] == "harness":
        argv = [py, HARNESS_CLI, *harness_args(exp, search, sysr, run_dir, tg)]
        log = os.path.join(run_dir, "logs", f"{tg}.log")
        outputs = {"csv": os.path.join(legs, f"{tg}.csv"),
                   "clock": os.path.join(legs, f"{tg}.clock.json"),
                   "topk": os.path.join(legs, f"{tg}.topk.npz")}
    else:
        out = novabf_output(exp, run_dir, tg)
        cfg = novabf_config(exp, search, sysr, run_dir, tg, out, e2e=e2e)
        text = open(cfg).read()
        env_add["NOVA_BF_DEVICE"] = timing.DEV_NOVABF_DEVICE if dev else timing.NOVABF_DEVICE
        io = leg_io(exp, sysr)
        for var, get in NOVABF_CONFIG_ENV.items():
            if "${" + var in text:
                v = get(exp, sysr["knobs"], io)
                if v is not None:
                    env_add[var] = systems_base._fmt(_x(v))
        if "${NOVABF_OUT" in text:
            env_add["NOVABF_OUT"] = out
        if exp.get("subset"):
            env_add["NOVABF_MAX_FILES"] = str(exp["subset"]["files"])
        if not e2e:
            env_add["NOVABF_WARMUP_FILES"] = str(timing.WARMUP_FILES)
        # run by path, so the cwd is not on sys.path
        argv = [py, NOVABF_ADAPTER, "compute", cfg, "--hooks", "clock",
                "--clock-json", os.path.join(legs, f"{tg}.clock.json"), "--boundary"]
        log = os.path.join(run_dir, "logs", f"{tg}.novabf.log")
        outputs = {"clock": os.path.join(legs, f"{tg}.clock.json"), "novabf_out": out,
                   "config": cfg, "topk": os.path.join(legs, f"{tg}.topk.npz")}
    return LegSpec(tag=tg, kind=sysr["kind"], system=system, search=search_name, round=rnd,
                   e2e=e2e, argv=argv, env_add=env_add, cwd=CB, log=log, outputs=outputs,
                   knobs=sysr["knobs"], preset=sysr["preset"], system_kind=sysr["system"],
                   chosen_from=sysr.get("chosen_from"))


# ---------------------------------------------------------------------------------------------
# 2. run it
# ---------------------------------------------------------------------------------------------
def spawn(spec: LegSpec, base_env: dict | None = None) -> dict:
    """Run the engine command under /usr/bin/time -v, output to the leg's log; nothing timed."""
    env = dict(os.environ if base_env is None else base_env)
    env.pop("BENCH_ENGINE_PYTHON", None)   # a test hook of plan(); never part of a leg's env
    env.update(spec["env_add"])
    os.makedirs(os.path.dirname(spec["log"]), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(spec["log"])), "legs"), exist_ok=True)
    cmd = timing.timev_prefix() + list(spec["argv"])
    t0 = time.time()
    with open(spec["log"], "wb") as lg:
        p = subprocess.run(cmd, env=env, cwd=spec["cwd"], stdout=lg, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"exit_code": p.returncode, "wrapper": cmd[: len(cmd) - len(spec["argv"])],
            "started_unix": t0, "ended_unix": time.time(),
            "children_maxrss_bytes": ru.ru_maxrss * 1024}


GPU_APPS_QUERY = ["--query-compute-apps=pid,process_name", "--format=csv,noheader"]


GPU_FREE_TRIES = 6      # a process that just exited can stay listed for a moment:
GPU_FREE_WAIT_S = 1.0   # re-check up to 5 s before refusing a leg


def gpu_foreign_procs(spec: LegSpec, dev: bool, tries: int = GPU_FREE_TRIES,
                      wait_s: float = GPU_FREE_WAIT_S) -> tuple:
    """The other processes on the GPU, checked before a GPU leg's process starts (so never
    inside its timed region). Returns (record value, refusal or None):
      []                    nvidia-smi lists no compute process: the leg runs
      [{pid, name}, ...]    it lists some: the leg is refused
      "not checked"         a GPU leg on a node without nvidia-smi: the leg runs
      "n/a"                 a --dev leg or a CPU-backend leg (no GPU used): not checked
    Any non-empty line is a process; nothing is ignored. A listed process is re-checked a few
    times (up to ~5 s) first, since one that just exited can still show. An nvidia-smi that
    fails refuses the leg too: a GPU leg never runs unchecked when the check is available."""
    if dev or spec["system_kind"] in systems_base.HOST_BACKENDS:
        return "n/a", None
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return "not checked", None
    for i in range(max(1, tries)):
        rec, why = _gpu_procs_once(exe)
        if not isinstance(rec, list) or not rec:   # free, or the check itself failed
            return rec, why
        if i + 1 < tries:
            time.sleep(wait_s)
    return rec, why


def _gpu_procs_once(exe: str) -> tuple:
    """One `nvidia-smi` query: (record value, refusal or None), as gpu_foreign_procs."""
    try:
        p = subprocess.run([exe, *GPU_APPS_QUERY], capture_output=True, text=True, timeout=60,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        return "check failed", f"refused: could not list the GPU's processes ({type(exc).__name__}: {exc})"
    if p.returncode != 0:
        why = (p.stderr or p.stdout or "").strip()[:200]
        return "check failed", (f"refused: `nvidia-smi {' '.join(GPU_APPS_QUERY)}` exited "
                                f"{p.returncode}" + (f" ({why})" if why else ""))
    procs = []
    for line in p.stdout.splitlines():
        if line.strip():
            pid, _, name = line.partition(",")
            pid = pid.strip()
            procs.append({"pid": int(pid) if pid.isdigit() else pid, "name": name.strip()})
    if procs:
        return procs, ("refused: the GPU is in use by another process: "
                       + ", ".join(f"pid {x['pid']} ({x['name'] or '?'})" for x in procs))
    return [], None


# ---------------------------------------------------------------------------------------------
# 3. check + record
# ---------------------------------------------------------------------------------------------
RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
EXIT = re.compile(r"^\s*Exit status: (-?\d+)\s*$", re.M)
MARK = re.compile(r"novabf_launch: TIMED_RUN_START warmup_files=(\d+)")
BENCH = re.compile(r"bf-bench io_workers=\d+ .*?files=(\d+) .*?gb=([\d.]+) wall_s=([\d.]+) "
                   r".*?io_wait_s=([\d.]+) gpu_s=([\d.]+)")
SEARCH = re.compile(r"bf-bench-search search_s=([\d.]+) scan_s=([\d.]+) result_sort_s=([\d.]+)")
WORKER_FILES = re.compile(r"corpus files: \d+ total, (\d+) for this worker")
MAX_FILES = re.compile(r"--max-files=(\d+): benchmarking on the first")
FALLBACK = re.compile(r"fell back to pyarrow|using pyarrow for the rest")


def _f(v):
    """float(v), or None when v is blank or not a number."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Allowed gap between LegClock and the harness's own total_s. The clock starts right after the
# pass's t_start stamp and stops right after its t_end stamp (bench/systems/loop.py run_pass),
# so the two differ by two perf_counter calls; more means they did not time the same span.
HARNESS_CLOCK_TOL_S = 0.01


def _json_or_fail(path, failures):
    """Load a JSON file; None if absent; an unreadable file is also added to ``failures``."""
    if not path or not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except ValueError as exc:
        failures.append(f"unreadable {os.path.basename(path)}: {exc}")
        return None


# Harness CSV counters that must be 0 in a reported run: nonzero in either pass fails the leg.
MUST_BE_ZERO = {
    "cuvs_filter_small_k_calls": "a cuVS prefilter call with k <= 256 can meet raft's CSR "
                                 "select_k short-row bug (baselines/README.md)",
    "filter_mask_overflows": "a mask sent outside the reader's ring slot (bench/data/filters/reader.py)",
}
HARNESS_EVENT_COLS = ("h2d_overlapped", "sync_waits")   # the EVENT_FIELDS not named *_s
# the mode columns the attribution reads: batching part, device widen, overlap, cuVS device merge
HARNESS_MODE_COLS = ("spare_mode", "spare_events", "pylate_mode", "widen_where",
                     "h2d_mode", "d2h_mode", "cuvs_merge", "cuvs_sync")
OVERLAP_MODE = "pinned-overlap"   # harness_cli h2d_mode / d2h_mode for --h2d- / --d2h-overlap
TRACE_TOL_S = 1e-6   # the trace's stamps and total_s are the same perf_counter's, written in full


def harness_boundary(spec: LegSpec, total_s, failures: list) -> dict:
    """Check that the timed pass's trace rows (legs/<tag>.trace.csv, warmup 0) lie inside the
    pass: no reader started a file before t_start (w_start_s < 0 means corpus bytes were
    requested before the clock), and no row ends after total_s. Stamps are relative to the
    pass start (loop.trace_rows)."""
    p = spec["outputs"]["csv"][:-4] + ".trace.csv"
    b = {"trace": os.path.basename(p), "rows": 0, "min_start_s": None, "max_end_s": None,
         "total_s": total_s, "violations": []}
    if not os.path.exists(p):
        b["violations"].append("no trace CSV")
    else:
        starts, ends = [], []
        for r in csv.DictReader(open(p)):
            if str(r.get("warmup")) not in ("0", "False", "false"):
                continue
            b["rows"] += 1
            for k in ("w_start_s", "w_filter_start_s"):
                v = _f(r.get(k))
                if v is not None:
                    starts.append(v)
                    if v < 0:
                        b["violations"].append(f"file {r.get('file_index')}: {k}={v} < 0 "
                                               "(read before the timed pass started)")
            for k in ("t_done_s", "w_end_s", "w_publish_end_s", "w_filter_end_s"):
                v = _f(r.get(k))
                if v is not None:
                    ends.append(v)
                    if total_s is not None and v > total_s + TRACE_TOL_S:
                        b["violations"].append(f"file {r.get('file_index')}: {k}={v} after the "
                                               f"pass's total_s={total_s}")
        b["min_start_s"] = min(starts) if starts else None
        b["max_end_s"] = max(ends) if ends else None
        if not b["rows"]:
            b["violations"].append("no timed rows in the trace CSV")
    for v in b["violations"]:
        failures.append(f"boundary: {v}")
    return b


ATTRIBUTION_DOC = "bench/leg.py attribution_harness / attribution_novabf"

HOST_BACKENDS = systems_base.HOST_BACKENDS   # backends with no device and no library span
# A harness leg's LegClock time is split two ways, both from the timed row's own columns
# (nothing new is timed):
# (1) the host thread's wall time, the stacked bar: loop.run_pass's buckets
#     (total_s = source_stall + engine_call + merge + other, serial or overlapped);
# (2) the GPU compute stream: gpu_idle_s = total_s minus the stream's busy spans (widen +
#     normalize + search; one stream, so they never overlap), search = lib + our work in it.
# Per-part busy times that overlap (h2d, d2h, merge, batching) are information (`busy`),
# never stacked.
HOST_PARTS = ("io_wait_s", "engine_call_s", "merge_s", "harness_other_s")
GPU_PARTS = ("engine_s", "batching_s", "engine_and_batching_s", "device_merge_s", "ids_s",
             "widen_s", "normalize_s", "device_other_s", "gpu_idle_s")
HOST_LABELS = {"io_wait_s": "io wait (reader)", "engine_call_s": "engine call (host wall)",
               "merge_s": "merge (FAISS ResultHeap)", "harness_other_s": "harness other"}
FILTERED_READER_LABEL = "reader (read + filter eval)"
# cuVS's own recipe (cuvs_merge device:*, as NVIDIA calc_truth): the running top-k is merged on
# the GPU, not per file on the host. GPU view: its own part, device_merge_s (CUDA events inside
# search_s). Host view: inside engine_call, which waits for it; the host merge bucket holds only
# the one copy back at the end of the pass (plus cosine's 1 - d).
DEVICE_MERGE_LABEL = "device merge (cuVS recipe)"
DEVICE_MERGE_HOST_LABEL = "device merge (cuVS recipe): the one copy back"
# --cuvs-sync none: no per-file sync; engine_call is the host's enqueue of each block plus its
# wait for the block before (one block in flight), not the GPU time of each block
NOSYNC_ENGINE_CALL_LABEL = "engine call (host wall; no per-file sync: enqueue + wait for the block before)"
BUSY_INFO = ("h2d_s", "h2d_exposed_s", "d2h_s", "merge_s", "lib_s", "search_s", "ids_s",
             "build_s", "engine_gap_s", "engine_host_s", "sync_wait_s", "device_merge_s")
# A view is additive when its parts sum to the LegClock time within this. Both views sum to the
# pass's total_s, so their residual is LegClock - total_s, which check_harness bounds with
# HARNESS_CLOCK_TOL_S; a smaller tolerance here would flag correct legs. A non-additive
# measurable view is a summary warning.
ADDITIVE_TOL_S = HARNESS_CLOCK_TOL_S
# Our batching wrappers are reported as their own part: SPARe's batched modes (our search loop,
# baselines/spare/spare_batched.py, timed in plan / score / topk stages) and PyLate's
# scores-batched mode (our loop around PyLate's colbert_scores).
SPARE_BATCHED_MODES = ("dot", "iterative")


def _num(v):
    """v if it is an int or float (not a bool), else None."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def batching_part(g: dict) -> tuple:
    """(seconds, where, how) of our batching wrapper's own work in one harness timed row, or
    (None, where, why) when the leg has one but it is not measured. where = "lib" (inside lib_s:
    SPARe's batched loop runs inside the library span) or "search" (inside search_s - lib_s)."""
    spare_mode, pylate_mode = g.get("spare_mode"), g.get("pylate_mode")
    lib = _num(g.get("lib_s"))
    if spare_mode in SPARE_BATCHED_MODES:
        st = [_num(g.get(f"spare_{n}_s")) for n in ("score", "topk")]
        if lib is None or any(x is None for x in st) or g.get("spare_events") != "on":
            # the wrapper runs inside lib_s and only the stage spans split it off; without
            # them the split is unknown (never assume all of lib_s is engine)
            return None, "lib", "no_spare_events: SPARe batched with its stage events off: the " \
                                "engine / batching split of lib_s is not measurable"
        # lib_s = SPARe's model build + our batched loop. The loop's scoring (SpMM / gather +
        # index_add_, with the batch's dense query build) and top-k spans are engine work; the
        # rest (planning, and the gaps between stages: loop, group slicing, torch.cat) is ours
        build = _num(g.get("build_s")) or 0.0
        return lib - build - st[0] - st[1], "lib", (
            "SPARe batched: lib_s - build_s - spare_score_s - spare_topk_s (the planning span + "
            "the loop's gaps between its stages)")
    if pylate_mode == "scores-batched":
        s = _num(g.get("search_s"))
        if lib is None or s is None:
            return None, "search", "PyLate scores-batched: not measured (no lib_s)"
        # lib_s brackets only colbert_scores; the rest of the search span, minus the id step,
        # is our wrapper (length sort, padded gathers, index_copy_, masks, topk)
        return s - lib - (_num(g.get("ids_s")) or 0.0), "search", (
            "PyLate scores-batched: search_s - lib_s - ids_s (sorting, padded gathers, "
            "index_copy_, masks, topk around colbert_scores)")
    return 0.0, None, "no batching wrapper"


def cuvs_device_merge(g: dict) -> bool:
    """True for a cuVS row run with its own recipe (cuvs_merge device:file)."""
    return str(g.get("cuvs_merge") or "").startswith("device")


def _view(parts: dict, clock_s, tol=ADDITIVE_TOL_S) -> dict:
    """Sum the parts and compare with the LegClock time (residual, additive)."""
    vals = [v for v in parts.values() if v is not None]
    s = sum(vals) if vals else None
    res = clock_s - s if clock_s is not None and s is not None else None
    return {"parts": parts, "sum_s": s, "clock_s": clock_s, "residual_s": res,
            "additive": res is not None and abs(res) <= tol}


def host_view(g: dict, clock_s, filtered: bool) -> dict:
    """View (1), the stacked bar: the host thread's wall time over the timed pass, from
    run_pass's buckets, which tile the pass in every mode. A filtered block's handoff
    (filter_handoff_s, timed inside engine_call) is harness work, so it is moved from
    engine_call to harness_other."""
    stall, call = _num(g.get("source_stall_s")), _num(g.get("engine_call_s"))
    merge, other = _num(g.get("merge_s")), _num(g.get("other_s"))
    hand = (_num(g.get("filter_handoff_s")) or 0.0) if filtered else 0.0
    parts = {"io_wait_s": stall,
             "engine_call_s": call - hand if call is not None else None,
             "merge_s": merge,
             "harness_other_s": other + hand if other is not None else None}
    v = _view(parts, clock_s)
    v["measurable"] = all(parts[k] is not None for k in HOST_PARTS)
    v["labels"] = {**HOST_LABELS, **({"io_wait_s": FILTERED_READER_LABEL} if filtered else {}),
                   **({"merge_s": DEVICE_MERGE_HOST_LABEL} if cuvs_device_merge(g) else {}),
                   **({"engine_call_s": NOSYNC_ENGINE_CALL_LABEL}
                      if cuvs_device_merge(g) and g.get("cuvs_sync") == "none" else {})}
    v["formula"] = ("io_wait = source_stall_s; engine_call = engine_call_s"
                    + (" - filter_handoff_s" if filtered else "") + "; merge = merge_s; "
                    "harness_other = other_s" + (" + filter_handoff_s" if filtered else "")
                    + " (run_pass: total_s = their sum, in every mode); residual = LegClock - sum"
                    + ("; cuVS's own recipe: merge = the one copy back (the GPU merge is "
                       "inside engine_call, split off in the GPU view as device_merge_s)"
                       if cuvs_device_merge(g) else "")
                    + ("; --cuvs-sync none: engine_call = each block's enqueue + the wait for "
                       "the block before (one in flight), not the GPU time per block"
                       if cuvs_device_merge(g) and g.get("cuvs_sync") == "none" else ""))
    return v


def gpu_view(g: dict, clock_s, bat, bat_where, bat_how) -> dict:
    """View (2): the compute stream's time over the timed pass. gpu_idle_s = total_s - busy,
    where busy = device widen + normalize + search (CUDA events, one stream) and search = lib +
    our work in the search span (ids, padding, CSR views, PyLate's batching). So engine + our
    device work + gpu_idle sums to total_s. Not measurable without device spans
    (--no-gpu-events, host backends)."""
    lib, search, idle = _num(g.get("lib_s")), _num(g.get("search_s")), _num(g.get("gpu_idle_s"))
    parts = dict.fromkeys(GPU_PARTS)
    if lib is None or search is None or idle is None:
        v = _view(parts, clock_s)
        v.update(additive=None, measurable=False,
                 why="no device spans (lib_s / gpu_idle_s blank: --no-gpu-events, a host "
                     "backend, or a path without the resident event chain)")
        return v
    ids = _num(g.get("ids_s")) or 0.0
    norm = _num(g.get("normalize_s")) or 0.0
    # device widen only (the host widen is reader-side, not on the stream)
    widen = (_num(g.get("widen_s")) or 0.0) if g.get("widen_where") == "device" else 0.0
    split = not (bat is None and bat_where == "lib")
    if split:
        parts["engine_s"] = lib - (bat if bat_where == "lib" and bat is not None else 0.0)
        parts["batching_s"] = bat if bat is not None else (None if bat_where else 0.0)
    else:   # lib_s holds the engine and our batching loop; not separable without events
        parts["engine_and_batching_s"] = lib
    in_search = bat if bat_where == "search" and bat is not None else 0.0
    # cuVS's own recipe: the GPU merge of the running top-k, its own part (inside search_s)
    dmerge = (_num(g.get("device_merge_s")) or 0.0) if cuvs_device_merge(g) else None
    parts.update(device_merge_s=dmerge, ids_s=ids, widen_s=widen, normalize_s=norm,
                 device_other_s=search - lib - ids - in_search - (dmerge or 0.0),
                 gpu_idle_s=idle)
    v = _view(parts, clock_s)
    v["labels"] = {"device_merge_s": DEVICE_MERGE_LABEL}
    v.update(measurable=True, split_measurable=split,
             why=None if split else bat_how,
             formula="engine = lib_s (minus our batching inside it); batching = our wrapper; "
                     "ids = ids_s; widen = widen_s (widen_where device, else 0); normalize = "
                     "normalize_s; device_merge = device_merge_s (cuVS's own recipe only); "
                     "device_other = search_s - lib_s - ids_s (- batching in the search span, - "
                     "device_merge); gpu_idle = gpu_idle_s (total_s - the stream's busy spans); "
                     "residual = LegClock - sum")
    return v


def attribution_harness(internal: dict, clock_s, footer_s, backend=None, gpu_events=None,
                        modes: dict | None = None) -> dict:
    """Where a harness leg's LegClock time goes, as two views of the timed row's own columns:
    `host_view` (the stacked bar: io_wait | engine_call | merge | harness_other) and `gpu_view`
    (engine + our device work + gpu_idle). Each says whether its parts sum to the LegClock time
    (`additive`, within ADDITIVE_TOL_S). Overlapping busy times (h2d, d2h, merge, batching) are
    in `busy`, information only. A filtered leg's reader part is labelled
    "reader (read + filter eval)", with filter_s / filter_eval_s beside it."""
    g = {**internal, **(modes or {})}
    filtered = g.get("filter_handoff_s") is not None
    bat, bat_where, bat_how = batching_part(g)
    hv = host_view(g, clock_s, filtered)
    gv = gpu_view(g, clock_s, bat, bat_where, bat_how)
    pipe = g.get("engine_gap_s") in ("", None) and _num(g.get("lib_s")) is not None
    busy = {k: _num(g.get(k)) for k in BUSY_INFO}
    busy["batching_s"] = bat
    host = backend in HOST_BACKENDS
    return {"kind": "harness", "total_s": clock_s,
            "host_view": hv, "gpu_view": gv,
            "measurable": hv["measurable"],
            "engine_s": gv["parts"]["engine_s"],
            "gpu_idle_s": _num(g.get("gpu_idle_s")), "footer_read_s": footer_s,
            "busy": busy,
            "filter": {"reader_label": FILTERED_READER_LABEL, "filter_s": _num(g.get("filter_s")),
                       "filter_eval_s": _num(g.get("filter_eval_s")),
                       "filter_handoff_s": _num(g.get("filter_handoff_s"))} if filtered else None,
            "batching": {"seconds": bat, "inside": bat_where, "how": bat_how},
            # from the recorded modes (with events off, engine_gap_s / lib_s / h2d_overlapped
            # are blank); the event-based signs cover a row without recorded modes
            "overlap_mode": (OVERLAP_MODE in (g.get("h2d_mode"), g.get("d2h_mode")) or pipe
                             or bool(_num(g.get("h2d_overlapped")))),
            "formula": "two views, each summing to the LegClock time: host_view = the host "
                       "thread (io_wait | engine_call | merge | harness_other), gpu_view = the "
                       "compute stream (engine + our device work + gpu_idle); busy = per-part "
                       "device / merge times, overlapping, never stacked; footer_read_s is "
                       "outside the pass",
            "backend": backend, "gpu_events": gpu_events,
            "clocks": "host_view: host perf_counter; gpu_view: CUDA events (gpu_idle = host "
                      "total - the events' busy spans)" if not host
                      else "host perf_counter (every part and the total; no device)",
            "doc": ATTRIBUTION_DOC}


def attribution_novabf(clock_s) -> dict:
    """No breakdown for nova-bf: its leg reports its timed total. nova-bf's own timers (search /
    scan / sort seconds, the bf-bench record) are kept in `internal` as information only."""
    return {"kind": "novabf", "total_s": clock_s, "host_view": None, "gpu_view": None,
            "measurable": False,
            "formula": "none: nova-bf reports its timed total only; its own timers are in "
                       "`internal` (information)",
            "doc": ATTRIBUTION_DOC}


def _origins_ok(crec, out: dict, failures: list) -> None:
    """Check the engine process's import-origins record (bench from its tree, nova_bf under the
    hashed nova-bf root); a missing or not-ok record fails the leg."""
    if crec is None:
        return
    o = crec.get("import_origins")
    out["import_origins"] = o
    if not o:
        failures.append("the engine process recorded no import origins")
    elif not o.get("ok"):
        failures.append("import origins: " + "; ".join(o.get("problems") or ["not ok"]))


def check_harness(spec: LegSpec, exp: dict, txt: str, failures: list) -> dict:
    """Check a harness leg: LegClock (the child's --clock-json record of its one timed pass), the
    CSV's numbers and the protocol: exactly one warmup pass over 1 file, exactly one timed
    pass, and the clock agreeing with total_s."""
    out = {"time_source": "LegClock"}
    cp = spec["outputs"].get("clock")
    crec = _json_or_fail(cp, failures)
    _origins_ok(crec, out, failures)   # first: a refused engine process writes only this
    p = spec["outputs"]["csv"]
    if not os.path.exists(p):
        failures.append("no harness CSV (the run died before its first pass)")
        return out
    rows = list(csv.DictReader(open(p)))
    warm = [r for r in rows if str(r.get("warmup")) not in ("0", "False", "false")]
    timed = [r for r in rows if str(r.get("warmup")) in ("0", "False", "false")]
    if len(warm) != timing.WARMUP_FILES and timing.WARMUP_FILES:
        failures.append(f"{len(warm)} warmup pass(es) in the CSV (want 1)")
    for r in warm:
        if str(r.get("warmup_files")) != str(timing.WARMUP_FILES) or str(r.get("n_files")) != str(timing.WARMUP_FILES):
            failures.append(f"warmup pass over warmup_files={r.get('warmup_files')} "
                            f"n_files={r.get('n_files')} (want {timing.WARMUP_FILES})")
    if len(timed) != timing.REPEATS:
        failures.append(f"{len(timed)} timed pass(es) in the CSV (want {timing.REPEATS})")
    if not timed:
        return out
    r = timed[-1]
    if r.get("verified") in ("failed", "error"):
        failures.append(f"the harness's --verify says {r.get('verified')}")
    for row in rows:   # documented "must be 0" counters (blank = n/a for the leg)
        for col in MUST_BE_ZERO:
            v = str(row.get(col) if row.get(col) is not None else "").strip()
            if v and v not in ("0", "0.0"):
                failures.append(f"{col} = {v} in the {'warmup' if row in warm else 'timed'} "
                                f"pass (must be 0: {MUST_BE_ZERO[col]})")
    tot = _f(r.get("total_s"))
    if tot is None:
        failures.append("the timed row has no total_s")
    n_files = r.get("n_files")
    if exp.get("subset") and str(n_files) != str(exp["subset"]["files"]):
        failures.append(f"timed pass covered n_files={n_files}, the subset is {exp['subset']['files']}")
    elif not str(n_files).isdigit():   # coverage unknown: never accept a time for it
        failures.append(f"the timed row has no n_files ({n_files!r}): its coverage is unknown")
    elif not exp.get("subset"):   # the whole corpus: the run's listing (manifest.json)
        from bench.report.summary import expected_files
        want, _ = expected_files(os.path.dirname(os.path.dirname(os.path.abspath(p))), exp)
        if want is not None and int(n_files) != want:
            failures.append(f"timed pass covered n_files={n_files}, the run's corpus listing has "
                            f"{want} files")
    clocks = (crec or {}).get("clocks") or []
    if len(clocks) != timing.REPEATS:
        failures.append(f"{len(clocks)} LegClock record(s) (want {timing.REPEATS}: the timed pass)")
    c = clocks[-1] if clocks else {}
    out["clock"] = c or None
    if clocks and (c.get("state") != "stopped" or c.get("events") != ["start", "stop"]):
        failures.append(f"LegClock {c.get('state')} {c.get('events')} (want start, stop)")
    secs = c.get("seconds")
    # the harness's footer pre-read (read_footers on every file, before the warmup) is
    # recorded beside the headline, never folded into it (timed_seconds is the LegClock pass)
    foot = (crec or {}).get("footer_read_s")
    if crec is not None and foot is None:
        failures.append("the clock record has no footer_read_s (the footer pre-read was not measured)")
    out["footer_read_s"] = foot   # its own column, never summed with secs
    if secs is not None and tot is not None:
        diff = secs - tot
        out["clock_agreement"] = {"ok": abs(diff) <= HARNESS_CLOCK_TOL_S, "diff_seconds": diff,
                                  "harness_total_s": tot,
                                  "allowed": [-HARNESS_CLOCK_TOL_S, HARNESS_CLOCK_TOL_S]}
        if not abs(diff) <= HARNESS_CLOCK_TOL_S:   # a NaN difference fails too
            failures.append(f"LegClock {secs:.6f} s disagrees with the harness total_s {tot:.6f} s")
    internal = {k: _f(v) for k, v in r.items()          # every timing / event column
                if k is not None and (k.endswith("_s") or k in HARNESS_EVENT_COLS)}
    out["boundary"] = harness_boundary(spec, tot, failures)
    out.update(timed_seconds=secs, n_files=int(n_files) if str(n_files).isdigit() else None,
               bytes_read=int(_f(r.get("bytes")) or 0),
               internal=internal,
               attribution=attribution_harness(internal, secs, foot, r.get("backend"), r.get("gpu_events"),
                                               {k: r.get(k) for k in HARNESS_MODE_COLS}),
               counters={k: r.get(k) for k in (
                   "rows", "readers", "native_fallbacks", "read_mode_effective", "row_groups",
                   "verified", "rss_hwm_bytes", "warmup_files", "omp_threads", *MUST_BE_ZERO)})
    if not os.path.exists(spec["outputs"]["topk"]):
        failures.append("no saved top-k (--save-results)")
    return out


def check_novabf(spec: LegSpec, exp: dict, txt: str, failures: list) -> dict:
    """Check a nova-bf leg: LegClock (the clock json), the boundary record, the clock
    cross-check, and the log checks (warmup marker, one timed bf-bench line covering the
    expected files)."""
    out = {"time_source": "LegClock"}
    rec = _json_or_fail(spec["outputs"]["clock"], failures)
    if rec is None:
        failures.append("no clock record (the process died before writing it)")
        return out
    out["clock"] = rec.get("clock")
    out["clock_agreement"] = rec.get("clock_agreement")
    out["boundary"] = rec.get("boundary")
    _origins_ok(rec, out, failures)
    if not rec.get("ok"):
        failures.append(f"nova-bf failed: {rec.get('error')}")
    c = rec.get("clock") or {}
    if c.get("state") != "stopped":
        failures.append(f"LegClock {c.get('state')}" + (f": {c.get('failure')}" if c.get("failure") else ""))
    ev = c.get("events") or []
    if ev and (ev[0] != "start" or ev[-1] != "stop" or ev[1:-1] != ["pause", "resume"] * ((len(ev) - 2) // 2)):
        failures.append(f"clock events out of shape: {ev}")
    b = rec.get("boundary")
    if b is None:
        failures.append("no boundary record")
    else:
        for v in b.get("violations") or []:
            failures.append(f"boundary: {v}")
        if not any(k.split("@")[1] == "running" and not k.startswith("read_schema")
                   for k in (b.get("corpus_reads") or {})):
            failures.append("boundary: no corpus data read while the clock ran")
    ag = rec.get("clock_agreement")
    if rec.get("ok") and (not ag or not ag.get("ok")):
        failures.append(f"LegClock disagrees with nova-bf's search_seconds: {ag}")
    # the log checks
    want_warm = 0 if spec["e2e"] else timing.WARMUP_FILES
    marks = list(MARK.finditer(txt))
    if not marks:
        failures.append("no TIMED_RUN_START marker (the warmup died?)")
        sec = ""
    else:
        if marks[-1].group(1) != str(want_warm):
            failures.append(f"the marker says warmup_files={marks[-1].group(1)} (want {want_warm})")
        sec = txt[marks[-1].end():]
    ms = list(BENCH.finditer(sec))
    files = None
    if marks and len(ms) != 1:
        failures.append(f"{len(ms)} bf-bench lines in the timed run (want 1)")
    elif ms:
        files = int(ms[0].group(1))
        out["bytes_read"] = int(float(ms[0].group(2)) * 1e9)
        mf, wf = MAX_FILES.search(sec), WORKER_FILES.search(sec)
        want = (int(mf.group(1)) if mf else int(wf.group(1)) if wf else None)
        if exp.get("subset"):
            want = exp["subset"]["files"]
        if want is not None and files != want:
            failures.append(f"the timed bf-bench line covers files={files}, expected {want}")
        ss = list(SEARCH.finditer(sec))
        if len(ss) != 1 or ss[0].start() < ms[0].start():
            failures.append(f"{len(ss)} bf-bench-search lines in the timed run, or one before its bf-bench line")
    out["n_files"] = files
    out["internal"] = {**(rec.get("novabf_timing") or {}), "mv_sample_seconds": rec.get("mv_sample_seconds"),
                       "bf_bench": rec.get("novabf_bench")}
    out["attribution"] = attribution_novabf(c.get("seconds") if c.get("state") == "stopped" else None)
    out["native_textscan"] = rec.get("native_textscan")
    out["stage_dirs_importable"] = rec.get("stage_dirs_importable")
    out["counters"] = {"fallbacks": len(FALLBACK.findall(sec)), "results": rec.get("results")}
    if c.get("state") == "stopped" and not failures:
        out["timed_seconds"] = c["seconds"]
    if spec["e2e"]:
        out["e2e"] = e2e_manifest_times(rec.get("results") or {})
    return out


def e2e_manifest_times(results: dict) -> dict | None:
    """The cold e2e times from nova-bf's manifest (elapsed_seconds covers query load, setup
    and compile, scan, result write)."""
    try:
        import pyarrow.fs as pafs
        first = next(iter(results.values()))
        d = first.rsplit("/", 1)[0]
        fs, p = pafs.FileSystem.from_uri(d) if "://" in d else (pafs.LocalFileSystem(), d)
        for i in fs.get_file_info(pafs.FileSelector(p)):
            if "_bf_manifest" in i.path and i.path.endswith(".json"):
                t = json.loads(fs.open_input_stream(i.path).read())["timing"]
                return {k: t.get(k) for k in ("elapsed_seconds", "scan_seconds", "search_seconds",
                                              "io_wait_seconds", "gpu_seconds")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return None


def novabf_topk(results: dict, path: str) -> str | None:
    """Convert nova-bf's result parquet(s) to .topk.npz (untimed): per search `D` (hit_scores),
    `ids` (hit_ids: nova-bf's md5 point ids, or corpus ids with id_column) and `query_ids`. Rows
    are padded to the longest hit list (D with -inf, ids with ""). Ids are not mapped to corpus
    rows here; verify/agree.py does that."""
    import numpy as np
    import pyarrow.parquet as pq
    arrays = {}
    for name, p in sorted(results.items()):
        t = pq.read_table(p)
        ids = t.column("hit_ids").to_pylist()
        sc = t.column("hit_scores").to_pylist()
        w = max((len(x) for x in ids), default=0)
        D = np.full((len(ids), w), -np.inf, dtype=np.float32)
        I = np.full((len(ids), w), "", dtype=object)
        for i, (a, b) in enumerate(zip(ids, sc)):
            D[i, :len(b)] = b
            I[i, :len(a)] = a
        pre = "" if len(results) == 1 else f"{name}__"
        arrays[pre + "D"] = D
        arrays[pre + "ids"] = I.astype(str)
        arrays[pre + "query_ids"] = np.asarray(t.column("query_id").to_pylist(), dtype=str)
    np.savez(path, **arrays)
    return path


_ERR_LINE = re.compile(r"^(?:[\w.]+(?:Error|Exception|Exit)|error|bench[\w ]*?): .+$", re.M)


def last_error_line(txt: str, width: int = 300) -> str | None:
    """The last `SomeError: ...` (a Python traceback's last line) or `error: ...` line of a
    leg's log, cut to ``width`` characters; None when there is none."""
    m = _ERR_LINE.findall(txt or "")
    return m[-1].strip()[:width] if m else None


def run_leg(exp: dict, search: str | None, system: str, rnd: int | None, run_dir: str,
            dev: bool = False, e2e: bool = False, base_env: dict | None = None,
            spec: LegSpec | None = None) -> dict:
    """Plan, run, check and record one leg. Returns the record (also legs/<tag>.json)."""
    failures: list[str] = []
    spec = spec or plan(exp, search, system, rnd, run_dir, dev=dev, e2e=e2e)
    # a re-run leg makes any earlier checks.json stale: removed, so it cannot vouch for it
    stale = [spec["log"], *spec["outputs"].values(), os.path.join(run_dir, "checks.json")]
    if "csv" in spec["outputs"]:   # the harness appends to its CSVs: remove any stale rows
        c = spec["outputs"]["csv"]
        stale += [c[:-4] + ".trace.csv", c + ".pending", c[:-4] + ".trace.csv.pending"]
    for p in stale:   # a stale file must never be read as this leg's
        if isinstance(p, str) and os.path.isfile(p) and p != spec["outputs"].get("config"):
            os.remove(p)
    cold = e2e and not (exp.get("e2e") or {}).get("keep_cache")
    # a cold leg whose caches an env variable moves elsewhere is refused, not run
    redirects = timing.e2e_cache_redirects(
        {**(os.environ if base_env is None else base_env), **spec["env_add"]}) if cold else []
    if redirects:
        failures += [f"e2e refused: {x}" for x in redirects]
        return _write_record(exp, spec, run_dir, failures, {}, {
            "exit_code": None, "wrapper": [], "started_unix": time.time(),
            "ended_unix": time.time(), "children_maxrss_bytes": None}, [], None, dev, e2e, [])
    # no other process may share the GPU: checked here, before the leg process exists
    foreign, refusal = gpu_foreign_procs(spec, dev)
    if refusal:
        failures.append(refusal)
        return _write_record(exp, spec, run_dir, failures, {}, {
            "exit_code": None, "wrapper": [], "started_unix": time.time(),
            "ended_unix": time.time(), "children_maxrss_bytes": None}, [], None, dev, e2e, [],
            gpu_foreign=foreign)
    cleared = timing.clear_e2e_caches() if cold else []
    run = spawn(spec, base_env)
    txt = open(spec["log"], errors="replace").read() if os.path.exists(spec["log"]) else ""
    if run["exit_code"] != 0:
        # keep the engine's last error line, so a refusal (e.g. cuVS's "empty after
        # query-vocab projection") shows in the record and summary.md, not only the log
        err = last_error_line(txt)
        failures.append(f"exit status {run['exit_code']}" + (f" ({err})" if err else ""))
    m = EXIT.findall(txt)
    if m and int(m[-1]) != 0 and run["exit_code"] == 0:
        failures.append(f"/usr/bin/time says exit status {m[-1]}")
    res = (check_harness if spec["kind"] == "harness" else check_novabf)(spec, exp, txt, failures)
    rss = RSS.findall(txt)
    if spec["kind"] == "novabf" and not failures:
        try:
            novabf_topk((res.get("counters") or {}).get("results") or {}, spec["outputs"]["topk"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"could not write the top-k npz: {type(exc).__name__}: {exc}")
    return _write_record(exp, spec, run_dir, failures, res, run, rss, process_wall_s(txt), dev,
                         e2e, cleared, gpu_foreign=foreign)


ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)")


def process_wall_s(txt: str) -> float | None:
    """The leg process's wall time in seconds, from the last /usr/bin/time -v `Elapsed (wall
    clock) time` line in the log; None without /usr/bin/time."""
    m = ELAPSED.findall(txt or "")
    if not m:
        return None
    sec = 0.0
    for part in m[-1].split(":"):
        sec = sec * 60 + float(part)
    return sec


def _write_record(exp, spec, run_dir, failures, res, run, rss, wall, dev, e2e, cleared,
                  gpu_foreign=None) -> dict:
    """Write legs/<tag>.json atomically and return it; a failed leg gets no timed_seconds."""
    rec = {
        "schema": RECORD_SCHEMA, "experiment": exp["name"], "tag": spec["tag"],
        "search": spec["search"], "system": spec["system"], "round": spec["round"],
        "e2e": spec["e2e"], "kind": spec["kind"], "system_kind": spec["system_kind"],
        "preset": spec["preset"], "knobs": spec["knobs"], "chosen_from": spec["chosen_from"],
        "ok": not failures, "failures": failures,
        "timed_seconds": res.get("timed_seconds") if not failures else None,
        "footer_read_s": res.get("footer_read_s"),
        "time_source": res.get("time_source"),
        "attribution": res.get("attribution"),
        "internal": res.get("internal"), "counters": res.get("counters"),
        "n_files": res.get("n_files"), "bytes_read": res.get("bytes_read"),
        "peak_rss_bytes": int(rss[-1]) * 1024 if rss else run["children_maxrss_bytes"],
        # the whole leg process's wall time (/usr/bin/time -v), information beside the
        # LegClock headline (includes imports, warmup or JIT, result writing)
        "process_wall_s": wall,
        "exit_code": run["exit_code"], "wrapper": run["wrapper"],
        "argv": spec["argv"], "env_added": spec["env_add"], "cwd": spec["cwd"],
        "log": os.path.relpath(spec["log"], run_dir),
        "topk": os.path.relpath(spec["outputs"]["topk"], run_dir)
        if os.path.exists(spec["outputs"]["topk"]) else None,
        "protocol": {"warmup_files": 0 if e2e else timing.WARMUP_FILES,
                     "repeats": timing.REPEATS, "fresh_process": True,
                     "caches_cleared": cleared},
        "started_unix": run["started_unix"], "ended_unix": run["ended_unix"],
        "dev": dev,
        # other GPU processes when the leg started ([] = none; see gpu_foreign_procs)
        "gpu_foreign_procs": gpu_foreign,
    }
    for k in ("clock", "clock_agreement", "boundary", "e2e", "native_textscan", "stage_dirs_importable",
              "import_origins"):
        if k in res:
            rec[k if k != "e2e" else "e2e_manifest_timing"] = res[k]
    path = os.path.join(run_dir, "legs", f"{spec['tag']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)
    return rec


def main(argv=None) -> int:
    """bench leg RUN_DIR --system S [--search X --round N | --e2e] [--dev]: one leg of a run
    whose resolved experiment is RUN_DIR/experiment.json (written by `bench run`)."""
    import argparse
    ap = argparse.ArgumentParser(prog="bench leg", description=main.__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--system", required=True)
    ap.add_argument("--search", default=None)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--e2e", action="store_true")
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    exp = json.load(open(os.path.join(a.run_dir, "experiment.json")))
    paths.use(exp)
    if not a.e2e and (a.search is None or a.round is None):
        ap.error("--search and --round (or --e2e)")
    rec = run_leg(exp, a.search, a.system, a.round, a.run_dir, dev=a.dev, e2e=a.e2e)
    print(f"bench leg {rec['tag']}: {'ok' if rec['ok'] else 'FAILED'} "
          + (f"{rec['timed_seconds']:.3f} s ({rec['time_source']})" if rec["ok"] else "; ".join(rec["failures"])))
    return 0 if rec["ok"] else 1
