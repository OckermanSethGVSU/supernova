"""Run experiments and parameter sweeps.

Validates the configuration, checks the environment, and executes each leg
in a separate process. Runs verification checks and generates a summary,
failing if any leg or check fails.

Sweeps use the same execution process and support selecting winning
configurations without modifying experiment files.

Dev mode uses the CPU and reports environment issues without failing.
Per-run overrides are applied before validation and recorded in the manifest.
"""

from __future__ import annotations

import copy
import csv
import datetime
import glob
import hashlib
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys

from bench import experiment as experiment_mod
from bench import paths
from bench import timing
from bench.report import summary as summary_mod
from bench.systems import base as systems_base
from bench.verify import agree

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CB)
RESULTS = os.path.join(CB, "results")
NOISE_FRAC = 0.02   # --pick: a runner-up within 2 % of the winner is "within noise"
EVENTS_MIN_FRAC = 0.005   # events table: "worth a look" above max(0.5 %, spread)


# ---------------------------------------------------------------------------------------------
def _git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _set(raw: dict, assignments: list[str]) -> dict:
    """Apply `--set a.b.c=VALUE` to a copy of the raw experiment (VALUE is parsed as YAML)."""
    import yaml
    raw = copy.deepcopy(raw)
    for a in assignments or []:
        if "=" not in a:
            raise experiment_mod.ExperimentError(f"--set {a!r}: KEY=VALUE")
        k, v = a.split("=", 1)
        node = raw
        parts = k.split(".")
        for p in parts[:-1]:
            if node.get(p) is None:          # a missing mapping is created
                node[p] = {}
            elif node is raw.get("systems") and isinstance(node[p], str):
                node[p] = {"preset": node[p]}   # `name: PRESET` short form -> its mapping
            if not isinstance(node.get(p), dict):
                raise experiment_mod.ExperimentError(f"--set {k}: {p} is not a mapping")
            node = node[p]
        node[parts[-1]] = yaml.safe_load(v)
    return raw


def env_check(mode: str, run_dir: str) -> dict:
    if mode == "skip":
        return {"mode": "skip", "exit": None}
    cmd = [sys.executable, os.path.join(CB, "env", "check.py"), "--versions",
           os.path.join(CB, "env", "versions.env"), "--json", os.path.join(run_dir, "env_check.json")]
    if mode == "report":
        cmd.append("--report-only")
    with open(os.path.join(run_dir, "logs", "env_check.txt"), "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=CB)
    return {"mode": mode, "exit": rc}


def _gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _imds(path):
    if os.environ.get("HARNESS_NO_IMDS"):
        return None
    import urllib.request
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=1).read().decode()
        return urllib.request.urlopen(urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": tok}), timeout=1).read().decode()
    except Exception:  # noqa: BLE001
        return None


def gpus(text: str | None) -> list[dict]:
    """nvidia-smi --query-gpu=name,uuid,driver_version lines -> [{name, uuid, driver}]."""
    out = []
    for line in (text or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3:
            out.append({"name": f[0], "uuid": f[1], "driver": f[2]})
    return out


def staged() -> dict | None:
    """The commit, dirty flag and code sha from CB/STAGED, which `bench launch` writes (a node
    has no .git). None outside a stage."""
    p = os.path.join(CB, "STAGED")
    if not os.path.isfile(p):
        return None
    text = open(p).read()
    out = {"text": text, "commit": None, "dirty": None, "code_sha": None}
    for line in text.splitlines():
        if line.startswith("staged ") and "commit=" in line:
            out["commit"] = line.split("commit=", 1)[1].split()[0]
        elif line.startswith("code sha256: "):
            out["code_sha"] = line.split(": ", 1)[1].strip()
        elif line.startswith("dirty: "):
            out["dirty"] = line.split(": ", 1)[1].strip() == "yes"
    return out


def node_info() -> dict:
    out = {}
    try:
        for line in open("/proc/stat"):
            if line.startswith("btime "):
                out["booted_utc"] = datetime.datetime.fromtimestamp(
                    int(line.split()[1]), datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                out["mem_total_kib"] = int(line.split()[1])
        out["transparent_hugepage"] = open("/sys/kernel/mm/transparent_hugepage/enabled").read().strip()
    except (OSError, ValueError):
        pass
    return out


def dirty_paths(exp: dict) -> list[str]:
    """Uncommitted changes in the code a local run executes: clean_benchmarks (minus results/),
    python/nova-bf, and for a filtered experiment crates/nova-textscan. These are the paths
    `bench launch` ships."""
    paths = [CB, os.path.join(REPO, "python", "nova-bf")]
    if exp.get("modality") == "filtered":   # plus the workspace's root Cargo files
        paths += [os.path.join(REPO, "crates", "nova-textscan"), os.path.join(REPO, "Cargo.lock"),
                  os.path.join(REPO, "Cargo.toml")]
    out = _git("status", "--porcelain", "--", *paths, f":(exclude){os.path.join(CB, 'results')}")
    return [x for x in out.splitlines() if x.strip()]


def _sha256_file(path: str, drop_cache: bool = False) -> str:
    """sha256 of a file. With ``drop_cache`` (a corpus file hashed before the legs), the file's
    pages are dropped from the page cache afterwards (POSIX_FADV_DONTNEED), so hashing does not
    warm the cache for the first leg."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
        if drop_cache and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return h.hexdigest()


# A local corpus up to this many bytes is sha256'd file by file before the legs (hashing reads
# every byte). Above it, only size and mtime are recorded, with a note.
LOCAL_CORPUS_SHA_MAX_BYTES = 8 << 30


def _s3_listing(uri: str, region: str | None, recursive: bool) -> list[dict]:
    """[{key, size, etag, last_modified}] of the *.parquet objects under an s3:// prefix (boto3,
    one LIST per 1000 keys)."""
    import re
    import boto3
    # Send no request that is bound to fail: keys that are not AWS key ids (e.g. "<secret>"
    # placeholders), or no keys at all, are recorded as a skipped listing.
    keyid = os.environ.get("AWS_ACCESS_KEY_ID")
    if keyid is not None and not re.fullmatch(r"(AKIA|ASIA)[A-Z0-9]{16}", keyid):
        raise RuntimeError("skipped: AWS_ACCESS_KEY_ID is not an AWS key id (no request sent)")
    if keyid is None and os.environ.get("HARNESS_NO_IMDS"):
        raise RuntimeError("skipped: no AWS keys in the environment and HARNESS_NO_IMDS set")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    prefix = prefix.split("?", 1)[0]
    if prefix and not prefix.endswith("/") and not prefix.endswith(".parquet"):
        prefix += "/"
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    out = []
    kw = {"Bucket": bucket, "Prefix": prefix, **({} if recursive else {"Delimiter": "/"})}
    for page in s3.get_paginator("list_objects_v2").paginate(**kw):
        for o in page.get("Contents") or []:
            if o["Key"].endswith(".parquet"):
                out.append({"key": f"s3://{bucket}/{o['Key']}", "size": o["Size"],
                            "etag": o["ETag"].strip('"'), "last_modified": o["LastModified"].isoformat()})
    return sorted(out, key=lambda r: r["key"])


def _local_listing(path: str, recursive: bool, sha: bool = False) -> list[dict]:
    from bench.data.sources import list_parquet
    _, refs = list_parquet(path, recursive=recursive)
    out = []
    for r in refs:
        st = os.stat(r.path)
        out.append({"key": r.path, "size": st.st_size,
                    "mtime": datetime.datetime.fromtimestamp(st.st_mtime, datetime.UTC).isoformat(),
                    **({"sha256": _sha256_file(r.path)} if sha else {})})
    return out


def data_identity(exp: dict) -> dict:
    """What data the run read, for the manifest. Gathered before the legs, outside every timed
    window: the corpus listing (key, size, S3 ETag or local mtime) and the queries' sha256 and
    size (on S3: ETag and size; nothing is downloaded). Relative paths resolve from the
    clean_benchmarks root, as in the legs. Never fails a run: an error is recorded instead."""
    def resolve(p):
        p = os.path.expanduser(paths.expand(str(p)))
        return p if "://" in p or os.path.isabs(p) else os.path.join(CB, p)
    out = {}
    c = exp["corpus"]
    try:
        p = resolve(c["path"])
        files = _s3_listing(p, c.get("region"), bool(c.get("recursive"))) if p.startswith("s3://") \
            else _local_listing(p, bool(c.get("recursive")))
        tot = sum(f["size"] for f in files)
        out["corpus"] = {"path": c["path"], "n_files": len(files), "total_bytes": tot,
                         "files": files}
        if not p.startswith("s3://"):
            if tot <= LOCAL_CORPUS_SHA_MAX_BYTES:
                for f in files:
                    f["sha256"] = _sha256_file(f["key"], drop_cache=True)
                out["corpus"]["hashed"] = "sha256 of every file"
            else:
                out["corpus"]["hashed"] = (f"not hashed: {tot} bytes > the {LOCAL_CORPUS_SHA_MAX_BYTES}"
                                           "-byte cap (runner.LOCAL_CORPUS_SHA_MAX_BYTES); size + "
                                           "mtime only")
            dl = (exp.get("prep") or {}).get("download")
            if dl:
                out["corpus"]["downloaded_from"] = download_identity(dl, files)
    except Exception as exc:  # noqa: BLE001  provenance never fails a run
        out["corpus"] = {"path": c["path"], "error": f"{type(exc).__name__}: {exc}"}
    q = exp["queries"]
    try:
        p = resolve(q["path"])
        files = _s3_listing(p, c.get("region"), False) if p.startswith("s3://") \
            else _local_listing(p, False, sha=True)
        out["queries"] = {"path": q["path"], "files": files}
    except Exception as exc:  # noqa: BLE001
        out["queries"] = {"path": q["path"], "error": f"{type(exc).__name__}: {exc}"}
    gt = (exp.get("checks") or {}).get("gt")
    if gt:   # the dataset ground truth the checks compare against
        try:
            p = resolve(gt)
            if p.startswith("s3://"):
                out["gt"] = {"path": gt, "files": _s3_listing(p, c.get("region"), False)}
            else:
                out["gt"] = {"path": gt, "size": os.path.getsize(p), "sha256": _sha256_file(p)}
        except Exception as exc:  # noqa: BLE001
            out["gt"] = {"path": gt, "error": f"{type(exc).__name__}: {exc}"}
    srcs = prep_source_identity(exp)
    if srcs:
        out["prep_sources"] = srcs
    return out


def prep_source_identity(exp: dict) -> list[dict]:
    """The identity of each prep.stage source: the sha256 `bench launch` wrote into STAGED (on a
    node, where the source is absent), else the source file's own sha256, else `absent`."""
    stage = (staged() or {}).get("text") or ""
    out = []
    for d in (exp.get("prep") or {}).get("stage") or []:
        rec = {"key": d.get("key"), "path": d.get("path"), "source": d.get("source")}
        m = [ln for ln in stage.splitlines() if ln.startswith(f"data {d.get('path')}: ")]
        if m and " sha256 " in m[-1]:
            rec.update(sha256=m[-1].rsplit(" sha256 ", 1)[1].split()[0], from_="STAGED")
        else:
            try:
                from bench.prep import steps
                src = steps.prep_source(d["source"])
                rec.update(sha256=_sha256_file(src), size=os.path.getsize(src), from_="source file")
            except (OSError, KeyError, TypeError) as exc:
                rec.update(sha256=None, from_=f"absent ({type(exc).__name__})")
        rec["from"] = rec.pop("from_")
        out.append(rec)
    return out


def download_identity(dl: dict, local: list[dict]) -> dict:
    """The S3 objects prep.download fetched (the realfile corpus), with ETag and size, each
    matched to its local file by name and size. Uses the same keys the download script picks
    (the first `per_dir` *.parquet of each dir, sorted). Listed when the run starts, right after
    the download in a launched job; a skipped or failed listing is recorded as such."""
    try:
        objs = []
        src = paths.expand(dl["src"]).rstrip("/")
        for dd in dl["dirs"]:
            got = _s3_listing(f"{src}/{dd}/", dl.get("region"), False)
            objs += got[:int(dl["per_dir"])]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    by_name = {os.path.basename(f["key"]): f for f in local}
    for o in objs:
        f = by_name.get(os.path.basename(o["key"]))
        o["local"] = None if f is None else f["key"]
        o["local_size_matches"] = f is not None and f["size"] == o["size"]
    return {"objects": objs}


def import_origins() -> dict:
    """Where this process and the legs' interpreter import bench / nova_bf from."""
    from bench.record import provenance
    # The legs' interpreter is this one. BENCH_ENGINE_PYTHON, a test hook of leg.plan that may
    # point at a stub, is deliberately not used.
    return {"runner": provenance.check(CB), "engine": provenance.probe(CB, sys.executable)}


def manifest(exp: dict, run_dir: str, argv: list, overrides: list, dev: bool, envc: dict,
             launch_id: str | None = None) -> dict:
    stage = staged()
    commit = _git("rev-parse", "HEAD") or (stage or {}).get("commit") or ""
    files = {}
    for p in [exp["_file"], *exp.get("_base", [])]:
        files[os.path.relpath(p, CB)] = open(p).read()
    for sy in exp["systems"].values():
        cfg = sy["knobs"].get("config")
        if cfg:
            files[os.path.relpath(cfg, CB)] = open(cfg).read()
    if exp.get("filter"):
        files[os.path.relpath(exp["filter"]["config"], CB)] = open(exp["filter"]["config"]).read()
    dpaths = dirty_paths(exp)
    return {
        "schema": 1, "experiment": exp["name"], "run_id": os.path.basename(run_dir),
        # the --launch-id of the `bench launch` that started this run (None when started by
        # hand); `bench launch --sync` stamps and counts only runs that carry its id
        "launch_id": launch_id,
        # the private-path variables the experiment uses and a sha256 of their values
        "paths": paths.record(exp),
        "argv": argv, "overrides": overrides, "dev": dev,
        "commit": commit,
        "dirty": (stage or {}).get("dirty") if stage and not _git("rev-parse", "HEAD")
        else bool(dpaths),
        "dirty_paths": None if stage and not _git("rev-parse", "HEAD") else dpaths,
        # the launch's data steps, as the experiment states them
        "prep": exp.get("prep"),
        "stage": stage,
        # the code hash: paper_run.sh's PAPER_CODE_SHA on a node, else the stage's
        "code_sha": os.environ.get("PAPER_CODE_SHA") or (stage or {}).get("code_sha"),
        "stack_sha": os.environ.get("PAPER_STACK_SHA"),
        "paper": {k: os.environ.get(k) for k in ("PAPER_STACK_SHA", "PAPER_CODE_SHA",
                                                  "PAPER_ENV_PREFIX", "PAPER_RUN_DIR")},
        "env_check": envc,
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.executable, "cpu_count": os.cpu_count(),
        "instance": {"type": _imds("instance-type"), "id": _imds("instance-id"),
                     "az": _imds("placement/availability-zone")} if not dev else None,
        "node": node_info(),   # boot time, memory, transparent huge pages
        "gpu": _gpu(), "gpus": gpus(_gpu()),
        "files": files,
        "protocol": {"rounds": timing.rounds(exp), "warmup_files": timing.WARMUP_FILES,
                     "repeats": timing.REPEATS, "rotation": "by one per round (timing_rotate)"},
        "schedule": [list(x) for x in timing.schedule(exp)],
        "data": data_identity(exp),   # before the legs (execute writes this first)
        # where this process imports bench / nova_bf from, and what an engine process of the
        # legs' interpreter resolves (`python -P`); each leg also checks its own and refuses
        "import_origins": import_origins(),
        "started_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


CUDA_LIB_COLS = ("cuda_libraries", "cuda_lib_versions", "cublas_version", "cuda_lib_duplicates")


def leg_cuda_libs(run_dir: str) -> dict:
    """The CUDA libraries each leg mapped, as its engine recorded them after its pass. Harness
    legs: from the CSV's timed row (provenance.cuda_libs_fields; blank means no CUDA used).
    nova-bf legs: from nova-bf's own manifest next to the leg's result, when readable from here."""
    import csv
    out = {}
    legs = os.path.join(run_dir, "legs")
    for f in sorted(os.listdir(legs)):
        tg, ext = os.path.splitext(f)
        if ext == ".csv" and not f.endswith(".trace.csv"):
            rows = [r for r in csv.DictReader(open(os.path.join(legs, f)))
                    if str(r.get("warmup")) in ("0", "False", "false")]
            if rows:
                out[tg] = {k: rows[-1].get(k, "") for k in CUDA_LIB_COLS}
        elif ext == ".json" and not f.endswith(".clock.json"):
            try:
                rec = json.load(open(os.path.join(legs, f)))
            except ValueError:
                continue
            res = (rec.get("counters") or {}).get("results") if rec.get("kind") == "novabf" else None
            if res:
                out[tg] = _novabf_cuda_libs(res)
    return out


def _novabf_cuda_libs(results: dict) -> dict:
    """CUDA library fields from the nova-bf manifest next to a result; an error dict if none."""
    try:
        import pyarrow.fs as pafs
        first = next(iter(results.values()))
        d = first.rsplit("/", 1)[0]
        fs, p = pafs.FileSystem.from_uri(d) if "://" in d else (pafs.LocalFileSystem(), d)
        for i in fs.get_file_info(pafs.FileSelector(p)):
            if "_bf_manifest" in i.path and i.path.endswith(".json"):
                c = json.loads(fs.open_input_stream(i.path).read()).get("compute") or {}
                return {"manifest": i.path, **{k: c.get(k) for k in (
                    "cuda_libraries", "cuda_library_duplicates", "cublas_version")}}
    except Exception as exc:  # noqa: BLE001  provenance never fails a run
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"error": "no nova-bf manifest next to the result"}


def leg_field(run_dir: str, key: str) -> dict:
    """{leg tag: record[key]} over the run's leg records that have it."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "legs", "*.json"))):
        if p.endswith(".clock.json"):
            continue
        try:
            rec = json.load(open(p))
        except (OSError, ValueError):
            continue
        if rec.get(key) is not None:
            out[rec.get("tag")] = rec[key]
    return out


def finish_manifest(run_dir: str) -> None:
    """After the legs, add what only the legs know (CUDA libraries, native textscan, end time)."""
    p = os.path.join(run_dir, "manifest.json")
    m = json.load(open(p))
    m["cuda_libs"] = leg_cuda_libs(run_dir)
    m["novabf_native_textscan"] = leg_field(run_dir, "native_textscan")
    m["finished_utc"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(p, "w") as f:
        json.dump(paths.mask(m), f, indent=1, sort_keys=True)


def short_sha() -> str:
    """The run dir's commit tag: git's short HEAD, else the STAGED commit's first 7 characters
    (a node has no .git), else "nogit"."""
    return _git("rev-parse", "--short", "HEAD") or ((staged() or {}).get("commit") or "")[:7] \
        or "nogit"


def new_run_dir(root: str) -> str:
    utc = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = short_sha()
    d = os.path.join(root, f"{utc}_{sha}")
    n = 1
    while os.path.exists(d):
        n += 1
        d = os.path.join(root, f"{utc}_{sha}-{n}")
    os.makedirs(os.path.join(d, "legs"))
    os.makedirs(os.path.join(d, "logs"))
    return d


def dev_refusals(exp: dict) -> list[str]:
    out = []
    for name, sy in exp["systems"].items():
        if not systems_base.SYSTEMS[sy["system"]].cpu_ok:
            out.append(f"{name} ({sy['system']}) does not run on a CPU dev box")
    return out


def run_legs(exp: dict, run_dir: str, dev: bool, call=subprocess.call) -> list[str]:
    """Run the e2e leg (if any), then the schedule, each `bench leg` in its own process. Returns
    one problem per leg that exited non-zero (its record, if any, says why)."""
    problems = []
    base = [sys.executable, "-m", "bench", "leg", run_dir] + (["--dev"] if dev else [])
    todo = []
    if exp.get("e2e"):
        todo.append((f"e2e {exp['e2e']['system']}", ["--system", exp["e2e"]["system"], "--e2e"]))
    for r, s, sy in timing.schedule(exp):
        todo.append((f"{s} {sy} r{r}", ["--system", sy, "--search", s, "--round", str(r)]))
    for what, args in todo:
        rc = call(base + args, cwd=CB)
        if rc != 0:
            problems.append(f"bench leg {what}: exit {rc}")
    return problems


def execute(exp: dict, run_dir: str, argv: list, overrides: list, dev: bool, env_mode: str,
            launch_id: str | None = None) -> int:
    """Steps 2-6 of `bench run` in an existing run dir; returns the exit code."""
    envc = env_check(env_mode, run_dir)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(paths.mask(manifest(exp, run_dir, argv, overrides, dev, envc, launch_id)), f,
                  indent=1, sort_keys=True)
    with open(os.path.join(run_dir, "experiment.json"), "w") as f:
        f.write(experiment_mod.dumps(exp))
    if env_mode == "strict" and envc["exit"] != 0:
        print(f"bench: env/check.py failed (exit {envc['exit']}; {run_dir}/logs/env_check.txt): "
              "not the pinned environment, nothing run", file=sys.stderr)
        return 3
    leg_problems = run_legs(exp, run_dir, dev)
    finish_manifest(run_dir)
    checks = agree.run_checks(run_dir, exp)
    with open(os.path.join(run_dir, "checks.json"), "w") as f:
        json.dump(paths.mask(checks), f, indent=1, sort_keys=True)
    rows, problems, excluded, e2e = summary_mod.summarize(run_dir, exp)
    problems = leg_problems + problems
    print(summary_mod.write(run_dir, rows, problems, excluded, e2e), end="")
    print(f"bench: results in {run_dir}")
    return 1 if problems else 0


def _prepare(path, a, sweep=False):
    if sweep:
        raw = experiment_mod.load_raw(path)
        raw = _set(raw, a.set)
        # a `--set protocol.rounds=N` is the user's choice, so the sweep keeps it
        keep = any(x.split("=", 1)[0].strip() == "protocol.rounds" for x in a.set or [])
        e_raw, variants = experiment_mod.sweep_variants(raw, path, keep_rounds=keep)
        exp = experiment_mod.finish(e_raw, path)
        return _paths_known(exp), variants, raw["sweep"]
    raw = experiment_mod.load_raw(path)
    if "sweep" in raw:
        raise experiment_mod.ExperimentError(f"{path}: a sweep file: use `bench sweep`")
    raw = _set(raw, a.set)
    for nm, sy in (raw.get("systems") or {}).items():   # per-system io is for sweeps only
        if isinstance(sy, dict) and "sweep_io" in sy:
            raise experiment_mod.ExperimentError(f"{path}: systems.{nm}.sweep_io: set by `bench "
                                                 "sweep` only (a paper run shares one `io:`)")
    return _paths_known(experiment_mod.finish(raw, path)), None, None


def _paths_known(exp: dict) -> dict:
    """The experiment, refused when it names a private path (`${READ}` / `${WRITE}`) that has no
    value for it in the environment or in experiments/local_paths.env."""
    try:
        paths.check(exp)
    except paths.PathsError as exc:
        raise experiment_mod.ExperimentError(str(exc)) from None
    return exp


def _common_args(ap):
    ap.add_argument("--dev", action="store_true", help="CPU dev box")
    ap.add_argument("--env-check", choices=["strict", "report", "skip"], default=None,
                    help="default strict (--dev: report)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--rounds", type=positive_int, default=None, help="override protocol.rounds")
    ap.add_argument("--results", default=RESULTS, help="results root (default clean_benchmarks/results)")
    ap.add_argument("--allow-placeholders", action="store_true", help=argparse_suppress())


def launch_id_arg(v: str) -> str:
    """`--launch-id`, the id `bench launch` gives a launch (<UTC>_<sha>[-N])."""
    import argparse
    from bench.launch import LAUNCH_ID_RE
    if not LAUNCH_ID_RE.fullmatch(v):
        raise argparse.ArgumentTypeError(f"{v!r}: a launch id <UTC>_<sha>[-N] (bench launch's)")
    return v


def add_launch_id(ap) -> None:
    ap.add_argument("--launch-id", type=launch_id_arg, default=None,
                    help="set by `bench launch` on the node: recorded in manifest.json, matched "
                         "by `bench launch --sync`")


def positive_int(v: str) -> int:
    """--rounds is a positive integer."""
    import argparse
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v!r}: a positive integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{v!r}: a positive integer")
    return n


def argparse_suppress():
    import argparse
    return argparse.SUPPRESS


def io_hint(exp: dict, ph: list[str]) -> str:
    """The pointer to the io sweep, for a refusal that lists io placeholders."""
    if not any(p.startswith("io.") for p in ph):
        return ""
    return ("; pick the io values with an io sweep over io.* (a sweep file): "
            "`bench sweep FILE`, then `bench sweep RUN_DIR --pick` prints the YAML change")


def _checks_before_run(exp, a) -> int:
    ph = experiment_mod.placeholders(exp)
    if ph and not a.allow_placeholders:
        print("bench: placeholder value(s) (TODO), set them in the file or with --set: "
              + "; ".join(ph) + io_hint(exp, ph), file=sys.stderr)
        return 2
    if a.dev:
        bad = dev_refusals(exp)
        if bad:
            print("bench --dev: " + "; ".join(bad), file=sys.stderr)
            return 2
    return 0


def cmd_run(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bench run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    _common_args(ap)
    add_launch_id(ap)
    a = ap.parse_args(argv)
    try:
        exp, _, _ = _prepare(a.experiment, a)
    except experiment_mod.ExperimentError as exc:
        print(f"bench run: {exc}", file=sys.stderr)
        return 2
    if a.rounds is not None:
        exp["protocol"]["rounds"] = a.rounds
    rc = _checks_before_run(exp, a)
    if rc:
        return rc
    run_dir = new_run_dir(os.path.join(a.results, exp["name"]))
    return execute(exp, run_dir, ["bench", "run", *argv], a.set, a.dev,
                   a.env_check or ("report" if a.dev else "strict"), a.launch_id)


# ---- sweeps ---------------------------------------------------------------------------------
def pairs_blame(pairs: dict, listed: list[str]) -> dict:
    """{system: why} for a failed pairs check, from compare.judge's per-system counters: list
    defects (`system_defects[nm].failed`), float64-checked disagreements (`f64_wrong_total[nm]`)
    and order-failed pairs (see order_blame). If the compare could not run (no `compare`
    document) or no counter explains the failure, every listed system is blamed."""
    res = pairs.get("compare")
    why_all = "; ".join(pairs.get("fails") or []) or "not ok"
    if not isinstance(res, dict):
        return {nm: why_all for nm in listed}
    out: dict = {}
    for nm in listed:
        w = []
        if ((res.get("system_defects") or {}).get(nm) or {}).get("failed"):
            w.append("list defects (duplicate or unmapped ids)")
        c = (res.get("f64_wrong_total") or {}).get(nm)
        if c:
            w.append(f"{c} float64-checked disagreements")
        w += order_blame(res, listed).get(nm, [])
        w += shared_score_blame(res, listed).get(nm, [])
        if w:
            out[nm] = "; ".join(w)
    return out or {nm: why_all for nm in listed}


def _pair_of(pk: str, listed: list[str]):
    """The (x, y) of the systems listed in a round that compare.judge's pair key x_vs_y names."""
    for i, x in enumerate(listed):
        for y in listed[i + 1:]:
            if pk == f"{x}_vs_{y}":
                return x, y
            if pk == f"{y}_vs_{x}":
                return y, x
    return None


def order_blame(res: dict, listed: list[str]) -> dict:
    """{system: [why]} for the pairs that failed the float64 order check. A swap between two
    lists does not say which list is wrong, so blame is assigned in three steps:
    (1) a side whose own list is provably out of order (f64_order_inversions_by_system > 0: a
        later rank beats an earlier one beyond the window) is blamed, the other side is not;
    (2) for the remaining failed pairs, the one system common to all of them is blamed, if some
        other pair passed (3 variants, only v1 wrong: v1_vs_v2 and v1_vs_v3 fail, v2_vs_v3
        passes, so v1);
    (3) otherwise (two systems only, or no single common system) both sides are blamed."""
    failed = {pk: c for pk, c in (res.get("f64_order_failed_pairs") or {}).items() if c}
    if not failed:
        return {}
    pairs_doc = res.get("pairs") or {}
    out: dict = {}
    unresolved = {}
    for pk, c in failed.items():
        xy = _pair_of(pk, listed)
        if xy is None:   # a pair this round does not list: blame everyone, never drop it
            for nm in listed:
                out.setdefault(nm, []).append(f"{c} order disagreements in {pk} (not a pair of "
                                              "the listed systems)")
            continue
        inv = (pairs_doc.get(pk) or {}).get("f64_order_inversions_by_system") or {}
        own = [nm for nm in xy if (inv.get(nm) or 0) > 0]
        if own:
            for nm in own:
                out.setdefault(nm, []).append(f"{c} order disagreements in {pk} ({inv[nm]} "
                                              "inversions in its own list)")
        else:
            unresolved[pk] = (xy, c)
    if unresolved:
        common = set.intersection(*(set(xy) for xy, _ in unresolved.values()))
        passed = [(x, y) for i, x in enumerate(listed) for y in listed[i + 1:]
                  if f"{x}_vs_{y}" not in failed and f"{y}_vs_{x}" not in failed]
        if len(common) == 1 and passed:
            (nm,) = common
            for pk, (xy, c) in unresolved.items():
                out.setdefault(nm, []).append(
                    f"{c} order disagreements in {pk} (in every order-failed pair, while "
                    f"{passed[0][0]}_vs_{passed[0][1]} passed)")
        else:
            for pk, (xy, c) in unresolved.items():
                for nm in xy:
                    out.setdefault(nm, []).append(
                        f"{c} order disagreements in {pk} (ambiguous: blamed on both)")
    return out


def shared_score_blame(res: dict, listed: list[str]) -> dict:
    """{system: [why]} for the pairs whose shared-id scores disagree (compare.judge
    `shared_score_failed_pairs`). Two scores of one id do not say which is wrong: the one
    system common to every failed pair is blamed if some other pair passed (as order_blame's
    step 2), otherwise both sides of each failed pair."""
    failed = {pk: c for pk, c in (res.get("shared_score_failed_pairs") or {}).items() if c}
    if not failed:
        return {}
    out: dict = {}
    pairs = {}
    for pk, c in failed.items():
        xy = _pair_of(pk, listed)
        if xy is None:
            for nm in listed:
                out.setdefault(nm, []).append(f"{c} shared-id score mismatches in {pk} (not a "
                                              "pair of the listed systems)")
        else:
            pairs[pk] = (xy, c)
    if pairs:
        common = set.intersection(*(set(xy) for xy, _ in pairs.values()))
        passed = [(x, y) for i, x in enumerate(listed) for y in listed[i + 1:]
                  if f"{x}_vs_{y}" not in failed and f"{y}_vs_{x}" not in failed]
        for pk, (xy, c) in pairs.items():
            who = sorted(common) if len(common) == 1 and passed else list(xy)
            for nm in who:
                out.setdefault(nm, []).append(f"{c} shared-id score mismatches in {pk}")
    return out


def scheduled_systems(run_dir: str, names: list[str]):
    """A function search -> [systems scheduled in it], read from the run's experiment.json. For
    a search it does not name, the answer is every sweep.json variant (or ``names``)."""
    per = {}
    try:
        exp = json.load(open(os.path.join(run_dir, "experiment.json")))
        paths.use(exp)
        per = {s["name"]: list(s["systems"]) for s in exp["searches"]}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    every = list(names)
    try:
        sw = json.load(open(os.path.join(run_dir, "sweep.json")))
        every = sorted(set(every) | {v["variant"] for v in sw.get("variants") or []})
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return lambda search: per.get(search, every)


def variant_checks(run_dir: str, names: list[str]) -> dict:
    """{variant: (ok, why)}: did its checks pass? A variant fails when checks.json is missing or
    unreadable, its oracle verdict failed in any round, a round that lists it has no oracle
    verdict for it or blames it in the pairs check, it was never checked, or a checks problem
    names it."""
    import re
    try:
        c = json.load(open(os.path.join(run_dir, "checks.json")))
    except (OSError, ValueError) as exc:
        return {n: (False, f"no readable checks.json ({type(exc).__name__})") for n in names}
    seen, bad = {}, {}
    sched = scheduled_systems(run_dir, names)
    for s, per in (c.get("searches") or {}).items():
        for rk, rd in (per or {}).items():
            rd = rd or {}
            if "systems" not in rd:
                # check_round crashed before listing the round's systems: every variant
                # scheduled in this search fails (a missing list never means "ok")
                why = "; ".join(rd.get("fails") or []) or "the round's checks did not run"
                for nm in sched(s):
                    bad.setdefault(nm, []).append(f"{s} {rk}: the round's checks crashed ({why})")
                continue
            orc = rd.get("oracle") or {}
            per_sys = orc.get("systems") or {}
            for nm, v in per_sys.items():
                seen[nm] = True
                if not v.get("ok"):
                    bad.setdefault(nm, []).append(f"{s} {rk} oracle: " + "; ".join(v.get("fails") or []))
            # A listed system with no oracle verdict (the oracle could not run or had no
            # inputs) fails. A failed pairs check is blamed per system (pairs_blame).
            pairs = rd.get("pairs") or {}
            listed = rd.get("systems") or []
            blamed = pairs_blame(pairs, listed) if pairs.get("ok") is not True else {}
            for nm in listed:
                seen.setdefault(nm, False)
                if nm not in per_sys:
                    bad.setdefault(nm, []).append(
                        f"{s} {rk}: no oracle verdict ({'; '.join(orc.get('fails') or []) or 'not run'})")
                if nm in blamed:
                    bad.setdefault(nm, []).append(f"{s} {rk} pairs: {blamed[nm]}")
    out = {}
    for n in names:
        # a name standing alone, or inside a leg tag search__variant__rN
        pat = re.compile(rf"(?:(?<![\w-])|(?<=__)){re.escape(n)}(?:(?![\w-])|(?=__))")
        why = list(bad.get(n, [])) + [p for p in c.get("problems") or [] if pat.search(p)]
        if not seen.get(n):   # never given an oracle verdict in any round
            why.append("never checked (every leg failed or was skipped)")
        out[n] = (not why, "; ".join(why))
    return out


def _md_cell(v) -> str:
    """A value as one Markdown table cell (`|` escaped, line breaks flattened)."""
    return str(v).replace("|", "\\|").replace("\n", " ")


def sweep_table(run_dir: str, exp: dict, variants: list[dict]) -> list[dict]:
    """One row per variant (median, rounds, complete, checks) into sweep_table.csv / .md."""
    rows, problems, excluded, _ = summary_mod.summarize(run_dir, exp)
    ck = variant_checks(run_dir, [v["variant"] for v in variants])
    by = {}
    for r in rows:
        by.setdefault(r["system"], []).append(r)
    R = int(exp["protocol"]["rounds"])
    out = []
    for v in variants:
        rs = by.get(v["variant"], [])
        meds = [r["median_s"] for r in rs if r["median_s"] is not None]
        complete = bool(rs) and all(r["n"] == R for r in rs)
        out.append({"variant": v["variant"], "base_system": v["base_system"],
                    "setting": json.dumps(v["setting"], sort_keys=True),
                    "included": bool(v.get("included")),
                    "median_s": sum(meds) if meds and complete else None,   # summed over searches
                    # one search: its rounds; several: each labelled with its search and
                    # median, joined by "; " (not "|", which splits a Markdown cell)
                    "rounds": rs[0]["rounds_s"] if len(rs) == 1 else "; ".join(
                        f"{r['search']}: median "
                        f"{'-' if r['median_s'] is None else format(r['median_s'], '.4f')}, "
                        f"rounds {r['rounds_s']}" for r in rs),
                    "complete": complete,
                    "checks_ok": ck[v["variant"]][0], "checks_why": ck[v["variant"]][1]})
    with open(os.path.join(run_dir, "sweep_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["variant"])
        w.writeheader()
        w.writerows(paths.mask(out))
    lines = ["| variant | system | setting | median s | per round | complete | checks |",
             "|---|---|---|---|---|---|---|"]
    for r in out:
        med = f"{r['median_s']:.4f}" if r["median_s"] is not None else "-"
        chk = "ok" if r["checks_ok"] else f"FAILED: {r['checks_why']}"
        cells = [r["variant"], r["base_system"], r["setting"], med, r["rounds"], r["complete"], chk]
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    with open(os.path.join(run_dir, "sweep_table.md"), "w") as f:
        f.write(paths.mask("\n".join(lines) + "\n"))
    return out


def cmd_sweep(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bench sweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a sweep file, or (with --pick) a sweep run dir / sweep file")
    ap.add_argument("--pick", action="store_true", help="print the winner + the YAML change")
    _common_args(ap)
    add_launch_id(ap)
    a = ap.parse_args(argv)
    if a.pick:
        return pick(a.target, a.results)
    try:
        exp, variants, sweep_raw = _prepare(a.target, a, sweep=True)
    except experiment_mod.ExperimentError as exc:
        print(f"bench sweep: {exc}", file=sys.stderr)
        return 2
    if a.rounds is not None:
        exp["protocol"]["rounds"] = a.rounds
    rc = _checks_before_run(exp, a)
    if rc:
        return rc
    stem = os.path.splitext(os.path.basename(a.target))[0]
    run_dir = new_run_dir(os.path.join(a.results, "sweeps", stem))
    with open(os.path.join(run_dir, "sweep.json"), "w") as f:
        json.dump(paths.mask({"sweep_file": os.path.abspath(a.target), "base": exp.get("_base"),
                              "sweep": sweep_raw, "variants": variants, "subset": exp.get("subset")}),
                  f, indent=1, sort_keys=True)
    rc = execute(exp, run_dir, ["bench", "sweep", *argv], a.set, a.dev,
                 a.env_check or ("report" if a.dev else "strict"), a.launch_id)
    table = sweep_table(run_dir, exp, variants)
    print(open(os.path.join(run_dir, "sweep_table.md")).read(), end="")
    if events_pairs(variants):
        rows = events_table(run_dir, exp, variants)
        txt = format_events(rows, int(exp["protocol"]["rounds"]))
        with open(os.path.join(run_dir, "events_table.md"), "w") as f:
            f.write(paths.mask(txt))
        with open(os.path.join(run_dir, "events_table.json"), "w") as f:
            json.dump(paths.mask(rows), f, indent=1, sort_keys=True)
        print("\n" + txt, end="")
    return rc if table else 1


# ---- the GPU-events check: a system swept over gpu_events [on, off] (report only) ----------
def events_pairs(variants: list[dict]) -> dict:
    """{base system: {"on": variant, "off": variant}} of the systems swept over gpu_events alone."""
    pairs = {}
    for v in variants:
        if set(v["setting"]) == {"gpu_events"}:
            pairs.setdefault(v["base_system"], {})[v["setting"]["gpu_events"]] = v["variant"]
    return {b: p for b, p in pairs.items() if set(p) == {"on", "off"}}


def _csv_mode(run_dir, tg):
    """The gpu_events value in a leg's timed CSV row; None without one."""
    p = os.path.join(run_dir, "legs", f"{tg}.csv")
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if str(r.get("warmup")) in ("0", "False", "false")]
    return rows[-1].get("gpu_events") if rows else None


def events_table(run_dir: str, exp: dict, variants: list[dict]) -> list[dict]:
    """One row per search and engine: median on / off, difference, spread, worth a look.

    "worth a look" when |diff| > max(EVENTS_MIN_FRAC, spread), where spread is the larger
    round-to-round spread of the two sides. With one run per side there is no spread: the row
    is not decidable and never marked. An engine with no CUDA events on its path (a CPU
    backend: its CSV says gpu_events n/a) is n/a."""
    recs = {}
    for f in os.listdir(os.path.join(run_dir, "legs")):
        if f.endswith(".json") and not f.endswith(".clock.json"):
            try:
                r = json.load(open(os.path.join(run_dir, "legs", f)))
            except ValueError:
                continue
            recs[os.path.splitext(f)[0]] = r
    R = timing.rounds(exp)
    rows = []
    for s in exp["searches"]:
        for base, vv in events_pairs(variants).items():
            if vv["on"] not in s["systems"]:
                continue
            row = {"search": s["name"], "system": base}
            for mode in ("on", "off"):
                secs, modes = [], set()
                for r in range(1, R + 1):
                    tg = timing.tag(s["name"], vv[mode], r)
                    rec = recs.get(tg) or {}
                    if rec.get("ok") and rec.get("timed_seconds") is not None:
                        secs.append(float(rec["timed_seconds"]))
                    modes.add(_csv_mode(run_dir, tg))
                row[f"{mode}_s"] = statistics.median(secs) if secs else None
                row[f"{mode}_n"] = len(secs)
                row[f"{mode}_spread"] = ((max(secs) - min(secs)) / statistics.median(secs)
                                         if len(secs) > 1 and statistics.median(secs) else None)
                row[f"{mode}_mode"] = "/".join(sorted(m or "?" for m in modes))
            on, off = row["on_s"], row["off_s"]
            row["na"] = row["on_mode"] == "n/a" and row["off_mode"] == "n/a"
            row["diff"] = (on - off) / off if on is not None and off else None
            spreads = [x for x in (row["on_spread"], row["off_spread"]) if x is not None]
            row["spread"] = max(spreads) if spreads else None
            row["decidable"] = row["on_n"] > 1 and row["off_n"] > 1
            bar = max(EVENTS_MIN_FRAC, row["spread"] or 0.0)
            row["worth_a_look"] = (not row["na"] and row["decidable"] and row["diff"] is not None
                                   and abs(row["diff"]) > bar)
            rows.append(row)
    return rows


def format_events(rows: list[dict], rounds: int) -> str:
    """The events table as Markdown, with how to read it."""
    def pct(x):
        return "" if x is None else f"{100 * x:+.2f} %"
    lines = [f"GPU events on vs off ({rounds} run(s) each; REPORT ONLY: nothing "
             "is decided or changed)",
             "| search | engine | events | on s | off s | diff (on-off)/off | spread | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = (lambda x: "-" if x is None else f"{x:.4f}")
        ev = "n/a" if r["na"] else f"{r['on_mode']} / {r['off_mode']}"
        sp = "" if r["spread"] is None else f"{100 * r['spread']:.2f} %"
        note = "n/a (no CUDA events on this path)" if r["na"] else (
            "no spread: not decidable" if not r.get("decidable", True) else
            "worth a look" if r["worth_a_look"] else "")
        if r["on_s"] is None or r["off_s"] is None:
            note = "a leg failed (see the summary)"
        lines.append(f"| {r['search']} | {r['system']} | {ev} | {f(r['on_s'])} | {f(r['off_s'])} | "
                     f"{pct(r['diff'])} | {sp} | {note} |")
    lines.append(f"'worth a look' = |diff| > max({100 * EVENTS_MIN_FRAC:.1f} %, spread). With one "
                 "run per side there is no spread: 'no spread: not decidable'; run "
                 "`bench sweep --rounds 3` to judge the difference.")
    lines.append("Order: the sweep file alternates the on / off legs per engine (1st engine on "
                 "then off, 2nd off then on, ...); they then rotate by one per round.")
    return "\n".join(lines) + "\n"


def _latest_run(target: str, results: str) -> str:
    """A run dir as given, or the newest finished run of a sweep file."""
    if os.path.isdir(target):
        return target
    stem = os.path.splitext(os.path.basename(target))[0]
    runs = sorted(glob.glob(os.path.join(results, "sweeps", stem, "*")))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "sweep_table.csv"))]
    if not runs:
        raise FileNotFoundError(f"no finished run of sweep {stem} under {results}/sweeps/{stem}/")
    return runs[-1]


def pick(target: str, results: str = RESULTS) -> int:
    """Print, per swept system, the fastest complete variant and the YAML change; never edits."""
    try:
        run_dir = _latest_run(target, results)
    except FileNotFoundError as exc:
        print(f"bench sweep --pick: {exc}", file=sys.stderr)
        return 2
    sw = json.load(open(os.path.join(run_dir, "sweep.json")))
    exp = json.load(open(os.path.join(run_dir, "experiment.json")))
    paths.use(exp)
    rows = list(csv.DictReader(open(os.path.join(run_dir, "sweep_table.csv"))))
    run_abs, res_abs = os.path.abspath(run_dir), os.path.abspath(results)
    rel = os.path.relpath(run_abs, res_abs) if run_abs.startswith(res_abs + os.sep) else run_abs
    base_file = (sw.get("base") or [None])[-1]
    report_only = events_pairs(sw.get("variants") or [])
    rc = 0
    for base_sys in dict.fromkeys(r["base_system"] for r in rows if r["included"] != "True"):
        if base_sys in report_only:
            print(f"{base_sys}: gpu_events on vs off is report only (events_table.md in "
                  f"{run_dir}): no winner; set gpu_events by hand if needed")
            continue
        mine = [r for r in rows if r["base_system"] == base_sys and r["included"] != "True"]
        for r in mine:   # a variant whose checks failed is never picked
            if r.get("checks_ok") != "True":
                print(f"  {r['variant']}: REFUSED, its checks failed: "
                      f"{r.get('checks_why') or 'no checks verdict in the sweep table'}")
        cand = [r for r in mine if r["complete"] == "True" and r["median_s"] not in ("", None)
                and r.get("checks_ok") == "True"]
        if not cand:
            print(f"{base_sys}: no complete variant with passing checks (every variant failed, "
                  "missed a round or failed its checks)")
            rc = 1
            continue
        cand.sort(key=lambda r: float(r["median_s"]))
        win = cand[0]
        wmed = float(win["median_s"])
        print(f"{base_sys}: winner {win['variant']}  median {wmed:.4f} s  setting {win['setting']}")
        for r in cand[1:]:
            d = (float(r["median_s"]) - wmed) / wmed if wmed else 0.0
            note = "  WITHIN NOISE: re-run the finalists at 3 rounds before picking" if d <= NOISE_FRAC else ""
            print(f"  {r['variant']}: {float(r['median_s']):.4f} s (+{100 * d:.1f} %){note}")
        entry = dict(exp["systems"][win["variant"]])
        knobs = {k: v for k, v in entry["knobs"].items()}
        y = {"system": entry["system"], "preset": entry["preset"], **{
            k: v for k, v in knobs.items()
            if k not in systems_base.SYSTEMS[entry["system"]].presets[entry["preset"]]
            or systems_base.SYSTEMS[entry["system"]].presets[entry["preset"]][k] != v},
            "chosen_from": rel}
        if "config" in y and base_file:
            y["config"] = os.path.relpath(y["config"], os.path.dirname(base_file))
        flow = ", ".join(f"{k}: {_yv(v)}" for k, v in y.items())
        print(f"YAML change ({os.path.relpath(base_file, CB) if base_file else 'the base experiment'}, "
              f"apply by hand and record why):")
        print("  systems:")
        print(f"    {base_sys}: {{{flow}}}")
        if entry.get("io"):   # an io grid's winner changes the io shared by every system
            print(f"  io: {{{', '.join(f'{k}: {_yv(v)}' for k, v in entry['io'].items())}, "
                  f"chosen_from: {rel}}}   # shared by every system: compare the systems' winners")
    return rc


def _yv(v):
    import yaml
    return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
