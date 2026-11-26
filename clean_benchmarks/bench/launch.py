"""Launch experiments or sweeps on a SkyPilot node.

Stages code from a git commit, prepares the node, runs the benchmark, and
syncs results locally. Uncommitted changes require --allow-dirty.

Launches use on-demand instances and restrict S3 writes to the experiment's ${WRITE}.
The experiment's private path values (READ / WRITE, see bench/paths.py) are shipped beside the
staged code, never through git; the launch record keeps the placeholders. So is each optional
local-only directory in LOCAL_DIRS the working tree has (git-ignored); its files' sha256 is
recorded.
Results are matched to their launch ID to avoid collecting stale runs.

--dry-run stages code and generates launch files without starting a node.
--no-wait launches without waiting; --sync retrieves results later.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import re
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import io as _io

from bench import experiment as experiment_mod
from bench import paths as private_paths
from bench.prep import steps as prep
from bench.prep.steps import RES, WORKDIR

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CB)
RESULTS = os.path.join(CB, "results")
NODE_YAML = os.path.join(CB, "env", "node.yaml")
CB_REL = os.path.relpath(CB, REPO)   # "clean_benchmarks"
NOVABF_REL = "python/nova-bf"
TEXTSCAN_PATHS = ["crates/nova-textscan", "Cargo.lock", "Cargo.toml"]
# outputs and build products: not staged from the clean_benchmarks tree
STAGE_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
STAGE_EXCLUDE_SUFFIX = (".so", ".pyc")
STAGE_EXCLUDE_GLOB_DIRS = (".egg-info",)
# local-only dirs (git-ignored) a stage ships when the working tree has them
LOCAL_DIRS = ("s8",)
# the stage's .skyignore (see stage_code): no patterns, so SkyPilot uploads every staged file
SKYIGNORE = "# written by bench launch: upload the whole stage (SkyPilot reads this, not .gitignore)\n"
# code dirs named data/ (code_sha.py skips that name) and the test fixtures, each hashed as
# its own root (must match env/paper_run.sh DATA_CODE_DIRS)
DATA_CODE_DIRS = ("bench/data", "tests/data", "tests/fixtures")
# reads the private path values bench launch ships beside the code (unexported: the legs
# never see them; the prep lines' ${READ} / ${WRITE} are filled in by this shell)
# --move-strays (off by default): the upload never deletes, so a node's ~/sky_workdir can hold
# code files from before it; with the flag they are moved aside first (env/stray_files.py), so
# the node runs and hashes only the stage
STRAY_LINE = ('( . $W/env/activate.sh && python $W/env/stray_files.py $W $W/STAGED_files.txt '
              '~/sky_workdir_stray_$(date -u +%Y%m%dT%H%M%SZ) ) 2>&1 | tee -a ~/res/progress.log')
PATHS_LINE = '. $W/experiments/local_paths.env || { log "FAIL no $W/experiments/local_paths.env (bench launch ships it)"; exit 1; }'
RUN_BEGIN = "# --- run: env/paper_run.sh around bench (begin) ---"
RUN_END = "# --- run (end) ---"


class LaunchError(RuntimeError):
    """A launch refused or failed; `bench launch` prints it and exits 2."""
    pass


def _utc():
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")


def _git(*a, check=True) -> str:
    p = subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=120)
    if check and p.returncode != 0:
        raise LaunchError(f"git {' '.join(a)}: {p.stderr.strip()}")
    return p.stdout


# ---------------------------------------------------------------------------------------------
# per experiment (the `prep:` data steps themselves are in bench/prep/steps.py)
# ---------------------------------------------------------------------------------------------
def _family(exp: dict) -> str:
    """"filtered" for every filtered* experiment, else the experiment's name."""
    n = exp["name"]
    if n.startswith("filtered"):
        return "filtered"
    return n


# ---------------------------------------------------------------------------------------------
# the node and the task
# ---------------------------------------------------------------------------------------------
def load_node() -> dict:
    """env/node.yaml, refused unless it asks for on-demand instances only."""
    import yaml
    node = yaml.safe_load(open(NODE_YAML))
    res = node.get("resources") or {}
    if res.get("use_spot") is not False:
        raise LaunchError(f"{NODE_YAML}: resources.use_spot must be false (on-demand only)")
    if any(k in res for k in ("spot_recovery", "job_recovery")):
        raise LaunchError(f"{NODE_YAML}: spot recovery keys are not allowed (on-demand only)")
    return node


def cluster_name(exp: dict) -> str:
    """The default SkyPilot cluster name, bench-<experiment>."""
    return "bench-" + exp["name"].replace("_", "-").lower()


def build_lines(exp: dict) -> list[str]:
    """The build step: rustup if there is no cargo, then the staged nova-bf, rust_io and
    bench into the paper env with --no-deps; with prep.textscan also the staged nova_textscan."""
    L = ['log "STEP build (staged nova-bf + rust_fetch + bench into the paper env, --no-deps)"',
         "command -v cargo >/dev/null || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q; }",
         '. "$HOME/.cargo/env"',
         't0=$(date +%s)',
         f'( . $W/env/activate.sh && python -m pip install --no-input --no-deps -e "$W/nova-bf" \\',
         '    "$W/nova-bf/rust_io" && python -m pip install --no-input --no-deps -e "$W" ) \\',
         f'    > {RES}/install.log 2>&1 \\',
         '  || { log "FAIL install into the paper env (see install.log)"; tail -30 '
         f'{RES}/install.log; exit 1; }}']
    L += prep.textscan_lines(exp)
    L.append('log "build $(( $(date +%s) - t0 ))s"')
    return L


LAUNCH_ID_RE = re.compile(r"\d{8}T\d{6}Z_(?:[0-9a-f]{7,40}|nogit)(?:-\d+)?")   # launch()'s ids


def run_line(exp_rel: str, job: str, sets: list[str], launch_id: str | None = None) -> str:
    """The job command: the stage's paper_run.sh around `python -m bench`.
    `--launch-id ID` (launch() always passes one) is an argument of bench run / bench sweep,
    never an environment variable: the run's manifest records it, sync matches it, and nothing
    reaches the environment of paper_run.sh or the legs."""
    setargs = "".join(f" --set {shlex.quote(a)}" for a in sets)
    lid = ""
    if launch_id is not None:
        if not LAUNCH_ID_RE.fullmatch(launch_id):
            raise LaunchError(f"launch id {launch_id!r}: not <UTC>_<sha>[-N]")
        lid = f" --launch-id {launch_id}"
    cmd = "sweep" if job == "sweep" else "run"
    return (f"cd {WORKDIR} && bash env/paper_run.sh --run-dir {RES}/paper_run_$(date -u +%Y%m%dT%H%M%SZ) "
            f"python -m bench {cmd} {exp_rel}{lid} --results {RES}{setargs} 2>&1 | tee ~/run.log")


def run_block(exp: dict, exp_rel: str, job: str, sets: list[str], launch_id: str | None = None,
              move_strays: bool = False) -> str:
    """The task's `run:` script: setup, build, node prep, then the job command between markers."""
    L = ["set -uo pipefail",
         f"W={WORKDIR}; mkdir -p {RES} ~/data; cd $W",
         f'log() {{ echo "[$(date -u +%FT%TZ)] $*" | tee -a {RES}/progress.log; }}',
         PATHS_LINE,
         *([STRAY_LINE] if move_strays else []),
         *build_lines(exp),
         *prep.node_prep(exp),
         RUN_BEGIN,
         run_line(exp_rel, job, sets, launch_id),
         RUN_END]
    return "\n".join(L) + "\n"


def extract_run_line(task_yaml_text: str) -> str:
    """The job command between the RUN markers of a generated task."""
    import yaml
    run = yaml.safe_load(task_yaml_text)["run"]
    lines = run.splitlines()
    i, j = lines.index(RUN_BEGIN), lines.index(RUN_END)
    return "\n".join(lines[i + 1:j])


def task(exp: dict, exp_rel: str, job: str, sets: list[str], launch_id: str | None = None,
         move_strays: bool = False) -> dict:
    """The SkyPilot task: env/node.yaml's keys plus the name and the generated run block."""
    node = load_node()
    t = {"name": cluster_name(exp), "resources": copy.deepcopy(node["resources"])}
    if node.get("secrets"):
        t["secrets"] = copy.deepcopy(node["secrets"])
    for k in ("file_mounts", "envs", "num_nodes", "setup"):
        if node.get(k) is not None:
            t[k] = copy.deepcopy(node[k])
    t["run"] = run_block(exp, exp_rel, job, sets, launch_id, move_strays)
    return t


def dump_task(t: dict) -> str:
    """The task as YAML, multi-line strings as `|` blocks, under a do-not-edit header."""
    import yaml

    class _D(yaml.SafeDumper):
        pass

    def _str(d, s):
        return d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)
    _D.add_representer(str, _str)
    head = ("# generated by `bench launch` (bench/launch.py) from env/node.yaml + the experiment;\n"
            "# do not edit: re-run `bench launch --dry-run`\n")
    return head + yaml.dump(t, Dumper=_D, sort_keys=False, default_flow_style=False, width=200)


def sky_command(task_path: str, cluster: str, stage: str, node: dict) -> list[str]:
    cmd = ["sky", "launch", "-d", "-y", "-c", cluster, task_path, "--workdir", stage]
    for s in (node.get("secrets") or {}):
        cmd += ["--secret", s]
    return cmd


# ---------------------------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------------------------
def shipped_paths(exp: dict) -> list[str]:
    """The repo paths a stage ships for this experiment."""
    p = [CB_REL, NOVABF_REL]
    if _family(exp) == "filtered":
        p += TEXTSCAN_PATHS
    return p


def dirty_paths(paths: list[str]) -> list[str]:
    """`git status --porcelain` lines for those paths, minus clean_benchmarks/results."""
    spec = [*paths, f":(exclude){CB_REL}/results"]
    out = _git("status", "--porcelain", "--", *spec)
    return [x for x in out.splitlines() if x.strip()]


def _keep(rel: str) -> bool:
    """False for caches and build products (STAGE_EXCLUDE_*)."""
    parts = rel.split("/")
    if any(p in STAGE_EXCLUDE_DIRS or p.endswith(STAGE_EXCLUDE_GLOB_DIRS) for p in parts[:-1]):
        return False
    return not rel.endswith(STAGE_EXCLUDE_SUFFIX)


def _place(rel: str) -> str | None:
    """The stage path of a repo path; None when it is not staged."""
    if rel.startswith(CB_REL + "/"):
        sub = rel[len(CB_REL) + 1:]
        if sub.split("/")[0] == "results":
            return None
        return sub
    if rel.startswith(NOVABF_REL + "/"):
        return "nova-bf/" + rel[len(NOVABF_REL) + 1:]
    if rel.startswith("crates/nova-textscan/"):
        return "nova_textscan/" + rel[len("crates/nova-textscan/"):]
    if rel == "Cargo.lock":
        return "nova_textscan/Cargo.lock"
    if rel == "Cargo.toml":
        return None   # only its [profile.release-textscan] is used (appended below)
    return None


def _textscan_profile(cargo_toml: str) -> str:
    """The workspace root's [profile.release-textscan] section, to copy into the staged crate."""
    out, on = [], False
    for line in cargo_toml.splitlines():
        if line.strip() == "[profile.release-textscan]":
            on = True
            out.append(line)
            continue
        if line.startswith("["):
            on = False
        if on:
            out.append(line)
    if not out:
        raise LaunchError("no [profile.release-textscan] in Cargo.toml")
    return "\n".join(out)


def stage_code(exp: dict, stage: str, rev: str, allow_dirty: bool) -> dict:
    """Write the stage's code; returns {"commit", "rev", "dirty", "dirty_paths", "source"}."""
    paths = shipped_paths(exp)
    commit = _git("rev-parse", rev).strip()
    dirty = dirty_paths(paths)
    if dirty and not allow_dirty:
        shown = "\n".join("    " + d for d in dirty[:40])
        more = f"\n    ... ({len(dirty)} paths)" if len(dirty) > 40 else ""
        raise LaunchError(
            f"REFUSING: {' '.join(paths)} has uncommitted changes, and the stage ships the code "
            f"from git ({rev}):\n{shown}{more}\n  Commit them (then re-run), or pass --allow-dirty "
            "to stage the working tree as it is (recorded in the manifest).")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    cargo_toml = None
    if not dirty:   # git archive REV: the committed tree
        data = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", commit, *paths],
                              capture_output=True, timeout=600)
        if data.returncode != 0:
            raise LaunchError(f"git archive: {data.stderr.decode()[-500:]}")
        with tarfile.open(fileobj=_io.BytesIO(data.stdout)) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.name == "Cargo.toml":
                    cargo_toml = tf.extractfile(m).read().decode()
                dst = _place(m.name)
                if dst is None or not _keep(dst):
                    continue
                p = os.path.join(stage, dst)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(tf.extractfile(m).read())
                os.chmod(p, m.mode & 0o777)
        source = f"git archive {commit}"
    else:           # --allow-dirty: the working tree (tracked + untracked, not ignored)
        files = _git("ls-files", "-co", "--exclude-standard", "--", *paths).splitlines()
        for rel in files:
            src = os.path.join(REPO, rel)
            if not os.path.isfile(src):
                continue   # deleted in the working tree
            if rel == "Cargo.toml":
                cargo_toml = open(src).read()
            dst = _place(rel)
            if dst is None or not _keep(dst):
                continue
            p = os.path.join(stage, dst)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy2(src, p)
        source = "working tree (--allow-dirty)"
    if _family(exp) == "filtered":
        if cargo_toml is None:
            raise LaunchError("Cargo.toml not staged (needed for nova_textscan's profile)")
        with open(os.path.join(stage, "nova_textscan", "Cargo.toml"), "a") as f:
            f.write("\n# staged standalone (outside the workspace): the workspace root's profile, "
                    "copied\n" + _textscan_profile(cargo_toml) + "\n")
    check_patches(stage)
    # the private path values, beside the code (never through git; not in the code hash)
    with open(os.path.join(stage, os.path.relpath(private_paths.FILE, private_paths.CB)), "w") as f:
        f.write(private_paths.file_text(private_paths.values()))
    # SkyPilot uploads the workdir minus what its .gitignore files exclude, and the stage's own
    # .gitignore excludes the paths file just written. A .skyignore in the workdir replaces the
    # .gitignore files: this one excludes nothing, so the whole stage reaches the node
    with open(os.path.join(stage, ".skyignore"), "w") as f:
        f.write(SKYIGNORE)
    local = {d: ship_local_dir(stage, d) for d in LOCAL_DIRS}
    return {"commit": commit, "rev": rev, "dirty": bool(dirty), "dirty_paths": dirty, "source": source,
            "local_dirs": {d: v for d, v in local.items() if v}}


def ship_local_dir(stage: str, rel: str) -> dict | None:
    """Copy clean_benchmarks/rel (a git-ignored local dir) into the stage, as it is in the
    working tree. Returns {"files": {path: sha256}, "sha256": over the sorted "sha256  path"
    lines}, or None when the working tree has no such dir (a public clone)."""
    root = os.path.join(REPO, CB_REL)
    src = os.path.join(root, rel)
    if not os.path.isdir(src):
        return None
    files = {}
    for d, dirs, names in os.walk(src):
        dirs[:] = sorted(x for x in dirs if x not in STAGE_EXCLUDE_DIRS)
        for n in sorted(names):
            r = os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")
            if not os.path.isfile(os.path.join(d, n)) or not _keep(r):
                continue
            p = os.path.join(stage, r)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy2(os.path.join(d, n), p)
            files[r] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    text = "".join(f"{h}  {r}\n" for r, h in sorted(files.items()))
    return {"files": files, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def check_patches(stage: str):
    """Check every CUVS_PATCHES file is staged with versions.env's CUVS_PATCH_SHA256."""
    env = parse_env(os.path.join(stage, "env", "versions.env"))
    pins = dict(x.split("=", 1) for x in env.get("CUVS_PATCH_SHA256", "").split())
    for p in env.get("CUVS_PATCHES", "").split():
        f = os.path.join(stage, "baselines", "cuvs", "patches", p)
        if not os.path.isfile(f):
            raise LaunchError(f"{f} not staged")
        if hashlib.sha256(open(f, "rb").read()).hexdigest() != pins.get(p):
            raise LaunchError(f"{p} sha256 is not versions.env's CUVS_PATCH_SHA256 pin ({pins.get(p)})")


def parse_env(path: str) -> dict:
    """KEY=VALUE lines of a versions.env-style file, comments and quotes stripped."""
    out = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def stack_sha(versions_env: str) -> str:
    """PAPER_STACK_SHA as paper_run.sh / activate.sh compute it: sha256 of versions.env without
    STACK_FROZEN_ON, comments and blank lines (`grep -v | sed | grep -v | sha256sum`)."""
    import re
    lines = []
    for line in open(versions_env).read().splitlines():
        if line.startswith("STACK_FROZEN_ON="):
            continue
        line = re.sub(r"[ \t]*#.*$", "", line)
        if not line.strip():
            continue
        lines.append(line + "\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def code_sha(stage: str) -> str:
    """The stage's code sha256, as env/paper_run.sh's stage branch computes it (code_sha.py
    --base STAGE over the stage and its DATA_CODE_DIRS)."""
    roots = [stage] + [os.path.join(stage, d) for d in DATA_CODE_DIRS
                       if os.path.isdir(os.path.join(stage, d))]
    p = subprocess.run([sys.executable, os.path.join(stage, "env", "code_sha.py"), "--base", stage,
                        "--json", os.path.join(stage, "STAGED_code.json"), *roots],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise LaunchError(f"code_sha.py: {p.stderr[-500:]}")
    return p.stdout.strip()


def staged_files(stage: str) -> list[str]:
    """Every file in the stage, relative and sorted."""
    out = []
    for d, dirs, files in os.walk(stage):
        dirs.sort()
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(d, f), stage))
    return out


# ---------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------
def _manifest(run_dir: str) -> dict:
    """A run's manifest.json, or {} when it cannot be read."""
    try:
        return json.load(open(os.path.join(run_dir, "manifest.json")))
    except (OSError, ValueError):
        return {}


def run_verdict(run_dir: str) -> list[str]:
    """What failed in a synced run (empty means it passed). Its checks.json must say ok and its
    summary.md must list no PROBLEMS: the same rule as bench run's exit code (a failed leg, a
    missing round, a failed check)."""
    out = []
    try:
        c = json.load(open(os.path.join(run_dir, "checks.json")))
    except (OSError, ValueError) as exc:
        out.append(f"no readable checks.json ({type(exc).__name__})")
    else:
        if c.get("ok") is not True:
            out.append("checks failed: " + "; ".join((c.get("problems") or ["ok is not true"])[:5]))
    try:
        md = open(os.path.join(run_dir, "summary.md")).read()
    except OSError:
        out.append("no summary.md (the run did not finish)")
    else:
        if "\nPROBLEMS (" in "\n" + md:
            # only the PROBLEMS section: report/summary.py writes its "- " items right under
            # the header, and the section ends at the first other line (before e.g. the
            # "Warnings (N, not failures):" section, which are not problems)
            tail = ("\n" + md).split("\nPROBLEMS (", 1)[1].splitlines()[1:]
            items = []
            for x in tail:
                if not x.startswith("- "):
                    break
                items.append(f"summary: {x[2:]}")
            out += items[:10] or ["summary: a PROBLEMS section with no items"]
    return out


def sweep_verdict(run_dir: str) -> list[str]:
    """What failed in a synced sweep run: run_verdict's rule, plus its sweep_table.csv."""
    out = run_verdict(run_dir)
    if not os.path.isfile(os.path.join(run_dir, "sweep_table.csv")):
        out.append("no sweep_table.csv (the sweep did not finish)")
    return out


def results_rel(rec: dict) -> str:
    """Where a launch's runs go under the results root (and ~/res): <exp>, or sweeps/<sweep>."""
    return rec.get("results_dir") or rec["experiment"]


def sync(launch_dir: str, runner=subprocess.call) -> int:
    """rsync the node's results back and link each run of this launch to its launch record.
    Returns non-zero when an rsync failed or the synced run failed.

    A run launch's results are in ~/res/<exp>/, a `--job sweep` launch's in
    ~/res/sweeps/<sweep file stem>/ (results_rel). A run belongs to this launch only when its
    manifest's `launch_id` matches (the run line passes `--launch-id`). Any other run, such as an earlier launch's on a reused
    cluster, is `not_this_launch`: never stamped, and it does not count as arrived."""
    rec = json.load(open(os.path.join(launch_dir, "launch.json")))
    cl = rec["cluster"]
    job = rec.get("job", "run")
    lid = rec.get("launch_id")
    rel = results_rel(rec)
    node_dir = os.path.join(launch_dir, "node")
    os.makedirs(node_dir, exist_ok=True)
    # the runs go where a `chosen_from:` points (relative to the results root)
    dest = os.path.join(os.path.expanduser(rec["results_root"]), *rel.split("/"))
    src = f"{cl}:{RES}/{rel}/"
    os.makedirs(dest, exist_ok=True)
    before = set(os.listdir(dest))
    cmds = [["rsync", "-a", src, dest + "/"],
            ["rsync", "-a", "--exclude", f"/{rel}/", f"{cl}:{RES}/", os.path.join(node_dir, "res") + "/"],
            ["rsync", "-a", f"{cl}:run.log", os.path.join(node_dir, "run.log")]]
    rcs = [runner(c) for c in cmds]
    runs = sorted(d for d in os.listdir(dest) if d != "launch" and
                  os.path.isfile(os.path.join(dest, d, "manifest.json")))
    # only a run whose manifest carries this launch's id is linked to it; an older run (local,
    # or still on a reused cluster) is never stamped with this launch
    mine, others = [], []
    for d in runs:
        (mine if lid and _manifest(os.path.join(dest, d)).get("launch_id") == lid
         else others).append(d)
    for d in mine:
        p = os.path.join(dest, d, "launch.json")
        if not os.path.exists(p):
            shutil.copy2(os.path.join(launch_dir, "launch.json"), p)
    problems = [f"rsync exit {r}: {' '.join(c)}" for c, r in zip(cmds, rcs) if r != 0]
    if not lid:
        problems.append("launch.json has no launch_id: no run can be matched to this launch")
    what = f"bench {job}"
    if not mine:
        problems.append(f"no run of this launch ({lid}) arrived in {dest} (the job died before "
                        f"`{what}` wrote one? see {os.path.join(node_dir, 'run.log')})"
                        + (f"; not this launch's: {', '.join(others)}" if others else ""))
    verdict = sweep_verdict if job == "sweep" else run_verdict
    for d in mine:
        problems += [f"{d}: {x}" for x in verdict(os.path.join(dest, d))]
    with open(os.path.join(launch_dir, "synced.json"), "w") as f:
        json.dump({"rsync": cmds, "exit": rcs, "launch_id": lid, "runs": mine,
                   "new": [d for d in mine if d not in before],
                   "not_this_launch": others, "problems": problems}, f, indent=1)
    print(f"bench launch: synced into {dest} (rsync exits {rcs}); tear down with `sky down {cl}`")
    if problems:
        print(f"bench launch: the launched run FAILED ({len(problems)} problem(s)):", file=sys.stderr)
        for x in problems:
            print(f"  - {x}", file=sys.stderr)
        return 1
    return 0


def _home_masked(x):
    """A launch record with this machine's home directory written as ~ (the record is tracked)."""
    home = os.path.expanduser("~").rstrip("/")
    if not home or home == "/":
        return x
    if isinstance(x, str):
        return re.sub(re.escape(home) + r"(?=/|$)", "~", x)
    if isinstance(x, dict):
        return {k: _home_masked(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_home_masked(v) for v in x]
    return x


# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    """`bench launch` argument parsing; the work is in launch() and sync()."""
    import argparse
    ap = argparse.ArgumentParser(prog="bench launch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?")
    ap.add_argument("--sync", metavar="LAUNCH_DIR", help="rsync a launch's results back")
    ap.add_argument("--rev", default="HEAD", help="the commit staged (default HEAD)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="stage the working tree although it has uncommitted changes (recorded)")
    ap.add_argument("--job", choices=["run", "sweep"], default="run",
                    help="what the node runs under paper_run.sh: bench run EXPERIMENT (default) "
                         "or bench sweep SWEEP")
    ap.add_argument("--cluster", default=None, help="SkyPilot cluster name (default bench-<exp>)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="passed to bench run / sweep on the node (and validated here)")
    ap.add_argument("--data-src", action="append", default=[], metavar="KEY=PATH",
                    help="override a staged data source (a prep.stage entry's key)")
    ap.add_argument("--queries-src", default=None,
                    help="= --set prep.queries_src=URI (sparse / multivector: fetched to queries.path)")
    ap.add_argument("--textscan", action="store_true",
                    help="= --set prep.textscan=true (filtered: also build the staged nova_textscan)")
    ap.add_argument("--stage", default=None, help="stage dir (default <launch dir>/stage)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--dry-run", action="store_true", help="write the task + file list, launch nothing")
    ap.add_argument("--no-wait", action="store_true", help="launch detached and return (sync later)")
    ap.add_argument("--move-strays", action="store_true",
                    help="first move code files in the node's workdir that the stage does not list "
                         "to ~/sky_workdir_stray_<UTC>/ (env/stray_files.py; nothing is deleted)")
    a = ap.parse_args(argv)
    if a.sync:
        return sync(a.sync)
    if not a.experiment:
        ap.error("an experiment file (or --sync LAUNCH_DIR)")
    # these flags are experiment values (prep:), passed on as --set so the node's `bench run`
    # gets them too and the manifest records them
    if a.queries_src is not None:
        a.set.append(f"prep.queries_src={a.queries_src}")
    if a.textscan:
        a.set.append("prep.textscan=true")
    try:
        return launch(a)
    except (LaunchError, prep.PrepError, experiment_mod.ExperimentError) as exc:
        print(f"bench launch: {exc}", file=sys.stderr)
        return 2


def launch(a) -> int:
    """Validate, stage, write the launch record and task, then `sky launch` (unless --dry-run)."""
    from bench import runner
    path = os.path.abspath(a.experiment)
    if not path.startswith(CB + os.sep):
        raise LaunchError(f"{a.experiment}: the experiment must be a file of this clean_benchmarks tree")
    exp_rel = os.path.relpath(path, CB)
    raw = runner._set(experiment_mod.load_raw(path), a.set)
    if a.job == "sweep":
        if "sweep" not in raw:
            raise LaunchError(f"{exp_rel}: not a sweep file (--job sweep launches one)")
        exp = experiment_mod.finish(experiment_mod.sweep_variants(raw, path)[0], path)
    else:
        if "sweep" in raw:
            raise LaunchError(f"{exp_rel}: a sweep file (launch it with --job sweep)")
        exp = experiment_mod.finish(raw, path)
    ph = experiment_mod.placeholders(exp)
    if ph:
        raise LaunchError("placeholder value(s) (TODO): " + "; ".join(ph) + runner.io_hint(exp, ph))
    try:
        private_paths.check([exp, a.set])
        out = private_paths.expand(exp.get("novabf_output"))
    except private_paths.PathsError as exc:
        raise LaunchError(str(exc)) from None
    if out and out.startswith("s3://"):
        safe = private_paths.values().get("WRITE")
        dest = out.split("?", 1)[0].rstrip("/")
        if not safe or not (dest == safe or dest.startswith(safe + "/")):
            shown = private_paths.mask(exp["novabf_output"])
            why = "" if safe else f" (WRITE is not set for this experiment: see {private_paths.FILE})"
            raise LaunchError(f"novabf_output {shown}: S3 writes go under the experiment's ${{WRITE}} only{why}")
    keys = [d["key"] for d in exp["prep"]["stage"]]
    srcs = {}
    for kv in a.data_src:
        k, _, v = kv.partition("=")
        if k not in keys:
            raise LaunchError(f"--data-src {k}: one of {sorted(keys)} (this experiment's prep.stage keys)")
        srcs[k] = os.path.abspath(v)
    if a.allow_dirty and _git("rev-parse", a.rev).strip() != _git("rev-parse", "HEAD").strip():
        # --allow-dirty stages the working tree (HEAD plus the edits), never REV, so a launch
        # that says --rev REV must not ship another commit's code
        raise LaunchError(f"--allow-dirty stages the working tree (HEAD + its changes); --rev {a.rev} "
                          "is not HEAD. Check REV out first, or drop --allow-dirty.")
    node = load_node()
    sha = _git("rev-parse", "--short", a.rev).strip()
    dirty = dirty_paths(shipped_paths(exp))
    if dirty and not a.allow_dirty:   # refuse before anything is written
        stage_code(exp, None, a.rev, False)
    launch_id = f"{_utc()}_{sha}"
    # where bench run / bench sweep put the runs (and this launch's record)
    rel = (exp["name"] if a.job == "run" else
           "sweeps/" + os.path.splitext(os.path.basename(path))[0])
    launch_dir = os.path.join(a.results, *rel.split("/"), "launch", launch_id)
    n = 1
    while os.path.exists(launch_dir):
        n += 1
        launch_dir = os.path.join(a.results, *rel.split("/"), "launch", f"{launch_id}-{n}")
    launch_id = os.path.basename(launch_dir)
    os.makedirs(launch_dir)
    stage = os.path.abspath(a.stage or os.path.join(launch_dir, "stage"))
    st = stage_code(exp, stage, a.rev, a.allow_dirty)
    plan = prep.data_plan(exp, srcs)
    data_lines = [f"data {d['path']}: WOULD {d['how']} {d['source']} (dry run: not copied)"
                  for d in plan] if a.dry_run else prep.stage_data(stage, plan)
    csha = code_sha(stage)
    staged = [f"staged {_utc()} commit={st['commit']} ({a.rev}) from {st['source']}",
              f"code sha256: {csha}", f"dirty: {'yes' if st['dirty'] else 'no'}",
              *[f"  uncommitted (STAGED AS IS, --allow-dirty): {d}" for d in st["dirty_paths"]],
              *[f"local {d}/ (not from git): sha256 {v['sha256']} ({len(v['files'])} files)"
                for d, v in sorted(st["local_dirs"].items())],
              *data_lines]
    with open(os.path.join(stage, "STAGED"), "w") as f:
        f.write(_home_masked("\n".join(staged) + "\n"))
    shutil.copy2(os.path.join(stage, "STAGED"), os.path.join(launch_dir, "STAGED"))
    t = task(exp, exp_rel, a.job, a.set, launch_id, a.move_strays)
    task_path = os.path.join(launch_dir, "task.yaml")
    with open(task_path, "w") as f:
        f.write(dump_task(t))
    # the stage's own file list, shipped with it: env/stray_files.py on the node moves aside any
    # code file in the workdir that it does not name (not code itself: .txt)
    files = sorted(staged_files(stage) + ["STAGED_files.txt"])
    with open(os.path.join(stage, "STAGED_files.txt"), "w") as f:
        f.write("\n".join(files) + "\n")
    with open(os.path.join(launch_dir, "staged_files.txt"), "w") as f:
        f.write("\n".join(files + [f"{d['path']}  (data: {d['how']} of {d['source']})" for d in plan]
                          if a.dry_run else files) + "\n")
    cluster = a.cluster or cluster_name(exp)
    cmd = sky_command(task_path, cluster, stage, node)
    with open(os.path.join(launch_dir, "sky_command.txt"), "w") as f:
        f.write(shlex.join(cmd) + "\n")
    rec = {"schema": 1, "experiment": exp["name"], "experiment_file": exp_rel,
           "experiment_text": open(path).read(), "sets": a.set, "job": a.job, "results_dir": rel,
           "commit": st["commit"], "rev": a.rev, "dirty": st["dirty"], "dirty_paths": st["dirty_paths"],
           "allow_dirty": a.allow_dirty, "move_strays": a.move_strays,
           "stage_source": st["source"], "stage": stage,
           "code_sha": csha, "stack_sha": stack_sha(os.path.join(stage, "env", "versions.env")),
           "node": node, "cluster": cluster, "sky_command": cmd, "run_line": run_line(exp_rel, a.job, a.set, launch_id),
           "data": plan, "prep": exp["prep"], "results_root": os.path.abspath(a.results), "dry_run": a.dry_run,
           "launch_id": launch_id, "paths": private_paths.record([exp, a.set]),
           "local_dirs": st["local_dirs"]}
    with open(os.path.join(launch_dir, "launch.json"), "w") as f:
        json.dump(_home_masked(private_paths.mask(rec)), f, indent=1, sort_keys=True)
    print(f"bench launch: {len(files)} staged files, code sha256 {csha}, commit {st['commit'][:12]}"
          + (" (DIRTY: working tree staged)" if st["dirty"] else ""))
    print(f"  task {task_path}\n  run: {rec['run_line']}\n  {shlex.join(cmd)}")
    if a.dry_run:
        print(f"bench launch --dry-run: nothing launched ({launch_dir})")
        return 0
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"bench launch: sky launch exit {rc}", file=sys.stderr)
        return rc
    if a.no_wait:
        print(f"bench launch: detached; then `bench launch --sync {launch_dir}`")
        return 0
    subprocess.call(["sky", "logs", cluster])   # follows the job until it ends
    return sync(launch_dir)


if __name__ == "__main__":
    sys.exit(main())
