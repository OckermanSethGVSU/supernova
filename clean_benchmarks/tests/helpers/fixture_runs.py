"""`bench run --dev` of the four fixture experiments, built once per machine and shared.

Several test modules read finished runs of tests/fixtures/experiments/{dense,sparse,multivector,
filtered}.yaml (each a few minutes of real legs). They are built once into a cache under
~/.cache/bench_test_runs/<key>/ and every module gets its own COPY (`runs(tmp_path_factory, ...)`),
so a test may change its run dirs freely and never touches the cache.

The key is a content hash of everything that decides the outputs (`cache_key`): this file, the
code a run executes (bench/, baselines/, env/, nova-bf's source as `nova_bf` is found, the
installed rust_fetch module), the fixture experiments, configs and data, the Python executable,
its version and key package versions, the clean_benchmarks path (leg records hold it) and the
environment variables the code can read. Any change gives a new key, so a cached run is reused
only while nothing it depends on has changed. Not in the key: the git commit (manifest.json and
the run id record the commit the cache was built at; no test compares it with HEAD).

Concurrency: an fcntl lock on ~/.cache/bench_test_runs/.lock. Readers copy under a shared lock;
the first process that finds no cache takes the exclusive lock, builds into a `.build-*` dir
beside it and renames it into place (a crashed or interrupted build never looks complete),
then prunes. xdist workers and parallel pytest sessions build at most once per key.

Every shape's `bench run` must exit 0 (its checks pass). A failed build (any other exit) is NOT
cached under the key: a failure may be
transient (an OOM kill, a busy box) and must not fail later sessions without a retry. Within one
xdist session the workers share it (`.failed-<key>-<session>`), so they all see the same failure
without building six times; the next session builds again.

The copy is made self-consistent, as if `bench run` had written to it: the build dir's absolute
path is replaced by the copy's in every text file and in the recorded stdout / stderr, and
checks.json's input hashes are re-taken for the rewritten files (only when they matched the
build, checked at build time; a stale record stays stale).

Environment:
  BENCH_TEST_RUNS_REBUILD=1   rebuild once in this session (xdist workers share the rebuild)
  BENCH_TEST_RUNS_CACHE=0     no cache: each module builds into its own tmp dir, as before
The 3 most recently used keys are kept; older keys, leftover `.build-*` dirs and other
sessions' `.failed-*` dirs are removed when a new key is built.
"""

import contextlib
import fcntl
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(CB)
FIX = os.path.join(CB, "tests", "fixtures", "experiments")
NBFIX = os.path.join(CB, "tests", "fixtures", "novabf")
SHAPES = ("dense", "sparse", "multivector", "filtered")
ROOT = os.path.join(os.path.expanduser("~"), ".cache", "bench_test_runs")
KEEP = 3

PACKAGES = ("numpy", "pyarrow", "torch", "faiss-cpu", "faiss-gpu", "faiss", "duckdb", "scipy",
            "pyyaml", "pandas", "triton", "rust-fetch", "nova-bf", "cupy-cuda12x", "cuvs-cu12")
# env vars that reach the run but are named by prefix (not as a literal in the code)
ENV_PREFIXES = ("HARNESS_", "BENCH_", "ARROW_", "OMP_", "MKL_", "CUDA_", "CUBLAS_", "TORCH_",
                "TRITON_", "KMP_")
ENV_ALWAYS = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "HOME", "TMPDIR", "CONDA_PREFIX")
# never in the key: rotating credentials (the fixture data is local), pytest's own and this
# cache's switches (a forced rebuild makes the same key)
ENV_NEVER = ("AWS_", "PYTEST_", "BENCH_TEST_RUNS_")
_ENV_LITERAL = re.compile(rb"""["']([A-Z][A-Z0-9_]{2,})["']""")
_TEXT = (".json", ".log", ".md", ".csv", ".txt", ".yaml", ".yml", ".env")


# the environment as the session started (tests/conftest.py imports this module first): bench code
# run in-process by other tests sets os.environ (pylate_engine: PYLATE_SCORES_BACKEND), which would
# otherwise change the key, and the build, by which worker builds
_ENVIRON = dict(os.environ)


def _env():
    env = {k: v for k, v in _ENVIRON.items() if not k.startswith(("NOVA_BF_", "NOVABF_", "PAPER_"))}
    env.update(HARNESS_NO_IMDS="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    return env


class Proc:
    """What the tests read of a `bench run` CompletedProcess."""

    def __init__(self, args, returncode, stdout, stderr):
        self.args, self.returncode, self.stdout, self.stderr = args, returncode, stdout, stderr


def _bench_run(shape, res):
    args = [sys.executable, "-m", "bench", "run", os.path.join(FIX, f"{shape}.yaml"), "--dev",
            "--results", res]
    p = subprocess.run(args, cwd=CB, env=_env(), capture_output=True, text=True)
    return Proc(args, p.returncode, p.stdout, p.stderr)


def _only_run(results, name):
    runs = sorted(os.listdir(os.path.join(results, name)))
    assert len(runs) == 1, runs
    return os.path.join(results, name, runs[0])


# ---------------------------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------------------------
def _files(root):
    if os.path.isfile(root):
        return [root]
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", ".pytest_cache")
                         and not x.endswith(".egg-info"))
        out += [os.path.join(d, f) for f in files if not f.endswith((".pyc", ".lock"))
                or f.endswith(".lock.json")]
    return sorted(out)


def _origin_dir(module):
    spec = importlib.util.find_spec(module)   # finds, does not import
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    return spec.origin


def _roots():
    nb = _origin_dir("nova_bf")
    return [os.path.abspath(__file__), os.path.join(CB, "bench"), os.path.join(CB, "baselines"),
            os.path.join(CB, "env"), os.path.join(CB, "pyproject.toml"),
            os.path.join(CB, "experiments", "local_paths.env"), FIX, NBFIX,
            *([nb] if nb else []), *([r] if (r := _origin_dir("rust_fetch")) else [])]


def cache_key(roots=None, environ=None):
    """(key, the hashed description): sha256 of the files under `roots` (path + content), the
    interpreter, the package versions and the env vars the code can read."""
    names = set()
    roots = _roots() if roots is None else roots
    trees = {}
    for root in roots:
        h = hashlib.sha256()
        files = _files(root) if os.path.exists(root) else []
        for p in files:
            data = open(p, "rb").read()
            h.update(b"\0f\0" + os.path.relpath(p, root).encode() + b"\0"
                     + hashlib.sha256(data).digest())
            if p.endswith((".py", ".sh")):
                names.update(m.decode() for m in _ENV_LITERAL.findall(data))
        trees[root] = f"{len(files)} files {h.hexdigest()}" if os.path.exists(root) else "missing"
    vers = {}
    for pkg in PACKAGES:
        try:
            vers[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            vers[pkg] = None
    env = _env() if environ is None else environ
    used = {k: v for k, v in env.items()
            if (k in names or k in ENV_ALWAYS or k.startswith(ENV_PREFIXES))
            and not k.startswith(ENV_NEVER)}
    desc = {"cb": CB, "python": sys.executable, "version": sys.version, "packages": vers,
            "env": dict(sorted(used.items())), "trees": trees}
    return hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()[:24], desc


# ---------------------------------------------------------------------------------------------
# build, lock, prune
# ---------------------------------------------------------------------------------------------
def _session():
    return _ENVIRON.get("PYTEST_XDIST_TESTRUNUID")


def _build(dest):
    """Every shape into dest/res; meta.json records each shape's process and checks state."""
    res = os.path.join(dest, "res")
    os.makedirs(res)
    shapes = {}
    for shape in SHAPES:
        p = _bench_run(shape, res)
        shapes[shape] = {"args": p.args, "returncode": p.returncode, "stdout": p.stdout,
                         "stderr": p.stderr}
    inputs_ok = {}
    from bench.verify import check
    for d in glob.glob(os.path.join(res, "*", "*", "checks.json")):
        try:
            inputs_ok[os.path.relpath(os.path.dirname(d), res)] = \
                not check.input_problems(os.path.dirname(d), json.load(open(d)))
        except Exception:  # noqa: BLE001 - an unreadable checks.json stays as written
            inputs_ok[os.path.relpath(os.path.dirname(d), res)] = False
    meta = {"build_root": res, "built_unix": time.time(), "session": _session(),
            "ok": all(s["returncode"] == 0 for s in shapes.values()),
            "shapes": shapes, "inputs_ok": inputs_ok}
    json.dump(meta, open(os.path.join(dest, "meta.json"), "w"), indent=1)
    return meta


def _prune(keep):
    """Under the exclusive lock: all but the KEEP most recently used keys (`keep`, the one just
    built, among them), every leftover .build-* dir and other sessions' .failed-* dirs."""
    keys = [d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isfile(os.path.join(d, "meta.json"))]
    keys.sort(key=lambda d: os.path.getmtime(os.path.join(d, "meta.json")), reverse=True)
    old = [d for d in keys if d != keep][KEEP - 1:]
    old += glob.glob(os.path.join(ROOT, ".build-*"))   # under the exclusive lock: nobody builds
    old += [d for d in glob.glob(os.path.join(ROOT, ".failed-*"))
            if not (_session() and d.endswith("-" + _session()))]
    for d in old:
        if d != keep:
            shutil.rmtree(d, ignore_errors=True)


def _cached(key, rebuild):
    """The built dir for `key` to use (the good build, or this session's failed one), or None:
    with `rebuild`, only one built in this xdist session counts."""
    d = os.path.join(ROOT, key)
    failed = os.path.join(ROOT, f".failed-{key}-{_session()}") if _session() else None
    for cand in (d, failed):
        if cand and os.path.isfile(os.path.join(cand, "meta.json")):
            m = json.load(open(os.path.join(cand, "meta.json")))
            if not rebuild or (_session() and m.get("session") == _session()):
                return cand
    return None


_MEMO = {}   # key -> cache dir, this process (a forced rebuild happens once per process / session)


@contextlib.contextmanager
def locked_cache():
    """The cache dir holding every shape's run (built under the exclusive lock if missing),
    held under the lock while the caller reads it, so no prune can remove it meanwhile."""
    key, desc = cache_key()
    rebuild = _ENVIRON.get("BENCH_TEST_RUNS_REBUILD") == "1"
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, ".lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        got = _MEMO.get(key)
        if got is None or not os.path.isfile(os.path.join(got, "meta.json")):
            got = _cached(key, rebuild and key not in _MEMO)
        if got is None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            fcntl.flock(lock, fcntl.LOCK_EX)
            got = _cached(key, rebuild and key not in _MEMO)
            if got is None:
                tmp = os.path.join(ROOT, f".build-{key}-{os.getpid()}")
                shutil.rmtree(tmp, ignore_errors=True)
                meta = _build(tmp)
                json.dump(desc, open(os.path.join(tmp, "key.json"), "w"), indent=1, default=str)
                if meta["ok"]:
                    got = os.path.join(ROOT, key)
                else:   # this session only (see the module doc)
                    got = os.path.join(ROOT, f".failed-{key}-{_session() or os.getpid()}")
                shutil.rmtree(got, ignore_errors=True)   # a forced rebuild replaces it
                os.rename(tmp, got)
                _prune(got)
        os.utime(os.path.join(got, "meta.json"))   # most recently used: pruned last
        _MEMO[key] = got
        yield got


# ---------------------------------------------------------------------------------------------
# a module's own copy
# ---------------------------------------------------------------------------------------------
def _rewrite_tree(top, old, new):
    ob, nb = old.encode(), new.encode()
    for d, _, files in os.walk(top):
        for f in files:
            p = os.path.join(d, f)
            data = open(p, "rb").read()
            if ob not in data:
                continue
            if not f.endswith(_TEXT):   # a binary holding the path: cannot be rewritten safely
                raise RuntimeError(f"fixture_runs: {p} holds the build path and is not text")
            with open(p, "wb") as fh:
                fh.write(data.replace(ob, nb))


def _rehash_checks(run_dir):
    """checks.json's input record re-taken for the rewritten files, written as bench run does."""
    from bench.verify import check
    from bench.verify import load
    from bench import paths
    p = os.path.join(run_dir, "checks.json")
    doc = json.load(open(p))
    if isinstance(doc.get("inputs"), dict):
        doc["inputs"] = {k: load.file_id(run_dir, paths.expand(k)) for k in doc["inputs"]}
        check.write(run_dir, doc)
        assert not check.input_problems(run_dir, doc), check.input_problems(run_dir, doc)


def _copy(src, dst):
    meta = json.load(open(os.path.join(src, "meta.json")))
    res = os.path.join(dst, "res")
    shutil.copytree(os.path.join(src, "res"), res, symlinks=True)
    old = meta["build_root"]
    _rewrite_tree(res, old, res)
    for rel, ok in meta["inputs_ok"].items():
        if ok:
            _rehash_checks(os.path.join(res, rel))
    procs = {s: Proc([a.replace(old, res) for a in m["args"]], m["returncode"],
                     m["stdout"].replace(old, res), m["stderr"].replace(old, res))
             for s, m in meta["shapes"].items()}
    return res, procs


def _materialize(tmp_path_factory, name):
    dst = str(tmp_path_factory.mktemp(name))
    if _ENVIRON.get("BENCH_TEST_RUNS_CACHE") == "0":
        res = os.path.join(dst, "res")
        return res, {s: _bench_run(s, res) for s in SHAPES}
    with locked_cache() as src:
        return _copy(src, dst)


def runs(tmp_path_factory, name):
    """{shape: {"proc": Proc, "dir": run dir}} for every shape, in this caller's own tmp copy."""
    res, procs = _materialize(tmp_path_factory, name)
    return {s: {"proc": procs[s], "dir": _only_run(res, f"s4_{s}")} for s in SHAPES}


def ok_runs(tmp_path_factory, name, shapes=SHAPES):
    """{shape: run dir} of `shapes`, each one's `bench run` asserted to have passed."""
    res, procs = _materialize(tmp_path_factory, name)
    out = {}
    for shape in shapes:
        p = procs[shape]
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        out[shape] = _only_run(res, f"s4_{shape}")
    return out
