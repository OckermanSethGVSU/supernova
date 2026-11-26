"""`bench launch` (stage at a commit, the task, the job, results sync), the manifest fields
and the code-hash roots. No real `sky` / `rsync` / cloud call:
a recording fake `sky` / `rsync` is first on PATH wherever a launch could reach them."""

import glob
import json
import os
from bench import experiment as E
import shutil
import subprocess
import sys

import pytest
import yaml

from bench import launch as L
from bench import launch as LA
from bench import runner

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")
DUMMIES = {"sparse_fineweb": ["corpus.path=s3://DUMMY/corpus/", "queries.nq=1000"],
           "dense_pubmed": ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"],
           "filtered_structured": ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"],
           "filtered_text": ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"]}
NAMES = ["dense_coyo", "dense_pes2o", "sparse_pubmed", "sparse_fineweb", "multivector_pubmed",
         "dense_pubmed", "filtered_structured",
         "filtered_text"]
TODO = set()                    # experiments with TODO placeholders beyond io.* (not launchable yet)


@pytest.fixture
def dirty_index(tmp_path):
    """A git index (a copy of the checkout's) in which one clean_benchmarks file is not tracked:
    git run with GIT_INDEX_FILE set to it sees an uncommitted change in the shipped paths, as on
    a dev box, also in a clean clone. The checkout itself is not touched."""
    g = ["git", "-C", L.REPO]
    idx = subprocess.run(g + ["rev-parse", "--path-format=absolute", "--git-path", "index"],
                         capture_output=True, text=True, check=True).stdout.strip()
    copy = tmp_path / "index"
    shutil.copy(idx, copy)
    env = {**os.environ, "GIT_INDEX_FILE": str(copy)}
    one = subprocess.run(g + ["ls-files", "--", L.CB_REL], env=env, capture_output=True, text=True,
                         check=True).stdout.splitlines()[0]
    subprocess.run(g + ["update-index", "--force-remove", "--", one], env=env, check=True, capture_output=True)
    return str(copy)


@pytest.fixture
def fake_cloud(tmp_path, dirty_index):
    """`sky` and `rsync` that only record their argv (and rsync fakes one synced run dir); git
    sees a dirty tree (dirty_index), which every launch here stages with --allow-dirty."""
    b = tmp_path / "fakebin"
    b.mkdir()
    log = tmp_path / "calls.jsonl"
    for name in ("sky", "rsync", "aws"):
        (b / name).write_text(
            f"#!{sys.executable}\nimport glob, json, os, sys\n"
            f"open({str(log)!r}, 'a').write(json.dumps([{name!r}] + sys.argv[1:]) + '\\n')\n"
            "if sys.argv[0].endswith('rsync') and sys.argv[-1].endswith('/') and ':' in sys.argv[-2] "
            "and '--exclude' not in sys.argv:\n"
            "    d = os.path.join(sys.argv[-1], '20991231T000000Z_abc1234'); os.makedirs(d, exist_ok=True)\n"
            # the run names the launch that started it (bench run --launch-id)
            "    lj = glob.glob(os.path.join(sys.argv[-1], 'launch', '*', 'launch.json'))\n"
            "    lid = json.load(open(lj[0]))['launch_id'] if lj else None\n"
            "    open(os.path.join(d, 'manifest.json'), 'w').write(json.dumps({'launch_id': lid}))\n"
            "    ok = os.environ.get('FAKE_RUN_FAILS') is None\n"
            "    open(os.path.join(d, 'checks.json'), 'w').write(json.dumps({'ok': ok, 'problems': "
            "[] if ok else ['dense r1: oracle cuvs: 3 float64 disagreements']}))\n"
            "    open(os.path.join(d, 'summary.md'), 'w').write('| x |\\n' + ('' if ok else "
            "'\\nPROBLEMS (1):\\n- dense cuvs r2: excluded: exit status 1\\n'))\n")
        (b / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "PYTHONDONTWRITEBYTECODE": "1",
           "GIT_INDEX_FILE": dirty_index}
    return env, log


def calls(log):
    return [json.loads(x) for x in open(log)] if os.path.exists(log) else []


def bench(*args, env=None, timeout=600):
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=CB, capture_output=True,
                          text=True, timeout=timeout, env=env or {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def dry(name, res, env=None, extra=()):
    sets = [z for s in DUMMIES.get(name, []) for z in ("--set", s)]
    p = bench("launch", os.path.join(CB, E.path_of(name)), "--dry-run", "--allow-dirty",
              "--results", str(res), *sets, *extra, env=env)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    d = glob.glob(os.path.join(str(res), "*", "launch", "*"))
    assert len(d) == 1
    return d[0]


# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_dry_run_writes_the_task_and_launches_nothing(name, tmp_path, fake_cloud):
    if name in TODO:
        pytest.skip(f"{name} still has TODO placeholders")
    env, log = fake_cloud
    d = dry(name, tmp_path / "res", env=env)
    for f in ("task.yaml", "staged_files.txt", "STAGED", "launch.json", "sky_command.txt"):
        assert os.path.isfile(os.path.join(d, f)), f
    assert calls(log) == []                                    # nothing reached sky / rsync / aws
    # SkyPilot uploads the workdir minus its .gitignore patterns; the stage's .skyignore (no
    # patterns) replaces them, so the gitignored paths file reaches the node
    st = os.path.join(d, "stage")
    assert os.path.isfile(os.path.join(st, "experiments", "local_paths.env"))
    sky = [x.strip() for x in open(os.path.join(st, ".skyignore")) if x.strip()]
    assert sky and all(x.startswith("#") for x in sky)
    t = yaml.safe_load(open(os.path.join(d, "task.yaml")))
    node = yaml.safe_load(open(L.NODE_YAML))
    assert t["resources"] == node["resources"] and t["resources"]["use_spot"] is False
    assert t["secrets"] == node["secrets"]
    rec = json.load(open(os.path.join(d, "launch.json")))
    assert rec["dirty"] is True and rec["allow_dirty"] is True and rec["dry_run"] is True
    assert rec["experiment_text"] == open(os.path.join(CB, E.path_of(name))).read()
    assert rec["prep"]["stage"] == yaml.safe_load(open(os.path.join(CB, E.path_of(name))))["prep"].get("stage", [])
    assert [d["path"] for d in rec["data"]] == [d["path"] for d in rec["prep"]["stage"]]
    cmd = rec["sky_command"]
    assert cmd[:6] == ["sky", "launch", "-d", "-y", "-c", L.cluster_name({"name": name})]
    assert "--use-spot" not in cmd and "-i" not in cmd and "--workdir" in cmd
    run = t["run"]
    line = L.extract_run_line(open(os.path.join(d, "task.yaml")).read())
    assert line == rec["run_line"]
    assert line.startswith("cd ~/sky_workdir && bash env/paper_run.sh --run-dir ~/res/paper_run_$(date")
    assert f" python -m bench run {E.path_of(name)} --launch-id {rec['launch_id']} --results ~/res" in line
    assert run.index("pip install") < run.index(L.RUN_BEGIN)   # build before the launcher
    files = open(os.path.join(d, "staged_files.txt")).read().splitlines()
    for need in ("env/paper_run.sh", "env/versions.env", "bench/launch.py", "bench/data/sources.py",
                 "nova-bf/pyproject.toml", "pyproject.toml", E.path_of(name),
                 "tests/test_launch.py", "baselines/cuvs/patches/0002-cutile-opt-out.diff"):
        assert need in files, need
    assert not any(f.startswith("results/") or "__pycache__" in f or f.endswith(".so") for f in files)
    if name.startswith("filtered"):
        assert "nova_textscan/Cargo.lock" in files
        assert any("(data: filtered_" in f for f in files)            # listed, not copied
        assert not os.path.exists(os.path.join(rec["stage"], "data", "corpus"))


def test_dirty_tree_is_refused_and_nothing_is_written(tmp_path):
    if not L.dirty_paths(L.shipped_paths({"name": "dense_coyo"})):
        pytest.skip("the checkout is clean")
    p = bench("launch", os.path.join(CB, E.path_of("dense_coyo")), "--dry-run", "--results", str(tmp_path))
    assert p.returncode == 2 and "REFUSING" in p.stderr and "--allow-dirty" in p.stderr
    assert not os.listdir(tmp_path)


def test_placeholders_refused_and_a_sweep_launches_with_job_sweep(tmp_path):
    p = bench("launch", os.path.join(CB, E.path_of("filtered_structured")), "--dry-run", "--allow-dirty",
              "--results", str(tmp_path))
    assert p.returncode == 2 and "TODO" in p.stderr
    assert "an io sweep over io.*" in p.stderr
    sw = os.path.join(EXP, "sweeps", "filtered_structured_io.yaml")
    p = bench("launch", sw, "--dry-run", "--allow-dirty", "--results", str(tmp_path / "t"))
    assert p.returncode == 2 and "--job sweep" in p.stderr
    p = bench("launch", os.path.join(CB, E.path_of("dense_coyo")), "--job", "sweep", "--dry-run",
              "--allow-dirty", "--results", str(tmp_path / "t"))
    assert p.returncode == 2 and "not a sweep file" in p.stderr
    p = bench("launch", sw, "--job", "sweep", "--dry-run", "--allow-dirty",
              "--results", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr
    ld, = glob.glob(str(tmp_path / "t" / "sweeps" / "filtered_structured_io" / "launch" / "*"))
    rec = json.load(open(os.path.join(ld, "launch.json")))
    assert rec["job"] == "sweep" and rec["results_dir"] == "sweeps/filtered_structured_io"
    assert (" python -m bench sweep tests/fixtures/experiments/sweeps/filtered_structured_io.yaml "
            f"--launch-id {rec['launch_id']} "
            "--results ~/res") in rec["run_line"]


def test_s3_writes_outside_the_out_prefix_are_refused(tmp_path):
    p = bench("launch", os.path.join(CB, E.path_of("dense_coyo")), "--dry-run", "--allow-dirty",
              "--set", "novabf_output=${READ}/coyo", "--results", str(tmp_path))
    assert p.returncode == 2 and "${WRITE} only" in p.stderr


def test_a_missing_data_source_is_refused_before_sky(tmp_path, fake_cloud):
    """a prep step's refusal (bench/prep/steps.py) exits 2 like a launch refusal; no sky call"""
    env, log = fake_cloud
    p = bench("launch", os.path.join(CB, E.path_of("dense_pes2o")), "--allow-dirty", "--results",
              str(tmp_path / "r"), "--data-src", f"PES2O_QUERIES={tmp_path / 'nope.parquet'}", env=env)
    assert p.returncode == 2, p.stderr
    assert "bench launch: data source " in p.stderr and "--data-src PES2O_QUERIES=PATH" in p.stderr
    assert not calls(log)


def test_spot_is_refused(tmp_path, monkeypatch):
    y = tmp_path / "node.yaml"
    node = yaml.safe_load(open(L.NODE_YAML))
    node["resources"]["use_spot"] = True
    y.write_text(yaml.safe_dump(node))
    monkeypatch.setattr(L, "NODE_YAML", str(y))
    with pytest.raises(L.LaunchError, match="on-demand"):
        L.load_node()
    node["resources"].pop("use_spot")
    y.write_text(yaml.safe_dump(node))
    with pytest.raises(L.LaunchError, match="on-demand"):
        L.load_node()


def test_a_launch_calls_sky_follows_the_job_and_syncs_back(tmp_path, fake_cloud):
    """The real (non dry-run) path, with sky / rsync faked: the old `sky launch -d -y -c ...
    --workdir STAGE --secret ...` command, `sky logs`, then the three rsyncs."""
    env, log = fake_cloud
    res = tmp_path / "res"
    p = bench("launch", os.path.join(CB, E.path_of("dense_coyo")), "--allow-dirty", "--results", str(res),
              env=env)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    c = calls(log)
    d = glob.glob(str(res / "dense_coyo" / "launch" / "*"))[0]
    assert c[0][:7] == ["sky", "launch", "-d", "-y", "-c", "bench-dense-coyo", os.path.join(d, "task.yaml")]
    assert c[0][7:] == ["--workdir", os.path.join(d, "stage"), "--secret", "AWS_ACCESS_KEY_ID",
                        "--secret", "AWS_SECRET_ACCESS_KEY", "--secret", "AWS_SESSION_TOKEN"]
    assert c[1] == ["sky", "logs", "bench-dense-coyo"]
    assert [x[0] for x in c[2:]] == ["rsync"] * 3
    assert c[2][-2:] == ["bench-dense-coyo:~/res/dense_coyo/", str(res / "dense_coyo") + "/"]
    assert not any(x[0] == "aws" or "down" in x for x in c)          # never torn down / no aws
    run = res / "dense_coyo" / "20991231T000000Z_abc1234"
    assert json.load(open(run / "launch.json"))["commit"] == json.load(open(os.path.join(d, "launch.json")))["commit"]
    assert json.load(open(os.path.join(d, "synced.json")))["runs"] == ["20991231T000000Z_abc1234"]
    # the stage was really built (code + hash), outside a dry run
    staged = open(os.path.join(d, "stage", "STAGED")).read()
    assert "code sha256: " in staged and "dirty: yes" in staged


# ---------------------------------------------------------------------------------------------
# staging from a clean commit (a scratch repo with the same layout)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def scratch_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    cb = repo / "clean_benchmarks"
    for d in ("env", "baselines"):
        shutil.copytree(os.path.join(CB, d), cb / d, ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    for d in ("bench/data", "tests/data", "experiments"):
        (cb / d).mkdir(parents=True)
    (cb / "bench" / "data" / "sources.py").write_text("X = 1\n")
    (cb / "bench" / "launch.py").write_text("Y = 1\n")
    (cb / "tests" / "data" / "test_x.py").write_text("def test(): pass\n")
    (cb / "experiments" / "e.yaml").write_text("name: e\n")
    (cb / "pyproject.toml").write_text("[project]\nname='bench'\n")
    nb = repo / "python" / "nova-bf"
    nb.mkdir(parents=True)
    (nb / "pyproject.toml").write_text("[project]\nname='nova-bf'\n")
    g = ["git", "-C", str(repo)]
    subprocess.run(g + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"], check=True)
    monkeypatch.setattr(L, "REPO", str(repo))
    return repo


def test_a_clean_commit_is_staged_by_git_archive(scratch_repo, tmp_path):
    st = L.stage_code({"name": "dense_coyo"}, str(tmp_path / "s"), "HEAD", allow_dirty=False)
    assert st["dirty"] is False and st["source"].startswith("git archive")
    s = tmp_path / "s"
    assert (s / "bench" / "data" / "sources.py").read_text() == "X = 1\n"
    assert (s / "nova-bf" / "pyproject.toml").exists() and (s / "tests" / "data" / "test_x.py").exists()
    # an uncommitted edit is refused without --allow-dirty, staged as is with it
    (scratch_repo / "clean_benchmarks" / "bench" / "data" / "sources.py").write_text("X = 2\n")
    with pytest.raises(L.LaunchError, match="REFUSING"):
        L.stage_code({"name": "dense_coyo"}, str(tmp_path / "s2"), "HEAD", allow_dirty=False)
    st = L.stage_code({"name": "dense_coyo"}, str(tmp_path / "s3"), "HEAD", allow_dirty=True)
    assert st["dirty"] and any("sources.py" in x for x in st["dirty_paths"])
    assert (tmp_path / "s3" / "bench" / "data" / "sources.py").read_text() == "X = 2\n"


def test_a_staged_patch_with_the_wrong_hash_is_refused(scratch_repo, tmp_path):
    p = scratch_repo / "clean_benchmarks" / "baselines" / "cuvs" / "patches" / "0002-cutile-opt-out.diff"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(L.LaunchError, match="CUVS_PATCH_SHA256"):
        L.stage_code({"name": "dense_coyo"}, str(tmp_path / "s"), "HEAD", allow_dirty=True)


# ---------------------------------------------------------------------------------------------
# the code hash
# ---------------------------------------------------------------------------------------------
def test_paper_run_and_launch_hash_the_same_data_named_dirs():
    text = open(os.path.join(CB, "env", "paper_run.sh")).read()
    line = next(x for x in text.splitlines() if x.strip().startswith("DATA_CODE_DIRS="))
    assert tuple(line.split("=", 1)[1].strip('" ').split()) == L.DATA_CODE_DIRS
    for d in L.DATA_CODE_DIRS:
        assert os.path.isdir(os.path.join(CB, d)), d


@pytest.mark.parametrize("edit", ["bench/data/sources.py", "tests/data/test_x.py", "bench/launch.py"])
def test_a_change_in_a_code_root_changes_the_stage_hash(scratch_repo, tmp_path, edit):
    s = tmp_path / "s"
    L.stage_code({"name": "dense_coyo"}, str(s), "HEAD", allow_dirty=False)
    h0 = L.code_sha(str(s))
    with open(s / edit, "a") as f:
        f.write("# edited\n")
    assert L.code_sha(str(s)) != h0
    roots = json.load(open(s / "STAGED_code.json"))["roots"]
    assert "bench/data" in roots and "tests/data" in roots


def test_the_staged_hash_is_what_paper_run_recomputes_on_the_node(tmp_path):
    """env/paper_run.sh's stage branch over a real `bench launch` stage == STAGED's hash."""
    d = dry("dense_coyo", tmp_path / "res")
    stage = os.path.expanduser(json.load(open(os.path.join(d, "launch.json")))["stage"])   # recorded with ~ for a home path
    act, chk = tmp_path / "act.sh", tmp_path / "chk.py"
    act.write_text(f"export CONDA_PREFIX={os.path.dirname(os.path.dirname(sys.executable))}\n"
                   f"export PATH={os.path.dirname(sys.executable)}:$PATH\n")
    chk.write_text("import sys; sys.exit(0)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PAPER_")}
    env.update(PAPER_ACTIVATE=str(act), PAPER_CHECK_ENV=str(chk), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(["bash", os.path.join(stage, "env", "paper_run.sh"), "--run-dir",
                        str(tmp_path / "pr"), "true"], env=env, cwd=stage, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    staged = open(os.path.join(stage, "STAGED")).read().split("code sha256: ")[1].split()[0]
    assert json.load(open(tmp_path / "pr" / "code.json"))["code_sha"] == staged
    assert "differs from the stage" not in r.stderr


# ---------------------------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------------------------
def test_manifest_reads_the_stage_record(tmp_path, monkeypatch):
    (tmp_path / "STAGED").write_text("staged 20261119T000000Z commit=abcdef0123 (HEAD) from git archive abcdef0123\n"
                                     "code sha256: " + "c" * 64 + "\ndirty: no\n")
    monkeypatch.setattr(runner, "CB", str(tmp_path))
    st = runner.staged()
    assert st["commit"] == "abcdef0123" and st["code_sha"] == "c" * 64 and st["dirty"] is False
    assert runner.gpus("NVIDIA A10G, GPU-1234, 570.86.15\n") == [
        {"name": "NVIDIA A10G", "uuid": "GPU-1234", "driver": "570.86.15"}]


@pytest.mark.slow
def test_manifest_fields_on_a_cpu_run(tmp_path_factory):
    from helpers import fixture_runs   # the shared, cached `bench run --dev` of dense.yaml
    d = fixture_runs.ok_runs(tmp_path_factory, "manifest_run", ("dense",))["dense"]
    m = json.load(open(os.path.join(d, "manifest.json")))
    for k in ("commit", "dirty", "stage", "code_sha", "stack_sha", "paper", "files", "instance",
              "gpu", "gpus", "node", "cuda_libs", "finished_utc"):
        assert k in m, k
    assert "tests/fixtures/experiments/dense.yaml" in m["files"]          # a copy of the experiment
    tags = set(m["cuda_libs"])
    assert {"dense__cpu__r1", "dense__novabf__r1"} <= tags
    assert set(m["cuda_libs"]["dense__cpu__r1"]) == set(runner.CUDA_LIB_COLS)


# ---------------------------------------------------------------------------------------------
# the old-vs-new launch comparison and the env fingerprint's run block
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# sync links only this launch's runs; a failed run fails the launch
# ---------------------------------------------------------------------------------------------
LAUNCH_ID = "20991119T000000Z_bbbbbbb"


def _launch_dir(root, code_sha="c0de"):
    ld = os.path.join(root, "dense_coyo", "launch", LAUNCH_ID)
    os.makedirs(ld)
    json.dump({"cluster": "bench-x", "experiment": "dense_coyo", "results_root": root,
               "commit": "bbbbbbb", "code_sha": code_sha, "job": "run", "launch_id": LAUNCH_ID},
              open(os.path.join(ld, "launch.json"), "w"))
    return ld


def _arrive(root, name, ok=True, code_sha=None, launch_id=LAUNCH_ID):
    """a run's manifest names the launch that started it (None: by hand)."""
    d = os.path.join(root, "dense_coyo", name)
    os.makedirs(d, exist_ok=True)
    json.dump({"commit": name[-7:], "launch_id": launch_id,
               **({"stage": {"code_sha": code_sha}} if code_sha else {})},
              open(os.path.join(d, "manifest.json"), "w"))
    json.dump({"ok": ok, "problems": [] if ok else ["s r1: oracle x: boom"]}, open(os.path.join(d, "checks.json"), "w"))
    open(os.path.join(d, "summary.md"), "w").write("| x |\n" + ("" if ok else "\nPROBLEMS (1):\n- s x r1: excluded: boom\n"))
    return d


def test_sync_links_only_the_runs_of_this_launch(tmp_path, capsys):
    """sync_probe.py's shape: an OLD local run of the experiment (no launch.json) must not be
    stamped with this launch's record; the run that arrives is."""
    root = str(tmp_path)
    old = _arrive(root, "20991101T000000Z_aaaaaaa", launch_id=None)
    ld = _launch_dir(root)

    def rsync(cmd):
        if cmd[2].endswith("/dense_coyo/"):
            _arrive(root, "20991119T010000Z_bbbbbbb")
        return 0
    assert L.sync(ld, runner=rsync) == 0
    assert not os.path.exists(os.path.join(old, "launch.json"))
    new = os.path.join(root, "dense_coyo", "20991119T010000Z_bbbbbbb", "launch.json")
    assert json.load(open(new))["commit"] == "bbbbbbb"
    s = json.load(open(os.path.join(ld, "synced.json")))
    assert s["runs"] == ["20991119T010000Z_bbbbbbb"] and s["not_this_launch"] == ["20991101T000000Z_aaaaaaa"]
    # a second sync: nothing new, but the run names this launch's id: still this launch's
    # (the launch id, not the staged code sha, says so)
    os.remove(new)
    L.sync(ld, runner=lambda c: 0)
    assert os.path.exists(new) and not os.path.exists(os.path.join(old, "launch.json"))


def test_sync_fails_when_the_synced_run_failed_or_none_arrived(tmp_path, capsys):
    root = str(tmp_path)
    ld = _launch_dir(root)
    assert L.sync(ld, runner=lambda c: (_arrive(root, "20991119T010000Z_bbbbbbb", ok=False)
                                        if c[2].endswith("/dense_coyo/") else None) or 0) == 1
    err = capsys.readouterr().err
    assert "the launched run FAILED" in err and "checks failed: s r1: oracle x: boom" in err
    assert "summary: s x r1: excluded: boom" in err
    ld2 = os.path.join(root, "dense_coyo", "launch", "20991119T000000Z_ccccccc")
    shutil.copytree(ld, ld2)
    rec = json.load(open(os.path.join(ld2, "launch.json")))
    json.dump({**rec, "launch_id": "20991119T000000Z_ccccccc"}, open(os.path.join(ld2, "launch.json"), "w"))
    assert L.sync(ld2, runner=lambda c: 0) == 1                 # no run of THIS launch arrived
    assert "no run of this launch" in capsys.readouterr().err


def test_a_launch_whose_run_failed_exits_non_zero(tmp_path, fake_cloud):
    env, log = fake_cloud
    p = bench("launch", os.path.join(CB, E.path_of("dense_coyo")), "--allow-dirty", "--results",
              str(tmp_path / "res"), env={**env, "FAKE_RUN_FAILS": "1"})
    assert p.returncode == 1, p.stdout[-2000:] + p.stderr[-2000:]
    assert "the launched run FAILED" in p.stderr and "excluded: exit status 1" in p.stderr


def test_allow_dirty_refuses_another_rev(tmp_path):
    """--allow-dirty stages the working tree (HEAD + edits); --rev REV != HEAD is refused."""
    prev = subprocess.run(["git", "-C", LA.REPO, "rev-parse", "HEAD~1"], capture_output=True, text=True)
    if prev.returncode != 0:
        pytest.skip("no parent commit")
    p = subprocess.run([sys.executable, "-m", "bench", "launch", os.path.join(CB, E.path_of("dense_coyo")),
                        "--dry-run", "--allow-dirty", "--rev", "HEAD~1", "--results", str(tmp_path)],
                       cwd=CB, capture_output=True, text=True)
    assert p.returncode == 2 and "is not HEAD" in p.stderr
    assert not os.listdir(tmp_path)


LID = "20261120T100000Z_bbbbbbb"


def test_the_run_line_passes_the_launch_id_as_an_argument():
    from bench import launch as LA
    line = LA.run_line("experiments/dense/coyo.yaml", "run", ["a=b"], LID)
    assert f"python -m bench run experiments/dense/coyo.yaml --launch-id {LID} --results ~/res --set a=b" in line
    assert "BENCH_LAUNCH_ID" not in line and "=" + LID not in line
    assert f"python -m bench sweep experiments/sweeps/dense_coyo_io.yaml --launch-id {LID} " in \
        LA.run_line("experiments/sweeps/dense_coyo_io.yaml", "sweep", [], LID)
    with pytest.raises(LA.LaunchError):
        LA.run_line("experiments/dense/coyo.yaml", "run", [], "x; rm -rf ~")


RV_SUMMARY = """run: ...

PROBLEMS (1):
- s r1: pairs: x

Warnings (2, not failures):
- rounds s cuvs: r1 vs r2 DIFFERENT
- s__cuvs__r1: the attribution gpu_view does not add up to the LegClock time (sum 1 s, LegClock 1.02 s, residual 0.02 s; README.md#attribution)
"""


def test_rv_run_verdict_lists_problems_not_warnings(tmp_path):
    from bench import launch
    (tmp_path / "checks.json").write_text(json.dumps({"ok": False, "problems": ["s r1: pairs: x"]}))
    (tmp_path / "summary.md").write_text(RV_SUMMARY)
    v = launch.run_verdict(str(tmp_path))
    assert v == ["checks failed: s r1: pairs: x", "summary: s r1: pairs: x"], v


def test_warnings_alone_are_not_a_failed_run(tmp_path):
    from bench import launch
    (tmp_path / "checks.json").write_text(json.dumps({"ok": True}))
    (tmp_path / "summary.md").write_text("run: ...\n\nWarnings (1, not failures):\n- w\n")
    assert launch.run_verdict(str(tmp_path)) == []


def test_stray_code_files_in_the_workdir_are_moved_not_deleted(tmp_path):
    """the node's workdir keeps files from before the upload: every code file the stage does not
    list moves to DEST (relative path kept); staged files, data and build trees stay."""
    import subprocess
    w = tmp_path / "w"
    for rel in ("bench/a.py", "python/nova-bf/src/x.py", "benchmarks/run.sh", "data/q.parquet",
                "nova-bf/rust_io/target/t.rs", "notes.txt"):
        p = w / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    (w / "STAGED_files.txt").write_text("bench/a.py\nSTAGED_files.txt\n")
    dest = tmp_path / "stray"
    r = subprocess.run([sys.executable, os.path.join(CB, "env", "stray_files.py"), str(w),
                        str(w / "STAGED_files.txt"), str(dest)], capture_output=True, text=True)
    assert r.returncode == 0 and "2 not in the stage" in r.stdout, r.stdout + r.stderr
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == \
        ["benchmarks/run.sh", "python/nova-bf/src/x.py"]
    for kept in ("bench/a.py", "data/q.parquet", "nova-bf/rust_io/target/t.rs", "notes.txt"):
        assert (w / kept).is_file(), kept
    r = subprocess.run([sys.executable, os.path.join(CB, "env", "stray_files.py"), str(w),
                        str(w / "STAGED_files.txt"), str(dest)], capture_output=True, text=True)
    assert "stray code files: none" in r.stdout


def test_the_stage_lists_its_own_files_and_move_strays_is_opt_in(tmp_path, fake_cloud):
    """the stage always carries STAGED_files.txt; the job moves strays only with --move-strays,
    right after reading the paths file and before the build."""
    env, _ = fake_cloud
    d = dry("dense_coyo", tmp_path / "res", env=env)
    st = os.path.join(d, "stage")
    listed = open(os.path.join(st, "STAGED_files.txt")).read().split()
    assert "STAGED_files.txt" in listed and "bench/launch.py" in listed and "env/stray_files.py" in listed
    run = yaml.safe_load(open(os.path.join(d, "task.yaml")))["run"].splitlines()
    assert L.STRAY_LINE not in run
    assert json.load(open(os.path.join(d, "launch.json")))["move_strays"] is False
    e = E.load(os.path.join(CB, E.path_of("dense_coyo")))
    run = L.run_block(e, "experiments/dense/coyo.yaml", "run", [], move_strays=True).splitlines()
    i = run.index(L.STRAY_LINE)
    assert run[i - 1] == L.PATHS_LINE and "STEP build" in " ".join(run[i + 1:i + 3])
