"""The paper-stack plumbing (environment/README.md, "Launching a paper job"):
harness/provenance.py, environment/env/code_sha.py, check_env.py's --json OUT,
and environment/env/paper_run.sh in a stubbed dry run (no conda env, no GPU:
a stub activation, a stub check and a fake nvidia-smi)."""

import io
import json
import os
import subprocess
import sys
import textwrap

import pytest

from bench.record import provenance

ENV_DIR = os.path.join(provenance.ROOT, "env")
LAUNCHER = os.path.join(ENV_DIR, "paper_run.sh")
sys.path.insert(0, ENV_DIR)
import code_sha  # noqa: E402


def _maps(paths):
    return "".join(f"7f00-7f01 r-xp 00000000 08:01 1 {p}\n" for p in paths)


# --- versions.env's hash ---------------------------------------------------------------------

def _shell_sha(path):
    """activate.sh's / build_env.sh's own pipeline, run by bash."""
    cmd = (f"grep -v '^STACK_FROZEN_ON=' '{path}' | sed 's/[[:space:]]*#.*$//' "
           "| grep -v '^[[:space:]]*$' | sha256sum | cut -d' ' -f1")
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                          check=True).stdout.strip()


def test_versions_sha_is_the_shell_hash(tmp_path):
    assert provenance.versions_sha() == _shell_sha(provenance.VERSIONS_ENV)
    # comments, blank lines, trailing comments and the freeze date do not count
    a, b = tmp_path / "a.env", tmp_path / "b.env"
    a.write_text("# head\nX=1   # one\n\nSTACK_FROZEN_ON=TO-FILL\nY=\"a b\"\t# two\n")
    b.write_text("X=1\nSTACK_FROZEN_ON=2026-12-01\n   \nY=\"a b\"")   # no final newline
    assert provenance.versions_sha(a) == provenance.versions_sha(b) == _shell_sha(a) == _shell_sha(b)
    b.write_text("X=2\nY=\"a b\"\n")
    assert provenance.versions_sha(b) != provenance.versions_sha(a)
    assert provenance.versions_sha(tmp_path / "missing.env") is None


def test_stack_problem(tmp_path):
    good = {"PAPER_STACK_SHA": provenance.versions_sha(), "PAPER_ENV_PREFIX": sys.prefix}
    assert provenance.stack_problem(environ=good) is None
    assert "not set" in provenance.stack_problem(environ={})
    assert "another stack" in provenance.stack_problem(environ=dict(good, PAPER_STACK_SHA="0" * 64))
    assert "stub" in provenance.stack_problem(environ=dict(good, PAPER_LAUNCH_STUBBED="1"))
    assert "PAPER_ENV_PREFIX is not set" in provenance.stack_problem(
        environ={"PAPER_STACK_SHA": good["PAPER_STACK_SHA"]})
    assert "not the paper env" in provenance.stack_problem(
        environ=dict(good, PAPER_ENV_PREFIX=str(tmp_path)))
    # the prefix compares as a real path (a symlinked env is the same env)
    link = tmp_path / "envlink"
    os.symlink(sys.prefix, link)
    assert provenance.stack_problem(environ=dict(good, PAPER_ENV_PREFIX=str(link))) is None
    # a harness without a versions.env (nothing to compare): only presence + prefix
    assert provenance.stack_problem(tmp_path / "none.env",
                                    environ=dict(good, PAPER_STACK_SHA="x")) is None


# --- EC2 instance metadata -------------------------------------------------------------------

class _Resp(io.BytesIO):
    pass


def test_ec2_identity_reads_imdsv2(monkeypatch):
    monkeypatch.delenv(provenance.NO_IMDS_ENV, raising=False)
    seen = []

    def opener(req, timeout):
        seen.append((req.get_method(), req.full_url, dict(req.header_items()), timeout))
        path = req.full_url.split("169.254.169.254", 1)[1]
        return _Resp({"/latest/api/token": b"TOKEN\n",
                      "/latest/meta-data/instance-id": b"i-0123",
                      "/latest/meta-data/instance-type": b"g5.8xlarge",
                      "/latest/meta-data/placement/availability-zone": b"us-east-1b"}[path])

    got = provenance.ec2_identity(opener=opener, cache=False)
    assert got == {"instance_id": "i-0123", "instance_type": "g5.8xlarge",
                   "availability_zone": "us-east-1b"}
    assert seen[0][0] == "PUT" and seen[0][1].endswith("/latest/api/token")
    assert all(h.get("X-aws-ec2-metadata-token") == "TOKEN" for _, _, h, _ in seen[1:])
    assert all(t <= 1 for *_, t in seen)   # short timeouts


def test_ec2_identity_is_blank_off_ec2(monkeypatch):
    monkeypatch.delenv(provenance.NO_IMDS_ENV, raising=False)

    def refused(req, timeout):
        raise OSError("no route to host")
    blank = {"instance_id": "", "instance_type": "", "availability_zone": ""}
    assert provenance.ec2_identity(opener=refused, cache=False) == blank

    def half(req, timeout):   # a token, then a failure: nothing half-filled
        if req.full_url.endswith("token"):
            return _Resp(b"T")
        raise TimeoutError
    assert provenance.ec2_identity(opener=half, cache=False) == blank
    monkeypatch.setenv(provenance.NO_IMDS_ENV, "1")

    def never(req, timeout):
        raise AssertionError("looked up with HARNESS_NO_IMDS set")
    assert provenance.ec2_identity(opener=never, cache=False) == blank


# --- what is mapped / installed --------------------------------------------------------------

def test_thread_libs_finds_every_openmp_runtime_and_blas():
    libs = provenance.thread_libs(_maps([
        "/site/torch/lib/libgomp-a34b3233.so.1", "/site/torch/lib/libgomp-a34b3233.so.1",
        "/env/lib/libomp.so", "/env/lib/libiomp5.so", "/env/lib/libmkl_core.so.2",
        "/env/lib/libmkl_intel_thread.so.2", "/site/numpy.libs/libscipy_openblas64_-ff651d7f.so",
        "/env/lib/libtbb.so.12", "/env/lib/libomptarget.so", "/usr/lib/libc.so.6",
        "/env/lib/libgomp.so.1"]))
    assert list(libs) == ["libgomp", "libiomp5", "libmkl_core", "libmkl_intel_thread", "libomp",
                          "libscipy_openblas64", "libtbb"]
    assert libs["libgomp"] == ["/site/torch/lib/libgomp-a34b3233.so.1", "/env/lib/libgomp.so.1"]
    f = provenance.thread_libs_fields(libs)
    assert f["openmp_runtimes"] == "libgomp,libiomp5,libomp"
    assert f["thread_libs"].startswith("libgomp=/site/torch/lib/libgomp-a34b3233.so.1;"
                                       "libgomp=/env/lib/libgomp.so.1;libiomp5=")
    assert provenance.thread_libs_fields({}) == {"thread_libs": "", "openmp_runtimes": ""}


def test_env_str():
    env = {"OMP_NUM_THREADS": "32", "KMP_AFFINITY": "compact", "MKL_THREADING_LAYER": "GNU",
           "HOME": "/h", "CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "x"}
    assert provenance.env_str(provenance.THREAD_ENV_PREFIXES, env) == \
        "KMP_AFFINITY=compact;MKL_THREADING_LAYER=GNU;OMP_NUM_THREADS=32"
    assert provenance.env_str(provenance.CUDA_ENV_PREFIXES, env) == \
        "CUDA_VISIBLE_DEVICES=0;PYTORCH_CUDA_ALLOC_CONF=x"
    assert provenance.env_str(provenance.THREAD_ENV_PREFIXES, {}) == ""


class _Faiss:
    @staticmethod
    def get_compile_options():
        return "AVX2 GPU "


def test_faiss_build_names_the_conda_build_string(tmp_path):
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    for f in ("faiss-gpu-1.16.0-py3.12_hc84be16_0_cuda13.2.json",
              "libfaiss-1.16.0-hc84be16_0_cuda13.2.json", "faiss-gpu-cuvs-x.json.bak",
              "libcublas-13.3.0-h0.json"):
        (meta / f).write_text("{}")
    got = provenance.faiss_build(_Faiss, prefix=str(tmp_path))
    assert got == {"faiss_build": "conda:faiss-gpu-1.16.0-py3.12_hc84be16_0_cuda13.2,"
                                  "libfaiss-1.16.0-hc84be16_0_cuda13.2",
                   "faiss_compile_options": "AVX2 GPU"}
    # not a conda env: the pip distribution, if any (whatever this box has), never raises
    got = provenance.faiss_build(object(), prefix=str(tmp_path / "nothing"))
    assert got["faiss_compile_options"] == ""
    assert got["faiss_build"] == "" or got["faiss_build"].startswith("pip:faiss")


def test_cuvs_fields_only_when_cuvs_is_imported(tmp_path, monkeypatch):
    import types
    monkeypatch.delitem(sys.modules, "cuvs", raising=False)
    maps = _maps(["/env/lib/libcuvs.so", "/env/lib/libcuvs_c.so", "/env/lib/libcuvs.so"])
    assert set(provenance.cuvs_fields(maps).values()) == {""}
    monkeypatch.setitem(sys.modules, "cuvs", types.SimpleNamespace(__version__="26.10.00"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "check_env.json").write_text(json.dumps(
        {"sha256": {"/env/lib/libcuvs.so": "ab" * 32, "/env/lib/libfaiss.so": "cd" * 32}}))
    (tmp_path / provenance.CUVS_PATCH_STAMP).write_text(
        "1111  0001-sparse-knn-ip-merge-and-short-batch.diff\n")
    got = provenance.cuvs_fields(maps, run_dir=str(run_dir), prefix=str(tmp_path))
    assert got == {"cuvs_version": "26.10.00", "libcuvs": "/env/lib/libcuvs.so;/env/lib/libcuvs_c.so",
                   "libcuvs_sha256": "ab" * 32 + ";",
                   "cuvs_patches": "0001-sparse-knn-ip-merge-and-short-batch.diff=1111"}
    # no launcher record / no stamp: the version and path only, no hash invented
    got = provenance.cuvs_fields(maps, run_dir="", prefix=str(tmp_path / "x"))
    assert got["libcuvs_sha256"] == got["cuvs_patches"] == "" and got["cuvs_version"] == "26.10.00"


# --- code_sha.py -----------------------------------------------------------------------------

def _tree(root):
    (root / "harness" / "__pycache__").mkdir(parents=True)
    (root / "harness" / "run.py").write_text("print(1)\n")
    (root / "harness" / "__pycache__" / "run.cpython-312.pyc").write_bytes(b"\0")
    (root / "nova-bf" / "src" / "nova_bf.egg-info").mkdir(parents=True)
    (root / "nova-bf" / "src" / "nova_bf.egg-info" / "x.toml").write_text("")
    (root / "nova-bf" / "rust_io" / "target").mkdir(parents=True)
    (root / "nova-bf" / "rust_io" / "target" / "build.rs").write_text("")
    (root / "nova-bf" / "rust_io" / "lib.rs").write_text("fn main() {}\n")
    (root / "nova-bf" / "rust_io" / "fetch.so").write_bytes(b"\x7fELF")
    (root / "data").mkdir()
    (root / "data" / "q.py").write_text("")          # under data/: never code
    (root / "data" / "q.parquet").write_bytes(b"PAR1")
    (root / "run.sh").write_text("echo\n")
    (root / "cfg.yaml").write_text("k: 1\n")
    (root / "STAGED").write_text("staged ...\n")     # the stage's record: not code
    (root / "notes.md").write_text("doc\n")


def test_code_sha_hashes_code_files_only(tmp_path):
    root = tmp_path / "stage"
    _tree(root)
    sha, rows = code_sha.tree_sha([str(root)], str(root))
    assert [r for r, _ in rows] == ["cfg.yaml", "harness/run.py", "nova-bf/rust_io/lib.rs",
                                    "run.sh"]
    # caches, build outputs, data, records and docs change nothing
    (root / "harness" / "__pycache__" / "run.cpython-312.pyc").write_bytes(b"\1")
    (root / "STAGED").write_text("code sha256: x\n")
    (root / "data" / "q.py").write_text("x = 1\n")
    (root / "nova-bf" / "rust_io" / "fetch.so").write_bytes(b"\x7fELF2")
    assert code_sha.tree_sha([str(root)], str(root))[0] == sha
    # an edit, a new code file, or a move all do
    (root / "run.sh").write_text("echo 2\n")
    edited = code_sha.tree_sha([str(root)], str(root))[0]
    assert edited != sha
    (root / "harness" / "new.py").write_text("")
    assert code_sha.tree_sha([str(root)], str(root))[0] != edited
    # the same files elsewhere hash the same (relative paths)
    other = tmp_path / "copy"
    _tree(other)
    assert code_sha.tree_sha([str(other)], str(other))[0] == sha
    # a file root counts only if it is code
    assert [r for r, _ in code_sha.tree_sha(
        [str(root / "run.sh"), str(root / "data" / "q.parquet")], str(root))[1]] == ["run.sh"]


def test_code_sha_cli_json(tmp_path, capsys):
    root = tmp_path / "stage"
    _tree(root)
    out = tmp_path / "code.json"
    assert code_sha.main(["--base", str(root), "--json", str(out), str(root)]) == 0
    sha = capsys.readouterr().out.strip()
    rec = json.loads(out.read_text())
    assert rec["code_sha"] == sha and rec["n_files"] == 4 and rec["roots"] == ["."]
    assert rec["git"] == {".": None}   # not a checkout
    assert code_sha.main([str(tmp_path / "missing")]) == 2


def test_code_sha_git_record_for_a_checkout():
    """In this repository: HEAD, dirty or not, and the dirty hash (read-only git)."""
    nova = os.path.join(provenance.ROOT, "..", "python", "nova-bf")
    info = code_sha.git_info(os.path.realpath(nova))
    if info is None:
        pytest.skip("not a git checkout")
    assert len(info["head"]) == 40 and isinstance(info["dirty"], bool)
    assert (len(info["dirty_sha256"]) == 64) == info["dirty"]


# --- check_env.py --json OUT -----------------------------------------------------------------

def test_check_env_writes_json_to_a_path_and_reports_thread_libs(tmp_path, capsys, monkeypatch):
    """--report-only on this box (no GPU, no paper env): never fails, writes the
    JSON next to the printed table, and records the OpenMP / env snapshot."""
    import check as check_env
    monkeypatch.setenv("KMP_BLOCKTIME", "0")
    out = tmp_path / "check_env.json"
    rc = check_env.main(["--versions", provenance.VERSIONS_ENV, "--report-only",
                         "--engines", "faiss", "--json", str(out)])
    assert rc == 0
    table = capsys.readouterr().out
    assert "OpenMP runtimes mapped" in table and "NOT THE PINNED STACK" in table
    rec = json.loads(out.read_text())
    assert rec["stack_sha"] == provenance.versions_sha() and rec["ok"] is False
    assert rec["thread_env"]["KMP_BLOCKTIME"] == "0"
    assert isinstance(rec["thread_libraries"], dict) and rec["ec2"]["instance_type"] == ""
    names = {c["name"]: c for c in rec["checks"]}
    assert names["EC2 instance type"]["ok"] is False   # off EC2 (no IMDS here)
    assert "faiss conda build (exact)" in names        # this box: not the pinned build
    # --json alone: the JSON on stdout instead of the table (build_env.sh's freeze)
    check_env.main(["--versions", provenance.VERSIONS_ENV, "--report-only", "--engines",
                    "faiss", "--json"])
    assert json.loads(capsys.readouterr().out)["stack_sha"] == rec["stack_sha"]


# --- the launcher ----------------------------------------------------------------------------

def test_launcher_scripts_parse():
    for f in ("paper_run.sh", "activate.sh", "build.sh"):
        subprocess.run(["bash", "-n", os.path.join(ENV_DIR, f)], check=True)


# --- the cuVS patches: git-ignored files, pinned by sha256 in versions.env ------------------

PATCH = "0001-sparse-knn-ip-merge-and-short-batch.diff"


def _check_env():
    import check as check_env
    return check_env


def test_library_patches_found_in_the_checkout_and_match_their_pins():
    """The patches live in baselines/cuvs/patches/, and every CUVS_PATCHES entry has a
    CUVS_PATCH_SHA256 pin equal to the file's hash: check.py / build.sh accept this tree."""
    ce = _check_env()
    d = provenance.library_patches_dir()
    assert d == os.path.join(provenance.ROOT, "baselines", "cuvs", "patches")
    want = ce.read_versions(provenance.VERSIONS_ENV)
    names = want["CUVS_PATCHES"].split()
    pins = dict(e.split("=", 1) for e in want["CUVS_PATCH_SHA256"].split())
    assert names and list(pins) == names
    for n in names:
        assert ce.sha256(os.path.join(d, n)) == pins[n], n
    # and the file row of check_env's patch check passes on this tree
    rows = ce.cuvs_patch_checks(want, "", d)
    assert rows[1][3] is True, rows[1]


def _patch_tree(tmp_path, content=b"the patch\n"):
    d = tmp_path / "library_patches"
    (d / "cuvs").mkdir(parents=True)
    (d / "cuvs" / PATCH).write_bytes(content)
    import hashlib
    return str(d / "cuvs"), hashlib.sha256(content).hexdigest()


def test_cuvs_patch_check_match(tmp_path):
    ce = _check_env()
    d, h = _patch_tree(tmp_path)
    want = {"CUVS_PATCHES": PATCH, "CUVS_PATCH_SHA256": f"{PATCH}={h}"}
    stamp, files = ce.cuvs_patch_checks(want, f"{PATCH}={h}", d)
    assert stamp[3] and files[3], (stamp, files)


def test_cuvs_patch_check_mismatch(tmp_path):
    """A stamp or a file whose hash is not the pinned one fails (whatever is in the
    directory does not define the expectation); an unpinned patch fails too."""
    ce = _check_env()
    d, h = _patch_tree(tmp_path, b"edited patch\n")
    want = {"CUVS_PATCHES": PATCH, "CUVS_PATCH_SHA256": f"{PATCH}={'0' * 64}"}
    stamp, files = ce.cuvs_patch_checks(want, f"{PATCH}={h}", d)
    assert not stamp[3] and not files[3]
    assert stamp[2] == files[2] == f"{PATCH}={'0' * 64}"
    # the env stamped with the pinned hash, the tree's file edited since: only the file row fails
    stamp, files = ce.cuvs_patch_checks(want, f"{PATCH}={'0' * 64}", d)
    assert stamp[3] and not files[3]
    # no pin for a listed patch, or no stamp at all
    stamp, files = ce.cuvs_patch_checks({"CUVS_PATCHES": PATCH}, f"{PATCH}={h}", d)
    assert not stamp[3] and not files[3] and "not pinned" in stamp[2]
    stamp, _ = ce.cuvs_patch_checks({"CUVS_PATCHES": PATCH, "CUVS_PATCH_SHA256": f"{PATCH}={h}"},
                                    "", d)
    assert not stamp[3] and "not built by build.sh" in stamp[1]


def test_cuvs_patch_check_missing_file(tmp_path):
    ce = _check_env()
    want = {"CUVS_PATCHES": PATCH, "CUVS_PATCH_SHA256": f"{PATCH}={'a' * 64}"}
    stamp, files = ce.cuvs_patch_checks(want, f"{PATCH}={'a' * 64}", str(tmp_path / "none"))
    assert stamp[3] and not files[3]
    assert "missing" in files[1]


def _build_env_patches(tmp_path, patch_dir):
    env = dict(os.environ, WORK=str(tmp_path / "work"), CONDA=str(tmp_path / "conda"),
               CUVS_PATCH_DIR=patch_dir)
    return subprocess.run(["bash", os.path.join(ENV_DIR, "build.sh"), "patches"], env=env,
                          capture_output=True, text=True, timeout=60)


def test_build_env_verifies_patches_before_applying(tmp_path):
    """build_env.sh's `patches` step (also run first in `cuvs`): the checkout's patches pass;
    a changed file and a missing one are refused, naming the file."""
    r = _build_env_patches(tmp_path, provenance.library_patches_dir())
    assert r.returncode == 0 and "verified against CUVS_PATCH_SHA256" in r.stdout, r.stdout + r.stderr
    d, _ = _patch_tree(tmp_path, b"not the pinned patch\n")
    r = _build_env_patches(tmp_path, d)
    assert r.returncode != 0 and "PATCH MISMATCH" in r.stdout and PATCH in r.stdout, r.stdout
    r = _build_env_patches(tmp_path, str(tmp_path / "empty"))
    assert r.returncode != 0 and "not found" in r.stdout and PATCH in r.stdout, r.stdout


FAKE_SMI = """#!/usr/bin/env bash
# a fake nvidia-smi: -q, one-shot queries, and the -lms logger (until killed)
echo "$*" >> "$SMI_CALLS"
q=""; lms=""
for a in "$@"; do case $a in --query-gpu=*) q=${a#*=} ;; -lms) lms=1 ;; esac; done
[ "$1" = "-q" ] && { echo "fake nvidia-smi -q"; exit 0; }
case ,$q, in *,clocks_event_reasons.active,*) exit 2 ;; esac   # an "older driver"
if [ -n "$lms" ]; then
  echo "$PPID" > "$SMI_LOGGER_PPID"; echo $$ > "$SMI_LOGGER_PID"
  echo "$q"; while :; do echo "2026/11/13 00:00:00.000, 0, GPU-x"; sleep 0.05; done
fi
echo "1"
"""


@pytest.fixture
def stub_env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text(FAKE_SMI)
    (bin_ / "nvidia-smi").chmod(0o755)
    act = tmp_path / "stub_activate.sh"
    act.write_text(textwrap.dedent(f"""\
        export CONDA_PREFIX={sys.prefix}
        export PATH={bin_}:{os.path.dirname(sys.executable)}:$PATH
        """))
    chk = tmp_path / "stub_check.py"
    chk.write_text(textwrap.dedent("""\
        import json, os, sys
        out = sys.argv[sys.argv.index("--json") + 1]
        rc = int(os.environ.get("STUB_CHECK_RC", "0"))
        json.dump({"ok": rc == 0, "sha256": {}}, open(out, "w"))
        print("stub check", "ok" if rc == 0 else "FAIL")
        sys.exit(rc)
        """))
    code = tmp_path / "code"
    code.mkdir()
    (code / "a.py").write_text("x = 1\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PAPER_")}
    env.update(PAPER_ACTIVATE=str(act), PAPER_CHECK_ENV=str(chk), PAPER_CODE_ROOTS=str(code),
               SMI_CALLS=str(tmp_path / "smi_calls"), SMI_LOGGER_PID=str(tmp_path / "smi_pid"),
               SMI_LOGGER_PPID=str(tmp_path / "smi_ppid"),
               PAPER_STACK_SHA="stale-from-an-outer-job")   # must not survive the launcher
    return tmp_path, env, code


def _job(tmp_path, body):
    job = tmp_path / "job.py"
    job.write_text(textwrap.dedent(body))
    return job


def test_launcher_stubbed_dry_run(stub_env):
    tmp_path, env, code = stub_env
    env["PAPER_SMI_MS"] = "200"   # the logger is opt-in
    job = _job(tmp_path, """\
        import json, os, sys, time
        time.sleep(0.4)   # the logger samples meanwhile
        json.dump({k: v for k, v in os.environ.items() if k.startswith("PAPER_")},
                  open(sys.argv[1], "w"))
        sys.exit(7)
        """)
    run_dir = tmp_path / "run"
    seen = tmp_path / "seen.json"
    r = subprocess.run(["bash", LAUNCHER, "--run-dir", str(run_dir), "--", sys.executable,
                        str(job), str(seen)], env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 7, r.stderr          # the job's own exit status
    assert "STUBBED launch" in r.stderr
    got = json.loads(seen.read_text())
    assert got["PAPER_STACK_SHA"] == provenance.versions_sha() == _shell_sha(provenance.VERSIONS_ENV)
    assert got["PAPER_ENV_PREFIX"] == sys.prefix and got["PAPER_RUN_DIR"] == str(run_dir)
    assert got["PAPER_LAUNCH_STUBBED"] == "1" and got["PAPER_GPU_LOG"] == str(run_dir / "gpu_smi.csv")
    # the code hash: the roots given, plus the job's own script
    rec = json.loads((run_dir / "code.json").read_text())
    assert got["PAPER_CODE_SHA"] == rec["code_sha"] == code_sha.tree_sha(
        [str(code), str(job)], provenance.ROOT)[0]
    # what the run dir keeps
    for f in ("check_env.json", "check_env.txt", "launch.env", "launch.log", "gpu_smi.csv",
              "gpu_state_start.csv", "gpu_state_end.csv", "nvidia_smi_q_start.txt",
              "nvidia_smi_q_end.txt"):
        assert (run_dir / f).exists(), f
    assert "stubbed=1" in (run_dir / "launch.env").read_text()
    log = (run_dir / "gpu_smi.csv").read_text().splitlines()
    assert log[0].startswith("timestamp,index,uuid,memory.used,utilization.gpu")
    assert "clocks_throttle_reasons.active" in log[0]   # the older spelling, probed
    assert len(log) > 2
    # the logger was stopped at exit
    pid = int((tmp_path / "smi_pid").read_text())
    with pytest.raises(OSError):
        os.kill(pid, 0)
    # a harness GPU row under it would be refused: the stub never checked the stack
    assert "stub" in provenance.stack_problem(environ=got)


def test_launcher_runs_no_gpu_logger_by_default(stub_env):
    """without PAPER_SMI_MS no nvidia-smi logger runs (no gpu_smi.csv, no -lms call,
    no PAPER_GPU_LOG); the GPU state at job start and end is still recorded."""
    tmp_path, env, _ = stub_env
    job = _job(tmp_path, """\
        import json, os, sys
        json.dump({k: v for k, v in os.environ.items() if k.startswith("PAPER_")},
                  open(sys.argv[1], "w"))
        """)
    run_dir = tmp_path / "run"
    seen = tmp_path / "seen.json"
    r = subprocess.run(["bash", LAUNCHER, "--run-dir", str(run_dir), "--", sys.executable,
                        str(job), str(seen)], env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert "PAPER_GPU_LOG" not in json.loads(seen.read_text())
    assert not (run_dir / "gpu_smi.csv").exists()
    assert "-lms" not in (tmp_path / "smi_calls").read_text()
    for f in ("gpu_state_start.csv", "gpu_state_end.csv", "nvidia_smi_q_start.txt",
              "nvidia_smi_q_end.txt"):
        assert (run_dir / f).exists(), f


def test_launcher_refuses_when_check_env_fails(stub_env):
    tmp_path, env, _ = stub_env
    marker = tmp_path / "ran"
    job = _job(tmp_path, f"open({str(marker)!r}, 'w').write('x')\n")
    env["STUB_CHECK_RC"] = "1"
    run_dir = tmp_path / "run"
    r = subprocess.run(["bash", LAUNCHER, "--run-dir", str(run_dir), sys.executable, str(job)],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "check.py FAILED" in r.stderr
    assert not marker.exists()                    # the job never ran
    assert not (run_dir / "gpu_smi.csv").exists()  # nor the logger
    assert not os.path.exists(tmp_path / "smi_calls") or "-lms" not in (
        tmp_path / "smi_calls").read_text()


def test_launcher_forwards_term_and_stops_the_logger(stub_env):
    import signal
    import time
    tmp_path, env, _ = stub_env
    env["PAPER_SMI_MS"] = "200"   # the logger is opt-in
    job = _job(tmp_path, """\
        import signal, sys, time
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(42))
        open(sys.argv[1], "w").write("up")
        time.sleep(30)
        """)
    up = tmp_path / "up"
    p = subprocess.Popen(["bash", LAUNCHER, "--run-dir", str(tmp_path / "run"), sys.executable,
                          str(job), str(up)], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    while not (up.exists() and (tmp_path / "smi_pid").exists()) and time.time() - t0 < 30:
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 42, err   # the job's own exit, after the forwarded TERM
    pid = int((tmp_path / "smi_pid").read_text())
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_launcher_default_roots_in_a_repo_checkout(stub_env):
    """Without PAPER_CODE_ROOTS, in this checkout: clean_benchmarks' bench (and its data/,
    which code_sha.py would skip by name), env, baselines, experiments, and python/nova-bf."""
    tmp_path, env, _ = stub_env
    del env["PAPER_CODE_ROOTS"]
    run_dir = tmp_path / "run"
    r = subprocess.run(["bash", LAUNCHER, "--run-dir", str(run_dir), "true"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    roots = json.loads((run_dir / "code.json").read_text())["roots"]
    for want in ("clean_benchmarks/bench", "clean_benchmarks/bench/data",
                 "clean_benchmarks/env", "clean_benchmarks/baselines",
                 "clean_benchmarks/experiments", "python/nova-bf",
                 "clean_benchmarks/tests", "clean_benchmarks/tests/data",
                 "clean_benchmarks/tests/fixtures",
                 "clean_benchmarks/pyproject.toml"):
        assert want in roots, roots
    assert "crates/nova-textscan" not in roots          # not a filtered experiment


def test_launcher_repo_roots_add_the_text_scanner_for_a_filtered_experiment(stub_env):
    """a dev-box run of a filtered experiment hashes crates/nova-textscan too."""
    tmp_path, env, _ = stub_env
    del env["PAPER_CODE_ROOTS"]
    run_dir = tmp_path / "run"
    exp = os.path.join(os.path.dirname(ENV_DIR), "tests", "fixtures", "experiments", "filtered", "structured.yaml")
    r = subprocess.run(["bash", LAUNCHER, "--run-dir", str(run_dir), "true", exp], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    roots = json.loads((run_dir / "code.json").read_text())["roots"]
    assert "crates/nova-textscan" in roots and "clean_benchmarks/pyproject.toml" in roots, roots


def test_launcher_in_a_stage_matches_the_staged_hash(stub_env):
    """A stage (a copy of clean_benchmarks; no python/nova-bf one level up): the whole stage is
    hashed, and a code change after staging is flagged against STAGED."""
    import shutil
    tmp_path, env, _ = stub_env
    del env["PAPER_CODE_ROOTS"]
    stage = tmp_path / "stage"
    stage.mkdir(parents=True)
    shutil.copytree(ENV_DIR, stage / "env",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "harness").mkdir()
    (stage / "harness" / "run.py").write_text("print('harness')\n")
    (stage / "run.sh").write_text("echo job\n")
    staged = code_sha.tree_sha([str(stage)], str(stage))[0]
    (stage / "STAGED").write_text(f"staged x\ncode sha256: {staged}\n")
    launcher = str(stage / "env" / "paper_run.sh")
    r = subprocess.run(["bash", launcher, "--run-dir", str(tmp_path / "r1"), "bash",
                        str(stage / "run.sh")], env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "r1" / "code.json").read_text())["code_sha"] == staged
    assert "differs from the stage" not in r.stderr
    (stage / "harness" / "run.py").write_text("print('edited on the node')\n")
    r = subprocess.run(["bash", launcher, "--run-dir", str(tmp_path / "r2"), "true"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "differs from the stage" in r.stderr
