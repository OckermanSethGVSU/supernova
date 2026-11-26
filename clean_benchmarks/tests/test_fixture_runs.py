"""tests/helpers/fixture_runs.py: the cache key follows the content it hashes, and a copy is
rewritten to its own path (no `bench run` here)."""

import json
import os
import shutil

import pytest

from helpers import fixture_runs as F


def test_the_key_changes_with_a_hashed_file_and_only_then(tmp_path):
    tree = tmp_path / "bench"
    shutil.copytree(os.path.join(F.CB, "bench"), tree,
                    ignore=shutil.ignore_patterns("__pycache__"))
    env = {"PATH": "/usr/bin", "HARNESS_NO_IMDS": "1"}
    k0, desc = F.cache_key([str(tree)], env)
    assert F.cache_key([str(tree)], env)[0] == k0                   # stable
    assert desc["trees"][str(tree)].startswith(f"{len(F._files(str(tree)))} files ")
    p = tree / "runner.py"
    orig = p.read_bytes()
    p.write_bytes(orig + b"\n# a change\n")
    assert F.cache_key([str(tree)], env)[0] != k0                   # content
    p.write_bytes(orig)
    assert F.cache_key([str(tree)], env)[0] == k0
    (tree / "new_module.py").write_text("x = 1\n")
    assert F.cache_key([str(tree)], env)[0] != k0                   # a new file
    os.remove(tree / "new_module.py")
    (tree / "__pycache__").mkdir(exist_ok=True)
    (tree / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"junk")
    assert F.cache_key([str(tree)], env)[0] == k0                   # caches are not code


def test_the_key_follows_the_env_the_code_reads():
    roots = [os.path.join(F.CB, "bench")]
    env = {"PATH": "/usr/bin", "HARNESS_NO_IMDS": "1"}
    k0 = F.cache_key(roots, env)[0]
    assert F.cache_key(roots, {**env, "HARNESS_NO_IMDS": "0"})[0] != k0
    assert F.cache_key(roots, {**env, "OMP_NUM_THREADS": "4"})[0] != k0       # by prefix
    assert F.cache_key(roots, {**env, "CUDA_VISIBLE_DEVICES": "0"})[0] != k0
    for k in ("BENCH_TEST_RUNS_REBUILD", "PYTEST_XDIST_WORKER", "AWS_SESSION_TOKEN",
              "SOME_UNREAD_VAR"):
        assert F.cache_key(roots, {**env, k: "1"})[0] == k0, k


def test_the_default_roots_cover_the_code_a_run_executes():
    roots = F._roots()
    for rel in ("bench", "baselines", "env", os.path.join("tests", "fixtures", "experiments"),
                os.path.join("tests", "fixtures", "novabf")):
        assert os.path.join(F.CB, rel) in roots, rel
    assert any(r.endswith(os.path.join("src", "nova_bf")) for r in roots)
    assert os.path.abspath(F.__file__) in roots


def test_a_copy_names_its_own_path_and_refuses_a_binary_holding_the_old_one(tmp_path):
    src = tmp_path / "cache"
    old = str(tmp_path / ".build-x" / "res")
    run = src / "res" / "s4_dense" / "r1"
    (run / "legs").mkdir(parents=True)
    (run / "legs" / "a.json").write_text(json.dumps({"p": f"{old}/s4_dense/r1/x.parquet"}))
    (run / "logs").mkdir()
    (run / "logs" / "a.log").write_text(f"wrote {old}/s4_dense/r1\n")
    meta = {"build_root": old, "inputs_ok": {}, "shapes": {
        "dense": {"args": ["bench", "--results", old], "returncode": 0,
                  "stdout": f"results in {old}/s4_dense/r1", "stderr": ""}}}
    (src / "meta.json").write_text(json.dumps(meta))
    res, procs = F._copy(str(src), str(tmp_path / "copy"))
    assert res == str(tmp_path / "copy" / "res")
    got = json.load(open(os.path.join(res, "s4_dense", "r1", "legs", "a.json")))
    assert got["p"] == f"{res}/s4_dense/r1/x.parquet"
    assert open(os.path.join(res, "s4_dense", "r1", "logs", "a.log")).read() == f"wrote {res}/s4_dense/r1\n"
    assert procs["dense"].stdout == f"results in {res}/s4_dense/r1" and procs["dense"].args[-1] == res
    assert old in (run / "legs" / "a.json").read_text()             # the cache is untouched
    (run / "legs" / "a.topk.npz").write_bytes(b"PK\0" + old.encode())
    with pytest.raises(RuntimeError, match="not text"):
        F._copy(str(src), str(tmp_path / "copy2"))
