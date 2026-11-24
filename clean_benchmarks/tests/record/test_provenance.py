"""bench/record/provenance.py: where nova_bf and the bench code are imported from; a child
process run from a copy of the tree is refused."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from bench import leg as L


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


@pytest.fixture
def tree_copy(tmp_path):
    """A copy of bench/ (and env/check.py) at tmp/cb: its scripts, run by path, import the
    INSTALLED bench (this tree's), not the copy's."""
    cb = tmp_path / "cb"
    shutil.copytree(os.path.join(CB, "bench"), cb / "bench",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (cb / "env").mkdir()
    for f in ("check.py", "versions.env"):
        shutil.copy(os.path.join(CB, "env", f), cb / "env" / f)
    return cb


def _env():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "HARNESS_NO_IMDS": "1"}


def test_the_novabf_child_from_a_copy_is_refused(tree_copy, tmp_path):
    out = tmp_path / "clock.json"
    p = subprocess.run([PY, str(tree_copy / "bench" / "systems" / "novabf.py"), "compute",
                        str(tmp_path / "no_such.yaml"), "--clock-json", str(out)],
                       cwd=str(tree_copy), capture_output=True, text=True, env=_env(), timeout=300)
    assert p.returncode == 4 and "REFUSED" in p.stderr, p.stderr[-2000:]
    rec = json.load(open(out))
    o = rec["import_origins"]
    assert rec["ok"] is False and o["ok"] is False
    assert o["bench"].startswith(os.path.join(CB, "bench")) and "not this tree's" in o["problems"][0]
    assert o["expected"]["bench"] == str(tree_copy / "bench")


@pytest.mark.slow
def test_the_harness_child_from_a_copy_is_refused_before_reading(tree_copy, tmp_path):
    out = tmp_path / "clock.json"
    p = subprocess.run([PY, str(tree_copy / "bench" / "systems" / "harness_cli.py"), "--regime",
                        "memory", "--corpus", str(tmp_path / "nothing"), "--nq", "8", "--backend",
                        "cpu", "--csv", str(tmp_path / "x.csv"), "--clock-json", str(out)],
                       cwd=str(tree_copy), capture_output=True, text=True, env=_env(), timeout=300)
    assert p.returncode == 4 and "harness_cli: REFUSED" in p.stderr, p.stderr[-2000:]
    assert not (tmp_path / "x.csv").exists()
    o = json.load(open(out))["import_origins"]
    assert o["ok"] is False and o["expected"]["bench"] == str(tree_copy / "bench")
    # the leg turns that record into its failure
    spec = L.LegSpec(outputs={"csv": str(tmp_path / "x.csv"), "clock": str(out),
                              "topk": str(tmp_path / "x.npz")}, e2e=False)
    fails = []
    L.check_harness(spec, {"subset": None}, "", fails)
    assert any(f.startswith("import origins: bench is imported from") for f in fails), fails


def test_this_tree_passes_and_the_leg_records_both_origins():
    from bench.record import provenance
    o = provenance.check(CB)
    assert o["ok"], o["problems"]
    assert o["bench"] == os.path.join(CB, "bench", "__init__.py")
    assert o["nova_bf"].startswith(os.path.join(os.path.dirname(CB), "python", "nova-bf"))
    p = provenance.probe(CB)
    assert p["ok"] and p["bench"] == o["bench"]
    fails, out = [], {}
    L._origins_ok({"clocks": []}, out, fails)          # an engine that recorded none fails
    assert fails == ["the engine process recorded no import origins"]


@pytest.mark.slow
def test_check_py_from_a_copy_fails_its_import_origins_row(tree_copy, tmp_path):
    j = tmp_path / "chk.json"
    subprocess.run([PY, str(tree_copy / "env" / "check.py"), "--versions",
                    str(tree_copy / "env" / "versions.env"), "--report-only", "--json", str(j),
                    "--engines", "nova_bf"], cwd=str(tmp_path), capture_output=True, text=True,
                   env=_env(), timeout=600)
    doc = json.load(open(j))
    hit = [r for r in doc["checks"] if r["name"] == "import origins (bench / nova_bf, python -P)"]
    assert len(hit) == 1 and hit[0]["ok"] is False and "not this tree's" in hit[0]["got"]
    assert "import origins (bench / nova_bf, python -P)" in doc["mismatches"]


@pytest.mark.parametrize("rel,ok", [
    ("src/nova_bf/__init__.py", True),
    ("src/nova_bf/compute.py", True),
    (".venv/lib/python3.12/site-packages/nova_bf/__init__.py", False),
    ("build/lib/nova_bf/__init__.py", False),
    ("tests/nova_bf/__init__.py", False),
    ("src/nova_bf_copy/__init__.py", False),
])
def test_nova_bf_resolves_only_under_src_nova_bf(rel, ok):
    from bench.record import provenance
    root = provenance.novabf_root(CB)
    r = provenance.judge(CB, {"bench": os.path.join(CB, "bench", "__init__.py"),
                           "nova_bf": os.path.join(root, rel)}, True)
    assert r["ok"] is ok, r["problems"]
    assert r["expected"]["nova_bf"] == os.path.join(root, "src", "nova_bf")


def test_this_env_still_resolves_nova_bf_under_src():
    from bench.record import provenance
    r = provenance.probe(CB)
    assert r["ok"], r["problems"]
