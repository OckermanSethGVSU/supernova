"""env/paper_run.sh and env/check.py."""

import json
import os
import subprocess
import sys

import pytest

from bench import experiment as E


CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("base,textscan", [("filtered_structured", True), ("sparse_fineweb", False)])
def test_paper_run_hashes_textscan_for_a_filtered_sweep_file(tmp_path, base, textscan):
    """A sweep file names no `modality:` itself (its base does): the old grep missed it."""
    repo = os.path.dirname(CB)
    if not os.path.isdir(os.path.join(repo, "crates", "nova-textscan")):
        pytest.skip("no crates/nova-textscan in this checkout")
    sw = tmp_path / "sw.yaml"
    sw.write_text(f"name: sw\nbase: {os.path.join(CB, E.path_of(base))}\n"
                  "sweep: {system: novabf}\nsubset: {files: 1}\n")
    act, chk = tmp_path / "act.sh", tmp_path / "chk.py"
    act.write_text(f"export CONDA_PREFIX={os.path.dirname(os.path.dirname(sys.executable))}\n"
                   f"export PATH={os.path.dirname(sys.executable)}:$PATH\n")
    chk.write_text("import sys; sys.exit(0)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PAPER_")}
    env.update(PAPER_ACTIVATE=str(act), PAPER_CHECK_ENV=str(chk), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(["bash", os.path.join(CB, "env", "paper_run.sh"), "--run-dir", str(tmp_path / "pr"),
                        "true", str(sw)], env=env, cwd=str(tmp_path), capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    roots = json.load(open(tmp_path / "pr" / "code.json"))["roots"]
    want = {os.path.join(repo, "crates", "nova-textscan"), os.path.join(repo, "Cargo.lock"),
            os.path.join(repo, "Cargo.toml")}
    got = {os.path.join(repo, x) if not os.path.isabs(x) else x for x in roots}
    assert (want <= got) is textscan and not (want & got and not textscan), roots
