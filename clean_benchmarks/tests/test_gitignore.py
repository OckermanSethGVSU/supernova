""".gitignore: which result, sweep and launch files are tracked."""

import os
import subprocess

import pytest


CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_gitignore_tracks_the_run_identity_files_but_not_legs_or_logs():
    import subprocess
    d = os.path.join(CB, "results", "_gitignore_probe", "20990101T000000Z_x")
    os.makedirs(os.path.join(d, "legs"), exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    names = ["manifest.json", "experiment.json", "checks.json", "launch.json", "summary.csv",
             "summary.md", "legs/a.json", "logs/a.log", "other.json"]
    try:
        for n in names:
            open(os.path.join(d, n), "w").write("{}")
        rel = [os.path.relpath(os.path.join(d, n), CB) for n in names]
        p = subprocess.run(["git", "-C", CB, "check-ignore", "--no-index", *rel],
                           capture_output=True, text=True)
        if p.returncode not in (0, 1):
            pytest.skip(f"git check-ignore unavailable: {p.stderr}")
        ignored = {os.path.basename(x) if "/legs/" not in x and "/logs/" not in x
                   else x.split("/")[-2] + "/" + x.split("/")[-1] for x in p.stdout.split()}
        assert ignored == {"legs/a.json", "logs/a.log", "other.json"}
    finally:
        import shutil
        shutil.rmtree(os.path.join(CB, "results", "_gitignore_probe"))


@pytest.mark.parametrize("rel,tracked", [
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/sweep.json", True),
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/sweep_table.csv", True),
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/sweep_table.md", True),
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/events_table.md", True),
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/events_table.json", True),
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/manifest.json", True),
    ("results/sweeps/coyo_cuvs/20991119T000000Z_abc1234/legs/x.json", False),
])
def test_gitignore_tracks_the_sweep_evidence(rel, tracked):
    p = subprocess.run(["git", "-C", CB, "check-ignore", "--no-index", "-q", rel])
    assert (p.returncode == 1) is tracked, rel     # 1 = not ignored


LD = "results/dense_coyo/launch/20261120T100000Z_abc1234"
EV = "results/sweeps/dense_coyo_events/20261120T100000Z_abc1234"
SLD = "results/sweeps/dense_coyo_io/launch/20261120T100000Z_abc1234"


@pytest.mark.parametrize("rel,tracked", [
    (f"{LD}/launch.json", True),
    (f"{LD}/stage/tests/fixtures/experiments/run/manifest.json", False),
    (f"{LD}/stage/experiments/x/checks.json", False),
    (f"{LD}/stage/results/dense_coyo/r/summary.md", False),
    (f"{LD}/node/res/dense_coyo/r/summary.md", False),
    (f"{LD}/node/res/sweeps/dense_coyo_io/r/sweep_table.csv", False),
    (f"{SLD}/launch.json", True),
    (f"{SLD}/stage/experiments/x/checks.json", False),
    (f"{SLD}/node/res/manifest.json", False),
    (f"{LD}/node/res/manifest.json", False),
    (f"{EV}/events_table.md", True),
    (f"{EV}/events_table.json", True),
    (f"{EV}/manifest.json", True),
    (f"{EV}/legs/x.json", False),
    ("results/dense_coyo/20261120T100000Z_abc1234/summary.md", True),
    ("results/dense_coyo/20261120T100000Z_abc1234/launch.json", True),
])
def test_gitignore_launch_dirs_and_events_tables(rel, tracked):
    p = subprocess.run(["git", "-C", CB, "check-ignore", "--no-index", "-q", rel])
    assert (p.returncode == 1) is tracked, rel     # 1 = not ignored
