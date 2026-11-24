"""bench/launch.py: syncing a launch's results back, and which runs belong to the launch."""

import json
import os
import sys

import pytest


def _sweep_launch(root, ok=True, table=True):
    ld = os.path.join(root, "sweeps", "dense_coyo_io", "launch", "20991119T000000Z_bbbbbbb")
    os.makedirs(ld)
    json.dump({"cluster": "bench-x", "experiment": "dense_coyo_io", "results_root": root,
               "commit": "bbbbbbb", "code_sha": "c0de", "job": "sweep",
               "results_dir": "sweeps/dense_coyo_io", "launch_id": "20991119T000000Z_bbbbbbb"},
              open(os.path.join(ld, "launch.json"), "w"))
    calls = []

    def rsync(cmd):
        calls.append(cmd)
        if cmd[2] == "bench-x:~/res/sweeps/dense_coyo_io/":      # the node's ~/res/sweeps/<sweep>/
            d = os.path.join(cmd[3], "20991119T010000Z_bbbbbbb")
            os.makedirs(d)
            json.dump({"launch_id": "20991119T000000Z_bbbbbbb"}, open(os.path.join(d, "manifest.json"), "w"))
            json.dump({"ok": ok, "problems": [] if ok else ["x failed"]},
                      open(os.path.join(d, "checks.json"), "w"))
            open(os.path.join(d, "summary.md"), "w").write("| a |\n")
            if table:
                open(os.path.join(d, "sweep_table.csv"), "w").write("variant\n")
        return 0
    return ld, calls, rsync


def test_a_sweep_launch_syncs_its_results_where_chosen_from_points(tmp_path):
    from bench import launch as LA
    root = str(tmp_path)
    ld, calls, rsync = _sweep_launch(root)
    assert LA.sync(ld, runner=rsync) == 0
    run = os.path.join(root, "sweeps", "dense_coyo_io", "20991119T010000Z_bbbbbbb")
    assert os.path.isfile(os.path.join(run, "launch.json"))
    assert ["rsync", "-a", "bench-x:~/res/sweeps/dense_coyo_io/",
            os.path.join(root, "sweeps", "dense_coyo_io") + "/"] in calls
    assert ["rsync", "-a", "--exclude", "/sweeps/dense_coyo_io/", "bench-x:~/res/",
            os.path.join(ld, "node", "res") + "/"] in calls


@pytest.mark.parametrize("ok,table,want", [(False, True, "checks failed"),
                                           (True, False, "no sweep_table.csv")])
def test_a_failed_sweep_is_a_sync_problem(tmp_path, ok, table, want):
    from bench import launch as LA
    ld, _, rsync = _sweep_launch(str(tmp_path), ok, table)
    assert LA.sync(ld, runner=rsync) == 1
    assert any(want in p for p in json.load(open(os.path.join(ld, "synced.json")))["problems"])


LID = "20261120T100000Z_bbbbbbb"


def _launch_dir(root, job="run", exp="dense_coyo", lid=LID, rel=None):
    res = os.path.join(root, "results")
    ld = os.path.join(res, *(rel or exp).split("/"), "launch", lid)
    os.makedirs(ld)
    json.dump({"cluster": "bench-x", "experiment": exp, "results_root": res, "job": job,
               "code_sha": "NEWSHA", "launch_id": lid, **({"results_dir": rel} if rel else {})},
              open(os.path.join(ld, "launch.json"), "w"))
    return res, ld


def _run(d, lid, ok=True):
    os.makedirs(d, exist_ok=True)
    json.dump({"code_sha": "NEWSHA", "launch_id": lid}, open(os.path.join(d, "manifest.json"), "w"))
    json.dump({"ok": ok}, open(os.path.join(d, "checks.json"), "w"))
    open(os.path.join(d, "summary.md"), "w").write("ok\n")


def _missing_source(cmd):
    print(f'rsync: [sender] change_dir "/home/ubuntu/res/{cmd[2].split("/res/")[1]}" failed: No '
          "such file or directory (2)\nrsync error: some files/attrs were not transferred (see "
          "previous errors) (code 23) at main.c(1338) [Receiver=3.2.7]", file=sys.stderr)
    return 23


def test_a_sweep_launch_never_rsyncs_the_run_results_dir(tmp_path):
    from bench import launch as LA
    res, ld = _launch_dir(str(tmp_path), job="sweep", exp="dense_coyo_io", rel="sweeps/dense_coyo_io")
    calls = []

    def rsync(cmd):
        calls.append(cmd)
        if cmd[2] == "bench-x:~/res/dense_coyo_io/":   # a sweep writes nothing there
            return _missing_source(cmd)
        if cmd[2] == "bench-x:~/res/sweeps/dense_coyo_io/":
            d = os.path.join(cmd[3], "20261120T110000Z_bbbbbbb")
            _run(d, LID)
            open(os.path.join(d, "sweep_table.csv"), "w").write("variant\n")
        return 0
    assert LA.sync(ld, runner=rsync) == 0
    assert not any(c[2] == "bench-x:~/res/dense_coyo_io/" for c in calls)
    s = json.load(open(os.path.join(ld, "synced.json")))
    assert s["runs"] == ["20261120T110000Z_bbbbbbb"] and s["problems"] == []
    assert os.path.isfile(os.path.join(res, "sweeps", "dense_coyo_io", "20261120T110000Z_bbbbbbb",
                                       "launch.json"))


def test_p_sync_an_earlier_launchs_run_on_a_reused_cluster_is_not_this_launch(tmp_path):
    """p_sync.py: the cluster's ~/res/<exp> holds an EARLIER launch's run; this launch's job died."""
    from bench import launch as LA
    res, ld = _launch_dir(str(tmp_path))

    def rsync(cmd):
        if cmd[2] == "bench-x:~/res/dense_coyo/":
            _run(os.path.join(cmd[3], "20261119T090000Z_aaaaaaa"), "20261119T080000Z_aaaaaaa")
        return 0
    assert LA.sync(ld, runner=rsync) != 0
    s = json.load(open(os.path.join(ld, "synced.json")))
    assert s["runs"] == [] and s["not_this_launch"] == ["20261119T090000Z_aaaaaaa"]
    assert any("no run of this launch" in p for p in s["problems"])
    assert not os.path.exists(os.path.join(res, "dense_coyo", "20261119T090000Z_aaaaaaa", "launch.json"))


def test_only_the_run_naming_this_launch_id_is_stamped(tmp_path):
    from bench import launch as LA
    res, ld = _launch_dir(str(tmp_path))
    _run(os.path.join(res, "dense_coyo", "20261101T000000Z_aaaaaaa"), None)   # an old local run

    def rsync(cmd):
        if cmd[2] == "bench-x:~/res/dense_coyo/":
            _run(os.path.join(cmd[3], "20261119T090000Z_aaaaaaa"), "20261119T080000Z_aaaaaaa")
            _run(os.path.join(cmd[3], "20261120T110000Z_bbbbbbb"), LID)
        return 0
    assert LA.sync(ld, runner=rsync) == 0
    s = json.load(open(os.path.join(ld, "synced.json")))
    assert s["runs"] == ["20261120T110000Z_bbbbbbb"] and s["new"] == ["20261120T110000Z_bbbbbbb"]
    assert s["not_this_launch"] == ["20261101T000000Z_aaaaaaa", "20261119T090000Z_aaaaaaa"]
    stamped = [d for d in os.listdir(os.path.join(res, "dense_coyo"))
               if os.path.exists(os.path.join(res, "dense_coyo", d, "launch.json"))]
    assert stamped == ["20261120T110000Z_bbbbbbb"]
    # a second sync of the same launch still finds its run (already here) and passes
    assert LA.sync(ld, runner=lambda c: 0) == 0
