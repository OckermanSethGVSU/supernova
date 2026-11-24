"""The shared-GPU check re-checks a listed process for a few seconds before refusing a leg: a
process that just exited can still show in nvidia-smi for a moment."""
import os
import stat

from bench import leg as L


def _fake_smi(tmp_path, outputs):
    """An nvidia-smi on PATH that prints outputs[0], outputs[1], ... on successive calls."""
    state = tmp_path / "calls"
    state.write_text("0")
    for i, out in enumerate(outputs):
        (tmp_path / f"out{i}").write_text(out)
    exe = tmp_path / "nvidia-smi"
    exe.write_text(f"""#!/bin/sh
n=$(cat {state}); echo $((n + 1)) > {state}
last={len(outputs) - 1}; [ "$n" -gt "$last" ] && n=$last
cat {tmp_path}/out$n
""")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return state


def _spec():
    return {"system_kind": "gpu"}


def test_a_process_that_goes_away_does_not_refuse_the_leg(tmp_path, monkeypatch):
    state = _fake_smi(tmp_path, ["123, python\n", "123, python\n", ""])
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert L.gpu_foreign_procs(_spec(), dev=False, wait_s=0) == ([], None)
    assert state.read_text().strip() == "3"   # listed twice, free on the third look


def test_a_process_that_stays_refuses_after_all_tries(tmp_path, monkeypatch):
    state = _fake_smi(tmp_path, ["123, python\n"])
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    procs, why = L.gpu_foreign_procs(_spec(), dev=False, tries=4, wait_s=0)
    assert procs == [{"pid": 123, "name": "python"}] and "pid 123 (python)" in why
    assert state.read_text().strip() == "4"
