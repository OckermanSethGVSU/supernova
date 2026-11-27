"""The CLI: every subcommand is listed, `bench check` passes its arguments through."""

import pytest

from bench import cli


def test_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for name in ("run", "leg", "summarize", "sweep", "launch", "check", "plot", "verify"):
        assert name in out
    assert "tune" not in out and "check-copies" not in out and "check-verify" not in out


def test_tune_is_not_a_command(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["tune", "x.yaml"])
    assert e.value.code == 2 and "invalid choice: 'tune'" in capsys.readouterr().err


def test_check_passes_versions_and_args_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call", lambda argv: seen.setdefault("argv", argv) and 0)
    assert cli.main(["check", "--report-only"]) == 0
    argv = seen["argv"]
    assert argv[1].endswith("env/check.py")
    assert argv[2:] == ["--versions", cli.os.path.join(cli.ENV_DIR, "versions.env"),
                        "--report-only"]
