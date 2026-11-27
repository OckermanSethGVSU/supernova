"""Command-line interface for benchmark execution.

Commands:
    run            Run an experiment.
    leg            Execute a single system leg.
    summarize      Summarize benchmark results.
    sweep          Run parameter sweeps.
    check          Validate the environment.
    verify         Re-run the checks on a finished run dir (checks.json).
    plot           Generate figures from saved results.
    launch         Stage and run benchmarks on a SkyPilot node.

Engines run in separate processes to keep CLI imports lightweight.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # clean_benchmarks
ENV_DIR = os.path.join(ROOT, "env")

# name -> one-line help
COMMANDS = {
    "run": "run one experiment: every system, every round, then checks and summary",
    "leg": "one leg (one system, one round) in a fresh process; what `run` starts",
    "summarize": "medians, per-round values and excluded legs of a results dir",
    "sweep": "run a sweep file's grid of variants; --pick prints the winner + YAML change",
    "launch": "run an experiment on a SkyPilot node (env/node.yaml) and sync results back",
    "plot": "figures from saved results only (one time attribution per search)",
    "check": "check the pinned paper environment (env/check.py); extra ARGS go to it",
    "verify": "the checks again on a finished run dir (CHECKS_REDESIGN.md): rewrites checks.json",
}


def cmd_check(rest):
    """Run env/check.py in a fresh interpreter (as paper_run.sh does): the check loads the
    engines' CUDA libraries, which must not leak into this process."""
    check = os.path.join(ENV_DIR, "check.py")
    versions = os.path.join(ENV_DIR, "versions.env")
    if not os.path.isfile(check):
        print(f"bench check: {check} not found (bench must run from a clean_benchmarks checkout)",
              file=sys.stderr)
        return 2
    rest = list(rest)
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if "--versions" not in rest:
        rest = ["--versions", versions] + rest
    return subprocess.call([sys.executable, check] + rest)


def build_parser():
    """The top-level parser, used only for --help and unknown commands."""
    ap = argparse.ArgumentParser(prog="bench", description="clean_benchmarks: one command per "
                                 "paper experiment")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    for name, help_ in COMMANDS.items():
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    """Dispatch to the subcommand, which parses its own arguments."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["check"]:   # every later argument is check.py's (`bench check --help` too)
        return cmd_check(argv[1:])
    # each subcommand parses its own arguments and imports only what it needs (no engine)
    if argv[:1] == ["run"]:
        from bench import runner
        return runner.cmd_run(argv[1:])
    if argv[:1] == ["sweep"]:
        from bench import runner
        return runner.cmd_sweep(argv[1:])
    if argv[:1] == ["leg"]:
        from bench import leg
        return leg.main(argv[1:])
    if argv[:1] == ["summarize"]:
        from bench.report import summary
        return summary.main(argv[1:])
    if argv[:1] == ["verify"]:
        from bench.verify import check
        return check.main(argv[1:])
    if argv[:1] == ["launch"]:
        from bench import launch
        return launch.main(argv[1:])
    if argv[:1] == ["plot"]:
        from bench.report import plots
        return plots.main(argv[1:])
    ap = build_parser()
    ap.parse_known_args(argv)   # --help / an unknown command exit here
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
