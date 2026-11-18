"""Run a test file again in a child pytest under the Triton interpreter.

`TRITON_INTERPRET` must be set before `triton` is imported, so the
interpreted kernel tests run in a SUBPROCESS. The interpreter is pure Python
and single-threaded, so the child is spread over pytest-xdist workers when
xdist is installed: the block-skip suite drops from ~43 min to ~8 min at 8
workers (2026-11-17, 24 cores).

The child gets the WHOLE machine, not the slice of CPUs its parent xdist
worker is pinned to (see `_pin_xdist_worker` in conftest.py): the parent is
one test that just waits, and the rest of the parallel pass finishes within
minutes, leaving the cores idle.

Width: `NOVA_BF_INTERPRET_WORKERS` (1 = serial), else half the CPUs, 1..16.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

# Set by conftest.py before an xdist worker narrows its affinity.
ALL_CPUS_ENV = "NOVA_BF_ALL_CPUS"
WORKERS_ENV = "NOVA_BF_INTERPRET_WORKERS"
_XDIST_ENV = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT",
              "PYTEST_XDIST_TESTRUNUID")


def all_cpus() -> set[int]:
    raw = os.environ.get(ALL_CPUS_ENV, "")
    if raw:
        return {int(c) for c in raw.split(",")}
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return set(range(os.cpu_count() or 1))


def workers() -> int:
    raw = os.environ.get(WORKERS_ENV, "").strip()
    if raw:
        n = int(raw)
        if n < 1:
            raise ValueError(f"{WORKERS_ENV} must be >= 1, got {raw!r}")
    else:
        n = max(1, min(16, len(all_cpus()) // 2))
    if n > 1 and importlib.util.find_spec("xdist") is None:
        n = 1
    return n


def run(path: str, child_flag: str, *, env_drop=(), markexpr: str = "",
        timeout: float = 3600) -> subprocess.CompletedProcess:
    """Run the test file `path` with TRITON_INTERPRET=1 and `child_flag`=1."""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env[child_flag] = "1"
    for k in (*env_drop, *_XDIST_ENV):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    n = workers()
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header",
           "-p", "no:cacheprovider", os.path.abspath(path)]
    if n > 1:
        cmd += ["-n", str(n), "--dist", "load"]
    if markexpr:
        cmd += ["-m", markexpr]
    cpus = all_cpus()

    def widen():                       # in the child, before exec
        try:
            os.sched_setaffinity(0, cpus)
        except (AttributeError, OSError):
            pass

    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=timeout, preexec_fn=widen)
