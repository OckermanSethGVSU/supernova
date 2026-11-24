"""Timing protocol and utilities for benchmark execution.

Each experiment runs three rounds by default, rotating system order between
rounds. Each system runs in a fresh process with one untimed warmup over the
first corpus file, followed by one timed pass over the corpus. Results are
reported as the median across rounds.

LegClock measures each system's execution time, with engine-specific timing
used to verify the measurement. Stages tracks per-stage wall-clock time.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from contextlib import contextmanager

# =============================================================================================
# 1. The timing rules
# =============================================================================================

ROUNDS_DEFAULT = 3
WARMUP_FILES = 1          # every leg: an untimed warmup over the first corpus file
REPEATS = 1               # ONE timed pass
HARNESS_PROTOCOL_ARGS = ("--warmup", str(WARMUP_FILES), "--warmup-files", str(WARMUP_FILES),
                         "--repeats", str(REPEATS))
# harness flags only the timing rules / the leg runner may pass (the leg runner also owns the
# output paths and the subset)
HARNESS_RESERVED = ("--warmup", "--warmup-files", "--repeats", "--csv", "--save-results",
                    "--max-files")
TIMEV = "/usr/bin/time"
NOVABF_DEVICE = "cuda"    # NOVA_BF_DEVICE of every nova-bf leg
DEV_NOVABF_DEVICE = "cpu"  # bench run --dev: a CPU dev box
# the kernel caches the cold end-to-end leg clears first (Triton, CUDA JIT)
E2E_CACHES = ("~/.triton/cache", "~/.nv/ComputeCache")


def rounds(exp: dict) -> int:
    return int(exp["protocol"].get("rounds", ROUNDS_DEFAULT))


def rotated(order: list, r: int) -> list:
    """The system order of round r (1-based): round 1 is `order`, each later round is rotated
    left by one more."""
    n = len(order)
    if n == 0:
        return []
    s = (r - 1) % n
    return list(order[s:]) + list(order[:s])


def schedule(exp: dict, n_rounds: int | None = None) -> list[tuple[int, str, str]]:
    """Every (round, search, system) in run order: rounds outer, searches in file order,
    each search's systems in that round's rotated order."""
    R = rounds(exp) if n_rounds is None else n_rounds
    out = []
    for r in range(1, R + 1):
        for s in exp["searches"]:
            for sy in rotated(s["systems"], r):
                out.append((r, s["name"], sy))
    return out


def tag(search: str, system: str, rnd: int) -> str:
    """A leg's file stem: <search>__<system>__r<N>."""
    return f"{search}__{system}__r{rnd}"


def e2e_tag(system: str) -> str:
    return f"e2e__{system}"


def leg_env(exp: dict) -> dict:
    """Env vars every leg process gets on top of the job env.
    PATH is left alone: activate.sh already put the paper env's bin first."""
    return {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(omp_threads(exp))}


def omp_threads(exp: dict) -> int:
    """The CPU thread count every system gets: the experiment's `threads`, or for 0 (auto) the
    CPUs this machine lets the leg use (affinity and cgroup quota, nova-bf's own rule). Never
    left to a library default: SkyPilot sets OMP_NUM_THREADS=1."""
    if exp["threads"] == 0:
        from bench.data.sources import usable_cpu_count
        return usable_cpu_count()
    return exp["threads"]


def timev_prefix() -> list[str]:
    return [TIMEV, "-v"] if os.access(TIMEV, os.X_OK) else []


# Env vars that move those caches: var -> (default dir, cache subdir). The cold e2e leg
# refuses to run if one points anywhere clear_e2e_caches() does not empty, because it never
# deletes a directory it was not written to clear.
E2E_CACHE_VARS = {"TRITON_CACHE_DIR": ("~/.triton/cache", ""),
                  "TRITON_HOME": ("~", ".triton/cache"),
                  "CUDA_CACHE_PATH": ("~/.nv/ComputeCache", "")}


def e2e_cache_redirects(env: dict) -> list[str]:
    """The E2E_CACHE_VARS in ``env`` that send a cache somewhere clear_e2e_caches() does not
    clear (empty: the leg may run)."""
    out = []
    for var, (default, sub) in E2E_CACHE_VARS.items():
        v = env.get(var)
        if v is None or v == "":
            continue
        got = os.path.realpath(os.path.join(os.path.expanduser(v), sub))
        want = os.path.realpath(os.path.expanduser(os.path.join(default, sub)))
        if got != want:
            out.append(f"{var}={v} moves a cache to {got}, which the cold e2e leg does not clear "
                       f"(it clears {', '.join(E2E_CACHES)}): unset it or point it there")
    return out


def clear_e2e_caches() -> list[str]:
    """For the cold e2e leg: delete the Triton kernel cache and the CUDA JIT cache."""
    done = []
    for p in E2E_CACHES:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
            done.append(p)
    return done


# Times search execution, excluding paused intervals.
# Starts before corpus reading and stops when the final top-k is in host memory.
# Invalid state transitions permanently fail the clock.
NEW, RUNNING, PAUSED, STOPPED, FAILED = "new", "running", "paused", "stopped", "failed"


class ClockError(RuntimeError):
    """An out-of-order clock call, or reading the time of a clock that did not stop."""


class LegClock:
    def __init__(self, now=time.perf_counter):
        self._now = now
        self.state = NEW
        self.events: list[tuple[str, float]] = []   # (transition, perf_counter time)
        self.failure: str | None = None
        self._t_start = self._t_pause = None
        self._paused = 0.0
        self.pauses = 0

    # -- transitions ---------------------------------------------------------------
    def _move(self, name: str, allowed: tuple[str, ...], to: str) -> float:
        if self.state not in allowed:
            msg = f"LegClock.{name}() while {self.state}"
            if self.state != FAILED:
                self.fail(msg)
            raise ClockError(msg)
        t = self._now()
        self.events.append((name, t))
        self.state = to
        return t

    def start(self) -> None:
        self._t_start = self._move("start", (NEW,), RUNNING)

    def pause(self) -> None:
        self._t_pause = self._move("pause", (RUNNING,), PAUSED)
        self.pauses += 1

    def resume(self) -> None:
        t = self._move("resume", (PAUSED,), RUNNING)
        self._paused += t - self._t_pause
        self._t_pause = None

    def stop(self) -> None:
        self._t_stop = self._move("stop", (RUNNING,), STOPPED)

    def fail(self, reason: str) -> None:
        """Mark the leg failed (a crash, an exception in the system, a bad transition).
        The first reason is kept."""
        if self.state != FAILED:
            self.failure = reason
            self.state = FAILED

    # -- readings ------------------------------------------------------------------
    @property
    def stopped(self) -> bool:
        return self.state == STOPPED

    @property
    def seconds(self) -> float:
        """Timed seconds: stop - start - paused. Only for a stopped clock."""
        if self.state != STOPPED:
            raise ClockError(f"LegClock has no time: state {self.state}"
                             + (f" ({self.failure})" if self.failure else ""))
        return (self._t_stop - self._t_start) - self._paused

    @property
    def paused_seconds(self) -> float:
        if self.state != STOPPED:
            raise ClockError(f"LegClock has no time: state {self.state}")
        return self._paused

    def segments(self) -> list[float]:
        """Durations of the running intervals (start/resume -> pause/stop), in order.
        Only for a stopped clock; they sum to `seconds`."""
        if self.state != STOPPED:
            raise ClockError(f"LegClock has no time: state {self.state}")
        out, t0 = [], None
        for name, t in self.events:
            if name in ("start", "resume"):
                t0 = t
            else:
                out.append(t - t0)
        return out

    def record(self) -> dict:
        """What a leg record stores about the clock (no time unless stopped)."""
        rec = {"state": self.state, "pauses": self.pauses, "failure": self.failure,
               "events": [name for name, _ in self.events]}
        if self.state == STOPPED:
            rec.update(seconds=self.seconds, paused_seconds=self._paused,
                       wall_seconds=self._t_stop - self._t_start, segments=self.segments())
        return rec


class ClockHooks:
    """A LegClock behind nova-bf's `timing_hooks` interface (corpus_start / pause / resume /
    done, nova_bf/timing_hooks.py). Observe only."""

    __slots__ = ("clock",)

    def __init__(self, clock: LegClock):
        self.clock = clock

    def corpus_start(self) -> None:
        self.clock.start()

    def pause(self) -> None:
        self.clock.pause()

    def resume(self) -> None:
        self.clock.resume()

    def done(self) -> None:
        self.clock.stop()


# =============================================================================================
# 3. The harness's stage timer
# =============================================================================================

now = time.perf_counter


class Stages:
    """Stage name -> summed seconds (or counts). ``add`` sums, ``peak`` keeps the maximum.
    Lock-protected, so prefetch worker threads can feed the instance the consumer reads."""

    def __init__(self):
        self._lock = threading.Lock()
        self.s = {}

    def add(self, name, seconds):
        with self._lock:
            self.s[name] = self.s.get(name, 0.0) + seconds

    def peak(self, name, value):
        with self._lock:
            self.s[name] = max(self.s.get(name, value), value)

    @contextmanager
    def span(self, name):
        t0 = now()
        try:
            yield
        finally:
            self.add(name, now() - t0)

    def get(self, name):
        with self._lock:
            return self.s.get(name, 0.0)

    def as_dict(self):
        with self._lock:
            return dict(self.s)
