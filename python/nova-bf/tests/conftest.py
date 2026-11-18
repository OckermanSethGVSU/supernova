"""Shared fixtures for the nova-bf test suite."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slice_only: a two-pass test of slice-mode-only machinery (padded "
        "GEMM heights, shape verification); skipped in the tiles-mode run")
    config.addinivalue_line(
        "markers",
        "slow: unique but expensive on CPU; `pytest -m 'not slow'` is the fast "
        "tier, a plain `pytest` (and `make test`) still runs everything. "
        "Assigned centrally in `_SLOW` below")
    config.addinivalue_line(
        "markers",
        "timing: verdict depends on thread/process scheduling. Runs in the "
        "normal (parallel) pass; `pytest -m timing` without -n reruns them solo "
        "if one looks flaky. Assigned centrally in `_TIMING` below")
    _check_expected_src()
    _pin_xdist_worker()


def _pin_xdist_worker():
    """Give each pytest-xdist worker a separate set of CPUs to prevent
    oversubscription. nova-bf sizes its thread pools using CPU affinity,
    and child processes inherit the same CPU limits.

    Set NOVA_BF_XDIST_PIN=0 to disable pinning.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not worker.startswith("gw") or os.environ.get("NOVA_BF_XDIST_PIN") == "0":
        return
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return
    # Remember the whole set: tests/interpret_child.py hands it back to the
    # interpreted child suites this worker spawns.
    os.environ.setdefault("NOVA_BF_ALL_CPUS", ",".join(map(str, cpus)))
    n = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    i = int(worker[2:])
    if n < 2 or len(cpus) < n:
        return
    share = cpus[i * len(cpus) // n:(i + 1) * len(cpus) // n]
    os.sched_setaffinity(0, share)
    try:
        import torch

        torch.set_num_threads(len(share))
    except Exception:  # noqa: BLE001 - torch is optional for some suites
        pass
    try:
        import pyarrow as pa

        pa.set_cpu_count(len(share))
    except Exception:  # noqa: BLE001
        pass

# Tests excluded from the fast tier, with their cost and faster coverage.
# All still run in the full tier.
_SLOW = (
    # Slow: end-to-end execution with interpreted GPU kernels and repeated
    # model probes. Fast tests cover kernel correctness and skip behavior;
    # GPU end-to-end tests provide additional coverage when available.
    ("test_sparse_blockskip_interpret.py::test_e2e_",
     "interpreted end-to-end run_compute"),
    # Other slow cases are marked directly in their test files.
)

# Tests that depend on thread or process scheduling, such as timing
# deadlines, overlap checks, or concurrency assertions.
# These normally run in parallel but may fail under heavy system load.
# If flaky, rerun alone with `pytest -m timing` to check for load issues.
# Excludes generous timeout guards and load-independent assertions.
# Entries are test nodeid substrings.
_TIMING = (
    # Thread drain / interrupt / permit tests built on sleeps and join timeouts.
    "test_merge_regressions.py::",
    # Exchange: overlap counts (max_active > 1), stuck-GET deadlines,
    # elapsed-time bounds, background-poller waits, and the spawned
    # concurrent-ranks scan that relies on interleaving.
    "test_threshold_exchange.py::test_publish_then_peer_polls_lists",
    "test_threshold_exchange.py::test_background_poller_delivers_without_manual_poll",
    "test_threshold_exchange.py::test_other_run_directory_is_invisible",
    "test_threshold_exchange.py::test_s3_style_write_path",
    "test_threshold_exchange.py::test_publisher_seq_is_monotone_and_writes_are_atomic",
    "test_threshold_exchange.py::test_spawn_helper_fails_fast_when_a_child_dies_silently",
    "test_threshold_exchange.py::test_concurrent_ranks_exchange_mid_scan",
    "test_threshold_exchange.py::test_concurrent_fetch_folds_every_changed_peer_in_one_version",
    "test_threshold_exchange.py::test_slow_peer_get_does_not_block_the_others",
    "test_threshold_exchange.py::test_close_does_not_hang_on_inflight_get_and_final_refresh_works",
    "test_threshold_exchange.py::test_fetch_concurrency_one_is_sequential",
    "test_threshold_exchange.py::test_stuck_get_round_hits_deadline_and_folds_what_arrived",
    "test_threshold_exchange.py::test_single_peer_stuck_get_is_bounded_on_the_calling_thread",
    "test_threshold_exchange.py::test_sequential_stuck_peer_is_skipped_others_still_fetched",
    "test_threshold_exchange.py::test_stuck_peer_is_not_resubmitted_and_pool_queue_stays_bounded",
    "test_threshold_exchange.py::test_task_queued_behind_stuck_workers_never_blocks_its_peer",
    "test_threshold_exchange.py::test_fetch_pool_cancel_only_unstarted_tasks",
    "test_threshold_exchange.py::test_exchange_thread_start_failure_stops_the_publisher",
    "test_threshold_exchange.py::test_fetch_pool_thread_start_failure_stops_started_workers",
    "test_threshold_exchange.py::test_exit_drain_stops_exchanges_and_waits_for_inflight_io",
    "test_threshold_exchange.py::test_exit_drain_is_bounded_by_its_timeout",
    # Sleep-widened races and overlap counts elsewhere.
    "test_run_identity.py::test_the_merge_window_reaches_run_merge_from_the_config",
    "test_run_identity.py::test_the_fanout_child_table_is_keyed_per_child_not_per_name",
    "test_read_path_opts.py::test_copy_range_releases_the_gil",
    "test_read_path_opts.py::test_ranged_download_honours_per_file_concurrency",
    "test_read_path_opts.py::test_ranged_download_drains_inflight_ranges_before_closing",
    "test_read_timing.py::test_the_split_is_reset_before_any_reader_reports",
    "test_native_dense.py::test_interpreter_exit_with_daemon_readers_mid_call_does_not_abort",
    "test_native_dense.py::test_drain_cancels_inflight_native_calls",
    "test_packed_token_grid.py::test_a_failing_scan_batch_waits_for_its_siblings",
)

def pytest_collection_modifyitems(config, items):
    for item in items:
        for needle, _why in _SLOW:
            if needle in item.nodeid:
                item.add_marker(pytest.mark.slow)
                break
        if any(needle in item.nodeid for needle in _TIMING):
            item.add_marker(pytest.mark.timing)


def _check_expected_src():
    """Mutation testing runs the suite against a mutated COPY of nova_bf. The
    repo's `pythonpath = ["src"]` puts the real one first unless the caller
    overrides it, and a mutant run that silently imported the real source
    would report "mutant survived" for every mutation. So the caller names
    the copy it means (`NOVA_BF_PARITY_EXPECT_SRC`), and a mismatch is a hard
    error, not a skip.

    Compared as PATHS, not string prefixes: `startswith("/x/src")` also
    accepted `/x/src_orig/nova_bf/__init__.py` -- the pristine copy a mutation
    driver keeps right next to the mutant."""
    expect = os.environ.get("NOVA_BF_PARITY_EXPECT_SRC")
    if not expect:
        return
    import nova_bf

    got = os.path.realpath(nova_bf.__file__)
    root = os.path.realpath(expect)
    if os.path.commonpath([got, root]) != root:
        raise pytest.UsageError(
            f"NOVA_BF_PARITY_EXPECT_SRC={expect} but nova_bf was imported from {got}")

_MERGE_ENV = (
    "NOVA_BF_MERGE_FORCE", "NOVA_BF_MERGE_WINDOW",
    # Same property as the merge overrides: both are meant to be EXPORTED to
    # get a run through a bad day (throttle the `tiebreak='id'` startup pass,
    # or put it back on whole-file ranged reads), so they linger in the shell
    # afterwards. An exported NOVA_BF_ID_PASS_RANGED_GET is the quiet one --
    # the ids come out identical either way, so every id-pass test would go on
    # reporting green while exercising the whole-file path the change exists
    # to avoid.
    "NOVA_BF_ID_PASS_RANGED_GET", "NOVA_BF_ID_PASS_WORKERS",
)


@pytest.fixture(autouse=True)
def _tf32_flag_is_restored():
    """Every test leaves torch's process-global TF32 matmul flag as it found it.

    Tests set `torch.backends.cuda.matmul.allow_tf32` directly and rely on this
    to restore it BY VALUE. `monkeypatch.setattr` cannot be used for it: on
    torch >= 2.15 the flag is a property of `cuBLASModule`, monkeypatch's undo
    deletes the attribute, that raises at teardown, and TF32 stays ON for every
    later test -- silently changing GEMM bits under the two-pass suites.
    """
    try:
        import torch
        matmul = torch.backends.cuda.matmul
        before = matmul.allow_tf32
    except Exception:  # noqa: BLE001 - no torch / no such flag: nothing to guard
        yield
        return
    yield
    matmul.allow_tf32 = before


@pytest.fixture(autouse=True)
def _merge_env_is_hermetic(monkeypatch):
    """No test inherits the operator's run-rescue overrides.

    `NOVA_BF_MERGE_FORCE` is meant to be EXPORTED for a rescue run, so it will
    be set in the shell of whoever is working on this code -- exactly when
    these tests get run. Exported, it turns six of them red for the wrong
    reason and, more quietly, makes every other merge test in the repo
    exercise the forced path while still reporting green. Tests that want it
    set it themselves.
    """
    for name in _MERGE_ENV:
        monkeypatch.delenv(name, raising=False)


# Remove fixed delays from tests that do not check them.
# Skip slow AWS metadata probes outside EC2 and disable threshold
# publish spacing. Tests that check these behaviors use explicit settings.
# Set NOVA_BF_TEST_REAL_TIMINGS=1 to restore both delays, or use
# "imds" or "spacing" to restore either one.
# On EC2, metadata probes remain enabled for instance credentials.
_IMDS_HOST = "169.254.169.254"
_REAL_TIMINGS = os.environ.get("NOVA_BF_TEST_REAL_TIMINGS", "")


def _on_ec2() -> bool:
    """True on an EC2 instance, from local DMI/hypervisor files only (no
    network): Nitro instances report "Amazon EC2" as the system vendor, older
    Xen ones an "ec2"-prefixed hypervisor uuid."""
    for path, test in (("/sys/class/dmi/id/sys_vendor", lambda v: "Amazon EC2" in v),
                       ("/sys/class/dmi/id/board_vendor", lambda v: "Amazon EC2" in v),
                       ("/sys/hypervisor/uuid", lambda v: v.lower().startswith("ec2"))):
        try:
            with open(path) as f:
                if test(f.read().strip()):
                    return True
        except OSError:
            pass
    return False


_ON_EC2 = _on_ec2()
# "1" = both real; "imds" / "spacing" = only that one real.
_REAL_IMDS = _REAL_TIMINGS in ("1", "imds") or _ON_EC2
_REAL_SPACING = _REAL_TIMINGS in ("1", "spacing")


@pytest.fixture(autouse=True)
def _no_imds_probe(monkeypatch):
    if _REAL_IMDS:
        yield
        return
    import urllib.error
    import urllib.request

    real = urllib.request.urlopen

    def urlopen(url, *a, **kw):
        full = getattr(url, "full_url", url)
        if isinstance(full, str) and _IMDS_HOST in full:
            raise urllib.error.URLError("IMDS probe disabled under pytest")
        return real(url, *a, **kw)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    # Disable slow AWS metadata credential checks outside EC2.
    # Existing credentials still work; missing credentials fail faster.
    # Preserve user settings and allow real timing tests to restore checks.
    if "AWS_EC2_METADATA_DISABLED" not in os.environ:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    yield


@pytest.fixture(autouse=True)
def _unspaced_exchange_publish(monkeypatch):
    if _REAL_SPACING:
        yield
        return
    try:
        from nova_bf import thresholds
    except Exception:  # noqa: BLE001 - optional deps missing: nothing to patch
        yield
        return
    cls = thresholds.ThresholdExchange
    real_init = cls.__init__

    def __init__(self, store, run_id, rank, num_jobs, sizes, *args, **kw):
        # Positional order after `sizes`: refresh_s, publish_min_interval_s, ...
        if len(args) < 2:
            kw.setdefault("publish_min_interval_s", 0.0)
        real_init(self, store, run_id, rank, num_jobs, sizes, *args, **kw)

    monkeypatch.setattr(cls, "__init__", __init__)
    yield
