"""Reader PROCESSES (ParquetSource with prefetch > 0): GIL
decoupling, shared-memory handoff and cleanup, pool sizing, and the
prefetch semantics (the cap itself is tested in
test_sources.py, test_prefetch_caps_reads_in_flight_processes; results in
every existing correctness test, which now run on process readers)."""

import ctypes
import functools
import gc
import json
import mmap
import os
import signal
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import make_dummy
import readhooks
from bench.systems import harness_cli as run
import faiss
from bench.systems import loop
from bench.record import schema
from bench.verify import oracle as verify
from bench.systems import cpu_reference
from bench.data import sources
from conftest import ROWS, arrow_default_io_threads
from bench.data.sources import ParquetSource, ReaderDied, shm_segments
from bench.timing import Stages

K = 10
LIBC = ctypes.PyDLL(None)  # PyDLL: the C call KEEPS the GIL, as cuVS 26.10's search does


class FakeEngine:
    """run_pass backend whose search takes ``dur`` seconds either holding the
    GIL for the whole call (``hold``: libc usleep through ctypes.PyDLL, a C
    call that never releases it, like ``cuvs.neighbors.brute_force.search``
    in cuvs 26.10, which has no ``with nogil:``) or releasing it
    (``time.sleep``, like FAISS's SWIG wrapper). A pure-Python busy loop
    would be a weaker model: the eval loop hands the GIL over every
    ``sys.getswitchinterval()`` (5 ms) when another thread asks."""

    name, engine, events = "cpu", "fake", True
    last_wait, last_build = (0.0, False), 0.0

    def __init__(self, hold, dur=0.05, fail_at=None, exc=RuntimeError):
        self.hold, self.dur, self.stats = hold, dur, Stages()
        self.fail_at, self.exc, self.calls = fail_at, exc, 0

    def alloc_peak_reset(self):
        pass

    def alloc_peak(self):
        return ""

    def search(self, xb):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise self.exc("engine died")
        if self.hold:
            LIBC.usleep(int(self.dur * 1e6))
        else:
            time.sleep(self.dur)
        return np.zeros((4, K), np.float32), np.zeros((4, K), np.int64)


@pytest.fixture(scope="module")
def gil_corpus(tmp_path_factory):
    """8 files x 40k x 64-D fp32 (~10 MB each): a read is ~15-20 ms, a fake
    engine call 50 ms, so 2 readers keep up whenever they can run."""
    out = tmp_path_factory.mktemp("gil")
    make_dummy.make_corpus(str(out), 8, 40000, 64, seed=1)
    return str(out)


def _stall_after_first(info):
    """Consumer stall summed over files 2.. (file 1 is a cold read for any
    design; the steady state is what GIL coupling inflates)."""
    return sum(f[2] for f in info["trace"][1][1:])


@pytest.mark.slow
def test_gil_holding_engine_does_not_stall_readers(gil_corpus):
    """Prefetch 2, small part size (more Python glue per read). With reader
    PROCESSES a GIL-holding engine must stall the loop no more than a
    GIL-releasing one of the same duration, and must not inflate the readers'
    own decode times (dev box, 2026-10-14, stall over files 2-8: in-process
    thread readers, the removed design, ~0.001 s releasing vs ~0.07 s
    holding; processes ~0.008 s both ways, decode ~0.14 s both ways)."""
    reader = "process"
    src = ParquetSource(gil_corpus, prefetch=2, workers=2, range_bytes=1 << 18).start()
    try:
        loop.run_pass(src, FakeEngine(False), 4, K)  # warmup (page cache, first call)
        got = {False: [], True: []}
        for _ in range(3):
            for hold in (False, True):  # interleaved
                info = loop.run_pass(src, FakeEngine(hold), 4, K)[2]
                got[hold].append((_stall_after_first(info), info["decode_s"]))
    finally:
        src.close()
    stall = {h: statistics.median(s for s, _ in v) for h, v in got.items()}
    decode = {h: statistics.median(d for _, d in v) for h, v in got.items()}
    msg = f"{reader}: stall release {stall[False]:.4f} hold {stall[True]:.4f}; " \
          f"decode release {decode[False]:.3f} hold {decode[True]:.3f}"
    # relative + absolute slack for a loaded box: under 24-48 busy loops
    # the process design read stall release ~0.003-0.015 vs hold
    # ~0.017-0.058 (scheduling noise, not the GIL; 3 of 20 runs failed the
    # old 1.5x + 0.01). Still below the coupling thread readers showed at
    # rest (hold ~0.07 vs release ~0.001: 2 x 0.001 + 0.04 = 0.042)
    assert stall[True] <= 2 * stall[False] + 0.04, msg
    assert decode[True] <= 1.5 * decode[False] + 0.05, msg


# ------------------------------------------------------------- shm handoff

def _live(src):
    return shm_segments(src.shm_prefix)


def _all_free(src):
    """Every ring slot is back on the free list (nothing in flight, nothing
    held by the consumer)."""
    gc.collect()  # a traceback may still pin a block
    return len(src.ring.free) == len(src.ring)


def test_blocks_are_views_of_a_persistent_ring(corpus):
    """Blocks are numpy views of the ring's shared mappings (no copy); the
    ring is prefetch + 1 slots of the largest block, sized from the footers,
    created once and reused for every file of every pass: run_pass drops each
    block before asking for the next, so the ring never grows. close()
    unlinks it."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=2, workers=2).start()
    try:
        prefix = src.shm_prefix
        assert prefix.startswith(sources.SHM_PREFIX + str(os.getpid()))
        ring = src.ring
        assert len(ring) == 3 and ring.slot_bytes == max(ROWS) * 32 * 4
        names = [ring.name(i) for i in range(3)]
        assert _live(src) == sorted(names)
        it = iter(src)
        got = []
        for b in it:
            base = b.vectors.base
            assert isinstance(base, mmap.mmap) and base in ring.mms  # a mapping, not a copy
            assert b.vectors.flags.c_contiguous and b.vectors.dtype == np.float32
            assert b.info["handoff"] >= 0 and b.info["publish"] >= 0
            assert b.info["publish_end"] >= b.info["w_end"]
            got.append(b.vectors.copy())
            del b, base
        np.testing.assert_array_equal(np.concatenate(got), xb)
        st = src.stats.as_dict()
        assert st["handoff"] > 0 and st["publish"] > 0
        assert st["worker_cpu_user"] + st["worker_cpu_sys"] > 0
        for _ in range(2):
            loop.run_pass(src, FakeEngine(False, dur=0.0), 4, K)
        assert ring.grown == 0 and len(ring) == 3 and _live(src) == sorted(names)
        assert _all_free(src)
    finally:
        src.close()
    assert shm_segments(prefix) == []


def test_held_block_is_never_overwritten(corpus):
    """A slot is reused only once the consumer dropped every reference to the
    array over it: a consumer that keeps a block (or a view of it) while the
    stream goes on sees its content unchanged, the ring grows instead."""
    path, xb = corpus
    with ParquetSource(path, prefetch=1, workers=1).start() as src:
        ring = src.ring
        assert len(ring) == 2
        it = iter(src)
        first = next(it).vectors
        view = first[:5]
        want = first.copy()
        rest = []
        for b in it:  # five more files through the ring
            rest.append(b.vectors.copy())
            del b
            np.testing.assert_array_equal(first, want)
        np.testing.assert_array_equal(np.concatenate([want] + rest), xb)
        assert ring.grown == 1 and len(ring) == 3
        slot = ring.mms.index(first.base)
        assert slot not in ring.free
        del first
        assert slot not in ring.free  # the view still holds it
        del view
        assert slot in ring.free


def test_reader_decodes_multi_row_group_file_straight_into_slot(tmp_path):
    """A reader process decodes a multi-row-group file straight into the ring
    slot (one copy per block): no ``combine_chunks`` and no separate publish
    copy, and the slot holds exactly the file's vectors."""
    mode = "native"
    out = tmp_path / "c"
    paths = make_dummy.make_corpus(str(out), 3, 700, 16, row_group_size=128)  # 6 rgs/file
    assert pq.ParquetFile(paths[0]).metadata.num_row_groups == 6
    xb = np.array([v for p in paths
                   for v in pq.read_table(p).column("embedding").to_pylist()], np.float32)
    with ParquetSource(str(out), prefetch=2, workers=2, read_mode=mode,
                       read_fn=readhooks.combine_spy).start() as src:
        got, infos = [], []
        for b in src:
            assert b.vectors.base in src.ring.mms
            got.append(b.vectors.copy())
            infos.append(b.info)
            del b
    np.testing.assert_array_equal(np.concatenate(got), xb)
    assert [i["combines"] for i in infos] == [0, 0, 0]
    assert all(i["in_out"] for i in infos)


def test_reader_host_widen_publishes_widened_block(tmp_path):
    out = tmp_path / "c"
    make_dummy.make_corpus(str(out), 2, 300, 16, value_dtype=np.float16, row_group_size=100)
    ref = np.concatenate([b.vectors.astype(np.float32)
                          for b in ParquetSource(str(out), prefetch=0)])
    with ParquetSource(str(out), prefetch=1, dtype=np.float32,
                       read_fn=readhooks.combine_spy).start() as src:
        got, infos = [], []
        for b in src:
            assert b.vectors.dtype == np.float32
            got.append(b.vectors.copy())
            infos.append(b.info)
            del b
    np.testing.assert_array_equal(np.concatenate(got), ref)
    assert not any(i["in_out"] for i in infos)


def test_ring_must_fit_in_dev_shm(corpus, monkeypatch):
    """/dev/shm is checked before the ring is made: a clear error, nothing
    left behind."""
    monkeypatch.setattr(sources, "shm_free_bytes", lambda: 1000)
    src = ParquetSource(corpus[0], prefetch=2, workers=2)
    with pytest.raises(OSError, match="block ring needs 3 x"):
        src.start()
    prefix = src.shm_prefix
    src.close()
    assert shm_segments(prefix) == []


def test_ring_slot_size_follows_stored_or_host_dtype(tmp_path):
    out = tmp_path / "c"
    make_dummy.make_corpus(str(out), 2, 300, 16, value_dtype=np.float16)
    make_dummy.write_vectors(str(out / "part-00002.parquet"),
                             make_dummy.unit_vectors(np.random.default_rng(0), 500, 16),
                             value_dtype=np.float16)
    assert ParquetSource(str(out), prefetch=1).max_block_bytes() == 500 * 16 * 2
    src = ParquetSource(str(out), prefetch=1, dtype=np.float32)
    assert src.max_block_bytes() == 500 * 16 * 4
    blocks = [b.vectors.copy() for b in src]
    assert blocks[2].shape == (500, 16) and blocks[2].dtype == np.float32
    src.close()


def test_blocks_outlive_the_source(corpus):
    """A consumer that keeps blocks (list(src)) keeps valid arrays: the
    mapping lives as long as the array, not as long as the segment's name."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=3, workers=2)
    blocks = list(src)
    src.close()
    np.testing.assert_array_equal(np.concatenate([b.vectors for b in blocks]), xb)


def test_no_leak_after_consumer_exception_and_pool_recovers(corpus, queries):
    path, xb = corpus
    src = ParquetSource(path, prefetch=3, workers=2,
                        read_fn=functools.partial(readhooks.slow_read, default=0.05)).start()
    try:
        with pytest.raises(RuntimeError, match="engine died"):
            loop.run_pass(src, FakeEngine(False, dur=0.0, fail_at=2), 4, K)
        assert src._procs.busy == {} and _all_free(src)
        # start() respawns the terminated readers (untimed); the pass is right
        src.start()
        b = cpu_reference.CpuBackend(K)
        b.setup(queries)
        D, I, _ = loop.run_pass(src, b, len(queries), K)
        D_ref, I_ref = faiss.knn(queries, xb, K + 1, metric=schema.IP)
        assert verify.compare_results(D, I, D_ref, I_ref) == []
        assert _all_free(src) and src.ring.grown == 0
    finally:
        src.close()


def test_no_leak_after_keyboard_interrupt(corpus):
    path, _ = corpus
    src = ParquetSource(path, prefetch=2, workers=2,
                        read_fn=functools.partial(readhooks.slow_read, default=0.05))
    try:
        with pytest.raises(KeyboardInterrupt):
            loop.run_pass(src, FakeEngine(False, dur=0.0, fail_at=1, exc=KeyboardInterrupt),
                         4, K)
        assert _all_free(src)
    finally:
        src.close()


def test_no_leak_after_worker_exception(corpus):
    path, _ = corpus
    third = os.path.basename(sources.list_parquet(path)[1][2].path)
    src = ParquetSource(path, prefetch=3, workers=3,
                        read_fn=functools.partial(readhooks.failing_read, fail=third))
    try:
        with pytest.raises(OSError, match="s3 went away"):
            loop.run_pass(src, FakeEngine(False, dur=0.01), 4, K)
        assert _all_free(src)
    finally:
        src.close()


@pytest.mark.parametrize("hook", [readhooks.crashing_read, readhooks.publish_then_crash])
def test_worker_crash_surfaces_promptly_without_leak(corpus, hook):
    """A reader that dies (os._exit: no cleanup, like the OOM killer) raises
    ReaderDied naming the file, at once, even while the consumer waits on
    another file; the slot it was writing (or wrote, before dying without a
    reply) goes back to the ring, and close() leaves nothing in /dev/shm."""
    path, _ = corpus
    second = os.path.basename(sources.list_parquet(path)[1][1].path)
    src = ParquetSource(path, prefetch=2, workers=2,
                        read_fn=functools.partial(hook, crash=second)).start()
    try:
        t0 = time.perf_counter()
        with pytest.raises(ReaderDied, match=second):
            loop.run_pass(src, FakeEngine(False, dur=0.01), 4, K)
        assert time.perf_counter() - t0 < 5
        assert _all_free(src)
        prefix = src.shm_prefix
    finally:
        src.close()
    assert shm_segments(prefix) == []


def test_close_does_not_wait_for_inflight_reads(corpus):
    """Closing early (break / GeneratorExit) terminates readers with a file
    in flight instead of waiting for (possibly minutes of) S3 reads, and
    gives their slots back."""
    path, _ = corpus
    files = sources.list_parquet(path)[1]
    delays = {os.path.basename(r.path): 2.0 for r in files[1:]}
    src = ParquetSource(path, prefetch=2, workers=2,
                        read_fn=functools.partial(readhooks.slow_read, delays=delays)).start()
    try:
        it = iter(src)
        next(it)  # first block; the next read is running (2 s)
        time.sleep(0.05)
        t0 = time.perf_counter()
        it.close()
        assert time.perf_counter() - t0 < 1.0
        assert _all_free(src) and src._procs.busy == {}
    finally:
        src.close()


def test_close_stops_reader_processes(corpus):
    path, _ = corpus
    src = ParquetSource(path, prefetch=2, workers=2).start()
    procs = [p for p, _ in src._procs.procs]
    assert len(procs) == 2 and all(p.is_alive() for p in procs)
    list(src)
    list(src)  # kept across passes: the same processes
    assert [p for p, _ in src._procs.procs] == procs
    src.close()
    assert not any(p.is_alive() for p in procs)


# ------------------------------------------------------- pools and settings

@pytest.mark.parametrize("usable,readers", [(1, 1), (2, 1), (3, 2), (8, 3), (24, 5),
                                            (23, 8), (4, 8), (12, 12)])
@pytest.mark.parametrize("cpu,io", [(0, 0), (7, 5), (13, 3), (1, 20)])
def test_reader_pools_total_what_one_shared_pool_gets(monkeypatch, usable, readers, cpu, io):
    """Process readers TOGETHER get the decode and IO threads the one shared
    pool of thread readers gets (configure_arrow_pools with the same
    arguments): no CPU held back in one mode only, the remainder spread
    (first readers +1) rather than dropped. Only a total below the reader
    count gives more (1 each)."""
    monkeypatch.setattr(sources, "usable_cpu_count", lambda: usable)
    shared_cpu, shared_io = sources.configure_arrow_pools(cpu, io)  # thread readers' pool
    per_cpu, per_io = sources.reader_pool_threads(cpu, io, readers)
    assert len(per_cpu) == len(per_io) == readers
    assert sum(per_cpu) == max(shared_cpu, readers)
    assert per_cpu == sorted(per_cpu, reverse=True) and per_cpu[0] - per_cpu[-1] <= 1
    if io:
        assert sum(per_io) == max(shared_io, readers) and per_io[0] - per_io[-1] <= 1
    else:  # each fresh reader takes its share of Arrow's default, by index
        assert per_io == [0] * readers
        default = arrow_default_io_threads()
        assert shared_io == default
        assert sum(sources.split_threads(default, readers)) == max(default, readers)


def test_pool_split_str():
    assert sources.pool_split_str([3, 2, 2]) == "1x3+2x2"
    assert sources.pool_split_str([4, 4]) == "2x4"
    assert sources.pool_split_str([5]) == "1x5"
    assert sources.pool_split_str([None, 2]) is None and sources.pool_split_str(None) is None


def test_reader_pool_threads_warns_below_one_each(corpus):
    with pytest.warns(UserWarning, match="each gets 1 decode thread"):
        ParquetSource(corpus[0], prefetch=3, workers=3, pa_cpu_threads=2)


def test_readers_apply_pool_sizes_and_range_settings(corpus):
    """Each reader process sizes its own Arrow pools (the totals split across
    readers, remainder to the first) and reads with the parent's read mode
    and per-file range concurrency."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=3, workers=3, read_mode="arrow", range_bytes=777,
                        range_concurrency=24, pa_cpu_threads=7, pa_io_threads=5,
                        read_fn=readhooks.slow_read).start()
    try:
        assert src.reader_pa_threads_effective == ([3, 2, 2], [2, 2, 1])
        pools = {p.pid: src._procs.pa_threads[r] for r, (p, _) in enumerate(src._procs.procs)}
        blocks = list(src)
        np.testing.assert_array_equal(np.concatenate([b.vectors for b in blocks]), xb)
        for b in blocks:
            a = b.info["args"]
            assert (a["mode"], a["range_concurrency"], a["range_bytes"]) == ("arrow", 8, 777)
            assert (a["pa_cpu"], a["pa_io"]) == pools[a["pid"]]
            assert a["pid"] != os.getpid()
    finally:
        src.close()


def test_default_io_pool_is_split_across_readers(corpus):
    """--io-thread-count 0 splits Arrow's default IO pool across the reader
    processes (8 over 3: 3 + 3 + 2), so the readers together issue the GETs
    one shared pool would, not readers x 8 and not 3 x 2."""
    want = sources.split_threads(arrow_default_io_threads(), 3)
    src = ParquetSource(corpus[0], prefetch=3, workers=3, pa_io_threads=0,
                        read_fn=readhooks.slow_read).start()
    try:
        assert src.reader_pa_threads_effective[1] == want
        assert sum(want) == arrow_default_io_threads()
        assert {b.info["args"]["pa_io"] for b in src} <= set(want)
    finally:
        src.close()


def test_readers_warm_up_their_filesystem_before_ready(corpus, tmp_path):
    """Each reader touches the corpus through its OWN filesystem object (on
    S3: credentials, client, first TLS connection) before it says ready, so
    that cost is not in its first timed read: by the time start() returns,
    every reader process has opened a corpus file, no pass having run."""
    path, xb = corpus
    log = tmp_path / "fs.log"
    files = sources.list_parquet(path)[1]
    fs = pafs.PyFileSystem(readhooks.LoggedLocalFS(str(log)))
    # arrow: the native reader takes local / s3:// filesystems only, not this logging one
    with ParquetSource(path, prefetch=2, workers=2, fs=fs, files=files,
                       read_mode="arrow").start() as src:
        pids = {p.pid for p, _ in src._procs.procs}
        touched = {int(line.split()[0]) for line in log.read_text().splitlines()
                   if line.split()[1] == "open"}
        assert pids <= touched
        np.testing.assert_array_equal(np.concatenate([b.vectors.copy() for b in src]), xb)


def test_readers_warm_up_before_ready(corpus):
    """A reader's one-off costs (pyarrow.dataset's lazy import, ~0.1 s, and the
    first parquet read) are paid at start-up, untimed, not in its first timed
    read: every read, the first of each reader included, finds pyarrow.dataset
    already imported."""
    with ParquetSource(corpus[0], prefetch=2, workers=2, read_mode="arrow",
                       read_fn=readhooks.warm_probe).start() as src:
        infos = [b.info for b in src]
    assert len({i["pid"] for i in infos}) == 2
    assert all(i["dataset_imported"] for i in infos)


def test_readers_and_ring_capped_by_file_count(corpus):
    """No more reader processes, and no more ring slots, than files: prefetch
    8 / workers 8 over the 6-file corpus is 6 readers (so the range-GET budget
    is split 6 ways, not 8) and a 7-slot ring, not 9."""
    path, xb = corpus
    with ParquetSource(path, prefetch=8, workers=8).start() as src:
        assert src.readers == len(ROWS) and \
            src.range_concurrency == sources.NATIVE_CONCURRENCY // len(ROWS)
        assert len(src._procs.procs) == len(ROWS) and len(src.ring) == len(ROWS) + 1
        got = []
        for b in src:
            got.append(b.vectors.copy())
            del b
        np.testing.assert_array_equal(np.concatenate(got), xb)
        assert src.ring.grown == 0


def test_abort_is_reentrant(corpus):
    """An abort cut short (Ctrl-C after a reader was reaped and nulled, before
    the rest) is finished by the next abort / close: no TypeError on the
    reaped entry, every reader stopped, the slots back, the pool usable."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=2, workers=2,
                        read_fn=functools.partial(readhooks.slow_read, default=0.3)).start()
    try:
        procs = src._procs
        it = iter(src)
        first = next(it)  # files 1 and 2 are now in flight
        del first
        # an abort interrupted right after reaping reader 0 and nulling it
        p0, c0 = procs.procs[0]
        p0.terminate()
        p0.join(5)
        c0.close()
        procs.procs[0] = None
        others = [e[0] for e in procs.procs if e is not None]
        procs.abort()  # finishes the job
        procs.abort()  # and again: nothing left to do
        assert procs.procs == [None, None] and procs.busy == {}
        assert not any(p.is_alive() for p in others)
        it.close()  # the generator's own abort, a third one, then the slots
        assert _all_free(src)
        prefix = src.shm_prefix
        procs.read_fn = None  # respawned readers read at full speed
        src.start()
        np.testing.assert_array_equal(np.concatenate([b.vectors.copy() for b in src]), xb)
    finally:
        src.close()
    assert shm_segments(prefix) == []


def test_arrow_mode_footers_cached_in_parent(corpus):
    """arrow mode reads each file's footer once (first pass) for its byte /
    row-group counts; with readers in other processes the parent keeps the
    cache and ships the footer with every later task."""
    path, _ = corpus
    with ParquetSource(path, prefetch=2, workers=2, read_mode="arrow") as src:
        list(src)
        assert set(src.footers) == {r.path for r in src.files}
        first = src.stats.as_dict()
        list(src)
        again = src.stats.as_dict()
        assert again["bytes"] == first["bytes"] and again["row_groups"] == len(ROWS)


def test_prefetch0_reads_inline(corpus):
    path, _ = corpus
    src = ParquetSource(path, prefetch=0)
    assert src.reader == "inline"
    src.start()  # nothing to start
    assert src._procs is None
    blocks = list(src)
    assert all(b.info.get("handoff") is None for b in blocks)


def test_prefetch_above_the_cap_refused(corpus):
    """--prefetch above MAX_PREFETCH would not fit the task pipe (a blocking
    submit write while the readers block on unread replies: a deadlock, seen
    at 6000 with tiny files); the harness and ParquetSource both refuse it."""
    cap = sources.MAX_PREFETCH
    assert cap * sources._TASK.size <= 65536 // 4  # a quarter of the default pipe
    base = ["--regime", "stream", "--corpus", "x", "--csv", "y", "--prefetch"]
    assert run.parse_args(base + [str(cap)]).prefetch == cap
    with pytest.raises(SystemExit):
        run.parse_args(base + [str(cap + 1)])
    with pytest.raises(ValueError, match="prefetch must be in"):
        ParquetSource(corpus[0], prefetch=cap + 1)


@pytest.mark.skipif(sources._F_SETPIPE_SZ is None, reason="Linux pipe sizing only")
def test_task_pipe_is_sized_for_every_task_in_flight():
    """A pipe smaller than the tasks in flight (a user past
    pipe-user-pages-soft gets one-page pipes) is grown to hold them all."""
    import fcntl
    r, w = os.pipe()
    try:
        fcntl.fcntl(w, sources._F_SETPIPE_SZ, 4096)
        assert fcntl.fcntl(w, sources._F_GETPIPE_SZ) == 4096
        need = sources.MAX_PREFETCH * sources._TASK.size
        assert sources.ensure_pipe_room(w, need) >= need
        assert fcntl.fcntl(w, sources._F_GETPIPE_SZ) >= need
        for _ in range(sources.MAX_PREFETCH):  # never blocks
            os.set_blocking(w, False)
            os.write(w, sources._TASK.pack(1, 1))
    finally:
        os.close(r)
        os.close(w)


def test_fd_room_checked_before_python_3_13(monkeypatch):
    """Before 3.13 each ring slot's mapping pins an fd per process: a ring
    RLIMIT_NOFILE cannot hold is refused up front with a clear message."""
    monkeypatch.setattr(sources, "_MMAP_KW", {})
    monkeypatch.setattr(sources.resource, "getrlimit", lambda what: (300, 4096))
    sources._check_fd_room(300 - sources.FD_MARGIN)
    with pytest.raises(OSError, match="ulimit -n"):
        sources._check_fd_room(301 - sources.FD_MARGIN)
    monkeypatch.setattr(sources, "_MMAP_KW", {"trackfd": False})
    sources._check_fd_room(10_000)  # 3.13+: the ring keeps no fds


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    """600 one-row files: a deep prefetch with nothing to read."""
    out = tmp_path_factory.mktemp("tiny")
    x = make_dummy.unit_vectors(np.random.default_rng(5), 600, 4)
    for i in range(600):
        make_dummy.write_vectors(str(out / f"f{i:04d}.parquet"), x[i:i + 1])
    return str(out), x


def test_deep_prefetch_completes_and_ring_keeps_no_fds(tiny_corpus):
    """--prefetch 600 over 600 tiny files: 600 records submitted before the
    first reply is read, a 601-slot ring. Completes in order, and (Python >=
    3.13, mmap trackfd=False) the ring's mappings pin no fd in the parent
    (before, one dup per slot: 601 fds per process)."""
    path, x = tiny_corpus
    fds0 = len(os.listdir("/proc/self/fd"))
    with ParquetSource(path, prefetch=600, workers=2).start() as src:
        assert len(src.ring) == 601
        fds = len(os.listdir("/proc/self/fd")) - fds0
        got = np.concatenate([b.vectors.copy() for b in src])
        np.testing.assert_array_equal(got, x)
    if sys.version_info >= (3, 13):
        assert fds < 50, fds  # pipes + sentinels, no per-slot fd


# ------------------------------------------- dispatch ahead, dead readers

@pytest.mark.slow
@pytest.mark.parametrize("workers,slow,engine", [(1, 0.3, 0.25), (2, 0.6, 0.3)])
def test_prefetch_beyond_readers_keeps_readers_busy(gil_corpus, workers, slow, engine):
    """With prefetch > readers the tasks wait IN the readers (queued on the
    shared task pipe at submit time), so a reader that finishes during the
    engine call starts its next file without the parent running any Python.
    Before, tasks waiting for a free reader were handed out only inside the
    next next(), and the readers idled through every engine call.

    8 files, slow (``slow`` s) and fast (0.02 s) reads alternating, prefetch
    4, an engine call of ``engine`` s per file (the engine total is at least
    the read total per reader). Dispatched ahead, the consumer waits for about
    the first slow read only (dev box: workers 1 0.32 s, workers 2 0.62 s).
    With the old hand-out-in-next() dispatch, workers 1 waits for every read
    in series (1.41 s), which the bound, 1.1 s, catches with room for a
    loaded box (absolute, so the old thread-reader baseline is not needed).
    Workers 2 under the old dispatch measured 0.94 s, too close to catch with
    a robust bound: that case checks only that two readers keep up."""
    files = sources.list_parquet(gil_corpus)[1]
    delays = {os.path.basename(r.path): (0.02 if i % 2 else slow)
              for i, r in enumerate(files)}
    read_fn = functools.partial(readhooks.slow_read, delays=delays)
    src = ParquetSource(gil_corpus, prefetch=4, workers=workers, read_fn=read_fn).start()
    try:
        loop.run_pass(src, FakeEngine(False, dur=engine), 4, K)  # warmup
        stall = min(loop.run_pass(src, FakeEngine(False, dur=engine), 4, K)[2]
                    ["source_stall_s"] for _ in range(2))
    finally:
        src.close()
    assert stall >= 0.9 * slow, stall   # the first slow read is waited for either way
    assert stall < 1.1, stall


@pytest.fixture(scope="module")
def outlier_corpus(tmp_path_factory):
    """12 small files (the reads are slowed by a hook)."""
    out = tmp_path_factory.mktemp("outlier")
    make_dummy.make_corpus(str(out), 12, 2000, 16, seed=3)
    return str(out)


@pytest.mark.slow
def test_slow_outlier_file_does_not_hold_up_others(outlier_corpus):
    """Work-conserving dispatch: with prefetch > readers a file must never wait
    behind a slow file while another reader idles. An earlier design pinned
    each task to one reader at submit time, so the files queued behind the
    outlier waited for it while the other reader sat idle. Readers now pull
    from one shared task pipe, so the first free reader takes the next file.

    12 files, file 1 a 2.5 s read, the others 0.2 s, prefetch 11, 2 readers, a
    0.01 s engine call. Work-conserving: one reader does the outlier while
    the other does the 11 others (2.2 s), so the consumer waits for the
    outlier only (dev box: 2.50 s). Pinned to readers in turn (file i to
    reader i % 2), files 3, 5, 7, 9 and 11 queue behind the outlier (3.42 s).
    The bound, 3.0 s, sits half-way (absolute, so the old thread-reader
    baseline is not needed)."""
    files = sources.list_parquet(outlier_corpus)[1]
    delays = {os.path.basename(r.path): (2.5 if i == 1 else 0.2)
              for i, r in enumerate(files)}
    read_fn = functools.partial(readhooks.slow_read, delays=delays)
    src = ParquetSource(outlier_corpus, prefetch=11, workers=2, read_fn=read_fn).start()
    try:
        stall = min(loop.run_pass(src, FakeEngine(False, dur=0.01), 4, K)[2]
                    ["source_stall_s"] for _ in range(2))
    finally:
        src.close()
    assert stall > 2.4, stall  # the outlier is waited for either way
    assert stall < 3.0, stall




def test_reader_dead_between_passes_is_not_respawned_in_a_pass(corpus):
    """A reader that died while idle between passes is NOT silently respawned
    inside the next (timed) pass: the pass raises ReaderDied naming it, hands
    out nothing, and an explicit start() (untimed) respawns it."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=2, workers=2).start()
    try:
        procs = src._procs
        list(src)
        p, _ = procs.procs[1]
        p.kill()
        p.join(5)
        with pytest.raises(ReaderDied, match=rf"{p.pid} \(reader 1\).*between passes"):
            next(iter(src))
        assert procs.procs[1][0] is p and procs.outstanding == {} and _all_free(src)
        src.start()
        assert procs.procs[1][0] is not p and procs.procs[1][0].is_alive()
        np.testing.assert_array_equal(np.concatenate([b.vectors.copy() for b in src]), xb)
    finally:
        src.close()


@pytest.mark.slow
def test_pass_after_abort_needs_start(corpus):
    """A pass aborted with reads in flight stops every reader; the next pass
    does not respawn them inside its timed span but raises ReaderDied asking
    for start(); after start() it runs."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=2, workers=2,
                        read_fn=functools.partial(readhooks.slow_read, default=0.1)).start()
    try:
        it = iter(src)
        next(it)
        it.close()  # file 2 in flight: abort
        assert src._procs.procs == [None, None] and _all_free(src)
        with pytest.raises(ReaderDied, match=r"reader 0 is not running.*start\(\)"):
            next(iter(src))
        assert _all_free(src)
        src.start()
        np.testing.assert_array_equal(np.concatenate([b.vectors.copy() for b in src]), xb)
    finally:
        src.close()


@pytest.mark.slow
def test_slots_return_when_abort_is_interrupted(corpus):
    """An abort cut short (Ctrl-C inside it) still gives the unconsumed
    tasks' slots back; the readers it left running are killed by the next
    start() before any slot is handed out again, and the pool works."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=3, workers=2,
                        read_fn=functools.partial(readhooks.slow_read, default=0.3)).start()
    try:
        procs = src._procs
        it = iter(src)
        first = next(it)  # files 2 and 3 now in flight
        del first
        old = [p for p, _ in procs.procs]

        def interrupted():
            procs._aborting = True  # what a real abort does first
            raise KeyboardInterrupt

        procs.abort = interrupted
        with pytest.raises(KeyboardInterrupt):
            it.close()
        del procs.abort
        assert _all_free(src)
        with pytest.raises(ReaderDied, match="start\\(\\)"):
            next(iter(src))  # nothing handed out while old readers may still write
        src.start()
        assert not any(p.is_alive() for p in old)
        procs.read_fn = None
        np.testing.assert_array_equal(np.concatenate([b.vectors.copy() for b in src]), xb)
        assert _all_free(src)
    finally:
        src.close()


def test_reader_dead_while_idle_is_reader_died(corpus):
    """A reader that died while it had nothing to do is found when it is
    handed a file: ReaderDied naming the reader and that file, not a bare
    BrokenPipeError."""
    path, _ = corpus
    src = ParquetSource(path, prefetch=2, workers=2).start()
    try:
        procs = src._procs
        p, _ = procs.procs[0]
        p.kill()
        p.join(5)
        ref = src.files[0]
        with pytest.raises(ReaderDied, match=rf"reader 0\).*{os.path.basename(ref.path)}"):
            procs.submit(0, ref.path, 0)
        assert procs.busy == {}
    finally:
        src.close()


# --------------------------------------------- signals, main re-import

def _wait_for(cond, timeout=30.0):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_run_cleans_up_on_sigterm(gil_corpus, tmp_path, sig):
    """The harness turns SIGTERM / SIGHUP into SystemExit, so its cleanup runs:
    the readers stop and the ring leaves /dev/shm (a plain SIGTERM used to
    kill it with segments behind)."""
    cmd = [sys.executable, os.path.join(os.path.dirname(run.__file__), "harness_cli.py"),
           "--regime", "stream", "--backend", "cpu", "--corpus", gil_corpus,
           "--nq", "4", "--prefetch", "2", "--workers", "2", "--warmup", "0",
           "--repeats", "100000", "--csv", str(tmp_path / "r.csv")]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env=dict(os.environ, OMP_NUM_THREADS="2"))
    prefix = f"{sources.SHM_PREFIX}{proc.pid}-"
    try:
        assert _wait_for(lambda: shm_segments(prefix)), "no ring appeared"
        time.sleep(0.5)  # in the passes
        proc.send_signal(sig)
        rc = proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    err = proc.stderr.read().decode()
    assert rc == 128 + sig, err
    assert shm_segments(prefix) == []


@pytest.mark.slow
def test_nohup_run_survives_sighup(gil_corpus, tmp_path):
    """Started with SIGHUP ignored (as under nohup), the run and its reader
    processes keep it ignored: a hangup to the whole process group (a closed
    terminal) leaves the run going, readers and ring included. It used to
    turn into SystemExit(129) in the run, the readers dying by SIG_DFL."""
    cmd = [sys.executable, os.path.join(os.path.dirname(run.__file__), "harness_cli.py"),
           "--regime", "stream", "--backend", "cpu", "--corpus", gil_corpus,
           "--nq", "4", "--prefetch", "2", "--workers", "2", "--warmup", "0",
           "--repeats", "100000", "--csv", str(tmp_path / "r.csv")]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env=dict(os.environ, OMP_NUM_THREADS="2"),
                            start_new_session=True,  # its own group, killpg-able
                            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    prefix = f"{sources.SHM_PREFIX}{proc.pid}-"
    try:
        assert _wait_for(lambda: shm_segments(prefix)), "no ring appeared"
        time.sleep(0.5)  # in the passes
        os.killpg(proc.pid, signal.SIGHUP)  # the run and its readers
        time.sleep(1.5)
        assert proc.poll() is None, proc.stderr.read().decode()
        rows = len(open(tmp_path / "r.csv").readlines())
        assert _wait_for(lambda: len(open(tmp_path / "r.csv").readlines()) > rows, 10)
        proc.send_signal(signal.SIGTERM)  # still ends cleanly
        rc = proc.wait(30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    err = proc.stderr.read().decode()
    assert rc == 128 + signal.SIGTERM, err
    assert shm_segments(prefix) == []


SPAWN_SCRIPT = """
import json, os, signal, sys, time
sys.path[:0] = [{harness!r}, {tests!r}]
import faiss  # what the harness imports at the top: the readers must not
import readhooks
from bench.data import sources
if __name__ == "__main__":
    src = sources.ParquetSource({corpus!r}, prefetch=2, workers=2,
                                read_fn=readhooks.module_probe).start()
    if sys.argv[1] == "mods":
        print(json.dumps([b.info["mods"] for b in src]))
        src.close()
    else:  # hang with the ring up, until killed
        print(src.shm_prefix, flush=True)
        time.sleep(60)
"""


def _spawn_script(tmp_path, corpus_dir):
    p = tmp_path / "script.py"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tests/
    p.write_text(SPAWN_SCRIPT.format(harness=os.path.dirname(here), tests=here,
                                     corpus=corpus_dir))
    return str(p)


def test_readers_do_not_reimport_main(corpus, tmp_path):
    """Readers must not re-run the parent's __main__ (on the node that imports the GPU faiss
    build): they import only what they need."""
    out = subprocess.run([sys.executable, _spawn_script(tmp_path, corpus[0]), "mods"],
                         capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, OMP_NUM_THREADS="2"))
    assert out.returncode == 0, out.stderr
    mods = json.loads(out.stdout.strip().splitlines()[-1])
    assert len(mods) == len(ROWS) and all(m == [] for m in mods), mods


def test_readers_remove_the_ring_when_the_parent_is_killed(corpus, tmp_path):
    """A SIGKILLed parent runs no cleanup at all; its readers see EOF on their
    pipe and remove the pool's segments themselves."""
    proc = subprocess.Popen([sys.executable, _spawn_script(tmp_path, corpus[0]), "hang"],
                            stdout=subprocess.PIPE, text=True,
                            env=dict(os.environ, OMP_NUM_THREADS="2"))
    try:
        prefix = proc.stdout.readline().strip()
        assert prefix and shm_segments(prefix)
        proc.kill()
        proc.wait(10)
        assert _wait_for(lambda: shm_segments(prefix) == [], 10), shm_segments(prefix)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("usable,readers", [(1, 1), (3, 2), (8, 2), (8, 3), (4, 8)])
@pytest.mark.parametrize("cpu,io", [(0, 0), (7, 5), (1, 20)])
def test_full_reader_pool_gives_every_reader_the_total(monkeypatch, usable, readers,
                                                       cpu, io):
    """--reader-pool full: every reader process gets the whole decode total
    (what one shared pool has); the IO pool is split exactly as in split mode
    (total S3 GETs in flight unchanged); split is unchanged."""
    monkeypatch.setattr(sources, "usable_cpu_count", lambda: usable)
    shared_cpu, _ = sources.configure_arrow_pools(cpu, 0)
    split_cpu, split_io = sources.reader_pool_threads(cpu, io, readers)
    assert sources.reader_pool_threads(cpu, io, readers, "split") == (split_cpu, split_io)
    full_cpu, full_io = sources.reader_pool_threads(cpu, io, readers, "full")
    assert full_cpu == [shared_cpu] * readers and full_io == split_io
    assert split_cpu == sources.split_threads(shared_cpu, readers)


def test_reader_pool_validated(corpus):
    with pytest.raises(ValueError, match="reader pool"):
        sources.reader_pool_threads(4, 0, 2, "half")
    with pytest.raises(ValueError, match="reader pool"):
        ParquetSource(corpus[0], prefetch=2, workers=2, reader_pool="half")


def test_full_reader_pool_does_not_warn_below_one_each(corpus):
    """The below-one-each warning is split mode's: full gives each reader the
    whole total, however small."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        src = ParquetSource(corpus[0], prefetch=3, workers=3, pa_cpu_threads=2,
                            reader_pool="full")
    assert src.reader_pa_threads[0] == [2, 2, 2]


@pytest.mark.parametrize("prefetch,want", [(0, "n/a"), (2, "full"), (1, "split")])
def test_reader_pool_recorded_for_process_readers_only(corpus, prefetch, want):
    """n/a without reader processes; with ONE reader process full is the same
    pools as split (its share is the whole total), recorded as split."""
    src = ParquetSource(corpus[0], prefetch=prefetch, workers=2, reader_pool="full")
    assert src.reader_pool == want


@pytest.mark.parametrize("pool", ["split", "full"])
def test_one_reader_records_split_either_way(corpus, pool):
    """One reader (a single file, however large --prefetch): full and split
    give it the same pool (the whole total), so both record split, one config;
    two readers keep the pool asked for."""
    fs, files = sources.list_parquet(corpus[0])
    one = ParquetSource(corpus[0], prefetch=4, workers=4, fs=fs, files=files[:1],
                        pa_cpu_threads=5, reader_pool=pool)
    assert one.readers == 1 and one.reader_pool == "split"
    assert one.reader_pa_threads[0] == [5]
    two = ParquetSource(corpus[0], prefetch=4, workers=4, fs=fs, files=files[:2],
                        pa_cpu_threads=5, reader_pool=pool)
    assert two.readers == 2 and two.reader_pool == pool


@pytest.mark.parametrize("pool,want", [("split", [2, 3]), ("full", [5, 5])])
def test_reader_pools_configured_before_warm_up(corpus, tmp_path, pool, want):
    """Each reader process sizes its Arrow pools BEFORE its warm-up read: the
    decode pool the warm-up read runs with (recorded from inside it, in the
    reader) is the one assigned (split 5 over 2 = [3, 2]; full = 5 each), as
    is the IO pool (5 over 2 either way); not the fresh process's default."""
    log = tmp_path / "warm.log"
    src = ParquetSource(corpus[0], prefetch=2, workers=2, pa_cpu_threads=5,
                        pa_io_threads=5, reader_pool=pool,
                        read_fn=readhooks.WarmUpPoolProbe(str(log))).start()
    try:
        seen = [tuple(map(int, line.split())) for line in log.read_text().splitlines()]
        assert len(seen) == 2 and len({pid for pid, _, _ in seen}) == 2
        assert sorted(cpu for _, cpu, _ in seen) == want
        assert sorted(io for _, _, io in seen) == [2, 3]
        assert sorted(src.reader_pa_threads_effective[0]) == want
    finally:
        src.close()


def test_full_reader_pool_live_pool_sizes(corpus):
    """Each reader process, with --reader-pool full, sizes its decode pool to
    the WHOLE total (and reads with it); its IO pool is still its share."""
    path, xb = corpus
    src = ParquetSource(path, prefetch=3, workers=3, pa_cpu_threads=7, pa_io_threads=5,
                        reader_pool="full", read_fn=readhooks.slow_read).start()
    try:
        assert src.reader_pa_threads_effective == ([7, 7, 7], [2, 2, 1])
        assert sources.pool_split_str(src.reader_pa_threads_effective[0]) == "3x7"
        blocks = list(src)
        np.testing.assert_array_equal(np.concatenate([b.vectors for b in blocks]), xb)
        assert {b.info["args"]["pa_cpu"] for b in blocks} == {7}
    finally:
        src.close()
