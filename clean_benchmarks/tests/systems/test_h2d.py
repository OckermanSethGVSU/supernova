"""--h2d pinned / --h2d-overlap / --warmup-files (docs/io.md "Host-to-device",
"Device-to-host"). CPU only: torch's CPU device stands in for the GPU,
cudaHostRegister is mocked (a fake cudart), CUDA streams / events are faked
where the ordering on the device is what is checked."""

import csv
import functools
import gc
import importlib
import time

import faiss
import numpy as np
import pytest

import readhooks
from bench.systems import harness_cli as run
import argparse
from bench.systems import guards, loop
from bench.record import results_io, schema
from bench.verify import oracle as verify
from bench.systems import streaming as base, faiss_engine
from bench.data import sources
from conftest import ROWS
from bench.data.sources import NOT_READY, ParquetSource, shm_segments

torch = pytest.importorskip("torch")

K = 10
BASE = ["--regime", "stream", "--corpus", "x", "--csv", "x.csv"]


# ------------------------------------------------------------ flag checks

@pytest.mark.parametrize("extra, msg", [
    (["--h2d-overlap"], "--h2d-overlap needs --h2d pinned"),
    (["--h2d", "pinned", "--backend", "cpu"], "needs a GPU backend"),
    (["--h2d", "pinned", "--backend", "cuvs", "--prefetch", "0"], "needs --prefetch > 0"),
    (["--h2d", "pinned", "--backend", "cpu", "--modality", "sparse", "--query-file", "q"],
     "needs a GPU backend"),
    (["--warmup-files", "-1"], "--warmup-files must be >= 0"),
    (["--warmup-files", "2", "--warmup", "0"], "needs --warmup >= 1"),
])
def test_refused_combinations(capsys, extra, msg):
    with pytest.raises(SystemExit):
        run.parse_args(BASE + extra)
    assert msg in capsys.readouterr().err


@pytest.mark.parametrize("extra, mode", [
    ([], "pageable"),
    (["--h2d", "pinned"], "pinned"),
    (["--h2d", "pinned", "--h2d-overlap"], "pinned-overlap"),
])
@pytest.mark.parametrize("backend", ["gpu", "cuvs"])
def test_accepted_and_recorded(extra, mode, backend):
    a = run.parse_args(BASE + ["--backend", backend] + extra)
    assert run.h2d_mode(a) == mode
    assert run.h2d_mode(run.parse_args(BASE + ["--backend", "cpu"])) == "n/a"


def test_backend_refuses_overlap_without_pinned():
    b = _gpu_backend()
    with pytest.raises(ValueError, match="needs --h2d pinned"):
        b.configure_h2d("pageable", True)
    with pytest.raises(ValueError, match="ahead= needs --h2d-overlap"):
        b.search(np.zeros((3, 4), np.float32), ahead=np.zeros((3, 4), np.float32))


# -------------------------------------------------------- pinning the ring

class FakeCudart:
    """torch.cuda.cudart(): cudaHostRegister / cudaHostUnregister, logged;
    ``fail_register`` / ``fail_unregister`` return a non-zero cudaError."""

    def __init__(self, log=None):
        self.log = [] if log is None else log
        self.fail_register = self.fail_unregister = False

    def cudaHostRegister(self, ptr, size, flags):
        self.log.append(("register", ptr, size, flags))
        return 2 if self.fail_register else 0

    def cudaHostUnregister(self, ptr):
        self.log.append(("unregister", ptr))
        return 1 if self.fail_unregister else 0


def _pinner(log=None):
    cr = FakeCudart(log)
    return base.HostPinner(cr, sync=lambda: cr.log.append(("sync",))), cr


def _slot_addrs(ring):
    return [sources.buffer_address(mm) for mm in ring.mms]


@pytest.mark.parametrize("extra", [0, 1])
def test_each_slot_pinned_once_and_unpinned_on_close(corpus, extra):
    """Every ring slot is registered exactly once, in the consumer process at
    start(), with its full size and flags 0; passes register nothing more;
    close() syncs, then unregisters each slot once, before unlinking."""
    path, xb = corpus
    pin, cr = _pinner()
    src = ParquetSource(path, prefetch=2, workers=2, pin=pin, extra_slots=extra).start()
    prefix = src.shm_prefix
    try:
        ring = src.ring
        assert len(ring) == 3 + extra
        addrs = _slot_addrs(ring)
        assert cr.log == [("register", a, ring.slot_bytes, 0) for a in addrs]
        for _ in range(2):
            loop.run_pass(src, _gpu_backend(), 4, K)
        assert [e for e in cr.log if e[0] == "register"] == cr.log[:len(addrs)]
        assert len(cr.log) == len(addrs)  # nothing more during the passes
    finally:
        src.close()
    tail = cr.log[len(addrs):]
    assert tail[0] == ("sync",)
    assert sorted(tail[1:]) == sorted(("unregister", a) for a in addrs)
    assert pin.registered == {}
    assert shm_segments(prefix) == []
    src.close()  # a second close unregisters nothing twice
    assert cr.log[len(addrs):] == tail


def test_unpinned_after_consumer_error(corpus):
    """A pass that dies mid-way (the consumer's error aborts the readers):
    close() still unregisters every slot exactly once."""
    path, _ = corpus
    pin, cr = _pinner()
    src = ParquetSource(path, prefetch=2, workers=2, pin=pin, extra_slots=1).start()
    b = _gpu_backend(overlap=True)
    calls = []
    real = b._engine_search

    def dying(xq, xb_t):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt  # what Ctrl-C / SIGTERM's SystemExit do
        return real(xq, xb_t)

    b._engine_search = dying
    try:
        with pytest.raises(KeyboardInterrupt):
            loop.run_pass(src, b, 4, K)
        assert b._ahead is None  # drained before the iterator was closed
    finally:
        src.close()
    regs = [e[1] for e in cr.log if e[0] == "register"]
    unregs = [e[1] for e in cr.log if e[0] == "unregister"]
    assert len(regs) == 4 and sorted(unregs) == sorted(regs)


def test_unpinned_by_finalizer_when_never_closed(corpus):
    """A source dropped without close(): the ring's finalizer unregisters the
    slots (and holds the mappings until it has)."""
    path, _ = corpus
    pin, cr = _pinner()
    src = ParquetSource(path, prefetch=1, workers=1, pin=pin).start()
    ring = src.ring
    n = len(ring)
    procs = src._procs
    procs.abort()
    procs._fin()  # the readers' own finalizer (stops them, sweeps the names)
    del src, procs, ring
    gc.collect()
    assert len([e for e in cr.log if e[0] == "unregister"]) == n
    assert pin.registered == {}


def test_grown_slot_is_pinned_too(corpus):
    """A consumer that keeps blocks grows the ring: the new slot is
    registered as well, and unregistered with the rest."""
    path, _ = corpus
    pin, cr = _pinner()
    with ParquetSource(path, prefetch=1, workers=1, pin=pin).start() as src:
        held = list(src)  # keeps every block: the ring grows
        assert src.ring.grown > 0
        n = len(src.ring)
        assert len(pin.registered) == n
        del held
    assert len([e for e in cr.log if e[0] == "register"]) == n
    assert len([e for e in cr.log if e[0] == "unregister"]) == n


def test_register_failure_is_loud_and_cleans_up(corpus):
    path, _ = corpus
    pin, cr = _pinner()
    cr.fail_register = True
    src = ParquetSource(path, prefetch=2, workers=2, pin=pin)
    try:
        with pytest.raises(RuntimeError, match="cudaHostRegister.*failed"):
            src.start()
    finally:
        src.close()
    assert shm_segments(sources.SHM_PREFIX + str(__import__("os").getpid())) == []


def test_unregister_failure_only_warns(corpus, capsys):
    path, _ = corpus
    pin, cr = _pinner()
    cr.fail_unregister = True
    src = ParquetSource(path, prefetch=1, workers=1, pin=pin).start()
    src.close()
    assert "cudaHostUnregister" in capsys.readouterr().err


def test_pin_needs_reader_processes(corpus):
    path, _ = corpus
    pin, _ = _pinner()
    for kw in (dict(prefetch=0),):
        with pytest.raises(ValueError, match="reader processes"):
            ParquetSource(path, pin=pin, **kw)


# ------------------------------------------------------------ the poll

def test_poll_never_blocks_and_hands_over_like_next(corpus):
    """poll(): NOT_READY while the head is still being read (returning at
    once), the block once it is there, None at the end."""
    path, xb = corpus
    with ParquetSource(path, prefetch=2, workers=2, read_fn=functools.partial(
            readhooks.slow_read, default=0.3)).start() as src:
        it = src.stream(pollable=True)
        t = time.perf_counter()
        assert it.poll() is NOT_READY
        assert time.perf_counter() - t < 0.1
        got = []
        while True:
            b = it.poll()
            if b is None:
                break
            if b is NOT_READY:
                time.sleep(0.01)
                continue
            assert "handoff" in b.info
            got.append(b.vectors.copy())
            del b
        assert next(it, None) is None
        np.testing.assert_array_equal(np.concatenate(got), xb)


def test_stream_prefix(corpus):
    path, xb = corpus
    with ParquetSource(path, prefetch=2, workers=2).start() as src:
        got = [b.vectors.copy() for b in src.stream(2)]
        assert len(got) == 2
        np.testing.assert_array_equal(np.concatenate(got), xb[:ROWS[0] + ROWS[1]])
        assert len(list(src)) == len(ROWS)  # a full pass after a short one


# ------------------------------------------------------- overlap ordering

def _gpu_backend(overlap=False, pinned=None, metric="ip", dtype=np.float32, events=True,
                 log=None, engine_s=0.0):
    """A resident GpuBackend on torch's CPU device, knn_gpu = faiss.knn (as
    test_timing_detail), the uploads and engine calls logged in ``log``."""
    faiss_res = getattr(faiss, "StandardGpuResources", None)
    faiss.StandardGpuResources = lambda: object()
    try:
        b = faiss_engine.GpuBackend(K, dtype=dtype, device="cpu",
                           events=events, metric=metric)
    finally:
        if faiss_res is None:
            del faiss.StandardGpuResources
        else:
            faiss.StandardGpuResources = faiss_res
    if pinned or overlap:
        b.configure_h2d("pinned", overlap)
    log = [] if log is None else log

    def engine(xq, xb_t):
        log.append(("search", xb_t.shape[0], xb_t.data_ptr()))
        if engine_s:
            time.sleep(engine_s)
        D, I = faiss.knn(xq.numpy().astype(np.float32), xb_t.numpy().astype(np.float32),
                         b.k, metric=faiss.METRIC_INNER_PRODUCT)
        return torch.from_numpy(D), torch.from_numpy(I)

    real_upload = b._upload

    def upload(x):
        log.append(("upload", x.shape[0], x.ctypes.data))
        return real_upload(x)

    b._engine_search = engine
    b._upload = upload
    b.log = log
    if not hasattr(b, "xq"):
        b.setup(np.random.default_rng(0).standard_normal((4, 32)).astype(np.float32))
    return b


def test_copy_of_next_issued_before_search_when_ready(corpus):
    """Readers far ahead of a slow engine: every block after the first has
    been read when the search before it is launched, so its upload is
    issued BEFORE that search; the first is uploaded on demand."""
    path, _ = corpus
    b = _gpu_backend(overlap=True, engine_s=0.05)
    # file 0 slow, so file 1 (the other reader) is read by the time search 0
    # is launched; after that the 50 ms engine calls outlast every read
    slow = functools.partial(readhooks.slow_read, delays={"part-00000.parquet": 0.3})
    with ParquetSource(path, prefetch=2, workers=2, extra_slots=1,
                       read_fn=slow).start() as src:
        _, _, info = loop.run_pass(src, b, 4, K)
        assert src.ring.grown == 0  # two blocks held, the extra slot covers it
    ops = [op for op, *_ in b.log]
    # upload 0, upload 1, search 0, upload 2, search 1, ... search last
    assert ops[:3] == ["upload", "upload", "search"]
    rows = [n for op, n, _ in b.log if op == "upload"]
    assert rows == ROWS
    for i in range(1, len(ROWS)):
        up = [j for j, e in enumerate(b.log) if e[0] == "upload"][i]
        se = [j for j, e in enumerate(b.log) if e[0] == "search"][i - 1]
        assert up < se, f"block {i}'s copy not issued before search {i - 1}"
    assert info["h2d_overlapped"] == len(ROWS) - 1
    _, files = info["trace"]
    assert [f[10][2] for f in files] == [0] + [1] * (len(ROWS) - 1)


def test_on_demand_when_next_not_ready(corpus):
    """Reads slower than the engine: block i+1 is never there when search i
    is launched, so each block is uploaded on demand at its own turn
    (never waited for before search i)."""
    path, _ = corpus
    b = _gpu_backend(overlap=True)
    with ParquetSource(path, prefetch=1, workers=1, extra_slots=1, read_fn=functools.partial(
            readhooks.slow_read, default=0.15)).start() as src:
        t = time.perf_counter()
        _, _, info = loop.run_pass(src, b, 4, K)
        wall = time.perf_counter() - t
    ops = [op for op, *_ in b.log]
    assert ops == ["upload", "search"] * len(ROWS)
    assert info["h2d_overlapped"] == 0
    assert info["h2d_exposed_s"] == pytest.approx(info["h2d_s"])
    # the stall is the reads, and nothing waited for block i+1 inside a call
    assert info["engine_call_s"] < 0.1 * wall


def test_slot_not_released_before_its_copy_is_awaited(corpus):
    """The ring slot of a block whose upload was issued ahead stays owned
    (not on the free list) until the compute stream has waited on its copy,
    i.e. through the next search; the device tensor is held as long."""
    path, _ = corpus
    b = _gpu_backend(overlap=True, engine_s=0.05)
    checks = []
    slow = functools.partial(readhooks.slow_read, delays={"part-00000.parquet": 0.3})
    with ParquetSource(path, prefetch=2, workers=2, extra_slots=1,
                       read_fn=slow).start() as src:
        ring = src.ring
        real_await = b._await

        def awaiting(up):
            slot = ring.mms.index(up[0].base)
            checks.append(slot not in ring.free)
            return real_await(up)

        real_engine = b._engine_search

        def engine(xq, xb_t):
            if b._ahead is not None:
                slot = ring.mms.index(b._ahead[0].base)
                checks.append(slot not in ring.free)
                checks.append(b._ahead[1] is not None)
            return real_engine(xq, xb_t)

        b._await, b._engine_search = awaiting, engine
        loop.run_pass(src, b, 4, K)
        assert all(checks) and len(checks) >= len(ROWS) + 2 * (len(ROWS) - 1)
        assert ring.grown == 0 and len(ring.free) == len(ring)


@pytest.mark.slow
@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("events", [True, False])
def test_overlap_results_identical(corpus, queries, metric, events):
    """pinned-overlap gives bit-for-bit the pageable path's top-k (cosine
    included: the in-place normalise must not touch the host block)."""
    path, xb = corpus
    out = {}
    for name, kw in (("pageable", {}), ("pinned", dict(pinned=True)),
                     ("overlap", dict(overlap=True, engine_s=0.01))):
        b = _gpu_backend(metric=metric, events=events, **kw)
        b.setup(queries)
        with ParquetSource(path, prefetch=2, workers=2,
                           extra_slots=int(name == "overlap")).start() as src:
            D, I, info = loop.run_pass(src, b, len(queries), K)
            out[name] = (D, I, info)
    for name in ("pinned", "overlap"):
        np.testing.assert_array_equal(out[name][1], out["pageable"][1])
        np.testing.assert_array_equal(out[name][0], out["pageable"][0])
    D_ref, I_ref = faiss.knn(base.normalized_queries(queries) if metric == "cosine"
                             else queries, base.l2_normalized(xb) if metric == "cosine"
                             else xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(out["overlap"][0], out["overlap"][1], D_ref, I_ref) == []
    info = out["overlap"][2]
    if events:
        assert info["h2d_overlapped"] > 0
        assert info["h2d_exposed_s"] >= 0 and info["h2d_s"] >= 0
        assert out["pageable"][2]["h2d_exposed_s"] == out["pageable"][2]["h2d_s"]
    else:
        for i in out.values():
            assert i[2]["h2d_exposed_s"] == "" and i[2]["h2d_overlapped"] == ""
    # the bucket identity still holds
    for _, _, i in out.values():
        assert i["total_s"] == pytest.approx(i["source_stall_s"] + i["engine_call_s"]
                                             + i["merge_s"] + i["other_s"])
        _, files = i["trace"]
        assert sum(f[3] for f in files) == pytest.approx(i["engine_call_s"])
        assert sum(f[2] for f in files) <= i["source_stall_s"] + 1e-9


def test_mismatched_ahead_is_refused():
    b = _gpu_backend(overlap=True)
    x0, x1, x2 = (np.full((5, 32), v, np.float32) for v in (1, 2, 3))
    b.search(x0, ahead=x1)
    with pytest.raises(RuntimeError, match="not the block being searched"):
        b.search(x2)


# ------------------------------------------- device ordering (faked CUDA)

class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_event(self, ev):
        self.log.append(("wait_event", self.name, ev.name))

    def synchronize(self):
        self.log.append(("stream_sync", self.name))


class FakeTensor:
    def __init__(self, x, log):
        self.x, self.log = x, log

    def record_stream(self, s):
        self.log.append(("record_stream", int(self.x[0, 0]), s.name))


def test_device_ordering_with_fake_cuda(monkeypatch):
    """On CUDA: both uploads go to the COPY stream (c0 [H2D] c1, the next
    block's issued behind this one's), the compute stream WAITS on this
    block's c1 (a stream wait, no host sync) and adopts the tensor
    (record_stream); drain() synchronizes c1 of an upload never searched."""
    b = _gpu_backend(overlap=True)
    log = []
    current = ["compute"]
    b._cuda = True
    b.stream, b.copy_stream = FakeStream("compute", log), FakeStream("copy", log)

    class Ctx:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            current.append(self.s.name)

        def __exit__(self, *e):
            current.pop()

    n = [0]

    class Ev:
        def __init__(self, enable_timing=False):
            n[0] += 1
            self.name = f"ev{n[0]}"

        def record(self):
            log.append(("record", self.name, current[-1]))

        def synchronize(self):
            log.append(("ev_sync", self.name))

    monkeypatch.setattr(torch.cuda, "stream", Ctx)
    monkeypatch.setattr(torch.cuda, "Event", Ev)
    b._to_dev = lambda x: log.append(("h2d", int(x[0, 0]), current[-1])) or FakeTensor(x, log)
    del b._upload  # the real one (the instance attribute was the logging wrapper)
    x0, x1 = np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)
    up, issued = b._search_upload(x0, x1)
    assert not issued
    assert log == [("record", "ev1", "copy"), ("h2d", 0, "copy"), ("record", "ev2", "copy"),
                   ("record", "ev3", "copy"), ("h2d", 1, "copy"), ("record", "ev4", "copy")]
    log.clear()
    with b._on_stream():
        b._await(up)
    assert log == [("wait_event", "compute", "ev2"), ("record_stream", 0, "compute")]
    log.clear()
    b.drain()
    assert log == [("ev_sync", "ev4")] and b._ahead is None


# -------------------------------------------------- GPU memory guard

def test_overlap_gpu_memory_guard(monkeypatch, corpus):
    path, _ = corpus
    a = run.parse_args(["--regime", "stream", "--corpus", path, "--csv", "x.csv",
                        "--backend", "cuvs", "--h2d", "pinned", "--h2d-overlap"])
    b = _gpu_backend(overlap=True)
    src = ParquetSource(path, prefetch=1, workers=1)
    block = max(ROWS) * 32 * 4
    need = guards.overlap_gpu_need(block, max(ROWS), 32, 4, a.k, 4, 4, 0)
    assert need == 2 * block + 4 * a.k * guards.OVERLAP_RESULT_BYTES_PER_CELL + guards.OVERLAP_GPU_MARGIN
    guards.overlap_gpu_memory_guard(a, b, src, 4, np.float32)  # not CUDA: skipped
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (need - 1, 24 << 30))
    with pytest.raises(SystemExit, match="two 2000 x 32 blocks"):
        guards.overlap_gpu_memory_guard(a, b, src, 4, np.float32)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (need, 24 << 30))
    guards.overlap_gpu_memory_guard(a, b, src, 4, np.float32)


# ------------------------------------------------------- run.main end to end

def _gpu_factory(monkeypatch, pinners):
    """run.make_backend with --backend gpu on torch's CPU device (knn_gpu =
    faiss.knn), and base.host_pinner capturing a fake-cudart HostPinner."""
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)

    def knn_gpu(res, xq, xb_t, k, metric=None, **kw):
        D, I = faiss.knn(xq.numpy().astype(np.float32), xb_t.numpy().astype(np.float32), k,
                         metric=faiss.METRIC_INNER_PRODUCT)
        return torch.from_numpy(D), torch.from_numpy(I)

    def make(args, dtype):
        b = faiss_engine.GpuBackend(args.k, dtype, device="cpu",
                           events=not args.no_gpu_events, metric=args.metric)
        return b

    def pinner(backend):
        p, cr = _pinner()
        pinners.append((p, cr))
        return p

    monkeypatch.setattr(run, "make_backend", make)
    monkeypatch.setattr(base, "host_pinner", pinner)
    importlib.import_module("faiss.contrib.torch_utils")  # patches knn_gpu; replaced next
    monkeypatch.setattr(faiss, "knn_gpu", knn_gpu)


@pytest.mark.parametrize("events", [True, False])
def test_main_columns_and_warmup_files(monkeypatch, tmp_path, corpus, events):
    path, _ = corpus
    pinners = []
    _gpu_factory(monkeypatch, pinners)
    out = tmp_path / "r.csv"
    common = ["--regime", "stream", "--corpus", path, "--nq", "8", "--backend", "gpu",
              "--csv", str(out), "--verify", "--warmup", "1", "--repeats", "1",
              "--prefetch", "2", "--workers", "2"] + ([] if events else ["--no-gpu-events"])
    run.main(common)
    run.main(common + ["--h2d", "pinned", "--h2d-overlap", "--warmup-files", "2"])
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0]) == schema.CSV_FIELDS and len(rows) == 4
    assert [r["h2d_mode"] for r in rows] == ["pageable"] * 2 + ["pinned-overlap"] * 2
    assert [r["warmup_files"] for r in rows] == ["0", "0", "2", "2"]
    # warmup over 2 files: never verified, no recall; the timed pass verified
    warm, timed = rows[2], rows[3]
    assert (warm["warmup"], warm["n_files"], warm["verified"]) == ("1", "2", "not_run")
    assert int(warm["rows"]) == ROWS[0] + ROWS[1]
    assert (timed["n_files"], timed["verified"]) == (str(len(ROWS)), "ok")
    assert rows[0]["verified"] == rows[1]["verified"] == "ok"
    for r in rows:
        if events:
            assert float(r["h2d_exposed_s"]) >= 0 and r["h2d_overlapped"] != ""
        else:
            assert r["h2d_exposed_s"] == r["h2d_overlapped"] == r["h2d_s"] == ""
    if events:
        assert rows[0]["h2d_exposed_s"] == rows[0]["h2d_s"]
        assert int(rows[0]["h2d_overlapped"]) == 0
    trows = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert list(trows[0]) == schema.TRACE_FIELDS
    p_warm = [t for t in trows if t["t_start_unix"] == warm["t_start_unix"]]
    p_timed = [t for t in trows if t["t_start_unix"] == timed["t_start_unix"]]
    assert len(p_warm) == 2 and len(p_timed) == len(ROWS)
    assert {t["h2d_mode"] for t in p_timed} == {"pinned-overlap"}
    if events:
        assert sum(int(t["h2d_overlapped"]) for t in p_timed) == int(timed["h2d_overlapped"])
        assert sum(float(t["h2d_exposed_s"]) for t in p_timed) == \
            pytest.approx(float(timed["h2d_exposed_s"]))
    else:
        assert {t["h2d_exposed_s"] for t in p_timed} == {""}
    # one pinner for the pinned run: every slot registered, all unregistered
    (pin, cr), = pinners
    regs = [e for e in cr.log if e[0] == "register"]
    assert len(regs) == 2 + 1 + 1 and pin.registered == {}
    assert sorted(e[1] for e in cr.log if e[0] == "unregister") == sorted(e[1] for e in regs)


def test_main_unpins_on_signal_exit(monkeypatch, tmp_path, corpus):
    """SIGTERM / SIGHUP become SystemExit in the harness; the run's finally closes
    the source, which unregisters every slot."""
    path, _ = corpus
    pinners = []
    _gpu_factory(monkeypatch, pinners)
    real = base.ResidentBackend.search
    calls = []

    def dying(self, xb, ahead=None, offset=0):
        calls.append(1)
        if len(calls) == 3:
            raise SystemExit(143)
        return real(self, xb, ahead, offset)

    monkeypatch.setattr(base.ResidentBackend, "search", dying)
    with pytest.raises(SystemExit):
        run.main(["--regime", "stream", "--corpus", path, "--nq", "8", "--backend", "gpu",
                  "--csv", str(tmp_path / "r.csv"), "--h2d", "pinned", "--h2d-overlap"])
    (pin, cr), = pinners
    assert pin.registered == {} and any(e[0] == "unregister" for e in cr.log)


# ------------------------------------------------------------- --d2h pinned

@pytest.mark.parametrize("extra, msg", [
    (["--d2h", "pinned", "--backend", "cpu"], "needs a GPU backend"),
    (["--d2h", "pinned", "--backend", "cpu", "--modality", "sparse", "--query-file", "q"],
     "needs a GPU backend"),
])
def test_d2h_refused_combinations(capsys, extra, msg):
    with pytest.raises(SystemExit):
        run.parse_args(BASE + extra)
    assert msg in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--h2d", "pinned"], ["--h2d", "pinned", "--h2d-overlap"], ["--d2h", "pinned"],
    ["--d2h", "pinned", "--d2h-overlap"],
    ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap",
     "--read-mode", "native"]])
def test_sparse_cuvs_accepts_the_transfer_modes(extra):
    """--modality sparse --backend cuvs runs ResidentBackend's submit/complete
    (SparseCuvsBackend overrides only _block_search), so every dense --h2d /
    --d2h mode applies (tests/test_sparse.py runs them end to end)."""
    a = run.parse_args(BASE + ["--backend", "cuvs", "--modality", "sparse", "--query-file",
                               "q"] + extra)
    assert a.modality == "sparse"


@pytest.mark.parametrize("backend", ["gpu", "cuvs"])
def test_d2h_accepted_any_reader_and_regime(backend):
    """Unlike --h2d pinned, nothing about the host block matters: its own buffers."""
    common = ["--corpus", "x", "--csv", "x.csv", "--backend", backend, "--d2h", "pinned"]
    for extra in (["--regime", "stream"], ["--regime", "stream", "--prefetch", "0"]):
        a = run.parse_args(extra + common)
        assert a.d2h == "pinned"


def test_d2h_buffers_alternate_and_are_reused(corpus, queries):
    """Two buffers per result shape, allocated once, used alternately: block i's
    results are still intact while block i+1's arrive (run_pass merges each
    before the next search; this is the margin)."""
    path, _ = corpus
    b = _gpu_backend()
    b.configure_d2h("pinned")
    b.setup(queries)
    xb = np.random.default_rng(1).standard_normal((50, 32)).astype(np.float32)
    D1, I1 = b.search(xb)
    keep = (D1.copy(), I1.copy())
    D2, I2 = b.search(xb[::-1].copy())
    np.testing.assert_array_equal(D1, keep[0])   # not overwritten by the next call
    np.testing.assert_array_equal(I1, keep[1])
    assert D1.ctypes.data != D2.ctypes.data
    D3, _ = b.search(xb)
    assert D3.ctypes.data == D1.ctypes.data       # the first buffer again
    assert len(b._d2h_bufs) == 1                  # one shape, allocated once
    # a short block (< k rows): still k columns, -1 past its rows
    Ds, Is = b.search(xb[:3])
    assert Ds.shape == (len(queries), K) and (Is[:, 3:] == -1).all()


def test_d2h_short_shapes_reuse_the_full_buffers():
    """Results of distinct short widths (k_eff < k: blocks of fewer than k
    rows) are views into the one flat pinned pair, never new allocations, and
    still alternate (block i intact while block i+1 is written)."""
    b = _gpu_backend()
    b.configure_d2h("pinned")
    t = b.torch
    full = (t.ones(4, K), t.ones(4, K, dtype=t.int64))
    b._to_host(*full)
    b._to_host(*full)
    bases = {p[0].data_ptr() for p in b._d2h_bufs[(t.float32, t.int64)]}
    prev = None
    for w in (3, 1, 2, 3):
        D, I = t.full((4, w), float(w)), t.full((4, w), w, dtype=t.int64)
        Dh, Ih = b._to_host(D, I)
        assert Dh.shape == (4, w) and (Ih == w).all() and Dh.data_ptr() in bases
        if prev is not None:
            assert (prev[0] == prev[1]).all()   # the previous result not overwritten
        prev = (Dh, float(D[0, 0]))
    assert len(b._d2h_bufs) == 1


def test_d2h_narrow_first_result_allocates_full_k():
    """A narrow FIRST result (a first block of fewer than k rows) still
    allocates the full nq x k pair, so the later full-width blocks reuse it
    (no regrow, no stale page-locked pair left in torch's host cache)."""
    b = _gpu_backend()
    b.configure_d2h("pinned")
    t = b.torch
    Dh, _ = b._to_host(t.zeros(4, 2), t.zeros(4, 2, dtype=t.int64))
    pair = b._d2h_bufs[(t.float32, t.int64)]
    assert pair[0][0].numel() == 4 * K and pair[0][1].numel() == 4 * K
    D2, _ = b._to_host(t.ones(4, K), t.ones(4, K, dtype=t.int64))
    assert b._d2h_bufs[(t.float32, t.int64)] is pair and (Dh == 0).all()


def test_d2h_pinned_bytes_charges_power_of_two_rounding():
    a = argparse.Namespace(d2h="pinned", k=1000)
    assert guards.d2h_pinned_bytes(a, 25000) == 2 * ((128 << 20) + (256 << 20))
    assert guards.d2h_pinned_bytes(a, 1) == 2 * (4096 + 8192)


@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("events", [True, False])
@pytest.mark.parametrize("overlap", [False, True])
def test_d2h_pinned_results_identical(corpus, queries, metric, events, overlap):
    """--d2h pinned (alone and with --h2d pinned --h2d-overlap) gives bit for
    bit the pageable top-k, and the bucket identity still holds."""
    path, _ = corpus
    out = {}
    for name in ("pageable", "pinned"):
        b = _gpu_backend(metric=metric, events=events, overlap=overlap,
                         engine_s=0.01 if overlap else 0.0)
        if name == "pinned":
            b.configure_d2h("pinned")
        b.setup(queries)
        with ParquetSource(path, prefetch=2, workers=2, extra_slots=int(overlap)).start() as src:
            out[name] = loop.run_pass(src, b, len(queries), K)
    np.testing.assert_array_equal(out["pinned"][0], out["pageable"][0])
    np.testing.assert_array_equal(out["pinned"][1], out["pageable"][1])
    i = out["pinned"][2]
    assert i["total_s"] == pytest.approx(i["source_stall_s"] + i["engine_call_s"]
                                         + i["merge_s"] + i["other_s"])


def test_main_records_d2h_mode(monkeypatch, tmp_path, corpus):
    path, _ = corpus
    _gpu_factory(monkeypatch, [])
    out = tmp_path / "r.csv"
    common = ["--regime", "stream", "--corpus", path, "--nq", "8", "--backend", "gpu",
              "--csv", str(out), "--verify", "--warmup", "0", "--repeats", "1",
              "--prefetch", "2", "--workers", "2"]
    run.main(common)
    run.main(common + ["--d2h", "pinned"])
    rows = list(csv.DictReader(open(out)))
    assert [r["d2h_mode"] for r in rows] == ["pageable", "pinned"]
    assert [r["verified"] for r in rows] == ["ok", "ok"]
    trows = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert {t["d2h_mode"] for t in trows} == {"pageable", "pinned"}
