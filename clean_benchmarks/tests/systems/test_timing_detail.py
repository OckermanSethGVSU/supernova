"""REVISION 4 annotations: the CUDA-event chain inside engine_call (mocked:
no GPU here), the guarded end-of-call wait, the read_table / convert split
of decode, the always-on per-file trace, /proc/net/dev deltas, the
wall-clock pass stamps and gpu_busy.py's join. None of these may change the
buckets.

GPU mocks: GpuBackend.__init__ (resident) imports faiss.contrib.torch_utils,
which on its FIRST import in the process replaces faiss.knn_gpu (keeping
what was there as knn_gpu_numpy). A knn_gpu mock installed before that
construction is silently overridden, and whether it is depends on test
order, so every mock here is installed AFTER the backend is built."""

import csv
import os
import time

import faiss
import numpy as np
import pytest

from bench.systems import harness_cli as run
import resource
from bench.systems import loop
from bench.record import results_io, schema
from bench.verify import oracle as verify
from bench.systems import cpu_reference, faiss_engine
from bench.data import sources
from conftest import ROWS
from bench.data.sources import InMemorySource, ParquetSource, list_parquet, read_file

K = 10
DEV = ("h2d_s", "widen_s", "search_s", "d2h_s")


# ------------------------------------------------- CUDA event chain (mocked)

class Clock:
    """Fake device timeline in ms: each mocked op advances it by a distinct
    amount, so every event pair's elapsed time says which ops it spans."""

    def __init__(self, query_done=True):
        self.ms, self.log, self.query_done = 0.0, [], query_done

    def op(self, name, ms):
        self.ms += ms
        self.log.append(name)


def fake_event_cls(clock):
    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing

        def record(self):
            self.t = clock.ms
            clock.log.append("ev")

        def query(self):
            clock.log.append("query")
            return clock.query_done

        def synchronize(self):
            clock.log.append("sync")
            time.sleep(0.002)

        def elapsed_time(self, other):
            return other.t - self.t
    return FakeEvent


class Out:
    """knn_gpu's D / I: .cpu() is the D2H (+8 ms each on the fake clock)."""

    def __init__(self, t, clock):
        self.t, self.clock = t, clock

    def cpu(self):
        self.clock.op("d2h", 8)
        return self.t


def cuda_backend(monkeypatch, clock, dtype=np.float32, events=True):
    """A resident GpuBackend on torch's CPU device with the ops on the fake
    clock. knn_gpu is mocked after construction (see the module docstring)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    b = faiss_engine.GpuBackend(K, dtype=dtype, device="cpu",
                       events=events)
    real_to_dev, real_cast = b._to_dev, b._cast

    def to_dev(x):
        clock.op("h2d", 1)
        return real_to_dev(x)

    def cast(x):
        clock.op("cast", 2)
        return real_cast(x)

    def knn_gpu(res, xq, xb_t, k, metric=None, **kw):
        clock.op("knn", 4)
        Dn, In = faiss.knn(xq.numpy().astype(np.float32),
                           xb_t.numpy().astype(np.float32), k,
                           metric=faiss.METRIC_INNER_PRODUCT)
        return Out(torch.from_numpy(Dn), clock), Out(torch.from_numpy(In), clock)

    monkeypatch.setattr(b, "_to_dev", to_dev)
    monkeypatch.setattr(b, "_cast", cast)
    monkeypatch.setattr(faiss, "knn_gpu", knn_gpu)
    monkeypatch.setattr(b, "_globalize", globalize_out(b._globalize, clock))
    fake_alloc_counters(monkeypatch, torch, clock)
    return b, torch


def globalize_out(real, clock):
    """The device-side id offset (ResidentBackend._globalize) applied to the
    tensor an ``Out`` wraps; no fake-clock op (it is part of the search span,
    and the event-chain tests pin the op sequence)."""
    def g(I, offset):
        return Out(real(I.t, offset), clock) if isinstance(I, Out) else real(I, offset)
    return g


def fake_alloc_counters(monkeypatch, torch, clock, peak=12345):
    """torch.cuda's peak counters (host-side, no sync) for tests that flip
    ``_cuda`` on: logged outside the per-block op sequence."""
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda *a: clock.resets.append(a))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: peak)
    clock.resets = []


@pytest.mark.parametrize("stored", ["f16", "f32"])
def test_event_chain_order_and_pairing(monkeypatch, tmp_path, queries, stored):
    """e0 [H2D] e1 [cast] e2 [knn_gpu] e3 [D/I .cpu()] e4, then the guard:
    each column is exactly its own pair's span; no cast => e2 is e1."""
    from test_run import _fp16_pair
    f16, f32, xb = _fp16_pair(tmp_path)
    q = np.ascontiguousarray(queries[:, :24] / np.linalg.norm(queries[:, :24], axis=1,
                                                               keepdims=True))
    clock = Clock()
    b, torch = cuda_backend(monkeypatch, clock)
    b.setup(q)  # on torch's CPU device: plain upload, no CUDA sync
    b._cuda = True  # from here on: CUDA events (faked)
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a: pytest.fail("no device-wide sync in search"))
    clock.log.clear()
    src = InMemorySource.load(f16 if stored == "f16" else f32)
    D, I, info = loop.run_pass(src, b, len(q), K)
    n = len(src.blocks)
    cast = stored == "f16"
    per_block = ["ev", "h2d", "ev"] + (["cast", "ev"] if cast else []) + \
        ["ev", "knn", "ev", "ev", "ev", "d2h", "d2h", "ev", "query"]
    assert clock.log == per_block * n
    assert info["h2d_s"] == pytest.approx(n * 1e-3)
    assert info["widen_s"] == pytest.approx(n * 2e-3 if cast else 0.0)
    assert info["search_s"] == pytest.approx(n * 4e-3)
    # the library call (knn) is all of search_s here; no harness work beside it
    assert info["lib_s"] == pytest.approx(n * 4e-3)
    assert info["search_harness_s"] == pytest.approx(0.0, abs=1e-12)
    assert info["d2h_s"] == pytest.approx(n * 16e-3)
    assert info["engine_gap_s"] == pytest.approx(
        info["engine_call_s"] - sum(info[c] for c in DEV))
    assert info["sync_wait_s"] == 0 and info["sync_waits"] == 0
    # gpu_idle_s: the pass minus the compute stream's busy spans (cast + search;
    # the H2D / D2H on this stream and the host side are idle)
    assert info["gpu_idle_s"] == pytest.approx(
        info["total_s"] - info["widen_s"] - info["search_s"])
    D_ref, I_ref = faiss.knn(q, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []


def test_guarded_wait_only_when_e4_not_done(monkeypatch, corpus, queries):
    clock = Clock(query_done=False)
    b, torch = cuda_backend(monkeypatch, clock)
    b.setup(queries)
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a: pytest.fail("no device-wide sync in search"))
    clock.log.clear()
    src = InMemorySource.load(corpus[0])
    info = loop.run_pass(src, b, len(queries), K)[2]
    n = len(src.blocks)
    # every block: query() said "not done" -> exactly one synchronize on e4,
    # after the last event and before the elapsed times are read
    assert clock.log.count("sync") == n and clock.log.count("query") == n
    assert clock.log[-3:] == ["ev", "query", "sync"]
    assert info["sync_waits"] == n
    assert info["sync_wait_s"] >= n * 0.002  # the wait is timed
    assert info["sync_wait_s"] < info["engine_call_s"]  # and inside engine_call
    assert b.last_wait[1] is True


def test_device_columns_zero_off_the_resident_gpu_path(corpus, queries, monkeypatch):
    """The CPU backend records no events at all: the event
    columns are blank (not measured, never 0) and the trace's wait is blank
    too, as with --no-gpu-events (widen_s too: an f32 corpus has nothing to
    widen on the host)."""
    b = cpu_reference.CpuBackend(K)
    b.setup(queries)
    info = loop.run_pass(ParquetSource(corpus[0], prefetch=2, workers=2), b,
                        len(queries), K)[2]
    for c in DEV + ("engine_gap_s", "gpu_idle_s", "sync_wait_s", "sync_waits"):
        assert info[c] == "", c
    trace = loop.trace_rows(dict.fromkeys(schema.TRACE_CONFIG, ""), 0, 0, 0.0, info["trace"])
    assert {r["sync_wait_s"] for r in trace} == {""}


def test_resident_on_host_device_never_queries(monkeypatch, corpus, queries):
    """torch's CPU device (the other GPU mock tests): host stamps, no
    event.query(), wait 0, but search / d2h are populated."""
    clock = Clock()
    b, _ = cuda_backend(monkeypatch, clock)
    b.setup(queries)
    info = loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)[2]
    assert "query" not in clock.log and info["sync_waits"] == 0
    assert info["search_s"] > 0 and info["d2h_s"] > 0 and info["sync_wait_s"] == 0


# ------------------------------------------------------ --no-gpu-events

def test_no_gpu_events_creates_no_events(monkeypatch, tmp_path, queries):
    """--no-gpu-events: no Event object, no record / query / event sync; one
    device-wide synchronize per block; event columns empty, buckets intact."""
    from test_run import _fp16_pair
    f16, _, xb = _fp16_pair(tmp_path)
    q = np.ascontiguousarray(queries[:, :24] / np.linalg.norm(queries[:, :24], axis=1,
                                                               keepdims=True))
    clock = Clock()
    b, torch = cuda_backend(monkeypatch, clock, events=False)
    b.setup(q)
    b._cuda = True

    def no_event(*a, **kw):
        raise AssertionError("a CUDA event was created")

    monkeypatch.setattr(torch.cuda, "Event", no_event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: clock.log.append("devsync"))
    clock.log.clear()
    src = InMemorySource.load(f16)
    D, I, info = loop.run_pass(src, b, len(q), K)
    assert clock.log == ["h2d", "cast", "knn", "d2h", "d2h", "devsync"] * len(src.blocks)
    for c in schema.EVENT_FIELDS:
        assert info[c] == "", c
    assert info["engine_call_s"] > 0 and info["other_s"] >= -1e-9
    D_ref, I_ref = faiss.knn(q, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []


def gpu_backend_factory(monkeypatch, knn_gpu):
    """run.make_backend replacement: a GpuBackend on torch's CPU device, with
    knn_gpu mocked only once it is built (see the module docstring)."""
    def make(args, dtype):
        b = faiss_engine.GpuBackend(args.k, dtype, device="cpu",
                           events=not args.no_gpu_events)
        monkeypatch.setattr(faiss, "knn_gpu", knn_gpu)
        return b
    return make


def _torch_knn(torch):
    def knn_gpu(res, xq, xb_t, k, metric=None, **kw):
        Dn, In = faiss.knn(xq.numpy(), xb_t.numpy(), k, metric=faiss.METRIC_INNER_PRODUCT)
        return torch.from_numpy(Dn), torch.from_numpy(In)
    return knn_gpu


def test_engine_gap_stored_raw_when_negative(monkeypatch, corpus, queries):
    """Event time larger than the wall-clock call (event granularity on a
    real GPU, exaggerated here): engine_gap_s is call - events, NOT clamped."""
    clock = Clock()
    b, torch = cuda_backend(monkeypatch, clock)
    b.setup(queries)
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    monkeypatch.setattr(b, "_elapsed", lambda a, z: 10.0)  # 10 s per event pair
    src = InMemorySource.load(corpus[0])
    info = loop.run_pass(src, b, len(queries), K)[2]
    dev = sum(info[c] for c in DEV)
    assert dev == pytest.approx(3 * 10.0 * len(src.blocks))  # no cast: widen 0
    assert info["engine_gap_s"] < 0
    assert info["engine_gap_s"] == pytest.approx(info["engine_call_s"] - dev)


@pytest.mark.slow
def test_no_gpu_events_blanks_round_trip_csv_and_plot(monkeypatch, corpus, tmp_path, capsys):
    import pandas as pd
    from bench.report import plots as plot
    torch = pytest.importorskip("torch")
    knn_gpu = _torch_knn(torch)
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    monkeypatch.setattr(run, "make_backend", gpu_backend_factory(monkeypatch, knn_gpu))
    out, tr = tmp_path / "r.csv", tmp_path / "r.trace.csv"
    base = ["--regime", "stream", "--corpus", corpus[0], "--nq", "16", "--warmup", "0",
            "--repeats", "3", "--csv", str(out), "--verify"]
    run.main(base + ["--no-gpu-events"])
    run.main(base + ["--no-gpu-events"])  # header check passes with blanks
    run.main(base)  # events on (host stamps on torch's CPU device)
    rows = list(csv.DictReader(open(out)))
    off, on = rows[:6], rows[6:]
    assert {r["gpu_events"] for r in off} == {"off"} and {r["gpu_events"] for r in on} == {"on"}
    for r in off:
        assert all(r[c] == "" for c in schema.EVENT_FIELDS)
        assert float(r["engine_call_s"]) > 0 and float(r["decode_s"]) > 0
    for r in on:
        assert float(r["search_s"]) > 0 and r["sync_waits"] == "0"
    trows = list(csv.DictReader(open(tr)))
    assert {(r["sync_wait_s"], r["sync_waited"]) for r in trows[:6 * len(ROWS)]} == {("", "")}
    assert {r["sync_waited"] for r in trows[6 * len(ROWS):]} == {"0"}
    med = plot.median_rows(pd.read_csv(out, keep_default_na=False))
    assert sorted(med["gpu_events"]) == ["off", "on"]  # never one bar
    png = tmp_path / "a.png"
    assert plot.attribution([str(out)], str(png)) == 0
    printed = capsys.readouterr().out
    assert png.stat().st_size > 10_000
    num = plot.numeric_annotations(med).set_index("gpu_events")
    ev = [c for c in plot.ANNOTATIONS if c in schema.EVENT_FIELDS and c != "widen_s"]
    assert num.loc["off", ev].isna().all() and num.loc["on", ev].notna().all()
    assert "off" in printed and "on" in printed
    # sync_waits is a count: printed as an integer, blank when not measured
    lines = plot.table(num.reset_index(), ["gpu_events", "sync_waits", "search_s"]).splitlines()
    cells = {ln.split()[0]: ln.split()[1:] for ln in lines[1:]}
    assert cells["on"][0] == "0" and len(cells["on"]) == 2
    assert cells["off"] == []  # both blank


# ------------------------------------------------------------ decode split

@pytest.mark.parametrize("mode", ["arrow", "native"])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_read_table_plus_convert_is_decode(corpus, mode, prefetch):
    src = ParquetSource(corpus[0], prefetch=prefetch, workers=2, read_mode=mode)
    infos = [b.info for b in src]
    st = src.stats.as_dict()
    assert st["read_table"] > 0 and st["convert"] > 0
    assert st["read_table"] + st["convert"] == pytest.approx(st["decode"], rel=1e-9)
    assert sum(i["read_table"] for i in infos) == pytest.approx(st["read_table"])
    assert sum(i["convert"] for i in infos) == pytest.approx(st["convert"])
    assert sum(i["fetch"] for i in infos) == pytest.approx(st["fetch"])
    for i in infos:
        assert i["w_end"] >= i["w_start"]
        assert i["w_end"] - i["w_start"] == pytest.approx(
            i["fetch"] + i["read_table"] + i["convert"] + i["widen"], abs=1e-6)
        # arrow: fetch is the footer read, done in-pass on an inline first
        # pass; process readers get every footer from start(), read before it
        assert (i["fetch"] > 0) == (mode != "arrow" or prefetch == 0)


def test_arrow_footer_read_is_fetch_not_widen(corpus, monkeypatch):
    """arrow mode's (uncached) footer read is a GET: it is timed as fetch,
    inside [w_start, w_end], and never lands in widen (host widen on) or in
    no bucket at all (widen off)."""
    fs, refs = list_parquet(corpus[0])
    orig = sources.read_footer

    def slow_footer(*a, **kw):
        time.sleep(0.3)
        return orig(*a, **kw)

    monkeypatch.setattr(sources, "read_footer", slow_footer)
    for dtype in (None, np.float16):
        rec = {}
        read_file(fs, refs[0].path, "embedding", dtype, size=refs[0].size, mode="arrow",
                  footers={}, rec=rec)
        assert rec["fetch"] >= 0.3 and rec["widen"] < 0.2, rec
        assert rec["w_end"] - rec["w_start"] >= 0.3


def test_read_file_rec_and_memory_blocks_have_no_info(corpus):
    fs, refs = list_parquet(corpus[0])
    rec = {}
    read_file(fs, refs[0].path, "embedding", np.float16, size=refs[0].size, rec=rec)
    assert set(rec) == {"w_start", "w_end", "fetch", "read_table", "convert", "widen"}
    assert rec["widen"] > 0
    assert all(b.info is None for b in InMemorySource.load(corpus[0]))


# ------------------------------------------------------ per-file trace

def _main(regime, corpus, out, extra=()):
    """run.main on the CPU backend; returns the trace path it wrote."""
    run.main(["--regime", regime, "--corpus", corpus, "--backend", "cpu", "--nq", "16",
              "--warmup", "1", "--repeats", "2", "--csv", str(out)] + list(extra))
    return results_io.trace_path(str(out))


@pytest.mark.parametrize("csv_path,trace", [
    ("fw_x.csv", "fw_x.trace.csv"),
    ("/a/b.c/res.csv", "/a/b.c/res.trace.csv"),
    ("res", "res.trace.csv"),
    ("res.tsv", "res.tsv.trace.csv"),
])
def test_trace_path(csv_path, trace):
    assert results_io.trace_path(csv_path) == trace


@pytest.mark.parametrize("regime,extra", [
    ("stream", ["--prefetch", "0"]),
    ("stream", ["--prefetch", "3", "--workers", "2"]),
])
def test_trace_rows_per_file_per_pass(corpus, tmp_path, regime, extra):
    out = tmp_path / "r.csv"
    tr = _main(regime, corpus[0], out, extra)
    runs = list(csv.DictReader(open(out)))
    rows = list(csv.DictReader(open(tr)))
    assert list(rows[0].keys()) == schema.TRACE_FIELDS
    assert len(rows) == len(ROWS) * 3  # files x (1 warmup + 2 repeats)
    assert [r["warmup"] for r in rows] == ["1"] * len(ROWS) + ["0"] * 2 * len(ROWS)
    for p in runs:  # each pass's per-file rows add up to its buckets
        mine = [r for r in rows if r["t_start_unix"] == p["t_start_unix"]]
        assert [int(r["file_index"]) for r in mine] == list(range(len(ROWS)))
        assert [int(r["rows"]) for r in mine] == ROWS
        assert sum(float(r["engine_call_s"]) for r in mine) == \
            pytest.approx(float(p["engine_call_s"]))
        assert sum(float(r["stall_s"]) for r in mine) <= float(p["source_stall_s"]) + 1e-9
        assert sum(float(r["merge_s"]) for r in mine) <= float(p["merge_s"]) + 1e-9
        for c in schema.TRACE_CONFIG:
            assert all(r[c] == p[c] for r in mine), c
    for r in rows:
        assert 0 <= float(r["t_got_s"]) <= float(r["t_done_s"])
        if regime == "memory":
            assert r["w_start_s"] == r["read_table_s"] == r["widen_host_s"] == ""
        else:
            # same clock: a file is read before the consumer gets it
            assert 0 <= float(r["w_start_s"]) <= float(r["w_end_s"]) \
                <= float(r["t_got_s"]) + 1e-9
            assert float(r["read_table_s"]) > 0 and float(r["convert_s"]) > 0
            if r["reader"] == "process":
                # the copy into the ring slot follows the read, before the handoff
                assert float(r["w_end_s"]) <= float(r["w_publish_end_s"]) \
                    <= float(r["t_got_s"]) + 1e-9
                assert float(r["w_publish_end_s"]) - float(r["w_end_s"]) == \
                    pytest.approx(float(r["publish_s"]), abs=1e-3)
            else:
                assert r["w_publish_end_s"] == ""


def test_trace_header_mismatch_refused(corpus, tmp_path):
    (tmp_path / "r.trace.csv").write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        _main("stream", corpus[0], tmp_path / "r.csv")
    assert not (tmp_path / "r.csv").exists()  # refused before any pass


def test_trace_always_written_next_to_csv(corpus, tmp_path):
    """No flag: every run writes <csv stem>.trace.csv beside the run CSV,
    one row per file per pass, appended across runs like the run CSV."""
    out = tmp_path / "res"  # no .csv suffix: .trace.csv appended
    tr = _main("stream", corpus[0], out)
    assert sorted(os.listdir(tmp_path)) == ["res", "res.trace.csv"]
    _main("stream", corpus[0], out)
    assert len(list(csv.DictReader(open(out)))) == 2 * 3
    assert len(list(csv.DictReader(open(tr)))) == 2 * 3 * len(ROWS)
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", corpus[0], "--csv", "x.csv",
                        "--trace", "t.csv"])  # the old flag is gone


def test_trace_carries_the_run_config(corpus, tmp_path):
    """Trace rows carry every config column the plot separates runs by (all
    but the total rows), so plot_trace can filter on them (--where corpus=,
    max_files=, dtype=, ...) and titles each panel with the ones that vary
    between the passes drawn."""
    import pandas as pd
    from bench.report import plots as plot
    assert set(schema.TRACE_CONFIG) == set(plot.CONFIG) - {"rows"}
    out = tmp_path / "r.csv"
    tr = _main("stream", corpus[0], out)
    _main("stream", corpus[0], out, ["--max-files", "2"])
    df = pd.read_csv(tr, keep_default_na=False, dtype=str)
    for c in ("backend", "corpus", "rows_per_file", "dtype", "widen_where", "gpu_events",
              "queries_mode", "range_concurrency", "max_files"):
        assert c in df.columns, c
    assert set(df["max_files"]) == {"0", "2"} and set(df["backend"]) == {"cpu"}


# ------------------------------------------------- NIC counters, unix stamps

NET_DEV_TEXT = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets
    lo: {lo} 10 0 0 0 0 0 0 999 10 0 0 0 0 0 0
  eth0: {eth0} 20 0 0 0 0 0 0 5 1 0 0 0 0 0 0
ens5:{ens5} 30 0 0 0 0 0 0 5 1 0 0 0 0 0 0
"""


def test_net_rx_bytes_parsing(tmp_path):
    f = tmp_path / "dev"
    f.write_text(NET_DEV_TEXT.format(lo=10**9, eth0=1234, ens5=100))
    assert loop.net_rx_bytes(str(f)) == 1334  # lo excluded, no-space "ens5:" ok
    assert loop.net_rx_bytes(str(tmp_path / "missing")) is None


def test_net_rx_delta_and_rate_per_pass(corpus, queries, tmp_path, monkeypatch):
    f = tmp_path / "dev"
    real_read = loop.net_rx_bytes
    calls = []

    def bump(path=None):
        # each read sees 5 MB more on eth0 than the last
        f.write_text(NET_DEV_TEXT.format(lo=7, eth0=5_000_000 * len(calls), ens5=0))
        calls.append(1)
        return real_read(str(f))

    monkeypatch.setattr(loop, "net_rx_bytes", bump)
    b = cpu_reference.CpuBackend(K)
    b.setup(queries)
    info = loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)[2]
    assert len(calls) == 2 and info["net_rx_bytes"] == 5_000_000
    assert info["net_rx_MBps"] == pytest.approx(5.0 / info["total_s"])
    monkeypatch.setattr(schema, "NET_DEV", str(tmp_path / "missing"))
    monkeypatch.setattr(loop, "net_rx_bytes", real_read)
    info = loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)[2]
    assert info["net_rx_bytes"] == "" and info["net_rx_MBps"] == ""


def test_net_and_rusage_reads_outside_the_timed_span(corpus, queries, monkeypatch):
    """The /proc/net/dev and getrusage reads are taken outside [t_start,
    t_end]: made 50 ms slow each, none of them lands inside the pass and
    total_s does not grow by them."""
    import resource
    spans = []

    def slow(fn):
        def wrapped(*a, **kw):
            t0 = time.time()
            time.sleep(0.05)
            r = fn(*a, **kw)
            spans.append((t0, time.time()))
            return r
        return wrapped

    monkeypatch.setattr(loop, "net_rx_bytes", slow(loop.net_rx_bytes))
    monkeypatch.setattr(resource, "getrusage", slow(resource.getrusage))
    b = cpu_reference.CpuBackend(K)
    b.setup(queries)
    t0 = time.perf_counter()
    info = loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)[2]
    wall = time.perf_counter() - t0
    assert len(spans) == 4  # net + rusage, before and after
    for a, z in spans:
        assert z <= info["t_start_unix"] or a >= info["t_end_unix"]
    assert info["total_s"] <= wall - 4 * 0.05


def test_unix_stamps_bracket_the_pass(corpus, queries):
    b = cpu_reference.CpuBackend(K)
    b.setup(queries)
    t0 = time.time()
    info = loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)[2]
    t1 = time.time()
    assert t0 <= info["t_start_unix"] <= info["t_end_unix"] <= t1
    assert info["t_end_unix"] - info["t_start_unix"] == pytest.approx(
        info["total_s"], abs=0.01)


def test_cli_columns_populated(corpus, tmp_path):
    out = tmp_path / "r.csv"
    _main("stream", corpus[0], out, ["--prefetch", "2", "--workers", "2"])
    r = list(csv.DictReader(open(out)))[1]
    assert float(r["read_table_s"]) + float(r["convert_s"]) == \
        pytest.approx(float(r["decode_s"]))
    assert float(r["t_end_unix"]) > float(r["t_start_unix"]) > 1.7e9
    if os.path.exists(schema.NET_DEV):
        assert int(r["net_rx_bytes"]) >= 0 and float(r["net_rx_MBps"]) >= 0
    # cpu: no events, so not measured: blank, never 0
    assert r["search_s"] == r["d2h_s"] == r["engine_gap_s"] == ""
    assert r["sync_waits"] == "" and r["gpu_events"] == "n/a"


def test_device_id_offset_equals_the_host_offset(monkeypatch):
    """ResidentBackend._globalize (GPU backends: ids made global on the device,
    before D2H) == run_pass's host path (np.add in place), -1 kept as -1, and
    run_pass hands each block its own offset: a 3-block pass returns the same
    global ids whichever side adds the offset."""
    torch = pytest.importorskip("torch")
    b = faiss_engine.GpuBackend.__new__(faiss_engine.GpuBackend)
    b.torch = torch
    I = torch.tensor([[3, 0, -1], [7, -1, -1]], dtype=torch.int64)
    got = b._globalize(I, 40000).numpy()
    host = I.numpy().copy()
    np.add(host, 40000, out=host, where=host >= 0)
    np.testing.assert_array_equal(got, host)
    np.testing.assert_array_equal(got, [[40003, 40000, -1], [40007, -1, -1]])
    assert b._globalize(I, 0) is I  # offset 0: untouched, no kernel

    rng = np.random.default_rng(0)
    xq = rng.standard_normal((5, 8)).astype(np.float32)
    blocks = [rng.standard_normal((n, 8)).astype(np.float32) for n in (6, 2, 9)]
    src = sources.InMemorySource([sources.Block(o, x) for o, x in
                              zip(np.cumsum([0, 6, 2]), blocks)])

    class Fake:  # a host backend and a "device" one, same engine
        returns_distance = False
        overlap = False
        stats = None
        last_wait = (None, None)
        last_build = None

        def __init__(self, on_device):
            self.applies_offset = on_device
            self.seen = []

        def alloc_peak_reset(self):
            pass

        def alloc_peak(self):
            return ""

        def search(self, xb, ahead=None, offset=0):
            assert offset == 0 or self.applies_offset
            self.seen.append(offset)
            D, I = faiss.knn(xq, xb, 4, metric=faiss.METRIC_INNER_PRODUCT)
            if self.applies_offset:
                I = b._globalize(torch.from_numpy(I), offset).numpy()
            return D, I

    outs = []
    for on_device in (False, True):
        f = Fake(on_device)
        D, I, *_ = loop.run_pass(src, f, 5, 4)
        outs.append((D, I))
        assert f.seen == ([0, 6, 8] if on_device else [0, 0, 0])
    np.testing.assert_array_equal(outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])
    want = faiss.knn(xq, np.concatenate(blocks), 4, metric=faiss.METRIC_INNER_PRODUCT)[1]
    np.testing.assert_array_equal(np.sort(outs[1][1], 1), np.sort(want, 1))
