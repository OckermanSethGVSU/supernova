"""--cuvs-sync none (cuVS device merge without calc_truth's per-file resources.sync()), on the
CPU with the fake cuvs package and the numpy CuPy stand-in.

Covered: the top-k equals --cuvs-sync per-file and per_file_host (dense, sparse, every filtered
mode); no resources.sync() under none and one per file under per-file; at most one block in
flight (block i-1 is completed only after block i is submitted); with events off the host waits
on one ordering event per block, never a device sync; the ring needs no extra slot beyond the
one counted; the CLI refusals, the recorded column, the label; the guards' arithmetic.

What the mocks cannot show (GPU): real stream ordering, the overlap it buys (gpu_idle_s).
"""

import csv
import types

import numpy as np
import pytest

from bench.systems import harness_cli as run
from bench.systems import loop, cuvs_engine, guards
from bench.systems import base as systems
from bench.data.sources import InMemorySource, ParquetSource
from test_cuvs_backend import _cuvs_factory, fake_cuvs  # noqa: F401 (fixture)
from test_sparse import FakeRes, fake_knn, sparse_corpus  # noqa: F401 (fixture)
from test_cuvs_recipe import FILTER_CASES, filt_corpus  # noqa: F401 (fixture)
import fake_cupy

torch = pytest.importorskip("torch")
K = 10


def _backend(merge="device", sync="per-file", metric="ip", **kw):
    b = cuvs_engine.CuvsBackend(K, device="cpu", metric=metric, **kw)
    b.configure_merge(merge)
    if merge == "device":
        b.configure_sync(sync)
    return b


def _trace_calls(b, order):
    """Log submit / complete on the backend instance (search() goes through both)."""
    real_s, real_c = b.submit, b.complete
    n = {"s": 0, "c": 0}

    def submit(*a, **kw):
        tok = real_s(*a, **kw)
        tok["_i"] = n["s"]
        order.append(("s", n["s"]))
        n["s"] += 1
        return tok

    def complete(tok):
        order.append(("c", tok["_i"]))
        return real_c(tok)
    b.submit, b.complete = submit, complete


VARIANTS = (("host", None), ("device", "per-file"), ("device", "none"))


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_dense_none_equals_per_file_and_host(fake_cuvs, corpus, queries, metric):
    path, _ = corpus
    fake_cuvs.metric = "inner_product" if metric == "ip" else "cosine"
    out = {}
    for merge, sync in VARIANTS:
        b = _backend(merge, sync or "per-file", metric)
        b.setup(queries)
        s0 = getattr(fake_cuvs.capi, "syncs", 0)
        fake_cuvs.log.clear()
        D, I, _ = loop.run_pass(ParquetSource(path, prefetch=2, workers=2), b, len(queries), K)
        built = [n for op, _, n in fake_cuvs.log if op == "build"]
        out[(merge, sync)] = (D, I, getattr(fake_cuvs.capi, "syncs", 0) - s0, len(built))
    Dh, Ih, _, _ = out[("host", None)]
    for v in VARIANTS[1:]:
        D, I, _, _ = out[v]
        assert np.array_equal(Ih, I) and np.array_equal(Dh, D), v
    # resources.sync(): one per file under per-file (calc_truth), none under none
    _, _, syncs, files = out[("device", "per-file")]
    assert syncs == files > 1
    assert out[("device", "none")][2] == 0


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_sparse_none_equals_per_file_and_host(monkeypatch, metric, sparse_corpus):
    from bench.data.formats import sparse
    cdir, qp, parts, x = sparse_corpus
    cuvs = types.ModuleType("cuvs")
    cuvs.__version__ = "26.10.00-fake"
    common = types.ModuleType("cuvs.common")
    common.Resources = FakeRes
    cuvs.common = common
    import sys
    monkeypatch.setitem(sys.modules, "cuvs", cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.common", common)
    fake_cupy.install(monkeypatch)
    syncs = []
    monkeypatch.setattr(FakeRes, "sync", lambda self: syncs.append(1))
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    out = {}
    for merge, sync in VARIANTS:
        calls = []
        b = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=fake_knn(calls), metric=metric)
        b.configure_merge(merge)
        if merge == "device":
            b.configure_sync(sync)
        b.setup(q)
        del calls[:], syncs[:]
        src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
        D, I, _ = loop.run_pass(src, b, q.n_rows, K)
        out[(merge, sync)] = (D, I, len(syncs), len(calls))
    Dh, Ih, _, _ = out[("host", None)]
    for v in VARIANTS[1:]:
        np.testing.assert_array_equal(Ih, out[v][1])
        np.testing.assert_array_equal(Dh, out[v][0])
    assert out[("device", "per-file")][2] == out[("device", "per-file")][3] == len(parts)
    assert out[("device", "none")][2] == 0
    assert out[("host", None)][2] == 0


@pytest.mark.parametrize("sync", ["per-file", "none"])
def test_at_most_one_block_in_flight(fake_cuvs, corpus, queries, sync):
    """per-file: each block completed before the next is submitted. none: block i-1 completed
    right after block i is submitted, never later (one block in flight), the last at the end."""
    path, _ = corpus
    b = _backend(sync=sync)
    b.setup(queries)
    order = []
    _trace_calls(b, order)
    loop.run_pass(InMemorySource.load(path), b, len(queries), K)
    n = sum(1 for o in order if o[0] == "s")
    assert n > 2
    if sync == "per-file":
        want = [x for i in range(n) for x in (("s", i), ("c", i))]
    else:
        want = [("s", 0)] + [x for i in range(1, n) for x in (("s", i), ("c", i - 1))] \
            + [("c", n - 1)]
    assert order == want


class _OrderEvent:
    """torch.cuda.Event stand-in for events-off runs: counts synchronize()."""
    log = []

    def __init__(self, enable_timing=False):
        self.timing = enable_timing

    def record(self):
        pass

    def query(self):
        return True

    def synchronize(self):
        _OrderEvent.log.append("event.synchronize")


@pytest.mark.parametrize("sync", ["per-file", "none"])
def test_events_off_waits_on_one_event_per_block(fake_cuvs, corpus, queries, monkeypatch, sync):
    """--no-gpu-events with a (faked) CUDA device: none waits on one ordering-only event per
    block and makes no device sync during the pass; per-file syncs the device per block (as
    before) and records no event."""
    path, _ = corpus
    b = _backend(sync=sync, events=False)
    b.setup(queries)
    b._cuda = True
    _OrderEvent.log = []
    dev = []
    monkeypatch.setattr(torch.cuda, "Event", _OrderEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: dev.append(1))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    src = InMemorySource.load(path)
    D, I, info = loop.run_pass(src, b, len(queries), K)
    b._cuda = False
    n = len(src.blocks)
    if sync == "none":
        assert _OrderEvent.log == ["event.synchronize"] * n and dev == []
    else:
        assert _OrderEvent.log == [] and len(dev) == n
    b2 = _backend()
    b2.setup(queries)
    D2, I2, _ = loop.run_pass(InMemorySource.load(path), b2, len(queries), K)
    assert np.array_equal(I, I2) and np.array_equal(D, D2)


def test_events_on_spans_and_buckets(fake_cuvs, corpus, queries, monkeypatch):
    """With (faked) CUDA events under none: device_merge_s inside search_s, the host buckets
    tile total_s, engine_gap_s blank (pipelined), gpu_idle_s measured."""
    from test_timing_detail import Clock, fake_alloc_counters, fake_event_cls
    path, _ = corpus
    b = _backend(sync="none")
    b.setup(queries)
    clock = Clock()
    b._cuda = True
    monkeypatch.setattr(torch.cuda, "Event", fake_event_cls(clock))
    fake_alloc_counters(monkeypatch, torch, clock)
    D, I, info = loop.run_pass(InMemorySource.load(path), b, len(queries), K)
    b._cuda = False
    assert isinstance(info["device_merge_s"], float)
    assert 0 <= info["device_merge_s"] <= info["search_s"]
    parts = info["source_stall_s"] + info["engine_call_s"] + info["merge_s"] + info["other_s"]
    assert parts == pytest.approx(info["total_s"], abs=1e-9)
    assert info["engine_gap_s"] == "" and isinstance(info["gpu_idle_s"], float)


def test_ring_holds_the_block_in_flight(fake_cuvs, corpus, queries):
    """The one extra slot (overlap_extra_slots) covers the block in flight: the ring never
    grows."""
    path, _ = corpus
    b = _backend(sync="none")
    b.setup(queries)
    extra = guards.overlap_extra_slots(types.SimpleNamespace(cuvs_sync="none"))
    assert extra == 1
    src = ParquetSource(path, prefetch=2, workers=2, extra_slots=extra).start()
    try:
        loop.run_pass(src, b, len(queries), K)
        assert src.ring.grown == 0
    finally:
        src.close()


# ------------------------------------------------------------- the CLI

BASE = ["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "cuvs"]


def test_cli_defaults_and_refusals():
    assert run.parse_args(BASE).cuvs_sync == "per-file"
    assert run.parse_args(BASE + ["--cuvs-sync", "none"]).cuvs_sync == "none"
    assert run.parse_args(BASE + ["--cuvs-merge", "host"]).cuvs_sync is None
    for bad in (["--cuvs-merge", "host", "--cuvs-sync", "none"],
                ["--cuvs-merge", "host", "--cuvs-sync", "per-file"],
                ["--cuvs-sync", "never"]):
        with pytest.raises(SystemExit):
            run.parse_args(BASE + bad)
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y", "--backend", "gpu",
                        "--cuvs-sync", "none"])
    b = cuvs_engine.CuvsBackend.__new__(cuvs_engine.CuvsBackend)
    b.configure_merge("host", xp=fake_cupy.make())
    with pytest.raises(ValueError, match="--cuvs-merge device"):
        b.configure_sync("none")


@pytest.mark.slow
def test_cli_dense_rows_record_cuvs_sync(fake_cuvs, monkeypatch, tmp_path):
    """run.main with pinned + overlapped copies both ways: per-file, none and per_file_host
    give the same top-k; the CSV records cuvs_sync per-file / none / n/a."""
    from test_run import _fp16_pair
    from test_h2d import _pinner
    f16, _, _ = _fp16_pair(tmp_path)
    _cuvs_factory(monkeypatch)
    monkeypatch.setattr(cuvs_engine.base, "host_pinner", lambda backend: _pinner()[0])
    out = tmp_path / "r.csv"
    res = {}
    for name, extra in (("per-file", []), ("none", ["--cuvs-sync", "none"]),
                        ("host", ["--cuvs-merge", "host"])):
        run.main(["--regime", "stream", "--corpus", f16, "--nq", "8", "--warmup", "0",
                  "--repeats", "1", "--csv", str(out), "--backend", "cuvs", "--prefetch", "3",
                  "--workers", "2", "--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned",
                  "--d2h-overlap", "--verify", "--save-results", str(tmp_path / f"{name}.npz"),
                  *extra])
        res[name] = np.load(tmp_path / f"{name}.npz")
    rows = list(csv.DictReader(open(out)))
    assert [r["cuvs_sync"] for r in rows] == ["per-file", "none", "n/a"]
    assert [r["cuvs_merge"] for r in rows] == ["device:file", "device:file", "host"]
    assert all(r["verified"] == "ok" for r in rows)
    for n in ("none", "host"):
        assert np.array_equal(res["per-file"]["I"], res[n]["I"]), n
        assert np.array_equal(res["per-file"]["D"], res[n]["D"]), n


@pytest.mark.slow
@pytest.mark.parametrize("name,extra", FILTER_CASES)
def test_filtered_none_equals_per_file_and_host(fake_cuvs, monkeypatch, tmp_path, filt_corpus,
                                                name, extra):
    """Every filtered mode (bitmap static / per-query, union, files, gather): none gives the
    per-file and per_file_host top-k, with pinned + overlapped copies (masks and kept rows
    ride in the ring slots and are uploaded ahead); --verify passes."""
    import test_filtered_run as T
    from test_h2d import _pinner
    root, fpaths = filt_corpus
    _cuvs_factory(monkeypatch)
    monkeypatch.setattr(cuvs_engine.base, "host_pinner", lambda backend: _pinner()[0])
    if "cosine" in extra:
        fake_cuvs.metric = "cosine"
    out = {}
    for variant, ex in (("host", ["--cuvs-merge", "host"]), ("per-file", []),
                        ("none", ["--cuvs-sync", "none"])):
        res = tmp_path / f"{variant}.npz"
        csvp = tmp_path / f"{variant}.csv"
        run.main(["--regime", "stream", "--corpus", str(root / "corpus"), "--column",
                  "embedding", "--backend", "cuvs", "--query-file", str(root / "q.parquet"),
                  "--query-column", "embedding", "--nq", str(T.NQ), "--k", str(T.K),
                  "--filter-config", fpaths[name], "--filter-search", "f",
                  "--csv", str(csvp), "--save-results", str(res), "--verify", "--warmup", "0",
                  "--repeats", "1", "--prefetch", "3", "--workers", "2", "--h2d", "pinned",
                  "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap", *extra, *ex])
        out[variant] = np.load(res)
        r = list(csv.DictReader(open(csvp)))[-1]
        assert r["verified"] == "ok", variant
        assert r["cuvs_sync"] == {"host": "n/a"}.get(variant, variant)
    for v in ("per-file", "none"):
        assert np.array_equal(out["host"]["I"], out[v]["I"]), v
        assert np.array_equal(out["host"]["D"], out[v]["D"]), v


# ------------------------------------------------------------- knobs, labels, guards

def test_knob_label_and_attribution():
    spec = systems.SYSTEMS["cuvs"]
    assert spec.knobs["cuvs_sync"].flag == "--cuvs-sync" and spec.knobs["cuvs_sync"].sweepable
    for p, kn in spec.presets.items():
        assert "cuvs_sync" not in kn, p   # no preset turns it on
    assert systems.display_label("stock", {"cuvs_sync": "none"}) == \
        "stock (ours: no per-file sync)"
    assert systems.display_label("stock", {"cuvs_sync": "per-file"}) == "stock"
    assert systems.harness_knob_args(spec, {"cuvs_sync": "none"}, ["cuvs_sync"]) == \
        ["--cuvs-sync", "none"]
    from bench import leg as L
    row = {"total_s": 10.0, "source_stall_s": 1.0, "engine_call_s": 8.0, "merge_s": 0.5,
           "other_s": 0.5, "search_s": 9.0, "lib_s": 8.0, "ids_s": 0.1, "device_merge_s": 0.5,
           "engine_gap_s": "", "gpu_idle_s": 1.0, "widen_s": 0.0}
    for sync, label in (("per-file", L.HOST_LABELS["engine_call_s"]),
                        ("none", L.NOSYNC_ENGINE_CALL_LABEL)):
        a = L.attribution_harness(row, 10.0, 0.0, "cuvs", "on",
                                  {"cuvs_merge": "device:file", "cuvs_sync": sync})
        assert a["host_view"]["labels"]["engine_call_s"] == label
        assert a["host_view"]["additive"] and a["gpu_view"]["additive"]
        assert ("--cuvs-sync none" in a["host_view"]["formula"]) == (sync == "none")


def test_guards_count_the_block_in_flight(fake_cuvs, corpus, monkeypatch):
    ns = types.SimpleNamespace
    assert guards.overlap_extra_slots(ns(h2d_overlap=False, d2h_overlap=False)) == 0
    assert guards.overlap_extra_slots(ns(h2d_overlap=True, d2h_overlap=True)) == 2
    assert guards.overlap_extra_slots(ns(h2d_overlap=True, d2h_overlap=True,
                                         cuvs_sync="none")) == 2   # the same slot
    assert guards.overlap_extra_slots(ns(h2d_overlap=True, d2h_overlap=False,
                                         cuvs_sync="none")) == 2
    assert guards.overlap_extra_slots(ns(h2d_overlap=False, d2h_overlap=False,
                                         cuvs_sync="per-file")) == 0
    path, _ = corpus
    b = _backend()
    b._cuda = True
    src = ParquetSource(path, prefetch=0, workers=1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (0, 24 << 30))

    def args(**kw):
        a = dict(k=K, modality="dense", filter_config=None, h2d_overlap=False, widen="device",
                 backend="cuvs", cuvs_rmm_pool="off")
        a.update(kw)
        return ns(**a)
    for kw, blocks in (({"cuvs_sync": "none"}, 1), ({"h2d_overlap": True}, 2),
                       ({"h2d_overlap": True, "cuvs_sync": "per-file"}, 2),
                       ({"h2d_overlap": True, "cuvs_sync": "none"}, 3)):
        with pytest.raises(SystemExit, match=f"{blocks} block"):
            guards.cuvs_device_merge_guard(args(**kw), b, src, 64, np.float32)
