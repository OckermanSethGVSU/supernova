"""GpuBackend plumbing with faiss's GPU entry points mocked.

There is no GPU (and only faiss-cpu) on the dev box, so this checks the
harness's side only: StandardGpuResources configuration, queries uploaded
once in resident mode, the stored-dtype upload then cast on the "device"
(the device widen), tensors (not numpy) and matching dtypes handed to knn_gpu,
D/I brought back as numpy, and the guards. The
"device" is torch's CPU device, so h2d/widen timing falls back to host
timestamps. It does NOT exercise CUDA, CUDA-event timing, torch H2D, FAISS's
bfKnn, or stream synchronisation.

Mock order: GpuBackend.__init__ in resident mode imports
faiss.contrib.torch_utils (the harness, GpuBackend.__init__), whose FIRST import
in the process re-patches faiss.knn_gpu (the previous one is kept as
knn_gpu_numpy). A knn_gpu mock installed BEFORE that construction is
silently overridden, and only when no earlier test imported torch_utils, so
the tests pass or fail by order. Install knn_gpu mocks AFTER constructing the
backend.
"""

import faiss
import numpy as np
import pytest

from bench.systems import harness_cli as run
from bench.systems import loop
from bench.verify import oracle as verify
from bench.systems import faiss_engine
from bench.data.sources import InMemorySource

torch = pytest.importorskip("torch")
K = 10


class FakeRes:
    def __init__(self):
        self.calls = []

    def noTempMemory(self):
        self.calls.append(("noTempMemory",))

    def setTempMemory(self, n):
        self.calls.append(("setTempMemory", n))

    def setLogMemoryAllocations(self, on):
        self.calls.append(("log", on))


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(faiss, "StandardGpuResources", FakeRes, raising=False)


def test_no_gpu_build_is_a_clear_error(monkeypatch):
    monkeypatch.delattr(faiss, "StandardGpuResources", raising=False)
    with pytest.raises(RuntimeError, match="StandardGpuResources"):
        faiss_engine.GpuBackend(K)


@pytest.mark.parametrize("temp_mem,expect", [
    (None, []),
    (0, [("noTempMemory",)]),
    (1 << 28, [("setTempMemory", 1 << 28)]),
])
def test_resources_configured(fake_gpu, temp_mem, expect):
    b = faiss_engine.GpuBackend(K, temp_mem=temp_mem, device="cpu")
    assert b.res.calls == expect


def test_resident_path(fake_gpu, monkeypatch, corpus, queries):
    path, xb = corpus
    b = faiss_engine.GpuBackend(K, device="cpu")
    uploads = []
    real_to_dev = b._to_dev

    def counting_to_dev(x):
        uploads.append(x.shape)
        return real_to_dev(x)

    monkeypatch.setattr(b, "_to_dev", counting_to_dev)
    calls = []

    def fake_knn_gpu(res, xq, xb_t, k, D=None, I=None, metric=faiss.METRIC_L2, **kw):
        assert res is b.res and metric == faiss.METRIC_INNER_PRODUCT and not kw
        assert isinstance(xq, torch.Tensor) and isinstance(xb_t, torch.Tensor)
        assert xq.dtype == xb_t.dtype
        assert xq.is_contiguous() and xb_t.is_contiguous()
        calls.append(xb_t.shape[0])
        Dn, In = faiss.knn(xq.numpy().astype(np.float32),
                           xb_t.numpy().astype(np.float32), k,
                           metric=faiss.METRIC_INNER_PRODUCT)
        return torch.from_numpy(Dn), torch.from_numpy(In)

    monkeypatch.setattr(faiss, "knn_gpu", fake_knn_gpu)
    setup_s = b.setup(queries)
    assert setup_s > 0 and len(uploads) == 1  # queries uploaded once
    src = InMemorySource.load(path)
    D, I, info = loop.run_pass(src, b, len(queries), K)
    assert len(uploads) == 1 + len(src.blocks)  # + one block H2D per file
    assert calls == [blk.vectors.shape[0] for blk in src.blocks]
    D_ref, I_ref = faiss.knn(queries, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []


def test_resident_every_op_runs_on_the_explicit_stream(fake_gpu, monkeypatch, corpus,
                                                       queries):
    """Resident FAISS runs exactly like CuvsBackend: setup()'s upload and each
    block's H2D / knn_gpu / events inside torch.cuda.stream(backend.stream),
    one context per call (torch_utils enqueues bfKnn on the current stream)."""
    from test_timing_detail import Clock, fake_alloc_counters, fake_event_cls

    class Stream:
        cuda_stream = 0xbeef

    stream = Stream()
    active, entered = [], []

    class StreamCtx:
        def __init__(self, s):
            assert s is stream
            entered.append(s)

        def __enter__(self):
            active.append(1)

        def __exit__(self, *exc):
            active.pop()

    monkeypatch.setattr(faiss_engine.GpuBackend, "_make_stream", lambda self: stream)
    b = faiss_engine.GpuBackend(K, device="cpu")
    assert b.stream is stream
    monkeypatch.setattr(torch.cuda, "stream", StreamCtx)
    real_to_dev = b._to_dev

    def to_dev(x):
        assert active, "H2D off the explicit stream"
        return real_to_dev(x)

    def fake_knn_gpu(res, xq, xb_t, k, metric=faiss.METRIC_L2, **kw):
        assert active, "knn_gpu off the explicit stream"
        Dn, In = faiss.knn(xq.numpy(), xb_t.numpy(), k, metric=metric)
        return torch.from_numpy(Dn), torch.from_numpy(In)

    monkeypatch.setattr(b, "_to_dev", to_dev)
    monkeypatch.setattr(faiss, "knn_gpu", fake_knn_gpu)
    b.setup(queries)
    clock = Clock()
    b._cuda = True  # events (faked) from here on
    fake_alloc_counters(monkeypatch, torch, clock)

    class CheckedEvent(fake_event_cls(clock)):
        def record(self):
            assert active, "event recorded off the explicit stream"
            super().record()

    monkeypatch.setattr(torch.cuda, "Event", CheckedEvent)
    src = InMemorySource.load(corpus[0])
    D, I, _ = loop.run_pass(src, b, len(queries), K)
    assert len(entered) == 1 + len(src.blocks) and not active
    D_ref, I_ref = faiss.knn(queries, corpus[1], K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []


GiB = 1 << 30


def _fake_knn(log):
    def fake_knn_gpu(res, xq, xb_t, k, metric=None, **kw):
        assert xq.is_contiguous() and xb_t.is_contiguous()
        log.append(("knn", xq.dtype, xb_t.dtype))
        Dn, In = faiss.knn(xq.numpy().astype(np.float32),
                           xb_t.numpy().astype(np.float32), k,
                           metric=faiss.METRIC_INNER_PRODUCT)
        return torch.from_numpy(Dn), torch.from_numpy(In)
    return fake_knn_gpu


def _logging_backend(monkeypatch, log, dtype=np.float32):
    b = faiss_engine.GpuBackend(K, dtype=dtype, device="cpu")
    real = b._to_dev

    def to_dev(x):
        log.append(("h2d", x.dtype))
        return real(x)

    monkeypatch.setattr(b, "_to_dev", to_dev)
    monkeypatch.setattr(faiss, "knn_gpu", _fake_knn(log))
    return b


@pytest.mark.parametrize("regime", ["memory", "stream"])
def test_device_widen_uploads_stored_dtype_then_casts(fake_gpu, monkeypatch, tmp_path,
                                                      queries, regime):
    """--widen device: the fp16-STORED block crosses H2D as fp16, is cast on
    the device, and knn_gpu sees fp32 xb with fp32 xq. Results are identical
    to the fp32-stored copy of the same values (widening is exact)."""
    from test_run import _fp16_pair
    from bench.data.sources import ParquetSource
    f16, f32, xb = _fp16_pair(tmp_path)
    q = np.ascontiguousarray(queries[:, :24] / np.linalg.norm(queries[:, :24], axis=1,
                                                               keepdims=True))
    out = {}
    for name, path in (("f16", f16), ("f32", f32)):
        log = []
        b = _logging_backend(monkeypatch, log)
        b.setup(q)
        assert log == [("h2d", np.float32)]  # queries: fp32, once
        log.clear()
        src = InMemorySource.load(path) if regime == "memory" else \
            ParquetSource(path, prefetch=2, workers=2)
        stored = np.float16 if name == "f16" else np.float32
        if regime == "memory":  # host RAM holds the STORED dtype
            assert {blk.vectors.dtype for blk in src.blocks} == {np.dtype(stored)}
        out[name] = loop.run_pass(src, b, len(q), K)
        assert log == [("h2d", np.dtype(stored)), ("knn", torch.float32, torch.float32)] * 3
        info = out[name][2]
        assert info["h2d_s"] > 0
        assert (info["widen_s"] > 0) == (name == "f16")  # no cast when stored f32
    np.testing.assert_array_equal(out["f16"][0], out["f32"][0])
    np.testing.assert_array_equal(out["f16"][1], out["f32"][1])
    D_ref, I_ref = faiss.knn(q, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(out["f16"][0], out["f16"][1], D_ref, I_ref) == []


def tmp_path_corpus(tmp_path):
    import make_dummy
    make_dummy.make_corpus(str(tmp_path / "c"), 2, 50, 32)
    return str(tmp_path / "c")


def test_main_gpu_device_widen_end_to_end(fake_gpu, monkeypatch, tmp_path, capsys):
    """run.main with the GPU backend on torch's CPU device: CSV records the
    device widen, h2d_s / widen_s are populated, --verify passes."""
    from test_run import _fp16_pair
    f16, _, _ = _fp16_pair(tmp_path, payload_bytes=300)

    def make_backend(args, dtype):
        b = faiss_engine.GpuBackend(args.k, dtype, device="cpu")
        monkeypatch.setattr(faiss, "knn_gpu", _fake_knn([]))  # after construction
        return b

    monkeypatch.setattr(run, "make_backend", make_backend)
    out = tmp_path / "r.csv"
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--prefetch", pf, "--corpus", f16, "--nq", "16", "--warmup", "0",
                  "--repeats", "1", "--csv", str(out), "--verify"])
    assert capsys.readouterr().out.count("verify OK") == 2
    import csv
    for r in csv.DictReader(open(out)):
        assert r["backend"] == "gpu" and r["queries_mode"] == "resident"
        assert r["widen_where"] == "device" and r["stored_dtype"] == "f16"
        assert float(r["h2d_s"]) > 0 and float(r["widen_s"]) > 0

