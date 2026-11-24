"""--d2h-overlap: each block's results are copied back on a separate stream and
merged one block late, while the next block is searched.

On torch's CPU device every op is synchronous, so these tests pin what can be
pinned off-GPU: the call order (submit i+1 BEFORE complete i), results identical
to the serial path (the two pinned buffers alternate correctly: a block's
results are merged before its buffer is reused), the error path (a pending
block is completed, never abandoned), and the CLI contract. The stream /
copy-engine overlap itself is a GPU property, checked on the node.
"""
import faiss
import numpy as np
import pytest

from bench.systems import harness_cli as run
from bench.systems import loop
from bench.verify import oracle as verify
from bench.data.sources import InMemorySource

torch = pytest.importorskip("torch")
K = 5  # < the smallest real file (1 row is < k: -1 padding is exercised)


def _faiss_resident(monkeypatch, metric="ip"):
    from test_metric import _faiss_resident as mk
    return mk(monkeypatch, metric=metric)


def _pass(monkeypatch, corpus, queries, overlap, metric="ip", spy=None):
    b = _faiss_resident(monkeypatch, metric)
    b.setup(queries)
    b.configure_d2h("pinned", overlap=overlap)
    if spy is not None:
        sub, comp = b.submit, b.complete

        def submit(xb, *a, **kw):
            tok = sub(xb, *a, **kw)
            spy.append(("S", kw.get("offset", 0)))
            tok["_off"] = kw.get("offset", 0)
            return tok

        def complete(tok):
            spy.append(("C", tok["_off"]))
            return comp(tok)
        b.submit, b.complete = submit, complete
    src = InMemorySource.load(corpus[0])
    D, I, info = loop.run_pass(src, b, len(queries), K)
    return D, I, info, src


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_overlapped_results_equal_the_serial_path_and_the_oracle(monkeypatch, corpus,
                                                                 queries, metric):
    Ds, Is, _, _ = _pass(monkeypatch, corpus, queries, overlap=False, metric=metric)
    Do, Io, info, _ = _pass(monkeypatch, corpus, queries, overlap=True, metric=metric)
    np.testing.assert_array_equal(Io, Is)
    np.testing.assert_array_equal(Do, Ds)
    D_ref, I_ref = faiss.knn(queries, corpus[1], K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(Do, Io, D_ref, I_ref) == []
    assert info["merge_s"] >= 0 and info["engine_call_s"] > 0


def test_submit_next_before_completing_previous(monkeypatch, corpus, queries):
    """S0 S1 C0 S2 C1 ... S5 C4 C5: block i is completed (copy back waited
    for, merged) only after block i+1 has been submitted, and every block
    exactly once, in order."""
    spy = []
    _, _, _, src = _pass(monkeypatch, corpus, queries, overlap=True, spy=spy)
    offs = [blk.offset for blk in src.blocks]
    want = [("S", offs[0])]
    for i in range(1, len(offs)):
        want += [("S", offs[i]), ("C", offs[i - 1])]
    want.append(("C", offs[-1]))
    assert spy == want


def test_serial_path_is_unchanged(monkeypatch, corpus, queries):
    """Without --d2h-overlap every block is submitted and completed back to
    back (S0 C0 S1 C1 ...): the search() path the paper runs used."""
    spy = []
    _pass(monkeypatch, corpus, queries, overlap=False, spy=spy)
    assert [k for k, _ in spy] == ["S", "C"] * (len(spy) // 2)


def test_a_failing_block_still_completes_the_pending_one(monkeypatch, corpus, queries):
    """An error while block i+1 is submitted must not abandon block i's copy
    back: the finally completes it before anything is released."""
    b = _faiss_resident(monkeypatch)
    b.setup(queries)
    b.configure_d2h("pinned", overlap=True)
    sub, comp = b.submit, b.complete
    n_sub, completed = [0], []

    def submit(xb, *a, **kw):
        n_sub[0] += 1
        if n_sub[0] == 3:
            raise RuntimeError("engine died")
        tok = sub(xb, *a, **kw)
        tok["_n"] = n_sub[0]
        return tok

    def complete(tok):
        completed.append(tok["_n"])
        return comp(tok)
    b.submit, b.complete = submit, complete
    with pytest.raises(RuntimeError, match="engine died"):
        loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)
    assert completed == [1, 2]  # 1 merged normally, 2 completed by the finally


def test_cli_refuses_overlap_without_pinned_d2h(corpus, capsys):
    with pytest.raises(SystemExit):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--nq", "8", "--backend", "cpu",
                  "--d2h-overlap", "--csv", "/dev/null"])
    assert "--d2h-overlap needs --d2h pinned" in capsys.readouterr().err


def test_configure_refuses_overlap_with_pageable(monkeypatch):
    b = _faiss_resident(monkeypatch)
    with pytest.raises(ValueError, match="needs --d2h pinned"):
        b.configure_d2h("pageable", overlap=True)
    assert b.d2h_stream is None  # off CUDA there is no copy-back stream at all


@pytest.mark.parametrize("h2d_overlap", [False, True])
@pytest.mark.parametrize("engine_s", [0.0, 0.05])
def test_the_ring_never_grows_with_d2h_overlap(corpus, queries, h2d_overlap, engine_s):
    """Process readers + the shared-memory ring: the consumer holds the block
    whose copy back is pending, the one being searched and (h2d overlap) the
    one uploading ahead; extra_slots budgets exactly that, so the ring must
    never grow (a stale reference to the previous pending block would grow it)."""
    from bench.data.sources import ParquetSource
    from test_h2d import _gpu_backend
    b = _gpu_backend(overlap=h2d_overlap, pinned=True, engine_s=engine_s)
    b.setup(queries)
    b.configure_d2h("pinned", overlap=True)
    with ParquetSource(corpus[0], prefetch=2, workers=2,
                       extra_slots=int(h2d_overlap) + 1).start() as src:
        D, I, info = loop.run_pass(src, b, len(queries), K)
        assert src.ring.grown == 0
    D_ref, I_ref = faiss.knn(queries, corpus[1], K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []
    assert info["engine_gap_s"] == ""  # overlapped device spans: no gap figure
    # ...but the GPU's idle time is defined in pipe mode too
    busy = info["widen_s"] + info["search_s"]
    assert info["gpu_idle_s"] == pytest.approx(info["total_s"] - busy)


def test_pipe_refuses_a_backend_that_does_not_globalise(monkeypatch, corpus, queries):
    b = _faiss_resident(monkeypatch)
    b.setup(queries)
    b.configure_d2h("pinned", overlap=True)
    monkeypatch.setattr(type(b), "applies_offset", property(lambda self: False))
    with pytest.raises(ValueError, match="globalises ids"):
        loop.run_pass(InMemorySource.load(corpus[0]), b, len(queries), K)
