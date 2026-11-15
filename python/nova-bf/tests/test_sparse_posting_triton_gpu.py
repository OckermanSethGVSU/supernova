"""The sparse posting-list kernel on real hardware (skipped without CUDA).

The interpreted suite (`test_sparse_posting_triton_interpret.py`) covers the kernel's
logic; this file covers what only a GPU can: FMA contraction and the CTA
barrier between tokens, agreement with the cuSPARSE reference within the fp32
summation-order bound (cuSPARSE sums in a tree order, so bitwise agreement is
reported, not required), run-to-run determinism, and int64 output offsets
past 2^31 elements.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from nova_bf import sparse_posting_triton as sp  # noqa: E402

DEV = "cuda"


@pytest.fixture(autouse=True)
def _fresh():
    sp.reset()
    yield
    sp.reset()


def _zipf_csr(n, vocab, nnz_mean, g, signed=False, zeros=0.0):
    """Rows whose tokens follow a Zipf-like popularity (so a few tokens have
    posting lists spanning most of the slice, as in real sparse data)."""
    w = 1.0 / torch.arange(1, vocab + 1, dtype=torch.float64) ** 1.1
    k = torch.poisson(torch.full((n,), float(nnz_mean)), generator=g).clamp(1, vocab).long()
    crow, cols, vals = [0], [], []
    for i in range(n):
        c = torch.multinomial(w, int(k[i]), replacement=False, generator=g).sort().values
        v = torch.rand(len(c), generator=g) + 1e-3
        if signed:
            v *= torch.where(torch.rand(len(c), generator=g) < 0.5, -1.0, 1.0)
        if zeros:
            v = torch.where(torch.rand(len(c), generator=g) < zeros, 0.0, v)
        cols.append(c)
        vals.append(v.float())
        crow.append(crow[-1] + len(c))
    return torch.sparse_csr_tensor(
        torch.tensor(crow), torch.cat(cols), torch.cat(vals),
        size=(n, vocab)).to(DEV)


def _within_order_bound(got, ref, q, c, rows=None):
    """fp32 summation-order bound: |got - ref| <= (2n+2) eps (|q| . |c|) per
    cell with n shared tokens (the self-check's bound).
    cuSPARSE sums a row in a different (tree) order, so bitwise equality is
    not the contract; a wrong or dropped product exceeds this bound."""
    mag = sp.reference(q, c, rows, absolute=True)
    _, shared = sp.reference(q, c, rows, counts=True)
    live = ~torch.isneginf(ref)
    tol = (2 * shared + 2) * torch.finfo(torch.float32).eps * mag + 1e-30
    return bool((~((got - ref).abs() <= tol))[live].sum() == 0)


def _ulp_diff(a, b):
    ai = a.view(torch.int32).long()
    bi = b.view(torch.int32).long()
    ai = torch.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
    bi = torch.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
    return (ai - bi).abs()


@pytest.mark.parametrize("signed,zeros", [(False, 0.0), (True, 0.0), (True, 0.2)])
def test_matches_cusparse(signed, zeros, capsys):
    g = torch.Generator().manual_seed(0)
    q = _zipf_csr(1500, 4000, 8, g, signed, zeros)
    c = _zipf_csr(8192, 4000, 60, g, signed, zeros)
    got = sp.scores(q, c)
    ref = sp.reference(q, c)
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref))
    live = ~torch.isneginf(ref)
    ulp = _ulp_diff(got[live], ref[live])
    exact = float((ulp == 0).float().mean())
    with capsys.disabled():
        print(f"\n[signed={signed} zeros={zeros}] bit-exact {exact:.6f} of "
              f"{int(live.sum())} live cells, max ulp {int(ulp.max())}")
    assert _within_order_bound(got, ref, q, c)


def test_deterministic_run_to_run():
    g = torch.Generator().manual_seed(1)
    q = _zipf_csr(800, 3000, 8, g)
    c = _zipf_csr(8192, 3000, 80, g)
    a = sp.scores(q, c)
    b = sp.scores(q, c)
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_hot_tokens_span_many_posting_blocks():
    g = torch.Generator().manual_seed(2)
    q = _zipf_csr(300, 200, 6, g)                 # tiny vocab: every token is hot
    c = _zipf_csr(sp.BLOCK_P * 6 + 13, 200, 40, g)
    got, ref = sp.scores(q, c), sp.reference(q, c)
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref))
    assert _within_order_bound(got, ref, q, c)


@pytest.mark.parametrize("zero_gate_ok", [True, False])
@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_slice_score_posting_vs_cusparse(zero_gate_ok, metric):
    from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

    g = torch.Generator().manual_seed(3)
    q = _zipf_csr(700, 3000, 8, g, signed=not zero_gate_ok)
    c = _zipf_csr(4096, 3000, 70, g, signed=not zero_gate_ok)
    Q = q.to_dense()
    qn = Q.norm(dim=1)
    rn = (torch.rand(4096, generator=g) + 0.5).to(DEV)
    res = {}
    for kern in ("posting", "cusparse"):
        sp.reset()
        cache = _SparseQueryCache()
        cache.kernel = kern
        res[kern] = SparseBatchSlice(c, rn, zero_gate_ok, cache).score(Q, metric, qn)
    a, b = res["posting"], res["cusparse"]
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))
    mag = sp.reference(q, c, absolute=True)
    _, shared = sp.reference(q, c, counts=True)
    if metric == "cosine":
        mag = mag / rn.clamp_min(1e-12)[None, :] / qn[:, None]
    live = ~torch.isneginf(a)
    tol = (2 * shared + 4) * torch.finfo(torch.float32).eps * mag + 1e-30
    assert bool((~((a - b).abs() <= tol))[live].sum() == 0)


@pytest.mark.parametrize("doc_block", [0, 512])
def test_output_offsets_past_int32(doc_block, monkeypatch):
    monkeypatch.setattr(sp, "DOC_BLOCK", doc_block)
    free, _ = torch.cuda.mem_get_info()
    n_rows = 8192
    n_q = (2**31 // n_rows) + 300                 # last rows start past 2^31 elements
    if free < n_q * n_rows * 4 * 1.3:
        pytest.skip("not enough GPU memory for a >2^31-element output")
    g = torch.Generator().manual_seed(4)
    base = _zipf_csr(64, 2000, 6, g)
    reps = -(-n_q // 64)
    crow = base.crow_indices().cpu()
    nnz = int(crow[-1])
    crow_all = torch.cat([crow[:-1] + i * nnz for i in range(reps)] + [torch.tensor([reps * nnz])])
    q = torch.sparse_csr_tensor(crow_all[: n_q + 1].to(DEV),
                                base.col_indices().repeat(reps)[: int(crow_all[n_q])],
                                base.values().repeat(reps)[: int(crow_all[n_q])],
                                size=(n_q, 2000))
    c = _zipf_csr(n_rows, 2000, 50, g)
    got = sp.scores(q, c)
    tail = list(range(n_q - 64, n_q))
    ref = sp.reference(q, c, tail)
    sub = got[n_q - 64:]
    assert torch.equal(torch.isneginf(sub), torch.isneginf(ref))
    assert _within_order_bound(sub, ref, q, c, tail)
    del got
    torch.cuda.empty_cache()


def test_determinism_stress_hot_tokens_long_queries():
    """Intermittent races between successive tokens would show as run-to-run
    bit differences on hot, long-posting data."""
    g = torch.Generator().manual_seed(5)
    q = _zipf_csr(2000, 300, 30, g)              # long queries over a tiny vocab
    c = _zipf_csr(8192, 300, 60, g)              # every token posts to most rows
    first = sp.scores(q, c).view(torch.int32).clone()
    for _ in range(10):
        assert torch.equal(sp.scores(q, c).view(torch.int32), first)


@pytest.mark.parametrize("doc_block", [0, 512])
def test_row_reads_are_not_noncoherent_loads(doc_block, monkeypatch):
    monkeypatch.setattr(sp, "DOC_BLOCK", doc_block)
    """The row read-modify-write must not use the read-only (non-coherent)
    cache path; it is compiled with `.cg`."""
    g = torch.Generator().manual_seed(6)
    q = _zipf_csr(10, 100, 5, g)
    c = _zipf_csr(500, 100, 20, g)
    sp.scores(q, c)
    kern = sp._posting_scores_blocked if sp.DOC_BLOCK else sp._posting_scores
    caches = getattr(kern, "device_caches", None) or getattr(kern, "cache", None)
    ptx = []
    if isinstance(caches, dict):
        for v in caches.values():
            entries = v[0] if isinstance(v, tuple) else v
            for ck in (entries.values() if isinstance(entries, dict) else []):
                asm = getattr(ck, "asm", {})
                if "ptx" in asm:
                    ptx.append(asm["ptx"])
    if not ptx:
        pytest.skip("compiled PTX not reachable through this Triton version's cache API")
    text = "\n".join(ptx)
    assert "ld.global.cg" in text
    print("\nPTX loads:", sorted(set(l.split()[0] for l in text.splitlines() if l.strip().startswith("ld.global"))))


@pytest.mark.parametrize("doc_block", [16, 512, 1024, 2048])
def test_layouts_agree_bitwise(doc_block, monkeypatch):
    """Both kernel layouts sum each cell in the same order, so they must agree
    bit for bit with each other (not just within the cuSPARSE bound)."""
    g = torch.Generator().manual_seed(7)
    q = _zipf_csr(900, 3000, 9, g, signed=True)
    c = _zipf_csr(8192 - 37, 3000, 70, g, signed=True)
    monkeypatch.setattr(sp, "DOC_BLOCK", 0)
    base = sp.scores(q, c).view(torch.int32).clone()
    sp.reset()
    monkeypatch.setattr(sp, "DOC_BLOCK", doc_block)
    assert torch.equal(sp.scores(q, c).view(torch.int32), base)


def test_device_mismatch_declines():
    g = torch.Generator().manual_seed(8)
    q = _zipf_csr(5, 50, 3, g).cpu()
    c = _zipf_csr(20, 50, 5, g)
    assert "different devices" in sp.decline_reason(q, c)



def test_hot_path_on_gpu(monkeypatch):
    monkeypatch.setattr(sp, "HOT_TOKENS", 64)
    sp._HOT_PLAN.clear()
    g = torch.Generator().manual_seed(20)
    q = _zipf_csr(1500, 4000, 8, g)
    c = _zipf_csr(8192, 4000, 60, g)
    got = sp.scores(q, c, cmin=float(c.values().min()))
    assert sp.usage()["hot_launches"] == 1
    ref = sp.reference(q, c)
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref))
    assert _within_order_bound(got, ref, q, c)
    again = sp.scores(q, c, cmin=float(c.values().min()))
    assert torch.equal(got.view(torch.int32), again.view(torch.int32))


def test_threshold_select_identical_state_on_gpu(monkeypatch):
    """pack_topk + the merge fold with the select fast path on and off: the
    running state after the fold must be bit-identical, and some rows must
    really take the select path (n_ge <= k) while others take the full one."""
    from nova_bf import compute as cm
    from nova_bf import topk_triton as tt
    from nova_bf.tiebreak import pack, pack_topk

    g = torch.Generator().manual_seed(21)
    n_q, n_cols, k = 3000, 8192, 1000
    # ordinal != column, so a select path that writes ordinals is caught
    ordinal = torch.arange(n_cols, dtype=torch.int64, device=DEV) * 3 + 1000
    scores = torch.randn(n_q, n_cols, generator=g).to(DEV)
    srt = scores.sort(dim=1, descending=True).values
    # row i's state k-th sits at the slice's r_i-th best score, r_i in [1, 2k]
    r = (torch.arange(n_q, device=DEV) % (2 * k)) + 1
    kth = srt[torch.arange(n_q, device=DEV), r - 1]
    base = torch.linspace(5.0, 0.0, k, device=DEV)
    state = pack(kth[:, None] + base[None, :], torch.arange(10**6, 10**6 + k, device=DEV))
    enc0 = torch.zeros_like(state)
    results = []
    for env in ("", "1"):
        if env:
            monkeypatch.setenv("NOVA_BF_NO_TOPK_SELECT", "1")
        else:
            monkeypatch.delenv("NOVA_BF_NO_TOPK_SELECT", raising=False)
        tt.reset_usage()
        thr = state.min(dim=1).values.contiguous().clone()
        pk, pe, live = pack_topk(scores, ordinal, k, thr=thr)
        st, se = cm._merge_topk(state.clone(), enc0.clone(), [(pk, pe, live)], k, thr)
        results.append((st, se, live, tt.usage()["select_launches"]))
    assert results[0][3] == 1 and results[1][3] == 0      # select compiled only when on
    assert torch.equal(results[0][2], results[1][2])
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])
    n_ge = (scores >= kth[:, None]).sum(dim=1)
    assert bool(((n_ge > 0) & (n_ge <= k)).any()) and bool((n_ge > k).any())


def test_select_launch_counter_needs_a_threshold():
    from nova_bf import topk_triton as tt
    from nova_bf.tiebreak import pack_topk

    s_ = torch.randn(8, 4096, device=DEV)
    o = torch.arange(4096, dtype=torch.int64, device=DEV)
    tt.reset_usage()
    pack_topk(s_, o, 16)
    assert tt.usage()["select_launches"] == 0


def test_hot_path_subnormal_weights_are_present(monkeypatch):
    monkeypatch.setattr(sp, "HOT_TOKENS", 4)
    sp._HOT_PLAN.clear()
    q = torch.sparse_csr_tensor(torch.tensor([0, 1]), torch.tensor([0]),
                                torch.tensor([1.0]), size=(1, 2)).to(DEV)
    tiny = 1e-40                                       # float32 subnormal
    c = torch.sparse_csr_tensor(torch.tensor([0, 1, 2]), torch.tensor([0, 1]),
                                torch.tensor([tiny, 1.0]), size=(2, 2)).to(DEV)
    out = sp.scores(q, c, cmin=tiny)
    assert sp.usage()["hot_launches"] == 1
    assert out[0, 0].item() > 0.0 and torch.isneginf(out[0, 1])


def _assert_staged():
    from nova_bf import compute as cm
    ring = cm._sparse_staging(torch.device(DEV))
    assert isinstance(ring, cm._PinnedStagingRing) and any(ring._slots)


def test_pinned_staging_ring():
    import numpy as np

    from nova_bf import compute as cm

    ring = cm._PinnedStagingRing()
    arrs = [np.arange(n, dtype=np.int64) * 3 for n in (100, 5000, 70, 9000, 9000)]
    outs = [ring.upload({"a": a, "b": a.astype(np.float32)}, DEV) for a in arrs]
    torch.cuda.synchronize()
    for a, o in zip(arrs, outs):
        assert torch.equal(o["a"].cpu(), torch.from_numpy(a))
        assert torch.equal(o["b"].cpu(), torch.from_numpy(a.astype(np.float32)))


def test_staged_transfer_matches_pageable(monkeypatch):
    import numpy as np

    from nova_bf.compute import SparseCorpusBatch

    rng = np.random.default_rng(0)
    n = 3000
    counts = rng.integers(1, 20, size=n)
    offs = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    idx = np.concatenate([np.sort(rng.choice(500, size=c, replace=False)) for c in counts]).astype(np.int64)
    val = rng.random(int(offs[-1])).astype(np.float32) + 0.01
    import nova_bf.compute as _cm
    monkeypatch.setattr(_cm, "_SPARSE_STAGING", None)   # fresh ring: _assert_staged proves this test staged
    b = SparseCorpusBatch(offs, idx, val, None, np.arange(500), False)
    a = b.transfer(100, 2900, DEV).Cb
    _assert_staged()
    monkeypatch.setenv("NOVA_BF_NO_SPARSE_PINNED", "1")
    p = b.transfer(100, 2900, DEV).Cb
    assert a.shape == p.shape
    assert torch.equal(a.crow_indices(), p.crow_indices())
    assert torch.equal(a.col_indices(), p.col_indices())
    assert torch.equal(a.values(), p.values())


def test_staged_transfer_and_scoring_make_no_host_syncs(monkeypatch):
    import numpy as np

    from nova_bf.compute import SparseBatchSlice, SparseCorpusBatch, _SparseQueryCache

    rng = np.random.default_rng(1)
    n, vocab = 8192, 2000
    counts = rng.integers(5, 60, size=n)
    offs = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    idx = np.concatenate([np.sort(rng.choice(vocab, size=c, replace=False)) for c in counts]).astype(np.int64)
    val = rng.random(int(offs[-1])).astype(np.float32) + 0.01
    cache = _SparseQueryCache()
    cache.kernel = "posting"
    b = SparseCorpusBatch(offs, idx, val, None, np.arange(vocab), False,
                          zero_gate_ok=True, q_cache=cache)
    g = torch.Generator().manual_seed(22)
    Q = _zipf_csr(400, vocab, 8, g).to_dense()
    import nova_bf.compute as _cm
    monkeypatch.setattr(_cm, "_SPARSE_STAGING", None)   # fresh ring: _assert_staged proves this test staged
    b.transfer(0, n, DEV).score(Q, "dot")          # warm: query caches, compile, self-check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        sl = b.transfer(0, n, DEV)
        sl.score(Q, "dot")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _assert_staged()



def test_pinned_ring_never_overwrites_a_pending_copy():
    """Delay the stream, then upload A, B and C: C reuses A's slot and must
    wait for A's copy, so A's device data stays intact."""
    import numpy as np

    from nova_bf import compute as cm

    ring = cm._PinnedStagingRing()
    a = np.arange(4_000_000, dtype=np.int64)
    b_ = a * 2
    c = a * 3
    torch.cuda._sleep(int(1e9))
    da = ring.upload({"x": a}, DEV)["x"]
    ring.upload({"x": b_}, DEV)
    ring.upload({"x": c}, DEV)
    torch.cuda.synchronize()
    assert torch.equal(da.cpu(), torch.from_numpy(a))


def test_cutfill_register_report(capsys):
    """Not an assertion on speed: record registers/spills of every compiled
    `_cutfill` variant (the select branch sits near the register limit)."""
    import gc

    from nova_bf.tiebreak import pack_topk

    s_ = torch.randn(64, 8192, device=DEV)
    o = torch.arange(8192, dtype=torch.int64, device=DEV)
    thr = torch.full((64,), -(1 << 62), dtype=torch.int64, device=DEV)
    enc = torch.arange(8192, dtype=torch.int64, device=DEV)
    pack_topk(s_, o, 1000, thr=thr.clone(), encoded=enc)
    pack_topk(s_, o, 1000, thr=thr.clone())
    rows = []
    for obj in gc.get_objects():
        if type(obj).__name__ == "CompiledKernel" and "cutfill" in str(getattr(obj, "name", "")):
            rows.append((getattr(obj, "n_regs", None), getattr(obj, "n_spills", None)))
    with capsys.disabled():
        print("\n_cutfill compiled variants (n_regs, n_spills):", rows)
    assert rows, "no compiled _cutfill found"
