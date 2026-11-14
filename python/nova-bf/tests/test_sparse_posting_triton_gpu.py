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
