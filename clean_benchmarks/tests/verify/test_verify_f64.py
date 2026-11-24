"""The float64 --verify references (dense: verify.oneshot_reference; filtered:
verify.filtered_oneshot_reference; both verify.exact_topk_f64)
against a naive float64 numpy brute force, a constructed case whose float32
and float64 rankings differ, and the float32 `--backend masked` ENGINE
(masked_topk), which must be unchanged."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from bench.data.filters import duckdb_filter as dfm
from bench.data.filters import search as fs
from bench.systems import guards
from bench.verify import oracle as verify
from bench.data.filters.reader import FilterBlock
from bench.data.sources import Block


def naive(xq, xb, k, metric="ip", allowed=None):
    """float64 brute force, best first, ties to the lowest id; (finfo(f32).min,
    -1) past the allowed rows."""
    q = np.asarray(xq, np.float64)
    x = np.asarray(xb, np.float64)
    if metric == "cosine":
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        q = np.divide(q, qn, out=np.zeros_like(q), where=qn > 0)
        x = np.divide(x, xn, out=np.zeros_like(x), where=xn > 0)
    S = q @ x.T
    if allowed is not None:
        S = np.where(allowed, S, -np.inf)
    D = np.full((len(q), k), float(np.finfo(np.float32).min), np.float64)
    I = np.full((len(q), k), -1, np.int64)
    for i in range(len(q)):
        o = [j for j in np.lexsort((np.arange(len(x)), -S[i])) if S[i, j] > -np.inf][:k]
        D[i, :len(o)] = S[i, o]
        I[i, :len(o)] = o
    return D, I


def _blocks(rng, sizes, d, dtype, ints):
    xs = [(rng.integers(-2, 3, (n, d)) if ints else rng.standard_normal((n, d))).astype(dtype)
          for n in sizes]
    out, off = [], 0
    for x in xs:
        out.append(Block(off, x))
        off += len(x)
    return out, np.concatenate(xs)


SIZES = [(50, 7, 30), (3,), (1, 2, 1, 9), (64, 64), (130,)]


@pytest.mark.parametrize("sizes", SIZES, ids=str)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_dense_reference_matches_naive_float64(sizes, dtype, metric):
    """Several block layouts (uneven, one smaller than k + 1, single-row
    blocks), fp16- and fp32-stored, k + 1 columns: equal to the naive float64
    brute force. Integer-valued ip data has many EXACT ties, which must go to
    the lowest id."""
    rng = np.random.default_rng(len(sizes) * 7 + (dtype == np.float16))
    d, k = 6, 5
    ints = metric == "ip"
    blocks, full = _blocks(rng, sizes, d, dtype, ints)
    xq = (rng.integers(-2, 3, (9, d)) if ints else rng.standard_normal((9, d))).astype(np.float32)
    D, I, xq_ref, xb = verify.oneshot_reference(xq, blocks, k + 1, metric)
    Dn, In = naive(xq, full, k + 1, metric)
    assert D.dtype == np.float64 and D.shape == (9, k + 1)
    np.testing.assert_array_equal(I, In)
    np.testing.assert_allclose(D, Dn, rtol=1e-13, atol=0)
    # small explicit tiles: the same answer
    D2, I2 = verify.exact_topk_f64(xq_ref, [b.vectors for b in blocks], k + 1,
                               unit=metric == "cosine", q_chunk=2, r_tile=3)
    np.testing.assert_array_equal(I2, np.where(In >= 0, In, -1))
    # the rescoring rows: stored values in float64 (unit for cosine)
    want = full.astype(np.float64)
    if metric == "cosine":
        want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert len(xb) == len(full)
    for i in (0, len(full) // 2, len(full) - 1):
        np.testing.assert_allclose(xb[i], want[i], rtol=1e-15)


def test_dense_reference_pads_like_faiss():
    xq = np.ones((2, 3), np.float32)
    D, I, _, _ = verify.oneshot_reference(xq, [Block(0, np.ones((2, 3), np.float32))], 4)
    assert I[:, 2:].tolist() == [[-1, -1]] * 2
    assert (D[:, 2:] == np.finfo(np.float32).min).all() and (D[:, :2] == 3).all()


def _f32_f64_case():
    """Two rows whose dot products with q = (1, 1) are 2^24 + 0.25 (id 0) and
    2^24 + 0.5 (id 1), every value exact in float32: float32 rounds BOTH to
    2^24 (spacing 2 there; a two-term sum, so in any order), a tie that a
    float32 reference resolves to id 0; float64 ranks id 1 first."""
    big = float(2 ** 24)
    xb = np.array([[big, 0.25], [big, 0.5], [1.0, 1.0]], np.float32)
    xq = np.array([[1.0, 1.0]], np.float32)
    s32 = (xq @ xb.T)[0]
    assert s32[0] == s32[1] == big          # float32 cannot tell them apart
    return xq, xb


def test_dense_reference_is_really_float64():
    """The old fp32 faiss.knn oracle returned ids 0, 1 with equal scores 2^24;
    the float64 reference returns 1 before 0 with their exact scores."""
    xq, xb = _f32_f64_case()
    D, I, _, _ = verify.oneshot_reference(xq, [Block(0, xb)], 3)
    assert I[0].tolist() == [1, 0, 2]
    assert D[0, :2].tolist() == [2 ** 24 + 0.5, 2 ** 24 + 0.25]


def test_compare_results_rescores_a_stray_id_from_the_blocks():
    """An id outside the reference's k + 1 is rescored in float64 from its
    block (Float64Rows), not from a corpus copy."""
    rng = np.random.default_rng(4)
    blocks, full = _blocks(rng, (20, 13), 4, np.float32, False)
    xq = rng.standard_normal((1, 4)).astype(np.float32)
    D, I, xq_ref, xb = verify.oneshot_reference(xq, blocks, 4)       # k = 3, k + 1 = 4
    s = (xq.astype(np.float64) @ full.astype(np.float64).T)[0]
    stray = int(np.argsort(-s)[10])
    # a wrong engine answer: the stray id with its TRUE score in place of the 3rd
    De, Ie = D[:, :3].astype(np.float32).copy(), I[:, :3].copy()
    Ie[0, 2], De[0, 2] = stray, s[stray]
    probs = verify.compare_results(De, Ie, D, I, xq=xq_ref, xb=xb)
    assert probs and "cannot be checked" not in " ".join(probs)
    # the score it was rescored with is the float64 one
    assert verify._ref_score(0, stray, {}, xq_ref, xb) == pytest.approx(s[stray], rel=1e-15)


# ------------------------------------------------------------------ filtered

def _fblock(mask_rows, per_query=None):
    """A FilterBlock: static (mask_rows: (n,) bool) or per-query (per_query:
    (nq, n) bool, distinct masks deduplicated)."""
    if per_query is None:
        m = np.asarray(mask_rows, bool)[None, :]
        return FilterBlock("static", m.shape[1], np.packbits(m, axis=1, bitorder="little"),
                           m.sum(1))
    uniq, moq = np.unique(per_query, axis=0, return_inverse=True)
    return FilterBlock("per_query", per_query.shape[1],
                       np.packbits(uniq, axis=1, bitorder="little"), uniq.sum(1),
                       moq.reshape(-1).astype(np.int32))


@pytest.mark.parametrize("kind", ["static", "per_query"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_filtered_reference_matches_naive_float64(kind, dtype, metric):
    rng = np.random.default_rng(11)
    sizes, d, k, nq = (40, 3, 25, 1), 5, 6, 7
    ints = metric == "ip"
    blocks, full = _blocks(rng, sizes, d, dtype, ints)
    xq = (rng.integers(-2, 3, (nq, d)) if ints else rng.standard_normal((nq, d))).astype(
        np.float32)
    allow = rng.random((nq, len(full))) < 0.3
    if kind == "static":
        allow[:] = allow[0]
    fblocks, off = [], 0
    for b in blocks:
        n = len(b.vectors)
        a = allow[:, off:off + n]
        fblocks.append(NS(vectors=b.vectors, filter=_fblock(a[0]) if kind == "static"
                          else _fblock(None, a)))
        off += n
    D, I, xq_ref, xb, allowed = verify.filtered_oneshot_reference(
        xq, iter(fblocks), k + 1, metric, n_rows=len(full))
    Dn, In = naive(xq, full, k + 1, metric, allowed=allow)
    assert D.dtype == np.float64
    np.testing.assert_array_equal(I, In)
    np.testing.assert_allclose(D, Dn, rtol=1e-13, atol=0)
    assert allowed.disallowed(I) == []
    assert len(xb) == len(full)
    bad = np.where(allow, -1, np.arange(len(full)))[0]
    bad = int(bad[bad >= 0][0])
    assert allowed.disallowed(np.array([[bad]] + [[-1]] * (nq - 1))) == [(0, bad)]


@pytest.mark.parametrize("kind", ["static", "per_query"])
def test_filtered_reference_is_really_float64(kind):
    """The f32/f64 case with a mask that keeps both close rows: float64 order
    (id 1 before id 0); the masked-out row never appears."""
    xq, xb = _f32_f64_case()
    xq = np.repeat(xq, 2, axis=0)
    m = np.array([True, True, False])
    fb = _fblock(m) if kind == "static" else _fblock(None, np.stack([m, m]))
    D, I, _, _, _ = verify.filtered_oneshot_reference(
        xq, iter([NS(vectors=xb, filter=fb)]), 3, "ip", n_rows=3)
    assert I.tolist() == [[1, 0, -1]] * 2
    assert D[:, :2].tolist() == [[2 ** 24 + 0.5, 2 ** 24 + 0.25]] * 2
    assert (D[:, 2] == np.finfo(np.float32).min).all()


def test_masked_engine_stays_float32():
    """--backend masked (masked_topk) is an engine, not the oracle: float32
    scores, so it ties the two close rows (both 2^24) and keeps the lower id
    first, exactly as before; and on random data it equals a float32
    brute force (xq @ xb.T in float32, lexsort, lowest id)."""
    xq, xb = _f32_f64_case()
    D, I = fs.masked_topk(xq, xb, 3)
    assert D.dtype == np.float32 and I[0].tolist() == [0, 1, 2]
    assert D[0, :2].tolist() == [2 ** 24, 2 ** 24]
    rng = np.random.default_rng(2)
    xq = rng.standard_normal((11, 7)).astype(np.float32)
    xb = rng.integers(-2, 3, (300, 7)).astype(np.float32)
    m = rng.random((11, 300)) < 0.5
    D, I = fs.masked_topk(xq, xb, 9, lambda q0, q1, r0, r1: m[q0:q1, r0:r1], 4, 64)
    s = np.where(m, xq @ xb.T, -np.inf).astype(np.float32)
    assert D.dtype == np.float32
    for q in range(11):
        want = np.lexsort((np.arange(300), -s[q]))[:9]
        assert I[q].tolist() == want.tolist()
        np.testing.assert_allclose(D[q], s[q][want], rtol=0, atol=1e-5)


def test_oracle_scratch_is_bounded_and_charged():
    """exact_topk_f64's numpy allocations (tracemalloc) stay within what the
    guard charges for it (oracle_bytes: float64 queries, D / I, the tiles)."""
    import tracemalloc
    rng = np.random.default_rng(9)
    nq, d, k = 300, 64, 10
    xs = [rng.standard_normal((n, d)).astype(np.float16) for n in (5000, 3000)]
    xq = rng.standard_normal((nq, d)).astype(np.float32)
    tracemalloc.start()
    t0 = tracemalloc.get_traced_memory()[0]
    D, I, xq_ref, xb = verify.oneshot_reference(xq, [Block(0, xs[0]), Block(5000, xs[1])],
                                             k + 1, "cosine")
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    charged = guards.oracle_bytes(d, nq, k, 5000)
    assert peak <= charged, (peak, charged)
    assert peak > charged / 4, (peak, charged)   # an upper estimate, not a wild one
