"""The sparse posting-list kernel, executed for real on CPU in the interpreter.

`nova_bf.sparse_posting_triton._posting_scores` only runs on a GPU in production.
`TRITON_INTERPRET=1` runs the real `@triton.jit` body as numpy, which covers
the posting walk, the per-token chunking past BLOCK_P, the untouched-cell
sentinel, the masks of partial blocks and the row initialisation. It does NOT
cover GPU numerics (FMA contraction, the CTA barrier), so score checks here
are "the right products were summed" within the fp32 summation-order bound
of a float64 reference; agreement with cuSPARSE on hardware is in
`test_sparse_posting_triton_gpu.py`.

As in `test_tile_rescore_interpret.py`: the variable must be set before
`triton` is imported, so the interpreted tests run in a SUBPROCESS.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

_CHILD = os.environ.get("NOVA_BF_SPARSE_POSTING_INTERPRET_CHILD") == "1"
_MIN_CHILD_TESTS = 171


def test_the_interpreted_suite_passes():
    """Re-run this file with `TRITON_INTERPRET=1` set before `import triton`."""
    if _CHILD:
        pytest.skip("this process IS the interpreted child")
    import interpret_child

    # In parallel on the whole machine: see tests/interpret_child.py.
    proc = interpret_child.run(__file__, "NOVA_BF_SPARSE_POSTING_INTERPRET_CHILD",
                               env_drop=("NOVA_BF_NO_SPARSE_POSTING",))
    assert proc.returncode == 0, (
        f"the interpreted sparse-posting suite failed:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"(\d+) passed", proc.stdout)
    assert m, f"the child reported no passing tests:\n{proc.stdout}"
    assert int(m.group(1)) >= _MIN_CHILD_TESTS, (
        f"the interpreted child ran only {m.group(1)} tests:\n{proc.stdout}")


# --- tests that run in the ordinary (non-interpreted) CPU process ------------

def test_config_knob_values():
    from pydantic import ValidationError

    from nova_bf.config import ParamsConfig

    assert ParamsConfig().sparse_kernel == "auto"
    for v in ("auto", "posting", "cusparse"):
        assert ParamsConfig(sparse_kernel=v).sparse_kernel == v
    with pytest.raises(ValidationError):
        ParamsConfig(sparse_kernel="triton")


@pytest.mark.skipif(_CHILD, reason="non-interpreted behaviour")
def test_cpu_slice_declines_and_auto_takes_cusparse():
    import torch

    from nova_bf import sparse_posting_triton as sp
    from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

    if os.environ.get("TRITON_INTERPRET"):
        pytest.skip("interpreter set in the parent environment")
    Q = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    Cb = torch.tensor([[1.0, 1.0, 0.0], [0.0, 3.0, 0.0]]).to_sparse_csr()
    cache = _SparseQueryCache()
    assert cache.kernel == "cusparse"          # class default; run_compute applies config
    cache.kernel = "auto"
    assert sp.decline_reason(cache.values(Q), Cb) == "not a CUDA slice"
    sl = SparseBatchSlice(Cb, None, False, cache)
    out = sl.score(Q, "dot")
    assert out[0, 0] == 1.0 and torch.isneginf(out[0, 1]) and torch.isneginf(out[1]).all()


@pytest.mark.skipif(_CHILD, reason="non-interpreted behaviour")
def test_posting_mode_fails_loudly_where_it_cannot_run():
    import torch

    from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

    if os.environ.get("TRITON_INTERPRET"):
        pytest.skip("interpreter set in the parent environment")
    Q = torch.tensor([[1.0, 0.0, 2.0]])
    Cb = torch.tensor([[1.0, 1.0, 0.0]]).to_sparse_csr()
    cache = _SparseQueryCache()
    cache.kernel = "posting"
    with pytest.raises(RuntimeError, match="posting-list kernel cannot run"):
        SparseBatchSlice(Cb, None, False, cache).score(Q, "dot")


def test_sample_rows_is_deterministic_and_bounded():
    from nova_bf.sparse_posting_triton import _sample_rows

    assert _sample_rows(0) == []
    assert _sample_rows(5) == [0, 1, 2, 3, 4]
    s = _sample_rows(100_000)
    assert s[0] == 0 and s[-1] == 99_999 and len(s) <= 256
    assert s == sorted(set(s)) == _sample_rows(100_000)


# --- interpreted child -------------------------------------------------------

if _CHILD:
    assert os.environ.get("TRITON_INTERPRET") == "1"

    import triton.runtime.jit as _jit

    _orig_normalize = _jit._normalize_ty

    def _normalize(ty):
        out = _orig_normalize(ty)
        if isinstance(out, str) and out.rsplit(".", 1)[-1] == "constexpr":
            return "constexpr"
        return out

    _jit._normalize_ty = _normalize

    import torch                                          # noqa: E402

    from nova_bf import sparse_posting_triton as sp              # noqa: E402

    # Both kernels: one program per row (0), a tiny block (16, so most slices
    # span many blocks and posting lists split across them), the default.
    LAYOUTS = [0, 16, 512]

    @pytest.fixture(autouse=True)
    def _fresh():
        sp.reset()
        sp.set_policy("auto")
        yield
        sp.reset()

    def _layout(monkeypatch, db):
        monkeypatch.setattr(sp, "DOC_BLOCK", db)

    def _rand_csr(n, vocab, nnz_lo, nnz_hi, g, signed, zeros=0.0, hot=()):
        """Random CSR rows with unique sorted columns. `hot` tokens appear in
        (almost) every row, to force posting lists longer than BLOCK_P."""
        crow, cols, vals = [0], [], []
        for _ in range(n):
            k = int(torch.randint(nnz_lo, nnz_hi + 1, (1,), generator=g))
            c = set(torch.randperm(vocab, generator=g)[:k].tolist())
            for h in hot:
                if torch.rand(1, generator=g) < 0.97:
                    c.add(h)
            c = sorted(c)
            v = torch.rand(len(c), generator=g) + 0.01
            if signed:
                v = v * torch.where(torch.rand(len(c), generator=g) < 0.5, -1.0, 1.0)
            if zeros:
                v = torch.where(torch.rand(len(c), generator=g) < zeros, 0.0, v)
            cols += c
            vals += v.tolist()
            crow.append(len(cols))
        return torch.sparse_csr_tensor(
            torch.tensor(crow, dtype=torch.int64), torch.tensor(cols, dtype=torch.int64),
            torch.tensor(vals, dtype=torch.float32), size=(n, vocab))

    def _csr(rows, vocab):
        """CSR from [(cols, vals)] rows."""
        crow, cols, vals = [0], [], []
        for c, v in rows:
            cols += c
            vals += v
            crow.append(len(cols))
        return torch.sparse_csr_tensor(torch.tensor(crow), torch.tensor(cols, dtype=torch.int64),
                                       torch.tensor(vals, dtype=torch.float32),
                                       size=(len(rows), vocab))

    def _ind(x):
        m = torch.zeros(x.shape, dtype=torch.float64)
        rows = torch.repeat_interleave(torch.arange(x.shape[0]), x.crow_indices().diff())
        m[rows, x.col_indices().long()] = 1
        return m

    def _check(out, q, c):
        """Mask exactly the structural rule; scores within the fp32 order bound
        of the float64 truth."""
        hit = (_ind(q) @ _ind(c).T) > 0
        s64 = q.to_dense().double() @ c.to_dense().double().T
        assert out.dtype == torch.float32 and tuple(out.shape) == tuple(hit.shape)
        assert torch.equal(~torch.isneginf(out), hit), "no-overlap mask differs"
        n = (_ind(q) @ _ind(c).T)[hit]
        bound = (n + 2) * torch.finfo(torch.float32).eps * (
            q.to_dense().abs().double() @ c.to_dense().abs().double().T)[hit] + 1e-30
        err = (out[hit].double() - s64[hit]).abs()
        assert (err <= bound).all(), float(err.max())

    def _bad_reference(monkeypatch, delta):
        """Patch `reference` so the plain scores are off by `delta` (absolute
        and counts calls pass through untouched)."""
        real = sp.reference

        def bad(*a, **k):
            r = real(*a, **k)
            if k.get("absolute"):
                return r
            if k.get("counts"):
                return r[0] + delta, r[1]
            return r + delta
        monkeypatch.setattr(sp, "reference", bad)

    # --- the kernel itself ----------------------------------------------------

    def test_both_kernels_are_the_ones_being_interpreted():
        from triton.runtime.interpreter import InterpretedFunction

        assert sp._posting_scores is not None, sp._UNAVAILABLE
        assert isinstance(sp._posting_scores, InterpretedFunction)
        assert isinstance(sp._posting_scores_blocked, InterpretedFunction)

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("seed", [0, 1])
    @pytest.mark.parametrize("signed", [False, True])
    def test_random_matches_float64_truth(seed, signed, layout, monkeypatch):
        _layout(monkeypatch, layout)
        g = torch.Generator().manual_seed(seed)
        q = _rand_csr(37, 60, 0, 9, g, signed)
        c = _rand_csr(130, 60, 0, 25, g, signed)
        _check(sp.scores(q, c), q, c)

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_posting_lists_longer_than_a_block(layout, monkeypatch):
        _layout(monkeypatch, layout)
        g = torch.Generator().manual_seed(3)
        n_rows = 1024 * 2 + 77                   # hot tokens span several BLOCK_P chunks
        q = _rand_csr(9, 40, 1, 5, g, False, hot=(7,))
        c = _rand_csr(n_rows, 40, 1, 6, g, False, hot=(7, 11))
        _check(sp.scores(q, c), q, c)

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_row_init_covers_partial_and_multiple_init_blocks(layout, monkeypatch):
        _layout(monkeypatch, layout)
        g = torch.Generator().manual_seed(4)
        n_rows = 2 * 512 + 5                     # not a multiple of any block size
        q = _rand_csr(3, 50, 0, 2, g, False)    # mostly-empty queries: most cells untouched
        c = _rand_csr(n_rows, 50, 0, 1, g, False)
        out = sp.scores(q, c)
        _check(out, q, c)
        assert torch.isneginf(out).float().mean() > 0.5

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_summation_is_ascending_token_order(layout, monkeypatch):
        # Terms 1e8, -1e8, 1 in token order 0, 1, 2: ascending gives exactly 1,
        # any other order loses the 1 to rounding (1 - 1e8 == -1e8 in fp32).
        _layout(monkeypatch, layout)
        q = _csr([([0, 1, 2], [1e8, -1e8, 1.0])], 3)
        c = _csr([([0, 1, 2], [1.0, 1.0, 1.0])], 3)
        assert sp.scores(q, c).tolist() == [[1.0]]

    def test_init_stays_inside_the_programs_block(monkeypatch):
        # DOC_BLOCK 768 with BLOCK_I 512: the init loop's second step reaches
        # past the block. It must not overwrite the next block's cells.
        monkeypatch.setattr(sp, "DOC_BLOCK", 768)
        g = torch.Generator().manual_seed(23)
        q = _rand_csr(4, 30, 3, 6, g, False)
        c = _rand_csr(768 * 2 + 40, 30, 3, 8, g, False)
        _check(sp.scores(q, c), q, c)
        # and launching only block 0 leaves block 1 untouched
        ptr, prow, pval = sp.corpus_postings(c)
        bp, nblk = sp.block_ptr(ptr, prow, 30, int(c.shape[0]), 768)
        out = torch.full((4, int(c.shape[0])), 7.0)
        sp._posting_scores_blocked[(4, 1)](
            q.crow_indices(), q.col_indices(), q.values(), bp, prow, pval, out,
            int(c.shape[0]), out.stride(0), nblk, DB=768, BLOCK_P=256, BLOCK_I=512)
        assert (out[:, 768:] == 7.0).all()

    def test_stored_zeros_are_candidates():
        # A shared dimension whose weight is a stored 0.0 still makes the pair
        # a candidate scoring 0.0 (the structural rule), never -inf.
        q = _csr([([0, 1], [0.0, 1.0])], 3)
        c = _csr([([0], [5.0]), ([1], [0.0]), ([2], [2.0])], 3)
        assert sp.scores(q, c).tolist() == [[0.0, 0.0, float("-inf")]]

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_random_with_stored_zeros(layout, monkeypatch):
        _layout(monkeypatch, layout)
        g = torch.Generator().manual_seed(5)
        q = _rand_csr(20, 30, 1, 8, g, True, zeros=0.3)
        c = _rand_csr(90, 30, 1, 12, g, True, zeros=0.3)
        _check(sp.scores(q, c), q, c)

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_empty_shapes(layout, monkeypatch):
        _layout(monkeypatch, layout)
        g = torch.Generator().manual_seed(6)
        q = _rand_csr(0, 10, 1, 3, g, False)
        c = _rand_csr(5, 10, 1, 3, g, False)
        assert sp.scores(q, c).shape == (0, 5)
        q = _rand_csr(4, 10, 1, 3, g, False)
        c = _rand_csr(0, 10, 1, 3, g, False)
        assert sp.scores(q, c).shape == (4, 0)

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_int32_indices_both_sides(layout, monkeypatch):
        _layout(monkeypatch, layout)
        g = torch.Generator().manual_seed(7)
        q64 = _rand_csr(10, 40, 1, 6, g, False)
        c64 = _rand_csr(50, 40, 1, 9, g, False)
        q = torch.sparse_csr_tensor(q64.crow_indices().int(), q64.col_indices().int(),
                                    q64.values(), size=tuple(q64.shape))
        c = torch.sparse_csr_tensor(c64.crow_indices().int(), c64.col_indices().int(),
                                    c64.values(), size=tuple(c64.shape))
        _check(sp.scores(q, c), q64, c64)

    def test_postings_are_token_grouped_and_row_sorted():
        g = torch.Generator().manual_seed(8)
        c = _rand_csr(70, 25, 0, 10, g, True)
        ptr, rows, vals = sp.corpus_postings(c)
        d = c.to_dense()
        for t in range(25):
            r = rows[ptr[t]:ptr[t + 1]].long()
            assert torch.equal(r, torch.sort(r).values)
            exp = (c.col_indices() == t)
            exp_rows = torch.repeat_interleave(torch.arange(70), c.crow_indices().diff())[exp]
            assert torch.equal(r, exp_rows)
            assert torch.equal(vals[ptr[t]:ptr[t + 1]], d[r, t])

    @pytest.mark.parametrize("db", [16, 64, 512])
    def test_block_ptr_bounds_each_token_block(db):
        g = torch.Generator().manual_seed(24)
        n_rows = 300
        c = _rand_csr(n_rows, 25, 0, 10, g, False)
        ptr, rows, _ = sp.corpus_postings(c)
        bp, nblk = sp.block_ptr(ptr, rows, 25, n_rows, db)
        assert nblk == -(-n_rows // db)
        for t in range(25):
            for b in range(nblk):
                seg = rows[bp[t * nblk + b]:bp[t * nblk + b + 1]].long()
                exp = rows[ptr[t]:ptr[t + 1]].long()
                exp = exp[(exp // db) == b]
                assert torch.equal(seg, exp)

    # --- declines ---------------------------------------------------------------

    def test_env_kill_switch(monkeypatch):
        g = torch.Generator().manual_seed(11)
        q = _rand_csr(3, 10, 1, 3, g, False)
        c = _rand_csr(4, 10, 1, 3, g, False)
        monkeypatch.setenv("NOVA_BF_NO_SPARSE_POSTING", "1")
        assert sp.decline_reason(q, c) == "NOVA_BF_NO_SPARSE_POSTING"

    def test_declines_non_float32_and_vocab_mismatch():
        g = torch.Generator().manual_seed(12)
        q = _rand_csr(3, 10, 1, 3, g, False)
        c = _rand_csr(4, 10, 1, 3, g, False)
        cd = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(),
                                     c.values().double(), size=tuple(c.shape))
        assert "float32" in sp.decline_reason(q, cd)
        c11 = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(), c.values(),
                                      size=(4, 11))
        assert "vocabularies" in sp.decline_reason(q, c11)
        assert sp.usage()["declines"] == {"values are not float32": 1,
                                          "query and corpus vocabularies differ": 1}

    def test_declines_four_byte_non_float_values():
        """itemsize == 4 is not float32: int32 (both sides) must decline."""
        g = torch.Generator().manual_seed(13)
        q = _rand_csr(3, 10, 1, 3, g, False)
        c = _rand_csr(4, 10, 1, 3, g, False)
        qi = torch.sparse_csr_tensor(q.crow_indices(), q.col_indices(),
                                     q.values().to(torch.int32), size=tuple(q.shape))
        ci = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(),
                                     c.values().to(torch.int32), size=tuple(c.shape))
        assert sp.decline_reason(qi, ci) == "values are not float32"

    def test_skip_thr_requires_return_skip():
        g = torch.Generator().manual_seed(14)
        q = _rand_csr(3, 10, 1, 3, g, False)
        c = _rand_csr(4, 10, 1, 3, g, False)
        with pytest.raises(ValueError, match="return_skip"):
            sp.scores(q, c, skip_thr=torch.zeros(3, dtype=torch.int64))

    def test_overflow_scale_declines():
        # q = {t0: 1e20, t1: 1}, doc = {t0: -1e20, t1: 2}: the first partial sum
        # overflows to -inf, which would read as "untouched". Must decline.
        q = _csr([([0, 1], [1e20, 1.0])], 2)
        c = _csr([([0, 1], [-1e20, 2.0])], 2)
        assert "overflow" in sp.decline_reason(q, c)

    def test_overflow_bound_uses_the_row_l1_norm():
        # 10 tokens x 1e17 (L1 1e18) against 5e18: L1 bound 5e36 >= 1e36 must
        # decline; a max-|q| bound (5e35) would wrongly accept.
        q = _csr([(list(range(10)), [1e17] * 10)], 10)
        c = _csr([([0], [5e18])], 10)
        assert "overflow" in sp.decline_reason(q, c)
        c = _csr([([0], [5e17])], 10)
        assert sp.decline_reason(q, c) is None

    @pytest.mark.parametrize("side,val", [("q", float("inf")), ("q", float("nan")),
                                          ("c", float("-inf")), ("c", float("nan"))])
    def test_non_finite_weights_decline(side, val):
        good = _csr([([0, 1], [1.0, 2.0])], 2)
        bad = _csr([([0, 1], [1.0, val])], 2)
        q, c = (bad, good) if side == "q" else (good, bad)
        assert "non-finite" in sp.decline_reason(q, c)

    def test_query_bound_cache_holds_no_tensor():
        import gc
        import weakref

        g = torch.Generator().manual_seed(25)
        q = _rand_csr(5, 10, 1, 3, g, False)
        c = _rand_csr(5, 10, 1, 3, g, False)
        assert sp.decline_reason(q, c) is None
        ref = weakref.ref(q)
        del q
        gc.collect()
        assert ref() is None

    # --- the self-check ----------------------------------------------------------

    def test_selfcheck_runs_once_per_configuration():
        g = torch.Generator().manual_seed(9)
        q = _rand_csr(12, 30, 1, 5, g, False)
        c = _rand_csr(40, 30, 1, 8, g, False)
        sp.scores(q, c)
        sp.scores(q, c)
        u = sp.usage()
        assert u["launches"] == 2 and u["selfchecks"] == 1 and u["selfcheck_failures"] == 0

    def test_tail_slice_with_another_block_count_is_checked_again():
        g = torch.Generator().manual_seed(26)
        q = _rand_csr(12, 30, 1, 5, g, False)
        sp.scores(q, _rand_csr(512 * 16, 30, 1, 3, g, False))   # nblk 16
        sp.scores(q, _rand_csr(512 * 16, 30, 1, 3, g, False))   # same key
        # 512 rows: n_rows and the row stride specialise exactly as 8192 does
        # (divisible by 16, not 1), only nblk changes (16 -> 1): new compile.
        sp.scores(q, _rand_csr(512, 30, 1, 3, g, False))
        assert sp.usage()["selfchecks"] == 2

    def test_selfcheck_failure_disables_and_raises(monkeypatch):
        g = torch.Generator().manual_seed(10)
        q = _rand_csr(12, 30, 1, 5, g, False)
        c = _rand_csr(40, 30, 1, 8, g, False)
        _bad_reference(monkeypatch, 1.0)
        with pytest.raises(sp.KernelMismatch):
            sp.scores(q, c)
        assert sp.kernel_off() is not None and sp.usage()["selfcheck_failures"] == 1
        assert sp.decline_reason(q, c) is not None

    def test_selfcheck_catches_nan():
        g = torch.Generator().manual_seed(20)
        q = _rand_csr(12, 30, 2, 5, g, False)
        c = _rand_csr(40, 30, 3, 8, g, False)
        out = sp.scores(q, c)
        r, k = map(int, (~torch.isneginf(out)).nonzero()[0])
        out[r, k] = float("nan")
        with pytest.raises(sp.KernelMismatch):
            sp._self_check(q, c, out)

    @pytest.mark.parametrize("flip", ["live_to_dead", "dead_to_live"])
    def test_selfcheck_catches_mask_errors(flip):
        g = torch.Generator().manual_seed(27)
        q = _rand_csr(12, 30, 2, 5, g, False)
        c = _rand_csr(60, 30, 1, 4, g, False)
        out = sp.scores(q, c)
        dead = torch.isneginf(out)
        r, k = map(int, ((~dead) if flip == "live_to_dead" else dead).nonzero()[0])
        out[r, k] = float("-inf") if flip == "live_to_dead" else 0.0
        with pytest.raises(sp.KernelMismatch):
            sp._self_check(q, c, out)

    def test_selfcheck_tolerance_edges():
        # Half the bound passes; three times the bound on a cell with >= 3
        # shared tokens fails -- so the tolerance is neither loose nor
        # missing its dependence on the shared count.
        g = torch.Generator().manual_seed(21)
        q = _rand_csr(20, 30, 6, 6, g, True)
        c = _rand_csr(60, 30, 10, 10, g, True)
        rows = list(range(20))
        ref, shared = sp.reference(q, c, rows, counts=True)
        mag = sp.reference(q, c, rows, absolute=True)
        live = ~torch.isneginf(ref)
        tol = (2 * shared + 2) * torch.finfo(torch.float32).eps * mag
        sp._self_check(q, c, torch.where(live, ref + 0.5 * tol, ref))
        r, k = map(int, (live & (shared >= 3)).nonzero()[0])
        bad = ref.clone()
        bad[r, k] += 3 * float(tol[r, k])
        with pytest.raises(sp.KernelMismatch):
            sp._self_check(q, c, bad)

    def test_selfcheck_samples_beyond_the_first_rows():
        # 300 queries > 256 sampled: the sample includes the last row, so an
        # error there is caught; and `_select_rows` picks the right rows.
        g = torch.Generator().manual_seed(28)
        q = _rand_csr(300, 30, 2, 5, g, False)
        c = _rand_csr(50, 30, 2, 6, g, False)
        rows = sp._sample_rows(300)
        assert len(rows) == 256 and rows[-1] == 299 and rows != list(range(256))
        full = sp.reference(q, c)
        part = sp.reference(q, c, rows)
        assert torch.equal(part, full[rows])
        out = sp.scores(q, c)
        r = 299
        k = int((~torch.isneginf(out[r])).nonzero()[0])
        out[r, k] += 1.0
        with pytest.raises(sp.KernelMismatch):
            sp._self_check(q, c, out)

    def test_selfcheck_error_disables_the_kernel(monkeypatch):
        g = torch.Generator().manual_seed(29)
        q = _rand_csr(8, 20, 1, 4, g, False)
        c = _rand_csr(30, 20, 1, 5, g, False)

        def boom(*a, **k):
            raise ValueError("reference broke")
        monkeypatch.setattr(sp, "reference", boom)
        with pytest.raises(ValueError):
            sp.scores(q, c)
        assert "self-check could not run" in sp.kernel_off()

    # --- integration: SparseBatchSlice -------------------------------------------

    @pytest.mark.parametrize("zero_gate_ok", [True, False])
    @pytest.mark.parametrize("metric", ["dot", "cosine"])
    def test_slice_score_posting_equals_cusparse(zero_gate_ok, metric):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(13)
        signed = not zero_gate_ok
        q = _rand_csr(25, 50, 1, 8, g, signed)
        c = _rand_csr(120, 50, 1, 14, g, signed)
        Q = q.to_dense()
        qn = Q.norm(dim=1)
        rn = torch.rand(120, generator=g) + 0.5
        res = {}
        for kern in ("posting", "cusparse"):
            sp.reset()
            cache = _SparseQueryCache()
            cache.kernel = kern
            res[kern] = SparseBatchSlice(c, rn, zero_gate_ok, cache).score(Q, metric, qn)
            assert (sp.usage()["launches"] > 0) == (kern == "posting")
        a, b = res["posting"], res["cusparse"]
        assert torch.equal(torch.isneginf(a), torch.isneginf(b))
        m = ~torch.isneginf(a)
        assert torch.allclose(a[m], b[m], rtol=1e-6, atol=1e-7)

    def _ref_dot(Q, c):
        ref = Q @ c.to_dense().T
        ref[ref == 0] = float("-inf")
        return ref

    def test_auto_falls_back_after_first_slice_mismatch(monkeypatch):
        from nova_bf.compute import _SPARSE_BRANCHES, SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(14)
        q = _rand_csr(10, 30, 1, 6, g, False)
        c = _rand_csr(60, 30, 1, 9, g, False)
        Q = q.to_dense()
        _bad_reference(monkeypatch, 1.0)
        before = _SPARSE_BRANCHES["scored_posting"]
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        out = SparseBatchSlice(c, None, True, cache).score(Q, "dot")
        assert torch.allclose(out, _ref_dot(Q, c))
        assert _SPARSE_BRANCHES["scored_posting"] == before
        assert sp.kernel_off() is not None
        cache2 = _SparseQueryCache()
        cache2.kernel = "posting"
        sp.reset()
        with pytest.raises(sp.KernelMismatch):
            SparseBatchSlice(c, None, True, cache2).score(Q, "dot")

    def test_auto_falls_back_on_a_later_configuration_mismatch(monkeypatch):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(22)
        q = _rand_csr(10, 30, 1, 6, g, False)
        c1 = _rand_csr(64, 30, 1, 9, g, False)
        c2 = _rand_csr(37, 30, 1, 9, g, False)    # different shape -> new compile key
        Q = q.to_dense()
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        SparseBatchSlice(c1, None, True, cache).score(Q, "dot")
        assert sp.usage()["launches"] == 1
        _bad_reference(monkeypatch, 1.0)
        out = SparseBatchSlice(c2, None, True, cache).score(Q, "dot")
        assert torch.allclose(out, _ref_dot(Q, c2))
        assert "earlier slices of this run used the kernel" in sp.kernel_off()

    @pytest.mark.parametrize("mode", ["auto", "posting"])
    def test_launch_failure(monkeypatch, mode):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(30)
        q = _rand_csr(10, 30, 1, 6, g, False)
        c = _rand_csr(60, 30, 1, 9, g, False)
        Q = q.to_dense()

        def boom(*a, **k):
            raise RuntimeError("launch failed")
        monkeypatch.setattr(sp, "_launch", boom)
        cache = _SparseQueryCache()
        cache.kernel = mode
        sl = SparseBatchSlice(c, None, True, cache)
        if mode == "posting":
            with pytest.raises(RuntimeError, match="launch failed"):
                sl.score(Q, "dot")
        else:
            assert torch.allclose(sl.score(Q, "dot"), _ref_dot(Q, c))
        assert "launch failed" in sp.kernel_off()

    @pytest.mark.parametrize("mode", ["auto", "posting"])
    def test_oom_falls_back_in_auto_and_does_not_disable(monkeypatch, mode):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(31)
        q = _rand_csr(10, 30, 1, 6, g, False)
        c = _rand_csr(60, 30, 1, 9, g, False)
        Q = q.to_dense()

        def oom(*a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3 GiB")
        monkeypatch.setattr(sp, "_launch", oom)
        cache = _SparseQueryCache()
        cache.kernel = mode
        sl = SparseBatchSlice(c, None, True, cache)
        if mode == "posting":
            with pytest.raises(torch.OutOfMemoryError):
                sl.score(Q, "dot")
        else:
            assert torch.allclose(sl.score(Q, "dot"), _ref_dot(Q, c))
            assert sp.usage()["declines"] == {"out of memory": 1}
        assert sp.kernel_off() is None

    def test_postings_build_failure_disables(monkeypatch):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(32)
        q = _rand_csr(10, 30, 1, 6, g, False)
        c = _rand_csr(60, 30, 1, 9, g, False)

        def boom(*a, **k):
            raise IndexError("postings broke")
        monkeypatch.setattr(sp, "corpus_postings", boom)
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        out = SparseBatchSlice(c, None, True, cache).score(q.to_dense(), "dot")
        assert torch.allclose(out, _ref_dot(q.to_dense(), c))
        assert "postings broke" in sp.kernel_off()

    # --- integration: run_compute --------------------------------------------------

    def _write(path, rows, lang=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols = {
            "sparse_embedding": pa.array(
                [{"indices": i, "values": v} for i, v in rows],
                type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                                pa.field("values", pa.list_(pa.float32()))])),
            "id": pa.array([f"r{path.stem}_{j}" for j in range(len(rows))]),
        }
        if lang is not None:
            cols["language"] = pa.array(lang)
        pq.write_table(pa.table(cols), str(path))

    def _dataset(tmp_path, signed, n_files=3, per_file=150, seed=15):
        rng = np.random.default_rng(seed)

        def row(vocab, lo, hi):
            k = int(rng.integers(lo, hi + 1))
            idx = sorted(rng.choice(vocab, size=k, replace=False).tolist())
            v = rng.random(k).astype(np.float32) + 0.05
            if signed:
                v *= rng.choice([-1.0, 1.0], size=k).astype(np.float32)
            return idx, v.tolist()

        cdir = tmp_path / "corpus"
        cdir.mkdir(exist_ok=True)
        for f in range(n_files):
            _write(cdir / f"f{f}.parquet", [row(80, 1, 12) for _ in range(per_file)],
                   lang=[("eng" if (f + j) % 3 else "fra") for j in range(per_file)])
        _write(tmp_path / "q.parquet", [row(80, 1, 6) for _ in range(30)])
        return cdir, tmp_path / "q.parquet"

    def _cfg(cdir, qpath, out, kern, searches, batch=128):
        from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                    ParamsConfig, QueriesConfig)

        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
            queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="id"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, sparse_batch_size=batch, sparse_kernel=kern),
            searches=searches,
        )

    def _same_results(a, b):
        """Same queries; same hits except exact-tie swaps at the k-th score."""
        assert a["query_id"] == b["query_id"]
        for ha, hb, sa, sb in zip(a["hit_ids"], b["hit_ids"], a["hit_scores"], b["hit_scores"]):
            assert len(ha) == len(hb)
            np.testing.assert_allclose(sa, sb, rtol=1e-6, atol=1e-7)
            if not sa:
                continue
            kth = min(sb)
            tol = 1e-6 * max(1.0, abs(kth))
            above_a = {i for i, s in zip(ha, sa) if s > kth + tol}
            above_b = {i for i, s in zip(hb, sb) if s > kth + tol}
            assert above_a == above_b, "hits clearly above the k-th score differ"

    def _searches(filt=None):
        from nova_bf.config import SearchSpec

        return [SearchSpec(name="dot", k=10, metric="dot", vector_type="sparse", filter=filt),
                SearchSpec(name="cos", k=10, metric="cosine", vector_type="sparse", filter=filt)]

    @pytest.mark.parametrize("signed", [False, True])
    @pytest.mark.parametrize("filtered", [False, True])
    @pytest.mark.parametrize("layout", [64, 512])
    def test_run_compute_posting_equals_cusparse(tmp_path, signed, filtered, layout, monkeypatch):
        import pyarrow.parquet as pq

        from nova_bf.compute import _SPARSE_BRANCHES, run_compute
        from nova_bf.config import Filter, FilterCondition

        _layout(monkeypatch, layout)      # 64: slices of 150 rows span 3 blocks + a tail
        cdir, qpath = _dataset(tmp_path, signed)
        filt = Filter(must=[FilterCondition(field="language", match="eng")]) if filtered else None
        out = {}
        for kern in ("posting", "cusparse"):
            paths = run_compute(_cfg(cdir, qpath, tmp_path / kern, kern, _searches(filt), batch=150))
            used = _SPARSE_BRANCHES["scored_posting"]    # reset per run by run_compute
            assert (used > 0) == (kern == "posting"), (kern, used)
            out[kern] = {n: pq.read_table(p).to_pydict() for n, p in paths.items()}
        for n in ("dot", "cos"):
            _same_results(out["posting"][n], out["cusparse"][n])

    def test_run_compute_resets_and_records_policy(tmp_path, monkeypatch):
        import json

        from nova_bf.compute import _SPARSE_BRANCHES, run_compute

        cdir, qpath = _dataset(tmp_path, False, n_files=2, per_file=60)
        # run 1 (auto): forced mismatch -> kernel off, cuSPARSE used
        _bad_reference(monkeypatch, 1.0)
        run_compute(_cfg(cdir, qpath, tmp_path / "r1", "auto", _searches()))
        assert _SPARSE_BRANCHES["scored_posting"] == 0
        monkeypatch.undo()
        # run 2 (auto): reset per run, so the kernel is retried and used
        run_compute(_cfg(cdir, qpath, tmp_path / "r2", "auto", _searches()))
        assert _SPARSE_BRANCHES["scored_posting"] > 0
        m2 = json.loads(next((tmp_path / "r2").rglob("*manifest*.json")).read_text())
        k2 = m2["params"]["kernels"]["sparse_posting_kernel"]
        assert k2["policy"] == "auto" and k2["permitted"] and k2["launches"] > 0
        # run 3 (cusparse): policy reaches the manifest
        run_compute(_cfg(cdir, qpath, tmp_path / "r3", "cusparse", _searches()))
        m3 = json.loads(next((tmp_path / "r3").rglob("*manifest*.json")).read_text())
        k3 = m3["params"]["kernels"]["sparse_posting_kernel"]
        assert k3["policy"] == "cusparse" and k3["permitted"] is False and k3["launches"] == 0

    def test_manifest_records_policy():
        sp.set_policy("cusparse")
        try:
            u = sp.usage()
            assert u["policy"] == "cusparse" and u["permitted"] is False
        finally:
            sp.set_policy("auto")
        assert sp.usage()["permitted"] is True


if _CHILD:
    # --- round-2 mutation killers ------------------------------------------------

    def _kcsr(rows, vocab, dt=torch.float32, it=torch.int64):
        crow, cols, vals = [0], [], []
        for c, v in rows:
            cols += c
            vals += v
            crow.append(len(cols))
        return torch.sparse_csr_tensor(torch.tensor(crow, dtype=it), torch.tensor(cols, dtype=it),
                                       torch.tensor(vals, dtype=dt), size=(len(rows), vocab))

    def _krand(n, vocab, k, g, signed=True):
        rows = []
        for _ in range(n):
            c = sorted(torch.randperm(vocab, generator=g)[:k].tolist())
            v = torch.rand(len(c), generator=g) + 0.01
            if signed:
                v = v * torch.where(torch.rand(len(c), generator=g) < .5, -1., 1.)
            rows.append((c, v.tolist()))
        return _kcsr(rows, vocab)

    def test_query_side_double_declines():
        q = _kcsr([([0, 1], [1.0, 2.0])], 2, torch.float64)
        c = _kcsr([([0, 1], [1.0, 2.0])], 2)
        assert "float32" in sp.decline_reason(q, c)

    def test_both_double_declines():
        q = _kcsr([([0, 1], [1.0, 2.0])], 2, torch.float64)
        c = _kcsr([([0, 1], [1.0, 2.0])], 2, torch.float64)
        assert "float32" in sp.decline_reason(q, c)

    def test_l1_bound_uses_abs_on_signed_queries():
        q = _kcsr([(list(range(10)), [1e17, -1e17] * 5)], 10)
        c = _kcsr([([0], [5e18])], 10)
        assert "overflow" in sp.decline_reason(q, c)

    def test_bound_not_shared_across_query_matrices():
        c = _kcsr([([0], [5e18])], 10)
        small = _kcsr([([0], [1.0])], 10)
        big = _kcsr([(list(range(10)), [1e17] * 10)], 10)
        assert sp.decline_reason(small, c) is None
        assert "overflow" in sp.decline_reason(big, c)

    def test_decline_counts_by_reason_and_reset_clears():
        q = _kcsr([([0, 1], [1e20, 1.0])], 2)
        c = _kcsr([([0, 1], [-1e20, 2.0])], 2)
        sp.decline_reason(q, c)
        sp.decline_reason(q, c)
        assert sp.usage()["declines"] == {"weights large enough to overflow fp32 sums": 2}
        sp.reset()
        assert sp.usage()["declines"] == {}

    def test_index_dtype_change_rechecks():
        g = torch.Generator().manual_seed(1)
        q = _krand(8, 30, 4, g)
        c = _krand(64, 30, 6, g)
        sp.scores(q, c)
        q32 = torch.sparse_csr_tensor(q.crow_indices().int(), q.col_indices().int(),
                                      q.values(), size=q.shape)
        sp.scores(q32, c)
        assert sp.usage()["selfchecks"] == 2

    def test_misaligned_values_rechecks():
        g = torch.Generator().manual_seed(2)
        q = _krand(8, 30, 4, g)
        c = _krand(64, 30, 6, g)
        sp.scores(q, c)
        buf = torch.empty(q.values().numel() + 1, dtype=torch.float32)
        v = buf[1:]
        v.copy_(q.values())
        assert v.data_ptr() % 16 != 0
        sp.scores(torch.sparse_csr_tensor(q.crow_indices(), q.col_indices(), v, size=q.shape), c)
        assert sp.usage()["selfchecks"] == 2

    def test_launch_config_change_rechecks(monkeypatch):
        g = torch.Generator().manual_seed(33)
        q = _krand(8, 30, 4, g)
        c = _krand(64, 30, 6, g)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "DOC_BLOCK", 256)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "NUM_WARPS", 4)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "DOC_BLOCK", 0)
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 4

    def test_selfcheck_oom_keeps_kernel_and_rechecks(monkeypatch):
        g = torch.Generator().manual_seed(3)
        q = _krand(8, 30, 4, g)
        c = _krand(64, 30, 6, g)
        real = sp.reference

        def oom(*a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3 GiB")
        monkeypatch.setattr(sp, "reference", oom)
        with pytest.raises(torch.OutOfMemoryError):
            sp.scores(q, c)
        assert sp.kernel_off() is None and sp.usage()["launches"] == 0
        monkeypatch.setattr(sp, "reference", real)
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 1 and sp.usage()["launches"] == 1

    def test_failed_check_is_not_counted_as_a_launch(monkeypatch):
        g = torch.Generator().manual_seed(4)
        q = _krand(8, 30, 4, g)
        c = _krand(64, 30, 6, g)
        _bad_reference(monkeypatch, 1.0)
        with pytest.raises(sp.KernelMismatch):
            sp.scores(q, c)
        assert sp.usage()["launches"] == 0
        assert "earlier slices" not in sp.kernel_off()

    def test_first_disable_reason_wins():
        sp.kernel_disable("first")
        sp.kernel_disable("second")
        assert sp.kernel_off() == "first"

    def test_reset_forgets_checked():
        g = torch.Generator().manual_seed(5)
        q = _krand(8, 30, 4, g)
        c = _krand(64, 30, 6, g)
        sp.scores(q, c)
        sp.reset()
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 1

    def test_env_kill_switch_not_permitted(monkeypatch):
        monkeypatch.setenv("NOVA_BF_NO_SPARSE_POSTING", "1")
        assert sp.usage()["permitted"] is False

    def test_trailing_unused_token():
        q = _kcsr([([0, 2], [1.0, 1.0])], 4)
        c = _kcsr([([0], [2.0]), ([1], [3.0])], 4)     # token 3 never posted
        ptr, rows, vals = sp.corpus_postings(c)
        assert ptr.tolist() == [0, 1, 2, 2, 2]
        assert sp.scores(q, c).tolist() == [[2.0, float("-inf")]]


if _CHILD:
    # --- round-3 mutation killers + round-3 behaviour ------------------------------

    def _r3rand(n, vocab, k, g):
        rows = []
        for _ in range(n):
            c = sorted(torch.randperm(vocab, generator=g)[:k].tolist())
            v = (torch.rand(len(c), generator=g) + 0.01) * torch.where(
                torch.rand(len(c), generator=g) < .5, -1., 1.)
            rows.append((c, v.tolist()))
        return _kcsr(rows, vocab)

    def _tol_case():
        g = torch.Generator().manual_seed(21)
        q = _r3rand(20, 30, 6, g)
        c = _r3rand(60, 30, 10, g)
        rows = list(range(20))
        ref, shared = sp.reference(q, c, rows, counts=True)
        mag = sp.reference(q, c, rows, absolute=True)
        live = ~torch.isneginf(ref)
        tol = (2 * shared + 2) * torch.finfo(torch.float32).eps * mag
        return q, c, ref, shared, live, tol

    def test_tolerance_accepts_0p9_of_the_bound_everywhere():
        q, c, ref, shared, live, tol = _tol_case()
        assert bool((live & (shared == 1)).any())
        sp._self_check(q, c, torch.where(live, ref + 0.9 * tol, ref))

    def test_tolerance_rejects_1p1_of_the_bound():
        q, c, ref, shared, live, tol = _tol_case()
        r, k = map(int, (live & (shared >= 2)).nonzero()[0])
        bad = ref.clone()
        bad[r, k] += 1.1 * float(tol[r, k])
        with pytest.raises(sp.KernelMismatch):
            sp._self_check(q, c, bad)

    @pytest.mark.parametrize("db", [0, 16, 512])
    def test_query_token_unused_by_the_corpus(db, monkeypatch):
        monkeypatch.setattr(sp, "DOC_BLOCK", db)
        q = _kcsr([([0, 3], [1.0, 5.0])], 4)
        c = _kcsr([([0], [2.0]), ([1], [3.0])], 4)
        assert sp.scores(q, c).tolist() == [[2.0, float("-inf")]]

    def test_block_ptr_full_length_when_last_token_unused():
        c = _kcsr([([0], [2.0]), ([1], [3.0])], 4)
        ptr, rows, _ = sp.corpus_postings(c)
        for db in (1, 16, 512):
            bp, nblk = sp.block_ptr(ptr, rows, 4, 2, db)
            assert bp.numel() == 4 * nblk + 1 and int(bp[-1]) == 2

    def test_n_rows_one_specialisation_rechecks():
        g = torch.Generator().manual_seed(3)
        q = _r3rand(8, 30, 4, g)
        sp.scores(q, _r3rand(17, 30, 6, g))
        sp.scores(q, _r3rand(1, 30, 6, g))
        assert sp.usage()["selfchecks"] == 2

    def test_row_layout_block_sizes_are_in_the_key(monkeypatch):
        g = torch.Generator().manual_seed(33)
        q = _r3rand(8, 30, 4, g)
        c = _r3rand(64, 30, 6, g)
        monkeypatch.setattr(sp, "DOC_BLOCK", 0)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "BLOCK_P", 64)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "BLOCK_I", 128)
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 3

    def test_doc_block_is_in_the_key_even_with_equal_tiles(monkeypatch):
        g = torch.Generator().manual_seed(34)
        q = _r3rand(8, 30, 4, g)
        c = _r3rand(64, 30, 6, g)
        monkeypatch.setattr(sp, "DOC_BLOCK", 1024)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "DOC_BLOCK", 2048)
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 2

    def test_empty_query_rows_and_empty_corpus_are_accepted():
        q = _kcsr([([], []), ([], [])], 4)
        c = _kcsr([([0], [2.0])], 4)
        assert sp.decline_reason(q, c) is None
        assert torch.isneginf(sp.scores(q, c)).all()
        q = _kcsr([([0], [1.0])], 4)
        c = _kcsr([([], []), ([], [])], 4)
        assert sp.decline_reason(q, c) is None
        assert torch.isneginf(sp.scores(q, c)).all()

    def test_sample_rows_endpoints_many_sizes():
        for n in list(range(257, 3000, 7)) + [16326]:
            s_ = sp._sample_rows(n)
            assert s_[0] == 0 and s_[-1] == n - 1 and len(s_) == 256, n

    def test_sample_cap_read_at_call_time(monkeypatch):
        monkeypatch.setattr(sp, "SELF_CHECK_ROWS", 4)
        assert sp._sample_rows(10) == [0, 3, 6, 9]

    def test_slice_returns_the_kernel_output(monkeypatch):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        q = _kcsr([([0], [1.0])], 4)
        c = _kcsr([([0], [2.0]), ([1], [3.0])], 4)
        monkeypatch.setattr(sp, "scores", lambda qc, cb, cmin=None: torch.full((1, 2), 7.0))
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        assert SparseBatchSlice(c, None, True, cache).score(q.to_dense(), "dot").tolist() \
            == [[7.0, 7.0]]

    def test_reset_clears_failures(monkeypatch):
        g = torch.Generator().manual_seed(10)
        q = _r3rand(12, 30, 5, g)
        c = _r3rand(40, 30, 8, g)
        _bad_reference(monkeypatch, 1.0)
        with pytest.raises(sp.KernelMismatch):
            sp.scores(q, c)
        sp.reset()
        assert sp.usage()["selfcheck_failures"] == 0

    def test_reference_multi_chunk(monkeypatch):
        import nova_bf.compute as cm

        g = torch.Generator().manual_seed(40)
        q = _r3rand(5, 30, 4, g)
        c = _r3rand(30, 30, 6, g)
        full = sp.reference(q, c)
        monkeypatch.setattr(cm, "_SPARSE_SWAP_MAX_DENSE_BYTES", 30 * 4 * 7)
        assert torch.equal(sp.reference(q, c), full)

    def test_slice_too_wide_for_the_grid_declines(monkeypatch):
        monkeypatch.setattr(sp, "MAX_GRID_Y", 2)
        q = _kcsr([([0], [1.0])], 4)
        c = _kcsr([([0], [1.0])] * (2 * sp.DOC_BLOCK + 1), 4)
        assert "too wide" in sp.decline_reason(q, c)

    def test_failing_slice_is_counted_as_a_decline(monkeypatch):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(41)
        q = _r3rand(6, 30, 4, g)
        c = _r3rand(40, 30, 6, g)
        _bad_reference(monkeypatch, 1.0)
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        SparseBatchSlice(c, None, True, cache).score(q.to_dense(), "dot")
        assert sp.usage()["declines"] == {"kernel failure": 1}

    def test_oom_streak_warns_and_then_switches_off(monkeypatch, caplog):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(42)
        q = _r3rand(6, 30, 4, g)
        Q = q.to_dense()

        def oom(*a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3 GiB")
        monkeypatch.setattr(sp, "_launch", oom)
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        import logging
        with caplog.at_level(logging.WARNING, logger="nova_bf.sparse_posting_triton"):
            for i in range(sp.OOM_STREAK_LIMIT):
                assert sp.kernel_off() is None
                SparseBatchSlice(_r3rand(40, 30, 6, g), None, True, cache).score(Q, "dot")
        assert sum("ran out of GPU memory" in r.message for r in caplog.records) == 1
        assert "consecutive out-of-memory" in sp.kernel_off()
        assert sp.usage()["declines"] == {"out of memory": sp.OOM_STREAK_LIMIT}

    def test_a_kernel_slice_resets_the_oom_streak(monkeypatch):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(43)
        q = _r3rand(6, 30, 4, g)
        Q = q.to_dense()
        real = sp._launch
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] % 2:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3 GiB")
            return real(*a, **k)
        monkeypatch.setattr(sp, "_launch", flaky)
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        for _ in range(2 * sp.OOM_STREAK_LIMIT):
            SparseBatchSlice(_r3rand(40, 30, 6, g), None, True, cache).score(Q, "dot")
        assert sp.kernel_off() is None

    def test_decline_check_oom_falls_back(monkeypatch):
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        g = torch.Generator().manual_seed(44)
        q = _r3rand(6, 30, 4, g)
        c = _r3rand(40, 30, 6, g)

        def oom(*a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3 GiB")
        monkeypatch.setattr(sp, "_magnitude_reason", oom)
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        out = SparseBatchSlice(c, None, True, cache).score(q.to_dense(), "dot")
        assert torch.allclose(out, _ref_dot(q.to_dense(), c))


if _CHILD:
    # --- round-4 mutation killers + round-4 behaviour ------------------------------

    def _r4rand(n, vocab, k, seed):
        g = torch.Generator().manual_seed(seed)
        rows = []
        for _ in range(n):
            c = sorted(torch.randperm(vocab, generator=g)[:k].tolist())
            rows.append((c, (torch.rand(len(c), generator=g) + 0.01).tolist()))
        return _kcsr(rows, vocab)

    def test_row_layout_num_warps_in_key(monkeypatch):
        q, c = _r4rand(8, 30, 4, 1), _r4rand(64, 30, 6, 2)
        monkeypatch.setattr(sp, "DOC_BLOCK", 0)
        sp.scores(q, c)
        monkeypatch.setattr(sp, "NUM_WARPS", 4)
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 2

    @pytest.mark.parametrize("knob,val", [("BLOCK_P", 64), ("BLOCK_I", 128), ("NUM_WARPS", 4)])
    def test_blocked_launch_values_in_key(monkeypatch, knob, val):
        q, c = _r4rand(8, 30, 4, 1), _r4rand(64, 30, 6, 2)
        monkeypatch.setattr(sp, "DOC_BLOCK", 512)
        sp.scores(q, c)
        monkeypatch.setattr(sp, knob, val)
        sp.scores(q, c)
        assert sp.usage()["selfchecks"] == 2

    def test_int_divisible_by_16_not_8():
        q = _r4rand(8, 30, 4, 1)
        sp.scores(q, _r4rand(16, 30, 6, 2))
        sp.scores(q, _r4rand(24, 30, 6, 3))   # % 8 == 0 but % 16 != 0
        assert sp.usage()["selfchecks"] == 2

    def test_pointer_aligned_8_not_16():
        q, c = _r4rand(8, 30, 4, 1), _r4rand(64, 30, 6, 2)
        sp.scores(q, c)
        buf = torch.empty(q.values().numel() + 2, dtype=torch.float32)
        v = buf[2:]
        v.copy_(q.values())
        assert v.data_ptr() % 16 == 8
        sp.scores(torch.sparse_csr_tensor(q.crow_indices(), q.col_indices(), v, size=q.shape), c)
        assert sp.usage()["selfchecks"] == 2

    def test_grid_exactly_max_blocks_accepted(monkeypatch):
        monkeypatch.setattr(sp, "MAX_GRID_Y", 2)
        q = _kcsr([([0], [1.0])], 4)
        c = _kcsr([([0], [1.0])] * (2 * sp.DOC_BLOCK), 4)
        assert sp.decline_reason(q, c) is None

    def test_constants():
        assert sp.MAX_GRID_Y == 65535 and sp.OOM_STREAK_LIMIT == 3 and sp.MAX_SUM == 1e36

    def test_l1_bound_is_the_max_over_rows():
        q = _kcsr([(list(range(10)), [1e17] * 10), ([], [])], 10)
        c = _kcsr([([0], [1.5e18])], 10)
        assert "overflow" in sp.decline_reason(q, c)

    def test_reset_clears_oom_streak_and_warning():
        sp.note_oom()
        sp.note_oom()
        sp.reset()
        sp.note_oom()
        assert sp.kernel_off() is None and sp._OOM_WARNED

    def test_oom_warning_is_once_per_run(caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="nova_bf.sparse_posting_triton"):
            sp.note_oom()
            sp.note_kernel_slice()
            sp.note_oom()
            sp.note_kernel_slice()
            sp.note_oom()
        assert sum("ran out of GPU memory" in r.message for r in caplog.records) == 1
        assert sp.kernel_off() is None          # streaks were broken by kernel slices

    def test_usage_unavailable_reports_env(monkeypatch):
        monkeypatch.setenv("NOVA_BF_NO_SPARSE_POSTING", "1")
        assert sp.usage()["unavailable"] == "NOVA_BF_NO_SPARSE_POSTING"

    def test_reference_counts_are_shared_counts():
        q = _kcsr([([0, 1, 2], [1.0, 2.0, 3.0])], 4)
        c = _kcsr([([0, 1, 2], [1.0, 1.0, 1.0]), ([1], [1.0]), ([3], [1.0])], 4)
        ref, shared = sp.reference(q, c, counts=True)
        assert shared.tolist() == [[3.0, 1.0, 0.0]]
        assert ref.tolist() == [[6.0, 2.0, float("-inf")]]

    def test_slice_scored_once_across_repeat_calls():
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache

        q, c = _r4rand(6, 30, 4, 1), _r4rand(40, 30, 6, 2)
        cache = _SparseQueryCache()
        cache.kernel = "auto"
        sl = SparseBatchSlice(c, None, True, cache)
        sl.score(q.to_dense(), "dot")
        sl.score(q.to_dense(), "dot")
        sl.score(q.to_dense(), "cosine", q.to_dense().norm(dim=1)) if sl.row_norms is not None else None
        assert sp.usage()["launches"] == 1

    def test_selfcheck_live_where_reference_is_dead():
        q = _kcsr([([0], [1.0])], 2)
        c = _kcsr([([0], [1.0]), ([1], [1.0])], 2)
        out = sp.scores(q, c)
        out[0, 1] = 5.0
        with pytest.raises(sp.KernelMismatch):
            sp._self_check(q, c, out)

    def test_empty_slices_are_not_launches():
        q = _r4rand(4, 30, 3, 5)
        e = _kcsr([], 30)
        assert sp.scores(q, e).shape == (4, 0)
        assert sp.usage()["launches"] == 0 and sp.usage()["selfchecks"] == 0

    def test_reset_restores_the_default_policy():
        sp.set_policy("cusparse")
        sp.reset()
        assert sp.usage()["policy"] == "auto" and sp.usage()["permitted"]

    def test_interpreter_detection_is_by_kernel_not_env(monkeypatch):
        assert sp._INTERPRETED            # this child really interprets
        monkeypatch.setattr(sp, "_INTERPRETED", False)
        q = _kcsr([([0], [1.0])], 2)
        c = _kcsr([([0], [1.0])], 2)
        assert sp.decline_reason(q, c) == "not a CUDA slice"


if _CHILD:
    # --- dense hot-token path ------------------------------------------------------

    def _pos_csr(n, vocab, lo, hi, g, hot=()):
        return _rand_csr(n, vocab, lo, hi, g, False, hot=hot)

    def _hot(monkeypatch, n):
        monkeypatch.setattr(sp, "HOT_TOKENS", n)
        sp._HOT_PLAN.clear()

    @pytest.mark.parametrize("db", [16, 512])
    @pytest.mark.parametrize("n_hot", [1, 4, 64, 1000])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_hot_path_matches_float64_truth(db, n_hot, seed, monkeypatch):
        _layout(monkeypatch, db)
        _hot(monkeypatch, n_hot)
        g = torch.Generator().manual_seed(60 + seed)
        q = _pos_csr(37, 60, 0, 9, g)                  # includes empty queries
        c = _pos_csr(700, 60, 0, 25, g, hot=(3, 7))    # long postings, n_rows % db != 0
        before = sp.usage()["hot_launches"]
        out = sp.scores(q, c, cmin=float(c.values().min()))
        assert sp.usage()["hot_launches"] == before + 1
        _check(out, q, c)

    def test_hot_path_all_hot_and_all_cold_queries(monkeypatch):
        _layout(monkeypatch, 64)
        _hot(monkeypatch, 3)
        # tokens 0,1,2 are the most frequent (hot); 5,6 are cold
        q = _kcsr([([0, 1, 2], [1.0, 2.0, 3.0]), ([5, 6], [1.0, 1.0]),
                   ([0], [1.0]), ([0], [1.0]), ([1], [1.0]), ([2], [1.0]), ([], [])], 8)
        g = torch.Generator().manual_seed(61)
        c = _pos_csr(150, 8, 1, 6, g)
        _check(sp.scores(q, c, cmin=float(c.values().min())), q, c)
        hotidx, n_hot = sp.hot_plan(q)
        assert n_hot == 3 and sorted(int(i) for i in (hotidx >= 0).nonzero().flatten()) == [0, 1, 2]

    def test_hot_path_is_deterministic(monkeypatch):
        _layout(monkeypatch, 64)
        _hot(monkeypatch, 8)
        g = torch.Generator().manual_seed(62)
        q = _pos_csr(20, 40, 2, 8, g)
        c = _pos_csr(300, 40, 2, 12, g)
        a = sp.scores(q, c, cmin=1e-3)
        b = sp.scores(q, c, cmin=1e-3)
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))

    @pytest.mark.parametrize("cmin", [None, 0.0, -1.0])
    def test_hot_path_needs_a_positive_slice_minimum(cmin, monkeypatch):
        _hot(monkeypatch, 8)
        g = torch.Generator().manual_seed(63)
        q = _pos_csr(10, 30, 2, 5, g)
        c = _pos_csr(80, 30, 2, 6, g)
        sp.scores(q, c, cmin=cmin)
        assert sp.usage()["hot_launches"] == 0

    def test_hot_plan_rules(monkeypatch):
        _layout(monkeypatch, 512)
        _hot(monkeypatch, 2)
        q = _kcsr([([1, 2, 3], [1.0, 1.0, 1.0]), ([2, 3], [1.0, 1.0]), ([3], [1.0])], 5)
        hotidx, n = sp.hot_plan(q)
        assert n == 2 and hotidx.tolist() == [-1, -1, 1, 0, -1]      # by frequency
        qt = _kcsr([([0, 1], [1.0, 1.0]), ([0, 1], [1.0, 1.0])], 3)  # tie -> lower id first
        sp._HOT_PLAN.clear()
        assert sp.hot_plan(qt)[0].tolist() == [0, 1, -1]
        neg = _kcsr([([0, 1], [1.0, -1.0])], 3)                       # a negative query weight
        sp._HOT_PLAN.clear()
        assert sp.hot_plan(neg) == (None, 0)
        monkeypatch.setattr(sp, "DOC_BLOCK", 0)                        # row layout: no hot path
        sp._HOT_PLAN.clear()
        assert sp.hot_plan(q) == (None, 0)
        monkeypatch.setattr(sp, "DOC_BLOCK", 512)
        _hot(monkeypatch, 0)                                            # disabled
        assert sp.hot_plan(q) == (None, 0)

    def test_hot_plan_cache_follows_config(monkeypatch):
        _layout(monkeypatch, 512)
        _hot(monkeypatch, 1)
        q = _kcsr([([0, 1], [1.0, 1.0]), ([1], [1.0])], 3)
        assert sp.hot_plan(q)[1] == 1
        monkeypatch.setattr(sp, "HOT_TOKENS", 2)       # no explicit clear: key must change
        assert sp.hot_plan(q)[1] == 2

    def test_hot_table_contents():
        c = _kcsr([([0, 2], [1.5, 2.5]), ([2], [4.0]), ([1], [9.0])], 3)
        ptr, prow, pval = sp.corpus_postings(c)
        hotidx = torch.tensor([-1, -1, 0], dtype=torch.int32)
        t = sp.hot_table(ptr, prow, pval, hotidx, 1, 3, 3)
        assert t[:3].tolist() == [2.5, 4.0, 0.0]         # token 2 (hot row 0) per doc
        assert t.numel() == 1 * 3 + int(prow.numel())    # + one scratch slot per posting

    def test_hot_table_destinations_are_distinct():
        """No two postings may share an index_put_ destination (unspecified
        winner), even when many cold postings hit the same doc."""
        g = torch.Generator().manual_seed(5)
        c = _pos_csr(40, 30, 6, 12, g)
        ptr, prow, pval = sp.corpus_postings(c)
        hotidx = torch.full((30,), -1, dtype=torch.int32)
        hotidx[torch.tensor([3, 7, 11])] = torch.tensor([0, 1, 2], dtype=torch.int32)
        t = sp.hot_table(ptr, prow, pval, hotidx, 3, 40, 30)
        dense = c.to_dense()
        for tok, h in ((3, 0), (7, 1), (11, 2)):
            assert torch.equal(t[h * 40:(h + 1) * 40], dense[:, tok])
        # scratch region holds each cold posting exactly once
        cold = hotidx.index_select(0, torch.repeat_interleave(
            torch.arange(30), ptr.diff())) < 0
        assert torch.equal(t[3 * 40:][cold], pval[cold])

    def test_selfcheck_catches_a_bad_hot_table(monkeypatch):
        _layout(monkeypatch, 64)
        _hot(monkeypatch, 4)
        g = torch.Generator().manual_seed(64)
        q = _pos_csr(12, 30, 2, 5, g)
        c = _pos_csr(120, 30, 2, 6, g)
        real = sp.hot_table
        monkeypatch.setattr(sp, "hot_table", lambda *a, **k: real(*a, **k) * 1.5)
        with pytest.raises(sp.KernelMismatch):
            sp.scores(q, c, cmin=1e-3)

    def test_hot_and_cold_configurations_are_checked_separately(monkeypatch):
        _layout(monkeypatch, 64)
        _hot(monkeypatch, 4)
        g = torch.Generator().manual_seed(65)
        q = _pos_csr(12, 30, 2, 5, g)
        c = _pos_csr(128, 30, 2, 6, g)
        sp.scores(q, c, cmin=None)
        sp.scores(q, c, cmin=1e-3)
        assert sp.usage()["selfchecks"] == 2

    @pytest.mark.parametrize("signed", [False, True])
    def test_run_compute_takes_the_hot_path_only_on_positive_data(tmp_path, signed, monkeypatch):
        from nova_bf.compute import run_compute
        from nova_bf.config import SearchSpec

        _hot(monkeypatch, 8)
        cdir, qpath = _dataset(tmp_path, signed)
        run_compute(_cfg(cdir, qpath, tmp_path / "o", "auto",
                         [SearchSpec(name="d", k=5, metric="dot", vector_type="sparse")], batch=150))
        assert (sp.usage()["hot_launches"] > 0) == (not signed)

    # --- threshold select in _cutfill ------------------------------------------------

    def _cutfill_launch(scores, thr, k, sel, enc=None):
        from nova_bf import topk_triton as tt

        n_q, n_cols = scores.shape
        ordinal = torch.arange(n_cols, dtype=torch.int64)
        rank = tt.rank_of(ordinal)
        block = tt._triton.next_power_of_2(n_cols)
        outk = torch.full((n_q, k), 777, dtype=torch.int64)
        outi = torch.full((n_q, k), 777, dtype=torch.int64 if enc is not None else torch.int32)
        live = torch.full((n_q,), 9, dtype=torch.uint8)
        tt._cutfill[(n_q,)](
            scores, scores, scores, rank, ordinal, enc if enc is not None else scores,
            outk, outi, thr, live, scores.stride(0), n_cols, k,
            BLOCK=block, RBITS=max(1, int(n_cols).bit_length()),
            HAS_NRM=False, HAS_CNRM=False, HAS_THR=True, HAS_ENC=enc is not None,
            HAS_SEL=sel)
        return outk, outi, live

    def _fold_ref(state, part, live, k):
        """Reference fold: top-k of state ∪ part for live rows; dead rows keep state."""
        out = state.clone()
        for r in range(state.shape[0]):
            if live[r]:
                out[r] = torch.cat([state[r], part[r]]).sort(descending=True).values[:k]
        return out

    def _state_and_thr(n_q, k, kth_scores):
        from nova_bf.tiebreak import pack

        # a full state whose k-th (smallest) key has the given score
        st = torch.stack([pack(torch.linspace(float(s_) + 5, float(s_), k)[None, :],
                               torch.arange(10 ** 6, 10 ** 6 + k))[0] for s_ in kth_scores])
        return st, st.min(dim=1).values.contiguous()

    @pytest.mark.parametrize("enc", [False, True])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_threshold_select_folds_identically(enc, seed):
        g = torch.Generator().manual_seed(70 + seed)
        n_q, n_cols, k = 30, 300, 20
        scores = torch.randn(n_q, n_cols, generator=g) * 2
        # thresholds spanning dead rows, few survivors (select), many (> k: full path)
        kth = torch.quantile(scores, torch.linspace(0.6, 1.05, n_q).clamp(max=1.0), dim=1).diagonal()
        kth[-3:] = scores.max(dim=1).values[-3:] + 1.0                 # dead rows
        state, thr = _state_and_thr(n_q, k, kth)
        e = torch.arange(n_cols, dtype=torch.int64) * 7 + 3 if enc else None
        a = _cutfill_launch(scores, thr, k, False, e)
        b = _cutfill_launch(scores, thr, k, True, e)
        assert torch.equal(a[2], b[2])                                  # identical liveness
        assert torch.equal(_fold_ref(state, a[0], a[2], k), _fold_ref(state, b[0], b[2], k))
        n_ge = (scores >= kth[:, None]).sum(dim=1)
        assert ((n_ge > 0) & (n_ge <= k)).any() and (n_ge > k).any()  # both paths ran

    def test_threshold_select_ties_and_boundaries():
        from nova_bf.tiebreak import NEUTRAL_KEY, pack

        k = 4
        # row 0: exactly k lanes at/above; row 1: k+1 (full path); row 2: ties at the cut
        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
                               [5.0, 4.0, 3.0, 2.0, 2.0, 0.0, -1.0, -2.0],
                               [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, -1.0, float("-inf")]])
        kth = torch.tensor([2.0, 2.0, 2.0])
        state, thr = _state_and_thr(3, k, kth)
        a = _cutfill_launch(scores, thr, k, False)
        b = _cutfill_launch(scores, thr, k, True)
        assert torch.equal(_fold_ref(state, a[0], a[2], k), _fold_ref(state, b[0], b[2], k))
        # row 2 took the select path: 3 real keys then NEUTRAL padding
        assert (b[0][2] == NEUTRAL_KEY).sum() == 1

    def test_threshold_select_skips_rows_whose_state_is_not_full():
        # an under-filled state carries the -inf sentinel threshold: every lane
        # counts, so the row must take the full path (no NEUTRAL padding)
        from nova_bf.tiebreak import NEUTRAL_KEY, SENTINEL_KEY

        scores = torch.tensor([[float("-inf")] * 6 + [1.0, 2.0]])
        thr = torch.tensor([SENTINEL_KEY], dtype=torch.int64)
        a = _cutfill_launch(scores, thr, 4, False)
        b = _cutfill_launch(scores, thr, 4, True)
        assert torch.equal(a[0], b[0]) and not (b[0] == NEUTRAL_KEY).any()

    def test_threshold_select_kill_switch(monkeypatch):
        from nova_bf import topk_triton as tt

        assert tt._select_on()
        monkeypatch.setenv("NOVA_BF_NO_TOPK_SELECT", "1")
        assert not tt._select_on()

    # --- pinned staging ----------------------------------------------------------------

    def test_pinned_staging_is_off_on_cpu(monkeypatch):
        from nova_bf import compute as cm

        assert cm._sparse_staging("cpu") is None
        monkeypatch.setenv("NOVA_BF_NO_SPARSE_PINNED", "1")
        assert cm._sparse_staging("cuda:0") is None

    def test_transfer_computes_host_value_range():
        import numpy as np

        from nova_bf.compute import SparseCorpusBatch

        offs = np.array([0, 2, 3, 5], dtype=np.int64)
        idx = np.array([0, 1, 1, 0, 2], dtype=np.int64)
        val = np.array([1.0, 7.0, 2.0, 3.0, 0.5], dtype=np.float32)
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False).transfer(0, 3, "cpu")
        assert sl.value_absmax == 7.0 and sl.value_min == 0.5
        val[1] = -9.0
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False).transfer(0, 3, "cpu")
        assert sl.value_absmax == 9.0 and sl.value_min == -9.0
        val[1] = np.nan
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False).transfer(0, 3, "cpu")
        assert np.isnan(sl.value_absmax)

    def test_decline_uses_the_host_absmax():
        q = _kcsr([(list(range(10)), [1e17] * 10)], 10)
        c = _kcsr([([0], [1.0])], 10)
        assert sp.decline_reason(q, c) is None
        assert "overflow" in sp.decline_reason(q, c, cmax=5e18)
        assert "non-finite" in sp.decline_reason(q, c, cmax=float("nan"))


if _CHILD:
    # --- round-1 mutation killers (select, hot path, host range) ------------------
    import math
    import numpy as np

    def _sel_launch(scores, thr, k, sel, enc=None, guard=False):
        from nova_bf import topk_triton as tt
        n_q, n_cols = scores.shape
        ordinal = torch.arange(n_cols, dtype=torch.int64)
        rank = tt.rank_of(ordinal)
        block = tt._triton.next_power_of_2(n_cols)
        big_k = torch.full((n_q + 1, k), 777, dtype=torch.int64)
        big_i = torch.full((n_q + 1, k), 777, dtype=torch.int64 if enc is not None else torch.int32)
        outk, outi = big_k[:n_q], big_i[:n_q]
        live = torch.full((n_q,), 9, dtype=torch.uint8)
        tt._cutfill[(n_q,)](
            scores, scores, scores, rank, ordinal, enc if enc is not None else scores,
            outk, outi, thr, live, scores.stride(0), n_cols, k,
            BLOCK=block, RBITS=max(1, int(n_cols).bit_length()),
            HAS_NRM=False, HAS_CNRM=False, HAS_THR=True, HAS_ENC=enc is not None, HAS_SEL=sel)
        return outk, outi, live, big_k[n_q], big_i[n_q]


    def _sel_setup(seed):
        from nova_bf.tiebreak import pack
        g = torch.Generator().manual_seed(seed)
        n_q, n_cols, k = 12, 40, 6
        scores = torch.randn(n_q, n_cols, generator=g)
        srt = scores.sort(dim=1, descending=True).values
        kth = srt[torch.arange(n_q), torch.arange(n_q) % 5]          # n_ge in 1..5 <= k: select
        st = torch.stack([pack(torch.linspace(float(s) + 5, float(s), k)[None, :],
                               torch.arange(10 ** 6, 10 ** 6 + k))[0] for s in kth])
        return scores, st.min(dim=1).values.contiguous(), k


    @pytest.mark.parametrize("enc", [False, True])
    def test_select_ids_match_full_path(enc):
        scores, thr, k = _sel_setup(5)
        e = torch.arange(scores.shape[1], dtype=torch.int64) * 7 + 3 if enc else None
        a = _sel_launch(scores, thr, k, False, e)
        b = _sel_launch(scores, thr, k, True, e)
        from nova_bf.tiebreak import NEUTRAL_KEY
        for r in range(scores.shape[0]):
            ma = {int(x): int(y) for x, y in zip(a[0][r], a[1][r])}
            mb = {int(x): int(y) for x, y in zip(b[0][r], b[1][r]) if int(x) != NEUTRAL_KEY}
            for key, idx in mb.items():
                assert ma[key] == idx, (r, key, idx, ma.get(key))


    def test_select_never_writes_past_its_row():
        scores, thr, k = _sel_setup(6)
        for e in (None, torch.arange(scores.shape[1], dtype=torch.int64)):
            o = _sel_launch(scores, thr, k, True, e)
            assert (o[3] == 777).all() and (o[4] == 777).all()


    def test_select_writes_liveness():
        scores, thr, k = _sel_setup(7)
        live = _sel_launch(scores, thr, k, True)[2]
        assert set(live.tolist()) <= {0, 1}


    def test_slice_passes_host_absmax_to_decline():
        from nova_bf.compute import SparseBatchSlice, _SparseQueryCache
        q = _kcsr([([0], [1.0])], 4)
        c = _kcsr([([0], [2.0]), ([1], [3.0])], 4)
        cache = _SparseQueryCache()
        cache.kernel = "posting"
        with pytest.raises(RuntimeError, match="non-finite"):
            SparseBatchSlice(c, None, True, cache, value_absmax=float("nan")).score(q.to_dense(), "dot")


    def test_hot_plan_not_reused_for_another_query_csr(monkeypatch):
        _layout(monkeypatch, 16)
        monkeypatch.setattr(sp, "HOT_TOKENS", 2)
        sp._HOT_PLAN.clear()
        q1 = _kcsr([([0, 1], [1.0, 1.0])], 3)
        c1 = _kcsr([([0, 1], [1.0, 2.0])], 3)
        sp.scores(q1, c1, cmin=1.0)
        q2 = _kcsr([([5, 50], [1.0, 2.0])], 60)
        c2 = _kcsr([([5, 50], [1.0, 2.0]), ([7], [1.0])], 60)
        out = sp.scores(q2, c2, cmin=1.0)
        _check(out, q2, c2)
        h, n = sp.hot_plan(q2)
        assert h.numel() == 60 and int(h[5]) >= 0 and int(h[50]) >= 0


    def test_hot_underflow_product_is_zero_not_untouched(monkeypatch):
        _layout(monkeypatch, 16)
        monkeypatch.setattr(sp, "HOT_TOKENS", 4)
        sp._HOT_PLAN.clear()
        q = _kcsr([([0], [1e-30])], 2)
        c = _kcsr([([0], [1e-30]), ([1], [1.0])], 2)
        out = sp.scores(q, c, cmin=1e-30)
        assert sp.usage()["hot_launches"] == 1
        assert out[0, 0].item() == 0.0 and math.isinf(out[0, 1].item())


    def test_transfer_range_is_the_slice_only():
        from nova_bf.compute import SparseCorpusBatch
        offs = np.array([0, 2, 3, 5], dtype=np.int64)
        idx = np.array([0, 1, 1, 0, 2], dtype=np.int64)
        val = np.array([-9.0, 7.0, 2.0, 3.0, 0.5], dtype=np.float32)
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False).transfer(1, 2, "cpu")
        assert sl.value_absmax == 2.0 and sl.value_min == 2.0


    def test_transfer_skips_range_for_cusparse_and_empty():
        from nova_bf.compute import SparseCorpusBatch, _SparseQueryCache
        offs = np.array([0, 0, 1], dtype=np.int64)
        idx = np.array([0], dtype=np.int64)
        val = np.array([1.0], dtype=np.float32)
        b = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False)
        sl = b.transfer(0, 1, "cpu")
        assert sl.value_absmax is None and sl.value_min is None
        cache = _SparseQueryCache()
        cache.kernel = "cusparse"
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False, q_cache=cache).transfer(0, 2, "cpu")
        assert sl.value_absmax is None


    def test_nan_slice_min_propagates():
        from nova_bf.compute import SparseCorpusBatch
        offs = np.array([0, 2], dtype=np.int64)
        idx = np.array([0, 1], dtype=np.int64)
        val = np.array([np.nan, 1.0], dtype=np.float32)
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(2), False).transfer(0, 1, "cpu")
        assert np.isnan(sl.value_min)


    def test_select_kplus1_lanes_take_the_full_path():
        from nova_bf.tiebreak import pack
        k = 2
        # flat state at score 1.0: three lanes >= thr (k+1); the best lane is last in column order
        st = pack(torch.tensor([[1.0, 1.0]]), torch.tensor([10 ** 6, 10 ** 6 + 1]))
        thr = st.min(dim=1).values.contiguous()
        scores = torch.tensor([[1.5, 2.0, 0.0, 3.0]])
        o = _sel_launch(scores, thr, k, True)
        full = _sel_launch(scores, thr, k, False)
        assert sorted(o[0][0].tolist()) == sorted(full[0][0].tolist())
        assert (o[3] == 777).all()


    def test_liveness_is_exact():
        from nova_bf.tiebreak import pack
        k = 2
        st = pack(torch.tensor([[1.0, 1.0], [5.0, 5.0]]), torch.tensor([10 ** 6, 10 ** 6 + 1]))
        thr = st.min(dim=1).values.contiguous()
        scores = torch.tensor([[1.5, 2.0, 0.0, 3.0], [1.5, 2.0, 0.0, 3.0]])
        for sel in (False, True):
            assert _sel_launch(scores, thr, k, sel)[2].tolist() == [1, 0]


    def test_stored_zero_slice_never_takes_the_hot_path(monkeypatch):
        from nova_bf.compute import SparseCorpusBatch, _SparseQueryCache
        _layout(monkeypatch, 16)
        monkeypatch.setattr(sp, "HOT_TOKENS", 4)
        sp._HOT_PLAN.clear()
        offs = np.array([0, 1, 2], dtype=np.int64)
        idx = np.array([0, 1], dtype=np.int64)
        val = np.array([0.0, 2.0], dtype=np.float32)          # row 0: a stored zero
        cache = _SparseQueryCache()
        cache.kernel = "posting"
        sl = SparseCorpusBatch(offs, idx, val, None, np.arange(3), False, q_cache=cache).transfer(0, 2, "cpu")
        raw = sl._posting_raw(torch.tensor([[1.0, 1.0, 0.0]]), cache)
        assert sp.usage()["hot_launches"] == 0
        assert raw[0].tolist() == [0.0, 2.0]                  # stored zero is a candidate (0.0, not -inf)

    def test_threshold_select_under_an_exchange_threshold():
        # A rank's state not yet full (sentinel slots) but a finite peer bound:
        # the select path drops lanes below the bound; folding must still give
        # the same entries at or above the bound as the full path.
        from nova_bf.tiebreak import SENTINEL_KEY, pack

        k = 4
        state = torch.full((1, k), SENTINEL_KEY, dtype=torch.int64)
        state[0, :2] = pack(torch.tensor([[9.0, 8.0]]), torch.tensor([10**6, 10**6 + 1]))[0]
        peer = pack(torch.tensor([[2.0]]), torch.tensor([0]))[:, 0]       # peer's k-th
        thr = torch.maximum(state.min(dim=1).values, peer).contiguous()
        scores = torch.tensor([[5.0, 1.0, 3.0, 0.5, 2.5, 1.5, -1.0, 0.0]])
        a = _sel_launch(scores, thr, k, False)
        b = _sel_launch(scores, thr, k, True)
        fa = torch.cat([state[0], a[0][0]]).sort(descending=True).values[:k]
        fb = torch.cat([state[0], b[0][0]]).sort(descending=True).values[:k]
        bound = int(thr[0])
        assert fa[fa >= bound].tolist() == fb[fb >= bound].tolist()


if _CHILD:
    # --- round-2 mutation killers ------------------------------------------------

    def _sel_launch_ord(scores, thr, k, sel, ordinal):
        from nova_bf import topk_triton as tt
        n_q, n_cols = scores.shape
        rank = tt.rank_of(ordinal)
        outk = torch.full((n_q, k), 777, dtype=torch.int64)
        outi = torch.full((n_q, k), 777, dtype=torch.int32)
        live = torch.full((n_q,), 9, dtype=torch.uint8)
        tt._cutfill[(n_q,)](
            scores, scores, scores, rank, ordinal, scores,
            outk, outi, thr, live, scores.stride(0), n_cols, k,
            BLOCK=tt._triton.next_power_of_2(n_cols), RBITS=max(1, int(n_cols).bit_length()),
            HAS_NRM=False, HAS_CNRM=False, HAS_THR=True, HAS_ENC=False, HAS_SEL=sel)
        return outk, outi, live

    def test_select_ids_are_columns_not_ordinals():
        # Without `encoded`, OUTI holds the winning COLUMN (the caller maps it to
        # an ordinal); ordinals != columns here, unlike every other select test.
        from nova_bf.tiebreak import NEUTRAL_KEY, pack
        g = torch.Generator().manual_seed(9)
        n_q, n_cols, k = 12, 40, 6
        scores = torch.randn(n_q, n_cols, generator=g)
        ordinal = torch.arange(n_cols, dtype=torch.int64) * 3 + 1000
        kth = scores.sort(dim=1, descending=True).values[torch.arange(n_q), torch.arange(n_q) % 5]
        st = torch.stack([pack(torch.linspace(float(s) + 5, float(s), k)[None, :],
                               torch.arange(10 ** 6, 10 ** 6 + k))[0] for s in kth])
        thr = st.min(dim=1).values.contiguous()
        a = _sel_launch_ord(scores, thr, k, False, ordinal)
        b = _sel_launch_ord(scores, thr, k, True, ordinal)
        for r in range(n_q):
            ma = {int(x): int(y) for x, y in zip(a[0][r], a[1][r])}
            got = [(int(x), int(y)) for x, y in zip(b[0][r], b[1][r]) if int(x) != NEUTRAL_KEY]
            assert got
            for key, idx in got:
                assert 0 <= idx < n_cols and ma[key] == idx, (r, key, idx)

    def test_staging_fits_is_per_array_and_inclusive(monkeypatch):
        from nova_bf import compute as cm
        monkeypatch.setattr(cm._PinnedStagingRing, "MAX_ELEMS", 4)
        r = cm._PinnedStagingRing()
        assert r.fits({"a": np.zeros(4), "b": np.zeros(1)})            # == MAX fits
        assert not r.fits({"a": np.zeros(5), "b": np.zeros(1)})
        assert not r.fits({"a": np.zeros(1), "b": np.zeros(5)})        # every array

    class _FakeRing:
        def __init__(self, ok):
            self.ok, self.calls = ok, 0

        def fits(self, parts):
            return self.ok

        def upload(self, parts, device):
            self.calls += 1
            return {n: torch.from_numpy(np.ascontiguousarray(v)).clone() for n, v in parts.items()}

    @pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
    def test_staged_csr_branch_matches_pageable_on_cpu(idx_dtype, monkeypatch):
        from nova_bf import compute as cm
        offs = np.array([0, 2, 3, 3, 6, 7], dtype=np.int64)
        idx = np.array([0, 3, 1, 0, 2, 4, 3], dtype=idx_dtype)
        val = np.arange(1, 8, dtype=np.float32)
        vocab = np.arange(5)
        ref = cm._sparse_batch_to_csr(offs, idx, val, 1, 5, vocab, "cpu").to_dense()
        for ok in (True, False):
            ring = _FakeRing(ok)
            monkeypatch.setattr(cm, "_sparse_staging", lambda d, ring=ring: ring)
            got = cm._sparse_batch_to_csr(offs, idx, val, 1, 5, vocab, "cpu")
            assert ring.calls == int(ok)                     # oversize slice: pageable
            assert got.shape == (4, 5) and torch.equal(got.to_dense(), ref)

    def test_reset_drops_the_hot_plan(monkeypatch):
        _hot(monkeypatch, 2)
        q = _kcsr([([0, 1], [1.0, 1.0])], 3)
        sp.hot_plan(q)
        assert sp._HOT_PLAN
        sp.reset()
        assert not sp._HOT_PLAN

    def test_hot_path_with_a_stored_zero_query_weight(monkeypatch):
        # probe for S09 (positivity check v>0 -> v>=0): correctness only
        _layout(monkeypatch, 16)
        _hot(monkeypatch, 4)
        q = _kcsr([([0, 1], [0.0, 1.0]), ([0], [0.0])], 3)
        c = _kcsr([([0], [2.0]), ([1], [3.0]), ([2], [1.0])], 3)
        _check(sp.scores(q, c, cmin=1.0), q, c)
