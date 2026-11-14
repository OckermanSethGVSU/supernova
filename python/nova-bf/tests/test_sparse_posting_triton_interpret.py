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
import subprocess
import sys

import numpy as np
import pytest

_CHILD = os.environ.get("NOVA_BF_SPARSE_POSTING_INTERPRET_CHILD") == "1"
_MIN_CHILD_TESTS = 120


def test_the_interpreted_suite_passes():
    """Re-run this file with `TRITON_INTERPRET=1` set before `import triton`."""
    if _CHILD:
        pytest.skip("this process IS the interpreted child")
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["NOVA_BF_SPARSE_POSTING_INTERPRET_CHILD"] = "1"
    env.pop("NOVA_BF_NO_SPARSE_POSTING", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "-p", "no:cacheprovider", os.path.abspath(__file__)],
        env=env, capture_output=True, text=True, timeout=3600,
    )
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
        monkeypatch.setattr(sp, "scores", lambda qc, cb: torch.full((1, 2), 7.0))
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
