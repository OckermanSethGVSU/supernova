"""Block-max sub-block skipping in the sparse posting kernel,
executed for real on CPU in the Triton interpreter.

What is checked (see python/nova-bf/docs/sparse_block_skip_prototype.md):
  (a) skip posting kernel: every kept cell BIT-identical to the unskipped
      hot kernel, every dropped cell strictly below the threshold, every cell
      at or above it kept; elided dead cells never written; random and
      adversarial data (ties at the threshold, fp32 sums rounding above a
      tight bound, one huge weight, unfull states, all-dead / all-live,
      ragged tails, several sub-block sizes, hot-only and hot+cold);
  (b) the bound kernels: BM exact, the fp32 skip test strict, the margin
      constants, the margin covering fp32 round-up;
  (c) the mask-aware top-k (`_cutfill` with HAS_LM) giving the same keys,
      ids and live flags as the unmasked kernel on the materialized matrix,
      and the non-skip top-k unchanged;
  (d) robustness: skip failures (first-launch check mismatch, exceptions,
      OOM, budget) never disable the posting kernel and the slice is
      re-scored identically; each compiled variant is checked once;
  (e) end to end through `run_compute`: outputs identical on / overhead /
      off / on-without-elision, with skipping engaged, the masked top-k used
      (CPU top-k availability patched in), warm-up gate, and auto-disabled for
      shared raw matrices.

As in `test_sparse_posting_triton_interpret.py`, `TRITON_INTERPRET` must be set
before `triton` is imported, so the interpreted tests run in a SUBPROCESS.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

_CHILD = os.environ.get("NOVA_BF_SPARSE_BLOCKSKIP_INTERPRET_CHILD") == "1"
_MIN_CHILD_TESTS = 180
_MIN_CHILD_TESTS_FAST = 150   # `-m "not slow"`: 290 minus 66 e2e minus 60 grid cells
_ENV_KEYS = ("NOVA_BF_NO_SPARSE_POSTING", "NOVA_BF_SPARSE_BLOCK_SKIP",
             "NOVA_BF_SPARSE_BLOCK_SKIP_SUB", "NOVA_BF_SPARSE_BLOCK_SKIP_ELIDE",
             "NOVA_BF_SPARSE_BLOCK_SKIP_BUDGET_MIB", "NOVA_BF_NO_TOPK_SELECT",
             "NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD", "NOVA_BF_SPARSE_BLOCK_SKIP_REPROBE",
             "NOVA_BF_SPARSE_BLOCK_SKIP_CHECK")


def test_the_interpreted_suite_passes(request):
    """Re-run this file with `TRITON_INTERPRET=1` set before `import triton`.

    The parent's `-m` expression is forwarded, so `pytest -m "not slow"`
    (the fast tier) also drops the child's `slow` cases -- the end-to-end
    runs and two thirds of the random-contract grid, ~33 of its ~44 serial
    minutes on CPU; a plain `pytest` still runs all of them. The child runs
    in parallel (tests/interpret_child.py)."""
    if _CHILD:
        pytest.skip("this process IS the interpreted child")
    import interpret_child

    markexpr = request.config.getoption("markexpr") or ""
    # Spread over xdist workers on the whole machine (~43 -> ~8 min at 8):
    # see tests/interpret_child.py, NOVA_BF_INTERPRET_WORKERS=1 for serial.
    proc = interpret_child.run(__file__, "NOVA_BF_SPARSE_BLOCKSKIP_INTERPRET_CHILD",
                               env_drop=_ENV_KEYS, markexpr=markexpr)
    assert proc.returncode == 0, (
        f"the interpreted block-skip suite failed:\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"(\d+) passed", proc.stdout)
    assert m, f"the child reported no passing tests:\n{proc.stdout}"
    # A floor for the full run and for the fast tier; any other `-m`
    # selection is the caller's choice.
    floor = {"": _MIN_CHILD_TESTS, "notslow": _MIN_CHILD_TESTS_FAST}.get(
        markexpr.replace(" ", ""))
    if floor is not None:
        assert int(m.group(1)) >= floor, (
            f"the interpreted child ran only {m.group(1)} tests:\n{proc.stdout}")


# --- ordinary (non-interpreted) process --------------------------------------

def test_config_knobs():
    from pydantic import ValidationError

    from nova_bf.config import ParamsConfig

    p = ParamsConfig()
    assert (p.sparse_block_skip, p.sparse_block_skip_sub,
            p.sparse_block_skip_elide) == ("on", 64, True)
    assert ParamsConfig(sparse_block_skip=True).sparse_block_skip == "on"
    assert ParamsConfig(sparse_block_skip=False).sparse_block_skip == "off"
    assert ParamsConfig(sparse_block_skip="overhead").sparse_block_skip == "overhead"
    for bad in ("yes", "skip"):
        with pytest.raises(ValidationError):
            ParamsConfig(sparse_block_skip=bad)
    for bad in (0, 48, 1024):
        with pytest.raises(ValidationError):
            ParamsConfig(sparse_block_skip_sub=bad)
    with pytest.raises(ValidationError):           # the audit knob is gone
        ParamsConfig(sparse_block_skip_audit_every=1)


def test_env_override_and_effective_mode(monkeypatch):
    from nova_bf import sparse_block_skip as bs

    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    bs.reset()
    try:
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP", "overhead")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_SUB", "32")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_ELIDE", "no")
        bs.configure("off", 64, True)
        u = bs.usage()
        assert (u["mode"], u["configured"], u["env_override"], u["sub"],
                u["elide"]) == ("overhead", "off", True, 32, False)
        for spell, want in (("1", "on"), ("TRUE", "on"), ("yes", "on"), ("on", "on"),
                            ("0", "off"), ("false", "off"), ("", "off"), ("no", "off")):
            monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP", spell)
            bs.configure("overhead")
            assert bs.mode() == want, spell
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP", "bogus")
        with pytest.raises(ValueError):
            bs.configure("on")
        monkeypatch.delenv("NOVA_BF_SPARSE_BLOCK_SKIP")
        for bad in ("overhead", "maybe"):
            monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_ELIDE", bad)
            with pytest.raises(ValueError):
                bs.configure("on")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_ELIDE", "off")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_SUB", "48")
        with pytest.raises(ValueError):
            bs.configure("on")
    finally:
        bs.reset()


def test_env_knobs_parse_strictly(monkeypatch):
    """R1/R3: env-only knobs are parsed once, in `configure`, strictly."""
    from nova_bf import sparse_block_skip as bs

    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    try:
        for name in ("CHECK",):
            key = f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}"
            for spell, want in (("1", True), ("true", True), ("yes", True), ("ON", True),
                                ("0", False), ("false", False), ("no", False),
                                ("off", False), ("", False)):
                monkeypatch.setenv(key, spell)
                bs.configure("on")
                got = bs._ENV["check"]
                assert got is want, (name, spell)
            for bad in ("maybe", "2", "overhead"):
                monkeypatch.setenv(key, bad)
                with pytest.raises(ValueError, match=key):
                    bs.configure("on")
            monkeypatch.delenv(key)
        for name, bad in (("MIN_DEAD", "abc"), ("MIN_DEAD", "1.5"), ("MIN_DEAD", "nan"),
                          ("REPROBE", "0"), ("REPROBE", "x"), ("BUDGET_MIB", "-1")):
            key = f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}"
            monkeypatch.setenv(key, bad)
            with pytest.raises(ValueError, match=key):
                bs.configure("on")
            monkeypatch.delenv(key)
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD", "0.3")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_REPROBE", "4")
        bs.configure("on")
        assert bs._min_dead() == 0.3 and bs._reprobe() == 4
        monkeypatch.delenv("NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD")
        bs.configure("on")
        assert bs._min_dead() == 0.15                 # P6 default
    finally:
        bs.reset()


def test_margin_constants():
    """F = 1 + (m+1) 2^-21 (exact in fp32), SIG = (m+1) 2^-120 (normal fp32),
    F = +inf past MAX_QUERY_NNZ."""
    import torch

    from nova_bf import sparse_block_skip as bs

    bs.reset()
    q = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]).to_sparse_csr()
    _, qtok, F, SIG = bs.query_plan(q)
    assert qtok.tolist() == [0, 1, 2]
    assert F.dtype is torch.float32 and SIG.dtype is torch.float32
    assert F.double().tolist() == [1 + 3 * 2.0 ** -21, 1 + 2 * 2.0 ** -21]
    assert SIG.double().tolist() == [3 * 2.0 ** -120, 2 * 2.0 ** -120]
    assert (SIG >= 2.0 ** -126).all()                 # normal: survives FTZ
    bs.reset()
    big = torch.ones((1, bs.MAX_QUERY_NNZ + 1)).to_sparse_csr()
    assert torch.isinf(bs.query_plan(big)[2]).all()
    bs.reset()


def test_skip_readers_gate():
    """`_sparse_skip_readers` admits a sparse batch whose readers are ALL
    unscaled dot searches with pruning on (filters allowed), with one
    threshold per selected row; any other metric, scaling or a threshold of
    the wrong height turns skipping off for the group."""
    import torch

    from nova_bf import sparse_block_skip as bs
    from nova_bf.compute import SparseCorpusBatch, _sparse_skip_readers
    from nova_bf.config import Filter, FilterCondition, SearchSpec

    bs.reset()
    batch = SparseCorpusBatch.__new__(SparseCorpusBatch)
    dot = SearchSpec(name="d", k=5, metric="dot", vector_type="sparse")
    cos = SearchSpec(name="c", k=5, metric="cosine", vector_type="sparse")
    filt = SearchSpec(name="f", k=5, metric="dot", vector_type="sparse",
                      filter=Filter(must=[FilterCondition(field="x", match="y")]))

    def gate(ms, specs, qsel=None, scale=None, prune=True, **k):
        n = len(specs)
        return _sparse_skip_readers(batch, ms, specs, qsel or [None] * n,
                                    scale or [None] * n, prune, **k)
    assert gate([0], [dot]) is None                                  # mode off
    bs.configure("on")
    try:
        assert gate([0], [dot]) == [0]
        assert gate([0], [dot], prune=False) is None
        assert gate([0, 1], [dot, dot]) == [0, 1]
        assert gate([0, 1], [dot, filt]) == [0, 1]
        assert gate([0], [filt]) == [0]
        assert gate([0], [cos]) is None
        assert gate([0, 1], [dot, cos]) is None
        assert gate([0], [dot], qsel=[[1]]) is None                  # heights unknown
        assert gate([0], [dot], scale=[1.0]) is None
        assert _sparse_skip_readers(object(), [0], [dot], [None], [None], True) is None
        g = bs.usage()["gates"]
        assert g["enabled"] == 4 and g["metric cosine"] == 2, g
        assert g["pruning off"] == 1 and g["query subset"] == 1 and g["query scaling"] == 1
        bs.disable("test")
        assert gate([0], [dot]) is None
        assert bs.usage()["gates"]["skipping disabled for the run"] == 1
    finally:
        bs.reset()


def test_skip_readers_gate_selection_heights():
    """With the query matrix and thresholds given, every reader's selection
    must be a `_row_selector` selector whose length is its threshold's height."""
    import torch

    from nova_bf import sparse_block_skip as bs
    from nova_bf.compute import (SparseCorpusBatch, _selection_len, _sparse_skip_readers,
                                 _whole_selection)
    from nova_bf.config import SearchSpec

    batch = SparseCorpusBatch.__new__(SparseCorpusBatch)
    dot = SearchSpec(name="d", k=5, metric="dot", vector_type="sparse")
    Q = torch.zeros(7, 3)
    bs.reset()
    bs.configure("on")
    try:
        def gate(sels, heights):
            n = len(sels)
            return _sparse_skip_readers(
                batch, list(range(n)), [dot] * n, sels, [None] * n, True, [Q] * n,
                [torch.zeros(h, dtype=torch.int64) for h in heights])
        assert gate([None], [7]) == [0]
        assert gate([slice(0, 7)], [7]) == [0]
        assert gate([slice(2, 5), torch.tensor([0, 3, 6])], [3, 3]) == [0, 1]
        assert gate([None, slice(0, 4)], [7, 4]) == [0, 1]
        for sel, h in ((None, 6), (slice(0, 7), 6), (slice(0, 7, 2), 4),
                       (slice(3, 9), 6), (slice(5, 2), 0), (slice(0, None), 7),
                       (torch.tensor([0, 1], dtype=torch.int32), 2), ([0, 1], 2)):
            assert gate([sel], [h]) is None, (sel, h)
        assert _selection_len(slice(None, 7), 7) == 7
        assert _whole_selection(None, None) and not _whole_selection(slice(0, 7), None)
        assert _whole_selection(slice(0, 7), 7) and not _whole_selection(slice(0, 6), 7)
    finally:
        bs.reset()


def test_skip_threshold_is_the_per_row_minimum():
    """`_skip_threshold`: per raw row, the lowest key among the readers
    selecting it (slice, index tensor or whole), the sentinel where none does."""
    import torch

    from nova_bf.compute import _skip_threshold
    from nova_bf.tiebreak import sentinel_key

    def keys(v):
        return torch.tensor(v, dtype=torch.int64)
    thr = [keys([50, 10, 70, 30, 90, 20]), keys([40, 60]), keys([5, 80, 95])]
    sel = [None, slice(2, 4), torch.tensor([1, 3, 5])]
    got = _skip_threshold([0, 1, 2], thr, sel, 6, "cpu")
    assert got.tolist() == [50, 5, 40, 30, 90, 20]
    got = _skip_threshold([1, 2], thr, sel, 6, "cpu")
    s = int(sentinel_key((1,), "cpu")[0])
    assert got.tolist() == [s, 5, 40, 60, s, 95]
    assert _skip_threshold([0], thr, sel, 6, "cpu") is thr[0]


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

    from nova_bf import sparse_block_skip as bs           # noqa: E402
    from nova_bf import sparse_posting_triton as sp       # noqa: E402
    from nova_bf import topk_triton as tt                 # noqa: E402
    from nova_bf.tiebreak import pack, sentinel_key       # noqa: E402

    POISON = 3e38

    @pytest.fixture(autouse=True)
    def _fresh(monkeypatch):
        """Fresh run state; the adaptive gate off (every test exercises the
        skip path unless it says otherwise); the uniqueness check on; and
        the score buffer POISONED (+3e38, which wins any selection) before
        every skip launch, so a reader of an unwritten elided cell that was
        not materialized changes the output."""
        for k in _ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD", "0")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_CHECK", "1")
        real = bs.try_skip

        def poisoned(q_csr, Cb, ptr, prow, pval, bptr, hot, n_rows, out, thr, cmin=None):
            out.fill_(POISON)
            return real(q_csr, Cb, ptr, prow, pval, bptr, hot, n_rows, out, thr, cmin)
        monkeypatch.setattr(bs, "try_skip", poisoned)
        sp.reset()
        sp.set_policy("auto")
        yield
        sp.reset()

    def _pos_csr(n, vocab, nnz_lo, nnz_hi, g, hot=(), dup_every=0):
        """Strictly positive CSR rows (the hot path's domain). `hot` tokens
        appear in most rows; `dup_every` > 0 repeats the previous row every so
        often (exact score ties)."""
        crow, cols, vals = [0], [], []
        prev = None
        for i in range(n):
            if dup_every and prev is not None and i % dup_every == 0:
                c, v = prev
            else:
                k = int(torch.randint(nnz_lo, nnz_hi + 1, (1,), generator=g))
                c = set(torch.randperm(vocab, generator=g)[:k].tolist())
                for h in hot:
                    if torch.rand(1, generator=g) < 0.9:
                        c.add(h)
                c = sorted(c)
                v = (torch.rand(len(c), generator=g) + 0.01).tolist()
            prev = (c, v)
            cols += c
            vals += v
            crow.append(len(cols))
        return torch.sparse_csr_tensor(
            torch.tensor(crow, dtype=torch.int64), torch.tensor(cols, dtype=torch.int64),
            torch.tensor(vals, dtype=torch.float32), size=(n, vocab))

    def _setup(monkeypatch, db, sub, hot_tokens, elide=True, mode="on"):
        monkeypatch.setattr(sp, "DOC_BLOCK", db)
        monkeypatch.setattr(sp, "HOT_TOKENS", hot_tokens)
        bs.configure(mode, sub, elide)

    def _kth(base, k):
        return torch.topk(base, min(k, base.shape[1]), dim=1).values[:, -1].contiguous()

    def _thr_key(scores):
        return pack(scores.float().contiguous(),
                    torch.zeros(scores.shape[0], dtype=torch.int64))

    def _base(q, c, cmin):
        return sp.scores(q, c, cmin=cmin)              # no skip_thr: unskipped

    def _skip(q, c, cmin, thr_key):
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=thr_key, return_skip=True)
        if mk is not None:
            mk.materialize()
        return out, mk

    def _assert_contract(out, base, thr):
        """Kept cells bit-identical; dropped cells strictly below thr; every
        cell >= thr kept; no-overlap cells stay -inf."""
        assert out.dtype == torch.float32 and out.shape == base.shape
        kept = ~torch.isneginf(out)
        assert torch.equal(out[kept].view(torch.int32), base[kept].view(torch.int32)), \
            "a kept cell differs from the unskipped kernel"
        dropped = torch.isneginf(out) & ~torch.isneginf(base)
        t = thr[:, None].expand_as(base)
        assert (base[dropped] < t[dropped]).all(), "dropped a cell at/above thr"
        ge = base >= t
        assert torch.equal(out[ge], base[ge])
        assert torch.isneginf(out[torch.isneginf(base)]).all()
        return int(dropped.sum())

    SHAPES = [(64, 16), (64, 64), (512, 64), (256, 8), (128, 32)]  # (256, 8): the 32-flag maximum

    # --- (a) skip posting kernel ---------------------------------------------

    # The full 5 x 2 x 3 x 3 grid (90 cells, ~5.8 s each interpreted) runs in
    # the full tier. `k` only moves the threshold -- the launches are the same
    # for every k -- so the fast tier keeps a Latin-square third of it (one k
    # per (shape, hot, n_rows) cell, k index = (shape + hot + n_rows) % 3):
    # every shape x hot x n_rows cell still runs, every (shape, k), (hot, k)
    # and (n_rows, k) pair still occurs, and every shape still sees both
    # elision modes. The other 60 cells are marked `slow`.
    _RC_HOT, _RC_ROWS, _RC_K = [64, 3], [300, 64 * 5 + 1, 1000 + 37], [1, 7, 40]
    _RC_GRID = [
        pytest.param(db, sub, h, n, k, id=f"{k}-{n}-{h}-{db}-{sub}",
                     marks=() if ki == (si + hi + ni) % 3 else pytest.mark.slow)
        for si, (db, sub) in enumerate(SHAPES)
        for hi, h in enumerate(_RC_HOT)
        for ni, n in enumerate(_RC_ROWS)
        for ki, k in enumerate(_RC_K)
    ]

    @pytest.mark.parametrize("db,sub,hot_tokens,n_rows,k", _RC_GRID)
    def test_random_contract(monkeypatch, db, sub, hot_tokens, n_rows, k):
        elide = (n_rows + k) % 2 == 0          # both elision modes across the grid
        _setup(monkeypatch, db, sub, hot_tokens, elide)
        g = torch.Generator().manual_seed(db * 7 + sub + hot_tokens + n_rows + k)
        q = _pos_csr(24, 60, 1, 6, g, hot=(0, 1))
        c = _pos_csr(n_rows, 60, 1, 9, g, hot=(0,), dup_every=11)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr = _kth(base, k)
        out, mk = _skip(q, c, cmin, _thr_key(thr))
        assert mk is not None, bs.usage()
        _assert_contract(out, base, thr)
        u = bs.usage()
        assert u["slices"] == 1 and u["declines"] == {} and u["check_failures"] == 0
        assert u["subblocks"] == 24 * -(-n_rows // sub)
        assert u["elided_slices"] == int(elide)

    def test_skips_a_lot_with_a_mature_threshold(monkeypatch):
        _setup(monkeypatch, 512, 16, 64)
        g = torch.Generator().manual_seed(5)
        q = _pos_csr(30, 400, 2, 6, g)
        c = _pos_csr(2048, 400, 2, 8, g)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr = _kth(base, 1)
        out, mk = _skip(q, c, cmin, _thr_key(thr))
        assert _assert_contract(out, base, thr) > 0
        u = bs.usage()
        assert u["dead"] == int((mk.live == 0).sum()) > 0
        assert u["checks"] == 1 and u["checked_dead_cells"] > 0

    def _master_and_skip(monkeypatch, db, sub, n_rows, pattern, elide, seed):
        """Run master's `_posting_scores_hot` and the skip kernel (given an
        explicit flag pattern) on POISONED buffers; return both outputs and the
        dead-cell mask."""
        from types import SimpleNamespace

        _setup(monkeypatch, db, sub, 3, elide)
        g = torch.Generator().manual_seed(seed)
        q = _pos_csr(12, 60, 1, 6, g, hot=(0, 1))
        c = _pos_csr(n_rows, 60, 1, 9, g, hot=(0,))
        ptr, prow, pval = sp.corpus_postings(c)
        bptr = sp.block_ptr(ptr, prow, 60, n_rows, db)
        hotidx, n_hot = sp.hot_plan(q)
        hot = (hotidx, sp.hot_table(ptr, prow, pval, hotidx, n_hot, n_rows, 60))
        master = torch.full((12, n_rows), POISON)
        sp._launch(q, ptr, prow, pval, n_rows, master, bptr, hot)
        n_sub = -(-n_rows // sub)
        nsb = db // sub
        sb = torch.arange(n_sub)
        if pattern == "all_live":
            live = torch.ones((12, n_sub), dtype=torch.int8)
        elif pattern == "all_dead":
            live = torch.zeros((12, n_sub), dtype=torch.int8)
        elif pattern == "partly":             # every program partly dead
            live = ((sb % nsb) % 2 == 0).to(torch.int8).expand(12, -1).contiguous()
        else:                                 # mixed: whole programs live/dead/partly
            prog = sb // nsb
            lv = torch.where(prog % 3 == 0, 1, torch.where(prog % 3 == 1, 0,
                                                           (sb % 3 != 1).long()))
            live = lv.to(torch.int8).expand(12, -1).contiguous()
            live = live.clone()
            live[::2] = 1 - live[::2]         # rows differ too
        ctx = SimpleNamespace(klive=live, n_sub=n_sub, sub=sub, elided=elide)
        out = torch.full((12, n_rows), POISON)
        bs.launch(q, prow, pval, n_rows, out, hot, bptr, ctx)
        dead = live.repeat_interleave(sub, dim=1)[:, :n_rows] == 0
        return master, out, dead

    @pytest.mark.parametrize("pattern", ["all_live", "partly", "all_dead", "mixed"])
    @pytest.mark.parametrize("elide", [True, False])
    @pytest.mark.parametrize("db,sub,n_rows", [(64, 16, 300), (512, 64, 1037),
                                               (128, 8, 257), (512, 16, 1100)])
    def test_skip_kernel_paths_bitwise_vs_master(monkeypatch, pattern, elide, db, sub,
                                                 n_rows):
        """All three program paths (all live = master's code, partly dead =
        tile + bitmask, all dead) against master's hot kernel on poisoned
        buffers: live cells bit-identical; dead cells untouched (elided) or
        -inf (not elided)."""
        master, out, dead = _master_and_skip(monkeypatch, db, sub, n_rows, pattern,
                                             elide, seed=db + sub + n_rows)
        assert torch.equal(out[~dead].view(torch.int32), master[~dead].view(torch.int32))
        if elide:
            assert (out[dead] == POISON).all()
        else:
            assert torch.isneginf(out[dead]).all()

    def test_more_than_32_sub_blocks_declines(monkeypatch):
        _setup(monkeypatch, 512, 8, 64)                  # 64 sub-blocks per program
        q, c, cmin = _slice(42)
        base = _base(q, c, cmin)
        out, mk = _skip(q, c, cmin, _thr_key(_kth(base, 1)))
        assert mk is None and torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert any("more than 32" in k for k in bs.usage()["declines"])

    @pytest.mark.parametrize("db,sub", [(64, 16), (512, 64)])
    def test_elided_cells_are_never_written(monkeypatch, db, sub):
        """With elision, the kernel writes no dead cell (a finite sentinel
        prefill survives bit-exactly on the dead sub-blocks; a stray hot store
        or cold fma would change it) and live cells match."""
        _setup(monkeypatch, db, sub, 3)
        g = torch.Generator().manual_seed(6)
        q = _pos_csr(16, 60, 1, 6, g, hot=(0,))
        c = _pos_csr(777, 60, 1, 9, g, hot=(0,))
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr_key = _thr_key(_kth(base, 1))
        ptr, prow, pval = sp.corpus_postings(c)
        bptr = sp.block_ptr(ptr, prow, 60, 777, db)
        hotidx, n_hot = sp.hot_plan(q)
        hot = (hotidx, sp.hot_table(ptr, prow, pval, hotidx, n_hot, 777, 60))
        out = torch.full((16, 777), 12345.0)
        ctx = bs.prepare(q, ptr, prow, pval, hot, 777, thr_key, out)
        bs.launch(q, prow, pval, 777, out, hot, bptr, ctx)
        dead = ctx.cells(ctx.live == 0)
        assert dead.any() and (~dead).any()
        assert (out[dead] == 12345.0).all()
        assert torch.equal(out[~dead].view(torch.int32), base[~dead].view(torch.int32))

    @pytest.mark.parametrize("db,sub", SHAPES)
    def test_overhead_mode_is_identical_everywhere(monkeypatch, db, sub):
        _setup(monkeypatch, db, sub, 4, mode="overhead")
        g = torch.Generator().manual_seed(11 + sub)
        q = _pos_csr(20, 80, 1, 6, g, hot=(0,))
        c = _pos_csr(777, 80, 1, 9, g, hot=(0,))
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        out, mk = _skip(q, c, cmin, _thr_key(_kth(base, 1)))
        assert mk is not None and not mk.elided and bool((mk.klive == 1).all())
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        u = bs.usage()
        assert u["overhead_slices"] == 1 and u["slices"] == 0 and u["dead"] > 0
        assert u["checked_dead_cells"] > 0          # would-be-dead cells checked

    @pytest.mark.parametrize("db,sub", SHAPES)
    def test_all_live_sentinel(monkeypatch, db, sub):
        _setup(monkeypatch, db, sub, 64)
        g = torch.Generator().manual_seed(12)
        q = _pos_csr(16, 50, 1, 5, g)
        c = _pos_csr(333, 50, 1, 7, g)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        out, mk = _skip(q, c, cmin, sentinel_key((16,), "cpu"))
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["dead"] == 0 and bool((mk.rowany == 1).all())

    @pytest.mark.parametrize("db,sub", SHAPES)
    @pytest.mark.parametrize("elide", [False, True])
    def test_all_dead(monkeypatch, db, sub, elide):
        _setup(monkeypatch, db, sub, 3, elide)
        g = torch.Generator().manual_seed(13)
        q = _pos_csr(10, 40, 1, 5, g, hot=(0,))
        c = _pos_csr(500, 40, 1, 7, g, hot=(0,))
        cmin = float(c.values().min())
        out, mk = _skip(q, c, cmin, _thr_key(torch.full((10,), 1e6)))
        assert torch.isneginf(out).all() and bool((mk.rowany == 0).all())
        u = bs.usage()
        assert u["dead"] == u["subblocks"] == 10 * -(-500 // sub)
        assert u["checks"] == 1 and u["check_failures"] == 0

    def test_mixed_full_and_unfull_rows(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        g = torch.Generator().manual_seed(14)
        q = _pos_csr(12, 50, 1, 5, g)
        c = _pos_csr(400, 50, 1, 7, g)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr_s = _kth(base, 1)
        key = _thr_key(thr_s)
        unfull = torch.arange(12) % 3 == 0
        key[unfull] = sentinel_key((int(unfull.sum()),), "cpu")
        out, _ = _skip(q, c, cmin, key)
        assert torch.equal(out[unfull].view(torch.int32), base[unfull].view(torch.int32))
        assert _assert_contract(out, base, torch.where(unfull, float("-inf"), thr_s)) > 0

    def test_query_with_fewer_than_k_overlaps(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        g = torch.Generator().manual_seed(15)
        q = torch.cat([_pos_csr(8, 50, 1, 4, g).to_dense(), torch.zeros(8, 1)], 1)
        q[0] = 0
        q[0, 50] = 1.0
        q = q.to_sparse_csr()
        c = torch.cat([_pos_csr(300, 50, 1, 7, g).to_dense(), torch.zeros(300, 1)], 1)
        c[[5, 100, 250], 50] = 0.3
        c = c.to_sparse_csr()
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr = _kth(base, 10)
        assert torch.isneginf(thr[0])
        out, _ = _skip(q, c, cmin, _thr_key(thr))
        assert torch.equal(out[0].view(torch.int32), base[0].view(torch.int32))
        assert int((~torch.isneginf(out[0])).sum()) == 3
        _assert_contract(out, base, thr)

    @pytest.mark.parametrize("sub", [8, 16, 64])
    def test_ties_exactly_at_threshold(monkeypatch, sub):
        _setup(monkeypatch, 64, sub, 64)
        g = torch.Generator().manual_seed(16)
        q = _pos_csr(10, 40, 2, 5, g)
        c = _pos_csr(640, 40, 2, 6, g, dup_every=2)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr = _kth(base, 3)
        out, _ = _skip(q, c, cmin, _thr_key(thr))
        _assert_contract(out, base, thr)
        ties = base == thr[:, None]
        assert int(ties.sum()) > thr.numel()
        assert torch.equal(out[ties], base[ties])

    def _round_up_case(g, n_q=300, m=9):
        """Every corpus row is ONE document, so the block-max bound is tight
        (UB == exact score); some queries' fp32 sums round ABOVE the exact
        sum. With thr = the kernel's own score, only the margin keeps the
        tie live."""
        vocab = m + 1
        doc = torch.rand(vocab, generator=g, dtype=torch.float64).float() + 0.01
        Q = torch.rand(n_q, vocab, generator=g) + 0.01
        Q[:, -1] = 0
        return Q, doc

    @pytest.mark.parametrize("hot_tokens", [64, 2])
    def test_margin_covers_fp32_round_up(monkeypatch, hot_tokens):
        _setup(monkeypatch, 64, 16, hot_tokens)
        g = torch.Generator().manual_seed(17)
        Q, doc = _round_up_case(g)
        C = doc.expand(200, -1).clone()
        q, c = Q.to_sparse_csr(), C.to_sparse_csr()
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        up = base[:, 0].double() > (Q.double() @ doc.double())
        assert int(up.sum()) >= 5
        out, _ = _skip(q, c, cmin, _thr_key(base[:, 0].clone()))
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["dead"] == 0

    def test_margin_covers_kernel_vs_bound_order(monkeypatch):
        """The kernel sums hot tokens first, the bound in query-column order.
        Make the HIGH token ids hot so the orders differ, every corpus row one
        document (tight bound), thr = the kernel's own score. Positive
        control: with the margin removed (F = 1, SIG = 0) some tie IS proven
        dead; with the real margin none is."""
        _setup(monkeypatch, 64, 16, 4)
        g = torch.Generator().manual_seed(171)
        m = 9
        Q = torch.rand(400, m + 1, generator=g) * 3 + 0.01
        Q[:, -1] = 0
        extra = torch.zeros(60, m + 1)
        extra[:, 5:9] = torch.rand(60, 4, generator=g) + 0.01   # tokens 5..8 hot
        Q = torch.cat([Q, extra])
        doc = torch.rand(m + 1, generator=g) * 3 + 0.01
        C = doc.expand(128, -1).clone()
        q, c = Q.to_sparse_csr(), C.to_sparse_csr()
        cmin = float(c.values().min())
        hotidx, _ = sp.hot_plan(q)
        assert sorted(torch.nonzero(hotidx >= 0).flatten().tolist()) == [5, 6, 7, 8]
        base = _base(q, c, cmin)
        thr = _thr_key(base[:, 0].clone())
        real = bs.query_plan

        def no_margin(qc):
            qmap, qtok, F, SIG = real(qc)
            return qmap, qtok, torch.ones_like(F), torch.zeros_like(SIG)
        monkeypatch.setattr(bs, "query_plan", no_margin)
        ptr, prow, pval = sp.corpus_postings(c)
        hot = (hotidx, sp.hot_table(ptr, prow, pval, hotidx, 4, 128, m + 1))
        ctx = bs.prepare(q, ptr, prow, pval, hot, 128, thr, torch.empty(1))
        assert int((ctx.live == 0).sum()) > 0, "control: no order-induced rounding gap"
        monkeypatch.setattr(bs, "query_plan", real)
        out, _ = _skip(q, c, cmin, thr)
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["dead"] == 0

    def test_one_huge_weight_keeps_only_its_sub_block(monkeypatch):
        _setup(monkeypatch, 512, 64, 64)
        g = torch.Generator().manual_seed(18)
        q = _pos_csr(6, 30, 2, 4, g, hot=(0,))
        C = _pos_csr(1024, 30, 2, 5, g, hot=(0,)).to_dense()
        C[700, 0] = 1e4                                  # sub-block 10
        c = C.to_sparse_csr()
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr = _kth(base, 1)
        out, _ = _skip(q, c, cmin, _thr_key(thr))
        _assert_contract(out, base, thr)
        has0 = q.to_dense()[:, 0] > 0
        assert has0.any()
        for i in torch.nonzero(has0).flatten().tolist():
            assert torch.isneginf(out[i, :640]).all() and torch.isneginf(out[i, 704:]).all()
            assert out[i, 700] == base[i, 700]

    # --- (b) the bound ---------------------------------------------------------

    @pytest.mark.parametrize("sub", [1, 16, 64])
    @pytest.mark.parametrize("hot_tokens", [64, 3, 0])
    def test_block_max_is_exact(monkeypatch, sub, hot_tokens):
        """BM over hot tokens (dense-table reduce), cold query tokens (posting
        walk + int32 atomic max) and with no hot tokens at all; tokens no
        query uses are absent; the last sub-block is ragged."""
        monkeypatch.setattr(sp, "HOT_TOKENS", hot_tokens)
        g = torch.Generator().manual_seed(41 + sub + hot_tokens)
        q0 = _pos_csr(20, 60, 1, 8, g, hot=(0,))
        q = torch.sparse_csr_tensor(q0.crow_indices(), q0.col_indices(), q0.values(),
                                    size=(20, 70))
        c = _pos_csr(301, 70, 1, 9, g, hot=(0, 65))
        n_rows, vocab = 301, 70
        ptr, prow, pval = sp.corpus_postings(c)
        if hot_tokens:
            hotidx, n_hot = sp.hot_plan(q)
            table = sp.hot_table(ptr, prow, pval, hotidx, n_hot, n_rows, vocab)
        else:
            hotidx = torch.full((vocab,), -1, dtype=torch.int32)
            table = torch.zeros(1)
        qmap, qtok, _, _ = bs.query_plan(q)
        n_sub = -(-n_rows // sub)
        bm = torch.zeros(qtok.numel() * n_sub, dtype=torch.int32)
        bs._block_max[(qtok.numel(),)](qtok, hotidx, table, ptr, prow, pval, bm,
                                       n_rows, n_sub, SUB=sub, BLOCK_P=bs.BLOCK_P)
        pad = torch.zeros(n_sub * sub, vocab)
        pad[:n_rows] = c.to_dense()
        want = pad.view(n_sub, sub, vocab).amax(1).T[qtok]
        assert torch.equal(bm.view(torch.float32).view(-1, n_sub), want)

    def _bound_once(qv, bmv, thr):
        """Run `_bound` for one single-term query: ub = qv * bmv."""
        bs.reset()
        q = torch.sparse_csr_tensor(torch.tensor([0, 1]), torch.tensor([0]),
                                    torch.tensor([qv], dtype=torch.float32), size=(1, 1))
        qmap, qtok, F, SIG = bs.query_plan(q)
        bm = torch.tensor([bmv], dtype=torch.float32)
        live = torch.empty((1, 1), dtype=torch.int8)
        rowany = torch.empty(1, dtype=torch.int8)
        cnt = torch.empty(1, dtype=torch.int32)
        bs._bound[(1,)](q.crow_indices(), q.col_indices(), q.values(), qmap, bm, F, SIG,
                        _thr_key(torch.tensor([thr], dtype=torch.float32)), live,
                        rowany, cnt, 1, NSP=1)
        return bool(live[0, 0] == 0), F, SIG

    def test_bound_test_is_strict():
        """dead iff fl(fl(ub F) + SIG) < thr, strictly: at exact equality the
        sub-block is live, one ulp above it is dead."""
        _, F, SIG = _bound_once(1.0, 0.75, 0.0)
        t = (torch.tensor([0.75], dtype=torch.float32) * F + SIG)[0]
        assert not _bound_once(1.0, 0.75, float(t))[0]
        up = torch.nextafter(t, torch.tensor(9.0))
        assert _bound_once(1.0, 0.75, float(up))[0]
        assert not _bound_once(1.0, 0.75, float("-inf"))[0]
        assert not _bound_once(1.0, 0.75, float("nan"))[0]
        # the margin really inflates: a thr one ulp above the raw product is live
        assert not _bound_once(1.0, 0.75, float(torch.nextafter(
            torch.tensor(0.75), torch.tensor(9.0))))[0]

    # --- (c) mask-aware top-k ----------------------------------------------------

    @pytest.mark.parametrize("sub", [16, 64])
    @pytest.mark.parametrize("k", [1, 5, 60])
    @pytest.mark.parametrize("sel", [True, False])
    def test_masked_cutfill_matches_unmasked(monkeypatch, sub, k, sel):
        """`_cutfill` with the skip mask over an UNWRITTEN (NaN-filled) dead
        region equals the unmasked kernel over the materialized matrix: live
        flags, and every live row's keys and ids."""
        _setup(monkeypatch, 64, sub, 64)
        if not sel:
            monkeypatch.setenv("NOVA_BF_NO_TOPK_SELECT", "1")
        g = torch.Generator().manual_seed(50 + sub + k)
        q = _pos_csr(30, 80, 1, 5, g)
        c = _pos_csr(450, 80, 1, 8, g, dup_every=7)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr_key = _thr_key(_kth(base, 3))
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=thr_key, return_skip=True)
        assert mk.elided and (mk.rowany == 1).any()
        garbage = out.clone()
        garbage[mk.cells(mk.klive == 0)] = POISON
        ordv = torch.arange(450, dtype=torch.int64) * 3 + 1
        enc = torch.arange(450, dtype=torch.int64) + 1000
        k1, i1, l1 = tt.topk(garbage, ordv, k, thr=thr_key, enc=enc, lmask=mk)
        mk.materialize()
        k0, i0, l0 = tt.topk(out, ordv, k, thr=thr_key, enc=enc)
        assert torch.equal(l0, l1)
        lv = l0.bool()
        assert lv.any()
        o0, o1 = k0[lv].argsort(1), k1[lv].argsort(1)
        assert torch.equal(k0[lv].gather(1, o0), k1[lv].gather(1, o1))
        assert torch.equal(i0[lv].gather(1, o0), i1[lv].gather(1, o1))
        assert bs.usage()["masked_topk"] == 1

    def test_masked_cutfill_fully_dead_rows(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        g = torch.Generator().manual_seed(59)
        q = _pos_csr(10, 40, 1, 5, g)
        c = _pos_csr(200, 40, 1, 6, g)
        cmin = float(c.values().min())
        base = _base(q, c, cmin)
        thr = _kth(base, 1)
        thr[::2] = 1e6                                   # even rows: all dead
        key = _thr_key(thr)
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=key, return_skip=True)
        assert bool((mk.rowany[::2] == 0).all())
        out[mk.cells(mk.klive == 0)] = POISON
        _, _, live = tt.topk(out, torch.arange(200, dtype=torch.int64), 3, thr=key,
                             lmask=mk)
        assert bool((live[::2] == 0).all()) and bool((live[1::2] == 1).all())

    def test_unmasked_cutfill_matches_portable_pack_topk():
        """The top-k kernel without a skip mask (every non-skip caller) still
        selects what the portable packed-key path selects."""
        from nova_bf.tiebreak import pack_topk

        g = torch.Generator().manual_seed(60)
        s = torch.rand(20, 300, generator=g)
        s[s < 0.3] = float("-inf")
        s[3] = s[4]
        ordv = torch.randperm(300, generator=g).to(torch.int64)
        thr = _thr_key(torch.full((20,), 0.97))
        k0, i0, l0 = tt.topk(s, ordv, 7, thr=thr)
        k1, i1, l1 = pack_topk(s, ordv, 7, thr=thr)       # portable on CPU
        assert torch.equal(l0, l1)
        for r in torch.nonzero(l0).flatten().tolist():
            lo = int((thr[r] >> 32) << 32)
            assert set(k0[r][k0[r] >= lo].tolist()) == set(k1[r][k1[r] >= lo].tolist())
        k0, i0, _ = tt.topk(s, ordv, 7)
        k1, i1, _ = pack_topk(s, ordv, 7)
        assert torch.equal(k0.sort(1).values, k1.sort(1).values)

    def test_lmask_contract_is_enforced():
        g = torch.Generator().manual_seed(61)
        s = torch.rand(4, 64, generator=g)

        class M:
            klive = torch.ones((4, 4), dtype=torch.int8)
            rowany = torch.ones(4, dtype=torch.int8)
            sub, n_rows = 16, 64

            def note_masked(self):
                pass
        ordv = torch.arange(64, dtype=torch.int64)
        with pytest.raises(ValueError):
            tt.topk(s, ordv, 3, lmask=M())                        # no threshold
        with pytest.raises(ValueError):
            tt.topk(s, ordv, 3, thr=_thr_key(torch.zeros(4)), lmask=M(),
                    scale=torch.ones(4))
        bad = M()
        bad.n_rows = 63
        with pytest.raises(ValueError):
            tt.topk(s, ordv, 3, thr=_thr_key(torch.zeros(4)), lmask=bad)

    def test_portable_pack_topk_materializes():
        from nova_bf.tiebreak import pack_topk

        class M:
            done = 0

            def materialize(self):
                M.done += 1
        pack_topk(torch.rand(3, 20), torch.arange(20, dtype=torch.int64), 4, lmask=M())
        assert M.done == 1

    # --- (d) robustness ----------------------------------------------------------

    def _slice(seed=21, n_rows=300):
        g = torch.Generator().manual_seed(seed)
        q = _pos_csr(8, 30, 1, 4, g)
        c = _pos_csr(n_rows, 30, 1, 5, g)
        return q, c, float(c.values().min())

    def test_unsound_bound_is_caught_and_only_skipping_turns_off(monkeypatch):
        """A too-small bound (F scaled down): the first-launch check sees a skipped cell
        whose reference reaches thr, switches SKIPPING off, the slice is
        re-scored unskipped (bit-identical), and the posting kernel stays on."""
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice()
        base = _base(q, c, cmin)
        real = bs.query_plan

        def small(qc):
            qmap, qtok, F, SIG = real(qc)
            return qmap, qtok, F * 0.5, SIG
        monkeypatch.setattr(bs, "query_plan", small)
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=_thr_key(_kth(base, 1)),
                            return_skip=True)
        assert mk is None
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        u = bs.usage()
        assert u["check_failures"] == 1 and "first-launch check" in u["disabled"]
        assert sp.kernel_off() is None
        out2, _mk = sp.scores(q, c, cmin=cmin, skip_thr=_thr_key(_kth(base, 1)), return_skip=True)
        assert _mk is None              # fell back: fully written
        assert torch.equal(out2.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["declines"].get("disabled") == 1

    # --- model probe (adversarial check of the bound's fp model) ------------

    def _probe_run(monkeypatch, db=64, sub=16, elide=True, mode="on", seed=31, n_rows=300):
        _setup(monkeypatch, db, sub, 64, elide=elide, mode=mode)
        q, c, cmin = _slice(seed, n_rows)
        base = _base(q, c, cmin)
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=_thr_key(_kth(base, 2)),
                            return_skip=True)
        if mk is not None:
            mk.materialize()
        return out, mk, base

    @pytest.mark.parametrize("db,sub,elide,mode,n_rows", [
        (64, 16, True, "on", 300), (64, 32, False, "on", 300),
        (128, 16, True, "overhead", 300), (64, 64, True, "on", 70),
        (64, 16, True, "on", 1), (64, 16, True, "on", 17)])
    def test_model_probe_passes_once_per_variant(monkeypatch, db, sub, elide, mode, n_rows):
        """A clean kernel passes the probe: once per compiled variant, every
        probe cell inside the model, worst error at most the model's (round
        to nearest sits near half of it), skipping stays on."""
        out, mk, base = _probe_run(monkeypatch, db, sub, elide, mode, n_rows=n_rows)
        u = bs.usage()
        assert u["disabled"] is None and mk is not None
        assert u["model_probes"] == 1 and u["model_probe_failures"] == 0
        assert u["model_probe_tokens"] > 900
        assert 0.0 < u["model_probe_worst"] <= 1.0
        _assert_contract(out, base, _kth(base, 2))
        _probe_run(monkeypatch, db, sub, elide, mode, seed=32, n_rows=n_rows)
        assert bs.usage()["model_probes"] == 1             # same variant: not re-probed

    @pytest.mark.parametrize("frac,sig", [(0.0, 1.0), (2.0, 1.0), (5.0, 1.0), (8.0, 0.0),
                                          (8.0, 0.25)])
    def test_model_probe_catches_a_kernel_using_less_margin(monkeypatch, frac, sig):
        """The bound kernel run with a smaller F than the proof's (even 5/8 of
        it, SIG intact), or the proof's F with SIG dropped or quartered,
        violates (B): the probe turns skipping off before first_check.
        (B) uses the model's worst case, so it flags an F short by about
        2(m+1)u or more: a quarter of the proof's 8(m+1)u."""
        real = bs.make_plan

        def weak(qc):
            qmap, qtok, F, SIG = real(qc)
            m = qc.crow_indices().diff().double()
            return qmap, qtok, (1 + frac * (m + 1) * 2.0 ** -24).float(), SIG * sig
        monkeypatch.setattr(bs, "make_plan", weak)
        out, mk, base = _probe_run(monkeypatch)
        u = bs.usage()
        assert mk is None and "model probe" in u["disabled"]
        assert u["model_probe_failures"] == 1 and u["checks"] == 0
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))

    def test_model_probe_catches_posting_arithmetic_outside_the_model(monkeypatch):
        """Posting scores 2^-16 relative high (a coarser accumulator): (A)."""
        real = bs.launch
        n_probe = len(bs._probe_rows())

        def coarse(q_, prow, pval, n_rows, out, hot, bptr, ctx):
            h = real(q_, prow, pval, n_rows, out, hot, bptr, ctx)
            if out.shape[0] == n_probe:
                out.mul_(1 + 2.0 ** -16)
            return h
        monkeypatch.setattr(bs, "launch", coarse)
        out, mk, base = _probe_run(monkeypatch)
        assert mk is None and "posting kernel outside the model" in bs.usage()["disabled"]
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))

    def test_model_probe_catches_a_low_block_max(monkeypatch):
        """BM one ulp under the true block maximum: (D)."""
        real = bs.build_bm

        def low(*a, **k):
            bm = real(*a, **k)
            bm[bm > 0] -= 1
            return bm
        monkeypatch.setattr(bs, "build_bm", low)
        out, mk, base = _probe_run(monkeypatch)
        assert mk is None and "exact block maximum" in bs.usage()["disabled"]

    def test_model_probe_catches_skip_path_divergence(monkeypatch):
        """A live cell that differs only on the partly-dead path."""
        real = bs.launch
        n_probe = len(bs._probe_rows())

        def diverge(q_, prow, pval, n_rows, out, hot, bptr, ctx):
            h = real(q_, prow, pval, n_rows, out, hot, bptr, ctx)
            if out.shape[0] == n_probe and bool((ctx.klive == 0).any()):
                live = ~torch.isneginf(out) & ~torch.isnan(out)
                out[live] = torch.nextafter(out[live], torch.tensor(float("inf")))
            return h
        monkeypatch.setattr(bs, "launch", diverge)
        out, mk, base = _probe_run(monkeypatch)
        assert mk is None and "skip paths" in bs.usage()["disabled"]

    def test_bound_row_flags_must_match_live(monkeypatch):
        """A ROWANY that drops a live row (the masked top-k would skip it) is
        caught by the probe; with the probe bypassed, by first_check."""
        real = bs.run_bound

        def bad(*a, **k):
            live, rowany, dcnt = real(*a, **k)
            return live, torch.zeros_like(rowany), dcnt
        monkeypatch.setattr(bs, "run_bound", bad)
        out, mk, base = _probe_run(monkeypatch)
        assert mk is None and "ROWANY" in bs.usage()["disabled"]
        assert bs.usage()["model_probe_failures"] == 1
        sp.reset()
        monkeypatch.setattr(bs, "model_probe", lambda *a, **k: None)
        out, mk, base = _probe_run(monkeypatch)
        u = bs.usage()
        assert mk is None and "ROWANY" in u["disabled"] and u["check_failures"] == 1
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))

    def test_model_probe_refuses_a_variant_it_cannot_reproduce(monkeypatch):
        real = bs.compile_key

        def other(*a, plan=None, **k):
            key = real(*a, plan=plan, **k)
            return key + ("probe",) if plan is not None else key
        monkeypatch.setattr(bs, "compile_key", other)
        out, mk, base = _probe_run(monkeypatch)
        assert mk is None and "could not reproduce" in bs.usage()["disabled"]
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))

    def test_live_cell_corruption_is_caught(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(22)
        base = _base(q, c, cmin)
        real = bs.launch

        def corrupt(q_, prow, pval, n_rows, out, hot, bptr, ctx):
            real(q_, prow, pval, n_rows, out, hot, bptr, ctx)
            live = ctx.cells(ctx.klive == 1) & ~torch.isneginf(out) & ~torch.isnan(out)
            out[live] += 1.0
        monkeypatch.setattr(bs, "launch", corrupt)
        out, _mk = sp.scores(q, c, cmin=cmin, skip_thr=_thr_key(_kth(base, 2)), return_skip=True)
        assert _mk is None              # fell back: fully written
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["disabled"] is not None and sp.kernel_off() is None

    def test_exception_in_skip_prep_falls_back_to_unskipped(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(23)
        base = _base(q, c, cmin)

        def boom(*a, **k):
            raise IndexError("bound broke")
        monkeypatch.setattr(bs, "prepare", boom)
        out, _mk = sp.scores(q, c, cmin=cmin, skip_thr=_thr_key(_kth(base, 1)), return_skip=True)
        assert _mk is None              # fell back: fully written
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert "bound broke" in bs.usage()["disabled"] and sp.kernel_off() is None

    def test_oom_in_skip_prep_declines_the_slice_only(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(24)
        base = _base(q, c, cmin)
        real = bs.prepare
        calls = []

        def oom_once(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 3 GiB")
            return real(*a, **k)
        monkeypatch.setattr(bs, "prepare", oom_once)
        thr = _thr_key(_kth(base, 1))
        out, _mk = sp.scores(q, c, cmin=cmin, skip_thr=thr, return_skip=True)
        assert _mk is None              # fell back: fully written
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        u = bs.usage()
        assert u["declines"] == {"out of memory": 1} and u["disabled"] is None
        out2, mk = _skip(q, c, cmin, thr)
        assert mk is not None
        _assert_contract(out2, base, _kth(base, 1))

    def test_budget_declines_skipping(monkeypatch):
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_BUDGET_MIB", "0")
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(25)
        base = _base(q, c, cmin)
        out, mk = _skip(q, c, cmin, _thr_key(_kth(base, 1)))
        assert mk is None and torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["declines"] == {"over the memory budget": 1}

    def test_declines(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(26)
        base = _base(q, c, cmin)
        thr = _thr_key(_kth(base, 1))
        assert _skip(q, c, None, thr)[1] is None                 # no hot path
        tiny = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(),
                                       c.values().clone(), size=tuple(c.shape))
        tiny.values()[0] = 1e-39                                 # subnormal weight
        assert _skip(q, tiny, float(tiny.values().min()), thr)[1] is None
        bs.configure("on", 128)
        assert _skip(q, c, cmin, thr)[1] is None                 # 128 does not divide 64
        bs.configure("on", 16)
        assert _skip(q, c, cmin, thr[:3])[1] is None             # wrong threshold shape
        assert bs.usage()["declines"] == {
            "not on the hot path": 1, "subnormal corpus weights": 1,
            "sub-block 128 does not divide the document block 64": 1,
            "threshold shape": 1}

    def test_zero_row_slice_unchanged(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, _, _ = _slice(27)
        c0 = torch.sparse_csr_tensor(torch.zeros(1, dtype=torch.int64),
                                     torch.zeros(0, dtype=torch.int64),
                                     torch.zeros(0), size=(0, 30))
        out, mk = sp.scores(q, c0, cmin=0.5, skip_thr=sentinel_key((8,), "cpu"),
                            return_skip=True)
        assert out.shape == (8, 0) and mk is None
        assert sp.usage()["hot_launches"] == 0 and sp.usage()["launches"] == 0

    def _fake_ctx(n_rows, sub=64, elided=True, n_q=4):
        from types import SimpleNamespace

        n_sub = -(-n_rows // sub)
        z8 = torch.zeros((n_q, n_sub), dtype=torch.int8)
        return SimpleNamespace(n_rows=n_rows, n_sub=n_sub, sub=sub, elided=elided,
                               klive=z8, live=z8, rowany=torch.zeros(n_q, dtype=torch.int8),
                               deadcnt=torch.zeros(n_q, dtype=torch.int32),
                               thr_key=torch.zeros(n_q, dtype=torch.int64))

    def test_compile_key_covers_bound_nsp(monkeypatch):
        """K1: n_rows 65536 and 6144 are the same specialisation class for every
        size argument, but `_bound`'s NSP (and num_warps) differ: distinct keys.
        The key is built from the same meta helpers the launches use."""
        _setup(monkeypatch, 512, 64, 64)
        q, c, cmin = _slice(35)
        ptr, prow, pval = sp.corpus_postings(c)
        hot = (torch.zeros(30, dtype=torch.int32), torch.zeros(1))
        bptr = (torch.zeros(2, dtype=torch.int64), 1)
        keys = []
        for n in (65536, 6144):
            ctx = _fake_ctx(n)
            out = torch.empty((8, n))
            keys.append(bs.compile_key("cpu", q, ptr, prow, pval, out, hot, bptr, ctx))
        assert bs._bound_meta(1024)["NSP"] != bs._bound_meta(96)["NSP"]
        assert keys[0] != keys[1]
        assert "NSP" in repr(keys[0]) and "num_warps" in repr(keys[0])

    def test_bound_variant_change_is_checked(monkeypatch):
        """K1: two slices whose size classes match but whose `_bound` NSP
        differs (1024 vs 4096 rows at sub 16: n_sub 64 vs 256) each get a
        first-launch check."""
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(36, 4096)
        _skip(q, c.to_dense()[:1024].to_sparse_csr(), cmin, sentinel_key((8,), "cpu"))
        assert bs.usage()["checks"] == 1
        _skip(q, c, cmin, sentinel_key((8,), "cpu"))
        assert bs.usage()["checks"] == 2

    def test_materialize_failure_rescores_unskipped(monkeypatch):
        """K5: an OOM while materializing: skipping goes off, the slice is
        re-scored with the unskipped kernel (bit-identical), kernel stays on."""
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(37)
        base = _base(q, c, cmin)
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=_thr_key(_kth(base, 1)),
                            return_skip=True)
        assert mk is not None and mk.elided

        def oom(self, rows):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
        monkeypatch.setattr(bs.SkipMask, "cells", oom)
        mk.materialize()
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        u = bs.usage()
        assert u["rescored"] == 1 and "materialize failed" in u["disabled"]
        assert sp.kernel_off() is None

    def test_rescore_self_checks_a_new_hot_variant(monkeypatch):
        """L1: the materialize-OOM `rescore` goes through the same first-launch
        compile-key check as the ordinary path, so a never-seen specialisation
        class (288 rows, %16 == 0, after 300) is self-checked exactly once.
        (299 after 300 would be the SAME class: neither is ==1 or %16.)"""
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(41, 300)
        base = _base(q, c, cmin)                       # checks the 300-row variant
        before = set(sp._CHECKED)
        d = c.to_dense()[:288].to_sparse_csr()
        out, mk = sp.scores(q, d, cmin=cmin,
                            skip_thr=_thr_key(_kth(base[:, :288].contiguous(), 1)),
                            return_skip=True)
        assert mk is not None, bs.usage()["declines"]
        skip_keys = set(sp._CHECKED) - before
        calls = []
        real = sp._self_check

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)
        monkeypatch.setattr(sp, "_self_check", spy)

        def oom(self, rows):
            raise torch.OutOfMemoryError("CUDA out of memory.")
        monkeypatch.setattr(bs.SkipMask, "cells", oom)
        mk.materialize()
        assert len(calls) == 1
        assert len(set(sp._CHECKED) - before - skip_keys) == 1
        assert torch.equal(out.view(torch.int32), _base(q, d, cmin).view(torch.int32))

    def test_masked_cutfill_failure_falls_back_without_disabling_triton(monkeypatch):
        """K5: the HAS_LM `_cutfill` variant fails (e.g. compile error): skipping
        goes off, the slice is materialized, the PLAIN Triton top-k runs, and
        the Triton top-k is NOT disabled for the run."""
        from nova_bf.tiebreak import pack_topk

        _setup(monkeypatch, 64, 16, 64)
        _cpu_topk(monkeypatch)
        q, c, cmin = _slice(38)
        base = _base(q, c, cmin)
        key = _thr_key(_kth(base, 1))
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=key, return_skip=True)
        real = tt._cutfill

        class Wrap:
            def __getitem__(self, grid):
                def run(*a, **kw):
                    if kw.get("HAS_LM"):
                        raise RuntimeError("HAS_LM variant failed to compile")
                    return real[grid](*a, **kw)
                return run
        monkeypatch.setattr(tt, "_cutfill", Wrap())
        ordv = torch.arange(300, dtype=torch.int64)
        k1, i1, l1 = pack_topk(out, ordv, 3, thr=key, lmask=mk)
        k0, i0, l0 = pack_topk(base, ordv, 3, thr=key)
        assert torch.equal(l0, l1)
        lv = l0.bool()
        assert torch.equal(k0[lv].sort(1).values, k1[lv].sort(1).values)
        assert "masked top-k failed" in bs.usage()["disabled"]
        assert tt._cutfill is not None and tt._UNAVAILABLE is None

    def test_compile_key_tracks_specialisation_class_only(monkeypatch):
        """A tail of the same specialisation class (n_rows % 16 == 0) reuses
        the compiled kernel and its check; a tail of another class is a new
        key and is checked itself."""
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(31, 400)
        d = c.to_dense()
        _skip(q, c, cmin, sentinel_key((8,), "cpu"))                    # 400: %16
        n = len(sp._CHECKED)
        _skip(q, d[:384].to_sparse_csr(), cmin, sentinel_key((8,), "cpu"))  # %16 too
        assert len(sp._CHECKED) == n and bs.usage()["checks"] == 1
        _skip(q, d[:301].to_sparse_csr(), cmin, sentinel_key((8,), "cpu"))  # other class
        assert len(sp._CHECKED) == n + 1 and bs.usage()["checks"] == 2

    def test_skip_side_error_in_probe_poll_disables_skipping_only(monkeypatch):
        """An exception in the adaptive gate's probe polling (now inside the
        skip path's protection): skipping turns off, the posting kernel stays
        on, and the slice is scored unskipped (bit-identical)."""
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(34)
        base = _base(q, c, cmin)

        def boom():
            raise ZeroDivisionError("probe broke")
        monkeypatch.setattr(bs, "_poll_probes", boom)
        out, mk = _skip(q, c, cmin, _thr_key(_kth(base, 1)))
        assert mk is None and torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert "probe broke" in bs.usage()["disabled"] and sp.kernel_off() is None

    def test_adaptive_gate(monkeypatch):
        """Below NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD the next slices build no
        tables (declined, identical output), re-probing every REPROBE."""
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD", "0.99")
        monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_REPROBE", "3")
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(32, 300)
        base = _base(q, c, cmin)
        thr = _thr_key(_kth(base, 2))
        got = [_skip(q, c, cmin, thr) for _ in range(5)]
        assert [mk is not None for _, mk in got] == [True, False, False, True, False]
        for out, _ in got:
            _assert_contract(out, base, _kth(base, 2))
        u = bs.usage()["adaptive"]
        assert u["declines"] == 3 and u["probes"] == 1 and u["last_fraction"] < 0.99

    def test_probe_slot_keeps_the_in_flight_read(monkeypatch):
        """The adaptive gate's single pinned slot: a new probe never replaces
        one that has not been consumed; polling consumes it."""
        _setup(monkeypatch, 64, 16, 64)
        bs._probe_async(torch.tensor(3), 10)
        bs._probe_async(torch.tensor(9), 10)            # slot busy: not sampled
        bs._poll_probes()
        assert bs._ADAPT["probe"] is None and bs._ADAPT["frac"] == 0.3
        bs._probe_async(torch.tensor(9), 10)
        bs._poll_probes()
        assert bs._ADAPT["frac"] == 0.9

    def test_uniqueness_check(monkeypatch):
        ptr = torch.tensor([0, 2, 3])
        bs.check_unique(ptr, torch.tensor([0, 1, 1], dtype=torch.int32))
        with pytest.raises(ValueError):
            bs.check_unique(ptr, torch.tensor([1, 1, 0], dtype=torch.int32))

    def test_masked_topk_counted_only_after_success(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(33)
        base = _base(q, c, cmin)
        key = _thr_key(_kth(base, 1))
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=key, return_skip=True)

        class Boom:
            def __getitem__(self, grid):
                def run(*a, **k):
                    raise RuntimeError("launch failed")
                return run
        monkeypatch.setattr(tt, "_cutfill", Boom())
        with pytest.raises(RuntimeError):
            tt.topk(out, torch.arange(300, dtype=torch.int64), 3, thr=key, lmask=mk)
        assert bs.usage()["masked_topk"] == 0

    def test_threshold_is_snapshotted(monkeypatch):
        _setup(monkeypatch, 64, 16, 64)
        q, c, cmin = _slice(29)
        base = _base(q, c, cmin)
        key = _thr_key(_kth(base, 2))
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=key, return_skip=True)
        key.fill_(0x7FFFFFFF00000000)
        assert not torch.equal(mk.thr_key, key)
        mk.materialize()
        _assert_contract(out, base, _kth(base, 2))

    # --- (e) end to end through run_compute -----------------------------------

    def _write(path, rows, lang=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        cols = {
            "sparse_embedding": pa.array(
                [{"indices": i, "values": v} for i, v in rows],
                type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                                pa.field("values", pa.list_(pa.float32()))])),
            "id": pa.array([f"r{path.stem}_{j:04d}" for j in range(len(rows))]),
        }
        if lang is not None:
            cols["language"] = pa.array(lang)
        pq.write_table(pa.table(cols), str(path))

    def _dataset(tmp_path, n_files=4, per_file=400, seed=15):
        """Positive weights, duplicated rows (exact ties within and across
        files), two queries on a rare token (fewer than k documents), and a
        token-82 query whose four hits span three files best-first."""
        rng = np.random.default_rng(seed)

        def row(vocab, lo, hi):
            k = int(rng.integers(lo, hi + 1))
            idx = sorted(rng.choice(vocab, size=k, replace=False).tolist())
            v = rng.random(k).astype(np.float32) + 0.05
            return idx, v.tolist()

        cdir = tmp_path / "corpus"
        cdir.mkdir(exist_ok=True)
        dup = [row(80, 2, 10) for _ in range(20)]
        for f in range(n_files):
            rows = [row(80, 1, 12) for _ in range(per_file)]
            for j in range(0, per_file, 37):
                rows[j] = dup[(j // 37 + f) % len(dup)]
            if f == 1:
                rows[3] = ([80], [0.7])
                rows[9] = ([3, 80], [0.2, 0.4])
            for (ff, j, w) in ((0, 5, 0.9), (2, 7, 0.3), (3, 11, 0.2), (3, 300, 0.1)):
                if f == ff:
                    rows[j] = ([82], [w])
            _write(cdir / f"f{f}.parquet", rows,
                   lang=[("eng" if (f + j) % 3 else "fra") for j in range(per_file)])
        qs = [row(80, 1, 6) for _ in range(30)] + [([80], [1.0]), ([80, 81], [1.0, 2.0]),
                                                    ([82], [1.0])]
        _write(tmp_path / "q.parquet", qs)
        return cdir, tmp_path / "q.parquet"

    def _cfg(cdir, qpath, out, skip, searches, batch=256, tiebreak="ordinal", elide=True):
        from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                    ParamsConfig, QueriesConfig)

        return BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding",
                                id_column="id"),
            queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding",
                                  id_column="id"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, sparse_batch_size=batch,
                                sparse_kernel="posting", sparse_block_skip=skip,
                                sparse_block_skip_sub=16, sparse_block_skip_elide=elide,
                                tiebreak=tiebreak),
            searches=searches,
        )

    def _cpu_topk(monkeypatch):
        """Let the Triton top-k run on CPU (interpreted), so the mask-aware
        path is exercised end to end. Applies to the baseline run too."""
        def avail(scores, ordinal, k, scale=None, thr=None, enc=None, cscale=None):
            n_q, n_cols = scores.shape
            return bool(scores.ndim == 2 and ordinal.ndim == 1
                        and scores.dtype is torch.float32 and scores.is_contiguous()
                        and ordinal.dtype is torch.int64 and ordinal.numel() == n_cols
                        and n_q > 0 and 0 < k <= n_cols <= tt.MAX_BLOCK
                        and (thr is None or thr.is_contiguous())
                        and (scale is None or scale.is_contiguous())
                        and (cscale is None or cscale.is_contiguous())
                        and (enc is None or tt._enc_ok(enc, n_cols, scores.device)))
        monkeypatch.setattr(tt, "_available", avail)

    def _run(tmp_path, monkeypatch, searches, tag, skip, tiebreak="ordinal",
             elide=True, batch=256):
        import json

        import pyarrow.parquet as pq

        from nova_bf.compute import run_compute

        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        cdir, qpath = _dataset(tmp_path)
        out = tmp_path / f"{tag}_{skip}_{elide}"
        paths = run_compute(_cfg(cdir, qpath, out, skip, searches, batch, tiebreak, elide))
        tables = {n: pq.read_table(p) for n, p in paths.items()}
        man = json.loads(next(out.rglob("*manifest*.json")).read_text())
        bs_u = man["params"]["kernels"]["sparse_posting_kernel"]["block_skip"]
        assert man["params"]["sparse_block_skip"] == skip
        assert bs_u["configured"] == skip and bs_u["mode"] == skip
        return tables, bs_u

    def _dot(name="dot", k=5, filt=None):
        from nova_bf.config import SearchSpec

        return SearchSpec(name=name, k=k, metric="dot", vector_type="sparse", filter=filt)

    def _same(a, b):
        assert set(a) == set(b)
        for n in a:
            assert a[n].to_pydict() == b[n].to_pydict(), f"search {n} differs"

    @pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
    @pytest.mark.parametrize("k", [1, 5])
    @pytest.mark.parametrize("cpu_topk", [False, True])
    def test_e2e_single_dot_identical_and_engaged(tmp_path, monkeypatch, tiebreak, k,
                                                  cpu_topk):
        if cpu_topk:
            _cpu_topk(monkeypatch)
        searches = [_dot(k=k)]
        off, b_off = _run(tmp_path, monkeypatch, searches, "a", "off", tiebreak)
        on, b_on = _run(tmp_path, monkeypatch, searches, "a", "on", tiebreak)
        ovh, b_ovh = _run(tmp_path, monkeypatch, searches, "a", "overhead", tiebreak)
        unel, b_un = _run(tmp_path, monkeypatch, searches, "a", "on", tiebreak, elide=False)
        for other in (on, ovh, unel):
            _same(off, other)
        assert b_off["slices"] == 0 and b_off["gates"] == {}
        assert set(b_on["gates"]) == {"enabled"}
        assert b_on["slices"] > 0 and b_on["dead"] > 0, b_on
        assert b_on["checks"] >= 1
        assert b_on["check_failures"] == 0 and b_on["disabled"] is None
        assert b_on["elided_slices"] == b_on["slices"] and b_un["elided_slices"] == 0
        if cpu_topk:
            assert b_on["masked_topk"] > 0, b_on
        else:
            assert b_on["masked_topk"] == 0 and b_on["materialized"] > 0
        assert b_ovh["overhead_slices"] > 0 and b_ovh["slices"] == 0
        assert b_ovh["dead"] == b_on["dead"]
        hits = off["dot"].to_pydict()["hit_ids"]
        assert len(hits[-2]) == len(hits[-3]) == min(k, 2)
        assert len(hits[-1]) == min(k, 4)

    def test_e2e_narrow_tail_slices_materialize(tmp_path, monkeypatch):
        """k=200 with 256-row slices: each file's 144-row tail is narrower than
        k, so it takes `pack` + `live_rows`, which must see materialized -inf
        cells. Wide slices use the mask-aware top-k (CPU top-k patched in),
        so every materialization comes from the narrow path."""
        _cpu_topk(monkeypatch)
        searches = [_dot(k=200)]
        off, _ = _run(tmp_path, monkeypatch, searches, "n", "off")
        on, b = _run(tmp_path, monkeypatch, searches, "n", "on")
        _same(off, on)
        assert b["dead"] > 0 and b["masked_topk"] > 0, b
        assert b["materialized"] > 0, b

    def test_e2e_warm_up_gate(tmp_path, monkeypatch):
        """k larger than the first slices: no tables are built until k rows
        have been folded (declined as warm-up), results identical."""
        searches = [_dot(k=300)]
        off, _ = _run(tmp_path, monkeypatch, searches, "w", "off")
        on, b = _run(tmp_path, monkeypatch, searches, "w", "on")
        _same(off, on)
        assert b["declines"].get("state cannot be full yet") == 2   # 0 then 256 rows
        assert b["slices"] > 0

    def test_e2e_skip_config_fails_before_any_read(tmp_path, monkeypatch):
        """K4: bad skip settings raise from `run_compute` before the queries
        or the corpus are touched (both paths here do not exist), naming the
        variable."""
        from nova_bf import compute
        from nova_bf.compute import run_compute

        def too_late(*a, **k):
            raise AssertionError("skip config validated only after thread pools / reads")
        # the first step of run_compute after the per-run reset, before any read
        monkeypatch.setattr(compute, "configure_thread_pools", too_late)
        cfg = _cfg(tmp_path / "no_corpus", tmp_path / "no_q.parquet", tmp_path / "o",
                   "on", [_dot()])
        for name, bad in (("SUB", "abc"), ("REPROBE", "x"), ("MIN_DEAD", "abc"),
                          ("ELIDE", "maybe"), ("", "bogus")):
            key = "NOVA_BF_SPARSE_BLOCK_SKIP" + (f"_{name}" if name else "")
            monkeypatch.setenv(key, bad)
            with pytest.raises(ValueError, match=key):
                run_compute(cfg)
            monkeypatch.delenv(key)

    # The reviewer's poisoned differential grid (round 2), folded in: each
    # case runs off vs on (elided), on (unelided) and overhead.
    _GRID = [
        dict(seed=1, files=3, per=150, vocab=40, nq=12, k=1, batch=64, tb="ordinal",
             sub=8, cpu_topk=True, nosel=False),
        dict(seed=2, files=3, per=150, vocab=40, nq=12, k=4, batch=96, tb="id",
             sub=16, cpu_topk=True, nosel=True),
        dict(seed=3, files=3, per=150, vocab=40, nq=12, k=10, batch=128, tb="ordinal",
             sub=64, cpu_topk=False, nosel=False),
        dict(seed=4, files=3, per=160, vocab=40, nq=12, k=4, batch=150, tb="id",
             sub=8, cpu_topk=False, nosel=True),
        dict(seed=5, files=3, per=130, vocab=30, nq=12, k=10, batch=64, tb="id",
             sub=16, cpu_topk=True, nosel=False),
        dict(seed=6, files=3, per=150, vocab=40, nq=12, k=1, batch=200, tb="ordinal",
             sub=64, cpu_topk=False, nosel=False),
        dict(seed=9, files=3, per=170, vocab=36, nq=12, k=4, batch=64, tb="ordinal",
             sub=16, cpu_topk=True, nosel=True),
        dict(seed=12, files=3, per=100, vocab=40, nq=12, k=10, batch=96, tb="id",
             sub=8, cpu_topk=True, nosel=False),
    ]

    def _grid_dataset(root, seed, n_files, per_file, vocab, n_q):
        """Odd seeds: ragged files (narrow tails); seed % 3 == 0: quantized
        weights (many exact ties); a few strong documents late in each file;
        ids NOT in file/row order (tiebreak=id differs from ordinal)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(seed)

        def row(lo, hi, quant):
            k = int(rng.integers(lo, hi + 1))
            idx = sorted(rng.choice(vocab, size=k, replace=False).tolist())
            v = rng.random(k).astype(np.float32) + 0.05
            if quant:
                v = (np.round(v * 4) / 4 + 0.25).astype(np.float32)
            return idx, v.tolist()

        def write(path, rows):
            ids = [f"{(j * 7919 + len(path.stem) * 104729) % 10**6:07d}_{path.stem}_{j}"
                   for j in range(len(rows))]
            pq.write_table(pa.table({
                "sparse_embedding": pa.array(
                    [{"indices": i, "values": v} for i, v in rows],
                    type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                                    pa.field("values", pa.list_(pa.float32()))])),
                "id": pa.array(ids)}), str(path))

        cdir = root / "corpus"
        cdir.mkdir()
        for f in range(n_files):
            n = per_file + (37 * f if seed % 2 else 0)
            rows = [row(1, 12, seed % 3 == 0) for _ in range(n)]
            for j in range(n - 5, n):
                rows[j] = (list(range(0, vocab, 3)), [2.0] * len(range(0, vocab, 3)))
            write(cdir / f"f{f}.parquet", rows)
        write(root / "q.parquet", [row(1, 6, False) for _ in range(n_q)])
        return cdir, root / "q.parquet"

    @pytest.mark.parametrize("case", range(len(_GRID)))
    def test_e2e_poisoned_differential_grid(tmp_path, monkeypatch, case):
        import json

        import pyarrow.parquet as pq

        from nova_bf.compute import run_compute
        from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                    ParamsConfig, QueriesConfig, SearchSpec)

        c = _GRID[case]
        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        if c["cpu_topk"]:
            _cpu_topk(monkeypatch)
        if c["nosel"]:
            monkeypatch.setenv("NOVA_BF_NO_TOPK_SELECT", "1")
        cdir, qpath = _grid_dataset(tmp_path, c["seed"], c["files"], c["per"],
                                    c["vocab"], c["nq"])

        def run(tag, skip, elide):
            out = tmp_path / tag
            cfg = BruteForceConfig(
                corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding",
                                    id_column="id"),
                queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding",
                                      id_column="id"),
                output=OutputConfig(path=str(out)),
                params=ParamsConfig(io_workers=1, sparse_batch_size=c["batch"],
                                    sparse_kernel="posting", sparse_block_skip=skip,
                                    sparse_block_skip_sub=c["sub"],
                                    sparse_block_skip_elide=elide,
                                    tiebreak=c["tb"]),
                searches=[SearchSpec(name="dot", k=c["k"], metric="dot",
                                     vector_type="sparse")])
            paths = run_compute(cfg)
            t = {n: pq.read_table(p).to_pydict() for n, p in paths.items()}
            man = json.loads(next(out.rglob("*manifest*.json")).read_text())
            return t, man["params"]["kernels"]["sparse_posting_kernel"]["block_skip"]

        base, _ = run("off", "off", True)
        dead = None
        for tag, skip, elide in (("on", "on", True), ("un", "on", False),
                                 ("ov", "overhead", True)):
            got, u = run(tag, skip, elide)
            assert got == base, (case, tag)
            assert u["check_failures"] == 0 and u["disabled"] is None
            if skip == "on" and elide:
                dead = u["dead"]
                assert u["slices"] > 0
                if c["cpu_topk"]:
                    assert u["masked_topk"] > 0
        assert dead and dead > 0, c

    def _shared_dataset(tmp_path, layout, n_files=3, per_file=300, n_q=36, seed=41):
        """Corpus with dense + sparse + language columns; queries with dense,
        sparse and a `query_set` column (`layout`: "blocks" = sorted by set,
        "interleaved" = round robin), as in a shared dense / sparse /
        filtered ground-truth run."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(seed)
        sp_t = pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                          pa.field("values", pa.list_(pa.float32()))])

        def srow(lo, hi):
            k = int(rng.integers(lo, hi + 1))
            idx = sorted(rng.choice(60, size=k, replace=False).tolist())
            return {"indices": idx,
                    "values": (rng.random(k).astype(np.float32) + 0.05).tolist()}

        def drow():
            return rng.standard_normal(8).astype(np.float32).tolist()

        cdir = tmp_path / "corpus"
        cdir.mkdir(exist_ok=True)
        for f in range(n_files):
            pq.write_table(pa.table({
                "sparse_embedding": pa.array([srow(1, 12) for _ in range(per_file)], sp_t),
                "dense_embedding": pa.array([drow() for _ in range(per_file)],
                                            pa.list_(pa.float32())),
                "language": pa.array([("eng" if (f + j) % 3 else "fra")
                                      for j in range(per_file)]),
                "id": pa.array([f"c{f}_{j:04d}" for j in range(per_file)]),
            }), str(cdir / f"f{f}.parquet"))
        sets = ["dense", "sparse", "filtered"]
        if layout == "blocks":
            qs = [sets[i * 3 // n_q] for i in range(n_q)]
        else:
            qs = [sets[i % 3] for i in range(n_q)]
        pq.write_table(pa.table({
            "sparse_embedding": pa.array([srow(1, 6) for _ in range(n_q)], sp_t),
            "dense_embedding": pa.array([drow() for _ in range(n_q)], pa.list_(pa.float32())),
            "query_set": pa.array(qs),
            "lang": pa.array([("eng" if i % 4 else "fra") for i in range(n_q)]),
            "id": pa.array([f"q{i:03d}" for i in range(n_q)]),
        }), str(tmp_path / "q.parquet"))
        return cdir, tmp_path / "q.parquet"

    def _shared_run(tmp_path, cdir, qpath, searches, tag, skip, batch=128):
        import json

        import pyarrow.parquet as pq

        from nova_bf.compute import run_compute
        from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                    ParamsConfig, QueriesConfig)

        out = tmp_path / f"{tag}_{skip}"
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="id"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, sparse_batch_size=batch,
                                dense_batch_size=batch, sparse_kernel="posting",
                                sparse_block_skip=skip, sparse_block_skip_sub=16),
            searches=searches)
        paths = run_compute(cfg)
        tables = {n: pq.read_table(p).to_pydict() for n, p in paths.items()}
        man = json.loads(next(out.rglob("*manifest*.json")).read_text())
        return tables, man["params"]["kernels"]["sparse_posting_kernel"]["block_skip"]

    def _shared_searches(k, sparse_rows=("sparse",), second_sparse=False):
        from nova_bf.config import Filter, FilterCondition, RowSelector, SearchSpec

        def rows(*v):
            return RowSelector(column="query_set", isin=list(v))
        out = [
            SearchSpec(name="dense_cos", k=k, metric="cosine", vector_type="dense",
                       rows=rows("dense", "sparse")),
            SearchSpec(name="sparse_dot", k=k, metric="dot", vector_type="sparse",
                       rows=rows(*sparse_rows)),
            SearchSpec(name="filtered", k=k, metric="cosine", vector_type="dense",
                       rows=rows("filtered"),
                       filter=Filter(must=[FilterCondition(field="language", match="eng")])),
        ]
        if second_sparse:
            out.append(SearchSpec(name="sparse_dot2", k=k, metric="dot",
                                  vector_type="sparse", rows=rows("filtered")))
        return out

    @pytest.mark.parametrize("layout", ["blocks", "interleaved"])
    @pytest.mark.parametrize("k", [1, 5])
    @pytest.mark.parametrize("cpu_topk", [False, True])
    def test_e2e_shared_mode_sparse_rows_subset_engages(tmp_path, monkeypatch, layout, k,
                                                        cpu_topk):
        """Dense, row-subset sparse dot and filtered dense searches in one run
        (the shared ground-truth layout): the sparse search is its
        vector_type's only reader, so skipping engages, every search's output
        is identical to skipping off, and (CPU top-k patched in) the
        mask-aware top-k is used, i.e. the whole-matrix selection was dropped."""
        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        if cpu_topk:
            _cpu_topk(monkeypatch)
        cdir, qpath = _shared_dataset(tmp_path, layout)
        searches = _shared_searches(k)
        off, b_off = _shared_run(tmp_path, cdir, qpath, searches, "s", "off")
        on, b = _shared_run(tmp_path, cdir, qpath, searches, "s", "on")
        assert set(off) == {"dense_cos", "sparse_dot", "filtered"}
        assert off == on
        assert b_off["slices"] == 0
        assert set(b["gates"]) == {"enabled"}, b["gates"]
        assert b["slices"] > 0 and b["dead"] > 0, b
        assert b["checks"] >= 1 and b["check_failures"] == 0 and b["disabled"] is None
        assert b["model_probes"] >= 1 and b["model_probe_failures"] == 0
        if cpu_topk:
            assert b["masked_topk"] > 0, b
        # each search answered exactly its own query rows
        n_rows = {"dense_cos": 24, "sparse_dot": 12, "filtered": 12}
        for n, t in off.items():
            assert len(t["hit_ids"]) == n_rows[n], n

    def _sparse_specs(case):
        """Sparse reader mixes over the shared dataset (dense + filtered dense
        searches always run beside them)."""
        from nova_bf.config import Filter, FilterCondition, RowSelector, SearchSpec

        def rows(*v):
            return RowSelector(column="query_set", isin=list(v))
        eng = Filter(must=[FilterCondition(field="language", match="eng")])
        perq = Filter(must=[FilterCondition(field="language", match_from_query="lang")])

        def dot(name, k=5, r=None, f=None):
            return SearchSpec(name=name, k=k, metric="dot", vector_type="sparse",
                              rows=r, filter=f)
        return {
            # disjoint / overlapping proper subsets of the raw matrix
            "two_subsets": [dot("a", r=rows("sparse")), dot("b", r=rows("filtered"))],
            "overlapping": [dot("a", r=rows("sparse", "filtered")),
                            dot("b", k=2, r=rows("filtered", "dense"))],
            # dot + filtered dot (uniform), whole and subset
            "dot+filtered": [dot("u"), dot("f", f=eng)],
            "dot+filtered_subset": [dot("u", r=rows("sparse")),
                                    dot("f", r=rows("sparse", "filtered"), f=eng)],
            # dot + per-query filtered dot
            "dot+perquery": [dot("u", k=3), dot("p", f=perq)],
            # one filtered reader alone, whole-matrix rows subset, with its
            # per-query filter SHARED with a dense search over other rows (its
            # filter mask lives in that union's row space: the selector must
            # not be dropped)
            "perquery_alone_shared_filter": [dot("p", r=rows("sparse"), f=perq)],
            "two_k": [dot("k1", k=1), dot("k20", k=20)],
        }[case]

    @pytest.mark.parametrize("case", ["two_subsets", "overlapping", "dot+filtered",
                                      "dot+filtered_subset", "dot+perquery",
                                      "perquery_alone_shared_filter", "two_k"])
    @pytest.mark.parametrize("cpu_topk", [False, True])
    def test_e2e_several_dot_readers_engage(tmp_path, monkeypatch, case, cpu_topk):
        """Several dot readers of one sparse raw matrix (filters, `rows`
        subsets, different k), beside dense and filtered dense searches: the
        slice skips against the per-row minimum threshold, every output is
        identical to skipping off."""
        from nova_bf.config import Filter, FilterCondition, RowSelector, SearchSpec

        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        if cpu_topk:
            _cpu_topk(monkeypatch)
        cdir, qpath = _shared_dataset(tmp_path, "interleaved")
        searches = _shared_searches(5)[:1] + [_shared_searches(5)[2]] + _sparse_specs(case)
        if case == "perquery_alone_shared_filter":
            perq = Filter(must=[FilterCondition(field="language", match_from_query="lang")])
            searches.append(SearchSpec(
                name="dense_perq", k=5, metric="cosine", vector_type="dense",
                rows=RowSelector(column="query_set", isin=["dense", "filtered"]),
                filter=perq))
        off, _ = _shared_run(tmp_path, cdir, qpath, searches, case, "off")
        on, b = _shared_run(tmp_path, cdir, qpath, searches, case, "on")
        assert off == on
        assert set(b["gates"]) == {"enabled"}, b["gates"]
        assert b["slices"] > 0 and b["dead"] > 0, b
        assert b["check_failures"] == 0 and b["disabled"] is None, b
        if cpu_topk and case in ("two_k", "dot+filtered", "dot+perquery"):
            assert b["masked_topk"] > 0, b          # an unfiltered whole reader
        if case.startswith("perquery_alone"):
            assert b["masked_topk"] == 0 and b["materialized"] > 0, b

    def test_e2e_dot_plus_cosine_sparse_stays_off(tmp_path, monkeypatch):
        """A cosine reader of the raw matrix keeps skipping off (its threshold
        is in cosine units), outputs identical."""
        from nova_bf.config import SearchSpec

        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        cdir, qpath = _shared_dataset(tmp_path, "interleaved")
        searches = _shared_searches(5) + [SearchSpec(
            name="sparse_cos", k=5, metric="cosine", vector_type="sparse")]
        off, _ = _shared_run(tmp_path, cdir, qpath, searches, "dc", "off")
        on, b = _shared_run(tmp_path, cdir, qpath, searches, "dc", "on")
        assert off == on
        assert b["slices"] == 0 and "enabled" not in b["gates"], b
        assert b["gates"].get("metric cosine", 0) > 0, b

    # --- reviewer round 2: multi-reader differentials ----------------------

    _REV_WORDS = ["alpha", "beta", "gamma", "delta", "eps"]


    def _rev_data(root, n_files=4, per_file=400, seed=7, n_q=30):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(seed)
        st = pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                        pa.field("values", pa.list_(pa.float32()))])

        def srow(lo, hi, vocab=80):
            k = int(rng.integers(lo, hi + 1))
            idx = sorted(rng.choice(vocab, size=k, replace=False).tolist())
            v = rng.random(k).astype(np.float32) + 0.05
            if rng.random() < 0.3:
                v = (np.round(v * 4) / 4 + 0.25).astype(np.float32)
            return {"indices": idx, "values": v.tolist()}

        cdir = root / "corpus"
        cdir.mkdir()
        for f in range(n_files):
            rows = [srow(1, 12) for _ in range(per_file)]
            for j in range(per_file - 4, per_file):   # strong late rows
                rows[j] = {"indices": list(range(0, 80, 3)), "values": [2.0] * 27}
            body = [" ".join(rng.choice(_REV_WORDS, size=int(rng.integers(1, 4)))) for _ in range(per_file)]
            pq.write_table(pa.table({
                "sparse_embedding": pa.array(rows, st),
                "language": pa.array([("eng" if (f + j) % 3 else "fra") for j in range(per_file)]),
                "rare": pa.array([("x" if (j * 7 + f) % 23 == 0 else "y") for j in range(per_file)]),
                "body": pa.array(body),
                "id": pa.array([f"{(j * 7919 + f * 104729) % 10**6:07d}_{f}_{j}" for j in range(per_file)]),
            }), str(cdir / f"f{f}.parquet"))
        qs = [srow(1, 6) for _ in range(n_q)]
        qs[3] = {"indices": [], "values": []}          # empty query
        qs[7] = {"indices": [79], "values": [1.0]}     # rare-ish token
        pq.write_table(pa.table({
            "sparse_embedding": pa.array(qs, st),
            "query_set": pa.array([["a", "b", "c"][i % 3] for i in range(n_q)]),
            "lang": pa.array([("eng" if i % 4 else "fra") for i in range(n_q)]),
            "phrase": pa.array([_REV_WORDS[i % 5] if i % 6 else "alpha beta" for i in range(n_q)]),
            "id": pa.array([f"q{i:03d}" for i in range(n_q)]),
        }), str(root / "q.parquet"))
        return cdir, root / "q.parquet"


    def _rev_specs(case):
        from nova_bf.config import Filter, FilterCondition, RowSelector, SearchSpec

        def rows(*v):
            return RowSelector(column="query_set", isin=list(v))

        def dot(name, k=5, r=None, f=None):
            return SearchSpec(name=name, k=k, metric="dot", vector_type="sparse", rows=r, filter=f)
        fra = Filter(must=[FilterCondition(field="language", match="fra")])
        rare = Filter(must=[FilterCondition(field="rare", match="x")])
        txt = Filter(must=[FilterCondition(field="body", match_text="alpha")])
        perq = Filter(must=[FilterCondition(field="language", match_from_query="lang")])
        ptxt = Filter(must=[FilterCondition(field="body", match_text_from_query="phrase")])
        return {
            "filters_only_coalesce": [dot("fra", 3, f=fra), dot("rare", 4, r=rows("a", "c"), f=rare),
                                      dot("txt", 2, r=rows("b"), f=txt)],
            "perquery_text_only": [dot("pq", 5, f=perq), dot("pt", 3, r=rows("a", "b"), f=ptxt),
                                   dot("pt2", 1, r=rows("c"), f=ptxt)],
            "empty_filter_mix": [dot("empty", 4, f=Filter()), dot("pt", 2, r=rows("b", "c"), f=ptxt),
                                 dot("huge", 5000, r=rows("a")), dot("u", 3, r=rows("c")),
                                 dot("rare", 6, f=rare)],
            "mixed_and_none": [dot("u", 7), dot("f", 2, r=rows("a"), f=perq),
                               dot("fr", 9, r=rows("b"), f=fra)],
        }[case]


    def _rev_run(root, cdir, qpath, searches, tag, skip, batch, num_jobs=None, elide=True):
        import json

        import pyarrow.parquet as pq

        from nova_bf.compute import run_compute
        from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                    ParamsConfig, QueriesConfig)
        from nova_bf.merge import run_merge

        out = root / f"{tag}_{skip}_{elide}"
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding", id_column="id"),
            queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding", id_column="id"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, sparse_batch_size=batch, sparse_kernel="posting",
                                sparse_block_skip=skip, sparse_block_skip_sub=16,
                                sparse_block_skip_elide=elide, threshold_exchange=False),
            searches=searches)
        usages = []
        if num_jobs is None:
            paths = run_compute(cfg)
        else:
            for r in range(num_jobs):
                run_compute(cfg, num_jobs=num_jobs, job_rank=r)
            paths = run_merge(cfg)
        for mf in list(out.rglob("*manifest*.json")) + list(out.glob("_bf_manifest_*_compute/*.json")):
            man = json.loads(mf.read_text())
            try:
                usages.append(man["params"]["kernels"]["sparse_posting_kernel"]["block_skip"])
            except KeyError:
                pass
        t = {n: pq.read_table(p).to_pydict() for n, p in paths.items()}
        for n in t:
            t[n].pop("src", None)
        return t, usages


    @pytest.mark.parametrize("case", ["filters_only_coalesce", "perquery_text_only",
                                      "empty_filter_mix", "mixed_and_none"])
    @pytest.mark.parametrize("cpu_topk", [False, True])
    @pytest.mark.parametrize("batch", [64, 512])
    def test_e2e_readers_differential(tmp_path, monkeypatch, case, cpu_topk, batch):
        """Reviewer round 2 (2026-11-16): several dot readers with no unfiltered
        search (union-compacted, coalesced batches), text and per-query filters on
        index-tensor `rows` subsets, `filter: {}`, an empty query, k beyond the
        corpus: off vs on byte-identical on the poisoned buffer."""
        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        if cpu_topk:
            _cpu_topk(monkeypatch)
        cdir, qpath = _rev_data(tmp_path)
        s = _rev_specs(case)
        off, _ = _rev_run(tmp_path, cdir, qpath, s, case, "off", batch)
        on, us = _rev_run(tmp_path, cdir, qpath, s, case, "on", batch)
        assert off == on
        u = us[0]
        assert u["check_failures"] == 0 and u["disabled"] is None, u
        assert u["slices"] > 0 and u["dead"] > 0, u


    @pytest.mark.parametrize("case", ["empty_filter_mix", "mixed_and_none"])
    def test_e2e_readers_multirank_merge(tmp_path, monkeypatch, case):
        """Two ranks + merge, several dot readers: merged outputs identical."""
        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        _cpu_topk(monkeypatch)
        cdir, qpath = _rev_data(tmp_path)
        s = _rev_specs(case)
        off, _ = _rev_run(tmp_path, cdir, qpath, s, case, "off", 64, num_jobs=2)
        on, us = _rev_run(tmp_path, cdir, qpath, s, case, "on", 64, num_jobs=2)
        assert off == on
        assert len(us) >= 2, us                       # both ranks' manifests
        assert all(u["check_failures"] == 0 and u["disabled"] is None for u in us), us
        assert any(u["dead"] > 0 for u in us)

    @pytest.mark.parametrize("case", ["dot+cos", "two_k", "filtered", "filtered+unfiltered"])
    def test_e2e_shared_raw(tmp_path, monkeypatch, case):
        from nova_bf.config import Filter, FilterCondition, SearchSpec

        eng = Filter(must=[FilterCondition(field="language", match="eng")])
        searches = {
            "dot+cos": [_dot(), SearchSpec(name="cos", k=5, metric="cosine",
                                           vector_type="sparse")],
            "two_k": [_dot("k2", 2), _dot("k20", 20)],
            "filtered": [_dot("f", 5, eng)],
            "filtered+unfiltered": [_dot("u", 5), _dot("f", 5, eng)],
        }[case]
        off, _ = _run(tmp_path, monkeypatch, searches, case, "off")
        on, b = _run(tmp_path, monkeypatch, searches, case, "on")
        _same(off, on)
        if case == "dot+cos":       # a cosine reader: off
            assert b["slices"] == 0 and b["dead"] == 0
            assert "enabled" not in b["gates"] and sum(b["gates"].values()) > 0, b
        else:                       # dot readers only: skip against the minimum
            assert set(b["gates"]) == {"enabled"} and b["slices"] > 0, b
            assert b["check_failures"] == 0 and b["disabled"] is None, b

    # --- dense row reorder + mixed-metric two-pass beside sparse skipping ---

    def _combined_data(root, n_files=4, per_file=200, dim=64, n_q=48, seed=57):
        """Dense (fp16-representable) + sparse + language corpus; queries
        whose `query_set` cycles a, b, c row by row, so every `rows` subset is
        interleaved in file order and only the dense reorder makes them
        contiguous."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(seed)
        st = pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                        pa.field("values", pa.list_(pa.float32()))])

        def srow(lo, hi):
            k = int(rng.integers(lo, hi + 1))
            idx = sorted(rng.choice(60, size=k, replace=False).tolist())
            return {"indices": idx,
                    "values": (rng.random(k).astype(np.float32) + 0.05).tolist()}

        def dense(n):
            v = rng.standard_normal((n, dim)).astype(np.float16).astype(np.float32)
            return pa.array(v.tolist(), pa.list_(pa.float32()))

        cdir = root / "corpus"
        cdir.mkdir()
        for f in range(n_files):
            pq.write_table(pa.table({
                "sparse_embedding": pa.array([srow(1, 12) for _ in range(per_file)], st),
                "dense_embedding": dense(per_file),
                "language": pa.array([("eng" if (f + j) % 3 else "fra")
                                      for j in range(per_file)]),
                "id": pa.array([f"c{f}_{j:04d}" for j in range(per_file)]),
            }), str(cdir / f"f{f}.parquet"))
        pq.write_table(pa.table({
            "sparse_embedding": pa.array([srow(1, 6) for _ in range(n_q)], st),
            "dense_embedding": dense(n_q),
            "query_set": pa.array(["abc"[i % 3] for i in range(n_q)]),
            "id": pa.array([f"q{i:03d}" for i in range(n_q)]),
        }), str(root / "q.parquet"))
        return cdir, root / "q.parquet"

    def _combined_searches():
        from nova_bf.config import Filter, FilterCondition, RowSelector, SearchSpec

        def rows(*v):
            return RowSelector(column="query_set", isin=list(v))
        eng = Filter(must=[FilterCondition(field="language", match="eng")])
        return [
            # dense: dot + cosine in one batch (mixed-metric two-pass), on
            # laminar interleaved subsets (the reorder), plus a filtered one
            SearchSpec(name="d_dot", k=5, metric="dot", vector_type="dense", rows=rows("a")),
            SearchSpec(name="d_cos", k=5, metric="cosine", vector_type="dense",
                       rows=rows("b", "c")),
            SearchSpec(name="d_cos_eng", k=4, metric="cosine", vector_type="dense",
                       rows=rows("c"), filter=eng),
            # sparse: dot readers on the whole matrix (the mask-aware top-k),
            # a subset and a filtered subset, all different k (min-threshold
            # skipping)
            SearchSpec(name="s_all", k=5, metric="dot", vector_type="sparse"),
            SearchSpec(name="s_dot", k=2, metric="dot", vector_type="sparse", rows=rows("b")),
            SearchSpec(name="s_dot_eng", k=3, metric="dot", vector_type="sparse",
                       rows=rows("a", "c"), filter=eng),
        ]

    def _combined_run(root, monkeypatch, cdir, qpath, tag, on):
        import glob
        import json

        import pyarrow.parquet as pq

        from nova_bf import twopass
        from nova_bf.compute import run_compute
        from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                    ParamsConfig, QueriesConfig)

        if on:
            monkeypatch.delenv("NOVA_BF_NO_ROW_REORDER", raising=False)
            monkeypatch.delenv("NOVA_BF_NO_TWOPASS", raising=False)
        else:
            monkeypatch.setenv("NOVA_BF_NO_ROW_REORDER", "1")
            monkeypatch.setenv("NOVA_BF_NO_TWOPASS", "1")
        out = root / f"{tag}_{'on' if on else 'off'}"
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), id_column="id"),
            queries=QueriesConfig(path=str(qpath), id_column="id"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, dense_batch_size=64, sparse_batch_size=64,
                                sparse_kernel="posting",
                                sparse_block_skip="on" if on else "off",
                                sparse_block_skip_sub=16, threshold_exchange=False),
            searches=_combined_searches())
        twopass.reset()
        paths = run_compute(cfg)
        st = twopass.stats()
        tables = {n: pq.read_table(p) for n, p in paths.items()}
        (mf,) = glob.glob(str(out / "_bf_manifest_*_compute.json"))
        man = json.loads(open(mf).read())
        return tables, man, st

    @pytest.mark.parametrize("cpu_topk", [False, True])
    def test_e2e_row_reorder_and_mixed_metric_twopass_beside_sparse_skip(
            tmp_path, monkeypatch, cpu_topk):
        """The merged dense and sparse changes in one run: dense dot + cosine
        searches on interleaved `rows` subsets (reordered to contiguous blocks
        and two-passed per metric, slice mode), a filtered dense search, and
        three sparse dot readers (whole matrix, subset, filtered subset;
        different k) skipping against
        their per-row minimum threshold. Every output is byte-identical to the
        run with the reorder, the two-pass and skipping all off."""
        from nova_bf import twopass

        monkeypatch.setattr(sp, "DOC_BLOCK", 64)
        monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "slice")
        monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
        monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
        monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
        monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
        monkeypatch.setattr(twopass, "PAD_SMALL", ())
        monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 8)
        if cpu_topk:
            _cpu_topk(monkeypatch)
        cdir, qpath = _combined_data(tmp_path)
        off, m_off, st_off = _combined_run(tmp_path, monkeypatch, cdir, qpath, "c", False)
        on, m_on, st_on = _combined_run(tmp_path, monkeypatch, cdir, qpath, "c", True)
        twopass.reset()

        assert set(off) == set(on) == {"d_dot", "d_cos", "d_cos_eng", "s_all", "s_dot",
                                         "s_dot_eng"}
        for n in off:
            assert off[n].equals(on[n], check_metadata=True), n
        assert m_on["counts"] == m_off["counts"]

        lay = m_on["params"]["query_row_layout"]
        assert lay["dense"]["status"] == "reordered", lay
        assert st_off["slices_twopass"] == 0, st_off
        assert st_on["slices_twopass"] > 0, st_on
        assert st_on["groups_shared_gram_twopass"] > 0, st_on
        assert st_on["groups_shared_gram_kept"] == 0, st_on

        b = m_on["params"]["kernels"]["sparse_posting_kernel"]["block_skip"]
        assert set(b["gates"]) == {"enabled"}, b["gates"]
        assert b["slices"] > 0 and b["dead"] > 0, b
        assert b["check_failures"] == 0 and b["disabled"] is None, b
        assert b["model_probe_failures"] == 0, b
        if cpu_topk:
            assert b["masked_topk"] > 0, b


if _CHILD:
    def test_dead_program_count_matches_bruteforce(monkeypatch):
        """book() groups the bound's verdict into (query, DOC_BLOCK) programs:
        a program is dead iff every in-slice sub-block is dead; the padded tail
        of the last program counts as not live (the kernel's bits == 0 test)."""
        from types import SimpleNamespace
        from nova_bf import sparse_block_skip as bs
        from nova_bf import sparse_posting_triton as sp

        g = torch.Generator().manual_seed(7)
        for sub, n_sub, n_q, p in [(64, 128, 9, 0.9), (64, 13, 5, 0.5),
                                   (16, 70, 4, 0.97), (512, 3, 6, 0.3)]:
            bs.reset()
            live = (torch.rand(n_q, n_sub, generator=g) > p).to(torch.int8)
            live[0] = 0                                   # one fully dead row
            ctx = SimpleNamespace(live=live.flatten(), sub=sub, n_sub=n_sub,
                                  deadcnt=(live == 0).sum(1).to(torch.int32),
                                  elided=True)
            monkeypatch.setattr(bs, "_probe_async", lambda *a, **k: None)
            bs.book(ctx, n_q)
            nsb = max(1, sp.DOC_BLOCK // sub)
            want_dead = want_prog = 0
            for q in range(n_q):
                for b0 in range(0, n_sub, nsb):
                    want_prog += 1
                    want_dead += int(not live[q, b0:b0 + nsb].any())
            u = bs.usage()
            assert u["programs"] == want_prog, (sub, n_sub)
            assert u["dead_programs"] == want_dead, (sub, n_sub)
            assert u["dead"] == int((live == 0).sum())
