"""Posting-list scoring for sparse dot products.

Unlike the cuSPARSE path, this kernel scores directly from token posting
lists without densifying the corpus or running a separate overlap check.

The corpus slice is grouped by token (CSC). Each GPU program owns a query
and a block of documents, accumulating scores from the query's shared
tokens. Scores start at -inf, so documents with no shared token remain
excluded. Stored zero weights still count as shared tokens.

Scores are accumulated in FP32 in ascending token order. Each cell has
one owner, avoiding atomics and ensuring deterministic results. A barrier
between tokens prevents conflicting updates. Scores may differ slightly
from cuSPARSE due to summation order.

The kernel requires finite weights and bounds partial sums to prevent
overflow into the -inf sentinel.

Each compiled configuration is checked against cuSPARSE on its first
launch using sampled queries. Masks must match exactly and scores must
agree within FP32 rounding bounds. In "auto" mode, a failed check or
kernel failure disables the kernel for the rest of the run; "posting"
mode raises an error instead.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

BLOCK_P = 256            # postings per vectorised step
BLOCK_I = 512            # elements per step of the row initialisation
NUM_WARPS = 2
DOC_BLOCK = 512          # documents per program (0 = one program per whole row)
SELF_CHECK_ROWS = 256

_OFF: str | None = None
_UNAVAILABLE: str | None = None
_LAUNCHES = 0
_SELFCHECKS = 0
_SELFCHECK_FAILURES = 0
_CHECKED: set = set()
_POLICY = "auto"          # params.sparse_kernel, for the manifest
# Query-side magnitude bound for the run-wide query CSR, held through a weak
# reference.
_Q_BOUND: dict = {}
_DECLINES: dict = {}      # decline reason -> slices (manifest)
_OOM_STREAK = 0           # consecutive out-of-memory fallbacks
_OOM_WARNED = False       # the out-of-memory warning is logged once per run
OOM_STREAK_LIMIT = 3      # then the kernel is switched off for the run
MAX_GRID_Y = 65535        # CUDA grid dimension y (document blocks per slice)
MAX_SUM = 1e36            # |partial sum| bound that keeps the -inf sentinel exact

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _posting_scores(QCROW, QCOL, QVAL, CPTR, CROW, CVAL, OUT,
                        n_rows, stride_out,
                        BLOCK_P: tl.constexpr, BLOCK_I: tl.constexpr):
        q = tl.program_id(0)
        base = OUT + q.to(tl.int64) * stride_out

        # Every cell starts untouched.
        for d0 in range(0, n_rows, BLOCK_I):
            o = d0 + tl.arange(0, BLOCK_I)
            tl.store(base + o, tl.full([BLOCK_I], float("-inf"), tl.float32),
                     mask=o < n_rows)

        j0 = tl.load(QCROW + q)
        j1 = tl.load(QCROW + q + 1)
        for j in range(j0, j1):
            # Make the previous token's (or the initialisation's) stores
            # visible to every thread before this token reads the row.
            tl.debug_barrier()
            t = tl.load(QCOL + j)
            qv = tl.load(QVAL + j)
            p0 = tl.load(CPTR + t)
            p1 = tl.load(CPTR + t + 1)
            for p in range(p0, p1, BLOCK_P):
                offs = p + tl.arange(0, BLOCK_P)
                m = offs < p1
                d = tl.load(CROW + offs, mask=m, other=0)
                cv = tl.load(CVAL + offs, mask=m, other=0.0)
                # `.cg`: read through L2, never a stale L1 line, after the
                # barrier that orders the previous token's stores.
                cur = tl.load(base + d, mask=m, other=0.0, cache_modifier=".cg")
                # Untouched cells start the sum from +0.0, as the reference's
                # zero-initialised output does.
                acc = tl.where(cur == float("-inf"), 0.0, cur)
                tl.store(base + d, tl.fma(tl.broadcast_to(qv, [BLOCK_P]), cv, acc),
                         mask=m)

    @triton.jit
    def _posting_scores_blocked(QCROW, QCOL, QVAL, BPTR, CROW, CVAL, OUT,
                                n_rows, stride_out, nblk,
                                DB: tl.constexpr, BLOCK_P: tl.constexpr,
                                BLOCK_I: tl.constexpr):
        """As `_posting_scores`, but program (q, b) owns documents
        [b*DB, (b+1)*DB) of row q; `BPTR[t*nblk + b]` bounds token t's
        postings inside that block (postings are row-sorted per token)."""
        q = tl.program_id(0)
        b = tl.program_id(1)
        base = OUT + q.to(tl.int64) * stride_out
        lo = b * DB
        for d0 in range(0, DB, BLOCK_I):
            o = lo + d0 + tl.arange(0, BLOCK_I)
            # Never past this program's block: a neighbour owns those cells.
            tl.store(base + o, tl.full([BLOCK_I], float("-inf"), tl.float32),
                     mask=(o < n_rows) & (o < lo + DB))
        j0 = tl.load(QCROW + q)
        j1 = tl.load(QCROW + q + 1)
        for j in range(j0, j1):
            tl.debug_barrier()
            t = tl.load(QCOL + j)
            qv = tl.load(QVAL + j)
            k = t.to(tl.int64) * nblk + b
            p0 = tl.load(BPTR + k)
            p1 = tl.load(BPTR + k + 1)
            for p in range(p0, p1, BLOCK_P):
                offs = p + tl.arange(0, BLOCK_P)
                m = offs < p1
                d = tl.load(CROW + offs, mask=m, other=0)
                cv = tl.load(CVAL + offs, mask=m, other=0.0)
                cur = tl.load(base + d, mask=m, other=0.0, cache_modifier=".cg")
                acc = tl.where(cur == float("-inf"), 0.0, cur)
                tl.store(base + d, tl.fma(tl.broadcast_to(qv, [BLOCK_P]), cv, acc),
                         mask=m)

except Exception as exc:  # no triton, or an API that moved
    triton = tl = None
    _posting_scores = None
    _posting_scores_blocked = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"

# Running under Triton's CPU interpreter (tests): the kernels accept CPU
# tensors.
_INTERPRETED = type(_posting_scores).__name__ == "InterpretedFunction"


class KernelMismatch(RuntimeError):
    """A launch disagreed with the cuSPARSE reference in its self-check.
    The kernel is already disabled for the run when this is raised."""


def reset() -> None:
    """Per-run state: a kernel disabled by one run is retried by the next."""
    global _OFF, _LAUNCHES, _SELFCHECKS, _SELFCHECK_FAILURES
    _OFF = None
    _LAUNCHES = 0
    _SELFCHECKS = 0
    _SELFCHECK_FAILURES = 0
    _CHECKED.clear()
    _Q_BOUND.clear()
    _DECLINES.clear()
    global _OOM_STREAK, _OOM_WARNED, _POLICY
    _OOM_STREAK = 0
    _OOM_WARNED = False
    _POLICY = "auto"          # run_compute sets the run's policy after reset()


def note_oom() -> None:
    """A slice ran out of memory on the kernel path and fell back. Warn once;
    after OOM_STREAK_LIMIT in a row, stop trying for the run (each failed
    attempt costs a postings build plus a launch before the cuSPARSE retry).
    Declined slices are not kernel attempts and neither extend nor break the
    streak; a slice scored by the kernel ends it."""
    global _OOM_STREAK, _OOM_WARNED
    _OOM_STREAK += 1
    note_decline("out of memory")
    if not _OOM_WARNED:
        _OOM_WARNED = True
        logger.warning(
            "sparse posting-list kernel ran out of GPU memory on a slice; that "
            "slice is scored on the cuSPARSE path (consider a smaller "
            "params.sparse_batch_size)")
    if _OOM_STREAK >= OOM_STREAK_LIMIT:
        kernel_disable(f"{_OOM_STREAK} consecutive out-of-memory fallbacks")


def note_kernel_slice() -> None:
    """A slice was scored by the kernel (resets the out-of-memory streak)."""
    global _OOM_STREAK
    _OOM_STREAK = 0

def set_policy(policy: str) -> None:
    """Record `params.sparse_kernel` so the manifest says what was asked for."""
    global _POLICY
    _POLICY = policy


def kernel_off() -> str | None:
    """Why the kernel is not in use for this run, or `None`."""
    if os.environ.get("NOVA_BF_NO_SPARSE_POSTING"):
        return "NOVA_BF_NO_SPARSE_POSTING"
    if _posting_scores is None:
        return f"triton unavailable ({_UNAVAILABLE})"
    return _OFF


def kernel_disable(reason: str) -> None:
    global _OFF
    if _OFF is None:
        _OFF = reason
        logger.warning(
            "sparse posting-list kernel is off for the rest of this run: %s. "
            "Sparse slices use the cuSPARSE path from here on.", reason)


def usage() -> dict:
    return {"permitted": (_POLICY != "cusparse"
                          and not os.environ.get("NOVA_BF_NO_SPARSE_POSTING")),
            "policy": _POLICY,
            "launches": _LAUNCHES, "unavailable": kernel_off(),
            "selfchecks": _SELFCHECKS,
            "selfcheck_failures": _SELFCHECK_FAILURES,
            "declines": dict(_DECLINES)}


def note_decline(reason: str) -> None:
    """Count a slice that took the cuSPARSE path, by reason (manifest)."""
    key = reason.split(" (")[0]
    _DECLINES[key] = _DECLINES.get(key, 0) + 1


def decline_reason(q_csr, Cb) -> str | None:
    """Why this slice cannot use the kernel, or `None`. Per-slice reasons are
    counted for the manifest (`usage()["declines"]`)."""
    why = _decline_reason(q_csr, Cb)
    if why is not None:
        note_decline(why)
    return why


def _decline_reason(q_csr, Cb) -> str | None:
    off = kernel_off()
    if off is not None:
        return off
    if q_csr.device != Cb.device:
        return "query and corpus on different devices"
    if Cb.device.type != "cuda" and not _INTERPRETED:
        return "not a CUDA slice"
    if q_csr.values().dtype != Cb.values().dtype or Cb.values().dtype.itemsize != 4:
        return "values are not float32"
    if int(q_csr.shape[1]) != int(Cb.shape[1]):
        return "query and corpus vocabularies differ"
    if DOC_BLOCK and -(-int(Cb.shape[0]) // DOC_BLOCK) > MAX_GRID_Y:
        return f"slice too wide for the launch grid ({int(Cb.shape[0])} rows)"
    return _magnitude_reason(q_csr, Cb)


def _query_bound(q_csr):
    """(max over queries of sum |q|, all query weights finite), cached per run."""
    import torch

    import weakref

    hit = _Q_BOUND.get("q")
    if hit is not None and hit[0]() is q_csr:
        return hit[1], hit[2]
    v = q_csr.values()
    finite = bool(torch.isfinite(v).all()) if v.numel() else True
    if v.numel() and finite:
        rows = torch.repeat_interleave(
            torch.arange(int(q_csr.shape[0]), device=v.device),
            q_csr.crow_indices().diff())
        l1 = torch.zeros(int(q_csr.shape[0]), dtype=torch.float64, device=v.device)
        l1.index_add_(0, rows, v.abs().double())
        qmax = float(l1.max())
    else:
        qmax = 0.0
    _Q_BOUND["q"] = (weakref.ref(q_csr), qmax, finite)   # one run-wide query matrix
    return qmax, finite


def _magnitude_reason(q_csr, Cb) -> str | None:
    """Refuse slices whose partial sums could overflow fp32 (where the `-inf`
    untouched sentinel would stop being exact) or that hold non-finite weights.
    One scalar device read per slice."""
    import torch

    qmax, qfinite = _query_bound(q_csr)
    if not qfinite:
        return "non-finite query weight"
    cv = Cb.values()
    if cv.numel() == 0:
        return None
    cmax = float(cv.abs().max())
    if cmax != cmax or cmax == float("inf"):
        return "non-finite corpus weight"
    if qmax * cmax >= MAX_SUM:
        return f"weights large enough to overflow fp32 sums ({qmax:.3g} x {cmax:.3g})"
    return None


def corpus_postings(Cb):
    """Regroup a CSR slice `(rows, vocab)` by token.

    Returns `(ptr, rows, vals)`: for token `t`, its documents are
    `rows[ptr[t]:ptr[t+1]]` in ascending order with weights `vals[...]`.
    """
    import torch

    n_rows, vocab = int(Cb.shape[0]), int(Cb.shape[1])
    crow = Cb.crow_indices()
    col = Cb.col_indices().to(torch.int64)
    vals = Cb.values()
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=col.device, dtype=torch.int64), crow.diff())
    # (token, row) keys are unique after `_remap_sparse_file`'s coalesce, so a
    # stable sort is not needed for determinism; it is used anyway so equal
    # keys (if any slipped through) keep CSR order.
    order = torch.argsort(col * max(1, n_rows) + row, stable=True)
    counts = torch.bincount(col, minlength=vocab)
    ptr = torch.zeros(vocab + 1, dtype=torch.int64, device=col.device)
    torch.cumsum(counts, 0, out=ptr[1:])
    return ptr, row.index_select(0, order).to(torch.int32), vals.index_select(0, order)


def block_ptr(ptr, prow, vocab: int, n_rows: int, db: int):
    """Per-(token, document block) posting offsets: entry `t*nblk + b` starts
    token t's postings with row in block b. Postings are (token, row) sorted,
    so (token, block) keys are already in order."""
    import torch

    nblk = max(1, -(-n_rows // db))
    tok = torch.repeat_interleave(
        torch.arange(vocab, device=ptr.device, dtype=torch.int64), ptr.diff())
    key = tok * nblk + prow.to(torch.int64) // db
    counts = torch.bincount(key, minlength=vocab * nblk)
    bptr = torch.zeros(vocab * nblk + 1, dtype=torch.int64, device=ptr.device)
    torch.cumsum(counts, 0, out=bptr[1:])
    return bptr, nblk


def _launch(q_csr, ptr, prow, pval, n_rows, out, bptr=None):
    qcrow = q_csr.crow_indices()
    qcol = q_csr.col_indices()
    qval = q_csr.values()
    n_q = int(q_csr.shape[0])
    if n_q == 0 or n_rows == 0:
        return
    if DOC_BLOCK:
        if bptr is None:
            bptr = block_ptr(ptr, prow, int(q_csr.shape[1]), n_rows, DOC_BLOCK)
        bp, nblk = bptr
        _posting_scores_blocked[(n_q, nblk)](
            qcrow, qcol, qval, bp, prow, pval, out,
            n_rows, out.stride(0), nblk,
            DB=DOC_BLOCK, BLOCK_P=min(BLOCK_P, DOC_BLOCK),
            BLOCK_I=min(BLOCK_I, DOC_BLOCK), num_warps=NUM_WARPS,
        )
        return
    _posting_scores[(n_q,)](
        qcrow, qcol, qval, ptr, prow, pval, out,
        n_rows, out.stride(0),
        BLOCK_P=BLOCK_P, BLOCK_I=BLOCK_I, num_warps=NUM_WARPS,
    )


def scores(q_csr, Cb):
    """Raw dot scores `(n_q, n_rows)` float32, `-inf` where no token is shared.

    `q_csr` is the run-wide query CSR (`_SparseQueryCache.values`), `Cb` the
    slice CSR. Callers check `decline_reason` first. Raises `KernelMismatch`
    when the first launch of a configuration fails its self-check.
    """
    import torch

    global _LAUNCHES

    n_q, n_rows = int(q_csr.shape[0]), int(Cb.shape[0])
    from nova_bf.profiling import slice_mark

    out = torch.empty((n_q, n_rows), dtype=torch.float32, device=Cb.device)
    if n_q == 0 or n_rows == 0:
        return out            # nothing to score: no launch, nothing to check
    with slice_mark("bf_sparse_postings"):
        ptr, prow, pval = corpus_postings(Cb)
        bptr = (block_ptr(ptr, prow, int(Cb.shape[1]), n_rows, DOC_BLOCK)
                if DOC_BLOCK and n_rows else None)
    try:
        with slice_mark("bf_sparse_posting_kernel"):
            _launch(q_csr, ptr, prow, pval, n_rows, out, bptr)
    except Exception as exc:
        from nova_bf.twopass import is_oom

        if not is_oom(exc):
            kernel_disable(f"{type(exc).__name__}: {exc}")
        raise
    nblk = bptr[1] if bptr is not None else 0
    key = _compile_key(Cb.device,
                       (n_rows, out.stride(0), nblk),
                       (q_csr.crow_indices(), q_csr.col_indices(), q_csr.values(),
                        bptr[0] if bptr is not None else ptr, prow, pval, out),
                       _launch_config())
    if key not in _CHECKED:
        try:
            _self_check(q_csr, Cb, out)
        except KernelMismatch:
            raise
        except Exception as exc:
            from nova_bf.twopass import is_oom

            # OOM: leave the key unchecked; the caller falls back for this
            # slice (auto) and a later slice re-checks. Any other failure
            # means this configuration is unverified: off.
            if not is_oom(exc):
                kernel_disable(f"self-check could not run: {type(exc).__name__}: {exc}")
            raise
        _CHECKED.add(key)
    _LAUNCHES += 1                # slices actually scored by the kernel
    return out


def _launch_config() -> tuple:
    """The kernel and its constexpr / launch parameters, as `_launch` uses them."""
    if DOC_BLOCK:
        return ("blocked", DOC_BLOCK, min(BLOCK_P, DOC_BLOCK),
                min(BLOCK_I, DOC_BLOCK), NUM_WARPS)
    return ("row", 0, BLOCK_P, BLOCK_I, NUM_WARPS)


def _compile_key(dev, ints, tensors, config=()):
    """What selects a separately compiled kernel: the device, the kernel and
    its constexpr/launch values (raw), Triton's specialisations of the runtime
    integer arguments (== 1, divisible by 16), and every pointer argument's
    alignment and dtype."""
    return (str(dev), tuple(config),
            tuple((int(v) == 1, int(v) % 16 == 0) for v in ints),
            tuple((t.data_ptr() % 16 == 0, str(t.dtype)) for t in tensors))


def _sample_rows(n_q: int, cap: int | None = None) -> list[int]:
    """Deterministic spread of query rows: the first, the last, and evenly
    spaced rows between. `cap` defaults to `SELF_CHECK_ROWS` at call time."""
    cap = SELF_CHECK_ROWS if cap is None else cap
    if n_q <= cap:
        return list(range(n_q))
    step = (n_q - 1) / (cap - 1)
    return sorted({int(round(i * step)) for i in range(cap)})


def reference(q_csr, Cb, rows=None, absolute=False, counts=False):
    """Compute cuSPARSE reference scores with no-overlap masking.
    `absolute=True` uses absolute weights for error bounds.
    `counts=True` also returns shared-token counts."""
    import torch

    from nova_bf.compute import (_dense_slice_t, _iter_csr_row_chunks,
                                 _dense_corpus_rows_per_chunk)

    if rows is not None:
        q_csr = _select_rows(q_csr, rows)
    if absolute:
        q_csr = torch.sparse_csr_tensor(
            q_csr.crow_indices(), q_csr.col_indices(), q_csr.values().abs(),
            size=tuple(q_csr.shape), check_invariants=False)
        Cb = torch.sparse_csr_tensor(
            Cb.crow_indices(), Cb.col_indices(), Cb.values().abs(),
            size=tuple(Cb.shape), check_invariants=False)
    ind = torch.sparse_csr_tensor(
        q_csr.crow_indices(), q_csr.col_indices(),
        torch.ones_like(q_csr.values()), size=tuple(q_csr.shape),
        check_invariants=False)
    per = _dense_corpus_rows_per_chunk(Cb, q_csr.values().element_size())
    ones = torch.ones(Cb.values().numel(), dtype=Cb.values().dtype, device=Cb.device)
    raw, hit = [], []
    for sub in _iter_csr_row_chunks(Cb, per, Cb.values()):
        raw.append(torch.matmul(q_csr, _dense_slice_t(sub, sub.values())))
    for sub in _iter_csr_row_chunks(Cb, per, ones):
        hit.append(torch.matmul(ind, _dense_slice_t(sub, sub.values())))
    raw = torch.cat(raw, 1) if len(raw) > 1 else raw[0]
    shared = torch.cat(hit, 1) if len(hit) > 1 else hit[0]
    raw = raw.masked_fill_(shared == 0, float("-inf"))
    return (raw, shared) if counts else raw


def _select_rows(q_csr, rows):
    import torch

    crow = q_csr.crow_indices()
    idx = torch.as_tensor(rows, dtype=torch.int64, device=crow.device)
    lo, hi = crow.index_select(0, idx), crow.index_select(0, idx + 1)
    counts = hi - lo
    new_crow = torch.zeros(len(rows) + 1, dtype=crow.dtype, device=crow.device)
    torch.cumsum(counts, 0, out=new_crow[1:])
    take = torch.cat([torch.arange(int(a), int(b), device=crow.device)
                      for a, b in zip(lo.tolist(), hi.tolist())]) if len(rows) else \
        torch.zeros(0, dtype=torch.int64, device=crow.device)
    return torch.sparse_csr_tensor(
        new_crow, q_csr.col_indices().index_select(0, take),
        q_csr.values().index_select(0, take),
        size=(len(rows), int(q_csr.shape[1])), check_invariants=False)


def _self_check(q_csr, Cb, out) -> None:
    """Check sampled query rows against cuSPARSE. No-overlap masks must
    match exactly, and scores must agree within FP32 rounding bounds
    based on the number and magnitude of shared-token contributions.
    NaNs fail the check."""
    import torch

    global _SELFCHECKS, _SELFCHECK_FAILURES

    rows = _sample_rows(int(q_csr.shape[0]))
    if not rows:
        return
    ref, shared = reference(q_csr, Cb, rows, counts=True)
    mag = reference(q_csr, Cb, rows, absolute=True)
    got = out.index_select(0, torch.as_tensor(rows, device=out.device))
    _SELFCHECKS += 1
    ref_dead, got_dead = torch.isneginf(ref), torch.isneginf(got)
    bad = ref_dead != got_dead
    live = ~ref_dead & ~got_dead
    eps = torch.finfo(torch.float32).eps
    tol = (2 * shared + 2) * eps * mag.clamp_min(0) + 1e-30
    bad |= live & ~((got - ref).abs() <= tol)
    n_bad = int(bad.sum())
    if n_bad:
        _SELFCHECK_FAILURES += 1
        i = int(bad.nonzero()[0, 0]), int(bad.nonzero()[0, 1])
        msg = (f"self-check: {n_bad} of {bad.numel()} sampled cells disagree "
               f"(first at sampled row {i[0]}, column {i[1]}: kernel "
               f"{float(got[i])!r} vs reference {float(ref[i])!r})")
        if _LAUNCHES > 0:
            msg += ("; earlier slices of this run used the kernel (from "
                    "configurations spot-checked on their first slice), later "
                    "ones use cuSPARSE")
        kernel_disable(msg)
        raise KernelMismatch(msg)
