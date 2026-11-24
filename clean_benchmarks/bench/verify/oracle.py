"""The exact --verify reference, and the check of each timed pass against it.

The reference is the float64 top-k of the stored values (exact widening),
computed with numpy and no engine code; ties go to the lowest id.
``oneshot_reference`` is the plain search; ``filtered_oneshot_reference`` is
the same search with every file's filter masks applied (it checks results
against those masks, not the filter's correctness). ``compare_results`` /
``verify_passes`` compare a pass's top-k against it (ties, padding, tie zones).

Imports only numpy, so any check can use it.
"""

import sys

import numpy as np


Q_CHUNK = 256      # queries per score tile
R_TILE = 65536     # corpus rows per score tile


# ---------------------------------------------------------------- exact top-k

def _topk_rows(scores: np.ndarray, k: int):
    """Exact top-k per row of ``scores`` (-inf = not allowed), in the scores' dtype.
    Best first, ties to the lower id; empty slots are (-inf, -1)."""
    nq, n = scores.shape
    D = np.full((nq, k), -np.inf, scores.dtype)
    I = np.full((nq, k), -1, np.int64)
    if n == 0 or nq == 0:
        return D, I
    kk = min(k, n)
    if np.isnan(scores).any():
        # NaN scores rank as not allowed (the tie logic below needs a total order)
        scores = np.where(np.isnan(scores), -np.inf, scores).astype(scores.dtype, copy=False)
    if kk < n:
        # exactly kk rows per query: all rows above the kk-th best value, then the lowest-id
        # rows tied with it (argpartition alone keeps an arbitrary subset of ties).
        # np.nonzero walks row-major, so each query's picks come out in id order.
        v = -np.partition(-scores, kk - 1, axis=1)[:, kk - 1:kk]
        gt = scores > v
        eq = scores == v
        need = kk - gt.sum(axis=1, keepdims=True)
        sel = gt | (eq & (np.cumsum(eq, axis=1) <= need))
        part = np.nonzero(sel)[1].reshape(nq, kk)
    else:
        part = np.broadcast_to(np.arange(n), (nq, n))
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.lexsort((part, -vals), axis=1)   # best first, then lower id
    part = np.take_along_axis(part, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    ok = vals > -np.inf
    D[:, :kk] = np.where(ok, vals, -np.inf)
    I[:, :kk] = np.where(ok, part, -1)
    return D, I


def _merge_topk(D1, I1, D2, I2, k):
    """Merge two (nq, k) top-k lists (global ids): best first, ties to the lower id."""
    D = np.concatenate([D1, D2], axis=1)
    I = np.concatenate([I1, I2], axis=1)
    big = np.iinfo(np.int64).max
    order = np.lexsort((np.where(I >= 0, I, big), -D), axis=1)[:, :k]
    return np.take_along_axis(D, order, axis=1), np.take_along_axis(I, order, axis=1)


# ------------------------------------------------- the float64 --verify oracle
#
# Every --verify reference: float64 scores of the stored values (exact widening), computed
# with numpy (no engine code), ties to the lowest id. Separate from the float32 masked engine.

# max bytes of one float64 row tile (rows x d x 8)
ORACLE_TILE_BYTES = 256 << 20
# peak host bytes per (query, row) score cell: float64 tile (8) + np.partition's negated copy
# and result (16) + filter mask (1) = 25; rounded up to 32
ORACLE_BYTES_PER_CELL = 32


def oracle_row_tile(d, max_rows=None):
    """Rows per float64 tile: fits ORACLE_TILE_BYTES, at most R_TILE and ``max_rows``."""
    t = max(1, min(R_TILE, ORACLE_TILE_BYTES // (8 * max(1, d))))
    return t if max_rows is None else max(1, min(t, max_rows))


def oracle_scratch_bytes(d, nq, max_rows, q_chunk=Q_CHUNK):
    """Scratch host bytes of exact_topk_f64 (beyond queries and D / I): one float64 row tile
    plus one score tile at ORACLE_BYTES_PER_CELL."""
    rt = oracle_row_tile(d, max_rows)
    return rt * d * 8 + min(q_chunk, max(1, nq)) * rt * ORACLE_BYTES_PER_CELL


def unit_rows_f64(x):
    """Normalise float64 rows to unit L2 in place (zero rows stay zero); einsum avoids a
    (rows x d) temporary."""
    n = np.sqrt(np.einsum("ij,ij->i", x, x))[:, None]
    np.divide(x, n, out=x, where=n > 0)
    return x


def f64_rows(x, unit=False):
    """Float64 copy of ``x`` (exact widening), normalised when ``unit``."""
    out = np.array(x, dtype=np.float64)
    return unit_rows_f64(out) if unit else out


class Float64Rows:
    """``rows[i]``: global row i of ``blocks`` (in corpus order) widened to float64, unit L2
    when ``unit``. Lets the result check rescore any id without copying the corpus."""

    def __init__(self, blocks, unit=False):
        self.blocks = list(blocks)
        self.unit = unit
        self.starts = np.cumsum([0] + [len(b) for b in self.blocks])

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, i):
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(i)
        j = int(np.searchsorted(self.starts, i, side="right") - 1)
        return f64_rows(self.blocks[j][i - self.starts[j]][None, :], self.unit)[0]


def exact_topk_f64(xq, blocks, k, unit=False, allowed=None, q_chunk=Q_CHUNK, r_tile=None):
    """Exact float64 top-k of ``xq`` (float64, pre-normalised for cosine) over ``blocks``
    (stored dtype, corpus order; global id = position in their concatenation).

    Each row tile is widened (and normalised if ``unit``) once, scored per query chunk, and
    merged into a running top-k (best first, ties to the lowest id). ``allowed(q0, q1, g0, g1)``
    masks disallowed pairs. Returns (D float64, I int64); empty slots are (-inf, -1)."""
    xq = np.asarray(xq, np.float64)
    nq, d = xq.shape
    D = np.full((nq, k), -np.inf, np.float64)
    I = np.full((nq, k), -1, np.int64)
    rt = r_tile or oracle_row_tile(d, max((len(b) for b in blocks), default=1))
    g = 0
    for b in blocks:
        n = len(b)
        for r0 in range(0, n, rt):
            r1 = min(n, r0 + rt)
            x = f64_rows(b[r0:r1], unit)
            g0, g1 = g + r0, g + r1
            for q0 in range(0, nq, q_chunk):
                q1 = min(nq, q0 + q_chunk)
                s = xq[q0:q1] @ x.T
                if allowed is not None:
                    s = np.where(allowed(q0, q1, g0, g1), s, -np.inf)
                dd, ii = _topk_rows(s, k)
                del s
                ii = np.where(ii >= 0, ii + g0, -1)
                D[q0:q1], I[q0:q1] = _merge_topk(D[q0:q1], I[q0:q1], dd, ii, k)
            del x
        g += n
    return D, I


def oneshot_reference(xq, blocks, k, metric="ip"):
    """The dense --verify oracle: exact float64 top-k (numpy BLAS, no FAISS), block by
    block, each row tile widened exactly to float64; ties go to the lowest id. For cosine,
    queries and rows are normalised in float64 from the values the engines got.
    Memory: one row tile and one score tile beside the float64 queries (no corpus copy).
    Returns ``(D float64, I, xq_ref, xb)``: the float64 (unit for cosine) queries and a
    Float64Rows over the blocks, so compare_results can rescore ids outside the top-k.
    An empty slot (fewer rows than k) is (np.finfo(float32).min, -1), as faiss.knn and
    every engine pad it."""
    unit = metric == "cosine"
    vecs = [b.vectors for b in blocks]
    xq64 = f64_rows(xq, unit)
    D, I = exact_topk_f64(xq64, vecs, k, unit)
    D[I < 0] = np.finfo(np.float32).min
    return D, I, xq64, Float64Rows(vecs, unit)


# ---------------------------------------------------------------- filtered

def _mask_tile(packed, r0, r1):
    """Rows [r0, r1) of packed little-order masks (m, nbytes) -> (m, r1-r0) bool."""
    b0 = r0 >> 3
    bits = np.unpackbits(packed[:, b0:(r1 + 7) >> 3], axis=1, bitorder="little")
    return bits[:, r0 - (b0 << 3):r1 - (b0 << 3)].astype(bool)


class _Allowed:
    """Which (query, global id) pairs the filter allows, over the whole corpus (one
    FilterBlock per file, starting at ``starts``)."""

    def __init__(self, starts, fblocks):
        self.starts = np.asarray(starts, np.int64)
        self.fblocks = fblocks

    def bits(self, qs, gids):
        """Bool per (query, global id) pair."""
        out = np.zeros(len(gids), bool)
        f = np.searchsorted(self.starts, gids, side="right") - 1
        for fi in np.unique(f):
            sel = f == fi
            fb = self.fblocks[fi]
            r = gids[sel] - self.starts[fi]
            # index one byte per pair: gathering whole mask rows first costs
            # len(r) x n_rows/8 bytes (127 GiB on a real run)
            byte = fb.masks[0][r >> 3] if fb.kind == "static" else \
                fb.masks[fb.mask_of_query[qs[sel]], r >> 3]
            out[sel] = ((byte >> (r & 7)) & 1).astype(bool)
        return out

    def disallowed(self, I):
        """(query, id) pairs in a (nq, k) result that the filter forbids."""
        qs, js = np.nonzero(I >= 0)
        gids = I[qs, js]
        ok = self.bits(qs, gids)
        return [(int(q), int(g)) for q, g in zip(qs[~ok], gids[~ok])]


def filtered_oneshot_reference(xq, blocks, k, metric, n_rows=None):
    """Filtered --verify oracle: exact float64 top-k over the whole corpus with every file's
    masks applied (``blocks`` in file order; ``n_rows`` checked against what was read).
    Returns (D_ref, I_ref, float64 queries, Float64Rows over the blocks for rescoring,
    _Allowed)."""
    unit = metric == "cosine"
    starts, fblocks, vecs, n0 = [], [], [], 0
    for b in blocks:
        x = np.asarray(b.vectors)
        vecs.append(x)
        starts.append(n0)
        fblocks.append(b.filter)
        n0 += len(x)
        del b, x
    if n_rows is not None and n0 != n_rows:
        raise RuntimeError(f"oracle: {n0} rows read, {n_rows} expected from the footers")
    xq64 = f64_rows(xq, unit)
    allowed = _Allowed(starts, fblocks)
    st = allowed.starts

    def allowed_tile(q0, q1, r0, r1):
        m = np.zeros((q1 - q0, r1 - r0), bool)
        f0 = int(np.searchsorted(st, r0, side="right") - 1)
        for fi in range(f0, len(fblocks)):
            s0 = int(st[fi])
            if s0 >= r1:
                break
            fb = fblocks[fi]
            a, b_ = max(r0, s0), min(r1, s0 + fb.n_rows)
            if a >= b_:
                continue
            packed = fb.masks[0][None, :] if fb.kind == "static" else \
                fb.masks[fb.mask_of_query[q0:q1]]
            m[:, a - r0:b_ - r0] = _mask_tile(packed, a - s0, b_ - s0)
        return m

    D, I = exact_topk_f64(xq64, vecs, k, unit, allowed_tile)
    # empty slots match the engines' merged results: lowest float32 with id -1
    D[I < 0] = np.finfo(np.float32).min
    return D, I, xq64, Float64Rows(vecs, unit), allowed


# ---------------------------------------------------------------- checking a pass

VERIFY_RTOL = 1e-5  # --verify's relative tolerance on D (np.isclose's rtol)


def compare_results(D, I, D_ref, I_ref, atol=1e-5, xq=None, xb=None):
    """Check an engine's (D, I) against the reference; returns a list of problems (empty = match).

    D_ref / I_ref may carry k+1 columns. tol(s) = atol + VERIFY_RTOL * |s|.
    Every returned (score, id) pair must be within tol of that id's reference score (from the
    reference, or ``xq[q] . xb[id]`` in float64 when the id is outside it), so a wrong order
    or a score paired with the wrong id fails even when the id set is right.

    Two ids whose scores are within tol(a) + tol(b) may legally swap. A query whose k-th
    reference score is more than 2 * tol above the (k+1)-th (or has no (k+1)-th) is
    SEPARATED: its id set must equal the reference top-k. Otherwise it is in a TIE ZONE and
    the ids must be distinct, include every id more than 2 * tol above the (k+1)-th score,
    and every dropped reference id a must be no better than every substituted id b:
    s(a) - tol(a) <= s(b) + tol(b)."""
    k = D.shape[1]
    problems = []
    if not np.allclose(D, D_ref[:, :k], atol=atol, rtol=VERIFY_RTOL):
        bad = np.argwhere(~np.isclose(D, D_ref[:, :k], atol=atol, rtol=VERIFY_RTOL))
        problems.append(f"D mismatch at {len(bad)} entries, first {bad[:3].tolist()}")
    if D_ref.shape[1] > k:
        kth, nxt = D_ref[:, k - 1].astype(np.float64), D_ref[:, k].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            tol = atol + VERIFY_RTOL * np.maximum(np.abs(kth), np.abs(nxt))
            # no (k+1)-th row (-1 padding): nothing to tie with
            sep = (I_ref[:, k] < 0) | (kth - nxt > 2 * tol)
    else:
        sep = np.ones(len(D), bool)
    for q in range(len(D)):
        if sep[q]:
            if set(I[q].tolist()) != set(I_ref[q, :k].tolist()):
                problems.append(f"I mismatch on query {q}")
            else:
                why = _pair_problem(q, D[q], I[q], D_ref[q], I_ref[q], k, atol, xq, xb)
                if why:
                    problems.append(f"I mismatch on query {q} ({why})")
        else:
            why = _tie_zone_problem(q, D[q], I[q], D_ref[q], I_ref[q], k, tol[q], xq, xb,
                                    atol)
            if why:
                problems.append(f"I mismatch on query {q} (tie zone: {why})")
        if len(problems) > 5:
            break
    return problems


def _entry_tol(s, atol):
    """The tolerance of one reference score ``s``: atol + VERIFY_RTOL * |s|."""
    return atol + VERIFY_RTOL * abs(s)


def _ref_score(q, i, ref, xq, xb):
    """Reference score of id ``i`` for query ``q``: from the reference row
    (``ref``: id -> score), else float64 ``xq[q] . xb[i]``; None when it
    cannot be computed."""
    s = ref.get(i)
    if s is None and i >= 0 and xq is not None and xb is not None and i < len(xb):
        if hasattr(xb, "pair_dot"):  # sparse.Rows: sparse float64 dot, no densify
            s = xb.pair_dot(xq, q, i)
        else:
            s = float(np.dot(np.asarray(xq[q], np.float64), np.asarray(xb[i], np.float64)))
    return s


def _pair_problem(q, d, ids, d_ref, i_ref, k, atol, xq, xb, skip_padding=True):
    """Why some returned (score, id) pair of query ``q`` disagrees with the
    id's reference score (see compare_results), or None. -1 padding is
    skipped (the id-set / positional D checks own it) unless
    ``skip_padding`` is False."""
    ref = {int(i): float(s) for i, s in zip(i_ref, d_ref) if i >= 0}
    for pos, i in enumerate(int(i) for i in ids):
        if i < 0 and skip_padding:
            continue
        s = _ref_score(q, i, ref, xq, xb)
        if s is None:
            return f"id {i} outside the reference's top-{k + 1}, cannot be checked"
        if abs(float(d[pos]) - s) > _entry_tol(s, atol):
            return f"id {i} returned with {float(d[pos]):.7g}, its score is {s:.7g}"
    return None


def _tie_zone_problem(q, d, ids, d_ref, i_ref, k, tol, xq, xb, atol=1e-5):
    """Why query ``q``'s ids are not a valid choice within the tolerance (see
    compare_results), or None. ``tol`` is the query's tie-zone tolerance
    (the must-have check), ``atol`` the per-entry one's absolute part."""
    got = [int(i) for i in ids]
    if len(set(got)) != len(got):
        return "duplicate ids"
    must = {int(i) for i, s in zip(i_ref[:k], d_ref[:k]) if i >= 0 and s > d_ref[k] + 2 * tol}
    if must - set(got):
        return f"ids {sorted(must - set(got))} missing (above the tie zone)"
    why = _pair_problem(q, d, ids, d_ref, i_ref, k, atol, xq, xb, skip_padding=False)
    if why:
        return why
    ref = {int(i): float(s) for i, s in zip(i_ref, d_ref) if i >= 0}
    top = {int(i) for i in i_ref[:k] if i >= 0}
    dropped = [(i, ref[i]) for i in top - set(got)]
    extra = [(i, _ref_score(q, i, ref, xq, xb)) for i in set(got) - top]
    if dropped and extra:
        a, sa = max(dropped, key=lambda e: e[1] - _entry_tol(e[1], atol))
        b, sb = min(extra, key=lambda e: e[1] + _entry_tol(e[1], atol))
        if sa - _entry_tol(sa, atol) > sb + _entry_tol(sb, atol):
            return (f"id {a} (score {sa:.7g}) dropped for id {b} (score {sb:.7g}): "
                    "more than their tolerances apart")
    return None


def verify_passes(results, names, D_ref, I_ref, atol, xq, xb):
    """compare_results for every pass: the last pass, and each earlier pass that is not
    bitwise equal to it, against the oracle (a pass that alone went wrong is caught; a legal
    tie-zone difference between passes is reported, not failed). Returns problems per pass."""
    D_last, I_last = results[-1]
    last = compare_results(D_last, I_last, D_ref, I_ref, atol=atol, xq=xq, xb=xb)
    out = []
    for (Dp, Ip), name in zip(results[:-1], names[:-1]):
        if np.array_equal(Dp, D_last) and np.array_equal(Ip, I_last):
            out.append(last)
            continue
        out.append(compare_results(Dp, Ip, D_ref, I_ref, atol=atol, xq=xq, xb=xb))
        print(f"NOTE: --verify: {name}'s results differ from the last pass's; checked "
              f"against the oracle on their own ({'FAILED' if out[-1] else 'ok'})",
              file=sys.stderr)
    return out + [last]

