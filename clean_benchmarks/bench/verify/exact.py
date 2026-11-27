"""Independent float64 scoring and error bounds for correctness verification.

Each modality (dense, sparse, multivector) implements a common interface
for scoring selected query-row pairs, computing float32 error bounds,
and generating exhaustive top-k references. Filtered search uses the
dense scorer.

Corpus files are streamed in bounded batches to limit memory usage.
Scoring uses PyArrow and NumPy independently of the benchmarked systems,
with optional GPU acceleration via PyTorch for dense reference scans.

Exhaustive reference results are cached on disk.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile

import numpy as np

from bench.verify import oracle as O

U32 = 2.0 ** -24          # float32 unit roundoff
PAIR_CELLS = 1 << 24      # (pairs x dims) float64 cells per scoring chunk (128 MiB)
STREAM_BATCH_BYTES = 4 << 20    # stored bytes of one streamed batch of a corpus file
STREAM_BUFFER_BYTES = 32 << 20  # bytes read at a time from a corpus file (one ranged GET on S3)
READ_GROUPS = 16                # needed row groups read at once (one S3 stream each)
BIG_GROUP_BYTES = 512 << 20     # a row group this large (one per file, e.g. FineWeb) is streamed
SPARE = 32                # reference rows kept past k (deep tie groups)
CACHE_VERSION = 4         # bump when the reference's meaning changes (old cache files then miss)
DEVICE_ENV = "BENCH_CHECK_DEVICE"   # auto | cpu | cuda | torch-cpu (scan_device)
TORCH_TILE_CELLS = 1 << 25          # float64 (query x row) scores per torch tile (256 MiB)


def gamma(m: int) -> float:
    """The textbook bound factor of an m-term float32 sum: m u / (1 - m u)."""
    return m * U32 / (1.0 - m * U32)


# ---------------------------------------------------------------------------------------------
# where the dense reference scan computes its score tiles
# ---------------------------------------------------------------------------------------------
def scan_device():
    """(name, (torch, device) or None) for the dense reference scan, from BENCH_CHECK_DEVICE:
      auto       (default) CUDA when torch sees a GPU, else numpy
      cpu        numpy (the code path every check used before the GPU path)
      cuda       torch on the GPU; refused when torch sees none
      torch-cpu  the torch code path on the CPU (to test it without a GPU)
    The torch path (Dense._torch_tile) keeps everything of a file on the device: float64 score
    tiles, masks, score - bound and both running top-K lists (_device_merge); the lists come back
    once per file. Its scores are the device's float64 GEMM, not numpy's: they can differ from
    the numpy path in the last bits (~1e-16 relative), so rows tied to that precision can come
    out in another order; ties among EQUAL device values go to the lowest id, deterministically."""
    want = os.environ.get(DEVICE_ENV, "auto") or "auto"
    if want not in ("auto", "cpu", "cuda", "torch-cpu"):
        raise ValueError(f"{DEVICE_ENV}={want!r}: auto | cpu | cuda | torch-cpu")
    if want == "cpu":
        return "cpu", None
    from bench.record import cudalibs
    torch = cudalibs.torch_first()   # None without torch
    if want == "torch-cpu":
        if torch is None:
            raise ValueError(f"{DEVICE_ENV}=torch-cpu: torch is not installed")
        return "torch-cpu", (torch, torch.device("cpu"))
    if torch is not None and torch.cuda.is_available():
        return "cuda", (torch, torch.device("cuda"))
    if want == "cuda":
        raise ValueError(f"{DEVICE_ENV}=cuda: torch sees no CUDA device")
    return "cpu", None


def _device_merge(torch, rv, ri, v, r0):
    """One tile merged into running top-K lists, on the device. ``rv, ri`` (nq, K): the running
    values and ids (best first, ties to the lowest id, empty slots (-inf, -1)); ``v`` (nq, w): the
    tile's values of rows r0 .. r0 + w - 1 (-inf = not a candidate, NaN read as -inf), all ids
    above the running ones. -> the new (values, ids), same rules.

    float64 leaves no spare bits to pack an id into a sort key, so two steps: (1) pick exactly K
    entries per query, every value above the K-th largest plus the first (lowest position) of
    those equal to it; (2) a stable descending sort of those K. In the concatenation [running,
    tile] equal values sit in ascending id order (the running list is ordered so, the tile's ids
    rise and exceed it), so position order is id order and both steps send ties to the lowest
    id. Only comparisons, a cumulative count and a stable sort: deterministic, whatever order
    topk returns ties in."""
    K = rv.shape[1]
    nq, w = v.shape
    vals = torch.cat([rv, torch.where(torch.isnan(v), float("-inf"), v)], dim=1)
    kth = torch.topk(vals, K, dim=1, sorted=False).values.amin(dim=1, keepdim=True)
    above = vals > kth
    tied = vals == kth
    need = K - above.sum(dim=1, keepdim=True)          # how many ties with the K-th to keep
    take = above | (tied & (tied.cumsum(dim=1) <= need))
    del above, tied
    pos = take.nonzero()[:, 1].view(nq, K)             # exactly K per query, in position order
    del take
    sv = vals.gather(1, pos)
    si = torch.where(pos < K, ri.gather(1, pos.clamp(max=K - 1)), pos - K + r0)
    o = torch.sort(sv, dim=1, descending=True, stable=True).indices
    sv, si = sv.gather(1, o), si.gather(1, o)
    return sv, torch.where(sv > float("-inf"), si, -1)


# ---------------------------------------------------------------------------------------------
# dense scorer
# ---------------------------------------------------------------------------------------------
class Dense:
    """Dense rows (a list<float> column), metric ip or cosine.

    bound(q, ||x||) = gamma(2d + 8) * S, S = ||q|| ||x|| (ip) or 1 (cosine): a float32 dot product
    of d terms is off by at most gamma(d) * sum|q_i x_i| <= gamma(d) ||q|| ||x|| for any summation
    order (cuBLAS split-K included); cosine adds the float32 normalisation of both sides (about
    gamma(d) more: per side about (d/2 + 2) u of scale plus u per component), and 8 u covers the
    final roundings and a score conversion such as 1 - distance."""

    def __init__(self, xq, metric: str, vcol: str):
        if metric not in ("ip", "cosine"):
            raise ValueError(f"metric {metric!r}: ip | cosine")
        self.unit = metric == "cosine"
        self.vcol = vcol
        xq = np.asarray(xq, np.float64)
        self.d = xq.shape[1]
        self.qnorm = np.sqrt(np.einsum("ij,ij->i", xq, xq))
        self.q = O.f64_rows(xq, self.unit)
        self.n = len(self.q)
        self.g = gamma(2 * self.d + 8)
        self._qdev = None      # (device, the queries on it), for the torch scan

    floor = -np.inf        # a pad stands for no row

    def subset(self, qs):
        """The same scorer over queries ``qs`` only (query i of it is query qs[i])."""
        c = copy.copy(self)
        c.q, c.qnorm, c.n, c._qdev = self.q[qs], self.qnorm[qs], len(qs), None
        return c

    def stream_width(self, pf):
        """Stored values per row, for stream_batches."""
        return self.d

    def run_problems(self, corpus):
        return []

    def bound(self, qi, xnorm):
        """Float32 error bound of the scores of queries ``qi`` with rows of norm ``xnorm``."""
        if self.unit:
            return np.full(np.shape(qi), self.g)
        return self.g * self.qnorm[qi] * np.asarray(xnorm, np.float64)

    def batch_rows(self, col, path):
        """One streamed batch's ``vcol`` (a list / fixed-size list array) as an (rows, d) array
        of the stored dtype; a null row or another dim is refused."""
        import pyarrow.compute as pc
        if not len(col):
            return np.zeros((0, self.d), np.float32)
        lens = pc.list_value_length(col)
        if col.null_count or pc.any(pc.not_equal(lens, self.d)).as_py():
            bad = [x for x in lens.to_pylist() if x != self.d][0]
            raise ValueError(f"{path}: corpus dim {bad}, queries {self.d}")
        return pc.list_flatten(col).to_numpy(zero_copy_only=False).reshape(len(col), self.d)

    def score_rows(self, qi, x):
        """exact(q, x) for pairs (queries ``qi``, float64 rows ``x`` in pair order, already
        normalised for cosine by prepare_rows)."""
        return np.einsum("ij,ij->i", self.q[qi], x)

    def prepare_rows(self, x):
        """||x|| of float64 rows ``x``, and (cosine) ``x`` divided by it IN PLACE, a row of norm
        0 or NaN left as it is: bit for bit ``x / np.where(xn > 0, xn, 1.0)[:, None]``."""
        xn = np.sqrt(np.einsum("ij,ij->i", x, x))
        if self.unit:
            np.divide(x, xn[:, None], out=x, where=(xn > 0)[:, None])
        return xn

    def file_pairs(self, path, fs, local, qi, inv):
        """(exact score, ||row||) of pairs (queries ``qi``, rows ``local[inv]``) of one file,
        ``local`` its sorted, unique needed local rows (needed_rows, streamed)."""
        e, xn = np.empty(len(qi)), np.empty(len(qi))
        x = needed_rows(self, path, fs, local)
        xu = self.prepare_rows(x)               # norms; cosine: x normalised in place
        step = max(1, PAIR_CELLS // max(1, self.d))
        for c0 in range(0, len(qi), step):
            c = slice(c0, c0 + step)
            e[c] = self.score_rows(qi[c], x[inv[c]])
            xn[c] = xu[inv[c]]
        return e, xn

    def _torch_tile(self, dev, lists, x, r0, allowed):
        """The numpy tile loop of scan_file on a torch device, in query chunks of
        TORCH_TILE_CELLS scores: the rows widened, the float64 scores (GEMM), the masks
        (non-finite rows / scores, the filter -> -inf), score - bound as Dense.bound computes it,
        and the merge into the running lists (_device_merge), all on the device; nothing comes
        back per tile. ``lists``: [[values, ids], [values, ids]] (nq, K) device tensors, by score
        and by score - bound, updated in place. ``x``: the tile's rows in the stored dtype
        (float32 / float16), copied to the device as they are (half the bytes of float64, or a
        quarter) and widened there (exact). The finite-row flags, the norms and (cosine) the unit
        rows are computed there with the CPU path's definitions: a row with a NaN / inf value is
        never a candidate, a zero row stays zero. -> the tile's norms (a device tensor; the
        caller copies them back once per file). The device's sum of squares may differ from
        numpy's in the last bit (~1e-16 relative), as its scores do."""
        torch, where = dev
        r1 = r0 + len(x)
        if self._qdev is None or self._qdev[0] != where:
            self._qdev = (where, torch.from_numpy(self.q).to(where),
                          torch.from_numpy(self.qnorm).to(where))
        _, qd, qn = self._qdev
        # the stored values to the device; the widened copy is a new tensor, so the caller's
        # buffer (which torch-cpu shares) is never written
        x_d = torch.from_numpy(np.ascontiguousarray(x)).to(where).to(torch.float64)
        fin_d = torch.isfinite(x_d).all(dim=1)
        xn_d = (x_d * x_d).sum(dim=1).sqrt()
        if self.unit:   # as O.unit_rows_f64: divided by the norm iff it is > 0 (NaN: left as is)
            x_d = x_d / torch.where(xn_d > 0, xn_d, 1.0)[:, None]
        step = max(1, TORCH_TILE_CELLS // max(1, r1 - r0))
        for q0 in range(0, self.n, step):
            q1 = min(self.n, q0 + step)
            s = qd[q0:q1] @ x_d.T
            assert s.dtype == torch.float64   # (TF32 only ever applies to float32)
            ok = fin_d[None, :] & torch.isfinite(s)
            if allowed is not None:
                ok &= torch.from_numpy(np.ascontiguousarray(allowed(q0, q1, r0, r1))).to(where)
            s = torch.where(ok, s, float("-inf"))
            del ok
            # the bound as Dense.bound computes it: g, or (g * ||q||) * ||x||
            p = s - (self.g if self.unit else (self.g * qn[q0:q1, None]) * xn_d[None, :])
            for (V, I), v in zip(lists, (s, p)):
                V[q0:q1], I[q0:q1] = _device_merge(torch, V[q0:q1], I[q0:q1], v, r0)
            del s, p
        del x_d, fin_d
        return xn_d

    def scan_file(self, path, fs, K, allowed=None, dev=None):
        """One file's two running lists per query (local row ids), best first, ties to the lowest
        id, empty slots (-inf, -1):
          D, I   the top-K by exact score (the reference);
          P, PI  the top-K by exact score minus the row's bound (what the set rule needs: the best
                 row a system dropped, judged pessimistically; for cosine the same rows as D, I);
        and the norms of the file's rows. ``allowed(q0, q1, r0, r1) -> bool (q1-q0, r1-r0)`` masks
        pairs out; a row with a NaN / inf value, or a score that is not finite, is never a candidate.

        The file is streamed (stream_batches: about STREAM_BATCH_BYTES of stored values at a
        time); its rows are scored in tiles of oracle_row_tile rows counted from the file's first
        row, a tile filled across batches, so the tiles (and the float64 arithmetic) are the
        same as scoring the whole column at once: one batch and one tile in memory.
        ``dev`` (torch, device), from scan_device: the tiles scored and both lists kept there
        (_torch_tile), copied back once at the end of the file; None: numpy."""
        import pyarrow.parquet as pq
        with pq.ParquetFile(path, filesystem=fs, buffer_size=STREAM_BUFFER_BYTES,
                            pre_buffer=False) as pf:
            nr = pf.metadata.num_rows
            n = len(self.q)
            byD, byP = RunningTopK(n, K), RunningTopK(n, K)   # by score; by score - bound
            if dev is not None:   # the same two lists, on the device
                torch, where = dev
                lists = [[torch.full((n, K), float("-inf"), dtype=torch.float64, device=where),
                          torch.full((n, K), -1, dtype=torch.int64, device=where)] for _ in range(2)]
            xn = np.zeros(nr)
            xn_dev = []     # the device path's per-tile norms, copied back once at the end
            rt = O.oracle_row_tile(self.d, max(1, nr))

            def tile(x, r0):
                if dev is not None:   # stored values to the device, widened there
                    xn_dev.append(self._torch_tile(dev, lists, x, r0, allowed))
                    return
                r1 = r0 + len(x)
                fin = np.isfinite(x).all(axis=1)
                raw = np.array(x, np.float64)
                xn[r0:r1] = np.sqrt(np.einsum("ij,ij->i", raw, raw))
                xt = O.unit_rows_f64(raw) if self.unit else raw
                for q0 in range(0, n, O.Q_CHUNK):
                    q1 = min(n, q0 + O.Q_CHUNK)
                    s = self.q[q0:q1] @ xt.T
                    m = fin[None, :] & np.isfinite(s)
                    if allowed is not None:
                        m = m & allowed(q0, q1, r0, r1)
                    s = np.where(m, s, -np.inf)
                    p = s - self.bound(np.arange(q0, q1)[:, None], xn[None, r0:r1])
                    byD.add(q0, q1, s, r0)
                    byP.add(q0, q1, p, r0)
            # batches copied into one tile-sized buffer, scored when full (tile() widens it into
            # its own float64 copy, so the buffer is reused); batches of about a tile's stored
            # bytes, far fewer than the pair reads' small ones
            r0, buf, fill = 0, None, 0
            for b in scan_batches(pf, path, fs, self.vcol, self.d, rt * self.d * 4):
                x = self.batch_rows(b, path)
                del b
                if buf is None:
                    buf = np.empty((rt, self.d), x.dtype)
                j = 0
                while j < len(x):
                    c = min(len(x) - j, rt - fill)
                    buf[fill:fill + c] = x[j:j + c]
                    fill, j = fill + c, j + c
                    if fill == rt:
                        tile(buf, r0)
                        r0, fill = r0 + rt, 0
            if fill:
                tile(buf[:fill], r0)
                r0 += fill
            byD.flush()
            byP.flush()
            D, I, P, PI = byD.A, byD.AI, byP.A, byP.AI
            if dev is not None:   # the file's one device -> host copy (lists and norms)
                (D, I), (P, PI) = ((v.cpu().numpy(), i.cpu().numpy()) for v, i in lists)
                if xn_dev:
                    xn = torch.cat(xn_dev).cpu().numpy()
                del lists, xn_dev
            if r0 != nr:
                raise ValueError(f"{path}: streamed {r0} rows, its footer {nr}")
        return D, I, P, PI, xn, nr


class RunningTopK:
    """A running top-K per query (best first, ties to the lowest id; empty slots (-inf, -1)),
    fed one tile of scores at a time, tiles in row order.

    Until every list of a query chunk is full, a tile goes through the full selection
    (_topk_rows, then a merge). After that only entries strictly above a query's current K-th
    value can still enter: a tie with the K-th loses to it (the tile's rows have higher ids than
    every row already seen) and anything lower never enters. Those few candidates are held and
    merged in one go when PENDING_MAX of them have gathered, or at flush(). Admitting against a
    K-th that is not yet updated only admits more, and the merge is an exact top-K of the union,
    so the lists equal the full selection's, at a fraction of its cost (a sort per tile of the
    whole list was most of the scan's time)."""

    PENDING_MAX = 1 << 21

    def __init__(self, n, K):
        self.K = K
        self.A = np.full((n, K), -np.inf)
        self.AI = np.full((n, K), -1, np.int64)
        self._q, self._i, self._v, self._n = [], [], [], 0

    def add(self, q0, q1, v, r0):
        K = self.K
        kth = self.A[q0:q1, K - 1]
        if not np.isfinite(kth).all():            # a list not yet full (an empty slot is -inf)
            dd, ii = O._topk_rows(v, K)
            ii = np.where(ii >= 0, ii + r0, -1)
            self.A[q0:q1], self.AI[q0:q1] = O._merge_topk(self.A[q0:q1], self.AI[q0:q1], dd, ii, K)
            return
        rows, cols = np.nonzero(v > kth[:, None])
        if len(rows):
            self._q.append(rows + q0)
            self._i.append(cols + r0)
            self._v.append(v[rows, cols])
            self._n += len(rows)
            if self._n >= self.PENDING_MAX:
                self.flush()

    def flush(self):
        if not self._n:
            return
        q, i, v = (np.concatenate(z) for z in (self._q, self._i, self._v))
        self._q, self._i, self._v, self._n = [], [], [], 0
        # a query with more than 4 K pending (its list filled early with a low K-th, e.g. a filter
        # that follows the file's row order) keeps only its best K (ties to the lowest id), so
        # the merge below never pads every query to one query's huge count; the rest (almost
        # always all of them) need no sort beyond grouping
        over = np.bincount(q) > 4 * self.K        # (up to 4 K pending is normal and cheap)
        if over.any():
            big = over[q]
            o = np.lexsort((i[big], -v[big], q[big]))
            qb, ib, vb = q[big][o], i[big][o], v[big][o]
            _, sb, cb = np.unique(qb, return_index=True, return_counts=True)
            kb = (np.arange(len(qb)) - np.repeat(sb, cb)) < self.K
            q, i, v = (np.concatenate([z[~big], zb[kb]]) for z, zb in ((q, qb), (i, ib), (v, vb)))
        o = np.argsort(q, kind="stable")          # grouped by query (the merge sorts each row)
        q, i, v = q[o], i[o], v[o]
        qs, start, cnt = np.unique(q, return_index=True, return_counts=True)
        w = int(cnt.max())
        cd = np.full((len(qs), w), -np.inf)
        ci = np.full((len(qs), w), -1, np.int64)
        pos = np.arange(len(q)) - np.repeat(start, cnt)
        which = np.repeat(np.arange(len(qs)), cnt)
        cd[which, pos], ci[which, pos] = v, i
        self.A[qs], self.AI[qs] = O._merge_topk(self.A[qs], self.AI[qs], cd, ci, self.K)


# ---------------------------------------------------------------------------------------------
# sparse and multivector scorers
# ---------------------------------------------------------------------------------------------
SCAN_CELLS = 1 << 22      # float64 (query x row) scores per ragged scan tile (32 MiB)
SPARSE_PAIRS = 1 << 16    # sparse pairs scored at once
MV_TILE_CELLS = 1 << 22   # float64 token scores per maxsim_matrix tile
MV_PAIR_TOKENS = 1 << 14  # query tokens gathered at once for one document's pairs


class _Block:
    """One decoded piece of a file: ``rows``, per row ``aux`` (the bound's input) and ``fin``
    (no NaN / inf stored value: a row with one is never a candidate), plus the scorer's own parts."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Ragged:
    """What Sparse and MultiVector share: rows decoded by the harness's formats module (one
    streamed batch, or one file's needed rows, at a time) and scored in float64."""

    floor = -np.inf

    def stream_width(self, pf):
        """About how many 4-byte values a row stores (the column's uncompressed bytes in the
        footer / rows / 4), for stream_batches."""
        md = pf.metadata
        tot = sum(md.row_group(i).column(c).total_uncompressed_size
                  for i in range(md.num_row_groups) for c in range(md.num_columns)
                  if md.row_group(i).column(c).path_in_schema.split(".")[0] == self.vcol)
        return max(1, tot // max(1, 4 * md.num_rows))

    def run_problems(self, corpus):
        return []

    def file_pairs(self, path, fs, local, qi, inv):
        """(exact score, aux) of pairs (queries ``qi``, rows ``local[inv]``) of one file."""
        import pyarrow as pa
        cols = [c for _, _, c in needed_arrays(self, path, fs, local)]
        return self.pairs(self.decode(pa.concat_arrays(cols), path), qi, inv)

    def scan_file(self, path, fs, K, allowed=None):
        """As Dense.scan_file (D, I, P, PI, aux of the file's rows, rows), each streamed batch
        decoded and scored in tiles of at most SCAN_CELLS scores. No filter (dense only).
        Always numpy / scipy on the CPU: sparse and MaxSim scores have no GPU path here."""
        import pyarrow.parquet as pq
        if allowed is not None:
            raise ValueError("a filter applies to dense rows only")
        n = self.n
        with pq.ParquetFile(path, filesystem=fs, buffer_size=STREAM_BUFFER_BYTES,
                            pre_buffer=False) as pf:
            nr = pf.metadata.num_rows
            D, P = np.full((n, K), -np.inf), np.full((n, K), -np.inf)
            I, PI = np.full((n, K), -1, np.int64), np.full((n, K), -1, np.int64)
            aux = np.zeros(nr)
            step = max(1, SCAN_CELLS // max(1, min(n, O.Q_CHUNK)))
            r0 = 0
            for b in _batches_of(scan_batches(pf, path, fs, self.vcol, self.stream_width(pf))):
                if not b.num_rows:
                    continue
                blk = self.decode(b.column(0), path)
                del b
                aux[r0:r0 + blk.rows] = blk.aux
                for a in range(0, blk.rows, step):
                    z = min(blk.rows, a + step)
                    for q0 in range(0, n, O.Q_CHUNK):
                        q1 = min(n, q0 + O.Q_CHUNK)
                        s = self.scores(blk, q0, q1, a, z)
                        s = np.where(blk.fin[None, a:z] & np.isfinite(s), s, -np.inf)
                        p = s - self.bound(np.arange(q0, q1)[:, None], blk.aux[None, a:z])
                        for (A, AI), v in (((D, I), s), ((P, PI), p)):
                            dd, ii = O._topk_rows(v, K)
                            ii = np.where(ii >= 0, ii + r0 + a, -1)
                            A[q0:q1], AI[q0:q1] = O._merge_topk(A[q0:q1], AI[q0:q1], dd, ii, K)
                r0 += blk.rows
            if r0 != nr:
                raise ValueError(f"{path}: streamed {r0} rows, its footer {nr}")
        return D, I, P, PI, aux, nr


def _row_finite(values, lengths):
    """Per row: every one of its ``lengths`` (consecutive) ``values`` entries / rows finite."""
    fin = np.isfinite(values)
    if fin.ndim > 1:
        fin = fin.all(axis=1)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    return np.bincount(rows[~fin], minlength=len(lengths)) == 0


class Sparse(_Ragged):
    """Sparse rows (struct<indices, values>), metric ip (dot) or cosine. Queries: the harness's
    canonical ones (sparse.load_queries: duplicates summed, float32); a row's duplicate indices
    are summed in float64. exact = the float64 sum over shared indices; cosine divides by each
    side's float64 norm iff it is nonzero (a zero row scores 0).

    floor 0: a pad stands for score 0 (a row sharing no term with the query scores exactly 0 and
    is a candidate), so a short list fails only when a dropped row clearly beats 0. That and
    nova-bf's candidate rule assume non-negative weights (run_problems).

    bound, q with nq stored nonzeros, row x with nx:
      ip      aux = ||x||, B = gamma(nq + 8) ||q|| ||x||   (a sum of at most nq products)
      cosine  aux = nx,    B = gamma(2 nq + nx + 8)        (the dot, both float32 norms, roundings)"""

    floor = 0.0

    def __init__(self, q, metric: str, vcol: str):
        import scipy.sparse as sp
        from bench.data.formats import sparse as F
        if metric not in ("ip", "cosine"):
            raise ValueError(f"metric {metric!r}: ip | cosine")
        self.unit, self.vcol, self.qv = metric == "cosine", vcol, q.values
        self.n, self.ncols = q.n_rows, max(1, q.n_cols)
        self.Q = sp.csr_matrix((q.values.astype(np.float64), q.indices, q.indptr),
                               shape=(self.n, self.ncols))
        self.qnorm = F.row_norms(self.Q)
        self.nq = np.diff(q.indptr).astype(np.int64)

    def subset(self, qs):
        """The same scorer over queries ``qs`` only (run_problems still sees every query)."""
        c = copy.copy(self)
        c.Q, c.qnorm, c.nq, c.n = self.Q[qs], self.qnorm[qs], self.nq[qs], len(qs)
        return c

    def bound(self, qi, aux):
        qi = np.asarray(qi, np.int64)
        if self.unit:
            return gamma(2 * self.nq[qi] + np.asarray(aux, np.float64) + 8)
        return gamma(self.nq[qi] + 8) * self.qnorm[qi] * np.asarray(aux, np.float64)

    def decode(self, col, path):
        import pyarrow as pa
        import scipy.sparse as sp
        from bench.data.formats import sparse as F
        c = F.unpack(F.table_csr(pa.table({"v": col}), "v"))
        w = max(self.ncols, c.n_cols)
        nx = np.diff(c.indptr).astype(np.int64)            # stored nonzeros, duplicates included
        X = F.block_matrix(c, w, np.float64)
        X.sum_duplicates()
        xn = F.row_norms(X)
        Qw = self.Q if w == self.ncols else \
            sp.csr_matrix((self.Q.data, self.Q.indices, self.Q.indptr), shape=(self.n, w))
        fin = _row_finite(c.values, nx)
        aux = np.where(fin, nx if self.unit else xn, 0.0)   # a non-candidate's bound: 0
        return _Block(rows=c.n_rows, aux=aux, fin=fin, X=X, xn=xn, Qw=Qw)

    def _cos(self, s, qn, xn):
        """Cosine: ``s`` divided by each side's norm iff nonzero, in place."""
        np.divide(s, qn, out=s, where=qn > 0)
        np.divide(s, xn, out=s, where=xn > 0)
        return s

    def scores(self, blk, q0, q1, a, z):
        s = (blk.Qw[q0:q1] @ blk.X[a:z].T).toarray()
        return self._cos(s, self.qnorm[q0:q1, None], blk.xn[None, a:z]) if self.unit else s

    def pairs(self, blk, qi, inv):
        e = np.empty(len(qi))
        for c0 in range(0, len(qi), SPARSE_PAIRS):
            q, r = qi[c0:c0 + SPARSE_PAIRS], inv[c0:c0 + SPARSE_PAIRS]
            s = np.asarray(blk.Qw[q].multiply(blk.X[r]).sum(axis=1), np.float64).ravel()
            if self.unit:
                s = self._cos(s, self.qnorm[q], blk.xn[r])
            e[c0:c0 + len(q)] = np.where(blk.fin[r] & np.isfinite(s), s, -np.inf)
        return e, blk.aux[inv]

    def run_problems(self, corpus):
        """The floor-0 rule and nova-bf's candidate rule assume non-negative weights: every query
        and the FIRST corpus file (a cheap sample; the whole corpus would be a full pass)."""
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        out = []
        if self.qv.size and float(np.min(self.qv)) < 0:
            out.append(f"{int((self.qv < 0).sum())} negative query weight(s) (min "
                       f"{float(np.min(self.qv))!r}): the checks assume non-negative weights")
        if corpus.paths:
            t = pq.read_table(corpus.paths[0], columns=[self.vcol], filesystem=corpus.fs)
            v = pc.list_flatten(pc.struct_field(t.column(self.vcol).combine_chunks(), "values"))
            mn = pc.min(v).as_py() if len(v) else None
            if mn is not None and mn < 0:
                out.append(f"{pc.sum(pc.less(v, 0)).as_py()} negative corpus weight(s) (min {mn!r}) "
                           f"in {corpus.paths[0]} (the first corpus file): the checks assume "
                           "non-negative weights")
        return out


class MultiVector(_Ragged):
    """Multivector rows (list<list<float>>, one inner list per token), MaxSim, metric ip or
    cosine (every token unit-normalised in float64, a zero token stays zero): score(q, doc) = the
    sum over the query's T tokens of the max over the doc's tokens of the token dot product. A
    doc with no token is not a candidate; a query with no token has no candidate. floor -inf.

    bound (d = token dims): each token dot is off by at most gamma(2d + 8) ||q_t|| ||d_j||
    (normalisation included), the max keeps that per-candidate bound, and the sum of T maxima
    adds gamma(T) sum_t |max_t| <= gamma(T) sum_t ||q_t|| maxnorm:
      ip      aux = the doc's largest token norm, B = (gamma(2d + 8) + gamma(T)) (sum_t ||q_t||) aux
      cosine  aux = 1 (unused),                   B = (gamma(2d + 8) + gamma(T)) T"""

    def __init__(self, q, metric: str, vcol: str):
        from bench.data.formats import multivector as F
        if metric not in ("ip", "cosine"):
            raise ValueError(f"metric {metric!r}: ip | cosine")
        self.unit, self.vcol = metric == "cosine", vcol
        self.n, self.d = q.n_rows, q.dim
        self.qoff = np.asarray(q.offsets, np.int64)
        self.T = np.diff(self.qoff)
        t = np.asarray(q.tokens, np.float64)
        self.qtok = F.unit_tokens(t, np.float64) if self.unit else t
        tn = np.sqrt(np.einsum("ij,ij->i", t, t))
        self.qsum = np.bincount(np.repeat(np.arange(self.n), self.T), weights=tn, minlength=self.n)
        self.g = gamma(2 * self.d + 8)

    def subset(self, qs):
        """The same scorer over queries ``qs`` only (their tokens gathered in that order)."""
        qs = np.asarray(qs, np.int64)
        c = copy.copy(self)
        c.T = self.T[qs]
        c.qoff = np.concatenate([[0], np.cumsum(c.T)]).astype(np.int64)
        c.qtok = self.qtok[np.repeat(self.qoff[qs] - c.qoff[:-1], c.T) + np.arange(c.qoff[-1])]
        c.qsum, c.n = self.qsum[qs], len(qs)
        return c

    def bound(self, qi, aux):
        qi = np.asarray(qi, np.int64)
        f = self.g + gamma(self.T[qi])
        if self.unit:
            return f * self.T[qi]
        return f * self.qsum[qi] * np.asarray(aux, np.float64)

    def decode(self, col, path):
        import pyarrow as pa
        from bench.data.formats import multivector as F
        r = F.unpack(F.table_ragged(pa.table({"v": col}), "v"))
        if r.n_tok and len(self.qtok) and r.dim != self.d:
            raise ValueError(f"{path}: corpus token width {r.dim}, queries {self.d}")
        off = np.asarray(r.offsets, np.int64)
        lens = np.diff(off)
        t = np.asarray(r.tokens, np.float64)
        fin = _row_finite(t, lens)
        mx = np.zeros(r.n_rows)                      # the largest token norm of each doc
        nz = lens > 0
        if nz.any():
            mx[nz] = np.maximum.reduceat(np.sqrt(np.einsum("ij,ij->i", t, t)), off[:-1][nz])
        mx = np.where(fin, mx, 0.0)                  # a non-candidate's bound: 0
        return _Block(rows=r.n_rows, aux=np.ones(r.n_rows) if self.unit else mx, fin=fin,
                      off=off, tok=F.unit_tokens(t, np.float64) if self.unit else t)

    def scores(self, blk, q0, q1, a, z):
        from bench.data.formats import multivector as F
        lo, hi = blk.off[a], blk.off[z]
        return F.maxsim_matrix(self.qoff[q0:q1 + 1] - self.qoff[q0],
                               self.qtok[self.qoff[q0]:self.qoff[q1]],
                               blk.off[a:z + 1] - lo, blk.tok[lo:hi], np.float64, MV_TILE_CELLS)

    def pairs(self, blk, qi, inv):
        """One doc at a time against the queries paired with it (at most MV_PAIR_TOKENS query
        tokens gathered per call), by the scan's maxsim_matrix."""
        from bench.data.formats import multivector as F
        e = np.full(len(qi), -np.inf)
        o = np.argsort(inv, kind="stable")
        cut = np.flatnonzero(np.diff(inv[o])) + 1
        for grp in np.split(o, cut):
            j = int(inv[grp[0]])
            lo, hi = int(blk.off[j]), int(blk.off[j + 1])
            if hi == lo or not blk.fin[j]:
                continue                           # not a candidate for any query
            dtok, doff = blk.tok[lo:hi], np.array([0, hi - lo], np.int64)
            ctok = np.cumsum(self.T[qi[grp]])
            for part in np.split(grp, np.searchsorted(ctok, np.arange(MV_PAIR_TOKENS, ctok[-1],
                                                                       MV_PAIR_TOKENS), side="right")):
                if not len(part):
                    continue
                qs = qi[part]
                qo = np.concatenate([[0], np.cumsum(self.T[qs])])
                take = np.repeat(self.qoff[qs] - qo[:-1], self.T[qs]) + np.arange(qo[-1])
                s = F.maxsim_matrix(qo, self.qtok[take], doff, dtok, np.float64, MV_TILE_CELLS)[:, 0]
                e[part] = np.where(np.isfinite(s), s, -np.inf)
        return e, blk.aux[inv]


# ---------------------------------------------------------------------------------------------
# the corpus as the checks see it
# ---------------------------------------------------------------------------------------------
class Corpus:
    """The run's corpus files (the first ``subset.files`` of them, as the legs read) and the
    global row id of each file's first row."""

    def __init__(self, fs, paths, offsets):
        self.fs, self.paths = fs, list(paths)
        self.offsets = np.asarray(offsets, np.int64)
        if len(self.offsets) != len(self.paths) + 1:
            raise ValueError(f"{len(self.paths)} files, {len(self.offsets) - 1} indexed")

    @property
    def rows(self) -> int:
        return int(self.offsets[-1])

    def file_of(self, gids):
        return np.searchsorted(self.offsets, np.asarray(gids, np.int64), side="right") - 1

    def identity(self) -> list:
        """(path, size, mtime, footer hash) per file: what the reference cache key holds. The
        footer (parquet metadata incl. column statistics) is cheap to read and changes with the
        values far more often than size and mtime do (a same-size rewrite with the time kept)."""
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor

        def one(p):
            info = self.fs.get_file_info(p)
            with self.fs.open_input_file(p) as fh:
                md = pq.read_metadata(fh)
            foot = hashlib.sha256(repr(md.to_dict()).encode()).hexdigest()
            return [p, int(info.size or 0), str(info.mtime), foot]
        # files read on threads (one footer GET each on S3), listed in their order; at most the
        # checks' thread count (check.run sets Arrow's CPU count to check_threads)
        import pyarrow as pa
        with ThreadPoolExecutor(max(1, min(32, pa.cpu_count(), len(self.paths)))) as ex:
            return list(ex.map(one, self.paths))


def stream_batches(pf, vcol, d, row_groups=None, batch_bytes=None, threads=False):
    """Record batches of ``vcol`` of an open ParquetFile (``row_groups``, default all) of about
    ``batch_bytes`` (default STREAM_BATCH_BYTES) stored values each (at least one row), decoded
    one at a time."""
    typ = pf.schema_arrow.field(vcol).type     # KeyError for a missing column
    try:
        item = max(1, typ.value_type.bit_width // 8)
    except (AttributeError, ValueError):
        item = 4
    rows = max(1, (batch_bytes or STREAM_BATCH_BYTES) // max(1, int(d) * item))
    groups = list(range(pf.metadata.num_row_groups)) if row_groups is None else list(row_groups)
    if not groups:
        return iter(())
    return pf.iter_batches(batch_size=rows, row_groups=groups, columns=[vcol], use_threads=threads)


def needed_arrays(scorer, path, fs, local):
    """The stored ``vcol`` of the file's rows ``local`` (sorted, unique local row numbers), as
    (first, end, arrow array of rows local[first:end]) pieces in order: the row groups holding
    one are streamed in batches (stream_batches), each batch's needed rows taken, the batch
    dropped before the next, so a file stored as one row group is never held whole."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not len(local):
        return
    with pq.ParquetFile(path, filesystem=fs, buffer_size=STREAM_BUFFER_BYTES,
                        pre_buffer=False) as pf:
        md = pf.metadata
        sizes = np.array([md.row_group(i).num_rows for i in range(md.num_row_groups)], np.int64)
        ends = np.cumsum(sizes)
        if not len(ends) or local[-1] >= ends[-1] or local[0] < 0:
            raise ValueError(f"{path}: row {int(local[-1])} beyond its "
                             f"{int(ends[-1]) if len(ends) else 0} rows")
        starts = ends - sizes
        rg = np.searchsorted(ends, local, side="right")
        sel = np.unique(rg)
        base = np.zeros(md.num_row_groups, np.int64)  # where each selected group starts in the read
        base[sel] = np.concatenate([[0], np.cumsum(sizes[sel])[:-1]])
        pos = local - starts[rg] + base[rg]          # each needed row's place in the read (sorted)
        if len(sel) > 1 and max(md.row_group(int(g)).total_byte_size for g in sel) <= BIG_GROUP_BYTES:
            yield from _parallel_groups(path, fs, scorer.vcol, sel, starts, rg, local)
            return
        p0 = j = 0
        for b in stream_batches(pf, scorer.vcol, scorer.stream_width(pf), sel.tolist()):
            n = b.num_rows
            j1 = int(np.searchsorted(pos, p0 + n, side="left"))
            if j1 > j:
                yield j, j1, b.column(0).take(pa.array(pos[j:j1] - p0))
                j = j1
            del b
            p0 += n
        if j != len(local):
            raise ValueError(f"{path}: read {p0} rows of its selected row groups, "
                             f"{len(local) - j} needed rows missing")


def scan_batches(pf, path, fs, vcol, d, batch_bytes=None):
    """A whole file's ``vcol`` in row order, as arrow arrays: its row groups read READ_GROUPS at a
    time (a sliding window, one S3 stream each), or, when a row group exceeds BIG_GROUP_BYTES (a
    file stored as one row group), streamed in batches so it is never held whole."""
    import collections
    import concurrent.futures as cf
    import pyarrow.parquet as pq
    md = pf.metadata
    groups = list(range(md.num_row_groups))
    if len(groups) <= 1 or max(md.row_group(g).total_byte_size for g in groups) > BIG_GROUP_BYTES:
        for b in stream_batches(pf, vcol, d, batch_bytes=batch_bytes, threads=True):
            yield b.column(0)
        return

    def read(g):
        with pq.ParquetFile(path, filesystem=fs, pre_buffer=True) as f:
            return f.read_row_group(g, columns=[vcol], use_threads=False).column(0).combine_chunks()
    with cf.ThreadPoolExecutor(min(READ_GROUPS, len(groups))) as ex:
        window = collections.deque(ex.submit(read, g) for g in groups[:READ_GROUPS])
        nxt = len(window)
        while window:
            col = window.popleft().result()
            if nxt < len(groups):
                window.append(ex.submit(read, groups[nxt]))
                nxt += 1
            yield col
            del col


class _Batch:
    """An arrow array posing as a one-column record batch (the ragged scans read b.column(0),
    b.num_rows)."""

    def __init__(self, col):
        self._c, self.num_rows = col, len(col)

    def column(self, i):
        return self._c


def _batches_of(cols):
    for c in cols:
        yield _Batch(c)


def _parallel_groups(path, fs, vcol, sel, starts, rg, local):
    """needed_arrays' pieces read READ_GROUPS row groups at a time (a sliding window, so at most
    that many are held): S3 serves one stream at a time far below its bandwidth. Each row
    group is read whole, its needed rows taken; the same rows, values and order as the
    streamed path."""
    import collections
    import concurrent.futures as cf
    import pyarrow as pa
    import pyarrow.parquet as pq

    def read(g):
        with pq.ParquetFile(path, filesystem=fs, pre_buffer=True) as pf:
            return pf.read_row_group(int(g), columns=[vcol], use_threads=False).column(0).combine_chunks()
    bounds = np.searchsorted(rg, np.append(sel, sel[-1] + 1), side="left")
    with cf.ThreadPoolExecutor(min(READ_GROUPS, len(sel))) as ex:
        window = collections.deque()
        it = iter(range(len(sel)))
        for _ in range(READ_GROUPS):
            i = next(it, None)
            if i is None:
                break
            window.append((i, ex.submit(read, sel[i])))
        while window:
            i, fut = window.popleft()
            col = fut.result()
            nxt = next(it, None)
            if nxt is not None:
                window.append((nxt, ex.submit(read, sel[nxt])))
            j, j1 = int(bounds[i]), int(bounds[i + 1])
            yield j, j1, col.take(pa.array(local[j:j1] - starts[sel[i]]))
            del col


def needed_rows(scorer, path, fs, local):
    """Dense: float64 ``(len(local), d)`` of the stored ``vcol`` of the file's rows ``local``
    (needed_arrays), each piece widened straight into the result. The same stored values in the
    same order as reading those rows whole (widening float16 / float32 to float64 is exact)."""
    out = np.empty((len(local), scorer.d), np.float64)
    for j, j1, col in needed_arrays(scorer, path, fs, local):
        out[j:j1] = scorer.batch_rows(col, path)
    return out


def score_pairs(scorer, corpus: Corpus, qi, gids):
    """exact(q, row) and the row's ``aux`` (what its bound needs: dense its norm) of each pair
    (arrays in, arrays out, in pair order). Reads only the row groups that hold a pair, streamed
    (needed_arrays), one file at a time."""
    import pyarrow.parquet as pq
    qi = np.asarray(qi, np.int64)
    gids = np.asarray(gids, np.int64)
    e = np.full(len(gids), np.nan)
    xn = np.full(len(gids), np.nan)
    if not len(gids):
        return e, xn
    if gids.min() < 0 or gids.max() >= corpus.rows:
        raise ValueError("score_pairs: a row id outside the corpus")
    f_of = corpus.file_of(gids)
    for f in np.unique(f_of):
        sel = np.flatnonzero(f_of == f)
        p = corpus.paths[f]
        with corpus.fs.open_input_file(p) as fh:
            nr = pq.read_metadata(fh).num_rows
        if nr != corpus.offsets[f + 1] - corpus.offsets[f]:
            raise ValueError(f"{p}: {nr} rows, the corpus index "
                             f"{corpus.offsets[f + 1] - corpus.offsets[f]}")
        ur, inv = np.unique(gids[sel] - corpus.offsets[f], return_inverse=True)
        e[sel], xn[sel] = scorer.file_pairs(p, corpus.fs, ur, qi[sel], inv)
    return e, xn


# ---------------------------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------------------------
class Filter:
    """The search's filter, evaluated per file by the harness's own filter code (DuckDB, the
    nova-bf filter config): ``allowed(qi, gids)`` for pairs, ``tile(f)`` for the scan."""

    def __init__(self, job, flt, corpus: Corpus):
        self.job, self.flt, self.corpus = job, flt, corpus
        self._res = {}
        self.qsel = None     # subset(): query i here is query qsel[i] of the search

    def subset(self, qs):
        """The same filter for a scorer over queries ``qs`` only (Dense.subset)."""
        c = copy.copy(self)    # shares the evaluated file
        c.qsel = np.asarray(qs, np.int64)
        return c

    def _q(self, qi):
        return qi if self.qsel is None else self.qsel[qi]

    def _file(self, f):
        import pyarrow.parquet as pq
        if f not in self._res:
            self._res.clear()   # one file's result held at a time
            self._res[f] = self.flt.evaluate(
                pq.read_table(self.corpus.paths[f], columns=self.job.fields, filesystem=self.corpus.fs))
        return self._res[f]

    def allowed(self, qi, gids):
        qi = np.asarray(qi, np.int64)
        gids = np.asarray(gids, np.int64)
        out = np.zeros(len(gids), bool)
        f_of = self.corpus.file_of(gids)
        for f in np.unique(f_of):
            sel = np.flatnonzero(f_of == f)
            r = gids[sel] - self.corpus.offsets[f]
            res = self._file(int(f))
            byte = res.static[r >> 3] if not res.per_query else \
                res.combo_masks[res.combo_ids[self._q(qi[sel])], r >> 3]
            out[sel] = ((byte >> (r & 7)) & 1).astype(bool)   # little bit order
        return out

    def tile(self, f):
        """allowed(q0, q1, r0, r1) over file f's local rows, for Dense.scan_file."""
        res = self._file(f)

        def allowed(q0, q1, r0, r1):
            packed = res.static[None, :] if not res.per_query else \
                res.combo_masks[res.combo_ids[self._q(np.arange(q0, q1))]]
            m = O._mask_tile(packed, r0, r1)
            return np.broadcast_to(m, (q1 - q0, r1 - r0)) if m.shape[0] == 1 else m
        return allowed


# ---------------------------------------------------------------------------------------------
# the exhaustive reference
# ---------------------------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def reference_key(corpus: Corpus, queries_path, query_rows, vcol, qcol, metric, modality,
                  filter_id, K, sample="all") -> str:
    """The cache key of a reference. ``sample``: the queries it covers, "all" or
    {"seed", "size", "n"} (check.sample_queries; the sample is a function of these three)."""
    if "://" in queries_path:   # an object store: its size and modification time
        import pyarrow.fs as pafs
        fs, p = pafs.FileSystem.from_uri(queries_path)
        info = fs.get_file_info(p)
        q = [queries_path, int(info.size or 0), str(info.mtime)]
    else:
        q = _sha256(queries_path)
    doc = {"v": CACHE_VERSION, "files": corpus.identity(), "queries": q, "query_rows": query_rows,
           "vcol": vcol, "qcol": qcol, "metric": metric, "modality": modality,
           "filter": filter_id, "K": int(K), "sample": sample}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


def build_reference(scorer, corpus: Corpus, K, flt: Filter | None = None):
    """The true float64 top-K of every query over every file (one file at a time, a running
    top-K; filter applied before the top-K). -> dict of (n, K) arrays: D, I the reference (exact
    score, global ids), N its rows' ``aux`` (dense: norms), and P, PI the top-K by exact score minus bound (see
    Dense.scan_file), and "device": where the score tiles were computed (scan_device; sparse
    and multivector always "cpu"). Empty slot: (-inf, -1)."""
    n = scorer.n
    name, dev = scan_device() if isinstance(scorer, Dense) else ("cpu", None)
    D = np.full((n, K), -np.inf)
    I = np.full((n, K), -1, np.int64)
    N = np.full((n, K), np.nan)
    P = np.full((n, K), -np.inf)
    PI = np.full((n, K), -1, np.int64)
    for f, p in enumerate(corpus.paths):
        Df, If, Pf, PIf, xn, nrows = scorer.scan_file(p, corpus.fs, K, flt.tile(f) if flt else None,
                                                      **({"dev": dev} if dev else {}))
        if nrows != corpus.offsets[f + 1] - corpus.offsets[f]:
            raise ValueError(f"{p}: {nrows} rows, the corpus index {corpus.offsets[f + 1] - corpus.offsets[f]}")
        Nf = np.where(If >= 0, xn[np.maximum(If, 0)], np.nan) if nrows else np.full(If.shape, np.nan)
        Ig = np.where(If >= 0, If + corpus.offsets[f], -1)
        P, PI = O._merge_topk(P, PI, Pf, np.where(PIf >= 0, PIf + corpus.offsets[f], -1), K)
        # merge (best first, ties to the lowest id), carrying the norms
        DD, II, NN = (np.concatenate(z, axis=1) for z in ((D, Df), (I, Ig), (N, Nf)))
        big = np.iinfo(np.int64).max
        o = np.lexsort((np.where(II >= 0, II, big), -DD), axis=1)[:, :K]
        D, I, N = (np.take_along_axis(z, o, axis=1) for z in (DD, II, NN))
    return {"D": D, "I": I, "N": N, "P": P, "PI": PI, "device": name}


def cached_reference(cache_dir, key, build, shape):
    """(reference dict, hit): the cached reference for ``key`` (every array of ``shape``), else
    ``build()`` saved atomically. A cache file that cannot be read or has another shape is rebuilt.
    ``device`` is where the reference was built ("?" for a cache file older than that record)."""
    path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.isfile(path):
        try:
            z = np.load(path)
            ref = {k: z[k] for k in ("D", "I", "N", "P", "PI")}
            if all(v.shape == tuple(shape) for v in ref.values()):
                ref["device"] = str(z["device"]) if "device" in z.files else "?"
                return ref, True
        except (OSError, ValueError, KeyError, EOFError):
            pass   # unreadable or wrong shape: rebuilt and replaced below
    ref = build()
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".npz.tmp")
    os.close(fd)
    with open(tmp, "wb") as fh:
        np.savez(fh, **ref)
    os.replace(tmp, path)
    return ref, False
