"""Sparse data handling and exact CPU top-k search.

Decodes sparse vectors into CSR blocks, loads queries, and provides an
exact CPU search engine and verification oracle.

Input rows contain column indices and values of equal length. Indices
must fit int32; values retain float16 or float32, with float64 converted
to float32.

Blocks are packed into a single uint8 buffer with 256-byte-aligned
sections for the header, row offsets, indices, and values. This allows
each block to be transferred to the GPU in one copy.

Queries are canonicalized at load time by sorting indices and summing
duplicates, as required for cuVS. Corpus rows remain unchanged.
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from bench.data.sources import _check_slot_room, arrow_array, list_parquet
from bench.timing import now

MAGIC = int.from_bytes(b"NVSPCSR1", "little")
HEADER_BYTES = 256
ALIGN = 256
INT32_MAX = 2**31 - 1
STORED_VALUE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))
# exact CPU engine / oracle: max (query x corpus row) scores per dense tile
CPU_TILE_CELLS = 1 << 24
# exact_topk refuses wider vocabularies (scipy allocates O(columns) per tile)
CPU_MAX_COLS = 1 << 26


def _al(n):
    """`n` rounded up to a multiple of ALIGN."""
    return -(-int(n) // ALIGN) * ALIGN


def packed_nbytes(n_rows, nnz, value_itemsize):
    """Bytes of a packed block of ``n_rows`` rows and ``nnz`` nonzeros."""
    return (HEADER_BYTES + _al(4 * (n_rows + 1)) + _al(4 * nnz)
            + _al(value_itemsize * nnz))


@dataclass
class Csr:
    """A block's CSR parts: views into its packed bytes (no copy)."""
    n_rows: int
    n_cols: int
    nnz: int
    indptr: np.ndarray   # int32 (n_rows + 1,)
    indices: np.ndarray  # int32 (nnz,)
    values: np.ndarray   # float16 / float32 (nnz,)


def _header(buf):
    return np.ndarray((8,), np.int64, buffer=buf)


def is_packed(x):
    """True for a packed sparse block (1-D uint8 starting with MAGIC)."""
    return (isinstance(x, np.ndarray) and x.ndim == 1 and x.dtype == np.uint8
            and x.nbytes >= HEADER_BYTES and int(_header(x)[0]) == MAGIC)


def packed_rows(x):
    return int(_header(x)[1])


def packed_nnz(x):
    return int(_header(x)[2])


def unpack(x):
    """``Csr`` views of packed block ``x``."""
    if not is_packed(x):
        raise ValueError("not a packed sparse block")
    _, n, nnz, n_cols, vb, o_p, o_i, o_v = (int(v) for v in _header(x))
    vdt = {2: np.float16, 4: np.float32}[vb]
    return Csr(n, n_cols, nnz,
               np.ndarray((n + 1,), np.int32, buffer=x, offset=o_p),
               np.ndarray((nnz,), np.int32, buffer=x, offset=o_i),
               np.ndarray((nnz,), vdt, buffer=x, offset=o_v))


def new_packed(n_rows, nnz, value_dtype, out=None):
    """An empty packed block, placed at the start of ``out`` (e.g. a ring slot) or in a new
    page-aligned Arrow array. Only the header is written (n_cols 0); fill via ``unpack``."""
    vdt = np.dtype(value_dtype)
    nbytes = packed_nbytes(n_rows, nnz, vdt.itemsize)
    if out is None:
        x = arrow_array((nbytes,), np.uint8)
    else:
        _check_slot_room(nbytes, out)
        x = np.ndarray((nbytes,), np.uint8, buffer=out)
    o_p = HEADER_BYTES
    o_i = o_p + _al(4 * (n_rows + 1))
    o_v = o_i + _al(4 * nnz)
    _header(x)[:] = (MAGIC, n_rows, nnz, 0, vdt.itemsize, o_p, o_i, o_v)
    _proj(x)[:] = 0  # not projected (ProjectingCodec marks its blocks)
    return x


def set_n_cols(x, n_cols):
    _header(x)[3] = n_cols


# --sparse-project query-vocab: header int64 slots 8 / 9 hold PROJ_MAGIC and the file's
# nonzero count before projection; both 0 in an unprojected block.
PROJ_MAGIC = int.from_bytes(b"NVSPPRJ1", "little")


def _proj(buf):
    return np.ndarray((2,), np.int64, buffer=buf, offset=64)


def is_projected(x):
    return int(_proj(x)[0]) == PROJ_MAGIC


def packed_stored_nnz(x):
    """Nonzeros as stored in the file (the pre-projection count for a projected block)."""
    p = _proj(x)
    return int(p[1]) if int(p[0]) == PROJ_MAGIC else packed_nnz(x)


# ----------------------------------------------------------------- schema

def sparse_fields(typ):
    """``(indices list type, values list type)`` of a sparse struct type, or None if ``typ`` is not one."""
    if not pa.types.is_struct(typ):
        return None
    names = [typ.field(i).name for i in range(typ.num_fields)]
    if "indices" not in names or "values" not in names:
        return None
    it, vt = typ.field("indices").type, typ.field("values").type
    lists = (pa.types.is_list, pa.types.is_large_list)
    if not (any(f(it) for f in lists) and any(f(vt) for f in lists)):
        return None
    if not pa.types.is_integer(it.value_type) or not pa.types.is_floating(vt.value_type):
        return None
    return it, vt


def stored_value_dtype(typ):
    """Value dtype of a decoded block: float16 / float32 as stored, float64 -> float32."""
    dt = np.dtype(sparse_fields(typ)[1].value_type.to_pandas_dtype())
    return dt if dt in STORED_VALUE_DTYPES else np.dtype(np.float32)


def _check_type(typ, column):
    if sparse_fields(typ) is None:
        raise TypeError(f"column {column!r} is {typ}, expected a sparse struct<indices: "
                        "list<int>, values: list<float>>")


# ----------------------------------------------------------------- decode

def _chunk_parts(chunk, column):
    """(offsets int64 rebased to 0, indices, values) of one struct chunk.

    Refuses null rows, null lists or elements, and rows whose two lists differ in length."""
    if chunk.null_count:
        raise ValueError(f"column {column!r} has {chunk.null_count} null rows")
    il, vl = pc.struct_field(chunk, "indices"), pc.struct_field(chunk, "values")
    if il.null_count or vl.null_count:
        raise ValueError(f"column {column!r}: null indices / values lists")
    oi = il.offsets.to_numpy().astype(np.int64, copy=False)
    ov = vl.offsets.to_numpy().astype(np.int64, copy=False)
    if not np.array_equal(oi - oi[0], ov - ov[0]):
        raise ValueError(f"column {column!r}: a row's indices and values differ in length")
    fi, fv = il.flatten(), vl.flatten()
    if fi.null_count or fv.null_count:
        raise ValueError(f"column {column!r} has null index / value elements")
    return (oi - oi[0], fi.to_numpy(zero_copy_only=False),
            fv.to_numpy(zero_copy_only=False))


def table_csr(table, column, out=None):
    """Decode an Arrow table's sparse column into one packed block.

    Each chunk is copied once, straight into ``out`` (a ring slot) or a new Arrow array.
    Sets n_cols; refuses indices outside [0, 2^31) or more than 2^31 - 1 nonzeros."""
    chunked = table.column(column)
    _check_type(chunked.type, column)
    n = len(chunked)
    if n == 0:
        raise ValueError(f"column {column!r} has no rows")
    parts = [_chunk_parts(c, column) for c in chunked.chunks if len(c)]
    nnz = sum(int(o[-1]) for o, _, _ in parts)
    if nnz > INT32_MAX:
        raise ValueError(f"column {column!r}: {nnz} nonzeros in one file, more than "
                         "int32 CSR (cuVS's sparse brute force) can index")
    vdt = stored_value_dtype(chunked.type)
    x = new_packed(n, nnz, vdt, out)
    c = unpack(x)
    c.indptr[0] = 0
    row = at = 0
    n_cols = 0
    for offs, idx, val in parts:
        m, z = len(offs) - 1, int(offs[-1])
        np.copyto(c.indptr[row + 1:row + m + 1], offs[1:] + at, casting="unsafe")
        if z:
            lo, hi = idx.min(), idx.max()
            if lo < 0 or hi > INT32_MAX:
                raise ValueError(f"column {column!r}: indices in [{lo}, {hi}], outside "
                                 "[0, 2^31)")
            n_cols = max(n_cols, int(hi) + 1)
            np.copyto(c.indices[at:at + z], idx, casting="unsafe")
            np.copyto(c.values[at:at + z], val, casting="same_kind")
        row += m
        at += z
    set_n_cols(x, n_cols)
    return x


def widen_packed(x, dtype):
    """``x`` with its values cast to ``dtype`` (a new packed block); ``x`` itself if already so."""
    c = unpack(x)
    if c.values.dtype == np.dtype(dtype):
        return x
    y = new_packed(c.n_rows, c.nnz, dtype)
    d = unpack(y)
    d.indptr[:] = c.indptr
    d.indices[:] = c.indices
    np.copyto(d.values, c.values, casting="same_kind")
    set_n_cols(y, c.n_cols)
    _proj(y)[:] = _proj(x)  # a projected block stays marked
    return y


class SparseCodec:
    """sources.py's codec for sparse columns: decode, widen values, count rows, and bound the
    block size from the footer. Stateless, so it pickles into reader processes."""

    name = "sparse"

    def decode(self, table, column, dtype, out):
        """Decode straight into ``out``, unless a host cast to ``dtype`` follows (then into a
        new array that the caller widens and copies into the slot)."""
        typ = table.schema.field(column).type
        if out is not None and dtype is not None and sparse_fields(typ) is not None \
                and stored_value_dtype(typ) != np.dtype(dtype):
            out = None
        return table_csr(table, column, out)

    def needs_widen(self, x, dtype):
        return dtype is not None and unpack(x).values.dtype != np.dtype(dtype)

    def widen(self, x, dtype):
        return widen_packed(x, dtype)

    def rows(self, x):
        return packed_rows(x)

    def block_bytes(self, footer, dtype):
        """Upper bound on the packed block a file decodes to (its ring slot size).

        Nonzeros are bounded by the footer's indices value count (it also counts empty
        lists, so it is >= nnz). None when the footer is not a sparse column's."""
        if footer.sparse_nnz is None:
            return None
        vb = np.dtype(dtype).itemsize if dtype is not None else \
            {16: 2}.get(footer.sparse_value_bits, 4)
        return packed_nbytes(footer.num_rows, footer.sparse_nnz, vb)


CODEC = SparseCodec()


# ------------------------------------- query-vocabulary projection (nova-bf's)

PROJECT_MODES = ("off", "query-vocab")


def nova_bf_projection():
    """nova-bf's own projection functions, imported (not copied) from ``nova_bf.compute``.

    Returns ``(_build_query_vocab, _build_vocab_lut, _remap_sparse_file)``.
    Imported only for ``--sparse-project query-vocab``: it pulls in nova-bf's whole compute
    stack (~0.3 s)."""
    try:
        from nova_bf.compute import _build_query_vocab, _build_vocab_lut, _remap_sparse_file
    except ImportError as e:
        raise SystemExit(f"--sparse-project query-vocab imports nova-bf's projection "
                         f"(nova_bf.compute) and it is not importable here ({e}); install "
                         "python/nova-bf in this venv") from e
    return _build_query_vocab, _build_vocab_lut, _remap_sparse_file


def table_csr_projected(table, column, vocab, lut, remap, out=None, stats=None):
    """``table_csr`` with nova-bf's query-vocabulary projection between decode and pack.

    ``remap`` (nova-bf's ``_remap_sparse_file``, with ``lut``) drops columns no query uses,
    renumbers the rest to their position in ``vocab``, and sums duplicates. Every row is kept
    (possibly empty), so row counts and ids do not change. Same input checks as ``table_csr``
    on the stored data; the int32 limit applies to kept nonzeros. n_cols = ``len(vocab)``;
    the header records the stored nonzero count. ``stats.project``: concatenation + remap time."""
    chunked = table.column(column)
    _check_type(chunked.type, column)
    n = len(chunked)
    if n == 0:
        raise ValueError(f"column {column!r} has no rows")
    parts = [_chunk_parts(c, column) for c in chunked.chunks if len(c)]
    stored = sum(int(o[-1]) for o, _, _ in parts)
    for _, idx, _ in parts:
        if len(idx):
            lo, hi = idx.min(), idx.max()
            if lo < 0 or hi > INT32_MAX:
                raise ValueError(f"column {column!r}: indices in [{lo}, {hi}], outside "
                                 "[0, 2^31)")
    vdt = stored_value_dtype(chunked.type)
    t0 = now()
    if len(parts) == 1:
        offs, idx, val = parts[0]
    else:
        at = np.cumsum([0] + [int(o[-1]) for o, _, _ in parts[:-1]])
        offs = np.concatenate([np.zeros(1, np.int64)]
                              + [o[1:] + a for (o, _, _), a in zip(parts, at)])
        idx = np.concatenate([i for _, i, _ in parts])
        val = np.concatenate([v for _, _, v in parts])
    if val.dtype != vdt:  # float64 -> float32, as nova-bf narrows before its remap
        val = val.astype(vdt)
    offs, idx, val = remap(offs, idx, val, vocab, lut)
    if stats is not None:
        stats.add("project", now() - t0)
    kept = len(idx)
    if kept > INT32_MAX:
        raise ValueError(f"column {column!r}: {kept} nonzeros kept in one file, more than "
                         "int32 CSR (cuVS's sparse brute force) can index")
    x = new_packed(n, kept, vdt, out)
    c = unpack(x)
    np.copyto(c.indptr, offs, casting="unsafe")
    np.copyto(c.indices, idx, casting="unsafe")
    np.copyto(c.values, val, casting="same_kind")
    set_n_cols(x, len(vocab))
    _proj(x)[:] = (PROJ_MAGIC, stored)
    return x


class ProjectingCodec(SparseCodec):
    """``SparseCodec`` for ``--sparse-project query-vocab``.

    Decodes each file through ``table_csr_projected`` against ``vocab`` (the queries' sorted
    distinct columns). Only ``vocab`` is pickled; each process builds the lookup table once,
    at reader start-up, outside every timed pass. ``decode_stats`` asks sources.py to pass
    ``stats``. Slot sizes stay the unprojected bounds (a projected block is never larger)."""

    decode_stats = True

    def __init__(self, vocab):
        self.vocab = np.ascontiguousarray(vocab)
        self._prepare()

    def _prepare(self):
        _, build_lut, self._remap = nova_bf_projection()
        self.lut = build_lut(self.vocab)  # None when nova-bf would use searchsorted

    def __getstate__(self):
        return {"vocab": self.vocab}

    def __setstate__(self, state):
        self.vocab = state["vocab"]
        self._prepare()

    def decode(self, table, column, dtype, out, stats=None):
        typ = table.schema.field(column).type
        if out is not None and dtype is not None and sparse_fields(typ) is not None \
                and stored_value_dtype(typ) != np.dtype(dtype):
            out = None
        return table_csr_projected(table, column, self.vocab, self.lut, self._remap, out,
                                   stats)


def project_queries(q):
    """Project the queries for ``--sparse-project query-vocab``: ``(projected Queries, vocab)``.

    Every query entry is in the vocabulary, so nothing is dropped; the mapping is monotone,
    so rows stay sorted and unique. Width ``len(vocab)`` (at least 1)."""
    build_vocab, build_lut, remap = nova_bf_projection()
    vocab = build_vocab(q.indices)
    offs, idx, val = remap(q.indptr.astype(np.int64), q.indices, q.values, vocab,
                           build_lut(vocab))
    if len(idx) != q.nnz:
        raise RuntimeError("query projection dropped query entries (never expected)")
    return Queries(offs.astype(np.int32), np.ascontiguousarray(idx, np.int32),
                   np.ascontiguousarray(val, np.float32), max(1, len(vocab))), vocab


# ----------------------------------------------------------------- queries

@dataclass
class Queries:
    """Canonical CSR queries: float32 values, int32 indices sorted and unique per row."""
    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    n_cols: int

    @property
    def n_rows(self):
        return len(self.indptr) - 1

    @property
    def nnz(self):
        return len(self.indices)

    @property
    def nbytes(self):
        return self.indptr.nbytes + self.indices.nbytes + self.values.nbytes

    def matrix(self, n_cols=None, dtype=np.float32):
        import scipy.sparse as sp
        return sp.csr_matrix((self.values.astype(dtype, copy=False), self.indices,
                              self.indptr), shape=(self.n_rows, n_cols or self.n_cols))


def canonical(indptr, indices, values):
    """(indptr, indices, values) with every row's columns sorted and unique.

    Duplicates are summed in float64, then cast to float32 (nova-bf's
    ``_coalesce_by_row_col``); explicit zeros are kept."""
    import scipy.sparse as sp
    n = len(indptr) - 1
    ncols = int(indices.max()) + 1 if len(indices) else 1
    m = sp.csr_matrix((values.astype(np.float64), indices.astype(np.int64),
                       indptr.astype(np.int64)), shape=(n, ncols))
    m.sum_duplicates()  # sorts too
    return (m.indptr.astype(np.int32), m.indices.astype(np.int32),
            m.data.astype(np.float32))


def load_queries(uri, column, nq=None):
    """The first ``nq`` sparse queries of ``uri``, canonicalised; read once, untimed.

    Reads every *.parquet under ``uri`` in sorted order. Returns ``(Queries, warnings)``."""
    fs, refs = list_parquet(uri)
    parts, n = [], 0
    for r in refs:
        if nq and n >= nq:
            break
        t = pq.read_table(r.path, columns=[column], filesystem=fs)
        c = unpack(table_csr(t, column))
        parts.append((c.indptr.astype(np.int64), c.indices.copy(),
                      c.values.astype(np.float32)))
        n += c.n_rows
    ptr = [np.zeros(1, np.int64)]
    base = 0
    for p, _, _ in parts:
        ptr.append(p[1:] + base)
        base += int(p[-1])
    indptr = np.concatenate(ptr)
    indices = np.concatenate([i for _, i, _ in parts])
    values = np.concatenate([v for _, _, v in parts])
    warn = []
    if nq:
        if nq > len(indptr) - 1:
            warn.append(f"WARNING: --nq {nq} > the {len(indptr) - 1} rows in {uri}; "
                        f"using all {len(indptr) - 1}")
        indptr = indptr[:nq + 1]
        indices, values = indices[:indptr[-1]], values[:indptr[-1]]
    nnz_before = len(indices)
    indptr, indices, values = canonical(indptr, indices, values)
    if len(indices) != nnz_before:
        warn.append(f"NOTE: sparse queries: {nnz_before - len(indices)} duplicate column "
                    "indices summed into their row's first (cuVS would keep one value)")
    empty = int(np.sum(np.diff(indptr) == 0))
    if empty:
        warn.append(f"WARNING: {empty} sparse queries have no nonzeros: every corpus row "
                    "scores 0 for them (cuVS cosine scores two empty rows 1)")
    n_cols = int(indices.max()) + 1 if len(indices) else 1
    return Queries(indptr, indices, values, n_cols), warn


# ------------------------------------------------- exact CPU engine / oracle

def row_norms(m):
    """float64 L2 norm of each row of scipy CSR ``m`` (duplicates summed first)."""
    m = m.astype(np.float64)
    m.sum_duplicates()
    return np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())


def topk_tile(S, k):
    """Top ``min(k, cols)`` of every row of the dense tile ``S``, highest score first.

    Ties go to the lowest column id, including at the k-th place (within this tile; merging
    across blocks uses FAISS's ResultHeap tie order). Returns ``(D, I)``, local ids."""
    nq, c = S.shape
    kk = min(k, c)
    if kk == 0:
        return np.empty((nq, 0), S.dtype), np.empty((nq, 0), np.int64)
    if kk == c:
        sel = np.broadcast_to(np.arange(c, dtype=np.int64), (nq, c))
    else:
        kth = -np.partition(-S, kk - 1, axis=1)[:, kk - 1]
        gt = S > kth[:, None]
        eq = S == kth[:, None]
        need = kk - gt.sum(axis=1)
        take = eq & (np.cumsum(eq, axis=1, dtype=np.int32) <= need[:, None])
        sel = np.nonzero(gt | take)[1].reshape(nq, kk).astype(np.int64)
    vals = np.take_along_axis(S, sel, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(vals, order, axis=1), np.take_along_axis(sel, order, axis=1)


def merge_sorted(D0, I0, D1, I1, k):
    """Merge two per-row top lists whose ids are all ``I0`` < all ``I1``
    (consecutive corpus ranges), keeping the lowest id on equal scores."""
    D = np.concatenate([D0, D1], axis=1)
    I = np.concatenate([I0, I1], axis=1)
    order = np.argsort(-D, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(D, order, axis=1), np.take_along_axis(I, order, axis=1)


def exact_topk(Q, X, k, metric="ip", dtype=np.float32, q_norms=None,
               tile_cells=CPU_TILE_CELLS):
    """Exact top-``min(k, rows)`` of scipy CSR queries ``Q`` against CSR corpus ``X``.

    Tiled over corpus rows (``tile_cells`` scores at once). ``dtype``: float32 for the CPU
    engine, float64 for the oracle. cosine = dot / (|q| |x|), float64 norms, zero norm scores
    0. Ties as in ``topk_tile``. Returns ``(D, I int64)``, ids local to ``X``: D in float32 for
    the engine, float64 for the oracle (never rounded)."""
    if X.shape[1] > CPU_MAX_COLS:
        # scipy's product allocates O(columns) per tile: 16 GiB at 2^31 columns
        raise ValueError(f"the scipy sparse engine / --verify oracle needs the vocabulary "
                         f"width <= {CPU_MAX_COLS} (2^26), got {X.shape[1]} (the largest "
                         "index + 1): remap the indices to a dense vocabulary first")
    Q = Q.astype(dtype)
    X = X.astype(dtype)
    nq, rows = Q.shape[0], X.shape[0]
    if metric == "cosine":
        qn = row_norms(Q) if q_norms is None else q_norms
        xn = row_norms(X)
    step = max(1, tile_cells // max(1, nq))
    D = I = None
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        S = (Q @ X[r0:r1].T).toarray()
        if metric == "cosine":
            den = qn[:, None] * xn[None, r0:r1]
            with np.errstate(divide="ignore", invalid="ignore"):
                S = np.where(den > 0, S / np.where(den > 0, den, 1), 0).astype(dtype)
        d, i = topk_tile(S, k)
        i = i + r0
        D, I = (d, i) if D is None else merge_sorted(D, I, d, i, k)
        del S
    return (D if np.dtype(dtype) == np.float64 else D.astype(np.float32)), I


def block_matrix(csr, n_cols, dtype=np.float32):
    """scipy CSR over a block's ``Csr`` views (``n_cols`` wide)."""
    import scipy.sparse as sp
    return sp.csr_matrix((csr.values.astype(dtype, copy=False), csr.indices, csr.indptr),
                         shape=(csr.n_rows, n_cols))


class Rows:
    """Dense float64 rows of a scipy CSR matrix on demand (``xq[q]`` / ``xb[i]``).

    Unit-L2 with ``normalize`` (cosine). Lets the result check rescore single ids without
    densifying the whole matrix."""

    def __init__(self, m, normalize=False):
        self.m = m.astype(np.float64)
        self.m.sum_duplicates()
        self.norms = row_norms(self.m) if normalize else None

    def __len__(self):
        return self.m.shape[0]

    def __getitem__(self, i):
        r = self.m.getrow(int(i)).toarray().ravel()
        if self.norms is not None and self.norms[i] > 0:
            r /= self.norms[i]
        return r

    def pair_dot(self, other, q, i):
        """float64 score of row ``q`` of ``other`` (the queries' Rows) against row ``i`` here.

        Same value as ``np.dot(other[q], self[i])`` without densifying (~10 ms a pair at a
        250k vocabulary). A zero norm is not divided by, as in the dense form."""
        a, b = other.m.getrow(int(q)), self.m.getrow(int(i))
        s = float(a.multiply(b).sum())
        if other.norms is not None and other.norms[int(q)] > 0:
            s /= other.norms[int(q)]
        if self.norms is not None and self.norms[int(i)] > 0:
            s /= self.norms[int(i)]
        return s


def oneshot_reference(q, blocks, k, metric="ip"):
    """The sparse --verify oracle: all blocks stacked into one scipy matrix, searched in float64.

    Returns ``(D, I, xq, xb)``, with ``xq`` / ``xb`` as ``Rows`` for rescoring."""
    import scipy.sparse as sp
    csrs = [unpack(b.vectors) for b in blocks]
    n_cols = max([q.n_cols] + [c.n_cols for c in csrs])
    X = sp.vstack([block_matrix(c, n_cols, np.float64) for c in csrs], format="csr")
    Q = q.matrix(n_cols, np.float64)
    D, I = exact_topk(Q, X, k, metric, np.float64)
    D, I = pad(D, I, k)  # corpus smaller than k
    return D, I, Rows(Q, metric == "cosine"), Rows(X, metric == "cosine")


def pad(D, I, k):
    """k columns, padded the way FAISS pads (lowest float32, -1)."""
    short = k - D.shape[1]
    if short <= 0:
        return D, I
    return (np.pad(D, ((0, 0), (0, short)), constant_values=np.finfo(np.float32).min),
            np.pad(I, ((0, 0), (0, short)), constant_values=-1))


def corpus_nnz(blocks):
    return sum(unpack(b.vectors).nnz for b in blocks)

