"""Multi-vector data handling and exact MaxSim search.

Decodes ragged token embeddings, loads queries, and provides CPU top-k
search and a float64 verification oracle.

Each document contains a variable number of token vectors. Values are
stored as float16 or float32, with float64 converted to float32.
MaxSim sums each query token's highest similarity to a document token.
Cosine search normalizes tokens first; empty queries or documents score -inf.

Blocks are packed into a single uint8 buffer with 256-byte-aligned
sections for the header, document offsets, and token embeddings.
"""


import math
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bench.data.sources import _check_slot_room, arrow_array, list_parquet

MAGIC = int.from_bytes(b"NVMVRAG1", "little")
HEADER_BYTES = 256
ALIGN = 256
STORED_VALUE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))
# exact CPU engine / oracle: max (query token x corpus token) scores per tile
CPU_TILE_CELLS = 1 << 24
NEG_INF = float("-inf")
FLT_LOWEST = float(np.finfo(np.float32).min)


def _al(n):
    """`n` rounded up to a multiple of ALIGN."""
    return -(-int(n) // ALIGN) * ALIGN


def packed_nbytes(n_rows, n_values, value_itemsize):
    """Bytes of a packed block of ``n_rows`` documents and ``n_values`` (= n_tok * dim) values."""
    return HEADER_BYTES + _al(8 * (n_rows + 1)) + _al(value_itemsize * n_values)


@dataclass
class Ragged:
    """A block's ragged parts: views into its packed bytes (no copy)."""
    n_rows: int
    n_tok: int
    dim: int
    offsets: np.ndarray  # int64 (n_rows + 1,)
    tokens: np.ndarray   # float16 / float32 (n_tok, dim)

    @property
    def lengths(self):
        return np.diff(self.offsets)


def _header(buf):
    return np.ndarray((8,), np.int64, buffer=buf)


def is_packed(x):
    """True for a packed multi-vector block (1-D uint8 starting with MAGIC)."""
    return (isinstance(x, np.ndarray) and x.ndim == 1 and x.dtype == np.uint8
            and x.nbytes >= HEADER_BYTES and int(_header(x)[0]) == MAGIC)


def packed_rows(x):
    return int(_header(x)[1])


def packed_tokens(x):
    return int(_header(x)[2])


def header_fields(x):
    """``(n_rows, n_tok, dim, value_itemsize, offsets_off, tokens_off)``."""
    h = _header(x)
    return tuple(int(v) for v in h[1:7])


def unpack(x):
    """``Ragged`` views of packed block ``x``."""
    if not is_packed(x):
        raise ValueError("not a packed multi-vector block")
    n, nt, dim, vb, o_o, o_t = header_fields(x)
    vdt = {2: np.float16, 4: np.float32}[vb]
    return Ragged(n, nt, dim, np.ndarray((n + 1,), np.int64, buffer=x, offset=o_o),
                  np.ndarray((nt, dim), vdt, buffer=x, offset=o_t))


def new_packed(n_rows, n_tok, dim, value_dtype, out=None):
    """An empty packed block for ``n_rows`` documents / ``n_tok`` tokens of width ``dim``.

    Placed at the start of ``out`` (e.g. a ring slot) or in a new page-aligned Arrow array.
    Only the header is written; fill via ``unpack``."""
    vdt = np.dtype(value_dtype)
    nbytes = packed_nbytes(n_rows, n_tok * dim, vdt.itemsize)
    if out is None:
        x = arrow_array((nbytes,), np.uint8)
    else:
        _check_slot_room(nbytes, out)
        x = np.ndarray((nbytes,), np.uint8, buffer=out)
    o_o = HEADER_BYTES
    o_t = o_o + _al(8 * (n_rows + 1))
    _header(x)[:] = (MAGIC, n_rows, n_tok, dim, vdt.itemsize, o_o, o_t, 0)
    return x


# ----------------------------------------------------------------- schema

def _is_list(t):
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def mv_value_type(typ):
    """Float value type of a ``list<list<float>>`` type (inner may be fixed_size_list), else None."""
    if not _is_list(typ):
        return None
    inner = typ.value_type
    if not (_is_list(inner) or pa.types.is_fixed_size_list(inner)):
        return None
    vt = inner.value_type
    return vt if pa.types.is_floating(vt) else None


def stored_value_dtype(typ):
    """Value dtype of a decoded block: float16 / float32 as stored, float64 -> float32."""
    dt = np.dtype(mv_value_type(typ).to_pandas_dtype())
    return dt if dt in STORED_VALUE_DTYPES else np.dtype(np.float32)


def _check_type(typ, column):
    if mv_value_type(typ) is None:
        raise TypeError(f"column {column!r} is {typ}, expected a multi-vector "
                        "list<list<float>> (one inner list per token)")


# ----------------------------------------------------------------- decode

def _chunk_parts(chunk, column):
    """(token counts per document, tokens, dim) of one ``list<list<float>>`` chunk.

    ``tokens`` is (n_tok, dim) (a view when contiguous); both None without tokens. A null
    document counts 0 tokens by its validity bit, not its offsets (Arrow does not promise a
    null slot's offsets are equal); any tokens it carries are dropped. Refuses null token
    vectors / values and mixed widths."""
    outer_off = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    lengths = np.diff(outer_off)
    keep = None
    if chunk.null_count:
        valid = np.asarray(chunk.is_valid())
        if not bool((lengths[~valid] == 0).all()):
            keep = np.repeat(valid, lengths)  # drop tokens a null slot carries
        lengths = np.where(valid, lengths, 0)
    tok_lo, tok_hi = int(outer_off[0]), int(outer_off[-1])
    if tok_hi == tok_lo:
        return lengths, None, None
    inner = chunk.values  # one entry per token, including tokens of null slots
    # Keep all tokens [tok_lo, tok_hi), or, if a null slot carries tokens (never from parquet),
    # only valid documents' tokens, so the checks below never refuse a dropped token.
    vals = inner.values
    fixed = pa.types.is_fixed_size_list(inner.type)
    inner_off = None if fixed else \
        inner.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if keep is None:
        # common case: kept tokens are contiguous, no per-token index needed
        if inner.null_count and bool(np.asarray(inner.is_null())[tok_lo:tok_hi].any()):
            raise ValueError(f"column {column!r}: a null token vector")
        if fixed:
            dim = inner.type.list_size
            v_lo = (inner.offset + tok_lo) * dim
        else:
            dim = int(inner_off[tok_lo + 1] - inner_off[tok_lo])
            if not bool((np.diff(inner_off[tok_lo:tok_hi + 1]) == dim).all()):
                raise ValueError(f"column {column!r}: token vectors of different widths "
                                 f"(the first is {dim} wide)")
            v_lo = int(inner_off[tok_lo])
        n_rows = tok_hi - tok_lo
        starts = None
    else:
        rows = np.arange(tok_lo, tok_hi)[keep]
        if not len(rows):
            return lengths, None, None
        if inner.null_count and bool(np.asarray(inner.is_null())[rows].any()):
            raise ValueError(f"column {column!r}: a null token vector")
        if fixed:
            dim = inner.type.list_size
            starts = (inner.offset + rows) * dim
        else:
            dim = int(inner_off[rows[0] + 1] - inner_off[rows[0]])
            if not bool((inner_off[rows + 1] - inner_off[rows] == dim).all()):
                raise ValueError(f"column {column!r}: token vectors of different widths "
                                 f"(the first is {dim} wide)")
            starts = inner_off[rows]
        v_lo = int(starts[0])
        n_rows = len(rows)
    if dim == 0:
        raise ValueError(f"column {column!r}: zero-width token vectors")
    contiguous = starts is None
    v_hi = v_lo + n_rows * dim if contiguous else int(starts[-1]) + dim
    if vals.null_count:
        nulls = np.asarray(vals.is_null())
        bad = nulls[v_lo:v_hi].any() if contiguous else \
            nulls[(starts[:, None] + np.arange(dim)).ravel()].any()
        if bool(bad):
            raise ValueError(f"column {column!r} has null token values")
    flat = vals.to_numpy(zero_copy_only=False)
    if contiguous:
        tok = flat[v_lo:v_hi].reshape(n_rows, dim)
    else:
        tok = flat[starts[:, None] + np.arange(dim)]  # gathered: a copy
    return lengths, tok, dim


def table_ragged(table, column, out=None):
    """Decode an Arrow table's multi-vector column into one packed block.

    Each chunk's tokens are copied once, straight into ``out`` (a ring slot) or a new Arrow
    array."""
    chunked = table.column(column)
    _check_type(chunked.type, column)
    n = len(chunked)
    if n == 0:
        raise ValueError(f"column {column!r} has no rows")
    parts = [_chunk_parts(c, column) for c in chunked.chunks if len(c)]
    dims = {d for _, _, d in parts if d is not None}
    if len(dims) > 1:
        raise ValueError(f"column {column!r}: token widths {sorted(dims)} in one file")
    dim = dims.pop() if dims else 0
    n_tok = sum(int(l.sum()) for l, _, _ in parts)
    vdt = stored_value_dtype(chunked.type)
    x = new_packed(n, n_tok, dim, vdt, out)
    r = unpack(x)
    r.offsets[0] = 0
    row = at = 0
    for lengths, tok, _ in parts:
        m = len(lengths)
        np.cumsum(lengths, out=r.offsets[row + 1:row + m + 1])
        r.offsets[row + 1:row + m + 1] += at
        if tok is not None:
            z = tok.shape[0]
            np.copyto(r.tokens[at:at + z], tok, casting="same_kind")
            at += z
        row += m
    return x


def widen_packed(x, dtype):
    """``x`` with its tokens cast to ``dtype`` (a new packed block); ``x`` itself if already so."""
    r = unpack(x)
    if r.tokens.dtype == np.dtype(dtype):
        return x
    y = new_packed(r.n_rows, r.n_tok, r.dim, dtype)
    s = unpack(y)
    s.offsets[:] = r.offsets
    np.copyto(s.tokens, r.tokens, casting="same_kind")
    return y


class MultiVectorCodec:
    """sources.py's codec for multi-vector columns: decode, widen values, count rows, and
    bound the block size from the footer. Stateless, so it pickles into reader processes."""

    name = "multivector"

    def decode(self, table, column, dtype, out):
        typ = table.schema.field(column).type
        if out is not None and dtype is not None and mv_value_type(typ) is not None \
                and stored_value_dtype(typ) != np.dtype(dtype):
            out = None  # a host cast follows: decode into a new array, widen it later
        return table_ragged(table, column, out)

    def needs_widen(self, x, dtype):
        return dtype is not None and unpack(x).tokens.dtype != np.dtype(dtype)

    def widen(self, x, dtype):
        return widen_packed(x, dtype)

    def rows(self, x):
        return packed_rows(x)

    def block_bytes(self, footer, dtype):
        """Upper bound on the packed block a file decodes to (its ring slot size).

        Values are bounded by the footer's leaf value count (it also counts null / empty
        lists, so it is >= n_tok * dim). None when the footer is not a multi-vector column's."""
        if footer.mv_values is None:
            return None
        vb = np.dtype(dtype).itemsize if dtype is not None else \
            {16: 2}.get(footer.mv_value_bits, 4)
        return packed_nbytes(footer.num_rows, footer.mv_values, vb)


CODEC = MultiVectorCodec()


# ----------------------------------------------------------------- queries

@dataclass
class Queries:
    """Multi-vector queries: float32 tokens, ragged by ``offsets``."""
    offsets: np.ndarray  # int64 (n + 1,)
    tokens: np.ndarray   # float32 (n_tok, dim)

    @property
    def n_rows(self):
        return len(self.offsets) - 1

    @property
    def n_tok(self):
        return self.tokens.shape[0]

    @property
    def dim(self):
        return self.tokens.shape[1]

    @property
    def lengths(self):
        return np.diff(self.offsets)

    @property
    def nbytes(self):
        return self.offsets.nbytes + self.tokens.nbytes


def unit_tokens(tokens, dtype=np.float32):
    """L2-normalise every token (norm in float64, result in ``dtype``); zero stays zero."""
    t = np.asarray(tokens, np.float64)
    n = np.sqrt(np.einsum("ij,ij->i", t, t))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(n[:, None] > 0, t / np.where(n > 0, n, 1)[:, None], 0.0)
    return u.astype(dtype)


def load_queries(uri, column, nq=None):
    """The first ``nq`` multi-vector queries of ``uri``, tokens as float32; read once, untimed.

    Reads every *.parquet under ``uri`` in sorted order. Returns ``(Queries, warnings)``."""
    fs, refs = list_parquet(uri)
    offs, toks, n, dim = [], [], 0, None
    for r in refs:
        if nq and n >= nq:
            break
        t = pq.read_table(r.path, columns=[column], filesystem=fs)
        g = unpack(table_ragged(t, column))
        if g.n_tok:
            if dim is not None and g.dim != dim:
                raise ValueError(f"{r.path}: query token width {g.dim}, earlier files {dim}")
            dim = g.dim
        offs.append(g.offsets.copy())
        toks.append(g.tokens.astype(np.float32))
        n += g.n_rows
    ptr, base = [np.zeros(1, np.int64)], 0
    for o in offs:
        ptr.append(o[1:] + base)
        base += int(o[-1])
    offsets = np.concatenate(ptr)
    dim = dim or 0
    tokens = np.concatenate([t.reshape(-1, dim) for t in toks]) if base else \
        np.zeros((0, dim), np.float32)
    warn = []
    if nq:
        if nq > len(offsets) - 1:
            warn.append(f"WARNING: --nq {nq} > the {len(offsets) - 1} rows in {uri}; "
                        f"using all {len(offsets) - 1}")
        offsets = offsets[:nq + 1]
        tokens = tokens[:offsets[-1]]
    tokens = np.ascontiguousarray(tokens)
    empty = int(np.sum(np.diff(offsets) == 0))
    if empty:
        warn.append(f"WARNING: {empty} multi-vector queries have no tokens: they retrieve "
                    "nothing (-inf against every document, as nova-bf)")
    return Queries(offsets, tokens), warn


# ------------------------------------------------- exact CPU engine / oracle

def topk_tile(S, k):
    """``sparse.topk_tile``: top ``min(k, cols)`` of every row, highest first, ties by ascending column."""
    from bench.data.formats import sparse
    return sparse.topk_tile(S, k)


def maxsim_matrix(q_off, q_tok, d_off, d_tok, dtype=np.float32,
                  tile_cells=CPU_TILE_CELLS):
    """Dense ``(nq, n_docs)`` MaxSim scores of ragged queries against ragged documents.

    ``dtype``: float32 for the CPU engine, float64 for the oracle. Per (query group, document
    tile): token scores ``q_tok @ d_tok.T``, max over each document's tokens, sum over each
    query's tokens. Tiles hold at most ``tile_cells`` scores, unless a single query x document
    pair is larger. Zero-token documents and queries score ``-inf``."""
    nq, n = len(q_off) - 1, len(d_off) - 1
    S = np.full((nq, n), NEG_INF, dtype)
    q_len, d_len = np.diff(q_off), np.diff(d_off)
    if not n or not int(q_len.sum()):
        return S
    gmax = max(1, math.isqrt(tile_cells))
    a = 0
    while a < nq:
        # queries [a, b): at most gmax tokens, at least one query
        b = int(np.searchsorted(q_off, q_off[a] + gmax, side="right")) - 1
        b = min(max(b, a + 1), nq)
        qs = a + np.flatnonzero(q_len[a:b])  # queries with tokens
        if len(qs):
            # their token rows are contiguous (empty queries hold none)
            Q = np.asarray(q_tok[q_off[qs[0]]:q_off[qs[-1] + 1]], dtype)
            q_starts = q_off[qs] - q_off[qs[0]]
            step = max(1, tile_cells // Q.shape[0])
            i = 0
            while i < n:
                j = int(np.searchsorted(d_off, d_off[i] + step, side="right")) - 1
                j = min(max(j, i + 1), n)
                lo, hi = int(d_off[i]), int(d_off[j])
                ds = np.flatnonzero(d_len[i:j])  # documents with tokens, tile-local
                if len(ds):
                    tok = Q @ np.asarray(d_tok[lo:hi], dtype).T     # (q tokens, d tokens)
                    m = np.maximum.reduceat(tok, d_off[i + ds] - lo, axis=1)
                    del tok
                    S[np.ix_(qs, i + ds)] = np.add.reduceat(m, q_starts, axis=0)
                i = j
        a = b
    return S


def exact_topk(q_off, q_tok, d_off, d_tok, k, dtype=np.float32, tile_cells=CPU_TILE_CELLS):
    """Exact top-``min(k, docs)`` by MaxSim (``maxsim_matrix``), ties as in ``topk_tile``.

    ``-inf`` scores come back as FAISS's padding (lowest float32, -1). Returns
    ``(D float32, I int64)``, ids local to the block."""
    S = maxsim_matrix(q_off, q_tok, d_off, d_tok, dtype, tile_cells)
    D, I = topk_tile(S, k)
    return to_padding(D.astype(np.float32), I)


def to_padding(D, I):
    """``-inf`` entries -> FAISS's ``(lowest float32, -1)`` padding (in place)."""
    neg = np.isneginf(D)
    if neg.any():
        D[neg] = FLT_LOWEST
        I[neg] = -1
    return D, I


class Rows:
    """float64 MaxSim of one (query, document) pair on demand.

    Used by the result check as ``xb.pair_dot(xq, q, i)``. Tokens are widened (and normalised
    for cosine) per call, never all at once."""

    def __init__(self, offsets, tokens_of, n, normalize=False):
        self.offsets = offsets      # global token offsets (n + 1,), or None
        self.tokens_of = tokens_of  # i -> (len_i, dim) array of row i's tokens
        self.n = n
        self.normalize = normalize

    @classmethod
    def of_queries(cls, q, normalize=False):
        return cls(q.offsets, lambda i: q.tokens[q.offsets[i]:q.offsets[i + 1]],
                   q.n_rows, normalize)

    @classmethod
    def of_blocks(cls, blocks, normalize=False):
        """Rows over every block's documents, in corpus order."""
        rag = [unpack(b.vectors) for b in blocks]
        starts = np.cumsum([0] + [r.n_rows for r in rag])

        def tokens_of(i):
            b = int(np.searchsorted(starts, i, side="right")) - 1
            r, j = rag[b], i - int(starts[b])
            return r.tokens[r.offsets[j]:r.offsets[j + 1]]
        return cls(None, tokens_of, int(starts[-1]), normalize)

    def __len__(self):
        return self.n

    def tokens(self, i):
        t = np.asarray(self.tokens_of(int(i)), np.float64)
        return unit_tokens(t, np.float64) if self.normalize else t

    def pair_dot(self, other, q, i):
        a, b = other.tokens(q), self.tokens(i)
        if not len(a) or not len(b):
            return NEG_INF
        return float((a @ b.T).max(axis=1).sum())


def oneshot_reference(q, blocks, k, metric="ip"):
    """The multi-vector --verify oracle: every block's MaxSim scores in float64.

    Blocks are merged in corpus order (ties to the lowest id). Returns ``(D, I, xq, xb)``, with
    ``xq`` / ``xb`` as ``Rows`` for rescoring."""
    from bench.data.formats import sparse
    cos = metric == "cosine"
    q_tok = unit_tokens(q.tokens, np.float64) if cos else q.tokens.astype(np.float64)
    D = I = None
    base = 0
    for b in blocks:
        r = unpack(b.vectors)
        d_tok = unit_tokens(r.tokens, np.float64) if cos else r.tokens.astype(np.float64)
        S = maxsim_matrix(q.offsets, q_tok, r.offsets, d_tok, np.float64)
        d, i = topk_tile(S, k)
        i = i + base
        D, I = (d, i) if D is None else sparse.merge_sorted(D, I, d, i, k)
        base += r.n_rows
        del S, d_tok
    D, I = to_padding(D, I)   # float64 scores, never rounded (as the dense oracle)
    D, I = sparse.pad(D, I, k)
    return (D, I, Rows.of_queries(q, cos), Rows.of_blocks(blocks, cos))


def corpus_tokens(blocks):
    return sum(unpack(b.vectors).n_tok for b in blocks)
