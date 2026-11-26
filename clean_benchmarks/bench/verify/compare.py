#!/usr/bin/env python3
"""Compare top-k results across nova-bf, baseline engines, and dataset ground truth.

Maps all results to global corpus row IDs and compares rankings, overlap,
scores, and result validity.

Optional float64 rescoring resolves disagreements independently of the
engines. Differences within a floating-point tie tolerance are accepted;
real ranking disagreements, score mismatches, and invalid results fail.
Dataset ground truth is reference-only and never causes a failure.

Supports dense, sparse, multivector, and filtered search.
"""

import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

KS = (1, 10, 100, 1000)
SHARED_SCORE_ATOL = 1e-5   # = agree.py SCORE_ATOL (the oracle's per-score tolerance)
SHARED_SCORE_RTOL = 1e-5   # = oracle.py VERIFY_RTOL
MODALITIES = ("dense", "sparse", "multivector")


def harness_modules():
    """The harness's ``(sparse, multivec)`` decode and float64 scoring modules (imported
    lazily; they do not import faiss)."""
    from bench.data.formats import multivector as multivec
    from bench.data.formats import sparse
    return sparse, multivec


def uuid_of(key, row):
    h = hashlib.md5(f"{key}:{row}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def corpus_index(uri, threads=32, need_ids=True, max_files=None):
    """(keys, rows, offsets, id column as one arrow array or None). Without
    ``need_ids`` only the footers are read (row counts). ``max_files``: only the first N files
    by sorted path (a subset run's files: the legs' --max-files)."""
    fs, root = pafs.FileSystem.from_uri(uri)
    root = root.rstrip("/")
    infos = [i for i in fs.get_file_info(pafs.FileSelector(root, recursive=True))
             if i.type == pafs.FileType.File and i.path.endswith(".parquet")]
    infos.sort(key=lambda i: i.path)
    if max_files:
        infos = infos[:int(max_files)]
    s3 = fs.type_name == "s3"
    keys = [i.path.split("/", 1)[1] if s3 else i.path for i in infos]

    def ids(i):
        return pq.read_table(i.path, filesystem=fs, columns=["id"]).column("id").combine_chunks()

    def nrows(i):
        with fs.open_input_file(i.path) as fh:
            return pq.ParquetFile(fh).metadata.num_rows
    with ThreadPoolExecutor(threads) as ex:
        if need_ids:
            cols = list(ex.map(ids, infos))
            rows = np.array([len(c) for c in cols], dtype=np.int64)
            id_col = pa.concat_arrays([c.cast(pa.string()) for c in cols])
        else:
            rows = np.array(list(ex.map(nrows, infos)), dtype=np.int64)
            id_col = None
    offsets = np.concatenate([[0], np.cumsum(rows)])
    return keys, rows, offsets, id_col


def uuid_table(keys, rows, offsets) -> pa.Array:
    """nova-bf's point id (uuid_of) of every indexed corpus row, one string array whose
    position is the global row id. DuckDB's md5 (multi-threaded, the same hex digest of the
    same UTF-8 bytes as hashlib's) computes them in bulk: uuid_of per row in Python took most
    of a check's time on a large corpus."""
    import duckdb
    files = pa.table({"k": pa.array([str(k) for k in keys], pa.string()),
                      "n": pa.array(np.asarray(rows, np.int64)),
                      "o": pa.array(np.asarray(offsets[:-1], np.int64))})
    h = "md5(k || ':' || CAST(r AS VARCHAR))"
    u = (f"substr({h}, 1, 8) || '-' || substr({h}, 9, 4) || '-' || substr({h}, 13, 4) || '-' "
         f"|| substr({h}, 17, 4) || '-' || substr({h}, 21, 12)")
    con = duckdb.connect()
    try:
        con.register("files", files)
        t = con.execute(f"SELECT o + r AS gid, {u} AS u FROM "
                        "(SELECT k, o, unnest(range(n)) AS r FROM files) ORDER BY gid").arrow()
    finally:
        con.close()
    if hasattr(t, "read_all"):   # a RecordBatchReader (newer duckdb)
        t = t.read_all()
    gid = t.column("gid").to_numpy()
    if not np.array_equal(gid, np.arange(int(offsets[-1]), dtype=gid.dtype)):
        raise SystemExit("uuid_table: the generated row ids are not 0..rows-1")
    return t.column("u").combine_chunks().cast(pa.string())


def uuid_gids(uuids, table: pa.Array) -> np.ndarray:
    """Global row ids of nova-bf hit ids (an arrow string array): their position in
    uuid_table's array; -1 for an id no indexed row has."""
    idx = pc.index_in(uuids.cast(pa.string()), value_set=table)
    return pc.fill_null(idx, -1).to_numpy(zero_copy_only=False).astype(np.int64)


def load_novabf(path, u2g):
    """nova-bf's result with its hit ids as global row ids. ``u2g``: uuid_table's array."""
    t = pq.read_table(path, columns=["query_id", "hit_ids", "hit_scores"])
    h = t.column("hit_ids").combine_chunks()
    s = t.column("hit_scores").combine_chunks()
    g = uuid_gids(pc.list_flatten(h), u2g)
    oh, os_ = h.offsets.to_numpy(), s.offsets.to_numpy()
    sf = pc.list_flatten(s).to_numpy(zero_copy_only=False)
    return (t.column("query_id").to_pylist(),
            [g[oh[i]:oh[i + 1]] for i in range(len(h))],
            [sf[os_[i]:os_[i + 1]] for i in range(len(s))])


def load_gt(path, id_col):
    t = pq.read_table(path, columns=["query_id", "hit_ids", "hit_scores"])
    h = t.column("hit_ids").combine_chunks()
    s = t.column("hit_scores").combine_chunks()
    flat = pc.list_flatten(h).cast(pa.string())
    g = pc.fill_null(pc.index_in(flat, value_set=id_col), -1).to_numpy().astype(np.int64)
    oh, os_ = h.offsets.to_numpy(), s.offsets.to_numpy()
    sf = pc.list_flatten(s).to_numpy(zero_copy_only=False)
    return (t.column("query_id").to_pylist(),
            [g[oh[i]:oh[i + 1]] for i in range(len(h))],
            [sf[os_[i]:os_[i + 1]] for i in range(len(s))])


def load_gt_npz(path):
    """Pes2o-style GT: query_id (N,), ids (N, k) GLOBAL row ids (-1 = none),
    distances (N, k) COSINE DISTANCE -> scores as similarity 1 - d."""
    z = np.load(path)
    ids = z["ids"].astype(np.int64)
    sims = 1.0 - z["distances"].astype(np.float64)
    return z["query_id"].tolist(), [r for r in ids], [r for r in sims]


OUT_OF_RANGE = -2   # replaces a harness id outside [0, rows) other than the -1 pad: a list defect


def load_npz(path, n_ids=None):
    """The harness's saved top-k: (ids rows, score rows). With ``n_ids`` (the corpus rows), an id
    that is neither -1 (padding) nor in [0, n_ids) becomes OUT_OF_RANGE: list_defects counts it
    (a FAIL), and every later step treats it as no id (never an IndexError)."""
    z = np.load(path)
    I = z["I"].astype(np.int64)
    if n_ids is not None:
        I = np.where((I == -1) | ((I >= 0) & (I < int(n_ids))), I, OUT_OF_RANGE)
    return [row for row in I], [row for row in z["D"]]


def valid(ids, scores, kmax):
    """(ids, float64 scores) of one query's top-kmax, -1 entries dropped from both with one
    mask so each id keeps its own score even when a -1 sits mid-list."""
    A = np.asarray(ids[:kmax])
    S = np.asarray(scores[:kmax], dtype=np.float64)
    if len(S) != len(A):
        raise SystemExit(f"a list has {len(A)} ids but {len(S)} scores")
    m = A >= 0
    return A[m], S[m]


def list_defects(ids, kmax, mapped):
    """Per-system list checks (see the module docstring): duplicate ids,
    unmapped entries (-1 before a valid id; with ``mapped``, any -1) and
    out-of-range ids (load_npz's OUT_OF_RANGE)."""
    dq = dn = uq = un = oq = on = 0
    ex = []
    for q, row in enumerate(ids):
        A = np.asarray(row[:kmax])
        oor = np.nonzero(A == OUT_OF_RANGE)[0]
        if len(oor):
            oq += 1
            on += len(oor)
            if len(ex) < 5:
                ex.append({"query": q, "out_of_range_positions": [int(i) for i in oor[:5]]})
        good = A[A >= 0]
        u, c = np.unique(good, return_counts=True)
        if (c > 1).any():
            dq += 1
            dn += int((c - 1).sum())
            if len(ex) < 5:
                ex.append({"query": q, "duplicate_ids": [int(i) for i in u[c > 1][:5]]})
        neg = np.nonzero((A < 0) & (A != OUT_OF_RANGE))[0]
        if len(neg):
            last_valid = np.nonzero(A >= 0)[0]
            bad = neg if mapped else neg[neg < (last_valid[-1] if len(last_valid) else -1)]
            if len(bad):
                uq += 1
                un += len(bad)
                if len(ex) < 5:
                    ex.append({"query": q, "unmapped_positions": [int(i) for i in bad[:5]]})
    out = {"queries_with_duplicate_ids": dq, "duplicate_entries": dn,
           "queries_with_unmapped_entries": uq, "unmapped_entries": un,
           "failed": bool(dq or uq or oq), "examples": ex}
    if oq:   # keys added only when present, so the usual document shape is unchanged
        out.update(queries_with_out_of_range_ids=oq, out_of_range_entries=on)
    return out


def kth_and_strays(A, B):
    """(B's k-th id, ids A has that B does not) for valid-id arrays A, B (best first)."""
    return (int(B[-1]) if len(B) else None), [int(i) for i in set(A.tolist()) - set(B.tolist())]


def mismatched_ranks(A, B):
    """Ranks r < min(len(A), len(B)) where valid-id arrays A and B differ."""
    m = min(len(A), len(B))
    return np.nonzero(A[:m] != B[:m])[0]


def needed_ids(systems, n, kmax, order=False):
    """{query: set of global ids} the float64 adjudication must score: for every
    pair of systems, the ids one list has and the other does not, plus each
    list's k-th id; with ``order`` (the order check) also both ids at every
    rank where the two lists differ."""
    need = [set() for _ in range(n)]
    names = list(systems)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            ia, ib = systems[names[x]][0], systems[names[y]][0]
            for q in range(n):
                A, B = np.asarray(ia[q][:kmax]), np.asarray(ib[q][:kmax])
                A, B = A[A >= 0], B[B >= 0]
                if order:
                    r = mismatched_ranks(A, B)
                    if len(r):
                        need[q].update(A[r].tolist())
                        need[q].update(B[r].tolist())
                for P, Q in ((A, B), (B, A)):
                    kth, strays = kth_and_strays(P, Q)
                    if strays:
                        need[q].update(strays)
                        if kth is not None:
                            need[q].add(kth)
    return need


PAIR_BYTES = 200   # one (query, gid) -> float entry of the score table, Python objects incl.


def rescore_bytes(need, offsets, d, itemsize, threads):
    """Upper-bound estimate of exact_scores' peak memory, in bytes, before any
    vector is read: per file in flight its whole decoded vector column
    (read_table + combine_chunks: 2 copies at the stored width), its needed
    rows widened and normalised (2 float64 copies) and one scoring chunk (two
    gathered float64 operands of DENSE_PAIR_CELLS cells), the score table, and
    the float64 queries. Returns (bytes, n_gids, n_files)."""
    gids = np.asarray(sorted({g for s in need for g in s}), np.int64)
    n_pairs = sum(len(s) for s in need)
    if not len(gids):
        return 0, 0, 0
    files = np.searchsorted(offsets, gids, side="right") - 1
    fu, per_file = np.unique(files, return_counts=True)
    rows = np.diff(offsets)[fu]
    inflight = min(threads, len(fu))
    big_rows = np.sort(rows)[-inflight:].sum()           # the largest files at once
    big_need = np.sort(per_file)[-inflight:].sum()
    est = (big_rows * d * itemsize * 2                    # decoded columns in flight
           + big_need * d * 8 * 2                         # widened + normalised rows
           + inflight * 2 * min(DENSE_PAIR_CELLS, n_pairs * d) * 8   # scoring chunks
           + n_pairs * PAIR_BYTES                         # exact {(q, g): score}
           + len(need) * d * 8 * 2)                       # q64 (+ its normalised copy)
    return int(est), len(gids), len(fu)


def vector_itemsize(path, fs, vcol):
    """Bytes per stored vector element (list / fixed-size list of floats), 4 if unknown."""
    try:
        t = pq.read_schema(path, filesystem=fs).field(vcol).type
        return max(1, t.value_type.bit_width // 8)
    except Exception:
        return 4


DENSE_PAIR_CELLS = 1 << 22  # float64 cells of one gathered operand per scoring chunk (32 MiB)


def exact_scores(need, corpus, vcol, offsets, keys_paths, fs, xq, metric, threads=8):
    """{(query, gid): float64 score} from the STORED vectors (widened exactly to
    float64; cosine: both sides normalised in float64), reading only the files
    that hold a needed id. Each file is scored as it is read (no float64 copy
    of the needed vectors is kept), its (query, id) pairs in chunks of at most
    DENSE_PAIR_CELLS gathered float64 cells per operand."""
    by_file, by_g = _need_maps(need, offsets)
    q64 = np.asarray(xq, np.float64)
    if metric == "cosine":
        nq = np.linalg.norm(q64, axis=1, keepdims=True)
        q64 = np.divide(q64, nq, out=np.zeros_like(q64), where=nq > 0)
    chunk = max(1, DENSE_PAIR_CELLS // max(1, q64.shape[1]))

    def load(f):
        gs = by_file[f]
        t = pq.read_table(keys_paths[f], filesystem=fs, columns=[vcol])
        a = t.column(vcol).combine_chunks()
        del t
        sub = a.take(pa.array(np.asarray(gs, np.int64) - int(offsets[f])))
        del a
        v = pc.list_flatten(sub).to_numpy(zero_copy_only=False).astype(np.float64)
        v = v.reshape(len(gs), -1)
        if v.shape[1] != q64.shape[1]:
            raise SystemExit(f"{keys_paths[f]}: corpus dim {v.shape[1]}, queries {q64.shape[1]}")
        if metric == "cosine":
            nv = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.divide(v, nv, out=np.zeros_like(v), where=nv > 0)
        qi = np.fromiter((x for g in gs for x in by_g[g]), np.int64)
        ri = np.repeat(np.arange(len(gs), dtype=np.int64), [len(by_g[g]) for g in gs])
        s = np.empty(len(qi), np.float64)
        for lo in range(0, len(qi), chunk):
            hi = lo + chunk
            s[lo:hi] = np.einsum("ij,ij->i", q64[qi[lo:hi]], v[ri[lo:hi]])
        g_of = np.asarray(gs, np.int64)[ri]
        return list(zip(zip(qi.tolist(), g_of.tolist()), s.tolist()))
    out = {}
    with ThreadPoolExecutor(threads) as ex:
        for part in ex.map(load, sorted(by_file)):
            out.update(part)
    return out


# ------------------------------------------------- sparse / multivector rescoring

SPARSE_NZ_BYTES = 48       # per needed nonzero: packed (int32 + value), float64 CSR
#                            (int32 + float64) and its sum_duplicates / astype copies
SPARSE_PAIR_CHUNK = 1 << 14  # (query, id) pairs scored per scipy call
MV_Q_TOKENS = 4096         # query tokens gathered per MaxSim call (one document at a time)


def _need_files(need, offsets):
    """(sorted needed gids, their files' indices (unique), needed ids per file, pairs)."""
    gids = np.asarray(sorted({g for s in need for g in s}), np.int64)
    files = np.searchsorted(offsets, gids, side="right") - 1
    fu, per_file = np.unique(files, return_counts=True)
    return gids, fu, per_file, sum(len(s) for s in need)


def _need_maps(need, offsets):
    """({file: sorted needed gids}, {gid: sorted queries needing it})."""
    by_file, by_g = {}, {}
    gids = np.asarray(sorted({g for s in need for g in s}), np.int64)
    files = np.searchsorted(offsets, gids, side="right") - 1
    for g, f in zip(gids.tolist(), files.tolist()):
        by_file.setdefault(f, []).append(g)
    for q, s in enumerate(need):
        for g in s:
            by_g.setdefault(g, []).append(q)
    return by_file, by_g


def column_footer(path, fs, col):
    """``(rows, {leaf path: value count})`` of column ``col`` from one file's
    footer (no data read). Parquet counts a null / empty list's placeholder
    too, so each count bounds the stored nonzeros / token values from above."""
    with fs.open_input_file(path) as fh:
        md = pq.ParquetFile(fh).metadata
    leaves = {}
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            p = cc.path_in_schema
            if p == col or p.startswith(col + "."):
                leaves[p] = leaves.get(p, 0) + int(cc.num_values)
    return md.num_rows, leaves


def ragged_file_stats(paths, files, fs, col, threads=32):
    """{file: (rows, values)} for ``files`` (indices into ``paths``): values =
    the largest leaf count of ``col`` (sparse: nonzeros; multivector: token
    values, n_tok x dim), from the footers."""
    def one(f):
        rows, leaves = column_footer(paths[f], fs, col)
        return f, (rows, max(leaves.values()) if leaves else 0)
    with ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(one, [int(f) for f in files]))


def rescore_bytes_sparse(need, offsets, stats, index_bytes, value_bytes, q_nnz, n_q, threads):
    """Estimate of exact_scores_sparse's peak memory, in bytes, before any
    vector is read (``stats``: ragged_file_stats of the files holding a needed
    id). Per file in flight (the largest ``threads``): its whole decoded
    column at the stored width (2 copies: read_table + take / table_csr), its
    needed rows at the file's average nonzeros per row in float64
    (SPARSE_NZ_BYTES a nonzero), and one chunk of SPARSE_PAIR_CHUNK pairs'
    gathered query and document rows plus their product (12 B a nonzero, 3
    copies); beside them the float64 queries (+ their norms) and the score
    table. Returns (bytes, n_gids, n_files)."""
    gids, fu, per_file, n_pairs = _need_files(need, offsets)
    if not len(gids):
        return 0, 0, 0
    per, d_avg = [], 0.0
    for f, c in zip(fu.tolist(), per_file.tolist()):
        rows, nnz = stats[f]
        avg = nnz / max(1, rows)
        d_avg = max(d_avg, avg)
        per.append(nnz * (index_bytes + value_bytes) * 2 + c * avg * SPARSE_NZ_BYTES)
    inflight = min(threads, len(fu))
    q_avg = q_nnz / max(1, n_q)
    est = (sum(sorted(per)[-inflight:])
           + inflight * SPARSE_PAIR_CHUNK * (q_avg + d_avg) * 12 * 3
           + q_nnz * (4 + 4 + 12 * 2) + n_q * 8 * 2       # Queries + float64 CSR copies, norms
           + n_pairs * PAIR_BYTES)
    return int(est), len(gids), len(fu)


def rescore_bytes_mv(need, offsets, stats, value_bytes, q_tokens, dim, threads, tile_cells):
    """Estimate of exact_scores_mv's peak memory, in bytes, before any vector
    is read (``stats``: ragged_file_stats, values = token values). Per file in
    flight (the largest ``threads``): its whole decoded column at the stored
    width (2 copies), its needed documents at the file's average tokens per
    document (the packed subset at the stored width, then widened and, for
    cosine, unit-normalised: 2 float64 copies), and the MaxSim scratch: at
    most MV_Q_TOKENS gathered float64 query tokens and the float64 token
    score tile with its row maxima (2 x the larger of ``tile_cells`` and
    MV_Q_TOKENS x the average document's tokens); beside them the queries
    (float32 + float64 + unit copy) and the score table. Returns (bytes,
    n_gids, n_files)."""
    gids, fu, per_file, n_pairs = _need_files(need, offsets)
    if not len(gids):
        return 0, 0, 0
    per, tok_avg = [], 0.0
    for f, c in zip(fu.tolist(), per_file.tolist()):
        rows, vals = stats[f]
        avg = vals / max(1, rows)                            # token values per document
        tok_avg = max(tok_avg, avg / max(1, dim))
        per.append(vals * value_bytes * 2 + c * avg * (value_bytes + 8 * 2))
    inflight = min(threads, len(fu))
    est = (sum(sorted(per)[-inflight:])
           + inflight * (MV_Q_TOKENS * dim * 8
                         + 2 * 8 * max(tile_cells, MV_Q_TOKENS * tok_avg))
           + q_tokens * dim * (4 + 8 + 8)
           + n_pairs * PAIR_BYTES)
    return int(est), len(gids), len(fu)


def _take_rows(path, fs, col, rows):
    """Table of column ``col`` holding only ``rows`` (file-local, sorted) of one file."""
    return pq.read_table(path, filesystem=fs, columns=[col]).take(pa.array(rows))


def exact_scores_sparse(need, offsets, keys_paths, fs, vcol, q, metric, threads=8):
    """{(query, gid): float64 score} for a sparse corpus, as the harness's
    float64 --verify reference scores them (sparse.oneshot_reference /
    Rows.pair_dot): the needed rows of each file holding one, decoded by
    sparse.table_csr (values at the stored width, float64 narrowed to
    float32), widened to float64 and their duplicate columns summed, against
    ``q`` (sparse.load_queries' canonical queries) in float64; cosine divides
    by each side's float64 norm (sparse.row_norms) when it is nonzero. Each
    file is scored as it is read."""
    import scipy.sparse as sp
    sparse, _ = harness_modules()
    by_file, by_g = _need_maps(need, offsets)
    cos = metric == "cosine"
    qv = q.values.astype(np.float64)
    qn = sparse.row_norms(q.matrix(q.n_cols, np.float64)) if cos else None

    def load(f):
        gs = by_file[f]
        t = _take_rows(keys_paths[f], fs, vcol, np.asarray(gs, np.int64) - int(offsets[f]))
        c = sparse.unpack(sparse.table_csr(t, vcol))
        del t
        width = max(q.n_cols, c.n_cols, 1)
        X = sparse.block_matrix(c, width, np.float64)
        X.sum_duplicates()
        Q = sp.csr_matrix((qv, q.indices, q.indptr), shape=(q.n_rows, width))
        xn = sparse.row_norms(X) if cos else None
        qi = np.fromiter((x for g in gs for x in by_g[g]), np.int64)
        ri = np.repeat(np.arange(len(gs), dtype=np.int64), [len(by_g[g]) for g in gs])
        s = np.empty(len(qi), np.float64)
        for a in range(0, len(qi), SPARSE_PAIR_CHUNK):
            b = a + SPARSE_PAIR_CHUNK
            s[a:b] = np.asarray(Q[qi[a:b]].multiply(X[ri[a:b]]).sum(axis=1)).ravel()
        if cos:
            # each side divided by its own norm iff that norm is nonzero (Rows.pair_dot)
            for nv in (qn[qi], xn[ri]):
                np.divide(s, nv, out=s, where=nv > 0)
        g_of = np.asarray(gs, np.int64)[ri]
        return list(zip(zip(qi.tolist(), g_of.tolist()), s.tolist()))
    out = {}
    with ThreadPoolExecutor(threads) as ex:
        for part in ex.map(load, sorted(by_file)):
            out.update(part)
    return out


def exact_scores_mv(need, offsets, keys_paths, fs, vcol, q, metric, threads=8):
    """{(query, gid): float64 MaxSim} for a multi-vector corpus, as the
    harness's float64 --verify reference scores them (multivec.
    oneshot_reference): the needed documents of each file holding one,
    decoded by multivec.table_ragged, tokens widened to float64 (cosine:
    multivec.unit_tokens, a zero token stays zero), against ``q``
    (multivec.load_queries) by multivec.maxsim_matrix in float64, one
    document at a time against the queries that need it (at most MV_Q_TOKENS
    query tokens per call). A zero-token document or query scores -inf. Each
    file is scored as it is read."""
    _, multivec = harness_modules()
    by_file, by_g = _need_maps(need, offsets)
    cos = metric == "cosine"
    q64 = multivec.unit_tokens(q.tokens, np.float64) if cos else q.tokens.astype(np.float64)
    q_off, q_len, dim = q.offsets, q.lengths, q64.shape[1]

    def groups(qs):
        cur, tok = [], 0
        for x in qs:
            n_t = int(q_len[x])
            if cur and tok + n_t > MV_Q_TOKENS:
                yield cur
                cur, tok = [], 0
            cur.append(x)
            tok += n_t
        if cur:
            yield cur

    def load(f):
        gs = by_file[f]
        t = _take_rows(keys_paths[f], fs, vcol, np.asarray(gs, np.int64) - int(offsets[f]))
        r = multivec.unpack(multivec.table_ragged(t, vcol))
        del t
        if r.n_tok and q.n_tok and r.dim != dim:
            raise SystemExit(f"{keys_paths[f]}: corpus token width {r.dim}, queries {dim}")
        out = []
        for j, g in enumerate(gs):
            lo, hi = int(r.offsets[j]), int(r.offsets[j + 1])
            if hi == lo:  # a zero-token document: a non-candidate for every query
                out += [((x, g), multivec.NEG_INF) for x in by_g[g]]
                continue
            d = r.tokens[lo:hi].astype(np.float64)
            if cos:
                d = multivec.unit_tokens(d, np.float64)
            d_off = np.array([0, hi - lo], np.int64)
            for grp in groups(by_g[g]):
                off = np.zeros(len(grp) + 1, np.int64)
                np.cumsum(q_len[grp], out=off[1:])
                tok = np.concatenate([q64[q_off[x]:q_off[x + 1]] for x in grp]) if off[-1] \
                    else np.zeros((0, dim), np.float64)
                S = multivec.maxsim_matrix(off, tok, d_off, d, np.float64)
                out += [((x, g), float(s)) for x, s in zip(grp, S[:, 0])]
        return out
    out = {}
    with ThreadPoolExecutor(threads) as ex:
        for part in ex.map(load, sorted(by_file)):
            out.update(part)
    return out


def ragged_query_ids(uri):
    """query_id of every row of the query files, in sparse / multivec
    load_queries' order (sources.list_parquet), or None without the column."""
    harness_modules()
    from bench.data.sources import list_parquet
    fs, refs = list_parquet(uri)
    if "query_id" not in pq.read_schema(refs[0].path, filesystem=fs).names:
        return None
    return [str(x) for r in refs for x in
            pq.read_table(r.path, columns=["query_id"], filesystem=fs).column("query_id")
            .to_pylist()]


def ulps32(x):
    """One float32 ULP at |x| (float32 spacing), as a float."""
    return float(np.spacing(np.float32(abs(x))))


def order_check(q, A, B, exact, tie_ulps, st):
    """The float64 order check of one query (module docstring, "Order check"):
    at every rank r < min(len(A), len(B)) where A[r] != B[r], both ids'
    exact scores must lie within the tie window of each other; accumulates
    into ``st`` (see compare)."""
    r_ = mismatched_ranks(A, B)
    if not len(r_):
        return
    real_q = False
    for r in r_.tolist():
        a, b = int(A[r]), int(B[r])
        ea, eb = exact[(q, a)], exact[(q, b)]
        st["ranks"] += 1
        if np.isnan(ea) or np.isnan(eb):
            st["not_judged"] += 1
            continue
        if ea == eb:          # an exact tie (also two -inf non-candidates): any order is right
            st["ties"] += 1
            continue
        big = max(abs(ea), abs(eb))
        if np.isfinite(ea) and np.isfinite(eb):
            gap_ulps = abs(ea - eb) / ulps32(big)
            if gap_ulps <= tie_ulps:
                st["ties"] += 1
                st["max_tie_gap"] = max(st["max_tie_gap"], gap_ulps)
                continue
        else:
            gap_ulps = float("inf")   # a -inf non-candidate against a candidate: never a tie
        st["real"] += 1
        real_q = True
        st["max_real_gap"] = max(st["max_real_gap"], gap_ulps)
        if len(st["examples"]) < 5:
            st["examples"].append({"query": q, "rank": r, "ids": [a, b], "exact": [ea, eb],
                                   "gap_ulps": gap_ulps})
    st["queries_real"] += real_q
    # diagnostic: an inversion beyond the window inside one list (a later mismatched rank
    # beating an earlier one), the list's own fault whatever the other list holds
    for nm, L in st["sides"]:
        lo = None
        for r in r_.tolist():
            e = exact[(q, int(L[r]))]
            if np.isnan(e):
                continue
            if lo is not None and e > lo and (
                    not np.isfinite(lo) or e - lo > tie_ulps * ulps32(max(abs(e), abs(lo)))):
                st["inversions"][nm] += 1
            lo = e if lo is None else min(lo, e)


# judge / compare document keys added on top of compare.py's original --json format;
# legacy_view drops them
D_S8_40_TOP = ("filter_masked_pairs", "score_tolerance", "shared_score_failed_pairs",
               "f64_max_error_abs")
D_S8_40_PAIR = ("shared_score_mismatches", "shared_score_examples", "shared_score_tolerance",
                "f64_max_error_abs")


def legacy_view(doc):
    """judge()'s --json document without those keys: the original --json format."""
    if not isinstance(doc, dict):
        return doc
    out = {k: v for k, v in doc.items() if k not in D_S8_40_TOP}
    out["pairs"] = {pk: {k: v for k, v in r.items() if k not in D_S8_40_PAIR}
                    for pk, r in (doc.get("pairs") or {}).items()}
    return out


def shared_score_bad(x, y, atol, rtol) -> bool:
    """Two scores of one id: off by more than 2 x (atol + rtol x |larger|)? Equal = never."""
    if x == y:
        return False
    if not (np.isfinite(x) and np.isfinite(y)):   # unequal and one is +-inf / NaN
        return True
    return not abs(x - y) <= 2 * (atol + rtol * max(abs(x), abs(y)))


def compare(a, b, n, tol, kmax, exact=None, names=("a", "b"), tie_ulps=128, modality="dense",
            order=False, score_tol=None):
    """Compare two systems a, b = (ids list, scores list) over the same n queries.

    ``exact`` {(query, gid): float64 score}: also adjudicate every id only one list has (the
    --rescore and short-list rules of the module docstring) and measure each side's own error
    in float32 ULPs over its adjudicated ids. ``order`` (``exact`` must cover
    needed_ids(..., order=True)): run order_check. ``score_tol`` (atol, rtol): check every
    shared id with shared_score_bad."""
    (ia, sa), (ib, sb) = a, b
    res = {f"k{k}": {"set_equal": 0, "overlap": 0.0} for k in KS if k <= kmax}
    at_k = {"set_equal": 0, "overlap": 0.0}
    ost = {"ranks": 0, "ties": 0, "real": 0, "not_judged": 0, "queries_real": 0,
           "max_tie_gap": 0.0, "max_real_gap": 0.0, "examples": [],
           "inversions": {names[0]: 0, names[1]: 0}}
    first_diff, dmax, nonb, examples, short = [], 0.0, 0, [], 0
    f64_real, f64_ties, f64_blame, f64_examples = 0, 0, {names[0]: 0, names[1]: 0}, []
    f64_short = 0        # strays of a short Q left unjudged (exact score NaN)
    f64_short_judged = 0  # strays of a short Q judged against the modality's floor
    f64_err = {names[0]: 0.0, names[1]: 0.0}   # max |own - exact| in ULPs, per side
    f64_abs = {names[0]: 0.0, names[1]: 0.0}   # max |own - exact|, per side
    sh_bad, sh_ex = 0, []                       # shared ids whose two scores disagree
    dups = {names[0]: 0, names[1]: 0}      # queries whose list repeats an id
    dropped = {names[0]: 0, names[1]: 0}   # queries whose list had a -1 masked out
    for q in range(n):
        (A, SA), (B, SB) = valid(ia[q], sa[q], kmax), valid(ib[q], sb[q], kmax)
        for nm, X, raw in ((names[0], A, ia[q]), (names[1], B, ib[q])):
            dups[nm] += len(np.unique(X)) != len(X)
            dropped[nm] += len(X) != len(np.asarray(raw[:kmax]))
        if len(A) < kmax or len(B) < kmax:
            short += 1
        for k in KS:
            if k > kmax:
                continue
            x, y = set(A[:k].tolist()), set(B[:k].tolist())
            res[f"k{k}"]["set_equal"] += x == y
            res[f"k{k}"]["overlap"] += len(x & y) / max(1, k)
        x, y = set(A.tolist()), set(B.tolist())
        at_k["set_equal"] += x == y
        at_k["overlap"] += len(x & y) / max(1, kmax)
        neq = mismatched_ranks(A, B)
        if len(neq):
            first_diff.append(int(neq[0]) + 1)
        sa_q = dict(zip(A.tolist(), SA))
        sb_q = dict(zip(B.tolist(), SB))
        if exact is not None and order and len(neq):
            ost["sides"] = ((names[0], A), (names[1], B))
            order_check(q, A, B, exact, tie_ulps, ost)
            for nm, X, own in ((names[0], A, sa_q), (names[1], B, sb_q)):
                for g in X[neq].tolist():
                    e = exact[(q, g)]
                    if np.isfinite(e):
                        f64_err[nm] = max(f64_err[nm], abs(own[g] - e) / ulps32(e))
                        f64_abs[nm] = max(f64_abs[nm], abs(own[g] - e))
        common = sa_q.keys() & sb_q.keys()
        if common:
            dmax = max(dmax, max(abs(sa_q[i] - sb_q[i]) for i in common))
            if score_tol is not None:
                for i in sorted(common):
                    if shared_score_bad(sa_q[i], sb_q[i], *score_tol):
                        sh_bad += 1
                        if len(sh_ex) < 5:
                            sh_ex.append({"query": q, "id": int(i), names[0]: sa_q[i],
                                          names[1]: sb_q[i]})
        # boundary rule: an id only one side has must sit at the other side's cutoff
        for mine, other in ((sa_q, sb_q), (sb_q, sa_q)):
            if not other:
                continue
            kth = min(other.values())
            for i in mine.keys() - other.keys():
                if abs(mine[i] - kth) > tol:
                    nonb += 1
                    if len(examples) < 5:
                        examples.append({"query": q, "id": int(i), "score": mine[i],
                                         "other_kth": kth})
        if exact is not None:
            # float64 adjudication: id i is in P, not Q; s_k = exact score of Q's k-th id.
            # Within the tie window either choice is right; above it Q dropped a better id
            # (Q is wrong); below it P kept a worse id (P is wrong).
            own = {names[0]: sa_q, names[1]: sb_q}
            for (P, Q, pname, qname) in ((A, B, names[0], names[1]), (B, A, names[1], names[0])):
                kth_id, strays = kth_and_strays(P, Q)
                if not strays:
                    continue
                # each side's own float32 error on what is being judged (finite scores:
                # a -inf non-candidate has no rounding error to measure)
                for nm, g in [(pname, i) for i in strays] + \
                        ([(qname, kth_id)] if kth_id is not None else []):
                    e = exact[(q, g)]
                    if np.isfinite(e):
                        f64_err[nm] = max(f64_err[nm], abs(own[nm][g] - e) / ulps32(e))
                        f64_abs[nm] = max(f64_abs[nm], abs(own[nm][g] - e))
                # a dense Q holding a non-candidate (exact -inf: a filter-excluded row, set by
                # judge's exact_hook) has fewer than k real hits: judged as a short list, so a
                # finite stray of P is Q's fault (it filled the slot with an excluded row)
                q_noncand = modality == "dense" and any(
                    exact.get((q, int(g)), 0.0) == -np.inf for g in Q.tolist())
                if len(Q) < kmax or q_noncand:
                    # short Q: its k-th score is the modality's floor; every stray is judged
                    for i in strays:
                        s_i = exact[(q, i)]
                        if np.isnan(s_i):
                            f64_short += 1
                            continue
                        f64_short_judged += 1
                        if modality == "sparse":
                            # floor 0: within the window of 0 is a tie (zero-score filler,
                            # or a document sharing no term); a nonzero score needs a
                            # shared term, so Q, which had room, dropped a scoring document
                            if abs(s_i) <= tie_ulps * ulps32(abs(s_i)):
                                f64_ties += 1
                                continue
                            at_fault, floor = qname, 0.0
                        else:
                            # dense / multivector: every finite-scoring row is a candidate
                            # a short Q should hold; a -inf one (zero-token) is a
                            # non-candidate P should never have returned
                            at_fault, floor = (qname if np.isfinite(s_i) else pname), None
                        f64_real += 1
                        f64_blame[at_fault] += 1
                        if len(f64_examples) < 5:
                            f64_examples.append({"query": q, "id": i, "exact": s_i,
                                                 "other_kth_exact": floor, "short_list": True,
                                                 "wrong": at_fault})
                    continue
                s_k = exact[(q, kth_id)]
                for i in strays:
                    s_i = exact[(q, i)]
                    if np.isfinite(s_i) and np.isfinite(s_k):
                        zone = tie_ulps * ulps32(max(abs(s_i), abs(s_k)))
                        if abs(s_i - s_k) <= zone:
                            f64_ties += 1
                            continue
                    # (a -inf score, multivector only: no tie window; the -inf side is
                    # the worse one)
                    f64_real += 1
                    at_fault = qname if s_i > s_k else pname
                    f64_blame[at_fault] += 1
                    if len(f64_examples) < 5:
                        f64_examples.append({"query": q, "id": i, "exact": s_i,
                                             "other_kth_exact": s_k, "wrong": at_fault})
    for k in list(res):
        res[k]["set_equal"] /= n
        res[k]["overlap"] /= n
    res["at_k"] = {"k": kmax, "set_equal": at_k["set_equal"] / max(1, n),
                   "overlap": at_k["overlap"] / max(1, n)}
    if exact is not None and order:
        res["f64_order_mismatched_ranks"] = ost["ranks"]
        res["f64_order_tie_swaps"] = ost["ties"]
        res["f64_order_real_disagreements"] = ost["real"]
        res["f64_order_queries_with_real"] = ost["queries_real"]
        res["f64_order_not_judged"] = ost["not_judged"]
        res["f64_order_max_tie_gap_ulps"] = ost["max_tie_gap"]
        res["f64_order_max_real_gap_ulps"] = ost["max_real_gap"]
        res["f64_order_inversions_by_system"] = ost["inversions"]
        res["f64_order_examples"] = ost["examples"]
    if exact is not None:
        res["f64_tie_zone_differences"] = f64_ties
        res["f64_real_disagreements"] = f64_real
        res["f64_wrong_by_system"] = f64_blame
        res["f64_examples"] = f64_examples
        res["f64_not_judged_short_list"] = f64_short
        res["f64_judged_short_list"] = f64_short_judged
        res["f64_max_error_ulps"] = f64_err
        res["f64_max_error_abs"] = f64_abs
    if score_tol is not None:
        res["shared_score_mismatches"] = sh_bad
        res["shared_score_examples"] = sh_ex
        res["shared_score_tolerance"] = (f"2 x ({score_tol[0]:g} + {score_tol[1]:g} x |s|)")
    res["queries_differing_order"] = len(first_diff)
    res["first_diff_rank_min"] = min(first_diff) if first_diff else None
    res["first_diff_rank_median"] = statistics.median(first_diff) if first_diff else None
    res["max_abs_score_delta_shared"] = dmax
    res["non_boundary_disagreements"] = nonb
    res["non_boundary_examples"] = examples
    res["queries_short_of_k"] = short
    res["queries_with_duplicate_ids"] = dups
    res["queries_with_masked_minus1"] = dropped
    return res


def corpus_paths(corpus, n_keys, max_files=None):
    """(fs, the corpus's *.parquet paths, sorted as corpus_index lists them; the first
    ``max_files`` of them for a subset run)."""
    fs, root = pafs.FileSystem.from_uri(corpus)
    infos = sorted((i for i in fs.get_file_info(pafs.FileSelector(root.rstrip("/"),
                                                                    recursive=True))
                    if i.type == pafs.FileType.File and i.path.endswith(".parquet")),
                   key=lambda i: i.path)
    paths = [i.path for i in infos]
    if max_files:
        paths = paths[:int(max_files)]
    if len(paths) != n_keys:
        raise SystemExit(f"--rescore: {len(paths)} corpus files now, {n_keys} indexed")
    return fs, paths


def rescore_ragged(a, systems, n, qn, offsets, n_keys):
    """--rescore for --modality sparse / multivector: the harness's queries
    (sparse: canonicalised) and the float64 scores of every needed (query,
    id) pair, after the memory estimate (refused above --rescore-max-gib)."""
    sparse, multivec = harness_modules()
    mod = sparse if a.modality == "sparse" else multivec
    qid = ragged_query_ids(a.queries)
    if qid is not None and qid != [str(x) for x in qn]:
        raise SystemExit("--queries order != nova-bf query order")
    q, warn = mod.load_queries(a.queries, a.qcol)
    for w in warn:
        print(w, flush=True)
    if q.n_rows != n:
        raise SystemExit(f"--queries has {q.n_rows} rows, nova-bf {n}")
    fs, paths = corpus_paths(a.corpus, n_keys, getattr(a, "max_files", None))
    need = needed_ids(systems, n, a.k, order=a.order_check)
    _, fu, _, _ = _need_files(need, offsets)
    if len(fu):
        typ = pq.read_schema(paths[int(fu[0])], filesystem=fs).field(a.vcol).type
    stats = ragged_file_stats(paths, fu, fs, a.vcol)
    if a.modality == "sparse":
        if len(fu) and sparse.sparse_fields(typ) is None:
            raise SystemExit(f"--modality sparse: --vcol {a.vcol!r} is {typ}, not a "
                             "struct<indices: list<int>, values: list<float>>")
        ib, vb = ((t.value_type.bit_width // 8 for t in sparse.sparse_fields(typ))
                  if len(fu) else (4, 4))
        est, n_g, n_f = rescore_bytes_sparse(need, offsets, stats, ib, vb, q.nnz, q.n_rows,
                                             a.rescore_threads)
        what = f"{n_g} sparse rows' nonzeros (from the footers)"
    else:
        if len(fu) and multivec.mv_value_type(typ) is None:
            raise SystemExit(f"--modality multivector: --vcol {a.vcol!r} is {typ}, not a "
                             "list<list<float>>")
        vb = multivec.mv_value_type(typ).bit_width // 8 if len(fu) else 4
        est, n_g, n_f = rescore_bytes_mv(need, offsets, stats, vb, q.n_tok, q.dim,
                                         a.rescore_threads, multivec.CPU_TILE_CELLS)
        what = f"{n_g} documents' tokens of {q.dim} dims (from the footers)"
    print(f"float64 rescoring ({a.modality}): {n_g} ids in {n_f} files, "
          f"{sum(len(x) for x in need)} (query, id) pairs, estimated peak "
          f"{est / 2**30:.2f} GiB (limit --rescore-max-gib {a.rescore_max_gib:g})", flush=True)
    if est > a.rescore_max_gib * 2**30:
        raise SystemExit(
            f"--rescore refused: estimated {est / 2**30:.2f} GiB > --rescore-max-gib "
            f"{a.rescore_max_gib:g} ({what} + {min(a.rescore_threads, n_f)} files' columns "
            "in flight + the score table). Raise --rescore-max-gib if the machine has the "
            "memory, lower --rescore-threads to shrink the read buffers, or compare fewer "
            "systems")
    fn = exact_scores_sparse if a.modality == "sparse" else exact_scores_mv
    return fn(need, offsets, paths, fs, a.vcol, q, a.metric, threads=a.rescore_threads)


def parser():
    ap = argparse.ArgumentParser(
        description="Per-id agreement of nova-bf, harness engines and an optional dataset GT, "
                    "with --rescore's float64 adjudication (see the module docstring).")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--gt", default=None,
                    help="the dataset's ground truth (.parquet with COYO-style ids in the "
                         "corpus `id` column, or a Pes2o-style .npz of global row ids); "
                         "optional; a reference, not a system under test: its float64 "
                         "disagreements are printed, never a FAIL")
    ap.add_argument("--novabf", required=True)
    ap.add_argument("--modality", choices=MODALITIES, default="dense",
                    help="stored vector layout and float64 score for --rescore: dense "
                         "(list<float>), sparse (struct<indices, values>: ip / cosine, short "
                         "nova-bf lists judged against the floor 0), multivector "
                         "(list<list<float>>: MaxSim, zero-token = -inf); sparse / "
                         "multivector reuse the harness's decode and float64 helpers "
                         "(bench.data.formats)")
    ap.add_argument("--novabf-ids", choices=["uuid", "corpus"], default="uuid",
                    help="what nova-bf's hit_ids are: uuid (md5 of key:row, a config without "
                         "a corpus id_column; the default) or corpus (the corpus `id` column's "
                         "values, a config with id_column: id; must be unique)")
    ap.add_argument("--query-rows", default=None, metavar="COLUMN=V1,V2",
                    help="--rescore with a `rows`-selected search: take from --queries only the "
                         "rows whose COLUMN is one of the values (file order), as nova-bf's "
                         "`rows: {column, isin}` selects them")
    ap.add_argument("--harness", action="append", default=[], help="NAME=path.npz")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--max-files", type=int, default=None,
                    help="the run read only the first N corpus files (sorted by path; a subset "
                         "run's --max-files): index, map and rescore those files only")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="cosine-score tolerance for the cutoff-tie rule on each system's "
                         "OWN scores (the legacy non_boundary count; see --rescore)")
    ap.add_argument("--rescore", action="store_true",
                    help="adjudicate in float64: every id one list has and the other does not, "
                         "and each list's k-th id, rescored from the stored vectors (needs "
                         "--vcol, --queries, --qcol); reads only the files holding such ids")
    ap.add_argument("--no-order-check", dest="order_check", action="store_false",
                    help="with --rescore, skip the float64 order check (on by default: at "
                         "every rank where two lists differ, both ids' exact scores must be "
                         "within the --tie-ulps window, else a FAIL; it rescores those ids "
                         "too, so it costs more memory -- see --rescore-max-gib)")
    ap.add_argument("--vcol", default=None, help="corpus vector column (--rescore)")
    ap.add_argument("--queries", default=None,
                    help="query parquet (sparse / multivector: a file or a directory of "
                         "*.parquet, as the harness lists it), nova-bf's order (--rescore)")
    ap.add_argument("--qcol", default=None, help="query vector column (--rescore)")
    ap.add_argument("--rescore-max-gib", type=float, default=32.0,
                    help="refuse --rescore when its estimated peak memory (float64 copies of "
                         "the needed vectors / nonzeros / tokens + read buffers + score "
                         "table) exceeds this")
    ap.add_argument("--rescore-threads", type=int, default=8,
                    help="files read at once by --rescore (each holds its whole decoded "
                         "vector column)")
    ap.add_argument("--metric", choices=["cosine", "ip"], default="cosine",
                    help="ip is nova-bf's `dot`; multivector cosine = per-token cosine MaxSim")
    ap.add_argument("--tie-ulps", type=float, default=128,
                    help="--rescore tie window: two ids whose exact scores differ by at most "
                         "this many float32 ULPs (of the larger) may be swapped by any engine "
                         "(default 128 = 2 x the worst measured engine error, 54, rounded up)")
    ap.add_argument("--score-atol", type=float, default=SHARED_SCORE_ATOL,
                    help="shared-id score check (every query, every pair): an id both lists "
                         "report fails when its two scores differ by more than "
                         "2 x (--score-atol + --score-rtol x |s|)")
    ap.add_argument("--score-rtol", type=float, default=SHARED_SCORE_RTOL)
    ap.add_argument("--json", default=None)
    return ap


def parse_args(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    a.order_check = a.order_check and a.rescore
    if a.rescore and not (a.vcol and a.queries and a.qcol):
        ap.error("--rescore needs --vcol, --queries and --qcol")
    return a


def load_inputs(a, t0):
    """main()'s input half: the corpus index, nova-bf's result (ids mapped to global rows), the
    GT and the harness results, as ``inputs`` for judge()."""
    gt_npz = bool(a.gt) and a.gt.endswith(".npz")
    # the corpus `id` column is read only to map a parquet GT's ids, or nova-bf's
    # (--novabf-ids corpus)
    corpus_ids = a.novabf_ids == "corpus"
    keys, rows, offsets, id_col = corpus_index(
        a.corpus, need_ids=(bool(a.gt) and not gt_npz) or corpus_ids, max_files=getattr(a, "max_files", None))
    n_ids = int(offsets[-1])
    dup = 0 if id_col is None else n_ids - len(pc.unique(id_col))
    print(f"corpus: {len(keys)} files, {n_ids} rows, duplicate ids {dup} "
          f"({time.time() - t0:.0f} s)", flush=True)
    if corpus_ids:
        if dup:
            raise SystemExit(f"--novabf-ids corpus: the corpus `id` column has {dup} duplicate "
                             "ids, so a nova-bf hit id does not name one row")
        qn, ids, sc = load_gt(a.novabf, id_col)   # same layout: query_id, hit_ids, hit_scores
    else:
        u2g = uuid_table(keys, rows, offsets)
        print(f"uuid map built ({time.time() - t0:.0f} s)", flush=True)
        qn, ids, sc = load_novabf(a.novabf, u2g)

    systems = {}
    unmapped = sum(int((x < 0).sum()) for x in ids)
    systems["novabf"] = (ids, sc)
    gt_missing = None
    if a.gt:
        qg, gids, gsc = load_gt_npz(a.gt) if gt_npz else load_gt(a.gt, id_col)
        gt_missing = sum(int((x < 0).sum()) for x in gids)
        # nova-bf writes query_id as a string; a GT may hold ints (Pes2o): compare as text
        if [str(x) for x in qn] != [str(x) for x in qg]:
            raise SystemExit("nova-bf query order != GT query order")
        systems["gt"] = (gids, gsc)
    for spec in a.harness:
        name, path = spec.split("=", 1)
        systems[name] = load_npz(path, n_ids)
        if len(systems[name][0]) != len(qn):
            raise SystemExit(f"harness {name} ({path}) has {len(systems[name][0])} result rows, "
                             f"nova-bf {len(qn)}: not the same queries")
    n = len(qn)
    return {"keys": keys, "offsets": offsets, "n_ids": n_ids, "dup": dup, "qn": qn,
            "systems": systems, "n": n, "unmapped": unmapped, "gt_missing": gt_missing,
            "mapped": {"novabf": True, "gt": not gt_npz}}


def judge(a, inputs, t0, exact_hook=None):
    """main()'s verdict half: list defects, float64 rescoring, every pair, and the FAILs.
    Returns ``(out, fails)``: the --json document and the FAIL reasons (empty = pass).
    ``inputs``: load_inputs()'s dict; ``mapped`` names the systems whose ids were mapped
    (nova-bf, a parquet GT), where any -1 is a list defect. ``exact_hook(exact) -> int``
    (filtered searches): runs on the float64 score table before judging; it sets each
    filter-excluded (query, row) to -inf, so the system that returned it is blamed, and
    returns how many."""
    keys, offsets, n_ids, dup = inputs["keys"], inputs["offsets"], inputs["n_ids"], inputs["dup"]
    qn, systems, n = inputs["qn"], inputs["systems"], inputs["n"]
    unmapped, gt_missing, mapped = inputs["unmapped"], inputs["gt_missing"], inputs["mapped"]
    # per-system list defects: a FAIL each (exit 1 at the end), printed before the pairs
    defects = {nm: list_defects(sy[0], a.k, mapped.get(nm, False)) for nm, sy in systems.items()}
    for nm, df in defects.items():
        if df["failed"]:
            print(f"FAIL {nm}: {df['queries_with_duplicate_ids']} queries with duplicate ids "
                  f"({df['duplicate_entries']} extra entries), "
                  f"{df['queries_with_unmapped_entries']} queries with unmapped (-1) entries "
                  f"({df['unmapped_entries']})"
                  + (f", {df['queries_with_out_of_range_ids']} queries with out-of-range ids "
                     f"({df['out_of_range_entries']})" if "out_of_range_entries" in df else "")
                  + f"; examples {df['examples'][:3]}", flush=True)
    exact = None
    if a.rescore and a.modality != "dense":
        exact = rescore_ragged(a, systems, n, qn, offsets, len(keys))
        print(f"float64 rescoring: {len(exact)} (query, id) pairs ({time.time() - t0:.0f} s)",
              flush=True)
    elif a.rescore:
        qt = pq.read_table(a.queries)
        if a.query_rows:
            col, _, vals = a.query_rows.partition("=")
            if not col or not vals:
                raise SystemExit("--query-rows wants COLUMN=V1,V2,...")
            if col not in qt.column_names:
                raise SystemExit(f"--query-rows: --queries has no column {col!r}")
            want = set(vals.split(","))
            idx = [i for i, v in enumerate(qt.column(col).to_pylist()) if str(v) in want]
            qt = qt.take(pa.array(idx, pa.int64()))
            print(f"--query-rows {a.query_rows}: {len(idx)} of the queries file's rows", flush=True)
        if "query_id" in qt.column_names and \
                [str(x) for x in qt.column("query_id").to_pylist()] != [str(x) for x in qn]:
            raise SystemExit("--queries order != nova-bf query order")
        qa = qt.column(a.qcol).combine_chunks()
        xq = pc.list_flatten(qa).to_numpy(zero_copy_only=False).reshape(len(qa), -1)
        if len(xq) != n:
            raise SystemExit(f"--queries has {len(xq)} rows, nova-bf {n}")
        fs, paths = corpus_paths(a.corpus, len(keys), getattr(a, "max_files", None))
        need = needed_ids(systems, n, a.k, order=a.order_check)
        est, n_g, n_f = rescore_bytes(need, offsets, xq.shape[1],
                                      vector_itemsize(paths[0], fs, a.vcol), a.rescore_threads)
        print(f"float64 rescoring: {n_g} ids in {n_f} files, {sum(len(x) for x in need)} "
              f"(query, id) pairs, estimated peak {est / 2**30:.2f} GiB "
              f"(limit --rescore-max-gib {a.rescore_max_gib:g})", flush=True)
        if est > a.rescore_max_gib * 2**30:
            raise SystemExit(
                f"--rescore refused: estimated {est / 2**30:.2f} GiB > --rescore-max-gib "
                f"{a.rescore_max_gib:g} ({n_g} float64 vectors of {xq.shape[1]} dims + "
                f"{min(a.rescore_threads, n_f)} files' columns in flight + the score table). "
                "Raise --rescore-max-gib if the machine has the memory, lower "
                "--rescore-threads to shrink the read buffers, or compare fewer systems")
        exact = exact_scores(need, a.corpus, a.vcol, offsets, paths, fs,
                             xq, a.metric, threads=a.rescore_threads)
        print(f"float64 rescoring: {len(exact)} (query, id) pairs ({time.time() - t0:.0f} s)",
              flush=True)
    masked = None
    if exact is not None and exact_hook is not None:
        masked = exact_hook(exact)
        print(f"filter: {masked} filter-excluded (query, id) pairs scored -inf", flush=True)
    out = {"queries": n, "k": a.k, "corpus_rows": n_ids, "corpus_duplicate_ids": dup,
           "novabf_unmapped_hits": unmapped, "gt_ids_not_in_corpus": gt_missing,
           "tol": a.tol, "rescore": a.rescore, "order_check": a.order_check,
           "tie_ulps": a.tie_ulps,
           "modality": a.modality, "system_defects": defects, "filter_masked_pairs": masked,
           "score_tolerance": [a.score_atol, a.score_rtol],
           "ok": not any(df["failed"] for df in defects.values()), "pairs": {}}
    names = list(systems)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            out["pairs"][f"{x}_vs_{y}"] = r = compare(
                systems[x], systems[y], n, a.tol, a.k, exact=exact, names=(x, y),
                tie_ulps=a.tie_ulps, modality=a.modality, order=a.order_check,
                score_tol=(a.score_atol, a.score_rtol))
            print(f"{x:>6} vs {y:<6} " + "  ".join(
                f"@{k}: eq {r[f'k{k}']['set_equal']:.4f} ov {r[f'k{k}']['overlap']:.6f}"
                for k in KS if k <= a.k)
                + f"  first diff rank min {r['first_diff_rank_min']}"
                + f"  max|d| {r['max_abs_score_delta_shared']:.2e}"
                + f"  NON-BOUNDARY {r['non_boundary_disagreements']}"
                + (f"  F64 REAL {r['f64_real_disagreements']} {r['f64_wrong_by_system']}"
                   f" (tie zone {r['f64_tie_zone_differences']}, not judged short "
                   f"{r['f64_not_judged_short_list']}"
                   + (f", judged short {r['f64_judged_short_list']}"
                      if a.modality != "dense" or r["f64_judged_short_list"] else "")
                   + ")" if exact is not None else "")
                + (f"  F64 ORDER REAL {r['f64_order_real_disagreements']} (tie swaps "
                   f"{r['f64_order_tie_swaps']}, max tie gap "
                   f"{r['f64_order_max_tie_gap_ulps']:.1f} ULPs, not judged "
                   f"{r['f64_order_not_judged']})" if a.order_check else "")
                + "".join(f"  FAIL {nm}: DUP-IDS {r['queries_with_duplicate_ids'][nm]} "
                          f"UNMAPPED {defects[nm]['queries_with_unmapped_entries']}"
                          for nm in (x, y) if defects[nm]["failed"]),
                flush=True)
    wrong = {}
    if exact is not None:
        # each system's own float32 error over everything adjudicated, all pairs; and
        # every real disagreement, by the system it is blamed on
        err, err_abs = {}, {}
        for r in out["pairs"].values():
            for nm, e in r["f64_max_error_ulps"].items():
                err[nm] = max(err.get(nm, 0.0), e)
            for nm, e in r["f64_max_error_abs"].items():
                err_abs[nm] = max(err_abs.get(nm, 0.0), e)
            for nm, c in r["f64_wrong_by_system"].items():
                wrong[nm] = wrong.get(nm, 0) + c
        out["f64_max_error_ulps"], out["f64_wrong_total"] = err, wrong
        out["f64_max_error_abs"] = err_abs
        print("float64: own error on the adjudicated ids (float32 ULPs of the exact score): "
              + ", ".join(f"{nm} {e:.1f}" for nm, e in err.items())
              + f"; tie window {a.tie_ulps:g} ULPs", flush=True)
        for nm, e in err.items():
            if 2 * e > a.tie_ulps:
                print(f"WARNING: {nm}'s own error {e:.1f} ULPs x 2 > --tie-ulps {a.tie_ulps:g}: "
                      "the window is too tight for its honest rounding, so real "
                      "disagreements may be rounding (raise --tie-ulps and report it)",
                      flush=True)
        # the dataset's GT is a reference, not a system under test: its own
        # disagreements are printed, not a FAIL
        gt_wrong = wrong.pop("gt", 0)
        if gt_wrong:
            print(f"note: the dataset GT has {gt_wrong} float64-checked disagreements outside "
                  "the tie window (not a system under test)", flush=True)
        out["f64_wrong_gt"] = gt_wrong
        for nm, c in wrong.items():
            if c:
                print(f"FAIL {nm}: {c} float64-checked disagreements outside the "
                      f"{a.tie_ulps:g}-ULP tie window (examples in the JSON)", flush=True)
    order_bad = {}
    if a.order_check:
        # every pair's real order disagreements are a FAIL of that pair, except a pair with
        # the dataset GT (a reference, not a system under test: printed as a note)
        for pk, r in out["pairs"].items():
            c = r["f64_order_real_disagreements"]
            if not c:
                continue
            ex = r["f64_order_examples"][:2]
            if "gt" in pk.split("_vs_"):
                print(f"note: {pk}: {c} float64-checked ORDER disagreements outside the tie "
                      f"window (the dataset GT is not a system under test); e.g. {ex}", flush=True)
                continue
            order_bad[pk] = c
            print(f"FAIL {pk}: {c} float64-checked ORDER disagreements at mismatched ranks "
                  f"outside the {a.tie_ulps:g}-ULP tie window, in "
                  f"{r['f64_order_queries_with_real']} queries (max gap "
                  f"{r['f64_order_max_real_gap_ulps']:.1f} ULPs; own-list inversions "
                  f"{r['f64_order_inversions_by_system']}); e.g. {ex}", flush=True)
        out["f64_order_failed_pairs"] = order_bad
    score_bad = {}
    for pk, r in out["pairs"].items():
        c = r["shared_score_mismatches"]
        if not c:
            continue
        if "gt" in pk.split("_vs_"):
            print(f"note: {pk}: {c} shared ids whose scores differ by more than "
                  f"{r['shared_score_tolerance']} (the dataset GT is not a system under test); "
                  f"e.g. {r['shared_score_examples'][:2]}", flush=True)
            continue
        score_bad[pk] = c
        print(f"FAIL {pk}: {c} shared ids whose two scores differ by more than "
              f"{r['shared_score_tolerance']}; e.g. {r['shared_score_examples'][:2]}", flush=True)
    out["shared_score_failed_pairs"] = score_bad
    print(f"unmapped nova-bf hits {unmapped}; "
          + (f"GT ids not in corpus {gt_missing}; " if a.gt else "no GT; ")
          + f"{time.time() - t0:.0f} s", flush=True)
    out["ok"] = out["ok"] and not any(wrong.values()) and not order_bad and not score_bad
    bad = [nm for nm, df in defects.items() if df["failed"]]
    fails = ([f"list defects in {', '.join(bad)} (duplicate or unmapped ids)"] if bad else []) + \
        ([f"float64 disagreements: {', '.join(f'{n} {c}' for n, c in wrong.items() if c)}"]
         if any(wrong.values()) else []) + \
        ([f"float64 order disagreements: {', '.join(f'{p} {c}' for p, c in order_bad.items())}"]
         if order_bad else []) + \
        ([f"shared-id score mismatches (> 2 x ({a.score_atol:g} + {a.score_rtol:g} x |s|), every "
          f"query): {', '.join(f'{p} {c}' for p, c in score_bad.items())}"] if score_bad else [])
    return out, fails


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    out, fails = judge(a, load_inputs(a, t0), t0)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    if fails:
        raise SystemExit("compare: FAIL: " + "; ".join(fails) +
                         " (see the FAIL lines above and the JSON)")


if __name__ == "__main__":
    main()
