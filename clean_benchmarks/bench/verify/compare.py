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
import collections
import hashlib
import json
import os
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


def _corpus_files(fs, root, recursive=True):
    """The corpus's *.parquet file infos, sorted by path: every level (recursive, as nova-bf
    always lists and the harness with --corpus-recursive), or only those directly under the
    root (the harness without it)."""
    root = root.rstrip("/")
    infos = [i for i in fs.get_file_info(pafs.FileSelector(root, recursive=recursive))
             if i.type == pafs.FileType.File and i.path.endswith(".parquet")]
    return sorted(infos, key=lambda i: i.path)


def refuse_flat_with_nested(uri) -> None:
    """A flat listing (corpus.recursive false) with nova-bf in the run: nova-bf always lists
    every level, so *.parquet files below the root would be read by nova-bf but not by the
    harness legs or the checks. Refused (set corpus.recursive true, or move the files)."""
    fs, root = pafs.FileSystem.from_uri(uri)
    n = len(_corpus_files(fs, root)) - len(_corpus_files(fs, root, recursive=False))
    if n:
        raise SystemExit(f"corpus.recursive is false but {n} *.parquet files sit in sub-folders "
                         f"of {uri}: nova-bf lists every level, the harness legs and the checks "
                         "only the root's own, so they would not read the same corpus. Set "
                         "corpus.recursive true (or move the files)")


def corpus_index(uri, threads=32, need_ids=True, max_files=None, recursive=True):
    """(keys, rows, offsets, id column as one arrow array or None). Without
    ``need_ids`` only the footers are read (row counts). ``max_files``: only the first N files
    by sorted path (a subset run's files: the legs' --max-files). ``recursive`` False: only
    the files directly under the root (the harness's listing without --corpus-recursive)."""
    fs, root = pafs.FileSystem.from_uri(uri)
    root = root.rstrip("/")
    infos = _corpus_files(fs, root, recursive)
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
    # large_string (64-bit offsets): 36 bytes per id pass string's 2 GiB limit at ~59.6M rows
    # (Pes2o has 88M); index_in's int32 positions then hold up to 2^31 rows
    if int(offsets[-1]) >= 2**31:
        raise SystemExit(f"uuid_table: {int(offsets[-1])} rows, more than the 2^31 the lookup holds")
    return pa.chunked_array([c.cast(pa.large_string()) for c in t.column("u").chunks],
                            pa.large_string()).combine_chunks()


def uuid_gids(uuids, table: pa.Array) -> np.ndarray:
    """Global row ids of nova-bf hit ids (an arrow string array): their position in
    uuid_table's array; -1 for an id no indexed row has."""
    idx = pc.index_in(uuids.cast(table.type), value_set=table)
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


def _beats(x, y, tie_ulps):
    """Exact scores x better than y beyond the tie window, element-wise (broadcast): both finite,
    by more than tie_ulps float32 ULPs of the larger magnitude (each comparison its own window:
    it is not monotone at a binade edge); otherwise plainly x > y (a -inf non-candidate is the
    worse one, a NaN never better or worse)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        fin = np.isfinite(x) & np.isfinite(y)
        big = np.where(fin, np.maximum(np.abs(x), np.abs(y)), 0.0)
        w = tie_ulps * np.spacing(big.astype(np.float32)).astype(np.float64)
        return np.where(fin, x - y > w, x > y)


def _class_extremes(v):
    """The values ``_beats`` against ``v`` can turn on: each float32 spacing class's max and
    min, and the non-finite values' max and min (NaN dropped). Within a spacing class the
    window against any one x is constant (the spacing of the larger magnitude is the larger
    of the two spacings), so _beats is monotone in v there and its extremes decide it."""
    v = np.asarray(v, np.float64)
    fin = np.isfinite(v)
    out = []
    if fin.any():
        f = v[fin]
        _, inv = np.unique(np.spacing(np.abs(f).astype(np.float32)), return_inverse=True)
        hi = np.full(inv.max() + 1, -np.inf)
        lo = np.full(inv.max() + 1, np.inf)
        np.maximum.at(hi, inv, f)
        np.minimum.at(lo, inv, f)
        out += [hi, lo]
    nf = v[~fin & ~np.isnan(v)]
    if len(nf):
        out.append(np.array([nf.max(), nf.min()]))
    return np.concatenate(out) if out else np.zeros(0)


def stray_verdicts(s_i, s_k, s_q, tie_ulps):
    """(P wrong, Q wrong) boolean arrays for P's strays (ids Q does not have; exact scores
    ``s_i``, both lists full, s_i and s_k finite): Q is wrong where a stray beats an id Q kept
    (its k-th id, exact ``s_k``, or one of Q's own strays, exact ``s_q``): Q dropped a better
    id; P is wrong where one of Q's strays (ids P dropped) beats it: P kept a worse id. Neither
    is a tie at the cutoff. Q's k-th id is its LAST entry, not necessarily its lowest score (Q
    may be out of order), so it proves nothing against P on its own. With Q in order (the k-th
    id its lowest score) this is the k-th rule, up to a binade edge (each comparison has its
    own window) and except that P is blamed too when one of Q's strays beats its stray."""
    si = np.asarray(s_i, np.float64)
    q_wrong = _beats(si, s_k, tie_ulps)
    p_wrong = np.zeros(len(si), bool)
    sq = _class_extremes(s_q)     # every s_q decides the same as these (O(classes), not O(|Q|))
    if len(sq):
        step = max(1, (1 << 20) // len(sq))   # at most ~1M comparisons at once
        for lo in range(0, len(si), step):
            blk = si[lo:lo + step, None]
            q_wrong[lo:lo + step] |= _beats(blk, sq[None, :], tie_ulps).any(axis=1)
            p_wrong[lo:lo + step] = _beats(sq[None, :], blk, tie_ulps).any(axis=1)
    return p_wrong, q_wrong


def mismatched_ranks(A, B):
    """Ranks r < min(len(A), len(B)) where valid-id arrays A and B differ."""
    m = min(len(A), len(B))
    return np.nonzero(A[:m] != B[:m])[0]


def system_pairs(names):
    """Every pair of systems, in round order (what a normal run compares)."""
    names = list(names)
    return [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]


def system_groups(a, names):
    """--group SYSTEM=GROUP (a sweep: each variant's base system) as {group: [systems in round
    order]}; a system without one (the dataset GT, a run's own system) is its own group. None
    without --group: every pair is compared (a normal run)."""
    given = dict(g.split("=", 1) for g in (getattr(a, "group", None) or []))
    if not given:
        return None
    out = {}
    for nm in names:   # keys never collide: a named group vs a system on its own
        out.setdefault(("group", given[nm]) if nm in given else ("system", nm), []).append(nm)
    return out


def group_reps(groups):
    """{system: its group's representative (the group's first system in round order)}."""
    return {m: g[0] for g in groups.values() for m in g}


def same_lists(x, y, n, kmax):
    """Per query (a bool array): are systems x and y = (ids, scores) the same input to
    compare() there? The valid() ids and float64 scores equal (scores bit for bit) and the
    raw top-kmax the same length (compare's masked -1 count)."""
    (ia, sa), (ib, sb) = x, y
    out = np.zeros(n, bool)
    for q in range(n):
        (A, SA), (B, SB) = valid(ia[q], sa[q], kmax), valid(ib[q], sb[q], kmax)
        out[q] = (len(np.asarray(ia[q][:kmax])) == len(np.asarray(ib[q][:kmax]))
                  and np.array_equal(A, B)
                  and np.array_equal(SA.view(np.int64), SB.view(np.int64)))
    return out


def differing_queries(systems, n, kmax, rep_of):
    """{system: [queries where its lists are not its representative's]} (none for a
    representative)."""
    return {nm: ([] if rep_of[nm] == nm else
                 np.flatnonzero(~same_lists(systems[nm], systems[rep_of[nm]], n, kmax)).tolist())
            for nm in systems}


def grouped_plan(systems, n, kmax, groups):
    """A sweep's pairs, per query (see judge): (rep_of, differing, items). ``items`` are what
    needed_ids takes: every pair of distinct representatives over every query, and every
    pair of systems (x, y, queries) over the queries where x or y differs from its
    representative. Their needed ids are exactly every pair's (all-pairs): where both lists
    equal their representatives', a pair needs what its representatives' pair needs (a
    representative with itself needs nothing)."""
    rep_of = group_reps(groups)
    names = list(systems)
    differing = differing_queries(systems, n, kmax, rep_of)
    reps = [nm for nm in names if rep_of[nm] == nm]
    items = list(system_pairs(reps))
    for x, y in system_pairs(names):
        qs = sorted(set(differing[x]) | set(differing[y]))
        if qs:
            items.append((x, y, qs))
    return rep_of, differing, items


def pair_plan(a, systems, n):
    """The pairs judge's rescoring covers (needed_ids ``pairs``): every pair, or with --group
    grouped_plan's items. agree.prefetch_scores uses the same list."""
    names = list(systems)
    groups = system_groups(a, names)
    if groups is None:
        return system_pairs(names)
    return grouped_plan(systems, n, a.k, groups)[2]


def needed_ids(systems, n, kmax, order=False, pairs=None):
    """{query: set of global ids} the float64 adjudication must score: for every
    pair of systems, the ids one list has and the other does not, plus each
    list's k-th id; with ``order`` (the order check) also both ids at every
    rank where the two lists differ. ``pairs``: (x, y) over every query or (x, y, queries)
    over those only (default every pair, every query); (x, y, queries, "full"): every id of
    every system's list on those queries (order_full_check's); (x, y, queries, "shared",
    (atol, rtol)): the ids x and y share whose two scores are shared_score_bad on those
    queries (shared_score_check's), or (x, y, queries, "shared", {query: shared_score_ids
    there}): those ids as already listed."""
    need = [set() for _ in range(n)]
    for item in (system_pairs(systems) if pairs is None else pairs):
        x, y = item[0], item[1]
        ia, ib = systems[x][0], systems[y][0]
        for q in (item[2] if len(item) > 2 else range(n)):
            if len(item) > 3 and item[3] == "shared":
                ids = item[4].get(q, ()) if isinstance(item[4], dict) else \
                    shared_score_ids(systems[x], systems[y], q, kmax, item[4])
                need[q].update(g for g, _, _ in ids)
                continue
            A, B = np.asarray(ia[q][:kmax]), np.asarray(ib[q][:kmax])
            A, B = A[A >= 0], B[B >= 0]
            if len(item) > 3 and item[3] == "full":
                for nm in systems:
                    L = np.asarray(systems[nm][0][q][:kmax])
                    need[q].update(L[L >= 0].tolist())
                continue
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


def _without(need, have):
    """``need`` without the (query, id) pairs ``have`` (a score table) already holds."""
    if not have:
        return need
    return [{g for g in s if (q, g) not in have} for q, s in enumerate(need)]


PAIR_BYTES = 200   # one (query, gid) -> float entry of the score table, Python objects incl.
# per pair beyond its table entry while exact_scores runs: the per-file (query, id, score)
# result lists the reader threads hand back before they are merged
# (~64 B), and each pair's query number as a Python int (28 B past the small-int cache) plus
# dict growth slack: measured 388 B/pair in RSS without a score cache
PAIR_WORK_BYTES = 192
# a decoded column in flight, in copies of its stored bytes: read + combine_chunks + decoder
# buffers. Measured at d=2560: float32 4.02-4.04 (list and fixed-size list, 1-8 row groups,
# both read paths), float16 4.95-5.77; one copy more each for margin
COLUMN_COPIES = 5
COLUMN_COPIES_F16 = 7


def column_copies(itemsize) -> int:
    """COLUMN_COPIES for a stored element of ``itemsize`` bytes (narrower decodes cost more)."""
    return COLUMN_COPIES_F16 if itemsize < 4 else COLUMN_COPIES
READ_FIXED_BYTES = 4 << 20   # per file in flight: the reader's fixed buffers (measured ~1.2 MiB)


def rescore_bytes(need, offsets, d, itemsize, threads, group_rows=None, held_pairs=0,
                  returned_pairs=0):
    """Upper-bound estimate of exact_scores' peak memory, in bytes, before any
    vector is read: per file in flight its decoded vector column (read +
    combine_chunks and decoder overhead: COLUMN_COPIES copies at the stored width) — the whole
    column, or with
    ``group_rows`` (row_group_rows: each needed file's row-group sizes) and ROWGROUP_READ only
    the row groups holding a needed row, as _needed_rows reads them — its needed
    rows widened and normalised (2 float64 copies) and one scoring chunk (two
    gathered float64 operands of DENSE_PAIR_CELLS cells), the score table (plus each pair's
    PAIR_WORK_BYTES while it is computed, and ``held_pairs`` entries a score cache already
    holds beside it, and ``returned_pairs`` entries of the new table exact_scores returns from
    a score cache), and the float64 queries. Returns (bytes, n_gids, n_files)."""
    gids = np.asarray(sorted({g for s in need for g in s}), np.int64)
    n_pairs = sum(len(s) for s in need)
    if not len(gids):
        return (held_pairs + returned_pairs) * PAIR_BYTES, 0, 0
    files = np.searchsorted(offsets, gids, side="right") - 1
    fu, per_file = np.unique(files, return_counts=True)
    rows = np.diff(offsets)[fu]
    if group_rows is not None and ROWGROUP_READ:
        rows = rows.copy()
        for j, f in enumerate(fu.tolist()):
            if group_rows.get(f) is None:   # footer unreadable: keep the whole file
                continue
            sizes = np.asarray(group_rows[f], np.int64)
            ends = np.cumsum(sizes)
            if not len(sizes) or ends[-1] != rows[j]:   # footer disagrees: keep the whole file
                continue
            local = gids[files == f] - offsets[f]
            rows[j] = int(sizes[np.unique(np.searchsorted(ends, local, side="right"))].sum())
    inflight = min(threads, len(fu))
    big_rows = np.sort(rows)[-inflight:].sum()           # the largest files at once
    big_need = np.sort(per_file)[-inflight:].sum()
    est = (big_rows * (d * itemsize + 8) * column_copies(itemsize)  # columns (+ list offsets)
           + inflight * READ_FIXED_BYTES                  # each read's fixed buffers
           + big_need * d * 8 * 2                         # widened + normalised rows
           + inflight * 2 * min(DENSE_PAIR_CELLS, n_pairs * d) * 8   # scoring chunks
           + n_pairs * (PAIR_BYTES + PAIR_WORK_BYTES)     # exact {(q, g): score} + results
           + (held_pairs + returned_pairs) * PAIR_BYTES   # a score cache + its returned table
           + len(need) * d * 8 * 2)                       # q64 (+ its normalised copy)
    return int(est), len(gids), len(fu)


def needed_files(need, offsets):
    """The sorted file indices holding a needed id."""
    gids = np.asarray(sorted({g for s in need for g in s}), np.int64)
    return np.unique(np.searchsorted(offsets, gids, side="right") - 1).tolist() if len(gids) \
        else []


def row_group_rows(paths, fs, files, threads=8):
    """{file index: [rows of each row group]} from the footers of ``files`` (only footers are
    read, each path once per process: _ROW_GROUPS). A file whose footer cannot be read maps to
    None (rescore_bytes counts it whole). None when ROWGROUP_READ is off (whole columns)."""
    if not ROWGROUP_READ or not files:
        return None

    def one(f):
        p = paths[f]
        if p not in _ROW_GROUPS:
            try:
                md = pq.ParquetFile(p, filesystem=fs).metadata
                _ROW_GROUPS[p] = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
            except Exception:  # noqa: BLE001 - the estimate falls back; the read reports it
                _ROW_GROUPS[p] = None
        return _ROW_GROUPS[p]
    todo = [f for f in files if paths[f] not in _ROW_GROUPS]
    if todo:
        with ThreadPoolExecutor(max(1, min(threads, len(todo)))) as ex:
            list(ex.map(one, todo))
    return {f: one(f) for f in files}


# path -> its row-group sizes, or None (unreadable). A corpus file does not change during a
# run; a rewritten file only makes the estimate stale (the read re-reads its footer)
_ROW_GROUPS = {}


def pool_warning() -> None:
    """Print once if Arrow's pool is jemalloc: the rescoring estimates are measured with the
    mimalloc and system pools (jemalloc keeps freed columns resident: 1.3x measured)."""
    global _POOL_WARNED
    if _POOL_WARNED:
        return
    _POOL_WARNED = True
    if pa.default_memory_pool().backend_name == "jemalloc":
        print("WARNING: Arrow's memory pool is jemalloc: the float64 rescoring memory estimate "
              "holds for the mimalloc and system pools only (set ARROW_DEFAULT_MEMORY_POOL="
              "mimalloc or system)", flush=True)


_POOL_WARNED = False


def dense_rescore_plan(a, need, offsets, paths, fs, d, held_extra=0):
    """(threads, estimated bytes, n_gids, n_files) for exact_scores(need, ...) under ``a``'s
    limit (--rescore-max-gib): the estimate covers only the pairs the run's score cache does
    not hold yet (the ones exact_scores reads), plus the cache and the table it returns; the
    files read at once are lowered from --rescore-threads until the estimate fits (fewer reads
    in flight, slower, same scores). The estimate returned is the fitting one, or the
    one-thread one when even that is over the limit (the caller refuses). ``held_extra``:
    score-table entries held beside it besides the cache (a first pass's table)."""
    key = score_cache_key(a)
    store = SCORE_CACHE.get(key) if SCORE_CACHE is not None else None
    todo = _without(need, store) if store else need
    held = (len(store) if store else 0) + held_extra
    returned = sum(len(s) for s in need) if SCORE_CACHE is not None else 0
    pool_warning()
    itemsize = vector_itemsize(paths[0], fs, a.vcol)
    groups = row_group_rows(paths, fs, needed_files(todo, offsets), a.rescore_threads)
    limit = a.rescore_max_gib * 2**30
    for t in range(max(1, int(a.rescore_threads)), 0, -1):
        est, n_g, n_f = rescore_bytes(todo, offsets, d, itemsize, t, groups, held, returned)
        if est <= limit:
            return t, est, n_g, n_f
    return 1, est, n_g, n_f


def vector_itemsize(path, fs, vcol):
    """Bytes per stored vector element (list / fixed-size list of floats), 4 if unknown."""
    try:
        t = pq.read_schema(path, filesystem=fs).field(vcol).type
        return max(1, t.value_type.bit_width // 8)
    except Exception:
        return 4


DENSE_PAIR_CELLS = 1 << 22  # float64 cells of one gathered operand per scoring chunk (32 MiB)
# read only the row groups that hold a needed row (the same stored values as the whole column);
# False reads each file's whole column (the reference the tests compare it with)
ROWGROUP_READ = True
# a run's float64 scores, shared by its checks (agree.run_checks sets it for one run; None =
# off): {score_cache_key(a): {(query, gid): score}}. A score is computed once; a later check or
# round reads only the rows it has no score for yet.
SCORE_CACHE: dict | None = None


def score_cache_key(a) -> tuple:
    """What a (query, gid) score depends on besides the pair: the corpus files, the column, the
    metric and the query set (its rows after --query-rows; filtered searches number them apart)."""
    return (a.corpus, a.vcol, a.metric, a.queries, a.qcol, getattr(a, "query_rows", None),
            getattr(a, "max_files", None), getattr(a, "corpus_flat", False))


def _needed_rows(path, fs, vcol, local):
    """The stored ``vcol`` of the file's rows ``local`` (sorted local row numbers), reading only
    the row groups that hold one (ROWGROUP_READ), else the whole column."""
    if not ROWGROUP_READ:
        t = pq.read_table(path, filesystem=fs, columns=[vcol])
        a = t.column(vcol).combine_chunks()
        return a.take(pa.array(local))
    pf = pq.ParquetFile(path, filesystem=fs)
    md = pf.metadata
    ends = np.cumsum([md.row_group(i).num_rows for i in range(md.num_row_groups)])
    starts = ends - np.array([md.row_group(i).num_rows for i in range(md.num_row_groups)])
    rg = np.searchsorted(ends, local, side="right")
    if len(local) and (local[-1] >= ends[-1] or local[0] < 0):
        raise SystemExit(f"{path}: row {int(local[-1])} beyond its {int(ends[-1])} rows")
    sel = np.unique(rg)
    base = np.zeros(md.num_row_groups, np.int64)      # where each selected group starts in the read
    base[sel] = np.concatenate([[0], np.cumsum((ends - starts)[sel])[:-1]])
    a = pf.read_row_groups(sel.tolist(), columns=[vcol]).column(vcol).combine_chunks()
    return a.take(pa.array(local - starts[rg] + base[rg]))


def _rss_bytes():
    """This process's resident memory in bytes (Linux /proc), or None where unreadable."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE
    except (OSError, ValueError, IndexError):
        return None


_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def release_pool(budget=None) -> None:
    """Hand the Arrow pool's freed buffers back to the OS before a file is read: mimalloc
    (pyarrow's default; the estimates are measured with it and the system pool, NOT jemalloc)
    keeps them resident otherwise, so the process would grow with files READ, not files in
    flight (measured 1.3-1.7x RSS without it). Releasing costs time (the call, and the pages
    faulted in again on the next read: 7-63% of the read measured), so with ``budget``
    (limit bytes, the job's estimated peak) it is skipped while the process's resident memory
    plus the whole estimate still fits under the limit: memory is only handed back when it
    could matter. Without a budget, or where /proc is unreadable, it always releases."""
    if budget is not None:
        rss = _rss_bytes()
        if rss is not None and rss + budget[1] <= budget[0]:
            return
    pa.default_memory_pool().release_unused()


def exact_scores(need, corpus, vcol, offsets, keys_paths, fs, xq, metric, threads=8,
                 cache_key=None, budget=None):
    """{(query, gid): float64 score} from the STORED vectors (widened exactly to
    float64; cosine: both sides normalised in float64), reading only the files
    that hold a needed id. Each file is scored as it is read (no float64 copy
    of the needed vectors is kept), its (query, id) pairs in chunks of at most
    DENSE_PAIR_CELLS gathered float64 cells per operand. With SCORE_CACHE and a ``cache_key``,
    only pairs without a score yet are computed; the result (a new dict, in the same order as
    computing all of ``need``) holds exactly ``need``'s pairs. ``budget``: release_pool's
    (limit bytes, estimated peak)."""
    store = None
    if SCORE_CACHE is not None and cache_key is not None:
        store = SCORE_CACHE.setdefault(cache_key, {})
        todo = [{g for g in s if (q, g) not in store} for q, s in enumerate(need)]
        new = _exact_scores(todo, offsets, keys_paths, fs, xq, metric, vcol, threads,
                            budget) if any(todo) else {}
        store.update(new)
        by_file, by_g = _need_maps(need, offsets)
        return {(q, g): store[(q, g)] for f in sorted(by_file) for g in by_file[f]
                for q in by_g[g]}
    return _exact_scores(need, offsets, keys_paths, fs, xq, metric, vcol, threads, budget)


def _exact_scores(need, offsets, keys_paths, fs, xq, metric, vcol, threads, budget=None):
    by_file, by_g = _need_maps(need, offsets)
    q64 = np.asarray(xq, np.float64)
    if metric == "cosine":
        nq = np.linalg.norm(q64, axis=1, keepdims=True)
        q64 = np.divide(q64, nq, out=np.zeros_like(q64), where=nq > 0)
    chunk = max(1, DENSE_PAIR_CELLS // max(1, q64.shape[1]))

    def load(f):
        gs = by_file[f]
        release_pool(budget)   # the pool's freed column buffers back first, if needed
        sub = _needed_rows(keys_paths[f], fs, vcol, np.asarray(gs, np.int64) - int(offsets[f]))
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


# a ragged (sparse / multivector) column in flight, in copies of its stored bytes: read_table +
# take + the decoder's buffers. Measured: one file, one reader: sparse 3.4-5.3 (by size and
# types; fewer with many row groups), multivector float32 4.95 (1 row group) - 5.41 (30: take
# concatenates the chunks), float16 5.4-5.6; mimalloc fragmentation over many files adds ~0.3
# before it plateaus. Over one copy more each for margin (the narrowest stored width decides,
# as column_copies). Valid for the mimalloc and system pools, NOT jemalloc (1.3x measured)
RAGGED_COLUMN_COPIES = 7
RAGGED_COLUMN_COPIES_F16 = 8
# the longest document a multivector file may hold, in tokens (bge-m3's 8192 cap): the
# footers give only the average, and the longest documents are the ones MaxSim favours
MV_MAX_DOC_TOKENS = 8192


def ragged_column_copies(itemsize) -> int:
    """RAGGED_COLUMN_COPIES for a stored element of ``itemsize`` bytes."""
    return RAGGED_COLUMN_COPIES_F16 if itemsize < 4 else RAGGED_COLUMN_COPIES


def rescore_bytes_sparse(need, offsets, stats, index_bytes, value_bytes, q_nnz, n_q, threads):
    """Estimate of exact_scores_sparse's peak memory, in bytes, before any
    vector is read (``stats``: ragged_file_stats of the files holding a needed
    id). Per file in flight (the largest ``threads``): its whole decoded
    column (ragged_column_copies copies at the stored width) and the reader's fixed buffers,
    its needed rows at the file's average nonzeros per row in float64
    (SPARSE_NZ_BYTES a nonzero), and one chunk of SPARSE_PAIR_CHUNK pairs'
    gathered query and document rows plus their product (12 B a nonzero, 3
    copies); beside them the float64 queries (+ their norms) and the score
    table (+ each pair's PAIR_WORK_BYTES while it is computed). Returns (bytes, n_gids,
    n_files)."""
    gids, fu, per_file, n_pairs = _need_files(need, offsets)
    if not len(gids):
        return 0, 0, 0
    per, d_avg = [], 0.0
    copies = ragged_column_copies(min(index_bytes, value_bytes))
    for f, c in zip(fu.tolist(), per_file.tolist()):
        rows, nnz = stats[f]
        avg = nnz / max(1, rows)
        d_avg = max(d_avg, avg)
        per.append(nnz * (index_bytes + value_bytes) * copies + c * avg * SPARSE_NZ_BYTES)
    inflight = min(threads, len(fu))
    q_avg = q_nnz / max(1, n_q)
    est = (sum(sorted(per)[-inflight:])
           + inflight * READ_FIXED_BYTES
           + inflight * SPARSE_PAIR_CHUNK * (q_avg + d_avg) * 12 * 3
           + q_nnz * (4 + 4 + 12 * 2) + n_q * 8 * 2       # Queries + float64 CSR copies, norms
           + n_pairs * (PAIR_BYTES + PAIR_WORK_BYTES))
    return int(est), len(gids), len(fu)


def rescore_bytes_mv(need, offsets, stats, value_bytes, q_tokens, dim, threads, tile_cells):
    """Estimate of exact_scores_mv's peak memory, in bytes, before any vector
    is read (``stats``: ragged_file_stats, values = token values). Per file in
    flight (the largest ``threads``): its whole decoded column
    (ragged_column_copies copies at the stored width) and the reader's fixed buffers, its
    needed documents at the file's average tokens per document (the taken subset at the
    stored width and its unpacked float32 tokens: value_bytes + 4 a value), one document
    widened and, for cosine, unit-normalised (2 float64 copies of the longest document the
    file can hold: MV_MAX_DOC_TOKENS, or the file's tokens if fewer), and the MaxSim scratch:
    at most MV_Q_TOKENS gathered float64 query tokens and the float64 token score tile with
    its row maxima (2 x the larger of ``tile_cells`` and those query tokens x that longest
    document); beside them the queries
    (float32 + float64 + unit copy) and the score table (+ each pair's PAIR_WORK_BYTES).
    Returns (bytes, n_gids, n_files)."""
    gids, fu, per_file, n_pairs = _need_files(need, offsets)
    if not len(gids):
        return 0, 0, 0
    per, doc_max = [], 0
    copies = ragged_column_copies(value_bytes)
    for f, c in zip(fu.tolist(), per_file.tolist()):
        rows, vals = stats[f]
        avg = vals / max(1, rows)                            # token values per document
        doc_max = max(doc_max, min(MV_MAX_DOC_TOKENS, vals // max(1, dim)))
        per.append(vals * value_bytes * copies + c * avg * (value_bytes + 4))
    inflight = min(threads, len(fu))
    q_tok = min(MV_Q_TOKENS, max(1, q_tokens))
    est = (sum(sorted(per)[-inflight:])
           + inflight * READ_FIXED_BYTES
           + inflight * (q_tok * dim * 8
                         + 2 * 8 * max(tile_cells, q_tok * doc_max)
                         + 2 * 8 * doc_max * dim)              # one document in float64
           + q_tokens * dim * (4 + 8 + 8)
           + n_pairs * (PAIR_BYTES + PAIR_WORK_BYTES))
    return int(est), len(gids), len(fu)


def _take_rows(path, fs, col, rows, budget=None):
    """Table of column ``col`` holding only ``rows`` (file-local, sorted) of one file. The
    pool's freed buffers are handed back first if needed (release_pool(budget))."""
    release_pool(budget)
    return pq.read_table(path, filesystem=fs, columns=[col]).take(pa.array(rows))


def exact_scores_sparse(need, offsets, keys_paths, fs, vcol, q, metric, threads=8,
                        budget=None):
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
        t = _take_rows(keys_paths[f], fs, vcol, np.asarray(gs, np.int64) - int(offsets[f]),
                       budget)
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


def exact_scores_mv(need, offsets, keys_paths, fs, vcol, q, metric, threads=8, budget=None):
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
        t = _take_rows(keys_paths[f], fs, vcol, np.asarray(gs, np.int64) - int(offsets[f]),
                       budget)
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
    if "contradictions" in st:
        order_contradictions(q, r_, exact, tie_ulps, st)


def _beyond(e, lo, tie_ulps) -> bool:
    """e beats lo beyond the tie window (order_check's inversion test)."""
    return e > lo and (not np.isfinite(lo) or e - lo > tie_ulps * ulps32(max(abs(e), abs(lo))))


def _contradicted(p, ev, ranks, eu, tie_ulps) -> np.ndarray:
    """For each u (rank ranks[j] of a list X, exact score eu[j]): does some v (position p[i] in
    X, len(X) when X lacks it; exact score ev[i]) sit after u in X and beat it beyond the tie
    window (_beyond)? Every v is decided: within one group of v sharing a float32 ULP
    (ulps32), the window test is monotone in e_v, so that group's best v after u is the one to
    test; groups are few (a binade each)."""
    out = np.zeros(len(ranks), bool)
    if not len(p) or not len(ranks):
        return out
    with np.errstate(invalid="ignore"):
        key = np.spacing(np.abs(ev).astype(np.float32)).astype(np.float64)
    key = np.where(np.isfinite(ev), key, -1.0)
    key = np.where(np.isnan(ev), np.nan, key)   # a NaN v beats nothing: never a group's best
    for gk in np.unique(key[~np.isnan(key)]):
        m = key == gk
        o = np.argsort(p[m], kind="stable")
        ps, es = p[m][o], ev[m][o]
        suf = np.append(np.maximum.accumulate(es[::-1])[::-1], -np.inf)   # best at >= ps[j]
        js = np.searchsorted(ps, ranks, side="right")   # first v placed after u
        for j in np.flatnonzero(~out & (suf[js] > eu)).tolist():   # (a NaN e_u: False)
            if _beyond(float(suf[js[j]]), float(eu[j]), tie_ulps):
                out[j] = True
    return out


def _positions(L, ids) -> np.ndarray:
    """Each of ``ids``' first rank in list L (len(L) when L lacks it)."""
    pos = {}
    for i, g in enumerate(L.tolist()):
        pos.setdefault(int(g), i)
    return np.array([pos.get(int(g), len(L)) for g in ids], np.int64)


def order_contradictions(q, r_, exact, tie_ulps, st):
    """Per side X, count X's ids u at the mismatched ranks ``r_`` that X keeps above (or keeps
    while lacking) some id v at a mismatched rank of EITHER list whose exact score beats u's
    beyond the tie window: X's own list then contradicts the float64 scores (v must come
    before u in any correct list), whatever the other list holds. A superset of the inversions
    (v later in X at a mismatched rank); also v missing from X, or later in X at another
    rank. Ids with a NaN score are skipped. A count > 0 proves X wrong; 0 proves nothing (u at
    a matched rank is not looked at: see order_full_check). Feeds runner.order_blame."""
    sides = st["sides"]
    U = {}
    for _, L in sides:
        for r in r_.tolist():
            g = int(L[r])
            e = exact[(q, g)]
            if not np.isnan(e):
                U[g] = e
    if not U:
        return
    ug = np.fromiter(U.keys(), np.int64, len(U))
    ue = np.fromiter(U.values(), np.float64, len(U))
    for nm, L in sides:
        eu = np.array([exact[(q, int(L[r]))] for r in r_.tolist()], np.float64)
        st["contradictions"][nm] += int(_contradicted(_positions(L, ug), ue, r_, eu,
                                                      tie_ulps).sum())


def order_failed_queries(x, y, n, kmax, exact, tie_ulps) -> list:
    """The queries on which lists x, y = (ids, scores) have a real float64 order disagreement
    (order_check's ``real``), in query order."""
    qs = []
    for q in range(n):
        A, B = valid(x[0][q], x[1][q], kmax)[0], valid(y[0][q], y[1][q], kmax)[0]
        if not len(mismatched_ranks(A, B)):
            continue
        st = {"ranks": 0, "ties": 0, "real": 0, "not_judged": 0, "queries_real": 0,
              "max_tie_gap": 0.0, "max_real_gap": 0.0, "examples": [],
              "inversions": {0: 0, 1: 0}, "sides": ((0, A), (1, B))}
        order_check(q, A, B, exact, tie_ulps, st)
        if st["real"]:
            qs.append(q)
    return qs


def order_full_check(systems, names, n, queries, kmax, exact, tie_ulps, modality) -> dict:
    """The full-rank order check of the pair ``names`` (x, y) on ``queries`` (its order-failed
    ones), against every id ANY system lists there (U, exact float64 scores; ``exact`` is
    looked up with .get). Per side X and query, X's contradictions are (each a proof X is
    wrong): every u at EVERY rank of X beaten beyond the tie window by some v of U placed
    after u in X or missing from it (_contradicted); a short X (fewer than k ids) missing a v
    of U it should hold (compare()'s short-list rule: dense / multivector a finite score,
    sparse one beyond the window of 0); a dense / multivector X holding a -inf non-candidate.
    A query where some id of U has no score or a NaN one is unjudged for both sides (nothing
    proven there, either way). -> the pair document's keys f64_order_full_queries (queries
    checked), f64_order_full_contradictions_by_system, f64_order_full_unjudged_by_system and
    f64_order_short_queries_by_system (over ALL n queries: those where the side holds fewer
    than k ids, which may be missing rows no system returned). A side with 0 of each is
    consistent with every id any system returned on the order-failed queries and full
    everywhere (runner.order_blame may then clear it)."""
    con = {names[0]: 0, names[1]: 0}
    unj = {names[0]: 0, names[1]: 0}
    for q in queries:
        lists = {nm: valid(systems[nm][0][q], systems[nm][1][q], kmax)[0] for nm in systems}
        ids = list(dict.fromkeys(g for nm in systems for g in lists[nm].tolist()))
        ev = np.array([exact.get((q, int(g)), np.nan) for g in ids], np.float64)
        if np.isnan(ev).any():
            for nm in names:
                unj[nm] += 1
            continue
        ug = np.array(ids, np.int64)
        e_of = dict(zip(ids, ev.tolist()))
        for nm in names:
            L = lists[nm]
            eu = np.array([e_of[int(g)] for g in L.tolist()], np.float64)
            bad = _contradicted(_positions(L, ug), ev, np.arange(len(L)), eu, tie_ulps)
            if modality != "sparse":
                bad |= eu == -np.inf
            n_bad = int(bad.sum())
            if len(L) < kmax:
                held = set(L.tolist())
                for g, e in zip(ids, ev.tolist()):
                    if g in held:
                        continue
                    if (abs(e) > tie_ulps * ulps32(abs(e)) if modality == "sparse"
                            else np.isfinite(e)):
                        n_bad += 1
            con[nm] += n_bad
    short = {nm: sum(len(valid(systems[nm][0][q], systems[nm][1][q], kmax)[0]) < kmax
                     for q in range(n)) for nm in names}
    return {"f64_order_full_queries": len(queries),
            "f64_order_full_contradictions_by_system": con,
            "f64_order_full_unjudged_by_system": unj,
            "f64_order_short_queries_by_system": short}


# judge / compare document keys added on top of compare.py's original --json format;
# legacy_view drops them
D_S8_40_TOP = ("filter_masked_pairs", "score_tolerance", "shared_score_failed_pairs",
               "f64_max_error_abs")
D_S8_40_PAIR = ("shared_score_mismatches", "shared_score_examples", "shared_score_tolerance",
                "f64_max_error_abs", "f64_order_contradictions_by_system",
                "f64_order_full_queries", "f64_order_full_contradictions_by_system",
                "f64_order_full_unjudged_by_system", "f64_order_short_queries_by_system",
                "shared_score_checked", "shared_score_wrong_by_system",
                "shared_score_unjudged_by_system")


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


# the shared-id score loop of compare() in numpy (same float64 values, same order); False runs
# the Python loop it replaced (kept as the reference the tests compare it with)
VECTOR_SHARED = True


def _shared_scores(A, SA, B, SB):
    """(shared ids sorted, A's scores, B's scores) as arrays, or None when the numpy path could
    differ from the Python one: a list repeats an id (its dict keeps the last score) or a
    shared score difference is NaN (Python's max then depends on set order)."""
    if len(np.unique(A)) != len(A) or len(np.unique(B)) != len(B):
        return None
    cid, ia, ib = np.intersect1d(A, B, assume_unique=True, return_indices=True)
    xa, xb = SA[ia], SB[ib]
    if np.isnan(xa - xb).any():
        return None
    return cid, xa, xb


def shared_scores_bad(x, y, atol, rtol) -> np.ndarray:
    """shared_score_bad over arrays (the same float64 operations, element by element)."""
    fin = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore"):
        lim = 2 * (atol + rtol * np.maximum(np.abs(x), np.abs(y)))
        within = np.abs(x - y) <= lim
    return (x != y) & (~fin | ~within)


def shared_score_ids(x, y, q, kmax, score_tol) -> list:
    """[(id, x's score, y's score)] on query q for the ids lists x, y = (ids, scores) share
    whose two scores are shared_score_bad: the ids compare()'s shared_score_mismatches counts
    there (the same valid() lists and per-id scores, the last of a repeated id), in id order."""
    (A, SA), (B, SB) = valid(x[0][q], x[1][q], kmax), valid(y[0][q], y[1][q], kmax)
    fast = _shared_scores(A, SA, B, SB)
    if fast is not None:
        cid, xa, xb = fast
        return [(int(cid[j]), float(xa[j]), float(xb[j]))
                for j in np.flatnonzero(shared_scores_bad(xa, xb, *score_tol))]
    sa_q, sb_q = dict(zip(A.tolist(), SA)), dict(zip(B.tolist(), SB))
    return [(int(i), float(sa_q[i]), float(sb_q[i])) for i in sorted(sa_q.keys() & sb_q.keys())
            if shared_score_bad(sa_q[i], sb_q[i], *score_tol)]


def score_off(s, e, atol, rtol) -> bool:
    """A system's score s of an id against its exact float64 score e (not NaN): off by more
    than the oracle's per-score tolerance atol + rtol x |e|? Equal (incl. the same infinity) =
    never; unequal with either one +-inf / NaN = off."""
    if s == e:
        return False
    if not (np.isfinite(s) and np.isfinite(e)):
        return True
    return not abs(s - e) <= atol + rtol * abs(e)


def shared_score_check(systems, names, queries, kmax, exact, score_tol, listed=None) -> dict:
    """The float64 check of the shared-id score mismatches of the pair ``names`` (x, y) on
    ``queries``: each mismatched id (shared_score_ids) is scored against ``exact`` (looked up
    with .get); a side whose own score is score_off is proven wrong there; an id without an
    exact score (or a NaN one) is unjudged for both sides. -> the pair document's keys
    shared_score_checked (ids checked), shared_score_wrong_by_system and
    shared_score_unjudged_by_system (runner.shared_score_blame's proofs: a side with 0 wrong
    and 0 unjudged agrees with the float64 scores on every mismatched id). ``listed``:
    {query: shared_score_ids there} already computed (only those ids are checked)."""
    x, y = names
    wrong, unj, checked = {x: 0, y: 0}, {x: 0, y: 0}, 0
    for q in queries:
        ids = listed.get(q, ()) if listed is not None else \
            shared_score_ids(systems[x], systems[y], q, kmax, score_tol)
        for g, sx, sy in ids:
            checked += 1
            e = exact.get((q, g), np.nan)
            if np.isnan(e):
                unj[x] += 1
                unj[y] += 1
                continue
            for nm, s in ((x, sx), (y, sy)):
                wrong[nm] += score_off(s, e, *score_tol)
    return {"shared_score_checked": checked, "shared_score_wrong_by_system": wrong,
            "shared_score_unjudged_by_system": unj}


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
           "inversions": {names[0]: 0, names[1]: 0},
           "contradictions": {names[0]: 0, names[1]: 0}}
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
        fast = _shared_scores(A, SA, B, SB) if common and VECTOR_SHARED else None
        if fast is not None:
            cid, xa, xb = fast
            dmax = max(dmax, np.abs(xa - xb).max())
            if score_tol is not None:
                bad = np.flatnonzero(shared_scores_bad(xa, xb, *score_tol))
                sh_bad += len(bad)
                for j in bad[:max(0, 5 - len(sh_ex))]:
                    sh_ex.append({"query": q, "id": int(cid[j]), names[0]: xa[j],
                                  names[1]: xb[j]})
        elif common:
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
                # Q's own strays (ids P dropped; scored, as P's are): with Q's k-th id, the
                # ids a stray of P is weighed against. Q's k-th id is its LAST entry, not
                # necessarily its lowest score (Q may be out of order), so it alone proves
                # nothing about P: stray_verdicts
                q_ids = [int(j) for j in set(Q.tolist()) - set(P.tolist())]
                s_q = np.array([exact[(q, j)] for j in q_ids], np.float64)
                s_all = np.array([exact[(q, i)] for i in strays], np.float64)
                fin = np.isfinite(s_all) & np.isfinite(s_k)
                pw, qw = stray_verdicts(s_all[fin], s_k, s_q, tie_ulps) if fin.any() \
                    else (np.zeros(0, bool), np.zeros(0, bool))
                jf = 0
                for i, s_i, f in zip(strays, s_all.tolist(), fin.tolist()):
                    if f:
                        p_w, q_w = bool(pw[jf]), bool(qw[jf])
                        jf += 1
                        if not (p_w or q_w):
                            f64_ties += 1
                            continue
                        at = [nm for nm, w in ((pname, p_w), (qname, q_w)) if w]
                    else:
                        # (a -inf score, multivector only: no tie window; the -inf side is
                        # the worse one)
                        at = [qname if s_i > s_k else pname]
                    f64_real += 1
                    for nm in at:
                        f64_blame[nm] += 1
                    if len(f64_examples) < 5:
                        f64_examples.append({"query": q, "id": i, "exact": s_i,
                                             "other_kth_exact": s_k,
                                             "wrong": at[0] if len(at) == 1 else at})
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
        res["f64_order_contradictions_by_system"] = ost["contradictions"]
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


# compare()'s names in a grouped judge's documents, renamed to the pair's systems afterwards
# (a representative compared with itself would otherwise collapse the per-system dicts)
SIDES = ("\x00side-a", "\x00side-b")
# compare() document keys, by how one query's documents combine into all queries' (fold_queries)
_SUM = {"f64_order_mismatched_ranks", "f64_order_tie_swaps", "f64_order_real_disagreements",
        "f64_order_queries_with_real", "f64_order_not_judged", "f64_tie_zone_differences",
        "f64_real_disagreements", "f64_not_judged_short_list", "f64_judged_short_list",
        "shared_score_mismatches", "queries_differing_order", "non_boundary_disagreements",
        "queries_short_of_k"}
_SUM_SIDES = {"f64_order_inversions_by_system", "f64_order_contradictions_by_system",
              "f64_wrong_by_system",
              "queries_with_duplicate_ids", "queries_with_masked_minus1"}
_MAX = {"f64_order_max_tie_gap_ulps", "f64_order_max_real_gap_ulps", "max_abs_score_delta_shared"}
_MAX_SIDES = {"f64_max_error_ulps", "f64_max_error_abs"}
_EXAMPLES = {"f64_order_examples", "f64_examples", "shared_score_examples", "non_boundary_examples"}


def fold_queries(recs, kmax):
    """compare()'s document over n queries from its n one-query documents ``recs`` (rec q =
    compare() on query q alone, its exact table keyed (0, gid), names SIDES), with compare()'s
    own arithmetic in its own order: counts summed, overlap sums added query by query, maxima
    folded from 0 with max(), the first 5 examples in query order (their query renumbered),
    the first-difference ranks listed in query order. Equal to compare() over the n queries."""
    n = len(recs)
    first = [r["first_diff_rank_min"] for r in recs if r["first_diff_rank_min"] is not None]
    out = {}
    for key, v0 in recs[0].items():
        vals = [r[key] for r in recs]
        if key.startswith("k") and key[1:].isdigit() or key == "at_k":
            se, ov = 0, 0.0
            for v in vals:
                se += int(v["set_equal"])
                ov += v["overlap"]
            d = max(1, n) if key == "at_k" else n
            out[key] = ({"k": kmax} if key == "at_k" else {}) | {"set_equal": se / d,
                                                                 "overlap": ov / d}
        elif key in _SUM:
            out[key] = sum(vals)
        elif key in _SUM_SIDES:
            out[key] = {s: sum(v[s] for v in vals) for s in v0}
        elif key in _MAX:
            m = 0.0
            for v in vals:
                m = max(m, v)
            out[key] = m
        elif key in _MAX_SIDES:
            out[key] = {}
            for s in v0:
                m = 0.0
                for v in vals:
                    m = max(m, v[s])
                out[key][s] = m
        elif key in _EXAMPLES:
            out[key] = [{**e, "query": q} for q, v in enumerate(vals) for e in v][:5]
        elif key == "shared_score_tolerance":
            out[key] = v0
        elif key == "first_diff_rank_min":
            out[key] = min(first) if first else None
        elif key == "first_diff_rank_median":
            out[key] = statistics.median(first) if first else None
        else:
            raise KeyError(f"fold_queries: no rule for compare() key {key!r}")
    return out


def rename_sides(doc, x, y):
    """A copy of a SIDES-named compare() document with its sides named x, y."""
    to = {SIDES[0]: x, SIDES[1]: y}
    if isinstance(doc, dict):
        return {to.get(k, k) if isinstance(k, str) else k: rename_sides(v, x, y)
                for k, v in doc.items()}
    if isinstance(doc, list):
        return [rename_sides(v, x, y) for v in doc]
    if isinstance(doc, str):
        return to.get(doc, doc)
    return doc


class GroupedPairs:
    """A sweep's pair documents, each equal to compare() on that pair (judge's --group).

    A member's lists are compared with its representative's per query (same_lists). On a
    query where both systems of a pair (x, y) hold their representatives' lists exactly,
    compare() sees the same input as on the representatives' pair, so it gives that pair's
    outcome there; on every other query the pair is compared for real. A pair whose two
    systems equal their representatives on every query takes the representatives' document
    (one compare() over all queries, shared by every such pair); one differing on every query
    is one compare() of its own; otherwise compare() runs query by query (the representatives'
    pair on the equal queries, the pair itself on the rest) and fold_queries combines them.
    ``exact`` must hold the needed ids of grouped_plan's items (= every pair's)."""

    def __init__(self, systems, n, kmax, exact, rep_of, differing, **kw):
        self.systems, self.n, self.kmax, self.exact = systems, n, kmax, exact
        self.rep_of, self.differing, self.kw = rep_of, differing, kw
        self._full, self._recs, self._exq = {}, {}, None
        self.stats = {"pairs": 0, "as_representatives": 0, "full_compares": 0,
                      "query_compares": 0}

    def _compare(self, x, y):
        self.stats["full_compares"] += 1
        return compare(self.systems[x], self.systems[y], self.n, kmax=self.kmax,
                       exact=self.exact, names=SIDES, **self.kw)

    def _exact_of(self, q):
        """The float64 table's query-q slice, keyed (0, gid); built from one scan for the queries
        some member differs on, and on demand for any other."""
        if q not in self._exq:
            self._exq[q] = {(0, g): v for (qq, g), v in self.exact.items() if qq == q}
        return self._exq[q]

    def _rec(self, x, y, q):
        """compare() of x and y on query q alone (memoised: the representatives' pairs)."""
        key = (x, y, q)
        if key not in self._recs:
            if self._exq is None and self.exact is not None:
                # one query's slice of the float64 table, keyed (0, gid), built only for the
                # queries some member differs on (not a copy of the whole table)
                want = {qq for qs in self.differing.values() for qq in qs}
                self._exq = {qq: {} for qq in want}
                for (qq, g), v in self.exact.items():
                    if qq in want:
                        self._exq[qq][(0, g)] = v
            (ia, sa), (ib, sb) = self.systems[x], self.systems[y]
            self.stats["query_compares"] += 1
            self._recs[key] = compare(([ia[q]], [sa[q]]), ([ib[q]], [sb[q]]), 1,
                                      kmax=self.kmax,
                                      exact=None if self.exact is None else self._exact_of(q),
                                      names=SIDES, **self.kw)
        return self._recs[key]

    def doc(self, x, y):
        """compare()'s document of x vs y (names x, y)."""
        self.stats["pairs"] += 1
        rx, ry = self.rep_of[x], self.rep_of[y]
        qs = set(self.differing[x]) | set(self.differing[y])
        if not qs:
            self.stats["as_representatives"] += 1
            if (rx, ry) not in self._full:
                self._full[(rx, ry)] = self._compare(rx, ry)
            d = self._full[(rx, ry)]
        elif len(qs) == self.n:
            d = self._compare(x, y)
        else:
            recs = [self._rec(x, y, q) if q in qs else self._rec(rx, ry, q)
                    for q in range(self.n)]
            for q in qs:   # the pair's own queries are not reused by another pair
                self._recs.pop((x, y, q), None)
            d = fold_queries(recs, self.kmax)
        return rename_sides(d, x, y)


def corpus_paths(corpus, n_keys, max_files=None, recursive=True):
    """(fs, the corpus's *.parquet paths, sorted as corpus_index lists them; the first
    ``max_files`` of them for a subset run)."""
    fs, root = pafs.FileSystem.from_uri(corpus)
    paths = [i.path for i in _corpus_files(fs, root, recursive)]
    if max_files:
        paths = paths[:int(max_files)]
    if len(paths) != n_keys:
        raise SystemExit(f"--rescore: {len(paths)} corpus files now, {n_keys} indexed")
    return fs, paths


def rescore_ragged(a, systems, n, qn, offsets, n_keys, pairs=None, have=None, live=0,
                   qcache=None):
    """--rescore for --modality sparse / multivector: the harness's queries
    (sparse: canonicalised) and the float64 scores of every needed (query,
    id) pair of ``pairs`` (default every pair) not already in ``have``, after the
    memory estimate (refused above --rescore-max-gib; ``live`` score-table entries held beside
    it count too). ``qcache``: a dict keeping the loaded queries for a later call."""
    sparse, multivec = harness_modules()
    mod = sparse if a.modality == "sparse" else multivec
    if qcache is not None and "q" in qcache:
        q = qcache["q"]
    else:
        qid = ragged_query_ids(a.queries)
        if qid is not None and qid != [str(x) for x in qn]:
            raise SystemExit("--queries order != nova-bf query order")
        q, warn = mod.load_queries(a.queries, a.qcol)
        for w in warn:
            print(w, flush=True)
        if q.n_rows != n:
            raise SystemExit(f"--queries has {q.n_rows} rows, nova-bf {n}")
        if qcache is not None:
            qcache["q"] = q
    fs, paths = corpus_paths(a.corpus, n_keys, getattr(a, "max_files", None),
                        not getattr(a, "corpus_flat", False))
    need = _without(needed_ids(systems, n, a.k, order=a.order_check, pairs=pairs), have)
    thr, est, n_g, n_f, what = ragged_rescore_plan(a, need, offsets, paths, fs, q, live)
    print(f"float64 rescoring ({a.modality}): {n_g} ids in {n_f} files, "
          f"{sum(len(x) for x in need)} (query, id) pairs, estimated peak "
          f"{est / 2**30:.2f} GiB (limit --rescore-max-gib {a.rescore_max_gib:g})", flush=True)
    if thr < a.rescore_threads and est <= a.rescore_max_gib * 2**30:
        print(f"float64 rescoring: {thr} files read at once (not --rescore-threads "
              f"{a.rescore_threads}) to fit the limit", flush=True)
    if est > a.rescore_max_gib * 2**30:
        raise SystemExit(
            f"--rescore refused: estimated {est / 2**30:.2f} GiB > --rescore-max-gib "
            f"{a.rescore_max_gib:g} ({what} + one file's columns in flight + the score "
            "table, already at one file read at a time). Raise --rescore-max-gib if the "
            "machine has the memory, or compare fewer systems")
    fn = exact_scores_sparse if a.modality == "sparse" else exact_scores_mv
    return fn(need, offsets, paths, fs, a.vcol, q, a.metric, threads=thr,
              budget=(a.rescore_max_gib * 2**30, est))


def ragged_rescore_plan(a, need, offsets, paths, fs, q, live=0):
    """(threads, estimated bytes, n_gids, n_files, what) for exact_scores_sparse / _mv(need,
    ...) under ``a``'s limit (--rescore-max-gib), as dense_rescore_plan: the files read at once
    are lowered from --rescore-threads until the estimate fits (fewer reads in flight, slower,
    same scores); the estimate returned is the fitting one, or the one-thread one when even
    that is over the limit (the caller refuses). ``live``: score-table entries held beside it
    (a first pass's table). ``q``: the loaded queries. ``what``: the refusal's description."""
    sparse, multivec = harness_modules()
    _, fu, _, _ = _need_files(need, offsets)
    if len(fu):
        sch = pq.read_schema(paths[int(fu[0])], filesystem=fs)
        if a.vcol not in sch.names:
            raise SystemExit(f"--vcol {a.vcol!r}: {paths[int(fu[0])]} has no such column")
        typ = sch.field(a.vcol).type
    pool_warning()
    stats = ragged_file_stats(paths, fu, fs, a.vcol)
    if a.modality == "sparse":
        if len(fu) and sparse.sparse_fields(typ) is None:
            raise SystemExit(f"--modality sparse: --vcol {a.vcol!r} is {typ}, not a "
                             "struct<indices: list<int>, values: list<float>>")
        ib, vb = ((t.value_type.bit_width // 8 for t in sparse.sparse_fields(typ))
                  if len(fu) else (4, 4))

        def est_at(t):
            return rescore_bytes_sparse(need, offsets, stats, ib, vb, q.nnz, q.n_rows, t)
        what = "sparse rows' nonzeros (from the footers)"
    else:
        if len(fu) and multivec.mv_value_type(typ) is None:
            raise SystemExit(f"--modality multivector: --vcol {a.vcol!r} is {typ}, not a "
                             "list<list<float>>")
        vb = multivec.mv_value_type(typ).bit_width // 8 if len(fu) else 4

        def est_at(t):
            return rescore_bytes_mv(need, offsets, stats, vb, q.n_tok, q.dim, t,
                                    multivec.CPU_TILE_CELLS)
        what = f"documents' tokens of {q.dim} dims (from the footers)"
    limit = a.rescore_max_gib * 2**30
    thr = max(1, int(a.rescore_threads))
    while True:
        est, n_g, n_f = est_at(thr)
        est += live * PAIR_BYTES
        if est <= limit or thr == 1:
            return thr, est, n_g, n_f, f"{n_g} {what}"
        thr -= 1


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
    ap.add_argument("--group", action="append", default=[], metavar="SYSTEM=GROUP",
                    help="a sweep's variant and its base system (repeatable). With any --group, "
                         "every pair is still judged (the same documents and verdicts), but a "
                         "member is compared for real only on the queries where its lists are "
                         "not its group's first system's exactly; elsewhere it takes that "
                         "system's outcomes")
    ap.add_argument("--corpus-flat", action="store_true",
                    help="list only the files directly under the corpus root (the experiment's "
                         "corpus.recursive is false: the harness legs list that way)")
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
    if getattr(a, "corpus_flat", False) and a.novabf != "-":
        refuse_flat_with_nested(a.corpus)
    keys, rows, offsets, id_col = corpus_index(
        a.corpus, need_ids=(bool(a.gt) and not gt_npz) or corpus_ids, max_files=getattr(a, "max_files", None),
        recursive=not getattr(a, "corpus_flat", False))
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


def order_full_checks(a, systems, n, pairs_doc, order_bad, exact, rescore,
                      defects=None) -> None:
    """The full-rank order check (order_full_check) of the order-failed pairs where clearing
    is in question (runner.order_clearing_candidates: a side the common-system rule blames has
    no proof against it), on each such pair's order-failed queries only: the ids any list
    holds there that the first pass did not score are rescored (``rescore``, judge's, a second
    pass over the missing ids only, its memory estimate counting the first pass's table still
    held), and the pair's document gains the f64_order_full_* keys. A passing order check, or
    one whose failures are all already explained by proofs, costs nothing. If the rescoring
    is refused, the keys are left out (runner.order_blame then never clears)."""
    from bench import runner
    by_key = {f"{x}_vs_{y}": (x, y) for x, y in system_pairs(list(systems))}
    listed = [nm for nm in systems if nm != "gt"]
    # a side short of k on any query is never cleared (order_full_check's short count): known
    # before any rescoring, so such sides put no pair in question
    short = {nm for nm in listed
             if any(len(valid(systems[nm][0][q], systems[nm][1][q], a.k)[0]) < a.k
                    for q in range(n))}
    cand = runner.order_clearing_candidates(
        {"pairs": pairs_doc, "f64_order_failed_pairs": order_bad,
         "system_defects": defects or {}}, listed, unclearable=short)
    todo = []
    for pk in order_bad:
        xy = by_key.get(pk)
        if xy is None or pk not in cand:
            continue
        x, y = xy
        qs = order_failed_queries(systems[x], systems[y], n, a.k, exact, a.tie_ulps)
        todo.append((x, y, qs, "full"))
    if not todo:
        return
    need = _without(needed_ids(systems, n, a.k, pairs=todo), exact)
    extra = {}
    if any(need):
        try:
            extra = rescore(todo, exact, live=len(exact))
        except SystemExit as exc:
            print(f"note: the full-rank order check was not run ({exc}); order blame stays "
                  "the common-system rule's", flush=True)
            return
        if extra is None:
            return
    look = collections.ChainMap(extra, exact)
    for x, y, qs, _ in todo:
        pairs_doc[f"{x}_vs_{y}"].update(order_full_check(systems, (x, y), n, qs, a.k, look,
                                                         a.tie_ulps, a.modality))


# the most mismatched (query, id) pairs one pair's shared-score check rescores (~64 MiB of
# score table); past it the pair is sampled (shared_score_checks)
SHARED_CHECK_MAX_IDS = 200_000


def shared_score_checks(a, systems, n, pairs_doc, score_bad, exact, rescore) -> None:
    """The float64 check (shared_score_check) of the shared-id score-failed pairs
    ``score_bad`` (judge's, the dataset GT's pairs left out), on each pair's mismatched
    (query, id) pairs only: those the first pass did not score are rescored (``rescore``,
    judge's, a pass over the missing ids only, its memory estimate counting the first pass's
    table still held), and the pair's document gains the shared_score_checked / _wrong_ /
    _unjudged_by_system keys (runner.shared_score_blame's proofs). No failed pair, no pass. If
    the rescoring is refused, the keys are left out (the blame stays the common-system
    rule's). A pair with more than SHARED_CHECK_MAX_IDS mismatched ids (e.g. a system
    reporting float16-rounded scores: every shared id) is checked on its first that many (in
    query order) only: a proof there is a proof, but no side is cleared (shared_score_checked
    < shared_score_mismatches), so the rule's blame stands for the rest."""
    by_key = {f"{x}_vs_{y}": (x, y) for x, y in system_pairs(list(systems))}
    tol = (a.score_atol, a.score_rtol)
    todo = []
    for pk in score_bad:
        xy = by_key.get(pk)
        if xy is None:
            continue
        x, y = xy
        listed, left = {}, SHARED_CHECK_MAX_IDS
        for q in range(n):
            if left <= 0:
                break
            ids = shared_score_ids(systems[x], systems[y], q, a.k, tol)
            if ids:
                listed[q] = ids[:left]
                left -= len(listed[q])
        todo.append((x, y, sorted(listed), "shared", listed))
    if not todo:
        return
    # (only whether anything is missing: the rescore builds its own need, counted by its gate)
    missing = any(any((q, g) not in exact for g, _, _ in ids)
                  for _, _, _, _, listed in todo for q, ids in listed.items())
    extra = {}
    if missing:
        try:
            extra = rescore(todo, exact, live=len(exact))
        except SystemExit as exc:
            print(f"note: the float64 shared-id score check was not run ({exc}); shared-id "
                  "score blame stays the common-system rule's", flush=True)
            return
        if extra is None:
            return
    look = collections.ChainMap(extra, exact)
    for x, y, qs, _, listed in todo:
        r = shared_score_check(systems, (x, y), qs, a.k, look, tol, listed)
        pairs_doc[f"{x}_vs_{y}"].update(r)
        print(f"float64 shared-id score check {x}_vs_{y}: {r['shared_score_checked']} ids, "
              f"off the float64 score (> {tol[0]:g} + {tol[1]:g} x |exact|): "
              f"{r['shared_score_wrong_by_system']}, unjudged "
              f"{r['shared_score_unjudged_by_system']}", flush=True)


def judge(a, inputs, t0, exact_hook=None):
    """main()'s verdict half: list defects, float64 rescoring, every pair (with --group, a
    sweep's pairs computed per query by identity: GroupedPairs, the same documents), and the
    FAILs.
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
    xq = fs = paths = None
    if a.rescore and a.modality == "dense":
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
        fs, paths = corpus_paths(a.corpus, len(keys), getattr(a, "max_files", None),
                        not getattr(a, "corpus_flat", False))
    masked = None

    ragged_q = {}   # rescore_ragged's queries, loaded once for both passes

    def rescore(pairs, have=None, live=0):
        """The float64 scores of ``pairs``' needed (query, id) pairs not already in ``have``
        (exact_hook applied to them), or None without --rescore. ``live``: score-table entries
        held beside it meanwhile (the first pass's, during order_full_checks' pass)."""
        nonlocal masked
        if not a.rescore:
            return None
        if a.modality != "dense":
            ex = rescore_ragged(a, systems, n, qn, offsets, len(keys), pairs, have,
                                live=live, qcache=ragged_q)
        else:
            need = _without(needed_ids(systems, n, a.k, order=a.order_check, pairs=pairs), have)
            thr, est, n_g, n_f = dense_rescore_plan(a, need, offsets, paths, fs, xq.shape[1],
                                                    held_extra=live)
            print(f"float64 rescoring: {n_g} ids in {n_f} files, {sum(len(x) for x in need)} "
                  f"(query, id) pairs, estimated peak {est / 2**30:.2f} GiB "
                  f"(limit --rescore-max-gib {a.rescore_max_gib:g})", flush=True)
            if thr < a.rescore_threads and est <= a.rescore_max_gib * 2**30:
                print(f"float64 rescoring: {thr} files read at once (not --rescore-threads "
                      f"{a.rescore_threads}) to fit the limit", flush=True)
            if est > a.rescore_max_gib * 2**30:
                raise SystemExit(
                    f"--rescore refused: estimated {est / 2**30:.2f} GiB > --rescore-max-gib "
                    f"{a.rescore_max_gib:g} ({n_g} float64 vectors of {xq.shape[1]} dims + "
                    f"one file's columns in flight + the score table, already at one file read "
                    "at a time). Raise --rescore-max-gib if the machine has the memory, or "
                    "compare fewer systems")
            ex = exact_scores(need, a.corpus, a.vcol, offsets, paths, fs,
                              xq, a.metric, threads=thr,
                              cache_key=score_cache_key(a),
                              budget=(a.rescore_max_gib * 2**30, est))
        print(f"float64 rescoring: {len(ex)} (query, id) pairs ({time.time() - t0:.0f} s)",
              flush=True)
        if exact_hook is not None:
            m = exact_hook(ex)
            masked = (masked or 0) + m
            print(f"filter: {m} filter-excluded (query, id) pairs scored -inf", flush=True)
        return ex

    out = {"queries": n, "k": a.k, "corpus_rows": n_ids, "corpus_duplicate_ids": dup,
           "novabf_unmapped_hits": unmapped, "gt_ids_not_in_corpus": gt_missing,
           "tol": a.tol, "rescore": a.rescore, "order_check": a.order_check,
           "tie_ulps": a.tie_ulps,
           "modality": a.modality, "system_defects": defects, "filter_masked_pairs": None,
           "score_tolerance": [a.score_atol, a.score_rtol],
           "ok": not any(df["failed"] for df in defects.values()), "pairs": {}}
    exact = None

    def show(x, y, r):
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

    kw = dict(tol=a.tol, tie_ulps=a.tie_ulps, modality=a.modality, order=a.order_check,
              score_tol=(a.score_atol, a.score_rtol))
    names = list(systems)
    groups = system_groups(a, names)
    if groups is None:   # a normal run: every pair
        pairs = system_pairs(names)
        exact = rescore(pairs)
        for x, y in pairs:
            out["pairs"][f"{x}_vs_{y}"] = r = compare(
                systems[x], systems[y], n, kmax=a.k, exact=exact, names=(x, y), **kw)
            show(x, y, r)
    else:
        # a sweep (--group): every pair's document, each equal to compare() on that pair, but
        # computed per query by identity (GroupedPairs): on a query where a member's lists are
        # its group representative's (its group's first system) exactly, its outcomes there
        # are the representative's; elsewhere it is compared for real. The rescoring covers
        # the same (query, id) pairs as every pair would (grouped_plan), so the float64 table,
        # every pair document and every verdict below equal the all-pairs ones.
        rep_of, differing, items = grouped_plan(systems, n, a.k, groups)
        exact = rescore(items)
        gp = GroupedPairs(systems, n, a.k, exact, rep_of, differing, **kw)
        for x, y in system_pairs(names):
            out["pairs"][f"{x}_vs_{y}"] = r = gp.doc(x, y)
            show(x, y, r)
        out["pair_mode"] = "grouped"
        out["group_of"] = rep_of
        out["group_differing_queries"] = {m: len(differing[m]) for m in names
                                          if rep_of[m] != m}
        out["grouped_compares"] = gp.stats
        for m, c in out["group_differing_queries"].items():
            print(f"group {rep_of[m]}: {m} has its lists on {n - c} of {n} queries "
                  f"({c} compared for real)", flush=True)
        print(f"grouped pairs: {gp.stats}", flush=True)
        gp._exq, gp._recs = {}, {}   # its per-query slices and documents: not held further
    out["filter_masked_pairs"] = masked
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
        if order_bad and exact is not None:
            order_full_checks(a, systems, n, out["pairs"], order_bad, exact, rescore, defects)
            out["filter_masked_pairs"] = masked   # (+ the full check's rows; same key place)
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
    if score_bad and exact is not None:
        # who is wrong: the mismatched shared ids against the float64 scores (a pass over
        # those not scored yet); grouped or not, the same pairs, ids and table
        shared_score_checks(a, systems, n, out["pairs"], score_bad, exact, rescore)
        out["filter_masked_pairs"] = masked   # (+ this pass's rows; same key place)
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
