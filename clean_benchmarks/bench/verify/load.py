"""Load inputs for correctness verification: run records, corpus metadata,
queries, filters, and each system's first-round results mapped to global
row IDs. Record input hashes and thread count for reproducibility.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS = ("cosine", "ip")   # ip is nova-bf's dot
OUT_OF_RANGE = -2   # replaces a harness id outside [0, rows) other than the -1 pad: a list defect
INPUT_SKIP = (".tmp", ".pending")
UUID_CACHE_VERSION = 1      # bump when uuid_table's meaning changes (old cache files then miss)


class CheckError(Exception):
    pass


# corpus_index / uuid_table results, cached for one check.run call (every search reads the
# same corpus); check.run clears it
CACHE: dict = {}


# ---------------------------------------------------------------------------------------------
# what one search's check reads
# ---------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    corpus: str                     # the corpus root, as a leg resolves it
    vcol: str                       # corpus vector column
    queries: str                    # the queries parquet
    qcol: str                       # query vector column
    metric: str                     # cosine | ip
    k: int
    query_rows: str | None = None   # COLUMN=V1,V2: a filtered search's `rows` selection
    max_files: int | None = None    # a subset run: the first N corpus files (sorted by path)
    recursive: bool = False         # list every level below the corpus root, else only its own
    modality: str = "dense"         # dense | filtered | sparse | multivector


def abs_path(p: str) -> str:
    """A path as a leg resolves it: placeholders filled in, ~ expanded, relative = to the leg
    cwd (clean_benchmarks)."""
    from bench import paths
    p = paths.expand(p)
    if "://" in p:
        return p
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(CB, p))


def _yaml(path):
    import yaml
    return yaml.safe_load(open(path))


def query_rows(exp: dict, search: dict) -> str | None:
    """filtered: COLUMN=V1,V2 from the filter config's `rows` selector for this search."""
    if exp["modality"] != "filtered":
        return None
    cfg = _yaml(exp["filter"]["config"])
    name = search.get("filter_search") or search["name"]
    s = next((x for x in cfg.get("searches") or [] if x.get("name") == name), None)
    if s is None:
        raise CheckError(f"filter config {exp['filter']['config']}: no search {name!r}")
    r = s.get("rows")
    if not r:
        return None
    return r["column"] + "=" + ",".join(map(str, r["isin"]))


def search_args(exp: dict, search: dict) -> Args:
    """What one search's check reads (``exp``: filtered with the run's own filter config)."""
    if exp["search"]["metric"] not in METRICS:
        raise CheckError(f"search.metric {exp['search']['metric']!r}: the checks score cosine | ip")
    sub = (exp.get("subset") or {}).get("files")
    return Args(corpus=abs_path(exp["corpus"]["path"]), vcol=exp["corpus"]["column"],
                queries=abs_path(exp["queries"]["path"]), qcol=exp["queries"]["column"],
                metric=exp["search"]["metric"], k=int(exp["search"]["k"]),
                query_rows=query_rows(exp, search),
                max_files=int(sub) if sub else None,
                # the harness lists only the root's own level unless corpus.recursive
                recursive=bool(exp["corpus"].get("recursive")),
                modality=exp["modality"])


# ---------------------------------------------------------------------------------------------
# leg records
# ---------------------------------------------------------------------------------------------
def records(run_dir: str) -> dict:
    """{tag: leg record, or None when unreadable} of the run's legs/."""
    out = {}
    d = os.path.join(run_dir, "legs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".json") and not f.endswith(".clock.json"):
            try:
                out[f[:-5]] = json.load(open(os.path.join(d, f)))
            except (OSError, ValueError):
                out[f[:-5]] = None
    return out


def novabf_config(rec: dict) -> str | None:
    """The nova-bf config path a leg ran (`compute CONFIG` in its argv)."""
    argv = rec.get("argv") or []
    return argv[argv.index("compute") + 1] if "compute" in argv else None


def novabf_result(rec: dict, search: dict) -> str:
    """The nova-bf result parquet of ``search`` a leg wrote (counters.results)."""
    res = (rec.get("counters") or {}).get("results") or {}
    if len(res) == 1:
        return next(iter(res.values()))
    for key in (search.get("filter_search"), search["name"]):
        if key in res:
            return res[key]
    raise CheckError(f"{rec.get('tag')}: {len(res)} nova-bf results {sorted(res)}, none for search "
                     f"{search['name']!r}")


def _is_f16(rec: dict) -> bool:
    argv = rec.get("argv") or []
    return any(a == "--dtype" and i + 1 < len(argv) and argv[i + 1] == "f16" for i, a in enumerate(argv))


def usable(run_dir, recs, tag, search, skipped):
    """The leg record of ``tag`` with its result paths (_result / _config, or _topk), or None
    after appending why to ``skipped`` (no readable record, a failed leg, fp16 scores)."""
    rec = recs.get(tag)
    if rec is None:
        skipped.append({"leg": tag, "why": "no readable leg record"})
        return None
    if not rec.get("ok"):
        skipped.append({"leg": tag, "why": "leg failed (summary lists it)"})
        return None
    rec = dict(rec)
    if rec.get("kind") == "novabf":
        rec["_result"] = novabf_result(rec, search)
        rec["_config"] = novabf_config(rec)
    else:
        if _is_f16(rec):
            skipped.append({"leg": tag, "why": "fp16 scores, inexact by design"})
            return None
        rec["_topk"] = os.path.join(run_dir, rec["topk"]) if rec.get("topk") else None
        if not rec["_topk"] or not os.path.isfile(rec["_topk"]):
            raise CheckError(f"{tag}: no saved top-k")
    return rec


# ---------------------------------------------------------------------------------------------
# the corpus as the legs read it
# ---------------------------------------------------------------------------------------------
def uuid_of(key, row):
    """nova-bf's point id of a corpus row: the md5 of `key:row` as a uuid string."""
    h = hashlib.md5(f"{key}:{row}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _corpus_files(fs, root, recursive=True):
    """The corpus's *.parquet file infos, sorted by path: every level (recursive, as nova-bf
    always lists and the harness with --corpus-recursive), or only those directly under the
    root (the harness without it)."""
    import pyarrow.fs as pafs
    root = root.rstrip("/")
    infos = [i for i in fs.get_file_info(pafs.FileSelector(root, recursive=recursive))
             if i.type == pafs.FileType.File and i.path.endswith(".parquet")]
    return sorted(infos, key=lambda i: i.path)


def refuse_flat_with_nested(uri) -> None:
    """A flat listing (corpus.recursive false) with nova-bf in the run: nova-bf always lists
    every level, so *.parquet files below the root would be read by nova-bf but not by the
    harness legs or the checks. Refused (set corpus.recursive true, or move the files)."""
    import pyarrow.fs as pafs
    fs, root = pafs.FileSystem.from_uri(uri)
    n = len(_corpus_files(fs, root)) - len(_corpus_files(fs, root, recursive=False))
    if n:
        raise CheckError(f"corpus.recursive is false but {n} *.parquet files sit in sub-folders "
                         f"of {uri}: nova-bf lists every level, the harness legs and the checks "
                         "only the root's own, so they would not read the same corpus. Set "
                         "corpus.recursive true (or move the files)")


def corpus_index(uri, threads=32, need_ids=True, max_files=None, recursive=True):
    """(keys, rows, offsets, id column as one arrow array or None). Without ``need_ids`` only
    the footers are read (row counts). ``max_files``: only the first N files by sorted path (a
    subset run's files: the legs' --max-files). ``recursive`` False: only the files directly
    under the root (the harness's listing without --corpus-recursive)."""
    import pyarrow as pa
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
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


def corpus_paths(corpus, n_keys, max_files=None, recursive=True):
    """(fs, the corpus's *.parquet paths, sorted as corpus_index lists them; the first
    ``max_files`` of them for a subset run)."""
    import pyarrow.fs as pafs
    fs, root = pafs.FileSystem.from_uri(corpus)
    paths = [i.path for i in _corpus_files(fs, root, recursive)]
    if max_files:
        paths = paths[:int(max_files)]
    if len(paths) != n_keys:
        raise CheckError(f"{len(paths)} corpus files now, {n_keys} indexed")
    return fs, paths


def uuid_table(keys, rows, offsets):
    """nova-bf's point id (uuid_of) of every indexed corpus row, one string array whose
    position is the global row id. DuckDB's md5 (multi-threaded, the same hex digest of the
    same UTF-8 bytes as hashlib's) computes them in bulk: uuid_of per row in Python took most
    of a check's time on a large corpus."""
    import duckdb
    import pyarrow as pa
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
        raise CheckError("uuid_table: the generated row ids are not 0..rows-1")
    # large_string (64-bit offsets): 36 bytes per id pass string's 2 GiB limit at ~59.6M rows
    # (Pes2o has 88M); index_in's int32 positions then hold up to 2^31 rows
    if int(offsets[-1]) >= 2**31:
        raise CheckError(f"uuid_table: {int(offsets[-1])} rows, more than the 2^31 the lookup holds")
    return pa.chunked_array([c.cast(pa.large_string()) for c in t.column("u").chunks],
                            pa.large_string()).combine_chunks()


def uuid_cache_path(cache_dir, keys, identity) -> str:
    """The on-disk uuid_table of a corpus: <cache_dir>/uuids-<key>.arrow, the key hashing
    UUID_CACHE_VERSION, the files' identity (exact.Corpus.identity: path, size, mtime and footer
    hash of each file, in order) and the keys nova-bf names their rows by."""
    doc = {"v": UUID_CACHE_VERSION, "files": identity, "keys": [str(k) for k in keys]}
    key = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]
    return os.path.join(cache_dir, f"uuids-{key}.arrow")


def cached_uuid_table(keys, rows, offsets, path):
    """(uuid_table(keys, rows, offsets), hit): read from ``path`` (an Arrow IPC file, memory
    mapped) when it holds a large_string array of one id per indexed row and no null, else
    built and written there atomically (temp file + os.replace); an unreadable or wrong-length
    file is rebuilt and replaced. Building is a DuckDB md5 per corpus row (about 1 s per 1.6M
    rows); the file is reused by every check of the same corpus files."""
    import tempfile
    import pyarrow as pa
    if os.path.isfile(path):
        try:
            with pa.memory_map(path, "r") as src:
                col = pa.ipc.open_file(src).read_all().column("u")
            if col.type == pa.large_string() and len(col) == int(offsets[-1]) and not col.null_count:
                return col.combine_chunks(), True
        except (OSError, ValueError, KeyError, pa.ArrowException):
            pass   # unreadable or wrong: rebuilt and replaced below
    tab = uuid_table(keys, rows, offsets)
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".arrow.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            t = pa.table({"u": tab})
            with pa.ipc.new_file(fh, t.schema) as w:
                w.write_table(t)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return tab, False


def _index_in(values, value_set) -> np.ndarray:
    """Position of each of ``values`` in ``value_set`` (arrow arrays of one type), -1 for none:
    one pyarrow index_in call. (Splitting it across threads rebuilt the hash set of the whole
    value set per piece, all at once: ~90 bytes per corpus row per piece."""
    import pyarrow.compute as pc
    return pc.fill_null(pc.index_in(values, value_set=value_set), -1).to_numpy(
        zero_copy_only=False).astype(np.int64)


def uuid_gids(uuids, table) -> np.ndarray:
    """Global row ids of nova-bf hit ids (an arrow string array): their position in
    uuid_table's array; -1 for an id no indexed row has."""
    return _index_in(uuids.cast(table.type), table)


def _cached_index(uri, need_ids, max_files=None, recursive=True):
    key = ("index", uri, need_ids, max_files, recursive)
    if key not in CACHE:
        CACHE[key] = corpus_index(uri, need_ids=need_ids, max_files=max_files, recursive=recursive)
    return CACHE[key]


def _cached_uuids(a, keys, rows, offsets, cache_dir=None):
    """uuid_table of the search's corpus: once per check.run (CACHE), and with ``cache_dir``
    (check.run's reference cache dir) once per corpus on disk (cached_uuid_table)."""
    key = ("u2g", a.corpus, len(keys))
    if key not in CACHE:
        if cache_dir:
            from bench.verify import exact
            fs, paths = corpus_paths(a.corpus, len(keys), a.max_files, a.recursive)
            ident = exact.Corpus(fs, paths, offsets).identity()
            CACHE[key] = cached_uuid_table(keys, rows, offsets,
                                           uuid_cache_path(cache_dir, keys, ident))[0]
        else:
            CACHE[key] = uuid_table(keys, rows, offsets)
    return CACHE[key]


# ---------------------------------------------------------------------------------------------
# each system's lists
# ---------------------------------------------------------------------------------------------
def _hits(t, g):
    """(query ids, per query: global row ids from ``g`` (flat), per query: scores)."""
    import pyarrow.compute as pc
    h = t.column("hit_ids").combine_chunks()
    s = t.column("hit_scores").combine_chunks()
    oh, os_ = h.offsets.to_numpy(), s.offsets.to_numpy()
    sf = pc.list_flatten(s).to_numpy(zero_copy_only=False)
    return (t.column("query_id").to_pylist(),
            [g[oh[i]:oh[i + 1]] for i in range(len(h))],
            [sf[os_[i]:os_[i + 1]] for i in range(len(s))])


def load_novabf(path, u2g):
    """nova-bf's result with its hit ids (uuids) as global row ids. ``u2g``: uuid_table's array."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["query_id", "hit_ids", "hit_scores"])
    return _hits(t, uuid_gids(pc.list_flatten(t.column("hit_ids").combine_chunks()), u2g))


def load_novabf_ids(path, id_col):
    """nova-bf's result with its hit ids (the corpus `id` column's values, id_column: id) as
    global row ids; -1 for an id no row has."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["query_id", "hit_ids", "hit_scores"])
    flat = pc.list_flatten(t.column("hit_ids").combine_chunks()).cast(pa.string())
    return _hits(t, _index_in(flat, id_col))


def load_npz(path, n_ids=None):
    """The harness's saved top-k: (ids rows, score rows). With ``n_ids`` (the corpus rows), an id
    that is neither -1 (padding) nor in [0, n_ids) becomes OUT_OF_RANGE: list_defects counts it
    (a FAIL), and every later step treats it as no id (never an IndexError)."""
    z = np.load(path)
    I = z["I"].astype(np.int64)
    if n_ids is not None:
        I = np.where((I == -1) | ((I >= 0) & (I < int(n_ids))), I, OUT_OF_RANGE)
    return [row for row in I], [row for row in z["D"]]


def list_defects(ids, kmax, mapped):
    """Per-system list checks: duplicate ids, unmapped entries (-1 before a valid id; with
    ``mapped``, any -1) and out-of-range ids (load_npz's OUT_OF_RANGE)."""
    return list_defects_and_bad(ids, kmax, mapped)[0]


def list_defects_and_bad(ids, kmax, mapped):
    """(list_defects' document, per query: any R1 defect as check.defects defines it: a
    duplicate id, an id below -1, or a negative entry before the last valid id (any, with
    ``mapped``)). Vectorised over an (queries x longest list) matrix of each list cut to
    ``kmax``; positions past a list's end are not entries (a nova-bf list may be shorter than
    k; a harness list is padded with -1 entries). The same document as the per-query loop it
    replaced (tests/verify/test_list_defects.py keeps that loop and compares)."""
    rows = [np.asarray(r[:kmax], np.int64) for r in ids]
    n = len(rows)
    lens = np.array([len(r) for r in rows], np.int64)
    w = int(lens.max(initial=0))
    pos = np.arange(w)
    valid = pos[None, :] < lens[:, None]          # (n, w): a real entry of the list
    M = np.full((n, w), -1, np.int64)
    if n and w:
        M[valid] = np.concatenate(rows)
    good = valid & (M >= 0)
    oor = valid & (M == OUT_OF_RANGE)
    # duplicates: equal neighbours among each row's sorted good ids (the rest sorted as -1)
    srt = np.sort(np.where(good, M, -1), axis=1)
    dup = (srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)
    neg = valid & (M < 0)
    last = np.where(good, pos[None, :], -1).max(axis=1, initial=-1)   # last valid position
    before = pos[None, :] < last[:, None]
    unm = neg & (M != OUT_OF_RANGE)
    unm = unm if mapped else unm & before
    oq_, dq_, uq_ = oor.any(axis=1), dup.any(axis=1), unm.any(axis=1)
    bad = dq_ | (valid & (M < -1)).any(axis=1) | (neg if mapped else neg & before).any(axis=1)
    ex = []
    for q in np.flatnonzero(oq_ | dq_ | uq_):    # examples in query order, at most 5
        if len(ex) >= 5:
            break
        q = int(q)
        if oq_[q]:
            ex.append({"query": q, "out_of_range_positions": [int(i) for i in np.flatnonzero(oor[q])[:5]]})
        if dq_[q] and len(ex) < 5:
            u, c = np.unique(M[q][good[q]], return_counts=True)
            ex.append({"query": q, "duplicate_ids": [int(i) for i in u[c > 1][:5]]})
        if uq_[q] and len(ex) < 5:
            ex.append({"query": q, "unmapped_positions": [int(i) for i in np.flatnonzero(unm[q])[:5]]})
    oq, dq, uq = int(oq_.sum()), int(dq_.sum()), int(uq_.sum())
    out = {"queries_with_duplicate_ids": dq, "duplicate_entries": int(dup.sum()),
           "queries_with_unmapped_entries": uq, "unmapped_entries": int(unm.sum()),
           "failed": bool(dq or uq or oq), "examples": ex}
    if oq:   # keys added only when present, so the usual document shape is unchanged
        out.update(queries_with_out_of_range_ids=oq, out_of_range_entries=int(oor.sum()))
    return out, bad


def _selected_queries(a, columns=None):
    """The queries table after the query-row selection (a.query_rows), in file order."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    qt = pq.read_table(a.queries, columns=columns)
    if a.query_rows:
        col, _, vals = a.query_rows.partition("=")
        want = set(vals.split(","))
        idx = [i for i, v in enumerate(qt.column(col).to_pylist()) if str(v) in want]
        qt = qt.take(pa.array(idx, pa.int64()))
    return qt


def ragged_query_ids(uri):
    """query_id of every row of the query files in sparse / multivector load_queries' order
    (sources.list_parquet: every *.parquet under ``uri``, sorted), or None without the column."""
    import pyarrow.parquet as pq
    from bench.data.sources import list_parquet
    fs, refs = list_parquet(uri)
    if "query_id" not in pq.read_schema(refs[0].path, filesystem=fs).names:
        return None
    return [str(x) for r in refs for x in
            pq.read_table(r.path, columns=["query_id"], filesystem=fs).column("query_id").to_pylist()]


def queries_qn(a) -> list:
    """query_id of the queries rows (after the query-row selection), else 0..n-1: the query
    order a run with no nova-bf leg is checked against."""
    if a.modality in ("sparse", "multivector"):
        qid = ragged_query_ids(a.queries)
        if qid is not None:
            return qid
        import pyarrow.parquet as pq
        from bench.data.sources import list_parquet
        fs, refs = list_parquet(a.queries)
        return [str(i) for i in range(sum(pq.read_metadata(r.path, filesystem=fs).num_rows
                                          for r in refs))]
    qt = _selected_queries(a)
    if "query_id" in qt.column_names:
        return [str(x) for x in qt.column("query_id").to_pylist()]
    return [str(i) for i in range(qt.num_rows)]


def dense_queries(a, n, qn):
    """The query vectors the results are for (after the query-row selection), in their order."""
    import pyarrow.compute as pc
    qt = _selected_queries(a)
    if "query_id" in qt.column_names and \
            [str(x) for x in qt.column("query_id").to_pylist()] != [str(x) for x in qn]:
        raise CheckError("--queries order != nova-bf query order")
    qa = qt.column(a.qcol).combine_chunks()
    xq = pc.list_flatten(qa).to_numpy(zero_copy_only=False).reshape(len(qa), -1)
    if len(xq) != n:
        raise CheckError(f"--queries has {len(xq)} rows, the results {n}")
    return xq


def ragged_queries(a, n, qn):
    """The sparse (canonical: duplicates summed) / multivector queries as the harness loads them
    (bench.data.formats load_queries), in the results' query order."""
    from bench.data.formats import multivector, sparse
    qid = ragged_query_ids(a.queries)
    if qid is not None and qid != [str(x) for x in qn]:
        raise CheckError("--queries order != nova-bf query order")
    q, _ = (sparse if a.modality == "sparse" else multivector).load_queries(a.queries, a.qcol)
    if q.n_rows != n:
        raise CheckError(f"--queries has {q.n_rows} rows, the results {n}")
    return q


def load_round(a: Args, legs: list[tuple[str, dict, str]], id_mode: str, cache_dir=None) -> dict:
    """The round's legs [(name, record, kind)], kind novabf | harness, as global row ids: any
    number of nova-bf systems (each id-mapped the same way: ``id_mode`` uuid | corpus) including
    none (then the query order is the queries file's, see queries_qn). -> {keys, offsets, n, qn,
    systems {name: (ids rows, score rows)}, mapped {name: True for nova-bf}}. ``cache_dir``:
    where the uuid table of the corpus is cached on disk (cached_uuid_table); None: built."""
    import pyarrow.compute as pc
    corpus_ids = id_mode == "corpus"
    has_nb = any(k == "novabf" for _, _, k in legs)
    if has_nb and not a.recursive:
        refuse_flat_with_nested(a.corpus)
    keys, rows, offsets, id_col = _cached_index(a.corpus, corpus_ids and has_nb, a.max_files,
                                                a.recursive)
    n_ids = int(offsets[-1])
    dup = 0 if id_col is None else n_ids - len(pc.unique(id_col))
    systems, mapped, qn, u2g = {}, {}, None, None
    for name, rec, kind in legs:
        if kind != "novabf":
            continue
        path = rec["_result"]
        if corpus_ids:
            if dup:
                raise CheckError(f"nova-bf id_column id: the corpus `id` column has {dup} duplicate "
                                 "ids, so a nova-bf hit id does not name one row")
            q_, ids, sc = load_novabf_ids(path, id_col)
        else:
            if u2g is None:
                u2g = _cached_uuids(a, keys, rows, offsets, cache_dir)
            q_, ids, sc = load_novabf(path, u2g)
        if qn is None:
            qn = q_
        elif [str(x) for x in q_] != [str(x) for x in qn]:
            raise CheckError(f"{name}: query order != {legs[0][0]}'s")
        systems[name] = (ids, sc)
        mapped[name] = True
    if qn is None:
        qn = queries_qn(a)
    for name, rec, kind in legs:
        if kind == "novabf":
            continue
        systems[name] = load_npz(rec["_topk"], n_ids)   # ids range-checked
        if len(systems[name][0]) != len(qn):
            raise CheckError(f"harness {name} ({rec['_topk']}) has {len(systems[name][0])} result "
                             f"rows, nova-bf {len(qn)}: not the same queries")
    return {"keys": keys, "offsets": offsets, "n": len(qn), "qn": qn, "systems": systems,
            "mapped": mapped}


# ---------------------------------------------------------------------------------------------
# the search's filter
# ---------------------------------------------------------------------------------------------
def filter_setup(exp: dict, search: dict, a: Args, n: int):
    """The harness's own filter evaluation for one search (FilterJob / DuckFilter from the
    nova-bf filter config + search, the same query-row selection) -> (job, built filter)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from bench.data.filters import duckdb_filter as dfm
    from bench.data.filters.reader import FilterJob
    from bench.data.filters.search import load_filter_config, query_row_index
    fdict, cdates, qdates, rows, _ = load_filter_config(exp["filter"]["config"],
                                                        search.get("filter_search") or search["name"])
    spec = dfm.parse_filter(fdict, cdates)
    qcols = spec.query_columns()
    qtable = None
    if spec.per_query:
        need_cols = sorted(set(qcols) | ({rows["column"]} if rows else set()))
        qt = pq.read_table(a.queries, columns=need_cols)
        idx = query_row_index(qt, rows)
        if idx is not None:
            qt = qt.take(pa.array(idx))
        if qt.num_rows != n:
            raise CheckError(f"filter: {qt.num_rows} selected query rows, the results {n}")
        qtable = qt.select(qcols)
    job = FilterJob(fdict, qtable, cdates, qdates)
    return job, job.build()


# ---------------------------------------------------------------------------------------------
# the e2e leg vs round 1
# ---------------------------------------------------------------------------------------------
def same_table(a: str, b: str) -> bool:
    """True when two nova-bf result parquets hold equal tables."""
    import pyarrow.parquet as pq
    return bool(pq.read_table(a).equals(pq.read_table(b)))


def same_scores(a: str, b: str) -> bool:
    """True when two nova-bf result parquets hold equal hit_scores, query by query. Ids are not
    compared: used when the two runs break ties differently (or name hits differently)."""
    import pyarrow.parquet as pq
    return bool(pq.read_table(a, columns=["hit_scores"]).equals(
        pq.read_table(b, columns=["hit_scores"])))


# ---------------------------------------------------------------------------------------------
# the checks' inputs: hashed, so a later `bench summarize` can refuse a stale checks.json
# ---------------------------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def input_files(run_dir: str) -> list[str]:
    """Every file the checks and summarize read: experiment.json, every file under legs/, and
    each nova-bf leg's result parquet(s) (counters.results). The run dir's own files are keyed
    relative to it (a synced or copied run dir still matches); a nova-bf result by the path its
    leg record names (absolute, or a URI), the file the checks read."""
    out = ["experiment.json"] if os.path.isfile(os.path.join(run_dir, "experiment.json")) else []
    d = os.path.join(run_dir, "legs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if os.path.isfile(os.path.join(d, f)) and not f.endswith(INPUT_SKIP):
            out.append(f"legs/{f}")
    for rec in records(run_dir).values():
        for p in (((rec or {}).get("counters") or {}).get("results") or {}).values() \
                if isinstance(rec, dict) else ():
            if isinstance(p, str):   # as the record names it: the file the checks read
                out.append(p)
    return sorted(set(out))


def file_id(run_dir: str, key: str) -> dict:
    """size + sha256 of a local file (relative = under the run dir); size + mtime of a URI."""
    if "://" in key:
        import pyarrow.fs as pafs
        fs, path = pafs.FileSystem.from_uri(key)
        info = fs.get_file_info(path)
        if info.type != pafs.FileType.File:
            return {"missing": True}
        return {"size": info.size, "mtime": str(info.mtime)}
    p = key if os.path.isabs(key) else os.path.join(run_dir, key)
    if not os.path.isfile(p):
        return {"missing": True}
    return {"size": os.path.getsize(p), "sha256": sha256(p)}


# ---------------------------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------------------------
def check_threads(exp: dict) -> int:
    """The checks' thread count: the experiment's `threads` (0 = every CPU). Not the
    process's OMP_NUM_THREADS, which SkyPilot sets to 1 in every job (the legs set their own)."""
    from bench.timing import omp_threads
    # threads, or for 0 / unset (an older experiment record) the CPUs the process may use
    return omp_threads({"threads": int(exp.get("threads") or 0)})


@contextlib.contextmanager
def threads(n: int):
    """Arrow's CPU / IO pools and the BLAS / OpenMP pools at ``n`` for the checks only, put back
    after (set at run time: the process's environment, which the legs inherit, is unchanged)."""
    import pyarrow as pa
    cpu, io_ = pa.cpu_count(), pa.io_thread_count()
    # threadpoolctl sets KMP_DUPLICATE_LIB_OK in os.environ when imported; legs started later
    # from this process would inherit it, so the environment is put back as it was
    kmp = os.environ.get("KMP_DUPLICATE_LIB_OK")
    try:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(n)
    except ImportError:   # no threadpoolctl: BLAS keeps its own count
        limits = contextlib.nullcontext()
    finally:
        if kmp is None:
            os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
        else:
            os.environ["KMP_DUPLICATE_LIB_OK"] = kmp
    try:
        pa.set_cpu_count(n)
        pa.set_io_thread_count(max(n, io_))
        with limits:
            yield
    finally:
        pa.set_cpu_count(cpu)
        pa.set_io_thread_count(io_)
