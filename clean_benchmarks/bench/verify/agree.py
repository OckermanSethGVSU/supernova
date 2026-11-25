"""Verify benchmark results across systems and rounds.

Runs three checks on saved results:

- Pairs: Compare every pair of systems, using float64 rescoring to
  resolve disagreements.
- Oracle: Compare each system against exact float64 results on a
  sample of queries.
- Rounds: Check consistency across rounds and against the cold e2e run.

Pairwise and oracle disagreements can fail verification. Differences
between regular rounds are warnings; e2e mismatches are failures.
Dataset ground truth is reference-only.

Records score errors and hashes the checked inputs so stale results
cannot be reused.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time

import numpy as np

from bench.verify import compare as C
from bench.verify import oracle as O

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPLEMENTED = True
ORACLE_QUERIES_DEFAULT = 500   # random queries in the oracle sample
ORACLE_SEED = 0
ORACLE_MAX_DIFF = 500          # plus up to this many queries where systems differ
SCORE_ATOL = 1e-5              # the harness --verify tolerance (oracle.compare_results)
ORACLE_HEAD = 3                # checks.json shows the oracle list of this many sampled queries
SCORE_RTOL = O.VERIFY_RTOL     # 1e-5
METRICS = {"cosine": "cosine", "ip": "ip"}   # compare.py --metric (ip = nova-bf's dot)


class CheckError(Exception):
    pass


# corpus_index / uuid_to_gid results, cached for one run_checks call (every round reads the same corpus)
_CACHE: dict = {}


def _corpus_index(uri, need_ids):
    key = ("index", uri, need_ids)
    if key not in _CACHE:
        _CACHE[key] = C.corpus_index(uri, need_ids=need_ids)
    return _CACHE[key]


def _uuid_to_gid(uri, keys, rows, offsets):
    key = ("u2g", uri)
    if key not in _CACHE:
        _CACHE[key] = C.uuid_to_gid(keys, rows, offsets)
    return _CACHE[key]


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------
def _abs(p: str) -> str:
    """A path as a leg resolves it: placeholders filled in, ~ expanded, relative = to the leg
    cwd (clean_benchmarks)."""
    from bench import paths
    p = paths.expand(p)
    if "://" in p:
        return p
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(CB, p))


def _records(run_dir: str) -> dict:
    out = {}
    d = os.path.join(run_dir, "legs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".json") and not f.endswith(".clock.json"):
            try:
                out[f[:-5]] = json.load(open(os.path.join(d, f)))
            except (OSError, ValueError):
                out[f[:-5]] = None
    return out


def _novabf_config(rec: dict) -> str | None:
    argv = rec.get("argv") or []
    return argv[argv.index("compute") + 1] if "compute" in argv else None


def _novabf_result(rec: dict, search: dict) -> str:
    res = (rec.get("counters") or {}).get("results") or {}
    if len(res) == 1:
        return next(iter(res.values()))
    for key in (search.get("filter_search"), search["name"]):
        if key in res:
            return res[key]
    raise CheckError(f"{rec.get('tag')}: {len(res)} nova-bf results {sorted(res)}, none for search "
                     f"{search['name']!r}")


def _yaml(path):
    import yaml
    return yaml.safe_load(open(path))


def _id_mode(cfg_path: str | None) -> str:
    """compare.py --novabf-ids: corpus when the nova-bf config has a corpus id_column."""
    if not cfg_path or not os.path.isfile(cfg_path):
        return "uuid"
    idc = ((_yaml(cfg_path) or {}).get("corpus") or {}).get("id_column")
    if idc is None:
        return "uuid"
    if idc != "id":
        raise CheckError(f"{cfg_path}: corpus id_column {idc!r}: compare.py maps corpus ids "
                         "through the `id` column only")
    return "corpus"


def _query_rows(exp: dict, search: dict) -> str | None:
    """filtered: compare.py --query-rows COLUMN=V1,V2 from the filter config's `rows` selector
    for this search."""
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


def _is_f16(rec: dict) -> bool:
    argv = rec.get("argv") or []
    return any(a == "--dtype" and i + 1 < len(argv) and argv[i + 1] == "f16" for i, a in enumerate(argv))


def compare_args(exp: dict, search: dict, novabf_path: str | None, id_mode: str) -> list[str]:
    """compare.py's argv for one round (the --harness list is added by the caller)."""
    if exp["search"]["metric"] not in METRICS:
        raise CheckError(f"search.metric {exp['search']['metric']!r}: compare.py scores cosine | ip")
    mod = "dense" if exp["modality"] == "filtered" else exp["modality"]
    argv = ["--corpus", _abs(exp["corpus"]["path"]), "--novabf", novabf_path or "-",
            "--modality", mod, "--novabf-ids", id_mode, "--k", str(exp["search"]["k"]),
            "--rescore", "--vcol", exp["corpus"]["column"], "--queries", _abs(exp["queries"]["path"]),
            "--qcol", exp["queries"]["column"], "--metric", METRICS[exp["search"]["metric"]]]
    gt = (exp.get("checks") or {}).get("gt")
    if gt:
        argv += ["--gt", _abs(gt)]
    rmg = (exp.get("checks") or {}).get("rescore_max_gib")
    if rmg is not None:   # the rescoring's memory cap (the pairs' and the oracle's)
        argv += ["--rescore-max-gib", f"{float(rmg):g}"]
    qr = _query_rows(exp, search)
    if qr:
        argv += ["--query-rows", qr]
    return argv


def _queries_qn(a) -> list:
    """query_id of the --queries rows (after --query-rows), else 0..n-1: the query order a
    run with no nova-bf leg is checked against."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if a.modality != "dense":
        qid = C.ragged_query_ids(a.queries)
        if qid is not None:
            return qid
        from bench.data.sources import list_parquet
        fs, refs = list_parquet(a.queries)
        return [str(i) for i in range(sum(pq.read_metadata(r.path, filesystem=fs).num_rows
                                          for r in refs))]
    qt = pq.read_table(a.queries)
    if a.query_rows:
        col, _, vals = a.query_rows.partition("=")
        want = set(vals.split(","))
        idx = [i for i, v in enumerate(qt.column(col).to_pylist()) if str(v) in want]
        qt = qt.take(pa.array(idx, pa.int64()))
    if "query_id" in qt.column_names:
        return [str(x) for x in qt.column("query_id").to_pylist()]
    return [str(i) for i in range(qt.num_rows)]


def load_round(a, legs: list[tuple[str, dict, str]], t0: float) -> dict:
    """Load the round's legs [(name, record, kind)], kind novabf | harness, as
    compare.load_inputs() does, for any number of nova-bf systems (each id-mapped the same way)
    including none (then the query order is the queries file's, see _queries_qn)."""
    import pyarrow.compute as pc
    gt_npz = bool(a.gt) and a.gt.endswith(".npz")
    corpus_ids = a.novabf_ids == "corpus"
    has_nb = any(k == "novabf" for _, _, k in legs)
    keys, rows, offsets, id_col = _corpus_index(
        a.corpus, (bool(a.gt) and not gt_npz) or (corpus_ids and has_nb))
    n_ids = int(offsets[-1])
    dup = 0 if id_col is None else n_ids - len(pc.unique(id_col))
    print(f"corpus: {len(keys)} files, {n_ids} rows, duplicate ids {dup} "
          f"({time.time() - t0:.0f} s)", flush=True)
    systems, mapped, qn, unmapped, u2g = {}, {}, None, 0, None
    for name, rec, kind in legs:
        if kind != "novabf":
            continue
        path = rec["_result"]
        if corpus_ids:
            if dup:
                raise CheckError(f"--novabf-ids corpus: the corpus `id` column has {dup} duplicate "
                                 "ids, so a nova-bf hit id does not name one row")
            q_, ids, sc = C.load_gt(path, id_col)
        else:
            if u2g is None:
                u2g = _uuid_to_gid(a.corpus, keys, rows, offsets)
                print(f"uuid map built ({time.time() - t0:.0f} s)", flush=True)
            q_, ids, sc = C.load_novabf(path, u2g)
        if qn is None:
            qn = q_
            unmapped = sum(int((x < 0).sum()) for x in ids)
        elif [str(x) for x in q_] != [str(x) for x in qn]:
            raise CheckError(f"{name}: query order != {legs[0][0]}'s")
        systems[name] = (ids, sc)
        mapped[name] = True
    if qn is None:
        qn = _queries_qn(a)
    gt_missing = None
    if a.gt:
        if "gt" in systems:
            raise CheckError("a system is named `gt`, the dataset GT's name")
        qg, gids, gsc = C.load_gt_npz(a.gt) if gt_npz else C.load_gt(a.gt, id_col)
        gt_missing = sum(int((x < 0).sum()) for x in gids)
        if [str(x) for x in qn] != [str(x) for x in qg]:
            raise CheckError("nova-bf query order != GT query order")
        systems["gt"] = (gids, gsc)
        mapped["gt"] = not gt_npz
    for name, rec, kind in legs:
        if kind == "novabf":
            continue
        systems[name] = C.load_npz(rec["_topk"], n_ids)   # ids range-checked
        if len(systems[name][0]) != len(qn):
            raise CheckError(f"harness {name} ({rec['_topk']}) has {len(systems[name][0])} result "
                             f"rows, nova-bf {len(qn)}: not the same queries")
    return {"keys": keys, "offsets": offsets, "n_ids": n_ids, "dup": dup, "qn": qn,
            "systems": systems, "n": len(qn), "unmapped": unmapped, "gt_missing": gt_missing,
            "mapped": mapped}


# ---------------------------------------------------------------------------------------------
# 2. the float64 oracle on a query sample
# ---------------------------------------------------------------------------------------------
def oracle_sample(systems: dict, n: int, kmax: int, n_random: int) -> tuple[list[int], dict]:
    """The oracle sample: n_random seeded random queries + the first ORACLE_MAX_DIFF whose
    top-k id sets differ between any two systems."""
    names = [x for x in systems if x != "gt"]
    diff = []
    for q in range(n):
        sets = [frozenset(int(i) for i in np.asarray(systems[nm][0][q][:kmax]) if i >= 0) for nm in names]
        if len(set(sets)) > 1:
            diff.append(q)
    rng = np.random.default_rng(ORACLE_SEED)
    sample = sorted(set(rng.choice(n, size=min(n_random, n), replace=False).tolist())
                    | set(diff[:ORACLE_MAX_DIFF])) if n else []
    return sample, {"random": min(n_random, n), "differing": len(diff),
                    "differing_included": min(len(diff), ORACLE_MAX_DIFF), "seed": ORACLE_SEED}


def _exact_for(a, need, inputs):
    """compare.py's float64 rescoring of ``need`` ({query: gids}), the same functions and memory
    refusal as --rescore (dense: exact_scores; sparse / multivector: rescore_ragged's body)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    offsets, n_keys, n, qn = inputs["offsets"], len(inputs["keys"]), inputs["n"], inputs["qn"]
    fs, paths = C.corpus_paths(a.corpus, n_keys)
    if a.modality != "dense":
        sparse, multivec = C.harness_modules()
        mod = sparse if a.modality == "sparse" else multivec
        q, _ = mod.load_queries(a.queries, a.qcol)
        if q.n_rows != n:
            raise CheckError(f"--queries has {q.n_rows} rows, the results {n}")
        fn = C.exact_scores_sparse if a.modality == "sparse" else C.exact_scores_mv
        return fn(need, offsets, paths, fs, a.vcol, q, a.metric, threads=a.rescore_threads)
    xq = _dense_queries(a, n, qn)
    est, _, _ = C.rescore_bytes(need, offsets, xq.shape[1],
                                C.vector_itemsize(paths[0], fs, a.vcol), a.rescore_threads)
    if est > a.rescore_max_gib * 2**30:
        raise CheckError(f"oracle rescoring refused: estimated {est / 2**30:.2f} GiB > "
                         f"{a.rescore_max_gib:g} GiB")
    return C.exact_scores(need, a.corpus, a.vcol, offsets, paths, fs, xq, a.metric,
                          threads=a.rescore_threads)


def _dense_queries(a, n, qn):
    """The dense --queries rows the results are for (after --query-rows), in their order."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    qt = pq.read_table(a.queries)
    if a.query_rows:
        col, _, vals = a.query_rows.partition("=")
        want = set(vals.split(","))
        idx = [i for i, v in enumerate(qt.column(col).to_pylist()) if str(v) in want]
        qt = qt.take(pa.array(idx, pa.int64()))
    if "query_id" in qt.column_names and \
            [str(x) for x in qt.column("query_id").to_pylist()] != [str(x) for x in qn]:
        raise CheckError("--queries order != nova-bf query order")
    qa = qt.column(a.qcol).combine_chunks()
    xq = pc.list_flatten(qa).to_numpy(zero_copy_only=False).reshape(len(qa), -1)
    if len(xq) != n:
        raise CheckError(f"--queries has {len(xq)} rows, the results {n}")
    return xq


# `checks.exact_scan: true` scans every row of the run's files for the sampled queries, keeping a
# running float64 top-k per query. Dense (unfiltered): oracle.py's float64 reference per file.
# Sparse / multivector / filtered: one file in memory at a time, scored in chunks of at most
# EXACT_SCAN_CELLS (query, row) cells with compare.py's float64 arithmetic (filtered: the file's
# filter mask applied first); each chunk's top-k merged in (best first, ties to the lowest id).
EXACT_SCAN_CELLS = 1 << 23     # 64 MiB of float64 scores per chunk


def scan_rows(inputs, scan_files) -> int | None:
    """Rows of the files the run read: the first ``scan_files`` (a subset) or all; None when
    the inputs carry no corpus index."""
    off = inputs.get("offsets")
    if off is None:
        return None
    nf = len(off) - 1 if scan_files is None else min(int(scan_files), len(off) - 1)
    return int(off[nf])


def _chunk_topk(s, gids, k):
    """(nq, r) float64 scores of rows ``gids`` -> their (nq, k) top-k (best first, ties to the
    lowest id; -inf and NaN are never in it: padded -inf / -1)."""
    nq, r = s.shape
    D = np.full((nq, k), -np.inf, np.float64)
    I = np.full((nq, k), -1, np.int64)
    if not r:
        return D, I
    s = np.where(np.isnan(s), -np.inf, s)
    kth = np.partition(s, r - k, axis=1)[:, r - k] if r > k else np.full(nq, -np.inf)
    for j in range(nq):
        c = np.flatnonzero(np.isfinite(s[j]) & (s[j] >= kth[j]))
        o = np.lexsort((gids[c], -s[j, c]))[:k]
        D[j, :len(o)] = s[j, c[o]]
        I[j, :len(o)] = gids[c[o]]
    return D, I


def _scan_scorer(a, inputs, sample):
    """file table -> a generator of (row0, (len(sample), rows) float64 scores) chunks, the same
    float64 arithmetic as compare.exact_scores / _sparse / _mv (dense covers filtered)."""
    nq = len(sample)
    step = max(1, EXACT_SCAN_CELLS // max(1, nq))
    cos = a.metric == "cosine"
    if a.modality in ("dense", "filtered"):
        import pyarrow.compute as pc
        q64 = np.asarray(_dense_queries(a, inputs["n"], inputs["qn"])[np.asarray(sample, np.int64)],
                         np.float64)
        if cos:
            nn = np.linalg.norm(q64, axis=1, keepdims=True)
            q64 = np.divide(q64, nn, out=np.zeros_like(q64), where=nn > 0)

        def score(t):
            col = t.column(a.vcol).combine_chunks()
            x = pc.list_flatten(col).to_numpy(zero_copy_only=False).reshape(len(col), -1)
            if len(col) and x.shape[1] != q64.shape[1]:
                raise CheckError(f"exact_scan: corpus dim {x.shape[1]}, queries {q64.shape[1]}")
            for r0 in range(0, len(col), step):
                v = x[r0:r0 + step].astype(np.float64)
                if cos:
                    nv = np.linalg.norm(v, axis=1, keepdims=True)
                    v = np.divide(v, nv, out=np.zeros_like(v), where=nv > 0)
                yield r0, q64 @ v.T
        return score
    sparse, multivec = C.harness_modules()
    if a.modality == "sparse":
        import scipy.sparse as sp
        q, _ = sparse.load_queries(a.queries, a.qcol)
        if q.n_rows != inputs["n"]:
            raise CheckError(f"--queries has {q.n_rows} rows, the results {inputs['n']}")

        def score(t):
            c = sparse.unpack(sparse.table_csr(t, a.vcol))
            width = max(q.n_cols, c.n_cols, 1)
            X = sparse.block_matrix(c, width, np.float64)
            X.sum_duplicates()
            Q = sp.csr_matrix((q.values.astype(np.float64), q.indices, q.indptr),
                              shape=(q.n_rows, width))[np.asarray(sample, np.int64)]
            qn = sparse.row_norms(Q) if cos else None
            xn = sparse.row_norms(X) if cos else None
            for r0 in range(0, c.n_rows, step):
                s = np.asarray((Q @ X[r0:r0 + step].T).todense(), np.float64)
                if cos:   # each side divided by its own norm iff nonzero (Rows.pair_dot)
                    np.divide(s, qn[:, None], out=s, where=qn[:, None] > 0)
                    np.divide(s, xn[None, r0:r0 + step], out=s, where=xn[None, r0:r0 + step] > 0)
                yield r0, s
        return score
    q, _ = multivec.load_queries(a.queries, a.qcol)
    if q.n_rows != inputs["n"]:
        raise CheckError(f"--queries has {q.n_rows} rows, the results {inputs['n']}")
    q64 = multivec.unit_tokens(q.tokens, np.float64) if cos else q.tokens.astype(np.float64)
    sel = np.asarray(sample, np.int64)
    q_len = q.lengths[sel]
    q_off = np.zeros(nq + 1, np.int64)
    np.cumsum(q_len, out=q_off[1:])
    q_tok = np.concatenate([q64[q.offsets[x]:q.offsets[x + 1]] for x in sel]) if q_off[-1] \
        else np.zeros((0, q64.shape[1]), np.float64)

    def score(t):
        r = multivec.unpack(multivec.table_ragged(t, a.vcol))
        if r.n_tok and q_tok.shape[0] and r.dim != q_tok.shape[1]:
            raise CheckError(f"exact_scan: corpus token width {r.dim}, queries {q_tok.shape[1]}")
        for r0 in range(0, r.n_rows, step):
            r1 = min(r.n_rows, r0 + step)
            lo, hi = int(r.offsets[r0]), int(r.offsets[r1])
            d = r.tokens[lo:hi].astype(np.float64)
            if cos:
                d = multivec.unit_tokens(d, np.float64)
            # a zero-token document scores -inf (a non-candidate), as exact_scores_mv
            yield r0, multivec.maxsim_matrix(q_off, q_tok, np.asarray(r.offsets[r0:r1 + 1], np.int64) - lo,
                                             d, np.float64)
    return score


def exact_scan_candidates(a, inputs, sample, scan_files, filtered=False, mask_fn=None) -> dict:
    """{sampled query: candidate gids} of a full float64 scan of the run's files: the
    exact top-k of every scanned row, one file at a time. A filtered search (``filtered``) takes
    ``mask_fn(f, sample) -> bool (len(sample) | 1, rows)``, the search's filter on file f's rows
    (filter_scan_mask), applied BEFORE the top-k: a row the filter removes is never a
    candidate."""
    rows = scan_rows(inputs, scan_files)
    if rows is None:
        raise CheckError("exact_scan: no corpus index")
    if filtered and mask_fn is None:
        raise CheckError("exact_scan: a filtered search needs its filter mask (mask_fn)")
    import pyarrow.parquet as pq
    fs, paths = C.corpus_paths(a.corpus, len(inputs["keys"]))
    nf = len(paths) if scan_files is None else min(int(scan_files), len(paths))
    off = np.asarray(inputs["offsets"], np.int64)
    D = np.full((len(sample), a.k), -np.inf, np.float64)
    I = np.full((len(sample), a.k), -1, np.int64)
    if a.modality == "dense" and not filtered:
        import pyarrow.compute as pc
        xq = _dense_queries(a, inputs["n"], inputs["qn"])
        # per file: oracle.py's float64 reference (same row tiles), merged with the same tie
        # rule, so the ids equal a one-shot reference over all files
        unit = a.metric == "cosine"
        rt = O.oracle_row_tile(xq.shape[1], int(np.diff(off[:nf + 1]).max()) if nf else 1)
        xq64 = O.f64_rows(xq[np.asarray(sample, np.int64)], unit)
        for f, p in enumerate(paths[:nf]):
            col = pq.read_table(p, columns=[a.vcol], filesystem=fs).column(a.vcol).combine_chunks()
            x = pc.list_flatten(col).to_numpy(zero_copy_only=False).reshape(len(col), -1)
            del col
            Df, If = O.exact_topk_f64(xq64, [x], a.k, unit, r_tile=rt)
            del x
            D, I = O._merge_topk(D, I, Df, np.where(If >= 0, If + off[f], -1), a.k)
        return {q: {int(g) for g in I[j] if g >= 0} for j, q in enumerate(sample)}
    score = _scan_scorer(a if not filtered else _with(a, modality="filtered"), inputs, sample)
    for f, p in enumerate(paths[:nf]):
        t = pq.read_table(p, columns=[a.vcol], filesystem=fs)
        n_f = int(off[f + 1] - off[f])
        if t.num_rows != n_f:
            raise CheckError(f"exact_scan: {p} has {t.num_rows} rows, the corpus index {n_f}")
        m = np.asarray(mask_fn(f, sample), bool) if filtered else None
        if m is not None and m.shape[-1] != n_f:
            raise CheckError(f"exact_scan: the filter mask of {p} covers {m.shape[-1]} rows, not {n_f}")
        for r0, s in score(t):
            r1 = r0 + s.shape[1]
            if m is not None:
                s = np.where(m[:, r0:r1], s, -np.inf)
            Df, If = _chunk_topk(s, np.arange(r0, r1, dtype=np.int64) + off[f], a.k)
            D, I = O._merge_topk(D, I, Df, If, a.k)
        del t
    return {q: {int(g) for g, d in zip(I[j], D[j]) if g >= 0 and np.isfinite(d)}
            for j, q in enumerate(sample)}


def _with(a, **kw):
    import types
    return types.SimpleNamespace(**{**vars(a), **kw})


def _filter_setup(exp: dict, search: dict, a, inputs):
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
        if qt.num_rows != inputs["n"]:
            raise CheckError(f"filter: {qt.num_rows} selected query rows, the results {inputs['n']}")
        qtable = qt.select(qcols)
    job = FilterJob(fdict, qtable, cdates, qdates)
    return job, job.build()


def filter_scan_mask(exp: dict, search: dict, a, inputs):
    """-> mask_fn(f, sample) for exact_scan_candidates: file f's rows that pass the
    search's filter, per sampled query (len(sample), rows) or static (1, rows), by the same
    filter evaluation as filter_allowed (built once per call of this function)."""
    import pyarrow.parquet as pq
    from bench.data.filters import duckdb_filter as dfm
    job, flt = _filter_setup(exp, search, a, inputs)
    fs, paths = C.corpus_paths(a.corpus, len(inputs["keys"]))

    def mask_fn(f, sample):
        res = flt.evaluate(pq.read_table(paths[f], columns=job.fields, filesystem=fs))
        if res.per_query:
            return dfm.unpack(res.packed_per_query()[np.asarray(sample, np.int64)], res.n_rows)
        return np.asarray(res.row_mask(), bool)[None, :]
    return mask_fn


def filter_allowed(exp: dict, search: dict, a, inputs, need) -> dict:
    """{(q, gid): passes the search's filter} for every candidate in ``need``, by the harness's
    own filter evaluation (the FilterJob / DuckFilter its readers build from the nova-bf filter
    config). Reads only the filter columns of files that hold a candidate."""
    import pyarrow.parquet as pq
    from bench.data.filters import duckdb_filter as dfm
    job, flt = _filter_setup(exp, search, a, inputs)
    offsets = np.asarray(inputs["offsets"], np.int64)
    fs, paths = C.corpus_paths(a.corpus, len(inputs["keys"]))
    by_file: dict = {}
    for q, gs in enumerate(need):
        for g in gs:
            f = int(np.searchsorted(offsets, g, side="right") - 1)
            by_file.setdefault(f, []).append((q, g))
    out = {}
    for f, pairs in sorted(by_file.items()):
        t = pq.read_table(paths[f], columns=job.fields, filesystem=fs)
        res = flt.evaluate(t)
        n_rows = res.n_rows
        if res.per_query:
            packed = res.packed_per_query()
            for q, g in pairs:
                r = g - int(offsets[f])
                out[(q, g)] = bool(dfm.unpack(packed[q:q + 1], n_rows)[0][r])
        else:
            m = res.row_mask()
            for q, g in pairs:
                out[(q, g)] = bool(m[g - int(offsets[f])])
    return out


def mask_excluded(exp: dict, search: dict, a, inputs, exact: dict) -> int:
    """The pairs' float64 table of a filtered search: every (query, row) the search's filter
    excludes (filter_allowed) scored -inf, in place, so compare.judge blames the system that
    returned it (as the oracle does). -> how many pairs were set."""
    need = [set() for _ in range(inputs["n"])]
    for (q, g) in exact:
        need[q].add(int(g))
    n = 0
    for (q, g), ok in filter_allowed(exp, search, a, inputs, need).items():
        if not ok:
            exact[(q, g)] = -np.inf
            n += 1
    return n


SCORE_ERROR_KEYS = ("max_ulps", "max_abs", "max_ulps_adjudicated")


def round_score_error(out: dict) -> dict:
    """{system: {max_ulps, max_abs, max_ulps_adjudicated}} of one round, each system's own
    score error against float64 (float32 ULPs of the exact score; absolute):
      max_ulps / max_abs     over every score measured: each reported score of the oracle
                             sample, and the pairs' adjudicated ids;
      max_ulps_adjudicated   over the ids the tie window judges only (strays, k-th ids,
                             mismatched ranks; pairs and oracle), compare.judge's own
                             "own error" (its WARNING rule).
    A ULP count is relative to the score, so a score near 0 shows many ULPs at an absolute
    error of ~1e-7; max_abs says which it is."""
    err: dict = {}

    def put(nm, u=None, ab=None, adj=None):
        if nm == "gt":
            return
        e = err.setdefault(nm, dict.fromkeys(SCORE_ERROR_KEYS, 0.0))
        for key, v in (("max_ulps", u), ("max_abs", ab), ("max_ulps_adjudicated", adj)):
            if v is not None:
                e[key] = max(e[key], float(v))
        if adj is not None:
            e["max_ulps"] = max(e["max_ulps"], float(adj))
    pr = (out.get("pairs") or {}).get("compare") or {}
    for nm, u in (pr.get("f64_max_error_ulps") or {}).items():
        put(nm, ab=(pr.get("f64_max_error_abs") or {}).get(nm), adj=u)
    for nm, v in ((out.get("oracle") or {}).get("systems") or {}).items():
        oc = v.get("compare") or {}
        put(nm, v.get("max_score_error_ulps"), v.get("max_score_error_abs"),
            (oc.get("f64_max_error_ulps") or {}).get(nm))
    return err


# ---------------------------------------------------------------------------------------------
# the checks' inputs: hashed, so a later `bench summarize` can refuse a stale checks.json
# ---------------------------------------------------------------------------------------------
_INPUT_SKIP = (".tmp", ".pending")


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _input_files(run_dir: str) -> list[str]:
    """Every file the checks and summarize read: experiment.json, every file under legs/, and
    each nova-bf leg's result parquet(s) (counters.results). The run dir's own files are keyed
    relative to it (a synced or copied run dir still matches); a nova-bf result by the path its
    leg record names (absolute, or a URI), the file the checks read."""
    out = ["experiment.json"] if os.path.isfile(os.path.join(run_dir, "experiment.json")) else []
    d = os.path.join(run_dir, "legs")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if os.path.isfile(os.path.join(d, f)) and not f.endswith(_INPUT_SKIP):
            out.append(f"legs/{f}")
    for rec in _records(run_dir).values():
        for p in (((rec or {}).get("counters") or {}).get("results") or {}).values() \
                if isinstance(rec, dict) else ():
            if isinstance(p, str):   # as the record names it: the file the checks read
                out.append(p)
    return sorted(set(out))


def _file_id(run_dir: str, key: str) -> dict:
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
    return {"size": os.path.getsize(p), "sha256": _sha256(p)}


def input_hashes(run_dir: str) -> dict:
    """{file: {size, sha256}} of every file _input_files names, as they are now."""
    return {k: _file_id(run_dir, k) for k in _input_files(run_dir)}


def input_problems(run_dir: str, doc: dict) -> list[str]:
    """Why a checks.json no longer describes the run dir: no `inputs` record, a recorded file
    changed or gone, or a file the checks would now read that it did not record."""
    rec = doc.get("inputs")
    if not isinstance(rec, dict) or not rec:
        return ["checks.json records no inputs (written before input hashing, or by hand): "
                "re-run the checks"]
    # checks.json keys hold private paths in placeholder form (bench/paths.py)
    from bench import paths
    now = {paths.mask(k): k for k in _input_files(run_dir)}
    out = []
    for k in sorted(set(rec) | set(now)):
        if k not in rec:
            out.append(f"{k}: not read by the checks (added after them)")
            continue
        cur = _file_id(run_dir, now.get(k) or paths.expand(k))
        if cur.get("missing"):
            out.append(f"{k}: missing (the checks read it)")
        elif cur != rec[k]:
            out.append(f"{k}: changed since the checks read it")
    return out


def short_queries(systems: dict, under_test: list, n: int, k: int, rows: int | None) -> tuple:
    """the queries where EVERY system returned fewer than min(k, rows) hits
    (rows unknown: k). -> (want, [queries])."""
    want = k if rows is None else min(k, rows)
    out = []
    for q in range(n):
        if under_test and all(int((np.asarray(systems[nm][0][q][:k]) >= 0).sum()) < want
                              for nm in under_test):
            out.append(q)
    return want, out


def _finite_hits(systems, nm, q, k) -> int:
    ids = np.asarray(systems[nm][0][q][:k])
    sc = systems[nm][1][q] if systems[nm][1] is not None else None
    ok = ids >= 0
    if sc is not None:
        sc = np.asarray(sc[:k], np.float64)
        ok = ok & np.isfinite(sc[:len(ok)])
    return int(ok.sum())


def mv_short_queries(a, inputs, systems: dict, under_test: list, n: int, k: int,
                     scan_files) -> tuple:
    """The multivector fewer-hits guard. A document is a candidate when it scores finite (it
    and the query both have tokens); a system's hits are entries with an id and a finite score.
    A query fails when every system returned fewer than min(k, documents with tokens) (0 for a
    zero-token query). Token counts are read only if some query is short against min(k, rows).
    -> (want, [queries])."""
    rows = scan_rows(inputs, scan_files)
    bound = k if rows is None else min(k, rows)
    sus = [q for q in range(n) if under_test and
           all(_finite_hits(systems, nm, q, k) < bound for nm in under_test)]
    if not sus:
        return bound, []
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from bench.data.formats import multivector as multivec
    fs, paths = C.corpus_paths(a.corpus, len(inputs["keys"]))
    nf = len(paths) if scan_files is None else min(int(scan_files), len(paths))
    docs = 0
    for p in paths[:nf]:
        col = pq.read_table(p, columns=[a.vcol], filesystem=fs).column(a.vcol).combine_chunks()
        docs += int(pc.sum(pc.greater(pc.fill_null(pc.list_value_length(col), 0), 0)).as_py() or 0)
        del col
    try:
        qlen = multivec.load_queries(a.queries, a.qcol)[0].lengths
        qlen = qlen if len(qlen) == n else None
    except Exception:  # noqa: BLE001  unknown query lengths: every query counts as having tokens
        qlen = None
    want = min(k, docs)
    out = [q for q in sus if (want if qlen is None or qlen[q] > 0 else 0) > 0 and
           all(_finite_hits(systems, nm, q, k) < want for nm in under_test)]
    return want, out


def oracle_check(a, inputs, n_random: int, allowed_fn=None, exact_scan: bool = False,
                 scan_files: int | None = None, short_guard: bool | None = None,
                 scan_mask_fn=None) -> dict:
    """Check every system against the float64 oracle on a query sample.

    ``allowed_fn(need) -> {(q, gid): bool}`` (filtered): a candidate failing the filter scores
    -inf and is never in the oracle list, so the system that returned it is blamed.
    ``exact_scan``: candidates also include a full float64 scan (first ``scan_files`` files,
    else all), so a row every system missed still reaches the oracle list; otherwise the
    candidates are the union of the systems' lists. ``short_guard``: a query where every system
    returned fewer than min(k, rows) hits fails every system."""
    systems, n, k = inputs["systems"], inputs["n"], a.k
    under_test = [x for x in systems if x != "gt"]
    sample, info = oracle_sample(systems, n, k, n_random)
    need = [set() for _ in range(n)]
    for q in sample:
        for nm in under_test:
            A = np.asarray(systems[nm][0][q][:k])
            need[q].update(int(i) for i in A[A >= 0])
    scanned = None
    if exact_scan and sample:
        kw = {"mask_fn": scan_mask_fn} if scan_mask_fn is not None else {}
        scanned = exact_scan_candidates(a, inputs, sample, scan_files, allowed_fn is not None, **kw)
        for q, gs in scanned.items():
            need[q].update(gs)
    exact = _exact_for(a, need, inputs) if sample else {}
    masked = 0
    if allowed_fn is not None and sample:
        ok = allowed_fn(need)
        for (q, g), v in ok.items():
            if not v:
                exact[(q, g)] = -np.inf   # filtered out: not a candidate, the system is blamed
                masked += 1
    # oracle list per sampled query: exact top-k over the sorted candidate union; a stable
    # sort puts the lowest id first among equal scores
    o_ids, o_sc, ex_j = [], [], {}
    for j, q in enumerate(sample):
        cand = np.asarray(sorted(need[q]), np.int64)
        s = np.asarray([exact[(q, int(g))] for g in cand], np.float64)
        keep = np.isfinite(s)        # -inf non-candidates and NaN are never in the oracle's list
        cand, s = cand[keep], s[keep]
        order = np.argsort(-s, kind="stable")[:k]
        o_ids.append(cand[order])
        o_sc.append(s[order])
        for g in need[q]:
            ex_j[(j, g)] = exact[(q, g)]
    if short_guard is None:
        short_guard = a.modality in ("dense", "multivector")
    rows = scan_rows(inputs, scan_files)
    if short_guard and a.modality == "multivector":   # finite-score documents
        want, short = mv_short_queries(a, inputs, systems, under_test, n, k, scan_files)
    else:
        want, short = short_queries(systems, under_test, n, k, rows) if short_guard else (None, [])
    out = {"sample": {"queries": len(sample), **info}, "systems": {},
           "exact_scan": {"on": bool(exact_scan), "rows": rows if exact_scan else None,
                          "candidates": sum(len(v) for v in scanned.values()) if scanned else None},
           "short_guard": {"on": bool(short_guard), "want": want, "queries": short[:20],
                           "n": len(short)},
           "filter_masked_candidates": masked if allowed_fn is not None else None,
           # the oracle's own list for the first few sampled queries (its tie rule is visible)
           "oracle_lists_head": {str(int(sample[j])): [int(g) for g in o_ids[j]]
                                 for j in range(min(ORACLE_HEAD, len(sample)))}}
    if not sample:
        if short:
            for nm in under_test:
                out["systems"][nm] = {"ok": False, "fails": [_short_fail(short, want)]}
        return out
    for nm in under_test:
        ids = [systems[nm][0][q] for q in sample]
        sc = [systems[nm][1][q] for q in sample]
        defects = C.list_defects(ids, k, inputs["mapped"].get(nm, False))
        r = C.compare((ids, sc), (o_ids, o_sc), len(sample), a.tol, k, exact=ex_j,
                      names=(nm, "oracle"), tie_ulps=a.tie_ulps, modality=a.modality, order=True)
        # reported scores vs exact: the harness --verify per-entry tolerance
        bad, worst, worst_abs, ex = 0, 0.0, 0.0, []
        for j in range(len(sample)):
            A, S = C.valid(ids[j], sc[j], k)
            for g, s_ in zip(A.tolist(), S.tolist()):
                e = ex_j[(j, g)]
                if not np.isfinite(e):
                    continue          # a -inf non-candidate: judged by the blame rule above
                err = abs(s_ - e)
                worst = max(worst, err / C.ulps32(e))
                worst_abs = max(worst_abs, err)
                if not err <= SCORE_ATOL + SCORE_RTOL * abs(e):
                    bad += 1
                    if len(ex) < 5:
                        ex.append({"query": int(sample[j]), "id": int(g), "score": s_, "exact": e})
        fails = []
        if defects["failed"]:
            fails.append("list defects (out-of-range ids)" if defects.get("out_of_range_entries")
                         and not (defects["queries_with_duplicate_ids"]
                                  or defects["queries_with_unmapped_entries"])
                         else "list defects (duplicate or unmapped ids)"
                         + (" + out-of-range ids" if defects.get("out_of_range_entries") else ""))
        if r["f64_wrong_by_system"][nm]:
            fails.append(f"{r['f64_wrong_by_system'][nm]} float64 disagreements with the oracle "
                         f"outside the {a.tie_ulps:g}-ULP tie window")
        if r["f64_wrong_by_system"]["oracle"]:
            fails.append(f"{r['f64_wrong_by_system']['oracle']} disagreements blamed on the oracle "
                         "list (a candidate it could not rank)")
        if r["f64_order_real_disagreements"]:
            fails.append(f"{r['f64_order_real_disagreements']} float64 ORDER disagreements")
        if bad:
            fails.append(f"{bad} scores off their float64 score by more than "
                         f"{SCORE_ATOL:g} + {SCORE_RTOL:g} x |exact|")
        if short:
            fails.append(_short_fail(short, want))
        out["systems"][nm] = {"ok": not fails, "fails": fails, "defects": defects,
                              "scores_out_of_tolerance": bad, "max_score_error_ulps": worst,
                              "max_score_error_abs": worst_abs,
                              "score_examples": ex, "compare": r}
    return out


def _short_fail(short, want):
    return (f"{len(short)} queries where every system returned fewer than {want} hits "
            f"(min(k, corpus rows); multivector: documents with tokens, finite scores), "
            f"e.g. query {short[0]}")


# ---------------------------------------------------------------------------------------------
# 3. rounds
# ---------------------------------------------------------------------------------------------
def same_topk(ref: str, other: str) -> bool:
    """True when two saved top-k files have array-equal I and D."""
    a, b = np.load(ref), np.load(other)
    return bool(np.array_equal(a["I"], b["I"]) and np.array_equal(a["D"], b["D"]))


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
# one round
# ---------------------------------------------------------------------------------------------
def _usable(run_dir, recs, tag, search, skipped):
    rec = recs.get(tag)
    if rec is None:
        skipped.append({"leg": tag, "why": "no readable leg record"})
        return None
    if not rec.get("ok"):
        skipped.append({"leg": tag, "why": "leg failed (summary lists it)"})
        return None
    rec = dict(rec)
    if rec.get("kind") == "novabf":
        rec["_result"] = _novabf_result(rec, search)
        rec["_config"] = _novabf_config(rec)
    else:
        if _is_f16(rec):
            skipped.append({"leg": tag, "why": "fp16 scores, inexact by design"})
            return None
        rec["_topk"] = os.path.join(run_dir, rec["topk"]) if rec.get("topk") else None
        if not rec["_topk"] or not os.path.isfile(rec["_topk"]):
            raise CheckError(f"{tag}: no saved top-k")
    return rec


def sparse_weight_problems(a, inputs) -> list[str]:
    """nova-bf's sparse candidate rule assumes non-negative weights (a row sharing no term with
    a query scores 0, never above a positive one). Checks every query and only the FIRST corpus
    file (a cheap sample; the whole corpus would cost a full pass). -> problems (empty = ok).
    Cached per run."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    key = ("sparse_weights", a.corpus, a.queries, a.vcol, a.qcol)
    if key in _CACHE:
        return _CACHE[key]
    out = []
    sparse, _ = C.harness_modules()
    q, _ = sparse.load_queries(a.queries, a.qcol)
    if q.values.size and float(np.min(q.values)) < 0:
        out.append(f"{int((q.values < 0).sum())} negative query weight(s) (min {float(np.min(q.values))!r}) "
                   f"in {a.queries}: the candidate rule assumes non-negative weights")
    fs, paths = C.corpus_paths(a.corpus, len(inputs["keys"]))
    if paths:
        t = pq.read_table(paths[0], columns=[a.vcol], filesystem=fs)
        v = pc.list_flatten(pc.struct_field(t.column(a.vcol).combine_chunks(), "values"))
        mn = pc.min(v).as_py() if len(v) else None
        if mn is not None and mn < 0:
            out.append(f"{pc.sum(pc.less(v, 0)).as_py()} negative corpus weight(s) (min {mn!r}) in "
                       f"{paths[0]} (the first corpus file): the candidate rule assumes "
                       "non-negative weights")
    _CACHE[key] = out
    return out


def check_round(exp, run_dir, recs, search, r, skipped, log_dir) -> dict:
    from bench import timing
    legs = []
    for sy in search["systems"]:
        tag = timing.tag(search["name"], sy, r)
        rec = _usable(run_dir, recs, tag, search, skipped)
        if rec is not None:
            legs.append((sy, rec, "novabf" if rec.get("kind") == "novabf" else "harness"))
    nb = [x for x in legs if x[2] == "novabf"]
    id_mode = _id_mode(nb[0][1]["_config"]) if nb else "uuid"
    for _, rec, _ in nb[1:]:
        if _id_mode(rec["_config"]) != id_mode:
            raise CheckError("nova-bf systems of one search disagree on id_column")
    argv = compare_args(exp, search, nb[0][1]["_result"] if nb else None, id_mode)
    harness_argv = [f"{x[0]}={x[1]['_topk']}" for x in legs if x[2] == "harness"]
    a = C.parse_args(argv)
    out = {"systems": [x[0] for x in legs],
           "compare_argv": argv + [z for h in harness_argv for z in ("--harness", h)]}
    log = os.path.join(log_dir, f"{search['name']}__r{r}.log")
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            inputs = load_round(a, legs, t0)
            hook = None
            if exp["modality"] == "filtered":
                def hook(exact, _s=search, _a=a, _i=inputs):
                    return mask_excluded(exp, _s, _a, _i, exact)
            res, fails = C.judge(a, inputs, t0, exact_hook=hook)
            out["pairs"] = {"ok": not fails, "fails": fails, "compare": res}
        except (CheckError, SystemExit) as exc:
            out["pairs"] = {"ok": False, "fails": [f"compare could not run: {exc}"], "compare": None}
            inputs = None
        try:
            if inputs is not None:
                n_random = (exp.get("checks") or {}).get("oracle_queries") or ORACLE_QUERIES_DEFAULT
                allowed_fn = scan_mask_fn = None
                ck = exp.get("checks") or {}
                if exp["modality"] == "filtered":
                    def allowed_fn(need, _s=search, _a=a, _i=inputs):
                        return filter_allowed(exp, _s, _a, _i, need)
                    if ck.get("exact_scan"):   # the scan applies the filter per file
                        scan_mask_fn = filter_scan_mask(exp, search, a, inputs)
                o = oracle_check(a, inputs, int(n_random), allowed_fn,
                                 exact_scan=bool(ck.get("exact_scan")),
                                 scan_files=(exp.get("subset") or {}).get("files"),
                                 short_guard=exp["modality"] in ("dense", "multivector"),
                                 scan_mask_fn=scan_mask_fn)
                o["ok"] = all(v["ok"] for v in o["systems"].values())
                out["oracle"] = o
            else:
                out["oracle"] = {"ok": False, "fails": ["not run: the inputs did not load"]}
        except (CheckError, SystemExit) as exc:
            out["oracle"] = {"ok": False, "fails": [f"oracle could not run: {exc}"]}
    with open(log, "w") as f:
        f.write(buf.getvalue())
    out["log"] = os.path.relpath(log, run_dir)
    if exp["modality"] == "sparse":
        try:
            out["sparse_weights"] = sparse_weight_problems(a, inputs) if inputs is not None else []
        except Exception as exc:  # noqa: BLE001  a check that cannot run fails
            out["sparse_weights"] = [f"could not check: {type(exc).__name__}: {exc}"]
    out["fails"] = [f"sparse weights: {f}" for f in out.get("sparse_weights") or []] + \
        [f"pairs: {f}" for f in out["pairs"]["fails"]] + \
        [f"oracle: {f}" for f in out["oracle"].get("fails", [])] + \
        [f"oracle {nm}: {f}" for nm, v in out["oracle"].get("systems", {}).items() for f in v["fails"]]
    out["ok"] = not out["fails"]
    return out


def run_checks(run_dir: str, exp: dict) -> dict:
    """-> the checks.json document; `ok` false fails the run."""
    from bench import timing
    _CACHE.clear()
    try:
        return _run_checks(run_dir, exp, timing)
    finally:
        _CACHE.clear()


def _run_checks(run_dir: str, exp: dict, timing) -> dict:
    inputs = input_hashes(run_dir)   # first: what the checks below read
    recs = _records(run_dir)
    log_dir = os.path.join(run_dir, "logs", "checks")
    os.makedirs(log_dir, exist_ok=True)
    R = timing.rounds(exp)
    doc = {"implemented": True, "stage": "S5", "ok": True, "searches": {}, "rounds": {},
           "e2e": None, "skipped": [], "problems": [], "warnings": [],
           "settings": {"tol": 1e-4, "tie_ulps": 128, "rescore": True, "order_check": True,
                        "oracle_queries": (exp.get("checks") or {}).get("oracle_queries")
                        or ORACLE_QUERIES_DEFAULT,
                        "exact_scan": bool((exp.get("checks") or {}).get("exact_scan")),
                        "oracle_sample_rule": f"numpy default_rng({ORACLE_SEED}).choice "
                        f"+ the first {ORACLE_MAX_DIFF} differing queries",
                        "oracle_score_tolerance": f"{SCORE_ATOL:g} + {SCORE_RTOL:g} x |exact|",
                        "gt": (exp.get("checks") or {}).get("gt"),
                        "max_score_ulps": (exp.get("checks") or {}).get("max_score_ulps")},
           "score_error": {}, "inputs": inputs}
    max_ulps = (exp.get("checks") or {}).get("max_score_ulps")
    tie_ulps = doc["settings"]["tie_ulps"]
    for s in exp["searches"]:
        per = doc["searches"].setdefault(s["name"], {})
        for r in range(1, R + 1):
            try:
                per[f"r{r}"] = check_round(exp, run_dir, recs, s, r, doc["skipped"], log_dir)
            except Exception as exc:   # noqa: BLE001  a check that cannot run is a failed check
                per[f"r{r}"] = {"ok": False, "fails": [f"{type(exc).__name__}: {exc}"]}
            if not per[f"r{r}"]["ok"]:
                doc["problems"].append(f"{s['name']} r{r}: " + "; ".join(per[f"r{r}"]["fails"]))
            se = doc["score_error"].setdefault(s["name"], {})
            for nm, e in round_score_error(per[f"r{r}"]).items():
                cur = se.setdefault(nm, dict.fromkeys(SCORE_ERROR_KEYS, 0.0))
                for key in SCORE_ERROR_KEYS:
                    cur[key] = max(cur[key], e[key])
        for nm, e in sorted(doc["score_error"].get(s["name"], {}).items()):
            # compare.judge's WARNING (printed to the checks log), surfaced here
            if 2 * e["max_ulps_adjudicated"] > tie_ulps:
                doc["warnings"].append(
                    f"score error {s['name']} {nm}: own error {e['max_ulps_adjudicated']:.1f} "
                    f"float32 ULPs x 2 > the {tie_ulps:g}-ULP tie window (on the adjudicated ids): "
                    "the window is too tight for its rounding, so a real disagreement may be "
                    "rounding (raise tie_ulps and report it)")
            if max_ulps is not None and e["max_ulps"] > float(max_ulps):
                doc["problems"].append(
                    f"score error {s['name']} {nm}: {e['max_ulps']:.1f} float32 ULPs > "
                    f"checks.max_score_ulps {float(max_ulps):g}")
        # 3. rounds: round N == round 1, per system
        rr = doc["rounds"].setdefault(s["name"], {})
        for sy in s["systems"]:
            first = recs.get(timing.tag(s["name"], sy, 1))
            if not first or not first.get("ok"):
                continue
            for r in range(2, R + 1):
                other = recs.get(timing.tag(s["name"], sy, r))
                if not other or not other.get("ok"):
                    continue
                try:
                    if first.get("kind") == "novabf":
                        same = same_table(_novabf_result(first, s), _novabf_result(other, s))
                    else:
                        same = same_topk(os.path.join(run_dir, first["topk"]),
                                         os.path.join(run_dir, other["topk"]))
                except (CheckError, OSError, KeyError, ValueError) as exc:
                    # could not compare at all: a failed check (not a round-to-round difference)
                    rr.setdefault(sy, {})[f"r{r}"] = "ERROR"
                    doc["problems"].append(f"rounds {s['name']} {sy}: r1 vs r{r} could not be "
                                           f"compared ({exc})")
                    continue
                rr.setdefault(sy, {})[f"r{r}"] = "IDENTICAL" if same else "DIFFERENT"
                if not same:   # a warning: checks 1 and 2 enforce correctness every round
                    doc["warnings"].append(f"rounds {s['name']} {sy}: r1 vs r{r} DIFFERENT")
    if exp.get("e2e"):
        sy = exp["e2e"]["system"]
        e2e = recs.get(f"e2e__{sy}")
        s0 = exp["searches"][0]
        first = recs.get(timing.tag(s0["name"], sy, 1))
        if e2e and e2e.get("ok") and first and first.get("ok"):
            # a cold leg that orders ties by a corpus field names its hits by that field, so
            # only its scores can be compared with round 1 (they never depend on tie order)
            by_id = (exp["e2e"].get("tiebreak") or {}).get("mode") == "id"
            cmp = same_scores if by_id else same_table
            try:
                same = cmp(_novabf_result(e2e, s0), _novabf_result(first, s0))
            except (CheckError, OSError) as exc:
                doc["e2e"] = "ERROR"
                doc["problems"].append(f"e2e: {exc}")
            else:
                doc["e2e"] = ("SCORES IDENTICAL" if by_id else "IDENTICAL") if same else "DIFFERENT"
                if not same:   # a FAILURE (a round's DIFFERENT is only a warning)
                    doc["problems"].append(f"e2e vs {s0['name']} r1 result: DIFFERENT")
    doc["ok"] = not doc["problems"]
    return doc
