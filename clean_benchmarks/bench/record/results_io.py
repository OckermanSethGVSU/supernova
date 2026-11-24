"""Inputs and outputs of one harness run: load or generate the queries, read --gt-ids-column
ground truth and score recall against it, find the corpus dimension, append CSV / trace rows,
and keep the --verify pending sidecars.
"""

import csv
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bench.systems import loop
from bench.record import schema
from bench.data.sources import list_parquet, read_file


# ---------------------------------------------------------------- helpers

def load_queries(args, dtype):
    """The ``--query-file`` queries (read once, untimed, cast to ``dtype`` on the host), or
    None when queries are generated instead (random_queries, which needs d)."""
    if not args.query_file:
        return None
    fs, refs = list_parquet(args.query_file)
    xq = np.concatenate([read_file(fs, r.path, args.query_column, size=r.size,
                                   mode="arrow") for r in refs])
    if args.nq:
        if args.nq > len(xq):
            print(f"WARNING: --nq {args.nq} > the {len(xq)} rows in "
                  f"{args.query_file}; using all {len(xq)}", file=sys.stderr)
        xq = xq[:args.nq]
    return np.ascontiguousarray(xq, dtype=dtype)


QUERY_STREAM = 0x5155  # 'Q': generated queries' RNG stream tag (see random_queries)


def random_queries(args, d, dtype):
    """``--nq`` random unit-norm queries from ``default_rng([seed, QUERY_STREAM])``.

    The QUERY_STREAM tag keeps them apart from tests/make_dummy.py's corpus vectors
    (``[seed, file, 0x56]``) for the same seed. A bare ``seed`` would not: it equals corpus
    file 0's ``[seed, 0]``, since trailing zeros do not change a SeedSequence."""
    rng = np.random.default_rng([args.seed, QUERY_STREAM])
    xq = rng.standard_normal((args.nq, d), dtype=np.float32)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return np.ascontiguousarray(xq, dtype=dtype)


def gt_id_matrix(col, what):
    """An Arrow list / large_list / fixed_size_list column of integer ids as an int64
    (rows, width) matrix; width is the shortest row's length (longer rows are cut). Nulls are
    refused."""
    if col.null_count:
        raise ValueError(f"{what}: null rows")
    t = col.type
    if not (pa.types.is_list(t) or pa.types.is_large_list(t) or
            pa.types.is_fixed_size_list(t)) or not pa.types.is_integer(t.value_type):
        raise ValueError(f"{what}: {t} is not a list of integer ids")
    if pa.types.is_fixed_size_list(t):
        n, w = len(col), t.list_size
        flat = col.flatten()  # the rows' values (slice offsets respected)
        if flat.null_count:
            raise ValueError(f"{what}: null ids")
        return flat.to_numpy(zero_copy_only=False).astype(np.int64, copy=False).reshape(n, w)
    offs = col.offsets.to_numpy()
    lens = np.diff(offs)
    w = int(lens.min()) if len(lens) else 0
    vals = col.values
    if vals.null_count:
        raise ValueError(f"{what}: null ids")
    vals = vals.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return vals[offs[:-1, None] + np.arange(w)]


def load_gt_ids(args, nq):
    """The ``--gt-ids-column`` of ``--query-file`` for the first ``nq`` queries, row for row,
    as an int64 (nq, width) matrix; read once, untimed.

    The ids must be global row positions in this harness's corpus order (sorted file
    listing, rows in file order), the same ids the run returns."""
    fs, refs = list_parquet(args.query_file)
    parts, n = [], 0
    for r in refs:
        if n >= nq:
            break
        t = pq.read_table(r.path, columns=[args.gt_ids_column], filesystem=fs)
        col = t.column(0).combine_chunks()
        parts.append(gt_id_matrix(col[:nq - n], f"{r.path}: {args.gt_ids_column}"))
        n += len(parts[-1])
    if n < nq:
        raise SystemExit(f"--gt-ids-column {args.gt_ids_column}: {n} rows, fewer than the "
                         f"{nq} queries")
    w = min(p.shape[1] for p in parts)
    if w < args.k:
        raise SystemExit(f"--gt-ids-column {args.gt_ids_column}: {w} ids per query, fewer "
                         f"than --k {args.k} (recall@k needs k <= the ground truth width)")
    return np.concatenate([p[:, :w] for p in parts])


def gt_recall(I, gt):
    """Mean recall@k over queries, k = I's columns.

    Per query: |returned ids & first k ground-truth ids| / k', where k' counts the valid
    (>= 0) ids among those first k; queries with none are skipped. -1 padding in I never
    matches. Vectorised: each (query, id) pair becomes one int64 key, and the two key sets
    are intersected."""
    nq, k = I.shape
    g = gt[:nq, :k]
    span = int(max(I.max(initial=0), g.max(initial=0))) + 2
    row = np.arange(nq, dtype=np.int64)[:, None] * span
    got = (I + 1 + row)[I >= 0]
    want = np.unique((g + 1 + row)[g >= 0])  # distinct per query
    hits = np.intersect1d(np.unique(got), want, assume_unique=True)
    per_q = np.bincount(hits // span, minlength=nq)
    denom = (g >= 0).sum(axis=1)
    ok = denom > 0
    return float(np.mean(per_q[ok] / denom[ok])) if ok.any() else float("nan")


def corpus_dim(args, xq, footer, fs, ref, mode):
    """The corpus dimension d, without an untimed column read when possible.

    Taken from --query-file's queries, else --dim, else the first file's footer; every source
    that knows d must agree. Only when none knows is the first file read (with the selected
    mode). run_pass still checks every block."""
    d = xq.shape[1] if xq is not None else args.dim
    if footer.dim is not None and d is not None and footer.dim != d:
        what = "queries" if xq is not None else "--dim"
        raise ValueError(f"{what} are {d}-D, corpus is {footer.dim}-D")
    if args.dim and xq is not None and args.dim != d:
        raise ValueError(f"--dim {args.dim} but queries are {d}-D")
    if d is None:
        d = footer.dim
    if d is None:
        d = read_file(fs, ref.path, args.column, size=ref.size, mode=mode).shape[1]
    return d


def trace_path(csv_path):
    """The per-file trace path of ``csv_path``: ``x.csv`` gives ``x.trace.csv``; a path
    without ``.csv`` gets ``.trace.csv`` appended."""
    stem = csv_path[:-len(".csv")] if csv_path.endswith(".csv") else csv_path
    return stem + ".trace.csv"


def check_csv(path, fields=schema.CSV_FIELDS):
    """True if ``path`` already has rows; refuses a foreign header."""
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:
            raise SystemExit(f"{path}: existing header differs from this "
                             "harness's columns (e.g. a CSV from before a column "
                             "was added); write to a new CSV")
    return exists


def append_csv(path, rows, fields=schema.CSV_FIELDS):
    exists = check_csv(path, fields)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _blank(v):
    return "" if v is None else v


def pending_paths(csv_path, trace_csv):
    """The --verify sidecars of ``csv_path`` and its trace: ``x.csv.pending``
    and ``x.trace.csv.pending``."""
    return csv_path + ".pending", trace_csv + ".pending"


def check_pending(csv_path, trace_csv):
    """Refuse to start a --verify run while an earlier run's sidecars hold rows: that run was
    killed (SIGKILL, OOM) before its check, and appending would mix two runs."""
    for p in pending_paths(csv_path, trace_csv):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            raise SystemExit(f"{p} exists: the rows of an earlier --verify run that was "
                             "killed before its check (verified=pending, never judged); "
                             "inspect it, then move it away (or delete it)")


def append_pending(csv_path, trace_csv, row, warm, rep, t_start_unix, trace):
    """Write a --verify pass's rows to the sidecars, marked pending, as soon as the pass ends,
    so a run killed outright (SIGKILL) still leaves its timings on disk, outside the CSV."""
    pend, tpend = pending_paths(csv_path, trace_csv)
    append_csv(pend, [dict(row, verified=schema.PENDING)])
    append_csv(tpend, loop.trace_rows(row, warm, rep, t_start_unix, trace, schema.PENDING), schema.TRACE_FIELDS)
