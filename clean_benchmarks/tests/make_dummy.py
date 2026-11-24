"""Write a dummy parquet corpus (dense, sparse or multi-vector) and its queries, for tests.

    python tests/make_dummy.py OUT_DIR --n-files 8 --rows-per-file 10000 --dim 2560
    python tests/make_dummy.py OUT_DIR --n-files 0 --queries-out q.parquet --nq 20000

Keep the queries file outside OUT_DIR: every *.parquet there is read as corpus. One file is
built and freed at a time (peak memory ~ one file of float32, x2 while Arrow builds it).

Dense (default): column ``embedding``, ``list<float32>``, snappy, one row group per file
(nova-embed's layout). Rows per file is the benchmark's block size (one file = one search
call) and row groups per file cap decode threads (one thread per row group), so sweep each
by writing one corpus per value. Variants:
  FineWeb-like:  --value-type f16 --payload-bytes N   (half floats + a ``text`` column)
  Pes2o-VE-like: --list-type fixed --compression none --row-group-size 8192

Sparse (``--modality sparse``): one ``struct<indices: list<uint32>, values: list<...>>``
column, indices sorted and unique from a Zipf-like vocabulary (so queries share columns with
documents), values > 0. Multi-vector (``--modality multivector``): one
``list<list<float>>`` column, ~``--tokens`` unit-norm tokens per document; zero-token
documents are null. For both, the queries file also gets ``gt_ids``: the exact top-``--gt-k``
over the whole corpus (float64, lowest id first on equal scores).

    python tests/make_dummy.py OUT_DIR --modality sparse --n-files 4 --rows-per-file 5000 \\
        --vocab 30522 --nnz 64 --queries-out q.parquet --nq 200 --query-nnz 16
    python tests/make_dummy.py OUT_DIR --modality multivector --n-files 4 \\
        --rows-per-file 500 --dim 128 --tokens 64 --queries-out q.parquet --nq 100
"""

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == "__main__":  # run as a script: put the clean_benchmarks root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def unit_vectors(rng, n, d, dtype=np.float32):
    """`n` random unit-norm `d`-dim vectors (drawn and normalised in float32, then cast)."""
    x = rng.standard_normal((n, d), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(dtype, copy=False)


VALUE_TYPES = {"f32": np.float32, "f16": np.float16}
# list = list<...> with int32 offsets; fixed = fixed_size_list<...>[dim] (no offsets)
LIST_TYPES = ("list", "fixed")


def check_int32_offsets(n, width, what):
    """Raise before allocating when ``n`` rows of ``width`` values each would
    overflow an Arrow int32 offset buffer (list<...> / string)."""
    if n * width >= 2**31:
        raise ValueError(f"{n} rows x {width} {what} = {n * width} >= 2^31 overflows "
                         "Arrow's int32 offsets; use fewer rows per file")


def payload_array(rng, n, nbytes):
    """``n`` strings of ``nbytes`` random lowercase letters.

    Snappy barely compresses them, so the column really is large on disk. Type
    ``string`` (int32 offsets, like FineWeb's text column): n * nbytes must stay
    below 2^31."""
    check_int32_offsets(n, nbytes, "payload bytes")
    data = rng.integers(97, 123, size=n * nbytes, dtype=np.uint8)
    offsets = np.arange(0, (n + 1) * nbytes, nbytes, dtype=np.int32)
    return pa.StringArray.from_buffers(n, pa.py_buffer(offsets), pa.py_buffer(data))


def vectors_table(x, column="embedding", value_dtype=np.float32, payload_bytes=0,
                  payload_seed=(0, 7), list_type="list"):
    """Arrow table of vectors ``x`` (plus an optional ``text`` payload column).

    ``payload_seed``: the payload RNG's seed sequence (make_corpus gives every
    file its own, distinct from every vector stream).
    ``list_type``: ``list`` (offsets) or ``fixed`` (fixed_size_list[d])."""
    if list_type not in LIST_TYPES:
        raise ValueError(f"list type {list_type!r} not in {LIST_TYPES}")
    n, d = x.shape
    flat = pa.array(np.ascontiguousarray(x, dtype=value_dtype).reshape(-1))
    if list_type == "fixed":  # no offsets: nothing to overflow
        col = pa.FixedSizeListArray.from_arrays(flat, d)
    else:
        check_int32_offsets(n, d, "vector values")
        offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32))
        col = pa.ListArray.from_arrays(offsets, flat)
    cols = {column: col}
    if payload_bytes:
        cols["text"] = payload_array(np.random.default_rng(list(payload_seed)), n,
                                     payload_bytes)
    return pa.table(cols)


def write_vectors(path, x, column="embedding", row_group_size=None,
                  compression="snappy", value_dtype=np.float32, payload_bytes=0,
                  payload_seed=(0, 7), list_type="list"):
    """Write vectors ``x`` to one parquet file (one row group unless ``row_group_size``)."""
    table = vectors_table(x, column, value_dtype, payload_bytes, payload_seed,
                          list_type)
    pq.write_table(table, path, compression=compression,
                   row_group_size=row_group_size or max(1, x.shape[0]))
    return path


def make_corpus(out, n_files, rows_per_file, dim, seed=0, column="embedding",
                row_group_size=None, compression="snappy", value_dtype=np.float32,
                payload_bytes=0, list_type="list"):
    """Write ``n_files`` dense corpus files to ``out``; returns their paths."""
    os.makedirs(out, exist_ok=True)
    paths = []
    for i in range(n_files):
        # Distinct RNG streams: vectors [seed, i, 0x56], payload [seed, i, 7], harness-made
        # queries [seed, 0x5155]. (A trailing 0 does not change a SeedSequence.)
        rng = np.random.default_rng([seed, i, 0x56])
        x = unit_vectors(rng, rows_per_file, dim)
        p = os.path.join(out, f"part-{i:05d}.parquet")
        write_vectors(p, x, column, row_group_size, compression, value_dtype,
                      payload_bytes, (seed, i, 7), list_type)
        del x
        paths.append(p)
    return paths


# ----------------------------------------------------------------- sparse

def random_sparse(rng, n, vocab, nnz_mean, empty_frac=0.0):
    """``n`` sparse rows as CSR parts ``(indptr int64, indices uint32, values
    float32)``: per row about ``nnz_mean`` distinct columns (at least 1, at
    most ``vocab``), drawn half Zipf-like (popular columns, so rows overlap)
    and half uniform, sorted and unique; values in (0, 1] (non-negative, as
    a ReLU'd embedder's). ``empty_frac`` of the rows get no nonzeros."""
    want = np.minimum(np.maximum(1, rng.poisson(nnz_mean, n)), vocab)
    if empty_frac:
        want[rng.random(n) < empty_frac] = 0
    tot = int(want.sum())
    rows = np.repeat(np.arange(n, dtype=np.int64), want)
    zipf = (rng.zipf(1.3, tot) - 1) % vocab
    unif = rng.integers(0, vocab, tot)
    cols = np.where(rng.random(tot) < 0.5, zipf, unif).astype(np.int64)
    key = np.unique(rows * vocab + cols)  # sorted by row, then column; distinct
    rows, cols = key // vocab, key % vocab
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    values = (1.0 - rng.random(len(cols))).astype(np.float32)  # (0, 1]
    return indptr, cols.astype(np.uint32), values


def sparse_column(indptr, indices, values, value_dtype=np.float32):
    """The struct<indices: list<uint32>, values: list<...>> column of CSR
    parts (int32 offsets: nnz must stay below 2^31)."""
    check_int32_offsets(1, len(indices), "sparse nonzeros")
    offs = pa.array(np.asarray(indptr, np.int32))
    il = pa.ListArray.from_arrays(offs, pa.array(np.asarray(indices, np.uint32)))
    vl = pa.ListArray.from_arrays(offs, pa.array(np.asarray(values, value_dtype)))
    return pa.StructArray.from_arrays([il, vl], names=["indices", "values"])


def write_sparse(path, parts, column="sparse_embedding", row_group_size=None,
                 compression="snappy", value_dtype=np.float32, extra=None):
    """Write CSR ``parts`` (plus ``extra`` columns) to one parquet file."""
    indptr = parts[0]
    n = len(indptr) - 1
    cols = {column: sparse_column(*parts, value_dtype=value_dtype)}
    cols.update(extra or {})
    pq.write_table(pa.table(cols), path, compression=compression,
                   row_group_size=row_group_size or max(1, n))
    return path


def sparse_ground_truth(corpus_parts, query_parts, k, metric="ip", value_dtype=np.float32):
    """Exact top-``k`` ids (int64, -1 padded) of every query against the whole corpus.

    Uses the harness's sparse oracle (sparse.exact_topk) in float64, lowest id
    first on equal scores, on the values as stored: the corpus rounded to
    ``value_dtype``, the queries to float32 (what make_sparse_queries writes)."""
    import scipy.sparse as ssp

    from bench.data.formats import sparse
    vocab = 1 + max(int(p[1].max()) if len(p[1]) else 0 for p in corpus_parts + query_parts)

    def mat(parts, dt):
        return ssp.vstack([ssp.csr_matrix(
            (np.asarray(v, dt).astype(np.float64), i.astype(np.int64), p),
            shape=(len(p) - 1, vocab)) for p, i, v in parts], format="csr")

    Q, X = mat(query_parts, np.float32), mat(corpus_parts, value_dtype)
    _, I = sparse.exact_topk(Q, X, k, metric, np.float64)
    if I.shape[1] < k:
        I = np.pad(I, ((0, 0), (0, k - I.shape[1])), constant_values=-1)
    return I


def make_sparse_corpus(out, n_files, rows_per_file, vocab, nnz_mean, seed=0,
                       column="sparse_embedding", row_group_size=None, compression="snappy",
                       value_dtype=np.float32, empty_frac=0.0):
    """Sparse corpus files ([seed, i, 0x53] per file); returns (paths, parts)."""
    os.makedirs(out, exist_ok=True)
    paths, parts = [], []
    for i in range(n_files):
        rng = np.random.default_rng([seed, i, 0x53])
        p = random_sparse(rng, rows_per_file, vocab, nnz_mean, empty_frac)
        path = os.path.join(out, f"part-{i:05d}.parquet")
        write_sparse(path, p, column, row_group_size, compression, value_dtype)
        paths.append(path)
        parts.append(p)
    return paths, parts


def make_sparse_queries(path, nq, vocab, nnz_mean, seed=1, column="sparse_embedding",
                        corpus_parts=None, gt_k=10, gt_metric="ip",
                        value_dtype=np.float32):
    """Sparse queries ([seed, 2**31, 0x53]) plus, with ``corpus_parts`` and
    ``gt_k`` > 0, a ``gt_ids`` list<int64> column of exact ground truth."""
    rng = np.random.default_rng([seed, 2**31, 0x53])
    q = random_sparse(rng, nq, vocab, nnz_mean)
    extra = {}
    if corpus_parts is not None and gt_k:
        I = sparse_ground_truth(corpus_parts, [q], gt_k, gt_metric, value_dtype)
        extra["gt_ids"] = pa.array(list(I), pa.list_(pa.int64()))
    return write_sparse(path, q, column, value_dtype=np.float32, extra=extra)


def random_multivector(rng, n, dim, tok_mean, empty_frac=0.0, nonneg=False, max_tok=None):
    """``n`` ragged documents as ``(offsets int64 (n + 1,), tokens float32
    (n_tok, dim))``: about ``tok_mean`` tokens each (at least 1, at most
    ``max_tok``), unit-norm Gaussian tokens (``nonneg``: their absolute
    values, renormalised: every token score >= 0). ``empty_frac`` of the
    documents get no tokens (written as null)."""
    want = np.maximum(1, rng.poisson(tok_mean, n))
    if max_tok:
        want = np.minimum(want, max_tok)
    if empty_frac:
        want[rng.random(n) < empty_frac] = 0
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(want, out=offsets[1:])
    tok = unit_vectors(rng, int(offsets[-1]), dim)
    if nonneg:
        tok = np.abs(tok)
        tok /= np.linalg.norm(tok, axis=1, keepdims=True)
    return offsets, tok


def multivector_column(offsets, tokens, value_dtype=np.float32):
    """The list<list<...>> column of ragged parts; zero-token documents are
    NULL (nova-embed's on_empty_input="null")."""
    n_tok, dim = tokens.shape
    check_int32_offsets(1, n_tok * dim, "multi-vector token values")
    vt = {np.dtype(np.float16): pa.float16(), np.dtype(np.float32): pa.float32()}[
        np.dtype(value_dtype)]
    vals = pa.array(np.asarray(tokens, value_dtype).ravel(), vt)
    inner = pa.ListArray.from_arrays(pa.array(np.arange(n_tok + 1, dtype=np.int32) * dim,
                                              pa.int32()), vals)
    lens = np.diff(offsets)
    mask = pa.array(lens == 0) if (lens == 0).any() else None
    return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32), pa.int32()), inner,
                                    mask=mask)


def write_multivector(path, parts, column="multivector_embedding", row_group_size=None,
                      compression="snappy", value_dtype=np.float32, extra=None):
    """Write ragged multi-vector ``parts`` (plus ``extra`` columns) to one parquet file."""
    offsets = parts[0]
    n = len(offsets) - 1
    cols = {column: multivector_column(*parts, value_dtype=value_dtype)}
    cols.update(extra or {})
    pq.write_table(pa.table(cols), path, compression=compression,
                   row_group_size=row_group_size or max(1, n))
    return path


def mv_ground_truth(corpus_parts, query_parts, k, metric="ip", value_dtype=np.float32):
    """Exact top-``k`` MaxSim ids (int64, -1 padded) of every query against the whole corpus.

    float64, lowest id first on equal scores, on the tokens as stored; the same
    building blocks as the harness's oracle (multivector.oneshot_reference)."""
    from bench.data.formats import multivector as multivec
    from bench.data.formats import sparse
    q_off, q_tok = query_parts
    q_tok = np.asarray(q_tok, np.float32).astype(np.float64)
    if metric == "cosine":
        q_tok = multivec.unit_tokens(q_tok, np.float64)
    D = I = None
    base = 0
    for off, tok in corpus_parts:
        t = np.asarray(tok, value_dtype).astype(np.float64)
        if metric == "cosine":
            t = multivec.unit_tokens(t, np.float64)
        S = multivec.maxsim_matrix(q_off, q_tok, off, t, np.float64)
        d, i = multivec.topk_tile(S, k)
        i = i + base
        D, I = (d, i) if D is None else sparse.merge_sorted(D, I, d, i, k)
        base += len(off) - 1
    D, I = multivec.to_padding(D.astype(np.float32), I)
    return sparse.pad(D, I, k)[1]


def make_mv_corpus(out, n_files, rows, dim, tok_mean, seed=0, column="multivector_embedding",
                   row_group_size=None, compression="snappy", value_dtype=np.float32,
                   empty_frac=0.0, nonneg=False):
    """Multi-vector corpus files ([seed, i, 0x4D56] per file); returns (paths, parts)."""
    os.makedirs(out, exist_ok=True)
    paths, parts = [], []
    for i in range(n_files):
        rng = np.random.default_rng([seed, i, 0x4D56])
        p = random_multivector(rng, rows, dim, tok_mean, empty_frac, nonneg)
        paths.append(write_multivector(os.path.join(out, f"part-{i:05d}.parquet"), p, column,
                                       row_group_size, compression, value_dtype))
        parts.append(p)
    return paths, parts


def make_mv_queries(path, nq, dim, tok_mean, seed=1, column="multivector_embedding",
                    corpus_parts=None, gt_k=10, gt_metric="ip", value_dtype=np.float32,
                    nonneg=False, empty_frac=0.0):
    """Multi-vector queries ([seed, 2**31, 0x4D56]) plus, with
    ``corpus_parts`` and ``gt_k`` > 0, a ``gt_ids`` list<int64> column of
    exact MaxSim ground truth."""
    rng = np.random.default_rng([seed, 2**31, 0x4D56])
    q = random_multivector(rng, nq, dim, tok_mean, empty_frac, nonneg)
    extra = {}
    if corpus_parts is not None and gt_k:
        I = mv_ground_truth(corpus_parts, q, gt_k, gt_metric, value_dtype)
        extra["gt_ids"] = pa.array(list(I), pa.list_(pa.int64()))
    return write_multivector(path, q, column, value_dtype=np.float32, extra=extra)


def make_queries(path, nq, dim, seed=1, column="embedding", list_type="list"):
    """Dense unit-norm queries ([seed, 2**31]) written to ``path``."""
    rng = np.random.default_rng([seed, 2**31])
    return write_vectors(path, unit_vectors(rng, nq, dim), column,
                         list_type=list_type)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output directory")
    ap.add_argument("--n-files", type=int, default=4)
    ap.add_argument("--rows-per-file", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modality", choices=["dense", "sparse", "multivector"], default="dense")
    ap.add_argument("--tokens", type=int, default=64,
                    help="multivector: mean tokens per document")
    ap.add_argument("--query-tokens", type=int, default=32,
                    help="multivector: mean tokens per query")
    ap.add_argument("--nonneg", action="store_true",
                    help="multivector: non-negative tokens (every token score >= 0)")
    ap.add_argument("--column", default=None,
                    help="vector column (default: embedding; sparse_embedding with "
                         "--modality sparse)")
    ap.add_argument("--vocab", type=int, default=30522,
                    help="sparse: column count (indices in [0, vocab))")
    ap.add_argument("--nnz", type=int, default=64, help="sparse: mean nonzeros per row")
    ap.add_argument("--query-nnz", type=int, default=16,
                    help="sparse: mean nonzeros per query")
    ap.add_argument("--empty-rows", type=float, default=0.0,
                    help="sparse: share of corpus rows with no nonzeros")
    ap.add_argument("--gt-k", type=int, default=10,
                    help="sparse: exact ground-truth ids per query written to the "
                         "queries file's gt_ids column (0 = none; needs the corpus in "
                         "RAM: small corpora)")
    ap.add_argument("--gt-metric", choices=["ip", "cosine"], default="ip",
                    help="sparse: the metric gt_ids is exact for")
    ap.add_argument("--row-group-size", type=int, default=None,
                    help="rows per row group (default: one row group per file)")
    ap.add_argument("--compression", default="snappy",
                    help="parquet codec (snappy, zstd, none = uncompressed, ...)")
    ap.add_argument("--value-type", choices=sorted(VALUE_TYPES), default="f32",
                    help="stored vector element type (f16 = list<halffloat>)")
    ap.add_argument("--list-type", choices=LIST_TYPES, default="list",
                    help="vector column layout: list<...> or fixed_size_list<...>[dim] "
                         "(corpus and queries)")
    ap.add_argument("--payload-bytes", type=int, default=0,
                    help="add a 'text' column of this many bytes per row")
    ap.add_argument("--queries-out", default=None,
                    help="also write --nq query vectors to this parquet path "
                         "(must not be inside OUT)")
    ap.add_argument("--nq", type=int, default=20000)
    args = ap.parse_args()
    if args.column is None:
        args.column = {"sparse": "sparse_embedding",
                       "multivector": "multivector_embedding"}.get(args.modality, "embedding")
    if args.queries_out:
        qdir = os.path.dirname(os.path.abspath(args.queries_out))
        if qdir == os.path.abspath(args.out):
            raise SystemExit("--queries-out must not be inside the corpus dir")
    if args.modality == "multivector":
        vdt = VALUE_TYPES[args.value_type]
        paths, parts = make_mv_corpus(args.out, args.n_files, args.rows_per_file, args.dim,
                                      args.tokens, args.seed, args.column,
                                      args.row_group_size, args.compression, vdt,
                                      args.empty_rows, args.nonneg)
        print(f"wrote {len(paths)} multi-vector files x {args.rows_per_file} documents "
              f"(~{args.tokens} tokens x {args.dim}-D) to {args.out}")
        if args.queries_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.queries_out)), exist_ok=True)
            make_mv_queries(args.queries_out, args.nq, args.dim, args.query_tokens,
                            args.seed + 1, args.column, parts if parts else None,
                            args.gt_k, args.gt_metric, vdt, args.nonneg)
            print(f"wrote {args.nq} multi-vector queries to {args.queries_out}")
        return
    if args.modality == "sparse":
        vdt = VALUE_TYPES[args.value_type]
        paths, parts = make_sparse_corpus(args.out, args.n_files, args.rows_per_file,
                                          args.vocab, args.nnz, args.seed, args.column,
                                          args.row_group_size, args.compression, vdt,
                                          args.empty_rows)
        print(f"wrote {len(paths)} sparse files x {args.rows_per_file} rows "
              f"(vocab {args.vocab}, ~{args.nnz} nnz/row) to {args.out}")
        if args.queries_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.queries_out)), exist_ok=True)
            make_sparse_queries(args.queries_out, args.nq, args.vocab, args.query_nnz,
                                args.seed + 1, args.column, parts if parts else None,
                                args.gt_k, args.gt_metric, vdt)
            print(f"wrote {args.nq} sparse queries"
                  f"{f' (+ gt_ids, exact top-{args.gt_k} {args.gt_metric})' if parts and args.gt_k else ''}"
                  f" to {args.queries_out}")
        return

    if args.n_files:
        paths = make_corpus(args.out, args.n_files, args.rows_per_file, args.dim,
                            args.seed, args.column, args.row_group_size,
                            args.compression, VALUE_TYPES[args.value_type],
                            args.payload_bytes, args.list_type)
        print(f"wrote {len(paths)} files x {args.rows_per_file} rows x "
              f"{args.dim}-D to {args.out}")
    if args.queries_out:
        qdir = os.path.dirname(os.path.abspath(args.queries_out))
        if qdir == os.path.abspath(args.out):
            raise SystemExit("--queries-out must not be inside the corpus dir")
        os.makedirs(qdir, exist_ok=True)
        make_queries(args.queries_out, args.nq, args.dim, args.seed + 1, args.column,
                     args.list_type)
        print(f"wrote {args.nq} queries to {args.queries_out}")


if __name__ == "__main__":
    main()
