"""Tiny sparse / multivector / dense corpora and compare inputs for the scan tests."""

import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bench.verify import agree
from bench.verify import compare as C


def _sparse_fixture(tmp_path, n_files=3, rows=6, seed=1):
    """Small sparse corpus + queries with integer weights, so many scores tie exactly."""
    rng = np.random.default_rng(seed)
    (tmp_path / "c").mkdir()

    def rowset(n):
        out = []
        for _ in range(n):
            m = int(rng.integers(0, 4))
            idx = rng.choice(8, size=m, replace=False).astype(np.uint32).tolist()
            out.append((idx, [float(rng.integers(1, 3)) for _ in idx]))
        return out

    def write(path, rs):
        arr = pa.StructArray.from_arrays(
            [pa.array([r[0] for r in rs], pa.list_(pa.uint32())),
             pa.array([r[1] for r in rs], pa.list_(pa.float32()))], ["indices", "values"])
        pq.write_table(pa.table({"sp": arr}), path)
    for f in range(n_files):
        write(tmp_path / "c" / f"f{f}.parquet", rowset(rows))
    write(tmp_path / "q.parquet", rowset(5))
    return n_files, rows


def _mv_fixture(tmp_path, n_files=3, rows=5, seed=2, d=3):
    rng = np.random.default_rng(seed)
    (tmp_path / "c").mkdir()
    t = pa.list_(pa.list_(pa.float32()))

    def docs(n):
        return [[rng.integers(-1, 2, d).astype(np.float32).tolist()
                 for _ in range(int(rng.integers(0, 3)))] for _ in range(n)]
    for f in range(n_files):
        pq.write_table(pa.table({"mv": pa.array(docs(rows), t)}), tmp_path / "c" / f"f{f}.parquet")
    q = docs(5)
    q[0] = q[0] or [[1.0, 0.0, 0.0]]
    pq.write_table(pa.table({"mv": pa.array(q, t)}), tmp_path / "q.parquet")
    return n_files, rows


def _args(tmp_path, modality, col, metric="ip", k=3):
    return types.SimpleNamespace(k=k, modality=modality, metric=metric, corpus=str(tmp_path / "c"),
                                 vcol=col, queries=str(tmp_path / "q.parquet"), qcol=col,
                                 query_rows=None)


def _inputs(n_files, rows, n=5):
    return {"keys": [f"k{i}" for i in range(n_files)],
            "offsets": [rows * i for i in range(n_files + 1)], "n": n,
            "qn": [str(i) for i in range(n)]}


def _oneshot(a, inputs, sample, files, allowed=None):
    """The one-shot reference: every (query, row) of the first `files` files scored at once by
    compare.py's own float64 rescorer, then sorted best first, ties to the lowest id."""
    off = inputs["offsets"]
    need = [set() for _ in range(inputs["n"])]
    for q in sample:
        need[q] = set(range(off[files]))
    fs, paths = C.corpus_paths(a.corpus, len(inputs["keys"]))
    sparse, mv = C.harness_modules()
    if a.modality == "sparse":
        q, _ = sparse.load_queries(a.queries, a.qcol)
        ex = C.exact_scores_sparse(need, off, paths, fs, a.vcol, q, a.metric, threads=1)
    elif a.modality == "multivector":
        q, _ = mv.load_queries(a.queries, a.qcol)
        ex = C.exact_scores_mv(need, off, paths, fs, a.vcol, q, a.metric, threads=1)
    else:
        xq = agree._dense_queries(a, inputs["n"], inputs["qn"])
        ex = C.exact_scores(need, a.corpus, a.vcol, off, paths, fs, xq, a.metric, threads=1)
    out = {}
    for q in sample:
        cand = [(g, ex[(q, g)]) for g in range(off[files])
                if np.isfinite(ex[(q, g)]) and (allowed is None or allowed[q][g])]
        cand.sort(key=lambda x: (-x[1], x[0]))
        out[q] = {g for g, _ in cand[:a.k]}
    return out


def _tiny_corpus(tmp_path, n_files=3, rows=7, d=4):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(5)
    xs = []
    (tmp_path / "c").mkdir()
    for f in range(n_files):
        x = rng.standard_normal((rows, d)).astype(np.float32)
        xs.append(x)
        pq.write_table(pa.table({"emb": pa.array(list(x), pa.list_(pa.float32(), d))}),
                       tmp_path / "c" / f"f{f}.parquet")
    q = rng.standard_normal((6, d)).astype(np.float32)
    pq.write_table(pa.table({"emb": pa.array(list(q), pa.list_(pa.float32(), d))}), tmp_path / "q.parquet")
    return np.concatenate(xs), q
