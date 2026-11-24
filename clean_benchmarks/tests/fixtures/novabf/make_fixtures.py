"""Deterministic tiny CPU fixtures for the nova-bf reference (part 3).

    python make_fixtures.py OUT_DIR          # writes OUT_DIR/{dense,sparse,multivector,filtered}/

Every file is a pure function of the fixed seeds below (numpy PCG64), written by
pyarrow with fixed options, so two runs give byte-identical parquet files (with the
same pyarrow version: parquet's `created_by` footer names it).

Shapes mirror the paper corpora's column types, only tiny:
  dense        4 files x 1000 rows, list<float32>[32]           (Coyo/Pes2o: list<float>)
  sparse       4 files x 1000 rows, struct<indices: list<uint32>, values: list<float32>>
               (FineWeb gte), Zipf-ish vocabulary of 2000, ~16 nnz, a few empty rows
  multivector  4 files x 200 docs, list<list<float32>>[16], 3..12 tokens, a few null docs
  filtered     4 files x 1000 rows: dense_embedding + id + FineWeb's filter payload
               (text, url, language_score, date rfc3339 string, dump); the queries carry
               the ms-marco shared query schema (query_set, keyword_phrase, domain_slot_1..9,
               ls_gte, date_gte, date_lt, dump_set list<string>) with the same sentinel
               values the real file uses ('zzznomatchzzz000' slots, 1970/2100 date bounds).
Queries files sit OUTSIDE the corpus directories (every *.parquet there is corpus).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_FILES = 4
WRITE_KW = dict(compression="snappy", use_dictionary=True, write_statistics=True)


def _write(table: pa.Table, path: str, row_group_size: int | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size, **WRITE_KW)


def _unit(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _dense_list(x: np.ndarray) -> pa.Array:
    n, d = x.shape
    return pa.ListArray.from_arrays(pa.array(np.arange(0, n * d + 1, d, dtype=np.int32)),
                                    pa.array(x.reshape(-1), pa.float32()))


def dense(out: str) -> None:
    rng = np.random.default_rng(1001)
    d, rows = 32, 1000
    centers = _unit(rng, 8, d)
    for f in range(N_FILES):
        # clustered (non-uniform) vectors with some exact duplicates, so ties exist
        x = centers[rng.integers(0, 8, rows)] + 0.35 * rng.standard_normal((rows, d)).astype(np.float32)
        x = x.astype(np.float32)
        x[rows - 20:] = x[:20]  # duplicate rows -> exactly tied scores
        _write(pa.table({"embedding": _dense_list(x)}), f"{out}/dense/corpus/part-{f:03d}.parquet")
    nq = 300
    q = (centers[rng.integers(0, 8, nq)] + 0.35 * rng.standard_normal((nq, d))).astype(np.float32)
    qset = pa.array(["head"] * 60 + ["tail"] * (nq - 60))  # rows: subset (dense_multi)
    _write(pa.table({"query_id": pa.array(np.arange(nq, dtype=np.int64)), "query_set": qset,
                     "embedding": _dense_list(q)}),
           f"{out}/dense/queries.parquet")


def _sparse_rows(rng, n, vocab, nnz, empty_frac):
    p = 1.0 / np.arange(1, vocab + 1) ** 0.9
    p /= p.sum()
    idx_lists, val_lists = [], []
    for _ in range(n):
        if rng.random() < empty_frac:
            idx_lists.append(np.zeros(0, np.uint32))
            val_lists.append(np.zeros(0, np.float32))
            continue
        k = int(rng.integers(max(1, nnz // 2), nnz * 2))
        ids = np.unique(rng.choice(vocab, size=k, p=p)).astype(np.uint32)
        # quantized weights (multiples of 1/16) -> exact ties are likely
        vals = (np.ceil(rng.exponential(1.0, ids.size) * 16) / 16).astype(np.float32)
        idx_lists.append(ids)
        val_lists.append(vals)
    return pa.StructArray.from_arrays(
        [pa.array(idx_lists, pa.list_(pa.uint32())), pa.array(val_lists, pa.list_(pa.float32()))],
        names=["indices", "values"])


def sparse(out: str) -> None:
    rng = np.random.default_rng(2002)
    for f in range(N_FILES):
        col = _sparse_rows(rng, 1000, 2000, 16, 0.01)
        _write(pa.table({"sparse_embedding": col}), f"{out}/sparse/corpus/part-{f:03d}.parquet")
    nq = 120
    _write(pa.table({"query_id": pa.array(np.arange(nq, dtype=np.int64)),
                     "sparse_embedding": _sparse_rows(rng, nq, 2000, 8, 0.0)}),
           f"{out}/sparse/queries.parquet")


def _mv_rows(rng, n, d, lo, hi, null_frac):
    docs = []
    for _ in range(n):
        if rng.random() < null_frac:
            docs.append(None)
            continue
        t = int(rng.integers(lo, hi + 1))
        docs.append(_unit(rng, t, d).tolist())
    return pa.array(docs, pa.list_(pa.list_(pa.float32())))


def multivector(out: str) -> None:
    rng = np.random.default_rng(3003)
    for f in range(N_FILES):
        _write(pa.table({"multivector_embedding": _mv_rows(rng, 200, 16, 3, 12, 0.02)}),
               f"{out}/multivector/corpus/part-{f:03d}.parquet")
    nq = 40
    _write(pa.table({"query_id": pa.array(np.arange(nq, dtype=np.int64)),
                     "multivector_embedding": _mv_rows(rng, nq, 16, 2, 6, 0.0)}),
           f"{out}/multivector/queries.parquet")


WORDS = ("health history music physics dna science art river cell energy market law city "
         "garden ocean the of and data model court policy virus star").split()
DOMAINS = "com org net edu gov co io us arxiv jstor springer wiki blogspot nih".split()
DUMPS = ["CC-MAIN-2013-20", "CC-MAIN-2013-48", "CC-MAIN-2014-10", "CC-MAIN-2014-15",
         "CC-MAIN-2015-06", "CC-MAIN-2016-07"]
NOMATCH = "zzznomatchzzz000"


def _date(rng):
    # 2013-05-01 .. 2016-12-31, second resolution, RFC 3339 'Z'
    t = int(rng.integers(1367366400, 1483228800))
    return np.datetime_as_string(np.datetime64(t, "s"), unit="s") + "Z"


def filtered(out: str) -> None:
    rng = np.random.default_rng(4004)
    d, rows = 32, 1000
    centers = _unit(rng, 6, d)
    rid = 0
    for f in range(N_FILES):
        x = (centers[rng.integers(0, 6, rows)] + 0.4 * rng.standard_normal((rows, d))).astype(np.float32)
        text = [" ".join(rng.choice(WORDS, size=int(rng.integers(4, 15)))).capitalize()
                for _ in range(rows)]
        url = [f"https://www.{rng.choice(['alpha', 'beta', 'gamma', 'delta'])}"
               f".{rng.choice(DOMAINS)}/{rng.choice(WORDS)}/{int(rng.integers(0, 999))}"
               for _ in range(rows)]
        ls = rng.uniform(0.6, 1.0, rows)
        dates = [_date(rng) for _ in range(rows)]
        # dump-partitioned-ish like FineWeb: most rows of a file share one dump
        dump = [DUMPS[f] if rng.random() < 0.8 else str(rng.choice(DUMPS)) for _ in range(rows)]
        _write(pa.table({
            "id": pa.array(np.arange(rid, rid + rows, dtype=np.int64)),
            "dense_embedding": _dense_list(x),
            "text": pa.array(text), "url": pa.array(url),
            "language_score": pa.array(ls), "date": pa.array(dates), "dump": pa.array(dump),
        }), f"{out}/filtered/corpus/part-{f:03d}.parquet", row_group_size=500)
        rid += rows

    qs = (["filtered_text"] * 120 + ["structured"] * 120 + ["dense_only"] * 40)
    nq = len(qs)
    q = (centers[rng.integers(0, 6, nq)] + 0.4 * rng.standard_normal((nq, d))).astype(np.float32)
    kw, slots, ls_gte, dgte, dlt, dset = [], [[] for _ in range(9)], [], [], [], []
    for s in qs:
        if s == "filtered_text":
            kw.append(" ".join(rng.choice(WORDS, size=int(rng.integers(1, 3)))))
            m = int(rng.integers(1, 9))
            doms = list(rng.choice(DOMAINS, size=m, replace=False)) + [NOMATCH] * (9 - m)
            for i in range(9):
                slots[i].append(str(doms[i]))
            ls_gte.append(0.0); dgte.append("1970-01-01T00:00:00Z"); dlt.append("2100-01-01T00:00:00Z")
            dset.append(["zzznomatchzzz"])
        else:
            kw.append("")
            for i in range(9):
                slots[i].append(NOMATCH)
            ls_gte.append(float(rng.uniform(0.6, 0.98)))
            if rng.random() < 0.5:
                dgte.append("1970-01-01T00:00:00Z"); dlt.append("2100-01-01T00:00:00Z")
            else:
                a = _date(rng); b = _date(rng)
                dgte.append(min(a, b)); dlt.append(max(a, b))
            dset.append([str(x) for x in rng.choice(DUMPS, size=int(rng.integers(1, 5)), replace=False)])
    cols = {"query_id": pa.array(np.arange(nq, dtype=np.int64)), "query_set": pa.array(qs),
            "dense_embedding": _dense_list(q), "keyword_phrase": pa.array(kw)}
    for i in range(9):
        cols[f"domain_slot_{i + 1}"] = pa.array(slots[i])
    cols.update(ls_gte=pa.array(ls_gte), date_gte=pa.array(dgte), date_lt=pa.array(dlt),
                dump_set=pa.array(dset, pa.list_(pa.string())))
    _write(pa.table(cols), f"{out}/filtered/queries/queries.parquet")


def main(out: str) -> None:
    for fn in (dense, sparse, multivector, filtered):
        fn(out)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
