#!/usr/bin/env python3
"""Write FineWeb's 10,000 sparse queries as a nova-bf queries file.

    python prep_queries_fineweb.py SRC OUT_DIR

SRC is the MS MARCO shared queries file (ms_marco_shared_queries_fp32.parquet): every query set
of the FineWeb-10B benchmark in one file. The sparse benchmark's queries are its
`query_set == "dense_sparse"` rows, taken in file order, which is the row order of the public
release's queries/gt_sparse_k1000.parquet (Qdrant/FineWeb-10B on Hugging Face).

Writes OUT_DIR/queries_fineweb.parquet with
    query_id          int64   (equal to msmarco_query_id for this set)
    msmarco_config    string  \\  the key the public release names each query by
    msmarco_split     string  |
    msmarco_query_id  string  /
    sparse_embedding  struct<indices: list<uint32>, values: list<float>>

Rows are selected by query_set, never by query_id alone: the shared file reuses query ids
across query sets for different queries.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

SET = "dense_sparse"
ROWS = 10000
COLS = ["query_id", "msmarco_config", "msmarco_split", "msmarco_query_id", "sparse_embedding"]


def main(src: str, out_dir: str) -> None:
    fs, path = (pafs.FileSystem.from_uri(src) if "://" in src
                else (pafs.LocalFileSystem(), src))
    t = pq.read_table(path, filesystem=fs, columns=["query_set"] + COLS)
    print(f"{src}: {t.num_rows} rows")
    t = t.filter(pc.equal(t["query_set"], SET)).select(COLS)
    if t.num_rows != ROWS:
        raise SystemExit(f"query_set {SET!r}: {t.num_rows} rows, expected {ROWS}")
    qid = t["query_id"].to_numpy()
    if len(np.unique(qid)) != len(qid):
        raise SystemExit(f"query_id is not unique within query_set {SET!r}")
    nnz = pc.list_value_length(pc.struct_field(t["sparse_embedding"], "indices"))
    if pc.min(nnz).as_py() == 0:
        raise SystemExit("a query has an empty sparse embedding")
    out = f"{out_dir}/queries_fineweb.parquet"
    pq.write_table(t, out, row_group_size=2048)
    print(f"wrote {out} ({t.num_rows} rows, nnz min {pc.min(nnz).as_py()} max {pc.max(nnz).as_py()})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
