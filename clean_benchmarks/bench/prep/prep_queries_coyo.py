#!/usr/bin/env python3
"""Split Coyo-VE's queries_gt_k1000.parquet into what the runs read and the GT.

    python prep_queries_coyo.py SRC OUT_DIR

SRC: ${READ}/queries/queries_gt_k1000.parquet (the dense_coyo experiment's READ)
(25,000 rows: query_id, query, id, qwen_multimodal_embedding, hit_ids, hit_scores).

Writes, rows in SRC order (nova-bf keeps the queries file's order; the harness
takes the first --nq rows in file order -- so row i is the same query in both):
  OUT_DIR/queries_25k.parquet  query_id:string, qwen_multimodal_embedding:list<float>
  OUT_DIR/gt_25k.parquet       query_id, hit_ids:list<string> (COYO image ids), hit_scores
"""
import os
import sys

import pyarrow.fs as pafs
import pyarrow.parquet as pq


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    fs, path = pafs.FileSystem.from_uri(src)
    t = pq.read_table(path, filesystem=fs,
                      columns=["query_id", "qwen_multimodal_embedding", "hit_ids", "hit_scores"])
    if t.column("query_id").null_count:
        raise SystemExit("null query_id")
    if len(set(t.column("query_id").to_pylist())) != t.num_rows:
        raise SystemExit("duplicate query_id")
    pq.write_table(t.select(["query_id", "qwen_multimodal_embedding"]),
                   os.path.join(out, "queries_25k.parquet"))
    pq.write_table(t.select(["query_id", "hit_ids", "hit_scores"]),
                   os.path.join(out, "gt_25k.parquet"))
    lens = {len(x) for x in t.column("hit_ids").to_pylist()[:100]}
    print(f"queries {t.num_rows}; GT lists (first 100 rows) of length {sorted(lens)}")


if __name__ == "__main__":
    main()
