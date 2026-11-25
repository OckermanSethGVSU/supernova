#!/usr/bin/env python3
"""Split a PubMed-VE queries file into what the runs read and the GT.

    python prep_queries_pubmed.py SRC OUT_DIR MODALITY

SRC: ${READ}/queries/bf_queries_<...>_k1000.parquet (10,000 rows: query_id, query, doc_id,
category, the three bge_m3_* embeddings, hit_ids (PMIDs), hit_snippet_ids, hit_scores).
MODALITY: dense | sparse | multivector (the bge_m3_<MODALITY>_embedding column).

Writes, rows in SRC order:
  OUT_DIR/queries_pubmed_<m>.parquet  query_id, bge_m3_<MODALITY>_embedding
  OUT_DIR/gt_pubmed_<m>.parquet       query_id, hit_ids, hit_scores
with <m> = dense | sparse | mv. The GT's hit_ids are the file's hit_snippet_ids: the corpus is
keyed by the MedRAG snippet id (its `id` column), not by PMID, and several snippets share a PMID.
"""
import os
import sys

import pyarrow.fs as pafs
import pyarrow.parquet as pq

SHORT = {"dense": "dense", "sparse": "sparse", "multivector": "mv"}


def main():
    src, out, mod = sys.argv[1], sys.argv[2], sys.argv[3]
    if mod not in SHORT:
        raise SystemExit(f"MODALITY: one of {sorted(SHORT)}, not {mod!r}")
    col = f"bge_m3_{mod}_embedding"
    os.makedirs(out, exist_ok=True)
    fs, path = pafs.FileSystem.from_uri(src)
    t = pq.read_table(path, filesystem=fs,
                      columns=["query_id", col, "hit_snippet_ids", "hit_scores"])
    if t.column("query_id").null_count:
        raise SystemExit("null query_id")
    if len(set(t.column("query_id").to_pylist())) != t.num_rows:
        raise SystemExit("duplicate query_id")
    m = SHORT[mod]
    pq.write_table(t.select(["query_id", col]), os.path.join(out, f"queries_pubmed_{m}.parquet"))
    gt = t.select(["query_id", "hit_snippet_ids", "hit_scores"]) \
          .rename_columns(["query_id", "hit_ids", "hit_scores"])
    pq.write_table(gt, os.path.join(out, f"gt_pubmed_{m}.parquet"))
    lens = {len(x) for x in gt.column("hit_ids").to_pylist()[:100]}
    print(f"queries {t.num_rows}; GT lists (first 100 rows) of length {sorted(lens)}")


if __name__ == "__main__":
    main()
