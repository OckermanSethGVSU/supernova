#!/usr/bin/env python3
"""Write the first N rows of Pes2o-VE's labelled_queries.parquet as a nova-bf
queries file, plus the ground truth for those rows.

    python prep_queries_pes2o.py SRC OUT_DIR N [N ...]

For each N writes
    OUT_DIR/queries_{tag}.parquet   query_id:int64, query:fixed_size_list<float>[2560]
    OUT_DIR/gt_{tag}.npz            query_id (N,), ids (N,1000) int64, distances (N,1000)
where tag = "20k" for 20000, "500" for 500, else str(N).

The GT stays OUT of the queries parquet so nova-bf's ranged reader does not
pull the 8 KB/row `ids` column into the query load. Rows are taken in file
order (the harness's --nq takes the first N the same way).
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


def tag(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 and n >= 1000 else str(n)


def main(src: str, out_dir: str, ns: list[int]) -> None:
    fs, path = (pafs.FileSystem.from_uri(src) if "://" in src
                else (pafs.LocalFileSystem(), src))
    nmax = max(ns)
    pf = pq.ParquetFile(path, filesystem=fs)
    print(f"{src}: {pf.metadata.num_rows} rows, schema {pf.schema_arrow}")
    want = ["query_id", "query", "ids", "distances"]
    have = set(pf.schema_arrow.names)
    missing = [c for c in want if c not in have]
    if missing:
        raise SystemExit(f"missing columns {missing}; have {sorted(have)}")
    if pf.metadata.num_rows < nmax:
        raise SystemExit(f"only {pf.metadata.num_rows} rows, need {nmax}")
    batches, got = [], 0
    for b in pf.iter_batches(batch_size=4096, columns=want):
        batches.append(b)
        got += b.num_rows
        if got >= nmax:
            break
    t = pa.Table.from_batches(batches).slice(0, nmax)
    qid = t["query_id"].to_numpy()
    if len(np.unique(qid)) != len(qid):
        raise SystemExit("query_id is not unique in the first N rows")
    ids_col = t["ids"].combine_chunks()
    width = ids_col.type.list_size if pa.types.is_fixed_size_list(ids_col.type) else None
    if width is None:
        raise SystemExit(f"expected fixed_size_list ids, got {ids_col.type}")
    ids = ids_col.flatten().to_numpy().reshape(-1, width)
    dist = t["distances"].combine_chunks().flatten().to_numpy(
        zero_copy_only=False).reshape(len(t), -1)
    for n in ns:
        q = t.select(["query_id", "query"]).slice(0, n)
        qp = f"{out_dir}/queries_{tag(n)}.parquet"
        # Several row groups so pyarrow can decode in parallel; uncompressed
        # like the corpus (2560 random floats do not compress).
        pq.write_table(q, qp, compression="none", row_group_size=2048)
        gp = f"{out_dir}/gt_{tag(n)}.npz"
        np.savez(gp, query_id=qid[:n], ids=ids[:n], distances=dist[:n])
        print(f"wrote {qp} ({n} rows) and {gp} (ids {ids[:n].shape})")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
