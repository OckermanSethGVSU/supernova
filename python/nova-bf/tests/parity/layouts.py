"""Corpus VARIANTS for the parity harness: storage dtypes, file layouts, edge rows.

`corpus.py` builds one shared, well-behaved dataset — float32 dense, one row
group per file, ordinary signed sparse weights, 2..6-token multivectors. That
is the right fixture for the filter x modality matrix and the wrong one for
every code path that only wakes up on a DIFFERENT layout: a float16 corpus, a
file split into row groups, a 0-row shard, non-negative sparse weights (the
zero-gate), a 1-token or 300-token document, an id derived from the file name
instead of read from a column. Those paths are what this module builds data
for, one small dataset per variant, each with its own in-memory truth.

Same contract as the rest of the harness:

  * the truth (`LayoutDS.docs` / `.queries`) is in nova-bf's GLOBAL ROW ORDER
    (sorted file path, then row within file), in the `naive.Doc` form the
    plain-Python oracle reads — so `naive.topk` scores it directly;
  * the truth holds the values AS STORED. A float16 corpus's truth is the
    float16-ROUNDED values (rounded here with numpy's IEEE half cast, never by
    asking nova-bf), because those are the numbers nova-bf is obliged to
    score. Scoring the unrounded float32 originals would demand an accuracy
    the stored file cannot carry;
  * nothing here imports a scoring, id or decoding helper from `nova_bf`. The
    one place that matters most is `expected_point_id`, which re-derives
    nova-load's `vf_point_id` from its DEFINITION (md5 of `filename:row`,
    formatted as a UUID) with `hashlib` — importing `nova_bf.ids` would turn
    the check into "make_point_id equals make_point_id".

`run` is a sibling of `runner.run` rather than a call to it, because these
variants need what the shared runner hard-codes: `corpus.id_column` unset (the
md5 id path), extra corpus/queries keys (`include`/`exclude`), and the raw
string ids back rather than ints.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import compare, naive

if os.environ.get("NOVA_BF_PARITY_SHOW_SRC"):
    # Used by the mutation-testing driver to PROVE which nova_bf a run
    # imported: pytest's `pythonpath = ["src"]` ini setting is inserted ahead
    # of PYTHONPATH, so a mutant copy is only exercised when the driver also
    # overrides `-o pythonpath=...`. A mutation "caught" by the real source
    # would be a lie, so the driver greps for this line.
    import nova_bf as _nb

    print(f"\n[layouts] nova_bf imported from {_nb.__file__}", flush=True)

SPARSE_TYPE = pa.struct([
    pa.field("indices", pa.list_(pa.uint32())),
    pa.field("values", pa.list_(pa.float32())),
])
MV_TYPE = pa.list_(pa.list_(pa.float32()))


# ------------------------------------------------------------- arrow builders


def dense_array(rows, dtype=pa.float32(), fixed: bool = False) -> pa.Array:
    rows = [None if r is None else [float(x) for x in r] for r in rows]
    if fixed:
        width = len(next(r for r in rows if r is not None))
        return pa.array(rows, type=pa.list_(dtype, width))
    return pa.array(rows, type=pa.list_(dtype))


def sparse_array(rows) -> pa.Array:
    """`rows` are `(indices, values)` pairs, or `None` for a NULL row. The
    pairs are written EXACTLY as given — duplicates, zeros and order kept —
    because the raw form is what nova-bf's coalescing must handle."""
    return pa.array(
        [None if r is None else {"indices": [int(i) for i in r[0]],
                                 "values": [float(v) for v in r[1]]}
         for r in rows],
        type=SPARSE_TYPE,
    )


def mv_array(docs) -> pa.Array:
    return pa.array(
        [None if d is None else [[float(x) for x in t] for t in d] for d in docs],
        type=MV_TYPE,
    )


def as_stored(values, dtype: str) -> list[float]:
    """`values` as the corpus file holds them, widened to Python floats.

    float16 rounds with numpy's IEEE round-to-nearest-even half cast (what
    pyarrow writes for a float16 column); float32 likewise; float64 is
    NARROWED to float32 because that is the documented scoring width
    (`io.dense_to_2d` keeps float16/float32 and converts everything else to
    float32), so the obligation on a float64 corpus is to score its float32
    rounding."""
    arr = np.asarray(values, dtype=np.float64)
    if dtype == "float16":
        arr = arr.astype(np.float16)
    elif dtype in ("float32", "float64"):
        arr = arr.astype(np.float32)
    else:
        raise ValueError(dtype)
    return [float(x) for x in arr.astype(np.float64)]


# -------------------------------------------------------------------- dataset


@dataclass
class LayoutDS:
    tmp: str
    corpus_dir: str
    queries_path: str
    docs: list[naive.Doc]
    queries: list[dict]
    files: list[str]                 # absolute corpus file paths, sorted
    file_rows: list[int]
    date_fields: dict = field(default_factory=dict)
    query_date_fields: dict = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    def file_of(self, row: int) -> tuple[str, int]:
        """Global row -> (file path, row within file)."""
        base = 0
        for path, n in zip(self.files, self.file_rows):
            if row < base + n:
                return path, row - base
            base += n
        raise IndexError(row)


def build(tmp, name: str, file_tables: list[pa.Table], docs: list[naive.Doc],
          query_table: pa.Table, queries: list[dict], *,
          row_group_size: int | None = None,
          file_names: list[str] | None = None) -> LayoutDS:
    """Write `file_tables` (one parquet each) + the queries file under
    `tmp/name`. `docs` must already be in global row order."""
    root = os.path.join(str(tmp), name)
    cdir = os.path.join(root, "corpus")
    os.makedirs(cdir, exist_ok=True)
    names = file_names or [f"part-{i:03d}.parquet" for i in range(len(file_tables))]
    paths = []
    for fname, t in zip(names, file_tables):
        p = os.path.join(cdir, fname)
        kw = {} if row_group_size is None else {"row_group_size": row_group_size}
        pq.write_table(t, p, **kw)
        paths.append(p)
    order = sorted(range(len(paths)), key=lambda i: paths[i])
    assert order == list(range(len(paths))), (
        "file_tables must be given in sorted-path order, or `docs` would not "
        "be in nova-bf's global row order")
    qpath = os.path.join(root, "queries.parquet")
    pq.write_table(query_table, qpath)
    assert sum(t.num_rows for t in file_tables) == len(docs)
    return LayoutDS(tmp=root, corpus_dir=cdir, queries_path=qpath, docs=docs,
                    queries=queries, files=paths,
                    file_rows=[t.num_rows for t in file_tables])


def row_groups(path: str) -> int:
    return pq.ParquetFile(path).metadata.num_row_groups


# ---------------------------------------------------------------------- runs


def build_config(ds: LayoutDS, specs: list[dict], *, tag: str,
                 params: dict | None = None, corpus: dict | None = None,
                 queries: dict | None = None):
    from nova_bf.config import BruteForceConfig, _normalize_static_date_bounds

    corpus_cfg = {"path": ds.corpus_dir, "id_column": "rid"}
    corpus_cfg.update(corpus or {})
    corpus_cfg = {k: v for k, v in corpus_cfg.items() if v is not None}
    data = {
        "corpus": corpus_cfg,
        "queries": {"path": ds.queries_path, "id_column": "qid", **(queries or {})},
        "output": {"path": os.path.join(ds.tmp, f"out_{tag}")},
        "params": {"io_workers": 2, **(params or {})},
        "searches": specs,
    }
    return BruteForceConfig.model_validate(_normalize_static_date_bounds(data))


def run(ds: LayoutDS, specs: list[dict], *, tag: str, device: str | None = "cpu",
        params: dict | None = None, corpus: dict | None = None,
        env: dict | None = None, raw_ids: bool = False):
    """`{search: {query_index: [(id, score), …]}}`, rank order kept.

    ids are ints (the global row, read back from the `rid` column every
    builder writes) unless `raw_ids`, which returns the strings nova-bf wrote —
    what the md5-id tests need."""
    from nova_bf.compute import run_compute

    from .runner import env as env_ctx
    from .runner import pinned_device

    cfg = build_config(ds, specs, tag=f"{tag}_{device or 'auto'}", params=params,
                       corpus=corpus)
    with env_ctx(**(env or {})), pinned_device(device):
        paths = run_compute(cfg)
    out = {}
    for name, path in paths.items():
        t = pq.read_table(path).to_pydict()
        out[name] = {
            int(qid): [((s_id if raw_ids else int(s_id)), float(sc))
                       for s_id, sc in zip(ids, scores)]
            for qid, ids, scores in zip(t["query_id"], t["hit_ids"], t["hit_scores"])
        }
    return out


def spec(name, *, vector_type="dense", metric="dot", k=10, filter=None, rows=None):
    out = {"name": name, "vector_type": vector_type, "metric": metric, "k": k}
    if filter is not None:
        out["filter"] = filter
    if rows is not None:
        out["rows"] = rows
    return out


# ---------------------------------------------------------------- the oracle


def naive_topk(ds: LayoutDS, *, vector_type: str, metric: str, k: int,
               keep=None, queries=None, docs=None) -> dict[int, list]:
    """`naive.topk` per query. `keep(doc, query) -> bool` is an optional
    corpus predicate standing in for a filter (these variants use only
    trivially-decidable ones, written out in Python at the call site)."""
    docs = ds.docs if docs is None else docs
    out = {}
    for qi in (range(len(ds.queries)) if queries is None else queries):
        q = ds.queries[qi]
        eligible = docs if keep is None else [d for d in docs if keep(d, q)]
        out[qi] = naive.topk(q[vector_type], eligible, vector_type=vector_type,
                             metric=metric, k=k)
    return out


def assert_topk(got, want, *, metric: str, label: str, tol: float | None = None):
    """The strict form every test in the layout files uses.

    `compare.assert_scores_agree` is boundary-aware about WHICH ids appear,
    which is right, but on its own it accepts a result that is too SHORT
    when the missing tail sits near the other side's floor, a result with a
    repeated id (its dict collapses them), and an out-of-order list. Each of
    those is a wrong ground truth, so this adds, before delegating:

      * the same length (a correct engine returns min(k, eligible)
        candidates, and the oracle computed exactly that);
      * no duplicate ids;
      * scores non-increasing (larger is nearer, for every metric);
      * both-empty is an agreement (a query with no candidate at all), not
        an error.
    """
    ids = [i for i, _ in got]
    assert len(ids) == len(set(ids)), f"{label}: duplicate ids in result {ids}"
    scores = [s for _, s in got]
    for a, b in zip(scores, scores[1:]):
        assert a >= b or (math.isnan(a) and math.isnan(b)), (
            f"{label}: scores are not non-increasing: {scores}")
    assert len(got) == len(want), (
        f"{label}: returned {len(got)} hits, the oracle has {len(want)} "
        f"eligible-and-ranked\n  got:  {got[:12]}\n  want: {want[:12]}")
    if not got and not want:
        return
    compare.assert_scores_agree(got, want, metric=metric, label=label, tol=tol)


def assert_all(got_by_q, want_by_q, *, metric, label, tol=None):
    assert set(got_by_q) == set(want_by_q), (
        f"{label}: query sets differ {sorted(got_by_q)} vs {sorted(want_by_q)}")
    for qi in sorted(want_by_q):
        assert_topk(got_by_q[qi], want_by_q[qi], metric=metric,
                    label=f"{label} q{qi}", tol=tol)


# ----------------------------------------------------------------- point ids


def expected_point_id(filename: str, row: int) -> str:
    """nova-load's `vf_point_id` re-stated from its definition (AGENTS.md:
    `vf_uuid_from_hex(md5(filename || ':' || row))`): md5 over the UTF-8 of
    `<filename>:<row>`, hex, cut 8-4-4-4-12. `filename` is the loader's key,
    which for a LOCAL corpus is the file's absolute path."""
    h = hashlib.md5((filename + ":" + str(int(row))).encode("utf-8")).hexdigest()
    return "-".join((h[0:8], h[8:12], h[12:16], h[16:20], h[20:32]))


# ------------------------------------------------------------------- qdrant


def qdrant_collection(client, ds: LayoutDS, *, dense: dict | None = None,
                      mv: dict | None = None, sparse: tuple[str, ...] = (),
                      payload_index: dict | None = None) -> str:
    """One collection for one layout variant, points keyed by global row.

    `dense` / `mv`: `{vector_name: (distance, dim)}`. `sparse`: vector names;
    a name ending in `_cos` stores each vector pre-scaled to unit norm (Qdrant
    has no sparse cosine; a dot over unit vectors IS cosine). Each doc's
    sparse vector is `doc.sparse` — the COALESCED vector — unless the doc's
    payload carries `_qdrant_sparse` (underscore keys never reach Qdrant's
    payload). Coalescing is forced, not chosen: Qdrant 1.18.2 REJECTS a sparse
    vector with a repeated index (HTTP 422 "indices must be unique",
    measured), so the raw duplicate-index form cannot be given to it at all.
    """
    import uuid

    from qdrant_client import models as M

    name = f"nova_bf_layout_{uuid.uuid4().hex[:8]}"
    vcfg = {}
    for vname, (dist, dim) in (dense or {}).items():
        vcfg[vname] = M.VectorParams(size=dim, distance=dist)
    for vname, (dist, dim) in (mv or {}).items():
        vcfg[vname] = M.VectorParams(
            size=dim, distance=dist,
            multivector_config=M.MultiVectorConfig(
                comparator=M.MultiVectorComparator.MAX_SIM))
    client.create_collection(
        name, vectors_config=vcfg,
        sparse_vectors_config={s: M.SparseVectorParams() for s in sparse})
    for fname, schema in (payload_index or {}).items():
        client.create_payload_index(name, field_name=fname, field_schema=schema,
                                    wait=True)
    points = []
    for d in ds.docs:
        vec = {}
        for vname in dense or {}:
            vec[vname] = d.dense
        for vname in mv or {}:
            if d.multivector:
                vec[vname] = d.multivector
        for vname in sparse:
            sv = d.payload.get("_qdrant_sparse", d.sparse)
            if sv is not None and vname.endswith("_cos"):
                sv = sparse_cosine_store(sv)
            if sv is not None:
                items = sorted(sv.items())
                vec[vname] = M.SparseVector(indices=[t for t, _ in items],
                                            values=[float(w) for _, w in items])
        payload = {k: v for k, v in d.payload.items() if not k.startswith("_")}
        points.append(M.PointStruct(id=d.row, vector=vec, payload=payload))
    for s in range(0, len(points), 128):
        client.upsert(name, points=points[s:s + 128], wait=True)
    return name


def qdrant_topk(client, coll: str, ds: LayoutDS, *, using: str, vector_type: str,
                k: int, qfilter=None, normalize_sparse: bool = False,
                sign: float = 1.0):
    from qdrant_client import models as M

    out = {}
    for qi, q in enumerate(ds.queries):
        if vector_type == "sparse":
            vec = q["sparse"]
            if normalize_sparse:
                n = math.sqrt(sum(w * w for w in vec.values()))
                vec = vec if n == 0 else {t: w / n for t, w in vec.items()}
            items = sorted(vec.items())
            query = M.SparseVector(indices=[t for t, _ in items],
                                   values=[float(w) for _, w in items])
        elif vector_type == "multivector":
            query = q["multivector"]
        else:
            query = q["dense"]
        pts = client.query_points(
            coll, query=query, using=using, limit=k, query_filter=qfilter,
            search_params=M.SearchParams(exact=True), with_payload=False).points
        out[qi] = [(int(p.id), sign * float(p.score)) for p in pts]
    return out


def sparse_cosine_store(vec: dict[int, float]) -> dict[int, float]:
    n = math.sqrt(sum(w * w for w in vec.values()))
    return dict(vec) if n == 0 else {t: w / n for t, w in vec.items()}
