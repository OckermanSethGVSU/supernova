"""The live-Qdrant oracle: one collection, every modality, exact search.

All five scored configurations nova-bf can express live as NAMED VECTORS on
the SAME points of ONE collection:

    dense_dot / dense_cos / dense_euc   dense, the three metrics
    mv_dot / mv_cos                     multivector, MaxSim comparator
    sparse_dot / sparse_cos             sparse

so the payload — and therefore every filter — is shared across modalities and
a filter × modality cross-product costs one upsert instead of one per cell.

Two things Qdrant cannot do natively, and how they are handled rather than
skipped:

  * **sparse cosine.** Qdrant has no cosine distance for sparse vectors, only
    dot. `sparse_cos` therefore stores each vector pre-scaled to unit norm, so
    a dot product over it IS cosine similarity. Queries against it are scaled
    the same way. This is an identity, not an approximation.
  * **euclidean sign.** Qdrant's `EUCLID` score is a distance (smaller =
    nearer); nova-bf negates it so larger is always nearer. `_topk` applies
    that negation, so every score this module returns is already in nova-bf's
    convention and comparable without further thought at the call site.

Every search runs with `exact=True`, so what comes back is Qdrant's true
brute-force ranking rather than an HNSW approximation — otherwise a
disagreement would be unattributable between "nova-bf is wrong" and "the index
missed a neighbour".
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from . import corpus as corpus_mod

DENSE_VECTORS = {"dot": "dense_dot", "cosine": "dense_cos", "euclidean": "dense_euc"}
MV_VECTORS = {"dot": "mv_dot", "cosine": "mv_cos"}
SPARSE_VECTORS = {"dot": "sparse_dot", "cosine": "sparse_cos"}

# Payload fields, and the index each needs. `title` MUST have a text index —
# `MatchText` against an unindexed field is not a slow query, it is an error —
# and the tokenizer/lowercase settings here are the ones `nova_bf.tokenize`
# documents itself as matching.
_INDEXES = {
    "language": models.PayloadSchemaType.KEYWORD,
    "category": models.PayloadSchemaType.KEYWORD,
    "tier": models.PayloadSchemaType.KEYWORD,
    "views": models.PayloadSchemaType.INTEGER,
    "rating": models.PayloadSchemaType.FLOAT,
    "published_at": models.PayloadSchemaType.DATETIME,
    # the edge columns (see corpus.EDGE_SEED)
    "event_at": models.PayloadSchemaType.DATETIME,
    "score_i": models.PayloadSchemaType.INTEGER,
    "flag": models.PayloadSchemaType.BOOL,
    "weight": models.PayloadSchemaType.FLOAT,
    "lang_dict": models.PayloadSchemaType.KEYWORD,
}
# Full-text fields, all with the tokenizer settings `nova_bf.tokenize`
# documents itself as matching.
_TEXT_FIELDS = ("title", "note")


def create_collection(client: QdrantClient, ds) -> str:
    name = f"nova_bf_parity_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name,
        vectors_config={
            "dense_dot": models.VectorParams(
                size=corpus_mod.DIM, distance=models.Distance.DOT),
            "dense_cos": models.VectorParams(
                size=corpus_mod.DIM, distance=models.Distance.COSINE),
            "dense_euc": models.VectorParams(
                size=corpus_mod.DIM, distance=models.Distance.EUCLID),
            "mv_dot": models.VectorParams(
                size=corpus_mod.MV_DIM, distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM)),
            "mv_cos": models.VectorParams(
                size=corpus_mod.MV_DIM, distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM)),
        },
        sparse_vectors_config={
            "sparse_dot": models.SparseVectorParams(),
            "sparse_cos": models.SparseVectorParams(),
        },
    )
    for text_field in _TEXT_FIELDS:
        client.create_payload_index(
            name, field_name=text_field,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
            ),
            wait=True,
        )
    for field, schema in _INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema,
                                    wait=True)

    points = []
    for doc in ds.docs:
        vectors = {
            "dense_dot": doc.dense,
            "dense_cos": doc.dense,
            "dense_euc": doc.dense,
            "sparse_dot": _sparse_vector(doc.sparse),
            "sparse_cos": _sparse_vector(
                corpus_mod.l2_normalized_sparse(doc.sparse)),
        }
        # A point with no (or an empty) multivector simply omits those named
        # vectors — which is exactly what makes it a non-candidate for a
        # multivector search while staying a normal dense/sparse point, the
        # behaviour nova-bf has to reproduce.
        if doc.multivector:
            vectors["mv_dot"] = doc.multivector
            vectors["mv_cos"] = doc.multivector
        points.append(models.PointStruct(
            id=doc.row, vector=vectors, payload=dict(doc.payload)))
    for start in range(0, len(points), 128):
        client.upsert(name, points=points[start:start + 128], wait=True)
    return name


def _sparse_vector(vec: dict[int, float]) -> models.SparseVector:
    """Qdrant wants each token id ONCE. The corpus deliberately stores rows
    with a repeated index (one dimension, summed) — `naive.sparse_from_pairs`
    has already coalesced them here, so what Qdrant is given is the same
    mathematical vector nova-bf's own coalescing must produce."""
    items = sorted(vec.items())
    return models.SparseVector(indices=[t for t, _ in items],
                               values=[float(w) for _, w in items])


# ----------------------------------------------------------- filter transla-
# tion: a RAW filter dict (exactly as a YAML config writes it) -> Qdrant's own
# filter language, resolved for ONE query (per-query conditions become that
# query's literal). Never nova-bf's parsed `Filter` model: its date bounds are
# already nova-bf-converted epoch µs and its range bounds already floats, and
# grading nova-bf's conversion with nova-bf's conversion is no grade at all. A
# date bound reaches Qdrant as the very RFC-3339 string the config holds.

# A condition that matches NO point, for a per-query condition whose value is
# null for this query (nova-bf: matches nothing). `has_id: []` is the
# measured way to say it: under `must` it matches nothing, under `must_not`
# it excludes nothing, under `should` it contributes nothing.
_NOTHING = models.HasIdCondition(has_id=[])


def to_qdrant_filter(filt, query_row: dict, date_fields: dict[str, str]):
    """`None` for no filter. Per-query conditions are resolved against
    `query_row` here — Qdrant has no notion of a filter that varies per query,
    so the translation is per query by construction, and that is precisely the
    check: nova-bf's single fused pass over `(n_queries, rows)` must agree,
    query by query, with N independently-filtered Qdrant searches.

    `filt` is the raw dict (preferred) or an object exposing it as `.raw`."""
    if filt is None:
        return None
    raw = getattr(filt, "raw", filt)
    if not isinstance(raw, dict):
        raise TypeError(
            "qdrant_ref translates RAW filter dicts only, not nova-bf's parsed "
            f"Filter model (got {type(raw).__name__})")
    kwargs = {}
    for group in ("must", "should", "must_not"):
        conds = [_condition(c, query_row, date_fields) for c in raw.get(group) or ()]
        if conds:
            kwargs[group] = conds
    return models.Filter(**kwargs) if kwargs else None


def _condition(cond: dict, query_row: dict, date_fields: dict[str, str]):
    field = cond["field"]
    is_date = field in date_fields
    if cond.get("match") is not None:
        return _match(field, cond["match"])
    if cond.get("match_from_query") is not None:
        return _match(field, query_row[cond["match_from_query"]])
    if cond.get("match_text") is not None:
        return models.FieldCondition(
            key=field, match=models.MatchText(text=cond["match_text"]))
    if cond.get("match_text_from_query") is not None:
        phrase = query_row[cond["match_text_from_query"]]
        if not isinstance(phrase, str):
            return _NOTHING
        return models.FieldCondition(key=field, match=models.MatchText(text=phrase))
    if cond.get("range") is not None:
        return _range(field, dict(cond["range"]), is_date)
    if cond.get("range_from_query") is not None:
        bounds = {}
        for b, col in cond["range_from_query"].items():
            if col is None:
                continue
            v = query_row[col]
            if v is None:
                return _NOTHING      # a null per-query bound never matches
            bounds[b] = v
        return _range(field, bounds, is_date)
    raise ValueError(f"cannot translate condition {cond!r}")


def _match(field: str, value):
    if value is None:
        return _NOTHING
    if isinstance(value, (list, tuple)):
        # A null ELEMENT is simply absent (it can equal nothing); an empty list
        # is sent as-is — Qdrant's MatchAny([]) matches nothing (measured).
        return models.FieldCondition(
            key=field, match=models.MatchAny(any=[v for v in value if v is not None]))
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


def _range(field: str, bounds: dict, is_date: bool):
    bounds = {b: v for b, v in bounds.items() if v is not None}
    if is_date:
        # The raw literal: an RFC-3339 string from the config or the queries
        # column, handed to Qdrant's own parser. A numeric bound on a date
        # field would be nova-bf's µs convention — not something Qdrant can
        # be told — so it is refused rather than silently converted here.
        for b, v in bounds.items():
            if not isinstance(v, str):
                raise TypeError(f"date bound {b}={v!r} on {field!r} is not a string")
        return models.FieldCondition(key=field, range=models.DatetimeRange(**bounds))
    return models.FieldCondition(key=field, range=models.Range(**bounds))


def count(client, collection, filt, query_row: dict, date_fields: dict[str, str]) -> int:
    """How many points Qdrant itself says satisfy `filt` for this query — an
    eligibility count independent of both scoring and of the naive oracle."""
    return client.count(
        collection, count_filter=to_qdrant_filter(filt, query_row, date_fields),
        exact=True).count


# ------------------------------------------------------------------ querying


def topk(client, collection, ds, *, vector_type, metric, k, filt=None,
         queries=None):
    """`{query_index: [(point_id, score), …]}`, best-first, in nova-bf's
    larger-is-nearer convention."""
    indices = range(len(ds.queries)) if queries is None else queries
    out = {}
    for qi in indices:
        q = ds.queries[qi]
        using, vector = _query_vector(q, vector_type, metric)
        pts = client.query_points(
            collection,
            query=vector,
            using=using,
            limit=k,
            query_filter=to_qdrant_filter(filt, q["payload"], ds.date_fields),
            search_params=models.SearchParams(exact=True),
            with_payload=False,
        ).points
        sign = -1.0 if (vector_type == "dense" and metric == "euclidean") else 1.0
        out[qi] = [(int(p.id), sign * float(p.score)) for p in pts]
    return out


def _query_vector(query: dict, vector_type: str, metric: str):
    if vector_type == "dense":
        return DENSE_VECTORS[metric], list(query["dense"])
    if vector_type == "multivector":
        return MV_VECTORS[metric], [list(t) for t in query["multivector"]]
    if vector_type == "sparse":
        vec = query["sparse"]
        if metric == "cosine":
            vec = corpus_mod.l2_normalized_sparse(vec)
        return SPARSE_VECTORS[metric], _sparse_vector(vec)
    raise ValueError(f"unknown vector_type {vector_type!r}")
