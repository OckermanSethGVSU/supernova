"""Multivector (MaxSim) at the shapes the shared corpus never produces.

`corpus.py`'s multivectors are 2..6 tokens, one row group per file, no
zero-norm token. This file builds a corpus that has every awkward shape at
once, written with SMALL row groups so each file is read back as a chunked
Arrow column and decoded through `io._stitch_ragged_chunks` (the per-chunk
decode + re-prefix-summed offsets) rather than the single-array path:

  * 1-token documents and a 1-token query (the segmented max over one
    element);
  * 230- and 260-token documents (a single doc larger than a whole slice's
    token budget, which `multivector_budget_mib` must process ALONE) and a
    40-token query;
  * a zero-norm token inside an otherwise normal document, a document made
    only of zero tokens, and a query containing a zero token — under cosine
    these must score 0 for that token (Qdrant's rule, measured: a zero vector
    stays zero under its normalization) rather than NaN;
  * null and empty documents (never candidates), and a 0-row and a 1-row
    file;

each checked against the naive oracle and live Qdrant, under the knobs that
change how the work is cut: tiny pinned tiles (3 docs x 3 queries, test-only
via `mv_tiles`), a tiny `multivector_budget_mib` (64 score elements),
`multivector_double_buffer` (CUDA-only staging ring; on CPU it must be a
no-op — on a GPU box this same test exercises the ring), and a static filter.
`rows:` query subsets are REJECTED for multivector at config validation
(`rows is not supported with vector_type='multivector'`), which
`test_rows_on_multivector_is_rejected_loudly` pins — the audit's "rows never
run on multivector" gap is a missing feature, not missing coverage.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

import mv_tiles

import nova_bf.io as io_mod

from . import layouts, naive

D = 8
FILE_SIZES = (19, 0, 1, 26, 14)
ROW_GROUP = 7
N_Q = 10
K = 8
KALL = 200
METRICS = ("dot", "cosine")
LONG_DOCS = {10: 230, 40: 260}


def _build(tmp):
    rng = np.random.default_rng(4242)
    tables, docs, g = [], [], 0
    for fi, n in enumerate(FILE_SIZES):
        mvs, payload = [], []
        for r in range(n):
            if g in LONG_DOCS:
                mv = rng.standard_normal((LONG_DOCS[g], D))
            elif g % 17 == 3:
                mv = None                                   # null doc
            elif g % 19 == 5:
                mv = np.zeros((0, D))                       # empty doc
            elif g % 4 == 0:
                mv = rng.standard_normal((1, D))            # 1-token doc
            else:
                mv = rng.standard_normal((int(rng.integers(2, 7)), D))
            if mv is not None and g % 7 == 2 and len(mv) > 1:
                mv[1] = 0.0                                 # a zero-norm token
            if g == 23:
                mv = np.zeros((2, D))                       # only zero tokens
            mv = None if mv is None else mv.astype(np.float32)
            mvs.append(mv)
            payload.append({"grp": "abc"[g % 3]})
            docs.append(naive.Doc(
                row=g, payload={"grp": "abc"[g % 3]},
                multivector=None if mv is None else [[float(x) for x in t] for t in mv]))
            g += 1
        tables.append(pa.table({
            "multivector_embedding": layouts.mv_array(mvs),
            "grp": pa.array([p["grp"] for p in payload], type=pa.string()),
            "rid": pa.array([str(g - n + i) for i in range(n)], type=pa.string()),
        }))
    qmv, queries = [], []
    for qi in range(N_Q):
        ntok = {0: 1, 3: 40}.get(qi, int(rng.integers(2, 6)))
        mv = rng.standard_normal((ntok, D)).astype(np.float32)
        if qi == 2:
            mv[0] = 0.0                                     # a zero query token
        qmv.append(mv)
        queries.append({"multivector": [[float(x) for x in t] for t in mv],
                        "payload": {"query_set": "even" if qi % 2 == 0 else "odd"}})
    qt = pa.table({
        "multivector_embedding": layouts.mv_array(qmv),
        "qid": pa.array([str(i) for i in range(N_Q)]),
        "query_set": pa.array([q["payload"]["query_set"] for q in queries]),
    })
    return layouts.build(tmp, "mv_edges", tables, docs, qt, queries,
                         row_group_size=ROW_GROUP)


@pytest.fixture(scope="module")
def mv_ds(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("mvedge"))


def _specs(filter=None, rows=None, tag=""):
    return [layouts.spec(f"mv{tag}_{m}{sfx}", vector_type="multivector", metric=m,
                         k=k, filter=filter, rows=rows)
            for m in METRICS for sfx, k in (("", K), ("_all", KALL))]


STATIC = {"must": [{"field": "grp", "match": ["a", "c"]}]}
EVEN = {"column": "query_set", "isin": ["even"]}
ODD = {"column": "query_set", "isin": ["odd"]}

# knob -> (params, pinned (docs_per_slice, queries_per_tile) or None)
KNOBS = {
    "default": ({}, None),
    "tiny_tiles": ({}, (3, 3)),
    # Far below one long document's 230 x 40 token product, so the budget
    # must split the corpus axis per doc and process each long doc alone.
    "token_budget": ({"multivector_budget_mib": mv_tiles.mib(64)}, None),
    "double_buffer": ({"multivector_double_buffer": True}, (5, 10**6)),
    "prune_off": ({"multivector_prune": "off"}, (5, 10**6)),
}


@pytest.fixture(scope="module")
def mv_runs(mv_ds, device):
    """One run per knob set, each carrying the unfiltered searches and a
    filtered pair sharing the pass."""
    out = {}
    mp = pytest.MonkeyPatch()
    try:
        calls = {"stitch": 0}
        real = io_mod._stitch_ragged_chunks

        def spy(chunks):
            calls["stitch"] += 1
            return real(chunks)

        mp.setattr(io_mod, "_stitch_ragged_chunks", spy)
        for name, (params, tiles) in KNOBS.items():
            calls["stitch"] = 0
            specs = _specs() + _specs(filter=STATIC, tag="f")
            hits = layouts.run(mv_ds, specs, tag=f"mv_{name}", device=device,
                               params=params, mv_tiles=tiles)
            out[name] = (hits, calls["stitch"])
    finally:
        mp.undo()
    return out


def _keep(tag):
    if tag == "f":
        return lambda d, q: d.payload["grp"] in ("a", "c")
    return None


def _queries(tag):
    if tag == "e":
        return [qi for qi in range(N_Q) if qi % 2 == 0]
    if tag == "o":
        return [qi for qi in range(N_Q) if qi % 2 == 1]
    return None


def test_the_fixture_has_the_shapes_it_claims(mv_ds):
    for path in mv_ds.files:
        if pq.ParquetFile(path).metadata.num_rows > ROW_GROUP:
            assert layouts.row_groups(path) > 1, path
    lens = [len(d.multivector) for d in mv_ds.docs if d.multivector]
    assert min(lens) == 1 and max(lens) >= 230
    assert any(d.multivector is None for d in mv_ds.docs)
    assert any(d.multivector == [] for d in mv_ds.docs)


@pytest.mark.parametrize("knob", list(KNOBS))
def test_multi_row_group_files_go_through_the_stitch(knob, mv_runs):
    """Every file with more than one row group must be decoded chunk by
    chunk; if the reader ever combined chunks first this test — not a wrong
    answer — is what would say the stitch is no longer under test."""
    _, n_stitch = mv_runs[knob]
    assert n_stitch >= 3, f"{knob}: _stitch_ragged_chunks ran {n_stitch} times"


CELLS = [(knob, tag, m, sfx) for knob in KNOBS for tag in ("", "f")
         for m in METRICS for sfx in ("", "_all")]


@pytest.mark.parametrize("knob,tag,metric,sfx", CELLS,
                         ids=[f"{a}-{b or 'nf'}-{c}{d}" for a, b, c, d in CELLS])
def test_mv_edges_agree_with_the_naive_oracle(knob, tag, metric, sfx, mv_ds, mv_runs, device):
    hits, _ = mv_runs[knob]
    got = hits[f"mv{tag}_{metric}{sfx}"]
    want = layouts.naive_topk(mv_ds, vector_type="multivector", metric=metric,
                              k=K if sfx == "" else KALL, keep=_keep(tag),
                              queries=_queries(tag))
    layouts.assert_all(got, want, metric=metric,
                       label=f"[{device}] mv {knob}/{tag or 'nf'}/{metric}{sfx} vs naive")


@pytest.mark.parametrize("knob", [k for k in KNOBS if k != "default"])
def test_cutting_knobs_do_not_change_the_answer(knob, mv_runs, device):
    """Same ids in the same order as the default cut, scores to tolerance
    (a different cut sums the same products in a different order)."""
    from . import compare

    base, _ = mv_runs["default"]
    other, _ = mv_runs[knob]
    for name in base:
        metric = "dot" if "_dot" in name else "cosine"
        for qi in base[name]:
            a, b = other[name][qi], base[name][qi]
            layouts.assert_topk(a, b, metric=metric,
                                label=f"[{device}] {knob} vs default {name} q{qi}")


def test_zero_norm_tokens_score_zero_not_nan(mv_runs):
    hits, _ = mv_runs["default"]
    for name, by_q in hits.items():
        for qi, lst in by_q.items():
            assert all(s == s for _, s in lst), f"{name} q{qi}: NaN score"
    # row 23 is all-zero tokens: under cosine its MaxSim is exactly 0
    cos = dict(hits["mv_cosine_all"][1])
    assert 23 in cos and abs(cos[23]) < 1e-6, cos.get(23)


@pytest.fixture(scope="module")
def mv_collection(client, collection, mv_ds):
    from qdrant_client import models as M

    name = layouts.qdrant_collection(
        client, mv_ds, mv={"mv_dot": (M.Distance.DOT, D), "mv_cos": (M.Distance.COSINE, D)},
        payload_index={"grp": M.PayloadSchemaType.KEYWORD})
    yield name
    client.delete_collection(name)


QCELLS = [(knob, tag, m, sfx) for knob in ("default", "token_budget")
          for tag in ("", "f") for m in METRICS for sfx in ("", "_all")]


@pytest.mark.qdrant
@pytest.mark.parametrize("knob,tag,metric,sfx", QCELLS,
                         ids=[f"{a}-{b or 'nf'}-{c}{d}" for a, b, c, d in QCELLS])
def test_mv_edges_agree_with_qdrant(knob, tag, metric, sfx, mv_ds, mv_runs, client,
                                    mv_collection, device):
    from qdrant_client import models as M

    hits, _ = mv_runs[knob]
    qf = (M.Filter(must=[M.FieldCondition(key="grp", match=M.MatchAny(any=["a", "c"]))])
          if tag == "f" else None)
    want = layouts.qdrant_topk(client, mv_collection, mv_ds,
                               using="mv_dot" if metric == "dot" else "mv_cos",
                               vector_type="multivector", k=K if sfx == "" else KALL,
                               qfilter=qf)
    layouts.assert_all(hits[f"mv{tag}_{metric}{sfx}"], want, metric=metric,
                       label=f"[{device}] mv {knob}/{tag or 'nf'}/{metric}{sfx} vs qdrant")


def test_rows_on_multivector_is_rejected_loudly(mv_ds):
    """Pinned: a `rows:` selector on a multivector search is a config error,
    not a silently-ignored key (which would compute every query and report
    it as the subset)."""
    with pytest.raises(Exception, match="rows.*not supported.*multivector"):
        layouts.build_config(mv_ds, _specs(rows=EVEN, tag="e"), tag="mv_rows")
