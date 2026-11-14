"""Sparse: the zero-score gate, and the edge rows real sparse data contains.

THE ZERO-GATE
-------------
A sparse candidate is a document sharing at least one token with the query.
nova-bf decides that per corpus FILE, two ways (`compute._zero_gate_file_ok`):

  * non-negative queries AND a strictly-positive file whose smallest possible
    query x corpus product stays a normal float (`_ZERO_GATE_MIN_PRODUCT`):
    a dot product of 0 can then only mean "no shared token", so the free
    `raw == 0` test IS the overlap test (`_SPARSE_BRANCHES["gate_zero"]`);
  * anything else — a signed value, a STORED zero, a product that could
    underflow — pays for an indicator matmul that counts shared tokens
    exactly (`gate_structural`).

The first is the production path for every SPLADE / mGTE style corpus
(ReLU'd, non-negative weights) and the shared parity corpus never takes it —
its weights are standard-normal. So this file builds non-negative corpora and
checks, per file, which gate was chosen and that the answer is right either
way:

  * `pos`    — strictly positive everywhere: every non-empty file must take
               the cheap gate;
  * `zeros`  — one file stores explicit 0.0 weights, including documents whose
               ONLY shared token is a stored zero. That file must fall back
               (a stored zero is an overlap that dots to 0, which `raw == 0`
               would wrongly drop); the others keep the cheap gate;
  * `tiny`   — one query carries a 1e-25 weight and one file 1e-27-scale
               weights: their products underflow float32 to exactly 0.0, so
               that file must fall back; a second file sits JUST above the
               underflow fence (products ~2e-30, normal) and must keep the
               cheap gate.

each under no filter, a static filter (all searches filtered, so small files
are COALESCED into shared batches — `_concat_sparse_batches` must AND the
per-file gates), a per-query filter, and a starved dense-operand budget
(the chunked scoring path), for dot and cosine, against the naive oracle and
live Qdrant.

QDRANT SEMANTICS (measured on 1.18.2, see the module report)
------------------------------------------------------------
Qdrant's exact sparse search is not one rule but two:

  * UNFILTERED it drops every candidate whose score is exactly 0.0 — a
    stored zero, a cancelling pair, an underflowed product — even though the
    token is shared;
  * under ANY filter (even one admitting every point, indexed or not) it
    keeps them, at 0.0.

nova-bf keeps them in both cases (structural overlap). The filtered half
therefore agrees and is asserted in full; the unfiltered half disagrees on
exactly the stored-zero / underflow rows, so those rows — derived from the
data, not from either engine's output — are removed from both sides and
everything else is still compared, while a separate test pins the divergence
itself (nova-bf has them at 0.0, Qdrant has none, in exactly the measured
cells), so the day either side changes, this file says so. (This replaced a
strict xfail over whole cells, which would also have absorbed any UNRELATED
failure in them.)

EDGE ROWS
---------
`test_edge_*`: empty and null rows, a stored 0.0 weight, a repeated index
(one dimension, summed), a repeated index cancelling to 0.0, token ids at 4e9
and at u32 max, unsorted indices, a query with a repeated index, an empty
query, a query with a zero weight.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("torch")

import nova_bf.compute as compute_mod

from . import layouts, naive

VOCAB = 50
OOV = VOCAB + 5          # a token no query uses
T_ZERO = 3               # the token the `zeros` file stores as 0.0
T_TINY = 7               # the token the tiny query weight sits on
FILE_SIZES = (23, 0, 31, 1, 17, 29)   # includes a 0-row and a 1-row shard
N_Q = 8
K = 12
GROUPS = ("a", "b", "c")
METRICS = ("dot", "cosine")
# k=K is the ordinary top-K; k=KALL exceeds the corpus, so EVERY candidate
# must come back, in order. That is what makes a dropped ~0-score candidate
# (stored zero, underflowed product — always at the bottom of the ranking)
# visible at all: in a k=12 list it would never have been there to drop.
KALL = 200
SPECS = [layouts.spec(f"sp_{m}{sfx}", vector_type="sparse", metric=m, k=k)
         for m in ("dot", "cosine") for sfx, k in (("", 12), ("_all", KALL))]
TOPKS = {"": 12, "_all": KALL}

STATIC = {"must": [{"field": "grp", "match": ["a", "b", "c"]}]}   # admits ALL rows
STATIC_NARROW = {"must": [{"field": "grp", "match": ["a", "c"]}]}
PERQ = {"must": [{"field": "grp", "match_from_query": "q_grp"}]}


def _keep_for(regime):
    if regime == "static":
        return None                     # admits everything
    if regime == "static_narrow":
        return lambda d, q: d.payload["grp"] in ("a", "c")
    if regime == "perquery":
        return lambda d, q: d.payload["grp"] == q["payload"]["q_grp"]
    return None


def _filter_for(regime):
    return {"static": STATIC, "static_narrow": STATIC_NARROW,
            "perquery": PERQ}.get(regime)


# --------------------------------------------------------------- the corpus


def _build(tmp, variant: str):
    """Non-negative SPLADE-like corpus + queries (see module docstring)."""
    rng = np.random.default_rng({"pos": 11, "zeros": 12, "tiny": 13}[variant])
    tables, docs = [], []
    g = 0
    for fi, n in enumerate(FILE_SIZES):
        rows, payload = [], []
        for r in range(n):
            nnz = int(rng.integers(3, 9))
            idx = rng.choice(VOCAB, size=nnz, replace=False).tolist()
            val = rng.uniform(0.05, 3.0, size=nnz).tolist()
            if g % 11 == 0:                      # repeated index, positive sum
                idx.append(idx[0]); val.append(0.25)
            if g % 13 == 0:                      # a token no query has
                idx.append(OOV); val.append(1.5)
            if variant == "zeros" and fi == 2:
                if r % 5 == 0:
                    # ONLY shared token is a stored zero (T_ZERO is in every
                    # query's support, see below): a candidate scoring 0.0.
                    idx, val = [T_ZERO], [0.0]
                elif r % 3 == 0:
                    idx.append(T_ZERO if T_ZERO not in idx else OOV + 1)
                    val.append(0.0)
            if variant == "tiny" and fi == 4:
                # ~1e-12 weights: products against the 1e-34 query weight are
                # ~1e-46, below float32's smallest denormal, so they round to
                # exactly 0.0. (Norms stay >= 2e-12, clear of the 1e-12 norm
                # clamp, so cosine is not perturbed by THAT instead.)
                if r % 2 == 0:
                    idx, val = [T_TINY], [float(rng.uniform(2.0, 5.0)) * 1e-12]
                else:
                    val = [(2.0 + v) * 1e-12 for v in val]
            if variant == "tiny" and fi == 5:
                # Just above the fence: 1e-34 * >=2e4 = >=2e-30 > 1e-30, and a
                # normal float32 (min normal ~1.2e-38).
                val = [2e4 + v * 1e3 for v in val]
                if r % 2 == 0:
                    idx, val = [T_TINY], [2.5e4]
            if g == 5 and variant != "tiny":
                idx, val = [], []                # empty row: never a candidate
            rows.append((idx, val))
            payload.append({"grp": GROUPS[g % 3], "rid": str(g)})
            docs.append(naive.Doc(row=g, payload={"grp": GROUPS[g % 3]},
                                  sparse=naive.sparse_from_pairs(idx, val)))
            g += 1
        if variant != "tiny" and fi == 2 and n:
            # a NULL row in the middle of a file
            rows[4] = None
            docs[g - n + 4].sparse = None
        tables.append(pa.table({
            "sparse_embedding": layouts.sparse_array(rows),
            "grp": pa.array([p["grp"] for p in payload], type=pa.string()),
            "rid": pa.array([p["rid"] for p in payload], type=pa.string()),
        }))

    qrows, queries = [], []
    for qi in range(N_Q):
        nnz = int(rng.integers(4, 10))
        idx = [t for t in rng.choice(VOCAB, size=nnz, replace=False).tolist()
               if t not in (T_ZERO, T_TINY)]
        val = rng.uniform(0.1, 2.0, size=len(idx)).tolist()
        idx.append(T_ZERO); val.append(0.7)      # every query hits T_ZERO
        if qi == 3:
            idx.append(idx[0]); val.append(0.4)  # repeated query index: summed
        if variant == "tiny" and qi == 5:
            idx.append(T_TINY); val.append(1e-34)
        if qi == 7:
            idx, val = [], []                    # empty query: no candidates
        qrows.append((idx, val))
        queries.append({"sparse": naive.sparse_from_pairs(idx, val),
                        "payload": {"q_grp": GROUPS[qi % 3]}})
    qt = pa.table({
        "sparse_embedding": layouts.sparse_array(qrows),
        "qid": pa.array([str(i) for i in range(N_Q)]),
        "q_grp": pa.array([q["payload"]["q_grp"] for q in queries]),
    })
    return layouts.build(tmp, f"sp_{variant}", tables, docs, qt, queries)


VARIANTS = ("pos", "zeros", "tiny")


@pytest.fixture(scope="module")
def sp_ds(tmp_path_factory):
    return {v: _build(tmp_path_factory.mktemp(f"spgate_{v}"), v) for v in VARIANTS}


# The file whose gate MUST fall back, per variant, identified by a property
# of its raw values that the spy records (min value), and the file that sits
# just above the underflow fence.
def _classify(vmin):
    if vmin is None:
        return "empty"
    if vmin == 0.0:
        return "stored_zero"
    if vmin < 1e-9:
        return "underflow"
    if vmin > 1e3:
        return "near_fence"
    return "normal"


@contextmanager
def _gate_spy(monkeypatch, log):
    real = compute_mod._zero_gate_file_ok

    def spy(values, q_nonneg, q_min_pos):
        ok = real(values, q_nonneg, q_min_pos)
        vmin = float(values.min()) if len(values) else None
        log.append((_classify(vmin), bool(ok), bool(q_nonneg)))
        return ok

    real_concat = compute_mod._concat_sparse_batches

    def concat_spy(batches):
        out = real_concat(batches)
        log.append(("group", tuple(bool(b.zero_gate_ok) for b in batches),
                    bool(out.zero_gate_ok)))
        return out

    monkeypatch.setattr(compute_mod, "_zero_gate_file_ok", spy)
    monkeypatch.setattr(compute_mod, "_concat_sparse_batches", concat_spy)
    yield


REGIMES = ("nofilter", "static", "static_narrow", "perquery", "chunked")
TINY_BUDGET = 4 * 60 * 5     # ~5 rows of a <=60-wide query vocab per chunk


def _run(ds, regime, device, kernel="cusparse"):
    f = _filter_for(regime)
    specs = [dict(s, filter=f) if f else dict(s) for s in SPECS]
    log: list = []
    mp = pytest.MonkeyPatch()
    try:
        with _gate_spy(mp, log):
            if regime == "chunked":
                mp.setattr(compute_mod, "_SPARSE_SWAP_MAX_DENSE_BYTES", TINY_BUDGET)
            hits = layouts.run(ds, specs, tag=f"gate_{regime}_{kernel}", device=device,
                               params={"sparse_batch_size": 20, "sparse_kernel": kernel})
            branches = dict(compute_mod._SPARSE_BRANCHES)
    finally:
        mp.undo()
    return hits, branches, log


@pytest.fixture(scope="module")
def gate_runs(sp_ds, device):
    # Pinned to the cuSPARSE path: the gate/branch assertions below are about
    # it, and on CUDA `auto` would take the posting-list kernel instead.
    return {(v, r): _run(sp_ds[v], r, device) for v in VARIANTS for r in REGIMES}


@pytest.fixture(scope="module")
def posting_runs(sp_ds, device):
    """The same runs under `sparse_kernel: auto` (the posting-list kernel on
    CUDA, or under TRITON_INTERPRET; the cuSPARSE path elsewhere)."""
    return {(v, r): _run(sp_ds[v], r, device, kernel="auto")
            for v in VARIANTS for r in REGIMES}


# ------------------------------------------------------------ reachability


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_each_file_took_the_gate_its_values_require(variant, regime, gate_runs, device):
    """Without this every comparison below could pass with the cheap gate
    never taken (all structural) — which is the shared corpus's state today.

    Per file: a strictly-positive file with a safe product floor MUST take
    `raw == 0`; a stored-zero or underflow-risk file MUST NOT."""
    _, branches, log = gate_runs[(variant, regime)]
    kinds = {}
    for kind, ok, q_nonneg in (e for e in log if e[0] != "group"):
        assert q_nonneg, f"{variant}: the query side is non-negative by construction"
        kinds.setdefault(kind, set()).add(ok)
    # The fence is RUN-WIDE on the query side: a 1e-34 query weight puts every
    # ordinary file (min weight 0.05 -> product 5e-36) under it too.
    assert kinds.get("normal") == ({False} if variant == "tiny" else {True}), \
        (variant, regime, kinds)
    if variant == "zeros":
        assert kinds.get("stored_zero") == {False}, kinds
    if variant == "tiny":
        assert kinds.get("underflow") == {False}, kinds
        assert kinds.get("near_fence") == {True}, kinds
    groups = [e for e in log if e[0] == "group"]
    coalesced = regime in ("static", "static_narrow")
    if coalesced:
        # Every sparse search is filtered by a uniform filter and a batch
        # size is set, so small files MUST have been coalesced — otherwise
        # `_concat_sparse_batches`' AND over the per-file gates is untested.
        assert any(len(parts) >= 2 for _, parts, _ in groups), (variant, groups)
        for _, parts, ok in groups:
            assert ok == all(parts), f"coalesced gate {ok} != AND of parts {parts}"
    else:
        assert not groups, f"{regime}: nothing should coalesce ({groups})"
    if variant == "tiny" and coalesced:
        # The near-fence file (gate on) is coalesced with the underflow file
        # (gate off): the group must take the structural gate, so NO slice of
        # this run may use the cheap one.
        assert any(len(set(parts)) == 2 for _, parts, _ in groups), groups
        assert branches["gate_zero"] == 0, branches
    else:
        assert branches["gate_zero"] > 0, (variant, regime, branches)
    if variant == "pos":
        assert branches["gate_structural"] == 0, (regime, branches)
    else:
        assert branches["gate_structural"] > 0, (variant, regime, branches)
    if regime == "chunked":
        assert branches["scored_chunked"] > 0, branches
        assert branches["scored_swapped"] == 0, branches


# -------------------------------------------------------------- the oracle


@pytest.mark.parametrize("sfx", list(TOPKS))
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_gate_agrees_with_the_naive_oracle(variant, regime, metric, sfx, sp_ds,
                                           gate_runs, device):
    """Structural overlap: a stored 0.0 or an underflowed product is still a
    candidate (score ~0), an empty / null row never is, and an empty query
    has no candidates. Length, order and uniqueness are checked too
    (`layouts.assert_topk`), so a dropped zero-score candidate cannot hide
    at the top-K boundary."""
    ds = sp_ds[variant]
    hits, _, _ = gate_runs[(variant, regime)]
    want = layouts.naive_topk(ds, vector_type="sparse", metric=metric, k=TOPKS[sfx],
                              keep=_keep_for(regime))
    layouts.assert_all(hits[f"sp_{metric}{sfx}"], want, metric=metric,
                       label=f"[{device}] {variant}/{regime}/{metric}{sfx} vs naive")


@pytest.mark.parametrize("sfx", list(TOPKS))
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_posting_kernel_agrees_with_the_naive_oracle(variant, regime, metric, sfx, sp_ds,
                                                     posting_runs, device):
    """The `auto` arm: stored zeros, underflow, empty rows and queries, filters
    and coalescing through the posting-list kernel, against the same oracle."""
    ds = sp_ds[variant]
    hits, branches, _ = posting_runs[(variant, regime)]
    import os as _os
    if device == "cuda" or _os.environ.get("TRITON_INTERPRET"):
        assert branches["scored_posting"] > 0, (variant, regime, branches)
    want = layouts.naive_topk(ds, vector_type="sparse", metric=metric, k=TOPKS[sfx],
                              keep=_keep_for(regime))
    layouts.assert_all(hits[f"sp_{metric}{sfx}"], want, metric=metric,
                       label=f"[{device}/auto] {variant}/{regime}/{metric}{sfx} vs naive")


def test_the_fixture_really_has_zero_score_candidates(sp_ds):
    """The stored-zero / underflow rows must actually reach the top-K in the
    oracle, or a mutant that drops them would pass every test above."""
    for variant in ("zeros", "tiny"):
        ds = sp_ds[variant]
        want = layouts.naive_topk(ds, vector_type="sparse", metric="dot", k=KALL)
        n_zero = sum(1 for qi in want for _, s in want[qi] if abs(s) < 1e-40)
        assert n_zero >= 3, f"{variant}: only {n_zero} ~0-score hits in the oracle top-K"


# -------------------------------------------------------------- qdrant


@pytest.fixture(scope="module")
def sp_collections(client, collection, sp_ds):
    """`collection` (the shared one) is requested only so the session's
    Qdrant gate (`conftest._QDRANT["graded"]`) records that the oracle ran —
    this file grades against its own collections."""
    from qdrant_client import models as M

    out = {}
    for v, ds in sp_ds.items():
        out[v] = layouts.qdrant_collection(
            client, ds, sparse=("s_dot", "s_cos"),
            payload_index={"grp": M.PayloadSchemaType.KEYWORD})
    yield out
    for name in out.values():
        client.delete_collection(name)


def _qfilter(regime, q):
    from qdrant_client import models as M

    if regime == "static":
        return M.Filter(must=[M.FieldCondition(key="grp", match=M.MatchAny(any=["a", "b", "c"]))])
    if regime == "static_narrow":
        return M.Filter(must=[M.FieldCondition(key="grp", match=M.MatchAny(any=["a", "c"]))])
    if regime == "perquery":
        return M.Filter(must=[M.FieldCondition(key="grp", match=M.MatchValue(value=q["payload"]["q_grp"]))])
    return None


def _qdrant(client, coll, ds, metric, regime, k):
    out = {}
    for qi, q in enumerate(ds.queries):
        one = layouts.LayoutDS(**{**ds.__dict__, "queries": [q]})
        got = layouts.qdrant_topk(client, coll, one,
                                  using="s_dot" if metric == "dot" else "s_cos",
                                  vector_type="sparse", k=k,
                                  qfilter=_qfilter(regime, q),
                                  normalize_sparse=(metric == "cosine"))
        out[qi] = got[0]
    return out


# Qdrant 1.18.2's UNFILTERED exact sparse search drops every candidate whose
# score is exactly 0.0 (stored zero / underflowed product); nova-bf keeps them
# (structural overlap), as Qdrant itself does under any filter. The unfiltered
# regimes ("chunked" is unfiltered too, with a starved budget) are therefore
# compared with exactly those rows — derived from the DATA by the naive oracle,
# never from nova-bf's output — removed from both sides, and the divergence
# itself is pinned separately (`test_the_qdrant_zero_score_divergence_is_what_it_was`).
# Everything else about those cells, and every other cell, is compared in full.
_UNFILTERED = ("nofilter", "chunked")

# Measured, then pinned: the (variant, metric) pairs whose unfiltered answer
# contains a 0.0-score candidate. `tiny` diverges for dot only: Qdrant's sparse
# cosine is a dot over PRE-NORMALIZED vectors (see `layouts.qdrant_collection`),
# and normalizing lifts the ~1e-12 weights back to O(1), so its products no
# longer underflow.
_DIV_PAIRS = {("zeros", "dot"), ("zeros", "cosine"), ("tiny", "dot")}


def _zero_score_candidates(ds, metric, regime, qi):
    """Rows that ARE candidates (they share a token with the query) but score
    0.0 in float32 — a stored zero, or a product that underflows. Only the
    unfiltered regimes, where Qdrant drops them."""
    if regime not in _UNFILTERED:
        return set()
    want = layouts.naive_topk(ds, vector_type="sparse", metric=metric, k=KALL,
                              queries=[qi])[qi]
    assert len(want) < KALL, "KALL must exceed the candidate count"
    return {r for r, s in want if abs(s) < 1e-40}


def _without(hits, rows):
    return [(r, s) for r, s in hits if r not in rows]


QDRANT_CELLS = [
    pytest.param(v, r, m, sfx, id=f"{v}-{r}-{m}{sfx}")
    for v in VARIANTS for r in REGIMES for m in METRICS for sfx in TOPKS
]


@pytest.mark.qdrant
@pytest.mark.parametrize("variant,regime,metric,sfx", QDRANT_CELLS)
def test_gate_agrees_with_qdrant(variant, regime, metric, sfx, sp_ds, gate_runs,
                                 client, sp_collections, device):
    """Qdrant is asked for EVERY candidate (k=KALL exceeds the corpus), the
    documented zero-score rows are dropped from both full lists, and both are
    then cut to this cell's k — so a k=12 list is compared against Qdrant's
    true top-12 of the remaining rows, not against a list the dropped rows
    shortened. nova-bf's own k=12 list must be the prefix of its full list."""
    ds = sp_ds[variant]
    hits, _, _ = gate_runs[(variant, regime)]
    k = TOPKS[sfx]
    got_k = hits[f"sp_{metric}{sfx}"]
    got_all = hits[f"sp_{metric}_all"]
    want_all = _qdrant(client, sp_collections[variant], ds, metric, regime, KALL)
    label = f"[{device}] {variant}/{regime}/{metric}{sfx} vs qdrant"
    for qi in range(len(ds.queries)):
        assert [r for r, _ in got_k[qi]] == [r for r, _ in got_all[qi][:k]], (
            f"{label} q{qi}: the k={k} list is not the prefix of the full list")
        drop = _zero_score_candidates(ds, metric, regime, qi)
        layouts.assert_topk(_without(got_all[qi], drop)[:k], _without(want_all[qi], drop)[:k],
                            metric=metric, label=f"{label} q{qi}")


@pytest.mark.qdrant
def test_the_qdrant_zero_score_divergence_is_what_it_was(sp_ds, gate_runs, client,
                                                         sp_collections, device):
    """The divergence the comparison above removes, asserted as itself:
    unfiltered, every zero-score candidate is in nova-bf's full list AT 0.0
    and in none of Qdrant's — and it happens in exactly the measured
    (variant, metric) pairs. The day Qdrant starts keeping them (or nova-bf
    starts dropping them) this fails, as the strict xfail it replaces did."""
    seen = set()
    for v in VARIANTS:
        ds = sp_ds[v]
        for metric in METRICS:
            for regime in _UNFILTERED:
                got = gate_runs[(v, regime)][0][f"sp_{metric}_all"]
                want = _qdrant(client, sp_collections[v], ds, metric, regime, KALL)
                for qi in range(len(ds.queries)):
                    drop = _zero_score_candidates(ds, metric, regime, qi)
                    if not drop:
                        continue
                    seen.add((v, metric))
                    mine = dict(got[qi])
                    assert all(mine.get(r) == 0.0 for r in drop), (
                        f"[{device}] {v}/{regime}/{metric} q{qi}: nova-bf should keep "
                        f"{sorted(drop)} at 0.0, has {[(r, mine.get(r)) for r in sorted(drop)]}")
                    assert not drop & {r for r, _ in want[qi]}, (
                        f"[{device}] {v}/{regime}/{metric} q{qi}: Qdrant now returns "
                        f"zero-score candidates {sorted(drop & {r for r, _ in want[qi]})}")
    assert seen == _DIV_PAIRS, seen


# ===================================================================== edges
#
# A signed corpus (so the structural gate is the one in play) built from
# hand-written rows, each one an edge case, plus random filler so every query
# has a real ranking around them.

E_HUGE = 4_000_000_000
E_MAX = 4_294_967_295
EDGE_ROWS = {
    # row: (indices, values)   -- None = NULL row
    0: ([1, 2], [1.0, 1.0]),
    1: ([], []),                        # empty: never a candidate
    2: None,                            # null: never a candidate
    3: ([1], [0.0]),                    # stored zero: candidate at 0.0
    4: ([1, 1], [0.5, 0.5]),            # repeated index: ONE dim worth 1.0
    5: ([1, 1], [1.0, -1.0]),           # repeated index cancelling to 0.0
    6: ([E_HUGE], [1.5]),               # token id past 2^31
    7: ([E_MAX, 2], [2.0, 0.25]),       # u32 max, unsorted
    8: ([2, 1], [2.0, 0.5]),            # unsorted indices
    9: ([1], [-3.0]),                   # overlap with a negative score
    10: ([9, 1], [1.0, 1.0]),           # a non-query token inflating the norm
    11: ([E_HUGE, E_HUGE], [1.0, 2.0]), # repeated huge index
}
N_EDGE_FILLER = 37
EDGE_QUERIES = [
    ([1, 2], [1.0, 1.0]),               # 0 plain
    ([E_HUGE, 2], [2.0, 0.5]),          # 1 huge token
    ([E_MAX], [1.0]),                   # 2 u32-max token
    ([1, 1, 2], [0.5, 0.5, 1.0]),       # 3 repeated QUERY index (summed: 1 -> 1.0)
    ([], []),                           # 4 empty query
    ([1, 2], [0.0, 1.0]),               # 5 zero QUERY weight on token 1
    ([31337], [1.0]),                   # 6 no candidate anywhere
]
EK = 9


def _build_edges(tmp):
    rng = np.random.default_rng(99)
    rows, docs = [], []
    for g in range(len(EDGE_ROWS) + N_EDGE_FILLER):
        if g in EDGE_ROWS:
            r = EDGE_ROWS[g]
        else:
            nnz = int(rng.integers(2, 6))
            idx = rng.choice(np.arange(1, 12), size=nnz, replace=False).tolist()
            r = (idx, rng.standard_normal(nnz).tolist())
        rows.append(r)
        docs.append(naive.Doc(row=g, payload={"grp": GROUPS[g % 3]},
                              sparse=None if r is None else naive.sparse_from_pairs(*r)))
    # Two files (13 rows, rest) so the huge-id rows and the fillers straddle a
    # file boundary.
    cut = 13
    tables = [pa.table({
        "sparse_embedding": layouts.sparse_array(rows[a:b]),
        "grp": pa.array([GROUPS[g % 3] for g in range(a, b)]),
        "rid": pa.array([str(g) for g in range(a, b)]),
    }) for a, b in ((0, cut), (cut, len(rows)))]
    queries = [{"sparse": naive.sparse_from_pairs(i, v), "payload": {}}
               for i, v in EDGE_QUERIES]
    qt = pa.table({"sparse_embedding": layouts.sparse_array(EDGE_QUERIES),
                   "qid": pa.array([str(i) for i in range(len(EDGE_QUERIES))])})
    return layouts.build(tmp, "sp_edges", tables, docs, qt, queries)


@pytest.fixture(scope="module")
def edge_ds(tmp_path_factory):
    return _build_edges(tmp_path_factory.mktemp("spedge"))


@pytest.fixture(scope="module")
def edge_runs(edge_ds, device):
    specs = [layouts.spec(f"e_{m}{sfx}", vector_type="sparse", metric=m, k=k)
             for m in METRICS for sfx, k in (("", EK), ("_all", KALL))]
    out = {"nofilter": layouts.run(edge_ds, specs, tag="edge", device=device,
                                   params={"sparse_batch_size": 5})}
    fspecs = [dict(s, filter=STATIC) for s in specs]
    out["static"] = layouts.run(edge_ds, fspecs, tag="edge_f", device=device,
                                params={"sparse_batch_size": 5})
    return out


# Query 5 is the one semantic asymmetry with the naive oracle, and it is not a
# float question: nova-bf densifies the query (`_sparse_rows_to_dense`), so a
# query token stored with weight 0.0 vanishes and is not an overlap, while a
# CORPUS token stored with 0.0 IS one (row 3, kept at 0.0). naive treats both
# sides as stored structure. So the naive comparison drops exactly the rows
# that share ONLY zero-weight query tokens (derived from the data, and empty
# for every other query), and `test_a_zero_weight_query_token_is_not_an_overlap`
# pins the divergence itself.
_Q_ZERO_WEIGHT = 5


def _zero_weight_only_rows(ds, qi):
    """Rows whose every token shared with query `qi` carries query weight 0.0."""
    q = ds.queries[qi]["sparse"]
    live = {t for t, w in q.items() if w != 0.0}
    dead = {t for t, w in q.items() if w == 0.0}
    return {d.row for d in ds.docs if d.sparse
            and not (set(d.sparse) & live) and set(d.sparse) & dead}


def _edge_zero_score_rows(ds, qi):
    """Rows that ARE candidates under nova-bf's rule (share a nonzero-weight
    query token) but score exactly 0.0: what UNFILTERED Qdrant drops."""
    q = ds.queries[qi]["sparse"]
    live = {t for t, w in q.items() if w != 0.0}
    return {d.row for d in ds.docs if d.sparse and set(d.sparse) & live
            and sum(q[t] * d.sparse[t] for t in set(d.sparse) & set(q)) == 0.0}


def _edge_cells():
    for regime in ("nofilter", "static"):
        for m in METRICS:
          for sfx in TOPKS:
            for qi in range(len(EDGE_QUERIES)):
                yield pytest.param(regime, m, sfx, qi, id=f"{regime}-{m}{sfx}-q{qi}")


@pytest.mark.parametrize("regime,metric,sfx,qi", list(_edge_cells()))
def test_edge_rows_agree_with_the_naive_oracle(regime, metric, sfx, qi, edge_ds, edge_runs, device):
    got = edge_runs[regime][f"e_{metric}{sfx}"][qi]
    drop = _zero_weight_only_rows(edge_ds, qi)
    assert bool(drop) == (qi == _Q_ZERO_WEIGHT), (qi, drop)
    want = layouts.naive_topk(edge_ds, vector_type="sparse", metric=metric,
                              k=EK if sfx == "" else KALL, queries=[qi],
                              docs=[d for d in edge_ds.docs if d.row not in drop])[qi]
    layouts.assert_topk(got, want, metric=metric,
                        label=f"[{device}] edges {regime}/{metric} q{qi} vs naive")


@pytest.mark.parametrize("regime", ["nofilter", "static"])
@pytest.mark.parametrize("metric", METRICS)
def test_a_zero_weight_query_token_is_not_an_overlap(regime, metric, edge_ds, edge_runs,
                                                     device):
    """The divergence from naive, asserted as itself: the rows that share only
    q5's 0.0-weight token are candidates to naive (so the comparison above
    really did remove something) and absent from nova-bf's full list."""
    qi = _Q_ZERO_WEIGHT
    drop = _zero_weight_only_rows(edge_ds, qi)
    assert 3 in drop and 5 in drop, drop          # the stored-zero rows on token 1
    naive_all = layouts.naive_topk(edge_ds, vector_type="sparse", metric=metric,
                                   k=KALL, queries=[qi])[qi]
    assert drop <= {r for r, _ in naive_all}, (drop, naive_all)
    got = {r for r, _ in edge_runs[regime][f"e_{metric}_all"][qi]}
    assert not drop & got, (
        f"[{device}] {regime}/{metric}: nova-bf now counts a zero-weight query token "
        f"as an overlap (returns {sorted(drop & got)})")


@pytest.mark.parametrize("regime", ["nofilter", "static"])
def test_edge_rows_pinned_semantics(regime, edge_runs, device):
    """The individual rules, stated as rules (the oracle comparison above
    proves the numbers; this says WHICH rule each row exercises)."""
    dot = edge_runs[regime]["e_dot_all"]
    q0 = dict(dot[0])
    assert 1 not in q0 and 2 not in q0, "empty / null rows are never candidates"
    assert q0.get(3) == 0.0, "a stored 0.0 weight is an overlap scoring 0.0"
    assert math.isclose(q0[4], 1.0, abs_tol=1e-6), "a repeated index is ONE dimension, summed"
    assert q0.get(5) == 0.0, "a repeated index cancelling to 0.0 is still an overlap"
    assert math.isclose(dict(dot[1])[6], 3.0, abs_tol=1e-6), "token id 4e9 survives"
    assert math.isclose(dict(dot[1])[11], 6.0, abs_tol=1e-5), "repeated 4e9 index summed"
    assert math.isclose(dict(dot[2])[7], 2.0, abs_tol=1e-6), "token id u32-max survives"
    # repeated QUERY index: q3 = {1: 1.0, 2: 1.0}, so row 4 ({1: 1.0}) scores 1.0
    assert math.isclose(dict(dot[3])[4], 1.0, abs_tol=1e-6), "a repeated query index is summed"
    assert dot[4] == [], "an empty query has no candidates"
    assert dot[6] == [], "a query sharing no token has no candidates"
    assert 3 not in dict(dot[5]), (
        "pinned: a ZERO-weight query token is not an overlap (row 3 shares only it)")


# Qdrant, from the measured semantics above:
#   * unfiltered: Qdrant drops score-0.0 candidates (rows 3 and 5 for q0/q3)
#     that nova-bf keeps (`_edge_zero_score_rows`);
#   * filtered:   Qdrant keeps them, AND counts a zero-weight QUERY token as
#     an overlap (q5's rows 3, 5, ...), which nova-bf does not
#     (`_zero_weight_only_rows`). (Unfiltered, q5's two rules remove the SAME
#     rows, so there the two agree — for different reasons.)
# Each cell drops exactly its documented rows from both full candidate lists,
# then cuts both to k; `test_the_edge_qdrant_divergence_is_what_it_was` pins
# the divergence itself.
def _edge_qdrant_drop(ds, regime, qi):
    if regime == "nofilter":
        return _edge_zero_score_rows(ds, qi)
    return _zero_weight_only_rows(ds, qi)


def _edge_qdrant_cells():
    for regime in ("nofilter", "static"):
        for m in METRICS:
          for sfx in TOPKS:
            for qi in range(len(EDGE_QUERIES)):
                yield pytest.param(regime, m, sfx, qi, id=f"{regime}-{m}{sfx}-q{qi}")


def _edge_qdrant(client, coll, ds, regime, metric, qi):
    one = layouts.LayoutDS(**{**ds.__dict__, "queries": [ds.queries[qi]]})
    return layouts.qdrant_topk(
        client, coll, one, using="s_dot" if metric == "dot" else "s_cos",
        vector_type="sparse", k=KALL,
        qfilter=_qfilter(regime, None) if regime == "static" else None,
        normalize_sparse=(metric == "cosine"))[0]


@pytest.fixture(scope="module")
def edge_collection(client, collection, edge_ds):
    from qdrant_client import models as M

    name = layouts.qdrant_collection(
        client, edge_ds, sparse=("s_dot", "s_cos"),
        payload_index={"grp": M.PayloadSchemaType.KEYWORD})
    yield name
    client.delete_collection(name)


@pytest.mark.qdrant
@pytest.mark.parametrize("regime,metric,sfx,qi", list(_edge_qdrant_cells()))
def test_edge_rows_agree_with_qdrant(regime, metric, sfx, qi, edge_ds, edge_runs,
                                     client, edge_collection, device):
    """Qdrant is given the COALESCED vectors: it rejects a repeated index
    outright (422, measured — corpus and query side alike), so the raw form
    cannot be adjudicated by it and the summing rule rests on the naive
    oracle alone."""
    k = EK if sfx == "" else KALL
    want_all = _edge_qdrant(client, edge_collection, edge_ds, regime, metric, qi)
    got_all = edge_runs[regime][f"e_{metric}_all"][qi]
    got = edge_runs[regime][f"e_{metric}{sfx}"][qi]
    label = f"[{device}] edges {regime}/{metric}{sfx} q{qi} vs qdrant"
    assert [r for r, _ in got] == [r for r, _ in got_all[:k]], (
        f"{label}: the k={k} list is not the prefix of the full list")
    drop = _edge_qdrant_drop(edge_ds, regime, qi)
    layouts.assert_topk(_without(got_all, drop)[:k], _without(want_all, drop)[:k],
                        metric=metric, label=label)


@pytest.mark.qdrant
@pytest.mark.parametrize("metric", METRICS)
def test_the_edge_qdrant_divergence_is_what_it_was(metric, edge_ds, edge_runs, client,
                                                   edge_collection, device):
    """The rows the comparison above drops, asserted as the divergence they
    are: unfiltered, nova-bf keeps q0/q3's zero-score rows at 0.0 and Qdrant
    returns none of them; filtered, Qdrant returns q5's zero-weight-only rows
    at 0.0 and nova-bf none of them. Exactly those (regime, query) pairs."""
    seen = set()
    for regime in ("nofilter", "static"):
        for qi in range(len(EDGE_QUERIES)):
            drop = _edge_qdrant_drop(edge_ds, regime, qi)
            if not drop:
                continue
            seen.add((regime, qi))
            mine = dict(edge_runs[regime][f"e_{metric}_all"][qi])
            theirs = dict(_edge_qdrant(client, edge_collection, edge_ds, regime, metric, qi))
            keeper, dropper = (mine, theirs) if regime == "nofilter" else (theirs, mine)
            assert all(keeper.get(r) == 0.0 for r in drop), (
                f"[{device}] {regime}/{metric} q{qi}: expected {sorted(drop)} at 0.0, "
                f"got {[(r, keeper.get(r)) for r in sorted(drop)]}")
            assert not drop & set(dropper), (
                f"[{device}] {regime}/{metric} q{qi}: the divergence closed "
                f"({sorted(drop & set(dropper))} now on both sides)")
    assert seen == {("nofilter", 0), ("nofilter", 3), ("static", _Q_ZERO_WEIGHT)}, seen


# ------------------------------------------------------------- non-finite


def test_a_nan_sparse_row_is_an_input_error(tmp_path_factory, device):
    """A NaN sparse weight is rejected by the reader, naming the file, the
    column and the rows. (It used to make that row's score NaN, rank above
    every real score and cost the query a result slot -- the strict xfail
    this replaces.)"""
    tmp = tmp_path_factory.mktemp("spnan")
    rng = np.random.default_rng(5)
    rows, docs = [], []
    for g in range(30):
        idx = [1, 2, int(rng.integers(3, 9))]
        val = rng.uniform(0.1, 2.0, size=3).tolist()
        if g in (4, 11, 19):
            val[0] = float("nan")
        rows.append((idx, val))
        docs.append(naive.Doc(row=g, sparse=None if g in (4, 11, 19)
                              else naive.sparse_from_pairs(idx, val)))
    t = pa.table({"sparse_embedding": layouts.sparse_array(rows),
                  "rid": pa.array([str(g) for g in range(30)])})
    qs = [([1, 2], [1.0, 0.5])]
    ds = layouts.build(tmp, "spnan", [t], docs,
                       pa.table({"sparse_embedding": layouts.sparse_array(qs),
                                 "qid": pa.array(["0"])}),
                       [{"sparse": naive.sparse_from_pairs(*qs[0]), "payload": {}}])
    k = 6
    for m in METRICS:
        with pytest.raises(RuntimeError) as e:
            layouts.run(ds, [layouts.spec(f"n_{m}", vector_type="sparse", metric=m, k=k)],
                        tag=f"spnan_{m}", device=device)
        msg = str(e.value)
        assert "part-000.parquet" in msg and "column `sparse_embedding.values`" in msg
        assert "3 row(s)" in msg and "rows 4, 11, 19;" in msg
        assert "NaN/inf vectors corrupt results; clean or drop these rows." in msg
