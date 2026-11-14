"""The regimes the matrix structurally cannot reach: dead, short and mixed-k.

`test_parity_matrix` only ever asks for a full top-25 out of a filter that
leaves more than 25 rows — its fixture guard insists on it, because there a
filter matching (almost) nothing would make a whole matrix row vacuous. That
guard is right for the matrix and leaves a hole everywhere else:

  * **dead queries** — a query whose filter admits NO row (a per-query value
    no corpus row holds, a null per-query value, a token-less phrase). The
    answer is an empty list, and an empty list is exactly what a broken mask
    also produces, so it is asserted against both oracles with
    `allow_empty=True` AND with the per-query eligible count pinned;
  * **short results** — fewer eligible rows than `k`: every eligible row
    must come back, no more (a top-K padded with ineligible rows, or cut
    short, is wrong);
  * **k = 1**, and **k past a file and past the whole corpus** (k = 400 on 300
    rows, files of 61-97) — at that size the comparison stops being a ranking
    comparison and becomes an exact eligible-set comparison;
  * **mixed k in one pass** — all three k share one batch grid (the grid's
    floor is the LARGEST k of its vector_type), so a k = 1 search is merged
    through slices sized for k = 400;
  * the **compacted** regime — no unfiltered search in a vector_type, so the
    batch grid is the union of the filters' rows and every reported row
    number round-trips through that compaction — and the **legacy** routing
    (`NOVA_BF_NO_GPU_COMBO`: `PackedRowMask`s and `_gpu_evaluate` instead of
    `FilterPlan`s), in both grids;
  * the same dead / short / mixed-k specs **sharded** W = 3 and 5 and merged.

The filters are `cases.EDGE_FILTERS` (nulls, type-strict int/bool/dictionary
match, empty/null phrases, every plan shape); every cell runs on every modality.
`test_regime_paths_were_reached` pins that each arm really took its route.
"""

from __future__ import annotations

import pytest

from nova_bf.compute import run_compute
from nova_bf.merge import run_merge

from . import cases as cases_mod
from . import compare, naive, qdrant_ref
from .corpus import FILE_SIZES
from .runner import (MV_23_DOCS_MIB, build_config, pinned_device, probe,
                     read_results, run, spec)

EDGE = {**cases_mod.EDGE_FILTERS, **cases_mod.UNFILTERED_EDGE_FILTERS}
KS = cases_mod.EDGE_KS
MODS = cases_mod.MODALITIES
MATRIX_PARAMS = {"dense_batch_size": 37, "sparse_batch_size": 29,
                 "multivector_budget_mib": MV_23_DOCS_MIB}

# arm -> (filters it runs, NOVA_BF_NO_GPU_COMBO set?)
ARMS = {
    "baseline": (EDGE, False),
    "compacted": ({f: cases_mod.EDGE_FILTERS[f] for f in cases_mod.COMPACT_EDGE}, False),
    "legacy_baseline": (EDGE, True),
    "legacy_compacted": ({f: cases_mod.EDGE_FILTERS[f] for f in cases_mod.COMPACT_EDGE}, True),
}


def name_of(vt, metric, fname, k) -> str:
    return f"{vt[:2]}{metric[:3]}_{fname}_k{k}"


def _specs(filters):
    out = []
    for fname, fdict in filters.items():
        for vt, metric in MODS:
            for k in KS:
                out.append(spec(name_of(vt, metric, fname, k), vector_type=vt,
                                metric=metric, k=k, filter=fdict))
    if filters is EDGE:
        # the baseline arm's whole-file grid comes from a no-filter search
        out += [spec(f"{vt[:2]}{m[:3]}_nofilter", vector_type=vt, metric=m, k=7)
                for vt, m in MODS]
    return out


def _path_probe():
    from nova_bf import compute as C
    from nova_bf import filters as F

    def plan_hook(p, out, plan, rows, true_rows, *a, **k):
        kind = ("mixed" if plan.text is not None and (plan.struct_must or plan.struct_should
                                                      or plan.struct_not)
                else "text" if plan.text is not None else "struct")
        p.note(f"plan_{kind}_{'slice' if isinstance(true_rows, slice) else 'gather'}")
        if plan.static_or is not None:
            p.note("plan_static_or")
        if plan.struct_should:
            p.note("plan_struct_should")
        if plan.struct_not:
            p.note("plan_struct_not")
        if out is None:
            p.note("plan_dead_slice")

    def union_hook(p, out, filters, keeps, n_rows):
        p.note("union_rows_total", int(n_rows))
        p.note("union_rows_kept", int(out.sum()))

    return probe(
        lines={
            "packed_gather": (C, "bits = pm.packed[:, true_rows >> 3] & sel"),
            "packed_slice": (C, "sub = pm.packed[:, r0 >> 3 : (r1 + 7) >> 3]"),
            "corpus_null_code": (C, "codes[nulls] = -1               # -1: matches no query"),
            "corpus_null_nan": (C, "vals[nulls] = np.nan"),
            "matchany_empty_vocab": (C, "return torch.zeros((qstate.shape[0], arr.shape[0]), "
                                        "dtype=torch.bool, device=arr.device)"),
            "zero_combo": (F, "ands, ors = (ZERO,), (ONES,)"),
            "union_generic": (F, "own = {c: set(self.and_idx[c].tolist()) - common - {self.ONES}"),
        },
        calls={
            "plan_mask_device": (C, "_plan_mask_device", plan_hook),
            "gpu_evaluate": (C, "_gpu_evaluate", None),
            "union_keep": (C, "_union_keep", union_hook),
            "vocab_lut": (C, "_build_vocab_lut", None),
        })


@pytest.fixture(scope="session")
def regimes(ds, device):
    """`arm -> (filters, results, probe)`, each arm ONE nova-bf run of its edge
    filters × every modality × every k — run lazily, the first time a test
    asks for that arm, and then shared."""
    done: dict = {}

    def get(arm):
        if arm not in done:
            filters, legacy = ARMS[arm]
            mp = pytest.MonkeyPatch()
            try:
                if legacy:
                    mp.setenv("NOVA_BF_NO_GPU_COMBO", "1")
                else:
                    mp.delenv("NOVA_BF_NO_GPU_COMBO", raising=False)
                with _path_probe() as p:
                    got = run(ds, _specs(filters), out_tag=f"regime_{arm}", device=device,
                              params=MATRIX_PARAMS)
            finally:
                mp.undo()
            done[arm] = (filters, got, p)
        return done[arm]
    return get


ARM_CELLS = [(a, f) for a in ARMS for f in ARMS[a][0]]
ARM_IDS = [f"{a}-{f}" for a, f in ARM_CELLS]
LIVE_CELLS = [(a, f) for a, f in ARM_CELLS if not a.startswith("legacy")]


def _assert_against(got, want, *, ds, metric, label):
    assert set(got) == set(range(len(ds.queries))), (
        f"{label}: nova-bf answered queries {sorted(got)}, expected every query "
        f"(a dead query must be present with an EMPTY list, not missing)")
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(got[qi], want[qi], metric=metric, allow_empty=True,
                                    label=f"{label} q{qi}")


@pytest.mark.parametrize("arm,fname", ARM_CELLS, ids=ARM_IDS)
def test_regime_matches_naive(arm, fname, regimes, ds, oracle, device):
    _, got, _ = regimes(arm)
    for vt, metric in MODS:
        for k in KS:
            want = oracle.topk(vector_type=vt, metric=metric, k=k, filt=EDGE[fname])
            _assert_against(got[name_of(vt, metric, fname, k)], want, ds=ds, metric=metric,
                            label=f"[{device}/{arm}] {vt}-{metric} {fname} k={k} vs naive")


@pytest.fixture(scope="session")
def qdrant_edge(client, collection, ds):
    """Every edge cell's Qdrant answer, computed once and shared by the arms."""
    return {
        (fname, vt, metric, k): qdrant_ref.topk(client, collection, ds, vector_type=vt,
                                                metric=metric, k=k, filt=fdict)
        for fname, fdict in EDGE.items() for vt, metric in MODS for k in KS
    }


@pytest.mark.qdrant
@pytest.mark.parametrize("arm,fname", LIVE_CELLS, ids=[f"{a}-{f}" for a, f in LIVE_CELLS])
def test_regime_matches_qdrant(arm, fname, regimes, ds, qdrant_edge, device):
    """The live (FilterPlan) routing against the engine, both grids. The
    legacy arms are graded against naive only: they are an operator fallback,
    and the eligibility-count test below pins naive to Qdrant per query."""
    _, got, _ = regimes(arm)
    for vt, metric in MODS:
        for k in KS:
            _assert_against(got[name_of(vt, metric, fname, k)],
                            qdrant_edge[(fname, vt, metric, k)], ds=ds, metric=metric,
                            label=f"[{device}/{arm}] {vt}-{metric} {fname} k={k} vs qdrant")


@pytest.mark.qdrant
@pytest.mark.parametrize("fname", list(EDGE))
def test_edge_eligibility_counts_match_qdrant(fname, ds, client, collection):
    """The two oracles agree on WHICH points each filter admits, per query —
    counted by Qdrant itself (`client.count`), independent of any scoring.
    This is what makes the per-cell comparisons above mean the engine's
    semantics and not merely the harness's."""
    fdict = EDGE[fname]
    for qi, q in enumerate(ds.queries):
        want = sum(naive.filter_matches(fdict, d.payload, q["payload"], ds.date_fields,
                                        ds.query_date_fields) for d in ds.docs)
        got = qdrant_ref.count(client, collection, fdict, q["payload"], ds.date_fields)
        assert got == want, f"{fname} q{qi}: Qdrant counts {got} eligible, naive {want}"


def test_edge_fixture_reaches_every_regime(ds):
    """A guard on the FIXTURE: the regimes this file exists for must actually
    occur in it, for every plan shape — otherwise the tests above would pass
    on the easy cases alone."""
    def counts(fdict):
        return [sum(naive.filter_matches(fdict, d.payload, q["payload"], ds.date_fields,
                                         ds.query_date_fields) for d in ds.docs)
                for q in ds.queries]

    by = {f: counts(fd) for f, fd in EDGE.items()}
    n = len(ds.docs)
    kmid = sorted(KS)[1]
    shapes = {"static": "e_dead_static", "struct": "e_pq_int", "text": "e_pqtext_note",
              "mixed": "e_mixed_and"}
    for shape, f in shapes.items():
        assert 0 in by[f], f"{shape} shape ({f}) has no dead query"
        assert any(c > 0 for c in by[f]) or shape == "static", f"{f} is dead everywhere"
    assert any(0 < c < kmid for cs in by.values() for c in cs), "no short (0 < eligible < k) query"
    assert max(KS) > n > max(FILE_SIZES), "k=400 must exceed a file and the corpus"
    for f, cs in by.items():
        if f in cases_mod.UNFILTERED_EDGE_FILTERS:
            assert cs == [n] * len(cs)
        else:
            assert min(cs) < n, f"{f} admits every row for every query — it constrains nothing"


@pytest.mark.parametrize("arm", list(ARMS))
def test_regime_paths_were_reached(arm, regimes):
    """Each arm took the route it is named for. Without this a regime could
    silently degrade into another (e.g. a compacted arm that kept a baseline
    search, or a legacy arm the env switch never reached) and every
    comparison above would still pass."""
    _, _, p = regimes(arm)
    legacy = arm.startswith("legacy")
    compacted = arm.endswith("compacted")
    assert p.hit("corpus_null_code") or legacy, "a null corpus value never reached a pq match leaf"
    if compacted:
        assert p.calls["union_keep"] > 0, "no union compaction ran in the compacted arm"
        assert p.notes["union_rows_kept"] < p.notes["union_rows_total"], (
            "the compacted arm's row union kept every row — nothing was compacted")
    else:
        assert p.calls["union_keep"] == 0, "the baseline arm compacted — it has no baseline"
    if legacy:
        assert p.calls["plan_mask_device"] == 0, "legacy arm built FilterPlans"
        assert p.calls["gpu_evaluate"] > 0, "legacy arm never ran _gpu_evaluate"
        assert p.hit("packed_gather" if compacted else "packed_slice"), (
            "legacy arm never read a PackedRowMask on its own grid")
        assert p.combo["packed_masks"] > 0
    else:
        assert p.calls["gpu_evaluate"] == 0
        for kind in ("struct", "text", "mixed"):
            got = p.notes.get(f"plan_{kind}_{'gather' if compacted else 'slice'}", 0)
            assert got > 0, f"no {kind} FilterPlan was evaluated on the {arm} grid"
        for note in ("plan_static_or", "plan_struct_should", "plan_struct_not",
                     "plan_dead_slice"):
            assert p.notes.get(note, 0) > 0, f"{note} never reached"
        assert p.hit("corpus_null_nan"), "a null corpus value never reached a pq range leaf"
        assert p.hit("zero_combo"), "no null/token-less required phrase reached the combo"
        assert p.hit("union_generic"), "no combination with >= 2 own parts in the union"
        assert p.calls["vocab_lut"] > 0, "no per-query match vocab lookup table was built"
        if not compacted:
            assert p.hit("matchany_empty_vocab"), "no MatchAny with an empty vocabulary"


# ------------------------------------------------------------------ sharded

SHARD_FILTERS = ("e_dead_static", "e_short_static", "e_pq_int", "e_pqtext_note",
                 "e_mixed_and", "e_pq_langs_n_not")
SHARD_MODS = (("dense", "cosine"), ("sparse", "dot"), ("multivector", "dot"))


def _shard_specs():
    return [spec(name_of(vt, m, f, k), vector_type=vt, metric=m, k=k, filter=EDGE[f])
            for f in SHARD_FILTERS for vt, m in SHARD_MODS for k in KS]


@pytest.fixture(scope="session", params=[3, 5], ids=["W3", "W5"])
def sharded_edge(request, ds, device):
    """compute × W ranks over the compacted (all-filtered) edge specs, then one
    merge. W=5 exceeds the 4 corpus files, so a rank holds no rows at all."""
    w = request.param
    cfg = build_config(ds, _shard_specs(), out_tag=f"shard_edge{w}_{device}",
                       params=MATRIX_PARAMS)
    with pinned_device(device):
        for rank in range(w):
            run_compute(cfg, num_jobs=w, job_rank=rank)
        return w, read_results(run_merge(cfg))


@pytest.mark.parametrize("fname", SHARD_FILTERS)
def test_sharded_regimes_match_naive(fname, sharded_edge, ds, oracle, device):
    w, got = sharded_edge
    for vt, metric in SHARD_MODS:
        for k in KS:
            want = oracle.topk(vector_type=vt, metric=metric, k=k, filt=EDGE[fname])
            _assert_against(got[name_of(vt, metric, fname, k)], want, ds=ds, metric=metric,
                            label=f"[{device}/W{w}] {vt}-{metric} {fname} k={k} vs naive")

