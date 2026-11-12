"""Filter code paths a small fixture routes around unless it is MADE to take them.

Each group below forces one path, proves (with `runner.probe`) that the path
really ran, and checks the answer against the oracles. The "proves it ran"
half is not optional: every one of these paths is an alternative way of
computing the SAME mask, so a run that quietly took the ordinary path instead
would pass every comparison and test nothing.

  * **budget fallback** — `params.combo_parts_max_bytes: 0`: per-query text
    parts are combined per combination on the CPU (`_combination_rows_mask`)
    instead of shipped to the device as phrase rows;
  * **`rows:` narrowing** — per-query structured / mixed / text filters used
    only by `rows:`-selected searches: the shared leaf state is narrowed to
    the filter's own query rows (`_narrow_gpu_leaf_state`), on every
    modality including multivector;
  * **windowed compacted gather** — the device copy of a text part held in
    one-byte column windows, read through a compacted (gappy) row grid, so the
    gather crosses windows (`_gather_part_bytes`);
  * **struct_union** — a compacted grid built from a per-query range's
    cross-query extreme bound, for `gt` and for `lte`;
  * **Arrow tokenizer fallback** — `NOVA_BF_NO_NATIVE_TOKENIZER`;
  * **legacy routing over the whole matrix** — `NOVA_BF_NO_GPU_COMBO`, with
    and without a baseline (unfiltered) search.
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare, qdrant_ref
from .runner import probe, run, spec

EDGE = cases_mod.EDGE_FILTERS
MODS = cases_mod.MODALITIES
PARAMS = {"dense_batch_size": 37, "sparse_batch_size": 29, "multivector_batch_size": 23}
TEXTY = [f for f, fd in EDGE.items()
         if any("match_text_from_query" in c for g in fd.values() for c in g)]


def _name(tag, vt, metric, fname, k):
    return f"{tag}_{vt[:2]}{metric[:3]}_{fname}_k{k}"


def _grid(tag, fnames, ks=(7, 400), rows=None, mods=MODS):
    return [spec(_name(tag, vt, m, f, k), vector_type=vt, metric=m, k=k, filter=EDGE[f],
                 rows=rows) for f in fnames for vt, m in mods for k in ks]


def _check(got, oracle, ds, tag, fnames, *, device, ks=(7, 400), queries=None, mods=MODS):
    queries = list(range(len(ds.queries))) if queries is None else queries
    for f in fnames:
        for vt, m in mods:
            for k in ks:
                want = oracle.topk(vector_type=vt, metric=m, k=k, filt=EDGE[f], queries=queries)
                res = got[_name(tag, vt, m, f, k)]
                assert sorted(res) == sorted(queries), (
                    f"[{device}/{tag}] {f}: answered {sorted(res)}, expected {queries}")
                for qi in queries:
                    compare.assert_scores_agree(
                        res[qi], want[qi], metric=m, allow_empty=True,
                        label=f"[{device}/{tag}] {vt}-{m} {f} k={k} q{qi} vs naive")


# ------------------------------------------------------- budget fallback

@pytest.fixture(scope="module")
def budget_run(ds, device):
    from nova_bf import filters as F

    with probe(calls={"rows_mask": (F, "_combination_rows_mask", None)}) as p:
        got = run(ds, _grid("bud", TEXTY), out_tag="budget", device=device,
                  params={**PARAMS, "combo_parts_max_bytes": 0})
    return got, p


def test_budget_fallback_was_taken(budget_run):
    _, p = budget_run
    assert p.calls["rows_mask"] > 0 and p.combo["budget_fallbacks"] > 0, (
        "combo_parts_max_bytes=0 never sent a filter to the per-combination fallback")


@pytest.mark.parametrize("fname", TEXTY)
def test_budget_fallback_matches_naive(fname, budget_run, ds, oracle, device):
    _check(budget_run[0], oracle, ds, "bud", [fname], device=device)


# ------------------------------------------------------- rows: narrowing

ROWS_FILTERS = {
    "even": ("e_pq_int", "e_pq_intany", "e_pq_bool", "e_pq_range_frac", "e_pq_should_not",
             "e_mixed_should", "e_pqtext_note"),
    "g1": ("e_pq_lang_n", "e_pq_dict", "e_pq_weight_not", "e_mixed_not", "e_mixed_and"),
}
ROWS_COL = {"even": "query_set", "g1": "query_third"}
# `rows` is rejected for multivector at config load (pinned below), so the
# narrowing is exercised on the dense and sparse modalities.
ROWS_MODS = tuple(m for m in MODS if m[0] != "multivector")


def _selected(ds, sel):
    return [qi for qi, q in enumerate(ds.queries) if q["payload"][ROWS_COL[sel]] == sel]


@pytest.fixture(scope="module")
def rows_run(ds, device):
    from nova_bf import compute as C

    specs = []
    for sel, fnames in ROWS_FILTERS.items():
        specs += _grid(f"rows{sel}", fnames, rows={"column": ROWS_COL[sel], "isin": [sel]},
                       mods=ROWS_MODS)
    with probe(calls={"narrow": (C, "_narrow_gpu_leaf_state", None)}) as p:
        got = run(ds, specs, out_tag="rowsnarrow", device=device, params=PARAMS)
    return got, p


def test_rows_narrowing_was_taken(rows_run):
    assert rows_run[1].calls["narrow"] > 0, "no per-query leaf state was narrowed to `rows`"


@pytest.mark.parametrize("sel", list(ROWS_FILTERS))
def test_rows_narrowing_matches_naive(sel, rows_run, ds, oracle, device):
    _check(rows_run[0], oracle, ds, f"rows{sel}", ROWS_FILTERS[sel], device=device,
           queries=_selected(ds, sel), mods=ROWS_MODS)


def test_rows_on_multivector_is_rejected_not_ignored(ds):
    """The one modality `rows` does not support must FAIL at load: a selector
    silently ignored would run every query and report it as the subset."""
    from .runner import build_config

    with pytest.raises(Exception, match="rows.*multivector"):
        build_config(ds, [spec("mv_rows", vector_type="multivector", metric="dot",
                               rows={"column": "query_set", "isin": ["even"]})],
                     out_tag="unused")


@pytest.mark.qdrant
@pytest.mark.parametrize("sel", list(ROWS_FILTERS))
def test_rows_narrowing_matches_qdrant(sel, rows_run, ds, client, collection, device):
    qs = _selected(ds, sel)
    for f in ROWS_FILTERS[sel]:
        for vt, m in ROWS_MODS:
            for k in (7, 400):
                want = qdrant_ref.topk(client, collection, ds, vector_type=vt, metric=m, k=k,
                                       filt=EDGE[f], queries=qs)
                res = rows_run[0][_name(f"rows{sel}", vt, m, f, k)]
                for qi in qs:
                    compare.assert_scores_agree(
                        res[qi], want[qi], metric=m, allow_empty=True,
                        label=f"[{device}] rows={sel} {vt}-{m} {f} k={k} q{qi} vs qdrant")


# ------------------------------------------- windowed compacted gather

WINDOW_FILTERS = ("e_pqtext_two_phrases", "e_mixed_and", "e_mixed_two_phrases", "e_pq_int")


@pytest.fixture(scope="module", params=[1, 64], ids=["w1byte", "wmulti"])
def window_run(request, ds, device):
    """Two window sizes: one byte-column per window (every row byte its own
    window), and a few byte-columns per window (rows from different bytes of
    ONE window, which a gather that mixed up positions within a window would
    scramble — invisible at one byte per window)."""
    from nova_bf import compute as C

    def gather(p, out, cm, dev, cols, w, nb):
        p.note("gathers")
        if w < nb and len(set((cols // w).tolist())) > 1:
            p.note("windowed_multi")

    def union(p, out, filters, keeps, n_rows):
        p.note("rows_total", int(n_rows))
        p.note("rows_kept", int(out.sum()))

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(C, "_RESIDENT_WINDOW_BYTES", request.param)
        with probe(calls={"gather": (C, "_gather_part_bytes", gather),
                          "union": (C, "_union_keep", union)}) as p:
            got = run(ds, _grid("win", WINDOW_FILTERS), out_tag=f"window{request.param}",
                      device=device, params=PARAMS)
    finally:
        mp.undo()
    return got, p


def test_windowed_compacted_gather_was_taken(window_run):
    _, p = window_run
    assert p.notes.get("rows_kept", 0) < p.notes.get("rows_total", 0), (
        "the row union kept every row: the grid was never compacted")
    assert p.notes.get("windowed_multi", 0) > 0, (
        "no compacted gather read across more than one device window")


@pytest.mark.parametrize("fname", WINDOW_FILTERS)
def test_windowed_compacted_gather_matches_naive(fname, window_run, ds, oracle, device):
    _check(window_run[0], oracle, ds, "win", [fname], device=device)


# ------------------------------------------------------- struct_union

STRUCT_UNION = {"gt": "e_pq_range_frac", "lte": "e_pq_range_frac_lte"}


@pytest.fixture(scope="module", params=list(STRUCT_UNION))
def struct_union_run(request, ds, device):
    """ONE per-query range filter alone, so the compacted grid is exactly its
    struct_union (any second filter would OR its own rows in and could hide a
    struct_union that dropped rows)."""
    from nova_bf import compute as C

    def hook(p, out, f, *a, **k):
        if out is not None:
            p.note("unions")

    fname = STRUCT_UNION[request.param]
    with probe(calls={"su": (C, "_row_union_from_gpu_leaves", hook)}) as p:
        got = run(ds, _grid("su", [fname]), out_tag=f"su_{request.param}", device=device,
                  params=PARAMS)
    return fname, got, p


def test_struct_union_matches_naive(struct_union_run, ds, oracle, device):
    fname, got, p = struct_union_run
    assert p.notes.get("unions", 0) > 0, "no struct_union was built"
    _check(got, oracle, ds, "su", [fname], device=device)


# ------------------------------------------------------- Arrow tokenizer

TOKEN_FILTERS = ("e_text_note", "e_text_note_multi", "e_mustnot_text_note", "e_pqtext_note",
                 "e_pqtext_two_phrases", "e_mixed_not")


@pytest.fixture(scope="module")
def arrow_tok_run(ds, device):
    from nova_bf import filters as F

    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("NOVA_BF_NO_NATIVE_TOKENIZER", "1")
        with probe(lines={
                "arrow": (F, "toks = pc.split_pattern_regex(chunk, pattern=TOKEN_SPLIT_PATTERN)"),
                "native": (F, "nativetok.scan_into(chunk, native, sub_grid)")}) as p:
            got = run(ds, _grid("tok", TOKEN_FILTERS), out_tag="arrowtok", device=device,
                      params=PARAMS)
    finally:
        mp.undo()
    return got, p


def test_arrow_tokenizer_fallback_was_taken(arrow_tok_run):
    _, p = arrow_tok_run
    assert p.hit("arrow") and not p.hit("native")


@pytest.mark.parametrize("fname", TOKEN_FILTERS)
def test_arrow_tokenizer_fallback_matches_naive(fname, arrow_tok_run, ds, oracle, device):
    _check(arrow_tok_run[0], oracle, ds, "tok", [fname], device=device)


# ------------------------------------------- legacy routing, whole matrix

@pytest.fixture(scope="module", params=["baseline", "compacted"])
def legacy_matrix(request, ds, device):
    """The main matrix under `NOVA_BF_NO_GPU_COMBO`, with its `nofilter`
    searches (whole-file grid) and without them (every vector_type compacted
    to its filters' union)."""
    from nova_bf import compute as C

    cs = [c for c in cases_mod.CASES
          if request.param == "baseline" or c.filter_dict is not None]
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("NOVA_BF_NO_GPU_COMBO", "1")
        with probe(calls={"gpu_eval": (C, "_gpu_evaluate", None),
                          "plan": (C, "_plan_mask_device", None),
                          "union": (C, "_union_keep", None)}) as p:
            got = run(ds, [c.spec() for c in cs], out_tag=f"legacy_{request.param}",
                      device=device, params=PARAMS)
    finally:
        mp.undo()
    return request.param, cs, got, p


def test_legacy_matrix_matches_naive(legacy_matrix, ds, oracle, device):
    from .test_parity_matrix import _filter_from_dict

    regime, cs, got, p = legacy_matrix
    assert p.calls["plan"] == 0 and p.calls["gpu_eval"] > 0
    assert (p.calls["union"] > 0) == (regime == "compacted")
    for c in cs:
        want = oracle.topk(vector_type=c.vector_type, metric=c.metric, k=c.k,
                           filt=_filter_from_dict(ds, c.filter_dict))
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(got[c.name][qi], want[qi], metric=c.metric,
                                        label=f"[{device}/legacy-{regime}] {c.id} q{qi}")
