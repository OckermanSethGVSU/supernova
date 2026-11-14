"""The `params` knobs nothing else in the harness turns.

`test_parity_invariance` pins the axes the matrix run fixes — tiling, sharing,
query-row subsetting. It does not cover every knob, and a knob that silently
changes ground truth is exactly as bad whether or not a test happens to set it.
Cross-referencing `ParamsConfig` against the suite leaves five never varied
here; these are the three that gate a distinct CODE PATH rather than only a
pool size:

  * `multivector_budget_mib` — splits the MaxSim similarity tile by
    `max_doc_tokens = elements // max_query_tokens`. At the default a parity
    slice is one tile and the per-block loop runs once.
  * `merge_batch_size` — rows per fold in `nova bf merge`. The default resolves
    to a ceiling far above any parity corpus, so the merge always folds in ONE
    batch and the multi-batch fold is never exercised against an oracle.
  * `merge_ranged_reads` — how a partial's bytes are fetched. Every production
    config sets it true; the harness has only ever run it false.

Each is checked against the naive oracle, not merely against another nova-bf
run, so a knob that corrupted both arms identically still fails.
"""

from __future__ import annotations

import pytest

from nova_bf.merge import run_merge

from . import cases as cases_mod
from . import compare
from .runner import build_config, read_results, pinned_device, run, spec

import mv_tiles

try:  # `run_compute` moved once; keep the import obvious if it moves again
    from nova_bf.compute import run_compute
except ImportError:  # pragma: no cover
    from nova_bf import run_compute  # type: ignore

MV_PROBES = [
    cases_mod.CASES_BY_NAME[n] for n in (
        "mudot_nofilter", "mucos_match", "mudot_pqmatch", "mucos_matchtext",
    )
]
MV_IDS = [c.id for c in MV_PROBES]


def _filter_of(case, ds):
    from .test_parity_matrix import _filter_from_dict

    return _filter_from_dict(ds, case.filter_dict)


# ---------------------------------------------------------------------------
# multivector_budget_mib
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_spy(monkeypatch):
    """Record what the token budget actually did to the blocking.

    Without this the budget tests are ONE-SIDED: they assert the answer is
    unchanged, which a knob that does NOTHING satisfies perfectly. That was not
    hypothetical — disabling the split branch, and even forcing the config
    field to `None`, left every test in this file passing.

    `_ragged_batch_ranges` is where the budget lands, so spy there and record
    both the blocking it produced and the blocking it WOULD have produced
    unbudgeted. A test can then assert the tile really split.
    """
    from nova_bf import compute

    seen: list[tuple[int | None, int, int]] = []
    real = compute._ragged_batch_ranges

    def spy_ranges(offsets, max_rows, max_tokens):
        got = real(offsets, max_rows, max_tokens)
        unbudgeted = real(offsets, max_rows, None)
        seen.append((max_tokens, len(got), len(unbudgeted)))
        return got

    monkeypatch.setattr(compute, "_ragged_batch_ranges", spy_ranges)
    return seen


def _assert_the_budget_split_something(seen, budget):
    assert seen, "the multivector blocking never ran; this case is not multivector"
    assert any(mt is not None for mt, _, _ in seen), (
        f"budget={budget} never reached `_ragged_batch_ranges` — the knob is "
        f"being ignored, and an 'answer unchanged' assertion cannot see that")
    assert any(got > unb for _, got, unb in seen), (
        f"budget={budget} produced no more blocks than an unbudgeted run "
        f"({[(m, g, u) for m, g, u in seen[:4]]}) — the tile never split, so "
        f"this case proves nothing about split correctness")


@pytest.mark.parametrize("budget", [16, 64, 256, 4096])
@pytest.mark.parametrize("case", MV_PROBES, ids=MV_IDS)
def test_a_token_budget_does_not_change_the_answer(budget, case, ds, oracle, device,
                                                   budget_spy, monkeypatch):
    """Splitting the MaxSim tile must be invisible.

    `budget` is in score ELEMENTS (converted to `multivector_budget_mib`).
    The budget caps `n_query_tokens x n_doc_tokens` per tile, so a small one
    turns each slice into many blocks and the per-query MaxSim is assembled
    from several partial reductions instead of one. `8` is small enough that
    the parity corpus (ragged token counts, some documents with none at all)
    splits on nearly every slice.
    """
    # The derived item caps would otherwise bind before the token budget does
    # (the even split gives each axis ~sqrt(elements) tokens, and with 8
    # short queries the doc axis is then capped by ITEMS) -- precisely why
    # these tests were once one-sided. So the shape is pinned (test-only) to
    # 64 docs per slice and all queries in one tile, giving the budget room to
    # be the constraint that splits.
    mv_tiles.pin(monkeypatch, 64, 10**6)
    got = run(ds, [case.spec()], out_tag=f"tb{budget}_{case.name}", device=device,
              params={"multivector_budget_mib": mv_tiles.mib(budget)})[case.name]
    _assert_the_budget_split_something(budget_spy, budget)
    want = oracle.topk(vector_type=case.vector_type, metric=case.metric,
                       k=case.k, filt=_filter_of(case, ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] token_budget={budget} {case.id} q{qi}")


@pytest.mark.parametrize("queries_per_tile", [1, 2, 4, 10**6])
@pytest.mark.parametrize("case", MV_PROBES, ids=MV_IDS)
def test_a_tiny_token_budget_equals_an_unbudgeted_run(case, ds, device, budget_spy,
                                                     monkeypatch, queries_per_tile):
    """Exact hit-for-hit, not merely oracle-tolerable.

    A tile split changes the ORDER the per-token maxima are reduced in, so
    scores may move by an ulp — but a document may not enter or leave the
    ranking, and with the parity corpus's well-separated scores the hit list
    must be identical.
    """
    # 64, not 8: at budget 8 `max_doc_tokens` is 1 on this corpus, so every
    # tile is a single row and a tile-BOUNDARY bug cannot show up. 64 gives
    # ~12 tokens per tile, which really blocks.
    # queries_per_tile < all: the small run ALSO tiles the query axis (1 / 2 /
    # 4 queries per tile, what the old budget-derived query block gave at
    # budgets 16 / 64 / 256), so query-tile boundaries are checked hit-for-hit
    # against the all-queries reference too.
    mv_tiles.pin(monkeypatch, 64, queries_per_tile)   # see the test above
    small = run(ds, [case.spec()], out_tag=f"tbsmall{queries_per_tile}_{case.name}",
                device=device,
                params={"multivector_budget_mib": mv_tiles.mib(16)})[case.name]
    _assert_the_budget_split_something(budget_spy, 16)
    # Same row batching, every query in one tile, default budget (never binds
    # here): the only differences are the token split and the query tiling.
    mv_tiles.pin(monkeypatch, 64, 10**6)
    none = run(ds, [case.spec()], out_tag=f"tbnone_{case.name}", device=device,
               params={})[case.name]
    for qi in range(len(ds.queries)):
        assert [r for r, _ in small[qi]] == [r for r, _ in none[qi]], (
            f"[{device}] {case.id} q{qi}: a token budget changed the ranking")


# ---------------------------------------------------------------------------
# the merge knobs
# ---------------------------------------------------------------------------

MERGE_SPECS = [
    ("mk_dense", "dense", "cosine", None),
    ("mk_sparse", "sparse", "dot", {"must": [{"field": "language", "match": "eng"}]}),
    ("mk_mv", "multivector", "dot", None),
]
MERGE_IDS = [s[0] for s in MERGE_SPECS]


def _sharded(ds, device, *, tag, params=None, num_jobs=3):
    """`num_jobs` compute ranks, then one merge — the production shape."""
    specs = [spec(n, vector_type=vt, metric=m, k=cases_mod.K, filter=f)
             for n, vt, m, f in MERGE_SPECS]
    cfg = build_config(ds, specs, out_tag=f"{tag}_{device or 'auto'}",
                       params=params)
    with pinned_device(device):
        for rank in range(num_jobs):
            run_compute(cfg, num_jobs=num_jobs, job_rank=rank)
        return read_results(run_merge(cfg))


@pytest.fixture(scope="module")
def merge_default(ds, device):
    return _sharded(ds, device, tag="mk_default")


@pytest.mark.parametrize("batch_rows", [1, 2, 3])
@pytest.mark.parametrize("name", MERGE_IDS)
def test_a_small_merge_batch_gives_the_same_merged_answer(
    batch_rows, name, ds, device, merge_default
):
    """`merge_batch_size` folds the partials in row batches. The default
    resolves to a ceiling far above any parity corpus, so the merge has only
    ever folded in ONE batch here — the multi-batch fold, which is what every
    production merge actually runs, was untested against an oracle.

    Batches of 1-3 rows against 8 queries force several folds per search.
    """
    got = _sharded(ds, device, tag=f"mk_b{batch_rows}",
                   params={"merge_batch_size": batch_rows})
    for qi in range(len(ds.queries)):
        assert [r for r, _ in got[name][qi]] == [r for r, _ in merge_default[name][qi]], (
            f"[{device}] merge_batch_size={batch_rows} changed {name} at q{qi}")


def _ranged_merge(ds, device, *, tag, ranged):
    """`_sharded` with `merge_ranged_reads=ranged`, and the ranged path made
    REACHABLE: `io._ranged_download` is gated on a file of at least
    `_RANGED_GET_MIN_BYTES` (256 MiB), which no parity partial comes near, so
    the gate is lowered to 1 byte and the range size to 4 KiB (so a partial is
    fetched as several ranges, not one). Returns the merged results and how
    many partial files were read through the ranged path."""
    from nova_bf import io as io_mod

    calls = {"n": 0}
    real = io_mod.Store._ranged_download

    def counting(self, read_path, size):
        calls["n"] += 1
        return real(self, read_path, size)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(io_mod, "_RANGED_GET_MIN_BYTES", 1)
        mp.setattr(io_mod, "_RANGED_GET_BYTES", 4096)
        mp.setattr(io_mod.Store, "_ranged_download", counting)
        got = _sharded(ds, device, tag=tag,
                       params={"merge_batch_size": 2, "merge_ranged_reads": ranged})
    finally:
        mp.undo()
    return got, calls["n"]


@pytest.fixture(scope="module")
def merge_knobs_run(ds, device):
    return _ranged_merge(ds, device, tag="mk_both", ranged=True)


def test_the_ranged_merge_read_path_is_reached_only_by_its_knob(ds, device,
                                                                merge_knobs_run):
    """The knob, not the lowered gate, is what routes the merge through the
    ranged reader: on, every search's partials (3 searches x 3 ranks) go
    through it; off, with the same lowered gate, none do."""
    _, n_on = merge_knobs_run
    assert n_on >= len(MERGE_SPECS) * 3, n_on
    _, n_off = _ranged_merge(ds, device, tag="mk_unranged", ranged=False)
    assert n_off == 0, n_off


@pytest.mark.parametrize("name", MERGE_IDS)
def test_the_merge_knobs_still_agree_with_the_oracle(name, ds, oracle, device,
                                                     merge_knobs_run, merge_default):
    """Both knobs at once (a 2-row merge batch, ranged partial reads), against
    the oracle rather than against another nova-bf run — two merges agreeing
    with each other and both being wrong is the failure a self-comparison
    cannot see — and then against the default merge, which must not move."""
    got, _ = merge_knobs_run
    idx = MERGE_IDS.index(name)
    _, vt, metric, filt = MERGE_SPECS[idx]
    want = oracle.topk(vector_type=vt, metric=metric, k=cases_mod.K,
                       filt=None if filt is None
                       else _filter_of(cases_mod.CASES_BY_NAME["decos_match"], ds))
    for qi in range(len(ds.queries)):
        compare.assert_scores_agree(
            got[name][qi], want[qi], metric=metric,
            label=f"[{device}] merge knobs {name} q{qi}")
        assert [r for r, _ in got[name][qi]] == [r for r, _ in merge_default[name][qi]], (
            f"[{device}] ranged merge changed {name} at q{qi}")
