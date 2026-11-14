"""The float16 multivector prune, held to the INDEPENDENT oracles.

`tests/test_mv_fp16.py` pins the bound's arithmetic and
`tests/test_survivor_oracle.py` pins `score_survivors` against an independent
pair-at-a-time reference. Neither can see a prune that
is wrong in a way both arms share — a bound that drops the same real candidate
whichever implementation runs it agrees with itself perfectly. So the pruned
path is also held here, against the naive oracle and, where a server is up,
against live Qdrant.

The prune is CUDA-only in practice: certification delegates to
`twopass.certify_closed_form`, whose accumulation probe runs the Triton kernel,
so a CPU run declines and there is nothing to grade. Hence `_CUDA_ONLY`.

Three knobs, none of them a correctness parameter:

  * `MV_DIM = 64`. The shared `ds` fixture uses 8, and at that width the bound
    is +inf by construction: `closed_form.eps_dot` refuses any dimension below
    the tensor-core MMA K-dimension, because the accumulation envelope is not
    derived there. +inf is the SAFE refusal -- nothing can ever be pruned --
    but it also makes every assertion here vacuous, so this file builds its own
    corpus at a width the bound actually covers.
  * `multivector_min_prune_rate=0.0` disables the throughput gate, which would
    otherwise disengage below the 0.70 production floor.
  * `NOVA_BF_MV_PRUNE_AUDIT=1` grades every slice. It costs the exact MaxSim
    the prune exists to avoid, which is why it is off in production.

EVERY test here asserts that pruning actually happened. A prune that pruned
nothing agrees with every oracle perfectly, and that is the failure mode this
file exists to rule out.
"""

from __future__ import annotations

import contextlib

import pytest

from . import compare, naive, qdrant_ref, runner
from .devices import has_cuda

_CUDA_ONLY = pytest.mark.skipif(
    not has_cuda(), reason="the float16 multivector prune certifies on CUDA only")

K = 25
MV_DIM = 64            # see the module docstring: below this the bound is +inf
MV_CASES = [("mv_dot", "dot"), ("mv_cos", "cosine")]
MV_IDS = [f"{n}-{m}" for n, m in MV_CASES]

PRUNE_PARAMS = {
    "multivector_prune": "fp16",
    "multivector_min_prune_rate": 0.0,
}
# Several slices per file (37 docs each, pinned test-only via `mv_tiles`), so
# the top-K fills and the threshold the prune works against is genuinely a
# running one.
PRUNE_TILES = (37, 10**6)


@contextlib.contextmanager
def _wide_tokens():
    """Widen the corpus generator's token width for the duration of a call.

    `corpus.MV_DIM` is a module constant read by both the generator and
    `qdrant_ref.create_collection`, so the dataset and its Qdrant collection
    have to be built while it is patched. Restored immediately afterwards so
    nothing else in the session sees the wider width.
    """
    from . import corpus as corpus_mod

    was = corpus_mod.MV_DIM
    corpus_mod.MV_DIM = MV_DIM
    try:
        yield
    finally:
        corpus_mod.MV_DIM = was


@pytest.fixture(scope="session")
def mv_ds(tmp_path_factory):
    from . import corpus as corpus_mod

    with _wide_tokens():
        return corpus_mod.build(tmp_path_factory.mktemp("bf_mvprune"))


@pytest.fixture(scope="session")
def mv_oracle(mv_ds):
    return naive.Oracle(mv_ds.docs, mv_ds.queries, mv_ds.date_fields,
                        mv_ds.query_date_fields)


@pytest.fixture(scope="session")
def mv_collection(client, mv_ds):
    with _wide_tokens():
        name = qdrant_ref.create_collection(client, mv_ds)
    yield name
    client.delete_collection(name)


@pytest.fixture(autouse=True)
def _clean_prune_state():
    """The prune's counters and any audit disable are process-wide."""
    from nova_bf import compute, mv_fp16

    mv_fp16.reset_audit()
    before = dict(compute._MV_PRUNE)
    compute._MV_PRUNE.update({k: (0.0 if isinstance(v, float) else 0)
                              for k, v in compute._MV_PRUNE.items()})
    yield
    compute._MV_PRUNE.update(before)
    mv_fp16.reset_audit()


def _specs():
    return [{"name": n, "vector_type": "multivector", "metric": m, "k": K}
            for n, m in MV_CASES]


def _run(ds, tag, **extra):
    return runner.run(ds, _specs(), out_tag=tag, device="cuda",
                      params=PRUNE_PARAMS | extra, mv_tiles=PRUNE_TILES)


def _pruned_fraction():
    from nova_bf import compute

    p = compute._MV_PRUNE
    assert p["calls"], "the prune was never called — it declined every slice"
    assert p["pairs"], "the prune ran but graded no pairs"
    return p["pruned"] / p["pairs"]


# --- the fixture has to be able to show a difference --------------------------

@_CUDA_ONLY
def test_the_fixture_actually_prunes(mv_ds):
    """Non-vacuity. Everything below compares a pruned run against an oracle;
    if nothing is ever pruned those comparisons prove only that the unpruned
    path works, which other files already cover."""
    _run(mv_ds, "mvprune_vacuity")
    frac = _pruned_fraction()
    assert frac > 0.01, (
        f"only {frac:.4%} of pairs were pruned — every parity assertion in "
        "this file is vacuous at that rate")


# --- against the independent oracles ------------------------------------------

@_CUDA_ONLY
@pytest.mark.parametrize("name,metric", MV_CASES, ids=MV_IDS)
def test_the_pruned_path_agrees_with_the_naive_oracle(mv_ds, mv_oracle, name, metric):
    got = _run(mv_ds, f"mvprune_naive_{name}")[name]
    assert _pruned_fraction() > 0.01
    want = mv_oracle.topk(vector_type="multivector", metric=metric, k=K, filt=None)
    for qi in range(len(mv_ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"[cuda] pruned {name} q{qi}: nova-bf vs naive")


@_CUDA_ONLY
@pytest.mark.qdrant
@pytest.mark.parametrize("name,metric", MV_CASES, ids=MV_IDS)
def test_the_pruned_path_agrees_with_live_qdrant(mv_ds, client, mv_collection,
                                                 name, metric):
    """nova-bf's job is grading Qdrant recall, so ground truth its own exact
    MaxSim disagrees with is wrong by definition — pruned or not."""
    got = _run(mv_ds, f"mvprune_qdrant_{name}")[name]
    assert _pruned_fraction() > 0.01
    want = qdrant_ref.topk(client, mv_collection, mv_ds, vector_type="multivector",
                           metric=metric, k=K, filt=None)
    for qi in range(len(mv_ds.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=metric,
            label=f"[cuda] pruned {name} q{qi}: nova-bf vs qdrant")


@_CUDA_ONLY
@pytest.mark.parametrize("name,metric", MV_CASES, ids=MV_IDS)
def test_pruning_changes_nothing_a_prune_disabled_run_returns(mv_ds, name, metric):
    """The tightest comparison available: same engine, same inputs, prune the
    only difference."""
    pruned = _run(mv_ds, f"mvprune_on_{name}")[name]
    assert _pruned_fraction() > 0.01
    plain = runner.run(mv_ds, _specs(), out_tag=f"mvprune_off_{name}",
                       device="cuda",
                       params={"multivector_prune": "off"},
                       mv_tiles=PRUNE_TILES)[name]
    for qi in range(len(mv_ds.queries)):
        compare.assert_scores_agree(
            pruned[qi], plain[qi], metric=metric,
            label=f"[cuda] {name} q{qi}: prune on vs prune off")


# --- the audit ----------------------------------------------------------------

@_CUDA_ONLY
def test_the_audit_grades_a_real_run_and_finds_nothing_wrong(mv_ds, monkeypatch):
    """The audit scores every pair exactly and checks BOTH the bound
    (`upper >= exact`) and the decision (nothing pruned that reaches its
    threshold), against the UNPRUNED scorer — which is also the claim that the
    bound is scorer-agnostic."""
    from nova_bf import mv_fp16

    monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", "1")
    _run(mv_ds, "mvprune_audit_clean")

    a = mv_fp16.audit_stats()
    assert a["graded_slices"] > 0, "the audit never ran"
    assert a["graded_pairs"] > 1000, f"only {a['graded_pairs']} pairs graded"
    assert a["correct_prune"] > 0, "nothing was pruned, so nothing was proved"
    assert a["false_prune"] == 0, a["disabled"]
    assert a["bound_violations"] == 0, a["disabled"]
    # The check that DISABLES pruning process-wide, so a spurious failure here
    # would silently turn the prune off for the rest of the run while the
    # assertions above still passed on the first graded slice.
    assert a["graded_scores"] > 0, "no survivor scores were graded"
    assert a["score_mismatch"] == 0, a["disabled"]
    assert not mv_fp16.audit_disabled(), mv_fp16.audit_disabled()
    assert a["min_headroom"] is not None and a["min_headroom"] >= 0.0


@_CUDA_ONLY
def test_the_audit_catches_a_bound_that_is_too_small(mv_ds, monkeypatch):
    """The audit's value is that it FAILS. Break the bound by subtracting a
    constant from every upper bound and the run must be caught and stopped —
    a grader that only ever reports success on a healthy run is untested."""
    from nova_bf import mv_fp16

    real_upper = mv_fp16.upper_bound
    monkeypatch.setattr(
        mv_fp16, "upper_bound",
        lambda approx, slack_: real_upper(approx, slack_) - 1.0e3)
    monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", "1")

    _run(mv_ds, "mvprune_audit_broken")

    a = mv_fp16.audit_stats()
    assert a["bound_violations"] > 0, "a bound 1000 below the truth went unnoticed"
    assert a["false_prune"] > 0, "candidates above their threshold were dropped"
    # NOT `graded_scores > 0` here: a bound 1000 below the truth prunes
    # everything, so there are no survivors left to grade and 0 is correct.
    # That assertion belongs in the clean-run test above, where survivors exist.
    assert a["disabled"], "the audit found violations but did not disable pruning"
    assert a["worst_bound_gap"] > 0.0
