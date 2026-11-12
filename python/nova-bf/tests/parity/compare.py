"""Comparing two engines' top-K without pretending float reductions agree.

The naive way to compare two top-K lists — assert the id sequences are equal —
fails for a reason that has nothing to do with correctness: three
implementations (nova-bf's tiled matmuls, Qdrant's own reductions, this
harness's Python loops) sum the same products in three different orders, and
float addition is not associative. Two documents whose true scores differ by
less than the accumulated error can legitimately swap, and one sitting exactly
at the K-th position can legitimately fall out of one engine's list and into
another's.

So the comparison here asserts the two properties that actually distinguish a
correct implementation from a broken one:

  1. **Agreement where both engines answered.** For every id in both lists,
     the scores match to tolerance. A wrong metric, a wrong norm, a dropped
     token, a mis-mapped row — all of them move a score far more than a
     reduction-order ULP.
  2. **Disagreement only at the boundary.** An id in one list and not the
     other is accepted ONLY if its score sits within tolerance of the OTHER
     list's K-th score. That is the signature of a near-tie straddling the
     cutoff. An id missing from the middle of the ranking — a filter that
     wrongly excluded a row, a non-candidate wrongly included — is not at the
     boundary and fails.

`assert_same_membership` is the stricter variant for cases where the two sides
must agree exactly on WHICH rows were eligible (filter semantics), independent
of how they ranked them.
"""

from __future__ import annotations

import os

# THE score tolerance of the harness — every boundary-aware comparison reads it
# from here. Applied as `tol * (1 + |score|)`: relative for large scores, so
# euclidean (scores are -distance, ~-11 in the shared corpus) gets a
# distance-scaled allowance, and absolute near zero.
#
# Measured worst `|nova-bf - naive| / (1 + |naive|)` on CPU (fp32 matmul vs a
# float64 Python loop, matrix-style shared-Gram run, dense_batch_size=37):
#   dot 3.8e-7, cosine 1.1e-7, euclidean 1.2e-7 (the Gram expansion
#   ‖q‖² − 2q·c + ‖c‖² included).
# 1e-5 is ~25x the worst of those, and still ~10x BELOW the error of scoring
# fp16-rounded inputs (≈1e-4 relative), which the old 2e-3 euclidean allowance
# let through unnoticed (mutant `eucl_fp16_inputs_shared`).
#
# GPU (cuBLAS fp32, TF32 off) reduces in a different order and must still pass
# at this tolerance; if a run needs more, set NOVA_BF_PARITY_SCORE_RTOL to the
# measured error and record it, rather than widening this default. (TF32 tests
# pass their own explicit, TF32-sized tolerance.)
SCORE_RTOL = float(os.environ.get("NOVA_BF_PARITY_SCORE_RTOL", "") or 1e-5)
TOL = {"dot": SCORE_RTOL, "cosine": SCORE_RTOL, "euclidean": SCORE_RTOL}


def scores_are_reproducible(device: str, vector_type: str,
                            multivector_kernel: str = "torch") -> bool:
    """Can two runs over the same input be expected to produce identical score
    BITS — the precondition for any bit-equality assertion?
    """
    if vector_type != "multivector" or device != "cuda":
        return True
    return multivector_kernel == "triton_reduce"


def assert_same_ranking(a, b, *, metric: str, label: str,
                        device: str = "cpu", vector_type: str = "dense",
                        multivector_kernel: str = "torch"):
    """`assert_identical` where score bits are reproducible, and the strongest
    available claim where they are not: same ids, in the same order, with
    scores agreeing to the metric's tolerance.

    Note what is NOT weakened even in the non-reproducible case — the id
    sequence is still compared exactly. The observed nondeterminism moves the
    last mantissa bit and has never reordered the result, so a reordering
    remains a failure."""
    if scores_are_reproducible(device, vector_type, multivector_kernel):
        assert_identical(a, b, label=label)
        return
    assert_same_membership(a, b, label=label)
    assert [int(i) for i, _ in a] == [int(i) for i, _ in b], (
        f"{label}: ranking differs (scores here are not bit-reproducible, but "
        f"the ORDER still must not move)")
    assert_scores_agree(a, b, metric=metric, label=label)


def as_dict(hits) -> dict[int, float]:
    return {int(i): float(s) for i, s in hits}


def assert_well_formed(hits, *, label: str, side: str = "result"):
    """A top-K list on its own: ids unique, scores non-increasing.

    Both are properties the boundary-aware comparison CANNOT see — it reads the
    lists through `as_dict`, which collapses a duplicated id to one entry and
    forgets the order — so a result with the same hit twice, or written
    worst-first, passed it. Scores are compared exactly (no tolerance): every
    engine sorts by its OWN scores, so its own list is never out of order."""
    ids = [int(i) for i, _ in hits]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, f"{label}: {side} returns id(s) {dup} more than once"
    scores = [float(s) for _, s in hits]
    for r in range(len(scores) - 1):
        assert scores[r] >= scores[r + 1], (
            f"{label}: {side} is not best-first at rank {r}: "
            f"{scores[r]!r} then {scores[r + 1]!r} (ids {ids[r]}, {ids[r + 1]})")


def assert_scores_agree(a, b, *, metric: str, label: str, tol: float | None = None,
                        allow_empty: bool = False, check_len: bool = True):
    """The full boundary-aware comparison — see this module's docstring.

    `a` and `b` are `[(id, score), …]` lists from the two engines being
    compared; neither is privileged. On top of the boundary rule this asserts:

      * each side is well-formed (`assert_well_formed`: unique ids, best-first);
      * both sides return the SAME NUMBER of hits (`check_len`). An exact
        top-K is `min(k, eligible)` long whatever the float noise, so a length
        difference is always an eligibility disagreement — and the boundary
        rule alone accepted a short list whose missing tail sat at the cutoff;
      * both returning nothing is a failure unless `allow_empty` — the default
        keeps the vacuity guard (a filter that matched nothing on both sides
        tests nothing), while dead-query tests opt in, since there an empty
        list on both sides IS the answer.
    """
    tol = TOL[metric] if tol is None else tol
    a, b = list(a), list(b)
    assert_well_formed(a, label=label, side="left")
    assert_well_formed(b, label=label, side="right")
    if check_len:
        assert len(a) == len(b), (
            f"{label}: left returned {len(a)} hits, right {len(b)} — an exact "
            f"top-K has one length, so the two disagree on what was eligible")
    da, db = as_dict(a), as_dict(b)
    if not da and not db:
        assert allow_empty, f"{label}: both sides returned nothing"
        return
    assert da, f"{label}: left side returned nothing while right returned {len(db)}"
    assert db, f"{label}: right side returned nothing while left returned {len(da)}"

    for i in sorted(set(da) & set(db)):
        assert abs(da[i] - db[i]) <= tol * (1 + abs(db[i])), (
            f"{label}: id={i} scored {da[i]!r} vs {db[i]!r} "
            f"(tolerance {tol} relative)")

    a_floor, b_floor = min(da.values()), min(db.values())
    for i in sorted(set(da) - set(db)):
        assert abs(da[i] - b_floor) <= tol * (1 + abs(b_floor)), (
            f"{label}: id={i} (score {da[i]!r}) is in the left top-K only, and is "
            f"not at the right side's top-K boundary {b_floor!r} — this is a "
            f"ranking disagreement, not a boundary tie")
    for i in sorted(set(db) - set(da)):
        assert abs(db[i] - a_floor) <= tol * (1 + abs(a_floor)), (
            f"{label}: id={i} (score {db[i]!r}) is in the right top-K only, and is "
            f"not at the left side's top-K boundary {a_floor!r} — this is a "
            f"ranking disagreement, not a boundary tie")


def assert_same_membership(a, b, *, label: str):
    """Exact set equality of returned ids — no boundary allowance.

    Only for comparisons where a difference could not possibly be a float
    artefact: the same run on two devices, or a filter's ELIGIBLE set checked
    against the naive predicate over the whole corpus.
    """
    a, b = list(a), list(b)
    ia, ib = {int(i) for i, _ in a}, {int(i) for i, _ in b}
    assert len(ia) == len(a) and len(ib) == len(b), (
        f"{label}: a side repeats an id (left {len(a)} hits / {len(ia)} distinct, "
        f"right {len(b)} / {len(ib)})")
    assert ia == ib, (
        f"{label}: id sets differ — only left: {sorted(ia - ib)}, "
        f"only right: {sorted(ib - ia)}")


def assert_identical(a, b, *, label: str, tol: float = 0.0):
    """Same ids, same order, same scores. Used where nothing legitimate may
    differ — e.g. a `rows` subset against the same search without one."""
    la, lb = list(a), list(b)
    assert [int(i) for i, _ in la] == [int(i) for i, _ in lb], (
        f"{label}: ranking differs\n  left:  {[int(i) for i, _ in la][:12]}\n"
        f"  right: {[int(i) for i, _ in lb][:12]}")
    for (ia, sa), (_, sb) in zip(la, lb):
        assert abs(sa - sb) <= tol * (1 + abs(sb)), (
            f"{label}: id={ia} scored {sa!r} vs {sb!r}")
