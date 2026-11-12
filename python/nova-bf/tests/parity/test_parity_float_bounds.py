"""Static float `range` bounds placed EXACTLY on a stored value.

The matrix's float range (`rating > 2.5`) sits between stored values, so
`gt` and `gte` admit the same rows there and a `gt` evaluated as `gte` (or
`lt` as `lte`) passes every comparison. The only bound that tells the
operators apart is one EQUAL to a stored value: then the row holding it is
in for `gte`/`lte` and out for `gt`/`lt`, and nothing else differs.

So each operator, and the two-sided forms, is put on values read back from
the corpus itself — float64 `rating` (no nulls) and nullable float64 `weight`
— at rows spread across the value range. k exceeds the corpus, so the
result IS the eligible set, compared exactly (membership, no boundary
allowance) against the naive predicate and against Qdrant, plus the row that
holds the bound pinned in or out by name.
"""

from __future__ import annotations

import pytest

from . import compare, naive, qdrant_ref
from .runner import run, spec
from .test_parity_matrix import _filter_from_dict

K_ALL = 400          # > the 300-row corpus: every eligible row comes back
# Rows whose stored value becomes the bound: spread over the value range and
# over all four corpus files (97, 61, 79, 63 rows).
BOUND_ROWS = (5, 123, 170, 250)
FIELDS = ("rating", "weight")
OPS = ("gt", "gte", "lt", "lte")


def _value(ds, field, row):
    v = ds.doc(row).payload[field]
    assert isinstance(v, float), (field, row, v)
    return v


def _cells(ds):
    """`[(name, filter_dict, field, row, op)]` — one per (field, bound row,
    operator), plus two-sided ranges with BOTH ends on stored values, and a
    `must_not` of a strict bound (whose complement is the non-strict one)."""
    out = []
    for field in FIELDS:
        rows = [r for r in BOUND_ROWS if ds.doc(r).payload[field] is not None]
        assert len(rows) >= 3, (field, rows)
        for row in rows:
            v = _value(ds, field, row)
            for op in OPS:
                out.append((f"fb_{field}_{row}_{op}",
                            {"must": [{"field": field, "range": {op: v}}]}, field, row, op))
            out.append((f"fb_{field}_{row}_notgt",
                        {"must_not": [{"field": field, "range": {"gt": v}}]},
                        field, row, "not_gt"))
        lo, hi = sorted((_value(ds, field, rows[0]), _value(ds, field, rows[1])))
        out.append((f"fb_{field}_gt_lte", {"must": [{"field": field,
                    "range": {"gt": lo, "lte": hi}}]}, field, None, None))
        out.append((f"fb_{field}_gte_lt", {"must": [{"field": field,
                    "range": {"gte": lo, "lt": hi}}]}, field, None, None))
    return out


@pytest.fixture(scope="module")
def cells(ds):
    return _cells(ds)


@pytest.fixture(scope="module")
def bound_run(ds, cells, device):
    return run(ds, [spec(n, vector_type="dense", metric="dot", k=K_ALL, filter=f)
                    for n, f, *_ in cells],
               out_tag="float_bounds", device=device, params={"dense_batch_size": 37})


def _eligible(ds, fdict, qi):
    q = ds.queries[qi]
    return [d.row for d in ds.docs
            if naive.filter_matches(fdict, d.payload, q["payload"], ds.date_fields,
                                    ds.query_date_fields)]


def test_the_bound_rows_decide_something(ds, cells):
    """Guards the fixture: the bound row's own value is held by that row ONLY
    (so `gt` vs `gte` differ by exactly that row), and every cell leaves a
    non-trivial eligible set."""
    for name, fdict, field, row, op in cells:
        n = len(_eligible(ds, fdict, 0))
        assert 0 < n < len(ds.docs), (name, n)
        if row is not None:
            v = _value(ds, field, row)
            holders = [d.row for d in ds.docs if d.payload[field] == v]
            assert holders == [row], (name, holders)


def test_float_bounds_on_stored_values_match_naive_exactly(ds, cells, bound_run, oracle,
                                                          device):
    for name, fdict, field, row, op in cells:
        filt = _filter_from_dict(ds, fdict)
        want = oracle.topk(vector_type="dense", metric="dot", k=K_ALL, filt=filt)
        for qi in range(len(ds.queries)):
            got = bound_run[name][qi]
            label = f"[{device}] {name} q{qi}"
            assert sorted(r for r, _ in got) == sorted(_eligible(ds, fdict, qi)), (
                f"{label}: returned rows are not the eligible set")
            compare.assert_same_membership(got, want[qi], label=label)
            compare.assert_scores_agree(got, want[qi], metric="dot", label=label)
            if row is not None:
                inside = op in ("gte", "lte", "not_gt")
                assert (row in {r for r, _ in got}) == inside, (
                    f"{label}: row {row} holds the bound {_value(ds, field, row)!r}; "
                    f"`{op}` must {'admit' if inside else 'exclude'} it")


@pytest.mark.qdrant
def test_float_bounds_on_stored_values_match_qdrant_exactly(ds, cells, bound_run,
                                                           client, collection, device):
    for name, fdict, *_ in cells:
        want = qdrant_ref.topk(client, collection, ds, vector_type="dense",
                               metric="dot", k=K_ALL, filt=_filter_from_dict(ds, fdict))
        for qi in range(len(ds.queries)):
            label = f"[{device}] {name} q{qi} vs qdrant"
            compare.assert_same_membership(bound_run[name][qi], want[qi], label=label)
            compare.assert_scores_agree(bound_run[name][qi], want[qi], metric="dot",
                                        label=label)
