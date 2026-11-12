"""Datetime bounds placed exactly on stored microsecond instants.

The matrix's date filters sit on midnight bounds over a day-granular column,
so a date conversion that is off by one MICROSECOND is invisible there — and
one was: `dates.parse_scalar_epoch_us` used `int(dt.timestamp() * 1e6)`,
which is a float of seconds scaled up, and lands one µs low (or high) for
about 1 % of sub-second literals. The corpus column is converted exactly (by
Arrow), so the static bound and the stored value disagreed by one µs and a
`gt`/`lte` (or `gte`/`lt`) at exactly a stored instant silently flipped the
rows holding it.

Here the corpus column `event_at` carries µs-precision instants, about half
before 1970, and `corpus.EVENT_AT_PINS` stores the instants where float
arithmetic is wrong in each direction (twice, plus once in another offset
spelling). `cases.DATE_FILTERS` bounds on each with each operator, static and
per-query. Every search runs at k=400 > corpus size, so each comparison is
an exact comparison of the WHOLE eligible set:

  * against naive, whose RFC-3339 parser is integer-only and hand-written
    (`naive.rfc3339_to_epoch_us`), sharing no code with nova-bf or Arrow;
  * against Qdrant, which is handed the RAW bound string, exactly as the
    config holds it (`qdrant_ref` never sees nova-bf's converted µs);
  * against Qdrant's own eligibility count.
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare, corpus, naive, qdrant_ref
from .runner import run, spec

DATES = cases_mod.DATE_FILTERS
K = cases_mod.DATE_K
MODS = cases_mod.MODALITIES


def _name(vt, metric, fname):
    return f"{vt[:2]}{metric[:3]}_{fname}"


@pytest.fixture(scope="session")
def date_run(ds, device):
    specs = [spec(_name(vt, m, f), vector_type=vt, metric=m, k=K, filter=fd)
             for f, fd in DATES.items() for vt, m in MODS]
    return run(ds, specs, out_tag="dates", device=device,
               params={"dense_batch_size": 37, "sparse_batch_size": 29,
                       "multivector_batch_size": 23})


def _eligible(ds, fdict, query):
    return {d.row for d in ds.docs
            if naive.filter_matches(fdict, d.payload, query["payload"], ds.date_fields,
                                    ds.query_date_fields)}


def test_pinned_instants_are_where_the_fixture_says(ds):
    """Guards on the FIXTURE. The equality windows select exactly the pinned
    rows — so the instants really are stored, and A really is stored in two
    spellings of one instant — and float-seconds arithmetic really is off by
    one µs at A, B and C (otherwise this file would test nothing)."""
    from datetime import datetime, timezone

    q0 = ds.queries[0]
    assert _eligible(ds, DATES["d_eq_a"], q0) == {17, 200, 201}
    assert _eligible(ds, DATES["d_eq_a_offset"], q0) == {17, 200, 201}
    assert _eligible(ds, DATES["d_eq_b"], q0) == {40, 250}
    for instant, direction in ((corpus.EVENT_A, -1), (corpus.EVENT_B, +1),
                               (corpus.EVENT_C, -1)):
        exact = naive.rfc3339_to_epoch_us(instant)
        dt = datetime.fromisoformat(instant.replace("Z", "+00:00")).astimezone(timezone.utc)
        assert int(dt.timestamp() * 1_000_000) - exact == direction, instant
    # pre-1970 and post-1970 rows both present, nulls present
    vals = [d.payload["event_at"] for d in ds.docs]
    us = [naive.to_epoch_us(v) for v in vals if v is not None]
    assert min(us) < 0 < max(us) and None in vals


@pytest.mark.parametrize("fname", list(DATES))
def test_date_bounds_match_naive_exactly(fname, date_run, ds, oracle, device):
    for vt, metric in MODS:
        want = oracle.topk(vector_type=vt, metric=metric, k=K, filt=DATES[fname])
        got = date_run[_name(vt, metric, fname)]
        for qi, q in enumerate(ds.queries):
            label = f"[{device}] {vt}-{metric} {fname} q{qi} vs naive"
            assert qi in got, f"{label}: query missing from the output"
            compare.assert_scores_agree(got[qi], want[qi], metric=metric, label=label,
                                        allow_empty=True)
            # k > corpus: the hits ARE the eligible set (no boundary allowance)
            if vt == "dense":
                assert {i for i, _ in got[qi]} == _eligible(ds, DATES[fname], q), label


@pytest.mark.qdrant
@pytest.mark.parametrize("fname", list(DATES))
def test_date_bounds_match_qdrant_exactly(fname, date_run, ds, client, collection, device):
    for vt, metric in MODS:
        want = qdrant_ref.topk(client, collection, ds, vector_type=vt, metric=metric, k=K,
                               filt=DATES[fname])
        got = date_run[_name(vt, metric, fname)]
        for qi in range(len(ds.queries)):
            compare.assert_scores_agree(got[qi], want[qi], metric=metric, allow_empty=True,
                                        label=f"[{device}] {vt}-{metric} {fname} q{qi} vs qdrant")
    for qi, q in enumerate(ds.queries):
        assert qdrant_ref.count(client, collection, DATES[fname], q["payload"],
                                ds.date_fields) == len(_eligible(ds, DATES[fname], q)), (
            f"{fname} q{qi}: Qdrant's eligible count differs from naive's")
