"""The harness's own instruments, mutated: each must REJECT a wrong answer.

A comparison helper that accepts everything makes every parity test pass, so
the helpers are tested the way the suite tests nova-bf — by feeding them
answers that are wrong in a realistic way and requiring a failure:

  * `compare.assert_scores_agree` against a reversed list, a duplicated hit, a
    dropped tail, a hit with a wrong score, an extra non-boundary hit, and two
    empty lists without `allow_empty`;
  * `naive` against the semantics it claims (type-strict match, empty phrase,
    empty MatchAny, null per-query values, integer-exact dates, raw-dict
    parsing that cannot be fed nova-bf's converted bounds by accident).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from . import compare, naive

GOOD = [(10, 0.9), (11, 0.8), (12, 0.7), (13, 0.6), (14, 0.5)]


def _rejects(a, b, **kw):
    with pytest.raises(AssertionError):
        compare.assert_scores_agree(a, b, metric="cosine", label="mutant", **kw)


def test_identical_lists_agree():
    compare.assert_scores_agree(GOOD, list(GOOD), metric="cosine", label="same")


def test_rejects_a_reversed_ranking():
    _rejects(GOOD[::-1], GOOD)
    _rejects(GOOD, GOOD[::-1])


def test_rejects_a_duplicated_hit():
    dup = GOOD[:4] + [GOOD[3]]
    _rejects(dup, GOOD)
    with pytest.raises(AssertionError):
        compare.assert_same_membership(dup, GOOD[:4], label="dup")


def test_rejects_a_dropped_tail_even_at_the_boundary():
    """The boundary rule alone accepts a missing last hit (it sits AT the
    cutoff); only the length check catches an eligibility disagreement."""
    _rejects(GOOD[:4], GOOD)
    _rejects(GOOD, GOOD[:4])


def test_rejects_a_wrong_score():
    _rejects([(i, s + (0.01 if i == 12 else 0.0)) for i, s in GOOD], GOOD)


def test_rejects_an_extra_hit_off_the_boundary():
    _rejects([(99, 0.95)] + GOOD[:4], GOOD)


def test_accepts_a_genuine_boundary_tie():
    tie = GOOD[:4] + [(77, 0.5 + 1e-7)]
    compare.assert_scores_agree(tie, GOOD, metric="cosine", label="tie")


def test_empty_needs_allow_empty():
    _rejects([], [])
    compare.assert_scores_agree([], [], metric="cosine", label="dead", allow_empty=True)
    _rejects([], GOOD, allow_empty=True)
    _rejects(GOOD, [], allow_empty=True)


# ----------------------------------------------------------- naive itself

def _m(cond, value, row=None):
    return naive.filter_matches({"must": [cond]}, {"f": value}, row or {}, {})


def test_match_is_type_strict():
    assert _m({"field": "f", "match": 1}, 1)
    assert not _m({"field": "f", "match": 1}, True)
    assert not _m({"field": "f", "match": True}, 1)
    assert not _m({"field": "f", "match": "1"}, 1)
    assert _m({"field": "f", "match": [1, 2]}, 2)
    assert not _m({"field": "f", "match": [True]}, 1)
    assert not _m({"field": "f", "match_from_query": "c"}, 1, {"c": True})
    assert _m({"field": "f", "match": 3.0}, 3)       # whole-number float on an int


def test_empty_and_null_values_match_nothing():
    assert not _m({"field": "f", "match": []}, "a")
    assert not _m({"field": "f", "match_from_query": "c"}, "a", {"c": []})
    assert not _m({"field": "f", "match_from_query": "c"}, "a", {"c": None})
    assert _m({"field": "f", "match_from_query": "c"}, "a", {"c": ["a", None]})
    for phrase in ("", "!!!", None):
        assert not _m({"field": "f", "match_text_from_query": "c"}, "a b", {"c": phrase})
        # ... and so a must_not on it excludes nothing
        assert naive.filter_matches({"must_not": [{"field": "f", "match_text_from_query": "c"}]},
                                    {"f": "a b"}, {"c": phrase}, {})
    assert not _m({"field": "f", "range_from_query": {"gt": "c"}}, 5, {"c": None})
    assert not _m({"field": "f", "range": {"gt": 0}}, None)
    assert naive.filter_matches({"must_not": [{"field": "f", "match": "a"}]}, {"f": None}, {}, {})


def test_empty_groups_constrain_nothing():
    for f in ({}, {"should": []}, {"must": [], "should": [], "must_not": []}, None):
        assert naive.filter_matches(f, {"f": 1}, {}, {})


def test_dates_are_integer_exact():
    """The hand-written integer parser agrees with datetime arithmetic (also
    integer) on random µs instants either side of 1970, in several offset
    spellings, and truncates past-µs digits as Qdrant does."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    rng = random.Random(7)
    for _ in range(3000):
        us = rng.randrange(-70 * 365 * 86400 * 10**6, 130 * 365 * 86400 * 10**6)
        off = rng.choice((0, 330, -480))
        dt = (epoch + timedelta(microseconds=us)).astimezone(timezone(timedelta(minutes=off)))
        lit = dt.isoformat()
        assert naive.rfc3339_to_epoch_us(lit) == us, lit
        assert naive.rfc3339_to_epoch_us(lit.replace("+00:00", "Z")) == us
    assert naive.rfc3339_to_epoch_us("2005-04-18T02:42:41.147961999Z") == 1113792161147961
    assert naive.rfc3339_to_epoch_us("2005-04-18") == 1113782400000000


def test_static_date_bounds_are_parsed_from_the_raw_string():
    """`gt` on exactly a stored instant excludes it, `gte` includes it — with
    the bound given as the RFC-3339 string the config holds."""
    at = "2005-04-18T02:42:41.147961Z"
    df = {"f": "rfc3339"}
    keep = lambda op: naive.filter_matches(  # noqa: E731
        {"must": [{"field": "f", "range": {op: at}}]}, {"f": at}, {}, df)
    assert not keep("gt") and keep("gte") and keep("lte") and not keep("lt")


def test_parse_filter_is_independent_of_nova_bf_models():
    """The oracle's own parse of a raw dict keeps the literal values (no float
    coercion, no date conversion) and is hashable for the cache."""
    nf = naive.parse_filter({"must": [{"field": "d", "range": {"gte": "2014-01-01T00:00:00Z"}},
                                      {"field": "x", "match": [1, True]}]})
    assert nf.must[0].arg == (("gte", "2014-01-01T00:00:00Z"),)
    assert nf.must[1].arg == (1, True) and type(nf.must[1].arg[1]) is bool
    hash(nf)


def test_matrix_filters_reach_the_oracles_raw(ds):
    """`_filter_from_dict` hands the oracles the config dict untouched — the
    date prepass converts IN PLACE, and a shared `cases.FILTERS` entry once
    came back with its RFC-3339 bounds already turned into nova-bf's floats."""
    from .cases import FILTERS
    from .runner import build_config, spec
    from .test_parity_matrix import _filter_from_dict

    build_config(ds, [spec("x", filter=FILTERS["datetime"])], out_tag="unused")
    assert FILTERS["datetime"]["must"][0]["range"]["gte"] == "2014-01-01T00:00:00Z"
    raw = _filter_from_dict(ds, FILTERS["datetime"])
    assert raw == FILTERS["datetime"] and raw is not FILTERS["datetime"]
