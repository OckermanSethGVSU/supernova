"""`dates.parse_scalar_epoch_us` is EXACT, and agrees with the column path.

A static `range` bound on a date field is converted by
`parse_scalar_epoch_us` (config load); the corpus column holding the same
instant is converted by `to_epoch_us_array` (Arrow). The two must produce the
same integer for the same literal, or a bound placed exactly on a stored
instant flips that row. They did not: the scalar path computed
`int(dt.timestamp() * 1_000_000)` — a float of seconds scaled to µs — which
truncates one µs low (or high) for ~1 % of sub-second literals, e.g.
2005-04-18T02:42:41.147961Z -> ...147960.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pytest

from nova_bf.dates import parse_scalar_epoch_us, to_epoch_us_array

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _literal(us: int, offset_min: int = 0) -> str:
    dt = (EPOCH + timedelta(microseconds=us)).astimezone(
        timezone(timedelta(minutes=offset_min)))
    s = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    if offset_min == 0:
        return s + "Z"
    sign = "+" if offset_min > 0 else "-"
    return s + f"{sign}{abs(offset_min) // 60:02d}:{abs(offset_min) % 60:02d}"


@pytest.mark.parametrize("literal,want", [
    ("2005-04-18T02:42:41.147961Z", 1113792161147961),   # float: ...960
    ("1961-06-16T09:02:42.182308Z", -269621837817692),   # float: one µs high
    ("1970-02-19T17:23:49.839604Z", 4296229839604),      # float: ...603
    ("1969-12-31T23:59:59.999999Z", -1),
    ("1970-01-01T00:00:00Z", 0),
    ("1900-01-01T00:00:00.000001+01:00", -2208992399999999),
    ("2005-04-18T08:12:41.147961+05:30", 1113792161147961),
])
def test_known_literals_are_exact(literal, want):
    assert parse_scalar_epoch_us(literal) == want


def test_random_us_literals_match_the_column_path_exactly():
    """5000 random µs instants 1900..2100 in UTC and offset spellings: the
    scalar (config) path, the column (Arrow) path and integer arithmetic all
    give the same µs. Under the old float formula ~1 % of these failed."""
    rng = random.Random(20261111)
    lo = (datetime(1900, 1, 1, tzinfo=timezone.utc) - EPOCH) // timedelta(microseconds=1)
    hi = (datetime(2100, 1, 1, tzinfo=timezone.utc) - EPOCH) // timedelta(microseconds=1)
    instants = [rng.randrange(lo, hi) for _ in range(5000)]
    offsets = [rng.choice((0, 0, 330, -480, 60)) for _ in instants]
    literals = [_literal(u, o) for u, o in zip(instants, offsets)]
    column = to_epoch_us_array(pa.array(literals)).to_pylist()
    bad = [(s, u, parse_scalar_epoch_us(s), c)
           for s, u, c in zip(literals, instants, column)
           if not (parse_scalar_epoch_us(s) == u == c)]
    assert not bad, f"{len(bad)} literals disagree, e.g. {bad[:3]}"


def test_strptime_format_is_exact_too():
    fmt = "%Y/%m/%d %H:%M:%S"
    assert parse_scalar_epoch_us("2005/04/18 02:42:41", fmt) == 1113792161000000
    assert to_epoch_us_array(pa.array(["2005/04/18 02:42:41"]), fmt)[0].as_py() \
        == 1113792161000000


@pytest.mark.xfail(strict=True, reason=(
    "DIVERGENCE (left for a decision): a strptime format with %f parses a "
    "static bound (Python strptime) but not the corpus column (Arrow's strptime "
    "has no %f), so such a config fails at the first file instead of at load"))
def test_strptime_fractional_seconds_work_on_both_paths():
    fmt = "%Y/%m/%d %H:%M:%S.%f"
    assert parse_scalar_epoch_us("2005/04/18 02:42:41.147961", fmt) == 1113792161147961
    assert to_epoch_us_array(pa.array(["2005/04/18 02:42:41.147961"]), fmt)[0].as_py() \
        == 1113792161147961


@pytest.mark.xfail(strict=True, reason=(
    "DIVERGENCE (left for a decision): the module docstring says naive strings "
    "(no Z/offset) are read as UTC. True for a static bound; the corpus/query "
    "COLUMN path (Arrow cast to timestamp[us, UTC]) rejects them with "
    "'expected a zone offset'"))
def test_naive_rfc3339_column_values_are_utc():
    assert to_epoch_us_array(pa.array(["2005-04-18T02:42:41.147961"]))[0].as_py() \
        == parse_scalar_epoch_us("2005-04-18T02:42:41.147961")
