"""A plain-Python reference implementation of everything nova-bf computes.

Deliberately slow and deliberately dumb: nested loops over Python floats, one
document at a time, no numpy vectorization, no batching, no shared helper
imported from `nova_bf`. That is the whole point — it can only agree with
nova-bf by both of them implementing the same *meaning*, never by both calling
the same code. Anything clever here (a vectorized rewrite, reusing
`nova_bf.filters.evaluate`, reusing `nova_bf.tokenize`) would quietly turn
this oracle into a mirror.

The one thing it does NOT try to reproduce is nova-bf's tie-break rule.
Reproducing it would mean agreeing on which scores are exactly equal, and two
independent float reductions do not agree on that — so `topk` ranks by
(score desc, corpus row asc) and the comparison helpers in `compare.py` treat
the top-K boundary with a tolerance instead. Ties as such are covered by the
dedicated tie-break suite (`tests/test_tiebreak*.py`,
`tests/test_qdrant_tiebreak_parity.py`), not here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------- documents


@dataclass
class Doc:
    """One corpus point, carrying every modality plus its payload.

    `row` is the point's global corpus row number in file-then-row order —
    nova-bf's own ordinal, and the id used throughout the harness so a hit can
    be traced back to the row that produced it.
    """

    row: int
    payload: dict = field(default_factory=dict)
    dense: list[float] | None = None
    # token id -> weight, ALREADY coalesced (a repeated raw index is one
    # dimension whose value is the sum of its occurrences — see
    # `sparse_from_pairs`).
    sparse: dict[int, float] | None = None
    # list of token vectors; [] and None both mean "has no multivector", which
    # makes the point a non-candidate for a multivector search.
    multivector: list[list[float]] | None = None


def sparse_from_pairs(indices, values) -> dict[int, float]:
    """`(indices, values)` as stored on disk -> `{token: weight}`.

    A token id repeated within one row is ONE dimension whose value is the sum
    of its occurrences (a hash collision is not two separate dimensions), so
    duplicates are summed rather than last-write-wins. nova-bf coalesces the
    same way (`compute._coalesce_by_row_col`); getting this wrong here would
    make the harness accept a corpus where they disagree.
    """
    out: dict[int, float] = {}
    for i, v in zip(indices, values):
        out[int(i)] = out.get(int(i), 0.0) + float(v)
    return out


# ------------------------------------------------------------------ scoring


def _l2(vec) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def dense_score(q: list[float], d: list[float], metric: str) -> float:
    """nova-bf's convention: LARGER is nearer, for every metric. Euclidean is
    therefore the NEGATED L2 distance (not squared, not the raw distance) —
    the form `compute._scores` returns and the form Qdrant's `EUCLID` score
    has to be negated into before comparison."""
    if metric == "dot":
        return sum(float(a) * float(b) for a, b in zip(q, d))
    if metric == "cosine":
        nq, nd = _l2(q), _l2(d)
        if nq == 0.0 or nd == 0.0:
            return 0.0
        return sum(float(a) * float(b) for a, b in zip(q, d)) / (nq * nd)
    if metric == "euclidean":
        return -math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(q, d)))
    raise ValueError(f"unknown dense metric {metric!r}")


def sparse_overlaps(q: dict[int, float], d: dict[int, float]) -> bool:
    """Do these two sparse vectors share at least one stored token?

    This — not the score — is what decides whether a sparse point is a
    CANDIDATE at all. A sparse search is an inverted-index intersection: a
    document that shares no token with the query is never visited, so it is
    absent from the results rather than present with a score of 0. Qdrant
    works that way and nova-bf reproduces it (`compute._zero_gate_file_ok`
    and the structural indicator path it falls back to).

    The distinction is invisible on strictly-positive data — where no overlap
    and a zero score coincide — and very visible on signed embeddings, where
    a real overlap can dot to a NEGATIVE score and must still outrank the
    non-candidates. Note it is STRUCTURAL: a token stored with the value 0.0
    is still an overlap.
    """
    return any(t in d for t in q)


def sparse_score(q: dict[int, float], d: dict[int, float], metric: str) -> float:
    """Dot over the shared support; cosine divides by both rows' TRUE norms.

    "True" matters: nova-bf truncates each corpus row to the union of the
    queries' token ids before scoring, which cannot change a dot product (the
    dropped dimensions are zero in every query anyway) but WOULD shrink the
    norm if the norm were taken after truncation, inflating cosine. So the
    norm here is over the full row, matching `compute._sparse_file_norms`.
    """
    dot = sum(w * d[t] for t, w in q.items() if t in d)
    if metric == "dot":
        return dot
    if metric == "cosine":
        nq = math.sqrt(sum(w * w for w in q.values()))
        nd = math.sqrt(sum(w * w for w in d.values()))
        if nq == 0.0 or nd == 0.0:
            return 0.0
        return dot / (nq * nd)
    raise ValueError(f"unknown sparse metric {metric!r}")


def maxsim(qtok: list[list[float]], dtok: list[list[float]], metric: str) -> float:
    """Late-interaction MaxSim: for each QUERY token take its best similarity
    against any document token, then sum those maxima over query tokens.

    `cosine` is MaxSim over per-token-normalized vectors (Qdrant's own
    definition of a cosine multivector), not a cosine of anything pooled.
    """
    if metric == "cosine":
        qtok = [_normalize(t) for t in qtok]
        dtok = [_normalize(t) for t in dtok]
    elif metric != "dot":
        raise ValueError(f"unknown multivector metric {metric!r}")
    total = 0.0
    for qt in qtok:
        best = None
        for dt in dtok:
            s = sum(float(a) * float(b) for a, b in zip(qt, dt))
            if best is None or s > best:
                best = s
        total += 0.0 if best is None else best
    return total


def _normalize(vec: list[float]) -> list[float]:
    n = _l2(vec)
    return list(vec) if n == 0.0 else [float(x) / n for x in vec]


# ------------------------------------------------------------------ filters
#
# The oracle reads the RAW filter dict — exactly what a YAML config carries
# (dates as RFC-3339 strings, numbers as written, `match: true` as a bool) —
# and never nova-bf's validated `Filter` model. Feeding it the model would let
# every coercion nova-bf's config layer performs (date bounds to epoch µs,
# range bounds to float, `match` values through pydantic's union) happen
# invisibly on BOTH sides of the comparison, which is how a ±1 µs date-bound
# bug once sat undetected under a fully green suite.
#
# Semantics below are Qdrant's, MEASURED against a live 1.18.2 server (see
# the README's "measured Qdrant semantics" table), not guessed:
#
#   * a null payload value never satisfies a condition (so `must_not` keeps it);
#   * `match` is TYPE-STRICT: `true` never equals `1`, `"1"` never equals `1`;
#   * `match: []` (MatchAny over nothing) matches nothing;
#   * a `match_text` phrase with no tokens (``""``, ``"!!!"``) matches nothing
#     (so as a `must_not` it excludes nothing, and as a `should` member it
#     adds nothing);
#   * a per-query value that is null makes that condition match nothing for
#     that query (a null element inside a MatchAny list is just absent);
#   * an empty `should` is no constraint; `{}` keeps every row;
#   * datetimes compare as integer epoch microseconds; digits past the sixth
#     fractional place are TRUNCATED.


_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")


def tokens(text) -> list[str]:
    """Qdrant `word`-tokenizer semantics: split the ORIGINAL text on maximal
    runs of non-alphanumerics, then lowercase each token.

    Written against ASCII on purpose. nova-bf splits on a measured Qdrant
    word-character table and lowercases per token like Rust `to_lowercase`
    (`İ` expansion, Final_Sigma); its non-ASCII behaviour is checked against
    the live engine directly by `test_parity_tokenizer.py`. The shared corpus
    is ASCII-only so this reference is exact rather than approximately exact.
    """
    if not isinstance(text, str):
        return []
    return [t.lower() for t in _TOKEN_SPLIT.split(text) if t]


# ---- datetimes, in integer arithmetic only (no float seconds anywhere)

_RFC3339 = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?"
    r"(Z|z|[+-]\d{2}:?\d{2})?$")


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Days since 1970-01-01 of a proleptic-Gregorian date (H. Hinnant's
    algorithm) — integer-only, so it cannot share a float rounding with
    `datetime.timestamp()` or with Arrow's parser."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def rfc3339_to_epoch_us(text: str) -> int:
    """An RFC-3339 / ISO-8601 literal as integer epoch µs, parsed by hand.
    A missing offset means UTC; fractional digits past µs are truncated
    (Qdrant's measured behaviour)."""
    m = _RFC3339.match(text.strip())
    if m is None:
        raise ValueError(f"not an RFC-3339 datetime: {text!r}")
    y, mo, d, hh, mm, ss, frac, tz = m.groups()
    days = _days_from_civil(int(y), int(mo), int(d))
    secs = days * 86400 + int(hh or 0) * 3600 + int(mm or 0) * 60 + int(ss or 0)
    us = int(((frac or "") + "000000")[:6])
    if tz and tz not in ("Z", "z"):
        sign = -1 if tz[0] == "-" else 1
        digits = tz[1:].replace(":", "")
        secs -= sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    return secs * 1_000_000 + us


def to_epoch_us(value, fmt: str = "rfc3339") -> int | None:
    """A declared date field's value as int64 epoch microseconds — Qdrant's own
    internal datetime representation, so a nova-bf `range` over a date field
    and a Qdrant `DatetimeRange` are value-for-value comparable."""
    if value is None:
        return None
    if fmt == "epoch_s":
        return int(round(float(value) * 1_000_000))
    if fmt == "epoch_ms":
        return int(round(float(value) * 1_000))
    if fmt == "epoch_us":
        return int(value)
    if fmt == "rfc3339":
        return rfc3339_to_epoch_us(value)
    dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
    return rfc3339_to_epoch_us(dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))


# ---- the oracle's own filter representation, parsed from the raw dict

_BOUND_OPS = ("gt", "gte", "lt", "lte")


@dataclass(frozen=True)
class Cond:
    """One condition, exactly as written: `kind` is the key that was set
    (`match`, `match_text`, `range`, or a `*_from_query` variant), `arg` its
    raw value (a list made a tuple, a range a tuple of `(op, bound)` pairs)."""

    field: str
    kind: str
    arg: object


@dataclass(frozen=True)
class NFilter:
    must: tuple = ()
    should: tuple = ()
    must_not: tuple = ()


_KINDS = ("match", "match_text", "range", "match_from_query",
          "match_text_from_query", "range_from_query")


def _freeze(v):
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple((op, v[op]) for op in _BOUND_OPS if v.get(op) is not None)
    return v


def parse_filter(fdict) -> NFilter | None:
    """The oracle's OWN parse of a raw filter dict. `None` stays `None`."""
    if fdict is None:
        return None
    if isinstance(fdict, NFilter):
        return fdict
    if not isinstance(fdict, dict):
        return _from_model(fdict)
    groups = {}
    for g in ("must", "should", "must_not"):
        conds = []
        for c in fdict.get(g) or ():
            set_kinds = [k for k in _KINDS if c.get(k) is not None]
            if len(set_kinds) != 1:
                raise ValueError(f"condition {c!r} must set exactly one predicate")
            kind = set_kinds[0]
            conds.append(Cond(c["field"], kind, _freeze(c[kind])))
        groups[g] = tuple(conds)
    return NFilter(**groups)


def _from_model(model) -> NFilter:
    """Back-compat for a caller still holding a nova-bf `Filter` model.
    Discouraged — the model's bounds are already nova-bf-converted, which is
    exactly what the raw-dict path exists to avoid — and marked as such so a
    date bound arriving as a number is taken as µs, not re-parsed."""
    groups = {}
    for g in ("must", "should", "must_not"):
        conds = []
        for c in getattr(model, g):
            for kind in _KINDS:
                v = getattr(c, kind)
                if v is None:
                    continue
                if kind in ("range", "range_from_query"):
                    v = tuple((op, getattr(v, op)) for op in _BOUND_OPS
                              if getattr(v, op) is not None)
                conds.append(Cond(c.field, kind, _freeze(v)))
        groups[g] = tuple(conds)
    return NFilter(**groups)


def _same(a, b) -> bool:
    """Qdrant's type-strict equality: a bool only equals a bool, a string
    only a string, and numbers compare by value (`3 == 3.0`, the one place
    nova-bf's static check deliberately lets an int column take a float)."""
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is bool and type(b) is bool and a == b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    return a == b


def _matches_any(value, wanted) -> bool:
    if isinstance(wanted, (list, tuple)):
        return any(_same(value, w) for w in wanted)
    return _same(value, wanted)


def _text_holds(value, phrase) -> bool:
    if value is None or not isinstance(phrase, str):
        return False
    want = tokens(phrase)
    if not want:
        return False          # measured: a token-less phrase matches nothing
    have = set(tokens(value))
    return all(t in have for t in want)


def _bound_us(bound, fmt: str):
    """A static date bound: a string is parsed; a number is already µs for
    rfc3339/strptime/epoch_us fields and in the field's own unit for
    epoch_s/epoch_ms (nova-bf's documented rule for numeric date bounds)."""
    if isinstance(bound, str):
        return to_epoch_us(bound, fmt)
    if fmt in ("epoch_s", "epoch_ms"):
        return to_epoch_us(bound, fmt)
    return bound


def condition_matches(cond, payload: dict, query_row: dict,
                      date_fields: dict[str, str],
                      query_date_fields: dict[str, str] | None = None) -> bool:
    """Does ONE condition hold for one point (for one query)?

    `date_fields` declares which CORPUS columns are datetimes, `query_date_fields`
    which QUERIES columns are. A null payload value never matches anything.
    """
    query_date_fields = query_date_fields or {}
    value = payload.get(cond.field)
    kind, arg = cond.kind, cond.arg
    if kind == "match_from_query":
        return value is not None and _matches_any(value, query_row[arg])
    if kind == "match":
        return value is not None and _matches_any(value, arg)
    if kind == "match_text_from_query":
        return _text_holds(value, query_row[arg])
    if kind == "match_text":
        return _text_holds(value, arg)
    if kind in ("range", "range_from_query"):
        if value is None:
            return False
        fmt = date_fields.get(cond.field)
        num = value if fmt is None else to_epoch_us(value, fmt)
        for op, b in arg:
            if kind == "range_from_query":
                col = b
                b = query_row[col]
                if b is not None and col in query_date_fields:
                    b = to_epoch_us(b, query_date_fields[col])
            elif fmt is not None:
                b = _bound_us(b, fmt)
            if b is None or not _cmp(num, op, b):
                return False
        return True
    raise ValueError(f"unknown condition kind {kind!r}")


def _cmp(value, op: str, bound) -> bool:
    if op == "gt":
        return value > bound
    if op == "gte":
        return value >= bound
    if op == "lt":
        return value < bound
    return value <= bound


def filter_matches(filt, payload: dict, query_row: dict, date_fields: dict[str, str],
                   query_date_fields: dict[str, str] | None = None) -> bool:
    """`must` = AND, `should` = at-least-one, `must_not` = AND-NOT — Qdrant's
    own composition. An EMPTY `should` is no constraint at all (not an OR over
    nothing, which would reject everything); an empty filter keeps every row.

    `filt` is a raw filter dict (preferred), an `NFilter`, or None.
    """
    filt = parse_filter(filt)
    if filt is None:
        return True
    for cond in filt.must:
        if not condition_matches(cond, payload, query_row, date_fields, query_date_fields):
            return False
    for cond in filt.must_not:
        if condition_matches(cond, payload, query_row, date_fields, query_date_fields):
            return False
    if filt.should:
        if not any(
            condition_matches(c, payload, query_row, date_fields, query_date_fields)
            for c in filt.should
        ):
            return False
    return True


# -------------------------------------------------------------------- top-K


def score_one(query, doc: Doc, vector_type: str, metric: str) -> float | None:
    """This query's score against this point, or None when the point is not a
    candidate for this search at all.

    Non-candidacy is not the same as scoring zero, and the difference shows up
    the moment real scores can be negative:

      * dense — every point is a candidate;
      * multivector — a point with a null or empty token set has nothing to
        MaxSim against, so it is absent rather than scoring 0;
      * sparse — a point sharing no token with the query is never visited by
        the inverted index, so it is absent too (`sparse_overlaps`).
    """
    if vector_type == "dense":
        return None if doc.dense is None else dense_score(query, doc.dense, metric)
    if vector_type == "sparse":
        if doc.sparse is None or not sparse_overlaps(query, doc.sparse):
            return None
        return sparse_score(query, doc.sparse, metric)
    if vector_type == "multivector":
        if not doc.multivector:
            return None
        return maxsim(query, doc.multivector, metric)
    raise ValueError(f"unknown vector_type {vector_type!r}")


def topk(
    query,
    docs: list[Doc],
    *,
    vector_type: str,
    metric: str,
    k: int,
    filt=None,
    query_row: dict | None = None,
    date_fields: dict[str, str] | None = None,
    query_date_fields: dict[str, str] | None = None,
) -> list[tuple[int, float]]:
    """`[(row, score), …]` — the eligible points ranked by score descending,
    truncated to `k`. Ordinal (corpus row) breaks ties, which is nova-bf's
    `tiebreak: ordinal` rule; see this module's docstring for why that
    agreement is not what the parity tests lean on."""
    query_row = query_row or {}
    date_fields = date_fields or {}
    filt = parse_filter(filt)
    scored = []
    for doc in docs:
        if not filter_matches(filt, doc.payload, query_row, date_fields,
                              query_date_fields):
            continue
        s = score_one(query, doc, vector_type, metric)
        if s is None:
            continue
        scored.append((doc.row, s))
    scored.sort(key=lambda rs: (-rs[1], rs[0]))
    return scored[:k]


# ------------------------------------------------------------------- oracle


class Oracle:
    """`topk` for many (metric, filter) combinations over one corpus, without
    recomputing what does not change.

    A corpus-side filter is a predicate on the POINT, evaluated before scoring
    and independent of the score (that is what makes it a pre-filter rather
    than a re-rank). So the score of a (query, point) pair is the same in every
    case that shares a vector_type and metric, and the eligibility of a point
    is the same in every case that shares a filter. This caches those two
    axes separately and combines them per case.

    It calls exactly the same `score_one` and `filter_matches` as the direct
    path — this is memoization, not a second implementation, and
    `test_oracle_matches_the_direct_path` pins that by checking a sample of
    cases against `topk` itself.
    """

    def __init__(self, docs: list[Doc], queries: list[dict], date_fields: dict[str, str],
                 query_date_fields: dict[str, str] | None = None):
        self.docs = docs
        self.queries = queries
        self.date_fields = date_fields
        self.query_date_fields = query_date_fields or {}
        self._scores: dict[tuple, list[list[float | None]]] = {}
        self._masks: dict[tuple, list[list[bool]]] = {}

    def _score_matrix(self, vector_type: str, metric: str, query_key: str):
        key = (vector_type, metric)
        if key not in self._scores:
            self._scores[key] = [
                [score_one(q[query_key], d, vector_type, metric) for d in self.docs]
                for q in self.queries
            ]
        return self._scores[key]

    def _mask(self, filt):
        # The oracle's own parsed form is frozen and hashable, so it is its
        # own cache key (a raw dict is parsed first; it is not hashable).
        key = filt = parse_filter(filt)
        if key not in self._masks:
            self._masks[key] = [
                [filter_matches(filt, d.payload, q["payload"], self.date_fields,
                                self.query_date_fields)
                 for d in self.docs]
                for q in self.queries
            ]
        return self._masks[key]

    def topk(self, *, vector_type: str, metric: str, k: int, filt=None,
             queries=None, query_key: str | None = None):
        """`{query_index: [(row, score), …]}` for every query (or just
        `queries`)."""
        query_key = query_key or vector_type
        scores = self._score_matrix(vector_type, metric, query_key)
        mask = self._mask(filt)
        out = {}
        for qi in (range(len(self.queries)) if queries is None else queries):
            ranked = [
                (doc.row, scores[qi][j])
                for j, doc in enumerate(self.docs)
                if mask[qi][j] and scores[qi][j] is not None
            ]
            ranked.sort(key=lambda rs: (-rs[1], rs[0]))
            out[qi] = ranked[:k]
        return out
