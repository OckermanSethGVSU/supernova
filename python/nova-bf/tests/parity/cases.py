"""The matrix under test: every filter shape × every modality/metric.

`FILTERS` enumerates the filter language — each of the six condition kinds,
each of the three groups, both the static and the per-query form of the four
that have one, plus the shapes where they compose. `MODALITIES` enumerates the
seven scored configurations (dense × 3 metrics, sparse × 2, multivector × 2).
`CASES` is the cross-product: 7 × the filter table, every cell checked against
BOTH oracles.

The cross-product is the point. A filter is evaluated once per corpus file and
shared by every search that names it, and all searches of one vector_type
share a single batch grid — so "does `match_text` work" and "does `match_text`
work for the multivector search sharing a pass with two sparse ones" are
different questions, and only the second is the one production asks.

Filters are written as plain dicts, exactly as they appear in a YAML config
(dates as RFC-3339 strings, not epoch microseconds) — `runner.build_config`
puts them through the same normalization `load_config` does.
"""

from __future__ import annotations

# name -> filter dict (or None). Names are also part of the nova-bf output
# filename, so they stay [A-Za-z0-9_-].
FILTERS: dict[str, dict | None] = {
    # --- no filter: the control. Every parity failure elsewhere has to be
    # attributable to the filter, which requires knowing the unfiltered search
    # is right.
    "nofilter": None,

    # --- static, one condition per kind
    "match": {"must": [{"field": "language", "match": "eng"}]},
    "matchany": {"must": [{"field": "category", "match": ["news", "paper"]}]},
    "rangeint": {"must": [{"field": "views", "range": {"gte": 2000, "lt": 8000}}]},
    "rangefloat": {"must": [{"field": "rating", "range": {"gt": 2.5}}]},
    "matchtext": {"must": [{"field": "title", "match_text": "vector search"}]},
    # A declared date field: the bound is an RFC-3339 string here and epoch µs
    # by the time nova-bf sees it, against Qdrant's DatetimeRange.
    "datetime": {"must": [{"field": "published_at",
                           "range": {"gte": "2014-01-01T00:00:00Z",
                                     "lt": "2015-01-01T00:00:00Z"}}]},
    # `tier` is null on a third of the corpus — this pins "a null payload
    # value never matches" on a field a filter actually reads.
    "nullable": {"must": [{"field": "tier", "match": "gold"}]},

    # --- the other two groups
    "mustnot": {"must_not": [{"field": "language", "match": "eng"}]},
    "should": {"should": [{"field": "category", "match": "news"},
                          {"field": "category", "match": "blog"}]},
    # must AND should AND must_not at once, over three different fields and
    # three different condition kinds.
    "compound": {
        "must": [{"field": "views", "range": {"gte": 1000}}],
        "should": [{"field": "language", "match": "eng"},
                   {"field": "language", "match": "fra"}],
        "must_not": [{"field": "tier", "match": "gold"}],
    },

    # --- per-query: the mask is (n_queries, rows), and every query wants a
    # different subset (see corpus._build_queries), so a per-query filter that
    # collapsed to one shared mask would fail rather than pass by accident.
    "pqmatch": {"must": [{"field": "language", "match_from_query": "q_language"}]},
    "pqmatchany": {"must": [{"field": "language",
                             "match_from_query": "q_languages"}]},
    "pqrange": {"must": [{"field": "views",
                          "range_from_query": {"gte": "q_min_views",
                                               "lt": "q_max_views"}}]},
    "pqtext": {"must": [{"field": "title", "match_text_from_query": "q_phrase"}]},
    "pqdatetime": {"must": [{"field": "published_at",
                             "range_from_query": {"gte": "q_after"}}]},
    # Static and per-query conditions in the same filter, across two groups —
    # the case where the mask starts 1-D and is promoted to 2-D mid-evaluation.
    "pqmixed": {
        "must": [{"field": "category", "match": ["news", "blog", "paper"]},
                 {"field": "language", "match_from_query": "q_language"}],
        "must_not": [{"field": "views", "range": {"lt": 500}}],
    },

    # --- per-query text mixed with per-query structured conditions (and static
    # text with per-query structured): today these take the CPU paths; the
    # unified GPU path (step 3) must reproduce them exactly.
    "pqtext_pqmatch_and": {"must": [
        {"field": "title", "match_text_from_query": "q_phrase"},
        {"field": "language", "match_from_query": "q_language"}]},
    "pqtext_should_pqrange": {"should": [
        {"field": "title", "match_text_from_query": "q_phrase"},
        {"field": "views", "range_from_query": {"gte": "q_min_views", "lt": "q_max_views"}},
        {"field": "category", "match": "news"}]},
    "pqtext_not_pqmatch": {
        "must": [{"field": "title", "match_text_from_query": "q_phrase"}],
        "must_not": [{"field": "language", "match_from_query": "q_language"}]},
    "pqmatch_not_pqtext": {
        "must": [{"field": "language", "match_from_query": "q_language"}],
        "must_not": [{"field": "title", "match_text_from_query": "q_phrase"}]},
    "pqrange_statictext": {"must": [
        {"field": "views", "range_from_query": {"gte": "q_min_views", "lt": "q_max_views"}},
        {"field": "title", "match_text": "search"}]},
    "pqtext_should_only_struct": {
        "must": [{"field": "title", "match_text_from_query": "q_phrase"}],
        "should": [{"field": "language", "match_from_query": "q_language"},
                   {"field": "language", "match_from_query": "q_languages"}]},
}

# (vector_type, metric) — every scored configuration nova-bf supports.
# Euclidean is dense-only by construction (`SearchSpec._no_euclidean_for_non_dense`).
MODALITIES = (
    ("dense", "dot"),
    ("dense", "cosine"),
    ("dense", "euclidean"),
    ("sparse", "dot"),
    ("sparse", "cosine"),
    ("multivector", "dot"),
    ("multivector", "cosine"),
)

K = 25  # < the smallest filtered result set, so every case is a real ranking


class Case:
    """One cell of the matrix: a nova-bf search, and the two oracle queries
    that must agree with it."""

    __slots__ = ("name", "vector_type", "metric", "filter_name", "filter_dict", "k")

    def __init__(self, vector_type, metric, filter_name, filter_dict, k=K):
        self.name = f"{vector_type[:2]}{metric[:3]}_{filter_name}"
        self.vector_type = vector_type
        self.metric = metric
        self.filter_name = filter_name
        self.filter_dict = filter_dict
        self.k = k

    def spec(self) -> dict:
        from .runner import spec

        return spec(self.name, vector_type=self.vector_type, metric=self.metric,
                    k=self.k, filter=self.filter_dict)

    @property
    def id(self) -> str:
        return f"{self.vector_type}-{self.metric}-{self.filter_name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Case {self.id}>"


CASES = [
    Case(vt, metric, fname, fdict)
    for vt, metric in MODALITIES
    for fname, fdict in FILTERS.items()
]

CASES_BY_NAME = {c.name: c for c in CASES}


# ------------------------------------------------------------------ EDGES
#
# The filter language at its edges, over the EDGE payload columns
# (`corpus.EDGE_SEED`): nulls on both sides, type-strict match on int / bool /
# dictionary columns, empty and token-less phrases, dead queries, fractional
# bounds on integer fields, every group shape the matrix does not reach.
#
# Kept OUT of `FILTERS` on purpose: many of these leave fewer than K rows (or
# none) for some query, which is the point of them and which the matrix's
# fixture guard (`test_the_filters_actually_discriminate`) rightly forbids
# there. They are run by `test_parity_regimes.py` at k = 1, 7 and 400 (one
# pass, so k varies inside one shared batch grid; 400 is past both a file and
# the whole 300-row corpus, which turns every comparison into an exact
# eligible-set comparison). Each entry's comment says which code path it is
# for; `test_parity_filter_paths.py` asserts those paths were really taken.
EDGE_FILTERS: dict[str, dict] = {
    # --- static, type-strict match
    "e_match_int": {"must": [{"field": "score_i", "match": 7}]},
    "e_matchany_int": {"must": [{"field": "score_i", "match": list(range(1, 13))}]},
    "e_match_bool": {"must": [{"field": "flag", "match": True}]},
    "e_mustnot_bool": {"must_not": [{"field": "flag", "match": True}]},
    "e_dict_match": {"must": [{"field": "lang_dict", "match": "eng"}]},
    "e_dict_matchany": {"must": [{"field": "lang_dict", "match": ["fra", "spa"]}]},
    "e_dict_mustnot": {"must_not": [{"field": "lang_dict", "match": "eng"}]},
    # --- ranges: fractional bounds on an int field; nullable float field
    "e_range_int_frac": {"must": [{"field": "score_i", "range": {"gt": 10.5, "lte": 25.5}}]},
    # integer bounds on an int field: here gt/gte (and lt/lte) differ
    "e_range_int_exact": {"must": [{"field": "score_i", "range": {"gt": 10, "lt": 20}}]},
    "e_range_int_exact2": {"must": [{"field": "score_i", "range": {"gte": 10, "lte": 20}}]},
    "e_range_weight": {"must": [{"field": "weight", "range": {"gte": 0.3, "lt": 0.8}}]},
    "e_mustnot_weight": {"must_not": [{"field": "weight", "range": {"lt": 0.5}}]},
    # --- static text on a nullable field with "" and punctuation-only rows
    "e_text_note": {"must": [{"field": "note", "match_text": "alpha"}]},
    "e_text_note_multi": {"must": [{"field": "note", "match_text": "Gamma, ALPHA!"}]},
    "e_mustnot_text_note": {"must_not": [{"field": "note", "match_text": "beta"}]},
    # --- group shapes
    "e_two_mustnot": {"must_not": [{"field": "language", "match": "eng"},
                                   {"field": "category", "match": "news"}]},
    "e_same_field_twice": {"must": [{"field": "score_i", "range": {"gte": 5}},
                                    {"field": "score_i", "range": {"lte": 20}},
                                    {"field": "views", "range": {"gte": 2000}},
                                    {"field": "views", "range": {"lt": 7000}}]},
    "e_single_should": {"should": [{"field": "category", "match": "news"}]},
    "e_should_mustnot": {"should": [{"field": "flag", "match": True},
                                    {"field": "score_i", "range": {"lt": 5}}],
                         "must_not": [{"field": "lang_dict", "match": "eng"}]},
    "e_dead_static": {"must": [{"field": "language", "match": "xxx"}]},
    "e_short_static": {"must": [{"field": "views", "range": {"gte": 9800}}]},

    # --- per-query STRUCTURED only (a struct FilterPlan; `_gpu_evaluate` when
    # the combine is off). Query columns carry nulls and dead values.
    "e_pq_int": {"must": [{"field": "score_i", "match_from_query": "q_score"}]},
    "e_pq_intany": {"must": [{"field": "score_i", "match_from_query": "q_scores"}]},
    "e_pq_bool": {"must": [{"field": "flag", "match_from_query": "q_flag"}]},
    "e_pq_lang_n": {"must": [{"field": "language", "match_from_query": "q_lang_n"}]},
    "e_pq_lang_n_not": {"must_not": [{"field": "language", "match_from_query": "q_lang_n"}]},
    "e_pq_langs_n": {"must": [{"field": "language", "match_from_query": "q_langs_n"}]},
    "e_pq_langs_n_not": {"must_not": [{"field": "language", "match_from_query": "q_langs_n"}]},
    "e_pq_nolist": {"must": [{"field": "language", "match_from_query": "q_nolist"}]},
    "e_pq_nolist_not": {"must_not": [{"field": "language", "match_from_query": "q_nolist"}],
                        "must": [{"field": "score_i", "match_from_query": "q_score"}]},
    "e_pq_dict": {"must": [{"field": "lang_dict", "match_from_query": "q_lang_n"}]},
    "e_pq_range_frac": {"must": [{"field": "score_i", "range_from_query": {"gt": "q_score_lo"}}]},
    "e_pq_range_frac_lte": {"must": [{"field": "score_i",
                                      "range_from_query": {"lte": "q_score_lo"}}]},
    "e_pq_weight": {"must": [{"field": "weight", "range_from_query": {"gte": "q_weight_lo"}}]},
    "e_pq_weight_not": {"must_not": [{"field": "weight",
                                      "range_from_query": {"gte": "q_weight_lo"}}]},
    # struct plan with a STATIC should row (`static_or`) next to a pq must
    "e_pq_static_should": {"must": [{"field": "score_i", "match_from_query": "q_scores"}],
                           "should": [{"field": "flag", "match": True},
                                      {"field": "category", "match": "news"}]},
    # a per-query leaf INSIDE should (struct_should), next to a static member
    "e_pq_in_should": {"should": [{"field": "language", "match_from_query": "q_lang_n"},
                                  {"field": "flag", "match": True}]},
    # per-query leaves in should AND must_not, static must
    "e_pq_should_not": {"must": [{"field": "views", "range": {"gte": 1000}}],
                        "should": [{"field": "score_i", "match_from_query": "q_scores"},
                                   {"field": "weight", "range_from_query": {"gte": "q_weight_lo"}}],
                        "must_not": [{"field": "flag", "match_from_query": "q_flag"}]},

    # selective struct plans with a per-query must_not / should leaf (they
    # keep the COMPACTED arm's row union small enough to leave gaps)
    "e_pq_must_not_pq": {"must": [{"field": "score_i", "match_from_query": "q_scores"}],
                         "must_not": [{"field": "flag", "match_from_query": "q_flag"}]},
    "e_pq_must_should_pq": {"must": [{"field": "score_i", "match_from_query": "q_score"}],
                            "should": [{"field": "language", "match_from_query": "q_lang_n"},
                                       {"field": "flag", "match": True}]},

    # --- per-query TEXT only (a combo TextPart)
    "e_pqtext_note": {"must": [{"field": "note", "match_text_from_query": "q_note_phrase"}]},
    "e_pqtext_note_not": {"must_not": [{"field": "note",
                                        "match_text_from_query": "q_note_phrase"}]},
    "e_pqtext_note_should": {"should": [{"field": "note", "match_text_from_query": "q_note_phrase"},
                                        {"field": "category", "match": "news"}]},
    "e_pqtext_should_only": {"should": [{"field": "note", "match_text_from_query": "q_note_phrase"},
                                        {"field": "title", "match_text_from_query": "q_phrase"}]},
    # two per-query phrases ANDed: every combination owns >= 2 parts, the
    # `generic` branch of the union factoring
    "e_pqtext_two_phrases": {"must": [{"field": "note", "match_text_from_query": "q_note_phrase"},
                                      {"field": "title", "match_text_from_query": "q_phrase"}]},

    # --- MIXED: per-query text AND per-query structured (a mixed FilterPlan)
    "e_mixed_and": {"must": [{"field": "note", "match_text_from_query": "q_note_phrase"},
                             {"field": "score_i", "match_from_query": "q_scores"}]},
    "e_mixed_should": {"must": [{"field": "flag", "match": True}],
                       "should": [{"field": "note", "match_text_from_query": "q_note_phrase"},
                                  {"field": "language", "match_from_query": "q_lang_n"}]},
    "e_mixed_not": {"must": [{"field": "lang_dict", "match_from_query": "q_lang_n"}],
                    "must_not": [{"field": "note", "match_text_from_query": "q_note_phrase"}]},
    "e_mixed_two_phrases": {"must": [{"field": "note", "match_text_from_query": "q_note_phrase"},
                                     {"field": "title", "match_text_from_query": "q_phrase"},
                                     {"field": "weight", "range_from_query": {"gte": "q_weight_lo"}}]},
}

# Filters with no condition at all. nova-bf treats them exactly like no
# filter (`compute._is_unfiltered`), so they force the whole-file batch grid —
# kept apart so the COMPACTED regime (no unfiltered search in a vector_type)
# can be run without them.
UNFILTERED_EDGE_FILTERS: dict[str, dict] = {
    "e_empty": {},
    "e_empty_should": {"should": []},
}

# The COMPACTED regime runs only SELECTIVE filters. The batch grid of a
# vector_type with no unfiltered search is the union of ALL its filters' rows,
# so one broad filter (`e_mustnot_bool` keeps half the corpus for everyone)
# would make the union the whole file again — and a whole-file union is read
# as a contiguous slice, which is the baseline path wearing another name.
# Every plan shape is still represented (static, struct with static_or /
# struct_should / struct_not, text, mixed).
COMPACT_EDGE = ("e_match_int", "e_short_static", "e_dead_static", "e_text_note_multi",
                "e_pq_int", "e_pq_intany", "e_pq_static_should", "e_pq_must_not_pq",
                "e_pq_must_should_pq", "e_pqtext_two_phrases", "e_mixed_and",
                "e_mixed_two_phrases")

EDGE_KS = (1, 7, 400)


# ------------------------------------------------------------------ DATES
#
# Bounds placed EXACTLY on stored µs-precision instants (`corpus.EVENT_AT_PINS`),
# for each of gt / gte / lt / lte, static and per-query. At such a bound an
# implementation one µs off flips exactly the rows holding that instant — and
# for EVENT_A/EVENT_C float-seconds arithmetic is one µs LOW (gt wrongly keeps
# them, lte wrongly drops them), for EVENT_B one µs HIGH (gte drops, lt keeps).
# Run at k=400 (> corpus), so each comparison is of the whole eligible set.
def _date_filters() -> dict[str, dict]:
    from .corpus import (EVENT_A, EVENT_A_OFFSET, EVENT_B, EVENT_C, EVENT_M1,
                         EVENT_ZERO)

    out: dict[str, dict] = {}
    pins = {"a": EVENT_A, "b": EVENT_B, "c": EVENT_C, "m1": EVENT_M1, "zero": EVENT_ZERO}
    for tag, instant in pins.items():
        for op in ("gt", "gte", "lt", "lte"):
            out[f"d_{op}_{tag}"] = {"must": [{"field": "event_at", "range": {op: instant}}]}
    # equality windows: exactly the rows holding that instant (A is stored
    # twice as "...Z" and once as the same instant written "+05:30")
    out["d_eq_a"] = {"must": [{"field": "event_at",
                               "range": {"gte": EVENT_A, "lte": EVENT_A}}]}
    out["d_eq_b"] = {"must": [{"field": "event_at",
                               "range": {"gte": EVENT_B, "lte": EVENT_B}}]}
    # the same bound in three spellings: offset, no zone (UTC), date-only
    out["d_eq_a_offset"] = {"must": [{"field": "event_at",
                                      "range": {"gte": EVENT_A_OFFSET, "lte": EVENT_A_OFFSET}}]}
    naive_a = EVENT_A.rstrip("Z")
    out["d_eq_a_nozone"] = {"must": [{"field": "event_at",
                                      "range": {"gte": naive_a, "lte": naive_a}}]}
    out["d_day_a"] = {"must": [{"field": "event_at",
                                "range": {"gte": "2005-04-18", "lt": "2005-04-19"}}]}
    # across the epoch, both bounds exclusive
    out["d_window_epoch"] = {"must": [{"field": "event_at",
                                       "range": {"gt": EVENT_M1, "lt": EVENT_C}}]}
    out["d_mustnot_lte_b"] = {"must_not": [{"field": "event_at", "range": {"lte": EVENT_B}}]}
    # per-query: each query's bound is one of the pinned instants (or null)
    for op in ("gt", "gte", "lt", "lte"):
        out[f"dq_{op}"] = {"must": [{"field": "event_at",
                                     "range_from_query": {op: "q_event_at"}}]}
    out["dq_eq"] = {"must": [{"field": "event_at",
                              "range_from_query": {"gte": "q_event_at", "lte": "q_event_at"}}]}
    out["dq_not_gt"] = {"must_not": [{"field": "event_at",
                                      "range_from_query": {"gt": "q_event_at"}}]}
    return out


DATE_FILTERS: dict[str, dict] = _date_filters()
DATE_K = 400
