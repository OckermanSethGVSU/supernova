"""duckdb_filter parity: every mask must equal nova-bf's `filters.evaluate`
bit for bit (static and per-query, every leaf kind, nulls/NaN, dates, text).

nova-bf is the REFERENCE here, imported only by the tests (the harness itself
never imports it). Skipped when nova_bf is not importable.
"""
import copy
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench.data.filters import duckdb_filter as dfm

nova_bf = pytest.importorskip("nova_bf")
from nova_bf import filters as nbf  # noqa: E402
from nova_bf.compute import _to_query_array  # noqa: E402
from nova_bf.config import Filter, _normalize_static_date_bounds  # noqa: E402
from nova_bf.dates import (convert_table_date_columns, normalize_date_fields,  # noqa: E402
                           to_epoch_us_array)

CDATES = {"d": "rfc3339"}
QDATES = {"qd_lo": "rfc3339", "qd_hi": "rfc3339"}


# ---------------------------------------------------------------------------
# reference + comparison
# ---------------------------------------------------------------------------

def nova_mask(fdict, corpus, queries=None, cdates=CDATES, qdates=QDATES):
    data = {"corpus": {"date_fields": cdates}, "searches": [{"filter": copy.deepcopy(fdict)}]}
    _normalize_static_date_bounds(data)
    filt = Filter.model_validate(data["searches"][0]["filter"])
    # nova-bf's start-of-run check (compute.py -> check_match_types) on static
    # match values that don't fit the column. (Its match_from_query half also
    # rejects float/decimal/temporal columns, which the engine still evaluates
    # exactly; only its uint64 refusal is mirrored, see
    # test_uint64_match_from_query_is_rejected.)
    nbf.check_static_match_types([("s", filt)], corpus.schema,
                                 skip_fields=tuple(normalize_date_fields(cdates)))
    table = convert_table_date_columns(corpus, normalize_date_fields(cdates))
    qv = None
    if filt.is_per_query():
        qconv = convert_table_date_columns(queries, normalize_date_fields(qdates))
        qv = {c: _to_query_array(qconv[c].to_pylist()) for c in filt.query_fields()}
    m = nbf.evaluate(filt, table, qv)
    return (m.unpack() if isinstance(m, nbf.PackedRowMask) else np.asarray(m)), filt.is_per_query()


def duck_mask(fdict, corpus, queries=None, cdates=CDATES, qdates=QDATES, **kw):
    spec = dfm.parse_filter(fdict, cdates)
    kw.setdefault("threads", 2)
    eng = dfm.DuckFilter(spec, queries if spec.per_query else None,
                         corpus_date_fields=cdates, query_date_fields=qdates, **kw)
    res = eng.evaluate(corpus)
    return (res.query_masks() if res.per_query else res.row_mask()), res


def assert_parity(fdict, corpus, queries=None, **kw):
    ref, ref_pq = nova_mask(fdict, corpus, queries, **{k: v for k, v in kw.items() if k in ("cdates", "qdates")})
    got, res = duck_mask(fdict, corpus, queries, **kw)
    assert res.per_query == ref_pq, fdict
    # Padding bits past n_rows must be 0: a bitmap engine (cuVS) is handed the
    # packed bytes as-is.
    packed = res.static[None, :] if res.static is not None else res.combo_masks
    n = corpus.num_rows
    if n % 8 and packed.size:
        assert not (packed[:, -1] & (~dfm.tail_mask(n) & 0xFF)).any(), "padding bits set"
    if ref_pq and ref.ndim == 1:  # nova-bf may return a 1-D mask broadcast over queries
        ref = np.broadcast_to(ref, got.shape)
    assert got.shape == ref.shape, (got.shape, ref.shape, fdict)
    if not np.array_equal(got, ref):
        bad = np.argwhere(got != ref)[:5]
        pytest.fail(f"mask mismatch for {fdict}: {len(np.argwhere(got != ref))} cells, e.g. "
                    f"{[(tuple(b), bool(got[tuple(b)]), bool(ref[tuple(b)])) for b in bad]}")
    return res


# ---------------------------------------------------------------------------
# synthetic data with every edge the semantics care about
# ---------------------------------------------------------------------------

STRS = ["a", "b", "c", "Ünï", "x y", "O'Brien"]
WORDS = ["Alpha", "beta", "GAMMA", "delta", "épée", "naïve", "Σίσυφος", "ΣΑΣ", "x_y",
         "42", "foo-bar", "İstanbul", "ﬁne"]
SEPS = [" ", ", ", "-", "_", "!! ", "\n", "/", "."]
HOSTS = ["example", "news", "wiki", "blog", "shop"]


def _maybe(rng, v, p=0.1):
    return None if rng.random() < p else v


def make_corpus(n, seed=0):
    rng = random.Random(seed)

    def text():
        if rng.random() < 0.05:
            return ""
        k = rng.randint(1, 6)
        return "".join(rng.choice(WORDS) + rng.choice(SEPS) for _ in range(k))

    def date():
        y = rng.choice([2013, 2014, 2015])
        return f"{y}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00Z"

    return pa.table({
        "s": pa.array([_maybe(rng, rng.choice(STRS)) for _ in range(n)], pa.string()),
        "i": pa.array([_maybe(rng, rng.randint(-3, 5)) for _ in range(n)], pa.int64()),
        "f": pa.array([_maybe(rng, rng.choice([-1.5, 0.0, 0.5, 2.0, float("nan")])) for _ in range(n)],
                      pa.float64()),
        "b": pa.array([_maybe(rng, rng.random() < 0.5) for _ in range(n)], pa.bool_()),
        "d": pa.array([_maybe(rng, date()) for _ in range(n)], pa.string()),
        "text": pa.array([_maybe(rng, text()) for _ in range(n)], pa.large_string()),
        "url": pa.array([_maybe(rng, f"http://{rng.choice(HOSTS)}.{rng.choice(['com', 'org'])}/"
                                      f"{rng.choice(WORDS).lower()}") for _ in range(n)], pa.string()),
    })


def make_queries(nq, seed=1):
    rng = random.Random(seed)

    def phrase():
        r = rng.random()
        if r < 0.08:
            return None
        if r < 0.13:
            return rng.choice(["!!!", "", "   "])  # token-less: matches nothing
        k = rng.randint(1, 3)
        ws = [rng.choice(WORDS) for _ in range(k)]
        ws = [w.upper() if rng.random() < 0.3 else w for w in ws]
        return rng.choice(SEPS).join(ws)

    def date():
        return f"{rng.choice([2013, 2014, 2015])}-{rng.randint(1, 12):02d}-01T00:00:00Z"

    def flt():
        r = rng.random()
        return None if r < 0.08 else float("nan") if r < 0.14 else rng.choice([-2.0, -1.5, 0.0, 0.25, 0.5, 2.0])

    return pa.table({
        "qs": pa.array([_maybe(rng, rng.choice(STRS + ["zzz"])) for _ in range(nq)], pa.string()),
        "qi": pa.array([_maybe(rng, rng.randint(-3, 5)) for _ in range(nq)], pa.int64()),
        "qb": pa.array([_maybe(rng, rng.random() < 0.5) for _ in range(nq)], pa.bool_()),
        "qf_lo": pa.array([flt() for _ in range(nq)], pa.float64()),
        "qf_hi": pa.array([flt() for _ in range(nq)], pa.float64()),
        "qd_lo": pa.array([_maybe(rng, date()) for _ in range(nq)], pa.string()),
        "qd_hi": pa.array([_maybe(rng, date()) for _ in range(nq)], pa.string()),
        "qlist": pa.array([_maybe(rng, rng.sample(STRS + ["zzz"], rng.randint(0, 3))) for _ in range(nq)],
                          pa.list_(pa.string())),
        "qflist": pa.array([_maybe(rng, [rng.choice([-1.5, 0.5, 2.0, float("nan")])
                                         for _ in range(rng.randint(0, 3))]) for _ in range(nq)],
                           pa.list_(pa.float64())),
        "qilist": pa.array([_maybe(rng, rng.sample(range(-3, 6), rng.randint(0, 3))) for _ in range(nq)],
                           pa.list_(pa.int64())),
        "qphrase": pa.array([phrase() for _ in range(nq)], pa.string()),
        "qurl": pa.array([_maybe(rng, rng.choice(HOSTS) + rng.choice(["", " com", ".org"])) for _ in range(nq)],
                         pa.string()),
    })


STATIC_LEAVES = [
    {"field": "s", "match": "a"},
    {"field": "s", "match": ["a", "Ünï", "O'Brien"]},
    {"field": "i", "match": 3},
    {"field": "i", "match": [1, 2]},
    {"field": "f", "match": 0.5},
    {"field": "f", "match": float("nan")},
    {"field": "b", "match": True},
    {"field": "i", "range": {"gte": 0, "lt": 4}},
    {"field": "f", "range": {"gt": 0}},
    {"field": "f", "range": {"lte": 0.5}},
    {"field": "d", "range": {"gte": "2014-01-01T00:00:00Z"}},
    {"field": "d", "range": {"gte": "2013-06-01T00:00:00Z", "lt": "2015-02-01T00:00:00+02:00"}},
    {"field": "text", "match_text": "alpha beta"},
    {"field": "text", "match_text": "ΣΑΣ"},
    {"field": "url", "match_text": "example"},
]
PQ_LEAVES = [
    {"field": "s", "match_from_query": "qs"},
    {"field": "i", "match_from_query": "qi"},
    {"field": "b", "match_from_query": "qb"},
    {"field": "s", "match_from_query": "qlist"},
    {"field": "i", "match_from_query": "qilist"},
    {"field": "f", "match_from_query": "qflist"},   # NaN in a per-query list matches nothing
    {"field": "f", "match_from_query": "qf_lo"},    # NaN per-query scalar matches nothing
    {"field": "f", "range_from_query": {"gte": "qf_lo"}},
    {"field": "f", "range_from_query": {"gte": "qf_lo", "lt": "qf_hi"}},
    {"field": "i", "range_from_query": {"gt": "qf_lo"}},
    {"field": "d", "range_from_query": {"gte": "qd_lo", "lt": "qd_hi"}},
    {"field": "text", "match_text_from_query": "qphrase"},
    {"field": "url", "match_text_from_query": "qurl"},
]


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(301)


@pytest.fixture(scope="module")
def queries():
    return make_queries(60)


# ---------------------------------------------------------------------------
# leaf-by-leaf, in every group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("group", ["must", "should", "must_not"])
@pytest.mark.parametrize("leaf", STATIC_LEAVES + PQ_LEAVES, ids=lambda l: f"{l['field']}-{[k for k in l if k != 'field'][0]}")
def test_leaf(group, leaf, corpus, queries):
    assert_parity({group: [leaf]}, corpus, queries)


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 64, 65])
def test_row_count_edges(n, queries):
    c = make_corpus(n, seed=n + 5)
    assert_parity({"must": [STATIC_LEAVES[0], PQ_LEAVES[5]], "should": [PQ_LEAVES[9], PQ_LEAVES[3]]}, c, queries)
    assert_parity({"must": [STATIC_LEAVES[7]], "must_not": [STATIC_LEAVES[12]]}, c, queries)


def test_empty_filter(corpus):
    assert_parity({}, corpus)


@pytest.mark.parametrize("seed", range(150))
def test_random_filters(seed, corpus, queries):
    rng = random.Random(1000 + seed)
    pool = STATIC_LEAVES + PQ_LEAVES
    f = {}
    for g in ("must", "should", "must_not"):
        k = rng.choice([0, 0, 1, 2, 3])
        if k:
            f[g] = [rng.choice(pool) for _ in range(k)]
    assert_parity(f, corpus, queries)


def test_wide_projection_blocks(corpus):
    """More distinct per-query values than one projection block holds."""
    q = make_queries(400, seed=9)
    res = assert_parity({"must": [PQ_LEAVES[6], PQ_LEAVES[8]]}, corpus, q, block_cols=7)
    assert res.stats["sql_exprs"] > 7


def test_combo_dedup_matches_queries(corpus, queries):
    spec = dfm.parse_filter({"must": [PQ_LEAVES[0]]}, CDATES)
    eng = dfm.DuckFilter(spec, queries, corpus_date_fields=CDATES, query_date_fields=QDATES)
    res = eng.evaluate(corpus)
    distinct = len({v for v in queries["qs"].to_pylist()})
    assert len(res.combo_masks) == distinct
    assert res.packed_per_query().shape == (queries.num_rows, dfm.nbytes(corpus.num_rows))


# ---------------------------------------------------------------------------
# dates, literals, bitmaps, tokeniser
# ---------------------------------------------------------------------------

VALID_DATES = ["2013-05-18T05:48:54Z", "2013-05-18T05:48:54.123456Z", "2013-05-18T05:48:54+02:00",
               "2013-05-18T05:48:54-0530", "2013-05-18 05:48:54Z", "1969-12-31T23:59:59.5Z",
               "2100-01-01T00:00:00Z", None]


@pytest.mark.parametrize("fmt,vals,typ", [
    ("rfc3339", VALID_DATES, pa.string()),
    ("epoch_s", [0, 1, 1368856134, -5, None], pa.int64()),
    ("epoch_s", [1.0000005, 2.5e-6, 1368856134.25, -0.0000005, None], pa.float64()),
    ("epoch_ms", [0.0005, 0.0015, 1368856134123.5, None], pa.float64()),
    ("epoch_us", [0, 1368856134123456, None], pa.int64()),
    ("%Y%m%d", ["20130518", "19700101", None], pa.string()),
    ("%Y-%m-%d %H:%M", ["2013-05-18 05:48", None], pa.string()),
])
def test_date_conversion(fmt, vals, typ):
    import duckdb
    ref = to_epoch_us_array(pa.array(vals, typ), fmt).to_pylist()
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("t", pa.table({"v": pa.array(vals, typ)}))
    got = con.execute(f"SELECT {dfm._date_expr('v', fmt)} FROM t").fetchall()
    assert [g[0] for g in got] == ref


@pytest.mark.parametrize("bound", ["2014-01-01T00:00:00Z", "2014-01-01T00:00:00.000001Z",
                                   "2014-03-01T12:30:00.5+01:00", "2014-01-01T00:00:00"])
def test_static_date_bound(bound):
    from nova_bf.dates import parse_scalar_epoch_us
    assert dfm._static_bound(bound, "rfc3339") == float(parse_scalar_epoch_us(bound, "rfc3339"))


def test_sql_literal_roundtrip():
    import duckdb
    con = duckdb.connect()
    for v in [0.1, 0.9674920439720154, 1e-300, 5e-324, 1.7976931348623157e308, -0.0, 123456789.123456789]:
        assert con.execute(f"SELECT {dfm.sql_literal(v)}").fetchone()[0] == v
    assert con.execute(f"SELECT {dfm.sql_literal(chr(39) + 'x' + chr(39))}").fetchone()[0] == "'x'"
    assert con.execute(f"SELECT {dfm.sql_literal(2**62)}").fetchone()[0] == 2**62


@pytest.mark.parametrize("n", [0, 1, 7, 8, 13, 64, 1001])
def test_pack_bool(n):
    rng = np.random.default_rng(n)
    bits = rng.random(n + 11) < 0.5
    arr = pa.array(bits)
    for a in (arr.slice(0, n), arr.slice(3, n), arr.slice(8, n),
              pa.chunked_array([arr.slice(0, n // 2), arr.slice(n // 2, n - n // 2)])):
        ref = a.to_numpy(zero_copy_only=False) if isinstance(a, pa.Array) else \
            np.concatenate([c.to_numpy(zero_copy_only=False) for c in a.chunks])
        p = dfm.pack_bool(a, n)
        assert p.shape == (dfm.nbytes(n),)
        assert np.array_equal(dfm.unpack(p, n), ref)
        if n % 8:
            assert int(p[-1]) & (~dfm.tail_mask(n) & 0xFF) == 0  # padding bits zero
    assert np.array_equal(dfm.unpack(dfm.ones(n), n), np.ones(n, bool))


PREFILTER_PHRASES = ["alp", "ALPHA beta", "eta", "ΣΑΣ", "σας", "istanbul", "İSTANBUL", "ﬁne", "fine",
                     "x_y", "x y", "foo bar", "42 delta", "épée naïve", "a"]


@pytest.mark.parametrize("max_phrases", [0, 8, 10_000])
@pytest.mark.parametrize("phrase", PREFILTER_PHRASES)
def test_substring_prefilter_static(phrase, max_phrases, corpus):
    """The pre-check may only skip rows: on, off and forced, masks match nova-bf."""
    for fld in ("text", "url"):
        for g in ("must", "should", "must_not"):
            assert_parity({g: [{"field": fld, "match_text": phrase}]}, corpus,
                          prefilter_max_phrases=max_phrases)


@pytest.mark.parametrize("seed", range(20))
def test_substring_prefilter_forced_per_query(seed, corpus, queries):
    rng = random.Random(seed)
    leaves = [l for l in STATIC_LEAVES + PQ_LEAVES if "match_text" in l or "match_text_from_query" in l]
    f = {g: [rng.choice(leaves) for _ in range(rng.randint(1, 2))] for g in ("must", "should")}
    assert_parity(f, corpus, queries, prefilter_max_phrases=10_000)


def test_lower_selfcheck():
    import duckdb
    assert dfm.selfcheck_lower(duckdb.connect()) == []


@pytest.mark.slow
def test_tokenizer_selfcheck():
    import duckdb
    assert dfm.selfcheck_tokenizer(duckdb.connect()) == []


def test_tokenizer_matches_nova_on_words():
    import duckdb
    from nova_bf.tokenize import tokenize_many
    vals = [rng_s for rng_s in WORDS] + [s.join(WORDS) for s in SEPS] + [None, "", "!!!", "ﬁne ΣΑΣ σας"]
    assert dfm.tokenize_many(duckdb.connect(), vals) == tokenize_many(vals)


MULTILINGUAL = [
    "Hello, WORLD! hello_world it's", "हिन्दी भाषा", "مَرْحَبًا بِكُمْ", "สวัสดีครับ",
    "שָׁלוֹם עוֹלָם", "ⓐⓑⓒ ⓗⓔⓛⓛⓞ", "中文分词测试 日本語のテキスト 한국어", "he\u0301llo café",
    "ΟΔΟΣ σίσυφος", "İstanbul Straße ǅemal", "Ⅻ ½ x² ＡＢＣ ﬁ", "emoji \U0001f600 test",
    "\u1c89\u1c8a \U00010d50x \U00016ea0y", "\U0002ebf0\U0002ee5d", None, "", "!!! ---",
]


def test_tokenizer_matches_nova_on_multilingual_text():
    """Combining marks, circled letters, Unicode 16 capitals that DuckDB's
    lower() misses (`_lower_fix`), CJK Ext I: DuckDB's tokens == nova-bf's."""
    import duckdb
    from nova_bf.tokenize import tokenize_many
    vals = MULTILINGUAL + [" ".join(v for v in MULTILINGUAL if v)]
    assert dfm.tokenize_many(duckdb.connect(), vals) == tokenize_many(vals)


# Qdrant's per-token Rust lowercasing: `İ` -> i + U+0307, Final_Sigma.
LOWER_CASES = [
    "ΟΔΟΣ", "ΟΔΟΣΑ", "Σ", "ΑΣ", "ΣΑ", "ΑΣ'Β", "Α'Σ", "ΑΣ.Β", "Α\u0345Σ", "ΑΣ\u0345",
    "\u0345Σ", "Α\u02b0Σ", "\u02b0Σ", "ΑΣ\u02b0Β", "ΑΣ\u02b0", "1Σ", "Α1Σ", "ΑΣ1", "ΑΣ1Β",
    "ΑΣΣ", "ΑΣΣΑ", "ΑΣ\u0345Σ", "Α\u0345Σ\u0345Α", "ΑΣ1ΑΣ", "ΑΣΑΣ", "ΑΣ\u0301", "Α\u0301Σ",
    "ⒶΣ", "ǅΣ", "ΑσΒ", "Ας", "ς σ", "hello ΣΟΦΟΣ world", "İ", "İSTANBUL", "İstanbul",
    "iİ", "Aİ", "İΣ", "ΑΣİ", "i\u0307stanbul", "istanbul", "ıstanbul", "ΣΣΣ", "ΑΣΣΣ",
    "Ⴀ\u1c89Σ", "\U00010d50Σ \u1c89Σ",
]
_LOWER_ALPH = (["Σ", "Σ", "Σ", "σ", "ς", "İ", "İ", "i", "ı", "Α", "Β", "α", "a", "Z", "1",
                "Ⓐ", "ǅ", "中", " ", "'", ".", "\u0301", "\u0307", "-", "\u1c89"]
               + [chr(c) for c in (0x0345, 0x02B0, 0x1D2C, 0x05B0, 0x064E, 0x0E31)])


def test_tokenizer_matches_nova_on_lowercase_rules():
    import duckdb
    from nova_bf.tokenize import tokenize_many
    rng = random.Random(77)
    vals = LOWER_CASES + [" ".join(LOWER_CASES)] + [
        "".join(rng.choice(_LOWER_ALPH) for _ in range(rng.randint(0, 30))) for _ in range(3000)]
    got, want = dfm.tokenize_many(duckdb.connect(), vals), tokenize_many(vals)
    bad = [(v, g, w) for v, g, w in zip(vals, got, want) if g != w]
    assert not bad, f"{len(bad)} rows differ, e.g. {bad[:5]}"


def test_tokenizer_matches_nova_on_every_final_sigma_context():
    """Every Case_Ignorable / Cased word char around a Σ and after an İ."""
    import duckdb
    from nova_bf.tokenize import tokenize_many
    word = {c for lo, hi in dfm.WORD_CHAR_RANGES for c in range(lo, hi + 1)}
    ctx = sorted((dfm._CI | dfm._CASED) & word)
    vals = [v for c in ctx for v in ("Α" + chr(c) + "Σ", chr(c) + "Σ", "ΑΣ" + chr(c),
                                     "İ" + chr(c), chr(c) + "Σ" + chr(c) + "Σ")]
    assert dfm.tokenize_many(duckdb.connect(), vals) == tokenize_many(vals)


@pytest.mark.slow
@pytest.mark.parametrize("prefilter", [0, 10_000])
def test_text_masks_follow_the_lowercase_rules(prefilter):
    """Static and per-query match_text on İ / Σ words, with and without the
    substring pre-check (a word holding ς or U+0307 is checked folded)."""
    rows = LOWER_CASES + [None, "", "σοφος", "ΣΟΦΟΣ", "σοφοσ", "i̇stanbul"]
    corpus = pa.table({"t": pa.array(rows, pa.large_string())})
    phrases = ["οδος", "οδοσ", "ΟΔΟΣ", "ας", "ασ", "ασς", "ας1", "α\u02b0ς", "σ", "ς",
               "İstanbul", "istanbul", "iİ", "ii", "σοφος", "σοφοσ", "ⓐς", "ΑΣ'Β", "ii\u0307"]
    for ph in phrases:
        assert_parity({"must": [{"field": "t", "match_text": ph}]}, corpus,
                      prefilter_max_phrases=prefilter)
        assert_parity({"must_not": [{"field": "t", "match_text": ph}]}, corpus,
                      prefilter_max_phrases=prefilter)
    queries = pa.table({"ph": pa.array(phrases + [None], pa.large_string())})
    assert_parity({"must": [{"field": "t", "match_text_from_query": "ph"}]}, corpus, queries,
                  prefilter_max_phrases=prefilter)


def test_final_sigma_regex_matches_the_python_rule():
    """The SQL rewrite alone (no tokenizer around it) on random tokens."""
    import duckdb
    rng = random.Random(3)
    alph = ["Σ", "Σ", "Α", "1", "\u0345", "\u02b0", "σ", "ǅ", "中"]
    toks = sorted({"".join(rng.choice(alph) for _ in range(rng.randint(1, 8))) for _ in range(4000)})
    con = duckdb.connect()
    con.register("__t", pa.table({"i": pa.array(range(len(toks))), "x": pa.array(toks)}))
    got = con.sql(f"SELECT {dfm._special_token_sql('x')} AS l FROM __t ORDER BY i") \
        .to_arrow_table().column("l").to_pylist()
    import pyarrow.compute as pc
    simple = pc.utf8_lower(pa.array(toks)).to_pylist()
    want = [dfm.rust_lower_token(t, s) for t, s in zip(toks, simple)]
    assert got == want


def test_rust_case_props_copy_is_nova_bfs():
    import nova_bf.tokenizer_tables.rust_case_props as ref
    from bench.data.filters.text_tables import _rust_case_props as mine
    with open(ref.__file__, "rb") as a, open(mine.__file__, "rb") as b:
        assert a.read() == b.read()


def test_rust_lower_token_is_nova_bfs_rule():
    """The harness's own statement of the rule == nova-bf's, on random tokens."""
    import pyarrow.compute as pc
    from nova_bf.tokenize import qdrant_lower_token
    rng = random.Random(9)
    toks = ["".join(rng.choice(_LOWER_ALPH[:20] + _LOWER_ALPH[25:]) for _ in range(rng.randint(1, 9)))
            for _ in range(3000)]
    toks = [t for t in toks if " " not in t]
    simple = pc.utf8_lower(pa.array(toks)).to_pylist()
    assert [dfm.rust_lower_token(t, s) for t, s in zip(toks, simple)] == \
        [qdrant_lower_token(t, s) for t, s in zip(toks, simple)]


def test_word_char_table_copy_is_nova_bfs():
    """The harness vendors nova-bf's measured table; the copies must be identical."""
    import nova_bf.tokenizer_tables.qdrant_word_chars as ref
    from bench.data.filters.text_tables import _qdrant_word_chars as mine
    with open(ref.__file__, "rb") as a, open(mine.__file__, "rb") as b:
        assert a.read() == b.read()
    from nova_bf.tokenize import TOKEN_SPLIT_PATTERN
    assert dfm.TOKEN_SPLIT == TOKEN_SPLIT_PATTERN

def test_parse_errors():
    with pytest.raises(ValueError):
        dfm.parse_filter({"must": [{"field": "x", "match": 1, "range": {"gt": 0}}]})
    with pytest.raises(ValueError):
        dfm.parse_filter({"must": [{"field": "x", "range": {}}]})
    with pytest.raises(ValueError):
        dfm.parse_filter({"bogus": []})
    with pytest.raises(ValueError):
        dfm.parse_filter({"must": [{"field": "x", "range": {"gt": "2014-01-01"}}]})  # undeclared date


# ---------------------------------------------------------------------------
# real FineWeb file + the real query set (skipped when absent)
# ---------------------------------------------------------------------------

FINEWEB = os.path.join(os.path.dirname(__file__), "../../../../../fineweb_sample/train-part0__0200.parquet")
QUERIES = os.path.join(os.path.dirname(__file__), "../../../../data/ms_marco/ms_marco_shared_queries.parquet")
REAL_CDATES = {"date": "rfc3339"}
REAL_QDATES = {"date_gte": "rfc3339", "date_lt": "rfc3339"}
SLOTS = [f"domain_slot_{i}" for i in range(1, 10)]

FILTERED_TEXT = {"must": [{"field": "text", "match_text_from_query": "keyword_phrase"}],
                 "should": [{"field": "url", "match_text_from_query": s} for s in SLOTS]}
STRUCTURED = {"must": [{"field": "language_score", "range_from_query": {"gte": "ls_gte"}},
                       {"field": "date", "range_from_query": {"gte": "date_gte", "lt": "date_lt"}},
                       {"field": "dump", "match_from_query": "dump_set"}]}
REAL_STATIC = [
    {"must": [{"field": "language_score", "range": {"gte": 0.9}}]},
    {"must": [{"field": "dump", "match": ["CC-MAIN-2013-20", "CC-MAIN-2014-10"]}]},
    {"must": [{"field": "date", "range": {"gte": "2013-05-20T00:00:00Z", "lt": "2013-05-24T00:00:00Z"}}]},
    {"must": [{"field": "text", "match_text": "history of the"}]},
    {"must": [{"field": "language_score", "range": {"gte": 0.7}}],
     "should": [{"field": "url", "match_text": "wiki"}, {"field": "text", "match_text": "music"}],
     "must_not": [{"field": "url", "match_text": "blogspot"}]},
]

real = pytest.mark.skipif(not (os.path.exists(FINEWEB) and os.path.exists(QUERIES)),
                          reason="local FineWeb sample / query parquet not present")


@pytest.fixture(scope="module")
def real_corpus():
    return pq.read_table(FINEWEB, columns=["text", "url", "dump", "date", "language_score"])


def _real_queries(query_set):
    q = pq.read_table(QUERIES, columns=["query_set", "keyword_phrase", *SLOTS,
                                         "ls_gte", "date_gte", "date_lt", "dump_set"])
    import pyarrow.compute as pc
    return q.filter(pc.equal(q["query_set"], query_set))


@real
@pytest.mark.parametrize("fdict", REAL_STATIC, ids=range(len(REAL_STATIC)))
def test_real_static(fdict, real_corpus):
    res = assert_parity(fdict, real_corpus, cdates=REAL_CDATES, qdates=REAL_QDATES)
    kept = res.row_mask().sum()
    if fdict["must"][0]["field"] == "dump":
        # The corpus is dump-partitioned: one file = one dump, so a dump filter
        # is all-or-nothing per file (here: all).
        assert kept == real_corpus.num_rows
    else:
        assert 0 < kept < real_corpus.num_rows  # non-degenerate


@pytest.mark.slow
@real
@pytest.mark.parametrize("name,fdict,qset", [("filtered_text", FILTERED_TEXT, "filtered_text"),
                                             ("structured", STRUCTURED, "structured")])
def test_real_per_query(name, fdict, qset, real_corpus):
    q = _real_queries(qset)
    res = assert_parity(fdict, real_corpus, q, cdates=REAL_CDATES, qdates=REAL_QDATES)
    hits = res.query_masks().sum(axis=1)
    assert (hits > 0).any() and (hits < real_corpus.num_rows).any()


# ---------------------------------------------------------------------------
# cases that each reproduced a real disagreement
# ---------------------------------------------------------------------------

import datetime as _dt  # noqa: E402
import decimal as _dec  # noqa: E402

F32 = pa.array([0.1, 0.5, -0.0, 0.0, None, float("nan")], pa.float32())
F64 = pa.array([0.0, -0.0, None, float("nan"), 0.1, 1.5])
BOOL = pa.array([True, False, None])
I8 = pa.array([1, 0, None, 127], pa.int8())
U8 = pa.array([1, 255, None], pa.uint8())
I64 = pa.array([1, 0, 3, 2, None])
DEC = pa.array([_dec.Decimal("0.10"), _dec.Decimal("1.50"), None], pa.decimal128(5, 2))
STR = pa.array(["a", "3", "true", None, "a\x00b"])

STATIC_TYPE_CASES = [
    (F32, 0.1), (F32, 0.5), (F32, [0.1, 0.5]), (F32, 0.0), (F32, -0.0), (F32, float("nan")), (F32, 1e300),
    (F64, 0.0), (F64, -0.0), (F64, [0.0, float("nan")]), (F64, 0), (F64, True),
    (BOOL, 2), (BOOL, -1), (BOOL, 0.5), (BOOL, [0, 2]), (BOOL, 0), (BOOL, True),
    (I8, 257), (I8, [257, 1]), (I8, 127), (U8, -1), (U8, [255, -1]),
    (I64, 0.5), (I64, [0.5, 1]), (I64, 2.0), (I64, True), (I64, []),
    (DEC, 0.1), (DEC, 1.5), (DEC, [1.5, 0.1]),
    (STR, "a"), (STR, "a\x00b"), (STR, ["a\x00b", "3"]), (STR, []),
]


@pytest.mark.parametrize("group", ["must", "must_not"])
@pytest.mark.parametrize("col,val", STATIC_TYPE_CASES,
                         ids=[f"{c.type}-{v!r}" for c, v in STATIC_TYPE_CASES])
def test_static_match_types(col, val, group):
    t = pa.table({"x": col})
    try:
        nova_mask({group: [{"field": "x", "match": val}]}, t)
    except Exception:
        pytest.skip("reference raises on this input (engine may raise or not)")
    assert_parity({group: [{"field": "x", "match": val}]}, t)


PQ_TYPE_CASES = [
    (I64, pa.array(["3", "1", None, "0", "2"])),                 # str key, numeric column: never equal
    (I64, pa.array([["1"], ["3"], None, [], ["0", "2"]])),        # list form
    (BOOL, pa.array(["true", "f", None])),
    (BOOL, pa.array([2, 1, 0])),                                  # bool == 2 is False under numpy
    (BOOL, pa.array([0.5, 1.0, 0.0])),
    (I64, pa.array([True, False, None, True, False])),            # 1 == True under numpy
    (STR, pa.array([3, 1, None, 0, 2])),                          # int key, string column
    (STR, pa.array([True, False, None, True, False])),
    (STR, pa.array(["a\x00b", "a", None, "3", "zzz"])),
    (STR, pa.array([["a\x00b"], ["3", "a"], None, [], ["true"]])),
    (DEC, pa.array([0.1, 1.5, None])),                            # Decimal('0.10') != 0.1 exactly
    (DEC, pa.array([_dec.Decimal("0.1"), _dec.Decimal("1.50"), None], pa.decimal128(4, 2))),
    (F32, pa.array([0.1, float(np.float32(0.1)), -0.0, 0.0, None, float("nan")])),
    (F64, pa.array([[0.0], [-0.0], [float("nan")], None, [0.1, 1.5], []])),
]


@pytest.mark.parametrize("col,qcol", PQ_TYPE_CASES, ids=[f"{c.type}-{q.type}" for c, q in PQ_TYPE_CASES])
def test_per_query_match_types(col, qcol):
    corpus, queries = pa.table({"x": col}), pa.table({"q": qcol})
    assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, corpus, queries)
    assert_parity({"must_not": [{"field": "x", "match_from_query": "q"}]}, corpus, queries)


def test_per_query_match_temporal_and_binary_query_columns():
    ts = [_dt.datetime(2013, 5, 18, 5), _dt.datetime(2014, 1, 1)]
    for col, q in [
        (pa.array(ts + [None], pa.timestamp("us")), pa.array([ts[0], None, ts[1]], pa.timestamp("us"))),
        (pa.array([_dt.date(2013, 5, 18), None], pa.date32()), pa.array([_dt.date(2013, 5, 18), None], pa.date32())),
        (pa.array([b"ab", b"\x00", None], pa.binary()), pa.array([b"\x00", b"zz", None], pa.binary())),
    ]:
        assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, pa.table({"x": col}),
                      pa.table({"q": q}))


@pytest.mark.parametrize("name", ["__lf", "__r", "__f0", "__cv", 'we"ird'])
def test_internal_name_collisions(name):
    t = pa.table({name: pa.array(["hello world", "HELLO", None, "x"]), "y": pa.array([1, 2, 3, 4])})
    q = pa.table({"p": pa.array(["Hello", "world", None])})
    for f in ({"must": [{"field": name, "match_text": "hello"}]},
              {"must": [{"field": name, "match": "HELLO"}, {"field": "y", "range": {"gt": 1}}]},
              {"should": [{"field": name, "match_text_from_query": "p"}]}):
        assert_parity(f, t, q)


@pytest.mark.parametrize("cdates", [{"d": None}, {"d": ""}, ["d"], {"d": "rfc3339"}])
def test_date_fields_normalisation(cdates, corpus, queries):
    assert_parity({"must": [{"field": "d", "range": {"gte": "2014-01-01T00:00:00Z"}}],
                   "should": [{"field": "d", "range_from_query": {"gte": "qd_lo"}}]},
                  corpus, queries, cdates=cdates)


def test_rfc3339_declared_on_int64_column():
    t = pa.table({"d": pa.array([1368856134000000, 0, None, 1400000000000000])})
    q = pa.table({"lo": pa.array(["2013-01-01T00:00:00Z", None, "1970-01-01T00:00:00Z"])})
    assert_parity({"must": [{"field": "d", "range": {"gte": "2013-01-01T00:00:00Z"}}]}, t,
                  cdates={"d": "rfc3339"})
    assert_parity({"must": [{"field": "d", "range_from_query": {"gte": "lo"}}]}, t, q,
                  cdates={"d": "rfc3339"}, qdates={"lo": "rfc3339"})


def test_int64_above_2_53_with_nulls_is_exact():
    """nova-bf's per-query match converts an int column WITH NULLS to float64
    (`to_numpy`), so 2^53 and 2^53+1 compare equal there (and its CPU and GPU
    paths disagree on the scalar case). The engine is exact by design; this
    pins that choice rather than copying the precision loss."""
    big = 2 ** 53
    t = pa.table({"x": pa.array([big, big + 1, None])})
    q = pa.table({"q": pa.array([big, big + 1])})
    got, _ = duck_mask({"must": [{"field": "x", "match_from_query": "q"}]}, t, q)
    assert got.tolist() == [[True, False, False], [False, True, False]]


# ---------------------------------------------------------------------------
# fairness: parallel scan, factorisation, mask dedup
# ---------------------------------------------------------------------------

ENGINE_VARIANTS = [
    dict(threads=4, chunk_rows=16, factorize="never"),     # many chunks -> rows out of order
    dict(threads=4, chunk_rows=7, factorize="always"),
    dict(threads=1, factorize="always"),
    dict(threads=3, chunk_rows=64, factorize="auto", prefilter_max_phrases=0),
]


@pytest.mark.parametrize("variant", ENGINE_VARIANTS, ids=range(len(ENGINE_VARIANTS)))
@pytest.mark.parametrize("seed", range(40))
def test_random_filters_engine_variants(seed, variant, corpus, queries):
    rng = random.Random(5000 + seed)
    pool = STATIC_LEAVES + PQ_LEAVES
    f = {g: [rng.choice(pool) for _ in range(rng.choice([0, 1, 2, 3]))] for g in ("must", "should", "must_not")}
    assert_parity({g: v for g, v in f.items() if v}, corpus, queries, **variant)


@pytest.mark.parametrize("variant", ENGINE_VARIANTS, ids=range(len(ENGINE_VARIANTS)))
@pytest.mark.parametrize("col,qcol", PQ_TYPE_CASES, ids=[f"{c.type}-{q.type}" for c, q in PQ_TYPE_CASES])
def test_per_query_match_types_variants(col, qcol, variant):
    corpus, queries = pa.table({"x": col}), pa.table({"q": qcol})
    for g in ("must", "must_not"):
        assert_parity({g: [{"field": "x", "match_from_query": "q"}]}, corpus, queries, **variant)


def test_factorized_path_is_used(corpus):
    q = make_queries(300, seed=4)
    res = assert_parity({"must": [{"field": "s", "match_from_query": "qlist"},
                                  {"field": "i", "match_from_query": "qilist"}]},
                        corpus, q, factorize="auto")
    assert res.stats["factorized_exprs"] > 0


def test_multichunk_scan_out_of_order(corpus, queries):
    """A many-chunk table with threads > 1 makes DuckDB return rows out of
    order; masks must still be exact (scattered by row id)."""
    big = pa.concat_tables([corpus] * 20)
    assert_parity({"must": [PQ_LEAVES[6]], "should": [STATIC_LEAVES[1], PQ_LEAVES[0]]},
                  big, queries, threads=4, chunk_rows=32)


def test_distinct_masks(corpus, queries):
    _, res = duck_mask({"must": [PQ_LEAVES[0], PQ_LEAVES[5]]}, corpus, queries)
    masks, mid, counts = res.distinct_masks()
    full = res.packed_per_query()
    assert np.array_equal(masks[mid], full)
    assert len({m.tobytes() for m in masks}) == len(masks)
    assert counts.tolist() == [int(dfm.unpack(m, corpus.num_rows).sum()) for m in masks]


# ---------------------------------------------------------------------------
# integer ranges: exact float64 semantics via integer thresholds
# ---------------------------------------------------------------------------

BIG = 2 ** 53
INT_RANGE_COLS = [
    pa.array([BIG - 2, BIG - 1, BIG, BIG + 1, BIG + 2, BIG + 3, None, -BIG - 1, 0, 2 ** 63 - 1, -2 ** 63],
             pa.int64()),
    pa.array([0, 1, 127, -128, None, -1], pa.int8()),
    pa.array([0, 1, 2 ** 64 - 1, 2 ** 63, None, 2 ** 53 + 1], pa.uint64()),
    pa.array([0, 255, None, 7], pa.uint8()),
]
INT_BOUNDS = [float(BIG), float(BIG) + 2, float(BIG + 1), -float(BIG + 1), 0.5, -0.5, 0.0, -0.0, 127.0, 127.5,
              -128.5, 255.5, float(2 ** 63), float(2 ** 64), -float(2 ** 63), 1e300, -1e300,
              float("inf"), float("-inf"), 9007199254740993.0]


def parity_unless_ref_raises(fdict, corpus, queries=None, **kw):
    """assert_parity, except inputs the reference itself rejects (e.g. Arrow's
    safe int64 -> double cast above 2**53) are left free. Returns whether the
    case was compared."""
    try:
        nova_mask(fdict, corpus, queries)
    except Exception:
        return False
    assert_parity(fdict, corpus, queries, **kw)
    return True


@pytest.mark.parametrize("col", INT_RANGE_COLS, ids=lambda c: str(c.type))
@pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
def test_integer_range_thresholds(col, op):
    compared = 0
    # Without the >2**53 values too, so the static path is exercised where Arrow accepts it.
    for c in (col, col.filter(pa.array([v is None or abs(v) < BIG for v in col.to_pylist()]))):
        t = pa.table({"x": c})
        for b in INT_BOUNDS:
            compared += parity_unless_ref_raises({"must": [{"field": "x", "range": {op: b}}]}, t)
        q = pa.table({"q": pa.array(INT_BOUNDS + [None, float("nan")])})
        for g in ("must", "must_not"):
            compared += parity_unless_ref_raises({g: [{"field": "x", "range_from_query": {op: "q"}}]}, t, q)
    assert compared >= len(INT_BOUNDS)


def test_integer_range_two_sided(corpus):
    t = pa.table({"x": INT_RANGE_COLS[0]})
    q_lo = pa.array([random.Random(i).choice(INT_BOUNDS) for i in range(200)])
    q_hi = pa.array([random.Random(1000 + i).choice(INT_BOUNDS) for i in range(200)])
    q = pa.table({"lo": q_lo, "hi": q_hi})
    for lo_op in ("gt", "gte"):
        for hi_op in ("lt", "lte"):
            assert parity_unless_ref_raises(
                {"must": [{"field": "x", "range_from_query": {lo_op: "lo", hi_op: "hi"}}]}, t, q)


# ---------------------------------------------------------------------------
# last mismatches + loud refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col,val", [
    (pa.array([_dec.Decimal("1.00"), _dec.Decimal("3.00"), None], pa.decimal128(5, 2)), [1, 2, 3]),
    (pa.array([_dec.Decimal("1.00"), _dec.Decimal("2.50"), None], pa.decimal128(5, 2)), [1, 2]),  # ref raises
    (pa.array([_dec.Decimal("1"), _dec.Decimal("2")], pa.decimal128(5, 0)), [1.0, 2]),
], ids=["dec-ints", "dec-ints-frac", "dec-float"])
def test_static_match_decimal_exact(col, val):
    # Arrow raises when an int value set meets a decimal column holding
    # fractional values (it rescales the column); there the engine is free.
    for g in ("must", "must_not"):
        parity_unless_ref_raises({g: [{"field": "x", "match": val}]}, pa.table({"x": col}))


def test_per_query_decimal_38_digits():
    big = _dec.Decimal("9" * 38)
    t = pa.table({"x": pa.array([big, _dec.Decimal(1)], pa.decimal128(38, 0))})
    q = pa.table({"q": pa.array([big, _dec.Decimal(1)], pa.decimal128(38, 0))})
    assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, t, q)


@pytest.mark.parametrize("qcol", [pa.array([2 ** 62]), pa.array([[2 ** 62]]), pa.array([2 ** 62, None]),
                                  pa.array([1], pa.uint64()), pa.array([[1]], pa.list_(pa.uint64()))])
@pytest.mark.parametrize("ctype", [pa.uint64(), pa.int64(), pa.dictionary(pa.int32(), pa.uint64())])
def test_uint64_match_from_query_is_rejected(qcol, ctype):
    """nova-bf rejects match_from_query when either side is uint64 (numpy
    would compare it as float64 next to int64); so does the engine."""
    vals = pa.array([2 ** 62 + 1, 2 ** 62], pa.uint64())
    t = pa.table({"x": vals.cast(ctype) if not pa.types.is_dictionary(ctype)
                  else vals.dictionary_encode().cast(ctype)})
    f = {"must": [{"field": "x", "match_from_query": "q"}]}
    q = pa.table({"q": qcol})
    if not (_is_u64(ctype) or _is_u64(qcol.type)):
        assert_parity(f, t, q)
        return
    filt = Filter.model_validate(f)
    with pytest.raises(ValueError, match="uint64"):
        nbf.check_match_types([("s", filt)], t.schema, q.schema)
    with pytest.raises(ValueError, match="uint64"):
        duck_mask(f, t, q)


def _is_u64(t):
    if pa.types.is_dictionary(t):
        t = t.value_type
    if pa.types.is_list(t):
        t = t.value_type
    return pa.types.is_uint64(t)


@pytest.mark.parametrize("fdict,corpus_col,qcol", [
    ({"must": [{"field": "x", "match_text": "hello"}]}, pa.array([b"hello", None]), None),
    ({"must": [{"field": "x", "match": 5}]}, pa.array([5, 6], pa.duration("s")), None),
    ({"must": [{"field": "x", "match_from_query": "q"}]},
     pa.array([_dt.datetime(2013, 5, 18)], pa.timestamp("us", tz="UTC")),
     pa.array([_dt.datetime(2013, 5, 18, tzinfo=_dt.timezone.utc)], pa.timestamp("us", tz="UTC"))),
    ({"must": [{"field": "x", "match_from_query": "q"}]},
     pa.array([_dt.datetime(2013, 5, 18)], pa.timestamp("ns")), pa.array([_dt.datetime(2013, 5, 18)], pa.timestamp("ns"))),
], ids=["text-binary", "duration-match", "pq-tz-ts", "pq-ns-ts"])
def test_unsupported_types_refuse_loudly(fdict, corpus_col, qcol):
    """Inputs whose nova-bf results are numpy/Arrow accidents: the engine must
    raise NotImplementedError, never return a (possibly different) mask."""
    q = pa.table({"q": qcol}) if qcol is not None else None
    with pytest.raises(NotImplementedError):
        duck_mask(fdict, pa.table({"x": corpus_col}), q, cdates={}, qdates={})




# ---------------------------------------------------------------------------
# byte-permutation reorder of shuffled DuckDB result runs
# ---------------------------------------------------------------------------

def _runs_order(n, run_len, rng, misalign=False):
    starts = list(range(0, n, run_len))
    if misalign and len(starts) > 2:
        starts[1] += 3
    bounds = starts + [n]
    runs = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(starts))]
    rng.shuffle(runs)
    return np.concatenate(runs) if runs else np.zeros(0, np.int64)


@pytest.mark.parametrize("n", [0, 1, 8, 13, 2048, 5000, 20001])
@pytest.mark.parametrize("run_len", [8, 16, 2048])
@pytest.mark.parametrize("misalign", [False, True])
def test_byte_permutation_matches_scatter(n, run_len, misalign):
    rng = np.random.default_rng(n + run_len)
    r = _runs_order(n, run_len, random.Random(n), misalign)
    vals = rng.random(n) < 0.5                       # value of FILE row r[i] is vals[r[i]]
    res_bits = pa.array(vals[r]) if n else pa.array([], pa.bool_())
    perm = dfm._byte_permutation(r, n)
    if misalign and n > 2 * run_len:
        assert perm is None
    eng = dfm.DuckFilter(dfm.FilterSpec(), selfcheck=False)
    out = np.zeros((1, dfm.nbytes(n)), np.uint8)
    eng._scatter_pack(pa.table({"__r": pa.array(r, pa.int64()), "f": res_bits}), ["f"], n, out, [0])
    assert np.array_equal(dfm.unpack(out[0], n), vals)
    if n % 8:
        assert int(out[0, -1]) & (~dfm.tail_mask(n) & 0xFF) == 0


@pytest.mark.parametrize("n_distinct", [1, 3, 64, 65, 300])
def test_factorized_few_and_many_distinct(n_distinct):
    """Both broadcast branches of the factorised path (<= 64 distinct values:
    OR of per-value bitmaps; more: per-row gather)."""
    rng = random.Random(n_distinct)
    vals = [rng.randrange(n_distinct) if rng.random() > 0.1 else None for _ in range(2000)]
    t = pa.table({"x": pa.array(vals, pa.int64()), "s": pa.array([None if v is None else f"v{v}" for v in vals])})
    q = pa.table({"qi": pa.array([rng.sample(range(n_distinct + 2), min(3, n_distinct)) for _ in range(40)]),
                  "qs": pa.array([None if rng.random() < 0.1 else f"v{rng.randrange(n_distinct + 2)}" for _ in range(40)])})
    for fz in ("always", "never"):
        assert_parity({"must": [{"field": "x", "match_from_query": "qi"}],
                       "should": [{"field": "s", "match_from_query": "qs"},
                                  {"field": "x", "range": {"gte": 1}}]}, t, q, factorize=fz, threads=4)


# ---------------------------------------------------------------------------
# more disagreement cases
# ---------------------------------------------------------------------------

def _rechunk(t, rows):
    return pa.Table.from_batches(t.to_batches(max_chunksize=rows), schema=t.schema)


@pytest.mark.parametrize("seed", range(25))
def test_multichunk_queries_table(seed, corpus, queries):
    """a many-chunk queries table + threads > 1 must not shuffle per-query
    values (date conversion ran without ORDER BY under preserve_insertion_order=false)."""
    rng = random.Random(9000 + seed)
    pool = STATIC_LEAVES + PQ_LEAVES
    f = {g: [rng.choice(pool) for _ in range(rng.choice([1, 2]))] for g in ("must", "should")}
    f["must"].append({"field": "d", "range_from_query": {"gte": "qd_lo", "lt": "qd_hi"}})
    assert_parity(f, corpus, _rechunk(queries, rng.choice([1, 3, 7, 10])), threads=rng.choice([2, 4, 8]))


@real
def test_real_structured_multichunk_queries(real_corpus):
    q = _rechunk(_real_queries("structured"), 256)
    assert q["date_gte"].num_chunks > 1
    assert_parity(STRUCTURED, real_corpus, q, cdates=REAL_CDATES, qdates=REAL_QDATES, threads=8)


DEC_COLS = [pa.array([_dec.Decimal(x) for x in ("1.15", "0.30", "-8.77", "2.50")] + [None], pa.decimal128(10, 2)),
            pa.array([_dec.Decimal("9007199254740993.1"), _dec.Decimal("0.1"), None], pa.decimal128(18, 1)),
            pa.array([_dec.Decimal("-8.7686"), _dec.Decimal("0.3000")], pa.decimal128(5, 4))]


@pytest.mark.parametrize("col", DEC_COLS, ids=lambda c: str(c.type))
def test_decimal_ranges_static_and_per_query(col):
    """static ranges use Arrow's (inexact) decimal->double; per-query
    ranges use float(Decimal) (correctly rounded)."""
    t = pa.table({"x": col})
    bounds = sorted({float(v) for v in col.to_pylist() if v is not None} |
                    {float(np.nextafter(float(v), d)) for v in col.to_pylist() if v is not None for d in (-1e300, 1e300)}
                    | {9007199254740994.0, 0.3, 1.15})
    for op in ("gt", "gte", "lt", "lte"):
        for b in bounds:
            parity_unless_ref_raises({"must": [{"field": "x", "range": {op: b}}]}, t)
        q = pa.table({"q": pa.array(bounds)})
        parity_unless_ref_raises({"must": [{"field": "x", "range_from_query": {op: "q"}}]}, t, q)


@pytest.mark.slow
def test_decimal_ranges_fuzz():
    rng = random.Random(77)
    for _ in range(60):
        p = rng.randint(2, 38); sc = rng.randint(0, min(p - 1, 20))
        vals = [_dec.Decimal(rng.randrange(-10 ** (p - 1), 10 ** (p - 1))).scaleb(-sc) for _ in range(30)]
        t = pa.table({"x": pa.array(vals, pa.decimal128(p, sc))})
        bs = [float(v) for v in vals[:10]] + [float(np.nextafter(float(v), 1e300)) for v in vals[:5]]
        for op in ("gt", "lte"):
            parity_unless_ref_raises({"must": [{"field": "x", "range": {op: bs[0]}}]}, t)
            parity_unless_ref_raises({"must": [{"field": "x", "range_from_query": {op: "q"}}]}, t,
                                     pa.table({"q": pa.array(bs)}))


@pytest.mark.parametrize("fdict,col", [
    ({"should": [{"field": "x", "match": 0}, {"field": "x", "match": -0.0}]}, pa.array([0.0, -0.0, None])),
    ({"must": [{"field": "x", "match": 16777217.0}], "must_not": [{"field": "x", "match": 16777217}]},
     pa.array([16777216.0, 1.0], pa.float32())),
    ({"should": [{"field": "x", "match": [1, 0]}, {"field": "x", "match": [1.0, -0.0]}]}, pa.array([0.0, -0.0, 1.0])),
], ids=["zero-signs", "f32-int-vs-float", "lists"])
def test_static_leaves_not_merged_by_python_equality(fdict, col):
    """Cond(0) == Cond(-0.0) in Python, but their static-match SQL differs."""
    assert_parity(fdict, pa.table({"x": col}))


@pytest.mark.parametrize("col,vals", [
    (pa.array([0, 1, 127, None], pa.int8()), [128, -0.0]),
    (pa.array([0, 1, 5], pa.int64()), [0.5, -0.0]),
    (pa.array([0, 1, 255], pa.uint8()), [-1.0, -0.0, 1.0]),
    (pa.array([_dec.Decimal("0.00"), _dec.Decimal("1.15")], pa.decimal128(5, 2)), [-0.0, 1e300, 1.15]),
    (pa.array([True, False]), [-0.0, 2.0]),
    (pa.array([0, 1, 127], pa.int8()), [128, 1]),
], ids=["i8-128-negzero", "i64-half-negzero", "u8-neg", "dec-fallback", "bool", "i8-int-fallback"])
def test_static_match_cast_fallback(col, vals):
    """when part of the value set fails to cast, Arrow compares in a common type."""
    for g in ("must", "must_not"):
        parity_unless_ref_raises({g: [{"field": "x", "match": vals}]}, pa.table({"x": col}))






@pytest.mark.parametrize("fdict", [
    {"must": [{"field": "x", "range": {"gt": 0}}]},
    {"must_not": [{"field": "x", "range": {"gt": 0}}]},
    {"must": [{"field": "x", "match": 1}]},
    {"must": [{"field": "x", "range_from_query": {"gte": "qf"}}]},
    {"must_not": [{"field": "x", "match_from_query": "qi"}]},
    {"must": [{"field": "x", "match_text": "hello"}]},
])
def test_all_null_column(fdict):
    """F6c: an all-null (pa.null()) column matches nothing."""
    t = pa.table({"x": pa.nulls(5)})
    q = pa.table({"qf": pa.array([0.5, None]), "qi": pa.array([1, 2])})
    parity_unless_ref_raises(fdict, t, q)




# ---------------------------------------------------------------------------
# semantic (original) vs DuckDB-side types
# ---------------------------------------------------------------------------





def test_per_query_decimal_zero_with_small_exponent():
    """R4: Decimal('0E-10') equals 0 in a scale-9 column."""
    t = pa.table({"x": pa.array([_dec.Decimal(0), _dec.Decimal("1.5")], pa.decimal128(20, 9))})
    q = pa.table({"q": pa.array([_dec.Decimal("0E-10"), _dec.Decimal("1.5")], pa.decimal128(38, 10))})
    ql = pa.table({"q": pa.array([[_dec.Decimal("0E-10")], [_dec.Decimal("1.5")]], pa.list_(pa.decimal128(38, 10)))})
    assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, t, q)
    assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, t, ql)


def test_decimal_38_38():
    """R5: DECIMAL(38,38) literals."""
    v = _dec.Decimal("0." + "12345678901234567890123456789012345678")
    t = pa.table({"x": pa.array([v, _dec.Decimal(0), None], pa.decimal128(38, 38))})
    q = pa.table({"q": pa.array([v, _dec.Decimal(0)], pa.decimal128(38, 38))})
    assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, t, q)
    for m in (0.0, 0.5, [0.0, 0.5]):
        parity_unless_ref_raises({"must": [{"field": "x", "match": m}]}, t)


def test_decimal_epoch_date_field_refused():
    """R6: epoch_s/ms date fields stored as decimals are refused (nova-bf rescales
    through Arrow's inexact decimal -> float64)."""
    t = pa.table({"d": pa.array([_dec.Decimal("93.2775000")], pa.decimal128(9, 7))})
    with pytest.raises(NotImplementedError):
        duck_mask({"must": [{"field": "d", "range": {"gte": 0}}]}, t, cdates={"d": "epoch_ms"})




@pytest.mark.parametrize("vals", [0, 1, -1, 15843, [0, 2 ** 31 - 1]])
def test_static_match_date32_by_days(vals):
    t = pa.table({"x": pa.array([-1, 0, 15843, None], pa.date32())})
    parity_unless_ref_raises({"must": [{"field": "x", "match": vals}]}, t)




# ---------------------------------------------------------------------------
# type allow-list: exotic types are refused up front
# ---------------------------------------------------------------------------

EXOTIC_COLUMNS = [
    pa.array([1500, 1000], pa.timestamp("ns", tz="UTC")),
    pa.array([1500, 1000], pa.timestamp("ns")),
    pa.array([0, 1], pa.timestamp("s")),
    pa.array([0, 1], pa.timestamp("ms", tz="UTC")),
    pa.array([0, 86400000], pa.date64()),
    pa.array([_dec.Decimal("1.00"), _dec.Decimal("2.00")], pa.decimal256(5, 2)),
    pa.array([_dec.Decimal("1E+2"), _dec.Decimal("2E+2")], pa.decimal128(5, -2)),
    pa.array(np.array([0.1, 0.5], np.float16)),
    pa.array(["a", "bb"], pa.string_view()),
    pa.array([b"a", b"bb"], pa.binary_view()),
    pa.array(["a", "bb"], pa.string_view()).dictionary_encode(),
    pa.array([5, 6], pa.duration("s")),
]


@pytest.mark.parametrize("col", EXOTIC_COLUMNS, ids=lambda c: str(c.type))
@pytest.mark.parametrize("fdict", [
    {"must": [{"field": "x", "match": 1}]},
    {"must": [{"field": "x", "match_from_query": "q"}]},
    {"must": [{"field": "x", "range": {"gt": 0}}]},
], ids=["static-match", "pq-match", "range"])
def test_exotic_corpus_types_refused(col, fdict):
    t = pa.table({"x": col})
    q = pa.table({"q": pa.array([1, 2])})
    with pytest.raises(NotImplementedError, match="not supported by the DuckDB baseline"):
        duck_mask(fdict, t, q, cdates={}, qdates={})


@pytest.mark.parametrize("qcol,qdates", [
    (pa.array([3_600_000, 86_400_001], pa.date64()), {"lo": "rfc3339"}),
    (pa.array(np.array([1.0, 2.0], np.float16)), {"lo": "epoch_s"}),
    (pa.array([_dec.Decimal(1), _dec.Decimal(2)], pa.decimal256(20, 0)), {"lo": "epoch_s"}),
    (pa.array([_dec.Decimal(1), _dec.Decimal(2)], pa.decimal128(20, 0)), {"lo": "epoch_ms"}),
    (pa.array([1, 2], pa.timestamp("ns")), {}),
], ids=["date64-rfc3339", "f16-epoch", "dec256-epoch", "dec-epoch_ms", "ts-ns"])
def test_exotic_query_types_refused(qcol, qdates):
    t = pa.table({"x": pa.array([0, 5_000_000, 10 ** 12])})
    with pytest.raises(NotImplementedError, match="not supported by the DuckDB baseline"):
        duck_mask({"must": [{"field": "x", "range_from_query": {"gte": "lo"}}]}, t,
                  pa.table({"lo": qcol}), cdates={"x": "epoch_us"}, qdates=qdates)


# exact-range and type edge cases
def test_int_value_on_float_column_beyond_exact_range():
    for col, v in [(pa.array([16777218.0, 1.0, None], pa.float32()), 16777218),
                   (pa.array([float(2 ** 53 + 2), 1.0]), 2 ** 53 + 2),
                   (pa.array([float(2 ** 53), 1.0]), 2 ** 53 + 1)]:
        for g in ("must", "must_not"):
            parity_unless_ref_raises({g: [{"field": "x", "match": v}]}, pa.table({"x": col}))
            parity_unless_ref_raises({g: [{"field": "x", "match": [v, 1]}]}, pa.table({"x": col}))


def test_epoch_seconds_exact_halves():
    # DuckDB's round_even() is wrong on exact .5 near 1e15; the engine uses the
    # half-to-even DOUBLE -> BIGINT cast instead.
    vals = [1388534400.0000005, 1388534400.0000015, 1388534400.0000025, 1.5e-6, 2.5e-6]
    t = pa.table({"d": pa.array(vals)})
    q = pa.table({"q": pa.array([1388534400000000, 1388534400000001, 1388534400000002, 2, 3])})
    assert_parity({"must": [{"field": "d", "match_from_query": "q"}]}, t, q,
                  cdates={"d": "epoch_s"}, qdates={"q": "epoch_us"})


def test_per_query_decimal_value_too_large_for_column():
    t = pa.table({"x": pa.array([_dec.Decimal("0.5"), None], pa.decimal128(38, 38))})
    for qv in ([1.5, 0.5], [10 ** 9, 0.5]):
        assert_parity({"must": [{"field": "x", "match_from_query": "q"}]}, t, pa.table({"q": pa.array(qv)}))


def test_decimal_match_value_coerced_like_pydantic():
    spec = dfm.parse_filter({"must": [{"field": "x", "match": _dec.Decimal("0.1")}]})
    assert isinstance(spec.must[0].value, float)


def test_list_corpus_columns_refused():
    """A-1: list-typed CORPUS fields are refused (lists are for MatchAny query
    columns only); nova-bf's numpy compare of one-element list cells is accidental."""
    t = pa.table({"x": pa.array([[5], [6], None, [5]])})
    q = pa.table({"q": pa.array([5, 6, 7])})
    with pytest.raises(NotImplementedError, match="not supported by the DuckDB baseline"):
        duck_mask({"must": [{"field": "x", "match_from_query": "q"}]}, t, q, cdates={}, qdates={})


def test_dictionary_int_rfc3339_date_field():
    """A-2: a dictionary-encoded int column declared rfc3339 (epoch us)."""
    t = pa.table({"d": pa.array([1368856134000000, 0, None, 1368856134000000]).dictionary_encode()})
    q = pa.table({"lo": pa.array([1368856134000000, 1]).dictionary_encode()})
    assert_parity({"must": [{"field": "d", "range": {"gte": "2013-01-01T00:00:00Z"}}]}, t,
                  cdates={"d": "rfc3339"})
    assert_parity({"must": [{"field": "d", "range_from_query": {"gte": "lo"}}]}, t, q,
                  cdates={"d": "rfc3339"}, qdates={"lo": "rfc3339"})


# ---------------------------------------------------------------------------
# integer semantics of nova-bf 990d74e: differential over every int width
# ---------------------------------------------------------------------------

_INT_TYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64(),
              pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]


def _int_column(t, rng, n=64, dictionary=False):
    lo, hi = dfm._int_bounds(t)
    edge = [lo, lo + 1, hi - 1, hi, 0, 1, -1, 2 ** 53, 2 ** 53 + 1, 2 ** 63, -2 ** 53 - 1]
    vals = [v if lo <= v <= hi else 0 for v in edge]
    vals += [rng.randint(lo, hi) for _ in range(n - len(vals) - 4)] + [None] * 4
    rng.shuffle(vals)
    a = pa.array(vals, t)
    return a.dictionary_encode() if dictionary else a


def _int_bound(t, rng):
    lo, hi = dfm._int_bounds(t)
    c = rng.random()
    if c < 0.1:
        return rng.choice([float("inf"), float("-inf"), float("nan"), 1e300, -1e300])
    v = rng.choice([lo, hi, lo - 1, hi + 1, 0, 2 ** 53, 2 ** 63, rng.randint(lo, hi)])
    return float(v) + rng.choice([0.0, 0.0, 0.5, -0.5, 0.25])


def _int_value(t, rng):
    lo, hi = dfm._int_bounds(t)
    c = rng.random()
    if c < 0.1:
        return rng.choice([2.5, float("inf"), float("nan"), -0.0, 1e300])
    v = rng.choice([lo, hi, lo - 1, hi + 1, 0, rng.randint(lo, hi)])
    return float(v) if (c < 0.3 and abs(v) < 2 ** 53) else v


@pytest.mark.parametrize("dictionary", [False, True], ids=["plain", "dict"])
@pytest.mark.parametrize("t", _INT_TYPES, ids=str)
def test_random_static_int_filters_match_nova(t, dictionary):
    """Random static match/range trees on int8..uint64 (plain and dictionary
    encoded): the engine's mask == nova-bf's, exactly (no ref-raises escape:
    every one of these is a valid filter for nova-bf)."""
    rng = random.Random(_INT_TYPES.index(t) * 2 + dictionary)
    for _ in range(25):
        corpus = pa.table({"x": _int_column(t, rng, dictionary=dictionary)})
        leaves = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                ops = rng.sample(["gt", "gte", "lt", "lte"], rng.randint(1, 2))
                leaves.append({"field": "x", "range": {op: _int_bound(t, rng) for op in ops}})
            else:
                k = rng.randint(1, 4)
                v = [_int_value(t, rng) for _ in range(k)]
                leaves.append({"field": "x", "match": v if k > 1 else v[0]})
        g = rng.choice(["must", "should", "must_not"])
        assert_parity({g: leaves}, corpus)


@pytest.mark.parametrize("t", [pa.int8(), pa.int32(), pa.int64(), pa.uint8(), pa.uint32()], ids=str)
def test_random_per_query_int_filters_match_nova(t):
    """Per-query range (float64 semantics) and int64 match_from_query on the
    same columns, dictionary encoded too."""
    rng = random.Random(len(str(t)))
    for dictionary in (False, True):
        corpus = pa.table({"x": _int_column(t, rng, dictionary=dictionary)})
        q = pa.table({"lo": pa.array([_int_bound(t, rng) for _ in range(30)] + [None]),
                      "m": pa.array([rng.choice(corpus["x"].to_pylist()) for _ in range(30)] + [None],
                                    pa.int64())})
        assert_parity({"must": [{"field": "x", "range_from_query": {"gte": "lo"}}]}, corpus, q)
        assert_parity({"must_not": [{"field": "x", "range_from_query": {"lt": "lo"}}]}, corpus, q)
        assert_parity({"should": [{"field": "x", "match_from_query": "m"},
                                  {"field": "x", "range": {"gt": _int_bound(t, rng)}}]}, corpus, q)


@pytest.mark.parametrize("col,vals", [
    (pa.array([0, 1, 2], pa.int64()), [True]),
    (pa.array([0, 1, 2], pa.int8()), ["1"]),
    (pa.array([0.0, 1.0]), [False]),
    (pa.array(["a", "1"]), [1]),
    (pa.array([True, False]), [1]),
    (pa.array([1, 2], pa.int32()).dictionary_encode(), [True, 1]),
], ids=["int-bool", "int-str", "float-bool", "str-int", "bool-int", "dict-int-bool"])
def test_static_match_values_of_the_wrong_kind_are_rejected(col, vals):
    t = pa.table({"x": col})
    f = {"must": [{"field": "x", "match": vals}]}
    with pytest.raises(ValueError):
        nova_mask(f, t)
    with pytest.raises(ValueError, match="don't fit"):
        duck_mask(f, t)
