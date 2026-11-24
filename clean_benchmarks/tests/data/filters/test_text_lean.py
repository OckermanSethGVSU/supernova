"""Lean tokenisation (`DuckFilter(text_guards="auto")`, `lean_tokens_ok`): with
an all-ASCII query vocabulary the engine lowercases with DuckDB's plain lower()
and guards only `İ`. It must give the masks of the full (nova-bf-exact) SQL."""

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from bench.data.filters import duckdb_filter as dfm
from ref_text_numpy import IMPLS, duck_filter

FIX = dfm._lower_fix()[0]
CAP = FIX[0] if FIX else "Σ"      # a capital DuckDB's lower() leaves alone
DOCS = ["İstanbul trip", "istanbul trip", "ISTANBUL", "i̇stanbul", "İ", "i", "İn in IN",
        "ΟΔΟΣ odos", "οδος", "οδοσ", "Σ", "aΣ b", f"{CAP}x x", f"x{CAP}", f"a {CAP} b",
        "KELVIN K k", "ſ s", "ǅ", "plain ascii words", "", None, "naïve naive", "日本 x1",
        "BEĞENDİM begendim", "İİ ii", "aİb aib", "σ ς Σ.", "ẋ x"]
ASCII_Q = ["istanbul", "i", "in", "odos", "x", "b", "a b", "k", "s", "naive", "x1", "ii", "aib",
           "begendim", "trip istanbul", "absent", "plain words"]
NON_ASCII_Q = ["οδος", "i̇stanbul", "naïve", "日本"]

ASCII_ONLY = "x -> strlen(x) = length(x)"


def test_lean_ok_only_for_ascii_words():
    assert dfm.lean_tokens_ok(["abc", "x1"])
    assert dfm.lean_tokens_ok([])
    assert not dfm.lean_tokens_ok(["abc", "naïve"])
    assert not any(c.isascii() for c in "".join(dfm._lower_fix()))


def test_i_dot_needs_its_guard():
    con = duckdb.connect()
    assert dfm.tokenize_many(con, ["İstanbul"]) == [["i̇stanbul"]]
    assert con.sql("SELECT lower('İstanbul')").fetchone()[0] == "istanbul"


@pytest.mark.slow
def test_ascii_tokens_equal_on_every_codepoint():
    """Every codepoint alone, inside a token, next to İ and in the Final_Sigma
    contexts: the ASCII tokens of the lean SQL == those of the full SQL."""
    con = duckdb.connect()
    full = dfm.tokens_sql("s")
    lean = dfm._tokens_sql("s", "lower(s)", True)
    cps = [c for c in range(0x110000) if not (0xD800 <= c < 0xE000)]
    shapes = ["{c}", "a{c}b", "x {c}y z", "Α{c}Σ w", "{c}Σ", "ΑΣ{c}",
              "İ{c} i", "i{c}İ"]
    bad = 0
    for b0 in range(0, len(cps), 1 << 16):
        chunk = cps[b0:b0 + (1 << 16)]
        vals = [sh.format(c=chr(c)) for c in chunk for sh in shapes]
        con.register("__t", pa.table({"s": pa.array(vals, pa.large_string())}))
        bad += con.sql(f"""SELECT count(*) FROM __t WHERE
            list_filter({full}, {ASCII_ONLY}) IS DISTINCT FROM
            list_filter({lean}, {ASCII_ONLY})""").fetchone()[0]
        con.unregister("__t")
    assert bad == 0


def masks(phrases, guards, impl, prefilter, field_docs=DOCS):
    spec = dfm.parse_filter({"must": [{"field": "text", "match_text_from_query": "q"}]})
    f = duck_filter(spec, pa.table({"q": pa.array(phrases, pa.large_string())}), threads=2,
                    selfcheck=False, text_impl=impl, text_guards=guards,
                    prefilter_max_phrases=prefilter)
    return f.evaluate(pa.table({"text": pa.array(field_docs, pa.large_string())})).query_masks()


@pytest.mark.parametrize("impl", list(IMPLS))
@pytest.mark.parametrize("prefilter", [0, 10**6])
@pytest.mark.parametrize("phrases", [ASCII_Q, ASCII_Q + NON_ASCII_Q, ASCII_Q[:1]])
def test_auto_masks_equal_full(impl, prefilter, phrases):
    a = masks(phrases, "auto", impl, prefilter)
    b = masks(phrases, "full", impl, prefilter)
    assert np.array_equal(a, b)
    if phrases is ASCII_Q:       # the cases the lean path must not get wrong
        q = {p: i for i, p in enumerate(phrases)}
        d = dfm.unpack(a, len(DOCS)) if a.dtype == np.uint8 and a.shape[1] != len(DOCS) else a
        assert not d[q["istanbul"], DOCS.index("İstanbul trip")]
        assert d[q["istanbul"], DOCS.index("ISTANBUL")]
        assert not d[q["i"], DOCS.index("İ")]
        assert d[q["k"], DOCS.index("KELVIN K k")]


def test_random_corpus_auto_equals_full():
    rng = np.random.default_rng(7)
    pool = [w for d in DOCS if d for w in d.split()]
    docs = [" ".join(rng.choice(pool, int(rng.integers(0, 7))).tolist()) for _ in range(3000)]
    a = masks(ASCII_Q, "auto", "bitmap", 0, docs)
    b = masks(ASCII_Q, "full", "bitmap", 0, docs)
    assert np.array_equal(a, b) and a.any()


def test_bad_text_guards():
    with pytest.raises(ValueError):
        dfm.DuckFilter(dfm.parse_filter(None), text_guards="lean", selfcheck=False)


# ---- popcount_rows / DuckDB-time attribution (same engine module) ----------

@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (3, 0), (1, 1), (7, 3), (40, 17), (300, 1025)])
def test_popcount_rows_equals_unpackbits(shape, monkeypatch):
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    m = rng.integers(0, 256, shape, dtype=np.uint8)
    want = np.unpackbits(m, axis=1).sum(axis=1).astype(np.int64) if m.size \
        else np.zeros(shape[0], np.int64)
    got = dfm.popcount_rows(m)
    assert got.dtype == np.int64 and np.array_equal(got, want)
    monkeypatch.setattr(dfm, "_POPCOUNT_CHUNK", 64)      # many small steps
    assert np.array_equal(dfm.popcount_rows(m), want)
    monkeypatch.delattr(np, "bitwise_count", raising=False)   # the old-numpy fallback
    assert np.array_equal(dfm.popcount_rows(m), want)


def test_factorized_leaf_statements_count_as_duckdb_time():
    """The low-cardinality path's own statements (count distinct, temp table)
    are DuckDB time (sql_query_s), not harness time."""
    n, nq = 4000, 40
    rng = np.random.default_rng(5)
    corpus = pa.table({"s": pa.array(rng.choice(list("abcdefgh"), n).tolist())})
    q = pa.table({"qs": pa.array(rng.choice(list("abcdefghxyz"), nq).tolist())})
    spec = dfm.parse_filter({"must": [{"field": "s", "match_from_query": "qs"}]})
    eng = dfm.DuckFilter(spec, q, threads=2, selfcheck=False, factorize="always")
    calls = []
    plain = eng._x
    eng._x = lambda sql: (calls.append(sql), plain(sql))[1]
    res = eng.evaluate(corpus)
    st = res.stats
    assert st["factorized_exprs"] > 0 and len(calls) >= 3
    assert st["sql_query_s"] > 0 and st["sql_post_s"] >= 0
    assert st["sql_query_s"] + st["sql_post_s"] == pytest.approx(st["sql_s"])
    ref = dfm.DuckFilter(spec, q, threads=2, selfcheck=False, factorize="never").evaluate(corpus)
    assert np.array_equal(res.query_masks(), ref.query_masks())
