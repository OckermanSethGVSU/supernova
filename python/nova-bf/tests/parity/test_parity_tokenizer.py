"""`match_text` tokenization against the live engine, on non-ASCII text.

The shared corpus is ASCII on purpose (so `naive.tokens` can be exact), which
leaves the tokenizer's Unicode behaviour checked by nobody but itself. Here it
is checked against Qdrant directly: a small corpus of awkward strings (case,
punctuation, combining marks, Indic / Arabic / Thai / Hebrew, circled letters,
final sigma, dotted capital I, CJK, emoji, fullwidth, ligatures) and one
per-query phrase per probe, with nova-bf's eligible set compared to the one a
Qdrant 1.18.2 `word` tokenizer (`lowercase: true`) returned.

`QDRANT_ANSWERS` is that measurement, recorded, so the nova-bf half runs
anywhere; `test_recorded_answers_are_still_qdrants` re-measures it against the
live server, so a Qdrant upgrade that changes tokenization fails loudly
instead of leaving this file asserting stale truth.

Token BOUNDARIES now follow the measured Qdrant word-character table
(`nova_bf/tokenizer_tables/qdrant_word_chars.py`; `test_word_char_table_is_still_qdrants`
re-probes it on the live server), so combining vowel signs, harakat, niqqud
and circled letters agree. Lowercasing follows Qdrant's per-token Rust
`to_lowercase` (`İ` -> `i` U+0307, Final_Sigma within the token; see
`nova_bf.tokenize`), so there are no known divergences: `DIVERGENT` is empty.
A future one goes there as `xfail(strict=True)` with the measured answer in
the reason. `test_final_sigma_contexts_are_still_qdrants` re-measures the
Final_Sigma context classes on the live server.
"""

from __future__ import annotations

import types
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from .runner import build_config, env, pinned_device, read_results

DOCS = [
    "Hello World", "hello_world", "it's", "don't stop", "a-b c.d e,f", "x1y2 42",
    "caf\xe9", "café", "Stra\xdfe", "İstanbul", "ΟΔΟΣ ΟΔΌΣ",
    "σίσυφος", "हिन्दी भाषा",
    "مَرْحَبًا بِكُمْ",
    "สวัสดีครับ",
    "שָׁלוֹם עוֹלָם",
    "ⓐⓑⓒ ⓗⓔⓛⓛⓞ", "中文分词测试",
    "日本語のテキスト", "한국어 텍스트",
    "emoji \U0001f600 test", "\xdf", "ǅemal", "Ⅻ roman", "\xbd fraction", "x\xb2",
    "ＡＢＣ fullwidth", "ﬁ ligature", "\xd6 ö \xf6", "", "!!! ---",
    # 31..: lowercasing context (Final_Sigma, İ)
    "ΟΔΟΣΑ", "ΑΣ'Β", "ΑΣΣ", "ΑΣ1", "Α\u02b0Σ", "\u02b0Σ", "ΑΣ\u0345", "iİ", "ⒶΣ", "Α'Σ", "ΣΑ",
]

# phrase -> doc indices Qdrant 1.18.2 matched (measured 2026-11-11)
QDRANT_ANSWERS = [
    ('hello', [0, 1]), ('HELLO world', [0, 1]), ('hello_world', [0, 1]),
    ("it's", [2]), ('s', [2]), ('don', [3]), ('x1y2', [5]), ('42', [5]), ('c.d', [4]),
    ('caf\xe9', [6]), ('cafe', [7]), ('café', [7]), ('stra\xdfe', [8]),
    ('strasse', []), ('istanbul', []), ('İstanbul', [9]),
    ('ΟΔΟΣ', [10]), ('οδος', [10]),
    ('οδοσ', []), ('σίσυφος', [11]),
    ('हिन्दी', [12]), ('दी', [12]), ('ह', []),
    ('مَرْحَبًا', [13]),
    ('مرحبا', []), ('م', []),
    ('بِكُمْ', [13]),
    ('สวัสดีครับ', [14]), ('สว', []),
    ('שָׁלוֹם', [15]), ('שלום', []),
    ('ש', []), ('ⓐⓑⓒ', [16]), ('ⓐ', []),
    ('中文分词测试', [17]), ('中', []), ('한국어', [19]),
    ('emoji test', [20]), ('\U0001f600', []), ('\xdf', [21]), ('ǆemal', [22]),
    ('ⅻ', [23]), ('\xbd', [24]), ('x\xb2', [25]), ('x', []), ('ａｂｃ', [26]),
    ('abc', []), ('ﬁ', [27]), ('fi', []), ('\xf6', [28]), ('ö', [28]),
    ('', []), ('!!!', []),
    ('οδοσα', [31]), ('οδοςα', []), ('ΑΣ', [32]), ('ας', [32]), ('ασ', []), ('β', [32]),
    ('ασς', [33]), ('ασσ', []), ('ας1', [34]), ('ασ1', []), ('α\u02b0ς', [35]),
    ('α\u02b0σ', []), ('\u02b0σ', [36]), ('\u02b0ς', []), ('ας\u0345', [37]),
    ('ασ\u0345', []), ('iİ', [38]), ('ii', []), ('ⓐς', [39]), ('ⓐσ', []), ('σ', [40]),
    ('ς', []), ('α', [40]), ('İSTANBUL', [9]), ('σα', [41]), ('ςα', []),
]

# Measured divergences: phrase -> what nova-bf does instead, and why. None
# known (the former three, 'istanbul' / 'οδος' / 'οδοσ', came from Arrow's
# simple lowercasing and now pass as ordinary cases).
DIVERGENT: dict[str, str] = {}


@pytest.fixture(scope="module")
def tok_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tok")
    (tmp / "corpus").mkdir()
    n = len(DOCS)
    pq.write_table(pa.table({
        "dense_embedding": pa.array([[1.0, float(i)] for i in range(n)], pa.list_(pa.float32())),
        "id": pa.array([str(i) for i in range(n)]),
        "t": pa.array(DOCS)}), str(tmp / "corpus" / "part-000.parquet"))
    phrases = [p for p, _ in QDRANT_ANSWERS]
    pq.write_table(pa.table({
        "dense_embedding": pa.array([[1.0, 0.0]] * len(phrases), pa.list_(pa.float32())),
        "qid": pa.array([str(i) for i in range(len(phrases))]),
        "ph": pa.array(phrases)}), str(tmp / "queries.parquet"))
    return types.SimpleNamespace(tmp=str(tmp), corpus_dir=str(tmp / "corpus"),
                                 queries_path=str(tmp / "queries.parquet"),
                                 date_fields={}, query_date_fields={})


@pytest.fixture(scope="module", params=["native", "arrow"])
def tok_run(request, tok_data, device):
    """nova-bf's eligible set per phrase, with the native scanner and with the
    Arrow fallback (`NOVA_BF_NO_NATIVE_TOKENIZER`) — both must match Qdrant."""
    from nova_bf.compute import run_compute

    search = {"name": "tok", "vector_type": "dense", "metric": "dot", "k": len(DOCS) + 5,
              "filter": {"must": [{"field": "t", "match_text_from_query": "ph"}]}}
    cfg = build_config(tok_data, [search], out_tag=f"tok_{request.param}_{device}")
    with env(NOVA_BF_NO_NATIVE_TOKENIZER="1" if request.param == "arrow" else None), \
            pinned_device(device):
        res = read_results(run_compute(cfg))["tok"]
    return {qi: sorted(i for i, _ in hits) for qi, hits in res.items()}


def _params():
    out = []
    for qi, (phrase, want) in enumerate(QDRANT_ANSWERS):
        marks = []
        if phrase in DIVERGENT:
            marks.append(pytest.mark.xfail(strict=True, reason=(
                f"DIVERGENCE vs Qdrant (measured: {want}): {DIVERGENT[phrase]}")))
        out.append(pytest.param(qi, phrase, want, marks=marks,
                                id=phrase.encode("unicode_escape").decode() or "<empty>"))
    return out


@pytest.mark.parametrize("qi,phrase,want", _params())
def test_match_text_tokenization_matches_qdrant(qi, phrase, want, tok_run):
    assert tok_run[qi] == want, (
        f"phrase {phrase!r}: nova-bf matched {[DOCS[i] for i in tok_run[qi]]}, "
        f"Qdrant {[DOCS[i] for i in want]}")


def test_static_match_text_on_circled_letters_is_accepted(tok_data):
    """Circled letters are Alphabetic, so `ⓐⓑⓒ` is a token (Qdrant matches doc
    16) and a static match_text on it is a valid filter, not a config error."""
    from nova_bf.tokenize import tokenize

    assert tokenize("ⓐⓑⓒ") == ["ⓐⓑⓒ"]
    build_config(tok_data, [{"name": "s", "vector_type": "dense", "metric": "dot", "k": 3,
                             "filter": {"must": [{"field": "t", "match_text": "ⓐⓑⓒ"}]}}],
                 out_tag="unused")


@pytest.mark.qdrant
def test_recorded_answers_are_still_qdrants(client):
    """Re-measure QDRANT_ANSWERS on the live server."""
    from qdrant_client import models

    name = f"nova_bf_tok_{uuid.uuid4().hex[:8]}"
    client.create_collection(name, vectors_config=models.VectorParams(
        size=2, distance=models.Distance.DOT))
    try:
        client.create_payload_index(name, field_name="t", field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD,
            lowercase=True), wait=True)
        client.upsert(name, points=[models.PointStruct(id=i, vector=[1.0, 0.0],
                                                       payload={"t": t})
                                    for i, t in enumerate(DOCS)], wait=True)
        from .conftest import _QDRANT

        _QDRANT["graded"] = True       # an oracle collection is loaded (see conftest)
        for phrase, want in QDRANT_ANSWERS:
            pts = client.scroll(name, scroll_filter=models.Filter(must=[models.FieldCondition(
                key="t", match=models.MatchText(text=phrase))]), limit=100)[0]
            assert sorted(p.id for p in pts) == want, f"Qdrant now answers {phrase!r} differently"
    finally:
        client.delete_collection(name)


def test_divergence_table_names_only_recorded_phrases():
    assert set(DIVERGENT) <= {p for p, _ in QDRANT_ANSWERS}


def _word_char_probe_set() -> list[int]:
    """Every range edge of the recorded table (first/last and the codepoints
    just outside), known cases, and a fixed random sample."""
    import random

    from nova_bf.tokenizer_tables.qdrant_word_chars import WORD_CHAR_RANGES

    cps = {0x78, 0x2D, 0x5F, 0x20, 0x093F, 0x094D, 0x24D0, 0x0301, 0x1F600}
    for lo, hi in WORD_CHAR_RANGES:
        cps |= {lo - 1, lo, hi, hi + 1}
    rng = random.Random(1234)
    cps |= {rng.randrange(0x110000) for _ in range(4000)}
    return sorted(c for c in cps if 0 <= c < 0x110000 and not 0xD800 <= c < 0xE000)


@pytest.mark.qdrant
def test_word_char_table_is_still_qdrants(client):
    """Re-probe the recorded word-character table on the live server (same
    `a<cp>b` / `MatchText("a")` method as tools/gen_qdrant_word_chars.py, on
    every range edge plus a sample). A failure means the server's tokenizer
    changed: re-run the generator."""
    from qdrant_client import models

    from nova_bf.tokenizer_tables.qdrant_word_chars import WORD_CHAR_RANGES

    cps = _word_char_probe_set()
    word = {c for lo, hi in WORD_CHAR_RANGES for c in range(lo, hi + 1)}
    name = f"nova_bf_wc_{uuid.uuid4().hex[:8]}"
    client.create_collection(name, vectors_config=models.VectorParams(
        size=1, distance=models.Distance.DOT))
    try:
        client.create_payload_index(name, field_name="t", field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD,
            lowercase=True), wait=True)
        for i in range(0, len(cps), 2000):
            client.upsert(name, points=[models.PointStruct(id=c, vector=[1.0],
                                                           payload={"t": "a" + chr(c) + "b"})
                                        for c in cps[i:i + 2000]], wait=True)
        from .conftest import _QDRANT

        _QDRANT["graded"] = True
        cond = models.FieldCondition(key="t", match=models.MatchText(text="a"))
        got, offset = [], None
        while True:
            pts, offset = client.scroll(name, scroll_filter=models.Filter(must_not=[cond]),
                                        limit=10_000, offset=offset)
            got += [p.id for p in pts]
            if offset is None:
                break
        want = sorted(c for c in cps if c in word)
        diff = sorted(set(got) ^ set(want))
        assert not diff, (f"Qdrant's word characters changed at "
                          f"{[f'U+{c:04X}' for c in diff[:20]]} ({len(diff)} probed): "
                          f"re-run tools/gen_qdrant_word_chars.py")
    finally:
        client.delete_collection(name)


@pytest.mark.qdrant
def test_final_sigma_contexts_are_still_qdrants(client):
    """Final_Sigma and the İ expansion on the live server, per context class:
    every Case_Ignorable word character plus samples of the Cased ones and of
    the rest, each as `Α c Σ`, `c Σ`, `ΑΣ c`, `İ c`. Phrases are nova-bf's
    tokens for each doc and their σ/ς twins; Qdrant must return exactly the
    docs nova-bf's tokens predict. (The full sweep, all 4364 Cased word chars
    and 3000 others, also matched: 26k docs, 43k phrases, 2026-11-11.)"""
    import collections
    import random

    from qdrant_client import models

    from nova_bf.tokenizer_tables.qdrant_word_chars import WORD_CHAR_RANGES
    from nova_bf.tokenizer_tables.rust_case_props import CASE_IGNORABLE_RANGES, CASED_RANGES
    from nova_bf.tokenize import tokenize_many

    word = {c for lo, hi in WORD_CHAR_RANGES for c in range(lo, hi + 1)}
    ci = sorted(c for lo, hi in CASE_IGNORABLE_RANGES for c in range(lo, hi + 1) if c in word)
    cased = sorted(c for lo, hi in CASED_RANGES for c in range(lo, hi + 1) if c in word)
    rng = random.Random(42)
    cps = ci + rng.sample(cased, 600) + rng.sample(sorted(word - set(ci) - set(cased)), 300)
    docs = [d for c in cps for d in ("Α" + chr(c) + "Σ", chr(c) + "Σ", "ΑΣ" + chr(c),
                                     "İ" + chr(c))]
    dtoks = [set(t) for t in tokenize_many(docs)]
    inv = collections.defaultdict(set)
    for i, ts in enumerate(dtoks):
        for t in ts:
            inv[t].add(i)
    phrases = sorted({v for ts in dtoks for t in ts
                      for v in (t, t.replace("ς", "σ"), t.replace("σ", "ς"))})
    ptoks = tokenize_many(phrases)
    name = f"nova_bf_sigma_{uuid.uuid4().hex[:8]}"
    client.create_collection(name, vectors_config=models.VectorParams(
        size=1, distance=models.Distance.DOT))
    try:
        client.create_payload_index(name, field_name="t", field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD,
            lowercase=True), wait=True)
        for i in range(0, len(docs), 2000):
            client.upsert(name, points=[models.PointStruct(id=j, vector=[1.0],
                                                           payload={"t": docs[j]})
                                        for j in range(i, min(i + 2000, len(docs)))], wait=True)
        from .conftest import _QDRANT

        _QDRANT["graded"] = True
        bad = []
        for p, pt in zip(phrases, ptoks):
            if not pt:
                continue
            want = set.intersection(*[inv.get(t, set()) for t in pt])
            got, offset = set(), None
            while True:
                pts, offset = client.scroll(name, scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="t", match=models.MatchText(text=p))]),
                    limit=10_000, offset=offset, with_payload=False)
                got |= {q.id for q in pts}
                if offset is None:
                    break
            if got != want:
                bad.append((p, sorted(got)[:3], sorted(want)[:3]))
        assert not bad, f"{len(bad)} phrases differ from Qdrant, e.g. {bad[:5]}"
    finally:
        client.delete_collection(name)
