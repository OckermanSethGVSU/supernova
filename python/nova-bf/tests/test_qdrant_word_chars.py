"""The measured Qdrant word-character table drives every tokenizer path.

`nova_bf/tokenizer_tables/qdrant_word_chars.py` records which codepoints Qdrant's `word`
tokenizer keeps inside a token. Both corpus paths (Arrow `split_pattern_regex`
on `TOKEN_SPLIT_PATTERN`, and the native scanner) and the query-side
`tokenize_many` must split exactly there, on EVERY Unicode scalar value. Each
codepoint is probed as `a<cp>b`: a word character leaves one token, a
separator leaves `a` and `b`. Run in chunks to bound memory.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pytest

from nova_bf import nativetok
from nova_bf.tokenizer_tables.qdrant_word_chars import N_WORD_CHARS, WORD_CHAR_RANGES
from nova_bf.tokenize import TOKEN_SPLIT_PATTERN, tokenize, tokenize_many

N = 0x110000
CPS = np.array([c for c in range(N) if not 0xD800 <= c < 0xE000])
WORD = np.zeros(N, dtype=bool)
for _lo, _hi in WORD_CHAR_RANGES:
    WORD[_lo:_hi + 1] = True
CHUNK = 1 << 16


def _chunks():
    for i in range(0, len(CPS), CHUNK):
        cps = CPS[i:i + CHUNK]
        yield cps, pa.array(["a" + chr(c) + "b" for c in cps], type=pa.large_string())


def test_table_is_well_formed():
    prev = -2
    for lo, hi in WORD_CHAR_RANGES:
        assert lo <= hi and lo > prev + 1, (hex(lo), hex(hi))  # sorted, disjoint, non-adjacent
        assert not (lo <= 0xDFFF and hi >= 0xD800), "surrogates are not scalar values"
        prev = hi
    assert prev < N
    assert int(WORD.sum()) == N_WORD_CHARS
    # the probe bases themselves must be word characters
    assert all(WORD[ord(c)] for c in "abqz")


@pytest.mark.parametrize("cp,word", [
    (ord("x"), True), (ord("-"), False), (ord("_"), False), (0x093F, True),
    (0x094D, False), (0x24D0, True), (0x0301, False), (0x05B0, True),
    (0x064E, True), (0x0E31, True), (0x1F600, False),
])
def test_known_codepoints(cp, word):
    assert WORD[cp] == word
    assert (tokenize("a" + chr(cp) + "b") == ["a", "b"]) != word


def test_arrow_split_path_matches_the_table_on_every_codepoint():
    """`arrow_grid` is the corpus Arrow fallback's exact kernel sequence."""
    for cps, arr in _chunks():
        sep = nativetok.arrow_grid(arr, ["a"])[0]
        bad = cps[sep == WORD[cps]]
        assert not bad.size, [f"U+{c:04X}" for c in bad[:20]]


def test_query_tokenizer_matches_the_table_on_every_codepoint():
    for cps, arr in _chunks():
        toks = tokenize_many(arr.to_pylist())
        sep = np.array([t[0] == "a" for t in toks])
        assert all(len(t) == (1 if w else 2) for t, w in zip(toks, WORD[cps]))
        bad = cps[sep == WORD[cps]]
        assert not bad.size, [f"U+{c:04X}" for c in bad[:20]]


def test_native_scanner_matches_the_table_on_every_codepoint():
    if nativetok._ns is None or os.environ.get("NOVA_BF_NO_NATIVE_TOKENIZER"):
        pytest.skip(f"native tokenizer unavailable: {nativetok.unavailable_reason()}")
    # Built but disabled (table derivation or self-check failed) is a failure:
    # the self-check disagreeing means the tables are wrong.
    assert nativetok.available(), nativetok.unavailable_reason()
    vocab = nativetok.prepare(["a"])
    assert vocab is not None
    assert np.array_equal(nativetok._alnum_cp_table().astype(bool), WORD)
    for cps, arr in _chunks():
        grid = np.zeros((1, len(arr)), dtype=bool)
        nativetok.scan_into(arr, vocab, grid)
        bad = cps[grid[0] == WORD[cps]]
        assert not bad.size, [f"U+{c:04X}" for c in bad[:20]]


def test_split_pattern_is_built_from_the_table_only():
    """No Unicode property classes: the boundary must not depend on the Unicode
    tables of whichever RE2 Arrow (or DuckDB) bundles."""
    assert "\\p" not in TOKEN_SPLIT_PATTERN and "\\P" not in TOKEN_SPLIT_PATTERN
    assert TOKEN_SPLIT_PATTERN.startswith("[^") and TOKEN_SPLIT_PATTERN.endswith("]+")


def test_qdrant_measured_examples():
    assert tokenize("हिन्दी भाषा") == ["हिन", "दी", "भाषा"]
    assert tokenize("ⓐⓑⓒ ⓗⓔⓛⓛⓞ") == ["ⓐⓑⓒ", "ⓗⓔⓛⓛⓞ"]
    assert tokenize("مَرْحَبًا") == ["مَرْحَبًا"]
    assert tokenize("he\u0301llo") == ["he", "llo"]   # U+0301 is not Alphabetic
