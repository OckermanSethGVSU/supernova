"""THE tokenizer for `match_text`/`match_text_from_query` — one definition,
used by both sides of every text match.

Rule. Qdrant's `word` tokenizer with `lowercase: true`, in its order of
operations: split the ORIGINAL text into maximal runs of word characters,
then lowercase each token (token boundaries never depend on case mapping). A
row matches a phrase iff every one of the phrase's tokens is one of the row's
tokens.

Word characters. Exactly the codepoints in `tokenizer_tables.qdrant_word_chars.WORD_CHAR_RANGES`,
a table MEASURED against a live Qdrant (version/commit and date in that
module's header) by `tools/gen_qdrant_word_chars.py`: every Unicode scalar
value was probed as `a<cp>b` in a `word`/lowercase full-text index and counted
as a word character iff `MatchText("a")` did not match (repeated with a second
probe base). The measured set is what Rust's `char::is_alphanumeric`
(Unicode Alphabetic | Numeric) gives on Unicode 17 — e.g. combining vowel
signs (Devanagari, Arabic harakat, Thai, Hebrew niqqud) and circled letters
stay INSIDE a token, while marks that are not Alphabetic (the Devanagari
virama U+094D, U+0301 COMBINING ACUTE) split it: `हिन्दी` -> `हिन`, `दी`.
`TOKEN_SPLIT_PATTERN` spells the table out as an explicit codepoint class,
so it does not depend on the Unicode tables of the RE2 bundled with Arrow (or
DuckDB); the native scanner (`nativetok`) is built from the same ranges. To
follow a newer Qdrant, re-run the generator.

Lowercasing. Qdrant lowercases each TOKEN with Rust's `str::to_lowercase`
(measured: the context below stops at the token's edges, e.g. `ΑΣ'Β` gives
`ας`, `β`). That is Arrow's simple per-codepoint `utf8_lower` (compared on
every word character, pyarrow 24 vs Rust 1.97 / Unicode 17, 2026-11-11) plus
exactly two rules, both applied here (`lower_tokens`):

* `İ` U+0130 lowercases to two codepoints, `i` U+0069 + U+0307 (the only
  multi-codepoint lowercase mapping). U+0307 is not a word character, so a
  query can only reach such a token through a phrase that itself has `İ`:
  `istanbul` does not match `İstanbul`.
* Final_Sigma: `Σ` U+03A3 becomes `ς` U+03C2 iff, within the token, the chars
  before it are Case_Ignorable* then a Cased char AND the chars after it are
  not Case_Ignorable* then a Cased char; otherwise `σ` (alloc's
  `map_uppercase_sigma`). The two properties come from
  `tokenizer_tables/rust_case_props.py`, derived from rustc's own `to_lowercase` by
  `tools/gen_rust_case_props.py`. So `ΟΔΟΣ` -> `οδος`, `ΑΣΣ` -> `ασς`,
  `ΑΣ1` -> `ας1` (a digit is not Cased), `Σ` alone -> `σ`.

The fix-up runs on Arrow's output only for tokens holding U+0130 or U+03A3
(vectorised detection), so the common path stays one Arrow kernel. The
native scanner (`nativetok`) implements the same two rules in Rust from the
same tables, and both are checked against `qdrant_lower_token` (the plain
Python statement of the rule) and against a live Qdrant.

One definition for corpus and query. The corpus-column path
(`filters._token_row_masks`, Arrow or native scanner) and the query-side
helpers here use the same split class and the same `lower_tokens`, so corpus
and query tokens agree BY CONSTRUCTION. (An earlier draft tokenized queries
with Python `str.lower`/`str.isalnum`; Python and Arrow disagree on thousands
of codepoints, so query tokens could silently never match corpus tokens.)

Divergences vs Qdrant: none known for the `word` tokenizer with
`lowercase: true` on Qdrant 1.18.2. Measured scope (2026-11-11): token
boundaries on every Unicode scalar value; lowercasing on every word
character in isolation; Final_Sigma in the before/after contexts of every
Case_Ignorable and every Cased word character plus 3000 other word
characters (26k documents, 43k phrases, 0 mismatches); the hand-picked cases
in `tests/parity/test_parity_tokenizer.py`. Other Arrow, Qdrant or Rust
builds bundle other Unicode versions and could differ on recently encoded
characters; re-run the two generators and the live tests after an upgrade.
Other Qdrant tokenizer options (stemmer, stopwords, ascii_folding, token
length limits) are not modelled.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

import numpy as np

from nova_bf.tokenizer_tables.qdrant_word_chars import WORD_CHAR_RANGES
from nova_bf.tokenizer_tables.rust_case_props import CASE_IGNORABLE_RANGES, CASED_RANGES


def _re2_class(ranges) -> str:
    """RE2 class body for inclusive codepoint ranges (`\\x{..}-\\x{..}` ...)."""
    return "".join(f"\\x{{{lo:X}}}" if lo == hi else f"\\x{{{lo:X}}}-\\x{{{hi:X}}}"
                   for lo, hi in ranges)


# Maximal runs of anything OUTSIDE the measured word characters separate
# tokens (splitting happens BEFORE lowercasing, see module docstring).
TOKEN_SPLIT_PATTERN = "[^" + _re2_class(WORD_CHAR_RANGES) + "]+"

CAPITAL_SIGMA = "\u03a3"
DOTTED_CAPITAL_I = "\u0130"
# A token needs `qdrant_lower_token` beyond `utf8_lower` iff it holds one of these.
SPECIAL_LOWER_PATTERN = "[\\x{130}\\x{3A3}]"

_CASE_IGNORABLE = frozenset(c for lo, hi in CASE_IGNORABLE_RANGES for c in range(lo, hi + 1))
_CASED = frozenset(c for lo, hi in CASED_RANGES for c in range(lo, hi + 1))


def _ignorable_then_cased(chars) -> bool:
    """alloc's `case_ignorable_then_cased`: skip Case_Ignorable chars; is the
    next char present and Cased? (`_CASED` excludes case-ignorables, which
    are never asked.)"""
    for ch in chars:
        cp = ord(ch)
        if cp not in _CASE_IGNORABLE:
            return cp in _CASED
    return False


def qdrant_lower_token(token: str, simple: str) -> str:
    """Rust `token.to_lowercase()`, given `simple` = Arrow `utf8_lower(token)`
    (one codepoint per codepoint): replace each `İ` by `i` U+0307 and each
    `Σ` by `ς`/`σ` per Final_Sigma within `token` (see module docstring)."""
    if len(simple) != len(token):
        raise ValueError(f"utf8_lower changed the length of {token!r}; the "
                         f"per-codepoint lowercase fix-up cannot be applied")
    out = list(simple)
    for k, ch in enumerate(token):
        if ch == CAPITAL_SIGMA:
            final = (_ignorable_then_cased(reversed(token[:k]))
                     and not _ignorable_then_cased(token[k + 1:]))
            out[k] = "\u03c2" if final else "\u03c3"
        elif ch == DOTTED_CAPITAL_I:
            out[k] = "i\u0307"
    return "".join(out)


def lower_tokens(tokens):
    """Lowercase an Arrow string array of TOKENS exactly like Qdrant: Arrow
    `utf8_lower`, then `qdrant_lower_token` on just the tokens holding
    U+0130 / U+03A3 (found with one vectorised regex match)."""
    lowered = pc.utf8_lower(tokens)
    hit = pc.match_substring_regex(tokens, SPECIAL_LOWER_PATTERN)
    if not pc.any(hit).as_py():
        return lowered
    mask = pc.fill_null(hit, False)
    idx = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
    fixed = [qdrant_lower_token(t, s) for t, s in zip(
        tokens.take(idx).to_pylist(), lowered.take(idx).to_pylist())]
    return pc.replace_with_mask(lowered, mask, pa.array(fixed, type=lowered.type))


def tokenize_many(values) -> list[list[str] | None]:
    """Token lists for a whole sequence of strings in one set of Arrow kernel
    calls — per-string round-trips would dominate on query-log-scale inputs.
    A non-string entry (`None`, `nan`, anything else) yields `None` (the
    "matches nothing" convention for null per-query values); leading/trailing
    separators' empty-string split artifacts are dropped, so a string with no
    word characters yields `[]`."""
    clean = [v if isinstance(v, str) else None for v in values]
    toks = pc.split_pattern_regex(
        pa.array(clean, type=pa.large_string()), pattern=TOKEN_SPLIT_PATTERN,
    )
    lowered = lower_tokens(pc.list_flatten(toks)).to_pylist()
    lengths = pc.list_value_length(toks).to_pylist()
    out: list[list[str] | None] = []
    i = 0
    for n in lengths:
        if n is None:
            out.append(None)
        else:
            out.append([t for t in lowered[i:i + n] if t])
            i += n
    return out


def tokenize(text: str) -> list[str]:
    """Single-string convenience over `tokenize_many` — config validation and
    static `match_text` phrases; hot paths use the batched form."""
    toks = tokenize_many([text])[0]
    return toks if toks is not None else []
