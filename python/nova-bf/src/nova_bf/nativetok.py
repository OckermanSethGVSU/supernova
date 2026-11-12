"""Native `match_text` corpus scanning.

`nova_textscan` performs tokenization, lowercasing, vocabulary lookup, and
row-mask construction in Rust with the GIL released. This module prepares the
per-codepoint tables and buffers required by that scanner and decides whether
the native path is safe to use.

Token boundaries come from the measured Qdrant word-character table
(`tokenizer_tables.qdrant_word_chars.WORD_CHAR_RANGES`, the same ranges `TOKEN_SPLIT_PATTERN`
spells out for the Arrow path). Lowercasing is Rust `str::to_lowercase` per
token, as Qdrant does it (`nova_bf.tokenize`): the one-codepoint map is
derived from the installed Arrow build's `utf8_lower` rather than hardcoded,
the scanner adds the `İ` -> `i` U+0307 expansion and Final_Sigma within the
token, and the Case_Ignorable / Cased flags Final_Sigma needs come from
`tokenizer_tables.rust_case_props` (the same tables `tokenize.qdrant_lower_token` uses).
Hashes are used only to locate vocabulary candidates; matches are confirmed
against the full lowered token.

If the extension is unavailable or too old (its `features()` must list every
entry of `REQUIRED_FEATURES`; an older build is refused with one warning, not
used with older lowercasing), table derivation is unsupported, the vocabulary
cannot be represented, or the native self-check disagrees with the Arrow
implementation, the native path is disabled and callers fall back to Arrow.

Set `NOVA_BF_NO_NATIVE_TOKENIZER` to disable the native path explicitly.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nova_bf.tokenizer_tables.qdrant_word_chars import WORD_CHAR_RANGES
from nova_bf.tokenizer_tables.rust_case_props import CASE_IGNORABLE_RANGES, CASED_RANGES
from nova_bf.tokenize import TOKEN_SPLIT_PATTERN, lower_tokens

logger = logging.getLogger(__name__)

__all__ = ["available", "prepare", "scan_into", "unavailable_reason"]

N_CODEPOINTS = 0x110000

# Lowercasing rules the extension must implement (`nova_textscan.features()`).
REQUIRED_FEATURES = ("dotted_capital_i_expands", "final_sigma_in_token")
# `case` table bits, as in crates/nova-textscan/src/tokenizer.rs.
CASE_IGNORABLE, CASED = 1, 2

def _alnum_cp_table() -> np.ndarray:
    """Word-character flag per codepoint, from the measured Qdrant table — the
    same `WORD_CHAR_RANGES` `TOKEN_SPLIT_PATTERN` is built from, so the native
    and Arrow paths split identically (not derived from Arrow's RE2)."""
    alnum = np.zeros(N_CODEPOINTS, dtype=np.uint8)
    for lo, hi in WORD_CHAR_RANGES:
        alnum[lo:hi + 1] = 1
    return alnum

def _case_cp_table() -> np.ndarray:
    """`CASE_IGNORABLE` / `CASED` bits per codepoint for Final_Sigma."""
    case = np.zeros(N_CODEPOINTS, dtype=np.uint8)
    for lo, hi in CASE_IGNORABLE_RANGES:
        case[lo:hi + 1] |= CASE_IGNORABLE
    for lo, hi in CASED_RANGES:
        case[lo:hi + 1] |= CASED
    return case


def _missing_features(ns) -> list[str] | None:
    """Required lowercasing rules `ns` lacks (`None` = all present)."""
    try:
        have = set(ns.features())
    except Exception:
        have = set()
    missing = [f for f in REQUIRED_FEATURES if f not in have]
    return missing or None


try:
    import nova_textscan as _ns
except Exception as exc:  # not built, or built for another ABI
    _ns = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _IMPORT_ERROR = None

# Serialize one-time table derivation and self-check across reader threads.
# Reentrant because the self-check can call back into table access.
_INIT = threading.RLock()

_TABLES: object = None
_TABLES_REASON: str | None = None
_PROBED = False
_SELFCHECK: str | None = None


def unavailable_reason() -> str | None:
    """Why the native scan is not in use, or `None` if it is usable."""
    if os.environ.get("NOVA_BF_NO_NATIVE_TOKENIZER"):
        return "NOVA_BF_NO_NATIVE_TOKENIZER"
    if _ns is None:
        return f"nova_textscan not importable ({_IMPORT_ERROR})"
    if _tables() is None:
        return _TABLES_REASON
    return _probe()


def available() -> bool:
    return unavailable_reason() is None

def _lower_cp_table() -> np.ndarray | None:
    """Derive Arrow's per-codepoint lowercase map, or `None` if it expands."""
    cps = [c for c in range(N_CODEPOINTS) if not (0xD800 <= c < 0xE000)]
    arr = pa.array([chr(c) for c in cps], type=pa.large_string())
    low = pc.utf8_lower(arr).to_pylist()

    # Surrogates cannot occur in valid Arrow UTF-8 strings, so their entries
    # remain identity mappings.
    out = np.arange(N_CODEPOINTS, dtype=np.uint32)

    for c, s in zip(cps, low):
        if s is None or len(s) != 1:
            logger.info(
                "native tokenizer: this Arrow build lowercases U+%04X to %r, "
                "which is not a single codepoint; a per-codepoint case table "
                "cannot express it", c, s,
            )
            return None
        out[c] = ord(s)

    return out


def _tables():
    """The two derived tables, built once. `None` if any assertion fails."""
    global _TABLES, _TABLES_REASON

    if _TABLES is not None or _TABLES_REASON is not None:
        return _TABLES
    with _INIT:
        if _TABLES is not None or _TABLES_REASON is not None:
            return _TABLES
        return _derive()
def _derive():
    """Build and cache the native scanner tables."""
    global _TABLES, _TABLES_REASON

    if _ns is None:
        _TABLES_REASON = f"nova_textscan not importable ({_IMPORT_ERROR})"
        return None

    missing = _missing_features(_ns)
    if missing:
        ver = getattr(_ns, "version", lambda: "?")()
        _TABLES_REASON = (f"nova_textscan {ver} is too old: it lacks {missing}; "
                          f"rebuild it (crates/nova-textscan/build_native.sh)")
        logger.warning("native tokenizer disabled: %s. The Arrow pipeline computes "
                       "the same masks, more slowly.", _TABLES_REASON)
        return None

    try:
        alnum_cp = np.ascontiguousarray(_alnum_cp_table(), dtype=np.uint8)
        if alnum_cp.shape != (N_CODEPOINTS,):
            _TABLES_REASON = f"ALNUM_CP has shape {alnum_cp.shape}"
            return None

        lower_cp = _lower_cp_table()
        if lower_cp is None:
            _TABLES_REASON = "utf8_lower is not a per-codepoint mapping here"
            return None

        # The Rust ASCII fast path requires ASCII to lowercase to ASCII.
        bad_ascii = np.flatnonzero(lower_cp[:128] >= 0x80)
        if bad_ascii.size:
            cp = int(bad_ascii[0])
            _TABLES_REASON = (
                f"utf8_lower maps ASCII U+{cp:04X} to non-ASCII "
                f"U+{int(lower_cp[cp]):04X}"
            )
            return None

        # Token boundaries are determined before lowercasing on both paths, so
        # mappings that cross the alphanumeric class do not affect tokenization.
        moved = int(np.count_nonzero(alnum_cp != alnum_cp[lower_cp]))
        if moved:
            logger.debug(
                "native tokenizer: %d codepoint(s) change alphanumeric class "
                "under this build's case map; token boundaries are decided "
                "before lowering on both sides, so this does not affect the "
                "result", moved,
            )

        _TABLES = _ns.ScanTables(alnum_cp, lower_cp, _case_cp_table())

    except Exception as exc:
        _TABLES_REASON = f"{type(exc).__name__}: {exc}"
        logger.info("native tokenizer: table derivation failed: %s", _TABLES_REASON)
        return None

    return _TABLES

# Fixed self-check covering case folding, separators, combining marks (split
# and kept), circled letters, non-ASCII letters/scripts, digits, special
# lowercase mappings (İ expansion, Final_Sigma in and out of context), nulls,
# empty strings, and token-length distinctions.
_SELFCHECK_ROWS = [
    "Hello, WORLD! hello",
    ",leading and trailing,",
    "snake_case CamelCase",
    "héllo café café",
    "İSTANBUL and 100K and ı",
    "текст hello мир 42",
    "a" * 9 + " " + "a" * 8,
    "",
    None,
    "ǅungla ǄUNGLA",
    "हिन्दी ⓐⓑⓒ مَرْحَبًا",
    "ΟΔΟΣ ΣΑ Σ ΑΣΣ ΑΣ1 Α\u02b0Σ \u02b0Σ ΑΣ\u0345 ΑΣ\u02b0Β ΑΣ'Β",
    "iİ İ ⒶΣ",
]

_SELFCHECK_VOCAB = [
    "hello", "world", "leading", "trailing", "snake", "case", "camelcase",
    "he", "llo", "café", "café", "istanbul", "100k", "текст", "мир",
    "42", "aaaaaaaa", "aaaaaaaaa", "ǆungla", "i", "ı",
    "हिन", "दी", "ह", "ⓐⓑⓒ", "مَرْحَبًا", "م",
    "οδος", "οδοσ", "σα", "ςα", "σ", "ς", "ασς", "ασσ", "ας1", "ασ1",
    "α\u02b0ς", "\u02b0σ", "\u02b0ς", "ας\u0345", "ασ\u02b0β", "ας", "β",
    "i\u0307stanbul", "ii\u0307", "ii", "i\u0307", "ⓐς", "ⓐσ",
]

def _probe() -> str | None:
    """Run the self-check once per process; cache and return its verdict."""
    global _PROBED, _SELFCHECK

    if _PROBED:
        return _SELFCHECK
    with _INIT:
        if _PROBED:
            return _SELFCHECK
        _SELFCHECK = _self_check()
        _PROBED = True
        if _SELFCHECK is not None:
            logger.warning(
                "native tokenizer disabled: %s. The Arrow pipeline computes "
                "the same masks, more slowly.", _SELFCHECK,
            )
    return _SELFCHECK

def _self_check() -> str | None:
    """Compare the native scanner's output against the Arrow reference."""
    global _SELFCHECK

    vocab = sorted(set(_SELFCHECK_VOCAB))
    arr = pa.array(_SELFCHECK_ROWS, type=pa.large_string())

    try:
        v = _ns.ScanVocab([t.encode() for t in vocab])
    except Exception as exc:
        return f"self-check vocabulary refused: {type(exc).__name__}: {exc}"

    got = np.zeros((len(vocab), len(arr)), dtype=bool)
    try:
        scan_into(arr, v, got)
    except Exception as exc:
        return f"self-check scan raised {type(exc).__name__}: {exc}"

    try:
        want = arrow_grid(arr, vocab)
    except Exception as exc:
        return f"self-check Arrow reference raised {type(exc).__name__}: {exc}"
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)
        return (f"self-check disagrees with Arrow at {len(bad)} cell(s), "
                f"first {tuple(bad[0])} (token {vocab[bad[0][0]]!r}, "
                f"row {_SELFCHECK_ROWS[bad[0][1]]!r})")

    return None

def arrow_grid(arr, vocab) -> np.ndarray:
    """Return the reference Arrow `(n_tokens, n_rows)` match grid."""
    toks = pc.split_pattern_regex(arr, pattern=TOKEN_SPLIT_PATTERN)
    lowered = lower_tokens(pc.list_flatten(toks))
    parent = pc.list_parent_indices(toks)
    codes = pc.index_in(lowered, value_set=pa.array(vocab, type=pa.large_string()))
    valid = pc.is_valid(codes)
    c = pc.filter(codes, valid).to_numpy(zero_copy_only=False)
    r = pc.filter(parent, valid).to_numpy(zero_copy_only=False)

    out = np.zeros((len(vocab), len(arr)), dtype=bool)
    out[c, r] = True
    return out


def prepare(tokens):
    """Build the native vocabulary, or return `None` to use Arrow."""
    if os.environ.get("NOVA_BF_NO_NATIVE_TOKENIZER"):
        return None
    if _ns is None or _tables() is None or _probe() is not None:
        return None

    try:
        return _ns.ScanVocab([t.encode("utf-8") for t in tokens])
    except Exception as exc:
        # Unsupported vocabularies fall back to the Arrow path.
        logger.debug("native tokenizer declines this vocabulary: %s", exc)
        return None

def _buffers(arr):
    """Return `(values, offsets, valid)` buffers for a string array.

    String bytes are viewed zero-copy; offsets are rebased to the selected
    slice so the scanner can handle sliced arrays directly.
    """
    n = len(arr)
    bufs = arr.buffers()
    if bufs[1] is None or bufs[2] is None:
        return None

    off_dtype = np.int64 if pa.types.is_large_string(arr.type) else np.int32
    offs = np.frombuffer(bufs[1], dtype=off_dtype)[
        arr.offset : arr.offset + n + 1
    ].astype(np.int64, copy=True)

    base = int(offs[0])
    end = int(offs[-1])
    vals = np.frombuffer(bufs[2], dtype=np.uint8)[base:end]
    offs -= base

    valid = None
    if arr.null_count:
        valid = pc.is_valid(arr).to_numpy(zero_copy_only=False).astype(bool, copy=False)
        valid = np.ascontiguousarray(valid)

    return vals, offs, valid


def _as_array(chunk):
    """Collapse a `ChunkedArray` to a single Arrow `Array`."""
    if isinstance(chunk, pa.ChunkedArray):
        chunk = chunk.combine_chunks()

    if isinstance(chunk, pa.ChunkedArray):
        if chunk.num_chunks == 0:
            return pa.array([], type=chunk.type)
        chunk = (
            chunk.chunk(0)
            if chunk.num_chunks == 1
            else pa.concat_arrays(chunk.chunks)
        )

    return chunk

def scan_into(chunk, vocab, grid) -> None:
    """Fill the preallocated `(n_tokens, n_rows)` match grid in place."""
    arr = _as_array(chunk)
    n = len(arr)
    if n == 0:
        return

    got = _buffers(arr)
    if got is None:
        # No value buffer means there is nothing to match.
        return

    vals, offs, valid = got
    _ns.scan_into(vals, offs, valid, _tables(), vocab, grid)


def tokens_of(text: str):
    """Return the scanner's lowered tokens for testing."""
    t = _tables()
    if t is None:
        return None

    return [
        b.decode("utf-8", "replace")
        for b in _ns.tokens_of(text.encode("utf-8"), t)
    ]