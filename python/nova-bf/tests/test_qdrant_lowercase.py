"""Qdrant's per-token `str::to_lowercase`: `İ` expansion and Final_Sigma.

Qdrant 1.18.2 (commit 44ad62f8) tokenizes `word` text as
`text.split(|c| !char::is_alphanumeric(c))` (tokenizers/mod.rs:43) and
lowercases each token with `token_cow.to_lowercase()`
(tokenizers/tokens_processor.rs:84). Beyond Arrow's one-codepoint
`utf8_lower` that adds exactly two rules (`nova_bf.tokenize` docstring); this
file pins them on all three implementations:

* `tokenize.qdrant_lower_token` / `lower_tokens` (the Arrow path, used by the
  query side and by the corpus fallback in `filters._token_row_masks`);
* the native scanner (`nativetok`, Rust), when built;
* an independent oracle: Rust's own `str::to_lowercase`, compiled from a
  10-line program when `rustc` is on PATH (skipped otherwise). The live
  Qdrant check is in tests/parity/test_parity_tokenizer.py.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from nova_bf import nativetok
from nova_bf.tokenizer_tables.qdrant_word_chars import WORD_CHAR_RANGES
from nova_bf.tokenizer_tables.rust_case_props import CASE_IGNORABLE_RANGES, CASED_RANGES
from nova_bf.tokenize import TOKEN_SPLIT_PATTERN, tokenize, tokenize_many

# Built = importable with the required features. Token tests then run against
# the Rust code directly (`tokens_of` does not consult the self-check), and
# `test_a_built_extension_passes_its_self_check` makes a failing self-check (which
# would silently route everything to Arrow) a test failure.
BUILT = nativetok._ns is not None and nativetok._missing_features(nativetok._ns) is None
NATIVE = BUILT
needs_native = pytest.mark.skipif(
    not BUILT, reason=f"nova_textscan unavailable: {nativetok.unavailable_reason()}")

# text -> Qdrant/Rust tokens (each measured on Qdrant 1.18.2, 2026-11-11).
CASES = [
    ("ΟΔΟΣ", ["οδος"]), ("ΟΔΟΣΑ", ["οδοσα"]), ("Σ", ["σ"]), ("ΑΣ", ["ας"]),
    ("ΣΑ", ["σα"]), ("ΑΣ'Β", ["ας", "β"]), ("Α'Σ", ["α", "σ"]), ("ΑΣ.Β", ["ας", "β"]),
    ("ᾼΣ", ["ᾳς"]), ("ΑΣͅ", ["αςͅ"]), ("ͅΣ", ["ͅσ"]),
    ("ΑʰΣ", ["αʰς"]), ("ʰΣ", ["ʰσ"]), ("ΑΣʰΒ", ["ασʰβ"]),
    ("ΑΣʰ", ["αςʰ"]), ("1Σ", ["1σ"]), ("Α1Σ", ["α1σ"]), ("ΑΣ1", ["ας1"]),
    ("ΑΣ1Β", ["ας1β"]), ("ΑΣΣ", ["ασς"]), ("ΑΣΣΑ", ["ασσα"]),
    ("ᾼΣͅΑ", ["ᾳσͅα"]), ("ΑΣ́", ["ας"]), ("ΆΣ", ["α", "σ"]),
    ("ⒶΣ", ["ⓐς"]), ("Ⓐ Σ", ["ⓐ", "σ"]), ("ǅΣ", ["ǆς"]), ("ΑσΒ", ["ασβ"]), ("Ας", ["ας"]),
    ("ς", ["ς"]), ("σ", ["σ"]), ("hello ΣΟΦΟΣ world", ["hello", "σοφος", "world"]),
    ("İ", ["i̇"]), ("İSTANBUL", ["i̇stanbul"]), ("İstanbul", ["i̇stanbul"]),
    ("iİ", ["ii̇"]), ("Aİ", ["ai̇"]), ("İΣ", ["i̇ς"]), ("ΑΣİ", ["ασi̇"]),
    ("i̇stanbul", ["i", "stanbul"]), ("istanbul", ["istanbul"]), ("ıstanbul", ["ıstanbul"]),
]


def _arrow_tokens(texts):
    with pytest.MonkeyPatch.context() as m:
        m.setenv("NOVA_BF_NO_NATIVE_TOKENIZER", "1")
        return tokenize_many(texts)


@pytest.mark.parametrize("text,want", CASES, ids=[c[0].encode("unicode_escape").decode()
                                                  for c in CASES])
def test_measured_cases_arrow_path(text, want):
    assert tokenize(text) == want


@needs_native
@pytest.mark.parametrize("text,want", CASES, ids=[c[0].encode("unicode_escape").decode()
                                                  for c in CASES])
def test_measured_cases_native(text, want):
    assert nativetok.tokens_of(text) == want


def _token_masks(rows, vocab, native: bool):
    from nova_bf.filters import _token_row_masks

    col = pa.chunked_array([pa.array(rows, pa.large_string())])
    with pytest.MonkeyPatch.context() as m:
        if native:
            m.delenv("NOVA_BF_NO_NATIVE_TOKENIZER", raising=False)
        else:
            m.setenv("NOVA_BF_NO_NATIVE_TOKENIZER", "1")
        g = _token_row_masks(col, set(vocab), len(rows))
    return np.stack([g.mask(t) for t in sorted(vocab)])


def _want_masks(rows, vocab, row_tokens):
    out = np.zeros((len(vocab), len(rows)), bool)
    for j, toks in enumerate(row_tokens):
        for i, t in enumerate(sorted(vocab)):
            out[i, j] = toks is not None and t in toks
    return out


def test_corpus_masks_follow_the_measured_cases_on_both_paths():
    """The corpus side (`filters._token_row_masks`, Arrow fallback and
    native) reaches exactly the measured tokens, and no simple-lowercase
    neighbour (`ασ` for `ας`, `istanbul` for `i̇stanbul`)."""
    rows = [t for t, _ in CASES]
    vocab = sorted({w for _, ws in CASES for w in ws}
                   | {"οδοσ", "ασ", "ασσ", "σοφοσ", "istanbul", "ii", "i", "ας1", "ασ1"})
    want = _want_masks(rows, vocab, [set(ws) for _, ws in CASES])
    assert np.array_equal(_token_masks(rows, vocab, native=False), want)
    if NATIVE:
        assert np.array_equal(_token_masks(rows, vocab, native=True), want)


@needs_native
def test_a_built_extension_passes_its_self_check(monkeypatch):
    monkeypatch.delenv("NOVA_BF_NO_NATIVE_TOKENIZER", raising=False)
    assert nativetok.unavailable_reason() is None


# -- random strings mixing the context classes --------------------------------

_CI_WORD = [0x0345, 0x02B0, 0x02B7, 0x1D2C, 0x05B0, 0x064E, 0x0E31, 0x093F]
_ALPH = (["Σ", "Σ", "Σ", "σ", "ς", "İ", "İ", "i", "ı", "Α", "Β", "α", "a", "Z", "1", "٣",
          "Ⓐ", "ǅ", "中", " ", "'", ".", "́", "̇", "-", "·"]
         + [chr(c) for c in _CI_WORD])


def _random_rows(seed, n=60):
    rng = random.Random(seed)
    return ["".join(rng.choice(_ALPH) for _ in range(rng.randint(0, 24))) for _ in range(n)]


@pytest.mark.parametrize("seed", range(30))
def test_random_mix_native_equals_arrow_path(seed):
    rows = _random_rows(seed)
    toks = _arrow_tokens(rows)
    vocab = sorted({t for ts in toks for t in ts}
                   | {t.replace("ς", "σ") for ts in toks for t in ts}
                   | {t.replace("σ", "ς") for ts in toks for t in ts})
    want = _want_masks(rows, vocab, [set(ts) for ts in toks])
    assert np.array_equal(_token_masks(rows, vocab, native=False), want)
    if NATIVE:
        for r, ts in zip(rows, toks):
            assert nativetok.tokens_of(r) == ts, r
        assert np.array_equal(_token_masks(rows, vocab, native=True), want)


# -- independent oracle: rustc's own to_lowercase -----------------------------

_RUST = r"""
use std::io::BufRead;
fn main() {
    for line in std::io::stdin().lock().lines() {
        let l = line.unwrap();
        // each token's to_lowercase as hex codepoints: "," within, " " between
        let v: Vec<String> = l.split('\u{1}').map(|t| t.to_lowercase().chars()
            .map(|c| format!("{:x}", c as u32)).collect::<Vec<_>>().join(",")).collect();
        println!("{}", v.join(" "));
    }
}
"""


@pytest.fixture(scope="module")
def rust_lower(tmp_path_factory):
    """Lowercase each token (as Qdrant does) with the local rustc's
    `str::to_lowercase`. Tokens of one row are joined by U+0001 (a separator,
    never inside a token)."""
    if shutil.which("rustc") is None:
        pytest.skip("rustc not on PATH")
    d = tmp_path_factory.mktemp("rs")
    (d / "l.rs").write_text(_RUST)
    subprocess.run(["rustc", "-O", "-o", str(d / "l"), str(d / "l.rs")], check=True)

    def run(token_lists):
        inp = "\n".join("\x01".join(ts) for ts in token_lists) + "\n"
        out = subprocess.run([str(d / "l")], input=inp, capture_output=True, text=True,
                             check=True).stdout.splitlines()
        res = []
        for line, ts in zip(out, token_lists):
            res.append(["".join(chr(int(h, 16)) for h in t.split(",") if h)
                        for t in line.split(" ")] if ts else [])
        assert len(res) == len(token_lists)
        return res
    return run


def _split_original(texts):
    toks = pc.split_pattern_regex(pa.array(texts, pa.large_string()), pattern=TOKEN_SPLIT_PATTERN)
    return [[t for t in ts if t] for ts in toks.to_pylist()]


def _context_strings():
    word = {c for lo, hi in WORD_CHAR_RANGES for c in range(lo, hi + 1)}
    ci = [c for lo, hi in CASE_IGNORABLE_RANGES for c in range(lo, hi + 1) if c in word]
    cased = [c for lo, hi in CASED_RANGES for c in range(lo, hi + 1) if c in word]
    rng = random.Random(11)
    other = rng.sample(sorted(word - set(ci) - set(cased)), 1500)
    out = []
    for c in ci + cased + other:
        ch = chr(c)
        out += ["Α" + ch + "Σ", ch + "Σ", "ΑΣ" + ch, "İ" + ch, ch + "İ"]
    return out


def test_every_case_context_matches_rust_to_lowercase(rust_lower):
    """Every Case_Ignorable and every Cased word character (and 1500 others)
    before and after a `Σ`, and next to an `İ`: the Arrow path and the native
    scanner give exactly Rust's per-token `to_lowercase`."""
    texts = _context_strings() + [t for t, _ in CASES] + [
        r for s in range(40) for r in _random_rows(500 + s)]
    want = rust_lower(_split_original(texts))
    got = _arrow_tokens(texts)
    bad = [(t, g, w) for t, g, w in zip(texts, got, want) if g != w]
    assert not bad, f"{len(bad)} Arrow-path mismatches, e.g. {bad[:5]}"
    if NATIVE:
        bad = [(t, w) for t, w in zip(texts, want) if nativetok.tokens_of(t) != w]
        assert not bad, f"{len(bad)} native mismatches, e.g. {bad[:5]}"


def test_case_tables_are_rusts(rust_lower):
    """`tokenizer_tables.rust_case_props` against rustc directly: `A c Σ` / `c Σ` / `AΣ c`
    end in final sigma exactly as the recorded classes predict."""
    ci = {c for lo, hi in CASE_IGNORABLE_RANGES for c in range(lo, hi + 1)}
    cased = {c for lo, hi in CASED_RANGES for c in range(lo, hi + 1)}
    rng = random.Random(5)
    cps = sorted(ci | cased | {rng.randrange(0x110000) for _ in range(20000)})
    cps = [c for c in cps if not 0xD800 <= c < 0xE000 and c not in (0x0A, 0x0D, 0x01)]
    rows = [[f"A{chr(c)}Σ", f"{chr(c)}Σ", f"AΣ{chr(c)}"] for c in cps]
    for c, (fa, fc, ff) in zip(cps, rust_lower(rows)):
        is_ci, is_cased = c in ci, c in cased
        assert fa.endswith("ς") == (is_ci or is_cased), hex(c)
        assert fc.endswith("ς") == (not is_ci and is_cased), hex(c)
        assert (ff[1] == "ς") == (is_ci or not is_cased), hex(c)


# -- an old extension is refused -----------------------------------------------

class _OldScanner:
    """Stands in for nova_textscan 0.0.12: no `features()`, two-table ScanTables."""

    def __init__(self, real):
        self._real = real

    def version(self):
        return "0.0.12"

    def __getattr__(self, name):
        if name == "features":
            raise AttributeError(name)
        return getattr(self._real, name)


@needs_native
def test_an_extension_without_the_lowercase_features_is_refused(monkeypatch, caplog):
    monkeypatch.delenv("NOVA_BF_NO_NATIVE_TOKENIZER", raising=False)
    monkeypatch.setattr(nativetok, "_ns", _OldScanner(nativetok._ns))
    for name, val in [("_TABLES", None), ("_TABLES_REASON", None), ("_PROBED", False),
                      ("_SELFCHECK", None)]:
        monkeypatch.setattr(nativetok, name, val)
    with caplog.at_level(logging.WARNING, logger="nova_bf.nativetok"):
        assert not nativetok.available()
        assert "too old" in nativetok.unavailable_reason()
        assert nativetok.prepare(["a"]) is None
        assert nativetok.prepare(["b"]) is None
    warns = [r for r in caplog.records if "too old" in r.getMessage()]
    assert len(warns) == 1, "the refusal is logged exactly once"
    # ... and the corpus path still answers (Arrow), with the exact rules.
    assert np.array_equal(_token_masks(["ΟΔΟΣ"], ["οδος", "οδοσ"], native=True),
                          np.array([[True], [False]]))


@needs_native
def test_a_partial_feature_list_is_refused(monkeypatch):
    class Partial(_OldScanner):
        def features(self):
            return ["final_sigma_in_token"]

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(nativetok, "_ns", Partial(nativetok._ns))
    for name, val in [("_TABLES", None), ("_TABLES_REASON", None), ("_PROBED", False),
                      ("_SELFCHECK", None)]:
        monkeypatch.setattr(nativetok, name, val)
    reason = nativetok.unavailable_reason()
    assert reason and "dotted_capital_i_expands" in reason


def test_required_features_are_what_the_crate_declares():
    src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "crates",
                       "nova-textscan", "src", "lib.rs")
    if not os.path.exists(src):
        pytest.skip("crate source not in this checkout")
    text = open(src).read()
    for f in nativetok.REQUIRED_FEATURES:
        assert f'"{f}"' in text, f


def test_rust_case_props_harness_copy_is_identical():
    here = os.path.dirname(__file__)
    ours = os.path.join(here, "..", "src", "nova_bf", "tokenizer_tables", "rust_case_props.py")
    theirs = os.path.join(here, "..", "..", "..", "benchmarks", "streaming", "harness",
                          "_rust_case_props.py")
    if not os.path.exists(theirs):
        pytest.skip("harness not in this checkout")
    assert open(ours, "rb").read() == open(theirs, "rb").read()
