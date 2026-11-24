"""DuckDB filter engine for the filtered-search baseline (DuckDB -> Arrow -> cuVS/FAISS).

Evaluates nova-bf's Qdrant-style filters on each corpus file and produces
packed row masks for static or per-query filters. DuckDB handles predicates
and text tokenization; NumPy combines the resulting bitmaps.

Masks must match nova-bf's CPU filtering behavior or raise an error.
Null values never match, date fields use UTC epoch microseconds, and
integer comparisons remain exact rather than reproducing NumPy coercion
errors. Unsupported date formats or ambiguous coercions raise errors.

Masks use Arrow's little-endian bit order, padded with zeros to whole bytes.
"""

from __future__ import annotations

import datetime
import decimal
import functools
import math
import time
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pyarrow as pa

from bench.data.filters.text_tables._qdrant_word_chars import WORD_CHAR_RANGES
from bench.data.filters.text_tables._rust_case_props import CASE_IGNORABLE_RANGES, CASED_RANGES

# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------

# Runs of non-word characters separate tokens. Word characters come from the codepoint table
# measured against Qdrant's `word` tokenizer (a byte-identical copy of nova-bf's
# `nova_bf/tokenizer_tables/qdrant_word_chars.py`), spelled out as an explicit class so
# DuckDB's and Arrow's bundled RE2 Unicode tables do not matter.
TOKEN_SPLIT = "[^" + "".join(
    f"\\x{{{lo:X}}}" if lo == hi else f"\\x{{{lo:X}}}-\\x{{{hi:X}}}"
    for lo, hi in WORD_CHAR_RANGES) + "]+"


@functools.lru_cache(maxsize=None)
def _lower_fix() -> tuple[str, str]:
    """(from, to) strings for `translate`: word characters whose DuckDB lower() differs from
    Arrow's utf8_lower (nova-bf's lowercasing). DuckDB bundles older case tables (DuckDB 1.5.3
    vs pyarrow 24: 55 Unicode 16 capitals, e.g. U+1C89, that DuckDB leaves unchanged).
    Derived per process from the installed libraries (~0.5 s), so upgrades need no edit."""
    import pyarrow.compute as pc

    word = [chr(c) for lo, hi in WORD_CHAR_RANGES for c in range(lo, hi + 1)
            if not (0xD800 <= c < 0xE000)]
    want = pc.utf8_lower(pa.array(word, pa.large_string())).to_pylist()
    con = duckdb.connect()
    try:
        con.register("__lfix", pa.table({"i": pa.array(range(len(word)), pa.int64()),
                                         "s": pa.array(word, pa.large_string())}))
        got = con.sql("SELECT lower(s) AS l FROM __lfix ORDER BY i").to_arrow_table() \
            .column("l").to_pylist()
    finally:
        con.close()
    pairs = [(c, w) for c, w, g in zip(word, want, got)
             if w != g and len(w) == 1 and len(g) == 1 and g == c]
    return "".join(c for c, _ in pairs), "".join(w for _, w in pairs)


# --- the two non-per-codepoint lowercase rules (as in nova-bf's tokenize.py) -------
#
# Rust's `to_lowercase`, which Qdrant applies to each token, is the per-codepoint map except:
# `İ` U+0130 -> `i` U+0307, and `Σ` U+03A3 -> `ς` iff Final_Sigma holds within the token
# (before it: Case_Ignorable* then Cased; after it: NOT Case_Ignorable* then Cased), else
# `σ`. The case property tables are derived from rustc.

SPECIAL_CHARS = "\u0130\u03a3"   # İ, Σ
_CI = frozenset(c for lo, hi in CASE_IGNORABLE_RANGES for c in range(lo, hi + 1))
_CASED = frozenset(c for lo, hi in CASED_RANGES for c in range(lo, hi + 1))


def _ci_then_cased(chars) -> bool:
    for ch in chars:
        if ord(ch) not in _CI:
            return ord(ch) in _CASED
    return False


def rust_lower_token(tok: str, simple: str) -> str:
    """Rust's `tok.to_lowercase()`, given `simple` = its per-codepoint lowercase."""
    assert len(simple) == len(tok)
    out = list(simple)
    for k, ch in enumerate(tok):
        if ch == "\u03a3":
            out[k] = ("\u03c2" if _ci_then_cased(reversed(tok[:k]))
                      and not _ci_then_cased(tok[k + 1:]) else "\u03c3")
        elif ch == "\u0130":
            out[k] = "i\u0307"
    return "".join(out)


def _class(cps) -> str:
    rng = []
    for c in sorted(cps):
        if rng and rng[-1][1] == c - 1:
            rng[-1][1] = c
        else:
            rng.append([c, c])
    return "".join(f"\\x{{{lo:X}}}" if lo == hi else f"\\x{{{lo:X}}}-\\x{{{hi:X}}}"
                   for lo, hi in rng)


@functools.lru_cache(maxsize=None)
def _final_sigma_re() -> str:
    """RE2 pattern whose global replacement by `\\1\\2ς\\3\\4` turns exactly the word-final
    `Σ`s of one token into `ς`: (Cased)(Case_Ignorable*)Σ(Case_Ignorable*)(end | a word char
    that is neither). Matches never need to overlap: a word-final Σ is never followed by
    Case_Ignorable* then Σ, so no match consumes the Cased char a later Σ needs."""
    word = {c for lo, hi in WORD_CHAR_RANGES for c in range(lo, hi + 1)}
    ci, cased = _class(_CI & word), _class(_CASED & word)
    return f"([{cased}])([{ci}]*)\u03a3([{ci}]*)($|[^{cased}{ci}])"


def _special_token_sql(tok: str) -> str:
    """Lowercase one TOKEN `tok` (SQL expr) that may hold İ / Σ."""
    fixed = (f"replace(regexp_replace({tok}, '{_final_sigma_re()}', "
             f"'\\1\\2\u03c2\\3\\4', 'g'), '\u0130', 'i\u0307')")
    return lower_sql(fixed, regex_guard=True)   # `fixed` is long: name it once


def _special_tokens_sql(expr: str) -> str:
    """list<varchar> tokens of `expr`: split the ORIGINAL text, then lowercase
    each token with the two rules (`_special_token_sql`)."""
    return (f"list_transform(list_filter(regexp_split_to_array({expr}, '{TOKEN_SPLIT}'), "
            f"x -> x <> ''), x -> {_special_token_sql('x')})")


# Rare-character guards use literal substring searches instead of a regex class: same truth
# value, far cheaper (90 MB FineWeb file, 1 thread: 203 -> 5 ms for `İ`/`Σ`, 203 -> 60 ms for
# the 55 `_lower_fix` characters). Above this many characters, a regex class is used.
GUARD_CONTAINS_MAX = 128


def _has_any_sql(expr: str, chars: str, regex_guard: bool = False) -> str:
    """SQL boolean: `expr` contains at least one of `chars` (codepoints)."""
    if regex_guard or len(chars) > GUARD_CONTAINS_MAX:
        return (f"regexp_matches({expr}, '["
                + "".join(f"\\x{{{ord(c):X}}}" for c in chars) + "]')")
    return "(" + " OR ".join(f"contains({expr}, {_str_literal(c)})" for c in chars) + ")"


def lower_sql(expr: str, regex_guard: bool = False) -> str:
    """SQL lowercasing equal to Arrow's utf8_lower on every word character: DuckDB's lower(),
    plus `translate` of the codepoints DuckDB misses, only on rows holding one (translate
    costs ~10x lower()). `regex_guard`: guard with one regex class (for a long `expr`)."""
    src, dst = _lower_fix()
    if not src:
        return f"lower({expr})"
    return (f"CASE WHEN {_has_any_sql(expr, src, regex_guard)} "
            f"THEN translate(lower({expr}), '{src}', '{dst}') ELSE lower({expr}) END")


def _fold(w: str) -> str:
    return w.replace("\u03c2", "\u03c3").replace("\u0307", "")


def _fold_sql(expr: str) -> str:
    return f"translate({expr}, '\u03c2\u0307', '\u03c3')"


def tokens_sql(expr: str) -> str:
    """SQL list<varchar> of `expr`'s tokens, drop ''.

    nova-bf splits, then lowercases. Lowercasing first gives the same tokens when lower() is
    per-codepoint, context-free, length-preserving and keeps each character's word class
    (all verified by the self-checks), and is ~12% cheaper. Rows holding `İ`/`Σ` (context- or
    length-changing) take `_special_tokens_sql`."""
    return _tokens_sql(expr, lower_sql(expr))


def _tokens_sql(expr: str, lowered: str, lean: bool = False) -> str:
    """`tokens_sql` given `lowered` = `lower_sql(expr)`, or with `lean` plain `lower(expr)`
    (then only rows holding `İ` take `_special_tokens_sql`)."""
    fast = f"list_filter(regexp_split_to_array({lowered}, '{TOKEN_SPLIT}'), x -> x <> '')"
    return (f"CASE WHEN {_has_any_sql(expr, LEAN_CHARS if lean else SPECIAL_CHARS)} "
            f"THEN {_special_tokens_sql(expr)} ELSE {fast} END")


# Lean tokenisation, for an all-ASCII query vocabulary: plain lower(), with only rows holding
# `İ` sent through the exact path. It differs from the full SQL only at a `Σ` or a
# `_lower_fix` capital, where both spell a non-ASCII character, and token boundaries are the
# same; so an ASCII word matches the same rows. `İ` stays guarded because plain lower() turns
# it into ASCII `i` where nova-bf has `i` U+0307.
LEAN_CHARS = "\u0130"


def lean_tokens_ok(words) -> bool:
    """True when lean tokenisation is exact for `words`: all ASCII, and no `_lower_fix`
    character is ASCII."""
    src, dst = _lower_fix()
    return all(w.isascii() for w in words) and not any(c.isascii() for c in src + dst)


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# match_text implementations (`DuckFilter(text_impl=...)`).
#   bitmap  all in DuckDB: one bitmap per needed word (bitstring_agg over the
#           row ids), each phrase the bit_and of its words' bitmaps. Python
#           only repacks DuckDB's BIT values into the harness's bit order.
# The tests compare it with an independent implementation that overrides _phrase_masks
# (tests/data/filters/ref_text_numpy.py).
TEXT_IMPLS = ("bitmap",)

# DuckDB BIT -> harness packing: byte value -> the same byte bit-reversed
_BIT_REV = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], np.uint8)


def bit_blob_to_packed(data: np.ndarray, n: int) -> np.ndarray:
    """(P, 1 + nbytes) DuckDB BIT values of n bits (a padding-count byte p, then MSB-first
    bits with p leading padding bits) -> (P, nbytes) little bit order, padding 0. Raises on
    any other layout."""
    nb = (n + 7) // 8
    if data.shape[0] == 0:
        return np.zeros((0, nb), np.uint8)
    p = int(data[0, 0])
    if data.shape[1] != nb + 1 or not (data[:, 0] == p).all() or p != nb * 8 - n:
        raise RuntimeError(f"unexpected DuckDB BIT layout for {n} bits "
                           f"(padding {sorted(set(data[:, 0].tolist()))[:4]}, "
                           f"width {data.shape[1]})")
    rev = _BIT_REV[data[:, 1:]]         # bitstream position j -> bit j%8 of byte j//8
    if p:
        nxt = np.zeros_like(rev)
        nxt[:, :-1] = rev[:, 1:]
        rev = ((rev >> p) | (nxt << (8 - p))).astype(np.uint8)
    if n % 8:
        rev[:, -1] &= (1 << (n % 8)) - 1
    return rev


def _fetch(con, sql: str) -> pa.Table:
    """Run a SELECT and return its result as an Arrow table.

    Uses the relation API, whose Arrow export is parallel; `con.execute(...).arrow()` and
    the fetch_* calls export serially (1M rows x 199 bool columns: 0.73 s at any thread count,
    vs 0.66 / 0.21 / 0.16 s at 1 / 4 / 8 threads here)."""
    return con.sql(sql).to_arrow_table()


def tokenize_many(con, values) -> list[list[str] | None]:
    """Tokenise Python strings with the corpus side's SQL, so tokens agree. Non-strings -> None."""
    clean = [v if isinstance(v, str) else None for v in values]
    tbl = pa.table({"i": pa.array(range(len(clean)), pa.int64()),
                    "s": pa.array(clean, pa.large_string())})
    con.register("__tok_in", tbl)
    try:
        out = _fetch(con, (f"SELECT {tokens_sql('s')} AS t FROM __tok_in ORDER BY i"))
    finally:
        con.unregister("__tok_in")
    return out.column("t").to_pylist()


def selfcheck_tokenizer(con, reference=None) -> list[int]:
    """Codepoints where DuckDB's tokeniser disagrees with `reference` (default: nova-bf's
    tokenizer built from pyarrow). Empty = exact. ~3 s; run at startup since each library
    bundles its own Unicode tables. Checks `a<cp>b` for every codepoint, plus Final_Sigma
    contexts (`Α c Σ`, `c Σ`, `ΑΣ c`, `İ c`) for every Case_Ignorable / Cased word char c."""
    import pyarrow.compute as pc

    cps = [c for c in range(0x110000) if not (0xD800 <= c < 0xE000)]
    vals = ["a" + chr(c) + "b" for c in cps]
    if reference is None:
        def reference(vs):
            toks = pc.split_pattern_regex(pa.array(vs, pa.large_string()),
                                          pattern=TOKEN_SPLIT)
            flat = pc.list_flatten(toks)
            orig = flat.to_pylist()
            low = pc.utf8_lower(flat).to_pylist()
            low = [rust_lower_token(o, s) if ("\u03a3" in o or "\u0130" in o) else s
                   for o, s in zip(orig, low)]
            lens = pc.list_value_length(toks).to_pylist()
            out, i = [], 0
            for n in lens:
                out.append([t for t in low[i:i + n] if t])
                i += n
            return out
    bad = []
    for b0 in range(0, len(vals), _SELFCHECK_CHUNK):  # chunked: bounded memory
        chunk = vals[b0:b0 + _SELFCHECK_CHUNK]
        ref, got = reference(chunk), tokenize_many(con, chunk)
        bad += [cps[b0 + i] for i in range(len(chunk)) if ref[i] != got[i]]
    word = {c for lo, hi in WORD_CHAR_RANGES for c in range(lo, hi + 1)}
    ctx = sorted((_CI | _CASED) & word)
    vals = [v for c in ctx for v in ("\u0391" + chr(c) + "\u03a3", chr(c) + "\u03a3",
                                     "\u0391\u03a3" + chr(c), "\u0130" + chr(c))]
    ref, got = reference(vals), tokenize_many(con, vals)
    bad += sorted({ctx[i // 4] for i in range(len(vals)) if ref[i] != got[i]} - set(bad))
    return bad


def selfcheck_lower(con) -> list[int]:
    """Codepoints where DuckDB's lower() is context-dependent or changes length (either would
    make `_text_masks`' substring pre-check unsound). Empty = sound. Each codepoint is lowered
    alone and next to capital sigma."""
    la, ls = con.execute("SELECT lower('A'), lower(chr(931))").fetchone()
    all_cps = [c for c in range(0x110000) if not (0xD800 <= c < 0xE000)]
    bad = []
    for b0 in range(0, len(all_cps), _SELFCHECK_CHUNK):  # chunked: bounded memory
        cps = all_cps[b0:b0 + _SELFCHECK_CHUNK]
        con.register("__lc", pa.table({"c": pa.array(cps, pa.int32()),
                                       "s": ["A" + chr(c) + "\u03a3" + chr(c) + "\u03a3 " for c in cps],
                                       "one": [chr(c) for c in cps]}))
        try:
            rows = con.execute("SELECT c, lower(s), lower(one) FROM __lc ORDER BY c").fetchall()
        finally:
            con.unregister("__lc")
        bad += [c for c, w, lo in rows if len(lo) != 1 or w != la + lo + ls + lo + ls + " "]
    return bad


_SELFCHECK_CHUNK = 1 << 16
_SELFCHECKED: set[str] = set()


def ensure_selfchecks(con) -> None:
    """Run both Unicode self-checks once per process (~5 s); raise if this DuckDB / pyarrow
    pair breaks tokeniser parity or the pre-check's soundness."""
    key = f"{duckdb.__version__}/{pa.__version__}"
    if key in _SELFCHECKED:
        return
    tok = selfcheck_tokenizer(con)
    if tok:
        raise RuntimeError(f"DuckDB tokeniser disagrees with Arrow on {len(tok)} codepoints, "
                           f"e.g. {[hex(c) for c in tok[:5]]}; update TOKEN_SPLIT")
    low = selfcheck_lower(con)
    if low:
        raise RuntimeError(f"DuckDB lower() is context-dependent/length-changing on {len(low)} "
                           f"codepoints, e.g. {[hex(c) for c in low[:5]]}: substring pre-check unsound")
    _SELFCHECKED.add(key)


# ---------------------------------------------------------------------------
# filter spec (the harness's own copy of nova-bf's Filter shape)
# ---------------------------------------------------------------------------

KINDS = ("match", "range", "match_text",
         "match_from_query", "range_from_query", "match_text_from_query")
RANGE_OPS = ("gt", "gte", "lt", "lte")
SQL_OP = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
GROUPS = ("must", "should", "must_not")


@dataclass(frozen=True)
class Cond:
    field: str
    kind: str
    # match: scalar or tuple; range: tuple of (op, float); match_text: str;
    # match_from_query / match_text_from_query: query column; range_from_query:
    # tuple of (op, query column)
    value: object

    @property
    def per_query(self) -> bool:
        return self.kind.endswith("_from_query")

    @property
    def text(self) -> bool:
        return self.kind in ("match_text", "match_text_from_query")

    def query_columns(self) -> list[str]:
        if self.kind == "range_from_query":
            return [c for _, c in self.value]
        if self.per_query:
            return [self.value]
        return []


@dataclass(frozen=True)
class FilterSpec:
    must: tuple[Cond, ...] = ()
    should: tuple[Cond, ...] = ()
    must_not: tuple[Cond, ...] = ()

    def conds(self) -> tuple[Cond, ...]:
        return (*self.must, *self.should, *self.must_not)

    def fields(self) -> list[str]:
        return sorted({c.field for c in self.conds()})

    def query_columns(self) -> list[str]:
        return sorted({q for c in self.conds() for q in c.query_columns()})

    @property
    def per_query(self) -> bool:
        return any(c.per_query for c in self.conds())


def parse_filter(d: dict | None, date_fields: dict[str, str] | None = None) -> FilterSpec:
    """A nova-bf `filter:` mapping -> FilterSpec. Date-string range bounds on declared date
    fields become epoch microseconds, as nova-bf does at config load."""
    if not d:
        return FilterSpec()
    unknown = set(d) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown filter keys {sorted(unknown)}")
    date_fields = normalize_date_fields(date_fields)
    groups = {}
    for g in GROUPS:
        conds = []
        for raw in d.get(g) or ():
            raw = dict(raw)
            fld = raw.pop("field")
            if len(raw) != 1 or next(iter(raw)) not in KINDS:
                raise ValueError(f"condition on {fld!r} must set exactly one of {KINDS}, got {sorted(raw)}")
            kind, val = next(iter(raw.items()))
            if kind == "match":
                # nova-bf's config turns a Decimal match value into a float; so here too
                def _mv(x):
                    return float(x) if isinstance(x, decimal.Decimal) else x
                val = tuple(_mv(x) for x in val) if isinstance(val, (list, tuple)) else _mv(val)
            elif kind in ("range", "range_from_query"):
                bad = set(val) - set(RANGE_OPS)
                if bad or not val:
                    raise ValueError(f"range on {fld!r}: bounds must be a non-empty subset of {RANGE_OPS}")
                if kind == "range":
                    fmt = date_fields.get(fld)
                    val = tuple((op, _static_bound(val[op], fmt)) for op in RANGE_OPS if val.get(op) is not None)
                else:
                    val = tuple((op, val[op]) for op in RANGE_OPS if val.get(op) is not None)
            conds.append(Cond(fld, kind, val))
        groups[g] = tuple(conds)
    return FilterSpec(**groups)


_EPOCH_MULT = {"epoch_s": 1_000_000, "epoch_ms": 1_000, "epoch_us": 1}


def _static_bound(v, fmt: str | None) -> float:
    """A static range bound -> float, by a copy of nova-bf's config-load parser (not DuckDB,
    whose parsing can differ by 1 us on fractional seconds). Strings are parsed; numbers are
    rescaled only for epoch_s / epoch_ms."""
    if isinstance(v, str) and fmt is None:
        raise ValueError(f"string range bound {v!r} on a field not declared in date_fields")
    if fmt is None or not (isinstance(v, str) or fmt in _EPOCH_MULT):
        return float(v)
    from datetime import datetime, timezone

    if fmt in _EPOCH_MULT:
        if _EPOCH_MULT[fmt] == 1:
            return float(int(v))
        return float(int(round(float(v) * _EPOCH_MULT[fmt])))
    s = str(v)
    dt = (datetime.fromisoformat(s.replace("Z", "+00:00")) if fmt == "rfc3339"
          else datetime.strptime(s, fmt))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(int(dt.timestamp() * 1_000_000))


def normalize_date_fields(raw) -> dict[str, str]:
    """`list | dict | None` -> {field: format}; list entries and empty formats mean rfc3339,
    as in nova-bf."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {k: (v or "rfc3339") for k, v in raw.items()}
    return {k: "rfc3339" for k in raw}


def _date_expr(expr: str, fmt: str, typ: pa.DataType | None = None) -> str:
    """SQL: a declared date column / literal -> BIGINT epoch microseconds, matching nova-bf.
    `typ`: the column's Arrow type if known (an int column declared rfc3339 is already epoch
    us, as Arrow treats it)."""
    if fmt == "rfc3339":
        if typ is not None and pa.types.is_integer(_base_type(typ)):
            return f"CAST({expr} AS BIGINT)"
        return f"epoch_us(CAST({expr} AS TIMESTAMPTZ))"
    if fmt == "epoch_us":
        return f"CAST({expr} AS BIGINT)"
    if fmt in ("epoch_s", "epoch_ms"):
        if typ is not None and pa.types.is_decimal(_base_type(typ)):
            # nova-bf rescales through Arrow's (not correctly rounded) decimal
            # -> float64; DuckDB's cast differs at half-microsecond boundaries.
            raise NotImplementedError(f"{fmt} date field stored as {typ}")
        mult = 1_000_000 if fmt == "epoch_s" else 1_000
        # DOUBLE -> BIGINT rounds half-to-even like Arrow's `round`; DuckDB's
        # round_even() itself is wrong on exact halves near 1e15 (epoch us!).
        return f"CAST(CAST({expr} AS DOUBLE) * {mult} AS BIGINT)"
    if "%" in fmt:
        return f"epoch_us(strptime({expr}, {sql_literal(fmt)}))"
    raise ValueError(f"unsupported date format {fmt!r}")


# ---------------------------------------------------------------------------
# literals
# ---------------------------------------------------------------------------

def _is_nan(v) -> bool:
    return isinstance(v, (float, np.floating)) and math.isnan(v)


def _str_literal(s: str) -> str:
    # A NUL cannot appear inside SQL text; splice it in with chr(0).
    parts = ["'" + p.replace("'", "''") + "'" for p in s.split("\x00")]
    return " || chr(0) || ".join(parts)


def sql_literal(v, as_float32: bool = False) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        # String -> DOUBLE parsing is correctly rounded; a bare 0.123 literal
        # would be a DECIMAL first. A float32 value is exactly representable
        # as a double, so DOUBLE -> FLOAT is exact (no double rounding).
        f = float(v)
        s = "NaN" if math.isnan(f) else ("Infinity" if f > 0 else "-Infinity") if math.isinf(f) else repr(f)
        lit = f"CAST('{s}' AS DOUBLE)"
        return f"CAST({lit} AS FLOAT)" if as_float32 else lit
    if isinstance(v, str):
        return _str_literal(v)
    if isinstance(v, decimal.Decimal):
        s = format(v, "f")
        intpart, _, frac = s.lstrip("-").partition(".")
        scale = len(frac)
        prec = max(1, scale, len(intpart.lstrip("0")) + scale)   # DECIMAL(38,38) is valid
        if prec > 38:
            raise TypeError(f"decimal literal {v} exceeds DECIMAL(38)")
        return f"CAST('{s}' AS DECIMAL({prec}, {scale}))"
    if isinstance(v, datetime.datetime):
        kind = "TIMESTAMPTZ" if v.tzinfo is not None else "TIMESTAMP"
        return f"{kind} '{v.isoformat(sep=' ')}'"
    if isinstance(v, datetime.date):
        return f"DATE '{v.isoformat()}'"
    if isinstance(v, (bytes, bytearray)):
        return f"from_hex('{bytes(v).hex()}')"
    raise TypeError(f"unsupported literal {v!r} ({type(v).__name__})")


# ---------------------------------------------------------------------------
# type-aware equality (the static vs per-query paths of nova-bf differ)
# ---------------------------------------------------------------------------

def _base_type(t: pa.DataType) -> pa.DataType:
    return t.value_type if pa.types.is_dictionary(t) else t


def _family(t: pa.DataType) -> str:
    t = _base_type(t)
    if pa.types.is_boolean(t) or pa.types.is_integer(t) or pa.types.is_floating(t):
        return "num"
    if pa.types.is_decimal(t):
        return "dec"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or \
            getattr(pa.types, "is_string_view", lambda _: False)(t):
        return "str"
    if pa.types.is_temporal(t):
        return "time"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_fixed_size_binary(t):
        return "bin"
    return "other"


def _value_family(v) -> str:
    if isinstance(v, (bool, np.bool_, int, np.integer, float, np.floating)):
        return "num"
    if isinstance(v, decimal.Decimal):
        return "dec"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (datetime.datetime, datetime.date)):
        return "time"
    if isinstance(v, (bytes, bytearray)):
        return "bin"
    return "other"


def _supported_type(t: pa.DataType, declared_date_fmt: str | None = None,
                    allow_list: bool = False) -> bool:
    """Allow-list of supported column types (corpus fields and per-query value columns).
    Anything else is refused up front: nova-bf's results on exotic types (tz/ns timestamps,
    date64, decimal256, negative-scale decimals, float16, string/binary views) follow
    numpy/Arrow coercion accidents, not defined rules."""
    if pa.types.is_dictionary(t):
        t = t.value_type
        if pa.types.is_dictionary(t):
            return False
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        # lists only as per-query MatchAny value columns (queries side), never
        # as corpus fields (nova-bf's numpy compare of list cells is accidental)
        if declared_date_fmt is not None or not allow_list:
            return False
        return _supported_type(t.value_type)            # MatchAny value lists
    if declared_date_fmt is not None:
        if declared_date_fmt in ("rfc3339",) or "%" in declared_date_fmt:
            return (pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_null(t)
                    or (declared_date_fmt == "rfc3339" and (pa.types.is_int64(t)
                        or (pa.types.is_timestamp(t) and t.unit == "us") or pa.types.is_date32(t))))
        return pa.types.is_integer(t) or pa.types.is_float32(t) or pa.types.is_float64(t) \
            or pa.types.is_null(t)                   # epoch_s / epoch_ms / epoch_us
    return (pa.types.is_string(t) or pa.types.is_large_string(t)
            or pa.types.is_binary(t) or pa.types.is_large_binary(t)
            or pa.types.is_integer(t) or pa.types.is_float32(t) or pa.types.is_float64(t)
            or pa.types.is_boolean(t) or pa.types.is_null(t)
            or (pa.types.is_decimal128(t) and t.scale >= 0)
            or (pa.types.is_timestamp(t) and t.unit == "us")
            or pa.types.is_date32(t))


def _check_supported(table: pa.Table, cols, dates: dict[str, str], side: str) -> None:
    for c in cols:
        t = table.schema.field(c).type
        if not _supported_type(t, dates.get(c), allow_list=(side == "queries")):
            what = f" declared {dates[c]}" if c in dates else ""
            raise NotImplementedError(
                f"{side} column {c!r} of type {t}{what} is not supported by the DuckDB "
                f"baseline (see `_supported_type`)")


def _static_match_sql(col: str, t: pa.DataType, values, col_f64: str | None = None) -> str:
    """nova-bf's static `match` is Arrow `pc.is_in(col, pa.array(values))`.

    `col_f64` names a column holding Arrow's own float64 conversion of a decimal column
    (Arrow's decimal -> double is not correctly rounded, so it must be Arrow's).

    When every value casts to the column type, Arrow compares in that type: values are
    safe-cast (uncastable ones dropped) and floats compare bit patterns (NaN matches NaN,
    -0.0 does not match 0.0). Integer values on a decimal column compare exactly. When some
    value fails the cast, a float value set compares as float64 bit patterns against the
    column cast to double."""
    import pyarrow.compute as pc

    bt = _base_type(t)
    if pa.types.is_null(bt):
        return "FALSE"
    if pa.types.is_duration(bt) or pa.types.is_time(bt) or pa.types.is_interval(bt):
        raise NotImplementedError(f"match on a {bt} column is not supported")
    if pa.types.is_integer(bt):
        ints = _int_match_values(values, bt)
        if ints is not None:
            return f"{col} IN ({', '.join(str(v) for v in ints)})" if ints else "FALSE"
    vals = pa.array(list(values))  # same inference as nova-bf; raises on mixed types like it
    if pa.types.is_decimal(bt) and pa.types.is_integer(vals.type):
        # is_in compares an integer value set with a decimal column through a
        # common (wider) decimal type, i.e. exactly -- a cast into the column's
        # own type would wrongly drop values Arrow keeps.
        lits = []
        for v in vals.to_pylist():
            if v is None:
                continue
            try:
                lits.append(sql_literal(decimal.Decimal(v)))
            except TypeError:
                continue            # wider than DECIMAL(38): equals no value
        return f"{col} IN ({', '.join(lits)})" if lits else "FALSE"
    casted, failed = [], False
    for i in range(len(vals)):
        if not vals[i].is_valid:
            continue
        try:
            casted.append(pc.cast(vals.slice(i, 1), bt, safe=True))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            failed = True

    def bit_eq(expr: str, v: float, f32: bool) -> str:
        if math.isnan(v):
            return f"isnan({expr})"
        return (f"({expr} = {sql_literal(v, as_float32=f32)} AND "
                f"signbit({expr}) = {'TRUE' if math.copysign(1, v) < 0 else 'FALSE'})")

    if failed and pa.types.is_floating(vals.type) and \
            (pa.types.is_integer(bt) or pa.types.is_decimal(bt) or pa.types.is_boolean(bt)):
        if pa.types.is_decimal(bt):
            if col_f64 is None:
                raise NotImplementedError("decimal match fallback without Arrow's float64 column")
            expr = col_f64
        else:
            expr = f"CAST({col} AS DOUBLE)"
        parts = [bit_eq(expr, float(v), False) for v in vals.to_pylist() if v is not None]
        return "(" + " OR ".join(parts) + ")" if parts else "FALSE"
    if failed and pa.types.is_integer(vals.type) and pa.types.is_floating(bt):
        # Integer value set that does not cast into a float column (beyond
        # +-2**24 / +-2**53): Arrow instead casts the COLUMN to int64 and
        # compares exactly (it raises if the column holds fractional/NaN/inf,
        # where the engine is free).
        ints = [str(v) for v in vals.to_pylist() if v is not None]
        return f"TRY_CAST({col} AS HUGEINT) IN ({', '.join(ints)})" if ints else "FALSE"
    if not casted:
        return "FALSE"
    if pa.types.is_timestamp(bt):
        # Compare stored integers via each unit's exact epoch function, never via Python
        # datetimes (they cannot hold every int64 timestamp and would truncate ns).
        fn = {"s": "epoch", "ms": "epoch_ms", "us": "epoch_us", "ns": "epoch_ns"}[bt.unit]
        to_us = {"s": 10**6, "ms": 10**3, "us": 1, "ns": 1}[bt.unit]
        ints = [pc.cast(cv, pa.int64())[0].as_py() for cv in casted]
        for x in ints:
            # DuckDB reserves +-(2**63 - 1) for +-infinity and cannot hold
            # instants beyond ~2**63 microseconds.
            if abs(x) >= 2**63 - 1 or abs(x) * to_us >= 2**63 - 1:
                raise NotImplementedError(f"timestamp value {x} ({bt.unit}) is outside DuckDB's range")
        return f"{fn}({col}) IN ({', '.join(str(x) for x in ints)})"
    if pa.types.is_date32(bt):
        # compare stored day counts (Python dates cannot hold every value)
        ints = [pc.cast(cv, pa.int32())[0].as_py() for cv in casted]
        return f"datediff('day', DATE '1970-01-01', {col}) IN ({', '.join(str(x) for x in ints)})"
    kept = [cv[0].as_py() for cv in casted]
    if pa.types.is_floating(bt):
        f32 = not pa.types.is_float64(bt)   # float32
        return "(" + " OR ".join(bit_eq(col, float(v), f32) for v in kept) + ")"
    return f"{col} IN ({', '.join(sql_literal(v) for v in kept)})"


def _int_match_values(values, bt: pa.DataType) -> list[int] | None:
    """Static match values on an int column as exact ints of its type (as nova-bf's
    `_int_match_values`): non-whole / non-finite floats and out-of-range values are dropped.
    None when a value is not int / float (`check_static_match_kinds` rejects those)."""
    lo, hi = _int_bounds(bt)
    out = []
    for v in values:
        if type(v) is float:
            if not (math.isfinite(v) and v.is_integer()):
                continue
            v = int(v)
        elif type(v) is not int:
            return None
        if lo <= v <= hi:
            out.append(v)
    return out


# Python value types a static `match` may use per column kind (nova-bf's
# `_STATIC_MATCH_OK`).
_STATIC_MATCH_OK = {"int": (int, float), "float": (int, float), "string": (str,), "bool": (bool,)}


def _static_match_kind(t: pa.DataType) -> str | None:
    bt = _base_type(t)
    if pa.types.is_floating(bt):
        return "float"
    if pa.types.is_integer(bt):
        return "int"
    if pa.types.is_string(bt) or pa.types.is_large_string(bt) or \
            getattr(pa.types, "is_string_view", lambda _: False)(bt):
        return "string"
    if pa.types.is_boolean(bt):
        return "bool"
    return None


def check_static_match_kinds(spec: "FilterSpec", schema: pa.Schema, skip_fields=()) -> None:
    """Reject static `match` values whose kind does not fit the column, like
    nova-bf's `check_static_match_types` (ValueError naming every offender).
    Declared date fields and other column types are not checked."""
    bad = []
    for g in GROUPS:
        for c in getattr(spec, g):
            if c.kind != "match" or c.field in skip_fields or c.field not in schema.names:
                continue
            t = schema.field(c.field).type
            ok = _STATIC_MATCH_OK.get(_static_match_kind(t))
            if ok is None:
                continue
            vals = c.value if isinstance(c.value, tuple) else (c.value,)
            wrong = [v for v in vals if type(v) not in ok]
            if wrong:
                bad.append(f"{c.field!r} ({t}) with {wrong!r}")
    if bad:
        raise ValueError("static match values don't fit their column's type: " + "; ".join(bad))


def _is_uint64(t: pa.DataType) -> bool:
    t = _base_type(t)
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        t = _base_type(t.value_type)
    return pa.types.is_uint64(t)


def _strip_frac_zeros(d: decimal.Decimal) -> decimal.Decimal:
    """Drop trailing fractional zeros exactly (`Decimal.normalize()` rounds to 28 digits)."""
    sign, digits, exp = d.as_tuple()
    if not any(digits):
        return decimal.Decimal(0)          # 0E-10 is zero, whatever its exponent
    digits = list(digits)
    while exp < 0 and len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exp += 1
    return decimal.Decimal((sign, tuple(digits), exp))


def _int_in_type(v: int, bt: pa.DataType) -> bool:
    lo, hi = _int_bounds(bt)
    return lo <= v <= hi


def _pq_eq_literal(t: pa.DataType, v, exact: bool, is_list: bool = False):
    """SQL for `col == v` as nova-bf's per-query match computes it. Returns
    (column_cast, literal), or None for "matches nothing".

      exact=True  -- Python `==` (MatchAny lists, and query columns holding a null, which
                     numpy keeps as object arrays): ints and floats compare as exact reals.
      exact=False -- numpy `==` on a typed query array: int vs float compares in float64
                     (lossy above 2**53); `v` was taken from `np.array(values)`.
    Different kinds never match; bools are 0/1; a Decimal equals a number only exactly."""
    cf, vf = _family(t), _value_family(v)
    bt = _base_type(t)
    if isinstance(v, (bool, np.bool_)):
        v = int(v)
    if cf == "num" and vf == "dec":            # number column, Decimal value: exact
        if not v.is_finite():
            return None
        if pa.types.is_floating(bt):
            f = float(v)
            return ("DOUBLE", sql_literal(f)) if decimal.Decimal(f) == v else None
        if v != v.to_integral_value():
            return None
        v = int(v)
        vf = "num"
    if cf == "num" and vf == "num":
        is_float_col = pa.types.is_floating(bt)
        if pa.types.is_boolean(bt):
            if float(v) not in (0.0, 1.0):
                return None
            return "INTEGER", str(int(v))
        if exact:
            if is_float_col:
                if isinstance(v, int) and float(v) != v:
                    return None                 # Python: 2**53+1 != 2.0**53
                return "DOUBLE", sql_literal(float(v))
            if isinstance(v, float):
                if not v.is_integer():
                    return None
                v = int(v)
            return (None, str(v)) if _int_in_type(v, bt) else None
        # numpy promotion (NumPy 2 compares mixed-sign integers exactly)
        if is_float_col or isinstance(v, float):
            return "DOUBLE", sql_literal(float(v))
        return (None, str(v)) if _int_in_type(v, bt) else None
    if cf == "dec" and vf in ("num", "dec"):
        # Decimal(float) is the float's EXACT binary value, as Python compares it.
        d = decimal.Decimal(v)
        if not d.is_finite():
            return None
        d = _strip_frac_zeros(d)
        if max(0, -d.as_tuple().exponent) > bt.scale:
            return None  # more fractional digits than any column value has
        # copy_abs(), not abs(): abs() rounds to the 28-digit context precision
        int_digits = len(format(d.copy_abs(), "f").partition(".")[0].lstrip("0"))
        if int_digits > bt.precision - bt.scale:
            return None  # larger than any column value (DuckDB would raise)
        try:
            return None, sql_literal(d)
        except TypeError:
            return None  # wider than DECIMAL(38): equals no column value
    if cf == "time" or vf == "time":
        # nova-bf's results for mixed temporal kinds are numpy accidents: support only the
        # pairs known to agree, refuse the rest
        naive_ts = pa.types.is_timestamp(bt) and bt.tz is None and bt.unit != "ns"
        if not is_list and naive_ts and isinstance(v, datetime.datetime) and v.tzinfo is None:
            return None, sql_literal(v)
        if not is_list and pa.types.is_date32(bt) and type(v) is datetime.date:
            return None, sql_literal(v)
        if cf != "time" and cf != "other":
            return None    # a date/time value never equals a str/number/bytes cell
        raise NotImplementedError(f"per-query match of {type(v).__name__} values on a {bt} column "
                                  f"(declare it in date_fields to compare as epoch microseconds)")
    if cf == vf and cf in ("str", "bin"):
        return None, sql_literal(v)
    return None


def _num(col: str) -> str:
    return f"CAST({col} AS DOUBLE)"


def _int_bounds(t: pa.DataType) -> tuple[int, int]:
    if pa.types.is_unsigned_integer(t):
        return 0, (1 << t.bit_width) - 1
    return -(1 << (t.bit_width - 1)), (1 << (t.bit_width - 1)) - 1


def _first_int_passing(b: float, strict: bool, lo: int, hi: int) -> int | None:
    """Smallest integer d in [lo, hi] with float64(d) > b (strict) / >= b. float64(d) is
    non-decreasing, so binary search is exact, also above 2**53."""
    def ok(d: int) -> bool:
        f = float(d)
        return f > b if strict else f >= b
    if not ok(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _exact_int_range_sql(col: str, bounds, bt: pa.DataType) -> str:
    """Static range on an int column, exact in the column type (as nova-bf's
    `_int_range_part`): gt b -> x > floor(b), gte -> x >= ceil(b), lt -> x < ceil(b),
    lte -> x <= floor(b). Out-of-type bounds keep all non-null rows or none; NaN keeps none."""
    lo, hi = _int_bounds(bt)
    parts = []
    for op, b in bounds:
        b = float(b)
        if math.isnan(b):
            return "FALSE"
        up = op in ("gt", "gte")
        if math.isinf(b):
            if (b < 0) == up:
                continue                       # every value passes
            return "FALSE"
        k = math.floor(b) if op in ("gt", "lte") else math.ceil(b)
        if up:
            first = k + 1 if op == "gt" else k   # keep x >= first
            if first > hi:
                return "FALSE"
            if first > lo:
                parts.append(f"{col} >= {first}")
        else:
            end = k if op == "lt" else k + 1     # keep x < end
            if end <= lo:
                return "FALSE"
            if end <= hi:
                parts.append(f"{col} < {end}")
    return " AND ".join(parts) if parts else f"{col} IS NOT NULL"


def _range_sql(col: str, bounds, typ: pa.DataType | None = None) -> str:
    """`bounds`: (op, float) pairs, compared in float64 like nova-bf. NaN corpus values never
    pass (DuckDB orders NaN above every number, so float columns are guarded); a NaN bound
    passes nothing. Int columns (including declared dates) compare natively against the
    exact integer threshold of the same float64 comparison (~3.5x cheaper than casting)."""
    if any(_is_nan(b) for _, b in bounds):
        return "FALSE"
    bt = _base_type(typ) if typ is not None else None
    if bt is not None and pa.types.is_integer(bt):
        lo, hi = _int_bounds(bt)
        parts = []
        for op, b in bounds:
            b = float(b)
            if op in ("gt", "gte"):   # keep d >= first passing
                d0 = _first_int_passing(b, op == "gt", lo, hi)
                if d0 is None:
                    return "FALSE"
                if d0 > lo:
                    parts.append(f"{col} >= {d0}")
            else:                     # lt b: keep d < first d with f(d) >= b; lte: d < first with f(d) > b
                d1 = _first_int_passing(b, op == "lte", lo, hi)
                if d1 is not None:
                    if d1 == lo:
                        return "FALSE"
                    parts.append(f"{col} < {d1}")
        return " AND ".join(parts) if parts else f"{col} IS NOT NULL"
    num = col if bt is not None and pa.types.is_float64(bt) else _num(col)
    parts = [f"NOT isnan({num})"]
    for op, b in bounds:
        parts.append(f"{num} {SQL_OP[op]} {sql_literal(float(b))}")
    return " AND ".join(parts)


# ---------------------------------------------------------------------------
# packed bitmaps (little bit order)
# ---------------------------------------------------------------------------

def nbytes(n_rows: int) -> int:
    return (n_rows + 7) // 8


def tail_mask(n_rows: int) -> int:
    r = n_rows % 8
    return 0xFF if r == 0 else (1 << r) - 1


def ones(n_rows: int) -> np.ndarray:
    out = np.full(nbytes(n_rows), 0xFF, np.uint8)
    if len(out):
        out[-1] = tail_mask(n_rows)
    return out


def pack_bool(arr: pa.ChunkedArray | pa.Array, n_rows: int) -> np.ndarray:
    """Null-free Arrow bool column -> packed little-order bytes (zero-copy read if aligned)."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks() if arr.num_chunks != 1 else arr.chunk(0)
    if arr.null_count:
        raise ValueError("pack_bool needs a null-free column (coalesce in SQL)")
    nb = nbytes(n_rows)
    if arr.offset % 8 == 0 and len(arr) == n_rows:
        buf = np.frombuffer(arr.buffers()[1], np.uint8)
        start = arr.offset // 8
        out = buf[start:start + nb].copy()
        if nb:
            out[-1] &= tail_mask(n_rows)
        return out
    return np.packbits(arr.to_numpy(zero_copy_only=False), bitorder="little")


_IDENTITY = object()
_FEW_DISTINCT = 64


def _byte_permutation(r: np.ndarray, n: int):
    """For result rows in order `r` (a permutation of range(n)): `perm` with
    packed_file_order = packed_result_order[perm]; `_IDENTITY` if in order; None when a run
    of consecutive rows is not byte aligned."""
    if n == 0:
        return _IDENTITY
    breaks = np.flatnonzero(np.diff(r) != 1) + 1
    if len(breaks) == 0 and r[0] == 0:
        return _IDENTITY
    pos = np.r_[0, breaks]                       # run starts in RESULT order
    lens = np.diff(np.r_[pos, n])
    src0 = r[pos]                                # run starts in FILE order
    ends_file = src0 + lens == n
    if (pos % 8).any() or (src0 % 8).any() or ((lens % 8 != 0) & ~ends_file).any():
        return None
    nbr = (lens + 7) // 8                        # bytes per run
    perm = np.empty(nbytes(n), np.int64)
    within = np.arange(int(nbr.sum())) - np.repeat(np.cumsum(nbr) - nbr, nbr)
    perm[np.repeat(src0 // 8, nbr) + within] = np.repeat(pos // 8, nbr) + within
    return perm


_POPCOUNT_CHUNK = 32 << 20   # bytes of masks counted per step (bounds the temporary)


def popcount_rows(packed: np.ndarray) -> np.ndarray:
    """Set bits per row of (m, nbytes) uint8 -> (m,) int64. numpy's bitwise_count (>= 2.0)
    is ~8x faster than unpackbits (used on older numpy)."""
    packed = np.asarray(packed, np.uint8)
    out = np.zeros(packed.shape[0], np.int64)
    if not packed.size:
        return out
    step = max(1, _POPCOUNT_CHUNK // max(1, packed.shape[1]))
    count = getattr(np, "bitwise_count", None)
    for i in range(0, packed.shape[0], step):
        part = packed[i:i + step]
        out[i:i + step] = (count(part) if count is not None else np.unpackbits(part, axis=1)) \
            .sum(axis=1, dtype=np.int64)
    return out


def unpack(packed: np.ndarray, n_rows: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=n_rows, bitorder="little").astype(bool)


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """One file's mask.

    Static filter: `static` is the packed (nbytes,) row mask.
    Per-query: query q's mask is `combo_masks[combo_ids[q]]`."""
    n_rows: int
    static: np.ndarray | None = None
    combo_ids: np.ndarray | None = None
    combo_masks: np.ndarray | None = None
    stats: dict = field(default_factory=dict)

    @property
    def per_query(self) -> bool:
        return self.static is None

    def row_mask(self) -> np.ndarray:
        assert self.static is not None
        return unpack(self.static, self.n_rows)

    def packed_per_query(self) -> np.ndarray:
        assert self.combo_ids is not None
        return self.combo_masks[self.combo_ids]

    def query_masks(self) -> np.ndarray:
        """(n_queries, n_rows) bool. Allocates the full mask: tests/small only."""
        return unpack(self.packed_per_query(), self.n_rows)

    def distinct_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dedup by the actual mask (different values often select the same rows). Returns
        (masks (m, nbytes), mask id per query, popcount per mask)."""
        assert self.combo_masks is not None
        seen: dict[bytes, int] = {}
        combo_to_mask = np.empty(len(self.combo_masks), np.int64)
        for i, m in enumerate(self.combo_masks):
            combo_to_mask[i] = seen.setdefault(m.tobytes(), len(seen))
        masks = np.empty((len(seen), self.combo_masks.shape[1]), np.uint8)
        for b, j in seen.items():
            masks[j] = np.frombuffer(b, np.uint8)
        counts = popcount_rows(masks)
        return masks, combo_to_mask[self.combo_ids], counts


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Leaf:
    group: str
    cond: Cond


class DuckFilter:
    """Evaluate one FilterSpec against corpus files.

    `queries`: Arrow table of the search's queries with at least `spec.query_columns()`.
    Query-side work (dates, tokenising, combo dedup) is done once here, not per file.
    """

    def __init__(self, spec: FilterSpec, queries: pa.Table | None = None, *,
                 corpus_date_fields: dict[str, str] | None = None,
                 query_date_fields: dict[str, str] | None = None,
                 threads: int | None = None, block_cols: int = 256,
                 prefilter_max_phrases: int = 8, selfcheck: bool = True,
                 factorize: str = "auto", chunk_rows: int | None = None,
                 text_impl: str = "bitmap", text_guards: str = "auto",
                 con: duckdb.DuckDBPyConnection | None = None):
        self.spec = spec
        self.corpus_dates = normalize_date_fields(corpus_date_fields)
        self.block_cols = block_cols
        # Corpus fields are projected under internal aliases so no user field
        # name (e.g. "__r", "__lf") can collide with the engine's own columns.
        self._alias = {f: f"__f{i}" for i, f in enumerate(spec.fields())}
        self._types: dict[str, pa.DataType] = {}
        self.prefilter_max_phrases = prefilter_max_phrases
        self.con = con or duckdb.connect()
        self.con.execute("SET TimeZone='UTC'")
        # Results are scattered by row id, never by position, so DuckDB may
        # return rows in any order (lets it parallelise freely).
        self.con.execute("SET preserve_insertion_order=false")
        if threads:
            self.con.execute(f"SET threads={int(threads)}")
        self.threads = int(self.con.execute("SELECT current_setting('threads')").fetchone()[0])
        if factorize not in ("auto", "always", "never"):
            raise ValueError("factorize must be auto/always/never")
        self.factorize = factorize
        self.chunk_rows = chunk_rows
        if text_impl not in TEXT_IMPLS:
            raise ValueError(f"text_impl must be one of {'/'.join(TEXT_IMPLS)}")
        self.text_impl = text_impl
        # auto: lean tokenisation when `lean_tokens_ok` (same masks); full: never
        if text_guards not in ("auto", "full"):
            raise ValueError("text_guards must be auto/full")
        self.text_guards = text_guards
        if selfcheck:
            ensure_selfchecks(self.con)

        self.leaves = [_Leaf(g, c) for g in GROUPS for c in getattr(spec, g)]
        self.pq_leaves = [l for l in self.leaves if l.cond.per_query]
        # static match_text phrases -> token sets (config literals)
        self.static_tokens: dict[Cond, frozenset | None] = {}
        for l in self.leaves:
            if l.cond.kind == "match_text":
                toks = tokenize_many(self.con, [l.cond.value])[0]
                if not toks:
                    raise ValueError(f"match_text on {l.cond.field!r} has no tokens")
                self.static_tokens[l.cond] = frozenset(toks)

        self.n_queries = None
        self.leaf_keys: list[list] = []        # per pq leaf: key per query (None = matches nothing)
        self.combo_ids = None
        self.combo_keys: list[tuple] = []
        if self.pq_leaves:
            if queries is None:
                raise ValueError("per-query filter needs the queries table")
            self._prepare_queries(queries, normalize_date_fields(query_date_fields))

    # -- query side -------------------------------------------------------

    def _query_values(self, queries: pa.Table, qdates: dict[str, str]) -> dict[str, list]:
        cols = self.spec.query_columns()
        missing = [c for c in cols if c not in queries.column_names]
        if missing:
            raise ValueError(f"queries table lacks filter columns {missing}")
        _check_supported(queries, cols, qdates, "queries")
        for l in self.pq_leaves:
            c = l.cond
            if c.kind == "match_from_query" and c.value not in qdates \
                    and _is_uint64(queries.schema.field(c.value).type):
                raise ValueError(f"match_from_query query column {c.value!r} is uint64: "
                                 f"rejected (as by nova-bf: cast it to int64)")
        out = {}
        for c in cols:
            if c in qdates:
                # preserve_insertion_order is off, so a multi-chunk queries
                # table can come back shuffled: carry the row index and sort.
                qt = queries.select([c]).append_column(
                    "__i", pa.array(np.arange(queries.num_rows, dtype=np.int64)))
                self.con.register("__q", qt)
                try:
                    t = self._q((
                        f"SELECT {_date_expr(ident(c), qdates[c], queries.schema.field(c).type)} AS v "
                        f"FROM __q ORDER BY __i"))
                finally:
                    self.con.unregister("__q")
                out[c] = t.column("v").to_pylist()
            else:
                out[c] = queries[c].to_pylist()
        return out

    def _prepare_queries(self, queries: pa.Table, qdates: dict[str, str]) -> None:
        vals = self._query_values(queries, qdates)
        self.n_queries = queries.num_rows
        for l in self.pq_leaves:
            c = l.cond
            if c.kind == "match_from_query":
                col = vals[c.value]
                listy = any(isinstance(v, (list, tuple)) for v in col)
                keys = []
                if listy:
                    for v in col:
                        if v is None or _is_nan(v):
                            keys.append(None)
                            continue
                        if not isinstance(v, (list, tuple)):
                            raise ValueError(f"{c.value!r} mixes scalar and list values")
                        s = frozenset(x for x in v if x is not None and not _is_nan(x))
                        keys.append(("any", s) if s else None)
                else:
                    # nova-bf compares against `np.array(values)`: an object array (Python ==)
                    # if a null is present, else numpy's dtype, reproduced by taking each
                    # value from that array
                    exact = any(v is None for v in col)
                    arr = None if exact else np.array(col)
                    for q, v in enumerate(col):
                        if not exact:
                            v = arr[q].item() if arr.dtype != object else arr[q]
                        keys.append(None if v is None or _is_nan(v) else ("eq", v, exact))
            elif c.kind == "range_from_query":
                keys = []
                for q in range(self.n_queries):
                    bounds = []
                    for op, qc in c.value:
                        b = vals[qc][q]
                        if b is None or _is_nan(b):
                            bounds = None
                            break
                        bounds.append((op, float(b)))
                    keys.append(tuple(bounds) if bounds is not None else None)
            else:  # match_text_from_query
                toks = tokenize_many(self.con, vals[c.value])
                keys = [frozenset(t) if t else None for t in toks]
            self.leaf_keys.append(keys)
        combos: dict[tuple, int] = {}
        ids = np.empty(self.n_queries, np.int64)
        for q in range(self.n_queries):
            k = tuple(keys[q] for keys in self.leaf_keys)
            ids[q] = combos.setdefault(k, len(combos))
        self.combo_ids = ids
        self.combo_keys = list(combos)

    # -- corpus side ------------------------------------------------------

    _query_s = 0.0   # seconds inside DuckDB (`_q`) since _evaluate last read it
    _text_cols = _text_lean = 0   # match_text columns evaluated / lean, per evaluate()

    def _q(self, sql: str) -> pa.Table:
        """`_fetch`, timed as DuckDB time (statement + Arrow export); the rest of a stage is
        counted as *_post_s."""
        t = time.perf_counter()
        try:
            return _fetch(self.con, sql)
        finally:
            self._query_s += time.perf_counter() - t

    def _x(self, sql: str):
        """`con.execute` for a statement returning no Arrow table, timed like `_q`."""
        t = time.perf_counter()
        try:
            return self.con.execute(sql)
        finally:
            self._query_s += time.perf_counter() - t

    def _register(self, table: pa.Table) -> int:
        fields = self.spec.fields()
        missing = [f for f in fields if f not in table.column_names]
        if missing:
            raise ValueError(f"corpus table lacks filter fields {missing}")
        n = table.num_rows
        _check_supported(table, fields, self.corpus_dates, "corpus")
        orig = table
        self._types = {f: table.schema.field(f).type for f in fields}
        check_static_match_kinds(self.spec, table.schema, tuple(self.corpus_dates))
        for l in self.pq_leaves:
            c = l.cond
            if c.kind == "match_from_query" and c.field not in self.corpus_dates \
                    and _is_uint64(self._types[c.field]):
                raise ValueError(f"match_from_query on uint64 column {c.field!r} is rejected "
                                 f"(as by nova-bf: cast it to int64)")
        self._duck = dict(self._types)
        al = [self._alias[f] for f in fields]
        t = table.select(fields).rename_columns(al).append_column(
            "__r", pa.array(np.arange(n, dtype=np.int64)))
        # Arrow's own (not correctly rounded) decimal -> float64, which nova-bf's static
        # range / match fallback compare against
        import pyarrow.compute as pc
        self._f64 = {}
        for f, a in zip(fields, al):
            if pa.types.is_decimal(_base_type(self._types[f])) and f not in self.corpus_dates:
                self._f64[f] = a + "_af"
                t = t.append_column(a + "_af", pc.cast(orig[f], pa.float64()))
        # DuckDB parallelises an Arrow scan per record batch, and a one-row-group file is one
        # batch: re-batch (zero-copy) into ~2 batches per thread
        cs = self.chunk_rows or max(2048, -(-n // (2 * self.threads)))
        t = pa.Table.from_batches(t.to_batches(max_chunksize=cs), schema=t.schema) if n else t
        self.con.register("__c", t)
        sel = ", ".join(
            (f"{_date_expr(a, self.corpus_dates[f], self._types[f])} AS {a}" if f in self.corpus_dates
             else a) for f, a in zip(fields, al))
        extra = "".join(f", {x}" for x in self._f64.values())
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW __cv AS SELECT __r, {sel}{extra} FROM __c")
        for f in self.corpus_dates:
            if f in self._types:
                self._types[f] = self._duck[f] = pa.int64()  # declared dates are epoch us from here on
        return n

    def _scatter_pack(self, t: pa.Table, names: list[str], n: int, out: np.ndarray, rows) -> None:
        """Columns of `t` (any row order, row id in __r) -> packed bitmaps in out[rows[i]].

        Multi-threaded DuckDB returns shuffled runs of consecutive rows. When every run is byte
        aligned, reordering is a byte permutation of the packed bits; else a per-row scatter."""
        r = t.column("__r").to_numpy()
        if len(r) != n:
            raise RuntimeError(f"DuckDB returned {len(r)} rows for a {n}-row file")
        perm = _byte_permutation(r, n)
        nb = nbytes(n)
        for j, name in zip(rows, names):
            col = t.column(name)
            if perm is not None:
                packed = pack_bool(col, n) if perm is _IDENTITY else pack_bool(col, n)[perm]
                if nb:
                    packed[-1] &= tail_mask(n)
                out[j] = packed
            else:
                b = np.zeros(n, bool)
                b[r] = col.to_numpy(zero_copy_only=False)
                out[j] = np.packbits(b, bitorder="little")

    def _eval_exprs(self, exprs: list[str], n: int) -> np.ndarray:
        """Boolean SQL expressions over __cv -> (len(exprs), nbytes) packed."""
        out = np.zeros((len(exprs), nbytes(n)), np.uint8)
        for b0 in range(0, len(exprs), self.block_cols):
            block = exprs[b0:b0 + self.block_cols]
            cols = ", ".join(f"coalesce(({e}), FALSE) AS f{i}" for i, e in enumerate(block))
            t = self._q((f"SELECT __r, {cols} FROM __cv"))
            self._scatter_pack(t, [f"f{i}" for i in range(len(block))], n,
                               out, range(b0, b0 + len(block)))
        return out

    def _factorizable(self, fld: str, n_exprs: int, n: int) -> bool:
        """Whether to evaluate a field's per-query match expressions on its distinct values
        instead of every row (as nova-bf's CPU path does). Never for floats: grouping on
        float keys has NaN / signed-zero corner cases."""
        if self.factorize == "never" or n == 0:
            return False
        if pa.types.is_floating(_base_type(self._types[fld])):
            return False
        if self.factorize == "always":
            return True
        if n_exprs < 4:
            return False
        a = self._alias[fld]
        nd = self._x(f"SELECT count(DISTINCT {a}) FROM __cv").fetchone()[0]
        return nd * 4 <= n

    def _eval_exprs_factorized(self, fld: str, exprs: list[str], n: int) -> np.ndarray | None:
        """`_eval_exprs` for expressions over one column: evaluate them on the column's distinct
        values, then broadcast to rows by value id. None (caller falls back) if the join is
        not 1:1."""
        a = self._alias[fld]
        self._x(f"""CREATE OR REPLACE TEMP TABLE __d AS
            SELECT (row_number() OVER ()) - 1 AS __did, {a}
            FROM (SELECT DISTINCT {a} FROM __cv WHERE {a} IS NOT NULL)""")
        try:
            codes_t = self._q((
                f"SELECT __cv.__r, __d.__did FROM __cv LEFT JOIN __d ON __cv.{a} = __d.{a}"))
            r = codes_t.column("__r").to_numpy()
            # __d's keys are DISTINCT under "=" (floats are never factorised),
            # so the LEFT JOIN yields exactly one row per corpus row.
            if len(r) != n:
                return None
            codes = np.full(n, -1, np.int64)
            did = codes_t.column("__did").to_numpy(zero_copy_only=False)
            did = np.where(np.isnan(did.astype(np.float64)), -1, did).astype(np.int64) \
                if did.dtype.kind == "f" else did.astype(np.int64)
            codes[r] = did
            nd = self._x("SELECT count(*) FROM __d").fetchone()[0]
            member = np.zeros((len(exprs), nd + 1), bool)  # last column: null rows (code -1)
            for b0 in range(0, len(exprs), self.block_cols):
                block = exprs[b0:b0 + self.block_cols]
                cols = ", ".join(f"coalesce(({e}), FALSE) AS f{i}" for i, e in enumerate(block))
                t = self._q((f"SELECT __did, {cols} FROM __d"))
                ids = t.column("__did").to_numpy()
                for i in range(len(block)):
                    member[b0 + i, ids] = t.column(f"f{i}").to_numpy(zero_copy_only=False)
        finally:
            self._x("DROP TABLE IF EXISTS __d")
        out = np.empty((len(exprs), nbytes(n)), np.uint8)
        if nd <= _FEW_DISTINCT:
            # few distinct values: OR the per-value bitmaps each expression accepts
            per_value = np.stack([np.packbits(codes == d, bitorder="little") for d in range(nd)]) \
                if nd else np.zeros((0, nbytes(n)), np.uint8)
            for j in range(len(exprs)):
                ids = np.flatnonzero(member[j, :nd])
                out[j] = (np.bitwise_or.reduce(per_value[ids], axis=0) if len(ids)
                          else 0)
            return out
        for j in range(len(exprs)):
            out[j] = np.packbits(member[j][codes], bitorder="little")
        return out

    def _text_masks(self, fld: str, token_sets: list[frozenset], n: int) -> np.ndarray:
        """Rows containing every word of each phrase.

        One DuckDB pass tokenises `fld`, keeping (row, word) pairs only for words some phrase
        needs; phrases are then combined by _phrase_masks."""
        out = np.zeros((len(token_sets), nbytes(n)), np.uint8)
        if not token_sets or n == 0:
            return out
        vocab = sorted(set().union(*token_sets))
        wid = {w: i for i, w in enumerate(vocab)}
        self.con.register("__vocab", pa.table({"wid": pa.array(range(len(vocab)), pa.int64()),
                                               "w": pa.array(vocab, pa.large_string())}))
        # Substring pre-check (few phrases only), exact: a row can hold token w only if
        # lower(row) contains w, since lower() is per-codepoint, context-free and
        # length-preserving (self-checked). Rows failing every phrase skip the costly split.
        # With many phrases nearly every row passes, so it is skipped. The two special rules
        # only put `ς` for `σ` and add U+0307 after `i`, so words holding them are checked with
        # both sides folded (`_fold_sql`). Under lean tokenisation the ASCII words are spelled
        # identically by plain lower(), so the check stays sound.
        if pa.types.is_null(_base_type(self._types[fld])):
            return out              # all-null column: no row has any word
        if _family(self._types[fld]) != "str":
            raise NotImplementedError(f"match_text on {fld!r} of type {self._types[fld]} "
                                      f"(only string columns are supported)")
        col = self._alias[fld]
        lean = self.text_guards == "auto" and lean_tokens_ok(vocab)
        self._text_cols += 1
        self._text_lean += int(lean)
        lowered = f"lower({col})" if lean else lower_sql(col)
        where = ""
        if len(token_sets) <= self.prefilter_max_phrases:
            def check(w):
                if "\u03c2" in w or "\u0307" in w:
                    return f"contains({_fold_sql('__lf')}, {sql_literal(_fold(w))})"
                return f"contains(__lf, {sql_literal(w)})"
            where = "WHERE " + " OR ".join(
                "(" + " AND ".join(check(w) for w in sorted(ts)) + ")"
                for ts in token_sets)
        def pairs(distinct: str) -> str:
            return f"""
                SELECT {distinct} p.r, v.wid FROM (
                    SELECT __r AS r, unnest({_tokens_sql('__t', '__lf', lean)}) AS w
                    FROM (SELECT __r, {col} AS __t, {lowered} AS __lf FROM __cv) {where}
                ) p JOIN __vocab v USING (w)
            """
        return self._phrase_masks(pairs, token_sets, wid, vocab, n, out)

    def _phrase_masks(self, pairs, token_sets, wid, vocab, n, out) -> np.ndarray:
        """Each phrase's rows (bit-packed into `out`) from the (row, word) pairs query
        `pairs(distinct)`: DuckDB word bitmaps, combined with bit_and."""
        # (phrase, word, word count): a phrase survives only if all its words occur (HAVING)
        pid, wids, nw = [], [], []
        for i, ts in enumerate(token_sets):
            for x in ts:
                pid.append(i)
                wids.append(wid[x])
                nw.append(len(ts))
        self.con.register("__pw", pa.table({"pid": pa.array(pid, pa.int64()),
                                            "wid": pa.array(wids, pa.int64()),
                                            "nw": pa.array(nw, pa.int64())}))
        try:
            # no DISTINCT: bitstring_agg sets a bit once however often it is fed
            t = self._q(f"""
                WITH wb AS (SELECT wid, bitstring_agg(r, 0, {n - 1}) AS b
                            FROM ({pairs('')}) GROUP BY wid)
                SELECT pw.pid, bit_and(wb.b) AS b FROM __pw pw JOIN wb USING (wid)
                GROUP BY pw.pid, pw.nw HAVING count(*) = pw.nw""")
        finally:
            self.con.unregister("__pw")
            self.con.unregister("__vocab")
        width = out.shape[1] + 1
        for pc, bc in zip(t.column("pid").chunks, t.column("b").chunks):
            m = len(bc)
            if not m:
                continue
            if bc.null_count:
                raise RuntimeError("NULL bitmap from DuckDB")
            wide = pa.types.is_large_binary(bc.type) or pa.types.is_large_string(bc.type)
            offs = np.frombuffer(bc.buffers()[1], np.int64 if wide else np.int32,
                                 count=m + 1, offset=bc.offset * (8 if wide else 4))
            if not (np.diff(offs) == width).all():
                raise RuntimeError("DuckDB BIT values of unexpected length")
            data = np.frombuffer(bc.buffers()[2], np.uint8, count=width * m,
                                 offset=int(offs[0])).reshape(m, width)
            out[pc.to_numpy()] = bit_blob_to_packed(data, n)
        return out

    def _leaf_sql(self, c: Cond, key=None) -> str:
        """One leaf as SQL over __cv. Rebuilt per file: the column type decides the semantics."""
        col, typ = self._alias[c.field], self._types[c.field]
        if pa.types.is_null(_base_type(typ)):
            return "FALSE"          # an all-null column matches nothing
        if c.kind == "match":
            vals = c.value if isinstance(c.value, tuple) else (c.value,)
            return _static_match_sql(col, typ, vals, self._f64.get(c.field))
        if c.kind in ("range", "range_from_query"):
            if _family(typ) not in ("num", "dec"):
                raise NotImplementedError(f"range on {c.field!r} of type {typ}: declare it in "
                                          f"date_fields (temporal) or use a numeric column")
            bounds = c.value if c.kind == "range" else key
            if pa.types.is_decimal(_base_type(typ)):
                # static: Arrow's inexact decimal -> double; per query: correctly rounded
                # float(Decimal), which DuckDB gets via the exact text form
                expr = (self._f64[c.field] if c.kind == "range"
                        else f"CAST(CAST({col} AS VARCHAR) AS DOUBLE)")
                return _range_sql(expr, bounds, pa.float64())
            if c.kind == "range" and pa.types.is_integer(_base_type(typ)):
                return _exact_int_range_sql(col, bounds, _base_type(typ))
            return _range_sql(col, bounds, typ)
        if c.kind == "match_from_query":
            if key[0] == "eq":
                items, exact, is_list = [key[1]], key[2], False
            else:
                items, exact, is_list = list(key[1]), True, True
            by_cast: dict[str | None, list[str]] = {}
            for x in items:
                lit = _pq_eq_literal(self._duck.get(c.field, typ), x, exact, is_list)
                if lit is not None:
                    by_cast.setdefault(lit[0], []).append(lit[1])
            if not by_cast:
                return "FALSE"
            parts = []
            for cast, lits in by_cast.items():
                lhs = f"CAST({col} AS {cast})" if cast else col
                parts.append(f"{lhs} = {lits[0]}" if len(lits) == 1 else f"{lhs} IN ({', '.join(lits)})")
            return "(" + " OR ".join(parts) + ")"
        raise AssertionError(c.kind)

    def evaluate(self, table: pa.Table) -> FilterResult:
        t0 = time.perf_counter()
        n = self._register(table)
        try:
            return self._evaluate(n, t0)
        finally:
            self.con.execute("DROP VIEW IF EXISTS __cv")
            self.con.unregister("__c")

    def _evaluate(self, n: int, t0: float) -> FilterResult:
        stats = {"sql_exprs": 0, "phrases": 0}
        nb = nbytes(n)
        # Every distinct (leaf, key) needing evaluation, batched by kind.
        expr_slots: dict[tuple, int] = {}
        exprs: list[str] = []
        expr_field: list[str | None] = []   # set for per-query match exprs (factorisable)
        text_slots: dict[str, dict[frozenset, int]] = {}

        def want_expr(c: Cond, key, slot=None) -> tuple:
            # static leaves are keyed by position: Cond equality merges values Python calls
            # equal (0 == -0.0) whose static-match SQL differs
            k = ("e", c, key) if slot is None else ("s", slot)
            if k not in expr_slots:
                expr_slots[k] = len(exprs)
                exprs.append(self._leaf_sql(c, key))
                expr_field.append(c.field if c.kind == "match_from_query" else None)
            return k

        def want_text(fld: str, ts: frozenset) -> tuple:
            d = text_slots.setdefault(fld, {})
            d.setdefault(ts, len(d))
            return ("t", fld, ts)

        static_ref = {}
        for li, l in enumerate(self.leaves):
            c = l.cond
            if c.per_query:
                continue
            static_ref[li] = (want_text(c.field, self.static_tokens[c]) if c.kind == "match_text"
                              else want_expr(c, None, slot=li))
        pq_refs = []
        for l, keys in zip(self.pq_leaves, self.leaf_keys):
            refs = {}
            for k in set(keys):
                if k is None:
                    continue
                refs[k] = (want_text(l.cond.field, k) if l.cond.text else want_expr(l.cond, k))
            pq_refs.append(refs)

        t1 = time.perf_counter()
        # prep_s: file registration + slot planning
        stats["prep_s"] = t1 - t0
        self._query_s = 0.0
        expr_masks = np.zeros((len(exprs), nb), np.uint8)
        groups: dict[str | None, list[int]] = {}
        for i, f in enumerate(expr_field):
            groups.setdefault(f, []).append(i)
        stats["factorized_exprs"] = 0
        for f, idx in groups.items():
            sub = [exprs[i] for i in idx]
            m = None
            if f is not None and self._factorizable(f, len(idx), n):
                m = self._eval_exprs_factorized(f, sub, n)
                if m is not None:
                    stats["factorized_exprs"] += len(idx)
            expr_masks[idx] = m if m is not None else self._eval_exprs(sub, n)
        t2 = time.perf_counter()
        # each stage split into DuckDB (query + Arrow export) and the rest
        sql_q, self._query_s = self._query_s, 0.0
        self._text_cols = self._text_lean = 0
        text_masks = {}
        for fld, d in text_slots.items():
            sets = [None] * len(d)
            for ts, i in d.items():
                sets[i] = ts
            text_masks[fld] = self._text_masks(fld, sets, n)
        t3 = time.perf_counter()
        text_q = self._query_s
        stats.update(sql_query_s=sql_q, sql_post_s=max(0.0, t2 - t1 - sql_q),
                     text_query_s=text_q, text_post_s=max(0.0, t3 - t2 - text_q),
                     text_cols=self._text_cols, text_lean_cols=self._text_lean)
        stats["sql_exprs"] = len(exprs)
        stats["phrases"] = sum(len(d) for d in text_slots.values())

        def get(ref) -> np.ndarray:
            if ref[0] in ("e", "s"):
                return expr_masks[expr_slots[ref]]
            return text_masks[ref[1]][text_slots[ref[1]][ref[2]]]

        zeros = np.zeros(nb, np.uint8)
        full = ones(n)
        keep = full.copy()
        static_or = None
        for li, l in enumerate(self.leaves):
            if l.cond.per_query:
                continue
            m = get(static_ref[li])
            if l.group == "must":
                keep &= m
            elif l.group == "must_not":
                keep &= ~m
            else:
                static_or = m.copy() if static_or is None else (static_or | m)
        if nb:
            keep[-1] &= tail_mask(n)

        if not self.pq_leaves:
            if self.spec.should:
                keep &= static_or
            stats.update(filter_s=t3 - t0, sql_s=t2 - t1, text_s=t3 - t2,
                         combine_s=time.perf_counter() - t3)
            return FilterResult(n, static=keep, stats=stats)

        combos = np.empty((len(self.combo_keys), nb), np.uint8)
        for ci, ck in enumerate(self.combo_keys):
            m = keep.copy()
            s = None if static_or is None else static_or.copy()
            for li, (l, key) in enumerate(zip(self.pq_leaves, ck)):
                part = zeros if key is None else get(pq_refs[li][key])
                if l.group == "must":
                    m &= part
                elif l.group == "must_not":
                    m &= ~part
                else:
                    s = part.copy() if s is None else (s | part)
            if self.spec.should:
                m &= s  # never None: a non-empty `should` has a static or per-query member
            if nb:
                m[-1] &= tail_mask(n)
            combos[ci] = m
        stats.update(combos=len(self.combo_keys), filter_s=time.perf_counter() - t0,
                     sql_s=t2 - t1, text_s=t3 - t2, combine_s=time.perf_counter() - t3)
        return FilterResult(n, combo_ids=self.combo_ids, combo_masks=combos, stats=stats)
