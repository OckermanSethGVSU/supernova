"""Derive the two Unicode properties Rust's `str::to_lowercase` consults for
Greek final sigma, and write them to `src/nova_bf/tokenizer_tables/rust_case_props.py`.

    python tools/gen_rust_case_props.py [--out PATH]

Why. Qdrant's `word` tokenizer (`lowercase: true`) lowercases each token with
Rust's `str::to_lowercase`, whose only context-dependent rule is Final_Sigma
(alloc/src/str.rs, `map_uppercase_sigma`): a capital `Σ` U+03A3 becomes `ς`
U+03C2 iff

    case_ignorable_then_cased(chars BEFORE it, reversed)
    && !case_ignorable_then_cased(chars AFTER it)

where `case_ignorable_then_cased(it)` skips `Case_Ignorable` chars and asks
whether the next char exists and is `Cased`. Otherwise it becomes `σ`.
Qdrant applies it to the token alone (it splits first; measured, see
`tests/parity/test_parity_tokenizer.py`), so "before"/"after" stop at the
token's edges.

Method. Measured from the local rustc's own `to_lowercase` (so it is Rust's
behaviour by construction, not a re-reading of the Unicode data files): for
every scalar value `c`,

    fa = to_lowercase("A" c "Σ") ends in 'ς'    (= Case_Ignorable(c) | Cased(c))
    fc = to_lowercase(c "Σ")     ends in 'ς'    (= !Case_Ignorable(c) & Cased(c))
    ff = to_lowercase("AΣ" c)[1] == 'ς'         (= Case_Ignorable(c) | !Cased(c))

so `Case_Ignorable(c) = fa & !fc` and, for a char that is not case-ignorable,
`Cased(c) = fc`; `ff` must agree with both (checked). `Cased` of a
case-ignorable char is never consulted by the rule (it is skipped first), so
`CASED_RANGES` holds exactly the cased chars that are NOT case-ignorable.

The same program also checks the two facts the tokenizer relies on: the only
lowercase mapping longer than one codepoint is U+0130 -> U+0069 U+0307, and
that `char::UNICODE_VERSION` is recorded. The live-server cross-check of the
whole rule (every case-ignorable and cased word character, in both contexts)
is `test_final_sigma_contexts_are_still_qdrants`.

Output is deterministic (sorted inclusive ranges); the header records the
rustc version, its Unicode version and the date.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "nova_bf", "tokenizer_tables", "rust_case_props.py")
# The DuckDB baseline harness vendors a byte-identical copy (it does not import
# nova_bf); written too when the harness is present in this checkout.
HARNESS_COPY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks",
                            "streaming", "harness", "_rust_case_props.py")

RUST = r"""
fn main() {
    let (a, b, c) = char::UNICODE_VERSION;
    println!("{}.{}.{}", a, b, c);
    for cp in 0u32..0x110000 {
        let ch = match char::from_u32(cp) { Some(x) => x, None => continue };
        let fa = format!("A{ch}\u{3A3}").to_lowercase().ends_with('\u{3C2}');
        let fc = format!("{ch}\u{3A3}").to_lowercase().ends_with('\u{3C2}');
        let ff = format!("A\u{3A3}{ch}").to_lowercase().chars().nth(1) == Some('\u{3C2}');
        let n = ch.to_lowercase().count();
        println!("{} {} {} {} {}", cp, fa as u8, fc as u8, ff as u8, n);
    }
}
"""

# (codepoint, case_ignorable, cased-and-not-case-ignorable) facts to reproduce.
SPOT = [(ord("A"), False, True), (ord("a"), False, True), (ord("1"), False, False),
        (ord("'"), True, False), (ord("."), True, False), (0x0345, True, False),
        (0x02B0, True, False), (0x0301, True, False), (0x03A3, False, True),
        (0x24B6, False, True), (0x01C5, False, True), (0x4E2D, False, False)]


def _ranges(chars):
    out = []
    for c in sorted(chars):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [tuple(r) for r in out]


def measure():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.rs")
        with open(p, "w") as f:
            f.write(RUST)
        subprocess.run(["rustc", "-O", "-o", os.path.join(d, "a"), p], check=True)
        lines = subprocess.run([os.path.join(d, "a")], check=True, capture_output=True,
                               text=True).stdout.splitlines()
    ver = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    ci, cased, multi = set(), set(), []
    for line in lines[1:]:
        cp, fa, fc, ff, n = (int(x) for x in line.split())
        is_ci = bool(fa) and not fc
        if bool(ff) != (is_ci or not fc):
            sys.exit(f"U+{cp:04X}: following-context probe disagrees ({fa} {fc} {ff})")
        if is_ci:
            ci.add(cp)
        elif fc:
            cased.add(cp)
        if n != 1:
            multi.append(cp)
    if multi != [0x0130]:
        sys.exit(f"multi-codepoint lowercase mappings changed: {[hex(c) for c in multi]}; "
                 f"the tokenizer special-cases only U+0130")
    return ver, lines[0], ci, cased


def _emit(name, rng, lines):
    lines.append(f"{name} = (")
    row = []
    for lo, hi in rng:
        row.append(f"(0x{lo:04X}, 0x{hi:04X}),")
        if len(row) == 4:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append(")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    ver, uver, ci, cased = measure()
    bad = [hex(c) for c, want_ci, want_cased in SPOT
           if (c in ci) != want_ci or (c in cased) != want_cased]
    if bad:
        sys.exit(f"spot checks failed: {bad}")
    lines = [
        "# GENERATED by python/nova-bf/tools/gen_rust_case_props.py -- do not edit.",
        f"# Derived from {ver} (char::UNICODE_VERSION {uver}) on "
        f"{datetime.date.today().isoformat()}",
        "# by probing its own str::to_lowercase Final_Sigma behaviour. See the script.",
        '"""Case_Ignorable / Cased as Rust\'s `to_lowercase` Final_Sigma rule sees them."""',
        "",
        f"RUSTC_VERSION = {ver!r}",
        f"UNICODE_VERSION = {uver!r}",
        f"MEASURED = {datetime.date.today().isoformat()!r}",
        f"N_CASE_IGNORABLE = {len(ci)}",
        f"N_CASED = {len(cased)}",
        "",
        "# Sorted, disjoint, non-adjacent inclusive (first, last) codepoint ranges.",
        "# Case_Ignorable scalar values:",
    ]
    _emit("CASE_IGNORABLE_RANGES", _ranges(ci), lines)
    lines.append("# Cased scalar values that are NOT Case_Ignorable (the rule skips")
    lines.append("# case-ignorables before asking Cased, so only these matter):")
    _emit("CASED_RANGES", _ranges(cased), lines)
    outs = [a.out] + ([HARNESS_COPY] if a.out == OUT and os.path.isdir(
        os.path.dirname(HARNESS_COPY)) else [])
    for out in outs:
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {len(ci)} case-ignorable / {len(cased)} cased to "
              f"{os.path.abspath(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
