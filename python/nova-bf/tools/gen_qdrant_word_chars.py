"""Measure which codepoints Qdrant's `word` tokenizer keeps INSIDE a token and
write them to `src/nova_bf/tokenizer_tables/qdrant_word_chars.py`.

Run against a live Qdrant (any scratch instance: one collection is created and
deleted):

    python tools/gen_qdrant_word_chars.py --url http://localhost:6333 [--rustc]

Method. One collection with a full-text index (`tokenizer: word`,
`lowercase: true`) on payload field `t`. Point `cp` gets `t = "a" + chr(cp) +
"b"` for every Unicode scalar value (U+0000..U+10FFFF minus surrogates). If
`cp` separates tokens the row's tokens are `a`, `b` and `MatchText("a")`
matches it; if `cp` is a word character the row is the single token
`a<cp>b` and it does not. So the points the filter does NOT match are exactly
the word characters. The whole measurement is repeated with base `q`/`z` and
must agree; known codepoints are spot-checked; the two filtered sets (must /
must_not) must partition the collection.

`--rustc` additionally compiles a tiny Rust program printing
`char::is_alphanumeric` for every scalar value and reports where the local
toolchain disagrees with the server (informational only: the live server is
authoritative, its toolchain's Unicode version may differ).

Output is deterministic (sorted inclusive ranges); the header records the
server version/commit and the measurement date.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

N = 0x110000
CPS = [c for c in range(N) if not (0xD800 <= c < 0xE000)]
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "nova_bf", "tokenizer_tables", "qdrant_word_chars.py")
# The DuckDB baseline harness vendors a byte-identical copy (it does not import
# nova_bf); written too when the harness is present in this checkout.
HARNESS_COPY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks",
                            "streaming", "harness", "_qdrant_word_chars.py")

# (codepoint, is word char) facts the measurement must reproduce.
SPOT = [(ord("x"), True), (ord("7"), True), (ord("-"), False), (ord("_"), False),
        (ord(" "), False), (0x093F, True), (0x094D, False), (0x24D0, True),
        (0x00E9, True), (0x0301, False), (0x1F600, False)]


def _req(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()  # ensure_ascii: safe for controls
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read())


def _ranges(chars):
    out = []
    for c in sorted(chars):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [tuple(r) for r in out]


def measure(url, cps, base=("a", "b"), batch=5000):
    """Return the subset of `cps` Qdrant keeps inside a word token."""
    name = f"wordchars_{uuid.uuid4().hex[:8]}"
    col = f"{url}/collections/{name}"
    _req(col, "PUT", {"vectors": {"size": 1, "distance": "Dot"}})
    try:
        _req(f"{col}/index?wait=true", "PUT", {"field_name": "t", "field_schema": {
            "type": "text", "tokenizer": "word", "lowercase": True}})
        for i in range(0, len(cps), batch):
            pts = [{"id": c, "vector": [1.0], "payload": {"t": base[0] + chr(c) + base[1]}}
                   for c in cps[i:i + batch]]
            _req(f"{col}/points?wait=true", "PUT", {"points": pts})
        while True:
            info = _req(col)["result"]
            if info["status"] == "green" and info["points_count"] == len(cps):
                break
            time.sleep(1)
        cond = {"key": "t", "match": {"text": base[0]}}
        got = {}
        for kind in ("must", "must_not"):
            ids, offset = [], None
            while True:
                body = {"filter": {kind: [cond]}, "limit": 50000,
                        "with_payload": False, "with_vector": False}
                if offset is not None:
                    body["offset"] = offset
                res = _req(f"{col}/points/scroll", "POST", body)["result"]
                ids += [p["id"] for p in res["points"]]
                offset = res.get("next_page_offset")
                if offset is None:
                    break
            got[kind] = set(ids)
        sep, word = got["must"], got["must_not"]
        assert not (sep & word) and (sep | word) == set(cps), "filters do not partition"
        return word
    finally:
        _req(col, "DELETE")


def rustc_alnum():
    """`char::is_alphanumeric` per scalar value from the local rustc, plus its versions."""
    src = ('fn main(){let (a,b,c)=char::UNICODE_VERSION;println!("{}.{}.{}",a,b,c);'
           'for c in 0u32..0x110000 { if let Some(ch)=char::from_u32(c) {'
           'if ch.is_alphanumeric(){println!("{}",c);} } } }')
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.rs")
        open(p, "w").write(src)
        subprocess.run(["rustc", "-O", "-o", os.path.join(d, "a"), p], check=True)
        lines = subprocess.run([os.path.join(d, "a")], check=True, capture_output=True,
                               text=True).stdout.split()
    ver = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    return ver, lines[0], {int(x) for x in lines[1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--rustc", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    server = _req(url + "/")

    t0 = time.time()
    word = measure(url, CPS, ("a", "b"))
    word2 = measure(url, CPS, ("q", "z"))
    if word != word2:
        sys.exit(f"probe bases disagree on {sorted(word ^ word2)[:20]}")
    bad = [hex(c) for c, want in SPOT if (c in word) != want]
    if bad:
        sys.exit(f"spot checks failed: {bad}")
    print(f"measured {len(word)} word chars in {time.time() - t0:.0f}s", file=sys.stderr)

    if a.rustc:
        ver, uver, rs = rustc_alnum()
        diff = sorted(rs ^ word)
        print(f"{ver} (Unicode {uver}): {len(rs)} alphanumeric, "
              f"{len(diff)} differ from Qdrant: "
              f"{[f'U+{c:04X}' for c in diff[:40]]}", file=sys.stderr)

    rng = _ranges(word)
    lines = [
        "# GENERATED by python/nova-bf/tools/gen_qdrant_word_chars.py -- do not edit.",
        f"# Measured against Qdrant {server['version']} (commit {server['commit']}) "
        f"on {datetime.date.today().isoformat()}:",
        "# word tokenizer, lowercase: true; each scalar value probed as 'a<cp>b' and",
        "# 'q<cp>z', word char iff MatchText('a'/'q') does NOT match. See the script.",
        '"""Codepoints Qdrant\'s `word` tokenizer keeps inside a token (measured)."""',
        "",
        f"QDRANT_VERSION = {server['version']!r}",
        f"QDRANT_COMMIT = {server['commit']!r}",
        f"MEASURED = {datetime.date.today().isoformat()!r}",
        f"N_WORD_CHARS = {len(word)}",
        "",
        "# Sorted, disjoint, non-adjacent inclusive (first, last) codepoint ranges.",
        "WORD_CHAR_RANGES = (",
    ]
    row = []
    for lo, hi in rng:
        row.append(f"(0x{lo:04X}, 0x{hi:04X}),")
        if len(row) == 4:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append(")")
    outs = [a.out] + ([HARNESS_COPY] if a.out == OUT and os.path.isdir(
        os.path.dirname(HARNESS_COPY)) else [])
    for out in outs:
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {len(rng)} ranges to {os.path.abspath(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
