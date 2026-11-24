"""DuckFilter's match_text, "bitmap" (DuckDB word bitmaps + bit_and), must give the
same masks as the numpy reference (sorted-list intersection, ref_text_numpy.py) and as a
naive Python reference, on every row count around a byte boundary, with absent words,
chunked input, the substring pre-check on and off, and through evaluate()."""

import numpy as np
import pyarrow as pa
import pytest

from bench.data.filters import duckdb_filter as dfm
from ref_text_numpy import IMPLS, duck_filter

WORDS = ["alpha", "beta", "gamma", "delta", "straße", "ΟΔΟΣ", "İstanbul", "naïve", "x1", "日本"]
ABSENT = ["zeta", "omega"]


def corpus(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = int(rng.integers(0, 6))
        s = " ".join(rng.choice(WORDS, k).tolist())
        out.append(None if rng.random() < 0.05 else s)
    return out


def phrases(seed, m):
    rng = np.random.default_rng(seed)
    pool = WORDS + ABSENT
    return [" ".join(rng.choice(pool, int(rng.integers(1, 4)), replace=False).tolist())
            for _ in range(m)]


def engine(ph, impl, **kw):
    spec = dfm.parse_filter({"must": [{"field": "text", "match_text_from_query": "q"}]})
    return duck_filter(spec, pa.table({"q": pa.array(ph, pa.large_string())}),
                       threads=2, selfcheck=False, text_impl=impl, **kw)


def table(texts, chunks=1):
    arr = pa.array(texts, pa.large_string())
    if chunks > 1 and len(texts) > chunks:
        step = -(-len(texts) // chunks)
        arr = pa.chunked_array([arr[i:i + step] for i in range(0, len(texts), step)])
    return pa.table({"text": arr})


def naive(con, texts, ph):
    toks = dfm.tokenize_many(con, texts)
    want = np.zeros((len(ph), len(texts)), bool)
    for i, p in enumerate(dfm.tokenize_many(con, ph)):
        for r, t in enumerate(toks):
            want[i, r] = bool(p) and t is not None and set(p) <= set(t)
    return want


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 257, 1000, 4099])
@pytest.mark.parametrize("prefilter", [0, 8, 10**6])
def test_bitmap_equals_numpy_and_naive(n, prefilter):
    texts, ph = corpus(n, n), phrases(n + 1, 40)
    res = {}
    for impl in IMPLS:
        eng = engine(ph, impl, prefilter_max_phrases=prefilter)
        r = eng.evaluate(table(texts, chunks=3))
        res[impl] = r.query_masks()
        pk = r.packed_per_query()
        assert pk.shape[1] == dfm.nbytes(n)
        if n % 8:                       # padding bits of the last byte stay 0
            assert not (pk[:, -1] & ~np.uint8(dfm.tail_mask(n))).any()
    assert np.array_equal(res["bitmap"], res["numpy"])
    assert np.array_equal(res["bitmap"], naive(eng.con, texts, ph))


@pytest.mark.parametrize("n", [5, 8, 13, 4099])
def test_text_masks_direct(n):
    """_text_masks itself, incl. phrases whose words never occur and words
    with SQL / format metacharacters in the literals."""
    texts = corpus(n, 7 * n) + ["a{b}c it's 100% {distinct}", "{0} a{b}c"]
    ph = phrases(3, 25) + ["zeta", "zeta omega", "alpha zeta", "a{b}c", "{distinct}", "it's"]
    out = {}
    for impl in IMPLS:
        eng = engine(ph, impl)
        ts = [frozenset(t) for t in dfm.tokenize_many(eng.con, ph) if t]
        m = eng._register(table(texts))
        out[impl] = eng._text_masks("text", ts, m)
    assert np.array_equal(out["bitmap"], out["numpy"])
    assert out["bitmap"].any()


def test_all_null_and_empty():
    for impl in IMPLS:
        eng = engine(["alpha", "beta gamma"], impl)
        r = eng.evaluate(pa.table({"text": pa.array([None, None, None], pa.large_string())}))
        assert not r.packed_per_query().any()
        r = eng.evaluate(pa.table({"text": pa.array(["", " ", "!!"], pa.large_string())}))
        assert not r.packed_per_query().any()


@pytest.mark.parametrize("n", [1, 7, 8, 9, 31, 33])
def test_bit_blob_to_packed(n):
    rng = np.random.default_rng(n)
    bits = rng.random((6, n)) < 0.5
    nb = dfm.nbytes(n)
    p = nb * 8 - n
    stream = np.concatenate([np.ones((6, p), bool), bits], axis=1)   # padding bits are 1s
    data = np.concatenate([np.full((6, 1), p, np.uint8),
                           np.packbits(stream, axis=1, bitorder="big")], axis=1)
    got = dfm.bit_blob_to_packed(data, n)
    assert np.array_equal(got, np.packbits(bits, axis=1, bitorder="little"))
    bad = data.copy()
    bad[0, 0] = (p + 1) % 8
    with pytest.raises(RuntimeError):
        dfm.bit_blob_to_packed(bad, n)


def test_bad_impl():
    with pytest.raises(ValueError):
        engine(["alpha"], "nope")
    spec = dfm.parse_filter({"must": [{"field": "text", "match_text_from_query": "q"}]})
    with pytest.raises(ValueError):   # bench's DuckFilter has the bitmap path only
        dfm.DuckFilter(spec, pa.table({"q": pa.array(["a"], pa.large_string())}),
                       selfcheck=False, text_impl="numpy")
