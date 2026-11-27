"""The vectorised list-defect check (load.list_defects_and_bad, check.defects) against the
per-query loop it replaced, kept here as the reference: the same document (counts, examples,
their order) and the same per-query bad mask for every input."""

import numpy as np
import pytest

from bench.verify import check as V
from bench.verify import load as L

OOR = L.OUT_OF_RANGE


# ---------------------------------------------------------------------------------------------
# the reference: the loops as they were before vectorising
# ---------------------------------------------------------------------------------------------
def ref_list_defects(ids, kmax, mapped):
    dq = dn = uq = un = oq = on = 0
    ex = []
    for q, row in enumerate(ids):
        A = np.asarray(row[:kmax])
        oor = np.nonzero(A == OOR)[0]
        if len(oor):
            oq += 1
            on += len(oor)
            if len(ex) < 5:
                ex.append({"query": q, "out_of_range_positions": [int(i) for i in oor[:5]]})
        good = A[A >= 0]
        u, c = np.unique(good, return_counts=True)
        if (c > 1).any():
            dq += 1
            dn += int((c - 1).sum())
            if len(ex) < 5:
                ex.append({"query": q, "duplicate_ids": [int(i) for i in u[c > 1][:5]]})
        neg = np.nonzero((A < 0) & (A != OOR))[0]
        if len(neg):
            last_valid = np.nonzero(A >= 0)[0]
            bad = neg if mapped else neg[neg < (last_valid[-1] if len(last_valid) else -1)]
            if len(bad):
                uq += 1
                un += len(bad)
                if len(ex) < 5:
                    ex.append({"query": q, "unmapped_positions": [int(i) for i in bad[:5]]})
    out = {"queries_with_duplicate_ids": dq, "duplicate_entries": dn,
           "queries_with_unmapped_entries": uq, "unmapped_entries": un,
           "failed": bool(dq or uq or oq), "examples": ex}
    if oq:
        out.update(queries_with_out_of_range_ids=oq, out_of_range_entries=on)
    return out


def ref_bad(ids, k, mapped):
    bad = np.zeros(len(ids), bool)
    for q, row in enumerate(ids):
        A = np.asarray(row[:k], np.int64)
        v = A >= 0
        last = np.flatnonzero(v)[-1] if v.any() else -1
        neg = np.flatnonzero(A < 0)
        bad[q] = (len(np.unique(A[v])) < v.sum() or (A < -1).any()
                  or (len(neg) > 0 and (mapped or neg[0] < last)))
    return bad


def ref_defects(ids, k, mapped):
    d = ref_list_defects(ids, k, mapped)
    return {"_bad": ref_bad(ids, k, mapped), "failed": d["failed"],
            "duplicate_ids": d["queries_with_duplicate_ids"],
            "holes_or_unmapped": d["queries_with_unmapped_entries"],
            "out_of_range": d.get("queries_with_out_of_range_ids", 0),
            "examples": d["examples"][:V.EXAMPLES]}


# ---------------------------------------------------------------------------------------------
# random lists
# ---------------------------------------------------------------------------------------------
def _lists(rng, n, k):
    """n lists of every kind: clean, short, empty, longer than k, trailing -1 pads, -1 holes
    before real ids, duplicates, -2 and other negatives, ids from a tiny or a large range (so
    duplicates are common or rare), numpy arrays and Python lists."""
    out = []
    for _ in range(n):
        ln = int(rng.choice([0, 1, k // 2, k - 1, k, k + 3, int(rng.integers(0, 2 * k + 2))]))
        hi = int(rng.choice([3, 10, 1000, 2**40]))
        a = rng.integers(0, hi, ln).astype(np.int64)
        r = rng.random(ln)
        a[r < rng.choice([0.0, 0.05, 0.3])] = -1                      # holes
        a[(r > 0.97) & (rng.random(ln) < 0.5)] = OOR                  # out of range
        if rng.random() < 0.1 and ln:
            a[rng.integers(0, ln)] = -int(rng.integers(3, 9))         # another negative
        if rng.random() < 0.3 and ln:                                 # trailing pads
            a[int(rng.integers(0, ln)):] = -1
        if rng.random() < 0.2 and ln > 1:                             # a planted duplicate
            i, j = rng.choice(ln, 2, replace=False)
            a[j] = a[i]
        out.append(a.tolist() if rng.random() < 0.15 else a)
    return out


@pytest.mark.parametrize("seed", range(200))
def test_the_vectorised_defects_equal_the_per_query_loop(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.choice([1, 2, 5, 12]))
    ids = _lists(rng, int(rng.choice([0, 1, 7, 40])), k)
    for kmax in (k, k + 5):           # also a k larger than every list
        for mapped in (False, True):
            got = L.list_defects(ids, kmax, mapped)
            assert got == ref_list_defects(ids, kmax, mapped)
            assert [type(x["query"]) for x in got["examples"]] == [int] * len(got["examples"])
            d, want = V.defects(ids, kmax, mapped), ref_defects(ids, kmax, mapped)
            assert np.array_equal(d.pop("_bad"), want.pop("_bad"))
            assert d == want


def test_the_documented_cases():
    ids = [np.array([5, 5, 7]), np.array([1, -1, 2]), np.array([3, 4, -1]), np.array([], np.int64),
           np.array([OOR, 1]), np.array([9, -4])]
    d = L.list_defects(ids, 3, False)
    assert d == ref_list_defects(ids, 3, False)
    assert d["queries_with_duplicate_ids"] == 1 and d["queries_with_unmapped_entries"] == 1
    assert d["queries_with_out_of_range_ids"] == 1
    _, bad = L.list_defects_and_bad(ids, 3, False)
    assert bad.tolist() == [True, True, False, False, True, True]
    _, bad = L.list_defects_and_bad(ids, 3, True)       # nova-bf: any -1 is an unmapped id
    assert bad.tolist() == [True, True, True, False, True, True]
