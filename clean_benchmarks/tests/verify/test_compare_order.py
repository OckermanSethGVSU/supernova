"""bench/verify/compare.py's (moved from runs/common/compare.py) float64 ORDER check (--rescore; --no-order-check
turns it off): at every rank where two lists differ, both ids' exact scores
must be within the --tie-ulps window, else a FAIL. Unit level on compare()
with a hand-made exact table, and end to end (compare.py as a subprocess) on
tiny local dense / sparse / multivector corpora with a nova-bf-shaped parquet
(md5 UUIDs of "<local path>:<row>") and a harness-shaped npz (-1 padded).
Each corpus holds a NEAR-TIED pair (row 1 = row 0 with one value nudged by a
float32 ULP), so swapping those two is a tie swap, while swapping two
clearly different ids is a real order disagreement. No GPU, no network."""
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import make_dummy
from bench.data.formats import multivector as multivec
from bench.data.formats import sparse
from bench.verify import compare as cmp

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.dirname(os.path.dirname(HERE))
# the CLI as a module, run from clean_benchmarks (the old test ran the runs/common/compare.py script)
COMPARE = ["-m", "bench.verify.compare"]
LOWEST = np.finfo(np.float32).min
K = 10


def a64(*x):
    return np.array(x, np.int64)


def ulp(x):
    return float(np.spacing(np.float32(abs(x))))


# ------------------------------------------------------------------ unit: compare()

def one(A, B, exact, kmax=None, **kw):
    """compare() on one query with own scores = the exact ones (float32)."""
    sa = [np.array([exact[(0, int(i))] for i in A], np.float64)]
    sb = [np.array([exact[(0, int(i))] for i in B], np.float64)]
    return cmp.compare(([np.asarray(A)], sa), ([np.asarray(B)], sb), 1, 1e-4,
                       kmax or max(len(A), len(B)), exact=exact, names=("p", "q"),
                       order=True, **kw)


def test_identical_lists_have_no_order_mismatch():
    ex = {(0, i): 1.0 - 0.1 * i for i in range(5)}
    r = one(a64(0, 1, 2, 3, 4), a64(0, 1, 2, 3, 4), ex)
    assert r["f64_order_mismatched_ranks"] == 0
    assert r["f64_order_real_disagreements"] == 0 and r["f64_order_tie_swaps"] == 0
    assert r["at_k"] == {"k": 5, "set_equal": 1.0, "overlap": 1.0}


def test_near_tied_swap_is_a_tie():
    s = 0.5
    ex = {(0, 0): 0.9, (0, 1): s, (0, 2): s + 3 * ulp(s), (0, 3): 0.1}
    r = one(a64(0, 2, 1, 3), a64(0, 1, 2, 3), ex)
    assert r["f64_order_mismatched_ranks"] == 2
    assert r["f64_order_tie_swaps"] == 2 and r["f64_order_real_disagreements"] == 0
    assert 2.9 < r["f64_order_max_tie_gap_ulps"] < 3.1
    assert r["f64_real_disagreements"] == 0     # same sets: the stray rule sees nothing


def test_exactly_equal_scores_are_a_tie():
    ex = {(0, 0): 0.9, (0, 1): 0.5, (0, 2): 0.5}
    r = one(a64(0, 2, 1), a64(0, 1, 2), ex)
    assert r["f64_order_tie_swaps"] == 2 and r["f64_order_max_tie_gap_ulps"] == 0.0


def test_window_edge():
    s = 0.25
    for gap, real in ((128, 0), (129, 2)):
        ex = {(0, 1): s + gap * ulp(s), (0, 2): s}
        r = one(a64(2, 1), a64(1, 2), ex)
        assert r["f64_order_real_disagreements"] == real, gap
    ex = {(0, 1): s + 200 * ulp(s), (0, 2): s}
    r = one(a64(2, 1), a64(1, 2), ex, tie_ulps=256)
    assert r["f64_order_real_disagreements"] == 0


def test_clear_swap_is_real_with_examples():
    ex = {(0, i): 1.0 - 0.1 * i for i in range(5)}
    r = one(a64(0, 1, 2, 3, 4), a64(0, 3, 2, 1, 4), ex)
    assert r["f64_order_mismatched_ranks"] == 2 and r["f64_order_real_disagreements"] == 2
    assert r["f64_order_queries_with_real"] == 1
    e = r["f64_order_examples"][0]
    assert e["query"] == 0 and e["rank"] == 1 and e["ids"] == [1, 3]
    assert e["exact"] == [0.9, pytest.approx(0.7)]
    assert e["gap_ulps"] == pytest.approx(0.2 / ulp(0.9))
    # diagnostic: q put 3 (0.7) above 1 (0.9) -> an inversion in q's own list, none in p's
    assert r["f64_order_inversions_by_system"] == {"p": 0, "q": 1}


def test_minus_inf_against_finite_is_real_two_minus_inf_tie():
    ex = {(0, 0): 0.9, (0, 1): -np.inf, (0, 2): -np.inf, (0, 3): 0.1}
    r = one(a64(0, 1, 2), a64(0, 2, 1), ex)            # two -inf non-candidates swapped
    assert r["f64_order_tie_swaps"] == 2 and r["f64_order_real_disagreements"] == 0
    r = one(a64(0, 3, 1), a64(0, 1, 3), ex)            # -inf vs 0.1
    assert r["f64_order_real_disagreements"] == 2
    assert r["f64_order_max_real_gap_ulps"] == float("inf")


def test_nan_not_judged():
    ex = {(0, 0): 0.9, (0, 1): np.nan, (0, 2): 0.5}
    r = one(a64(0, 1, 2), a64(0, 2, 1), ex)
    assert r["f64_order_not_judged"] == 2 and r["f64_order_real_disagreements"] == 0


def test_short_lists_order_checked_only_to_the_shorter_end():
    # filtered-like: P has 3 valid ids then -1 padding, Q has 3 ids (nova-bf short list)
    ex = {(0, i): 1.0 - 0.1 * i for i in range(6)}
    A = np.array([0, 2, 1, -1, -1], np.int64)
    B = a64(0, 1, 2)
    sa = [np.array([0.9, 0.8, 0.9, LOWEST, LOWEST], np.float64)]
    sb = [np.array([1.0, 0.9, 0.8], np.float64)]
    r = cmp.compare(([A], sa), ([B], sb), 1, 1e-4, 5, exact=ex, names=("h", "nb"), order=True)
    assert r["f64_order_mismatched_ranks"] == 2 and r["f64_order_real_disagreements"] == 2
    assert r["at_k"]["set_equal"] == 1.0 and r["f64_real_disagreements"] == 0
    # different lengths: only ranks < 3 are order-checked; the extra ids are strays
    B2 = a64(0, 1, 2, 3, 4)
    r = cmp.compare(([A], sa), ([B2], [np.array([1, .9, .8, .7, .6])]), 1, 1e-4, 5, exact=ex,
                    names=("h", "nb"), order=True)
    assert r["f64_order_mismatched_ranks"] == 2
    assert r["f64_wrong_by_system"] == {"h": 2, "nb": 0}   # dense: the short list dropped them
    # all -1 / empty lists on both sides (a query no row passes)
    r = cmp.compare(([np.full(5, -1, np.int64)], [np.full(5, LOWEST)]),
                    ([np.array([], np.int64)], [np.array([])]), 1, 1e-4, 5, exact={},
                    names=("h", "nb"), order=True)
    assert r["f64_order_mismatched_ranks"] == 0 and r["at_k"]["set_equal"] == 1.0


def test_needed_ids_order_adds_mismatched_ranks():
    sys_ = {"a": ([a64(0, 1, 2, 3, -1)], None), "b": ([a64(0, 2, 1, 3)], None)}
    assert cmp.needed_ids(sys_, 1, 5) == [set()]
    assert cmp.needed_ids(sys_, 1, 5, order=True) == [{1, 2}]


def test_rescore_bytes_dense_keeps_no_vector_copies():
    need = [{0, 1, 2}, {1, 150}]
    offs = np.array([0, 100, 200])
    est, ng, nf = cmp.rescore_bytes(need, offs, 1024, 4, 8)
    want = (200 * 1024 * 4 * 2 + 4 * 1024 * 8 * 2 + 2 * 2 * 5 * 1024 * 8
            + 5 * cmp.PAIR_BYTES + 2 * 1024 * 8 * 2)
    assert (est, ng, nf) == (want, 4, 2)


# ------------------------------------------------------------------ end to end

def nudge(v):
    return np.nextafter(np.float32(v), np.float32(np.inf), dtype=np.float32)


def dense_corpus(tmp, metric):
    """2 files x 30 rows, dim 16; row 1 = row 0 with one value one float32 ULP up."""
    rng = np.random.default_rng(5)
    x = make_dummy.unit_vectors(rng, 60, 16)
    x[1] = x[0]
    x[1, 3] = nudge(x[1, 3])
    os.makedirs(f"{tmp}/corpus")
    for f in range(2):
        make_dummy.write_vectors(f"{tmp}/corpus/part-{f:05d}.parquet", x[30 * f:30 * f + 30])
    xq = np.concatenate([x[:1], make_dummy.unit_vectors(rng, 4, 16)])
    qids = [f"q{i}" for i in range(len(xq))]
    pq.write_table(pa.table({"query_id": qids, "embedding": make_dummy.vectors_table(xq)
                             .column("embedding")}), f"{tmp}/q.parquet")
    keys, _, offsets, _ = cmp.corpus_index(f"{tmp}/corpus", need_ids=False)
    need = [set(range(60)) for _ in xq]
    ex = cmp.exact_scores(need, None, "embedding", offsets, keys, pafs.LocalFileSystem(), xq,
                          metric, threads=2)
    # the per-file chunked float64 scoring == a plain float64 numpy product
    q64, x64 = xq.astype(np.float64), x.astype(np.float64)
    if metric == "cosine":
        q64 /= np.linalg.norm(q64, axis=1, keepdims=True)
        x64 /= np.linalg.norm(x64, axis=1, keepdims=True)
    S = q64 @ x64.T
    assert max(abs(ex[(q, i)] - S[q, i]) for q in range(len(xq)) for i in range(60)) < 1e-14
    return qids, keys, offsets, ex, ["--vcol", "embedding", "--qcol", "embedding"]


def sparse_corpus(tmp, metric):
    """2 files x 30 rows; row 0 scaled x4 (so a query equal to it ranks rows 0 / 1
    first), row 1 = row 0 with one value one float32 ULP up."""
    rng = np.random.default_rng(3)
    ip, ix, iv = make_dummy.random_sparse(rng, 60, 80, 6)
    rows = [(ix[ip[r]:ip[r + 1]].copy(), iv[ip[r]:ip[r + 1]].copy()) for r in range(60)]
    rows[0] = (rows[0][0], (rows[0][1] * 4).astype(np.float32))
    v1 = rows[0][1].copy()
    v1[0] = nudge(v1[0])
    rows[1] = (rows[0][0].copy(), v1)

    def parts(rs):
        p = np.concatenate([[0], np.cumsum([len(i) for i, _ in rs])]).astype(np.int64)
        return p, np.concatenate([i for i, _ in rs]), np.concatenate([v for _, v in rs])
    os.makedirs(f"{tmp}/corpus")
    for f in range(2):
        make_dummy.write_sparse(f"{tmp}/corpus/part-{f:05d}.parquet", parts(rows[30 * f:30 * f + 30]))
    qr = make_dummy.random_sparse(np.random.default_rng(9), 4, 80, 6)
    qrows = [rows[0]] + [(qr[1][qr[0][r]:qr[0][r + 1]], qr[2][qr[0][r]:qr[0][r + 1]])
                         for r in range(4)]
    qids = [f"q{i}" for i in range(len(qrows))]
    make_dummy.write_sparse(f"{tmp}/q.parquet", parts(qrows), extra={"query_id": pa.array(qids)})
    keys, _, offsets, _ = cmp.corpus_index(f"{tmp}/corpus", need_ids=False)
    q, _ = sparse.load_queries(f"{tmp}/q.parquet", "sparse_embedding")
    ex = cmp.exact_scores_sparse([set(range(60)) for _ in qrows], offsets, keys,
                                 pafs.LocalFileSystem(), "sparse_embedding", q, metric, threads=2)
    return qids, keys, offsets, ex, ["--modality", "sparse", "--vcol", "sparse_embedding",
                                     "--qcol", "sparse_embedding"]


def mv_corpus(tmp, metric):
    """2 files x 20 documents of unit tokens, dim 8; document 1 = document 0 with
    one value one float32 ULP up; query 0 = document 0's tokens."""
    rng = np.random.default_rng(4)
    off, tok = make_dummy.random_multivector(rng, 40, 8, 3)
    docs = [tok[off[r]:off[r + 1]].copy() for r in range(40)]
    docs[1] = docs[0].copy()
    docs[1][0, 0] = nudge(docs[1][0, 0])

    def parts(ds):
        o = np.concatenate([[0], np.cumsum([len(d) for d in ds])]).astype(np.int64)
        return o, np.concatenate(ds).astype(np.float32)
    os.makedirs(f"{tmp}/corpus")
    for f in range(2):
        make_dummy.write_multivector(f"{tmp}/corpus/part-{f:05d}.parquet", parts(docs[20 * f:20 * f + 20]))
    qo, qt = make_dummy.random_multivector(np.random.default_rng(8), 4, 8, 3)
    qdocs = [docs[0]] + [qt[qo[r]:qo[r + 1]] for r in range(4)]
    qids = [f"q{i}" for i in range(len(qdocs))]
    make_dummy.write_multivector(f"{tmp}/q.parquet", parts(qdocs),
                                 extra={"query_id": pa.array(qids)})
    keys, _, offsets, _ = cmp.corpus_index(f"{tmp}/corpus", need_ids=False)
    q, _ = multivec.load_queries(f"{tmp}/q.parquet", "multivector_embedding")
    ex = cmp.exact_scores_mv([set(range(40)) for _ in qdocs], offsets, keys,
                             pafs.LocalFileSystem(), "multivector_embedding", q, metric,
                             threads=2)
    return qids, keys, offsets, ex, ["--modality", "multivector", "--vcol",
                                     "multivector_embedding", "--qcol", "multivector_embedding"]


def write_novabf(path, qids, keys, offsets, lists):
    def uuid(g):
        f = int(np.searchsorted(offsets, g, side="right") - 1)
        return cmp.uuid_of(keys[f], int(g - offsets[f]))
    pq.write_table(pa.table({
        "query_id": qids,
        "hit_ids": pa.array([[uuid(g) for g, _ in L] for L in lists], pa.list_(pa.string())),
        "hit_scores": pa.array([[float(np.float32(s)) for _, s in L] for L in lists],
                               pa.list_(pa.float32()))}), path)


def write_npz(path, lists, k):
    I = np.full((len(lists), k), -1, np.int64)
    D = np.full((len(lists), k), LOWEST, np.float32)
    for q, L in enumerate(lists):
        for j, (g, s) in enumerate(L):
            I[q, j], D[q, j] = g, s
    np.savez(path, I=I, D=D)


def run_compare(tmp, extra, *args):
    p = subprocess.run([sys.executable, *COMPARE, "--corpus", f"{tmp}/corpus", "--novabf",
                        f"{tmp}/nb.parquet", "--harness", f"h={tmp}/h.npz", "--k", str(K),
                        "--rescore", "--queries", f"{tmp}/q.parquet", *extra,
                        "--json", f"{tmp}/o.json", *args],
                       capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"), cwd=CB)
    j = json.load(open(f"{tmp}/o.json")) if os.path.exists(f"{tmp}/o.json") else None
    if os.path.exists(f"{tmp}/o.json"):
        os.remove(f"{tmp}/o.json")
    return p, j


BUILD = {"dense": dense_corpus, "sparse": sparse_corpus, "multivector": mv_corpus}


@pytest.mark.slow
@pytest.mark.parametrize("modality,metric", [("dense", "ip"), ("dense", "cosine"),
                                             ("sparse", "ip"), ("multivector", "ip")])
def test_end_to_end(tmp_path, modality, metric):
    tmp = str(tmp_path)
    qids, keys, offsets, ex, extra = BUILD[modality](tmp, metric)
    extra = extra + ["--metric", metric]
    nq = len(qids)
    n = int(offsets[-1])

    def topk(q):
        order = sorted(range(n), key=lambda i: (-ex[(q, i)], i))
        return [(i, ex[(q, i)]) for i in order[:K]]
    lists = [topk(q) for q in range(nq)]
    # the near-tied pair: rows 0 and 1, both in query 0's top-k, a nonzero gap within the window
    pos = {g: j for j, (g, _) in enumerate(lists[0])}
    assert 0 in pos and 1 in pos
    e0, e1 = ex[(0, 0)], ex[(0, 1)]
    gap = abs(e0 - e1) / ulp(max(abs(e0), abs(e1)))
    assert 0 < gap < 8, (e0, e1, gap)
    # query 1's ranks 2 and 6: clearly different (far outside the window)
    s2, s6 = lists[1][2][1], lists[1][6][1]
    assert abs(s2 - s6) / ulp(max(abs(s2), abs(s6))) > 1000

    def swap(ls, q, i, j):
        out = [list(L) for L in ls]
        out[q][i], out[q][j] = out[q][j], out[q][i]
        return out

    write_novabf(f"{tmp}/nb.parquet", qids, keys, offsets, lists)
    # 1. identical: PASS, nothing mismatched
    write_npz(f"{tmp}/h.npz", lists, K)
    p, j = run_compare(tmp, extra)
    assert p.returncode == 0, p.stdout + p.stderr
    r = j["pairs"]["novabf_vs_h"]
    assert r["f64_order_mismatched_ranks"] == 0 and j["ok"] and j["order_check"]
    assert r["at_k"]["set_equal"] == 1.0 and "F64 ORDER REAL 0" in p.stdout
    assert j["novabf_unmapped_hits"] == 0
    # 2. the harness swaps the near-tied pair: a tie swap at both ranks, PASS
    write_npz(f"{tmp}/h.npz", swap(lists, 0, pos[0], pos[1]), K)
    p, j = run_compare(tmp, extra)
    assert p.returncode == 0, p.stdout + p.stderr
    r = j["pairs"]["novabf_vs_h"]
    assert r["f64_order_tie_swaps"] == 2 and r["f64_order_real_disagreements"] == 0
    assert r["f64_order_max_tie_gap_ulps"] == pytest.approx(gap)
    # 3. the harness swaps two clearly different ids: a real order disagreement, FAIL
    bad = swap(lists, 1, 2, 6)
    write_npz(f"{tmp}/h.npz", bad, K)
    p, j = run_compare(tmp, extra)
    assert p.returncode == 1, p.stdout + p.stderr
    r = j["pairs"]["novabf_vs_h"]
    assert r["f64_order_real_disagreements"] == 2 and r["f64_order_tie_swaps"] == 0
    assert r["f64_real_disagreements"] == 0          # same sets: only the order is wrong
    assert {(e["query"], e["rank"]) for e in r["f64_order_examples"]} == {(1, 2), (1, 6)}
    assert r["f64_order_inversions_by_system"] == {"novabf": 0, "h": 1}
    assert j["ok"] is False and j["f64_order_failed_pairs"] == {"novabf_vs_h": 2}
    assert "FAIL novabf_vs_h: 2 float64-checked ORDER" in p.stdout
    assert "float64 order disagreements: novabf_vs_h 2" in p.stderr
    # ... and --no-order-check does not look at the order (exit 0, no order fields)
    p, j = run_compare(tmp, extra, "--no-order-check")
    assert p.returncode == 0 and not j["order_check"]
    assert "f64_order_real_disagreements" not in j["pairs"]["novabf_vs_h"]
    # 4. short lists (a filtered query with fewer than k allowed rows): nova-bf returns
    # the first 6, the harness the same 6 then -1 padding -> PASS; a clear swap inside
    # the short list still FAILs
    short = [L[:6] for L in lists]
    write_novabf(f"{tmp}/nb.parquet", qids, keys, offsets, short)
    write_npz(f"{tmp}/h.npz", short, K)
    p, j = run_compare(tmp, extra)
    assert p.returncode == 0, p.stdout + p.stderr
    r = j["pairs"]["novabf_vs_h"]
    assert r["queries_short_of_k"] == nq and r["f64_order_mismatched_ranks"] == 0
    write_npz(f"{tmp}/h.npz", swap(short, 1, 2, 5), K)
    p, j = run_compare(tmp, extra)
    assert p.returncode == 1
    r = j["pairs"]["novabf_vs_h"]
    assert r["f64_order_real_disagreements"] == 2 and r["f64_real_disagreements"] == 0


# ------------------------------------------- filtered runs: corpus ids + a rows subset

@pytest.mark.slow
def test_corpus_ids_and_query_subset(tmp_path):
    """runs/filtered's shape: a corpus WITH an `id` column (nova-bf config id_column: id, so
    hit_ids are those ids, --novabf-ids corpus) and a search whose `rows` selector keeps
    only some queries of the shared file (--query-rows). Like the MS MARCO shared file, one
    query_id is reused across query sets for a DIFFERENT query (rows 0 and 3), so only the
    rows selector, not the ids, picks the right vectors."""
    tmp = str(tmp_path)
    rng = np.random.default_rng(11)
    x = make_dummy.unit_vectors(rng, 40, 8)
    os.makedirs(f"{tmp}/corpus")
    cid = [f"<urn:doc:{i:03d}>" for i in range(40)]
    for f in range(2):
        pq.write_table(pa.table({"id": cid[20 * f:20 * f + 20],
                                 "embedding": make_dummy.vectors_table(x[20 * f:20 * f + 20])
                                 .column("embedding")}), f"{tmp}/corpus/part-{f:05d}.parquet")
    xq = make_dummy.unit_vectors(rng, 6, 8)
    qids = ["q3", "q1", "q2", "q3", "q4", "q5"]
    sets = ["B", "A", "B", "A", "A", "B"]
    pq.write_table(pa.table({"query_id": qids, "query_set": sets,
                             "embedding": make_dummy.vectors_table(xq).column("embedding")}),
                   f"{tmp}/q.parquet")
    keep = [1, 3, 4]                      # query_set == A, file order
    S = xq.astype(np.float64) @ x.astype(np.float64).T

    def topk(q):
        o = sorted(range(40), key=lambda i: (-S[q, i], i))[:K]
        return [(i, S[q, i]) for i in o]
    lists = [topk(q) for q in keep]

    def nb(path, qs, ls):
        pq.write_table(pa.table({
            "query_id": qs,
            "hit_ids": pa.array([[cid[g] for g, _ in L] for L in ls], pa.list_(pa.string())),
            "hit_scores": pa.array([[float(np.float32(s)) for _, s in L] for L in ls],
                                   pa.list_(pa.float32()))}), path)
    extra = ["--vcol", "embedding", "--qcol", "embedding", "--metric", "ip"]
    sub = ["--novabf-ids", "corpus", "--query-rows", "query_set=A"]
    nb(f"{tmp}/nb.parquet", [qids[q] for q in keep], lists)
    write_npz(f"{tmp}/h.npz", lists, K)
    # 1. identical: PASS, every hit mapped
    p, j = run_compare(tmp, extra, *sub)
    assert p.returncode == 0, p.stdout + p.stderr
    assert j["novabf_unmapped_hits"] == 0 and j["queries"] == 3
    assert j["pairs"]["novabf_vs_h"]["at_k"]["set_equal"] == 1.0
    # 2. the harness drops query 3's best id for a worse one: a real float64 disagreement
    bad = [list(L) for L in lists]
    worst = sorted(range(40), key=lambda i: (-S[3, i], i))[K]
    bad[1][0] = (worst, S[3, worst])
    write_npz(f"{tmp}/h.npz", bad, K)
    p, j = run_compare(tmp, extra, *sub)
    assert p.returncode == 1 and j["pairs"]["novabf_vs_h"]["f64_real_disagreements"] >= 1
    # 3. without the flags: uuid mapping finds nothing (FAIL), and the queries file's
    # order does not match nova-bf's subset (refused)
    write_npz(f"{tmp}/h.npz", lists, K)
    p, _ = run_compare(tmp, extra, "--query-rows", "query_set=A")
    assert p.returncode == 1 and "unmapped" in p.stdout
    p, _ = run_compare(tmp, extra, "--novabf-ids", "corpus")
    assert p.returncode != 0 and "--queries order != nova-bf query order" in p.stderr
    # 4. nova-bf's queries out of file order, or a harness with other rows: refused
    nb(f"{tmp}/nb.parquet", [qids[q] for q in (3, 1, 4)], [lists[1], lists[0], lists[2]])
    p, _ = run_compare(tmp, extra, *sub)
    assert p.returncode != 0 and "--queries order != nova-bf query order" in p.stderr
    # ... and the wrong selector (set B holds q3 too, a different query): refused
    p, _ = run_compare(tmp, extra, "--novabf-ids", "corpus", "--query-rows", "query_set=B")
    assert p.returncode != 0 and "--queries order != nova-bf query order" in p.stderr
    nb(f"{tmp}/nb.parquet", [qids[q] for q in keep], lists)
    write_npz(f"{tmp}/h.npz", lists[:2], K)
    p, _ = run_compare(tmp, extra, *sub)
    assert p.returncode != 0 and "result rows, nova-bf 3" in p.stderr
    # 5. duplicate corpus ids: refused (a hit id would not name one row)
    t = pq.read_table(f"{tmp}/corpus/part-00001.parquet")
    pq.write_table(t.set_column(0, "id", pa.array([cid[0]] + cid[21:40])),
                   f"{tmp}/corpus/part-00001.parquet")
    write_npz(f"{tmp}/h.npz", lists, K)
    p, _ = run_compare(tmp, extra, *sub)
    assert p.returncode != 0 and "duplicate ids" in p.stderr


def test_uuid_table_is_uuid_of_for_every_row():
    """DuckDB's bulk md5 gives nova-bf's point id (uuid_of) for every row, keys with spaces,
    unicode and slashes included; uuid_gids maps hit ids back to global rows (-1: unknown)."""
    import pyarrow as pa
    keys = ["b/x 1.parquet", "a/ünï/ç.parquet", "c.parquet", "empty.parquet"]
    rows = np.array([3, 5, 1, 0], np.int64)
    offsets = np.concatenate([[0], np.cumsum(rows)])
    t = cmp.uuid_table(keys, rows, offsets)
    want = [cmp.uuid_of(k, r) for k, n in zip(keys, rows) for r in range(n)]
    assert t.to_pylist() == want
    hits = pa.array([want[4], "not-an-id", want[0], want[8]])
    assert cmp.uuid_gids(hits, t).tolist() == [4, -1, 0, 8]
