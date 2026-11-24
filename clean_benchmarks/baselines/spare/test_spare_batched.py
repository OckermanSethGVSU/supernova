"""Batched SPARe vs SPARe as released vs an independent float64 oracle.

    <paper env python> -m pytest -q test_spare_batched.py

Runs on CPU (and on GPU with NOVA_SPARE_DEVICE=cuda). Needs the `spare`
package at commit e7f4ad1 (see baselines/README.md).

On CPU the batched modes are checked bitwise against SPARe as released wherever
that holds (iterative, all objectives); on GPU, where index_add_ is atomic in
both, the same checks are tie-aware to float rounding.
"""

import os
import warnings

import numpy as np
import pytest
import scipy.sparse as ssp
import torch

import spare_batched as sb
from spare_batched import (BatchedDot, BatchedIterative, CsrCollection, _group_queries,
                           queries_to_padded, run_released)

DEV = os.environ.get("NOVA_SPARE_DEVICE", "cpu")
CPU = DEV == "cpu"
REL_GAP = 1e-5  # tie zone at the k-th score: |s - s_k| <= REL_GAP * max(1, |s_k|)


# --------------------------------------------------------------------- data

def make_data(seed=0, n_docs=3000, vocab=400, nnz_doc=20, n_q=64, max_q_terms=12, zipf=1.1,
              rare_queries=0):
    """Non-negative learned-sparse-like data: Zipf-ish term frequencies (a few
    terms in most documents, long tail), random float32 weights. The last
    `rare_queries` queries use only the rarest terms (few matching documents,
    so at large k most of their top-k are zero-score ties)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** zipf
    p /= p.sum()
    rows, cols = [], []
    for d in range(n_docs):
        c = np.unique(rng.choice(vocab, size=rng.integers(1, nnz_doc + 1), p=p))
        rows += [d] * len(c)
        cols += c.tolist()
    vals = rng.uniform(0.01, 3.0, len(rows)).astype(np.float32)
    X = ssp.csr_matrix((vals, (rows, cols)), shape=(n_docs, vocab), dtype=np.float32)
    X.sort_indices()
    df = np.diff(X.tocsc().indptr)
    rare = np.argsort(df)[: max(5, vocab // 20)]
    rare = rare[df[rare] > 0]
    queries = []
    for j in range(n_q):
        if j >= n_q - rare_queries:
            terms = np.unique(rng.choice(rare, size=int(rng.integers(1, 4))))
        else:
            terms = np.unique(rng.choice(vocab, size=int(rng.integers(1, max_q_terms + 1)), p=p))
        queries.append({int(t): float(w) for t, w in
                        zip(terms, rng.uniform(0.01, 2.0, len(terms)).astype(np.float32))})
    queries.append({})                     # empty query: scores all zero
    return X, queries


def oracle_scores(X, queries):
    """float64 dense scores, independent of SPARe and torch."""
    Q = np.zeros((len(queries), X.shape[1]))
    for i, q in enumerate(queries):
        for t, w in q.items():
            Q[i, t] += np.float32(w)
    return Q @ X.astype(np.float64).toarray().T


def collection(X, index_dtype=torch.int32):
    # built on CPU, the model moved afterwards (as SPARe does)
    return CsrCollection.from_csr(X.indptr, X.indices, X.data, X.shape[1],
                                  index_dtype=index_dtype)


def model(M, X, **kw):
    return M(collection(X), **kw).to(DEV)


# --------------------------------------------------------------------- checks

def assert_topk_correct(scores, ids, S_full, k, atol=1e-4):
    """Strict exact-top-k check, tolerant only of genuine ties: each row has k
    distinct valid ids; returned scores are descending and equal to the true
    sorted top-k scores; each returned id's true score equals its returned
    score; every document strictly above the tie zone at the k-th score is
    returned and nothing below it is."""
    scores, ids = scores.cpu().double().numpy(), ids.cpu().numpy()
    assert scores.shape == ids.shape == (S_full.shape[0], k)
    for r in range(S_full.shape[0]):
        true = S_full[r]
        assert len(set(ids[r].tolist())) == k, f"query {r}: duplicate ids {ids[r][:20]}"
        assert ids[r].min() >= 0 and ids[r].max() < true.shape[0], f"query {r}: id out of range"
        assert np.all(np.diff(scores[r]) <= 1e-6 * max(1.0, abs(scores[r]).max())), \
            f"query {r}: scores not descending"
        top = np.sort(true)[::-1][:k]
        np.testing.assert_allclose(scores[r], top, atol=atol, rtol=1e-5,
                                   err_msg=f"query {r}: scores != true top-k scores")
        np.testing.assert_allclose(true[ids[r]], scores[r], atol=atol, rtol=1e-5,
                                   err_msg=f"query {r}: returned scores != true scores")
        kth = top[-1]
        gap = REL_GAP * max(1.0, abs(kth))
        must = set(np.nonzero(true > kth + gap)[0].tolist())
        got = set(ids[r].tolist())
        assert must <= got, f"query {r}: missing {sorted(must - got)[:5]}"
        assert all(true[i] >= kth - gap for i in got), f"query {r}: returned a non-top-k id"


def assert_same_as_released(s1, i1, s2, i2, atol=1e-5, bitwise=None):
    """Batched vs released. bitwise=True: identical scores and ids. Otherwise:
    scores equal to `atol`, k distinct ids per row in both, and ids differing
    only inside runs of (near-)equal scores."""
    if bitwise is None:
        bitwise = False
    if bitwise:
        assert torch.equal(s1.cpu(), s2.cpu()), "scores differ from released"
        assert torch.equal(i1.cpu(), i2.cpu()), "ids differ from released"
        return
    s1, s2 = s1.cpu().double().numpy(), s2.cpu().double().numpy()
    i1, i2 = i1.cpu().numpy(), i2.cpu().numpy()
    np.testing.assert_allclose(s1, s2, atol=atol, rtol=0)
    for r in range(s1.shape[0]):
        assert len(set(i1[r].tolist())) == s1.shape[1], f"query {r}: duplicate ids (batched)"
        assert len(set(i2[r].tolist())) == s1.shape[1], f"query {r}: duplicate ids (released)"
        for c in np.nonzero(i1[r] != i2[r])[0]:
            assert np.sum(np.abs(s1[r] - s1[r][c]) <= atol) > 1, f"query {r} rank {c}: id differs"


def note_bitwise(a, b, what):
    """fp32 vs released: bitwise equality is observed on CPU but not part of
    the contract (summation order is not guaranteed). Reported, never failed."""
    if CPU and not torch.equal(a.cpu(), b.cpu()):
        warnings.warn(f"{what}: scores not bitwise equal to SPARe's on CPU (allowed)")


# --------------------------------------------------------------------- tests

@pytest.mark.parametrize("k", [1, 10, 100])
def test_iterative_matches_released_and_oracle(k):
    X, queries = make_data()
    S = oracle_scores(X, queries)
    m = model(BatchedIterative, X, objective="accuracy")
    t, w = queries_to_padded(queries, DEV)
    s_b, i_b = m.search(t, w, k)
    s_r, i_r = run_released(m, queries, k)
    note_bitwise(s_b, s_r, f"iterative k={k}")
    assert_same_as_released(s_b, i_b, s_r, i_r)
    assert_topk_correct(s_b, i_b, S, k)


@pytest.mark.parametrize("k", [1, 10, 100])
def test_dot_matches_released_and_oracle(k):
    X, queries = make_data(seed=1)
    S = oracle_scores(X, queries)
    m = model(BatchedDot, X)
    t, w = queries_to_padded(queries, DEV)
    s_b, i_b = m.search(t, w, k)
    s_r, i_r = run_released(m, queries, k)
    assert_same_as_released(s_b, i_b, s_r, i_r)
    assert_topk_correct(s_b, i_b, S, k)


@pytest.mark.parametrize("M", [BatchedIterative, BatchedDot])
def test_k1000_with_many_zero_score_ties(M):
    """Production-like k: several queries match fewer than k documents, so
    their top-k is mostly zero-score ties. Ids must still be k distinct
    documents with the right scores."""
    X, queries = make_data(seed=13, n_docs=2500, n_q=40, rare_queries=10)
    S = oracle_scores(X, queries)
    assert sum(int((row > 0).sum()) < 1000 for row in S) >= 5
    m = model(M, X)
    t, w = queries_to_padded(queries, DEV)
    s_b, i_b = m.search(t, w, 1000)
    assert_topk_correct(s_b, i_b, S, 1000)
    s_r, i_r = run_released(m, queries, 1000)
    assert_same_as_released(s_b, i_b, s_r, i_r)


@pytest.mark.parametrize("M,kw", [(BatchedIterative, {"max_bytes": 1}),
                                  (BatchedIterative, {"max_bytes": 3000 * 4 * 5 + 64 * 9000,
                                                      "max_batch": 3}),
                                  (BatchedIterative, {"max_batch": 7}),
                                  (BatchedDot, {"max_batch": 1}), (BatchedDot, {"max_batch": 9}),
                                  (BatchedDot, {"max_bytes": 1})])
def test_multi_group_runs_against_oracle_and_single_group(M, kw):
    """Forced splitting (many groups / batches, single queries over budget):
    results stay in query order, correct against the oracle, and equal the
    one-group run (bitwise on CPU for iterative)."""
    X, queries = make_data(seed=2, n_q=50)
    S = oracle_scores(X, queries)
    m = model(M, X)
    t, w = queries_to_padded(queries, DEV)
    s0, i0 = m.search(t, w, 20)
    s1, i1 = m.search(t, w, 20, **kw)
    assert_topk_correct(s1, i1, S, 20)
    assert_same_as_released(s1, i1, s0, i0, bitwise=CPU and M is BatchedIterative)


@pytest.mark.parametrize("objective", ["half", "performance"])
def test_low_precision_objectives_match_released(objective):
    """SPARe's fp16 / uint8 accumulators (lossy by design): batched must
    reproduce the released iterative mode (not the oracle). Bitwise, scores
    and ids, on CPU; on GPU a few accumulator ulps."""
    X, queries = make_data(seed=6, n_q=30)
    m = model(BatchedIterative, X, objective=objective)
    t, w = queries_to_padded(queries, DEV)
    s_b, i_b = m.search(t, w, 10)
    s_r, i_r = run_released(m, queries, 10)
    assert s_b.dtype == s_r.dtype == {"half": torch.float16, "performance": torch.uint8}[objective]
    if CPU:
        assert_same_as_released(s_b, i_b, s_r, i_r, bitwise=True)
    else:
        assert_same_as_released(s_b, i_b, s_r, i_r,
                                atol=2.0 if objective == "performance" else 6e-2)


def test_k_equals_n_docs_and_padding_width():
    X, queries = make_data(seed=4, n_docs=60, n_q=10)
    S = oracle_scores(X, queries)
    for M in (BatchedIterative, BatchedDot):
        m = model(M, X)
        t, w = queries_to_padded(queries, DEV)
        # extra all-padding columns must change nothing
        t2 = torch.cat([t, torch.full((t.shape[0], 5), -1, dtype=t.dtype, device=t.device)], 1)
        w2 = torch.cat([w, torch.zeros((w.shape[0], 5), dtype=w.dtype, device=w.device)], 1)
        s, i = m.search(t, w, 60)
        s2, i2 = m.search(t2, w2, 60)
        assert_same_as_released(s2, i2, s, i, bitwise=CPU and M is BatchedIterative)
        assert_topk_correct(s, i, S, 60)


def test_padding_with_nonzero_weights_is_ignored():
    X, _ = make_data(seed=14, n_q=1)
    for M in (BatchedIterative, BatchedDot):
        m = model(M, X)
        s, i = m.search(torch.tensor([[1, -1, 4, -1]], device=DEV),
                        torch.tensor([[1.0, 5.0, 0.5, float("nan")]], device=DEV), 10)
        sr, ir = m.search(torch.tensor([[1, 4]], device=DEV),
                          torch.tensor([[1.0, 0.5]], device=DEV), 10)
        assert_same_as_released(s, i, sr, ir)


def test_duplicate_query_terms_are_summed():
    """SPARe's queries are dicts (no duplicates); our padded form could carry a
    term twice. Both modes must sum duplicates, i.e. equal the merged query."""
    X, _ = make_data(seed=5, n_q=1)
    t = torch.tensor([[3, 3, 7, -1]], device=DEV)
    w = torch.tensor([[0.5, 0.25, 1.0, 0.0]], device=DEV)
    tm = torch.tensor([[3, 7]], device=DEV)
    wm = torch.tensor([[0.75, 1.0]], device=DEV)
    for M in (BatchedIterative, BatchedDot):
        m = model(M, X)
        s, i = m.search(t, w, 10)
        sm, im = m.search(tm, wm, 10)
        assert_same_as_released(s, i, sm, im)


def test_long_queries_and_float64_weights():
    """ArguAna-like long queries (up to 200 terms over a 600-term vocab), with
    the weights handed in as float64: must still match SPARe and the oracle."""
    X, queries = make_data(seed=11, n_docs=1500, vocab=600, n_q=20, max_q_terms=200, zipf=0.6)
    S = oracle_scores(X, queries)
    t, w = queries_to_padded(queries, DEV, dtype=torch.float64)
    assert t.shape[1] > 100
    for M, kw in ((BatchedIterative, {"objective": "accuracy"}), (BatchedDot, {})):
        m = model(M, X, **kw)
        s_b, i_b = m.search(t, w, 50)
        s_r, i_r = run_released(m, queries, 50)
        assert_same_as_released(s_b, i_b, s_r, i_r)
        assert_topk_correct(s_b, i_b, S, 50)


def test_term_in_every_document_int32_noncontiguous_inputs_and_int64_indices():
    X, queries = make_data(seed=15, n_docs=800, n_q=12)
    X = X.tolil()
    X[:, 0] = 0.5                          # term 0 in every document
    X = X.tocsr()
    X.sort_indices()
    queries = [{**q, 0: 1.25} for q in queries]
    S = oracle_scores(X, queries)
    t, w = queries_to_padded(queries, DEV)
    t32 = t.to(torch.int32).T.contiguous().T          # int32 and non-contiguous
    w_nc = w.T.contiguous().T
    for M in (BatchedIterative, BatchedDot):
        for idt in (torch.int32, torch.int64):
            m = M(collection(X, index_dtype=idt)).to(DEV)
            s, i = m.search(t32, w_nc, 30)
            assert_topk_correct(s, i, S, 30)


def test_empty_batch_and_out_device():
    X, queries = make_data(seed=9, n_docs=40, n_q=4)
    for M in (BatchedIterative, BatchedDot):
        m = model(M, X)
        s, i = m.search(torch.empty((0, 3), dtype=torch.int64, device=DEV),
                        torch.empty((0, 3), dtype=torch.float32, device=DEV), 5)
        assert s.shape == (0, 5) and i.shape == (0, 5) and i.dtype == torch.int64
        t, w = queries_to_padded(queries, DEV)
        s, i = m.search(t, w, 5, out_device="cpu", max_batch=2)
        assert s.device.type == "cpu" and i.device.type == "cpu" and s.shape == (5, 5)


def test_invalid_inputs_rejected():
    X, _ = make_data(seed=10, n_docs=50, n_q=3)
    vocab = X.shape[1]
    w2 = torch.ones((2, 2), device=DEV)
    for M in (BatchedIterative, BatchedDot):
        m = model(M, X)
        for bad in ([[0, 1], [0, vocab]], [[0, 1], [0, -2]], [[vocab + 7, 1], [0, 1]]):
            with pytest.raises(ValueError):
                m.search(torch.tensor(bad, device=DEV), w2, 5)
        with pytest.raises(TypeError):                    # float term ids
            m.search(torch.tensor([[0.0, 1.0]], device=DEV), w2[:1], 5)
        with pytest.raises(TypeError):                    # integer weights
            m.search(torch.tensor([[0, 1]], device=DEV), torch.tensor([[1, 2]], device=DEV), 5)
        t, w = torch.tensor([[0, 1]], device=DEV), w2[:1]
        for kw in ({"max_bytes": 0}, {"max_batch": 0}, {"max_bytes": -5}, {"max_batch": 2.5}):
            with pytest.raises(ValueError):
                m.search(t, w, 5, **kw)
        for k in (0, 51, True, 2.0):
            with pytest.raises(ValueError):
                m.search(t, w, k)
        m.search(t, w, np.int64(5), max_batch=np.int32(4))   # numpy ints are fine
        with pytest.raises(ValueError):
            m.search(t, torch.ones((1, 3), device=DEV), 5)    # shape mismatch
    with pytest.raises(ValueError):
        queries_to_padded([{-1: 0.5}])
    queries_to_padded([{np.int64(3): 0.5}])                     # numpy ints are fine


@pytest.mark.parametrize("dt", [torch.float16, torch.uint8])
def test_iterative_refuses_non_float32_collections(dt):
    """SPARe's per-posting product for these follows promotion with its 0-d
    float32 weight and differs CPU vs GPU; refused rather than approximated."""
    X, queries = make_data(seed=16, n_docs=30, n_q=2)
    coll = CsrCollection.from_csr(X.indptr, X.indices, (X.data * 10).astype(np.float32),
                                  X.shape[1], dtype=dt)
    m = BatchedIterative(coll).to(DEV)
    t, w = queries_to_padded(queries, DEV)
    with pytest.raises(TypeError):
        m.search(t, w, 5)


def test_grouping_respects_budgets():
    """_group_queries: groups cover every query once, in order, within the
    cost and batch budgets (a single query over budget gets its own group)."""
    rng = np.random.default_rng(8)
    for _ in range(300):
        costs = torch.from_numpy(rng.integers(0, 50, int(rng.integers(1, 40))))
        max_cost, max_batch = int(rng.integers(1, 120)), int(rng.integers(1, 10))
        groups = _group_queries(costs, max_cost, max_batch)
        assert groups[0][0] == 0 and groups[-1][1] == len(costs)
        assert all(a[1] == b[0] for a, b in zip(groups, groups[1:]))
        for lo, hi in groups:
            assert 0 < hi - lo <= max_batch
            assert hi - lo == 1 or int(costs[lo:hi].sum()) <= max_cost


def test_iterative_plan_counts_accumulator_and_postings():
    """BatchedIterative.plan: cost includes both the posting scratch and the
    (B, n_docs) accumulator, so accumulator-dominated data (many documents,
    tiny queries) is split by the accumulator."""
    X, _ = make_data(seed=17, n_docs=5000, n_q=1)
    m = model(BatchedIterative, X)                        # fp32 accumulator
    postings = torch.tensor([3, 0, 5, 2, 1, 4] * 5, dtype=torch.int64)
    acc = 5000 * 4
    for idt, per in (("int64", m.BYTES_PER_POSTING), ("auto", m.BYTES_PER_POSTING_INT32)):
        for max_bytes in (acc, 2 * acc + 64 * 6, 7 * acc + 1000, 10**9):
            groups, cost = m.plan(postings, max_bytes, 4096, idt)
            assert torch.equal(cost, postings * per + acc)
            for lo, hi in groups:
                assert hi - lo == 1 or int(cost[lo:hi].sum()) <= max_bytes
            if max_bytes == acc:
                assert all(hi - lo == 1 for lo, hi in groups)   # accumulator alone fills it


def test_group_queries_count_cap():
    """The int32 path's second cap: summed counts per group <= max_count
    (a single query over it alone)."""
    costs = torch.zeros(6, dtype=torch.int64)
    counts = torch.tensor([3, 3, 5, 9, 1, 1])
    g = _group_queries(costs, 10, 100, counts, 6)
    assert g == [(0, 2), (2, 3), (3, 4), (4, 6)]


def test_dot_batch_size_counts_everything():
    X, _ = make_data(seed=18, n_docs=2000, vocab=400, n_q=1)
    m = model(BatchedDot, X)
    per_query = (400 + m.DOT_DOCS_FACTOR * 2000) * 4
    for max_bytes, max_batch in ((per_query * 7 + 3, 100), (per_query - 1, 100),
                                 (10**12, 9), (per_query * 50, 4096)):
        b = m.batch_size(max_bytes, max_batch)
        assert 1 <= b <= max_batch
        assert b == 1 or b * per_query <= max_bytes
        assert b == max_batch or (b + 1) * per_query > max_bytes


def test_run_released_requires_model_on_spare_device():
    from spare.backend_torch import TorchBackend
    X, queries = make_data(seed=19, n_docs=30, n_q=2)
    m = BatchedIterative(collection(X)).to("meta" if TorchBackend().devices[0] != "meta" else "cpu")
    with pytest.raises(RuntimeError):
        run_released(m, queries, 5)


@pytest.mark.skipif(CPU, reason="peak-memory accounting needs CUDA")
@pytest.mark.parametrize("case", ["long_queries", "accumulator_dominated"])
@pytest.mark.parametrize("k", [100, 1000])
def test_memory_budget_holds_on_gpu(case, k):
    """Measured peak scratch (beyond model + inputs) stays within max_bytes,
    for posting-dominated and accumulator-dominated data, generous and tight
    budgets."""
    if case == "long_queries":
        X, queries = make_data(seed=12, n_docs=200_000, vocab=2000, nnz_doc=60, n_q=300,
                               max_q_terms=40)
    else:
        X, queries = make_data(seed=20, n_docs=400_000, vocab=20000, nnz_doc=30, n_q=300,
                               max_q_terms=3, zipf=0.3)
    t, w = queries_to_padded(queries, DEV)
    for M in (BatchedIterative, BatchedDot):
        m = model(M, X)
        for budget in (256 << 20, 48 << 20):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            m.search(t, w, k, max_bytes=budget, out_device="cpu")
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            assert peak <= budget, f"{M.__name__} {case} k={k}: peak {peak} > budget {budget}"


@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_dot_topk_layouts_agree(layout):
    """Both top-k layouts give the exact top-k (and the same as each other)."""
    X, Q = make_data(seed=41, n_docs=900, vocab=300, n_q=37)
    m = model(BatchedDot, X)
    ids, w = queries_to_padded(Q, device=DEV)
    s, i = m.search(ids, w, 10, max_bytes=40_000, topk_layout=layout)
    s0, i0 = m.search(ids, w, 10, max_bytes=40_000, topk_layout="strided")
    torch.testing.assert_close(s, s0)
    assert_topk_correct(s, i, oracle_scores(X, Q), 10)


@pytest.mark.parametrize("index_dtype", ["auto", "int64"])
def test_iterative_index_dtypes_and_given_postings(index_dtype):
    """int32 / int64 expansion indices give the same exact top-k; postings
    passed in (the harness's precomputed counts) give the same result, and
    wrong ones fail loudly."""
    X, Q = make_data(seed=43, n_docs=800, vocab=250, n_q=29)
    m = model(BatchedIterative, X)
    ids, w = queries_to_padded(Q, device=DEV)
    q = sb.CsrQueries.from_padded(ids, w)
    s, i = m.search_csr(q, 10, max_bytes=30_000, index_dtype=index_dtype)
    assert_topk_correct(s, i, oracle_scores(X, Q), 10)
    post = m.query_postings(q).cpu()
    s2, i2 = m.search_csr(q, 10, max_bytes=30_000, index_dtype=index_dtype, postings=post)
    torch.testing.assert_close(s2, s)
    assert torch.equal(i2, i)
    bad = post.clone()
    bad[3] += 1
    with pytest.raises(RuntimeError):
        m.search_csr(q, 10, max_bytes=30_000, index_dtype=index_dtype, postings=bad)


def test_csr_queries_from_csr_matches_padded():
    X, Q = make_data(seed=47, n_docs=300, vocab=120, n_q=11)
    ids, w = queries_to_padded(Q)
    a = sb.CsrQueries.from_padded(ids, w)
    indptr = np.concatenate([[0], np.cumsum([len(d) for d in Q])])
    terms = np.array([t for d in Q for t in d], np.int64)
    vals = np.array([v for d in Q for v in d.values()], np.float32)
    b = sb.CsrQueries.from_csr(indptr, terms, vals)
    assert (a.indptr == b.indptr).all()
    for f in ("terms", "weights", "rows", "indptr_dev"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    with pytest.raises(ValueError):
        sb.CsrQueries.from_csr(np.array([0, 3, 2]), terms[:2], vals[:2])


def test_released_nosync_bitwise_equals_released():
    """IterativeReleasedNoSync.forward_nosync (host bounds, device
    accumulator) gives SPARe's own released forward's result bit for bit
    (same slices, product, cast, index_add_ order and topk), incl. empty
    queries and duplicate terms."""
    from spare.backend_torch import CSRSparseRetrievalModelIterative
    X, Q = make_data(seed=53, n_docs=700, vocab=200, n_q=25)
    Q = Q + [{}, {5: 0.5, 7: 1.5}]
    ref = model(CSRSparseRetrievalModelIterative, X)
    ns = model(sb.IterativeReleasedNoSync, X)
    ns.host_bounds()
    for m in (ref, ns):
        m.top_k = 10
    for d in Q:
        if d:
            ind = torch.tensor(list(d.keys()), dtype=torch.int64).unsqueeze(0).to(DEV)
            val = torch.tensor(list(d.values()), dtype=torch.float32).unsqueeze(0).to(DEV)
        else:
            ind = torch.tensor([], dtype=torch.int32).unsqueeze(0).to(DEV)
            val = torch.tensor([], dtype=torch.int32).unsqueeze(0).to(DEV)
        a = ref(ind, val)
        b = ns.forward_nosync(list(d.keys()), val)
        if CPU:
            assert torch.equal(a.values, b.values) and torch.equal(a.indices, b.indices)
        else:   # GPU atomics: order of equal-score ids may differ
            torch.testing.assert_close(a.values, b.values)


def test_int32_policy_follows_the_budget():
    """int32 only when no group the budget allows could leave int32 range, so
    the int32 caps never shrink a group; larger budgets use int64."""
    X, _ = make_data(seed=59, n_docs=1000, n_q=1)
    m = model(BatchedIterative, X)
    assert m._int32("auto", 4 << 30)
    assert m._int32("auto", 8 << 30)           # 2^31 cells: indices 0 .. 2^31 - 1
    assert not m._int32("auto", (8 << 30) + 4)  # one cell more could leave int32
    assert not m._int32("int64", 1 << 20)


def test_dot_batch_size_charges_topk_sort_for_few_long_slices():
    """At <= 10 queries per batch over >= 100k rows CUDA topk sorts the
    whole slice: batch_size then charges the sort's extra buffers."""
    X, _ = make_data(seed=61, n_docs=1000, vocab=50, n_q=1)
    m = model(BatchedDot, X)
    m.shape = (200_000, 50)                      # only the planner reads it here
    per = (50 + m.DOT_DOCS_FACTOR * 200_000) * 4
    sorted_per = (50 + (m.DOT_DOCS_FACTOR + sb.TOPK_SORT_FACTOR) * 200_000) * 4
    assert m.batch_size(per * 5, 100) == max(1, (per * 5) // sorted_per)
    assert m.batch_size(per * 50, 100) == 50     # > 10 slices: radix select, no sort
