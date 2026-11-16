"""Block-max sub-block skipping (PROTOTYPE) on real hardware; skipped
without CUDA. The interpreted suite (`test_sparse_blockskip_interpret.py`)
covers the logic; this file covers what only a GPU can:

  - identity: every kept cell bit-identical to the unskipped hot kernel with
    real FMA contraction and barriers, every dropped cell strictly below the
    threshold, overhead mode bit-identical, the mask-aware top-k equal to the
    unmasked one, the first-launch checks passing;
  - FTZ margin case; the adversarial fp-model probe per variant (and its
    cost); poisoned run_compute differential;
  - zero-skip gates (finite thr = 0 kills nothing): skip posting kernel
    (overhead, on) <= 1.03x off and masked `_cutfill` <= 1.03x plain, prep
    printed; registers / spills of both new kernel variants;
  - timing (`test_time_one_slice`, run with `-s`): per-kernel CUDA-event
    times, setup / first-launch check / compile warm-up excluded, for
      off       the unskipped hot posting kernel,
      bound     BM build + bound kernel (`sparse_block_skip.prepare`),
      overhead  the skip posting kernel with every flag live,
      on        the skip posting kernel (elided),
      top-k     `_cutfill` on the unskipped matrix vs mask-aware on the
                elided one,
    plus the fractions of dead sub-blocks, fully dead (query, 512-block)
    programs and fully dead (query, slice) rows.

Real data (optional; otherwise a synthetic Zipf slice):
  NOVA_BF_BLOCKSKIP_CORPUS      parquet with a sparse struct column (indices, values)
  NOVA_BF_BLOCKSKIP_QUERIES     parquet with the same column (queries)
  NOVA_BF_BLOCKSKIP_COLUMN      column name (default sparse_embedding)
  NOVA_BF_BLOCKSKIP_ROWS        corpus rows in the slice (default 8192)
  NOVA_BF_BLOCKSKIP_NQ          queries (default 100000)
  NOVA_BF_BLOCKSKIP_THRESHOLDS  per-query k-th scores: .npy (float32, one per
                                query) or a ground-truth parquet whose
                                `hit_scores` lists give min() per query (the
                                true LATE threshold). Default: quantile
                                extrapolation as in oracle.py at 1f / 10f /
                                150f production files (FILE_ROWS rows each,
                                default 1.09M) over POOL_ROWS pooled corpus
                                rows (default: the whole file); a regime the
                                pool cannot represent (rank < 1) is REFUSED
                                and reported, never clamped.
  NOVA_BF_BLOCKSKIP_FILE_ROWS / _POOL_ROWS  see above
  NOVA_BF_BLOCKSKIP_SUB / _REPS sub-block width (64) / timed repetitions (10)
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from nova_bf import sparse_block_skip as bs  # noqa: E402
from nova_bf import sparse_posting_triton as sp  # noqa: E402
from nova_bf import topk_triton as tt  # noqa: E402
from nova_bf.tiebreak import pack  # noqa: E402

DEV = "cuda"


@pytest.fixture(autouse=True)
def _fresh():
    sp.reset()
    sp.set_policy("auto")
    yield
    sp.reset()


def _zipf_pos_csr(n, vocab, nnz_mean, g):
    w = 1.0 / torch.arange(1, vocab + 1, dtype=torch.float64) ** 1.1
    k = torch.poisson(torch.full((n,), float(nnz_mean)), generator=g).clamp(1, vocab).long()
    crow, cols, vals = [0], [], []
    for i in range(n):
        c = torch.multinomial(w, int(k[i]), replacement=False, generator=g).sort().values
        cols.append(c)
        vals.append((torch.rand(len(c), generator=g) + 1e-3).float())
        crow.append(crow[-1] + len(c))
    return torch.sparse_csr_tensor(
        torch.tensor(crow), torch.cat(cols), torch.cat(vals), size=(n, vocab)).to(DEV)


def _read_sparse(path, column, limit):
    """(crow, cols, vals) of the first `limit` rows; columns sorted per row."""
    import numpy as np
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=[column])[column].combine_chunks()[:limit]
    idx, val = col.field("indices"), col.field("values")
    offs = np.asarray(idx.offsets).astype(np.int64)
    offs = offs - offs[0]
    ind = np.asarray(idx.values)[: offs[-1]].astype(np.int64)
    vv = np.asarray(val.values)[: offs[-1]].astype(np.float32)
    row = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    order = np.lexsort((ind, row))
    return (torch.from_numpy(offs), torch.from_numpy(ind[order]),
            torch.from_numpy(vv[order]))


def _real_slice():
    cpath = os.environ.get("NOVA_BF_BLOCKSKIP_CORPUS")
    qpath = os.environ.get("NOVA_BF_BLOCKSKIP_QUERIES")
    if not (cpath and qpath):
        return None
    column = os.environ.get("NOVA_BF_BLOCKSKIP_COLUMN", "sparse_embedding")
    rows = int(os.environ.get("NOVA_BF_BLOCKSKIP_ROWS", "8192"))
    nq = int(os.environ.get("NOVA_BF_BLOCKSKIP_NQ", "100000"))
    cc, ci, cv = _read_sparse(cpath, column, rows)
    qc, qi, qv = _read_sparse(qpath, column, nq)
    vocab = int(max(ci.max(), qi.max())) + 1
    c = torch.sparse_csr_tensor(cc, ci, cv, size=(cc.numel() - 1, vocab)).to(DEV)
    q = torch.sparse_csr_tensor(qc, qi, qv, size=(qc.numel() - 1, vocab)).to(DEV)
    return q, c


def _resize(x, vocab):
    return torch.sparse_csr_tensor(x.crow_indices(), x.col_indices(), x.values(),
                                   size=(x.shape[0], vocab))


def _real_pool(q, c):
    """More corpus rows for threshold quantiles (NOVA_BF_BLOCKSKIP_POOL_ROWS,
    default: the whole file). Returns (q, c, pool) on one common vocabulary."""
    cpath = os.environ.get("NOVA_BF_BLOCKSKIP_CORPUS")
    column = os.environ.get("NOVA_BF_BLOCKSKIP_COLUMN", "sparse_embedding")
    rows = int(os.environ.get("NOVA_BF_BLOCKSKIP_POOL_ROWS", str(1 << 40)))
    cc, ci, cv = _read_sparse(cpath, column, rows)
    vocab = max(int(q.shape[1]), int(ci.max()) + 1)
    pool = torch.sparse_csr_tensor(cc, ci, cv, size=(cc.numel() - 1, vocab)).to(DEV)
    return _resize(q, vocab), _resize(c, vocab), pool


def _thresholds_from_env(n_q):
    import numpy as np

    p = os.environ.get("NOVA_BF_BLOCKSKIP_THRESHOLDS")
    if not p:
        return None
    if p.endswith(".npy"):
        t = np.load(p).astype(np.float32)[:n_q]
    else:
        import pyarrow.parquet as pq

        hs = pq.read_table(p, columns=["hit_scores"])["hit_scores"].to_pylist()[:n_q]
        t = np.array([min(h) if h else -np.inf for h in hs], dtype=np.float32)
    assert len(t) == n_q, "thresholds file has fewer rows than queries"
    return torch.from_numpy(t).to(DEV)


def _key(s):
    return pack(s.float().contiguous(), torch.zeros(s.shape[0], dtype=torch.int64,
                                                    device=s.device))


def _synthetic(seed=0, n_q=2000, n_rows=8192, vocab=30000):
    g = torch.Generator().manual_seed(seed)
    return _zipf_pos_csr(n_q, vocab, 7, g), _zipf_pos_csr(n_rows, vocab, 120, g)


@pytest.mark.parametrize("sub", [32, 64, 128])
@pytest.mark.parametrize("k", [10, 1000])
@pytest.mark.parametrize("elide", [False, True])
def test_identity_on_gpu(sub, k, elide):
    q, c = _synthetic(1, n_q=1000)
    cmin = float(c.values().min())
    base = sp.scores(q, c, cmin=cmin)
    thr = torch.topk(base, k, dim=1).values[:, -1].contiguous()
    ordv = torch.arange(c.shape[0], dtype=torch.int64, device=DEV)
    for mode in ("overhead", "on"):
        sp.reset()
        sp.set_policy("auto")
        bs.configure(mode, sub, elide)
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=_key(thr), return_skip=True)
        assert mk is not None, bs.usage()
        if mode == "overhead":
            assert torch.equal(out.view(torch.int32), base.view(torch.int32))
            continue
        if elide:
            k1, _, l1 = tt.topk(out, ordv, 50, thr=_key(thr), lmask=mk)
        mk.materialize()
        kept = ~torch.isneginf(out)
        assert torch.equal(out[kept].view(torch.int32), base[kept].view(torch.int32))
        dropped = torch.isneginf(out) & ~torch.isneginf(base)
        t = thr[:, None].expand_as(base)
        assert (base[dropped] < t[dropped]).all()
        assert torch.equal(out[base >= t], base[base >= t])
        if elide:
            k0, _, l0 = tt.topk(out, ordv, 50, thr=_key(thr))
            assert torch.equal(l0, l1)
            lv = l0.bool()
            assert torch.equal(k0[lv].sort(1).values, k1[lv].sort(1).values)
        u = bs.usage()
        assert u["check_failures"] == 0 and u["checks"] >= 1
        # Positive control only where a dead sub-block is guaranteed: at k=10
        # the slice's own 10th score prunes the synthetic slice heavily; at
        # k=1000 (and wide sub-blocks) the bound may legitimately prove nothing.
        if k <= 10:
            assert u["dead"] > 0, u


def _poison(monkeypatch, value=3e38):
    real = bs.try_skip

    def poisoned(q_csr, Cb, ptr, prow, pval, bptr, hot, n_rows, out, thr, cmin=None):
        out.fill_(value)
        return real(q_csr, Cb, ptr, prow, pval, bptr, hot, n_rows, out, thr, cmin)
    monkeypatch.setattr(bs, "try_skip", poisoned)


def _write_dataset(root, seed=3, n_files=4, per_file=9000, vocab=3000, n_q=600):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, vocab + 1) ** 1.1
    w /= w.sum()

    def rows(n, nnz):
        out = []
        for _ in range(n):
            k = max(1, int(rng.poisson(nnz)))
            idx = np.sort(rng.choice(vocab, size=min(k, vocab), replace=False, p=w))
            out.append((idx.tolist(), (rng.random(len(idx)) + 1e-3).astype(np.float32).tolist()))
        return out

    def write(path, rs):
        pq.write_table(pa.table({
            "sparse_embedding": pa.array(
                [{"indices": i, "values": v} for i, v in rs],
                type=pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                                pa.field("values", pa.list_(pa.float32()))])),
            "id": pa.array([f"{(j * 7919) % 10**6:07d}_{path.stem}_{j}"
                            for j in range(len(rs))])}), str(path))

    cdir = root / "corpus"
    cdir.mkdir()
    for f in range(n_files):
        write(cdir / f"f{f}.parquet", rows(per_file + 333 * f, 60))
    write(root / "q.parquet", rows(n_q, 7))
    return cdir, root / "q.parquet"


@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("k", [10, 100])
def test_e2e_poisoned_differential_on_gpu(tmp_path, monkeypatch, tiebreak, k):
    """run_compute on the GPU: off vs on (elided), on (unelided), overhead,
    with the score buffer poisoned before every skip launch."""
    import json

    import pyarrow.parquet as pq

    from nova_bf.compute import run_compute
    from nova_bf.config import (BruteForceConfig, CorpusConfig, OutputConfig,
                                ParamsConfig, QueriesConfig, SearchSpec)

    monkeypatch.setenv("NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD", "0")
    _poison(monkeypatch)
    cdir, qpath = _write_dataset(tmp_path)

    def run(tag, skip, elide):
        out = tmp_path / tag
        cfg = BruteForceConfig(
            corpus=CorpusConfig(path=str(cdir), sparse_column="sparse_embedding",
                                id_column="id"),
            queries=QueriesConfig(path=str(qpath), sparse_column="sparse_embedding",
                                  id_column="id"),
            output=OutputConfig(path=str(out)),
            params=ParamsConfig(io_workers=1, sparse_kernel="posting",
                                sparse_block_skip=skip, sparse_block_skip_elide=elide,
                                tiebreak=tiebreak),
            searches=[SearchSpec(name="dot", k=k, metric="dot", vector_type="sparse")])
        paths = run_compute(cfg)
        t = {n: pq.read_table(p).to_pydict() for n, p in paths.items()}
        man = json.loads(next(out.rglob("*manifest*.json")).read_text())
        return t, man["params"]["kernels"]["sparse_posting_kernel"]["block_skip"]

    base, _ = run("off", "off", True)
    for tag, skip, elide in (("on", "on", True), ("un", "on", False),
                             ("ov", "overhead", True)):
        got, u = run(tag, skip, elide)
        assert got == base, tag
        assert u["check_failures"] == 0 and u["disabled"] is None, u
        if tag == "on":
            assert u["dead"] > 0 and u["masked_topk"] > 0


def test_ftz_margin_on_gpu():
    """Flush-to-zero: weights 2^-63 (normal), products near 2^-126 so cold
    terms flush to zero on their own while a large hot term sums first in
    the kernel but last in the bound. Every corpus row is the same document
    (tight bound) and thr = the kernel's own score: nothing may be dead."""
    import numpy as np

    rng = np.random.default_rng(0)
    m = 40
    vocab = m + 1
    n_q = 512
    Q = torch.full((n_q, vocab), 2.0 ** -63)
    # make the LAST token the only hot one: it appears in extra queries too
    Q = torch.cat([Q, torch.zeros(256, vocab)])
    Q[n_q:, -1] = 2.0 ** -63
    doc = torch.tensor(np.concatenate([rng.uniform(0.5, 0.999, m), rng.uniform(1, 2, 1)])
                       * 2.0 ** -63, dtype=torch.float32)
    C = doc.expand(2048, -1).contiguous()
    q, c = Q.to_sparse_csr().to(DEV), C.to_sparse_csr().to(DEV)
    old = sp.HOT_TOKENS
    sp.HOT_TOKENS = 1
    try:
        sp.reset()
        sp.set_policy("auto")
        cmin = float(c.values().min())
        base = sp.scores(q, c, cmin=cmin)
        hotidx, _ = sp.hot_plan(q)
        assert int(hotidx[-1]) == 0 and int((hotidx >= 0).sum()) == 1
        bs.configure("on", 64, True)
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=_key(base[:, 0].clone()),
                            return_skip=True)
        assert mk is not None, bs.usage()
        mk.materialize()
        assert torch.equal(out.view(torch.int32), base.view(torch.int32))
        assert bs.usage()["dead"] == 0
    finally:
        sp.HOT_TOKENS = old


@pytest.mark.parametrize("sub,elide,mode,n_rows", [
    (64, True, "on", 8192), (64, False, "on", 8192), (32, True, "on", 8191),
    (64, True, "overhead", 8192), (128, True, "on", 100), (64, True, "on", 1)])
def test_model_probe_on_gpu(sub, elide, mode, n_rows, capsys):
    """The adversarial fp-model probe passes on real hardware (single-rounding
    fma, FTZ as compiled, contracted ub*F+SIG) for each variant; its cost
    (after the variant is compiled) is printed."""
    import time

    q, c = _synthetic(3, n_q=500, n_rows=n_rows)
    cmin = float(c.values().min())
    base = sp.scores(q, c, cmin=cmin)
    thr = torch.topk(base, min(10, n_rows), dim=1).values[:, -1].contiguous()
    sp.reset()
    sp.set_policy("auto")
    bs.configure(mode, sub, elide)
    real, spent = bs.model_probe, []

    def timed(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        real(*a, **k)
        torch.cuda.synchronize()
        spent.append(time.perf_counter() - t0)
    bs.model_probe = timed
    try:
        out, mk = sp.scores(q, c, cmin=cmin, skip_thr=_key(thr), return_skip=True)
    finally:
        bs.model_probe = real
    u = bs.usage()
    assert mk is not None and u["disabled"] is None, u
    assert u["model_probes"] == 1 and u["model_probe_failures"] == 0, u
    assert 0.0 < u["model_probe_worst"] <= 1.0, u
    with capsys.disabled():
        print(f"\nmodel probe sub={sub} elide={elide} mode={mode} n_rows={n_rows}: "
              f"{spent[0] * 1e3:.1f} ms, worst {u['model_probe_worst']:.3f} of the model")


def _ev_time(fn, reps):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()                                         # warm-up: compile, caches
    torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def _pool_topk(q, c_pool, cmin, need):
    """Per-query `need` best scores over a pool of corpus rows (scored in
    8192-row slices with the unskipped kernel)."""
    best = None
    n = int(c_pool.shape[0])
    crow = c_pool.crow_indices()
    for r0 in range(0, n, 8192):
        r1 = min(n, r0 + 8192)
        lo, hi = int(crow[r0]), int(crow[r1])
        sl = torch.sparse_csr_tensor(crow[r0:r1 + 1] - lo, c_pool.col_indices()[lo:hi],
                                     c_pool.values()[lo:hi], size=(r1 - r0, c_pool.shape[1]))
        s = sp.scores(q, sl, cmin=cmin)
        cand = s if best is None else torch.cat([best, s], 1)
        best = torch.topk(cand, min(need, cand.shape[1]), dim=1).values
    return best


def _quantile_thresholds(q, c_pool, cmin, k, file_rows, files=(1, 10, 150)):
    """The oracle's extrapolation (benchmarks/experiments/sparse_blockmax_
    oracle/oracle.py): after F production files (F * file_rows rows) a rank's
    k-th score ~ the (k * N / (F * file_rows))-th best of N sampled rows.
    Returns (representable regimes, refused regime messages): a regime whose
    rank falls below 1 (the pool is too small to represent it) is REFUSED,
    never clamped to rank 1."""
    N = int(c_pool.shape[0])
    ranks = {F: k * N / (F * file_rows) for F in files}
    ok = {F: r for F, r in ranks.items() if r >= 1}
    refused = [f"{F}f needs rank {r:.3g} of {N} pooled rows (< 1): load more rows "
               f"(NOVA_BF_BLOCKSKIP_POOL_ROWS) or pass NOVA_BF_BLOCKSKIP_THRESHOLDS"
               for F, r in ranks.items() if r < 1]
    if not ok:
        return [], refused
    best = _pool_topk(q, c_pool, cmin, int(round(max(ok.values()))))
    out = [(f"{F}f (rank {int(round(r))} of {N})",
            best[:, int(round(r)) - 1].contiguous()) for F, r in sorted(ok.items())]
    return out, refused


def test_time_one_slice(capsys):
    reps = int(os.environ.get("NOVA_BF_BLOCKSKIP_REPS", "10"))
    sub = int(os.environ.get("NOVA_BF_BLOCKSKIP_SUB", "64"))
    k = int(os.environ.get("NOVA_BF_BLOCKSKIP_K", "1000"))
    # production rows per corpus file (FineWeb gte-sparse: ~1.09M)
    file_rows = int(os.environ.get("NOVA_BF_BLOCKSKIP_FILE_ROWS", "1090000"))
    real = _real_slice()
    q, c = real if real is not None else _synthetic(2)
    pool = None
    if real is not None:
        q, c, pool = _real_pool(q, c)
    n_q, n_rows, vocab = int(q.shape[0]), int(c.shape[0]), int(c.shape[1])
    cmin = float(c.values().min())
    assert cmin > 0, "the skip path needs strictly positive corpus weights"
    assert sp.decline_reason(q, c) is None
    base = sp.scores(q, c, cmin=cmin)            # also runs the first-launch check
    kk = min(k, n_rows)
    env_thr = _thresholds_from_env(n_q)
    # Zero-skip: a FINITE threshold that kills nothing (every T >= SIG > 0).
    regimes = [("zero-skip (thr=0)", torch.zeros(n_q, device=DEV))]
    refused = []
    if env_thr is not None:
        regimes.append(("given (GT file)", env_thr))
    else:
        got, refused = _quantile_thresholds(q, pool if pool is not None else c, cmin, kk,
                                            file_rows)
        regimes += got

    ptr, prow, pval = sp.corpus_postings(c)
    bptr = sp.block_ptr(ptr, prow, vocab, n_rows, sp.DOC_BLOCK)
    hotidx, n_hot = sp.hot_plan(q)
    hot = (hotidx, sp.hot_table(ptr, prow, pval, hotidx, n_hot, n_rows, vocab))
    out = torch.empty((n_q, n_rows), dtype=torch.float32, device=DEV)
    ordv = torch.arange(n_rows, dtype=torch.int64, device=DEV)
    enc = ordv + 7
    rank = tt.rank_of(ordv)
    t_off = _ev_time(lambda: sp._launch(q, ptr, prow, pval, n_rows, out, bptr, hot), reps)

    lines = [f"\nblock skip, slice {n_rows} x {vocab}, {n_q} queries, sub={sub}, k={kk}, "
             f"{'real data' if real is not None else 'synthetic'}",
             f"  off posting kernel: {t_off:.3f} ms"]
    regs = {}
    zero = None
    for name, thr in regimes:
        key = _key(thr)
        res = {}
        bs.reset()
        bs.configure("on", sub, True)
        n_sub = -(-n_rows // sub)
        res["bm"] = _ev_time(lambda: bs.build_bm(q, ptr, prow, pval, hot, n_rows, sub), reps)
        bm = bs.build_bm(q, ptr, prow, pval, hot, n_rows, sub)
        res["bound"] = _ev_time(lambda: bs.run_bound(q, bm, n_sub, key), reps)
        res["prep"] = _ev_time(lambda: bs.prepare(q, ptr, prow, pval, hot, n_rows, key, out),
                               reps)
        ctx = bs.prepare(q, ptr, prow, pval, hot, n_rows, key, out)
        res["on"] = _ev_time(lambda: bs.launch(q, prow, pval, n_rows, out, hot, bptr, ctx),
                             reps)
        res["on_unelided"] = _ev_time(
            lambda: bs.launch(q, prow, pval, n_rows, out, hot, bptr, ctx, elide=False), reps)
        h = bs.launch(q, prow, pval, n_rows, out, hot, bptr, ctx)
        regs.setdefault("skip posting kernel", (getattr(h, "n_regs", None),
                                                getattr(h, "n_spills", None)))
        bs.configure("overhead", sub, True)
        cto = bs.prepare(q, ptr, prow, pval, hot, n_rows, key, out)
        res["overhead"] = _ev_time(
            lambda: bs.launch(q, prow, pval, n_rows, out, hot, bptr, cto), reps)
        res["topk_off"] = _ev_time(lambda: tt.topk(base, ordv, kk, thr=key, enc=enc,
                                                   rank=rank), reps)
        res["topk_on"] = _ev_time(lambda: tt.topk(out, ordv, kk, thr=key, enc=enc,
                                                  rank=rank, lmask=ctx), reps)
        nsb = sp.DOC_BLOCK // sub
        live = ctx.live
        pad = torch.zeros((n_q, -(-n_sub // nsb) * nsb), dtype=torch.int8, device=DEV)
        pad[:, :n_sub] = live
        prog_dead = float((pad.view(n_q, -1, nsb).amax(2) == 0).float().mean())
        row_dead = float((ctx.rowany == 0).float().mean())
        sb_dead = float((live == 0).float().mean())
        off_total = t_off + res["topk_off"]
        on_total = res["prep"] + res["on"] + res["topk_on"]
        lines.append(
            f"  {name}: BM {res['bm']:.3f} + bound {res['bound']:.3f} (prep "
            f"{res['prep']:.3f}) ms | posting on {res['on']:.3f}, unelided "
            f"{res['on_unelided']:.3f}, overhead {res['overhead']:.3f} ms | top-k off "
            f"{res['topk_off']:.3f}, masked {res['topk_on']:.3f} ms | dead: sub-blocks "
            f"{sb_dead:.3f}, programs {prog_dead:.3f}, rows {row_dead:.3f} | "
            f"posting+top-k+prep off {off_total:.3f} vs on {on_total:.3f} ms "
            f"({on_total / off_total:.3f}x)")
        if name.startswith("zero-skip"):
            zero = {"overhead": res["overhead"] / t_off, "on": res["on"] / t_off,
                    "topk": res["topk_on"] / res["topk_off"], "prep_ms": res["prep"]}
    # the HAS_LM _cutfill variant's registers / spills
    import triton

    outk = torch.empty((n_q, kk), dtype=torch.int64, device=DEV)
    outi = torch.empty((n_q, kk), dtype=torch.int64, device=DEV)
    lv = torch.empty(n_q, dtype=torch.uint8, device=DEV)
    blk = triton.next_power_of_2(n_rows)
    hc = tt._cutfill[(n_q,)](
        out, out, out, rank, ordv, enc, outk, outi, key, lv, out.stride(0), n_rows, kk,
        BLOCK=blk, RBITS=max(1, n_rows.bit_length()), HAS_NRM=False, HAS_CNRM=False,
        HAS_THR=True, HAS_ENC=True, HAS_SEL=True, HAS_LM=True, LSUB=ctx.sub,
        LMASK=ctx.klive, ROWANY=ctx.rowany, n_sub_l=ctx.klive.shape[1],
        num_warps=tt._warps_for(blk))
    regs["_cutfill HAS_LM"] = (getattr(hc, "n_regs", None), getattr(hc, "n_spills", None))
    lines.append("  registers / spills: " + ", ".join(f"{k}: {v[0]} / {v[1]}"
                                                     for k, v in regs.items()))
    lines += [f"  REFUSED regime: {r}" for r in refused]
    lines.append(f"  zero-skip ratios vs off: posting overhead {zero['overhead']:.3f}x, "
                 f"posting on {zero['on']:.3f}x, masked top-k {zero['topk']:.3f}x; "
                 f"prep {zero['prep_ms']:.3f} ms (adaptive gate skips prep when "
                 f"nothing dies)")
    with capsys.disabled():
        print("\n".join(lines))
    # zero-skip regression gates, per component
    assert zero["overhead"] <= 1.03 and zero["on"] <= 1.03, zero
    assert zero["topk"] <= 1.03, zero
