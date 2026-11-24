"""GPU validation of the cuVS sparse binding (run on a GPU node, in the cuvs env, after
build.sh).

Compares cuVS's sparse brute force, called through libnova_cuvs_sparse.so exactly as the
harness calls it (cuvs Resources bound to cudaStreamPerThread, torch tensors on that
stream), against the harness's float64 scipy oracle (bench/data/formats/sparse.py), with the
harness's --verify check (bench/verify/oracle.py compare_results: scores within
atol + 1e-5 |s|, ids exact outside tie zones).

    python validate_gpu.py                       # synthetic cases
    python validate_gpu.py --corpus-file F.parquet --query-file Q.parquet \\
        --column sparse_embedding --rows 200000 --nq 500 --k 1000

The harness runs cuVS at its stock sparse_search_params (2 << 14 each), which needs the
patched cuVS (baselines/cuvs/patches/0001, the sparse_knn.cuh fix): every case, including
the stock-params inner-product ones over several index batches, must pass. ``--unpatched``
turns those into EXPECTED-FAIL diagnostics instead (the stock cuVS 26.10 wheel's multi-batch
IP merge keeps the smallest scores), and adds the workaround parameters (one index batch
per block) as cases that must pass there. Exit status is non-zero when a non-diagnostic
case fails. Cases that might crash the process (an all-empty block) run in a subprocess.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the harness code is the bench package of this tree (clean_benchmarks/bench)
HARNESS = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # clean_benchmarks
sys.path[:0] = [HERE, HARNESS, os.path.join(HARNESS, "tests")]   # tests/: make_dummy

from bench.record import cudalibs  # noqa: E402

cudalibs.torch_first()  # load torch's pip CUDA libraries before cuvs maps conda's

import numpy as np  # noqa: E402
import scipy.sparse as ssp  # noqa: E402

import cuvs_sparse  # noqa: E402
from bench.verify import oracle as verify  # noqa: E402
from bench.systems import cuvs_engine  # noqa: E402
from bench.data.formats import sparse  # noqa: E402
import make_dummy  # noqa: E402

PER_THREAD = cuvs_sparse.CUDA_STREAM_PER_THREAD


def setup():
    import torch
    from cuvs.common import Resources
    res = Resources()
    cuvs_engine.cuvs_bind_stream(res, PER_THREAD)
    stream = torch.cuda.ExternalStream(PER_THREAD)
    return torch, res, stream


def gpu_knn(torch, res, stream, X, Q, k, metric, bi=None, bq=None, n_cols=None,
            raw_queries=False):
    """cuVS sparse top-k of CSR Q (scipy) against CSR X (scipy), on the
    per-thread stream; returns (D similarity float32, I int64, seconds)."""
    dbi, dbq = cuvs_sparse.default_batches()
    bi, bq = bi or dbi, bq or dbq
    n_cols = n_cols or max(X.shape[1], Q.shape[1])
    with torch.cuda.stream(stream):
        dev = [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in (
            X.indptr.astype(np.int32), X.indices.astype(np.int32), X.data.astype(np.float32),
            Q.indptr.astype(np.int32), Q.indices.astype(np.int32), Q.data.astype(np.float32))]
        nq, kk = Q.shape[0], min(k, X.shape[0])
        I = torch.empty((nq, kk), dtype=torch.int32, device="cuda")
        D = torch.empty((nq, kk), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        cuvs_sparse.knn(res.get_c_obj(), (dev[0], dev[1], dev[2], X.shape[0], X.nnz),
                        (dev[3], dev[4], dev[5], nq, Q.nnz), n_cols, kk,
                        1 if metric == "cosine" else 0, bi, bq, I, D)
        stream.synchronize()
        dt = time.perf_counter() - t
        Dn, In = D.cpu().numpy(), I.cpu().numpy().astype(np.int64)
    if metric == "cosine":
        Dn = 1 - Dn
    return sparse.pad(Dn, In, k) + (dt,)


def oracle(X, Q, k, metric):
    D, I = sparse.exact_topk(Q.astype(np.float64), X.astype(np.float64), k + 1, metric,
                             np.float64)
    D, I = sparse.pad(D, I, k + 1)
    return D, I, sparse.Rows(Q, metric == "cosine"), sparse.Rows(X, metric == "cosine")


def check(name, torch, res, stream, X, Q, k, metric, expect_fail=False, atol=1e-5, **kw):
    D, I, dt = gpu_knn(torch, res, stream, X, Q, k, metric, **kw)
    Dr, Ir, xq, xb = oracle(X, Q, k, metric)
    probs = verify.compare_results(D, I, Dr, Ir, atol=atol, xq=xq, xb=xb)
    ok = not probs
    tag = ("EXPECTED-FAIL" if not ok else "UNEXPECTED-PASS") if expect_fail else \
        ("PASS" if ok else "FAIL")
    maxd = float(np.max(np.abs(D - Dr[:, :k])[I >= 0])) if (I >= 0).any() else 0.0
    print(f"{tag:15s} {name}: {X.shape[0]} rows x {Q.shape[0]} queries, k={k}, {metric}, "
          f"{dt * 1e3:.1f} ms, max|dD| {maxd:.2e}{'' if ok else ' :: ' + probs[0]}",
          flush=True)
    return ok or expect_fail  # a diagnostic that passes (cuVS fixed it) is no failure


def synth(seed, n, vocab, nnz, empty=0.0):
    p = make_dummy.random_sparse(np.random.default_rng(seed), n, vocab, nnz, empty)
    return ssp.csr_matrix((p[2], p[1].astype(np.int64), p[0]), shape=(n, vocab))


def empty_block_case():
    """An index block without any nonzero (subprocess: may crash)."""
    torch, res, stream = setup()
    X = ssp.csr_matrix((50, 100), dtype=np.float32)
    Q = synth(1, 5, 100, 4)
    D, I, _ = gpu_knn(torch, res, stream, X, Q, 10, "ip")
    assert np.all(D[:, :10] == 0), D
    print("all-empty block: OK (all scores 0)")


def install_counted_pool(spec):
    """The harness's pool (cuvs_engine.cuvs_set_rmm_pool) with RMM's statistics
    adaptor on top as the current device resource; returns the adaptor (kept
    alive by the caller for the whole process: cuVS may hold its address)."""
    import rmm.mr
    size = cuvs_engine.cuvs_set_rmm_pool(spec)
    stats = rmm.mr.StatisticsResourceAdaptor(rmm.mr.get_current_device_resource())
    rmm.mr.set_current_device_resource(stats)
    print(f"RMM pool {size >> 20} MiB installed under an allocation counter", flush=True)
    return stats


def pool_report(stats):
    """Allocations cuVS's sparse calls made through the counted pool: True
    when there were any (rmm >= 24.06 returns an object, older a dict)."""
    c = stats.allocation_counts
    get = (lambda k: c[k]) if isinstance(c, dict) else (lambda k: getattr(c, k))
    n, peak = int(get("total_count")), int(get("peak_bytes"))
    ok = n > 0
    print(f"[{'PASS' if ok else 'FAIL'}] RMM pool reached by cuVS sparse: {n} allocations, "
          f"peak {peak / 2**20:.1f} MiB through the pool", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-file", default=None)
    ap.add_argument("--query-file", default=None)
    ap.add_argument("--column", default="sparse_embedding")
    ap.add_argument("--query-column", default=None)
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--unpatched", action="store_true",
                    help="stock cuVS 26.10 (no sparse_knn.cuh fix): the multi-batch IP cases "
                         "are EXPECTED-FAIL; also check the old one-index-batch workaround")
    ap.add_argument("--workspace-mib", type=int, default=4096,
                    help="--unpatched only: the workaround's tile cap")
    ap.add_argument("--race-probe", type=int, default=0,
                    help="also run N calls on a torch NON-blocking stream (diagnostic: "
                         "cuVS's stream-less cudaMemset may then race; counts mismatches)")
    ap.add_argument("--rmm-pool", default=None, metavar="SIZE",
                    help="install the harness's RMM pool (cuvs_engine.cuvs_set_rmm_pool, e.g. 1GiB) "
                         "under an allocation counter before anything else, run the "
                         "synthetic cases, and FAIL unless cuVS's sparse call allocated "
                         "through it (the check that --cuvs-rmm-pool reaches sparse cuVS)")
    ap.add_argument("--_empty_block", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._empty_block:
        empty_block_case()
        return
    pool_stats = install_counted_pool(args.rmm_pool) if args.rmm_pool else None
    torch, res, stream = setup()
    print(f"cuvs {__import__('cuvs').__version__}, torch {torch.__version__}, "
          f"{torch.cuda.get_device_name()}; default sparse batches "
          f"{cuvs_sparse.default_batches()}", flush=True)
    good = True

    if args.corpus_file:
        import pyarrow.parquet as pq
        t = pq.read_table(args.corpus_file, columns=[args.column]).slice(0, args.rows)
        x = sparse.unpack(sparse.table_csr(t, args.column))
        q, notes = sparse.load_queries(args.query_file, args.query_column or args.column,
                                       args.nq)
        for n in notes:
            print(n)
        n_cols = max(x.n_cols, q.n_cols)
        X = sparse.block_matrix(x, n_cols)
        Q = q.matrix(n_cols)
        rows = X.shape[0]
        print(f"real data: {rows} rows, {x.nnz} nnz, vocab {n_cols}")
        for metric in ("ip", "cosine"):   # the harness's parameters: cuVS's stock ones
            good &= check("real rows, STOCK params (the harness's)", torch, res, stream,
                          X, Q, min(args.k, 1024), metric,
                          expect_fail=args.unpatched and rows > 32768 and (
                              metric == "ip" or 0 < rows % 32768 < min(args.k, 1024)))
        if args.unpatched:
            bq = max(1, min(32768, Q.shape[0], (args.workspace_mib << 20) // (8 * rows)))
            for metric in ("ip", "cosine"):
                good &= check(f"real rows, old workaround (bi=rows bq={bq})", torch, res,
                              stream, X, Q, args.k, metric, bi=max(32768, rows), bq=bq)
        if pool_stats is not None:
            good &= pool_report(pool_stats)
        sys.exit(0 if good else 1)

    X, Q = synth(1, 5000, 3000, 40, 0.01), synth(2, 64, 3000, 8)
    for metric in ("ip", "cosine"):
        good &= check("small, stock params", torch, res, stream, X, Q, 10, metric)
        good &= check("small, k=1000", torch, res, stream, X, Q, 1000, metric)
        good &= check("small, 7-query batches", torch, res, stream, X, Q, 10, metric, bq=7)
    Xv, Qv = synth(3, 20000, 250002, 200), synth(4, 100, 250002, 30)
    for metric in ("ip", "cosine"):
        good &= check("gte-like vocab 250002 (hash strategy)", torch, res, stream, Xv, Qv, 100,
                  metric)
    Xb, Qb = synth(5, 70000, 5000, 30), synth(6, 50, 5000, 8)
    good &= check("70000 rows, bi=rows (old workaround)", torch, res, stream, Xb, Qb, 100,
              "ip", bi=70000)
    good &= check("70000 rows, cosine, stock bi", torch, res, stream, Xb, Qb, 100, "cosine")
    good &= check("70000 rows, IP, STOCK bi=32768 (3 index batches: the merge fix)",
              torch, res, stream, Xb, Qb, 100, "ip", expect_fail=args.unpatched)
    # 32768 x 2 + 500 rows at k=1000: the last index batch is short of k (wrong for
    # every metric without the patch's stride fix)
    Xs, Qs = synth(8, 2 * 32768 + 500, 5000, 20), synth(9, 30, 5000, 8)
    for metric in ("ip", "cosine"):
        good &= check("65536+500 rows, k=1000, STOCK (last index batch < k: the stride fix)",
                      torch, res, stream, Xs, Qs, 1000, metric, expect_fail=args.unpatched)
    # duplicate column in a query row: cuVS keeps one value, the oracle sums
    Qd = ssp.csr_matrix((np.array([1., 1., 1.], np.float32), np.array([3, 3, 7]),
                         np.array([0, 3])), shape=(1, 3000))
    Dg, Ig, _ = gpu_knn(torch, res, stream, X, Qd, 10, "ip")
    Qs = Qd.copy()
    Qs.sum_duplicates()
    Ds, Is, _ = gpu_knn(torch, res, stream, X, Qs, 10, "ip")
    same = np.allclose(Dg, Ds)
    print(f"{'NOTE':15s} duplicate query index: cuVS {'SUMS' if same else 'does NOT sum'} "
          f"duplicates (harness canonicalises queries either way)")
    p = subprocess.run([sys.executable, __file__, "--_empty_block"], capture_output=True, text=True)
    rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    sys.stdout.write(out)
    # a known, deliberately unpatched cuVS 26.10 bug: an index block without nonzeros raises a
    # CUDA error in select_warpsort (stock wheel and patched build alike); the harness refuses
    # the one way it could make such a block (--sparse-project). Only that failure is
    # KNOWN-FAIL; any other outcome (wrong answer, assert, segfault, other error) FAILs, and
    # a clean pass is reported as UNEXPECTED-PASS
    known = rc != 0 and "select_warpsort" in out
    tag = "UNEXPECTED-PASS" if rc == 0 else ("KNOWN-FAIL" if known else "FAIL")
    print(f"{tag:15s} all-empty index block (subprocess rc {rc}"
          f"{'; pre-existing cuVS select_warpsort bug, documented' if known else ''})")
    good &= rc == 0 or known
    if args.race_probe:
        bad = 0
        s2 = torch.cuda.Stream()
        cuvs_engine.cuvs_bind_stream(res, s2.cuda_stream)
        for _ in range(args.race_probe):
            D, I, _ = gpu_knn(torch, res, s2, Xv, Qv, 100, "ip")
            Dr, Ir, xq, xb = oracle(Xv, Qv, 100, "ip")
            bad += bool(verify.compare_results(D, I, Dr, Ir, xq=xq, xb=xb))
        cuvs_engine.cuvs_bind_stream(res, PER_THREAD)
        print(f"{'NOTE':15s} non-blocking stream: {bad} of {args.race_probe} calls wrong")
    if pool_stats is not None:
        good &= pool_report(pool_stats)
    print("ALL OK" if good else "SOME CASES FAILED")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
