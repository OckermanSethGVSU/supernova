"""--backend spare (SPARe, both modes, query-batched: baselines/spare/
spare_batched) through the harness on torch's CPU device: every --h2d /
--d2h transfer mode end to end (--verify, the ground truth, identical to the
pageable run, equal to a dense brute force), blocks shorter than k, an
fp16-stored corpus widened on the device, small scratch budgets (many query
groups / batches per block), the query padding, the CLI refusals, the CSV
columns and the --h2d-overlap device guard's SPARe term. The real GPU run is
owed (tests/test_sparse_gpu.py covers cuVS only)."""

import types

import numpy as np
import pytest
import torch

pytest.importorskip("spare")

import make_dummy  # noqa: E402
from bench.systems import harness_cli as run  # noqa: E402
from bench.systems import guards, loop
from bench.systems import cpu_reference, spare_engine
from bench.data.formats import sparse  # noqa: E402
from bench.data.sources import InMemorySource, ParquetSource  # noqa: E402
from test_sparse import (K, ROWS, TRANSFER_MODES, VOCAB, brute, csr_parts,  # noqa: E402,F401
                         dense_of, read_rows, sparse_corpus)

MODES = spare_engine.SPARE_MODES


def spare_backend(mode, **kw):
    def make(args, dtype):
        return spare_engine.SparseSpareBackend(args.k, args.spare_mode, device="cpu",
                                      events=not args.no_gpu_events, metric=args.metric,
                                      max_bytes=kw.get("max_bytes", args.spare_max_mib << 20),
                                      stage_events=not args.no_spare_events)
    return make


def queries_dense(qp):
    return sparse.load_queries(qp, "sparse_embedding")[0].matrix(VOCAB).toarray()


@pytest.mark.slow
@pytest.mark.parametrize("events", [True, False])
@pytest.mark.parametrize("tmode", sorted(TRANSFER_MODES))
@pytest.mark.parametrize("mode", MODES)
def test_cli_spare_transfer_modes(sparse_corpus, monkeypatch, tmp_path, mode, tmode, events):
    """SPARe through ResidentBackend's submit/complete in every --h2d / --d2h
    mode: --verify passes, the ground truth is recalled exactly, the CSV
    records engine / mode / budget / transfer modes, the results are identical
    to the pageable run's (ids globalised on the device) and equal a dense
    float64 brute force."""
    cdir, qp, _, x = sparse_corpus
    monkeypatch.setattr(run, "make_backend", spare_backend(mode))
    res = {}
    for name in ("pageable", tmode):
        out = tmp_path / f"{name}.csv"
        run.main(["--modality", "sparse", "--backend", "spare", "--spare-mode", mode,
                  "--regime", "stream", "--corpus", cdir, "--column", "sparse_embedding",
                  "--query-file", qp, "--query-column", "sparse_embedding", "--k", str(K),
                  "--csv", str(out), "--warmup", "1", "--repeats", "2", "--verify",
                  "--gt-ids-column", "gt_ids", "--prefetch", "2",
                  "--save-results", str(tmp_path / f"res_{name}_{{engine}}"),
                  *([] if events else ["--no-gpu-events"]), *TRANSFER_MODES[name]])
        rows = read_rows(out)
        assert {r["verified"] for r in rows if r["verified"]} == {"ok"}
        assert {float(r["gt_recall"]) for r in rows if r["gt_recall"]} == {1.0}
        r = rows[-1]
        assert (r["engine"], r["backend"], r["spare_mode"], r["sparse_batches"],
                r["queries_mode"], r["widen_where"]) == \
            ("spare", "spare", mode, "per-query" if spare_engine.spare_released(mode) else
             "max_bytes=8192MiB," + ("topk=contiguous" if mode == "dot" else "index=auto"),
             "resident", "device")
        assert r["engine_version"].endswith("+" + spare_engine.spare_batched_module().SPARE_COMMIT)
        a = TRANSFER_MODES[name]
        assert r["h2d_mode"] == ("pinned-overlap" if "--h2d-overlap" in a else
                                 "pinned" if "--h2d" in a else "pageable")
        assert r["d2h_mode"] == ("pinned-overlap" if "--d2h-overlap" in a else
                                 "pinned" if "--d2h" in a else "pageable")
        res[name] = np.load(tmp_path / f"res_{name}_spare.npz")
    np.testing.assert_array_equal(res[tmode]["I"], res["pageable"]["I"])
    np.testing.assert_array_equal(res[tmode]["D"], res["pageable"]["D"])
    Dr, _ = brute(queries_dense(qp), x, K)
    np.testing.assert_allclose(res[tmode]["D"], Dr, rtol=1e-5, atol=1e-6)


KNOBS = {"dot": [{}], "iterative": [{}],
         "dot-released": [{}], "iterative-released": [{}],
         "iterative-released-nosync": [{}]}


@pytest.mark.parametrize("events", [True, False])
@pytest.mark.parametrize("max_bytes", [1, 30_000, 4 << 30])
@pytest.mark.parametrize("mode, knob", [(m, kn) for m in KNOBS for kn in KNOBS[m]])
def test_backend_budgets_and_short_blocks(sparse_corpus, mode, knob, max_bytes, events):
    """A whole pass (run_pass: blocks of 900, 3, 650, 1, 1200 rows, two of
    them shorter than k) at any scratch budget (1 B: one query per call;
    30 kB: several groups; 4 GiB: one) gives the exact global top-k: the
    scores of a dense float64 brute force, every id's true score, k distinct
    ids per row, and the same result as the scipy engine."""
    cdir, qp, _, x = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    Q = q.matrix(VOCAB).toarray().astype(np.float64)
    src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    if spare_engine.spare_released(mode) and max_bytes != 30_000:
        pytest.skip("released modes have no budget")
    b = spare_engine.SparseSpareBackend(K, mode, device="cpu", max_bytes=max_bytes, events=events,
                               **knob)
    b.setup(q)
    D, I, _ = loop.run_pass(src, b, q.n_rows, K)
    Dr, _ = brute(Q, x, K)
    np.testing.assert_allclose(D, Dr, rtol=1e-5, atol=1e-6)
    S = Q @ x.T
    np.testing.assert_allclose(np.take_along_axis(S, I, 1), D, rtol=1e-5, atol=1e-6)
    assert all(len(set(r)) == K for r in I.tolist())
    cb = cpu_reference.SparseCpuBackend(K)
    cb.setup(q)
    Dc, Ic, _ = loop.run_pass(src, cb, q.n_rows, K)
    np.testing.assert_allclose(D, Dc, rtol=1e-6, atol=1e-6)
    assert b.stats.as_dict().get("build", 0) > 0


@pytest.mark.parametrize("mode", MODES)
def test_fp16_stored_corpus_widened_on_device(tmp_path, mode, monkeypatch):
    """An fp16-stored corpus (--widen device): values widened to float32 on
    the device before SPARe (iterative refuses fp16 collections; dot in fp16
    would not be exact), verified exact."""
    (tmp_path / "c").mkdir()
    parts = []
    for i, n in enumerate((300, 2, 250)):
        p = csr_parts(np.random.default_rng([9, i]), n, VOCAB, 12, 0.02)
        make_dummy.write_sparse(str(tmp_path / "c" / f"p{i}.parquet"), p,
                                value_dtype=np.float16)
        parts.append(p)
    qp = str(tmp_path / "q.parquet")
    make_dummy.make_sparse_queries(qp, 20, VOCAB, 6, seed=4, corpus_parts=parts, gt_k=K)
    monkeypatch.setattr(run, "make_backend", spare_backend(mode))
    out = tmp_path / "r.csv"
    run.main(["--modality", "sparse", "--backend", "spare", "--spare-mode", mode,
              "--regime", "stream", "--corpus", str(tmp_path / "c"), "--column",
              "sparse_embedding", "--query-file", qp, "--query-column", "sparse_embedding",
              "--k", str(K), "--csv", str(out), "--warmup", "0", "--repeats", "1",
              "--verify"])
    r = read_rows(out)[0]
    assert (r["verified"], r["stored_dtype"], r["widen_where"]) == ("ok", "f16", "device")
    assert r["widen_s"] != ""   # measured (iterative would refuse an fp16 collection)


def test_setup_queries_like_spare(sparse_corpus):
    """setup()'s CsrQueries equal spare_batched's own conversion of SPARe's
    {term: weight} dicts (queries_to_padded -> CsrQueries.from_padded); the
    released modes' per-query tensors are what SPARe's QuestionDataset +
    DataLoader yield (1, terms), int64 / float32."""
    _, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    sb = spare_engine.spare_batched_module()
    dicts = [dict(zip(q.indices[q.indptr[r]:q.indptr[r + 1]].tolist(),
                      q.values[q.indptr[r]:q.indptr[r + 1]].tolist()))
             for r in range(q.n_rows)]
    ref = sb.CsrQueries.from_padded(*sb.queries_to_padded(dicts))
    b = spare_engine.SparseSpareBackend(K, "dot", device="cpu")
    b.setup(q)
    assert (b.qc.indptr == ref.indptr).all()
    for f in ("terms", "weights", "rows"):
        assert torch.equal(getattr(b.qc, f), getattr(ref, f)), f
    from spare.backend_torch import QuestionDataset
    import torch.utils.data as tud
    r = spare_engine.SparseSpareBackend(K, "iterative-released", device="cpu")
    r.setup(q)
    dl = tud.DataLoader(QuestionDataset(dicts, lambda x: x, VOCAB), batch_size=1)
    n = 0
    for (ind, val), (a, v) in zip(dl, r.q_list):
        assert ind.dtype == a.dtype and val.dtype == v.dtype and torch.equal(ind, a)
        assert torch.equal(val, v)
        n += 1
    assert n == q.n_rows


def _argv(cdir, qp, *extra):
    return ["--modality", "sparse", "--regime", "stream", "--corpus", cdir, "--column",
            "sparse_embedding", "--query-file", qp, "--csv", "x.csv", *extra]


@pytest.mark.parametrize("extra, msg", [
    (["--backend", "spare"], "needs --spare-mode"),
    (["--backend", "spare", "--spare-mode", "dot", "--metric", "cosine"], "inner product only"),
    (["--backend", "spare", "--spare-mode", "dot", "--spare-max-mib", "0"], ">= 1"),
    (["--backend", "spare", "--spare-mode", "dot", "--temp-mem", "5"], "FAISS option"),
    (["--backend", "cuvs", "--spare-mode", "dot"], "--backend spare options"),
    (["--backend", "cpu", "--spare-max-mib", "8"], "--backend spare options"),
    (["--backend", "cuvs", "--spare-max-mib", "4096"], "--backend spare options"),
    (["--backend", "spare", "--spare-mode", "dot-released", "--spare-max-mib", "8"],
     "does not apply"),
])
def test_cli_spare_refusals(sparse_corpus, capsys, extra, msg):
    cdir, qp, _, _ = sparse_corpus
    with pytest.raises(SystemExit):
        run.parse_args(_argv(cdir, qp, *extra))
    assert msg in capsys.readouterr().err


def test_cli_spare_refuses_dense(sparse_corpus, capsys):
    cdir, qp, _, _ = sparse_corpus
    argv = [a for a in _argv(cdir, qp, "--backend", "spare", "--spare-mode", "dot")
            if a not in ("--modality", "sparse")]
    with pytest.raises(SystemExit):
        run.parse_args(argv)
    assert "--modality sparse only" in capsys.readouterr().err


def test_cli_spare_accepts_and_defaults(sparse_corpus):
    cdir, qp, _, _ = sparse_corpus
    a = run.parse_args(_argv(cdir, qp, "--backend", "spare", "--spare-mode", "iterative",
                             "--k", "2000", "--d2h", "pinned"))
    assert (a.spare_mode, a.spare_max_mib, a.widen, a.k) == ("iterative", 8192, "device", 2000)


def test_spare_engine_bytes_terms():
    feet = [types.SimpleNamespace(sparse_nnz=1000), types.SimpleNamespace(sparse_nnz=5000)]
    args = types.SimpleNamespace(spare_max_mib=2, spare_mode="dot")
    assert spare_engine.spare_engine_bytes(args, feet, 10, pairs=70, q_cols=100) == (2 << 20) + 70 * 64
    args.spare_mode = "iterative"
    assert spare_engine.spare_engine_bytes(args, feet, 10, pairs=70, q_cols=100) == \
        (2 << 20) + 70 * 64 + 5000 * spare_engine.SPARE_CSC_BYTES_PER_NNZ + 101 * 20
    feet = [types.SimpleNamespace(sparse_nnz=1000, num_rows=40),
            types.SimpleNamespace(sparse_nnz=5000, num_rows=300)]
    args.spare_mode = "iterative-released"   # no budget charged
    assert spare_engine.spare_engine_bytes(args, feet, 10, pairs=70, q_cols=100) == \
        (3 * 300 + 100) * 4 + 70 * 64 + 5000 * spare_engine.SPARE_CSC_BYTES_PER_NNZ + 101 * 20
    args.spare_mode = "iterative-released-nosync"
    assert spare_engine.spare_engine_bytes(args, feet, 10, pairs=70, q_cols=100) == \
        (3 * 300 + 100) * 4 + 70 * 64 + 5000 * spare_engine.SPARE_CSC_BYTES_PER_NNZ + 101 * 20
    args.spare_mode = "dot-released"
    assert spare_engine.spare_engine_bytes(args, feet, 10, pairs=70, q_cols=100) == \
        (3 * 300 + 100) * 4 + 70 * 64


@pytest.mark.parametrize("mode", MODES)
def test_spare_overlap_gpu_guard(sparse_corpus, monkeypatch, mode):
    """The --h2d-overlap guard charges SPARe's term (CUDA faked): exactly fits
    passes, one byte less is refused with SPARe named."""
    cdir, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    b = spare_engine.SparseSpareBackend(K, mode, device="cpu")
    b.setup(q)
    src = ParquetSource(cdir, "sparse_embedding", prefetch=0, codec=sparse.CODEC,
                        read_mode="arrow")
    try:
        args = run.parse_args(_argv(cdir, qp, "--backend", "spare", "--spare-mode", mode,
                                    "--k", str(K), "--h2d", "pinned", "--h2d-overlap",
                                    "--prefetch", "2"))
        feet = src.read_footers()
        f = [feet[r.path] for r in src.files]
        block = max(sparse.CODEC.block_bytes(g, None) for g in f)
        nq = q.n_rows
        need = (2 * block + nq * K * guards.OVERLAP_RESULT_BYTES_PER_CELL
                + spare_engine.spare_engine_bytes(args, f, nq, q.nnz, q.n_cols)
                + guards.OVERLAP_GPU_MARGIN)
        guards._sparse_overlap_guard(args, src, feet, nq, need, need, b)
        with pytest.raises(SystemExit, match="SPARe's model"):
            guards._sparse_overlap_guard(args, src, feet, nq, need - 1, need, b)
    finally:
        src.close()


def test_nosync_loop_equals_released_loop(sparse_corpus):
    """iterative-released-nosync returns the untouched released loop's
    result bit for bit on every block (same arithmetic, same order)."""
    cdir, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    out = {}
    for mode in ("iterative-released", "iterative-released-nosync"):
        b = spare_engine.SparseSpareBackend(K, mode, device="cpu")
        b.setup(q)
        out[mode] = loop.run_pass(src, b, q.n_rows, K)[:2]
    np.testing.assert_array_equal(out["iterative-released"][0],
                                  out["iterative-released-nosync"][0])
    np.testing.assert_array_equal(out["iterative-released"][1],
                                  out["iterative-released-nosync"][1])


STAGES = ("spare_plan_s", "spare_score_s", "spare_topk_s")


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_stage_events_columns(sparse_corpus, monkeypatch, tmp_path, mode):
    """The batched and nosync modes record plan / score / topk spans inside
    search_s (events between the harness's own calls; a host clock on the
    CPU); SPARe's own forward modes record none. --no-spare-events and
    --no-gpu-events leave the columns blank, the results unchanged."""
    cdir, qp, _, _ = sparse_corpus
    monkeypatch.setattr(run, "make_backend", spare_backend(mode))
    staged = mode in spare_engine.SPARE_STAGED_MODES
    variants = {"on": [], "gpuoff": ["--no-gpu-events"]}
    if staged:
        variants["off"] = ["--no-spare-events"]
    res, rows = {}, {}
    for name, extra in variants.items():
        out = tmp_path / f"{name}.csv"
        run.main(["--modality", "sparse", "--backend", "spare", "--spare-mode", mode,
                  "--regime", "stream", "--corpus", cdir, "--column", "sparse_embedding",
                  "--query-file", qp, "--query-column", "sparse_embedding", "--k", str(K),
                  "--csv", str(out), "--warmup", "0", "--repeats", "1",
                  "--save-results", str(tmp_path / f"res_{name}_{{engine}}"), *extra])
        rows[name] = read_rows(out)[-1]
        res[name] = np.load(tmp_path / f"res_{name}_spare.npz")
    r = rows["on"]
    if not staged:
        assert r["spare_events"] == rows["gpuoff"]["spare_events"] == "n/a"
        assert [r[c] for c in STAGES] == ["", "", ""]
        return
    assert r["spare_events"] == (f"sampled:{spare_engine.SPARE_NOSYNC_EVENT_STRIDE}"
                                 if mode.endswith("nosync") else "on")
    plan, score, topk = (float(r[c]) for c in STAGES)
    assert score > 0 and topk > 0 and plan >= 0
    assert (plan > 0) == (mode != "dot")       # dot has no planning step
    if not mode.endswith("nosync"):            # measured, not a scaled sample
        assert plan + score + topk + float(r["build_s"]) <= float(r["search_s"]) * 1.001
    for name in ("off", "gpuoff"):
        assert rows[name]["spare_events"] == "off"
        assert [rows[name][c] for c in STAGES] == ["", "", ""]
        np.testing.assert_array_equal(res[name]["I"], res["on"]["I"])
        np.testing.assert_array_equal(res[name]["D"], res["on"]["D"])


def test_nosync_stage_events_are_sampled(sparse_corpus):
    """nosync times one query in SPARE_NOSYNC_EVENT_STRIDE (three marks each)
    and scales the sums to every query; the batched modes time every group."""
    cdir, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    b = spare_engine.SparseSpareBackend(K, "iterative-released-nosync", device="cpu")
    b.setup(q)
    marks = []
    real = b._mark
    b._mark = lambda: marks.append(1) or real()
    seen = []
    real_search = b._engine_search

    def spy(*a):
        out = real_search(*a)
        seen.append(b._stage_ev)
        return out
    b._engine_search = spy
    loop.run_pass(src, b, q.n_rows, K)
    timed = -(-q.n_rows // spare_engine.SPARE_NOSYNC_EVENT_STRIDE)
    for ev in seen:
        assert len(ev["spare_score"][0]) == len(ev["spare_topk"][0]) == timed
        assert ev["spare_score"][1] == ev["spare_topk"][1] == q.n_rows / timed
        assert ev["spare_plan"][1] == 1.0 and len(ev["spare_plan"][0]) == 1


def test_no_spare_events_refusals(sparse_corpus, capsys):
    cdir, qp, _, _ = sparse_corpus
    base = ["--modality", "sparse", "--regime", "stream", "--corpus", cdir, "--column",
            "sparse_embedding", "--query-file", qp, "--query-column", "sparse_embedding",
            "--csv", "x.csv", "--no-spare-events"]
    for extra in (["--backend", "spare", "--spare-mode", "dot-released"],
                  ["--backend", "cpu"]):
        with pytest.raises(SystemExit):
            run.parse_args(base + extra)
        assert "--no-spare-events" in capsys.readouterr().err
