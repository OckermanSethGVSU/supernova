"""--modality sparse, CPU only: the packed CSR block, its decode from the
struct<indices, values> column (every read mode, every reader kind, global
row offsets), the canonical queries, the exact scipy engine / float64 oracle
(against a dense numpy brute force, ties to the lowest id), the harness's CLI
wiring (--verify, --gt-ids-column, --save-results, the CSV columns, the
refusals), and SparseCuvsBackend's plumbing with the cuVS call faked on
torch's CPU device (the real call is GPU only: tests/data/formats/test_sparse_gpu.py and
baselines/cuvs/sparse_binding/validate_gpu.py)."""

import csv
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import make_dummy
from bench.systems import harness_cli as run
from bench.systems import guards, loop
from bench.record import results_io, schema
from bench.systems import cpu_reference, cuvs_engine
from bench.data.formats import sparse
from bench.data.sources import InMemorySource, ParquetSource, read_file, read_footer, list_parquet

K = 10


def csr_parts(rng, n, vocab, nnz=12, empty=0.0):
    return make_dummy.random_sparse(rng, n, vocab, nnz, empty)


def dense_of(parts, vocab, dtype=np.float64):
    indptr, idx, val = parts
    x = np.zeros((len(indptr) - 1, vocab), dtype)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    np.add.at(x, (rows, idx.astype(np.int64)), val.astype(dtype))
    return x


def brute(q, x, k, metric="ip"):
    """Dense numpy reference: score desc, lowest id first on ties."""
    s = q @ x.T
    if metric == "cosine":
        qn, xn = np.linalg.norm(q, axis=1), np.linalg.norm(x, axis=1)
        den = qn[:, None] * xn[None, :]
        s = np.where(den > 0, s / np.where(den > 0, den, 1), 0)
    order = np.argsort(-s, axis=1, kind="stable")
    I = order[:, :k]
    return np.take_along_axis(s, I, axis=1), I


ROWS = [900, 3, 650, 1, 1200]
VOCAB = 700


@pytest.fixture(scope="module")
def sparse_corpus(tmp_path_factory):
    """Five sparse files (row groups of 400; some rows empty), the queries
    (with exact gt_ids) and the dense float64 corpus."""
    out = tmp_path_factory.mktemp("sp")
    (out / "c").mkdir()
    parts = []
    for i, n in enumerate(ROWS):
        p = csr_parts(np.random.default_rng([5, i]), n, VOCAB, 15, 0.02)
        make_dummy.write_sparse(str(out / "c" / f"part-{i:05d}.parquet"), p,
                                row_group_size=400)
        parts.append(p)
    qp = str(out / "q.parquet")
    make_dummy.make_sparse_queries(qp, 40, VOCAB, 6, seed=3, corpus_parts=parts, gt_k=K)
    x = np.concatenate([dense_of(p, VOCAB) for p in parts])
    return str(out / "c"), qp, parts, x


# ------------------------------------------------------------ packed block

def test_pack_unpack_roundtrip_and_alignment():
    x = sparse.new_packed(5, 7, np.float16)
    c = sparse.unpack(x)
    assert sparse.is_packed(x) and sparse.packed_rows(x) == 5 and sparse.packed_nnz(x) == 7
    assert (c.n_rows, c.nnz, c.n_cols) == (5, 7, 0)
    assert c.values.dtype == np.float16 and c.indices.dtype == np.int32
    for a in (c.indptr, c.indices, c.values):
        assert (a.ctypes.data - x.ctypes.data) % sparse.ALIGN == 0
    assert x.nbytes == sparse.packed_nbytes(5, 7, 2)
    sparse.set_n_cols(x, 99)
    assert sparse.unpack(x).n_cols == 99
    assert not sparse.is_packed(np.zeros((4, 4), np.float32))
    assert not sparse.is_packed(np.zeros(1024, np.uint8))


def test_decode_into_buffer_and_room_check():
    t = pa.table({"s": make_dummy.sparse_column(np.array([0, 2, 3]), np.array([1, 5, 2]),
                                                np.array([.5, .25, 1.], np.float32))})
    buf = bytearray(4096)
    x = sparse.table_csr(t, "s", out=memoryview(buf))
    assert x.base is not None and np.shares_memory(x, np.frombuffer(buf, np.uint8))
    c = sparse.unpack(x)
    assert c.indptr.tolist() == [0, 2, 3] and c.indices.tolist() == [1, 5, 2]
    assert c.values.tolist() == [.5, .25, 1.] and c.n_cols == 6
    with pytest.raises(ValueError, match="larger than a shared-memory ring slot"):
        sparse.table_csr(t, "s", out=memoryview(bytearray(300)))


def _struct(indices, values, offs, ityp=pa.uint32(), vtyp=pa.float32(), large=False):
    mk = pa.LargeListArray if large else pa.ListArray
    o = pa.array(offs, pa.int64() if large else pa.int32())
    return pa.StructArray.from_arrays([mk.from_arrays(o, pa.array(indices, ityp)),
                                       mk.from_arrays(o, pa.array(values, vtyp))],
                                      names=["indices", "values"])


def test_decode_chunks_slices_and_types():
    a = _struct([3, 1, 4, 1, 5, 9], [1, 2, 3, 4, 5, 6], [0, 2, 2, 6], ityp=pa.int64(),
                vtyp=pa.float64(), large=True)
    b = _struct([2, 6], [7, 8], [0, 1, 2], ityp=pa.int64(), vtyp=pa.float64(), large=True)
    t = pa.Table.from_batches([pa.record_batch([a], names=["s"]),
                               pa.record_batch([b], names=["s"])])
    c = sparse.unpack(sparse.table_csr(t, "s"))
    assert c.indptr.tolist() == [0, 2, 2, 6, 7, 8]  # rebased per chunk, then global
    assert c.indices.tolist() == [3, 1, 4, 1, 5, 9, 2, 6]
    assert c.values.dtype == np.float32  # f64 narrowed, as nova-bf
    assert c.n_cols == 10
    # a sliced table (offset struct + list arrays) decodes its own rows only
    s = sparse.unpack(sparse.table_csr(t.slice(1, 3), "s"))
    assert s.indptr.tolist() == [0, 0, 4, 5] and s.indices.tolist() == [4, 1, 5, 9, 2]
    f16 = pa.table({"s": _struct([0], [0.5], [0, 1], vtyp=pa.float16())})
    assert sparse.unpack(sparse.table_csr(f16, "s")).values.dtype == np.float16


@pytest.mark.parametrize("bad, match", [
    (lambda: pa.table({"s": pa.StructArray.from_arrays(
        [pa.array([[1]]), pa.array([[1.0]])], names=["indices", "values"],
        mask=pa.array([True]))}), "null rows"),
    (lambda: pa.table({"s": pa.StructArray.from_arrays(
        [pa.array([[1, 2]]), pa.array([[1.0]])], names=["indices", "values"])}),
     "differ in length"),
    (lambda: pa.table({"s": _struct([-1], [1.0], [0, 1], ityp=pa.int32())}), "outside"),
    (lambda: pa.table({"s": _struct([2**31], [1.0], [0, 1], ityp=pa.uint32())}), "outside"),
    (lambda: pa.table({"s": pa.array([[1.0]])}), "expected a sparse struct"),
])
def test_decode_refusals(bad, match):
    with pytest.raises((ValueError, TypeError), match=match):
        sparse.table_csr(bad(), "s")


def test_footer_bounds_the_block(sparse_corpus):
    cdir, qp, parts, _ = sparse_corpus
    fs, refs = list_parquet(cdir)
    for r, p in zip(refs, parts):
        f = read_footer(fs, r.path, "sparse_embedding")
        assert f.sparse_nnz >= len(p[1]) and f.sparse_value_bits == 32 and f.dim is None
        x = read_file(fs, r.path, "sparse_embedding", codec=sparse.CODEC)
        assert sparse.CODEC.block_bytes(f, None) >= x.nbytes
        assert sparse.CODEC.block_bytes(f, np.float32) >= x.nbytes



def test_dense_footer_has_no_sparse_fields(corpus):
    fs, refs = list_parquet(corpus[0])
    f = read_footer(fs, refs[0].path, "embedding")
    assert f.sparse_nnz is None and f.sparse_value_bits is None and f.dim


@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_read_modes_and_host_widen(tmp_path, mode):
    p = csr_parts(np.random.default_rng(1), 300, 200)
    path = make_dummy.write_sparse(str(tmp_path / "a.parquet"), p, value_dtype=np.float16,
                                   row_group_size=70)
    fs, refs = list_parquet(path)
    from bench.timing import Stages
    st, rec = Stages(), {}
    x = read_file(fs, refs[0].path, "sparse_embedding", stats=st, size=refs[0].size,
                  mode=mode, range_bytes=1000, codec=sparse.CODEC, rec=rec)
    c = sparse.unpack(x)
    assert c.values.dtype == np.float16 and st.as_dict()["rows"] == 300
    np.testing.assert_array_equal(c.indptr, p[0])
    np.testing.assert_array_equal(c.indices, p[1])
    w = read_file(fs, refs[0].path, "sparse_embedding", dtype=np.float32, stats=st,
                  size=refs[0].size, mode=mode, range_bytes=1000, codec=sparse.CODEC, rec=rec)
    cw = sparse.unpack(w)
    assert cw.values.dtype == np.float32 and rec["widen"] > 0
    np.testing.assert_array_equal(cw.values, c.values.astype(np.float32))


@pytest.mark.parametrize("prefetch", [2, 1, 0])
def test_sources_offsets_and_blocks(sparse_corpus, prefetch):
    cdir, _, parts, _ = sparse_corpus
    src = ParquetSource(cdir, "sparse_embedding", prefetch=prefetch, workers=2,
                        codec=sparse.CODEC)
    try:
        offs, got = [], []
        for b in src:
            c = sparse.unpack(b.vectors)
            offs.append(b.offset)
            got.append((c.indptr.copy(), c.indices.copy(), c.values.copy()))
            del b, c
    finally:
        src.close()
    assert offs == list(np.cumsum([0] + ROWS[:-1]))
    for (ip, ix, iv), (pp, px, pv) in zip(got, parts):
        np.testing.assert_array_equal(ip, pp)
        np.testing.assert_array_equal(ix, px)
        np.testing.assert_array_equal(iv, pv)
    mem = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    assert [b.offset for b in mem.blocks] == offs


# ------------------------------------------------------------ queries

def test_canonical_sums_duplicates_and_sorts():
    ip, ix, iv = sparse.canonical(np.array([0, 3, 4]), np.array([5, 2, 5, 1]),
                                  np.array([1., 2., 3., 4.], np.float32))
    assert ip.tolist() == [0, 2, 3] and ix.tolist() == [2, 5, 1]
    assert iv.tolist() == [2., 4., 4.]


def test_load_queries_notes(tmp_path):
    p = make_dummy.write_sparse(str(tmp_path / "q.parquet"),
                                (np.array([0, 2, 2]), np.array([3, 3], np.uint32),
                                 np.array([1., 2.], np.float32)))
    q, notes = sparse.load_queries(p, "sparse_embedding")
    assert q.indices.tolist() == [3] and q.values.tolist() == [3.]
    assert q.n_rows == 2 and q.n_cols == 4
    assert any("duplicate" in n for n in notes) and any("no nonzeros" in n for n in notes)
    q1, _ = sparse.load_queries(p, "sparse_embedding", nq=1)
    assert q1.n_rows == 1


# ------------------------------------------------------------ exact engine

@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_topk_matches_dense_bruteforce(sparse_corpus, metric, dtype):
    cdir, qp, parts, x = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    import scipy.sparse as ssp
    X = ssp.csr_matrix(x)
    Q = q.matrix(VOCAB)
    D, I = sparse.exact_topk(Q, X, K, metric, dtype, tile_cells=40 * 333)  # many tiles
    Dr, Ir = brute(Q.toarray().astype(np.float64), x, K, metric)
    np.testing.assert_allclose(D, Dr, rtol=1e-5, atol=1e-6)
    if dtype is np.float64:
        np.testing.assert_array_equal(I, Ir)
    gt = np.stack(pq.read_table(qp).column("gt_ids").to_pylist())
    if metric == "ip" and dtype is np.float64:
        np.testing.assert_array_equal(I, gt)


def test_ties_go_to_the_lowest_id_across_tiles():
    S = np.array([[1, 3, 3, 0, 3, 3, 2]], np.float32)
    D, I = sparse.topk_tile(S, 3)
    assert I.tolist() == [[1, 2, 4]] and D.tolist() == [[3, 3, 3]]
    import scipy.sparse as ssp
    X = ssp.csr_matrix(np.ones((9, 2)))
    Q = ssp.csr_matrix(np.ones((1, 2)))
    for tile in (1, 2, 4, 100):
        _, I = sparse.exact_topk(Q, X, 4, "ip", np.float64, tile_cells=tile)
        assert I.tolist() == [[0, 1, 2, 3]]
    # zero rows (no overlap) still fill k, lowest ids first, as cuVS fills them
    X = ssp.csr_matrix(np.array([[0, 1], [1, 0], [0, 2], [0, 0]], np.float64))
    D, I = sparse.exact_topk(ssp.csr_matrix(np.array([[1., 0]])), X, 3, "cosine", np.float64)
    assert I.tolist() == [[1, 0, 2]] and D.tolist() == [[1, 0, 0]]


# ------------------------------------------------------------ CLI

def base_args(cdir, qp, csv_path, *extra):
    return ["--modality", "sparse", "--backend", "cpu", "--corpus", cdir,
            "--column", "sparse_embedding", "--query-file", qp, "--query-column",
            "sparse_embedding", "--k", str(K), "--csv", str(csv_path), "--warmup", "0",
            "--repeats", "2", "--workers", "2", *extra]


def read_rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("regime, extra", [("stream", ["--prefetch", "2"]),
                                           ("stream", ["--prefetch", "0"])])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_cli_sparse_verify_gt_and_csv(sparse_corpus, tmp_path, regime, extra, metric, capsys):
    cdir, qp, parts, x = sparse_corpus
    out = tmp_path / "r.csv"
    run.main(base_args(cdir, qp, out, "--regime", regime, "--metric", metric, "--verify",
                       "--gt-ids-column", "gt_ids", "--save-results",
                       str(tmp_path / "res_{engine}"), *extra))
    err = capsys.readouterr()
    assert "verify OK" in err.out and "float64 scipy sparse" in err.out
    rows = read_rows(out)
    assert len(rows) == 2 and list(rows[0]) == schema.CSV_FIELDS
    r = rows[0]
    assert (r["modality"], r["engine"], r["verified"], r["sparse_batches"]) == \
        ("sparse", "scipy", "ok", "n/a")
    assert int(r["nnz"]) == sum(len(p[1]) for p in parts)
    assert int(r["rows"]) == sum(ROWS) and int(r["rows_per_file"]) == max(ROWS)
    assert r["stored_dtype"] == "f32" and r["d"] != ""
    if metric == "ip":
        assert float(r["gt_recall"]) == 1.0
    res = np.load(tmp_path / "res_scipy.npz")
    Dr, Ir = brute(sparse.load_queries(qp, "sparse_embedding")[0].matrix(VOCAB).toarray(),
                   x, K, metric)
    np.testing.assert_allclose(res["D"], Dr, rtol=1e-5, atol=1e-6)
    trows = read_rows(results_io.trace_path(str(out)))
    assert {t["modality"] for t in trows} == {"sparse"}
    assert sorted({int(t["rows"]) for t in trows}) == sorted(set(ROWS))


def test_cli_f16_values_widen_on_host(tmp_path):
    p = csr_parts(np.random.default_rng(4), 500, 300)
    cdir = tmp_path / "c"
    cdir.mkdir()
    make_dummy.write_sparse(str(cdir / "a.parquet"), p, value_dtype=np.float16)
    qp = make_dummy.make_sparse_queries(str(tmp_path / "q.parquet"), 20, 300, 5,
                                        corpus_parts=[p], value_dtype=np.float16)
    out = tmp_path / "r.csv"
    run.main(base_args(str(cdir), qp, out, "--regime", "stream", "--verify",
                       "--gt-ids-column", "gt_ids"))
    r = read_rows(out)[0]
    assert r["stored_dtype"] == "f16" and r["widen_where"] == "host"
    assert float(r["widen_s"]) > 0 and float(r["gt_recall"]) == 1.0


@pytest.mark.parametrize("extra, match", [
    (["--backend", "gpu"], "FAISS has no sparse"),
    (["--dim", "8"], "dense only"),
])
def test_cli_refusals(sparse_corpus, tmp_path, extra, match, capsys):
    cdir, qp, _, _ = sparse_corpus
    with pytest.raises(SystemExit):
        run.parse_args(base_args(cdir, qp, tmp_path / "r.csv", "--regime", "stream",
                                 *extra))
    assert match in capsys.readouterr().err


def test_cli_needs_query_file_and_a_sparse_column(sparse_corpus, corpus, tmp_path, capsys):
    cdir, qp, _, _ = sparse_corpus
    with pytest.raises(SystemExit):
        run.parse_args(["--modality", "sparse", "--backend", "cpu", "--regime", "stream",
                        "--corpus", cdir, "--csv", str(tmp_path / "r.csv")])
    assert "needs --query-file" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="not a struct"):
        run.main(["--modality", "sparse", "--backend", "cpu", "--regime", "stream",
                  "--corpus", corpus[0], "--query-file", qp, "--query-column",
                  "sparse_embedding", "--csv", str(tmp_path / "r.csv")])
    with pytest.raises(SystemExit):
        run.parse_args(["--cuvs-sparse-workspace-mib", "5", "--regime", "stream",
                        "--corpus", cdir, "--csv", str(tmp_path / "r.csv")])


def test_dense_rows_have_modality_dense(corpus, tmp_path):
    out = tmp_path / "d.csv"
    run.main(["--backend", "cpu", "--regime", "stream", "--corpus", corpus[0], "--nq", "5",
              "--k", "3", "--csv", str(out), "--warmup", "0", "--repeats", "1"])
    r = read_rows(out)[0]
    assert (r["modality"], r["nnz"], r["sparse_batches"]) == ("dense", "", "n/a")


def test_sparse_verify_guard_refuses(sparse_corpus, tmp_path, monkeypatch):
    cdir, qp, _, _ = sparse_corpus
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: 1 << 20)
    with pytest.raises(SystemExit, match="sparse run needs about"):
        run.main(base_args(cdir, qp, tmp_path / "r.csv", "--regime", "stream", "--verify"))


# ------------------------------------------- SparseCuvsBackend, faked call

torch = pytest.importorskip("torch")


class FakeRes:
    def __init__(self):
        self.c_obj = 0x5000

    def get_c_obj(self):
        return self.c_obj

    def sync(self):
        """calc_truth's per-batch ``resources.sync()`` (cuVS's own recipe)."""


def fake_knn(calls):
    """What the shim does, on torch CPU tensors: cuVS's sparse brute force
    (inner product largest first; cosine 1 - cos smallest first), int32 ids,
    written in place; asserts the call contract."""
    import scipy.sparse as ssp

    def knn(res, index, queries, n_cols, k, metric, bi, bq, I_t, D_t):
        ip, ix, iv, n, nnz = index
        qp, qx, qv, nq, qnnz = queries
        assert res == 0x5000
        for t, dt in ((ip, torch.int32), (ix, torch.int32), (iv, torch.float32),
                      (qp, torch.int32), (qx, torch.int32), (qv, torch.float32)):
            assert t.dtype == dt and t.is_contiguous()
        assert ip.numel() == n + 1 and ix.numel() == nnz == iv.numel()
        assert I_t.dtype == torch.int32 and D_t.dtype == torch.float32
        assert tuple(I_t.shape) == (nq, k) and 1 <= k <= n
        assert bi >= 1 and bq >= 1
        assert int(ix.max()) < n_cols if nnz else True
        calls.append((n, k, metric, bi, bq))
        X = ssp.csr_matrix((iv.numpy(), ix.numpy(), ip.numpy()), shape=(n, n_cols))
        Q = ssp.csr_matrix((qv.numpy(), qx.numpy(), qp.numpy()), shape=(nq, n_cols))
        D, I = sparse.exact_topk(Q, X, k, "cosine" if metric == 1 else "ip", np.float32)
        if metric == 1:
            D = 1 - D
        I_t.copy_(torch.from_numpy(I.astype(np.int32)))
        D_t.copy_(torch.from_numpy(D.astype(np.float32)))
    return knn


@pytest.fixture
def fake_cuvs_pkg(monkeypatch):
    cuvs = types.ModuleType("cuvs")
    cuvs.__version__ = "26.10.00-fake"
    common = types.ModuleType("cuvs.common")
    common.Resources = FakeRes
    cuvs.common = common
    monkeypatch.setitem(sys.modules, "cuvs", cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.common", common)
    import fake_cupy
    return fake_cupy.install(monkeypatch)   # cuVS's own recipe merges with CuPy


def make_cuvs(calls, **kw):
    return cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=fake_knn(calls), **kw)


@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("events", [True, False])
def test_cuvs_backend_matches_cpu_engine(sparse_corpus, fake_cuvs_pkg, metric, events):
    cdir, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    calls = []
    gb = make_cuvs(calls, metric=metric, events=events)
    cb = cpu_reference.SparseCpuBackend(K, metric=metric)
    gb.setup(q)
    cb.setup(q)
    src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    Dg, Ig, _ = loop.run_pass(src, gb, q.n_rows, K)
    Dc, Ic, _ = loop.run_pass(src, cb, q.n_rows, K)
    np.testing.assert_allclose(Dg, Dc, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(Ig, Ic)
    probe, calls = calls[0], calls[1:]      # setup()'s self-test (ip, k 3, bi 2) first
    assert probe[1:4] == (3, 0, 2)
    assert [c[1] for c in calls] == [min(K, n) for n in ROWS]  # k_eff, padded on the host
    assert all(c[2] == (1 if metric == "cosine" else 0) for c in calls)
    assert all(c[3:] == cuvs_engine.SPARSE_STOCK_BATCHES for c in calls)  # stock params
    st = gb.stats.as_dict()
    if events:
        assert st["search"] > 0 and st["h2d"] > 0 and st["widen"] == 0
    else:
        assert "search" not in st


def test_cuvs_backend_widens_f16_values_on_device(tmp_path, fake_cuvs_pkg):
    p = csr_parts(np.random.default_rng(8), 400, 120)
    path = make_dummy.write_sparse(str(tmp_path / "a.parquet"), p, value_dtype=np.float16)
    q = sparse.Queries(*sparse.canonical(*csr_parts(np.random.default_rng(9), 7, 120, 4)),
                       n_cols=120)
    calls = []
    b = make_cuvs(calls)
    b.setup(q)
    src = InMemorySource.load(path, "sparse_embedding", codec=sparse.CODEC)
    assert sparse.unpack(src.blocks[0].vectors).values.dtype == np.float16
    D, I, _ = loop.run_pass(src, b, q.n_rows, K)
    assert b.stats.as_dict()["widen"] > 0
    cb = cpu_reference.SparseCpuBackend(K)
    cb.setup(q)
    D2, I2, _ = loop.run_pass(InMemorySource.load(path, "sparse_embedding", np.float32,
                                                 codec=sparse.CODEC), cb, q.n_rows, K)
    np.testing.assert_array_equal(I, I2)


def test_cuvs_batch_params_are_stock(fake_cuvs_pkg):
    """cuVS's own sparse_search_params, whatever the block or query count."""
    b = make_cuvs([])
    for rows, nq in ((100, 10), (1_000_000, 50_000), (1_090_356, 10_000)):
        assert b.params(rows, nq) == cuvs_engine.SPARSE_STOCK_BATCHES == (32768, 32768)


def buggy_knn(calls, bug="merge"):
    """Unfixed cuVS 26.10 sparse inner product, on CPU. bug="merge": per index
    batch the largest k, then knn_merge_parts keeping the SMALLEST k
    candidates. bug="stride" (merge fixed): a batch of n < k rows writes its
    select_k result densely as (query rows, n) into the stride-k merge part,
    which the merge then reads at stride k (the unwritten tail is 0, id 0)."""
    import scipy.sparse as ssp
    if bug == "stride":
        return _stride_buggy_knn(calls, ssp)

    def knn(res, index, queries, n_cols, k, metric, bi, bq, I_t, D_t):
        ip, ix, iv, n, nnz = index
        qp, qx, qv, nq, qnnz = queries
        calls.append((n, k, metric, bi, bq))
        X = ssp.csr_matrix((iv.numpy(), ix.numpy(), ip.numpy()), shape=(n, n_cols))
        Q = ssp.csr_matrix((qv.numpy(), qx.numpy(), qp.numpy()), shape=(nq, n_cols))
        S = (Q @ X.T).toarray()
        for r in range(nq):
            cand = []
            for b0 in range(0, n, bi):
                seg = S[r, b0:b0 + bi]
                o = np.argsort(-seg, kind="stable")[:min(k, len(seg))]
                cand += [(seg[i], b0 + i) for i in o]
            keep = sorted(cand)[:k]
            D_t[r] = torch.tensor([d for d, _ in keep], dtype=torch.float32)
            I_t[r] = torch.tensor([i for _, i in keep], dtype=torch.int32)
    return knn


def _stride_buggy_knn(calls, ssp):
    def knn(res, index, queries, n_cols, k, metric, bi, bq, I_t, D_t):
        ip, ix, iv, n, nnz = index
        qp, qx, qv, nq, qnnz = queries
        calls.append((n, k, metric, bi, bq))
        X = ssp.csr_matrix((iv.numpy(), ix.numpy(), ip.numpy()), shape=(n, n_cols))
        Q = ssp.csr_matrix((qv.numpy(), qx.numpy(), qp.numpy()), shape=(nq, n_cols))
        S = (Q @ X.T).toarray()
        for q0 in range(0, nq, bq):
            rows = min(bq, nq - q0)
            cand = [[] for _ in range(rows)]
            for b0 in range(0, n, bi):
                m = min(k, min(bi, n - b0))
                fd, fi = np.zeros(rows * k), np.zeros(rows * k, np.int64)
                for r in range(rows):              # dense (rows, m) write
                    seg = S[q0 + r, b0:b0 + bi]
                    o = np.argsort(-seg, kind="stable")[:m]
                    fd[r * m:(r + 1) * m], fi[r * m:(r + 1) * m] = seg[o], o
                for r in range(rows):              # stride-k read, ids translated
                    cand[r] += zip(fd[r * k:(r + 1) * k], fi[r * k:(r + 1) * k] + b0)
            for r in range(rows):
                keep = sorted(cand[r], key=lambda c: -c[0])[:k]
                D_t[q0 + r] = torch.tensor([d for d, _ in keep], dtype=torch.float32)
                I_t[q0 + r] = torch.tensor([i for _, i in keep], dtype=torch.int32)
    return knn


@pytest.mark.parametrize("bug", ["merge", "stride"])
def test_self_test_refuses_each_unfixed_bug(fake_cuvs_pkg, sparse_corpus, bug):
    """Each cuVS bug alone (min-only merge; short-batch stride) fails setup()."""
    _, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding", 5)
    bad = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=buggy_knn([], bug))
    with pytest.raises(RuntimeError, match="lacks the sparse_knn.cuh fix"):
        bad.setup(q)


@pytest.mark.parametrize("backend, k, refused", [
    ("cuvs", 1025, True), ("cuvs", 1024, False), ("cpu", 2048, False)])
def test_sparse_cuvs_refuses_k_over_merge_limit(sparse_corpus, tmp_path, capsys,
                                                backend, k, refused):
    cdir, qp, _, _ = sparse_corpus
    argv = ["--modality", "sparse", "--backend", backend, "--regime", "stream",
            "--corpus", cdir, "--column", "sparse_embedding", "--query-file", qp,
            "--k", str(k), "--csv", str(tmp_path / "r.csv")]
    if not refused:
        assert run.parse_args(argv).k == k
        return
    with pytest.raises(SystemExit):
        run.parse_args(argv)
    assert "knn_merge_parts" in capsys.readouterr().err


def test_self_test_refuses_unfixed_cuvs_and_passes_fixed(fake_cuvs_pkg, sparse_corpus):
    """setup() trips both bugs the patched cuVS fixes (multi-batch IP merge, a
    batch of fewer than k rows): an unfixed build is refused before any pass;
    a correct one passes (its probe call is cuVS's own knn with bi = 2)."""
    _, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding", 5)
    calls = []
    bad = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=buggy_knn(calls))
    with pytest.raises(RuntimeError, match="lacks the sparse_knn.cuh fix"):
        bad.setup(q)
    assert calls and calls[0][3] == 2          # the probe used batch_size_index 2
    calls = []
    good = make_cuvs(calls)
    good.setup(q)                               # fake_knn is exact: passes
    assert calls[0][:4] == (5, 3, 0, 2)


def test_cli_refuses_unfixed_cuvs(sparse_corpus, fake_cuvs_pkg, monkeypatch, tmp_path):
    cdir, qp, _, _ = sparse_corpus
    monkeypatch.setattr(run, "make_backend", lambda args, dtype: cuvs_engine.SparseCuvsBackend(
        args.k, device="cpu", events=not args.no_gpu_events, metric=args.metric,
        knn=buggy_knn([])))
    out = tmp_path / "r.csv"
    with pytest.raises(RuntimeError, match="self-test failed"):
        run.main(["--modality", "sparse", "--backend", "cuvs", "--regime", "stream",
                  "--corpus", cdir, "--column", "sparse_embedding", "--query-file", qp,
                  "--query-column", "sparse_embedding", "--k", str(K), "--csv", str(out),
                  "--warmup", "0", "--repeats", "1"])
    assert not out.exists() or not read_rows(out)


def test_cuvs_binds_the_per_thread_stream(fake_cuvs_pkg, monkeypatch):
    """On CUDA the backend runs on cudaStreamPerThread (0x2): wrapped for
    torch as an ExternalStream and bound to the Resources via the C API
    (read back)."""
    streams = {}

    class CApi:
        def cuvsStreamSet(self, res, stream):
            streams[res.value] = stream.value
            return cuvs_engine.CUVS_SUCCESS

        def cuvsStreamGet(self, res, out):
            out._obj.value = streams[res.value]
            return cuvs_engine.CUVS_SUCCESS

    made = []
    monkeypatch.setattr(cuvs_engine, "cuvs_c_api", lambda: CApi())
    monkeypatch.setattr(torch.cuda, "ExternalStream",
                        lambda h, device=None: made.append((h, device)) or object())
    b = cuvs_engine.SparseCuvsBackend(K, device="cuda", knn=fake_knn([]))
    assert made == [(0x2, "cuda")] and streams == {0x5000: 0x2}
    assert b.stream is not None


def test_cli_cuvs_sparse_with_fake_call(sparse_corpus, fake_cuvs_pkg, monkeypatch, tmp_path):
    cdir, qp, _, _ = sparse_corpus
    calls = []
    monkeypatch.setattr(run, "make_backend", lambda args, dtype: cuvs_engine.SparseCuvsBackend(
        args.k, device="cpu", events=not args.no_gpu_events, metric=args.metric,
        knn=fake_knn(calls)))
    out = tmp_path / "r.csv"
    run.main(["--modality", "sparse", "--backend", "cuvs", "--regime", "stream",
              "--corpus", cdir, "--column", "sparse_embedding", "--query-file", qp,
              "--query-column", "sparse_embedding", "--k", str(K), "--csv", str(out),
              "--warmup", "0", "--repeats", "1", "--verify", "--gt-ids-column", "gt_ids",
              "--metric", "cosine"])
    r = read_rows(out)[0]
    assert (r["engine"], r["verified"], r["sparse_batches"], r["widen_where"]) == \
        ("cuvs", "ok", "32768x32768", "device")
    assert r["rmm_pool"] == "pool:1GiB" and calls   # the default, as dense cuVS
    assert all(c[3:] == (32768, 32768) for c in calls[1:])   # the passes: stock params

TRANSFER_MODES = {
    "pageable": [],
    "h2d_pinned": ["--h2d", "pinned"],
    "h2d_overlap": ["--h2d", "pinned", "--h2d-overlap"],
    "d2h_pinned": ["--d2h", "pinned"],
    "d2h_overlap": ["--d2h", "pinned", "--d2h-overlap"],
    "all_overlap": ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap"],
}


@pytest.mark.parametrize("events", [True, False])
@pytest.mark.parametrize("mode", sorted(TRANSFER_MODES))
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_cli_cuvs_sparse_transfer_modes(sparse_corpus, fake_cuvs_pkg, monkeypatch, tmp_path,
                                        mode, metric, events):
    """Sparse cuVS through ResidentBackend's submit/complete in every --h2d /
    --d2h mode (the call faked on torch's CPU device; pinning is a counted
    no-op there, as for the dense tests): --verify passes, the ground truth
    is recalled exactly, the CSV records the modes, and the results are
    identical to the pageable run's (ids globalised on the device)."""
    cdir, qp, _, x = sparse_corpus
    calls = []
    monkeypatch.setattr(run, "make_backend", lambda args, dtype: cuvs_engine.SparseCuvsBackend(
        args.k, device="cpu", events=not args.no_gpu_events, metric=args.metric,
        knn=fake_knn(calls)))
    res = {}
    for name in ("pageable", mode):
        out = tmp_path / f"{name}.csv"
        run.main(["--modality", "sparse", "--backend", "cuvs", "--regime", "stream",
                  "--corpus", cdir, "--column", "sparse_embedding", "--query-file", qp,
                  "--query-column", "sparse_embedding", "--k", str(K), "--csv", str(out),
                  "--warmup", "1", "--repeats", "2", "--verify", "--gt-ids-column", "gt_ids",
                  "--metric", metric, "--prefetch", "2",
                  "--save-results", str(tmp_path / f"res_{name}_{{engine}}"),
                  *([] if events else ["--no-gpu-events"]), *TRANSFER_MODES[name]])
        rows = read_rows(out)
        assert {r["verified"] for r in rows if r["verified"]} == {"ok"}
        if metric == "ip":
            assert {float(r["gt_recall"]) for r in rows if r["gt_recall"]} == {1.0}
        r = rows[-1]
        a = TRANSFER_MODES[name]
        assert r["h2d_mode"] == ("pinned-overlap" if "--h2d-overlap" in a else
                                 "pinned" if "--h2d" in a else "pageable")
        # cuVS's own recipe: one copy back at the end whatever --d2h says
        assert r["d2h_mode"] == "final-once" and r["cuvs_merge"] == "device:file"
        res[name] = np.load(tmp_path / f"res_{name}_cuvs.npz")
    np.testing.assert_array_equal(res[mode]["I"], res["pageable"]["I"])
    np.testing.assert_array_equal(res[mode]["D"], res["pageable"]["D"])
    Dr, Ir = brute(sparse.load_queries(qp, "sparse_embedding")[0].matrix(VOCAB).toarray(),
                   x, K, metric)
    np.testing.assert_allclose(res[mode]["D"], Dr, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("normalize", [False, True])
def test_rows_pair_dot_equals_dense_dot(normalize):
    """sparse.Rows.pair_dot (the --verify rescoring, no densify) equals the
    dense float64 np.dot of the two rows it replaces, incl. empty / zero-norm
    rows and duplicate columns, to float64 rounding."""
    import scipy.sparse as ssp
    rng = np.random.default_rng(31)
    X = ssp.random(60, 400, density=0.05, random_state=1, format="csr") * 3
    X = ssp.vstack([X, ssp.csr_matrix((2, 400))], format="csr")       # two empty rows
    dup = ssp.csr_matrix((np.array([1.0, 2.0]), np.array([5, 5]), np.array([0, 2])),
                         shape=(1, 400))                              # duplicate column
    X = ssp.vstack([X, dup], format="csr")
    Q = ssp.vstack([ssp.random(9, 400, density=0.08, random_state=2, format="csr"),
                    ssp.csr_matrix((1, 400)), dup], format="csr")
    xq, xb = sparse.Rows(Q, normalize), sparse.Rows(X, normalize)
    for q in range(Q.shape[0]):
        for i in rng.choice(X.shape[0], 25).tolist() + [X.shape[0] - 1, X.shape[0] - 2]:
            dense = float(np.dot(xq[q], xb[i]))
            assert abs(xb.pair_dot(xq, q, i) - dense) <= 1e-12 * max(1.0, abs(dense)), (q, i)


def test_rows_pair_dot_mixed_normalization():
    """Each side divides by its own norm iff normalized (as other[q] / self[i]
    densified would), whichever side it is."""
    import scipy.sparse as ssp
    Q = ssp.random(5, 80, density=0.2, random_state=3, format="csr")
    X = ssp.random(12, 80, density=0.2, random_state=4, format="csr")
    for nq_, nx_ in ((True, False), (False, True)):
        xq, xb = sparse.Rows(Q, nq_), sparse.Rows(X, nx_)
        for q in range(5):
            for i in range(12):
                d = float(np.dot(xq[q], xb[i]))
                assert abs(xb.pair_dot(xq, q, i) - d) <= 1e-12 * max(1.0, abs(d)), (nq_, nx_)


def test_sparse_overlap_gpu_guard(sparse_corpus, fake_cuvs_pkg, monkeypatch):
    """--h2d-overlap's device-memory guard for sparse (it runs on CUDA only, so
    CUDA is faked here): sized from the packed blocks, it neither crashes on a
    sparse footer (no dim / value_bits) nor lets an oversize run through."""
    cdir, qp, _, _ = sparse_corpus
    b = make_cuvs([])
    src = ParquetSource(cdir, "sparse_embedding", prefetch=0, codec=sparse.CODEC,
                        read_mode="arrow")
    try:
        args = run.parse_args(["--modality", "sparse", "--backend", "cuvs", "--regime", "stream",
                               "--corpus", cdir, "--column", "sparse_embedding", "--query-file",
                               qp, "--k", str(K), "--csv", "x.csv", "--h2d", "pinned",
                               "--h2d-overlap"])
        monkeypatch.setattr(b, "_cuda", True)
        for free, ok in ((40 << 30, True), (1 << 20, False)):
            monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None, f=free: (f, 48 << 30))
            if ok:
                guards.overlap_gpu_memory_guard(args, b, src, 40, np.float32)
            else:
                with pytest.raises(SystemExit, match="packed sparse blocks"):
                    guards.overlap_gpu_memory_guard(args, b, src, 40, np.float32)
    finally:
        src.close()


def test_warmup_files_with_gt_recall_does_not_crash(sparse_corpus, tmp_path, capsys):
    """A shortened warmup pass (--warmup-files) has no recall: its summary line
    must not try to format one (crashed after the warmup row, any modality)."""
    cdir, qp, _, _ = sparse_corpus
    out = tmp_path / "r.csv"
    run.main(base_args(cdir, qp, out, "--regime", "stream", "--warmup", "1",
                       "--warmup-files", "2", "--gt-ids-column", "gt_ids"))
    rows = read_rows(out)
    assert [r["gt_recall"] for r in rows if r["warmup"] in ("1", "True")] == [""]
    assert float([r for r in rows if r["warmup"] not in ("1", "True")][-1]["gt_recall"]) == 1.0


def test_sparse_overlap_guard_mixed_corpus_and_f64(tmp_path, fake_cuvs_pkg, monkeypatch):
    """_sparse_overlap_guard: a later non-sparse file does not crash the guard
    (its read fails with the codec's error instead), and fp64-stored values
    (narrowed on the host) are not charged a device widen."""
    from types import SimpleNamespace as NS
    cdir = tmp_path / "c"
    cdir.mkdir()
    p = csr_parts(np.random.default_rng(7), 300, 200)
    make_dummy.write_sparse(str(cdir / "a.parquet"), p, value_dtype=np.float64)
    pq.write_table(pa.table({"sparse_embedding": pa.array([[1.0, 2.0]] * 50)}),
                   str(cdir / "b.parquet"))
    fs, refs = list_parquet(str(cdir))
    feet = {r.path: read_footer(fs, r.path, "sparse_embedding") for r in refs}
    src = NS(files=refs, read_footers=lambda: feet)
    args = NS(widen="device", k=K, modality="sparse")
    guards._sparse_overlap_guard(args, src, feet, 20, 40 << 30, 48 << 30)   # no TypeError
    f = feet[refs[0].path]
    assert f.sparse_value_bits == 64
    tile = min(32768, 20) * min(32768, max(x.num_rows for x in feet.values())) * 8 + min(32768, 20) * K * 32
    need_free = (2 * sparse.CODEC.block_bytes(f, None) + 20 * K * guards.OVERLAP_RESULT_BYTES_PER_CELL
                 + tile + guards.OVERLAP_GPU_MARGIN)
    guards._sparse_overlap_guard(args, src, feet, 20, need_free, 48 << 30)  # exactly fits: no widen
    with pytest.raises(SystemExit):
        guards._sparse_overlap_guard(args, src, feet, 20, need_free - 1, 48 << 30)


def test_sparse_cuvs_returns_int64_global_ids(sparse_corpus, fake_cuvs_pkg):
    """Ids are widened to int64 and globalised on the device (no int32 past
    2^31 global ids)."""
    cdir, qp, _, _ = sparse_corpus
    b = make_cuvs([])
    q, _ = sparse.load_queries(qp, "sparse_embedding", 40)
    b.setup(q)
    src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    blk = src.blocks[2]
    D, I = b.search(blk.vectors, offset=3_000_000_000)
    assert I.dtype == np.int64
    valid = I >= 0
    assert valid.any() and I[valid].min() >= 3_000_000_000


def test_sparse_verify_need_terms(sparse_corpus, tmp_path):
    """sparse_verify_need against hand arithmetic: decode factor, the native
    image term, the overlap ring slots, --d2h pinned buffers, cosine's CSR
    copies and the merge term."""
    from types import SimpleNamespace as NS
    cdir, qp, _, _ = sparse_corpus
    fs, refs = list_parquet(cdir)
    feet = {r.path: read_footer(fs, r.path, "sparse_embedding") for r in refs}
    q, _ = sparse.load_queries(qp, "sparse_embedding", 40)
    args = NS(prefetch=2, workers=2, warmup=1, repeats=2,
              k=K, backend="cuvs", metric="cosine", h2d_overlap=True, d2h_overlap=True,
              d2h="pinned")
    t = guards.sparse_verify_need(args, refs, feet, q, None)
    f = [feet[r.path] for r in refs]
    packed = [sparse.CODEC.block_bytes(x, None) for x in f]
    b = max(range(len(f)), key=packed.__getitem__)
    fb = f[b]
    raw = min(refs[b].size, fb.column_bytes + (1 << 20) + (4096 + 8192 + 4096) * 2 * fb.num_row_groups)
    stored = max({16: 2, 64: 8}.get(x.sparse_value_bits, 4) for x in f)
    transient = 5 * (fb.sparse_nnz * (8 + stored) + 8 * fb.num_rows) + packed[b] + raw
    assert t["raw"] == raw and t["transient"] == transient
    ring = (min(2, len(refs)) + 1 + 2) * max(packed)
    assert t["in_passes"] == ring + 2 * transient
    shared = (3 * q.nbytes + 2 * q.nnz * 12 + 3 * q.n_rows * K * 12 + guards.VERIFY_PROCESS_ALLOWANCE
              + guards.d2h_pinned_bytes(args, q.n_rows) + guards.CUDA_HOST_ALLOWANCE)
    assert t["shared"] == shared
    rows, nnz = sum(x.num_rows for x in f), sum(x.sparse_nnz for x in f)
    csr = nnz * 12 + rows * 8
    step = max(1, sparse.CPU_TILE_CELLS // q.n_rows)
    oracle = (sum(packed) + 5 * csr + sparse.CPU_TILE_CELLS * 40
              + q.n_rows * (K + 1 + step) * 40 + q.n_rows * (K + 1) * 12)
    assert t["oracle"] == oracle
    assert t["need"] == int((shared + max(ring + 2 * transient, oracle)) * guards.VERIFY_SAFETY)
    ip = guards.sparse_verify_need(NS(**{**vars(args), "metric": "ip", "d2h": "pageable",
                                      "h2d_overlap": False, "d2h_overlap": False}),
                                refs, feet, q, None)
    assert ip["raw"] == raw and ip["oracle"] == oracle - 3 * csr
    assert ip["in_passes"] == (min(2, len(refs)) + 1) * max(packed) + 2 * transient


def test_repeats0_warmup_files_recall_and_verify(sparse_corpus, tmp_path, capsys):
    """--repeats 0 with a --warmup-files warmup: the only pass reads a prefix.
    The final recall line must not format the empty recall, and --verify must
    refuse clearly (it has no full pass) instead of an IndexError."""
    cdir, qp, _, _ = sparse_corpus
    base = ["--modality", "sparse", "--backend", "cpu", "--corpus", cdir, "--column",
            "sparse_embedding", "--query-file", qp, "--query-column", "sparse_embedding",
            "--k", str(K), "--regime", "stream", "--warmup", "1", "--repeats", "0",
            "--warmup-files", "2", "--workers", "2", "--gt-ids-column", "gt_ids"]
    run.main(base + ["--csv", str(tmp_path / "a.csv")])            # no ValueError
    with pytest.raises(SystemExit, match="no full pass"):
        run.main(base + ["--csv", str(tmp_path / "b.csv"), "--verify"])



def test_cpu_engine_refuses_huge_vocabulary():
    import scipy.sparse as ssp
    X = ssp.csr_matrix((np.array([1.0, 2.0]), np.array([0, (1 << 26) + 5]), np.array([0, 2])),
                       shape=(1, (1 << 26) + 6))
    Q = ssp.csr_matrix((np.array([1.0]), np.array([0]), np.array([0, 1])), shape=(1, (1 << 26) + 6))
    with pytest.raises(ValueError, match="vocabulary"):
        sparse.exact_topk(Q, X, 1)


# ------------------------------------------- --cuvs-sparse-batches (the tile)

@pytest.mark.parametrize("text,want", [
    ("8192x32768", (8192, 32768)), ("1x1", (1, 1)), ("8192xstock", (8192, "stock")),
    ("stockx2500", ("stock", 2500)), ("stockxstock", ("stock", "stock")),
    ("filexstock", ("file", "stock")), ("filex64", ("file", 64)),
    ("2147483647x1", (2**31 - 1, 1)),
])
def test_cuvs_sparse_batches_parses(text, want):
    assert cuvs_engine.parse_cuvs_sparse_batches(text) == want


@pytest.mark.parametrize("text", [
    "8192X32768", "8192", "8192x", "x8192", "8192x32768x1", "0x5", "5x0", "-1x5", "+1x5",
    "08192x5", "1.5x5", "8192 x5", " 8192x5", "stockxfile", "filexfile", "Stockx5", "fx5",
    "2147483648x1", "",
])
def test_cuvs_sparse_batches_rejects_malformed(text):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cuvs_engine.parse_cuvs_sparse_batches(text)


def _tile_argv(cdir, qp, csv_path, *extra):
    return ["--modality", "sparse", "--backend", "cuvs", "--regime", "stream",
            "--corpus", cdir, "--column", "sparse_embedding", "--query-file", qp,
            "--query-column", "sparse_embedding", "--k", str(K), "--csv", str(csv_path),
            "--warmup", "0", "--repeats", "1", *extra]


def test_cuvs_sparse_batches_cli_parsing_and_refusals(capsys):
    base = ["--regime", "stream", "--corpus", "c", "--csv", "x.csv", "--query-file", "q.parquet"]
    args = run.parse_args(base + ["--modality", "sparse", "--backend", "cuvs",
                                  "--cuvs-sparse-batches", "filex2500"])
    assert args.cuvs_sparse_batches == ("file", 2500)
    assert run.parse_args(base + ["--modality", "sparse", "--backend", "cuvs"]
                          ).cuvs_sparse_batches is None
    for bad in (["--modality", "sparse", "--backend", "cpu"],
                ["--modality", "sparse", "--backend", "spare", "--spare-mode", "dot"],
                ["--backend", "cuvs"],                      # dense cuVS
                ["--modality", "sparse", "--backend", "cuvs", "--cuvs-sparse-batches", "0x1"],
                ["--modality", "sparse", "--backend", "cuvs", "--cuvs-sparse-batches", "8192"],
                ["--modality", "sparse", "--backend", "cuvs", "--cuvs-sparse-batches",
                 "stockxfile"]):
        flag = [] if "--cuvs-sparse-batches" in bad else ["--cuvs-sparse-batches", "8192x32768"]
        with pytest.raises(SystemExit) as e:
            run.parse_args(base + bad + flag)
        assert e.value.code == 2, bad
        assert "--cuvs-sparse-batches" in capsys.readouterr().err, bad


def tiling_knn(calls):
    """fake_knn, but TILED like cuVS: the queries in batches of ``bq`` rows,
    the index in batches of ``bi`` rows (the last one short, possibly < k),
    each index batch's exact top-min(k, rows) merged across batches by
    (score, lowest id) as the patched sparse_knn.cuh does. Exact, so it must
    reproduce the untiled answer for any tile."""
    import scipy.sparse as ssp

    def knn(res, index, queries, n_cols, k, metric, bi, bq, I_t, D_t):
        ip, ix, iv, n, nnz = index
        qp, qx, qv, nq, qnnz = queries
        assert bi >= 1 and bq >= 1 and 1 <= k <= n
        calls.append((n, k, metric, bi, bq))
        X = ssp.csr_matrix((iv.numpy(), ix.numpy(), ip.numpy()), shape=(n, n_cols))
        Q = ssp.csr_matrix((qv.numpy(), qx.numpy(), qp.numpy()), shape=(nq, n_cols))
        m = "cosine" if metric == 1 else "ip"
        D = np.empty((nq, k), np.float32)
        I = np.empty((nq, k), np.int64)
        for q0 in range(0, nq, bq):
            q1 = min(nq, q0 + bq)
            parts_d, parts_i = [], []
            for r0 in range(0, n, bi):
                r1 = min(n, r0 + bi)
                d, i = sparse.exact_topk(Q[q0:q1], X[r0:r1], k, m, np.float32)
                parts_d.append(d)
                parts_i.append(i + r0)
            d, i = np.concatenate(parts_d, 1), np.concatenate(parts_i, 1)
            for r in range(q1 - q0):
                o = np.lexsort((i[r], -d[r]))[:k]
                D[q0 + r], I[q0 + r] = d[r][o], i[r][o]
        if metric == 1:
            D = 1 - D
        I_t.copy_(torch.from_numpy(I.astype(np.int32)))
        D_t.copy_(torch.from_numpy(D.astype(np.float32)))
    return knn


# 3 < K: an index batch smaller than k; 7 / 333 divide none of ROWS; 1 query a batch
TILES = [None, (3, "stock"), (7, 11), (333, 1), ("file", "stock"), ("stock", 16),
         ("file", 5)]


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_cuvs_sparse_tiles_reach_every_call_and_change_nothing(sparse_corpus, fake_cuvs_pkg,
                                                               metric):
    """The effective pair reaches every engine call (``file``: that block's
    row count; ``stock``: the build default; None: stock on both sides), the
    self-test keeps batch_size_index 2, and a tiled exact engine gives the
    same answer at every tile."""
    cdir, qp, _, _ = sparse_corpus
    q, _ = sparse.load_queries(qp, "sparse_embedding")
    src = InMemorySource.load(cdir, "sparse_embedding", codec=sparse.CODEC)
    stock = cuvs_engine.SPARSE_STOCK_BATCHES
    res = {}
    for tile in TILES:
        calls = []
        b = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=tiling_knn(calls), metric=metric,
                                  batches=tile)
        b.setup(q)
        res[tile] = loop.run_pass(src, b, q.n_rows, K)[:2]
        probe, calls = calls[0], calls[1:]
        assert probe[3] == 2                       # the self-test ignores the flag
        ri, rq = tile or ("stock", "stock")
        want = [(stock[0] if ri == "stock" else n if ri == "file" else ri,
                 stock[1] if rq == "stock" else rq) for n in ROWS]
        assert [c[3:] for c in calls] == want, tile
        assert b.batches == want                # the self-test's call is not recorded
    D0, I0 = res[None]
    for tile, (D, I) in res.items():
        np.testing.assert_allclose(D, D0, rtol=1e-6, atol=1e-6, err_msg=str(tile))
        np.testing.assert_array_equal(I, I0, err_msg=str(tile))


@pytest.mark.parametrize("flag,want", [
    (None, "32768x32768"), ("8192xstock", "8192x32768"), ("stockx2500", "32768x2500"),
    ("filexstock", "filex32768"), ("filex5", "filex5"), ("3x7", "3x7"),
])
def test_cli_sparse_batches_column_records_the_effective_pair(
        sparse_corpus, fake_cuvs_pkg, monkeypatch, tmp_path, flag, want):
    cdir, qp, _, _ = sparse_corpus
    calls = []
    monkeypatch.setattr(run, "make_backend", lambda args, dtype: cuvs_engine.SparseCuvsBackend(
        args.k, device="cpu", events=not args.no_gpu_events, metric=args.metric,
        knn=tiling_knn(calls), batches=args.cuvs_sparse_batches))
    out = tmp_path / "r.csv"
    run.main(_tile_argv(cdir, qp, out, "--verify", "--gt-ids-column", "gt_ids",
                        *([] if flag is None else ["--cuvs-sparse-batches", flag])))
    (r,) = read_rows(out)
    assert (r["sparse_batches"], r["verified"]) == (want, "ok")
    eff = cuvs_engine.parse_cuvs_sparse_batches(flag) if flag else ("stock", "stock")
    passes = calls[1:]
    assert [c[3:] for c in passes] == [cuvs_engine.cuvs_sparse_effective_batches(
        eff, cuvs_engine.SPARSE_STOCK_BATCHES, n) for n in ROWS]


def test_make_backend_passes_the_flag(fake_cuvs_pkg, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, k, **kw):
            seen.update(kw)
    monkeypatch.setattr(cuvs_engine, "SparseCuvsBackend", Fake)
    monkeypatch.setattr(cuvs_engine, "cuvs_set_rmm_pool", lambda spec: None)
    args = run.parse_args(["--regime", "stream", "--corpus", "c", "--csv", "x.csv",
                           "--query-file", "q.parquet", "--modality", "sparse", "--backend", "cuvs",
                           "--cuvs-sparse-batches", "filex2500"])
    run.make_backend(args, np.float32)
    assert seen["batches"] == ("file", 2500)


# nq 100000, k 10, largest file 1200 rows: the tile is
# min(bq, nq) x min(bi, 1200) x 8 B + min(bq, nq) x k x 32 B, by hand
@pytest.mark.parametrize("flag,tile", [
    (None, 32768 * 1200 * 8 + 32768 * K * 32),                 # stock
    ((65536, 65536), 65536 * 1200 * 8 + 65536 * K * 32),       # a LARGER tile is charged
    (("file", 100_000), 100_000 * 1200 * 8 + 100_000 * K * 32),  # file: the largest file
    ((8, "stock"), 32768 * 8 * 8 + 32768 * K * 32),
])
def test_overlap_guard_charges_the_effective_tile(sparse_corpus, fake_cuvs_pkg, flag, tile):
    """_sparse_overlap_guard sizes cuVS's tile from the EFFECTIVE pair, so a
    larger tile cannot slip past a guard computed for the stock one: it
    passes at exactly the need and refuses one byte less, with the pair
    named. The same need with or without the backend (args only)."""
    from types import SimpleNamespace as NS
    cdir, qp, _, _ = sparse_corpus
    src = ParquetSource(cdir, "sparse_embedding", prefetch=0, codec=sparse.CODEC,
                        read_mode="arrow")
    try:
        feet = src.read_footers()
        f = [feet[r.path] for r in src.files]
        assert max(g.num_rows for g in f) == 1200
        block = max(sparse.CODEC.block_bytes(g, None) for g in f)
        nq = 100_000
        need = (2 * block + nq * K * guards.OVERLAP_RESULT_BYTES_PER_CELL + tile
                + guards.OVERLAP_GPU_MARGIN)
        args = NS(widen="device", backend="cuvs", k=K, cuvs_sparse_batches=flag)
        b = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=fake_knn([]), batches=flag)
        bi, bq = b.params(1200, nq)
        for backend in (b, None):
            guards._sparse_overlap_guard(args, src, feet, nq, need, need, backend)
            with pytest.raises(SystemExit, match=f"{bi}x{bq}"):
                guards._sparse_overlap_guard(args, src, feet, nq, need - 1, need, backend)
    finally:
        src.close()
