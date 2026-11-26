"""--modality multivector: the packed ragged block and its decode (nulls,
empty lists, row groups / chunks, fp16 / fp64, an inner fixed_size_list,
ring-slot decode, refusals), the footer bound, the queries, the exact numpy
MaxSim engine and the float64 oracle against a per-pair brute force, every
read mode (native: nova-bf's rust_fetch FETCH path, byte-identical blocks),
and --backend pylate (PyLate's exact float32 torch path; its three modes) on
torch's CPU device through the harness: every --h2d / --d2h transfer mode
(--verify, the ground truth, identical to the pageable run), ip and cosine,
an fp16 corpus widened on the device, tiny scratch budgets, zero-token
documents / queries, the released mode's zero-padding inexactness, the
fp32-matmul refusal, the CLI refusals, the CSV columns and the memory
guards. The real GPU run is owed."""

import csv
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import make_dummy
from bench.data.formats import multivector as multivec
from bench.systems import harness_cli as run
from bench.systems import guards
from bench.record import results_io
from bench.systems import streaming as base, pylate_engine
from bench.data.sources import InMemorySource, ParquetSource, list_parquet, read_file, read_footer
from bench.timing import Stages

COL = "multivector_embedding"
K = 10
DIM = 16
ROWS = [70, 2, 55, 1, 90]


def brute_maxsim(q_off, q_tok, d_off, d_tok, metric="ip"):
    """Per (query, document) pair float64 MaxSim, -inf for an empty side."""
    q = np.asarray(q_tok, np.float64)
    d = np.asarray(d_tok, np.float64)
    if metric == "cosine":
        q, d = multivec.unit_tokens(q, np.float64), multivec.unit_tokens(d, np.float64)
    nq, n = len(q_off) - 1, len(d_off) - 1
    S = np.full((nq, n), -np.inf)
    for a in range(nq):
        qa = q[q_off[a]:q_off[a + 1]]
        for b in range(n):
            db = d[d_off[b]:d_off[b + 1]]
            if len(qa) and len(db):
                S[a, b] = (qa @ db.T).max(axis=1).sum()
    return S


def topk_ref(S, k):
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    D = np.take_along_axis(S, order, axis=1)
    I = order.astype(np.int64)
    D, I = multivec.to_padding(D.astype(np.float32), I)
    return D, I


@pytest.fixture(scope="module")
def mv_corpus(tmp_path_factory):
    """Five multi-vector files (row groups of 25; ~4% zero-token documents),
    the queries (one zero-token query, exact gt_ids) and the parts. Tokens
    non-negative, so every token score is >= 0 and rerank-released's zero
    padding is exact here too (test_cli_negative_scores covers the rest)."""
    out = tmp_path_factory.mktemp("mv")
    (out / "c").mkdir()
    parts = []
    for i, n in enumerate(ROWS):
        p = make_dummy.random_multivector(np.random.default_rng([7, i]), n, DIM, 5, 0.04,
                                          nonneg=True)
        make_dummy.write_multivector(str(out / "c" / f"part-{i:05d}.parquet"), p,
                                     row_group_size=25)
        parts.append(p)
    q = make_dummy.random_multivector(np.random.default_rng(3), 24, DIM, 4, nonneg=True)
    # one zero-token query
    q_off = np.concatenate([q[0][:6], q[0][5:]])
    q = (q_off, q[1])
    qp = str(out / "q.parquet")
    gt = make_dummy.mv_ground_truth(parts, q, K)
    make_dummy.write_multivector(qp, q, extra={"gt_ids": pa.array(list(gt),
                                                                   pa.list_(pa.int64()))})
    return str(out / "c"), qp, parts, q


def concat(parts):
    offs, toks, base = [np.zeros(1, np.int64)], [], 0
    for o, t in parts:
        offs.append(o[1:] + base)
        toks.append(t)
        base += int(o[-1])
    return np.concatenate(offs), np.concatenate(toks)


def read_rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------- block / decode

def test_pack_unpack_roundtrip_and_alignment():
    x = multivec.new_packed(5, 9, 3, np.float16)
    r = multivec.unpack(x)
    assert multivec.is_packed(x) and multivec.packed_rows(x) == 5
    assert multivec.packed_tokens(x) == 9 and r.tokens.shape == (9, 3)
    assert r.tokens.dtype == np.float16
    n, nt, dim, vb, o_o, o_t = multivec.header_fields(x)
    assert (n, nt, dim, vb) == (5, 9, 3, 2) and o_o % 256 == 0 and o_t % 256 == 0
    assert x.nbytes == multivec.packed_nbytes(5, 27, 2)
    assert not multivec.is_packed(np.zeros(300, np.uint8))
    from bench.data.formats import sparse
    assert not sparse.is_packed(x) and not multivec.is_packed(sparse.new_packed(2, 2, np.float32))


def _table(docs, dtype=pa.float32(), inner_fixed=None, chunks=1):
    """docs: list of None | list of token lists."""
    inner = pa.list_(dtype, inner_fixed) if inner_fixed else pa.list_(dtype)
    arr = pa.array(docs, pa.list_(inner))
    if chunks > 1:
        step = -(-len(docs) // chunks)
        arr = pa.chunked_array([arr[i:i + step] for i in range(0, len(docs), step)])
    return pa.table({COL: arr})


DOCS = [[[1, 2], [3, 4]], None, [], [[5, 6]], [[7, 8], [9, 10], [11, 12]]]


@pytest.mark.parametrize("chunks", [1, 2, 5])
@pytest.mark.parametrize("inner_fixed", [None, 2])
def test_decode_nulls_empties_chunks_fixed(chunks, inner_fixed):
    x = multivec.table_ragged(_table(DOCS, inner_fixed=inner_fixed, chunks=chunks), COL)
    r = multivec.unpack(x)
    assert r.n_rows == 5 and r.n_tok == 6 and r.dim == 2
    np.testing.assert_array_equal(r.offsets, [0, 2, 2, 2, 3, 6])
    np.testing.assert_array_equal(r.tokens, np.arange(1, 13, dtype=np.float32).reshape(6, 2))


def test_decode_sliced_chunk_and_null_slot_carrying_tokens():
    # a sliced array (non-zero offset) and a null slot whose offsets span tokens
    vals = pa.array(np.arange(12, dtype=np.float32))
    inner = pa.ListArray.from_arrays(pa.array([0, 2, 4, 6, 8, 10, 12], pa.int32()), vals)
    outer = pa.ListArray.from_arrays(pa.array([0, 2, 4, 6], pa.int32()), inner,
                                     mask=pa.array([False, True, False]))
    x = multivec.table_ragged(pa.table({COL: outer}), COL)
    r = multivec.unpack(x)
    np.testing.assert_array_equal(r.offsets, [0, 2, 2, 4])
    np.testing.assert_array_equal(r.tokens.ravel(), [0, 1, 2, 3, 8, 9, 10, 11])
    sl = multivec.table_ragged(pa.table({COL: outer.slice(1)}), COL)
    np.testing.assert_array_equal(multivec.unpack(sl).offsets, [0, 0, 2])
    np.testing.assert_array_equal(multivec.unpack(sl).tokens.ravel(), [8, 9, 10, 11])


@pytest.mark.parametrize("dt, want", [(pa.float16(), np.float16), (pa.float32(), np.float32),
                                      (pa.float64(), np.float32)])
def test_decode_value_types(dt, want):
    r = multivec.unpack(multivec.table_ragged(_table(DOCS, dt), COL))
    assert r.tokens.dtype == want
    np.testing.assert_array_equal(r.tokens.astype(np.float64).ravel(), np.arange(1, 13))


def test_decode_into_buffer_and_room_check():
    t = _table(DOCS)
    need = multivec.packed_nbytes(5, 12, 4)
    buf = bytearray(need + 512)
    x = multivec.table_ragged(t, COL, out=memoryview(buf))
    assert x.base is not None and multivec.unpack(x).tokens.sum() == 78
    with pytest.raises(ValueError, match="ring"):
        multivec.table_ragged(t, COL, out=memoryview(bytearray(need - 1)))


@pytest.mark.parametrize("docs, match", [
    ([[[1, 2], [3]]], "different widths"),
    ([[[1, None]]], "null token values"),
    ([[[1, 2], None]], "null token vector"),
    ([[[]]], "zero-width"),
])
def test_decode_refusals(docs, match):
    with pytest.raises(ValueError, match=match):
        multivec.table_ragged(_table(docs), COL)


def test_decode_refuses_non_multivector_and_widths_across_chunks():
    with pytest.raises(TypeError, match="list<list<float>>"):
        multivec.table_ragged(pa.table({COL: pa.array([[1.0, 2.0]])}), COL)
    a = pa.array([[[1.0, 2.0]]], pa.list_(pa.list_(pa.float32())))
    b = pa.array([[[1.0, 2.0, 3.0]]], pa.list_(pa.list_(pa.float32())))
    with pytest.raises(ValueError, match="token widths"):
        multivec.table_ragged(pa.table({COL: pa.chunked_array([a, b])}), COL)


def test_all_empty_block():
    r = multivec.unpack(multivec.table_ragged(_table([None, []]), COL))
    assert r.n_rows == 2 and r.n_tok == 0 and r.dim == 0
    np.testing.assert_array_equal(r.offsets, [0, 0, 0])


def test_footer_bounds_the_block(mv_corpus, tmp_path):
    cdir, _, _, _ = mv_corpus
    fs, refs = list_parquet(cdir)
    for ref in refs:
        f = read_footer(fs, ref.path, COL)
        x = read_file(fs, ref.path, COL, codec=multivec.CODEC)
        assert f.mv_value_bits == 32 and f.dim is None and f.sparse_nnz is None
        assert multivec.CODEC.block_bytes(f, None) >= x.nbytes
        assert f.mv_values >= multivec.packed_tokens(x) * DIM
    p = make_dummy.write_vectors(str(tmp_path / "d.parquet"),
                                 np.ones((3, 4), np.float32), "embedding")
    dfs, drefs = list_parquet(p)
    f = read_footer(dfs, drefs[0].path, "embedding")
    assert f.mv_values is None and f.mv_value_bits is None and f.dim == 4
    assert multivec.CODEC.block_bytes(f, None) is None


@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_read_modes_and_host_widen(tmp_path, mode):
    p = make_dummy.random_multivector(np.random.default_rng(1), 60, 8, 4, 0.1)
    path = make_dummy.write_multivector(str(tmp_path / "a.parquet"), p,
                                        value_dtype=np.float16, row_group_size=13)
    fs, refs = list_parquet(path)
    st, rec = Stages(), {}
    x = read_file(fs, refs[0].path, COL, stats=st, size=refs[0].size, mode=mode,
                  range_bytes=1000, codec=multivec.CODEC, rec=rec)
    r = multivec.unpack(x)
    assert r.tokens.dtype == np.float16 and st.as_dict()["rows"] == 60
    np.testing.assert_array_equal(r.offsets, p[0])
    np.testing.assert_array_equal(r.tokens, p[1].astype(np.float16))
    w = read_file(fs, refs[0].path, COL, dtype=np.float32, stats=st, size=refs[0].size,
                  mode=mode, range_bytes=1000, codec=multivec.CODEC, rec=rec)
    assert multivec.unpack(w).tokens.dtype == np.float32 and rec["widen"] > 0
    np.testing.assert_array_equal(multivec.unpack(w).tokens, r.tokens.astype(np.float32))


@pytest.mark.parametrize("prefetch", [2, 0])
def test_sources_offsets_and_blocks(mv_corpus, prefetch):
    cdir, _, parts, _ = mv_corpus
    src = ParquetSource(cdir, COL, prefetch=prefetch, workers=2, codec=multivec.CODEC)
    try:
        offs, got = [], []
        for b in src:
            r = multivec.unpack(b.vectors)
            offs.append(b.offset)
            got.append((r.offsets.copy(), r.tokens.copy()))
            del b, r
    finally:
        src.close()
    assert offs == list(np.cumsum([0] + ROWS[:-1]))
    for (o, t), (po, pt) in zip(got, parts):
        np.testing.assert_array_equal(o, po)
        np.testing.assert_array_equal(t, pt)
    mem = InMemorySource.load(cdir, COL, codec=multivec.CODEC)
    assert [b.offset for b in mem.blocks] == offs


def test_native_read_identical_to_arrow(tmp_path, monkeypatch):
    pytest.importorskip("rust_fetch")
    from bench.data import sources
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_CANARY", "1")
    p = make_dummy.random_multivector(np.random.default_rng(4), 80, 12, 6, 0.05)
    path = make_dummy.write_multivector(
        str(tmp_path / "n.parquet"), p, row_group_size=17,
        extra={"text": pa.array([f"doc {i} " * 30 for i in range(80)])})
    fs, refs = list_parquet(path)
    try:
        a = read_file(fs, refs[0].path, COL, size=refs[0].size, mode="arrow",
                      codec=multivec.CODEC)
        rec = {}
        b = read_file(fs, refs[0].path, COL, size=refs[0].size, mode="native",
                      codec=multivec.CODEC, rec=rec)
    finally:
        sources.native_drop_clients()
    assert rec["read_path"] == "native-fetch"
    # the header and every part (the alignment padding between them is not written)
    assert multivec.header_fields(a) == multivec.header_fields(b)
    ra, rb = multivec.unpack(a), multivec.unpack(b)
    np.testing.assert_array_equal(ra.offsets, rb.offsets)
    np.testing.assert_array_equal(ra.tokens, rb.tokens)
    np.testing.assert_array_equal(ra.offsets, p[0])
    np.testing.assert_array_equal(ra.tokens, p[1])


# ------------------------------------------------------------ queries

def test_load_queries_truncates_and_warns(mv_corpus):
    _, qp, _, q = mv_corpus
    got, notes = multivec.load_queries(qp, COL)
    np.testing.assert_array_equal(got.offsets, q[0])
    np.testing.assert_array_equal(got.tokens, q[1])
    assert got.tokens.dtype == np.float32 and any("no tokens" in n for n in notes)
    few, notes = multivec.load_queries(qp, COL, nq=5)
    assert few.n_rows == 5 and few.n_tok == int(q[0][5]) and not any("no tokens" in n
                                                                     for n in notes)
    _, notes = multivec.load_queries(qp, COL, nq=1000)
    assert any("--nq 1000" in n for n in notes)


# ------------------------------------------------ exact engine / oracle

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("tile", [multivec.CPU_TILE_CELLS, 7, 1])
def test_maxsim_matrix_matches_bruteforce(mv_corpus, dtype, tile):
    _, _, parts, q = mv_corpus
    off, tok = concat(parts)
    S = multivec.maxsim_matrix(q[0], q[1], off, tok, dtype, tile_cells=tile)
    ref = brute_maxsim(q[0], q[1], off, tok)
    assert np.array_equal(np.isneginf(S), np.isneginf(ref))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(S[fin], ref[fin], rtol=1e-5 if dtype == np.float32 else 1e-12,
                               atol=1e-5)
    assert np.isneginf(S[5]).all()  # the zero-token query
    assert np.isneginf(S[:, np.diff(off) == 0]).all()  # zero-token documents


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_oneshot_reference_and_rows(mv_corpus, metric):
    cdir, _, parts, q = mv_corpus
    src = InMemorySource.load(cdir, COL, codec=multivec.CODEC)
    qq = multivec.Queries(q[0], q[1])
    D, I, xq, xb = multivec.oneshot_reference(qq, src.blocks, K + 1, metric)
    off, tok = concat(parts)
    S = brute_maxsim(q[0], q[1], off, tok, metric)
    Dr, Ir = topk_ref(S, K + 1)
    np.testing.assert_array_equal(I, Ir)
    np.testing.assert_allclose(D, Dr, rtol=1e-6)
    assert (I[5] == -1).all()  # the zero-token query retrieves nothing
    assert len(xb) == sum(ROWS)
    for a, b in [(0, 3), (7, 100), (2, int(I[2, 0]))]:
        assert xb.pair_dot(xq, a, b) == pytest.approx(S[a, b], rel=1e-12)
    empty_doc = int(np.flatnonzero(np.diff(off) == 0)[0])
    assert xb.pair_dot(xq, 0, empty_doc) == -np.inf


def test_ground_truth_writer_equals_oracle(mv_corpus):
    cdir, qp, parts, q = mv_corpus
    gt = np.stack(pq.read_table(qp, columns=["gt_ids"]).column(0).to_pylist())
    src = InMemorySource.load(cdir, COL, codec=multivec.CODEC)
    _, I, _, _ = multivec.oneshot_reference(multivec.Queries(*q), src.blocks, K)
    np.testing.assert_array_equal(gt, I)


# ------------------------------------------------------------ CLI: cpu

def base_args(cdir, qp, out, *extra):
    return ["--modality", "multivector", "--corpus", cdir, "--column", COL,
            "--query-file", qp, "--query-column", COL, "--k", str(K), "--csv", str(out),
            "--repeats", "1", *extra]


@pytest.mark.parametrize("regime, extra", [("stream", ["--prefetch", "2"]),
                                           ("stream", ["--prefetch", "0"])])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_cli_cpu_verify_gt_and_csv(mv_corpus, tmp_path, regime, extra, metric, capsys):
    cdir, qp, parts, q = mv_corpus
    out = tmp_path / "r.csv"
    run.main(base_args(cdir, qp, out, "--backend", "cpu", "--regime", regime,
                       "--metric", metric, "--verify", "--gt-ids-column", "gt_ids",
                       "--save-results", str(tmp_path / "res"), *extra))
    rows = read_rows(out)
    r = rows[-1]
    assert r["verified"] == "ok" and r["modality"] == "multivector" and r["engine"] == "numpy"
    assert int(r["tokens"]) == sum(int(o[-1]) for o, _ in parts) and r["nnz"] == ""
    assert r["d"] == str(DIM) and r["pylate_mode"] == "n/a" and r["stored_dtype"] == "f32"
    if metric == "ip":
        assert float(r["gt_recall"]) == 1.0
    res = np.load(tmp_path / "res.npz")
    off, tok = concat(parts)
    _, Ir = topk_ref(brute_maxsim(q[0], q[1], off, tok, metric), K)
    np.testing.assert_array_equal(res["I"], Ir)
    assert "verify OK" in capsys.readouterr().out


def test_cli_f16_corpus_widens_on_host(tmp_path):
    parts = []
    (tmp_path / "c").mkdir()
    for i in range(2):
        p = make_dummy.random_multivector(np.random.default_rng([9, i]), 40, 8, 5)
        make_dummy.write_multivector(str(tmp_path / "c" / f"p{i}.parquet"), p,
                                     value_dtype=np.float16)
        parts.append(p)
    qp = make_dummy.make_mv_queries(str(tmp_path / "q.parquet"), 9, 8, 3, corpus_parts=parts,
                                    gt_k=K, value_dtype=np.float16)
    out = tmp_path / "r.csv"
    run.main(base_args(str(tmp_path / "c"), qp, out, "--backend", "cpu", "--regime",
                       "stream", "--verify", "--gt-ids-column", "gt_ids"))
    r = read_rows(out)[-1]
    assert (r["verified"], r["stored_dtype"], r["widen_where"]) == ("ok", "f16", "host")
    assert float(r["gt_recall"]) == 1.0


# --------------------------------------------------------- CLI: pylate

TRANSFER_MODES = {
    "pageable": [],
    "h2d_pinned": ["--h2d", "pinned"],
    "h2d_overlap": ["--h2d", "pinned", "--h2d-overlap"],
    "d2h_pinned": ["--d2h", "pinned"],
    "d2h_overlap": ["--d2h", "pinned", "--d2h-overlap"],
    "all_overlap": ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned", "--d2h-overlap"],
}
MODES = pylate_engine.PYLATE_MODES


def pylate_backend(max_bytes=None):
    def make(args, dtype):
        return pylate_engine.MvPylateBackend(args.k, args.pylate_mode, device="cpu",
                                   events=not args.no_gpu_events, metric=args.metric,
                                   max_bytes=max_bytes or args.pylate_max_mib << 20,
                                   rerank_results=args.pylate_rerank_results or "device")
    return make


@pytest.mark.slow
@pytest.mark.parametrize("events", [True, False])
@pytest.mark.parametrize("tmode", sorted(TRANSFER_MODES))
@pytest.mark.parametrize("mode", MODES)
def test_cli_pylate_transfer_modes(mv_corpus, monkeypatch, tmp_path, mode, tmode, events):
    """PyLate through ResidentBackend's submit/complete in every --h2d / --d2h
    mode: --verify passes, the ground truth is recalled exactly, the CSV
    records engine / mode / budget, the results are identical to the
    pageable run's (ids globalised on the device) and equal the brute force."""
    cdir, qp, parts, q = mv_corpus
    monkeypatch.setattr(run, "make_backend", pylate_backend())
    res = {}
    for name in ("pageable", tmode):
        out = tmp_path / f"{name}.csv"
        run.main(base_args(cdir, qp, out, "--backend", "pylate", "--pylate-mode", mode,
                           "--regime", "stream", "--warmup", "1", "--repeats", "2",
                           "--verify", "--gt-ids-column", "gt_ids", "--prefetch", "2",
                           "--save-results", str(tmp_path / f"res_{name}_{{engine}}"),
                           *([] if events else ["--no-gpu-events"]), *TRANSFER_MODES[name]))
        rows = read_rows(out)
        assert {r["verified"] for r in rows if r["verified"]} == {"ok"}
        assert {float(r["gt_recall"]) for r in rows if r["gt_recall"]} == {1.0}
        r = rows[-1]
        assert (r["engine"], r["backend"], r["pylate_mode"], r["pylate_batches"],
                r["queries_mode"], r["widen_where"], r["temp_mem"]) == \
            ("pylate", "pylate", mode,
             "max_bytes=8192MiB" + (",per-query" if pylate_engine.pylate_released(mode)
                                    else ",query_buckets=1"),
             "resident", "device", "n/a")
        assert r["engine_version"] == f"1.6.0+{pylate_engine.PYLATE_COMMIT[:7]}"
        assert r["pylate_rerank_results"] == ("device" if pylate_engine.pylate_released(mode) else "n/a")
        # one call per block with tokens at the default budget (a 1-document
        # file may hold none)
        assert len(ROWS) - 1 <= int(r["pylate_calls"]) <= len(ROWS)
        assert r["pylate_calls_max_block"] == "1"
        res[name] = np.load(tmp_path / f"res_{name}_pylate.npz")
    np.testing.assert_array_equal(res[tmode]["I"], res["pageable"]["I"])
    np.testing.assert_array_equal(res[tmode]["D"], res["pageable"]["D"])
    off, tok = concat(parts)
    Dr, Ir = topk_ref(brute_maxsim(q[0], q[1], off, tok), K)
    np.testing.assert_array_equal(res["pageable"]["I"], Ir)
    np.testing.assert_allclose(res["pageable"]["D"], Dr, rtol=1e-5)


RERANK_MODES = [m for m in MODES if pylate_engine.pylate_released(m)]


@pytest.mark.slow
@pytest.mark.parametrize("events", [True, False])
@pytest.mark.parametrize("tmode", sorted(TRANSFER_MODES))
@pytest.mark.parametrize("mode", RERANK_MODES)
def test_cli_pylate_rerank_results_host_equals_device(mv_corpus, monkeypatch, tmp_path, mode,
                                                      tmode, events):
    """--pylate-rerank-results host (results kept on the host, ids globalised
    there, no upload / copy back) against the default device, in the same
    transfer mode, with a 3000 B budget (several rerank calls a block, merged
    by FAISS's ResultHeap): identical saved results, --verify ok, recall 1,
    the choice in the CSV and the trace; host's d2h_s is 0 when measured,
    blank with events off (as every event column)."""
    cdir, qp, _, _ = mv_corpus
    monkeypatch.setattr(run, "make_backend", pylate_backend(max_bytes=3000))
    res, rows = {}, {}
    for where in pylate_engine.PYLATE_RERANK_RESULTS:
        out = tmp_path / f"{where}.csv"
        run.main(base_args(cdir, qp, out, "--backend", "pylate", "--pylate-mode", mode,
                           "--pylate-rerank-results", where, "--regime", "stream",
                           "--warmup", "1", "--repeats", "2", "--verify", "--gt-ids-column",
                           "gt_ids", "--prefetch", "2",
                           "--save-results", str(tmp_path / f"res_{where}_{{engine}}"),
                           *([] if events else ["--no-gpu-events"]), *TRANSFER_MODES[tmode]))
        rows[where] = read_rows(out)
        res[where] = np.load(tmp_path / f"res_{where}_pylate.npz")
        trows = read_rows(results_io.trace_path(str(out)))
        assert {t["pylate_rerank_results"] for t in trows} == {where}
    for where, rs in rows.items():
        assert {r["verified"] for r in rs if r["verified"]} == {"ok"}
        assert {float(r["gt_recall"]) for r in rs if r["gt_recall"]} == {1.0}
        r = rs[-1]
        assert r["pylate_rerank_results"] == where
        assert int(r["pylate_calls_max_block"]) > 1  # the merge ran
        if events:
            assert float(r["search_s"]) > 0
            if where == "host":
                assert float(r["d2h_s"]) == 0.0
        else:
            assert (r["search_s"], r["d2h_s"]) == ("", "")
    np.testing.assert_array_equal(res["host"]["I"], res["device"]["I"])
    np.testing.assert_array_equal(res["host"]["D"], res["device"]["D"])


def test_pylate_host_results_flow():
    """The host path through submit / complete: results stay CPU tensors over
    the backend's own arrays (no _to_host, no pinned buffer even with --d2h
    pinned), ids globalised in place (-1 kept), e4 is e3; the device path is
    untouched (host_results False)."""
    q = make_dummy.random_multivector(np.random.default_rng(4), 3, 4, 2, nonneg=True)
    q = (np.concatenate([q[0][:2], q[0][1:]]), q[1])  # query 1: no tokens
    d = (np.array([0, 2, 2, 5, 5, 6]), np.abs(make_dummy.unit_vectors(
        np.random.default_rng(3), 6, 4)))
    out = {}
    for where in pylate_engine.PYLATE_RERANK_RESULTS:
        b = pylate_engine.MvPylateBackend(4, "rerank-released", device="cpu", rerank_results=where)
        assert b.host_results == (where == "host")
        b.configure_h2d("pageable")
        b.configure_d2h("pinned")
        b.setup(multivec.Queries(*q))
        tok = b.submit(_block(d), offset=100)
        e3, e4 = tok["e"][4:]
        assert (e4 is e3) == (where == "host")
        out[where] = b.complete(tok)
        assert (b._d2h_bufs == {}) == (where == "host")  # host: nothing copied into one
    for where, (D, I) in out.items():
        assert (I[1] == -1).all() and (I[:, 3:] == -1).all()  # 3 documents with tokens
        assert set(I[0, :3].tolist()) == {100, 102, 104}
    np.testing.assert_array_equal(out["host"][0], out["device"][0])
    np.testing.assert_array_equal(out["host"][1], out["device"][1])
    assert not any(c.host_results for c in (base.ResidentBackend(4, np.float32, "cpu", True),
                                            pylate_engine.MvPylateBackend(4, "scores-batched",
                                                                device="cpu")))
    with pytest.raises(ValueError, match="rerank modes only"):
        pylate_engine.MvPylateBackend(4, "scores-batched", device="cpu", rerank_results="host")
    with pytest.raises(ValueError):
        pylate_engine.MvPylateBackend(4, "rerank-released", device="cpu", rerank_results="gpu")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_cli_pylate_metrics_inline_reader(mv_corpus, monkeypatch, tmp_path, mode, metric):
    cdir, qp, _, _ = mv_corpus
    monkeypatch.setattr(run, "make_backend", pylate_backend())
    out = tmp_path / "r.csv"
    run.main(base_args(cdir, qp, out, "--backend", "pylate", "--pylate-mode", mode,
                       "--regime", "stream", "--prefetch", "0", "--metric", metric,
                       "--verify"))
    r = read_rows(out)[-1]
    assert r["verified"] == "ok" and r["metric"] == metric
    assert (r["normalize_s"] != "") == (metric == "cosine")


@pytest.mark.parametrize("mode", MODES)
def test_cli_pylate_f16_corpus(tmp_path, monkeypatch, mode):
    widen = "device"
    (tmp_path / "c").mkdir()
    parts = []
    for i in range(2):
        p = make_dummy.random_multivector(np.random.default_rng([11, i]), 30, 8, 4, 0.1,
                                          nonneg=True)
        make_dummy.write_multivector(str(tmp_path / "c" / f"p{i}.parquet"), p,
                                     value_dtype=np.float16)
        parts.append(p)
    qp = make_dummy.make_mv_queries(str(tmp_path / "q.parquet"), 7, 8, 3, corpus_parts=parts,
                                    gt_k=K, value_dtype=np.float16, nonneg=True)
    monkeypatch.setattr(run, "make_backend", pylate_backend())
    out = tmp_path / "r.csv"
    run.main(base_args(str(tmp_path / "c"), qp, out, "--backend", "pylate", "--pylate-mode",
                       mode, "--regime", "stream", "--verify", "--gt-ids-column", "gt_ids"))
    r = read_rows(out)[-1]
    assert (r["verified"], r["stored_dtype"], r["widen_where"]) == ("ok", "f16", widen)
    assert float(r["gt_recall"]) == 1.0
    if widen == "device":
        assert float(r["widen_s"]) > 0


# ------------------------------------------------- backend, direct

def _block(parts_or_docs):
    off, tok = parts_or_docs
    x = multivec.new_packed(len(off) - 1, len(tok), tok.shape[1] if len(tok) else 0,
                            np.float32)
    r = multivec.unpack(x)
    r.offsets[:] = off
    if len(tok):
        r.tokens[:] = tok
    return x


def _direct(mode, q, xb, k=K, max_bytes=8192 << 20, metric="ip", query_buckets=1):
    b = pylate_engine.MvPylateBackend(k, mode, device="cpu", metric=metric, max_bytes=max_bytes,
                            query_buckets=query_buckets)
    b.setup(multivec.Queries(*q))
    D, I = b.search(xb)
    # what run_pass's ResultHeap makes of -inf (zero-token documents /
    # queries): FAISS's padding
    return multivec.to_padding(D.copy(), I.copy())


@pytest.mark.parametrize("max_bytes", [1, 600, 5000, 40000])
def test_batched_tiny_budgets_equal_oracle(mv_corpus, max_bytes):
    """Budgets that force a single document per range and one query per
    batch, and a few in between: the same results as the oracle."""
    _, _, parts, q = mv_corpus
    p = parts[4]
    D, I = _direct("scores-batched", q, _block(p), max_bytes=max_bytes)
    Dr, Ir = topk_ref(brute_maxsim(q[0], q[1], *p), K)
    np.testing.assert_array_equal(I, Ir)
    np.testing.assert_allclose(D, Dr, rtol=1e-5)


@pytest.mark.parametrize("buckets", [2, 3, 7, 1000])
@pytest.mark.parametrize("max_bytes", [1, 3000, 60000, 8192 << 20])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_query_buckets_equal_oracle(gauss_corpus, buckets, max_bytes, metric):
    """--pylate-query-buckets: the length-sorted queries cut into buckets,
    each planned at its own longest query (Gaussian tokens: negative maxima,
    varied query lengths; budgets down to one query / one document a call;
    more buckets than queries): the same top-k as one bucket and the oracle."""
    _, qp, parts = gauss_corpus
    q, _ = multivec.load_queries(qp, COL)
    q = (q.offsets, q.tokens)
    p = parts[0]
    D1, I1 = _direct("scores-batched", q, _block(p), k=40, max_bytes=max_bytes, metric=metric)
    D, I = _direct("scores-batched", q, _block(p), k=40, max_bytes=max_bytes, metric=metric,
                   query_buckets=buckets)
    np.testing.assert_array_equal(I, I1)
    np.testing.assert_allclose(D, D1, rtol=1e-6, atol=1e-6)
    if metric == "ip":
        Dr, Ir = topk_ref(brute_maxsim(q[0], q[1], *p), 40)
        np.testing.assert_allclose(D, Dr, rtol=1e-5, atol=1e-6)


def test_query_buckets_pad_less(gauss_corpus, monkeypatch):
    """Bucketing pads the query batches to their bucket's longest, not the
    longest of all: fewer padded query tokens through colbert_scores."""
    _, qp, parts = gauss_corpus
    q, _ = multivec.load_queries(qp, COL)
    q = (q.offsets, q.tokens)
    import pylate.scores
    real = pylate.scores.colbert_scores
    for buckets in (1, 4):
        seen = []

        def spy(Q, D, **kw):
            seen.append(Q.shape[0] * Q.shape[1])
            return real(Q, D, **kw)
        b = pylate_engine.MvPylateBackend(40, "scores-batched", device="cpu", query_buckets=buckets)
        monkeypatch.setattr(b._scores, "colbert_scores", spy)
        b.setup(multivec.Queries(*q))
        b.search(_block(parts[0]))
        if buckets == 1:
            one = sum(seen)
        else:
            assert sum(seen) < one, (sum(seen), one)


def test_doc_chunks_plan():
    lens = np.array([3, 1, 7, 2, 2, 0, 5])
    ch = pylate_engine.mv_doc_chunks(lens, 10, 4, 100)
    assert ch[0][0] == 0 and ch[-1][1] == len(lens)
    for (a, b, L), (c, _, _) in zip(ch, ch[1:] + [(len(lens), 0, 0)]):
        assert b == c and L == lens[a:b].max()
        assert b - a == 1 or (b - a) * (L * 10 + 4) <= 100
    assert pylate_engine.mv_doc_chunks(lens, 1, 0, 10**9) == [(0, 7, 7)]
    assert [b - a for a, b, _ in pylate_engine.mv_doc_chunks(lens, 10**6, 0, 1)] == [1] * 7


def test_own_token_index():
    off = torch.tensor([0, 3, 3, 4, 6])
    ln = off[1:] - off[:-1]
    idx = pylate_engine.own_token_index(torch, off, ln, slice(0, 4), 3, 6)
    assert idx.tolist() == [[0, 1, 2], [3, 3, 3], [3, 3, 3], [4, 5, 5]]
    # an index tensor (scores-batched's length-sorted documents)
    assert pylate_engine.own_token_index(torch, off, ln, torch.tensor([3, 0]), 3, 6).tolist() == \
        [[4, 5, 5], [0, 1, 2]]
    # an empty last document indexes inside the block
    off = torch.tensor([0, 2, 2])
    assert pylate_engine.own_token_index(torch, off, off[1:] - off[:-1], slice(1, 2), 2, 2).tolist() \
        == [[1, 1]]


def _old_rerank_merge(calls, nq, qs, k_eff, k):
    """The rerank modes' block result as the harness built it before the
    merge was FAISS's: per recorded rerank call, the first min(k_eff, len)
    scores / ids of each query with tokens (-inf / -1 rows for the rest),
    then a stable descending sort of their concatenation, padded to k as
    complete() pads (lowest float32 / -1)."""
    Ds, Is = [], []
    for res in calls:
        kk = min(k_eff, len(res[0]))
        Dh = np.full((nq, kk), -np.inf, np.float32)
        Ih = np.full((nq, kk), -1, np.int64)
        for qi, r in zip(qs, res):
            Dh[qi] = [x["score"] for x in r[:kk]]
            Ih[qi] = [x["id"] for x in r[:kk]]
        Ds.append(Dh)
        Is.append(Ih)
    D, I = np.concatenate(Ds, axis=1), np.concatenate(Is, axis=1)
    order = np.argsort(-D, axis=1, kind="stable")[:, :k_eff]
    D, I = np.take_along_axis(D, order, axis=1), np.take_along_axis(I, order, axis=1)
    pad = k - k_eff
    return (np.pad(D, ((0, 0), (0, pad)), constant_values=np.finfo(np.float32).min),
            np.pad(I, ((0, 0), (0, pad)), constant_values=-1))


def _assert_same_up_to_ties(D, I, Dr, Ir, scores):
    """Identical scores (positions too: both sorted descending), and the same
    ids except among exactly tied scores, where each id must carry that
    score (``scores[q]``: id -> its score in the block's rerank lists) and
    appear once per query. A score no other candidate of the query shares
    must keep its id at its position."""
    np.testing.assert_array_equal(D, Dr)
    ties = 0
    for qi in range(D.shape[0]):
        cand = np.array(list(scores[qi].values()), np.float32)
        for j, (s, i) in enumerate(zip(D[qi], I[qi])):
            if i < 0:
                continue
            if (cand == s).sum() == 1:
                assert i == Ir[qi][j], (qi, j, s)
            else:
                ties += 1
        real = I[qi][I[qi] >= 0]
        assert len(set(real.tolist())) == len(real)
        assert all(scores[qi][i] == D[qi][j] for j, i in enumerate(I[qi]) if i >= 0)
        assert ((I[qi] < 0) == (Ir[qi] < 0)).all()
    return ties  # result slots whose score another candidate shares


@pytest.mark.parametrize("mode", RERANK_MODES)
@pytest.mark.parametrize("k", [5, 40])
@pytest.mark.parametrize("max_bytes", [1, 700, 3000, 8192 << 20])
def test_rerank_heap_merge_equals_old_merge(gauss_corpus, monkeypatch, mode, k, max_bytes):
    """FAISS's ResultHeap merge of the per-range lists against the old stable
    host sort, both fed the SAME recorded rerank results: identical scores,
    ids identical except among exact ties. Gaussian tokens, zero-token
    documents, a zero-token query, duplicated documents (exact ties across
    ranges), budgets down to one document per call, a block shorter than k
    (k = 40 > its 32 documents) and a single range."""
    _, qp, _ = gauss_corpus
    q, _ = multivec.load_queries(qp, COL)
    q = (np.concatenate([q.offsets[:4], q.offsets[3:]]), q.tokens)  # query 3: no tokens
    rng = np.random.default_rng(21)
    off, tok = make_dummy.random_multivector(rng, 20, DIM, 3, 0.1)
    lens = np.diff(off)
    lens = np.concatenate([lens, lens[:10], [0, 0]])  # documents 20..29 copy 0..9
    tok = np.concatenate([tok, tok[:int(off[10])]])
    off = np.concatenate([[0], np.cumsum(lens)])
    n_live = int((lens > 0).sum())
    calls = []
    import pylate.rank
    real = pylate.rank.rerank

    def spy(**kw):
        out = real(**kw)
        calls.append(out)
        return out
    monkeypatch.setattr(pylate.rank, "rerank", spy)
    b = pylate_engine.MvPylateBackend(k, mode, device="cpu", max_bytes=max_bytes)
    b.setup(multivec.Queries(*q))
    D, I = b.search(_block((off, tok)))
    nq = len(q[0]) - 1
    qs = np.flatnonzero(np.diff(q[0]) > 0)
    assert (max_bytes == 8192 << 20) == (len(calls) == 1) and len(qs) == nq - 1
    Dr, Ir = _old_rerank_merge(calls, nq, qs, min(k, n_live), k)
    scores = [{} for _ in range(nq)]
    for res in calls:
        for qi, r in zip(qs, res):
            scores[qi].update((x["id"], np.float32(x["score"])) for x in r)
    ties = _assert_same_up_to_ties(D, I, Dr, Ir, scores)
    if mode == "rerank-released-ownpad" and k > n_live:
        assert ties > 0  # every document returned: the duplicates tie exactly
    # the zero-token query and the slots beyond the block: -inf / lowest, -1, as before
    assert (I[3] == -1).all() and np.isneginf(D[3, :min(k, n_live)]).all()
    assert (D[:, n_live:] == np.finfo(np.float32).min).all() and (I[:, n_live:] == -1).all()
    if mode == "rerank-released-ownpad":  # exact: ranges cannot change a score
        S = brute_maxsim(q[0], q[1], off, tok)
        Do, _ = topk_ref(S, k)
        Dc, _ = multivec.to_padding(D.copy(), I.copy())
        np.testing.assert_allclose(Dc[:, :Do.shape[1]], Do, rtol=1e-5, atol=1e-6)


def _negative_case():
    """Document 0 has ONE token scoring -1 against the query; document 1 two
    tokens (so rerank pads document 0 with a zero token, which scores 0)."""
    q = (np.array([0, 1]), np.array([[1.0, 0.0]], np.float32))
    d = (np.array([0, 1, 3]), np.array([[-1.0, 0.0], [0.5, 0.0], [0.2, 0.0]], np.float32))
    return q, d


def test_rerank_released_zero_padding_is_not_exact():
    """The disclosed inexactness of rank.rerank as released: a document whose
    tokens all score below 0 against a query token is scored with the zero
    padding's 0 instead; ownpad and scores-batched are exact."""
    q, d = _negative_case()
    xb = _block(d)
    D, I = _direct("rerank-released", q, xb, k=2)
    assert I.tolist() == [[1, 0]] and D[0].tolist() == [0.5, 0.0]  # 0.0: should be -1
    for mode in ("rerank-released-ownpad", "scores-batched"):
        D, I = _direct(mode, q, xb, k=2)
        assert I.tolist() == [[1, 0]] and D[0].tolist() == [0.5, -1.0], mode


def test_cli_released_inexact_case_fails_verify(tmp_path, monkeypatch):
    q, d = _negative_case()
    (tmp_path / "c").mkdir()
    make_dummy.write_multivector(str(tmp_path / "c" / "p.parquet"), d)
    make_dummy.write_multivector(str(tmp_path / "q.parquet"), q)
    monkeypatch.setattr(run, "make_backend", pylate_backend())
    args = ["--modality", "multivector", "--corpus", str(tmp_path / "c"), "--column", COL,
            "--query-file", str(tmp_path / "q.parquet"), "--query-column", COL, "--k", "2",
            "--regime", "stream", "--repeats", "1", "--verify", "--backend", "pylate"]
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        run.main(args + ["--pylate-mode", "rerank-released", "--csv", str(tmp_path / "a.csv")])
    run.main(args + ["--pylate-mode", "rerank-released-ownpad", "--csv",
                     str(tmp_path / "b.csv")])
    assert read_rows(tmp_path / "b.csv")[-1]["verified"] == "ok"


@pytest.fixture(scope="module")
def gauss_corpus(tmp_path_factory):
    """Unit Gaussian tokens (half the token scores negative), short
    documents: some document's tokens all score < 0 against some query
    token, the case rerank-released's zero padding gets wrong."""
    out = tmp_path_factory.mktemp("mvg")
    (out / "c").mkdir()
    parts = []
    for i, n in enumerate([60, 45]):
        p = make_dummy.random_multivector(np.random.default_rng([13, i]), n, DIM, 3, 0.05)
        make_dummy.write_multivector(str(out / "c" / f"p{i}.parquet"), p)
        parts.append(p)
    qp = make_dummy.make_mv_queries(str(out / "q.parquet"), 12, DIM, 4, corpus_parts=parts,
                                    gt_k=40)
    return str(out / "c"), qp, parts


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_cli_negative_scores(gauss_corpus, monkeypatch, tmp_path, mode, metric):
    """k = 40 reaches documents whose best tokens score below 0: ownpad and
    scores-batched verify; rerank-released fails --verify (its zero padding
    scores such a document's max as 0: the disclosed inexactness)."""
    cdir, qp, _ = gauss_corpus
    monkeypatch.setattr(run, "make_backend", pylate_backend())
    args = base_args(cdir, qp, tmp_path / "r.csv", "--backend", "pylate", "--pylate-mode",
                     mode, "--regime", "stream", "--verify", "--metric", metric)
    args[args.index("--k") + 1] = "40"
    if mode == "rerank-released":
        with pytest.raises(SystemExit, match="VERIFY FAILED"):
            run.main(args)
    else:
        run.main(args)
        assert read_rows(tmp_path / "r.csv")[-1]["verified"] == "ok"


@pytest.mark.parametrize("mode", MODES)
def test_zero_token_documents_and_queries(mode):
    """Zero-token documents are never returned, a zero-token query returns
    nothing (padding), a block of only zero-token documents returns padding,
    and k beyond the block's documents pads."""
    rng = np.random.default_rng(5)
    # non-negative tokens: rerank-released's zero padding is exact here
    q = make_dummy.random_multivector(rng, 4, 6, 3, nonneg=True)
    q = (np.array([0, 2, 2, 5, 7, 9]), q[1][:9])  # query 1: no tokens
    off = np.array([0, 0, 3, 3, 4, 4])            # documents 0, 2, 4: no tokens
    d = (off, np.abs(make_dummy.unit_vectors(rng, 4, 6)))
    D, I = _direct(mode, q, _block(d), k=4)
    assert (I[1] == -1).all() and (D[1] == multivec.FLT_LOWEST).all()
    for r in (0, 2, 3, 4):
        assert sorted(I[r][:2].tolist()) == [1, 3] and (I[r][2:] == -1).all()
    Dr, Ir = topk_ref(brute_maxsim(q[0], q[1], *d), 4)
    np.testing.assert_array_equal(I, Ir)
    np.testing.assert_allclose(D, Dr, rtol=1e-6)
    empty = (np.zeros(3, np.int64), np.zeros((0, 6), np.float32))
    D, I = _direct(mode, q, _block(empty), k=4)
    assert (I == -1).all()


@pytest.mark.parametrize("mode", MODES)
def test_calls_counted_and_zero_token_documents_never_handed_to_rerank(mode, monkeypatch):
    d = (np.array([0, 2, 2, 5, 5, 6]), np.abs(make_dummy.unit_vectors(
        np.random.default_rng(3), 6, 4)))
    q = make_dummy.random_multivector(np.random.default_rng(4), 3, 4, 2, nonneg=True)
    seen = []
    import pylate.rank
    real = pylate.rank.rerank

    def spy(documents_ids, **kw):
        seen.append(documents_ids[0])
        return real(documents_ids=documents_ids, **kw)
    monkeypatch.setattr(pylate.rank, "rerank", spy)
    for budget, calls in ((8 << 20, 1), (1, 3)):
        b = pylate_engine.MvPylateBackend(K, mode, device="cpu", max_bytes=budget)
        b.setup(multivec.Queries(*q))
        b.search(_block(d))
        n = int(b.stats.as_dict().get("pylate_calls", 0))
        assert int(b.stats.get("pylate_calls_max_block")) == n  # one block
        if mode == "scores-batched":
            assert n >= calls
        else:
            assert n == calls
    if mode != "scores-batched":
        assert all(1 not in ids and 3 not in ids for ids in seen) and \
            sorted(i for ids in seen[-3:] for i in ids) == [0, 2, 4]


def test_all_queries_without_tokens():
    q = (np.zeros(4, np.int64), np.zeros((0, 0), np.float32))
    d = make_dummy.random_multivector(np.random.default_rng(1), 5, 6, 3)
    for mode in MODES:
        D, I = _direct(mode, q, _block(d), k=3)
        assert (I == -1).all() and D.shape == (3, 3), mode


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("max_bytes", [1, 3000, 60000])
def test_tiny_budgets_negative_scores_equal_oracle(gauss_corpus, mode, max_bytes):
    """Gaussian tokens (negative maxima occur), varied query lengths, budgets
    that force one document per range / one query per batch (scores-batched)
    or one rerank call per range (merged across ranges): the exact modes equal
    the oracle; rerank-released matches it where no negative maximum is hit."""
    _, qp, parts = gauss_corpus
    q, _ = multivec.load_queries(qp, COL)
    q = (q.offsets, q.tokens)
    p = parts[0]
    D, I = _direct(mode, q, _block(p), k=40, max_bytes=max_bytes)
    S = brute_maxsim(q[0], q[1], *p)
    Dr, Ir = topk_ref(S, 40)
    if mode == "rerank-released":
        # zero padding to the candidate range's longest can only RAISE a score
        # (max(real tokens, 0) >= max(real tokens)), so its scores depend on
        # which documents share the range: never below the true score
        ok = I >= 0
        true = S[np.nonzero(ok)[0], I[ok]]
        assert (D[ok] >= true - 1e-5).all()
        if max_bytes == 1:  # one candidate per call: nothing padded, exact
            np.testing.assert_allclose(D, Dr, rtol=1e-5, atol=1e-6)
        else:
            assert (D[ok] > true + 1e-4).any()
        return
    np.testing.assert_allclose(D, Dr, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(I, Ir)


@pytest.mark.parametrize("mode", MODES)
def test_cosine_never_writes_the_host_block(mode):
    """torch's CPU device hands the host block's own memory to the backend:
    the in-place token normalisation must work on a copy (fp32 storage)."""
    d = make_dummy.random_multivector(np.random.default_rng(8), 9, 6, 3)
    d = (d[0], d[1] * 3.0)  # not unit: a normalisation would change them
    xb = _block(d)
    before = xb.copy()
    q = make_dummy.random_multivector(np.random.default_rng(9), 3, 6, 2)
    _direct(mode, q, xb, k=4, metric="cosine")
    n, nt, dim, _, o_o, o_t = multivec.header_fields(xb)
    np.testing.assert_array_equal(multivec.unpack(xb).tokens, multivec.unpack(before).tokens)


def test_decode_keeps_null_slots_out_of_the_checks():
    """A null document whose (dropped) span holds a null token vector or a
    token of another width is not refused; the kept tokens decode."""
    vals = pa.array(np.arange(9, dtype=np.float32))
    inner = pa.ListArray.from_arrays(pa.array([0, 2, 5, 7, 9], pa.int32()), vals,
                                     mask=pa.array([False, False, False, False]))
    outer = pa.ListArray.from_arrays(pa.array([0, 1, 2, 4], pa.int32()), inner,
                                     mask=pa.array([False, True, False]))
    r = multivec.unpack(multivec.table_ragged(pa.table({COL: outer}), COL))
    np.testing.assert_array_equal(r.offsets, [0, 1, 1, 3])
    np.testing.assert_array_equal(r.tokens, [[0, 1], [5, 6], [7, 8]])
    inner2 = pa.ListArray.from_arrays(pa.array([0, 2, 4, 6], pa.int32()),
                                      pa.array(np.arange(6, dtype=np.float32)),
                                      mask=pa.array([False, True, False]))
    outer2 = pa.ListArray.from_arrays(pa.array([0, 1, 2, 3], pa.int32()), inner2,
                                      mask=pa.array([False, True, False]))
    r = multivec.unpack(multivec.table_ragged(pa.table({COL: outer2}), COL))
    np.testing.assert_array_equal(r.tokens, [[0, 1], [4, 5]])


def test_maxsim_tiles_bounded_with_many_query_tokens(monkeypatch):
    """Many query tokens against long documents: every token-score tile
    (the array the per-document max reduces) stays within tile_cells unless
    one query x one document is wider; results unchanged."""
    rng = np.random.default_rng(2)
    q = make_dummy.random_multivector(rng, 50, 4, 6)
    d = make_dummy.random_multivector(rng, 6, 4, 40)
    seen = []

    class Spy:  # multivec's numpy, with maximum.reduceat recorded
        maximum = types.SimpleNamespace(
            reduceat=lambda a, i, axis=0: (seen.append(a.size),
                                           np.maximum.reduceat(a, i, axis=axis))[1],
            accumulate=np.maximum.accumulate)

        def __getattr__(self, name):
            return getattr(np, name)
    monkeypatch.setattr(multivec, "np", Spy())
    cells = 64
    S = multivec.maxsim_matrix(q[0], q[1], d[0], d[1], np.float64, tile_cells=cells)
    monkeypatch.undo()
    np.testing.assert_allclose(S, brute_maxsim(q[0], q[1], *d), rtol=1e-12)
    longest = int(np.diff(q[0]).max()) * int(np.diff(d[0]).max())
    assert seen and max(seen) <= max(cells, longest)
    assert sum(np.diff(q[0])) * int(np.diff(d[0]).max()) > max(cells, longest)  # would break


def test_refuses_reduced_precision_matmul(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="exact float32"):
        pylate_engine.MvPylateBackend(K, "scores-batched", device="cpu")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with pytest.raises(RuntimeError, match="float32_matmul_precision"):
            pylate_engine.MvPylateBackend(K, "scores-batched", device="cpu")
    finally:
        torch.set_float32_matmul_precision(old)


def test_forces_pylates_torch_scoring_backend(monkeypatch):
    monkeypatch.setenv("PYLATE_SCORES_BACKEND", "auto")
    pylate_engine.MvPylateBackend(K, "rerank-released", device="cpu")
    import os
    assert os.environ["PYLATE_SCORES_BACKEND"] == "torch"


# ------------------------------------------------------------- refusals

@pytest.mark.parametrize("extra, match", [
    (["--backend", "gpu"], "no multi-vector"),
    (["--backend", "cuvs"], "no multi-vector"),
    (["--backend", "spare", "--spare-mode", "dot"], "sparse only"),
    (["--backend", "pylate"], "needs --pylate-mode"),
    (["--backend", "pylate", "--pylate-mode", "scores-batched", "--pylate-max-mib", "0"],
     ">= 1"),
    (["--backend", "cpu", "--pylate-mode", "scores-batched"], "--backend pylate options"),
    (["--backend", "cpu", "--pylate-rerank-results", "host"], "--backend pylate options"),
    (["--backend", "pylate", "--pylate-mode", "scores-batched", "--pylate-rerank-results",
      "host"], "does not apply to --pylate-mode scores-batched"),
    (["--backend", "pylate", "--pylate-mode", "scores-batched", "--pylate-query-buckets", "0"],
     ">= 1"),
    (["--backend", "pylate", "--pylate-mode", "rerank-released-ownpad",
      "--pylate-query-buckets", "4"], "is for --pylate-mode scores-batched"),
    (["--backend", "cpu", "--pylate-query-buckets", "4"], "--backend pylate options"),
    (["--backend", "pylate", "--pylate-mode", "scores-batched", "--pylate-rerank-results",
      "device"], "does not apply to --pylate-mode scores-batched"),
    (["--backend", "pylate", "--pylate-mode", "rerank-released", "--pylate-rerank-results",
      "gpu"], "invalid choice"),
    (["--backend", "pylate", "--pylate-mode", "scores-batched", "--dim", "8"], "dense only"),
    (["--backend", "pylate", "--pylate-mode", "scores-batched", "--temp-mem", "1"],
     "FAISS option"),
])
def test_cli_refusals(mv_corpus, tmp_path, extra, match, capsys):
    cdir, qp, _, _ = mv_corpus
    with pytest.raises(SystemExit):
        run.parse_args(base_args(cdir, qp, tmp_path / "r.csv", "--regime", "stream", *extra))
    assert match in capsys.readouterr().err


def test_cli_pylate_needs_multivector_and_query_file(mv_corpus, tmp_path, capsys):
    cdir, qp, _, _ = mv_corpus
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", cdir, "--csv", str(tmp_path / "r"),
                        "--backend", "pylate", "--pylate-mode", "scores-batched"])
    assert "--modality multivector only" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", cdir, "--csv", str(tmp_path / "r"),
                        "--modality", "multivector", "--backend", "cpu"])
    assert "needs --query-file" in capsys.readouterr().err
    a = run.parse_args(base_args(cdir, qp, tmp_path / "r.csv", "--regime", "stream",
                                 "--backend", "pylate", "--pylate-mode", "scores-batched"))
    assert a.pylate_max_mib == pylate_engine.PYLATE_DEFAULT_MAX_MIB and a.widen == "device"
    assert a.pylate_rerank_results is None  # scores-batched: not applicable
    for mode in RERANK_MODES:
        a = run.parse_args(base_args(cdir, qp, tmp_path / "r.csv", "--regime", "stream",
                                     "--backend", "pylate", "--pylate-mode", mode))
        assert a.pylate_rerank_results == "device"  # the default


def test_cli_refuses_non_multivector_column(mv_corpus, tmp_path):
    _, qp, _, _ = mv_corpus
    p = make_dummy.write_vectors(str(tmp_path / "d.parquet"), np.ones((3, 4), np.float32),
                                 COL)
    with pytest.raises(SystemExit, match="not a list<list<float>>"):
        run.main(base_args(p, qp, tmp_path / "r.csv", "--regime", "stream",
                           "--backend", "cpu"))


def test_save_results_engine_names(tmp_path):
    assert run.save_results_path(str(tmp_path / "x_{engine}"), "pylate", "multivector") \
        .endswith("x_pylate.npz")
    assert run.save_results_path(str(tmp_path / "x_{engine}"), "cpu", "multivector") \
        .endswith("x_numpy.npz")


# ---------------------------------------------------------------- guards

def _feet(cdir):
    fs, refs = list_parquet(cdir)
    return refs, {r.path: read_footer(fs, r.path, COL) for r in refs}


def _foot(rows, values, bits=32):
    return types.SimpleNamespace(num_rows=rows, mv_values=values, mv_value_bits=bits)


def test_pylate_engine_bytes():
    a = types.SimpleNamespace(pylate_mode="rerank-released", pylate_max_mib=8, k=10)
    assert pylate_engine.pylate_engine_bytes(a, [_foot(5, 100)], 24) == 8 * 2**20 + 24 * 10 * 12
    a.pylate_mode = "scores-batched"
    assert pylate_engine.pylate_engine_bytes(a, [_foot(5, 100)], 3) == 8388608 + 360


@pytest.mark.parametrize("bits, widen", [(32, 0), (16, 1)])
def test_mv_overlap_guard_hand_arithmetic(bits, widen):
    """Two files: 4 documents / 1000 values and 2 / 3000 values. The largest
    packed block is the second: 256 header + 256 (offsets 24 B, aligned) +
    values (3000 x 4 = 12000 -> 12032 aligned; f16: 6000 -> 6144). Need =
    2 blocks + device widen (f16 only: 3000 x 4) + 5 queries x k 10 x 41 B
    + the 1 MiB budget + 5 x 10 x 12 B + the 512 MiB margin."""
    refs = [types.SimpleNamespace(path="a"), types.SimpleNamespace(path="b")]
    footers = {"a": _foot(4, 1000, bits), "b": _foot(2, 3000, bits)}
    block = 256 + 256 + (12032 if bits == 32 else 6144)
    need = 2 * block + widen * 12000 + 5 * 10 * 41 + (1 << 20) + 600 + (512 << 20)
    a = types.SimpleNamespace(widen="device", pylate_mode="scores-batched", pylate_max_mib=1,
                              k=10, backend="pylate")
    src = types.SimpleNamespace(files=refs)
    guards._mv_overlap_guard(a, src, footers, 5, need, need)
    with pytest.raises(SystemExit, match="multi-vector blocks"):
        guards._mv_overlap_guard(a, src, footers, 5, need - 1, need)


def test_mv_verify_need_and_guard(mv_corpus, tmp_path, monkeypatch):
    cdir, qp, _, _ = mv_corpus
    refs, footers = _feet(cdir)
    q, _ = multivec.load_queries(qp, COL)
    a = run.parse_args(base_args(cdir, qp, tmp_path / "r.csv", "--regime", "stream",
                                 "--backend", "cpu", "--verify"))
    t = guards.mv_verify_need(a, refs, footers, q, np.float32)
    assert t["need"] == int((t["shared"] + max(t["in_passes"], t["oracle"]))
                            * guards.VERIFY_SAFETY)
    corpus = sum(multivec.CODEC.block_bytes(f, None) for f in footers.values())
    assert t["oracle"] > corpus and t["in_passes"] > corpus
    monkeypatch.setattr(guards, "mem_available_bytes", lambda: t["need"] - 1)
    with pytest.raises(SystemExit, match="multi-vector run needs"):
        guards.mv_verify_memory_guard(a, refs, footers, q, np.float32)
    monkeypatch.setattr(guards, "mem_available_bytes", lambda: t["need"])
    guards.mv_verify_memory_guard(a, refs, footers, q, np.float32)


