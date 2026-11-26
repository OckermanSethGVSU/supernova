"""--sparse-project query-vocab (the fairness ablation: the sparse baselines
given nova-bf's own query-vocabulary projection, imported from
nova_bf.compute), CPU only.

Checked: the projected block is exactly what nova-bf's ``_remap_sparse_file``
returns for the same file (sorted, unsorted / duplicate and float64 inputs,
several row groups), the stored / kept nonzero counts against numpy, the
projected queries; run.main end to end for the scipy engine (every reader
kind, the native read path), cuVS (call faked on torch's
CPU device, as tests/test_sparse.py) and every SPARe mode (real, torch CPU):
--verify passes and the scores equal the unprojected run's and a float64
brute force; rows whose entries are all dropped stay eligible as zero-score
results; the CSV / trace columns; the refusals; cuVS refuses an all-empty
projected block; with the flag off nothing of nova-bf is imported and the
results equal the projected run's. The GPU run is owed."""

import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench.systems import harness_cli as run
from bench.systems import loop
from bench.record import results_io
from bench.systems import cpu_reference, cuvs_engine, spare_engine
from bench.data.formats import sparse
from bench.data.sources import InMemorySource, Stages, list_parquet, read_file
from test_sparse import (K, VOCAB, brute, csr_parts, fake_cuvs_pkg,  # noqa: F401
                         fake_knn, read_rows, sparse_corpus)

compute = pytest.importorskip("nova_bf.compute")
COL = "sparse_embedding"


def vocab_of(qp, nq=None):
    q, _ = sparse.load_queries(qp, COL, nq)
    return compute._build_query_vocab(q.indices)


def nova_bf_block(path, vocab):
    """nova-bf's own decode + projection of one file (io.sparse_to_coo_parts,
    then compute._remap_sparse_file with its run-wide LUT)."""
    from nova_bf.io import sparse_to_coo_parts
    t = pq.read_table(path, columns=[COL])
    offs, idx, val = sparse_to_coo_parts(t[COL])
    return compute._remap_sparse_file(offs, idx, val, vocab, compute._build_vocab_lut(vocab))


def _struct_file(path, rows, vtyp=pa.float32(), rg=None):
    """A file of explicit (indices, values) rows (any order, duplicates allowed)."""
    arr = pa.StructArray.from_arrays(
        [pa.array([r[0] for r in rows], pa.list_(pa.uint32())),
         pa.array([r[1] for r in rows], pa.list_(vtyp))], ["indices", "values"])
    pq.write_table(pa.table({COL: arr}), path, row_group_size=rg or len(rows))
    return path


# ------------------------------------------------------------ the block

@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_block_equals_nova_bf_remap_and_counts(sparse_corpus, mode):
    cdir, qp, parts, _ = sparse_corpus
    vocab = vocab_of(qp)
    codec = sparse.ProjectingCodec(vocab)
    fs, refs = list_parquet(cdir)
    for ref, p in zip(refs, parts):
        st = Stages()
        x = read_file(fs, ref.path, COL, stats=st, size=ref.size, mode=mode, codec=codec)
        c = sparse.unpack(x)
        offs, idx, val = nova_bf_block(ref.path, vocab)
        np.testing.assert_array_equal(c.indptr, offs)
        np.testing.assert_array_equal(c.indices, idx)
        np.testing.assert_array_equal(c.values, val)
        assert c.values.dtype == np.float32 and c.n_cols == len(vocab)
        # counts against numpy: stored = the file's, kept = entries in the vocab
        assert sparse.packed_stored_nnz(x) == len(p[1])
        assert c.nnz == sparse.packed_nnz(x) == int(np.isin(p[1], vocab).sum())
        assert sparse.is_projected(x) and c.n_rows == len(p[0]) - 1
        d = st.as_dict()
        assert 0 <= d["project"] <= d["convert"]


def test_unsorted_duplicate_and_f64_rows_match_nova_bf(tmp_path):
    """Rows nova-bf must coalesce (unsorted columns, a duplicate column),
    float64 values (narrowed before the remap, as nova-bf), two row groups,
    a row whose entries are all outside the vocabulary and an empty row."""
    rows = [([5, 1, 9], [1.0, 2.0, 3.0]), ([7, 7, 2], [0.5, 0.25, 4.0]),
            ([100, 200], [1.0, 1.0]), ([], []), ([2, 5, 900], [1.5, -2.0, 3.0])]
    path = _struct_file(str(tmp_path / "a.parquet"), rows, pa.float64(), rg=2)
    vocab = np.array([1, 2, 5, 7, 9], np.uint32)
    fs, refs = list_parquet(path)
    x = read_file(fs, refs[0].path, COL, codec=sparse.ProjectingCodec(vocab))
    c = sparse.unpack(x)
    offs, idx, val = nova_bf_block(path, vocab)
    np.testing.assert_array_equal(c.indptr, offs)
    np.testing.assert_array_equal(c.indices, idx)
    np.testing.assert_array_equal(c.values, val)
    assert c.indptr.tolist() == [0, 3, 5, 5, 5, 7]     # every row kept, two empty
    assert c.indices[3:5].tolist() == [1, 3]           # (2, 7+7) sorted, summed
    assert c.values[3:5].tolist() == [4.0, 0.75]
    assert sparse.packed_stored_nnz(x) == 11 and c.nnz == 7


def test_codec_pickles_without_its_lut_and_rebuilds_it():
    vocab = np.array([3, 8, 40], np.int32)
    codec = sparse.ProjectingCodec(vocab)
    blob = pickle.dumps(codec)
    assert len(blob) < 2000                       # the vocabulary only
    back = pickle.loads(blob)
    assert back.lut is not None and back.lut.vocab is back.vocab
    np.testing.assert_array_equal(back.lut.table, codec.lut.table)


def test_projected_queries():
    q = sparse.Queries(*sparse.canonical(np.array([0, 3, 3, 5]), np.array([40, 3, 8, 3, 3]),
                                         np.array([1, 2, 3, 4, 5], np.float32)), n_cols=41)
    pq_, vocab = sparse.project_queries(q)
    assert vocab.tolist() == [3, 8, 40] and pq_.n_cols == 3
    assert pq_.indptr.tolist() == [0, 3, 3, 4] and pq_.indices.tolist() == [0, 1, 2, 0]
    assert pq_.values.tolist() == [2, 3, 1, 9]     # canonical sums kept
    assert pq_.indices.dtype == np.int32 and pq_.indptr.dtype == np.int32


# ------------------------------------------------------------ run.main

def cli(cdir, qp, out, *extra, backend="cpu"):
    return ["--modality", "sparse", "--backend", backend, "--corpus", cdir, "--column", COL,
            "--query-file", qp, "--query-column", COL, "--k", str(K), "--csv", str(out),
            "--warmup", "0", "--repeats", "1", "--workers", "2", *extra]


def pair(tmp_path, cdir, qp, *extra, backend="cpu", name="r"):
    """The unprojected and the projected run (--verify, --save-results);
    returns their CSV rows and saved results."""
    rows, res = {}, {}
    for proj in ("off", "query-vocab"):
        out = tmp_path / f"{name}_{proj}.csv"
        run.main(cli(cdir, qp, out, "--verify", "--gt-ids-column", "gt_ids",
                     "--sparse-project", proj, "--save-results",
                     str(tmp_path / f"res_{name}_{proj}"), *extra, backend=backend))
        rows[proj] = read_rows(out)[-1]
        res[proj] = np.load(tmp_path / f"res_{name}_{proj}.npz")
    return rows, res


def check_pair(rows, res, parts, vocab, x, Q, exact_ids=True):
    off, on = rows["off"], rows["query-vocab"]
    assert off["verified"] == on["verified"] == "ok"
    assert (off["sparse_project"], on["sparse_project"]) == ("off", "query-vocab")
    stored = sum(len(p[1]) for p in parts)
    kept = sum(int(np.isin(p[1], vocab).sum()) for p in parts)
    assert int(off["nnz"]) == int(off["nnz_kept"]) == int(on["nnz"]) == stored
    assert int(on["nnz_kept"]) == kept < stored
    assert off["project_s"] == "" and float(on["project_s"]) >= 0
    assert float(on["project_s"]) <= float(on["convert_s"])
    assert int(on["d"]) == len(vocab) and on["rows"] == off["rows"]
    assert float(on["gt_recall"]) == float(off["gt_recall"]) == 1.0
    if exact_ids:  # the same products, the same order: bit-identical
        np.testing.assert_array_equal(res["query-vocab"]["D"], res["off"]["D"])
    else:  # SPARe: dropped zero products may regroup a device reduction
        np.testing.assert_allclose(res["query-vocab"]["D"], res["off"]["D"], rtol=1e-6,
                                   atol=1e-7)
    Dr, _ = brute(Q, x, K)
    np.testing.assert_allclose(res["query-vocab"]["D"], Dr, rtol=1e-5, atol=1e-6)
    I = res["query-vocab"]["I"]
    np.testing.assert_allclose(np.take_along_axis(Q @ x.T, I, 1), Dr, rtol=1e-5, atol=1e-6)
    if exact_ids:
        np.testing.assert_array_equal(I, res["off"]["I"])


def corpus_dense(sparse_corpus):
    cdir, qp, parts, x = sparse_corpus
    Q = sparse.load_queries(qp, COL)[0].matrix(VOCAB).toarray()
    return cdir, qp, parts, x, Q, vocab_of(qp)


@pytest.mark.parametrize("regime, extra", [
    ("stream", ["--prefetch", "2"]),
    ("stream", ["--prefetch", "0"]),
])
def test_cli_cpu_projected_equals_unprojected(sparse_corpus, tmp_path, regime, extra):
    cdir, qp, parts, x, Q, vocab = corpus_dense(sparse_corpus)
    rows, res = pair(tmp_path, cdir, qp, "--regime", regime, *extra)
    check_pair(rows, res, parts, vocab, x, Q)
    on = rows["query-vocab"]
    assert float(on["project_s"]) > 0     # measured in the readers (processes too)
    trows = read_rows(results_io.trace_path(str(tmp_path / "r_query-vocab.csv")))
    assert {t["sparse_project"] for t in trows} == {"query-vocab"}


def test_cli_native_read_path(sparse_corpus, tmp_path):
    pytest.importorskip("rust_fetch")
    cdir, qp, parts, x, Q, vocab = corpus_dense(sparse_corpus)
    rows, res = pair(tmp_path, cdir, qp, "--regime", "stream", "--read-mode", "native",
                     "--prefetch", "2")
    check_pair(rows, res, parts, vocab, x, Q)
    assert rows["query-vocab"]["read_mode_effective"] == "native"
    assert int(rows["query-vocab"]["native_fallbacks"]) == 0


def test_cli_cuvs_fake(sparse_corpus, fake_cuvs_pkg, monkeypatch, tmp_path):
    cdir, qp, parts, x, Q, vocab = corpus_dense(sparse_corpus)
    calls = []
    monkeypatch.setattr(run, "make_backend", lambda args, dtype: cuvs_engine.SparseCuvsBackend(
        args.k, device="cpu", events=not args.no_gpu_events, metric=args.metric,
        knn=fake_knn(calls)))
    rows, res = pair(tmp_path, cdir, qp, "--regime", "stream", "--prefetch", "2",
                     backend="cuvs")
    check_pair(rows, res, parts, vocab, x, Q)
    assert calls


@pytest.mark.slow
def test_cli_spare_every_mode(sparse_corpus, monkeypatch, tmp_path):
    pytest.importorskip("spare")
    from test_spare import spare_backend
    cdir, qp, parts, x, Q, vocab = corpus_dense(sparse_corpus)
    for mode in spare_engine.SPARE_MODES:
        monkeypatch.setattr(run, "make_backend", spare_backend(mode))
        rows, res = pair(tmp_path, cdir, qp, "--regime", "stream", "--prefetch", "2",
                         "--spare-mode", mode, backend="spare", name=mode)
        check_pair(rows, res, parts, vocab, x, Q, exact_ids=False)
        assert rows["query-vocab"]["spare_mode"] == mode


def test_fully_dropped_rows_stay_zero_score_results(tmp_path):
    """A corpus where only 2 rows share a column with the query, k = 10:
    the other 8 results are zero-score rows, most of them rows whose every
    entry the projection dropped; the projected run returns exactly the
    unprojected one (lowest-id ties, scipy engine) and --verify passes."""
    cdir = tmp_path / "c"
    cdir.mkdir()
    rows = [([50, 60], [1.0, 2.0])] * 6 + [([1, 70], [2.0, 1.0])] + [([], [])] * 3 + \
        [([80], [3.0])] * 4 + [([2, 90], [0.5, 9.0])]
    _struct_file(str(cdir / "a.parquet"), rows)
    _struct_file(str(cdir / "b.parquet"), [([55], [1.0])] * 5)
    qp = str(tmp_path / "q.parquet")
    qarr = pa.StructArray.from_arrays([pa.array([[1, 2]], pa.list_(pa.uint32())),
                                       pa.array([[1.0, 1.0]], pa.list_(pa.float32()))],
                                      ["indices", "values"])
    pq.write_table(pa.table({COL: qarr}), qp)
    res = {}
    for proj in ("off", "query-vocab"):
        out = tmp_path / f"{proj}.csv"
        run.main(cli(str(cdir), qp, out, "--regime", "stream", "--prefetch", "2",
                     "--verify", "--sparse-project", proj, "--save-results",
                     str(tmp_path / f"res_{proj}")))
        assert read_rows(out)[0]["verified"] == "ok"
        res[proj] = np.load(tmp_path / f"res_{proj}.npz")
    np.testing.assert_array_equal(res["query-vocab"]["I"], res["off"]["I"])
    np.testing.assert_array_equal(res["query-vocab"]["D"], res["off"]["D"])
    assert res["off"]["D"][0, :2].tolist() == [2.0, 0.5]
    assert (res["off"]["D"][0, 2:] == 0).all() and (res["off"]["I"][0] >= 0).all()


def test_cuvs_refuses_an_all_empty_projected_block(tmp_path, fake_cuvs_pkg):
    cdir = tmp_path / "c"
    cdir.mkdir()
    _struct_file(str(cdir / "a.parquet"), [([1, 2], [1.0, 1.0])] * 4)
    _struct_file(str(cdir / "b.parquet"), [([300], [1.0])] * 4)   # no query column
    q = sparse.Queries(np.array([0, 2], np.int32), np.array([1, 2], np.int32),
                       np.ones(2, np.float32), 3)
    qp_, vocab = sparse.project_queries(q)
    b = cuvs_engine.SparseCuvsBackend(K, device="cpu", knn=fake_knn([]))
    b.setup(qp_)
    src = InMemorySource.load(str(cdir), COL, codec=sparse.ProjectingCodec(vocab))
    # the refusal says what happened in the words of the label (+proj)
    with pytest.raises(ValueError, match="empty after query-vocab projection: it has no nonzeros left"):
        loop.run_pass(src, b, 1, K)
    # the cpu engine searches it (zero scores) without complaint
    cb = cpu_reference.SparseCpuBackend(K)
    cb.setup(qp_)
    D, I, _ = loop.run_pass(src, cb, 1, K)
    assert D[0, :4].tolist() == [2.0] * 4 and (D[0, 4:8] == 0).all()


# ------------------------------------------------------------ refusals / off

@pytest.mark.parametrize("extra, match", [
    (["--metric", "cosine"], "--metric ip only"),
    (["--modality", "dense", "--backend", "cpu"], "--modality sparse option"),
])
def test_refusals(sparse_corpus, tmp_path, capsys, extra, match):
    cdir, qp, _, _ = sparse_corpus
    with pytest.raises(SystemExit):
        run.parse_args(cli(cdir, qp, tmp_path / "r.csv", "--regime", "stream",
                           "--sparse-project", "query-vocab", *extra))
    assert match in capsys.readouterr().err


def test_off_imports_nothing_and_matches(sparse_corpus, tmp_path, monkeypatch):
    """Flag off: nova-bf's projection is never imported or run, no block is
    marked projected, the new columns read as without it."""
    cdir, qp, parts, _ = sparse_corpus

    def boom():
        raise AssertionError("nova-bf projection imported with --sparse-project off")
    monkeypatch.setattr(sparse, "nova_bf_projection", boom)
    out = tmp_path / "r.csv"
    run.main(cli(cdir, qp, out, "--regime", "stream", "--prefetch", "2", "--verify"))
    r = read_rows(out)[0]
    assert (r["sparse_project"], r["project_s"], r["verified"]) == ("off", "", "ok")
    assert int(r["nnz"]) == int(r["nnz_kept"]) == sum(len(p[1]) for p in parts)
    src = InMemorySource.load(cdir, COL, codec=sparse.CODEC)
    assert not any(sparse.is_projected(b.vectors) for b in src.blocks)
    assert sparse.packed_stored_nnz(src.blocks[0].vectors) == len(parts[0][1])


def test_dense_row_columns(corpus, tmp_path):
    out = tmp_path / "d.csv"
    run.main(["--backend", "cpu", "--regime", "stream", "--corpus", corpus[0], "--nq", "5",
              "--k", "3", "--csv", str(out), "--warmup", "0", "--repeats", "1"])
    r = read_rows(out)[0]
    assert (r["sparse_project"], r["project_s"], r["nnz"], r["nnz_kept"]) == ("n/a", "", "", "")
