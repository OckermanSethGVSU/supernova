"""bench/verify/exact.py reads corpus files streamed (stream_batches). The scores and lists must
be bit for bit those of reading the whole column (or whole row groups) at once: each test runs
the streamed code and a small copy of the whole-read code it replaced on the same files (one,
several and many row groups; float32 and float16; list and fixed-size list; with and without a
filter; NaN and zero rows), with batches and tiles small enough that they never line up."""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from bench.verify import exact as X
from bench.verify import oracle as O

D_ = 40       # dims
ROWS = 61     # rows per file


# ---------------------------------------------------------------------------------------------
# the whole-read reference (exact.py before streaming)
# ---------------------------------------------------------------------------------------------
def _rows(sc, col):
    if not len(col):
        return np.zeros((0, sc.d), np.float32)
    x = pc.list_flatten(col).to_numpy(zero_copy_only=False).reshape(len(col), -1)
    assert x.shape[1] == sc.d
    return x


def ref_scan_file(sc, path, fs, K, allowed=None):
    col = pq.read_table(path, columns=[sc.vcol], filesystem=fs).column(sc.vcol).combine_chunks()
    x = _rows(sc, col)
    n, nr = len(sc.q), len(x)
    fin = np.isfinite(x).all(axis=1)
    D, P = np.full((n, K), -np.inf), np.full((n, K), -np.inf)
    I, PI = np.full((n, K), -1, np.int64), np.full((n, K), -1, np.int64)
    xn = np.zeros(nr)
    rt = O.oracle_row_tile(sc.d, max(1, nr))
    for r0 in range(0, nr, rt):
        r1 = min(nr, r0 + rt)
        raw = np.array(x[r0:r1], np.float64)
        xn[r0:r1] = np.sqrt(np.einsum("ij,ij->i", raw, raw))
        xt = O.unit_rows_f64(raw) if sc.unit else raw
        for q0 in range(0, n, O.Q_CHUNK):
            q1 = min(n, q0 + O.Q_CHUNK)
            s = sc.q[q0:q1] @ xt.T
            m = fin[None, r0:r1] & np.isfinite(s)
            if allowed is not None:
                m = m & allowed(q0, q1, r0, r1)
            s = np.where(m, s, -np.inf)
            p = s - sc.bound(np.arange(q0, q1)[:, None], xn[None, r0:r1])
            for (A, AI), v in (((D, I), s), ((P, PI), p)):
                dd, ii = O._topk_rows(v, K)
                ii = np.where(ii >= 0, ii + r0, -1)
                A[q0:q1], AI[q0:q1] = O._merge_topk(A[q0:q1], AI[q0:q1], dd, ii, K)
    return D, I, P, PI, xn, nr


def ref_score_pairs(sc, corpus, qi, gids):
    qi, gids = np.asarray(qi, np.int64), np.asarray(gids, np.int64)
    e, xn = np.full(len(gids), np.nan), np.full(len(gids), np.nan)
    f_of = corpus.file_of(gids)
    step = max(1, X.PAIR_CELLS // max(1, sc.d))
    for f in np.unique(f_of):
        sel = np.flatnonzero(f_of == f)
        local = gids[sel] - corpus.offsets[f]
        pf = pq.ParquetFile(corpus.fs.open_input_file(corpus.paths[f]))
        rg_end = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
        rg_of = np.searchsorted(rg_end, local, side="right")
        for g in np.unique(rg_of):
            s2 = sel[rg_of == g]
            r = gids[s2] - corpus.offsets[f] - (rg_end[g - 1] if g else 0)
            col = pf.read_row_group(int(g), columns=[sc.vcol]).column(sc.vcol).combine_chunks()
            ur, inv = np.unique(r, return_inverse=True)
            x = _rows(sc, col.take(pa.array(ur))).astype(np.float64)
            for c0 in range(0, len(s2), step):
                c = slice(c0, c0 + step)
                xx = x[inv[c]]
                nn = np.sqrt(np.einsum("ij,ij->i", xx, xx))
                xs = xx / np.where(nn > 0, nn, 1.0)[:, None] if sc.unit else xx
                e[s2[c]], xn[s2[c]] = np.einsum("ij,ij->i", sc.q[qi[s2[c]]], xs), nn
    return e, xn


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------
def _write(path, x, dtype, kind, row_group_size):
    vt = pa.float16() if dtype == "f16" else pa.float32()
    t = pa.list_(vt, D_) if kind == "fixed" else pa.list_(vt)
    arr = pa.array([r.astype(np.float16 if dtype == "f16" else np.float32).tolist() for r in x], t)
    pq.write_table(pa.table({"v": arr}), path, row_group_size=row_group_size)


def _corpus(tmp_path, dtype, kind, groups, files=3):
    rng = np.random.default_rng(7)
    rgs = {"one": ROWS, "several": 23, "many": 4}[groups]
    paths, xs = [], []
    for f in range(files):
        x = rng.standard_normal((ROWS, D_)).astype(np.float32)
        x[3] = 0.0                      # a zero row (cosine: stays zero)
        x[10, 2] = np.nan               # a non-finite row: never a candidate
        x[20:24] = x[30:34]             # exact ties
        p = str(tmp_path / f"f{f}.parquet")
        _write(p, x, dtype, kind, rgs)
        paths.append(p)
        xs.append(x)
    q = rng.standard_normal((9, D_))
    q[4] = q[5]
    return X.Corpus(pafs.LocalFileSystem(), paths, [ROWS * i for i in range(files + 1)]), q


@pytest.fixture
def small_tiles(monkeypatch):
    """Batches of 5 rows and tiles of 13 (and 4 queries per tile): never aligned."""
    monkeypatch.setattr(X, "STREAM_BATCH_BYTES", 5 * D_ * 2)   # f16: 5 rows, f32: 2 rows
    monkeypatch.setattr(O, "R_TILE", 13)
    monkeypatch.setattr(O, "Q_CHUNK", 4)


def _allowed(seed):
    m = np.random.default_rng(seed).random((9, ROWS)) < 0.6

    def allowed(q0, q1, r0, r1):
        return m[q0:q1, r0:r1]
    return allowed


CASES = [(dt, kind, g) for dt in ("f32", "f16") for kind in ("list", "fixed")
         for g in ("one", "several", "many")]


@pytest.mark.parametrize("dtype,kind,groups", CASES)
@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("filtered", [False, True])
def test_the_streamed_scan_equals_the_whole_column_scan(tmp_path, small_tiles, dtype, kind, groups,
                                                         metric, filtered):
    corpus, q = _corpus(tmp_path, dtype, kind, groups)
    sc = X.Dense(q, metric, "v")
    for f, p in enumerate(corpus.paths):
        al = _allowed(f) if filtered else None
        got = sc.scan_file(p, corpus.fs, 8, al)
        want = ref_scan_file(sc, p, corpus.fs, 8, al)
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)   # bit for bit (NaN norms included)


@pytest.mark.parametrize("dtype,kind,groups", CASES)
@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_the_streamed_pair_scores_equal_the_whole_row_group_scores(tmp_path, small_tiles, dtype,
                                                                    kind, groups, metric):
    corpus, q = _corpus(tmp_path, dtype, kind, groups)
    sc = X.Dense(q, metric, "v")
    rng = np.random.default_rng(3)
    gids = rng.integers(0, corpus.rows, 400)            # duplicates and every file
    gids[:3] = [0, corpus.rows - 1, ROWS + 10]          # the edges and a NaN row
    qi = rng.integers(0, len(q), 400)
    e, xn = X.score_pairs(sc, corpus, qi, gids)
    e0, xn0 = ref_score_pairs(sc, corpus, qi, gids)
    np.testing.assert_array_equal(e, e0)
    np.testing.assert_array_equal(xn, xn0)


def test_the_reference_built_from_streamed_files_equals_the_whole_read(tmp_path, small_tiles,
                                                                        monkeypatch):
    corpus, q = _corpus(tmp_path, "f32", "fixed", "many")
    sc = X.Dense(q, "cosine", "v")
    got = X.build_reference(sc, corpus, 10)
    monkeypatch.setattr(sc, "scan_file", lambda p, fs, K, al=None: ref_scan_file(sc, p, fs, K, al))
    want = X.build_reference(sc, corpus, 10)
    for k in want:
        np.testing.assert_array_equal(got[k], want[k])


def test_a_streamed_read_refuses_another_dim_and_a_row_beyond_the_file(tmp_path):
    corpus, q = _corpus(tmp_path, "f32", "list", "several", files=1)
    with pytest.raises(ValueError, match="corpus dim"):
        X.score_pairs(X.Dense(q[:, :4], "ip", "v"), corpus, [0], [5])
    with pytest.raises(ValueError, match="beyond"):
        X.needed_rows(X.Dense(q, "ip", "v"), corpus.paths[0], corpus.fs, np.array([ROWS]))
    with pytest.raises(ValueError, match="corpus dim"):
        X.Dense(q[:, :4], "ip", "v").scan_file(corpus.paths[0], corpus.fs, 3)


def test_an_empty_file_streams_to_empty_lists(tmp_path):
    p = str(tmp_path / "e.parquet")
    pq.write_table(pa.table({"v": pa.array([], pa.list_(pa.float32(), D_))}), p)
    D, I, P, PI, xn, nr = X.Dense(np.ones((2, D_)), "ip", "v").scan_file(p, pafs.LocalFileSystem(), 3)
    assert nr == 0 and len(xn) == 0 and (I == -1).all() and (PI == -1).all()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("pending_max", [1, 7, 1 << 21])
def test_running_top_k_equals_the_full_selection(seed, pending_max, monkeypatch):
    """RunningTopK (full selection until a list is full, then candidates strictly above the K-th,
    merged in batches) gives exactly the lists of a full selection of every tile, ties to the
    lowest id: integer scores (many exact ties), -inf masks, tiles in row order."""
    from bench.verify import exact as X, oracle as O
    monkeypatch.setattr(X.RunningTopK, "PENDING_MAX", pending_max)
    rng = np.random.default_rng(seed)
    n, K, tile, tiles = 9, 6, 13, 11
    rk = X.RunningTopK(n, K)
    A = np.full((n, K), -np.inf)
    AI = np.full((n, K), -1, np.int64)
    for t in range(tiles):
        v = rng.integers(0, 5, (n, tile)).astype(np.float64)
        v[rng.random((n, tile)) < 0.2] = -np.inf
        r0 = t * tile
        for q0 in range(0, n, 4):                       # query chunks, as the scan
            q1 = min(n, q0 + 4)
            rk.add(q0, q1, v[q0:q1], r0)
            dd, ii = O._topk_rows(v[q0:q1], K)
            ii = np.where(ii >= 0, ii + r0, -1)
            A[q0:q1], AI[q0:q1] = O._merge_topk(A[q0:q1], AI[q0:q1], dd, ii, K)
    rk.flush()
    assert np.array_equal(rk.AI, AI) and np.array_equal(rk.A, A)


def test_running_top_k_flush_never_pads_beyond_k_per_query(monkeypatch):
    """One query whose list fills early with a low K-th gets nearly every later row as a
    candidate; the flush keeps only each query's best K, so its block stays (queries, <= K)."""
    from bench.verify import exact as X, oracle as O
    n, K, tile = 50, 20, 400
    rk = X.RunningTopK(n, K)
    rng = np.random.default_rng(0)
    seen = []
    real = O._merge_topk

    def spy(D1, I1, D2, I2, k):
        seen.append(D2.shape[1])
        return real(D1, I1, D2, I2, k)
    monkeypatch.setattr(O, "_merge_topk", spy)
    A = np.full((n, K), -np.inf)
    AI = np.full((n, K), -1, np.int64)
    for t in range(6):
        v = rng.random((n, tile))
        if t:
            v[0] += 10.0 * t          # query 0: every later row beats its early K-th
        rk.add(0, n, v, t * tile)
        dd, ii = O._topk_rows(v, K)
        A, AI = real(A, AI, dd, np.where(ii >= 0, ii + t * tile, -1), K)
    rk.flush()
    assert np.array_equal(rk.AI, AI) and np.array_equal(rk.A, A)
    assert max(seen) <= max(4 * K, tile)
