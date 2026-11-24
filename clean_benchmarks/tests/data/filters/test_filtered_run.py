"""The harness --filter end to end (CPU): every filter mode / backend combination
must return the exact filtered top-k -- checked against an INDEPENDENT
reference built from nova-bf's own filter masks (nova_bf.filters.evaluate)
and a plain numpy brute force, and by the harness's own --verify."""
import csv

import duckdb

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from bench.systems import harness_cli as run
from bench.record import results_io
from test_duckdb_filter import make_corpus, make_queries, nova_mask

D = 16
ROWS = [301, 7, 512, 1, 200]
NQ = 40
K = 10


def _emb(x):
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("frun")
    corpus = root / "corpus"
    corpus.mkdir()
    payloads, vecs = [], []
    for i, n in enumerate(ROWS):
        p = make_corpus(n, seed=200 + i)
        x = np.random.default_rng(i).standard_normal((n, D)).astype(np.float32)
        pq.write_table(p.append_column("embedding", _emb(x)), str(corpus / f"part-{i:05d}.parquet"))
        payloads.append(p)
        vecs.append(x)
    q = make_queries(NQ, seed=11)
    xq = np.random.default_rng(99).standard_normal((NQ, D)).astype(np.float32)
    q = q.append_column("embedding", _emb(xq)).append_column(
        "query_set", pa.array(["a" if i % 3 else "b" for i in range(NQ)]))
    qpath = str(root / "queries.parquet")
    pq.write_table(q, qpath)
    return root, str(corpus), qpath, payloads, vecs, q, xq


DATES = {"corpus_date_fields": {"d": "rfc3339"},
         "query_date_fields": {"qd_lo": "rfc3339", "qd_hi": "rfc3339"}}
FILTERS = {
    "static": {"must": [{"field": "i", "range": {"gte": 0}}],
               "should": [{"field": "text", "match_text": "alpha"}, {"field": "s", "match": "a"}]},
    "per_query": {"must": [{"field": "d", "range_from_query": {"gte": "qd_lo", "lt": "qd_hi"}}],
                  "should": [{"field": "text", "match_text_from_query": "qphrase"},
                             {"field": "s", "match_from_query": "qlist"}]},
}


def write_filter(root, name, rows=None, flt=...):
    """A nova-bf config with ONE search, named "f" (--filter-config PATH
    --filter-search f): filter FILTERS[name] (or ``flt``), the DATES fields."""
    path = root / f"{name}.yaml"
    search = {"name": "f", "vector_type": "dense",
              "filter": FILTERS[name] if flt is ... else flt}
    if rows:
        search["rows"] = rows
    doc = {"corpus": {"path": "x", "date_fields": DATES["corpus_date_fields"]},
           "queries": {"path": "x", "date_fields": DATES["query_date_fields"]},
           "searches": [search]}
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def reference(name, payloads, vecs, q, xq, sel=None):
    """Independent: nova-bf's masks per file + numpy exact top-k (ip)."""
    if sel is not None:
        q, xq = q.take(pa.array(sel)), xq[sel]
    cd, qd = DATES["corpus_date_fields"], DATES["query_date_fields"]
    allowed, xb = [], np.concatenate(vecs)
    for p in payloads:
        m, per_query = nova_mask(FILTERS[name], p, q, cdates=cd, qdates=qd)
        allowed.append(m if per_query else np.broadcast_to(m, (len(xq), p.num_rows)))
    allowed = np.concatenate(allowed, axis=1)
    s = np.where(allowed, xq @ xb.T, -np.inf)
    I = np.argsort(-s, axis=1, kind="stable")[:, :K]
    ok = np.take_along_axis(s, I, axis=1) > -np.inf
    return np.where(ok, I, -1)


def allowed_pairs(name, payloads, q):
    """Independent: nova-bf's masks, allowed (query, row) pairs over the corpus."""
    cd, qd = DATES["corpus_date_fields"], DATES["query_date_fields"]
    total = 0
    for p in payloads:
        m, per_query = nova_mask(FILTERS[name], p, q, cdates=cd, qdates=qd)
        total += int(m.sum()) if per_query else int(m.sum()) * q.num_rows
    return total


@pytest.mark.slow
@pytest.mark.parametrize("name,mode,extra", [
    ("static", "gather", []), ("static", "bitmap", []), ("static", "bitmap", ["--filter-skip", "none"]),
    ("per_query", "bitmap", []), ("per_query", "bitmap", ["--filter-skip", "none"]),
    ("per_query", "bitmap", ["--filter-skip", "files"]),
    ("per_query", "bitmap", ["--filter-compact", "union"])])
def test_filter_allowed_pairs(tmp_path, data, name, mode, extra):
    """filter_allowed_pairs = the filter's selectivity numerator, whatever is
    skipped or compacted afterwards."""
    root, corpus, qpath, payloads, vecs, q, xq = data
    rows, _ = run_cli(tmp_path, data, write_filter(root, name), "masked", mode, extra)
    want = allowed_pairs(name, payloads, q)
    assert 0 < want < NQ * sum(ROWS)
    assert int(rows[-1]["filter_allowed_pairs"]) == want


def run_cli(tmp_path, data, fpath, backend, mode, extra=()):
    root, corpus, qpath, *_ = data
    out = tmp_path / "out.csv"
    res = tmp_path / "res.npz"
    argv = ["--regime", "stream", "--corpus", corpus, "--column", "embedding",
            "--backend", backend, "--query-file", qpath, "--query-column", "embedding",
            "--nq", str(NQ), "--k", str(K), "--filter-config", fpath, "--filter-search", "f", "--filter-mode", mode,
            "--prefetch", "2", "--workers", "2", "--csv", str(out), "--save-results", str(res),
            "--verify", *extra]
    run.main(argv)
    rows = list(csv.DictReader(open(out)))
    return rows, np.load(res)


def same_topk(I, ref):
    for q in range(len(ref)):
        assert set(I[q][I[q] >= 0]) == set(ref[q][ref[q] >= 0]), q


@pytest.mark.slow
@pytest.mark.parametrize("backend,mode", [("masked", "bitmap"), ("masked", "gather"), ("cpu", "gather")])
def test_static_filter_end_to_end(tmp_path, data, backend, mode):
    root, corpus, qpath, payloads, vecs, q, xq = data
    rows, res = run_cli(tmp_path, data, write_filter(root, "static"), backend, mode)
    same_topk(res["I"], reference("static", payloads, vecs, q, xq))
    r = rows[-1]
    assert r["verified"] == "ok" and r["filter_kind"] == "static" and r["filter_mode"] == mode
    assert int(r["filter_blocks"]) == len(ROWS)
    assert float(r["filter_s"]) > 0
    # DuckFilter's stage breakdown, summed over files, nested inside filter_eval_s
    sub = [float(r[c]) for c in ("filter_prep_s", "filter_sql_s", "filter_text_s",
                                 "filter_combine_s")]
    assert all(v > 0 for v in sub) and sum(sub) <= float(r["filter_eval_s"])
    # each of the two DuckDB stages = time inside DuckDB + the harness's code around it
    for stage in ("filter_sql", "filter_text"):
        q, post = float(r[f"{stage}_query_s"]), float(r[f"{stage}_post_s"])
        assert q > 0 and post >= 0
        assert q + post == pytest.approx(float(r[f"{stage}_s"]), rel=1e-6, abs=1e-9)
    assert r["filter_text_tokens"] in ("lean", "full", "mixed")
    assert r["filter_text_impl"] == "bitmap" and r["duckdb_version"] == duckdb.__version__
    assert float(r["filter_setup_s"]) > 0   # engine start-up, outside the pass
    assert int(r["filter_overflow_bytes"]) == 0 and r["filter_bits_s"] == ""  # not cuVS bitmap


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["native"])
def test_per_query_filter_end_to_end(tmp_path, data, mode):
    root, corpus, qpath, payloads, vecs, q, xq = data
    rows, res = run_cli(tmp_path, data, write_filter(root, "per_query"), "masked", "bitmap",
                        ["--read-mode", mode])
    ref = reference("per_query", payloads, vecs, q, xq)
    same_topk(res["I"], ref)
    r = rows[-1]
    assert r["verified"] == "ok" and r["filter_kind"] == "per_query"
    live = int(r["filter_live_queries"])
    assert 0 < live <= NQ * len(ROWS)
    # file-level dead-query skipping: each file searched only its live queries
    assert live < NQ * len(ROWS)
    if mode == "native":
        assert float(r["payload_read_s"]) > 0


@pytest.mark.slow
def test_filter_config_and_search(tmp_path, data, monkeypatch):
    """--filter-config / --filter-search read a nova-bf config: the search's
    filter and rows, corpus / queries date_fields, ${VAR:-default} expansion."""
    root, corpus, qpath, payloads, vecs, q, xq = data
    cfg = {"corpus": {"path": "${CORPUS_UNUSED:-x}", "date_fields": ["d"]},
           "queries": {"path": "x", "date_fields": ["qd_lo", "qd_hi"]},
           "searches": [{"name": "other", "vector_type": "dense", "k": 1},
                        {"name": "pq", "vector_type": "dense", "k": 10,
                         "filter": FILTERS["per_query"],
                         "rows": {"column": "query_set", "isin": ["a"]}}]}
    path = tmp_path / "bf.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out.csv"
    res = tmp_path / "res.npz"
    run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding",
              "--backend", "masked", "--query-file", qpath, "--query-column", "embedding",
              "--nq", "1000", "--k", str(K), "--filter-config", str(path), "--filter-search", "pq",
              "--filter-mode", "bitmap", "--csv", str(out), "--save-results", str(res), "--verify"])
    sel = np.array([i for i in range(NQ) if i % 3])
    r = list(csv.DictReader(open(out)))[-1]
    assert r["filter"].endswith(":pq") and r["query_rows"] == "query_set=a"
    same_topk(np.load(res)["I"], reference("per_query", payloads, vecs, q, xq, sel))


@pytest.mark.slow
def test_gather_and_bitmap_agree(tmp_path, data):
    root, *_ = data
    f = write_filter(root, "static")
    _, a = run_cli(tmp_path / "a", data, f, "cpu", "gather") if (tmp_path / "a").mkdir() is None else None
    _, b = run_cli(tmp_path / "b", data, f, "masked", "bitmap") if (tmp_path / "b").mkdir() is None else None
    same_topk(a["I"], b["I"])


@pytest.mark.parametrize("argv,msg", [
    (["--backend", "gpu", "--filter-mode", "bitmap"], "FAISS has no prefilter"),
    (["--backend", "cpu", "--filter-mode", "bitmap"], "FAISS has no prefilter"),
    (["--backend", "masked"], "needs --filter-mode"),
    (["--backend", "masked", "--filter-mode", "bitmap", "--h2d-overlap"], "overlap"),
])
def test_validation_errors(tmp_path, data, capsys, argv, msg):
    root, corpus, qpath, *_ = data
    base = ["--regime", "stream", "--corpus", corpus, "--column", "embedding", "--query-file",
            qpath, "--query-column", "embedding", "--filter-config", write_filter(root, "static"), "--filter-search", "f",
            "--csv", str(tmp_path / "o.csv")]
    with pytest.raises(SystemExit):
        run.main(base + argv)
    assert msg in capsys.readouterr().err


def test_per_query_with_gather_or_faiss_refused(tmp_path, data):
    root, corpus, qpath, *_ = data
    f = write_filter(root, "per_query")
    for backend, mode in (("masked", "gather"),):
        with pytest.raises(SystemExit, match="static filters only"):
            run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding",
                      "--backend", backend, "--query-file", qpath, "--query-column", "embedding",
                      "--filter-config", f, "--filter-search", "f", "--filter-mode", mode, "--csv", str(tmp_path / "o.csv")])


def test_unfiltered_run_unchanged(tmp_path, data):
    root, corpus, qpath, payloads, vecs, q, xq = data
    out = tmp_path / "out.csv"
    run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding", "--backend",
              "cpu", "--query-file", qpath, "--query-column", "embedding", "--nq", str(NQ),
              "--k", str(K), "--csv", str(out), "--verify"])
    r = list(csv.DictReader(open(out)))[-1]
    assert r["verified"] == "ok" and r["filter"] == "n/a" and r["filter_s"] == ""


@pytest.mark.slow
@pytest.mark.parametrize("skip", ["queries", "files", "none"])
def test_filter_skip_levels_same_results(tmp_path, data, skip):
    """--filter-skip only changes what is searched, never the results."""
    root, corpus, qpath, payloads, vecs, q, xq = data
    rows, res = run_cli(tmp_path, data, write_filter(root, "per_query"), "masked", "bitmap",
                        ["--filter-skip", skip])
    same_topk(res["I"], reference("per_query", payloads, vecs, q, xq))
    r = rows[-1]
    assert r["verified"] == "ok" and r["filter_skip"] == skip
    searched, live = int(r["filter_searched_queries"]), int(r["filter_live_queries"])
    if skip == "queries":
        assert searched == live
    elif skip == "none":
        assert searched == NQ * len(ROWS) and r["filter_skipped_files"] == "0"
    else:
        assert searched == NQ * (len(ROWS) - int(r["filter_skipped_files"]))


# ---------------------------------------------------------------------------
# the harness wiring
# ---------------------------------------------------------------------------

def test_gt_ids_with_yaml_rows_refused(tmp_path, data):
    root, corpus, qpath, *_ = data
    f = write_filter(root, "per_query", rows={"column": "query_set", "isin": ["b"]})
    with pytest.raises(SystemExit):   # parse_args: "with a query row selection is not supported"
        run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding",
                  "--backend", "masked", "--query-file", qpath, "--query-column", "embedding",
                  "--filter-config", f, "--filter-search", "f", "--filter-mode", "bitmap", "--gt-ids-column", "qi",
                  "--csv", str(tmp_path / "o.csv")])


@pytest.mark.parametrize("doc", [{"filter": {}}, {"filter": None}, {"filter": {"must": []}}])
def test_empty_filter_refused(tmp_path, data, doc):
    root, corpus, qpath, *_ = data
    p = write_filter(tmp_path, "empty", flt=doc["filter"])
    with pytest.raises(SystemExit, match="has no filter"):
        run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding",
                  "--backend", "cpu", "--filter-config", p, "--filter-search", "f", "--filter-mode", "gather",
                  "--csv", str(tmp_path / "o.csv")])


@pytest.mark.slow
def test_counters_match_independent_counts(tmp_path, data):
    """filter_live_queries / searched / rows_kept / skipped against nova-bf masks."""
    root, corpus, qpath, payloads, vecs, q, xq = data
    cd, qd = DATES["corpus_date_fields"], DATES["query_date_fields"]
    for name, mode, skip in (("static", "gather", None), ("static", "bitmap", "none"),
                             ("static", "bitmap", "files"), ("per_query", "bitmap", "queries"),
                             ("per_query", "bitmap", "none")):
        extra = ["--filter-skip", skip] if skip else []
        d = tmp_path / f"{name}-{mode}-{skip}"
        d.mkdir()
        rows, _ = run_cli(d, data, write_filter(root, name), "masked" if mode == "bitmap"
                          else "cpu", mode, extra)
        r = rows[-1]
        masks = [nova_mask(FILTERS[name], p, q, cdates=cd, qdates=qd)[0] for p in payloads]
        if name == "static":
            kept = [int(m.sum()) for m in masks]
            assert int(r["filter_rows_kept"]) == sum(kept) and r["filter_distinct_masks"] == ""
            assert int(r["filter_live_queries"]) == NQ * sum(k > 0 for k in kept)
            skipped = 0 if skip == "none" else sum(k == 0 for k in kept)
            assert int(r["filter_skipped_files"]) == skipped
            assert int(r["filter_searched_queries"]) == NQ * (len(ROWS) - skipped)
        else:
            live = [int(m.any(axis=1).sum()) for m in masks]
            assert int(r["filter_live_queries"]) == sum(live) and r["filter_rows_kept"] == ""
            want = sum(live) if skip == "queries" else NQ * len(ROWS)
            assert int(r["filter_searched_queries"]) == want


def test_filter_flags_need_filter(tmp_path, data):
    root, corpus, qpath, *_ = data
    for extra in (["--filter-skip", "none"], ["--filter-prefilter-max-phrases", "3"]):
        with pytest.raises(SystemExit):
            run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding",
                      "--backend", "cpu", "--csv", str(tmp_path / "o.csv"), *extra])


def test_missing_columns_give_clean_errors(tmp_path, data):
    root, corpus, qpath, *_ = data
    base = ["--regime", "stream", "--corpus", corpus, "--column", "embedding", "--backend",
            "masked", "--query-file", qpath, "--query-column", "embedding", "--csv",
            str(tmp_path / "o.csv"), "--filter-mode", "bitmap"]
    p = write_filter(tmp_path, "nosuch", flt={"must": [{"field": "nosuch", "match": 1}]})
    with pytest.raises(SystemExit, match="no column"):
        run.main(base + ["--filter-config", p, "--filter-search", "f"])
    with pytest.raises(SystemExit, match="filter config"):
        run.main(base + ["--filter-config", str(tmp_path / "missing.yaml"), "--filter-search", "f"])


@pytest.mark.parametrize("extra", [
    ["--backend", "gpu", "--filter-mode", "gather", "--h2d", "pinned"],
    ["--backend", "gpu", "--filter-mode", "gather", "--h2d", "pinned", "--h2d-overlap"],
    ["--backend", "gpu", "--filter-mode", "gather", "--h2d", "pinned", "--h2d-overlap",
     "--d2h", "pinned", "--d2h-overlap"],
    ["--backend", "cuvs", "--filter-mode", "bitmap", "--filter-compact", "union", "--h2d",
     "pinned", "--h2d-overlap"],
])
def test_compacting_modes_accept_pinned_and_overlap(tmp_path, data, extra):
    """gather / union blocks are compacted by the READER in their ring slot,
    so --h2d pinned / --h2d-overlap / --d2h-overlap are accepted for them;
    only the generic transfer checks still apply (reader processes)."""
    root, corpus, *_ = data
    base = ["--regime", "stream", "--corpus", corpus, "--filter-config", write_filter(root, "static"), "--filter-search", "f",
            "--csv", str(tmp_path / "o.csv")]
    a = run.parse_args(base + extra)
    assert run.h2d_mode(a) in ("pinned", "pinned-overlap")
    spec = run.filter_compact_spec(a)
    assert spec.mode == ("gather" if "gather" in extra else "union")
    with pytest.raises(SystemExit):   # the inline reader hands over pageable arrays
        run.parse_args(base + extra + ["--prefetch", "0"])


def test_filter_compact_spec():
    a = run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "x.csv", "--filter-config",
                        "f.yaml", "--filter-search", "f", "--filter-mode", "bitmap", "--backend", "masked"])
    assert run.filter_compact_spec(a) is None
    a = run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "x.csv", "--filter-config",
                        "f.yaml", "--filter-search", "f", "--filter-mode", "bitmap", "--backend", "masked",
                        "--filter-compact", "union", "--filter-skip", "files"])
    assert (run.filter_compact_spec(a).mode, run.filter_compact_spec(a).skip) == ("union", "files")


def test_rows_selector_matches_nova_bf_string_compare():
    from bench.data.filters import search as fsm
    # nova-bf: np.isin(np.array(values).astype(str), isin) -- with a null the
    # array is object dtype, so ints stay "1" and the null reads "None"
    t = pa.table({"c": pa.array([1, None, 2, 1])})
    assert fsm.query_row_index(t, {"column": "c", "isin": ["1"]}).tolist() == [0, 3]
    assert fsm.query_row_index(t, {"column": "c", "isin": ["1.0"]}).tolist() == []
    assert fsm.query_row_index(t, {"column": "c", "isin": ["None"]}).tolist() == [1]
    # a float column reads "1.0"
    t = pa.table({"c": pa.array([1.0, 2.0])})
    assert fsm.query_row_index(t, {"column": "c", "isin": ["1.0"]}).tolist() == [0]
    for vals in ([1, None, 2, 1], [1.0, 2.0], ["a", None], [True, False]):
        want = np.nonzero(np.isin(np.array(vals).astype(str), np.asarray(["1", "1.0", "a",
                          "True", "None"], dtype=str)))[0].tolist()
        got = fsm.query_row_index(pa.table({"c": pa.array(vals)}),
                                  {"column": "c", "isin": ["1", "1.0", "a", "True", "None"]})
        assert got.tolist() == want, vals


@pytest.mark.parametrize("kinds", ["per_query", "static", "mixed"])
def test_oracle_disallowed_vectorised_matches_slow(kinds):
    from bench.verify import oracle as O
    rng = np.random.default_rng(0)
    from bench.data.filters.reader import FilterBlock
    from bench.data.filters import duckdb_filter as dfm
    fbs, starts, s0 = [], [], 0
    for i, n in enumerate((37, 1, 512, 9, 263)):
        static = kinds == "static" or (kinds == "mixed" and i % 2 == 1)
        masks = np.packbits(rng.random((1 if static else 5, n)) < 0.3, axis=1,
                            bitorder="little")
        fbs.append(FilterBlock("static", n, masks, masks.sum(axis=1)) if static else
                   FilterBlock("per_query", n, masks, masks.sum(axis=1),
                               rng.integers(0, 5, 20).astype(np.int32)))
        starts.append(s0)
        s0 += n
    a = O._Allowed(starts, fbs)
    I = rng.integers(-1, s0, (20, 15))
    got = set(a.disallowed(I))
    slow = set()
    for q in range(20):
        for g in I[q]:
            if g < 0:
                continue
            f = max(i for i, st in enumerate(starts) if st <= g)
            r = g - starts[f]
            m = 0 if fbs[f].kind == "static" else fbs[f].mask_of_query[q]
            if not dfm.unpack(fbs[f].masks[m], fbs[f].n_rows)[r]:
                slow.add((q, int(g)))
    assert slow and got == slow


def test_oracle_disallowed_memory_is_per_pair():
    """bits() reads one byte per (query, id) pair: a 1M-row per-query file with
    many returned ids must not gather whole mask rows (the 11-11 127 GiB OOM)."""
    from bench.verify import oracle as O
    from bench.data.filters.reader import FilterBlock
    import tracemalloc
    rng = np.random.default_rng(1)
    n, nq, k = 1_000_000, 200, 1000
    masks = np.packbits(rng.random((4, n)) < 0.5, axis=1, bitorder="little")
    fb = FilterBlock("per_query", n, masks, masks.sum(axis=1),
                     rng.integers(0, 4, nq).astype(np.int32))
    a = O._Allowed([0], [fb])
    I = rng.integers(0, n, (nq, k))
    tracemalloc.start()
    a.disallowed(I)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # a full nq x n bitmap per pair would be nq*k*n/8 = 25 GB; per-pair indexing is a few arrays of nq*k
    assert peak < 200 * nq * k, peak


def test_masked_topk_tiling_matches_one_tile():
    from bench.data.filters import search as fsm
    rng = np.random.default_rng(1)
    xq = rng.standard_normal((30, 8)).astype(np.float32)
    xb = rng.integers(-2, 3, (700, 8)).astype(np.float32)   # many exact ties
    m = rng.random((30, 700)) < 0.4
    full = fsm.masked_topk(xq, xb, 25, lambda q0, q1, r0, r1: m[q0:q1, r0:r1], 1000, 10 ** 9)
    tiled = fsm.masked_topk(xq, xb, 25, lambda q0, q1, r0, r1: m[q0:q1, r0:r1], 7, 64)
    # same picks; scores equal up to BLAS rounding (a GEMM's float order depends on its
    # shape, so tiles may differ from one big product in the last bit on some BLAS builds)
    assert np.array_equal(full[1], tiled[1])
    assert np.allclose(full[0], tiled[0], rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# --filter-compact union
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("skip", ["queries", "files", "none"])
def test_compact_union_same_results(tmp_path, data, skip):
    """Union compaction only changes what is searched, never the results; the
    rows searched per file are exactly the union of nova-bf's masks over the
    searched queries."""
    root, corpus, qpath, payloads, vecs, q, xq = data
    from bench.data.filters.reader import COMPACT_MAX_FRAC as frac
    rows, res = run_cli(tmp_path, data, write_filter(root, "per_query"), "masked", "bitmap",
                        ["--filter-skip", skip, "--filter-compact", "union"])
    same_topk(res["I"], reference("per_query", payloads, vecs, q, xq))
    r = rows[-1]
    assert r["verified"] == "ok" and r["filter_compact"] == "union"
    cd, qd = DATES["corpus_date_fields"], DATES["query_date_fields"]
    from bench.data.filters import search as fsm
    union = floor_files = floor_rows = whole = 0
    for p in payloads:
        m = nova_mask(FILTERS["per_query"], p, q, cdates=cd, qdates=qd)[0]
        if skip != "none" and not m.any():
            continue                       # skipped file: nothing searched
        wanted = int(m.any(axis=0).sum())
        if wanted > frac * p.num_rows:
            # COMPACT_MAX_FRAC: the union keeps most of it, left whole
            whole += 1
            union += p.num_rows
            continue
        # the COMPACT_MIN_ROWS floor: a file is never compacted below 257 rows
        # (or below its own size) unless nothing is wanted
        n = max(wanted, min(fsm.COMPACT_MIN_ROWS, p.num_rows)) if wanted else 0
        floor_files += n > wanted
        floor_rows += n - wanted
        whole += n == p.num_rows     # every row searched (all wanted, or the floor)
        union += n
    assert int(r["filter_compact_rows"]) == union
    assert (int(r["filter_compact_floor_files"]), int(r["filter_compact_floor_rows"])) == \
        (floor_files, floor_rows)
    assert int(r["filter_compact_whole_files"]) == whole
    assert float(r["filter_compact_max_frac"]) == frac


def test_compact_union_refusals(tmp_path, data):
    root, corpus, qpath, *_ = data
    base = ["--regime", "stream", "--corpus", corpus, "--column", "embedding", "--query-file",
            qpath, "--query-column", "embedding", "--csv", str(tmp_path / "o.csv"),
            "--filter-compact", "union"]
    with pytest.raises(SystemExit, match="per-query filters"):
        run.main(base + ["--backend", "masked", "--filter-config", write_filter(root, "static"), "--filter-search", "f",
                         "--filter-mode", "bitmap"])
    with pytest.raises(SystemExit):
        run.main(base + ["--backend", "cpu", "--filter-config", write_filter(root, "static"), "--filter-search", "f",
                         "--filter-mode", "gather"])


def test_compact_union_repacks_exactly():
    from bench.data.filters import duckdb_filter as dfm
    from bench.data.filters import search as fsm
    from bench.data.filters.reader import FilterBlock
    rng = np.random.default_rng(3)
    for n in (1, 7, 8, 9, 300, 1031):
        m = rng.random((70, n)) < 0.05          # > MASK_CHUNK masks, sparse
        m[3] = False                             # an empty mask
        masks = np.packbits(m, axis=1, bitorder="little")
        moq = rng.integers(0, 70, 40).astype(np.int32)
        fb = FilterBlock("per_query", n, masks, m.sum(axis=1), moq)
        for queries in (None, np.array(sorted(set(rng.integers(0, 40, 9).tolist())))):
            keep, fc, _ = fsm._compact_union(fb, queries, min_rows=0)
            used = np.unique(moq if queries is None else moq[queries])
            assert keep.tolist() == np.flatnonzero(m[used].any(axis=0)).tolist()
            for j in used:
                assert np.array_equal(dfm.unpack(fc.masks[j], fc.n_rows), m[j][keep])
            if fc.n_rows % 8:
                assert not (fc.masks[:, -1] & (~dfm.tail_mask(fc.n_rows) & 0xFF)).any()


@pytest.mark.slow
def test_trace_carries_filter_segments_and_plots(tmp_path, data):
    root, corpus, qpath, *_ = data
    rows, _ = run_cli(tmp_path, data, write_filter(root, "per_query"), "masked", "bitmap",
                      ["--read-mode", "native"])
    trace = list(csv.DictReader(open(results_io.trace_path(str(tmp_path / "out.csv")))))
    r = [t for t in trace if t["warmup"] == "0"][0]
    assert float(r["w_filter_start_s"]) <= float(r["w_filter_end_s"])
    assert float(r["w_end_s"]) <= float(r["w_filter_start_s"])
    assert float(r["payload_read_s"]) > 0



@pytest.mark.slow
def test_mask_slots_auto_never_overflow(tmp_path, data):
    """Default (--filter-mask-slots auto): room for every possible mask, so no
    file's masks travel through the pipe."""
    root, corpus, qpath, payloads, vecs, q, xq = data
    rows, res = run_cli(tmp_path, data, write_filter(root, "per_query"), "masked", "bitmap")
    same_topk(res["I"], reference("per_query", payloads, vecs, q, xq))
    r = rows[-1]
    assert r["verified"] == "ok" and r["filter_mask_slots"] == "auto"
    assert int(r["filter_mask_overflows"]) == 0


def test_mask_slots_auto_refuses_a_ring_that_does_not_fit(tmp_path, data, monkeypatch):
    from bench.data.filters import reader as filtered
    root, *_ = data
    monkeypatch.setattr(filtered, "mem_available", lambda: 1024)   # 1 KiB available
    with pytest.raises((RuntimeError, SystemExit), match="use fewer readers"):
        run_cli(tmp_path, data, write_filter(root, "per_query"), "masked", "bitmap")


def test_masked_topk_ties_keep_lowest_ids():
    """argpartition alone keeps an arbitrary subset of rows tied at the k-th
    place; the masked engine must keep the lowest ids (nova-bf ordinal)."""
    from bench.data.filters import search as fs
    rng = np.random.default_rng(5)
    xb = rng.integers(-1, 2, (700, 6)).astype(np.float32)
    xq = rng.integers(-1, 2, (9, 6)).astype(np.float32)
    for r_tile in (700, 64):
        D, I = fs.masked_topk(xq, xb, 10, r_tile=r_tile)
        s = xq @ xb.T
        for q in range(len(xq)):
            want = np.lexsort((np.arange(len(xb)), -s[q]))[:10]
            assert list(I[q]) == list(want), (r_tile, q)
            assert np.array_equal(D[q], s[q][want])
    D, I = fs.masked_topk(np.ones((1, 6), np.float32), np.ones((300, 6), np.float32), 10)
    assert list(I[0]) == list(range(10))


def test_topk_rows_nan_rows_do_not_crash():
    """Fewer non-NaN scores than k: NaN ranks as not allowed (-inf, -1)."""
    from bench.verify import oracle as O
    s = np.array([[np.nan] * 6 + [1, 2, 3, 0], [np.nan] * 10], np.float32)
    D, I = O._topk_rows(s, 5)
    assert list(I[0]) == [8, 7, 6, 9, -1] and list(I[1]) == [-1] * 5
    assert np.isneginf(D[0, 4]) and np.isneginf(D[1]).all()


def test_compact_union_floor():
    """--filter-compact union never shrinks a file below COMPACT_MIN_ROWS rows
    (topped up with rows no query wants, whose bits stay 0), keeps a smaller
    file whole, and keeps an all-unwanted file empty (skipped)."""
    from bench.data.filters import search as fsm
    from bench.data.filters.reader import FilterBlock
    from bench.data.filters import duckdb_filter as dfm
    rng = np.random.default_rng(3)
    for n_rows, n_want in ((1000, 5), (1000, 300), (100, 5), (1000, 0), (257, 3)):
        wanted = np.zeros(n_rows, bool)
        wanted[rng.choice(n_rows, n_want, replace=False)] = True
        m = np.stack([wanted & (rng.random(n_rows) < 0.7), wanted])
        masks = np.packbits(m, axis=1, bitorder="little")
        fb = FilterBlock("per_query", n_rows, masks, m.sum(1), np.array([0, 1, 1], np.int32))
        keep, fc, added = fsm._compact_union(fb)
        want_n = 0 if n_want == 0 else max(n_want, min(fsm.COMPACT_MIN_ROWS, n_rows))
        assert len(keep) == want_n and added == want_n - n_want, (n_rows, n_want)
        assert np.all(np.diff(keep) > 0) and set(np.flatnonzero(wanted)) <= set(keep)
        if want_n:
            bits = dfm.unpack(fc.masks, len(keep))
            assert np.array_equal(bits, m[:, keep])        # added rows: 0 for every mask
            assert np.array_equal(fc.counts, m.sum(1))     # popcounts unchanged


@pytest.mark.slow
@pytest.mark.parametrize("name,mode,extra", [("static", "gather", []),
                                             ("per_query", "bitmap", ["--filter-compact", "union"])])
def test_verify_oracle_reload_never_compacts(tmp_path, data, monkeypatch, name, mode, extra):
    """The passes' blocks are compacted by the readers (gather / union); the
    --verify oracle's own reload must see WHOLE blocks and the file's masks."""
    from bench.verify import oracle as O
    root, corpus, qpath, payloads, vecs, q, xq = data
    seen = []
    real = O.filtered_oneshot_reference

    def spy(xq_, blocks, *a, **kw):
        def watch():
            for b in blocks:
                seen.append((b.compact, len(b.vectors), b.filter.n_rows))
                yield b
        return real(xq_, watch(), *a, **kw)

    monkeypatch.setattr(O, "filtered_oneshot_reference", spy)
    rows, _ = run_cli(tmp_path, data, write_filter(root, name), "masked", mode, extra)
    assert rows[-1]["verified"] == "ok" and float(rows[-1]["filter_compact_s"]) > 0
    assert [s[0] for s in seen] == [None] * len(ROWS)
    assert [s[1] for s in seen] == [s[2] for s in seen] == ROWS


def test_compact_union_whole_file_and_fast_repack():
    """--filter-compact-max-frac: a file whose union keeps more than the
    fraction is left whole (keep = every row, masks untouched); below it,
    the compress-based re-pack equals gathering the bits at ``keep``."""
    from bench.data.filters import duckdb_filter as dfm
    from bench.data.filters import search as fsm
    from bench.data.filters.reader import FilterBlock
    rng = np.random.default_rng(11)
    for n in (9, 300, 1031):
        for density in (0.002, 0.3):
            m = rng.random((70, n)) < density
            masks = np.packbits(m, axis=1, bitorder="little")
            moq = rng.integers(0, 70, 40).astype(np.int32)
            fb = FilterBlock("per_query", n, masks, m.sum(axis=1), moq)
            frac_kept = m[np.unique(moq)].any(axis=0).mean()
            for max_frac in (0.5, 1.0):
                keep, fc, added = fsm._compact_union(fb, None, max_frac=max_frac)
                if frac_kept > max_frac:
                    assert keep.tolist() == list(range(n)) and fc is fb and added == 0
                    continue
                for j in np.unique(moq):
                    assert np.array_equal(dfm.unpack(fc.masks[j], fc.n_rows), m[j][keep])
