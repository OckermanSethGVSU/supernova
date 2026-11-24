"""End-to-end parity with nova-bf ITSELF: the same corpus files, queries and
nova-bf config are searched by nova-bf (`run_compute`, on CPU) and by the
harness (`harness_cli.py --filter-config ... --filter-search ...`) in every filtered
mode; the top-k ids per query must be the same, and the scores close.

nova-bf runs in a SUBPROCESS: importing nova_bf.compute pulls in torch, whose
~3 GB of mapped address space (on top of DuckDB's) would push this pytest
process past the suite's `ulimit -v` and break the memory-measuring tests that
run after it. Skipped when nova_bf.compute (torch) is not importable."""
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# every test here runs nova-bf / the old compare in a subprocess (seconds each)
pytestmark = pytest.mark.slow
import yaml

from bench.systems import harness_cli as run
from test_duckdb_filter import make_corpus, make_queries

_probe = subprocess.run([sys.executable, "-c", "import nova_bf.compute"], capture_output=True)
if _probe.returncode != 0:
    pytest.skip("nova_bf.compute is not importable: "
                + _probe.stderr.decode(errors="replace")[-300:], allow_module_level=True)


def run_nova_bf(cfg_path):
    """nova-bf's own run (single node, CPU unless NOVA_BF_DEVICE says otherwise)."""
    env = dict(os.environ)
    env.setdefault("NOVA_BF_DEVICE", "cpu")
    code = ("from nova_bf.compute import run_compute; from nova_bf.config import load_config; "
            f"run_compute(load_config({str(cfg_path)!r}))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-2000:]

D = 16
ROWS = [301, 7, 512, 200]
NQ = 30
K = 10
KBIG = 200

STATIC = {"must": [{"field": "i", "range": {"gte": 0}}],
          "should": [{"field": "text", "match_text": "alpha"}, {"field": "s", "match": "a"}]}
PER_QUERY = {"must": [{"field": "d", "range_from_query": {"gte": "qd_lo", "lt": "qd_hi"}}],
             "should": [{"field": "text", "match_text_from_query": "qphrase"},
                        {"field": "s", "match_from_query": "qlist"}]}


def _emb(x):
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def _vectors(kind, rng, n):
    """normal: no ties. ties: entries in {-1, 0, 1}, so many rows score
    exactly the same (as duplicate documents do). The harness merges with
    FAISS's ResultHeap and adds no tie-breaking of its own, so on ties only
    the SCORES must match nova-bf's (which tied ids survive may differ)."""
    if kind == "ties":
        return rng.integers(-1, 2, (n, D)).astype(np.float32)
    return rng.standard_normal((n, D)).astype(np.float32)


@pytest.fixture(scope="module", params=["normal", "ties"])
def setup(tmp_path_factory, request):
    kind = request.param
    root = tmp_path_factory.mktemp(f"nova_parity_{kind}")
    corpus = root / "corpus"
    corpus.mkdir()
    ids = []
    for i, n in enumerate(ROWS):
        p = make_corpus(n, seed=300 + i)
        x = _vectors(kind, np.random.default_rng(10 + i), n)
        fid = [f"f{i}r{r}" for r in range(n)]
        t = p.append_column("embedding", _emb(x)).append_column("id", pa.array(fid))
        pq.write_table(t, str(corpus / f"part-{i:05d}.parquet"))
        ids += fid
    q = make_queries(NQ, seed=21)
    xq = _vectors(kind, np.random.default_rng(77), NQ)
    q = q.append_column("embedding", _emb(xq)).append_column(
        "query_id", pa.array([f"q{i}" for i in range(NQ)]))
    qdir = root / "queries"
    qdir.mkdir()
    qpath = str(qdir / "queries.parquet")
    pq.write_table(q, qpath)
    cfg = {
        "corpus": {"path": str(corpus), "dense_column": "embedding", "id_column": "id",
                   "date_fields": {"d": "rfc3339"}},
        "queries": {"path": qpath, "dense_column": "embedding", "id_column": "query_id",
                    "date_fields": {"qd_lo": "rfc3339", "qd_hi": "rfc3339"}},
        "output": {"path": str(root / "out")},
        "params": {"io_workers": 1, "cpu_thread_count": 1},
        "searches": [
            # one reader thread on nova-bf's side: files in order, ordinal
            # ties exactly as a single pass gives them
            {"name": "static", "vector_type": "dense", "metric": "dot", "k": K, "filter": STATIC},
            {"name": "per_query", "vector_type": "dense", "metric": "dot", "k": K,
             "filter": PER_QUERY},
            # k above most queries' allowed rows: short result lists
            {"name": "per_query_big", "vector_type": "dense", "metric": "dot", "k": KBIG,
             "filter": PER_QUERY},
        ],
    }
    cpath = root / "bf.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    run_nova_bf(cpath)
    nova = {}
    for name, k in (("static", K), ("per_query", K), ("per_query_big", KBIG)):
        t = pq.read_table(str(root / "out" / f"bf_queries_{name}_k{k}.parquet")).to_pydict()
        nova[name] = {qid: (list(h), list(s)) for qid, h, s in
                      zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
    return root, str(corpus), qpath, str(cpath), np.array(ids), nova, kind


def harness(tmp_path, setup, search, backend, mode, extra=(), k=K):
    root, corpus, qpath, cpath, ids, nova, _ = setup
    res = tmp_path / "res.npz"
    run.main(["--regime", "stream", "--corpus", corpus, "--column", "embedding",
              "--backend", backend, "--query-file", qpath, "--query-column", "embedding",
              "--nq", str(NQ), "--k", str(k), "--metric", "ip",
              "--filter-config", cpath, "--filter-search", search, "--filter-mode", mode,
              "--csv", str(tmp_path / "o.csv"), "--save-results", str(res), "--verify", *extra])
    r = np.load(res)
    return r["D"], r["I"]


def test_nova_output_is_non_trivial(setup):
    """The comparison has teeth: hits exist, and per-query results include short
    lists (fewer allowed rows than k) and empty ones."""
    *_, nova, kind = setup
    lens = {s: [len(h) for h, _ in nova[s].values()] for s in nova}
    assert all(n == K for n in lens["static"])
    if kind == "ties":   # ties really straddle the k-th place
        assert any(s[K - 1] == s[K - 2] for _, s in nova["static"].values())
    assert sum(lens["per_query"]) > 0
    big = lens["per_query_big"]
    assert any(0 < n < KBIG for n in big) and any(n == 0 for n in big), big


def assert_same_as_nova(setup, search, D, I):
    """Same scores per query; without ties also the same ids in the same
    order. On the tie corpus the tied ids are the engine's / heap's choice."""
    *_, ids, nova, kind = setup
    for q in range(NQ):
        want_ids, want_scores = nova[search][f"q{q}"]
        got = I[q][I[q] >= 0]
        assert len(got) == len(want_ids), (q, len(got), len(want_ids))
        assert np.allclose(D[q][I[q] >= 0], want_scores, atol=1e-4), q
        if kind != "ties":
            assert list(ids[got]) == list(want_ids), (q, list(ids[got]), want_ids)


@pytest.mark.parametrize("backend,mode", [("masked", "bitmap"), ("masked", "gather"),
                                          ("cpu", "gather")])
def test_static_matches_nova_bf(tmp_path, setup, backend, mode):
    D, I = harness(tmp_path, setup, "static", backend, mode)
    assert_same_as_nova(setup, "static", D, I)


@pytest.mark.parametrize("extra", [[], ["--filter-skip", "files"], ["--filter-skip", "none"],
                                   ["--filter-compact", "union"],
                                   ["--filter-compact", "union", "--filter-skip", "files"],
                                   ["--read-mode", "native"]])
def test_per_query_matches_nova_bf(tmp_path, setup, extra):
    D, I = harness(tmp_path, setup, "per_query", "masked", "bitmap", extra)
    assert_same_as_nova(setup, "per_query", D, I)


@pytest.mark.parametrize("extra", [[], ["--filter-compact", "union"], ["--filter-skip", "none"]])
def test_short_results_match_nova_bf(tmp_path, setup, extra):
    """k above most queries' allowed rows: every short list matches nova-bf's."""
    D, I = harness(tmp_path, setup, "per_query_big", "masked", "bitmap", extra, k=KBIG)
    assert_same_as_nova(setup, "per_query_big", D, I)
