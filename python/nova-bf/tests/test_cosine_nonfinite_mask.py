"""Non-finite corpus/query vectors under a cosine per-query filter.

nova-bf now rejects NaN/inf vectors at read time (`io.check_finite`), so the
tests below assert that error. The history, kept because the guards it
describes are still in the scorer (now unreachable from `run_compute`): a
per-query filter had to keep excluded rows excluded under cosine even when a
corpus or query vector had a non-finite norm.

Cosine divides by the corpus norm (fused cnorm, the default) and by the query
norm in the PACKER, i.e. after the per-query mask set excluded cells to -inf.
-inf / NaN and -inf / inf are NaN, so before the fix an excluded broken row came
back as a NaN candidate for every query: it took a top-k slot (then was dropped
at write time, leaving the query a hit short) and was counted in `nan_hits`.

Oracles here do not depend on NaN sign conventions (which differ CPU vs GPU):
  * a query whose filter EXCLUDES the broken row must get exactly the results
    of a run on the corpus without that row;
  * default (fused) output must equal `NOVA_BF_FUSE_CNORM=0` output;
  * a NaN query's `nan_hits` can never exceed the rows its filter allows.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from nova_bf.compute import run_compute
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
    QueriesConfig, SearchSpec,
)

N, D, NQ, BAD = 120, 16, 40, 57
TENANTS = list("ABCD")
F = Filter(must=[FilterCondition(field="tenant", match_from_query="tenant")])


def _write(tmp_path, tag, *, corpus_bad=None, drop_bad=False, nan_queries=()):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((N, D)).astype(np.float32)
    if corpus_bad == "nan":
        X[BAD] = np.nan
    elif corpus_bad == "inf":
        X[BAD, 0] = np.inf
    tenant = np.array([TENANTS[i % 4] for i in range(N)])
    keep = np.ones(N, bool)
    if drop_bad:
        keep[BAD] = False
    cdir = tmp_path / f"corpus_{tag}"
    cdir.mkdir()
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(N) if keep[i]],
        "dense": pa.array(X[keep].tolist(), pa.list_(pa.float32())),
        "tenant": tenant[keep].tolist(),
    }), cdir / "part0.parquet")
    Q = rng.standard_normal((NQ, D)).astype(np.float32)
    for r in nan_queries:
        Q[r] = np.nan
    (tmp_path / f"queries_{tag}").mkdir()
    qpath = tmp_path / f"queries_{tag}" / "q.parquet"     # same output names in every run
    pq.write_table(pa.table({
        "qid": [f"q{i}" for i in range(NQ)],
        "dense": pa.array(Q.tolist(), pa.list_(pa.float32())),
        "tenant": [TENANTS[i % 4] for i in range(NQ)],
    }), qpath)
    return str(cdir), str(qpath), tenant


def _run(tmp_path, monkeypatch, tag, *, fuse=True, sharded=False, k=12, batch=32, **kw):
    cdir, qpath, tenant = _write(tmp_path, tag, **kw)
    out = tmp_path / f"out_{tag}"
    if fuse:
        monkeypatch.delenv("NOVA_BF_FUSE_CNORM", raising=False)
    else:
        monkeypatch.setenv("NOVA_BF_FUSE_CNORM", "0")
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id", dense_column="dense"),
        queries=QueriesConfig(path=qpath, id_column="qid", dense_column="dense",
                              payload_fields=["tenant"]),
        output=OutputConfig(path=str(out)),
        searches=[SearchSpec(name="cos", vector_type="dense", metric="cosine", k=k, filter=F)],
        params={"dense_batch_size": batch, "io_workers": 1, "threshold_exchange": False},
    )
    run_compute(cfg, **({"num_jobs": 1, "job_rank": 0} if sharded else {}))
    files = {os.path.relpath(p, out): pq.read_table(p)
             for p in glob.glob(f"{out}/**/*.parquet", recursive=True)}
    return files, tenant


def _final(files):
    (name,) = [n for n in files if "partial" not in n and "rank" not in n] or list(files)
    return files[name].to_pydict()


_MSG = "NaN/inf vectors corrupt results; clean or drop these rows."


@pytest.mark.parametrize("bad", ["nan", "inf"])
@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("batch", [32, 8192])     # bad row mid-slice / one slice
def test_broken_corpus_row_is_an_input_error(tmp_path, monkeypatch, bad, fuse, batch):
    """Non-finite vectors are rejected in the reader, before any scoring, so
    the cosine-mask NaN handling this file used to pin is never reached:
    the run fails naming the file, the column and the row."""
    with pytest.raises(RuntimeError) as e:
        _run(tmp_path, monkeypatch, "bad", corpus_bad=bad, fuse=fuse, batch=batch)
    msg = str(e.value)
    assert "part0.parquet" in msg and "column `dense`" in msg
    assert f"1 row(s)" in msg and f"rows {BAD};" in msg and _MSG in msg


def test_nan_queries_are_an_input_error(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as e:
        _run(tmp_path, monkeypatch, "nq", sharded=True, k=40, nan_queries=(5, 6, 13))
    msg = str(e.value)
    assert "queries file" in msg and "q.parquet" in msg and "column `dense`" in msg
    assert "3 row(s)" in msg and "rows 5, 6, 13;" in msg and _MSG in msg


def test_finite_data_unchanged_by_the_guards(tmp_path, monkeypatch):
    """Clean data: fused and unfused already agree; the guards must not move
    anything (the full identity harness covers this at scale)."""
    a, _ = _run(tmp_path, monkeypatch, "f1", fuse=True)
    b, _ = _run(tmp_path, monkeypatch, "f2", fuse=False)
    for n in a:
        assert a[n].equals(b[n]), n
