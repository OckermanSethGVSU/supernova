"""--modality sparse --backend cuvs on a REAL GPU (skipped unless torch sees
CUDA, the cuvs package imports and baselines/cuvs/sparse_binding's shim is
built): the harness end to end with --verify against the float64 oracle, both
regimes, both metrics, events on and off, fp16-stored values (widened on the
device), files larger than cuVS's stock 32768-row index batch."""

import csv
import os

import numpy as np
import pytest

import make_dummy
from bench.systems import harness_cli as run
from bench.systems import cuvs_engine

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("cuvs")
try:
    cuvs_engine.cuvs_sparse_binding().load()
except RuntimeError as e:
    pytest.skip(f"cuvs sparse shim not built: {e}", allow_module_level=True)

K = 10


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    out = tmp_path_factory.mktemp("spg")
    made = {}
    for name, rows, vdt in (("f32", [5000, 3, 40000, 1200], np.float32),
                            ("f16", [3000, 2500], np.float16)):
        cdir = out / name
        cdir.mkdir()
        parts = []
        for i, n in enumerate(rows):
            p = make_dummy.random_sparse(np.random.default_rng([11, i]), n, 4000, 30, 0.01)
            make_dummy.write_sparse(str(cdir / f"part-{i:05d}.parquet"), p, value_dtype=vdt,
                                    row_group_size=1500)
            parts.append(p)
        for metric in ("ip", "cosine"):
            make_dummy.make_sparse_queries(str(out / f"q_{name}_{metric}.parquet"), 64,
                                           4000, 8, corpus_parts=parts, gt_k=K,
                                           gt_metric=metric, value_dtype=vdt)
        made[name] = (str(cdir), str(out))
    return made


@pytest.mark.parametrize("name", ["f32", "f16"])
@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("regime", ["stream", "memory"])
@pytest.mark.parametrize("events", [True, False])
def test_cuvs_sparse_end_to_end(corpora, tmp_path, name, metric, regime, events):
    cdir, qdir = corpora[name]
    out = tmp_path / "r.csv"
    run.main(["--modality", "sparse", "--backend", "cuvs", "--regime", regime,
              "--corpus", cdir, "--column", "sparse_embedding",
              "--query-file", os.path.join(qdir, f"q_{name}_{metric}.parquet"),
              "--query-column", "sparse_embedding", "--k", str(K), "--metric", metric,
              "--csv", str(out), "--warmup", "1", "--repeats", "1", "--verify",
              "--gt-ids-column", "gt_ids"]
             + ([] if events else ["--no-gpu-events"]))
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["verified"] for r in rows} == {"ok"}
    r = rows[-1]
    assert r["engine"] == "cuvs" and r["modality"] == "sparse"
    assert float(r["gt_recall"]) >= 0.99  # ties at the k-th place may differ
    if events:
        assert float(r["search_s"]) > 0 and float(r["h2d_s"]) > 0
        assert (float(r["widen_s"]) > 0) == (name == "f16")
    else:
        assert r["search_s"] == ""
