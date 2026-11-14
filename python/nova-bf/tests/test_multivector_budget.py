"""`params.multivector_budget_mib`: the ONE multivector batching setting.

It is the size, in MiB, of the fp32 score block `(block_query_tokens x
slice_doc_tokens)`. The element budget is `int(mib * 2**20 / 4)`, floored at 1
and capped at the fused reducer's int32 limit; the tile shape (docs per slice,
queries per tile) is derived from it by an even split. These pin the
conversion, the split, validation, the removal of the three settings it
replaced, the manifest record, and that the answer does not depend on it.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf import compute
from nova_bf.compute import (
    _multivector_budget_elements,
    _mv_max_doc_tokens,
    _resolve_multivector_tiles,
    run_compute,
)
from nova_bf.config import ParamsConfig, load_config

import mv_tiles

INT32_MAX = (1 << 31) - 1
DIM = 8


# --------------------------------------------------------------- conversion
@pytest.mark.parametrize("mib,elements", [
    (1, 262_144),                    # 2**20 / 4
    (0.5, 131_072),
    (648.5, 170_000_384),            # the shipped pubmed configs (~170M, measured on A10G)
    (mv_tiles.mib(64), 64),
    (mv_tiles.mib(1), 1),
    (8191.999, int(8191.999 * 2**20 / 4)),
])
def test_mib_converts_to_fp32_elements(mib, elements):
    assert _multivector_budget_elements(mib) == elements


def test_the_default_lands_exactly_on_the_int32_cap():
    """8192 MiB is exactly 2**31 elements, one more than the fused kernel's
    int32 limit; the cap brings it to 2**31 - 1."""
    default = ParamsConfig().multivector_budget_mib
    assert default == 8192
    assert int(default * 2**20 / 4) == 2**31
    assert _multivector_budget_elements(default) == INT32_MAX


@pytest.mark.parametrize("mib", [8192.0001, 16384, 10**9])
def test_larger_budgets_are_capped(mib):
    assert _multivector_budget_elements(mib) == INT32_MAX


def test_a_sub_element_budget_still_makes_progress():
    assert _multivector_budget_elements(1e-12) == 1


# --------------------------------------------------------------- even split
@pytest.mark.parametrize("elements,mean_q,mean_d,want", [
    (10_000, 4.0, 2.0, (50, 25)),        # 100 tokens per axis
    (10_000, 1.0, 1.0, (100, 100)),
    (1, 3.0, 3.0, (1, 1)),               # floored at one item per axis
    (64, 100.0, 0.5, (16, 1)),           # 8 tokens per axis
    (INT32_MAX, 32.0, 180.0, (int(math.sqrt(INT32_MAX) / 180), int(math.sqrt(INT32_MAX) / 32))),
])
def test_the_even_split(elements, mean_q, mean_d, want):
    assert _resolve_multivector_tiles(elements, mean_q, mean_d) == want


def test_the_even_split_spends_about_the_budget_on_each_axis():
    elements, mean_q, mean_d = 4_000_000, 32.0, 120.0
    docs, queries = _resolve_multivector_tiles(elements, mean_q, mean_d)
    axis = math.sqrt(elements)
    assert docs * mean_d <= axis < (docs + 1) * mean_d
    assert queries * mean_q <= axis < (queries + 1) * mean_q
    assert docs * mean_d * queries * mean_q <= elements


def test_doc_token_bound_is_the_budget_when_far_from_int32():
    assert _mv_max_doc_tokens(1000, 10) == 100
    assert _mv_max_doc_tokens(5, 10) == 1        # a single over-budget doc goes alone
    assert _mv_max_doc_tokens(1000, 0) == 1000   # an all-zero-token tile


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, float("nan"), float("inf"),
                                 8192.001, 16384, 1e306, True, False])
def test_non_positive_non_finite_over_cap_or_boolean_budgets_are_rejected(bad):
    """Above 8192 MiB the element count is already at the int32 cap, so a
    larger value could only be silently ignored (and 1e306 used to overflow
    the conversion inside run_compute); a YAML boolean is not a size."""
    with pytest.raises(ValidationError):
        ParamsConfig(multivector_budget_mib=bad)


@pytest.mark.parametrize("ok", [8192, 8192.0, 648.5, 1, 1e-9])
def test_budgets_up_to_the_cap_are_accepted(ok):
    assert ParamsConfig(multivector_budget_mib=ok).multivector_budget_mib == ok


def test_multivector_docs_per_slice_below_k_is_info_not_warning(caplog):
    """The derived docs-per-slice is below a k of 1000 on almost every run and
    has no knob to raise, so it logs INFO; dense / sparse keep the WARNING."""
    from nova_bf.compute import _resolve_vt_batch_size
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        assert _resolve_vt_batch_size(39, 1000, "multivector", device="cpu") == 39
    recs = [r for r in caplog.records if "below k" in r.getMessage()
            or "multivector" in r.getMessage()]
    assert recs and all(r.levelno == logging.INFO for r in recs)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nova_bf.compute"):
        _resolve_vt_batch_size(39, 1000, "dense", device="cpu")
    assert any(r.levelno == logging.WARNING and "below k" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("old", [
    "multivector_batch_size", "multivector_query_block", "multivector_token_budget",
])
def test_the_removed_settings_fail_loudly(old, tmp_path):
    """`ParamsConfig` forbids unknown keys, so a config still carrying one of
    the three replaced settings is a hard error, not a silent no-op."""
    with pytest.raises(ValidationError, match=old):
        ParamsConfig(**{old: 4})
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
corpus: {{path: {tmp_path}, multivector_column: mv}}
queries: {{path: {tmp_path}/q.parquet, multivector_column: mv}}
output: {{path: {tmp_path}/out}}
params: {{{old}: 4}}
searches:
  - {{name: s, k: 1, metric: dot, vector_type: multivector}}
""")
    with pytest.raises(ValidationError, match=old):
        load_config(str(cfg))


# --------------------------------------------------------------- end to end
def _mv_array(docs):
    tok = [len(d) for d in docs]
    total = sum(tok)
    flat = (np.concatenate([d.reshape(-1) for d in docs if len(d)])
            if total else np.empty(0, np.float32))
    inner = pa.ListArray.from_arrays(
        pa.array(np.arange(0, total * DIM + 1, DIM, dtype=np.int32)),
        pa.array(flat.astype(np.float32), pa.float32()),
    )
    outer = np.zeros(len(docs) + 1, dtype=np.int32)
    np.cumsum(tok, out=outer[1:])
    return pa.ListArray.from_arrays(pa.array(outer), inner)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """3 files (40, 37, 23 docs) of ragged 0..7-token docs, 9 ragged queries."""
    rng = np.random.default_rng(42)
    tmp = tmp_path_factory.mktemp("mv_budget")
    cdir = tmp / "corpus"
    cdir.mkdir()
    g = 0
    for fi, n in enumerate((40, 37, 23)):
        docs = [rng.standard_normal((int(rng.integers(0, 8)), DIM)).astype(np.float32)
                for _ in range(n)]
        pq.write_table(pa.table({"mv": _mv_array(docs),
                                 "id": pa.array([str(g + i) for i in range(n)])}),
                       str(cdir / f"c{fi}.parquet"))
        g += n
    queries = [rng.standard_normal((int(rng.integers(1, 6)), DIM)).astype(np.float32)
               for _ in range(9)]
    pq.write_table(pa.table({"mv": _mv_array(queries),
                             "qid": pa.array([str(i) for i in range(9)])}),
                   str(tmp / "q.parquet"))
    return tmp


def _run(tmp, tag, budget_mib, metric):
    out = tmp / f"out_{tag}"
    out.mkdir()
    params = "{io_workers: 2" + (
        "" if budget_mib is None else f", multivector_budget_mib: {budget_mib!r}") + "}"
    p = tmp / f"cfg_{tag}.yaml"
    p.write_text(f"""
corpus: {{path: {tmp / "corpus"}, multivector_column: mv, id_column: id}}
queries: {{path: {tmp / "q.parquet"}, multivector_column: mv, id_column: qid}}
output: {{path: {out}}}
params: {params}
searches:
  - {{name: mv, k: 12, metric: {metric}, vector_type: multivector}}
""")
    t = pq.read_table(run_compute(load_config(str(p)))["mv"]).to_pydict()
    hits = {q: (list(ids), list(sc))
            for q, ids, sc in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
    (manifest,) = out.glob("*_compute.json")
    return hits, json.loads(manifest.read_text())


@pytest.fixture
def tiling_spy(monkeypatch):
    """Record (doc slices, query tiles) per scored batch, so a 'same answer'
    assertion cannot pass because the budget did nothing."""
    seen = {"slices": [], "max_block_tokens": []}
    real = compute._ragged_batch_ranges

    def spy(offsets, max_rows, max_tokens):
        got = real(offsets, max_rows, max_tokens)
        seen["slices"].append(len(got))
        return got

    monkeypatch.setattr(compute, "_ragged_batch_ranges", spy)
    return seen


@pytest.mark.parametrize("metric", ["dot", "cosine"])
def test_results_are_identical_across_budgets(data, metric, tiling_spy):
    """tiny (1 element: one doc per slice, one query per tile), small (64),
    medium (4096), and the 8192 MiB default give the SAME hits and scores."""
    runs = {}
    slices = {}
    for name, mib in (("tiny", mv_tiles.mib(1)), ("small", mv_tiles.mib(64)),
                      ("medium", mv_tiles.mib(4096)), ("default", None)):
        tiling_spy["slices"].clear()
        runs[name] = _run(data, f"{metric}_{name}", mib, metric)
        slices[name] = sum(tiling_spy["slices"])
    base_hits, base_doc = runs["default"]
    # The tilings really differ: the default scores each file as one slice,
    # the tiny budget each document alone.
    assert slices["default"] == 3
    assert slices["tiny"] == 100
    assert slices["default"] < slices["medium"] <= slices["small"] < slices["tiny"]
    assert runs["tiny"][1]["params"]["multivector_derived_queries_per_tile"] == 1
    assert runs["small"][1]["params"]["multivector_derived_queries_per_tile"] < 9
    assert base_doc["params"]["multivector_derived_queries_per_tile"] >= 9
    for name, (hits, _) in runs.items():
        assert hits.keys() == base_hits.keys()
        for q in base_hits:
            assert hits[q][0] == base_hits[q][0], f"{metric} {name} q{q} hit ids differ"
            np.testing.assert_allclose(hits[q][1], base_hits[q][1], rtol=1e-6, atol=1e-6,
                                       err_msg=f"{metric} {name} q{q} scores differ")


def test_the_manifest_records_the_budget_and_the_derived_shape(data):
    _, doc = _run(data, "manifest", mv_tiles.mib(4096), "dot")
    params = doc["params"]
    assert params["multivector_budget_mib"] == mv_tiles.mib(4096)
    assert params["multivector_budget_elements"] == 4096
    # mean tokens per doc of the FIRST file and per query, as the run sees them
    first = pq.read_table(data / "corpus" / "c0.parquet")["mv"].combine_chunks()
    mean_d = sum(len(x) for x in first.to_pylist()) / len(first)
    qs = pq.read_table(data / "q.parquet")["mv"].to_pylist()
    mean_q = sum(len(x) for x in qs) / len(qs)
    want = _resolve_multivector_tiles(4096, mean_q, mean_d)
    assert (params["multivector_derived_docs_per_slice"],
            params["multivector_derived_queries_per_tile"]) == want
    assert params["batch_size_by_vector_type"]["multivector"] == want[0]
    for old in ("multivector_batch_size", "multivector_query_block",
                "multivector_token_budget"):
        assert old not in params


def test_the_default_manifest_records_the_capped_budget(data):
    _, doc = _run(data, "manifest_default", None, "dot")
    assert doc["params"]["multivector_budget_mib"] == 8192
    assert doc["params"]["multivector_budget_elements"] == INT32_MAX
