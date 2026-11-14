"""Query ids that are NOT the query's row index.

Every other fixture in the harness names its queries `str(index)` — "0",
"1", … — so an output that wrote the query's POSITION in place of its id is
indistinguishable from a correct one: `runner.read_results` read `int(qid)`
back as the index and the rows lined up perfectly. Production query files are
keyed by arbitrary strings, so that bug would hand every query another query's
ground truth.

`ds_qids` is the shared corpus with only the query ids changed, to
`q1000, q999, …` — not integers, not in index order, and not in lexicographic
index order either ("q1000" < "q993" < "q994"), so writing positions, sorting
the ids, or re-deriving them all fail. Results are mapped back through the
ids (`runner.qid_index`), never through their position, and an id nova-bf
wrote that names no query is a hard failure.

Run through it: a spread of the matrix (every modality; no / static /
per-query filters), a sharded compute + merge, and a `rows` query subset —
each against the naive oracle, and the matrix spread against Qdrant.
"""

from __future__ import annotations

import pytest

from . import cases as cases_mod
from . import compare, corpus, naive, qdrant_ref
from .runner import (MV_23_DOCS_MIB, build_config, pinned_device, qid_index,
                     read_results, run, spec)
from .test_parity_matrix import _filter_from_dict


def _qid(i: int) -> str:
    return f"q{1000 - i}"


@pytest.fixture(scope="module")
def ds_qids(tmp_path_factory):
    return corpus.build(tmp_path_factory.mktemp("bf_parity_qids"), qid=_qid)


@pytest.fixture(scope="module")
def oracle_qids(ds_qids):
    return naive.Oracle(ds_qids.docs, ds_qids.queries, ds_qids.date_fields,
                        ds_qids.query_date_fields)


FILTER_NAMES = ("nofilter", "match", "rangefloat", "pqmatch", "pqtext", "pqmixed")
CASES = [c for c in cases_mod.CASES if c.filter_name in FILTER_NAMES]
IDS = [c.id for c in CASES]


@pytest.fixture(scope="module")
def qids_run(ds_qids, device):
    return run(ds_qids, [c.spec() for c in CASES], out_tag="qids", device=device,
               params={"dense_batch_size": 37, "sparse_batch_size": 29,
                       "multivector_budget_mib": MV_23_DOCS_MIB})


def test_the_fixture_ids_are_not_positions(ds, ds_qids):
    """Guards the fixture: ids are not the index, not in index order, and
    sorting them does not recover index order — and nothing ELSE about the
    queries changed, so the shared Qdrant collection and every expectation
    about `ds` carry over."""
    ids = [q["qid"] for q in ds_qids.queries]
    assert all(q != str(i) for i, q in enumerate(ids))
    assert sorted(ids) != ids and sorted(ids, reverse=True) != ids
    assert len(set(ids)) == len(ids)
    assert len(ds_qids.queries) == len(ds.queries)
    for a, b in zip(ds_qids.queries, ds.queries):
        assert {k: v for k, v in a.items() if k != "qid"} == \
               {k: v for k, v in b.items() if k != "qid"}
    assert [d.dense for d in ds_qids.docs] == [d.dense for d in ds.docs]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_results_are_keyed_by_query_id(case, ds_qids, oracle_qids, qids_run, device):
    got = qids_run[case.name]
    assert sorted(got) == list(range(len(ds_qids.queries))), (
        f"{case.id}: queries answered {sorted(got)}")
    want = oracle_qids.topk(vector_type=case.vector_type, metric=case.metric,
                            k=case.k, filt=_filter_from_dict(ds_qids, case.filter_dict))
    for qi in range(len(ds_qids.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] {case.id} {_qid(qi)}: nova-bf vs naive")


@pytest.mark.qdrant
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_results_keyed_by_query_id_match_qdrant(case, ds_qids, qids_run, client,
                                                collection, device):
    """`collection` holds the shared corpus's points, which `ds_qids` shares
    byte for byte (see the fixture guard)."""
    got = qids_run[case.name]
    want = qdrant_ref.topk(client, collection, ds_qids, vector_type=case.vector_type,
                           metric=case.metric, k=case.k,
                           filt=_filter_from_dict(ds_qids, case.filter_dict))
    for qi in range(len(ds_qids.queries)):
        compare.assert_scores_agree(
            got[qi], want[qi], metric=case.metric,
            label=f"[{device}] {case.id} {_qid(qi)}: nova-bf vs qdrant")


SHARD_CASES = [cases_mod.CASES_BY_NAME[n] for n in
               ("decos_nofilter", "deeuc_pqmatch", "spdot_match", "mudot_pqtext")]


@pytest.mark.parametrize("num_jobs", [3, 5])
def test_a_sharded_merge_is_keyed_by_query_id(num_jobs, ds_qids, oracle_qids, device):
    """Partials and the merge both carry the query id: a merge that re-derived
    it from row position would only show up here."""
    cfg = build_config(ds_qids, [c.spec() for c in SHARD_CASES],
                       out_tag=f"qids_shard{num_jobs}_{device}")
    from nova_bf.compute import run_compute
    from nova_bf.merge import run_merge

    with pinned_device(device):
        for rank in range(num_jobs):
            run_compute(cfg, num_jobs=num_jobs, job_rank=rank)
        got = read_results(run_merge(cfg), qid_index(ds_qids))
    for case in SHARD_CASES:
        assert sorted(got[case.name]) == list(range(len(ds_qids.queries)))
        want = oracle_qids.topk(vector_type=case.vector_type, metric=case.metric,
                                k=case.k,
                                filt=_filter_from_dict(ds_qids, case.filter_dict))
        for qi in range(len(ds_qids.queries)):
            compare.assert_scores_agree(
                got[case.name][qi], want[qi], metric=case.metric,
                label=f"[{device}] {case.id} W={num_jobs} {_qid(qi)} vs naive")


# `rows` is not supported for multivector (a config error), so no mv here.
ROWS_CASES = [cases_mod.CASES_BY_NAME[n] for n in
              ("decos_nofilter", "deeuc_pqmatch", "dedot_pqtext", "spdot_match")]


@pytest.mark.parametrize("case", ROWS_CASES, ids=[c.id for c in ROWS_CASES])
def test_a_rows_subset_is_keyed_by_query_id(case, ds_qids, oracle_qids, device):
    """`rows` selects a subset of the query file; its output must name the
    SELECTED queries by their ids, not by their position within the subset."""
    halves = {h: spec(f"{case.name}_{h}", vector_type=case.vector_type,
                      metric=case.metric, k=case.k, filter=case.filter_dict,
                      rows={"column": "query_set", "isin": [h]})
              for h in ("even", "odd")}
    got = run(ds_qids, list(halves.values()), out_tag=f"qids_rows_{case.name}",
              device=device)
    want = oracle_qids.topk(vector_type=case.vector_type, metric=case.metric,
                            k=case.k, filt=_filter_from_dict(ds_qids, case.filter_dict))
    for h in halves:
        mine = [qi for qi, q in enumerate(ds_qids.queries)
                if q["payload"]["query_set"] == h]
        res = got[f"{case.name}_{h}"]
        assert sorted(res) == mine, (
            f"{case.id} rows={h}: answered queries {sorted(res)}, selected {mine}")
        for qi in mine:
            compare.assert_scores_agree(
                res[qi], want[qi], metric=case.metric,
                label=f"[{device}] {case.id} rows={h} {_qid(qi)} vs naive")
