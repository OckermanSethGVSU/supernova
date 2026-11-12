"""`_SliceFilters`: the per-slice source of every per-query cell mask.

Two layers:

* unit tests with a fake mask builder pin memoisation, release after the last
  member, and the `select` contract;
* an end-to-end test runs `run_compute` on a tiny corpus and checks, at EVERY
  slice and member, that the mask `select` hands to score masking equals the
  mask recomputed independently from `filters.evaluate` over the true corpus
  rows of that slice -- with and without union compaction (`orig_rows`), with
  slices that start off a byte boundary, for a CPU text filter, a GPU-eligible
  filter and a uniform filter, under `rows:` selectors -- and that contiguous
  rows take the slice fast path.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute
from nova_bf.compute import _SliceFilters, run_compute, _to_query_array
from nova_bf.config import (
    BruteForceConfig, CorpusConfig, Filter, FilterCondition, OutputConfig,
    QueriesConfig, RowSelector, SearchSpec,
)
from nova_bf.filters import PackedRowMask, evaluate


# --- unit layer ---------------------------------------------------------------

F_A = Filter(must=[FilterCondition(field="tenant", match_from_query="tenant")])
F_B = Filter(must=[FilterCondition(field="score", range_from_query={"gte": "lo"})])
F_U = Filter(must=[FilterCondition(field="tenant", match="A")])


def _spec(name, f):
    return SearchSpec(name=name, vector_type="dense", metric="dot", k=3, filter=f)


def _make(masks, *, members, specs, uniform=None, n_cols=10):
    calls = []

    def build(f, rows, true_rows):
        calls.append(f)
        m = masks[f]
        return None if m is None else torch.from_numpy(m.copy())

    def uni(f, true_rows):
        return torch.from_numpy(np.nonzero(uniform[f][true_rows])[0])

    sf = _SliceFilters(
        specs, members, torch.arange(n_cols), slice(0, n_cols), n_cols, "cpu",
        build_mask=build, uniform_idx=uni,
        per_query=lambda f: f in (F_A, F_B),
    )
    return sf, calls


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    mA = rng.random((6, 10)) < 0.15          # F_A: shared by members 0 and 1
    mB = rng.random((4, 10)) < 0.3           # F_B: member 2
    specs = [_spec("a0", F_A), _spec("a1", F_A), _spec("b", F_B), _spec("u", F_U),
             _spec("none", None)]
    keep_u = np.zeros(10, dtype=bool)
    keep_u[[4, 9]] = True
    return dict(masks={F_A: mA, F_B: mB}, specs=specs,
                uniform={F_U: keep_u}, members=[0, 1, 2, 3, 4])


def _args(setup):
    return {k: setup[k] for k in ("masks", "members", "specs", "uniform")}


def test_empty_mask_selects_nothing(setup):
    setup["masks"][F_B] = None
    sf, _ = _make(**_args(setup))
    assert sf.select(2) == (None, None, None)


def test_uniform_with_no_column_selects_nothing(setup):
    setup["uniform"][F_U][:] = False
    sf, _ = _make(**_args(setup))
    assert sf.select(3) == (None, None, None)


def test_select_contract_and_memoisation(setup):
    sf, calls = _make(**_args(setup))
    r0 = sf.select(0)
    assert F_A in sf._masks                   # member 1 still to come
    r1 = sf.select(1)
    assert r0[2] is r1[2]                     # one tensor backs both F_A members
    np.testing.assert_array_equal(r0[2].numpy(), setup["masks"][F_A])  # FILTER rows
    assert r0[1] is None and torch.equal(r0[0], torch.arange(10))
    sf.select(2)
    assert calls == [F_A, F_B] and sf.builds == 2   # built once per filter
    assert sf._masks == {}                    # released after the last select
    rows, cols, cm = sf.select(3)             # uniform: column selection
    np.testing.assert_array_equal(cols.numpy(), [4, 9])
    np.testing.assert_array_equal(rows.numpy(), [4, 9])
    assert cm is None
    assert sf.select(4) == (sf._rows, None, None)   # unfiltered


def test_process_batch_group_needs_exactly_one_selector():
    batch = compute.DenseCorpusBatch(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="exactly one"):       # both
        compute._process_batch_group(
            batch, [], [], [], [], [], [], [], None, 0, "cpu", None,
            lambda *a: None, [], [], [], slice_filters=lambda *a: None)
    with pytest.raises(ValueError, match="exactly one"):       # neither
        compute._process_batch_group(
            batch, [], [], [], [], [], [], [], None, 0, "cpu", None,
            None, [], [], [])


# --- end-to-end layer ---------------------------------------------------------

N, D = 23, 8          # 23 rows, dense_batch_size 5 -> partial slices, off-byte starts
TENANTS = ["A", "B", "C", "A", "B", "A", "C", "C", "B", "A", "A", "B",
           "C", "A", "B", "B", "A", "C", "A", "B", "C", "A", "B"]
WORDS = ["red", "blue", "green", "red blue", "blue", "green red", "x", "red",
         "blue green", "x", "red", "green", "blue", "red x", "x", "green",
         "blue red", "x", "green", "red", "blue", "x", "green blue"]


def _dataset(tmp_path):
    rng = np.random.default_rng(1)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(N)],
        "dense": pa.array(rng.standard_normal((N, D)).astype(np.float32).tolist(),
                          pa.list_(pa.float32())),
        "tenant": TENANTS,
        "score": rng.random(N).round(3).tolist(),
        "text": WORDS,
    }), cdir / "part0.parquet")
    nq = 9
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "qid": [f"q{i}" for i in range(nq)],
        "dense": pa.array(rng.standard_normal((nq, D)).astype(np.float32).tolist(),
                          pa.list_(pa.float32())),
        "grp": ["t", "s", "t", "s", "t", "s", "u", "u", "t"],
        # No query names tenant C (the GPU filter's row union is built from EVERY
        # query's values, see `_row_union_from_gpu_leaves`), so tenant-C rows the
        # text queries reject are outside every filter -> union compaction.
        "tenant": ["A", "B", "Z", "Z", "Z", "B", "A", "B", "B"],
        "lo": [0.1, 0.5, 0.9, 0.6, 0.3, 0.7, 0.2, 0.4, 0.6],
        "kw": ["red", "green", "x", "x", "nothing", "red", "x", "blue", "green"],
    }), qpath)
    return str(cdir), str(qpath)


F_TEXT = Filter(must=[FilterCondition(field="text", match_text_from_query="kw")])
F_GPU = Filter(must=[FilterCondition(field="tenant", match_from_query="tenant"),
                     FilterCondition(field="score", range_from_query={"gte": "lo"})])
F_STATIC = Filter(must=[FilterCondition(field="tenant", match="A")])


@pytest.mark.parametrize("with_baseline", [False, True])
def test_select_masks_match_independent_masks_end_to_end(tmp_path, monkeypatch, with_baseline):
    cdir, qpath = _dataset(tmp_path)
    # Each per-query filter is SHARED: the text filter by two specs over
    # different `rows:` (so its mask spans both), the GPU filter by a `rows:`
    # spec and a spec with no `rows:` (so the filter's rows are ALL queries).
    searches = [
        SearchSpec(name="text_t", vector_type="dense", metric="dot", k=4, filter=F_TEXT,
                   rows=RowSelector(column="grp", isin=["t"])),
        SearchSpec(name="text_s", vector_type="dense", metric="dot", k=4, filter=F_TEXT,
                   rows=RowSelector(column="grp", isin=["s"])),
        SearchSpec(name="gpu", vector_type="dense", metric="dot", k=4, filter=F_GPU,
                   rows=RowSelector(column="grp", isin=["s"])),
        SearchSpec(name="gpu_all", vector_type="dense", metric="dot", k=4, filter=F_GPU),
        SearchSpec(name="static", vector_type="dense", metric="dot", k=4, filter=F_STATIC,
                   rows=RowSelector(column="grp", isin=["u"])),
    ]
    if with_baseline:   # an unfiltered member -> no union compaction, slice true_rows
        searches.append(SearchSpec(name="all", vector_type="dense", metric="dot", k=4))
    cfg = BruteForceConfig(
        corpus=CorpusConfig(path=cdir, id_column="id", dense_column="dense"),
        queries=QueriesConfig(path=qpath, id_column="qid", dense_column="dense",
                              payload_fields=["grp", "tenant", "lo", "kw"]),
        output=OutputConfig(path=str(tmp_path / "out")),
        searches=searches,
        params={"dense_batch_size": 5, "io_workers": 1, "threshold_exchange": False},
    )

    seen = []

    class Rec(_SliceFilters):
        __slots__ = ()

        def select(self, m):
            tr = self._true_rows
            cols = (np.arange(tr.start, tr.stop) if isinstance(tr, slice)
                    else np.asarray(tr).copy())
            got = super().select(m)
            sel_cols = got[1]
            cm = got[2]
            seen.append((m, cols, isinstance(tr, slice), got[0] is None,
                         None if sel_cols is None else sel_cols.cpu().numpy().copy(),
                         None if cm is None else cm.cpu().numpy().copy()))
            return got

    monkeypatch.setattr(compute, "_SliceFilters", Rec)
    run_compute(cfg)

    # Independent oracle: the CPU evaluator over the whole file and ALL queries
    # (a query's allowed rows do not depend on the other queries). A per-query
    # filter's mask is in FILTER rows: the union of its members' own query rows,
    # in query order (all queries when any member has no `rows:`).
    table = pq.read_table(f"{cdir}/part0.parquet")
    qt = pq.read_table(qpath)
    grp = np.asarray(qt["grp"].to_pylist())
    qv_all = {c: _to_query_array(qt[c].to_pylist()) for c in ("tenant", "lo", "kw")}

    def own_rows(s):
        return (np.arange(len(grp)) if s.rows is None
                else np.nonzero(np.isin(grp, s.rows.isin))[0])

    full = {}
    filter_rows = {}
    for s in searches:
        if s.filter is None:
            continue
        got = evaluate(s.filter, table, qv_all)
        if isinstance(got, PackedRowMask):
            full[s.filter] = np.unpackbits(got.packed, axis=1, count=N).astype(bool)
        else:
            full[s.filter] = np.broadcast_to(got, (len(grp), N))
        prev = filter_rows.get(s.filter, np.zeros(0, dtype=np.int64))
        filter_rows[s.filter] = np.union1d(prev, own_rows(s))

    checked = 0
    saw_compacted = saw_offbyte = False
    kinds = set()
    for m, cols, as_slice, none_sel, sel_cols, cm in seen:
        # Contiguous-rows fast path: a slice whose rows are one contiguous run
        # (including a contiguous run inside a union-compacted batch) must be
        # handed on as a `slice`; only a genuinely scattered one as an array.
        contiguous = np.array_equal(cols, np.arange(cols[0], cols[0] + len(cols)))
        assert as_slice == contiguous, (cols, as_slice)
        kinds.add(as_slice)
        s = searches[m]
        if s.filter is None:
            assert not none_sel and sel_cols is None and cm is None
            continue
        if s.filter == F_STATIC:
            keep = full[F_STATIC][0, cols]
            if keep.any():
                np.testing.assert_array_equal(sel_cols, np.nonzero(keep)[0])
                assert cm is None
            else:
                assert none_sel
        else:
            want = full[s.filter][filter_rows[s.filter]][:, cols]
            if cm is None:
                assert none_sel and not want.any()
            else:
                np.testing.assert_array_equal(cm, want)
        checked += 1
        saw_compacted |= not contiguous
        saw_offbyte |= bool(cols[0] % 8)
    assert checked >= 3
    assert saw_offbyte
    if not with_baseline:
        assert saw_compacted, "expected union compaction to give non-contiguous rows"
        # Both paths exercised inside one compacted batch: contiguous slices
        # take the packed-slice path, scattered ones the bit gather.
        assert kinds == {True, False}
    else:
        assert kinds == {True}
