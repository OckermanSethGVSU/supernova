"""`SearchSpec.rows` subsets laid out contiguously (`_vt_row_layout`).

Where a vector_type's searches select non-contiguous query rows (interleaved
query sets in one file), the DEVICE query matrix is permuted so every search's
rows are one block: its selector is then a `slice` (a view, and two-pass
eligible) instead of a gather. Each search's top-K state runs in that block
order and is put back in file order once, at the end of the scan; the
threshold exchange converts at its two hooks.

Everything a run writes must be what the unpermuted run writes. The reference
here is `NOVA_BF_NO_ROW_REORDER=1`, which takes exactly the pre-change code
path (file-order matrix, gather selectors). Results are compared as whole
Arrow tables, byte for byte, with the pass two pinned to slice mode, which is
bit-identical to one-pass. Under the default `tiles` pass two, groups the
layout newly makes two-pass eligible get tiles-mode results -- the same
near-tie guarantee every contiguous subset already gets (`twopass_compare`),
not byte identity; `test_tiles_mode_is_within_the_tiles_tolerance` pins that.
Only dense matrices are reordered (sparse has no two-pass to gain).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")
import pyarrow as pa
import pyarrow.parquet as pq

from nova_bf import compute as compute_mod
from nova_bf import thresholds, twopass
from nova_bf.compute import (
    _contiguous_order,
    _positions_in,
    _row_selector,
    _vt_row_layout,
    run_compute,
)
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    Filter,
    FilterCondition,
    OutputConfig,
    ParamsConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import run_merge

DIM, VOCAB, NNZ = 64, 40, 8
NQ, K, BATCH = 48, 5, 64


# --------------------------------------------------------------------- units
def _blocks_ok(order, sets):
    inv = {int(r): i for i, r in enumerate(order)}
    for s in sets:
        p = sorted(inv[int(r)] for r in s)
        assert p == list(range(p[0], p[0] + len(p))), (s, order)


def test_row_selector_needs_unit_steps_not_just_endpoints():
    """`[0, 2, 1, 3]` spans 0..3 with no gap in COUNT -- the old endpoint
    check collapsed it to `slice(0, 4)`, a different row order."""
    import torch

    sel = _row_selector(np.array([0, 2, 1, 3]), "cpu")
    assert isinstance(sel, torch.Tensor) and sel.tolist() == [0, 2, 1, 3]
    assert _row_selector(np.array([4, 5, 6]), "cpu") == slice(4, 7)


@pytest.mark.parametrize("seed", range(40))
def test_laminar_families_always_get_a_contiguous_order(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    universe = np.arange(n, dtype=np.int64)
    # A random laminar family: recursively split random shuffled groups.
    sets, stack = [], [rng.permutation(n)]
    while stack and len(sets) < 12:
        g = stack.pop()
        if len(g) == 0:
            continue
        sets.append(np.sort(g))
        if len(g) > 1:
            cut = sorted(rng.choice(np.arange(1, len(g)), size=min(2, len(g) - 1),
                                    replace=False))
            stack.extend(np.split(g, cut))
    order, why = _contiguous_order(sets, universe)
    assert why is None
    assert sorted(order.tolist()) == universe.tolist()
    _blocks_ok(order, sets)
    again, _ = _contiguous_order(list(reversed(sets)), universe)
    assert np.array_equal(order, again), "order depends on the specs' order"


def test_contiguous_family_maps_to_the_identity():
    u = np.arange(10)
    order, _ = _contiguous_order([np.arange(0, 4), np.arange(4, 10), u, np.arange(1, 3)], u)
    assert order.tolist() == list(range(10))


def test_properly_overlapping_sets_fall_back():
    u = np.arange(6)
    order, why = _contiguous_order([np.array([0, 2, 4]), np.array([2, 3, 5])], u)
    assert order is None and "laminar" in why


def test_layout_status():
    n = 8
    every = None
    odd, even = np.arange(1, n, 2), np.arange(0, n, 2)
    assert _vt_row_layout([0, 1], [np.arange(4), np.arange(4, 8)], every, n)[1] == "file_order"
    order, st, _ = _vt_row_layout([0, 1, 2], [None, odd, even], every, n)
    assert st == "reordered"
    _blocks_ok(order, [odd, even])
    _, st, why = _vt_row_layout([0, 1], [np.array([0, 2, 4]), np.array([2, 3, 5])],
                                np.array([0, 2, 3, 4, 5]), n)
    assert st == "fallback" and why


def test_positions_in_refuses_a_missing_row():
    with pytest.raises(RuntimeError):
        _positions_in(np.array([1, 5]), np.array([0, 1, 2]), 8)


# ---------------------------------------------------------------- end to end
def _sparse_rows(rng, n):
    out = []
    for _ in range(n):
        idx = np.sort(rng.choice(VOCAB, size=NNZ, replace=False))
        out.append({"indices": idx.tolist(),
                    "values": rng.standard_normal(NNZ).astype(np.float32).tolist()})
    return out


SPARSE_T = pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                      pa.field("values", pa.list_(pa.float32()))])


def _write(path, dense, sparse, **cols):
    data = {"dense_embedding": pa.array(dense.tolist(), pa.list_(pa.float32())),
            "sparse_embedding": pa.array(sparse, type=SPARSE_T)}
    data.update({k: pa.array(v) for k, v in cols.items()})
    pq.write_table(pa.table(data), str(path))


# Query-set column layouts (row i's set). `cycle3` makes the nested/overlap
# families; `holes` leaves rows no search owns (a strict-subset matrix).
LAYOUTS = {
    "interleaved": lambda i: "ab"[i % 2],
    "blocked": lambda i: "a" if i < NQ // 2 else "b",
    "cycle3": lambda i: "abc"[i % 3],
    "holes": lambda i: "ab-"[i % 3],
}


@pytest.fixture
def ds(tmp_path):
    rng = np.random.default_rng(23)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    g = 0
    tenants = "ABC"
    for fi in range(6):
        n = 160
        v = rng.standard_normal((n, DIM)).astype(np.float32)
        v = v.astype(np.float16).astype(np.float32)
        _write(cdir / f"f{fi}.parquet", v, _sparse_rows(rng, n),
               id=[f"c{g + r:06d}" for r in range(n)],
               tenant=[tenants[(g + r) % 3] for r in range(n)],
               text=[f"w{(g + r) % 7} " + ("red" if (g + r) % 2 else "blue")
                     for r in range(n)])
        g += n
    q = rng.standard_normal((NQ, DIM)).astype(np.float32)
    qs = _sparse_rows(rng, NQ)
    paths = {}
    for name, f in LAYOUTS.items():
        p = tmp_path / f"q_{name}.parquet"
        _write(p, q, qs, qid=[f"q{i:03d}" for i in range(NQ)],
               qset=[f(i) for i in range(NQ)],
               want=[tenants[i % 3] for i in range(NQ)],
               phrase=["red" if i % 4 else "blue" for i in range(NQ)])
        paths[name] = str(p)
    return {"cdir": str(cdir), "q": paths, "tmp": tmp_path}


@pytest.fixture(autouse=True)
def _cpu_twopass(monkeypatch):
    """Two-pass forced on CPU at fixture size, slice mode (bit-identical)."""
    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "slice")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.delenv("NOVA_BF_NO_ROW_REORDER", raising=False)
    monkeypatch.delenv("NOVA_BF_NO_TWOPASS", raising=False)
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 8)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 8)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 8)
    twopass.reset()
    yield
    twopass.reset()


def _sel(*sets):
    return {"column": "qset", "isin": list(sets)}


FAMILIES = {
    # name: (layout, [(search name, vector_type, metric, rows, filter key)])
    "interleaved_with_full": ("interleaved", [
        ("all", "dense", "cosine", None, None),
        ("a", "dense", "cosine", _sel("a"), None),
        ("b", "dense", "dot", _sel("b"), None)]),
    "interleaved_disjoint": ("interleaved", [
        ("a", "dense", "cosine", _sel("a"), None),
        ("b", "dense", "cosine", _sel("b"), None)]),
    "blocked": ("blocked", [
        ("all", "dense", "cosine", None, None),
        ("a", "dense", "cosine", _sel("a"), None),
        ("b", "dense", "cosine", _sel("b"), None)]),
    "nested": ("cycle3", [
        ("all", "dense", "cosine", None, None),
        ("ab", "dense", "cosine", _sel("a", "b"), None),
        ("a", "dense", "cosine", _sel("a"), None),
        ("c", "dense", "dot", _sel("c"), None)]),
    "overlapping": ("cycle3", [
        ("ab", "dense", "cosine", _sel("a", "b"), None),
        ("bc", "dense", "cosine", _sel("b", "c"), None)]),
    "holes": ("holes", [
        ("a", "dense", "cosine", _sel("a"), None),
        ("b", "dense", "dot", _sel("b"), None)]),
    "filters": ("interleaved", [
        ("all", "dense", "cosine", None, None),
        # shares a_gpu's filter, so that filter's mask is full height and
        # a_gpu's rows are a block in it only if the mask follows the layout
        ("all_gpu", "dense", "dot", None, "gpu"),
        ("a_gpu", "dense", "cosine", _sel("a"), "gpu"),
        ("b_text", "dense", "cosine", _sel("b"), "text"),
        ("b_uni", "dense", "dot", _sel("b"), "uniform")]),
    # A per-query filter shared by a reordered dense matrix and a file-order
    # sparse one keeps FILE order; the whole-matrix dense search must still
    # get its mask rows mapped into the permuted matrix order.
    "cross_vt_filter": ("interleaved", [
        ("dall", "dense", "cosine", None, "gpu"),
        ("da", "dense", "cosine", _sel("a"), None),
        ("sb", "sparse", "dot", _sel("b"), "gpu")]),
    "sparse": ("interleaved", [
        ("sall", "sparse", "dot", None, None),
        ("sa", "sparse", "cosine", _sel("a"), None),
        ("sb", "sparse", "dot", _sel("b"), "gpu"),
        ("da", "dense", "cosine", _sel("a"), None)]),
    # cdist (euclidean-only) is row-position dependent: never reordered.
    "euclidean_only": ("interleaved", [
        ("ea", "dense", "euclidean", _sel("a"), None),
        ("eb", "dense", "euclidean", _sel("b"), "gpu")]),
    # A whole-file euclidean search beside cosine/dot subsets: the shared
    # Gram path (no cdist), so it IS reordered.
    "euclidean_mixed": ("interleaved", [
        ("eall", "dense", "euclidean", None, None),
        ("ca", "dense", "cosine", _sel("a"), None),
        ("db", "dense", "dot", _sel("b"), None)]),
}

EXPECT_STATUS = {
    "euclidean_only": {"dense": "fallback"},
    "euclidean_mixed": {"dense": "reordered"},
    "interleaved_with_full": {"dense": "reordered"},
    "interleaved_disjoint": {"dense": "reordered"},
    "blocked": {"dense": "file_order"},
    "nested": {"dense": "reordered"},
    "overlapping": {"dense": "fallback"},
    "holes": {"dense": "reordered"},
    "filters": {"dense": "reordered"},
    "sparse": {"sparse": "fallback", "dense": "file_order"},
    "cross_vt_filter": {"dense": "reordered", "sparse": "file_order"},
}


def _filter(key):
    if key is None:
        return None
    if key == "gpu":
        return Filter(must=[FilterCondition(field="tenant", match_from_query="want")])
    if key == "text":
        return Filter(must=[FilterCondition(field="text", match_text_from_query="phrase")])
    return Filter(must=[FilterCondition(field="tenant", match="A")])


def _cfg(ds, family, out, **params):
    layout, searches = FAMILIES[family]
    out.mkdir(parents=True, exist_ok=True)
    return BruteForceConfig(
        corpus=CorpusConfig(path=ds["cdir"], id_column="id"),
        queries=QueriesConfig(path=ds["q"][layout], id_column="qid",
                              payload_fields=["qset", "want"]),
        output=OutputConfig(path=str(out)),
        params=ParamsConfig(io_workers=1, dense_batch_size=BATCH,
                            sparse_batch_size=BATCH, **params),
        searches=[SearchSpec(name=n, vector_type=vt, metric=mt, k=K, rows=r,
                             filter=_filter(f))
                  for n, vt, mt, r, f in searches],
    )


def _manifest(out):
    (p,) = glob.glob(os.path.join(str(out), "_bf_manifest_*_compute.json"))
    return json.load(open(p))


def _results(out):
    got = {}
    for p in sorted(glob.glob(os.path.join(str(out), "**", "*.parquet"), recursive=True)):
        got[os.path.relpath(p, str(out))] = pq.read_table(p)
    assert got
    return got


def _assert_tables_identical(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].equals(b[k], check_metadata=True), k


def _run(cfg, monkeypatch, reorder, **kw):
    if reorder:
        monkeypatch.delenv("NOVA_BF_NO_ROW_REORDER", raising=False)
    else:
        monkeypatch.setenv("NOVA_BF_NO_ROW_REORDER", "1")
    twopass.reset()
    run_compute(cfg, **kw)
    return twopass.stats()


@pytest.mark.parametrize("family", list(FAMILIES))
def test_outputs_identical_to_file_order(ds, monkeypatch, family):
    on_out, off_out = ds["tmp"] / f"{family}_on", ds["tmp"] / f"{family}_off"
    st_on = _run(_cfg(ds, family, on_out), monkeypatch, True)
    st_off = _run(_cfg(ds, family, off_out), monkeypatch, False)
    _assert_tables_identical(_results(on_out), _results(off_out))

    m_on, m_off = _manifest(on_out), _manifest(off_out)
    status = {vt: v["status"] for vt, v in m_on["params"]["query_row_layout"].items()}
    assert status == EXPECT_STATUS[family], m_on["params"]["query_row_layout"]
    assert m_on["counts"] == m_off["counts"]
    strip = lambda e: {k: v for k, v in e.items() if k not in ("output_path",)}
    assert [strip(e) for e in m_on["searches"]] == [strip(e) for e in m_off["searches"]]

    # (euclidean beside cosine/dot keeps the shared Gram's two-pass off)
    if status.get("dense") == "reordered" and family != "euclidean_mixed":
        # The point of the reorder: the dense groups now two-pass.
        # (sparse groups are never two-pass eligible, hence a comparison)
        assert st_on["groups_refused"] < st_off["groups_refused"], (st_on, st_off)
        # ("filters" mixes cosine+dot in one shared-Gram group, whose summed
        # live hints stay above the engage gate at this fixture size)
        if family != "filters":
            assert st_on["slices_twopass"] > 0, st_on
        assert st_off["groups_refused"] > 0, (
            f"the file-order reference already two-passed, so this layout does "
            f"not exercise the gather path: {st_off}")


def test_reordered_selectors_are_slices(ds, monkeypatch):
    """Spy the selectors `run_compute` hands the scoring loop."""
    seen = {}
    real = compute_mod._process_batch_group

    def spy(batch, member_idxs, specs, *a, **kw):
        for m in member_idxs:
            seen[specs[m].name] = (kw["spec_qsel"][m], kw["spec_qrows"][m])
        return real(batch, member_idxs, specs, *a, **kw)

    monkeypatch.setattr(compute_mod, "_process_batch_group", spy)
    _run(_cfg(ds, "filters", ds["tmp"] / "sel"), monkeypatch, True)
    assert seen["all"][0] is None
    for name in ("a_gpu", "b_text", "b_uni"):
        assert isinstance(seen[name][0], slice), (name, seen[name])
    for name in ("a_gpu", "b_text"):       # per-query filter masks: views too
        assert isinstance(seen[name][1], slice), (name, seen[name])


@pytest.mark.parametrize("exchange", [False, True])
def test_multi_rank_partials_merge_and_exchange_identical(ds, monkeypatch, exchange):
    """Two ranks (run in sequence, so rank 1 applies rank 0's thresholds),
    then the merge: every partial (incl. `applied_threshold`), the merged
    result, and the decoded threshold-exchange files are identical."""
    outs = {}
    for reorder in (True, False):
        out = ds["tmp"] / f"mr_{exchange}_{reorder}"
        cfg = _cfg(ds, "interleaved_with_full", out, threshold_exchange=exchange,
                   threshold_exchange_refresh_s=0.2)
        for r in range(2):
            _run(cfg, monkeypatch, reorder, num_jobs=2, job_rank=r)
        if reorder:
            assert _manifest_rank(out, 1)["params"]["query_row_layout"]["dense"]["status"] == "reordered"
        exch = _exchange_arrays(out, cfg) if exchange else None
        run_merge(cfg)
        outs[reorder] = (_results(out), exch)
    _assert_tables_identical(outs[True][0], outs[False][0])
    if exchange:
        a, b = outs[True][1], outs[False][1]
        assert a and set(a) == set(b)
        for k in a:
            for name in a[k]:
                assert np.array_equal(a[k][name], b[k][name]), (k, name)
        applied = [t for p, t in outs[True][0].items()
                   if p.endswith("rank001.parquet") and "applied_threshold" in t.column_names]
        assert applied and any(np.isfinite(t.column("applied_threshold").to_numpy()).any()
                               for t in applied), "rank 1 never applied a peer threshold"


def _manifest_rank(out, r):
    (p,) = glob.glob(os.path.join(str(out), "_bf_manifest_*_compute", f"rank{r:03d}.json"))
    return json.load(open(p))


def _exchange_arrays(out, cfg):
    sizes = {s.name: None for s in cfg.searches}
    got = {}
    for p in sorted(glob.glob(os.path.join(str(out), thresholds.SUBDIR, "**", "*"),
                              recursive=True)):
        if os.path.isdir(p):
            continue
        buf = open(p, "rb").read()
        h = {s.name: _height(cfg, s) for s in cfg.searches}
        sz = {n: (h[n], thresholds.publish_width(K, 2)) for n in sizes}
        _, arrays = thresholds.decode(buf, sz)
        got[os.path.relpath(p, str(out))] = arrays
    return got


def _height(cfg, s):
    if s.rows is None:
        return NQ
    layout = FAMILIES["interleaved_with_full"][0]
    return sum(LAYOUTS[layout](i) in s.rows.isin for i in range(NQ))


@pytest.mark.parametrize("family", ["interleaved_with_full", "nested", "holes"])
def test_tiles_mode_is_within_the_tiles_tolerance(ds, monkeypatch, family):
    """Default pass-two mode: newly engaged groups differ from the file-order
    (one-pass) reference only by tiles mode's summation-order near-ties."""
    from twopass_compare import assert_same, tolerance

    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "tiles")
    monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
    monkeypatch.setattr(twopass, "CPU_TILE", 16)
    monkeypatch.setattr(twopass, "TILE_CROSSOVER", 1.0)
    on_out, off_out = ds["tmp"] / f"t_{family}_on", ds["tmp"] / f"t_{family}_off"
    st = _run(_cfg(ds, family, on_out), monkeypatch, True)
    _run(_cfg(ds, family, off_out), monkeypatch, False)
    assert st["slices_tiles"] > 0, st
    a, b = _results(on_out), _results(off_out)
    assert set(a) == set(b)
    tol = tolerance(DIM, 13.0 ** 2)
    for k in a:
        assert_same("tiles", a[k], b[k], k, abs_tol=tol)


def test_reorder_bit_check_accepts_a_true_permutation_and_rejects_a_wrong_one():
    import torch

    from nova_bf.compute import _reorder_bit_check

    Q = torch.randn(40, DIM)
    order = np.random.default_rng(0).permutation(40)
    assert _reorder_bit_check(Q, Q[torch.from_numpy(order)], order, None) is None
    bad = Q[torch.from_numpy(order)].clone()
    bad[3, 0] += 1.0
    assert "not a row permutation" in _reorder_bit_check(Q, bad, order, None)
    # a strict-subset union: `order` holds FILE rows, the matrix their positions
    union = np.arange(0, 80, 2)
    rows = union[order]
    assert _reorder_bit_check(Q, Q[torch.from_numpy(order)], rows, union) is None


def test_a_failed_reorder_self_check_keeps_file_order(ds, monkeypatch):
    """A device whose GEMM bits depend on row position (e.g. MKL on CPU at
    d % 4 != 0) must fall back to file order, recorded, outputs unchanged."""
    monkeypatch.setattr(compute_mod, "_reorder_bit_check",
                        lambda *a: "reorder self-check: test")
    on, off = ds["tmp"] / "chk_on", ds["tmp"] / "chk_off"
    _run(_cfg(ds, "interleaved_with_full", on), monkeypatch, True)
    _run(_cfg(ds, "interleaved_with_full", off), monkeypatch, False)
    lay = _manifest(on)["params"]["query_row_layout"]["dense"]
    assert lay == {"status": "fallback", "reason": "reorder self-check: test"}, lay
    _assert_tables_identical(_results(on), _results(off))
