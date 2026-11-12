"""The CUDA-only fast paths, each held to an ORACLE and each PROVEN to have run.

Written on a CPU-only box to be run on a GPU one (A10G). Two rules for every
test here:

  1. **Against an independent oracle, not only against the CPU.** A GPU path
     compared only with the CPU path of the same engine cannot see a bug both
     share (a wrong boundary convention, a mis-stated tie rule). Each test
     compares with `naive` (scores to tolerance, plus length/order/uniqueness
     via `layouts.assert_topk`) or, for the exact-tie fixtures, with an exact
     id sequence.
  2. **The path must be shown to have executed.** These paths fail OPEN — a
     declining kernel silently falls back to the portable path and gives the
     same answer — so each test records executed nova_bf lines with
     `reach.LineRecorder` (or reads the run's own counters) and REQUIRES the
     lines only that path reaches.

On a machine without CUDA every test in the `cuda` group skips with a reason;
the `reach` self-tests at the bottom run everywhere (they prove the recorder
itself would catch a path that did not run).

Run on the GPU box (from python/nova-bf/):

    NOVA_BF_PARITY_REACH_OUT=/tmp/reach.json \\
    QDRANT_URL=... pytest tests/parity/test_parity_gpu_paths.py -v
    python -m tests.parity.reach /tmp/reach.json   # OK/FAIL per requirement

REQUIRED LINES (what each test proves ran — `reach.require` snippets)
  triton_reduce MaxSim   multivector_kernels.py `def fused_ragged_maxsim_reduce(` body
                         + compute.py `fused_ragged_maxsim_reduce(` call
  MV fp16 prune          compute._MV_PRUNE["calls"] > 0 and ["pruned"] > 0
  MV device residency    compute.py `flat_gpu = torch.from_numpy(batch.flat_tokens).to(device)`
  MV double buffer ring  compute.py `stage_tokens = max(` (and NOT the residency line)
  dense pinned prefetch  dense_transfer_usage(): pinned_slices>0, copy_stream_slices>0;
                         compute.py `def _run_dense_prefetch(` body
  dense pageable (PINNED=0)  pinned_slices == 0
  pipelined prefetch     compute.py `    _run_dense_prefetch_pipelined(batch` (the call, not the def)
  two-pass fused rowmax  twopass.stats()["slices_fused"] > 0 (NO_FUSED_ROWMAX arm: == 0)
  two-pass tile kernel   tile_rescore._LAUNCHES > 0
  merge device fold      merge._DEVICE_FOLDS[0] > 0
  GPU id ordinals        tiebreak._gpu_perm_from_lanes / _gpu_perm called
  resident text windows  filters.combo_usage(): resident_uploads > 0, device_builds > 0
  budget fallback        filters.combo_usage(): budget_fallbacks > 0
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from . import compare, layouts, reach
from .devices import has_cuda, parity_devices

import os

# `NOVA_BF_PARITY_GPU_DRYRUN=1` runs every `cuda` test body on the CPU with the
# GPU-ONLY assertions (counters, `reach` requirements) switched off and the
# oracle comparisons kept. It exists to validate the plumbing of this file on
# a box with no GPU — fixture construction, spec names, oracle calls — so the
# first A10G run fails only for GPU reasons. It proves nothing about the GPU.
DRYRUN = os.environ.get("NOVA_BF_PARITY_GPU_DRYRUN", "") not in ("", "0")
DEV = "cpu" if DRYRUN else "cuda"

needs_cuda = pytest.mark.skipif(
    not (has_cuda() or DRYRUN),
    reason=f"CUDA-only path (parity devices: {parity_devices()})")


def gpu_check(cond, msg=""):
    """A GPU-path assertion (a counter, a kernel launch): skipped in DRYRUN."""
    if not DRYRUN:
        assert cond, msg


def require(rec, *a, **kw):
    if not DRYRUN:
        require(rec, *a, **kw)


def forbid(rec, *a, **kw):
    if not DRYRUN:
        forbid(rec, *a, **kw)


# ============================================================ multivector


@pytest.fixture(scope="module")
def mv_ds(tmp_path_factory):
    from .test_parity_mv_edges import _build

    return _build(tmp_path_factory.mktemp("gpu_mv"))


def _mv_specs(k_small=8, k_all=200):
    return [layouts.spec(f"mv_{m}{sfx}", vector_type="multivector", metric=m, k=k)
            for m in ("dot", "cosine") for sfx, k in (("", k_small), ("_all", k_all))]


def _assert_mv_vs_naive(ds, hits, label, k_small=8, k_all=200):
    for name, by_q in hits.items():
        metric = "dot" if "_dot" in name else "cosine"
        k = k_all if name.endswith("_all") else k_small
        want = layouts.naive_topk(ds, vector_type="multivector", metric=metric, k=k)
        layouts.assert_all(by_q, want, metric=metric, label=f"{label} {name}")


@needs_cuda
@pytest.mark.parametrize("prune", ["off", "fp16"])
def test_triton_maxsim_on_awkward_shapes_matches_the_oracle(prune, mv_ds, monkeypatch):
    """1-token and 230/260-token docs, zero-norm tokens, null/empty docs, a
    40-token query, row-group-chunked files: the fused ragged reduction's
    boundary arithmetic (per-doc token spans inside one tile, per-query token
    spans inside one block) is exactly what these shapes stress.

    Mutants this catches (not runnable on CPU): an off-by-one doc boundary in
    `_fused_ragged_reduce_kernel` (a 1-token doc borrowing its neighbour's
    token), an empty doc not left at -inf (it would surface at 0.0 in the
    k=200 list), a query span cut at the block edge (the 40-token query), a
    per-block `query_start` offset error (multivector_query_block=3)."""
    from nova_bf import multivector_kernels as mk

    calls = {"n": 0}
    real = mk.fused_ragged_maxsim_reduce

    def spy(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(mk, "fused_ragged_maxsim_reduce", spy)
    with reach.LineRecorder(f"triton_maxsim[{prune}]") as rec:
        hits = layouts.run(mv_ds, _mv_specs(), tag=f"gpu_triton_{prune}", device=DEV,
                           params={"multivector_kernel": "torch" if DRYRUN else "triton_reduce",
                                   "multivector_prune": prune,
                                   "multivector_batch_size": 5,
                                   "multivector_query_block": 3})
    gpu_check(calls["n"] > 0, "triton_reduce never launched — every block declined")
    require(rec, "multivector_kernels.py", "_fused_ragged_reduce_kernel[",
                "the Triton reducer must have been launched")
    _assert_mv_vs_naive(mv_ds, hits, f"[cuda] triton_reduce prune={prune}")


@needs_cuda
@pytest.mark.parametrize("kernel", ["torch", "triton_reduce"])
def test_mv_fp16_prune_on_awkward_shapes_matches_the_oracle(kernel, mv_ds):
    """The fp16 certified prune (`mv_fp16`) with zero-norm tokens and 1-token
    docs — whose fp16 maxima are exactly 0 / a single product, the edge of its
    rounding bound. `multivector_min_prune_rate=0` keeps the gate open so the
    prune runs on every slice. Catches a bound that prunes a live doc (a doc
    missing from the k=8 list), and a prune that mishandles a -inf (empty)
    doc."""
    import nova_bf.compute as compute_mod

    hits = layouts.run(mv_ds, _mv_specs(k_small=3, k_all=200), tag=f"gpu_mvprune_{kernel}",
                       device=DEV,
                       params={"multivector_kernel": "torch" if DRYRUN else kernel,
                               "multivector_prune": "fp16",
                               "multivector_min_prune_rate": 0.0,
                               "multivector_batch_size": 5})
    st = dict(compute_mod._MV_PRUNE)
    gpu_check(st["calls"] > 0 and st["pruned"] > 0, f"the fp16 prune never pruned: {st}")
    _assert_mv_vs_naive(mv_ds, hits, f"[cuda] mv fp16 prune {kernel}", k_small=3)


@needs_cuda
def test_mv_device_residency_matches_the_oracle(mv_ds):
    """Whole-batch residency: the batch's tokens are uploaded once and every
    slice is a device VIEW with rebased offsets (`off_gpu[r0:r1+1] -
    off_gpu[r0]`). A rebase error shifts every doc after the first slice;
    with 5-doc slices over the 1/230/260-token docs that moves real hits."""
    with reach.LineRecorder("mv_residency") as rec:
        hits = layouts.run(mv_ds, _mv_specs(), tag="gpu_mvres", device=DEV,
                           params={"multivector_batch_size": 5, "multivector_prune": "off"})
    require(rec, "compute.py", "flat_gpu = torch.from_numpy(batch.flat_tokens).to(device)",
                "residency declined (free VRAM?) — the slice loop ran instead")
    _assert_mv_vs_naive(mv_ds, hits, "[cuda] mv residency")


@needs_cuda
def test_mv_double_buffer_ring_matches_the_oracle(mv_ds):
    """The pinned two-buffer ring + transfer stream. Mutants it catches: the
    ring index swapped (`stages[(index + 1) % 2]` -> slice i scored on slice
    i+1's tokens), the offsets rebased against the wrong slice start, the
    staging buffer sized from the FIRST slice instead of the largest (the
    260-token doc lands mid-file), a missing wait_event (scoring a buffer
    still being copied — flaky, but the k=200 list sees it)."""
    with reach.LineRecorder("mv_double_buffer") as rec:
        hits = layouts.run(mv_ds, _mv_specs(), tag="gpu_mvdb", device=DEV,
                           params={"multivector_double_buffer": True,
                                   "multivector_batch_size": 3, "multivector_prune": "off"})
    require(rec, "compute.py", "stage_tokens = max(", "the double-buffer ring did not run")
    forbid(rec, "compute.py", "flat_gpu = torch.from_numpy(batch.flat_tokens).to(device)",
               "residency must not replace the ring when double buffering is requested")
    _assert_mv_vs_naive(mv_ds, hits, "[cuda] mv double buffer")


# ================================================================== dense


@pytest.fixture(scope="module")
def dense_ds(tmp_path_factory):
    from .test_parity_layouts import STORAGE, _dense_ds

    return {n: _dense_ds(tmp_path_factory.mktemp(f"gpu_{n}"), n, **STORAGE[n])
            for n in ("f16", "many", "mixed")}


def _dense_vs_naive(ds, hits, label, k):
    for name, by_q in hits.items():
        metric = name.split("_")[1]
        want = layouts.naive_topk(ds, vector_type="dense", metric=metric, k=k)
        layouts.assert_all(by_q, want, metric=metric, label=f"{label} {name}")


TRANSFER = [("2", "0"), ("2", "1"), ("0", "0")]


@needs_cuda
@pytest.mark.parametrize("name", ["f16", "many", "mixed"])
@pytest.mark.parametrize("pinned,inline", TRANSFER, ids=["overlapped", "inline", "pageable"])
def test_dense_transfer_modes_match_the_oracle(pinned, inline, name, dense_ds, monkeypatch):
    """`NOVA_BF_PINNED` 2 (pinned staging ring + copy stream, prefetching the
    next slice) / 0 (pageable), and `NOVA_BF_PREFETCH_INLINE`. float16 files
    stay float16 across PCIe and widen on the device; the mixed corpus puts
    float16/float32/float64 batches through the SAME pinned size class.
    Catches: a prefetched slice consumed for the wrong (r0, r1) (`_prefetched`
    key mix-up), the staging buffer reused before its copy finished, a
    float16 slice widened from the wrong half of a staging buffer."""
    import nova_bf.compute as compute_mod

    monkeypatch.setenv("NOVA_BF_PINNED", pinned)
    monkeypatch.setenv("NOVA_BF_PREFETCH_INLINE", inline)
    ds = dense_ds[name]
    specs = [layouts.spec(f"d_{m}", metric=m, k=7) for m in ("dot", "cosine", "euclidean")]
    with reach.LineRecorder(f"dense_transfer[{name},{pinned},{inline}]") as rec:
        hits = layouts.run(ds, specs, tag=f"gpu_xfer_{name}_{pinned}_{inline}",
                           device=DEV, params={"dense_batch_size": 8, "two_pass": "off"})
    usage = compute_mod.dense_transfer_usage()
    if pinned == "2":
        gpu_check(usage["pinned_slices"] > 0 and usage["copy_stream_slices"] > 0, usage)
        require(rec, "compute.py", "    _run_dense_prefetch(batch, ranges, device, process_slice)",
                    "the overlapped prefetch loop did not run")
    else:
        assert usage["pinned_slices"] == 0, usage
    assert usage["prefetch_inline"] == (inline == "1"), usage
    _dense_vs_naive(ds, hits, f"[cuda] transfer {name} PINNED={pinned} INLINE={inline}", 7)


@pytest.fixture
def forced_twopass(monkeypatch):
    from nova_bf import twopass

    monkeypatch.setattr(twopass, "PAD_QUANTUM", 2)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 2)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.delenv("NOVA_BF_TWOPASS_NO_VERIFY", raising=False)
    if DRYRUN:
        monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
        monkeypatch.setattr(twopass, "CPU_TILE", 4)
    twopass.reset()
    yield twopass
    twopass.reset()


@pytest.fixture(scope="module")
def tp_ds(tmp_path_factory):
    from .test_parity_layouts import _dense_ds

    return {st: _dense_ds(tmp_path_factory.mktemp(f"gpu_tp_{st}"), f"tp_{st}",
                          [(97, st), (61, st), (79, st)], seed=7, n_q=26)
            for st in ("float16", "float32")}


TP = [(mode, pipe, fused) for mode in ("slice", "tiles") for pipe in ("0", "1")
      for fused in ("fused", "unfused")]


@needs_cuda
@pytest.mark.parametrize("storage", ["float16", "float32"])
@pytest.mark.parametrize("mode,pipe,fused", TP, ids=[f"{a}-pipe{b}-{c}" for a, b, c in TP])
def test_two_pass_cuda_paths_match_the_oracle(mode, pipe, fused, storage, tp_ds,
                                              forced_twopass, monkeypatch):
    """The two-pass on its home device: the fused fp16 pass-one row-max
    kernel (or, with `NOVA_BF_NO_FUSED_ROWMAX`, cuBLAS + row max), slice or
    TILE rescoring (the Triton tile kernel), pass one pipelined on a side
    stream or not — over a float16 corpus (`exact_fp16`, no corpus-side
    conversion term) and a float32 one. Held to the naive oracle on the
    STORED values, and required to have pruned.

    Catches (GPU-only): a fused row-max reading the wrong tile column range
    (a live row pruned -> missing hit), the tile kernel masking the wrong
    (query, tile) pair, a pipelined pass one using the PREVIOUS slice's
    thresholds after they were raised, an fp16 corpus being double-rounded."""
    import nova_bf.tile_rescore as tr

    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", mode)
    monkeypatch.setenv("NOVA_BF_PIPELINE_PASS_ONE", pipe)
    if mode == "tiles":
        monkeypatch.setenv("NOVA_BF_TWOPASS_TILE_THRESHOLD", "1.0")
        monkeypatch.setattr(forced_twopass, "TILE_CROSSOVER", 1.0)
    if fused == "unfused":
        monkeypatch.setenv("NOVA_BF_NO_FUSED_ROWMAX", "1")
    ds = tp_ds[storage]
    for m in ("dot", "cosine"):
        forced_twopass.reset()
        tr.reset()
        with reach.LineRecorder(f"twopass[{storage},{mode},{pipe},{fused},{m}]") as rec:
            hits = layouts.run(ds, [layouts.spec(f"t_{m}", metric=m, k=5)],
                               tag=f"gpu_tp_{storage}_{mode}_{pipe}_{fused}_{m}",
                               device=DEV, params={"dense_batch_size": 8})
        st = forced_twopass.stats()
        assert st["slices_twopass"] > 0 and st["rows_full"] > st["rows_live"], st
        if fused == "fused":
            gpu_check(st["slices_fused"] > 0, f"fused row-max never ran: {st}")
        else:
            assert st["slices_fused"] == 0, st
            gpu_check(st["slices_unfused"] > 0, st)
        if mode == "tiles":
            gpu_check(tr._LAUNCHES > 0, "the Triton tile-rescore kernel never launched")
        if pipe == "1":
            require(rec, "compute.py", "    _run_dense_prefetch_pipelined(batch",
                        "pass one was not pipelined")
        _dense_vs_naive(ds, hits, f"[cuda] two-pass {storage}/{mode}/pipe{pipe}/{fused}", 5)


# ================================================= merge fold / id ordinals


@pytest.fixture(scope="module")
def tie_ds(tmp_path_factory):
    from .test_parity_layouts import _dense_ds

    # Same exact-tie construction as test_parity_layouts, ids ZERO-PADDED so
    # they are fixed-width (the GPU ordinal tiers decline ragged ids).
    return _dense_ds(tmp_path_factory.mktemp("gpu_ties"), "ties_pad",
                     [(6, "float32"), (4, "float32"), (0, "float32"), (9, "float32"),
                      (5, "float32"), (7, "float32"), (1, "float32"), (8, "float32"),
                      (6, "float32")], seed=9, dup_pool=6, id_fmt="{:06d}")


def _run_sharded(ds, specs, *, num_jobs, tag, params):
    from nova_bf.compute import run_compute
    from nova_bf.merge import run_merge

    from .runner import pinned_device

    cfg = layouts.build_config(ds, specs, tag=tag, params=params)
    with pinned_device(DEV):
        for rank in range(num_jobs):
            run_compute(cfg, num_jobs=num_jobs, job_rank=rank)
        paths = run_merge(cfg)
    import pyarrow.parquet as pq

    out = {}
    for name, path in paths.items():
        t = pq.read_table(path).to_pydict()
        out[name] = {int(q): [(int(i), float(s)) for i, s in zip(ids, sc)]
                     for q, ids, sc in zip(t["query_id"], t["hit_ids"], t["hit_scores"])}
    return out


@needs_cuda
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
@pytest.mark.parametrize("metric", ["dot", "euclidean"])
def test_merge_device_fold_and_gpu_ordinals_on_exact_ties(tiebreak, metric, tie_ds, monkeypatch):
    """compute --num-jobs 3 (each rank's ties ranked by its LOCAL ordinals) and
    a merge whose device fold must resolve CROSS-rank ties by the global rule.
    Exact id sequences against the exact oracle. Requires the merge's device
    fold and, under `tiebreak: id`, the GPU ordinal tier.

    Catches: the device fold resolving a cross-rank tie by rank order instead
    of id/ordinal (a different id at every tied boundary), the GPU id-rank
    lanes ordering bytes little-endian, a fold that drops the tail of a
    partial when two partials tie at the k-th score."""
    import nova_bf.merge as merge_mod
    import nova_bf.tiebreak as tb

    from .test_parity_layouts import _exact_oracle

    seen = []
    real, real_dev = tb._gpu_perm, tb._gpu_perm_from_lanes
    monkeypatch.setattr(tb, "_gpu_perm", lambda l, m: (seen.append(m), real(l, m))[1])
    monkeypatch.setattr(tb, "_gpu_perm_from_lanes",
                        lambda l: (seen.append("device"), real_dev(l))[1])
    specs = [layouts.spec(f"t{k}", metric=metric, k=k) for k in (3, 7, 13)]
    hits = _run_sharded(tie_ds, specs, num_jobs=3, tag=f"gpu_merge_{tiebreak}_{metric}",
                        params={"dense_batch_size": 10, "tiebreak": tiebreak})
    gpu_check(merge_mod._DEVICE_FOLDS[0] > 0, "the merge never took the device fold")
    if tiebreak == "id":
        gpu_check(seen, "no GPU ordinal tier ran")
    for k in (3, 7, 13):
        want = _exact_oracle(tie_ds, metric, k, tiebreak, id_fmt="{:06d}")
        for qi in want:
            assert [r for r, _ in hits[f"t{k}"][qi]] == [r for r, _ in want[qi]], (
                f"[cuda] merged {tiebreak}/{metric} k={k} q{qi}\n"
                f"  got:  {hits[f't{k}'][qi]}\n  want: {want[qi]}")


@needs_cuda
@pytest.mark.parametrize("tiebreak", ["ordinal", "id"])
def test_single_rank_gpu_ties_with_coalescing(tiebreak, tie_ds):
    """One rank, statically filtered (cross-file coalescing), on CUDA with the
    Triton pre-top-K and fold kernels live: the device-side tie handling of a
    coalesced batch, exact against the oracle."""
    from .test_parity_layouts import STATIC, _exact_oracle, _keep

    specs = [layouts.spec(f"t{k}", metric="dot", k=k, filter=STATIC) for k in (3, 7, 13)]
    hits = layouts.run(tie_ds, specs, tag=f"gpu_ties1_{tiebreak}", device=DEV,
                       params={"dense_batch_size": 10, "tiebreak": tiebreak})
    for k in (3, 7, 13):
        want = _exact_oracle(tie_ds, "dot", k, tiebreak, _keep("static"), id_fmt="{:06d}")
        for qi in want:
            assert [r for r, _ in hits[f"t{k}"][qi]] == [r for r, _ in want[qi]], (
                tiebreak, k, qi, hits[f"t{k}"][qi], want[qi])


# ======================================================= filter combos on GPU


def _shared_filtered(ds, oracle, names, hits, label):
    from .cases import FILTERS
    from .test_parity_matrix import _filter_from_dict

    for name, vt, metric, fname in names:
        want = oracle.topk(vector_type=vt, metric=metric, k=25,
                           filt=_filter_from_dict(ds, FILTERS[fname]))
        for qi in range(len(ds.queries)):
            layouts.assert_topk(hits[name][qi], want[qi], metric=metric,
                                label=f"{label} {name} q{qi}")


COMBO_SPECS = [("c_dense_pqtext", "dense", "cosine", "pqtext"),
               ("c_sparse_pqtext", "sparse", "dot", "pqtext"),
               ("c_mv_pqtext", "multivector", "cosine", "pqtext"),
               ("c_dense_mixed", "dense", "dot", "pqtext_pqmatch_and"),
               ("c_sparse_mixed", "sparse", "cosine", "pqrange_statictext")]


def _combo_run(ds, tag, params):
    from .cases import FILTERS
    from .runner import run, spec

    present = [c for c in COMBO_SPECS if c[3] in FILTERS]
    hits = run(ds, [spec(n, vector_type=vt, metric=m, k=25, filter=FILTERS[f])
                    for n, vt, m, f in present],
               out_tag=tag, device=DEV, params=params)
    return hits, present


@needs_cuda
def test_resident_text_windows_on_device_match_the_oracle(ds, oracle, monkeypatch):
    """Per-query text filters combined ON the device from pooled, windowed
    phrase rows (`_RESIDENT_POOLS`). Catches a window offset error (the phrase
    rows of the wrong file or the wrong row range combined into a query's
    mask) and a pooled buffer reused across files without being rewritten."""
    import nova_bf.compute as compute_mod
    from nova_bf import filters

    # 16 bytes of packed phrase rows per window: ~8 rows per window per part,
    # so each 61..97-row corpus file is uploaded in many windows.
    monkeypatch.setattr(compute_mod, "_RESIDENT_WINDOW_MAX_BYTES", 16)
    hits, present = _combo_run(ds, "gpu_resident",
                               {"dense_batch_size": 37, "sparse_batch_size": 29,
                                "multivector_batch_size": 23})
    u = filters.combo_usage()
    from .corpus import FILE_SIZES
    n_files = len(FILE_SIZES)
    gpu_check(u["device_builds"] > 0, u)
    # Several WINDOWS per file (the window is shrunk below), so a window
    # offset error has somewhere to show.
    gpu_check(u["resident_uploads"] > 4 * n_files, u)
    _shared_filtered(ds, oracle, present, hits, "[cuda] resident windows")


@needs_cuda
def test_budget_fallback_on_device_matches_the_oracle(ds, oracle):
    """`combo_parts_max_bytes: 0` forces every combination off the lazy
    device path onto the host-materialized fallback, on CUDA. Catches a
    fallback mask built at the wrong query height or uploaded transposed —
    which the CPU run of the same knob cannot see, because there the mask
    never crosses a device boundary."""
    from nova_bf import filters

    hits, present = _combo_run(ds, "gpu_budget", {"combo_parts_max_bytes": 0,
                                                  "dense_batch_size": 37})
    u = filters.combo_usage()
    assert u["budget_fallbacks"] > 0, u          # also true on CPU
    _shared_filtered(ds, oracle, present, hits, "[cuda] budget fallback")


# ============================================ the recorder itself (runs anywhere)


def test_reach_recorder_sees_a_path_that_ran(tmp_path_factory):
    """On CPU: a 3-doc multivector run with a token budget MUST execute the
    budgeted branch of `_ragged_batch_ranges`; the recorder must see it, and
    must NOT see a CUDA-only line. If this fails the GPU requirements above
    are meaningless."""
    from .test_parity_mv_edges import _build

    ds = _build(tmp_path_factory.mktemp("reach_self"))
    with reach.LineRecorder("self-test") as rec:
        layouts.run(ds, _mv_specs(), tag="reach_self", device="cpu",
                    params={"multivector_token_budget": 64})
    rec.require("compute.py", "token_cap_row = int(np.searchsorted(offsets, token_cap",
                "the budgeted range split must run under multivector_token_budget")
    rec.forbid("compute.py", "stage_tokens = max(", "CUDA-only ring on a CPU run")
    with pytest.raises(AssertionError, match="did NOT run"):
        rec.require("compute.py", "stage_tokens = max(")


def test_reach_rejects_a_stale_snippet():
    rec = reach.LineRecorder("stale")
    with pytest.raises(AssertionError, match="matches no line"):
        rec.hit("compute.py", "this text appears nowhere in compute.py 9f3a")


def test_every_required_snippet_exists():
    """Every `require` snippet in this file names a real line TODAY — checked
    on CPU, so a refactor that renames one is caught before the GPU run."""
    for f, s in [("multivector_kernels.py", "_fused_ragged_reduce_kernel["),
                 ("compute.py", "flat_gpu = torch.from_numpy(batch.flat_tokens).to(device)"),
                 ("compute.py", "stage_tokens = max("),
                 ("compute.py", "    _run_dense_prefetch(batch, ranges, device, process_slice)"),
                 ("compute.py", "    _run_dense_prefetch_pipelined(batch"),
                 ("compute.py", "token_cap_row = int(np.searchsorted(offsets, token_cap")]:
        assert reach.lines_matching(f, s), (f, s)
    from nova_bf import merge, tiebreak, tile_rescore
    assert hasattr(merge, "_DEVICE_FOLDS") and hasattr(tile_rescore, "_LAUNCHES")
    assert hasattr(tiebreak, "_gpu_perm") and hasattr(tiebreak, "_gpu_perm_from_lanes")
