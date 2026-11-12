"""End-to-end wiring for `params.multivector_prune`, on CPU.

The prune's ~130 lines of `compute.py` plumbing -- the config gate, the
vector-type gate, the `MultiVectorBatchSlice` gate, the `NOVA_BF_NO_PRUNE` kill
switch, the `score_cache` bypass and the decline path -- had NO executable
coverage on a machine without a GPU: every other test of this feature is
CUDA-gated. This file closes that, because the wiring is not CUDA-specific even
though the kernels are.

WHAT CPU CAN AND CANNOT SHOW. Without a GPU the float16 pass declines (the
two-pass certification refuses a non-CUDA device), so these cannot exercise the
bound or the kernels. What they CAN prove is the contract the whole design
rests on: **declining is invisible.** A run with the prune enabled must produce
byte-identical output to one with it off, must not raise, and must record the
decline rather than silently scoring something different. A wiring bug that
routed the prune into a dense search, or that let a declined prune leave
`scores` as `None`, or that shared pruned scores through `score_cache`, would
show up here as a changed answer or a crash.
"""
from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from nova_bf import compute as cm
from nova_bf import mv_fp16
from nova_bf.compute import run_compute
from nova_bf.config import load_config

DIM = 8


def _mv_array(docs):
    """`list<list<float32>>` from a list of (n_tokens, DIM) arrays."""
    toks = [t for d in docs for t in d]
    inner = pa.ListArray.from_arrays(
        pa.array(list(range(0, len(toks) * DIM + 1, DIM)), type=pa.int32()),
        pa.array(np.concatenate(toks).astype(np.float32) if toks
                 else np.zeros(0, np.float32), type=pa.float32()),
    )
    outer, n = [0], 0
    for d in docs:
        n += len(d)
        outer.append(n)
    return pa.ListArray.from_arrays(pa.array(outer, type=pa.int32()), inner)


@pytest.fixture
def corpus_and_queries(tmp_path):
    rng = np.random.default_rng(5)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # A zero-token document and a single-token one, because those are the rows
    # the prune's -inf handling is most likely to get wrong.
    docs = [rng.standard_normal((n, DIM)) for n in (3, 1, 5, 2)]
    docs.insert(1, np.zeros((0, DIM)))
    pq.write_table(
        pa.table({"id": pa.array([f"d{i}" for i in range(len(docs))]),
                  "multivector_embedding": _mv_array(docs)}),
        cdir / "c.parquet")
    queries = [rng.standard_normal((n, DIM)) for n in (2, 4, 1)]
    qpath = tmp_path / "q.parquet"
    pq.write_table(
        pa.table({"qid": pa.array([f"q{i}" for i in range(len(queries))]),
                  "multivector_embedding": _mv_array(queries)}),
        qpath)
    return cdir, qpath


def _run(tmp_path, cdir, qpath, prune, *, k=3, tag="r"):
    out = tmp_path / f"out_{tag}"
    out.mkdir(exist_ok=True)
    cfg = tmp_path / f"cfg_{tag}.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: {prune}
  # 0.0 pins the gate open. These fixtures are a handful of slices, and the
  # gate needs three consecutive good batch-group probes to engage — on a run
  # this short it would never latch, and every test here would pass vacuously
  # by never reaching the prune at all.
  multivector_min_prune_rate: 0.0
searches:
  - name: mv
    k: {k}
    metric: dot
    vector_type: multivector
""")
    cm._MV_PRUNE.update(pairs=0, pruned=0, calls=0, declined=0, sec=0.0)
    t = pq.read_table(run_compute(load_config(str(cfg)))["mv"]).to_pydict()
    return ({q: (tuple(h), tuple(round(float(x), 6) for x in s))
             for q, h, s in zip(t["query_id"], t["hit_ids"], t["hit_scores"])},
            dict(cm._MV_PRUNE))


def test_enabling_the_prune_does_not_change_the_answer(tmp_path,
                                                       corpus_and_queries):
    """The whole contract in one assertion. On CPU the prune declines, so this
    is really "declining is invisible" -- but that is exactly the property a
    wiring bug would break, and it holds on a GPU too because the bound is
    admissible."""
    cdir, qpath = corpus_and_queries
    off, _ = _run(tmp_path, cdir, qpath, "off", tag="off")
    on, _ = _run(tmp_path, cdir, qpath, "fp16", tag="on")
    assert on == off


def test_the_prune_is_reached_and_declines_on_cpu(tmp_path, corpus_and_queries):
    """Guard against the test above passing because the prune was never wired
    in at all: it must actually be CALLED, and must decline rather than run."""
    cdir, qpath = corpus_and_queries
    _, tally = _run(tmp_path, cdir, qpath, "fp16", tag="called")
    assert tally["calls"] > 0, "the prune was never reached from run_compute"
    assert tally["declined"] == tally["calls"], (
        "the prune claims to have run on a CPU-only machine")
    assert tally["pruned"] == 0


def test_off_never_reaches_the_prune(tmp_path, corpus_and_queries):
    cdir, qpath = corpus_and_queries
    _, tally = _run(tmp_path, cdir, qpath, "off", tag="never")
    assert tally["calls"] == 0


def test_the_kill_switch_disables_it(tmp_path, corpus_and_queries, monkeypatch):
    """`NOVA_BF_NO_PRUNE` is a pre-existing escape hatch; the multivector prune
    is gated on it at the call site and must honour it."""
    cdir, qpath = corpus_and_queries
    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    _, tally = _run(tmp_path, cdir, qpath, "fp16", tag="killed")
    assert tally["calls"] == 0


def test_a_dense_search_never_reaches_the_multivector_prune(tmp_path):
    """The state is forwarded only for `vt == "multivector"`. A dense run with
    the flag set must not touch it -- and must not crash on the attempt."""
    rng = np.random.default_rng(1)
    cdir = tmp_path / "dense"
    cdir.mkdir()
    pq.write_table(pa.table({
        "id": pa.array([f"d{i}" for i in range(6)]),
        "embedding": pa.array(list(rng.standard_normal((6, DIM))),
                              type=pa.list_(pa.float32(), DIM)),
    }), cdir / "c.parquet")
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "qid": pa.array(["q0", "q1"]),
        "embedding": pa.array(list(rng.standard_normal((2, DIM))),
                              type=pa.list_(pa.float32(), DIM)),
    }), qpath)
    out = tmp_path / "out"
    out.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  dense_column: embedding
  id_column: id
queries:
  path: {qpath}
  dense_column: embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: fp16
searches:
  - name: d
    k: 2
    metric: dot
    vector_type: dense
""")
    cm._MV_PRUNE.update(pairs=0, pruned=0, calls=0, declined=0, sec=0.0)
    run_compute(load_config(str(cfg)))
    assert cm._MV_PRUNE["calls"] == 0


def test_two_searches_sharing_a_metric_still_agree(tmp_path,
                                                   corpus_and_queries):
    """Pruned scores are member-specific (they depend on that member's running
    threshold) and must bypass `score_cache`. Two members over the same metric
    must each get their own correct answer, not one leaking into the other."""
    cdir, qpath = corpus_and_queries
    out = tmp_path / "out_two"
    out.mkdir()
    cfg = tmp_path / "cfg_two.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: fp16
searches:
  - name: a
    k: 2
    metric: dot
    vector_type: multivector
  - name: b
    k: 4
    metric: dot
    vector_type: multivector
""")
    res = run_compute(load_config(str(cfg)))
    a = pq.read_table(res["a"]).to_pydict()
    b = pq.read_table(res["b"]).to_pydict()
    # `a` asked for 2 and `b` for 4; `a`'s hits must be `b`'s first two.
    for ha, hb in zip(a["hit_ids"], b["hit_ids"]):
        assert list(ha) == list(hb)[:len(ha)]


def test_the_prune_is_on_by_default():
    """It is opt-OUT: a config that says nothing about it gets the prune.
    Safe because it declines wherever it cannot run, but it is a default that
    changes tie ordering, so pin it."""
    from nova_bf.config import ParamsConfig

    assert ParamsConfig().multivector_prune == "fp16"


def test_the_default_reaches_the_prune_without_being_asked(tmp_path,
                                                           corpus_and_queries):
    """The default must actually plumb through, not just parse."""
    cdir, qpath = corpus_and_queries
    out = tmp_path / "out_default"
    out.mkdir()
    cfg = tmp_path / "cfg_default.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
searches:
  - name: mv
    k: 3
    metric: dot
    vector_type: multivector
""")
    cm._MV_PRUNE.update(pairs=0, pruned=0, calls=0, declined=0, sec=0.0)
    mv_fp16.reset_gate_stats()
    run_compute(load_config(str(cfg)))

    # The GATE is what proves the plumbing here, not `calls`. This config
    # deliberately sets no `multivector_min_prune_rate`, so the default 0.70
    # floor applies, and a fixture this short never gets the three consecutive
    # good probes needed to engage — `calls` would be 0 whether the default
    # plumbed through or not. `closed` counts slices that REACHED the gate and
    # were turned away, which separates "not wired" from "wired but gated".
    assert mv_fp16.gate_stats()["closed"] > 0, (
        "the default did not reach the prune")


def test_the_setting_is_recorded_in_the_output_metadata(tmp_path,
                                                        corpus_and_queries):
    """A ground-truth file must say whether it was produced with the prune:
    two files that disagree only in tie order should be able to explain why."""
    cdir, qpath = corpus_and_queries
    for mode in ("off", "fp16"):
        out = tmp_path / f"out_meta_{mode}"
        out.mkdir()
        cfg = tmp_path / f"cfg_meta_{mode}.yaml"
        cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: {mode}
searches:
  - name: mv
    k: 3
    metric: dot
    vector_type: multivector
""")
        res = run_compute(load_config(str(cfg)))
        meta = pq.read_schema(res["mv"]).metadata
        assert meta[b"nova_bf.multivector_prune"] == mode.encode()


def test_bare_off_in_yaml_is_accepted():
    """YAML 1.1 parses a bare `off` as the boolean False, so the documented
    default spelling would otherwise be a ValidationError whose message says
    `input_value=False` and never mentions quoting. Found by writing the
    obvious config in the tests above."""
    from nova_bf.config import ParamsConfig

    assert ParamsConfig(multivector_prune=False).multivector_prune == "off"
    assert ParamsConfig(multivector_prune="off").multivector_prune == "off"
    assert ParamsConfig(multivector_prune="fp16").multivector_prune == "fp16"


def test_bare_on_in_yaml_is_still_rejected():
    """`on` is NOT mapped: there could be more than one way to enable this, so
    silently choosing one would be guessing at intent."""
    import pydantic

    from nova_bf.config import ParamsConfig

    with pytest.raises(pydantic.ValidationError):
        ParamsConfig(multivector_prune=True)


def test_the_probe_wiring_can_actually_engage_the_gate():
    """The seam nothing else covers: `_mv_note_unpruned_rate` measuring a
    slice that was scored WITHOUT the prune, and the gate acting on it.

    Every fixture above pins `multivector_min_prune_rate: 0.0`, which starts
    the gate latched — so the probe path and the latch transition are never
    exercised through a real config. `tests/test_mv_prune_gate.py` feeds
    `Fp16State` plain integers and never touches this function. Without this
    test, nothing verifies that a run using the SHIPPED default can ever
    engage at all, and the probe and the floor now come from two different
    estimators."""
    torch = pytest.importorskip("torch")

    from nova_bf.compute import _mv_note_unpruned_rate
    from nova_bf.mv_fp16 import Fp16State
    from nova_bf.tiebreak import pack

    class _Q:
        n_q = 4

    st = Fp16State(min_prune_rate=0.70)
    assert not st.gate_open(), "the default floor must start closed"

    # 4 queries x 5 documents, thresholds above every score: all prunable.
    scores = torch.zeros(4, 5)
    thr = pack(torch.full((4,), 1.0), torch.zeros(4, dtype=torch.int64))

    for i in range(Fp16State._LATCH_AFTER):
        assert st.wants_probe(), "a closed gate must still want measuring"
        assert _mv_note_unpruned_rate(st, scores, _Q(), thr, None), (
            f"probe {i} reported nothing measurable")
    assert st.gate_open(), "three good probes did not engage the prune"
    assert not st.wants_probe(), "a latched gate must stop paying for probes"


def test_the_probe_ignores_rows_the_member_does_not_own():
    """Unowned rows carry `-inf` thresholds, and `x >= -inf` is True — so
    counting them made an unowned row look permanently live. A member owning
    20% of the rows measured 0.20 even with every one of ITS pairs prunable,
    and could never clear the 0.70 floor."""
    torch = pytest.importorskip("torch")

    from nova_bf.compute import _mv_note_unpruned_rate
    from nova_bf.mv_fp16 import Fp16State
    from nova_bf.tiebreak import pack

    class _Q:
        n_q = 10

    st = Fp16State(min_prune_rate=0.70)
    scores = torch.zeros(10, 5)
    thr = pack(torch.full((2,), 1.0), torch.zeros(2, dtype=torch.int64))

    for _ in range(Fp16State._LATCH_AFTER):
        assert _mv_note_unpruned_rate(st, scores, _Q(), thr, slice(0, 2))
    assert st.gate_open(), (
        "a member owning 2 of 10 rows could not engage despite pruning all "
        "of its own pairs")


# --- the prune's statistics in the run manifest ----------------------------
#
# `params.kernels.multivector_prune` copies the same host-side counters
# `_log_mv_prune_summary` logs. These pin that the manifest and the log agree,
# that the block exists only for a run with a multivector search, and (via a
# CPU stand-in for the float16 pass) that pruned pairs and the audit reach it.


def _mv_manifest_cfg(tmp_path, cdir, qpath, prune, tag, *, k=3, extra=""):
    out = tmp_path / f"out_{tag}"
    out.mkdir(exist_ok=True)
    cfg = tmp_path / f"cfg_{tag}.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: {prune}
  multivector_min_prune_rate: 0.0
{extra}
searches:
  - name: mv
    k: {k}
    metric: dot
    vector_type: multivector
""")
    return load_config(str(cfg)), out


def _manifest(out):
    import json

    return json.loads(next(out.rglob("*manifest*.json")).read_text())


def _summary_line(caplog):
    import re

    for r in caplog.records:
        m = re.match(r"mv-prune calls=(\d+) declined=(\d+) pruned=([0-9.]+) "
                     r"\((\d+)/(\d+) pairs\)", r.getMessage())
        if m:
            return m
    return None


def test_the_prune_statistics_reach_the_manifest_on_cpu(
        tmp_path, many_docs, caplog, monkeypatch):
    """On CPU the prune declines every call (no fused float16 kernel); the
    manifest must say so with the same numbers the log line reports."""
    import logging

    monkeypatch.delenv("NOVA_BF_MV_PRUNE_AUDIT", raising=False)
    monkeypatch.delenv("NOVA_BF_PRUNE_TIMING", raising=False)
    cdir, qpath = many_docs
    cfg, out = _mv_manifest_cfg(tmp_path, cdir, qpath, "fp16", "mf_cpu", k=2)
    with caplog.at_level(logging.INFO, logger="nova_bf"):
        run_compute(cfg)
    mv = _manifest(out)["params"]["kernels"]["multivector_prune"]

    line = _summary_line(caplog)
    assert line is not None, "the summary log line is gone"
    assert (mv["calls"], mv["declined"], mv["pruned"], mv["pairs"]) == tuple(
        int(line.group(i)) for i in (1, 2, 4, 5))
    assert mv["calls"] == cm._MV_PRUNE["calls"] > 0
    assert mv["declined"] == mv["calls"] and mv["launches"] == 0
    assert mv["pairs"] == mv["pruned"] == 0 and mv["pruned_fraction"] is None
    assert mv["policy"] == "fp16" and mv["permitted"] is True
    assert mv["uncertified"] is None, mv
    assert mv["gate"] == mv_fp16.gate_stats()
    assert mv["timing_syncs"] is False and "seconds" not in mv
    assert mv["audit"] == {"enabled": False}


def test_off_records_the_block_as_not_permitted(tmp_path, corpus_and_queries):
    cdir, qpath = corpus_and_queries
    cfg, out = _mv_manifest_cfg(tmp_path, cdir, qpath, "off", "mf_off")
    run_compute(cfg)
    mv = _manifest(out)["params"]["kernels"]["multivector_prune"]
    assert mv["policy"] == "off" and mv["permitted"] is False
    assert mv["calls"] == mv["launches"] == mv["pairs"] == 0
    assert mv["uncertified"] is None and mv["audit"] == {"enabled": False}


def test_a_certification_refusal_is_named(tmp_path, many_docs, monkeypatch):
    monkeypatch.setattr(mv_fp16, "certify", lambda dim, device: "test refusal")
    cdir, qpath = many_docs
    cfg, out = _mv_manifest_cfg(tmp_path, cdir, qpath, "fp16", "mf_cert", k=2)
    run_compute(cfg)
    mv = _manifest(out)["params"]["kernels"]["multivector_prune"]
    assert mv["uncertified"] == "dim=8 on cpu: test refusal", mv
    assert mv["declined"] == mv["calls"] > 0


def test_a_run_without_a_multivector_search_has_no_block(tmp_path):
    rng = np.random.default_rng(2)
    cdir = tmp_path / "dense"
    cdir.mkdir()
    pq.write_table(pa.table({
        "id": pa.array([f"d{i}" for i in range(6)]),
        "embedding": pa.array(list(rng.standard_normal((6, DIM))),
                              type=pa.list_(pa.float32(), DIM)),
    }), cdir / "c.parquet")
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "qid": pa.array(["q0", "q1"]),
        "embedding": pa.array(list(rng.standard_normal((2, DIM))),
                              type=pa.list_(pa.float32(), DIM)),
    }), qpath)
    out = tmp_path / "out"
    out.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  dense_column: embedding
  id_column: id
queries:
  path: {qpath}
  dense_column: embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
searches:
  - name: d
    k: 2
    metric: dot
    vector_type: dense
""")
    run_compute(load_config(str(cfg)))
    kernels = _manifest(out)["params"]["kernels"]
    assert "multivector_prune" not in kernels, kernels.keys()


def _cpu_exact_prune(self, q_flat, c_flat, q_offsets, doc_offsets, thresholds):
    """CPU stand-in for `Fp16State.score`: the bound IS the exact MaxSim, so
    every prune is correct and the audit must grade it clean. Exercises the
    counting, the audit hook and the manifest without the float16 kernels."""
    import torch

    q_off = mv_fp16._as_offsets(q_offsets, c_flat.device).cpu().tolist()
    d_off = mv_fp16._as_offsets(doc_offsets, c_flat.device).cpu().tolist()
    sim = q_flat @ c_flat.T
    exact = torch.full((len(q_off) - 1, len(d_off) - 1), float("-inf"))
    for i in range(len(q_off) - 1):
        for j in range(len(d_off) - 1):
            if d_off[j + 1] > d_off[j]:
                blk = sim[q_off[i]:q_off[i + 1], d_off[j]:d_off[j + 1]]
                exact[i, j] = blk.max(dim=1).values.sum()
    upper = exact.clone()
    dead = mv_fp16.prune_mask(upper, thresholds)
    scores = torch.where(dead, torch.full_like(exact, float("-inf")), exact)
    return scores, dead, (upper if mv_fp16._audit_should_grade() else None)


@pytest.fixture
def many_docs(tmp_path):
    """Several files of documents, so later slices prune against thresholds
    the earlier ones set."""
    rng = np.random.default_rng(11)
    cdir = tmp_path / "corpus_many"
    cdir.mkdir()
    for f in range(4):
        docs = [rng.standard_normal((int(n), DIM))
                for n in rng.integers(1, 5, size=12)]
        pq.write_table(
            pa.table({"id": pa.array([f"f{f}d{i}" for i in range(len(docs))]),
                      "multivector_embedding": _mv_array(docs)}),
            cdir / f"c{f}.parquet")
    queries = [rng.standard_normal((n, DIM)) for n in (2, 3, 1, 4)]
    qpath = tmp_path / "q_many.parquet"
    pq.write_table(
        pa.table({"qid": pa.array([f"q{i}" for i in range(len(queries))]),
                  "multivector_embedding": _mv_array(queries)}),
        qpath)
    return cdir, qpath


def test_pruned_pairs_and_the_audit_reach_the_manifest(
        tmp_path, many_docs, caplog, monkeypatch):
    import logging

    cdir, qpath = many_docs
    monkeypatch.setattr(mv_fp16.Fp16State, "score", _cpu_exact_prune)
    monkeypatch.setenv("NOVA_BF_MV_PRUNE_AUDIT", "1")
    monkeypatch.setenv("NOVA_BF_PRUNE_TIMING", "1")
    mv_fp16.reset_audit()
    try:
        cfg, out = _mv_manifest_cfg(tmp_path, cdir, qpath, "fp16", "mf_audit",
                                    k=2)
        with caplog.at_level(logging.INFO, logger="nova_bf"):
            run_compute(cfg)
        mv = _manifest(out)["params"]["kernels"]["multivector_prune"]

        line = _summary_line(caplog)
        assert line is not None
        assert (mv["calls"], mv["declined"], mv["pruned"], mv["pairs"]) == tuple(
            int(line.group(i)) for i in (1, 2, 4, 5))
        assert mv["pruned"] > 0, "nothing pruned, so this proves nothing"
        assert mv["launches"] == mv["calls"] - mv["declined"] > 0
        assert mv["pruned_fraction"] == round(mv["pruned"] / mv["pairs"], 6)
        assert f"{mv['pruned_fraction']:.4f}" == line.group(3)
        assert mv["uncertified"] is None and mv["disabled_by_audit"] is None
        assert mv["timing_syncs"] is True
        assert mv["seconds"] == round(cm._MV_PRUNE["sec"], 3)

        audit = mv["audit"]
        assert audit["enabled"] is True and audit["rate"] == 1
        assert {k: v for k, v in audit.items()
                if k not in ("enabled", "rate")} == mv_fp16.audit_stats()
        assert audit["graded_slices"] == audit["offered"] > 0
        assert audit["correct_prune"] > 0
        assert audit["false_prune"] == audit["bound_violations"] == 0
        assert audit["score_mismatch"] == 0 and audit["disabled"] is None
        assert any(r.getMessage().startswith(
            f"mv-prune AUDIT slices={audit['graded_slices']}/{audit['offered']}")
            for r in caplog.records)
    finally:
        mv_fp16.reset_audit()
