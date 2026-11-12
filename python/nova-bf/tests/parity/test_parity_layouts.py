"""Dense storage dtypes, file layouts, read paths, point ids and tie-breaking.

The shared parity corpus is float32, one row group per file, four
similar-sized files, ids read from a column. Production corpora are not
always that, and every difference lands on a different code path:

STORAGE (`STORAGE` datasets)
  * float16 — kept as float16 all the way to the device and widened there
    (`DenseCorpusBatch.transfer`); the truth is the float16-ROUNDED vector
    (`layouts.as_stored`), so a reader that silently scored something else
    (the unrounded float32, a mis-widened half) is a wrong answer here;
  * float64 — narrowed to float32 at decode (`io.dense_to_2d`);
  * `fixed_size_list<float32>` instead of `list<float32>`;
  * a MIXED corpus (float16 / float32 / float64 files) whose small files are
    coalesced into shared batches (`compute._concat_dense_batches`, which must
    WIDEN to the common type — a batch that took the first file's float16
    would round every float32 row coalesced after it).

LAYOUT
  * row groups of 7 and 17 rows, so every file is a CHUNKED column (the
    per-chunk copy loop in `dense_to_2d`, and the native reader's per-row-
    group plan);
  * many small files including 0-row and 1-row shards (global row numbering
    and the coalescing buffer must skip an empty file without shifting);
  * both READ PATHS: the native Rust reader (`rust_fetch_dense`/`_columns`,
    on by default) and pyarrow (`rust_fetch_*: false`) — a spy proves which one
    actually decoded the vectors, since the native path silently falls back
    to pyarrow for layouts it does not support;
  * a DICTIONARY-encoded payload column, filtered through both read paths.

IDS AND TIES
  * `corpus.id_column` unset: every hit id is nova-load's `vf_point_id`
    (md5 of `<file>:<row>` as a UUID), re-derived here from its definition with
    `hashlib` and mapped back to the row — including across a 0-row file and
    an `include`/`exclude` file selection;
  * exact score TIES across files (small-integer vectors, so every fp32 dot
    and euclidean score is exact and ties are bit-for-bit), under
    `tiebreak: ordinal` and `tiebreak: id`, with cross-file coalescing, and
    with `NOVA_BF_NO_PRUNE` on and off — asserted as EXACT id sequences, not to
    tolerance, against an oracle that applies the tie rule itself;
  * a two-pass run over a float16 corpus (the two-pass is the default dense
    path on CUDA and reads `exact_fp16`).

NON-FINITE
  * NaN corpus rows: an input error naming the file, column and rows.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

import nova_bf.io as io_mod

from . import layouts, naive

DIM = 64
N_Q = 6
K = 7
KALL = 500
DENSE_METRICS = ("dot", "cosine", "euclidean")


# ---------------------------------------------------------------- builders


def _dense_ds(tmp, name, files, *, seed, row_group_size=None, n_q=N_Q,
              dict_payload=False, integer=False, dup_pool=None, nan_rows=(),
              file_names=None, id_fmt="{}"):
    """`files` = [(n_rows, storage)] with storage in float16 / float32 /
    float64 / fsl32. `integer` draws small-integer vectors (exact fp32
    arithmetic), `dup_pool` draws every corpus row from a pool of that many
    base vectors (exact ties)."""
    rng = np.random.default_rng(seed)
    pool = (rng.integers(-3, 4, size=(dup_pool, DIM)).astype(np.float64)
            if dup_pool else None)
    tables, docs, g = [], [], 0
    for n, storage in files:
        vecs, grp = [], []
        for r in range(n):
            if pool is not None:
                v = pool[int(rng.integers(0, len(pool)))]
            elif integer:
                v = rng.integers(-3, 4, size=DIM).astype(np.float64)
            else:
                v = rng.standard_normal(DIM)
            if g in nan_rows:
                v = v.copy(); v[5] = float("nan")
            vecs.append(v)
            grp.append("abc"[g % 3])
            dt = {"fsl32": "float32"}.get(storage, storage)
            docs.append(naive.Doc(row=g, payload={"grp": grp[-1]},
                                  dense=None if g in nan_rows else layouts.as_stored(v, dt)))
            g += 1
        pa_type = {"float16": pa.float16(), "float32": pa.float32(),
                   "float64": pa.float64(), "fsl32": pa.float32()}[storage]
        grp_arr = pa.array(grp, type=pa.string())
        if dict_payload:
            grp_arr = grp_arr.dictionary_encode()
        tables.append(pa.table({
            "dense_embedding": layouts.dense_array(vecs, pa_type, fixed=(storage == "fsl32"))
            if n else pa.array([], type=pa.list_(pa_type, DIM) if storage == "fsl32"
                               else pa.list_(pa_type)),
            "grp": grp_arr,
            "rid": pa.array([id_fmt.format(g - n + i) for i in range(n)], type=pa.string()),
        }))
    qv = (rng.integers(-3, 4, size=(n_q, DIM)).astype(np.float32) if (integer or pool is not None)
          else rng.standard_normal((n_q, DIM)).astype(np.float32))
    queries = [{"dense": [float(x) for x in q], "payload": {"q_grp": "abc"[i % 3]}}
               for i, q in enumerate(qv)]
    qt = pa.table({"dense_embedding": layouts.dense_array(qv),
                   "qid": pa.array([str(i) for i in range(n_q)]),
                   "q_grp": pa.array([q["payload"]["q_grp"] for q in queries])})
    return layouts.build(tmp, name, tables, docs, qt, queries,
                         row_group_size=row_group_size, file_names=file_names)


STORAGE = {
    "f16": dict(files=[(41, "float16"), (0, "float16"), (1, "float16"), (57, "float16"),
                       (33, "float16")], row_group_size=7, seed=1),
    "f64": dict(files=[(29, "float64"), (1, "float64"), (44, "float64")],
                row_group_size=17, seed=2),
    "fsl": dict(files=[(37, "fsl32"), (26, "fsl32")], row_group_size=17, seed=3),
    "mixed": dict(files=[(13, "float16"), (9, "float32"), (11, "float64"), (7, "float16"),
                         (12, "float32"), (8, "float64")], seed=4),
    "many": dict(files=[(5, "float32"), (0, "float32"), (1, "float32"), (8, "float32"),
                        (3, "float32"), (0, "float32"), (11, "float32"), (2, "float32"),
                        (1, "float32"), (9, "float32"), (4, "float32"), (16, "float32")],
                 row_group_size=7, seed=5, dict_payload=True),
}


@pytest.fixture(scope="module")
def storage_ds(tmp_path_factory):
    return {name: _dense_ds(tmp_path_factory.mktemp(f"lay_{name}"), name, **kw)
            for name, kw in STORAGE.items()}


STATIC = {"must": [{"field": "grp", "match": ["a", "c"]}]}
PERQ = {"must": [{"field": "grp", "match_from_query": "q_grp"}]}
READ_PATHS = {"native": {}, "pyarrow": {"rust_fetch_dense": False, "rust_fetch_columns": False}}


def _keep(regime):
    if regime == "static":
        return lambda d, q: d.payload["grp"] in ("a", "c")
    if regime == "perquery":
        return lambda d, q: d.payload["grp"] == q["payload"]["q_grp"]
    return None


def _specs(regime):
    f = {"static": STATIC, "perquery": PERQ}.get(regime)
    return [layouts.spec(f"d_{m}{sfx}", metric=m, k=k, filter=f)
            for m in DENSE_METRICS for sfx, k in (("", K), ("_all", KALL))]


@pytest.fixture(scope="module")
def storage_runs(storage_ds, device):
    """Every storage dataset x regime x read path, with a spy recording which
    reader decoded each file's vectors (`native` = the Rust reader returned an
    array; anything else fell back to pyarrow's `dense_to_2d`)."""
    out = {}
    mp = pytest.MonkeyPatch()
    log = []
    real_native = io_mod.Store._read_dense_native
    real_d2 = io_mod.dense_to_2d

    def native_spy(self, *a, **kw):
        r = real_native(self, *a, **kw)
        log.append("native" if r is not None else "native_declined")
        return r

    def d2_spy(col):
        log.append("pyarrow")
        return real_d2(col)

    mp.setattr(io_mod.Store, "_read_dense_native", native_spy)
    mp.setattr(io_mod, "dense_to_2d", d2_spy)
    import nova_bf.compute as compute_mod
    if getattr(compute_mod, "dense_to_2d", None) is real_d2:
        mp.setattr(compute_mod, "dense_to_2d", d2_spy)
    try:
        for name, ds in storage_ds.items():
            for regime in ("nofilter", "static", "perquery"):
                for rp, rp_params in READ_PATHS.items():
                    del log[:]
                    hits = layouts.run(ds, _specs(regime), tag=f"st_{regime}_{rp}",
                                       device=device,
                                       params={"dense_batch_size": 20, **rp_params})
                    out[(name, regime, rp)] = (hits, list(log))
    finally:
        mp.undo()
    return out


def test_layouts_are_what_they_claim(storage_ds):
    for path in storage_ds["f16"].files + storage_ds["many"].files:
        if pq.ParquetFile(path).metadata.num_rows > 7:
            assert layouts.row_groups(path) > 1, path
    assert 0 in storage_ds["many"].file_rows and 1 in storage_ds["many"].file_rows
    t = pq.read_table(storage_ds["many"].files[0])
    assert pa.types.is_dictionary(t.schema.field("grp").type)
    assert pa.types.is_fixed_size_list(pq.read_schema(storage_ds["fsl"].files[0])
                                       .field("dense_embedding").type)


@pytest.mark.parametrize("name", list(STORAGE))
@pytest.mark.parametrize("rp", list(READ_PATHS))
def test_the_read_path_spy_saw_the_path_it_was_asked_for(name, rp, storage_runs):
    """`pyarrow` must never touch the native reader; `native` must use it for
    exactly the layouts it supports (see below)."""
    for regime in ("nofilter", "static", "perquery"):
        _, log = storage_runs[(name, regime, rp)]
        print(f"[read-path] {name}/{regime}/{rp}: native={log.count('native')} "
              f"declined={log.count('native_declined')} pyarrow={log.count('pyarrow')}")
        if rp == "pyarrow":
            assert "native" not in log and "native_declined" not in log, log
            assert log.count("pyarrow") >= 1, log
        else:
            # Pinned (measured): the native reader decodes float16, float32 and
            # fixed_size_list files — row groups and all — and DECLINES float64,
            # which then goes through pyarrow. One declined call per float64
            # file; every other file natively. A layout silently moving to the
            # fallback (still correct, so no oracle test would notice) is what
            # this catches.
            n_f64 = sum(1 for _, st in STORAGE[name]["files"] if st == "float64")
            n_files = len(STORAGE[name]["files"])
            assert log.count("native_declined") == n_f64, (name, regime, log)
            assert log.count("native") == n_files - n_f64, (name, regime, log)


CELLS = [(n, r, rp, m, sfx) for n in STORAGE for r in ("nofilter", "static", "perquery")
         for rp in READ_PATHS for m in DENSE_METRICS for sfx in ("", "_all")]


@pytest.mark.parametrize("name,regime,rp,metric,sfx", CELLS,
                         ids=[f"{a}-{b}-{c}-{d}{e}" for a, b, c, d, e in CELLS])
def test_storage_layouts_agree_with_the_naive_oracle(name, regime, rp, metric, sfx,
                                                     storage_ds, storage_runs, device):
    """Against the AS-STORED truth: float16 files are scored on their
    float16-rounded values, float64 on their float32 narrowing."""
    ds = storage_ds[name]
    hits, _ = storage_runs[(name, regime, rp)]
    want = layouts.naive_topk(ds, vector_type="dense", metric=metric,
                              k=K if sfx == "" else KALL, keep=_keep(regime))
    layouts.assert_all(hits[f"d_{metric}{sfx}"], want, metric=metric,
                       label=f"[{device}] {name}/{regime}/{rp}/{metric}{sfx} vs naive")


def test_float16_rounding_is_visible_at_this_tolerance(storage_ds, storage_runs):
    """The f16 comparison above is only evidence that the float16 values were
    scored if scoring the UNROUNDED float32 originals would have FAILED it.
    Re-derive the originals (same seed) and check that they disagree with what
    nova-bf returned by more than the cosine tolerance."""
    from . import compare

    ds = storage_ds["f16"]
    rng = np.random.default_rng(STORAGE["f16"]["seed"])
    raw = [rng.standard_normal(DIM) for _ in range(ds.n_docs)]
    hits, _ = storage_runs[("f16", "nofilter", "native")]
    for metric in ("dot", "cosine"):
        worst = 0.0
        for qi, q in enumerate(ds.queries):
            for row, s in hits[f"d_{metric}_all"][qi]:
                exact = naive.dense_score(q["dense"], list(raw[row]), metric)
                worst = max(worst, abs(exact - s) / (1 + abs(exact)))
        assert worst > 2 * compare.TOL[metric], (
            f"float16 rounding moved {metric} by at most {worst:.2e}, under 2x the "
            f"tolerance — the f16 test could not tell rounded from unrounded")


# ----------------------------------------------------------------- qdrant


@pytest.fixture(scope="module")
def storage_collections(client, collection, storage_ds):
    from qdrant_client import models as M

    out = {}
    for name in ("f16", "mixed", "many"):
        out[name] = layouts.qdrant_collection(
            client, storage_ds[name],
            dense={"dot": (M.Distance.DOT, DIM), "cosine": (M.Distance.COSINE, DIM),
                   "euclidean": (M.Distance.EUCLID, DIM)},
            payload_index={"grp": M.PayloadSchemaType.KEYWORD})
    yield out
    for c in out.values():
        client.delete_collection(c)


QCELLS = [(n, r, m) for n in ("f16", "mixed", "many") for r in ("nofilter", "static")
          for m in DENSE_METRICS]


@pytest.mark.qdrant
@pytest.mark.parametrize("name,regime,metric", QCELLS,
                         ids=[f"{a}-{b}-{c}" for a, b, c in QCELLS])
def test_storage_layouts_agree_with_qdrant(name, regime, metric, storage_ds, storage_runs,
                                           client, storage_collections, device):
    """Qdrant is given the AS-STORED values (float16-rounded), so this checks
    nova-bf scored what the file holds, the way the engine would."""
    from qdrant_client import models as M

    ds = storage_ds[name]
    qf = (M.Filter(must=[M.FieldCondition(key="grp", match=M.MatchAny(any=["a", "c"]))])
          if regime == "static" else None)
    want = layouts.qdrant_topk(client, storage_collections[name], ds, using=metric,
                               vector_type="dense", k=K, qfilter=qf,
                               sign=-1.0 if metric == "euclidean" else 1.0)
    for rp in READ_PATHS:
        hits, _ = storage_runs[(name, regime, rp)]
        layouts.assert_all(hits[f"d_{metric}"], want, metric=metric,
                           label=f"[{device}] {name}/{regime}/{rp}/{metric} vs qdrant")


# --------------------------------------------------------- two-pass on fp16


@pytest.fixture
def forced_twopass(monkeypatch):
    """The same forcing `test_parity_twopass._force` documents (slice mode, no
    pipelining), re-stated here because that helper is private to its file."""
    from nova_bf import twopass

    monkeypatch.setenv("NOVA_BF_PIPELINE_PASS_ONE", "0")
    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "slice")
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 2)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 2)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.delenv("NOVA_BF_TWOPASS_NO_VERIFY", raising=False)
    twopass.reset()
    yield twopass
    twopass.reset()


@pytest.mark.parametrize("storage", ["float16", "float32"])
def test_two_pass_over_a_float16_corpus_agrees_with_the_naive_oracle(
        storage, forced_twopass, tmp_path_factory, device):
    """The two-pass reads `exact_fp16` to decide whether the corpus side
    pays the half-conversion term of its bound; a float16 corpus is the case
    that skips it. Asserts pruning really happened (a two-pass that pruned
    nothing agrees with every oracle vacuously)."""
    ds = _dense_ds(tmp_path_factory.mktemp(f"tp16_{storage}"), f"tp_{storage}",
                   [(97, storage), (61, storage), (79, storage)], seed=7, n_q=26)
    for m in ("dot", "cosine"):
        # One metric per run: a batch scored by two metrics shares one Gram
        # product (`share_gram`) and the two-pass declines it.
        forced_twopass.reset()
        hits = layouts.run(ds, [layouts.spec(f"t_{m}", metric=m, k=5)],
                           tag=f"tp16_{storage}_{m}", device=device,
                           params={"dense_batch_size": 8, "two_pass_rescore": "slice"})
        st = forced_twopass.stats()
        assert st["slices_twopass"] > 0 and st["rows_full"] > st["rows_live"] \
            and st["gemms"] > 0, {k: v for k, v in st.items() if not k.startswith("audit")}
        want = layouts.naive_topk(ds, vector_type="dense", metric=m, k=5)
        layouts.assert_all(hits[f"t_{m}"], want, metric=m,
                           label=f"[{device}] two-pass {storage}/{m} vs naive")


# ------------------------------------------------------------- md5 point ids


def _ids_ds(tmp):
    return _dense_ds(tmp, "ids", [(13, "float32"), (0, "float32"), (1, "float32"),
                                  (21, "float32"), (9, "float32")], seed=8)


@pytest.fixture(scope="module")
def ids_ds(tmp_path_factory):
    return _ids_ds(tmp_path_factory.mktemp("lay_ids"))


def _md5_to_row(ds, files=None):
    """expected point id -> global row, for the files a run actually read.
    The id is per-FILE (`<path>:<row-in-file>`) and stable under any file
    selection; the global row is the oracle's key."""
    out, g = {}, 0
    for path, n in zip(ds.files, ds.file_rows):
        if files is None or path in files:
            for r in range(n):
                out[layouts.expected_point_id(path, r)] = g + r
        g += n
    return out


SELECTIONS = {
    "all": ({}, None),
    # include first, then exclude (config.py) — drop the 21-row file
    "exclude": ({"exclude": "part-003"}, {0, 1, 2, 4}),
    "include": ({"include": r"part-00[034]"}, {0, 3, 4}),
}


@pytest.mark.parametrize("sel", list(SELECTIONS))
@pytest.mark.parametrize("vt", ["dense", "sparse"])
def test_default_point_ids_are_nova_loads_vf_point_id(sel, vt, ids_ds, tmp_path_factory, device):
    """`corpus.id_column` unset. Every hit id must be exactly
    `uuid(md5("<abs path>:<row in file>"))` for the row that scored it — the
    contract that makes a GT hit id equal to the Qdrant point id nova-load
    wrote. The rows are recovered from the ids and the ranking is then held to
    the oracle, so a correct-looking UUID for the WRONG row fails too."""
    ds = ids_ds
    if vt == "sparse":
        ds = _sparse_twin(ds, tmp_path_factory)
    extra, keep_files = SELECTIONS[sel]
    files = None if keep_files is None else {ds.files[i] for i in keep_files}
    id_map = _md5_to_row(ds, files)
    hits = layouts.run(ds, [layouts.spec("ids", vector_type=vt, metric="dot", k=KALL)],
                       tag=f"ids_{vt}_{sel}", device=device,
                       corpus={"id_column": None, **extra}, raw_ids=True)
    docs = ds.docs if files is None else [d for d in ds.docs if ds.file_of(d.row)[0] in files]
    want = layouts.naive_topk(ds, vector_type=vt, metric="dot", k=KALL, docs=docs)
    for qi, lst in hits["ids"].items():
        unknown = [i for i, _ in lst if i not in id_map]
        assert not unknown, (f"[{device}] {vt}/{sel} q{qi}: ids not equal to any "
                             f"md5('<file>:<row>') of the selected files: {unknown[:3]}")
        got = [(id_map[i], s) for i, s in lst]
        layouts.assert_topk(got, want[qi], metric="dot",
                            label=f"[{device}] md5 ids {vt}/{sel} q{qi}")


def _sparse_twin(ds, tmp_path_factory):
    """The ids corpus again, same files/row counts, with a sparse column."""
    rng = np.random.default_rng(81)
    tables, docs, g = [], [], 0
    for n in ds.file_rows:
        rows = []
        for _ in range(n):
            idx = rng.choice(30, size=4, replace=False).tolist()
            rows.append((idx, rng.uniform(0.1, 2.0, size=4).tolist()))
            docs.append(naive.Doc(row=g, sparse=naive.sparse_from_pairs(*rows[-1])))
            g += 1
        tables.append(pa.table({"sparse_embedding": layouts.sparse_array(rows),
                                "rid": pa.array([str(g - n + i) for i in range(n)])}))
    qs = [(rng.choice(30, size=6, replace=False).tolist(), [1.0] * 6) for _ in range(N_Q)]
    return layouts.build(tmp_path_factory.mktemp("lay_ids_sp"), "ids_sp", tables, docs,
                         pa.table({"sparse_embedding": layouts.sparse_array(qs),
                                   "qid": pa.array([str(i) for i in range(N_Q)])}),
                         [{"sparse": naive.sparse_from_pairs(*q), "payload": {}} for q in qs])


def test_the_md5_reference_is_the_documented_one():
    """The reference itself, against a hand-computed value, so a typo in
    `layouts.expected_point_id` cannot make the id test agree with a wrong
    nova-bf (both being wrong the same way is exactly what an independent
    reference exists to rule out)."""
    import hashlib

    h = hashlib.md5(b"/data/corpus/part-000.parquet:17").hexdigest()
    assert layouts.expected_point_id("/data/corpus/part-000.parquet", 17) == (
        f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}")
    assert len(layouts.expected_point_id("x", 0)) == 36


# ------------------------------------------------------- exact ties, tiebreak


def _tie_ds(tmp):
    # 9 files of 4..9 rows, every row one of 6 base integer vectors: each
    # query's ranking is a handful of score LEVELS with many rows tied on each.
    return _dense_ds(tmp, "ties", [(6, "float32"), (4, "float32"), (0, "float32"),
                                   (9, "float32"), (5, "float32"), (7, "float32"),
                                   (1, "float32"), (8, "float32"), (6, "float32")],
                     seed=9, dup_pool=6)


@pytest.fixture(scope="module")
def tie_ds(tmp_path_factory):
    return _tie_ds(tmp_path_factory.mktemp("lay_ties"))


def _exact_oracle(ds, metric, k, tiebreak, keep=None, id_fmt="{}"):
    """Exact ranking with the tie rule applied: scores are exact (integer
    vectors, fp32-exact sums; euclidean's sqrt of an exact integer is one
    correctly-rounded value), so equality is meaningful here."""
    out = {}
    for qi, q in enumerate(ds.queries):
        rows = []
        for d in ds.docs:
            if keep is not None and not keep(d, q):
                continue
            if metric == "dot":
                s = sum(a * b for a, b in zip(q["dense"], d.dense))
            else:
                s = -math.sqrt(sum((a - b) ** 2 for a, b in zip(q["dense"], d.dense)))
            rows.append((d.row, float(np.float32(s))))
        key = (lambda rs: (-rs[1], rs[0])) if tiebreak == "ordinal" else \
              (lambda rs: (-rs[1], id_fmt.format(rs[0])))  # string ids: LEXICOGRAPHIC
        rows.sort(key=key)
        out[qi] = rows[:k]
    return out


TIE_CELLS = [(tb, reg, prune, m) for tb in ("ordinal", "id") for reg in ("nofilter", "static")
             for prune in ("prune", "no_prune") for m in ("dot", "euclidean")]


@pytest.mark.parametrize("tiebreak,regime,prune,metric", TIE_CELLS,
                         ids=[f"{a}-{b}-{c}-{d}" for a, b, c, d in TIE_CELLS])
def test_exact_ties_follow_the_tiebreak_rule(tiebreak, regime, prune, metric, tie_ds, device):
    """Bit-exact ties straddling every K boundary, the files small enough that
    a statically-filtered run COALESCES several into one batch (the coalesced
    ordinal column is rebuilt per file in `_flush_coalesce_group`). The answer
    is an exact id sequence; nothing is left to tolerance."""
    keep = _keep(regime)
    f = STATIC if regime == "static" else None
    specs = [layouts.spec(f"t{k}", metric=metric, k=k, filter=f) for k in (3, 7, 13)]
    hits = layouts.run(tie_ds, specs, tag=f"ties_{tiebreak}_{regime}_{prune}_{metric}",
                       device=device, params={"dense_batch_size": 10, "tiebreak": tiebreak},
                       env={"NOVA_BF_NO_PRUNE": "1" if prune == "no_prune" else None})
    for k in (3, 7, 13):
        want = _exact_oracle(tie_ds, metric, k, tiebreak, keep)
        for qi in want:
            got = hits[f"t{k}"][qi]
            assert [r for r, _ in got] == [r for r, _ in want[qi]], (
                f"[{device}] {tiebreak}/{regime}/{prune}/{metric} k={k} q{qi}\n"
                f"  got:  {got}\n  want: {want[qi]}")
            for (_, a), (_, b) in zip(got, want[qi]):
                assert a == pytest.approx(b, abs=1e-4), (got, want[qi])


SHARD_TIE_KS = (1, 2, 4, 7, 13)
SHARD_TIE_CELLS = [(tb, reg, m, w) for tb in ("ordinal", "id") for reg in ("nofilter", "static")
                   for m in ("dot", "euclidean") for w in (2, 3, 4)]


@pytest.mark.parametrize("tiebreak,regime,metric,num_jobs", SHARD_TIE_CELLS,
                         ids=[f"{a}-{b}-{c}-W{d}" for a, b, c, d in SHARD_TIE_CELLS])
def test_exact_ties_survive_a_sharded_merge(tiebreak, regime, metric, num_jobs, tie_ds,
                                            device):
    """The same bit-exact ties, but SPLIT ACROSS RANKS: `compute --num-jobs W`
    for every rank, then `merge`. Tied rows now sit in different partials, so
    which of them survives is decided by the merge's own tie handling (its
    exact lexsort on (score, tie key) and the cutoff-ambiguity check that
    sends a query whose K-th score is tied past the kept rows down the exact
    path) — code the unsharded test above never reaches. k = 1 and 2 put the
    cutoff inside the very first tie level for most queries. The answer is an
    exact id sequence vs the exact oracle; nothing is left to tolerance."""
    from nova_bf.compute import run_compute
    from nova_bf.merge import run_merge

    from .runner import pinned_device, read_results

    f = STATIC if regime == "static" else None
    specs = [layouts.spec(f"t{k}", metric=metric, k=k, filter=f) for k in SHARD_TIE_KS]
    cfg = layouts.build_config(
        tie_ds, specs, tag=f"shties_{tiebreak}_{regime}_{metric}_W{num_jobs}_{device}",
        params={"dense_batch_size": 10, "tiebreak": tiebreak})
    with pinned_device(device):
        for rank in range(num_jobs):
            run_compute(cfg, num_jobs=num_jobs, job_rank=rank)
        hits = read_results(run_merge(cfg))
    keep = _keep(regime)
    for k in SHARD_TIE_KS:
        want = _exact_oracle(tie_ds, metric, k, tiebreak, keep)
        assert set(hits[f"t{k}"]) == set(want), (sorted(hits[f"t{k}"]), sorted(want))
        for qi in want:
            got = hits[f"t{k}"][qi]
            assert [r for r, _ in got] == [r for r, _ in want[qi]], (
                f"[{device}] W={num_jobs} {tiebreak}/{regime}/{metric} k={k} q{qi}\n"
                f"  got:  {got}\n  want: {want[qi]}")
            for (_, a), (_, b) in zip(got, want[qi]):
                assert a == pytest.approx(b, abs=1e-4), (got, want[qi])


def test_the_tie_fixture_ties_across_ranks(tie_ds):
    """The sharded tie test is only a test of the MERGE's tie rule if, at the
    k's it uses, rows tied on the K-th score live in files that land on
    different ranks — otherwise each rank's partial already settles the tie
    and the merge never has to."""
    file_of_row = {}
    for fi, n in enumerate(tie_ds.file_rows):
        base = sum(tie_ds.file_rows[:fi])
        for r in range(n):
            file_of_row[base + r] = fi
    for num_jobs in (2, 3, 4):
        cross = 0
        for metric in ("dot", "euclidean"):
            full = _exact_oracle(tie_ds, metric, 10 ** 6, "ordinal")
            for k in SHARD_TIE_KS:
                for qi in full:
                    kth = full[qi][k - 1][1]
                    ranks = {file_of_row[r] % num_jobs for r, s in full[qi] if s == kth}
                    cross += len(ranks) > 1
        assert cross >= 20, (num_jobs, cross)


def test_the_tie_fixture_really_ties_at_the_boundary(tie_ds):
    """At every k used above, the k-th and (k+1)-th oracle scores are EQUAL
    for most queries — otherwise the rule is never consulted."""
    n_tied = 0
    for metric in ("dot", "euclidean"):
        full = _exact_oracle(tie_ds, metric, 10 ** 6, "ordinal")
        for k in (3, 7, 13):
            n_tied += sum(1 for qi in full if full[qi][k - 1][1] == full[qi][k][1])
    assert n_tied >= 20, n_tied


# ------------------------------------------------------------ non-finite


@pytest.fixture(scope="module")
def nan_ds(tmp_path_factory):
    return _dense_ds(tmp_path_factory.mktemp("lay_nan"), "nan",
                     [(23, "float32"), (31, "float32")], seed=10,
                     nan_rows=(2, 9, 17, 30, 41))


# NaN rows per file (local row numbers): part-000 holds global rows 0..22,
# part-001 rows 23..53, so global 2, 9, 17 / 30, 41 are local 2, 9, 17 / 7, 18.
_NAN_LOCAL = {"part-000.parquet": "rows 2, 9, 17;", "part-001.parquet": "rows 7, 18;"}


@pytest.mark.parametrize("bs", [8, 64])
@pytest.mark.parametrize("metric", DENSE_METRICS)
def test_nan_rows_are_an_input_error(metric, bs, nan_ds, device):
    """NaN corpus vectors are rejected by the reader (they used to be dropped
    at write, and each NaN row in a dot/cosine top-K cost the query a real
    neighbour -- the strict xfail this replaces). The error names the file,
    the column and that file's bad rows, whichever file is read first."""
    with pytest.raises(RuntimeError) as e:
        layouts.run(nan_ds, [layouts.spec(f"n_{metric}", metric=metric, k=K)],
                    tag=f"nan_{bs}", device=device, params={"dense_batch_size": bs})
    msg = str(e.value)
    assert "column `dense_embedding`" in msg
    assert "NaN/inf vectors corrupt results; clean or drop these rows." in msg
    hit = [f for f in _NAN_LOCAL if f in msg]
    assert len(hit) == 1 and _NAN_LOCAL[hit[0]] in msg, msg
