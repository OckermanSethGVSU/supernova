"""Input validation: inputs nova-bf cannot grade correctly are rejected with
an error that says what is wrong, where (file / field / row / search), and
why, instead of crashing deep in a reader or scorer.

  1. a static `match: []` (rejected at config load);
  2. a filtered corpus field stored as a list column (first file at start-up,
     every later file when it is read);
  3. a filtered corpus field absent from a file;
  4. sparse rows whose `indices` and `values` lists differ in length (corpus
     and queries, pyarrow and rust_fetch download paths);
  5. NaN/inf in a dense vector, a sparse value or a multivector token
     (corpus, both read paths, and queries).

Errors raised inside a reader thread surface as the run's RuntimeError, whose
text repeats the cause.

Each has a valid-input control next to it. All CPU, tiny parquet files.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("torch")

from nova_bf.compute import run_compute
from nova_bf.config import (BruteForceConfig, CorpusConfig, Filter, FilterCondition,
                            OutputConfig, ParamsConfig, QueriesConfig, SearchSpec,
                            load_config)
from nova_bf.io import sparse_to_coo_parts

DENSE = pa.list_(pa.float32())
SPARSE = pa.struct([pa.field("indices", pa.list_(pa.uint32())),
                    pa.field("values", pa.list_(pa.float32()))])


def _dense_rows(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 4)).astype(np.float32)


def _corpus_file(path, n, seed=0, **cols):
    data = {"dense_embedding": pa.array(_dense_rows(n, seed).tolist(), DENSE),
            "id": pa.array([f"{path.stem}-{i}" for i in range(n)])}
    data.update(cols)
    pq.write_table(pa.table(data), str(path))


def _queries(path, n=2, **cols):
    data = {"dense_embedding": pa.array(_dense_rows(n, 99).tolist(), DENSE),
            "qid": pa.array([f"q{i}" for i in range(n)])}
    data.update(cols)
    pq.write_table(pa.table(data), str(path))


def _cfg(tmp_path, filt=None, *, vector_type="dense", rust_fetch=True, k=3, **corpus_kw):
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(tmp_path / "c"), id_column="id", **corpus_kw),
        queries=QueriesConfig(path=str(tmp_path / "q.parquet"), id_column="qid"),
        output=OutputConfig(path=str(tmp_path / "out")),
        params=ParamsConfig(io_workers=1, rust_fetch_dense=rust_fetch,
                            rust_fetch_columns=rust_fetch),
        searches=[SearchSpec(name="srch", vector_type=vector_type, metric="dot", k=k,
                             filter=filt)])


def _hits(path):
    t = pq.read_table(path).to_pydict()
    return dict(zip(t["query_id"], t["hit_ids"]))


# ----------------------------------------------------------- 1. static match: []

@pytest.mark.parametrize("group", ["must", "should", "must_not"])
def test_static_empty_match_rejected_at_config_load(tmp_path, group):
    filt = Filter(**{group: [FilterCondition(field="lang", match="eng"),
                             FilterCondition(field="lang", match=[])]})
    with pytest.raises(ValueError) as e:
        _cfg(tmp_path, filt)
    msg = str(e.value)
    assert "search `srch`" in msg and f"filter.{group}[1]" in msg
    assert "`lang`" in msg and "match: []" in msg and "Qdrant" in msg
    assert "at least one value" in msg


def test_static_empty_match_rejected_by_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "corpus: {path: /c}\nqueries: {path: /q}\noutput: {path: /o}\n"
        "searches:\n  - filter:\n      must:\n        - {field: tag, match: []}\n")
    with pytest.raises(ValueError, match=r"search `dense_cosine_filtered`.*`tag`.*match: \[\]"):
        load_config(str(p))


def test_nonempty_static_match_and_empty_match_from_query_still_run(tmp_path):
    """Controls: a one-value MatchAny runs as before, and an EMPTY per-query
    list (match_from_query) stays legal and matches nothing for that query."""
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 6, lang=pa.array(["eng", "fra"] * 3))
    q = pa.array([["eng"], []], pa.list_(pa.string()))
    _queries(tmp_path / "q.parquet", ql=q)
    out = run_compute(_cfg(tmp_path, Filter(must=[FilterCondition(field="lang", match=["eng"])])))
    hits = _hits(out["srch"])
    assert all(h.endswith(("-0", "-2", "-4")) for v in hits.values() for h in v)
    out = run_compute(_cfg(tmp_path, Filter(must=[FilterCondition(field="lang",
                                                                  match_from_query="ql")])))
    hits = _hits(out["srch"])
    assert len(hits["q0"]) == 3 and hits["q1"] == []


# ------------------------------------------------- 2. list-valued filter fields

LIST_FILTERS = {
    "match": Filter(must=[FilterCondition(field="tags", match="x")]),
    "range": Filter(must=[FilterCondition(field="tags", range={"gte": 1})]),
    "match_text": Filter(must=[FilterCondition(field="tags", match_text="x")]),
    "match_from_query": Filter(must=[FilterCondition(field="tags", match_from_query="qt")]),
}


def _list_col(kind, n):
    if kind in ("range",):
        return pa.array([[i, i + 1] for i in range(n)], pa.list_(pa.int64()))
    return pa.array([["x", "y"]] * n, pa.list_(pa.string()))


def _scalar_col(kind, n):
    if kind == "range":
        return pa.array(list(range(n)), pa.int64())
    return pa.array(["x"] * n)


@pytest.mark.parametrize("kind", list(LIST_FILTERS))
@pytest.mark.parametrize("list_type", ["list", "large_list", "fixed_size_list"])
def test_list_column_in_first_file_rejected_at_start(tmp_path, kind, list_type):
    (tmp_path / "c").mkdir()
    col = _list_col(kind, 4)
    t = {"list": pa.list_, "large_list": pa.large_list,
         "fixed_size_list": lambda v: pa.list_(v, 2)}[list_type](col.type.value_type)
    _corpus_file(tmp_path / "c" / "a.parquet", 4, tags=col.cast(t))
    _queries(tmp_path / "q.parquet", qt=pa.array(["x", "y"]))
    with pytest.raises(ValueError) as e:
        run_compute(_cfg(tmp_path, LIST_FILTERS[kind]))
    msg = str(e.value)
    assert "corpus file " in msg and "a.parquet" in msg and "filter field `tags`" in msg
    assert "is a list column" in msg and "list-valued payload fields are not supported" in msg
    assert "ANY element" in msg and "'srch'" in msg


@pytest.mark.parametrize("rust_fetch", [True, False])
@pytest.mark.parametrize("kind", ["match", "range"])
def test_list_column_in_a_later_file_rejected_when_read(tmp_path, kind, rust_fetch):
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 4, tags=_scalar_col(kind, 4))
    _corpus_file(tmp_path / "c" / "b.parquet", 4, seed=1, tags=_list_col(kind, 4))
    _queries(tmp_path / "q.parquet")
    with pytest.raises(RuntimeError) as e:   # reader error, cause in the text
        run_compute(_cfg(tmp_path, LIST_FILTERS[kind], rust_fetch=rust_fetch))
    msg = str(e.value)
    assert "corpus file " in msg and "b.parquet" in msg and "`tags`" in msg and "is a list column" in msg


def test_unfiltered_list_columns_are_fine(tmp_path):
    """Control: a list column nobody filters on is never inspected."""
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 4, tags=_list_col("match", 4),
                 lang=pa.array(["eng"] * 4))
    _queries(tmp_path / "q.parquet")
    out = run_compute(_cfg(tmp_path, Filter(must=[FilterCondition(field="lang", match="eng")])))
    assert all(len(v) == 3 for v in _hits(out["srch"]).values())


# ----------------------------------------------------- 3. missing filter field

@pytest.mark.parametrize("rust_fetch", [True, False])
def test_field_missing_from_a_later_file(tmp_path, rust_fetch):
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 4, lang=pa.array(["eng"] * 4))
    _corpus_file(tmp_path / "c" / "b.parquet", 4, seed=1, langs=pa.array(["eng"] * 4))
    _queries(tmp_path / "q.parquet")
    with pytest.raises(RuntimeError) as e:   # reader error, cause in the text
        run_compute(_cfg(tmp_path, Filter(must=[FilterCondition(field="lang", match="eng")]),
                         rust_fetch=rust_fetch))
    msg = str(e.value)
    assert "corpus file " in msg and "b.parquet" in msg and "`lang`" in msg and "'srch'" in msg
    assert "absent" in msg and "not in this file" in msg
    assert "did you mean `langs`" in msg and "Columns present" in msg and "'id'" in msg


def test_field_missing_from_the_first_file_fails_before_the_scan(tmp_path, monkeypatch):
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 4, other=pa.array([1, 2, 3, 4]))
    _queries(tmp_path / "q.parquet")
    from nova_bf.io import Store

    orig = Store.read_columns
    corpus_dir = str(tmp_path / "c")

    def no_corpus_reads(self, read_path, *a, **k):
        if read_path.startswith(corpus_dir):
            raise AssertionError("corpus data read before the start-up check")
        return orig(self, read_path, *a, **k)
    monkeypatch.setattr(Store, "read_columns", no_corpus_reads)
    monkeypatch.setattr(Store, "read_columns_fetched", lambda *a, **k: None)
    with pytest.raises(ValueError) as e:
        run_compute(_cfg(tmp_path, Filter(must_not=[FilterCondition(field="lang",
                                                                     match_text="x")])))
    msg = str(e.value)
    assert "corpus file " in msg and "a.parquet" in msg and "`lang`" in msg and "absent" in msg
    assert "did you mean" not in msg      # nothing close to `lang`


def test_unrelated_read_errors_pass_through(tmp_path, monkeypatch):
    """A read failure that is NOT a missing filter field is re-raised as is."""
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 4, lang=pa.array(["eng"] * 4))
    _queries(tmp_path / "q.parquet")
    from nova_bf.io import Store

    def boom(*a, **k):
        raise OSError("simulated transport failure")
    monkeypatch.setattr(Store, "read_columns", boom)
    monkeypatch.setattr(Store, "read_columns_fetched", lambda *a, **k: None)
    with pytest.raises(OSError, match="simulated transport failure"):
        run_compute(_cfg(tmp_path, Filter(must=[FilterCondition(field="lang", match="eng")]),
                         rust_fetch=False))


# ---------------------------------------- 4. sparse indices/values length mismatch

def _sparse(rows):
    """rows: list of (indices, values) with possibly different lengths."""
    idx = pa.array([r[0] for r in rows], pa.list_(pa.uint32()))
    val = pa.array([r[1] for r in rows], pa.list_(pa.float32()))
    return pa.StructArray.from_arrays([idx, val], fields=list(SPARSE))


GOOD = [([1, 2], [1.0, 2.0]), ([3], [0.5]), ([1, 3], [0.25, 4.0]), ([2], [1.0])]


def test_sparse_lengths_unit_error_names_rows():
    bad = GOOD[:1] + [([3, 4], [0.5])] + GOOD[2:] + [([1], [1.0, 2.0, 3.0])]
    with pytest.raises(ValueError) as e:
        sparse_to_coo_parts(pa.chunked_array([_sparse(bad)]), source="corpus file x.parquet")
    msg = str(e.value)
    assert msg.startswith("corpus file x.parquet:")
    assert "2 row(s)" in msg and "row 1: 2 indices vs 1 values" in msg
    assert "row 4: 1 indices vs 3 values" in msg


def test_sparse_lengths_unit_valid_and_rebased_offsets():
    ref = sparse_to_coo_parts(pa.chunked_array([_sparse(GOOD)]))
    # Same lengths, but the values child starts at a different physical
    # offset than the indices child: decoded as the same CSR.
    idx = pa.array([[9, 9, 9]] + [r[0] for r in GOOD], pa.list_(pa.uint32())).slice(1)
    val = pa.array([r[1] for r in GOOD], pa.list_(pa.float32()))
    s = pa.StructArray.from_arrays([idx, val], fields=list(SPARSE))
    got = sparse_to_coo_parts(pa.chunked_array([s]))
    for a, b in zip(ref, got):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert ref[0].tolist() == [0, 2, 3, 5, 6]


def _sparse_ds(tmp_path, corpus_rows_by_file, query_rows):
    (tmp_path / "c").mkdir()
    for name, rows in corpus_rows_by_file.items():
        pq.write_table(pa.table({"sparse_embedding": _sparse(rows),
                                 "id": pa.array([f"{name}-{i}" for i in range(len(rows))])}),
                       str(tmp_path / "c" / f"{name}.parquet"))
    pq.write_table(pa.table({"sparse_embedding": _sparse(query_rows),
                             "qid": pa.array([f"q{i}" for i in range(len(query_rows))])}),
                   str(tmp_path / "q.parquet"))


@pytest.mark.parametrize("rust_fetch", [True, False])
def test_sparse_corpus_mismatch_names_file_and_row(tmp_path, rust_fetch):
    _sparse_ds(tmp_path, {"a": GOOD, "b": GOOD[:2] + [([1, 2, 3], [1.0, 1.0])]}, GOOD[:2])
    with pytest.raises(RuntimeError) as e:   # reader error, cause in the text
        run_compute(_cfg(tmp_path, vector_type="sparse", rust_fetch=rust_fetch))
    msg = str(e.value)
    assert "corpus file " in msg and "b.parquet" in msg and "corpus.sparse_column `sparse_embedding`" in msg
    assert "row 2: 3 indices vs 2 values" in msg


def test_sparse_query_mismatch_names_file_and_row(tmp_path):
    _sparse_ds(tmp_path, {"a": GOOD}, [GOOD[0], ([1], [])])
    with pytest.raises(ValueError) as e:
        run_compute(_cfg(tmp_path, vector_type="sparse"))
    msg = str(e.value)
    assert "queries file " in msg and "q.parquet" in msg and "row 1: 1 indices vs 0 values" in msg


def test_sparse_valid_input_scores_as_before(tmp_path):
    """Control: valid sparse input gives the hand-computed dot top-k."""
    _sparse_ds(tmp_path, {"a": GOOD}, [([1, 3], [1.0, 1.0])])
    out = run_compute(_cfg(tmp_path, vector_type="sparse", k=4))
    t = pq.read_table(out["srch"]).to_pydict()
    # scores: a-0: 1.0, a-1: 0.5, a-2: 4.25, a-3: 0 overlap -> none
    assert t["hit_ids"][0] == ["a-2", "a-0", "a-1"]
    assert np.allclose(t["hit_scores"][0], [4.25, 1.0, 0.5])


# ------------------------------------------------- 5. non-finite vector values

NONFINITE = "NaN/inf vectors corrupt results; clean or drop these rows."
MV = pa.list_(pa.list_(pa.float32()))


@pytest.mark.parametrize("rust_fetch", [True, False])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_dense_corpus_nonfinite_names_file_rows_and_count(tmp_path, rust_fetch, bad, caplog):
    caplog.set_level("INFO", logger="nova_bf")
    (tmp_path / "c").mkdir()
    _corpus_file(tmp_path / "c" / "a.parquet", 5)
    v = _dense_rows(12, 1)
    for r in (1, 3, 4, 6, 8, 10, 11):
        v[r, r % 4] = bad
    pq.write_table(pa.table({"dense_embedding": pa.array(v.tolist(), DENSE),
                             "id": pa.array([f"b-{i}" for i in range(12)])}),
                   str(tmp_path / "c" / "b.parquet"))
    _queries(tmp_path / "q.parquet")
    with pytest.raises(RuntimeError) as e:
        run_compute(_cfg(tmp_path, rust_fetch=rust_fetch))
    # the native (rust_fetch) dense decode really ran when enabled
    assert ("rust_fetch read path" in caplog.text) == rust_fetch
    assert "fell back to pyarrow" not in caplog.text
    msg = str(e.value)
    assert "corpus file " in msg and "b.parquet" in msg and "column `dense_embedding`" in msg
    assert "7 row(s)" in msg and "rows 1, 3, 4, 6, 8, ... (7 in total);" in msg
    assert NONFINITE in msg and "Qdrant" not in msg


def test_float64_overflow_to_inf_is_caught(tmp_path):
    """A float64 value too large for float32 decodes to inf -- also caught."""
    (tmp_path / "c").mkdir()
    v = _dense_rows(4).astype(np.float64)
    v[2, 0] = 1e300
    pq.write_table(pa.table({"dense_embedding": pa.array(v.tolist(), pa.list_(pa.float64())),
                             "id": pa.array(list("abcd"))}), str(tmp_path / "c" / "a.parquet"))
    _queries(tmp_path / "q.parquet")
    with pytest.raises(RuntimeError, match=r"rows 2;.*NaN/inf vectors corrupt"):
        run_compute(_cfg(tmp_path))


def test_multivector_corpus_nonfinite_names_document_rows(tmp_path):
    (tmp_path / "c").mkdir()
    docs = [[[1.0, 0.0]], [[0.5, 0.5], [0.0, 1.0], [1.0, 1.0]], [[3.0, 3.0]],
            [[2.0, 1.0], [0.0, 0.0]]]
    docs[1][2][1] = float("nan")          # third token of document 1
    docs[3][1][0] = float("inf")          # second token of document 3
    pq.write_table(pa.table({"multivector_embedding": pa.array(docs, MV),
                             "id": pa.array(list("abcd"))}), str(tmp_path / "c" / "a.parquet"))
    pq.write_table(pa.table({"multivector_embedding": pa.array([[[1.0, 0.0]]], MV),
                             "qid": pa.array(["q0"])}), str(tmp_path / "q.parquet"))
    with pytest.raises(RuntimeError) as e:
        run_compute(_cfg(tmp_path, vector_type="multivector"))
    msg = str(e.value)
    assert "a.parquet" in msg and "column `multivector_embedding`" in msg
    assert "2 row(s)" in msg and "rows 1, 3;" in msg and NONFINITE in msg


@pytest.mark.parametrize("vector_type", ["dense", "sparse", "multivector"])
def test_nonfinite_queries_rejected_at_load(tmp_path, vector_type):
    if vector_type == "dense":
        (tmp_path / "c").mkdir()
        _corpus_file(tmp_path / "c" / "a.parquet", 4)
        q = _dense_rows(4, 3)
        q[2, 1] = float("nan")
        pq.write_table(pa.table({"dense_embedding": pa.array(q.tolist(), DENSE),
                                 "qid": pa.array(list("wxyz"))}), str(tmp_path / "q.parquet"))
        col = "dense_embedding"
    elif vector_type == "sparse":
        _sparse_ds(tmp_path, {"a": GOOD}, [GOOD[0], GOOD[1], ([1], [float("inf")]), GOOD[3]])
        col = "sparse_embedding.values"
    else:
        (tmp_path / "c").mkdir()
        pq.write_table(pa.table({"multivector_embedding": pa.array([[[1.0, 0.0]]] * 3, MV),
                                 "id": pa.array(list("abc"))}), str(tmp_path / "c" / "a.parquet"))
        pq.write_table(pa.table({
            "multivector_embedding": pa.array(
                [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0], [float("nan"), 0.0]]], MV),
            "qid": pa.array(list("xyz"))}), str(tmp_path / "q.parquet"))
        col = "multivector_embedding"
    with pytest.raises(ValueError) as e:
        run_compute(_cfg(tmp_path, vector_type=vector_type))
    msg = str(e.value)
    assert "queries file " in msg and "q.parquet" in msg and f"column `{col}`" in msg
    assert "1 row(s)" in msg and "rows 2;" in msg and NONFINITE in msg


def test_check_finite_chunking_counts_each_row_once(monkeypatch):
    """Tiny chunks: rows spanning a chunk boundary, several bad elements per
    row, bad data only in the last chunk -- each row still counted once."""
    import nova_bf.io as io_mod

    monkeypatch.setattr(io_mod, "_FINITE_CHUNK_ELEMS", 3)
    vals = np.ones(20, np.float32)
    offsets = np.array([0, 4, 4, 9, 13, 20])          # row 1 is empty
    vals[[5, 6, 8, 19]] = np.nan                      # rows 2, 2, 2, 4
    with pytest.raises(ValueError, match=r"has 2 row\(s\).*rows 2, 4;"):
        io_mod.check_finite(vals, "f", "c", row_offsets=offsets)
    dense = np.ones((9, 5), np.float16)
    dense[8, 4] = np.inf
    with pytest.raises(ValueError, match=r"has 1 row\(s\).*rows 8;"):
        io_mod.check_finite(dense, "f", "c")
    # float16 takes the exponent-bit fast path: NaN, +inf and -inf, in the
    # last chunk only, and the largest finite half (65504) is not flagged.
    for bad in (np.nan, np.inf, -np.inf):
        h = np.full(23, 65504, np.float16)
        h[[0, 1]] = -65504
        h[[17, 22]] = bad
        with pytest.raises(ValueError, match=r"has 2 row\(s\).*rows 3, 4;"):
            io_mod.check_finite(h, "f", "c", row_offsets=np.array([0, 5, 10, 15, 20, 23]))
    io_mod.check_finite(np.full((9, 5), 65504, np.float16), "f", "c")
    io_mod.check_finite(np.full((7, 3), 6e-8, np.float16)[:, ::2], "f", "c")  # non-contiguous
    nc = np.ones((7, 4), np.float16)
    nc[5, 2] = np.nan
    with pytest.raises(ValueError, match=r"rows 5;"):
        io_mod.check_finite(nc[:, ::2], "f", "c")
    io_mod.check_finite(np.ones((9, 5), np.float32), "f", "c")      # clean: no error
    io_mod.check_finite(np.zeros((0, 0), np.float32), "f", "c")
    io_mod.check_finite(np.arange(5), "f", "c")                      # ints: skipped
