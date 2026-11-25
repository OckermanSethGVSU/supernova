"""bench/prep/steps.py: the prep values, the download spec and stage_data."""

import os

import pytest

from bench import launch as LA
from bench import experiment as E
from bench.prep import steps as PS

from helpers.experiments import _exp


CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")


def test_prep_values_live_in_the_experiment_files_not_in_launch_py():
    """no data-source / prep tables in bench/launch.py or bench/prep/steps.py; the generated
    prep lines follow the YAML."""
    for gone in ("COYO_QSRC", "PES2O_QUERIES_SRC", "FILTERED_CORPUS_SRC", "FILTERED_QUERIES_SRC",
                 "STAGE_DATA", "DATA_SRC_DEFAULTS", "REALFILE_DOWNLOAD", "REALFILE_TEMPLATE",
                 "download_script"):
        assert not hasattr(LA, gone) and not hasattr(PS, gone), gone
    for f in (("launch.py",), ("prep", "steps.py")):
        src = open(os.path.join(CB, "bench", *f)).read()
        for value in ("queries_gt_k1000", "pes2o_sample", "fineweb_sample", "ms_marco_shared_queries",
                      "7001", "FineWeb-10B/resharded"):
            assert value not in src, (f, value)
    e = _exp("dense_coyo", ["prep.queries.args=[s3://other/q.parquet, ~/data]"])
    assert "prep_queries_coyo.py s3://other/q.parquet ~/data )" in "\n".join(PS.node_prep(e))
    e = _exp("sparse_pubmed", ["prep.queries_src=s3://x/q.parquet"])
    assert "s3://x/q.parquet" in "\n".join(PS.node_prep(e))
    assert [d["key"] for d in PS.data_plan(_exp("filtered_structured"), {})] == ["FILTERED_QUERIES_SRC"]


def test_a_filtered_experiment_always_builds_the_text_matcher():
    """nova-bf's native text matcher is installed before every filtered run (no switch)."""
    sets = ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16"]
    for name in ("filtered_structured", "filtered_text"):
        assert "nova_textscan" in "\n".join(LA.build_lines(_exp(name, sets)))
    assert "nova_textscan" not in "\n".join(LA.build_lines(_exp("dense_coyo")))
    for gone in ("replicate", "download", "textscan"):
        with pytest.raises(E.ExperimentError, match="prep"):
            _exp("dense_coyo", [f"prep.{gone}=1"])


def test_stage_data_writes_the_full_source_sha256(tmp_path):
    import hashlib
    from bench.prep import steps as PS
    src = tmp_path / "s.parquet"
    src.write_bytes(b"abc")
    lines = PS.stage_data(str(tmp_path / "stage"), [{"key": "K", "path": "data/s.parquet",
                                                    "source": str(src), "how": "copy"}])
    assert lines[0].endswith(f" sha256 {hashlib.sha256(b'abc').hexdigest()}")


def test_a_dense_experiment_with_queries_src_copies_it_to_queries_path():
    """a dense experiment with prep.queries_src and no prep.queries script copies it like sparse's."""
    e = _exp("dense_pubmed", ["prep.queries=null", "prep.queries_src=s3://x/q.parquet"])
    lines = "\n".join(PS.node_prep(e))
    assert 'log "STEP queries"' in lines and "pyarrow.fs" in lines
    assert lines.endswith(f'log "queries: s3://x/q.parquet -> {e["queries"]["path"]}"')


def test_pubmed_prep_writes_the_queries_and_a_gt_keyed_by_snippet_id(tmp_path):
    """the PubMed GT uses hit_snippet_ids (the corpus `id`), not the PMIDs in hit_ids."""
    import subprocess
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    src = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "query_id": ["a", "b"], "query": ["t1", "t2"],
        "bge_m3_dense_embedding": [[1.0, 0.0], [0.0, 1.0]],
        "hit_ids": [["111", "222"], ["333", "111"]],
        "hit_snippet_ids": [["s_1", "s_2"], ["s_3", "s_1"]],
        "hit_scores": pa.array([[0.9, 0.8], [0.7, 0.6]], pa.list_(pa.float32()))}), src)
    subprocess.run([sys.executable, os.path.join(CB, "bench", "prep", "prep_queries_pubmed.py"),
                    str(src), str(tmp_path / "out"), "dense"], check=True)
    q = pq.read_table(tmp_path / "out" / "queries_pubmed_dense.parquet")
    gt = pq.read_table(tmp_path / "out" / "gt_pubmed_dense.parquet")
    assert q.column_names == ["query_id", "bge_m3_dense_embedding"]
    assert gt.column_names == ["query_id", "hit_ids", "hit_scores"]
    assert gt.column("hit_ids").to_pylist() == [["s_1", "s_2"], ["s_3", "s_1"]]
    e = _exp("dense_pubmed")
    lines = "\n".join(PS.node_prep(e))
    assert "prep_queries_pubmed.py" in lines and lines.rstrip().endswith("progress.log")
    assert e["checks"]["gt"].endswith("gt_pubmed_dense.parquet")
