"""bench/prep/steps.py: the prep values, the download spec and stage_data."""

import os

import pytest

from bench import launch as LA
from bench.prep import steps as PS

from helpers.experiments import _exp


CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "experiments")


def test_prep_values_live_in_the_experiment_files_not_in_launch_py():
    """no data-source / prep tables in bench/launch.py or bench/prep/steps.py; the generated
    prep lines follow the YAML."""
    for gone in ("COYO_QSRC", "PES2O_QUERIES_SRC", "FILTERED_CORPUS_SRC", "FILTERED_QUERIES_SRC",
                 "STAGE_DATA", "DATA_SRC_DEFAULTS", "REALFILE_DOWNLOAD"):
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
    e = _exp("filtered_fineweb_smoke", ["prep.replicate.copies=2"])
    assert "seq 0 1" in "\n".join(PS.node_prep(e))
    e = _exp("filtered_fineweb_smoke", ["prep.textscan=true"])
    assert "nova_textscan" in "\n".join(LA.build_lines(e))
    assert [d["key"] for d in PS.data_plan(_exp("filtered_realfile"), {})] == \
        ["FILTERED_CORPUS_SRC", "FILTERED_QUERIES_SRC"]


def test_the_download_spec_is_pinned():
    e = _exp("filtered_realfile")
    assert "boto3" in PS.download_script(e["prep"]["download"])
    e = _exp("filtered_realfile", ["prep.download.per_dir=2"])
    with pytest.raises(PS.PrepError, match="pins"):
        PS.node_prep(e)


def test_stage_data_writes_the_full_source_sha256(tmp_path):
    import hashlib
    from bench.prep import steps as PS
    src = tmp_path / "s.parquet"
    src.write_bytes(b"abc")
    lines = PS.stage_data(str(tmp_path / "stage"), [{"key": "K", "path": "data/s.parquet",
                                                    "source": str(src), "how": "copy"}])
    assert lines[0].endswith(f" sha256 {hashlib.sha256(b'abc').hexdigest()}")


def test_a_dense_experiment_with_queries_src_copies_it_to_queries_path():
    """dense_pubmed has no prep.queries script: its prep.queries_src is copied like sparse's."""
    e = _exp("dense_pubmed")
    assert not e["prep"].get("queries") and e["prep"]["queries_src"]
    lines = "\n".join(PS.node_prep(e))
    assert 'log "STEP queries"' in lines and "pyarrow.fs" in lines
    assert lines.endswith(f'log "queries: {e["prep"]["queries_src"]} -> {e["queries"]["path"]}"')
    e = _exp("dense_pubmed", ["prep.queries_src=s3://x/q.parquet"])
    assert "s3://x/q.parquet" in "\n".join(PS.node_prep(e))
