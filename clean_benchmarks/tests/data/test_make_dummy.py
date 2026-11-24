import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import make_dummy


def test_layout_matches_nova_embed(tmp_path):
    paths = make_dummy.make_corpus(str(tmp_path / "c"), 2, 500, 48, seed=3)
    assert len(paths) == 2
    md = pq.ParquetFile(paths[0]).metadata
    assert md.num_row_groups == 1
    assert md.row_group(0).column(0).compression == "SNAPPY"
    t = pq.read_table(paths[0])
    assert t.schema.field("embedding").type == pa.list_(pa.float32())
    x = np.stack(t.column("embedding").to_numpy(zero_copy_only=False))
    assert x.shape == (500, 48)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-5)


def test_deterministic_and_files_differ(tmp_path):
    a = make_dummy.make_corpus(str(tmp_path / "a"), 2, 10, 8, seed=1)
    b = make_dummy.make_corpus(str(tmp_path / "b"), 2, 10, 8, seed=1)
    ta, tb = (pq.read_table(p) for p in (a[0], b[0]))
    assert ta.equals(tb)
    assert not pq.read_table(a[0]).equals(pq.read_table(a[1]))


def test_row_group_size_override(tmp_path):
    p = make_dummy.write_vectors(str(tmp_path / "x.parquet"),
                                 np.ones((100, 4), np.float32), row_group_size=30)
    assert pq.ParquetFile(p).metadata.num_row_groups == 4


def test_int32_offset_guard_fails_fast():
    """n * width >= 2^31 overflows Arrow's int32 offsets (list<float> and the
    string payload): refused before anything is allocated."""
    import pytest
    make_dummy.check_int32_offsets(2**20, 2**11 - 1, "x")
    with pytest.raises(ValueError, match="2\\^31"):
        make_dummy.check_int32_offsets(2**20, 2**11, "payload bytes")
    with pytest.raises(ValueError, match="payload bytes"):
        make_dummy.payload_array(None, 2**20, 2**11)  # guard runs before the RNG


def test_payload_rng_streams_are_distinct(tmp_path, monkeypatch):
    """Every file's payload stream differs from every file's vector stream
    (it once was [seed + i, 7], i.e. file 7's vector seed for file 0)."""
    seeds, real = [], np.random.default_rng
    monkeypatch.setattr(make_dummy.np.random, "default_rng",
                        lambda s=None: seeds.append(tuple(s)) or real(s))
    make_dummy.make_corpus(str(tmp_path / "c"), 9, 4, 4, seed=0, payload_bytes=3)
    assert len(seeds) == 18 and len(set(seeds)) == 18
