"""The nova-bf uuid table cached on disk per corpus (load.cached_uuid_table, keyed by
exact.Corpus.identity) and the threaded id lookup (load._index_in)."""

import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from bench.verify import exact as X
from bench.verify import load as L


def _corpus(tmp_path, rows=(50, 1, 0, 37)):
    d = tmp_path / "corpus"
    d.mkdir()
    for i, n in enumerate(rows):
        pq.write_table(pa.table({"v": pa.array(np.arange(n, dtype=np.float32))}),
                       str(d / f"f{i}.parquet"), row_group_size=16)
    return str(d)


def _ident(uri, keys, offsets):
    fs, paths = L.corpus_paths(uri, len(keys))
    return X.Corpus(fs, paths, offsets).identity()


def _same_bytes(a, b):
    """Equal large_string arrays, down to the offsets and the bytes they cover."""
    assert a.type == b.type == pa.large_string() and a.equals(b)
    oa = np.frombuffer(a.buffers()[1], np.int64)[a.offset:a.offset + len(a) + 1]
    ob = np.frombuffer(b.buffers()[1], np.int64)[b.offset:b.offset + len(b) + 1]
    assert np.array_equal(oa - oa[0], ob - ob[0])
    assert a.buffers()[2].to_pybytes()[oa[0]:oa[-1]] == b.buffers()[2].to_pybytes()[ob[0]:ob[-1]]


def test_the_cached_table_is_the_fresh_table_and_is_reused(tmp_path):
    uri = _corpus(tmp_path)
    keys, rows, offsets, _ = L.corpus_index(uri, need_ids=False)
    fresh = L.uuid_table(keys, rows, offsets)
    assert fresh.to_pylist()[50] == L.uuid_of(keys[1], 0) and fresh[87].as_py() == L.uuid_of(keys[3], 36)
    path = L.uuid_cache_path(str(tmp_path / "cache"), keys, _ident(uri, keys, offsets))
    built, hit = L.cached_uuid_table(keys, rows, offsets, path)
    assert not hit and os.path.isfile(path)
    assert not [f for f in os.listdir(tmp_path / "cache") if f.endswith(".tmp")]
    got, hit = L.cached_uuid_table(keys, rows, offsets, path)
    assert hit
    _same_bytes(built, fresh)
    _same_bytes(got, fresh)
    ids = pa.array([fresh[i].as_py() for i in (0, 87, 50, 3)] + ["nope"])
    assert L.uuid_gids(ids, got).tolist() == [0, 87, 50, 3, -1]


def test_the_key_changes_with_any_files_identity(tmp_path):
    uri = _corpus(tmp_path)
    keys, rows, offsets, _ = L.corpus_index(uri, need_ids=False)
    c = str(tmp_path / "cache")
    p0 = L.uuid_cache_path(c, keys, _ident(uri, keys, offsets))
    assert p0 == L.uuid_cache_path(c, keys, _ident(uri, keys, offsets))   # deterministic
    f = os.path.join(uri, "f1.parquet")
    st = os.stat(f)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))              # mtime only
    p1 = L.uuid_cache_path(c, keys, _ident(uri, keys, offsets))
    assert p1 != p0
    # same size, same mtime, other values: the footer hash (statistics) changes
    pq.write_table(pa.table({"v": pa.array([7.0], pa.float32())}), f, row_group_size=16)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    p2 = L.uuid_cache_path(c, keys, _ident(uri, keys, offsets))
    assert p2 not in (p0, p1)
    assert L.uuid_cache_path(c, keys[::-1], _ident(uri, keys, offsets)) != p2   # other keys
    os.remove(os.path.join(uri, "f2.parquet"))                               # a file less
    k2, _, o2, _ = L.corpus_index(uri, need_ids=False)
    assert L.uuid_cache_path(c, k2, _ident(uri, k2, o2)) not in (p0, p1, p2)


@pytest.mark.parametrize("bad", ["garbage", "short", "empty"])
def test_an_unreadable_or_wrong_length_cache_file_is_rebuilt(tmp_path, bad):
    uri = _corpus(tmp_path)
    keys, rows, offsets, _ = L.corpus_index(uri, need_ids=False)
    fresh = L.uuid_table(keys, rows, offsets)
    path = str(tmp_path / "u.arrow")
    if bad == "garbage":
        open(path, "wb").write(b"not an arrow file")
    elif bad == "empty":
        open(path, "wb").close()
    else:
        t = pa.table({"u": fresh.slice(0, 10)})
        with pa.OSFile(path, "wb") as fh, pa.ipc.new_file(fh, t.schema) as w:
            w.write_table(t)
    got, hit = L.cached_uuid_table(keys, rows, offsets, path)
    assert not hit
    _same_bytes(got, fresh)
    again, hit = L.cached_uuid_table(keys, rows, offsets, path)
    assert hit
    _same_bytes(again, fresh)


def test_load_round_reads_the_same_table_with_and_without_the_disk_cache(tmp_path):
    uri = _corpus(tmp_path)
    keys, rows, offsets, _ = L.corpus_index(uri, need_ids=False)
    a = L.Args(corpus=uri, vcol="v", queries="", qcol="", metric="ip", k=3)
    L.CACHE.clear()
    try:
        plain = L._cached_uuids(a, keys, rows, offsets)
        L.CACHE.clear()
        cached = L._cached_uuids(a, keys, rows, offsets, str(tmp_path / "c"))
        assert [f for f in os.listdir(tmp_path / "c") if f.startswith("uuids-")]
        L.CACHE.clear()
        again = L._cached_uuids(a, keys, rows, offsets, str(tmp_path / "c"))   # from the file
    finally:
        L.CACHE.clear()
    _same_bytes(cached, plain)
    _same_bytes(again, plain)


def test_the_lookup_equals_one_index_in():
    rng = np.random.default_rng(0)
    table = pa.array([f"id{i}" for i in rng.permutation(500)], pa.large_string())
    for n in (0, 1, 6, 7, 8, 333):
        vals = pa.array([f"id{i}" for i in rng.integers(0, 520, n)], pa.string())   # some missing
        want = pc.fill_null(pc.index_in(vals.cast(table.type), value_set=table), -1).to_numpy(
            zero_copy_only=False).astype(np.int64)
        for cpus in (1, 3, 8):
            old = pa.cpu_count()
            pa.set_cpu_count(cpus)
            try:
                got = L.uuid_gids(vals, table)
            finally:
                pa.set_cpu_count(old)
            assert got.dtype == np.int64 and np.array_equal(got, want), (n, cpus)
