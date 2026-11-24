import functools
import os
import random
import time
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import readhooks
from bench.data import sources
from conftest import ROWS
from bench.data.sources import InMemorySource, ParquetSource, list_parquet, read_file
from bench.timing import now


def collect(src):
    return [(b.offset, b.vectors.copy()) for b in src]


@pytest.mark.parametrize("prefetch,workers", [(0, 1), (1, 1), (2, 2), (3, 4), (10, 3)])
def test_order_offsets_and_content(corpus, prefetch, workers):
    path, xb = corpus
    src = ParquetSource(path, prefetch=prefetch, workers=workers)
    blocks = collect(src)
    assert [b[1].shape[0] for b in blocks] == ROWS
    assert [b[0] for b in blocks] == list(np.cumsum([0] + ROWS[:-1]))
    np.testing.assert_array_equal(np.concatenate([b[1] for b in blocks]), xb)
    st = src.stats.as_dict()
    assert st["files"] == len(ROWS) and st["rows"] == sum(ROWS)
    # native reads: bytes = the column's compressed bytes
    assert st["bytes"] == sum(sources.read_footer(src.fs, r.path, "embedding").column_bytes
                              for r in src.files)
    assert st["fetch"] > 0 and st["decode"] > 0
    # each iter() re-streams and resets stats
    assert len(collect(src)) == len(ROWS)
    assert src.stats.get("files") == len(ROWS)


def test_order_preserved_under_reverse_completion(corpus):
    """Later files finish first; output must still be in file order."""
    path, xb = corpus
    files = list_parquet(path)[1]
    delays = {os.path.basename(r.path): 0.05 * (len(files) - i) + random.random() * 0.01
              for i, r in enumerate(files)}
    src = ParquetSource(path, prefetch=4, workers=4,
                        read_fn=functools.partial(readhooks.slow_read, delays=delays))
    blocks = list(src)
    np.testing.assert_array_equal(np.concatenate([b.vectors for b in blocks]), xb)
    # it really did complete out of order
    ends = [b.info["w_end"] for b in blocks]
    assert ends[1] > ends[2] > ends[3]


def test_blocks_are_c_contiguous_float32(corpus):
    path, _ = corpus
    for b in ParquetSource(path, prefetch=2, workers=2):
        assert b.vectors.dtype == np.float32 and b.vectors.flags.c_contiguous


def test_dtype_cast(corpus):
    path, xb = corpus
    got = np.concatenate([b.vectors for b in ParquetSource(path, prefetch=1, dtype=np.float16)])
    assert got.dtype == np.float16
    np.testing.assert_array_equal(got, xb.astype(np.float16))


def test_in_memory_source_matches(corpus):
    path, xb = corpus
    src = InMemorySource.load(path)
    blocks = [(b.offset, b.vectors) for b in src.blocks]
    assert [b[0] for b in blocks] == list(np.cumsum([0] + ROWS[:-1]))
    np.testing.assert_array_equal(np.concatenate([b[1] for b in blocks]), xb)


def test_list_parquet_file_uri_and_single_file(corpus):
    path, _ = corpus
    fs, refs = list_parquet("file://" + path)
    assert len(refs) == len(ROWS) and refs == sorted(refs, key=lambda r: r.path)
    _, one = list_parquet(refs[0].path)
    assert len(one) == 1


def test_list_parquet_recursive_is_opt_in_and_path_sorted(tmp_path):
    """Sharded layouts (data/000/.., data/001/..): the default listing stays
    flat (a nested file is NOT picked up), --corpus-recursive finds every
    shard, ordered by FULL path -- so data/000/z sorts before data/001/a and
    global row ids are deterministic, the order nova-bf uses."""
    t = pa.table({"embedding": pa.array([[0.0, 1.0]] * 2, pa.list_(pa.float32()))})
    for rel in ("data/001/a.parquet", "data/000/z.parquet", "data/000/b.parquet",
                "data/top.parquet", "data/000/notes.txt"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".parquet"):
            pq.write_table(t, str(p))
        else:
            p.write_text("x")
    _, flat = list_parquet(str(tmp_path / "data"))
    assert [os.path.basename(r.path) for r in flat] == ["top.parquet"]
    _, rec = list_parquet(str(tmp_path / "data"), recursive=True)
    rel = [os.path.relpath(r.path, str(tmp_path / "data")) for r in rec]
    assert rel == ["000/b.parquet", "000/z.parquet", "001/a.parquet", "top.parquet"]


def test_non_parquet_files_ignored(tmp_path, corpus):
    (tmp_path / "README.txt").write_text("x")
    with pytest.raises(FileNotFoundError):
        list_parquet(str(tmp_path))


def _write(tmp_path, arr):
    p = str(tmp_path / "bad.parquet")
    pq.write_table(pa.table({"embedding": arr}), p)
    return p


def test_ragged_rejected(tmp_path):
    p = _write(tmp_path, pa.array([[1.0, 2.0], [1.0]], pa.list_(pa.float32())))
    with pytest.raises(ValueError, match="ragged"):
        list(ParquetSource(p, prefetch=0))


def test_null_row_rejected(tmp_path):
    p = _write(tmp_path, pa.array([[1.0, 2.0], None], pa.list_(pa.float32())))
    with pytest.raises(ValueError, match="null"):
        list(ParquetSource(p, prefetch=0))


def test_worker_exception_propagates(corpus, monkeypatch):
    path, _ = corpus

    def boom(*a, **kw):
        raise OSError("s3 went away")

    monkeypatch.setattr(sources, "read_file", boom)
    with pytest.raises(OSError, match="s3 went away"):
        list(ParquetSource(path, prefetch=0))
    # reader processes: the same exception type and message, with the
    # reader's traceback chained
    with pytest.raises(OSError, match="s3 went away") as ei:
        list(ParquetSource(path, prefetch=2, workers=2, read_fn=readhooks.failing_read))
    assert "failing_read" in str(ei.value.__cause__)


def test_multi_row_group_file(tmp_path):
    import make_dummy
    x = make_dummy.unit_vectors(np.random.default_rng(5), 1000, 16)
    make_dummy.write_vectors(str(tmp_path / "a.parquet"), x[:600], row_group_size=128)
    make_dummy.write_vectors(str(tmp_path / "b.parquet"), x[600:], row_group_size=50)
    for pf in (0, 2):
        src = ParquetSource(str(tmp_path), prefetch=pf, workers=2)
        blocks = collect(src)
        assert [b[0] for b in blocks] == [0, 600]
        got = np.concatenate([b[1] for b in blocks])
        assert got.flags.c_contiguous
        np.testing.assert_array_equal(got, x)
        assert src.stats.get("row_groups") == 5 + 8
        assert src.stats.get("max_row_groups") == 8


def peak_overlap(intervals):
    """Most intervals open at once (an end at t closes before a start at t)."""
    ev = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals],
                key=lambda e: (e[0], e[1]))
    peak = cur = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


@pytest.mark.parametrize("prefetch,workers,want", [(1, 8, 1), (2, 8, 2), (3, 2, 2),
                                                   (4, 4, 4)])
def test_prefetch_caps_reads_in_flight_processes(corpus, prefetch, workers, want):
    """The same cap with reader PROCESSES: counted from each read's own
    [w_start, w_end] interval (one CLOCK_MONOTONIC across processes), the
    read slowed to 30 ms by the hook, the consumer to 80 ms per block."""
    src = ParquetSource(corpus[0], prefetch=prefetch, workers=workers,
                        read_fn=functools.partial(readhooks.slow_read, default=0.03))
    assert src.reader == "process" and src.readers == want
    src.start()
    # parent-side stamps: when each task went on the pipe, when each block
    # was handed to the consumer (the ring view made for it)
    procs, ring = src._procs, src.ring
    submitted, handed = {}, []
    submit, view = procs.submit, ring.view
    procs.submit = lambda tid, path, slot: (submitted.setdefault(tid, now()),
                                            submit(tid, path, slot))[1]
    ring.view = lambda *a: (handed.append(now()), view(*a))[1]
    blocks = []
    try:
        for b in src:
            time.sleep(0.08)  # slow consumer
            blocks.append(b.info)
            del b
    finally:
        src.close()
    assert len(blocks) == len(ROWS) == len(handed) == len(submitted)
    assert peak_overlap([(r["w_start"], r["w_end"]) for r in blocks]) == want
    # exactly `want` reader processes did the work
    assert len({r["args"]["pid"] for r in blocks}) == want
    # never more than `prefetch` tasks out, those waiting in the pipe (which
    # the reader intervals cannot see once prefetch > readers) included: no
    # extra ring slot taken, and file i + prefetch is neither submitted nor
    # started before file i was handed to the consumer
    assert ring.grown == 0
    for i in range(len(ROWS) - prefetch):
        assert submitted[i + prefetch] >= handed[i], (i, prefetch)
        assert blocks[i + prefetch]["w_start"] >= handed[i], (i, prefetch)


@pytest.mark.parametrize("prefetch", [0, 2])
def test_start_reads_footers_for_every_reader(corpus, prefetch):
    """start() reads every footer, untimed, whatever reads the files: with
    --warmup 0 the first arrow pass fetches no footer (fetch 0), for the
    inline reader too."""
    src = ParquetSource(corpus[0], prefetch=prefetch, workers=2,
                        read_mode="arrow").start()
    try:
        assert set(src.footers) == {r.path for r in src.files}
        assert sum(1 for _ in src) == len(ROWS)
        st = src.stats.as_dict()
        assert st["fetch"] == 0 and st["files"] == len(ROWS)
    finally:
        src.close()


def test_read_mode_choice_and_concurrency(corpus):
    for bad in ("yes", "auto", "ranged", "sequential"):
        with pytest.raises(ValueError):
            ParquetSource(corpus[0], read_mode=bad)
    foot = sources.Footer(10, 1, 600, 4, "float")
    assert sources.choose_read_mode("native", foot, 1000) == ("native", 0.6)
    assert sources.choose_read_mode("arrow", foot, 1300) == ("arrow", 600 / 1300)
    # default: the native 128-GET budget split across concurrent readers, >= 1
    # each (readers = min(prefetch, workers, files): prefetch counts every read
    # in flight; never more readers than files)
    n = sources.NATIVE_CONCURRENCY
    assert ParquetSource(corpus[0], prefetch=0).range_concurrency == n
    assert ParquetSource(corpus[0], prefetch=4, workers=2).range_concurrency == n // 2
    assert ParquetSource(corpus[0], prefetch=2, workers=8).range_concurrency == n // 2
    assert ParquetSource(corpus[0], prefetch=64, workers=48).range_concurrency == \
        n // len(ROWS)
    fs, files = sources.list_parquet(corpus[0])
    assert ParquetSource(corpus[0], prefetch=200, workers=200, fs=fs,
                         files=files * 40).range_concurrency == 1
    assert ParquetSource(corpus[0], prefetch=4, workers=4,
                         range_concurrency=40).range_concurrency == 10
    # per-file: nova-bf's actual behaviour, the budget for every reader
    assert ParquetSource(corpus[0], prefetch=4, workers=4,
                         range_scope="per-file").range_concurrency == n
    with pytest.raises(ValueError):
        sources.range_concurrency_for(2, 24, "both")


def test_configure_arrow_pools():
    cpu, io = sources.configure_arrow_pools(3, 7)
    assert (cpu, io) == (3, 7) == (pa.cpu_count(), pa.io_thread_count())
    cpu, io = sources.configure_arrow_pools(0, 0)  # 0 -> usable CPUs; io untouched
    assert cpu == sources.usable_cpu_count() == pa.cpu_count() and io == 7
    assert 1 <= sources.usable_cpu_count() <= (__import__("os").cpu_count() or 1)


def test_usable_cpu_count_respects_affinity_and_quota(monkeypatch):
    import os
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(sources, "_cgroup_cpu_quota", lambda: None)
    assert sources.usable_cpu_count() == min(3, os.cpu_count())
    monkeypatch.setattr(sources, "_cgroup_cpu_quota", lambda: 2)
    assert sources.usable_cpu_count() == min(2, os.cpu_count())


def test_empty_file_is_an_error(tmp_path):
    zero = tmp_path / "zero.parquet"
    zero.write_bytes(b"")
    with pytest.raises(ValueError, match="zero.parquet: empty file"):
        list(ParquetSource(str(zero), prefetch=0))
    norows = _write(tmp_path, pa.array([], pa.list_(pa.float32())))
    with pytest.raises(ValueError, match="bad.parquet: empty file"):
        read_file(*list_parquet(norows)[0:1], norows, "embedding")
    with pytest.raises(ValueError, match="empty file"):
        list(ParquetSource(norows, prefetch=2, workers=2))


def test_trailing_slash_directory(corpus):
    path, xb = corpus
    for uri in (path + "/", "file://" + path + "/"):
        _, refs = list_parquet(uri)
        assert len(refs) == len(ROWS)
        np.testing.assert_array_equal(
            np.concatenate([b.vectors for b in ParquetSource(uri, prefetch=1)]), xb)


def test_trailing_slash_stripped_after_from_uri(corpus, monkeypatch):
    """Local from_uri already strips the slash; s3's may not, so check that
    list_parquet strips whatever from_uri hands back."""
    import types
    import pyarrow.fs as pafs
    path, _ = corpus
    seen = []

    class Fs:
        def get_file_info(self, p):
            seen.append(p if isinstance(p, str) else p.base_dir)
            return pafs.LocalFileSystem().get_file_info(p)

    fake = types.SimpleNamespace(
        FileType=pafs.FileType, FileSelector=pafs.FileSelector,
        FileSystem=types.SimpleNamespace(from_uri=lambda uri: (Fs(), path + "/")))
    monkeypatch.setattr(sources, "pafs", fake)
    _, refs = list_parquet("s3://bucket/prefix/")
    assert len(refs) == len(ROWS) and seen and all(not s.endswith("/") for s in seen)


def _fineweb_like(out, payload_bytes, value_dtype=np.float16, n_files=3, rows=400,
                  rg=150, d=16):
    import make_dummy
    xs = make_dummy.make_corpus(str(out), n_files, rows, d, seed=4,
                                column="dense_embedding", row_group_size=rg,
                                value_dtype=value_dtype, payload_bytes=payload_bytes)
    return [pq.read_table(p, columns=["dense_embedding"]) for p in xs]


def test_host_widen_is_timed_apart_from_decode(tmp_path):
    _fineweb_like(tmp_path / "c", payload_bytes=0)
    src = ParquetSource(str(tmp_path / "c"), column="dense_embedding", prefetch=2,
                        workers=2, dtype=np.float32)
    blocks = collect(src)
    assert all(x.dtype == np.float32 and x.flags.c_contiguous for _, x in blocks)
    st = src.stats.as_dict()
    assert st["widen"] > 0 and st["decode"] > 0
    keep = ParquetSource(str(tmp_path / "c"), column="dense_embedding", prefetch=0)
    collect(keep)
    assert keep.stats.get("widen") == 0  # stored dtype: no host cast at all


def test_non_half_single_floats_widen_on_host(tmp_path):
    p = _write(tmp_path, pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64())))
    x = read_file(*list_parquet(p)[0:1], p, "embedding")
    assert x.dtype == np.float32 and x.tolist() == [[1, 2], [3, 4]]


@pytest.mark.parametrize("prefetch", [0, 2])
def test_yielded_block_is_the_consumers_only_reference(corpus, prefetch):
    """The generator keeps no reference to the block it yielded: the
    consumer's ``del`` frees it there (run_pass's ``other``), not inside the
    next ``next()`` (``source_stall``) when the generator resumes. Every
    reader kind (prefetch 0 = inline)."""
    path, _ = corpus
    with ParquetSource(path, prefetch=prefetch, workers=2).start() as src:
        it = iter(src)
        for _ in range(2):
            b = next(it)
            ref = weakref.ref(b.vectors)
            del b
            assert ref() is None, f"{src.reader}: the generator still holds the block"
        it.close()
