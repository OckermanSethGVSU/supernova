"""The filtered baseline's reader integration (filtered.py + sources.py hooks):
every reader kind / read mode reads a file's filter columns with its vectors,
evaluates the filter, and hands the masks over with the block -- and they must
equal a direct DuckFilter.evaluate of that file (whose parity with nova-bf is
pinned by test_duckdb_filter.py)."""
import functools
import gc
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench.data.filters import duckdb_filter as dfm
from bench.data.filters import reader as filtered
from bench.data.sources import ParquetSource
import readhooks
from test_duckdb_filter import CDATES, QDATES, make_corpus, make_queries

D = 8
ROWS = [301, 7, 1024, 1, 77]


@pytest.fixture(scope="module")
def fcorpus(tmp_path_factory):
    """Files with vectors AND payload columns; returns (dir, [payload tables], vectors)."""
    out = tmp_path_factory.mktemp("fcorpus")
    payloads, vecs = [], []
    for i, n in enumerate(ROWS):
        p = make_corpus(n, seed=100 + i)
        x = np.random.default_rng(i).standard_normal((n, D)).astype(np.float32)
        emb = pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), D)
        t = p.append_column("embedding", pa.ListArray.from_arrays(
            pa.array(np.arange(0, n * D + 1, D, dtype=np.int32)), pa.array(x.reshape(-1))))
        pq.write_table(t, str(out / f"part-{i:05d}.parquet"), row_group_size=max(1, n // 3))
        payloads.append(p)
        vecs.append(x)
        del emb
    return str(out), payloads, vecs


QUERIES = make_queries(40, seed=7)

FILTERS = {
    "static": {"must": [{"field": "i", "range": {"gte": 0}}],
               "should": [{"field": "text", "match_text": "alpha"}, {"field": "s", "match": "a"}]},
    "per_query": {"must": [{"field": "d", "range_from_query": {"gte": "qd_lo", "lt": "qd_hi"}}],
                  "should": [{"field": "text", "match_text_from_query": "qphrase"},
                             {"field": "s", "match_from_query": "qlist"}]},
}


def job(name, **kw):
    return filtered.FilterJob(FILTERS[name], QUERIES, CDATES, QDATES, threads=2,
                              selfcheck=False, **kw)


def expected(name, payload):
    eng = job(name).build()
    res = eng.evaluate(payload)
    return res.row_mask() if not res.per_query else res.query_masks()


def check_block(b, name, payload, vecs):
    assert np.array_equal(np.asarray(b.vectors), vecs)
    f = b.filter
    assert f is not None and f.n_rows == payload.num_rows
    want = expected(name, payload)
    if name == "static":
        assert f.kind == "static"
        assert np.array_equal(f.row_mask(), want)
    else:
        assert f.kind == "per_query"
        got = dfm.unpack(f.query_masks_packed(), f.n_rows)
        assert np.array_equal(got, want)
        # distinct masks really are distinct, counts are their popcounts
        assert len({m.tobytes() for m in f.masks}) == len(f.masks)
        assert f.counts.tolist() == [int(dfm.unpack(m, f.n_rows).sum()) for m in f.masks]
        live = f.live_queries()
        assert np.array_equal(live, np.flatnonzero(want.any(axis=1)))
    assert {"filter", "filter_eval", "filter_dedup"} <= set(b.info)
    # DuckFilter's stage breakdown rides along, nested inside filter_eval
    sub = [b.info[k] for k in filtered.FILTER_STAGES]
    assert all(v >= 0 for v in sub) and sum(sub) <= b.info["filter_eval"]
    assert b.info["filter_sql"] > 0 and b.info["filter_text"] > 0  # both leaf kinds present


@pytest.mark.slow
@pytest.mark.parametrize("name", ["static", "per_query"])
@pytest.mark.parametrize("reader,prefetch", [("process", 2), ("process", 1), ("inline", 0)])
@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_blocks_carry_exact_masks(fcorpus, name, reader, prefetch, mode):
    path, payloads, vecs = fcorpus
    kw = dict(prefetch=prefetch, workers=2, read_mode=mode, filter_job=job(name))
    with ParquetSource(path, column="embedding", **kw).start() as src:
        for _ in range(2):  # two passes: readers / slots are reused
            n = 0
            for i, b in enumerate(src):
                check_block(b, name, payloads[i], vecs[i])
                n += 1
                del b
            assert n == len(ROWS)
            st = src.stats.as_dict()
            assert st.get("filter", 0) > 0 and st.get("filter_eval", 0) > 0
            # summed over files through every reader kind (process: worker stats merged)
            assert all(st.get(k, 0) > 0 for k in ("filter_prep", "filter_sql", "filter_text",
                                                  "filter_combine"))
            assert sum(st.get(k, 0) for k in filtered.FILTER_STAGES) <= st["filter_eval"]
            if mode == "native":
                assert st.get("payload_fetch", 0) > 0  # filter columns via rust_fetch


def test_native_filter_columns_come_from_rust_fetch(fcorpus):
    path, payloads, vecs = fcorpus
    with ParquetSource(path, column="embedding", prefetch=1, workers=1, read_mode="native",
                       filter_job=job("per_query")).start() as src:
        for b in src:
            assert b.info["read_path"] == "native"
            assert b.info["payload_path"] == "native-fetch"
            assert b.info.get("native_etag")


def test_masks_keep_their_slot_alive(fcorpus):
    """A FilterBlock held after its vectors are dropped must not be overwritten
    by later files (masks and vectors share one slot base)."""
    path, payloads, vecs = fcorpus
    with ParquetSource(path, column="embedding", prefetch=2, workers=2,
                       filter_job=job("per_query")).start() as src:
        it = iter(src)
        b0 = next(it)
        f0 = b0.filter
        snapshot = f0.masks.copy()
        del b0
        gc.collect()
        for b in it:
            del b
        assert np.array_equal(f0.masks, snapshot)
        assert src.ring.grown >= 1  # the held slot was not reused: the ring grew
        del f0
        gc.collect()


def test_slot_mask_capacity_covers_every_combo(fcorpus):
    path, _, _ = fcorpus
    j = job("per_query")
    eng = j.build(selfcheck=False)
    assert j.max_masks() == len(eng.combo_keys)
    assert job("static").max_masks() == 1
    with ParquetSource(path, column="embedding", prefetch=1, workers=1, filter_job=j).start() as src:
        offset, cap = src._mask_layout
        assert offset % filtered.MASK_ALIGN == 0 and offset >= src.max_block_bytes()
        assert cap >= len(eng.combo_keys) * dfm.nbytes(max(ROWS))


def test_extra_columns_need_dict():
    from bench.data.sources import read_file
    with pytest.raises(ValueError):
        read_file(None, "x", "embedding", extra_columns=["s"], extra=None, mode="arrow")


def test_reader_without_filter_unchanged(fcorpus):
    path, payloads, vecs = fcorpus
    with ParquetSource(path, column="embedding", prefetch=2, workers=2).start() as src:
        for i, b in enumerate(src):
            assert b.filter is None
            assert np.array_equal(np.asarray(b.vectors), vecs[i])
            assert "filter" not in (b.info or {})


# ---------------------------------------------------------------------------
# the reader plumbing: edge cases
# ---------------------------------------------------------------------------

from bench.data import sources  # noqa: E402


def _versioned_file(tmp_path, D=8, N=64):
    """One corpus file and ``write(version)``, which rewrites it as that version
    (picklable: a reader process can call it)."""
    p = str(tmp_path / "part-00000.parquet")
    return p, functools.partial(readhooks.write_version, p, D=D, N=N)


@pytest.mark.parametrize("prefetch", [0, 1], ids=["inline", "process"])
def test_native_filter_columns_refuse_a_changed_object(tmp_path, monkeypatch, prefetch):
    """The object changes between the native dense read and the filter columns' read: the
    filter columns are pinned to the dense read's ETag (sources.read_file ->
    _native_fetch_columns(pin_etag=...)), so the read is refused (NativeRefused, the leg
    fails), never one version's vectors with another version's mask. prefetch 0 reads
    inline (the patch is in this process); prefetch 1 reads in a reader process (the patch
    goes with the read_fn) and the refusal reaches the consumer."""
    p, write = _versioned_file(tmp_path)
    write(1)
    j = filtered.FilterJob({"must": [{"field": "s", "match": "a"}]}, threads=1, selfcheck=False)
    kw = dict(column="embedding", prefetch=prefetch, workers=1, read_mode="native",
              filter_job=j)
    if prefetch:
        kw["read_fn"] = functools.partial(readhooks.changes_after_dense_read,
                                          rewrite=functools.partial(write, 2))
    else:
        real = sources.read_native

        def changing(*a, **k):
            x = real(*a, **k)
            write(2)  # the object changes between the two reads
            return x

        monkeypatch.setattr(sources, "read_native", changing)
    with ParquetSource(str(tmp_path), **kw).start() as src:
        with pytest.raises(sources.NativeRefused, match="object changed since the dense read"):
            list(src)


@pytest.mark.slow
@pytest.mark.parametrize("reader,prefetch", [("inline", 0)])
def test_inline_filters_built_at_start(fcorpus, monkeypatch, reader, prefetch):
    """DuckFilters are built in start(), never inside a pass."""
    path, payloads, vecs = fcorpus
    builds = []
    orig = filtered.FilterJob.build

    def counting(self, *a, **k):
        builds.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(filtered.FilterJob, "build", counting)
    kw = dict(prefetch=prefetch, workers=2, filter_job=job("per_query"))
    with ParquetSource(path, column="embedding", **kw).start() as src:
        at_start = len(builds)
        assert at_start == 1
        got = []
        for _ in range(3):
            for i, b in enumerate(src):
                got.append((i, b.filter.query_masks_packed().copy(), b.filter.n_rows))
                del b
        assert len(builds) == at_start  # no builds inside the passes
    monkeypatch.undo()
    for i, packed, n in got:
        assert np.array_equal(dfm.unpack(packed, n), expected("per_query", payloads[i]))


@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_filter_timeline_stamps(fcorpus, mode):
    path, _, _ = fcorpus
    with ParquetSource(path, column="embedding", prefetch=2, workers=2, read_mode=mode,
                       filter_job=job("per_query")).start() as src:
        for b in src:
            r = b.info
            assert r["w_start"] <= r["w_end"] <= r["filter_start"] <= r["filter_end"]
            if mode == "native":
                assert r["payload_start"] <= r["payload_end"] == r["w_end"]


# ---------------------------------------------------------------------------
# files that change after start, and slot room
# ---------------------------------------------------------------------------

def _write_rows(p, n, D=4, value=2.0, s="a"):
    x = np.full((n, D), value, np.float32)
    t = pa.table({"s": [s] * n, "embedding": pa.FixedSizeListArray.from_arrays(
        pa.array(x.reshape(-1)), D)})
    pq.write_table(t, p)


@pytest.mark.parametrize("grow_to", [129, 5000])
@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_grown_file_never_spills_into_masks(tmp_path, mode, grow_to):
    """A file that grows after start() must fail LOUDLY -- the vector-region
    room check (large growth), the mask-capacity check (small growth that still
    fits the aligned vector region) or the native footer check -- never write
    vectors over the mask region (and masks over the vectors)."""
    p = str(tmp_path / "part-00000.parquet")
    _write_rows(p, 128)
    j = filtered.FilterJob({"must": [{"field": "s", "match": "a"}]}, threads=1, selfcheck=False)
    with ParquetSource(str(tmp_path), column="embedding", prefetch=1, workers=1,
                       read_mode=mode, filter_job=j).start() as src:
        _write_rows(p, grow_to)  # grows after the ring was sized from the footers
        try:
            for b in src:
                # handed over only if it is EXACTLY the new file: vectors all
                # 2.0 (none overwritten by masks) and a mask over every row
                v = np.asarray(b.vectors)
                assert v.shape[0] == grow_to and np.all(v == 2.0)
                assert b.filter.n_rows == grow_to and int(b.filter.counts[0]) == grow_to
                assert grow_to != 5000 or mode != "arrow"   # that one must not fit
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            assert ("larger than a shared-memory ring slot" in msg or "native refused" in msg), msg


# ---------------------------------------------------------------------------
# reader-side compaction (--filter-mode gather / --filter-compact union):
# differential against the consumer-side code it replaced (OldAdapter, a
# verbatim copy of FilterAdapter's old _prepare_gather / _prepare_bitmap) and
# against the uncompacted masked reference, across reader kinds and transfer
# modes
# ---------------------------------------------------------------------------

from bench.systems import loop  # noqa: E402
from bench.systems import faiss_engine  # noqa: E402
from bench.data.filters import search as fsm  # noqa: E402

CD = 8
CNQ, CK = 24, 10
# rows and the per-file pattern of the int column g (see _compaction_corpus):
#   mixed  g in 0..5        static partial; per-query: only qg=3 is live
#   all7   g = 7            static everything passes; per-query union = all
#   zero   g = 0            static nothing passes (skipped); per-query: no live
#   rare9  g = 9 on 5 rows  static 5 rows; union 5 wanted -> the 257-row floor
#   small  g = 9 on 7 rows  union floor = the whole (100-row) file
#   one7   one row, g = 7
CFILES = [("mixed", 600), ("all7", 300), ("zero", 400), ("rare9", 1000), ("small", 100),
          ("one7", 1)]
CFILTERS = {"static": {"must": [{"field": "g", "range": {"gte": 1}}]},
            "per_query": {"must": [{"field": "g", "match_from_query": "qg"}]}}


@pytest.fixture(scope="module")
def ccorpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("ccorpus")
    rng = np.random.default_rng(42)
    for i, (kind, n) in enumerate(CFILES):
        g = {"mixed": rng.integers(0, 6, n), "all7": np.full(n, 7), "zero": np.zeros(n, int),
             "one7": np.full(n, 7)}.get(kind)
        if g is None:
            g = np.zeros(n, int)
            g[rng.choice(n, 5 if kind == "rare9" else 7, replace=False)] = 9
        x = rng.standard_normal((n, CD)).astype(np.float32)
        pq.write_table(pa.table({
            "g": pa.array(g.astype(np.int64)),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), CD)}),
            str(out / f"part-{i:05d}.parquet"))
    qg = np.array([[7, 9, 3, 11][i % 4] for i in range(CNQ)], np.int64)  # 11: dead everywhere
    xq = rng.standard_normal((CNQ, CD)).astype(np.float32)
    return str(out), pa.table({"qg": pa.array(qg)}), xq


def cjob(name, queries):
    return filtered.FilterJob(CFILTERS[name], queries if name == "per_query" else None,
                              threads=1, selfcheck=False)


class OldAdapter(fsm.FilterAdapter):
    """The consumer-side compaction this change replaced, verbatim: blocks
    arrive whole and are gathered here."""

    def _prepare_gather(self, block, fb):
        keep = np.flatnonzero(fb.row_mask())
        self.stats.add("filter_rows_kept", len(keep))
        self.stats.add("filter_allowed_pairs", len(keep) * self.nq)
        if len(keep) == 0:
            return None
        self.stats.add("filter_live_queries", self.nq)
        self.stats.add("filter_searched_queries", self.nq)
        xc = np.take(block.vectors, keep, axis=0)
        return fsm.Prep(xc, None, None, keep, block.offset)

    def _prepare_bitmap(self, block, fb):
        live = fb.live_queries()
        self.stats.add("filter_allowed_pairs",
                       int(np.asarray(fb.counts, np.int64)[fb.mask_of_query].sum()))
        self.stats.add("filter_live_queries", len(live))
        self.stats.add("filter_distinct_masks", len(fb.masks))
        if len(live) == 0 and self.skip != "none":
            return None
        queries = live if self.skip == "queries" and len(live) < self.nq else None
        self.stats.add("filter_searched_queries", self.nq if queries is None else len(queries))
        n_file = fb.n_rows
        keep, fb, added = fsm._compact_union(fb, queries, max_frac=filtered.COMPACT_MAX_FRAC)
        self.stats.add("filter_compact_rows", len(keep))
        if len(keep) == n_file:
            self.stats.add("filter_compact_whole_files", 1)
        if added:
            self.stats.add("filter_compact_floor_files", 1)
            self.stats.add("filter_compact_floor_rows", added)
        if len(keep) == 0:
            return None
        return fsm.Prep(np.take(block.vectors, keep, axis=0), fb, queries, keep, block.offset)


def _counters(filt):
    return {k: v for k, v in filt.stats.as_dict().items() if k != "filter_handoff"}


def _src_kw(reader):
    return {"process": dict(prefetch=2, workers=2),
            "inline": dict(prefetch=0)}[reader]


def _cpass(path, queries, backend, nq, name, mode, skip, reader, impl, pin=None,
           extra_slots=0):
    """One run_pass. ``mode``: gather | union (per-query bitmap + union) |
    bitmap (no compaction); ``impl``: new (reader-compacted blocks) | old
    (whole blocks, OldAdapter compacts on the consumer)."""
    kind = "static" if name == "static" else "per_query"
    spec = None
    if impl == "new" and mode != "bitmap":
        spec = filtered.CompactSpec("gather") if mode == "gather" \
            else filtered.CompactSpec("union", skip)
    cls = OldAdapter if impl == "old" else fsm.FilterAdapter
    filt = cls("gather" if mode == "gather" else "bitmap", nq, CK, skip=skip, kind=kind,
               compact="union" if mode == "union" else "none")
    with ParquetSource(path, column="embedding", filter_job=cjob(name, queries),
                       filter_compact=spec, pin=pin, extra_slots=extra_slots,
                       **_src_kw(reader)).start() as src:
        D, I, info = loop.run_pass(src, backend, nq, CK, filt=filt)
        st = src.stats.as_dict()
    return D, I, info, _counters(filt), st


def _masked(xq):
    b = fsm.MaskedBackend(CK)
    b.setup(xq)
    return b


def _reference(path, queries, xq, name):
    """Uncompacted: the masked engine with the bitmap prefilter over whole blocks."""
    D, I, *_ = _cpass(path, queries, _masked(xq), len(xq), name, "bitmap", "none", "inline",
                      "new")
    return D, I


@pytest.mark.slow
@pytest.mark.parametrize("reader", ["process", "inline"])
@pytest.mark.parametrize("name,mode,skip", [("static", "gather", "queries"),
                                            ("per_query", "union", "queries"),
                                            ("per_query", "union", "files"),
                                            ("per_query", "union", "none")])
def test_reader_compaction_matches_consumer_compaction(ccorpus, reader, name, mode, skip):
    """Results IDENTICAL to the old consumer-side compaction and to the
    uncompacted masked reference; every counter equal to the old one; the
    union floor and an all-passing / nothing-passing file are in the corpus;
    the compaction is reader time (filter_compact, inside filter)."""
    path, queries, xq = ccorpus
    new = _cpass(path, queries, _masked(xq), CNQ, name, mode, skip, reader, "new")
    old = _cpass(path, queries, _masked(xq), CNQ, name, mode, skip, reader, "old")
    np.testing.assert_array_equal(new[0], old[0])
    np.testing.assert_array_equal(new[1], old[1])
    Dr, Ir = _reference(path, queries, xq, name)
    np.testing.assert_array_equal(new[1], Ir)
    # scores: the same rows in a different GEMM shape (BLAS rounding)
    np.testing.assert_allclose(new[0], Dr, rtol=0, atol=1e-5)
    assert new[3] == old[3]
    assert new[2]["rows"] == old[2]["rows"] == sum(n for _, n in CFILES)
    assert new[2]["rows_per_file"] == old[2]["rows_per_file"]
    assert new[2]["max_block_bytes"] == old[2]["max_block_bytes"]
    st = new[4]
    assert st["filter_compact"] > 0 and "filter_compact" not in old[4]
    assert st["filter"] >= st["filter_compact"] + st["filter_eval"]
    if mode == "union":
        # the fixture really hits the 257 floor (mixed: ~100 rows of g = 3;
        # rare9: 5 rows) and the small-file floor (small: 7 of 100 rows),
        # counted independently from the files' g column
        qg = set(queries.column("qg").to_pylist())
        files, rows = 0, 0
        for f in sorted(os.listdir(path)):
            g = pq.read_table(os.path.join(path, f), columns=["g"]).column("g").to_numpy()
            w = int(np.isin(g, list(qg)).sum())
            if w and w < min(257, len(g)):
                files, rows = files + 1, rows + min(257, len(g)) - w
        assert files == 3
        assert (new[3]["filter_compact_floor_files"], new[3]["filter_compact_floor_rows"]) == \
            (files, rows)
    else:
        assert new[3]["filter_skipped_files"] == 1   # the all-zero file


@pytest.mark.slow
@pytest.mark.parametrize("reader", ["process", "inline"])
def test_compacted_blocks_are_the_kept_rows_in_place(ccorpus, reader):
    """Each block is the kept rows (= the file rows ``compact.keep`` of the
    whole block), for process readers IN the ring slot (same base as the
    masks), union masks re-packed exactly as _compact_union does."""
    path, queries, xq = ccorpus
    full = {}
    with ParquetSource(path, column="embedding", filter_job=cjob("per_query", queries),
                       **_src_kw(reader)).start() as src:
        for b in src:
            full[b.path] = (np.array(b.vectors), b.filter)
    for spec in (filtered.CompactSpec("union", "queries"), filtered.CompactSpec("union", "none")):
        with ParquetSource(path, column="embedding", filter_job=cjob("per_query", queries),
                           filter_compact=spec, **_src_kw(reader)).start() as src:
            for _ in range(2):   # slots reused across passes
                for b in src:
                    x, fb = full[b.path]
                    c = b.compact
                    live = fb.live_queries()
                    q = live if spec.skip == "queries" and len(live) < CNQ else None
                    keep, fc, added = fsm._compact_union(fb, q)
                    assert c.keep.dtype == np.int64 and c.keep.tolist() == keep.tolist()
                    assert (c.n_rows, c.added) == (len(x), added)
                    np.testing.assert_array_equal(b.vectors, x[keep])
                    np.testing.assert_array_equal(b.filter.masks, fc.masks)
                    assert b.filter.counts.tolist() == fb.counts.tolist()
                    assert b.filter.n_rows == len(keep)
                    if reader == "process" and len(keep):
                        # compacted IN the ring slot: vectors at its start, keep
                        # in its keep region, masks in its mask region
                        a = b.vectors.__array_interface__["data"][0]
                        assert a in {sources.buffer_address(mm) for mm in src.ring.mms}
                        assert c.keep.__array_interface__["data"][0] == \
                            a + src._keep_layout[0]
                        assert b.filter.masks.__array_interface__["data"][0] == \
                            a + src._mask_layout[0]
                    del b, c
    # gather: the static mask's rows; the FilterBlock stays the file's
    with ParquetSource(path, column="embedding", filter_job=cjob("static", None),
                       filter_compact=filtered.CompactSpec("gather"),
                       **_src_kw(reader)).start() as src:
        for b in src:
            keep = np.flatnonzero(b.filter.row_mask())
            assert b.compact.keep.tolist() == keep.tolist()
            assert b.filter.n_rows == b.compact.n_rows
            np.testing.assert_array_equal(b.vectors, full[b.path][0][keep])


def test_compact_rows_in_place_is_a_gather():
    rng = np.random.default_rng(0)
    for n, p in ((1, 1.0), (50, 0.0), (50, 1.0), (1000, 0.3), (1000, 0.97), (4097, 0.5)):
        x = rng.standard_normal((n, 3)).astype(np.float32)
        keep = np.flatnonzero(rng.random(n) < p)
        want = x[keep].copy()
        y = x.copy()
        # tiny chunks: many chunk boundaries, sources never overwritten
        got = filtered.compact_rows(y, keep, chunk_bytes=7 * 12)
        assert got.base is y or got is y or len(keep) == 0
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(y[len(keep):], x[len(keep):])  # the tail untouched
    with pytest.raises(RuntimeError, match="ascending"):
        filtered.compact_rows(np.zeros((5, 2)), np.array([3, 1]))
    with pytest.raises(RuntimeError, match="ascending"):
        filtered.compact_rows(np.zeros((5, 2)), np.array([1, 5]))


def test_read_only_block_is_compacted_into_a_copy():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    x.flags.writeable = False
    masks = np.packbits(np.arange(10) % 3 == 0, bitorder="little")[None, :]
    info = {"kind": "static", "n_rows": 10, "m": 1, "nb": 2, "counts": np.array([4])}
    y, m2, i2, c = filtered.compact_block(filtered.CompactSpec("gather"), x, masks, info)
    np.testing.assert_array_equal(y, x[[0, 3, 6, 9]])
    assert c.keep.tolist() == [0, 3, 6, 9] and c.n_rows == 10 and m2 is masks
    with pytest.raises(RuntimeError, match="not writable"):
        filtered.compact_block(filtered.CompactSpec("gather"), x, masks, info, in_place=True)


def test_uncompacted_block_refused_by_compacting_adapter(ccorpus):
    """A gather / union adapter handed whole blocks (a source without
    filter_compact) fails loudly instead of searching unfiltered rows."""
    path, queries, xq = ccorpus
    for name, mode, compact in (("static", "gather", "none"), ("per_query", "bitmap", "union")):
        filt = fsm.FilterAdapter(mode, CNQ, CK, kind="static" if name == "static" else "per_query",
                                 compact=compact)
        with ParquetSource(path, column="embedding", prefetch=0,
                           filter_job=cjob(name, queries)).start() as src:
            with pytest.raises(RuntimeError, match="compacted by"):
                loop.run_pass(src, _masked(xq), CNQ, CK, filt=filt)


def test_filter_compact_needs_a_filter(ccorpus):
    with pytest.raises(ValueError, match="filter_job"):
        ParquetSource(ccorpus[0], column="embedding",
                      filter_compact=filtered.CompactSpec("gather"))


# ---- FAISS (resident, torch's CPU device): gather with every transfer mode

torch = pytest.importorskip("torch")


def _faiss(monkeypatch, xq, h2d="pageable", overlap=False, d2h=None):
    import faiss
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    b = faiss_engine.GpuBackend(CK, device="cpu", events=True, metric="ip")

    def knn_gpu(res, q, xb_t, k, metric=None, **kw):
        Dn, In = faiss.knn(q.numpy(), xb_t.numpy(), k, metric=faiss.METRIC_INNER_PRODUCT)
        return torch.from_numpy(Dn), torch.from_numpy(In)

    monkeypatch.setattr(faiss, "knn_gpu", knn_gpu)
    if h2d != "pageable" or overlap:
        b.configure_h2d(h2d, overlap)
    if d2h is not None:
        b.configure_d2h("pinned", overlap=d2h == "overlap")
    b.setup(xq)
    return b


@pytest.mark.slow
@pytest.mark.parametrize("reader,h2d,overlap,d2h", [
    ("process", "pageable", False, None), ("process", "pinned", False, None),
    ("process", "pinned", True, None), ("process", "pinned", True, "overlap"),
    ("process", "pageable", False, "overlap"), ("inline", "pageable", False, None)])
def test_gather_faiss_transfer_modes_identical(monkeypatch, ccorpus, reader, h2d, overlap, d2h):
    """FAISS gather over reader-compacted blocks (pinned ring slots,
    uploaded ahead, copies back overlapped): identical to the old consumer
    gather on the same engine, and to the masked reference ids."""
    from test_h2d import _pinner
    path, queries, xq = ccorpus
    pin = None
    if h2d == "pinned":
        pin, cr = _pinner()
    slots = int(overlap) + int(d2h == "overlap")
    new = _cpass(path, queries, _faiss(monkeypatch, xq, h2d, overlap, d2h), CNQ, "static",
                 "gather", "queries", reader, "new", pin=pin, extra_slots=slots)
    old = _cpass(path, queries, _faiss(monkeypatch, xq), CNQ, "static", "gather", "queries",
                 reader, "old")
    np.testing.assert_array_equal(new[1], old[1])
    np.testing.assert_array_equal(new[0], old[0])
    assert new[3] == old[3]
    Dr, Ir = _reference(path, queries, xq, "static")
    np.testing.assert_array_equal(new[1], Ir)
    np.testing.assert_allclose(new[0], Dr, rtol=0, atol=1e-5)
    if overlap:
        # two searched files are adjacent (mixed, all7; rare9, small, one7):
        # some compacted block was uploaded ahead, from its ring slot. Whether
        # block i+1 is already read when search i starts is a race (the native
        # reads of these tiny files can finish later): up to 3 more passes
        ahead = new[2]["h2d_overlapped"]
        for _ in range(3):
            if ahead:
                break
            again = _cpass(path, queries, _faiss(monkeypatch, xq, h2d, overlap, d2h), CNQ,
                           "static", "gather", "queries", reader, "new", pin=pin,
                           extra_slots=slots)
            np.testing.assert_array_equal(again[1], old[1])
            ahead = again[2]["h2d_overlapped"]
        assert ahead > 0
    if pin is not None:
        assert pin.registered == {}   # every slot unpinned at close


def test_a_reader_compacted_block_given_to_a_non_compacting_adapter_is_refused():
    """The other direction of the adapter's wiring check: compacted local ids
    read as file rows would be wrong ids, silently."""
    import types

    from bench.data.filters import search as filtered_search
    for mode, kw in [("bitmap", dict(kind="per_query")), ("bitmap", dict(kind="static"))]:
        ad = filtered_search.FilterAdapter(mode, nq=2, k=1, **kw)
        blk = types.SimpleNamespace(filter=object(), path="f.parquet",
                                    compact=types.SimpleNamespace(mode="union"))
        with pytest.raises(RuntimeError, match="filter_compact and the adapter disagree"):
            ad.prepare(blk)
