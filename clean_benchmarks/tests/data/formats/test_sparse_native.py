"""--modality sparse --read-mode native: nova-bf's rust_fetch FETCH path
(Store.read_columns_fetched: rust_fetch downloads the footer and the sparse
column's chunks into a lazily allocated image, pyarrow decodes it), CPU only,
local parquet.

Checked: the packed block (header + CSR parts) is identical to the arrow mode's (both value
widths, compressions, row-group layouts, a file with other wide columns
around the sparse one) with NOVA_BF_RUST_FETCH_CANARY=1 (every unfetched byte
of the image is 0xA5, so a decode that touched one would differ or fail);
only the column's chunks + footer are fetched; the fetched table is the one
nova-bf's own Store.read_columns_fetched returns for the same file; host
widen; every reader kind; refusals fail loudly; and
run.main end to end with --verify (whose oracle reloads with pyarrow)."""

import csv

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

pytest.importorskip("rust_fetch")

import make_dummy  # noqa: E402
from bench.systems import harness_cli as run  # noqa: E402
from bench.record import results_io
from bench.data.formats import sparse  # noqa: E402
from bench.data import sources  # noqa: E402
from bench.data.sources import NativeRefused, ParquetSource, Stages, list_parquet, read_file  # noqa: E402

FS = pafs.LocalFileSystem()
COL = "sparse_embedding"
K = 10


@pytest.fixture(autouse=True)
def canary_and_clients(monkeypatch):
    monkeypatch.setenv("NOVA_BF_RUST_FETCH_CANARY", "1")
    yield
    sources.native_drop_clients()


def _write(path, rng, n, vocab=500, nnz=14, value_dtype=np.float32, rg=None,
           compression="snappy", wide=False, empty=0.03):
    parts = make_dummy.random_sparse(rng, n, vocab, nnz, empty)
    extra = None
    if wide:   # other columns around the sparse one: a dense vector + text
        extra = {"dense": pa.array(list(rng.standard_normal((n, 64)).astype(np.float32)),
                                   pa.list_(pa.float32())),
                 "text": pa.array([f"doc {i} " * 20 for i in range(n)])}
    make_dummy.write_sparse(path, parts, row_group_size=rg, compression=compression,
                            value_dtype=value_dtype, extra=extra)
    return parts


CASES = {  # name: (rows, value dtype, row-group size, compression, wide)
    "f32_snappy_rgs": (1300, np.float32, 300, "snappy", False),
    "f16_snappy_wide": (900, np.float16, 200, "snappy", True),
    "f32_none_onerg": (700, np.float32, None, "none", True),
    "f32_zstd_rgs": (1100, np.float32, 128, "zstd", False),
}


@pytest.fixture(params=sorted(CASES))
def case(request, tmp_path):
    rows, vt, rg, comp, wide = CASES[request.param]
    path = str(tmp_path / "a.parquet")
    parts = _write(path, np.random.default_rng(len(request.param)), rows, value_dtype=vt,
                   rg=rg, compression=comp, wide=wide)
    fs, refs = list_parquet(path)
    return fs, refs[0], parts, vt, wide


def same_block(a, b):
    """Two packed blocks hold the same CSR: identical header (sizes, offsets,
    widths) and identical indptr / indices / values. Raw bytes are NOT
    compared: the alignment padding between the parts is never written
    (sparse.new_packed), so it differs between any two allocations."""
    ha, hb = sparse._header(a), sparse._header(b)
    if not np.array_equal(ha, hb) or a.nbytes != b.nbytes:
        return False
    ca, cb = sparse.unpack(a), sparse.unpack(b)
    return all(getattr(ca, f).dtype == getattr(cb, f).dtype
               and np.array_equal(getattr(ca, f), getattr(cb, f))
               for f in ("indptr", "indices", "values"))


def _read(fs, ref, mode, **kw):
    st, rec = Stages(), {}
    x = read_file(fs, ref.path, COL, stats=st, size=ref.size, mode=mode, codec=sparse.CODEC,
                  rec=rec, **kw)
    return x, st.as_dict(), rec


def test_block_byte_identical_to_arrow_and_only_the_column_fetched(case):
    fs, ref, parts, vt, wide = case
    xa, _, _ = _read(fs, ref, "arrow")
    xn, st, rec = _read(fs, ref, "native")
    assert same_block(xn, xa), "packed block differs from arrow"
    c = sparse.unpack(xn)
    np.testing.assert_array_equal(c.indptr, parts[0])
    np.testing.assert_array_equal(c.indices, parts[1])
    np.testing.assert_array_equal(c.values, np.asarray(parts[2]).astype(vt))
    assert rec["read_path"] == "native-fetch" and st["native_fallbacks"] == 0
    assert st["native_read"] > 0 and st["rows"] == len(parts[0]) - 1
    foot = sources.read_footer(fs, ref.path, COL)
    assert st["bytes"] == foot.column_bytes          # as arrow / dense native report
    # files under nova-bf's 1 MiB first tail read are fetched whole by it
    assert foot.column_bytes <= st["fetched_bytes"] <= ref.size
    assert st["row_groups"] == foot.num_row_groups


def test_only_the_sparse_column_is_downloaded(tmp_path):
    """A multi-MiB file with wide dense + text columns around the sparse one:
    the fetch moves the footer tail + the sparse chunks only."""
    path = str(tmp_path / "w.parquet")
    rng = np.random.default_rng(21)
    n = 4000
    parts = make_dummy.random_sparse(rng, n, 500, 14, 0.0)
    extra = {"dense": pa.array(list(rng.standard_normal((n, 384)).astype(np.float32)),
                               pa.list_(pa.float32())),
             "text": pa.array([f"document number {i} " * 30 for i in range(n)])}
    make_dummy.write_sparse(path, parts, row_group_size=800, compression="snappy",
                            extra=extra)
    fs, refs = list_parquet(path)
    assert refs[0].size > 4 * sources._FETCH_TAIL
    xa, _, _ = _read(fs, refs[0], "arrow")
    xn, st, _ = _read(fs, refs[0], "native")
    assert same_block(xa, xn)
    foot = sources.read_footer(fs, refs[0].path, COL)
    assert st["bytes"] == foot.column_bytes
    assert foot.column_bytes <= st["fetched_bytes"] <= foot.column_bytes + sources._FETCH_TAIL \
        + 2 * sources._FETCH_PAD * foot.num_row_groups + sources._FETCH_COALESCE * 2 \
        * foot.num_row_groups
    assert st["fetched_bytes"] < 0.5 * refs[0].size, (st["fetched_bytes"], refs[0].size)


def test_same_table_as_nova_bf_store(case):
    """The harness's fetch returns exactly what nova-bf's
    Store.read_columns_fetched returns for the same file and column."""
    nova_io = pytest.importorskip("nova_bf.io")
    fs, ref, _, _, _ = case
    store = nova_io.Store(str(__import__("os").path.dirname(ref.path)))
    got = store.read_columns_fetched(ref.path, [COL])
    assert got is not None, "nova-bf declined its own fetch path on this file"
    t_nova, _ = got
    t_h, _, _, _, _ = sources._native_fetch(fs, ref.path, COL, sources.NATIVE_CONCURRENCY,
                                            sources.NATIVE_PART_BYTES)
    assert t_h.equals(t_nova)
    assert t_h.equals(pq.read_table(ref.path, columns=[COL]))


def test_host_widen_identical(tmp_path):
    path = str(tmp_path / "a.parquet")
    _write(path, np.random.default_rng(9), 800, value_dtype=np.float16, rg=150)
    fs, refs = list_parquet(path)
    wa, _, _ = _read(fs, refs[0], "arrow", dtype=np.float32)
    wn, st, rec = _read(fs, refs[0], "native", dtype=np.float32)
    assert sparse.unpack(wn).values.dtype == np.float32 and rec["widen"] > 0
    assert same_block(wa, wn)


@pytest.mark.parametrize("prefetch", [2, 1, 0])
def test_source_blocks_identical_to_arrow(tmp_path, prefetch):
    root = tmp_path / "c"
    root.mkdir()
    rows = [600, 2, 900, 1, 450]
    for i, n in enumerate(rows):
        _write(str(root / f"part-{i:05d}.parquet"), np.random.default_rng([3, i]), n,
               rg=250, wide=bool(i % 2))
    blocks = {}
    for mode in ("arrow", "native"):
        src = ParquetSource(str(root), COL, prefetch=prefetch, workers=2,
                            codec=sparse.CODEC, read_mode=mode)
        try:
            blocks[mode] = [(b.offset, b.vectors.copy(), (b.info or {}).get("read_path"))
                            for b in src]
        finally:
            src.close()
    assert [o for o, _, _ in blocks["native"]] == list(np.cumsum([0] + rows[:-1]))
    for (oa, xa, _), (on, xn, path) in zip(blocks["arrow"], blocks["native"]):
        assert oa == on and same_block(xa, xn)
        assert path == "native-fetch", path


def test_refused_file_fails_loudly(tmp_path):
    d = tmp_path / "part=1"            # hive k=v directory: io.py reads it with pyarrow
    d.mkdir()
    path = str(d / "a.parquet")
    _write(path, np.random.default_rng(2), 300)
    fs, refs = list_parquet(path)
    with pytest.raises(NativeRefused, match="nova-bf would read it with pyarrow"):
        _read(fs, refs[0], "native")


def test_not_parquet_refused(tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"x" * 5000)
    with pytest.raises(NativeRefused):
        sources._native_fetch(FS, str(path), COL, 8, 1 << 20)


def test_column_ranges_cover_only_the_column(tmp_path):
    path = str(tmp_path / "a.parquet")
    _write(path, np.random.default_rng(4), 500, rg=100, wide=True)
    size = FS.get_file_info(path).size
    image = np.frombuffer(open(path, "rb").read(), np.uint8).copy()
    ranges = sources._column_ranges(image, size, [COL])
    md = pq.read_metadata(path)
    chunks = [(md.row_group(i).column(j).data_page_offset if md.row_group(i).column(j)
               .dictionary_page_offset is None else md.row_group(i).column(j)
               .dictionary_page_offset, md.row_group(i).column(j).total_compressed_size)
              for i in range(md.num_row_groups) for j in range(md.num_columns)
              if md.row_group(i).column(j).path_in_schema.startswith(COL + ".")]
    covered = lambda a, n: any(o <= a and a + n <= o + ln for o, ln, _ in ranges)  # noqa: E731
    assert all(covered(o, n) for o, n in chunks)
    others = [(md.row_group(i).column(j).data_page_offset,
               md.row_group(i).column(j).total_compressed_size)
              for i in range(md.num_row_groups) for j in range(md.num_columns)
              if not md.row_group(i).column(j).path_in_schema.startswith(COL + ".")]
    assert sum(ln for _, ln, _ in ranges) < sum(n for _, n in chunks) + sum(n for _, n in others)


# ------------------------------------------------------------ run.main


def _corpus(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    parts = []
    for i, n in enumerate([700, 3, 500]):
        p = _write(str(root / f"part-{i:05d}.parquet"), np.random.default_rng([8, i]), n,
                   vocab=400, rg=200, wide=i == 0)
        parts.append(p)
    qp = make_dummy.make_sparse_queries(str(tmp_path / "q.parquet"), 30, 400, 6, seed=2,
                                        corpus_parts=parts, gt_k=K)
    return str(root), qp


def _args(cdir, qp, out, *extra):
    return ["--modality", "sparse", "--backend", "cpu", "--corpus", cdir, "--column", COL,
            "--query-file", qp, "--query-column", COL, "--k", str(K), "--csv", str(out),
            "--warmup", "0", "--repeats", "1", "--workers", "2", "--regime", "stream",
            *extra]


@pytest.mark.parametrize("extra", [["--prefetch", "2"], ["--prefetch", "0"]])
def test_main_sparse_native_verified(tmp_path, extra, capsys):
    cdir, qp = _corpus(tmp_path)
    out = tmp_path / "r.csv"
    run.main(_args(cdir, qp, out, "--read-mode", "native", "--verify", "--gt-ids-column",
                   "gt_ids", *extra))
    assert "verify OK" in capsys.readouterr().out
    with open(out, newline="") as f:
        r = list(csv.DictReader(f))[0]
    assert r["read_mode"] == "native" and r["modality"] == "sparse"
    assert int(r["native_fallbacks"]) == 0 and float(r["native_read_s"]) > 0
    assert float(r["gt_recall"]) == 1.0


def test_main_sparse_native_refuses(tmp_path):
    cdir, qp = _corpus(tmp_path)
    import os
    hive = tmp_path / "h" / "k=v"
    hive.mkdir(parents=True)
    for f in os.listdir(cdir):
        os.replace(os.path.join(cdir, f), hive / f)
    out = tmp_path / "r.csv"
    with pytest.raises(NativeRefused):
        run.main(_args(str(hive), qp, out, "--read-mode", "native", "--prefetch", "0"))


# ------------------------------------------------------- range planning at scale
#
# Files well over the 1 MiB first tail read, with wide columns on both sides of
# the sparse one, so the column's chunks are really planned and fetched by
# ranges (and, with the canary, any byte the decode reads but the fetch missed
# is 0xA5): every layout of CASES plus large_list / int64 indices, a
# dictionary-heavy low-cardinality column, and a footer larger than the tail.


def _big_table(rng, n, list_kind="list", index_type=pa.uint32(), value_dtype=np.float32,
               vocab=500, nnz=14, low_card=False, meta_bytes=0):
    parts = make_dummy.random_sparse(rng, n, vocab, nnz, 0.03)
    indptr, idx, val = parts
    if low_card:
        val = np.round(np.asarray(val) * 4) / 4      # few distinct values: dictionary pages
    mk = pa.LargeListArray if list_kind == "large_list" else pa.ListArray
    offs = pa.array(np.asarray(indptr, np.int64 if list_kind == "large_list" else np.int32))
    il = mk.from_arrays(offs, pa.array(np.asarray(idx).astype(index_type.to_pandas_dtype()),
                                       index_type))
    vl = mk.from_arrays(offs, pa.array(np.asarray(val, value_dtype)))
    col = pa.StructArray.from_arrays([il, vl], names=["indices", "values"])
    t = pa.table({"before": pa.array(list(rng.standard_normal((n, 200)).astype(np.float32)),
                                     pa.list_(pa.float32())),
                  COL: col,
                  "after": pa.array([f"text {i} " * 25 for i in range(n)])})
    if meta_bytes:
        t = t.replace_schema_metadata({b"pad": b"x" * meta_bytes})
    return t


BIG = {
    "f32_snappy": dict(),
    "f16_zstd": dict(value_dtype=np.float16, compression="zstd"),
    "f32_none": dict(compression="none"),
    "large_list_int64": dict(list_kind="large_list", index_type=pa.int64()),
    "int32_lowcard_dict": dict(index_type=pa.int32(), low_card=True),
    "footer_over_tail": dict(meta_bytes=3 << 20),
}


@pytest.mark.parametrize("name", sorted(BIG))
def test_big_files_ranges_planned_and_identical(tmp_path, name):
    kw = dict(BIG[name])
    comp = kw.pop("compression", "snappy")
    path = str(tmp_path / f"{name}.parquet")
    pq.write_table(_big_table(np.random.default_rng(len(name)), 3000, **kw), path,
                   compression=comp, row_group_size=600)
    fs, refs = list_parquet(path)
    ref = refs[0]
    assert ref.size > 2 * sources._FETCH_TAIL
    xa, _, _ = _read(fs, ref, "arrow")
    xn, st, rec = _read(fs, ref, "native")
    assert same_block(xa, xn) and rec["read_path"] == "native-fetch"
    foot = sources.read_footer(fs, ref.path, COL)
    assert st["bytes"] == foot.column_bytes and st["row_groups"] == foot.num_row_groups
    # fetched = the footer reads (>= the 1 MiB tail, more for a long footer) +
    # the column's chunks with their 4 KiB pads and <= 8 KiB coalesced holes
    with open(ref.path, "rb") as f:
        f.seek(ref.size - 8)
        flen = int.from_bytes(f.read(4), "little")
    footer_read = max(min(sources._FETCH_TAIL, ref.size), flen + 8)
    slack = (sources._FETCH_PAD + sources._FETCH_COALESCE) * 2 * foot.num_row_groups
    assert foot.column_bytes <= st["fetched_bytes"] <= footer_read + foot.column_bytes + slack
    assert st["fetched_bytes"] < 0.6 * ref.size or flen > sources._FETCH_TAIL
    nova_io = pytest.importorskip("nova_bf.io")
    got = nova_io.Store(str(tmp_path)).read_columns_fetched(ref.path, [COL])
    assert got is not None and got[0].equals(pq.read_table(ref.path, columns=[COL]))


def test_image_mmap_enomem_is_a_refusal(tmp_path, monkeypatch):
    import errno
    path = str(tmp_path / "a.parquet")
    pq.write_table(_big_table(np.random.default_rng(1), 1500), path, row_group_size=500)
    fs, refs = list_parquet(path)

    def boom(size):
        raise OSError(errno.ENOMEM, "Cannot allocate memory")
    monkeypatch.setattr(sources, "_zeroed_image", boom)
    with pytest.raises(NativeRefused, match="image"):
        _read(fs, refs[0], "native")



# ------------------------------------------------------- image page release


def _image_rss_kb(buf):
    """Resident kB of the mmap mapping holding pyarrow buffer ``buf`` (Linux
    /proc/self/smaps), or None if no mapping contains it."""
    addr = buf.address
    with open("/proc/self/smaps") as f:
        hit = False
        for line in f:
            head = line.split()
            if head and "-" in head[0] and len(head) >= 5 and ":" not in head[0]:
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                hit = lo <= addr < hi
            elif hit and line.startswith("Rss:"):
                return int(line.split()[1])
    return None


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/smaps"), reason="Linux only")
def test_image_pages_released_after_read(tmp_path):
    """io.py's _release_image: once decoded, the kept-alive image holds no
    resident pages (an 8+ MB uncompressed file: pages really were touched)."""
    path = str(tmp_path / "big.parquet")
    pq.write_table(_big_table(np.random.default_rng(5), 6000), path, compression="none",
                   row_group_size=1500)
    fs, refs = list_parquet(path)
    assert refs[0].size > 8 << 20
    _, st, _ = _read(fs, refs[0], "native")
    assert st["fetched_bytes"] > 1 << 20          # unreleased, that much would stay resident
    rss = _image_rss_kb(sources._FETCH_KEEPALIVE[-1])
    # the smaps entry can include small neighbouring anonymous allocations the
    # kernel merged into the same mapping (observed: up to ~150 kB), so the
    # bound is a third of the fetched bytes, not zero (unreleased: ~4.8 MB)
    assert rss is not None and rss * 1024 < st["fetched_bytes"] / 3, (rss, st["fetched_bytes"])


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/smaps"), reason="Linux only")
def test_image_released_on_an_empty_file(tmp_path):
    path = str(tmp_path / "empty.parquet")
    t = _big_table(np.random.default_rng(6), 2000)
    pq.write_table(t.slice(0, 0), path)
    big = str(tmp_path / "pad.parquet")   # make the empty file's image an mmap: >= 1 MiB
    pq.write_table(t.slice(0, 0).replace_schema_metadata({b"pad": b"y" * (2 << 20)}), big)
    fs, refs = list_parquet(big)
    with pytest.raises(ValueError, match="empty file"):
        _read(fs, refs[0], "native")
    rss = _image_rss_kb(sources._FETCH_KEEPALIVE[-1])
    assert rss is not None and rss < 1024, rss    # the 2+ MiB footer read, released


def test_missing_column_is_a_keyerror_like_arrow(tmp_path):
    path = str(tmp_path / "a.parquet")
    pq.write_table(_big_table(np.random.default_rng(7), 1500), path, row_group_size=500)
    fs, refs = list_parquet(path)
    for mode in ("arrow", "native"):
        with pytest.raises(KeyError, match="no column"):
            read_file(fs, refs[0].path, "absent_sparse", size=refs[0].size, mode=mode,
                      codec=sparse.CODEC, stats=Stages())


# ------------------------------------------------------- refusal paths release


@pytest.fixture
def big_file(tmp_path):
    path = str(tmp_path / "rf.parquet")
    pq.write_table(_big_table(np.random.default_rng(11), 6000), path, compression="none",
                   row_group_size=1500)
    fs, refs = list_parquet(path)
    return fs, refs[0]


def _client_failing(monkeypatch, exc):
    """The process's rust_fetch Client whose read_ranges_into raises ``exc``
    on every column-chunk download (each file's 2nd call: these test files'
    footers fit the 1 MiB tail, so the 1st is the footer read), after really
    writing the chunks into the image."""
    real = sources.native_client
    calls = {"n": 0}

    class Wrap:
        def __init__(self, c):
            self.c = c

        def __getattr__(self, a):
            return getattr(self.c, a)

        def read_ranges_into(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                self.c.read_ranges_into(*a, **kw)   # really fetch, then fail
                raise exc
            return self.c.read_ranges_into(*a, **kw)
    monkeypatch.setattr(sources, "native_client", lambda *a: Wrap(real(*a)))


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/smaps"), reason="Linux only")
@pytest.mark.parametrize("exc", [ValueError("bad range"), OSError("object changed (ETag)")])
def test_fetch_failure_refuses_and_releases(big_file, monkeypatch, exc):
    """rust_fetch raising mid-download (a refusal type, or the object changing
    under the read): NativeRefused (the read fails; io.py would fall back),
    and the image's already-fetched pages are released."""
    fs, ref = big_file
    _client_failing(monkeypatch, exc)
    with pytest.raises(NativeRefused):
        _read(fs, ref, "native")


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/smaps"), reason="Linux only")
def test_fetch_failure_pages_released_measured(big_file, monkeypatch):
    """The image of a fetch that failed AFTER the column chunks were written
    holds no resident pages (would be MBs unreleased)."""
    fs, ref = big_file
    images = []
    real_zi = sources._zeroed_image
    monkeypatch.setattr(sources, "_zeroed_image", lambda n: images.append(real_zi(n)) or images[-1])
    _client_failing(monkeypatch, ValueError("bad range"))
    with pytest.raises(NativeRefused):
        _read(fs, ref, "native")
    img = images[-1]
    rss = _image_rss_kb(pa.py_buffer(img))
    assert rss is not None and rss * 1024 < 1 << 20, rss


@pytest.mark.skipif(not __import__("os").path.exists("/proc/self/smaps"), reason="Linux only")
@pytest.mark.parametrize("path_kind", ["missing_column", "read_table_fails"])
def test_other_failure_paths_release(big_file, monkeypatch, path_kind):
    fs, ref = big_file
    images = []
    real_zi = sources._zeroed_image
    monkeypatch.setattr(sources, "_zeroed_image", lambda n: images.append(real_zi(n)) or images[-1])
    if path_kind == "missing_column":
        with pytest.raises(KeyError):
            read_file(fs, ref.path, "absent", size=ref.size, mode="native", codec=sparse.CODEC)
    else:
        real_rt = sources.pq.read_table

        def boom(src, *a, **kw):
            if isinstance(src, pa.BufferReader):
                raise OSError("corrupt page")
            return real_rt(src, *a, **kw)
        monkeypatch.setattr(sources.pq, "read_table", boom)
        with pytest.raises(NativeRefused):
            _read(fs, ref, "native")
    rss = _image_rss_kb(pa.py_buffer(images[-1]))
    assert rss is not None and rss * 1024 < 1 << 20, (path_kind, rss)


def test_trace_records_native_read(tmp_path):
    cdir, qp = _corpus(tmp_path)
    out = tmp_path / "r.csv"
    run.main(_args(cdir, qp, out, "--read-mode", "native", "--prefetch", "2"))
    with open(results_io.trace_path(str(out)), newline="") as f:
        t = list(csv.DictReader(f))
    assert t and all(r["read_path"] == "native-fetch" and float(r["native_read_s"]) > 0
                     for r in t)
