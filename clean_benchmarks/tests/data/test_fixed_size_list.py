"""fixed_size_list<float|halffloat|double>[d] vector columns (the Pes2o-VE
layout: fixed_size_list<float>[2560], UNCOMPRESSED, 8,192-row row groups)
work everywhere list / large_list do: footer, decode (both paths), every
reader kind and read mode, --verify, --query-file."""
import argparse
import csv
import mmap
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import make_dummy
from bench.systems import harness_cli as run
from bench.systems import guards, loop
from bench.record import results_io
from bench.systems import cpu_reference
from bench.data import sources
from bench.data.sources import (InMemorySource, ParquetSource, block_dtype, list_parquet,
                     read_file, read_footer, shm_segments, table_vectors)
from test_run import MEASURE, check_against_oneshot

K = 10
D = 24
FROWS = [700, 3, 1100, 450]  # one file smaller than k
RG = 128  # many small row groups, as Pes2o-VE's 8,192 of 1,000,000


def fsl(x, vt=pa.float32()):
    """``x`` (n, d) as a fixed_size_list<vt>[d] array."""
    n, d = x.shape
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), vt), d)


def write_fixed_corpus(out, value_dtype=np.float32, rows=FROWS, d=D, seed=13,
                       list_types=None):
    """Uncompressed fixed_size_list corpus (``list_types`` per file
    overrides: e.g. ["fixed", "list", ...] for a mixed corpus); returns the
    concatenated vectors as stored (fp16-rounded for f16), widened to f32."""
    os.makedirs(out, exist_ok=True)
    xs = []
    for i, n in enumerate(rows):
        x = make_dummy.unit_vectors(np.random.default_rng([seed, i]), n, d)
        x = x.astype(value_dtype)
        lt = list_types[i] if list_types else "fixed"
        make_dummy.write_vectors(os.path.join(out, f"part-{i:05d}.parquet"), x,
                                 row_group_size=RG, compression="none",
                                 value_dtype=value_dtype, list_type=lt)
        xs.append(x.astype(np.float32))
    return np.concatenate(xs)


@pytest.fixture(scope="module", params=["f32", "f16"])
def fixed(request, tmp_path_factory):
    vt = make_dummy.VALUE_TYPES[request.param]
    root = tmp_path_factory.mktemp(f"fsl-{request.param}")
    xb = write_fixed_corpus(str(root / "c"), vt)
    q = make_dummy.make_queries(str(root / "q.parquet"), 40, D, list_type="fixed")
    return request.param, str(root / "c"), xb, q


# ------------------------------------------------------------- make_dummy

def test_make_dummy_fixed_layout(tmp_path):
    for vt, arrow_vt in (("f32", pa.float32()), ("f16", pa.float16())):
        out = tmp_path / vt
        sys_argv = [str(out), "--n-files", "1", "--rows-per-file", "100", "--dim", "8",
                    "--list-type", "fixed", "--compression", "none",
                    "--row-group-size", "30", "--value-type", vt,
                    "--queries-out", str(tmp_path / f"q{vt}.parquet"), "--nq", "5"]
        old = sys.argv
        sys.argv = ["make_dummy.py", *sys_argv]
        try:
            make_dummy.main()
        finally:
            sys.argv = old
        p = str(out / "part-00000.parquet")
        md = pq.ParquetFile(p).metadata
        assert md.num_row_groups == 4
        assert md.row_group(0).column(0).compression == "UNCOMPRESSED"
        assert pq.read_schema(p).field("embedding").type == pa.list_(arrow_vt, 8)
        # queries are always f32, in the chosen layout
        assert pq.read_schema(str(tmp_path / f"q{vt}.parquet")).field(
            "embedding").type == pa.list_(pa.float32(), 8)
    with pytest.raises(ValueError, match="list type"):
        make_dummy.vectors_table(np.ones((2, 2), np.float32), list_type="ragged")


def test_fixed_skips_the_int32_offset_guard(monkeypatch):
    """No offsets buffer, nothing to overflow: the guard is list-only."""
    calls = []
    monkeypatch.setattr(make_dummy, "check_int32_offsets", lambda *a: calls.append(a))
    make_dummy.vectors_table(np.ones((3, 2), np.float32), list_type="fixed")
    assert calls == []
    make_dummy.vectors_table(np.ones((3, 2), np.float32), list_type="list")
    assert len(calls) == 1


# ------------------------------------------------------------------ footer

@pytest.mark.parametrize("vt,bits", [(np.float16, 16), (np.float32, 32), (np.float64, 64)])
def test_footer_dim_itemsize_and_column_share(tmp_path, vt, bits):
    p = str(tmp_path / "a.parquet")
    make_dummy.write_vectors(p, np.ones((3000, 64), np.float32), row_group_size=640,
                             compression="none", value_dtype=vt, list_type="fixed")
    fs, refs = list_parquet(p)
    foot = read_footer(fs, p, "embedding")
    assert (foot.num_rows, foot.num_row_groups, foot.dim) == (3000, 5, 64)
    assert foot.value_bits == bits
    assert sources._block_itemsize(foot, None) == {16: 2, 32: 4, 64: 4}[bits]
    # vectors-only, uncompressed: the column is ~all of the file
    mode, share = sources.choose_read_mode("native", foot, refs[0].size)
    assert mode == "native" and share > 0.9


def test_footer_dim_from_the_type_not_the_leaf_count(tmp_path):
    """list_size is authoritative: a null element (one leaf value fewer
    stored) does not change d."""
    p = str(tmp_path / "a.parquet")
    a = pa.array([[1.0, None, 3.0], [4.0, 5.0, 6.0]], pa.list_(pa.float32(), 3))
    pq.write_table(pa.table({"embedding": a}), p)
    assert read_footer(*list_parquet(p)[:1], p, "embedding").dim == 3


# ------------------------------------------------------------------ decode

@pytest.mark.parametrize("vt", [pa.float16(), pa.float32(), pa.float64()])
def test_block_dtype(vt):
    want = np.float16 if vt == pa.float16() else np.float32
    assert block_dtype(pa.list_(vt, 7)) == want
    assert block_dtype(pa.list_(vt)) == want


def test_table_vectors_one_chunk_no_copy_past_combine(monkeypatch):
    """flatten / to_numpy / reshape add no copy of their own: the block
    aliases the (combined) array's values buffer, as for list<...>."""
    x = np.arange(60, dtype=np.float32).reshape(10, 6)
    seen = []
    real = sources._combine_chunks
    monkeypatch.setattr(sources, "_combine_chunks",
                        lambda c: seen.append(real(c)) or seen[-1])
    got = table_vectors(pa.table({"e": fsl(x)}), "e")
    np.testing.assert_array_equal(got, x)
    assert got.flags.c_contiguous


@pytest.mark.parametrize("vt", [pa.float16(), pa.float32(), pa.float64()])
def test_table_vectors_sliced_chunks_both_paths(vt):
    """Several chunks, each a SLICE of a bigger array (non-zero offset): the
    combine path and the out=slot path give the same rows; f64 is cast to
    f32 on the way, as for list<double>."""
    x = np.random.default_rng(3).standard_normal((50, 5)).astype(np.float32)
    big = fsl(x, vt)
    pieces = [big.slice(3, 10), big.slice(20, 1), big.slice(0, 0), big.slice(30, 17)]
    want = np.concatenate([x[3:13], x[20:21], x[30:47]]).astype(
        block_dtype(big.type)).astype(np.float32)
    t = pa.table({"e": pa.chunked_array(pieces, big.type)})
    a = table_vectors(t, "e")
    slot = bytearray(4096)
    b = table_vectors(t, "e", out=memoryview(slot))
    assert b.base is not None and b.shape == (28, 5) and b.dtype == block_dtype(big.type)
    for got in (a, b):
        np.testing.assert_array_equal(got.astype(np.float32), want)
    # the out path wrote the rows into the START of the slot
    np.testing.assert_array_equal(
        np.frombuffer(slot, b.dtype, b.size).reshape(b.shape), b)


def test_out_slot_too_small_is_refused():
    t = pa.table({"e": fsl(np.ones((10, 4), np.float32))})
    with pytest.raises(ValueError, match="ring slot"):
        table_vectors(t, "e", out=memoryview(bytearray(100)))


def test_null_rows_and_elements_rejected(tmp_path):
    typ = pa.list_(pa.float32(), 2)
    rows = pa.table({"e": pa.array([[1.0, 2.0], None, [3.0, 4.0]], typ)})
    elems = pa.table({"e": pa.array([[1.0, 2.0], [3.0, None]], typ)})
    for t in (rows, elems):
        for out in (None, memoryview(bytearray(256))):
            with pytest.raises(ValueError, match="null"):
                table_vectors(t, "e", out=out)
    # through a file (parquet cannot store a null fixed_size_list ROW whose
    # child slots are not null, so only the element case is writable here)
    p = str(tmp_path / "e.parquet")
    pq.write_table(elems, p)
    with pytest.raises(ValueError, match="null"):
        list(ParquetSource(p, column="e", prefetch=0))


def test_wrong_type_error_names_fixed_size_list():
    with pytest.raises(TypeError, match="fixed_size_list"):
        table_vectors(pa.table({"e": pa.array([1.0, 2.0])}), "e")


@pytest.mark.parametrize("mode", ["arrow", "native"])
def test_read_file_modes_and_out_slot(tmp_path, mode):
    x = make_dummy.unit_vectors(np.random.default_rng(4), 1000, 12)
    p = str(tmp_path / "a.parquet")
    make_dummy.write_vectors(p, x, row_group_size=RG, compression="none",
                             list_type="fixed")
    fs, refs = list_parquet(p)
    np.testing.assert_array_equal(read_file(fs, p, "embedding", size=refs[0].size,
                                            mode=mode), x)
    raw = bytearray(1000 * 12 * 4 + 64)
    y = read_file(fs, p, "embedding", dtype=np.float32, size=refs[0].size, mode=mode,
                  out=memoryview(raw))
    # decoded straight into the start of the slot
    assert y.ctypes.data == np.frombuffer(raw, np.uint8).ctypes.data
    np.testing.assert_array_equal(y, x)


@pytest.mark.parametrize("prefetch", [3, 0])
def test_parquet_source_readers(fixed, prefetch):
    _, path, xb, _ = fixed
    src = ParquetSource(path, prefetch=prefetch, workers=2)
    try:
        blocks = [(b.offset, b.vectors.copy()) for b in src]
    finally:
        src.close()
    assert [o for o, _ in blocks] == list(np.cumsum([0] + FROWS[:-1]))
    np.testing.assert_array_equal(np.concatenate([v for _, v in blocks]).astype(np.float32), xb)
    assert src.stats.get("row_groups") == sum(-(-n // RG) for n in FROWS)
    assert shm_segments(sources.SHM_PREFIX + str(os.getpid())) == []


# -------------------------------------------------------------- end to end

@pytest.mark.parametrize("prefetch", ["3", "0"])   # process readers, inline
def test_cli_end_to_end_fixed(fixed, tmp_path, capsys, prefetch):
    vt, path, xb, q = fixed
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu",
              "--query-file", q, "--nq", "30", "--k", str(K), "--warmup", "1",
              "--repeats", "1", "--csv", str(out), "--omp-threads", "2",
              "--prefetch", prefetch, "--workers", "2",
              "--read-mode", "native", "--verify"])
    assert "verify OK" in capsys.readouterr().out
    rows = list(csv.DictReader(open(out)))
    assert {r["verified"] for r in rows} == {"ok"}
    r = rows[-1]
    assert r["d"] == str(D) and int(r["rows"]) == sum(FROWS)
    assert r["stored_dtype"] == vt and r["dtype"] == "f32"
    assert r["read_mode_effective"] == "native"
    assert r["row_groups"] == str(sum(-(-n // RG) for n in FROWS))
    assert r["reader"] == ("inline" if prefetch == "0" else "process")
    assert shm_segments(sources.SHM_PREFIX + str(os.getpid())) == []


def test_results_match_oneshot_and_the_list_copy(fixed, tmp_path):
    """A fixed_size_list corpus searches exactly like the same vectors stored
    as list<...>, and both match a one-shot search."""
    vt, path, xb, q = fixed
    lst = str(tmp_path / "lst")
    xl = write_fixed_corpus(lst, make_dummy.VALUE_TYPES[vt],
                            list_types=["list"] * len(FROWS))
    np.testing.assert_array_equal(xl, xb)
    xq = results_io.load_queries(argparse.Namespace(query_file=q, query_column="embedding",
                                             nq=0), np.float32)
    b = cpu_reference.CpuBackend(K)
    b.setup(xq)
    out = []
    for p in (path, lst):
        src = ParquetSource(p, prefetch=2, workers=2, dtype=np.float32)
        try:
            out.append(loop.run_pass(src, b, len(xq), K, d=D))
        finally:
            src.close()
    np.testing.assert_array_equal(out[0][0], out[1][0])
    np.testing.assert_array_equal(out[0][1], out[1][1])
    check_against_oneshot(out[0][0], out[0][1], xq, xb)


# ------------------------------------------------------ mixed corpora / d

def test_mixed_list_and_fixed_corpus_allowed(tmp_path):
    """list and fixed_size_list files in ONE corpus: allowed when d (and so
    every block's shape) agrees; each file is decoded on its own."""
    path = str(tmp_path / "mix")
    xb = write_fixed_corpus(path, d=32, list_types=["fixed", "list", "fixed", "list"])
    out = tmp_path / "r.csv"
    for pf in ("0", "3"):
        run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu",
                  "--nq", "16", "--warmup", "0", "--repeats", "1", "--csv", str(out),
                  "--prefetch", pf, "--workers", "2", "--verify"])
    assert {r["verified"] for r in csv.DictReader(open(out))} == {"ok"}
    src = InMemorySource.load(path)
    np.testing.assert_array_equal(np.concatenate([b.vectors for b in src.blocks]), xb)


@pytest.mark.parametrize("prefetch", ["2", "0"])
def test_mixed_d_refused(tmp_path, prefetch):
    """A fixed_size_list file whose d differs from the others is refused, as
    for list files."""
    path = str(tmp_path / "c")
    os.makedirs(path)
    make_dummy.write_vectors(os.path.join(path, "part-00000.parquet"),
                             np.ones((20, 8), np.float32), list_type="list")
    make_dummy.write_vectors(os.path.join(path, "part-00001.parquet"),
                             np.ones((20, 12), np.float32), list_type="fixed")
    with pytest.raises(ValueError, match="12-D"):
        run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu",
                  "--nq", "4", "--warmup", "0", "--repeats", "1",
                  "--prefetch", prefetch, "--workers", "2",
                  "--csv", str(tmp_path / "r.csv")])
    assert shm_segments(sources.SHM_PREFIX + str(os.getpid())) == []


def test_fixed_query_dim_mismatch_refused(fixed, tmp_path):
    _, path, _, _ = fixed
    q = make_dummy.make_queries(str(tmp_path / "q.parquet"), 10, 16, list_type="fixed")
    with pytest.raises(ValueError, match=f"queries are 16-D, corpus is {D}-D"):
        run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu",
                  "--query-file", q, "--csv", str(tmp_path / "r.csv")])


def test_pes2o_style_query_file_column(tmp_path, fixed):
    """Pes2o-VE's labelled_queries.parquet: the vectors in ``query`` next to
    fixed_size_list ground-truth columns; --query-column picks the vectors."""
    _, path, xb, _ = fixed
    xq = make_dummy.unit_vectors(np.random.default_rng(8), 12, D)
    qp = str(tmp_path / "labelled.parquet")
    pq.write_table(pa.table({
        "query_id": pa.array(np.arange(12, dtype=np.int64)),
        "query": fsl(xq),
        "ids": fsl(np.zeros((12, 5), np.int64), pa.int64()),
        "distances": fsl(np.zeros((12, 5), np.float32)),
    }), qp, compression="zstd")
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu",
              "--query-file", qp, "--query-column", "query", "--k", "5",
              "--warmup", "0", "--repeats", "1", "--csv", str(out), "--verify"])
    r = next(csv.DictReader(open(out)))
    assert r["nq"] == "12" and r["query_column"] == "query" and r["verified"] == "ok"


# ---------------------------------------------------------- memory guard

@pytest.mark.slow
@pytest.mark.parametrize("vt", ["f16", "f32"])
def test_verify_memory_estimate_covers_the_fixed_oracle(tmp_path, monkeypatch, vt):
    """test_run's guard-vs-measurement check on an uncompressed,
    many-row-group fixed_size_list corpus: the guard (whose per-value decode
    figures were measured on list<...>) still covers what the oracle
    actually allocates."""
    dtype, n = {"f16": (np.float16, 4000), "f32": (np.float32, 20000)}[vt]
    corpus = str(tmp_path / "c")
    make_dummy.make_corpus(corpus, 3, n, 128, seed=2, value_dtype=dtype,
                           compression="none", row_group_size=2048, list_type="fixed")
    harness = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run([sys.executable, "-c", MEASURE, harness, corpus, "arrow", "16", "5"],
                         capture_output=True, text=True, check=True)
    measured = int(res.stdout)
    fs, files = sources.list_parquet(corpus)
    footers = {r.path: sources.read_footer(fs, r.path, "embedding") for r in files}
    assert {f.dim for f in footers.values()} == {128}
    args = argparse.Namespace(prefetch=0, k=5, warmup=0, repeats=0, workers=2, backend="cpu")
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: measured - 1)
    with pytest.raises(SystemExit, match="decode transient"):
        guards.verify_memory_guard(args, files, footers, 128, 16, None)


# A Pes2o-VE-shaped file at test scale: fixed_size_list<float>[512],
# UNCOMPRESSED, 50 row groups (Pes2o-VE: 50 of 8,192 rows per 409,600-row
# file), 20,000 rows (41 MB of vectors)
PES2O_ROWS, PES2O_D, PES2O_RG = 20000, 512, 400



PEAK_RUN = r"""
import contextlib, io, os, sys
sys.path.insert(0, sys.argv[1])
from bench.systems import harness_cli as run

def status(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024

with open("/proc/self/clear_refs", "w") as f:
    f.write("5")
rss0 = status("VmRSS")
with contextlib.redirect_stdout(io.StringIO()):
    run.main(sys.argv[2:])
print(status("VmHWM") - rss0, flush=True)
os._exit(0)
"""


@pytest.mark.slow
def test_verify_guard_tracks_measured_fixed_peak(tmp_path, monkeypatch):
    """A whole --verify run (CPU backend, --prefetch 0, native reads) on
    a Pes2o-VE-shaped file: the guard's estimate is at least the run
    process's real peak (VmHWM increase over run.main, fresh process) and at
    most 2x it. The old guard (list<...> figures, phases summed) charged
    2.1-3.0x this peak."""
    corpus = str(tmp_path / "c")
    make_dummy.make_corpus(corpus, 1, PES2O_ROWS, PES2O_D, seed=4, compression="none",
                           row_group_size=PES2O_RG, list_type="fixed")
    argv = ["--regime", "stream", "--corpus", corpus, "--backend", "cpu", "--nq", "16",
            "--k", "5", "--warmup", "0", "--repeats", "1", "--verify", "--prefetch", "0",
            "--read-mode", "native", "--csv", str(tmp_path / "r.csv")]
    harness = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run([sys.executable, "-c", PEAK_RUN, harness, *argv],
                         capture_output=True, text=True, check=True)
    measured = int(res.stdout)
    assert next(csv.DictReader(open(tmp_path / "r.csv")))["verified"] == "ok"
    seen, real = [], guards.verify_memory_need
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: 0)
    monkeypatch.setattr(guards, "verify_memory_need",
                        lambda *a, **kw: seen.append(real(*a, **kw)) or seen[-1])
    fs, files = list_parquet(corpus)
    footers = {r.path: read_footer(fs, r.path, "embedding") for r in files}
    with pytest.raises(SystemExit, match="needs about"):
        guards.verify_memory_guard(run.parse_args(argv), files, footers, PES2O_D, 16, None)
    assert measured <= seen[-1] <= 2 * measured, (seen[-1], measured)


PRELOAD_RSS = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
import pyarrow as pa
from bench.data.sources import InMemorySource, configure_arrow_pools, list_parquet

def status(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024

configure_arrow_pools(int(sys.argv[4]), None)
fs, files = list_parquet(sys.argv[2])
w = InMemorySource.load(sys.argv[2], "embedding", None, sys.argv[3], fs=fs, files=files[:1])
del w  # warm-up: lazy imports and first-read costs off the books
pool = pa.default_memory_pool()
pool.release_unused()
with open("/proc/self/clear_refs", "w") as f:
    f.write("5")
rss0 = status("VmRSS")
src = InMemorySource.load(sys.argv[2], "embedding", None, sys.argv[3], fs=fs, files=files)
print(status("VmHWM") - rss0, status("VmRSS") - rss0, pool.bytes_allocated(), flush=True)
os._exit(0)
"""


@pytest.mark.slow
def test_memory_preload_holds_about_one_corpus(tmp_path):
    """The --verify reload (3 x 30,000 x 768 f32 fixed_size_list,
    8 row groups each, 8 decode threads, fresh process): once loaded, the
    process holds about the corpus (RSS increase <= corpus + one block), and
    Arrow's pool holds the blocks and nothing else (each block is one
    page-aligned Arrow buffer, arrow_array, the row-group batches were copied
    into; one spare page each to align in). Before, every block lived in Arrow's pool and
    the pool kept each file's whole decode transient: RSS stayed at the
    preload's peak, corpus + 2-6 blocks. The peak is the corpus + the
    in-flight batches."""
    corpus = str(tmp_path / "c")
    rows, d = 30000, 768
    make_dummy.make_corpus(corpus, 3, rows, d, seed=5, compression="none",
                           row_group_size=3750, list_type="fixed")
    harness = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res = subprocess.run([sys.executable, "-c", PRELOAD_RSS, harness, corpus, "arrow", "8"],
                         capture_output=True, text=True, check=True)
    peak, after, pooled = map(int, res.stdout.split())
    block = rows * d * 4
    assert 3 * block <= pooled <= 3 * (block + mmap.PAGESIZE)
    assert after <= 3 * block + block, (after, block)
    # the peak: the corpus + the guard's preload transient (8 threads' record
    # batches) covers it
    fs, files = list_parquet(corpus)
    footers = {r.path: read_footer(fs, r.path, "embedding") for r in files}
    assert peak <= 3 * block + guards.preload_transient(files, footers, d, 4, 8), \
        (peak, block)


@pytest.mark.slow
@pytest.mark.parametrize("vt", ["f16", "f32", "f64"])
@pytest.mark.parametrize("lt", ["fixed", "list"])
def test_preload_batches_match_whole_file_read(tmp_path, vt, lt):
    """The preload's row-group / record-batch decode (tiny batches here, so
    every row group is many batches, several row groups at once) gives
    bit-identical blocks to the whole-file read the timed passes use, stored
    dtype or a host cast (f64 widened to f32 like table_vectors; to f16, f64
    through f32 like the stream path), and to the reader processes' blocks
    (decoded straight into their ring slots when there is no cast, the
    host-cast copy published into them otherwise). The f64 values are real
    f64 (not f32 values widened), so a direct f64 -> f16 cast would round
    differently."""
    mode = "arrow"
    out = str(tmp_path / "c")
    os.makedirs(out)
    dt = {"f16": np.float16, "f32": np.float32, "f64": np.float64}[vt]
    for i, n in enumerate([300, 7, 450]):
        rng = np.random.default_rng([9, i])
        x = make_dummy.unit_vectors(rng, n, D).astype(dt)
        if dt == np.float64:
            x += rng.standard_normal(x.shape) * 1e-5
        flat = pa.array(x.reshape(-1))
        col = fsl(x, flat.type) if lt == "fixed" else pa.ListArray.from_arrays(
            pa.array(np.arange(0, (n + 1) * D, D, dtype=np.int32)), flat)
        pq.write_table(pa.table({"embedding": col}),
                       os.path.join(out, f"part-{i:05d}.parquet"), row_group_size=64)
    for dtype in (None, np.float32, np.float16):
        a = InMemorySource.load(out, dtype=dtype, read_mode=mode, batch_bytes=0)
        b = InMemorySource.load(out, dtype=dtype, read_mode=mode, batch_bytes=7 * D * 4)
        src = ParquetSource(out, prefetch=2, workers=2, dtype=dtype, read_mode=mode).start()
        try:
            c = [(blk.offset, blk.vectors.copy()) for blk in src]
        finally:
            src.close()
        assert [x.offset for x in b.blocks] == [x.offset for x in a.blocks] == \
            [o for o, _ in c]
        for x, y, (_, z) in zip(a.blocks, b.blocks, c):
            for v in (y.vectors, z):
                assert v.dtype == x.vectors.dtype and v.flags.c_contiguous
                np.testing.assert_array_equal(v.view(np.uint8), x.vectors.view(np.uint8))


def test_preload_blocks_are_page_aligned_arrow_buffers(tmp_path):
    """--verify's reloaded blocks sit in
    Arrow-allocated, page-aligned buffers (arrow_array), like the stream
    path's blocks (Arrow buffers; page-aligned tmpfs ring slots): not numpy
    allocations, which numpy advises MADV_HUGEPAGE and starts 16 B past a
    page boundary. Writable, C-contiguous, and freed with the block."""
    out = str(tmp_path / "c")
    make_dummy.make_corpus(out, 2, 5000, 256, seed=3, list_type="fixed")
    for mode in ("arrow",):
        for dtype in (None, np.float16):
            src = InMemorySource.load(out, dtype=dtype, read_mode=mode)
            for blk in src.blocks:
                v = blk.vectors
                assert v.ctypes.data % mmap.PAGESIZE == 0
                assert not v.flags.owndata and v.flags.writeable and v.flags.c_contiguous
                base = v
                while isinstance(base, np.ndarray):
                    base = base.base
                assert isinstance(base, pa.Buffer), type(base)
    x = sources.arrow_array((3, 5), np.float32)
    assert x.shape == (3, 5) and x.dtype == np.float32 and x.ctypes.data % mmap.PAGESIZE == 0


def test_preload_empty_lists_error_like_the_stream_read(tmp_path):
    """A column whose rows are all empty lists: the preload and the timed
    whole-file read fail alike (d = 0), not with the preload's footer
    "ragged" message (the footer reads d = 1 off the empty lists' leaves)."""
    p = tmp_path / "e"
    p.mkdir()
    pq.write_table(pa.table({"embedding": pa.array([[], [], []], pa.list_(pa.float32()))}),
                   str(p / "f.parquet"))
    for bb in (0, sources.PRELOAD_BATCH_BYTES):
        with pytest.raises(ValueError, match=r"empty lists \(d = 0\)"):
            InMemorySource.load(str(p), batch_bytes=bb)


def test_preload_rejects_ragged_and_null_rows(tmp_path):
    """The preload's per-batch decode keeps the whole-file read's checks:
    ragged rows (within a row group, or a d that changes between row
    groups), null rows and null elements are errors."""
    def write(name, arr, rg=2):
        p = tmp_path / name
        p.mkdir()
        pq.write_table(pa.table({"embedding": arr}), str(p / "f.parquet"), row_group_size=rg)
        return str(p)

    lt = pa.list_(pa.float32())
    cases = {
        "ragged": ([[1.0, 2.0], [1.0]], "ragged"),
        "ragged_rg": ([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "ragged"),
        "null_row": ([[1.0, 2.0], None], "null"),
        "null_elem": ([[1.0, 2.0], [None, 2.0]], "null"),
    }
    for name, (rows, match) in cases.items():
        with pytest.raises(ValueError, match=match):
            InMemorySource.load(write(name, pa.array(rows, lt)))


def test_pes2o_verify_fits_a_g5_8xlarge(monkeypatch):
    """The real-data case: 2-3 Pes2o-VE files (409,600 x 2560 f32
    fixed_size_list, UNCOMPRESSED, 50 row groups, 4.19 GB) streamed at the
    defaults (--prefetch 2, two reader processes, native reads),
    20,000 queries, k=1000, 4 passes, GPU: a g5.8xlarge (32 vCPUs,
    MemAvailable 117.65 GiB) must be allowed to --verify them (the passes
    peak at ~44 GB: the 3-slot ring + two readers' slot transients; the
    oracle at the corpus + the reload's transient, the float64 oracle's
    ~1.1 GB of queries, D / I and tiles below it). The guard refused 2 files
    at ~134 GiB (list<...> per-value figures, a per-m fit charging a whole
    block per row group in flight, and the passes and the oracle summed).
    FIT_MAX = 22 files fit now with native reads (the fp32 faiss.knn oracle's
    corpus copy, n x 4.19 GB, refused anything past 11); 23 do not."""
    FIT_MAX = 22
    monkeypatch.setattr(guards, "usable_cpu_count", lambda: 32)
    monkeypatch.setattr(sources, "usable_cpu_count", lambda: 32)
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: int(117.65 * 2**30))

    def guard(n):
        files = [sources.FileRef(f"/pes2o/part-{i}.parquet", 4_194_700_000) for i in range(n)]
        footers = {r.path: sources.Footer(409600, 50, 4_194_300_000, 2560, "float", 32, 8192,
                                          "fixed_size_list", "UNCOMPRESSED") for r in files}
        args = argparse.Namespace(prefetch=2, k=1000, warmup=1, repeats=3, workers=31,
                                  backend="gpu", metric="ip")
        guards.verify_memory_guard(args, files, footers, 2560, 20000, None)

    for n in (2, 3):
        guard(n)  # no SystemExit
    guard(FIT_MAX)  # the old fp32-copy accounting refused anything past 11
    with pytest.raises(SystemExit, match="needs about"):
        guard(FIT_MAX + 1)


def test_verify_guard_applies_the_safety_factor(monkeypatch):
    """The guard refuses when the raw estimate fits but the estimate x
    VERIFY_SAFETY does not (the node peaks came within 3-10% of the raw
    estimate), and its message names the factor."""
    monkeypatch.setattr(guards, "usable_cpu_count", lambda: 32)
    monkeypatch.setattr(sources, "usable_cpu_count", lambda: 32)
    files = [sources.FileRef(f"/pes2o/part-{i}.parquet", 4_194_700_000) for i in range(2)]
    footers = {r.path: sources.Footer(409600, 50, 4_194_300_000, 2560, "float", 32, 8192,
                                      "fixed_size_list", "UNCOMPRESSED") for r in files}
    args = argparse.Namespace(prefetch=2, k=1000, warmup=1, repeats=3, workers=31,
                              backend="gpu", metric="cosine")
    captured = {}
    real_need = guards.verify_memory_need

    def spy(*a, **kw):
        captured["raw"] = real_need(*a, **kw)
        return captured["raw"]
    monkeypatch.setattr(guards, "verify_memory_need", spy)
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: 10**15)
    guards.verify_memory_guard(args, files, footers, 2560, 20000, None)
    raw = captured["raw"]
    # fits the raw estimate, not the scaled one -> refused
    monkeypatch.setattr(guards, "mem_available_bytes",
                        lambda path=None: int(raw * (1 + guards.VERIFY_SAFETY) / 2))
    with pytest.raises(SystemExit, match=f"x {guards.VERIFY_SAFETY} safety"):
        guards.verify_memory_guard(args, files, footers, 2560, 20000, None)
    # fits the scaled one -> allowed
    monkeypatch.setattr(guards, "mem_available_bytes",
                        lambda path=None: int(raw * guards.VERIFY_SAFETY) + 1)
    guards.verify_memory_guard(args, files, footers, 2560, 20000, None)
