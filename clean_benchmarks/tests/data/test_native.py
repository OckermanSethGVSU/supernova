"""--read-mode native: nova-bf's rust_fetch dense reader (CPU only, local
parquet). Blocks must be byte-identical to the arrow decode's for every reader
kind and both production layouts (Pes2o-VE: fixed_size_list<float>[2560],
uncompressed, multi-row-group; FineWeb: list<halffloat> 768, snappy, 11 row
groups), a refused file must fail loudly, and run.main must record the mode
end to end for both engines."""

import csv

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

pytest.importorskip("rust_fetch")

import make_dummy  # noqa: E402
from bench.systems import harness_cli as run  # noqa: E402
from bench.systems import guards
from bench.record import results_io, schema
from bench.data import sources  # noqa: E402
from bench.data.sources import NativeRefused, ParquetSource, Stages  # noqa: E402

try:  # the cuVS end-to-end test reuses test_cuvs_backend's fake cuvs (needs torch)
    import torch  # noqa: F401
    from test_cuvs_backend import _cuvs_factory, fake_cuvs  # noqa: F401
except ImportError:
    _cuvs_factory = None

FS = pafs.LocalFileSystem()


@pytest.fixture(autouse=True)
def drop_native_clients():
    """Each test's rust_fetch Clients go with it (see native_drop_clients)."""
    yield
    sources.native_drop_clients()

# (column, list_type, value dtype, compression, rows per file, d, row-group size):
# small stand-ins of the two production layouts, plus make_dummy's default
SHAPES = {
    "pes2o": ("embedding", "fixed", np.float32, "none", [300, 70], 2560, 128),
    "fineweb": ("dense_embedding", "list", np.float16, "snappy", [1100, 450], 768, 100),
    "onerg": ("embedding", "list", np.float32, "snappy", [400, 3], 32, None),
}


def _corpus(root, shape, seed=0):
    column, lt, vt, comp, rows, d, rg = SHAPES[shape]
    root.mkdir()
    for i, n in enumerate(rows):
        x = np.random.default_rng([seed, i]).standard_normal((n, d)).astype(vt)
        make_dummy.write_vectors(str(root / f"part-{i:05d}.parquet"), x, column,
                                 row_group_size=rg, compression=comp, value_dtype=vt,
                                 list_type=lt)
    return str(root), column


@pytest.fixture(scope="module", params=sorted(SHAPES))
def shaped(request, tmp_path_factory):
    path, column = _corpus(tmp_path_factory.mktemp("native") / request.param, request.param)
    return request.param, path, column


def _blocks(src):
    out = []
    for b in src:
        out.append((b.offset, b.vectors.dtype, b.vectors.shape, b.vectors.tobytes(), b.info))
        del b
    return out


def _arrow_blocks(path, column, dtype=None):
    return [(o, dt, sh, by) for o, dt, sh, by, _ in
            _blocks(ParquetSource(path, column, prefetch=0, read_mode="arrow", dtype=dtype))]


def test_read_file_byte_identical(shaped):
    name, path, column = shaped
    _, files = sources.list_parquet(path)
    for ref in files:
        a = sources.read_file(FS, ref.path, column, mode="arrow")
        rec, st = {}, Stages()
        n = sources.read_file(FS, ref.path, column, size=ref.size, mode="native",
                              rec=rec, stats=st)
        assert (n.dtype, n.shape) == (a.dtype, a.shape) and n.flags.c_contiguous
        assert n.tobytes() == a.tobytes()
        assert rec["read_path"] == "native" and rec["native_read"] == rec["read_table"] > 0
        assert st.get("native_fallbacks") == 0 and st.get("files") == 1
        md = pq.read_metadata(ref.path)
        assert st.get("row_groups") == md.num_row_groups
    if name == "fineweb":
        assert pq.read_metadata(files[0].path).num_row_groups == 11
        assert n.dtype == np.float16


@pytest.mark.parametrize("prefetch", [2, 0])
def test_source_blocks_identical_to_arrow(shaped, prefetch):
    name, path, column = shaped
    want = _arrow_blocks(path, column)
    with ParquetSource(path, column, prefetch=prefetch, workers=2,
                       read_mode="native").start() as src:
        got = []
        for b in src:
            if src.reader == "process":
                # decoded straight into the ring slot: nothing to publish
                assert b.vectors.base in src.ring.mms
            got.append((b.offset, b.vectors.dtype, b.vectors.shape, b.vectors.tobytes(),
                        b.info))
            del b
        st = src.stats.as_dict()
    assert [g[:4] for g in got] == want
    assert all(g[4]["read_path"] == "native" for g in got)
    assert st["native_fallbacks"] == 0 and st["native_read"] > 0
    assert st["bytes"] == sum(pq.read_metadata(p).row_group(i).column(0).total_compressed_size
                              for p in [r.path for r in src.files]
                              for i in range(pq.read_metadata(p).num_row_groups))


@pytest.mark.parametrize("prefetch", [2, 0])
def test_host_widen_identical(tmp_path, prefetch):
    """A host cast (f16 -> f32 widen) follows the native decode exactly as it
    follows arrow's."""
    path, column = _corpus(tmp_path / "fw", "fineweb")
    want = _arrow_blocks(path, column, dtype=np.float32)
    with ParquetSource(path, column, prefetch=prefetch, workers=2, read_mode="native",
                       dtype=np.float32).start() as src:
        got = [g[:4] for g in _blocks(src)]
    assert got == want


# ------------------------------------------------------------ refusals

def _refused_corpus(root):
    """part-0: plain list<float>; part-1: brotli (rust_fetch refuses the codec)."""
    root.mkdir()
    rng = np.random.default_rng(3)
    xs = [rng.standard_normal((n, 16)).astype(np.float32) for n in (200, 90)]
    make_dummy.write_vectors(str(root / "part-0.parquet"), xs[0])
    make_dummy.write_vectors(str(root / "part-1.parquet"), xs[1], compression="brotli")
    return str(root), np.concatenate(xs)


def test_refused_file_fails_loudly(tmp_path):
    path, _ = _refused_corpus(tmp_path / "c")
    bad = str(tmp_path / "c" / "part-1.parquet")
    with pytest.raises(NativeRefused, match="nova-bf would read it with pyarrow") as e:
        sources.read_file(FS, bad, "embedding", mode="native")
    assert "BROTLI" in str(e.value)
    for prefetch in (2, 0):
        with ParquetSource(path, prefetch=prefetch, workers=2,
                           read_mode="native").start() as src:
            with pytest.raises(NativeRefused):
                _blocks(src)


def test_float64_refused(tmp_path):
    p = str(tmp_path / "f64.parquet")
    x = np.random.default_rng(1).standard_normal((40, 8))
    pq.write_table(pa.table({"embedding": pa.array(list(x), pa.list_(pa.float64()))}), p)
    with pytest.raises(NativeRefused, match="DOUBLE"):
        sources.read_file(FS, p, "embedding", mode="native")


def test_schema_guard():
    plan = type("P", (), {"dtype": "float32", "n_rows": 10, "dim": 4})()
    foot = sources.Footer(10, 1, 160, 4, "float", 32, 10, "list", "SNAPPY")
    assert sources.native_schema_mismatch(foot, plan) is None
    for bad in (sources.Footer(10, 1, 160, 4, "halffloat", 16, 10, "list"),
                sources.Footer(11, 1, 160, 4, "float", 32, 11, "list"),
                sources.Footer(10, 1, 160, 5, "float", 32, 10, "fixed_size_list"),
                sources.Footer(10, 1, 160, 4, "float", 32, 10, None)):
        assert sources.native_schema_mismatch(bad, plan)


def test_custom_endpoint_and_other_fs_refused(monkeypatch):
    s3 = type("S3", (), {"type_name": "s3", "region": "us-east-1"})()
    assert sources.native_url(s3, "b/k.parquet") == ("s3://b/k.parquet", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio:9000")
    with pytest.raises(ValueError, match="AWS_ENDPOINT_URL"):
        sources.native_url(s3, "b/k.parquet")
    with pytest.raises(ValueError, match="gcs"):
        sources.native_url(type("G", (), {"type_name": "gcs"})(), "b/k")


def test_client_per_process_and_settings():
    a = sources.native_client(None, 64, 8 << 20)
    assert sources.native_client(None, 64, 8 << 20) is a
    assert (a.concurrency, a.part_size) == (64, 8 << 20)
    assert sources.native_client(None, 32, 8 << 20) is not a


def test_concurrency_mapping(shaped):
    """One Client per reader process at its share of the budget (total
    scope), or the whole budget per process (per-file); the inline reader
    has the whole budget."""
    _, path, column = shaped
    kw = dict(prefetch=2, workers=2, read_mode="native", range_concurrency=128)
    assert ParquetSource(path, column, **kw).range_concurrency == 64
    assert ParquetSource(path, column, range_scope="per-file", **kw).range_concurrency == 128
    assert ParquetSource(path, column, **dict(kw, prefetch=0)).range_concurrency == 128


# ------------------------------------------------------------ run.main

def test_parse_args_defaults():
    base = ["--regime", "stream", "--corpus", "x", "--csv", "y.csv", "--backend", "cpu"]
    a = run.parse_args(base + ["--read-mode", "native"])
    assert (a.read_mode, a.range_concurrency, a.range_mib) == ("native", 128, 16)
    a = run.parse_args(base)  # native is the default (and the only mode)
    assert (a.read_mode, a.range_concurrency, a.range_mib) == ("native", 128, 16)
    a = run.parse_args(base + ["--read-mode", "native", "--range-concurrency", "32",
                               "--range-mib", "8"])
    assert (a.range_concurrency, a.range_mib) == (32, 8)
    for gone in (["--native-fallback"], ["--read-mode", "arrow"], ["--read-mode", "auto"],
                 ["--read-mode", "ranged"], ["--read-mode", "sequential"]):
        with pytest.raises(SystemExit):
            run.parse_args(base + gone)
    # sparse: nova-bf's fetch path (rust_fetch download + pyarrow decode), same
    # rust_fetch defaults (tests/test_sparse_native.py covers the reads)
    a = run.parse_args(base + ["--read-mode", "native", "--modality", "sparse",
                               "--query-file", "q"])
    assert (a.read_mode, a.range_concurrency, a.range_mib) == ("native", 128, 16)


def _rows(out):
    rows = list(csv.DictReader(open(out)))
    return rows, list(csv.DictReader(open(results_io.trace_path(str(out)))))


@pytest.mark.parametrize("prefetch", ["2", "0"])
def test_main_faiss_native_verified(tmp_path, prefetch, capsys):
    path, column = _corpus(tmp_path / "fw", "fineweb")
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", path, "--column", column,
              "--read-mode", "native", "--nq", "8", "--k", "5", "--warmup", "0", "--backend", "cpu",
              "--repeats", "2", "--prefetch", prefetch, "--workers", "2", "--verify",
              "--csv", str(out)])
    assert capsys.readouterr().out.count("verify OK") == 1
    rows, trace = _rows(out)
    assert len(rows) == 2 and list(rows[0]) == schema.CSV_FIELDS
    for r in rows:
        assert (r["read_mode"], r["read_mode_effective"]) == ("native", "native")
        assert (r["range_mib"], r["native_fallback"], r["native_fallbacks"]) == \
            ("16", "refuse", "0")
        assert r["verified"] == "ok"
    assert rows[0]["range_concurrency"] == ("64" if prefetch == "2" else "128")
    assert all(float(r["native_read_s"]) > 0 for r in rows)
    assert float(rows[0]["native_read_s"]) == pytest.approx(float(rows[0]["read_table_s"]))
    assert {t["read_path"] for t in trace} == {"native"}


def test_main_refuses_a_file(tmp_path):
    path, _ = _refused_corpus(tmp_path / "c")
    out = tmp_path / "r.csv"
    args = ["--regime", "stream", "--corpus", path, "--read-mode", "native", "--nq", "4", "--backend", "cpu",
            "--k", "3", "--warmup", "0", "--repeats", "1", "--csv", str(out)]
    with pytest.raises(NativeRefused):
        run.main(args)


def test_main_cuvs_native(tmp_path, monkeypatch, capsys, request):
    """--backend cuvs (fake cuvs on torch's CPU device, as test_cuvs_backend)
    over the same native reads, process readers and the inline reader."""
    if _cuvs_factory is None:
        pytest.skip("torch not installed")
    request.getfixturevalue("fake_cuvs")
    _cuvs_factory(monkeypatch)
    path, column = _corpus(tmp_path / "fw", "fineweb")
    out = tmp_path / "r.csv"
    for prefetch in ("2", "0"):
        run.main(["--regime", "stream", "--prefetch", prefetch, "--corpus", path,
                  "--column", column, "--backend", "cuvs", "--read-mode", "native",
                  "--nq", "8", "--k", "5", "--warmup", "0", "--repeats", "1", "--verify",
                  "--csv", str(out)])
    assert capsys.readouterr().out.count("verify OK") == 2
    rows, trace = _rows(out)
    assert [(r["engine"], r["reader"], r["read_mode_effective"]) for r in rows] == \
        [("cuvs", "process", "native"), ("cuvs", "inline", "native")]
    assert {t["read_path"] for t in trace} == {"native"}


def test_memory_guard_models():
    foot = sources.Footer(1000, 10, 4_000_000, 1000, "float", 32, 100, "fixed_size_list")
    files = [sources.FileRef("a", 4_100_000)]
    footers = {"a": foot}
    staging = guards.NATIVE_STAGING_FACTOR * 8 * 400_000
    assert guards.native_read_transient(files, footers) == staging
    assert guards.native_read_bytes(files, footers, 1000, 4, slot=True) == staging
    assert guards.native_read_bytes(files, footers, 1000, 4) == staging + 1000 * 1000 * 4
