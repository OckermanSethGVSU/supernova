import argparse
import csv
import os
import statistics
import sys

import faiss
import numpy as np
import pytest

from bench.systems import harness_cli as run
from bench.systems import guards, loop
from bench.record import provenance, results_io, schema
from bench.verify import oracle as verify
from bench.systems import cpu_reference
from bench.data import sources
from conftest import D, ROWS, arrow_default_io_threads
from bench.data.sources import ParquetSource

K = 10


def check_against_oneshot(D, I, xq, xb, k=K):
    D_ref, I_ref = faiss.knn(xq, xb, k + 1, metric=faiss.METRIC_INNER_PRODUCT)
    assert verify.compare_results(D, I, D_ref, I_ref) == []
    # ids are global row ids: re-score each returned id against the corpus
    assert I.min() >= 0 and I.max() < len(xb)
    rescored = np.einsum("qd,qkd->qk", xq, xb[I])
    np.testing.assert_allclose(rescored, D, atol=1e-5)
    # ResultHeap(keep_max=True) result is sorted descending
    assert np.all(np.diff(D, axis=1) <= 1e-7)


def sources_for(path):
    return {
        "stream-pf0": ParquetSource(path, prefetch=0),
        "stream-pf2": ParquetSource(path, prefetch=2, workers=2),
        "stream-pf5": ParquetSource(path, prefetch=5, workers=3),
    }


@pytest.mark.parametrize("name", ["stream-pf0", "stream-pf2", "stream-pf5"])
def test_streamed_equals_oneshot(corpus, queries, name):
    path, xb = corpus
    src = sources_for(path)[name]
    backend = cpu_reference.CpuBackend(K)
    backend.setup(queries)
    D, I, info = loop.run_pass(src, backend, len(queries), K)
    check_against_oneshot(D, I, queries, xb)
    assert info["rows"] == sum(ROWS)
    assert info["rows_per_file"] == max(ROWS)


def test_k_larger_than_some_blocks_and_ids_offset(corpus, queries):
    """k=5 with files of 3 and 1 rows: their -1 padding must not leak as
    offset ids, and every id must point at the right global row."""
    path, xb = corpus
    backend = cpu_reference.CpuBackend(5)
    backend.setup(queries)
    D, I, _ = loop.run_pass(ParquetSource(path, prefetch=2, workers=2),
                           backend, len(queries), 5)
    check_against_oneshot(D, I, queries, xb, k=5)


def test_tiny_corpus_fewer_rows_than_k(tmp_path, queries):
    import make_dummy
    x = make_dummy.unit_vectors(np.random.default_rng(0), 4, queries.shape[1])
    make_dummy.write_vectors(str(tmp_path / "a.parquet"), x[:3])
    make_dummy.write_vectors(str(tmp_path / "b.parquet"), x[3:])
    backend = cpu_reference.CpuBackend(K)
    backend.setup(queries)
    D, I, _ = loop.run_pass(ParquetSource(str(tmp_path), prefetch=1), backend,
                           len(queries), K)
    assert np.all(I[:, 4:] == -1)
    assert np.all(np.sort(I[:, :4], axis=1) == np.arange(4))


@pytest.mark.parametrize("name", ["stream-pf0", "stream-pf2"])
def test_buckets_account_for_wall(corpus, queries, name):
    """other_s is DEFINED as total minus the timed buckets, so the buckets
    always sum to total; the real check is that the remainder is small and
    non-negative (time is not escaping the buckets, nor double counted)."""
    path, _ = corpus
    src = sources_for(path)[name]
    backend = cpu_reference.CpuBackend(K)
    backend.setup(queries)
    infos = [loop.run_pass(src, backend, len(queries), K)[2] for _ in range(4)][1:]
    for info in infos:
        assert info["other_s"] >= -1e-9
    frac = statistics.median(i["other_s"] / i["total_s"] for i in infos)
    assert frac < 0.05


def test_prefetch0_stall_is_fetch_plus_decode(corpus, queries):
    path, _ = corpus
    backend = cpu_reference.CpuBackend(K)
    backend.setup(queries)
    src = ParquetSource(path, prefetch=0)
    for info in [loop.run_pass(src, backend, len(queries), K)[2] for _ in range(3)][1:]:
        fd = info["fetch_s"] + info["decode_s"]
        assert info["source_stall_s"] >= fd
        assert info["source_stall_s"] - fd < 0.25 * fd + 0.002


def test_max_block_bytes_guard(corpus, queries):
    path, _ = corpus
    backend = cpu_reference.CpuBackend(K)
    backend.setup(queries)
    with pytest.raises(RuntimeError, match="per-file cap"):
        loop.run_pass(ParquetSource(path, prefetch=0), backend, len(queries), K,
                     max_block_bytes=1000)


def test_max_file_bytes_listing_guard(corpus, tmp_path, monkeypatch):
    """The listing-level cap (harness_cli.MAX_FILE_BYTES, not a flag): a corpus
    file larger than it on disk is refused before any read."""
    monkeypatch.setattr(run, "MAX_FILE_BYTES", 100)
    with pytest.raises(SystemExit, match=r"files exceed 100 B, e\.g\. "):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--csv", str(tmp_path / "r.csv")])
    assert not (tmp_path / "r.csv").exists()


def test_compare_results_detects_wrong_ids(corpus, queries):
    _, xb = corpus
    D, I = faiss.knn(queries, xb, K + 1, metric=faiss.METRIC_INNER_PRODUCT)
    I2 = I[:, :K].copy()
    I2[0, 0] = I[0, K]
    assert verify.compare_results(D[:, :K], I2, D, I)
    D2 = D[:, :K].copy()
    D2[3, 2] += 1e-3
    assert verify.compare_results(D2, I[:, :K], D, I)


@pytest.mark.parametrize("regime,extra", [
    ("stream", ["--prefetch", "0"]),
    ("stream", ["--prefetch", "3", "--workers", "2"]),
])
def test_cli_end_to_end(corpus, tmp_path, capsys, regime, extra):
    path, _ = corpus
    out = tmp_path / "r.csv"
    argv = ["--regime", regime, "--corpus", path, "--backend", "cpu", "--nq", "64",
            "--k", str(K), "--warmup", "1", "--repeats", "2", "--csv", str(out),
            "--omp-threads", "2", "--verify"] + extra
    run.main(argv)
    run.main(argv)  # appends under the same header
    assert "verify OK" in capsys.readouterr().out
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert list(rows[0].keys()) == schema.CSV_FIELDS
    # blank only where documented: no GPU, no query file (generated queries)
    blank = {"gpu_name", "gpu_uuid", "gpu_index", "driver", "query_file", "query_column"}
    # cpu: no device events, so the event columns are not measured (blank, not
    # 0); host widen_s too, as with --no-gpu-events, when nothing was widened
    blank |= set(schema.EVENT_FIELDS)
    # ip (the default): no normalize step; no --gt-ids-column: no gt_recall
    blank |= {"normalize_s", "gt_recall"}
    blank |= {"engine_host_s"}  # resident (GPU-path) backends only
    blank |= {"device_merge_s"}  # cuVS's own recipe (--cuvs-merge device) only
    blank |= {"nnz", "nnz_kept"}  # dense: no nonzeros to count (--modality sparse only)
    blank |= {"project_s"}  # --sparse-project query-vocab only
    blank |= {"tokens", "pylate_calls", "pylate_calls_max_block"}  # multivector / --backend pylate only
    blank |= {"spare_plan_s", "spare_score_s", "spare_topk_s"}  # --backend spare stage events only
    # --filter only (tests/test_filtered_run.py): measured only when filtering
    blank |= {"filter_s", "filter_eval_s", "filter_dedup_s", "filter_compact_s",
              "filter_publish_s",
              "filter_prep_s", "filter_sql_s", "filter_text_s", "filter_combine_s",
              "filter_sql_query_s", "filter_sql_post_s", "filter_text_query_s",
              "filter_text_post_s", "filter_text_tokens", "filter_setup_s",
              "filter_overflow_bytes", "filter_bits_s",
              "payload_read_s", "filter_handoff_s", "filter_blocks", "filter_rows_kept",
              "filter_allowed_pairs",
              "filter_live_queries", "filter_skipped_files", "filter_distinct_masks",
              "filter_searched_queries", "filter_compact_rows", "filter_compact_floor_files",
              "filter_compact_floor_rows", "filter_compact_whole_files",
              "filter_compact_max_frac", "filter_mask_slots", "filter_mask_overflows",
              "cuvs_filter_small_k_calls"}
    # the CUDA libraries the engine ran on: blank for a CPU run, whatever this
    # test process mapped (import torch); checked in test_provenance_columns
    blank |= set(provenance.CUDA_LIB_FIELDS)
    # not launched through paper_run.sh, not on EC2 (conftest: no IMDS lookup), not
    # --backend cuvs, not a GPU run (test_stack_fields_* check them set)
    blank |= {"paper_stack_sha", "paper_code_sha", "paper_run_dir", "instance_id",
              "instance_type", "availability_zone", "cuvs_version", "libcuvs",
              "libcuvs_sha256", "cuvs_patches", "gpu_state"}
    # depend on this box's install / what this test process has mapped / its env
    maybe = {"faiss_build", "thread_libs", "openmp_runtimes", "cuda_env"}
    procs = regime == "stream" and extra[1] != "0"
    if not procs:  # only reader processes report their own pools and CPU time
        blank |= {"reader_pa_cpu_threads", "reader_pa_io_threads",
                  "worker_cpu_user_s", "worker_cpu_sys_s",
                  "publish_s", "handoff_s"}  # no ring: not measured, not 0
    blank |= {"preload_rss_bytes"}  # no preload: no RSS after it
    assert all(rows[0][c] != "" for c in schema.CSV_FIELDS if c not in blank | maybe)
    assert all(rows[0][c] == "" for c in blank)
    assert "OMP_NUM_THREADS=2" in rows[0]["thread_env"].split(";")   # conftest sets it
    assert [r["warmup"] for r in rows[:3]] == ["1", "0", "0"]
    # the run process's VmHWM never shrinks
    hwm = [int(r["rss_hwm_bytes"]) for r in rows[:3]]
    assert hwm == sorted(hwm) and hwm[0] > 0
    assert {r["verified"] for r in rows} == {"ok"}
    r = rows[1]
    assert r["regime"] == regime and r["backend"] == "cpu" and r["d"] == "32"
    assert int(r["rows"]) == sum(ROWS) and int(r["n_files"]) == len(ROWS)
    assert r["omp_threads"] == "2" and r["corpus_fs"] == "local"
    assert int(r["pa_cpu_threads"]) >= 1 and int(r["pa_io_threads"]) >= 1
    # vectors-only corpus: the column is ~all of the file
    assert r["read_mode"] == r["read_mode_effective"] == "native"
    assert r["native_fallbacks"] == "0" and r["native_fallback"] == "refuse"
    assert float(r["column_share"]) > 0.9 and r["range_scope"] == "total"
    assert r["stored_dtype"] == "f32" and r["widen_where"] == "host"
    assert r["widen_s"] == "" and r["h2d_s"] == ""  # f32 -> f32, cpu: no events
    assert r["row_groups"] == str(len(ROWS)) and r["row_groups_per_file"] == "1"
    if "--prefetch" in extra and extra[1] == "0":
        assert r["workers"] == "0"
    assert r["reader"] == ("process" if procs else "inline")
    if procs:
        # 2 readers (min(prefetch 3, workers 2)) share the usable CPUs
        from bench.data import sources
        assert r["reader_pa_cpu_threads"] == sources.pool_split_str(
            sources.split_threads(sources.usable_cpu_count(), 2))
        # the readers' CPU is in cpu_*_s; reading happens there, not here
        assert float(r["worker_cpu_user_s"]) + float(r["worker_cpu_sys_s"]) > 0
        assert float(r["cpu_user_s"]) >= float(r["worker_cpu_user_s"])
        assert float(r["handoff_s"]) > 0 and float(r["publish_s"]) > 0
        # Arrow's default IO pool (8) SPLIT across the 2 readers, as one
        # shared pool would be, not 8 per reader
        assert r["reader_pa_io_threads"] == sources.pool_split_str(
            sources.split_threads(arrow_default_io_threads(), 2))
    else:
        assert r["handoff_s"] == "" and r["publish_s"] == ""


def test_cli_query_file(corpus, tmp_path):
    import make_dummy
    path, _ = corpus
    q = make_dummy.make_queries(str(tmp_path / "q.parquet"), 40, 32)
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", path, "--backend", "cpu",
              "--query-file", q, "--nq", "25", "--warmup", "0", "--repeats", "1",
              "--csv", str(out), "--verify"])
    assert next(csv.DictReader(open(out)))["nq"] == "25"


def test_max_files_limits_every_read(corpus, tmp_path, capsys, monkeypatch):
    """--max-files N: the timed passes and the --verify re-load read only the
    first N files of the sorted listing."""
    regime = "stream"
    from bench.data import sources
    path, _ = corpus
    read, real = [], sources.read_file
    monkeypatch.setattr(sources, "read_file",
                        lambda fs, p, *a, **kw: read.append(p) or real(fs, p, *a, **kw))
    out = tmp_path / "r.csv"
    run.main(["--regime", regime, "--corpus", path, "--backend", "cpu", "--nq", "16",
              "--warmup", "0", "--repeats", "1", "--csv", str(out), "--verify",
              "--max-files", "2", "--prefetch", "2", "--workers", "2"])
    assert f"over {sum(ROWS[:2])} rows" in capsys.readouterr().out
    first = sorted(sources.list_parquet(path)[1], key=lambda r: r.path)[:2]
    assert set(read) == {r.path for r in first}
    # the verify re-load here, the pass itself in the reader processes (which
    # the patch cannot see: the trace says what they read)
    assert len(read) == 2
    trace = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert [t["path"] for t in trace] == [r.path for r in first]
    r = next(csv.DictReader(open(out)))
    assert (r["n_files"], r["max_files"]) == ("2", "2")
    assert int(r["rows"]) == sum(ROWS[:2])


def test_nq_beyond_query_file_warns(corpus, tmp_path, capsys):
    import make_dummy
    q = make_dummy.make_queries(str(tmp_path / "q.parquet"), 10, 32)
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--query-file", q, "--nq", "25", "--warmup", "0", "--repeats", "1",
              "--csv", str(out)])
    assert "--nq 25 > the 10 rows" in capsys.readouterr().err
    assert next(csv.DictReader(open(out)))["nq"] == "10"


def test_csv_header_mismatch_refused(corpus, tmp_path):
    out = tmp_path / "r.csv"
    out.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--repeats", "1", "--warmup", "0", "--csv", str(out)])


@pytest.mark.parametrize("argv", [
    ["--backend", "gpu", "--k", "2049"],
    ["--backend", "cpu", "--dtype", "f16"],   # removed flags are refused
    ["--regime", "memory"],
    ["--reader", "thread"],
    ["--queries", "per-call"],
    ["--read-mode", "ranged"],
    ["--native-fallback"],
    ["--verify-external", "a=b.npz"],
    ["--k", "0"],
    ["--prefetch", "-1"],
    ["--max-files", "-1"],
])
def test_arg_guards(argv):
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y"] + argv)


def test_cpu_backend_refuses_f16():
    with pytest.raises(ValueError):
        cpu_reference.CpuBackend(K, np.float16)


def test_cli_records_pools_read_mode_and_row_groups(tmp_path):
    import make_dummy
    import pyarrow as pa
    corpus = tmp_path / "c"
    make_dummy.make_corpus(str(corpus), 3, 100, 8, row_group_size=30)  # 4 rgs/file
    out = tmp_path / "r.csv"
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--corpus", str(corpus), "--backend", "cpu",
                  "--nq", "8", "--warmup", "0", "--repeats", "1", "--csv", str(out),
                  "--read-mode", "native", "--prefetch", pf, "--workers", "2",
                  "--verify"])
    rows = list(csv.DictReader(open(out)))
    for r in rows:
        assert (r["pa_cpu_threads"], r["pa_io_threads"]) == (str(pa.cpu_count()),
                                                             str(pa.io_thread_count()))
        assert r["read_mode"] == r["read_mode_effective"] == "native"
        assert r["row_groups"] == "12" and r["row_groups_per_file"] == "4"
    assert [r["range_concurrency"] for r in rows] == ["128", "64"]  # 128 // 2 readers


def test_cli_range_scope_per_file(tmp_path, corpus):
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "0", "--repeats", "1", "--csv", str(out),
              "--read-mode", "native", "--range-concurrency", "6",
              "--range-scope", "per-file", "--prefetch", "3", "--workers", "3"])
    r = next(csv.DictReader(open(out)))
    assert r["range_concurrency"] == "6" and r["range_scope"] == "per-file"


def test_pools_configured_before_first_read(corpus, tmp_path, monkeypatch):
    from bench.data import sources
    order = []
    real_cfg, real_md = sources.configure_arrow_pools, sources.pq.read_metadata
    monkeypatch.setattr(run, "configure_arrow_pools",
                        lambda *a: order.append("pools") or real_cfg(*a))
    # the first file's footer is the first parquet read in main()
    monkeypatch.setattr(sources.pq, "read_metadata",
                        lambda *a, **kw: order.append("read") or real_md(*a, **kw))
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "0", "--repeats", "1",
              "--csv", str(tmp_path / "r.csv")])
    assert order[0] == "pools" and "read" in order


def test_dim_flag_mismatch_is_an_error(corpus, tmp_path):
    with pytest.raises(ValueError, match="16-D"):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--dim", "16", "--warmup", "0", "--repeats", "1",
                  "--csv", str(tmp_path / "r.csv")])


def test_query_file_dim_mismatch_is_an_error(corpus, tmp_path):
    import make_dummy
    q = make_dummy.make_queries(str(tmp_path / "q.parquet"), 10, 16)
    with pytest.raises(ValueError, match="16-D, corpus is 32-D"):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--query-file", q, "--csv", str(tmp_path / "r.csv")])


def test_verify_without_passes_refused():
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y",
                        "--verify", "--warmup", "0", "--repeats", "0"])


@pytest.mark.parametrize("regime", ["stream"])
def test_failed_verify_marks_rows_and_exits(corpus, tmp_path, monkeypatch, regime):
    """A pass whose --verify fails must never sit in the CSV as an ordinary
    row: the rows are held back until the check has run, then written with
    verified=failed (trace rows too), and the run exits non-zero. The plot
    drops them."""
    from bench.report import plots as plot
    monkeypatch.setattr(verify, "compare_results", lambda *a, **kw: ["I mismatch on query 3"])
    out = tmp_path / "r.csv"
    with pytest.raises(SystemExit, match="VERIFY FAILED: I mismatch on query 3"):
        run.main(["--regime", regime, "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--warmup", "1", "--repeats", "2", "--verify",
                  "--csv", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3 and {r["verified"] for r in rows} == {"failed"}
    trace = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert len(trace) == 3 * len(ROWS) and {r["verified"] for r in trace} == {"failed"}
    run.main(["--regime", regime, "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "0", "--repeats", "1", "--csv", str(out)])
    assert list(csv.DictReader(open(out)))[-1]["verified"] == "not_run"
    assert plot.read_csvs([str(out)])["verified"].tolist() == ["not_run"]


def test_verify_holds_rows_until_checked(corpus, tmp_path, monkeypatch):
    """No row reaches the CSV before the check has run."""
    out = tmp_path / "r.csv"
    seen = []
    real = verify.compare_results

    def spy(*a, **kw):
        seen.append(out.exists() and out.stat().st_size > 0)
        return real(*a, **kw)

    monkeypatch.setattr(verify, "compare_results", spy)
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "1", "--repeats", "1", "--verify",
              "--csv", str(out)])
    assert seen == [False]
    assert [r["verified"] for r in csv.DictReader(open(out))] == ["ok", "ok"]


def test_provenance_columns(corpus, tmp_path):
    import make_dummy
    import pyarrow as pa
    out = tmp_path / "r.csv"
    base = ["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
            "--warmup", "0", "--repeats", "1", "--csv", str(out)]
    run.main(base + ["--seed", "77"])
    q = make_dummy.make_queries(str(tmp_path / "q.parquet"), 40, 32)
    run.main(base + ["--query-file", q, "--query-column", "embedding"])
    gen, qf = list(csv.DictReader(open(out)))
    assert gen["column"] == qf["column"] == "embedding"
    assert (gen["query_file"], gen["query_column"], gen["seed"]) == ("", "", "77")
    assert (qf["query_file"], qf["query_column"], qf["seed"]) == (q, "embedding", "")
    for r in (gen, qf):
        assert r["pyarrow_version"] == pa.__version__
        assert r["numpy_version"] == np.__version__
        assert r["merge_faiss_version"] == faiss.__version__
        # a CPU run used no CUDA library, whatever this process maps (import torch)
        assert all(r[c] == "" for c in provenance.CUDA_LIB_FIELDS)
        assert r["torch_version"] == r["torch_cuda"] == "n/a"  # cpu: no torch
        assert r["gpu_name"] == r["gpu_uuid"] == r["gpu_index"] == r["driver"] == ""  # cpu


def _maps(paths):
    """/proc/self/maps text mapping each of ``paths`` (fake addresses)."""
    return "".join(f"7f00-7f01 r-xp 00000000 08:01 1 {p}\n" for p in paths)


# conda's cuSPARSE beside torch's: the case the harness refuses
DUP_MAPS = _maps([
    "/env/lib/libcublas.so.13.4.1.3", "/env/lib/libcublas.so.13.4.1.3",   # one file, 2 maps
    "/env/lib/libcusparse.so.12", "/site/nvidia/cu13/lib/libcusparse.so.12",
    "/site/nvidia/cu13/lib/libcudart.so.13", "/site/nvidia/cu13/lib/libnvJitLink.so.13",
    "/site/nvidia/cu13/lib/libcurand.so.10", "/env/lib/libtorch_cuda.so", "/usr/lib/libc.so.6",
    "/env/lib/libcusolverMg.so.12", "/env/lib/libnvrtc.alt.so.13"])   # other libraries


def test_cudalibs_finds_every_copy_and_the_duplicates_in_the_maps():
    from bench.record import cudalibs
    libs = cudalibs.mapped(DUP_MAPS)
    assert libs == {
        "libcublas": ["/env/lib/libcublas.so.13.4.1.3"],
        "libcudart": ["/site/nvidia/cu13/lib/libcudart.so.13"],
        "libcurand": ["/site/nvidia/cu13/lib/libcurand.so.10"],
        "libcusparse": ["/env/lib/libcusparse.so.12", "/site/nvidia/cu13/lib/libcusparse.so.12"],
        "libnvJitLink": ["/site/nvidia/cu13/lib/libnvJitLink.so.13"],
    }
    assert cudalibs.duplicates(libs) == {"libcusparse": libs["libcusparse"]}
    assert cudalibs.duplicates(cudalibs.mapped(_maps(["/a/libcublas.so.13"]))) == {}
    assert cudalibs.mapped("") == {}
    msg = cudalibs.duplicate_message(cudalibs.duplicates(libs))
    assert "libcusparse: /env/lib/libcusparse.so.12, /site/nvidia/cu13/lib/libcusparse.so.12" in msg


def test_cuda_libs_fields_names_the_one_copy_and_lists_duplicates():
    from bench.record import cudalibs
    got = provenance.cuda_libs_fields(True, cudalibs.mapped(DUP_MAPS))
    assert got["cuda_libraries"] == (
        "libcublas=/env/lib/libcublas.so.13.4.1.3;libcudart=/site/nvidia/cu13/lib/libcudart.so.13;"
        "libcurand=/site/nvidia/cu13/lib/libcurand.so.10;"
        "libnvJitLink=/site/nvidia/cu13/lib/libnvJitLink.so.13")
    # two copies: neither can be named as THE copy, so every copy goes here
    assert got["cuda_lib_duplicates"] == ("libcusparse=/env/lib/libcusparse.so.12@;"
                                          "libcusparse=/site/nvidia/cu13/lib/libcusparse.so.12@")
    # fake files cannot be asked (never loaded, never raises): no versions
    assert got["cuda_lib_versions"] == got["cublas_version"] == ""
    # a run that did not use CUDA: blank, whatever is mapped
    assert provenance.cuda_libs_fields(False, cudalibs.mapped(DUP_MAPS)) == \
        dict.fromkeys(provenance.CUDA_LIB_FIELDS, "")
    assert provenance.cuda_libs_fields(True, {}) == dict.fromkeys(provenance.CUDA_LIB_FIELDS, "")


def test_cuda_libs_fields_reads_the_real_versions_of_mapped_copies():
    """Each mapped copy reports its own version, through a handle on that very
    file (no GPU needed); a file that is not mapped is never loaded to ask it."""
    import ctypes
    import shutil
    import tempfile
    from bench.record import cudalibs
    paths = cudalibs.pip_lib_paths()
    if not {"libcublasLt", "libcublas", "libcudart"} <= set(paths):
        pytest.skip("no CUDA torch with pip CUDA wheels here")
    for fam in ("libcublasLt", "libcublas", "libcudart"):
        ctypes.CDLL(paths[fam])   # mapped already by torch_first: a no-op
    libs = {f: [os.path.realpath(paths[f])] for f in ("libcublas", "libcudart")}
    got = provenance.cuda_libs_fields(True, libs)
    assert got["cublas_version"].count(".") == 2, got
    vers = dict(kv.split("=") for kv in got["cuda_lib_versions"].split(";"))
    assert vers["libcublas"] == got["cublas_version"] and vers["libcudart"].count(".") == 1
    with tempfile.TemporaryDirectory() as d:   # another file: not the mapped one
        copy = os.path.realpath(os.path.join(d, "libcudart.so.13"))
        shutil.copyfile(paths["libcudart"], copy)
        assert cudalibs.version("libcudart", copy) == ""
        assert copy not in cudalibs.mapped().get("libcudart", [])


def _fake_backend(name, torch=None, cuda=False):
    import types
    return types.SimpleNamespace(name=name, torch=torch, _cuda=cuda)


def test_backend_uses_cuda():
    assert not provenance.backend_uses_cuda(_fake_backend("cpu"))
    assert not provenance.backend_uses_cuda(_fake_backend("masked"))
    assert provenance.backend_uses_cuda(_fake_backend("gpu"))   # FAISS per-call: no torch, CUDA
    assert provenance.backend_uses_cuda(_fake_backend("cuvs", torch=object(), cuda=True))
    assert not provenance.backend_uses_cuda(_fake_backend("spare", torch=object(), cuda=False))


def test_duplicate_cuda_libs_refuse_a_gpu_run_only():
    from bench.record import cudalibs
    libs = cudalibs.mapped(DUP_MAPS)
    args = argparse.Namespace(allow_duplicate_cuda_libs=False)
    with pytest.raises(SystemExit, match="(?s)two copies of a CUDA library.*libcusparse"):
        provenance.refuse_duplicate_cuda_libs(args, _fake_backend("cuvs", object(), True), libs)
    with pytest.raises(SystemExit, match="libcusparse"):
        provenance.refuse_duplicate_cuda_libs(args, _fake_backend("gpu"), libs)
    # not refused: a CPU run (it never used them), a CPU device, the escape hatch,
    # one copy of each
    provenance.refuse_duplicate_cuda_libs(args, _fake_backend("cpu"), libs)
    provenance.refuse_duplicate_cuda_libs(args, _fake_backend("pylate", object(), False), libs)
    provenance.refuse_duplicate_cuda_libs(argparse.Namespace(allow_duplicate_cuda_libs=True),
                                   _fake_backend("cuvs", object(), True), libs)
    provenance.refuse_duplicate_cuda_libs(args, _fake_backend("cuvs", object(), True),
                                   cudalibs.mapped(_maps(["/a/libcublas.so.13"])))


def test_main_refuses_a_cuda_run_with_two_copies_before_any_pass(corpus, tmp_path, monkeypatch):
    """main checks once the backend is set up, so no row is written; with
    --allow-duplicate-cuda-libs the rows record every copy."""
    from bench.record import cudalibs
    monkeypatch.setattr(provenance, "backend_uses_cuda", lambda b: True)   # stand in for a GPU run
    real = cudalibs.mapped
    monkeypatch.setattr(cudalibs, "mapped",
                        lambda maps_text=None: real(DUP_MAPS if maps_text is None else maps_text))
    out = tmp_path / "r.csv"
    # (a stand-in GPU run off the paper stack: that refusal is test_main_refuses_unpinned_*'s)
    base = ["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
            "--warmup", "0", "--repeats", "1", "--csv", str(out), "--allow-unpinned-stack"]
    with pytest.raises(SystemExit, match="two copies of a CUDA library"):
        run.main(base)
    assert not out.exists()
    run.main(base + ["--allow-duplicate-cuda-libs"])
    (r,) = csv.DictReader(open(out))
    assert r["cuda_lib_duplicates"].startswith("libcusparse=/env/lib/libcusparse.so.12@")
    assert "libcublas=/env/lib/libcublas.so.13.4.1.3" in r["cuda_libraries"]


def _paper_env(monkeypatch, tmp_path, **over):
    """The environment paper_run.sh exports (PAPER_*), for this interpreter."""
    env = {"PAPER_STACK_SHA": provenance.versions_sha(), "PAPER_CODE_SHA": "c0de" * 16,
           "PAPER_ENV_PREFIX": sys.prefix, "PAPER_RUN_DIR": str(tmp_path / "paper_run")}
    env.update(over)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    return env


def test_main_refuses_unpinned_stack_for_a_gpu_run(corpus, tmp_path, monkeypatch):
    """A CUDA run is refused before any pass unless the launcher's PAPER_* say
    it is on THIS harness's stack, in this interpreter; --allow-unpinned-stack
    runs anyway; the row records what was set."""
    monkeypatch.delenv(provenance.ALLOW_UNPINNED_ENV)   # conftest sets it for the GPU tests
    monkeypatch.setattr(provenance, "backend_uses_cuda", lambda b: True)   # stand in for a GPU run
    out = tmp_path / "r.csv"
    base = ["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
            "--warmup", "0", "--repeats", "1", "--csv", str(out), "--allow-duplicate-cuda-libs"]
    with pytest.raises(SystemExit, match="off the paper stack: PAPER_STACK_SHA is not set"):
        run.main(base)
    assert not out.exists()
    for over, why in (({"PAPER_STACK_SHA": "f" * 64}, "launched for another stack"),
                      ({"PAPER_ENV_PREFIX": str(tmp_path)}, "is not the paper env"),
                      ({"PAPER_ENV_PREFIX": None}, "PAPER_ENV_PREFIX is not set"),
                      ({"PAPER_LAUNCH_STUBBED": "1"}, "stub")):
        env = _paper_env(monkeypatch, tmp_path, **over)
        with pytest.raises(SystemExit, match=why):
            run.main(base)
        assert not out.exists()
    monkeypatch.delenv("PAPER_LAUNCH_STUBBED")
    env = _paper_env(monkeypatch, tmp_path)
    run.main(base)                                   # on the stack: runs
    for k in ("PAPER_STACK_SHA", "PAPER_CODE_SHA", "PAPER_ENV_PREFIX", "PAPER_RUN_DIR"):
        monkeypatch.delenv(k)
    run.main(base + ["--allow-unpinned-stack"])      # off it, with the override: runs
    on, off = csv.DictReader(open(out))
    assert (on["paper_stack_sha"], on["paper_code_sha"], on["paper_run_dir"]) == (
        env["PAPER_STACK_SHA"], env["PAPER_CODE_SHA"], env["PAPER_RUN_DIR"])
    assert off["paper_stack_sha"] == off["paper_code_sha"] == off["paper_run_dir"] == ""


def test_cpu_run_is_never_refused_and_records_the_launcher(corpus, tmp_path, monkeypatch):
    env = _paper_env(monkeypatch, tmp_path, PAPER_STACK_SHA="0" * 64)   # another stack's
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    out = tmp_path / "r.csv"
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
              "--warmup", "0", "--repeats", "1", "--csv", str(out)])
    (r,) = csv.DictReader(open(out))
    assert r["paper_stack_sha"] == "0" * 64 and r["paper_code_sha"] == env["PAPER_CODE_SHA"]
    assert "MKL_THREADING_LAYER=GNU" in r["thread_env"].split(";")
    assert "CUDA_LAUNCH_BLOCKING=1" in r["cuda_env"].split(";")
    assert r["instance_id"] == r["instance_type"] == r["availability_zone"] == ""   # no IMDS
    assert r["gpu_state"] == "" and r["cuvs_version"] == r["libcuvs"] == ""   # cpu, not cuvs
    assert r["faiss_compile_options"] == faiss.get_compile_options().strip()
    # what /proc/self/maps has of OpenMP / BLAS: faiss links one, numpy its OpenBLAS
    assert r["thread_libs"] != "" and all("=" in kv for kv in r["thread_libs"].split(";"))


def test_stack_fields_for_a_gpu_run(monkeypatch):
    import types
    monkeypatch.setattr(provenance, "ec2_identity", lambda: {
        "instance_id": "i-0abc", "instance_type": "g5.8xlarge",
        "availability_zone": "us-east-1a"})
    seen = []

    def smi_lines(field, dev=None):
        seen.append((field, dev))
        return [", ".join(f"v{i}" for i in range(len(provenance.GPU_STATE_FIELDS)))]
    monkeypatch.setattr(provenance, "nvidia_smi_lines", smi_lines)
    be = types.SimpleNamespace(name="gpu", torch=None)
    got = provenance.stack_fields(argparse.Namespace(backend="gpu"), be, "GPU-1234")
    assert (got["instance_id"], got["instance_type"], got["availability_zone"]) == (
        "i-0abc", "g5.8xlarge", "us-east-1a")
    assert got["gpu_state"].split(";")[0] == "persistence_mode=v0"
    assert got["gpu_state"].count("=") == len(provenance.GPU_STATE_FIELDS)
    assert seen == [(",".join(provenance.GPU_STATE_FIELDS), "GPU-1234")]
    # a driver that rejects one name: field by field, the rejected one left out
    monkeypatch.setattr(provenance, "nvidia_smi_lines", lambda field, dev=None: [])
    monkeypatch.setattr(provenance, "nvidia_smi", lambda f, dev=None: "" if f == "ecc.mode.current"
                        else f"<{f}>")
    st = provenance.gpu_state("0")
    assert "ecc.mode.current" not in st and "power.limit=<power.limit>" in st
    assert provenance.gpu_state("") == ""
    # a CPU backend: no GPU state, whatever nvidia-smi says
    assert provenance.stack_fields(argparse.Namespace(backend="cpu"),
                            types.SimpleNamespace(name="cpu"), "0")["gpu_state"] == ""


@pytest.mark.slow
def test_run_imports_torch_before_faiss():
    """the harness's import order (cudalibs.torch_first): torch -- and its pip CUDA
    libraries -- before faiss, or libfaiss's sonames would map conda's copies
    first. Checked in a fresh process: sys.modules keeps import-start order."""
    import importlib.util
    import subprocess
    import sys
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    harness = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(run.__file__))))
    code = (f"import sys; sys.path.insert(0, {harness!r}); "
            "from bench.systems import harness_cli as run; "
            "m = list(sys.modules); print(m.index('torch'), m.index('faiss'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="2"), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    t, f = map(int, out.stdout.split())
    assert t < f


def test_provenance_names_the_gpu_by_uuid_and_index(monkeypatch):
    """gpu_uuid / gpu_index name the GPU the run used as nvidia-smi does
    (gpu_busy.py's default --gpu). Torch path: torch's UUID for the run's
    device, the index looked up BY that UUID (CUDA's ordinal need not be
    nvidia-smi's); otherwise nvidia-smi's for the first visible GPU."""
    import argparse
    import types
    calls = []

    def smi(field, dev=None):
        calls.append((field, dev))
        return {"index": "3", "uuid": "GPU-smi", "name": "smi-name"}.get(field, "x")

    monkeypatch.setattr(provenance, "nvidia_smi", smi)
    monkeypatch.setattr(provenance, "nvidia_driver", lambda path=None: "580")
    args = argparse.Namespace(column="embedding", query_file=None, query_column="q", seed=1)
    props = types.SimpleNamespace(uuid="58fb7c4c-0000-1111-2222-333344445555")
    cuda = types.SimpleNamespace(get_device_name=lambda dev: "A10G",
                                 get_device_properties=lambda dev: props)
    torch = types.SimpleNamespace(cuda=cuda, __version__="2", version=types.SimpleNamespace(
        cuda="13"))
    be = types.SimpleNamespace(name="gpu", torch=torch, _cuda=True, device="cuda:1")
    p = provenance.provenance(args, be)
    assert (p["gpu_name"], p["gpu_uuid"], p["gpu_index"]) == \
        ("A10G", "GPU-58fb7c4c-0000-1111-2222-333344445555", "3")
    assert calls == [("index", "GPU-58fb7c4c-0000-1111-2222-333344445555")]
    calls.clear()
    be = types.SimpleNamespace(name="cuvs", torch=None)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # see the test below
    p = provenance.provenance(args, be)
    assert (p["gpu_name"], p["gpu_uuid"], p["gpu_index"]) == ("smi-name", "GPU-smi", "3")
    assert ("index", "GPU-smi") in calls


FAKE_SMI = """#!{python}
import os, sys
gpus = [dict(index=str(i), name=n, uuid="GPU-" + "ab"[i] * 8)
        for i, n in enumerate(os.environ["FAKE_SMI_NAMES"].split(","))]
field = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--query-gpu="))
dev = sys.argv[sys.argv.index("-i") + 1] if "-i" in sys.argv else None
sel = [g for g in gpus if dev in (None, g["index"], g["uuid"])]
if not sel:
    sys.exit(6)
print("\\n".join(g[field] for g in sel))
"""


@pytest.mark.parametrize("names,env,want", [
    # mixed models, CUDA's default fastest-first order: CUDA device 0 need
    # not be nvidia-smi's index 0, so nothing is named (and a warning)
    ("Tesla T4,NVIDIA A10G", {}, ("", "", "")),
    ("Tesla T4,NVIDIA A10G", {"CUDA_VISIBLE_DEVICES": "1"}, ("", "", "")),
    ("Tesla T4,NVIDIA A10G", {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
     ("NVIDIA A10G", "GPU-bbbbbbbb", "1")),
    ("Tesla T4,NVIDIA A10G", {"CUDA_VISIBLE_DEVICES": "GPU-bbbbbbbb,GPU-aaaaaaaa"},
     ("NVIDIA A10G", "GPU-bbbbbbbb", "1")),
    # one model: the default order of identical GPUs is unspecified, so
    # still nothing is named without PCI_BUS_ID (or a UUID)
    ("NVIDIA A10G,NVIDIA A10G", {"CUDA_VISIBLE_DEVICES": "1"}, ("", "", "")),
    ("NVIDIA A10G,NVIDIA A10G", {}, ("", "", "")),
    ("NVIDIA A10G,NVIDIA A10G", {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
     ("NVIDIA A10G", "GPU-bbbbbbbb", "1")),
    # exactly one GPU in total: the ordinal can only be it
    ("NVIDIA A10G", {}, ("NVIDIA A10G", "GPU-aaaaaaaa", "0")),
])
def test_provenance_without_torch_maps_cuda_ordinal_safely(tmp_path, monkeypatch, capsys,
                                                           names, env, want):
    """Without torch (FAISS per-call) the GPU is looked up in nvidia-smi from
    CUDA_VISIBLE_DEVICES, whose entries are CUDA ordinals (fastest first by
    default), not nvidia-smi indices: a UUID is used as is, an ordinal only
    with CUDA_DEVICE_ORDER=PCI_BUS_ID or exactly one GPU; otherwise blank
    and a warning naming the fix. Real nvidia_smi code, fake nvidia-smi."""
    import argparse
    import sys
    import types
    smi = tmp_path / "nvidia-smi"
    smi.write_text(FAKE_SMI.format(python=sys.executable))
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SMI_NAMES", names)
    for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(provenance, "nvidia_driver", lambda path=None: "580")
    args = argparse.Namespace(column="embedding", query_file=None, query_column="q", seed=1)
    p = provenance.provenance(args, types.SimpleNamespace(name="gpu", torch=None))
    assert (p["gpu_name"], p["gpu_uuid"], p["gpu_index"]) == want
    err = capsys.readouterr().err
    assert ("CUDA_DEVICE_ORDER=PCI_BUS_ID" in err) == (want[1] == "")


def test_nvidia_driver_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "nvidia_smi", lambda field: "smi-" + field)
    for text, want in [
        ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  Wed Aug 27 2025\n",
         "580.82.07"),
        ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.172.08  "
         "Release Build\n", "570.172.08"),
    ]:
        p = tmp_path / "version"
        p.write_text(text)
        assert provenance.nvidia_driver(str(p)) == want
    assert provenance.nvidia_driver(str(tmp_path / "missing")) == "smi-driver_version"


def test_verify_after_warmup_only(corpus, tmp_path, capsys):
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "1", "--repeats", "0", "--verify",
              "--csv", str(tmp_path / "r.csv")])
    assert "verify OK" in capsys.readouterr().out


def test_csv_written_per_pass(corpus, tmp_path, monkeypatch):
    """A crash in pass 3 keeps passes 1-2 on disk."""
    out = tmp_path / "r.csv"
    real, n = loop.run_pass, []

    def flaky(*a, **kw):
        n.append(1)
        if len(n) == 3:
            raise RuntimeError("crash")
        return real(*a, **kw)

    monkeypatch.setattr(loop, "run_pass", flaky)
    with pytest.raises(RuntimeError):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--warmup", "1", "--repeats", "3", "--csv", str(out)])
    assert len(list(csv.DictReader(open(out)))) == 2


def test_verify_crash_keeps_held_rows_as_not_run(corpus, tmp_path, monkeypatch):
    """--verify holds rows back for the check; a crash in a later pass (so no
    check ever runs) still writes the finished passes, marked not_run."""
    out = tmp_path / "r.csv"
    real, n = loop.run_pass, []

    def flaky(*a, **kw):
        n.append(1)
        if len(n) == 3:
            raise RuntimeError("crash")
        return real(*a, **kw)

    monkeypatch.setattr(loop, "run_pass", flaky)
    with pytest.raises(RuntimeError):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--warmup", "1", "--repeats", "3", "--verify",
                  "--csv", str(out)])
    assert [r["verified"] for r in csv.DictReader(open(out))] == ["not_run"] * 2
    trace = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert len(trace) == 2 * len(ROWS) and {r["verified"] for r in trace} == {"not_run"}


def test_no_abbreviated_options():
    """An abbreviation (--back for --backend) is refused, so sweep.py's
    pass-through check by full name cannot be dodged."""
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y",
                        "--back", "cuvs"])


def _fp16_pair(tmp_path, payload_bytes=0):
    """The same corpus twice: stored list<halffloat>, and stored list<float>
    holding the fp16-rounded values (what a host widen produces)."""
    import make_dummy
    xs = [make_dummy.unit_vectors(np.random.default_rng([11, i]), n, 24)
          for i, n in enumerate([500, 7, 900])]
    for name, vt in (("f16", np.float16), ("f32", np.float32)):
        os_dir = tmp_path / name
        os_dir.mkdir()
        for i, x in enumerate(xs):
            make_dummy.write_vectors(str(os_dir / f"part-{i:05d}.parquet"),
                                     x.astype(np.float16).astype(vt),
                                     row_group_size=200, value_dtype=vt,
                                     payload_bytes=payload_bytes)
    return str(tmp_path / "f16"), str(tmp_path / "f32"), \
        np.concatenate(xs).astype(np.float16).astype(np.float32)


@pytest.mark.parametrize("prefetch", [2, 0])
def test_fp16_stored_equals_fp32_widened(tmp_path, queries, prefetch):
    """Widening fp16 -> fp32 is exact: an fp16-stored corpus gives bit-for-bit
    the results of the fp32-stored copy of the same (rounded) values."""
    f16, f32, xb = _fp16_pair(tmp_path)
    q = queries[:, :24].copy()
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    b = cpu_reference.CpuBackend(K)
    b.setup(q)
    out = {}
    for name, path in (("f16", f16), ("f32", f32)):
        src = ParquetSource(path, prefetch=prefetch, workers=2, dtype=np.float32)
        out[name] = loop.run_pass(src, b, len(q), K)
    np.testing.assert_array_equal(out["f16"][0], out["f32"][0])
    np.testing.assert_array_equal(out["f16"][1], out["f32"][1])
    check_against_oneshot(out["f16"][0], out["f16"][1], q, xb)
    # cpu: no events, an f32 corpus is never widened: blank, not 0
    assert out["f16"][2]["widen_s"] > 0 and out["f32"][2]["widen_s"] == ""


def test_cli_fp16_corpus_host_widen(tmp_path):
    f16, _, _ = _fp16_pair(tmp_path, payload_bytes=300)
    out = tmp_path / "r.csv"
    for pf in ("0", "2"):
        run.main(["--regime", "stream", "--corpus", f16, "--backend", "cpu",
                  "--nq", "16", "--warmup", "0", "--repeats", "1", "--csv", str(out),
                  "--prefetch", pf, "--workers", "2", "--verify"])
    rows = list(csv.DictReader(open(out)))
    for r in rows:
        assert r["stored_dtype"] == "f16" and r["dtype"] == "f32"
        assert r["widen_where"] == "host" and r["read_mode_effective"] == "native"
        assert float(r["column_share"]) < 0.5 and r["d"] == "24"
        assert float(r["widen_s"]) > 0 and float(r["decode_s"]) > 0


def test_widen_is_on_the_device_for_gpu_backends():
    base = ["--regime", "stream", "--corpus", "x", "--csv", "y"]
    for gone in (["--widen", "device"], ["--queries", "per-call"]):
        with pytest.raises(SystemExit):
            run.parse_args(base + gone)
    assert run.parse_args(base).widen == "device"
    assert run.parse_args(base + ["--backend", "cpu"]).widen == "host"


def test_query_file_skips_corpus_probe(corpus, tmp_path, monkeypatch):
    """With --query-file, d comes from the queries (checked against the
    footer): no corpus column chunk is downloaded before the timed passes."""
    import make_dummy
    q = make_dummy.make_queries(str(tmp_path / "q.parquet"), 10, 32)
    seen = []
    real = sources.read_file
    monkeypatch.setattr(results_io, "read_file",
                        lambda fs, p, *a, **kw: seen.append(p) or real(fs, p, *a, **kw))
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--query-file", q, "--warmup", "0", "--repeats", "1",
              "--csv", str(tmp_path / "r.csv")])
    assert seen == [q]


def test_corpus_dim_falls_back_to_a_read(corpus, monkeypatch):
    import types
    from bench.data import sources
    fs, refs = sources.list_parquet(corpus[0])
    foot = sources.Footer(1, 1, 1, None, "float")
    args = types.SimpleNamespace(dim=None, column="embedding")
    assert results_io.corpus_dim(args, None, foot, fs, refs[0], "arrow") == 32
    args.dim = 32
    assert results_io.corpus_dim(args, None, sources.Footer(1, 1, 1, 32, "float"),
                          fs, refs[0], "arrow") == 32
    with pytest.raises(ValueError, match="--dim are 32-D, corpus is 16-D"):
        results_io.corpus_dim(args, None, sources.Footer(1, 1, 1, 16, "float"),
                       fs, refs[0], "arrow")


def test_other_share_warning_printed(corpus, tmp_path, monkeypatch, capsys):
    real = loop.run_pass

    def leaky(*a, **kw):
        D, I, info = real(*a, **kw)
        info["other_s"] = info["total_s"] * 0.5
        return D, I, info

    monkeypatch.setattr(loop, "run_pass", leaky)
    run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
              "--nq", "8", "--warmup", "1", "--repeats", "1",
              "--csv", str(tmp_path / "r.csv")])
    err = capsys.readouterr().err
    assert err.count("WARNING: other_s is 50.0% of total") == 1  # not for warmup


# ------------------------------------------------- --verify, guards

def test_compare_results_allows_swaps_within_tolerance():
    """Both the oracle and the engine may be off by atol + rtol * |D|: an id
    swap between scores 1.5e-5 apart (< 2 * tol) is legal, not an I
    mismatch; a wrong id in the same tie zone still fails."""
    D_ref = np.array([[1.0, 0.9, 0.9 - 1.5e-5]], np.float32)
    I_ref = np.array([[7, 8, 9]])
    D = np.array([[1.0, 0.9 - 0.75e-5]], np.float32)
    assert verify.compare_results(D, np.array([[7, 9]]), D_ref, I_ref) == []
    assert verify.compare_results(D, np.array([[7, 8]]), D_ref, I_ref) == []
    # id 3 is outside the oracle's k+1: rescored in xb (0.5, far below the
    # tie zone) it fails; without xb it cannot be checked and fails too
    xq = np.array([[1.0, 0.0]], np.float32)
    xb = np.zeros((10, 2), np.float32)
    xb[:, 0] = 0.5
    xb[7, 0], xb[8, 0], xb[9, 0] = 1.0, 0.9, 0.9 - 1.5e-5
    bad = np.array([[7, 3]])
    assert any("tie zone" in p for p in verify.compare_results(D, bad, D_ref, I_ref, xq=xq, xb=xb))
    assert verify.compare_results(D, bad, D_ref, I_ref)
    # ... but an id tied with the boundary outside the k+1 is a valid choice
    xb[3, 0] = 0.9 - 1.5e-5
    assert verify.compare_results(D, bad, D_ref, I_ref, xq=xq, xb=xb) == []
    # an id clearly above the tie zone (7) may never be dropped, and no id twice
    tol = 1e-5 + verify.VERIFY_RTOL * 0.9
    assert "missing" in verify._tie_zone_problem(0, D_ref[0, 1:], [8, 9], D_ref[0], I_ref[0],
                                              2, tol, None, None)
    assert "duplicate" in verify._tie_zone_problem(0, D[0], [7, 7], D_ref[0], I_ref[0], 2, tol,
                                                None, None)


def _verify_argv(corpus, out, *extra):
    return ["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
            "--warmup", "1", "--repeats", "2", "--verify", "--csv", str(out), *extra]


def test_verify_memory_guard_refuses_up_front(corpus, tmp_path, monkeypatch):
    """--verify whose oracle will not fit in MemAvailable is refused before
    any pass, naming the estimate and --max-files."""
    passes, avail = [], guards.mem_available_bytes
    monkeypatch.setattr(loop, "run_pass", lambda *a, **kw: passes.append(1))
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: 10_000)
    out = tmp_path / "r.csv"
    argv = _verify_argv(corpus, out)
    with pytest.raises(SystemExit, match=r"--verify: the run needs .*--max-files"):
        run.main(argv)
    assert passes == [] and not out.exists()
    need = guards.verify_memory_need(sum(ROWS), 32, 8, K, 3, 4)
    # corpus (4 B) + the float64 oracle (queries 8 B, D / I 16 B a cell, a
    # row tile of every row at 8 B + an 8-query score tile at 32 B a cell)
    # + the engine's fp32 queries + the passes' D / I + the allowance
    rt = sum(ROWS)
    assert need == (sum(ROWS) * 32 * 4 + 8 * 32 * 8 + 8 * (K + 1) * 16 + rt * 32 * 8
                    + 8 * rt * 32 + 8 * 32 * 4 + 3 * 8 * K * 12
                    + guards.VERIFY_PROCESS_ALLOWANCE)
    with open(tmp_path / "meminfo", "w") as f:
        f.write("MemTotal:  100 kB\nMemAvailable:  2048 kB\n")
    assert avail(str(tmp_path / "meminfo")) == 2048 * 1024
    assert avail(str(tmp_path / "missing")) is None


@pytest.mark.parametrize("exc", [MemoryError, KeyboardInterrupt, OSError])
def test_verify_check_raising_marks_error(corpus, tmp_path, monkeypatch, exc):
    """A check that raises never judged the results: rows say verified=error
    (not failed), the exception propagates, the sidecars are gone."""
    def boom(*a, **kw):
        raise exc("oracle died")

    monkeypatch.setattr(verify, "oneshot_reference", boom)
    out = tmp_path / "r.csv"
    with pytest.raises(exc):
        run.main(_verify_argv(corpus, out))
    assert [r["verified"] for r in csv.DictReader(open(out))] == ["error"] * 3
    trace = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert {r["verified"] for r in trace} == {"error"}
    assert not any(os.path.exists(p) for p in results_io.pending_paths(str(out),
                                                                results_io.trace_path(str(out))))


def test_verify_pending_sidecars(corpus, tmp_path, monkeypatch):
    """While the check runs the rows sit in <csv>.pending (verified=pending),
    not in the CSV; afterwards the CSV has them and the sidecars are gone. A
    surviving sidecar (a SIGKILLed run) blocks the next --verify run."""
    out = tmp_path / "r.csv"
    pend, tpend = results_io.pending_paths(str(out), results_io.trace_path(str(out)))
    seen = []
    real = verify.compare_results

    def spy(*a, **kw):
        seen.append(([r["verified"] for r in csv.DictReader(open(pend))],
                     len(list(csv.DictReader(open(tpend)))), out.exists()))
        return real(*a, **kw)

    monkeypatch.setattr(verify, "compare_results", spy)
    run.main(_verify_argv(corpus, out))
    assert seen == [(["pending"] * 3, 3 * len(ROWS), False)]
    assert [r["verified"] for r in csv.DictReader(open(out))] == ["ok"] * 3
    assert not os.path.exists(pend) and not os.path.exists(tpend)
    with open(pend, "w") as f:
        f.write("x\n")
    with pytest.raises(SystemExit, match="killed before its check"):
        run.main(_verify_argv(corpus, out))
    run.main(_verify_argv(corpus, out)[:-3] + ["--csv", str(out)])  # no --verify: unaffected


def test_verify_checks_every_pass(corpus, tmp_path, monkeypatch, capsys):
    """An earlier pass that alone went wrong is caught: its rows are failed,
    the others ok, and the run exits non-zero naming it."""
    real, n = loop.run_pass, []

    def corrupt_first(*a, **kw):
        D, I, info = real(*a, **kw)
        n.append(1)
        if len(n) == 1:
            I = I.copy()
            I[0] = I[0, ::-1] + 1  # wrong ids for query 0 (D untouched)
        return D, I, info

    monkeypatch.setattr(loop, "run_pass", corrupt_first)
    out = tmp_path / "r.csv"
    with pytest.raises(SystemExit, match=r"VERIFY FAILED: I mismatch on query 0.*warmup 0"):
        run.main(_verify_argv(corpus, out))
    assert [r["verified"] for r in csv.DictReader(open(out))] == ["failed", "ok", "ok"]
    assert "warmup 0's results differ from the last pass's" in capsys.readouterr().err


def test_readers_column(corpus, tmp_path):
    out = tmp_path / "r.csv"
    for argv in (["--regime", "stream", "--prefetch", "0"],
                 ["--regime", "stream", "--prefetch", "3", "--workers", "2"]):
        run.main(argv + ["--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
                         "--warmup", "0", "--repeats", "1", "--csv", str(out)])
    assert [r["readers"] for r in csv.DictReader(open(out))] == ["0", "2"]


@pytest.mark.parametrize("argv", [
    ["--nq", "0"], ["--warmup", "-1"], ["--repeats", "-1"], ["--warmup", "0", "--repeats", "0"],
    ["--temp-mem", "-1"], ["--omp-threads", "-1"],
    ["--dim", "0"], ["--workers", "0"], ["--range-concurrency", "0"],
])
def test_numeric_arg_guards(argv):
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y"] + argv)


def test_numeric_arg_zero_defaults_accepted():
    a = run.parse_args(["--regime", "stream", "--corpus", "x", "--csv", "y", "--omp-threads",
                        "0", "--temp-mem", "0", "--warmup", "0", "--repeats", "1"])
    assert a.omp_threads == 0 and a.temp_mem == 0


# ------------------------------------------------- --verify, write_held

def _sorted_case(seed=0, n=200, d=16, nq=5, k=4):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d)).astype(np.float32)
    xq = rng.standard_normal((nq, d)).astype(np.float32)
    from bench.data.sources import Block
    D_ref, I_ref, xq_ref, xbc = verify.oneshot_reference(xq, [Block(0, xb)], k + 1)
    return xq_ref, xbc, D_ref, I_ref, k


def test_compare_results_checks_every_pair():
    """Every returned (D, I) pair must carry the id's own reference score, so ids reversed
    (D still sorted) or one pair of ids swapped fail."""
    xq, xb, D_ref, I_ref, k = _sorted_case()
    D, I = D_ref[:, :k].copy(), I_ref[:, :k].copy()
    assert verify.compare_results(D, I, D_ref, I_ref, xq=xq, xb=xb) == []
    rev = verify.compare_results(D, I[:, ::-1].copy(), D_ref, I_ref, xq=xq, xb=xb)
    assert len(rev) == len(D) and all("returned with" in p for p in rev)
    swapped = I.copy()
    swapped[2, [0, 1]] = swapped[2, [1, 0]]
    probs = verify.compare_results(D, swapped, D_ref, I_ref, xq=xq, xb=xb)
    assert len(probs) == 1 and probs[0].startswith("I mismatch on query 2 (id ")
    # a swap within the tolerance is no failure (scores 1e-6 apart)
    D2, I2, D2r, I2r = (np.array([[1.0, 0.9, 0.9 - 1e-6]], np.float32)[:, :2],
                        np.array([[7, 9]]), np.array([[1.0, 0.9, 0.9 - 1e-6]], np.float32),
                        np.array([[7, 8, 9]]))
    assert verify.compare_results(D2, I2, D2r, I2r) == []


def _tie_case(a, b, kth=0.1, atol=1e-5):
    """Reference top-5 [1, 0.5, kth + a, kth, kth]; the engine returns ids
    0, 1, 3 and id 5 (score kth - b, outside the reference's k + 1), dropping
    id 2 (kth + a)."""
    xb = np.array([[1.0], [0.5], [kth + a], [kth], [kth], [kth - b]], np.float64)
    xq = np.ones((1, 1), np.float32)
    D_ref = np.array([[1.0, 0.5, kth + a, kth, kth]], np.float32)
    I_ref = np.array([[0, 1, 2, 3, 4]])
    D = np.array([[1.0, 0.5, kth, kth - b]], np.float32)
    return D, np.array([[0, 1, 3, 5]]), D_ref, I_ref, xq, xb.astype(np.float32), atol


def test_tie_zone_needs_a_consistent_choice():
    """cmp2: dropping id 2 (kth + 1.9 tol) for id 5 (kth - 1.9 tol), 3.8 tol
    apart, passed the old loose 2 * tol bounds; the dropped id must be within
    tol(a) + tol(b) of every extra one."""
    tol = 1e-5 + verify.VERIFY_RTOL * 0.1
    D, I, D_ref, I_ref, xq, xb, atol = _tie_case(1.9 * tol, 1.9 * tol)
    D[0, 2:] = [0.1 + 0.95 * tol, 0.1 - 0.95 * tol]  # every D within tol of its slot
    probs = verify.compare_results(D, I, D_ref, I_ref, atol=atol, xq=xq, xb=xb)
    assert len(probs) == 1 and "dropped for id 5" in probs[0] and "tie zone" in probs[0]
    # a legit boundary swap: 2 and 5 within 2 tol of each other
    D, I, D_ref, I_ref, xq, xb, atol = _tie_case(0.9 * tol, 0.9 * tol)
    D[0, 2:] = [0.1, 0.1 - 0.9 * tol]
    assert verify.compare_results(D, I, D_ref, I_ref, atol=atol, xq=xq, xb=xb) == []


def test_oneshot_reference_has_no_stored_dtype_concat(monkeypatch):
    """The float64 oracle searches the blocks in place: no np.concatenate
    (no corpus copy at all), and its row accessor gives the stored rows
    widened to float64."""
    from bench.data.sources import Block
    rng = np.random.default_rng(3)
    xs = [make_unit(rng, n) for n in (50, 7, 30)]
    blocks, off = [], 0
    for x in xs:
        blocks.append(Block(off, x.astype(np.float16)))
        off += len(x)
    full = np.concatenate([x.astype(np.float16) for x in xs]).astype(np.float32)

    real_cat = np.concatenate

    def boom(arrays, *a, **kw):
        # the running top-k merge concatenates (nq, k) lists; a CORPUS-shaped
        # concatenation (rows x d) would be a copy
        if any(np.ndim(x) == 2 and np.shape(x)[1] == 16 for x in arrays):
            raise AssertionError("np.concatenate of corpus rows in oneshot_reference")
        return real_cat(arrays, *a, **kw)

    monkeypatch.setattr(np, "concatenate", boom)
    q = make_unit(rng, 4)
    D, I, xq_ref, xb = verify.oneshot_reference(q, blocks, 5)
    monkeypatch.undo()
    assert len(xb) == len(full) and D.dtype == np.float64
    assert all(np.array_equal(xb[i], full[i].astype(np.float64)) for i in range(len(full)))
    S = q.astype(np.float64) @ full.astype(np.float64).T
    I2 = np.argsort(-S, axis=1, kind="stable")[:, :5]
    np.testing.assert_array_equal(I, I2)
    np.testing.assert_array_equal(D, np.take_along_axis(S, I2, 1))


def make_unit(rng, n, d=16):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


MEASURE = r"""
import sys, tracemalloc
sys.path.insert(0, sys.argv[1])
import numpy as np, pyarrow as pa
from bench.systems import harness_cli as run
from bench.verify import oracle as verify
from bench.data.sources import InMemorySource, list_parquet, read_footer
corpus, mode, nq, k = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
fs, files = list_parquet(corpus)
footers = {r.path: read_footer(fs, r.path, "embedding") for r in files}
d = footers[files[0].path].dim
xq = np.random.default_rng(0).standard_normal((nq, d)).astype(np.float32)
# warm-up on one file: lazy imports and caches off the books
w = InMemorySource.load(corpus, "embedding", None, mode, fs=fs, files=files[:1])
verify.oneshot_reference(xq, w.blocks, k + 1)
del w
pool = pa.default_memory_pool()
base = pool.bytes_allocated()
tracemalloc.start()
t0 = tracemalloc.get_traced_memory()[0]
src = InMemorySource.load(corpus, "embedding", None, mode, fs=fs, files=files)
D, I, xq_ref, xb = verify.oneshot_reference(xq, src.blocks, k + 1)
np_peak = tracemalloc.get_traced_memory()[1] - t0
# lifetime max (warm-up included): never below this load's own peak
arrow_peak = pool.max_memory() - base
print(np_peak + arrow_peak)
"""


# (value dtype, rows per file): one row group per file (make_dummy's
# default). 20,000 x 128 list<float> is a size where Arrow's builder growth
# peaks (~25 B per value, above the 18 B the guard once assumed for every
# width); ~10 MB blocks
VERIFY_SHAPES = {"f16": (np.float16, 4000), "f32": (np.float32, 20000)}


@pytest.mark.slow
@pytest.mark.parametrize("vt", sorted(VERIFY_SHAPES))
def test_verify_memory_estimate_covers_the_oracle(tmp_path, monkeypatch, vt):
    """The guard's estimate (stored-dtype blocks + fp32 copy + the decode
    transient: the oracle's arrow reload) is at least what the oracle
    actually allocates: numpy's peak (tracemalloc) plus Arrow's pool peak,
    measured in a fresh process. The old estimate (no transient, plus an
    fp16 concatenation it never counted) fell short of it."""
    import argparse
    import subprocess
    import sys
    import make_dummy
    dtype, n = VERIFY_SHAPES[vt]
    corpus = str(tmp_path / "c")
    make_dummy.make_corpus(corpus, 3, n, 128, seed=2, value_dtype=dtype)
    harness = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-c", MEASURE, harness, corpus, "arrow", "16", "5"],
                         capture_output=True, text=True, check=True)
    measured = int(out.stdout)
    rows, d = 3 * n, 128
    # the guard refuses whenever MemAvailable is below what the oracle takes
    fs, files = sources.list_parquet(corpus)
    footers = {r.path: sources.read_footer(fs, r.path, "embedding") for r in files}
    args = argparse.Namespace(prefetch=0, k=5, warmup=0, repeats=0, workers=2, backend="cpu")
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: measured - 1)
    with pytest.raises(SystemExit, match="decode transient"):
        guards.verify_memory_guard(args, files, footers, d, 16, None)


def test_verify_memory_guard_counts_readers_and_cuda(corpus, monkeypatch):
    """With prefetch: each reader process keeps its own pool (one file's
    native read transient each, kept until the readers are closed) and the
    ring, in the passes' phase; inline reads are this process's. The
    oracle's reload is the other phase. --backend gpu / cuvs add the CUDA
    host-context allowance to both."""
    import argparse
    fs, files = sources.list_parquet(corpus[0])
    footers = {r.path: sources.read_footer(fs, r.path, "embedding") for r in files}
    d = footers[files[0].path].dim
    seen, real = [], guards.verify_memory_need
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: 0)
    monkeypatch.setattr(guards, "verify_memory_need",
                        lambda *a, **kw: seen.append((a, kw, real(*a, **kw))) or seen[-1][2])

    def need(**kw):
        a = dict(prefetch=0, k=K, warmup=0, repeats=1, workers=8, backend="cpu") | kw
        with pytest.raises(SystemExit, match="needs about") as e:
            guards.verify_memory_guard(argparse.Namespace(**a), files, footers, d, 8, None)
        return seen[-1], str(e.value)

    one = guards.native_read_bytes(files, footers, d, 4, slot=True)
    fit = guards.native_read_bytes(files, footers, d, 4)
    block = max(footers[r.path].num_rows for r in files) * d * 4
    readers = min(2, len(files))
    (a0, kw0, base), _ = need()
    assert a0[6:9] == (0, fit, 0) and kw0["reload_bytes"] > 0  # inline reads; the reload
    (a, kw, got), msg = need(prefetch=2)
    # ring + transients; the run process decodes nothing during the passes
    assert a[6:9] == ((readers + 1) * block, 0, readers * one)
    assert "reader-process transients" in msg
    assert kw["reload_bytes"] == kw0["reload_bytes"]
    assert got == real(*a, **kw) >= base
    (a, _, got), msg = need(backend="gpu")
    assert got - base == guards.CUDA_HOST_ALLOWANCE and "CUDA host-context allowance" in msg
    assert a[9] == guards.CUDA_HOST_ALLOWANCE and need(backend="cuvs")[0][2] == got


def test_verify_memory_need_two_phases():
    """The passes (ring, reader / own transients) and the
    oracle (corpus + the larger of the reload transient and the float64
    oracle's own allocations) never overlap: the larger phase counts, plus
    what both hold."""
    rows, d, nq, k, passes = 1000, 64, 10, 5, 2
    shared = nq * d * 4 + passes * nq * k * 12 + guards.VERIFY_PROCESS_ALLOWANCE
    # float64 queries + float64 D / int64 I (k + 1) + a 1000-row float64 tile
    # + a 10 x 1000 score tile at 32 B a cell
    own = nq * d * 8 + nq * (k + 1) * 16 + rows * d * 8 + nq * rows * 32
    assert guards.oracle_bytes(d, nq, k, rows) == own
    oracle = rows * d * 2 + own
    need = guards.verify_memory_need
    assert need(rows, d, nq, k, passes, 2) == shared + oracle
    # a reload transient below the oracle's own is not added; above it, it replaces it
    assert need(rows, d, nq, k, passes, 2, reload_bytes=own - 1) == shared + oracle
    assert need(rows, d, nq, k, passes, 2, reload_bytes=own + 7) == shared + oracle + 7
    # the tile's rows are the largest file's (oracle_rows), not the corpus's
    assert need(rows, d, nq, k, passes, 2, oracle_rows=100) == shared + oracle - (
        900 * d * 8 + nq * 900 * 32)
    # passes smaller than the oracle: the oracle counts, not the sum
    assert need(rows, d, nq, k, passes, 2, 100, 200, 300) == shared + oracle
    big = 10 * oracle
    assert need(rows, d, nq, k, passes, 2, big, 200, 300, 9) == shared + 9 + big + 500


def test_write_held_defers_signals(tmp_path, monkeypatch):
    """SIGTERM arriving while the held rows are written takes effect only
    once the CSV, the trace and the sidecar removal are all done: the CSV
    never holds part of the held rows while the sidecars hold them all."""
    import signal
    out = str(tmp_path / "r.csv")
    trace_csv = results_io.trace_path(out)
    base = {c: "x" for c in schema.CSV_FIELDS}
    held = []
    for rep in range(3):
        row = dict(base, repeat=rep)
        trace = (0.0, [(f"f{rep}", 1, 0.0, 0.0, 0.0, 0.0, 0.0, None, (None, None), 0.0)])
        held.append((row, (0, rep, 0.0, trace)))
        results_io.append_pending(out, trace_csv, row, 0, rep, 0.0, trace)
    real, calls = results_io.append_csv, []

    def append_then_kill(path, rows, fields=schema.CSV_FIELDS):
        real(path, rows, fields)
        calls.append(len(rows))
        if len(calls) == 1:
            signal_main_thread(signal.SIGTERM)

    monkeypatch.setattr(results_io, "append_csv", append_then_kill)
    with run.exit_on_signals():
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as e:
            run.write_held(out, trace_csv, held, "ok")
        assert signal.getsignal(signal.SIGTERM) is handler  # restored
    assert e.value.code == 128 + signal.SIGTERM
    assert calls == [3, 3]  # one append each: all rows, all trace rows
    assert [r["repeat"] for r in csv.DictReader(open(out))] == ["0", "1", "2"]
    assert len(list(csv.DictReader(open(trace_csv)))) == 3
    assert not any(os.path.exists(p) for p in results_io.pending_paths(out, trace_csv))


def signal_main_thread(sig):
    """Send ``sig`` to the main thread, not the process. A process-directed
    os.kill(getpid()) is taken by whichever thread has it unblocked; once
    the pytest process holds native threads (imported OpenMP / BLAS pools,
    Arrow), the main thread's mask in signals_deferred sends it to one of
    those, whose C handler can run late: after signals_deferred has put the
    old handlers back, so the old handler, not the recorder, sees it (a
    SIGINT then outranks a deferred SIGTERM: test flake). Thread-directed,
    it stays pending on the blocked main thread until the unblock, which
    is the deterministic path these tests assert."""
    import signal
    import threading
    signal.pthread_kill(threading.main_thread().ident, sig)


def test_exit_on_signals_keeps_an_inherited_ignore():
    """A signal ignored when the run starts (SIGHUP under nohup) stays ignored:
    exit_on_signals must not install its SystemExit handler over SIG_IGN."""
    import signal
    old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with run.exit_on_signals():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_IGN
            os.kill(os.getpid(), signal.SIGHUP)
            with run.signals_deferred():  # deferred, then still ignored
                signal_main_thread(signal.SIGHUP)
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, old)


@pytest.mark.parametrize("order", [("SIGINT", "SIGTERM"), ("SIGTERM", "SIGINT"),
                                   ("SIGINT", "SIGHUP")])
def test_signals_deferred_reraises_the_most_severe(order):
    """Several signals deferred: the most severe is delivered (SIGTERM /
    SIGHUP over SIGINT), not merely the first recorded (on unblocking,
    pending signals arrive lowest number first: SIGINT before SIGTERM)."""
    import signal
    with run.exit_on_signals():
        # KeyboardInterrupt caught too: the wrong one must fail the test, not
        # stop the session
        with pytest.raises((SystemExit, KeyboardInterrupt)) as e:
            with run.signals_deferred():
                for name in order:
                    signal_main_thread(getattr(signal, name))
    worst = signal.SIGHUP if "SIGHUP" in order else signal.SIGTERM
    assert e.type is SystemExit and e.value.code == 128 + worst


@pytest.mark.parametrize("order", [("SIGINT", "SIGTERM"), ("SIGTERM", "SIGINT")])
def test_signals_deferred_records_a_signal_taken_by_another_thread(order):
    """A signal taken by another thread during the body (its mask is not
    blocked; process-directed signals land there) is recorded, not run
    through the old handler: the most severe still wins. The other
    thread signals itself, so its C handler has run (signal tripped)
    before the body goes on: deterministic, unlike os.kill(getpid())."""
    import signal
    import threading
    first, second = (getattr(signal, n) for n in order)
    go = threading.Event()

    def take():  # started before the body: unblocked mask inherited
        go.wait()
        signal.pthread_kill(threading.get_ident(), first)

    t = threading.Thread(target=take)
    t.start()
    with run.exit_on_signals():
        with pytest.raises((SystemExit, KeyboardInterrupt)) as e:
            with run.signals_deferred():
                go.set()
                t.join()
                signal_main_thread(second)
    assert e.type is SystemExit and e.value.code == 128 + signal.SIGTERM


def test_signals_deferred_skips_an_ignored_signal():
    """An ignored SIGHUP and a Ctrl-C deferred together: the Ctrl-C is
    delivered (the ignored one ranks higher but must not swallow it).
    SIGINT is set to Python's default handler for the test: run as a
    background job, the test process inherits it ignored."""
    import signal
    old = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    old_int = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            with run.signals_deferred():
                signal_main_thread(signal.SIGINT)
                signal_main_thread(signal.SIGHUP)
    finally:
        signal.signal(signal.SIGHUP, old)
        signal.signal(signal.SIGINT, old_int)


def test_pass_append_defers_signals(corpus, tmp_path, monkeypatch):
    """Without --verify, a SIGTERM arriving between a pass's CSV row and its
    trace rows takes effect once both are written: the CSV and the trace
    hold the same passes (both, never one without the other)."""
    import signal
    real, calls = results_io.append_csv, []

    def append_then_kill(path, rows, fields=schema.CSV_FIELDS):
        real(path, rows, fields)
        calls.append(path)
        if len(calls) == 1:  # the first pass's CSV row is out, its trace not yet
            signal_main_thread(signal.SIGTERM)

    monkeypatch.setattr(results_io, "append_csv", append_then_kill)
    out = tmp_path / "r.csv"
    with pytest.raises(SystemExit) as e:
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--warmup", "0", "--repeats", "3", "--csv", str(out)])
    assert e.value.code == 128 + signal.SIGTERM
    rows = list(csv.DictReader(open(out)))
    trows = list(csv.DictReader(open(results_io.trace_path(str(out)))))
    assert len(rows) == 1 and len(trows) == len(ROWS)
    assert {t["t_start_unix"] for t in trows} == {rows[0]["t_start_unix"]}


def test_generated_queries_are_not_corpus_file_0(tmp_path):
    """SeedSequence([S, 0]) == SeedSequence(S): make_dummy's file 0 and the harness's
    generated queries must not be the same vectors for one seed."""
    import argparse
    import pyarrow.parquet as pq
    import make_dummy
    paths = make_dummy.make_corpus(str(tmp_path / "c"), 1, 20, 8, seed=5)
    x = np.stack(pq.read_table(paths[0]).column("embedding").to_numpy(zero_copy_only=False))
    q = results_io.random_queries(argparse.Namespace(seed=5, nq=20), 8, np.float32)
    assert not np.allclose(q, x)
    assert np.allclose(q, results_io.random_queries(argparse.Namespace(seed=5, nq=20), 8,
                                             np.float32))  # still deterministic


# ------------------------------------------------- --range-mib / --save-results



def test_save_results_path_checked_up_front(tmp_path, capsys):
    """--save-results: {backend} / {engine} filled in, .npz appended (with a
    note) when missing, and a missing / unwritable directory or an unknown
    placeholder refused in parse_args, before any work."""
    base = ["--regime", "stream", "--corpus", "c", "--csv", "x.csv", "--backend", "cpu"]
    a = run.parse_args(base + ["--save-results", str(tmp_path / "res-{backend}-{engine}")])
    assert a.save_results == str(tmp_path / "res-cpu-faiss.npz")
    assert "res-cpu-faiss.npz" in capsys.readouterr().err
    a = run.parse_args(base[:-1] + ["cuvs", "--save-results", str(tmp_path / "{engine}.npz")])
    assert a.save_results == str(tmp_path / "cuvs.npz")
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        for bad in (tmp_path / "missing" / "r.npz", tmp_path / "{bogus}.npz", ro / "r.npz",
                    tmp_path):
            with pytest.raises(SystemExit):
                run.parse_args(base + ["--save-results", str(bad)])
            assert "--save-results" in capsys.readouterr().err, bad
    finally:
        ro.chmod(0o700)


@pytest.mark.parametrize("verify", [[], ["--verify"]])
def test_save_results_written_after_the_row(corpus, tmp_path, monkeypatch, verify):
    """The last pass's D / I go to --save-results AFTER its CSV row is
    written (or held for --verify); a failed write warns and loses nothing:
    every row is in the CSV and --verify still runs."""
    out = tmp_path / "r.csv"
    argv = ["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu", "--nq", "8",
            "--warmup", "1", "--repeats", "2", "--csv", str(out),
            "--save-results", str(tmp_path / "res-{backend}")] + verify
    run.main(argv)
    with np.load(tmp_path / "res-cpu.npz") as z:
        assert z["D"].shape == z["I"].shape == (8, K)
    rows_at_save = []
    real = np.savez

    def failing(path, **kw):
        rows_at_save.append(len(list(csv.DictReader(open(out)))))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np, "savez", failing)
    run.main(argv)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 6
    assert [r["verified"] for r in rows[3:]] == (["ok"] * 3 if verify else ["not_run"] * 3)
    # without --verify the row is in the CSV before the save (with it, held)
    assert rows_at_save == ([3] if verify else [6])
    monkeypatch.setattr(np, "savez", real)


def test_guard_counts_the_cosine_scratch(corpus, monkeypatch):
    """The host cosine normalise scratch (CPU backend: an fp32 buffer of the
    largest block) is in the guard's shared part; ip, gpu and cuvs have none."""
    fs, files = sources.list_parquet(corpus[0])
    footers = {r.path: sources.read_footer(fs, r.path, "embedding") for r in files}
    seen, real = [], guards.verify_memory_need
    monkeypatch.setattr(guards, "verify_memory_need",
                        lambda *a, **kw: seen.append(kw["scratch_bytes"]) or real(*a, **kw))
    monkeypatch.setattr(guards, "mem_available_bytes", lambda path=None: 1 << 50)
    values = max(ROWS) * D
    for backend, metric, want in (("cpu", "cosine", values * 4), ("gpu", "cosine", 0),
                                  ("cuvs", "cosine", 0), ("cpu", "ip", 0)):
        args = argparse.Namespace(prefetch=0, k=K, warmup=0, repeats=1, workers=2,
                                  backend=backend, metric=metric)
        guards.verify_memory_guard(args, files, footers, D, 8, None)
        assert seen[-1] == want, (backend, metric)
    assert real(10, 4, 1, 1, 1, 4, scratch_bytes=1000) - real(10, 4, 1, 1, 1, 4) == 1000
