"""Benchmark exact top-k search across Parquet files.

Supports FAISS, cuVS, SPARe, PyLate, and CPU reference backends for dense,
sparse, multivector, and filtered search. All backends share the same
data-loading, timing, and result-merging pipeline.

Files are prefetched by separate reader processes, and each file's results
are merged into a global top-k. Timing separates data stalls, engine
execution, and merging, with additional GPU and reader metrics.

Optional verification checks results against a float64 reference before
marking them valid.
"""

import argparse
import contextlib
import os
import signal
import socket
import sys
import threading

# Load torch's pip CUDA libraries BEFORE faiss / cuvs, so every engine runs on
# one copy of each library (faiss first would map conda's copies beside
# torch's; a GPU run with two copies is refused). No-op without CUDA torch.
from bench.record import cudalibs
cudalibs.torch_first()
import faiss  # noqa: E402
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bench.data.sources import (MAX_PREFETCH, NATIVE_CONCURRENCY, NATIVE_PART_BYTES, RANGE_SCOPES,
                          READ_MODES, native_import, native_url, InMemorySource, ParquetSource,
                          choose_read_mode, configure_arrow_pools, default_workers, list_parquet,
                          pool_split_str, read_footer, rss_bytes, usable_cpu_count)
from bench.data.formats import multivector as multivec
from bench.data.formats import sparse
from bench.timing import now
from bench.systems import guards, loop  # noqa: E402
from bench.record import provenance, results_io, schema  # noqa: E402
from bench.verify import oracle as verify  # noqa: E402
import bench.systems.streaming  # noqa: E402 - qualified in full: _main has a local `base` (the row)
from bench.systems import cpu_reference, cuvs_engine, faiss_engine, pylate_engine, spare_engine  # noqa: E402


def fmt(v, digits=3):
    """A number for the progress line; '-' for a column left empty."""
    return "-" if v == "" else f"{v:.{digits}f}"


def write_held(csv_path, trace_csv, held, status):
    """Write the rows a --verify run held back, marked with ``status`` (one
    string for all passes, or one per held pass), then remove the sidecars.

    CSV and trace rows each go out in one append, and the whole write +
    delete runs with signals deferred, so the CSV never holds part of the
    held rows while the sidecars still hold all of them."""
    statuses = [status] * len(held) if isinstance(status, str) else list(status)
    assert len(statuses) == len(held) and set(statuses) <= set(schema.VERIFIED)
    rows, trows = [], []
    for (row, (warm, rep, t0, trace)), st in zip(held, statuses):
        row["verified"] = st
        rows.append(row)
        trows.extend(loop.trace_rows(row, warm, rep, t0, trace, st))
    with signals_deferred():
        results_io.append_csv(csv_path, rows)
        results_io.append_csv(trace_csv, trows, schema.TRACE_FIELDS)
        for p in results_io.pending_paths(csv_path, trace_csv):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)


# --backend -> its class (save_results_path's {engine}), per --modality
BACKENDS = {"cpu": cpu_reference.CpuBackend, "gpu": faiss_engine.GpuBackend, "cuvs": cuvs_engine.CuvsBackend}
# --backend masked (bench/data/filters/search.py): the filtered CPU reference
from bench.data.filters.search import MaskedBackend as _MaskedBackend  # noqa: E402
BACKENDS["masked"] = _MaskedBackend
SPARSE_BACKENDS = {"cpu": cpu_reference.SparseCpuBackend, "cuvs": cuvs_engine.SparseCuvsBackend,
                   "spare": spare_engine.SparseSpareBackend}
MV_BACKENDS = {"cpu": cpu_reference.MvCpuBackend, "pylate": pylate_engine.MvPylateBackend}
MODALITIES = ("dense", "sparse", "multivector")
# fixed run settings (no flags): blocks are searched in float32; a file larger than
# MAX_FILE_BYTES (on disk) is refused; --verify's absolute tolerance on D
DTYPE = "f32"
MAX_FILE_BYTES = 8 << 30
VERIFY_ATOL = 1e-5
# duckdb_filter's substring pre-check: used when a text column has at most this many phrases
FILTER_PREFILTER_MAX_PHRASES = 8


def h2d_mode(args):
    """The h2d_mode column: pageable / pinned / pinned-overlap, n/a where the
    harness does no H2D of its own."""
    if args.backend in schema.HOST_BACKENDS:
        return "n/a"
    if args.h2d == "pinned":
        return "pinned-overlap" if args.h2d_overlap else "pinned"
    return "pageable"


def _pool_bytes(backend, n):
    """Record the RMM pool's initial size on a cuVS backend: the device-merge guard counts it as
    headroom for CuPy's allocations (rmm_cupy_allocator). A stand-in backend may not take it."""
    try:
        backend.rmm_pool_bytes = n
    except AttributeError:
        pass
    return backend


def make_backend(args, dtype):
    if getattr(args, "modality", "dense") == "multivector":
        if args.backend == "cpu":
            return cpu_reference.MvCpuBackend(args.k, metric=args.metric)
        return pylate_engine.MvPylateBackend(args.k, args.pylate_mode, events=not args.no_gpu_events,
                               metric=args.metric, max_bytes=args.pylate_max_mib << 20,
                               rerank_results=getattr(args, "pylate_rerank_results", None)
                               or "device",
                               query_buckets=getattr(args, "pylate_query_buckets", None) or 1)
    if getattr(args, "modality", "dense") == "sparse":
        if args.backend == "cpu":
            return cpu_reference.SparseCpuBackend(args.k, metric=args.metric)
        if args.backend == "spare":
            return spare_engine.SparseSpareBackend(args.k, args.spare_mode, events=not args.no_gpu_events,
                                      metric=args.metric, max_bytes=args.spare_max_mib << 20,
                                      stage_events=not args.no_spare_events)
        pool = cuvs_engine.cuvs_pool_spec(args)
        # before the backend (its Resources, its self-test) exists
        pool_bytes = cuvs_engine.cuvs_set_rmm_pool(pool) if pool else 0
        return _pool_bytes(cuvs_engine.SparseCuvsBackend(
            args.k, events=not args.no_gpu_events, metric=args.metric,
            batches=getattr(args, "cuvs_sparse_batches", None)), pool_bytes)
    if args.backend == "cpu":
        return cpu_reference.CpuBackend(args.k, dtype, metric=args.metric)
    if args.backend == "masked":
        from bench.data.filters.search import MaskedBackend
        return MaskedBackend(args.k, dtype, metric=args.metric)
    if args.backend == "cuvs":
        pool = cuvs_engine.cuvs_pool_spec(args)
        pool_bytes = cuvs_engine.cuvs_set_rmm_pool(pool) if pool else 0
        return _pool_bytes(cuvs_engine.CuvsBackend(args.k, dtype, events=not args.no_gpu_events,
                                                   metric=args.metric), pool_bytes)
    return faiss_engine.GpuBackend(args.k, dtype, args.temp_mem,
                                   events=not args.no_gpu_events, metric=args.metric)


def parse_args(argv=None):
    # no prefix abbreviations: a guessed option is no way to configure a benchmark
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["stream"], required=True)
    ap.add_argument("--corpus", required=True,
                    help="parquet file or directory; local path or s3://bucket/prefix")
    ap.add_argument("--corpus-recursive", action="store_true",
                    help="list *.parquet under --corpus recursively (sharded layouts "
                         "such as data/000/..data/009/); files are still ordered by "
                         "full path, so global row ids stay deterministic")
    ap.add_argument("--column", default="embedding")
    ap.add_argument("--modality", choices=MODALITIES, default="dense",
                    help="dense = list<float> vector columns (default); sparse = a "
                         "struct<indices: list<int>, values: list<float>> column (FineWeb "
                         "gte's sparse_embedding; --column / --query-column name it), "
                         "searched by --backend cpu (scipy, exact) or cuvs (cuVS's sparse "
                         "brute force through baselines/cuvs/sparse_binding); "
                         "multivector = a list<list<float>> column, one inner list per "
                         "token (pubmed bge-m3's multivector_embedding), scored by MaxSim, "
                         "searched by --backend cpu (numpy, exact) or pylate; both need "
                         "--query-file")
    ap.add_argument("--backend", choices=["gpu", "cpu", "cuvs", "spare", "pylate", "masked"],
                    default="gpu",
                    help="gpu = FAISS knn_gpu; cpu = FAISS knn (for a machine without a GPU); "
                         "cuvs = cuvs.neighbors.brute_force (stock defaults, "
                         "resident only); spare = SPARe (--modality sparse only; its "
                         "released per-query modes or query-batched (ours, "
                         "baselines/spare/spare_batched.py); needs --spare-mode); pylate "
                         "= PyLate's exact float32 MaxSim (--modality multivector only; "
                         "needs --pylate-mode)")
    ap.add_argument("--sparse-project", choices=sparse.PROJECT_MODES, default="off",
                    help="--modality sparse --metric ip: off (default: the engines score "
                         "the file as stored) or query-vocab (ABLATION: every corpus file and "
                         "the queries projected onto the queries' vocabulary by nova-bf's own "
                         "code, imported from nova_bf.compute: columns no query uses dropped, "
                         "the rest renumbered; reader side, per file, inside convert_s, "
                         "measured as project_s). Recorded as sparse_project")
    ap.add_argument("--pylate-mode", choices=pylate_engine.PYLATE_MODES, default=None,
                    help="--backend pylate: rerank-released (pylate.rank.rerank as "
                         "released, every document a candidate: zero padding, NOT exact "
                         "when a document's tokens all score < 0); "
                         "rerank-released-ownpad (the same with documents padded with "
                         "their own tokens: exact); scores-batched (ours: "
                         "colbert_scores over query x document batches + topk). Required, "
                         "recorded as pylate_mode")
    ap.add_argument("--pylate-max-mib", type=int, default=None,
                    help="--backend pylate: device scratch budget of one PyLate call, "
                         f"MiB (default {pylate_engine.PYLATE_DEFAULT_MAX_MIB}): scores-batched sizes its "
                         "query batches / document ranges to it; the rerank modes call "
                         "rerank once per candidate range that fits it (one call when the "
                         "block does); recorded in pylate_batches")
    ap.add_argument("--pylate-query-buckets", type=int, default=None, metavar="N",
                    help="--backend pylate --pylate-mode scores-batched: cut the "
                         "length-sorted queries into N equal-count buckets, each planned "
                         "at its own longest query (default 1: every query padded to the "
                         "longest of all). Same scores; a sweep axis. "
                         "Recorded in pylate_batches")
    ap.add_argument("--pylate-rerank-results", choices=pylate_engine.PYLATE_RERANK_RESULTS, default=None,
                    help="--backend pylate, the rerank modes (an A/B of the round trip): "
                         "device (default: rerank's host results uploaded (nq x k), ids "
                         "globalised on the device, copied back as every engine's) or host "
                         "(kept on the host: ids globalised there, no upload and no copy "
                         "back, d2h_s 0). Recorded as pylate_rerank_results")
    ap.add_argument("--spare-mode", choices=spare_engine.SPARE_MODES, default=None,
                    help="--backend spare: dot (block CSR @ dense query matrix, SpMM) "
                         "or iterative (a batched restructuring of the paper's Algorithm 1: "
                         "CSC + one index_add_ per query group), both batched by us; "
                         "dot-released / iterative-released: SPARe's own "
                         "forward, one query per call, as released; "
                         "iterative-released-nosync: that loop with host-side CSC bounds "
                         "(no per-term device reads). Required, recorded as spare_mode")
    ap.add_argument("--no-spare-events", action="store_true",
                    help="--backend spare, modes dot / iterative / "
                         "iterative-released-nosync: do not record the stage events "
                         "(spare_plan_s / spare_score_s / spare_topk_s stay blank); the "
                         "A/B for their overhead. Recorded as spare_events")
    ap.add_argument("--spare-max-mib", type=int, default=None,
                    help="--backend spare, batched modes: device scratch budget of one "
                         "search call (spare_batched's max_bytes: caps the query batch), "
                         f"MiB (default {spare_engine.SPARE_DEFAULT_MAX_MIB} = cuVS's stock tile); "
                         "recorded in sparse_batches")
    ap.add_argument("--query-file", default=None,
                    help="parquet queries (default: random unit-norm, --seed)")
    ap.add_argument("--query-column", default="embedding")
    ap.add_argument("--nq", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dim", type=int, default=None,
                    help="d for generated queries (default: the first file's "
                         "footer; with --query-file, the queries' d)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", choices=schema.METRICS, default="ip",
                    help="ip = raw inner product (default); cosine = cos(q, x): FAISS "
                         "normalises the queries once and every block inside the call "
                         "(measured as normalize_s), cuVS runs metric='cosine' natively; "
                         "both merge on the similarity")
    ap.add_argument("--gt-ids-column", default=None,
                    help="a list<int> column of --query-file holding each query's "
                         "ground-truth ids (global corpus row positions, best first); "
                         "after each pass recall@k of the merged ids against its first k "
                         "is printed and written as gt_recall (a sanity check, not a "
                         "pass/fail)")
    ap.add_argument("--save-results", default=None, metavar="PATH",
                    help="write the last pass's merged top-k to PATH (.npz with D, I; "
                         "after the pass and its CSV row, outside its timed span) for "
                         "cross-engine checks. {backend} / {engine} in PATH are filled "
                         "in (one file per engine when a sweep interleaves engines); a "
                         "PATH without .npz gets it appended. Its directory must exist "
                         "and be writable (checked up front); a failed write is a "
                         "warning, never a lost row or a skipped --verify")
    ap.add_argument("--prefetch", type=int, default=2,
                    help=f"file reads in flight, the head included (0 = read inline; at "
                         f"most {MAX_PREFETCH}: the reader task pipe and the block ring are "
                         "sized for it)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--cuvs-rmm-pool", default=cuvs_engine.CUVS_RMM_POOL_DEFAULT, metavar="SIZE",
                    help="--backend cuvs (dense and sparse): an RMM pool as the device resource (initial "
                         "size, e.g. 8GiB or 50%% of free memory; it may grow) instead of "
                         "RMM's default per-call cudaMalloc / cudaFree. ON by default "
                         "(%(default)s, what cuVS's own ground-truth tool sets); 'off' = RMM's "
                         "default resource. Recorded in rmm_pool.")
    ap.add_argument("--cuvs-sparse-batches", type=cuvs_engine.parse_cuvs_sparse_batches, default=None,
                    metavar="INDEXxQUERY",
                    help="--modality sparse --backend cuvs: cuVS's sparse_search_params "
                         "batch_size_index x batch_size_query (a parameter cuVS exposes, "
                         "not a source change), e.g. 8192x32768. Either side may be "
                         "'stock' (that side's default, read from the build: 2 << 14 = "
                         "32768 in cuVS 26.10); the index side may be 'file' (one index "
                         "batch per corpus file: its row count). Default: stock on both "
                         "sides. Recorded as sparse_batches")
    ap.add_argument("--cuvs-merge", choices=cuvs_engine.CUVS_MERGES, default=None,
                    help="--backend cuvs: device (default) = cuVS's own exact-search recipe, "
                         "NVIDIA's generate_groundtruth calc_truth: one build + search + "
                         "resources.sync() per file as the reader hands it, the running top-k "
                         "merged on the GPU with CuPy, one copy back at the end; host = the "
                         "per-file search with a host faiss.ResultHeap merge (the per_file_host "
                         "variant: NVIDIA's streaming compute_groundtruth_exact pattern). "
                         "Recorded in cuvs_merge")
    ap.add_argument("--cuvs-sync", choices=cuvs_engine.CUVS_SYNCS, default=None,
                    help="--backend cuvs --cuvs-merge device: per-file (default) = calc_truth's "
                         "resources.sync() after each file; none = no per-file sync (ours, "
                         "off by default): the merge is ordered by the stream alone and the "
                         "host keeps one block in flight, waiting for a block's end only "
                         "before its buffers are reused. Refused with --cuvs-merge host. "
                         "Recorded in cuvs_sync")
    ap.add_argument("--temp-mem", type=int, default=None,
                    help="bytes; StandardGpuResources.setTempMemory (gpu only; 0 = "
                         "noTempMemory; default = FAISS default)")
    ap.add_argument("--allow-duplicate-cuda-libs", action="store_true",
                    help="diagnosis only: run a GPU backend even though the process maps "
                         "two copies of one CUDA library (refused by default: every engine "
                         "must run on torch's pip copies); every copy goes to "
                         "the cuda_lib_duplicates column")
    ap.add_argument("--allow-unpinned-stack", action="store_true",
                    help="diagnosis only: run a GPU backend that was not launched through "
                         "env/paper_run.sh (refused by default: PAPER_STACK_SHA "
                         "unset, another stack's, a stubbed launch, or an interpreter outside "
                         "PAPER_ENV_PREFIX). CPU runs are never refused. Also "
                         "$HARNESS_ALLOW_UNPINNED_STACK (the test suite sets it). The paper_* "
                         "columns record what was set; such rows are not paper numbers")
    ap.add_argument("--no-gpu-events", action="store_true",
                    help="gpu / cuvs: create/record no CUDA events (minimal "
                         "instrumentation, for performance comparisons): "
                         "engine_call is plain wall time ending in one "
                         "torch.cuda.synchronize(); h2d_s / widen_s / search_s / "
                         "d2h_s / engine_gap_s / sync_wait_s / sync_waits are "
                         "left empty. Do not mix with events-on rows in one "
                         "comparison, nor engines measured in different modes")
    ap.add_argument("--omp-threads", type=int, default=None,
                    help="faiss.omp_set_num_threads (default or 0: faiss's current); "
                         "does NOT size Arrow's pool")
    ap.add_argument("--read-mode", choices=READ_MODES, default="native",
                    help="native = nova-bf's rust_fetch reader (the only mode): dense columns "
                         "its native dense path (params.rust_fetch_dense: ranged "
                         "GETs of the column chunks + Rust decode, GIL released), "
                         "sparse / multivector its fetch path (params.rust_fetch_columns: "
                         "rust_fetch downloads the column chunks, pyarrow decodes). A file "
                         "the native reader refuses fails the run")
    ap.add_argument("--range-concurrency", type=int, default=NATIVE_CONCURRENCY,
                    help="the in-flight GET budget of the rust_fetch Clients (default "
                         f"{NATIVE_CONCURRENCY}, nova-bf's NOVA_BF_RUST_FETCH_CONC), split "
                         "by --range-scope")
    ap.add_argument("--range-mib", type=int, default=NATIVE_PART_BYTES >> 20,
                    help="the rust_fetch part size in MiB (default "
                         f"{NATIVE_PART_BYTES >> 20}, nova-bf's NOVA_BF_RUST_FETCH_PART_MIB)")
    ap.add_argument("--range-scope", choices=RANGE_SCOPES, default="total",
                    help="total = the budget is split across concurrent readers "
                         "(the harness's choice); per-file = each reader gets it "
                         "(what nova-bf does: 24 per file x io_workers)")
    ap.add_argument("--max-files", type=int, default=0,
                    help="use only the first N files of the sorted listing (0 = "
                         "all); applies to --verify too")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--warmup-files", type=int, default=0,
                    help="warmup passes read only the first N files (0 = the whole "
                         "corpus, as the timed passes; recorded as warmup_files). "
                         "Warmup rows then cover N files (n_files), are never "
                         "--verify'd and get no gt_recall")
    ap.add_argument("--h2d", choices=schema.H2D_MODES, default="pageable",
                    help="how a block reaches the GPU (resident gpu / cuvs, stream "
                         "regime): pageable = torch's synchronous copy from the "
                         "block's pageable host memory (default); pinned "
                         "= every reader ring slot is page-locked once "
                         "(cudaHostRegister) and the copy is an async DMA. Needs "
                         "--prefetch > 0, dense. Recorded as "
                         "h2d_mode")
    ap.add_argument("--h2d-overlap", action="store_true",
                    help="--h2d pinned only: upload block i+1 on a dedicated copy "
                         "stream while block i is searched, if block i+1 has already "
                         "been read when search i is launched (a non-blocking poll); "
                         "otherwise it is uploaded on demand. The merge stays serial. "
                         "The GPU holds two blocks (refused up front if they do not "
                         "fit). h2d_mode = pinned-overlap")
    ap.add_argument("--d2h", choices=schema.D2H_MODES, default="pageable",
                    help="how the results (D, I: nq x k per block) come back to the "
                         "host for the merge (resident gpu / cuvs, dense): pageable = "
                         ".cpu() into a fresh pageable tensor (default); "
                         "pinned = into two reusable page-locked buffers, an async DMA "
                         "the call still waits for (the merge stays serial after it). "
                         "Recorded as d2h_mode")
    ap.add_argument("--d2h-overlap", action="store_true",
                    help="with --d2h pinned: copy each block's results back on a "
                         "separate stream and merge them while the NEXT block is "
                         "searched (one block in flight); the merge stays FAISS's "
                         "ResultHeap. d2h_mode records pinned-overlap")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--clock-json", default=None, metavar="PATH",
                    help="time every timed (non-warmup) pass with a bench LegClock too, "
                         "started / stopped at the pass's own t_start / t_end, and write the "
                         "clock records here as JSON (bench leg); off by default")
    ap.add_argument("--csv", required=True,
                    help="append rows here; one row per file per pass (warmup "
                         "included) always goes to the trace CSV next to it "
                         "(x.csv -> x.trace.csv)")
    ap.add_argument("--verify", action="store_true",
                    help="compare every pass to a one-shot FLOAT64 exact top-k computed "
                         "by the harness over the whole corpus (numpy / scipy, no engine's "
                         "code; lowest id on ties; filtered: with every file's masks), the "
                         "same reference for every engine; small corpora only: refused up "
                         "front when the run's estimated peak x VERIFY_SAFETY (1.25) would "
                         "not fit in MemAvailable")
    fl = ap.add_argument_group("filtered search (bench/data/filters)")
    fl.add_argument("--filter-config", default=None, metavar="NOVA_BF_YAML",
                    help="take the filter from a nova-bf brute-force config (with "
                         "--filter-search): its filter, rows and date_fields")
    fl.add_argument("--filter-search", default=None, metavar="NAME",
                    help="the search in --filter-config whose filter to use")
    fl.add_argument("--filter-mode", choices=["gather", "bitmap"], default=None,
                    help="gather = search only the passing rows, compacted by the "
                         "reader in the block's own buffer (static filters; FAISS or "
                         "cuVS); bitmap = the filter as a "
                         "prefilter (static or per-query; cuVS / masked)")
    fl.add_argument("--filter-skip", choices=["queries", "files", "none"], default=None,
                    help="bitmap mode's dead-work skipping: queries = drop each file's "
                         "dead queries (no allowed row) and skip a file with none live; "
                         "files = search every query, skip only a file where nothing "
                         "passes; none = search everything")
    fl.add_argument("--filter-compact", choices=["none", "union"], default=None,
                    help="per-query bitmap: union = shrink each file to the rows ANY query "
                         "wants (OR of the masks) before the prefilter search, in the "
                         "reader, as nova-bf compacts filtered-only runs; none = search "
                         "the whole block")
    fl.add_argument("--filter-threads", type=int, default=None,
                    help="DuckDB threads per reader (default: the usable CPUs, split "
                         "across the readers)")
    args = ap.parse_args(argv)
    if args.k < 1:
        ap.error("--k must be >= 1")
    if args.backend == "gpu" and args.k > schema.GPU_MAX_K:
        ap.error(f"--k {args.k} > {schema.GPU_MAX_K}: FAISS GPU k-select would abort the process")
    if (args.backend == "cuvs" and args.modality == "sparse"
            and args.k > cuvs_engine.SPARSE_CUVS_MAX_K):
        ap.error(f"--k {args.k} > {cuvs_engine.SPARSE_CUVS_MAX_K}: cuVS sparse merges index batches "
                 "with knn_merge_parts, which refuses k > 1024 (it would fail mid-run on "
                 "the first block over batch_size_index rows: "
                 f"{cuvs_engine.SPARSE_STOCK_BATCHES[0]} at the stock setting)")
    if args.cuvs_sparse_batches is not None and not (
            args.modality == "sparse" and args.backend == "cuvs"):
        ap.error("--cuvs-sparse-batches is a --modality sparse --backend cuvs option "
                 "(cuVS's sparse_search_params)")
    if args.backend == "cuvs":
        if args.cuvs_merge is None:
            args.cuvs_merge = "device"
    elif args.cuvs_merge is not None:
        ap.error("--cuvs-merge is a --backend cuvs option")
    if args.cuvs_sync is not None and args.backend != "cuvs":
        ap.error("--cuvs-sync is a --backend cuvs option")
    if args.cuvs_sync is not None and args.cuvs_merge != "device":
        ap.error("--cuvs-sync applies to --cuvs-merge device only (the host merge has no "
                 "per-file sync; it waits for each file's copy back)")
    if args.backend == "cuvs" and args.cuvs_merge == "device" and args.cuvs_sync is None:
        args.cuvs_sync = "per-file"
    if args.backend == "spare":
        if args.modality != "sparse":
            ap.error("--backend spare is --modality sparse only (SPARe is a sparse engine)")
        if args.spare_mode is None:
            ap.error("--backend spare needs --spare-mode dot or iterative")
        if args.metric != "ip":
            ap.error("--backend spare is inner product only (SPARe has no cosine)")
        if args.spare_max_mib is not None and args.spare_max_mib < 1:
            ap.error("--spare-max-mib must be >= 1")
        if args.spare_max_mib is not None and spare_engine.spare_released(args.spare_mode):
            ap.error(f"--spare-max-mib does not apply to --spare-mode {args.spare_mode}")
        if args.no_spare_events and args.spare_mode not in spare_engine.SPARE_STAGED_MODES:
            ap.error(f"--no-spare-events does not apply to --spare-mode {args.spare_mode} "
                     "(SPARe's own forward: no stage events are recorded)")
    elif args.no_spare_events or any(
            v is not None for v in (args.spare_mode, args.spare_max_mib)):
        ap.error("--spare-mode / --spare-max-mib / --no-spare-events are --backend spare "
                 "options")
    if args.backend == "pylate":
        if args.modality != "multivector":
            ap.error("--backend pylate is --modality multivector only (PyLate is a "
                     "late-interaction engine)")
        if args.pylate_mode is None:
            ap.error("--backend pylate needs --pylate-mode")
        if args.pylate_max_mib is not None and args.pylate_max_mib < 1:
            ap.error("--pylate-max-mib must be >= 1")
        if args.pylate_max_mib is None:
            args.pylate_max_mib = pylate_engine.PYLATE_DEFAULT_MAX_MIB
        if args.pylate_query_buckets is not None:
            if args.pylate_query_buckets < 1:
                ap.error("--pylate-query-buckets must be >= 1")
            if pylate_engine.pylate_released(args.pylate_mode):
                ap.error("--pylate-query-buckets is for --pylate-mode scores-batched (the "
                         "rerank modes take one query at a time)")
        if not pylate_engine.pylate_released(args.pylate_mode):
            if args.pylate_query_buckets is None:
                args.pylate_query_buckets = 1
            if args.pylate_rerank_results is not None:
                ap.error(f"--pylate-rerank-results does not apply to --pylate-mode "
                         f"{args.pylate_mode} (its results are device tensors already)")
        elif args.pylate_rerank_results is None:
            args.pylate_rerank_results = "device"
    elif any(v is not None for v in (args.pylate_mode, args.pylate_max_mib,
                                     args.pylate_rerank_results, args.pylate_query_buckets)):
        ap.error("--pylate-mode / --pylate-max-mib / --pylate-rerank-results / "
                 "--pylate-query-buckets are --backend pylate options")
    if args.backend == "spare" and args.spare_max_mib is None:
        args.spare_max_mib = spare_engine.SPARE_DEFAULT_MAX_MIB
    if args.backend in ("cuvs", "spare", "pylate") and args.temp_mem is not None:
        ap.error(f"--temp-mem is a FAISS option; --backend {args.backend} has no such knob")
    # where a stored-dtype block is cast to float32: on the device for the GPU
    # backends, on the host for the CPU ones (they need host float32 arrays)
    args.widen = "host" if args.backend in schema.HOST_BACKENDS else "device"
    if args.range_concurrency < 1:
        ap.error("--range-concurrency must be >= 1")
    if args.range_mib < 1:
        ap.error("--range-mib must be >= 1")
    if args.max_files < 0:
        ap.error("--max-files must be >= 0")
    if args.prefetch < 0 or args.workers < 1:
        ap.error("--prefetch >= 0 and --workers >= 1")
    if args.prefetch > MAX_PREFETCH:
        ap.error(f"--prefetch {args.prefetch} > {MAX_PREFETCH}: every in-flight read is a "
                 "record in the reader task pipe and a ring slot; more would not fit the "
                 "pipe")
    # counts and byte sizes: a negative one has no meaning (0 keeps its
    # documented meaning: default / off)
    for flag, v, lo in (("--nq", args.nq, 1), ("--warmup", args.warmup, 0),
                        ("--repeats", args.repeats, 0), ("--dim", args.dim, 1),
                        ("--temp-mem", args.temp_mem, 0),
                        ("--omp-threads", args.omp_threads, 0)):
        if v is not None and v < lo:
            ap.error(f"{flag} must be >= {lo}")
    if args.warmup + args.repeats < 1:
        ap.error("at least one pass is needed (--warmup + --repeats >= 1)")
    if args.warmup_files < 0:
        ap.error("--warmup-files must be >= 0")
    if args.warmup_files and not args.warmup:
        ap.error("--warmup-files needs --warmup >= 1 (it only shortens warmup passes)")
    if args.d2h_overlap and args.d2h != "pinned":
        ap.error("--d2h-overlap needs --d2h pinned (an async copy back needs "
                 "page-locked host buffers)")
    if args.d2h == "pinned" and args.backend in schema.HOST_BACKENDS:
        ap.error("--d2h pinned needs a GPU backend: the CPU backend has no D2H")
    if args.h2d_overlap and args.h2d != "pinned":
        ap.error("--h2d-overlap needs --h2d pinned (an async copy needs page-locked "
                 "host memory; from pageable memory torch's copy blocks the host)")
    if args.h2d == "pinned":
        # pinned memory that lives for the whole run exists only as the reader
        # processes' shared-memory ring
        if args.backend in schema.HOST_BACKENDS:
            ap.error("--h2d pinned needs a GPU backend: the CPU backend has no H2D")
        if args.prefetch == 0:
            ap.error("--h2d pinned needs --prefetch > 0: only the reader processes' "
                     "shared-memory ring slots live for the whole run and can be "
                     "page-locked once; the inline reader hands over a fresh pageable "
                     "array per file")
    if args.modality == "sparse":
        if args.backend == "gpu":
            ap.error("--modality sparse: FAISS has no sparse search; use --backend cuvs "
                     "(cuVS sparse brute force), spare (SPARe) or cpu (scipy)")
        if not args.query_file:
            ap.error("--modality sparse needs --query-file (sparse queries are not generated)")
        if args.dim:
            ap.error("--dim is dense only (a sparse corpus's width comes from its indices)")
    if args.sparse_project != "off":
        if args.modality != "sparse":
            ap.error("--sparse-project is a --modality sparse option")
        if args.metric != "ip":
            ap.error("--sparse-project is --metric ip only (cosine norms would have to come "
                     "from the unprojected values)")
    if args.modality == "multivector":
        if args.backend not in MV_BACKENDS:
            ap.error(f"--modality multivector: --backend {args.backend} has no multi-vector "
                     "search; use --backend pylate (PyLate) or cpu (numpy MaxSim)")
        if not args.query_file:
            ap.error("--modality multivector needs --query-file (multi-vector queries are "
                     "not generated)")
        if args.dim:
            ap.error("--dim is dense only (a multi-vector corpus's token width comes from "
                     "its data)")
    if args.gt_ids_column and not args.query_file:
        ap.error("--gt-ids-column needs --query-file (the ground truth is aligned with "
                 "its rows)")
    filtered = bool(args.filter_config)
    if not filtered and (args.filter_skip is not None or args.filter_compact is not None):
        ap.error("--filter-skip / --filter-compact need --filter-config")
    if args.filter_compact is None:
        args.filter_compact = "none"
    if args.filter_skip is None:
        args.filter_skip = "queries"
    if bool(args.filter_config) != bool(args.filter_search):
        ap.error("--filter-config and --filter-search go together")
    if args.backend == "masked" and args.modality != "dense":
        ap.error("--backend masked is dense only")
    if filtered:
        if not args.filter_mode:
            ap.error("--filter-config needs --filter-mode gather or bitmap")
        if args.modality != "dense":
            ap.error("--filter-config is dense only")
        # gather / union: the READER compacts each block in its own buffer (a
        # process reader's ring slot), so --h2d pinned / --h2d-overlap /
        # --d2h-overlap apply to them as to any block (their generic checks
        # above: GPU backend, reader processes)
        if args.backend not in ("gpu", "cpu", "cuvs", "masked"):
            ap.error(f"--filter-config: --backend {args.backend} is not a filtered-search backend "
                     "(gpu / cpu = FAISS gather, cuvs, masked)")
        if args.filter_mode == "bitmap" and args.backend in ("gpu", "cpu"):
            ap.error("--filter-mode bitmap: FAISS has no prefilter search on the GPU "
                     "(bfKnn takes no subset; GpuIndexFlat's single IDSelector runs only "
                     "through its cuVS backend); use --filter-mode gather or --backend cuvs")
        if args.filter_mode == "gather" and args.filter_compact != "none":
            ap.error("--filter-compact applies to --filter-mode bitmap (gather already "
                     "searches only the passing rows)")
        if args.filter_mode == "gather" and args.filter_skip != "queries":
            ap.error("--filter-skip applies to --filter-mode bitmap (gather searches only "
                     "the passing rows by construction)")
        if args.filter_threads is not None and args.filter_threads < 1:
            ap.error("--filter-threads must be >= 1")
        if args.gt_ids_column:
            ap.error("--gt-ids-column with a query row selection is not supported")
    elif args.filter_mode or args.filter_threads is not None:
        ap.error("--filter-mode / --filter-threads need --filter-config")
    if args.save_results:
        try:
            args.save_results = save_results_path(args.save_results, args.backend,
                                                  args.modality)
        except ValueError as e:
            ap.error(f"--save-results: {e}")
    return args


def save_results_path(path, backend, modality="dense"):
    """Resolve --save-results' PATH before any work.

    Fills in ``{backend}`` / ``{engine}``, appends ``.npz`` when missing (with
    a note: np.savez would do it silently), and checks the directory exists
    and is writable. Raises ValueError otherwise."""
    engine = {"sparse": SPARSE_BACKENDS, "multivector": MV_BACKENDS}.get(
        modality, BACKENDS)[backend].engine
    try:
        out = path.format(backend=backend, engine=engine)
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"{path!r}: only {{backend}} / {{engine}} placeholders are "
                         f"filled in ({type(e).__name__}: {e}); double a literal brace")
    if os.path.isdir(out):
        raise ValueError(f"{out} is a directory")
    if not out.endswith(".npz"):
        print(f"NOTE: --save-results {out} -> {out}.npz (np.savez writes .npz)",
              file=sys.stderr)
        out += ".npz"
        if os.path.isdir(out):
            raise ValueError(f"{out} is a directory")
    parent = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(parent):
        raise ValueError(f"directory {parent} does not exist")
    if not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"directory {parent} is not writable")
    return out


def save_results(path, D, I):
    """Save the last pass's merged top-k (D, I) to ``path``; True if written.

    Called after the pass's row is written or held, so a failed write is only
    a warning, never a lost row or a skipped --verify."""
    try:
        np.savez(path, D=D, I=I)
    except Exception as e:  # noqa: BLE001 - OSError, ...; Ctrl-C still propagates
        print(f"WARNING: --save-results {path} not written ({type(e).__name__}: {e}); "
              "the pass's rows are unaffected", file=sys.stderr)
        return False
    return True


EXIT_SIGNALS = ("SIGTERM", "SIGHUP")
# held off by signals_deferred; most severe first (the one re-raised when
# several arrived)
DEFERRED_SIGNALS = ("SIGTERM", "SIGHUP", "SIGINT")


@contextlib.contextmanager
def signals_deferred():
    """Run the body with SIGTERM / SIGHUP / SIGINT deferred.

    Afterwards the most severe signal that arrived (SIGTERM > SIGHUP > SIGINT)
    goes to the handler installed before; one that was SIG_IGN stays ignored.
    The signals are blocked in this thread, and their handlers are swapped
    for a recorder too, because a process-directed signal can land on another
    thread (readers, Arrow / OpenMP pools) and still run its Python handler
    here. Main thread only; elsewhere a no-op."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    sigs = [getattr(signal, n) for n in DEFERRED_SIGNALS if hasattr(signal, n)]
    got = []
    old = {sig: signal.signal(sig, lambda signum, frame: got.append(signum)) for sig in sigs}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
    try:
        yield
    finally:
        # unblocking runs any pending one's handler (still the recorder)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        for sig, h in old.items():
            signal.signal(sig, signal.SIG_DFL if h is None else h)
        live = [s for s in sigs if s in got and old[s] is not signal.SIG_IGN]
        if live:
            signal.raise_signal(live[0])  # sigs is in DEFERRED_SIGNALS' order


@contextlib.contextmanager
def exit_on_signals():
    """Turn SIGTERM / SIGHUP into ``SystemExit(128 + signum)`` in the main thread.

    So ``main``'s cleanup runs: readers stop, the shared-memory ring leaves
    /dev/shm, finished passes are in the CSV. A repeat while exiting is
    ignored. A signal already ignored at start (SIGHUP under ``nohup``) stays
    ignored, so the run and its readers (which inherit the ignore) keep
    going. Old handlers are restored on exit. Only SIGKILL skips cleanup."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    exiting = []

    def handler(signum, frame):
        if not exiting:
            exiting.append(signum)
            raise SystemExit(128 + signum)

    old = {}
    for name in EXIT_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
            old[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def filter_compact_spec(args):
    """The readers' compaction (ParquetSource filter_compact).

    gather: the static filter's passing rows; union: the rows any searched
    query wants; None: whole blocks (bitmap without union)."""
    from bench.data.filters.reader import COMPACT_MAX_FRAC, CompactSpec
    if args.filter_mode == "gather":
        return CompactSpec("gather")
    if args.filter_mode == "bitmap" and args.filter_compact == "union":
        return CompactSpec("union", args.filter_skip, COMPACT_MAX_FRAC)
    return None


def setup_filter(args, dtype, xq, n_files):
    """Load --filter-config's filter, its query rows, and the per-query filter
    values from the same rows as the query vectors (selection first, then --nq).

    Returns (FilterJob, xq, description, query-rows description)."""
    import copy

    from bench.data.filters import duckdb_filter as dfm
    from bench.data.filters.reader import FilterJob
    from bench.data.filters.search import load_filter_config, query_row_index

    fdict, cdates, qdates, rows, sspec = load_filter_config(args.filter_config,
                                                            args.filter_search)
    desc = f"{args.filter_config}:{args.filter_search}"
    if sspec is not None:
        vt = sspec.get("vector_type", "dense")
        if vt != "dense":
            raise SystemExit(f"--filter-search {args.filter_search!r} is a {vt} search; "
                             "the filtered baseline is dense only")
        for key, mine in (("metric", args.metric), ("k", args.k)):
            theirs = sspec.get(key)
            if key == "metric":   # nova-bf's "dot" is this harness's "ip"
                theirs = {"dot": "ip"}.get(theirs, theirs)
            if theirs is not None and str(theirs) != str(mine):
                print(f"WARNING: --filter-search {args.filter_search!r} has {key}={theirs} "
                      f"but this run uses --{key} {mine}", file=sys.stderr)
    spec = dfm.parse_filter(fdict, cdates)
    if spec.per_query and args.filter_mode == "gather":
        raise SystemExit("--filter-mode gather handles static filters only; this filter is "
                         "per-query (use --filter-mode bitmap)")
    if not spec.per_query and args.filter_compact != "none":
        raise SystemExit("--filter-compact union is for per-query filters (a static "
                         "filter's compaction is --filter-mode gather)")
    if spec.per_query and args.backend in ("gpu", "cpu"):
        raise SystemExit("--filter-config: FAISS has no per-query filtered search on the GPU; "
                         "per-query filters run on --backend cuvs (or masked)")
    qcols = spec.query_columns()
    qtable = None
    if qcols or rows:
        if not args.query_file:
            raise SystemExit("--filter-config: a per-query filter / query row selection needs "
                             "--query-file")
        a2 = copy.copy(args)
        a2.nq = 0  # every row: the selection comes first, then --nq
        xq = results_io.load_queries(a2, dtype)
        qfs, refs = list_parquet(args.query_file)
        need = sorted(set(qcols) | ({rows["column"]} if rows and rows.get("column") else set()))
        have = set(pq.read_schema(refs[0].path, filesystem=qfs).names)
        missing = [c for c in need if c not in have]
        if missing:
            raise SystemExit(f"--query-file {args.query_file}: no column(s) {missing} "
                             "(per-query filter values / the query row selector)")
        qt = pa.concat_tables([pq.read_table(r.path, filesystem=qfs, columns=need)
                               for r in refs])
        if qt.num_rows != len(xq):
            raise SystemExit(f"--query-file: {qt.num_rows} filter rows vs {len(xq)} vectors")
        idx = query_row_index(qt, rows)
        if idx is not None:
            if len(idx) == 0:
                raise SystemExit(f"--filter-config: the query row selection {rows} keeps no row")
            xq, qt = np.ascontiguousarray(xq[idx]), qt.take(pa.array(idx))
        if args.nq and args.nq < len(xq):
            # nova-bf searches EVERY selected row: say so when --nq cuts them
            print(f"WARNING: --nq {args.nq} keeps the first {args.nq} of the {len(xq)} "
                  "selected queries (nova-bf searches all of them; raise --nq)",
                  file=sys.stderr)
            xq, qt = xq[:args.nq], qt.slice(0, args.nq)
        qtable = qt.select(qcols) if qcols else None
    readers = min(args.prefetch, args.workers, n_files) if args.prefetch else 1
    threads = args.filter_threads or max(1, usable_cpu_count() // max(1, readers))
    job = FilterJob(fdict, qtable if spec.per_query else None, cdates, qdates, threads,
                    engine_kw={"prefilter_max_phrases": FILTER_PREFILTER_MAX_PHRASES,
                               "text_impl": "bitmap"})
    rows_desc = "" if not rows else f"{rows['column']}={','.join(map(str, rows['isin']))}"
    return job, xq, desc, rows_desc


def check_corpus_filter_fields(fs, files, job):
    """Refuse early if the first file lacks a filter field (not a reader traceback)."""
    names = set(pq.read_schema(files[0].path, filesystem=fs).names)
    missing = [f for f in job.fields if f not in names]
    if missing:
        raise SystemExit(f"--filter-config: corpus file {files[0].path} has no column(s) {missing}")


# where bench / nova_bf are imported from, checked against this file's tree
_CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORT_ORIGINS = None


def _origins_or_refuse(argv):
    """Refuse before any read unless bench is this tree's and nova_bf (if any) its hashed root.

    The refusal is also written to --clock-json, so the leg sees why."""
    global IMPORT_ORIGINS
    IMPORT_ORIGINS = provenance.check(_CB, need_novabf=False)
    if not IMPORT_ORIGINS["ok"]:
        av = list(sys.argv[1:] if argv is None else argv)
        out = next((x.split("=", 1)[1] for x in av if x.startswith("--clock-json=")), None)
        if out is None and "--clock-json" in av and av.index("--clock-json") + 1 < len(av):
            out = av[av.index("--clock-json") + 1]
        if out:
            import json
            with open(out, "w") as f:
                json.dump({"clocks": [], "import_origins": IMPORT_ORIGINS}, f, indent=1, sort_keys=True)
        provenance.refuse_unless_ok(IMPORT_ORIGINS, "harness_cli")


def main(argv=None):
    _origins_or_refuse(argv)
    with exit_on_signals():
        _main(argv)


def _main(argv=None):
    args = parse_args(argv)
    dtype = schema.DTYPES[DTYPE]
    host_dtype = dtype if args.widen == "host" else None  # None = stored dtype
    trace_csv = results_io.trace_path(args.csv)
    results_io.check_csv(args.csv)
    results_io.check_csv(trace_csv, schema.TRACE_FIELDS)
    if args.verify:
        results_io.check_pending(args.csv, trace_csv)
    # before the first parquet read (footer, queries); 0 = usable CPUs / Arrow's default
    pa_cpu, pa_io = configure_arrow_pools(0, 0)
    if args.omp_threads:
        # torch's lazy thread-pool init resets the OpenMP runtime it shares with
        # faiss, undoing an earlier count. Trigger it first, then set faiss's
        if "torch" in sys.modules:
            sys.modules["torch"].get_num_threads()
        faiss.omp_set_num_threads(args.omp_threads)
    omp_threads = faiss.omp_get_max_threads()

    fs, files = list_parquet(args.corpus, recursive=args.corpus_recursive)
    if args.max_files:
        files = files[:args.max_files]  # sorted by path: a deterministic prefix
    big = [r for r in files if r.size > MAX_FILE_BYTES]
    if big:
        raise SystemExit(f"{len(big)} files exceed {MAX_FILE_BYTES} B, e.g. {big[0].path} "
                         f"({big[0].size} B)")
    if files[0].size == 0:
        raise ValueError(f"{files[0].path}: empty file")
    footer = read_footer(fs, files[0].path, args.column)
    mode, share = choose_read_mode(args.read_mode, footer, files[0].size)
    native_import()  # rust_fetch importable, or a clear error now
    try:
        native_url(fs, files[0].path)  # local or plain s3:// only
    except ValueError as e:
        raise SystemExit(str(e)) from e
    # the --verify oracle reloads with pyarrow, so it checks the native decode
    # against an independent one
    oracle_mode = "arrow"
    is_sparse = args.modality == "sparse"
    is_mv = args.modality == "multivector"
    codec = sparse.CODEC if is_sparse else multivec.CODEC if is_mv else None
    if is_mv:
        if footer.mv_values is None:
            raise SystemExit(f"--modality multivector: {files[0].path}: column "
                             f"{args.column!r} is not a list<list<float>>")
        # float32 tokens, read once, untimed; d = their width
        xq, notes = multivec.load_queries(args.query_file, args.query_column, args.nq)
        for msg in notes:
            print(msg, file=sys.stderr)
        d, nq = xq.dim, xq.n_rows
    elif is_sparse:
        if footer.sparse_nnz is None:
            raise SystemExit(f"--modality sparse: {files[0].path}: column {args.column!r} is "
                             "not a struct<indices: list<int>, values: list<float>>")
        # canonical CSR (duplicates summed), read once, untimed; d = its width
        xq, notes = sparse.load_queries(args.query_file, args.query_column, args.nq)
        for msg in notes:
            print(msg, file=sys.stderr)
        if args.sparse_project == "query-vocab":
            # nova-bf's own projection: vocabulary from the queries, queries
            # renumbered into it, and a codec that projects every corpus file
            # the same way in the reader. The unprojected queries stay for
            # the --verify oracle.
            xq_raw = xq
            xq, vocab = sparse.project_queries(xq_raw)
            codec = sparse.ProjectingCodec(vocab)
            print(f"NOTE: --sparse-project query-vocab: {len(vocab)} query columns (of "
                  f"width {xq_raw.n_cols}); corpus columns outside them are dropped",
                  file=sys.stderr)
        d, nq = xq.n_cols, xq.n_rows
    else:
        xq = results_io.load_queries(args, dtype)
        d = results_io.corpus_dim(args, xq, footer, fs, files[0], mode)
        if xq is None:
            xq = results_io.random_queries(args, d, dtype)
        nq = xq.shape[0]
    fjob = fdesc = None
    # filter config columns: n/a without --filter-config (measurement columns stay blank)
    fbase = dict.fromkeys(("filter", "filter_mode", "filter_kind", "filter_skip",
                           "filter_compact", "filter_threads", "query_rows",
                           "filter_text_impl", "duckdb_version"), "n/a")
    if args.filter_config:
        fjob, xq, fdesc, frows = setup_filter(args, dtype, xq, len(files))
        nq = xq.shape[0]
        check_corpus_filter_fields(fs, files, fjob)
        fbase = {"filter": fdesc, "filter_mode": args.filter_mode,
                 "filter_kind": "per_query" if fjob.spec().per_query else "static",
                 "filter_skip": args.filter_skip if args.filter_mode == "bitmap" else "n/a",
                 "filter_compact": args.filter_compact if args.filter_mode == "bitmap"
                                   else "n/a",
                 "filter_threads": fjob.threads, "query_rows": frows,
                 # match_text implementation (n/a: the filter has no text leaf)
                 "filter_text_impl": fjob.engine_kw.get("text_impl", "bitmap")
                                     if any(c.text for c in fjob.spec().conds()) else "n/a",
                 "duckdb_version": provenance._duckdb_version()}
    # --gt-ids-column: read once, untimed, before any pass (fails early)
    gt = results_io.load_gt_ids(args, nq) if args.gt_ids_column else None
    if gt is not None and args.max_files:
        print(f"NOTE: --gt-ids-column with --max-files {args.max_files}: the ground "
              "truth is over the whole corpus, so recall over a prefix reads low",
              file=sys.stderr)
    footers = None
    verify_footer_s = 0.0   # --clock-json: footer reads before the passes
    if args.verify:
        # every footer (one small read each), so the guard sees the whole
        # corpus before any pass; the source reuses them
        t_foot = now()
        footers = {r.path: footer if i == 0 else read_footer(fs, r.path, args.column)
                   for i, r in enumerate(files)}
        verify_footer_s = now() - t_foot
        if is_sparse:
            guards.sparse_verify_memory_guard(args, files, footers, xq, host_dtype)
        elif is_mv:
            guards.mv_verify_memory_guard(args, files, footers, xq, host_dtype)
        else:
            guards.verify_memory_guard(args, files, footers, d, nq, host_dtype)

    backend = make_backend(args, dtype)
    if args.backend == "cuvs":
        # cuVS's own recipe (device, the default) or the per-file host merge
        backend.configure_merge(args.cuvs_merge)
        cuvs_device = backend.device_merge
        if cuvs_device:
            backend.configure_sync(args.cuvs_sync)
    else:
        cuvs_device = False
    # a GPU run on the paper stack (paper_run.sh), or none: before any pass
    provenance.refuse_unpinned_stack(args, backend)
    if args.h2d != "pageable" or args.h2d_overlap:
        backend.configure_h2d(args.h2d, args.h2d_overlap)  # before setup: the streams
    if args.d2h != "pageable" and not cuvs_device:
        # cuVS's recipe copies back once, at the end of the pass: there is no per-block D2H
        # for --d2h pinned / --d2h-overlap to change (a no-op, recorded as d2h_mode
        # final-once); FAISS / SPARe / PyLate keep them
        backend.configure_d2h(args.d2h, overlap=args.d2h_overlap)
    if cuvs_device and cuvs_engine.cuvs_pool_spec(args) and getattr(backend, "_cuda", False):
        # as cuVS's tool's main() does: CuPy's allocations from the same RMM pool
        # (generate_groundtruth/__main__.py:239-244)
        from rmm.allocators.cupy import rmm_cupy_allocator
        backend._xp.cuda.set_allocator(rmm_cupy_allocator)
    setup_s = backend.setup(xq)
    # the engine's CUDA libraries are mapped by now: one copy of each, or no run
    provenance.refuse_duplicate_cuda_libs(args, backend)
    used_cuda = provenance.backend_uses_cuda(backend)

    preload_s = 0.0   # (no preload: the corpus is streamed)
    source = ParquetSource(args.corpus, args.column, args.prefetch,
                           args.workers, host_dtype, fs=fs, files=files,
                           read_mode=mode, range_concurrency=args.range_concurrency,
                           range_scope=args.range_scope,
                           range_bytes=args.range_mib << 20, codec=codec,
                           pin=bench.systems.streaming.host_pinner(backend) if args.h2d == "pinned" else None,
                           # the consumer also holds: the block uploading
                           # ahead (--h2d-overlap) and the block whose copy
                           # back (or, --cuvs-sync none, merge) is in flight
                           extra_slots=guards.overlap_extra_slots(args),
                           filter_job=fjob,
                           # gather / union: compacted by the readers
                           filter_compact=filter_compact_spec(args) if fjob else None)
    range_conc = source.range_concurrency
    corpus_info = {}
    if footers:
        source.footers.update(footers)

    # cosine host scratch (CPU backend): pre-sized and pre-faulted to the
    # largest block when that is known before the passes (every footer under
    # --verify), in setup_s; otherwise the first block grows it (see NormScratch)
    if is_sparse or is_mv:
        largest = 0  # no host normalise scratch for sparse / multi-vector
    else:
        largest = max((f.num_rows for f in footers.values()), default=0) * d if footers else 0
    setup_s += backend.reserve(largest)

    # --verify: rows are held (and mirrored to the pending sidecars) until the
    # check runs, then written ok / failed / error, or not_run if the run dies
    # first. Without --verify each pass is written as it ends, marked not_run
    held, results = [], []
    try:
        # start the readers before any pass; inside the try so a failed start is cleaned up
        source.start()
        if args.h2d_overlap:
            guards.overlap_gpu_memory_guard(args, backend, source, nq, dtype)
        if cuvs_device:
            # the id dtype of cuVS's recipe (utils.neighbor_index_accumulator_dtype): from the
            # corpus rows, read from the footers before any pass
            foot_all = source.read_footers()
            backend.n_base = sum(foot_all[r.path].num_rows for r in source.files)
            guards.cuvs_device_merge_guard(args, backend, source, nq, dtype)
            # compile CuPy's merge kernels untimed, before the first pass
            setup_s += backend.warm_merge()
        reader_pools = getattr(source, "reader_pa_threads_effective", (None, None))
        prov = provenance.provenance(args, backend)
        base = {
            "regime": args.regime, "backend": backend.name, "engine": backend.engine,
            "engine_version": backend.engine_version,
            "queries_mode": "n/a" if args.backend in schema.HOST_BACKENDS else "resident",
            "nq": nq, "k": args.k, "metric": args.metric, "modality": args.modality,
            "d": d, "n_files": len(files),
            "max_files": args.max_files,
            "prefetch": args.prefetch,
            "workers": args.workers if args.prefetch else 0,
            # reader count actually used: min(prefetch, workers, files); 0 when
            # nothing reads ahead (prefetch 0)
            "readers": source.readers if args.prefetch else 0,
            # process (prefetch > 0) or inline (prefetch 0)
            "reader": source.reader,
            # split for reader processes; n/a for the inline reader
            "reader_pool": source.reader_pool,
            # Arrow pools of the reader processes, as they report them, in
            # pool_split_str's <readers>x<threads> form ("1x3+2x2")
            "reader_pa_cpu_threads": results_io._blank(pool_split_str(reader_pools[0])),
            "reader_pa_io_threads": results_io._blank(pool_split_str(reader_pools[1])),
            "vectors_mem_limit": 0,
            "temp_mem": "n/a" if args.backend == "cuvs" or is_sparse or is_mv else
                        "default" if args.temp_mem is None else args.temp_mem,
            # cuVS: device:file (its own recipe, each file as handed, merged on the GPU) or
            # host (per_file_host); n/a for the other engines
            "cuvs_merge": backend.merge_str() if args.backend == "cuvs" else "n/a",
            # cuVS device merge: per-file (calc_truth's resources.sync() per file) or none
            # (no per-file sync); n/a for the host merge and the other engines
            "cuvs_sync": backend.sync_str() if args.backend == "cuvs" else "n/a",
            # cuVS: --cuvs-rmm-pool's pool, or RMM's default device resource (off)
            "rmm_pool": ("n/a" if args.backend != "cuvs" else
                         f"pool:{cuvs_engine.cuvs_pool_spec(args)}" if cuvs_engine.cuvs_pool_spec(args) else "default"),
            # sparse cuVS: its EFFECTIVE batch_size_index x batch_size_query
            # (stock unless --cuvs-sparse-batches; `file` recorded as the word);
            # SPARe: its per-call scratch budget and A/B knob, or per-query
            "sparse_batches": (backend.batches_str()
                               if is_sparse and args.backend == "cuvs" else
                               spare_engine.spare_batches_str(args) if args.backend == "spare" else "n/a"),
            "spare_mode": args.spare_mode if args.backend == "spare" else "n/a",
            "spare_events": spare_engine.spare_events_str(args),
            # --sparse-project: off / query-vocab for sparse rows, n/a otherwise
            "sparse_project": args.sparse_project if is_sparse else "n/a",
            # PyLate: its mode, and scores-batched's per-block scratch budget
            "pylate_mode": args.pylate_mode if args.backend == "pylate" else "n/a",
            "pylate_batches": (f"max_bytes={args.pylate_max_mib}MiB" +
                               (",per-query" if pylate_engine.pylate_released(args.pylate_mode) else
                                f",query_buckets={args.pylate_query_buckets}"))
                              if args.backend == "pylate" else "n/a",
            # the rerank modes' --pylate-rerank-results (device / host); n/a otherwise
            "pylate_rerank_results": (args.pylate_rerank_results
                                      if args.backend == "pylate" and
                                      pylate_engine.pylate_released(args.pylate_mode) else "n/a"),
            # THIS process's Arrow pools (footer, queries, --verify; and the
            # reads themselves for the inline reader); reader processes size
            # their own, reader_pa_* above
            "omp_threads": omp_threads, "pa_cpu_threads": pa_cpu,
            "pa_io_threads": pa_io, "read_mode": args.read_mode,
            "read_mode_effective": mode, "column_share": round(share, 4),
            "range_concurrency": range_conc, "range_scope": args.range_scope,
            "range_mib": args.range_mib,
            # a file the native reader refuses fails the run
            "native_fallback": "refuse",
            # f16 / f32 / f64 from the float width; any other type as Arrow names it
            "stored_dtype": (f"f{footer.sparse_value_bits}" if is_sparse else
                             f"f{footer.mv_value_bits}" if is_mv else
                             f"f{footer.value_bits}" if footer.value_bits else
                             footer.value_type),
            "dtype": DTYPE, "widen_where": args.widen,
            # pageable / pinned / pinned-overlap for a GPU backend; n/a
            # where the harness does no H2D (CPU backend)
            "h2d_mode": h2d_mode(args),
            # pageable / pinned results copy back (n/a: no D2H of the harness's own);
            # cuVS device merge: final-once (one copy back at the end, whatever --d2h says)
            "d2h_mode": ("n/a" if h2d_mode(args) == "n/a" else
                         "final-once" if cuvs_device else
                         "pinned-overlap" if args.d2h_overlap else args.d2h),
            "warmup_files": args.warmup_files,
            # n/a: no CUDA events on this path at all (CPU backend)
            "gpu_events": "n/a" if args.backend in schema.HOST_BACKENDS
                          else "off" if args.no_gpu_events else "on",
            "setup_s": setup_s, "preload_s": preload_s,
            # (no preload: blank)
            "preload_rss_bytes": "",
            "corpus_fs": fs.type_name, "corpus": args.corpus,
            "host": socket.gethostname(), **prov, **fbase,
            # which stack / code / host (stack_fields)
            **provenance.stack_fields(args, backend, prov["gpu_uuid"] or prov["gpu_index"]),
        }
        filt = None
        if fjob is not None:
            from bench.data.filters.search import FilterAdapter
            filt = FilterAdapter(args.filter_mode, nq, args.k,
                                 bool(getattr(backend, "returns_distance", False)),
                                 skip=args.filter_skip,
                                 kind="per_query" if fjob.spec().per_query else "static",
                                 compact=args.filter_compact)
        D = I = None
        passes = [(1, i) for i in range(args.warmup)] + [(0, i) for i in range(args.repeats)]
        clock_recs = []   # --clock-json
        for warm, rep in passes:
            # --warmup-files N: a warmup pass over the first N files only; its
            # results cover a prefix, so it gets no gt_recall and no --verify
            short = args.warmup_files if warm and 0 < args.warmup_files < len(files) else 0
            clock = None
            if args.clock_json and not warm:   # the timed passes only, never a warmup
                from bench.timing import LegClock
                clock = LegClock()
            D, I, info = loop.run_pass(source, backend, nq, args.k, MAX_FILE_BYTES, d,
                                  n_files=short, filt=filt, clock=clock)
            if clock is not None:   # after the pass, outside its timed span
                import json
                clock_recs.append(dict(clock.record(), repeat=rep, total_s=info["total_s"]))
                # the untimed footer pre-read (source.start() plus --verify's), kept
                # as its own column, never in the timed seconds (nova-bf reads its
                # schemas inside its scan)
                foot = getattr(source, "footer_read_s", None)
                with open(args.clock_json, "w") as f:
                    json.dump({"clocks": clock_recs, "import_origins": IMPORT_ORIGINS,
                               "footer_read_s": None if foot is None and not args.verify
                               else (foot or 0.0) + verify_footer_s},
                              f, indent=1, sort_keys=True)
            trace = info.pop("trace")
            if args.sparse_project != "off" and info["project_s"] == "":
                info["project_s"] = 0.0
            # after the pass, outside its timed span
            recall = results_io.gt_recall(I, gt) if gt is not None and not short else ""
            # a library the pass loaded lazily: still one copy, or no row
            libs = cudalibs.mapped()
            provenance.refuse_duplicate_cuda_libs(args, backend, libs)
            row = dict(base, warmup=warm, repeat=rep, **dict(info, **corpus_info),
                       gt_recall=recall, verified="not_run",
                       # this process's RSS high-water mark so far (VmHWM; reader
                       # processes not included): the passes up to this one
                       rss_hwm_bytes=results_io._blank(rss_bytes("VmHWM")),
                       # the CUDA libraries this pass actually ran on (mapped by now)
                       **provenance.cuda_libs_fields(used_cuda, libs),
                       # the OpenMP runtimes / BLAS mapped by now (two runtimes = two pools)
                       **provenance.thread_libs_fields())
            del row["max_block_bytes"]  # = rows_per_file * d * itemsize; not a column
            if short:
                row["n_files"] = short  # files actually read
            if args.verify and not short:
                held.append((row, (warm, rep, info["t_start_unix"], trace)))
                results.append((D, I))
                results_io.append_pending(args.csv, trace_csv, row, warm, rep, info["t_start_unix"],
                               trace)
            else:
                # written per pass, signals deferred so a pass lands in both
                # the CSV and the trace or in neither
                with signals_deferred():
                    results_io.append_csv(args.csv, [row])
                    results_io.append_csv(trace_csv, loop.trace_rows(row, warm, rep, info["t_start_unix"],
                                                     trace), schema.TRACE_FIELDS)
            if args.save_results and (warm, rep) == passes[-1]:
                # the last pass's final merged top-k, for cross-engine checks:
                # after its row is in the CSV (or held), a failure only warns
                if warm and args.warmup_files:
                    print(f"WARNING: --save-results: the last pass is a --warmup-files "
                          f"warmup ({args.warmup_files} files): the saved top-k is over that "
                          "prefix of the corpus only (add --repeats >= 1)", file=sys.stderr)
                save_results(args.save_results, D, I)
            frac = info["other_s"] / info["total_s"] if info["total_s"] else 0.0
            print(f"[{args.regime}{' warmup' if warm else ''} {rep}] "
                  f"total {info['total_s']:.3f}s  stall {info['source_stall_s']:.3f}  "
                  f"{backend.engine} {info['engine_call_s']:.3f}  merge {info['merge_s']:.4f}  "
                  f"other {info['other_s']:.4f} ({frac:.1%})  "
                  f"[worker fetch {info['fetch_s']:.3f} decode {info['decode_s']:.3f}"
                  f" (read_table {info['read_table_s']:.3f} convert {info['convert_s']:.3f})"
                  f" publish {fmt(info['publish_s'])} handoff {fmt(info['handoff_s'])}"
                  f" | widen {fmt(info['widen_s'])} ({args.widen}) h2d {fmt(info['h2d_s'])}"
                  + (f" (exposed {fmt(info['h2d_exposed_s'])}, ahead x{info['h2d_overlapped']})"
                     if args.h2d_overlap else "")
                  + (f" normalize {fmt(info['normalize_s'])}" if args.metric == "cosine"
                     else "")
                  + f" search {fmt(info['search_s'])} (build {info['build_s']:.4f})"
                  f" d2h {fmt(info['d2h_s'])}"
                  f" sync_wait {fmt(info['sync_wait_s'], 4)} x{info['sync_waits']}]"
                  + (f"  gt_recall@{args.k} {recall:.4f}"
                     if gt is not None and recall != "" else ""),
                  flush=True)
            if not warm and frac > schema.OTHER_WARN_FRAC:
                print(f"WARNING: other_s is {frac:.1%} of total (> {schema.OTHER_WARN_FRAC:.0%}); "
                      "time is escaping the buckets", file=sys.stderr)

        if args.verify:
            # error unless the check finishes: MemoryError, an I/O error, a
            # signal (SystemExit) or Ctrl-C inside it never judged the results
            statuses = ["error"] * len(held)
            names = [f"{'warmup' if w else 'repeat'} {r}" for _, (w, r, _, _) in held]
            try:
                if D is None:
                    raise SystemExit("--verify: no pass ran")
                if not held:
                    # e.g. --repeats 0 with --warmup-files: the only pass read a
                    # prefix of the corpus, so nothing can be checked against it
                    raise SystemExit("--verify: no full pass ran (a --warmup-files warmup "
                                     "reads only a prefix; add --repeats >= 1)")
                # stop the readers and drop the ring before the oracle reload,
                # so the two never share memory (the guard counts them apart)
                source.close()
                # (a filtered run builds its own oracle from a fresh reader pass)
                # --sparse-project: the oracle reloads UNPROJECTED (plain codec,
                # the unprojected queries), so it checks the projection too
                ref_q = xq
                ref_codec = codec
                if args.sparse_project != "off":
                    ref_q, ref_codec = xq_raw, sparse.CODEC
                ref_src = None if fjob is not None else \
                    InMemorySource.load(args.corpus, args.column, None, oracle_mode,
                                        fs=fs, files=files, codec=ref_codec)
                if is_mv:
                    # float64 MaxSim over every block; xq_ref / xb rescore
                    # (query, document) pairs (unit tokens for cosine)
                    D_ref, I_ref, xq_ref, xb = multivec.oneshot_reference(
                        xq, ref_src.blocks, args.k + 1, args.metric)
                elif is_sparse:
                    # float64 scipy over every block's CSR; xq_ref / xb are
                    # row accessors (unit rows for cosine) for the rescoring
                    D_ref, I_ref, xq_ref, xb = sparse.oneshot_reference(
                        ref_q, ref_src.blocks, args.k + 1, args.metric)
                elif fjob is not None:
                    # filtered: exact top-k with every file's masks, re-evaluated
                    # by a fresh pyarrow reader pass; every returned id must
                    # also pass the filter
                    import dataclasses

                    from bench.verify.oracle import filtered_oneshot_reference
                    with ParquetSource(args.corpus, args.column, 0, 1, None, fs=fs,
                                       files=files, read_mode=oracle_mode,
                                       filter_job=dataclasses.replace(fjob, selfcheck=False)
                                       ).start() as osrc:
                        n_tot = sum(osrc.footers[r.path].num_rows for r in files)
                        # float64 (unit for cosine) queries and a row
                        # accessor over the kept blocks, for the rescoring
                        D_ref, I_ref, xq_ref, xb, f_allowed = filtered_oneshot_reference(
                            xq, iter(osrc), args.k + 1, args.metric, n_rows=n_tot)
                else:
                    # float64 numpy over every block (no FAISS); xq_ref / xb:
                    # the float64 (unit, for cosine) queries and a row accessor
                    # over the blocks, for the tie-zone rescoring
                    D_ref, I_ref, xq_ref, xb = verify.oneshot_reference(
                        xq, ref_src.blocks, args.k + 1, args.metric)
                n_ref = len(xb)
                problems = verify.verify_passes(results, names, D_ref, I_ref, VERIFY_ATOL,
                                         xq_ref, xb)
                if fjob is not None:
                    for i, (_, Ip) in enumerate(results):
                        bad = f_allowed.disallowed(Ip)
                        if bad:
                            problems[i] = list(problems[i]) + [
                                f"{len(bad)} returned ids are not allowed by the filter, "
                                f"e.g. (query, id) {bad[:3]}"]
                # the passes are judged: their rows are written now
                statuses = ["failed" if p else "ok" for p in problems]
                done, held = held, []
                write_held(args.csv, trace_csv, done, statuses)
                del xb
            finally:
                if held:  # the check died before judging: verified=error
                    done, held = held, []  # never written twice (the not_run path below)
                    write_held(args.csv, trace_csv, done, statuses)
            bad = [(n, p) for n, p in zip(names, problems) if p]
            oracle = ("float64 scipy sparse" if is_sparse else
                      "float64 numpy MaxSim" if is_mv else
                      "float64 numpy masked (bench/verify/oracle.py)" if fjob is not None
                      else "float64 numpy")
            if not bad:
                print(f"verify OK: {nq} queries x k={args.k} ({backend.engine}) match one-shot "
                      f"{oracle} over {n_ref} rows in all {len(names)} passes "
                      f"({args.metric}, atol {VERIFY_ATOL:g})")
            if bad:
                raise SystemExit("VERIFY FAILED: " + "; ".join(bad[-1][1]) +
                                 f" ({bad[-1][0]}; failed passes: "
                                 f"{', '.join(n for n, _ in bad)}; their rows written "
                                 f"with verified=failed to {args.csv})")
        if gt is not None and recall != "":  # "" after a short --warmup-files pass
            print(f"gt_recall@{args.k} ({args.metric}, last pass, {nq} queries vs "
                  f"{args.gt_ids_column}): {recall:.4f}")

    finally:
        try:
            if held:  # --verify, but the run died before the check
                done, held = held, []
                write_held(args.csv, trace_csv, done, "not_run")
        finally:
            # stop the reader processes now (also done at exit) and remove
            # the shared-memory ring
            close = getattr(source, "close", None)
            if close is not None:
                close()

if __name__ == "__main__":
    main()
