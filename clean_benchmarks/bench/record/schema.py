"""The harness's shared constants: the CSV and trace columns (CSV_FIELDS, TRACE_FIELDS, ...),
the metrics and dtypes, the CUDA-event fields, and the /proc paths read for network, memory
and driver figures.
"""

import faiss  # after cudalibs.torch_first(): harness_cli.py and tests/conftest.py call it first
import numpy as np



GPU_MAX_K = 2048  # GPU_MAX_SELECTION_K (faiss/gpu/utils/DeviceDefs.cuh); FAISS_ASSERT aborts above it
OTHER_WARN_FRAC = 0.05
IP = faiss.METRIC_INNER_PRODUCT
# --metric: ip = raw inner product, cosine = cos(q, x); both largest first. This is the score
# every engine's results are merged on.
METRICS = ("ip", "cosine")
DTYPES = {"f32": np.float32}
# --h2d: how a block reaches the device (pageable or pinned host memory)
H2D_MODES = ("pageable", "pinned")
D2H_MODES = ("pageable", "pinned")
# FAISS StandardGpuResources caps temp memory at 1.5 GiB on GPUs > 8 GiB (kMaxTempMem,
# getDefaultTempMemForGPU in faiss/gpu/StandardGpuResources.cpp), which is also its default.
# A larger setTempMemory request is clamped to it.
FAISS_MAX_TEMP_MEM = 1536 << 20
NET_DEV = "/proc/net/dev"  # Linux; elsewhere net_rx_* are left empty

CSV_FIELDS = [
    "regime", "backend", "engine", "engine_version", "queries_mode", "nq", "k", "metric",
    "modality", "d", "n_files", "max_files",
    "rows", "rows_per_file", "nnz", "nnz_kept", "tokens", "prefetch", "workers", "readers", "reader", "reader_pool",
    "reader_pa_cpu_threads",
    "reader_pa_io_threads", "vectors_mem_limit", "temp_mem",
    "rmm_pool", "cuvs_merge", "cuvs_sync", "sparse_batches", "spare_mode", "spare_events", "sparse_project", "pylate_mode",
    "pylate_batches", "pylate_rerank_results", "omp_threads", "pa_cpu_threads", "pa_io_threads",
    "read_mode",
    "read_mode_effective", "column_share", "range_concurrency", "range_scope", "range_mib",
    "stored_dtype", "dtype", "widen_where", "gpu_events", "warmup", "repeat",
    "h2d_mode", "d2h_mode", "warmup_files",
    "total_s",
    "source_stall_s", "engine_call_s", "merge_s", "other_s", "h2d_s", "h2d_exposed_s",
    "h2d_overlapped",
    "widen_s", "normalize_s", "search_s", "lib_s", "search_harness_s", "ids_s", "d2h_s",
    "build_s", "engine_host_s", "device_merge_s",
    "spare_plan_s", "spare_score_s", "spare_topk_s", "engine_gap_s", "gpu_idle_s",
    "sync_wait_s", "sync_waits", "pylate_calls", "pylate_calls_max_block", "gpu_alloc_peak_bytes",
    "fetch_s", "decode_s",
    "read_table_s", "convert_s", "project_s", "native_read_s", "native_fallbacks",
    "native_fallback",
    "publish_s", "handoff_s", "bytes", "net_rx_bytes",
    "net_rx_MBps",
    "row_groups", "row_groups_per_file", "setup_s", "preload_s", "preload_rss_bytes",
    "rss_hwm_bytes", "cpu_user_s", "cpu_sys_s", "worker_cpu_user_s", "worker_cpu_sys_s", "t_start_unix", "t_end_unix", "corpus_fs", "corpus", "column",
    "query_file", "query_column", "seed", "host", "gpu_name", "gpu_uuid", "gpu_index",
    "driver", "torch_version", "torch_cuda", "pyarrow_version", "numpy_version",
    "merge_faiss_version",
    # the CUDA libraries the pass actually ran on (cuda_libs_fields)
    "cublas_version", "cuda_libraries", "cuda_lib_versions", "cuda_lib_duplicates",
    # which stack / code / host (provenance.stack_fields): the launcher's PAPER_* values, the
    # EC2 instance (blank off EC2), the FAISS and cuVS builds, the GPU's state at the start,
    # the OpenMP / BLAS libraries mapped (per row) and the thread / CUDA environment
    "paper_stack_sha", "paper_code_sha", "paper_run_dir",
    "instance_id", "instance_type", "availability_zone",
    "faiss_build", "faiss_compile_options",
    "cuvs_version", "libcuvs", "libcuvs_sha256", "cuvs_patches",
    "gpu_state", "thread_libs", "openmp_runtimes", "thread_env", "cuda_env",
    # --filter-config (bench/data/filters/): the filter and how it was applied. Reader side:
    # filter_*_s and payload_read_s, summed over files in reader wall time (filter_compact_s
    # is gather / union's compaction in the reader). Consumer side: filter_handoff_s, inside
    # engine_call. Counts are summed over files.
    "filter", "filter_mode", "filter_kind", "filter_skip", "filter_compact", "filter_threads",
    "query_rows", "filter_text_impl", "duckdb_version",
    "filter_s", "filter_eval_s", "filter_dedup_s", "filter_compact_s", "filter_publish_s",
    "payload_read_s",
    # filter_eval_s broken down by DuckFilter stage (filters.reader.FILTER_STAGES)
    "filter_prep_s", "filter_sql_s", "filter_text_s", "filter_combine_s",
    # filter_sql_s / filter_text_s split into time inside DuckDB (statement + Arrow export)
    # and this harness's code around it
    "filter_sql_query_s", "filter_sql_post_s", "filter_text_query_s", "filter_text_post_s",
    # the match_text tokenisation used (lean / full / mixed; blank = no text leaf), the
    # readers' DuckFilter start-up (outside total_s), masks sent outside the ring slot, and cuVS's
    # mask upload + per-query expansion (inside search_s)
    "filter_text_tokens", "filter_setup_s", "filter_overflow_bytes", "filter_bits_s",
    "filter_handoff_s", "filter_blocks", "filter_rows_kept", "filter_allowed_pairs",
    "filter_live_queries",
    "filter_skipped_files", "filter_distinct_masks", "filter_searched_queries",
    "filter_compact_rows", "filter_compact_floor_files", "filter_compact_floor_rows",
    "filter_compact_whole_files", "filter_compact_max_frac", "filter_mask_slots",
    "filter_mask_overflows", "cuvs_filter_small_k_calls",
    "gt_recall", "verified",
]
# host-only backends: no device, no H2D / D2H / events of their own
HOST_BACKENDS = ("cpu", "masked")
# --verify outcome of the pass a row belongs to. A --verify run's rows are held back until the
# check has run, so a failed check never leaves unmarked rows; until then they sit in the
# sidecars (results_io.pending_paths) marked PENDING. "error" means the check itself raised
# (MemoryError, I/O, a signal, Ctrl-C), so the results were never judged.
VERIFIED = ("ok", "failed", "error", "not_run")
PENDING = "pending"
MEMINFO = "/proc/meminfo"  # Linux; elsewhere the --verify memory guard is skipped
NVIDIA_PROC = "/proc/driver/nvidia/version"  # Linux driver version, no subprocess
# written empty (not 0) with --no-gpu-events: not measured is not zero
EVENT_FIELDS = ("h2d_s", "widen_s", "search_s", "d2h_s", "engine_gap_s", "sync_wait_s",
                "sync_waits", "h2d_exposed_s", "h2d_overlapped", "gpu_idle_s", "ids_s",
                "lib_s", "search_harness_s")
# The trace (always written, to results_io.trace_path(--csv)): one row per file per pass.
# Consumer-side times (stall / engine_call / build / merge / t_got / t_done) and worker-side
# times (w_start / w_end / fetch / read_table / convert / widen_host) share the perf_counter
# clock (CLOCK_MONOTONIC, the same in the reader processes), as seconds since the pass began.
# handoff_s (the parent viewing the reader's ring slot) is inside stall_s. publish_s (the copy
# into the slot, w_end .. w_publish_end_s) and w_cpu_* belong to the reader process; all are
# empty for the inline reader (no such step).
# Each trace row carries its run's config: the columns plot.CONFIG separates bars by (so
# runs can be told apart, filtered with --where and panels titled), minus the run's total
# "rows", a name the trace uses for the file's own rows. A test keeps this equal to
# plot.CONFIG minus "rows".
TRACE_CONFIG = ["regime", "backend", "engine", "engine_version", "queries_mode", "nq", "k",
                "metric", "d", "n_files", "max_files", "rows_per_file", "prefetch", "workers",
                "readers", "reader", "reader_pool", "reader_pa_cpu_threads",
                "reader_pa_io_threads", "vectors_mem_limit", "temp_mem", "rmm_pool", "omp_threads", "pa_cpu_threads",
                "pa_io_threads", "read_mode", "read_mode_effective", "range_concurrency",
                "range_scope", "range_mib", "row_groups_per_file", "stored_dtype", "dtype",
                "widen_where", "gpu_events", "h2d_mode", "d2h_mode", "corpus", "column",
                "query_file",
                "query_column", "seed", "host",
                # --filter-config: filtered and unfiltered runs (and different filters) never
                # share a bar (plot.CONFIG)
                "filter", "filter_mode", "filter_kind", "filter_skip", "filter_compact",
                "filter_threads", "query_rows", "filter_text_impl", "duckdb_version",
                "torch_version", "torch_cuda", "pyarrow_version",
                "numpy_version", "merge_faiss_version", "gpu_name", "driver", "modality",
                "sparse_batches", "spare_mode", "spare_events", "sparse_project", "pylate_mode",
                "pylate_batches", "pylate_rerank_results", "cuvs_merge", "cuvs_sync"]
TRACE_FIELDS = TRACE_CONFIG + [
    "t_start_unix", "repeat", "warmup", "verified", "file_index", "path", "rows", "stall_s",
    "engine_call_s", "build_s", "merge_s", "sync_wait_s", "sync_waited", "h2d_s",
    "h2d_exposed_s", "h2d_overlapped", "t_got_s",
    "t_done_s", "w_start_s", "w_end_s",
    "fetch_s", "read_table_s", "convert_s", "widen_host_s", "publish_s", "w_publish_end_s",
    "handoff_s", "w_cpu_user_s", "w_cpu_sys_s", "read_path", "native_read_s",
    # --filter-config: the filter columns' own read (native mode) and the filter step (reader side,
    # after the vectors are published)
    "payload_read_s", "w_filter_start_s", "w_filter_end_s",
]
WORKER_TRACE = ("w_start_s", "w_end_s", "fetch_s", "read_table_s", "convert_s",
                "read_path", "native_read_s", "widen_host_s", "publish_s", "w_publish_end_s", "handoff_s", "w_cpu_user_s",
                "w_cpu_sys_s", "payload_read_s", "w_filter_start_s", "w_filter_end_s")
