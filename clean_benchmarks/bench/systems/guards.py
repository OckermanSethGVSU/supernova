"""Memory guards for a harness leg, checked before any pass: refuse a --verify run whose host
RAM peak (passes or oracle) would not fit, an --h2d-overlap run whose two device blocks would
not fit on the GPU, and a cuVS device-merge run whose merge would not fit. Per modality.
"""

import math

import numpy as np

from bench.record import schema
from bench.data.sources import (NATIVE_CHUNKS_IN_FLIGHT, PRELOAD_BUFFER_BYTES, preload_batch_rows,
                          preload_workers, reader_pool_threads, usable_cpu_count)
from bench.systems import cuvs_engine, pylate_engine, spare_engine
from bench.data.formats import multivector as multivec
from bench.data.formats import sparse


def mem_available_bytes(path=None):
    """MemAvailable from /proc/meminfo in bytes, or None where there is none."""
    try:
        with open(path or schema.MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def verify_memory_need(rows, d, nq, k, passes, block_itemsize, ring_bytes=0,
                       transient_bytes=0, reader_bytes=0, context_bytes=0,
                       query_copies=1, reload_bytes=0, scratch_bytes=0, oracle_rows=None):
    """Host bytes --verify holds at its peak, estimated from the footers.

    Two phases never overlap, so the larger counts, plus what both hold:
    the engine's fp32 queries (``query_copies``: 2 for cosine, the raw and
    the normalised copy), every pass's D / I (12 B a cell), ``context_bytes``
    (CUDA_HOST_ALLOWANCE on a GPU run), ``scratch_bytes`` (the CPU backend's
    cosine scratch, kept through the oracle) and VERIFY_PROCESS_ALLOWANCE.
    ``verify_memory_phases`` returns the three parts.

    * The passes: the reader ring (``ring_bytes``), the readers' decode
      transients (``reader_bytes``) and this process's own inline reads
      (``transient_bytes``).
    * The oracle: the reloaded corpus at the stored dtype
      (``block_itemsize``), searched in place, plus the larger of the
      reload's decode transient (``reload_bytes``; freed after every file)
      and the oracle's own allocations: float64 queries, float64 D / int64 I
      at k + 1 columns, and its tiles (oracle_scratch_bytes, rows bounded by
      ``oracle_rows``, else ``rows``). The readers and ring are closed
      before the reload."""
    shared, in_passes, oracle = verify_memory_phases(
        rows, d, nq, k, passes, block_itemsize, ring_bytes, transient_bytes, reader_bytes,
        context_bytes, query_copies, reload_bytes, scratch_bytes, oracle_rows)
    return shared + max(in_passes, oracle)


def oracle_bytes(d, nq, k, max_rows):
    """Host bytes the float64 oracle allocates beyond the corpus it searches
    (see verify_memory_need): its float64 queries, its D (float64) / I
    (int64) at k + 1 columns and its tile scratch."""
    from bench.verify.oracle import oracle_scratch_bytes
    return nq * d * 8 + nq * (k + 1) * 16 + oracle_scratch_bytes(d, nq, max_rows)


def verify_memory_phases(rows, d, nq, k, passes, block_itemsize, ring_bytes=0,
                         transient_bytes=0, reader_bytes=0, context_bytes=0,
                         query_copies=1, reload_bytes=0, scratch_bytes=0, oracle_rows=None):
    """``(shared, passes, oracle)`` bytes of verify_memory_need (same
    arguments): what both phases hold, the passes' phase, the oracle's."""
    corpus = rows * d * block_itemsize
    shared = (query_copies * nq * d * 4 + passes * nq * k * 12 + context_bytes
              + scratch_bytes + VERIFY_PROCESS_ALLOWANCE)
    in_passes = ring_bytes + transient_bytes + reader_bytes
    oracle = corpus + max(reload_bytes,
                          oracle_bytes(d, nq, k, oracle_rows if oracle_rows else rows))
    return shared, in_passes, oracle


# Arrow's pool peak while decoding a list<float> column, bytes per leaf value,
# by stored value width (decoded table, levels, pages, builder growth).
# Measured worst cases (pyarrow 24.0.0, mimalloc): f16 15.2 B, f32 25.5 B,
# f64 42.1 B, rounded up with a margin; other widths use the f32 figure.
# Used for the reload's per-batch decode (preload_transient).
ARROW_DECODE_BYTES_PER_VALUE = {2: 18, 4: 28, 8: 48}
# the --verify reload (InMemorySource.load): per decode thread, one record
# batch at PRELOAD_BATCH_FACTOR x the per-value figure above, plus the read
# buffer (PRELOAD_BUFFER_BYTES). Measured per thread (24 threads, 4 MiB
# batches): f32 26 B, f16 21 B per value; this charges 42 / 27 B
PRELOAD_BATCH_FACTOR = 1.5
# what the run process grows by over a --verify run beyond the terms counted
# (allocator slack, Python objects, faiss.knn's scratch, lazy imports):
# measured 50-75 MB on the dev box (CPU backend, small corpora)
VERIFY_PROCESS_ALLOWANCE = 128 << 20
# host memory a GPU run holds for the CUDA driver / runtime, torch and the
# engine's libraries next to the oracle: a generous allowance, not a
# measurement (the A10G node measured 0.1-0.5 GB)
CUDA_HOST_ALLOWANCE = 2 * 2**30
# the guard's margin over its own estimate: measured peaks on the A10G node
# (Pes2o-VE, g5.8xlarge) came within 3-10% of the unscaled estimate. With it
# a g5.8xlarge (MemAvailable 117.65 GiB) verifies at most 22 Pes2o-VE files with native
# reads
VERIFY_SAFETY = 1.25
# --read-mode native: per column chunk in flight, its staging buffer (the
# compressed chunk) plus the decoder's scratch (up to a decompressed page,
# never more than the chunk's decoded rows): an allowance of 3x the chunk
NATIVE_STAGING_FACTOR = 3


def native_read_transient(files, footers):
    """Host bytes rust_fetch's dense read holds beside the block, largest file.

    Up to NATIVE_CHUNKS_IN_FLIGHT column chunks at once, each charged at
    NATIVE_STAGING_FACTOR x the chunk. An allowance from the decoder's
    structure (dense.rs read_dense), not a measured fit."""
    best = 0
    for r in files:
        f = footers[r.path]
        chunk = math.ceil(f.column_bytes * (f.max_rg_rows or f.num_rows) / f.num_rows)
        best = max(best, NATIVE_STAGING_FACTOR * min(f.num_row_groups,
                                                     NATIVE_CHUNKS_IN_FLIGHT) * chunk)
    return best


def native_read_bytes(files, footers, d, itemsize, slot=False):
    """Host bytes one native read holds, for the largest file:
    ``native_read_transient`` plus the block itself unless ``slot`` (a reader
    process decoding into its ring slot: the block is the ring's)."""
    block = 0 if slot else max(footers[r.path].num_rows for r in files) * d * itemsize
    return native_read_transient(files, footers) + block


def preload_transient(files, footers, d, itemsize, threads=None):
    """Host bytes the --verify reload (InMemorySource.load) holds on top of
    the blocks already loaded, for the largest file.

    ``preload_workers`` threads, each one record batch at
    PRELOAD_BATCH_FACTOR x ARROW_DECODE_BYTES_PER_VALUE plus its read buffer.
    The block itself is counted by the caller; Arrow frees the rest after
    every file."""
    per_value = ARROW_DECODE_BYTES_PER_VALUE.get(itemsize, ARROW_DECODE_BYTES_PER_VALUE[4])
    best = 0
    for r in files:
        f = footers[r.path]
        rg_rows = f.max_rg_rows or f.num_rows
        batch = min(preload_batch_rows(d, itemsize), rg_rows) * d * per_value
        chunk = math.ceil(f.column_bytes * rg_rows / f.num_rows)  # one row group's, on disk
        decode = preload_workers(f.num_row_groups, threads) * math.ceil(
            PRELOAD_BATCH_FACTOR * batch + min(PRELOAD_BUFFER_BYTES, chunk))
        best = max(best, decode)
    return best


def verify_memory_guard(args, files, footers, d, nq, host_dtype):
    """Refuse a --verify run whose peak would not fit in host RAM (dense).

    Compares verify_memory_need x VERIFY_SAFETY with MemAvailable, before any
    pass. Skipped where /proc/meminfo is missing."""
    avail = mem_available_bytes()
    if avail is None:
        return
    rows = sum(footers[r.path].num_rows for r in files)
    first = footers[files[0].path]
    stored = (first.value_bits or 32) // 8
    held = np.dtype(host_dtype).itemsize if host_dtype is not None else stored
    values = max(footers[r.path].num_rows for r in files) * d
    total = usable_cpu_count()
    load = preload_transient(files, footers, d, stored, total)  # the oracle's reload
    ring = transient = reader_bytes = 0
    if not args.prefetch:
        # inline reads in this process, plus the host widen
        transient = native_read_bytes(files, footers, d, stored) + (
            values * held if held != stored else 0)
    else:
        readers = min(args.prefetch, args.workers, len(files))
        widened = values * held if held != stored else 0
        # --h2d-overlap / --d2h-overlap: one more slot each (the consumer
        # also holds the block uploading ahead / whose copy back is in
        # flight: ParquetSource's extra_slots)
        ring = (min(args.prefetch, len(files)) + 1 + overlap_extra_slots(args)) \
            * values * held
        # one read in flight per reader, straight into its slot unless a host
        # widen comes first (then its widened copy too); own pools, kept until
        # the readers are closed
        cpu, _ = reader_pool_threads(0, 0, readers, "split")
        reader_bytes = sum(native_read_bytes(files, footers, d, stored, slot=not widened)
                           + widened for _ in cpu)
    context = CUDA_HOST_ALLOWANCE if args.backend in ("gpu", "cuvs") else 0
    cosine = getattr(args, "metric", "ip") == "cosine"
    # the backend's host cosine scratch (NormScratch): an fp32 buffer of the
    # largest block (CPU backends; the GPU backends normalise on the device)
    scratch = values * 4 if cosine and args.backend in schema.HOST_BACKENDS else 0
    cos_scratch = scratch
    scratch += d2h_pinned_bytes(args, nq)
    pos = (rows, d, nq, args.k, args.warmup + args.repeats, stored,
           ring, transient, reader_bytes, context)
    # cosine: + the backend's normalised query copy (the float64 oracle's own
    # queries are in its phase: verify_memory_phases)
    kw = dict(query_copies=2 if cosine else 1, reload_bytes=load, scratch_bytes=scratch,
              oracle_rows=max(footers[r.path].num_rows for r in files))
    need = verify_memory_need(*pos, **kw)
    _, in_passes, oracle = verify_memory_phases(*pos, **kw)
    need = int(need * VERIFY_SAFETY)
    if need > avail:
        phase = ("the passes (the reads' decode transients"
                 f"{', the reader ring' if ring else ''})") if in_passes > oracle else \
            ("the oracle (the reloaded corpus plus the float64 reference's queries "
             "and tiles or the reload's decode transient)")
        raise SystemExit(
            f"--verify: the run needs about {need / 2**30:.2f} GiB of host RAM at its "
            f"peak, in {phase}: {rows} rows x {d}-D, estimate x {VERIFY_SAFETY} safety, "
            f"with the queries, results"
            f"{', the reader-process transients' if reader_bytes else ''}"
            f"{', the cosine normalise scratch' if cos_scratch else ''}"
            f"{', the --d2h pinned result buffers' if d2h_pinned_bytes(args, nq) else ''}"
            f"{', the CUDA host-context allowance' if context else ''}; more than the "
            f"{avail / 2**30:.2f} GiB available (MemAvailable). Verify a prefix of "
            "the corpus with --max-files N")


# --modality sparse, --verify guard: Arrow's decode of a sparse struct file as
# a multiple of its decoded column (nnz x (index + value width) + offsets): an
# allowance for the decoded table, a flatten copy and the packed block. On a
# 400k-row x 100-nnz file max-RSS was 1.4-1.6x what a factor of 3 allows
# (uint32 / int64 indices), so 5
SPARSE_DECODE_FACTOR = 5
# the sparse oracle's scratch: one dense score tile (sparse.CPU_TILE_CELLS
# cells) at float64 plus topk_tile's masks / cumsum / partition copy
SPARSE_TILE_BYTES_PER_CELL = 40
# merge_sorted per tile: the concatenated (nq, k + step) scores and ids, the
# argsort and the two gathers (float64 / int64): bytes per cell
SPARSE_MERGE_BYTES_PER_CELL = 40


def overlap_extra_slots(args):
    """Ring slots beyond prefetch + 1: one each for --h2d-overlap (block
    uploading ahead) and --d2h-overlap (block whose copy back is in flight);
    --cuvs-sync none holds the same one block in flight (one slot, shared with
    --d2h-overlap, which the device merge ignores)."""
    return int(bool(getattr(args, "h2d_overlap", False))) + int(
        bool(getattr(args, "d2h_overlap", False)) or getattr(args, "cuvs_sync", None) == "none")


def d2h_pinned_bytes(args, nq):
    """Host bytes of --d2h pinned's two (D float32, I int64) nq x k buffer
    pairs, 0 otherwise. Each is charged at the next power of two, as torch's
    CachingHostAllocator rounds pinned allocations up."""
    if getattr(args, "d2h", "pageable") != "pinned":
        return 0
    cells = nq * args.k
    return 2 * sum(1 << max(0, (cells * w - 1).bit_length()) for w in (4, 8))


def sparse_oracle_bytes(corpus, nnz, rows, nq, k, metric):
    """Peak host bytes of sparse.oneshot_reference beyond the queries.

    The packed corpus, float64 CSR copies (12 B a nonzero, 8 B a row each),
    the score tile, the per-tile merge and the k + 1 D / I. CSR copies alive
    at the peak: ip 2; cosine 4 (the norm transients), charged 5 because
    tracemalloc measured 0.94-1.01 of the 4-copy figure, rising with rows."""
    csr = nnz * 12 + rows * 8
    copies = 5 if metric == "cosine" else 2
    step = max(1, sparse.CPU_TILE_CELLS // max(1, nq))
    return (corpus + copies * csr + sparse.CPU_TILE_CELLS * SPARSE_TILE_BYTES_PER_CELL
            + nq * (k + 1 + step) * SPARSE_MERGE_BYTES_PER_CELL + nq * (k + 1) * 12)


def sparse_verify_memory_guard(args, files, footers, q, host_dtype):
    """verify_memory_guard for --modality sparse (arithmetic in sparse_verify_need).

    Same two phases and margin, from the footers' row and nonzero counts
    (an upper bound), with a rough decode allowance (SPARSE_DECODE_FACTOR)."""
    avail = mem_available_bytes()
    if avail is None:
        return
    t = sparse_verify_need(args, files, footers, q, host_dtype)
    need, in_passes, oracle = t["need"], t["in_passes"], t["oracle"]
    rows, nnz, held = t["rows"], t["nnz"], t["held"]
    if need > avail:
        phase = "the passes" if in_passes > oracle else "the oracle"
        raise SystemExit(
            f"--verify: the sparse run needs about {need / 2**30:.2f} GiB of host RAM at "
            f"its peak, in {phase}: {rows} rows, <= {nnz} nonzeros (held at {held} B "
            f"values), estimate x {VERIFY_SAFETY} safety; more than the "
            f"{avail / 2**30:.2f} GiB available (MemAvailable). Verify a prefix of the "
            "corpus with --max-files N")


def sparse_verify_need(args, files, footers, q, host_dtype):
    """sparse_verify_memory_guard's estimate as a dict: shared, in_passes,
    oracle, need = (shared + max(in_passes, oracle)) x VERIFY_SAFETY, plus
    rows / nnz / held for the message."""
    feet = [footers[r.path] for r in files]
    rows = sum(f.num_rows for f in feet)
    nnz = sum(f.sparse_nnz for f in feet)
    # value width as stored (the decode holds it before any narrowing), the
    # widest file's; indices charged at 8 B (Footer does not record their width)
    stored = max({16: 2, 64: 8}.get(f.sparse_value_bits, 4) for f in feet)
    held = np.dtype(host_dtype).itemsize if host_dtype is not None else stored
    packed = [sparse.CODEC.block_bytes(f, host_dtype) for f in feet]
    corpus = sum(sparse.CODEC.block_bytes(f, None) for f in feet)
    biggest = max(range(len(feet)), key=lambda i: packed[i])
    fb = feet[biggest]
    # the native read's fetched image, held until pyarrow has decoded it:
    # footer tail + the column's chunks + per-range padding (sources'
    # _FETCH_PAD / _FETCH_COALESCE, both leaves, page rounding)
    raw = min(files[biggest].size, fb.column_bytes + (1 << 20)
              + (4096 + (8 << 10) + 4096) * 2 * fb.num_row_groups)
    transient = SPARSE_DECODE_FACTOR * (fb.sparse_nnz * (8 + stored) + 8 * fb.num_rows) \
        + packed[biggest] + raw
    if not args.prefetch:
        in_passes = transient
    else:
        readers = min(args.prefetch, args.workers, len(files))
        ring = (min(args.prefetch, len(files)) + 1 + overlap_extra_slots(args)) * max(packed)
        in_passes = ring + readers * transient
    passes = args.warmup + args.repeats
    shared = (3 * q.nbytes + 2 * q.nnz * 12 + passes * q.n_rows * args.k * 12
              + VERIFY_PROCESS_ALLOWANCE + d2h_pinned_bytes(args, q.n_rows)
              + (CUDA_HOST_ALLOWANCE if args.backend in ("cuvs", "spare", "pylate") else 0))
    oracle = sparse_oracle_bytes(corpus, nnz, rows, q.n_rows, args.k, args.metric)
    need = int((shared + max(in_passes, oracle)) * VERIFY_SAFETY)
    return {"need": need, "shared": shared, "in_passes": in_passes, "oracle": oracle,
            "transient": transient, "raw": raw, "rows": rows, "nnz": nnz, "held": held}


# --h2d-overlap device guard: headroom left beyond the estimate (allocator
# fragmentation, the CUDA context's own growth, cuBLAS workspaces)
OVERLAP_GPU_MARGIN = 512 << 20


# device bytes per result cell at the peak of a block's results: D float32 4 +
# I int64 8 + _globalize's mask 1, I + offset temporary 8 and output 8, + the
# previous block's D / I (12) still alive under --d2h-overlap's record_stream
OVERLAP_RESULT_BYTES_PER_CELL = 41


def overlap_gpu_need(block_bytes, rows, d, nq, k, stored_itemsize, compute_itemsize,
                     engine_bytes):
    """Device bytes a --h2d-overlap run needs beyond the resident queries (an estimate).

    Two blocks at the stored width (searched + uploading), one widened copy
    if the device cast runs, the outputs (OVERLAP_RESULT_BYTES_PER_CELL a
    cell) and the engine's scratch (FAISS temp memory; cuVS's RMM workspace
    is not estimated)."""
    cast = rows * d * compute_itemsize if compute_itemsize != stored_itemsize else 0
    return (2 * block_bytes + cast + nq * k * OVERLAP_RESULT_BYTES_PER_CELL + engine_bytes
            + OVERLAP_GPU_MARGIN)


def overlap_gpu_memory_guard(args, backend, source, nq, dtype):
    """Refuse a --h2d-overlap run whose two device blocks (plus the rest of
    overlap_gpu_need) would not fit in the GPU's free memory, before any pass.
    Skipped off CUDA (tests on torch's CPU device)."""
    if not getattr(backend, "_cuda", False):
        return
    torch = backend.torch
    free, total = torch.cuda.mem_get_info(backend.device)
    footers = source.read_footers()
    if getattr(args, "modality", "dense") == "sparse":
        return _sparse_overlap_guard(args, source, footers, nq, free, total, backend)
    if getattr(args, "modality", "dense") == "multivector":
        return _mv_overlap_guard(args, source, footers, nq, free, total)
    big = max(source.files, key=lambda r: footers[r.path].num_rows)
    foot = footers[big.path]
    stored = (foot.value_bits or 32) // 8
    if args.widen == "host":
        stored = np.dtype(dtype).itemsize
    rows, d = foot.num_rows, foot.dim
    engine = 0
    if args.backend == "gpu":
        engine = schema.FAISS_MAX_TEMP_MEM if args.temp_mem is None else \
            min(args.temp_mem, schema.FAISS_MAX_TEMP_MEM)
    need = overlap_gpu_need(rows * d * stored, rows, d, nq, args.k, stored,
                            np.dtype(dtype).itemsize, engine)
    if need > free:
        raise SystemExit(
            f"--h2d-overlap: the GPU would need about {need / 2**30:.2f} GiB beyond the "
            f"queries (two {rows} x {d} blocks of {rows * d * stored / 2**30:.2f} GiB, "
            f"the outputs{', the device cast' if np.dtype(dtype).itemsize != stored else ''}"
            f"{', FAISS temp memory' if engine else ''}, {OVERLAP_GPU_MARGIN >> 20} MiB "
            f"margin; cuVS's RMM workspace not estimated), only {free / 2**30:.2f} of "
            f"{total / 2**30:.2f} GiB free. Run without --h2d-overlap")


def _sparse_overlap_guard(args, source, footers, nq, free, total, backend=None):
    """overlap_gpu_memory_guard for --modality sparse.

    Two of the largest packed CSR blocks, the device widen of fp16 values
    (4 B a nonzero), the outputs, and the engine's scratch: SPARe's model,
    or cuVS's dense score tile at the effective batch sizes plus its merge
    buffers (its other RMM scratch is not estimated, as for dense cuVS)."""
    host = np.float32 if args.widen == "host" else None
    feet = [footers[r.path] for r in source.files]
    # block_bytes is None for a file whose column is not a sparse struct: that
    # file then fails its read with the codec's clear error; not here
    size = [sparse.CODEC.block_bytes(f, host) or 0 for f in feet]
    i = max(range(len(feet)), key=size.__getitem__)
    f = feet[i]
    block = size[i]
    # only an fp16-stored block is widened on the device (fp64 is narrowed by
    # the host decode); any fp16 file may be widened next to the largest block
    widen = max(((g.sparse_nnz or 0) * 4 for g in feet if g.sparse_value_bits == 16),
                default=0) if host is None else 0
    if getattr(args, "backend", "cuvs") == "spare":
        q = getattr(backend, "q", None)
        tile = spare_engine.spare_engine_bytes(args, feet, nq, pairs=q.nnz if q is not None else 0,
                                  q_cols=q.n_cols if q is not None else 0)
        what = "SPARe's model + scratch"
    else:
        # cuVS's dense score tile at the effective batch sizes:
        # (query batch) x (index batch) x (float32 + int32)
        max_rows = max(g.num_rows for g in feet)
        if backend is not None and hasattr(backend, "params"):
            bi, bq = backend.params(max_rows, nq)
        else:
            bi, bq = cuvs_engine.cuvs_sparse_effective_batches(
                getattr(args, "cuvs_sparse_batches", None) or (cuvs_engine.SPARSE_BATCH_STOCK,) * 2,
                cuvs_engine.SPARSE_STOCK_BATCHES, max_rows)
        tile = min(bq, nq) * min(bi, max_rows) * 8
        # merge_buffer_{dists,indices} (batch_rows * k * 3 each, sparse_knn.cuh)
        # + the patched short-batch tmp_{dists,indices} (batch_rows * k): 32 B a cell
        tile += min(bq, nq) * args.k * 32
        what = f"cuVS's score tile at batch_size_index x batch_size_query {bi}x{bq}"
    need = (2 * block + widen + nq * args.k * OVERLAP_RESULT_BYTES_PER_CELL + tile
            + OVERLAP_GPU_MARGIN)
    if need > free:
        raise SystemExit(
            f"--h2d-overlap: the GPU would need about {need / 2**30:.2f} GiB beyond the "
            f"queries (two packed sparse blocks of {block / 2**30:.2f} GiB ({f.num_rows} rows, "
            f"<= {f.sparse_nnz} nonzeros){', the device widen' if widen else ''}, the "
            f"outputs, {what} ({tile >> 20} MiB), {OVERLAP_GPU_MARGIN >> 20} MiB "
            f"margin), only {free / 2**30:.2f} of {total / 2**30:.2f} GiB free. Run without "
            "--h2d-overlap")


def _mv_overlap_guard(args, source, footers, nq, free, total):
    """overlap_gpu_memory_guard for --modality multivector: two packed blocks,
    the device widen of fp16 tokens, the outputs and PyLate's scratch."""
    host = np.float32 if args.widen == "host" else None
    feet = [footers[r.path] for r in source.files]
    size = [multivec.CODEC.block_bytes(f, host) or 0 for f in feet]
    i = max(range(len(feet)), key=size.__getitem__)
    f, block = feet[i], size[i]
    widen = max(((g.mv_values or 0) * 4 for g in feet if g.mv_value_bits == 16),
                default=0) if host is None else 0
    tile = pylate_engine.pylate_engine_bytes(args, feet, nq)
    need = (2 * block + widen + nq * args.k * OVERLAP_RESULT_BYTES_PER_CELL + tile
            + OVERLAP_GPU_MARGIN)
    if need > free:
        raise SystemExit(
            f"--h2d-overlap: the GPU would need about {need / 2**30:.2f} GiB beyond the "
            f"queries (two packed multi-vector blocks of {block / 2**30:.2f} GiB "
            f"({f.num_rows} documents){', the device widen' if widen else ''}, the "
            f"outputs, PyLate's scratch ({tile >> 20} MiB), {OVERLAP_GPU_MARGIN >> 20} MiB "
            f"margin), only {free / 2**30:.2f} of {total / 2**30:.2f} GiB free. Run without "
            "--h2d-overlap")


# --modality multivector, --verify guard: Arrow's decode of a list<list<float>>
# file as a multiple of its token values: an allowance, not a measurement
MV_DECODE_FACTOR = 3


def mv_oracle_bytes(corpus, biggest_values, biggest_rows, nq, k, q_values):
    """Peak host bytes of multivec.oneshot_reference beyond the queries.

    The packed corpus, two float64 copies of the largest block's tokens, the
    token score tile (charged twice for reduceat's output), the (nq, block
    rows) score matrix at 40 B a cell, the queries' float64 copy and the
    k + 1 merge / D / I."""
    return (corpus + 2 * biggest_values * 8 + 2 * multivec.CPU_TILE_CELLS * 8
            + nq * biggest_rows * SPARSE_TILE_BYTES_PER_CELL + q_values * 8
            + nq * (k + 1) * (SPARSE_MERGE_BYTES_PER_CELL + 12))


def mv_verify_need(args, files, footers, q, host_dtype):
    """mv_verify_memory_guard's estimate as a dict, like sparse_verify_need."""
    feet = [footers[r.path] for r in files]
    rows = sum(f.num_rows for f in feet)
    values = sum(f.mv_values for f in feet)
    stored = max({16: 2, 64: 8}.get(f.mv_value_bits, 4) for f in feet)
    packed = [multivec.CODEC.block_bytes(f, host_dtype) for f in feet]
    corpus = sum(multivec.CODEC.block_bytes(f, None) for f in feet)
    b = max(range(len(feet)), key=lambda i: packed[i])
    fb = feet[b]
    raw = files[b].size   # the native read's fetched image
    transient = MV_DECODE_FACTOR * fb.mv_values * stored + packed[b] + raw
    if not args.prefetch:
        in_passes = transient
    else:
        readers = min(args.prefetch, args.workers, len(files))
        ring = (min(args.prefetch, len(files)) + 1 + overlap_extra_slots(args)) * max(packed)
        in_passes = ring + readers * transient
    passes = args.warmup + args.repeats
    shared = (3 * q.nbytes + passes * q.n_rows * args.k * 12 + VERIFY_PROCESS_ALLOWANCE
              + d2h_pinned_bytes(args, q.n_rows)
              + (CUDA_HOST_ALLOWANCE if args.backend == "pylate" else 0))
    oracle = mv_oracle_bytes(corpus, fb.mv_values, fb.num_rows, q.n_rows, args.k,
                             q.tokens.size)
    need = int((shared + max(in_passes, oracle)) * VERIFY_SAFETY)
    return {"need": need, "shared": shared, "in_passes": in_passes, "oracle": oracle,
            "transient": transient, "raw": raw, "rows": rows, "values": values}


def mv_verify_memory_guard(args, files, footers, q, host_dtype):
    """verify_memory_guard for --modality multivector (rough: MV_DECODE_FACTOR)."""
    avail = mem_available_bytes()
    if avail is None:
        return
    t = mv_verify_need(args, files, footers, q, host_dtype)
    if t["need"] > avail:
        phase = "the passes" if t["in_passes"] > t["oracle"] else "the oracle"
        raise SystemExit(
            f"--verify: the multi-vector run needs about {t['need'] / 2**30:.2f} GiB of "
            f"host RAM at its peak, in {phase}: {t['rows']} documents, <= {t['values']} "
            f"token values, estimate x {VERIFY_SAFETY} safety; more than the "
            f"{avail / 2**30:.2f} GiB available (MemAvailable). Verify a prefix of the "
            "corpus with --max-files N")


# cuVS's own recipe (CalcTruthMerge): device bytes at the merge's peak, per (query x 2k) cell,
# i.e. per column of the concatenation of the running top-k (k wide) and a block's (<= k wide).
# An array k wide counts half per 2k cell; ids are counted at 8 B (int64, the widest dtype
# offset_neighbor_indices gives):
#   running D (float32 4) + I (8), k wide                                     12 / 2 =   6
#   the block's new D (4) + I (torch int64 8), k wide (cupy.asarray is a view)  12 / 2 =   6
#   the block id copy: offset_neighbor_indices' astype + the offset add, both
#     alive during the add (8 + 8), k wide                                    16 / 2 =   8
#   the concatenated D (4) + I (8)                                                     12
#   the sort keys (-distances, inner product; float32)                                  4
#   argsort's data copy (float32; CuPy copies the input to C order)                     4
#   argsort's index (intp 8) and segment keys (intp 8, ndim > 1)                      16
#   thrust scratch: stable_sort_by_key on zip(segment key, data) with a comparator is
#     a merge sort that copies its keys (tuple<size_t, float>, 16 padded) and values
#     (intp 8): 24, budgeted twice for its other temporaries                          48
#   take_along_axis: its D (4) + I (8) output, k wide (12 / 2 = 6), and its fancy-index
#     temporaries (the broadcast row index + the flat index, int64, k wide: 16 / 2
#     = 8) for the larger of the two calls                                            14
#   = 118; rounded up to 140 for what is not itemised (pool rounding, CuPy's small
#     internal arrays, the -1 masks of result()).
DEVICE_MERGE_BYTES_PER_2K_CELL = 140


def cuvs_device_merge_need(args, source, footers, nq, dtype):
    """(bytes, parts) the GPU needs for cuVS's own recipe beyond the queries: the CuPy merge's
    peak (``merge``: nq x 2k cells x DEVICE_MERGE_BYTES_PER_2K_CELL) and, when a block is
    stored narrower than the compute dtype and widened on the device, the widened copy of the
    largest block (``widen``: dense rows x d at the compute width; sparse 4 B a nonzero of an
    fp16 file). An estimate: cuVS's own workspace (RMM) is not counted."""
    parts = {"merge": nq * 2 * args.k * DEVICE_MERGE_BYTES_PER_2K_CELL}
    feet = [footers[r.path] for r in source.files]
    widen = 0
    if getattr(args, "widen", "device") != "host":
        if getattr(args, "modality", "dense") == "sparse":
            widen = max(((f.sparse_nnz or 0) * 4 for f in feet if f.sparse_value_bits == 16),
                        default=0)
        else:
            compute = np.dtype(dtype).itemsize
            widen = max((f.num_rows * (f.dim or 0) * compute for f in feet
                         if (f.value_bits or 32) // 8 < compute), default=0)
    if widen:
        parts["widen"] = widen
    return sum(parts.values()), parts


def cuvs_device_merge_guard(args, backend, source, nq, dtype):
    """Refuse a cuVS device-merge run (its own recipe) whose merge would not fit in the GPU's
    free memory with one block of the largest file (two with --h2d-overlap, three with
    --h2d-overlap and --cuvs-sync none) and its device
    widen, before any pass. Skipped off CUDA (tests on torch's CPU device).

    With the RMM pool on, the pool's initial size (``backend.rmm_pool_bytes``) is already
    reserved from the GPU, so mem_get_info does not count it as free; CuPy allocates from it
    (rmm_cupy_allocator), so it is headroom for the merge, not used memory. The blocks and the
    widen are torch allocations (its own caching allocator) and must fit in the free memory."""
    if not getattr(backend, "_cuda", False):
        return
    free, total = backend.torch.cuda.mem_get_info(backend.device)
    footers = source.read_footers()
    need, parts = cuvs_device_merge_need(args, source, footers, nq, dtype)
    blocks = 1 + int(bool(args.h2d_overlap))
    if blocks > 1 and getattr(args, "cuvs_sync", None) == "none":
        # --cuvs-sync none: the block before may still be searched while block i+1 uploads
        # on the copy stream, so its memory is not yet free for that upload
        blocks += 1
    if getattr(args, "modality", "dense") == "sparse":
        host = np.float32 if args.widen == "host" else None
        block = max((sparse.CODEC.block_bytes(footers[r.path], host) or 0
                     for r in source.files), default=0)
    else:
        block = max(source.max_block_bytes(), 0)
    parts[f"{blocks} block(s)"] = blocks * block
    pool = int(getattr(backend, "rmm_pool_bytes", 0) or 0) \
        if cuvs_engine.cuvs_pool_spec(args) else 0
    torch_need = blocks * block + parts.get("widen", 0) + OVERLAP_GPU_MARGIN
    total_need = torch_need + max(0, parts["merge"] - pool)
    if total_need > free:
        detail = ", ".join(f"{k} {v / 2**30:.2f} GiB" for k, v in parts.items())
        raise SystemExit(
            f"--backend cuvs (cuVS's own recipe, --cuvs-merge device): the GPU would need "
            f"about {total_need / 2**30:.2f} GiB beyond the queries ({detail}, "
            f"{OVERLAP_GPU_MARGIN >> 20} MiB margin"
            + (f"; the merge may use the RMM pool's {pool / 2**30:.2f} GiB" if pool else "")
            + f"; cuVS's RMM workspace not estimated), only {free / 2**30:.2f} of "
            f"{total / 2**30:.2f} GiB free. Use --cuvs-merge host (per_file_host), fewer "
            "queries, or a larger GPU")
