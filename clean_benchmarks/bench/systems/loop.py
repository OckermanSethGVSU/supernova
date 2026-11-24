"""The timed loop of a harness leg: one pass over the source through an engine
(run_pass), its time split into stall / engine_call / merge / other, plus per-file trace rows.
"""

import resource
import time
import warnings

import faiss  # must import after cudalibs.torch_first() has loaded torch's CUDA libraries
import numpy as np

from bench.record import results_io, schema
from bench.data.sources import NOT_READY
from bench.timing import Stages, now
from bench.systems import streaming as base, cuvs_engine
from bench.data.formats import multivector as multivec
from bench.data.formats import sparse


def block_rows(x):
    """Rows of a block: a packed sparse / multi-vector block's header count,
    else shape[0]."""
    if sparse.is_packed(x):
        return sparse.packed_rows(x)
    if multivec.is_packed(x):
        return multivec.packed_rows(x)
    return x.shape[0]


# --------------------------------------------------------------- core loop

def net_rx_bytes(path=None):
    """Bytes received so far on every interface but ``lo`` (/proc/net/dev);
    None where that file does not exist. Read outside the timed span."""
    try:
        with open(path or schema.NET_DEV) as f:
            lines = f.readlines()[2:]  # two header lines
    except OSError:
        return None
    total = 0
    for line in lines:
        name, sep, rest = line.partition(":")
        if sep and name.strip() != "lo":
            total += int(rest.split()[0])
    return total


def run_pass(source, backend, nq, k, max_block_bytes=0, d=None, n_files=0, filt=None, clock=None):
    """One full pass over ``source``. Returns (D, I, info dict).

    ``d``, if given, is checked against every block (a wrong --dim would
    otherwise reach FAISS's asserts). ``n_files`` > 0: only the first
    n_files files (--warmup-files).

    --h2d-overlap: just before the engine call for block i, a non-blocking
    ``poll()`` asks whether block i+1 is already read; if so it is passed as
    ``ahead=`` so its upload is issued before search i, else it comes from
    the usual next() and is uploaded on demand. Poll time counts as
    source_stall. The merge stays serial after each call.

    ``info["trace"]`` holds one tuple per file, from the same stamps the
    buckets are summed from, times relative to the pass start; ``trace_rows``
    turns it into rows after the pass.

    ``clock``: an optional bench.timing.LegClock, observe only, started just
    after ``t_start`` and stopped just after ``t_end``."""
    # A distance engine (cuVS cosine: 1 - cos, ascending) is merged as
    # distances in a min-heap and turned into similarities once, after the
    # last block. 1 - d reverses order, so the same candidates win with the
    # same float32 scores. Exact ties at the k-th place may keep a different id.
    dist = bool(getattr(backend, "returns_distance", False))
    # --cuvs-merge device: the backend keeps the running top-k on the GPU (NVIDIA's calc_truth),
    # so there is no host heap; each block returns (None, None) and the merged top-k is copied
    # back once, after the last block
    dm = bool(getattr(backend, "device_merge", False))
    heap = None if dm else faiss.ResultHeap(nq, k, keep_max=not dist)

    def add(D, I):
        if D is None:
            return  # merged on the device
        heap.add_result(D, I)
    if filt is not None:
        # --filter-config: each block's search is shaped by its masks
        # (prepare, then call or submit, then finish)
        filt.stats = Stages()
    backend.stats = Stages()
    stall = call = merge = 0.0
    rows = max_rows = max_bytes = 0
    nnz = None  # sparse blocks: their nonzeros, summed
    nnz_kept = None  # ... the ones searched (after --sparse-project; else = nnz)
    tokens = None  # multi-vector blocks: their tokens, summed
    files = []
    rx0 = net_rx_bytes()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    backend.alloc_peak_reset()  # host-side counter, no device sync
    overlap = getattr(backend, "overlap", False)
    # GPU-resident backends add each block's global row offset to the ids on
    # the device, before the copy back; host backends get it added below
    on_device = bool(getattr(backend, "applies_offset", False))
    # --d2h-overlap: submit block i+1, then finish block i (wait for its copy
    # back and merge it) while the GPU works on i+1
    # --cuvs-sync none uses the same one-block pipeline: the host waits for block i-1's end
    # (the event behind its merge) only after submitting block i, and that wait is what
    # releases block i-1's host buffers (ring slot, masks) for reuse
    pipe = bool(getattr(backend, "d2h_overlap", False) or getattr(backend, "defer_wait", False))
    pending = None  # pipe: (token, block, n, file_stall, t1, submit seconds, filter Prep)
    next_prep = {}  # --filter-config + --h2d-overlap: id(block i+1) -> its Prep (None = skipped)
    if pipe and not on_device:
        # the pipelined branch relies on the backend globalising ids itself
        raise ValueError("--d2h-overlap needs a backend that globalises ids on the "
                         "device (applies_offset)")

    def finish(p):
        """Complete a submitted block: wait for its copy back (engine_call)
        and merge it (merge); its trace row is written now."""
        nonlocal call, merge
        tok, blk, nb, fstall, tg, sub, fprep = p
        tw = now()
        D, I = backend.complete(tok)
        if fprep is not None:
            D, I = filt.finish(fprep, D, I)   # filter_handoff, inside engine_call
        t2 = now()
        add(D, I)
        t3 = now()
        call += t2 - tw
        merge += t3 - t2
        files.append((blk.path, nb, fstall, sub + (t2 - tw), t3 - t2, tg - t_start, t3 - t_start,
                      blk.info, backend.last_wait, backend.last_build,
                      getattr(backend, "last_h2d", None)))
    t_start_unix = time.time()
    t_start = now()
    if clock is not None:
        clock.start()
    if dm:
        backend.begin_pass()  # a fresh running top-k, inside the clock
    it = source.stream(n_files or None, pollable=overlap) if n_files or overlap \
        else iter(source)
    ahead = None       # block i+1, taken by the poll before search i
    ahead_stall = 0.0  # poll seconds spent on the next block (its stall)
    ended = False      # a poll found the end of the stream
    try:
        while True:
            t0 = now()
            if ahead is not None:
                block, ahead = ahead, None
            else:
                block = None if ended else next(it, None)
            t1 = now()
            stall += t1 - t0
            file_stall, ahead_stall = t1 - t0 + ahead_stall, 0.0
            if block is None:
                if pending is not None:
                    p, pending = pending, None
                    finish(p)
                    del p
                break
            xb = block.vectors
            off_kw = {"offset": block.offset} if on_device else {}
            n = block_rows(xb)
            full_bytes = xb.nbytes  # the decoded file's bytes (the MAX_FILE_BYTES cap, max_block_bytes)
            comp = getattr(block, "compact", None)
            if comp is not None:
                # compacted by its reader (gather / union): ``xb`` holds the
                # kept rows; rows / bytes stay the file's
                n = comp.n_rows
                full_bytes = n * xb.shape[1] * xb.itemsize
                del comp
            if sparse.is_packed(xb):
                # header reads: the file's stored nonzeros, and the ones kept
                # by --sparse-project's projection (the same without it)
                nnz = (nnz or 0) + sparse.packed_stored_nnz(xb)
                nnz_kept = (nnz_kept or 0) + sparse.packed_nnz(xb)
            elif multivec.is_packed(xb):
                tokens = (tokens or 0) + multivec.packed_tokens(xb)  # a header read
            elif d is not None and xb.shape[1] != d:
                raise ValueError(f"{block.path}: block is {xb.shape[1]}-D, "
                                 f"queries are {d}-D")
            if max_block_bytes and full_bytes > max_block_bytes:
                raise RuntimeError(f"{block.path}: decoded block {full_bytes} B > "
                                   f"the {max_block_bytes} B per-file cap (MAX_FILE_BYTES)")
            tc = t1  # engine_call starts here (after the poll, with --h2d-overlap)
            if overlap:
                tp = now()
                if not ended:
                    got = it.poll()  # never blocks
                    if got is None:
                        ended = True
                    elif got is not NOT_READY:
                        ahead = got
                    # no second reference: a stale one would hold a ring
                    # slot past its block's turn and make the ring grow
                    got = None
                tc = now()  # the call starts after the poll (stall)
                stall += tc - tp
                ahead_stall += tc - tp
            akw = {"ahead": None if ahead is None else ahead.vectors} if overlap else {}
            prep = None
            if filt is not None:
                # this block's engine input (prepared ahead, when it was polled
                # during the previous block), then block i+1's, so its upload
                # can be issued before this search (--h2d-overlap); a skipped
                # block (prepare -> None) is never uploaded
                prep = next_prep.pop(id(block), None) if id(block) in next_prep \
                    else filt.prepare(block)
                fahead = None
                if ahead is not None:
                    fahead = next_prep[id(ahead)] = filt.prepare(ahead)
                if prep is None:
                    tf = tf_end = now()
                    if pending is not None:
                        # keep the trace in file order: finish block i-1 first.
                        # finish() charges its own wait / merge, so that span is
                        # left out of this skipped block's call (no double count)
                        p, pending = pending, None
                        finish(p)
                        del p
                        tf_end = now()
                    t2 = now()
                    own = (tf - tc) + (t2 - tf_end)
                    call += own
                    files.append((block.path, n, file_stall, own, 0.0, t1 - t_start,
                                  t2 - t_start, block.info, (None, None), 0.0, None))
                    rows += n
                    max_rows = max(max_rows, n)
                    max_bytes = max(max_bytes, full_bytes)
                    del block, xb
                    continue
            if pipe:
                if filt is not None:
                    tok = filt.submit(backend, prep, ahead=fahead)
                else:
                    tok = backend.submit(xb, **akw, **off_kw)
                t2 = now()
                call += t2 - tc
                if pending is not None:
                    p, pending = pending, None
                    finish(p)  # block i-1, while the GPU searches this one
                    del p  # else it pins block i-1's ring slot through next(it): the ring grows
                pending = (tok, block, n, file_stall, t1, t2 - tc, prep)
                rows += n
                max_rows = max(max_rows, n)
                max_bytes = max(max_bytes, full_bytes)
                del tok, block, xb, prep  # the token / pending keep what must live
                continue
            if filt is not None:
                # global ids, all nq queries
                got = filt.finish(prep, *filt.call(backend, prep, on_device, ahead=fahead))
                del prep
            else:
                got = backend.search(xb, **akw, **off_kw)
                if not on_device:
                    # host backends: global = local + offset, in place (no nq x k
                    # temporary); -1 = no result (block < k rows) stays -1
                    np.add(got[1], block.offset, out=got[1], where=got[1] >= 0)
            t2 = now()
            if got is not None:
                add(*got)
            t3 = now()
            call += t2 - tc
            merge += t3 - t2
            files.append((block.path, n, file_stall, t2 - tc, t3 - t2,
                          t1 - t_start, t3 - t_start, block.info, backend.last_wait,
                          backend.last_build, getattr(backend, "last_h2d", None)))
            rows += n
            max_rows = max(max_rows, n)
            max_bytes = max(max_bytes, full_bytes)
            del block, xb, got
    finally:
        try:
            # a submitted block whose copy back is still in flight (an error):
            # wait for it before its pinned buffer / ring slot can be reused
            try:
                if pending is not None:
                    tok, pending = pending[0], None
                    try:
                        backend.complete(tok)
                    except Exception:
                        pass  # the run already failed; this only waits for the copy
                    finally:
                        del tok
            finally:
                # an upload issued ahead for a block that will not be searched
                # (an error): wait for its copy before its slot can be released
                # -- also when waiting for the pending copy was interrupted
                drain = getattr(backend, "drain", None)
                if drain is not None:
                    drain()
        finally:
            ahead = None
            # a consumer-side error must shut the prefetch pool down now, not
            # whenever the traceback (which pins this frame's `it`) is freed
            close = getattr(it, "close", None)
            if close is not None:
                close()
    if dm:
        # the merge bucket: the one copy back of the merged top-k (and the cosine turn below)
        t0 = now()
        D_out, I_out = backend.pass_results(nq)
    else:
        t0 = now()
        heap.finalize()
        D_out, I_out = heap.D, heap.I
    if dist:
        # cos = 1 - d for the real columns; -1 columns get FAISS's lowest
        # float, the score every engine's empty slots carry
        D_out = np.where(I_out >= 0, 1 - D_out, np.finfo(np.float32).min).astype(np.float32)
    t_end = now()
    if clock is not None:
        clock.stop()
    t_end_unix = time.time()
    merge += t_end - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    rx1 = net_rx_bytes()
    alloc_peak = backend.alloc_peak()
    total = t_end - t_start
    st = source.stats.as_dict()
    bst = backend.stats.as_dict()
    # the H2D part inside engine_call is the exposed part: the whole copy
    # unless --h2d-overlap ran it during the previous search
    exposed = bst.get("h2d_exposed", bst.get("h2d", 0.0))
    dev = exposed + sum(bst.get(c, 0.0) for c in ("widen", "normalize", "search", "d2h"))
    # device spans only: a host engine (MaskedBackend) times "search" on the host
    busy = sum(bst.get(c, 0.0) for c in ("widen", "normalize", "search")) \
        if "search" in bst and isinstance(backend, base.ResidentBackend) and backend.events \
        else None
    if busy is not None and busy > 1.01 * total + 1e-3:
        # the compute stream's spans are sequential inside the pass: more
        # busy time than wall time means an event is on the wrong stream
        warnings.warn(f"run_pass: compute-stream busy {busy:.3f}s exceeds the pass's "
                      f"{total:.3f}s: device spans mis-measured (gpu_idle_s is negative)")
    net = rx1 - rx0 if rx0 is not None and rx1 is not None else None
    procs = getattr(source, "reader", None) == "process"
    w_user, w_sys = st.get("worker_cpu_user", 0.0), st.get("worker_cpu_sys", 0.0)
    info = {
        "total_s": total,
        "source_stall_s": stall,
        "engine_call_s": call,
        "merge_s": merge,
        "other_s": total - stall - call - merge,
        # annotations inside engine_call (device events, GPU only), or
        # worker-side (host widen)
        "h2d_s": bst.get("h2d", 0.0),
        # the part of the copies engine_call waited for (= h2d_s unless
        # --h2d-overlap), and the blocks whose copy was issued ahead
        "h2d_exposed_s": exposed,
        "h2d_overlapped": int(bst.get("h2d_overlapped", 0)),
        "widen_s": bst.get("widen", 0.0) + st.get("widen", 0.0),
        # FAISS cosine's per-block L2 normalise, inside engine_call (CUDA
        # events on the resident path, perf_counter on the host paths);
        # empty where not measured: ip, cuVS (native cosine), events off
        "normalize_s": bst["normalize"] if "normalize" in bst else "",
        "search_s": bst.get("search", 0.0),
        # the harness's id step inside search_s (global ids; a filtered block's
        # kept-row remap and query expansion), on the device for every resident
        # engine; CUDA events, blank without them
        "ids_s": bst.get("ids", 0.0),
        # the engine library's own calls inside search_s (_lib spans), and the
        # rest of search_s: the harness's work there (ids_s, filter_bits_s,
        # result extraction / padding, CSR views, ...). Blank where no library
        # span was recorded (events off, host backends)
        "lib_s": bst["lib"] if "lib" in bst else "",
        "search_harness_s": bst.get("search", 0.0) - bst["lib"] if "lib" in bst else "",
        "d2h_s": bst.get("d2h", 0.0),
        # cuVS's own recipe (--cuvs-merge device): the running top-k's merge on the GPU
        # (concatenate + argsort + take, CalcTruthMerge), CUDA events, a sub-span of
        # search_s; blank for every other engine / mode and without events
        "device_merge_s": bst["device_merge"] if "device_merge" in bst else "",
        # cuVS index build, host wall, inside engine_call (its device part,
        # if any, is in search_s); 0 for FAISS. Not an event column.
        "build_s": bst.get("build", 0.0),
        # resident backends: host wall of each block's engine step (submit's
        # _block_search), summed; compare with search_s. Host wall, so also
        # measured with --no-gpu-events; blank on the host backends
        "engine_host_s": bst["engine_host"] if "engine_host" in bst else "",
        # --backend spare, its batched / nosync modes: stage spans inside
        # search_s (CUDA events between the harness's own calls; nosync: a
        # 1-in-N query sample scaled up); blank where not measured
        **{f"spare_{n}_s": bst[f"spare_{n}"] if f"spare_{n}" in bst else ""
           for n in ("plan", "score", "topk")},
        # unattributed part of engine_call (launch / Python / host gaps); may
        # dip below 0 by event granularity. Blank under --d2h-overlap: device
        # spans then overlap the merge and are not inside engine_call
        "engine_gap_s": "" if pipe else (call - dev if "search" in bst else 0.0),
        # wall time in which the compute stream had none of the engine's work:
        # total_s minus its busy spans (widen + normalize + search; one stream
        # runs them in order, so their sum is their union). It is what the
        # host side costs the GPU in any mode. Host gaps inside the engine's
        # own call stay in search_s. Blank where nothing is timed on a device
        "gpu_idle_s": total - busy if busy is not None else "",
        # guarded end-of-call wait on e4 (inside engine_call; expected ~0)
        "sync_wait_s": bst.get("sync_wait", 0.0),
        "sync_waits": int(bst.get("sync_waits", 0)),
        # --backend pylate: PyLate calls over the pass (rerank: one per
        # candidate range; scores-batched: colbert_scores calls); blank elsewhere
        "pylate_calls": int(bst["pylate_calls"]) if "pylate_calls" in bst else "",
        "pylate_calls_max_block": int(bst["pylate_calls_max_block"])
                                  if "pylate_calls_max_block" in bst else "",
        # torch caching-allocator peak over the pass ("" off CUDA, n/a without torch)
        "gpu_alloc_peak_bytes": alloc_peak,
        "fetch_s": st.get("fetch", 0.0),
        "decode_s": st.get("decode", 0.0),
        "read_table_s": st.get("read_table", 0.0),
        "convert_s": st.get("convert", 0.0),
        # --sparse-project query-vocab: nova-bf's projection, reader side, a
        # sub-span of convert_s (not added to it); blank when not projected
        "project_s": st["project"] if "project" in st else "",
        # --read-mode native: time in rust_fetch calls (dense: GETs + decode,
        # also in read_table_s; sparse: the download, also in fetch_s), and
        # files the native reader refused (always 0: a refusal fails the run)
        "native_read_s": st["native_read"] if "native_read" in st else "",
        "native_fallbacks": int(st["native_fallbacks"]) if "native_fallbacks" in st else "",
        # process readers: copy into the shared block (reader wall) / attach it
        # in the consumer (inside source_stall); empty for the inline reader
        "publish_s": st.get("publish", 0.0) if procs else "",
        "handoff_s": st.get("handoff", 0.0) if procs else "",
        "bytes": int(st.get("bytes", 0)),
        "net_rx_bytes": "" if net is None else net,
        "net_rx_MBps": "" if net is None else (net / total / 1e6 if total else 0.0),
        "row_groups": int(st.get("row_groups", 0)),
        "row_groups_per_file": int(st.get("max_row_groups", 0)),
        "rows": rows,
        "rows_per_file": max_rows,
        "nnz": "" if nnz is None else nnz,  # sparse only (blank for dense): as stored
        "nnz_kept": "" if nnz_kept is None else nnz_kept,  # searched (--sparse-project)
        "tokens": "" if tokens is None else tokens,  # multi-vector only
        "max_block_bytes": max_bytes,
        # this process (all threads) plus the readers' own getrusage deltas
        # (RUSAGE_CHILDREN would count only reaped children)
        "cpu_user_s": ru1.ru_utime - ru0.ru_utime + w_user,
        "cpu_sys_s": ru1.ru_stime - ru0.ru_stime + w_sys,
        "worker_cpu_user_s": w_user if procs else "",
        "worker_cpu_sys_s": w_sys if procs else "",
        "t_start_unix": t_start_unix,
        "t_end_unix": t_end_unix,
        "trace": (t_start, files),
    }
    if filt is not None:
        fst = filt.stats.as_dict()
        info.update({
            "filter_s": st.get("filter", 0.0), "filter_eval_s": st.get("filter_eval", 0.0),
            "filter_dedup_s": st.get("filter_dedup", 0.0),
            # gather / union: the readers' compaction, inside filter_s; 0 when
            # nothing is compacted
            "filter_compact_s": st.get("filter_compact", 0.0),
            **{f"{k}_s": st.get(k, 0.0) for k in ("filter_prep", "filter_sql", "filter_text",
                                                  "filter_combine", "filter_sql_query",
                                                  "filter_sql_post", "filter_text_query",
                                                  "filter_text_post")},
            "filter_overflow_bytes": int(st.get("filter_overflow_bytes", 0)),
            "filter_publish_s": st.get("filter_publish", 0.0),
            "payload_read_s": st.get("payload_read", 0.0),
            "filter_mask_overflows": int(st.get("filter_mask_overflows", 0)),
            "filter_handoff_s": fst.get("filter_handoff", 0.0),
            **{c: int(fst.get(c, 0)) for c in ("filter_blocks", "filter_rows_kept",
                                               "filter_allowed_pairs",
                                               "filter_live_queries", "filter_skipped_files",
                                               "filter_distinct_masks",
                                               "filter_searched_queries")},
        })
        # the tokenisation the match_text columns used, over the pass's files
        tc, tl = int(st.get("filter_text_cols", 0)), int(st.get("filter_text_lean_cols", 0))
        info["filter_text_tokens"] = "" if not tc else \
            "lean" if tl == tc else "full" if not tl else "mixed"
        # readers' DuckFilter start-up (before the passes: not in total_s)
        info["filter_setup_s"] = results_io._blank(getattr(source, "filter_setup_s", None))
        # cuVS bitmap mode with device events: mask upload + expansion, a
        # sub-span of search_s; blank otherwise
        info["filter_bits_s"] = bst["filter_bits"] if "filter_bits" in bst else ""
        compacting = getattr(filt, "compact", "none") != "none"
        for c in ("filter_compact_rows", "filter_compact_floor_files", "filter_compact_floor_rows",
                  "filter_compact_whole_files"):
            info[c] = int(fst.get(c, 0)) if compacting else ""
        spec = getattr(source, "filter_compact", None)
        info["filter_compact_max_frac"] = spec.max_frac \
            if compacting and spec is not None and spec.mode == "union" else ""
        # masks a ring slot holds: "auto" (the run's upper bound: never
        # overflows); process readers only
        info["filter_mask_slots"] = "auto" if getattr(source, "reader", None) == "process" else ""
        # cuVS filtered calls with k_eff <= 256: raft's select_k uses warpsort there, where
        # cuVS 26.10 can go wrong for a query with fewer than k allowed rows; blank elsewhere
        info["cuvs_filter_small_k_calls"] = int(bst.get("cuvs_filter_small_k_calls", 0)) \
            if isinstance(backend, cuvs_engine.CuvsBackend) and filt.mode == "bitmap" else ""
        # undefined for the kind: rows kept is a static-filter count, distinct
        # masks a per-query one (blank, not 0)
        info["filter_rows_kept" if filt.kind == "per_query" else "filter_distinct_masks"] = ""
    if not getattr(backend, "events", True):
        # no events (--no-gpu-events, cpu): host widen_s is still measured
        info.update(dict.fromkeys(schema.EVENT_FIELDS, ""))
        if st.get("widen", 0.0):
            info["widen_s"] = st["widen"]
    return D_out, I_out, info


def trace_rows(base, warm, rep, t_start_unix, trace, verified="not_run"):
    """Trace rows for one pass (``info["trace"]``), built after the pass.

    Worker stamps are made relative to the same pass start as the consumer's;
    blocks without worker info leave those columns empty."""
    t_start, files = trace
    rows = []
    for i, (path, n, stall, call, merge, t_got, t_done, w, (wait, waited), build, *h2d) \
            in enumerate(files):
        # (h2d, h2d_exposed, overlapped) of a resident GPU call; empty when
        # not measured (events off, CPU backend)
        h2d = h2d[0] if h2d and h2d[0] is not None else (None, None, None)
        row = {c: base[c] for c in schema.TRACE_CONFIG}
        row.update(t_start_unix=t_start_unix, repeat=rep, warmup=warm,
                   verified=verified, file_index=i,
                   path=path, rows=n, stall_s=stall, engine_call_s=call,
                   build_s=build, merge_s=merge, sync_wait_s="" if wait is None else wait,
                   sync_waited="" if waited is None else int(waited),
                   t_got_s=t_got, t_done_s=t_done,
                   h2d_s=results_io._blank(h2d[0]), h2d_exposed_s=results_io._blank(h2d[1]),
                   h2d_overlapped=results_io._blank(h2d[2]))
        if w:
            row.update(w_start_s=w["w_start"] - t_start, w_end_s=w["w_end"] - t_start,
                       fetch_s=w["fetch"], read_table_s=w["read_table"],
                       convert_s=w["convert"], widen_host_s=w["widen"],
                       publish_s=w.get("publish", ""), handoff_s=w.get("handoff", ""),
                       # process readers: the copy into the ring slot runs from
                       # w_end to here (empty for the inline reader)
                       w_publish_end_s=w["publish_end"] - t_start if "publish_end" in w
                       else "",
                       w_cpu_user_s=w.get("cpu_user", ""), w_cpu_sys_s=w.get("cpu_sys", ""),
                       # native mode: native / native-fetch (empty otherwise)
                       read_path=w.get("read_path", ""),
                       native_read_s=w.get("native_read", ""),
                       payload_read_s=w.get("payload_read", ""),
                       w_filter_start_s=w["filter_start"] - t_start if "filter_start" in w
                       else "",
                       w_filter_end_s=w["filter_end"] - t_start if "filter_end" in w else "")
        else:
            row.update(dict.fromkeys(schema.WORKER_TRACE, ""))
        rows.append(row)
    return rows
