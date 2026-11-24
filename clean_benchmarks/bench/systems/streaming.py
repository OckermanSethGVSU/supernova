"""Code every engine shares: cosine L2 normalisation (NormScratch), the GPU engine base
class (ResidentBackend: CUDA event timing, guarded wait, H2D / D2H copies), page-locking of
the reader's ring slots (HostPinner) and the loader for baselines/ modules (sibling_module).
"""

import contextlib
import importlib
import importlib.util
import os
import sys
import warnings

import faiss  # must import after cudalibs.torch_first() has loaded torch's CUDA libraries
import numpy as np

from bench.record import schema
from bench.timing import Stages, now


# ----------------------------------------------------------------- backends

def l2_normalized(x):
    """A float32 copy of ``x`` with unit-L2 rows (``faiss.normalize_L2``,
    FAISS's own cosine recipe). A zero row stays zeros and scores 0. A copy,
    because a block may be a read-only Arrow buffer or a reader's ring slot."""
    y = np.array(x, dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(y)
    return y


def normalized_queries(xq):
    """``xq`` with unit-L2 rows (l2_normalized, in fp32), back in its own
    dtype: what a FAISS cosine deployment does once, before any search."""
    return l2_normalized(xq).astype(xq.dtype, copy=False)


class NormScratch:
    """Host cosine normalise of a block (CPU backend) into one reusable
    float32 buffer. A fresh copy of a 10 GB block is mostly allocation and
    page faults, which would show up in normalize_s as harness overhead.

    normalized(x) copies the block into a view of the buffer (the block
    itself is never written) and normalises in place there. The buffer grows,
    pre-faulted, only for a block larger than any before: reserve() sizes it
    up front when the largest block is known (under --verify, in setup_s);
    otherwise the first block pays it (the warmup pass, when there is one)."""

    def __init__(self):
        self.buf = np.empty(0, np.float32)
        self.grows = 0  # reallocations so far (tests: none for a same-size block)

    @staticmethod
    def _faulted(n, dtype):
        b = np.empty(n, dtype)
        b.fill(0)  # touch every page now, not inside a later normalize
        return b

    def reserve(self, values):
        """Grow (pre-faulted) to at least ``values`` elements; seconds spent."""
        t0 = now()
        if values > self.buf.size:
            self.buf = self._faulted(values, np.float32)
            self.grows += 1
        return now() - t0

    def normalized(self, x):
        """Unit-L2 rows of ``x`` (a zero row stays zeros, as l2_normalized),
        a C-contiguous view of the scratch, valid until the next call."""
        self.reserve(x.size)
        y = self.buf[:x.size].reshape(x.shape)
        np.copyto(y, x, casting="same_kind")
        faiss.normalize_L2(y)
        return y


class ResidentBackend:
    """Shared GPU execution and timing for resident backends.

    Subclasses implement _engine_search() to return device-side scores
    and indices. The base class handles data transfers, dtype conversion,
    global IDs, and CUDA event timing on a shared stream.

    Synchronization waits only for the final event rather than the entire
    device. Optional block normalization supports FAISS cosine search.
    """

    engine = None
    stream = None  # None = torch's current stream (non-CUDA)
    last_build = 0.0  # host seconds of the last call's index build (cuVS)
    normalize_blocks = False  # FAISS cosine: normalise each block on the device
    # True: the engine returns distances (cuVS cosine: 1 - cos, ascending);
    # run_pass turns them into similarities (1 - d) in its merge bucket
    returns_distance = False

    def __init__(self, k, dtype, device, events):
        self.k, self.dtype = k, np.dtype(dtype)
        self.device = device
        self.events = events
        self.stats = Stages()
        # (seconds, waited) of the last search's guarded wait; None = not measured
        self.last_wait = (0.0, False) if events else (None, None)
        self._cuda = False
        # zero-copy Arrow buffers are read-only; torch only reads them
        warnings.filterwarnings("ignore", message=".*not writable.*")

    def _init_torch(self):
        import torch
        self.torch = torch
        self.tdtype = {np.dtype(np.float32): torch.float32}[self.dtype]
        self._cuda = torch.device(self.device).type == "cuda"

    def _make_stream(self):
        """The explicit non-default torch stream the engine runs on (None off CUDA)."""
        return self.torch.cuda.Stream(device=self.device) if self._cuda else None

    def _on_stream(self):
        """Context that puts every op of a call on ``self.stream``; a no-op off CUDA."""
        if self.stream is None:
            return contextlib.nullcontext()
        return self.torch.cuda.stream(self.stream)

    def _sync(self):
        if self._cuda:
            self.torch.cuda.synchronize()

    def _mark(self):
        """A CUDA event recorded on the current stream (a host timestamp on a
        non-CUDA device, where every op is synchronous anyway)."""
        if not self._cuda:
            return now()
        ev = self.torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def _elapsed(self, a, b):
        return a.elapsed_time(b) / 1e3 if self._cuda else b - a

    # (start, end) event pairs of this block's engine-library calls (_lib)
    _lib_ev = None
    # an engine's own stage spans for this block: {name: ([(start, end), ...], scale)}
    _stage_ev = None
    # True (cuVS, --cuvs-merge device): results merged on the device, one copy back at the
    # end of the pass
    device_merge = False
    # True (cuVS, --cuvs-sync none): no per-file host wait; run_pass keeps one block in flight
    defer_wait = False

    def _add_stage(self, name, a, b):
        """Add one (start, end) event pair to this block's stage span ``name``."""
        if self._stage_ev is None:
            self._stage_ev = {}
        self._stage_ev.setdefault(name, ([], 1.0))[0].append((a, b))

    @contextlib.contextmanager
    def _lib(self):
        """Bracket one call into the engine library with a pair of events.

        Summed per block as lib_s, so search_s - lib_s is the harness's own
        work inside the search span. Nothing with events off."""
        if not self.events or self._lib_ev is None:
            yield
            return
        a = self._mark()
        try:
            yield
        finally:
            self._lib_ev.append((a, self._mark()))

    def _to_dev(self, x):
        # pinned (--h2d pinned: a registered ring slot): async DMA on the
        # current stream; pageable: torch's synchronous staged copy
        return self.torch.from_numpy(x).to(self.device, non_blocking=self._nb)

    def _cast(self, x):
        return x.to(self.tdtype)

    def _owned(self, x_t, xb):
        """True when device tensor ``x_t`` is this call's own memory, so the
        cosine normalise may write it in place. Always true on CUDA (an H2D or
        a cast makes a fresh tensor); on torch's CPU device (tests) ``.to()``
        can return a view of the host block ``xb``, which must never be written."""
        if x_t.device.type != "cpu":
            return True
        p, n = x_t.data_ptr(), x_t.numel() * x_t.element_size()
        lo = xb.ctypes.data
        return p + n <= lo or p >= lo + xb.nbytes

    def _normalize(self, x, inplace):
        """Device rows to unit L2 norm (torch version of faiss.normalize_L2).

        Norm and division run in fp32 whatever ``x``'s dtype, rounded once
        into it. ``inplace`` (see _owned) avoids a second block-sized tensor.
        A zero row stays zeros, as faiss.normalize_L2 leaves it."""
        torch = self.torch
        n = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32)
        n.masked_fill_(n == 0, 1)
        return x.div_(n) if inplace else (x / n).to(x.dtype)

    def _engine_search(self, xq_t, xb_t):
        raise NotImplementedError

    def reserve(self, values):
        """Host scratch pre-sizing (CpuBackend cosine); none here: 0 seconds."""
        return 0.0

    def _pad(self, D, I):
        """Pad to k columns as FAISS does for a short block (lowest float32 / -1).

        A distance engine (returns_distance) pads with the highest float32,
        its worst score; run_pass turns -1 slots back into the lowest float
        after the merge."""
        short = self.k - D.shape[1]
        if short <= 0:
            return D, I
        fill = np.finfo(np.float32).max if self.returns_distance else np.finfo(np.float32).min
        return (np.pad(D, ((0, 0), (0, short)), constant_values=fill),
                np.pad(I, ((0, 0), (0, short)), constant_values=-1))

    def alloc_peak_reset(self):
        """Reset torch's caching-allocator peak (a host-side counter, no device
        sync); called just before the pass's timed span."""
        if self._cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def alloc_peak(self):
        """torch's caching-allocator peak since alloc_peak_reset (xq, blocks,
        the widened copy, D/I); NOT FAISS's temp memory nor cuVS's RMM
        workspace, which bypass torch. Empty (not measured) off CUDA."""
        if not self._cuda:
            return ""
        return self.torch.cuda.max_memory_allocated(self.device)

    def setup(self, xq):
        t0 = now()
        with self._on_stream():
            self.xq = self._to_dev(xq)
        self._sync()
        return now() - t0

    # ------------------------------------------------------------- --d2h
    d2h_mode = "pageable"
    _d2h_bufs = None   # --d2h pinned: {(D dtype, I dtype): [(D, I), (D, I)] flat}
    _d2h_flip = 0
    d2h_overlap = False  # --d2h-overlap: copy back on its own stream, merged one block late
    d2h_stream = None

    def configure_d2h(self, mode="pageable", overlap=False):
        """Set how D, I come back to the host.

        pageable (default): ``.cpu()`` into a fresh tensor per call. pinned:
        an async copy into one of two reusable page-locked buffers, alternating
        per call; the call still waits for it, so D2H stays inside engine_call.
        The returned arrays are views valid until the second-next call.
        ``overlap`` (--d2h-overlap): the copy runs on its own stream and
        run_pass merges block i after submitting block i+1."""
        if mode not in schema.D2H_MODES:
            raise ValueError(mode)
        if overlap and mode != "pinned":
            raise ValueError("--d2h-overlap needs --d2h pinned")
        self.d2h_mode = mode
        self._d2h_bufs = {} if mode == "pinned" else None
        # --d2h-overlap: two pinned buffers alternate, block i's read by the
        # merge while block i+1's is written. No stream off CUDA (all synchronous).
        self.d2h_overlap = bool(overlap)
        self.d2h_stream = self.torch.cuda.Stream(device=self.device) \
            if overlap and self._cuda else None

    def _to_host(self, D, I):
        """D, I device tensors -> host tensors (see configure_d2h)."""
        if self._d2h_bufs is None:
            return D.cpu(), I.cpu()
        # one flat pinned pair per dtype pair, sized nq x k; a narrower result
        # uses a prefix view, so short shapes never add page-locked memory
        # beyond what d2h_pinned_bytes charges. Regrown only if a result
        # exceeds nq x k (not expected).
        key = (D.dtype, I.dtype)
        pair = self._d2h_bufs.get(key)
        if pair is None or pair[0][0].numel() < D.numel() or pair[0][1].numel() < I.numel():
            pin = bool(self._cuda)  # torch's CPU device (tests) cannot page-lock
            full = D.shape[0] * self.k if D.dim() == 2 else 0

            def mk(t):
                return self.torch.empty(max(t.numel(), full), dtype=t.dtype,
                                        pin_memory=pin)
            pair = self._d2h_bufs[key] = [(mk(D), mk(I)), (mk(D), mk(I))]
        Df, If = pair[self._d2h_flip]
        self._d2h_flip ^= 1
        Dh = Df[:D.numel()].view(D.shape)
        Ih = If[:I.numel()].view(I.shape)
        Dh.copy_(D, non_blocking=bool(self._cuda))
        Ih.copy_(I, non_blocking=bool(self._cuda))
        return Dh, Ih

    # ------------------------------------------------- --h2d / --h2d-overlap
    def configure_h2d(self, mode="pageable", overlap=False):
        """Set how a block reaches the GPU.

        pageable (default): torch's synchronous copy. pinned: the block is a
        ring slot page-locked by HostPinner, so the H2D is an async DMA.
        ``overlap`` (--h2d-overlap, pinned only): block i+1 is uploaded on a
        copy stream while block i is searched (see _search_upload)."""
        if mode not in schema.H2D_MODES:
            raise ValueError(mode)
        if overlap and mode != "pinned":
            raise ValueError("--h2d-overlap needs --h2d pinned")
        self.h2d_mode, self.overlap = mode, bool(overlap)
        self._nb = mode == "pinned"
        self.copy_stream = self.torch.cuda.Stream(device=self.device) \
            if overlap and self._cuda else None
        self._ahead = None

    h2d_mode = "pageable"
    overlap = False
    copy_stream = None
    _nb = False  # non_blocking H2D (pinned source)
    _ahead = None  # overlap: (host block, device tensor, c0, c1) of block i+1
    # (h2d, h2d_exposed, overlapped) of the last call, for the trace; None =
    # not measured (events off)
    last_h2d = (None, None, None)

    def _on_copy_stream(self):
        if self.copy_stream is None:
            return contextlib.nullcontext()
        return self.torch.cuda.stream(self.copy_stream)

    def _upload(self, x):
        """Overlap mode: issue ``x``'s H2D on the copy stream, ``c0 [H2D] c1``.

        c0 only with events on; c1 always (the compute stream waits on it).
        Returns at once (async from pinned memory). The device tensor is
        handed to the compute stream with record_stream (_await), so it is
        never reused while either stream uses it. Returns (x, x_t, c0, c1);
        the host block ``x`` is held until its search has returned."""
        with self._on_copy_stream():
            c0 = self._mark() if self.events else None
            x_t = self._to_dev(x)
            self._keep_upload(x)  # a filtered block's kept rows ride with it
            c1 = self._mark() if self.events else self._order_event()
        return (x, x_t, c0, c1)

    def _order_event(self):
        """An ordering-only CUDA event (no timing) on the current stream; None
        on a non-CUDA device (every op is synchronous there)."""
        if not self._cuda:
            return None
        ev = self.torch.cuda.Event()
        ev.record()
        return ev

    def _await(self, up):
        """Make the compute stream (not the host) wait for upload ``up``; returns its tensor."""
        _, x_t, _, c1 = up
        if self._cuda:
            self.stream.wait_event(c1)
            x_t.record_stream(self.stream)
        return x_t

    def _search_upload(self, xb, ahead):
        """Overlap mode, at the start of search(xb): get ``xb``'s upload.

        It is the one issued ahead during the previous search, or one issued
        now on demand. If ``ahead`` (block i+1, already read) is given, its
        upload is issued before this block's engine call so it runs during
        it. Returns (upload, issued_ahead)."""
        up, self._ahead = self._ahead, None
        issued = up is not None
        if issued and up[0] is not xb:
            raise RuntimeError("--h2d-overlap: the block uploaded ahead is not the block "
                               "being searched")
        if up is None:
            up = self._upload(xb)  # on demand (copy stream FIFO: before i+1's)
        if ahead is not None:
            self._ahead = self._upload(ahead)
        return up, issued

    def drain(self):
        """Drop an upload issued ahead that will never be searched (a pass
        ended early), after its copy is done, so its ring slot is never reused
        or unpinned under a copy still reading it."""
        up, self._ahead = self._ahead, None
        self._keep_dev = None  # an upload ahead that will never be searched
        if up is None:
            return
        if self._cuda and up[3] is not None:
            up[3].synchronize()  # c1: that copy (and everything before it) done
        del up

    # run_pass hands the block's global row offset to search() and gets GLOBAL
    # ids back (see _globalize); backends without it get the offset added on
    # the host by run_pass instead
    applies_offset = True

    def _globalize(self, I, offset):
        """File-local row ids to global ids (``offset + row``) on the device.

        One elementwise kernel (~1 ms for 25k x 1000 on an A10G) before the
        copy back, so it is inside engine_call and the merge bucket holds the
        merge alone (on the host it costs ~40-50 ms per block). -1 stays -1."""
        if not offset:
            return I
        return self.torch.where(I >= 0, I + offset, I)

    # True: the engine returns results already on the host (CPU tensors):
    # submit() globalises the ids there and makes no copy back (d2h 0).
    # Only MvPylateBackend's rerank modes with --pylate-rerank-results host
    host_results = False

    def _globalize_host(self, I, offset):
        """_globalize for host results: ``offset`` added in place (no nq x k
        temporary); -1 stays -1."""
        if offset:
            a = I.numpy()
            np.add(a, offset, out=a, where=a >= 0)
        return I

    def search(self, xb, ahead=None, offset=0, post=None, ahead_keep=None):
        """Search one block; = complete(submit(...)).

        ``ahead`` (--h2d-overlap only): the next block, already read, whose
        upload is issued before this engine call. ``offset``: the block's
        global row offset (returned ids are global). ``post`` /
        ``ahead_keep``: a filtered block's result shaping, done on the device."""
        return self.complete(self.submit(xb, ahead, offset, post, ahead_keep))

    # ---------------------------------------------- filtered: shaping on device
    # id(host block) -> (host block, kept rows): to upload with that block
    _keep_pending = None
    # id(host block) -> (host block, device kept rows), until it is searched
    _keep_dev = None

    def _keep_upload(self, x):
        """Upload block ``x``'s kept file rows (gather / union), if any, with the
        block itself (same stream, same h2d span)."""
        got = None if self._keep_pending is None else self._keep_pending.pop(id(x), None)
        if got is None or got[0] is not x:
            return
        if self._keep_dev is None:
            self._keep_dev = {}
        self._keep_dev[id(x)] = (x, self._upload_host(got[1]))

    def _device_post(self, xb, D, I, offset, post):
        """Shape a filtered block's results on the device, bit for bit as
        FilterAdapter.finish does on the host.

        Compacted ids map back to file rows (``post.keep``), then to global
        ids. With a searched subset (``post.queries``) the rows are padded to
        k and expanded to all ``post.nq`` queries, the others "no result"
        (+-inf, -1). Inside search_s."""
        torch = self.torch
        if post.keep is not None:
            got = None if self._keep_dev is None else self._keep_dev.pop(id(xb), None)
            if got is not None and got[0] is xb:
                keep_t = got[1]
                if self._cuda and self.copy_stream is not None:
                    keep_t.record_stream(self.stream)  # this stream waited on its c1
            else:
                keep_t = self._upload_host(post.keep)  # on demand, this stream
            ok = I >= 0
            if keep_t.numel():
                I = torch.where(ok, keep_t[torch.where(ok, I, torch.zeros_like(I))],
                                torch.full_like(I, -1))
            else:
                I = torch.full_like(I, -1)
        I = self._globalize(I, offset)
        if post.queries is None:
            return D, I
        f32 = np.finfo(np.float32)
        rows, k_eff = D.shape
        if k_eff < self.k:
            # _pad's fill for a block of fewer than k rows, before the expansion
            fill = float(f32.max if self.returns_distance else f32.min)
            Dk = torch.full((rows, self.k), fill, dtype=D.dtype, device=D.device)
            Ik = torch.full((rows, self.k), -1, dtype=I.dtype, device=I.device)
            Dk[:, :k_eff], Ik[:, :k_eff] = D, I
            D, I = Dk, Ik
        pad = float("inf") if self.returns_distance else float("-inf")
        Dn = torch.full((post.nq, self.k), pad, dtype=D.dtype, device=D.device)
        In = torch.full((post.nq, self.k), -1, dtype=I.dtype, device=I.device)
        q = self._upload_small(np.asarray(post.queries, np.int64))
        Dn.index_copy_(0, q, D)
        In.index_copy_(0, q, I)
        return Dn, In

    def _upload_small(self, a):
        """Copy a small host array to the device without blocking the host.

        Staged through torch's caching pinned allocator, then copied
        non-blocking, so the host goes on to enqueue the search. Dropping the
        staging tensor at once is safe: the allocator does not reuse it until
        the copy has finished. Off CUDA: a plain copy."""
        torch = self.torch
        src = torch.from_numpy(np.ascontiguousarray(a))
        if not getattr(self, "_cuda", False):
            return src.to(self.device)
        staged = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        staged.copy_(src)
        return staged.to(self.device, non_blocking=True)

    def _upload_host(self, a):
        """Copy a block's side array (masks, kept rows) to the device.

        From a page-locked ring slot (--h2d pinned) it is an async DMA straight
        from the slot, which stays mapped until the block's complete(). Any
        other array goes through pinned staging (_upload_small)."""
        torch = self.torch
        if self._cuda and self._nb:
            src = torch.from_numpy(np.ascontiguousarray(a))
            if src.is_pinned():
                return src.to(self.device, non_blocking=True)
        return self._upload_small(a)

    def submit(self, xb, ahead=None, offset=0, post=None, ahead_keep=None):
        """Phase 1 of a block: enqueue everything, with no host wait.

        Upload (or wait for the one issued ahead), cast / normalise, engine
        call, global ids, and the copy back of D / I. With --d2h-overlap the
        copy back runs on its own stream so the next search does not queue
        behind it. Returns a token for complete(); it keeps the host block and
        the events, while the device block is released here."""
        up = issued = None
        self._lib_ev = []  # this block's library spans (_lib)
        if post is not None or ahead_keep is not None:
            if self._keep_pending is None:
                self._keep_pending = {}
            if post is not None and post.keep is not None:
                self._keep_pending[id(xb)] = (xb, post.keep)
            if ahead is not None and ahead_keep is not None:
                self._keep_pending[id(ahead)] = (ahead, ahead_keep)
        if self.overlap:
            up, issued = self._search_upload(xb, ahead)
        elif ahead is not None:
            raise ValueError("ahead= needs --h2d-overlap")
        ev = self.events
        with self._on_stream():
            e0 = self._mark() if ev else None
            # e0..e1 is the copy itself (pageable / pinned), or under overlap
            # only this stream's wait for the copy stream (h2d_exposed).
            # The device block is handed over in a one-element list that
            # _block_search pops, so no reference stays here and a cast frees
            # the upload before the engine call (else both would coexist).
            # Under --h2d-overlap ``up`` holds the upload until the end of
            # submit (the GPU memory guard counts it)
            if up is None:
                xb_h = [self._to_dev(xb)]
                self._keep_upload(xb)  # inside e0..e1 too
            else:
                xb_h = [self._await(up)]
            e1 = self._mark() if ev else None
            th = now()
            D, I, cast, nrm, e2n = self._block_search(xb, xb_h, e1)
            # host wall the block's engine step held this thread (cast /
            # normalise enqueue, the engine call, its host-side prep): ~search_s
            # for an engine that blocks the host until its GPU work is done,
            # well below it for one that only enqueues (the host can run ahead)
            self.stats.add("engine_host", now() - th)
            e2 = cast[1]  # the end of the cast (e1 when there was none)
            build = self.last_build
            build_ev, self._build_ev = getattr(self, "_build_ev", None), None
            stage_ev, self._stage_ev = getattr(self, "_stage_ev", None), None
            host = self.host_results
            # the id step (global ids; a filtered block's shaping too) is the
            # harness's: ids = eg..e3, a sub-span of search_s
            eg = self._mark() if ev else None
            dm = self.device_merge
            moff = 0
            if post is not None:
                # a filtered block: ids -> file rows -> global, all nq rows
                D, I = self._device_post(xb, D, I, offset, post)
            elif dm:
                moff = offset  # calc_truth offsets the ids itself (offset_neighbor_indices)
            else:
                I = self._globalize_host(I, offset) if host else self._globalize(I, offset)
            em = None
            if dm:
                # cuVS's own recipe: the block's (D, I) into the running top-k on the
                # device (CalcTruthMerge); no per-block copy back
                em = self._mark() if ev else None
                self._merge_add(D, I, moff)
                D = I = None
            e3 = self._mark() if ev else None
            if em is not None:
                # this block's stage spans were taken above: add the merge to them
                stage_ev = dict(stage_ev or {})
                stage_ev.setdefault("device_merge", ([], 1.0))[0].append((em, e3))
            if D is None:
                # device merge: nothing to copy back for this block (d2h = e3..e3 = 0;
                # events off: e4 None, complete() syncs; --cuvs-sync none: an
                # ordering-only event, the one thing complete() waits on)
                e4 = e3 if ev or not self.defer_wait else self._order_event()
            elif host:
                # already on the host: no copy back, no pinned buffer, no d2h
                # stream (d2h = e3..e3 = 0; events off: e4 None, nothing to wait on)
                e4 = e3
            elif self.d2h_stream is not None:
                # the copy back on its own stream, after everything above
                s = self.d2h_stream
                s.wait_stream(self.stream)
                with self.torch.cuda.stream(s):
                    D.record_stream(s)  # D / I are not reused before the copy is done
                    I.record_stream(s)
                    D, I = self._to_host(D, I)
                    # events off: an ordering-only event, the one thing to wait on
                    e4 = self._mark() if ev else self._order_event()
            else:
                D, I = self._to_host(D, I)
                # events off: no Event object at all (complete() syncs)
                e4 = self._mark() if ev else None
            del xb_h
        lib_ev, self._lib_ev = self._lib_ev, None  # spans only inside a block's submit
        if self._keep_pending:
            self._keep_pending.clear()  # uploaded with their blocks, or not needed
        if up is not None:
            up = (up[0], None, up[2], up[3])  # keep the host block + events, not the device block
        return {"xb": xb, "up": up, "issued": issued, "D": D, "I": I, "build": build,
                "build_ev": build_ev, "stage_ev": stage_ev, "ids_ev": eg, "ids_end": em,
                "lib_ev": lib_ev,
                "e": (e0, e1, e2, e2n, e3, e4), "cast": cast, "nrm": nrm}

    def _block_search(self, xb, xb_h, e1):
        """The block-specific middle of submit(): cast / normalise, then search.

        ``xb_h`` is a one-element list holding the device copy of ``xb``; it
        is popped here so a cast frees the upload before the engine call.
        Returns ``(D, I, cast, nrm, e2n)``: device results (I int64,
        file-local ids), the event pairs summed as widen / normalize, and the
        event the search started behind. Sparse engines override this."""
        ev = self.events
        xb_t = xb_h.pop()
        e2 = e1  # no cast: e2 is e1
        if xb_t.dtype != self.tdtype:
            xb_t = self._cast(xb_t)
            if ev:
                e2 = self._mark()
        cast = (e1, e2)
        e2n = e2  # no normalize step (ip, cuVS): the chain is unchanged
        if self.normalize_blocks:
            xb_t = self._normalize(xb_t, self._owned(xb_t, xb))
            if ev:
                e2n = self._mark()
        nrm = (e2, e2n)
        D, I = self._engine_search(self.xq, xb_t)
        return D, I, cast, nrm, e2n

    def complete(self, tok):
        """Phase 2: wait for this token's copy back, record its timings, return (D, I).

        The wait is on an event, not a device or stream sync, so a later
        block's work keeps running. Arrays are numpy (views of a pinned buffer
        with --d2h pinned, valid until the second-next submit). Sets
        last_wait / last_h2d / last_build for the trace."""
        e0, e1, e2, e2n, e3, e4 = tok["e"]
        up, issued = tok["up"], tok["issued"]
        self.last_build = tok["build"]
        wait, waited = 0.0, False
        D, I = tok["D"], tok["I"]
        if not self.events:
            if self.d2h_stream is not None or self.defer_wait:
                if e4 is not None:
                    e4.synchronize()  # this block's copy back (or merge) only
            elif self.overlap and self._cuda:
                # this stream only: a device-wide sync would also wait for
                # block i+1's copy
                self.stream.synchronize()
            else:
                self._sync()
            self.last_wait = (None, None)
            self.last_h2d = (None, None, None)
            if D is None:
                return None, None  # merged on the device (cuVS's own recipe)
            return self._pad(D.numpy(), I.numpy())
        if self._cuda and e4 is not None and not e4.query():
            t = now()
            e4.synchronize()
            wait, waited = now() - t, True
        self.last_wait = (wait, waited)
        self.stats.add("sync_wait", wait)
        self.stats.add("sync_waits", int(waited))
        el = self._elapsed
        cast, nrm = tok["cast"], tok["nrm"]
        if up is None:
            h2d = exposed = el(e0, e1)  # the whole copy, on this stream
        else:
            # h2d: the copy stream's own pair; exposed: this stream's wait.
            # Off CUDA the copy ran on the host before e0, so all of it is
            # exposed unless it was issued ahead
            h2d = el(up[2], up[3])
            exposed = el(e0, e1) if self._cuda or issued else h2d
        self.stats.add("h2d", h2d)
        self.stats.add("h2d_exposed", exposed)
        self.stats.add("h2d_overlapped", int(bool(issued)))
        self.last_h2d = (h2d, exposed, int(bool(issued)))
        self.stats.add("widen", el(*cast) if cast[1] is not cast[0] else 0.0)
        if self.normalize_blocks:
            self.stats.add("normalize", el(*nrm))
        self.stats.add("search", el(e2n, e3))
        if tok.get("ids_ev") is not None:
            # the id step ends where a device merge starts (its own span, device_merge)
            self.stats.add("ids", el(tok["ids_ev"], tok.get("ids_end") or e3))
        lib = tok.get("lib_ev")
        if lib:
            self.stats.add("lib", sum(el(a, b) for a, b in lib))
        self.stats.add("d2h", el(e3, e4))
        if tok.get("build_ev") is not None:
            # an engine that timed its build with device events (SPARe): a
            # sub-span of search, read now that e4 (behind it) is complete
            self.last_build = el(*tok["build_ev"])
            self.stats.add("build", self.last_build)
        for name, (pairs, scale) in (tok.get("stage_ev") or {}).items():
            # an engine's own stage spans (SPARe: plan / score / topk), each a
            # sub-span of search, read now that e4 (behind them) is complete
            self.stats.add(name, scale * sum(el(a, b) for a, b in pairs))
        if D is None:
            return None, None  # merged on the device (cuVS's own recipe)
        return self._pad(D.numpy(), I.numpy())


class HostPinner:
    """--h2d pinned: page-locks the reader ring's slots.

    sources._BlockRing calls ``register`` once per slot right after creating
    it (in the consumer process) and ``unregister`` before any slot can be
    unmapped. ``cudart`` is torch.cuda.cudart(); None off CUDA (tests: calls
    are only counted). ``sync`` runs before unregistering, so no copy is
    still reading a slot."""

    def __init__(self, cudart=None, sync=None):
        self.cudart, self.sync = cudart, sync
        self.registered = {}  # address -> bytes, currently registered
        self.calls = []       # ("register" | "unregister", address): tests

    def register(self, addr, nbytes):
        self.calls.append(("register", addr))
        if addr in self.registered:
            raise RuntimeError(f"slot at {addr:#x} registered twice")
        if self.cudart is not None:
            rc = self.cudart.cudaHostRegister(addr, nbytes, 0)
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostRegister({addr:#x}, {nbytes}) failed: {rc} "
                                   "(--h2d pinned needs the ring slots page-locked; "
                                   "use --h2d pageable)")
        self.registered[addr] = nbytes

    def unregister(self, addrs):
        """Unregister ``addrs``; a failure only warns (cleanup path: it must not
        mask the error that got there)."""
        try:
            if self.sync is not None and self.cudart is not None:
                self.sync()
        except Exception as e:  # noqa: BLE001 - cleanup path
            print(f"WARNING: --h2d pinned: device sync before unregistering failed: {e}",
                  file=sys.stderr)
        for addr in addrs:
            self.calls.append(("unregister", addr))
            self.registered.pop(addr, None)
            if self.cudart is None:
                continue
            try:
                rc = self.cudart.cudaHostUnregister(addr)
                if int(rc) != 0:
                    print(f"WARNING: cudaHostUnregister({addr:#x}) failed: {rc}",
                          file=sys.stderr)
            except Exception as e:  # noqa: BLE001 - cleanup path
                print(f"WARNING: cudaHostUnregister({addr:#x}) raised: {e}", file=sys.stderr)


def host_pinner(backend):
    """The HostPinner for ``backend``: real on CUDA, a counting no-op otherwise."""
    if getattr(backend, "_cuda", False):
        torch = backend.torch
        return HostPinner(torch.cuda.cudart(), torch.cuda.synchronize)
    return HostPinner(None)


def csr_views(torch, buf_t, c, o):
    """Device views of a packed sparse block's CSR parts (no copy): indptr
    and indices int32, values float16 / float32 as stored; ``buf_t`` the
    block's bytes on the device, ``c`` its unpacked header, ``o`` the part
    offsets."""
    vb = c.values.dtype.itemsize
    ip = buf_t[o[0]:o[0] + 4 * (c.n_rows + 1)].view(torch.int32)
    ix = buf_t[o[1]:o[1] + 4 * c.nnz].view(torch.int32)
    iv = buf_t[o[2]:o[2] + vb * c.nnz].view(torch.float16 if vb == 2 else torch.float32)
    return ip, ix, iv


def sibling_module(name, attr, folder=None):
    """Import baselines/<folder>/<name>.py (``folder`` defaults to ``name``).

    An importable ``name`` is used only if it has ``attr``: ``import name``
    can silently find a directory of that name as an empty namespace
    package, so then the file is loaded by path."""
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, attr):
            return mod
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # bench/systems/ -> clean_benchmarks/
                        "baselines", folder or name, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or not os.path.exists(path):
        raise ImportError(f"{path} not found")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod
