"""SPARe engine for sparse legs (SparseSpareBackend): SPARe's released per-query modes
and our batched ones, plus its mode, result-column and memory helpers (leg -> engine)."""

import contextlib
import io

import numpy as np

from bench.timing import now
from bench.systems import streaming as base
from bench.systems.base import SPARE_MODES, SPARE_STAGED_MODES, spare_released  # noqa: F401 - the modes live with the knobs
from bench.data.formats import sparse


def spare_batched_module():
    """baselines/spare/spare_batched.py (needs the ``spare`` package), imported
    on first use."""
    return base.sibling_module("spare_batched", "BatchedDot", "spare")


# iterative-released-nosync times one query in this many and scales the sums up;
# the batched modes time every group
SPARE_NOSYNC_EVENT_STRIDE = 16


def spare_events_str(args):
    """The spare_events column: n/a (not SPARe, or a released mode run through
    SPARe's own forward), off (--no-spare-events / --no-gpu-events), on (the
    batched modes: every group timed) or sampled:N (nosync: one query in N)."""
    if args.backend != "spare" or args.spare_mode not in SPARE_STAGED_MODES:
        return "n/a"
    if args.no_gpu_events or getattr(args, "no_spare_events", False):
        return "off"
    return (f"sampled:{SPARE_NOSYNC_EVENT_STRIDE}"
            if args.spare_mode == "iterative-released-nosync" else "on")


SPARE_DEFAULT_MAX_MIB = 8192  # --spare-max-mib's default: cuVS's stock tile, 32768^2 x 8 B
# spare_batched's defaults, recorded in sparse_batches: dot top-k on a contiguous copy,
# int32 expansion indices where they fit
SPARE_TOPK_LAYOUT = "contiguous"
SPARE_INDEX_DTYPE = "auto"


class SparseSpareBackend(base.ResidentBackend):
    """SPARe exact sparse search on a single GPU (Almeida & Matos, ECIR 2024).

    Supports the original implementations and query-batched variants:

    - dot: CSR sparse-dense matrix multiplication (cuSPARSE SpMM).
    - iterative: CSC posting-list accumulation using index_add_.
    - *-released: original per-query implementations.
    - iterative-released-nosync: original implementation with a synchronization fix.

    Batched variants preserve SPARe's scoring and top-k operations while
    reducing per-query overhead. All modes use float32 and inner product.

    Queries are uploaded once, and each block builds its SPARe model.
    Batched execution limits scratch memory using max_bytes.
    """


    applies_offset = True  # ids globalised on the device by submit (_globalize)
    name = "spare"
    engine = "spare"
    returns_distance = False

    def __init__(self, k, mode, device="cuda", events=True, metric="ip",
                 max_bytes=SPARE_DEFAULT_MAX_MIB << 20, module=None, stage_events=True):
        try:
            import torch  # noqa: F401
            sb = module if module is not None else spare_batched_module()
        except ImportError as e:
            raise RuntimeError("--backend spare needs torch and the spare package "
                               "(env/build.sh, baselines/README.md; import "
                               f"failed: {e})") from e
        if mode not in SPARE_MODES:
            raise ValueError(mode)
        if metric != "ip":
            raise ValueError("--backend spare is inner product only (SPARe has no cosine)")
        super().__init__(k, np.float32, device, events)
        self.queries_mode = "resident"
        self.metric, self.mode, self.max_bytes = metric, mode, int(max_bytes)
        self.released = spare_released(mode)
        self._sb = sb
        if mode == "iterative-released-nosync":
            self._model_cls = sb.IterativeReleasedNoSync
        elif self.released:
            from spare.backend_torch import (CSRSparseRetrievalModel,
                                             CSRSparseRetrievalModelIterative)
            self._model_cls = CSRSparseRetrievalModel if mode.startswith("dot") \
                else CSRSparseRetrievalModelIterative
        else:
            self._model_cls = sb.BatchedDot if mode == "dot" else sb.BatchedIterative
        try:
            from importlib.metadata import version
            spare_v = version("spare")
        except Exception:
            spare_v = "unknown"
        self.engine_version = f"{spare_v}+{sb.SPARE_COMMIT}"
        self._init_torch()
        self.stream = self._make_stream()
        self._build_ev = None
        # stage spans (plan / score / topk) for the modes whose loop is ours:
        # device events between our own calls, read in complete()
        self.stage_events = bool(stage_events and events and mode in SPARE_STAGED_MODES)
        self._stage_ev = None

    def setup(self, q):
        """Upload the queries (a canonical sparse.Queries) once: CsrQueries for
        the batched modes, SPARe's per-query (1, terms) tensors for the
        released ones; validated once."""
        t0 = now()
        torch = self.torch
        self.q = q
        sb = self._sb
        with self._on_stream():
            self.qc = sb.CsrQueries.from_csr(q.indptr, q.indices, q.values,
                                             device=self.device)
            if self.released:
                # what SPARe's QuestionDataset + DataLoader(batch_size=1) yield
                # (int64 ids, float32 weights; int32 empties, as its __getitem__),
                # moved to the device as forward_retrieval does
                self.q_list = []
                for r in range(q.n_rows):
                    a, b = int(q.indptr[r]), int(q.indptr[r + 1])
                    if b > a:
                        ind = torch.from_numpy(q.indices[a:b].astype(np.int64))
                        val = torch.from_numpy(q.values[a:b].astype(np.float32))
                    else:
                        ind = torch.tensor([], dtype=torch.int32)
                        val = torch.tensor([], dtype=torch.int32)
                    self.q_list.append((ind.unsqueeze(0).to(self.device),
                                        val.unsqueeze(0).to(self.device)))
                # the nosync loop's host-side term ids
                self.q_terms_h = [q.indices[int(q.indptr[r]):int(q.indptr[r + 1])].tolist()
                                  for r in range(q.n_rows)]
            pin = bool(self._cuda)
            self._post_h = torch.empty(q.n_rows, dtype=torch.int64, pin_memory=pin)
        self._sync()
        sb.check_csr(self.qc, 1, 1, max(q.n_cols, 1), max_bytes=self.max_bytes, max_batch=1)
        return now() - t0

    def _views(self, buf_t, c, o):
        return base.csr_views(self.torch, buf_t, c, o)

    @staticmethod
    def _iterative_base():
        from spare.backend_torch import CSRSparseRetrievalModelIterative
        return CSRSparseRetrievalModelIterative

    def _build(self, coll, k_eff):
        """Build the SPARe model on this block, timing the construction
        (device events, or host seconds with --no-gpu-events)."""
        ev = self.events
        b0 = self._mark() if ev else None
        t = now()
        with contextlib.redirect_stdout(io.StringIO()):  # SPARe prints per conversion
            if issubclass(self._model_cls, self._iterative_base()):
                model = self._model_cls(coll, top_k=k_eff, objective="accuracy")
            else:
                model = self._model_cls(coll, top_k=k_eff)
        host = now() - t
        if ev:
            self._build_ev = (b0, self._mark())
        else:
            self.last_build = host
            self.stats.add("build", host)
        return model

    def _engine_search(self, ip, ix, iv, c):
        torch = self.torch
        sb = self._sb
        nq, k_eff = self.q.n_rows, min(self.k, c.n_rows)
        n_cols = max(self.q.n_cols, c.n_cols, 1)
        coll = sb.CsrCollection((ip, ix, iv), (c.n_rows, n_cols))
        post = None
        sp = sb.Spans(self._mark) if self.stage_events else None
        scale = 1.0
        if self.mode == "iterative":
            p0 = self._mark() if sp is not None else None
            # every query's postings count from the CSR column histogram, queued
            # before the conversion; the host waits only on its event (index_add_
            # of ones, not bincount: CUDA bincount syncs to read the input's max)
            df = torch.zeros(n_cols, dtype=torch.int64, device=ix.device)
            df.index_add_(0, ix, torch.ones(1, dtype=torch.int64,
                                            device=ix.device).expand(ix.numel()))
            dev_post = sb._segment_sums(df[self.qc.terms], self.qc.indptr_dev)
            self._post_h.copy_(dev_post, non_blocking=bool(self._cuda))
            post_ev = self.torch.cuda.Event() if self._cuda else None
            if post_ev is not None:
                post_ev.record()
            del df, dev_post
            if sp is not None:  # the histogram planning, before the conversion
                sp.plan.append((p0, self._mark()))
        model = self._build(coll, k_eff)
        if self.mode == "iterative-released-nosync":
            model.top_k = k_eff
            p0 = self._mark() if sp is not None else None
            model.host_bounds()   # one sync, after the conversion
            if sp is not None:
                sp.plan.append((p0, self._mark()))
            vals, ids = [], []
            stride = SPARE_NOSYNC_EVENT_STRIDE
            for j, (terms, (_, val)) in enumerate(zip(self.q_terms_h, self.q_list)):
                r = model.forward_nosync(terms, val,
                                         sp if sp is not None and j % stride == 0 else None)
                ids.append(r.indices)
                vals.append(r.values)
            if sp is not None and sp.score:
                scale = nq / len(sp.score)  # the timed sample -> every query
            D, I = torch.stack(vals), torch.stack(ids)
        elif self.released:
            model.top_k = k_eff
            vals, ids = [], []
            for ind, val in self.q_list:   # SPARe's forward_retrieval loop body
                r = model(ind, val)
                ids.append(r.indices)
                vals.append(r.values)
            D, I = torch.stack(vals), torch.stack(ids)
        elif self.mode == "dot":
            D, I = model.search_csr(self.qc, k_eff, max_bytes=self.max_bytes,
                                    max_batch=max(nq, 1), validate=False,
                                    topk_layout=SPARE_TOPK_LAYOUT, spans=sp)
        else:
            if post_ev is not None:
                post_ev.synchronize()
            D, I = model.search_csr(self.qc, k_eff, max_bytes=self.max_bytes,
                                    max_batch=max(nq, 1), validate=False,
                                    index_dtype=SPARE_INDEX_DTYPE, postings=self._post_h,
                                    spans=sp)
        if sp is not None:
            self._stage_ev = {"spare_plan": (sp.plan, 1.0), "spare_score": (sp.score, scale),
                              "spare_topk": (sp.topk, scale)}
        del model, coll
        return D, I

    def _block_search(self, xb, xb_h, e1):
        """Device views of the uploaded packed CSR block, fp16 values widened to
        float32, then SPARe model build + search; I is int64 already."""
        torch = self.torch
        xb_t = xb_h.pop()
        c = sparse.unpack(xb)  # header only: sizes and offsets (host)
        o = [int(v) for v in sparse._header(xb)[5:8]]
        ip, ix, iv = self._views(xb_t, c, o)
        e2 = e1
        if iv.dtype != torch.float32:
            iv = iv.to(torch.float32)
            if self.events:
                e2 = self._mark()
        with self._lib():  # SPARe's model build + search
            D, I = self._engine_search(ip, ix, iv, c)
        return D, I, (e1, e2), (e2, e2), e2


# --backend spare iterative: peak device bytes per nonzero of SPARe's CSR to
# CSC conversion beyond the block (~32-44 B transient) plus the resident CSC
# (~8-12 B): charged 56
SPARE_CSC_BYTES_PER_NNZ = 56


def spare_batches_str(args):
    """The sparse_batches column for --backend spare: the batched modes'
    budget and layout / index setting, ``per-query`` for the released modes."""
    if spare_released(args.spare_mode):
        return "per-query"
    knob = (f"topk={SPARE_TOPK_LAYOUT}" if args.spare_mode == "dot"
            else f"index={SPARE_INDEX_DTYPE}")
    return f"max_bytes={args.spare_max_mib}MiB,{knob}"


def spare_engine_bytes(args, feet, nq, pairs=0, q_cols=0):
    """Device bytes of SPARe's per-block work beyond the block, largest file.

    The call's scratch budget (--spare-max-mib; released modes: one query's
    dense vectors instead), the per-pair planning tensors (64 B per (query,
    term) pair) and, iterative, the CSC conversion plus 20 B per query
    column (histogram and column pointers). Not estimated: a single query
    over budget, or a block whose indices run wider than the queries'."""
    nnz = max((g.sparse_nnz or 0) for g in feet)
    if spare_released(args.spare_mode):
        # one query per call, no budget: a dense (vocab,) query + (rows,)
        # scores / accumulator + topk scratch, charged as 3 x rows + vocab
        rows = max((g.num_rows or 0) for g in feet)
        need = (3 * rows + q_cols) * 4 + max(pairs, 1) * 64
    else:
        need = (args.spare_max_mib << 20) + max(pairs, 1) * 64
    if args.spare_mode.startswith("iterative"):
        need += nnz * SPARE_CSC_BYTES_PER_NNZ + (q_cols + 1) * 20
    return need
