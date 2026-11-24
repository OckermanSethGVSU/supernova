"""PyLate engine for multivector legs (MvPylateBackend): exact float32 MaxSim through
PyLate's own calls, plus its memory-planning helpers and the fp32 matmul check
(leg -> engine)."""

import operator
import os

import faiss  # must import after cudalibs.torch_first() has loaded torch's CUDA libraries
import numpy as np

from bench.record import schema
from bench.timing import now
from bench.systems import streaming as base
from bench.systems.base import PYLATE_MODES, pylate_released  # noqa: F401 - the modes live with the knobs
from bench.data.formats import multivector as multivec


# --backend pylate: PyLate (LightOn; Chaffin & Sourty, arXiv 2508.03555)
# pinned at this commit (v1.6.0)
PYLATE_COMMIT = "3389837ce465ee66e46f369a3f821a96788b7f29"
PYLATE_DEFAULT_MAX_MIB = 8192
# --pylate-rerank-results (rerank modes): device (default: results uploaded,
# ids globalised on the device, copied back like every engine's) or host
# (kept on the host, no round trip)
PYLATE_RERANK_RESULTS = ("device", "host")
# the fields of rank.rerank's result entries ({"id": ..., "score": ...})
_RERANK_SCORE = operator.itemgetter("score")
_RERANK_ID = operator.itemgetter("id")


def pylate_module():
    """``(pylate.rank, pylate.scores, version)``, imported on first use."""
    import pylate
    from pylate import rank, scores
    return rank, scores, pylate.__version__


# cuBLAS environment switches that can change float32 GEMM arithmetic below
# torch's view (BF16x9 fp32 emulation on newer GPUs): refused when set at all
CUBLAS_FP32_ENV = ("CUBLAS_EMULATE_SINGLE_PRECISION", "CUBLAS_EMULATION_STRATEGY")


def check_fp32_matmul(torch):
    """Refuse a process whose float32 matmuls would not be IEEE float32.

    PyLate scores with an einsum (cuBLAS), so TF32 / bf16 matmul settings
    would silently change its arithmetic. Also refuses the cuBLAS
    fp32-emulation environment switches (CUBLAS_FP32_ENV), which torch
    cannot see."""
    why = "leave them at torch's defaults (False / 'highest')"
    try:
        prec = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
    except RuntimeError as e:  # the newer fp32_precision API set to a mixed state
        raise RuntimeError(f"--backend pylate is exact float32 only, and this process's "
                           f"float32 matmul precision cannot be read as IEEE ({e}); {why}") \
            from e
    if tf32 or prec != "highest":
        raise RuntimeError(
            "--backend pylate is exact float32 only, but this process's float32 matmuls "
            f"may run reduced precision (torch.backends.cuda.matmul.allow_tf32={tf32}, "
            f"float32_matmul_precision={prec!r}); {why}")
    env = [v for v in CUBLAS_FP32_ENV if os.environ.get(v)]
    if env:
        raise RuntimeError(f"--backend pylate is exact float32 only, but {', '.join(env)} "
                           "is set (cuBLAS float32 emulation); unset it")


def mv_views(torch, buf_t, xb):
    """Device views of a packed multi-vector block's parts (no copy): token
    offsets int64 (n_rows + 1,) and tokens (n_tok, dim) float16 / float32 as
    stored; ``buf_t`` the block's bytes on the device, ``xb`` the host block
    (its header)."""
    n, nt, dim, vb, o_o, o_t = multivec.header_fields(xb)
    off = buf_t[o_o:o_o + 8 * (n + 1)].view(torch.int64)
    tok = buf_t[o_t:o_t + vb * nt * dim].view(torch.float16 if vb == 2 else torch.float32)
    return off, tok.view(nt, dim)


def own_token_index(torch, off_d, len_d, docs, L, n_tok):
    """``(len(docs), L)`` token row ids that pad ``docs`` to ``L`` tokens with
    copies of each document's own last token (row ``t`` = token
    ``min(t, len_i - 1)``).

    Repeating a token leaves a max over tokens unchanged, so the padded batch
    scores exactly a masked MaxSim with no mask (zero padding would win the
    max for a document whose tokens all score below 0). Clamped into the
    block, so a zero-token document (excluded later) indexes a real row. All
    on the device, no sync."""
    t = torch.arange(L, device=off_d.device)
    last = (len_d[docs] - 1).clamp_min(0)
    idx = off_d[:-1][docs][:, None] + torch.minimum(t[None, :], last[:, None])
    return idx.clamp_(0, max(n_tok - 1, 0))


def mv_doc_chunks(lengths, per_token_bytes, per_doc_bytes, budget):
    """Split documents (in ``lengths``' order) into consecutive ranges
    ``[(d0, d1, L)]``, ``L`` the range's longest, so that
    ``(d1 - d0) * (L * per_token_bytes + per_doc_bytes) <= budget``.

    A single document over budget gets a range of its own. Host only."""
    n = len(lengths)
    out, i = [], 0
    while i < n:
        cm = np.maximum.accumulate(lengths[i:])
        cost = np.arange(1, n - i + 1) * (cm * per_token_bytes + per_doc_bytes)
        j = i + max(1, int(np.searchsorted(cost, budget, side="right")))
        out.append((i, j, int(cm[j - i - 1])))
        i = j
    return out


class MvPylateBackend(base.ResidentBackend):

    """PyLate exact MaxSim search on a single GPU using float32.

    Uses PyLate's torch scoring implementation to exhaustively score each
    corpus block. Supports three modes:

    - rerank-released: original rerank implementation, with a zero-padding
    issue that can produce incorrect scores.
    - rerank-released-ownpad: fixes padding to preserve exact MaxSim scores.
    - scores-batched: batches queries and documents for efficient scoring.

    Scratch memory is bounded by max_mib, with optional query-length bucketing
    to reduce padding overhead. Empty documents and queries score -inf.
    """

    name = engine = "pylate"
    applies_offset = True

    def __init__(self, k, mode, device="cuda", events=True, metric="ip",
                 max_bytes=PYLATE_DEFAULT_MAX_MIB << 20, module=None, rerank_results="device",
                 query_buckets=1):
        try:
            import torch  # noqa: F401
            rank, scores, version = module if module is not None else pylate_module()
        except ImportError as e:
            raise RuntimeError("--backend pylate needs torch and pylate (pip install "
                               "--no-deps pylate @ git+https://github.com/lightonai/"
                               f"pylate.git@{PYLATE_COMMIT}; import failed: {e})") from e
        if mode not in PYLATE_MODES:
            raise ValueError(mode)
        if metric not in schema.METRICS:
            raise ValueError(metric)
        if rerank_results not in PYLATE_RERANK_RESULTS:
            raise ValueError(rerank_results)
        if rerank_results == "host" and not pylate_released(mode):
            raise ValueError("rerank_results='host' is for the rerank modes only")
        if int(query_buckets) < 1 or (int(query_buckets) > 1 and pylate_released(mode)):
            raise ValueError("query_buckets: >= 1, and > 1 only for scores-batched")
        self.query_buckets = int(query_buckets)
        super().__init__(k, np.float32, device, events)
        self.queries_mode = "resident"
        self.metric, self.mode, self.max_bytes = metric, mode, int(max_bytes)
        self.released = pylate_released(mode)
        self.host_results = rerank_results == "host"
        self.normalize_blocks = metric == "cosine"
        self._rank, self._scores = rank, scores
        self.engine_version = f"{version}+{PYLATE_COMMIT[:7]}"
        self._init_torch()
        check_fp32_matmul(self.torch)
        # rerank() calls colbert_scores without a backend, and "auto" would
        # pick an fp16 / TF32 kernel; the variable is read per call, process-wide
        os.environ["PYLATE_SCORES_BACKEND"] = "torch"
        self.stream = self._make_stream()

    def setup(self, q):
        """Upload the queries (a multivec.Queries) once.

        Tokens are unit rows for cosine (normalised in float64, rounded once).
        Rerank modes get one (len, dim) tensor per query, as PyLate's encode
        yields; scores-batched gets the queries sorted by token count and
        zero-padded to the longest, plus the order to undo it. Also the
        zero-token mask."""
        t0 = now()
        torch = self.torch
        check_fp32_matmul(torch)
        self.q = q
        tok = multivec.unit_tokens(q.tokens) if self.metric == "cosine" else q.tokens
        lens = q.lengths
        self.q_empty_h = lens == 0
        self.q_lmax = max(int(lens.max(initial=0)), 1)
        with self._on_stream():
            tok_d = torch.from_numpy(np.ascontiguousarray(tok, np.float32)).to(self.device)
            if self.released:
                self.q_list = [tok_d[int(q.offsets[i]):int(q.offsets[i + 1])]
                               for i in range(q.n_rows)]
                self.q_pad = None
            else:
                order = np.argsort(lens, kind="stable")
                self.q_slens = lens[order]
                pad = np.zeros((q.n_rows, self.q_lmax, q.dim), np.float32)
                for j, i in enumerate(order):
                    a, b = int(q.offsets[i]), int(q.offsets[i + 1])
                    pad[j, :b - a] = tok[a:b]
                self.q_pad = torch.from_numpy(pad).to(self.device)
                self.q_order = torch.from_numpy(order.astype(np.int64)).to(self.device)
                self.q_empty_sorted = torch.from_numpy(self.q_slens == 0).to(self.device)
                self.q_list = None
                del tok_d
        self._sync()
        return now() - t0

    # ------------------------------------------------------------ per block

    def _block_search(self, xb, xb_h, e1):
        """Views of the packed block, fp16 tokens widened, cosine tokens
        normalised in place (copied first if the view is the host block, CPU
        device only), then PyLate."""
        torch = self.torch
        ev = self.events
        xb_t = xb_h.pop()
        off_d, tok = mv_views(torch, xb_t, xb)
        e2 = e1
        widened = tok.dtype != torch.float32
        if widened:
            tok = tok.to(torch.float32)
            if ev:
                e2 = self._mark()
        e2n = e2
        if self.normalize_blocks:
            if not widened and not self._owned(xb_t, xb):
                tok = tok.clone()  # never write the host block (CPU device only)
            nrm = torch.linalg.vector_norm(tok, dim=1, keepdim=True)
            tok.div_(torch.where(nrm > 0, nrm, torch.ones_like(nrm)))
            del nrm
            if ev:
                e2n = self._mark()
        off_h = multivec.unpack(xb).offsets  # host: the planning reads these
        D, I = self._engine_search(off_d, tok, off_h)
        return D, I, (e1, e2), (e2, e2n), e2n

    def _engine_search(self, off_d, tok, off_h):
        """The mode's search, counting PyLate calls: per pass (pylate_calls) and
        the most any one block took (pylate_calls_max_block)."""
        before = self.stats.get("pylate_calls")
        out = self._rerank(off_d, tok, off_h) if self.released else \
            self._batched(off_d, tok, off_h)
        self.stats.add("pylate_calls", 0)
        self.stats.peak("pylate_calls_max_block",
                        int(self.stats.get("pylate_calls") - before))
        return out

    def _empty_result(self, k_eff):
        torch = self.torch
        dev = "cpu" if self.host_results else self.device
        return (torch.full((self.q.n_rows, k_eff), float("-inf"), dtype=torch.float32,
                           device=dev),
                torch.full((self.q.n_rows, k_eff), -1, dtype=torch.int64, device=dev))

    def _rerank(self, off_d, tok, off_h):
        """rank.rerank over consecutive candidate ranges of the block.

        One range when the block fits --pylate-max-mib. Each query with tokens
        keeps the top k of its sorted list; zero-token documents are never
        candidates. Ranges are merged with FAISS's ResultHeap, as run_pass
        merges files. Returns device tensors, or CPU tensors (host_results)."""
        torch = self.torch
        nq, n, dim = self.q.n_rows, len(off_h) - 1, tok.shape[1]
        lens = np.diff(off_h)
        empty = lens == 0
        k_eff = min(self.k, n - int(empty.sum()))
        qs = np.flatnonzero(~self.q_empty_h)
        if k_eff <= 0 or not len(qs):
            return self._empty_result(max(k_eff, 0))
        own = self.mode == "rerank-released-ownpad"
        Lq = self.q_lmax
        chunks = mv_doc_chunks(lens, dim * 4 * (2 if own else 1) + Lq * 4, Lq * 12 + 32,
                               self.max_bytes)
        len_d = off_d[1:] - off_d[:-1]
        queries = [self.q_list[i] for i in qs]
        first = heap = None  # the first range's (D, I); FAISS's heap from the second on
        for d0, d1, L in chunks:
            # the candidates: the range's documents with tokens
            cand = d0 + np.flatnonzero(~empty[d0:d1])
            if not len(cand):
                continue
            if own:
                # the whole range padded (a zero-token document's rows are
                # clamped real tokens), its candidates picked on the host (no
                # device index, so no sync)
                idx = own_token_index(torch, off_d, len_d, slice(d0, d1), L, tok.shape[0])
                rows = tok.index_select(0, idx.reshape(-1)).view(d1 - d0, L, dim).unbind(0)
                docs = [rows[i - d0] for i in cand]
                del idx, rows
            elif len(cand) == d1 - d0:
                docs = list(tok[int(off_h[d0]):int(off_h[d1])].split(lens[d0:d1].tolist()))
            else:
                docs = [tok[int(off_h[i]):int(off_h[i + 1])] for i in cand]
            with self._lib():
                res = self._rank.rerank(documents_ids=[cand.tolist()] * len(qs),
                                        queries_embeddings=queries,
                                        documents_embeddings=[docs] * len(qs))
            self.stats.add("pylate_calls", 1)
            del docs
            kk = min(k_eff, len(cand))
            Dh = np.full((nq, kk), -np.inf, np.float32)
            Ih = np.full((nq, kk), -1, np.int64)
            for qi, r in zip(qs, res):
                # the first kk entries (each list holds len(cand) >= kk), read
                # with C-level iterators (fastest option measured)
                Dh[qi] = np.fromiter(map(_RERANK_SCORE, r), np.float32, count=kk)
                Ih[qi] = np.fromiter(map(_RERANK_ID, r), np.int64, count=kk)
            del res
            if heap is None and first is None:
                first = (Dh, Ih)
                continue
            if heap is None:
                # a multi-call block: FAISS's merge, as run_pass's across
                # files. -inf (a zero-token query's row) never enters its heap
                heap = faiss.ResultHeap(nq, k_eff, keep_max=True)
                heap.add_result(*first)
                first = None
            heap.add_result(Dh, Ih)
        if heap is None:
            Dh, Ih = first  # one range: no merge
        else:
            heap.finalize()
            Dh, Ih = heap.D, heap.I
            # the heap's empty slots (lowest float32, -1) back to this block's
            # own -inf fill, as with one range
            Dh[Ih < 0] = -np.inf
        if self.host_results:
            return torch.from_numpy(Dh), torch.from_numpy(Ih)  # CPU tensors, no copy
        return (torch.from_numpy(np.ascontiguousarray(Dh)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(Ih)).to(self.device))

    def _batched(self, off_d, tok, off_h):
        """scores-batched: every (query batch, document range) through
        colbert_scores into the batch's (bq, n) scores, then topk.

        Documents are sorted by token count (a stable argsort, same on host
        and device, so no sync) and own-token padded per range; queries are
        length-sorted and sliced per batch. Budget (--pylate-max-mib): the
        einsum's float32 scores, its max's values / indices (12 B a cell),
        the padded documents and the batch's scores; the query batch is
        halved until one longest document fits."""
        torch = self.torch
        nq, n, dim = self.q.n_rows, len(off_h) - 1, tok.shape[1]
        k_eff = min(self.k, n)
        if k_eff == 0 or nq == 0 or self.q.n_tok == 0:
            return self._empty_result(k_eff)  # nothing to score (or no query tokens)
        lens = np.diff(off_h)
        L_max = max(int(lens.max(initial=0)), 1)
        len_d = off_d[1:] - off_d[:-1]
        perm_d = torch.argsort(len_d, stable=True)
        slens = np.sort(lens)  # the ranges read only the sorted lengths
        empty_d = len_d == 0
        D = torch.empty((nq, k_eff), dtype=torch.float32, device=self.device)
        I = torch.empty((nq, k_eff), dtype=torch.int64, device=self.device)
        # --pylate-query-buckets: equal-count buckets of the length-sorted
        # queries, each planned at its own longest query
        edges = np.unique(np.linspace(0, nq, min(self.query_buckets, nq) + 1).round()
                          .astype(np.int64))
        for b0, b1 in zip(edges[:-1].tolist(), edges[1:].tolist()):
            Lq = max(int(self.q_slens[b1 - 1]), 1)

            def per_tok(bq, Lq=Lq):  # bytes per padded document token
                return bq * Lq * 4 + dim * 4

            bq = b1 - b0
            while bq > 1 and (L_max * per_tok(bq) + bq * Lq * 12 + bq * n * 4) > self.max_bytes:
                bq = (bq + 1) // 2
            chunks = mv_doc_chunks(slens, per_tok(bq), bq * Lq * 12,
                                   max(self.max_bytes - bq * n * 4, 1))
            for q0 in range(b0, b1, bq):
                q1 = min(b1, q0 + bq)
                Qb = self.q_pad[q0:q1, :max(int(self.q_slens[q1 - 1]), 1)]
                S = torch.empty((q1 - q0, n), dtype=torch.float32, device=self.device)
                for d0, d1, L in chunks:
                    if L == 0:
                        continue  # every document here is zero-token: masked below
                    sel = perm_d[d0:d1]
                    idx = own_token_index(torch, off_d, len_d, sel, L, tok.shape[0])
                    Dc = tok.index_select(0, idx.reshape(-1)).view(d1 - d0, L, dim)
                    del idx
                    with self._lib():
                        sc = self._scores.colbert_scores(Qb, Dc, backend="torch")
                    S.index_copy_(1, sel, sc)
                    del sc
                    self.stats.add("pylate_calls", 1)
                    del Dc
                S.masked_fill_(empty_d[None, :], float("-inf"))
                S.masked_fill_(self.q_empty_sorted[q0:q1, None], float("-inf"))
                d, i = torch.topk(S, k_eff, dim=1)
                rows = self.q_order[q0:q1]
                D.index_copy_(0, rows, d)
                I.index_copy_(0, rows, i)
                del S, d, i
        return D, I


def pylate_engine_bytes(args, feet, nq):
    """Device bytes of PyLate's per-block work beyond the block: the
    --pylate-max-mib budget plus the (nq, k) results (a single document
    over budget can exceed this; not estimated)."""
    return (args.pylate_max_mib << 20) + nq * args.k * 12
