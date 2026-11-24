"""Query-batched versions of SPARe's exact sparse search (Almeida & Matos, ECIR 2024).

Subclasses SPARe's original models without modifying the library:

- dot: Replaces per-query SpMV with batched SpMM.
- iterative: Batches posting-list accumulation into a single index_add_ call.

Both preserve SPARe's scoring operations, data types, and top-k selection.
Results agree within floating-point rounding, but are not guaranteed to be
bitwise identical due to differences in operation order.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import torch
from spare.backend_torch import CSRSparseRetrievalModel, CSRSparseRetrievalModelIterative

SPARE_COMMIT = "e7f4ad1"


@dataclass
class CsrCollection:
    """The two attributes SPARe's model constructors read from a SparseCollection:
    `sparse_vecs` = (crow_indices, col_indices, values) of the (n_docs, vocab) CSR
    collection, and `shape`. Stands in for SPARe's own builder, which loops over
    documents in Python and sizes its arrays from an estimate (an overflow risk on
    our files)."""

    sparse_vecs: tuple
    shape: tuple

    @classmethod
    def from_csr(cls, indptr, indices, values, n_cols, device="cpu", dtype=torch.float32,
                 index_dtype=torch.int32):
        """CSR arrays (numpy or torch) -> collection on `device`. Columns must be
        sorted and unique within each row (coalesced), as for any torch CSR.
        Defaults follow SPARe's SparseCollectionCSR (float32 values, int32
        indices). Build on CPU and move the model to the GPU afterwards, as SPARe
        does: the iterative model's CSR->CSC conversion runs on the collection's
        device and needs ~32-44 B/nnz of transient memory there."""
        if index_dtype == torch.int32 and int(indptr[-1]) > torch.iinfo(torch.int32).max:
            raise ValueError("nnz exceeds int32: pass index_dtype=torch.int64")
        def t(a, dt):
            a = a if isinstance(a, torch.Tensor) else torch.from_numpy(a)
            return a.to(device=device, dtype=dt)
        crow, col, val = t(indptr, index_dtype), t(indices, index_dtype), t(values, dtype)
        return cls((crow, col, val), (int(crow.numel() - 1), int(n_cols)))


def queries_to_padded(queries, device="cpu", dtype=torch.float32):
    """List of {term_id: weight} dicts (SPARe's query format) -> (term_ids, weights),
    both (n_q, max_terms), padded with term id -1 / weight 0."""
    for q in queries:
        for t in q:
            if isinstance(t, bool) or not isinstance(t, numbers.Integral) or t < 0:
                raise ValueError(f"query term ids must be non-negative ints, got {t!r}")
    n = len(queries)
    width = max((len(q) for q in queries), default=0)
    ids = torch.full((n, max(width, 1)), -1, dtype=torch.int64)
    w = torch.zeros((n, max(width, 1)), dtype=dtype)
    for i, q in enumerate(queries):
        if q:
            ids[i, : len(q)] = torch.tensor(list(q.keys()), dtype=torch.int64)
            w[i, : len(q)] = torch.tensor(list(q.values()), dtype=dtype)
    return ids.to(device), w.to(device)


@dataclass
class CsrQueries:
    """The queries as one flat list of (query, term, weight) pairs in row-major
    order (query by query, each query's terms in its own order: the order SPARe's
    per-query loop visits them). ``indptr`` is kept on the host (int64 numpy) so a
    group's pairs are the slice indptr[lo]:indptr[hi] without a device->host sync;
    ``indptr_dev`` is its device copy (for per-query sums on the device), ``rows``
    the query of every pair."""

    indptr: object        # numpy int64 (n + 1,), host
    indptr_dev: torch.Tensor
    terms: torch.Tensor   # int64 (pairs,)
    weights: torch.Tensor
    rows: torch.Tensor    # int64 (pairs,)

    @property
    def n(self):
        return len(self.indptr) - 1

    @classmethod
    def from_csr(cls, indptr, indices, values, device="cpu", dtype=torch.float32):
        """CSR arrays (numpy or torch; indptr host-readable) -> pairs on `device`."""
        import numpy as np
        ip = np.asarray(indptr.cpu() if isinstance(indptr, torch.Tensor) else indptr,
                        dtype=np.int64)
        if ip.ndim != 1 or len(ip) < 1 or ip[0] != 0 or (np.diff(ip) < 0).any():
            raise ValueError("indptr must be a non-decreasing 1-d array starting at 0")

        def t(a, dt):
            a = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.asarray(a))
            return a.to(device=device, dtype=dt)
        terms, w = t(indices, torch.int64), t(values, dtype)
        if terms.numel() != ip[-1] or w.numel() != ip[-1]:
            raise ValueError(f"indptr[-1] = {ip[-1]} but {terms.numel()} indices / "
                             f"{w.numel()} values")
        rows = torch.repeat_interleave(torch.arange(len(ip) - 1, device=device),
                                       torch.from_numpy(np.diff(ip)).to(device),
                                       output_size=int(ip[-1]))
        return cls(ip, torch.from_numpy(ip).to(device), terms, w, rows)

    @classmethod
    def from_padded(cls, term_ids, weights):
        """(n_q, T) -1 padded term ids / weights -> pairs (one mask, once)."""
        valid = term_ids >= 0
        counts = valid.sum(1).cpu()
        ip = torch.zeros(len(counts) + 1, dtype=torch.int64)
        torch.cumsum(counts, 0, out=ip[1:])
        dev = term_ids.device
        rows = torch.arange(term_ids.shape[0], device=dev).unsqueeze(1).expand_as(term_ids)
        return cls(ip.numpy(), ip.to(dev), term_ids[valid].long(), weights[valid],
                   rows[valid])


def _group_queries(costs, max_cost, max_batch, counts=None, max_count=None):
    """Consecutive query groups [(lo, hi)), each with summed cost <= max_cost
    (a single query over budget gets a group of its own), at most max_batch
    queries and, with ``counts``, summed counts <= max_count (again unless a
    single query). `costs` / `counts` are CPU int64 tensors."""
    groups, lo, acc, cnt = [], 0, 0, 0
    c = costs.tolist()
    n = counts.tolist() if counts is not None else None
    for i, ci in enumerate(c):
        ni = n[i] if n is not None else 0
        if i > lo and (acc + ci > max_cost or i - lo >= max_batch
                       or (n is not None and cnt + ni > max_count)):
            groups.append((lo, i))
            lo, acc, cnt = i, 0, 0
        acc += ci
        cnt += ni
    if lo < len(c):
        groups.append((lo, len(c)))
    return groups


def _check_budgets(k, n_docs, **budgets):
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 0 < k <= n_docs:
        raise ValueError(f"need an int 0 < k <= n_docs ({n_docs}), got k={k!r}")
    for name, v in budgets.items():
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1:
            raise ValueError(f"{name} must be a positive int, got {v!r}")


def _check(term_ids, weights, k, n_docs, vocab, **budgets):
    """Argument checks of the padded API (cheap: one device->host sync)."""
    if term_ids.dim() != 2 or term_ids.shape != weights.shape:
        raise ValueError(f"term_ids {tuple(term_ids.shape)} and weights {tuple(weights.shape)} "
                         "must both be (n_queries, max_terms)")
    if term_ids.is_floating_point() or term_ids.is_complex() or term_ids.dtype == torch.bool:
        raise TypeError(f"term_ids must be an integer tensor, got {term_ids.dtype}")
    if not weights.is_floating_point():
        raise TypeError(f"weights must be a floating-point tensor, got {weights.dtype}")
    _check_budgets(k, n_docs, **budgets)
    if term_ids.numel():
        lo, hi = int(term_ids.min()), int(term_ids.max())
        if lo < -1 or hi >= vocab:
            raise ValueError(f"term ids must be -1 (padding) or in [0, {vocab}); "
                             f"got min {lo}, max {hi}")


def check_csr(q, k, n_docs, vocab, **budgets):
    """Argument checks of the pair API (one device->host sync for the term
    range); a caller searching the SAME queries over many collections (the
    streaming harness) runs it once and passes validate=False after."""
    if not q.weights.is_floating_point():
        raise TypeError(f"weights must be a floating-point tensor, got {q.weights.dtype}")
    _check_budgets(k, n_docs, **budgets)
    if q.terms.numel():
        lo, hi = int(q.terms.min()), int(q.terms.max())
        if lo < 0 or hi >= vocab:
            raise ValueError(f"term ids must be in [0, {vocab}); got min {lo}, max {hi}")


def _empty(k, score_dtype, device):
    return (torch.empty((0, k), dtype=score_dtype, device=device),
            torch.empty((0, k), dtype=torch.int64, device=device))


TOPK_LAYOUTS = ("contiguous", "strided")
# torch's CUDA topk sorts the whole slice instead of a radix select when there
# are <= 10 slices of >= 100k elements (aten/src/ATen/native/cuda/TensorTopK.cpp
# should_use_sort): the sort's values + int64 indices + CUB double buffers are
# ~4 more score-sized buffers per slice (an estimate, not measured on a GPU)
TOPK_SORT_MAX_SLICES = 10
TOPK_SORT_MIN_LEN = 100_000
TOPK_SORT_FACTOR = 4
INDEX_DTYPES = ("auto", "int64")
INT32_MAX = 2 ** 31 - 1
# int32 expansion indices address [0, INT32_SPAN): at most this many
# accumulator cells / postings / pairs per group
INT32_SPAN = 2 ** 31


class Spans:
    """Optional stage timing for the batched / nosync searches, without a sync.

    ``mark()`` (the caller's) returns a point on the device timeline: a
    torch.cuda.Event recorded on the current stream, or a host timestamp on the
    CPU. The searches append (start, end) pairs of marks to ``plan`` (iterative:
    the planning tensors and the postings check), ``score`` (the SpMM / the gather
    + scatter into the accumulator) and ``topk`` (top-k, with dot's contiguous
    copy). The caller reads the elapsed times later, once the work is known to be
    complete. The marks sit between this module's calls into torch, never inside
    SPARe's own code. They measure elapsed time on the stream, not GPU busy time:
    a host stall between two launches falls inside its span."""

    __slots__ = ("mark", "plan", "score", "topk")

    def __init__(self, mark):
        self.mark = mark
        self.plan, self.score, self.topk = [], [], []


class BatchedDot(CSRSparseRetrievalModel):
    """SPARe `dot` mode, queries batched: collection CSR @ dense (vocab, B)."""

    def batch_size(self, max_bytes, max_batch, topk_layout="contiguous"):
        """Queries per SpMM: largest B with B * per_query_bytes <= max_bytes (>= 1)."""
        n_docs, vocab = self.shape
        item = self.values.element_size()
        factor = self.DOT_DOCS_FACTOR if topk_layout == "contiguous" else \
            self.DOT_DOCS_FACTOR_STRIDED
        b = max(1, min(max_batch, max_bytes // ((vocab + factor * n_docs) * item)))
        if b <= TOPK_SORT_MAX_SLICES and n_docs >= TOPK_SORT_MIN_LEN:
            # CUDA topk switches to a full sort here (should_use_sort): charge its
            # values + int64 indices + double buffers, TOPK_SORT_FACTOR more
            b = max(1, min(b, max_bytes // ((vocab + (factor + TOPK_SORT_FACTOR) * n_docs)
                                            * item)))
        return b

    # Device bytes per query in a batch, in units of the score dtype's size:
    # the (vocab, B) query matrix, the (n_docs, B) SpMM output, and up to two
    # more (n_docs, B)-sized buffers: the contiguous (B, n_docs) copy of the
    # output that topk reads (topk_layout="contiguous") and topk's workspace
    # (a deliberately generous allowance; the k-sized topk outputs are
    # negligible). Not measured on a GPU.
    DOT_DOCS_FACTOR = 3
    DOT_DOCS_FACTOR_STRIDED = 2  # no contiguous copy: the output + topk workspace

    def search(self, term_ids, weights, k, max_bytes=4 << 30, max_batch=4096, out_device=None,
               validate=True, topk_layout="contiguous"):
        """Exact top-k for every query. term_ids/weights: (n_q, T), -1 padded.
        Returns (scores, ids), both (n_q, k), scores descending, on
        `out_device` (default: the model's device; pass "cpu" to move each
        batch's results off the GPU as it finishes). Converts to CsrQueries
        (one mask, one sync) and runs search_csr.
        B is capped so B * (vocab + 3 * n_docs) * itemsize <= max_bytes.
        Not in the budget: the model itself (collection), the inputs, and the
        results (n_q * k * 12 B, doubled by the final cat, on out_device).
        validate=False skips the argument checks (_check: one device->host
        sync for the term-id range), for a caller that already validated the
        same queries and guarantees 0 < k <= n_docs and ids < vocab (the
        streaming harness, per block)."""
        n_docs, vocab = self.shape
        if validate:
            _check(term_ids, weights, k, n_docs, vocab, max_bytes=max_bytes,
                   max_batch=max_batch)
        dev = self.values.device
        if term_ids.shape[0] == 0:
            return _empty(k, self.values.dtype, out_device or dev)
        q = CsrQueries.from_padded(term_ids.to(dev), weights.to(dev))
        return self.search_csr(q, k, max_bytes, max_batch, out_device, validate=False,
                               topk_layout=topk_layout)

    def search_csr(self, q, k, max_bytes=4 << 30, max_batch=4096, out_device=None,
                   validate=True, topk_layout="contiguous", spans=None):
        """search() on CsrQueries: no masks, no device->host sync (a batch's
        pairs are a host-known slice). ``topk_layout``: "contiguous" (the
        (n_docs, B) SpMM output transposed to a contiguous (B, n_docs) copy,
        then topk along rows: the contiguous slices SPARe's own per-query topk
        reads) or "strided" (topk down the columns of the row-major output).
        ``spans``: a Spans to time score / topk per batch (two marks a batch)."""
        n_docs, vocab = self.shape
        if topk_layout not in TOPK_LAYOUTS:
            raise ValueError(topk_layout)
        if validate:
            check_csr(q, k, n_docs, vocab, max_bytes=max_bytes, max_batch=max_batch)
        dev, dt = self.values.device, self.values.dtype
        if q.n == 0:
            return _empty(k, dt, out_device or dev)
        b = self.batch_size(max_bytes, max_batch, topk_layout)
        C = torch.sparse_csr_tensor(self.crow, self.indice, self.values, self.shape, dtype=dt)
        out_s, out_i = [], []
        m = spans.mark() if spans is not None else None
        for lo in range(0, q.n, b):
            hi = min(q.n, lo + b)
            p0, p1 = int(q.indptr[lo]), int(q.indptr[hi])
            nb = hi - lo
            Q = torch.zeros((vocab, nb), dtype=dt, device=dev)
            if p1 > p0:
                # SPARe: sparse_coo_tensor(term ids, weights).to_dense() per
                # query (dtype = the collection's), which sums duplicate term
                # ids; accumulate=True does the same
                Q.index_put_((q.terms[p0:p1], q.rows[p0:p1] - lo),
                             q.weights[p0:p1].to(dt), accumulate=True)
            S = C @ Q                                     # (n_docs, nb)
            del Q
            if spans is not None:
                m1 = spans.mark()
                spans.score.append((m, m1))
            if topk_layout == "contiguous":
                St = S.T.contiguous()                     # (nb, n_docs)
                del S
                s, i = torch.topk(St, k=k, dim=1)
                del St
            else:
                s, i = torch.topk(S, k=k, dim=0)          # (k, nb)
                del S
                s, i = s.T, i.T
            if spans is not None:
                m = spans.mark()
                spans.topk.append((m1, m))
            out_s.append(s.to(out_device) if out_device else s)
            out_i.append(i.to(out_device) if out_device else i)
            if spans is not None and out_device:
                m = spans.mark()  # the copy off the device is in neither span
        return torch.cat(out_s), torch.cat(out_i)


class BatchedIterative(CSRSparseRetrievalModelIterative):
    """SPARe `iterative` mode, queries batched: a restructuring (ours) of the
    paper's Algorithm 1 with the same per-posting products and accumulate op."""

    def plan(self, postings, max_bytes, max_batch, index_dtype="auto"):
        """Query groups for `search` from each query's gathered-postings count
        (CPU int64). Returns (groups, cost): cost[q] = bytes_per_posting *
        postings[q] + n_docs * accumulator itemsize, and every group's summed
        cost <= max_bytes unless it is a single query. With 32-bit indices
        (``index_dtype`` "auto" and the collection small enough) groups are
        also kept to < 2^31 postings and B * n_docs < 2^31 (int32 range)."""
        acc_item = torch.empty((), dtype=self.storage_dtype).element_size()
        i32 = self._int32(index_dtype, max_bytes)
        per = self.BYTES_PER_POSTING_INT32 if i32 else self.BYTES_PER_POSTING
        cost = postings * per + self.shape[0] * acc_item
        if i32:
            max_batch = max(1, min(max_batch, INT32_SPAN // max(self.shape[0], 1)))
            return _group_queries(cost, max_bytes, max_batch, postings, INT32_SPAN), cost
        return _group_queries(cost, max_bytes, max_batch), cost

    def _int32(self, index_dtype, max_bytes):
        """Whether this call uses int32 expansion indices: "auto", the collection
        in int32 range, and a budget that can never plan a group past it (its
        accumulator <= max_bytes / acc_item cells, its postings <= max_bytes /
        BYTES_PER_POSTING_INT32). Then the plan's int32 caps never bind, so int32
        never shrinks a group; a larger budget uses int64 throughout."""
        if index_dtype not in INDEX_DTYPES:
            raise ValueError(index_dtype)
        acc_item = torch.empty((), dtype=self.storage_dtype).element_size()
        return (index_dtype == "auto" and self.shape[0] <= INT32_MAX
                and self.cvalues.numel() <= INT32_MAX
                and max_bytes // acc_item <= INT32_SPAN
                and max_bytes // self.BYTES_PER_POSTING_INT32 <= INT32_SPAN)

    # Peak device scratch per gathered posting, in bytes, of the expansion
    # below (index width I = 8 int64 / 4 int32; values in the collection dtype
    # <= 4 B, accumulator dtype <= 4 B): `pos` I + its arange temporary I;
    # then `vals` 4 + the expanded weights 4 while `pos` is alive; then
    # `target` I (+ the int32 rindices gather 4 before its widen, int64 only)
    # with `pos` and `vals` alive; then, `pos` freed, `vals` + `target` + the
    # expanded qbase I; then `vals` cast to the accumulator dtype (<= 4).
    # Worst: pos + vals + weights-or-target + temp: int64 8+4+8+4+8 = 32 B,
    # int32 4+4+4+4 = 16 B (+ the cast 4: ~16-20 B). Charged 48 / 32 (allocator
    # rounding, index_add_). Not measured on a GPU.
    BYTES_PER_POSTING = 48
    BYTES_PER_POSTING_INT32 = 32

    def search(self, term_ids, weights, k, max_bytes=4 << 30, max_batch=4096, out_device=None,
               validate=True, index_dtype="auto"):
        """Exact top-k for every query (same contract as BatchedDot.search,
        including what is not in the budget; also outside it: per-query
        planning tensors, ~40 B * n_q * max_terms). Converts to CsrQueries
        and runs search_csr.
        Queries are grouped so that each group's scratch,
        bytes_per_posting * (gathered postings) + B * n_docs * acc_itemsize
        (the dense accumulator), stays <= `max_bytes` (a single query over
        budget is scored alone), with at most `max_batch` queries per group."""
        n_docs, vocab = self.shape
        if validate:
            _check(term_ids, weights, k, n_docs, vocab, max_bytes=max_bytes,
                   max_batch=max_batch)
        self._check_dtype()
        dev = self.ccol.device
        if term_ids.shape[0] == 0:
            return _empty(k, self.storage_dtype, out_device or dev)
        q = CsrQueries.from_padded(term_ids.to(dev), weights.to(dev))
        return self.search_csr(q, k, max_bytes, max_batch, out_device, validate=False,
                               index_dtype=index_dtype)

    def _check_dtype(self):
        if self.cvalues.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"collection values are {self.cvalues.dtype}: only float32 / float64 "
                "collections reproduce SPARe's per-posting product exactly (for integer "
                "or reduced-precision collections SPARe's product dtype/rounding follows "
                "promotion with its 0-d float32 query weight and differs CPU vs GPU). "
                "Use objective='half'/'performance' for low-precision accumulation.")

    def query_postings(self, q):
        """Every query's gathered-postings count, on the device (no sync):
        the CSC column lengths of its terms, summed per query."""
        ccol = self.ccol.long()
        lens = ccol[q.terms + 1] - ccol[q.terms]
        return _segment_sums(lens, q.indptr_dev)

    def search_csr(self, q, k, max_bytes=4 << 30, max_batch=4096, out_device=None,
                   validate=True, index_dtype="auto", postings=None, spans=None):
        """search() on CsrQueries: no masks, so the only device->host sync is
        the planning one (each query's postings count). ``postings``: those
        counts already on the host (CPU int64, e.g. computed by the streaming
        harness from the CSR column histogram before the CSC conversion and
        copied asynchronously), so this call does not wait for the conversion;
        they are checked against the CSC on the device (torch._assert_async: a
        mismatch fails loudly, never silently).
        ``index_dtype``: "auto" (int32 expansion indices whenever they fit,
        as SPARe's own int32 row indices) or "int64". ``spans``: a Spans
        to time plan (once) and score / topk per group (two marks a group)."""
        n_docs, vocab = self.shape
        if validate:
            check_csr(q, k, n_docs, vocab, max_bytes=max_bytes, max_batch=max_batch)
        self._check_dtype()
        dev = self.ccol.device
        if q.n == 0:
            return _empty(k, self.storage_dtype, out_device or dev)
        m = spans.mark() if spans is not None else None
        ccol = self.ccol.long()
        lens_all = ccol[q.terms + 1] - ccol[q.terms]          # per pair, device
        if postings is None:
            postings = _segment_sums(lens_all, q.indptr_dev).cpu()   # the planning sync
        else:
            if postings.shape != (q.n,):
                raise ValueError(f"postings {tuple(postings.shape)} for {q.n} queries")
            # device-side check (no sync on CUDA; a mismatch raises there as a
            # device assert, on the CPU right here)
            torch._assert_async((postings.to(dev, non_blocking=True)
                                 == _segment_sums(lens_all, q.indptr_dev)).all())
        groups, _ = self.plan(postings, max_bytes, max_batch, index_dtype)
        i32 = self._int32(index_dtype, max_bytes)
        if spans is not None:
            m1 = spans.mark()
            spans.plan.append((m, m1))
            m = m1

        out_s, out_i = [], []
        for lo, hi in groups:
            p0, p1 = int(q.indptr[lo]), int(q.indptr[hi])
            nb = hi - lo
            acc = torch.zeros(nb * n_docs, dtype=self.storage_dtype, device=dev)
            total = int(postings[lo:hi].sum())            # host-side, no sync
            if total:
                idt = torch.int32 if (i32 and total <= INT32_SPAN and nb * n_docs <= INT32_SPAN
                                      and p1 - p0 <= INT32_SPAN) else torch.int64
                t, lens_v = q.terms[p0:p1], lens_all[p0:p1]
                # SPARe multiplies cvalues[s:e] by values[i], a 0-d float32
                # tensor (QuestionDataset: torch.tensor of python floats). For a
                # float32 / float64 collection PyTorch computes that in the
                # collection dtype, i.e. exactly cvalues * w cast to that
                # dtype, as here. (Other collection dtypes are refused.)
                w_v = q.weights[p0:p1].to(self.cvalues.dtype)
                # every per-pair quantity is expanded straight to its postings
                # (repeat_interleave), in row-major (query, term) order = SPARe's
                # loop order per query; gathers by index_select, which keeps an
                # int32 index (x[idx] would widen it to int64 first)
                lens_i = lens_v.to(idt)
                # posting position = start of the pair's range + offset within it
                delta = (ccol.index_select(0, t) - (torch.cumsum(lens_v, 0) - lens_v)).to(idt)
                pos = torch.repeat_interleave(delta, lens_i, output_size=total)
                del delta
                pos += torch.arange(total, device=dev, dtype=idt)
                # SPARe: (cvalues[s:e] * values[i]).type(storage_dtype)
                vals = self.cvalues.index_select(0, pos)
                vals *= torch.repeat_interleave(w_v, lens_i, output_size=total)
                target = self.rindices.index_select(0, pos).to(idt)
                del pos
                # the accumulator row of the pair's query, once per pair
                qbase = (q.rows[p0:p1] - lo).to(idt) * n_docs
                target += torch.repeat_interleave(qbase, lens_i, output_size=total)
                del qbase
                vals = vals.type(self.storage_dtype)
                acc.index_add_(0, target, vals)
                del target, vals
            if spans is not None:
                m1 = spans.mark()
                spans.score.append((m, m1))
            s, i = torch.topk(acc.view(nb, n_docs), k=k, dim=1)
            del acc
            if spans is not None:
                m = spans.mark()
                spans.topk.append((m1, m))
            out_s.append(s.to(out_device) if out_device else s)
            out_i.append(i.to(out_device) if out_device else i)
            if spans is not None and out_device:
                m = spans.mark()  # the copy off the device is in neither span
        return torch.cat(out_s), torch.cat(out_i)


class IterativeReleasedNoSync(CSRSparseRetrievalModelIterative):
    """SPARe's released iterative forward with one implementation fix.

    The released code (spare/backend_torch.py,
    CSRSparseRetrievalModelIterative.forward: one query per call, one index_add_
    per query term, Algorithm 1) iterates zip(ccol[indices], ccol[indices + 1])
    and slices with the resulting 0-d device tensors: four device->host scalar
    reads per term (backend_torch.py:472-476). Here the CSC column bounds are
    host ints (``host_bounds()``: ccol copied to the host once per collection)
    and the query's term ids are given on the host, so slicing ``rindices[s:e]``
    needs no device->host read. Nothing else changes: the same accumulator,
    allocated on the device as SPARe's does, and the same per-term slices,
    product, cast, index_add_ and topk, in the same order.
    (The ``torch.zeros(...).to(device)`` accumulator appears only in SPARe's
    ThreadSafe (multi-GPU, :383) and sharded (:681) iterative classes; the
    resident single-GPU class used here allocates on the device, :470.)

    Report it next to the untouched released loop, never instead of it."""

    def host_bounds(self):
        self._ccol_h = self.ccol.cpu().tolist()  # one sync, per collection

    def forward_nosync(self, terms, values, spans=None):
        """``terms``: the query's term ids (host ints); ``values``: its (1,
        terms) device weights, as SPARe's forward takes them. ``spans``: a
        Spans to time this query's score (accumulator + per-term index_add_)
        and topk: three marks, around the same operations in the same order."""
        m0 = spans.mark() if spans is not None else None
        values = values.squeeze(0)
        accumulated = torch.zeros(self.shape[0], dtype=self.storage_dtype,
                                  device=self.rindices.device)
        c = self._ccol_h
        for i, t in enumerate(terms):
            v_indices = self.rindices[c[t]: c[t + 1]]
            v_values = (self.cvalues[c[t]: c[t + 1]] * values[i]).type(self.storage_dtype)
            accumulated.index_add_(0, v_indices, v_values)
        if spans is None:
            return torch.topk(accumulated, k=self.top_k, dim=0)
        m1 = spans.mark()
        r = torch.topk(accumulated, k=self.top_k, dim=0)
        spans.score.append((m0, m1))
        spans.topk.append((m1, spans.mark()))
        return r


def _segment_sums(x, indptr_dev):
    """Per-segment sums of ``x`` (int64) for segments indptr[j]:indptr[j+1],
    on the device (a cumsum and two gathers, no sync)."""
    cs = torch.zeros(x.numel() + 1, dtype=torch.int64, device=x.device)
    torch.cumsum(x, 0, out=cs[1:])
    return cs[indptr_dev[1:]] - cs[indptr_dev[:-1]]


def run_released(model, queries, k):
    """SPARe exactly as released, for the comparison: its own single-device
    `forward_retrieval` loop (QuestionDataset + DataLoader, one query per
    call) on a SPARe model (`CSRSparseRetrievalModel` or
    `CSRSparseRetrievalModelIterative`, or the batched subclasses, whose
    inherited `forward` is SPARe's). `queries`: list of {term_id: weight}.
    Returns (scores, ids), both (n_q, k).

    SPARe's TorchBackend picks its devices itself (every visible GPU, else
    CPU) and forward_retrieval uses devices[0]; the model must already be on
    that device (checked). Timing is the caller's: wrap this call and the
    batched `search` in the same boundaries (synchronize, start before the
    first query, stop once results are on the host)."""
    from spare.backend_torch import QuestionDataset, TorchBackend

    backend = TorchBackend()
    if len(backend.devices) != 1:
        raise RuntimeError(f"SPARe sees {len(backend.devices)} devices; forward_retrieval (its "
                           "single-device path) would then collate several ragged queries per "
                           "call: expose one GPU (CUDA_VISIBLE_DEVICES; SPARe's multi-GPU path "
                           "is not used here)")
    dev = torch.device(backend.devices[0])
    p = next(model.parameters())
    if p.device != dev:
        raise RuntimeError(f"SPARe's forward_retrieval runs on {dev}, model is on {p.device}: "
                           "move it there (or set CUDA_VISIBLE_DEVICES) first")
    out = backend.forward_retrieval(
        model, QuestionDataset, queries, question_func=lambda q: q, top_k=k,
        collect_at=None, profiling=False, return_scores=True)
    return out.scores, out.ids
