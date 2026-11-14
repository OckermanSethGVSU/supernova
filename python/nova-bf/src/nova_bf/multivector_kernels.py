"""CUDA kernel for exact multivector (late-interaction) scoring.

This module is imported lazily by :mod:`nova_bf.compute`, so merge-only and
CPU installations do not need Triton.  The kernel folds a cuBLAS-materialized
token-similarity matrix into per-(query, document) MaxSim scores, using the
same prefix-sum offset arrays as the PyTorch reference path (the
``triton_reduce`` backend).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit(
    # None of these change the generated code — keeping them out of the
    # specialization key stops Triton from JIT-recompiling per slice shape.
    # Adaptive ragged slicing (`_ragged_batch_ranges`) deliberately varies the
    # per-slice document count, so `n_documents` as a compile-time constant
    # meant one ~0.3-1s compile per distinct slice size, per query-block size.
    do_not_specialize=[
        "query_start",
        "query_token_base",
        "similarity_row_stride",
        "n_queries",
        "n_documents",
        "out_row_stride",
    ],
)
def _fused_ragged_reduce_kernel(
    similarity_ptr,
    query_offsets_ptr,
    document_offsets_ptr,
    output_ptr,
    query_start,
    query_token_base,
    similarity_row_stride,
    n_queries,
    n_documents,
    out_row_stride,
    BLOCK_QUERY: tl.constexpr,
    BLOCK_DOCUMENT: tl.constexpr,
):
    """Fold a materialized token GEMM directly into query/document scores.

    The input is a row-major ``query_tokens x document_tokens`` matrix from
    cuBLAS. Each program owns one query/document pair, reads its contiguous
    ragged rectangle, computes max over document tokens and sum over query
    tokens, and writes one scalar. No atomics or token-by-document intermediate
    are required.
    """
    pair_index = tl.program_id(0)
    local_query = pair_index // n_documents
    document_index = pair_index % n_documents
    query_index = query_start + local_query

    query_begin = tl.load(query_offsets_ptr + query_index) - query_token_base
    query_end = tl.load(query_offsets_ptr + query_index + 1) - query_token_base
    document_begin = tl.load(document_offsets_ptr + document_index)
    document_end = tl.load(document_offsets_ptr + document_index + 1)

    query_lane = tl.arange(0, BLOCK_QUERY)
    document_lane = tl.arange(0, BLOCK_DOCUMENT)
    score = tl.zeros((), dtype=tl.float32)

    query_tile = query_begin
    while query_tile < query_end:
        query_row = query_tile + query_lane
        query_valid = query_row < query_end
        row_max = tl.full((BLOCK_QUERY,), float("-inf"), tl.float32)

        document_tile = document_begin
        while document_tile < document_end:
            document_column = document_tile + document_lane
            document_valid = document_column < document_end
            values = tl.load(
                similarity_ptr
                + query_row[:, None] * similarity_row_stride
                + document_column[None, :],
                mask=query_valid[:, None] & document_valid[None, :],
                other=float("-inf"),
            )
            row_max = tl.maximum(row_max, tl.max(values, axis=1))
            document_tile += BLOCK_DOCUMENT

        score += tl.sum(tl.where(query_valid, row_max, 0.0), axis=0)
        query_tile += BLOCK_QUERY

    score = tl.where(
        (query_end > query_begin) & (document_end > document_begin),
        score,
        float("-inf"),
    )
    # `out_row_stride` (not `n_documents`) so the caller can hand a row-slice
    # view of a larger output matrix and skip a separate device-to-device copy.
    tl.store(output_ptr + local_query * out_row_stride + document_index, score)


# Triton forms these offsets in int32, so oversized shapes can wrap and read 
# the wrong query's tokens without error. Decline those shapes instead.
_INT32_MAX = (1 << 31) - 1


def offsets_fit_int32(
    n_query_tokens: int,
    n_doc_tokens: int,
    n_queries: int,
    n_documents: int,
    out_row_stride: int,
    block_query: int = 8,
    block_document: int = 128,
) -> bool:
    """Return whether all kernel pointer offsets fit in int32. 
    Includes tile padding because masked lanes still form addresses. 
    Pure arithmetic so oversized cases can be tested without a GPU. 
    """
    if n_query_tokens <= 0 or n_doc_tokens <= 0:
        return True
    
    tile = (n_query_tokens + block_query) * n_doc_tokens + n_doc_tokens + block_document
    out_max = max(0, n_queries - 1) * max(0, out_row_stride) + max(0, n_documents - 1)
    return tile <= _INT32_MAX and out_max <= _INT32_MAX


def token_rows_fit_int32(n_rows: int, dim: int) -> bool:
    """Whether int32 row-stride address arithmetic is safe.

    Triton does not promote `tl.arange` row indices when multiplying by the
    stride; overflow can silently address the wrong token.
    """
    if n_rows <= 0 or dim <= 0:
        return True
    return n_rows * dim <= _INT32_MAX


def fused_ragged_maxsim_reduce(
    similarity,
    query_offsets,
    document_offsets,
    *,
    query_start: int,
    query_token_base: int,
    n_queries: int,
    out=None,
    block_query: int = 8,
    block_document: int = 128,
    num_warps: int = 4,
):
    """Reduce a cuBLAS token-similarity matrix into ragged MaxSim scores.

    `out` (optional) is written in place and must be a CUDA float32
    `(n_queries, n_documents)` tensor whose last dimension is contiguous —
    a row-slice view of a larger matrix qualifies, which is exactly the
    caller's use (`out[qs:qe]`), sparing a separate device-to-device copy.
    """
    import torch

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (similarity, query_offsets, document_offsets)
    ):
        raise TypeError("fused ragged reduction inputs must be torch tensors")
    if not all(
        tensor.is_cuda for tensor in (similarity, query_offsets, document_offsets)
    ):
        raise ValueError("fused ragged reduction requires CUDA tensors")
    # float16 is accepted as well as float32 — the output is float32 either way.
    if similarity.dtype not in (torch.float32, torch.float16) or similarity.ndim != 2:
        raise TypeError(
            "fused ragged reduction requires a 2D float32 or float16 similarity tensor")
    if query_offsets.dtype != torch.int64 or document_offsets.dtype != torch.int64:
        raise TypeError("fused ragged reduction requires int64 offset tensors")
    if (
        query_offsets.device != similarity.device
        or document_offsets.device != similarity.device
    ):
        raise ValueError("fused ragged reduction inputs must be on one CUDA device")
    if not similarity.is_contiguous():
        similarity = similarity.contiguous()

    n_documents = document_offsets.numel() - 1
    if out is None:
        out = torch.empty(
            (n_queries, n_documents), dtype=torch.float32, device=similarity.device
        )
    else:
        if out.shape != (n_queries, n_documents):
            raise ValueError(
                f"fused ragged reduction out shape {tuple(out.shape)} != "
                f"({n_queries}, {n_documents})"
            )
        if out.dtype != torch.float32 or not out.is_cuda:
            raise TypeError("fused ragged reduction out must be a CUDA float32 tensor")
        if n_documents > 0 and out.stride(1) != 1:
            raise ValueError("fused ragged reduction out must have a contiguous last dim")
        # A contiguous last dim does not by itself stop ROWS overlapping: a
        # stride-0 or short-strided view would alias several queries onto the
        # same output row and lose all but the last write.
        if n_queries > 1 and out.stride(0) < n_documents:
            raise ValueError(
                f"fused ragged reduction out row stride {out.stride(0)} < "
                f"{n_documents} columns, so its rows overlap")
    if n_queries == 0 or n_documents == 0:
        return out

    if not offsets_fit_int32(
        similarity.shape[0], similarity.shape[1], n_queries, n_documents,
        out.stride(0), block_query, block_document,
    ):
        # Refuse rather than launch
        raise ValueError(
            f"fused ragged reduction: a {tuple(similarity.shape)} similarity "
            f"tile makes the kernel's int32 pointer arithmetic overflow "
            f"(> {_INT32_MAX} elements). Lower params.multivector_budget_mib "
            "(it sizes this matrix), or set params.multivector_kernel="
            "'torch', whose reference path has no such limit."
        )

    _fused_ragged_reduce_kernel[(n_queries * n_documents,)](
        similarity,
        query_offsets,
        document_offsets,
        out,
        query_start,
        query_token_base,
        similarity.stride(0),
        n_queries,
        n_documents,
        out.stride(0),
        BLOCK_QUERY=block_query,
        BLOCK_DOCUMENT=block_document,
        num_warps=num_warps,
        num_stages=2,
    )
    return out

# ---------------------------------------------------------------------------
# Fused float16 pass one for multivector pruning.
#
# This kernel avoids materializing the full query-token × corpus-token
# similarity matrix. It performs the float16 GEMM directly, accumulates in
# float32, and takes the per-document token max in the epilogue.
#
# Each program owns one (query-token tile, document) pair and loops over that
# document's tokens, requiring no atomics or cross-tile reduction. Documents
# vary fastest in the grid to improve query-tile reuse.
#
# The output is one maximum per (query token, document); MaxSim's outer
# per-query sum is performed separately.

try:
    @triton.jit
    def _gemm_ragged_maxsim(
        Q, C, DOFF, OUT,
        M, K, n_documents,
        stride_qm, stride_cn, stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        """FP16 GEMM with FP32 accumulation and a per-document max epilogue.

        Writes the maximum token dot product for each (query token, document), or
        `-inf` for empty documents. Padded query rows are masked on store, and padded
        document lanes are clamped for safe loads then excluded from the max.
        """
        pid = tl.program_id(0)
        # Document varies fastest so consecutive programs share a query tile.
        pid_m = pid // n_documents
        doc = pid % n_documents

        d0 = tl.load(DOFF + doc).to(tl.int32)
        d1 = tl.load(DOFF + doc + 1).to(tl.int32)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rm = offs_m % M                      # wrapped rows keep loads in bounds
        offs_k = tl.arange(0, BLOCK_K)
        lane_n = tl.arange(0, BLOCK_N)

        best = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        n0 = d0
        while n0 < d1:
            offs_n = n0 + lane_n
            nm = offs_n < d1
            rn = tl.where(nm, offs_n, d0)    # clamp, then mask in the epilogue

            a_ptrs = Q + rm[:, None] * stride_qm + offs_k[None, :]
            b_ptrs = C + rn[None, :] * stride_cn + offs_k[:, None]
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                else:
                    km = (k0 + offs_k) < K
                    a = tl.load(a_ptrs, mask=km[None, :], other=0.0)
                    b = tl.load(b_ptrs, mask=km[:, None], other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K

            v = tl.where(nm[None, :], acc, float("-inf"))
            best = tl.maximum(best, tl.max(v, 1))
            n0 += BLOCK_N

        tl.store(OUT + offs_m * stride_om + doc, best, mask=offs_m < M)

except Exception:                            # no Triton, or a moved API
    _gemm_ragged_maxsim = None


def fused_fp16_token_maxima(
    q_half, c_half, document_offsets, *, out=None,
    block_m: int = 128, block_n: int = 64, block_k: int = 64, num_warps: int = 4,
    num_stages: int = 3,
):
    """Compute FP32 per-token document maxima with the fused FP16 pass.

    Returns `(n_query_tokens, n_documents)` maxima, or `None` when the kernel
    cannot safely run. Token inputs are FP16; accumulation is FP32.
    """
    import torch

    if _gemm_ragged_maxsim is None:
        return None
    if not all(t.is_cuda for t in (q_half, c_half, document_offsets)):
        return None
    if not (q_half.device == c_half.device == document_offsets.device):
        raise ValueError(
            "fused fp16 pass one inputs must be on one CUDA device")
    if q_half.dtype != torch.float16 or c_half.dtype != torch.float16:
        raise TypeError("fused fp16 pass one requires float16 token matrices")
    if q_half.ndim != 2 or c_half.ndim != 2:
        raise ValueError("fused fp16 pass one requires 2D token matrices")
    if q_half.shape[1] != c_half.shape[1]:
        raise ValueError(
            f"token dimension mismatch: queries {q_half.shape[1]} vs corpus "
            f"{c_half.shape[1]}")
    if document_offsets.dtype != torch.int64:
        raise TypeError("fused fp16 pass one requires int64 document offsets")
    q_half = q_half.contiguous()
    c_half = c_half.contiguous()

    M, K = int(q_half.shape[0]), int(q_half.shape[1])
    n_documents = int(document_offsets.numel()) - 1
    if M == 0 or K == 0 or n_documents <= 0:
        return None

    if out is None:
        out = torch.empty((M, n_documents), dtype=torch.float32,
                          device=q_half.device)
    elif out.shape != (M, n_documents) or out.dtype != torch.float32:
        raise ValueError("fused fp16 pass one out must be float32 "
                         f"({M}, {n_documents})")
    elif not out.is_cuda or out.device != q_half.device:
        raise ValueError(
            "fused fp16 pass one out must be on the input CUDA device")
    elif n_documents > 0 and out.stride(1) != 1:
        raise ValueError("fused fp16 pass one out must have a contiguous "
                         "last dimension")
    # As above: a contiguous last dim still permits overlapping ROWS, which
    # would alias several query tokens onto one output row.
    elif M > 1 and out.stride(0) < n_documents:
        raise ValueError(
            f"fused fp16 pass one out row stride {out.stride(0)} < "
            f"{n_documents} columns, so its rows overlap")

    # Triton forms these row-stride products in int32. Guard the padded
    # output tile too, since masked lanes still form addresses.
    padded_m = triton.cdiv(M, block_m) * block_m
    if (not token_rows_fit_int32(M, int(q_half.stride(0)))
            or not token_rows_fit_int32(int(c_half.shape[0]),
                                        int(c_half.stride(0)))
            or not token_rows_fit_int32(padded_m, int(out.stride(0)))):
        return None

    grid = (triton.cdiv(M, block_m) * n_documents,)
    _gemm_ragged_maxsim[grid](
        q_half, c_half, document_offsets, out,
        M, K, n_documents,
        q_half.stride(0), c_half.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        EVEN_K=(K % block_k == 0),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
