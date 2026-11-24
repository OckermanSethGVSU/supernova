"""FAISS GPU engine for dense legs: faiss.knn_gpu on the shared ResidentBackend
per-block flow (leg -> engine -> clock)."""

import importlib

import faiss  # must import after cudalibs.torch_first() has loaded torch's CUDA libraries
import numpy as np

from bench.record import schema
from bench.timing import now
from bench.systems import streaming as base


class GpuBackend(base.ResidentBackend):
    """faiss.knn_gpu on one GPU (engine "faiss").

    knn_gpu (through faiss.contrib.torch_utils) runs on torch's current stream;
    _on_stream sets that to the same explicit non-default stream cuVS uses, so
    the CUDA timing events bracket the call.

    Cosine: FAISS has no cosine metric, so the standard recipe is normalise +
    inner product: queries once on the host in setup(), each block on the
    device inside the timed call (``normalize``). knn_gpu always runs inner product.
    """

    name = "gpu"
    engine = "faiss"
    engine_version = faiss.__version__

    def __init__(self, k, dtype=np.float32, temp_mem=None, device="cuda", events=True,
                 metric="ip"):
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "gpu backend needs a faiss GPU build: this faiss has knn_gpu "
                "but no StandardGpuResources (faiss-cpu). Use --backend cpu.")
        if metric not in schema.METRICS:
            raise ValueError(metric)
        super().__init__(k, dtype, device, events)
        self.queries_mode = "resident"
        self.metric = metric
        # cosine: normalise each block on the device, inside the call
        self.normalize_blocks = metric == "cosine"
        self.res = faiss.StandardGpuResources()
        if temp_mem is not None:
            if temp_mem == 0:
                self.res.noTempMemory()
            else:
                self.res.setTempMemory(temp_mem)
        self._init_torch()
        # same explicit non-default stream as cuVS; knn_gpu uses torch's current stream
        self.stream = self._make_stream()
        # patches faiss.knn_gpu to accept torch tensors (contrib/torch_utils.py)
        importlib.import_module("faiss.contrib.torch_utils")

    def _engine_search(self, xq_t, xb_t):
        with self._lib():
            return faiss.knn_gpu(self.res, xq_t, xb_t, self.k, metric=schema.IP)

    def setup(self, xq):
        # cosine: normalise the queries once on the host; counted in setup_s
        t0 = now()
        if self.metric == "cosine":
            xq = base.normalized_queries(xq)
        norm_s = now() - t0
        return (norm_s if self.metric == "cosine" else 0.0) + super().setup(xq)
