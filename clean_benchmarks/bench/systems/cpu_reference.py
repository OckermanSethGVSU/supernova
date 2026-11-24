"""CPU engines, one per modality: CpuBackend (dense, faiss.knn), SparseCpuBackend
(scipy) and MvCpuBackend (numpy MaxSim). Used on machines without a GPU and in
tests (leg -> engine -> clock)."""

import faiss  # must import after cudalibs.torch_first() has loaded torch's CUDA libraries
import numpy as np

from bench.record import schema
from bench.timing import Stages, now
from bench.systems import streaming as base
from bench.data.formats import multivector as multivec
from bench.data.formats import sparse


class CpuBackend:
    """Dense exact top-k on the CPU: faiss.knn on numpy.

    Cosine: queries normalised once in setup(); each block normalised into a
    reusable scratch (NormScratch) inside search(), timed as ``normalize``."""

    name = "cpu"
    engine = "faiss"
    engine_version = faiss.__version__
    returns_distance = False  # scores are similarities, largest first

    def __init__(self, k, dtype=np.float32, metric="ip"):
        if np.dtype(dtype) != np.float32:
            raise ValueError("cpu backend is fp32 only (faiss.knn has no fp16)")
        if metric not in schema.METRICS:
            raise ValueError(metric)
        self.k = k
        self.metric = metric
        self.queries_mode = "resident"
        self.stats = Stages()  # no device: h2d / widen stay 0
        self.scratch = base.NormScratch() if metric == "cosine" else None

    def reserve(self, values):
        """Pre-size the cosine scratch to the largest block's value count
        (before the passes); seconds spent, 0 with ip."""
        return self.scratch.reserve(values) if self.scratch is not None else 0.0

    # no device, so no CUDA events: device stage times are left blank, never 0
    events = False

    def setup(self, xq):
        t0 = now()
        self.xq = base.normalized_queries(xq) if self.metric == "cosine" else xq
        return now() - t0 if self.metric == "cosine" else 0.0

    last_wait = (None, None)  # (seconds, waited) of the last search: not measured
    last_build = 0.0  # no build stage

    def alloc_peak_reset(self):
        pass

    def alloc_peak(self):
        return "n/a"  # no device

    def search(self, xb):
        if xb.dtype != np.float32:
            raise TypeError(f"cpu backend got a {xb.dtype} block; widen on the "
                            "host")
        if self.metric == "cosine":
            t = now()
            xb = self.scratch.normalized(xb)
            self.stats.add("normalize", now() - t)
        return faiss.knn(self.xq, xb, self.k, metric=schema.IP)


# ---------------------------------------------------------- sparse backends
#
# --modality sparse: blocks are packed CSR (compressed sparse row) byte arrays,
# queries a sparse.Queries. Same timing buckets, merge and checks as dense.

class SparseCpuBackend:
    """Exact sparse top-k on the CPU (engine "scipy").

    Per block, scipy sparse products in tiles of corpus rows (float32
    accumulation), ties to the lowest id within a block; across blocks
    run_pass's ResultHeap applies FAISS's tie order. Values must already be
    float32 on the host."""

    name = "cpu"
    engine = "scipy"
    queries_mode = "n/a"
    returns_distance = False
    events = False
    last_wait = (None, None)
    last_build = 0.0

    def __init__(self, k, metric="ip"):
        import scipy
        if metric not in schema.METRICS:
            raise ValueError(metric)
        self.k, self.metric = k, metric
        self.engine_version = scipy.__version__
        self.stats = Stages()
        self._qm = {}

    def reserve(self, values):
        return 0.0

    def setup(self, q):
        """``q`` a sparse.Queries; cosine: its float64 row norms, once."""
        t0 = now()
        self.q = q
        self.q_norms = sparse.row_norms(q.matrix()) if self.metric == "cosine" else None
        return now() - t0 if self.metric == "cosine" else 0.0

    def alloc_peak_reset(self):
        pass

    def alloc_peak(self):
        return "n/a"

    def _qmat(self, n_cols):
        if n_cols not in self._qm:  # the queries at a block's width (once per width)
            self._qm = {n_cols: self.q.matrix(n_cols)}
        return self._qm[n_cols]

    def search(self, xb):
        c = sparse.unpack(xb)
        if c.values.dtype != np.float32:
            raise TypeError(f"cpu backend got {c.values.dtype} sparse values; widen on "
                            "the host")
        n_cols = max(self.q.n_cols, c.n_cols, 1)
        D, I = sparse.exact_topk(self._qmat(n_cols), sparse.block_matrix(c, n_cols),
                                 self.k, self.metric, np.float32, self.q_norms)
        return sparse.pad(D, I, self.k)


# ------------------------------------------------------ multi-vector backends
#
# --modality multivector: blocks are packed variable-length token arrays, queries
# a multivec.Queries. Same timing buckets, merge and checks as dense. A document or
# query with zero tokens scores -inf and never enters the top-k.

class MvCpuBackend:
    """Exact MaxSim top-k on the CPU (engine "numpy").

    Per block, float32 token score tiles (max over each document's tokens,
    sum over each query's), ties to the lowest id within a block. Tokens must
    be float32 on the host. Cosine: query tokens normalised once in setup,
    each block's inside search() (timed as ``normalize``)."""

    name = "cpu"
    engine = "numpy"
    engine_version = np.__version__
    queries_mode = "n/a"
    returns_distance = False
    events = False
    last_wait = (None, None)
    last_build = 0.0

    def __init__(self, k, metric="ip"):
        if metric not in schema.METRICS:
            raise ValueError(metric)
        self.k, self.metric = k, metric
        self.stats = Stages()

    def reserve(self, values):
        return 0.0

    def setup(self, q):
        t0 = now()
        self.q = q
        self.q_tok = multivec.unit_tokens(q.tokens) if self.metric == "cosine" else q.tokens
        return now() - t0 if self.metric == "cosine" else 0.0

    def alloc_peak_reset(self):
        pass

    def alloc_peak(self):
        return "n/a"

    def search(self, xb):
        r = multivec.unpack(xb)
        if r.tokens.dtype != np.float32:
            raise TypeError(f"cpu backend got {r.tokens.dtype} tokens; widen on the host "
                            "")
        tok = r.tokens
        if self.metric == "cosine":
            t = now()
            tok = multivec.unit_tokens(tok)
            self.stats.add("normalize", now() - t)
        D, I = multivec.exact_topk(self.q.offsets, self.q_tok, r.offsets, tok, self.k)
        return sparse.pad(D, I, self.k)
