"""ResidentBackend.submit hands the uploaded block to _block_search without
keeping a reference of its own: when the block is cast (an fp16-stored block
widened to fp32 on the device), the upload must be
freed BEFORE the engine call, as when this code was inline in submit (else
the upload and its cast copy are both resident during the search: higher
alloc_peak, OOM risk). torch's CPU device stands in for the GPU."""

import gc
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bench.systems import streaming as base  # noqa: E402


class Probe(base.ResidentBackend):
    """A resident engine whose search records whether the raw upload is
    still alive at engine-call time."""

    engine = "probe"

    def __init__(self, k, dtype, normalize=False):
        super().__init__(k, dtype, "cpu", events=True)
        self._init_torch()
        self.normalize_blocks = normalize
        self.uploads, self.alive_at_search = [], []

    def _to_dev(self, x):
        t = super()._to_dev(x)
        if t.dim() == 2 and t.shape[0] != self.nq:   # a corpus block, not the queries
            t = t.clone()                             # a tensor nothing else holds
            self.uploads.append(weakref.ref(t))
        return t

    def setup(self, xq):
        self.nq = xq.shape[0]
        return super().setup(xq)

    def _engine_search(self, xq_t, xb_t):
        gc.collect()
        self.alive_at_search.append(self.uploads[-1]() is not None)
        s = xq_t.float() @ xb_t.float().T
        D, I = torch.topk(s, self.k, dim=1)
        return D, I


@pytest.mark.parametrize("stored, dtype, normalize, freed", [
    (np.float16, np.float32, False, True),   # widen on the device: upload freed
    (np.float32, np.float32, False, False),  # no cast: the upload IS the operand
])
def test_upload_freed_before_engine_call_when_cast(stored, dtype, normalize, freed):
    rng = np.random.default_rng(0)
    xq = rng.standard_normal((7, 16)).astype(np.float32 if dtype == np.float32 else np.float16)
    b = Probe(3, dtype, normalize=normalize)
    b.setup(xq)
    xb = rng.standard_normal((50, 16)).astype(stored)
    D, I = b.search(xb)
    assert D.shape == (7, 3) and I.shape == (7, 3)
    assert b.alive_at_search == [not freed], b.alive_at_search
