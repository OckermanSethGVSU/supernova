"""Test helpers for the multivector tile budget (`params.multivector_budget_mib`).

nova-bf has ONE multivector batching setting: a MiB budget for the fp32 score
block, from which the tile shape (docs per slice, queries per tile) is derived
with an even split. Tests on tiny synthetic data need two things from it:

* `mib(elements)` — the exact MiB value for an element budget, so a test can
  say "64 score elements" (`64 * 4 / 2**20` is exact in binary floating point,
  so `int(mib * 2**20 / 4)` gives back exactly `elements`).
* `pin(monkeypatch, docs, queries)` — a TEST-ONLY override of the derived item
  caps, for tests whose intent is a specific, independent shape on each axis
  (for example many query tiles over one document slice), which a single
  budget cannot express. It replaces `compute._resolve_multivector_tiles`, the
  one place the shape is derived; the element budget is still enforced against
  the actual ragged token counts exactly as in production.
"""

from __future__ import annotations

from contextlib import contextmanager

FP32_BYTES = 4


def mib(elements: int) -> float:
    """MiB value whose element budget is exactly `elements`."""
    value = elements * FP32_BYTES / 2**20
    assert int(value * 2**20 / FP32_BYTES) == elements, elements
    return value


def pin(monkeypatch, docs_per_slice: int, queries_per_tile: int) -> None:
    """Force the derived multivector tile shape for the rest of the test."""
    from nova_bf import compute

    monkeypatch.setattr(
        compute, "_resolve_multivector_tiles",
        lambda *_args, **_kwargs: (docs_per_slice, queries_per_tile),
    )


@contextmanager
def pinned(tiles: tuple[int, int] | None):
    """`pin` as a context manager, for fixtures without `monkeypatch`
    (session/module scope). `None` pins nothing."""
    if tiles is None:
        yield
        return
    import pytest

    mp = pytest.MonkeyPatch()
    try:
        pin(mp, *tiles)
        yield
    finally:
        mp.undo()
