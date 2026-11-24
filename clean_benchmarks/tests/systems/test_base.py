"""bench/systems/base.py: a system's display label."""

import pytest

from bench.systems import base as SB


@pytest.mark.parametrize("entry, label", [
    ({"system": "cuvs"}, "stock"),
    ({"system": "cuvs", "sparse_project": "query-vocab"}, "stock +proj"),
    ({"system": "cuvs", "sparse_project": "off"}, "stock"),
    ({"system": "spare", "preset": "batched_dot"}, "batched_dot (ours: SPARe batched)"),
    ({"system": "spare", "preset": "batched_iterative", "sparse_project": "query-vocab"},
     "batched_iterative +proj (ours: SPARe batched)"),
    ({"system": "spare", "preset": "released_dot"}, "released_dot"),
    ({"system": "spare", "preset": "released_iterative_nosync"},
     "released_iterative_nosync (ours: SPARe iterative nosync fix)"),
    ({"system": "spare", "spare_mode": "iterative"}, "stock (ours: SPARe batched)"),
    ({"system": "pylate", "preset": "scores_batched"}, "scores_batched (ours: PyLate scores-batched)"),
    ({"system": "pylate", "preset": "rerank_released"}, "rerank_released"),
    ({"system": "pylate", "preset": "rerank_released_ownpad"}, "rerank_released_ownpad (ours: PyLate ownpad)"),
    ({"system": "cuvs", "filter_mode": "bitmap", "filter_compact": "union", "d2h": "pinned",
      "d2h_overlap": True}, "stock (ours: union compaction, overlapped copies)"),
    ({"system": "novabf", "config": "x.yaml"}, "stock"),
])
def test_the_display_label(entry, label):
    r = SB.resolve("x", entry)
    assert SB.display_label(r["preset"], r["knobs"]) == label
