"""The numpy reference for DuckFilter's match_text: the same (row, word) pairs from DuckDB,
then each phrase's words' row lists intersected in numpy, rarest first. A second, independent
way to combine them, only for the tests (bench uses the DuckDB bitmaps)."""

import numpy as np

from bench.data.filters import duckdb_filter as dfm


class NumpyTextFilter(dfm.DuckFilter):
    """DuckFilter whose phrase masks are combined in numpy (text_impl "numpy")."""

    def __init__(self, *a, **kw):
        kw.pop("text_impl", None)
        super().__init__(*a, **kw)
        self.text_impl = "numpy"

    def _phrase_masks(self, pairs, token_sets, wid, vocab, n, out) -> np.ndarray:
        try:
            t = self._q(pairs("DISTINCT"))
        finally:
            self.con.unregister("__vocab")
        r = t.column("r").to_numpy()
        w = t.column("wid").to_numpy()
        order = np.lexsort((r, w))
        r, w = r[order], w[order]
        starts = np.searchsorted(w, np.arange(len(vocab) + 1))
        for i, ts in enumerate(token_sets):
            ids = sorted((wid[x] for x in ts), key=lambda k: starts[k + 1] - starts[k])
            rows = r[starts[ids[0]]:starts[ids[0] + 1]]
            for k in ids[1:]:
                if not len(rows):
                    break
                rows = np.intersect1d(rows, r[starts[k]:starts[k + 1]], assume_unique=True)
            if len(rows):
                np.bitwise_or.at(out[i], rows >> 3, np.left_shift(1, rows & 7).astype(np.uint8))
        return out


IMPLS = {"bitmap": dfm.DuckFilter, "numpy": NumpyTextFilter}


def duck_filter(*a, text_impl="bitmap", **kw):
    """dfm.DuckFilter, or the numpy reference for text_impl="numpy"."""
    if text_impl not in IMPLS:
        raise ValueError(f"text_impl must be one of {'/'.join(IMPLS)}")
    return IMPLS[text_impl](*a, text_impl=text_impl, **kw)
