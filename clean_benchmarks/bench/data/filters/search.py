"""Apply each block's filters before passing it to the search engine.

In gather mode, the reader compacts static-filter matches and the consumer
maps result IDs back to their original rows. In bitmap mode, the engine
applies filter masks, optionally skipping unnecessary searches or compacting
rows allowed by any query.

Also provides the NumPy masked backend (``--backend masked``), an exact
float32 engine. The float64 --verify reference is in bench/verify/oracle.py.

Tracks filter handoff time and the work actually searched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from bench.data.filters import duckdb_filter as dfm
from bench.data.filters import reader as _flt
from bench.timing import Stages, now
from bench.verify.oracle import Q_CHUNK, R_TILE, _mask_tile, _merge_topk, _topk_rows

FILTER_MODES = ("gather", "bitmap")
# --filter-skip (bitmap mode). A query is "dead" on a file when its mask allows no row there.
#   queries  search only live queries; skip a file with none (default)
#   files    search every query; skip only a file where nothing passes (as nova-bf does)
#   none     search every block with every query
SKIP_LEVELS = ("queries", "files", "none")
COMPACT_LEVELS = ("none", "union")
# --filter-compact union (per-query bitmap): the reader shrinks the block to the rows any
# query wants and re-packs the masks over them, as nova-bf does for all-filtered searches


# ---------------------------------------------------------------- exact top-k

def _unit_rows(x: np.ndarray, copy=True) -> np.ndarray:
    """Rows to unit L2 in float32 (zero rows stay zero), as faiss.normalize_L2."""
    x = np.array(x, dtype=np.float32, copy=True) if copy else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, n, out=x, where=n > 0)
    return x


def masked_topk(xq, xb, k, allowed=None, q_chunk=Q_CHUNK, r_tile=R_TILE):
    """Exact float32 top-k of xq . xb^T (pre-normalised for cosine). ``allowed``: None, a
    (rows,) bool mask, or ``allowed(q0, q1, r0, r1) -> bool (q1-q0, r1-r0)``. Tiled, so
    temporaries stay q_chunk x r_tile."""
    nq, n = xq.shape[0], xb.shape[0]
    D = np.full((nq, k), -np.inf, np.float32)
    I = np.full((nq, k), -1, np.int64)
    for q0 in range(0, nq, q_chunk):
        q1 = min(nq, q0 + q_chunk)
        Dq, Iq = D[q0:q1], I[q0:q1]
        for r0 in range(0, n, r_tile):
            r1 = min(n, r0 + r_tile)
            s = xq[q0:q1] @ xb[r0:r1].T
            if allowed is not None:
                m = allowed[None, r0:r1] if isinstance(allowed, np.ndarray) \
                    else allowed(q0, q1, r0, r1)
                s = np.where(m, s, -np.inf).astype(np.float32, copy=False)
            d, i = _topk_rows(s, k)
            i = np.where(i >= 0, i + r0, -1)
            Dq, Iq = (d, i) if r0 == 0 else _merge_topk(Dq, Iq, d, i, k)
        D[q0:q1], I[q0:q1] = Dq, Iq
    return D, I


class MaskedBackend:
    """``--backend masked``: exact float32 top-k in numpy, filter applied by masking scores to
    -inf. CPU only."""

    name = "masked"
    engine = "numpy"
    engine_version = np.__version__
    returns_distance = False
    events = False
    queries_mode = "resident"
    last_wait = (None, None)
    last_build = 0.0

    def __init__(self, k, dtype=np.float32, metric="ip"):
        if np.dtype(dtype) != np.float32:
            raise ValueError("masked backend is fp32 only")
        if metric not in ("ip", "cosine"):
            raise ValueError(f"masked backend: metric {metric!r} not supported (ip, cosine)")
        self.k, self.metric = k, metric
        self.stats = Stages()

    def reserve(self, values):
        return 0.0

    def alloc_peak_reset(self):
        pass

    def alloc_peak(self):
        return "n/a"

    def setup(self, xq):
        t0 = now()
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        self.xq = _unit_rows(xq) if self.metric == "cosine" else xq
        return now() - t0

    def _block(self, xb):
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        return _unit_rows(xb) if self.metric == "cosine" else xb

    def search(self, xb):
        """Unfiltered search (also the gather path, on a compacted block)."""
        t = now()
        out = masked_topk(self.xq, self._block(xb), self.k)
        self.stats.add("search", now() - t)
        return out

    def search_filtered(self, xb, fblock, queries=None):
        """Bitmap prefilter search for ``queries`` (indices into setup()'s queries; None = all)."""
        t = now()
        xq = self.xq if queries is None else self.xq[queries]
        if fblock.kind == "static":
            allowed = fblock.row_mask()
        else:
            mq = fblock.mask_of_query if queries is None else fblock.mask_of_query[queries]

            def allowed(q0, q1, r0, r1):
                return _mask_tile(fblock.masks[mq[q0:q1]], r0, r1)
        out = masked_topk(xq, self._block(xb), self.k, allowed)
        self.stats.add("search", now() - t)
        return out


# ---------------------------------------------------------------- per-block adapter

class FilterAdapter:
    """Shapes one block's search by its FilterBlock. ``search`` returns (D, I) for all nq
    queries with global ids, or None when the block is skipped."""

    def __init__(self, mode, nq, k, returns_distance=False, skip="queries", kind="per_query",
                 compact="none"):
        if mode not in FILTER_MODES:
            raise ValueError(f"--filter-mode {mode!r} not in {FILTER_MODES}")
        if skip not in SKIP_LEVELS:
            raise ValueError(f"--filter-skip {skip!r} not in {SKIP_LEVELS}")
        if mode == "gather" and skip != "queries":
            raise ValueError("--filter-skip applies to --filter-mode bitmap (gather "
                             "searches only passing rows by construction)")
        if compact not in COMPACT_LEVELS:
            raise ValueError(f"--filter-compact {compact!r} not in {COMPACT_LEVELS}")
        if compact != "none" and (mode != "bitmap" or kind != "per_query"):
            raise ValueError("--filter-compact union is for per-query --filter-mode bitmap "
                             "(a static filter's compaction is --filter-mode gather)")
        self.mode, self.nq, self.k, self.skip, self.kind = mode, nq, k, skip, kind
        self.compact = compact
        self.pad = np.inf if returns_distance else -np.inf
        self.stats = Stages()

    def _padded(self):
        return (np.full((self.nq, self.k), self.pad, np.float32),
                np.full((self.nq, self.k), -1, np.int64))

    # Three steps, so run_pass can overlap blocks (--d2h-overlap / --h2d-overlap):
    #   prepare(block) -> Prep | None (skipped)   masks -> engine input
    #   call / submit(backend, prep, ...)          the engine call
    #   finish(prep, D_l, I_l) -> (D, I)           local ids -> global, all nq rows
    # filter_handoff = prepare + finish time.

    def search(self, backend, block, on_device):
        prep = self.prepare(block)
        if prep is None:
            return None
        D_l, I_l = self.call(backend, prep, on_device)
        return self.finish(prep, D_l, I_l)

    def prepare(self, block):
        fb = block.filter
        if fb is None:
            raise RuntimeError(f"{block.path}: a filtered run got a block without filter masks")
        # a compacted block given to an adapter not expecting it would return wrong ids
        c = getattr(block, "compact", None)
        want = "gather" if self.mode == "gather" else \
            ("union" if self.compact == "union" else None)
        if c is not None and c.mode != want:
            raise RuntimeError(f"{block.path}: a block compacted by its reader ({c.mode}) given "
                               f"to a --filter-mode {self.mode} --filter-compact {self.compact} "
                               "adapter (the source's filter_compact and the adapter disagree)")
        self.stats.add("filter_blocks", 1)
        t = now()
        prep = self._prepare_gather(block, fb) if self.mode == "gather" \
            else self._prepare_bitmap(block, fb)
        if prep is None:
            self.stats.add("filter_skipped_files", 1)
        self.stats.add("filter_handoff", now() - t)
        return prep

    def _kw(self, prep, on_device, ahead):
        """Engine-call kwargs. On-device backends get the block's global offset and a
        DevicePost so results are shaped on the GPU; ``ahead`` is the next block's Prep
        (--h2d-overlap). Host backends get none: finish() shapes on the host."""
        kw = {}
        if on_device:
            kw = {"offset": prep.offset,
                  "post": DevicePost(prep.keep, prep.queries, self.nq)}
            prep.on_device = True
        if ahead is not None:
            kw["ahead"] = ahead.xb
            if ahead.fb is not None:
                kw["ahead_fblock"] = ahead.fb
            if on_device and ahead.keep is not None:
                kw["ahead_keep"] = ahead.keep
        return kw

    def call(self, backend, prep, on_device, ahead=None):
        """The engine call. ``ahead``: the next block's Prep (--h2d-overlap)."""
        kw = self._kw(prep, on_device, ahead)
        if prep.fb is None:
            if ahead is not None:
                kw.pop("ahead_fblock", None)
            return backend.search(prep.xb, **kw)
        return backend.search_filtered(prep.xb, prep.fb, prep.queries, **kw)

    def submit(self, backend, prep, ahead=None):
        """--d2h-overlap: first half of call(); the backend completes it later."""
        kw = self._kw(prep, True, ahead)
        if prep.fb is None:
            kw.pop("ahead_fblock", None)
            return backend.submit(prep.xb, **kw)
        return backend.submit_filtered(prep.xb, prep.fb, prep.queries, **kw)

    @staticmethod
    def _compacted(block, mode):
        """The reader's compaction of ``block``; raises if the block was not compacted by
        ``mode``."""
        c = getattr(block, "compact", None)
        if c is None or c.mode != mode:
            raise RuntimeError(f"{block.path}: {mode} compaction needs blocks compacted by "
                               f"their reader (ParquetSource filter_compact="
                               f"CompactSpec({mode!r})); got {'none' if c is None else c.mode}")
        if len(block.vectors) != len(c.keep):
            raise RuntimeError(f"{block.path}: {len(block.vectors)} compacted vectors vs "
                               f"{len(c.keep)} kept rows")
        return c

    def _prepare_gather(self, block, fb):
        if fb.kind != "static":
            raise RuntimeError("--filter-mode gather handles static filters only")
        keep = self._compacted(block, "gather").keep
        self.stats.add("filter_rows_kept", len(keep))
        self.stats.add("filter_allowed_pairs", len(keep) * self.nq)
        if len(keep) == 0:
            return None
        self.stats.add("filter_live_queries", self.nq)
        self.stats.add("filter_searched_queries", self.nq)
        return Prep(block.vectors, None, None, keep, block.offset)

    def _prepare_bitmap(self, block, fb):
        if fb.kind == "static":
            kept = int(fb.counts[0])
            self.stats.add("filter_rows_kept", kept)
            self.stats.add("filter_allowed_pairs", kept * self.nq)
            self.stats.add("filter_live_queries", self.nq if kept else 0)
            if kept == 0 and self.skip != "none":
                return None
            queries = None
        else:
            live = fb.live_queries()
            # allowed (query, row) pairs in this file, before skipping / compaction
            self.stats.add("filter_allowed_pairs",
                           int(np.asarray(fb.counts, np.int64)[fb.mask_of_query].sum()))
            self.stats.add("filter_live_queries", len(live))
            self.stats.add("filter_distinct_masks", len(fb.masks))
            if len(live) == 0 and self.skip != "none":
                return None
            queries = live if self.skip == "queries" and len(live) < self.nq else None
        self.stats.add("filter_searched_queries", self.nq if queries is None else len(queries))
        keep = None
        if self.compact == "union":
            # the reader compacted the block; popcounts (and the counts above) are unchanged
            c = self._compacted(block, "union")
            keep = c.keep
            self.stats.add("filter_compact_rows", len(keep))
            if len(keep) == c.n_rows:
                # left whole (union keeps more than COMPACT_MAX_FRAC of the rows)
                self.stats.add("filter_compact_whole_files", 1)
            if c.added:
                # padded up to COMPACT_MIN_ROWS
                self.stats.add("filter_compact_floor_files", 1)
                self.stats.add("filter_compact_floor_rows", c.added)
            if len(keep) == 0:
                return None   # no row any query wants (--filter-skip none): nothing to search
        return Prep(block.vectors, fb, queries, keep, block.offset)

    def finish(self, prep, D_l, I_l):
        t = now()
        if prep.on_device:
            # already shaped on the device
            self.stats.add("filter_handoff", now() - t)
            return D_l, I_l
        keep = prep.keep
        if keep is not None:   # compact-block ids -> file rows
            I_l = np.where(I_l >= 0, keep[np.where(I_l >= 0, I_l, 0)], -1) if len(keep) \
                else np.full_like(I_l, -1)   # nothing wanted (--filter-skip none)
        I_l = np.where(I_l >= 0, I_l + prep.offset, -1)
        if prep.queries is None:
            out = (D_l, I_l)
        else:
            D, I = self._padded()
            D[prep.queries], I[prep.queries] = D_l, I_l
            out = (D, I)
        self.stats.add("filter_handoff", now() - t)
        return out


@dataclass
class Prep:
    """One block's engine input: ``xb`` host vectors, ``fb`` the FilterBlock to filter with
    (None for gather: plain search), ``queries`` searched subset (None = all), ``keep`` file
    rows of compacted vectors (None = not compacted), ``offset`` the file's global row."""
    xb: np.ndarray
    fb: object
    queries: np.ndarray | None
    keep: np.ndarray | None
    offset: int
    on_device: bool = False   # results already shaped on the device


@dataclass
class DevicePost:
    """FilterAdapter.finish's steps for an on-device backend to run on the GPU."""
    keep: np.ndarray | None
    queries: np.ndarray | None
    nq: int


# re-exported from reader.py
COMPACT_MIN_ROWS = _flt.COMPACT_MIN_ROWS
_compact_union = _flt._compact_union


# ---------------------------------------------------------------- filter sources

def load_filter_config(path, search):
    """Read search ``search`` from nova-bf config ``path`` (env vars expanded as nova-bf does).
    Returns (filter, corpus date_fields, query date_fields, rows selector or None, spec)."""
    import yaml

    try:
        text = _expand_env(open(path).read())
    except OSError as e:
        raise SystemExit(f"filter config {path}: {e}") from e
    cfg = yaml.safe_load(text) or {}
    specs = [s for s in cfg.get("searches") or [] if s.get("name") == search]
    if len(specs) != 1:
        names = [s.get("name") for s in cfg.get("searches") or []]
        raise SystemExit(f"--filter-config {path}: no single search named {search!r} "
                         f"(searches: {names})")
    spec = specs[0]
    f = spec.get("filter")
    if not f or not any(f.get(g) for g in ("must", "should", "must_not")):
        raise SystemExit(f"--filter-search {search!r} has no filter")
    return (f, (cfg.get("corpus") or {}).get("date_fields"),
            (cfg.get("queries") or {}).get("date_fields"), spec.get("rows"), spec)


def _expand_env(text):
    """Copy of nova-bf's ``config.expand_env``: ``${VAR}`` / ``${VAR:-default}``; an empty
    variable counts as unset; no ``$$`` escape."""
    import re

    def repl(m):
        name, _, default = m.group(1).partition(":-")
        val = os.environ.get(name)
        if val:
            return val
        if ":-" in m.group(1):
            return default
        raise SystemExit(f"environment variable '{name}' referenced in config is not set; "
                         f"set it or supply a default with ${{{name}:-...}}")
    return re.sub(r"\$\{([^}]+)\}", repl, text)


def query_row_index(query_table, rows):
    """Sorted query-row indices selected by nova-bf ``rows: {column, isin}`` (None = all).
    Compares as strings exactly as nova-bf does, so a null reads "None" and an int column
    with nulls reads "1.0"."""
    if not rows:
        return None
    col = rows.get("column")
    isin = rows.get("isin") or ()
    if not col or not isin:
        raise SystemExit(f"query rows selector needs `column` and `isin`, got {rows}")
    if col not in query_table.column_names:
        raise SystemExit(f"query rows selector: the query file has no column {col!r}")
    vals = np.array(query_table.column(col).to_pylist())
    mask = np.isin(vals.astype(str), np.asarray(list(isin), dtype=str))
    return np.nonzero(mask)[0].astype(np.int64)
