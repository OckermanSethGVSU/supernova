"""Evaluate filters for each corpus file in the reader process.

Filtering runs after decoding, overlapping GPU execution. Masks are stored
alongside vectors in shared memory to avoid copying between processes.

Tracks filter evaluation, deduplication, compaction, and publication times.
Filter setup happens before timed passes. Filter-column reads are included
in decode or payload_read, depending on the read mode, and use the same
object version as the vectors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from bench.data.filters import duckdb_filter as dfm

MASK_ALIGN = 4096


def now():
    return time.perf_counter()


@dataclass
class FilterJob:
    """What a reader needs to build its own DuckFilter; picklable for spawned readers.
    ``queries``: the queries' filter columns (None for a static filter)."""
    filter: dict
    queries: pa.Table | None = None
    corpus_date_fields: dict | list | None = None
    query_date_fields: dict | list | None = None
    threads: int | None = None          # DuckDB threads per reader
    engine_kw: dict = field(default_factory=dict)   # e.g. prefilter_max_phrases
    selfcheck: bool = True

    def spec(self) -> dfm.FilterSpec:
        return dfm.parse_filter(self.filter, self.corpus_date_fields)

    @property
    def fields(self) -> list[str]:
        return self.spec().fields()

    def build(self, selfcheck: bool | None = None) -> dfm.DuckFilter:
        spec = self.spec()
        return dfm.DuckFilter(
            spec, self.queries if spec.per_query else None,
            corpus_date_fields=self.corpus_date_fields,
            query_date_fields=self.query_date_fields, threads=self.threads,
            selfcheck=self.selfcheck if selfcheck is None else selfcheck, **self.engine_kw)

    def max_masks(self) -> int:
        """Upper bound on one file's mask count: 1 for a static filter, else the number of
        distinct filter-value combinations. Builds the query side without self-checks."""
        eng = self.build(selfcheck=False)
        return 1 if not eng.pq_leaves else max(1, len(eng.combo_keys))


@dataclass
class FilterBlock:
    """One file's filter result.

    static:     ``masks`` is (1, nbytes), the file's row mask.
    per-query:  ``masks`` is (m, nbytes), the file's distinct masks; ``mask_of_query[q]`` is
                query q's mask and ``counts[j]`` mask j's popcount (0: its queries match no
                row of this file, i.e. are "dead" on it).
    Bits are little-endian (row r = bit r % 8 of byte r // 8); padding bits are 0."""
    kind: str                     # "static" | "per_query"
    n_rows: int
    masks: np.ndarray
    counts: np.ndarray
    mask_of_query: np.ndarray | None = None
    stats: dict | None = None

    @property
    def nbytes_per_mask(self) -> int:
        return self.masks.shape[1]

    def live_queries(self) -> np.ndarray:
        """Queries whose mask allows at least one row of this file (None if static)."""
        if self.kind == "static":
            return None
        return np.flatnonzero(self.counts[self.mask_of_query] > 0)

    def query_masks_packed(self, queries=None) -> np.ndarray:
        """(len(queries), nbytes) packed masks, expanded per query (allocates)."""
        if self.kind == "static":
            raise ValueError("a static filter has one row mask, not per-query masks")
        idx = self.mask_of_query if queries is None else self.mask_of_query[queries]
        return self.masks[idx]

    def row_mask(self) -> np.ndarray:
        """Static filter: the (n_rows,) bool row mask."""
        if self.kind != "static":
            raise ValueError("per-query filter: use masks / mask_of_query")
        return dfm.unpack(self.masks[0], self.n_rows)


# Parts of filter_eval (seconds per file): Stages key -> DuckFilter stats key. They do not sum
# exactly to filter_eval (call overhead is left out).
#   filter_prep    register the file's table, plan the distinct leaves
#   filter_sql     match / range leaves (SQL boolean columns)
#   filter_text    match_text leaves (tokenise, then intersect words per row)
#   filter_combine combine leaf bitmaps into the final masks
FILTER_STAGES = {"filter_prep": "prep_s", "filter_sql": "sql_s", "filter_text": "text_s",
                 "filter_combine": "combine_s"}
# filter_sql / filter_text each split into two parts that sum to it:
#   *_query  time inside DuckDB (statements and their Arrow export)
#   *_post   our code around them (registering inputs, packing / converting results)
FILTER_SPLITS = {"filter_sql_query": "sql_query_s", "filter_sql_post": "sql_post_s",
                 "filter_text_query": "text_query_s", "filter_text_post": "text_post_s"}
# counts summed over files: match_text columns evaluated, and how many used the lean
# tokenisation (duckdb_filter.lean_tokens_ok)
FILTER_COUNTS = {"filter_text_cols": "text_cols", "filter_text_lean_cols": "text_lean_cols"}

# max fraction of MemAvailable a filtered process-reader ring (shared memory) may use
MASK_RING_MEM_FRAC = 0.5


def mask_capacity(max_masks: int, max_rows: int) -> int:
    """Bytes a slot reserves after its vectors for masks: ``max_masks`` masks of the largest
    file, so no file overflows. An overflowing file sends its masks in the reply instead
    (counted as filter_mask_overflows, which must be 0)."""
    return max(1, max_masks * dfm.nbytes(max_rows))


def mem_available():
    """MemAvailable in bytes (/proc/meminfo), or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def aligned(n: int, a: int = MASK_ALIGN) -> int:
    return -(-int(n) // a) * a


def evaluate(flt: dfm.DuckFilter, table: pa.Table, stats=None, rec=None):
    """Run the filter on one file's table. Returns (masks (m, nb), info without the masks)."""
    t0 = now()
    res = flt.evaluate(table)
    t1 = now()
    if res.per_query:
        masks, mask_of_query, counts = res.distinct_masks()
        info = {"kind": "per_query", "mask_of_query": mask_of_query.astype(np.int32)}
    else:
        masks = res.static[None, :]
        counts = dfm.popcount_rows(masks)
        info = {"kind": "static"}
    t2 = now()
    info.update(n_rows=res.n_rows, m=int(masks.shape[0]), nb=int(masks.shape[1]),
                counts=np.asarray(counts, np.int64), stats=res.stats)
    sub = {k: float(res.stats.get(src, 0.0))
           for k, src in {**FILTER_STAGES, **FILTER_SPLITS}.items()}
    if stats is not None:
        stats.add("filter_eval", t1 - t0)
        stats.add("filter_dedup", t2 - t1)
        for k, v in sub.items():
            stats.add(k, v)
        for k, src in FILTER_COUNTS.items():
            stats.add(k, int(res.stats.get(src, 0)))
    if rec is not None:
        rec.update(filter_eval=t1 - t0, filter_dedup=t2 - t1, **sub)
    return masks, info


def publish_masks(masks: np.ndarray, mm, offset: int, capacity: int, path="", stats=None,
                  rec=None) -> bool:
    """Copy ``masks`` into slot mapping ``mm`` at ``offset``. False (nothing written) when they
    exceed ``capacity``: the caller then sends them in its reply."""
    t = now()
    if masks.nbytes > capacity:
        if stats is not None:
            stats.add("filter_mask_overflows", 1)
            stats.add("filter_overflow_bytes", int(masks.nbytes))
        if rec is not None:
            rec["filter_publish"] = now() - t
        return False
    if masks.size:
        dst = np.ndarray(masks.shape, np.uint8, buffer=mm, offset=offset)
        np.copyto(dst, masks)
        del dst
    dt = now() - t
    if stats is not None:
        stats.add("filter_publish", dt)
    if rec is not None:
        rec["filter_publish"] = dt
    return True


def finish_timing(stats=None, rec=None):
    """Per file: ``filter`` = eval + dedup + compact + publish (compact is 0 when off)."""
    if rec is not None:
        rec["filter"] = rec.get("filter_eval", 0) + rec.get("filter_dedup", 0) \
            + rec.get("filter_compact", 0) + rec.get("filter_publish", 0)
        if stats is not None:
            stats.add("filter", rec["filter"])


# ------------------------------------------------- reader-side compaction
# Static gather and per-query union compaction keep only rows needed for search.
# The reader moves these rows to the front of the block and records their
# original positions. For union, it also updates the filter masks.
# The consumer maps result IDs back to the original rows.
# Compaction time is recorded as filter_compact.

COMPACT_MODES = ("gather", "union")
# max bytes compact_rows gathers per step
COMPACT_CHUNK_BYTES = 32 << 20
# --filter-compact union never shrinks a file below this many rows (a file no query wants is
# skipped). Reason: cuVS 26.10's filtered search returns wrong results for queries with fewer
# than k allowed rows when raft's CSR select_k uses its warpsort path (k <= 256). The engine
# uses k_eff = min(k, rows), so >= 257 rows keeps k_eff > 256 (the exact radix path) for
# k >= 257. Padding rows have mask bit 0 for every query, so they are never returned.
COMPACT_MIN_ROWS = 257
MASK_CHUNK = 64    # masks re-packed at a time (bounds the unpacked temporary)
# --filter-compact union leaves a file whole when it would keep more than this fraction of
# rows: re-packing and moving most rows buys almost no search saving.
COMPACT_MAX_FRAC = 0.5


@dataclass(frozen=True)
class CompactSpec:
    """What a reader compacts; picklable for reader processes. ``mode``: gather (a static
    filter's passing rows) or union (rows any searched query wants; ``skip`` is --filter-skip,
    and "queries" leaves out dead queries)."""
    mode: str
    skip: str = "queries"
    max_frac: float = COMPACT_MAX_FRAC   # union only: above it, the file stays whole

    def __post_init__(self):
        if self.mode not in COMPACT_MODES:
            raise ValueError(f"compaction {self.mode!r} not in {COMPACT_MODES}")
        if not 0 < self.max_frac <= 1:
            raise ValueError(f"compaction max_frac {self.max_frac!r} not in (0, 1]")


@dataclass
class Compacted:
    """A block's compaction: ``vectors[i]`` is file row ``keep[i]`` (ascending int64),
    ``n_rows`` the file's row count, ``added`` the padding rows from COMPACT_MIN_ROWS."""
    mode: str
    keep: np.ndarray
    n_rows: int
    added: int = 0


def _compact_union(fb, queries=None, min_rows=COMPACT_MIN_ROWS, max_frac=1.0):
    """Returns (keep, FilterBlock over the kept rows, padding rows added).

    ``keep`` is the OR of the masks the searched queries use, padded to ``min_rows`` with the
    lowest unwanted rows. Masks are re-packed over ``keep`` by a boolean compress (same bits as
    ``[:, keep]`` since keep is ascending, ~4x faster); popcounts are unchanged. Above
    ``max_frac`` of the rows, the file stays whole (keep = every row, masks unchanged)."""
    used = np.unique(fb.mask_of_query if queries is None else fb.mask_of_query[queries])
    union = np.bitwise_or.reduce(fb.masks[used], axis=0) if len(used) else \
        np.zeros(fb.masks.shape[1], np.uint8)
    wanted = dfm.unpack(union, fb.n_rows)
    keep = np.flatnonzero(wanted)
    if fb.n_rows and len(keep) > max_frac * fb.n_rows:
        return np.arange(fb.n_rows, dtype=np.int64), fb, 0
    added = 0
    if 0 < len(keep) < min(min_rows, fb.n_rows):
        added = min(min_rows, fb.n_rows) - len(keep)
        extra = np.flatnonzero(~wanted)[:added]
        keep = np.sort(np.concatenate([keep, extra]))
        wanted = np.zeros(fb.n_rows, bool)
        wanted[keep] = True
    n2 = len(keep)
    masks = np.zeros((len(fb.masks), dfm.nbytes(n2)), np.uint8)
    for j0 in range(0, len(used), MASK_CHUNK):
        js = used[j0:j0 + MASK_CHUNK]
        if n2:
            bits = np.compress(wanted, dfm.unpack(fb.masks[js], fb.n_rows), axis=1)
            masks[js] = np.packbits(bits, axis=1, bitorder="little")
    counts = dfm.popcount_rows(masks)
    return keep, FilterBlock("per_query", n2, masks, counts, fb.mask_of_query, fb.stats), added


def plan_compaction(spec: CompactSpec, masks: np.ndarray, info: dict):
    """Which rows the block keeps. Returns (keep ascending int64, masks, info, added).
    gather: masks / info unchanged, keep = passing rows. union: masks re-packed over keep and
    info updated to match. Same rows FilterAdapter searches."""
    if spec.mode == "gather":
        if info["kind"] != "static":
            raise RuntimeError("--filter-mode gather handles static filters only")
        keep = np.flatnonzero(dfm.unpack(masks[0], info["n_rows"])).astype(np.int64)
        return keep, masks, info, 0
    if info["kind"] != "per_query":
        raise RuntimeError("--filter-compact union is for per-query filters")
    fb = block_from(info, masks)
    live = fb.live_queries()
    queries = live if spec.skip == "queries" and len(live) < len(fb.mask_of_query) else None
    keep, fc, added = _compact_union(fb, queries, max_frac=spec.max_frac)
    out = dict(info, n_rows=fc.n_rows, m=int(fc.masks.shape[0]), nb=int(fc.masks.shape[1]),
               counts=np.asarray(fc.counts, np.int64))
    return keep.astype(np.int64, copy=False), fc.masks, out, int(added)


def compact_rows(x: np.ndarray, keep: np.ndarray, chunk_bytes: int = COMPACT_CHUNK_BYTES):
    """In place ``x[:len(keep)] = x[keep]``; returns that view. ``keep`` strictly ascending.

    Safe because keep[t] >= t: a chunk's writes (positions i..j-1) are never a later chunk's
    sources (keep[t] >= t >= j); each chunk gathers into a temporary of ``chunk_bytes`` first.
    Rows already in place are skipped, so keeping every row copies nothing."""
    m = len(keep)
    if m and (int(keep[-1]) >= len(x) or (m > 1 and not np.all(keep[1:] > keep[:-1]))):
        raise RuntimeError("compact_rows: keep must be strictly ascending rows of the block")
    if m == 0:
        return x[:0]
    # rows already in place: keep[t] - t is >= 0 and non-decreasing, 0 on the prefix
    p = int(np.searchsorted(keep - np.arange(m), 0, side="right"))
    step = max(1, chunk_bytes // max(1, x[0].nbytes))
    for i in range(p, m, step):
        j = min(m, i + step)
        x[i:j] = x[keep[i:j]]
    return x[:m]


def compact_block(spec: CompactSpec, x: np.ndarray, masks: np.ndarray, info: dict,
                  stats=None, rec=None, in_place=False):
    """Plan the compaction, then compact ``x`` in place if writable, else (read-only
    Arrow-backed array) into a new array; ``in_place`` (a ring slot) forbids the copy.
    Returns (vectors, masks, info, Compacted); timed as ``filter_compact``."""
    t = now()
    n_rows = int(info["n_rows"])
    if len(x) != n_rows:
        raise RuntimeError(f"compaction: {len(x)} vectors vs {n_rows} filter rows")
    keep, masks, info, added = plan_compaction(spec, masks, info)
    if x.flags.writeable and x.flags.c_contiguous:
        x = compact_rows(x, keep)
    elif in_place:
        raise RuntimeError("compaction: the block's buffer is not writable in place")
    else:
        x = np.take(x, keep, axis=0)
    dt = now() - t
    if stats is not None:
        stats.add("filter_compact", dt)
    if rec is not None:
        rec["filter_compact"] = dt
    return x, masks, info, Compacted(spec.mode, keep, n_rows, added)


def block_from(info: dict, masks: np.ndarray) -> FilterBlock:
    return FilterBlock(kind=info["kind"], n_rows=info["n_rows"], masks=masks,
                       counts=info["counts"], mask_of_query=info.get("mask_of_query"),
                       stats=info.get("stats"))
