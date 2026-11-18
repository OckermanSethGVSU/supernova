"""Exact block-max skipping for sparse posting-list scoring
(prototype, on by default; `params.sparse_block_skip: off` disables it).

Skip document sub-blocks whose maximum possible score is below the
query's running top-k threshold (raised to the cross-worker threshold
under `threshold_exchange`). Live cells are scored exactly as in
the unskipped kernel, preserving the final results.

Steps:
1. Prepare the query vocabulary and FP32 rounding margins once per run.
2. Compute maximum token weights per sub-block using dense hot-token
   tables or cold-token posting lists.
3. Sum the query-weighted maxima to bound each sub-block's scores.
   Mark blocks below the threshold as dead, including rounding margins.
4. Skip scoring dead blocks. When write elision is enabled, leave their
   cells unwritten and mask them out during top-k selection.

Requires unique (document, token) pairs, as does the regular posting
kernel. NOVA_BF_SPARSE_BLOCK_SKIP_CHECK=1 verifies this for debugging.

Check each compiled variant on its first launch, twice: a model probe
runs the same variant on adversarial inputs built to break the bound's
floating-point model and checks every inequality its proof chains
(`model_probe`); then sampled real rows are checked against cuSPARSE:
live scores must agree within FP32 rounding bounds, overlap masks
must match, and skipped scores must be below the threshold.

A failed check or kernel error disables skipping for the run and
re-scores the slice without skipping. OOM only disables skipping
for the current slice.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MAX_QUERY_NNZ = 40000     # a query with more nonzeros is never skipped (see `_bound`)
MAX_NSUB = 8192           # sub-blocks per slice the bound kernel holds in registers
BLOCK_P = 256

_MODE = "off"             # effective: off | on | overhead
_CONFIGURED = "off"       # what params.sparse_block_skip said
_ENV_OVERRIDE = False
_SUB = 64
_ELIDE = True
_OFF: str | None = None   # why skipping was switched off for the run
_STATS: dict = {}
_DECLINES: dict = {}
_GATES: dict = {}
_QPLAN: dict = {}
_ROWS: dict = {}          # member -> rows folded so far (S3 warm-up gate)
_ADAPT: dict = {}
_ONES: dict = {}          # cached all-live masks (overhead mode)
# Environment-only knobs, parsed and validated once in `configure`.
_ENV: dict = {}
_ENV_DEFAULTS = {"min_dead": 0.15, "reprobe": 16, "budget_mib": 2048.0,
                 "check": False}
INT32_LIMIT = (1 << 31) - 1

try:
    import triton
    import triton.language as tl

    # Keep Triton specialization for alignment and masking, but not for nblk.
    # Each compiled variant is checked on its first launch.
    @triton.jit(do_not_specialize=["nblk"])
    def _posting_scores_hot_skip(QCROW, QCOL, QVAL, HOTIDX, HOT, BPTR, CROW, CVAL,
                                 OUT, LIVE, n_rows, stride_out, nblk, n_sub,
                                 DB: tl.constexpr, BLOCK_P: tl.constexpr,
                                 SUB: tl.constexpr, ELIDE: tl.constexpr):
        """Score only live document sub-blocks using the hot-token kernel.

        Load sub-block flags once into a register bitmask, then choose a path:
        - All dead: skip the block or write -inf.
        - All live: run the original hot-token kernel unchanged.
        - Partly dead: mask hot-token loads and cold posting updates.

        Live scores are bit-identical to the unskipped kernel. Dead cells
        are left unwritten when ELIDE is enabled, otherwise filled with -inf.
        """
        NSB: tl.constexpr = DB // SUB
        q = tl.program_id(0)
        b = tl.program_id(1)
        base = OUT + q.to(tl.int64) * stride_out
        lrow = LIVE + q.to(tl.int64) * n_sub
        lo = b * DB
        # Token plan first, so it is not serialised behind the flag load.
        j0 = tl.load(QCROW + q)
        j1 = tl.load(QCROW + q + 1)
        sbi = b * NSB + tl.arange(0, NSB)
        inb = sbi < n_sub
        lv = (tl.load(lrow + sbi, mask=inb, other=0) != 0) & inb          # bool [NSB]
        bits = tl.sum(lv.to(tl.int32) << tl.arange(0, NSB))                  # bitmask
        # Use the bitmask to detect all-dead and all-live blocks.
        # Use int64 to avoid overflow when building a 32-bit mask.
        n_in = tl.minimum(n_sub - b * NSB, NSB).to(tl.int64)
        inmask = (tl.full((), 1, tl.int64) << n_in) - 1
        all_live = (bits.to(tl.int64) & 0xFFFFFFFF) == inmask
        if bits == 0:
            if not ELIDE:
                o0 = lo + tl.arange(0, DB)
                tl.store(base + o0, tl.full([DB], float("-inf"), tl.float32),
                         mask=o0 < n_rows)
            return
        if all_live:
            # --- all live: verbatim `_posting_scores_hot` -------------------
            o = lo + tl.arange(0, DB)
            om = o < n_rows
            acc = tl.zeros([DB], tl.float32)
            hit = tl.zeros([DB], tl.int32)
            for j in range(j0, j1):
                h = tl.load(HOTIDX + tl.load(QCOL + j))
                if h >= 0:
                    qv = tl.load(QVAL + j)
                    tv = tl.load(HOT + h.to(tl.int64) * n_rows + o, mask=om, other=0.0)
                    hit = hit | (tv != 0.0).to(tl.int32)
                    acc = tl.fma(tl.broadcast_to(qv, [DB]), tv, acc)
            tl.store(base + o, tl.where(hit != 0, acc, float("-inf")), mask=om)
            for j in range(j0, j1):
                t = tl.load(QCOL + j)
                h = tl.load(HOTIDX + t)
                if h < 0:
                    tl.debug_barrier()
                    qv = tl.load(QVAL + j)
                    k = t.to(tl.int64) * nblk + b
                    p0 = tl.load(BPTR + k)
                    p1 = tl.load(BPTR + k + 1)
                    for p in range(p0, p1, BLOCK_P):
                        offs = p + tl.arange(0, BLOCK_P)
                        m = offs < p1
                        d = tl.load(CROW + offs, mask=m, other=0)
                        cv = tl.load(CVAL + offs, mask=m, other=0.0)
                        cur = tl.load(base + d, mask=m, other=0.0, cache_modifier=".cg")
                        a = tl.where(cur == float("-inf"), 0.0, cur)
                        tl.store(base + d,
                                 tl.fma(tl.broadcast_to(qv, [BLOCK_P]), cv, a), mask=m)
        else:
            # names differ from the all-live branch: Triton requires a value
            # assigned in both if/else arms to have one type ([DB] vs [NSB, SUB])
            # --- partly dead -------------------------------------------------
            o_t = (lo + tl.arange(0, NSB)[:, None] * SUB
                 + tl.arange(0, SUB)[None, :])                              # [NSB, SUB]
            om_t = o_t < n_rows
            lm = om_t & tl.broadcast_to(lv[:, None], [NSB, SUB])
            acc_t = tl.zeros([NSB, SUB], tl.float32)
            hit_t = tl.zeros([NSB, SUB], tl.int32)
            for j in range(j0, j1):
                h = tl.load(HOTIDX + tl.load(QCOL + j))
                if h >= 0:
                    qv = tl.load(QVAL + j)
                    tv_t = tl.load(HOT + h.to(tl.int64) * n_rows + o_t, mask=lm, other=0.0)
                    hit_t = hit_t | (tv_t != 0.0).to(tl.int32)
                    acc_t = tl.fma(tl.broadcast_to(qv, [NSB, SUB]), tv_t, acc_t)
            res = tl.where(hit_t != 0, acc_t, float("-inf"))
            if ELIDE:
                tl.store(base + o_t, res, mask=lm)
            else:
                tl.store(base + o_t, tl.where(lm, res, float("-inf")), mask=om_t)
            for j in range(j0, j1):
                t = tl.load(QCOL + j)
                h = tl.load(HOTIDX + t)
                if h < 0:
                    tl.debug_barrier()
                    qv = tl.load(QVAL + j)
                    k = t.to(tl.int64) * nblk + b
                    p0 = tl.load(BPTR + k)
                    p1 = tl.load(BPTR + k + 1)
                    for p in range(p0, p1, BLOCK_P):
                        offs = p + tl.arange(0, BLOCK_P)
                        m = offs < p1
                        d = tl.load(CROW + offs, mask=m, other=0)
                        # the posting's sub-block within this program, from
                        # the register bitmask (no memory access)
                        mm = m & (((bits >> ((d // SUB - b * NSB) & 31)) & 1) != 0)
                        cv = tl.load(CVAL + offs, mask=mm, other=0.0)
                        cur = tl.load(base + d, mask=mm, other=0.0, cache_modifier=".cg")
                        a = tl.where(cur == float("-inf"), 0.0, cur)
                        tl.store(base + d,
                                 tl.fma(tl.broadcast_to(qv, [BLOCK_P]), cv, a), mask=mm)

    @triton.jit
    def _block_max(QTOK, HOTIDX, HOT, PTR, CROW, CVAL, BM, n_rows, n_sub,
                   SUB: tl.constexpr, BLOCK_P: tl.constexpr):
        """Program c: BM[c, :] for query token QTOK[c], as fp32 bits (int32,
        zero-initialised = +0.0). Weights are strictly positive here, so the
        int32 order of their bit patterns is their value order."""
        c = tl.program_id(0)
        t = tl.load(QTOK + c)
        h = tl.load(HOTIDX + t)
        row = BM + c.to(tl.int64) * n_sub
        if h >= 0:
            hb = HOT + h.to(tl.int64) * n_rows
            for sb in range(0, n_sub):
                o = sb * SUB + tl.arange(0, SUB)
                v = tl.load(hb + o, mask=o < n_rows, other=0.0)
                tl.store(row + sb, tl.max(v, axis=0).to(tl.int32, bitcast=True))
        else:
            p0 = tl.load(PTR + t)
            p1 = tl.load(PTR + t + 1)
            for p in range(p0, p1, BLOCK_P):
                offs = p + tl.arange(0, BLOCK_P)
                m = offs < p1
                d = tl.load(CROW + offs, mask=m, other=0)
                v = tl.load(CVAL + offs, mask=m, other=0.0)
                # an order-free max: no ordering with other memory is needed
                tl.atomic_max(row + d // SUB, v.to(tl.int32, bitcast=True), mask=m,
                              sem="relaxed")

    @triton.jit
    def _bound(QCROW, QCOL, QVAL, QMAP, BM, F, SIG, THR, LIVE, ROWANY, DEADCNT,
               n_sub, NSP: tl.constexpr):
        """Compute a block-max score bound and skip blocks below the top-k threshold.

        For positive weights, the sum of query-weighted block maxima bounds
        every document's exact dot product. FP32 rounding can underestimate
        this bound or overestimate a document's score.

        To account for both errors, inflate the computed bound with a
        multiplicative margin F and additive slack SIG:
            F = 1 + (m + 1) * 2^-21
            SIG = (m + 1) * 2^-120
        where m is the number of stored query tokens. These margins cover
        FP32 accumulation error and underflow, including flush-to-zero,
        for m <= MAX_QUERY_NNZ.

        Skip only when the adjusted bound is strictly below the threshold,
        preserving ties. Invalid thresholds, unfinished top-k states, and
        queries exceeding MAX_QUERY_NNZ prevent skipping. The posting
        kernel's magnitude guard prevents overflow.
        """
        q = tl.program_id(0)
        s = tl.arange(0, NSP)
        sm = s < n_sub
        ub = tl.zeros([NSP], tl.float32)
        j0 = tl.load(QCROW + q)
        j1 = tl.load(QCROW + q + 1)
        for j in range(j0, j1):
            c = tl.load(QMAP + tl.load(QCOL + j)).to(tl.int64)
            qv = tl.load(QVAL + j)
            bm = tl.load(BM + c * n_sub + s, mask=sm, other=0.0)
            ub = tl.fma(tl.broadcast_to(qv, [NSP]), bm, ub)
        t = ub * tl.load(F + q) + tl.load(SIG + q)
        hi = (tl.load(THR + q) >> 32).to(tl.int32)
        bits = tl.where(hi >= 0, hi, hi ^ 0x7FFFFFFF)
        thr = bits.to(tl.float32, bitcast=True)
        dead = sm & (t < thr)
        live = sm & (dead == 0)
        tl.store(LIVE + q.to(tl.int64) * n_sub + s, live.to(tl.int8), mask=sm)
        tl.store(ROWANY + q, (tl.sum(live.to(tl.int32)) > 0).to(tl.int8))
        tl.store(DEADCNT + q, tl.sum(dead.to(tl.int32)))

except Exception as exc:  # no triton, or an API that moved
    triton = tl = None
    _posting_scores_hot_skip = _block_max = _bound = None
    _UNAVAILABLE = f"{type(exc).__name__}: {exc}"
else:
    _UNAVAILABLE = None


class SkipCheckMismatch(RuntimeError):
    """A skip kernel variant's first-launch check disagreed with cuSPARSE
    (a skipped cell that could have entered the top-k, or a wrong live
    cell). Skipping is turned off; the slice is re-scored unskipped."""


# --- run state ---------------------------------------------------------------

def reset() -> None:
    global _MODE, _CONFIGURED, _ENV_OVERRIDE, _SUB, _ELIDE, _OFF
    _MODE = _CONFIGURED = "off"
    _ENV_OVERRIDE = False
    _SUB, _ELIDE, _OFF = 64, True, None
    _STATS.clear()
    _STATS.update({"slices": 0, "overhead_slices": 0, "subblocks": 0, "dead": None,
                   "programs": 0, "dead_programs": None,
                   "checks": 0, "checked_dead_cells": 0, "check_failures": 0,
                   "model_probes": 0, "model_probe_tokens": 0,
                   "model_probe_failures": 0, "model_probe_worst": None,
                   "elided_slices": 0, "materialized": 0, "masked_topk": 0,
                   "rescored": 0, "adaptive_declines": 0, "adaptive_probes": 0})
    for d in (_DECLINES, _GATES, _QPLAN, _ROWS, _ONES):
        d.clear()
    _ADAPT.clear()
    # one pinned slot: (host buffer, event, total) of the latest dead count
    _ADAPT.update({"probe": None, "frac": None, "since_probe": 0})
    _ENV.clear()
    _ENV.update(_ENV_DEFAULTS)


reset()

_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no", ""}


def _parse_mode(v) -> str:
    if v is True:
        return "on"
    if v is False or v is None:
        return "off"
    s = str(v).strip().lower()
    if s in _TRUE:
        return "on"
    if s in _FALSE:
        return "off"
    if s == "overhead":
        return "overhead"
    raise ValueError(f"NOVA_BF_SPARSE_BLOCK_SKIP / params.sparse_block_skip must be "
                     f"off/on/overhead, got {v!r}")


def _parse_bool(v, what: str) -> bool:
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"{what} must be one of 1/true/on/yes or 0/false/off/no, got {v!r}")


def configure(mode="off", sub: int = 64, elide: bool = True) -> None:
    """Apply sparse block-skipping settings, giving environment overrides
    priority. Configure the mode, sub-block width, and write elision.
    Read additional environment-only settings via read_env."""
    global _MODE, _CONFIGURED, _ENV_OVERRIDE, _SUB, _ELIDE
    _CONFIGURED = _parse_mode(mode)
    env = os.environ.get("NOVA_BF_SPARSE_BLOCK_SKIP")
    _ENV_OVERRIDE = env is not None
    eff = _parse_mode(env) if env is not None else _CONFIGURED
    env_sub = os.environ.get("NOVA_BF_SPARSE_BLOCK_SKIP_SUB")
    try:
        sub = int(env_sub) if env_sub else int(sub)
    except ValueError:
        raise ValueError(f"NOVA_BF_SPARSE_BLOCK_SKIP_SUB={env_sub!r} is not an int") from None
    if sub < 1 or sub & (sub - 1):
        raise ValueError(f"NOVA_BF_SPARSE_BLOCK_SKIP_SUB / params.sparse_block_skip_sub "
                         f"must be a power of two, got {sub}")
    env_el = os.environ.get("NOVA_BF_SPARSE_BLOCK_SKIP_ELIDE")
    if env_el is not None:
        elide = _parse_bool(env_el, "NOVA_BF_SPARSE_BLOCK_SKIP_ELIDE")
    _ENV.clear()
    _ENV.update(read_env())
    _MODE, _SUB, _ELIDE = eff, sub, bool(elide)


def read_env() -> dict:
    """Read and validate environment settings for the minimum skipped fraction,
    reprobe interval, memory budget, and debug checks. Raise on invalid values."""
    env = dict(_ENV_DEFAULTS)

    def num(name, cast, lo, hi=None):
        raw = os.environ.get(f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}")
        if raw is None:
            return None
        try:
            v = cast(raw)
        except ValueError:
            raise ValueError(f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}={raw!r} is not a "
                             f"{cast.__name__}") from None
        if v < lo or (hi is not None and v > hi) or v != v:
            raise ValueError(f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}={raw!r} is out of range")
        return v

    for key, name, cast, lo, hi in (("min_dead", "MIN_DEAD", float, 0.0, 1.0),
                                    ("reprobe", "REPROBE", int, 1, None),
                                    ("budget_mib", "BUDGET_MIB", float, 0.0, None)):
        v = num(name, cast, lo, hi)
        if v is not None:
            env[key] = v
    for key, name in (("check", "CHECK"),):
        raw = os.environ.get(f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}")
        if raw is not None:
            env[key] = _parse_bool(raw, f"NOVA_BF_SPARSE_BLOCK_SKIP_{name}")
    return env


def mode() -> str:
    return _MODE


def active() -> bool:
    return _MODE != "off" and _OFF is None


def disable(reason: str) -> None:
    global _OFF
    if _OFF is None:
        _OFF = reason
        logger.warning("sparse block skipping is off for the rest of this run: %s. "
                       "The posting kernel keeps running without it.", reason)


def note_gate(reason: str) -> None:
    _GATES[reason] = _GATES.get(reason, 0) + 1


def decline(reason: str) -> None:
    _DECLINES[reason] = _DECLINES.get(reason, 0) + 1


def rows_seen(member: int) -> int:
    return _ROWS.get(member, 0)


def add_rows(member: int, n: int) -> None:
    _ROWS[member] = _ROWS.get(member, 0) + int(n)


def _min_dead() -> float:
    return float(_ENV["min_dead"])


def _reprobe() -> int:
    return int(_ENV["reprobe"])


def usage() -> dict:
    """Skip statistics; device counters are read back here, once."""
    st = _STATS
    dead = 0 if st["dead"] is None else int(st["dead"])
    dprog = 0 if st.get("dead_programs") is None else int(st["dead_programs"])
    tot = st["subblocks"]
    return {"mode": _MODE, "configured": _CONFIGURED, "env_override": _ENV_OVERRIDE,
            "sub": _SUB, "elide": _ELIDE,
            "disabled": _OFF,
            "slices": st["slices"], "overhead_slices": st["overhead_slices"],
            "subblocks": tot,
            # on: skipped sub-blocks; overhead: what WOULD have been skipped
            "dead": dead, "dead_fraction": round(dead / tot, 6) if tot else None,
            # (query, DOC_BLOCK) posting-kernel programs whose sub-blocks are
            # ALL dead (they exit early; partly-dead programs pay ~full latency)
            "programs": st.get("programs", 0), "dead_programs": dprog,
            "dead_program_fraction": (round(dprog / st["programs"], 6)
                                      if st.get("programs") else None),
            # first-launch checks of compiled skip-kernel variants
            "checks": st["checks"], "checked_dead_cells": st["checked_dead_cells"],
            "check_failures": st["check_failures"],
            # adversarial probes of the bound's fp model, one per variant;
            # worst = largest posting-kernel error as a fraction of the model's
            "model_probes": st["model_probes"],
            "model_probe_tokens": st["model_probe_tokens"],
            "model_probe_failures": st["model_probe_failures"],
            "model_probe_worst": st["model_probe_worst"],
            "elided_slices": st["elided_slices"],
            "materialized": st["materialized"], "masked_topk": st["masked_topk"],
            "rescored": st["rescored"],
            "adaptive": {"min_dead": _min_dead(), "reprobe": _reprobe(),
                         "declines": st["adaptive_declines"],
                         "probes": st["adaptive_probes"],
                         "last_fraction": _ADAPT["frac"]},
            "gates": dict(_GATES), "declines": dict(_DECLINES)}


# --- the skip mask handed to consumers ----------------------------------------

class SkipMask:
    """Block-skipping decisions for one slice. live marks sub-blocks that
    may contribute to top-k; klive is the mask passed to the posting kernel
    (all live in overhead mode).

    When elided, skipped cells are unwritten. Consumers must use the mask
    or call materialize() to fill them with -inf before reading.
    """

    def __init__(self, live, klive, rowany, deadcnt, thr_key, sub, n_sub, n_rows,
                 elided, out):
        self.live, self.klive, self.rowany = live, klive, rowany
        self.deadcnt, self.thr_key = deadcnt, thr_key
        self.sub, self.n_sub, self.n_rows = sub, n_sub, n_rows
        self.elided, self.out = elided, out
        self._materialized = not elided
        # Re-scores `out` with the UNSKIPPED hot kernel (every cell written);
        # set by `sparse_posting_triton.scores`. The fallback when
        # materializing fails.
        self.rescore = None

    def fail(self, why: str) -> None:
        """A consumer could not use the mask (e.g. the masked top-k failed):
        skipping goes off for the run, and the slice is materialized."""
        disable(why)
        self.materialize()

    def cells(self, mask_rows):
        """`(len(rows), n_rows)` bool expansion of a sub-block mask."""
        return mask_rows.repeat_interleave(self.sub, dim=1)[:, : self.n_rows]

    def note_masked(self) -> None:
        """Called by `topk_triton.topk` after a successful masked launch."""
        _STATS["masked_topk"] += 1

    def materialize(self) -> None:
        """Write -inf into every dead cell. On failure (e.g. out of memory)
        skipping goes off and the slice is re-scored with the unskipped hot
        kernel instead, which writes every cell (live cells bit-identical,
        dead cells their true scores, all < thr): the output is unchanged."""
        if self._materialized:
            return
        try:
            n_q = int(self.out.shape[0])
            step = max(1, (1 << 26) // max(1, self.n_rows))
            for r0 in range(0, n_q, step):
                dead = self.cells(self.klive[r0 : r0 + step] == 0)
                self.out[r0 : r0 + step].masked_fill_(dead, float("-inf"))
                del dead
        except Exception as exc:
            disable(f"materialize failed: {type(exc).__name__}: {exc}")
            if self.rescore is None:
                raise
            self.rescore()
            _STATS["rescored"] += 1
        self._materialized = True
        _STATS["materialized"] += 1


# --- host side -----------------------------------------------------------------

def query_plan(q_csr):
    """Run-wide: `(qmap int32 (vocab,), qtok int64 (n_qv,), F fp32, SIG fp32)`.
    One host sync (the query vocabulary size), once per run."""
    import weakref

    hit = _QPLAN.get("q")
    if hit is not None and hit[0]() is q_csr:
        return hit[1]
    plan = make_plan(q_csr)
    _QPLAN["q"] = (weakref.ref(q_csr), plan)
    return plan


def make_plan(q_csr):
    """`query_plan`'s computation, uncached (the model probe's own queries
    must not displace the run's cached plan)."""
    import torch

    vocab = int(q_csr.shape[1])
    col = q_csr.col_indices().to(torch.int64)
    dev = col.device
    present = torch.zeros(vocab, dtype=torch.bool, device=dev)
    present[col] = True
    qtok = present.nonzero().flatten()
    n_qv = int(qtok.numel())
    qmap = torch.full((vocab,), -1, dtype=torch.int32, device=dev)
    qmap[qtok] = torch.arange(n_qv, dtype=torch.int32, device=dev)
    m = q_csr.crow_indices().diff().double()
    F = torch.where(m > MAX_QUERY_NNZ, float("inf"), 1.0 + (m + 1.0) * 2.0 ** -21)
    SIG = (m + 1.0) * 2.0 ** -120
    return (qmap, qtok, F.float(), SIG.float())


MIN_NORMAL = 2.0 ** -126


def _query_min(q_csr) -> float:
    """Smallest query weight, cached per run (one sync per run)."""
    import weakref

    hit = _QPLAN.get("qmin")
    if hit is not None and hit[0]() is q_csr:
        return hit[1]
    v = q_csr.values()
    qmin = float(v.min()) if v.numel() else float("inf")
    _QPLAN["qmin"] = (weakref.ref(q_csr), qmin)
    return qmin


def budget_bytes() -> int:
    return int(float(_ENV["budget_mib"]) * (1 << 20))


def slice_reason(hot, bptr, n_q: int, skip_thr, cmin=None, q_csr=None) -> str | None:
    import torch

    from nova_bf import sparse_posting_triton as sp

    if _OFF is not None:
        return "disabled"
    if _posting_scores_hot_skip is None:
        return f"triton unavailable ({_UNAVAILABLE})"
    if hot is None or bptr is None:
        return "not on the hot path"
    # The margin's underflow term assumes no subnormal inputs (a flush of a
    # subnormal INPUT could cost more than 2^-126 per term).
    if cmin is None or cmin < MIN_NORMAL:
        return "subnormal corpus weights"
    if q_csr is not None and _query_min(q_csr) < MIN_NORMAL:
        return "subnormal query weights"
    db = sp.DOC_BLOCK
    if not db or db % _SUB or _SUB > db:
        return f"sub-block {_SUB} does not divide the document block {db}"
    if db // _SUB > 32:
        return f"more than 32 sub-blocks per program ({db // _SUB}; the flag bitmask is int32)"
    if not (getattr(skip_thr, "ndim", None) == 1 and skip_thr.numel() == n_q
            and skip_thr.dtype is torch.int64):
        return "threshold shape"
    return None


def size_reason(n_q: int, n_rows: int, out) -> str | None:
    """The skip kernels index with int32 sizes (Triton specialises an int
    argument >= 2^31 as int64, a variant the compile key does not track)."""
    if max(n_rows, int(out.stride(0)), -(-n_rows // max(1, _SUB)) * n_q) > INT32_LIMIT:
        return "sizes beyond int32"
    return None


def check_unique(ptr, prow) -> None:
    """Debug / test check of the (row, token) uniqueness precondition."""
    import torch

    n = int(prow.numel())
    if n < 2:
        return
    tok = torch.repeat_interleave(torch.arange(ptr.numel() - 1, device=ptr.device),
                                  ptr.diff(), output_size=n)
    key = tok * (int(prow.max()) + 1) + prow.to(torch.int64)
    if not bool((key[1:] > key[:-1]).all()):
        raise ValueError("sparse slice has duplicate (row, token) postings")


def _bm_meta(sub: int) -> dict:
    """`_block_max`'s constexpr / launch args. The ONE source for both the
    launch and `compile_key`, so the check key cannot drift from what runs."""
    from nova_bf import sparse_posting_triton as sp

    return {"SUB": sub, "BLOCK_P": BLOCK_P, "num_warps": sp.NUM_WARPS}


def _bound_meta(n_sub: int) -> dict:
    nsp = triton.next_power_of_2(n_sub)
    return {"NSP": nsp, "num_warps": max(1, min(8, nsp // 32))}


def _post_meta(sub: int, elide: bool) -> dict:
    from nova_bf import sparse_posting_triton as sp

    return {"DB": sp.DOC_BLOCK, "BLOCK_P": min(sp.BLOCK_P, sp.DOC_BLOCK), "SUB": sub,
            "ELIDE": bool(elide), "num_warps": sp.NUM_WARPS}


def build_bm(q_csr, ptr, prow, pval, hot, n_rows: int, sub: int, plan=None):
    """`BM` as int32 fp32-bit patterns, `(n_qv * n_sub,)`."""
    import torch

    qmap, qtok, _, _ = plan or query_plan(q_csr)
    n_qv = int(qtok.numel())
    n_sub = -(-n_rows // sub)
    hotidx, table = hot
    bm = torch.zeros(max(1, n_qv * n_sub), dtype=torch.int32, device=prow.device)
    if n_qv:
        _block_max[(n_qv,)](qtok, hotidx, table, ptr, prow, pval, bm, n_rows, n_sub,
                            **_bm_meta(sub))
    return bm


def run_bound(q_csr, bm, n_sub: int, thr_key, plan=None):
    import torch

    qmap, _, F, SIG = plan or query_plan(q_csr)
    n_q = int(q_csr.shape[0])
    dev = bm.device
    live = torch.empty((n_q, n_sub), dtype=torch.int8, device=dev)
    rowany = torch.empty(n_q, dtype=torch.int8, device=dev)
    deadcnt = torch.empty(n_q, dtype=torch.int32, device=dev)
    _bound[(n_q,)](q_csr.crow_indices(), q_csr.col_indices(), q_csr.values(), qmap,
                   bm.view(torch.float32), F, SIG, thr_key, live, rowany, deadcnt,
                   n_sub, **_bound_meta(n_sub))
    return live, rowany, deadcnt


def prepare(q_csr, ptr, prow, pval, hot, n_rows: int, thr_key, out):
    """Build BM, run the bound; returns a `SkipMask` or None (declined)."""
    import torch

    sub = _SUB
    n_q = int(q_csr.shape[0])
    n_sub = -(-n_rows // sub)
    if n_sub > MAX_NSUB:
        decline("too many sub-blocks")
        return None
    _, qtok, _, _ = query_plan(q_csr)
    n_qv = int(qtok.numel())
    need = n_qv * n_sub * 4 + n_q * n_sub * (2 if _MODE == "overhead" else 1) + n_q * 13
    if need > budget_bytes():
        decline("over the memory budget")
        return None
    if _ENV["check"]:
        check_unique(ptr, prow)
    dev = prow.device
    thr_key = thr_key.to(dev).clone()          # snapshot: the state rises in place
    bm = build_bm(q_csr, ptr, prow, pval, hot, n_rows, sub)
    live, rowany, deadcnt = run_bound(q_csr, bm, n_sub, thr_key)
    del bm
    if _MODE == "overhead":
        klive = _ONES.get((n_q, n_sub, str(dev)))
        if klive is None:
            klive = _ONES[(n_q, n_sub, str(dev))] = torch.ones(
                (n_q, n_sub), dtype=torch.int8, device=dev)
        elided = False
    else:
        klive, elided = live, _ELIDE
    return SkipMask(live, klive, rowany, deadcnt, thr_key, sub, n_sub, n_rows,
                    elided, out)


def launch(q_csr, prow, pval, n_rows, out, hot, bptr, ctx, elide=None):
    """Launch the skip posting kernel; returns Triton's launch handle (the
    compiled kernel on GPU, for register / spill reporting)."""
    from nova_bf import sparse_posting_triton as sp

    n_q = int(q_csr.shape[0])
    bp, nblk = bptr
    hotidx, table = hot
    return _posting_scores_hot_skip[(n_q, nblk)](
        q_csr.crow_indices(), q_csr.col_indices(), q_csr.values(), hotidx, table,
        bp, prow, pval, out, ctx.klive, n_rows, out.stride(0), nblk, ctx.n_sub,
        **_post_meta(ctx.sub, ctx.elided if elide is None else elide))


def compile_key(dev, q_csr, ptr, prow, pval, out, hot, bptr, ctx, plan=None):
    """Identify compiled skip-kernel variants by constexprs, warp counts,
    pointer alignment and dtype, and size specialization classes.

    Recheck only when these properties change, not when sizes change
    within the same class. Reject sizes that exceed int32 limits.
    """
    from nova_bf import sparse_posting_triton as sp

    qmap, qtok, F, SIG = plan or query_plan(q_csr)
    metas = (("bm", _bm_meta(ctx.sub)), ("bound", _bound_meta(ctx.n_sub)),
             ("post", _post_meta(ctx.sub, ctx.elided)))
    return sp._compile_key(
        dev, (ctx.n_rows, out.stride(0), ctx.n_sub),
        (q_csr.crow_indices(), q_csr.col_indices(), q_csr.values(), hot[0], hot[1],
         bptr[0], ptr, prow, pval, out, ctx.klive, ctx.live, ctx.rowany, ctx.deadcnt,
         ctx.thr_key, qmap, qtok, F, SIG),
        ("hot_skip",) + tuple((name, tuple(sorted(m.items()))) for name, m in metas))


# --- first-launch check and adaptive engagement -------------------------------

def first_check(q_csr, Cb, out, ctx) -> None:
    """Check each compiled skip-kernel variant against cuSPARSE on its
    first launch using sampled queries.

    Live scores must agree within FP32 rounding bounds, and overlap masks
    must match exactly. Skipped scores must be below the threshold and
    stored as -inf if written. Raise SkipCheckMismatch on failure.
    """
    import torch

    from nova_bf import sparse_posting_triton as sp
    from nova_bf.tiebreak import unpack_score

    rows = sp._sample_rows(int(q_csr.shape[0]))
    if not rows:
        return
    if not _flags_consistent(ctx.live, ctx.rowany, ctx.deadcnt):
        _STATS["check_failures"] += 1
        raise SkipCheckMismatch("skip-kernel first-launch check: the bound's "
                                "ROWANY / DEADCNT disagree with its LIVE flags")
    ref, shared = sp.reference(q_csr, Cb, rows, counts=True)
    mag = sp.reference(q_csr, Cb, rows, absolute=True)
    ridx = torch.as_tensor(rows, device=out.device)
    got = out.index_select(0, ridx)
    bdead = ctx.cells(ctx.live.index_select(0, ridx) == 0)      # bound's verdict
    kdead = ctx.cells(ctx.klive.index_select(0, ridx) == 0)     # what the kernel skipped
    if ctx.elided:
        got = got.masked_fill(kdead, float("-inf"))             # never written
    thr = unpack_score(ctx.thr_key.index_select(0, ridx))
    ref_dead = torch.isneginf(ref) | kdead
    got_dead = torch.isneginf(got)
    bad = ref_dead != got_dead
    bad |= bdead & ~(ref < thr[:, None])
    live = ~ref_dead & ~got_dead
    eps = torch.finfo(torch.float32).eps
    tol = (2 * shared + 2) * eps * mag.clamp_min(0) + 1e-30
    bad |= live & ~((got - ref).abs() <= tol)
    _STATS["checks"] += 1
    _STATS["checked_dead_cells"] += int(bdead.sum())
    n_bad = int(bad.sum())
    if n_bad:
        _STATS["check_failures"] += 1
        raise SkipCheckMismatch(f"skip-kernel first-launch check: {n_bad} of "
                                f"{bad.numel()} sampled cells disagree with cuSPARSE")

# Test each compiled variant with adversarial inputs that stress the
# floating-point assumptions behind the pruning bound.
# Check the score accumulation bounds, conservative pruning bound,
# exact block maxima, and consistency of live scores across skip paths.
# Unlike the real-data self-check, this targets numerical edge cases.
# Passing the probe does not prove correctness for every input.
_U = 2.0 ** -24           # fp32 unit roundoff
_ETA = 2.0 ** -126        # per-rounding underflow allowance (gradual or flushed)
_SLACK = 2.0 ** -40       # host fp64 evaluation slack (model terms are >= 2^-23 S)


def _probe_rows():
    """The probe's queries: each row a list of `(q, c_victim, c_companion,
    hot)` in token-id order (the bound kernel's order; the posting kernel adds
    every hot token first, then the cold ones). All values exact fp32 > 0;
    `c_companion` None: the companion document lacks the token."""
    import random

    import numpy as np

    def f32(x):
        return float(np.float32(x))

    rnd = random.Random(0x5EED)
    a = 2.0 ** -23                                  # one ulp of [1, 2)

    def full(scale=1.0):                            # full 24-bit mantissa
        return f32((1.0 + rnd.getrandbits(23) * a) * scale)

    def companion(c, i):
        return None if i % 3 == 2 else (f32(c * (1 + 2.0 ** -20)) if i % 2 else c)

    rows = []
    # 1. Order adversaries: the two kernels round in OPPOSITE directions.
    #    Posting: hot 1.0 + 1.5 first, then every cold term (just over half an
    #    ulp of [2, 4)) rounds UP; the bound sums the cold terms exactly first.
    up = f32(a * (1 + 2.0 ** -10))
    for n in (16, 256):
        rows.append([(1.0, up, None, False)] * n
                    + [(1.0, 1.0, None, True), (1.0, 1.5, None, True)])
    #    Bound: cold 1.0 + 1.5 first, then every hot term (just under half an
    #    ulp) rounds DOWN to nothing; the posting kernel sums them exactly first.
    down = f32(a * (1 - 2.0 ** -10))
    rows.append([(1.0, 1.0, None, False), (1.0, 1.5, None, False)]
                + [(1.0, down, None, True)] * 32)
    # 2. Same-direction adversaries: every update rounds by nearly half an ulp
    #    the same way, in both kernels (the model's upper / lower limits).
    for frac in (0.5 + 2.0 ** -8, 0.5 - 2.0 ** -8):
        t = f32(2.0 ** -12 + a * frac)
        rows.append([(1.0, 1.0, None, False)] + [(1.0, t, None, False)] * 256)
        rows.append([(1.0, 1.0, None, True)] + [(1.0, t, None, True)] * 15)
    # 3. Precision: full-mantissa products. A narrower multiply or accumulator
    #    errs by far more than the model's 2u per update at small m.
    for m, hot in ((1, False), (3, False), (8, False), (1, True), (3, True)):
        row = []
        for i in range(m):
            c = full()
            row.append((full(), c, companion(c, i), hot))
        rows.append(row)
    # 4. Underflow: products near 2^-130 are subnormal, and flushed to zero
    #    under FTZ. Their scores are far below 2m eta, so these rows test that
    #    SIG still keeps the cell live (not the eta terms of (A)).
    for m, hot in ((1, False), (64, False), (4, True)):
        rows.append([(2.0 ** -70, full(2.0 ** -60), None, hot) for _ in range(m)])
    # 5. Mixed: interleaved hot / cold tokens, magnitudes over 2^8.
    row = []
    for i in range(40):
        c = full(2.0 ** -rnd.randrange(9))
        row.append((full(2.0 ** -rnd.randrange(9)), c, companion(c, i), rnd.random() < 0.3))
    rows.append(row)
    return rows


def _flags_consistent(live, rowany, deadcnt) -> bool:
    """`_bound`'s per-row reductions agree with its LIVE flags (ROWANY = 0
    makes the masked top-k drop the whole row)."""
    import torch

    return (torch.equal(rowany.to(torch.bool), (live != 0).any(dim=1))
            and torch.equal(deadcnt.to(torch.int64), (live == 0).sum(dim=1)))


def _f32_down(x: float) -> float:
    """The largest fp32 <= x."""
    import numpy as np

    y = np.float32(x)
    if float(y) > x:
        y = np.nextafter(y, np.float32(-np.inf))
    return float(y)


def model_probe(real_key, dev, q_csr, n_rows: int, out, hot, ctx) -> None:
    """Run the model probe through the compiled variant `real_key` (the
    variant `ctx`'s slice just used). Raises `SkipCheckMismatch` on any
    violated inequality, or if the probe cannot reproduce the variant."""
    import math

    import torch

    from nova_bf import sparse_posting_triton as sp
    from nova_bf.tiebreak import pack

    rows = _probe_rows()
    sub, n_sub, elided = ctx.sub, ctx.n_sub, ctx.elided
    overhead = ctx.klive is not ctx.live
    # The victim document sits in sub-block 1 of program 1 when the slice has
    # one (else of program 0), so the second launch runs an all-dead program
    # before it, a partly-dead one with b >= 1 around it and all-dead ones
    # after; the companion shares its sub-block (BM from both).
    nsb = max(1, sp.DOC_BLOCK // sub)
    v = min(n_rows - 1, (nsb + 1 if n_sub > nsb + 1 else 1) * sub + 1)
    comp = v + 1 if v + 1 < n_rows and (v + 1) // sub == v // sub else None
    sbv = v // sub
    qcrow, qcol, qval, hot_tok, dv, dc = [0], [], [], [], [], []
    for r in rows:
        for qv, cv, cc, is_hot in r:
            t = len(qcol)                   # every token in exactly one query
            qcol.append(t)
            qval.append(qv)
            dv.append((t, cv))
            if comp is not None and cc is not None:
                dc.append((t, cc))
            if is_hot:
                hot_tok.append(t)
        qcrow.append(len(qcol))
    vocab, n_qp, n_hot = len(qcol), len(rows), len(hot_tok)
    qc, qi = q_csr.crow_indices(), q_csr.col_indices()
    q_p = torch.sparse_csr_tensor(
        torch.tensor(qcrow, dtype=qc.dtype, device=dev),
        torch.tensor(qcol, dtype=qi.dtype, device=dev),
        torch.tensor(qval, dtype=torch.float32, device=dev),
        size=(n_qp, vocab), check_invariants=False)
    cnt = torch.zeros(n_rows + 1, dtype=torch.int64)
    cnt[v + 1] = len(dv)
    if comp is not None:
        cnt[comp + 1] = len(dc)
    docs = dv + dc                          # rows in order: v < comp
    Cb = torch.sparse_csr_tensor(
        torch.cumsum(cnt, 0).to(dev),
        torch.tensor([t for t, _ in docs], dtype=torch.int64, device=dev),
        torch.tensor([c for _, c in docs], dtype=torch.float32, device=dev),
        size=(n_rows, vocab), check_invariants=False)
    ptr, prow, pval = sp.corpus_postings(Cb)
    bptr = sp.block_ptr(ptr, prow, vocab, n_rows, sp.DOC_BLOCK)
    hotidx = torch.full((vocab,), -1, dtype=hot[0].dtype, device=dev)
    hotidx[torch.tensor(hot_tok, dtype=torch.int64, device=dev)] = torch.arange(
        n_hot, dtype=hot[0].dtype, device=dev)
    hot_p = (hotidx, sp.hot_table(ptr, prow, pval, hotidx, n_hot, n_rows, vocab))
    plan = make_plan(q_p)
    o = torch.empty((n_qp, int(out.stride(0))), dtype=torch.float32, device=dev)[:, :n_rows]
    bm = build_bm(q_p, ptr, prow, pval, hot_p, n_rows, sub, plan)
    zero = torch.zeros(n_qp, dtype=torch.int64, device=dev)
    fails = []

    def bound(thr):
        tk = pack(torch.tensor(thr, dtype=torch.float32, device=dev), zero)
        live, rowany, dcnt = run_bound(q_p, bm, n_sub, tk, plan)
        if not _flags_consistent(live, rowany, dcnt):
            fails.append("the bound's ROWANY / DEADCNT disagree with its LIVE flags")
        klive = torch.ones_like(live) if overhead else live
        return SkipMask(live, klive, rowany, dcnt, tk, sub, n_sub, n_rows, elided, o)

    def post(c):
        o.fill_(float("nan"))
        launch(q_p, prow, pval, n_rows, o, hot_p, bptr, c)

    cols = [v] + ([comp] if comp is not None else [])

    def rest_ok(c):
        """Cells of documents other than `cols` (no shared token): -inf, or
        left unwritten (NaN) in a skipped sub-block of an elided launch."""
        rest = o.clone()
        rest[:, cols] = float("-inf")
        ok = torch.isneginf(rest)
        if elided:
            ok |= c.cells(c.klive == 0) & torch.isnan(rest)
        return bool(ok.all())

    # Launch 1: thr = -inf, everything live (the all-live path).
    c1 = bound([float("-inf")] * n_qp)
    key = compile_key(dev, q_p, ptr, prow, pval, o, hot_p, bptr, c1, plan=plan)
    if key != real_key:
        raise SkipCheckMismatch("model probe could not reproduce the compiled "
                                "variant it must check (alignment / size class)")
    post(c1)
    s1 = o[:, cols].clone()
    if not rest_ok(c1):
        fails.append("a cell without a shared token was written")
    # (D) BM: every token is a query token, so BM's row index is the token.
    bmv = bm.view(torch.float32).view(vocab, n_sub)
    exp_bm = [0.0] * vocab
    for t, c in docs:
        exp_bm[t] = max(exp_bm[t], c)
    if not torch.equal(bmv[:, sbv].cpu(), torch.tensor(exp_bm, dtype=torch.float32)):
        fails.append("BM is not the exact block maximum")
    other = bmv.clone()
    other[:, sbv] = 0
    if bool((other != 0).any()):
        fails.append("BM of an empty sub-block is not zero")
    # (A) per cell, and the second launch's thresholds for (B) and (C).
    got = s1.cpu().tolist()
    cval = [dict(dv), dict(dc)]
    thr2, worst, j0 = [], 0.0, 0
    for r, row in enumerate(rows):
        m = len(row)
        toks = range(j0, j0 + m)
        j0 += m
        g_hi = math.expm1(2 * m * math.log1p(_U))     # (1+u)^(2m) - 1
        g_lo = -math.expm1(2 * m * math.log1p(-_U))   # 1 - (1-u)^(2m)
        q_of = dict(zip(toks, (e[0] for e in row)))
        top = float("-inf")
        for j, docv in enumerate(cval[: len(cols)]):
            terms = [q_of[t] * docv[t] for t in toks if t in docv]
            if not terms:
                continue
            S = math.fsum(terms)                      # products exact in fp64
            x = got[r][j]
            dhi = S * g_hi + 2 * m * _ETA * (1 + g_hi)
            dlo = S * g_lo + 2 * m * _ETA
            pad = S * _SLACK
            if not (S - dlo * (1 + _SLACK) - pad <= x <= S + dhi * (1 + _SLACK) + pad):
                fails.append(f"posting kernel outside the model (row {r}: "
                             f"{x!r} vs exact {S!r})")
            else:
                worst = max(worst, (x - S) / dhi if x >= S else (S - x) / dlo)
            top = max(top, x)
        UB = math.fsum(q_of[t] * max(cval[0].get(t, 0.0), cval[1].get(t, 0.0))
                       for t in toks)
        ub_min = max(0.0, UB * (1 - g_lo) - 2 * m * _ETA)
        F, SIG = 1.0 + (m + 1) * 2.0 ** -21, (m + 1) * 2.0 ** -120
        t_min = ub_min * F * (1 - _U) ** 2 + SIG * (1 - _U) - 2 * _ETA
        t_min -= abs(t_min) * _SLACK
        thr2.append(max(_f32_down(t_min), top))
    # Launch 2: each row's threshold at the model's lowest T (B) or at the
    # sub-block's own scored maximum (C), whichever is higher: the victim's
    # sub-block must stay live. Every other sub-block is empty (UB = 0,
    # T = SIG) and must die at a threshold above SIG (the skip paths run).
    c2 = bound(thr2)
    live2 = c2.live.cpu()
    if not bool((live2[:, sbv] == 1).all()):
        fails.append("the bound killed a sub-block the model keeps live")
    if n_sub > 1:
        big = torch.tensor([t > 2.0 ** -100 for t in thr2])
        others = live2[big].clone()
        others[:, sbv] = 0
        if bool((others != 0).any()):
            fails.append("the bound kept an empty sub-block live")
    post(c2)
    if not torch.equal(o[:, cols].contiguous().view(torch.int32),
                       s1.contiguous().view(torch.int32)):
        fails.append("live cells differ between the all-live and skip paths")
    if not rest_ok(c2):
        fails.append("a skipped or untouched cell holds a score")
    _STATS["model_probes"] += 1
    _STATS["model_probe_tokens"] += sum(len(r) for r in rows)
    _STATS["model_probe_worst"] = max(_STATS["model_probe_worst"] or 0.0, worst)
    if fails:
        _STATS["model_probe_failures"] += 1
        raise SkipCheckMismatch(f"skip-kernel model probe: {fails[0]}"
                                + (f" (+{len(fails) - 1} more)" if len(fails) > 1 else ""))


def _probe_async(cnt, total: int) -> None:
    """Start a non-blocking read of a slice's dead count into the single
    pinned slot, unless an earlier read is still in flight (that one is
    kept; this slice's count is simply not sampled)."""
    import torch

    if _ADAPT["probe"] is not None:
        return
    if cnt.is_cuda:
        h = torch.empty((), dtype=torch.int64, pin_memory=True)
        stream = torch.cuda.current_stream(cnt.device)
        with torch.cuda.stream(stream):
            h.copy_(cnt, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(stream)
        _ADAPT["probe"] = (h, ev, total)
    else:
        _ADAPT["probe"] = (cnt, None, total)


def _poll_probes() -> None:
    """Consume the slot once its copy has landed (no sync)."""
    p = _ADAPT["probe"]
    if p is None:
        return
    h, ev, total = p
    if ev is not None and not ev.query():
        return
    _ADAPT["probe"] = None
    _ADAPT["frac"] = int(h) / max(1, total)


def _adaptive_skip() -> bool:
    """True: decline building tables for this slice because the latest
    sampled slice proved too little dead. Re-probe every `_reprobe()`."""
    _poll_probes()
    frac = _ADAPT["frac"]
    if frac is None or frac >= _min_dead():
        _ADAPT["since_probe"] = 0
        return False
    _ADAPT["since_probe"] += 1
    if _ADAPT["since_probe"] >= _reprobe():
        _ADAPT["since_probe"] = 0
        _STATS["adaptive_probes"] += 1
        return False
    return True


def book(ctx, n_q: int) -> None:
    import torch

    cnt = ctx.deadcnt.sum(dtype=torch.int64)
    _STATS["dead"] = cnt if _STATS["dead"] is None else _STATS["dead"] + cnt
    _STATS["subblocks"] += n_q * ctx.n_sub
    # Dead programs: group the bound's verdict into the posting kernel's
    # (query, DOC_BLOCK) programs. Padding past n_sub counts as not live,
    # matching the kernel's all-dead test (bits == 0 over in-slice flags).
    # Device-side only (no sync); read back once in usage().
    from nova_bf import sparse_posting_triton as sp
    nsb = max(1, sp.DOC_BLOCK // ctx.sub)
    nblk = -(-ctx.n_sub // nsb)
    lv = ctx.live.view(n_q, ctx.n_sub)
    if nblk * nsb != ctx.n_sub:
        lv = torch.nn.functional.pad(lv, (0, nblk * nsb - ctx.n_sub))
    dprog = (lv.view(n_q, nblk, nsb).amax(dim=2) == 0).sum(dtype=torch.int64)
    _STATS["dead_programs"] = (dprog if _STATS.get("dead_programs") is None
                               else _STATS["dead_programs"] + dprog)
    _STATS["programs"] = _STATS.get("programs", 0) + n_q * nblk
    _STATS["overhead_slices" if _MODE == "overhead" else "slices"] += 1
    _STATS["elided_slices"] += int(ctx.elided)
    _probe_async(cnt, n_q * ctx.n_sub)


def try_skip(q_csr, Cb, ptr, prow, pval, bptr, hot, n_rows, out, thr_key, cmin=None):
    """Score the slice with skipping into `out`; returns the `SkipMask`, or
    None when the caller must score it without skipping (declined, adaptive
    gate, out of memory, or skipping just switched off by a failure)."""
    from nova_bf import profiling
    from nova_bf import sparse_posting_triton as sp
    from nova_bf.twopass import is_oom

    n_q = int(q_csr.shape[0])
    try:
        # Inside the protection: any skip-side error (here or below) turns
        # off skipping only; the caller scores the slice unskipped.
        why = slice_reason(hot, bptr, n_q, thr_key, cmin, q_csr) or \
            size_reason(n_q, n_rows, out)
        if why is not None:
            decline(why)
            return None
        if _adaptive_skip():
            _STATS["adaptive_declines"] += 1
            return None
        with profiling.slice_mark("bf_sparse_block_bound"):
            ctx = prepare(q_csr, ptr, prow, pval, hot, n_rows, thr_key, out)
        if ctx is None:
            return None
        with profiling.slice_mark("bf_sparse_posting_kernel"):
            launch(q_csr, prow, pval, n_rows, out, hot, bptr, ctx)
        key = compile_key(Cb.device, q_csr, ptr, prow, pval, out, hot, bptr, ctx)
        if key not in sp._CHECKED:
            model_probe(key, Cb.device, q_csr, n_rows, out, hot, ctx)
            first_check(q_csr, Cb, out, ctx)
            sp._CHECKED.add(key)
        book(ctx, n_q)
        return ctx
    except SkipCheckMismatch as exc:
        # As the posting kernel treats its own mismatch, but scoped to
        # skipping: off for the run; the caller re-scores this slice unskipped.
        disable(str(exc))
        return None
    except Exception as exc:
        if is_oom(exc):
            decline("out of memory")
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        disable(f"{type(exc).__name__}: {exc}")
        return None
