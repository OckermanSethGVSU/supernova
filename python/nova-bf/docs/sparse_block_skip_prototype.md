# Sparse posting kernel: exact block-max sub-block skipping (PROTOTYPE, v3)

> **v3 (Seth's decision): the runtime audit machinery is removed.**
> - Gone: periodic and first-dead-slice audits, async probe queue,
>   first-phase per-slice sync, dead-row audit sampling, the run failure,
>   AUDIT_WAIVE, `.audit_failed` quarantine, the provenance stamp and the
>   merge refusal. `merge.py` and `results.py` are back to HEAD.
> - Kept: a first-launch model probe + check per compiled skip-kernel variant (see
>   "First-launch check" below).
> - The "Round 2-5" sections below are history; their audit / quarantine /
>   stamp items (A1, A2, C1, R2, K2, K3 and parts of A3, R1, R3) no longer
>   describe the code.

Status: prototype, ON by default since 2026-11-15 (`params.sparse_block_skip: off`
disables it). GPU-verified on A10G (2026-11-15, two runs: every 40-file FineWeb
scan bit-identical to skipping off, warm 40-file GPU time 1.12x, results in
`benchmarks/experiments/sparse_block_skip_gpu/`). Not yet run on a GPU: the
model probe, the ROWANY/DEADCNT check, and the posting-kernel review fixes
(hot-table scratch slots, `scores()` contract, float32-only). Background: `benchmarks/experiments/sparse_blockmax_oracle/README.md`
in the main checkout (64-doc bounds prove ~52% of posting work dead at a
10-file threshold, ~95% late).

v2 reworks v1 after three adversarial reviews (2 correctness, 1 speed). v1
used an 8 x 64 sub-block loop, an fp64 bound, a per-(token, sub-block) pointer
table, and wrote -inf into every dead cell. The speed review estimated v1 at
2-3x slower than today at zero skip, plus ~2-8 ms/slice for the bound.

Code: `src/nova_bf/sparse_block_skip.py` (all skip kernels, state, first-launch check).
Hooks are in `sparse_posting_triton.scores`, `topk_triton._cutfill/topk`,
`tiebreak.pack_topk`, `compute._process_batch_group` / `SparseBatchSlice`, and
`config.ParamsConfig`.

## Design

Per slice, when the gate admits it (below):

1. **Query plan** (once per run, `query_plan`): the query vocabulary (one host
   sync per run), `qmap` token -> compact id, and per-query `F = 1 + (m+1) 2^-21`
   and `SIG = (m+1) 2^-120` (m = nnz(q); F = +inf for m > 40000).
2. **BM** (`_block_max`, one program per query token): `BM[c, sb]` = max
   weight of query token c in sub-block sb, stored as fp32 bit patterns in
   int32. Hot tokens reduce their dense hot-table row (no atomics). Cold query
   tokens walk their own posting list with native int32 `atomic_max`, which is
   valid because weights are strictly positive on the hot path, so int32 order
   equals value order. Postings of tokens no query uses are never touched: no
   dump row, no fp64, no CAS loop, no token expansion.
3. **Bound** (`_bound`, one program per query): `ub = sum_t fma(q_t, BM[t,:])`
   in fp32, and `dead = ub * F + SIG < thr` (strict). It writes int8
   `live (n_q, n_sub)`, int8 `rowany (n_q,)` and an int32 per-row dead count
   (for stats). BM is freed right after. Timing mark: `bf_sparse_block_bound`.
4. **Posting kernel** (`_posting_scores_hot_skip`): exactly
   `_posting_scores_hot`'s program structure (512-doc programs, the
   ORIGINAL per-block BPTR, one chain per cold token).
   - It loads its `DB/SUB` flags once. If all are dead, it returns (or stores
     -inf when not eliding).
   - Otherwise hot-table loads are masked by `om & live[o // SUB]` and cold
     postings by `live[d // SUB]`. Masked lanes issue no memory transactions.
   - Live lanes execute the same loads, fma's and stores in the same order as
     `_posting_scores_hot`, so they are bit-identical.
   - Overhead mode runs the same kernel with an all-ones flag tensor.
5. **Elision + mask-aware top-k** (S4, sub-flag `sparse_block_skip_elide`,
   default true): dead cells are not written at all. The slice carries a
   `SkipMask`.
   - **Mask-aware path.** It is used for the single consumer when the read is the
     plain wide pruned top-k (no column gather, cell mask, two-pass plan,
     scaling or query subset). The call chain is `pack_topk(lmask=)` ->
     `topk_triton.topk(lmask=)` -> `_cutfill(HAS_LM=True)`.
     - A row with `rowany == 0` loads nothing and is dead (thr > -inf whenever
       anything is dead).
     - A partly dead row loads with `m & live[offs // SUB]`, other = -inf. That
       is exactly the value an unelided skip kernel would have stored, so every
       later step of `_cutfill` sees the same row: same keys, ids and live
       flags.
   - **Every other reader** calls `SkipMask.materialize()` first, which writes
     -inf into dead sub-blocks in row chunks. These readers are:
     - the portable `pack_topk` path (including any pre-enqueue Triton fallback);
     - narrow slices (`pack` + `live_rows`);
     - defensively, any second `SparseBatchSlice.score()` read.
   - **Non-skip callers** (dense, multivector, non-skip sparse) run `_cutfill`
     with `HAS_LM` false: the original load, unchanged. The new arguments are
     placeholders.

## Soundness

### fp32 margin (proof also in `_bound`'s docstring)

Notation: u = 2^-24, eta = 2^-126; q, c > 0 (strictly positive on the hot path,
and the skip path declines subnormal weights on either side); m = nnz(q).
Model, with gradual or flush-to-zero underflow, for x >= 0:
`x(1-u) - eta <= fl(x) <= x(1+u) + eta`.

- Exact: `S_d <= UB = sum_t q_t BM[t,sb]` for d in sb. The terms are
  non-negative, BM >= c_dt, and absent tokens add q_t BM >= 0. BM is an exact
  copy (a max).
- Kernel: at most m nonzero updates, at most 2 roundings each (fma, or
  mul+add), all values >= 0 (`fma(q, 0, acc) = acc` exactly):
  `S^ <= S (1+u)^(2m) + 2m eta (1+u)^(2m)`.
- Bound in fp32, any order: `ub^ >= UB (1-u)^(2m) - 2m eta`.
- For 4mu <= 0.01 (m <= 40000): `((1+u)/(1-u))^(2m) <= 1 + 4.05 m u`, and
  `(1+u)^(2m) <= 1.006`. So `S^ <= ub^ (1 + 4.05 m u) + 4.03 m eta`.
- Test: `T = fl(fl(ub^ F) + SIG)`, with F = 1 + 8(m+1)u (exact) and
  SIG = 64(m+1) eta (exact and normal):
  `T >= ub^ F (1-u)^2 + SIG (1-u) - 2 eta >= ub^ (1 + (8m+6)u - 16(m+1)u^2) + 63(m+1) eta >= S^`.
  If the compiler contracts to an fma there is one rounding, which is still
  covered.
- Hence `T < thr` implies every kernel score in the sub-block is `< thr`.
- The same holds for any summation order, so the cuSPARSE reference obeys it
  too; the audit checks this.
- NaN or -inf thresholds, and F = +inf, never mark anything dead.
- The posting kernel's magnitude guard (`< 1e36`) rules out overflow.

### Threshold validity, consumer gating, downstream readers

Unchanged from v1; see the audit notes in `compute._sparse_skip_member` and
`SparseBatchSlice`.

- **Threshold:** the member's OWN state k-th key `spec_thr[m]`, snapshotted.
  A heap that is not full gives -inf, so nothing is dead. The strict test keeps
  ties.
- **Gate:** exactly one unfiltered, unscaled, unsubset dot search over the
  vector type, with pruning on.
- **Downstream:**
  - `live_rows` / `_cutfill` liveness is unchanged: a row's max is >= thr iff
    it has a cell >= thr, and such cells are never skipped.
  - The fold sees the same candidate set.
  - Short-hit padding, tiebreak=id, threshold exchange and trimming read the
    unchanged state.
- **New in v2, for elision:** every reader of the raw matrix other than the
  mask-aware top-k materializes first (above). The posting self-check / audit
  fills dead cells itself before comparing.

### Warm-up gate (S3)

Until the member has folded `k` rows (a host counter, no sync), the state
cannot be full. So no tables are built and no skip kernel runs; the slice is
counted as declined ("state cannot be full yet").

### First-launch check and failure isolation (v3)

**Check.** Mirroring the posting kernel's `_self_check`, each compiled
skip-kernel variant (`compile_key`: every constexpr and num_warps of
`_block_max`, `_bound` and the skip posting kernel, pointer alignments, size
specialisation classes) is checked on its FIRST launch only (`first_check`).
It compares sampled rows with cuSPARSE, using the same rows and tolerance as
`_self_check`:
- live cells are within the fp32 order bound, with an exact no-overlap mask;
- every cell the bound marked dead has reference `< thr`;
- a written dead cell (not elided) reads -inf.

**Model probe (2026-11-15).** Before `first_check`, `model_probe` runs the
SAME compiled variant on synthetic inputs (it rebuilds the compile key and
refuses if it differs). It uses the real n_rows and stride, the same
constexprs, and fresh aligned buffers. The inputs are built to break the
proof's fp model: ~980 tokens over 18 query rows. The rows are:
- order adversaries, where the posting kernel (hot tokens first) rounds up
  on every update while the bound (token-id order) sums the same terms
  exactly, and the reverse;
- same-direction adversaries, where every update rounds by nearly half an
  ulp, cold (m=257) and hot;
- full-mantissa precision rows at m = 1..8;
- underflow rows, with products near 2^-130 (subnormal, flushed or not).
  These test that SIG keeps the cell live; their scores are far below the
  eta terms of (A);
- one mixed row.

Each row has a victim document in sub-block 1 of program 1 (program 0 if the
slice has only one), plus a companion document in the same sub-block (so BM
takes the max of both). The probe checks against
fp64 references:
- (A) every probe cell lies within the posting-kernel lemma
  `S(1-u)^(2m) - 2m eta <= S^ <= S(1+u)^(2m) + 2m eta (1+u)^(2m)`;
- (D) BM is bitwise the exact block maximum, and zero for empty sub-blocks;
- a second bound launch sets each row's thr to the larger of the model's
  lowest T (B: `ub_min F (1-u)^2 + SIG(1-u) - 2 eta`) and the sub-block's own
  kernel score (C: `T >= S^`, implied by A and B). The victim's sub-block
  must stay live, and every empty sub-block must die. ROWANY and DEADCNT
  must match LIVE in both launches (`first_check` also checks this on the
  real slice);
- the second posting launch runs the partly-dead and all-dead paths, and
  must leave live cells bit-identical to the all-live launch, with untouched
  cells -inf or, when elided and dead, unwritten.

A violation raises `SkipCheckMismatch`, which turns skipping off and
re-scores the slice, as for `first_check`. The manifest records
`model_probes`, `model_probe_tokens`, `model_probe_failures` and
`model_probe_worst`. `model_probe_worst` is the largest posting error as a
fraction of the lemma's allowance; round to nearest gives ~0.47. Like
twopass's accumulation probe, this exposes arithmetic that breaks the model
but does not prove the model for every input. (B) uses the model's worst
case, so it catches a kernel F that is short by more than ~2(m+1)u, a
quarter of the margin. Interpreter mutations caught:
- F = 1 / 1 + 2(m+1)u / 1 + 5(m+1)u with SIG intact, and SIG = 0 or SIG/4
  with the proof's F (an F at 7/8 of the margin or more is NOT caught);
- ROWANY zeroed (probe, and `first_check` when the probe is bypassed);
- posting scores 2^-16 high;
- BM one ulp low;
- a skip-path-only 1-ulp divergence;
- a variant the probe cannot reproduce.

There is no periodic or dead-slice-triggered re-check (removed in v3,
Seth's decision). A first launch usually happens early, when few or no cells
are dead, so the soundness of later skips rests on the proof plus the
exactness tests, as for the posting kernel itself.

**On mismatch.** The posting kernel disables itself on its own mismatch. Here
only skipping turns off for the run (`block_skip.disabled`,
`check_failures`), and the slice is re-scored with the unskipped hot kernel
(itself under `_self_check`), so that slice's output is right. Slices folded
earlier under the same variant are not revisited. Nothing is stamped or
quarantined, and the run does not fail.

**Other failure isolation.**
- Any exception in the skip path turns off skipping only and re-scores the
  slice unskipped.
- An OOM in skip prep declines that slice only.
- A `materialize()` failure re-scores via `_launch_checked`.
- A masked `_cutfill` failure falls back to materialize + the plain Triton
  top-k.

**Memory.**
- No token expansion; BM is freed before return.
- Skipping (not the kernel) is declined when
  `n_qv * n_sub * 4 + n_q * n_sub (+ n_q * n_sub in overhead) + 13 n_q` exceeds
  `NOVA_BF_SPARSE_BLOCK_SKIP_BUDGET_MIB` (2048), or when n_sub > 8192.

**Adaptive engagement.** One pinned host slot gets the latest skip slice's
dead count (non-blocking copy + event, polled without sync). A new read
starts only once the previous one has been consumed. While the last
fraction is below MIN_DEAD (0.15), slices build no tables, re-probing every
REPROBE (16).

## Config / manifest

Params:
- `params.sparse_block_skip: off|on|overhead` (YAML true/false accepted)
- `sparse_block_skip_sub: 64`
- `sparse_block_skip_elide: true`

Env overrides win (all parsed and validated in `configure`, which runs right
after the per-run reset; errors name the variable):
- `NOVA_BF_SPARSE_BLOCK_SKIP`: on = 1/true/on/yes; off = 0/false/off/no/empty;
  or overhead
- `..._SUB`
- `..._ELIDE` (same on/off spellings)
- `..._MIN_DEAD` (0.15), `..._REPROBE` (16), `..._BUDGET_MIB` (2048)
- `..._CHECK` (uniqueness debug check)

`params.kernels.sparse_posting_kernel.block_skip` records:
- `mode` (effective), `configured`, `env_override`;
- sub, elide, disabled;
- slices, overhead_slices, subblocks;
- dead (+ fraction; in overhead mode, the would-be dead);
- checks, checked_dead_cells, check_failures;
- elided_slices, materialized, masked_topk, rescored;
- adaptive, gates, declines.

## Review items

| item | v2 |
|---|---|
| S1 sub-block loop | replaced by the masked 512-wide program with the original BPTR; SB table gone |
| S2 fp64 bound | fp32 Triton bound kernel, proven margin; BM via hot-table reduce + int32 atomic_max over query tokens' postings only |
| S3 warm-up | no tables until k rows folded |
| S4 dead writes + re-read | elision + mask-aware `_cutfill` (sub-flag `sparse_block_skip_elide`); others materialize |
| C1 audit never sees dead | async-probe-triggered audit + periodic; stat `audited_dead_cells` |
| C2 failure kills kernel | only skipping turns off; slice re-scored unskipped |
| C3 memory | no second expansion; budget decline; OOM -> unskipped hot kernel |
| C4 n_rows | the hot-path condition is the original one again (posting module reverted to master + hooks); a 0-row slice returns before any table either way; test added |
| C5 manifest | effective/configured/env_override recorded; spellings documented |
| C6 timing mark | `bf_sparse_block_bound` |

## Known risks / owed on GPU

- **Speed is unmeasured.** Things to watch:
  - the per-lane flag gather in the hot loop, and the per-posting flag load in
    the cold loop, at zero skip (overhead vs off);
  - the BM kernel (hot rows: n_sub scalar stores per token; cold: atomics);
  - the bound kernel at 100k queries;
  - the masked `_cutfill` vs the plain one.
- `materialize` on fallback paths costs a bool expansion per row chunk. It
  should only run on CPU / portable paths.
- The first dead slice is unaudited (a one-slice lag, by design, to avoid a sync).
- Not done: exchange thresholds, a multi-consumer min-threshold, filtered /
  cosine consumers, the non-hot kernels.

## Round 2 (v2.1) changes

Round 2 found no output-changing bug. Changes:

- **T1 poisoning.** Every interpreted skip launch now runs on a score buffer
  pre-filled with +3e38 (autouse fixture wrapping `try_skip`), so a reader
  of an unwritten, unmaterialized cell wins selection and changes output.
  The masked-`_cutfill` tests use the same poison. The reviewer's poisoned
  differential grid is folded in (`test_e2e_poisoned_differential_grid`,
  8 cases covering k 1/4/10, ragged + narrow tails, sub 8/16/64, 3 files,
  tiebreak ordinal+id, quantized ties, audit_every 0-3, NO_TOPK_SELECT
  on/off, Triton (CPU-patched) + portable top-k; each runs off vs on,
  unelided, overhead).
- **T2 GPU file.** Adds:
  - a poisoned run_compute differential;
  - an FTZ margin case (2^-63 weights, a large hot term summed first by the
    kernel and last by the bound, thr = the kernel's own score);
  - a zero-skip regression gate (prep + skip kernel + masked top-k with
    nothing dead must be <= 1.03x of off posting + top-k);
  - register / spill printouts for the skip posting kernel and the HAS_LM
    `_cutfill`;
  - quantile-extrapolated thresholds at 1f / 10f / 150f (oracle.py);
  - separate timings for BM, bound, prep, elided and unelided posting,
    overhead, and plain vs masked top-k.
- **A1 audit failures.**
  - An audit mismatch still turns skipping off and re-scores the slice
    unskipped. The run then RAISES `SkipAuditFailure` after the manifest is
    written (`compute.run_compute` end), unless
    `NOVA_BF_SPARSE_BLOCK_SKIP_AUDIT_WAIVE=1` (`block_skip.audit_waived`).
  - The lag is closed: until a slice with dead cells has been audited, each
    skip slice reads its dead count synchronously (one scalar), and the first
    slice with any dead cell is audited itself.
- **A2.**
  - The audit sample adds up to 64 rows that have dead sub-blocks, so a dead
    slice's audit always sees dead cells.
  - A key audit with no dead cells does not end the first phase. A dead
    slice whose sample (impossibly) had none ends it anyway, counted as
    `audits_without_dead`, so the reference never runs every slice.
  - Probes are queued, never overwritten.
- **A3.** Probe copies are issued and their events recorded on `cnt.device`'s
  current stream.
- **A4.**
  - The ELIDE env var accepts only on/off spellings (`overhead` raises).
  - The gate counts "skipping disabled for the run" instead of "enabled"
    after a disable.
  - `masked_topk` is counted after a successful masked launch only.
- **A5.** The (row, token) uniqueness precondition is documented in the module
  docstring. `NOVA_BF_SPARSE_BLOCK_SKIP_CHECK=1` verifies it per slice (on
  in the tests).
- **P1.** `atomic_max(..., sem="relaxed")` in the BM kernel.
- **P2 adaptive engagement.**
  - The async dead counts (queued pinned copies, polled by event) give the
    latest dead fraction.
  - While it is below `NOVA_BF_SPARSE_BLOCK_SKIP_MIN_DEAD` (0.10), slices
    build no tables, re-probing every `NOVA_BF_SPARSE_BLOCK_SKIP_REPROBE`
    (16).
  - Recorded under `block_skip.adaptive`.
  - Output-neutral (a declined slice is scored unskipped).
- **P3.** The posting kernel's hot section is a 2-D `[DB/SUB, SUB]` tile, and
  each sub-block's flag is broadcast along its row (no per-lane flag
  gather). Cold postings, which are gathered rows, still gather their flag
  per lane. The per-lane arithmetic order is unchanged; the bit-identity
  contract tests cover it.
- **P4.** Size arguments are `do_not_specialize` in all three skip kernels,
  and the skip compile key no longer includes sizes, so ragged tails
  neither recompile nor re-audit.

## Next steps (deliberately NOT done; wait for GPU numbers + own soundness review)

- **Widen the gate to dot+cosine sharing a raw matrix, or to `rows`
  subsets.** This needs a per-consumer threshold in raw-dot units: the min
  over readers, with per-row / per-query norm factors for cosine. The
  reviewer's point: the shared raw is the riskiest part of the soundness
  argument, so it should only be widened once the single-consumer win is
  measured.
- **Skip against the exchanged `_thr`.** The cross-worker threshold is a
  valid global prune. But the Triton and portable `pack_topk` paths treat
  below-threshold candidates differently, so per-rank partials could
  differ; it needs its own argument about partial-output identity.
- **Live-program compaction** (launching only programs with live
  sub-blocks). This changes grid shape and scheduling; worth it only if the
  GPU shows many fully dead programs still costing launch and flag reads.
- **Fusing the bound into the posting kernel.** This saves a launch and the
  live-mask round trip, but couples the bound's rounding to the kernel's
  register tile. It needs a re-derived proof and a GPU measurement of
  whether the bound is even significant.

## Round 3 (v2.2) changes

- **R1.** `NOVA_BF_SPARSE_BLOCK_SKIP_AUDIT_WAIVE` and `..._CHECK` use the strict
  on/off parser (`0/false/no/off/empty` mean off; unknown spellings raise).
- **R2.** A failed audit is stamped on every output of that run as
  `nova_bf.block_skip_audit_failed=true` (`results.SKIP_AUDIT_KEY`, beside the
  run fingerprint), even when the compute-side raise was waived. `merge`
  refuses any such partial through `_refuse`, so the existing
  `NOVA_BF_MERGE_FORCE` convention is the only waiver. A forced merge stamps
  both `merge_forced=true` and the audit-failed key. With threshold exchange
  active nothing can be forced (existing rule).
- **R3.**
  - Every environment-only knob (MIN_DEAD, REPROBE, BUDGET_MIB, CHECK,
    AUDIT_WAIVE) is parsed and validated once in `configure` (`read_env`). Bad
    values raise there, naming the variable.
  - `slice_reason`, the size check and the adaptive gate (`_poll_probes`) now
    run inside `try_skip`'s protection, so a skip-side error turns off only
    skipping.
- **P5.**
  - `do_not_specialize` remains only for `nblk` in the skip posting kernel.
    n_rows, stride_out and n_sub (alignment / mask constancy; n_sub also feeds
    `_bound`'s BM row loads and int8 LIVE stores) are specialised again in all
    three kernels.
  - The compile key carries their specialisation class (==1, %16==0), so each
    variant gets its own first-launch audit (at most a few per run).
  - Sizes >= 2^31 decline skipping (`size_reason`), so no int64 variant goes
    untracked.
- **P6.** `MIN_DEAD` default 0.15.

  Cost model, per slice:
  - With skipping off: P + C, where P is the posting kernel and C the top-k
    consumer.
  - With skipping on: B + P(1 - f) + o P + C(1 - g f), where B is BM + bound
    (prep), f the dead sub-block fraction, o the zero-skip overhead of the
    masked kernel, and g <= 1 how much of C the mask saves (rows / lanes not
    loaded).
  - Break-even: f* = (B + o P) / (P + g C).
  - With the reviewer's estimates (B ~ 0.1-0.15 (P + C), o <= 0.03,
    g ~ 0.6-1), f* ~ 0.15-0.2. The GPU timing test prints every term to
    replace these estimates.
- **G1.** The zero-skip regime uses a finite threshold (0, nothing dies since
  T >= SIG > 0). Gates are per component: skip posting kernel (overhead and
  on) <= 1.03x off, masked `_cutfill` <= 1.03x plain, and prep printed.
- **G2.**
  - Quantile thresholds use production rows per file (1.09M default,
    configurable) over a pool of corpus rows (the whole loaded file by
    default).
  - A regime whose rank falls below 1 is refused with a message, not clamped.
    With the synthetic 8192-row fallback only 1f is representable.
  - True late thresholds come from a GT parquet (`NOVA_BF_BLOCKSKIP_THRESHOLDS`).

## Round 4 (v2.3) changes

- **K1.** The constexpr and num_warps arguments of all three skip kernels now
  come from one set of helpers (`_bm_meta`, `_bound_meta`, `_post_meta`). The
  launches and `compile_key` both use them, so the key cannot drift from what
  runs.
  - The key now includes `_bound`'s NSP and num_warps.
  - It now covers every pointer the three kernels take (including
    `_block_max`'s PTR, the bound outputs, the threshold snapshot, and the
    query-plan tensors), plus the size specialisation classes.
- **K2.** The audit-failed refusal is now one helper, `merge._refuse_skip_audit`.
  - Under threshold exchange it cannot be forced: `forced and not
    exchange_active`, matching `_validate_one_run`. Before, it could be
    forced there.
  - In that case the hint says to recompute every rank, instead of
    suggesting NOVA_BF_MERGE_FORCE.
- **K3.** With an UNWAIVED audit failure, every output of the run goes to
  `<name>.audit_failed`. That covers single-node results and per-rank partials
  alike; the manifest records the quarantined name, and the run then raises.
  With a WAIVED failure the outputs keep their names, are stamped, and merge
  refuses them.
- **K4.** `sparse_block_skip.configure` (config + environment) now runs
  immediately after the per-run reset in `run_compute`, before queries or the
  corpus are touched. Every parse error names its variable (mode, SUB,
  ELIDE, AUDIT_EVERY, MIN_DEAD, REPROBE, BUDGET_MIB, CHECK, AUDIT_WAIVE).
- **K5.**
  - If `SkipMask.materialize()` fails (e.g. OOM), skipping turns off and the
    slice is re-scored in place with the unskipped hot kernel (`rescore`, set
    by `scores`). That writes every cell, so the output is unchanged
    (`block_skip.rescored`).
  - If the HAS_LM `_cutfill` launch fails (any non-OOM error), `pack_topk`
    calls `SkipMask.fail`: skipping turns off, the slice is materialized, and
    the PLAIN Triton top-k runs. The Triton top-k stays enabled for the run.

## Round 5 (v2.4)

- **L1.** The materialize-OOM `rescore` now calls `_launch_checked`, the same
  launch + compile-key + `_self_check` path as the ordinary unskipped slice.
  So a never-seen hot-kernel variant (e.g. a ragged tail of a new
  specialisation class) is self-checked before its output is used. A
  `KernelMismatch` there propagates (loudly) out of `materialize`.

## Open items (documented, not changed)

- **(a) The masked `_cutfill` has no runtime check.** It is the HAS_LM
  variant, and like the plain `_cutfill` it has no first-launch self-check;
  that is the same policy as the existing top-k kernel. The GPU test file
  (`test_identity_on_gpu`, the poisoned e2e differential) must pass on
  hardware before shipping.
- **(b) Quarantine does not touch older files.** A quarantined run writes
  `<name>.audit_failed` and leaves any OLDER file under the real name as it
  is (deliberately not deleted). Consumers must check the exit code or the
  manifest (`block_skip.audit_failures`, the quarantined `output_file`)
  rather than the mere presence of the real name.
- **(c) The BM buffer's pointer is not in the compile key.** It is always a
  fresh, aligned `torch.zeros` allocation, so its alignment class cannot
  vary and including it would have no effect.

## Results (CPU, Triton interpreter, v2)

- `test_sparse_blockskip_interpret.py`: 175 interpreted child tests passed (1 skipped: the runner itself)
  (kernel contract grid 90, elision, overhead, all-live/all-dead, unfull
  rows, < k overlaps, ties, fp32 round-up, kernel-vs-bound order with a
  positive control, huge weight, BM exactness 9, strict bound, masked
  `_cutfill` 13 + fully dead rows, unmasked `_cutfill` vs portable, lmask
  contract, portable materialize, robustness 9, audits, snapshot, end to end
  11). Parent process: 5 tests (config, env/effective mode, margin
  constants, gate, child runner).
- Regressions: `test_sparse_posting_triton_interpret.py` child 190 passed;
  `test_manifest`, `test_config_searches`, `test_compute_sparse`,
  `test_dead_row_poison`, `test_prune_gpu_contract`, `test_topk_prune`,
  `test_tiebreak`: 192 passed, 50 skipped (CUDA-only).
- GPU files: skipped (no CUDA).

### Mutations (by hand, reverted)

| # | mutation | caught by |
|---|---|---|
| 1 | margin removed (F = 1, SIG = 0) | `test_margin_constants`, `test_bound_test_is_strict`, `test_margin_covers_kernel_vs_bound_order`. NOTE: the fp32 bound sums in the same order as the hot-only kernel, so the random/tie/round-up tests do NOT catch this any more (they did with v1's fp64 bound); only the order-difference test exposes it, through a positive control |
| 2 | strict `<` -> `<=` | `test_bound_test_is_strict` |
| 3 | heap-full: k-th ignoring -inf slots | 4 e2e single-dot (k=5) |
| 4 | consumer gating removed | e2e dot+cos, two_k, filtered+unfiltered; `test_skip_member_gate` |
| 5 | cold postings not masked by live | 18 random contracts (non-elided) |
| 6 | elided hot store writes dead lanes | `test_elided_cells_are_never_written` x2 |
| 7 | `_cutfill` lane mask dropped | 13 masked-cutfill tests |
| 8 | rowany off (>1 instead of >0) | fully-dead-rows test + 4 e2e (CPU top-k) |
| 9 | BM atomic_max -> store | 45 (block max + contracts) |
| 10 | compute skips materialize on non-mask paths | `test_e2e_narrow_tail_slices_materialize` (added after the first run missed it) |
| 11 | audits never run | 14 (audit counters) |
| 12 | audit failure disables the whole kernel | 2 robustness tests |
| 13 | warm-up gate removed | `test_e2e_warm_up_gate` |
| 14 | threshold not snapshotted | `test_threshold_is_snapshotted` |
| 15 | hot-row BM off by one sub-block | 95 |

## Results, round 2 (v2.1, CPU interpreter)

- New interpreted suite: 190 child tests passed (77 unit + 113 grid/e2e, incl.
  the 8 poisoned differential grid cases), 1 skipped (the runner); parent 4.
- Mutations (applied, run, reverted), all caught:

| # | mutation | caught by |
|---|---|---|
| N1 | reviewer (a): compute's non-mask-path `materialize()` -> pass | narrow-tail e2e + grid[4], grid[7] |
| N2 | reviewer (b): `_cutfill` `offs // LSUB` -> `offs // (LSUB*2)` | 22 (masked cutfill + grid + e2e) |
| N3 | end-of-run audit raise removed | `test_e2e_audit_failure_fails_the_run` |
| N4 | first-phase sync audit removed (lag back) | `test_audits_see_dead_cells_and_repeat` |
| N5 | dead rows not added to the audit sample | `test_audit_sample_includes_dead_rows` |
| N6 | adaptive gate inverted | `test_adaptive_gate` |
| N7 | hot section unmasked (`lm = om`) | `test_elided_cells_are_never_written` x2 |
| N8 | probe queue overwritten | `test_probes_are_queued_not_overwritten` |
| N9 | ELIDE env lenient again | `test_env_override_and_effective_mode` |
| N10 | masked-topk count removed | 12 masked-cutfill tests |
| N11 | margin removed | constants, order-difference, strict |
| N12 | cold postings unmasked | 16 random contracts |
| N13 | `_cutfill` lane mask dropped | 13 |
| N14 | BM atomic_max -> store | 4 block-max tests |

## Results, round 3 (v2.2, CPU interpreter)

- New interpreted suite: 193 child tests passed (79 unit + 114 grid/e2e), 1
  skipped (the runner); parent 5.
- Mutations (applied in place, run, reverted), all caught:

| # | mutation | caught by |
|---|---|---|
| R1 | WAIVE / CHECK parsed with `bool(raw)` again | `test_env_knobs_parse_strictly` |
| R2a | merge no longer refuses audit-failed partials | `test_e2e_audit_failure_partial_is_refused_by_merge` |
| R2b | compute no longer stamps the audit failure | same |
| R3a | probe polling moved outside the try protection | `test_skip_side_error_in_probe_poll_disables_skipping_only` |
| R3b | env range validation removed | `test_env_knobs_parse_strictly` |
| P5 | compile key drops the size specialisation classes | `test_compile_key_tracks_specialisation_class_only` |
| P6 | MIN_DEAD default back to 0.10 | `test_env_knobs_parse_strictly` |

## Results, round 4 (v2.3, CPU interpreter)

- New interpreted suite: 199 child tests passed, 1 skipped (the runner);
  parent 5.
- Mutations (applied in place, run, reverted), all caught:

| # | mutation | caught by |
|---|---|---|
| K1 | `_bound`'s NSP / num_warps dropped from the key | `test_compile_key_covers_bound_nsp`, `test_bound_variant_change_is_audited` |
| K2 | audit-failed refusal forceable under exchange | `test_merge_refusal_hint_and_exchange` |
| K3 | no quarantine name | `test_e2e_audit_failure_fails_the_run` |
| K4 | skip configure moved back after thread pools / reads | `test_e2e_skip_config_fails_before_any_read` (first version of the test missed it; strengthened with an ordering sentinel) |
| K5a | materialize failure re-raised | `test_materialize_failure_rescores_unskipped` |
| K5b | masked top-k failure re-raised | `test_masked_cutfill_failure_falls_back_without_disabling_triton` |

## Results, round 5

- L1 test `test_rescore_self_checks_a_new_hot_variant`: passes; the mutation
  (rescore calling `_launch` directly again) is caught by it.
