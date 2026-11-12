# nova-bf parity harness

Three-way correctness checking for nova-bf ground truth: **nova-bf** vs a
**plain-Python reference** vs a **live Qdrant exact search**, across all three
modalities (dense, sparse, multivector) and the whole filter language.

## Why three

Two implementations agreeing proves only that they agree. The two oracles here
fail differently on purpose:

| oracle | what it catches | needs |
| --- | --- | --- |
| `naive.py` | wrong *semantics* — a metric, a norm, a filter rule | nothing |
| `qdrant_ref.py` | wrong *fidelity to the engine the GT grades* | a Qdrant server |

They also check each other. During development the naive reference claimed a
sparse point with no shared token scores 0; nova-bf and Qdrant both said it is
not a candidate at all. Two independent oracles disagreeing with one
implementation localizes the error in one step — which is the whole argument
for paying for the second one.

## Running it

Everything is local. From `python/nova-bf/`:

```bash
# naive half only — no server, runs anywhere
pytest tests/parity -q

# add the live-Qdrant half
docker run -d --name qdrant-parity -p 6333:6333 qdrant/qdrant
QDRANT_URL=http://localhost:6333 pytest tests/parity -q
```

Without a reachable Qdrant the `@pytest.mark.qdrant` tests skip and the rest
still runs, so this is safe in CI.

## Running it on a GPU

Nothing to port. The suite parametrizes over whatever devices the machine has:

```bash
pytest tests/parity -q                       # cpu + cuda if present
NOVA_BF_PARITY_DEVICES=cuda pytest tests/parity -q   # cuda only
```

On a GPU box every case above runs twice, once per device, and
`test_parity_devices.py` wakes up to assert the extra things only a GPU can be
asked:

- CUDA and CPU select the **same** documents from the same bytes (membership
  exactly; scores to tolerance, since cuBLAS tiles differently);
- CUDA still agrees with both oracles directly, so a GPU-only defect cannot
  hide behind a CPU run sharing the same bug;
- `multivector_kernel="triton_reduce"`, the fused ragged MaxSim reduction,
  matches the portable torch path — this code is *unreachable* on CPU;
- `allow_tf32=True` does not reorder the ranking (it is explicitly **not**
  held to bit-equality: TF32 keeps 10 mantissa bits, and a test claiming
  otherwise would be wrong).

Device pinning goes through `NOVA_BF_DEVICE` (`cpu`/`cuda`), which
`compute._select_device` honors. Asking for `cuda` where torch sees no GPU is
an error rather than a silent downgrade — otherwise a device-comparison test
would pass by comparing CPU against CPU.

## What is covered

`cases.py` is the cross-product of

- **7 modality/metric pairs** — dense × {dot, cosine, euclidean}, sparse ×
  {dot, cosine}, multivector × {dot, cosine};
- **23 filters** — no filter; each static condition kind (`match`,
  `match` as MatchAny, numeric `range`, float `range`, `match_text`, datetime
  `range`, a nullable field); each group (`must`, `should`, `must_not`, and all
  three at once); each per-query condition kind (`match_from_query` scalar and
  list, `range_from_query`, `match_text_from_query`, a per-query datetime
  bound), a static/per-query mix, and six shapes mixing per-query text with
  per-query structured conditions.

= 161 cases, each asserted against both oracles **and** against an independent
check that the filter actually restricted something. That third assertion is
not ceremony: a filter that silently matched every row would make both parity
checks pass, because both engines would return the same unfiltered ranking.

`test_parity_invariance.py` covers the knobs the matrix holds fixed — tiling
(`*_batch_size`), search sharing (solo vs. one shared pass), and `rows`
query-row subsets — each re-checked against the oracles rather than only
against another nova-bf run.

`test_parity_combinations.py` covers **which searches share a run**, because
that changes the code that runs. All seven non-empty subsets of {dense,
sparse, multivector} × three filter regimes, plus spec-order invariance and one
filter shared across vector types.

The regime that matters most is the one the matrix structurally cannot reach.
If any search of a vector_type is unfiltered, that vt's batch grid is the whole
file; if EVERY one is filtered, the grid becomes the union of their surviving
rows (`_union_keep`) and every reported row number round-trips through that
compaction. The matrix always contains a `nofilter` spec, so it only ever
exercises the first branch. Injecting a bug into `_union_keep` (keep only the
first filter's rows) leaves `test_parity_matrix.py` and
`test_parity_invariance.py` passing 398/398 and fails the combinations suite 21
times — which is the argument for the file existing.

`test_parity_sharded.py` covers the shape production actually runs in: `compute
--num-jobs W --job-rank r` on W machines, then one `merge` reducing the
partials. Nothing else in the harness shards, so this is the only place the
partial/merge path is checked against the oracles at all. W = 1 (degenerate),
3 (does not divide the 4 corpus files) and 5 (more ranks than files, so some
rank gets nothing and must still write a well-formed empty partial), under both
tie-break rules. Plus the merge guards that no comparison could ever catch,
because a corpus subset's exact top-K is a perfectly self-consistent answer:
merge must refuse a search that lost one partial, a run missing a rank (rank 1
and the last rank, with the threshold exchange on and off — two different
guards), and partials from two different runs — each by `RuntimeError` and the
message naming the short search / missing rank / stale partial, not by "some
exception mentioning `run`". And the stride contract directly
(`test_rank_r_reads_exactly_the_stride_files`, W = 3 and 5 over 11 files, one
empty): with k above the corpus size each rank's partial lists every row it
scored, which must be exactly the rows of files `i % W == r`. Every merged
comparison is blind to how the files were split, so a chunked partition
(mutant `shard_chunking`) passed everything else.

Exact ties ACROSS ranks are in `test_parity_layouts.py`
(`test_exact_ties_survive_a_sharded_merge`): the integer tie corpus through
compute × W ∈ {2, 3, 4} + merge, both tie-breaks, dot and euclidean, no filter
and a static one, k ∈ {1, 2, 4, 7, 13}, asserted as exact id sequences against
the exact oracle. That is the only place the merge's own tie handling (its
lexsort on the tie key, the cutoff-ambiguity escalation) is reached with a
real tie; reversing the tie key or skipping the ambiguity check passed the
whole suite before it (mutants `merge_tie_reversed`,
`merge_cutoff_amb_skipped`). A fixture guard checks that the K-th score is
tied across ranks for most queries at every W.

`test_parity_kernels.py` covers the hand-written Triton kernels — `topk_triton`
(pre-top-K with the tie-break applied during selection) and `merge_triton` (the
running top-K fold) — against the portable paths they replace, end to end
through `run_compute` via the `NOVA_BF_NO_TOPK_KERNEL` /
`NOVA_BF_NO_FOLD_KERNEL` switches nova-bf already exposes to operators. These
are asserted BIT-IDENTICAL, not to tolerance: a kernel is an optimization of a
selection rule, not a different one. CUDA-only, plus a check that the kernels
were actually reached (every comparison passes trivially if a kernel silently
declined and fell back to the very path it is being compared against). The
tie-break-rule coverage in that file runs on any device.

`test_parity_mask_height.py` covers the three query-axis heights a run juggles
— the queries file, each vector_type's row union, and each filter's own row
union (the mask-height memory work). They coincide on most fixtures, which is
what makes confusing them dangerous: a mask read at the wrong height masks the
wrong queries and does not raise. It uses a taller query file than the rest of
the harness on purpose — the masks are bit-packed along the query axis, so at 8
queries every height is one byte and a wrong height is undetectable.

## The oracles read the RAW filter

Both oracles parse the filter dict exactly as a YAML config writes it —
`naive.parse_filter` and `qdrant_ref.to_qdrant_filter` — never nova-bf's
validated `Filter` model (`_filter_from_dict` validates through nova-bf, then
returns the raw dict). Before this, both oracles were fed nova-bf's own date
conversion and range coercion, and a ±1 µs bug in
`dates.parse_scalar_epoch_us` (`int(dt.timestamp() * 1e6)`) passed every
test. naive's RFC-3339 parser is now integer-only and hand-written; Qdrant
gets the raw bound string. `runner.build_config` deep-copies specs, because
the date prepass rewrites bounds in place and used to turn `cases.FILTERS`'
strings into nova-bf floats after the first run.

`compare.assert_scores_agree` also checks that each side has unique ids, is
best-first, and has the same length as the other (an exact top-K has one
length, so a length difference always means the two disagree on
eligibility). It only accepts two empty lists with `allow_empty=True`.

## Score tolerance

One number, `compare.SCORE_RTOL` = 1e-5, applied as `tol * (1 + |score|)`
(relative for large scores — so euclidean, whose scores are −distance, gets a
distance-scaled allowance — absolute near zero), for every metric. The
measured worst CPU error against naive's float64 loops is 3.8e-7 (dot),
1.1e-7 (cosine), 1.2e-7 (euclidean, shared-Gram expansion included), so the
default has ~25x headroom, while scoring fp16-rounded inputs (≈1e-4) fails it.
The old per-metric 2e-4 / 2e-5 / 2e-3 let exactly that through for euclidean
(mutant `eucl_fp16_inputs_shared`). A GPU (cuBLAS fp32, TF32 off) run must
pass at the default; if one does not, `NOVA_BF_PARITY_SCORE_RTOL=<measured>`
is the single override — record the measured error with it rather than
widening the default. TF32 tests carry their own explicit tolerance.

## Query ids are not positions

Every fixture used to name its queries `str(index)`, so writing the query's
POSITION instead of its id was invisible (`read_results` read `int(qid)` back
as the index). `runner.run` now maps written ids back through the dataset's
own `qid`s (`runner.qid_index`), failing on an id that names no query or a
query written twice, and `test_parity_qids.py` runs the shared corpus with ids
`q1000, q999, …` (not integers, not in index or lexicographic order) through a
matrix spread (every modality × none / static / per-query filters, vs naive and
Qdrant), a sharded merge (W = 3, 5) and a `rows` subset. Mutant
`out_qids_positional` was killed before only by an unrelated tokenizer test.

## Float bounds on stored values

`test_parity_float_bounds.py`: every `range` operator (and both two-sided
forms, and a `must_not` of `gt`) with the bound EQUAL to a stored float64
`rating` / nullable `weight` value, at k above the corpus size, compared as
exact eligible sets against the naive predicate and Qdrant, plus the row
holding the bound pinned in (`gte`/`lte`) or out (`gt`/`lt`). The matrix's
float range sits between stored values, where `gt` and `gte` agree — mutant
`float_range_gt_as_gte` survived everything else.

## Known divergences are narrowed, not xfailed

`test_parity_sparse_edges.py` used to pin its known divergences (unfiltered
Qdrant dropping 0.0-score sparse candidates; nova-bf not counting a
0.0-weight QUERY token as an overlap) as strict xfails over whole cells, which
would also have absorbed any unrelated failure in those cells. Now each cell
drops exactly the documented rows — derived from the data, never from an
engine's output — from both full candidate lists, cuts both to k, and is
compared in full; separate tests assert the divergence itself (who keeps
which rows, at 0.0, in exactly the measured cells). NaN inputs are a reader
error now (`test_nan_rows_are_an_input_error`,
`test_a_nan_sparse_row_is_an_input_error`); no NaN xfail remains. The only
strict xfails left in `tests/` are `test_dates_exact_us.py`'s two
"left for a decision" date-format divergences; the tokenizer's `DIVERGENT`
table is empty.

## Mutation runs

Run against a mutated copy with BOTH `PYTHONPATH=<copy>/src` and
`pytest -o pythonpath=<copy>/src` (the ini `pythonpath = ["src"]` otherwise
wins), plus `NOVA_BF_PARITY_EXPECT_SRC=<copy>/src`: `tests/conftest.py` then
refuses to run (UsageError) unless `nova_bf` was imported from inside that
directory — compared as paths, so a sibling `<copy>/src_orig` does not pass a
string-prefix check — for every test under `tests/`, not just the parity
package.

## Edge coverage (`cases.EDGE_FILTERS`, `DATE_FILTERS`)

The corpus has an EDGE payload (its own rng, so the original columns are
unchanged): `event_at` (µs RFC-3339, ~half pre-1970, with instants pinned
where float arithmetic is off by one µs), nullable `score_i` / `flag` /
`weight`, `note` text with null / "" / punctuation-only rows, and a
dictionary-encoded `lang_dict`. The per-query columns contain nulls, values
no row holds (dead queries), empty and null lists, and empty / token-less /
multi-token phrases. The edge filters are kept out of the matrix, because
many of them are dead or short on purpose. They run at k = 1, 7, 400 in one
pass on every modality, in four arms (whole-file / compacted grid × FilterPlan
/ legacy routing) and sharded. Every path-specific test uses `runner.probe`
(source-line hits by snippet, plus call counters) to assert that the path it
is named for actually ran.

## Measured Qdrant semantics (1.18.2), used by naive

| input | Qdrant |
| --- | --- |
| `match_text` "" or "!!!" | matches nothing (so `must_not` excludes nothing, `should` adds nothing) |
| `MatchAny([])` | matches nothing; `must_not` keeps all |
| `match` on int with `true`, on bool with `1`, on keyword with `1` | no match (type-strict) |
| `match` 3.0 on an int field | HTTP 400 (not expressible) |
| fractional `range` on an int field | numeric comparison (gt 2.5 keeps 3) |
| null / missing field | never satisfies a condition; `must_not` keeps it |
| datetime | integer µs; sub-µs digits truncated; offsets, no-zone, date-only and space-separated literals accepted |
| `{}`, `should: []` | no constraint |
| `has_id: []` | must: nothing, must_not: all — the translation used for a null per-query value |

Non-ASCII tokenization (combining marks, circled letters, final sigma, dotted
İ) is checked against the live engine in `test_parity_tokenizer.py`; no
divergences are known (a future one would be a strict xfail there).

## Extending it

Add a row to `cases.FILTERS` and it is immediately tested against all seven
modalities on every device, by both oracles. Add a `(vector_type, metric)` to
`cases.MODALITIES` and it picks up all twenty-three filters — you will also need
a named vector for it in `qdrant_ref.create_collection` and a branch in
`naive.score_one`.

Keep `naive.py` dumb. Its only value is being unable to share a bug with
nova-bf, and it loses that the moment it imports a helper from `nova_bf` or
gets rewritten in vectorized numpy.

## Layout

```
naive.py       plain-Python oracle: scoring, filters, top-K, and a memoizing Oracle
qdrant_ref.py  live-Qdrant oracle: one collection, all modalities, filter translation
corpus.py      the shared seeded synthetic dataset (parquet + in-memory truth)
cases.py       the filter × modality matrix
runner.py      running nova-bf on a pinned device, reading results back
compare.py     boundary-aware comparison (why exact id equality is the wrong assertion)
devices.py     which devices this machine can answer on

test_parity_matrix.py        161 cases vs both oracles, + the filter-really-filtered guard and Qdrant eligibility counts
test_parity_combinations.py  which searches share a run (all 7 vector_type subsets × 3 regimes)
test_parity_invariance.py    tiling, solo-vs-shared, `rows` subsets
test_parity_sharded.py       compute --num-jobs → merge, the merge refusal guards, the stride contract
test_parity_qids.py          query ids that are not row positions (matrix spread, sharded, `rows`)
test_parity_float_bounds.py  static float `range` bounds exactly on stored values, every operator
test_parity_kernels.py       the Triton kernels vs the portable paths (bit-identical)
test_parity_mask_height.py   the three query-axis heights, on a taller query file
test_parity_devices.py       CPU vs CUDA, triton_reduce MaxSim, TF32
test_parity_ordinals_paths.py all three tiebreak='id' ranking tiers agree end to end
test_qdrant_gate.py          the loud-skip gate itself (it had no coverage and no teeth)
test_parity_regimes.py       dead / short / k=1 / k>corpus / mixed-k, baseline+compacted, live+legacy, sharded W=3/5
test_parity_dates.py         µs-exact date bounds at stored instants, static + per-query, vs naive + Qdrant
test_parity_filter_paths.py  budget fallback, `rows` narrowing, windowed gather, struct_union, Arrow tokenizer, legacy matrix
test_parity_tokenizer.py     Unicode `match_text` vs measured Qdrant answers (divergences are strict xfails)
test_parity_layouts.py       storage dtypes / file layouts / md5 ids / exact ties (unsharded and sharded) / NaN
test_parity_sparse_edges.py  the sparse zero-gate and edge rows; Qdrant divergences narrowed + pinned
test_parity_knobs.py         batch / merge knobs, incl. the ranged merge reader (gate lowered to reach it)
test_harness_strength.py     compare.py and naive.py fed wrong answers: each must reject them
```

## A note on what is deliberately NOT asserted

Two mutations pass on purpose, and both would be tempting to "fix":

- `_union_keep` degenerating so it stops compacting the row union;
- a per-query filter mask unpacked TALLER than its own height.

Neither changes an answer — the first is slower, the second allocates padding
rows nothing reads (and is exactly what the code did before the mask-height
narrowing). Tests that failed on these would be pinning the optimizations
rather than the results, and would then have to be rewritten every time
someone tunes one.
