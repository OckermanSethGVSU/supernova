# clean_benchmarks

One tool that runs each experiment with one command, times every system with the same
clock, checks every answer against the same exact oracle, and records enough to rerun it.
Every command works on CPU (`launch` needs `--dry-run` or a real SkyPilot setup; no test calls
the cloud).

## Commands

```bash
bench run    experiments/<name>.yaml            # every system, every round, checks, summary
bench sweep  experiments/sweeps/<s>.yaml [--pick]   # a grid of variants; --pick prints the YAML change
bench launch experiments/<name>.yaml [--dry-run]    # run on a SkyPilot node (env/node.yaml) under
                                                #   env/paper_run.sh; results synced back
                                                #   (a sweep file with --job sweep: bench sweep there)
bench sweep  experiments/sweeps/<name>_io.yaml --pick   # I/O settings: an io grid on the leg
                                                #   runner; prints each system's own `io:`, never edits
bench sweep  experiments/sweeps/<name>_events.yaml      # GPU events on vs off per engine (report only)
bench check                                     # is this the pinned paper environment?
bench run tests/fixtures/experiments/dense.yaml --dev # CPU dev box: tiny fixtures, env check report-only
                                                #   (pytest makes the fixture data; by hand:
                                                #   python tests/fixtures/novabf/make_fixtures.py tests/fixtures/novabf)
```

Also: `bench leg RUN_DIR ...` (one leg, what `run` starts), `bench summarize RUN_DIR`,
`bench plot RUN_DIR`, `bench check-copies`, `bench check-verify`. `--set KEY=VALUE` overrides
one experiment value for one run. A sweep runs one warm run per variant (`--rounds N` for
more); a paper `bench run` uses 3 rounds.

## Per-dataset workflow

1. **Logic check on a few files:** every system, the full float64 scan as the oracle.
   `bench run experiments/<name>.yaml --set subset.files=4 --set protocol.rounds=1 --set checks.exact_scan=true`
2. **Pick settings on a subset:** `bench sweep experiments/sweeps/<name>_io.yaml` (plus the
   per-library knob sweeps), then `bench sweep RUN_DIR --pick`. A variant that fails its checks
   is never picked. Paste the printed change into the experiment file and commit it.
3. **The full run:** `bench launch experiments/<name>.yaml` (all files, 3 rounds, median).

On a node, all three can reuse one cluster (same node, same environment).

## Timing

The rules live in `bench/timing.py`, not in the experiment files: one fresh process per leg
(one system, one round), an untimed 1-file warmup, ONE timed pass, R rounds (default 3) with
the leg order rotated by one each round, the median reported. Every system is timed by
`LegClock`; for the baselines it starts and stops at the harness's own timed-pass stamps.

## Checks (the oracle)

Every run checks each system against a float64 oracle on a sample of queries
(`bench/verify/agree.py`).

- **Default:** the oracle ranks the union of the systems' own lists exactly. It cannot see a
  row that every system missed.
- **`checks.exact_scan: true`** (opt-in, or `--set checks.exact_scan=true`): a full float64
  scan of the run's files for the sampled queries, so such a row is caught. Dense streams one
  file at a time through a running top-k; other modalities rescore every row and refuse above
  2e7 query-row pairs.
- **Always on:** a dense query where every system returns fewer than min(k, rows) hits fails;
  multivector likewise, counting finite-score hits against min(k, documents with tokens).

`bench check-verify` is a separate check: the harness's own `--verify` run of each engine.

## Layout

| dir | what |
|---|---|
| `experiments/` | WHAT is measured: one YAML per experiment; `novabf/` its nova-bf configs; `sweeps/` few-file tuning grids |
| `bench/` | HOW it is measured: the one Python package (`pip install -e .`, no dependencies) |
| `baselines/` | every change made to third-party libraries (`baselines/README.md` lists them) |
| `env/` | the pinned environment: `versions.env`, `build.sh`, `activate.sh`, `check.py`, `node.yaml`, lock files |
| `tests/` | correctness tests, CPU by default; `tests/fixtures/` their experiment files (`experiments/`: tiny CPU ones plus copies of the real ones; tests never read the top-level `experiments/`) and data (not committed: `make_fixtures.py` writes it, and pytest does so when it is missing) |
| `results/` | written only by bench: `results/<experiment>/<UTC>_<sha>/` (data git-ignored, summaries tracked) |


## Environment

```bash
. env/activate.sh           # on a node built by env/build.sh; puts the env's bin first on PATH
bench check                 # fails unless the stack, python, nova-bf and bench are the env's
bench check --report-only   # print every row, never fail (a dev box)
```

## Tests

Two tiers, both CPU, run from this directory under a memory cap. The fast tier is for
iterating (about 1-2 min at `-n 6`). It covers every test file except the two that only hold
end-to-end comparisons (`test_filtered_nova_parity.py`, `test_s0_verdicts.py`). The full suite
(about 12 min) is the gate before a commit or a GPU run. Tests marked `slow` take more than
about 2 s, run `bench run` / `bench sweep --dev` end to end, drive nova-bf, or regenerate a
reference, or use a module fixture that does one of these.

```bash
# fast tier (iterating)
systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env OMP_NUM_THREADS=2 \
  python -m pytest -q -p no:cacheprovider -n 6 -m "not slow" tests
# full suite (the gate)
systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env OMP_NUM_THREADS=2 \
  python -m pytest -q -p no:cacheprovider -n 6 tests
```