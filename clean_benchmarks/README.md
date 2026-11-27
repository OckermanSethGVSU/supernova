# clean_benchmarks

A benchmarking framework for comparing vector search systems under a common experimental protocol. The framework provides consistent timing, correctness verification, parameter tuning, distributed experiment execution, and reproducible environments.

## Commands

```bash
# Run an experiment
bench run experiments/<name>.yaml

# Sweep parameters and select the best valid configuration
bench sweep experiments/sweeps/<name>.yaml --pick

# Run an experiment on SkyPilot
bench launch experiments/<name>.yaml

# Verify the experimental environment
bench check

# Process an existing run
bench summarize RUN_DIR
bench plot RUN_DIR
bench verify RUN_DIR [--exact-scan]
```

Use `--set KEY=VALUE` to override experiment settings, `--dev` for local CPU development, and `--dry-run` to inspect a SkyPilot launch.

## Experimental Workflow

1. **Validate:** Run each system on a small subset with exhaustive correctness checks.
2. **Tune:** Sweep system-specific search and I/O parameters. Select configurations that satisfy correctness requirements.
3. **Benchmark:** Run all systems on the full dataset using the selected configurations.

Experiments are defined in YAML files under `experiments/`.

## Timing

All systems follow the same timing protocol, implemented in `bench/timing.py`:

- A fresh process for each system and round.
- One untimed file warmup followed by one timed pass.
- Three rounds by default, with system execution order rotated between rounds.
- Median runtime reported using a common timing interface (`LegClock`).

## Correctness

The framework supports correctness verification for dense, sparse, multi-vector, and filtered search. Results are compared using float64 reference scores and numerical error bounds that account for float32 rounding.

Two verification modes are supported:

- **Default:** Validates returned IDs, filter compliance, and disagreements between systems across all queries.
- **Exhaustive (`checks.exact_scan: true`):** Additionally computes exhaustive float64 ground truth for a reproducible sample of queries (1,000 by default, or all queries).

Default verification cannot detect errors shared by all systems. Exhaustive verification addresses this limitation for the selected queries.

Verification is performed on first-round results, with failures recorded in `checks.json`. Exhaustive reference results are cached and reused across compatible experiments and sweeps.

## Repository Layout

| Directory | Description |
|---|---|
| `experiments/` | Experiment configurations and parameter sweeps |
| `bench/` | Experiment execution, timing, verification, and reporting |
| `baselines/` | Modifications to third-party libraries |
| `env/` | Pinned software environment and SkyPilot configuration |
| `tests/` | CPU correctness and integration tests |
| `results/` | Experiment records, verification results, and summaries |

## Environment

```bash
pip install -e .

# Activate and verify the pinned environment
. env/activate.sh
bench check

# Inspect the environment without enforcing requirements
bench check --report-only
```

The framework records experiment configurations, software versions, and code revisions to support reproducibility.

## Tests

Tests run on CPU and do not require cloud resources.

```bash
# Fast development tests
OMP_NUM_THREADS=2 python -m pytest -q -n 6 -m "not slow" tests

# Full correctness and integration suite
OMP_NUM_THREADS=2 python -m pytest -q -n 6 tests
```

Run the full suite before committing changes or launching GPU experiments.