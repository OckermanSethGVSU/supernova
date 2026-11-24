# Experiments

Each experiment defines a dataset, search parameters, and the systems to
compare. Use `bench run <file>` to run locally or `bench launch <file>`
to run on a GPU node.

## Layout

| Folder | Description |
|---|---|
| `dense/` | Dense vector search |
| `sparse/` | Sparse vector search |
| `multivector/` | Multi-vector search (MaxSim) |
| `filtered/` | Structured and text-filtered search |
| `novabf/` | nova-bf configurations |

Data locations are written as `${READ}` and `${WRITE}`. 

## Experimental Workflow

1. **Verify correctness.** Run each system on a small subset and
   compare its results against an exact float64 reference.

   ```bash
   bench run <file> --set subset.files=4 --set protocol.rounds=1 --set checks.exact_scan=true
   ```

   Additional correctness checks are available through `bench check-copies`
   and `bench check-verify`.

2. **Tune configurations.** Sweep configuration parameters on a subset
   of the dataset to identify the fastest correct settings for each system.

   ```bash
   bench sweep <sweep file> --pick
   ```

   Record the selected settings in the experiment file.

3. **Run the experiment.** Launch the benchmark using the selected
   configurations.

   ```bash
   bench launch <file>
   ```

   Each experiment runs three rounds with rotated system ordering,
   fresh processes, and an untimed warmup. Results are verified,
   and median performance is reported.
