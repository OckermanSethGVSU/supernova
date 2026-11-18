# nova-bf tests

## Running them

```bash
make test-bf-fast            # everything except `slow` tests (day to day)
make test                    # the whole repo, nova-bf's full suite included
```

Both run the suite in one parallel pass with pytest-xdist: one worker per
~4 cores (at least 1, at most 16). Override the width with `BF_WORKERS`, e.g.
`make test BF_WORKERS=1` for a plain serial run.

By hand, from `python/nova-bf/`:

```bash
MALLOC_ARENA_MAX=2 pytest -q -n 6 --dist loadfile   # everything, parallel
MALLOC_ARENA_MAX=2 pytest -q -m timing              # timing tier solo, if one flakes
pytest -q tests/test_s3_auth.py                     # one file
```

- `--dist loadfile` keeps each file on one worker, so the session fixtures in
  `tests/parity/conftest.py` are built once per worker instead of per test.
- `MALLOC_ARENA_MAX=2` matters only under a `ulimit -v` cap. glibc reserves a
  64 MB arena per thread, the suite accumulates ~150 threads, and the virtual
  size then trips the cap ("can't start new thread" on every later test).

## Markers

| Marker | Meaning | Where it is assigned |
|---|---|---|
| `slow` | Unique but expensive on CPU (the interpreted Triton block-skip end-to-end tests and part of its random grid). Skipped by `-m "not slow"`. | `_SLOW` in `conftest.py`, and the grid in `test_sparse_blockskip_interpret.py` |
| `timing` | The verdict depends on thread or process scheduling: sleeps that open overlap windows, wall-clock deadlines, "really ran concurrently" counts, a GIL probe. They run in the normal parallel pass (clean in 3 of 3 runs at 6 workers); rerun them solo with `-m timing` if one fails, to tell load from a real bug. | `_TIMING` in `conftest.py` |

## Parallel workers

`conftest.py` pins each xdist worker to its own slice of the CPUs. nova-bf
sizes its Arrow, torch and scan pools to the CPUs it may use, so unpinned
workers each start pools for the whole machine. With 6 workers on 24 cores
that was slower than one process. `NOVA_BF_XDIST_PIN=0` turns pinning off.

## What the test harness skips, and how to get it back

These only remove waiting; no test logic or comparison changes.

| What | In tests | Switch |
|---|---|---|
| The EC2 metadata lookup when writing a manifest (`manifest.detect_compute`) | Fails instantly instead of after a 0.5 s timeout | `NOVA_BF_TEST_REAL_TIMINGS=imds` |
| The AWS SDK's last credential source, the EC2 instance role (`AWS_EC2_METADATA_DISABLED=true`) | Skipped, instead of 6-20 s of timeouts per S3 filesystem | `NOVA_BF_TEST_REAL_TIMINGS=imds` |
| The threshold exchange's 1 s gap between publishes | 0 s, unless a test passes `publish_min_interval_s` | `NOVA_BF_TEST_REAL_TIMINGS=spacing` |

`NOVA_BF_TEST_REAL_TIMINGS=1` restores all three.

The tests' Qdrant clients also turn HTTP keep-alive on. `qdrant_client`
disables it for `localhost`, so every query opened a new connection, and the
parity tests' tens of thousands of queries ran the machine out of local ports
once several workers ran at once.

## S3 credentials

The offline tests never contact AWS: they use fake bucket names, local
stand-ins and fakes. The live ones are opt-in, and take their path from the
environment so no real bucket path is kept in the repo: point
`NOVA_BF_LIVE_S3_URI` at a publicly readable prefix of parquet files with a
dense column (`NOVA_BF_LIVE_S3_COLUMN`, default `dense_embedding`).

```bash
NOVA_BF_LIVE_S3_URI='s3://<bucket>/<prefix>/?region=<region>' pytest -q tests/test_s3_auth.py -k live
RUST_FETCH_TEST_S3_ENDPOINT=http://localhost:19000 pytest -q rust_io/tests   # Rust reader vs a local minio
```

Credentials resolve as they do in a real run. The AWS SDK checks, in order:

1. environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_SESSION_TOKEN`);
2. `~/.aws` profiles, including SSO;
3. the EC2 instance role.

Keys in a config's `s3:` block, or `anonymous: true`, bypass that chain.

- **With your own credentials (env vars, a profile, SSO):** nothing to set.
  These resolve before step 3, which is the only step the harness skips.
- **On an EC2 instance using its instance role:** nothing to set either.
  `conftest.py` recognises EC2 from local system files
  (`/sys/class/dmi/id/sys_vendor` reads "Amazon EC2", or an `ec2...`
  hypervisor uuid) and leaves the metadata lookup on, because there it
  answers in milliseconds and it is where the role's credentials come from.
- **If detection ever misses** (an unusual instance type or container), set
  `NOVA_BF_TEST_REAL_TIMINGS=imds`.
- If you exported `AWS_EC2_METADATA_DISABLED` yourself, the harness leaves
  your value alone.

## GPU machines

About 630 tests run only with a GPU. The parallel suite has not been tried on
a GPU box yet, and several workers sharing one GPU may run out of memory or
contend for it. Use `make test BF_WORKERS=1` there until it has been.

## Parity suite

See `tests/parity/README.md` and `make parity`.
