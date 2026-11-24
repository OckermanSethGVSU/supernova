import functools
import os
import subprocess
import sys

# ulimit -v + faiss's default (24) OMP threads segfaults on the dev box; keep
# faiss small and deterministic for every test. Must run before faiss import.
os.environ["OMP_NUM_THREADS"] = "2"
# no EC2 instance-metadata lookup from tests (rows' instance_* stay blank, as
# off EC2); tests/test_paper_stack.py drives the lookup through a fake opener
os.environ["HARNESS_NO_IMDS"] = "1"
# nor a launcher's stack identity (a test run inside a paper job): tests set it themselves
for _k in ("PAPER_STACK_SHA", "PAPER_CODE_SHA", "PAPER_ENV_PREFIX", "PAPER_RUN_DIR",
           "PAPER_LAUNCH_STUBBED", "PAPER_GPU_LOG"):
    os.environ.pop(_k, None)
# ... so the suite's GPU tests (a GPU node, no launcher) are not refused as off the
# paper stack; test_the harness's refusal tests unset it
os.environ["HARNESS_ALLOW_UNPINNED_STACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the private S3 prefixes (bench/paths.py): a checkout without experiments/local_paths.env gets dummy
# values, so experiments that name them still plan; a real file or environment value wins.
# EXAMPLE_PATHS: the dummies in use (set here, or inherited from the pytest-xdist controller
# that set them); never private, so test_no_private_paths skips them
from bench import paths as _paths  # noqa: E402

EXAMPLE_PATHS = {}
for _k, _v in (("READ", "s3://example-bucket/datasets"), ("WRITE", "s3://example-bucket/scratch")):
    _env = _paths.ENV[_k]   # BENCH_READ / BENCH_WRITE: used when no section gives a value
    if not os.environ.get(_env) and not any(_k in sec for sec in _paths.read_file().values()):
        os.environ[_env] = _v
    if os.environ.get(_env) == _v:
        EXAMPLE_PATHS[_k] = _v
# the test folders: test modules import each other's helpers by name (test_run,
# test_h2d, ...), also when one file runs alone; readhooks / conftest live here in tests/
for _d in ("data", os.path.join("data", "formats"), os.path.join("data", "filters"),
           "systems", "verify", "record", "report"):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), _d))

# the harness's own import order: torch (when installed; a CPU-only torch or none is
# fine) before faiss, so every test process maps the CUDA libraries as a run does
from bench.record import cudalibs  # noqa: E402

cudalibs.torch_first()
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pytest  # noqa: E402

import make_dummy  # noqa: E402

faiss.omp_set_num_threads(2)

D = 32
# uneven file sizes, incl. one smaller than k, so offsets and -1 padding matter
ROWS = [1200, 3, 700, 2000, 1, 950]


@functools.lru_cache(maxsize=None)
def arrow_default_io_threads():
    """Arrow's default IO pool size in a FRESH process (8 unless
    ARROW_IO_THREADS says otherwise): what an unconfigured reader starts with."""
    out = subprocess.run([sys.executable, "-c", "import pyarrow; print(pyarrow.io_thread_count())"],
                         capture_output=True, text=True, check=True)
    return int(out.stdout)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Six parquet files of unit-norm D-dim vectors; returns (dir, concat)."""
    out = tmp_path_factory.mktemp("corpus")
    xs = []
    for i, n in enumerate(ROWS):
        x = make_dummy.unit_vectors(np.random.default_rng([7, i]), n, D)
        make_dummy.write_vectors(str(out / f"part-{i:05d}.parquet"), x)
        xs.append(x)
    return str(out), np.concatenate(xs)


@pytest.fixture(scope="session")
def queries():
    return make_dummy.unit_vectors(np.random.default_rng(99), 150, D)


@pytest.fixture(autouse=True)
def restore_arrow_pools():
    """run.main() resizes Arrow's process-global pools; undo it per test."""
    cpu, io = pa.cpu_count(), pa.io_thread_count()
    yield
    pa.set_cpu_count(cpu)
    pa.set_io_thread_count(io)

# ---- fixture data: generated on demand (the parquet files are not committed) ----------------
FIXTURE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "novabf")
FIXTURE_KINDS = ("dense", "sparse", "multivector", "filtered")


def ensure_fixture_data():
    """Write the CPU fixture corpora (tests/fixtures/novabf/<kind>/) if any is missing, with
    their deterministic generator; a lock keeps parallel workers from writing at once."""
    if all(os.path.isdir(os.path.join(FIXTURE_DATA, k)) for k in FIXTURE_KINDS):
        return
    import fcntl
    import subprocess
    with open(os.path.join(FIXTURE_DATA, ".generate.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all(os.path.isdir(os.path.join(FIXTURE_DATA, k)) for k in FIXTURE_KINDS):
            subprocess.run([sys.executable, os.path.join(FIXTURE_DATA, "make_fixtures.py"),
                            FIXTURE_DATA], check=True, stdout=subprocess.DEVNULL)


def pytest_configure(config):
    ensure_fixture_data()
