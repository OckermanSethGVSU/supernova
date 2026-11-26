"""`bench run --dev` of the fixture experiments and the run-dir helpers the checks tests share."""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import fixture_runs
from bench.verify import agree
from bench.verify import compare as C


CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHAPES = ("dense", "sparse", "multivector", "filtered")


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NOVA_BF_", "NOVABF_", "PAPER_"))}
    env.update(HARNESS_NO_IMDS="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    return env


def bench(*args, env=None):
    return subprocess.run([sys.executable, "-m", "bench", *args], cwd=CB, env=env or _env(),
                          capture_output=True, text=True)


def _only_run(results, name):
    runs = sorted(os.listdir(os.path.join(results, name)))
    assert len(runs) == 1, runs
    return os.path.join(results, name, runs[0])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{shape: run dir}: this module's own copy of the shared, cached runs (fixture_runs)."""
    return fixture_runs.ok_runs(tmp_path_factory, "s5_results")


def checks(d):
    exp = json.load(open(os.path.join(d, "experiment.json")))
    return agree.run_checks(d, exp)


def copy_run(src, dst):
    """A run dir copy whose leg records point at the copy's own files (nova-bf result paths are
    absolute in counters.results)."""
    shutil.copytree(src, dst)
    for f in os.listdir(os.path.join(dst, "legs")):
        if f.endswith(".json"):
            p = os.path.join(dst, "legs", f)
            txt = open(p).read()
            open(p, "w").write(txt.replace(src, str(dst)))
    return str(dst)


def _npz_rewrite(d, search, system, fn):
    """fn(I, D) -> (I, D), applied to every round's saved top-k of one harness system."""
    for f in sorted(os.listdir(os.path.join(d, "legs"))):
        if f.startswith(f"{search}__{system}__r") and f.endswith(".topk.npz"):
            p = os.path.join(d, "legs", f)
            z = dict(np.load(p))
            z["I"], z["D"] = fn(z["I"].copy(), z["D"].copy())
            np.savez(p, **z)


def _nb_results(d, search, system="novabf"):
    out = []
    for f in sorted(os.listdir(os.path.join(d, "legs"))):
        if f.startswith(f"{search}__{system}__r") and f.endswith(".json") and not f.endswith("clock.json"):
            rec = json.load(open(os.path.join(d, "legs", f)))
            out += list(rec["counters"]["results"].values())
    return out


def _nb_rewrite(d, search, fn):
    """fn(ids: list[list[str]], scores: list[list[float]]) -> (ids, scores) on every round's
    nova-bf result parquet (schema and metadata kept)."""
    for p in _nb_results(d, search):
        t = pq.read_table(p)
        ids, sc = fn(t.column("hit_ids").to_pylist(), t.column("hit_scores").to_pylist())
        t2 = t.set_column(t.schema.get_field_index("hit_ids"), t.schema.field("hit_ids"),
                          pa.array(ids, t.schema.field("hit_ids").type))
        t2 = t2.set_column(t2.schema.get_field_index("hit_scores"), t2.schema.field("hit_scores"),
                           pa.array(sc, t2.schema.field("hit_scores").type))
        pq.write_table(t2.replace_schema_metadata(t.schema.metadata), p)


def _exact_ties(d, search, system, k):
    """(query, rank) pairs of `system`'s list whose two ids have EQUAL float64 scores, from the
    run's own oracle rescoring (compare's exact_scores via agree)."""
    exp = json.load(open(os.path.join(d, "experiment.json")))
    s = next(x for x in exp["searches"] if x["name"] == search)
    a = C.parse_args(agree.compare_args(exp, s, None, "uuid"))
    z = np.load(os.path.join(d, "legs", f"{search}__{system}__r1.topk.npz"))
    I = z["I"]
    need = [set(int(g) for g in row[:k] if g >= 0) for row in I]
    keys, rows, offsets, _ = C.corpus_index(a.corpus, need_ids=False)
    inputs = {"offsets": offsets, "keys": keys, "n": len(I), "qn": agree._queries_qn(a)}
    ex = agree._exact_for(a, need, inputs)
    return [(q, r) for q in range(len(I)) for r in range(k - 1)
            if I[q, r] >= 0 and I[q, r + 1] >= 0 and ex[(q, int(I[q, r]))] == ex[(q, int(I[q, r + 1]))]], ex


def _uuid(key, row):
    h = hashlib.md5(f"{key}:{row}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _filtered_rows(exp, search):
    import yaml
    cfg = yaml.safe_load(open(exp["filter"]["config"]))
    s = next(x for x in cfg["searches"] if x["name"] == search)
    return s["rows"]
