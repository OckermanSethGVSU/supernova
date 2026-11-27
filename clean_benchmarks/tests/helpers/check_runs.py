"""The finished `bench run --dev` runs the checks tests read (every shape, shared and cached:
helpers.fixture_runs) and the run-dir helpers that plant a mistake in a copy of one."""

import hashlib
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import fixture_runs

SHAPES = ("dense", "filtered", "sparse", "multivector")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{shape: run dir}: this module's own copy of the shared, cached runs (fixture_runs)."""
    return fixture_runs.ok_runs(tmp_path_factory, "check_results", SHAPES)


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


def _uuid(key, row):
    """nova-bf's point id of a corpus row, written out here (an independent copy of load.uuid_of)."""
    h = hashlib.md5(f"{key}:{row}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
