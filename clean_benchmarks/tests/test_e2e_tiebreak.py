"""The cold e2e leg can order ties by a corpus field (e2e.tiebreak); the timed rounds keep
nova-bf's position order, and the e2e check then compares scores only."""
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from bench import experiment as E
from bench import leg as L
from bench.verify import load

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COYO = os.path.join(CB, "tests", "fixtures", "experiments", "dense", "coyo.yaml")


def _coyo(tiebreak):
    raw = E.load_raw(COYO)
    raw["e2e"]["tiebreak"] = tiebreak
    return E.finish(raw, COYO)


def test_the_default_is_position_order():
    raw = E.load_raw(COYO)
    raw["e2e"].pop("tiebreak", None)
    assert E.finish(raw, COYO)["e2e"]["tiebreak"] == {"mode": "ordinal"}


@pytest.mark.parametrize("bad", [{"mode": "id"}, {"mode": "x"}, {"mode": "ordinal", "field": "id"},
                                 {"mode": "id", "field": "id", "extra": 1}])
def test_a_bad_tiebreak_is_refused(bad):
    with pytest.raises(E.ExperimentError, match="e2e.tiebreak"):
        _coyo(bad)


def test_only_the_e2e_leg_breaks_ties_by_the_field(tmp_path):
    e = _coyo({"mode": "id", "field": "id"})
    cold = L.plan(e, None, "novabf", None, str(tmp_path), e2e=True)
    cfg = yaml.safe_load(open([a for a in cold["argv"] if a.endswith(".yaml")][0]))
    assert cfg["params"]["tiebreak"] == "id" and cfg["corpus"]["id_column"] == "id"
    r1 = L.plan(e, e["searches"][0]["name"], "novabf", 1, str(tmp_path))
    cfg1 = yaml.safe_load(open([a for a in r1["argv"] if a.endswith(".yaml")][0]))
    assert cfg1["params"]["tiebreak"] == "ordinal" and "id_column" not in cfg1["corpus"]


def test_same_scores_ignores_ids_but_not_scores(tmp_path):
    def write(name, ids, scores):
        p = tmp_path / name
        pq.write_table(pa.table({"hit_ids": [ids], "hit_scores": pa.array([scores], pa.list_(pa.float32()))}), p)
        return str(p)
    a = write("a.parquet", ["x", "y"], [0.9, 0.9])
    b = write("b.parquet", ["y", "x"], [0.9, 0.9])
    c = write("c.parquet", ["x", "y"], [0.9, 0.8])
    assert load.same_scores(a, b) and not load.same_table(a, b)
    assert not load.same_scores(a, c)
