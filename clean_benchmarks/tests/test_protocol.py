"""bench/timing.py's rules and leg.plan: the rotation, the reserved harness flags, the clock
json of every harness leg, the protocol constants and the filtered io placeholder (no process
is run)."""

import os

from bench import experiment as E
from bench import leg as L
from bench import timing as P
from helpers.protocol_plans import _load

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")


def test_rotated_is_timing_rotate():
    assert P.rotated(["a", "b", "c"], 1) == ["a", "b", "c"]
    assert P.rotated(["a", "b", "c"], 2) == ["b", "c", "a"]
    assert P.rotated(["a", "b", "c"], 3) == ["c", "a", "b"]
    assert P.rotated(["a", "b", "c"], 4) == ["a", "b", "c"]


def test_harness_reserved_flags_cannot_come_from_a_system():
    from bench.systems import base as SB
    for spec in SB.SYSTEMS.values():
        for k in spec.knobs.values():
            assert k.flag not in P.HARNESS_RESERVED


def test_every_harness_leg_gets_the_clock_json(tmp_path):
    e = _load("dense_coyo", [])
    spec = L.plan(e, "coyo", "faiss", 1, str(tmp_path))
    i = spec["argv"].index("--clock-json")
    assert spec["argv"][i + 1] == os.path.join(str(tmp_path), "legs", "coyo__faiss__r1.clock.json")
    assert spec["argv"][-6:] == list(P.HARNESS_PROTOCOL_ARGS)


def test_protocol_constants():
    assert P.HARNESS_PROTOCOL_ARGS == ("--warmup", "1", "--warmup-files", "1", "--repeats", "1")
    assert (P.WARMUP_FILES, P.REPEATS, P.ROUNDS_DEFAULT) == (1, 1, 3)


def test_filtered_io_is_a_refused_placeholder_until_tuned():
    """the filtered io numbers are TODO (bench run refuses them)."""
    for name in ("filtered_structured", "filtered_text"):
        e = _load(name, [])
        ph = E.placeholders(e)
        assert sorted(p.split(" ")[0] for p in ph) == ["io.files_in_flight", "io.gets", "io.part_mib"], ph
        assert (e["io"]["read_mode"], e["io"]["range_scope"]) == ("native", "total")
    e = _load("filtered_structured", ["io.files_in_flight=4", "io.gets=512", "io.part_mib=16",
                                   "io.chosen_from=sweeps/x/y"])
    assert not E.placeholders(e) and e["io"]["chosen_from"] == "sweeps/x/y"
