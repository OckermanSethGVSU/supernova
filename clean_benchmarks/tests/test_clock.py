"""LegClock: arithmetic, refused out-of-order calls, no time from a failed/unfinished leg."""

import pytest

from bench.timing import ClockError, ClockHooks, LegClock


class FakeNow:
    def __init__(self, *ts):
        self.ts = list(ts)

    def __call__(self):
        return self.ts.pop(0)


def test_seconds_excludes_pauses():
    c = LegClock(now=FakeNow(10.0, 12.0, 15.0, 16.0, 16.5, 20.0))
    c.start(); c.pause(); c.resume(); c.pause(); c.resume(); c.stop()
    # 10 -> 20 is 10 s, paused 3 + 0.5
    assert c.seconds == pytest.approx(6.5)
    assert c.paused_seconds == pytest.approx(3.5)
    rec = c.record()
    assert rec["state"] == "stopped" and rec["pauses"] == 2
    assert rec["events"] == ["start", "pause", "resume", "pause", "resume", "stop"]
    assert rec["wall_seconds"] == pytest.approx(10.0)
    assert rec["segments"] == pytest.approx([2.0, 1.0, 3.5])


def test_real_clock_is_monotonic_and_positive():
    c = LegClock()
    c.start(); c.stop()
    assert c.seconds >= 0.0


@pytest.mark.parametrize("calls, bad", [
    ([], "pause"), ([], "resume"), ([], "stop"),
    (["start"], "start"), (["start"], "resume"),
    (["start", "pause"], "pause"), (["start", "pause"], "stop"), (["start", "pause"], "start"),
    (["start", "stop"], "start"), (["start", "stop"], "pause"), (["start", "stop"], "resume"),
    (["start", "stop"], "stop"),
])
def test_out_of_order_calls_are_refused_and_fail_the_clock(calls, bad):
    c = LegClock()
    for name in calls:
        getattr(c, name)()
    with pytest.raises(ClockError, match=f"{bad}\\(\\) while"):
        getattr(c, bad)()
    assert c.state == "failed" and bad in c.failure
    with pytest.raises(ClockError):
        c.seconds
    assert "seconds" not in c.record()
    # a failed clock refuses everything afterwards
    with pytest.raises(ClockError):
        c.stop()


def test_unfinished_clock_has_no_time():
    c = LegClock()
    c.start()
    with pytest.raises(ClockError, match="running"):
        c.seconds
    c.pause()
    with pytest.raises(ClockError, match="paused"):
        c.seconds


def test_fail_keeps_the_first_reason():
    c = LegClock()
    c.start()
    c.fail("crashed: reader died")
    c.fail("second reason")
    assert c.failure == "crashed: reader died"
    with pytest.raises(ClockError, match="reader died"):
        c.seconds


def test_clock_hooks_map_nova_bf_names():
    c = LegClock(now=FakeNow(1.0, 2.0, 4.0, 5.0))
    h = ClockHooks(c)
    h.corpus_start(); h.pause(); h.resume(); h.done()
    assert c.seconds == pytest.approx(2.0)
    with pytest.raises(ClockError):
        h.corpus_start()
