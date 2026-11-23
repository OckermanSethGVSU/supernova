"""`run_compute(timing_hooks=...)`: where the four hooks fire, and that they only observe.

The hooks (nova_bf/timing_hooks.py) let a benchmark clock time the search from just
before the scan's corpus readers start (`corpus_start`) to every search's sorted top-K
on the host (`done`), with `pause`/`resume` around each search's result decode + write.
These tests pin, for every scan shape (dense, sparse, multivector, filtered, several
searches in one pass, `tiebreak='id'`, two-pass on the CPU):

  * the order: corpus_start, (pause, resume) x n_searches, done -- never per file;
  * no corpus DATA read before corpus_start (Store's read entry points are recorded;
    the only pre-scan reads allowed are footer/schema reads and the multivector
    token sample, see `PRE_SCAN_OK`);
  * every search's top-K is sorted and on the host by its pause, and all are by done;
  * hooks on vs off: byte-identical result parquets;
  * failures: an exception leaves the sequence visibly incomplete (no done), and a
    hook that raises fails the run instead of being swallowed.
"""

from __future__ import annotations

import hashlib
import sys
import time

import numpy as np
import pytest

pytest.importorskip("torch")

from nova_bf import compute as compute_mod  # noqa: E402
from nova_bf import twopass  # noqa: E402
from nova_bf.compute import run_compute  # noqa: E402
from nova_bf.io import Store  # noqa: E402
from nova_bf.timing_hooks import NO_HOOKS, NoHooks  # noqa: E402
from parity import corpus as corpus_mod  # noqa: E402
from parity import runner  # noqa: E402

FILTER_TEXT = {"must": [{"field": "title", "match_text_from_query": "q_phrase"}]}
FILTER_MATCH = {"must": [{"field": "language", "match": "eng"}]}

# name -> (specs, params, env)
SHAPES = {
    "dense": ([runner.spec("cos", metric="cosine", k=7)], {}, {}),
    "dense_multi": ([runner.spec("cos", metric="cosine", k=7),
                     runner.spec("dot", metric="dot", k=5),
                     runner.spec("euc", metric="euclidean", k=3)],
                    {"dense_batch_size": 32}, {}),
    "sparse": ([runner.spec("sp", vector_type="sparse", metric="dot", k=6)], {}, {}),
    "multivector": ([runner.spec("mv", vector_type="multivector", metric="cosine", k=4)], {}, {}),
    "filtered": ([runner.spec("txt", metric="cosine", k=5, filter=FILTER_TEXT),
                  runner.spec("eng", metric="dot", k=5, filter=FILTER_MATCH)], {}, {}),
    "all_modalities": ([runner.spec("d", metric="dot", k=5),
                        runner.spec("s", vector_type="sparse", metric="dot", k=5),
                        runner.spec("m", vector_type="multivector", metric="dot", k=5,
                                    filter=FILTER_MATCH)], {}, {}),
    "tiebreak_id": ([runner.spec("cos", metric="cosine", k=7)], {"tiebreak": "id"}, {}),
}

# Store entry points that read corpus bytes. `read_schema` reads a footer only.
READ_METHODS = ("read_columns", "read_columns_fetched", "read_dense_versioned", "read_dense",
                "read_schema", "_ranged_download")
# Reads allowed BEFORE corpus_start (run_compute's pre-scan setup; D-S3-02 / D-S3-12 in
# clean_benchmarks/DECISIONS.md): footer / schema reads only. The multivector tile-size
# sample is NOT allowed: it is timed (corpus_start / pause around it).
PRE_SCAN_OK = {("read_schema", None)}


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return corpus_mod.build(tmp_path_factory.mktemp("bf_hooks"))


def _rc_locals() -> dict:
    """run_compute's local variables (the hook's caller, a few frames up)."""
    f = sys._getframe(1)
    while f is not None and f.f_code.co_name != "run_compute":
        f = f.f_back
    assert f is not None, "hook not called from run_compute"
    return f.f_locals


class Recorder:
    """Hooks that record each call, plus checks made from inside nova-bf's frame."""

    def __init__(self, fail_on: str | None = None):
        self.events: list[tuple[str, float]] = []
        self.fail_on = fail_on
        self.host_checks: list[str] = []
        self.sample_pauses = 0

    def _ev(self, name):
        self.events.append((name, time.perf_counter()))
        if name == self.fail_on:
            raise RuntimeError(f"hook {name} failed on purpose")

    @property
    def names(self):
        return [n for n, _ in self.events]

    @property
    def state(self):
        n = self.names
        return "done" if n and n[-1] == "done" else ("started" if n else "new")

    def corpus_start(self):
        self._ev("corpus_start")

    def pause(self):
        f = _rc_locals()
        if "sort_t0" not in f:
            # the multivector token sample's pause, before the scan (D-S3-12)
            assert "multivector" in f["vts_needed"] and f["hooks_started"] is False
            self.sample_pauses += 1
            self._ev("pause")
            return
        # The search just sorted: its top-K is on the host and its device state released.
        i = f["i"]
        assert f["spec_top_key"][i] is None and f["spec_top_enc"][i] is None
        sc, enc = f["sc"], f["enc"]
        assert isinstance(sc, np.ndarray) and isinstance(enc, np.ndarray)
        assert sc.shape == enc.shape and sc.shape[1] == f["s"].k
        # Sorted best-first: the packed keys sort descending, and the score is
        # monotone in the key, so each row's finite scores are non-increasing
        # (cosine/dot) or non-decreasing (euclidean distance).
        for row in sc:
            fin = row[np.isfinite(row)]
            d = np.diff(fin)
            assert (d <= 0).all() or (d >= 0).all(), row
        self.host_checks.append(f["s"].name)
        self._ev("pause")

    def resume(self):
        self._ev("resume")

    def done(self):
        f = _rc_locals()
        assert all(x is None for x in f["spec_top_key"]), "a search's top-K is still on the device"
        assert len(f["manifest_searches"]) == len(f["specs"])
        self._ev("done")


@pytest.fixture
def reads(monkeypatch):
    """Records every Store read as (method, caller function, path, hook phase)."""
    log: list[dict] = []
    phase = {"v": "pre"}

    def wrap(name):
        orig = getattr(Store, name)

        def w(self, *a, **kw):
            caller = sys._getframe(1).f_code.co_name
            path = a[0] if a else kw.get("read_path")
            log.append({"method": name, "caller": caller, "path": path, "phase": phase["v"]})
            return orig(self, *a, **kw)
        return w

    for name in READ_METHODS:
        if hasattr(Store, name):
            monkeypatch.setattr(Store, name, wrap(name))
    return log, phase


class PhasedRecorder(Recorder):
    def __init__(self, phase, **kw):
        super().__init__(**kw)
        self.phase = phase

    def corpus_start(self):
        super().corpus_start()
        self.phase["v"] = "timed"

    def pause(self):
        super().pause()
        self.phase["v"] = "paused"

    def resume(self):
        super().resume()
        self.phase["v"] = "timed"

    def done(self):
        super().done()
        self.phase["v"] = "after"


def _setup(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def _sha_outputs(paths: dict[str, str]) -> dict[str, str]:
    return {n: hashlib.sha256(open(p, "rb").read()).hexdigest() for n, p in paths.items()}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_hook_order_and_boundary(ds, shape, monkeypatch, reads):
    specs, params, env = SHAPES[shape]
    _setup(monkeypatch, env)
    log, phase = reads
    cfg = runner.build_config(ds, specs, out_tag=f"hooks_{shape}", params=params)
    rec = PhasedRecorder(phase)
    with runner.pinned_device("cpu"):
        run_compute(cfg, timing_hooks=rec)

    # order: once, (the multivector sample), per search, once -- independent of the 4 files
    has_mv = any(s.get("vector_type") == "multivector" for s in specs)
    assert rec.sample_pauses == int(has_mv)
    assert rec.names == ["corpus_start"] + ["pause", "resume"] * (len(specs) + has_mv) + ["done"]
    assert rec.host_checks == [s["name"] for s in specs]
    t = [ts for _, ts in rec.events]
    assert t == sorted(t)

    corpus_dir = ds.corpus_dir
    mine = [r for r in log if str(r["path"]).startswith(corpus_dir)]
    pre = [r for r in mine if r["phase"] == "pre"]
    bad = [r for r in pre
           if (r["method"], None) not in PRE_SCAN_OK and (r["method"], r["caller"]) not in PRE_SCAN_OK]
    assert not bad, f"corpus data read before corpus_start: {bad}"
    # the scan itself read the data after corpus_start: one data read per file at least
    timed_data = {r["path"] for r in mine if r["phase"] == "timed" and r["method"] != "read_schema"}
    assert len(timed_data) == len(corpus_mod.FILE_SIZES), timed_data
    # nothing reads corpus data after done
    assert not [r for r in mine if r["phase"] == "after" and r["method"] != "read_schema"]
    sample = [r for r in mine if r["caller"] == "_sample_mean_doc_tokens"]
    if has_mv:
        # the tile-size sample reads file 0 INSIDE the clock (D-S3-12)
        assert len(sample) == 1 and sample[0]["phase"] == "timed", sample
    else:
        assert not sample


@pytest.mark.parametrize("shape", ["dense_multi", "sparse", "multivector", "filtered"])
def test_hooks_on_and_off_write_identical_results(ds, shape, monkeypatch):
    specs, params, env = SHAPES[shape]
    _setup(monkeypatch, env)
    cfg = runner.build_config(ds, specs, out_tag=f"onoff_{shape}", params=params)
    with runner.pinned_device("cpu"):
        off = _sha_outputs(run_compute(cfg))
        on = _sha_outputs(run_compute(cfg, timing_hooks=Recorder()))
        default = _sha_outputs(run_compute(cfg, timing_hooks=NO_HOOKS))
    assert off == on == default


def test_nohooks_is_a_no_op():
    h = NoHooks()
    for name in ("corpus_start", "pause", "resume", "done"):
        assert getattr(h, name)() is None
    assert isinstance(NO_HOOKS, NoHooks)


def test_reader_failure_leaves_no_done(ds, monkeypatch):
    """A corpus read that fails mid-scan raises out of run_compute after corpus_start;
    `done` never fires, so a clock cannot report a time for the crashed run."""
    specs, params, _ = SHAPES["dense"]
    cfg = runner.build_config(ds, specs, out_tag="crash_read", params=params)
    orig = Store.read_dense_versioned
    seen = []

    def failing(self, read_path, column):
        seen.append(read_path)
        if len(seen) == 2:
            raise OSError("injected read failure")
        return orig(self, read_path, column)

    monkeypatch.setattr(Store, "read_dense_versioned", failing)
    rec = Recorder()
    with runner.pinned_device("cpu"), pytest.raises(RuntimeError, match="injected read failure"):
        run_compute(cfg, timing_hooks=rec)
    assert rec.names == ["corpus_start"]


def test_decode_failure_leaves_no_done(ds, monkeypatch):
    """An exception between pause and resume (result decode/write) ends the sequence
    on `pause`: still no `done`."""
    specs, params, _ = SHAPES["dense_multi"]
    cfg = runner.build_config(ds, specs, out_tag="crash_decode", params=params)

    def boom(*a, **kw):
        raise ValueError("injected decode failure")

    monkeypatch.setattr(compute_mod, "build_result_table", boom)
    rec = Recorder()
    with runner.pinned_device("cpu"), pytest.raises(ValueError, match="injected decode failure"):
        run_compute(cfg, timing_hooks=rec)
    assert rec.names == ["corpus_start", "pause"]


@pytest.mark.parametrize("hook", ["corpus_start", "pause", "resume", "done"])
def test_a_failing_hook_fails_the_run(ds, hook):
    """A hook that raises is never swallowed: run_compute raises it."""
    specs, params, _ = SHAPES["dense"]
    cfg = runner.build_config(ds, specs, out_tag=f"hookfail_{hook}", params=params)
    with runner.pinned_device("cpu"), pytest.raises(RuntimeError, match=f"hook {hook} failed"):
        run_compute(cfg, timing_hooks=Recorder(fail_on=hook))


def test_default_is_unchanged_signature():
    import inspect

    p = inspect.signature(run_compute).parameters
    assert list(p)[-1] == "timing_hooks" and p["timing_hooks"].default is None


def test_two_pass_on_cpu(tmp_path, monkeypatch, reads):
    """The two-pass dense path (inside the scan) with the hooks on: same order, same
    boundary, same bytes as with the hooks off. Fixture and knobs are
    tests/test_per_file_progress_line.py's, the ones that open the two-pass gate at
    fixture scale on the CPU (none changes a result)."""
    from test_per_file_progress_line import _cfg

    monkeypatch.setenv("NOVA_BF_TWOPASS_RESCORE", "slice")
    monkeypatch.setenv("NOVA_BF_TWOPASS_ON_CPU", "1")
    monkeypatch.setenv("NOVA_BF_TWOPASS_THRESHOLD", "1.0")
    monkeypatch.setattr(twopass, "PAD_QUANTUM", 2)
    monkeypatch.setattr(twopass, "PAD_FLOOR", 2)
    monkeypatch.setattr(twopass, "PAD_SMALL", ())
    monkeypatch.setattr(twopass, "MIN_QUERY_ROWS", 4)
    log, phase = reads
    cfg = _cfg(tmp_path, 5)
    off = _sha_outputs(run_compute(cfg))
    assert twopass.stats()["slices_twopass"] > 0, "two-pass did not engage"
    log.clear()
    rec = PhasedRecorder(phase)
    on = _sha_outputs(run_compute(cfg, timing_hooks=rec))
    assert twopass.stats()["slices_twopass"] > 0, "two-pass did not engage"
    assert off == on
    assert rec.names == ["corpus_start", "pause", "resume", "done"]
    corpus = str(tmp_path / "corpus")
    pre = [r for r in log if str(r["path"]).startswith(corpus) and r["phase"] == "pre"]
    assert all(r["method"] == "read_schema" for r in pre), pre
