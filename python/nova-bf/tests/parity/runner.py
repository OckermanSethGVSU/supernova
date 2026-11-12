"""Running nova-bf itself, on a chosen device, and reading its answer back.

Configs are built as plain dicts and validated through the SAME prepass
`load_config` uses (`_normalize_static_date_bounds` then `model_validate`),
not by constructing `BruteForceConfig` directly. That matters for one reason:
a static `range` bound on a declared date field is written as an RFC-3339
string in real YAML and converted to epoch microseconds by that prepass. A
harness that hand-constructed the model would have to pre-convert its own
bounds to µs, which would both be unlike every real config and quietly skip
the conversion step from coverage.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager

import pyarrow.parquet as pq

from nova_bf.compute import run_compute
from nova_bf.config import BruteForceConfig, _normalize_static_date_bounds


def spec(
    name: str,
    *,
    vector_type: str = "dense",
    metric: str = "cosine",
    k: int = 25,
    filter: dict | None = None,
    rows: dict | None = None,
) -> dict:
    """One search, in the dict form a YAML config would carry."""
    out = {"name": name, "vector_type": vector_type, "metric": metric, "k": k}
    if filter is not None:
        out["filter"] = filter
    if rows is not None:
        out["rows"] = rows
    return out


def build_config(ds, specs: list[dict], *, out_tag: str, params: dict | None = None):
    # Deep-copied: the date prepass rewrites `range` bounds IN PLACE, and the
    # spec dicts usually alias `cases.FILTERS`. Without the copy the first run
    # silently turned every shared filter's RFC-3339 bounds into nova-bf's own
    # epoch-µs floats — which the oracles then read back as if it were config.
    specs = copy.deepcopy(specs)
    data = {
        "corpus": {
            "path": ds.corpus_dir,
            "id_column": "id",
            "date_fields": ds.date_fields,
        },
        "queries": {
            "path": ds.queries_path,
            "id_column": "qid",
            # Declared so a `range_from_query` can draw a datetime bound from
            # `q_after`; the config rejects mixing a date corpus field with a
            # non-date query column, so this has to be declared to be usable.
            "date_fields": ds.query_date_fields,
        },
        "output": {"path": f"{ds.tmp}/out_{out_tag}"},
        "params": {"io_workers": 2, **(params or {})},
        "searches": specs,
    }
    return BruteForceConfig.model_validate(_normalize_static_date_bounds(data))


@contextmanager
def env(**overrides: str | None):
    """Set (or, with `None`, unset) environment variables for the block.

    The kernel switches nova-bf exposes to operators are environment variables
    — `NOVA_BF_NO_TOPK_KERNEL`, `NOVA_BF_NO_FOLD_KERNEL` — so an A/B between
    the fused and portable paths is an env A/B, and it has to restore whatever
    was there before or one test's setting leaks into the next.
    """
    previous = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def pinned_device(device: str | None):
    """Pin `run_compute` to `device` for the duration of the block.

    nova-bf auto-selects CUDA when it sees one, which is right for production
    and useless for a parity harness: on a GPU box there would otherwise be no
    way to ask for the CPU answer and compare the two. `None` restores
    auto-selection.
    """
    previous = os.environ.get("NOVA_BF_DEVICE")
    if device is None:
        os.environ.pop("NOVA_BF_DEVICE", None)
    else:
        os.environ["NOVA_BF_DEVICE"] = device
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NOVA_BF_DEVICE", None)
        else:
            os.environ["NOVA_BF_DEVICE"] = previous


def qid_index(ds) -> dict[str, int] | None:
    """`{query id string: query index}` for a dataset whose queries carry a
    `qid` (the shared corpus does); `None` for one whose queries do not, which
    `read_results` then reads as id == index."""
    if not all("qid" in q for q in ds.queries):
        return None
    out = {q["qid"]: i for i, q in enumerate(ds.queries)}
    assert len(out) == len(ds.queries), "query ids must be unique"
    return out


def read_results(paths: dict[str, str], qids: dict[str, int] | None = None
                 ) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """`{search_name: {query_index: [(corpus_row, score), …]}}`, hits left in
    the rank order nova-bf wrote them (they are already sorted best-first —
    keeping the order is what lets a caller check ranking, not just set
    membership).

    `qids` maps each written `query_id` back to its query index. Without it
    the id is read as the index itself (`int(qid)`), which is only right for a
    fixture whose ids ARE `str(index)`. With it, an id nova-bf wrote that is
    not one of the queries' ids — the position instead of the id, say — is a
    hard failure, as is the same query written twice."""
    out = {}
    for name, path in paths.items():
        t = pq.read_table(path).to_pydict()
        per_q = {}
        for qid, ids, scores in zip(t["query_id"], t["hit_ids"], t["hit_scores"]):
            if qids is None:
                qi = int(qid)
            else:
                assert qid in qids, (
                    f"{name}: nova-bf wrote query_id {qid!r}, which is not the id of "
                    f"any query (ids look like {sorted(qids)[:3]})")
                qi = qids[qid]
            assert qi not in per_q, f"{name}: query {qid!r} written twice"
            per_q[qi] = [(int(i), float(s)) for i, s in zip(ids, scores)]
        out[name] = per_q
    return out


def run(ds, specs: list[dict], *, out_tag: str, device: str | None = None,
        params: dict | None = None):
    """Run one nova-bf config on `device` and return its hits per search."""
    cfg = build_config(ds, specs, out_tag=f"{out_tag}_{device or 'auto'}", params=params)
    with pinned_device(device):
        paths = run_compute(cfg)
    return read_results(paths, qid_index(ds))


# ------------------------------------------------------------------ probes
#
# A parity test that never reaches the code path it is named after passes for
# free — and the paths that matter most here (a compacted batch, a budget
# fallback, a windowed gather) are exactly the ones a small fixture quietly
# routes around. `probe()` records, for one block, which nova-bf SOURCE LINES
# ran (by snippet, so a refactor that moves a line does not break the probe,
# and a refactor that deletes it fails loudly instead of passing vacuously)
# and how often chosen module functions were called. It works on whatever
# copy of `nova_bf` is imported — including a mutated scratch copy.

_PROBE_TOOL = 4  # sys.monitoring tool id (coverage.py uses 1, profilers 2/5)


def _locate(module, snippet: str) -> tuple[str, int]:
    """`(filename, line)` of the ONE line of `module`'s source containing
    `snippet`. Ambiguous or missing snippets are errors: a probe that
    silently watched the wrong line would be worse than none."""
    path = module.__file__
    with open(path) as fh:
        hits = [i + 1 for i, line in enumerate(fh) if snippet in line]
    if len(hits) != 1:
        raise LookupError(f"probe snippet {snippet!r} occurs {len(hits)} times in {path}")
    return path, hits[0]


class Probe:
    def __init__(self):
        self.lines: dict[str, bool] = {}
        self.calls: dict[str, int] = {}
        self.notes: dict[str, int] = {}
        self.combo: dict = {}

    def note(self, key: str, n: int = 1) -> None:
        self.notes[key] = self.notes.get(key, 0) + n

    def hit(self, name: str) -> bool:
        return self.lines[name]


@contextmanager
def probe(*, lines: dict | None = None, calls: dict | None = None):
    """`lines`: `{name: (module, snippet)}` — was that line executed?
    `calls`: `{name: (module, attr, hook)}` — wrap `module.attr`, count its
    calls, and call `hook(probe, result, *args, **kwargs)` (or None) after
    each. Also snapshots `filters.combo_usage()` for the block."""
    import sys

    from nova_bf import filters as filters_mod

    p = Probe()
    targets: dict[tuple[str, int], str] = {}
    for name, (module, snippet) in (lines or {}).items():
        targets[_locate(module, snippet)] = name
        p.lines[name] = False
    mon = sys.monitoring
    files = {f for f, _ in targets}

    def on_line(code, line):
        if code.co_filename in files:
            name = targets.get((code.co_filename, line))
            if name is not None:
                p.lines[name] = True
        return mon.DISABLE

    saved = []
    for name, (module, attr, hook) in (calls or {}).items():
        real = getattr(module, attr)
        p.calls[name] = 0

        def wrapper(*a, _real=real, _name=name, _hook=hook, **k):
            p.calls[_name] += 1
            out = _real(*a, **k)
            if _hook is not None:
                _hook(p, out, *a, **k)
            return out

        saved.append((module, attr, real))
        setattr(module, attr, wrapper)
    if targets:
        mon.use_tool_id(_PROBE_TOOL, "nova-bf-parity-probe")
        mon.register_callback(_PROBE_TOOL, mon.events.LINE, on_line)
        mon.set_events(_PROBE_TOOL, mon.events.LINE)
        mon.restart_events()
    filters_mod.reset_combo_usage()
    try:
        yield p
    finally:
        p.combo = filters_mod.combo_usage()
        if targets:
            mon.set_events(_PROBE_TOOL, 0)
            mon.register_callback(_PROBE_TOOL, mon.events.LINE, None)
            mon.free_tool_id(_PROBE_TOOL)
        for module, attr, real in reversed(saved):
            setattr(module, attr, real)
