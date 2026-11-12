"""Proving a code path RAN, not just that its answer was right.

Most GPU fast paths in nova-bf fail OPEN: when a kernel declines (no Triton,
an int32-overflowing tile, a shape it does not handle, not enough free VRAM),
the run quietly takes the portable path and returns the same ground truth. A
GPU test that only compares answers therefore passes whether or not the path
it is named after executed — so every test in `test_parity_gpu_paths.py`
records which nova_bf source lines executed during its run and REQUIRES the
ones that only that path reaches.

Lines are named by a SNIPPET of their source text, not a line number, so the
requirements survive edits above them; `require` fails if a snippet matches
no line (it was renamed — update the requirement) as well as when the line
did not run.

Mechanism: `sys.monitoring` (Python >= 3.12) LINE events, filtered to files
under `nova_bf/`, each location reported once per recording (DISABLE after the
first hit, `restart_events` at the start of the next recording). Events are
process-wide, so lines executed on nova-bf's reader and prefetch threads are
seen too.

`NOVA_BF_PARITY_REACH_OUT=/path/reach.json` additionally writes every
recording (test -> required snippets -> hit?, plus all executed lines) when
the test session ends, so a GPU run can be audited afterwards:

    NOVA_BF_PARITY_REACH_OUT=/tmp/reach.json pytest tests/parity/test_parity_gpu_paths.py
    python -m tests.parity.reach /tmp/reach.json      # from python/nova-bf/
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

_NAME = "nova_bf_parity_reach"
_REPORT: dict = {}


def _source_dir() -> Path:
    import nova_bf

    return Path(nova_bf.__file__).resolve().parent


def lines_matching(module_file: str, snippet: str) -> list[int]:
    """1-based line numbers of `nova_bf/<module_file>` containing `snippet`."""
    text = (_source_dir() / module_file).read_text().splitlines()
    return [i + 1 for i, line in enumerate(text) if snippet in line]


class LineRecorder:
    """`with LineRecorder("label") as rec: ...; rec.require("compute.py", "...")`"""

    def __init__(self, label: str):
        self.label = label
        self.hits: set[tuple[str, int]] = set()
        self.checks: list[dict] = []
        self._tid = None

    def __enter__(self):
        mon = sys.monitoring
        for tid in (4, 3, 2, 5, 1, 0):
            if mon.get_tool(tid) is None:
                mon.use_tool_id(tid, _NAME)
                self._tid = tid
                break
        else:  # pragma: no cover - every tool slot taken
            raise RuntimeError("no free sys.monitoring tool id for the reach recorder")
        src = str(_source_dir())

        def on_line(code, line):
            fn = code.co_filename
            if fn.startswith(src):
                self.hits.add((os.path.basename(fn), line))
            return mon.DISABLE

        mon.register_callback(self._tid, mon.events.LINE, on_line)
        mon.set_events(self._tid, mon.events.LINE)
        mon.restart_events()
        return self

    def __exit__(self, *exc):
        mon = sys.monitoring
        mon.set_events(self._tid, 0)
        mon.register_callback(self._tid, mon.events.LINE, None)
        mon.free_tool_id(self._tid)
        _REPORT[self.label] = {
            "checks": self.checks,
            "hits": sorted(f"{f}:{n}" for f, n in self.hits),
        }
        return False

    def hit(self, module_file: str, snippet: str) -> bool:
        nums = lines_matching(module_file, snippet)
        assert nums, (f"reach: snippet {snippet!r} matches no line of {module_file} — "
                      f"the code moved or was renamed; update the requirement")
        return any((module_file, n) in self.hits for n in nums)

    def require(self, module_file: str, snippet: str, why: str = ""):
        ok = self.hit(module_file, snippet)
        self.checks.append({"file": module_file, "snippet": snippet, "hit": ok, "why": why})
        assert ok, (f"{self.label}: the path under test did NOT run — no line of "
                    f"{module_file} containing {snippet!r} executed. {why}")

    def forbid(self, module_file: str, snippet: str, why: str = ""):
        ok = self.hit(module_file, snippet)
        self.checks.append({"file": module_file, "snippet": snippet, "hit": ok,
                            "why": why, "forbidden": True})
        assert not ok, f"{self.label}: {module_file} {snippet!r} ran but must not. {why}"


def _dump():
    out = os.environ.get("NOVA_BF_PARITY_REACH_OUT")
    if out and _REPORT:
        Path(out).write_text(json.dumps(_REPORT, indent=1))


atexit.register(_dump)


def summarize(path: str) -> int:
    """Print one line per (test, requirement); exit status 1 if any failed."""
    data = json.loads(Path(path).read_text())
    bad = 0
    for label, rec in sorted(data.items()):
        for c in rec["checks"]:
            want = not c.get("forbidden")
            ok = c["hit"] == want
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {label}: {c['file']} "
                  f"{'must not hit' if not want else 'hit'} {c['snippet']!r}")
        print(f"     {label}: {len(rec['hits'])} nova_bf lines executed")
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(summarize(sys.argv[1]))
