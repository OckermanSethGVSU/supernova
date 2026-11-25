"""The data steps of an experiment's `prep:` section.

Local steps (prep.stage), run by `bench launch` on this machine into the stage:
  copy              copy the source file
  filtered_queries  copy the queries without the given columns

Node steps, emitted as bash lines that the task runs before the job command:
  queries           dense: a bench/prep script splits queries and ground truth
  queries_src       copy a queries file to queries.path
  textscan          filtered: build the staged nova_textscan, always (part of the build step)

Every value comes from the YAML; this file keeps only the shape of each step.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil

from bench import paths

CB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(CB)
WORKDIR = "~/sky_workdir"   # SkyPilot's workdir on the node
RES = "~/res"


class PrepError(RuntimeError):
    """A prep step refused (`bench launch` prints it and exits 2)."""
    pass


def _word(s: str, quote=lambda x: x) -> str:
    """A value as a shell word: a private-path placeholder is left for the node's shell to
    fill in (the run block reads local_paths.env first); anything else as before."""
    return paths.shell_word(s) if paths.refs(s) else quote(s)


def prep_source(p: str) -> str:
    """A prep.stage source: absolute, or relative to the repo root."""
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO, p))


# ---------------------------------------------------------------------------------------------
# node steps
# ---------------------------------------------------------------------------------------------
def node_prep(exp: dict) -> list[str]:
    """The experiment's node-side data step as bash lines, from its `prep:` section. Runs before
    the launcher, with the paper env activated."""
    pr = exp.get("prep") or {}
    act = f'( . {WORKDIR}/env/activate.sh && '
    if pr.get("queries"):   # dense: a bench/prep script splits queries and ground truth
        q = pr["queries"]
        return [f'log "STEP queries"',
                f'{act}python {WORKDIR}/{q["script"]} {" ".join(_word(x) for x in q["args"])} ) '
                f'2>&1 | tee -a {RES}/progress.log']
    if exp["modality"] in ("sparse", "multivector") or pr.get("queries_src"):   # copy prep.queries_src to queries.path
        q = exp["queries"]["path"]
        src = pr.get("queries_src")
        if not src:
            return [f'log "queries: no QSRC, using {q} as is"']
        return [f'log "STEP queries"',
                f'{act}python -c "import sys,pyarrow.fs as f; fs,p=f.FileSystem.from_uri(sys.argv[1]); '
                f'open(sys.argv[2],\'wb\').write(fs.open_input_stream(p).read())" '
                f'{_word(src, shlex.quote)} {q} ) && log "queries: {src} -> {q}"']
    return []


def textscan_lines(exp: dict) -> list[str]:
    """A filtered experiment's build-step line that installs the staged nova_textscan, nova-bf's
    native text matcher (always: without it nova-bf would fall back to its slower Python path)."""
    if exp["modality"] == "filtered":
        return [f'( . $W/env/activate.sh && python -m pip install -q --no-deps --force-reinstall '
                f'./nova_textscan ) >> {RES}/textscan.log 2>&1 || {{ log "FAIL textscan (textscan.log)"; exit 1; }}']
    return []


# ---------------------------------------------------------------------------------------------
# local steps
# ---------------------------------------------------------------------------------------------
def data_plan(exp: dict, srcs: dict) -> list[dict]:
    """prep.stage with each source resolved (`--data-src KEY=PATH` overrides one)."""
    out = []
    for d in (exp.get("prep") or {}).get("stage") or []:
        out.append({**d, "source": srcs.get(d["key"]) or prep_source(d["source"])})
    return out


def stage_data(stage: str, plan: list[dict]) -> list[str]:
    """Copy the data into the stage; returns the lines for STAGED."""
    lines = []
    for d in plan:
        if not os.path.isfile(d["source"]):
            raise PrepError(f"data source {d['source']} not found ({d['key']}; --data-src {d['key']}=PATH)")
        raw = open(d["source"], "rb").read()
        md5 = hashlib.md5(raw).hexdigest()[:12]
        sha = hashlib.sha256(raw).hexdigest()   # the full source hash, in STAGED
        del raw
        if d["how"] == "copy":
            p = os.path.join(stage, d["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy2(d["source"], p)
        elif d["how"] == "filtered_queries":
            import pyarrow.parquet as pq
            q = pq.read_table(d["source"])
            drop = [c for c in d["drop_columns"] if c in q.column_names]
            if drop:
                q = q.drop_columns(drop)
            os.makedirs(os.path.dirname(os.path.join(stage, d["path"])), exist_ok=True)
            pq.write_table(q, os.path.join(stage, d["path"]))
        lines.append(f"data {d['path']}: {d['how']} of {d['source']} md5 {md5} sha256 {sha}")
    return lines
