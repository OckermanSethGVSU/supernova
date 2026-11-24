"""The data steps of an experiment's `prep:` section.

Local steps (prep.stage), run by `bench launch` on this machine into the stage:
  copy              copy the source file
  filtered_cut      cut the source's rows into part files at the given row offsets
  filtered_queries  copy the queries without the given columns

Node steps, emitted as bash lines that the task runs before the job command:
  queries           dense: a bench/prep script splits queries and ground truth
  queries_src       sparse / multivector: copy a queries file to queries.path
  replicate         filtered: copies of the staged files as one corpus
  download          realfile: the real files from S3 (a pinned generated script)
  textscan          filtered: build the staged nova_textscan (part of the build step)

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


REALFILE_TEMPLATE = r'''# parallel ranged GETs: 32 per object, all objects at once (single-part S3 objects stream
# slowly through one connection; ranged parallel GETs do not care how the object was written)
# argv[1]: the s3:// prefix (the shell fills it in from local_paths.env)
import os, sys, boto3
from concurrent.futures import ThreadPoolExecutor
from boto3.s3.transfer import TransferConfig
s3 = boto3.client("s3", region_name="{region}")
cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=32)
B, _, P = sys.argv[1][len("s3://"):].partition("/")
P = P.rstrip("/") + "/"
keys = []
for d in ({dirs}):
    r = s3.list_objects_v2(Bucket=B, Prefix=f"{{P}}{{d}}/")
    keys += sorted(o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".parquet"))[:{per_dir}]
def get(k):
    dst = os.path.expanduser("~/sky_workdir/{dst}/" + os.path.basename(k))
    want = s3.head_object(Bucket=B, Key=k)["ContentLength"]
    if not os.path.exists(dst) or os.path.getsize(dst) != want:
        s3.download_file(B, k, dst + ".part", Config=cfg)
        os.replace(dst + ".part", dst)
    return dst, os.path.getsize(dst), want
with ThreadPoolExecutor(4) as ex:
    for dst, got, want in ex.map(get, keys):
        print(dst, got, "OK" if got == want else f"SIZE MISMATCH want {{want}}")
'''


def download_script(d: dict) -> str:
    """The realfile download heredoc for prep.download. Refused unless
    its sha256 matches the YAML's pin, so any change to the spec or the template is deliberate."""
    text = REALFILE_TEMPLATE.format(region=d["region"],
                                    dirs=", ".join(f'"{x}"' for x in d["dirs"]),
                                    per_dir=d["per_dir"], dst=d["dst"])
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != d["sha256"]:
        raise PrepError(f"prep.download: the generated download script has sha256 {sha}, the "
                        f"experiment pins {d['sha256']} (update the pin if the change is meant)")
    return text


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
    if pr.get("replicate"):   # filtered: build the replicated corpus from copies of the files
        r = pr["replicate"]
        return ['log "timing: build replicated corpus"',
                f'cd {WORKDIR} && rm -rf {r["dst"]} && mkdir -p {r["dst"]}',
                f'for r in $(seq 0 {r["copies"] - 1}); do for f in {r["src"]}; do   '
                '# 32 x 30720 = 983040 rows, 128 files',
                f'  cp $f {r["dst"]}/r${{r}}_$(basename $f); done; done']
    if pr.get("download"):  # realfile: download the real files, with boto3 in a throwaway venv
        d = pr["download"]
        return ['log "real: download 4 files"',
                f'cd {WORKDIR} && mkdir -p {d["dst"]}',
                '{ [ -x ~/vdl/bin/python ] || uv venv -q --python 3.12 ~/vdl; } \\',
                f'  && uv pip install -q --python ~/vdl/bin/python boto3 >> {RES}/boto3.log 2>&1',
                f"~/vdl/bin/python - {_word(d['src'], shlex.quote)} <<'PY'", download_script(d).rstrip("\n"), "PY",
                f'log "real: files $(ls {d["dst"]} | wc -l), $(du -sh {d["dst"]} | cut -f1)"']
    return []


def textscan_lines(exp: dict) -> list[str]:
    """With prep.textscan, the build step's line that installs the staged nova_textscan."""
    if (exp.get("prep") or {}).get("textscan"):
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
    """Copy or cut the data into the stage; returns the lines for STAGED."""
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
        elif d["how"] == "filtered_cut":
            import pyarrow.parquet as pq
            t = pq.read_table(d["source"], columns=list(d["columns"]))
            if t.num_rows != d["rows"]:
                raise PrepError(f"{d['source']} has {t.num_rows} rows, the cut needs {d['rows']}")
            cuts = [0, *d["cuts"], t.num_rows]
            out_dir = os.path.join(stage, os.path.dirname(d["path"]))
            os.makedirs(out_dir, exist_ok=True)
            for i in range(len(cuts) - 1):
                pq.write_table(t.slice(cuts[i], cuts[i + 1] - cuts[i]),
                               os.path.join(out_dir, f"part-{i:05d}.parquet"))
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
