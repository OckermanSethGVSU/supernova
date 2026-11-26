"""No committed file names the private S3 bucket, its prefixes, or a local home directory.

Committed files use the placeholders of bench/paths.py (`${READ}`, `${WRITE}`) and `<HOME>`;
the values live in the untracked experiments/local_paths.env. The files checked are the ones
git would commit: tracked plus untracked-not-ignored under clean_benchmarks/, and the root
.gitignore. The forbidden words are kept here as sha256 digests only, and the values of this
machine's local_paths.env and home directory are checked as well.
"""

import hashlib
import os
import re
import subprocess

import pytest

from bench import paths
from conftest import EXAMPLE_PATHS

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CB)

# sha256 of each forbidden word (the bucket, its two prefixes, the user name in home paths)
FORBIDDEN_SHA256 = {
    "28e333cd326edf6cb1de87240b2a0939f57288c9b3a885b0b7739b9ddb30771a",
    "5b4100a569784a02c1d8fcba7ce9b1b664f20c6448fbb0ea77a114d564c778bf",
    "def452fd0130b8ae735addf4ab457835580bdc3831eff06cedb81342165da215",
    "bf3d7041abe6fda4bf56a412cbc1d0e5b4ceeae6018ea39391a81031a82b5540",
}
WORD = re.compile(rb"[A-Za-z0-9_-]+")


def committed_files():
    """Paths (relative to the repo) git would commit under clean_benchmarks/, plus .gitignore."""
    p = subprocess.run(["git", "-C", REPO, "ls-files", "--cached", "--others", "--exclude-standard",
                        "-z", "--", "clean_benchmarks", ".gitignore"],
                       capture_output=True, timeout=120)
    if p.returncode != 0:
        pytest.skip("not a git checkout")
    out = [f for f in p.stdout.decode().split("\0") if f]
    return [f for f in out if os.path.isfile(os.path.join(REPO, f))]


def local_values():
    """The literal strings of this machine that must not be committed: the values of a real
    local_paths.env or environment, never conftest's example dummies."""
    out = []
    try:
        secs = paths.read_file()
    except paths.PathsError:
        secs = {}
    vals = [v for sec in secs.values() for v in sec.values()]
    vals += [os.environ[e] for e in paths.ENV.values() if os.environ.get(e)]
    for v in vals:
        if v in EXAMPLE_PATHS.values() or not v.startswith("s3://"):
            continue
        # the bucket, and the whole prefix (dataset names inside it may appear on their own)
        bucket, _, prefix = v.rstrip("/")[len("s3://"):].partition("/")
        out += [bucket] + ([f"{bucket}/{prefix}"] if prefix else [])
    home = os.path.expanduser("~").rstrip("/")
    if home.startswith("/home/") and home != "/home/ubuntu":
        out.append(home)
    return sorted({x.encode() for x in out if len(x) >= 6})


def test_no_committed_file_names_a_private_path():
    files = committed_files()
    assert files and "clean_benchmarks/bench/paths.py" in files
    words = local_values()
    bad = []
    for f in files:
        data = open(os.path.join(REPO, f), "rb").read()
        hit = [w.decode() for w in words if w in data]
        if any(hashlib.sha256(w).hexdigest() in FORBIDDEN_SHA256
               for w in set(WORD.findall(data))):
            hit.append("a forbidden word")
        if hit:
            bad.append(f"{f}: {', '.join(hit)}")
    assert not bad, "private paths in committed files:\n" + "\n".join(bad[:50])


def test_the_values_file_is_ignored():
    p = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", "clean_benchmarks/experiments/local_paths.env"])
    assert p.returncode == 0


def test_mask_also_hides_the_schemeless_s3_form(tmp_path):
    """pyarrow / nova-bf name an S3 object `bucket/key`: a record holding that form is masked
    like the s3:// URI (the bucket name never reaches a recorded file)."""
    from bench import paths
    f = tmp_path / "local_paths.env"
    f.write_text("[x]\nREAD = s3://bkt-r/data\nWRITE = s3://bkt-w/out\n")
    old = paths._SECTIONS[:] if hasattr(paths, "_SECTIONS") else None
    try:
        paths.use({"name": "x"})
        rec = {"a": "s3://bkt-w/out/run/f.parquet", "b": "bkt-w/out/run/_bf_manifest.json",
               "c": "Cannot list 'bkt-r/data/part-0.parquet'"}
        got = paths.mask(rec, path=str(f))
        assert got == {"a": "${WRITE}/run/f.parquet", "b": "${WRITE}/run/_bf_manifest.json",
                       "c": "Cannot list '${READ}/part-0.parquet'"}
        assert "bkt" not in str(got)
    finally:
        if old is not None:
            paths._SECTIONS[:] = old
