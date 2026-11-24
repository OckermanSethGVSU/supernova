"""No committed file points at a local-only file.

The decision log, the working plan, the reference recordings and the tests that read them stay
local (git-ignored), so a committed doc, comment or message that names them points nowhere for a
reader of the public repo. Each such reference is written as the reason itself instead. The files
checked are the ones git would commit: tracked plus untracked-not-ignored under clean_benchmarks/,
and the root .gitignore (test_no_private_paths.committed_files).
"""

import os
import re

from test_no_private_paths import REPO, committed_files

# the words that name a local-only file or the tool these benchmarks replaced
LOCAL_ONLY = re.compile("|".join([
    r"DECISIONS\.md", r"\bD-S\d+-", r"tests/reference", r"tests/equivalence",
    r"old_scripts_snapshot", r"\.review_scratch", r"\bPLAN\.md", r"benchmarks/streaming",
]).encode())

SELF = "clean_benchmarks/tests/test_no_local_pointers.py"

# (file, exact line) pairs that may match: functional lines, not doc pointers
ALLOWED = {
    # the ignore rules that keep the local-only files out of git
    (".gitignore", "/clean_benchmarks/DECISIONS.md"),
    (".gitignore", "/clean_benchmarks/PLAN.md"),
    (".gitignore", "/clean_benchmarks/tests/reference/"),
    (".gitignore", "/clean_benchmarks/tests/equivalence/"),
    ("clean_benchmarks/.gitignore", "/.review_scratch/"),
    # pytest must not collect the local reference dir when it exists
    ("clean_benchmarks/pyproject.toml", 'norecursedirs = ["tests/reference"]'),
}


def test_no_committed_file_points_at_a_local_only_file():
    files = committed_files()
    assert files and "clean_benchmarks/bench/leg.py" in files
    bad = []
    for f in files:
        if f == SELF:   # this guard names the words it looks for
            continue
        data = open(os.path.join(REPO, f), "rb").read()
        if b"\0" in data[:4096]:   # binary (parquet fixtures)
            continue
        for n, line in enumerate(data.splitlines(), 1):
            if LOCAL_ONLY.search(line) and (f, line.decode(errors="replace").strip()) not in ALLOWED:
                bad.append(f"{f}:{n}: {line.decode(errors='replace').strip()[:120]}")
    assert not bad, "committed files point at local-only files:\n" + "\n".join(bad[:50])


def test_the_local_only_files_are_ignored():
    import subprocess
    for rel in ("clean_benchmarks/DECISIONS.md", "clean_benchmarks/PLAN.md",
                "clean_benchmarks/tests/reference/x", "clean_benchmarks/tests/equivalence/x",
                "clean_benchmarks/s8/x"):
        p = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", "--no-index", rel])
        assert p.returncode == 0, rel
