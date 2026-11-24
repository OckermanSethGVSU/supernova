#!/usr/bin/env python3
"""Compute a reproducible SHA256 hash of the benchmark code.

Hashes code and configuration files under the given roots, including
uncommitted changes. Excludes data, build outputs, caches, and environments.
Files are hashed by content and relative path, so identical code trees
produce the same hash even without Git metadata.

--json records individual file hashes and Git status for comparison between runs.

Usage:
    python code_sha.py ROOT [ROOT ...] [--base DIR] [--json OUT]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

CODE_SUFFIXES = {".py", ".pyi", ".sh", ".bash", ".yaml", ".yml", ".toml", ".cfg", ".ini",
                 ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".rs", ".diff", ".patch",
                 ".env", ".lock", ".in"}
CODE_NAMES = {"Makefile", "CMakeLists.txt", "Cargo.lock", "MANIFEST.in"}
# never code: caches, build trees, envs, VCS, and the data / output dirs of a stage
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", ".venv",
                "venv", "node_modules", "target", "build", "dist", "wheels", "data", "out",
                "res", ".ipynb_checkpoints"}
# never code: the private path values (bench/paths.py hashes them on their own)
EXCLUDE_NAMES = {"local_paths.env"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_code(name):
    if name in EXCLUDE_NAMES:
        return False
    return name in CODE_NAMES or os.path.splitext(name)[1] in CODE_SUFFIXES


def code_files(roots):
    """Real paths of every code file under ``roots`` (sorted, no duplicates)."""
    out = set()
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):   # a file root counts only if it is code too (not data)
            if is_code(os.path.basename(root)):
                out.add(root)
            continue
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS
                             and not x.endswith(".egg-info"))
            out.update(os.path.join(d, f) for f in files
                       if is_code(f) and os.path.isfile(os.path.join(d, f)))
    return sorted(out)


def tree_sha(roots, base):
    """(hash, [(relpath, sha256)])."""
    rows = []
    for p in code_files(roots):
        rows.append((os.path.relpath(p, base), sha256_file(p)))
    rows.sort()
    text = "".join(f"{h}  {rel}\n" for rel, h in rows)
    return hashlib.sha256(text.encode()).hexdigest(), rows


def _git(root, *args):
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip())
    return r.stdout


def git_info(root):
    """{toplevel, head, dirty, untracked_files, dirty_sha256} for a root inside a git
    checkout, else None. dirty_sha256 hashes ``git diff HEAD --binary -- root`` plus each
    untracked, not-ignored file under root (path and content); "" when clean."""
    d = root if os.path.isdir(root) else os.path.dirname(root)
    try:
        top = _git(d, "rev-parse", "--show-toplevel").decode().strip()
        head = _git(d, "rev-parse", "HEAD").decode().strip()
        diff = _git(top, "diff", "HEAD", "--binary", "--", os.path.abspath(root))
        untracked = _git(top, "ls-files", "--others", "--exclude-standard", "-z", "--",
                         os.path.abspath(root)).split(b"\0")
    except Exception:  # noqa: BLE001 - not a checkout, or no git
        return None
    h = hashlib.sha256(diff)
    n_untracked = 0
    for rel in sorted(u for u in untracked if u):
        p = os.path.join(top, rel.decode())
        if os.path.isfile(p):
            n_untracked += 1
            h.update(b"\0untracked\0" + rel + b"\0" + sha256_file(p).encode())
    dirty = bool(diff) or n_untracked > 0
    return {"toplevel": top, "head": head, "dirty": dirty,
            "untracked_files": n_untracked, "dirty_sha256": h.hexdigest() if dirty else ""}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--base", default=None,
                    help="paths in the hash are relative to this (default: the roots' common dir)")
    ap.add_argument("--json", default=None, help="write the record here")
    a = ap.parse_args(argv)
    missing = [r for r in a.roots if not os.path.exists(r)]
    if missing:
        print(f"code_sha.py: no such path: {' '.join(missing)}", file=sys.stderr)
        return 2
    roots = [os.path.abspath(r) for r in a.roots]
    base = os.path.abspath(a.base) if a.base else os.path.commonpath(
        [r if os.path.isdir(r) else os.path.dirname(r) for r in roots])
    sha, rows = tree_sha(roots, base)
    if a.json:
        rec = {"code_sha": sha, "base": base, "roots": [os.path.relpath(r, base) for r in roots],
               "n_files": len(rows), "git": {os.path.relpath(r, base): git_info(r) for r in roots},
               "files": [{"path": rel, "sha256": h} for rel, h in rows]}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
    print(sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
