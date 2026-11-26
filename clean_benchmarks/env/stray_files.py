#!/usr/bin/env python3
"""Move code files a stage does not contain out of the node's workdir.

SkyPilot's workdir upload adds and overwrites files but never deletes, so ~/sky_workdir keeps
whatever was there before. Those files sit beside the staged code, and code_sha.py would hash 
them, so the node's code hash would not be the stage's. Every code file (code_sha.py's rules: 
build trees, caches and data are left alone) that STAGED_files.txt does not list is MOVED to 
DEST, keeping its relative path. Nothing is deleted.

Usage:
    python stray_files.py WORKDIR STAGED_FILES DEST
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_sha  # noqa: E402


def strays(workdir: str, staged: set) -> list:
    """Code files under ``workdir`` (relative) that are not in ``staged``."""
    root = os.path.abspath(workdir)
    return sorted(r for r in (os.path.relpath(p, root) for p in code_sha.code_files([root]))
                  if r not in staged)


def move(workdir: str, rels: list, dest: str) -> None:
    for r in rels:
        dst = os.path.join(dest, r)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(workdir, r), dst)


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 3:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    workdir, listing, dest = a
    with open(listing) as f:
        staged = {x.strip() for x in f if x.strip()}
    found = strays(workdir, staged)
    if not found:
        print("stray code files: none")
        return 0
    move(workdir, found, dest)
    tops = sorted({r.split("/")[0] for r in found})
    print(f"stray code files: {len(found)} not in the stage, moved to {dest} "
          f"(top-level: {', '.join(tops)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
