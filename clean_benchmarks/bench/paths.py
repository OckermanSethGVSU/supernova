"""The private paths: where each experiment reads its data and writes nova-bf's results.

Committed files never say where data lives. An experiment names its locations only as
`${READ}` (the dataset's S3 prefix, e.g. `${READ}/data/`) and `${WRITE}` (the S3 prefix nova-bf
writes under). The values live in clean_benchmarks/experiments/local_paths.env (untracked;
local_paths.env.example shows the format), one section per experiment name:

    [dense_coyo]
    READ = s3://<bucket>/<prefix>/Coyo-VE
    WRITE = s3://<bucket>/<prefix>/coyo

A sweep or a derived experiment (`base:`) with no section of its own uses its base's. A value in
the environment (BENCH_READ / BENCH_WRITE) wins over the file. `bench launch` ships only the
launched experiment's two values to the node, as a flat READ= / WRITE= file.

A placeholder is resolved only where the path is used: a leg's argv and env, a listing, the
node's shell. Recorded files keep the placeholder form, plus `record()`: the names used and a
sha256 of their values, so two runs can be shown to have used the same paths.

The syntax is nova-bf's (`${VAR}`, `${VAR:-default}`); any other `${...}` is left as it is.
"""

from __future__ import annotations

import hashlib
import os
import re

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # clean_benchmarks
FILE = os.path.join(CB, "experiments", "local_paths.env")
EXAMPLE = os.path.join(CB, "experiments", "local_paths.env.example")

VARS = {
    "READ": "S3 prefix this experiment reads its corpus and queries from",
    "WRITE": "S3 prefix nova-bf writes its results under (the only S3 prefix a launch writes to)",
}
ENV = {"READ": "BENCH_READ", "WRITE": "BENCH_WRITE"}   # environment overrides
_SECTIONS: list[str] = []   # the active experiment's section names, most specific first
_REF = re.compile(r"\$\{(" + "|".join(VARS) + r")(:-[^}]*)?\}")
# a value goes into shell lines on the node unquoted, so only plain path characters
_SAFE = re.compile(r"[A-Za-z0-9@%+=:,./_-]+")


class PathsError(ValueError):
    """A placeholder with no value, or a bad values file."""


def read_file(path: str = FILE) -> dict:
    """{section: {KEY: value}} of a values file ("" = lines before any [section]; comments,
    blank lines, `export ` and quotes allowed); {} when there is no file."""
    out, sec = {}, ""
    if not os.path.isfile(path):
        return out
    for n, line in enumerate(open(path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip()
            out.setdefault(sec, {})
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        k, eq, v = line.partition("=")
        if not eq:
            raise PathsError(f"{path}:{n}: not KEY=VALUE: {line!r}")
        out.setdefault(sec, {})[k.strip()] = v.strip().strip('"').strip("'")
    return out


def use(exp: dict) -> list[str]:
    """Make `exp` the experiment whose section the placeholders resolve from: its name, then
    each `base:` experiment's name (nearest first). Returns the section names."""
    import yaml
    names = [exp.get("name")] if exp.get("name") else []
    for b in reversed(exp.get("_base") or []):
        try:
            with open(b) as f:
                n = (yaml.safe_load(f) or {}).get("name")
        except OSError:
            n = None
        if n and n not in names:
            names.append(n)
    _SECTIONS[:] = names
    return names


def values(path: str = FILE) -> dict:
    """{name: value} of READ / WRITE for the active experiment: the environment's, else its
    section's (or a base's), else the file's unsectioned lines (the node's flat file)."""
    f = read_file(path)
    out = {}
    for k in VARS:
        v = os.environ.get(ENV[k])
        for sec in [*_SECTIONS, ""]:
            if v:
                break
            v = (f.get(sec) or {}).get(k)
        if v:
            if not _SAFE.fullmatch(v):
                raise PathsError(f"{k}={v!r}: only letters, digits and @%+=:,./_- are allowed")
            out[k] = v.rstrip("/")
    return out


def _missing(name: str, path: str = FILE) -> PathsError:
    sec = _SECTIONS[0] if _SECTIONS else "<experiment name>"
    return PathsError(f"{name} is not set for experiment {sec!r}: add `[{sec}]` with "
                      f"`{name} = ...` to {path} (format: {os.path.basename(EXAMPLE)}), "
                      f"or export {ENV[name]}")


def refs(obj) -> set:
    """The VARS names a value (str, or dict / list of them, keys too) refers to."""
    out = set()
    if isinstance(obj, str):
        out.update(m.group(1) for m in _REF.finditer(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out |= refs(k) | refs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= refs(v)
    return out


def check(obj, path: str = FILE) -> dict:
    """The values a value refers to; refused (PathsError) when one has none and no default."""
    expand_all(obj, path)
    have = values(path)
    return {k: have[k] for k in sorted(refs(obj)) if k in have}


def expand(s, path: str = FILE):
    """A string with its placeholders filled in; anything else is returned as is."""
    if not isinstance(s, str) or "${" not in s:
        return s
    have = None

    def repl(m):
        nonlocal have
        if have is None:
            have = values(path)
        if m.group(1) in have:
            return have[m.group(1)]
        if m.group(2):
            return m.group(2)[2:]
        raise _missing(m.group(1), path)
    return _REF.sub(repl, s)


def expand_all(obj, path: str = FILE):
    """A value (str, or dict / list of them, keys too) with every placeholder filled in."""
    if isinstance(obj, str):
        return expand(obj, path)
    if isinstance(obj, dict):
        return {expand_all(k, path): expand_all(v, path) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_all(v, path) for v in obj]
    if isinstance(obj, tuple):
        return tuple(expand_all(v, path) for v in obj)
    return obj


def mask(obj, path: str = FILE):
    """A value with every resolved path value put back in placeholder form (keys too): what a
    recorded file holds."""
    try:
        have = values(path)
    except PathsError:
        have = {}
    pairs = sorted(((v, "${" + k + "}") for k, v in have.items()), key=lambda p: -len(p[0]))
    if not pairs:
        return obj

    def m(x):
        if isinstance(x, str):
            for v, ph in pairs:
                x = x.replace(v, ph)
            return x
        if isinstance(x, dict):
            return {m(k): m(v) for k, v in x.items()}
        if isinstance(x, list):
            return [m(v) for v in x]
        if isinstance(x, tuple):
            return tuple(m(v) for v in x)
        return x
    return m(obj)


def sha256(vals: dict) -> str | None:
    """sha256 over the sorted `NAME=value` lines; None for no values."""
    if not vals:
        return None
    return hashlib.sha256("".join(f"{k}={vals[k]}\n" for k in sorted(vals)).encode()).hexdigest()


def record(obj, path: str = FILE) -> dict:
    """What a recorded file says about the paths a value used: the names, and a sha256 of
    their values (never the values)."""
    have = values(path)
    used = {k: have[k] for k in sorted(refs(obj)) if k in have}
    return {"vars": sorted(refs(obj)), "sha256": sha256(used)}


def file_text(vals: dict) -> str:
    """A values file holding these values (what `bench launch` ships to the node)."""
    head = "# written by `bench launch`: the launched experiment's paths (from experiments/local_paths.env)\n"
    return head + "".join(f"{k}={vals[k]}\n" for k in sorted(vals))


def shell_word(s: str) -> str:
    """`s` as one shell word with its placeholders left for the shell to expand: unquoted when
    the rest is plain path characters (so the line reads as before), otherwise the literal
    parts quoted around "${VAR}"."""
    import shlex
    lit = _REF.sub("", s)
    if _SAFE.fullmatch(lit) or (lit == "" and s):
        return s
    out, i = [], 0
    for m in _REF.finditer(s):
        if s[i:m.start()]:
            out.append(shlex.quote(s[i:m.start()]))
        out.append(f'"{m.group(0)}"')
        i = m.end()
    if s[i:]:
        out.append(shlex.quote(s[i:]))
    return "".join(out)
