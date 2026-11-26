"""Experiment configuration schema and validation.

Unknown keys are rejected. Experiments define what to measure; timing and
execution are controlled by timing.py and leg.py.

Top-level fields:

  name        str: results directory (results/<name>/)
  modality    dense | sparse | multivector | filtered
  corpus      {path, column, recursive: false, region: null}
              Local or S3 path; local paths are relative to the repository root.
  queries     {path, column, nq: null}
              nq null: use the harness default.
  search      {k, metric}  metric: ip | cosine | l2
  io          {files_in_flight, gets, part_mib, read_mode, range_scope, chosen_from}: the
              default for every system; a system's own `io:` overrides it key by key
              Shared across systems; chosen_from identifies the I/O sweep.
  threads     int, required: OMP threads for all systems (0 = the usable CPU count)
  copies      {h2d, h2d_overlap, d2h, d2h_overlap}
              Host<->GPU copy modes of the GPU harness systems. Default: pinned and
              overlapped both ways (COPIES_DEFAULT); the file's values override it and a
              system's own copy knobs override both (e.g. `h2d: pageable, h2d_overlap: false`).
              The CPU references (cpu, masked) have no copies and never get these.
  filter      {config}: nova-bf filter configuration (filtered only)
  searches    [{name, systems, filter_search}]
              Each search specifies its systems and initial execution order.
  search_name str: default search name when searches is omitted
  systems     {name: preset | {system, preset, knobs..., io, chosen_from}}
              io: the system's own {files_in_flight, gets, part_mib, read_mode, range_scope}
  novabf_output  str: nova-bf output directory or S3 prefix
  e2e         {system, keep_cache: false, tiebreak}: cold end-to-end nova-bf run;
              tiebreak {mode: ordinal | id, field} (id: ties ordered by that corpus field)
  protocol    {rounds: 3}
  checks      {oracle_queries: 500, gt: null, exact_scan: false, max_score_ulps, rescore_max_gib}
              gt is reference-only; exact_scan uses a full float64 scan.
              max_score_ulps enables failure on excessive score error.
              groups: {system: group}, set by a sweep (each variant's base system): the pairs
              check still judges every pair (the same verdicts), but compares a variant for
              real only on the queries where its lists are not its group's first system's
              exactly (compare.py --group). Unset (a run): every pair, every query.
              rescore_max_gib: the float64 rescoring's memory cap (compare.py's default 32 GiB);
              a wide corpus on a big node raises it rather than checking fewer queries.
              copies_files / verify_files / check_repeats: bench check-copies /
              check-verify's corpus files and timed passes (their flags override them).
  prep        Data preparation steps used by bench launch:
              stage: files to copy or transform
              queries: {script, args}
              queries_src: optional query source URI

Sweep-only fields: base, sweep, subset.

Paths expand ~ at execution, and the private S3 prefixes `${READ}` / `${WRITE}` too
(bench/paths.py: their values are per experiment in experiments/local_paths.env, never in
these files). Values containing "TODO" are accepted during validation but rejected by
bench run.
"""


from __future__ import annotations

import copy
import itertools
import json
import os

from bench.systems import base as systems_base

MODALITIES = ("dense", "sparse", "multivector", "filtered")
METRICS = ("ip", "cosine", "l2")


# every GPU harness system copies through pinned buffers, overlapped with the search, both ways
COPIES_DEFAULT = {"h2d": "pinned", "h2d_overlap": True, "d2h": "pinned", "d2h_overlap": True}


class ExperimentError(ValueError):
    """An invalid experiment or sweep file."""
    pass


# section -> (required sub-keys, allowed sub-keys)
SECTIONS = {
    "corpus": ({"path", "column"}, {"path", "column", "recursive", "region"}),
    "queries": ({"path", "column"}, {"path", "column", "nq"}),
    "search": ({"k", "metric"}, {"k", "metric"}),
    "io": ({"files_in_flight", "gets", "part_mib"},
           {"files_in_flight", "gets", "part_mib", "read_mode", "range_scope", "chosen_from"}),
    "copies": (set(), {"h2d", "h2d_overlap", "d2h", "d2h_overlap"}),
    "filter": ({"config"}, {"config"}),
    "protocol": (set(), {"rounds"}),
    "checks": (set(), {"oracle_queries", "gt", "exact_scan", "max_score_ulps", "rescore_max_gib", "groups",
                       "copies_files", "verify_files", "check_repeats"}),
    "e2e": ({"system"}, {"system", "keep_cache", "tiebreak"}),
    "subset": ({"files"}, {"files"}),
    "prep": (set(), {"stage", "queries", "queries_src"}),
}
TOP = {"name", "modality", "corpus", "queries", "search", "io", "threads", "copies", "filter",
       "searches", "search_name", "systems", "novabf_output", "e2e", "protocol", "checks",
       "base", "sweep", "subset", "prep"}
REQUIRED = ("name", "modality", "corpus", "queries", "search", "systems")


def _section(doc, key, where):
    if key not in doc:
        return
    v = doc[key]
    req, allowed = SECTIONS[key]
    if not isinstance(v, dict):
        raise ExperimentError(f"{where}{key}: a mapping, not {v!r}")
    unknown = set(v) - allowed
    if unknown:
        raise ExperimentError(f"{where}{key}: unknown key(s) {sorted(unknown)} (allowed: {sorted(allowed)})")
    missing = req - set(v)
    if missing:
        raise ExperimentError(f"{where}{key}: missing {sorted(missing)}")


def _int(v, where, lo=1):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ExperimentError(f"{where}: an integer >= {lo}, not {v!r}")
    return v


COPY_SPEC = systems_base.SystemSpec("copies", "harness", None, (), systems_base.COPY_KNOBS)


def leg_copies(copies: dict, sysr: dict) -> dict:
    """The copy knobs a harness system runs with: the experiment's `copies` (defaults filled)
    under the system's own copy knobs. A CPU reference (cpu, masked) does no host<->GPU copy:
    it gets only what it sets itself (validate refuses pinned / overlap there)."""
    own = {k: v for k, v in sysr["knobs"].items() if k in systems_base.COPY_KNOBS}
    if systems_base.SYSTEMS[sysr["system"]].backend in systems_base.HOST_BACKENDS:
        return own
    return {**copies, **own}


def copies_problem(copies: dict, sysr: dict) -> str | None:
    """Why a system's resolved copy modes cannot run (the harness would refuse them), or None."""
    if sysr["kind"] != "harness":
        return None
    c = leg_copies(copies, sysr)
    if systems_base.SYSTEMS[sysr["system"]].backend in systems_base.HOST_BACKENDS:
        if c.get("h2d") == "pinned" or c.get("d2h") == "pinned" or c.get("h2d_overlap") \
                or c.get("d2h_overlap"):
            return "a CPU reference has no host<->GPU copies: no pinned / overlapped copy knobs"
        return None
    for side in ("h2d", "d2h"):
        if c.get(f"{side}_overlap") and c.get(side) != "pinned":
            return f"{side}_overlap: true needs {side}: pinned (set {side}_overlap: false with {side}: pageable)"
    return None


def validate(doc: dict, where: str = "") -> dict:
    """Check a merged experiment (base applied) and return its normalised form: defaults
    filled, systems resolved, `searches` explicit. Raises ExperimentError."""
    if not isinstance(doc, dict):
        raise ExperimentError(f"{where}an experiment is a mapping")
    unknown = set(doc) - TOP
    if unknown:
        raise ExperimentError(f"{where}unknown top-level key(s) {sorted(unknown)} (allowed: {sorted(TOP)})")
    for k in REQUIRED:
        if k not in doc:
            raise ExperimentError(f"{where}missing `{k}`")
    for k in SECTIONS:
        _section(doc, k, where)
    e = copy.deepcopy(doc)
    if not isinstance(e["name"], str) or not e["name"] or "/" in e["name"]:
        raise ExperimentError(f"{where}name: a plain string, not {e['name']!r}")
    if e["modality"] not in MODALITIES:
        raise ExperimentError(f"{where}modality: one of {MODALITIES}, not {e['modality']!r}")
    e["corpus"].setdefault("recursive", False)
    e["corpus"].setdefault("region", None)
    e["queries"].setdefault("nq", None)
    if e["queries"]["nq"] is not None and not _is_todo(e["queries"]["nq"]):
        _int(e["queries"]["nq"], f"{where}queries.nq")
    _int(e["search"]["k"], f"{where}search.k")
    if e["search"]["metric"] not in METRICS:
        raise ExperimentError(f"{where}search.metric: one of {METRICS}")
    if "io" in e:
        for k in ("files_in_flight", "gets", "part_mib"):
            if not _is_todo(e["io"][k]):      # a TODO placeholder: refused by `bench run`
                _int(e["io"][k], f"{where}io.{k}")
        e["io"].setdefault("read_mode", "native")
        e["io"].setdefault("range_scope", "total")
    else:
        e["io"] = None
    # required: SkyPilot sets OMP_NUM_THREADS=1, and only this value resets the baselines' CPU
    # thread pools (nova-bf sizes its own), so a missing value would starve the baselines.
    # 0 = auto: every leg uses the machine's usable CPU count (timing.omp_threads).
    if "threads" not in e:
        raise ExperimentError(f"{where}threads: required (CPU threads for every system, "
                              "e.g. 32, or 0 for the usable CPU count)")
    _int(e["threads"], f"{where}threads", lo=0)
    e["copies"] = {**COPIES_DEFAULT, **e.get("copies", {})}
    for k, v in e["copies"].items():
        try:
            systems_base.check_knob(COPY_SPEC, k, v, where=f"{where}copies.{k}")
        except systems_base.SystemError_ as exc:
            raise ExperimentError(str(exc)) from None
    e.setdefault("protocol", {})
    e["protocol"].setdefault("rounds", 3)
    _int(e["protocol"]["rounds"], f"{where}protocol.rounds")
    e.setdefault("checks", {})
    if not isinstance(e["checks"].get("exact_scan", False), bool):
        raise ExperimentError(f"{where}checks.exact_scan: true / false")
    msu = e["checks"].get("max_score_ulps")
    if msu is not None and (isinstance(msu, bool) or not isinstance(msu, (int, float)) or not msu >= 0):
        raise ExperimentError(f"{where}checks.max_score_ulps: a number >= 0 (or unset)")
    grp = e["checks"].get("groups")
    if grp is not None and not (isinstance(grp, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in grp.items())):
        raise ExperimentError(f"{where}checks.groups: a mapping system -> group name (or unset)")
    rmg = e["checks"].get("rescore_max_gib")
    if rmg is not None and (isinstance(rmg, bool) or not isinstance(rmg, (int, float)) or not rmg > 0):
        raise ExperimentError(f"{where}checks.rescore_max_gib: a number > 0 (or unset)")
    for k in ("copies_files", "verify_files", "check_repeats"):
        if k in e["checks"]:
            _int(e["checks"][k], f"{where}checks.{k}")
    e.setdefault("novabf_output", None)
    if e["modality"] == "filtered" and "filter" not in e:
        raise ExperimentError(f"{where}a filtered experiment needs `filter: {{config: ...}}`")
    if e["modality"] != "filtered" and "filter" in e:
        raise ExperimentError(f"{where}`filter` is for modality filtered only")

    # systems
    if not isinstance(e["systems"], dict) or not e["systems"]:
        raise ExperimentError(f"{where}systems: a non-empty mapping")
    resolved = {}
    for name, entry in e["systems"].items():
        if not isinstance(name, str) or "__" in name or "/" in name:
            raise ExperimentError(f"{where}systems: bad name {name!r} (no '__' or '/')")
        try:
            r = systems_base.resolve(name, entry)
        except systems_base.SystemError_ as exc:
            raise ExperimentError(f"{where}{exc}") from None
        spec = systems_base.SYSTEMS[r["system"]]
        if e["modality"] not in spec.modalities:
            raise ExperimentError(f"{where}systems.{name}: {r['system']} does not run "
                                  f"{e['modality']} (it runs {list(spec.modalities)})")
        problem = copies_problem(e["copies"], r)
        if problem:
            raise ExperimentError(f"{where}systems.{name}: {problem}")
        resolved[name] = r
    e["systems"] = resolved

    # searches
    if "searches" in e:
        if "search_name" in e:
            raise ExperimentError(f"{where}`searches` and `search_name` are exclusive")
        ss = e["searches"]
        if not isinstance(ss, list) or not ss:
            raise ExperimentError(f"{where}searches: a non-empty list")
        names = set()
        for i, s in enumerate(ss):
            if not isinstance(s, dict) or set(s) - {"name", "systems", "filter_search"} \
                    or {"name", "systems"} - set(s):
                raise ExperimentError(f"{where}searches[{i}]: {{name, systems, filter_search}}")
            if s["name"] in names or "__" in s["name"]:
                raise ExperimentError(f"{where}searches[{i}]: duplicate or bad name {s['name']!r}")
            names.add(s["name"])
            for sy in s["systems"]:
                if sy not in resolved:
                    raise ExperimentError(f"{where}searches[{i}].systems: {sy!r} not in systems")
            if len(set(s["systems"])) != len(s["systems"]):
                raise ExperimentError(f"{where}searches[{i}].systems: a system twice")
            s.setdefault("filter_search", s["name"] if e["modality"] == "filtered" else None)
    else:
        sn = e.pop("search_name", None) or e["name"]
        e["searches"] = [{"name": sn, "systems": list(resolved),
                          "filter_search": None}]
        if e["modality"] == "filtered":
            raise ExperimentError(f"{where}a filtered experiment lists its `searches` "
                                  "(each names the filter search)")
    e.pop("search_name", None)
    used = {sy for s in e["searches"] for sy in s["systems"]}
    unused = set(resolved) - used - ({e["e2e"]["system"]} if "e2e" in e else set())
    if unused:
        raise ExperimentError(f"{where}systems {sorted(unused)} are in no search")
    if "e2e" in e:
        if e["e2e"]["system"] not in resolved or resolved[e["e2e"]["system"]]["kind"] != "novabf":
            raise ExperimentError(f"{where}e2e.system: a nova-bf system of this experiment")
        e["e2e"].setdefault("keep_cache", False)
        tb = e["e2e"].setdefault("tiebreak", {"mode": "ordinal"})
        if not isinstance(tb, dict) or tb.get("mode") not in ("ordinal", "id") \
                or set(tb) - {"mode", "field"} or (tb["mode"] == "id") != bool(tb.get("field")):
            raise ExperimentError(f"{where}e2e.tiebreak: {{mode: ordinal}} or "
                                  "{mode: id, field: <corpus column>}")
    else:
        e["e2e"] = None
    if "subset" in e:
        _int(e["subset"]["files"], f"{where}subset.files")
    else:
        e["subset"] = None
    e["prep"] = _prep(e.get("prep") or {}, f"{where}prep")
    return e


PREP_HOW = {"copy": set(), "filtered_queries": {"drop_columns"}}


def _prep(p: dict, where: str) -> dict:
    out = {"stage": [], "queries": None, "queries_src": None}
    for i, d in enumerate(p.get("stage") or []):
        w = f"{where}.stage[{i}]"
        if not isinstance(d, dict) or {"key", "path", "source", "how"} - set(d):
            raise ExperimentError(f"{w}: {{key, path, source, how, ...}}")
        if d["how"] not in PREP_HOW:
            raise ExperimentError(f"{w}.how: one of {sorted(PREP_HOW)}")
        extra = set(d) - {"key", "path", "source", "how"} - PREP_HOW[d["how"]]
        if extra or PREP_HOW[d["how"]] - set(d):
            raise ExperimentError(f"{w}: {d['how']} takes {sorted(PREP_HOW[d['how']])}")
        out["stage"].append(dict(d))
    if p.get("queries") is not None:
        q = p["queries"]
        if not isinstance(q, dict) or set(q) != {"script", "args"} or not isinstance(q["args"], list):
            raise ExperimentError(f"{where}.queries: {{script, args: [...]}}")
        out["queries"] = {"script": q["script"], "args": [str(x) for x in q["args"]]}
    if p.get("queries_src") is not None:
        if not isinstance(p["queries_src"], str):
            raise ExperimentError(f"{where}.queries_src: a URI or null")
        out["queries_src"] = p["queries_src"]
    return out


def _is_todo(v) -> bool:
    """True for a placeholder value (a string containing TODO)."""
    return isinstance(v, str) and "TODO" in v


def placeholders(e: dict) -> list[str]:
    """Every value that still says TODO (dotted paths)."""
    out = []

    def walk(v, p):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{p}.{k}" if p else str(k))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{p}[{i}]")
        elif _is_todo(v):
            out.append(f"{p} = {v!r}")
    # look everywhere in the resolved experiment (system knobs, novabf_output, checks, filter,
    # prep, ...), except the bookkeeping keys (_file, _base)
    walk({k: v for k, v in e.items() if not str(k).startswith("_")}, "")
    # an `io:` TODO that every system's sweep io overrides is never used, so it is not one
    syss = [s for s in (e.get("systems") or {}).values() if isinstance(s, dict)]
    covered = {f"io.{k}" for k in systems_base.SWEEP_IO_KEYS
               if syss and all(k in (s.get("io") or {}) for s in syss)}
    return [p for p in out if p.split(" = ")[0] not in covered]


# ---------------------------------------------------------------------------------------------
# files: `base:` inheritance, sweeps
# ---------------------------------------------------------------------------------------------
def _read_yaml(path):
    """A YAML file that must hold a mapping."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ExperimentError(f"{path}: not a mapping")
    return doc


def _merge(base: dict, over: dict) -> dict:
    """Override merge: mappings merge key by key, anything else is replaced. `systems` and
    `searches` are replaced whole when the overriding file gives them."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in ("systems", "searches") or not isinstance(v, dict) or not isinstance(out.get(k), dict):
            out[k] = copy.deepcopy(v)
        else:
            out[k] = _merge(out[k], v)
    return out


def _abs_paths(doc: dict, here: str) -> dict:
    """Resolve the file references (nova-bf configs, the filter config) against the experiment
    file's directory. Data paths (corpus, queries) are left alone: they are relative to the
    leg's cwd."""
    d = copy.deepcopy(doc)
    for name, entry in (d.get("systems") or {}).items():
        if isinstance(entry, dict) and isinstance(entry.get("config"), str):
            entry["config"] = os.path.normpath(os.path.join(here, entry["config"]))
    if isinstance(d.get("filter"), dict) and isinstance(d["filter"].get("config"), str):
        d["filter"]["config"] = os.path.normpath(os.path.join(here, d["filter"]["config"]))
    return d


def load_raw(path: str, _seen=None) -> dict:
    """The file with its `base:` chain applied (file references made absolute), unvalidated.
    The sweep keys stay in the result."""
    path = os.path.abspath(path)
    _seen = _seen or []
    if path in _seen:
        raise ExperimentError(f"{path}: base: cycle {_seen}")
    doc = _abs_paths(_read_yaml(path), os.path.dirname(path))
    b = doc.pop("base", None)
    if b is not None:
        if not isinstance(b, str):
            raise ExperimentError(f"{path}: base: a relative path")
        parent = load_raw(os.path.join(os.path.dirname(path), b), _seen + [path])
        parent.pop("sweep", None)
        parent.pop("subset", None)
        doc = _merge(parent, doc)
        doc["_base"] = parent.get("_base", []) + [os.path.normpath(os.path.join(os.path.dirname(path), b))]
    return doc


EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
# experiment folders by modality; novabf/ holds nova-bf configs, sweeps/ tuning grids
EXPERIMENT_FOLDERS = ("dense", "sparse", "multivector", "filtered")


def experiment_files() -> list[str]:
    """Every experiment file (absolute paths, sorted): experiments/<folder>/*.yaml."""
    import glob
    return sorted(p for d in EXPERIMENT_FOLDERS
                  for p in glob.glob(os.path.join(EXPERIMENTS_DIR, d, "*.yaml")))


def path_of(name: str) -> str:
    """The experiment file whose `name:` is ``name`` (path relative to clean_benchmarks)."""
    import yaml
    for p in experiment_files():
        with open(p) as f:
            if (yaml.safe_load(f) or {}).get("name") == name:
                return os.path.relpath(p, os.path.dirname(EXPERIMENTS_DIR))
    raise ExperimentError(f"no experiment named {name!r} under {EXPERIMENTS_DIR}")


def load(path: str) -> dict:
    """An experiment file (not a sweep): validated. A file with `sweep:` is refused here."""
    raw = load_raw(path)
    if "sweep" in raw:
        raise ExperimentError(f"{path}: a sweep file (has `sweep:`): use `bench sweep`")
    return finish(raw, path)


def finish(raw: dict, path: str) -> dict:
    """Validate a raw experiment and record its file and base chain (_file, _base)."""
    raw = dict(raw)
    bases = raw.pop("_base", [])
    e = validate(raw, where=f"{os.path.basename(path)}: ")
    e["_file"] = os.path.abspath(path)
    e["_base"] = bases
    from bench import paths
    paths.use(e)          # its ${READ} / ${WRITE} resolve from its own section
    return e


# ---- sweeps ---------------------------------------------------------------------------------
SWEEP_KEYS = {"system", "grid", "searches"}   # searches: the entry's variants run only there
SWEEP_ROUNDS_DEFAULT = 1   # a sweep: one warm run per variant


def sweep_variants(raw: dict, path: str, keep_rounds: bool = False) -> tuple[dict, list[dict]]:
    """(the experiment that runs every variant, [variant descriptions]) of a sweep file.

    sweep: {system: NAME, grid: {knob: [values], ...}}, or a list of such entries (several
           systems, or one system over two grids). An entry without `grid` runs that base
           system unchanged at that position (e.g. nova-bf as a reference in the rotation). Each grid point is a copy of systems.<NAME> with the point's knobs set,
           named `<NAME>-<knob>-<value>[-<knob>-<value>...]`. Variants run in file order, each
           grid as itertools.product over its keys in file order; that list is round 1's
           order (timing.py rotates it). Knobs must be declared sweepable (systems/base.py).
           A grid key `io.<key>` (files_in_flight, gets, part_mib, read_mode, range_scope)
           sets that `io:` value for the variant only, for any system and modality (nova-bf:
           only the keys its config takes from ${NOVABF_*}).
    The sweep runs on the same leg runner as a final run."""
    sw = raw.get("sweep")
    entries = sw if isinstance(sw, list) else [sw]
    base_systems = raw.get("systems") or {}
    systems, variants = {}, []
    for i, ent in enumerate(entries):
        if not isinstance(ent, dict) or set(ent) - SWEEP_KEYS or "system" not in ent:
            raise ExperimentError(f"{path}: sweep[{i}]: {{system[, grid]}}")
        name = ent["system"]
        if name not in base_systems:
            raise ExperimentError(f"{path}: sweep[{i}].system {name!r} not in the base's systems")
        only = ent.get("searches")
        if only is not None:   # e.g. FAISS (static filters only) on the static searches
            known = [x.get("name") for x in raw.get("searches") or []]
            if not isinstance(only, list) or not only or any(x not in known for x in only) \
                    or len(set(only)) != len(only):
                raise ExperimentError(f"{path}: sweep[{i}].searches: a non-empty list of distinct "
                                      f"base search names (the base's: {known})")
        r = systems_base.resolve(name, base_systems[name])
        spec = systems_base.SYSTEMS[r["system"]]
        if "grid" not in ent:
            if name in systems:
                raise ExperimentError(f"{path}: sweep[{i}]: {name} twice")
            systems[name] = base_systems[name]
            variants.append({"variant": name, "base_system": name, "setting": {}, "included": True})
            if only is not None:
                variants[-1]["searches"] = list(only)
            continue
        grid = ent["grid"]
        if not isinstance(grid, dict) or not grid:
            raise ExperimentError(f"{path}: sweep[{i}].grid: a non-empty mapping")
        for k, vals in grid.items():
            if k.startswith("io."):   # an io key: any system, any modality
                ik = k[3:]
                if ik not in systems_base.SWEEP_IO_KEYS:
                    raise ExperimentError(f"{path}: sweep[{i}].grid: {k!r}: io keys are "
                                          f"{sorted('io.' + x for x in systems_base.SWEEP_IO_KEYS)}")
                if not raw.get("io"):
                    raise ExperimentError(f"{path}: sweep[{i}].grid: {k!r} needs the base's `io:`")
                if spec.kind == "novabf":
                    var = systems_base.SWEEP_IO_NOVABF_VAR.get(ik)
                    cfg = r["knobs"].get("config")
                    text = open(cfg).read() if cfg and os.path.isfile(cfg) else ""
                    if var is None or "${" + var not in text:
                        raise ExperimentError(
                            f"{path}: sweep[{i}].grid: {k!r} does not reach nova-bf "
                            f"({'no nova-bf setting' if var is None else f'its config has no ${{{var}}}'})")
                shadow = io_shadow(spec, r, ik, grid)
                if shadow:   # the message says how much is shadowed
                    raise ExperimentError(f"{path}: sweep[{i}].grid: {k!r} is {shadow}")
                if not isinstance(vals, list) or not vals:
                    raise ExperimentError(f"{path}: sweep[{i}].grid.{k}: a non-empty list")
                continue
            if k not in spec.knobs:
                raise ExperimentError(f"{path}: sweep[{i}].grid: {r['system']} has no knob {k!r}")
            if not spec.knobs[k].sweepable:
                raise ExperimentError(f"{path}: sweep[{i}].grid: knob {k!r} of {r['system']} "
                                      "is not sweepable (bench/systems/base.py)")
            if not isinstance(vals, list) or not vals:
                raise ExperimentError(f"{path}: sweep[{i}].grid.{k}: a non-empty list")
        keys = list(grid)
        entry = base_systems[name]
        entry = {"preset": entry} if isinstance(entry, str) else dict(entry)
        entry.setdefault("system", name)
        entry.pop("chosen_from", None)
        for combo in itertools.product(*(grid[k] for k in keys)):
            setting = dict(zip(keys, combo))
            vname = name + "".join(f"-{k}-{_vstr(v)}" for k, v in setting.items())
            if vname in systems:
                raise ExperimentError(f"{path}: duplicate variant {vname}")
            knobs = {k: v for k, v in setting.items() if not k.startswith("io.")}
            io = {k[3:]: v for k, v in setting.items() if k.startswith("io.")}
            systems[vname] = {**entry, **knobs, **({"sweep_io": io} if io else {})}
            variants.append({"variant": vname, "base_system": name, "setting": setting,
                             **({"searches": list(only)} if only is not None else {})})
    exp = {k: v for k, v in raw.items() if k not in ("sweep", "searches", "systems", "search_name")}
    exp["systems"] = systems
    order = list(systems)
    if raw.get("searches"):
        # a multi-search base: every search runs every variant, except that a variant whose
        # entry names `searches` runs only in those
        only_in = {v["variant"]: v.get("searches") for v in variants}
        exp["searches"] = [{**{k: v for k, v in s.items() if k != "systems"},
                            "systems": [x for x in order
                                        if only_in.get(x) is None or s["name"] in only_in[x]]}
                           for s in raw["searches"]]
        empty = [s["name"] for s in exp["searches"] if not s["systems"]]
        if empty:
            raise ExperimentError(f"{path}: no variant runs in search(es) {empty}")
    elif any(v.get("searches") for v in variants):
        raise ExperimentError(f"{path}: sweep `searches:` needs a multi-search base")
    elif raw.get("search_name"):
        exp["search_name"] = raw["search_name"]
    exp.pop("e2e", None)
    # the pairs check groups a sweep's variants by their base system (all pairs grow
    # quadratically: 435 for a 30-variant grid): a variant whose lists equal its group's first
    # system's on a query takes that system's outcomes there (compare.py --group)
    exp["checks"] = {**(exp.get("checks") or {}),
                     "groups": {v["variant"]: v["base_system"] for v in variants}}
    own = _read_yaml(os.path.abspath(path)) if os.path.isfile(path) else {}
    # a sweep is one warm run per variant, unless the sweep file itself sets protocol.rounds
    # (the base's rounds are the paper run's, so they do not count) or keep_rounds is set
    if not keep_rounds and "rounds" not in ((own.get("protocol") or {}) if isinstance(own, dict) else {}):
        exp["protocol"] = {**(exp.get("protocol") or {}), "rounds": SWEEP_ROUNDS_DEFAULT}
    return exp, variants


# the harness knobs that override an io key in the leg's args (leg.harness_args:
# --read-mode / --prefetch / --workers take the knob's value over the io value)
HARNESS_IO_SHADOW = {"files_in_flight": ("prefetch", "workers"), "read_mode": ("read_mode",)}


def io_shadow(spec, r: dict, ik: str, grid: dict) -> str | None:
    """What, if anything, overrides a swept `io.<ik>` in the leg the variant runs.

    Harness: a knob (the system's, a preset's, or the same grid's) that leg.harness_args uses
    in place of the io value. nova-bf: the merged config params (config file plus the `params`
    knob, as leg.novabf_config merges them) no longer referencing the io key's ${NOVABF_*}
    variable (e.g. params.io_workers / rust_fetch_*). Returns None, or the refusal text that
    follows "<key> is " ("shadowed by ..." / "partly shadowed by ...")."""
    if spec.kind != "novabf":
        # files_in_flight feeds two flags (--prefetch, --workers). Both pinned: every variant
        # runs the same value. One pinned: only the other flag would vary, a sweep of half
        # the setting. Both are refused, with a message that says which.
        flags = HARNESS_IO_SHADOW.get(ik, ())
        pins = {}
        for kn in flags:
            if kn in grid:
                pins[kn] = (f"the grid's own knob {kn!r} (--{kn.replace('_', '-')})",
                            f"--{kn.replace('_', '-')} is swept by the grid itself")
            elif kn in r["knobs"]:
                pins[kn] = (f"the system's knob {kn}={r['knobs'][kn]!r} (--{kn.replace('_', '-')})",
                            f"--{kn.replace('_', '-')} is fixed at {r['knobs'][kn]!r}")
        if not pins:
            return None
        by = " and ".join(p[0] for p in pins.values())
        if len(pins) == len(flags):
            return (f"shadowed by {by}: the variants would all run the same value (drop the "
                    "knob from the system, or sweep it instead)")
        free = [f"--{kn.replace('_', '-')}" for kn in flags if kn not in pins]
        names = " and ".join(f"--{kn.replace('_', '-')}" for kn in flags)
        return (f"partly shadowed by {by}: {'; '.join(p[1] for p in pins.values())}; only "
                f"{' and '.join(free)} would vary; set both {names} as knobs or neither")
    var = systems_base.SWEEP_IO_NOVABF_VAR.get(ik)
    cfg = r["knobs"].get("config")
    if var is None or not cfg or not os.path.isfile(cfg):
        return None
    c = _read_yaml(cfg)
    base = c.get("params") if isinstance(c.get("params"), dict) else {}
    knob = r["knobs"].get("params") or {}
    merged = {**base, **knob}
    ref = "${" + var
    used = [p for p, v in base.items() if isinstance(v, str) and ref in v]
    for p in used:
        if not (isinstance(merged.get(p), str) and ref in merged[p]):
            return (f"shadowed by the system's params.{p}={merged.get(p)!r} (merged over the "
                    f"config's {base[p]!r}): the variants would all run the same value (drop the "
                    "knob from the system, or sweep it instead)")
    return None


def _vstr(v) -> str:
    """A knob value as it appears in a variant name."""
    s = systems_base._fmt(v) if not isinstance(v, bool) else ("on" if v else "off")
    return s.replace("/", "_").replace(" ", "")


def load_sweep(path: str) -> tuple[dict, list[dict], dict]:
    """(validated experiment of the variants, the variants, the raw sweep section)."""
    raw = load_raw(path)
    if "sweep" not in raw:
        raise ExperimentError(f"{path}: no `sweep:` (not a sweep file)")
    sweep_raw = copy.deepcopy(raw["sweep"])
    exp_raw, variants = sweep_variants(raw, path)
    e = finish(exp_raw, path)
    return e, variants, sweep_raw


def dumps(e: dict) -> str:
    """The resolved experiment as the run dir's experiment.json text (private paths in
    their placeholder form)."""
    from bench import paths
    return json.dumps(paths.mask(e), indent=1, sort_keys=True)


def main(argv=None) -> int:
    """`python -m bench.experiment FILE --get KEY`: print one top-level value of the file with
    its `base:` chain applied (env/paper_run.sh asks for `modality`). Prints nothing and exits
    1 when the file is not an experiment or sweep file, or has no such key."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m bench.experiment")
    ap.add_argument("file")
    ap.add_argument("--get", required=True)
    a = ap.parse_args(argv)
    try:
        v = load_raw(a.file).get(a.get)
    except (ExperimentError, OSError, ValueError) as exc:
        print(f"bench.experiment: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001  a YAML error: not an experiment file
        print(f"bench.experiment: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if v is None:
        return 1
    print(v if isinstance(v, str) else json.dumps(v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
