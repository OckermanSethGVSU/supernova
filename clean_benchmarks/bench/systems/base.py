"""System definitions, presets, and configurable knobs.

Supports harness baselines and nova-bf. Experiment settings override preset
values, while unset knobs use library defaults.

Each knob defines its CLI mapping and whether it can be swept.
Unknown knobs are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Knob:
    flag: str | None          # harness flag ("--spare-mode"); None for nova-bf knobs
    type: type                # str / int / float / bool (bool: a store_true flag)
    sweepable: bool
    help: str
    choices: tuple | None = None
    flag_when: str | None = None   # a str knob that is a bare flag: passed (alone) iff value == this


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str                           # "harness" | "novabf"
    backend: str | None                 # harness --backend
    modalities: tuple                   # the experiment modalities it runs
    knobs: dict = field(default_factory=dict)
    presets: dict = field(default_factory=dict)
    cpu_ok: bool = False                # runs on a CPU dev box (bench run --dev)
    doc: str = ""
    mode_knob: str | None = None        # the knob that picks an engine mode (its choices = the modes)
    mode_only: dict = field(default_factory=dict)   # knob -> takes_it(mode), for knobs some modes refuse


# the harness backends with no device: no host<->GPU copies, no CUDA events
# (= bench.record.schema.HOST_BACKENDS; that module imports faiss)
HOST_BACKENDS = ("cpu", "masked")

# ---- engine modes (the engines and the harness CLI import these) ----------------------------
SPARE_MODES = ("dot", "iterative", "dot-released", "iterative-released",
               "iterative-released-nosync")
# modes whose search loop is our code, so stage timing events can go between torch
# calls without touching SPARe's own forward()
SPARE_STAGED_MODES = ("dot", "iterative", "iterative-released-nosync")
PYLATE_MODES = ("rerank-released", "rerank-released-ownpad", "scores-batched")


def spare_released(mode):
    """A --spare-mode that runs a per-query loop (no batch, no budget)."""
    return "released" in mode


def pylate_released(mode):
    return mode.startswith("rerank")


# ---- knobs shared by every harness system ------------------------------------------------
# host<->GPU copy modes; not sweepable: they are checked
# for identical answers (d2hcheck / overlapcheck), and the experiment sets them for everyone
# (default pinned + overlapped, bench/experiment.py COPIES_DEFAULT). The values unset here are
# the harness's own defaults (pageable, no overlap), the stock library usage.
COPY_KNOBS = {
    "h2d": Knob("--h2d", str, False, "host->GPU copy: pageable | pinned", ("pageable", "pinned")),
    "h2d_overlap": Knob("--h2d-overlap", bool, False, "overlap the upload with the previous search"),
    "d2h": Knob("--d2h", str, False, "GPU->host copy: pageable | pinned", ("pageable", "pinned")),
    "d2h_overlap": Knob("--d2h-overlap", bool, False, "overlap copy back + merge with the next search"),
}
# reader pipeline overrides of the experiment's shared `io:` (--prefetch N --workers N;
# filtered's per-variant --read-mode native). Not sweepable: a sweep sets shared I/O as `io.*`.
IO_KNOBS = {
    "read_mode": Knob("--read-mode", str, False, "reader mode (overrides io.read_mode)"),
    "prefetch": Knob("--prefetch", int, False, "files in flight (overrides io.files_in_flight)"),
    "workers": Knob("--workers", int, False, "reader processes (overrides io.files_in_flight)"),
}
FILTER_KNOBS = {
    "filter_mode": Knob("--filter-mode", str, True, "filtered search path", ("bitmap", "gather")),
    "filter_compact": Knob("--filter-compact", str, True, "row compaction", ("none", "union")),
    "filter_skip": Knob("--filter-skip", str, True, "dead-file / dead-query skipping",
                        ("queries", "files", "none")),
    "filter_threads": Knob("--filter-threads", int, True, "DuckDB filter threads"),
}
# CUDA event timing in the harness's GPU engines. Unset = on (the harness default); `off`
# passes --no-gpu-events. Set by hand only: the events sweeps just report on vs off.
EVENTS_KNOBS = {"gpu_events": Knob("--no-gpu-events", str, True,
                                   "CUDA event timing (on = harness default; off = --no-gpu-events)",
                                   ("on", "off"), flag_when="off")}
SPARSE_PROJECT = {"sparse_project": Knob("--sparse-project", str, True,
                                         "query-vocabulary projection", ("off", "query-vocab"))}


def _harness(name, backend, modalities, own, presets, cpu_ok=False, doc="", mode_knob=None,
             mode_only=None):
    knobs = {**COPY_KNOBS, **IO_KNOBS, **own, **EVENTS_KNOBS}   # CPU backends: gpu_events n/a
    if "filtered" in modalities:
        knobs.update(FILTER_KNOBS)
    return SystemSpec(name, "harness", backend, tuple(modalities), knobs,
                      {"stock": {}, **presets}, cpu_ok, doc, mode_knob, dict(mode_only or {}))


SYSTEMS: dict[str, SystemSpec] = {}


def _add(spec):
    SYSTEMS[spec.name] = spec


_add(_harness("faiss", "gpu", ("dense", "filtered"), {
    # FAISS's own temp-memory knob (StandardGpuResources.setTempMemory), the one
    # a competent FAISS user tunes; unset = not passed = FAISS's default (1.5 GiB cap on > 8 GiB GPUs)
    "temp_mem": Knob("--temp-mem", int, True,
                     "FAISS StandardGpuResources temp memory, bytes (unset = FAISS default; 0 = none)"),
}, {}, doc="stock FAISS faiss.knn_gpu (harness --backend gpu)"))
_add(_harness("cuvs", "cuvs", ("dense", "sparse", "filtered"), {
    "cuvs_rmm_pool": Knob("--cuvs-rmm-pool", str, True, "RMM pool size (harness default 1GiB; 0 = off)"),
    # unset = cuVS's stock sparse batch sizes
    "cuvs_sparse_batches": Knob("--cuvs-sparse-batches", str, True,
                                "sparse batch_size_index x batch_size_query (e.g. 8192x32768)"),
    # device = cuVS's own recipe (calc_truth: GPU merge, one copy back; every preset's default);
    # host = each file's results merged on the host (per_file_host)
    # sweepable: the dense cuVS sweep compares the recipe's device merge (with and without its
    # per-file sync) against the per-file host merge, and keeps the fastest
    "cuvs_merge": Knob("--cuvs-merge", str, True, "top-k merge: device (cuVS's own recipe) | "
                       "host (per-file + host ResultHeap)", ("device", "host")),
    # per-file = calc_truth's resources.sync() after each file (the default); none = no
    # per-file host wait, ordered by the stream alone (device merge only; ours, disclosed)
    "cuvs_sync": Knob("--cuvs-sync", str, True, "device merge: per-file (calc_truth's "
                      "resources.sync() per file) | none (no per-file host wait)",
                      ("per-file", "none")),
    **SPARSE_PROJECT,
}, {
    # an 8 GiB RMM pool
    "rmm_pool_8g": {"cuvs_rmm_pool": "8GiB"},
    # a comparison variant: each file searched, its results copied back and merged on the host
    # (faiss.ResultHeap), the streaming pattern of NVIDIA's other exact tool (cuvs_bench
    # synthesize_dataset _ground_truth.py compute_groundtruth_exact: per chunk, copy_to_host,
    # numpy merge)
    "per_file_host": {"cuvs_merge": "host"},
}, doc="stock cuVS brute force (dense, filtered) / the patched sparse cuVS (baselines/cuvs), "
       "run with cuVS's own exact-search recipe (calc_truth: GPU merge, one copy back)"))
_add(_harness("spare", "spare", ("sparse",), {
    "spare_mode": Knob("--spare-mode", str, True, "SPARe mode", SPARE_MODES),
    "spare_max_mib": Knob("--spare-max-mib", int, True, "batched SPARe memory budget (MiB)"),
    "no_spare_events": Knob("--no-spare-events", bool, True, "drop SPARe's CUDA stage events"),
    **SPARSE_PROJECT,
}, {
    # the sparse experiments' SPARe line: batched dot
    "batched_dot": {"spare_mode": "dot"},
    "batched_iterative": {"spare_mode": "iterative"},
    "released_dot": {"spare_mode": "dot-released"},
    "released_iterative": {"spare_mode": "iterative-released"},
    "released_iterative_nosync": {"spare_mode": "iterative-released-nosync"},
}, doc="SPARe; batched modes are ours (baselines/spare), released modes are SPARe as released",
   # the harness refuses these knobs in the other modes
   mode_knob="spare_mode", mode_only={"spare_max_mib": lambda m: not spare_released(m),
                                      "no_spare_events": lambda m: m in SPARE_STAGED_MODES}))
_add(_harness("pylate", "pylate", ("multivector",), {
    "pylate_mode": Knob("--pylate-mode", str, True, "PyLate mode", PYLATE_MODES),
    "pylate_max_mib": Knob("--pylate-max-mib", int, True, "PyLate memory budget (MiB)"),
    "pylate_rerank_results": Knob("--pylate-rerank-results", str, True,
                                  "where rerank ids are globalised", ("host", "device")),
    "pylate_query_buckets": Knob("--pylate-query-buckets", int, True, "scores-batched query buckets"),
}, {
    # the multivector experiment's PyLate line: scores-batched
    "scores_batched": {"pylate_mode": "scores-batched"},
    "rerank_released": {"pylate_mode": "rerank-released"},
    "rerank_released_ownpad": {"pylate_mode": "rerank-released-ownpad"},
}, doc="PyLate MaxSim; scores-batched is ours, rerank-released is PyLate as released",
   mode_knob="pylate_mode", mode_only={"pylate_rerank_results": pylate_released,
                                       "pylate_query_buckets": lambda m: not pylate_released(m)}))
_add(_harness("cpu", "cpu", ("dense", "sparse", "multivector"), {**SPARSE_PROJECT}, {}, cpu_ok=True,
              doc="the CPU references (faiss.knn / scipy / numpy MaxSim): dev box and tests"))
_add(_harness("masked", "masked", ("filtered",), {}, {}, cpu_ok=True,
              doc="the filtered CPU reference (harness --backend masked)"))

# nova-bf: the config file is the experiment's (systems.<name>.config); knobs only set per-leg
# values (paths, I/O, a few params); everything else comes from the config file.
_add(SystemSpec("novabf", "novabf", None, ("dense", "sparse", "multivector", "filtered"), {
    "config": Knob(None, str, False, "the nova-bf config file (relative to the experiment file)"),
    # set through $NOVABF_MV_BUDGET_MIB (the multivector config: ${NOVABF_MV_BUDGET_MIB:-648.5})
    "mv_budget_mib": Knob(None, float, True, "multivector_budget_mib via $NOVABF_MV_BUDGET_MIB"),
    # merged into the config's params block (the base config has none), e.g. {"io_workers": N}
    "params": Knob(None, dict, False, "params merged into the generated config (e.g. io_workers)"),
}, {"stock": {}}, cpu_ok=True, doc="nova-bf at its own defaults, 1-file warmup"))


# The label summary.md shows for a system. `+proj` marks the query-vocabulary projection
# (nova-bf's own functions, timed reader-side). "(ours: ...)" names every part of the leg that
# is our code, not the library as released: the SPARe batched modes and nosync fix, PyLate's
# scores-batched and ownpad variants, and harness optimizations an entry turns on explicitly
# (union compaction, overlapped copies). Defaults are never tagged.
OURS_KNOB_VALUES = {
    "spare_mode": {"dot": "SPARe batched", "iterative": "SPARe batched",
                   "iterative-released-nosync": "SPARe iterative nosync fix"},
    "pylate_mode": {"scores-batched": "PyLate scores-batched",
                    "rerank-released-ownpad": "PyLate ownpad"},
    "filter_compact": {"union": "union compaction"},
    "h2d_overlap": {True: "overlapped copies"},
    "d2h_overlap": {True: "overlapped copies"},
    "cuvs_sync": {"none": "no per-file sync"},
}


def display_label(preset: str, knobs: dict) -> str:
    """preset [+proj] [(ours: what)], e.g. `stock +proj`, `batched_dot (ours: SPARe batched)`."""
    out = preset or "stock"
    if (knobs or {}).get("sparse_project") == "query-vocab":
        out += " +proj"
    ours = []
    for k, vals in OURS_KNOB_VALUES.items():
        v = (knobs or {}).get(k)
        if not isinstance(v, (str, bool)):
            continue
        w = vals.get(v)
        if w and w not in ours:
            ours.append(w)
    if ours:
        out += f" (ours: {', '.join(ours)})"
    return out


def mode_variants(spec: SystemSpec, knobs: dict) -> list[tuple]:
    """[(mode, knobs)] for each mode of a system with modes (SPARe, PyLate): the mode knob set
    and the knobs that mode does not take dropped. [(None, knobs)] for any other system."""
    if spec.mode_knob is None:
        return [(None, dict(knobs))]
    return [(m, {**{k: v for k, v in knobs.items() if k not in spec.mode_only or spec.mode_only[k](m)},
                 spec.mode_knob: m}) for m in spec.knobs[spec.mode_knob].choices]


class SystemError_(ValueError):
    pass


def resolve(name: str, entry) -> dict:
    """An experiment's `systems.<name>` entry -> {name, system, kind, preset, knobs,
    chosen_from}. Validates the system, preset and every knob (type, choices)."""
    if isinstance(entry, str):
        entry = {"preset": entry}
    if not isinstance(entry, dict):
        raise SystemError_(f"systems.{name}: a preset name or a mapping, not {entry!r}")
    entry = dict(entry)
    sysname = entry.pop("system", name)
    if sysname not in SYSTEMS:
        raise SystemError_(f"systems.{name}: unknown system {sysname!r} (known: {sorted(SYSTEMS)})")
    spec = SYSTEMS[sysname]
    preset = entry.pop("preset", "stock")
    if preset not in spec.presets:
        raise SystemError_(f"systems.{name}: {sysname} has no preset {preset!r} "
                           f"(presets: {sorted(spec.presets)})")
    chosen_from = entry.pop("chosen_from", None)
    # the system's own io (`io:`, e.g. its I/O sweep's winner) over the experiment's `io:`; a
    # sweep variant's grid io (sweep_io, set by bench sweep) over both
    own_io = check_sweep_io(entry.pop("io", None), f"systems.{name}.io")
    io = {**(own_io or {}), **(check_sweep_io(entry.pop("sweep_io", None),
                                              f"systems.{name}.sweep_io") or {})}
    knobs = dict(spec.presets[preset])
    for k, v in entry.items():
        if k not in spec.knobs:
            raise SystemError_(f"systems.{name}: unknown key {k!r} for {sysname} "
                               f"(knobs: {sorted(spec.knobs)})")
        knobs[k] = check_knob(spec, k, v, where=f"systems.{name}.{k}")
    if spec.kind == "novabf" and "config" not in knobs:
        raise SystemError_(f"systems.{name}: nova-bf needs `config:` (its config file)")
    out = {"name": name, "system": sysname, "kind": spec.kind, "preset": preset,
           "knobs": knobs, "chosen_from": chosen_from}
    if io:
        out["io"] = io   # overlaid on the experiment's io for this system's legs (leg.leg_io)
    return out


# the `io:` keys a sweep may grid per variant (`io.<key>`); passed on exactly like `io:`
SWEEP_IO_KEYS = {"files_in_flight": int, "gets": int, "part_mib": int, "read_mode": str,
                 "range_scope": str}
# the nova-bf config variable each io key reaches (bench/leg.py NOVABF_CONFIG_ENV)
SWEEP_IO_NOVABF_VAR = {"files_in_flight": "NOVABF_IO_WORKERS", "gets": "NOVABF_RUST_FETCH_CONC",
                       "part_mib": "NOVABF_RUST_FETCH_PART_MIB"}


def check_sweep_io(io, where):
    if io is None:
        return None
    if not isinstance(io, dict) or not io or set(io) - set(SWEEP_IO_KEYS):
        raise SystemError_(f"{where}: a mapping of {sorted(SWEEP_IO_KEYS)}")
    for k, v in io.items():
        t = SWEEP_IO_KEYS[k]
        if isinstance(v, bool) or not isinstance(v, t) or (t is int and v < 1):
            raise SystemError_(f"{where}.{k}: {'a positive integer' if t is int else 'a string'}, not {v!r}")
    return dict(io)


def check_knob(spec: SystemSpec, k: str, v, where: str):
    kn = spec.knobs[k]
    if kn.type is bool:
        if not isinstance(v, bool):
            raise SystemError_(f"{where}: true/false, not {v!r}")
    elif kn.type is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SystemError_(f"{where}: a number, not {v!r}")
    elif kn.type is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise SystemError_(f"{where}: an integer, not {v!r}")
    elif kn.type is dict:
        if not isinstance(v, dict):
            raise SystemError_(f"{where}: a mapping, not {v!r}")
    elif not isinstance(v, str):
        raise SystemError_(f"{where}: a string, not {v!r}")
    if kn.choices and v not in kn.choices:
        raise SystemError_(f"{where}: {v!r} not in {list(kn.choices)}")
    return v


def harness_knob_args(spec: SystemSpec, knobs: dict, order) -> list[str]:
    """The harness flags for the knobs named in `order` that are set (bools: flag when true)."""
    out = []
    for k in order:
        if k not in knobs or k not in spec.knobs or spec.knobs[k].flag is None:
            continue
        v = knobs[k]
        kn = spec.knobs[k]
        if kn.type is bool:
            if v:
                out.append(kn.flag)
        elif kn.flag_when is not None:
            if v == kn.flag_when:
                out.append(kn.flag)
        else:
            out += [kn.flag, _fmt(v)]
    return out


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
