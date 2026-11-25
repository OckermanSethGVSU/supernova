"""bench/experiment.py + bench/systems/base.py: the experiment schema."""

import copy
import glob
import os
import subprocess
import sys

import pytest

from bench import experiment as E
from bench import runner
from bench.systems import base as SB
from helpers.experiments import _exp

CB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(CB, "tests", "fixtures", "experiments")
FIX = os.path.join(CB, "tests", "fixtures", "experiments")

MIN = {"name": "x", "modality": "dense", "corpus": {"path": "c", "column": "v"},
       "queries": {"path": "q", "column": "v"}, "search": {"k": 10, "metric": "ip"}, "threads": 4,
       "systems": {"cpu": "stock"}}


def _v(**over):
    d = copy.deepcopy(MIN)
    for k, v in over.items():
        if v is None:
            d.pop(k)
        else:
            d[k] = v
    return E.validate(d)


@pytest.mark.parametrize("path", E.experiment_files())
def test_every_experiment_file_validates(path):
    e = E.load(path)
    assert e["searches"] and e["systems"]
    for sy in e["systems"].values():
        if sy["kind"] == "novabf":
            assert os.path.isfile(sy["knobs"]["config"])


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(EXP, "sweeps", "*.yaml"))))
def test_every_sweep_file_expands(path):
    e, variants, _ = E.load_sweep(path)
    assert variants and e["subset"]["files"] >= 1
    for s in e["searches"]:   # a variant with `searches:` runs only in those
        assert [v["variant"] for v in variants
                if v.get("searches") is None or s["name"] in v["searches"]] == s["systems"]


def test_the_experiments_exist():
    names = {E.load_raw(p)["name"] for p in E.experiment_files()}
    assert names == {"dense_coyo", "dense_pes2o", "sparse_fineweb", "sparse_pubmed",
                     "multivector_pubmed",
                     "dense_pubmed", "filtered_structured", "filtered_text"}


def test_fineweb_filtered_io_defaults_are_novabf_own():
    """with the env unset, the params block expands to nova-bf's own config defaults."""
    import yaml
    pytest.importorskip("nova_bf")
    from nova_bf.config import ParamsConfig   # the defaults' home
    from bench.data.filters.search import _expand_env
    c = yaml.safe_load(_expand_env(open(os.path.join(EXP, "novabf", "fineweb_filtered_k1000.yaml")).read()))
    fields = ParamsConfig.model_fields
    for k in ("io_workers", "rust_fetch_concurrency", "rust_fetch_part_mib"):
        assert int(c["params"][k]) == fields[k].default, k
    assert "id_column" not in c["corpus"] and c["queries"]["id_column"] == "query_id"


@pytest.mark.parametrize("over,msg", [
    ({"bogus": 1}, "unknown top-level"),
    ({"corpus": {"path": "c", "column": "v", "colum": 1}}, "unknown key"),
    ({"corpus": {"path": "c"}}, "missing"),
    ({"modality": "graph"}, "modality"),
    ({"search": {"k": 0, "metric": "ip"}}, "search.k"),
    ({"search": {"k": 10, "metric": "dot"}}, "metric"),
    ({"systems": {"cpu": {"system": "cpu", "spare_mode": "dot"}}}, "unknown key 'spare_mode'"),
    ({"systems": {"x": {"system": "nope"}}}, "unknown system"),
    ({"systems": {"cpu": "fast"}}, "no preset"),
    ({"systems": {"spare": "batched_dot"}}, "does not run dense"),
    ({"systems": {"cpu": {"system": "cpu", "h2d": "zero"}}}, "not in"),
    ({"systems": {"cpu": {"system": "cpu", "h2d_overlap": "yes"}}}, "true/false"),
    ({"systems": {"novabf": {"system": "novabf"}}}, "needs `config:`"),
    ({"systems": {"a__b": "stock"}}, "bad name"),
    ({"protocol": {"rounds": 0}}, "protocol.rounds"),
    ({"protocol": {"rounds": 3, "warmup": 2}}, "unknown key"),   # the warmup is not a setting
    ({"io": {"files_in_flight": 4, "gets": 512}}, "missing"),
    ({"filter": {"config": "f.yaml"}}, "filtered only"),
])
def test_bad_experiments_are_refused(over, msg):
    with pytest.raises(E.ExperimentError, match=msg):
        _v(**over)


def test_defaults_and_single_search():
    e = _v(search_name="coyo")
    assert e["searches"] == [{"name": "coyo", "systems": ["cpu"], "filter_search": None}]
    assert e["protocol"]["rounds"] == 3 and e["io"] is None and e["queries"]["nq"] is None
    assert e["systems"]["cpu"] == {"name": "cpu", "system": "cpu", "kind": "harness",
                                   "preset": "stock", "knobs": {}, "chosen_from": None}


def test_filtered_needs_searches_and_systems_must_be_used():
    with pytest.raises(E.ExperimentError, match="lists its `searches`"):
        _v(modality="filtered", filter={"config": "f"}, systems={"masked": "stock"})
    with pytest.raises(E.ExperimentError, match="in no search"):
        _v(systems={"cpu": "stock", "cpu2": {"system": "cpu"}},
           searches=[{"name": "a", "systems": ["cpu"]}])


def test_preset_then_entry_knobs():
    r = SB.resolve("cuvs_pool", {"system": "cuvs", "preset": "rmm_pool_8g", "filter_threads": 8})
    assert r["knobs"] == {"cuvs_rmm_pool": "8GiB", "filter_threads": 8}


def test_placeholders_are_found():
    e = E.load(os.path.join(CB, E.path_of("sparse_fineweb")))
    assert E.placeholders(e) == ["queries.nq = 'TODO'"]   # its corpus is ${READ}/corpus/ now
    assert E.placeholders(E.load(os.path.join(CB, E.path_of("dense_coyo")))) == []


def test_base_inheritance_and_sweep_expansion():
    e, variants, raw = E.load_sweep(os.path.join(FIX, "sweep_sparse.yaml"))
    assert e["name"] == "s4_sweep_sparse" and e["subset"] == {"files": 2}
    assert e["protocol"]["rounds"] == 1 and e["corpus"]["column"] == "sparse_embedding"
    assert [v["variant"] for v in variants] == ["novabf", "cpu-sparse_project-off",
                                                 "cpu-sparse_project-query-vocab"]
    assert e["systems"]["cpu-sparse_project-query-vocab"]["knobs"] == {"sparse_project": "query-vocab"}
    assert e["_base"] == [os.path.join(FIX, "sparse.yaml")]


def test_sweep_refuses_non_sweepable_and_unknown_knobs(tmp_path):
    base = os.path.join(FIX, "dense.yaml")
    for grid, msg in [({"h2d": ["pinned"]}, "not sweepable"), ({"nope": [1]}, "no knob"),
                      ({"sparse_project": []}, "non-empty list")]:
        p = tmp_path / "s.yaml"
        import yaml
        p.write_text(yaml.safe_dump({"base": base, "sweep": {"system": "cpu", "grid": grid}}))
        with pytest.raises(E.ExperimentError, match=msg):
            E.load_sweep(str(p))
    p.write_text(yaml.safe_dump({"base": base, "sweep": {"system": "cpu", "grid": {"x": [1]}}}))
    with pytest.raises(E.ExperimentError, match="a sweep file"):
        E.load(str(p))


def test_threads_is_required_and_not_negative():
    with pytest.raises(E.ExperimentError, match="threads: required"):
        _v(threads=None)
    with pytest.raises(E.ExperimentError, match="an integer >= 0"):
        _v(threads=-1)
    assert _v(threads=0)["threads"] == 0


def test_threads_0_means_the_usable_cpu_count_for_every_system():
    from bench import timing
    from bench.data.sources import usable_cpu_count
    n = usable_cpu_count()
    assert timing.omp_threads({"threads": 0}) == n
    assert timing.leg_env({"threads": 0})["OMP_NUM_THREADS"] == str(n)
    assert timing.omp_threads({"threads": 32}) == 32


# ---- copies: pinned + overlapped both ways by default, GPU harness systems only -------------
COPY_FLAGS = ("--h2d", "--h2d-overlap", "--d2h", "--d2h-overlap")


def _copy_args(e, name, tmp_path):
    from bench import leg as L
    a = L.harness_args(e, e["searches"][0], e["systems"][name], str(tmp_path), "t")
    out = []
    for i, x in enumerate(a):
        if x in COPY_FLAGS:
            out += [x] + ([a[i + 1]] if x in ("--h2d", "--d2h") else [])
    return out


def test_copies_default_is_pinned_and_overlapped_for_gpu_systems(tmp_path):
    e = _v(systems={"faiss": "stock", "cpu": "stock"})
    assert e["copies"] == E.COPIES_DEFAULT == {"h2d": "pinned", "h2d_overlap": True,
                                               "d2h": "pinned", "d2h_overlap": True}
    assert _copy_args(e, "faiss", tmp_path) == ["--h2d", "pinned", "--h2d-overlap",
                                                "--d2h", "pinned", "--d2h-overlap"]
    assert _copy_args(e, "cpu", tmp_path) == []   # a CPU reference: no copies at all


def test_copies_file_block_and_variant_knobs_override_the_default(tmp_path):
    e = _v(copies={"d2h_overlap": False},
           systems={"faiss": "stock", "pageable": {"system": "faiss", "h2d": "pageable",
                                                   "h2d_overlap": False}})
    assert e["copies"] == {**E.COPIES_DEFAULT, "d2h_overlap": False}
    # a false bool emits no flag; h2d: pageable is passed as such
    assert _copy_args(e, "faiss", tmp_path) == ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned"]
    assert _copy_args(e, "pageable", tmp_path) == ["--h2d", "pageable", "--d2h", "pinned"]


@pytest.mark.parametrize("systems,copies,msg", [
    ({"g": {"system": "faiss", "h2d": "pageable"}}, None, "h2d_overlap: true needs h2d: pinned"),
    ({"faiss": "stock"}, {"d2h": "pageable"}, "d2h_overlap: true needs d2h: pinned"),
    ({"c": {"system": "cpu", "d2h": "pinned"}}, None, "CPU reference has no host<->GPU copies"),
    ({"c": {"system": "cpu", "h2d_overlap": True}}, None, "CPU reference"),
    ({"faiss": "stock"}, {"h2d": "fast"}, "not in"),
    ({"faiss": "stock"}, {"h2d_overlap": "yes"}, "true/false"),
])
def test_copies_that_the_harness_would_refuse_are_refused(systems, copies, msg):
    with pytest.raises(E.ExperimentError, match=msg):
        _v(systems=systems, **({"copies": copies} if copies is not None else {}))


def test_a_cpu_reference_may_say_pageable(tmp_path):
    e = _v(systems={"c": {"system": "cpu", "h2d": "pageable", "h2d_overlap": False}})
    assert _copy_args(e, "c", tmp_path) == ["--h2d", "pageable"]


def test_every_gpu_harness_leg_of_every_experiment_copies_pinned_and_overlapped(tmp_path):
    from bench import leg as L
    for path in E.experiment_files():
        e = E.load(path)
        for name, sy in e["systems"].items():
            if sy["kind"] != "harness":
                continue
            got = _copy_args(e, name, tmp_path)
            if SB.SYSTEMS[sy["system"]].backend in SB.HOST_BACKENDS:
                assert got == [], (path, name)
            else:
                assert got == ["--h2d", "pinned", "--h2d-overlap", "--d2h", "pinned",
                               "--d2h-overlap"], (path, name)
    assert L.HOST_BACKENDS == SB.HOST_BACKENDS


def test_host_backends_match_the_harness_schema():
    from bench.record import schema
    assert SB.HOST_BACKENDS == schema.HOST_BACKENDS


@pytest.mark.parametrize("sets,where", [
    (["systems.cuvs.cuvs_rmm_pool=TODO"], "systems.cuvs.knobs.cuvs_rmm_pool"),
    (["novabf_output=${WRITE}/TODO"], "novabf_output"),
    (["checks.gt=TODO.parquet"], "checks.gt"),
    (["prep.queries_src=s3://TODO/q.parquet"], "prep.queries_src"),
])
def test_todo_placeholders_anywhere_are_found(sets, where):
    """not only in corpus / queries / search / io."""
    ph = E.placeholders(_exp("dense_coyo", sets))
    assert any(p.startswith(where) for p in ph), ph


def test_prep_schema_is_validated():
    with pytest.raises(E.ExperimentError, match="prep"):
        _exp("dense_coyo", ["prep.bogus=1"])
    with pytest.raises(E.ExperimentError, match="stage"):
        _exp("dense_pes2o", ["prep.stage=[{key: K, path: p, source: s, how: copy, rows: 3}]"])


def test_a_sweep_entry_searches_must_name_base_searches(tmp_path):
    from bench import experiment as E
    p = tmp_path / "s.yaml"
    p.write_text(f"name: s\nbase: {os.path.join(CB, E.path_of('filtered_structured'))}\n"
                 "io: {files_in_flight: 4, gets: 512, part_mib: 16}\n"
                 "sweep:\n  - {system: faiss_gather, searches: [nope], grid: {io.files_in_flight: [2]}}\n")
    with pytest.raises(E.ExperimentError, match="searches"):
        E.load_sweep(str(p))


def test_exact_scan_is_a_validated_bool_in_the_schema():
    from bench import experiment as E
    raw = E.load_raw(os.path.join(CB, "tests", "fixtures", "experiments", "dense.yaml"))
    e = E.finish(runner._set(raw, ["checks.exact_scan=true"]), "x.yaml")
    assert e["checks"]["exact_scan"] is True
    with pytest.raises(E.ExperimentError, match="exact_scan"):
        E.finish(runner._set(raw, ["checks.exact_scan=7"]), "x.yaml")


@pytest.mark.slow
def test_io_grid_refusals(tmp_path):
    bad = tmp_path / "b.yaml"
    bad.write_text(f"name: b\nbase: {os.path.join(FIX, 'sparse.yaml')}\nsweep:\n  - system: novabf\n"
                   "    grid: {io.read_mode: [arrow]}\nsubset: {files: 2}\n")
    with pytest.raises(E.ExperimentError, match="does not reach nova-bf"):
        E.load_sweep(str(bad))
    bad.write_text(f"name: b\nbase: {os.path.join(FIX, 'sparse.yaml')}\nsweep:\n  - system: cpu\n"
                   "    grid: {io.prefetch: [2]}\nsubset: {files: 2}\n")
    with pytest.raises(E.ExperimentError, match="io keys are"):
        E.load_sweep(str(bad))
    # sweep_io is bench sweep's own key; a run sets a system's io as systems.<name>.io
    p = subprocess.run([sys.executable, "-m", "bench", "run", os.path.join(FIX, "sparse.yaml"), "--dev",
                        "--set", "systems.cpu.sweep_io={files_in_flight: 3}", "--results", str(tmp_path)],
                       cwd=CB, capture_output=True, text=True)
    assert p.returncode == 2 and "set by `bench sweep` only" in p.stderr


def test_a_systems_own_io_overrides_the_experiment_io(tmp_path):
    """systems.<name>.io (an I/O sweep's winner) replaces the experiment's io key by key, for that
    system's legs only; it reaches the harness and nova-bf."""
    from bench import leg as L
    path = os.path.join(FIX, "dense.yaml")
    raw = E.load_raw(path)
    raw["systems"]["cpu"] = {"system": "cpu", "io": {"files_in_flight": 7}}
    raw["systems"]["novabf"]["io"] = {"gets": 128, "part_mib": 8}
    e = E.finish(raw, path)
    assert L.leg_io(e, e["systems"]["cpu"])["files_in_flight"] == 7
    assert L.leg_io(e, e["systems"]["cpu"])["gets"] == 64                 # the experiment's
    nb = L.leg_io(e, e["systems"]["novabf"])
    assert (nb["files_in_flight"], nb["gets"], nb["part_mib"]) == (2, 128, 8)
    cpu = L.plan(e, "dense", "cpu", 1, str(tmp_path))
    assert cpu["argv"][cpu["argv"].index("--prefetch") + 1] == "7"
    nbp = L.plan(e, "dense", "novabf", 1, str(tmp_path))
    assert (nbp["env_add"]["NOVABF_RUST_FETCH_CONC"], nbp["env_add"]["NOVABF_RUST_FETCH_PART_MIB"],
            nbp["env_add"]["NOVABF_IO_WORKERS"]) == ("128", "8", "2")
    raw["systems"]["cpu"]["io"] = {"prefetch": 3}
    with pytest.raises(Exception, match="systems.cpu.io"):
        E.finish(raw, path)


def test_an_io_grid_covers_an_io_todo():
    e = {"io": {"files_in_flight": "TODO", "gets": 4, "part_mib": 1},
         "systems": {"a": {"io": {"files_in_flight": 2}}, "b": {"io": {"files_in_flight": 4}}}}
    assert E.placeholders(e) == []
    e["systems"]["b"] = {}
    assert E.placeholders(e) == ["io.files_in_flight = 'TODO'"]


def test_bench_experiment_get_reads_the_base_chain(tmp_path):
    sw = tmp_path / "sw.yaml"
    sw.write_text(f"name: sw\nbase: {os.path.join(CB, E.path_of('filtered_structured'))}\n"
                  "sweep: {system: novabf}\n")
    p = subprocess.run([sys.executable, "-m", "bench.experiment", str(sw), "--get", "modality"],
                       cwd=CB, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == "filtered"
    p = subprocess.run([sys.executable, "-m", "bench.experiment", str(tmp_path / "nope.yaml"), "--get",
                        "modality"], cwd=CB, capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout == ""


def _sweep(tmp_path, systems, grid="{io.files_in_flight: [2, 4, 8]}"):
    p = tmp_path / "s.yaml"
    p.write_text(f"name: s\nbase: {CB}/tests/fixtures/experiments/sparse/fineweb.yaml\n"
                 f"systems:\n  cuvs: {systems}\n"
                 f"sweep:\n  - system: cuvs\n    grid: {grid}\n"
                 "subset: {files: 8}\n")
    return str(p)


@pytest.mark.parametrize("knobs,want,not_want", [
    # p_io_prefetch.py / io_prefetch.yaml
    ("prefetch: 2", ["partly shadowed", "--prefetch is fixed at 2", "only --workers would vary",
                     "set both"], ["all run the same value"]),
    ("workers: 3", ["partly shadowed", "--workers is fixed at 3", "only --prefetch would vary"],
     ["all run the same value"]),
    ("prefetch: 2, workers: 3", ["shadowed by", "knob prefetch=2", "knob workers=3",
                                 "all run the same value"], ["partly"]),
])
def test_p_io_prefetch_the_refusal_says_how_much_is_shadowed(tmp_path, knobs, want, not_want):
    from bench import experiment as E
    with pytest.raises(E.ExperimentError) as ei:
        E.load_sweep(_sweep(tmp_path, "{system: cuvs, preset: stock, " + knobs + "}"))
    msg = str(ei.value)
    for w in want:
        assert w in msg, msg
    for w in not_want:
        assert w not in msg, msg


def test_a_grid_swept_flag_half_shadows_too(tmp_path):
    from bench import experiment as E
    with pytest.raises(E.ExperimentError, match="partly shadowed .*only --prefetch would vary"):
        E.load_sweep(_sweep(tmp_path, "{system: cuvs, preset: stock}",
                            "{io.files_in_flight: [2, 8], workers: [2, 4]}"))


def test_an_unpinned_files_in_flight_sweep_prepares(tmp_path):
    from bench import experiment as E
    _, v, _ = E.load_sweep(_sweep(tmp_path, "{system: cuvs, preset: stock}"))
    assert len(v) == 3
