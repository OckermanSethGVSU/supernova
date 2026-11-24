
import pandas as pd
import pytest

from bench.report import plots as plot
from bench.systems import harness_cli as run
from bench.record import schema
from bench.systems import cuvs_engine



@pytest.mark.slow
def test_plot_from_cli_csv(corpus, tmp_path):
    out = tmp_path / "r.csv"
    for argv in (["--regime", "stream", "--prefetch", "1", "--workers", "1"],
                 ["--regime", "stream", "--prefetch", "0"],
                 ["--regime", "stream", "--prefetch", "2", "--workers", "2"]):
        run.main(argv + ["--corpus", corpus[0], "--backend", "cpu", "--nq", "32",
                         "--repeats", "3", "--csv", str(out)])
    med = plot.median_rows(pd.read_csv(out, keep_default_na=False))
    assert len(med) == 3 and set(med["n_repeats"]) == {3}
    png = tmp_path / "a.png"
    assert plot.attribution([str(out)], str(png)) == 0
    assert png.stat().st_size > 10_000


def test_median_row_is_a_real_repeat():
    rows = []
    for rep, tot in enumerate([3.0, 1.0, 2.0, 9.0]):
        rows.append({c: 0 for c in plot.CONFIG} | {
            "warmup": 0, "repeat": rep, "total_s": tot, "source_stall_s": tot / 2,
            "engine_call_s": tot / 2, "merge_s": 0.0, "other_s": 0.0})
    rows.append(dict(rows[0], warmup=1, total_s=100.0))
    med = plot.median_rows(pd.DataFrame(rows))
    assert len(med) == 1 and med.iloc[0]["total_s"] == 2.0


def _csv_row(**kw):
    r = {c: 0 for c in schema.CSV_FIELDS}
    r.update(regime="stream", backend="gpu", engine="faiss", queries_mode="resident",
             temp_mem="default", rmm_pool="n/a", read_mode_effective="arrow",
             widen_where="device", gpu_events="off", corpus="c", host="h", warmup=0,
             total_s=1.0, engine_call_s=1.0, verified="not_run")
    r.update(kw)
    return r


def test_config_grouping_ignores_per_file_type_inference(tmp_path):
    """One CSV's temp_mem column is all numbers (int), another's mixes in
    "default" (object): 1536 and "1536" must still be one config, one bar."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame([_csv_row(temp_mem="default", repeat=0),
                  _csv_row(temp_mem=1536, repeat=0, total_s=3.0),
                  _csv_row(temp_mem=1536, repeat=1, total_s=4.0)]).to_csv(a, index=False)
    pd.DataFrame([_csv_row(temp_mem=1536, repeat=2, total_s=5.0)]).to_csv(b, index=False)
    med = plot.median_rows(plot.read_csvs([str(a), str(b)]))
    assert sorted(zip(med["temp_mem"], med["n_repeats"])) == [("1536", 3), ("default", 1)]
    assert med["total_s"].dtype.kind == "f"  # buckets stay numeric


def test_failed_verify_rows_are_dropped(tmp_path, capsys):
    p = tmp_path / "v.csv"
    pd.DataFrame([_csv_row(repeat=0, verified="ok"),
                  _csv_row(repeat=1, verified="failed", total_s=9.0),
                  _csv_row(repeat=2, verified="not_run", total_s=2.0)]).to_csv(p, index=False)
    df = plot.read_csvs([str(p)])
    assert sorted(df["verified"]) == ["not_run", "ok"]
    assert "dropping 1 row(s)" in capsys.readouterr().err


def test_provenance_splits_configs(tmp_path):
    """Query source, column, host and the library versions are config: rows
    that differ never share a bar."""
    p = tmp_path / "p.csv"
    pd.DataFrame([
        _csv_row(query_file="q1.parquet"), _csv_row(query_file="q2.parquet"),
        _csv_row(column="other"), _csv_row(host="h2"), _csv_row(pyarrow_version="99"),
        _csv_row(engine_call_s=2.0, pyarrow_version="99", repeat=1),
    ]).to_csv(p, index=False)
    med = plot.median_rows(plot.read_csvs([str(p)]))
    assert len(med) == 5
    for c in ("query_file", "query_column", "column", "host", "seed") + tuple(plot.VERSIONS):
        assert c in plot.CONFIG and c in schema.CSV_FIELDS


def test_every_column_the_plot_needs_is_in_the_csv_schema():
    """read_csvs refuses a CSV that lacks a needed column: every one is a harness CSV field."""
    need = plot.CONFIG + plot.BUCKETS + plot.ANNOTATIONS + ["warmup", "total_s", "verified",
                                                           "h2d_exposed_s", "gpu_idle_s"]
    assert [c for c in need if c not in schema.CSV_FIELDS] == []


def test_mixed_k_goes_on_the_bars(tmp_path):
    rows = [_csv_row(k=10), _csv_row(k=100)]
    med = plot.median_rows(pd.DataFrame(rows).astype(str).assign(
        warmup=0, total_s=1.0))
    assert plot.dk_title(med) == ("mixed d / k (see bars)", True)
    assert plot.dk_title(med.iloc[:1]) == ("d=0, k=10", False)
    assert "k=" not in plot.label(med.iloc[0])
    assert "k=100" in plot.label(med.iloc[1], dk=True)
    p = tmp_path / "k.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    png = tmp_path / "k.png"
    assert plot.attribution([str(p)], str(png)) == 0
    assert png.stat().st_size > 10_000


def test_labels_and_table_show_every_varying_config(corpus, tmp_path):
    """Two runs differing only in --range-concurrency are two bars with two
    different labels; the label shows readers (min(prefetch, workers,
    files)), and the table carries the varying column."""
    out = tmp_path / "r.csv"
    for rc in ("6", "12"):
        run.main(["--regime", "stream", "--corpus", corpus[0], "--backend", "cpu",
                  "--nq", "8", "--warmup", "0", "--repeats", "1", "--csv", str(out),
                  "--read-mode", "native", "--prefetch", "2", "--workers", "4",
                  "--range-concurrency", rc])
    med = plot.numeric_annotations(plot.median_rows(plot.read_csvs([str(out)])))
    assert len(med) == 2
    extra = plot.label_extras(med)
    labels = [plot.label(r, extra=extra) for _, r in med.iterrows()]
    assert len(set(labels)) == 2
    assert all("readers=2" in lb and "faiss/cpu" in lb for lb in labels)
    assert "range_concurrency" in plot.table_columns(med)
    # corpus by its last path component, only when it varies
    assert plot.compact("corpus", "s3://b/rpf10k/?region=x") == "corpus=rpf10k"


def test_csv_without_readers_refused_and_error_rows_kept(tmp_path, capsys):
    """No back-compat: a CSV from before the ``readers`` column is refused
    (re-run it), not patched up by deriving the column."""
    p = tmp_path / "o.csv"
    rows = [_csv_row(prefetch=4, workers=2, n_files=8, verified="error"),
            _csv_row(prefetch=4, workers=8, n_files=3, repeat=1),
            _csv_row(prefetch=0, workers=0, n_files=3, repeat=2)]
    pd.DataFrame(rows).drop(columns=["readers"]).to_csv(p, index=False)
    with pytest.raises(SystemExit, match=r"missing columns \['readers'\]"):
        plot.read_csvs([str(p)])
    pd.DataFrame(rows).to_csv(p, index=False)
    df = plot.read_csvs([str(p)])
    assert len(df) == 3 and "keeping 1 row(s)" in capsys.readouterr().err


def _labels(rows):
    med = pd.DataFrame(rows)
    extra, full = plot.label_extras(med), plot.full_paths(med)
    return [plot.label(r, extra=extra, full=full) for _, r in med.iterrows()]


def test_labels_show_full_paths_when_basenames_collide():
    """corpus by its last component, unless two plotted corpora share it:
    /data/fineweb and s3://bkt/fineweb/ get two different labels."""
    labels = _labels([_csv_row(corpus="/data/fineweb"),
                      _csv_row(corpus="s3://bkt/fineweb/")])
    assert len(set(labels)) == 2
    assert "corpus=/data/fineweb" in labels[0] and "corpus=s3://bkt/fineweb/" in labels[1]
    labels = _labels([_csv_row(corpus="/data/a"), _csv_row(corpus="s3://bkt/b/")])
    assert "corpus=a" in labels[0] and "corpus=b" in labels[1]


def test_labels_show_read_mode_and_row_groups_themselves():
    """read_mode_effective / row_groups_per_file are on every label: bars differing
    in them get different labels, with no key=value repeat."""
    labels = _labels([_csv_row(read_mode_effective="arrow"),
                      _csv_row(read_mode_effective="sequential")])
    assert len(set(labels)) == 2 and "read_mode_effective=" not in labels[1]
    assert "sequential" in labels[1]
    labels = _labels([_csv_row(row_groups_per_file=1), _csv_row(row_groups_per_file=4)])
    assert len(set(labels)) == 2 and "rg=4" in labels[1]
    assert "row_groups_per_file=" not in labels[1]


def test_reader_pool_splits_bars_and_labels(tmp_path):
    """split / full reader pools are config: never one bar, and each bar's
    label names its reader_pool when it varies. The per-reader pool sizes
    are the same here (--cpu-thread-count 1 over 2 readers: 2x1 either way),
    so only reader_pool tells the bars apart; two repeats each, so the
    grouping must also pool repeats within a reader_pool."""
    p = tmp_path / "rp.csv"
    rows = [_csv_row(reader="process", reader_pool=pool,
                     reader_pa_cpu_threads="2x1", repeat=rep, total_s=t + rep)
            for pool, t in (("split", 1.0), ("full", 3.0)) for rep in (0, 1)]
    pd.DataFrame(rows).to_csv(p, index=False)
    med = plot.median_rows(plot.read_csvs([str(p)]))
    assert len(med) == 2  # bars: one per reader_pool, both repeats in each
    assert list(med["n_repeats"]) == [2, 2]
    assert dict(zip(med["reader_pool"], med["total_s"])) == {"split": 1.0, "full": 3.0}
    assert "reader_pool" in plot.CONFIG and "reader_pool" in schema.CSV_FIELDS
    assert "reader_pool" in plot.table_columns(med)
    extra, full = plot.label_extras(med), plot.full_paths(med)
    labels = {r["reader_pool"]: plot.label(r, extra=extra, full=full)
              for _, r in med.iterrows()}
    assert len(set(labels.values())) == 2
    assert "reader_pool=split" in labels["split"] and "reader_pool=full" in labels["full"]
    assert "reader_pool=full" not in labels["split"]


def test_csv_without_range_mib_is_refused(tmp_path):
    """No column is filled in for an older CSV: one without range_mib is refused."""
    old = tmp_path / "old.csv"
    pd.DataFrame([_csv_row()]).drop(columns=["range_mib"]).to_csv(old, index=False)
    with pytest.raises(SystemExit, match=r"missing columns \['range_mib'\]"):
        plot.read_csvs([str(old)])



