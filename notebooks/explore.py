import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from health import schema, validate
    from health.cli import OUTPUT_DIR

    return OUTPUT_DIR, Path, alt, mo, pl, schema, validate


@app.cell
def _(mo):
    mo.md("""
    # Scratch pad

    Reads whatever `health-prep parse` last wrote to `data/processed/`.
    Nothing here is part of the pipeline, so break it freely.

    Run `uv run health-prep parse` first if the tables below are empty.
    """)
    return


@app.cell
def _(OUTPUT_DIR, Path, pl, schema):
    # Notebooks get launched from wherever, so anchor on the repo root rather
    # than the working directory.
    ROOT = Path(__file__).resolve().parent.parent
    PROCESSED = ROOT / OUTPUT_DIR

    parquet_files = sorted(PROCESSED.glob("*.parquet"))
    # One lazy frame over every output. Scanning keeps the six million rows on
    # disk until a cell actually asks for something.
    everything = pl.scan_parquet(parquet_files) if parquet_files else schema.empty_frame().lazy()
    return everything, parquet_files


@app.cell
def _(mo, parquet_files, pl):
    file_table = pl.DataFrame(
        {
            "file": [p.name for p in parquet_files],
            "rows": [pl.scan_parquet(p).select(pl.len()).collect().item() for p in parquet_files],
            "mb": [round(p.stat().st_size / 1e6, 1) for p in parquet_files],
        }
    )
    mo.ui.table(file_table, selection=None)
    return


@app.cell
def _(everything, mo):
    subjects = everything.select("subject_id").unique().collect().to_series().sort().to_list()
    all_metrics = everything.select("metric").unique().collect().to_series().sort().to_list()

    subject_pick = mo.ui.dropdown(
        subjects, value=subjects[0] if subjects else None, label="subject"
    )
    metric_pick = mo.ui.dropdown(
        all_metrics,
        value="HKQuantityTypeIdentifierStepCount"
        if "HKQuantityTypeIdentifierStepCount" in all_metrics
        else (all_metrics[0] if all_metrics else None),
        label="metric",
    )
    agg_pick = mo.ui.dropdown(
        {"daily total": "total", "daily mean": "mean"},
        value="daily total",
        label="aggregation",
    )
    source_types = everything.select("source_type").unique().collect().to_series().sort().to_list()
    source_pick = mo.ui.multiselect(source_types, value=source_types, label="source types")

    mo.vstack(
        [
            mo.hstack([subject_pick, metric_pick, agg_pick], justify="start", gap=2),
            source_pick,
        ],
        gap=1,
    )
    return agg_pick, metric_pick, source_pick, subject_pick


@app.cell
def _(everything, metric_pick, pl, subject_pick):
    picked = everything.filter(
        (pl.col("subject_id") == subject_pick.value) & (pl.col("metric") == metric_pick.value)
    )
    return (picked,)


@app.cell
def _(picked, pl):
    # Which source types carry this metric. Two of these usually means the
    # daily total is double counted, a watch and a phone both logging steps.
    by_source = (
        picked.group_by("source_type")
        .agg(
            pl.len().alias("rows"),
            pl.col("start_local").dt.date().n_unique().alias("days"),
            pl.col("value_num").sum().alias("total"),
        )
        .sort("rows", descending=True)
        .collect()
    )
    by_source
    return


@app.cell
def _(picked, pl, source_pick):
    selection = picked.filter(pl.col("source_type").is_in(source_pick.value))
    return (selection,)


@app.cell
def _(mo, selection):
    mo.ui.table(selection.head(200).collect(), selection=None)
    return


@app.cell
def _(pl, selection):
    # Daily buckets rather than raw rows: altair chokes well before six million
    # points, and the shape of a day is what we actually look at. Both
    # aggregates come out of one pass, so flipping the dropdown does not
    # rescan the parquet.
    daily = (
        selection.filter(pl.col("value_num").is_not_null())
        .group_by(pl.col("start_local").dt.date().alias("day"))
        .agg(
            pl.col("value_num").sum().alias("total"),
            pl.col("value_num").mean().alias("mean"),
            pl.len().alias("rows"),
        )
        .sort("day")
        .collect()
    )
    return (daily,)


@app.cell
def _(agg_pick, alt, daily, metric_pick, mo):
    chart = (
        alt.Chart(daily)
        .mark_line()
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y(f"{agg_pick.value}:Q", title=f"{agg_pick.selected_key} {metric_pick.value}"),
            tooltip=["day:T", "total:Q", "mean:Q", "rows:Q"],
        )
        .properties(height=220)
    )
    mo.ui.altair_chart(chart) if daily.height else mo.md("No numeric values for this pick.")
    return


@app.cell
def _(everything, mo, pl):
    mo.md("## What each subject actually has")

    coverage = (
        everything.group_by("subject_id", "source_type")
        .agg(
            pl.len().alias("rows"),
            pl.col("metric").n_unique().alias("metrics"),
            pl.col("start_local").dt.date().n_unique().alias("days"),
        )
        .sort("subject_id", "rows", descending=[False, True])
        .collect()
    )
    coverage
    return


@app.cell
def _(everything, validate):
    # The same gate the CLI runs. Handy after editing a reader: check the change
    # here before rerunning a full parse.
    findings = validate.check(everything.head(200_000).collect())
    findings or "clean"
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
