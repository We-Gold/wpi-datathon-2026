import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from health.cli import OUTPUT_DIR

    return OUTPUT_DIR, Path, alt, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Presentation visuals

    Figures for the slides. Each section builds one chart from what
    `health-prep parse` last wrote to `data/processed/`.
    """)
    return


@app.cell
def _(OUTPUT_DIR, Path, pl):
    # Notebooks get launched from wherever, so anchor on the repo root.
    ROOT = Path(__file__).resolve().parent.parent
    everything = pl.scan_parquet(sorted((ROOT / OUTPUT_DIR).glob("*.parquet")))
    return ROOT, everything


@app.cell
def data_overview(everything, mo, pl):
    # One row per metric in the Kaggle JSON file, so the slide can show how little
    # it covers before the richness chart makes the comparison.
    kaggle_json_overview = (
        everything.filter(pl.col("source_format") == "nested_json")
        .group_by("metric", "unit")
        .agg(
            pl.len().alias("records"),
            pl.col("start_local").dt.date().n_unique().alias("days_with_data"),
            pl.col("start_local").min().dt.date().alias("first_day"),
            pl.col("start_local").max().dt.date().alias("last_day"),
            pl.col("value_num").median().round(2).alias("median_value"),
        )
        .with_columns(
            pl.col("metric").cast(pl.String).str.replace(r"^HK\w+?Identifier", "").alias("metric")
        )
        .sort("records", descending=True)
        .collect()
    )

    mo.vstack(
        [
            mo.md("## Data overview: Kaggle JSON"),
            mo.ui.table(kaggle_json_overview, selection=None, show_download=False),
        ]
    )
    return (kaggle_json_overview,)


@app.cell
def kaggle_json_table(alt, kaggle_json_overview):
    # The same overview drawn as an Altair chart, so it exports as a sharp PNG like
    # the other figures. Values are formatted here because text marks show them as is.
    _headers = {
        "metric": "Metric",
        "unit": "Unit",
        "records": "Records",
        "days_with_data": "Days with data",
        "first_day": "First day",
        "last_day": "Last day",
        "median_value": "Median",
    }
    _cells = [
        {
            "row": _i,
            "column": _headers[_name],
            "text": f"{_value:,}" if isinstance(_value, int) else f"{_value:,.2f}".rstrip("0").rstrip(".")
            if isinstance(_value, float)
            else str(_value),
        }
        for _i, _row in enumerate(kaggle_json_overview.iter_rows(named=True))
        for _name, _value in _row.items()
    ]

    _x = alt.X(
        "column:N",
        sort=list(_headers.values()),
        title=None,
        axis=alt.Axis(orient="top", labelAngle=0, labelFontWeight="bold", ticks=False, domain=False),
    )
    _y = alt.Y("row:O", axis=None)
    _base = alt.Chart(alt.Data(values=_cells))

    _stripes = _base.mark_rect(color="#f2f2f2").encode(x=_x, y=_y).transform_filter("datum.row % 2 == 0")
    _text = _base.mark_text(fontSize=14).encode(x=_x, y=_y, text="text:N")

    kaggle_json_table_chart = (
        (_stripes + _text)
        .properties(
            width=880,
            height=36 * kaggle_json_overview.height,
            title=alt.Title(
                "Kaggle JSON: six metrics in total",
                subtitle="One row per metric in data/other/health_data.json",
            ),
        )
        .configure_axis(labelFontSize=14, grid=False)
        .configure_title(fontSize=20, subtitleFontSize=13, anchor="start")
        .configure_view(strokeWidth=0)
    )
    kaggle_json_table_chart
    return (kaggle_json_table_chart,)


@app.cell
def _(mo):
    mo.md("""
    ## Why we supplemented the Kaggle datasets

    The two Kaggle files in `data/other/` track far fewer signals than PMData or
    our own Apple exports. A metric only counts as tracked here if it has data on
    at least `MIN_DAYS` separate days, so a single height or BMI reading does not
    make a dataset look richer than it is.
    """)
    return


@app.cell
def _(everything, pl):
    MIN_DAYS = 30

    # One readable name per dataset. The 16 PMData subjects share one label so they
    # collapse to a single bar below.
    DATASET_LABELS = {
        "subject_01": "Our Apple export 1",
        "subject_02": "Our Apple export 2",
        "subject_03": "Kaggle CSV",
        "subject_04": "Kaggle JSON",
    }
    KAGGLE = {"Kaggle CSV", "Kaggle JSON"}

    metric_days = (
        everything.group_by("subject_id", "metric")
        .agg(pl.col("start_local").dt.date().n_unique().alias("days"))
        .collect()
    )

    per_subject = (
        metric_days.group_by("subject_id")
        .agg(
            pl.len().alias("any_data"),
            (pl.col("days") >= MIN_DAYS).sum().alias("tracked"),
        )
        .with_columns(
            pl.col("subject_id")
            .cast(pl.String)
            .replace_strict(DATASET_LABELS, default="PMData (median of 16)")
            .alias("dataset")
        )
    )

    richness = (
        per_subject.group_by("dataset")
        .agg(
            pl.col("tracked").median(),
            pl.col("any_data").median(),
            pl.col("tracked").min().alias("tracked_min"),
            pl.col("tracked").max().alias("tracked_max"),
        )
        .with_columns(pl.col("dataset").is_in(KAGGLE).alias("is_kaggle"))
        .sort("tracked", descending=True)
    )
    richness
    return MIN_DAYS, richness


@app.cell
def _(MIN_DAYS, alt, richness):
    _order = richness["dataset"].to_list()
    _y = alt.Y("dataset:N", sort=_order, title=None)

    _any = (
        alt.Chart(richness)
        .mark_bar(color="#d9d9d9", height=26)
        .encode(
            x=alt.X("any_data:Q", title="Distinct health metrics"),
            y=_y,
            tooltip=["dataset", alt.Tooltip("any_data:Q", title="metrics with any data")],
        )
    )

    _tracked = (
        alt.Chart(richness)
        .mark_bar(height=26)
        .encode(
            x="tracked:Q",
            y=_y,
            color=alt.condition(
                alt.datum.is_kaggle, alt.value("#C31230"), alt.value("#43A49B")
            ),
            tooltip=[
                "dataset",
                alt.Tooltip("tracked:Q", title=f"metrics on {MIN_DAYS}+ days"),
            ],
        )
    )

    _labels = _tracked.mark_text(align="left", dx=6, fontSize=15, fontWeight="bold").encode(
        text="tracked:Q", color=alt.value("#222")
    )

    richness_chart = (
        (_any + _tracked + _labels)
        .properties(
            width=620,
            height=280,
            title=alt.Title(
                "The Kaggle datasets track far fewer signals",
                subtitle=f"Colored bar: metrics recorded on {MIN_DAYS}+ days. Grey bar: metrics with any data.",
            ),
        )
        .configure_axis(labelFontSize=14, titleFontSize=14, grid=False)
        .configure_title(fontSize=20, subtitleFontSize=13, anchor="start")
        .configure_view(strokeWidth=0)
    )
    richness_chart
    return (richness_chart,)


@app.cell
def export_figures(ROOT, kaggle_json_table_chart, mo, richness_chart):
    # Writes every slide figure as a PNG. scale_factor multiplies the pixel size,
    # so 4 turns the 880 px wide table into about 3500 px.
    FIGURES = ROOT / "figures"
    FIGURES.mkdir(exist_ok=True)

    _exports = {
        "kaggle_json_overview.png": kaggle_json_table_chart,
        "dataset_richness.png": richness_chart,
    }
    for _name, _chart in _exports.items():
        _chart.save(FIGURES / _name, scale_factor=4)

    mo.md("Saved " + ", ".join(f"`figures/{_name}`" for _name in _exports))
    return


if __name__ == "__main__":
    app.run()
