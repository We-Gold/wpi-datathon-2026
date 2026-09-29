import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import textwrap
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from health.cli import OUTPUT_DIR

    return Counter, OUTPUT_DIR, Path, alt, mo, pl, textwrap


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
    return MIN_DAYS, per_subject, richness


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
def _(mo):
    mo.md("""
    ## Comparing the three data sources

    One card per source: where it comes from, a few numbers, and what it is good
    and bad at. The numbers come from the processed data. The text lives in
    `SOURCE_CARDS` in the chart cell, so edit the wording there.
    """)
    return


@app.cell
def source_stats(everything, per_subject, pl):
    SOURCE_GROUPS = {
        "flat_csv": "Kaggle",
        "nested_json": "Kaggle",
        "apple_xml": "Personal",
        "pmdata": "PMData",
    }

    # One row per person, joined to the metric counts from the richness section so
    # both figures agree on what "tracked" means.
    source_stats = (
        everything.group_by("subject_id", "source_format")
        .agg(
            pl.len().alias("records"),
            (pl.col("start_local").max() - pl.col("start_local").min())
            .dt.total_days()
            .alias("days_covered"),
        )
        .collect()
        .join(per_subject.select("subject_id", "tracked"), on="subject_id")
        .with_columns(
            pl.col("source_format").cast(pl.String).replace_strict(SOURCE_GROUPS).alias("source")
        )
        .sort("source", "subject_id")
    )
    source_stats
    return (source_stats,)


@app.cell
def source_comparison(Counter, alt, pl, source_stats, textwrap):
    SOURCE_CARDS = {
        "Kaggle": {
            "color": "#d1495b",
            "role": "The public data we started from",
            "context": (
                "Two Apple Health exports that anonymous users shared on Kaggle. "
                "One is a flat CSV and the other a nested JSON file, each from a "
                "different person."
            ),
            "device": "iPhone and wearables",
            "strength": "Real exports from outside our team",
            "limit": "Few signals, so little to learn from",
        },
        "Personal": {
            "color": "#30638e",
            "role": "Our own Apple Health exports",
            "context": (
                "Full Apple Health exports from two team members, covering phone, "
                "watch and scale data. We exported them as XML for this project."
            ),
            "device": "iPhone, watch and scale",
            "strength": "The richest and longest records",
            "limit": "Only two people",
        },
        "PMData": {
            "color": "#00798c",
            "role": "A research cohort with self-reports",
            "context": (
                "A public sports logging dataset from Simula Research Laboratory "
                "in Norway (ACM MMSys 2020). 16 people wore a Fitbit from Nov 2019 "
                "to Mar 2020 and logged mood, stress, soreness, training load and "
                "injuries in the PMSys app."
            ),
            "device": "Fitbit and PMSys app",
            "strength": "Many people on one protocol, plus wellness labels",
            "limit": "Five months only, and one device type",
        },
    }

    _W, _PAD, _WRAP = 330, 22, 44
    _TICK_MAX = 30


    def _count(n):
        return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


    def _per_person(values, fmt=str):
        # Two people read as "a and b". A cohort reads as a range.
        values = sorted(values, reverse=True)
        if len(values) <= 2:
            return " and ".join(fmt(v) for v in values)
        return f"{fmt(values[-1])} to {fmt(values[0])}"


    def _lines(text):
        return textwrap.fill(text, _WRAP).count("\n") + 1


    # Every card reserves the same height for each section, so the stats, the
    # tradeoffs and the strip sit at the same height in all three cards.
    _BODY_LINES = max(_lines(s["context"]) for s in SOURCE_CARDS.values())
    _PRO_LINES = max(_lines(f"+ {s['strength']}") + _lines(f"- {s['limit']}") for s in SOURCE_CARDS.values())
    _DOT = 12
    _STACK = max(
        max(Counter(source_stats.filter(pl.col("source") == n)["tracked"]).values())
        for n in SOURCE_CARDS
    )


    def _layout(name):
        spec = SOURCE_CARDS[name]
        rows = source_stats.filter(pl.col("source") == name)
        texts = {"title": [], "role": [], "body": [], "label": [], "value": [], "note": []}
        y = 26
        texts["title"].append({"x": _PAD, "y": y, "t": name})
        y += 32
        texts["role"].append({"x": _PAD, "y": y, "t": spec["role"]})
        y += 30
        _body = textwrap.fill(spec["context"], _WRAP)
        texts["body"].append({"x": _PAD, "y": y, "t": _body})
        y += 18 * _BODY_LINES + 18

        _stats = [
            ("People", str(rows.height)),
            ("Days per person", _per_person(rows["days_covered"], lambda v: f"{v:,}")),
            ("Records per person", _per_person(rows["records"], _count)),
            ("Metrics on 30+ days", _per_person(rows["tracked"])),
            ("Device", spec["device"]),
        ]
        _rules = [{"x": _PAD, "x2": _W - _PAD, "y": y - 8}]
        for _label, _value in _stats:
            texts["label"].append({"x": _PAD, "y": y, "t": _label})
            texts["value"].append({"x": _W - _PAD, "y": y, "t": _value})
            y += 26
        _rules.append({"x": _PAD, "x2": _W - _PAD, "y": y - 4})
        y += 12

        _pros_top = y
        for _mark, _key in [("+", "strength"), ("\u2212", "limit")]:
            _line = textwrap.fill(f"{_mark} {spec[_key]}", _WRAP)
            texts["body"].append({"x": _PAD, "y": y, "t": _line})
            y += 18 * (_line.count("\n") + 1) + 4
        y = _pros_top + 22 * _PRO_LINES + 18

        # A dot plot on a shared 0 to 30 scale, one dot per person. People with
        # the same count stack upward, so all 16 PMData subjects stay visible.
        texts["note"].append({"x": _PAD, "y": y, "t": "Metrics tracked on 30+ days, one dot per person"})
        _base = y + 20 + _DOT * _STACK
        _px = lambda v: _PAD + v / _TICK_MAX * (_W - 2 * _PAD)
        _seen = Counter()
        _ticks = []
        for _v in sorted(rows["tracked"]):
            _ticks.append({"x": _px(_v), "y": _base - _DOT / 2 - _DOT * _seen[_v]})
            _seen[_v] += 1
        _axis = [{"x": _PAD, "x2": _W - _PAD, "y": _base}]
        for _v in range(0, _TICK_MAX + 1, 10):
            texts["note"].append({"x": _px(_v), "y": _base + 6, "t": str(_v), "center": True})
        y = _base + 24
        return spec, texts, _rules, _ticks, _axis, y


    _layouts = {name: _layout(name) for name in SOURCE_CARDS}
    _H = max(_l[-1] for _l in _layouts.values()) + _PAD

    _px_x = lambda f="x": alt.X(f"{f}:Q", scale=None, axis=None)
    _px_y = lambda f="y": alt.Y(f"{f}:Q", scale=None, axis=None)
    _STYLES = {
        "title": dict(fontSize=24, fontWeight="bold", color="#222"),
        "role": dict(fontSize=14, fontStyle="italic", color="#555"),
        "body": dict(fontSize=13, color="#333", lineHeight=18),
        "label": dict(fontSize=13, color="#555"),
        "value": dict(fontSize=13, fontWeight="bold", color="#222", align="right"),
        "note": dict(fontSize=11, color="#666"),
    }


    def _card(name):
        spec, texts, rules, ticks, axis, _ = _layouts[name]
        layers = [
            alt.Chart(alt.Data(values=[{}]))
            .mark_rect(fill="#f6f6f6", cornerRadius=10)
            .encode(x=alt.value(0), x2=alt.value(_W), y=alt.value(0), y2=alt.value(_H)),
            alt.Chart(alt.Data(values=[{}]))
            .mark_rect(fill=spec["color"], cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
            .encode(x=alt.value(0), x2=alt.value(_W), y=alt.value(0), y2=alt.value(8)),
            alt.Chart(alt.Data(values=rules + axis))
            .mark_rule(color="#cccccc")
            .encode(x=_px_x(), x2="x2:Q", y=_px_y()),
            alt.Chart(alt.Data(values=ticks))
            .mark_circle(color=spec["color"], size=80, opacity=1)
            .encode(x=_px_x(), y=_px_y()),
        ]
        for style, values in texts.items():
            opts = {"align": "left", "baseline": "top", "lineBreak": "\n", **_STYLES[style]}
            left = [v for v in values if not v.get("center")]
            centered = [v for v in values if v.get("center")]
            for group, extra in [(left, {}), (centered, {"align": "center"})]:
                if group:
                    layers.append(
                        alt.Chart(alt.Data(values=group))
                        .mark_text(**{**opts, **extra})
                        .encode(x=_px_x(), y=_px_y(), text="t:N")
                    )
        return alt.layer(*layers).properties(width=_W, height=_H)


    source_comparison_chart = (
        alt.hconcat(*[_card(name) for name in SOURCE_CARDS], spacing=24)
        .properties(
            title=alt.Title(
                "Three data sources, three different jobs",
                subtitle="Kaggle shows the problem, our exports add depth, PMData adds people and labels",
            )
        )
        .configure_title(fontSize=22, subtitleFontSize=14, anchor="start", offset=16)
        .configure_view(strokeWidth=0)
    )
    source_comparison_chart
    return (source_comparison_chart,)


@app.cell
def export_figures(
    ROOT,
    kaggle_json_table_chart,
    mo,
    richness_chart,
    source_comparison_chart,
):
    # Writes every slide figure as a PNG. scale_factor multiplies the pixel size,
    # so 4 turns the 880 px wide table into about 3500 px.
    FIGURES = ROOT / "figures"
    FIGURES.mkdir(exist_ok=True)

    _exports = {
        "kaggle_json_overview.png": kaggle_json_table_chart,
        "dataset_richness.png": richness_chart,
        "source_comparison.png": source_comparison_chart,
    }
    for _name, _chart in _exports.items():
        _chart.save(FIGURES / _name, scale_factor=4)

    mo.md("Saved " + ", ".join(f"`figures/{_name}`" for _name in _exports))
    return


if __name__ == "__main__":
    app.run()
