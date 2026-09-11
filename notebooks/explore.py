import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from health import metrics, schema, validate
    from health.cli import OUTPUT_DIR

    return OUTPUT_DIR, Path, alt, metrics, mo, pl, schema, validate


@app.cell
def _(mo):
    mo.md("""
    # Scratch pad

    Reads whatever `health-prep parse` last wrote to `data/processed/`, which is
    every source together: the Apple exports, the two other formats, and the 16
    PMData subjects. Nothing here is part of the pipeline, so break it freely.

    PMData brings names HealthKit has no equivalent for, such as `PMSysMood` and
    `PMSysSessionRPE`. They behave like any other metric here, but only 16 of the 20
    subjects have them, so a series being empty for someone is normal.

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
    # One lazy frame over every output. Scanning keeps the thirty-odd million rows
    # on disk until a cell actually asks for something.
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
    # Daily buckets rather than raw rows: altair chokes long before the row counts
    # here, and the shape of a day is what we actually look at. Both
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


@app.cell
def _(mo):
    mo.md("""
    ## Daily aggregation

    One row per subject and day. Each source type keeps its own series, so a watch
    and a phone that both log steps stay side by side as `StepCount@wearable` and
    `StepCount@phone` rather than being reconciled into one number here.

    `daily_long` picks one source per day using the trust order below, for when a
    single number is wanted. The rule for each metric comes from
    `health.metrics.CATALOGUE`, so a metric nobody has thought about lands in the
    unhandled list below instead of being quietly summed.
    """)
    return


@app.cell
def _(metrics):
    # How each metric collapses into one number per day now comes from
    # health.metrics.CATALOGUE, which also carries the domain and the canonical unit.
    # It used to be four lists in this cell, where nothing tested it and a new metric
    # could sit unnoticed for a while.
    _by_rule = metrics.by_daily_rule()

    DAILY_SKIP = _by_rule["skip"]
    DAILY_RULES = {rule: _by_rule[rule] for rule in ("sum", "mean", "duration", "count")}
    None
    return DAILY_RULES, DAILY_SKIP


@app.cell
def _(everything, pl):
    # Everything a daily rollup needs, with the categoricals cast to plain strings
    # so the four rule frames can be concatenated.
    #
    # The day comes from start_local, which puts a night of sleep on the day it
    # started. Fine for now, but worth revisiting: sleep is usually reported against
    # the morning you wake up.
    daily_base = everything.with_columns(
        pl.col("start_local").dt.date().alias("day"),
        pl.col("metric").cast(pl.String),
        pl.col("value_str").cast(pl.String),
        pl.col("source_type").cast(pl.String),
    )
    return (daily_base,)


@app.cell
def _(metrics, pl):
    # Everything a series is grouped by. The source type stays in the key, so each
    # device ends up with its own column rather than being merged with the others.
    DAILY_KEYS = ["subject_id", "day", "metric", "value_str", "source_type"]

    def daily_intervals(frame, names):
        """Seconds covered by the named category metrics, per key.

        Overlapping intervals are merged rather than added. One of the sleep apps
        rewrites the same interval hundreds of times, which turned a seven hour night
        into twelve, and a watch that logs in-bed alongside asleep overlaps by design.
        Sorting by start and opening a new block whenever a row begins after every
        earlier row has ended gives the covered time instead of the recorded time.
        """
        starts_a_gap = pl.col("start_utc") > pl.col("end_utc").cum_max().shift(1).over(DAILY_KEYS)
        block_seconds = (pl.col("end_utc").max() - pl.col("start_utc").min()).dt.total_seconds()

        return (
            frame.filter(pl.col("metric").is_in(names))
            .sort(*DAILY_KEYS, "start_utc")
            .with_columns(starts_a_gap.fill_null(True).cum_sum().over(DAILY_KEYS).alias("block"))
            .group_by(*DAILY_KEYS, "block")
            .agg(block_seconds.alias("covered"))
            .group_by(*DAILY_KEYS)
            .agg(pl.col("covered").sum().cast(pl.Float64).alias("value"))
        )

    def daily_rule(frame, rule, names):
        """One value per subject, day, category value and source type."""
        if rule == "duration":
            per_source = daily_intervals(frame, names)
        else:
            _values = {
                "sum": pl.col("value_num").sum(),
                "mean": pl.col("value_num").mean(),
                # Distinct starts, not rows: the same event is often written twice.
                "count": pl.col("start_utc").n_unique(),
            }
            per_source = (
                frame.filter(pl.col("metric").is_in(names))
                .group_by(*DAILY_KEYS)
                .agg(_values[rule].cast(pl.Float64).alias("value"))
            )

        # Carried through so the sources can be combined later without looking the
        # rule up again.
        return per_source.with_columns(pl.lit(rule).alias("rule"))

    def series_name(metric, value):
        """A short name for a metric. Category metrics get one series per value."""
        name = metric.removeprefix(metrics.QUANTITY_PREFIX).removeprefix(metrics.CATEGORY_PREFIX)
        if value is None:
            return name
        stage = value.removeprefix(metrics.VALUE_PREFIX).removeprefix(name)
        return f"{name}/{stage}" if stage and stage != "NotApplicable" else f"{name}/Event"

    return daily_rule, series_name


@app.cell
def _(DAILY_RULES, daily_base, daily_rule, pl, series_name):
    daily_by_source = (
        pl.concat([daily_rule(daily_base, rule, names) for rule, names in DAILY_RULES.items()])
        .collect()
        .with_columns(
            pl.struct("metric", "value_str")
            .map_elements(
                lambda row: series_name(row["metric"], row["value_str"]),
                return_dtype=pl.String,
            )
            .alias("series")
        )
        .select("subject_id", "day", "series", "source_type", "rule", "value")
        .sort("subject_id", "day", "series", "source_type")
    )
    daily_by_source
    return (daily_by_source,)


@app.cell
def _(pl, schema):
    # Which source to believe when more than one recorded the same day. Most
    # trusted first. Nothing is averaged or added across sources: the best available
    # source wins the day outright and the others are dropped.
    #
    # A watch leads because it is worn, so it sees the whole day and measures at the
    # wrist rather than inferring from a pocket. A phone beats an app because the
    # apps here mostly re-import partial histories. Manual entry, anything derived
    # and anything unclassified come last.
    SOURCE_TRUST = ["wearable", "scale", "phone", "app", "manual", "derived", "unknown"]

    # Body composition is the exception. A scale is the instrument for it, and a
    # watch only ever holds a figure someone typed in somewhere else.
    BODY_TRUST = ["scale", "manual", "wearable", "phone", "app", "derived", "unknown"]
    BODY_SERIES = ["BodyMass", "BodyFatPercentage", "LeanBodyMass", "BodyMassIndex"]

    # Both orders have to name every source type the schema allows, or a day
    # recorded by an unlisted source would silently drop out of the ranking.
    assert set(SOURCE_TRUST) == set(schema.SOURCE_TYPES), sorted(schema.SOURCE_TYPES)
    assert set(BODY_TRUST) == set(schema.SOURCE_TYPES)

    def trust_rank():
        """Rank of each row's source, lowest is most trusted."""
        general = pl.col("source_type").replace_strict(
            {name: i for i, name in enumerate(SOURCE_TRUST)}, return_dtype=pl.Int8
        )
        body = pl.col("source_type").replace_strict(
            {name: i for i, name in enumerate(BODY_TRUST)}, return_dtype=pl.Int8
        )
        return pl.when(pl.col("series").is_in(BODY_SERIES)).then(body).otherwise(general)

    return (trust_rank,)


@app.cell
def _(daily_by_source, pl, trust_rank):
    # The same table with one source picked per day, for when a single number is
    # wanted. The chosen source is carried through, since a series that keeps
    # falling back to a phone is worth knowing about before it goes into a model.
    daily_long = (
        daily_by_source.with_columns(trust_rank().alias("trust"))
        .sort("subject_id", "day", "series", "trust")
        .group_by("subject_id", "day", "series", maintain_order=True)
        .agg(
            pl.col("value").first(),
            pl.col("source_type").first().alias("chosen"),
            pl.col("source_type").len().alias("available"),
        )
        .sort("subject_id", "day", "series")
    )
    daily_long
    return (daily_long,)


@app.cell
def _(DAILY_RULES, DAILY_SKIP, everything, pl):
    # Anything present in the data that no rule covers. Should stay empty.
    _covered = set(DAILY_SKIP).union(*DAILY_RULES.values())
    _present = set(everything.select("metric").unique().collect().to_series().cast(pl.String))
    sorted(_present - _covered) or "every metric has a rule"
    return


@app.cell
def _(daily_by_source, mo, pl):
    # Column per series and source type. Sorted by name so the two spellings of a
    # metric that more than one device records end up next to each other.
    _columns = daily_by_source.with_columns(
        (pl.col("series") + "@" + pl.col("source_type")).alias("column")
    )
    _pivoted = _columns.pivot(on="column", index=["subject_id", "day"], values="value")
    _index = ["subject_id", "day"]

    daily_wide = _pivoted.select(
        *_index, *sorted(c for c in _pivoted.columns if c not in _index)
    ).sort(*_index)

    mo.vstack(
        [
            mo.md(f"{daily_wide.height:,} subject-days, {daily_wide.width - 2} series"),
            mo.ui.table(daily_wide.head(300), selection=None),
        ]
    )
    return


@app.cell
def _(daily_by_source, mo):
    daily_series_pick = mo.ui.dropdown(
        sorted(daily_by_source["series"].unique()),
        value="StepCount",
        label="daily series",
    )
    daily_series_pick
    return (daily_series_pick,)


@app.cell
def _(alt, daily_by_source, daily_series_pick, mo, pl):
    # One line per source type, so a metric two devices both record shows the gap
    # between them rather than hiding it.
    _one = daily_by_source.filter(pl.col("series") == daily_series_pick.value)

    _chart = (
        alt.Chart(_one)
        .mark_line()
        .encode(
            x=alt.X("day:T", title=None),
            y=alt.Y("value:Q", title=daily_series_pick.value),
            color=alt.Color("source_type:N"),
            row=alt.Row("subject_id:N", title=None),
            tooltip=["subject_id:N", "source_type:N", "day:T", "value:Q"],
        )
        .properties(height=140)
    )
    mo.ui.altair_chart(_chart) if _one.height else mo.md("Nothing for this series.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Feature sets

    A feature set is only usable where every feature in it is present on the same
    day, so the question is not which features exist but how many days survive
    requiring all of them at once.

    Subjects are scored on their own timelines. Nothing here asks them to line up
    with each other, so a set is worth the sum of what each subject gets from it,
    and the run lengths matter more than the calendar.
    """)
    return


@app.cell
def _(daily_long, metrics, mo, pl):
    # One column per feature. Sleep needs a bridge first: subject_04 ships a daily
    # total, subject_02 only ever recorded time in bed, and the other two have per
    # stage intervals. SleepTotal takes the best available of those, in that order.
    # The detailed stages win over AsleepUnspecified rather than adding to it, since
    # a phone app and a watch describing the same night would otherwise stack into
    # fourteen hour nights.
    SLEEP_STAGES = [
        "SleepAnalysis/AsleepCore",
        "SleepAnalysis/AsleepDeep",
        "SleepAnalysis/AsleepREM",
    ]

    _staged = pl.any_horizontal([pl.col(c).is_not_null() for c in SLEEP_STAGES])
    _sleep_total = pl.coalesce(
        pl.when(_staged).then(pl.sum_horizontal([pl.col(c) for c in SLEEP_STAGES])),
        pl.col("SleepAnalysis/AsleepUnspecified"),
        pl.col("SleepAnalysis/InBed"),
        pl.col(metrics.SLEEP_DURATION_DAILY),
    )

    daily_features = (
        daily_long.pivot(on="series", index=["subject_id", "day"], values="value")
        .with_columns(_sleep_total.alias("SleepTotal"))
        .sort("subject_id", "day")
    )

    FEATURE_NAMES = sorted(c for c in daily_features.columns if c not in ("subject_id", "day"))
    mo.md(f"{daily_features.height:,} subject-days, {len(FEATURE_NAMES)} candidate features")
    return FEATURE_NAMES, SLEEP_STAGES, daily_features


@app.cell
def _(daily_features, pl):
    # How dense each feature is, measured against the subject's own span rather
    # than the calendar, so a subject who joined late is not penalised for it.
    _span = daily_features.group_by("subject_id").agg(
        (pl.col("day").max() - pl.col("day").min()).dt.total_days().alias("span")
    )

    feature_coverage = (
        daily_features.unpivot(
            index=["subject_id", "day"], variable_name="feature", value_name="value"
        )
        .drop_nulls("value")
        .group_by("subject_id", "feature")
        .agg(pl.col("day").n_unique().alias("days"))
        .join(_span, on="subject_id")
        .with_columns((pl.col("days") / (pl.col("span") + 1)).alias("density"))
        .select("subject_id", "feature", "days", "density")
        .sort("feature", "subject_id")
    )
    feature_coverage
    return (feature_coverage,)


@app.cell
def _(alt, feature_coverage, mo, pl):
    # Features nobody recorded for a month are dropped, or the chart is mostly
    # blank rows. The ordering puts the features shared by the most subjects first,
    # which is the order you want when picking a set.
    _usable = (
        feature_coverage.filter(pl.col("days") >= 30)
        .group_by("feature")
        .agg(pl.len().alias("subjects"), pl.col("density").sum().alias("weight"))
        .sort("subjects", "weight", descending=True)
    )
    _order = _usable["feature"].to_list()

    _grid = feature_coverage.filter(pl.col("feature").is_in(_order))

    mo.ui.altair_chart(
        alt.Chart(_grid)
        .mark_rect()
        .encode(
            x=alt.X("subject_id:N", title=None),
            y=alt.Y("feature:N", title=None, sort=_order),
            color=alt.Color("density:Q", scale=alt.Scale(scheme="viridis"), title="density"),
            tooltip=["feature:N", "subject_id:N", "days:Q", "density:Q"],
        )
        .properties(width=200, height=18 * len(_order))
    )
    return


@app.cell
def _(FEATURE_NAMES, SLEEP_STAGES):
    # Candidate sets, narrow to wide. Named so the tradeoff is legible: every
    # feature added costs subjects, days, or both.
    _ACTIVITY = [
        "StepCount",
        "DistanceWalkingRunning",
        "FlightsClimbed",
        "ActiveEnergyBurned",
        "BasalEnergyBurned",
    ]
    _GAIT = [
        "WalkingSpeed",
        "WalkingStepLength",
        "WalkingAsymmetryPercentage",
        "WalkingDoubleSupportPercentage",
    ]
    _STAGES = [*SLEEP_STAGES, "SleepAnalysis/Awake", "SleepAnalysis/InBed"]

    FEATURE_SETS = {
        "steps and heart": ["StepCount", "HeartRate"],
        "core": ["StepCount", "HeartRate", "SleepTotal"],
        "cardio": ["HeartRate", "RestingHeartRate", "HeartRateVariabilitySDNN"],
        "activity": _ACTIVITY,
        "gait": _GAIT,
        "sleep stages": _STAGES,
        "core plus activity": ["HeartRate", "SleepTotal", *_ACTIVITY],
        "rich": ["HeartRate", "RestingHeartRate", "SleepTotal", *_ACTIVITY, *_GAIT],
    }

    # Nothing here should be a typo.
    _unknown = {f for features in FEATURE_SETS.values() for f in features} - set(FEATURE_NAMES)
    assert not _unknown, sorted(_unknown)
    None
    return (FEATURE_SETS,)


@app.cell
def _(daily_features, pl):
    def longest_runs(complete):
        """Longest unbroken stretch of complete days, per subject.

        A new block starts wherever a day does not follow the one before it, so the
        biggest block is the longest run. Two subjects with the same total are not
        worth the same if one has it in a block and the other in scattered days,
        since anything with a lag or a rolling window needs the block.
        """
        breaks = (pl.col("day").diff().over("subject_id") > pl.duration(days=1)).fill_null(True)
        return (
            complete.with_columns(breaks.cum_sum().over("subject_id").alias("block"))
            .group_by("subject_id", "block")
            .len()
            .group_by("subject_id")
            .agg(pl.col("len").max().alias("run"))
        )

    def set_coverage(features):
        """Days where every one of these features is present, per subject."""
        complete = (
            daily_features.drop_nulls(subset=features)
            .select("subject_id", "day")
            .sort("subject_id", "day")
        )
        totals = complete.group_by("subject_id").agg(
            pl.len().alias("days"),
            pl.col("day").min().alias("first"),
            pl.col("day").max().alias("last"),
        )
        return (
            totals.join(longest_runs(complete), on="subject_id")
            .select("subject_id", "days", "run", "first", "last")
            .sort("subject_id")
        )

    def set_summary(name, features):
        """One row describing what a set is worth across all subjects."""
        per_subject = set_coverage(features)
        if per_subject.is_empty():
            return None

        return {
            "set": name,
            "features": len(features),
            "subjects": per_subject.height,
            "subject_days": int(per_subject["days"].sum()),
            "worst_subject": int(per_subject["days"].min()),
            "best_subject": int(per_subject["days"].max()),
            "worst_run": int(per_subject["run"].min()),
            "best_run": int(per_subject["run"].max()),
        }

    return set_coverage, set_summary


@app.cell
def _(FEATURE_SETS, mo, pl, set_summary):
    set_scores = pl.DataFrame(
        [row for name, features in FEATURE_SETS.items() if (row := set_summary(name, features))]
    )
    mo.ui.table(set_scores, selection=None)
    return


@app.cell
def _(FEATURE_SETS, alt, mo, pl, set_coverage):
    _bars = pl.concat(
        [
            set_coverage(features).with_columns(pl.lit(name).alias("set"))
            for name, features in FEATURE_SETS.items()
        ]
    )

    mo.ui.altair_chart(
        alt.Chart(_bars)
        .mark_bar()
        .encode(
            x=alt.X("subject_id:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("days:Q", title="complete days"),
            color=alt.Color("subject_id:N"),
            column=alt.Column("set:N", title=None, sort=list(FEATURE_SETS)),
            tooltip=["set:N", "subject_id:N", "days:Q", "first:T", "last:T"],
        )
        .properties(width=70, height=200)
    )
    return


@app.cell
def _(FEATURE_NAMES, FEATURE_SETS, mo):
    feature_build = mo.ui.multiselect(
        FEATURE_NAMES,
        value=FEATURE_SETS["core"],
        label="build a set",
    )
    feature_build
    return (feature_build,)


@app.cell
def _(alt, daily_features, feature_build, mo, set_coverage):
    _picked = list(feature_build.value)

    if not _picked:
        _out = mo.md("Pick at least one feature.")
    else:
        _per_subject = set_coverage(_picked)
        _complete = daily_features.drop_nulls(subset=_picked).select("subject_id", "day")

        # A tick per complete day. Gaps in a row are the thing to look at: a subject
        # with the same total spread over three years is worth less than one with it
        # in a solid block.
        _timeline = (
            alt.Chart(_complete)
            .mark_tick(thickness=1)
            .encode(
                x=alt.X("day:T", title=None),
                y=alt.Y("subject_id:N", title=None),
                color=alt.Color("subject_id:N", legend=None),
                tooltip=["subject_id:N", "day:T"],
            )
            .properties(height=110)
        )
        _out = mo.vstack(
            [
                mo.md(f"{_per_subject['days'].sum():,} subject-days across {_per_subject.height}"),
                mo.ui.table(_per_subject, selection=None),
                mo.ui.altair_chart(_timeline),
            ]
        )

    _out
    return


@app.cell
def _(mo):
    mo.md("""
    ## Which features travel together

    Most features come from one device recording a whole family of things at once,
    so requiring one often gets several more for nothing. These cells measure that
    directly, over pooled subject-days since the timelines are independent.

    `given_a` is the share of days holding the row feature that also hold the column
    feature, which is the asymmetric question worth asking: active energy nearly
    always brings basal energy, but plenty of days have basal energy alone.
    `jaccard` is the symmetric version, used to order the chart so families sit in
    blocks.
    """)
    return


@app.cell
def _(daily_features, pl):
    _present = (
        daily_features.unpivot(index=["subject_id", "day"], variable_name="feature", value_name="v")
        .drop_nulls("v")
        .select("subject_id", "day", "feature")
    )
    _totals = _present.group_by("feature").len().rename({"len": "days"})

    # Every feature against every other one on the same subject-day. The join is
    # wide but shallow, around fourteen features on a typical day.
    feature_pairs = (
        _present.join(_present, on=["subject_id", "day"], suffix="_b")
        .group_by("feature", "feature_b")
        .len()
        .rename({"len": "both"})
        .join(_totals, on="feature")
        .join(_totals.rename({"feature": "feature_b", "days": "days_b"}), on="feature_b")
        .with_columns(
            (pl.col("both") / pl.col("days")).alias("given_a"),
            (pl.col("both") / (pl.col("days") + pl.col("days_b") - pl.col("both"))).alias(
                "jaccard"
            ),
        )
        .select("feature", "feature_b", "both", "days", "days_b", "given_a", "jaccard")
    )
    feature_pairs
    return (feature_pairs,)


@app.function
def similar_order(pairs, counts):
    """Features arranged so the ones that co-occur end up next to each other.

    Starts at the most common feature and repeatedly takes whichever unused one
    overlaps it most. Crude next to real clustering, but enough to make the
    families show up as blocks and it needs no extra dependency.
    """
    overlap = {
        (row["feature"], row["feature_b"]): row["jaccard"]
        for row in pairs.select("feature", "feature_b", "jaccard").iter_rows(named=True)
    }

    remaining = dict(counts)
    current = max(remaining, key=remaining.get)
    order = []
    while remaining:
        del remaining[current]
        order.append(current)
        if not remaining:
            break
        current = max(remaining, key=lambda f: (overlap.get((order[-1], f), 0.0), remaining[f]))
    return order


@app.cell
def _(alt, feature_pairs, mo, pl):
    # Features seen on fewer than thirty days are left out, or the chart is mostly
    # empty rows nobody would build a set from.
    _common = feature_pairs.filter(pl.col("feature") == pl.col("feature_b")).filter(
        pl.col("days") >= 30
    )
    _counts = dict(zip(_common["feature"], _common["days"], strict=True))

    _grid = feature_pairs.filter(
        pl.col("feature").is_in(list(_counts)) & pl.col("feature_b").is_in(list(_counts))
    )
    feature_order = similar_order(_grid, _counts)

    mo.ui.altair_chart(
        alt.Chart(_grid)
        .mark_rect()
        .encode(
            x=alt.X("feature_b:N", title=None, sort=feature_order),
            y=alt.Y("feature:N", title=None, sort=feature_order),
            color=alt.Color(
                "given_a:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                title="share of row's days",
            ),
            tooltip=["feature:N", "feature_b:N", "both:Q", "days:Q", "given_a:Q", "jaccard:Q"],
        )
        .properties(width=16 * len(feature_order), height=16 * len(feature_order))
    )
    return


@app.cell
def _(daily_features, feature_build, mo, pl):
    # What a set picks up for nothing. Anything listed here is present on nearly
    # every day the chosen set is already complete, so adding it costs almost no
    # days. Driven by the multiselect above.
    _chosen = list(feature_build.value)
    _complete = daily_features.drop_nulls(subset=_chosen) if _chosen else daily_features.clear()

    if _complete.is_empty():
        free_features = daily_features.clear().select("subject_id")
        _out = mo.md("Pick a set above.")
    else:
        free_features = (
            _complete.select(pl.exclude("subject_id", "day"))
            .select(pl.all().is_not_null().sum())
            .unpivot(variable_name="feature", value_name="days")
            .filter(~pl.col("feature").is_in(_chosen))
            .with_columns((pl.col("days") / _complete.height).alias("kept"))
            .filter(pl.col("kept") > 0.5)
            .sort("kept", descending=True)
        )
        _out = mo.vstack(
            [
                mo.md(
                    f"{_complete.height:,} complete days for the {len(_chosen)} chosen features. "
                    f"Features below survive on more than half of them."
                ),
                mo.ui.table(free_features, selection=None),
            ]
        )

    _out
    return


if __name__ == "__main__":
    app.run()
