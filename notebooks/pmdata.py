import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from health import metrics, schema
    from health.cli import OUTPUT_DIR

    return OUTPUT_DIR, Path, alt, json, metrics, mo, pl, re, schema


@app.cell
def _(mo):
    mo.md("""
    # PMData: what is in it, and what of it we already have

    [PMData](https://datasets.simula.no/pmdata/) is 16 participants logged for five
    months. Three sources per participant: a Fitbit export, a PMSys athlete-report
    app, and a Google Forms food/weight log.

    This notebook answers two questions before we write a reader:

    1. What streams are there, at what grain, and for how many participants?
    2. Which of them land on a metric our schema already carries, and which do not?

    Nothing here writes to `data/processed/`.
    """)
    return


@app.cell
def _(OUTPUT_DIR, Path):
    # Notebooks get launched from wherever, so anchor on the repo root rather
    # than the working directory.
    ROOT = Path(__file__).resolve().parent.parent
    PMDATA = ROOT / "data" / "pmdata"
    PROCESSED = ROOT / OUTPUT_DIR

    participants = sorted(p.name for p in PMDATA.glob("p[0-9][0-9]") if p.is_dir())
    print(len(participants), "participants:", ", ".join(participants))
    return PMDATA, PROCESSED, participants


@app.cell
def _(Path, re):
    DATETIME_RE = re.compile(rb'"dateTime":\s*"([^"]+)"')

    def probe_json_span(path: Path, window: int = 8192) -> tuple[str | None, str | None]:
        """First and last dateTime in a JSON array, without parsing it.

        heart_rate.json is 128 MB per participant, two gigabytes across the cohort,
        and every record is one flat object on one line. Reading a window off each
        end gets the coverage span for the price of two seeks.
        """
        size = path.stat().st_size
        with path.open("rb") as fh:
            head = fh.read(window)
            fh.seek(max(0, size - window))
            tail = fh.read(window)
        first = DATETIME_RE.search(head)
        last = DATETIME_RE.findall(tail)
        return (
            first.group(1).decode() if first else None,
            last[-1].decode() if last else None,
        )

    return DATETIME_RE, probe_json_span


@app.cell
def _(mo):
    mo.md("""
    ## 1. What is on disk

    267 files, 1.9 GB, almost all of it `heart_rate.json`. Five months,
    2019-11-01 to 2020-03-31, the same window for everyone.
    """)
    return


@app.cell
def _(PMDATA, Path, participants, pl, probe_json_span):
    def _scan_file(participant: str, path: Path) -> dict[str, object]:
        """One row per file on disk. Cheap: no JSON is fully parsed here."""
        if path.suffix == ".json":
            first, last = probe_json_span(path)
            rows = None
        else:
            # Every PMData CSV puts its time column first and none is over a few
            # hundred rows, so reading the whole thing costs nothing.
            table = pl.read_csv(path, infer_schema_length=0)
            rows = table.height
            stamps = table[table.columns[0]].drop_nulls()
            first = stamps.first() if rows else None
            last = stamps.last() if rows else None
        return {
            "participant": participant,
            "source": path.parent.name,
            "stream": path.stem,
            "ext": path.suffix.lstrip("."),
            "mb": round(path.stat().st_size / 1e6, 2),
            "rows": rows,
            "first": first,
            "last": last,
        }

    files = pl.DataFrame(
        [
            _scan_file(person, path)
            for person in participants
            for path in sorted((PMDATA / person).rglob("*"))
            if path.is_file() and path.suffix in {".json", ".csv"}
        ]
    )
    files.head()
    return (files,)


@app.cell
def _(files, mo, pl):
    streams = (
        files.group_by("source", "stream", "ext")
        .agg(
            pl.col("participant").n_unique().alias("participants"),
            pl.col("mb").sum().round(1).alias("total_mb"),
            pl.col("first").min().alias("first"),
            pl.col("last").max().alias("last"),
        )
        .sort("source", "stream")
    )
    mo.ui.table(streams, selection=None)
    return


@app.cell
def _(files):
    # Four streams are not present for everyone. A reader has to treat a missing
    # file as "this person has no such data", not as a failure.
    missing = (
        files.select("participant", "stream")
        .unique()
        .pivot(on="stream", index="participant", values="stream", aggregate_function="len")
        .fill_null(0)
        .sort("participant")
    )
    gaps = [
        (person, stream)
        for stream in missing.columns[1:]
        for person, present in zip(missing["participant"], missing[stream], strict=True)
        if not present
    ]
    gaps
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Record shapes

    The Fitbit exports are all JSON arrays, but they are not one shape. Reading the
    first record of each shows four families, and only the first is uniform enough
    to read with a single code path.
    """)
    return


@app.cell
def _(PMDATA, Path, json, mo, participants, pl):
    def first_record(path: Path, window: int = 65536) -> object:
        """Decode the first element of a JSON array without reading the rest."""
        text = path.read_bytes()[:window].decode("utf-8", "ignore")
        start = text.index("[") + 1
        return json.JSONDecoder().raw_decode(text, start)[0]

    shapes = pl.DataFrame(
        [
            {
                "stream": path.stem,
                "record": json.dumps(first_record(path))[:220],
            }
            for path in sorted((PMDATA / participants[0] / "fitbit").glob("*.json"))
        ]
    )
    mo.ui.table(shapes, selection=None)
    return (first_record,)


@app.cell
def _(DATETIME_RE, PMDATA, Path, first_record, mo, participants, pl):
    def _family(record: object) -> str:
        if not isinstance(record, dict) or "dateTime" not in record:
            return "session"
        return "scalar" if isinstance(record["value"], str) else "nested"

    def _grain(path: Path) -> str | None:
        """Spacing between the first two records, which is the sampling grain.

        Only meaningful for the timeseries families. A session record nests its own
        dateTime keys (sleep stages, exercise levels), so scanning one would time
        the inside of a record rather than the gap between two.
        """
        text = path.read_bytes()[:65536]
        stamps = [m.decode() for m in DATETIME_RE.findall(text)[:2]]
        if len(stamps) < 2:
            return None
        seconds = pl.Series(stamps).str.to_datetime().diff()[1].total_seconds()
        return {5.0: "5s", 60.0: "1min", 86400.0: "1day"}.get(seconds, f"{seconds:g}s")

    def _describe(path: Path) -> dict[str, object]:
        family = _family(first_record(path))
        return {
            "stream": path.stem,
            "family": family,
            "grain": None if family == "session" else _grain(path),
            "mb_per_person": round(path.stat().st_size / 1e6, 1),
        }

    families = pl.DataFrame(
        [_describe(path) for path in sorted((PMDATA / participants[0] / "fitbit").glob("*.json"))]
    ).sort("family", "stream")
    mo.ui.table(families, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Where each stream would land in our schema

    Our vocabulary is HealthKit identifiers, so mapping PMData means naming a
    HealthKit metric for each stream. Some land exactly, some land approximately,
    and a good third of the dataset has nowhere to go.

    Two unit findings, checked against p01 rather than assumed:

    - `distance` is **centimetres**. A median day is 987,377 for 12,229 steps,
      which is 9.9 km. Our canonical length is `m`, so divide by 100.
    - `calories` is Fitbit's **total** burn, not active. A median day is 3,530,
      which is basal plus activity. It is not `ActiveEnergyBurned` and it is not
      `BasalEnergyBurned`; splitting it would mean subtracting an estimated BMR we
      do not have.
    """)
    return


@app.cell
def _(metrics, mo, pl):
    # One row per (stream, field) we could pull out. metric=None means the stream
    # has no HealthKit equivalent. Authored by hand; "fit" is the claim each row is
    # making and the thing to argue with.
    MAPPING = [
        # (source, stream, field, metric, unit, fit)
        # --- exact ---
        ("fitbit", "steps", "value", "StepCount", "count", "exact"),
        ("fitbit", "heart_rate", "value.bpm", "HeartRate", "count/min", "exact"),
        ("fitbit", "resting_heart_rate", "value.value", "RestingHeartRate", "count/min", "exact"),
        ("fitbit", "distance", "value", "DistanceWalkingRunning", "m", "exact, cm to m"),
        ("fitbit", "sleep", "levels.data[].level", "SleepAnalysis", None, "exact, stages renamed"),
        ("googledocs", "reporting", "weight", "BodyMass", "kg", "exact"),
        ("pmsys", "wellness", "sleep_duration_h", metrics.SLEEP_DURATION_DAILY, "s", "self-report"),
        # --- approximate ---
        ("fitbit", "very_active_minutes", "value", "AppleExerciseTime", "s", "different threshold"),
        ("fitbit", "moderately_active_minutes", "value", "AppleExerciseTime", "s", "double counts"),
        ("googledocs", "reporting", "glasses_of_fluid", "DietaryWater", "mL", "needs a glass size"),
        # --- no HealthKit equivalent ---
        ("fitbit", "calories", "value", None, "kcal", "total burn, neither Active nor Basal"),
        ("fitbit", "lightly_active_minutes", "value", None, "s", "no HK equivalent"),
        ("fitbit", "sedentary_minutes", "value", None, "s", "no HK equivalent"),
        ("fitbit", "time_in_heart_rate_zones", "value.valuesInZones", None, "s", "Fitbit zones"),
        ("fitbit", "sleep_score", "overall_score", None, None, "proprietary score"),
        ("fitbit", "exercise", "whole record", None, None, "a workout, and those are not done"),
        ("pmsys", "srpe", "perceived_exertion", None, None, "session RPE, no HK equivalent"),
        ("pmsys", "wellness", "fatigue, mood, readiness, ...", None, None, "1-5 self report"),
        ("pmsys", "injury", "injuries", None, None, "no HK equivalent"),
        ("googledocs", "reporting", "meals, alcohol_consumed", None, None, "free text"),
    ]

    # restore_prefix puts the HK prefix back, but it assumes anything unprefixed is
    # HealthKit, and SleepDurationDaily is ours, so derived names are held out.
    mapping = pl.DataFrame(
        MAPPING,
        schema={
            "source": pl.String,
            "stream": pl.String,
            "field": pl.String,
            "metric": pl.String,
            "unit": pl.String,
            "fit": pl.String,
        },
        orient="row",
    ).with_columns(
        pl.when(pl.col("metric").is_null() | pl.col("metric").is_in(list(metrics.DERIVED_METRICS)))
        .then(pl.col("metric"))
        .otherwise(metrics.restore_prefix(pl.col("metric")))
        .alias("metric")
    )
    mo.ui.table(mapping, selection=None)
    return (mapping,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Overlap with what we already parsed

    `have` is every metric currently in `data/processed/`, from the four subjects
    already run through the pipeline. The join below asks, per metric, whether
    PMData is adding a new column to our vocabulary or new rows under an existing
    one.
    """)
    return


@app.cell
def _(PROCESSED, pl, schema):
    parquet_files = sorted(PROCESSED.glob("*.parquet"))
    processed = pl.scan_parquet(parquet_files) if parquet_files else schema.empty_frame().lazy()

    have = (
        processed.group_by("metric")
        .agg(
            pl.col("subject_id").n_unique().alias("our_subjects"),
            pl.len().alias("our_rows"),
            pl.col("unit").first().alias("our_unit"),
        )
        .collect()
        .with_columns(pl.col("metric").cast(pl.String), pl.col("our_unit").cast(pl.String))
    )
    print(f"{have.height} metrics across {len(parquet_files)} processed files")
    return have, processed


@app.cell
def _(have, mapping, mo, pl):
    overlap = (
        mapping.filter(pl.col("metric").is_not_null())
        .select("source", "stream", "metric", pmdata_unit="unit", fit="fit")
        .unique(subset=["stream", "metric"], keep="first")
        .join(have, on="metric", how="left")
        .with_columns(
            pl.when(pl.col("our_rows").is_null())
            .then(pl.lit("new metric"))
            .when(pl.col("pmdata_unit") != pl.col("our_unit"))
            .then(pl.lit("unit mismatch"))
            .otherwise(pl.lit("more rows"))
            .alias("effect")
        )
        .sort("effect", "metric")
    )
    mo.ui.table(overlap, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    Every mapped metric is one we already carry. **PMData adds no new metric to the
    vocabulary**. It adds subjects and rows under names that already exist. That is
    the good case for a reader: nothing in `health/metrics.py` or `health/units.py`
    has to grow for the exact rows.

    ### How many rows

    Counting `"dateTime"` in a 1 MB window and scaling by file size lands within
    0.05% of a full parse of `steps.json`, so it is good enough to size the job
    without reading two gigabytes.
    """)
    return


@app.cell
def _(DATETIME_RE, PMDATA, Path, mo, participants, pl, processed):
    def estimate_records(path: Path, window: int = 1_000_000) -> int:
        """Records in a JSON array, from the density of dateTime keys in a window."""
        head = path.read_bytes()[:window]
        return round(len(DATETIME_RE.findall(head)) / len(head) * path.stat().st_size)

    volume = (
        pl.DataFrame(
            [
                {"stream": path.stem, "records": estimate_records(path)}
                for person in participants
                for path in sorted((PMDATA / person / "fitbit").glob("*.json"))
            ]
        )
        .group_by("stream")
        .agg(pl.col("records").sum().alias("cohort_records"))
        .sort("cohort_records", descending=True)
    )

    pmdata_rows = volume["cohort_records"].sum()
    our_rows = processed.select(pl.len()).collect().item()
    mo.md(f"""
    PMData would contribute about **{pmdata_rows:,} rows** against the
    **{our_rows:,}** we have now, a {pmdata_rows / our_rows:.0f}x increase, and
    {volume[0, "cohort_records"] / pmdata_rows:.0%} of it is `{volume[0, "stream"]}`.
    """)
    return estimate_records, volume


@app.cell
def _(mo, volume):
    mo.ui.table(volume, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Coverage is not the same for everyone

    The five-month window is identical for all 16, but the minute streams are not
    uniformly dense, and the reason differs by stream.

    `calories` is dense for almost everyone, a record for all 218,880 minutes,
    because Fitbit's per-minute burn includes a basal component that is never zero.
    `steps` and `distance` are sparse: most participants only get a record for
    minutes with movement, so their density sits near 0.4. p01 is the exception and
    writes the zero-step minutes too:

    ```
    p01  00:00  00:01  00:02  00:03   ...every minute, zeros included
    p02  21:11  21:12  21:41  21:42   ...only minutes with steps
    ```

    Two consequences for a reader. The `1min` grain in section 2 was read off p01
    and is that participant's convention, not the dataset's, so a row per minute
    cannot be assumed. And a missing minute means "no steps", not "not worn".
    `heart_rate` is the stream that tells you whether the watch was on, and it
    ranges from 2.9 to 7.4 records per minute across participants.
    """)
    return


@app.cell
def _(PMDATA, alt, estimate_records, participants, pl):
    MINUTES_IN_WINDOW = 152 * 24 * 60  # 2019-11-01 through 2020-03-31

    coverage = pl.DataFrame(
        [
            {
                "participant": person,
                "stream": stream,
                "records": estimate_records(PMDATA / person / "fitbit" / f"{stream}.json"),
            }
            for person in participants
            for stream in ("steps", "calories", "heart_rate")
        ]
    ).with_columns((pl.col("records") / MINUTES_IN_WINDOW).alias("per_minute"))

    coverage_chart = (
        alt.Chart(coverage)
        .mark_bar()
        .encode(
            x=alt.X("participant:N", title=None),
            y=alt.Y("per_minute:Q", title="records per minute of window"),
            color=alt.Color("stream:N", legend=None),
            tooltip=["participant", "stream", "records"],
        )
        .properties(height=110, width=440)
        .facet(row=alt.Row("stream:N", title=None))
        .resolve_scale(y="independent")
    )
    coverage_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Which clock are these timestamps on

    Our schema wants `start_utc` and `start_local` separately, so a reader has to
    know the offset. PMData presents three conventions and none of them carries one:

    | where | looks like |
    |---|---|
    | fitbit intraday, `sleep.startTime` | `2019-11-01 00:00:00`, naive |
    | `sleep_score.csv`, all of pmsys | `2019-11-01T06:29:30Z`, claims UTC |
    | `googledocs/reporting.csv` | `06/11/2019`, naive and day-first |

    The `Z` is not to be believed. Two independent checks below say every file is on
    the same naive wall clock.
    """)
    return


@app.cell
def _(PMDATA, json, participants, pl):
    # Check one. sleep.json and sleep_score.csv describe the same sleep sessions and
    # share a key, so the same instant is written both ways and the difference is
    # the offset.
    def _sleep_offsets(person: str) -> list[float]:
        fitbit = PMDATA / person / "fitbit"
        logs = json.loads((fitbit / "sleep.json").read_bytes())
        naive_end = {row["logId"]: row["endTime"].replace("T", " ")[:19] for row in logs}
        scores = pl.read_csv(fitbit / "sleep_score.csv", infer_schema_length=0)
        paired = scores.with_columns(
            pl.col("sleep_log_entry_id")
            .cast(pl.Int64)
            .replace_strict(naive_end, default=None)
            .alias("naive")
        ).drop_nulls("naive")
        claimed_utc = paired["timestamp"].str.to_datetime(time_zone="UTC")
        gap = paired["naive"].str.to_datetime() - claimed_utc.dt.replace_time_zone(None)
        return gap.dt.total_seconds().to_list()

    sleep_offsets = (
        pl.DataFrame(
            {"hours": [o / 3600 for person in participants for o in _sleep_offsets(person)]}
        )
        .group_by("hours")
        .len()
        .sort("len", descending=True)
    )
    sleep_offsets
    return


@app.cell
def _(PMDATA, json, mo, participants, pl):
    # Check two. pmsys is a different application from Fitbit, so agreeing with it
    # is a stronger claim than sleep.json agreeing with its own sibling. srpe logs
    # the end of a training session; exercise.json logs auto-detected sessions with
    # a naive start and a duration. Match each report to the nearest session.
    def _session_offsets(person: str) -> list[float]:
        sessions = json.loads((PMDATA / person / "fitbit" / "exercise.json").read_bytes())
        starts = pl.Series(
            [row["startTime"].replace("T", " ")[:19] for row in sessions]
        ).str.to_datetime()
        ran_for = pl.Series([row.get("activeDuration") or 0 for row in sessions]).cast(
            pl.Duration("ms")
        )
        ends = starts + ran_for

        reported = pl.read_csv(PMDATA / person / "pmsys" / "srpe.csv", infer_schema_length=0)
        stamps = reported["end_date_time"].str.to_datetime(time_zone="UTC")

        out = []
        for stamp in stamps.dt.replace_time_zone(None):
            gaps: list[float] = [(stamp - end).total_seconds() for end in ends]
            nearest: float = min(gaps, key=abs)
            if abs(nearest) < 3 * 3600:
                out.append(nearest / 3600)
        return out

    session_offsets = pl.Series(
        [offset for person in participants for offset in _session_offsets(person)]
    )
    _buckets = (
        pl.DataFrame({"bucket": (session_offsets * 4).round() / 4})
        .group_by("bucket")
        .len()
        .sort("len", descending=True)
    )
    mo.md(f"""
    {len(session_offsets)} matched sessions. Median offset
    **{session_offsets.median():.3f} h**, modal quarter-hour bucket
    **{_buckets[0, "bucket"]:.2f} h**, holding {_buckets[0, "len"]} of them. The
    spread either side is people filing the report a few minutes off the session,
    not a timezone: an hour offset would put the whole distribution on 1.00.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    So: **PMData has one clock, and it is naive local wall time.** The `Z` suffixes
    are decoration written by an exporter that had local time in hand. 1,831 of
    1,834 sleep logs agree exactly; the three that do not are off by 1.47, 1.99 and
    5.62 hours, which are not offsets any timezone has, so they are bad pairings
    rather than evidence of a second clock.

    The consequence is uncomfortable. Our schema needs `start_utc`, and PMData does
    not contain the information to produce one. The participants were recruited in
    Norway, so `Europe/Oslo` is the honest guess, and the five-month window crosses
    the 2020-03-29 DST change, so it cannot be a fixed offset either. Whatever a
    reader does here is an assumption, and it should be one line in one place rather
    than spread across twelve stream parsers.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Notes toward a reader

    **It is a directory reader, not a file reader.** The three existing readers each
    take one file. A PMData subject is a directory of 17 files across three
    subdirectories. Either `scan` learns to map a directory to a subject, or the
    reader takes `data/pmdata/p01/` as its unit. The second is less disruption.

    **Four parse shapes, not seventeen.** From section 2: `scalar`
    (`{dateTime, value: "12"}`, six streams), `nested` (`value` is an object, three
    streams), `session` (`sleep`, `exercise`), and the CSVs. A table from stream
    name to (metric, unit, extractor) covers the first two families in one code
    path, which is ten of the twelve JSON files.

    **Nothing new to add to the vocabulary.** Section 4: every stream that maps at
    all maps onto a metric already in `data/processed/`, with a unit already in
    `health/units.py`. The conversions needed are cm to m, minutes to s, and hours
    to s, all of which `units.py` already knows how to do.

    **Decide the timezone once.** Section 6. Suggest a module-level constant in the
    reader with the reasoning attached, feeding both `start_utc` and `start_local`,
    rather than a `tz` argument threaded through every stream.

    **`source_type` is `wearable` for all of fitbit** and `manual` for pmsys and
    googledocs. No `sources.local.toml` entries needed, since PMData has no
    free-text device names to place.

    **Sleep stages need a value mapping.** Fitbit writes `wake`, `light`, `deep`,
    `rem`; we store `HKCategoryValueSleepAnalysis*`. That is a new entry in
    `_VALUE_PREFIXES`-adjacent territory in `health/metrics.py`, the first one that
    is a rename rather than a prefix restoration.

    **Decide what to do with the two thirds that does not map.** `calories`,
    `sedentary_minutes`, `lightly_active_minutes`, heart-rate zones, sleep score,
    and all of the pmsys self-reports have no HealthKit name. Three options: drop
    them, invent non-HK metric names the way `SleepDurationDaily` already does, or
    keep them in a sidecar table. The schema tolerates the middle one today.

    **Size the run before starting it.** Section 4: about 27.5 million rows, 4.4x
    what we have, and 76% of it is `heart_rate` at 5-second grain. Worth deciding
    whether `heart_rate` gets downsampled on the way in, and worth streaming rather
    than `json.loads` on a 128 MB file per participant.
    """)
    return


if __name__ == "__main__":
    app.run()
