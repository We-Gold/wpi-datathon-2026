"""Reader for the PMData dataset.

https://datasets.simula.no/pmdata/

A subject is a directory, not a single file:

    p01/fitbit/steps.json          {"dateTime": "...", "value": "12"}
    p01/fitbit/heart_rate.json     {"dateTime": "...", "value": {"bpm": 54, ...}}
    p01/fitbit/sleep.json          one record per sleep session, stages nested
    p01/pmsys/wellness.csv         daily self report
    p01/googledocs/reporting.csv   food and weight log

Seventeen files, but only four shapes. Eight of them are a flat list of
timestamped values and come straight off the _TIMESERIES table below. Two more
have that shape but are summed together, see _EXERCISE_MINUTES. Sleep and the
four CSV files each need their own function.

exercise.json is the one file not read at all. It holds workouts, and workouts
are not done yet anywhere in this project.

Three things to know before changing this file.

PMData records local wall clock time and never writes an offset. Some files add
a "Z" suffix, which would normally mean UTC, but it is wrong. Sleep sessions
appear in both sleep.json (no suffix) and sleep_score.csv (with the suffix) and
the two agree to the second, so the suffix is decoration. See
PMDATA_OFFSET_MINUTES.

Files are missing for some subjects, so every read checks the path first. A
subject with no resting_heart_rate.json is normal, not an error.

"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import polars as pl

from health import anonymize, metrics, timestamps
from health.schema import PolarsDataType, conform, split_value

# Minutes east of UTC. PMData holds no offset, so this is supplied rather than
# read, and every UTC timestamp this reader produces depends on it. One constant
# rather than an argument threaded through the file.
#
# A fixed offset and not a named zone, because the clock that wrote these files
# did not follow daylight saving. Oslo jumped from 02:00 to 03:00 on 2020-03-29
# and the devices kept recording through that hour at their usual rate, which
# only happens if the clock never moved. A named zone would throw those rows
# away and shift the last three days of the window by an hour.
#
# Which fixed offset is a weaker claim. The files could be Norwegian winter time
# or plain UTC, and nothing in the dataset settles it. +01:00 fits best: bedtimes
# peak at 23:00 and morning wellness reports at 08:00, which read as local. If
# that is ever settled, this is the only line to change.
#
# start_local does not depend on any of this and is exact. Prefer start_local for
# anything grouped by day.
PMDATA_OFFSET_MINUTES = 60

# PMData ships with no device names in it, so there is nothing to classify and
# nothing lands in sources.local.toml. The other readers pass raw source strings
# through anonymize.SourceMap. This one does not need to, because the folder a
# file came from already says what recorded it.
_SOURCE_TYPE_BY_DIR = {
    "fitbit": "wearable",
    "pmsys": "manual",
    "googledocs": "manual",
}

# Short aliases, because the metric names below are already long.
_QUANTITY = metrics.QUANTITY_PREFIX
_CATEGORY = metrics.CATEGORY_PREFIX

_MINUTE = timedelta(minutes=1)
_DAY = timedelta(days=1)


@dataclass(frozen=True)
class _Timeseries:
    """One flat JSON file of timestamped values.

    pointer walks into the value object. An empty pointer means the value is the
    whole field, which is how the plain "value": "12" files are written.

    span is how long one record covers. It sets end_utc and end_local. A daily
    total covers a day even though it is stamped at midnight, and reading that
    span off the timestamps alone is not possible.

    zero_is_missing drops records whose value is 0. Fitbit writes 0 on days it
    has no reading for some streams, and a zero there is not a measurement.
    """

    filename: str
    metric: str
    unit: str | None
    span: timedelta
    pointer: tuple[str, ...] = ()
    dtype: PolarsDataType = pl.String
    zero_is_missing: bool = False

    def schema(self) -> dict[str, PolarsDataType]:
        """The exact shape to read, built from the pointer.

        Always passed to read_json rather than letting polars infer. Inference
        looks at the first records only, and these files change shape part way
        through. One subject's sleep summary gains a field the first time a
        classic log appears, which is enough to fail the whole read. Naming the
        shape also means anything not named is ignored.
        """
        leaf = self.dtype
        for key in reversed(self.pointer):
            leaf = pl.Struct({key: leaf})
        return {"dateTime": pl.String, "value": leaf}


# Units are the ones the file records, not the ones we store. health.units
# converts them afterwards, so "cm" and "min" belong here and metres and seconds
# do not.
_TIMESERIES: tuple[_Timeseries, ...] = (
    _Timeseries("steps.json", _QUANTITY + "StepCount", "count", _MINUTE),
    # Centimetres. A median day for p01 is 987,377 against 12,229 steps, which
    # is 9.9 km.
    _Timeseries("distance.json", _QUANTITY + "DistanceWalkingRunning", "cm", _MINUTE),
    _Timeseries(
        "heart_rate.json", _QUANTITY + "HeartRate", "count/min", timedelta(0), ("bpm",), pl.Int64
    ),
    # 0 on days Fitbit could not estimate it: 11 of 16 subjects have such days,
    # one of them 34. No one has a resting heart rate of 0.
    _Timeseries(
        "resting_heart_rate.json",
        _QUANTITY + "RestingHeartRate",
        "count/min",
        _DAY,
        ("value",),
        pl.Float64,
        zero_is_missing=True,
    ),
    # Fitbit's calorie figure is basal plus activity in one number. It is not
    # ActiveEnergyBurned and it is not BasalEnergyBurned, and splitting it would
    # need a BMR estimate we do not have.
    _Timeseries("calories.json", metrics.TOTAL_ENERGY_BURNED, "kcal", _MINUTE),
    _Timeseries("sedentary_minutes.json", "SedentaryTime", "min", _DAY),
    _Timeseries("lightly_active_minutes.json", "LightActivityTime", "min", _DAY),
)

# Apple's exercise time counts brisk activity and above, which is what these two
# Fitbit buckets hold between them. They are summed into one row per day. Two
# rows under one metric name would double count every daily total.
_EXERCISE_MINUTES = ("moderately_active_minutes.json", "very_active_minutes.json")

# Columns of wellness.csv that are 1 to 5 rating scales, and the metric each one
# becomes. The PMSys prefix is rule 3 in health/metrics.py: one system produces
# these, so the name says which. readiness is left out on purpose: it runs 0 to
# 10, so a zero there is a real answer rather than a skipped question.
_WELLNESS_SCALES = {
    "fatigue": "Fatigue",
    "mood": "Mood",
    "sleep_quality": "SleepQuality",
    "soreness": "Soreness",
    "stress": "Stress",
}

# Only levels.data is named, so levels.summary and levels.shortData are ignored
# on the way in. summary is the field that changes shape between a stages log
# and a classic one, and shortData would double count wake time.
_SLEEP_SCHEMA: dict[str, PolarsDataType] = {
    "levels": pl.Struct(
        {
            "data": pl.List(
                pl.Struct({"dateTime": pl.String, "level": pl.String, "seconds": pl.Int64})
            )
        }
    )
}

# The two active minute files, both plain "value": "58" records.
_MINUTES_SCHEMA: dict[str, PolarsDataType] = {"dateTime": pl.String, "value": pl.String}

# Only the overall score. Its three components sum to it exactly, so reading
# them as well would store the same number twice. See DROPPED_PMDATA_METRICS.
_SLEEP_SCORES = {"overall_score": "SleepScore"}


def _empty() -> pl.DataFrame:
    """A frame with the columns the builders below produce, and no rows."""
    return pl.DataFrame(
        schema={
            "metric": pl.String,
            "unit": pl.String,
            "value": pl.String,
            "start_local": pl.Datetime("us"),
            "end_local": pl.Datetime("us"),
            "source_dir": pl.String,
        }
    )


def _rows(
    frame: pl.DataFrame,
    metric: str,
    unit: str | None,
    value: pl.Expr,
    start: pl.Expr,
    end: pl.Expr,
    source_dir: str,
) -> pl.DataFrame:
    """Put one metric's rows into the shape _finish expects."""
    return frame.select(
        pl.lit(metric).alias("metric"),
        pl.lit(unit, dtype=pl.String).alias("unit"),
        value.cast(pl.String).alias("value"),
        start.alias("start_local"),
        end.alias("end_local"),
        pl.lit(source_dir).alias("source_dir"),
    )


def _read_timeseries(directory: Path, spec: _Timeseries) -> pl.DataFrame:
    """Read one flat JSON file of timestamped values."""
    path = directory / "fitbit" / spec.filename
    if not path.exists():
        return _empty()

    frame = pl.read_json(path, schema=spec.schema())
    value = pl.col("value")
    for key in spec.pointer:
        value = value.struct.field(key)
    if spec.zero_is_missing:
        frame = frame.filter(value != 0)

    start = pl.col("dateTime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", strict=False)
    return _rows(frame, spec.metric, spec.unit, value, start, start + spec.span, "fitbit")


def _read_exercise_minutes(directory: Path) -> pl.DataFrame:
    """Moderate and vigorous minutes, summed into one AppleExerciseTime row a day."""
    parts = [
        pl.read_json(directory / "fitbit" / name, schema=_MINUTES_SCHEMA)
        for name in _EXERCISE_MINUTES
        if (directory / "fitbit" / name).exists()
    ]
    if not parts:
        return _empty()

    daily = (
        pl.concat(parts)
        .group_by("dateTime")
        .agg(pl.col("value").cast(pl.Float64).sum().alias("minutes"))
    )
    start = pl.col("dateTime").str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", strict=False)
    return _rows(
        daily,
        _QUANTITY + "AppleExerciseTime",
        "min",
        pl.col("minutes"),
        start,
        start + _DAY,
        "fitbit",
    )


def _read_sleep(directory: Path) -> pl.DataFrame:
    """Sleep stages, one row per stage interval.

    Only levels.data is read. levels.shortData holds brief wake episodes that
    sit inside the same intervals, so reading both would count that time twice.

    Stage names come from FITBIT_SLEEP_STAGES. A name that is not in that table,
    such as the handful of "unknown" entries in the real files, is dropped
    rather than guessed at.
    """
    path = directory / "fitbit" / "sleep.json"
    if not path.exists():
        return _empty()

    frame = pl.read_json(path, schema=_SLEEP_SCHEMA)
    if not frame.height:
        return _empty()

    stages = (
        frame.select(pl.col("levels").struct.field("data"))
        # A sleep log with no stage list explodes to a null rather than being
        # dropped here, so drop_nulls below has something to catch.
        .explode("data", empty_as_null=True)
        .drop_nulls("data")
        .unnest("data")
    )
    if not stages.height:
        return _empty()

    stage = pl.col("level").replace_strict(metrics.FITBIT_SLEEP_STAGES, default=None)
    stages = stages.filter(stage.is_not_null())

    start = pl.col("dateTime").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", time_unit="us", strict=False)
    return _rows(
        stages,
        _CATEGORY + "SleepAnalysis",
        None,
        stage,
        start,
        start + pl.duration(seconds=pl.col("seconds")),
        "fitbit",
    )


def _read_sleep_score(directory: Path) -> pl.DataFrame:
    """The Fitbit sleep score and the restlessness fraction.

    Most of this file is skipped. The three component scores sum to the overall
    one, and deep_sleep_in_minutes and resting_heart_rate repeat what sleep.json
    and resting_heart_rate.json already said.
    """
    path = directory / "fitbit" / "sleep_score.csv"
    if not path.exists():
        return _empty()

    frame = pl.read_csv(path, infer_schema_length=0)
    # The trailing "Z" is not true. See the module docstring.
    start = (
        pl.col("timestamp")
        .str.slice(0, 19)
        .str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="us", strict=False)
    )

    parts = [
        _rows(frame, metric, "score", pl.col(column), start, start, "fitbit")
        for column, metric in _SLEEP_SCORES.items()
        if column in frame.columns
    ]
    if "restlessness" in frame.columns:
        parts.append(
            _rows(
                frame,
                "SleepRestlessness",
                "fraction",
                pl.col("restlessness"),
                start,
                start,
                "fitbit",
            )
        )
    return pl.concat(parts) if parts else _empty()


def _read_wellness(directory: Path) -> pl.DataFrame:
    """The daily self report.

    Zero on a 1 to 5 scale is out of range and means the question was skipped,
    so it becomes null. readiness runs 0 to 10 and keeps its zeros.
    """
    path = directory / "pmsys" / "wellness.csv"
    if not path.exists():
        return _empty()

    frame = pl.read_csv(path, infer_schema_length=0)
    start = timestamps.parse_iso8601("effective_time_frame")[1]

    parts = []
    for column, metric in _WELLNESS_SCALES.items():
        if column not in frame.columns:
            continue
        scale = pl.col(column).cast(pl.Float64, strict=False)
        skipped = pl.when(scale > 0).then(scale).otherwise(None)
        parts.append(_rows(frame, metric, "score", skipped, start, start, "pmsys"))

    if "readiness" in frame.columns:
        parts.append(_rows(frame, "Readiness", "score", pl.col("readiness"), start, start, "pmsys"))
    if "sleep_duration_h" in frame.columns:
        # The night before the report, so it is stamped as covering that day
        # rather than the moment the form was filled in.
        hours = pl.col("sleep_duration_h").cast(pl.Float64, strict=False)
        parts.append(
            _rows(
                frame,
                metrics.SLEEP_DURATION_DAILY,
                "hr",
                pl.when(hours > 0).then(hours).otherwise(None),
                start - _DAY,
                start,
                "pmsys",
            )
        )
    return pl.concat(parts) if parts else _empty()


def _read_srpe(directory: Path) -> pl.DataFrame:
    """Session rating of perceived exertion, and how long the session lasted."""
    path = directory / "pmsys" / "srpe.csv"
    if not path.exists():
        return _empty()

    frame = pl.read_csv(path, infer_schema_length=0)
    end = timestamps.parse_iso8601("end_date_time")[1]
    minutes = pl.col("duration_min").cast(pl.Float64, strict=False)
    start = end - pl.duration(minutes=minutes)

    return pl.concat(
        [
            _rows(
                frame,
                "PerceivedExertion",
                "score",
                pl.col("perceived_exertion"),
                start,
                end,
                "pmsys",
            ),
            _rows(frame, "TrainingDuration", "min", minutes, start, end, "pmsys"),
        ]
    )


def _injury_count(raw: str | None) -> int | None:
    """How many injuries one report lists.

    The column holds a Python dict written out as text, such as
    "{'left_foot': 'minor'}". Only the count is kept. The body parts are not
    read, so nothing about a person's body reaches the output.
    """
    if raw is None:
        return None
    try:
        parsed = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return None
    return len(parsed) if isinstance(parsed, dict) else None


def _read_injury(directory: Path) -> pl.DataFrame:
    path = directory / "pmsys" / "injury.csv"
    if not path.exists():
        return _empty()

    frame = pl.read_csv(path, infer_schema_length=0)
    start = timestamps.parse_iso8601("effective_time_frame")[1]
    counts = frame["injuries"].map_elements(_injury_count, return_dtype=pl.Int64)

    return _rows(
        frame.with_columns(counts.alias("count")),
        "InjuryCount",
        "count",
        pl.col("count"),
        start,
        start,
        "pmsys",
    )


def _read_reporting(directory: Path) -> pl.DataFrame:
    """The food and weight log.

    Only weight is read. The meals, alcohol and fluid columns are not kept, see
    DROPPED_PMDATA_METRICS. Weight is the reason this file matters: PMData has
    more weights in it than every other source put together.

    The date column is day first, which is why it is parsed rather than left to
    polars to guess. 06/11/2019 in this file is the sixth of November.
    """
    path = directory / "googledocs" / "reporting.csv"
    if not path.exists():
        return _empty()

    frame = pl.read_csv(path, infer_schema_length=0)
    start = (
        pl.col("date").str.strip_chars().str.to_datetime("%d/%m/%Y", time_unit="us", strict=False)
    )
    end = start + _DAY

    parts = []
    if "weight" in frame.columns:
        parts.append(
            _rows(
                frame,
                _QUANTITY + "BodyMass",
                "kg",
                pl.col("weight"),
                start,
                end,
                "googledocs",
            )
        )
    return pl.concat(parts) if parts else _empty()


def _finish(parts: list[pl.DataFrame], subject_id: str) -> pl.DataFrame:
    """Turn the per stream frames into the canonical schema."""
    frame = pl.concat([part for part in parts if part.height])
    value_num, value_str = split_value(pl.col("value"))

    local = pl.col("start_local")
    end_local = pl.col("end_local")
    offset = pl.lit(PMDATA_OFFSET_MINUTES, dtype=pl.Int32)
    start_utc = timestamps.to_utc(local, offset)
    end_utc = timestamps.to_utc(end_local, offset)

    metric = pl.col("metric")
    return conform(
        frame.lazy()
        .select(
            pl.lit(subject_id).alias("subject_id"),
            pl.lit("pmdata").alias("source_format"),
            metric.alias("metric"),
            pl.col("unit").alias("unit"),
            value_num,
            metrics.restore_value_prefix(metric, value_str).alias("value_str"),
            start_utc.alias("start_utc"),
            end_utc.alias("end_utc"),
            # PMData records no creation time. Left null rather than copied from
            # the start, which would claim the file said something it did not.
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_utc"),
            local.alias("start_local"),
            end_local.alias("end_local"),
            pl.col("source_dir")
            .replace_strict(_SOURCE_TYPE_BY_DIR, default="unknown")
            .alias("source_type"),
        )
        # A row with neither value fails validation, and every stream can
        # produce one from a blank cell in a CSV.
        .filter(pl.col("value_num").is_not_null() | pl.col("value_str").is_not_null())
        .collect()
    )


def source_names(path: Path) -> set[str]:
    """No device names appear anywhere in PMData, so there is nothing to classify."""
    return set()


def read(path: Path, subject_id: str, source_map: anonymize.SourceMap) -> pl.DataFrame:
    """Read one PMData subject directory into the canonical schema.

    source_map is unused. See _SOURCE_TYPE_BY_DIR.
    """
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory. A PMData subject is a folder of files.")

    parts = [_read_timeseries(path, spec) for spec in _TIMESERIES]
    parts += [
        _read_exercise_minutes(path),
        _read_sleep(path),
        _read_sleep_score(path),
        _read_wellness(path),
        _read_srpe(path),
        _read_injury(path),
        _read_reporting(path),
    ]

    if not any(part.height for part in parts):
        raise ValueError(f"{path} contains no readable PMData files")

    return _finish(parts, subject_id)
