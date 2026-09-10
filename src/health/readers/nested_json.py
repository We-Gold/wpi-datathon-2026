"""Reader for the nested JSON export.

One object keyed by HealthKit type:

    {"HKQuantityTypeIdentifierStepCount": {"records": {"data": [...]},
                                           "sources": [...], "devices": [...]},
     "metadata": {...}}

Only the record lists are read. The sources and devices lists hold device names
verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from health import anonymize, metrics, timestamps
from health.schema import conform, split_value

# Not a HealthKit type. The file's own bookkeeping, skipped rather than parsed.
_METADATA_KEY = "metadata"

# Read explicitly, so an unexpected extra field is ignored rather than carried
# into the output.
_FIELDS = ("type", "source_name", "unit", "creation_date", "start_date", "end_date", "value")

_PREAGGREGATED_SLEEP_UNIT = "hours"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the per-metric record lists into one list of records."""
    rows: list[dict[str, Any]] = []
    for key, section in payload.items():
        if key == _METADATA_KEY or not isinstance(section, dict):
            continue
        data = section.get("records", {}).get("data", [])
        for record in data:
            rows.append({field: record.get(field) for field in _FIELDS})
    return rows


def source_names(path: Path) -> set[str]:
    """Distinct source_name values across every metric's record list."""
    with path.open("rb") as handle:
        payload = json.load(handle)

    return {
        record["source_name"]
        for record in _rows(payload)
        if isinstance(record.get("source_name"), str)
    }


def read(path: Path, subject_id: str, source_map: anonymize.SourceMap) -> pl.DataFrame:
    """Read one nested JSON export into the canonical schema."""
    with path.open("rb") as handle:
        payload = json.load(handle)

    rows = _rows(payload)
    if not rows:
        raise ValueError(f"{path} contains no records")

    # Every field is held as text so the value column can carry both numbers and
    # sleep stage names, the same way the other formats do.
    raw = pl.DataFrame(
        rows,
        schema={field: pl.String for field in _FIELDS},
        strict=False,
        orient="row",
    )

    mapping = source_map.mapping(raw["source_name"].unique().to_list())

    start_utc, start_local = timestamps.parse_iso8601("start_date")
    end_utc, end_local = timestamps.parse_iso8601("end_date")
    created_utc, _ = timestamps.parse_iso8601("creation_date")

    value_num, value_str = split_value(pl.col("value"))

    # Sleep rolled up before export is a different variable, so it gets its own
    # name. The unit check avoids catching per-stage sleep if one ever appears.
    is_preaggregated_sleep = pl.col("type").str.contains("SleepAnalysis") & (
        pl.col("unit").str.to_lowercase() == _PREAGGREGATED_SLEEP_UNIT
    )
    metric = (
        pl.when(is_preaggregated_sleep)
        .then(pl.lit(metrics.SLEEP_DURATION_DAILY))
        .otherwise(metrics.restore_prefix(pl.col("type")))
    )

    parsed = raw.lazy().select(
        pl.lit(subject_id).alias("subject_id"),
        pl.lit("nested_json").alias("source_format"),
        metric.alias("metric"),
        pl.col("unit").alias("unit"),
        value_num,
        value_str,
        start_utc.alias("start_utc"),
        end_utc.alias("end_utc"),
        created_utc.alias("created_utc"),
        start_local.alias("start_local"),
        end_local.alias("end_local"),
        anonymize.source_type_expr("source_name", mapping),
    )

    return conform(parsed.collect())
