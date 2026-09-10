"""Reader for the flattened CSV export.

This format splits every timestamp across four columns, strips the HealthKit
prefix from metric names, and mixes numeric and categorical values in one value
column.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from health import anonymize, metrics, timestamps
from health.schema import conform, split_value

# The export's own column names. Checked on read so a differently shaped file
# fails immediately instead of producing a frame full of nulls.
EXPECTED_COLUMNS = frozenset(
    {
        "type",
        "sourceName",
        "sourceVersion",
        "unit",
        "m_creationDate",
        "m_creationTime",
        "m_creationTimeZone",
        "m_startDate",
        "m_startTime",
        "m_startTimeZone",
        "m_endDate",
        "m_endTime",
        "m_endTimeZone",
        "value",
    }
)


def source_names(path: Path) -> set[str]:
    """Distinct sourceName values, without parsing the rest of the file."""
    distinct = (
        pl.scan_csv(path, infer_schema_length=0)
        # Same repeated-header guard the reader applies, or scan reports the
        # literal column name as a source needing classification.
        .filter(pl.col("type") != "type")
        .select(pl.col("sourceName").unique())
        .collect()["sourceName"]
    )
    return {name for name in distinct.to_list() if name is not None}


def read(path: Path, subject_id: str, source_map: anonymize.SourceMap) -> pl.DataFrame:
    """Read one flattened CSV export into the canonical schema."""
    # Every column is read as text. The value column is deliberately mixed, and
    # letting polars infer types would either fail or coerce sleep stages away.
    frame = pl.scan_csv(path, infer_schema_length=0)

    found = set(frame.collect_schema().names())
    missing = EXPECTED_COLUMNS - found
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")

    # Guard against a header line repeated inside the data, which is what
    # concatenating two exports produces. The file at hand is clean, but more
    # are coming. No metric is ever the literal string "type".
    frame = frame.filter(pl.col("type") != "type")

    # Classification runs over the distinct source names first, so the mapping
    # is built from a few dozen values rather than a million rows.
    distinct = frame.select(pl.col("sourceName").unique()).collect()["sourceName"].to_list()
    mapping = source_map.mapping(distinct)

    start_utc, start_local = timestamps.parse_flat_csv(
        "m_startDate", "m_startTime", "m_startTimeZone"
    )
    end_utc, end_local = timestamps.parse_flat_csv("m_endDate", "m_endTime", "m_endTimeZone")
    created_utc, _ = timestamps.parse_flat_csv(
        "m_creationDate", "m_creationTime", "m_creationTimeZone"
    )

    value_num, value_str = split_value(pl.col("value"))

    # This format strips the HK prefix from both the metric name and the
    # category value, so both get it back here.
    metric = metrics.restore_prefix(pl.col("type"))

    parsed = frame.select(
        pl.lit(subject_id).alias("subject_id"),
        pl.lit("flat_csv").alias("source_format"),
        metric.alias("metric"),
        pl.col("unit").alias("unit"),
        value_num,
        metrics.restore_value_prefix(metric, value_str).alias("value_str"),
        start_utc.alias("start_utc"),
        end_utc.alias("end_utc"),
        created_utc.alias("created_utc"),
        start_local.alias("start_local"),
        end_local.alias("end_local"),
        anonymize.source_type_expr("sourceName", mapping),
    )

    return conform(parsed.collect())
