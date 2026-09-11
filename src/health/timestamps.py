"""Parsing the three timestamp encodings into a UTC instant and a local wall clock.

    apple_xml    "2017-01-27 16:09:39 -0400"
    flat_csv     "2023-06-13" + "09:25:02 PM" + "+0530"  (separate columns)
    nested_json  "2024-06-09T09:04:10+03:00"
    pmdata       "2019-11-01 00:00:00"                   (no offset at all)

The naive part is parsed as local time and the instant derived from it, so the
wall clock is whatever the file literally said.

The first three carry their own offset. PMData does not, so its reader supplies
one and calls to_utc directly. See health/readers/pmdata.py.
"""

from __future__ import annotations

import polars as pl

from health.schema import TIME_UNIT

# Offsets appear both as "+0530" and as "+05:30". Stripping the colon lets one
# set of slices handle both.


def offset_minutes(offset: pl.Expr) -> pl.Expr:
    """Minutes east of UTC from a "+0530", "-0400", or "+05:30" style offset.

    Null for anything malformed, so a bad row loses its timestamps rather than
    landing in the wrong hour.
    """
    clean = offset.str.strip_chars().str.replace_all(":", "")
    sign = pl.when(clean.str.slice(0, 1) == "-").then(-1).otherwise(1)
    hours = clean.str.slice(1, 2).cast(pl.Int32, strict=False)
    mins = clean.str.slice(3, 2).cast(pl.Int32, strict=False)

    valid = clean.str.contains(r"^[+-]\d{4}$")
    return pl.when(valid).then(sign * (hours * 60 + mins)).otherwise(None).cast(pl.Int32)


def to_utc(local: pl.Expr, offset_min: pl.Expr) -> pl.Expr:
    """The UTC instant for a local wall clock at a given offset."""
    return (local - pl.duration(minutes=offset_min)).dt.replace_time_zone("UTC")


def parse_apple_xml(column: str) -> tuple[pl.Expr, pl.Expr]:
    """Parse "2017-01-27 16:09:39 -0400" into (utc, local)."""
    text = pl.col(column).str.strip_chars()
    local = text.str.slice(0, 19).str.to_datetime(
        "%Y-%m-%d %H:%M:%S", time_unit=TIME_UNIT, strict=False
    )
    offset = offset_minutes(text.str.slice(-5, 5))
    return to_utc(local, offset), local


def parse_flat_csv(date_column: str, time_column: str, tz_column: str) -> tuple[pl.Expr, pl.Expr]:
    """Parse the CSV's split date, 12-hour time, and offset columns into (utc, local).

    The export's separate AM/PM column repeats the marker already in the time
    string, so it is ignored rather than trusted to agree.
    """
    joined = pl.col(date_column).str.strip_chars() + " " + pl.col(time_column).str.strip_chars()
    local = joined.str.to_datetime("%Y-%m-%d %I:%M:%S %p", time_unit=TIME_UNIT, strict=False)
    offset = offset_minutes(pl.col(tz_column))
    return to_utc(local, offset), local


def parse_iso8601(column: str) -> tuple[pl.Expr, pl.Expr]:
    """Parse "2024-06-09T09:04:10+03:00" into (utc, local). Trailing "Z" is UTC."""
    text = pl.col(column).str.strip_chars()
    local = text.str.slice(0, 19).str.to_datetime(
        "%Y-%m-%dT%H:%M:%S", time_unit=TIME_UNIT, strict=False
    )
    tail = text.str.slice(19)
    offset = offset_minutes(pl.when(tail == "Z").then(pl.lit("+0000")).otherwise(tail))
    return to_utc(local, offset), local
