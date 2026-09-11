"""The canonical schema every reader emits. One row is one observation."""

from __future__ import annotations

import polars as pl

# New formats go on the end. This list is a polars Enum and its order is part
# of the dtype, so adding one in the middle changes the sort order of the
# source_format column in every file already written.
SOURCE_FORMATS = ["apple_xml", "flat_csv", "nested_json", "pmdata"]
SOURCE_TYPES = ["wearable", "phone", "scale", "app", "manual", "derived", "unknown"]

TIME_UNIT = "us"

# polars takes a dtype either as an instance, pl.Datetime("us"), or as the class
# itself, pl.Float64. Both appear below.
PolarsDataType = pl.DataType | type[pl.DataType]

SCHEMA: dict[str, PolarsDataType] = {
    "subject_id": pl.Categorical,
    "source_format": pl.Enum(SOURCE_FORMATS),
    "metric": pl.Categorical,
    # Canonical, not as recorded. One export writes cm where another writes ft,
    # so health.units converts both before anything is written.
    "unit": pl.Categorical,
    # Exactly one is non-null per row: quantity types are numeric, category
    # types like sleep stages are strings.
    "value_num": pl.Float64,
    "value_str": pl.Categorical,
    "start_utc": pl.Datetime(TIME_UNIT, "UTC"),
    "end_utc": pl.Datetime(TIME_UNIT, "UTC"),
    "created_utc": pl.Datetime(TIME_UNIT, "UTC"),
    # Subjects span -0400 to +0530, so day boundaries drawn in UTC would be off
    # by up to five and a half hours.
    "start_local": pl.Datetime(TIME_UNIT),
    "end_local": pl.Datetime(TIME_UNIT),
    "source_type": pl.Enum(SOURCE_TYPES),
}

COLUMNS = list(SCHEMA)


def empty_frame() -> pl.DataFrame:
    """An empty frame with the canonical schema."""
    return pl.DataFrame(schema=SCHEMA)


def conform(frame: pl.DataFrame) -> pl.DataFrame:
    """Put a reader's output into canonical column order and dtypes."""
    missing = [name for name in COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"reader output is missing columns: {missing}")

    extra = [name for name in frame.columns if name not in SCHEMA]
    if extra:
        raise ValueError(f"reader output has unexpected columns: {extra}")

    return frame.select([pl.col(name).cast(dtype) for name, dtype in SCHEMA.items()])


def split_value(value: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Split a raw value column into the numeric and categorical columns."""
    text = value.cast(pl.String).str.strip_chars()
    present = text.is_not_null() & (text.str.len_chars() > 0)

    numeric = text.cast(pl.Float64, strict=False)
    is_numeric = present & numeric.is_not_null()

    value_num = pl.when(is_numeric).then(numeric).otherwise(None)
    value_str = pl.when(present & ~is_numeric).then(text).otherwise(None)
    return value_num.alias("value_num"), value_str.alias("value_str")
