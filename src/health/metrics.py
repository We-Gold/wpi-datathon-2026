"""Canonical metric names.

HealthKit identifiers are the vocabulary. The flat CSV strips the prefix and
writes "StepCount", so readers put it back. Anything without an HK prefix is not
a HealthKit metric.
"""

from __future__ import annotations

import polars as pl

QUANTITY_PREFIX = "HKQuantityTypeIdentifier"
CATEGORY_PREFIX = "HKCategoryTypeIdentifier"
VALUE_PREFIX = "HKCategoryValue"

# The only category type in the flat CSV. Everything else there is a quantity.
_CATEGORY_METRICS = frozenset({"SleepAnalysis"})

# The nested JSON ships sleep already rolled up to a daily total, while the
# other formats give per-stage intervals. Sharing a name would let one filter
# return both grains and sum them. The unit is canonicalized like any other, so
# the name does not carry one.
SLEEP_DURATION_DAILY = "SleepDurationDaily"

DERIVED_METRICS = frozenset({SLEEP_DURATION_DAILY})


# The flat CSV strips the prefix from category values the same way it strips it
# from metric names, so "AsleepCore" and "HKCategoryValueSleepAnalysisAsleepCore"
# are the same sleep stage. The prefix cannot be derived from the metric name:
# AudioExposureEvent spells its values HKCategoryValueEnvironmentalAudioExposure
# Event, and MindfulSession uses HKCategoryValueNotApplicable. So it is listed
# per metric, and a metric with no entry is left alone.
_VALUE_PREFIXES: dict[str, str] = {
    CATEGORY_PREFIX + "SleepAnalysis": VALUE_PREFIX + "SleepAnalysis",
}


def restore_prefix(metric: pl.Expr) -> pl.Expr:
    """Put the HK prefix back on a bare metric name. Prefixed names pass through."""
    return (
        pl.when(metric.str.starts_with("HK"))
        .then(metric)
        .when(metric.is_in(list(_CATEGORY_METRICS)))
        .then(pl.lit(CATEGORY_PREFIX) + metric)
        .otherwise(pl.lit(QUANTITY_PREFIX) + metric)
    )


def is_healthkit(name: str) -> bool:
    """Whether a canonical metric name came from HealthKit unmodified."""
    return name.startswith("HK")


def restore_value_prefix(metric: pl.Expr, value: pl.Expr) -> pl.Expr:
    """Put the HKCategoryValue prefix back on a bare category value.

    Values that already carry it, and metrics with no rule, pass through.
    """
    prefix = metric.replace_strict(_VALUE_PREFIXES, default=None, return_dtype=pl.String)
    bare = value.is_not_null() & ~value.str.starts_with(VALUE_PREFIX)
    return pl.when(bare & prefix.is_not_null()).then(prefix + value).otherwise(value)
