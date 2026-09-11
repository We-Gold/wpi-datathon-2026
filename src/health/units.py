"""Unit canonicalization.

Runs after the readers and before validation, on one file at a time. Every
value is converted in place and the recorded unit is dropped, so two subjects
can be compared without first checking which phone locale wrote the export.

The table is an allowlist, the same shape as source classification. A unit that
is not listed stops the run instead of passing through, so the first export
that arrives in an unfamiliar unit gets noticed rather than quietly mixing two
scales into one column.

SI where it helps. Length, speed and duration become m, m/s and s. Energy stays
in kcal and heart rate in count/min, because joules and hertz are the SI forms
and no one working with health data uses either.
"""

from __future__ import annotations

import polars as pl

from health.schema import conform

# Recorded unit -> the unit this pipeline uses, and what to multiply the value
# by to get there.
_CANONICAL: dict[str, tuple[str, float]] = {
    # Length.
    "m": ("m", 1.0),
    "cm": ("m", 0.01),
    "in": ("m", 0.0254),
    "ft": ("m", 0.3048),
    "km": ("m", 1000.0),
    "mi": ("m", 1609.344),
    # Speed.
    "m/s": ("m/s", 1.0),
    "ft/s": ("m/s", 0.3048),
    "km/hr": ("m/s", 1.0 / 3.6),
    "mi/hr": ("m/s", 0.44704),
    # Duration. Two exports spell the same hour as "hr" and "hours".
    "s": ("s", 1.0),
    "ms": ("s", 0.001),
    "min": ("s", 60.0),
    "hr": ("s", 3600.0),
    "hours": ("s", 3600.0),
    # Mass. Body mass arrives in kg or lb, nutrients in g or mg. No metric uses
    # both scales, so the small units keep their own rather than turning a
    # 45 gram carbohydrate reading into 0.045.
    "kg": ("kg", 1.0),
    "lb": ("kg", 0.45359237),
    "g": ("g", 1.0),
    "mg": ("mg", 1.0),
    "mL": ("mL", 1.0),
    # Energy. Apple writes "Cal" for the large calorie, which is the
    # kilocalorie, so this is a rename and not a factor of a thousand.
    "kcal": ("kcal", 1.0),
    "Cal": ("kcal", 1.0),
    # Dimensionless. Percentages are already recorded as decimals between 0 and
    # 1, so the name is corrected and the value left alone.
    "%": ("fraction", 1.0),
    "fraction": ("fraction", 1.0),
    "count": ("count", 1.0),
    # A point on a rating scale. Not a count, because the numbers are labels
    # with an order, and each metric uses its own scale. Do not sum across
    # metrics.
    "score": ("score", 1.0),
    # Clinical composites with no SI form anyone would want to read.
    "count/min": ("count/min", 1.0),
    "kcal/hr·kg": ("kcal/hr·kg", 1.0),
    "mL/min·kg": ("mL/min·kg", 1.0),
    "W": ("W", 1.0),
    "dBASPL": ("dBASPL", 1.0),
}

CANONICAL_UNITS = frozenset(unit for unit, _ in _CANONICAL.values())


class UnknownUnit(ValueError):
    """A unit the table has no rule for. Add one rather than guessing."""


def recorded_units(frame: pl.DataFrame) -> list[str]:
    """The non-null units present in a frame."""
    return [unit for unit in frame["unit"].cast(pl.String).unique().to_list() if unit is not None]


def canonicalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert every value to its canonical unit.

    Category rows such as sleep stages carry no unit and no numeric value. They
    pass through untouched.
    """
    present = recorded_units(frame)

    unknown = sorted(set(present) - set(_CANONICAL))
    if unknown:
        raise UnknownUnit(
            f"no conversion rule for {unknown}. Add them to health/units.py, "
            f"mapping each to one of {sorted(CANONICAL_UNITS)}."
        )

    names = {unit: _CANONICAL[unit][0] for unit in present}
    factors = {unit: _CANONICAL[unit][1] for unit in present}

    recorded = pl.col("unit").cast(pl.String)

    return conform(
        frame.with_columns(
            # A null unit means a category row, which has no value to scale.
            (
                pl.col("value_num")
                * recorded.replace_strict(factors, default=1.0, return_dtype=pl.Float64).fill_null(
                    1.0
                )
            ).alias("value_num"),
            recorded.replace_strict(names, default=None, return_dtype=pl.String).alias("unit"),
        )
    )
