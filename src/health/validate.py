"""Checks that run before anything is written.

Schema invariants, plus leakage: the output goes to other people, so a name that
slips through classification has to stop the run rather than land in a file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

from health import metrics
from health.schema import COLUMNS, SOURCE_TYPES

# How names are spelled in these exports. A possessive in any output column
# means a source string survived classification.
_POSSESSIVE = re.compile("['\u2019]s\\b", re.IGNORECASE)


_SUBJECT_ID = re.compile(r"^subject_[0-9a-z_]+$")

# Columns whose full distinct set is expected to stay small. If one of these
# grows unbounded, something per-record is leaking into it.
_BOUNDED_COLUMNS = ("subject_id", "source_format", "metric", "unit", "source_type")
_MAX_DISTINCT = 500


@dataclass(frozen=True)
class Finding:
    check: str
    detail: str


def _distinct_strings(frame: pl.DataFrame, column: str) -> list[str]:
    return [
        value for value in frame[column].cast(pl.String).unique().to_list() if value is not None
    ]


def check(frame: pl.DataFrame) -> list[Finding]:
    """Every problem found, rather than the first."""
    findings: list[Finding] = []

    if frame.columns != COLUMNS:
        findings.append(Finding("schema", f"columns are {frame.columns}, expected {COLUMNS}"))
        return findings

    findings.extend(_check_values(frame))
    findings.extend(_check_metrics(frame))
    findings.extend(_check_leakage(frame))
    return findings


def _check_values(frame: pl.DataFrame) -> list[Finding]:
    findings: list[Finding] = []

    both = frame.filter(pl.col("value_num").is_not_null() & pl.col("value_str").is_not_null())
    if both.height:
        findings.append(Finding("value", f"{both.height} rows set both value_num and value_str"))

    neither = frame.filter(pl.col("value_num").is_null() & pl.col("value_str").is_null())
    if neither.height:
        findings.append(Finding("value", f"{neither.height} rows set neither value column"))

    for column in ("subject_id", "source_format", "source_type", "metric"):
        nulls = frame[column].null_count()
        if nulls:
            findings.append(Finding("null", f"{column} has {nulls} nulls"))

    return findings


def _check_metrics(frame: pl.DataFrame) -> list[Finding]:
    findings: list[Finding] = []

    unexpected = [
        name
        for name in _distinct_strings(frame, "metric")
        if not metrics.is_healthkit(name) and name not in metrics.DERIVED_METRICS
    ]
    if unexpected:
        findings.append(
            Finding(
                "metric",
                f"names that are neither HealthKit identifiers nor registered "
                f"derived metrics: {sorted(unexpected)}",
            )
        )

    # Category values are the other vocabulary two formats can disagree on, the
    # same way they disagree on units. One format writing a bare "AsleepCore"
    # would split every sleep stage into two buckets, which is invisible in a
    # group-by, so it stops the run instead.
    bare_values = [
        name
        for name in _distinct_strings(frame, "value_str")
        if not name.startswith(metrics.VALUE_PREFIX)
    ]
    if bare_values:
        findings.append(
            Finding(
                "value_str",
                f"category values without the {metrics.VALUE_PREFIX} prefix: "
                f"{sorted(bare_values)[:5]}",
            )
        )

    bad_types = set(_distinct_strings(frame, "source_type")) - set(SOURCE_TYPES)
    if bad_types:
        findings.append(Finding("source_type", f"values outside the schema: {sorted(bad_types)}"))

    bad_subjects = [s for s in _distinct_strings(frame, "subject_id") if not _SUBJECT_ID.match(s)]
    if bad_subjects:
        findings.append(
            Finding(
                "subject_id",
                f"not pseudonyms of the form subject_NN: {sorted(bad_subjects)}",
            )
        )

    return findings


def _check_leakage(frame: pl.DataFrame) -> list[Finding]:
    findings: list[Finding] = []

    text_columns = [
        name
        for name, dtype in frame.schema.items()
        if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum)
    ]

    for column in text_columns:
        values = _distinct_strings(frame, column)

        possessives = [value for value in values if _POSSESSIVE.search(value)]
        if possessives:
            findings.append(
                Finding(
                    "leak",
                    f"{column} holds possessive strings, which is how names appear "
                    f"in these exports: {sorted(possessives)[:5]}",
                )
            )

        if column in _BOUNDED_COLUMNS and len(values) > _MAX_DISTINCT:
            findings.append(
                Finding(
                    "leak",
                    f"{column} has {len(values)} distinct values, over the {_MAX_DISTINCT} "
                    f"expected. A per-record value may be leaking into it.",
                )
            )

    return findings
