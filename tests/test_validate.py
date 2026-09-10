"""The gate that runs before anything is written."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from health import validate
from health.schema import SCHEMA, conform


def _row(**overrides: object) -> pl.DataFrame:
    base: dict[str, object] = {
        "subject_id": "subject_01",
        "source_format": "apple_xml",
        "metric": "HKQuantityTypeIdentifierStepCount",
        "unit": "count",
        "value_num": 412.0,
        "value_str": None,
        "start_utc": dt.datetime(2024, 1, 2, 7, 0, tzinfo=dt.UTC),
        "end_utc": dt.datetime(2024, 1, 2, 7, 15, tzinfo=dt.UTC),
        "created_utc": dt.datetime(2024, 1, 2, 7, 30, tzinfo=dt.UTC),
        "start_local": dt.datetime(2024, 1, 2, 12, 30),
        "end_local": dt.datetime(2024, 1, 2, 12, 45),
        "source_type": "wearable",
    }
    base.update(overrides)
    return conform(pl.DataFrame([base], schema_overrides=dict(SCHEMA)))


def _checks(frame: pl.DataFrame) -> set[str]:
    return {finding.check for finding in validate.check(frame)}


class TestAcceptsGoodData:
    def test_a_clean_row_passes(self) -> None:
        assert validate.check(_row()) == []

    def test_registered_derived_metric_passes(self) -> None:
        frame = _row(metric="SleepDurationDaily", unit="s")
        assert validate.check(frame) == []


class TestValueInvariant:
    def test_both_value_columns_set_is_a_finding(self) -> None:
        assert "value" in _checks(_row(value_str="AsleepCore"))

    def test_neither_value_column_set_is_a_finding(self) -> None:
        assert "value" in _checks(_row(value_num=None))


class TestLeakage:
    @pytest.mark.parametrize("column", ["unit", "metric", "subject_id"])
    def test_possessive_string_anywhere_is_caught(self, column: str) -> None:
        # This is how names actually appear in these exports, so it is the
        # signature worth catching regardless of which column it lands in.
        findings = _checks(_row(**{column: "Person\u2019s Apple Watch"}))
        assert "leak" in findings

    def test_straight_apostrophe_possessive_is_caught(self) -> None:
        assert "leak" in _checks(_row(unit="Person's Watch"))


class TestSchemaAndNames:
    def test_unregistered_non_healthkit_metric_is_a_finding(self) -> None:
        assert "metric" in _checks(_row(metric="StepCount"))

    def test_subject_id_that_is_not_a_pseudonym_is_a_finding(self) -> None:
        assert "subject_id" in _checks(_row(subject_id="weaver"))

    def test_missing_column_is_reported_without_cascading(self) -> None:
        frame = _row().drop("source_type")
        findings = validate.check(frame)
        # One clear finding rather than every name-based check failing after it.
        assert [finding.check for finding in findings] == ["schema"]


class TestCategoryValues:
    def test_a_bare_category_value_is_refused(self) -> None:
        """One format writing "AsleepCore" beside another's prefixed spelling
        splits every sleep stage in two, and no group-by would show it."""
        frame = _row(
            metric="HKCategoryTypeIdentifierSleepAnalysis",
            unit=None,
            value_num=None,
            value_str="AsleepCore",
        )
        assert "value_str" in _checks(frame)

    def test_the_prefixed_spelling_passes(self) -> None:
        frame = _row(
            metric="HKCategoryTypeIdentifierSleepAnalysis",
            unit=None,
            value_num=None,
            value_str="HKCategoryValueSleepAnalysisAsleepCore",
        )
        assert validate.check(frame) == []
