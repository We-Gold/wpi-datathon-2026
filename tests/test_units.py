"""Unit canonicalization."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from health import units
from health.schema import SCHEMA, conform


def _frame(rows: list[tuple[str, str | None, float | None, str | None]]) -> pl.DataFrame:
    """A frame of (metric, unit, value_num, value_str) rows, everything else fixed."""
    base: dict[str, object] = {
        "subject_id": "subject_01",
        "source_format": "apple_xml",
        "start_utc": dt.datetime(2024, 1, 2, 7, 0, tzinfo=dt.UTC),
        "end_utc": dt.datetime(2024, 1, 2, 7, 15, tzinfo=dt.UTC),
        "created_utc": dt.datetime(2024, 1, 2, 7, 30, tzinfo=dt.UTC),
        "start_local": dt.datetime(2024, 1, 2, 12, 30),
        "end_local": dt.datetime(2024, 1, 2, 12, 45),
        "source_type": "wearable",
    }
    records = [
        {**base, "metric": metric, "unit": unit, "value_num": num, "value_str": text}
        for metric, unit, num, text in rows
    ]
    return conform(pl.DataFrame(records, schema_overrides=dict(SCHEMA)))


def _converted(unit: str, value: float) -> tuple[str | None, float | None]:
    result = units.canonicalize(_frame([("HKQuantityTypeIdentifierStepCount", unit, value, None)]))
    return result["unit"].cast(pl.String)[0], result["value_num"][0]


class TestConversions:
    @pytest.mark.parametrize(
        ("unit", "value", "expected_unit", "expected_value"),
        [
            ("mi", 1.0, "m", 1609.344),
            ("km", 1.0, "m", 1000.0),
            ("cm", 180.0, "m", 1.8),
            ("in", 12.0, "m", 0.3048),
            ("ft", 1.0, "m", 0.3048),
            ("lb", 1.0, "kg", 0.45359237),
            ("mi/hr", 1.0, "m/s", 0.44704),
            ("km/hr", 3.6, "m/s", 1.0),
            ("min", 30.0, "s", 1800.0),
            ("hours", 8.0, "s", 28800.0),
            ("hr", 8.0, "s", 28800.0),
            ("ms", 50.0, "s", 0.05),
        ],
    )
    def test_values_scale(
        self, unit: str, value: float, expected_unit: str, expected_value: float
    ) -> None:
        got_unit, got_value = _converted(unit, value)
        assert got_unit == expected_unit
        assert got_value == pytest.approx(expected_value)

    def test_apple_calories_are_kilocalories(self) -> None:
        """Apple writes the large calorie, so Cal to kcal is a rename, not a factor."""
        assert _converted("Cal", 250.0) == ("kcal", 250.0)

    def test_percentages_keep_their_decimal_value(self) -> None:
        """Percent metrics are recorded as decimals, so only the label is wrong."""
        assert _converted("%", 0.95) == ("fraction", 0.95)

    def test_nutrient_grams_are_left_alone(self) -> None:
        """No metric mixes g with kg, so grams stay readable rather than becoming 0.045."""
        assert _converted("g", 45.0) == ("g", 45.0)

    def test_the_two_subjects_end_up_comparable(self) -> None:
        """The point of the whole step: a mile and a kilometre become one column."""
        frame = _frame(
            [
                ("HKQuantityTypeIdentifierDistanceWalkingRunning", "mi", 1.0, None),
                ("HKQuantityTypeIdentifierDistanceWalkingRunning", "km", 1.0, None),
            ]
        )
        result = units.canonicalize(frame)
        assert result["unit"].cast(pl.String).unique().to_list() == ["m"]


class TestPassthrough:
    def test_category_rows_survive(self) -> None:
        """Sleep stages have no unit and no number. Nothing should touch them."""
        frame = _frame([("HKCategoryTypeIdentifierSleepAnalysis", None, None, "asleepDeep")])
        result = units.canonicalize(frame)

        assert result["unit"][0] is None
        assert result["value_num"][0] is None
        assert result["value_str"].cast(pl.String)[0] == "asleepDeep"

    def test_schema_is_unchanged(self) -> None:
        frame = _frame([("HKQuantityTypeIdentifierStepCount", "count", 412.0, None)])
        assert units.canonicalize(frame).schema == frame.schema


class TestUnknownUnits:
    def test_an_unlisted_unit_stops_the_run(self) -> None:
        """Fail closed. Passing it through would silently mix two scales."""
        frame = _frame([("HKQuantityTypeIdentifierBodyMass", "stone", 11.0, None)])
        with pytest.raises(units.UnknownUnit, match="stone"):
            units.canonicalize(frame)

    def test_every_target_is_itself_a_known_unit(self) -> None:
        """Canonicalizing twice has to be a no-op, or the table has a cycle in it."""
        assert set(units._CANONICAL) >= units.CANONICAL_UNITS
        for unit in units.CANONICAL_UNITS:
            assert units._CANONICAL[unit] == (unit, 1.0)
