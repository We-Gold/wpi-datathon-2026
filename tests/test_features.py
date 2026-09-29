from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from health import metrics
from health.features import (
    add_interpretable_axes,
    add_smoothed_features,
    build_daily_features,
    select_dense_windows,
)


def _records(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("subject_id").cast(pl.Categorical),
        pl.col("metric").cast(pl.Categorical),
        pl.col("value_str").cast(pl.Categorical),
    )


def test_build_daily_features_uses_local_day_and_sleep_stages() -> None:
    start = datetime(2026, 1, 2, 23)
    rows: list[dict[str, object]] = [
        {
            "subject_id": "one",
            "metric": metrics.QUANTITY_PREFIX + "StepCount",
            "value_num": 100.0,
            "value_str": None,
            "start_local": datetime(2026, 1, 2, 8),
            "end_local": datetime(2026, 1, 2, 8),
        },
        {
            "subject_id": "one",
            "metric": metrics.QUANTITY_PREFIX + "StepCount",
            "value_num": 250.0,
            "value_str": None,
            "start_local": datetime(2026, 1, 2, 12),
            "end_local": datetime(2026, 1, 2, 12),
        },
        {
            "subject_id": "one",
            "metric": metrics.CATEGORY_PREFIX + "SleepAnalysis",
            "value_num": None,
            "value_str": metrics.VALUE_PREFIX + "SleepAnalysisAsleepCore",
            "start_local": start,
            "end_local": start + timedelta(hours=6),
        },
        {
            "subject_id": "one",
            "metric": metrics.CATEGORY_PREFIX + "SleepAnalysis",
            "value_num": None,
            "value_str": metrics.VALUE_PREFIX + "SleepAnalysisAwake",
            "start_local": start + timedelta(hours=6),
            "end_local": start + timedelta(hours=7),
        },
    ]
    daily = build_daily_features(_records(rows))
    day = daily.filter(pl.col("date") == date(2026, 1, 2)).row(0, named=True)
    assert day["steps"] == 350
    assert day["sleep_hours"] == 6
    assert day["sleep_efficiency"] == pytest.approx(6 / 7)


def test_overlapping_sleep_sources_are_not_double_counted() -> None:
    start = datetime(2026, 1, 2, 23)

    def sleep(stage: str, begin: float, end: float) -> dict[str, object]:
        return {
            "subject_id": "one",
            "metric": metrics.CATEGORY_PREFIX + "SleepAnalysis",
            "value_num": None,
            "value_str": metrics.VALUE_PREFIX + "SleepAnalysis" + stage,
            "start_local": start + timedelta(hours=begin),
            "end_local": start + timedelta(hours=end),
        }

    rows = [
        # A watch's stages for the night.
        sleep("AsleepCore", 0, 3),
        sleep("AsleepDeep", 3, 4),
        sleep("Awake", 4, 4.5),
        sleep("AsleepREM", 4.5, 7),
        # A phone app records the same night as one block, and an hour later.
        sleep("AsleepUnspecified", 0.5, 7.5),
    ]
    day = build_daily_features(_records(rows)).row(0, named=True)
    # The union of asleep time is 0 to 7.5 hours. The watch's awake half hour
    # is inside the app's asleep block, so it counts as asleep once.
    assert day["sleep_hours"] == pytest.approx(7.5)
    assert day["sleep_efficiency"] is None


def test_dense_window_finds_feature_rich_region() -> None:
    days = pl.date_range(date(2026, 1, 1), date(2026, 1, 12), eager=True)
    daily = pl.DataFrame(
        {
            "subject_id": ["one"] * 12,
            "date": days,
            "a": [None, None, 1, 1, 1, 1, 1, 1, 1, None, None, None],
            "b": [None, None, 1, 1, 1, 1, 1, 1, 1, None, None, None],
        }
    ).with_columns(pl.col("subject_id").cast(pl.Categorical))
    window = select_dense_windows(daily, ("a", "b"), minimum_days=3, minimum_completeness=1.0)[0]
    assert window.start == date(2026, 1, 3)
    assert window.end == date(2026, 1, 9)
    assert window.mean_completeness == 1
    assert window.meets_threshold


def test_dense_heart_rate_gets_bounded_spectral_features() -> None:
    start = datetime(2026, 1, 2)
    rows: list[dict[str, object]] = []
    for index in range(96):
        timestamp = start + timedelta(minutes=15 * index)
        rows.append(
            {
                "subject_id": "one",
                "metric": metrics.QUANTITY_PREFIX + "HeartRate",
                "value_num": 65 + 10 * math.sin(2 * math.pi * index / 96),
                "value_str": None,
                "start_local": timestamp,
                "end_local": timestamp,
            }
        )
    day = build_daily_features(_records(rows)).row(0, named=True)
    assert day["hr_samples"] == 96
    assert day["hr_p10_bpm"] < day["hr_mean_bpm"]
    assert day["hr_daily_cv"] == pytest.approx(day["hr_std_bpm"] / day["hr_mean_bpm"])
    assert 0 <= day["hr_spectral_entropy"] <= 1
    assert 0 <= day["hr_low_frequency_power"] <= 1


def test_smoothing_is_trailing_and_kept_separate_from_raw_values() -> None:
    dates = pl.date_range(date(2026, 1, 1), date(2026, 1, 28), eager=True).to_list()
    daily = pl.DataFrame(
        {
            "subject_id": ["one"] * 28 + ["two"] * 28,
            "date": dates * 2,
            "resting_hr_bpm": [float(value) for value in range(1, 29)] + [100.0] * 28,
        }
    ).with_columns(pl.col("subject_id").cast(pl.Categorical))
    result = add_smoothed_features(daily, ("resting_hr_bpm",))

    one = result.filter(pl.col("subject_id") == "one")
    assert one.item(1, "resting_hr_bpm_7d") is None
    assert one.item(2, "resting_hr_bpm_7d") == 2
    assert one.item(27, "resting_hr_bpm") == 28
    assert one.item(27, "resting_hr_bpm_7d") == 25
    assert one.item(27, "resting_hr_bpm_28d") == 14.5
    assert one.item(27, "resting_hr_change_bpm") == 10.5

    two = result.filter(pl.col("subject_id") == "two")
    assert two.item(2, "resting_hr_bpm_7d") == 100


def test_axes_are_directional_and_report_coverage() -> None:
    daily = pl.DataFrame(
        {
            "subject_id": ["one"] * 3,
            "date": pl.date_range(date(2026, 1, 1), date(2026, 1, 3), eager=True),
            "steps": [1000.0, 2000.0, 3000.0],
            "sedentary_hours": [10.0, 8.0, 6.0],
            "sleep_hours": [6.0, 7.0, 8.0],
            "resting_hr_bpm": [70.0, 65.0, 60.0],
        }
    ).with_columns(pl.col("subject_id").cast(pl.Categorical))
    result = add_interpretable_axes(daily)
    assert result["activity_score"].to_list() == [-1.0, 0.0, 1.0]
    assert result["recovery_score"].to_list() == [-1.0, 0.0, 1.0]
    assert result["activity_coverage"].to_list() == [1.0, 1.0, 1.0]
    assert result["recovery_coverage"].to_list() == [1.0, 1.0, 1.0]
