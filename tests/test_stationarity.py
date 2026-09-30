from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from health.stationarity import apply_differencing, check_stationarity


def test_check_stationarity_selects_zero_one_or_two_differences() -> None:
    random = np.random.default_rng(41)
    stationary = random.normal(size=500)
    first_order = np.cumsum(random.normal(size=500))
    second_order = np.cumsum(np.cumsum(random.normal(size=500)))

    assert check_stationarity(stationary).order == 0
    assert check_stationarity(first_order).order == 1
    assert check_stationarity(second_order).order == 2


def test_check_stationarity_leaves_short_series_unchanged() -> None:
    result = check_stationarity(np.arange(12, dtype=float))

    assert result.order == 0
    assert result.status == "insufficient_data"


def test_check_stationarity_does_not_join_separate_observation_runs() -> None:
    values = np.r_[np.arange(20, dtype=float), np.nan, np.arange(20, dtype=float)]

    result = check_stationarity(values)

    assert result.order == 0
    assert result.observations == 40
    assert result.status == "insufficient_contiguous_data"


def test_apply_differencing_does_not_cross_subject_boundaries() -> None:
    start = date(2026, 1, 1)
    daily = pl.DataFrame(
        {
            "subject_id": ["one"] * 3 + ["two"] * 3,
            "date": [start + timedelta(days=day) for day in range(3)] * 2,
            "score": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
        }
    )

    result = apply_differencing(daily, {"one": {"score": 1}, "two": {"score": 1}})

    assert result["score"].to_list() == [None, 1.0, 1.0, None, 1.0, 1.0]


def test_apply_differencing_inserts_missing_calendar_days() -> None:
    start = date(2026, 1, 1)
    daily = pl.DataFrame(
        {
            "subject_id": ["one", "one"],
            "date": [start, start + timedelta(days=2)],
            "score": [10.0, 12.0],
        }
    )

    result = apply_differencing(daily, {"one": {"score": 1}})

    assert result["date"].to_list() == [
        start,
        start + timedelta(days=1),
        start + timedelta(days=2),
    ]
    assert result["score"].to_list() == [None, None, None]