from __future__ import annotations

from datetime import date

import polars as pl

from health.forecast import (
    build_forecast_table,
    evaluate_forecasters,
    temporal_forecast_split,
)


def _daily() -> pl.DataFrame:
    days = pl.date_range(date(2026, 1, 1), date(2026, 2, 15), eager=True).to_list()
    return pl.DataFrame(
        {
            "subject_id": ["one"] * len(days) + ["two"] * len(days),
            "date": days * 2,
            "steps": list(range(len(days))) * 2,
            "sleep_hours": [7.0 + (index % 3) * 0.1 for index in range(len(days))] * 2,
            "hr_mean_bpm": [65.0 + index * 0.05 for index in range(len(days))] * 2,
            "activity_score": [index / 10 for index in range(len(days))] * 2,
            "recovery_score": [1 - index / 20 for index in range(len(days))] * 2,
        }
    ).with_columns(pl.col("subject_id").cast(pl.Categorical))


def test_forecast_table_uses_calendar_lags_and_future_targets() -> None:
    table = build_forecast_table(_daily(), lags=(0, 1, 2), horizon=1)
    # 46 days per subject become 45 one-day-ahead examples.
    assert table.frame.height == 45 * 2
    assert "steps__lag2" in table.input_columns
    assert table.frame["target_date"].min() == date(2026, 1, 2)


def test_forecast_evaluation_is_temporal_and_returns_three_baselines() -> None:
    table = build_forecast_table(_daily(), lags=(0, 1, 2, 3), horizon=1)
    train, test = temporal_forecast_split(table, test_fraction=0.2)
    report, predictions = evaluate_forecasters(train, test, table)
    assert train["target_date"].max() < test["target_date"].min()
    assert set(predictions.columns) >= {
        "persistence__activity_score",
        "rolling__recovery_score",
        "ridge__activity_score",
    }
    assert set(report["targets"]["activity_score"]) == {"persistence", "rolling", "ridge"}
