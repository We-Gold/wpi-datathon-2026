from __future__ import annotations

from datetime import date

import polars as pl

from health.forecast import (
    ForecastTable,
    baseline_predictions,
    build_forecast_table,
    evaluate_forecasters,
    temporal_forecast_split,
    undifference_forecasts,
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


def test_forecast_evaluation_is_temporal_and_returns_baselines() -> None:
    table = build_forecast_table(_daily(), lags=(0, 1, 2, 3), horizon=1)
    train, test = temporal_forecast_split(table, test_fraction=0.2)
    report, predictions = evaluate_forecasters(train, test, table)
    assert train["target_date"].max() < test["target_date"].min()
    assert set(predictions.columns) >= {
        "persistence__activity_score",
        "seasonal_persistence__activity_score",
        "rolling__recovery_score",
        "ridge__activity_score",
        "lightgbm__activity_score",
        "elasticnet__activity_score",
    }
    assert report["horizon_days"] == 1
    assert set(report["targets"]["activity_score"]) == {
        "persistence",
        "seasonal_persistence",
        "rolling",
        "ridge",
        "lightgbm",
        "elasticnet",
    }


def test_forecast_evaluation_supports_multi_day_horizon() -> None:
    table = build_forecast_table(_daily(), horizon=7)
    train, test = temporal_forecast_split(table)
    report, predictions = evaluate_forecasters(train, test, table)

    assert table.horizon == 7
    assert table.frame["target_date"].min() == date(2026, 1, 8)
    assert "activity_score__lag0" in table.frame.columns
    assert "seasonal_persistence__activity_score" in predictions.columns
    assert report["horizon_days"] == 7


def test_multi_day_baselines_use_values_available_at_origin() -> None:
    table = build_forecast_table(_daily(), horizon=3)
    frame = table.frame.filter(pl.col("subject_id") == "one").slice(10, 1)

    predictions = baseline_predictions(frame, table)

    assert predictions["persistence__activity_score"][0] == frame["activity_score__lag0"][0]
    assert predictions["seasonal_persistence__activity_score"][0] == frame[
        "activity_score__lag4"
    ][0]
    assert predictions["rolling__activity_score"][0] == sum(
        frame[f"activity_score__lag{lag}"][0] for lag in range(4)
    ) / 4


def test_undifference_forecasts_uses_only_origin_anchors() -> None:
    predictions = pl.DataFrame(
        {
            "subject_id": ["first", "second"],
            "target__score": [25.0, 40.0],
            "ridge__score": [2.0, 2.0],
            "anchor__score__origin": [10.0, 10.0],
            "anchor__score__previous": [7.0, 7.0],
        }
    )
    table = ForecastTable(
        pl.DataFrame(),
        (),
        ("score",),
        3,
        {"first": {"score": 1}, "second": {"score": 2}},
    )

    restored = undifference_forecasts(predictions, table)

    assert restored["target__score"].to_list() == [25.0, 40.0]
    assert restored["ridge__score"].to_list() == [16.0, 31.0]
