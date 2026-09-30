from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl

from health.export_cli import run
from health.features import add_interpretable_axes
from health.trajectory import (
    AXES,
    KEEP,
    RidgeForecast,
    add_positions,
    cluster_kind,
    contribution_name,
    factor_keys,
    fit_clusters,
)


def _daily(days: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    dates = [date.fromordinal(date(2026, 1, 1).toordinal() + i) for i in range(days)]
    frame = pl.DataFrame(
        {
            "subject_id": ["one"] * days + ["two"] * days,
            "date": dates * 2,
            "steps": rng.normal(8000, 2000, 2 * days),
            "exercise_minutes": rng.normal(30, 10, 2 * days),
            "sleep_hours": rng.normal(7, 1, 2 * days),
            "resting_hr_bpm": rng.normal(60, 3, 2 * days),
            "hr_active_excess_bpm": rng.normal(40, 8, 2 * days),
        }
    )
    # Missing days and missing values inside observed days.
    return frame.filter(pl.col("date") != date(2026, 2, 1)).with_columns(
        pl.when(pl.col("date") == date(2026, 2, 5))
        .then(None)
        .otherwise(pl.col("sleep_hours"))
        .alias("sleep_hours")
    )


def test_positions_follow_python_scores_and_contributions_sum_to_velocity() -> None:
    daily = _daily()
    filled = add_positions(daily)
    scores = add_interpretable_axes(daily).select(
        "subject_id", "date", "activity_score", "recovery_score"
    )
    joined = filled.join(scores, on=["subject_id", "date"], how="left")
    for axis in AXES:
        previous = pl.col(f"{axis}_position").shift(1).over("subject_id").fill_null(0)
        implied = joined.filter(pl.col(f"{axis}_score").is_not_null()).select(
            pl.col(f"{axis}_velocity") / (1 - KEEP) + previous - pl.col(f"{axis}_score")
        )
        assert implied.to_series().abs().max() < 1e-9
        total = joined.select(
            pl.sum_horizontal(contribution_name(axis, f) for f in factor_keys())
            - pl.col(f"{axis}_velocity")
        )
        assert total.to_series().abs().max() < 1e-12


def test_missing_calendar_day_holds_position() -> None:
    filled = add_positions(_daily()).filter(pl.col("subject_id") == "one")
    gap = filled.filter(pl.col("date").is_between(date(2026, 1, 31), date(2026, 2, 1)))
    assert gap["activity_velocity"][1] == 0
    assert gap["activity_position"][0] == gap["activity_position"][1]


def test_cluster_kind_uses_both_axes() -> None:
    assert cluster_kind(np.array([0.3, -0.1])) == "healthy"
    assert cluster_kind(np.array([0.1, -0.3])) == "unhealthy"
    clusters = fit_clusters(np.random.default_rng(1).normal(size=(200, 2)))
    assert clusters and all(np.array(c["covariance"]).shape == (2, 2) for c in clusters)


def test_ridge_forecast_has_ordered_bands_and_dates() -> None:
    filled = add_positions(_daily())
    forecast = RidgeForecast.fit(filled, horizons=range(1, 8))
    subject = filled.filter(pl.col("subject_id") == "one")
    days = forecast.predict_from(subject)
    assert [d["date"] for d in days][:2] == ["2026-05-01", "2026-05-02"]
    for d in days:
        for axis in AXES:
            assert d["low"][axis] <= d["high"][axis]
    assert 0 <= forecast.evaluation[1]["activity"]["band_coverage"] <= 1


def test_export_writes_index_and_subject_files(tmp_path) -> None:
    daily_path = tmp_path / "daily.parquet"
    _daily().write_parquet(daily_path)
    run(daily_path, tmp_path / "out")
    index = json.loads((tmp_path / "out" / "index.json").read_text())
    assert [s["subjectId"] for s in index["subjects"]] == ["one", "two"]
    subject = json.loads((tmp_path / "out" / "subjects" / "one.json").read_text())
    trajectory = subject["trajectory"]
    # Today is the last day with data.
    assert trajectory["asOf"] == "2026-04-30"
    assert trajectory["history"][-1]["date"] == trajectory["asOf"]
    assert trajectory["prediction"][0]["date"] == "2026-05-01"
    assert len(trajectory["prediction"]) == 30
    assert subject["scaling"]["iqr"]["steps"] > 0
    today = trajectory["history"][-1]["velocity"]
    for axis in AXES:
        assert abs(sum(c[axis] for c in trajectory["contributions"]) - today[axis]) < 1e-9
