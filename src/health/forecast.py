"""Leakage-safe supervised tables and first baselines for trajectory forecasting.

This module intentionally starts with transparent one-step baselines. A GRU or
transformer should only replace them after it wins rolling-origin evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

DEFAULT_LAGS: tuple[int, ...] = (0, 1, 2, 3, 7, 14, 28)
DEFAULT_TARGETS: tuple[str, ...] = ("activity_score", "recovery_score")
DEFAULT_INPUTS: tuple[str, ...] = (
    "steps",
    "sleep_hours",
    "hr_mean_bpm",
    "resting_hr_bpm",
    "hr_active_excess_bpm",
    "hrv_sdnn_ms",
    "resting_hr_change_bpm",
    "activity_score",
    "recovery_score",
)


def complete_daily_calendar(daily: pl.DataFrame) -> pl.DataFrame:
    """Insert missing calendar days as null rows, separately per subject."""
    if daily.is_empty():
        return daily
    source = daily.with_columns(pl.col("subject_id").cast(pl.String))
    parts: list[pl.DataFrame] = []
    for frame in source.partition_by("subject_id", maintain_order=True):
        subject = str(frame.item(0, "subject_id"))
        start, end = frame["date"].min(), frame["date"].max()
        calendar = pl.DataFrame(
            {
                "subject_id": [subject] * ((end - start).days + 1),
                "date": pl.date_range(start, end, eager=True),
            }
        )
        parts.append(calendar.join(frame, on=["subject_id", "date"], how="left", coalesce=True))
    return pl.concat(parts, how="diagonal_relaxed").sort("subject_id", "date")


@dataclass(frozen=True)
class ForecastTable:
    frame: pl.DataFrame
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


def build_forecast_table(
    daily: pl.DataFrame,
    *,
    inputs: tuple[str, ...] = DEFAULT_INPUTS,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    horizon: int = 1,
) -> ForecastTable:
    """Build calendar-aware lag features and a future target for each subject."""
    if horizon < 1:
        raise ValueError("horizon must be at least one day")
    available_inputs = tuple(name for name in inputs if name in daily.columns)
    available_targets = tuple(name for name in targets if name in daily.columns)
    if not available_inputs or not available_targets:
        raise ValueError("daily table does not contain requested forecast inputs/targets")
    grid = complete_daily_calendar(daily)
    expressions: list[pl.Expr] = []
    for name in available_inputs:
        for lag in lags:
            expressions.append(
                pl.col(name).shift(lag).over("subject_id").alias(f"{name}__lag{lag}")
            )
    expressions.append(pl.col("date").shift(-horizon).over("subject_id").alias("target_date"))
    for name in available_targets:
        expressions.append(pl.col(name).shift(-horizon).over("subject_id").alias(f"target__{name}"))
    frame = grid.with_columns(expressions).filter(
        pl.col("target_date").is_not_null()
        & pl.all_horizontal(
            *(pl.col(f"target__{name}").is_not_null() for name in available_targets)
        )
    )
    input_columns = tuple(f"{name}__lag{lag}" for name in available_inputs for lag in lags)
    return ForecastTable(frame, input_columns, available_targets)


def temporal_forecast_split(
    table: ForecastTable, test_fraction: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by each subject's target date, keeping future labels out of train."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    train_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []
    for frame in table.frame.sort("subject_id", "target_date").partition_by(
        "subject_id", maintain_order=True
    ):
        split = max(1, min(frame.height - 1, round(frame.height * (1 - test_fraction))))
        train_parts.append(frame[:split])
        test_parts.append(frame[split:])
    return pl.concat(train_parts), pl.concat(test_parts)


@dataclass
class RidgeForecaster:
    input_columns: tuple[str, ...]
    targets: tuple[str, ...]
    medians: np.ndarray
    models: dict[str, Ridge]

    @classmethod
    def fit(cls, train: pl.DataFrame, table: ForecastTable, alpha: float = 1.0) -> RidgeForecaster:
        x = train.select(table.input_columns).to_numpy().astype(float)
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        x = np.where(np.isfinite(x), x, medians)
        models: dict[str, Ridge] = {}
        for target in table.target_columns:
            y = train[f"target__{target}"].to_numpy().astype(float)
            valid = np.isfinite(y)
            models[target] = Ridge(alpha=alpha).fit(x[valid], y[valid])
        return cls(table.input_columns, table.target_columns, medians, models)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        x = frame.select(self.input_columns).to_numpy().astype(float)
        x = np.where(np.isfinite(x), x, self.medians)
        return pl.DataFrame(
            {f"ridge__{target}": model.predict(x) for target, model in self.models.items()}
        )


def baseline_predictions(frame: pl.DataFrame, table: ForecastTable) -> pl.DataFrame:
    """Persistence and short rolling-mean predictions from lagged state values."""
    output: dict[str, np.ndarray] = {}
    for target in table.target_columns:
        current = frame[f"{target}__lag0"].to_numpy().astype(float)
        previous = np.column_stack(
            [frame[f"{target}__lag{lag}"].to_numpy().astype(float) for lag in (0, 1, 2, 3)]
        )
        output[f"persistence__{target}"] = current
        with np.errstate(all="ignore"):
            output[f"rolling__{target}"] = np.nanmean(previous, axis=1)
    return pl.DataFrame(output)


def evaluate_forecasters(
    train: pl.DataFrame, test: pl.DataFrame, table: ForecastTable
) -> tuple[dict[str, object], pl.DataFrame]:
    """Evaluate persistence, rolling mean, and Ridge on the temporal holdout."""
    baseline = baseline_predictions(test, table)
    ridge = RidgeForecaster.fit(train, table)
    predictions = test.select(
        "subject_id", "date", "target_date", *[f"target__{t}" for t in table.target_columns]
    ).hstack(baseline.hstack(ridge.predict(test)))
    report: dict[str, object] = {"horizon_days": 1, "targets": {}}
    for target in table.target_columns:
        actual = predictions[f"target__{target}"].to_numpy().astype(float)
        target_report: dict[str, object] = {}
        for model in ("persistence", "rolling", "ridge"):
            predicted = predictions[f"{model}__{target}"].to_numpy().astype(float)
            valid = np.isfinite(actual) & np.isfinite(predicted)
            target_report[model] = {
                "mae": float(np.abs(actual[valid] - predicted[valid]).mean()),
                "rmse": float(np.sqrt(np.square(actual[valid] - predicted[valid]).mean())),
                "n": int(valid.sum()),
            }
        report["targets"][target] = target_report
    return report, predictions
