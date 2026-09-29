"""Leakage-safe supervised tables and model baselines for trajectory forecasting.

This module compares transparent persistence baselines with Ridge, LightGBM,
and ElasticNet regressors. A GRU or transformer should only replace them after
it wins rolling-origin evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from health.stationarity import (
    DifferencingOrders,
    apply_differencing,
    fit_differencing_plan,
)

DEFAULT_LAGS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 28)
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
    horizon: int
    differencing_orders: DifferencingOrders = field(default_factory=dict)
    stationarity_diagnostics: tuple[dict[str, object], ...] = ()


def build_forecast_table(
    daily: pl.DataFrame,
    *,
    inputs: tuple[str, ...] = DEFAULT_INPUTS,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    horizon: int = 1,
    train_fraction: float = 0.8,
    differencing_orders: DifferencingOrders | None = None,
    stationarity_diagnostics: tuple[dict[str, object], ...] = (),
) -> ForecastTable:
    """Build differenced lag features while retaining inversion anchors per target."""
    if horizon < 1:
        raise ValueError("horizon must be at least one day")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    available_inputs = tuple(name for name in inputs if name in daily.columns)
    available_targets = tuple(name for name in targets if name in daily.columns)
    if not available_inputs or not available_targets:
        raise ValueError("daily table does not contain requested forecast inputs/targets")

    stationarity_columns = tuple(dict.fromkeys((*available_inputs, *available_targets)))
    if differencing_orders is None:
        differencing_orders, fitted_diagnostics = fit_differencing_plan(
            daily, stationarity_columns, train_fraction=train_fraction
        )
        stationarity_diagnostics = tuple(fitted_diagnostics)
    transformed = apply_differencing(daily, differencing_orders)
    # Ensure our lag sequence includes the required baselines for this specific horizon
    seasonal_lag = (7 - horizon % 7) % 7
    required_lags = set(range(4)) | {seasonal_lag}
    missing_lags = required_lags - set(lags)
    if missing_lags:
        lags = tuple(sorted(set(lags) | required_lags))
    grid = complete_daily_calendar(transformed)
    raw_grid = complete_daily_calendar(daily)
    expressions: list[pl.Expr] = []
    for name in stationarity_columns:
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
    # Retain only information available at the forecast origin plus the raw future label.
    anchor_expressions: list[pl.Expr] = []
    for name in available_targets:
        anchor_expressions.extend(
            [
                pl.col(name).shift(-horizon).over("subject_id").alias(f"actual__{name}"),
                pl.col(name).alias(f"anchor__{name}__origin"),
                pl.col(name).shift(1).over("subject_id").alias(f"anchor__{name}__previous"),
            ]
        )
    if anchor_expressions:
        raw_anchors = raw_grid.with_columns(anchor_expressions).select(
            "subject_id", "date", *[expr.meta.output_name() for expr in anchor_expressions]
        )
        frame = frame.join(raw_anchors, on=["subject_id", "date"], how="left")
    return ForecastTable(
        frame,
        input_columns,
        available_targets,
        horizon,
        differencing_orders,
        stationarity_diagnostics,
    )

def baseline_predictions(frame: pl.DataFrame, table: ForecastTable) -> pl.DataFrame:
    """Multi-day horizon persistence and weekly seasonal persistence baselines."""
    output: dict[str, np.ndarray] = {}
    h = table.horizon
    
    # Forecast origin is frame.date; all baseline inputs must be known by then.
    persistence_lag = 0
    seasonal_lag = (7 - h % 7) % 7
    
    for target in table.target_columns:
        # Standard Multi-day Horizon Persistence
        current = frame[f"{target}__lag{persistence_lag}"].to_numpy().astype(float)
        output[f"persistence__{target}"] = current
        
        # Weekly Seasonal Persistence (same day of week from prior week)
        seasonal = frame[f"{target}__lag{seasonal_lag}"].to_numpy().astype(float)
        output[f"seasonal_persistence__{target}"] = seasonal
        
        # Horizon-adjusted 4-period rolling mean
        rolling_lags = list(range(4))
        previous = np.column_stack(
            [frame[f"{target}__lag{lag}"].to_numpy().astype(float) for lag in rolling_lags]
        )
        with np.errstate(all="ignore"):
            output[f"rolling__{target}"] = np.nanmean(previous, axis=1)
            
    return pl.DataFrame(output)

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
class MultiModelForecaster:
    input_columns: tuple[str, ...]
    targets: tuple[str, ...]
    medians: np.ndarray
    models: dict[str, dict[str, Any]]

    @classmethod
    def fit(cls, train: pl.DataFrame, table: ForecastTable) -> MultiModelForecaster:
        x = train.select(table.input_columns).to_numpy().astype(float)
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        x = np.where(np.isfinite(x), x, medians)
        models: dict[str, dict[str, Any]] = {}
        for target in table.target_columns:
            y = train[f"target__{target}"].to_numpy().astype(float)
            valid = np.isfinite(y)
            models[target] = {
                "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                "lightgbm": LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=0,
                    n_jobs=1,
                    verbosity=-1,
                ),
                "elasticnet": make_pipeline(
                    StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000)
                ),
            }
            for model in models[target].values():
                model.fit(x[valid], y[valid])
        return cls(table.input_columns, table.target_columns, medians, models)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        x = frame.select(self.input_columns).to_numpy().astype(float)
        x = np.where(np.isfinite(x), x, self.medians)
        return pl.DataFrame(
            {
                f"{model_name}__{target}": model.predict(x)
                for target, target_models in self.models.items()
                for model_name, model in target_models.items()
            }
        )


def undifference_forecasts(predictions: pl.DataFrame, table: ForecastTable) -> pl.DataFrame:
    """Reconstruct differenced predictions using forecast-origin values only.

    A terminal first difference is treated as a constant daily change across the
    horizon. A terminal second difference is treated as constant curvature.
    """
    prediction_columns = [
        column
        for column in predictions.columns
        if column.startswith(
            (
                "persistence__",
                "seasonal_persistence__",
                "rolling__",
                "ridge__",
                "lightgbm__",
                "elasticnet__",
            )
        )
    ]
    subject_ids = predictions["subject_id"].cast(pl.String).to_numpy()
    for target in table.target_columns:
        origin = predictions[f"anchor__{target}__origin"].to_numpy().astype(float)
        previous = predictions[f"anchor__{target}__previous"].to_numpy().astype(float)
        columns = [
            column
            for column in prediction_columns
            if column != f"target__{target}" and column.endswith(f"__{target}")
        ]
        for column in columns:
            values = predictions[column].to_numpy().astype(float)
            for row, subject in enumerate(subject_ids):
                order = table.differencing_orders.get(str(subject), {}).get(target, 0)
                if order == 1:
                    values[row] = origin[row] + table.horizon * values[row]
                elif order == 2:
                    values[row] = (
                        origin[row]
                        + table.horizon * (origin[row] - previous[row])
                        + table.horizon * (table.horizon + 1) / 2 * values[row]
                    )
                if order and not np.isfinite(origin[row]):
                    values[row] = float("nan")
                if order == 2 and not np.isfinite(previous[row]):
                    values[row] = float("nan")
            predictions = predictions.with_columns(pl.Series(column, values))
    return predictions


def evaluate_forecasters(
    train: pl.DataFrame, test: pl.DataFrame, table: ForecastTable
) -> tuple[dict[str, object], pl.DataFrame]:
    """Evaluate persistence baselines and three regressors on the temporal holdout."""
    baseline = baseline_predictions(test, table)
    forecaster = MultiModelForecaster.fit(train, table)
    predictions = test.select(
        "subject_id",
        "date",
        "target_date",
        *[
            pl.col(f"actual__{target}").alias(f"target__{target}")
            for target in table.target_columns
        ],
        *[
            column
            for target in table.target_columns
            for column in (f"anchor__{target}__origin", f"anchor__{target}__previous")
        ],
    ).hstack(baseline.hstack(forecaster.predict(test)))
    predictions = undifference_forecasts(predictions, table)
    report: dict[str, object] = {
        "horizon_days": table.horizon,
        "targets": {},
        "differencing_orders": table.differencing_orders,
        "stationarity_diagnostics": list(table.stationarity_diagnostics),
        "inverse_differencing": {
            "order_1": "origin level + horizon * predicted first difference",
            "order_2": (
                "origin level + horizon * origin slope + triangular horizon weight "
                "* predicted second difference"
            ),
        },
    }
    for target in table.target_columns:
        actual = predictions[f"target__{target}"].to_numpy().astype(float)
        target_report: dict[str, object] = {}
        for model in (
            "persistence",
            "seasonal_persistence",
            "rolling",
            "ridge",
            "lightgbm",
            "elasticnet",
        ):
            predicted = predictions[f"{model}__{target}"].to_numpy().astype(float)
            valid = np.isfinite(actual) & np.isfinite(predicted)
            target_report[model] = {
                "mae": float(np.abs(actual[valid] - predicted[valid]).mean()),
                "rmse": float(np.sqrt(np.square(actual[valid] - predicted[valid]).mean())),
                "n": int(valid.sum()),
            }
        report["targets"][target] = target_report
    return report, predictions
