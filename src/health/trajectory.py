"""Map positions, daily contributions, clusters, and Ridge forecasts for the dashboard.

A position is an exponentially weighted moving average of an axis score,
started at the person's usual level (zero). A velocity is one day's change in
position. The two sum exactly, and each velocity splits exactly into factor
contributions plus a pull back toward the usual level.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import ceil

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture

from health.features import AXIS_FEATURES, add_axis_components, axis_component_name
from health.forecast import complete_daily_calendar

AXES: tuple[str, ...] = ("activity", "recovery")
HALF_LIFE_DAYS = 7
KEEP = 0.5 ** (1 / HALF_LIFE_DAYS)
FORECAST_DAYS = 30
# Today and a week ago. More lags gave no accuracy gain and unstable weights.
FORECAST_LAGS: tuple[int, ...] = (0, 7)
BAND_QUANTILES = (0.1, 0.9)
RIDGE_ALPHA = 1.0

# Groups axis ingredients into the factors a person reads on the dashboard.
FACTORS: dict[str, str] = {
    "steps": "steps",
    "distance_m": "steps",
    "exercise_minutes": "exerciseTime",
    "active_energy_kcal": "activeEnergy",
    "light_activity_hours": "sitting",
    "sedentary_hours": "sitting",
    "hr_active_excess_bpm": "heartRate",
    "resting_hr_bpm": "heartRate",
    "resting_hr_change_bpm": "heartRate",
    "hrv_sdnn_ms": "hrv",
    "sleep_hours": "sleep",
    "sleep_efficiency": "sleep",
    "sleep_score": "sleep",
    "sleep_restlessness": "sleep",
    "respiratory_rate": "breathing",
}
PULL_FACTOR = "pull"


def position_name(axis: str) -> str:
    return f"{axis}_position"


def velocity_name(axis: str) -> str:
    return f"{axis}_velocity"


def contribution_name(axis: str, factor: str) -> str:
    return f"{axis}__factor__{factor}"


def factor_keys() -> list[str]:
    return [*dict.fromkeys(FACTORS.values()), PULL_FACTOR]


def add_positions(daily: pl.DataFrame) -> pl.DataFrame:
    """Fill the calendar and add positions, velocities, and factor contributions.

    `daily` needs the raw axis ingredients (as in `daily_features.parquet`).
    On a day with no score, position holds and every contribution is zero.
    """
    frame = complete_daily_calendar(add_axis_components(daily))
    factors = factor_keys()
    parts: list[pl.DataFrame] = []
    for subject in frame.partition_by("subject_id", maintain_order=True):
        columns: dict[str, np.ndarray] = {}
        for axis in AXES:
            present = [
                f for f in AXIS_FEATURES[axis] if axis_component_name(axis, f) in subject.columns
            ]
            values = np.column_stack(
                [subject[axis_component_name(axis, f)].to_numpy().astype(float) for f in present]
            )
            observed = np.isfinite(values)
            count = observed.sum(axis=1)
            shares = np.where(observed, values, 0.0) / np.maximum(count, 1)[:, None]
            position = np.zeros(subject.height)
            velocity = np.zeros(subject.height)
            pull = np.zeros(subject.height)
            previous = 0.0
            for day in range(subject.height):
                if count[day]:
                    score = shares[day].sum()
                    velocity[day] = (1 - KEEP) * (score - previous)
                    pull[day] = -(1 - KEEP) * previous
                previous += velocity[day]
                position[day] = previous
            columns[position_name(axis)] = position
            columns[velocity_name(axis)] = velocity
            columns[f"{axis}_count"] = count
            for factor in factors:
                columns[contribution_name(axis, factor)] = np.zeros(subject.height)
            columns[contribution_name(axis, PULL_FACTOR)] = pull
            for index, feature in enumerate(present):
                active = count > 0
                columns[contribution_name(axis, FACTORS[feature])][active] += (1 - KEEP) * shares[
                    active, index
                ]
        parts.append(subject.with_columns(pl.Series(k, v) for k, v in columns.items()))
    return pl.concat(parts, how="diagonal_relaxed")


def cluster_kind(center: np.ndarray) -> str:
    """Healthy when the cluster sits above the person's usual level on the two axes combined."""
    return "healthy" if center.sum() >= 0 else "unhealthy"


def fit_clusters(points: np.ndarray, max_clusters: int = 6) -> list[dict[str, object]]:
    """Fit a Gaussian mixture to map positions, choosing the count by BIC."""
    candidates = range(2, min(max_clusters, len(points) // 10) + 1)
    models = [
        GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0).fit(points)
        for k in candidates
    ]
    if not models:
        return []
    best = min(models, key=lambda model: model.bic(points))
    return [
        {
            "center": dict(zip(AXES, center.tolist(), strict=True)),
            "covariance": covariance.tolist(),
            "kind": cluster_kind(center),
        }
        for center, covariance in zip(best.means_, best.covariances_, strict=True)
    ]


def lag_feature_names() -> list[str]:
    return [f"{position_name(axis)}__lag{lag}" for axis in AXES for lag in FORECAST_LAGS]


def _lagged(frame: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Origin features from past positions and the position `horizon` days later."""
    lags = [
        pl.col(position_name(axis))
        .shift(lag)
        .over("subject_id")
        .alias(f"{position_name(axis)}__lag{lag}")
        for axis in AXES
        for lag in FORECAST_LAGS
    ]
    targets = [
        pl.col(position_name(axis)).shift(-horizon).over("subject_id").alias(f"target__{axis}")
        for axis in AXES
    ]
    target_date = pl.col("date").shift(-horizon).over("subject_id").alias("target_date")
    return frame.with_columns(*lags, *targets, target_date)


def _split_dates(frame: pl.DataFrame, fractions: dict[str, float]) -> pl.DataFrame:
    """For each subject and name, the first date of their latest `fraction` of days."""
    rows = []
    for subject in frame.partition_by("subject_id", maintain_order=True):
        dates = subject["date"].sort()
        starts = [dates[len(dates) - ceil(len(dates) * f)] for f in fractions.values()]
        rows.append((subject.item(0, "subject_id"), *starts))
    return pl.DataFrame(rows, schema=["subject_id", *fractions], orient="row")


def _residuals(
    rows: pl.DataFrame, train: pl.Expr, test: pl.Expr, features: list[str], targets: list[str]
) -> tuple[np.ndarray, pl.DataFrame]:
    """Errors on `test` rows of a Ridge fit on `train` rows."""
    fit_rows, test_rows = rows.filter(train), rows.filter(test)
    model = Ridge(alpha=RIDGE_ALPHA).fit(
        fit_rows.select(features).to_numpy(), fit_rows.select(targets).to_numpy()
    )
    predicted = model.predict(test_rows.select(features).to_numpy())
    return test_rows.select(targets).to_numpy() - predicted, test_rows


@dataclass
class RidgeForecast:
    """One direct Ridge model per horizon, predicting both positions from lagged positions."""

    models: dict[int, Ridge]
    bands: dict[int, np.ndarray]
    evaluation: dict[int, dict[str, dict[str, float]]]

    @classmethod
    def fit(
        cls,
        frame: pl.DataFrame,
        horizons: range = range(1, FORECAST_DAYS + 1),
        test_fraction: float = 0.2,
    ) -> RidgeForecast:
        """Fit on all days, with bands and errors from temporal holdouts.

        Splits use each row's target date, so no training label falls in a
        later period. Errors and bands come from a fit on days before the
        latest `test_fraction`, scored on that period. Band coverage is checked
        apart from that: bands from the period before it, scored on it.
        """
        features = lag_feature_names()
        target_cols = [f"target__{axis}" for axis in AXES]
        split = _split_dates(frame, {"test_start": test_fraction, "band_start": 2 * test_fraction})
        target = pl.col("target_date")
        in_test = target >= pl.col("test_start")
        in_band = (target >= pl.col("band_start")) & ~in_test
        models: dict[int, Ridge] = {}
        bands: dict[int, np.ndarray] = {}
        evaluation: dict[int, dict[str, dict[str, float]]] = {}
        for horizon in horizons:
            rows = (
                _lagged(frame, horizon)
                .drop_nulls([*features, *target_cols])
                .join(split, on="subject_id")
            )
            residual, test = _residuals(rows, ~in_test, in_test, features, target_cols)
            bands[horizon] = np.quantile(residual, BAND_QUANTILES, axis=0)
            early, _ = _residuals(
                rows, target < pl.col("band_start"), in_band, features, target_cols
            )
            check, _ = _residuals(
                rows, target < pl.col("band_start"), in_test, features, target_cols
            )
            lo, hi = np.quantile(early, BAND_QUANTILES, axis=0)
            inside = (check >= lo) & (check <= hi)
            actual = test.select(target_cols).to_numpy()
            hold = actual - test.select(f"{position_name(a)}__lag0" for a in AXES).to_numpy()
            evaluation[horizon] = {
                axis: {
                    "ridge_mae": float(np.abs(residual[:, i]).mean()),
                    "hold_today_mae": float(np.abs(hold[:, i]).mean()),
                    "usual_level_mae": float(np.abs(actual[:, i]).mean()),
                    "band_coverage": float(inside[:, i].mean()),
                    "test_rows": int(test.height),
                }
                for i, axis in enumerate(AXES)
            }
            models[horizon] = Ridge(alpha=RIDGE_ALPHA).fit(
                rows.select(features).to_numpy(), rows.select(target_cols).to_numpy()
            )
        return cls(models, bands, evaluation)

    def predict_from(self, origin: pl.DataFrame) -> list[dict[str, object]]:
        """Forecast from the last row of one subject's filled calendar."""
        features = _lagged(origin, 0).select(lag_feature_names()).tail(1).to_numpy()
        start = origin["date"].max()
        days: list[dict[str, object]] = []
        for horizon, model in sorted(self.models.items()):
            center = model.predict(features)[0]
            lo, hi = center + self.bands[horizon]
            days.append(
                {
                    "date": (start + timedelta(days=horizon)).isoformat(),
                    "position": dict(zip(AXES, center.tolist(), strict=True)),
                    "low": dict(zip(AXES, lo.tolist(), strict=True)),
                    "high": dict(zip(AXES, hi.tolist(), strict=True)),
                }
            )
        return days

    def coefficients(self) -> dict[str, dict[str, object]]:
        names = lag_feature_names()
        return {
            str(horizon): {
                axis: {
                    "intercept": float(model.intercept_[i]),
                    **dict(zip(names, model.coef_[i].tolist(), strict=True)),
                }
                for i, axis in enumerate(AXES)
            }
            for horizon, model in sorted(self.models.items())
        }
