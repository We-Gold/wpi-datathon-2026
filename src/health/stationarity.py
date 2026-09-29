"""Per-subject stationarity checks and leakage-safe differencing plans."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class StationarityCheck:
    order: int
    observations: int
    status: str
    tests: tuple[dict[str, float | int], ...]


DifferencingOrders = dict[str, dict[str, int]]


def _complete_daily_calendar(daily: pl.DataFrame) -> pl.DataFrame:
    """Insert null rows for calendar days absent from each subject's records."""
    if daily.is_empty():
        return daily
    source = daily.with_columns(pl.col("subject_id").cast(pl.String))
    parts: list[pl.DataFrame] = []
    for frame in source.partition_by("subject_id", maintain_order=True):
        subject = str(frame.item(0, "subject_id"))
        start, end = frame["date"].min(), frame["date"].max()
        days = (end - start).days + 1
        calendar = pl.DataFrame(
            {
                "subject_id": [subject] * days,
                "date": pl.date_range(start, end, eager=True),
            }
        )
        parts.append(calendar.join(frame, on=["subject_id", "date"], how="left", coalesce=True))
    return pl.concat(parts, how="diagonal_relaxed").sort("subject_id", "date")


def check_stationarity(
    values: np.ndarray | list[float], *, alpha: float = 0.05, min_observations: int = 30
) -> StationarityCheck:
    """Choose order using ADF/KPSS on the longest contiguous finite segment."""
    series = np.asarray(values, dtype=float)
    finite = np.isfinite(series)
    observations = int(finite.sum())
    boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    if len(boundaries):
        starts, stops = boundaries[::2], boundaries[1::2]
        longest = int(np.argmax(stops - starts))
        series = series[starts[longest] : stops[longest]]
    else:
        series = np.asarray([], dtype=float)
    if observations < min_observations:
        return StationarityCheck(0, observations, "insufficient_data", ())
    if series.size < min_observations:
        return StationarityCheck(0, observations, "insufficient_contiguous_data", ())

    stage_results: list[dict[str, float | int]] = []
    for order in range(3):
        if order:
            series = np.diff(series)
        if series.size < max(min_observations - order, 12):
            return StationarityCheck(
                max(0, order - 1),
                observations,
                "insufficient_after_differencing",
                tuple(stage_results),
            )
        if np.allclose(series, series[0], rtol=1e-9, atol=1e-10):
            stage_results.append({"order": order, "adf_pvalue": 0.0, "kpss_pvalue": 1.0})
            return StationarityCheck(order, observations, "stationary", tuple(stage_results))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_pvalue = float(adfuller(series, regression="c", autolag="AIC")[1])
                kpss_pvalue = float(kpss(series, regression="c", nlags="auto")[1])
        except (ValueError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError):
            return StationarityCheck(
                max(0, order - 1), observations, "inconclusive", tuple(stage_results)
            )

        stage_results.append(
            {"order": order, "adf_pvalue": adf_pvalue, "kpss_pvalue": kpss_pvalue}
        )
        if adf_pvalue <= alpha and kpss_pvalue > alpha:
            return StationarityCheck(order, observations, "stationary", tuple(stage_results))

    return StationarityCheck(
        2, observations, "not_stationary_after_second_difference", tuple(stage_results)
    )


def fit_differencing_plan(
    daily: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    train_fraction: float = 0.8,
    alpha: float = 0.05,
    min_observations: int = 30,
) -> tuple[DifferencingOrders, list[dict[str, object]]]:
    """Fit per-subject orders on each subject's early daily calendar."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")

    orders: DifferencingOrders = {}
    diagnostics: list[dict[str, object]] = []
    calendar = _complete_daily_calendar(daily)
    for frame in calendar.partition_by("subject_id", maintain_order=True):
        subject = str(frame.item(0, "subject_id"))
        train_size = max(1, min(frame.height, ceil(frame.height * train_fraction)))
        training = frame.head(train_size)
        subject_orders: dict[str, int] = {}
        for column in columns:
            if column not in training.columns:
                continue
            values = (
                training[column]
                .cast(pl.Float64)
                .fill_null(float("nan"))
                .to_numpy()
            )
            check = check_stationarity(
                values, alpha=alpha, min_observations=min_observations
            )
            subject_orders[column] = check.order
            diagnostics.append(
                {
                    "subject_id": subject,
                    "feature": column,
                    "differencing_order": check.order,
                    "observations": check.observations,
                    "status": check.status,
                    "tests": list(check.tests),
                }
            )
        orders[subject] = subject_orders
    return orders, diagnostics


def apply_differencing(
    daily: pl.DataFrame, orders: DifferencingOrders
) -> pl.DataFrame:
    """Apply fitted differences without crossing subjects or missing dates."""
    calendar = _complete_daily_calendar(daily)
    parts: list[pl.DataFrame] = []
    for frame in calendar.partition_by("subject_id", maintain_order=True):
        subject = str(frame.item(0, "subject_id"))
        transformed = frame
        for column, order in orders.get(subject, {}).items():
            for _ in range(order):
                transformed = transformed.with_columns(pl.col(column).diff().alias(column))
        parts.append(transformed)
    return pl.concat(parts, how="diagonal_relaxed").sort("subject_id", "date")