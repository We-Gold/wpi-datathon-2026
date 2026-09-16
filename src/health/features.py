"""Daily features and interpretable health axes.

The ingestion package deliberately preserves observations at their original
grain.  This module is the next boundary: it turns those observations into one
row per subject and local calendar day for embedding models.

Missing values remain missing here.  A missing wearable measurement is not a
physiological zero, and imputation belongs inside a train/test split rather
than in the feature table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from health import metrics

_Q = metrics.QUANTITY_PREFIX
_SLEEP = metrics.CATEGORY_PREFIX + "SleepAnalysis"


@dataclass(frozen=True)
class Feature:
    """Definition of a scalar daily feature."""

    name: str
    metric: str
    aggregation: str
    scale: float = 1.0


# Small enough to occur across Apple Health, Fitbit/PMData and the supplied
# exports.  Less portable features remain available as optional candidates.
CORE_FEATURES: tuple[str, ...] = (
    "steps",
    "distance_m",
    "exercise_minutes",
    "resting_hr_bpm",
    "sleep_hours",
)

SCALAR_FEATURES: tuple[Feature, ...] = (
    Feature("steps", _Q + "StepCount", "sum"),
    Feature("distance_m", _Q + "DistanceWalkingRunning", "sum"),
    Feature("exercise_minutes", _Q + "AppleExerciseTime", "sum", 1 / 60),
    Feature("active_energy_kcal", _Q + "ActiveEnergyBurned", "sum"),
    Feature("total_energy_kcal", metrics.TOTAL_ENERGY_BURNED, "sum"),
    Feature("sedentary_hours", "SedentaryTime", "sum", 1 / 3600),
    Feature("light_activity_hours", "LightActivityTime", "sum", 1 / 3600),
    Feature("resting_hr_bpm", _Q + "RestingHeartRate", "mean"),
    Feature("hrv_sdnn_ms", _Q + "HeartRateVariabilitySDNN", "mean", 1000),
    Feature("respiratory_rate", _Q + "RespiratoryRate", "mean"),
    Feature("sleep_score", "SleepScore", "mean"),
    Feature("sleep_restlessness", "SleepRestlessness", "mean"),
    Feature("readiness", "Readiness", "mean"),
)

MODEL_FEATURES: tuple[str, ...] = (
    *(feature.name for feature in SCALAR_FEATURES),
    "sleep_hours",
    "sleep_efficiency",
    "hr_mean_bpm",
    "hr_std_bpm",
    "hr_range_bpm",
    "hr_spectral_entropy",
    "hr_low_frequency_power",
)


@dataclass(frozen=True)
class SubjectWindow:
    subject_id: str
    start: date
    end: date
    days: int
    mean_completeness: float
    meets_threshold: bool


def _daily_scalar(records: pl.DataFrame, feature: Feature) -> pl.DataFrame:
    values = records.filter(
        (pl.col("metric") == feature.metric) & pl.col("value_num").is_not_null()
    )
    expression = (
        pl.col("value_num").sum() if feature.aggregation == "sum" else pl.col("value_num").mean()
    )
    return (
        values.with_columns(pl.col("start_local").dt.date().alias("date"))
        .group_by("subject_id", "date")
        .agg((expression * feature.scale).alias(feature.name))
    )


def _sleep_features(records: pl.DataFrame) -> pl.DataFrame:
    """Build duration and efficiency, preferring a source's daily rollup.

    Stage intervals count asleep time only; the containing ``InBed`` interval
    is excluded to avoid double counting.  Efficiency is available only when
    both asleep and awake stage intervals were recorded.
    """
    rolled = (
        records.filter(
            (pl.col("metric") == metrics.SLEEP_DURATION_DAILY) & pl.col("value_num").is_not_null()
        )
        .with_columns(pl.col("start_local").dt.date().alias("date"))
        .group_by("subject_id", "date")
        .agg((pl.col("value_num").sum() / 3600).alias("rolled_hours"))
    )

    sleep = records.filter((pl.col("metric") == _SLEEP) & pl.col("value_str").is_not_null())
    if sleep.is_empty():
        staged = pl.DataFrame(
            schema={
                "subject_id": pl.Categorical,
                "date": pl.Date,
                "staged_hours": pl.Float64,
                "sleep_efficiency": pl.Float64,
            }
        )
    else:
        staged = (
            sleep.with_columns(
                (pl.col("start_local") - pl.duration(hours=12)).dt.date().alias("date"),
                ((pl.col("end_local") - pl.col("start_local")).dt.total_seconds()).alias("seconds"),
                pl.col("value_str").cast(pl.String).str.contains("Asleep").alias("asleep"),
                pl.col("value_str").cast(pl.String).str.ends_with("Awake").alias("awake"),
            )
            .filter(pl.col("asleep") | pl.col("awake"))
            .group_by("subject_id", "date")
            .agg(
                pl.col("seconds").filter(pl.col("asleep")).sum().alias("asleep_seconds"),
                pl.col("seconds").filter(pl.col("awake")).sum().alias("awake_seconds"),
            )
            .with_columns(
                (pl.col("asleep_seconds") / 3600).alias("staged_hours"),
                pl.when(pl.col("awake_seconds") > 0)
                .then(
                    pl.col("asleep_seconds") / (pl.col("asleep_seconds") + pl.col("awake_seconds"))
                )
                .otherwise(None)
                .alias("sleep_efficiency"),
            )
            .select("subject_id", "date", "staged_hours", "sleep_efficiency")
        )

    if rolled.is_empty() and staged.is_empty():
        return pl.DataFrame(
            schema={
                "subject_id": pl.Categorical,
                "date": pl.Date,
                "sleep_hours": pl.Float64,
                "sleep_efficiency": pl.Float64,
            }
        )
    return (
        rolled.join(staged, on=["subject_id", "date"], how="full", coalesce=True)
        .with_columns(pl.coalesce("rolled_hours", "staged_hours").alias("sleep_hours"))
        .select("subject_id", "date", "sleep_hours", "sleep_efficiency")
    )


def _heart_rate_features(records: pl.DataFrame, bins: int = 96) -> pl.DataFrame:
    """Daily HR summaries plus FFT features on a regular 15-minute grid.

    Frequency features require observations in at least half the bins and a
    12-hour span.  This prevents interpolation across a short workout from
    masquerading as an all-day signal.  Spectral entropy is normalized to
    [0, 1]; low-frequency power is the fraction at 1--6 cycles/day.
    """
    hr = records.filter(
        (pl.col("metric") == _Q + "HeartRate") & pl.col("value_num").is_not_null()
    ).with_columns(pl.col("start_local").dt.date().alias("date"))
    if hr.is_empty():
        return pl.DataFrame(
            schema={
                "subject_id": pl.Categorical,
                "date": pl.Date,
                "hr_mean_bpm": pl.Float64,
                "hr_std_bpm": pl.Float64,
                "hr_range_bpm": pl.Float64,
                "hr_spectral_entropy": pl.Float64,
                "hr_low_frequency_power": pl.Float64,
            }
        )

    summaries = hr.group_by("subject_id", "date").agg(
        pl.col("value_num").mean().alias("hr_mean_bpm"),
        pl.col("value_num").std().alias("hr_std_bpm"),
        (pl.col("value_num").max() - pl.col("value_num").min()).alias("hr_range_bpm"),
    )
    spectral_rows: list[dict[str, object]] = []
    for group in hr.select("subject_id", "date", "start_local", "value_num").partition_by(
        ["subject_id", "date"], maintain_order=True
    ):
        subject = str(group.item(0, "subject_id"))
        day = group.item(0, "date")
        times = group["start_local"].to_list()
        values = np.asarray(group["value_num"].to_list(), dtype=float)
        minute = np.asarray([value.hour * 60 + value.minute + value.second / 60 for value in times])
        index = np.minimum((minute * bins / (24 * 60)).astype(int), bins - 1)
        grid = np.full(bins, np.nan)
        for position in np.unique(index):
            grid[position] = values[index == position].mean()
        observed = np.flatnonzero(~np.isnan(grid))
        entropy = low_power = None
        if len(observed) >= bins / 2 and minute.max() - minute.min() >= 12 * 60:
            grid = np.interp(np.arange(bins), observed, grid[observed])
            centered = np.asarray(grid - grid.mean(), dtype=np.float64)
            power = np.abs(np.fft.rfft(centered)) ** 2
            power = power[1:]
            total = power.sum()
            if total > 0:
                fractions = power / total
                raw_entropy = -(fractions * np.log(fractions + 1e-12)).sum() / np.log(len(power))
                entropy = float(np.clip(raw_entropy, 0, 1))
                frequencies = np.fft.rfftfreq(bins, d=1 / bins)[1:]
                low_power = float(power[frequencies <= 6].sum() / total)
        spectral_rows.append(
            {
                "subject_id": subject,
                "date": day,
                "hr_spectral_entropy": entropy,
                "hr_low_frequency_power": low_power,
            }
        )
    spectral = pl.DataFrame(spectral_rows).with_columns(pl.col("subject_id").cast(pl.Categorical))
    return summaries.join(spectral, on=["subject_id", "date"], how="left")


def build_daily_features(records: pl.DataFrame) -> pl.DataFrame:
    """Return one row per observed subject-day, without imputing missing data."""
    keys = (
        records.select("subject_id", pl.col("start_local").dt.date().alias("date"))
        .unique()
        .sort("subject_id", "date")
    )
    result = keys
    for feature in SCALAR_FEATURES:
        result = result.join(_daily_scalar(records, feature), on=["subject_id", "date"], how="left")
    result = result.join(_sleep_features(records), on=["subject_id", "date"], how="left")
    result = result.join(_heart_rate_features(records), on=["subject_id", "date"], how="left")
    available = [name for name in MODEL_FEATURES if name in result.columns]
    return result.with_columns(
        pl.mean_horizontal(
            *(pl.col(name).is_not_null().cast(pl.Float64) for name in available)
        ).alias("feature_completeness")
    ).sort("subject_id", "date")


def select_dense_windows(
    daily: pl.DataFrame,
    desired: tuple[str, ...] = CORE_FEATURES,
    *,
    minimum_days: int = 30,
    minimum_completeness: float = 0.6,
) -> list[SubjectWindow]:
    """Choose the longest dense interval independently for each subject.

    A date qualifies when the trailing ``minimum_days`` window meets the
    requested mean completeness. Consecutive qualifying windows are merged.
    If none qualify, the best minimum-length window is returned so the caller
    gets a diagnostic instead of silently losing the subject.
    """
    missing = [name for name in desired if name not in daily.columns]
    if missing:
        raise ValueError(f"desired features are absent from the table: {missing}")
    windows: list[SubjectWindow] = []
    for subject_frame in daily.partition_by("subject_id", maintain_order=True):
        subject = str(subject_frame.item(0, "subject_id"))
        observed = {
            row["date"]: row for row in subject_frame.select("date", *desired).iter_rows(named=True)
        }
        start, end = min(observed), max(observed)
        dates = [start + timedelta(days=index) for index in range((end - start).days + 1)]
        scores = np.asarray(
            [
                sum(observed.get(day, {}).get(name) is not None for name in desired) / len(desired)
                for day in dates
            ]
        )
        width = min(minimum_days, len(dates))
        rolling = np.convolve(scores, np.ones(width) / width, mode="valid")
        qualified = np.flatnonzero(rolling >= minimum_completeness)
        candidates: list[tuple[int, int]] = []
        if len(qualified):
            run_start = run_end = int(qualified[0])
            for position in qualified[1:]:
                if position == run_end + 1:
                    run_end = int(position)
                else:
                    candidates.append((run_start, run_end + width - 1))
                    run_start = run_end = int(position)
            candidates.append((run_start, run_end + width - 1))
        else:
            best = int(np.argmax(rolling))
            candidates.append((best, best + width - 1))
        best_start, best_end = max(
            candidates,
            key=lambda bounds: (
                bounds[1] - bounds[0] + 1,
                scores[bounds[0] : bounds[1] + 1].mean(),
            ),
        )
        windows.append(
            SubjectWindow(
                subject,
                dates[best_start],
                dates[best_end],
                best_end - best_start + 1,
                float(scores[best_start : best_end + 1].mean()),
                bool(len(qualified)),
            )
        )
    return windows


def apply_windows(daily: pl.DataFrame, windows: list[SubjectWindow]) -> pl.DataFrame:
    """Filter a daily table to the chosen inclusive interval per subject."""
    if not windows:
        return daily.clear()
    predicates = [
        (pl.col("subject_id").cast(pl.String) == window.subject_id)
        & pl.col("date").is_between(window.start, window.end)
        for window in windows
    ]
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = combined | predicate
    return daily.filter(combined)


AXIS_FEATURES: dict[str, dict[str, float]] = {
    "activity": {
        "steps": 1,
        "distance_m": 1,
        "exercise_minutes": 1,
        "active_energy_kcal": 1,
        "light_activity_hours": 1,
        "sedentary_hours": -1,
    },
    "recovery": {
        "sleep_hours": 1,
        "sleep_efficiency": 1,
        "hrv_sdnn_ms": 1,
        "sleep_score": 1,
        "sleep_restlessness": -1,
        "resting_hr_bpm": -1,
        "respiratory_rate": -1,
    },
}


def add_interpretable_axes(daily: pl.DataFrame) -> pl.DataFrame:
    """Add per-subject activity/recovery scores and their observed coverage.

    Each input is median-centered and divided by its per-subject interquartile
    range, then direction-aligned and averaged.  The axes therefore mean
    "above or below this person's usual level", not population fitness.
    """
    result = daily
    for axis, definitions in AXIS_FEATURES.items():
        available = {name: sign for name, sign in definitions.items() if name in result.columns}
        standardized: list[str] = []
        for name, sign in available.items():
            temp = f"__{axis}_{name}"
            median = pl.col(name).median().over("subject_id")
            iqr = (pl.col(name).quantile(0.75) - pl.col(name).quantile(0.25)).over("subject_id")
            result = result.with_columns(
                pl.when(iqr > 0)
                .then(sign * (pl.col(name) - median) / iqr)
                .otherwise(None)
                .alias(temp)
            )
            standardized.append(temp)
        result = result.with_columns(
            pl.mean_horizontal(*standardized).alias(f"{axis}_score"),
            pl.mean_horizontal(
                *(pl.col(name).is_not_null().cast(pl.Float64) for name in available)
            ).alias(f"{axis}_coverage"),
        ).drop(standardized)
    return result
