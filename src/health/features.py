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


@dataclass(frozen=True)
class AxisFeature:
    """A documented, direction-aligned ingredient of an interpretable axis."""

    name: str
    axis: str
    direction: float
    role: str
    outcome_only: bool = False


# The shared signals available in Apple Health, PMData/Fitbit and the supplied
# exports. Availability still varies by day; this is a common vocabulary, not
# a claim that every row contains all three values.
COMMON_FEATURES: tuple[str, ...] = (
    "steps",
    "sleep_hours",
    "hr_mean_bpm",
)

# Kept as the public name used by the window-selection pipeline.
CORE_FEATURES = COMMON_FEATURES

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
    "hr_p10_bpm",
    "hr_active_mean_bpm",
    "hr_active_excess_bpm",
    "hr_daily_cv",
    "hr_spectral_entropy",
    "hr_low_frequency_power",
)

# Rolling means are output alongside raw features. They are deliberately not
# included in MODEL_FEATURES: fitting on raw and multiple correlated smoothed
# copies would overweight these signals in PCA.
SMOOTH_FEATURES: tuple[str, ...] = (
    "steps",
    "sleep_hours",
    "hr_mean_bpm",
    "hr_p10_bpm",
    "hr_active_excess_bpm",
    "resting_hr_bpm",
    "activity_score",
    "recovery_score",
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
                "hr_samples": pl.UInt32,
                "hr_p10_bpm": pl.Float64,
                "hr_active_mean_bpm": pl.Float64,
                "hr_active_excess_bpm": pl.Float64,
                "hr_daily_cv": pl.Float64,
                "hr_spectral_entropy": pl.Float64,
                "hr_low_frequency_power": pl.Float64,
            }
        )

    summaries = hr.group_by("subject_id", "date").agg(
        pl.col("value_num").mean().alias("hr_mean_bpm"),
        pl.col("value_num").std().alias("hr_std_bpm"),
        (pl.col("value_num").max() - pl.col("value_num").min()).alias("hr_range_bpm"),
        pl.len().alias("hr_samples"),
        pl.when(pl.len() >= 10)
        .then(pl.col("value_num").quantile(0.1, interpolation="linear"))
        .otherwise(None)
        .alias("hr_p10_bpm"),
    )
    thresholds = hr.group_by("subject_id", "date").agg(
        pl.when(pl.len() >= 10)
        .then(pl.col("value_num").quantile(0.75, interpolation="linear"))
        .otherwise(None)
        .alias("hr_p75_bpm")
    )
    active_means = (
        hr.join(thresholds, on=["subject_id", "date"])
        .filter(pl.col("hr_p75_bpm").is_not_null() & (pl.col("value_num") >= pl.col("hr_p75_bpm")))
        .group_by("subject_id", "date")
        .agg(pl.col("value_num").mean().alias("hr_active_mean_bpm"))
    )
    summaries = summaries.join(active_means, on=["subject_id", "date"], how="left").with_columns(
        (pl.col("hr_active_mean_bpm") - pl.col("hr_p10_bpm")).alias("hr_active_excess_bpm"),
        pl.when((pl.col("hr_samples") >= 10) & (pl.col("hr_mean_bpm") > 0))
        .then(pl.col("hr_std_bpm") / pl.col("hr_mean_bpm"))
        .otherwise(None)
        .alias("hr_daily_cv"),
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
        "hr_active_excess_bpm": 1,
    },
    "recovery": {
        "sleep_hours": 1,
        "sleep_efficiency": 1,
        "hrv_sdnn_ms": 1,
        "sleep_score": 1,
        "sleep_restlessness": -1,
        "resting_hr_bpm": -1,
        "resting_hr_change_bpm": -1,
        "hr_active_excess_bpm": -1,
        "respiratory_rate": -1,
    },
}

AXIS_FEATURE_SPECS: tuple[AxisFeature, ...] = (
    AxisFeature("steps", "activity", 1, "movement volume"),
    AxisFeature("distance_m", "activity", 1, "movement volume"),
    AxisFeature("exercise_minutes", "activity", 1, "structured exercise"),
    AxisFeature("active_energy_kcal", "activity", 1, "energy expenditure"),
    AxisFeature("light_activity_hours", "activity", 1, "low-intensity movement"),
    AxisFeature("sedentary_hours", "activity", -1, "inactivity"),
    AxisFeature("hr_active_excess_bpm", "activity", 1, "active HR above resting proxy"),
    AxisFeature("sleep_hours", "recovery", 1, "sleep opportunity"),
    AxisFeature("sleep_efficiency", "recovery", 1, "sleep quality"),
    AxisFeature("hrv_sdnn_ms", "recovery", 1, "native HRV", outcome_only=False),
    AxisFeature("sleep_score", "recovery", 1, "device sleep score"),
    AxisFeature("sleep_restlessness", "recovery", -1, "sleep disruption"),
    AxisFeature("resting_hr_bpm", "recovery", -1, "resting physiology"),
    AxisFeature("resting_hr_change_bpm", "recovery", -1, "resting HR above baseline"),
    AxisFeature("hr_active_excess_bpm", "recovery", -1, "same-day cardiac load"),
    AxisFeature("respiratory_rate", "recovery", -1, "respiratory strain"),
    AxisFeature("readiness", "recovery", 1, "self-report outcome", outcome_only=True),
)


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


def add_smoothed_features(
    daily: pl.DataFrame,
    features: tuple[str, ...] = SMOOTH_FEATURES,
) -> pl.DataFrame:
    """Add trailing 7- and 28-calendar-day means without replacing raw values.

    Windows are causal (today and earlier only), calculated independently per
    subject, and require 3 or 7 real observations respectively. Missing days
    are not converted to zero. Change features compare the short trend with the
    28-day personal baseline.
    """
    result = daily.sort("subject_id", "date")
    available = [name for name in features if name in result.columns]
    expressions: list[pl.Expr] = []
    for name in available:
        expressions.extend(
            [
                pl.col(name)
                .rolling_mean_by("date", "7d", min_samples=3)
                .over("subject_id")
                .alias(f"{name}_7d"),
                pl.col(name)
                .rolling_mean_by("date", "28d", min_samples=7)
                .over("subject_id")
                .alias(f"{name}_28d"),
            ]
        )
    result = result.with_columns(expressions)
    changes: list[pl.Expr] = []
    change_names = {
        "resting_hr_bpm": "resting_hr_change_bpm",
        "hr_p10_bpm": "hr_p10_change_bpm",
    }
    for name, output in change_names.items():
        short, baseline = f"{name}_7d", f"{name}_28d"
        if short in result.columns and baseline in result.columns:
            changes.append((pl.col(short) - pl.col(baseline)).alias(output))
    return result.with_columns(changes)
