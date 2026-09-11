"""Canonical metric names.

HealthKit identifiers are the vocabulary. The flat CSV strips the prefix and
writes "StepCount", so readers put it back. Anything without an HK prefix is not
a HealthKit metric.

Every name follows one of two rules. A new reader picks the first that fits.

1. HealthKit has a name for it. Use that name exactly as HealthKit spells it,
   even when the spelling is odd: the Apple export writes
   HKDataTypeSleepDurationGoal, which is not one of the TypeIdentifier forms,
   and it is kept verbatim rather than tidied into one. Two sources measuring
   the same thing have to land on the same string or a group-by splits them.

2. HealthKit has no name for it. Use a plain name that says what was measured.

No vendor or source prefixes. Which device or app a row came from is already in
the source_type and source_format columns, and a prefix there would be wrong as
often as right: PMData's step counts come off a Fitbit but are HealthKit step
counts, so a Fitbit prefix would have described the concept, not the source.

No units in names either. health.units converts everything on the way in and
drops the recorded unit, so a name like SedentaryMinutes goes stale the moment
the value becomes seconds. Ask the unit column, or CATALOGUE below.

The one place this leaks is rule 1. Three HealthKit names end in Percentage but
hold fractions between 0 and 1, because Apple records them that way. Renaming
them would stop them matching the same metric from another export, so they keep
Apple's spelling and CATALOGUE carries the real unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

QUANTITY_PREFIX = "HKQuantityTypeIdentifier"
CATEGORY_PREFIX = "HKCategoryTypeIdentifier"
VALUE_PREFIX = "HKCategoryValue"

# Short aliases, used by the CATALOGUE keys below to keep them one line each.
_Q = QUANTITY_PREFIX
_C = CATEGORY_PREFIX

# The only category type in the flat CSV. Everything else there is a quantity.
_CATEGORY_METRICS = frozenset({"SleepAnalysis"})

# The nested JSON ships sleep already rolled up to a daily total, while the
# other formats give per-stage intervals. Sharing a name would let one filter
# return both grains and sum them. The unit is canonicalized like any other, so
# the name does not carry one.
SLEEP_DURATION_DAILY = "SleepDurationDaily"

DERIVED_METRICS = frozenset({SLEEP_DURATION_DAILY})

# PMData holds two kinds of measurement that HealthKit has no name for. They
# get names of their own here instead of being dropped, the way
# SLEEP_DURATION_DAILY already does. The prefix says where the number came from
# and warns that it is not the same thing as an Apple metric.

# Fitbit's own view of the body. None of these is an Apple metric under a new
# name. FitbitCaloriesTotal is basal plus activity, but Apple splits those into
# two metrics, and the heart rate zone limits are set by Fitbit.
TOTAL_ENERGY_BURNED = "TotalEnergyBurned"

VENDOR_METRICS = frozenset(
    {
        TOTAL_ENERGY_BURNED,
        "SedentaryTime",
        "LightActivityTime",
        # Fitbit's own sleep score. Its three components are not kept: they add
        # up to this number exactly, in all 1836 rows of the real dataset, so
        # storing them as well would store the same thing twice.
        "SleepScore",
        "SleepRestlessness",
    }
)

# Read from PMData and deliberately not kept. Listed so the reason survives and
# nobody adds them back wondering why they were missed.
#
#   FitbitSleepCompositionScore     the three parts of FitbitSleepScore, which
#   FitbitSleepRevitalizationScore  sum to it exactly
#   FitbitSleepDurationScore
#   FitbitTimeBelowZone1            heart rate zones, derivable from the heart
#   FitbitTimeInZone1               rate we already keep at 5 second grain, and
#   FitbitTimeInZone2               the limits are set per person so they do not
#   FitbitTimeInZone3               compare between subjects
#   MealsReported                   a count of meal labels with no time, no
#                                   amount and no nutrition, so nothing to
#                                   compare it against
#   AlcoholConsumed                 four answers, two of which are "Maybe" and a
#                                   refusal to say
#
# DietaryWater is dropped from PMData too, but it is not in the set below. The
# food log counts glasses rather than volume, so a figure in mL could only come
# from guessing a glass size. Other sources record the volume itself, so the
# metric stays in the vocabulary and only PMData stops contributing to it.
DROPPED_PMDATA_METRICS = frozenset(
    {
        "SleepCompositionScore",
        "SleepRevitalizationScore",
        "SleepDurationScore",
        "TimeBelowHeartRateZone1",
        "TimeInHeartRateZone1",
        "TimeInHeartRateZone2",
        "TimeInHeartRateZone3",
        "MealsReported",
        "AlcoholConsumed",
    }
)

# Answers a person typed into the PMSys athlete app, not a sensor reading.
# Worth keeping, because it is the one kind of data in PMData that no wearable
# export has.
SELF_REPORT_METRICS = frozenset(
    {
        "Fatigue",
        "Mood",
        "Readiness",
        "SleepQuality",
        "Soreness",
        "Stress",
        "PerceivedExertion",
        "TrainingDuration",
        "InjuryCount",
    }
)

# What validation accepts alongside HealthKit identifiers.
NON_HEALTHKIT_METRICS = DERIVED_METRICS | VENDOR_METRICS | SELF_REPORT_METRICS

# Fitbit's sleep names mapped onto HealthKit's. There are two sets of names
# because a "stages" log comes from the optical sensor and a "classic" log does
# not. A classic log only sees movement, so its sleep cannot claim a stage. It
# maps to Unspecified instead of being promoted to Core.
FITBIT_SLEEP_STAGES: dict[str, str] = {
    # type: "stages"
    "wake": "Awake",
    "light": "AsleepCore",
    "deep": "AsleepDeep",
    "rem": "AsleepREM",
    # type: "classic"
    "awake": "Awake",
    "asleep": "AsleepUnspecified",
    "restless": "AsleepUnspecified",
}


# The flat CSV strips the prefix from category values the same way it strips it
# from metric names, so "AsleepCore" and "HKCategoryValueSleepAnalysisAsleepCore"
# are the same sleep stage. The prefix cannot be derived from the metric name:
# AudioExposureEvent spells its values HKCategoryValueEnvironmentalAudioExposure
# Event, and MindfulSession uses HKCategoryValueNotApplicable. So it is listed
# per metric, and a metric with no entry is left alone.


# What every metric is, for code that has to decide something about one. Names
# say what was measured and nothing else, so anything a name used to imply
# lives here instead: the unit that was thrown away on conversion, the domain a
# metric belongs to, and how it collapses into one number a day.
#
# Feature engineering reads this rather than matching on name prefixes. The
# HealthKit names cannot be grouped by prefix anyway. HeartRate,
# RestingHeartRate, HeartRateVariabilitySDNN, RespiratoryRate and VO2Max are all
# cardiac and share no common string, and we do not get to rename them.


@dataclass(frozen=True)
class MetricInfo:
    """One row of the catalogue.

    unit is the canonical unit after health.units has run, which is what is
    actually stored. It is None for category metrics, which carry a value_str
    and no number.

    daily is how the metric collapses into one number per day:

        sum       cumulative over the day: steps, energy, distance, food
        mean      a level sampled many times: heart rate, speed, a rating
        duration  category intervals, added up in seconds: sleep stages
        count     category events with no meaningful length: stand hours
        skip      not a property of a day: height, waist, a goal setting

    subjective marks a rating on an opinion scale. The numbers are ordered
    labels, so a difference of one does not mean the same thing at both ends,
    and nothing may be compared across two of them. A measured score such as
    SleepScore is not subjective, even though it is also a score.
    """

    domain: str
    unit: str | None
    daily: str
    subjective: bool = False


DOMAINS = (
    "activity",
    "body",
    "cardiac",
    "energy",
    "exposure",
    "hygiene",
    "mobility",
    "nutrition",
    "sleep",
    "wellbeing",
    "workout",
)

DAILY_RULES = ("sum", "mean", "duration", "count", "skip")

CATALOGUE: dict[str, MetricInfo] = {
    # --- activity ---
    _C + "AppleStandHour": MetricInfo("activity", None, "count"),
    _Q + "AppleExerciseTime": MetricInfo("activity", "s", "sum"),
    _Q + "AppleStandTime": MetricInfo("activity", "s", "sum"),
    _Q + "DistanceWalkingRunning": MetricInfo("activity", "m", "sum"),
    _Q + "FlightsClimbed": MetricInfo("activity", "count", "sum"),
    _Q + "PhysicalEffort": MetricInfo("activity", "kcal/hr·kg", "mean"),
    _Q + "StepCount": MetricInfo("activity", "count", "sum"),
    "LightActivityTime": MetricInfo("activity", "s", "sum"),
    "SedentaryTime": MetricInfo("activity", "s", "sum"),
    # --- body ---
    _Q + "BodyFatPercentage": MetricInfo("body", "fraction", "mean"),
    _Q + "BodyMass": MetricInfo("body", "kg", "mean"),
    _Q + "BodyMassIndex": MetricInfo("body", "count", "mean"),
    _Q + "Height": MetricInfo("body", "m", "skip"),
    _Q + "LeanBodyMass": MetricInfo("body", "kg", "mean"),
    _Q + "WaistCircumference": MetricInfo("body", "m", "skip"),
    # --- cardiac ---
    _Q + "HeartRate": MetricInfo("cardiac", "count/min", "mean"),
    _Q + "HeartRateRecoveryOneMinute": MetricInfo("cardiac", "count/min", "mean"),
    _Q + "HeartRateVariabilitySDNN": MetricInfo("cardiac", "s", "mean"),
    _Q + "OxygenSaturation": MetricInfo("cardiac", "fraction", "mean"),
    _Q + "RespiratoryRate": MetricInfo("cardiac", "count/min", "mean"),
    _Q + "RestingHeartRate": MetricInfo("cardiac", "count/min", "mean"),
    _Q + "VO2Max": MetricInfo("cardiac", "mL/min·kg", "mean"),
    _Q + "WalkingHeartRateAverage": MetricInfo("cardiac", "count/min", "mean"),
    # --- energy ---
    _Q + "ActiveEnergyBurned": MetricInfo("energy", "kcal", "sum"),
    _Q + "BasalEnergyBurned": MetricInfo("energy", "kcal", "sum"),
    "TotalEnergyBurned": MetricInfo("energy", "kcal", "sum"),
    # --- exposure ---
    _C + "AudioExposureEvent": MetricInfo("exposure", None, "count"),
    _C + "HeadphoneAudioExposureEvent": MetricInfo("exposure", None, "count"),
    _Q + "EnvironmentalAudioExposure": MetricInfo("exposure", "dBASPL", "mean"),
    _Q + "HeadphoneAudioExposure": MetricInfo("exposure", "dBASPL", "mean"),
    _Q + "TimeInDaylight": MetricInfo("exposure", "s", "sum"),
    # --- hygiene ---
    _C + "ToothbrushingEvent": MetricInfo("hygiene", None, "count"),
    # --- mobility ---
    _Q + "AppleWalkingSteadiness": MetricInfo("mobility", "fraction", "mean"),
    _Q + "SixMinuteWalkTestDistance": MetricInfo("mobility", "m", "mean"),
    _Q + "StairAscentSpeed": MetricInfo("mobility", "m/s", "mean"),
    _Q + "StairDescentSpeed": MetricInfo("mobility", "m/s", "mean"),
    _Q + "WalkingAsymmetryPercentage": MetricInfo("mobility", "fraction", "mean"),
    _Q + "WalkingDoubleSupportPercentage": MetricInfo("mobility", "fraction", "mean"),
    _Q + "WalkingSpeed": MetricInfo("mobility", "m/s", "mean"),
    _Q + "WalkingStepLength": MetricInfo("mobility", "m", "mean"),
    # --- nutrition ---
    _Q + "DietaryCalcium": MetricInfo("nutrition", "mg", "sum"),
    _Q + "DietaryCarbohydrates": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryCholesterol": MetricInfo("nutrition", "mg", "sum"),
    _Q + "DietaryEnergyConsumed": MetricInfo("nutrition", "kcal", "sum"),
    _Q + "DietaryFatMonounsaturated": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryFatPolyunsaturated": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryFatSaturated": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryFatTotal": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryFiber": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryIron": MetricInfo("nutrition", "mg", "sum"),
    _Q + "DietaryPotassium": MetricInfo("nutrition", "mg", "sum"),
    _Q + "DietaryProtein": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietarySodium": MetricInfo("nutrition", "mg", "sum"),
    _Q + "DietarySugar": MetricInfo("nutrition", "g", "sum"),
    _Q + "DietaryVitaminC": MetricInfo("nutrition", "mg", "sum"),
    _Q + "DietaryWater": MetricInfo("nutrition", "mL", "sum"),
    # --- sleep ---
    _C + "SleepAnalysis": MetricInfo("sleep", None, "duration"),
    "HKDataTypeSleepDurationGoal": MetricInfo("sleep", "s", "skip"),
    "SleepDurationDaily": MetricInfo("sleep", "s", "sum"),
    "SleepQuality": MetricInfo("sleep", "score", "mean", subjective=True),
    "SleepRestlessness": MetricInfo("sleep", "fraction", "mean"),
    "SleepScore": MetricInfo("sleep", "score", "mean"),
    # --- wellbeing ---
    "Fatigue": MetricInfo("wellbeing", "score", "mean", subjective=True),
    _C + "MindfulSession": MetricInfo("wellbeing", None, "duration"),
    "InjuryCount": MetricInfo("wellbeing", "count", "mean"),
    "Mood": MetricInfo("wellbeing", "score", "mean", subjective=True),
    "Readiness": MetricInfo("wellbeing", "score", "mean", subjective=True),
    "Soreness": MetricInfo("wellbeing", "score", "mean", subjective=True),
    "Stress": MetricInfo("wellbeing", "score", "mean", subjective=True),
    # --- workout ---
    _Q + "CyclingCadence": MetricInfo("workout", "count/min", "mean"),
    _Q + "CyclingPower": MetricInfo("workout", "W", "mean"),
    _Q + "DistanceCycling": MetricInfo("workout", "m", "sum"),
    _Q + "DistancePaddleSports": MetricInfo("workout", "m", "sum"),
    _Q + "PaddleSportsSpeed": MetricInfo("workout", "m/s", "mean"),
    _Q + "RunningGroundContactTime": MetricInfo("workout", "s", "mean"),
    _Q + "RunningPower": MetricInfo("workout", "W", "mean"),
    _Q + "RunningSpeed": MetricInfo("workout", "m/s", "mean"),
    _Q + "RunningStrideLength": MetricInfo("workout", "m", "mean"),
    _Q + "RunningVerticalOscillation": MetricInfo("workout", "m", "mean"),
    "PerceivedExertion": MetricInfo("workout", "score", "mean", subjective=True),
    "TrainingDuration": MetricInfo("workout", "s", "sum"),
}


def in_domain(domain: str) -> list[str]:
    """Every catalogued metric in one domain, sorted."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}. Known: {list(DOMAINS)}")
    return sorted(name for name, info in CATALOGUE.items() if info.domain == domain)


def subjective_metrics() -> list[str]:
    """Every metric that is a rating rather than a measurement."""
    return sorted(name for name, info in CATALOGUE.items() if info.subjective)


def by_daily_rule() -> dict[str, list[str]]:
    """Catalogued metrics grouped by how they collapse into a day."""
    grouped: dict[str, list[str]] = {rule: [] for rule in DAILY_RULES}
    for name, info in CATALOGUE.items():
        grouped[info.daily].append(name)
    return {rule: sorted(names) for rule, names in grouped.items()}


_VALUE_PREFIXES: dict[str, str] = {
    CATEGORY_PREFIX + "SleepAnalysis": VALUE_PREFIX + "SleepAnalysis",
}


def restore_prefix(metric: pl.Expr) -> pl.Expr:
    """Put the HK prefix back on a bare metric name. Prefixed names pass through."""
    return (
        pl.when(metric.str.starts_with("HK"))
        .then(metric)
        .when(metric.is_in(list(_CATEGORY_METRICS)))
        .then(pl.lit(CATEGORY_PREFIX) + metric)
        .otherwise(pl.lit(QUANTITY_PREFIX) + metric)
    )


def is_healthkit(name: str) -> bool:
    """Whether a canonical metric name came from HealthKit unmodified."""
    return name.startswith("HK")


def restore_value_prefix(metric: pl.Expr, value: pl.Expr) -> pl.Expr:
    """Put the HKCategoryValue prefix back on a bare category value.

    Values that already carry it, and metrics with no rule, pass through.
    """
    prefix = metric.replace_strict(_VALUE_PREFIXES, default=None, return_dtype=pl.String)
    bare = value.is_not_null() & ~value.str.starts_with(VALUE_PREFIX)
    return pl.when(bare & prefix.is_not_null()).then(prefix + value).otherwise(value)
