"""Write the dashboard's static JSON from `daily_features.parquet`.

Run `health-model` first. Output holds health data, so the default output
directory is gitignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from health.features import AXIS_FEATURES
from health.trajectory import (
    AXES,
    BAND_QUANTILES,
    FORECAST_LAGS,
    HALF_LIFE_DAYS,
    KEEP,
    PULL_FACTOR,
    RIDGE_ALPHA,
    RidgeForecast,
    add_positions,
    contribution_name,
    factor_keys,
    fit_clusters,
    position_name,
    velocity_name,
)

HISTORY_DAYS = 180
SERIES_DAYS = 365
# Daily values shown in the metric cards and trend table.
CARD_METRICS: tuple[str, ...] = (
    "steps",
    "resting_hr_bpm",
    "sleep_hours",
    "hrv_sdnn_ms",
    "active_energy_kcal",
    "exercise_minutes",
)


def _vector(row: dict[str, object], name) -> dict[str, float]:
    return {axis: float(row[name(axis)]) for axis in AXES}


def subject_payload(
    raw: pl.DataFrame, filled: pl.DataFrame, forecast: RidgeForecast
) -> dict[str, object]:
    """Everything the dashboard needs for one subject."""
    subject_id = str(filled.item(0, "subject_id"))
    as_of = raw["date"].max()
    filled = filled.filter(pl.col("date") <= as_of)
    rows = filled.tail(HISTORY_DAYS).to_dicts()
    history = [
        {
            "date": row["date"].isoformat(),
            "position": _vector(row, position_name),
            "velocity": _vector(row, velocity_name),
        }
        for row in rows
    ]
    today = rows[-1]
    contributions = [
        {
            "factor": factor,
            **{axis: float(today[contribution_name(axis, factor)]) for axis in AXES},
        }
        for factor in factor_keys()
        if any(today[contribution_name(axis, factor)] != 0 for axis in AXES)
    ]
    points = filled.select(position_name(a) for a in AXES).to_numpy()
    series = [
        {
            "metric": metric,
            "points": [
                {"date": d.isoformat(), "value": float(v)}
                for d, v in raw.tail(SERIES_DAYS).select("date", metric).drop_nulls().iter_rows()
            ],
        }
        for metric in CARD_METRICS
        if metric in raw.columns and raw[metric].is_not_null().any()
    ]
    # Same spread `add_axis_components` divides by.
    iqr = {}
    for feature in {f for axis in AXES for f in AXIS_FEATURES[axis]} & set(raw.columns):
        if raw[feature].is_not_null().any():
            spread = raw[feature].quantile(0.75) - raw[feature].quantile(0.25)
            if spread > 0:
                iqr[feature] = float(spread)
    # A what-if changes today's score by its share of today's observed ingredients.
    last_observed = filled.filter(pl.all_horizontal(pl.col(f"{a}_count") > 0 for a in AXES)).tail(1)
    counts = {
        axis: int(last_observed.item(0, f"{axis}_count")) if last_observed.height else 0
        for axis in AXES
    }
    return {
        "trajectory": {
            "subjectId": subject_id,
            "asOf": as_of.isoformat(),
            "keep": KEEP,
            "history": history,
            "prediction": forecast.predict_from(filled),
            "clusters": fit_clusters(points),
            "contributions": contributions,
        },
        "series": series,
        "scaling": {"iqr": dict(sorted(iqr.items())), "counts": counts},
    }


def run(daily_path: Path, output_dir: Path) -> dict[str, object]:
    daily = pl.read_parquet(daily_path).with_columns(pl.col("subject_id").cast(pl.String))
    filled = add_positions(daily)
    forecast = RidgeForecast.fit(filled)

    subjects_dir = output_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    subjects = []
    for raw in daily.sort("subject_id", "date").partition_by("subject_id", maintain_order=True):
        subject_id = str(raw.item(0, "subject_id"))
        payload = subject_payload(raw, filled.filter(pl.col("subject_id") == subject_id), forecast)
        (subjects_dir / f"{subject_id}.json").write_text(json.dumps(payload) + "\n")
        subjects.append(
            {"subjectId": subject_id, "label": subject_id.replace("_", " ").capitalize()}
        )

    model = {
        "halfLifeDays": HALF_LIFE_DAYS,
        "keep": KEEP,
        "axes": AXIS_FEATURES,
        "factors": factor_keys(),
        "pullFactor": PULL_FACTOR,
        "clusterRule": "healthy when activity + recovery at the cluster center is at least 0",
        "forecast": {
            "model": "Ridge, one per horizon, pooled across subjects",
            "alpha": RIDGE_ALPHA,
            "lags": list(FORECAST_LAGS),
            "bandQuantiles": list(BAND_QUANTILES),
            "coefficients": forecast.coefficients(),
            "evaluation": {str(h): v for h, v in forecast.evaluation.items()},
        },
    }
    index = {"subjects": subjects, "model": model}
    (output_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    summary = {
        h: {a: round(v[a]["ridge_mae"], 3) for a in AXES}
        | {"coverage": round(float(np.mean([v[a]["band_coverage"] for a in AXES])), 2)}
        for h, v in forecast.evaluation.items()
        if h in (1, 7, 14, 30)
    }
    return {"subjects": len(subjects), "output": str(output_dir), "ridge": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=Path("data/model/daily_features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("dashboard/public/data"))
    args = parser.parse_args(argv)
    print(json.dumps(run(args.daily, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
