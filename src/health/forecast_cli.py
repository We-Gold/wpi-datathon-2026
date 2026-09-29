"""Run leakage-safe next-day trajectory baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from health.features import (
    CORE_FEATURES,
    add_interpretable_axes,
    add_smoothed_features,
    apply_windows,
    build_daily_features,
    select_dense_windows,
)
from health.forecast import (
    DEFAULT_INPUTS,
    DEFAULT_TARGETS,
    build_forecast_table,
    evaluate_forecasters,
    temporal_forecast_split,
)
from health.stationarity import fit_differencing_plan

DEFAULT_HORIZONS = (1, 3, 7, 14)


def run(
    input_dir: Path,
    output_dir: Path,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, object]:
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive day counts")
    horizons = tuple(sorted(set(horizons)))
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet files in {input_dir}")
    records = pl.concat([pl.read_parquet(path) for path in paths], how="vertical_relaxed")
    daily = build_daily_features(records)
    windows = select_dense_windows(daily, CORE_FEATURES)
    selected = apply_windows(daily, windows)
    selected = add_smoothed_features(selected)
    selected = add_interpretable_axes(selected)
    selected = add_smoothed_features(selected, ("activity_score", "recovery_score"))
    stationarity_columns = tuple(
        name
        for name in dict.fromkeys((*DEFAULT_INPUTS, *DEFAULT_TARGETS))
        if name in selected.columns
    )
    differencing_orders, stationarity_diagnostics = fit_differencing_plan(
        selected, stationarity_columns, train_fraction=0.8
    )
    horizon_reports: dict[str, object] = {}
    prediction_frames: list[pl.DataFrame] = []
    for horizon in horizons:
        table = build_forecast_table(
            selected,
            horizon=horizon,
            differencing_orders=differencing_orders,
            stationarity_diagnostics=tuple(stationarity_diagnostics),
        )
        train, test = temporal_forecast_split(table)
        horizon_report, predictions = evaluate_forecasters(train, test, table)
        horizon_report.update(
            {
                "supervised_rows": table.frame.height,
                "train_rows": train.height,
                "test_rows": test.height,
                "input_columns": list(table.input_columns),
            }
        )
        horizon_reports[str(horizon)] = horizon_report
        prediction_frames.append(predictions.with_columns(pl.lit(horizon).alias("horizon_days")))
    all_predictions = pl.concat(prediction_frames, how="diagonal_relaxed")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions.write_parquet(output_dir / "forecast_predictions.parquet")
    report: dict[str, object] = {
        "horizons": horizon_reports,
        "horizon_days": list(horizons),
        "subjects": selected["subject_id"].n_unique(),
        "daily_rows": selected.height,
        "differencing_orders": differencing_orders,
        "stationarity_diagnostics": stationarity_diagnostics,
        "windows": [window.__dict__ for window in windows],
    }
    report.update(
        {
            "supervised_rows": sum(
                int(horizon_report["supervised_rows"])
                for horizon_report in horizon_reports.values()
            ),
        }
    )
    (output_dir / "forecast_evaluation.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model"))
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        metavar="DAYS",
        help="forecast horizons to evaluate (default: 1 3 7 14)",
    )
    args = parser.parse_args(argv)
    result = run(args.input_dir, args.output_dir, tuple(args.horizons))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
