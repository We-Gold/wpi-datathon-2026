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
from health.forecast import build_forecast_table, evaluate_forecasters, temporal_forecast_split


def run(input_dir: Path, output_dir: Path) -> dict[str, object]:
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
    table = build_forecast_table(selected)
    train, test = temporal_forecast_split(table)
    report, predictions = evaluate_forecasters(train, test, table)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "forecast_predictions.parquet")
    report.update(
        {
            "subjects": selected["subject_id"].n_unique(),
            "daily_rows": selected.height,
            "supervised_rows": table.frame.height,
            "train_rows": train.height,
            "test_rows": test.height,
            "input_columns": list(table.input_columns),
            "windows": [window.__dict__ for window in windows],
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
    args = parser.parse_args(argv)
    print(json.dumps(run(args.input_dir, args.output_dir), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
