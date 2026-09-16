"""Build daily features and compare PCA with a small neural embedding model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from health.embeddings import (
    DenoisingAutoencoder,
    PCAEmbedding,
    embedding_frame,
    reconstruction_metrics,
    select_features,
    temporal_split,
)
from health.features import (
    CORE_FEATURES,
    MODEL_FEATURES,
    add_interpretable_axes,
    apply_windows,
    build_daily_features,
    select_dense_windows,
)


def run(input_dir: Path, output_dir: Path) -> dict[str, object]:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet files in {input_dir}")
    records = pl.concat([pl.read_parquet(path) for path in paths], how="vertical_relaxed")
    daily = build_daily_features(records)
    windows = select_dense_windows(daily, CORE_FEATURES)
    selected = add_interpretable_axes(apply_windows(daily, windows))
    train, test = temporal_split(selected)
    chosen = select_features(train, MODEL_FEATURES)

    pca = PCAEmbedding.fit(train, chosen)
    autoencoder = DenoisingAutoencoder.fit(train, chosen)
    pca_metrics = reconstruction_metrics(test, pca.preprocessor, pca.reconstruct(test))
    autoencoder_metrics = reconstruction_metrics(
        test, autoencoder.preprocessor, autoencoder.reconstruct(test)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected.write_parquet(output_dir / "daily_features.parquet")
    pca_rows = embedding_frame(selected, pca.transform(selected), "pc")
    neural_rows = embedding_frame(selected, autoencoder.transform(selected), "ae")
    pca_rows.join(neural_rows, on=["subject_id", "date"]).write_parquet(
        output_dir / "embeddings.parquet"
    )
    report: dict[str, object] = {
        "subjects": selected["subject_id"].n_unique(),
        "days": selected.height,
        "features": chosen,
        "windows": [asdict(window) for window in windows],
        "split": {
            "train_days": train.height,
            "test_days": test.height,
            "strategy": "latest 20% per subject",
        },
        "pca": {
            **pca_metrics,
            "explained_variance_ratio": pca.model.explained_variance_ratio_.tolist(),
            "loadings": {
                f"PC{component + 1}": dict(
                    zip(chosen, weights.tolist(), strict=True)
                )
                for component, weights in enumerate(pca.model.components_)
            },
        },
        "autoencoder": autoencoder_metrics,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/model"))
    args = parser.parse_args(argv)
    report = run(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
