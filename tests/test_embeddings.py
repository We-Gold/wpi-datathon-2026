from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl

from health.embeddings import (
    MaskedStandardizer,
    PCAEmbedding,
    fit_tsne,
    fit_umap,
    reconstruction_metrics,
    temporal_split,
)


def _daily() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": ["a"] * 10 + ["b"] * 10,
            "date": pl.date_range(date(2026, 1, 1), date(2026, 1, 10), eager=True).to_list() * 2,
            "one": [float(value) for value in range(10)] * 2,
            "two": [None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] * 2,
        }
    ).with_columns(pl.col("subject_id").cast(pl.Categorical))


def test_temporal_split_holds_out_latest_days_per_subject() -> None:
    train, test = temporal_split(_daily(), test_fraction=0.2)
    assert train.group_by("subject_id").len()["len"].to_list() == [8, 8]
    assert test.group_by("subject_id").agg(pl.col("date").min())["date"].to_list() == [
        date(2026, 1, 9),
        date(2026, 1, 9),
    ]


def test_standardizer_preserves_missingness_mask() -> None:
    daily = _daily()
    standardizer = MaskedStandardizer.fit(daily, ["one", "two"])
    matrix = standardizer.transform(daily[:2], include_mask=True)
    assert matrix.shape == (2, 4)
    assert matrix[0, 2:].tolist() == [1.0, 0.0]


def test_pca_scores_only_observed_cells() -> None:
    daily = _daily()
    model = PCAEmbedding.fit(daily, ["one", "two"], components=1)
    scores = reconstruction_metrics(daily, model.preprocessor, model.reconstruct(daily))
    assert scores["observed_cells"] == 38
    assert cast(float, scores["rmse_standardized"]) >= 0


def test_umap_and_tsne_return_two_dimensional_visualization_coordinates() -> None:
    daily = _daily()
    umap_values = fit_umap(daily, ["one", "two"], neighbors=5)
    tsne_values = fit_tsne(daily, ["one", "two"], perplexity=3)
    assert umap_values.shape == (20, 2)
    assert tsne_values.shape == (20, 2)
