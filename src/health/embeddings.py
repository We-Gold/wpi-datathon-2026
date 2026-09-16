"""Leakage-safe baselines for daily health embeddings.

PCA is the required linear baseline. ``DenoisingAutoencoder`` is intentionally
small: its job is to test whether non-linearity improves held-out
reconstruction before the project commits to a sequence model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor


def temporal_split(
    daily: pl.DataFrame, test_fraction: float = 0.2
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Put each subject's latest days in test; never randomly mix future into train."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    train_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []
    for frame in daily.sort("subject_id", "date").partition_by("subject_id", maintain_order=True):
        split = max(1, min(frame.height - 1, round(frame.height * (1 - test_fraction))))
        train_parts.append(frame[:split])
        test_parts.append(frame[split:])
    return pl.concat(train_parts), pl.concat(test_parts)


def select_features(
    train: pl.DataFrame, candidates: tuple[str, ...], minimum_coverage: float = 0.3
) -> list[str]:
    """Select using training data only, preventing test-set availability leakage."""
    present = [name for name in candidates if name in train.columns]
    return [
        name
        for name in present
        if float(train.select(pl.col(name).is_not_null().mean()).item()) >= minimum_coverage
    ]


@dataclass
class MaskedStandardizer:
    """Per-subject median imputation and scaling fitted on training rows only."""

    features: list[str]
    statistics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    fallback: tuple[np.ndarray, np.ndarray, np.ndarray]

    @classmethod
    def fit(cls, frame: pl.DataFrame, features: list[str]) -> MaskedStandardizer:
        if not features:
            raise ValueError("no features passed the coverage threshold")
        values = frame.select(features).to_numpy()
        fallback = cls._stats(values)
        statistics = {
            str(subject.item(0, "subject_id")): cls._stats(subject.select(features).to_numpy())
            for subject in frame.partition_by("subject_id", maintain_order=True)
        }
        return cls(features, statistics, fallback)

    @staticmethod
    def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            # An entire feature can be absent for one device ecosystem. It is
            # mapped to standardized zero while its mask remains false.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values, axis=0)
            mean = np.nanmean(values, axis=0)
            scale = np.nanstd(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        mean = np.where(np.isfinite(mean), mean, median)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return median, mean, scale

    def transform(self, frame: pl.DataFrame, *, include_mask: bool = False) -> np.ndarray:
        parts: list[np.ndarray] = []
        for row in frame.select("subject_id", *self.features).iter_rows(named=True):
            values = np.asarray([row[name] for name in self.features], dtype=float)
            missing = ~np.isfinite(values)
            median, mean, scale = self.statistics.get(str(row["subject_id"]), self.fallback)
            scaled = (np.where(missing, median, values) - mean) / scale
            parts.append(
                np.concatenate([scaled, (~missing).astype(float)]) if include_mask else scaled
            )
        width = len(self.features) * (2 if include_mask else 1)
        return np.vstack(parts) if parts else np.empty((0, width))


@dataclass
class PCAEmbedding:
    preprocessor: MaskedStandardizer
    model: PCA

    @classmethod
    def fit(cls, train: pl.DataFrame, features: list[str], components: int = 3) -> PCAEmbedding:
        preprocessor = MaskedStandardizer.fit(train, features)
        matrix = preprocessor.transform(train)
        count = min(components, matrix.shape[0], matrix.shape[1])
        if count < 1:
            raise ValueError("PCA needs at least one row and one feature")
        return cls(preprocessor, PCA(n_components=count, random_state=0).fit(matrix))

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.transform(self.preprocessor.transform(frame))

    def reconstruct(self, frame: pl.DataFrame) -> np.ndarray:
        embedded = self.transform(frame)
        return self.model.inverse_transform(embedded)


@dataclass
class DenoisingAutoencoder:
    preprocessor: MaskedStandardizer
    model: MLPRegressor
    latent_layer: int = 1

    @classmethod
    def fit(
        cls,
        train: pl.DataFrame,
        features: list[str],
        *,
        latent_dimensions: int = 3,
        corruption: float = 0.15,
        random_state: int = 0,
    ) -> DenoisingAutoencoder:
        preprocessor = MaskedStandardizer.fit(train, features)
        target = preprocessor.transform(train)
        observed_input = preprocessor.transform(train, include_mask=True)
        rng = np.random.default_rng(random_state)
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        feature_count = len(features)
        for _ in range(3):
            noisy = observed_input.copy()
            removable = noisy[:, feature_count:] > 0
            removed = (rng.random(removable.shape) < corruption) & removable
            noisy[:, :feature_count][removed] = 0.0
            noisy[:, feature_count:][removed] = 0.0
            inputs.append(noisy)
            targets.append(target)
        width = max(8, min(64, feature_count * 2))
        model = MLPRegressor(
            hidden_layer_sizes=(width, latent_dimensions, width),
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=500,
            random_state=random_state,
        )
        model.fit(np.vstack(inputs), np.vstack(targets))
        return cls(preprocessor, model)

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        activation = self.preprocessor.transform(frame, include_mask=True)
        for index, (weights, bias) in enumerate(
            zip(self.model.coefs_, self.model.intercepts_, strict=True)
        ):
            activation = np.maximum(0.0, activation @ weights + bias)
            if index == self.latent_layer:
                return activation
        raise RuntimeError("autoencoder does not contain the requested latent layer")

    def reconstruct(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(frame, include_mask=True))


def reconstruction_metrics(
    frame: pl.DataFrame, preprocessor: MaskedStandardizer, reconstructed: np.ndarray
) -> dict[str, object]:
    """Score only actually observed cells; imputed targets do not earn credit."""
    target = preprocessor.transform(frame)
    observed = frame.select(preprocessor.features).to_numpy()
    mask = np.isfinite(observed)
    squared = (target - reconstructed) ** 2
    absolute = np.abs(target - reconstructed)
    per_feature = {
        name: float(np.sqrt(squared[:, index][mask[:, index]].mean()))
        if mask[:, index].any()
        else None
        for index, name in enumerate(preprocessor.features)
    }
    return {
        "rmse_standardized": float(np.sqrt(squared[mask].mean())),
        "mae_standardized": float(absolute[mask].mean()),
        "rmse_by_feature": per_feature,
        "observed_cells": int(mask.sum()),
    }


def embedding_frame(frame: pl.DataFrame, values: np.ndarray, prefix: str) -> pl.DataFrame:
    columns = [
        pl.Series(f"{prefix}_{index + 1}", values[:, index]) for index in range(values.shape[1])
    ]
    return frame.select("subject_id", "date").with_columns(columns)
