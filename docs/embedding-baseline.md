# Daily health embedding baseline

This records the decisions behind the first reproducible embedding experiment.
They are defaults to test, not claims that the scores measure health or
clinical recovery.

## Decision summary

| Issue item | Decision |
|---|---|
| Feature set | Start with five cross-source core signals: steps, walking/running distance, exercise time, resting heart rate, and sleep duration. Keep richer cardiac, energy, sleep, and spectral signals as candidates and retain candidates with at least 30% coverage in training data. |
| Missing values | Preserve nulls in the feature artifact. Within each temporal training split, median-impute and standardize per subject. Carry an observed/missing mask into the neural model. Score reconstructions only where a real value was observed. Never treat an absent sensor value as zero. |
| Temporal aggregation | Use sums for cumulative measurements and means for sampled levels. Label sleep by a noon-to-noon day so a night crossing midnight stays together. For heart rate, add daily mean, standard deviation and range. Compute FFT features only with at least 48 of 96 15-minute bins and a 12-hour observed span. |
| Feature-rich periods | For each subject, find the longest interval whose trailing 30-day windows average at least 60% completeness over the core set. If no interval qualifies, retain the best 30-day diagnostic window and mark `meets_threshold=false`; do not silently drop that subject. |
| PCA evaluation | Hold out the latest 20% of each subject's selected interval. Fit feature selection, imputation, scaling, and PCA on earlier days only. Report explained variance plus held-out RMSE/MAE on observed cells, overall and per feature. |
| Neural embedding | Compare PCA to a small denoising autoencoder with the same temporal split and observed-cell metrics. It receives the missingness mask and randomly masks observed inputs during training. It is a feasibility benchmark, not the final model. |
| Interpretable axes | Activity and recovery are predefined composites of direction-aligned, per-person robust z-scores. They are kept separate from learned embeddings so their meaning does not rotate when a model is refit. |

## Activity and recovery axes

The activity score averages available, direction-aligned values for steps,
distance, exercise time, active energy, light activity, and negative sedentary
time. The recovery score does the same for sleep duration, sleep efficiency,
HRV, sleep score, negative restlessness, negative resting heart rate, and
negative respiratory rate. Every output row includes axis coverage.

These are **relative-to-self** scores. A value above zero means above that
subject's median in this dataset. It does not mean healthier than another
person, and it is not a medical recommendation. The equal weighting is
deliberately legible; learned or outcome-calibrated weights should only replace
it after an outcome and validation protocol are agreed.

## Why an embedding instead of one trend per metric?

Single-metric trends remain useful dashboard explanations and should be shown
beside the embedding. The embedding answers a different question: whether a
day's joint sleep, activity, and cardiac state resembles past multivariate
patterns. That shared state supports clustering, trajectory comparison,
next-day state prediction, and a decoder that can attribute movement back to
features. If those downstream tasks do not outperform simple per-feature
baselines, the embedding has not justified its complexity.

The next-day benchmark should compare:

1. persistence (tomorrow equals today),
2. a 7-day rolling mean,
3. a regularized linear model on the previous 7/30/90 days, and
4. only then an LSTM or transformer over embeddings.

Use rolling-origin evaluation and report error by subject as well as pooled.
Ninety days should be a tested context length, not a fixed assumption.

## Model acceptance criteria

PCA remains the default until a neural model:

- improves held-out observed-cell reconstruction and next-day forecasting;
- is stable across random seeds and subject-held-out sensitivity checks;
- does not primarily separate device/source type or missingness;
- yields clusters that recur across subjects and contiguous periods; and
- has decoder attributions that agree with the raw feature changes.

For very few subjects, leave-one-subject-out results are sensitivity analyses,
not evidence of population generalization. Bootstrap days in contiguous blocks
because adjacent health observations are correlated.

## Female-health data

Do not infer menstrual or pregnancy state from proxy signals. If participants
explicitly provide cycle data and consent to its use, keep it as optional
context and report performance/missingness by relevant groups. It should not be
a required core feature because that would systematically exclude subjects and
time periods without such data.

## Running the experiment

```sh
uv sync --all-groups
uv run health-model
```

This writes gitignored artifacts under `data/model/`:

- `daily_features.parquet`: selected daily rows, missingness, axis values and coverage;
- `embeddings.parquet`: PCA and autoencoder coordinates; and
- `evaluation.json`: windows, split, selected features, explained variance, and held-out errors.

After generating the artifacts, open the interactive visual report:

```sh
uv run marimo edit --watch --no-token notebooks/embeddings.py
```

It includes PCA/autoencoder trajectories, activity-versus-recovery movement,
the PCA scree chart and loading heatmap, per-feature held-out errors, and a
core-feature availability timeline.

The current local four-subject dataset is only a pipeline smoke test. On the
2026-09-16 run, PCA's first three components explained about 63% of training
variance. Held-out standardized RMSE was 0.730 for PCA and 0.703 for the
autoencoder. That small improvement supports further testing, not adoption of
the neural model.

## Branch audit

`main` contained PMData ingestion and unified metric names; it is now included
in this feature branch. `weaver-early-experimentation` contained a useful PCA
notebook prototype, including early sleep and HR spectral work, but it diverged
before the tested ingestion package and deletes that package if merged as-is.
The reusable ideas were reimplemented here against the canonical schema.
