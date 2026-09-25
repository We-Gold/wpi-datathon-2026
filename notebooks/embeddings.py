import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    alt.data_transformers.disable_max_rows()
    return Path, alt, json, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Daily health embeddings

    Explore the selected feature-rich periods, interpretable activity/recovery
    axes, and the PCA and autoencoder baselines. Scores are relative to each
    subject and are not clinical measurements.
    """)
    return


@app.cell
def _(Path, json, pl):
    ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR = ROOT / "data" / "model"
    DAILY_PATH = MODEL_DIR / "daily_features.parquet"
    EMBEDDING_PATH = MODEL_DIR / "embeddings.parquet"
    REPORT_PATH = MODEL_DIR / "evaluation.json"

    artifacts_exist = all(path.exists() for path in (DAILY_PATH, EMBEDDING_PATH, REPORT_PATH))
    if artifacts_exist:
        daily = pl.read_parquet(DAILY_PATH)
        embedding = pl.read_parquet(EMBEDDING_PATH)
        report = json.loads(REPORT_PATH.read_text())
        combined = daily.join(embedding, on=["subject_id", "date"], how="inner")
    else:
        daily = pl.DataFrame(
            schema={
                "subject_id": pl.String,
                "date": pl.Date,
                "activity_score": pl.Float64,
                "recovery_score": pl.Float64,
            }
        )
        embedding = pl.DataFrame()
        combined = daily
        report = {}
    return MODEL_DIR, artifacts_exist, combined, report


@app.cell
def _(MODEL_DIR, artifacts_exist, mo):
    mo.md(
        f"Loaded model artifacts from `{MODEL_DIR}`."
        if artifacts_exist
        else "Run `uv run health-model` first, then rerun this notebook."
    )
    return


@app.cell
def _(combined, mo):
    subjects = combined["subject_id"].cast(str).unique().sort().to_list() if combined.height else []
    subject_pick = mo.ui.dropdown(
        subjects,
        value=subjects[0] if subjects else None,
        label="Subject",
    )
    model_pick = mo.ui.dropdown(
        {"PCA": "pc", "Autoencoder": "ae", "UMAP": "umap", "t-SNE": "tsne"},
        value="PCA",
        label="Embedding",
    )
    history_pick = mo.ui.slider(
        start=30,
        stop=730,
        step=10,
        value=180,
        label="Latest days",
        show_value=True,
    )
    smoothing_pick = mo.ui.dropdown(
        {"Raw daily": 1, "7-day mean": 7, "28-day mean": 28},
        value="7-day mean",
        label="Smoothing",
    )
    mo.hstack(
        [subject_pick, model_pick, smoothing_pick, history_pick], justify="start", gap=2
    )
    return history_pick, model_pick, smoothing_pick, subject_pick


@app.cell
def _(combined, history_pick, pl, smoothing_pick, subject_pick):
    if combined.height and subject_pick.value is not None:
        subject_view = combined.filter(
            pl.col("subject_id").cast(pl.String) == subject_pick.value
        ).sort("date")
        smoothing_days = smoothing_pick.value
        if smoothing_days > 1:
            minimum = 3 if smoothing_days == 7 else 7
            coordinate_columns = [
                name
                for name in ("pc_1", "pc_2", "ae_1", "ae_2")
                if name in subject_view.columns
            ]
            subject_view = subject_view.with_columns(
                *(
                    pl.col(name)
                    .rolling_mean_by("date", f"{smoothing_days}d", min_samples=minimum)
                    .alias(name)
                    for name in coordinate_columns
                )
            )
            for axis in ("activity_score", "recovery_score"):
                smoothed = f"{axis}_{smoothing_days}d"
                if smoothed in subject_view.columns:
                    subject_view = subject_view.with_columns(pl.col(smoothed).alias(axis))
        subject_view = subject_view.tail(history_pick.value)
    else:
        subject_view = combined
        smoothing_days = smoothing_pick.value
    return smoothing_days, subject_view


@app.cell
def _(alt, mo, model_pick, smoothing_days, subject_view):
    prefix = model_pick.value
    x_name = f"{prefix}_1"
    y_name = f"{prefix}_2"
    if subject_view.height and x_name in subject_view.columns and y_name in subject_view.columns:
        trajectory_line = (
            alt.Chart(subject_view)
            .mark_line(opacity=0.35)
            .encode(x=alt.X(f"{x_name}:Q", title=f"{model_pick.selected_key} dimension 1"),
                    y=alt.Y(f"{y_name}:Q", title=f"{model_pick.selected_key} dimension 2"),
                    order="date:T")
        )
        trajectory_points = (
            alt.Chart(subject_view)
            .mark_circle(size=45)
            .encode(
                x=f"{x_name}:Q",
                y=f"{y_name}:Q",
                color=alt.Color("date:T", title="Date", scale=alt.Scale(scheme="viridis")),
                tooltip=["subject_id:N", "date:T", f"{x_name}:Q", f"{y_name}:Q"],
            )
        )
        _smoothing_label = "raw daily" if smoothing_days == 1 else f"{smoothing_days}-day mean"
        embedding_chart = (
            (trajectory_line + trajectory_points)
            .properties(
                title=f"{model_pick.selected_key} trajectory · {_smoothing_label}", height=340
            )
            .interactive()
        )
        embedding_output = mo.ui.altair_chart(embedding_chart)
    else:
        embedding_output = mo.md("No embedding coordinates are available for this selection.")
    embedding_output
    return


@app.cell
def _(alt, mo, smoothing_days, subject_view):
    axes_ready = (
        subject_view.height
        and "activity_score" in subject_view.columns
        and "recovery_score" in subject_view.columns
    )
    if axes_ready:
        axis_data = subject_view.drop_nulls(["activity_score", "recovery_score"])
        axis_line = (
            alt.Chart(axis_data)
            .mark_line(opacity=0.35)
            .encode(
                x=alt.X("activity_score:Q", title="Activity (relative to self)"),
                y=alt.Y("recovery_score:Q", title="Recovery (relative to self)"),
                order="date:T",
            )
        )
        axis_points = (
            alt.Chart(axis_data)
            .mark_circle(size=45)
            .encode(
                x="activity_score:Q",
                y="recovery_score:Q",
                color=alt.Color("date:T", title="Date", scale=alt.Scale(scheme="viridis")),
                tooltip=[
                    "date:T",
                    alt.Tooltip("activity_score:Q", format=".2f"),
                    alt.Tooltip("recovery_score:Q", format=".2f"),
                    alt.Tooltip("activity_coverage:Q", format=".0%"),
                    alt.Tooltip("recovery_coverage:Q", format=".0%"),
                ],
            )
        )
        _smoothing_label = "raw daily" if smoothing_days == 1 else f"{smoothing_days}-day mean"
        axes_output = mo.ui.altair_chart(
            (axis_line + axis_points)
            .properties(title=f"Interpretable trajectory · {_smoothing_label}", height=340)
            .interactive()
        )
    else:
        axes_output = mo.md("No activity/recovery scores are available for this selection.")
    axes_output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Model evaluation
    """)
    return


@app.cell
def _(alt, mo, pl, report):
    variance = report.get("pca", {}).get("explained_variance_ratio", [])
    variance_table = pl.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(len(variance))],
            "explained": variance,
            "cumulative": [sum(variance[: index + 1]) for index in range(len(variance))],
        }
    )
    if variance_table.height:
        bars = (
            alt.Chart(variance_table)
            .mark_bar()
            .encode(
                x=alt.X("component:N", title=None, sort=None),
                y=alt.Y("explained:Q", title="Explained variance", axis=alt.Axis(format="%")),
                tooltip=["component:N", alt.Tooltip("explained:Q", format=".1%")],
            )
        )
        cumulative = (
            alt.Chart(variance_table)
            .mark_line(point=True)
            .encode(
                x=alt.X("component:N", sort=None),
                y=alt.Y("cumulative:Q", title="Cumulative", axis=alt.Axis(format="%")),
                tooltip=["component:N", alt.Tooltip("cumulative:Q", format=".1%")],
            )
        )
        variance_output = mo.ui.altair_chart(
            alt.layer(bars, cumulative).resolve_scale(y="independent").properties(
                title="PCA variance captured", height=260
            )
        )
    else:
        variance_output = mo.md("No PCA evaluation is available.")
    variance_output
    return


@app.cell
def _(alt, mo, pl, report):
    error_rows = []
    for model_name in ("pca", "autoencoder"):
        by_feature = report.get(model_name, {}).get("rmse_by_feature", {})
        error_rows.extend(
            {"model": model_name.upper(), "feature": feature, "rmse": value}
            for feature, value in by_feature.items()
            if value is not None
        )
    errors = pl.DataFrame(error_rows) if error_rows else pl.DataFrame(
        schema={"model": pl.String, "feature": pl.String, "rmse": pl.Float64}
    )
    if errors.height:
        error_chart = (
            alt.Chart(errors)
            .mark_bar()
            .encode(
                x=alt.X("rmse:Q", title="Held-out standardized RMSE"),
                y=alt.Y("feature:N", title=None, sort="-x"),
                color=alt.Color("model:N", title=None),
                yOffset="model:N",
                tooltip=["model:N", "feature:N", alt.Tooltip("rmse:Q", format=".3f")],
            )
            .properties(title="Reconstruction error by feature", height=420)
        )
        error_output = mo.ui.altair_chart(error_chart)
    else:
        error_output = mo.md("No reconstruction errors are available.")
    error_output
    return


@app.cell
def _(alt, mo, pl, report):
    loading_rows = [
        {"component": component, "feature": feature, "loading": loading}
        for component, values in report.get("pca", {}).get("loadings", {}).items()
        for feature, loading in values.items()
    ]
    loadings = pl.DataFrame(loading_rows) if loading_rows else pl.DataFrame(
        schema={"component": pl.String, "feature": pl.String, "loading": pl.Float64}
    )
    if loadings.height:
        loading_chart = (
            alt.Chart(loadings)
            .mark_rect()
            .encode(
                x=alt.X("component:N", title=None, sort=None),
                y=alt.Y("feature:N", title=None),
                color=alt.Color(
                    "loading:Q",
                    title="Loading",
                    scale=alt.Scale(scheme="redblue", domainMid=0),
                ),
                tooltip=["component:N", "feature:N", alt.Tooltip("loading:Q", format=".3f")],
            )
            .properties(title="PCA loadings", height=420)
        )
        loading_output = mo.ui.altair_chart(loading_chart)
    else:
        loading_output = mo.md(
            "PCA loadings are not in this report. Rerun `uv run health-model` after updating."
        )
    loading_output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data coverage
    """)
    return


@app.cell
def _(alt, mo, pl, subject_view):
    core = [
        name
        for name in ("steps", "sleep_hours", "hr_mean_bpm")
        if name in subject_view.columns
    ]
    if subject_view.height and core:
        coverage = (
            subject_view.select("date", *core)
            .unpivot(index="date", on=core, variable_name="feature", value_name="value")
            .with_columns(pl.col("value").is_not_null().alias("observed"))
        )
        coverage_chart = (
            alt.Chart(coverage)
            .mark_rect()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("feature:N", title=None),
                color=alt.Color(
                    "observed:N",
                    title="Observed",
                    scale=alt.Scale(domain=[False, True], scheme="greys"),
                ),
                tooltip=["date:T", "feature:N", "observed:N"],
            )
            .properties(title="Core-feature availability", height=170)
        )
        coverage_output = mo.ui.altair_chart(coverage_chart)
    else:
        coverage_output = mo.md("No feature coverage is available for this selection.")
    coverage_output
    return


@app.cell
def _(mo, pl, report):
    windows = pl.DataFrame(report.get("windows", []))
    mo.ui.table(windows, selection=None) if windows.height else mo.md("No selected windows available.")
    return


if __name__ == "__main__":
    app.run()
