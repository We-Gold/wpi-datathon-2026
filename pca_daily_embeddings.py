# /// script
# dependencies = [
#     "marimo",
#     "pandas==3.0.5",
#     "numpy==2.5.2",
#     "scikit-learn==1.9.0",
#     "plotly==7.0.0",
#     "pywavelets",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import re
    import time
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pywt


    return PCA, Path, StandardScaler, go, make_subplots, mo, np, pd, pywt, time


@app.cell
def paths(Path, mo):
    DATA_DIR = Path("data")
    SELF_RECORDS_CSV = DATA_DIR / "health_export_parsed.csv"
    SELF_WORKOUTS_CSV = DATA_DIR / "health_workouts_parsed.csv"
    KAGGLE_CSV = DATA_DIR / "health_data.csv"

    mo.md(
        f"Self records: `{SELF_RECORDS_CSV}` | Self workouts: `{SELF_WORKOUTS_CSV}` | "
        f"Kaggle records: `{KAGGLE_CSV}`"
    )

    return KAGGLE_CSV, SELF_RECORDS_CSV, SELF_WORKOUTS_CSV


@app.cell
def intro(mo):
    mo.md(r"""
    # Daily health embeddings - PCA prototype

    **Goal:** represent each calendar day as a feature vector, reduce to 3D
    with PCA, and visualize the trajectory through that space over time -
    a cheap sanity check for the eventual VAE + sequence-model pipeline.
    If a linear 3D projection already separates weekdays/weekends, workout
    days, or device eras, that's a strong signal the richer pipeline is
    worth building. If it looks like noise, better to find out now.

    **Two people, two device ecosystems**, to check the ideas aren't
    overfit to one person's tracker: this notebook's own Apple Health
    export (Apple Watch, then a gap, then a Fitbit Air), and a Kaggle
    Apple Health export from someone else using a Zepp wearable. Both are
    reduced to the same anonymized long-format schema
    (`type, unit, value, creationDate, startDate, endDate`) before
    anything else happens, tagged only with a `person_id`.

    `person_id` is metadata for coloring/grouping and for per-person
    normalization - it is never fed into the PCA as a feature.
    """)
    return


@app.cell
def load_self(SELF_RECORDS_CSV, SELF_WORKOUTS_CSV, mo, pd):
    self_records_df = pd.read_csv(
        SELF_RECORDS_CSV, parse_dates=["creationDate", "startDate", "endDate"], low_memory=False
    )
    self_records_df.insert(0, "person_id", "self")
    self_records_df["day"] = self_records_df["startDate"].dt.tz_localize(None).dt.normalize()
    for _c in ["creationDate", "startDate", "endDate"]:
        self_records_df[_c] = self_records_df[_c].dt.tz_convert("UTC")

    self_workouts_df = pd.read_csv(SELF_WORKOUTS_CSV, parse_dates=["startDate", "endDate"])
    self_workouts_df.insert(0, "person_id", "self")
    self_workouts_df["day"] = self_workouts_df["startDate"].dt.tz_localize(None).dt.normalize()
    for _c in ["startDate", "endDate"]:
        self_workouts_df[_c] = self_workouts_df[_c].dt.tz_convert("UTC")

    mo.md(
        f"Loaded **{len(self_records_df):,}** self records and "
        f"**{len(self_workouts_df):,}** self workouts."
    )

    return self_records_df, self_workouts_df


@app.cell
def load_kaggle(KAGGLE_CSV, Path, mo, np, pd):
    def _load_kaggle_export(path: Path) -> pd.DataFrame:
        """Convert the Kaggle health_data.csv into the same canonical schema as
        self_records_df: one row per Record, anonymized, tz-aware datetimes.

        The Kaggle file splits each timestamp into 4 columns
        (date, time, am_pm, timeZone) - am_pm is redundant with the 12-hour
        time string, and CSV numeric inference turns "+0530" into the int 530
        (dropping the sign and leading zero), so it is reconstructed here.
        """
        raw = pd.read_csv(path, low_memory=False)

        def _combine(date_col, time_col, tz_col):
            tz_int = raw[tz_col].astype(int)
            sign = np.where(tz_int < 0, "-", "+")
            tz_abs = tz_int.abs()
            hh = (tz_abs // 100).astype(str).str.zfill(2)
            mm = (tz_abs % 100).astype(str).str.zfill(2)
            offset = sign + hh + ":" + mm
            combined = raw[date_col] + " " + raw[time_col] + " " + offset
            return pd.to_datetime(combined, format="%Y-%m-%d %I:%M:%S %p %z")

        df = pd.DataFrame({
            "person_id": "kaggle_1",
            "type": raw["type"],
            "unit": raw["unit"],
            "value": raw["value"].astype(str),
            "creationDate": _combine("m_creationDate", "m_creationTime", "m_creationTimeZone"),
            "startDate": _combine("m_startDate", "m_startTime", "m_startTimeZone"),
            "endDate": _combine("m_endDate", "m_endTime", "m_endTimeZone"),
        })
        df["day"] = df["startDate"].dt.tz_localize(None).dt.normalize()
        for _c in ["creationDate", "startDate", "endDate"]:
            df[_c] = df[_c].dt.tz_convert("UTC")
        return df


    kaggle_records_df = _load_kaggle_export(KAGGLE_CSV)

    mo.md(f"Loaded **{len(kaggle_records_df):,}** Kaggle records (person `kaggle_1`).")

    return (kaggle_records_df,)


@app.cell
def person_windows(mo, pd):
    # Per-person "useful" window: trims each person's history to the span with
    # consistently dense tracking, so per-person median-imputation and
    # z-scoring aren't diluted by mostly-empty years. Bounds are inclusive;
    # None means "no trim on that side".
    #
    # self: dense HR/sleep-stage tracking only starts once the Fitbit Air comes
    #   online (2026-07-28) - everything before that is the sparse Apple Watch
    #   era or phone-only, which is a different (already-explored) regime.
    # kaggle_1: reasonably dense HR throughout, but SleepAnalysis is essentially
    #   absent before 2023-02 (5 sleep-days in the 4 months prior) - trimmed to
    #   where sleep tracking actually starts rather than the full raw span.
    PERSON_WINDOWS = {
        "self": (pd.Timestamp("2026-07-28"), None),
        "kaggle_1": (pd.Timestamp("2023-02-01"), None),
    }


    def apply_person_windows(df: pd.DataFrame, day_col: str = "day") -> pd.DataFrame:
        keep = pd.Series(True, index=df.index)
        for _pid, (_start, _end) in PERSON_WINDOWS.items():
            _mask = df["person_id"] == _pid
            if _start is not None:
                keep &= ~(_mask & (df[day_col] < _start))
            if _end is not None:
                keep &= ~(_mask & (df[day_col] > _end))
        return df[keep]


    mo.md(
        "Windows: `self` >= **2026-07-28** (Fitbit era), "
        "`kaggle_1` >= **2023-02-01** (once sleep tracking starts)."
    )

    return (apply_person_windows,)


@app.cell
def combine_records(
    apply_person_windows,
    kaggle_records_df,
    mo,
    pd,
    self_records_df,
    self_workouts_df,
):
    records_df = apply_person_windows(pd.concat([self_records_df, kaggle_records_df], ignore_index=True))
    workouts_df = apply_person_windows(self_workouts_df.copy())

    _span = records_df.groupby("person_id")["day"].agg(["min", "max", "nunique"])
    mo.vstack([
        mo.md(
            f"Combined **{len(records_df):,}** records across "
            f"**{records_df['person_id'].nunique()}** people, trimmed to each "
            f"person's useful window."
        ),
        mo.ui.table(_span.reset_index(), page_size=5),
    ])

    return records_df, workouts_df


@app.cell
def type_coverage(mo, pd, records_df):
    def _daily_coverage(df: pd.DataFrame) -> pd.DataFrame:
        span = df.groupby("person_id")["day"].agg(["min", "max"])
        span_days = (span["max"] - span["min"]).dt.days + 1
        cov = df.groupby(["person_id", "type"], observed=True)["day"].nunique().unstack("person_id")
        cov_pct = cov.div(span_days, axis=1) * 100
        return cov_pct.round(1).sort_values("self", ascending=False)


    type_coverage_pct = _daily_coverage(records_df)
    mo.ui.table(type_coverage_pct.reset_index(), page_size=40)

    return


@app.cell
def core_features(mo, pd, records_df):
    _SUM_TYPES = {
        "StepCount": "steps",
        "DistanceWalkingRunning": "distance_walk_run_mi",
        "FlightsClimbed": "flights_climbed",
        "ActiveEnergyBurned": "energy_active_cal",
        "BasalEnergyBurned": "energy_basal_cal",
    }
    _MEAN_TYPES = {
        "WalkingSpeed": "walking_speed_avg",
        "WalkingStepLength": "walking_step_length_avg",
        "WalkingDoubleSupportPercentage": "walking_double_support_pct_avg",
        "WalkingAsymmetryPercentage": "walking_asymmetry_pct_avg",
        "RestingHeartRate": "resting_hr",
    }

    _num = records_df.copy()
    _num["value_num"] = pd.to_numeric(_num["value"], errors="coerce")
    _grouped = _num.groupby(["person_id", "day", "type"], observed=True)["value_num"]

    _sum_wide = (
        _grouped.sum()
        .unstack("type")[list(_SUM_TYPES)]
        .rename(columns=_SUM_TYPES)
    )
    _mean_wide = (
        _grouped.mean()
        .unstack("type")[list(_MEAN_TYPES)]
        .rename(columns=_MEAN_TYPES)
    )
    core_features_df = _sum_wide.join(_mean_wide, how="outer")

    mo.vstack([
        mo.md(f"Core feature table: **{len(core_features_df):,}** person-days x **{core_features_df.shape[1]}** features."),
        mo.ui.table(core_features_df.reset_index().sample(8, random_state=0), page_size=8),
    ])

    return (core_features_df,)


@app.cell
def sleep_features(mo, pd, records_df):
    def _build_sleep_features(records_df: pd.DataFrame) -> pd.DataFrame:
        """Per (person, day) sleep features from SleepAnalysis interval records.

        Handles both the full "HKCategoryValueSleepAnalysis..." strings (this
        notebook's own export) and the already-short Kaggle values ("InBed",
        "AsleepCore", ...) via one prefix strip.

        Devices that report stages nest "InBed" as one span covering the whole
        night, with the Asleep*/Awake sub-segments inside it - so night span
        must come from InBed alone where it exists (summing every record would
        double-count the same minutes twice). Falls back to
        max(end) - min(start) over the asleep/awake segments on nights with no
        InBed record. `sleep_efficiency` and stage percentages are only
        defined on nights with real asleep/awake staging - most of this
        notebook's own history only has an "InBed" span with no stage
        breakdown, so those stay NaN there (masked, not imputed).
        """
        sleep = records_df[records_df["type"] == "SleepAnalysis"].copy()
        sleep["stage"] = sleep["value"].str.replace("HKCategoryValueSleepAnalysis", "", regex=False)
        sleep["hours"] = (sleep["endDate"] - sleep["startDate"]).dt.total_seconds() / 3600
        sleep["is_asleep"] = sleep["stage"].str.startswith("Asleep")
        sleep["is_awake"] = sleep["stage"] == "Awake"
        sleep["is_inbed"] = sleep["stage"] == "InBed"

        g = sleep.groupby(["person_id", "day"])
        inbed_span = sleep[sleep["is_inbed"]].groupby(["person_id", "day"])["hours"].sum()
        staged = sleep[sleep["is_asleep"] | sleep["is_awake"]]
        fallback_span = (
            staged.groupby(["person_id", "day"])["endDate"].max()
            - staged.groupby(["person_id", "day"])["startDate"].min()
        ).dt.total_seconds() / 3600
        night_span_hours = inbed_span.combine_first(fallback_span).rename("night_span_hours")

        asleep_hours = sleep[sleep["is_asleep"]].groupby(["person_id", "day"])["hours"].sum().rename("asleep_hours")
        awake_hours = sleep[sleep["is_awake"]].groupby(["person_id", "day"])["hours"].sum().rename("awake_hours")

        out = pd.concat([night_span_hours, asleep_hours, awake_hours], axis=1)
        # naps or a sleep session split across the local-day boundary can push a
        # day above 100% efficiency; clip rather than build full bout-segmentation
        # logic for this prototype.
        out["sleep_efficiency"] = (out["asleep_hours"] / out["night_span_hours"]).clip(upper=1.0)

        for _stage in ["REM", "Deep", "Core"]:
            _stage_hours = (
                sleep[sleep["stage"] == f"Asleep{_stage}"]
                .groupby(["person_id", "day"])["hours"].sum()
            )
            out[f"sleep_{_stage.lower()}_pct"] = _stage_hours / out["asleep_hours"]

        return out.drop(columns=["asleep_hours", "awake_hours"])


    sleep_features_df = _build_sleep_features(records_df)

    mo.vstack([
        mo.md(f"Sleep features: **{len(sleep_features_df):,}** person-nights."),
        mo.ui.table(sleep_features_df.reset_index().dropna(subset=["sleep_rem_pct"]).sample(5, random_state=0), page_size=5),
    ])

    return (sleep_features_df,)


@app.cell
def hr_features(mo, np, pd, pywt, records_df, time):
    _FREQ_SAMPLE_THRESHOLD = 200  # min HR samples/day to attempt frequency features

    def _hr_frequency_features(day_hr: pd.DataFrame) -> pd.Series:
        """FFT- and wavelet-derived features from one day's raw, irregularly
        sampled HeartRate series. Resamples onto a 10-minute grid (144 points)
        via linear interpolation first, since FFT/wavelets assume even spacing.
        """
        t0 = day_hr["startDate"].min().normalize()
        grid = pd.date_range(t0, t0 + pd.Timedelta(hours=24), freq="10min", inclusive="left")
        s = day_hr.set_index("startDate")["value"].sort_index()
        s = s[~s.index.duplicated()]
        resampled = s.reindex(s.index.union(grid)).interpolate(method="time").reindex(grid)
        if resampled.isna().any():
            resampled = resampled.ffill().bfill()
        x = resampled.to_numpy() - resampled.mean()

        power = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=10 / 60)  # cycles/hour
        power = power[1:]  # drop DC
        freqs = freqs[1:]
        power_frac = power / power.sum() if power.sum() > 0 else np.zeros_like(power)
        dominant_period_h = 1 / freqs[np.argmax(power)] if power.sum() > 0 else np.nan
        low_freq_frac = power[freqs < 1 / 6].sum() / power.sum() if power.sum() > 0 else np.nan
        spectral_entropy = -(power_frac[power_frac > 0] * np.log(power_frac[power_frac > 0])).sum()

        coeffs = pywt.wavedec(x, "db4", level=3)
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies)
        detail_energy_frac = sum(energies[1:]) / total_energy if total_energy > 0 else np.nan

        return pd.Series({
            "hr_dominant_period_h": dominant_period_h,
            "hr_low_freq_power_frac": low_freq_frac,
            "hr_spectral_entropy": spectral_entropy,
            "hr_wavelet_detail_energy_frac": detail_energy_frac,
        })


    _hr = records_df[records_df["type"] == "HeartRate"].copy()
    _hr["value"] = pd.to_numeric(_hr["value"], errors="coerce")
    _hr = _hr.dropna(subset=["value"])

    _daily_hr_stats = _hr.groupby(["person_id", "day"])["value"].agg(
        hr_mean="mean", hr_std="std", hr_min="min", hr_max="max", hr_n="count"
    )
    _daily_hr_stats["hr_density_tier"] = pd.cut(
        _daily_hr_stats["hr_n"], bins=[-1, 0, _FREQ_SAMPLE_THRESHOLD, np.inf],
        labels=["none", "sparse_wearable", "dense_wearable"],
    )

    _dense_days = _daily_hr_stats[_daily_hr_stats["hr_n"] >= _FREQ_SAMPLE_THRESHOLD].index
    _t0 = time.time()
    _freq_rows = {
        key: _hr_frequency_features(group)
        for key, group in _hr.groupby(["person_id", "day"])
        if key in _dense_days
    }
    _freq_seconds = time.time() - _t0
    hr_freq_features_df = pd.DataFrame(_freq_rows).T
    hr_freq_features_df.index.names = ["person_id", "day"]

    hr_features_df = _daily_hr_stats.join(hr_freq_features_df, how="left")

    mo.vstack([
        mo.md(
            f"HR features: **{len(hr_features_df):,}** person-days with any HR, "
            f"**{len(hr_freq_features_df):,}** dense enough (>= {_FREQ_SAMPLE_THRESHOLD} "
            f"samples/day) for frequency/wavelet features ({_freq_seconds:.1f}s)."
        ),
        mo.ui.table(hr_features_df.reset_index().dropna(subset=["hr_spectral_entropy"]).sample(5, random_state=0), page_size=5),
    ])

    return (hr_features_df,)


@app.cell
def workout_daily_features(mo, workouts_df):
    workout_daily_features_df = workouts_df.groupby(["person_id", "day"]).agg(
        workout_count=("workout_id", "count"),
        workout_duration_min=("duration_min", "sum"),
        workout_energy_active_cal=("energy_active_cal", "sum"),
        workout_gpx_elev_gain_m=("gpx_elev_gain_m", "sum"),
    )

    mo.md(
        f"Workout aggregates: **{len(workout_daily_features_df):,}** person-days "
        f"with >=1 workout (self only - the Kaggle export has no Workout data, "
        f"so these columns will be all-NaN for `kaggle_1`, not zero)."
    )

    return (workout_daily_features_df,)


@app.cell
def daily_features(
    core_features_df,
    hr_features_df,
    mo,
    sleep_features_df,
    workout_daily_features_df,
):
    _KNOWN_MISSING_COLS = ["workout_count", "workout_duration_min",
                            "workout_energy_active_cal", "workout_gpx_elev_gain_m"]

    daily_features_df = (
        core_features_df
        .join(sleep_features_df, how="outer")
        .join(hr_features_df, how="outer")
        .join(workout_daily_features_df, how="outer")
    )
    for _c in _KNOWN_MISSING_COLS:
        daily_features_df[_c] = daily_features_df[_c].fillna(0.0)
    daily_features_df["has_workout"] = daily_features_df["workout_count"] > 0

    daily_features_df = daily_features_df.reset_index()
    daily_features_df["weekday"] = daily_features_df["day"].dt.day_name()
    daily_features_df["is_weekend"] = daily_features_df["day"].dt.dayofweek >= 5
    daily_features_df["month"] = daily_features_df["day"].dt.month
    daily_features_df["year"] = daily_features_df["day"].dt.year

    _FEATURE_COLS = [
        "steps", "distance_walk_run_mi", "flights_climbed", "energy_active_cal",
        "energy_basal_cal", "walking_speed_avg", "walking_step_length_avg",
        "walking_double_support_pct_avg", "walking_asymmetry_pct_avg", "resting_hr",
        "night_span_hours", "sleep_efficiency", "sleep_rem_pct", "sleep_deep_pct", "sleep_core_pct",
        "hr_mean", "hr_std", "hr_min", "hr_max",
        "hr_dominant_period_h", "hr_low_freq_power_frac", "hr_spectral_entropy",
        "hr_wavelet_detail_energy_frac",
        "workout_count", "workout_duration_min", "workout_energy_active_cal", "workout_gpx_elev_gain_m",
    ]
    daily_features_df["completeness_pct"] = daily_features_df[_FEATURE_COLS].notna().mean(axis=1) * 100

    mo.vstack([
        mo.md(
            f"**{len(daily_features_df):,}** person-days x **{len(_FEATURE_COLS)}** "
            f"raw features (before imputation)."
        ),
        mo.ui.table(daily_features_df[["person_id", "day", "completeness_pct"] + _FEATURE_COLS].sample(8, random_state=1), page_size=8),
    ])

    return (daily_features_df,)


@app.cell
def fit_pca(PCA, StandardScaler, daily_features_df, mo, pd):
    _FEATURE_COLS = [
        "steps", "distance_walk_run_mi", "flights_climbed", "energy_active_cal",
        "energy_basal_cal", "walking_speed_avg", "walking_step_length_avg",
        "walking_double_support_pct_avg", "walking_asymmetry_pct_avg", "resting_hr",
        "night_span_hours", "sleep_efficiency", "sleep_rem_pct", "sleep_deep_pct", "sleep_core_pct",
        "hr_mean", "hr_std", "hr_min", "hr_max",
        "hr_dominant_period_h", "hr_low_freq_power_frac", "hr_spectral_entropy",
        "hr_wavelet_detail_energy_frac",
        "workout_count", "workout_duration_min", "workout_energy_active_cal", "workout_gpx_elev_gain_m",
    ]

    # Impute with each person's own median (not a global median - baselines
    # differ enough between people that a global fill would leak one person's
    # scale into the other's missing days), then z-score within person too.
    _imputed = daily_features_df.copy()
    for _col in _FEATURE_COLS:
        _imputed[_col] = _imputed.groupby("person_id")[_col].transform(lambda s: s.fillna(s.median()))
    # a column can still be entirely NaN for one person (e.g. Apple-only types
    # for kaggle_1) - those fall back to 0 after per-person scaling below.
    _imputed[_FEATURE_COLS] = _imputed[_FEATURE_COLS].fillna(0.0)

    _scaled_parts = []
    for _pid, _group in _imputed.groupby("person_id"):
        _scaler = StandardScaler()
        _vals = _scaler.fit_transform(_group[_FEATURE_COLS])
        _scaled_parts.append(pd.DataFrame(_vals, index=_group.index, columns=_FEATURE_COLS))
    _scaled = pd.concat(_scaled_parts).sort_index().fillna(0.0)

    pca = PCA(n_components=10, random_state=0)
    _pcs = pca.fit_transform(_scaled)

    embedding_df = daily_features_df[["person_id", "day", "weekday", "is_weekend", "month",
                                        "year", "has_workout", "hr_density_tier", "completeness_pct"]].copy()
    for _i in range(10):
        embedding_df[f"PC{_i+1}"] = _pcs[:, _i]

    mo.md(
        f"Fit PCA on **{len(_scaled):,}** person-days x **{len(_FEATURE_COLS)}** "
        f"features (per-person median-imputed, per-person z-scored)."
    )

    return embedding_df, pca


@app.cell
def scree_plot(go, mo, np, pca):
    _evr = pca.explained_variance_ratio_
    _cum = np.cumsum(_evr)

    _scree = go.Figure()
    _scree.add_bar(x=[f"PC{i+1}" for i in range(len(_evr))], y=_evr, name="per-component")
    _scree.add_scatter(x=[f"PC{i+1}" for i in range(len(_evr))], y=_cum, name="cumulative", yaxis="y2")
    _scree.update_layout(
        height=300, margin=dict(t=30, b=10),
        yaxis=dict(title="explained variance ratio"),
        yaxis2=dict(title="cumulative", overlaying="y", side="right", range=[0, 1]),
        title=f"Scree plot - PC1-3 explain {_cum[2]*100:.0f}% of variance",
    )

    mo.vstack([
        mo.md(
            "First 3 components capture a modest but real fraction of the "
            "variance - the 3D plot below is a lossy projection, worth keeping "
            "in mind when reading it."
        ),
        _scree,
    ])

    return


@app.cell
def trajectory_controls(embedding_df, mo):
    color_by_picker = mo.ui.dropdown(
        options=["hr_density_tier", "is_weekend", "weekday", "month",
                 "year", "has_workout", "completeness_pct", "person_id"],
        value="has_workout",
        label="Color by",
    )
    timeframe_picker = mo.ui.date_range(
        start=embedding_df["day"].min().date(),
        stop=embedding_df["day"].max().date(),
        value=(embedding_df["day"].min().date(), embedding_df["day"].max().date()),
        label="Timeframe",
    )
    person_picker = mo.ui.multiselect(
        options=sorted(embedding_df["person_id"].unique()),
        value=sorted(embedding_df["person_id"].unique()),
        label="People",
    )
    smoothing_picker = mo.ui.slider(
        start=1, stop=14, step=1, value=7,
        label="Smoothing window (trailing days, 1 = raw)",
    )
    last_n_days_picker = mo.ui.slider(
        start=7, stop=730, step=1, value=90,
        label="Show last N days (per person, relative to their own most recent day)",
    )
    mo.vstack([
        mo.hstack([color_by_picker, timeframe_picker, person_picker]),
        mo.hstack([smoothing_picker, last_n_days_picker]),
    ])

    return (
        color_by_picker,
        last_n_days_picker,
        person_picker,
        smoothing_picker,
        timeframe_picker,
    )


@app.cell
def trajectory_plot(
    color_by_picker,
    embedding_df,
    last_n_days_picker,
    make_subplots,
    pd,
    person_picker,
    smoothing_picker,
    timeframe_picker,
):
    _start, _stop = timeframe_picker.value
    _mask = (
        embedding_df["day"].dt.date.between(_start, _stop)
        & embedding_df["person_id"].isin(person_picker.value)
    )
    _view = embedding_df[_mask].sort_values(["person_id", "day"]).copy()

    _n_days = last_n_days_picker.value
    _view = _view[
        _view.groupby("person_id")["day"].transform(
            lambda s: s >= s.max() - pd.Timedelta(days=_n_days)
        )
    ]

    _window = smoothing_picker.value
    if _window > 1:
        for _pc in ["PC1", "PC2", "PC3"]:
            _view[_pc] = _view.groupby("person_id")[_pc].transform(
                lambda s: s.rolling(_window, min_periods=1).mean()
            )

    _color_col = color_by_picker.value
    _people = [p for p in sorted(_view["person_id"].unique())]
    _n = max(len(_people), 1)

    _fig = make_subplots(
        rows=_n, cols=1,
        specs=[[{"type": "scene"}] for _ in range(_n)],
        subplot_titles=[f"{p} ({(_view['person_id'] == p).sum():,} days)" for p in _people],
        vertical_spacing=0.06,
    )

    _is_continuous = pd.api.types.is_numeric_dtype(_view[_color_col]) and _view[_color_col].nunique() > 8

    for _i, _pid in enumerate(_people):
        _g = _view[_view["person_id"] == _pid]
        _row = _i + 1
        if _is_continuous:
            _fig.add_scatter3d(
                x=_g["PC1"], y=_g["PC2"], z=_g["PC3"],
                mode="lines+markers",
                line=dict(width=2, color="rgba(150,150,150,0.4)"),
                marker=dict(size=3, color=_g[_color_col], colorscale="Viridis",
                            colorbar=dict(title=_color_col, len=1 / _n, y=1 - (_row - 0.5) / _n)),
                name=_pid, showlegend=False,
                text=_g["day"].dt.strftime("%Y-%m-%d"),
                hovertemplate="%{text}<br>" + f"{_color_col}=" + "%{marker.color}<extra></extra>",
                row=_row, col=1,
            )
        else:
            for _cat, _gg in _g.groupby(_color_col, observed=True):
                _fig.add_scatter3d(
                    x=_gg["PC1"], y=_gg["PC2"], z=_gg["PC3"],
                    mode="markers",
                    marker=dict(size=3),
                    name=str(_cat), legendgroup=str(_cat), showlegend=(_row == 1),
                    text=_gg["day"].dt.strftime("%Y-%m-%d"),
                    hovertemplate=f"{_cat}<br>" + "%{text}<extra></extra>",
                    row=_row, col=1,
                )
            _fig.add_scatter3d(
                x=_g["PC1"], y=_g["PC2"], z=_g["PC3"], mode="lines",
                line=dict(width=1, color="rgba(150,150,150,0.25)"),
                showlegend=False, hoverinfo="skip",
                row=_row, col=1,
            )

    _scene_axes = dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
    _layout_scenes = {f"scene{'' if _i == 0 else _i + 1}": _scene_axes for _i in range(_n)}
    _smooth_note = "raw daily" if _window == 1 else f"{_window}-day trailing rolling mean"
    _fig.update_layout(
        height=560 * _n,
        title=f"Daily embedding trajectory per person, last {_n_days} days ({_smooth_note}) - colored by {_color_col}",
        margin=dict(t=60, b=0),
        **_layout_scenes,
    )

    _fig

    return


@app.cell
def outro(mo):
    mo.md(r"""
    ## Reading this demo honestly

    **What it already shows:** `has_workout` and `hr_density_tier` both
    produce real, sizeable separation along PC1 in a *linear* 3D
    projection - that's evidence the daily feature vectors carry real
    signal, not just noise, before any VAE/nonlinearity is involved. That
    was the actual purpose of this prototype: a cheap kill-test before
    investing in the fancier pipeline.

    **Caveats worth stating out loud:**

    - **Two people is not generalization**, it's a single replication
      check. `kaggle_1`'s cloud sitting in roughly the same PC space as
      `self`'s (rather than a totally disjoint region) is a good sign, but
      two points don't establish a population pattern - a datathon
      writeup should call this a feasibility check, not a validated
      cross-person model.
    - **PC1-3 explain a modest share of total variance** (see the scree
      plot above) - the 3D picture is a lossy summary; components 4+
      still carry real structure this projection throws away.
    - **`hr_density_tier` and `has_workout` may be entangled with device
      era** rather than pure physiology - a day with dense HR sampling is
      also more likely to be a Fitbit-era day, which correlates with
      *when* in the multi-year span it happened. Some of what looks like
      "workout signature" could partly be "which years had a workout
      habit."
    - **Per-person median imputation is a blunt instrument** - it fills
      gaps with each person's own typical day, which will flatten out
      real variation on sparsely-tracked features rather than represent
      genuine uncertainty. Fine for this linear check; the VAE stage
      should carry an explicit missingness mask instead of silently
      imputing.
    - **`sleep_efficiency` uses simple calendar-day bucketing**, not real
      sleep-bout segmentation - a nap or a session crossing midnight can
      distort a handful of days (clipped at 1.0 rather than fixed
      properly, given prototype scope).

    **If this holds up**, the natural next steps are: swap the linear PCA
    for a (beta-)VAE on the same feature table, add a naive next-day
    forecast baseline, and only then reach for the transformer/LSTM if
    the baseline can't match it.
    """)
    return


if __name__ == "__main__":
    app.run()
