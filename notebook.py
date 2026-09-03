# /// script
# dependencies = [
#     "altair==6.2.2",
#     "marimo",
#     "pandas==3.0.5",
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
    import altair as alt
    from collections import Counter
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import time
    import re


    return Counter, ET, Path, alt, mo, pd, re, time


@app.cell
def paths(Path, mo):
    DATA_DIR = Path("data/apple_health_export 2")
    EXPORT_XML = DATA_DIR / "export.xml"
    EXPORT_CDA = DATA_DIR / "export_cda.xml"
    mo.md(f"Export XML: `{EXPORT_XML}` ({EXPORT_XML.stat().st_size / 1e6:.0f} MB)")

    return DATA_DIR, EXPORT_XML


@app.cell
def intro_cell(mo):
    mo.md("""
    # Apple Health export analysis

    Streams `export.xml` with `xml.etree.ElementTree.iterparse` so the ~1GB
    file is never fully loaded into memory — each `<Record>` is read, tallied
    into `Counter`s keyed by `(day, type)`, `(day, type, source)`, and
    `type`, then cleared before moving on. Only top-level `<Record>` elements
    are counted; per the DTD, records nested inside `<Correlation>` also
    appear as top-level records, so counting only the top-level ones avoids
    double-counting.

    `export_cda.xml` (351MB, HL7 CDA format) encodes the same underlying
    records as `export.xml` in a much more verbose clinical-document schema —
    it isn't parsed separately here since it's a redundant representation of
    the same data, not additional data.
    """)
    return


@app.cell
def parse_export(Counter, ET, EXPORT_XML, Path, mo, time):
    def _parse_export(path: Path):
        """Stream-parse the Apple Health export.xml without loading it into memory.

        Only top-level <Record> elements are counted (the DTD notes that records
        nested inside <Correlation> also appear as top-level Records, so this
        avoids double counting). Returns Counters keyed for downstream aggregation.
        """
        by_day_type = Counter()
        by_type = Counter()
        by_day_type_source = Counter()
        by_source = Counter()
        n = 0

        context = ET.iterparse(str(path), events=("end",))
        for event, elem in context:
            if elem.tag == "Record":
                n += 1
                rtype = elem.get("type")
                source = elem.get("sourceName")
                day = elem.get("startDate", "")[:10]
                by_day_type[(day, rtype)] += 1
                by_type[rtype] += 1
                by_day_type_source[(day, rtype, source)] += 1
                by_source[source] += 1
                elem.clear()
            elif elem.tag in ("Correlation", "Workout", "ActivitySummary",
                              "ClinicalRecord", "Audiogram", "VisionPrescription"):
                elem.clear()

        return n, by_day_type, by_type, by_day_type_source, by_source


    _t0 = time.time()
    n_records, by_day_type, by_type, by_day_type_source, by_source = _parse_export(EXPORT_XML)
    parse_seconds = time.time() - _t0
    mo.md(f"Parsed **{n_records:,}** records in {parse_seconds:.1f}s")

    return by_day_type, by_day_type_source, by_type


@app.cell
def day_totals_cell(alt, by_day_type, mo, pd):
    # Total records per day (all types combined)
    day_totals = (
        pd.DataFrame(
            [(day, t, c) for (day, t), c in by_day_type.items()],
            columns=["date", "type", "count"],
        )
        .groupby("date", as_index=False)["count"].sum()
        .sort_values("date")
    )
    day_totals["date"] = pd.to_datetime(day_totals["date"])

    _min_d = day_totals["date"].min().date()
    _max_d = day_totals["date"].max().date()

    mo.vstack([
        mo.md(
            f"**{len(day_totals):,}** distinct days with data, "
            f"from {_min_d} to {_max_d}.\n\n"
            f"Records/day: min **{day_totals['count'].min()}**, "
            f"median **{int(day_totals['count'].median())}**, "
            f"max **{day_totals['count'].max():,}**, "
            f"mean **{day_totals['count'].mean():.0f}**."
        ),
        alt.Chart(day_totals).mark_line(size=1).encode(
            x="date:T", y="count:Q"
        ).properties(height=250, title="Records per day"),
    ])

    return


@app.cell
def type_totals_cell(alt, by_type, mo, pd):
    # Total records per type (data point counts by measurement type)
    type_totals = (
        pd.DataFrame(by_type.items(), columns=["type", "count"])
        .assign(type=lambda d: d["type"].str.replace(
            r"^HK(QuantityType|CategoryType|DataType)Identifier", "", regex=True
        ))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    mo.vstack([
        mo.md(f"**{len(type_totals)}** distinct record types."),
        mo.ui.table(type_totals, page_size=40),
        alt.Chart(type_totals.head(20)).mark_bar().encode(
            x=alt.X("count:Q"),
            y=alt.Y("type:N", sort="-x"),
        ).properties(height=400, title="Top 20 record types by count"),
    ])

    return (type_totals,)


@app.cell
def multi_source_cell(by_day_type_source, mo, pd):
    # Which sources contribute records for which types (multi-source detection)
    type_source_totals = (
        pd.DataFrame(
            [(t, s, c) for (day, t, s), c in by_day_type_source.items()],
            columns=["type", "source", "count"],
        )
        .groupby(["type", "source"], as_index=False)["count"].sum()
        .assign(type=lambda d: d["type"].str.replace(
            r"^HK(QuantityType|CategoryType|DataType)Identifier", "", regex=True
        ))
    )

    # types with more than one contributing source, ranked by how "split" they are
    _n_sources = type_source_totals.groupby("type")["source"].nunique().rename("n_sources")
    multi_source_types = (
        type_source_totals.merge(_n_sources, on="type")
        .query("n_sources > 1")
        .sort_values(["type", "count"], ascending=[True, False])
    )

    mo.vstack([
        mo.md(
            f"**{_n_sources[_n_sources > 1].shape[0]}** of {_n_sources.shape[0]} record types "
            "are populated by more than one source app/device."
        ),
        mo.ui.table(multi_source_types, page_size=40),
    ])

    return


@app.cell
def hr_source_timeline(alt, by_day_type_source, mo, pd):
    # Heart-rate source handoff over time: which device/app was reporting HR each month
    _hr = pd.DataFrame(
        [(day, s, c) for (day, t, s), c in by_day_type_source.items()
         if t == "HKQuantityTypeIdentifierHeartRate"],
        columns=["date", "source", "count"],
    )
    _hr["date"] = pd.to_datetime(_hr["date"])
    _hr["month"] = _hr["date"].values.astype("datetime64[M]")

    hr_monthly_by_source = (
        _hr.groupby(["month", "source"], as_index=False)["count"].sum()
    )

    mo.vstack([
        mo.md("### Heart rate: Apple Watch vs. Google Health (Fitbit) over time"),
        alt.Chart(hr_monthly_by_source).mark_area().encode(
            x="month:T",
            y=alt.Y("count:Q", stack="normalize", title="share of HR records"),
            color=alt.Color("source:N"),
        ).properties(height=250, title="Monthly share of HeartRate records by source"),
        alt.Chart(hr_monthly_by_source).mark_bar().encode(
            x="month:T",
            y=alt.Y("count:Q", title="records"),
            color=alt.Color("source:N"),
        ).properties(height=250, title="Monthly HeartRate record volume by source"),
    ])

    return


@app.cell
def type_picker_cell(mo, type_totals):
    type_picker = mo.ui.dropdown(
        options=sorted(type_totals["type"].tolist()),
        value="HeartRate" if "HeartRate" in type_totals["type"].values else type_totals["type"].iloc[0],
        label="Record type",
    )
    type_picker

    return (type_picker,)


@app.cell
def type_drilldown_cell(alt, by_day_type_source, by_type, mo, pd, type_picker):
    # map the friendly (stripped) name back to the raw HK identifier
    _candidates = [t for t in by_type if t.endswith(type_picker.value) and (
        t.startswith("HKQuantityTypeIdentifier") or t.startswith("HKCategoryTypeIdentifier") or t.startswith("HKDataTypeIdentifier")
    )]
    _raw_type = _candidates[0] if _candidates else None

    _rows = pd.DataFrame(
        [(day, s, c) for (day, t, s), c in by_day_type_source.items() if t == _raw_type],
        columns=["date", "source", "count"],
    )
    _rows["date"] = pd.to_datetime(_rows["date"])

    _by_source_total = _rows.groupby("source", as_index=False)["count"].sum().sort_values("count", ascending=False)

    mo.vstack([
        mo.md(f"### `{type_picker.value}` — {int(_rows['count'].sum()):,} data points across {_rows['date'].nunique():,} days"),
        mo.ui.table(_by_source_total, page_size=10),
        alt.Chart(_rows).mark_bar().encode(
            x="date:T",
            y=alt.Y("count:Q", title="records/day"),
            color="source:N",
        ).properties(height=280, title=f"Daily {type_picker.value} record counts by source"),
    ])

    return


@app.cell(hide_code=True)
def outro_cell(mo):
    mo.md(f"""
    ### Notes on multi-source data

    **19** record types are populated from more than one
    source app/device (see the table above) — that's the expected pattern
    for anything trackable by both an Apple Watch and a synced third-party
    app/device (Fitbit data arrives here as source `"Google Health"`, since
    Fitbit → Google Health → Apple Health is this export's sync path).

    Heart rate is the clearest case: **0** records from
    Weaver's Apple Watch vs. **42,222** from Google Health. The
    timeline chart above shows the handoff over time rather than a clean
    split — there are stretches where each source dominates, plus overlap
    where both report on the same days. Any downstream analysis that treats
    heart rate as one continuous series should either pick one source per
    time window or explicitly dedupe/reconcile overlapping timestamps.
    """)
    return


@app.cell(hide_code=True)
def workout_section_header(mo):
    mo.md("""
    ## GPX route alignment & per-workout health summaries

    `export.xml` nests each workout's GPX route directly inside its
    `<Workout>` element (`<WorkoutRoute><FileReference path="..."/>`), so no
    filename/timestamp matching is needed — the link is structural. Of
    719 total workouts, 339 have an attached route file; the rest (mostly
    recent Fitbit-sourced workouts synced via Google Health) have none.

    This section re-scans `export.xml` once more (streamed the same way) to
    pull full workout records (with nested `WorkoutStatistics` and the GPX
    link) plus every `HeartRate` record, since heart rate isn't included in
    a workout's own statistics and has to be matched by overlapping
    timestamp instead.
    """)
    return


@app.cell
def parse_workouts_hr(ET, EXPORT_XML, Path, mo, pd, time):
    def _parse_workouts_and_hr(path: Path):
        """One streaming pass: collect full Workout records (+ nested
        WorkoutStatistics and GPX route link) and every HeartRate record
        (needed to compute per-workout heart rate since it is not part of
        WorkoutStatistics)."""
        workouts = []
        hr_rows = []

        context = ET.iterparse(str(path), events=("end",))
        for event, elem in context:
            tag = elem.tag
            if tag == "Workout":
                stats = {}
                gpx_path = None
                for child in elem:
                    if child.tag == "WorkoutStatistics":
                        stype = child.get("type", "").replace(
                            "HKQuantityTypeIdentifier", ""
                        )
                        stats[stype] = {
                            "sum": child.get("sum"),
                            "average": child.get("average"),
                            "minimum": child.get("minimum"),
                            "maximum": child.get("maximum"),
                            "unit": child.get("unit"),
                        }
                    elif child.tag == "WorkoutRoute":
                        for gc in child:
                            if gc.tag == "FileReference":
                                gpx_path = gc.get("path")
                workouts.append({
                    "activity": (elem.get("workoutActivityType") or "")
                        .replace("HKWorkoutActivityType", ""),
                    "source": elem.get("sourceName"),
                    "device": elem.get("device"),
                    "startDate": elem.get("startDate"),
                    "endDate": elem.get("endDate"),
                    "duration_min": float(elem.get("duration") or "nan"),
                    "stats": stats,
                    "gpx_path": gpx_path,
                })
                elem.clear()
            elif tag == "Record":
                if elem.get("type") == "HKQuantityTypeIdentifierHeartRate":
                    hr_rows.append((
                        elem.get("startDate"),
                        elem.get("value"),
                        elem.get("sourceName"),
                    ))
                elem.clear()
            elif tag in ("Correlation", "ActivitySummary", "ClinicalRecord",
                          "Audiogram", "VisionPrescription"):
                elem.clear()

        return workouts, hr_rows


    _t0 = time.time()
    _workouts_raw, _hr_rows = _parse_workouts_and_hr(EXPORT_XML)
    workouts_vitals_seconds = time.time() - _t0

    workouts_df = pd.DataFrame(_workouts_raw)
    workouts_df["startDate"] = pd.to_datetime(workouts_df["startDate"])
    workouts_df["endDate"] = pd.to_datetime(workouts_df["endDate"])
    workouts_df["has_gpx"] = workouts_df["gpx_path"].notna()
    workouts_df = workouts_df.sort_values("startDate").reset_index(drop=True)

    hr_df = pd.DataFrame(_hr_rows, columns=["startDate", "value", "source"])
    hr_df["startDate"] = pd.to_datetime(hr_df["startDate"])
    hr_df["value"] = pd.to_numeric(hr_df["value"], errors="coerce")
    hr_df = hr_df.sort_values("startDate").reset_index(drop=True)

    mo.md(
        f"Parsed **{len(workouts_df):,}** workouts "
        f"(**{int(workouts_df['has_gpx'].sum())}** with a GPX route) and "
        f"**{len(hr_df):,}** heart-rate records in {workouts_vitals_seconds:.1f}s"
    )

    return hr_df, workouts_df


@app.cell
def gpx_hr_helpers(DATA_DIR, ET, hr_df, mo, pd):
    _GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}


    def load_gpx_points(gpx_path: str) -> pd.DataFrame:
        """Parse a workout-route GPX file into a lat/lon/ele/time DataFrame."""
        full_path = DATA_DIR / gpx_path.lstrip("/")
        tree = ET.parse(full_path)
        rows = []
        for pt in tree.getroot().iter("{http://www.topografix.com/GPX/1/1}trkpt"):
            ele = pt.find("g:ele", _GPX_NS)
            time_el = pt.find("g:time", _GPX_NS)
            rows.append({
                "lat": float(pt.get("lat")),
                "lon": float(pt.get("lon")),
                "ele_m": float(ele.text) if ele is not None else None,
                "time": time_el.text if time_el is not None else None,
            })
        df = pd.DataFrame(rows)
        if len(df):
            df["time"] = pd.to_datetime(df["time"])
        return df


    def haversine_miles(lat1, lon1, lat2, lon2):
        import numpy as _np
        r = 3958.8  # earth radius, miles
        p1, p2 = _np.radians(lat1), _np.radians(lat2)
        dphi = _np.radians(lat2 - lat1)
        dlambda = _np.radians(lon2 - lon1)
        a = _np.sin(dphi / 2) ** 2 + _np.cos(p1) * _np.cos(p2) * _np.sin(dlambda / 2) ** 2
        return 2 * r * _np.arcsin(_np.sqrt(a))


    def gpx_route_summary(gpx_path: str) -> dict:
        pts = load_gpx_points(gpx_path)
        if len(pts) < 2:
            return {"gpx_points": len(pts), "gpx_distance_mi": 0.0, "gpx_elev_gain_m": 0.0}
        d = haversine_miles(
            pts["lat"].values[:-1], pts["lon"].values[:-1],
            pts["lat"].values[1:], pts["lon"].values[1:],
        )
        elev_gain = pts["ele_m"].diff().clip(lower=0).sum() if pts["ele_m"].notna().any() else 0.0
        return {
            "gpx_points": len(pts),
            "gpx_distance_mi": round(float(d.sum()), 2),
            "gpx_elev_gain_m": round(float(elev_gain), 1),
        }


    def hr_stats_for_window(start, end) -> dict:
        mask = (hr_df["startDate"] >= start) & (hr_df["startDate"] <= end)
        vals = hr_df.loc[mask, "value"]
        if len(vals) == 0:
            return {"hr_avg": None, "hr_min": None, "hr_max": None, "hr_n": 0}
        return {
            "hr_avg": round(float(vals.mean()), 1),
            "hr_min": float(vals.min()),
            "hr_max": float(vals.max()),
            "hr_n": int(len(vals)),
        }

    mo.md("GPX + heart-rate matching helpers defined: `load_gpx_points`, `gpx_route_summary`, `hr_stats_for_window`.")

    return gpx_route_summary, hr_stats_for_window, load_gpx_points


@app.cell
def workout_summary_table(
    gpx_route_summary,
    hr_stats_for_window,
    mo,
    pd,
    workouts_df,
):
    N_RECENT = 5

    recent_gpx_workouts = workouts_df[workouts_df["has_gpx"]].sort_values(
        "startDate", ascending=False
    ).head(N_RECENT)
    recent_no_gpx_workouts = workouts_df[~workouts_df["has_gpx"]].sort_values(
        "startDate", ascending=False
    ).head(N_RECENT)

    recent_workouts = pd.concat([recent_gpx_workouts, recent_no_gpx_workouts]).sort_values(
        "startDate", ascending=False
    )


    def summarize_workout(row) -> dict:
        stats = row["stats"]
        energy = stats.get("ActiveEnergyBurned", {})
        dist = stats.get("DistanceWalkingRunning") or stats.get("DistanceCycling") or {}
        out = {
            "date": row["startDate"].date(),
            "activity": row["activity"],
            "source": row["source"],
            "duration_min": round(row["duration_min"], 1),
            "active_energy_cal": float(energy["sum"]) if energy.get("sum") else None,
            "distance_mi": float(dist["sum"]) if dist.get("sum") else None,
            "has_gpx": row["has_gpx"],
        }
        out.update(hr_stats_for_window(row["startDate"], row["endDate"]))
        if row["has_gpx"]:
            out.update(gpx_route_summary(row["gpx_path"]))
        return out


    workout_summaries = pd.DataFrame(
        [summarize_workout(r) for _, r in recent_workouts.iterrows()]
    )

    mo.vstack([
        mo.md(
            f"### Recent workout health summaries\n\n"
            f"The **{N_RECENT}** most recent GPX-linked workouts and the "
            f"**{N_RECENT}** most recent workouts with no GPX route "
            "(mostly Fitbit-via-Google-Health, which doesn't sync routes), "
            "each with heart rate matched from overlapping `HeartRate` records."
        ),
        mo.ui.table(workout_summaries, page_size=N_RECENT * 2),
    ])
    return recent_workouts, summarize_workout


@app.cell
def workout_picker_cell(mo, recent_workouts):
    _workout_labels = []
    for _, _r in recent_workouts.iterrows():
        _gpx_note = "" if _r["has_gpx"] else ", no GPX"
        _workout_labels.append(
            f"{_r['startDate'].strftime('%Y-%m-%d %H:%M')} - {_r['activity']} "
            f"({_r['source']}{_gpx_note})"
        )
    workout_lookup = dict(zip(_workout_labels, recent_workouts.index))

    workout_picker = mo.ui.dropdown(
        options=_workout_labels, value=_workout_labels[0], label="Workout"
    )
    workout_picker

    return workout_lookup, workout_picker


@app.cell
def workout_detail_cell(
    alt,
    hr_df,
    load_gpx_points,
    mo,
    recent_workouts,
    summarize_workout,
    workout_lookup,
    workout_picker,
):
    _row = recent_workouts.loc[workout_lookup[workout_picker.value]]
    _summary = summarize_workout(_row)

    _gpx_note = "no route file for this workout"
    if _row["has_gpx"]:
        _gpx_note = (
            f"yes - {_summary.get('gpx_points')} points, "
            f"{_summary.get('gpx_distance_mi')} mi, "
            f"{_summary.get('gpx_elev_gain_m')} m gain"
        )

    _cards = mo.md(
        f"""
        **{_row['activity']}** via {_row['source']} - {_row['startDate']}

        | metric | value |
        |---|---|
        | duration | {_summary['duration_min']} min |
        | active energy | {_summary.get('active_energy_cal')} Cal |
        | distance (HealthKit) | {_summary.get('distance_mi')} mi |
        | heart rate | avg {_summary.get('hr_avg')} bpm (min {_summary.get('hr_min')} / max {_summary.get('hr_max')}, n={_summary.get('hr_n')}) |
        | GPX route | {_gpx_note} |
        """
    )

    if _row["has_gpx"]:
        _pts = load_gpx_points(_row["gpx_path"])
        _route_chart = alt.Chart(_pts).mark_line().encode(
            x=alt.X("lon:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("lat:Q", scale=alt.Scale(zero=False)),
            order="time:T",
        ).properties(height=300, width=300, title="Route (lon/lat)")
        _elev_chart = alt.Chart(_pts).mark_area(opacity=0.6).encode(
            x="time:T", y=alt.Y("ele_m:Q", title="elevation (m)")
        ).properties(height=200, title="Elevation profile")
        _extra = mo.hstack([_route_chart, _elev_chart])
    else:
        _extra = mo.md("_No GPX route was recorded for this workout._")

    _hr_window = hr_df[
        (hr_df["startDate"] >= _row["startDate"]) & (hr_df["startDate"] <= _row["endDate"])
    ]
    if len(_hr_window):
        _hr_chart = alt.Chart(_hr_window).mark_line(point=True).encode(
            x="startDate:T", y=alt.Y("value:Q", title="heart rate (bpm)"), color="source:N",
        ).properties(height=200, title="Heart rate during workout")
    else:
        _hr_chart = mo.md("_No HeartRate records overlap this workout's time window._")

    mo.vstack([_cards, _extra, _hr_chart])

    return


@app.cell(hide_code=True)
def anon_export_header(mo):
    mo.md(r"""
    ## Anonymized, Kaggle-format export from `export.xml`

    `data/health_data.csv` (a Kaggle Apple Health dataset) uses one row per
    `<Record>` with columns:

    `type, sourceName, sourceVersion, unit, m_creationDate, m_creationTime,
    m_creationTime_am_pm, m_creationTimeZone, m_startDate, m_startTime,
    m_startTime_am_pm, m_startTimeZone, m_endDate, m_endTime,
    m_endTime_am_pm, m_endTimeZone, value, device`

    This section builds a robust streaming parser that turns our own
    `export.xml` into the same *shape* of table (one row per Record), but:

    - **Anonymized** - `sourceName`, `sourceVersion`, and `device` are
      dropped entirely. `device` in particular can embed a device's
      user-given name (e.g. "Weaver's Apple Watch"), so it isn't kept in
      any form.
    - **De-duplicated** - each timestamp is a single tz-aware datetime column
      instead of 4 columns (`date`, `time`, `am_pm`, `timeZone`) that encode
      the exact same instant redundantly - the AM/PM marker in particular is
      already embedded in the 12-hour time string, so it never carries new
      information.
    - **Prefix-stripped types** - `HKQuantityTypeIdentifierHeartRate` becomes
      `HeartRate`; the `HK...TypeIdentifier` prefix is constant boilerplate
      on every single row.
    """)

    return


@app.cell
def anon_export_helpers(mo, re):
    TYPE_PREFIX_RE = re.compile(
        r"^HK(?:QuantityTypeIdentifier|CategoryTypeIdentifier|CorrelationTypeIdentifier)"
    )


    def clean_record_type(raw_type: str) -> str:
        """Strip the boilerplate HK...TypeIdentifier prefix shared by every record."""
        return TYPE_PREFIX_RE.sub("", raw_type) if raw_type else raw_type


    mo.md("Helper defined: `clean_record_type`.")

    return (clean_record_type,)


@app.cell
def anon_export_build(ET, EXPORT_XML, Path, clean_record_type, mo, pd, time):
    def _parse_export_anonymized(path: Path) -> pd.DataFrame:
        """Stream-parse export.xml into one row per Record, anonymized.

        Only top-level <Record> elements are read (records nested inside a
        <Correlation> also appear as top-level Records per the DTD, so no
        special-casing is needed there). Each element is cleared immediately
        after use so the ~1GB file is never held in memory at once.
        """
        types, units, values = [], [], []
        creation, start, end = [], [], []

        for _, elem in ET.iterparse(str(path), events=("end",)):
            if elem.tag == "Record":
                types.append(clean_record_type(elem.get("type")))
                units.append(elem.get("unit"))
                values.append(elem.get("value"))
                creation.append(elem.get("creationDate"))
                start.append(elem.get("startDate"))
                end.append(elem.get("endDate"))
                elem.clear()
            elif elem.tag in ("Correlation", "Workout", "ActivitySummary",
                              "ClinicalRecord", "Audiogram", "VisionPrescription"):
                elem.clear()

        df = pd.DataFrame({
            "type": pd.Categorical(types),
            "unit": pd.Categorical(units),
            "value": values,
            "creationDate": pd.to_datetime(creation, format="%Y-%m-%d %H:%M:%S %z", utc=False),
            "startDate": pd.to_datetime(start, format="%Y-%m-%d %H:%M:%S %z", utc=False),
            "endDate": pd.to_datetime(end, format="%Y-%m-%d %H:%M:%S %z", utc=False),
        })
        return df


    _t0 = time.time()
    export_records_df = _parse_export_anonymized(EXPORT_XML)
    anon_parse_seconds = time.time() - _t0
    mo.md(
        f"Parsed **{len(export_records_df):,}** records into the anonymized "
        f"table in {anon_parse_seconds:.1f}s."
    )

    return (export_records_df,)


@app.cell
def anon_export_write(DATA_DIR, EXPORT_XML, export_records_df, mo):
    EXPORT_PARSED_CSV = DATA_DIR.parent / "health_export_parsed.csv"

    export_records_df.to_csv(EXPORT_PARSED_CSV, index=False)
    _size_mb = EXPORT_PARSED_CSV.stat().st_size / 1e6
    _xml_mb = EXPORT_XML.stat().st_size / 1e6

    mo.md(
        f"""
        Wrote **{len(export_records_df):,}** rows to
        `{EXPORT_PARSED_CSV}`
        ({_size_mb:.0f} MB, vs. {_xml_mb:.0f} MB for the source `export.xml` -
        no `sourceName`/`sourceVersion` columns, a single tz-aware timestamp per
        date instead of 4 split columns, and boilerplate stripped from `type`).
        """
    )

    return


@app.cell
def anon_export_compare(export_records_df, mo, pd):
    _kaggle_cols = [
        "type", "sourceName", "sourceVersion", "unit",
        "m_creationDate", "m_creationTime", "m_creationTime_am_pm", "m_creationTimeZone",
        "m_startDate", "m_startTime", "m_startTime_am_pm", "m_startTimeZone",
        "m_endDate", "m_endTime", "m_endTime_am_pm", "m_endTimeZone",
        "value", "device",
    ]
    _schema_compare = pd.DataFrame({
        "kaggle health_data.csv column": _kaggle_cols + [""] * (len(export_records_df.columns) - len(_kaggle_cols)) if len(export_records_df.columns) > len(_kaggle_cols) else _kaggle_cols,
    })
    _schema_compare = _schema_compare.assign(
        **{"anonymized export_records_df column": list(export_records_df.columns) + [""] * (len(_schema_compare) - len(export_records_df.columns))}
    )

    mo.vstack([
        mo.md(
            f"**{len(_kaggle_cols)} columns -> {len(export_records_df.columns)} columns**, "
            f"**{len(export_records_df):,} rows** parsed from `export.xml`, "
            f"no `sourceName`/`sourceVersion`, device string reduced to "
            f"non-identifying fields."
        ),
        mo.ui.table(_schema_compare, page_size=18),
        mo.md("Sample rows:"),
        mo.ui.table(export_records_df.sample(5, random_state=0), page_size=5),
    ])

    return


@app.cell
def anon_workouts_header(mo):
    mo.md(r"""
    ## Separate, anonymized workouts table

    Workouts are a different grain than `export_records_df` (one row per
    *session*, not per instantaneous measurement), so rather than merging
    them into the Records table we keep a second table, `workouts_df_anon`,
    written to its own CSV. It can be joined to the Records time series
    later via `workout_id` + `[startDate, endDate]` overlap - e.g. to pull
    every `HeartRate` record that falls inside a workout, or to compute
    per-day workout aggregates - without duplicating anything into the
    Records table itself.

    This is also where GPX data gets combined in: each workout's own
    `WorkoutStatistics` only carries cumulative energy/distance, so the
    GPX track (when one exists) is parsed to add distance, point count,
    and elevation gain that aren't available anywhere else.

    Anonymization: `sourceName` and `device` are dropped, matching the
    Records table. Cumulative-only `WorkoutStatistics` (sum populated,
    average/min/max always empty for these types) are flattened into named
    sum columns instead of a nested per-row dict, with the unit folded into
    the column name (`energy_active_cal`, `distance_cycling_mi`, ...)
    rather than kept as a separate, always-identical unit column.
    """)
    return


@app.cell
def anon_workouts_build(gpx_route_summary, mo, pd, workouts_df):
    _STAT_COLUMNS = {
        "ActiveEnergyBurned": "energy_active_cal",
        "BasalEnergyBurned": "energy_basal_cal",
        "DistanceCycling": "distance_cycling_mi",
        "DistancePaddleSports": "distance_paddle_mi",
        "DistanceWalkingRunning": "distance_walking_running_mi",
    }

    _rows = []
    for _i, _w in workouts_df.sort_values("startDate").reset_index(drop=True).iterrows():
        _row = {
            "workout_id": _i,
            "activity": _w["activity"],
            "startDate": _w["startDate"],
            "endDate": _w["endDate"],
            "duration_min": _w["duration_min"],
        }
        for _stat_key, _col in _STAT_COLUMNS.items():
            _val = _w["stats"].get(_stat_key, {}).get("sum")
            _row[_col] = float(_val) if _val is not None else None
        _row["has_gpx"] = _w["has_gpx"]
        _row["gpx_path"] = _w["gpx_path"]
        if _w["has_gpx"]:
            _row.update(gpx_route_summary(_w["gpx_path"]))
        else:
            _row.update({"gpx_points": None, "gpx_distance_mi": None, "gpx_elev_gain_m": None})
        _rows.append(_row)

    workouts_df_anon = pd.DataFrame(_rows)

    mo.vstack([
        mo.md(f"**{len(workouts_df_anon):,}** workouts, anonymized and GPX-enriched."),
        mo.ui.table(workouts_df_anon, page_size=10),
    ])

    return (workouts_df_anon,)


@app.cell
def anon_workouts_write(DATA_DIR, mo, workouts_df_anon):
    WORKOUTS_PARSED_CSV = DATA_DIR.parent / "health_workouts_parsed.csv"
    workouts_df_anon.to_csv(WORKOUTS_PARSED_CSV, index=False)

    mo.md(
        f"Wrote **{len(workouts_df_anon):,}** workouts to `{WORKOUTS_PARSED_CSV}` "
        f"({WORKOUTS_PARSED_CSV.stat().st_size / 1e3:.0f} KB). Join to "
        f"`export_records_df` later via `[startDate, endDate]` overlap per "
        f"`workout_id`, e.g. for per-workout or per-day heart-rate detail that "
        f"a plain average would flatten away (FFT/wavelet features on the "
        f"matched `HeartRate` window, per workout or per day)."
    )

    return


if __name__ == "__main__":
    app.run()
