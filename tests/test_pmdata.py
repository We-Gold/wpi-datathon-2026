"""The PMData reader, against a small directory shaped like a real subject."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from health import anonymize, metrics, units, validate
from health.readers import pmdata
from health.schema import COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "pmdata_sample"


@pytest.fixture
def frame() -> pl.DataFrame:
    """Straight off the reader, so units are still the ones the files record."""
    return pmdata.read(FIXTURE, "subject_test", anonymize.SourceMap())


@pytest.fixture
def canonical(frame: pl.DataFrame) -> pl.DataFrame:
    """After unit conversion, which is what actually gets written."""
    return units.canonicalize(frame)


def rows(frame: pl.DataFrame, metric: str) -> pl.DataFrame:
    """One metric's rows, oldest first. metric is Categorical, so it is cast."""
    return frame.filter(pl.col("metric").cast(pl.String) == metric).sort("start_local")


class TestShape:
    def test_columns_are_canonical_and_ordered(self, frame: pl.DataFrame) -> None:
        assert frame.columns == COLUMNS

    def test_source_format_is_pmdata(self, frame: pl.DataFrame) -> None:
        assert frame["source_format"].unique().to_list() == ["pmdata"]

    def test_a_file_is_not_a_subject(self) -> None:
        # The other readers take a file. Handing this one a file is a mistake
        # worth naming rather than a confusing failure further down.
        with pytest.raises(ValueError, match="not a directory"):
            pmdata.read(FIXTURE / "fitbit" / "steps.json", "subject_test", anonymize.SourceMap())

    def test_output_passes_validation(self, canonical: pl.DataFrame) -> None:
        assert validate.check(canonical) == []


class TestMissingFiles:
    def test_a_missing_stream_is_not_an_error(self, frame: pl.DataFrame) -> None:
        # The fixture has no lightly_active_minutes.json, which is true of real
        # subjects too. The rest of the directory still has to read.
        assert rows(frame, "LightActivityTime").height == 0
        assert frame.height > 0

    def test_streams_that_are_present_still_read(self, frame: pl.DataFrame) -> None:
        assert rows(frame, "SedentaryTime").height == 1


class TestUnits:
    def test_distance_is_read_as_centimetres(self, frame: pl.DataFrame) -> None:
        assert (
            rows(frame, metrics.QUANTITY_PREFIX + "DistanceWalkingRunning")["unit"].item() == "cm"
        )

    def test_distance_becomes_metres(self, canonical: pl.DataFrame) -> None:
        row = rows(canonical, metrics.QUANTITY_PREFIX + "DistanceWalkingRunning")
        assert row["value_num"].item() == 25.0
        assert row["unit"].item() == "m"

    def test_daily_minute_counts_become_seconds(self, canonical: pl.DataFrame) -> None:
        row = rows(canonical, "SedentaryTime")
        assert row["value_num"].item() == 636 * 60
        assert row["unit"].item() == "s"

    def test_self_reported_sleep_hours_become_seconds(self, canonical: pl.DataFrame) -> None:
        row = rows(canonical, metrics.SLEEP_DURATION_DAILY)
        assert row["value_num"].to_list() == [6 * 3600, 7 * 3600]


class TestTimestamps:
    def test_local_is_what_the_file_said(self, frame: pl.DataFrame) -> None:
        row = rows(frame, metrics.QUANTITY_PREFIX + "StepCount").head(1)
        assert row["start_local"].item() == dt.datetime(2019, 11, 1, 0, 0, 0)

    def test_utc_is_one_hour_behind_the_wall_clock(self, frame: pl.DataFrame) -> None:
        row = rows(frame, metrics.QUANTITY_PREFIX + "StepCount").head(1)
        assert row["start_utc"].item() == dt.datetime(2019, 10, 31, 23, 0, tzinfo=dt.UTC)

    def test_the_offset_does_not_change_at_daylight_saving(self, frame: pl.DataFrame) -> None:
        # Oslo jumped 02:00 to 03:00 on 2020-03-29, but the devices kept
        # recording through that hour, so their clock did not move. A named zone
        # would drop this row and shift everything after it by an hour.
        row = rows(frame, metrics.QUANTITY_PREFIX + "StepCount").filter(
            pl.col("start_local") == dt.datetime(2020, 3, 29, 2, 30)
        )
        assert row.height == 1
        assert row["start_utc"].item() == dt.datetime(2020, 3, 29, 1, 30, tzinfo=dt.UTC)

    def test_no_row_loses_its_instant(self, frame: pl.DataFrame) -> None:
        assert frame["start_utc"].null_count() == 0

    def test_span_is_set_from_the_grain(self, frame: pl.DataFrame) -> None:
        # A minute stream covers a minute. Nothing in the file says so.
        row = rows(frame, metrics.QUANTITY_PREFIX + "StepCount").head(1)
        assert row["end_local"].item() - row["start_local"].item() == dt.timedelta(minutes=1)

    def test_created_is_null_because_pmdata_has_none(self, frame: pl.DataFrame) -> None:
        assert frame["created_utc"].null_count() == frame.height


class TestNestedValues:
    def test_heart_rate_is_taken_from_the_value_object(self, frame: pl.DataFrame) -> None:
        row = rows(frame, metrics.QUANTITY_PREFIX + "HeartRate")
        assert row["value_num"].to_list() == [54.0, 52.0]

    def test_resting_heart_rate_uses_the_inner_value(self, frame: pl.DataFrame) -> None:
        assert rows(frame, metrics.QUANTITY_PREFIX + "RestingHeartRate")["value_num"].item() == 53.5


class TestSleepStages:
    def test_stage_names_become_healthkit_values(self, frame: pl.DataFrame) -> None:
        stages = set(rows(frame, metrics.CATEGORY_PREFIX + "SleepAnalysis")["value_str"].to_list())
        assert metrics.VALUE_PREFIX + "SleepAnalysisAsleepCore" in stages
        assert metrics.VALUE_PREFIX + "SleepAnalysisAsleepDeep" in stages
        assert metrics.VALUE_PREFIX + "SleepAnalysisAsleepREM" in stages
        assert metrics.VALUE_PREFIX + "SleepAnalysisAwake" in stages

    def test_classic_sleep_is_not_promoted_to_a_stage(self, frame: pl.DataFrame) -> None:
        # A classic log only sees movement. Calling its sleep Core would invent
        # a measurement the device never made.
        stages = rows(frame, metrics.CATEGORY_PREFIX + "SleepAnalysis")
        classic = stages.filter(pl.col("start_local").dt.date() == dt.date(2019, 11, 3))
        assert classic["value_str"].to_list() == [
            metrics.VALUE_PREFIX + "SleepAnalysisAsleepUnspecified",
            metrics.VALUE_PREFIX + "SleepAnalysisAsleepUnspecified",
            metrics.VALUE_PREFIX + "SleepAnalysisAwake",
        ]

    def test_an_unmapped_stage_is_dropped(self, frame: pl.DataFrame) -> None:
        # The fixture holds one "unknown" level, as the real files do.
        stages = rows(frame, metrics.CATEGORY_PREFIX + "SleepAnalysis")
        assert stages.height == 7
        assert all("Unknown" not in value for value in stages["value_str"].to_list())

    def test_short_data_is_not_read(self, frame: pl.DataFrame) -> None:
        # shortData holds brief wake episodes that sit inside the same
        # intervals as data. Reading both counts that time twice.
        awake = rows(frame, metrics.CATEGORY_PREFIX + "SleepAnalysis").filter(
            pl.col("value_str").cast(pl.String) == metrics.VALUE_PREFIX + "SleepAnalysisAwake"
        )
        assert awake.filter(pl.col("start_local").dt.date() == dt.date(2019, 11, 2)).height == 1

    def test_stage_length_comes_from_the_seconds_field(self, frame: pl.DataFrame) -> None:
        stages = rows(frame, metrics.CATEGORY_PREFIX + "SleepAnalysis").head(1)
        assert stages["end_local"].item() - stages["start_local"].item() == dt.timedelta(seconds=30)


class TestExerciseMinutes:
    def test_moderate_and_vigorous_are_summed(self, canonical: pl.DataFrame) -> None:
        # Apple's exercise time is brisk activity and above, which is both
        # buckets. Two rows under one name would double count the day.
        row = rows(canonical, metrics.QUANTITY_PREFIX + "AppleExerciseTime")
        assert row.height == 1
        assert row["value_num"].item() == (58 + 72) * 60


class TestSelfReport:
    def test_a_skipped_answer_produces_no_row(self, frame: pl.DataFrame) -> None:
        # Zero is off the 1 to 5 scale and means the question was skipped. The
        # second wellness row in the fixture has fatigue 0 and mood 3, so
        # fatigue loses that day and mood keeps it. Storing the zero instead
        # would drag every average down.
        assert rows(frame, "Fatigue")["value_num"].to_list() == [2.0]
        assert rows(frame, "Mood")["value_num"].to_list() == [3.0, 3.0]

    def test_zero_readiness_is_a_real_answer(self, frame: pl.DataFrame) -> None:
        # readiness runs 0 to 10, so zero is in range and has to survive.
        assert rows(frame, "Readiness")["value_num"].to_list() == [5.0, 0.0]

    def test_session_rpe_and_duration_are_both_kept(self, canonical: pl.DataFrame) -> None:
        assert rows(canonical, "PerceivedExertion")["value_num"].item() == 7.0
        assert rows(canonical, "TrainingDuration")["value_num"].item() == 30 * 60

    def test_a_session_starts_where_its_duration_says(self, frame: pl.DataFrame) -> None:
        row = rows(frame, "PerceivedExertion")
        assert row["end_local"].item() - row["start_local"].item() == dt.timedelta(minutes=30)


class TestInjuries:
    def test_only_the_count_is_kept(self, frame: pl.DataFrame) -> None:
        assert rows(frame, "InjuryCount")["value_num"].to_list() == [0.0, 2.0]

    def test_body_parts_never_reach_the_output(self, frame: pl.DataFrame) -> None:
        text = " ".join(
            value
            for column in ("metric", "value_str", "unit")
            for value in frame[column].cast(pl.String).drop_nulls().to_list()
        )
        assert "left_foot" not in text
        assert "right_hand" not in text


class TestReporting:
    def test_the_date_is_read_day_first(self, frame: pl.DataFrame) -> None:
        # 06/11/2019 is the sixth of November, not the eleventh of June.
        row = rows(frame, metrics.QUANTITY_PREFIX + "BodyMass").head(1)
        assert row["start_local"].item() == dt.datetime(2019, 11, 6)

    def test_a_blank_weight_produces_no_row(self, frame: pl.DataFrame) -> None:
        # The middle day of the fixture has no weight. A row with no value at
        # all fails validation, so it must not be written.
        assert rows(frame, metrics.QUANTITY_PREFIX + "BodyMass")["value_num"].to_list() == [
            100.0,
            99.0,
        ]

    def test_fluid_is_not_read(self, frame: pl.DataFrame) -> None:
        # The column counts glasses, not volume, so any mL figure would be a
        # guess at a glass size. Other sources record the volume itself.
        assert rows(frame, metrics.QUANTITY_PREFIX + "DietaryWater").height == 0

    def test_dropped_columns_produce_no_rows(self, frame: pl.DataFrame) -> None:
        # meals and alcohol are read past on purpose. See
        # metrics.DROPPED_PMDATA_METRICS for why.
        present = set(frame["metric"].cast(pl.String).unique().to_list())
        assert not (present & metrics.DROPPED_PMDATA_METRICS)


class TestDroppedStreams:
    def test_only_the_overall_sleep_score_is_kept(self, frame: pl.DataFrame) -> None:
        # The three component scores add up to the overall one exactly, so
        # keeping them would store the same number twice.
        assert rows(frame, "SleepScore")["value_num"].item() == 76.0
        assert rows(frame, "SleepCompositionScore").height == 0

    def test_nothing_dropped_reaches_the_output(self, frame: pl.DataFrame) -> None:
        present = set(frame["metric"].cast(pl.String).unique().to_list())
        assert not (present & metrics.DROPPED_PMDATA_METRICS)

    def test_dropped_names_are_not_in_the_vocabulary(self) -> None:
        assert not (metrics.DROPPED_PMDATA_METRICS & metrics.NON_HEALTHKIT_METRICS)


class TestSourceType:
    def test_fitbit_is_a_wearable(self, frame: pl.DataFrame) -> None:
        row = rows(frame, metrics.QUANTITY_PREFIX + "StepCount").head(1)
        assert row["source_type"].item() == "wearable"

    def test_self_reported_sources_are_manual(self, frame: pl.DataFrame) -> None:
        # Both the athlete app and the food form are a person typing, so they
        # group together against anything a sensor measured.
        assert rows(frame, "Mood")["source_type"].unique().to_list() == ["manual"]
        weight = rows(frame, metrics.QUANTITY_PREFIX + "BodyMass")
        assert weight["source_type"].unique().to_list() == ["manual"]

    def test_pmdata_needs_no_source_classification(self) -> None:
        # There are no device names anywhere in the dataset, so scan has
        # nothing to ask a person about.
        assert pmdata.source_names(FIXTURE) == set()
