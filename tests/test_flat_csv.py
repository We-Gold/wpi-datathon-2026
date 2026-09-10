"""The flattened CSV reader, against a synthetic file shaped like the real one."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from health import anonymize, metrics
from health.readers import apple_xml, flat_csv
from health.schema import COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "flat_csv_sample.csv"


@pytest.fixture
def frame() -> pl.DataFrame:
    source_map = anonymize.SourceMap()
    return flat_csv.read(FIXTURE, subject_id="subject_test", source_map=source_map)


class TestShape:
    def test_columns_are_canonical_and_ordered(self, frame: pl.DataFrame) -> None:
        assert frame.columns == COLUMNS

    def test_repeated_header_row_is_dropped(self, frame: pl.DataFrame) -> None:
        # The fixture deliberately embeds a second header line, the way
        # concatenating two exports would. Five real records should survive.
        assert frame.height == 5
        assert "type" not in frame["metric"].to_list()

    def test_subject_id_is_applied(self, frame: pl.DataFrame) -> None:
        assert frame["subject_id"].unique().to_list() == ["subject_test"]


class TestMetrics:
    def test_quantity_prefix_restored(self, frame: pl.DataFrame) -> None:
        assert "HKQuantityTypeIdentifierStepCount" in frame["metric"].to_list()

    def test_category_prefix_restored_for_sleep(self, frame: pl.DataFrame) -> None:
        # Sleep is a category type, not a quantity type. Getting this wrong
        # produces a plausible-looking but nonexistent identifier.
        assert "HKCategoryTypeIdentifierSleepAnalysis" in frame["metric"].to_list()


class TestValues:
    def test_numeric_value_lands_in_value_num(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("metric").cast(pl.String).str.ends_with("StepCount"))
        assert row["value_num"].item() == 412.0
        assert row["value_str"].item() is None

    def test_categorical_value_lands_in_value_str(self, frame: pl.DataFrame) -> None:
        """This format writes a bare "AsleepCore". The prefix goes back on."""
        row = frame.filter(pl.col("metric").cast(pl.String).str.ends_with("SleepAnalysis"))
        assert row["value_str"].item() == "HKCategoryValueSleepAnalysisAsleepCore"
        assert row["value_num"].item() is None

    def test_exactly_one_value_column_is_set_per_row(self, frame: pl.DataFrame) -> None:
        both = frame.filter(pl.col("value_num").is_not_null() & pl.col("value_str").is_not_null())
        neither = frame.filter(pl.col("value_num").is_null() & pl.col("value_str").is_null())
        assert both.height == 0
        assert neither.height == 0


class TestTimestamps:
    def test_midnight_hour_and_offset_applied(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("metric").cast(pl.String).str.ends_with("SleepAnalysis"))
        # 12:51 AM at +0530 is 00:51 local and 19:21 the previous day in UTC.
        assert row["start_local"].item() == dt.datetime(2023, 1, 27, 0, 51, 0)
        assert row["start_utc"].item() == dt.datetime(2023, 1, 26, 19, 21, 0, tzinfo=dt.UTC)

    def test_local_and_utc_days_can_differ(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("metric").cast(pl.String).str.ends_with("SleepAnalysis"))
        assert row["start_local"].dt.date().item() == dt.date(2023, 1, 27)
        assert row["start_utc"].dt.date().item() == dt.date(2023, 1, 26)


class TestAnonymization:
    def test_source_names_do_not_survive(self, frame: pl.DataFrame) -> None:
        assert "sourceName" not in frame.columns
        assert "source_name" not in frame.columns

    def test_devices_classify_by_pattern_not_by_name(self, frame: pl.DataFrame) -> None:
        by_metric = dict(
            zip(frame["metric"].to_list(), frame["source_type"].to_list(), strict=True)
        )
        assert by_metric["HKQuantityTypeIdentifierHeight"] == "phone"
        assert by_metric["HKQuantityTypeIdentifierHeartRate"] == "wearable"
        assert by_metric["HKQuantityTypeIdentifierBodyMassIndex"] == "wearable"

    def test_unrecognized_nickname_becomes_unknown_and_is_reported(self) -> None:
        source_map = anonymize.SourceMap()
        result = flat_csv.read(FIXTURE, subject_id="subject_test", source_map=source_map)

        steps = result.filter(pl.col("metric").cast(pl.String).str.ends_with("StepCount"))
        assert steps["source_type"].item() == "unknown"
        # The scan command relies on this so a person can classify it locally.
        assert "Nickname" in source_map.unmatched

    def test_source_version_is_not_carried(self, frame: pl.DataFrame) -> None:
        # Version histories are a decent fingerprint and carry no analytic value.
        assert not any("version" in name.lower() for name in frame.columns)


def test_source_names_matches_the_full_read() -> None:
    """Scan's cheap path has to see the same names the reader classifies."""
    cheap = anonymize.SourceMap()
    for name in flat_csv.source_names(FIXTURE):
        cheap.classify(name)

    full = anonymize.SourceMap()
    flat_csv.read(FIXTURE, "subject_test", full)

    assert cheap.unmatched == full.unmatched


class TestCategoryVocabulary:
    """Sleep stages have to be spelled the same way here as in the XML.

    Two spellings in one column split every stage into two buckets, and a
    group-by shows no sign of it.
    """

    def test_agrees_with_the_apple_xml_reader(self, frame: pl.DataFrame) -> None:
        xml = apple_xml.read(
            FIXTURE.parent / "apple_xml_sample.xml", "subject_test", anonymize.SourceMap()
        )

        def stages(f: pl.DataFrame) -> set[str]:
            sleep = f.filter(pl.col("metric").cast(pl.String).str.ends_with("SleepAnalysis"))
            return set(sleep["value_str"].cast(pl.String).drop_nulls().to_list())

        assert stages(frame) & stages(xml)

    def test_every_category_value_carries_the_prefix(self, frame: pl.DataFrame) -> None:
        values = frame["value_str"].cast(pl.String).drop_nulls().to_list()
        assert values
        assert all(value.startswith(metrics.VALUE_PREFIX) for value in values)
