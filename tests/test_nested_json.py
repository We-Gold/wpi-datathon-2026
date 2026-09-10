"""The nested JSON reader, against a synthetic file shaped like the real one."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from health import anonymize, metrics
from health.readers import nested_json
from health.schema import COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "nested_json_sample.json"


@pytest.fixture
def frame() -> pl.DataFrame:
    return nested_json.read(FIXTURE, "subject_test", anonymize.SourceMap())


class TestShape:
    def test_columns_are_canonical_and_ordered(self, frame: pl.DataFrame) -> None:
        assert frame.columns == COLUMNS

    def test_records_from_every_metric_section_are_read(self, frame: pl.DataFrame) -> None:
        assert frame.height == 4

    def test_metadata_block_is_not_treated_as_a_metric(self, frame: pl.DataFrame) -> None:
        # It sits alongside the metric keys but describes the export, not the
        # subject. Parsing it as one would invent a metric named "metadata".
        assert "metadata" not in frame["metric"].to_list()


class TestSleepRename:
    def test_preaggregated_sleep_gets_its_own_metric_name(self, frame: pl.DataFrame) -> None:
        names = frame["metric"].to_list()
        assert metrics.SLEEP_DURATION_DAILY in names
        # The whole point: it must not share a name with per-stage sleep, or a
        # single filter returns two incompatible grains.
        assert "HKCategoryTypeIdentifierSleepAnalysis" not in names

    def test_the_renamed_metric_is_not_a_healthkit_identifier(self) -> None:
        assert not metrics.is_healthkit(metrics.SLEEP_DURATION_DAILY)

    def test_renamed_sleep_keeps_its_numeric_value_and_unit(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("metric") == metrics.SLEEP_DURATION_DAILY)
        assert row["value_num"].item() == 3.58
        assert row["unit"].item() == "hours"


class TestTimestamps:
    def test_offset_with_colon_is_applied(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("value_num") == 110)
        assert row["start_local"].item() == dt.datetime(2024, 6, 9, 10, 31, 50)
        assert row["start_utc"].item() == dt.datetime(2024, 6, 9, 7, 31, 50, tzinfo=dt.UTC)

    def test_z_suffix_is_read_as_utc(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("value_num") == 70.4)
        assert row["start_utc"].item() == dt.datetime(2024, 6, 10, 7, 0, 0, tzinfo=dt.UTC)


class TestAnonymization:
    def test_non_breaking_space_in_watch_name_still_classifies(self, frame: pl.DataFrame) -> None:
        # This is the case that silently fails without NFKC folding.
        row = frame.filter(pl.col("value_num") == 110)
        assert row["source_type"].item() == "wearable"

    def test_phone_and_scale_classify(self, frame: pl.DataFrame) -> None:
        assert frame.filter(pl.col("value_num") == 1)["source_type"].item() == "phone"
        assert frame.filter(pl.col("value_num") == 70.4)["source_type"].item() == "scale"

    def test_source_and_device_lists_are_not_read(self, frame: pl.DataFrame) -> None:
        # Those lists hold device names verbatim. Nothing from them should
        # appear anywhere in the output.
        assert "source_name" not in frame.columns
        assert "Watch4,2" not in str(frame.to_dicts())


def test_source_names_matches_the_full_read() -> None:
    """Scan's cheap path has to see the same names the reader classifies."""
    cheap = anonymize.SourceMap()
    for name in nested_json.source_names(FIXTURE):
        cheap.classify(name)

    full = anonymize.SourceMap()
    nested_json.read(FIXTURE, "subject_test", full)

    assert cheap.unmatched == full.unmatched
