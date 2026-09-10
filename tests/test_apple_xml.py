"""The streaming Apple Health reader, against a synthetic export."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import polars as pl
import pytest

from health import anonymize
from health.readers import apple_xml
from health.schema import COLUMNS

FIXTURE = Path(__file__).parent / "fixtures" / "apple_xml_sample.xml"


def _single_quoted(source: Path, tmp_path: Path) -> Path:
    """The same export with every attribute single quoted. Still valid XML."""
    rewritten = re.sub(
        rb'([\w:.-]+)="([^"]*)"',
        lambda match: b"%s='%s'" % (match.group(1), match.group(2)),
        source.read_bytes(),
    )
    target = tmp_path / "single_quoted.xml"
    target.write_bytes(rewritten)
    return target


@pytest.fixture
def frame() -> pl.DataFrame:
    return apple_xml.read(FIXTURE, "subject_test", anonymize.SourceMap())


class TestShape:
    def test_columns_are_canonical_and_ordered(self, frame: pl.DataFrame) -> None:
        assert frame.columns == COLUMNS

    def test_reads_top_level_records_only(self, frame: pl.DataFrame) -> None:
        # Four standalone records plus the two blood pressure records that the
        # export repeats at top level. The correlation's own copies are skipped.
        assert frame.height == 6

    def test_correlation_children_are_not_counted_twice(self, frame: pl.DataFrame) -> None:
        # The DTD states correlation children also appear at top level. Reading
        # both copies would silently double every correlated measurement.
        systolic = frame.filter(
            pl.col("metric").cast(pl.String).str.ends_with("BloodPressureSystolic")
        )
        assert systolic.height == 1

    def test_workouts_are_out_of_scope(self, frame: pl.DataFrame) -> None:
        assert not any("Workout" in name for name in frame["metric"].cast(pl.String).to_list())


class TestValues:
    def test_numeric_and_categorical_values_separate(self, frame: pl.DataFrame) -> None:
        sleep = frame.filter(pl.col("metric").cast(pl.String).str.ends_with("SleepAnalysis"))
        assert sleep["value_str"].item() == "HKCategoryValueSleepAnalysisAsleepCore"
        assert sleep["value_num"].item() is None

    def test_exactly_one_value_column_is_set_per_row(self, frame: pl.DataFrame) -> None:
        both = frame.filter(pl.col("value_num").is_not_null() & pl.col("value_str").is_not_null())
        neither = frame.filter(pl.col("value_num").is_null() & pl.col("value_str").is_null())
        assert both.height == 0
        assert neither.height == 0

    def test_units_are_kept_as_recorded(self, frame: pl.DataFrame) -> None:
        # This export writes height in feet while others use centimetres. That
        # disagreement is preserved rather than quietly converted here.
        height = frame.filter(pl.col("metric").cast(pl.String).str.ends_with("Height"))
        assert height["unit"].item() == "ft"
        assert height["value_num"].item() == pytest.approx(5.58071)


class TestTimestamps:
    def test_offset_applied(self, frame: pl.DataFrame) -> None:
        row = frame.filter(pl.col("value_num") == 72)
        assert row["start_local"].item() == dt.datetime(2024, 3, 1, 8, 59, 0)
        assert row["start_utc"].item() == dt.datetime(2024, 3, 1, 12, 59, 0, tzinfo=dt.UTC)


class TestAnonymization:
    def test_device_blob_is_not_read_at_all(self, frame: pl.DataFrame) -> None:
        # The fixture carries a full HKDevice blob with a device name, a memory
        # pointer, and a pairing date. None of it should reach the output.
        text = str(frame.to_dicts())
        for fragment in ("HKDevice", "0x8af56c540", "Apple Inc.", "Watch3,2"):
            assert fragment not in text

    def test_no_date_of_birth_or_sex_anywhere(self, frame: pl.DataFrame) -> None:
        # The Me element is excluded by scope, but scope is not a guarantee, so
        # the absence is asserted directly.
        text = str(frame.to_dicts())
        assert "1999-01-31" not in text
        assert "BiologicalSex" not in text

    def test_no_sync_identifiers_from_metadata_entries(self, frame: pl.DataFrame) -> None:
        assert "B0F3BCEC-B12A-4DB5-89C4-665838D51FAF" not in str(frame.to_dicts())

    def test_source_types_classify(self, frame: pl.DataFrame) -> None:
        by_value = dict(
            zip(
                frame["value_num"].to_list(),
                frame["source_type"].to_list(),
                strict=True,
            )
        )
        assert by_value[236.588] == "app"
        assert by_value[72.0] == "wearable"
        assert by_value[5.58071] == "scale"
        assert by_value[118.0] == "manual"


class TestBatching:
    def test_small_batches_give_the_same_result(self) -> None:
        # Batch size must not change the output, only the memory profile.
        one_batch = apple_xml.read(FIXTURE, "subject_test", anonymize.SourceMap())
        many = apple_xml.read(FIXTURE, "subject_test", anonymize.SourceMap(), batch_size=1)
        assert many.height == one_batch.height
        assert many.to_dicts() == one_batch.to_dicts()


class TestSourceNames:
    """The cheap path scan uses instead of a full parse."""

    def test_finds_every_source_name(self) -> None:
        names = apple_xml.source_names(FIXTURE)
        # Includes the workout and correlation names, which the record reader
        # skips. A wider net is what scan wants.
        assert names == {
            "WaterMinder",
            "Person\u2019s Apple Watch",
            "Nickname",
            "Withings",
            "Health",
            "Strava",
        }

    def test_decodes_character_references(self) -> None:
        # The fixture writes the curly apostrophe as &#8217;. Left encoded, a
        # nickname would not match the local overrides file.
        assert "Person\u2019s Apple Watch" in apple_xml.source_names(FIXTURE)

    def test_agrees_with_the_full_reader(self) -> None:
        cheap = anonymize.SourceMap()
        for name in apple_xml.source_names(FIXTURE):
            cheap.classify(name)

        full = anonymize.SourceMap()
        apple_xml.read(FIXTURE, "subject_test", full)

        assert cheap.unmatched >= full.unmatched

    def test_survives_a_chunk_boundary_mid_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expected = apple_xml.source_names(FIXTURE)
        # Sixteen bytes at a time, so most attributes in the file straddle a
        # read and the carryover has to stitch them back together.
        monkeypatch.setattr(apple_xml, "_READ_CHUNK", 16)
        assert apple_xml.source_names(FIXTURE) == expected


class TestFastPathAndFallback:
    """The byte scanner has to agree with the parser or get out of the way."""

    def test_byte_scanner_matches_iterparse(self) -> None:
        fast = apple_xml._read_with(
            apple_xml._scan_frames, FIXTURE, "subject_test", anonymize.SourceMap(), 10
        )
        slow = apple_xml._read_with(
            apple_xml._iterparse_frames, FIXTURE, "subject_test", anonymize.SourceMap(), 10
        )
        assert pl.concat(fast).equals(pl.concat(slow))

    def test_entity_references_are_resolved(self, frame: pl.DataFrame) -> None:
        # The fixture writes a curly apostrophe as &#8217;. The parser resolves
        # it for free; the byte scanner has to do it, or the watch stops being
        # classifiable as a wearable.
        watch = frame.filter(pl.col("source_type").cast(pl.String) == "wearable")
        assert watch.height == 1

    def test_single_quoted_attributes_fall_back(self, tmp_path: Path) -> None:
        # Not something Apple's exporter writes, but valid XML. The scanner
        # cannot read it, so it must hand off rather than return nothing.
        quirky = _single_quoted(FIXTURE, tmp_path)

        with pytest.raises(apple_xml._UnexpectedShape):
            list(apple_xml._scan_frames(quirky, 10))

        assert apple_xml.read(quirky, "subject_test", anonymize.SourceMap()).height == 6

    def test_fallback_does_not_leave_stale_classifications(self, tmp_path: Path) -> None:
        # The abandoned attempt already pushed names into unmatched. Left
        # there, scan would report the same source twice.
        quirky = _single_quoted(FIXTURE, tmp_path)

        source_map = anonymize.SourceMap()
        apple_xml.read(quirky, "subject_test", source_map)

        assert source_map.unmatched == {"Nickname"}
