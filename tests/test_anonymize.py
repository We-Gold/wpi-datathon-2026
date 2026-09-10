"""Source classification, including the exact strings the real exports contain.

The strings here are shapes taken from the real files with the names replaced.
No real source name is committed, which is the point of the module under test.
"""

from __future__ import annotations

import polars as pl
import pytest

from health import anonymize
from health.schema import SOURCE_TYPES


class TestNormalize:
    def test_folds_non_breaking_space(self) -> None:
        # The nested JSON writes "Apple\u00a0Watch" with a non-breaking space.
        # Without NFKC this never matches the wearable rule. Written as escapes
        # so the characters stay visible and cannot be lost in editing.
        #
        # NFKC folds the space but leaves the curly apostrophe alone, which is
        # fine: the apostrophe is never matched against, only discarded.
        assert anonymize.normalize("Person\u2019s Apple\u00a0Watch") == (
            "person\u2019s apple watch"
        )

    def test_strips_instance_suffix(self) -> None:
        assert anonymize.normalize("Nickname (17426)") == "nickname"

    def test_collapses_whitespace_and_case(self) -> None:
        assert anonymize.normalize("  Google   HEALTH ") == "google health"


class TestClassify:
    @pytest.fixture
    def source_map(self) -> anonymize.SourceMap:
        return anonymize.SourceMap()

    @pytest.mark.parametrize(
        "raw",
        [
            "Person\u2019s Apple Watch",
            "Person\u2019s Apple\u00a0Watch",
            "Apple Watch",
        ],
    )
    def test_watch_is_wearable_whatever_name_wraps_it(
        self, source_map: anonymize.SourceMap, raw: str
    ) -> None:
        assert source_map.classify(raw) == "wearable"

    @pytest.mark.parametrize(
        "raw",
        ["Person\u2019s iPhone (5647)", "iPhone Person", "iPhone - Person", "iPhone"],
    )
    def test_phone_is_phone_whatever_name_wraps_it(
        self, source_map: anonymize.SourceMap, raw: str
    ) -> None:
        assert source_map.classify(raw) == "phone"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Google Health", "wearable"),
            ("Zepp", "wearable"),
            ("Withings", "scale"),
            ("Strava", "app"),
            ("Health", "manual"),
            ("Aggregated", "derived"),
        ],
    )
    def test_known_products(self, source_map: anonymize.SourceMap, raw: str, expected: str) -> None:
        assert source_map.classify(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "Nickname",
            "RO-SLU_F2LSG7JQHG04",
            # The stray header row in the flat CSV puts the column name in the
            # column. It must not become a category.
            "sourceName",
            "",
            "   ",
        ],
    )
    def test_unrecognized_becomes_unknown(self, source_map: anonymize.SourceMap, raw: str) -> None:
        assert source_map.classify(raw) == "unknown"

    def test_null_becomes_unknown(self, source_map: anonymize.SourceMap) -> None:
        assert source_map.classify(None) == "unknown"

    def test_unmatched_values_are_recorded_for_scanning(
        self, source_map: anonymize.SourceMap
    ) -> None:
        source_map.classify("Nickname")
        source_map.classify("Strava")
        assert source_map.unmatched == {"Nickname"}

    def test_every_result_is_in_the_schema_enum(self, source_map: anonymize.SourceMap) -> None:
        # The leakage gate relies on the output vocabulary being closed. If a
        # rule ever returns something outside the enum, that assumption breaks.
        raws = ["Person\u2019s Apple Watch", "Zepp", "Health", "Nickname", None, "Aggregated"]
        assert {source_map.classify(raw) for raw in raws} <= set(SOURCE_TYPES)


class TestOverrides:
    def test_override_classifies_a_nickname(self) -> None:
        source_map = anonymize.SourceMap({"Nickname": "phone"})
        assert source_map.classify("Nickname") == "phone"

    def test_override_matches_after_normalization(self) -> None:
        # The local file can be written without the instance suffix and still
        # cover "Nickname (17426)".
        source_map = anonymize.SourceMap({"Nickname": "phone"})
        assert source_map.classify("Nickname (17426)") == "phone"

    def test_override_beats_builtin_rule(self) -> None:
        source_map = anonymize.SourceMap({"Strava": "wearable"})
        assert source_map.classify("Strava") == "wearable"

    def test_invalid_source_type_is_rejected_at_load(self) -> None:
        with pytest.raises(ValueError, match="not in the schema"):
            anonymize.SourceMap({"Nickname": "smartwatch"})


class TestSourceTypeExpr:
    def test_replaces_column_and_defaults_unmapped_to_unknown(self) -> None:
        mapping = {"Strava": "app"}
        got = (
            pl.DataFrame({"sourceName": ["Strava", "Something Else", None]})
            .select(anonymize.source_type_expr("sourceName", mapping))["source_type"]
            .to_list()
        )
        assert got == ["app", "unknown", "unknown"]
