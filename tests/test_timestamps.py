"""Timestamp parsing, including the shapes each real export actually contains."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import polars as pl
import pytest

from health import timestamps


def _eval(expr: pl.Expr, **columns: Sequence[str | None]) -> list:
    # Explicit String dtype so an all-null column stays a string column with a
    # null in it, which is what the real exports produce, rather than a Null
    # column that no string operation accepts.
    schema = dict.fromkeys(columns, pl.String)
    return pl.DataFrame(columns, schema=schema).select(expr.alias("out"))["out"].to_list()


class TestOffsetMinutes:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("-0400", -240),
            ("+0530", 330),
            ("+03:00", 180),
            ("+0000", 0),
            # A negative offset with non-zero minutes must not lose the sign on
            # the minutes part, which is the classic way to be 30 minutes off.
            ("-0930", -570),
        ],
    )
    def test_parses_known_offsets(self, text: str, expected: int) -> None:
        assert _eval(timestamps.offset_minutes(pl.col("tz")), tz=[text]) == [expected]

    @pytest.mark.parametrize("text", ["", "  ", "not an offset", "0400", "+04"])
    def test_rejects_malformed_offsets(self, text: str) -> None:
        assert _eval(timestamps.offset_minutes(pl.col("tz")), tz=[text]) == [None]

    def test_null_passes_through(self) -> None:
        assert _eval(timestamps.offset_minutes(pl.col("tz")), tz=[None]) == [None]


class TestAppleXml:
    def test_local_is_what_the_file_said(self) -> None:
        _, local = timestamps.parse_apple_xml("t")
        assert _eval(local, t=["2017-01-27 16:09:39 -0400"]) == [
            dt.datetime(2017, 1, 27, 16, 9, 39)
        ]

    def test_utc_is_shifted_by_the_offset(self) -> None:
        utc, _ = timestamps.parse_apple_xml("t")
        assert _eval(utc, t=["2017-01-27 16:09:39 -0400"]) == [
            dt.datetime(2017, 1, 27, 20, 9, 39, tzinfo=dt.UTC)
        ]

    def test_malformed_offset_yields_null_utc_but_keeps_local(self) -> None:
        utc, local = timestamps.parse_apple_xml("t")
        row = {"t": ["2017-01-27 16:09:39 wrong"]}
        assert _eval(utc, **row) == [None]
        assert _eval(local, **row) == [dt.datetime(2017, 1, 27, 16, 9, 39)]


class TestFlatCsv:
    def test_pm_time_converts_to_24_hour(self) -> None:
        _, local = timestamps.parse_flat_csv("d", "t", "tz")
        got = _eval(local, d=["2023-06-13"], t=["09:25:02 PM"], tz=["+0530"])
        assert got == [dt.datetime(2023, 6, 13, 21, 25, 2)]

    def test_midnight_is_hour_zero_not_twelve(self) -> None:
        # 12 AM is the case a naive 12-hour conversion gets wrong, and the CSV
        # has plenty of it.
        _, local = timestamps.parse_flat_csv("d", "t", "tz")
        got = _eval(local, d=["2023-01-27"], t=["12:30:00 AM"], tz=["+0530"])
        assert got == [dt.datetime(2023, 1, 27, 0, 30, 0)]

    def test_noon_stays_hour_twelve(self) -> None:
        _, local = timestamps.parse_flat_csv("d", "t", "tz")
        got = _eval(local, d=["2023-01-27"], t=["12:30:00 PM"], tz=["+0530"])
        assert got == [dt.datetime(2023, 1, 27, 12, 30, 0)]

    def test_half_hour_offset_applies_to_utc(self) -> None:
        utc, _ = timestamps.parse_flat_csv("d", "t", "tz")
        got = _eval(utc, d=["2023-06-13"], t=["09:25:02 PM"], tz=["+0530"])
        assert got == [dt.datetime(2023, 6, 13, 15, 55, 2, tzinfo=dt.UTC)]


class TestIso8601:
    def test_parses_offset_with_colon(self) -> None:
        utc, local = timestamps.parse_iso8601("t")
        row = {"t": ["2024-06-09T09:04:10+03:00"]}
        assert _eval(local, **row) == [dt.datetime(2024, 6, 9, 9, 4, 10)]
        assert _eval(utc, **row) == [dt.datetime(2024, 6, 9, 6, 4, 10, tzinfo=dt.UTC)]

    def test_z_suffix_is_utc(self) -> None:
        utc, local = timestamps.parse_iso8601("t")
        row = {"t": ["2024-06-09T09:04:10Z"]}
        assert _eval(local, **row) == [dt.datetime(2024, 6, 9, 9, 4, 10)]
        assert _eval(utc, **row) == [dt.datetime(2024, 6, 9, 9, 4, 10, tzinfo=dt.UTC)]


class TestAcrossFormats:
    def test_same_instant_from_all_three_encodings(self) -> None:
        # The whole point of the module: one moment, three spellings, one answer.
        xml_utc, _ = timestamps.parse_apple_xml("t")
        iso_utc, _ = timestamps.parse_iso8601("t")
        csv_utc, _ = timestamps.parse_flat_csv("d", "t", "tz")

        expected = dt.datetime(2024, 6, 9, 12, 0, 0, tzinfo=dt.UTC)
        assert _eval(xml_utc, t=["2024-06-09 08:00:00 -0400"]) == [expected]
        assert _eval(iso_utc, t=["2024-06-09T15:00:00+03:00"]) == [expected]
        assert _eval(csv_utc, d=["2024-06-09"], t=["05:30:00 PM"], tz=["+0530"]) == [expected]

    def test_local_day_differs_from_utc_day_near_midnight(self) -> None:
        # A +0530 subject recording just after local midnight is still on the
        # previous day in UTC. Bucketing on the instant would move the row.
        utc, local = timestamps.parse_flat_csv("d", "t", "tz")
        row = {"d": ["2023-01-27"], "t": ["01:51:00 AM"], "tz": ["+0530"]}
        assert _eval(local.dt.date(), **row) == [dt.date(2023, 1, 27)]
        assert _eval(utc.dt.date(), **row) == [dt.date(2023, 1, 26)]
