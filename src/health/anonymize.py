"""Stripping identifiers, in the pipeline rather than after it.

Runs between the reader and the writer with no way to skip it, so an identifying
frame is never persisted. Getting raw data means reparsing the source.

Classification is an allowlist: the output vocabulary is fixed and anything
unrecognized becomes "unknown". A denylist of known names would fail open the
first time an unfamiliar export arrives.
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from health.schema import SOURCE_TYPES

SUBJECTS_FILE = Path("subjects.toml")
SOURCES_FILE = Path("sources.local.toml")

# Product names, safe to commit because they identify software and hardware
# rather than people. Personal device nicknames belong in the local file.
_PRODUCTS: dict[str, str] = {
    # "Google Health" is the Fitbit app's export name.
    "google health": "wearable",
    "zepp": "wearable",
    "withings": "scale",
    "renpho health": "scale",
    # GymKit is gym equipment paired through the watch, not the worn device.
    "gymkit": "app",
    "strava": "app",
    "myfitnesspal": "app",
    "waterminder": "app",
    "headspace": "app",
    "seven": "app",
    "streaks": "app",
    "snorelab": "app",
    "sleep": "app",
    "sleep++": "app",
    "sleepwatch": "app",
    "zombies, run!": "app",
    "bevel": "app",
    "spire": "app",
    "clock": "app",
    # The Health app as a source means someone typed the value in.
    "health": "manual",
    # The nested JSON's marker for rows it rolled up before export.
    "aggregated": "derived",
}

# Matched as substrings, so the name wrapped around them is discarded. That is
# what lets these rules live in version control while the names never do.
_DEVICE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("apple watch", "wearable"),
    ("iphone", "phone"),
    ("ipad", "phone"),
)

# Apple appends an instance number to some source names. Two exports of the same
# device disagree on it.
_INSTANCE_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def normalize(raw: str) -> str:
    """Fold a raw source string to the form the matching rules expect.

    NFKC matters: one export writes "Apple Watch" with a non-breaking space, so
    a plain substring test misses it and that wearable silently becomes unknown.
    """
    folded = unicodedata.normalize("NFKC", raw)
    folded = _INSTANCE_SUFFIX.sub("", folded)
    return " ".join(folded.split()).casefold()


class SourceMap:
    """Classifies raw source strings into a source type.

    Overrides come from the uncommitted local file and win over the built-in
    rules, so a nickname can be classified without ever naming it in the repo.
    """

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self._overrides = {normalize(k): v for k, v in (overrides or {}).items()}

        unknown_types = set(self._overrides.values()) - set(SOURCE_TYPES)
        if unknown_types:
            raise ValueError(
                f"{SOURCES_FILE} uses source types that are not in the schema: "
                f"{sorted(unknown_types)}. Valid types: {SOURCE_TYPES}"
            )

        # What the scan command turns into a stub for someone to fill in.
        self.unmatched: set[str] = set()

    def classify(self, raw: str | None) -> str:
        if raw is None:
            return "unknown"

        folded = normalize(raw)
        if not folded:
            return "unknown"

        if folded in self._overrides:
            return self._overrides[folded]

        # Before the product table, so a nickname containing a product word
        # still resolves to the device.
        for needle, source_type in _DEVICE_PATTERNS:
            if needle in folded:
                return source_type

        if folded in _PRODUCTS:
            return _PRODUCTS[folded]

        self.unmatched.add(raw)
        return "unknown"

    def mapping(self, values: Iterable[str | None]) -> dict[str, str]:
        """A raw-to-type lookup for the distinct values in a column."""
        return {value: self.classify(value) for value in values if value is not None}


def source_type_expr(column: str, mapping: dict[str, str]) -> pl.Expr:
    """Replace a raw source column with its source type, dropping the original."""
    return (
        pl.col(column)
        .replace_strict(mapping, default="unknown", return_dtype=pl.String)
        .alias("source_type")
    )


def _load_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_overrides(source: Path, path: Path = SOURCES_FILE) -> dict[str, str]:
    """Source-name overrides for one input file.

    Keyed by source file, because nicknames are not unique across people.
    """
    table = _load_table(path)
    section = table.get(str(source), {})
    return {k: v for k, v in section.items() if isinstance(v, str)}


def load_subject_id(source: Path, path: Path = SUBJECTS_FILE) -> str:
    """The pseudonym for one input file.

    An explicit mapping, not derived from the filename (those are given names)
    and not hashed (the input space is a handful of guessable strings).
    """
    table = _load_table(path)
    section = table.get(str(source))
    if not section or "subject_id" not in section:
        raise KeyError(
            f"no subject_id for {source} in {path}. "
            f"Run `health-prep scan` to write a stub, then fill it in."
        )
    return str(section["subject_id"])
