"""The order the preparation steps run in.

One input becomes one shareable table by running these, in this order:

    1. read        parse the source format into the canonical schema
    2. anonymize   replace source names with source types
    3. units       convert every value to its canonical unit
    4. validate    check the schema and refuse anything identifying

Step 2 is not a separate call. Each reader classifies as it builds the frame,
so a frame holding raw source names never exists to be written by mistake.

Step 4 returns findings rather than raising, so one bad file reports everything
wrong with it instead of only the first problem. Writing is the caller's job
and only happens when there are no findings.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from health import anonymize, units, validate
from health.readers import apple_xml, flat_csv, nested_json, pmdata

# Which reader handles which format.
READERS = {
    "apple_xml": apple_xml.read,
    "flat_csv": flat_csv.read,
    "nested_json": nested_json.read,
    "pmdata": pmdata.read,
}

# Scan only needs the source names, so it uses these instead of the full
# reader. Reading a gigabyte export the cheap way takes a second; parsing it
# into the canonical schema takes half a minute, and scan throws that away.
SOURCE_NAMES = {
    "apple_xml": apple_xml.source_names,
    "flat_csv": flat_csv.source_names,
    "nested_json": nested_json.source_names,
    "pmdata": pmdata.source_names,
}

# The two "other" formats are told apart by extension alone, which is enough for
# the files at hand.
FORMAT_BY_SUFFIX = {
    ".xml": "apple_xml",
    ".csv": "flat_csv",
    ".json": "nested_json",
}


def source_format(path: Path) -> str:
    """Which format one input is in.

    A directory is a PMData subject. PMData holds .json and .csv files of its
    own, so extension alone cannot tell them apart from the other two formats,
    and the input being a folder is what settles it.
    """
    if path.is_dir():
        return "pmdata"

    suffix = path.suffix.lower()
    if suffix not in FORMAT_BY_SUFFIX:
        raise ValueError(f"no reader for {path}. Known extensions: {sorted(FORMAT_BY_SUFFIX)}")
    return FORMAT_BY_SUFFIX[suffix]


def prepare(
    path: Path, subject_id: str, source_map: anonymize.SourceMap
) -> tuple[pl.DataFrame, list[validate.Finding]]:
    """Run every step on one input. See the module docstring for the order."""
    frame = READERS[source_format(path)](path, subject_id, source_map)
    frame = units.canonicalize(frame)
    return frame, validate.check(frame)


def source_names(path: Path) -> set[str]:
    """Every raw source name in one input, without a full parse."""
    return SOURCE_NAMES[source_format(path)](path)
