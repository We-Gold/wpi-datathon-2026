"""The order the preparation steps run in.

One input file becomes one shareable table by running these, in this order:

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
from health.readers import apple_xml, flat_csv, nested_json

# Which reader handles which file, by extension. The two "other" formats are
# distinguished by extension alone, which is enough for the files at hand.
READERS = {
    ".xml": apple_xml.read,
    ".csv": flat_csv.read,
    ".json": nested_json.read,
}

# Scan only needs the source names, so it uses these instead of the full
# reader. Reading a gigabyte export the cheap way takes a second; parsing it
# into the canonical schema takes half a minute, and scan throws that away.
SOURCE_NAMES = {
    ".xml": apple_xml.source_names,
    ".csv": flat_csv.source_names,
    ".json": nested_json.source_names,
}


def prepare(
    path: Path, subject_id: str, source_map: anonymize.SourceMap
) -> tuple[pl.DataFrame, list[validate.Finding]]:
    """Run every step on one input file. See the module docstring for the order."""
    frame = READERS[path.suffix.lower()](path, subject_id, source_map)
    frame = units.canonicalize(frame)
    return frame, validate.check(frame)


def source_names(path: Path) -> set[str]:
    """Every raw source name in one input file, without a full parse."""
    return SOURCE_NAMES[path.suffix.lower()](path)
