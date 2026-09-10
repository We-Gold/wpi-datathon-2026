"""Reader for the Apple Health export.xml.

These files run to gigabytes. The document is read in bounded chunks and
records are pulled out of the raw bytes with a regex, which is about twice as
fast as building elements. The regex only understands the shape Apple's
exporter actually writes, so anything it cannot account for falls back to
iterparse rather than silently dropping rows.

Records nested in a <Correlation> also appear at top level, per the DTD, so
reading every Record would count them twice.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from html import unescape
from pathlib import Path

import polars as pl

from health import anonymize, metrics, timestamps
from health.schema import conform, split_value

# sourceVersion is deliberately absent: version histories fingerprint a device
# and carry nothing worth analyzing.
_ATTRIBUTES = (
    "type",
    "unit",
    "value",
    "sourceName",
    "creationDate",
    "startDate",
    "endDate",
)

# Large enough that per-batch overhead disappears, small enough that the raw
# string columns for one batch stay well under a gigabyte.
BATCH_SIZE = 1_000_000

# Read size for the byte scanner. Big enough that the per-chunk seam work
# disappears, small enough that a gigabyte export never lands in memory whole.
_READ_CHUNK = 1 << 24

# Attribute values are matched as whole quoted spans rather than "everything up
# to the next >". The device attribute embeds a > in its value, and a tag
# pattern that stopped there would cut records in half.
_TAG = re.compile(rb'<(Record|Correlation|/Correlation)((?:\s+[\w:.-]+="[^"]*")*)\s*/?>')
_ATTR = re.compile(rb'([\w:.-]+)="([^"]*)"')

# Counted per chunk and checked against what _TAG matched. Disagreement means
# the file is shaped in some way the pattern does not cover, which is the
# signal to hand the whole read back to iterparse.
_RECORD_OPEN = b"<Record"

_SOURCE_NAME = re.compile(rb'sourceName="([^"]*)"')

_TEXT_SCHEMA = dict.fromkeys(_ATTRIBUTES, pl.String)


class _UnexpectedShape(Exception):
    """The byte scanner found something it is not equipped to read."""


def _chunks(path: Path) -> Iterator[bytes]:
    """The file in pieces that never end mid-tag.

    Each piece is cut at the last complete tag, and the remainder is carried
    into the next one, so no attribute is ever split across a boundary.
    """
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            buffer = carry + chunk
            cut = buffer.rfind(b">") + 1
            if cut == 0:
                carry = buffer
                continue
            yield buffer[:cut]
            carry = buffer[cut:]

    if carry:
        yield carry


def source_names(path: Path) -> set[str]:
    """Distinct sourceName values, without building a single element.

    Picks up names on Workout and Correlation elements too, which the record
    reader skips. For classification that is a wider net, not a wrong one.
    """
    found: set[bytes] = set()
    for buffer in _chunks(path):
        found.update(match.group(1) for match in _SOURCE_NAME.finditer(buffer))

    # html.unescape rather than the five XML entities, because the export
    # writes curly quotes in device names as numeric character references.
    return {unescape(name.decode("utf-8")) for name in found}


def _scan_frames(path: Path, batch_size: int) -> Iterator[pl.DataFrame]:
    """Yield columnar batches of raw Record attributes, read from the bytes.

    Columns of lists rather than a list of dicts, which for millions of rows is
    the difference between comfortable and out of memory. Values stay as bytes;
    polars decodes them in one go, far cheaper than a decode call per value.
    """
    columns: dict[str, list[bytes | None]] = {name: [] for name in _ATTRIBUTES}
    count = 0
    correlation_depth = 0

    for buffer in _chunks(path):
        matched = 0
        for match in _TAG.finditer(buffer):
            tag = match.group(1)

            if tag == b"Correlation":
                correlation_depth += 1
                continue
            if tag != b"Record":
                correlation_depth -= 1
                continue

            matched += 1
            if correlation_depth:
                continue

            attributes = dict(_ATTR.findall(match.group(2)))
            for name in _ATTRIBUTES:
                columns[name].append(attributes.get(name.encode()))
            count += 1

        if matched != buffer.count(_RECORD_OPEN):
            raise _UnexpectedShape(f"{path}: record tags the byte scanner cannot read")

        if count >= batch_size:
            yield _decoded_frame(columns)
            columns = {name: [] for name in _ATTRIBUTES}
            count = 0

    if count:
        yield _decoded_frame(columns)


def _iterparse_frames(path: Path, batch_size: int) -> Iterator[pl.DataFrame]:
    """The same batches, built through the XML parser.

    Slower, but it handles anything well formed. Used when the byte scanner
    backs out.
    """
    columns: dict[str, list[str | None]] = {name: [] for name in _ATTRIBUTES}
    count = 0

    # Start events too, so a Record inside a Correlation can be recognized.
    context = ET.iterparse(path, events=("start", "end"))
    _, root = next(context)

    correlation_depth = 0

    for event, element in context:
        if element.tag == "Correlation":
            if event == "start":
                correlation_depth += 1
            else:
                correlation_depth -= 1
                element.clear()
            continue

        if event != "end":
            continue

        if element.tag == "Record":
            if correlation_depth == 0:
                for name in _ATTRIBUTES:
                    columns[name].append(element.get(name))
                count += 1

            element.clear()

            # The parser links finished elements under the root, so clearing
            # the element alone does not keep memory flat. Only safe outside a
            # correlation, whose children are still being walked.
            if correlation_depth == 0:
                root.clear()

            if count >= batch_size:
                yield pl.DataFrame(columns, schema=_TEXT_SCHEMA)
                columns = {name: [] for name in _ATTRIBUTES}
                count = 0
        else:
            # Out of scope, but still has to be released or it accumulates.
            element.clear()

    if count:
        yield pl.DataFrame(columns, schema=_TEXT_SCHEMA)


def _decode(name: str, values: list[bytes | None]) -> pl.Series:
    """One column of raw attribute bytes as text, entities resolved.

    The parser resolves entity references itself, so the byte scanner has to.
    Checking for an ampersand first keeps the slow per-value path off the
    columns that never contain one, which in practice is all of them.
    """
    column = pl.Series(name, values, dtype=pl.Binary).cast(pl.String)
    if column.str.contains("&", literal=True).any():
        return column.map_elements(unescape, return_dtype=pl.String)
    return column


def _decoded_frame(columns: dict[str, list[bytes | None]]) -> pl.DataFrame:
    """One batch of raw attribute bytes as a frame of text columns."""
    return pl.DataFrame([_decode(name, values) for name, values in columns.items()])


def _to_canonical(
    raw: pl.DataFrame, subject_id: str, source_map: anonymize.SourceMap
) -> pl.DataFrame:
    """Turn one batch of raw attributes into the canonical schema."""
    mapping = source_map.mapping(raw["sourceName"].unique().to_list())

    start_utc, start_local = timestamps.parse_apple_xml("startDate")
    end_utc, end_local = timestamps.parse_apple_xml("endDate")
    created_utc, _ = timestamps.parse_apple_xml("creationDate")

    value_num, value_str = split_value(pl.col("value"))

    return conform(
        raw.lazy()
        .select(
            pl.lit(subject_id).alias("subject_id"),
            pl.lit("apple_xml").alias("source_format"),
            metrics.restore_prefix(pl.col("type")).alias("metric"),
            pl.col("unit").alias("unit"),
            value_num,
            value_str,
            start_utc.alias("start_utc"),
            end_utc.alias("end_utc"),
            created_utc.alias("created_utc"),
            start_local.alias("start_local"),
            end_local.alias("end_local"),
            anonymize.source_type_expr("sourceName", mapping),
        )
        .collect()
    )


def read(
    path: Path,
    subject_id: str,
    source_map: anonymize.SourceMap,
    batch_size: int = BATCH_SIZE,
) -> pl.DataFrame:
    """Read one Apple Health export into the canonical schema."""
    try:
        frames = _read_with(_scan_frames, path, subject_id, source_map, batch_size)
    except (_UnexpectedShape, MemoryError):
        # Anything the fast path will not vouch for is read properly instead.
        # Classification state is per attempt, so it is rebuilt from scratch.
        source_map.unmatched.clear()
        frames = _read_with(_iterparse_frames, path, subject_id, source_map, batch_size)

    if not frames:
        raise ValueError(f"{path} contains no Record elements")

    return pl.concat(frames, rechunk=True)


def _read_with(
    batches: Callable[[Path, int], Iterator[pl.DataFrame]],
    path: Path,
    subject_id: str,
    source_map: anonymize.SourceMap,
    batch_size: int,
) -> list[pl.DataFrame]:
    return [_to_canonical(raw, subject_id, source_map) for raw in batches(path, batch_size)]
