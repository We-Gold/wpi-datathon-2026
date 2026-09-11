"""Command line entry point.

    health-prep scan     write stubs for anything needing a local decision
    health-prep parse    read every input file into data/processed/

Scan comes first on a new machine. It reads the inputs, reports every source
name the built-in rules could not place, and writes stubs to the two
uncommitted files so a person can supply the pseudonyms and classify their own
device nicknames. Neither file is ever committed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from health import anonymize, pipeline

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"


def find_inputs(data_dir: Path) -> list[Path]:
    """Every input, in a stable order so subject numbering is reproducible.

    Most inputs are one file. A PMData subject is a directory of files instead,
    so this returns folders as well, and everything downstream keys on the path
    either way.
    """
    paths = [
        path
        for subdir in ("ours", "other")
        for path in sorted((data_dir / subdir).glob("*"))
        if path.is_file() and path.suffix.lower() in pipeline.FORMAT_BY_SUFFIX
    ]
    paths += [path for path in sorted((data_dir / "pmdata").glob("p[0-9][0-9]")) if path.is_dir()]
    if not paths:
        raise FileNotFoundError(
            f"no inputs under {data_dir}/ours, {data_dir}/other or {data_dir}/pmdata"
        )
    return paths


def _scan(args: argparse.Namespace) -> int:
    """Report what needs a human decision, and write stubs for it."""
    paths = find_inputs(args.data_dir)

    subject_stub: list[str] = []
    source_stub: list[str] = []

    for index, path in enumerate(paths, start=1):
        source_map = anonymize.SourceMap(anonymize.load_overrides(path))
        try:
            subject_id = anonymize.load_subject_id(path)
        except KeyError:
            subject_id = f"subject_{index:02d}"
            subject_stub.append(f'["{path}"]\nsubject_id = "{subject_id}"\n')

        for raw in pipeline.source_names(path):
            source_map.classify(raw)

        # Distinct raw strings can differ only by an instance suffix, which
        # normalization already collapses, so one entry covers both.
        unclassified = sorted({anonymize.normalize(raw) for raw in source_map.unmatched})
        print(f"{path}: {subject_id}, {len(unclassified)} unclassified sources")
        if unclassified:
            lines = "\n".join(f'"{name}" = "unknown"' for name in unclassified)
            source_stub.append(f'["{path}"]\n{lines}\n')

    _report_stub(anonymize.SUBJECTS_FILE, subject_stub, "subject ids")
    _report_stub(
        anonymize.SOURCES_FILE,
        source_stub,
        "source classifications (edit the values, they default to unknown)",
    )
    return 0


def _report_stub(path: Path, sections: list[str], what: str) -> None:
    if not sections:
        return

    body = "\n".join(sections)
    if path.exists():
        print(f"\n{path} already exists. Add these {what} to it:\n\n{body}")
        return

    path.write_text(body)
    print(f"\nWrote {path} with {what}. Fill it in, then rerun.")


def _parse(args: argparse.Namespace) -> int:
    """Read every input into data/processed, refusing to write anything unsafe."""
    paths = find_inputs(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = False
    for path in paths:
        source_map = anonymize.SourceMap(anonymize.load_overrides(path))
        try:
            subject_id = anonymize.load_subject_id(path)
        except KeyError as error:
            print(f"{path}: {error}", file=sys.stderr)
            failed = True
            continue

        frame, findings = pipeline.prepare(path, subject_id, source_map)
        if findings:
            # Nothing is written. The whole point of validating before the write
            # is that a bad run leaves no shareable file behind.
            print(f"\n{path}: refusing to write, {len(findings)} problem(s):", file=sys.stderr)
            for finding in findings:
                print(f"  [{finding.check}] {finding.detail}", file=sys.stderr)
            failed = True
            continue

        destination = args.output_dir / f"{subject_id}_{pipeline.source_format(path)}.parquet"
        frame.write_parquet(destination)

        unclassified = sorted({anonymize.normalize(raw) for raw in source_map.unmatched})
        note = f", {len(unclassified)} unclassified sources" if unclassified else ""
        print(f"{path} -> {destination} ({frame.height:,} rows{note})")

    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="health-prep", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="report sources needing a local decision")
    scan.set_defaults(func=_scan)

    parse = subparsers.add_parser("parse", help="write canonical Parquet files")
    parse.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parse.set_defaults(func=_parse)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
