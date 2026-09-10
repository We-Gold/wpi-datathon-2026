# wpi-datathon-2026

Parses Apple Health exports and two related formats into one consistent schema.

## Setup

```sh
uv sync --all-groups
```

## Adding data

```
data/ours/     Apple Health export.xml files, one per person
data/other/    the other formats, .csv and .json
```

All of `data/` is gitignored.

## Running

```sh
uv run health-prep scan     # first time, and after adding a file
uv run health-prep parse
```

`scan` writes `subjects.toml` and `sources.local.toml` if they are missing. Fill
in any blanks it reports, then run `parse`.

`parse` writes one Parquet file per input to `data/processed/`, named
`<subject>_<format>.parquet`. If a file fails validation nothing is written for
it.

## The two local files

Both are gitignored. `scan` writes them if missing; these are the shapes.

`subjects.toml`, one section per input file:

```toml
["data/ours/alice.xml"]
subject_id = "subject_01"

["data/ours/bob.xml"]
subject_id = "subject_02"

["data/other/health_data.csv"]
subject_id = "subject_03"
```

`sources.local.toml`, for source names the built-in rules could not place:

```toml
["data/ours/alice.xml"]
"my phone nickname" = "phone"
"some fitness app" = "app"
"old fitness band" = "wearable"
```

Keys are lowercased and have any `(1234)` suffix stripped, which is how `scan`
writes them. Values must be one of `wearable`, `phone`, `scale`, `app`,
`manual`, `derived`, `unknown`.

## Output

One row per observation:

| column | notes |
|---|---|
| `subject_id`, `source_format` | |
| `metric` | full HealthKit identifier |
| `unit` | canonical, see below |
| `value_num`, `value_str` | exactly one is set per row |
| | `value_str` is always the full `HKCategoryValue...` name |
| `start_utc`, `end_utc`, `created_utc` | |
| `start_local`, `end_local` | wall clock, use this for daily buckets |
| `source_type` | `wearable`, `phone`, `scale`, `app`, `manual`, `derived`, `unknown` |

Units are converted on the way in, so the recorded unit is not kept. Length is
`m`, speed `m/s`, duration `s`, mass `kg` for body mass and `g`/`mg` for
nutrients, energy `kcal`, and percentages are `fraction` with decimal values
between 0 and 1. Anything else keeps the unit it was recorded in. A unit the
table in `health/units.py` has no rule for stops the run.

Not done yet: merging, deduplication, workouts.

## Pipeline order

`health/pipeline.py` runs the steps and is the place to look for the order they
happen in. Briefly: read, anonymize, canonicalize units, validate. The caller
writes only when validation returns nothing.

## Notebook

```sh
uv run marimo edit --watch --no-token notebooks/explore.py
```

A scratch pad over `data/processed/`. It imports from `health`, so editing a
reader and rerunning a cell picks the change up. marimo files store no cell
output, so nothing from the data lands in git.

`--watch` picks up edits to the notebook file made outside the browser. Cells
go stale rather than rerunning on their own, so click the run button to apply
them.

## Development

```sh
uv run pytest
uv run ruff check src tests notebooks
uv run pyrefly check
```
