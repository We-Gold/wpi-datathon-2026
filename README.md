# wpi-datathon-2026

Parses Apple Health exports, two related formats, and the PMData dataset into
one consistent schema.

## Setup

```sh
uv sync --all-groups
```

## Adding data

```
data/ours/     Apple Health export.xml files, one per person
data/other/    the other formats, .csv and .json
data/pmdata/   the PMData dataset, one p01..p16 folder per participant
```

All of `data/` is gitignored.

The first three formats are one file per person. A PMData subject is a folder of
seventeen files instead, so `data/pmdata/p01` is one input the same way
`data/ours/alice.xml` is. Everything downstream keys on the path either way.

## Running

```sh
uv run health-prep scan     # first time, and after adding a file
uv run health-prep parse
```

`scan` writes `subjects.toml` and `sources.local.toml` if they are missing. Fill
in any blanks it reports, then run `parse`.

`parse` writes one Parquet file per input to `data/processed/`, named
`<subject>_<format>.parquet`. If an input fails validation nothing is written
for it.

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

["data/pmdata/p01"]
subject_id = "subject_04"
```

`sources.local.toml`, for source names the built-in rules could not place:

```toml
["data/ours/alice.xml"]
"my phone nickname" = "phone"
"some fitness app" = "app"
"old fitness band" = "wearable"
```

PMData needs no entry in `sources.local.toml`. It holds no device names, so
there is nothing to classify. The folder a file came from says what recorded it:
`fitbit` is a wearable, `pmsys` and `googledocs` are what a person typed.

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

Metric names follow one of two rules, written out at the top of
`health/metrics.py`:

1. HealthKit has a name for it, so use that name exactly as HealthKit spells it.
   That includes odd spellings: the Apple export writes
   `HKDataTypeSleepDurationGoal`, which is not one of the `TypeIdentifier`
   forms, and it is kept as is. Two sources measuring the same thing have to
   land on the same string or a group-by splits them in two.
2. HealthKit has no name for it, so use a plain name that says what was
   measured: `TotalEnergyBurned`, `SedentaryTime`, `Mood`, `PerceivedExertion`.

Names carry no vendor and no unit. Which device or app a row came from is
already in `source_type` and `source_format`, and a vendor prefix would be wrong
as often as right, since PMData's step counts come off a Fitbit but are ordinary
HealthKit step counts. Units are converted on the way in, so a name like
`SedentaryMinutes` would go stale the moment the value became seconds.

The one place this leaks is rule 1. Three HealthKit names end in `Percentage`
but hold fractions between 0 and 1, because Apple records them that way:
`BodyFatPercentage` runs 0.051 to 0.12, not 5.1 to 12. Renaming them would stop
them matching the same metric from another export, so they keep Apple's spelling
and the catalogue carries the real unit.

## The metric catalogue

`CATALOGUE` in `health/metrics.py` describes all 81 metrics: the domain each one
belongs to, the canonical unit it is stored in, how it collapses into one number
a day, and whether it is a subjective rating rather than a measurement.

Read it instead of matching on name prefixes. The HealthKit names cannot be
grouped by prefix anyway: `HeartRate`, `RestingHeartRate`,
`HeartRateVariabilitySDNN`, `RespiratoryRate` and `VO2Max` are all cardiac and
share no common string, and we do not get to rename them.

```python
from health import metrics

metrics.in_domain("cardiac")     # every cardiac metric
metrics.subjective_metrics()     # the self ratings, where a difference of one
                                 # does not mean the same at both ends
metrics.by_daily_rule()          # grouped by sum, mean, duration, count, skip
```

The eleven domains are activity, body, cardiac, energy, exposure, hygiene,
mobility, nutrition, sleep, wellbeing and workout.

Not done yet: merging, deduplication, workouts.

## PMData timestamps

PMData records local wall clock time and never writes an offset. Some of its
files add a `Z` suffix, which normally means UTC, but it is not true: the same
sleep sessions appear in `sleep.json` without the suffix and in
`sleep_score.csv` with it, and the two agree to the second.

So `start_utc` for PMData comes from a fixed offset, `PMDATA_OFFSET_MINUTES` in
`health/readers/pmdata.py`, set to 60. That is supplied, not something the files
said. `start_local` is exact either way, so prefer it for anything grouped by
day.

Fixed rather than a named zone because the clock that wrote these files did not
follow daylight saving. Oslo jumped from 02:00 to 03:00 on 2020-03-29, and the
devices kept recording through that hour at their usual rate, which only happens
if the clock never moved. Using `Europe/Oslo` would throw those rows away and
shift the last three days of the window by an hour.

Which fixed offset is a weaker claim. The timestamps could be Norwegian winter
time or plain UTC and nothing in the dataset settles it. Bedtimes peaking at
23:00 and morning wellness reports at 08:00 both read as local, which is why it
is 60. It is one constant if that is ever settled.

## Pipeline order

`health/pipeline.py` runs the steps and is the place to look for the order they
happen in. Briefly: read, anonymize, canonicalize units, validate. The caller
writes only when validation returns nothing.

## Notebook

```sh
uv run marimo edit --watch --no-token notebooks/explore.py
uv run marimo edit --watch --no-token notebooks/pmdata.py
uv run marimo edit --watch --no-token notebooks/embeddings.py
```

`explore.py` is a scratch pad over `data/processed/`. `pmdata.py` is a survey of
the PMData dataset on disk, what it holds and how it lines up with the rest, and
it reads `data/pmdata/` directly rather than the parsed output.
`embeddings.py` visualizes the artifacts written by `health-model`, including
the learned trajectories, interpretable axes, coverage, PCA loadings, and
held-out reconstruction errors.

Both import from `health`, so editing a reader and rerunning a cell picks the
change up. marimo files store no cell output, so nothing from the data lands in
git.

`--watch` picks up edits to the notebook file made outside the browser. Cells
go stale rather than rerunning on their own, so click the run button to apply
them.

## Daily features and embeddings

The first modeling baseline is reproducible from the canonical Parquet files:

```sh
uv sync --all-groups
uv run health-model
```

It chooses a feature-rich interval per subject, builds daily and heart-rate
spectral features, adds interpretable relative-to-self activity and recovery
scores, and compares PCA with a small denoising autoencoder on a temporal
holdout. Outputs go to the gitignored `data/model/` directory.

The feature, missing-value, timeframe, axis, and evaluation decisions are in
[`docs/embedding-baseline.md`](docs/embedding-baseline.md). Read that before
interpreting the scores: they describe patterns in these records, not clinical
health or causal recommendations.

## Development

```sh
uv run pytest
uv run ruff check src tests
uv run pyrefly check
```

`notebooks/` is excluded from both ruff and pyrefly. marimo rewrites those files
itself and lays cells out its own way, so the formatter and the notebook take
turns undoing each other. Nothing in there ships.
