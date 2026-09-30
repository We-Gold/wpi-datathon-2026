## Data TOML files

These files are gitignored, and if you do create them,, they go in the root of the project.

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