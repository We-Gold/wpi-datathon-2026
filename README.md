# WPI Datathon 2026

Thanks to the Data Science club for hosting this inaugural event!

In this project, we explored the use of wearable fitness tracker data for short-term health forecasting.

Read more about the feature engineering and modeling process in the [Notebooks](#notebooks) section below, or check out our [final presentation](/assets/Datathon%202026.pdf). 

We created a dashboard to visualize our findings interactively:

![dashboard](/assets/dashboard-overview.png)

We also introduced a feature to see how a change to your routine can impact long term health:

| Additional 45 min workout | Additional late nights out |
| --- | --- |
| ![dashboard-positive](/assets/dashboard-positive.png) | ![dashboard-positive](/assets/dashboard-negative.png) |

## Setup

```bash
uv sync --all-groups
```

## Adding data

```
data/ours/     Apple Health export.xml files
data/other/    the other formats, .csv and .json
data/pmdata/   the PMData dataset, one p01..p16 folder per participant
```

All of `data/` is gitignored to maintain the privacy of the health data. 

Please contact me with any questions about the data.

## Data Preprocessing

```bash
uv run health-prep scan    
uv run health-prep parse
```

## Notebooks

```bash
uv run marimo edit notebooks/explore.py
uv run marimo edit notebooks/pmdata.py
uv run marimo edit notebooks/embeddings.py
```

## Daily Features, Embeddings, and Modeling

```bash
uv run health-model
uv run health-forecast
```

## Dashboard Data Export

```bash
uv run health-export
```

Run the dashboard with:

```bash
cd dashboard
npm run dev
```

**You MUST run the data export first before that, which requires running all prior steps.**

## Development

```sh
uv run pytest
uv run ruff check src tests
uv run pyrefly check
```