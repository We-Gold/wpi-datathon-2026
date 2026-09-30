# Dashboard

Prototype front end. Vite, React, d3, TypeScript, formatted and linted with Biome.

```sh
npm install
npm run dev
```

The dashboard reads JSON from `public/data/`, which holds health data and is
gitignored. Build it from the repository root with
`uv run health-model && uv run health-export`.

Run this after every chunk of work:

```sh
npm run quality-check   # biome check, then tsc -b
```

`npm run check` applies Biome's safe fixes and formatting.

## The idea

Fitness is drawn as a place on a map with two axes, activity and recovery.
Each day gives a velocity on both axes. The position on the activity axis is
fitness, and the position on the recovery axis is reserve.

- The **Aggregate** section has the map and the trend cards. The map is a
  landscape built from the person's clusters. Healthy clusters are blue peaks
  and unhealthy clusters are red valleys, and a tighter cluster makes a taller
  peak. Places are named against the person's usual level. On the map are the
  last 180 days as a fading trail, today as a ringed point with its velocity
  arrow, and the predicted next 30 days as a dotted line. "Today" is the last
  day with data for that subject. The view fits the paths (trail, forecast, and
  any what-if path), not the clusters, so terrain can run past the edges, where
  it fades out. The arrow shows direction; its length compares today with the
  subject's own typical day, within fixed screen limits.
- The **Daily** section has today's velocity on the same axes, the factors that
  made it up, and the daily cards.

`health-export` in `src/health/` does the math. A day's axis score is the
activity or recovery score from `add_interpretable_axes`. Position is a moving
average of it with a 7-day half-life, starting at the person's usual level, and
velocity is one day's change in position. Velocity splits exactly into factor
contributions plus a pull back toward the usual level. Clusters are a Gaussian
mixture over positions, healthy when activity plus recovery at the center is at
least zero. The prediction is a Ridge model per horizon, with an 80% range.

The what-if presets are the one piece of math done in the browser. A preset
changes raw features by fixed amounts (`src/lib/scenarios.ts`). The Python
score formula turns that into a score change, and the moving-average rule turns
it into a change in position, once or every day. This is added to the forecast;
it is not a new forecast.

## Where things are

- `src/config.ts` holds every name a person reads: the app name, the axis and
  position names, the factor labels, and the section names. Data uses fixed keys,
  so renaming a label never touches data. `index.html` gets the app name through
  a small plugin in `vite.config.ts`.
- `src/types.ts` has `DashboardIndex` and `SubjectData`, the shapes of the
  exported JSON, and the metric types that mirror the Python output.
- `src/data/source.ts` is the only place data is loaded.
- `src/lib/terrain.ts` turns clusters into heights and colors.
- `src/components/TerrainMap.tsx` is the map. `VelocityPlot.tsx` and
  `ContributionBars.tsx` are the daily view. React renders the SVG and d3 does
  scales, contours, and paths.

The page is light only on purpose, since the map look depends on the paper
colored background.
