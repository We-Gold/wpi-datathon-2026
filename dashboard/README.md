# Dashboard

Prototype front end. Vite, React, d3, TypeScript, formatted and linted with Biome.

```sh
npm install
npm run dev
```

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
  landscape built from the person's clusters. Healthy clusters are orange peaks
  and unhealthy clusters are slate valleys, and a tighter cluster makes a taller
  peak. On the map are the last 90 days as a fading trail, today as a ringed
  point with its velocity arrow, and the predicted next 30 days as a dotted line.
- The **Daily** section has today's velocity on the same axes, the factors that
  made it up, and the daily cards.

The server does all of the math: positions, velocities, clusters and their
labels, and the transformer prediction. The dashboard only draws the result.

## Where things are

- `src/config.ts` holds every name a person reads: the app name, the axis and
  position names, the factor labels, and the section names. Data uses fixed keys,
  so renaming a label never touches data. `index.html` gets the app name through
  a small plugin in `vite.config.ts`.
- `src/types.ts` has `TrajectoryResponse`, the shape the server will return for
  one subject, and the metric types that mirror the Python output.
- `src/data/source.ts` is the only place data is loaded. Both functions return
  seeded mock data for now and become fetches against the FastAPI server later.
- `src/lib/terrain.ts` turns clusters into heights and colors.
- `src/components/TerrainMap.tsx` is the map. `VelocityPlot.tsx` and
  `ContributionBars.tsx` are the daily view. React renders the SVG and d3 does
  scales, contours, and paths.

The page is light only on purpose, since the map look depends on the paper
colored background.
