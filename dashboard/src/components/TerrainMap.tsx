import { contours, Delaunay, format, geoIdentity, geoPath, max, scaleLinear, timeFormat } from "d3";
import { type PointerEvent, useMemo, useState } from "react";
import {
	AXIS_LABELS,
	CLUSTER_LABELS,
	HISTORY_DAYS,
	PREDICTION_DAYS,
	VELOCITY_ARROW_DAYS,
	WHAT_IF_LABELS,
} from "../config";
import { parseDay } from "../lib/stats";
import { regionName } from "../lib/story";
import { heightAt, TERRAIN_STOPS, terrainColor } from "../lib/terrain";
import { useElementWidth } from "../lib/useElementWidth";
import type { AxisVector, PredictedDay, TrajectoryResponse } from "../types";

interface Props {
	data: TrajectoryResponse;
	/** A simulated path from today, drawn apart from the model's prediction. */
	whatIf?: PredictedDay[];
}

const MARGIN = { top: 28, right: 16, bottom: 36, left: 40 };
/** Pixel size of one cell in the height grid. Smaller is smoother and slower. */
const CELL = 5;
/** Contour levels on each side of neutral ground. */
const LEVELS = 9;
/**
 * Extra cells sampled past each edge. Contour polygons close along the grid
 * edge, so without this their stroke draws a border around the map.
 */
const EDGE_CELLS = 2;
/**
 * How much wider than tall one data unit may be drawn. A little stretch lets
 * the terrain fill a wide page instead of leaving empty paper at the sides.
 */
const MAX_STRETCH = 1.4;
/** Named places on the map, most prominent first. */
const PEAK_LABELS = 3;
const VALLEY_LABELS = 2;
/** Room a place name needs, so names never overlap each other or the path ends. */
const LABEL_BOX = { width: 130, height: 26 };
const formatDate = timeFormat("%a %b %-d, %Y");
const formatShortDate = timeFormat("%b %-d");
const formatPosition = format("+.2f");
const formatTick = format("~g");

interface MarkPoint {
	date: string;
	position: AxisVector;
	kind: "past" | "predicted" | "whatIf";
	px: number;
	py: number;
}

const KIND_NOTE: Record<MarkPoint["kind"], string> = {
	past: "",
	predicted: " · predicted",
	whatIf: ` · ${WHAT_IF_LABELS.tag.toLowerCase()}`,
};

/**
 * The main view. The landscape comes from the subject's clusters, the trail is
 * where they have been, the dotted line is where the model expects them to go.
 */
export function TerrainMap({ data, whatIf }: Props) {
	const [containerRef, width] = useElementWidth<HTMLDivElement>();
	const [hovered, setHovered] = useState<MarkPoint | null>(null);

	const height = Math.round(Math.min(620, Math.max(340, width * 0.66)));
	const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
	const innerHeight = height - MARGIN.top - MARGIN.bottom;

	const map = useMemo(() => {
		if (innerWidth <= 0) return null;
		const { history, prediction, clusters } = data;
		const today = history.at(-1);
		if (!today) return null;

		// Fit the trail, the prediction, and the body of every cluster.
		const xs: number[] = [];
		const ys: number[] = [];
		for (const d of [...history, ...prediction]) {
			xs.push(d.position.activity);
			ys.push(d.position.recovery);
		}
		for (const c of clusters) {
			const sx = 2 * Math.sqrt(c.covariance[0][0]);
			const sy = 2 * Math.sqrt(c.covariance[1][1]);
			xs.push(c.center.activity - sx, c.center.activity + sx);
			ys.push(c.center.recovery - sy, c.center.recovery + sy);
		}
		let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
		let [y0, y1] = [Math.min(...ys), Math.max(...ys)];

		// Nearly the same data units per pixel on both axes, so peaks keep their
		// shape. The slack axis may stretch a little to use the space.
		const fitX = ((x1 - x0) / innerWidth) * 1.05;
		const fitY = ((y1 - y0) / innerHeight) * 1.05;
		const unitsPerPx = Math.max(fitX, fitY);
		const unitsPerPxX = fitX < unitsPerPx ? Math.max(fitX, unitsPerPx / MAX_STRETCH) : unitsPerPx;
		const unitsPerPxY = fitY < unitsPerPx ? Math.max(fitY, unitsPerPx / MAX_STRETCH) : unitsPerPx;
		const cx = (x0 + x1) / 2;
		const cy = (y0 + y1) / 2;
		x0 = cx - (unitsPerPxX * innerWidth) / 2;
		x1 = cx + (unitsPerPxX * innerWidth) / 2;
		y0 = cy - (unitsPerPxY * innerHeight) / 2;
		y1 = cy + (unitsPerPxY * innerHeight) / 2;
		const x = scaleLinear().domain([x0, x1]).range([0, innerWidth]);
		const y = scaleLinear().domain([y0, y1]).range([innerHeight, 0]);

		// Sample the height at cell centers, top row first.
		const nx = Math.ceil(innerWidth / CELL) + 2 * EDGE_CELLS;
		const ny = Math.ceil(innerHeight / CELL) + 2 * EDGE_CELLS;
		const values = new Float64Array(nx * ny);
		for (let j = 0; j < ny; j++) {
			const r = y.invert((j - EDGE_CELLS + 0.5) * CELL);
			for (let i = 0; i < nx; i++) {
				values[j * nx + i] = heightAt(clusters, x.invert((i - EDGE_CELLS + 0.5) * CELL), r);
			}
		}
		const maxAbs = max(values, (v) => Math.abs(v)) || 1;
		const step = maxAbs / LEVELS;
		// Neutral ground is the page itself, one band wide around zero. Peaks and
		// valleys are traced apart so no band covers the whole grid. Stacked full
		// size bands anti-alias into a visible line along the clip edge.
		const thresholds = Array.from({ length: LEVELS }, (_, k) => (k + 0.5) * step);
		const color = terrainColor(maxAbs);
		const path = geoPath(
			geoIdentity()
				.scale(CELL)
				.translate([-EDGE_CELLS * CELL, -EDGE_CELLS * CELL]),
		);
		const trace = (grid: number[], sign: 1 | -1) =>
			contours()
				.size([nx, ny])
				.thresholds(thresholds)(grid)
				.map((c) => ({
					value: sign * c.value,
					d: path(c) ?? "",
					fill: color(sign * (c.value + step / 2)),
				}));
		const bands = [
			...trace(
				Array.from(values, (v) => -v),
				-1,
			),
			...trace(Array.from(values), 1),
		];

		const toMark = (d: PredictedDay, kind: MarkPoint["kind"]): MarkPoint => ({
			date: d.date,
			position: d.position,
			kind,
			px: x(d.position.activity),
			py: y(d.position.recovery),
		});
		const past = history.map((d) => toMark(d, "past"));
		const future = [toMark(today, "predicted"), ...prediction.map((d) => toMark(d, "predicted"))];

		// Name the most prominent peaks and valleys, skipping any that would
		// crowd another name, the path ends, or repeat a name already shown.
		const taken = [past[past.length - 1], future[future.length - 1]].map((m) => ({
			x: m.px,
			y: m.py,
		}));
		const clear = (px: number, py: number) =>
			px > LABEL_BOX.width / 2 &&
			px < innerWidth - LABEL_BOX.width / 2 &&
			py > LABEL_BOX.height &&
			py < innerHeight - LABEL_BOX.height &&
			taken.every(
				(t) => Math.abs(t.x - px) > LABEL_BOX.width || Math.abs(t.y - py) > LABEL_BOX.height,
			);
		const byProminence = clusters
			.map((c) => ({ c, h: heightAt(clusters, c.center.activity, c.center.recovery) }))
			.sort((a, b) => Math.abs(b.h) - Math.abs(a.h));
		const names = new Set<string>();
		const places: { name: string; px: number; py: number; kind: "peak" | "valley" }[] = [];
		const quota = { peak: PEAK_LABELS, valley: VALLEY_LABELS };
		for (const { c, h } of byProminence) {
			const kind = h > 0 ? "peak" : "valley";
			const name = regionName(c.center);
			const px = x(c.center.activity);
			const py = y(c.center.recovery);
			if (quota[kind] === 0 || names.has(name) || !clear(px, py)) continue;
			quota[kind]--;
			names.add(name);
			taken.push({ x: px, y: py });
			places.push({ name, px, py, kind });
		}

		return {
			x,
			y,
			bands,
			toMark,
			places,
			past,
			future,
			today: past[past.length - 1],
			arrowEnd: {
				px: x(today.position.activity + today.velocity.activity * VELOCITY_ARROW_DAYS),
				py: y(today.position.recovery + today.velocity.recovery * VELOCITY_ARROW_DAYS),
			},
		};
	}, [data, innerWidth, innerHeight]);

	// Kept apart from the terrain, so trying presets does not redraw the map.
	const simulated = useMemo(
		() => (map && whatIf?.length ? [map.today, ...whatIf.map((d) => map.toMark(d, "whatIf"))] : []),
		[map, whatIf],
	);
	const hover = useMemo(() => {
		if (!map) return null;
		const marks = [...map.past, ...map.future.slice(1), ...simulated.slice(1)];
		return {
			marks,
			delaunay: Delaunay.from(
				marks,
				(m) => m.px,
				(m) => m.py,
			),
		};
	}, [map, simulated]);

	function handlePointer(event: PointerEvent<SVGRectElement>) {
		if (!hover) return;
		const rect = event.currentTarget.getBoundingClientRect();
		const px = event.clientX - rect.left;
		const py = event.clientY - rect.top;
		const nearest = hover.marks[hover.delaunay.find(px, py)];
		const close = nearest && Math.hypot(nearest.px - px, nearest.py - py) < 24;
		setHovered(close ? nearest : null);
	}

	const futurePath = map?.future.map((m) => `${m.px},${m.py}`).join("L");
	const whatIfPath = simulated.map((m) => `${m.px},${m.py}`).join("L");
	const whatIfEnd = simulated.at(-1);

	return (
		<div className="terrain">
			<div ref={containerRef} className="chart">
				{map && (
					<svg
						width={width}
						height={height}
						role="img"
						aria-label={`Map of ${AXIS_LABELS.activity.position} against ${AXIS_LABELS.recovery.position}`}
					>
						<defs>
							<clipPath id="terrain-clip">
								<rect width={innerWidth} height={innerHeight} />
							</clipPath>
							<marker
								id="terrain-arrow"
								viewBox="0 0 10 10"
								refX="8"
								refY="5"
								markerWidth="7"
								markerHeight="7"
								orient="auto-start-reverse"
							>
								<path d="M0,1 L9,5 L0,9 Z" className="arrow-head" />
							</marker>
						</defs>

						<text className="axis-title" x={MARGIN.left} y={16}>
							<tspan className="axis-key axis-key-recovery">■</tspan> ↑{" "}
							{AXIS_LABELS.recovery.position}
						</text>
						<text className="axis-title" x={width - MARGIN.right} y={height - 4} textAnchor="end">
							<tspan className="axis-key axis-key-activity">■</tspan>{" "}
							{AXIS_LABELS.activity.position} →
						</text>

						<g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
							<g clipPath="url(#terrain-clip)">
								{map.bands.map((b) => (
									<path key={b.value} d={b.d} fill={b.fill} className="contour" />
								))}

								{map.past.slice(1).map((m, i) => {
									const prev = map.past[i];
									// Older segments fade but stay readable. The newest is fully opaque.
									const age = (map.past.length - 2 - i) / HISTORY_DAYS;
									return (
										<line
											key={m.date}
											className="trail"
											x1={prev.px}
											y1={prev.py}
											x2={m.px}
											y2={m.py}
											strokeOpacity={0.2 + 0.8 * (1 - age)}
										/>
									);
								})}
								<path className="prediction" d={`M${futurePath}`} />
								<circle
									className="prediction-end"
									cx={map.future.at(-1)?.px}
									cy={map.future.at(-1)?.py}
									r={4}
								/>

								{whatIfEnd && (
									<>
										<path className="what-if-path" d={`M${whatIfPath}`} />
										<circle className="what-if-end" cx={whatIfEnd.px} cy={whatIfEnd.py} r={5} />
									</>
								)}

								<line
									className="velocity-arrow"
									x1={map.today.px}
									y1={map.today.py}
									x2={map.arrowEnd.px}
									y2={map.arrowEnd.py}
									markerEnd="url(#terrain-arrow)"
								/>
								<circle className="today-ring" cx={map.today.px} cy={map.today.py} r={11} />
								<circle className="today-dot" cx={map.today.px} cy={map.today.py} r={5} />

								{hovered && <circle className="hover-dot" cx={hovered.px} cy={hovered.py} r={5} />}
							</g>

							{map.places.map((p) => (
								<text
									key={p.name}
									className={`place-label place-${p.kind}`}
									x={p.px}
									y={p.py - 14}
									textAnchor="middle"
								>
									{p.name}
								</text>
							))}
							<PathLabels
								map={map}
								whatIfEnd={whatIfEnd}
								startDate={formatShortDate(parseDay(map.past[0].date))}
							/>

							{map.x.ticks(Math.max(2, Math.floor(innerWidth / 90))).map((t) => (
								<text
									key={t}
									className="tick"
									x={map.x(t)}
									y={innerHeight + 16}
									textAnchor="middle"
								>
									{formatTick(t)}
								</text>
							))}
							{map.y.ticks(Math.max(2, Math.floor(innerHeight / 70))).map((t) => (
								<text key={t} className="tick" x={-8} y={map.y(t)} dy="0.32em" textAnchor="end">
									{formatTick(t)}
								</text>
							))}

							<rect
								className="hit-area"
								width={innerWidth}
								height={innerHeight}
								onPointerMove={handlePointer}
								onPointerLeave={() => setHovered(null)}
							/>
						</g>
					</svg>
				)}
				{map && hovered && (
					<div
						className="tooltip"
						style={{
							left: MARGIN.left + hovered.px,
							top: MARGIN.top + hovered.py,
							transform: `translate(${hovered.px > innerWidth / 2 ? "calc(-100% - 14px)" : "14px"}, -50%)`,
						}}
					>
						<div className="tooltip-date">
							{formatDate(parseDay(hovered.date))}
							{KIND_NOTE[hovered.kind]}
							{hovered === map.today && " · today"}
						</div>
						<div className="tooltip-row">
							{AXIS_LABELS.activity.position}
							<strong>{formatPosition(hovered.position.activity)}</strong>
						</div>
						<div className="tooltip-row">
							{AXIS_LABELS.recovery.position}
							<strong>{formatPosition(hovered.position.recovery)}</strong>
						</div>
					</div>
				)}
			</div>

			<ul className="map-legend">
				<li>
					<span
						className="legend-ramp"
						style={{ background: `linear-gradient(to right, ${TERRAIN_STOPS.join(",")})` }}
					/>
					<span className="legend-ramp-labels">
						<span>{CLUSTER_LABELS.unhealthy} valley</span>
						<span>{CLUSTER_LABELS.healthy} peak</span>
					</span>
				</li>
				<li>
					<svg width="28" height="10" aria-hidden="true">
						<defs>
							<linearGradient id="legend-fade">
								<stop offset="0" stopOpacity="0.2" stopColor="currentColor" />
								<stop offset="1" stopOpacity="1" stopColor="currentColor" />
							</linearGradient>
						</defs>
						<rect
							x="2"
							y="4"
							width="24"
							height="2"
							rx="1"
							fill="url(#legend-fade)"
							className="legend-trail"
						/>
					</svg>
					Past {HISTORY_DAYS} days
				</li>
				<li>
					<svg width="28" height="10" aria-hidden="true">
						<line
							className="velocity-arrow"
							x1="2"
							x2="20"
							y1="5"
							y2="5"
							markerEnd="url(#terrain-arrow)"
						/>
					</svg>
					Today's velocity, {VELOCITY_ARROW_DAYS} days long
				</li>
				{whatIfEnd && (
					<li>
						<svg width="28" height="10" aria-hidden="true">
							<line className="what-if-path" x1="2" x2="26" y1="5" y2="5" />
						</svg>
						{WHAT_IF_LABELS.path} ({WHAT_IF_LABELS.tag.toLowerCase()})
					</li>
				)}
			</ul>
		</div>
	);
}

interface PathLabelsProps {
	map: { today: MarkPoint; past: MarkPoint[]; future: MarkPoint[] };
	whatIfEnd: MarkPoint | undefined;
	startDate: string;
}

/** A label just past the end of a path, continuing its direction. */
function endLabel(from: MarkPoint, to: MarkPoint) {
	const right = to.px >= from.px;
	return { x: to.px + (right ? 10 : -10), y: to.py, anchor: right ? "start" : "end" } as const;
}

/** Direct labels on the path, so the legend does not have to explain them. */
function PathLabels({ map, whatIfEnd, startDate }: PathLabelsProps) {
	const { today, past, future } = map;
	const predictedEnd = future[future.length - 1];
	// Put "Today" on the side away from where the path is heading.
	const todaySide = predictedEnd.px >= today.px ? "end" : "start";
	const ahead = endLabel(today, predictedEnd);
	const simulated = whatIfEnd && endLabel(today, whatIfEnd);
	const start = past[0];
	return (
		<g>
			<text
				className="path-label path-label-strong"
				x={today.px + (todaySide === "end" ? -16 : 16)}
				y={today.py}
				dy="0.32em"
				textAnchor={todaySide}
			>
				Today
			</text>
			<text className="path-label" x={ahead.x} y={ahead.y} dy="0.32em" textAnchor={ahead.anchor}>
				In {PREDICTION_DAYS} days
			</text>
			{simulated && (
				<text
					className="path-label"
					x={simulated.x}
					y={simulated.y}
					dy="0.32em"
					textAnchor={simulated.anchor}
				>
					{WHAT_IF_LABELS.title}
				</text>
			)}
			<text
				className="path-label path-label-faint"
				x={start.px}
				y={start.py - 10}
				textAnchor="middle"
			>
				{startDate}
			</text>
		</g>
	);
}
