import {
	contours,
	curveCatmullRom,
	Delaunay,
	format,
	geoIdentity,
	geoPath,
	line,
	max,
	median,
	scaleLinear,
	timeFormat,
} from "d3";
import { type PointerEvent, useMemo, useState } from "react";
import {
	AXIS_LABELS,
	CLUSTER_LABELS,
	HISTORY_DAYS,
	PREDICTION_DAYS,
	REGION_NOTE,
	TRAIL_OPACITY,
	TRAIL_SMOOTHING_DAYS,
	WHAT_IF_LABELS,
} from "../config";
import { smoothPath } from "../lib/shapes";
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
 * How much more one axis may be stretched than the other. The axes measure
 * different things, so equal units are not required. Some subjects move five
 * times more on one axis, and without the stretch they fill a thin strip.
 */
const MAX_STRETCH = 3;
/** Room kept free inside each edge, so the today ring and labels are not cut off. */
const FIT_PADDING = 28;
/**
 * The arrow's screen length. A day as big as the subject's typical day is
 * drawn at `typical`. Bigger and smaller days scale from there, within limits,
 * so one unusual day or subject never gives a huge or hidden arrow.
 */
const ARROW_PX = { min: 20, typical: 40, max: 84 };
/** Width of the fade at the map's edges, so a hill that runs off the map fades out. */
const EDGE_FADE = 24;
/** Named places on the map, most prominent first. */
const PEAK_LABELS = 3;
const VALLEY_LABELS = 2;
/** Room a place name needs, so names never overlap each other or the path ends. */
const LABEL_BOX = { width: 130, height: 26 };
const formatDate = timeFormat("%a %b %-d, %Y");
const formatShortDate = timeFormat("%b %-d");
const formatPosition = format("+.2f");
const formatTick = format("~g");
/** Smooth curve through the forecast and what-if points. */
const curve = line<{ px: number; py: number }>()
	.x((m) => m.px)
	.y((m) => m.py)
	.curve(curveCatmullRom.alpha(0.5));

type Bounds = [x0: number, x1: number, y0: number, y1: number];

/**
 * Scales that fit the bounds inside the padded plot. Similar data units per
 * pixel on both axes, so peaks keep a readable shape; the slack axis may
 * stretch, up to MAX_STRETCH, to use the space.
 */
function fitScales([x0, x1, y0, y1]: Bounds, innerWidth: number, innerHeight: number) {
	const fitX = (x1 - x0 || 1) / Math.max(1, innerWidth - 2 * FIT_PADDING);
	const fitY = (y1 - y0 || 1) / Math.max(1, innerHeight - 2 * FIT_PADDING);
	const unitsPerPx = Math.max(fitX, fitY);
	const unitsPerPxX = fitX < unitsPerPx ? Math.max(fitX, unitsPerPx / MAX_STRETCH) : unitsPerPx;
	const unitsPerPxY = fitY < unitsPerPx ? Math.max(fitY, unitsPerPx / MAX_STRETCH) : unitsPerPx;
	const cx = (x0 + x1) / 2;
	const cy = (y0 + y1) / 2;
	const halfX = (unitsPerPxX * innerWidth) / 2;
	const halfY = (unitsPerPxY * innerHeight) / 2;
	return {
		x: scaleLinear()
			.domain([cx - halfX, cx + halfX])
			.range([0, innerWidth]),
		y: scaleLinear()
			.domain([cy - halfY, cy + halfY])
			.range([innerHeight, 0]),
	};
}

/**
 * How many data units of today's velocity make the arrow, given the scales.
 * A day as big as the subject's typical day is drawn ARROW_PX.typical long.
 */
function arrowScaleFor(
	history: TrajectoryResponse["history"],
	velocity: AxisVector,
	x: (v: number) => number,
	y: (v: number) => number,
) {
	const screenLength = (v: AxisVector) => Math.hypot(x(v.activity) - x(0), y(v.recovery) - y(0));
	const typical = median(history, (d) => screenLength(d.velocity)) || 1;
	const length = screenLength(velocity);
	const target = Math.min(
		ARROW_PX.max,
		Math.max(ARROW_PX.min, (ARROW_PX.typical * length) / typical),
	);
	return length > 0 ? target / length : 0;
}

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

	// Fit the paths: the trail, the prediction, and any what-if path shown.
	// The terrain is the background and may run past the edges, where it fades.
	// Fitting whole clusters, or everywhere a preset could reach, left some
	// subjects' paths in a thin strip. The result is a string, so the terrain
	// is only rebuilt when the bounds change, not for a what-if inside them.
	const bounds = useMemo(() => {
		const points = [
			...data.history.map((d) => d.position),
			...data.prediction.map((d) => d.position),
			...(whatIf ?? []).map((d) => d.position),
		];
		const xs = points.map((p) => p.activity);
		const ys = points.map((p) => p.recovery);
		return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)].join(",");
	}, [data, whatIf]);

	const map = useMemo(() => {
		if (innerWidth <= 0) return null;
		const { history, prediction, clusters } = data;
		const today = history.at(-1);
		if (!today) return null;

		// Fit the paths, then make room for the arrow tip too. Its screen length
		// depends on the scales, so it is measured on a first fit.
		const pathBounds = bounds.split(",").map(Number) as Bounds;
		const first = fitScales(pathBounds, innerWidth, innerHeight);
		const k = arrowScaleFor(history, today.velocity, first.x, first.y);
		const tipX = today.position.activity + today.velocity.activity * k;
		const tipY = today.position.recovery + today.velocity.recovery * k;
		const { x, y } = fitScales(
			[
				Math.min(pathBounds[0], tipX),
				Math.max(pathBounds[1], tipX),
				Math.min(pathBounds[2], tipY),
				Math.max(pathBounds[3], tipY),
			],
			innerWidth,
			innerHeight,
		);

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
		// Marks sit on the drawn, smoothed trail. Their values stay the real ones.
		const past = smoothPath(
			history.map((d) => toMark(d, "past")),
			TRAIL_SMOOTHING_DAYS,
		);
		const todayMark = past[past.length - 1];
		const future = [todayMark, ...prediction.map((d) => toMark(d, "predicted"))];

		// Name the most prominent peaks and valleys, skipping any that would
		// crowd another name, the path ends, or repeat a name already shown.
		const taken = [todayMark, future[future.length - 1]].map((m) => ({
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

		// Today's direction, with a length set against the subject's own typical day.
		const arrowScale = arrowScaleFor(history, today.velocity, x, y);

		return {
			x,
			y,
			bands,
			toMark,
			places,
			past,
			future,
			today: todayMark,
			arrowEnd: {
				px: todayMark.px + (x(today.velocity.activity) - x(0)) * arrowScale,
				py: todayMark.py + (y(today.velocity.recovery) - y(0)) * arrowScale,
			},
		};
	}, [data, bounds, innerWidth, innerHeight]);

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

	const futurePath = map ? (curve(map.future) ?? "") : "";
	const whatIfPath = curve(simulated) ?? "";
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
							<filter
								id="terrain-fade-blur"
								filterUnits="userSpaceOnUse"
								x={0}
								y={0}
								width={innerWidth}
								height={innerHeight}
							>
								<feGaussianBlur stdDeviation={EDGE_FADE / 2} />
							</filter>
							<mask id="terrain-fade">
								<rect
									x={EDGE_FADE}
									y={EDGE_FADE}
									width={Math.max(0, innerWidth - 2 * EDGE_FADE)}
									height={Math.max(0, innerHeight - 2 * EDGE_FADE)}
									fill="white"
									filter="url(#terrain-fade-blur)"
								/>
							</mask>
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
								<g mask="url(#terrain-fade)">
									{map.bands.map((b) => (
										<path key={b.value} d={b.d} fill={b.fill} className="contour" />
									))}
								</g>

								{map.past.slice(1).map((m, i) => {
									const prev = map.past[i];
									// Older segments fade out quickly. Only recent weeks are strong.
									const age = (map.past.length - 2 - i) / HISTORY_DAYS;
									const recency = (1 - Math.min(1, age)) ** 2;
									return (
										<line
											key={m.date}
											className="trail"
											x1={prev.px}
											y1={prev.py}
											x2={m.px}
											y2={m.py}
											strokeOpacity={
												TRAIL_OPACITY.oldest +
												(TRAIL_OPACITY.newest - TRAIL_OPACITY.oldest) * recency
											}
										/>
									);
								})}
								<path className="prediction" d={futurePath} />
								<circle
									className="prediction-end"
									cx={map.future.at(-1)?.px}
									cy={map.future.at(-1)?.py}
									r={4}
								/>

								{whatIfEnd && (
									<>
										<path className="what-if-path" d={whatIfPath} />
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
								<stop offset="0" stopOpacity={TRAIL_OPACITY.oldest} stopColor="currentColor" />
								<stop offset="1" stopOpacity={TRAIL_OPACITY.newest} stopColor="currentColor" />
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
					Today's direction, longer on a bigger day than usual
				</li>
				<li>{REGION_NOTE}</li>
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
