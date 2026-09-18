import { contours, Delaunay, format, geoIdentity, geoPath, max, scaleLinear, timeFormat } from "d3";
import { type PointerEvent, useMemo, useState } from "react";
import {
	AXIS_LABELS,
	CLUSTER_LABELS,
	HISTORY_DAYS,
	PREDICTION_DAYS,
	VELOCITY_ARROW_DAYS,
} from "../config";
import { parseDay } from "../lib/stats";
import { heightAt, TERRAIN_STOPS, terrainColor } from "../lib/terrain";
import { useElementWidth } from "../lib/useElementWidth";
import type { AxisVector, TrajectoryResponse } from "../types";

interface Props {
	data: TrajectoryResponse;
}

const MARGIN = { top: 28, right: 16, bottom: 36, left: 40 };
/** Pixel size of one cell in the height grid. Smaller is smoother and slower. */
const CELL = 5;
/** Contour levels on each side of neutral ground. */
const LEVELS = 7;
const formatDate = timeFormat("%a %b %-d, %Y");
const formatPosition = format("+.2f");
const formatTick = format("~g");

interface MarkPoint {
	date: string;
	position: AxisVector;
	predicted: boolean;
	px: number;
	py: number;
}

/**
 * The main view. The landscape comes from the subject's clusters, the trail is
 * where they have been, the dotted line is where the model expects them to go.
 */
export function TerrainMap({ data }: Props) {
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

		// Same data units per pixel on both axes, so peaks are not stretched.
		const unitsPerPx = Math.max((x1 - x0) / innerWidth, (y1 - y0) / innerHeight) * 1.05;
		const cx = (x0 + x1) / 2;
		const cy = (y0 + y1) / 2;
		x0 = cx - (unitsPerPx * innerWidth) / 2;
		x1 = cx + (unitsPerPx * innerWidth) / 2;
		y0 = cy - (unitsPerPx * innerHeight) / 2;
		y1 = cy + (unitsPerPx * innerHeight) / 2;
		const x = scaleLinear().domain([x0, x1]).range([0, innerWidth]);
		const y = scaleLinear().domain([y0, y1]).range([innerHeight, 0]);

		// Sample the height at cell centers, top row first.
		const nx = Math.ceil(innerWidth / CELL);
		const ny = Math.ceil(innerHeight / CELL);
		const values = new Float64Array(nx * ny);
		for (let j = 0; j < ny; j++) {
			const r = y.invert((j + 0.5) * CELL);
			for (let i = 0; i < nx; i++) {
				values[j * nx + i] = heightAt(clusters, x.invert((i + 0.5) * CELL), r);
			}
		}
		const maxAbs = max(values, (v) => Math.abs(v)) || 1;
		const step = maxAbs / LEVELS;
		// Bands are centered on zero, so neutral ground is one band, not a line.
		const thresholds = Array.from({ length: 2 * LEVELS }, (_, k) => (k - LEVELS + 0.5) * step);
		const color = terrainColor(maxAbs);
		const path = geoPath(geoIdentity().scale(CELL));
		const bands = contours()
			.size([nx, ny])
			.thresholds(thresholds)(Array.from(values))
			.map((c) => ({
				value: c.value,
				d: path(c) ?? "",
				fill: color(c.value + step / 2),
			}));

		const toMark = (d: { date: string; position: AxisVector }, predicted: boolean): MarkPoint => ({
			date: d.date,
			position: d.position,
			predicted,
			px: x(d.position.activity),
			py: y(d.position.recovery),
		});
		const past = history.map((d) => toMark(d, false));
		const future = [toMark(today, true), ...prediction.map((d) => toMark(d, true))];
		const marks = [...past, ...future.slice(1)];

		return {
			x,
			y,
			bands,
			background: color(-maxAbs),
			past,
			future,
			marks,
			delaunay: Delaunay.from(
				marks,
				(m) => m.px,
				(m) => m.py,
			),
			today: past[past.length - 1],
			arrowEnd: {
				px: x(today.position.activity + today.velocity.activity * VELOCITY_ARROW_DAYS),
				py: y(today.position.recovery + today.velocity.recovery * VELOCITY_ARROW_DAYS),
			},
		};
	}, [data, innerWidth, innerHeight]);

	function handlePointer(event: PointerEvent<SVGRectElement>) {
		if (!map) return;
		const rect = event.currentTarget.getBoundingClientRect();
		const px = event.clientX - rect.left;
		const py = event.clientY - rect.top;
		const nearest = map.marks[map.delaunay.find(px, py)];
		const close = nearest && Math.hypot(nearest.px - px, nearest.py - py) < 24;
		setHovered(close ? nearest : null);
	}

	const futurePath = map?.future.map((m) => `${m.px},${m.py}`).join("L");

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
								<rect width={innerWidth} height={innerHeight} rx={6} />
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
								<rect width={innerWidth} height={innerHeight} fill={map.background} />
								{map.bands.map((b) => (
									<path key={b.value} d={b.d} fill={b.fill} className="contour" />
								))}

								{map.past.slice(1).map((m, i) => {
									const prev = map.past[i];
									// Older segments fade out. The newest is fully opaque.
									const age = (map.past.length - 2 - i) / HISTORY_DAYS;
									return (
										<line
											key={m.date}
											className="trail"
											x1={prev.px}
											y1={prev.py}
											x2={m.px}
											y2={m.py}
											strokeOpacity={Math.max(0.08, 1 - age)}
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

							<rect className="frame" width={innerWidth} height={innerHeight} rx={6} />

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
							{hovered.predicted && " · predicted"}
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
					<svg width="22" height="22" aria-hidden="true">
						<circle className="today-ring" cx="11" cy="11" r="9" />
						<circle className="today-dot" cx="11" cy="11" r="4" />
					</svg>
					Today
				</li>
				<li>
					<svg width="28" height="10" aria-hidden="true">
						<defs>
							<linearGradient id="legend-fade">
								<stop offset="0" stopOpacity="0.1" stopColor="currentColor" />
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
						<line className="prediction" x1="2" x2="26" y1="5" y2="5" />
					</svg>
					Predicted {PREDICTION_DAYS} days
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
			</ul>
		</div>
	);
}
