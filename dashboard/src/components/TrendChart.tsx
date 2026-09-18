import { bisector, extent, format, line, scaleLinear, scaleTime, timeFormat } from "d3";
import { type PointerEvent, useMemo, useState } from "react";
import type { DatedValue } from "../lib/stats";
import { useElementWidth } from "../lib/useElementWidth";

interface Props {
	daily: DatedValue[];
	smoothed: DatedValue[];
	domain: [Date, Date];
	unit: string;
	formatValue: (v: number) => string;
	height?: number;
}

const MARGIN = { top: 12, right: 12, bottom: 24, left: 44 };
const formatTooltipDate = timeFormat("%a %b %-d, %Y");
const byDate = bisector<DatedValue, Date>((d) => d.date).center;

/**
 * One metric over time: the daily values as a faint line, the rolling mean on
 * top. React owns the DOM; d3 only does scales, ticks, and path strings.
 */
export function TrendChart({ daily, smoothed, domain, unit, formatValue, height = 168 }: Props) {
	const [containerRef, width] = useElementWidth<HTMLDivElement>();
	const [hoverIndex, setHoverIndex] = useState<number | null>(null);

	const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
	const innerHeight = height - MARGIN.top - MARGIN.bottom;

	const { x, y, dailyPath, smoothedPath } = useMemo(() => {
		const x = scaleTime().domain(domain).range([0, innerWidth]);
		const [lo = 0, hi = 1] = extent(daily, (d) => d.value);
		const pad = (hi - lo) * 0.08 || 1;
		const y = scaleLinear()
			.domain([lo - pad, hi + pad])
			.nice(4)
			.range([innerHeight, 0]);
		const path = line<DatedValue>()
			.x((d) => x(d.date))
			.y((d) => y(d.value));
		return { x, y, dailyPath: path(daily), smoothedPath: path(smoothed) };
	}, [daily, smoothed, domain, innerWidth, innerHeight]);

	const xTicks = x.ticks(Math.max(2, Math.floor(innerWidth / 110)));
	const xTickFormat = x.tickFormat();
	const yTicks = y.ticks(4);
	const yTickFormat = format("~s");

	const last = smoothed.at(-1);
	const hovered = hoverIndex === null ? null : smoothed[hoverIndex];

	function handlePointer(event: PointerEvent<SVGRectElement>) {
		if (smoothed.length === 0) return;
		const rect = event.currentTarget.getBoundingClientRect();
		const date = x.invert(event.clientX - rect.left);
		setHoverIndex(byDate(smoothed, date));
	}

	return (
		<div ref={containerRef} className="chart">
			{width > 0 && (
				<svg width={width} height={height} role="img" aria-label={`Trend in ${unit}`}>
					<g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
						{yTicks.map((t) => (
							<g key={t} transform={`translate(0,${y(t)})`}>
								<line className="gridline" x2={innerWidth} />
								<text className="tick" x={-8} dy="0.32em" textAnchor="end">
									{yTickFormat(t)}
								</text>
							</g>
						))}
						<line className="baseline" y1={innerHeight} y2={innerHeight} x2={innerWidth} />
						{xTicks.map((t) => (
							<text
								key={t.getTime()}
								className="tick"
								x={x(t)}
								y={innerHeight + 16}
								textAnchor="middle"
							>
								{xTickFormat(t)}
							</text>
						))}

						<path className="line-daily" d={dailyPath ?? undefined} />
						<path className="line-smoothed" d={smoothedPath ?? undefined} />
						{last && !hovered && (
							<circle className="end-dot" cx={x(last.date)} cy={y(last.value)} r={4} />
						)}

						{hovered && (
							<g>
								<line
									className="crosshair"
									x1={x(hovered.date)}
									x2={x(hovered.date)}
									y2={innerHeight}
								/>
								<circle className="end-dot" cx={x(hovered.date)} cy={y(hovered.value)} r={4} />
							</g>
						)}

						<rect
							className="hit-area"
							width={innerWidth}
							height={innerHeight}
							onPointerMove={handlePointer}
							onPointerLeave={() => setHoverIndex(null)}
						/>
					</g>
				</svg>
			)}
			{hovered && hoverIndex !== null && (
				<div
					className="tooltip"
					style={{
						left: MARGIN.left + x(hovered.date),
						top: MARGIN.top,
						transform:
							x(hovered.date) > innerWidth / 2
								? "translateX(calc(-100% - 12px))"
								: "translateX(12px)",
					}}
				>
					<div className="tooltip-date">{formatTooltipDate(hovered.date)}</div>
					<div className="tooltip-row">
						<span className="swatch swatch-faint" />
						Daily
						<strong>{formatValue(daily[hoverIndex].value)}</strong>
					</div>
					<div className="tooltip-row">
						<span className="swatch" />
						7-day average
						<strong>{formatValue(hovered.value)}</strong>
					</div>
				</div>
			)}
		</div>
	);
}
