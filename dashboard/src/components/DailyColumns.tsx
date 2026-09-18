import { extent, scaleBand, scaleLinear, timeDay, timeFormat } from "d3";
import { useState } from "react";
import { verticalBar } from "../lib/shapes";
import type { DatedValue } from "../lib/stats";
import { useElementWidth } from "../lib/useElementWidth";

interface Props {
	points: DatedValue[];
	end: Date;
	days: number;
	formatValue: (v: number) => string;
	/** Totals are bars from zero. Levels are dots on a scale fitted to the values. */
	kind: "total" | "level";
	height?: number;
}

const MARGIN = { top: 8, right: 4, bottom: 20, left: 28 };
const formatDate = timeFormat("%a %b %-d");
const formatTick = timeFormat("%b %-d");

/** One column per day, the last one (today) at full strength. */
export function DailyColumns({ points, end, days, formatValue, kind, height = 120 }: Props) {
	const [containerRef, width] = useElementWidth<HTMLDivElement>();
	const [hovered, setHovered] = useState<DatedValue | null>(null);

	const innerWidth = Math.max(0, width - MARGIN.left - MARGIN.right);
	const innerHeight = height - MARGIN.top - MARGIN.bottom;
	const dayList = timeDay.range(timeDay.offset(end, -days + 1), timeDay.offset(end, 1));
	const byTime = new Map(points.map((p) => [p.date.getTime(), p]));

	const x = scaleBand<number>()
		.domain(dayList.map((d) => d.getTime()))
		.range([0, innerWidth])
		.paddingInner(0.2);
	const [lo = 0, hi = 1] = extent(points, (p) => p.value);
	const pad = (hi - lo) * 0.15 || 1;
	const y = scaleLinear()
		.domain(kind === "total" ? [0, hi] : [lo - pad, hi + pad])
		.nice(3)
		.range([innerHeight, 0]);
	const yTicks = kind === "level" ? y.ticks(3) : [];
	const barWidth = Math.min(24, x.bandwidth());
	const offset = (x.bandwidth() - barWidth) / 2;
	const endTime = end.getTime();

	return (
		<div ref={containerRef} className="chart">
			{width > 0 && (
				<svg width={width} height={height} role="img" aria-label={`Last ${days} days`}>
					<g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
						{yTicks.map((t) => (
							<g key={t} transform={`translate(0,${y(t)})`}>
								<line className="gridline" x2={innerWidth} />
								<text className="tick" x={-6} dy="0.32em" textAnchor="end">
									{t}
								</text>
							</g>
						))}
						{kind === "total" && (
							<line className="baseline" x2={innerWidth} y1={innerHeight} y2={innerHeight} />
						)}
						{dayList.map((day) => {
							const t = day.getTime();
							const p = byTime.get(t);
							const bx = x(t) ?? 0;
							return (
								<g
									key={t}
									onPointerEnter={() => setHovered(p ?? null)}
									onPointerLeave={() => setHovered(null)}
								>
									<rect className="hit-area" x={bx} width={x.step()} height={innerHeight} />
									{p && kind === "total" && (
										<path
											className={t === endTime ? "column column-today" : "column"}
											d={verticalBar(bx + offset, barWidth, innerHeight, y(p.value), 3)}
										/>
									)}
									{p && kind === "level" && (
										<circle
											className={t === endTime ? "level-dot level-dot-today" : "level-dot"}
											cx={bx + x.bandwidth() / 2}
											cy={y(p.value)}
											r={t === endTime ? 5 : 4}
										/>
									)}
								</g>
							);
						})}
						<text className="tick" x={0} y={innerHeight + 14}>
							{formatTick(dayList[0])}
						</text>
						<text className="tick" x={innerWidth} y={innerHeight + 14} textAnchor="end">
							Today
						</text>
					</g>
				</svg>
			)}
			{hovered && (
				<div
					className="tooltip"
					style={{
						left: MARGIN.left + (x(hovered.date.getTime()) ?? 0) + x.bandwidth() / 2,
						top: 0,
						transform: `translateX(${(x(hovered.date.getTime()) ?? 0) > innerWidth / 2 ? "calc(-100% - 8px)" : "8px"})`,
					}}
				>
					<div className="tooltip-date">{formatDate(hovered.date)}</div>
					<div className="tooltip-row">
						<strong>{formatValue(hovered.value)}</strong>
					</div>
				</div>
			)}
		</div>
	);
}
