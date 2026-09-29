import { bisector, extent, line, scaleLinear, scaleTime, timeFormat } from "d3";
import { type PointerEvent, useMemo, useState } from "react";
import type { DatedValue } from "../lib/stats";
import { useElementWidth } from "../lib/useElementWidth";

interface Props {
	daily: DatedValue[];
	smoothed: DatedValue[];
	domain: [Date, Date];
	label: string;
	formatValue: (v: number) => string;
}

const HEIGHT = 88;
const PAD = 8;
const formatTooltipDate = timeFormat("%a %b %-d, %Y");
const byDate = bisector<DatedValue, Date>((d) => d.date).center;

/** A small trend with no axes: daily values faint, the rolling mean on top. */
export function Sparkline({ daily, smoothed, domain, label, formatValue }: Props) {
	const [containerRef, width] = useElementWidth<HTMLDivElement>();
	const [hoverIndex, setHoverIndex] = useState<number | null>(null);

	const { x, y, dailyPath, smoothedPath } = useMemo(() => {
		const x = scaleTime()
			.domain(domain)
			.range([PAD, Math.max(PAD, width - PAD)]);
		// The 7-day average near the left edge includes days before the window,
		// so it can sit outside the daily values. Fit both.
		const [lo = 0, hi = 1] = extent([...daily, ...smoothed], (d) => d.value);
		const y = scaleLinear()
			.domain([lo, hi === lo ? lo + 1 : hi])
			.range([HEIGHT - PAD, PAD]);
		const path = line<DatedValue>()
			.x((d) => x(d.date))
			.y((d) => y(d.value));
		return { x, y, dailyPath: path(daily), smoothedPath: path(smoothed) };
	}, [daily, smoothed, domain, width]);

	const last = smoothed.at(-1);
	const hovered = hoverIndex === null ? null : smoothed[hoverIndex];

	function handlePointer(event: PointerEvent<SVGRectElement>) {
		if (smoothed.length === 0) return;
		const rect = event.currentTarget.getBoundingClientRect();
		setHoverIndex(byDate(smoothed, x.invert(event.clientX - rect.left)));
	}

	return (
		<div ref={containerRef} className="chart sparkline">
			{width > 0 && (
				<svg width={width} height={HEIGHT} role="img" aria-label={label}>
					<path className="line-daily" d={dailyPath ?? undefined} />
					<path className="line-smoothed" d={smoothedPath ?? undefined} />
					{hovered ? (
						<>
							<line className="crosshair" x1={x(hovered.date)} x2={x(hovered.date)} y2={HEIGHT} />
							<circle className="end-dot" cx={x(hovered.date)} cy={y(hovered.value)} r={4} />
						</>
					) : (
						last && <circle className="end-dot" cx={x(last.date)} cy={y(last.value)} r={4} />
					)}
					<rect
						className="hit-area"
						width={width}
						height={HEIGHT}
						onPointerMove={handlePointer}
						onPointerLeave={() => setHoverIndex(null)}
					/>
				</svg>
			)}
			{hovered && hoverIndex !== null && (
				<div
					className="tooltip"
					style={{
						left: x(hovered.date),
						top: HEIGHT + 4,
						transform: x(hovered.date) > width / 2 ? "translateX(-100%)" : undefined,
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
