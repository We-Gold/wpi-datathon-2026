import { format, median, timeDay, timeFormat } from "d3";
import { useMemo } from "react";
import { periodChange, rollingMean, toDated } from "../lib/stats";
import type { MetricInfo, MetricSeries } from "../types";
import { DailyColumns } from "./DailyColumns";
import { TrendChart } from "./TrendChart";

interface Props {
	info: MetricInfo;
	series: MetricSeries;
	/** Daily shows each day against the usual; aggregate shows the rolling trend. */
	variant: "daily" | "aggregate";
	/** Visible window. For daily cards only the end matters. */
	domain: [Date, Date];
}

const SMOOTHING_DAYS = 7;
const DAILY_DAYS = 30;
const formatTableDate = timeFormat("%b %-d");

function valueFormatter(info: MetricInfo) {
	const f =
		info.displayUnit === "steps" || info.displayUnit === "kcal" ? format(",.0f") : format(",.1f");
	return (v: number) => f(v);
}

function tone(info: MetricInfo, delta: number | undefined) {
	if (delta === undefined || delta === 0 || info.direction === "neutral") return "neutral";
	return delta > 0 === (info.direction === "up") ? "good" : "bad";
}

export function MetricCard({ info, series, variant, domain }: Props) {
	const formatValue = valueFormatter(info);
	const end = domain[1];

	const stats = useMemo(() => {
		const all = toDated(series.points, info.toDisplay);
		if (variant === "daily") {
			const start = timeDay.offset(end, -DAILY_DAYS + 1);
			const recent = all.filter((d) => d.date >= start && d.date <= end);
			const today = recent.at(-1);
			const isToday = today !== undefined && today.date.getTime() === end.getTime();
			const usual = median(
				recent.filter((d) => d !== today),
				(d) => d.value,
			);
			return {
				rows: recent,
				smoothed: undefined,
				headline: isToday ? today.value : undefined,
				delta: isToday && usual !== undefined ? today.value - usual : undefined,
				deltaLabel: `vs ${DAILY_DAYS}-day median`,
			};
		}
		const smoothed = rollingMean(all, SMOOTHING_DAYS);
		const inRange = (d: { date: Date }) => d.date >= domain[0] && d.date <= end;
		const { current, previous } = periodChange(all, end, SMOOTHING_DAYS);
		return {
			rows: all.filter(inRange),
			smoothed: smoothed.filter(inRange),
			headline: current,
			delta: current !== undefined && previous !== undefined ? current - previous : undefined,
			deltaLabel: "vs prior week",
		};
	}, [series, info, variant, domain, end]);

	const deltaTone = tone(info, stats.delta);
	const { rows, smoothed } = stats;

	return (
		<article className="card metric-card">
			<header className="card-header">
				<div>
					<h3>{info.label}</h3>
					<p className="card-subtitle">
						{variant === "daily" ? "Today" : `${SMOOTHING_DAYS}-day average`}
					</p>
				</div>
				<div className="headline">
					<span className="headline-value">
						{stats.headline === undefined ? "–" : formatValue(stats.headline)}
					</span>
					<span className="headline-unit">{info.displayUnit}</span>
					{stats.delta !== undefined && (
						<span className={`delta delta-${deltaTone}`}>
							{stats.delta > 0 ? "▲" : stats.delta < 0 ? "▼" : "•"}{" "}
							{formatValue(Math.abs(stats.delta))} {stats.deltaLabel}
						</span>
					)}
				</div>
			</header>
			{rows.length === 0 ? (
				<p className="empty">No data in this range.</p>
			) : smoothed ? (
				<TrendChart
					daily={rows}
					smoothed={smoothed}
					domain={domain}
					unit={info.displayUnit}
					formatValue={formatValue}
				/>
			) : (
				<DailyColumns
					points={rows}
					end={end}
					days={DAILY_DAYS}
					formatValue={formatValue}
					kind={info.kind}
				/>
			)}
			<details className="table-view">
				<summary>Show data</summary>
				<table>
					<thead>
						<tr>
							<th scope="col">Date</th>
							<th scope="col">Daily</th>
							{smoothed && <th scope="col">{SMOOTHING_DAYS}-day average</th>}
						</tr>
					</thead>
					<tbody>
						{rows
							.map((d, i) => (
								<tr key={d.date.getTime()}>
									<td>{formatTableDate(d.date)}</td>
									<td>{formatValue(d.value)}</td>
									{smoothed && <td>{formatValue(smoothed[i].value)}</td>}
								</tr>
							))
							.reverse()}
					</tbody>
				</table>
			</details>
		</article>
	);
}
