import { median, timeDay, timeFormat } from "d3";
import { useMemo } from "react";
import { deltaArrow, deltaTone, valueFormatter } from "../lib/format";
import { toDated } from "../lib/stats";
import type { MetricInfo, MetricSeries } from "../types";
import { DailyColumns } from "./DailyColumns";

interface Props {
	info: MetricInfo;
	series: MetricSeries;
	/** The present day. */
	end: Date;
}

const DAILY_DAYS = 30;
const formatTableDate = timeFormat("%b %-d");

/** One metric today, against the last month. */
export function MetricCard({ info, series, end }: Props) {
	const formatValue = valueFormatter(info);

	const stats = useMemo(() => {
		const all = toDated(series.points, info.toDisplay);
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
			headline: isToday ? today.value : undefined,
			delta: isToday && usual !== undefined ? today.value - usual : undefined,
		};
	}, [series, info, end]);

	const tone = deltaTone(info, stats.delta);
	const { rows } = stats;

	return (
		<article className="card metric-card">
			<header className="card-header">
				<div>
					<h3>{info.label}</h3>
					<p className="card-subtitle">Today</p>
				</div>
				<div className="headline">
					<span className="headline-value">
						{stats.headline === undefined ? "–" : formatValue(stats.headline)}
					</span>
					<span className="headline-unit">{info.displayUnit}</span>
					{stats.delta !== undefined && (
						<span className={`delta delta-${tone}`}>
							{deltaArrow(stats.delta)} {formatValue(Math.abs(stats.delta))} vs {DAILY_DAYS}-day
							median
						</span>
					)}
				</div>
			</header>
			{rows.length === 0 ? (
				<p className="empty">No data in this range.</p>
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
						</tr>
					</thead>
					<tbody>
						{rows
							.map((d) => (
								<tr key={d.date.getTime()}>
									<td>{formatTableDate(d.date)}</td>
									<td>{formatValue(d.value)}</td>
								</tr>
							))
							.reverse()}
					</tbody>
				</table>
			</details>
		</article>
	);
}
