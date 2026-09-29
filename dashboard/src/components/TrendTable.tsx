import { useMemo } from "react";
import { deltaArrow, deltaTone, valueFormatter } from "../lib/format";
import { periodChange, rollingMean, toDated } from "../lib/stats";
import type { MetricInfo, MetricSeries } from "../types";
import { Sparkline } from "./Sparkline";

interface Props {
	metrics: MetricInfo[];
	series: MetricSeries[];
	/** Visible window for the sparklines. */
	domain: [Date, Date];
}

const SMOOTHING_DAYS = 7;

/** Every metric in one table: this week's average, the change, and the trend. */
export function TrendTable({ metrics, series, domain }: Props) {
	const rows = useMemo(
		() =>
			metrics.flatMap((info) => {
				const s = series.find((x) => x.metric === info.metric);
				if (!s) return [];
				const all = toDated(s.points, info.toDisplay);
				const inRange = (d: { date: Date }) => d.date >= domain[0] && d.date <= domain[1];
				const { current, previous } = periodChange(all, domain[1], SMOOTHING_DAYS);
				return [
					{
						info,
						daily: all.filter(inRange),
						smoothed: rollingMean(all, SMOOTHING_DAYS).filter(inRange),
						current,
						delta: current !== undefined && previous !== undefined ? current - previous : undefined,
					},
				];
			}),
		[metrics, series, domain],
	);

	return (
		<table className="trend-table">
			<thead className="trend-head">
				<tr>
					<th scope="col">Metric</th>
					<th scope="col" className="num">
						{SMOOTHING_DAYS}-day average
					</th>
					<th scope="col" className="num">
						vs prior week
					</th>
					<th scope="col" className="trend-col">
						Trend
					</th>
				</tr>
			</thead>
			<tbody>
				{rows.map(({ info, daily, smoothed, current, delta }) => {
					const f = valueFormatter(info);
					return (
						<tr key={info.metric} className="trend-row">
							<th scope="row" className="trend-name">
								{info.label}
							</th>
							<td className="num">
								<span className="trend-value">{current === undefined ? "–" : f(current)}</span>{" "}
								<span className="headline-unit">{info.displayUnit}</span>
							</td>
							<td className={`num trend-delta delta-${deltaTone(info, delta)}`}>
								{delta === undefined ? "–" : `${deltaArrow(delta)} ${f(Math.abs(delta))}`}
							</td>
							<td className="trend-col">
								<Sparkline
									daily={daily}
									smoothed={smoothed}
									domain={domain}
									label={`${info.label} trend`}
									formatValue={f}
								/>
							</td>
						</tr>
					);
				})}
			</tbody>
		</table>
	);
}
