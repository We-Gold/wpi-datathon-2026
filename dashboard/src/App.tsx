import { timeDay, timeFormat } from "d3";
import { useEffect, useMemo, useState } from "react";
import { ContributionBars } from "./components/ContributionBars";
import { MetricCard } from "./components/MetricCard";
import { TerrainMap } from "./components/TerrainMap";
import { VelocityPlot } from "./components/VelocityPlot";
import { APP_NAME, AXIS_LABELS, SECTION_LABELS } from "./config";
import { loadDashboardData, loadTrajectory } from "./data/source";
import { parseDay } from "./lib/stats";
import type { DashboardData, TrajectoryResponse } from "./types";

const RANGES = [
	{ days: 30, label: "30 days" },
	{ days: 90, label: "90 days" },
	{ days: 180, label: "180 days" },
] as const;

const formatToday = timeFormat("%A, %B %-d");

export function App() {
	const [data, setData] = useState<DashboardData | null>(null);
	const [subjectId, setSubjectId] = useState<string | null>(null);
	const [trajectory, setTrajectory] = useState<TrajectoryResponse | null>(null);
	const [rangeDays, setRangeDays] = useState<number>(90);

	useEffect(() => {
		loadDashboardData().then((d) => {
			setData(d);
			setSubjectId(d.subjects[0]?.subjectId ?? null);
		});
	}, []);

	useEffect(() => {
		if (!subjectId) return;
		let current = true;
		loadTrajectory(subjectId).then((t) => {
			if (current) setTrajectory(t);
		});
		return () => {
			current = false;
		};
	}, [subjectId]);

	const asOf = useMemo(
		() => (trajectory ? parseDay(trajectory.asOf) : timeDay.floor(new Date())),
		[trajectory],
	);
	const aggregateDomain = useMemo<[Date, Date]>(
		() => [timeDay.offset(asOf, -rangeDays + 1), asOf],
		[asOf, rangeDays],
	);
	const dailyDomain = useMemo<[Date, Date]>(() => [asOf, asOf], [asOf]);

	const subjectSeries = data && subjectId ? data.series[subjectId] : [];
	const today = trajectory?.history.at(-1);

	function cards(variant: "daily" | "aggregate", domain: [Date, Date]) {
		return data?.metrics.map((info) => {
			const series = subjectSeries.find((s) => s.metric === info.metric);
			return series ? (
				<MetricCard
					key={info.metric}
					info={info}
					series={series}
					variant={variant}
					domain={domain}
				/>
			) : null;
		});
	}

	return (
		<div className="app">
			<header className="app-header">
				<h1>{APP_NAME}</h1>
				<span className="badge">Mock data</span>
				<label className="field">
					<span>Subject</span>
					<select value={subjectId ?? ""} onChange={(e) => setSubjectId(e.target.value)}>
						{data?.subjects.map((s) => (
							<option key={s.subjectId} value={s.subjectId}>
								{s.label}
							</option>
						))}
					</select>
				</label>
			</header>

			{data === null || trajectory === null || today === undefined ? (
				<p className="empty">Loading…</p>
			) : (
				<main className="sections">
					<section className="section section-aggregate" aria-labelledby="aggregate-title">
						<header className="section-header">
							<h2 id="aggregate-title">{SECTION_LABELS.aggregate}</h2>
						</header>
						<article className="card">
							<header className="card-header">
								<div>
									<h3>
										{AXIS_LABELS.activity.position} and {AXIS_LABELS.recovery.position}
									</h3>
									<p className="card-subtitle">Hover the path to see a day</p>
								</div>
							</header>
							<TerrainMap data={trajectory} />
						</article>

						<div className="subsection-header">
							<h3>Trends</h3>
							<fieldset className="segmented">
								<legend className="visually-hidden">Time range</legend>
								{RANGES.map((r) => (
									<label key={r.days} className={r.days === rangeDays ? "active" : undefined}>
										<input
											type="radio"
											name="range"
											value={r.days}
											checked={r.days === rangeDays}
											onChange={() => setRangeDays(r.days)}
										/>
										{r.label}
									</label>
								))}
							</fieldset>
						</div>
						<div className="card-grid">{cards("aggregate", aggregateDomain)}</div>
					</section>

					<section className="section section-daily" aria-labelledby="daily-title">
						<header className="section-header">
							<h2 id="daily-title">{SECTION_LABELS.daily}</h2>
							<span className="section-date">{formatToday(asOf)}</span>
						</header>
						<article className="card">
							<header className="card-header">
								<div>
									<h3>Today's velocity</h3>
									<p className="card-subtitle">Same axes as the map, starting from zero</p>
								</div>
							</header>
							<VelocityPlot velocity={today.velocity} />
							<ContributionBars contributions={trajectory.contributions} />
						</article>
						<div className="card-grid card-grid-narrow">{cards("daily", dailyDomain)}</div>
					</section>
				</main>
			)}
		</div>
	);
}
