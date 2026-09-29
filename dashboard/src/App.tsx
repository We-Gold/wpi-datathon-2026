import { timeDay, timeFormat } from "d3";
import { type KeyboardEvent, useEffect, useMemo, useState } from "react";
import { ContributionBars } from "./components/ContributionBars";
import { MetricCard } from "./components/MetricCard";
import { TerrainMap } from "./components/TerrainMap";
import { TrendTable } from "./components/TrendTable";
import { VelocityPlot } from "./components/VelocityPlot";
import { WhatIfPanel } from "./components/WhatIfPanel";
import { APP_NAME, AXIS_LABELS, SECTION_LABELS, WHAT_IF_LABELS } from "./config";
import { loadDashboardData, loadTrajectory } from "./data/source";
import { type ScenarioKey, scenarioDelta, simulatePath } from "./lib/scenarios";
import { parseDay } from "./lib/stats";
import { buildStory, pushSentence } from "./lib/story";
import type { DashboardData, TrajectoryResponse } from "./types";

const RANGES = [
	{ days: 30, label: "30 days" },
	{ days: 90, label: "90 days" },
	{ days: 180, label: "180 days" },
] as const;

const formatToday = timeFormat("%A, %B %-d");

const TABS = ["aggregate", "daily"] as const;
type Tab = (typeof TABS)[number];

/** The open tab lives in the URL hash, so a reload or a shared link keeps it. */
function tabFromHash(): Tab {
	const hash = window.location.hash.slice(1);
	return TABS.find((t) => t === hash) ?? "aggregate";
}

export function App() {
	const [data, setData] = useState<DashboardData | null>(null);
	const [subjectId, setSubjectId] = useState<string | null>(null);
	const [trajectory, setTrajectory] = useState<TrajectoryResponse | null>(null);
	const [rangeDays, setRangeDays] = useState<number>(90);
	const [tab, setTab] = useState<Tab>(tabFromHash);
	const [scenarios, setScenarios] = useState<ScenarioKey[]>([]);

	useEffect(() => {
		const sync = () => setTab(tabFromHash());
		window.addEventListener("hashchange", sync);
		return () => window.removeEventListener("hashchange", sync);
	}, []);

	function selectTab(next: Tab) {
		setTab(next);
		window.history.replaceState(null, "", `#${next}`);
	}

	function handleTabKey(event: KeyboardEvent<HTMLDivElement>) {
		const i = TABS.indexOf(tab);
		const next =
			event.key === "ArrowRight"
				? TABS[(i + 1) % TABS.length]
				: event.key === "ArrowLeft"
					? TABS[(i - 1 + TABS.length) % TABS.length]
					: event.key === "Home"
						? TABS[0]
						: event.key === "End"
							? TABS[TABS.length - 1]
							: undefined;
		if (!next) return;
		event.preventDefault();
		selectTab(next);
		document.getElementById(`tab-${next}`)?.focus();
	}

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

	const subjectSeries = data && subjectId ? data.series[subjectId] : [];
	const today = trajectory?.history.at(-1);

	const whatIf = useMemo(() => {
		if (!trajectory || !today || scenarios.length === 0) return undefined;
		const delta = scenarioDelta(scenarios);
		return {
			velocity: {
				activity: today.velocity.activity + delta.activity,
				recovery: today.velocity.recovery + delta.recovery,
			},
			path: simulatePath(trajectory.prediction, delta),
		};
	}, [trajectory, today, scenarios]);

	function toggleScenario(key: ScenarioKey) {
		setScenarios((current) =>
			current.includes(key) ? current.filter((k) => k !== key) : [...current, key],
		);
	}

	const story = useMemo(() => (trajectory ? buildStory(trajectory) : null), [trajectory]);

	function dailyCards() {
		return data?.metrics.map((info) => {
			const series = subjectSeries.find((s) => s.metric === info.metric);
			return series ? (
				<MetricCard key={info.metric} info={info} series={series} end={asOf} />
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
				<main>
					<nav className="tab-bar">
						<div
							className="tabs"
							role="tablist"
							aria-label="Dashboard views"
							onKeyDown={handleTabKey}
						>
							{TABS.map((t) => (
								<button
									key={t}
									id={`tab-${t}`}
									type="button"
									role="tab"
									className="tab"
									aria-selected={t === tab}
									aria-controls={`panel-${t}`}
									tabIndex={t === tab ? 0 : -1}
									onClick={() => selectTab(t)}
								>
									{SECTION_LABELS[t]}
								</button>
							))}
						</div>
						<span className="tab-date">{formatToday(asOf)}</span>
					</nav>

					{/* Both panels stay mounted, so switching tabs does not rebuild the charts. */}
					<section
						id="panel-aggregate"
						className="section"
						role="tabpanel"
						aria-labelledby="tab-aggregate"
						hidden={tab !== "aggregate"}
					>
						{story && (
							<header className="story">
								<h2 className="story-headline">{story.headline}</h2>
								<p className="story-dek">{story.dek}</p>
							</header>
						)}

						<div className="lead">
							<figure className="map-figure">
								<figcaption className="kicker">
									{AXIS_LABELS.activity.position} and {AXIS_LABELS.recovery.position}
								</figcaption>
								<TerrainMap data={trajectory} whatIf={whatIf?.path} />
							</figure>

							<aside className="margin-notes">
								<section className="note" aria-labelledby="today-note-title">
									<header className="note-header">
										<h3 id="today-note-title">Today</h3>
										<button
											type="button"
											className="icon-button"
											aria-label={`Open ${SECTION_LABELS.daily} for more detail`}
											title={`Open ${SECTION_LABELS.daily}`}
											onClick={() => {
												selectTab("daily");
												window.scrollTo({ top: 0 });
											}}
										>
											<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
												<path d="M6 4h8v8M14 4 4 14" />
											</svg>
										</button>
									</header>
									<p className="note-text">{pushSentence(today.velocity)}</p>
									<VelocityPlot velocity={today.velocity} whatIf={whatIf?.velocity} maxSize={220} />
									{whatIf && (
										<ul className="velocity-legend">
											<li>
												<svg width="28" height="10" aria-hidden="true">
													<line className="velocity-arrow" x1="2" x2="26" y1="5" y2="5" />
												</svg>
												Today
											</li>
											<li>
												<svg width="28" height="10" aria-hidden="true">
													<line className="what-if-arrow" x1="2" x2="26" y1="5" y2="5" />
												</svg>
												{WHAT_IF_LABELS.title} ({WHAT_IF_LABELS.tag.toLowerCase()})
											</li>
										</ul>
									)}
								</section>

								<WhatIfPanel
									active={scenarios}
									onToggle={toggleScenario}
									onReset={() => setScenarios([])}
									velocity={today.velocity}
									simulated={whatIf?.velocity ?? today.velocity}
								/>
							</aside>
						</div>

						<div className="subsection-header">
							<h2>Trends</h2>
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
						<TrendTable metrics={data.metrics} series={subjectSeries} domain={aggregateDomain} />
					</section>
					<section
						id="panel-daily"
						className="section"
						role="tabpanel"
						aria-labelledby="tab-daily"
						hidden={tab !== "daily"}
					>
						<article className="card">
							<header className="card-header">
								<div>
									<h2>Today's velocity</h2>
									<p className="card-subtitle">Same axes as the map, starting from zero</p>
								</div>
							</header>
							<div className="velocity-layout">
								<VelocityPlot velocity={today.velocity} />
								<ContributionBars contributions={trajectory.contributions} />
							</div>
						</article>
						<div className="subsection-header">
							<h2>Today against the usual</h2>
						</div>
						<div className="card-grid">{dailyCards()}</div>
					</section>
				</main>
			)}
		</div>
	);
}
