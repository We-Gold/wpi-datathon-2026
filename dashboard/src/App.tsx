import { timeDay } from "d3";
import { type KeyboardEvent, useEffect, useMemo, useState } from "react";
import { ContributionBars } from "./components/ContributionBars";
import { MetricCard } from "./components/MetricCard";
import { TerrainMap } from "./components/TerrainMap";
import { TrendTable } from "./components/TrendTable";
import { VelocityPlot } from "./components/VelocityPlot";
import { WhatIfPanel } from "./components/WhatIfPanel";
import { APP_NAME, AXIS_LABELS, SECTION_LABELS, WHAT_IF_LABELS } from "./config";
import { METRICS } from "./data/metrics";
import { loadIndex, loadSubject } from "./data/source";
import {
	type Persistence,
	positionEffect,
	type ScenarioKey,
	scoreDelta,
	simulatePath,
} from "./lib/scenarios";
import { parseDay } from "./lib/stats";
import { buildStory } from "./lib/story";
import type { DashboardIndex, SubjectData } from "./types";

const RANGES = [
	{ days: 30, label: "30 days" },
	{ days: 90, label: "90 days" },
	{ days: 180, label: "180 days" },
] as const;

const TABS = ["aggregate", "daily"] as const;
type Tab = (typeof TABS)[number];

/** The open tab lives in the URL hash, so a reload or a shared link keeps it. */
function tabFromHash(): Tab {
	const hash = window.location.hash.slice(1);
	return TABS.find((t) => t === hash) ?? "aggregate";
}

export function App() {
	const [index, setIndex] = useState<DashboardIndex | null>(null);
	const [subjectId, setSubjectId] = useState<string | null>(null);
	const [subject, setSubject] = useState<SubjectData | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [rangeDays, setRangeDays] = useState<number>(90);
	const [tab, setTab] = useState<Tab>(tabFromHash);
	const [scenarios, setScenarios] = useState<ScenarioKey[]>([]);
	const [persistence, setPersistence] = useState<Persistence>("once");

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
		loadIndex()
			.then((i) => {
				setIndex(i);
				setSubjectId(i.subjects[0]?.subjectId ?? null);
			})
			.catch((e: Error) => setError(e.message));
	}, []);

	useEffect(() => {
		if (!subjectId) return;
		let current = true;
		loadSubject(subjectId)
			.then((s) => {
				if (current) setSubject(s);
			})
			.catch((e: Error) => setError(e.message));
		return () => {
			current = false;
		};
	}, [subjectId]);

	const trajectory = subject?.trajectory ?? null;
	const asOf = useMemo(
		() => (trajectory ? parseDay(trajectory.asOf) : timeDay.floor(new Date())),
		[trajectory],
	);
	const aggregateDomain = useMemo<[Date, Date]>(
		() => [timeDay.offset(asOf, -rangeDays + 1), asOf],
		[asOf, rangeDays],
	);

	const subjectSeries = subject?.series ?? [];
	const today = trajectory?.history.at(-1);

	const whatIf = useMemo(() => {
		if (!index || !subject || !today || scenarios.length === 0) return undefined;
		const { keep, prediction } = subject.trajectory;
		const delta = scoreDelta(scenarios, subject.scaling, index.model.axes);
		const now = positionEffect(delta, keep, 0, persistence);
		return {
			velocity: {
				activity: today.velocity.activity + now.activity,
				recovery: today.velocity.recovery + now.recovery,
			},
			path: simulatePath(prediction, delta, keep, persistence),
		};
	}, [index, subject, today, scenarios, persistence]);

	function toggleScenario(key: ScenarioKey) {
		setScenarios((current) =>
			current.includes(key) ? current.filter((k) => k !== key) : [...current, key],
		);
	}

	const story = useMemo(() => (trajectory ? buildStory(trajectory) : null), [trajectory]);

	function dailyCards() {
		return METRICS.map((info) => {
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
				<div className="tabs" role="tablist" aria-label="Dashboard views" onKeyDown={handleTabKey}>
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
				<label className="field">
					<span>Subject</span>
					<select value={subjectId ?? ""} onChange={(e) => setSubjectId(e.target.value)}>
						{index?.subjects.map((s) => (
							<option key={s.subjectId} value={s.subjectId}>
								{s.label}
							</option>
						))}
					</select>
				</label>
			</header>

			{error ? (
				<p className="empty">{error}</p>
			) : subject === null || trajectory === null || today === undefined ? (
				<p className="empty">Loading…</p>
			) : (
				<main>
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
									<VelocityPlot
										velocity={today.velocity}
										whatIf={whatIf?.velocity}
										maxSize={220}
										crop
									/>
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
									persistence={persistence}
									onPersistence={setPersistence}
									scaling={subject.scaling}
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
						<TrendTable metrics={METRICS} series={subjectSeries} domain={aggregateDomain} />
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
