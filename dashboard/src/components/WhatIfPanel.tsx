import { format } from "d3";
import { AXIS_LABELS, PERSISTENCE_LABELS, SCENARIO_LABELS, WHAT_IF_LABELS } from "../config";
import { isUsable, type Persistence, type ScenarioKey } from "../lib/scenarios";
import type { Axis, AxisVector, Scaling } from "../types";

interface Props {
	active: ScenarioKey[];
	onToggle: (key: ScenarioKey) => void;
	onReset: () => void;
	persistence: Persistence;
	onPersistence: (next: Persistence) => void;
	/** The subject's feature spreads. A preset with no matching data is disabled. */
	scaling: Scaling;
	/** Today's real velocity. */
	velocity: AxisVector;
	/** Today's velocity with the presets applied. */
	simulated: AxisVector;
}

const KEYS = Object.keys(SCENARIO_LABELS) as ScenarioKey[];
const PERSISTENCE = Object.keys(PERSISTENCE_LABELS) as Persistence[];
const AXES: Axis[] = ["activity", "recovery"];
const formatValue = format("+.3f");

/**
 * Lets a person try changes to today, once or as a habit, and see where they
 * lead. Nothing here touches the real data or the model's prediction.
 */
export function WhatIfPanel({
	active,
	onToggle,
	onReset,
	persistence,
	onPersistence,
	scaling,
	velocity,
	simulated,
}: Props) {
	const on = active.length > 0;
	return (
		<section className="note what-if" aria-labelledby="what-if-title">
			<header className="note-header">
				<h3 id="what-if-title">{WHAT_IF_LABELS.title}</h3>
				<span className="badge badge-what-if">{WHAT_IF_LABELS.tag}</span>
			</header>
			<fieldset className="presets">
				<legend className="visually-hidden">Changes to today</legend>
				{KEYS.map((key) => {
					const usable = isUsable(key, scaling);
					return (
						<button
							key={key}
							type="button"
							className="pill"
							aria-pressed={usable && active.includes(key)}
							disabled={!usable}
							title={usable ? undefined : WHAT_IF_LABELS.noData}
							onClick={() => onToggle(key)}
						>
							{SCENARIO_LABELS[key]}
						</button>
					);
				})}
			</fieldset>
			<fieldset className="segmented segmented-fill">
				<legend className="visually-hidden">How long the change lasts</legend>
				{PERSISTENCE.map((p) => (
					<label key={p} className={p === persistence ? "active" : undefined}>
						<input
							type="radio"
							name="persistence"
							value={p}
							checked={p === persistence}
							onChange={() => onPersistence(p)}
						/>
						{PERSISTENCE_LABELS[p]}
					</label>
				))}
			</fieldset>

			<table className="what-if-table">
				<caption className="visually-hidden">Today's velocity, real and simulated</caption>
				<thead className="what-if-head">
					<tr>
						<th scope="col">Velocity</th>
						<th scope="col">Today</th>
						<th scope="col">{WHAT_IF_LABELS.title}</th>
					</tr>
				</thead>
				<tbody>
					{AXES.map((axis) => (
						<tr key={axis}>
							<th scope="row">
								<span className={`axis-key axis-key-${axis}`}>■</span> {AXIS_LABELS[axis].velocity}
							</th>
							<td>{formatValue(velocity[axis])}</td>
							<td>{on ? formatValue(simulated[axis]) : "–"}</td>
						</tr>
					))}
				</tbody>
			</table>

			<button type="button" className="text-button" onClick={onReset} disabled={!on}>
				Clear changes
			</button>
		</section>
	);
}
