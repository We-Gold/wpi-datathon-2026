import { format } from "d3";
import { AXIS_LABELS, SCENARIO_LABELS, WHAT_IF_LABELS } from "../config";
import type { ScenarioKey } from "../lib/scenarios";
import type { Axis, AxisVector } from "../types";

interface Props {
	active: ScenarioKey[];
	onToggle: (key: ScenarioKey) => void;
	onReset: () => void;
	/** Today's real velocity. */
	velocity: AxisVector;
	/** Today's velocity with the presets applied. */
	simulated: AxisVector;
}

const KEYS = Object.keys(SCENARIO_LABELS) as ScenarioKey[];
const AXES: Axis[] = ["activity", "recovery"];
const formatValue = format("+.3f");

/**
 * Lets a person try changes to today and see where they lead. Nothing here
 * touches the real data or the model's prediction.
 */
export function WhatIfPanel({ active, onToggle, onReset, velocity, simulated }: Props) {
	const on = active.length > 0;
	return (
		<section className="note what-if" aria-labelledby="what-if-title">
			<header className="note-header">
				<h3 id="what-if-title">{WHAT_IF_LABELS.title}</h3>
				<span className="badge badge-what-if">{WHAT_IF_LABELS.tag}</span>
			</header>
			<fieldset className="presets">
				<legend className="visually-hidden">Changes to today</legend>
				{KEYS.map((key) => (
					<button
						key={key}
						type="button"
						className="pill"
						aria-pressed={active.includes(key)}
						onClick={() => onToggle(key)}
					>
						{SCENARIO_LABELS[key]}
					</button>
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
