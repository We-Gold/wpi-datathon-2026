import type { PERSISTENCE_LABELS, SCENARIO_LABELS } from "../config";
import type { Axis, AxisVector, DashboardIndex, PredictedDay, Scaling } from "../types";

export type ScenarioKey = keyof typeof SCENARIO_LABELS;
export type Persistence = keyof typeof PERSISTENCE_LABELS;

const AXES: Axis[] = ["activity", "recovery"];

/**
 * How each preset changes today's raw daily features, in the pipeline's units.
 * These are rough, fixed sizes, the same for everyone. The subject's own
 * spread of each feature decides how much the change moves their scores.
 */
export const SCENARIOS: Record<ScenarioKey, Record<string, number>> = {
	workout: {
		exercise_minutes: 45,
		active_energy_kcal: 350,
		hr_active_excess_bpm: 5,
		sedentary_hours: -0.75,
	},
	walk: { steps: 5000, distance_m: 3800, sedentary_hours: -0.75 },
	earlySleep: { sleep_hours: 1 },
	restDay: {
		steps: -4000,
		distance_m: -3000,
		exercise_minutes: -30,
		active_energy_kcal: -250,
		hr_active_excess_bpm: -5,
	},
	lateNight: { sleep_hours: -2, resting_hr_bpm: 3, hrv_sdnn_ms: -10 },
};

/** Whether the subject has data for at least one feature the preset changes. */
export function isUsable(key: ScenarioKey, scaling: Scaling): boolean {
	return Object.keys(SCENARIOS[key]).some((feature) => feature in scaling.iqr);
}

/**
 * Change to today's scores from the chosen presets. This is the Python score
 * formula applied to the change: each ingredient moves by sign * change / IQR,
 * and a score is the mean over today's observed ingredients. Features the
 * subject has no data for are skipped.
 */
export function scoreDelta(
	keys: Iterable<ScenarioKey>,
	scaling: Scaling,
	axes: DashboardIndex["model"]["axes"],
): AxisVector {
	const delta: AxisVector = { activity: 0, recovery: 0 };
	for (const key of keys) {
		for (const [feature, change] of Object.entries(SCENARIOS[key])) {
			const iqr = scaling.iqr[feature];
			if (iqr === undefined) continue;
			for (const axis of AXES) {
				const sign = axes[axis][feature];
				if (sign !== undefined && scaling.counts[axis] > 0) {
					delta[axis] += (sign * change) / iqr / scaling.counts[axis];
				}
			}
		}
	}
	return delta;
}

/**
 * How far a score change moves the position `day` days from today (0 is
 * today), from the moving-average definition alone. A one-day change enters
 * at a share of 1 - keep and fades by `keep` each day. A daily habit keeps
 * adding, so its effect builds toward the full change.
 */
export function positionEffect(
	delta: AxisVector,
	keep: number,
	day: number,
	persistence: Persistence,
): AxisVector {
	const share = persistence === "once" ? (1 - keep) * keep ** day : 1 - keep ** (day + 1);
	return { activity: delta.activity * share, recovery: delta.recovery * share };
}

/**
 * The model's forecast plus the effect of the change. The forecast itself is
 * not refit, so this is "the same future, plus this change", not a new forecast.
 */
export function simulatePath(
	prediction: PredictedDay[],
	delta: AxisVector,
	keep: number,
	persistence: Persistence,
): PredictedDay[] {
	return prediction.map((d, i) => {
		const effect = positionEffect(delta, keep, i + 1, persistence);
		return {
			date: d.date,
			position: {
				activity: d.position.activity + effect.activity,
				recovery: d.position.recovery + effect.recovery,
			},
		};
	});
}
