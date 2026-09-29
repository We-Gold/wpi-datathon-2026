import { PREDICTION_DAYS, type SCENARIO_LABELS, WHAT_IF_HALF_LIFE_DAYS } from "../config";
import type { AxisVector, FactorKey, PredictedDay } from "../types";

export type ScenarioKey = keyof typeof SCENARIO_LABELS;

/**
 * How each preset changes today's contributions, per factor. These are rough
 * hand set sizes on the same scale as a mock day, not outputs of the model.
 */
export const SCENARIOS: Record<ScenarioKey, Partial<Record<FactorKey, AxisVector>>> = {
	workout: {
		exerciseTime: { activity: 0.02, recovery: -0.006 },
		activeEnergy: { activity: 0.012, recovery: -0.004 },
		heartRate: { activity: 0.006, recovery: -0.006 },
	},
	walk: {
		steps: { activity: 0.022, recovery: -0.003 },
		activeEnergy: { activity: 0.006, recovery: 0 },
	},
	earlySleep: {
		sleep: { activity: 0, recovery: 0.025 },
		hrv: { activity: 0, recovery: 0.008 },
	},
	restDay: {
		steps: { activity: -0.015, recovery: 0.006 },
		exerciseTime: { activity: -0.012, recovery: 0.01 },
		hrv: { activity: 0, recovery: 0.008 },
	},
	lateNight: {
		sleep: { activity: 0, recovery: -0.03 },
		hrv: { activity: 0, recovery: -0.012 },
		heartRate: { activity: 0, recovery: -0.01 },
	},
};

const ZERO: AxisVector = { activity: 0, recovery: 0 };

/** Total change to today's velocity from the chosen presets. */
export function scenarioDelta(keys: Iterable<ScenarioKey>): AxisVector {
	let delta = ZERO;
	for (const key of keys) {
		for (const change of Object.values(SCENARIOS[key])) {
			delta = {
				activity: delta.activity + change.activity,
				recovery: delta.recovery + change.recovery,
			};
		}
	}
	return delta;
}

const KEEP_RATE = 0.5 ** (1 / WHAT_IF_HALF_LIFE_DAYS);

/**
 * Shifts the model's prediction by the change. The change is full today and
 * then fades, as if the person keeps it up to some extent. Each later day adds
 * a smaller step, so the shift grows fast at first and then levels off. This is
 * simple addition on top of the prediction, not a new forecast.
 */
export function simulatePath(prediction: PredictedDay[], delta: AxisVector): PredictedDay[] {
	let kept = 0;
	return prediction.slice(0, PREDICTION_DAYS).map((d, i) => {
		kept += KEEP_RATE ** i;
		return {
			date: d.date,
			position: {
				activity: d.position.activity + delta.activity * kept,
				recovery: d.position.recovery + delta.recovery * kept,
			},
		};
	});
}
