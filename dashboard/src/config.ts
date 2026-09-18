/**
 * Every name a person reads lives here, so renaming is a one line change. Data
 * uses the fixed keys (activity, recovery, steps, ...) and never these labels.
 */

export const APP_NAME = "HealthTrajectory";

export const AXIS_LABELS = {
	activity: {
		/** What one day adds. */
		velocity: "Activity",
		/** Where the person is after all of their days. */
		position: "Fitness",
	},
	recovery: {
		velocity: "Recovery",
		position: "Reserve",
	},
} as const;

export const FACTOR_LABELS = {
	steps: "Steps",
	heartRate: "Heart rate",
	sleep: "Sleep",
	hrv: "Heart rate variability",
	activeEnergy: "Active energy",
	exerciseTime: "Exercise time",
} as const;

export const CLUSTER_LABELS = {
	healthy: "Healthy",
	unhealthy: "Unhealthy",
} as const;

export const SECTION_LABELS = {
	aggregate: "Aggregate",
	daily: "Daily",
} as const;

/** Days of past trail drawn on the map. */
export const HISTORY_DAYS = 90;

/** Days of prediction drawn on the map. */
export const PREDICTION_DAYS = 30;

/**
 * Today's velocity is tiny next to the map's scale, so the arrow on the map is
 * drawn this many days long, as if today repeated.
 */
export const VELOCITY_ARROW_DAYS = 7;
