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
export const HISTORY_DAYS = 180;

/** Days of prediction drawn on the map. */
export const PREDICTION_DAYS = 30;

/**
 * A what-if change happens today and is then kept up less and less. After this
 * many days it counts half as much as it did today.
 */
export const WHAT_IF_HALF_LIFE_DAYS = 7;

/**
 * Today's velocity is tiny next to the map's scale, so the arrow on the map is
 * drawn this many days long, as if today repeated.
 */
export const VELOCITY_ARROW_DAYS = 7;

export const SCENARIO_LABELS = {
	workout: "Add a 45 minute workout",
	walk: "Walk 5,000 more steps",
	earlySleep: "Sleep an hour earlier",
	restDay: "Take a rest day",
	lateNight: "Late night out",
} as const;

export const WHAT_IF_LABELS = {
	title: "What if",
	/** Marks every simulated number or path, so it is never read as a forecast. */
	tag: "Simulated",
	path: "What-if path",
} as const;

/**
 * Words for naming places on the map. A place far enough along an axis gets
 * that side's word; near the middle it gets none.
 */
export const REGION_WORDS = {
	activity: { high: "active", low: "inactive" },
	recovery: { high: "rested", low: "run down" },
	/** Near the middle on both axes. */
	middle: "balanced",
} as const;

/** How far from the middle, in axis units, before a place gets a word. */
export const REGION_THRESHOLD = 0.6;
