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

/** Keys match FACTORS in src/health/trajectory.py. */
export const FACTOR_LABELS = {
	steps: "Steps",
	exerciseTime: "Exercise time",
	activeEnergy: "Active energy",
	sitting: "Sitting and light activity",
	heartRate: "Heart rate",
	hrv: "Heart rate variability",
	sleep: "Sleep",
	breathing: "Breathing rate",
	pull: "Pull toward usual",
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

/**
 * Width, in days, of the smoothing applied when the past trail is drawn. The
 * data is not changed, and today stays exactly where it is.
 */
export const TRAIL_SMOOTHING_DAYS = 3;

/** Stroke opacity of the oldest and newest trail segments. */
export const TRAIL_OPACITY = { oldest: 0.04, newest: 0.75 } as const;

/** Days of prediction drawn on the map. */
export const PREDICTION_DAYS = 30;

export const SCENARIO_LABELS = {
	workout: "Add a 45 minute workout",
	walk: "Walk 5,000 more steps",
	earlySleep: "Sleep an hour earlier",
	restDay: "Take a rest day",
	lateNight: "Late night out",
} as const;

export const PERSISTENCE_LABELS = {
	once: "Just today",
	habit: "Every day",
} as const;

export const WHAT_IF_LABELS = {
	title: "What if",
	/** Shown on a preset the subject has no data for. */
	noData: "No data for this change",
	/** Marks every simulated number or path, so it is never read as a forecast. */
	tag: "Simulated",
	path: "What-if path",
} as const;

/**
 * Names for places on the map, by where a place sits against the person's
 * usual level: first on activity, then on recovery. Every name means
 * "compared with your usual", never an absolute level.
 */
export const REGION_NAMES = {
	high: { high: "Strong stretch", middle: "More active", low: "Pushing hard" },
	middle: { high: "Well rested", middle: "Your usual", low: "Less rested" },
	low: { high: "Resting", middle: "Less active", low: "Worn out" },
} as const;

/** Shown once on the map, so the place names are read as relative. */
export const REGION_NOTE = "Places are named against your usual level";

export const REGION_THRESHOLD = 0.25;
