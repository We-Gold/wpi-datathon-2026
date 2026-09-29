import type { FACTOR_LABELS } from "./config";

/** Mirrors the eleven domains in the Python metric catalogue. */
export type Domain =
	| "activity"
	| "body"
	| "cardiac"
	| "energy"
	| "exposure"
	| "hygiene"
	| "mobility"
	| "nutrition"
	| "sleep"
	| "wellbeing"
	| "workout";

export interface MetricInfo {
	/** Daily feature name, as written by the Python pipeline. */
	metric: string;
	label: string;
	domain: Domain;
	/** The pipeline already stores daily features in this unit. */
	displayUnit: string;
	/**
	 * A total adds up over the day (steps), so its daily chart is bars from zero.
	 * A level is sampled (resting heart rate), so it is dots on a fitted scale.
	 */
	kind: "total" | "level";
	/** Whether a rising value is good, bad, or neither. Drives delta wording only. */
	direction: "up" | "down" | "neutral";
}

export interface DailyPoint {
	/** Local calendar day, YYYY-MM-DD. */
	date: string;
	value: number;
}

export interface MetricSeries {
	metric: string;
	points: DailyPoint[];
}

export interface Subject {
	subjectId: string;
	label: string;
}

/** `index.json` from `health-export`. */
export interface DashboardIndex {
	subjects: Subject[];
	model: {
		/** Sign of each ingredient on each axis, from AXIS_FEATURES in Python. */
		axes: Record<Axis, Record<string, number>>;
	};
}

/** How a change to a raw feature becomes a change in a score, for one subject. */
export interface Scaling {
	/** The subject's interquartile range of each feature. Missing means no data. */
	iqr: Record<string, number>;
	/** Ingredients observed on the last full day. A score is their mean. */
	counts: AxisVector;
}

/** `subjects/<id>.json` from `health-export`. */
export interface SubjectData {
	trajectory: TrajectoryResponse;
	series: MetricSeries[];
	scaling: Scaling;
}

/** The two axes of the map. Labels for them are in config.ts. */
export type Axis = "activity" | "recovery";

export type AxisVector = Record<Axis, number>;

export type FactorKey = keyof typeof FACTOR_LABELS;

export interface TrajectoryDay {
	/** YYYY-MM-DD */
	date: string;
	position: AxisVector;
	velocity: AxisVector;
}

export interface PredictedDay {
	date: string;
	position: AxisVector;
}

export interface ForecastDay extends PredictedDay {
	/** The 80% range on each axis, from held-out errors at this horizon. */
	low: AxisVector;
	high: AxisVector;
}

export interface Cluster {
	center: AxisVector;
	/** 2x2, rows and columns in [activity, recovery] order. */
	covariance: [[number, number], [number, number]];
	/** Decided by the server. */
	kind: "healthy" | "unhealthy";
}

export interface Contribution {
	factor: FactorKey;
	/** Signed. The values on each axis add up exactly to that axis's velocity. */
	activity: number;
	recovery: number;
}

/** What the server returns for one subject. */
export interface TrajectoryResponse {
	subjectId: string;
	/** The present day, YYYY-MM-DD: the last day with data. The last history entry is this day. */
	asOf: string;
	/** Oldest first. */
	history: TrajectoryDay[];
	/**
	 * Position is a moving average of the daily score. Each day keeps this share
	 * of yesterday's position and takes the rest from today's score.
	 */
	keep: number;
	/** Oldest first, starting the day after asOf. */
	prediction: ForecastDay[];
	clusters: Cluster[];
	/** Today's velocity, split by factor. */
	contributions: Contribution[];
}
