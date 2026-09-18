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
	/** Canonical metric name, as written by the Python pipeline. */
	metric: string;
	label: string;
	domain: Domain;
	/** Canonical unit the values are stored in. */
	unit: string;
	/** Unit shown to people, with `toDisplay` converting from the canonical one. */
	displayUnit: string;
	toDisplay: (value: number) => number;
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

export interface DashboardData {
	subjects: Subject[];
	metrics: MetricInfo[];
	/** Keyed by subject id. */
	series: Record<string, MetricSeries[]>;
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

export interface Cluster {
	center: AxisVector;
	/** 2x2, rows and columns in [activity, recovery] order. */
	covariance: [[number, number], [number, number]];
	/** Decided by the server. */
	kind: "healthy" | "unhealthy";
}

export interface Contribution {
	factor: FactorKey;
	/** Signed. The values on each axis add up to that axis's velocity. */
	activity: number;
	recovery: number;
}

/** What the server returns for one subject. */
export interface TrajectoryResponse {
	subjectId: string;
	/** The present day, YYYY-MM-DD. The last history entry is this day. */
	asOf: string;
	/** Oldest first. */
	history: TrajectoryDay[];
	/** Oldest first, starting the day after asOf. */
	prediction: PredictedDay[];
	clusters: Cluster[];
	/** Today's velocity, split by factor. */
	contributions: Contribution[];
}
