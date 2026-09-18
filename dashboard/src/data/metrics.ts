import type { MetricInfo } from "../types";

const Q = "HKQuantityTypeIdentifier";

const identity = (v: number) => v;

/** The subset of the catalogue the prototype shows. The big three come first. */
export const METRICS: MetricInfo[] = [
	{
		metric: `${Q}StepCount`,
		label: "Steps",
		kind: "total",
		domain: "activity",
		unit: "count",
		displayUnit: "steps",
		toDisplay: identity,
		direction: "up",
	},
	{
		metric: `${Q}RestingHeartRate`,
		label: "Resting heart rate",
		kind: "level",
		domain: "cardiac",
		unit: "count/min",
		displayUnit: "bpm",
		toDisplay: identity,
		direction: "down",
	},
	{
		metric: "SleepDurationDaily",
		label: "Sleep",
		kind: "total",
		domain: "sleep",
		unit: "s",
		displayUnit: "h",
		toDisplay: (v) => v / 3600,
		direction: "up",
	},
	{
		metric: `${Q}HeartRateVariabilitySDNN`,
		label: "Heart rate variability",
		kind: "level",
		domain: "cardiac",
		unit: "s",
		displayUnit: "ms",
		toDisplay: (v) => v * 1000,
		direction: "up",
	},
	{
		metric: `${Q}ActiveEnergyBurned`,
		label: "Active energy",
		kind: "total",
		domain: "energy",
		unit: "kcal",
		displayUnit: "kcal",
		toDisplay: identity,
		direction: "up",
	},
	{
		metric: `${Q}AppleExerciseTime`,
		label: "Exercise time",
		kind: "total",
		domain: "activity",
		unit: "s",
		displayUnit: "min",
		toDisplay: (v) => v / 60,
		direction: "up",
	},
];
