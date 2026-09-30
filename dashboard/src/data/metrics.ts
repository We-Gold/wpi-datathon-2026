import type { MetricInfo } from "../types";

/**
 * The daily features the cards show, in the order shown. Names and units match
 * CARD_METRICS in src/health/export_cli.py.
 */
export const METRICS: MetricInfo[] = [
	{
		metric: "steps",
		label: "Steps",
		kind: "total",
		domain: "activity",
		displayUnit: "steps",
		direction: "up",
	},
	{
		metric: "resting_hr_bpm",
		label: "Resting heart rate",
		kind: "level",
		domain: "cardiac",
		displayUnit: "bpm",
		direction: "down",
	},
	{
		metric: "sleep_hours",
		label: "Sleep",
		kind: "total",
		domain: "sleep",
		displayUnit: "h",
		direction: "up",
	},
	{
		metric: "hrv_sdnn_ms",
		label: "Heart rate variability",
		kind: "level",
		domain: "cardiac",
		displayUnit: "ms",
		direction: "up",
	},
	{
		metric: "active_energy_kcal",
		label: "Active energy",
		kind: "total",
		domain: "energy",
		displayUnit: "kcal",
		direction: "up",
	},
	{
		metric: "exercise_minutes",
		label: "Exercise time",
		kind: "total",
		domain: "activity",
		displayUnit: "min",
		direction: "up",
	},
];
