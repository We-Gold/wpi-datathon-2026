import { timeDay, timeFormat } from "d3";
import type { DashboardData, MetricSeries, Subject } from "../types";
import { METRICS } from "./metrics";
import { mulberry32 } from "./random";

const DAYS = 180;
const formatDay = timeFormat("%Y-%m-%d");

interface Shape {
	base: number;
	/** Total drift across the whole window. */
	trend: number;
	noise: number;
	/** Amplitude of a weekly cycle. */
	weekly: number;
	min: number;
	max: number;
	round?: (v: number) => number;
	/** Chance a day has no value. */
	missing: number;
}

const SHAPES: Record<string, Shape> = {
	HKQuantityTypeIdentifierStepCount: {
		base: 7500,
		trend: 1500,
		noise: 2200,
		weekly: 1200,
		min: 300,
		max: 30000,
		round: Math.round,
		missing: 0.03,
	},
	HKQuantityTypeIdentifierRestingHeartRate: {
		base: 62,
		trend: -3,
		noise: 2.5,
		weekly: 0.8,
		min: 40,
		max: 100,
		round: Math.round,
		missing: 0.08,
	},
	HKQuantityTypeIdentifierHeartRateVariabilitySDNN: {
		base: 0.045,
		trend: 0.006,
		noise: 0.009,
		weekly: 0.002,
		min: 0.01,
		max: 0.15,
		missing: 0.1,
	},
	SleepDurationDaily: {
		base: 7 * 3600,
		trend: 0.3 * 3600,
		noise: 0.8 * 3600,
		weekly: 0.5 * 3600,
		min: 3 * 3600,
		max: 11 * 3600,
		missing: 0.06,
	},
	HKQuantityTypeIdentifierActiveEnergyBurned: {
		base: 520,
		trend: 60,
		noise: 180,
		weekly: 90,
		min: 50,
		max: 2000,
		round: Math.round,
		missing: 0.03,
	},
	HKQuantityTypeIdentifierAppleExerciseTime: {
		base: 32 * 60,
		trend: 6 * 60,
		noise: 20 * 60,
		weekly: 10 * 60,
		min: 0,
		max: 180 * 60,
		missing: 0.03,
	},
};

const SUBJECTS: Subject[] = [
	{ subjectId: "subject_01", label: "Subject 01" },
	{ subjectId: "subject_02", label: "Subject 02" },
	{ subjectId: "subject_03", label: "Subject 03" },
];

function makeSeries(
	metric: string,
	shape: Shape,
	days: Date[],
	random: () => number,
	personal: number,
): MetricSeries {
	const points = [];
	// A slow random walk on top of the trend, so the lines are not just noise.
	let walk = 0;
	for (const [i, day] of days.entries()) {
		walk = walk * 0.9 + (random() - 0.5) * shape.noise * 0.4;
		if (random() < shape.missing) continue;
		const progress = i / (days.length - 1);
		const weekday = Math.sin((day.getDay() / 7) * Math.PI * 2);
		const raw =
			shape.base * personal +
			shape.trend * progress +
			shape.weekly * weekday +
			walk +
			(random() - 0.5) * shape.noise;
		const clamped = Math.min(shape.max, Math.max(shape.min, raw));
		points.push({
			date: formatDay(day),
			value: shape.round ? shape.round(clamped) : clamped,
		});
	}
	return { metric, points };
}

export function buildMockData(today = new Date()): DashboardData {
	const end = timeDay.floor(today);
	const days = timeDay.range(timeDay.offset(end, -DAYS + 1), timeDay.offset(end, 1));

	const series: DashboardData["series"] = {};
	for (const [s, subject] of SUBJECTS.entries()) {
		const random = mulberry32(1000 + s);
		series[subject.subjectId] = METRICS.map((m) =>
			makeSeries(m.metric, SHAPES[m.metric], days, random, 0.9 + random() * 0.2),
		);
	}
	return { subjects: SUBJECTS, metrics: METRICS, series };
}
