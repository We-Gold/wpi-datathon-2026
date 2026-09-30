import { mean, timeDay, timeParse } from "d3";
import type { DailyPoint } from "../types";

export const parseDay = timeParse("%Y-%m-%d") as (s: string) => Date;

export interface DatedValue {
	date: Date;
	value: number;
}

/** Trailing mean over the previous `window` calendar days, missing days skipped. */
export function rollingMean(points: DatedValue[], window: number): DatedValue[] {
	const out: DatedValue[] = [];
	let start = 0;
	let sum = 0;
	for (const [i, p] of points.entries()) {
		sum += p.value;
		const cutoff = timeDay.offset(p.date, -window + 1);
		while (points[start].date < cutoff) {
			sum -= points[start].value;
			start++;
		}
		out.push({ date: p.date, value: sum / (i - start + 1) });
	}
	return out;
}

/** Mean of the last `days` days against the `days` before them. */
export function periodChange(points: DatedValue[], end: Date, days: number) {
	const split = timeDay.offset(end, -days + 1);
	const prevStart = timeDay.offset(split, -days);
	const current = mean(
		points.filter((p) => p.date >= split),
		(p) => p.value,
	);
	const previous = mean(
		points.filter((p) => p.date >= prevStart && p.date < split),
		(p) => p.value,
	);
	return { current, previous };
}

export function toDated(points: DailyPoint[]): DatedValue[] {
	return points.map((p) => ({ date: parseDay(p.date), value: p.value }));
}
