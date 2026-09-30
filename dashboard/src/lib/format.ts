import { format } from "d3";
import type { MetricInfo } from "../types";

export function valueFormatter(info: MetricInfo) {
	const f =
		info.displayUnit === "steps" || info.displayUnit === "kcal" ? format(",.0f") : format(",.1f");
	return (v: number) => f(v);
}

/** Whether a change is good or bad news for this metric. */
export function deltaTone(info: MetricInfo, delta: number | undefined) {
	if (delta === undefined || delta === 0 || info.direction === "neutral") return "neutral";
	return delta > 0 === (info.direction === "up") ? "good" : "bad";
}

export function deltaArrow(delta: number) {
	return delta > 0 ? "▲" : delta < 0 ? "▼" : "•";
}
