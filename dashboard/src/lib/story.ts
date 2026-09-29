import { AXIS_LABELS, PREDICTION_DAYS, REGION_THRESHOLD, REGION_WORDS } from "../config";
import type { Axis, AxisVector, Cluster, TrajectoryResponse } from "../types";
import { heightAt } from "./terrain";

const AXES: Axis[] = ["activity", "recovery"];

/** A name for a place on the map, from which side of each axis it is on. */
export function regionName(p: AxisVector): string {
	const words = AXES.flatMap((axis) => {
		if (p[axis] > REGION_THRESHOLD) return [REGION_WORDS[axis].high];
		if (p[axis] < -REGION_THRESHOLD) return [REGION_WORDS[axis].low];
		return [];
	});
	const name = words.length > 0 ? words.join(" and ") : REGION_WORDS.middle;
	return name[0].toUpperCase() + name.slice(1);
}

/** Height of the tallest peak or deepest valley, so heights can be compared. */
function reliefScale(clusters: Cluster[]) {
	return Math.max(
		...clusters.map((c) => Math.abs(heightAt(clusters, c.center.activity, c.center.recovery))),
		1e-9,
	);
}

/** Share of the map's relief that counts as a real climb or slide. */
const CLIMB = 0.08;
/** Weekly change on an axis, in axis units, that counts as a real move. */
const MOVE = 0.05;
const WEEK = 7;

function moveWord(delta: number) {
	if (delta > MOVE) return "rose";
	if (delta < -MOVE) return "fell";
	return "held steady";
}

export interface Story {
	headline: string;
	dek: string;
}

/**
 * The page's headline in plain words: where the prediction is taking the
 * person, then what changed this week and what today did.
 */
export function buildStory({ history, prediction, clusters }: TrajectoryResponse): Story | null {
	const today = history.at(-1);
	const weekAgo = history.at(-1 - WEEK);
	const end = prediction.at(-1);
	if (!today || !weekAgo || !end) return null;

	const scale = reliefScale(clusters);
	const height = (p: AxisVector) => heightAt(clusters, p.activity, p.recovery) / scale;
	const climb = height(end.position) - height(today.position);
	const destination = regionName(end.position).toLowerCase();
	const headline =
		climb > CLIMB
			? `You are climbing toward healthier ground`
			: climb < -CLIMB
				? `You are sliding toward a low point`
				: `You are holding steady`;

	const [a, r] = AXES.map((axis) => moveWord(today.position[axis] - weekAgo.position[axis]));
	const week =
		a === r
			? `Over the past week, ${AXIS_LABELS.activity.position} and ${AXIS_LABELS.recovery.position} both ${a}.`
			: `Over the past week, ${AXIS_LABELS.activity.position} ${a} and ${AXIS_LABELS.recovery.position} ${r}.`;
	const ahead = `In ${PREDICTION_DAYS} days the model expects you in ${destination} territory.`;
	return { headline, dek: `${week} ${ahead}` };
}

/** Which way today pushed, in words, for the margin note. */
export function pushSentence(v: AxisVector): string {
	const [main, other] =
		Math.abs(v.activity) >= Math.abs(v.recovery)
			? (["activity", "recovery"] as const)
			: (["recovery", "activity"] as const);
	const label = (axis: Axis) => AXIS_LABELS[axis].position;
	const dir = (axis: Axis) => (v[axis] >= 0 ? "more" : "less");
	const lead = `Today pushed you toward ${dir(main)} ${label(main)}`;
	return Math.abs(v[other]) > Math.abs(v[main]) * 0.5
		? `${lead}, and a little toward ${dir(other)} ${label(other)}.`
		: `${lead}.`;
}
