import { timeDay, timeFormat } from "d3";
import { HISTORY_DAYS, PREDICTION_DAYS } from "../config";
import type {
	AxisVector,
	Cluster,
	Contribution,
	FactorKey,
	PredictedDay,
	TrajectoryDay,
	TrajectoryResponse,
} from "../types";
import { mulberry32 } from "./random";

const formatDay = timeFormat("%Y-%m-%d");

/** Legs the mock history path is split into. */
const WAYPOINTS = 5;

const cluster = (
	activity: number,
	recovery: number,
	[sa, sb, sd]: [number, number, number],
	kind: Cluster["kind"],
): Cluster => ({
	center: { activity, recovery },
	covariance: [
		[sa, sb],
		[sb, sd],
	],
	kind,
});

/**
 * The landscape every mock subject gets, before per-subject jitter. Healthy
 * and unhealthy clusters overlap on purpose, so the map has ridges, saddles,
 * and pits inside hills, not only separate peaks and valleys.
 */
const BASE_CLUSTERS: Cluster[] = [
	cluster(1.1, 0.9, [0.18, 0.04, 0.14], "healthy"),
	cluster(-0.1, 1.5, [0.45, 0, 0.3], "healthy"),
	cluster(2.1, 1.9, [0.1, -0.02, 0.12], "healthy"),
	cluster(0.2, -0.3, [0.35, 0.15, 0.2], "healthy"),
	cluster(-1.9, 1.9, [0.2, 0.08, 0.25], "healthy"),
	cluster(1.8, -1.1, [0.16, -0.05, 0.22], "unhealthy"),
	cluster(-1.3, -0.7, [0.55, 0.1, 0.4], "unhealthy"),
	cluster(0.55, 0.45, [0.1, 0.02, 0.09], "unhealthy"),
	cluster(-0.9, 0.7, [0.12, -0.04, 0.1], "unhealthy"),
	cluster(0.3, -2.0, [0.4, 0, 0.15], "unhealthy"),
];

/**
 * Rough share of each factor on each axis. Signs say which way a good day
 * pushes: steps build activity and cost a little recovery, sleep only helps
 * recovery, heart rate touches both.
 */
const FACTOR_WEIGHTS: Record<FactorKey, AxisVector> = {
	steps: { activity: 0.035, recovery: -0.008 },
	heartRate: { activity: 0.015, recovery: -0.018 },
	sleep: { activity: 0, recovery: 0.03 },
	hrv: { activity: 0, recovery: 0.012 },
	activeEnergy: { activity: 0.02, recovery: -0.006 },
	exerciseTime: { activity: 0.018, recovery: -0.01 },
};

const add = (a: AxisVector, b: AxisVector, k = 1): AxisVector => ({
	activity: a.activity + b.activity * k,
	recovery: a.recovery + b.recovery * k,
});

function jitterClusters(random: () => number): Cluster[] {
	return BASE_CLUSTERS.map((c) => ({
		...c,
		center: {
			activity: c.center.activity + (random() - 0.5) * 0.5,
			recovery: c.center.recovery + (random() - 0.5) * 0.5,
		},
	}));
}

function todaysContributions(random: () => number): Contribution[] {
	return (Object.keys(FACTOR_WEIGHTS) as FactorKey[]).map((factor) => {
		const w = FACTOR_WEIGHTS[factor];
		// A day can go against the usual direction, so allow a flipped sign.
		const scale = () => 1.6 * (random() - 0.25);
		return { factor, activity: w.activity * scale(), recovery: w.recovery * scale() };
	});
}

/**
 * Builds a plausible response. The server computes the real thing; the pull
 * toward the origin here only keeps the mock path on the map.
 */
export function buildMockTrajectory(subjectId: string, seed: number, today = new Date()) {
	const random = mulberry32(seed);
	const end = timeDay.floor(today);
	const clusters = jitterClusters(random);
	const contributions = todaysContributions(random);

	// The path visits a few clusters in turn, one leg each, like phases of life:
	// a slump, a training block, a stressful month.
	const waypoints: AxisVector[] = [];
	while (waypoints.length < WAYPOINTS) {
		const next = clusters[Math.floor(random() * clusters.length)].center;
		if (next !== waypoints.at(-1)) waypoints.push(next);
	}
	const legDays = Math.ceil(HISTORY_DAYS / WAYPOINTS);

	const history: TrajectoryDay[] = [];
	let position: AxisVector = {
		activity: -1.6 + random() * 0.6,
		recovery: -0.6 + random() * 0.6,
	};
	let velocity: AxisVector = { activity: 0, recovery: 0 };
	for (let i = HISTORY_DAYS - 1; i >= 0; i--) {
		const day = HISTORY_DAYS - 1 - i;
		const target = waypoints[Math.floor(day / legDays)];
		const noise = { activity: random() - 0.5, recovery: random() - 0.5 };
		// A damped spring toward the waypoint, a little underdamped so it curves.
		velocity = add(
			{ activity: velocity.activity * 0.9, recovery: velocity.recovery * 0.9 },
			add(target, position, -1),
			0.006,
		);
		velocity = add(velocity, noise, 0.02);
		if (i === 0) {
			velocity = contributions.reduce<AxisVector>((sum, c) => add(sum, c), {
				activity: 0,
				recovery: 0,
			});
		}
		position = add(position, velocity);
		history.push({ date: formatDay(timeDay.offset(end, -i)), position, velocity });
	}

	// Drift toward the closest healthy cluster, the way a model might expect.
	const target = clusters
		.filter((c) => c.kind === "healthy")
		.reduce((best, c) =>
			Math.hypot(c.center.activity - position.activity, c.center.recovery - position.recovery) <
			Math.hypot(best.center.activity - position.activity, best.center.recovery - position.recovery)
				? c
				: best,
		).center;
	const prediction: PredictedDay[] = [];
	let predicted = position;
	let predictedVelocity = velocity;
	for (let i = 1; i <= PREDICTION_DAYS; i++) {
		const pull = add(target, predicted, -1);
		predictedVelocity = add(
			{ activity: predictedVelocity.activity * 0.9, recovery: predictedVelocity.recovery * 0.9 },
			pull,
			0.006,
		);
		predicted = add(predicted, predictedVelocity);
		prediction.push({ date: formatDay(timeDay.offset(end, i)), position: predicted });
	}

	return {
		subjectId,
		asOf: formatDay(end),
		history,
		prediction,
		clusters,
		contributions,
	} satisfies TrajectoryResponse;
}
