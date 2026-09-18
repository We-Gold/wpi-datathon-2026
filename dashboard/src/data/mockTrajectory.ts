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

/** The landscape every mock subject gets, before per-subject jitter. */
const BASE_CLUSTERS: Cluster[] = [
	{
		center: { activity: 1.1, recovery: 0.9 },
		covariance: [
			[0.18, 0.04],
			[0.04, 0.14],
		],
		kind: "healthy",
	},
	{
		center: { activity: -0.1, recovery: 1.5 },
		covariance: [
			[0.45, 0],
			[0, 0.3],
		],
		kind: "healthy",
	},
	{
		center: { activity: 1.8, recovery: -1.1 },
		covariance: [
			[0.16, -0.05],
			[-0.05, 0.22],
		],
		kind: "unhealthy",
	},
	{
		center: { activity: -1.3, recovery: -0.7 },
		covariance: [
			[0.55, 0.1],
			[0.1, 0.4],
		],
		kind: "unhealthy",
	},
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

	const history: TrajectoryDay[] = [];
	let position: AxisVector = {
		activity: -1.4 + random() * 0.6,
		recovery: 0.2 + random() * 0.6,
	};
	let velocity: AxisVector = { activity: 0.02, recovery: 0.01 };
	for (let i = HISTORY_DAYS - 1; i >= 0; i--) {
		const noise = { activity: random() - 0.45, recovery: random() - 0.5 };
		velocity = add(
			{ activity: velocity.activity * 0.85, recovery: velocity.recovery * 0.85 },
			noise,
			0.02,
		);
		velocity = add(velocity, position, -0.004);
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
