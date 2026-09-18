import type { DashboardData, TrajectoryResponse } from "../types";
import { buildMockData } from "./mock";
import { buildMockTrajectory } from "./mockTrajectory";

/*
 * The only places the dashboard gets data from. Both return mock data for now.
 * When the FastAPI server in src/ exists, these become fetches against it and
 * nothing that calls them has to change.
 */

export async function loadDashboardData(): Promise<DashboardData> {
	return buildMockData();
}

export async function loadTrajectory(subjectId: string): Promise<TrajectoryResponse> {
	const seed = [...subjectId].reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
	return buildMockTrajectory(subjectId, seed);
}
