import type { DashboardIndex, SubjectData } from "../types";

/*
 * The only places the dashboard gets data from. The files are written by
 * `uv run health-export` into public/data/, which is gitignored.
 */

const DATA_URL = `${import.meta.env.BASE_URL}data/`;

async function getJson<T>(path: string): Promise<T> {
	const response = await fetch(`${DATA_URL}${path}`);
	if (!response.ok) {
		throw new Error(`Could not load ${path} (${response.status}). Run \`uv run health-export\`.`);
	}
	return response.json() as Promise<T>;
}

export function loadIndex(): Promise<DashboardIndex> {
	return getJson("index.json");
}

export function loadSubject(subjectId: string): Promise<SubjectData> {
	return getJson(`subjects/${encodeURIComponent(subjectId)}.json`);
}
