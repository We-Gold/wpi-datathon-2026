import { interpolateLab, piecewise, scaleDiverging } from "d3";
import type { Cluster } from "../types";

/**
 * Height of the landscape at a point. Each cluster is a 2D Gaussian with the
 * same volume, so a tight cluster makes a tall narrow peak and a spread out one
 * makes a low broad hill. Healthy clusters add height, unhealthy ones remove it.
 */
export function heightAt(clusters: Cluster[], activity: number, recovery: number): number {
	let height = 0;
	for (const { center, covariance, kind } of clusters) {
		const [[a, b], [, d]] = covariance;
		const det = a * d - b * b;
		if (det <= 0) continue;
		const dx = activity - center.activity;
		const dy = recovery - center.recovery;
		// Mahalanobis distance squared, with the 2x2 inverse written out.
		const m = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
		const density = Math.exp(-0.5 * m) / (2 * Math.PI * Math.sqrt(det));
		height += kind === "healthy" ? density : -density;
	}
	return height;
}

/** Stops from deep valley, through neutral ground, to high peak. */
export const TERRAIN_STOPS = ["#7890a0", "#c3cfd3", "#f5f1e8", "#efc69c", "#cf7a3e"];

export function terrainColor(maxAbs: number) {
	return scaleDiverging(piecewise(interpolateLab, TERRAIN_STOPS)).domain([-maxAbs, 0, maxAbs]);
}
