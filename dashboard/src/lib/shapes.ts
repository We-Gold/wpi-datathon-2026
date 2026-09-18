/**
 * A horizontal bar from x0 to x1, square at x0 (the baseline) and rounded at
 * x1 (the data end). Works in either direction.
 */
export function horizontalBar(x0: number, x1: number, y: number, height: number, radius = 4) {
	const length = Math.abs(x1 - x0);
	if (length === 0) return "";
	const r = Math.min(radius, length, height / 2);
	const s = x1 > x0 ? 1 : -1;
	const top = y;
	const bottom = y + height;
	return [
		`M${x0},${top}`,
		`H${x1 - s * r}`,
		`Q${x1},${top} ${x1},${top + r}`,
		`V${bottom - r}`,
		`Q${x1},${bottom} ${x1 - s * r},${bottom}`,
		`H${x0}`,
		"Z",
	].join("");
}

/** A vertical bar rising from a baseline at y0 to y1, rounded at y1. */
export function verticalBar(x: number, width: number, y0: number, y1: number, radius = 4) {
	const length = Math.abs(y1 - y0);
	if (length === 0) return "";
	const r = Math.min(radius, length, width / 2);
	const s = y1 < y0 ? 1 : -1;
	const left = x;
	const right = x + width;
	return [
		`M${left},${y0}`,
		`V${y1 + s * r}`,
		`Q${left},${y1} ${left + r},${y1}`,
		`H${right - r}`,
		`Q${right},${y1} ${right},${y1 + s * r}`,
		`V${y0}`,
		"Z",
	].join("");
}
