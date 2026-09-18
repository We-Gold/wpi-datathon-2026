import { format, max, scaleLinear } from "d3";
import { AXIS_LABELS, FACTOR_LABELS } from "../config";
import { horizontalBar } from "../lib/shapes";
import { useElementWidth } from "../lib/useElementWidth";
import type { Axis, Contribution } from "../types";

interface Props {
	contributions: Contribution[];
}

const ROW = 26;
const BAR = 14;
const LABEL_WIDTH = 150;
const VALUE_ROOM = 48;
const formatValue = format("+.3f");
const AXES: Axis[] = ["activity", "recovery"];

/**
 * What made up today's velocity, one group per axis. Both groups share one
 * scale so a bar on one axis can be compared with a bar on the other.
 */
export function ContributionBars({ contributions }: Props) {
	const [containerRef, width] = useElementWidth<HTMLDivElement>();
	const extent =
		max(contributions, (c) => Math.max(Math.abs(c.activity), Math.abs(c.recovery))) || 1;
	const x = scaleLinear()
		.domain([-extent, extent])
		.range([LABEL_WIDTH + VALUE_ROOM, Math.max(LABEL_WIDTH + VALUE_ROOM + 40, width - VALUE_ROOM)]);
	const zero = x(0);

	return (
		<div ref={containerRef} className="contributions">
			{width > 0 &&
				AXES.map((axis) => {
					const rows = contributions.filter((c) => c[axis] !== 0);
					const total = contributions.reduce((sum, c) => sum + c[axis], 0);
					const height = (rows.length + 1) * ROW + 4;
					return (
						<figure key={axis} className="contribution-group">
							<figcaption>
								<span className={`axis-key axis-key-${axis}`}>■</span> {AXIS_LABELS[axis].velocity}
							</figcaption>
							<svg
								width={width}
								height={height}
								role="img"
								aria-label={`${AXIS_LABELS[axis].velocity} by factor, net ${formatValue(total)}`}
							>
								<line className="baseline" x1={zero} x2={zero} y1={0} y2={rows.length * ROW} />
								{rows.map((c, i) => {
									const value = c[axis];
									const end = x(value);
									const rowY = i * ROW;
									return (
										<g key={c.factor}>
											<title>{`${FACTOR_LABELS[c.factor]}: ${formatValue(value)}`}</title>
											<text className="row-label" x={0} y={rowY + ROW / 2} dy="0.32em">
												{FACTOR_LABELS[c.factor]}
											</text>
											<path
												className={`bar bar-${axis}`}
												d={horizontalBar(zero, end, rowY + (ROW - BAR) / 2, BAR)}
											/>
											<text
												className="value-label"
												x={end + (value >= 0 ? 6 : -6)}
												y={rowY + ROW / 2}
												dy="0.32em"
												textAnchor={value >= 0 ? "start" : "end"}
											>
												{formatValue(value)}
											</text>
										</g>
									);
								})}
								<line
									className="divider"
									x1={0}
									x2={width}
									y1={rows.length * ROW + 2}
									y2={rows.length * ROW + 2}
								/>
								<text
									className="row-label row-total"
									x={0}
									y={rows.length * ROW + ROW / 2 + 4}
									dy="0.32em"
								>
									Net
								</text>
								<text
									className="value-label row-total"
									x={zero}
									y={rows.length * ROW + ROW / 2 + 4}
									dy="0.32em"
									textAnchor="middle"
								>
									{formatValue(total)}
								</text>
							</svg>
						</figure>
					);
				})}
		</div>
	);
}
