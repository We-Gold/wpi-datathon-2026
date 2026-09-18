import { format, scaleLinear } from "d3";
import { AXIS_LABELS } from "../config";
import { useElementWidth } from "../lib/useElementWidth";
import type { AxisVector } from "../types";

interface Props {
	velocity: AxisVector;
}

const PAD = 34;
const formatValue = format("+.3f");

/**
 * Today's velocity on the same axes as the map, drawn from the origin. The two
 * colored legs are the part on each axis.
 */
export function VelocityPlot({ velocity }: Props) {
	const [containerRef, width] = useElementWidth<HTMLDivElement>();
	const size = Math.min(width, 320);
	const extent = Math.max(Math.abs(velocity.activity), Math.abs(velocity.recovery)) * 1.35 || 1;
	const scale = scaleLinear()
		.domain([-extent, extent])
		.range([PAD, size - PAD]);
	const cx = scale(0);
	const tipX = scale(velocity.activity);
	// Screen y grows downward, so recovery is mirrored around the center.
	const tipY = size - scale(velocity.recovery);

	return (
		<div ref={containerRef} className="chart velocity-plot">
			{size > 0 && (
				<svg
					width={size}
					height={size}
					role="img"
					aria-label={`Today: ${AXIS_LABELS.activity.velocity} ${formatValue(velocity.activity)}, ${AXIS_LABELS.recovery.velocity} ${formatValue(velocity.recovery)}`}
				>
					<defs>
						<marker
							id="velocity-arrow"
							viewBox="0 0 10 10"
							refX="8"
							refY="5"
							markerWidth="7"
							markerHeight="7"
							orient="auto-start-reverse"
						>
							<path d="M0,1 L9,5 L0,9 Z" className="arrow-head" />
						</marker>
					</defs>

					<line className="baseline" x1={PAD} x2={size - PAD} y1={cx} y2={cx} />
					<line className="baseline" x1={cx} x2={cx} y1={PAD} y2={size - PAD} />
					<text
						className="axis-title"
						x={size - PAD}
						y={velocity.recovery >= 0 ? cx + 18 : cx - 10}
						textAnchor="end"
					>
						{AXIS_LABELS.activity.velocity} →
					</text>
					<text
						className="axis-title"
						x={velocity.activity >= 0 ? cx - 8 : cx + 8}
						y={PAD - 12}
						textAnchor={velocity.activity >= 0 ? "end" : "start"}
					>
						↑ {AXIS_LABELS.recovery.velocity}
					</text>

					<line className="projection" x1={tipX} x2={tipX} y1={tipY} y2={cx} />
					<line className="projection" x1={tipX} x2={cx} y1={tipY} y2={tipY} />
					<line className="leg leg-activity" x1={cx} x2={tipX} y1={cx} y2={cx} />
					<line className="leg leg-recovery" x1={cx} x2={cx} y1={cx} y2={tipY} />
					<line
						className="velocity-arrow"
						x1={cx}
						y1={cx}
						x2={tipX}
						y2={tipY}
						markerEnd="url(#velocity-arrow)"
					/>
					<circle className="origin-dot" cx={cx} cy={cx} r={4} />

					<text
						className="value-label"
						x={tipX}
						y={velocity.recovery >= 0 ? cx - 10 : cx + 18}
						textAnchor="middle"
					>
						{formatValue(velocity.activity)}
					</text>
					<text
						className="value-label"
						x={velocity.activity >= 0 ? cx - 8 : cx + 8}
						y={tipY}
						dy="0.32em"
						textAnchor={velocity.activity >= 0 ? "end" : "start"}
					>
						{formatValue(velocity.recovery)}
					</text>
				</svg>
			)}
		</div>
	);
}
