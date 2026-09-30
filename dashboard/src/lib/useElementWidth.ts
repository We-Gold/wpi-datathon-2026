import { type RefObject, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

/**
 * Tracks an element's content width, so charts can size to their container.
 * Widths are applied before paint, so a chart never shows a frame at the old
 * size. A zero width (a hidden tab) is ignored, so the chart keeps its last
 * size and is ready the moment it is shown again.
 */
export function useElementWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
	const ref = useRef<T>(null);
	const [width, setWidth] = useState(0);
	useLayoutEffect(() => {
		const el = ref.current;
		if (!el) return;
		const update = (next: number) => {
			if (next > 0) setWidth(Math.floor(next));
		};
		update(el.getBoundingClientRect().width);
		const observer = new ResizeObserver(([entry]) => {
			// Resize callbacks run before paint, but React would render later.
			flushSync(() => update(entry.contentRect.width));
		});
		observer.observe(el);
		return () => observer.disconnect();
	}, []);
	return [ref, width];
}
