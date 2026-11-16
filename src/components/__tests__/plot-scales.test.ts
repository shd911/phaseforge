// b141.150: uPlot turns a static `range: [min, max]` into a constant function,
// so setScale on that scale is silently ignored — the GD tab's ± / ▲▼ / FIT
// buttons did nothing. Every zoomable scale must read mutable bounds.
import { describe, it, expect } from "vitest";

// No Node typings in the repo — fs through a non-literal specifier (as export-metrics.test.ts).
const FS_MODULE = "node:fs";
const fs: { readFileSync(u: URL, enc: "utf8"): string } = await import(/* @vite-ignore */ FS_MODULE);
const HERE = import.meta.url;

describe("plot scales", () => {
  it("no scale of the plot has a static range array", () => {
    const src = fs.readFileSync(new URL("../FrequencyPlot.tsx", HERE), "utf8");
    expect(src.match(/range:\s*\[/g) ?? []).toEqual([]);
  });
});

import { zoomSpanAllowed, ZOOM_MAX_SPAN_DB, ZOOM_MAX_SPAN_GD_MS } from "../../lib/plot-helpers";

describe("vertical zoom limits (b141.150)", () => {
  it("zooming in is allowed from a range wider than the cap", () => {
    // Real GD tab: −780..+124 ms; «+» (×0.6) gives 542 ms — refused before.
    expect(zoomSpanAllowed(904 * 0.6, 0.6, ZOOM_MAX_SPAN_DB)).toBe(true);
    expect(zoomSpanAllowed(904 * 0.6, 0.6, ZOOM_MAX_SPAN_GD_MS)).toBe(true);
  });
  it("zooming out stops at the cap, zooming in at 0.01", () => {
    expect(zoomSpanAllowed(600, 1.6, ZOOM_MAX_SPAN_DB)).toBe(false);
    expect(zoomSpanAllowed(1446, 1.6, ZOOM_MAX_SPAN_GD_MS)).toBe(true);
    expect(zoomSpanAllowed(0.005, 0.6, ZOOM_MAX_SPAN_DB)).toBe(false);
  });
});
