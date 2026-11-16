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
