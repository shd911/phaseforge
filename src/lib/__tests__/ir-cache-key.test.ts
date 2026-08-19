// b141.24 (audit): clicking a legend checkbox on IR/Step re-ran the whole DSP
// — ~65 IPC calls (4 bands × measurement/target/corrected plus three sums),
// 0.3–1.5 s of frozen UI, for a change that only affects which curves are
// drawn. Those toggles now replay the renderer over cached curves.
//
// The cache key is the safety property: it must change whenever the curves
// themselves would differ. A missed dependency here means the chart shows
// stale impulses — worse than the slowness the cache removes.

import { describe, it, expect } from "vitest";
import { irDataCacheKey } from "../plot-helpers";

const base = {
  mode: "ir",
  sumMode: true,
  bandId: null as string | null,
  excludedBands: [] as string[],
  bandsVersion: 7,
  sampleRate: 48_000,
};

describe("irDataCacheKey (b141.24)", () => {
  it("is stable for identical inputs", () => {
    expect(irDataCacheKey(base)).toBe(irDataCacheKey({ ...base }));
  });

  it("ignores the order bands were excluded in", () => {
    const a = irDataCacheKey({ ...base, excludedBands: ["Woofer", "Tweeter"] });
    const b = irDataCacheKey({ ...base, excludedBands: ["Tweeter", "Woofer"] });
    expect(a).toBe(b);
  });

  it("changes when a band leaves the coherent sum", () => {
    expect(irDataCacheKey({ ...base, excludedBands: ["Woofer"] })).not.toBe(irDataCacheKey(base));
  });

  it("changes on any band edit (bandsVersion bumps in markDirty)", () => {
    expect(irDataCacheKey({ ...base, bandsVersion: 8 })).not.toBe(irDataCacheKey(base));
  });

  it("changes when the export sample rate changes", () => {
    expect(irDataCacheKey({ ...base, sampleRate: 96_000 })).not.toBe(irDataCacheKey(base));
  });

  it("separates SUM from a band, and one band from another", () => {
    const sum = irDataCacheKey(base);
    const b1 = irDataCacheKey({ ...base, sumMode: false, bandId: "band-1" });
    const b2 = irDataCacheKey({ ...base, sumMode: false, bandId: "band-2" });
    expect(new Set([sum, b1, b2]).size).toBe(3);
  });

  it("separates the IR and Step tabs", () => {
    expect(irDataCacheKey({ ...base, mode: "step" })).not.toBe(irDataCacheKey(base));
  });

  it("does not depend on the band id while in SUM", () => {
    expect(irDataCacheKey({ ...base, bandId: "band-9" })).toBe(irDataCacheKey(base));
  });
});
