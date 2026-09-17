// b141.39: Export/GD rebuilt their chart with a hard-coded 20 Hz–30 kHz X
// range, so any parameter change (taps, rate, window, band edits) reset zoom.
import { describe, it, expect } from "vitest";
import { keptLogXRange } from "../plot-helpers";

describe("keptLogXRange", () => {
  it("keeps a saved frequency zoom", () => {
    expect(keptLogXRange(200, 2000, 20, 30000)).toEqual({ min: 200, max: 2000 });
  });
  it("falls back when nothing was saved", () => {
    expect(keptLogXRange(null, null, 20, 30000)).toEqual({ min: 20, max: 30000 });
  });
  it("ignores a range saved from a linear time chart (IR/Step ms)", () => {
    expect(keptLogXRange(-30, 30, 20, 30000)).toEqual({ min: 20, max: 30000 });
    expect(keptLogXRange(5, 5, 20, 30000)).toEqual({ min: 20, max: 30000 });
  });
});
