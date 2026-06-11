// b141.6 (audit, simplification): 5 near-identical binary-search interp
// implementations consolidated into grid.ts::interpOnGrid. Each former site
// maps to an option combo — this suite locks down every combo so the
// mechanical call-site translation is provably behavior-preserving.

import { describe, it, expect } from "vitest";
import { interpOnGrid, resampleOnLogGrid } from "../band-evaluator/grid";

const src = [100, 200, 400, 800];
const vals = [0, 10, 20, 30];

describe("interpOnGrid (b141.6)", () => {
  it("linear-space interpolation between points", () => {
    const [v] = interpOnGrid(src, vals, [150]);
    expect(v).toBeCloseTo(5, 10); // halfway 100..200 in linear f
  });

  it("log-space interpolation between points", () => {
    const [v] = interpOnGrid(src, vals, [Math.sqrt(100 * 200)], { logSpace: true });
    expect(v).toBeCloseTo(5, 10); // halfway in log f
  });

  it("outside=clamp holds boundary values (default)", () => {
    const out = interpOnGrid(src, vals, [50, 1600]);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(30);
  });

  it("outside=nan yields NaN beyond the source range", () => {
    const out = interpOnGrid(src, vals, [50, 1600], { outside: "nan" });
    expect(out[0]).toBeNaN();
    expect(out[1]).toBeNaN();
  });

  it("outside=<number> acts as a fence value", () => {
    const out = interpOnGrid(src, vals, [50, 1600], { outside: -200 });
    expect(out[0]).toBe(-200);
    expect(out[1]).toBe(-200);
  });

  it("exact boundary frequencies interpolate to edge values under any policy", () => {
    for (const outside of ["clamp", "nan", -200] as const) {
      const out = interpOnGrid(src, vals, [100, 800], { outside });
      expect(out[0]).toBe(0);
      expect(out[1]).toBe(30);
    }
  });

  it("null endpoints produce null (phase-wrap gaps preserved)", () => {
    const gappy = [0, null, 20, 30];
    const out = interpOnGrid(src, gappy, [150, 300, 600]);
    expect(out[0]).toBeNull(); // 100..200 has null hi
    expect(out[1]).toBeNull(); // 200..400 has null lo
    expect(out[2]).toBeCloseTo(25, 10); // 400..800 intact
  });

  it("duplicate source frequencies (dt=0) fall back to the matched lo value", () => {
    // Binary search advances lo past duplicates (same as all 5 original
    // copies) — the rightmost duplicate wins, no NaN from division by zero.
    const out = interpOnGrid([100, 100, 200], [1, 2, 3], [100]);
    expect(out[0]).toBe(2);
  });

  it("resampleOnLogGrid wrapper keeps its legacy edge cases", () => {
    expect(resampleOnLogGrid([], [], [10, 20])).toEqual([0, 0]);
    expect(resampleOnLogGrid([100], [7], [10, 100, 500])).toEqual([7, 7, 7]);
    const [v] = resampleOnLogGrid(src, vals, [Math.sqrt(100 * 200)]);
    expect(v).toBeCloseTo(5, 10);
  });
});
