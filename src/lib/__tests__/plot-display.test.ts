// b141.104 (audit 2026-10-23, section 4): display rules of the plot.
import { describe, it, expect } from "vitest";
import {
  polarityPhase, polarityAmplitude, peqHandleDb, peqGainAfterDrag,
  fitMagRange, fitExportMagRange, sumGroupDelayMs, phaseSlopeDegPerHz,
} from "../plot-display";
import { exportPhaseLabel } from "../export-metrics";
import { buildLogGrid } from "../band-evaluator/grid";
import { coherentSum } from "../band-evaluator/sum";
import { computeGroupDelay, unwrapDegrees } from "../plot-helpers";
import type { FilterConfig } from "../types";

const wrap = (d: number) => d - 360 * Math.round(d / 360);

describe("polarity (4.1/4.2)", () => {
  it("flips phase by 180° only when inverted, keeps wrap gaps", () => {
    expect(polarityPhase([10, null, -170], true)).toEqual([190, null, 10]);
    const ph = [1, 2];
    expect(polarityPhase(ph, false)).toBe(ph);
  });
  it("negates impulse/step only when inverted", () => {
    expect(polarityAmplitude([1, -2], true)).toEqual([-1, 2]);
    expect(polarityAmplitude([1, -2], false)).toEqual([1, -2]);
  });
});

describe("PEQ handle (4.6)", () => {
  it("sits at the filter's own level: full gain for peaking, half for shelves", () => {
    expect(peqHandleDb({ gain_db: 6, filter_type: "Peaking" })).toBe(6);
    expect(peqHandleDb({ gain_db: 6, filter_type: "LowShelf" })).toBe(3);
    expect(peqHandleDb({ gain_db: -8, filter_type: "HighShelf" })).toBe(-4);
  });
  it("drag moves the gain by the cursor travel — no jump, shelf keeps its gain", () => {
    // Grabbed 2 dB off the handle, not moved yet: gain unchanged.
    expect(peqGainAfterDrag(6, 0, "Peaking")).toBe(6);
    expect(peqGainAfterDrag(6, 1.5, "Peaking")).toBe(7.5);
    // Shelf handle at g/2: moving it by +1 dB means +2 dB of gain.
    expect(peqGainAfterDrag(6, 1, "LowShelf")).toBe(8);
    expect(peqGainAfterDrag(20, 10, "HighShelf")).toBe(24);
  });
});

describe("Y fit (4.12/4.18)", () => {
  it("ignores hidden curves", () => {
    const r = fitMagRange([
      { values: [0, -3, -18], visible: true },
      { values: [-150, -140], visible: false },
    ]);
    expect(r.min).toBe(-20);
  });
  it("keeps the highest visible point on screen", () => {
    const r = fitMagRange([{ values: [-18, 0, 8], visible: true }]);
    expect(r.min).toBe(-20);
    expect(r.max).toBeGreaterThanOrEqual(9);
  });
  it("keeps the old shape when nothing pokes above it", () => {
    expect(fitMagRange([{ values: [-40, 0], visible: true }])).toEqual({ min: -40, max: 10 });
  });
  it("export fit: visible data span + 10 %", () => {
    const r = fitExportMagRange([
      { values: [-30, 0], visible: true },
      { values: [-180], visible: false },
    ])!;
    expect(r.min).toBeCloseTo(-33);
    expect(r.max).toBeCloseTo(3);
    expect(fitExportMagRange([{ values: [1], visible: false }])).toBeNull();
  });
});

describe("export phase label (4.23)", () => {
  const f = (lin: boolean): FilterConfig =>
    ({ filter_type: "LinkwitzRiley", order: 4, freq_hz: 1000, shape: null, linear_phase: lin, q: null } as FilterConfig);
  it("names mixed phase", () => {
    expect(exportPhaseLabel(f(false), f(true), false)).toBe("Mixed-Phase");
    expect(exportPhaseLabel(f(true), f(false), true)).toBe("Mixed-Phase + Subsonic");
  });
  it("keeps the pure cases", () => {
    expect(exportPhaseLabel(f(true), f(true), false)).toBe("Linear-Phase");
    expect(exportPhaseLabel(f(false), null, false)).toBe("Min-Phase");
    expect(exportPhaseLabel(null, null, false)).toBe("Linear-Phase");
    expect(exportPhaseLabel(f(false), f(false), true)).toBe("Min-Phase + Subsonic");
  });
});

describe("Σ group delay (4.7)", () => {
  // Two flat bands, both with 1 ms of time of flight in their phase; the
  // second is aligned 3 ms later. Equal amplitudes: the sum's GD is the
  // mean, 2.5 ms, wherever the sum does not cancel.
  const freq = buildLogGrid(512, 20, 20000);
  const tof = 0.001;
  const band = (delay: number) => ({
    mag: freq.map(() => 0),
    phase: freq.map((f) => wrap(-360 * f * tof)),
    sign: 1 as const,
    delay,
  });
  const parts = [band(0), band(0.003)];

  it("reads 2.5 ms up to 20 kHz on the 512-point grid", () => {
    const gd = sumGroupDelayMs(freq, parts);
    const sum = coherentSum(freq, parts)!;
    let checked = 0;
    for (let j = 0; j < freq.length; j++) {
      if (freq[j] < 10000 || sum.mag[j] < 0) continue; // away from the comb nulls
      expect(gd[j]).toBeCloseTo(2.5, 1);
      checked++;
    }
    expect(checked).toBeGreaterThan(20);
  });

  it("the old unwrap of the summed phase aliased there", () => {
    const sum = coherentSum(freq, parts)!;
    const old = computeGroupDelay(freq, unwrapDegrees(sum.phase)).gdMs;
    let worst = 0;
    for (let j = 0; j < freq.length; j++) {
      if (freq[j] < 12300 || sum.mag[j] < 0) continue;
      worst = Math.max(worst, Math.abs(old[j] - 2.5) / 2.5);
    }
    expect(worst).toBeGreaterThan(0.1);
  });

  it("a single band reduces to its own GD, delay included", () => {
    const gd = sumGroupDelayMs(freq, [band(0.002)]);
    for (let j = 0; j < freq.length; j += 37) expect(gd[j]).toBeCloseTo(3, 3);
    const slope = phaseSlopeDegPerHz(freq, band(0).phase, band(0).mag);
    expect(-slope[400] / 360 * 1000).toBeCloseTo(1, 3);
  });

  it("polarity: an inverted band cancels its twin", () => {
    const inv = { ...band(0), sign: -1 as const };
    const gd = sumGroupDelayMs(freq, [band(0), inv]);
    expect(gd.every((v) => Number.isNaN(v))).toBe(true);
  });
});
