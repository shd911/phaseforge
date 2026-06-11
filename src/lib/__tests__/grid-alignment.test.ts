// b141.6 (audit MEDIUM): when a caller passes `req.freq` AND the band has a
// measurement on a different grid, evaluateBandFull used to mix the two
// index-wise: result.freq = req.freq, but measurementMag/correctedMag stayed
// on the measurement grid (corrected = measurement[i] + peq[i] + xs[i] with
// i indexing DIFFERENT grids). Latent until a consumer reads them — then
// silently garbage. Measurement curves must come back resampled onto `freq`.

import { describe, it, expect, vi } from "vitest";
import type { BandState } from "../../stores/bands";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      return { magnitude: freq.map(() => 0), phase: freq.map(() => 0) };
    }
    if (cmd === "compute_minimum_phase") {
      return new Array((args.magnitude as number[]).length).fill(0);
    }
    if (cmd === "compute_peq_complex") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(2), new Array(n).fill(0)]; // +2 dB everywhere
    }
    if (cmd === "get_smoothed") return args.magnitude;
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateBandFull } from "../band-evaluator";

function logGrid(n: number, fMin: number, fMax: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(fMin * Math.pow(fMax / fMin, i / (n - 1)));
  return out;
}

function band(): BandState {
  const freq = logGrid(300, 20, 20000); // measurement grid: 300 pts
  return {
    id: "b1", name: "b1",
    measurement: {
      name: "m", source_path: null, sample_rate: 48000,
      freq, magnitude: new Array(300).fill(10), phase: new Array(300).fill(0),
      metadata: { date: null, mic: null, notes: null, smoothing: null },
    },
    measurementFile: null,
    settings: { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false },
    target: {
      reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: null, low_pass: null, low_shelf: null, high_shelf: null,
    },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [{ freq_hz: 1000, gain_db: -3, q: 2, enabled: true, filter_type: "Peaking" }],
    peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  } as BandState;
}

describe("evaluateBandFull grid alignment (b141.6)", () => {
  it("measurement + corrected come back on req.freq when grids differ", async () => {
    const reqFreq = logGrid(128, 30, 15000); // different count AND bounds
    const result = await evaluateBandFull({ band: band(), freq: reqFreq });

    expect(result.freq.length).toBe(128);
    expect(result.measurementMag?.length).toBe(128);
    expect(result.correctedMag?.length).toBe(128);
    // Flat 10 dB measurement + flat +2 dB PEQ, no filters → corrected = 12
    // on EVERY bin of the request grid (inside the measurement range).
    for (let i = 0; i < 128; i++) {
      expect(Math.abs(result.correctedMag![i] - 12)).toBeLessThan(1e-6);
    }
  });

  it("measurement grid passthrough still works when grids match", async () => {
    const b = band();
    const result = await evaluateBandFull({ band: b });
    expect(result.measurementMag?.length).toBe(300);
    for (let i = 0; i < 300; i++) {
      expect(Math.abs(result.correctedMag![i] - 12)).toBeLessThan(1e-6);
    }
  });
});
