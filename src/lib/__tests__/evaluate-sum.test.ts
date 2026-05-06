// b140.3.0 — minimal evaluateSum: coherent sum of per-band Σ Target only.
//
// Old b140.2.x snapshots (avgRef, fence, extension, perBandResampled,
// sumIr, …) are intentionally gone — that pipeline accumulated 11
// fix iterations without parity. The new tests pin only what the
// minimal aggregator promises: Σ Target = coherent sum of per-band
// targets with polarity + alignment_delay phase rotation.

import { describe, it, expect, vi } from "vitest";
import type { BandState } from "../../stores/bands";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      const refLevel = args.target?.reference_level_db ?? 0;
      return {
        magnitude: freq.map(() => refLevel),
        phase: freq.map(() => 0),
      };
    }
    if (cmd === "compute_minimum_phase") {
      const m = args.magnitude as number[];
      return new Array(m.length).fill(0);
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateSum } from "../band-evaluator";

const N = 512;
function logGrid(fmin: number, fmax: number, n: number): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fmin * Math.pow(fmax / fmin, i / (n - 1));
  return out;
}

/** Flat target band with mag = 0 dB across 20 Hz – 20 kHz. */
function flatBand(id: string, overrides: Partial<BandState> = {}): BandState {
  const freq = logGrid(20, 20000, N);
  return {
    id,
    name: id,
    measurement: {
      name: id,
      source_path: null,
      sample_rate: 48000,
      freq,
      magnitude: new Array(N).fill(0),
      phase: new Array(N).fill(0),
      metadata: { date: null, mic: null, notes: null, smoothing: null },
    },
    measurementFile: null,
    settings: {
      smoothing: "off", delay_seconds: null, distance_meters: null,
      delay_removed: false, originalPhase: null, floorBounce: null,
      mergeSource: null, analysis: null, analysisDismissed: false,
    },
    target: {
      reference_level_db: 0,
      tilt_db_per_octave: 0,
      tilt_ref_freq: 1000,
      high_pass: null, low_pass: null, low_shelf: null, high_shelf: null,
    },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
    ...overrides,
  };
}

describe("evaluateSum (minimal, b140.3.0) — Σ Target only", () => {
  it("returns null sum when no band has targetEnabled", async () => {
    const result = await evaluateSum([
      flatBand("a", { targetEnabled: false }),
      flatBand("b", { targetEnabled: false }),
    ]);
    expect(result.sumTargetMag).toBeNull();
    expect(result.sumTargetPhase).toBeNull();
    expect(result.freq.length).toBe(512);
  });

  it("two flat-target bands → coherent sum +6 dB across the band", async () => {
    const result = await evaluateSum([flatBand("a"), flatBand("b")]);
    expect(result.sumTargetMag).not.toBeNull();
    const mag = result.sumTargetMag!;
    // 1 kHz sample
    const idx = mag.length / 2 | 0;
    expect(mag[idx]).toBeCloseTo(6.0206, 3);
  });

  it("two bands with opposite polarity → cancellation (mag → -∞)", async () => {
    const a = flatBand("a");
    const b = flatBand("b", { inverted: true });
    const result = await evaluateSum([a, b]);
    const mag = result.sumTargetMag!;
    // Floor for total cancellation in coherentSum is -200 dB.
    expect(mag[mag.length / 2 | 0]).toBeLessThan(-150);
  });

  it("alignment delay rotates Σ phase linearly with frequency", async () => {
    // 1 ms delay on a single band → phase = 360 · f · 0.001 deg
    const a = flatBand("a", { alignmentDelay: 0.001 });
    const result = await evaluateSum([a]);
    expect(result.sumTargetPhase).not.toBeNull();
    const f = result.freq;
    const ph = result.sumTargetPhase!;
    // Pick an early bin where the phase wrap stays linear (well below
    // ±180°). At 100 Hz, expected = 360·100·0.001 = 36°.
    const idx = f.findIndex((v) => v >= 100);
    expect(ph[idx]).toBeCloseTo(36, 0);
  });

  it("targetEnabled=false drops band from the sum", async () => {
    const a = flatBand("a"); // 0 dB
    const b = flatBand("b", { targetEnabled: false });
    const result = await evaluateSum([a, b]);
    const mag = result.sumTargetMag!;
    // Only one band contributes → 0 dB, not +6 dB.
    expect(mag[mag.length / 2 | 0]).toBeCloseTo(0, 3);
  });
});
