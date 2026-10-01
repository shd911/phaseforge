/**
 * b141.49 (audit 2026-10-01 H9): Σ Corrected with measurements that keep
 * their time of flight. Two flat bands, pure delays τ and τ + 0.1 ms: the
 * Σ magnitude must equal |1 + e^{-j2πf·0.1ms}| on the whole common grid.
 */
import { describe, it, expect, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    const n = (args.freq as number[] | undefined)?.length ?? 0;
    if (cmd === "evaluate_target") return { magnitude: new Array(n).fill(0), phase: new Array(n).fill(0) };
    if (cmd === "compute_peq_complex") return [new Array(n).fill(0), new Array(n).fill(0)];
    if (cmd === "compute_cross_section") return [new Array(n).fill(0), new Array(n).fill(0), 0];
    if (cmd === "compute_minimum_phase") return new Array(n).fill(0);
    throw new Error(`unmocked ${cmd}`);
  }),
}));

import { evaluateSum } from "../band-evaluator/sum";
import type { BandState } from "../../stores/bands";

const wrap = (d: number) => ((d + 180) % 360 + 360) % 360 - 180;

function band(id: string, tau: number): BandState {
  // REW-like dense native grid: 1/96 octave, 20 Hz – 20 kHz.
  const freq: number[] = [];
  for (let f = 20; f <= 20000; f *= Math.pow(2, 1 / 96)) freq.push(f);
  return {
    id, name: id,
    measurement: { name: id, source_path: null, sample_rate: 48000, freq,
      magnitude: freq.map(() => 0), phase: freq.map((f) => wrap(-360 * f * tau)),
      metadata: { date: null, mic: null, notes: null, smoothing: null } },
    measurementFile: null,
    settings: { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false },
    target: { reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: null, low_pass: null, low_shelf: null, high_shelf: null },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  } as unknown as BandState;
}

describe("Σ corrected keeps the delay phase", () => {
  for (const tau of [0.001, 0.003, 0.005]) {
    it(`τ = ${tau * 1000} ms`, async () => {
      const r = await evaluateSum([band("a", tau), band("b", tau + 0.0001)], { sampleRate: 48000 });
      let worst = 0;
      r.freq.forEach((f, j) => {
        const ref = 20 * Math.log10(Math.max(1e-9, Math.abs(2 * Math.cos(Math.PI * f * 0.0001))));
        if (ref < -20) return; // skip the notch itself
        worst = Math.max(worst, Math.abs(r.sumCorrectedMag![j] - ref));
      });
      expect(worst).toBeLessThan(0.1);
    });
  }
});
