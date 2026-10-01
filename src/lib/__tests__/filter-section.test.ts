/**
 * b141.48 (audit 2026-10-01 H5): «Corrected» must contain the whole filter
 * section the FIR bakes in — shelves, tilt and the export rate (ultrasonic
 * low-pass) reach compute_cross_section, and its result lands in corrected.
 */
import { describe, it, expect, vi } from "vitest";

const xsCalls: any[] = [];
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    const n = (args.freq as number[] | undefined)?.length ?? 0;
    if (cmd === "evaluate_target") {
      return { magnitude: new Array(n).fill(80), phase: new Array(n).fill(0) };
    }
    if (cmd === "compute_peq_complex") return [new Array(n).fill(0), new Array(n).fill(0)];
    if (cmd === "compute_cross_section") {
      xsCalls.push(args);
      // Stand-in for the Rust section: +4 dB when the low shelf arrives.
      const g = args.lowShelf ? args.lowShelf.gain_db : 0;
      return [new Array(n).fill(g), new Array(n).fill(0), 0];
    }
    if (cmd === "compute_minimum_phase") return new Array(n).fill(0);
    throw new Error(`unmocked ${cmd}`);
  }),
}));

import { evaluateBandFull } from "../band-evaluator";
import type { BandState } from "../../stores/bands";

function band(): BandState {
  const freq = Array.from({ length: 64 }, (_, i) => 20 * Math.pow(1000, i / 63));
  return {
    id: "fs", name: "fs",
    measurement: { name: "m", source_path: null, sample_rate: 48000, freq,
      magnitude: freq.map(() => 80), phase: freq.map(() => 0),
      metadata: { date: null, mic: null, notes: null, smoothing: null } },
    measurementFile: null,
    settings: { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false },
    target: {
      reference_level_db: 80, tilt_db_per_octave: -1, tilt_ref_freq: 1000,
      high_pass: null, low_pass: null,
      low_shelf: { freq_hz: 150, gain_db: 4, q: 0.707 }, high_shelf: null,
    },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  } as unknown as BandState;
}

describe("filter section in Corrected", () => {
  it("passes shelves, tilt and the export rate; corrected includes the shelf", async () => {
    xsCalls.length = 0;
    const r = await evaluateBandFull({ band: band(), sampleRate: 96000 });
    expect(xsCalls.length).toBeGreaterThan(0);
    const a = xsCalls[0];
    expect(a.lowShelf?.gain_db).toBe(4);
    expect(a.tiltDbPerOctave).toBe(-1);
    expect(a.sampleRate).toBe(96000);
    expect(r.correctedMag![10]).toBeCloseTo(84, 6);
  });
});
