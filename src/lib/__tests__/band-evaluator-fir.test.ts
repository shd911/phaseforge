// b141.70: the band → FIR request is assembled in Rust (fir::band, with its
// own tests for the config, the FIR grid and a flat target). This file pins
// the frontend side: the target travels WHOLE (tilt and shelves included),
// the settings map field for field, metadata-only requests stay cheap, and
// the realized curves come back on the caller's grid.

import { describe, it, expect, vi } from "vitest";
import type { BandState } from "../../stores/bands";

const calls: any[] = [];
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    const n = (args.freq as number[] | undefined)?.length ?? 0;
    if (cmd === "evaluate_target") return { magnitude: new Array(n).fill(0), phase: new Array(n).fill(0) };
    if (cmd === "compute_peq_complex") return [new Array(n).fill(0), new Array(n).fill(0)];
    if (cmd === "compute_cross_section") return [new Array(n).fill(0), new Array(n).fill(0), 0];
    if (cmd === "get_smoothed") return args.magnitude;
    if (cmd === "compute_minimum_phase") return new Array((args.magnitude as number[]).length).fill(0);
    if (cmd === "generate_band_fir") {
      calls.push(args);
      const g = Array.from({ length: 600 }, (_, i) => 5 * Math.pow(24000 / 5, i / 599));
      return {
        impulse: args.omitImpulse ? [] : new Array(args.settings.taps).fill(0),
        realized_mag: g.map(() => 0), realized_phase: g.map(() => 0),
        taps: args.settings.taps, sample_rate: args.settings.sample_rate, norm_db: 0, causality: 1,
        wav_delay_samples: args.settings.taps / 2, route: "cepstral", peak_boost_db: 1.5, freq: g,
        dev_mag: g.map(() => 0), dev_phase: g.map(() => 0),
      };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateBandFull } from "../band-evaluator";

function band(): BandState {
  const freq = Array.from({ length: 256 }, (_, i) => 20 * Math.pow(1000, i / 255));
  return {
    id: "b", name: "b",
    measurement: { name: "m", source_path: null, sample_rate: 48000, freq,
      magnitude: freq.map(() => 0), phase: freq.map(() => 0),
      metadata: { date: null, mic: null, notes: null, smoothing: null } },
    measurementFile: null,
    settings: { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false },
    target: { reference_level_db: 3, tilt_db_per_octave: -1, tilt_ref_freq: 1000,
      high_pass: null, low_pass: null,
      low_shelf: { freq_hz: 120, gain_db: 4, q: 0.7 }, high_shelf: null },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [
      { freq_hz: 50, gain_db: 3, q: 2, enabled: true, filter_type: "Peaking" },
      { freq_hz: 900, gain_db: -2, q: 1, enabled: false, filter_type: "Peaking" },
    ],
    peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  } as unknown as BandState;
}

const FIR = {
  taps: 16384, sampleRate: 96000, window: "Hann", maxBoostDb: 18, noiseFloorDb: -140,
  iterations: 2, freqWeighting: true, narrowbandLimit: true, nbSmoothingOct: 0.5, nbMaxExcessDb: 5,
};

describe("generate_band_fir request", () => {
  it("passes the whole target, the enabled PEQ and every setting", async () => {
    calls.length = 0;
    const r = await evaluateBandFull({ band: band(), fir: FIR });
    expect(calls.length).toBe(1);
    const a = calls[0];
    expect(a.target.tilt_db_per_octave).toBe(-1);
    expect(a.target.low_shelf.gain_db).toBe(4);
    expect(a.target.reference_level_db).toBe(3);
    expect(a.peq.map((p: any) => p.freq_hz)).toEqual([50]);
    expect(a.settings).toEqual({
      taps: 16384, sample_rate: 96000, window: "Hann", max_boost_db: 18, noise_floor_db: -140,
      iterations: 2, freq_weighting: true, narrowband_limit: true, nb_smoothing_oct: 0.5, nb_max_excess_db: 5,
    });
    expect(a.omitImpulse).toBe(false);
    expect(r.fir!.peakBoostDb).toBe(1.5);
    expect(r.fir!.impulse.length).toBe(16384);
  });

  it("metadata-only requests ask Rust to drop the impulse", async () => {
    calls.length = 0;
    const r = await evaluateBandFull({ band: band(), fir: { ...FIR, omitImpulse: true } });
    expect(calls[0].omitImpulse).toBe(true);
    expect(r.fir!.impulse.length).toBe(0);
  });

  it("realized curves come back on the caller's grid", async () => {
    const r = await evaluateBandFull({ band: band(), fir: FIR });
    expect(r.fir!.realizedMag.length).toBe(r.freq.length);
    expect(r.fir!.realizedPhase.length).toBe(r.freq.length);
  });
});
