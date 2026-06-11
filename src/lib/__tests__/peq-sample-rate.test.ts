// b141.5 (audit CRITICAL): every compute_peq_complex IPC call must carry the
// sample rate the realized correction will run at. Without it Rust defaults
// to 48 kHz while the IIR/FIR path realizes biquads at the export sample
// rate — bilinear warp diverges by several dB near Nyquist (e.g. ~3.5 dB at
// 20 kHz for a 15 kHz peaking filter at sr=96k).

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { BandState } from "../../stores/bands";

interface PeqCall { freq: number[]; sampleRate?: number }
const peqCalls: PeqCall[] = [];

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      return { magnitude: freq.map(() => 0), phase: freq.map(() => 0) };
    }
    if (cmd === "evaluate_target_standalone") {
      const n = args.nPoints ?? 512;
      const fmin = args.fMin ?? 5;
      const fmax = args.fMax ?? 40000;
      const f: number[] = [];
      for (let i = 0; i < n; i++) f.push(fmin * Math.pow(fmax / fmin, i / (n - 1)));
      return [f, { magnitude: new Array(n).fill(0), phase: new Array(n).fill(0) }];
    }
    if (cmd === "compute_minimum_phase") {
      return new Array((args.magnitude as number[]).length).fill(0);
    }
    if (cmd === "compute_peq_complex") {
      peqCalls.push({ freq: args.freq, sampleRate: args.sampleRate });
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0)];
    }
    if (cmd === "get_smoothed") return args.magnitude;
    if (cmd === "compute_impulse") {
      const n = 64;
      return {
        time: Array.from({ length: n }, (_, i) => i / 48000),
        impulse: new Array(n).fill(0),
        step: new Array(n).fill(0),
      };
    }
    if (cmd === "pick_fir_route") return "Cepstral";
    if (cmd === "generate_model_fir") {
      const taps = args.config.taps as number;
      const mag = args.targetMag as number[];
      return {
        impulse: new Array(taps).fill(0),
        time_ms: new Array(taps).fill(0),
        realized_mag: mag, realized_phase: new Array(mag.length).fill(0),
        taps, sample_rate: args.config.sample_rate, norm_db: 0, causality: 1,
      };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateBandFull, evaluateSum } from "../band-evaluator";

function bandWithPeq(): BandState {
  const n = 256;
  const freq: number[] = [];
  for (let i = 0; i < n; i++) freq.push(20 * Math.pow(20000 / 20, i / (n - 1)));
  return {
    id: "b1", name: "b1",
    measurement: {
      name: "m", source_path: null, sample_rate: 48000,
      freq, magnitude: new Array(n).fill(0), phase: new Array(n).fill(0),
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

const FIR_CFG = {
  taps: 4096, sampleRate: 96000, window: "Blackman",
  maxBoostDb: 24, noiseFloorDb: -150,
  iterations: 1, freqWeighting: false,
  narrowbandLimit: false, nbSmoothingOct: 0.333, nbMaxExcessDb: 6,
};

beforeEach(() => { peqCalls.length = 0; });

describe("compute_peq_complex sample-rate contract (b141.5)", () => {
  it("evaluateBandFull display path passes req.sampleRate", async () => {
    await evaluateBandFull({ band: bandWithPeq(), sampleRate: 96000 });
    expect(peqCalls.length).toBeGreaterThan(0);
    for (const c of peqCalls) expect(c.sampleRate).toBe(96000);
  });

  it("evaluateBandFull FIR path passes fir.sampleRate to every PEQ call", async () => {
    await evaluateBandFull({ band: bandWithPeq(), fir: FIR_CFG });
    expect(peqCalls.length).toBeGreaterThanOrEqual(2); // display grid + FIR grid
    for (const c of peqCalls) expect(c.sampleRate).toBe(96000);
  });

  it("evaluateBandFull IR path passes sampleRate on the IR grid too", async () => {
    await evaluateBandFull({ band: bandWithPeq(), sampleRate: 96000, includeIr: true });
    expect(peqCalls.length).toBeGreaterThanOrEqual(2); // display grid + IR grid
    for (const c of peqCalls) expect(c.sampleRate).toBe(96000);
  });

  it("evaluateSum threads sampleRate down to per-band PEQ calls (incl. IR)", async () => {
    await evaluateSum([bandWithPeq()], { includeIr: true, sampleRate: 96000 });
    expect(peqCalls.length).toBeGreaterThan(0);
    for (const c of peqCalls) expect(c.sampleRate).toBe(96000);
  });
});
