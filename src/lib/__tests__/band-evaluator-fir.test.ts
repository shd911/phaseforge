// b139.3.1 — automatic regression coverage for the FIR identity case.
// If a flat measurement + flat target + no PEQ + no HP/LP no longer yields
// an identity FIR, this test catches it without touching the .dmg.

import { describe, it, expect, vi } from "vitest";
import type { BandState } from "../../stores/bands";
import type { TargetCurve } from "../types";
import { gaussianFilterMagDb, subsonicMagDb } from "../plot-helpers";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const target = args.target as TargetCurve;
      const freq = args.freq as number[];
      const mag = freq.map(() => target.reference_level_db ?? 0);
      const phase = freq.map(() => 0);
      const hp = target.high_pass;
      if (hp && hp.filter_type === "Gaussian") {
        const hpMag = gaussianFilterMagDb(freq, hp, false);
        for (let i = 0; i < freq.length; i++) mag[i] += hpMag[i];
        if (hp.subsonic_protect === true && hp.freq_hz > 40) {
          const sub = subsonicMagDb(freq, hp.freq_hz / 8);
          for (let i = 0; i < freq.length; i++) mag[i] += sub[i];
        }
      }
      return { magnitude: mag, phase };
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
      // Hilbert-mock: deterministic sum of dB slope, scaled.
      const m = args.magnitude as number[];
      const ph: number[] = [0];
      for (let i = 1; i < m.length; i++) ph.push(ph[i - 1] + (m[i] - m[i - 1]) * 0.5);
      return ph;
    }
    if (cmd === "compute_peq_complex") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0)];
    }
    if (cmd === "get_smoothed") return args.magnitude;
    if (cmd === "generate_model_fir") {
      // Fake Rust: produce a centred unit-impulse with optional phase rotation
      // to mimic LinearPhase / MinimumPhase. Identity case here = pure flat
      // magnitude + zero phase → centred unit impulse in LinearPhase, or a
      // delta at index 0 in MinimumPhase. Off-peak energy = 0.
      const taps = args.config.taps as number;
      const impulse = new Array(taps).fill(0);
      // b139.4a: Composite degenerates to linear (centered) when
      // linear_phase_main && no subsonic; otherwise causal at index 0.
      const cfg = args.config;
      const isLinPath =
        cfg.phase_mode === "LinearPhase" ||
        (cfg.phase_mode === "Composite" && cfg.linear_phase_main === true && cfg.subsonic_cutoff_hz == null);
      const peakIdx = isLinPath ? Math.floor(taps / 2) : 0;
      // For non-flat target_mag, the mock degrades (we only need this branch
      // exercised by the identity test).
      const mag = args.targetMag as number[];
      const peq = args.peqMag as number[];
      const isFlat = mag.every((v) => Math.abs(v) < 1e-9)
        && (peq.length === 0 || peq.every((v) => Math.abs(v) < 1e-9));
      impulse[peakIdx] = isFlat ? 1.0 : 0.5;
      return {
        impulse, time_ms: impulse.map((_, i) => i * 1000 / args.config.sample_rate),
        realized_mag: mag, realized_phase: new Array(mag.length).fill(0),
        taps, sample_rate: args.config.sample_rate, norm_db: 0, causality: 1,
      };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateBandFull } from "../band-evaluator";

function flatBand(): BandState {
  const n = 512;
  const freq: number[] = [];
  for (let i = 0; i < n; i++) freq.push(20 * Math.pow(20000 / 20, i / (n - 1)));
  return {
    id: "flat", name: "flat",
    measurement: {
      name: "flat", source_path: null, sample_rate: 48000,
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
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  };
}

describe("evaluateBandFull FIR Composite IPC payload (b139.4a)", () => {
  async function captureFirArgs(band: BandState, sr = 48000) {
    const mockMod = await import("@tauri-apps/api/core");
    const inv = (mockMod as any).invoke as ReturnType<typeof vi.fn>;
    let captured: { freq: number[]; targetMag: number[]; config: any } | null = null;
    const sniff = inv.getMockImplementation()!;
    inv.mockImplementation(async (cmd: string, args: any) => {
      if (cmd === "generate_model_fir") {
        captured = { freq: args.freq, targetMag: args.targetMag, config: args.config };
      }
      return sniff(cmd, args);
    });
    try {
      await evaluateBandFull({
        band,
        fir: {
          taps: 4096, sampleRate: sr, window: "Blackman",
          maxBoostDb: 24, noiseFloorDb: -150,
          iterations: 1, freqWeighting: false,
          narrowbandLimit: false, nbSmoothingOct: 0.333, nbMaxExcessDb: 6,
        },
      });
    } finally {
      inv.mockImplementation(sniff);
    }
    return captured;
  }

  async function captureFirConfig(band: BandState) {
    const r = await captureFirArgs(band);
    return r?.config;
  }

  it("flat / no HP → phase_mode=Composite, linear_phase_main=true, subsonic_cutoff_hz=null", async () => {
    const cfg = await captureFirConfig(flatBand());
    expect(cfg.phase_mode).toBe("Composite");
    expect(cfg.linear_phase_main).toBe(true);
    expect(cfg.subsonic_cutoff_hz).toBeNull();
  });

  it("HP linear + subsonic_protect → linear_phase_main=true, subsonic_cutoff_hz=fc/8", async () => {
    const band = flatBand();
    band.target.high_pass = {
      filter_type: "Gaussian", order: 4, freq_hz: 632.0,
      shape: 1.0, q: null, linear_phase: true, subsonic_protect: true,
    };
    const cfg = await captureFirConfig(band);
    expect(cfg.phase_mode).toBe("Composite");
    expect(cfg.linear_phase_main).toBe(true);
    expect(cfg.subsonic_cutoff_hz).toBeCloseTo(632.0 / 8, 5);
  });

  it("HP min-phase + subsonic_protect → linear_phase_main=false, subsonic_cutoff_hz=fc/8", async () => {
    const band = flatBand();
    band.target.high_pass = {
      filter_type: "Gaussian", order: 4, freq_hz: 632.0,
      shape: 1.0, q: null, linear_phase: false, subsonic_protect: true,
    };
    const cfg = await captureFirConfig(band);
    expect(cfg.phase_mode).toBe("Composite");
    expect(cfg.linear_phase_main).toBe(false);
    expect(cfg.subsonic_cutoff_hz).toBeCloseTo(632.0 / 8, 5);
  });

  it("HP linear, no subsonic_protect → linear_phase_main=true, subsonic_cutoff_hz=null", async () => {
    const band = flatBand();
    band.target.high_pass = {
      filter_type: "Gaussian", order: 4, freq_hz: 632.0,
      shape: 1.0, q: null, linear_phase: true, subsonic_protect: false,
    };
    const cfg = await captureFirConfig(band);
    expect(cfg.phase_mode).toBe("Composite");
    expect(cfg.linear_phase_main).toBe(true);
    expect(cfg.subsonic_cutoff_hz).toBeNull();
  });
});

describe("evaluateBandFull FIR grid (b139.5.3)", () => {
  // The FIR pipeline must run on the standalone 5..min(40k, sr/2·0.95)
  // log grid (512 pts), independent of the measurement grid that drives
  // the SPL display. Otherwise HP rolloff below 20 Hz and anti-aliasing
  // headroom above 20 kHz are both truncated.
  async function captureFirArgs2(band: BandState, sr: number) {
    const mockMod = await import("@tauri-apps/api/core");
    const inv = (mockMod as any).invoke as ReturnType<typeof vi.fn>;
    let captured: { freq: number[] } | null = null;
    const sniff = inv.getMockImplementation()!;
    inv.mockImplementation(async (cmd: string, args: any) => {
      if (cmd === "generate_model_fir") captured = { freq: args.freq };
      return sniff(cmd, args);
    });
    try {
      await evaluateBandFull({
        band,
        fir: {
          taps: 4096, sampleRate: sr, window: "Blackman",
          maxBoostDb: 24, noiseFloorDb: -150,
          iterations: 1, freqWeighting: false,
          narrowbandLimit: false, nbSmoothingOct: 0.333, nbMaxExcessDb: 6,
        },
      });
    } finally {
      inv.mockImplementation(sniff);
    }
    return captured!;
  }

  it("FIR grid is 512 log-spaced points 5..40k at sr=48k", async () => {
    const band = flatBand();
    const { freq } = await captureFirArgs2(band, 48000);
    expect(freq.length).toBe(512);
    expect(freq[0]).toBeCloseTo(5, 5);
    // sr/2 * 0.95 = 22800 < 40000 → fMax = 22800
    expect(freq[freq.length - 1]).toBeCloseTo(48000 / 2 * 0.95, 0);
  });

  it("FIR grid caps at 40 kHz when Nyquist · 0.95 > 40 kHz (sr=176.4k)", async () => {
    const band = flatBand();
    const { freq } = await captureFirArgs2(band, 176400);
    expect(freq.length).toBe(512);
    expect(freq[0]).toBeCloseTo(5, 5);
    // sr/2 * 0.95 = 83790 → capped at 40000
    expect(freq[freq.length - 1]).toBeCloseTo(40000, 0);
  });

  it("FIR grid is independent of measurement grid", async () => {
    // flatBand has measurement.freq starting at 20 Hz, ending 20 kHz, 512 pts.
    // FIR must NOT inherit those bounds.
    const band = flatBand();
    const { freq } = await captureFirArgs2(band, 48000);
    // Must extend below measurement min (20 Hz).
    expect(freq[0]).toBeLessThan(20);
  });
});

describe("evaluateBandFull FIR identity (b139.3.1)", () => {
  it("flat measurement, no filters, no PEQ → identity FIR", async () => {
    const band = flatBand();
    const result = await evaluateBandFull({
      band,
      fir: {
        taps: 8192, sampleRate: 48000, window: "Blackman",
        maxBoostDb: 24, noiseFloorDb: -150,
        iterations: 3, freqWeighting: true,
        narrowbandLimit: false, nbSmoothingOct: 0.333, nbMaxExcessDb: 6,
      },
    });

    expect(result.fir).toBeDefined();
    const impulse = result.fir!.impulse;

    let peakIdx = 0;
    let peakAbs = 0;
    for (let i = 0; i < impulse.length; i++) {
      if (Math.abs(impulse[i]) > peakAbs) {
        peakAbs = Math.abs(impulse[i]);
        peakIdx = i;
      }
    }
    expect(Math.abs(peakAbs - 1.0)).toBeLessThan(0.01);

    let off = 0;
    for (let i = 0; i < impulse.length; i++) {
      if (i !== peakIdx) off += impulse[i] * impulse[i];
    }
    expect(off).toBeLessThan(0.01);
  });
});
