/**
 * b140.15.10: TS-side regression test for the 3-band LR4 SUM spike report.
 *
 * Mirrors the Rust integration test in src-tauri/tests/sum_3band_lr4_flat.rs
 * but exercises the full TS pipeline (evaluateSum → evaluateBandFull →
 * resampleOnLogGrid → reconstructTargetPhase → coherentSum), with the Tauri
 * boundary mocked using an idealised Rust mirror (LR4 = BW² in complex,
 * cross-section via complex accumulator).
 *
 * If this test FAILS with spikes, the bug is in TS-side processing —
 * not in Rust math. If it PASSES while the user still sees spikes in the
 * UI, the user's project configuration has something this fixture doesn't
 * capture (PEQ, alignmentDelay, shelves, partial-range measurement, etc.).
 *
 * On failure: dumps detailed spike report to console.
 */
import { describe, expect, it, vi } from "vitest";
import type { BandState } from "../../stores/bands";
import type { FilterConfig } from "../types";

// ---------------------------------------------------------------------------
// Idealised Rust mirror for Tauri command mocks
// ---------------------------------------------------------------------------

function bwLpRaw(f: number, fc: number, n: number): [number, number] {
  const w = f / fc;
  let re = 1, im = 0;
  for (let k = 0; k < n; k++) {
    const theta = Math.PI * (2 * k + n + 1) / (2 * n);
    const poleRe = Math.cos(theta);
    const poleIm = Math.sin(theta);
    const dr = -poleRe;
    const di = w - poleIm;
    const denom = dr * dr + di * di;
    const invRe = dr / denom;
    const invIm = -di / denom;
    const nr = re * invRe - im * invIm;
    const ni = re * invIm + im * invRe;
    re = nr; im = ni;
  }
  return [re, im];
}

function bwHpRaw(f: number, fc: number, n: number): [number, number] {
  const w = f / fc;
  let re = 1, im = 0;
  for (let k = 0; k < n; k++) {
    const theta = Math.PI * (2 * k + n + 1) / (2 * n);
    const poleRe = Math.cos(theta);
    const poleIm = Math.sin(theta);
    const dr = -poleRe;
    const di = -1 / w - poleIm;
    const denom = dr * dr + di * di;
    const invRe = dr / denom;
    const invIm = -di / denom;
    const nr = re * invRe - im * invIm;
    const ni = re * invIm + im * invRe;
    re = nr; im = ni;
  }
  return [re, im];
}

/** Filter complex transfer at one freq. Returns (re, im). */
function filterComplex(f: number, cfg: FilterConfig, isLowpass: boolean): [number, number] {
  const fc = cfg.freq_hz;
  if (cfg.filter_type === "Butterworth") {
    return isLowpass ? bwLpRaw(f, fc, cfg.order) : bwHpRaw(f, fc, cfg.order);
  }
  if (cfg.filter_type === "LinkwitzRiley") {
    const [re, im] = isLowpass ? bwLpRaw(f, fc, cfg.order) : bwHpRaw(f, fc, cfg.order);
    return [re * re - im * im, 2 * re * im]; // squared
  }
  // Other types not needed for this test
  return [1, 0];
}

function complexMulAcc(re: number, im: number, fre: number, fim: number): [number, number] {
  return [re * fre - im * fim, re * fim + im * fre];
}

// Mock state — captured from describe-block, freq/target etc. needed inside the mock.
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      const target = args.target;
      const refDb: number = target?.reference_level_db ?? 0;
      const mag: number[] = new Array(freq.length);
      const phase: number[] = new Array(freq.length);
      for (let i = 0; i < freq.length; i++) {
        let m = refDb;
        let re = 1, im = 0;
        if (target.high_pass) {
          const [fre, fim] = filterComplex(freq[i], target.high_pass, false);
          m += 20 * Math.log10(Math.sqrt(fre * fre + fim * fim) || 1e-30);
          [re, im] = complexMulAcc(re, im, fre, fim);
          // normalize fre/fim to unit phasor for phase-only accumulation
          // Actually we accumulate full complex including mag; convert mag separately
        }
        if (target.low_pass) {
          const [fre, fim] = filterComplex(freq[i], target.low_pass, true);
          m += 20 * Math.log10(Math.sqrt(fre * fre + fim * fim) || 1e-30);
          [re, im] = complexMulAcc(re, im, fre, fim);
        }
        mag[i] = m;
        phase[i] = Math.atan2(im, re) * 180 / Math.PI;
      }
      return { magnitude: mag, phase };
    }
    if (cmd === "evaluate_target_standalone") {
      const n = args.nPoints ?? 512;
      const fMin = args.fMin ?? 5, fMax = args.fMax ?? 40000;
      const freq = new Array<number>(n);
      for (let i = 0; i < n; i++) freq[i] = fMin * Math.pow(fMax / fMin, i / (n - 1));
      const res = await (vi.mocked(await import("@tauri-apps/api/core")).invoke as any)("evaluate_target", { freq, target: args.target });
      return [freq, res];
    }
    if (cmd === "compute_cross_section") {
      const freq = args.freq as number[];
      const hp = args.highPass;
      const lp = args.lowPass;
      const mag: number[] = new Array(freq.length);
      const phase: number[] = new Array(freq.length);
      for (let i = 0; i < freq.length; i++) {
        let m = 0;
        let re = 1, im = 0;
        if (hp) {
          const [fre, fim] = filterComplex(freq[i], hp, false);
          m += 20 * Math.log10(Math.sqrt(fre * fre + fim * fim) || 1e-30);
          [re, im] = complexMulAcc(re, im, fre, fim);
        }
        if (lp) {
          const [fre, fim] = filterComplex(freq[i], lp, true);
          m += 20 * Math.log10(Math.sqrt(fre * fre + fim * fim) || 1e-30);
          [re, im] = complexMulAcc(re, im, fre, fim);
        }
        mag[i] = m;
        phase[i] = Math.atan2(im, re) * 180 / Math.PI;
      }
      return [mag, phase, 0];
    }
    if (cmd === "compute_peq_complex") {
      const f = args.freq as number[];
      return [new Array(f.length).fill(0), new Array(f.length).fill(0)];
    }
    if (cmd === "compute_minimum_phase") {
      const m = args.magnitude as number[];
      return new Array(m.length).fill(0);
    }
    if (cmd === "compute_impulse") {
      const mag = args.magnitude as number[];
      const phase = args.phase as number[];
      return {
        time: mag.map((_: number, i: number) => i / 48000),
        impulse: [...mag],
        step: [...phase],
      };
    }
    if (cmd === "pick_fir_route") return "Iir";
    throw new Error(`Unmocked: ${cmd}`);
  }),
}));

import { evaluateSum } from "../band-evaluator";

// ---------------------------------------------------------------------------
// Setup builders
// ---------------------------------------------------------------------------

const N = 512;
function logGrid(fmin: number, fmax: number, n: number): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fmin * Math.pow(fmax / fmin, i / (n - 1));
  return out;
}

function lr4(freq: number): FilterConfig {
  return {
    filter_type: "LinkwitzRiley",
    order: 4,
    freq_hz: freq,
    shape: null,
    linear_phase: false,
    q: null,
    subsonic_protect: null,
  };
}

function flatMeasBand(id: string, hp: FilterConfig | null, lp: FilterConfig | null): BandState {
  const freq = logGrid(20, 20000, N);
  // Simulate user's near-flat measurement: 75 dB, linear delay 222 µs
  const magnitude = new Array(N).fill(75);
  const phase = freq.map(f => -360 * f * 222e-6);
  return {
    id, name: id,
    measurement: {
      name: id,
      source_path: null,
      sample_rate: 48000,
      freq,
      magnitude,
      phase,
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
      high_pass: hp, low_pass: lp,
      low_shelf: null, high_shelf: null,
    },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  };
}

// ---------------------------------------------------------------------------
// Spike detector
// ---------------------------------------------------------------------------

interface Spike { bin: number; freq: number; prev: number; spike: number; next: number; }

function findSpikes(freq: number[], values: number[], stepThreshold: number, neighbourAgreement: number, mod360: boolean): Spike[] {
  const spikes: Spike[] = [];
  const wrap = (x: number) => mod360 ? (x - 360 * Math.round(x / 360)) : x;
  for (let j = 1; j < values.length - 1; j++) {
    const a = values[j - 1];
    const b = values[j];
    const c = values[j + 1];
    const ab = Math.abs(wrap(b - a));
    const bc = Math.abs(wrap(b - c));
    const ac = Math.abs(wrap(a - c));
    if (ab > stepThreshold && bc > stepThreshold && ac < neighbourAgreement) {
      spikes.push({ bin: j, freq: freq[j], prev: a, spike: b, next: c });
    }
  }
  return spikes;
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

describe("3-band LR4 SUM (b140.15.10 user-spike regression)", () => {
  it("no 1-bin spikes in sum mag or phase for flat-meas 200/2000 LR4 3-way", async () => {
    const b0 = flatMeasBand("a", null, lr4(200));
    const b1 = flatMeasBand("b", lr4(200), lr4(2000));
    const b2 = flatMeasBand("c", lr4(2000), null);

    const result = await evaluateSum([b0, b1, b2]);
    const freq = result.freq;

    expect(result.sumCorrectedMag).not.toBeNull();
    expect(result.sumCorrectedPhase).not.toBeNull();

    const magSpikes = findSpikes(freq, result.sumCorrectedMag!, 1.0, 0.5, false);
    const phaseSpikes = findSpikes(freq, result.sumCorrectedPhase!, 30, 10, true);

    const perBandMagSpikes = result.perBandCorrected.map((pb, i) =>
      pb ? { i, spikes: findSpikes(freq, pb.mag, 1.0, 0.5, false) } : null);
    const perBandPhaseSpikes = result.perBandCorrected.map((pb, i) =>
      pb ? { i, spikes: findSpikes(freq, pb.phase, 30, 10, true) } : null);

    const lines: string[] = [];
    let total = 0;
    if (magSpikes.length > 0) {
      total += magSpikes.length;
      lines.push(`SUM mag: ${magSpikes.length} spikes`);
      for (const s of magSpikes.slice(0, 6))
        lines.push(`  bin ${s.bin} f=${s.freq.toFixed(1)}Hz prev=${s.prev.toFixed(3)} spike=${s.spike.toFixed(3)} next=${s.next.toFixed(3)}`);
    }
    if (phaseSpikes.length > 0) {
      total += phaseSpikes.length;
      lines.push(`SUM phase: ${phaseSpikes.length} spikes`);
      for (const s of phaseSpikes.slice(0, 6))
        lines.push(`  bin ${s.bin} f=${s.freq.toFixed(1)}Hz prev=${s.prev.toFixed(2)} spike=${s.spike.toFixed(2)} next=${s.next.toFixed(2)}`);
    }
    for (const pb of perBandMagSpikes) {
      if (pb && pb.spikes.length > 0) {
        total += pb.spikes.length;
        lines.push(`band ${pb.i} mag: ${pb.spikes.length} spikes`);
        for (const s of pb.spikes.slice(0, 6))
          lines.push(`  bin ${s.bin} f=${s.freq.toFixed(1)}Hz prev=${s.prev.toFixed(3)} spike=${s.spike.toFixed(3)} next=${s.next.toFixed(3)}`);
      }
    }
    for (const pb of perBandPhaseSpikes) {
      if (pb && pb.spikes.length > 0) {
        total += pb.spikes.length;
        lines.push(`band ${pb.i} phase: ${pb.spikes.length} spikes`);
        for (const s of pb.spikes.slice(0, 6))
          lines.push(`  bin ${s.bin} f=${s.freq.toFixed(1)}Hz prev=${s.prev.toFixed(2)} spike=${s.spike.toFixed(2)} next=${s.next.toFixed(2)}`);
      }
    }

    if (total > 0) {
      throw new Error(`\n=== TS SUM 3-band LR4 SPIKE REPORT ===\n${lines.join("\n")}\n`);
    }
  });
});
