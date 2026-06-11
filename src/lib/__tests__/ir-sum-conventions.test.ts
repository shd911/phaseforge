// b141.10: Σ IR/Step conventions.
//
// 1. Σ Corrected IR must weight bands exactly like the SPL Σ: per-band
//    normalize-to-target in the passband (measurements are merged/recorded
//    at arbitrary levels; the norm is "corrected sits at target", realised
//    live via mic + per-band attenuation). Raw-SPL weighting skewed the
//    step shape whenever band calibrations differed.
// 2. Positive alignmentDelay must mean LATER (HQPlayer delay convention):
//    phase ramp −360·f·τ. It was +360·f·τ (advance) — reference values
//    were sign-inverted vs what the user types into HQPlayer.
// 3. computeAutoAlign under the new convention: a LATE lower band gets 0,
//    the earlier bands get positive delay (all delays ≥ 0).
// 4. Saved projects carry the old convention — migrate via
//    new_i = max(old) − old_i (relative timing preserved, all ≥ 0).

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Mock } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import type { BandState } from "../../stores/bands";
import type { FilterConfig } from "../types";

interface ImpulseCall { freq: number[]; magnitude: number[]; phase: number[] }
const impulseCalls: ImpulseCall[] = [];

function bu2(w: number, hp: boolean): { mag: number; phase: number } {
  const re = 1 - w * w;
  const im = Math.SQRT2 * w;
  const den = Math.hypot(re, im);
  const denPh = Math.atan2(im, re);
  if (hp) return { mag: (w * w) / den, phase: Math.PI - denPh };
  return { mag: 1 / den, phase: -denPh };
}

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      const ref = args.target.reference_level_db ?? 0;
      return { magnitude: freq.map(() => ref), phase: freq.map(() => 0) };
    }
    if (cmd === "compute_cross_section") {
      const freq = args.freq as number[];
      const n = freq.length;
      const re = new Array(n).fill(1), im = new Array(n).fill(0);
      const mul = (fc: number, hp: boolean) => {
        for (let i = 0; i < n; i++) {
          const b = bu2(freq[i] / fc, hp);
          const m2 = b.mag * b.mag, p2 = 2 * b.phase;
          const fr = m2 * Math.cos(p2), fi = m2 * Math.sin(p2);
          const r2 = re[i] * fr - im[i] * fi;
          im[i] = re[i] * fi + im[i] * fr;
          re[i] = r2;
        }
      };
      const hp = args.highPass as FilterConfig | null;
      const lp = args.lowPass as FilterConfig | null;
      if (hp) mul(hp.freq_hz, true);
      if (lp) mul(lp.freq_hz, false);
      const mag = re.map((r, i) => 20 * Math.log10(Math.max(Math.hypot(r, im[i]), 1e-20)));
      const phase = re.map((r, i) => Math.atan2(im[i], r) * 180 / Math.PI);
      return [mag, phase, 0];
    }
    if (cmd === "compute_peq_complex") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0)];
    }
    if (cmd === "compute_minimum_phase") return (args.freq as number[]).map(() => 0);
    if (cmd === "get_smoothed") return args.magnitude;
    if (cmd === "compute_impulse") {
      impulseCalls.push({
        freq: [...args.freq], magnitude: [...args.magnitude], phase: [...args.phase],
      });
      return { time: [0, 1], impulse: [1, 0], step: [1, 1] };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateSum } from "../band-evaluator";
import { clearBandEvalCache } from "../band-evaluator/cache";
import { computeAutoAlign } from "../auto-align";
import { migrateDelayConvention } from "../types";

function grid96(fStart: number, fEnd: number): number[] {
  const out: number[] = [];
  for (let i = 0; ; i++) {
    const f = fStart * Math.pow(2, i / 96);
    if (f > fEnd) break;
    out.push(f);
  }
  return out;
}

function lrFilt(fc: number): FilterConfig {
  return {
    filter_type: "LinkwitzRiley", freq_hz: fc, order: 2,
    q: null, linear_phase: false, subsonic_protect: null,
  } as FilterConfig;
}

function bandWithMeas(
  name: string, freq: number[], levelDb: number,
  hp: FilterConfig | null, lp: FilterConfig | null,
  over?: Partial<BandState> & { phase?: number[] },
): BandState {
  const { phase, ...rest } = over ?? {};
  return {
    id: name, name,
    measurement: {
      name, source_path: null, sample_rate: 48000,
      freq, magnitude: freq.map(() => levelDb), phase: phase ?? freq.map(() => 0),
      metadata: { date: null, mic: null, notes: null, smoothing: null },
    },
    measurementFile: null,
    settings: { smoothing: "off" },
    target: {
      reference_level_db: 80, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: hp, low_pass: lp, low_shelf: null, high_shelf: null,
    },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
    ...rest,
  } as unknown as BandState;
}

const binNear = (freq: number[], f: number) => {
  let best = 0, bd = Infinity;
  for (let i = 0; i < freq.length; i++) {
    const d = Math.abs(freq[i] - f);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
};

beforeEach(() => {
  clearBandEvalCache();
  (invoke as Mock).mockClear();
  impulseCalls.length = 0;
});

async function correctedSumSpectrum(bands: BandState[]): Promise<ImpulseCall> {
  impulseCalls.length = 0;
  await evaluateSum(bands, { includeIr: true });
  // evaluateSum issues exactly meas → target → corrected compute_impulse.
  expect(impulseCalls.length).toBe(3);
  return impulseCalls[2];
}

describe("Σ corrected IR weighting (b141.10)", () => {
  it("band level miscalibration is normalized away (same as SPL Σ)", async () => {
    const mk = (midLevel: number) => [
      bandWithMeas("Woofer", grid96(20.29, 29812), 80, null, lrFilt(200)),
      bandWithMeas("Mid", grid96(69.98, 39930), midLevel, lrFilt(200), null),
    ];
    const base = await correctedSumSpectrum(mk(80));
    clearBandEvalCache();
    const hot = await correctedSumSpectrum(mk(86)); // mid measured +6 dB hot

    // Above the crossover the mid dominates: without normalization the hot
    // run sits ~6 dB higher; with SPL-parity normalization they match.
    for (const f of [500, 1000, 5000]) {
      const k = binNear(base.freq, f);
      expect(
        Math.abs(hot.magnitude[k] - base.magnitude[k]),
        `Σ corrected at ${f} Hz shifted by calibration`,
      ).toBeLessThan(1.0);
    }
  });
});

describe("alignmentDelay sign convention (b141.10 — HQPlayer parity)", () => {
  it("positive delay = LATER: phase ramp −360·f·τ on Σ corrected", async () => {
    const mk = (d: number) => [
      bandWithMeas("Woofer", grid96(20.29, 29812), 80, null, lrFilt(200), { alignmentDelay: d } as any),
    ];
    const r0 = await correctedSumSpectrum(mk(0));
    clearBandEvalCache();
    const r1 = await correctedSumSpectrum(mk(0.001));

    for (const f of [50, 100, 200]) {
      const k = binNear(r0.freq, f);
      let diff = r1.phase[k] - r0.phase[k];
      diff = ((diff + 180) % 360 + 360) % 360 - 180;
      const expected = -360 * r0.freq[k] * 0.001;
      expect(
        Math.abs(diff - expected),
        `phase ramp at ${f} Hz: got ${diff.toFixed(1)}°, expected ${expected.toFixed(1)}°`,
      ).toBeLessThan(2);
    }
  });
});

describe("computeAutoAlign under positive-is-late (b141.10)", () => {
  it("late woofer → woofer 0, earlier tweeter gets positive delay", async () => {
    const tau = 0.0005;
    const gridT = grid96(200, 20000);
    const gridW = grid96(20, 5000);
    const tweeter = bandWithMeas("T", gridT, 80, lrFilt(1000), null);
    const woofer = bandWithMeas("W", gridW, 80, null, lrFilt(1000), {
      phase: gridW.map(f => -360 * f * tau), // physically LATE by 0.5 ms
    });
    const res = await computeAutoAlign([tweeter, woofer]);
    expect(res.delays["W"]).toBeCloseTo(0, 4);
    expect(Math.abs(res.delays["T"] - tau)).toBeLessThan(1e-4);
  });
});

describe("migrateDelayConvention (b141.10)", () => {
  it("old advance values → equivalent positive-late delays", () => {
    expect(migrateDelayConvention([0.00015, 0.00019, 0]))
      .toEqual([0.00019 - 0.00015, 0, 0.00019].map(v => Math.round(v * 1e9) / 1e9));
  });
  it("zeros stay zeros; empty stays empty", () => {
    expect(migrateDelayConvention([0, 0])).toEqual([0, 0]);
    expect(migrateDelayConvention([])).toEqual([]);
  });
});
