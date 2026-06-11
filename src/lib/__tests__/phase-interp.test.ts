// b141.9: wrap-aware phase interpolation. Phases arrive from Rust wrapped
// to ±180° (atan2). At an LR crossover the phase passes through exactly
// ±180 at fc — linear lerp across the −179→+179 jump lands near 0° and
// poisons the bin, which partially cancels the coherent SUM (the narrow
// −3 dB dip at 200 Hz in the user's 3WAY project, 2026-06-11).
//
// Production parameters mirrored from test-fixtures/3way-dip/3WAY.pfproj:
// PF LR-2 (= squared BU-2) crossover at 200 Hz, band grids 1/96 oct
// starting at 20.29 Hz (woofer, n≈1010) and 69.98 Hz (mid, n≈879).

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Mock } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import type { BandState } from "../../stores/bands";
import type { FilterConfig } from "../types";

// ---------------------------------------------------------------------------
// Rust mirrors: PF LR-2 = (BU-2)², phase wrapped via final atan2 like
// target/mod.rs apply_filter (complex accumulation → single atan2).
// ---------------------------------------------------------------------------
function bu2(w: number, hp: boolean): { mag: number; phase: number } {
  const re = 1 - w * w;
  const im = Math.SQRT2 * w;
  const den = Math.hypot(re, im);
  const denPh = Math.atan2(im, re);
  if (hp) return { mag: (w * w) / den, phase: Math.PI - denPh };
  return { mag: 1 / den, phase: -denPh };
}

function pfLr2Db(freq: number[], fc: number, hp: boolean): { mag: number[]; phase: number[] } {
  const mag: number[] = [];
  const phase: number[] = [];
  for (const f of freq) {
    const b = bu2(f / fc, hp);
    const m2 = b.mag * b.mag;
    const ph2 = 2 * b.phase;
    mag.push(20 * Math.log10(Math.max(m2, 1e-20)));
    phase.push(Math.atan2(Math.sin(ph2), Math.cos(ph2)) * 180 / Math.PI); // wrapped
  }
  return { mag, phase };
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
      // Complex product of active filters, single atan2 at the end (Rust parity).
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
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateSum } from "../band-evaluator";
import { interpPhaseOnGrid } from "../band-evaluator/grid";
import { clearBandEvalCache } from "../band-evaluator/cache";

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

function bandWithMeas(name: string, freq: number[], hp: FilterConfig | null, lp: FilterConfig | null): BandState {
  return {
    id: name, name,
    measurement: {
      name, source_path: null, sample_rate: 48000,
      freq, magnitude: freq.map(() => 80), phase: freq.map(() => 0),
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
  } as unknown as BandState;
}

beforeEach(() => {
  clearBandEvalCache();
  (invoke as Mock).mockClear();
});

describe("interpPhaseOnGrid (b141.9)", () => {
  it("interpolates across a ±180 wrap along the shortest arc", () => {
    const src = [100, 200, 300];
    const ph = [179, -179, -177]; // wrap between 100 and 200
    const out = interpPhaseOnGrid(src, ph, [150]) as number[];
    // Shortest arc 179 → -179 is +2°; midpoint = 180 (≡ -180), NOT 0.
    const wrapped = Math.atan2(
      Math.sin(out[0] * Math.PI / 180), Math.cos(out[0] * Math.PI / 180),
    ) * 180 / Math.PI;
    expect(Math.abs(Math.abs(wrapped) - 180)).toBeLessThan(1e-9);
  });

  it("matches plain linear interpolation away from wraps", () => {
    const src = [100, 200];
    const ph = [10, 40];
    const out = interpPhaseOnGrid(src, ph, [150]) as number[];
    expect(out[0]).toBeCloseTo(25, 9);
  });

  it("honours the outside fence", () => {
    const out = interpPhaseOnGrid([100, 200], [10, 20], [50, 400], { outside: 0 }) as number[];
    expect(out).toEqual([0, 0]);
  });
});

describe("SUM coherent sum at LR crossover (b141.9 — user 3WAY dip)", () => {
  it("no narrow dip at the 200 Hz crossover when band grids differ", async () => {
    // Woofer: LP LR-2 @200 (PF convention), grid 20.29..29812 (1/96 oct).
    // Mid: HP LR-2 @200, grid 69.98..39930 — offset vs woofer grid, so the
    // ±180 wrap at fc lands between DIFFERENT common-grid bins per band.
    const woofer = bandWithMeas("Woofer", grid96(20.29, 29812), null, lrFilt(200));
    const mid = bandWithMeas("Mid", grid96(69.98, 39930), lrFilt(200), null);

    const r = await evaluateSum([woofer, mid], {});
    expect(r.sumCorrectedMag).not.toBeNull();
    const freq = r.freq;
    const mag = r.sumCorrectedMag!;

    // Local smoothness in 170..240 Hz: each bin vs the mean of its ±2
    // neighbours. The PF-LR complementarity residual is ≤ ~0.5 dB and
    // smooth; the wrap-lerp bug produced a 1.5–3 dB single-bin notch.
    for (let k = 2; k < freq.length - 2; k++) {
      if (freq[k] < 170 || freq[k] > 240) continue;
      const local = Math.abs(mag[k] - (mag[k - 2] + mag[k + 2]) / 2);
      expect(
        local,
        `narrow artifact at ${freq[k].toFixed(1)} Hz: ${local.toFixed(2)} dB`,
      ).toBeLessThan(0.6);
    }
  });
});
