// Tests for the b139.1 BandEvaluator. Two suites:
//   • snapshot — pin targetPhase / targetMag for the six canonical fixtures,
//   • equivalence — for SPL-style usage, evaluateBandFull must match the
//     legacy evaluateBand + addGaussianMinPhase pair to 1e-9.
// Tauri invoke is mocked so these run in vitest; the mock mirrors the Rust
// behaviour for the commands we actually call.

import { describe, it, expect, vi } from "vitest";
import { fixtureMeasurement, FIXTURE_CONFIGS } from "./fixtures/eval-fixtures";
import type { BandState } from "../../stores/bands";
import type { FilterConfig, TargetCurve } from "../types";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") return mockEvaluateTarget(args.target as TargetCurve, args.freq as number[]);
    if (cmd === "evaluate_target_standalone") {
      const f = mockLogGrid(5, 40000, 512);
      const r = mockEvaluateTarget(args.target as TargetCurve, f);
      return [f, r];
    }
    if (cmd === "compute_minimum_phase") return mockHilbert(args.freq as number[], args.magnitude as number[]);
    if (cmd === "compute_peq_complex") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0)];
    }
    if (cmd === "get_smoothed") return args.magnitude;
    if (cmd === "compute_impulse" || cmd === "compute_corrected_impulse") {
      return { time: [0, 1], impulse: [0, 0], step: [0, 0] };
    }
    if (cmd === "compute_cross_section") {
      // Mirror Rust apply_filter_public for Gaussian (phase=0, mag from
      // gaussianFilterMagDb). Non-Gaussian filters → zeros (fixtures 5/6
      // still snapshot deterministically because the freq grid + zero
      // contributions are reproducible).
      const freq = args.freq as number[];
      const hp = args.highPass as FilterConfig | null;
      const lp = args.lowPass as FilterConfig | null;
      const n = freq.length;
      const mag = new Array(n).fill(0);
      const phase = new Array(n).fill(0);
      if (hp && hp.filter_type === "Gaussian") {
        const m = gaussianFilterMagDb(freq, hp, false);
        for (let i = 0; i < n; i++) mag[i] += m[i];
      }
      if (lp && lp.filter_type === "Gaussian") {
        const m = gaussianFilterMagDb(freq, lp, true);
        for (let i = 0; i < n; i++) mag[i] += m[i];
      }
      return [mag, phase, 0];
    }
    if (cmd === "interpolate_log") {
      // Trivial pass-through — fixture grid already covers the requested span.
      return [args.freq, args.magnitude, args.phase ?? null];
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateBandFull } from "../band-evaluator";
import { gaussianFilterMagDb, subsonicMagDb } from "../plot-helpers";

function mockLogGrid(fmin: number, fmax: number, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(fmin * Math.pow(fmax / fmin, i / (n - 1)));
  return out;
}

/** Minimal Rust-mirror: applies HP magnitude (Gaussian + subsonic), LP not used
 *  in the test fixtures. Returns phase = 0 for Gaussian (Rust does), analytical
 *  phase for non-Gaussian filter types is not exercised by these fixtures. */
function mockEvaluateTarget(target: TargetCurve, freq: number[]): { magnitude: number[]; phase: number[] } {
  const mag = freq.map(() => target.reference_level_db ?? 0);
  const phase = freq.map(() => 0);
  const hp = target.high_pass;
  if (hp) {
    if (hp.filter_type === "Gaussian") {
      const hpMag = gaussianFilterMagDb(freq, hp, false);
      for (let i = 0; i < freq.length; i++) mag[i] += hpMag[i];
      if (hp.subsonic_protect === true && hp.freq_hz > 40) {
        const subDb = subsonicMagDb(freq, hp.freq_hz / 8);
        for (let i = 0; i < freq.length; i++) mag[i] += subDb[i];
      }
    }
    // Non-Gaussian: leave magnitude untouched in the mock (snapshot-tested
    // path; the real Rust behaviour is covered by cargo golden tests).
  }
  return { magnitude: mag, phase };
}

/** Deterministic Hilbert mock: integral of dB-magnitude slope, scaled. The
 *  exact value isn't physically meaningful — it just has to be a pure
 *  function of the magnitude so equivalence checks are stable. */
function mockHilbert(freq: number[], magnitude: number[]): number[] {
  const ph: number[] = [0];
  for (let i = 1; i < freq.length; i++) {
    const dy = magnitude[i] - magnitude[i - 1];
    ph.push(ph[i - 1] + dy * 0.5);
  }
  return ph;
}

function fixtureBand(hp: FilterConfig | null): BandState {
  const m = fixtureMeasurement();
  return {
    id: "test", name: "fixture",
    measurement: { name: "fixture", source_path: null, sample_rate: 48000, freq: m.freq, magnitude: m.magnitude, phase: m.phase, metadata: { date: null, mic: null, notes: null, smoothing: null } },
    measurementFile: null,
    settings: { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false },
    target: {
      reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: hp, low_pass: null, low_shelf: null, high_shelf: null,
    },
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  };
}

const round = (arr: number[]) => arr.map((v) => Math.round(v * 1e6) / 1e6);

describe("evaluateBandFull (b139.1)", () => {
  describe("snapshot — targetPhase + targetMag for 6 fixtures", () => {
    for (const cfg of FIXTURE_CONFIGS) {
      it(`${cfg.label} — targetMag`, async () => {
        const band = fixtureBand(cfg.hp);
        const r = await evaluateBandFull({ band });
        expect(r.targetMag).not.toBeNull();
        expect(round(r.targetMag!)).toMatchSnapshot();
      });
      it(`${cfg.label} — targetPhase`, async () => {
        const band = fixtureBand(cfg.hp);
        const r = await evaluateBandFull({ band });
        expect(r.targetPhase).not.toBeNull();
        expect(round(r.targetPhase!)).toMatchSnapshot();
      });
    }
  });

  // b139.4c: equivalence-with-legacy-evaluateBand suite removed — equivalence
  // baselined at b139.1 (snapshots above lock the canonical behaviour).
  // Legacy evaluateBand has been deleted from band-evaluation.ts.

  // b139.4c: pin correctedMag / correctedPhase for the same six fixtures.
  // Single-source-of-truth check — the four (linear × subsonic) Gaussian
  // combinations all flow through reconstructTargetPhase, so any
  // regression in that path shows up here.
  describe("snapshot — correctedMag + correctedPhase for 6 fixtures", () => {
    for (const cfg of FIXTURE_CONFIGS) {
      it(`${cfg.label} — correctedMag`, async () => {
        const band = fixtureBand(cfg.hp);
        const r = await evaluateBandFull({ band });
        expect(r.correctedMag).not.toBeNull();
        expect(round(r.correctedMag!)).toMatchSnapshot();
      });
      it(`${cfg.label} — correctedPhase`, async () => {
        const band = fixtureBand(cfg.hp);
        const r = await evaluateBandFull({ band });
        expect(r.correctedPhase).not.toBeNull();
        expect(round(r.correctedPhase!)).toMatchSnapshot();
      });
    }
  });
});
