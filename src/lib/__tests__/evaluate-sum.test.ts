// b140.3.0 — minimal evaluateSum: coherent sum of per-band Σ Target only.
//
// Old b140.2.x snapshots (avgRef, fence, extension, perBandResampled,
// sumIr, …) are intentionally gone — that pipeline accumulated 11
// fix iterations without parity. The new tests pin only what the
// minimal aggregator promises: Σ Target = coherent sum of per-band
// targets with polarity; alignment_delay rotates Σ corrected only (b141.113).

import { describe, it, expect, vi } from "vitest";
import { sumParts } from "./helpers/sum-parts";
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
    if (cmd === "evaluate_target_standalone") {
      const n = args.nPoints ?? 512;
      const fMin = args.fMin ?? 5, fMax = args.fMax ?? 40000;
      const refLevel = args.target?.reference_level_db ?? 0;
      const freq = new Array(n);
      for (let i = 0; i < n; i++) freq[i] = fMin * Math.pow(fMax / fMin, i / (n - 1));
      return [freq, { magnitude: freq.map(() => refLevel), phase: freq.map(() => 0) }];
    }
    if (cmd === "compute_minimum_phase") {
      const m = args.magnitude as number[];
      return new Array(m.length).fill(0);
    }
    if (cmd === "compute_peq_complex") {
      const f = args.freq as number[];
      return [new Array(f.length).fill(0), new Array(f.length).fill(0)];
    }
    if (cmd === "compute_cross_section") {
      const f = args.freq as number[];
      return [new Array(f.length).fill(0), new Array(f.length).fill(0), 0];
    }
    if (cmd === "compute_impulse" || cmd === "compute_sum_impulse") {
      // Deterministic non-trivial mock so different inputs → different outputs.
      const s = cmd === "compute_sum_impulse" ? sumParts(args.freq, args.parts) : null;
      const mag = (s ? s.magnitude : args.magnitude) as number[];
      const phase = (s ? s.phase : args.phase) as number[];
      return {
        time: mag.map((_, i) => i / 48000),
        impulse: [...mag],
        step: [...phase],
      };
    }
    if (cmd === "pick_fir_route") {
      const { hp, lp, linearMain, subsonicCutoffHz } = args;
      const realisable = (f: any) =>
        !f || f.filter_type === "LinkwitzRiley"
           || f.filter_type === "Butterworth"
           || f.filter_type === "Custom";
      if (linearMain) return "Cepstral";
      if (subsonicCutoffHz !== null) return "Cepstral";
      if (!realisable(hp)) return "Cepstral";
      if (!realisable(lp)) return "Cepstral";
      return "Iir";
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

describe("evaluateSum (b140.16.3) — no-measurement reuse path", () => {
  function noMeasBand(id: string, lpFc: number | null, hpFc: number | null): BandState {
    return {
      id, name: id,
      measurement: null,  // ← key: triggers Performance #2 reuse path
      measurementFile: null,
      settings: {
        smoothing: "off", delay_seconds: null, distance_meters: null,
        delay_removed: false, originalPhase: null, floorBounce: null,
        mergeSource: null, analysis: null, analysisDismissed: false,
      },
      target: {
        reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
        high_pass: hpFc ? {
          filter_type: "LinkwitzRiley", order: 4, freq_hz: hpFc,
          shape: null, linear_phase: false, q: null,
        } : null,
        low_pass: lpFc ? {
          filter_type: "LinkwitzRiley", order: 4, freq_hz: lpFc,
          shape: null, linear_phase: false, q: null,
        } : null,
        low_shelf: null, high_shelf: null,
      },
      targetEnabled: true, inverted: false, linkedToNext: false,
      peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
      firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
    };
  }

  it("3-band LR4 no-measurement: sumTarget populated, no corrected", async () => {
    // Mirrors user's flat.pfproj scenario
    const b0 = noMeasBand("a", 200, null);          // LP200
    const b1 = noMeasBand("b", 2000, 200);          // HP200 + LP2000
    const b2 = noMeasBand("c", null, 2000);         // HP2000
    const r = await evaluateSum([b0, b1, b2]);
    expect(r.sumTargetMag).not.toBeNull();
    expect(r.sumTargetPhase).not.toBeNull();
    expect(r.sumCorrectedMag).toBeNull();      // no measurement → no corrected
    expect(r.sumCorrectedPhase).toBeNull();
    expect(r.perBandTarget.length).toBe(3);
    for (const pb of r.perBandTarget) expect(pb).not.toBeNull();
  });
});

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

  it("b140.15.6: cancellation null produces finite phase (not NaN, not ulp-grid noise)", async () => {
    // Two antiphase flat bands with a 50ms alignmentDelay on one of them.
    // Pre-fix: large delay × high freq → cos/sin precision degrades to
    // ulp scale; when re/im are also ulp-scale (cancellation), atan2
    // returned random ±π values across adjacent bins (visible jaggedness
    // in the SUM phase trace around nulls).
    const a = flatBand("a");
    const b = flatBand("b", { inverted: true, alignmentDelay: 0.050 });
    const result = await evaluateSum([a, b]);
    const ph = result.sumTargetPhase!;
    for (let i = 0; i < ph.length; i++) {
      expect(Number.isFinite(ph[i])).toBe(true);
    }
  });

  it("alignment delay rotates Σ corrected linearly with frequency, not Σ target", async () => {
    // b141.10: positive delay = LATER (HQPlayer parity) →
    // phase = −360 · f · 0.001 deg for a 1 ms delay.
    // b141.113: DELAY compensates the measurement's offset — the target has
    // none, so Σ target stays at zero phase.
    const a = flatBand("a", { alignmentDelay: 0.001 });
    const result = await evaluateSum([a]);
    const f = result.freq;
    // Pick an early bin where the phase wrap stays linear (well below
    // ±180°). At 100 Hz, expected = −360·100·0.001 = −36°.
    const idx = f.findIndex((v) => v >= 100);
    expect(result.sumCorrectedPhase![idx]).toBeCloseTo(-36, 0);
    expect(result.sumTargetPhase![idx]).toBeCloseTo(0, 6);
  });

  it("b141.113: DELAY does not dig a hole in Σ target", async () => {
    // Jerez VIV: MF 0.53 ms / HF 0.67 ms on a 2621 Hz LR4 — applied to the
    // targets, the 0.14 ms difference fell Σ target −7.8 dB at 2.62 kHz.
    // Two flat targets with those delays must still sum to +6.02 dB.
    const result = await evaluateSum([
      flatBand("mf", { alignmentDelay: 0.00053 }),
      flatBand("hf", { alignmentDelay: 0.00067 }),
    ]);
    for (const v of result.sumTargetMag!) expect(v).toBeCloseTo(6.0206, 3);
  });

  it("targetEnabled=false drops band from the sum", async () => {
    const a = flatBand("a"); // 0 dB
    const b = flatBand("b", { targetEnabled: false });
    const result = await evaluateSum([a, b]);
    const mag = result.sumTargetMag!;
    // Only one band contributes → 0 dB, not +6 dB.
    expect(mag[mag.length / 2 | 0]).toBeCloseTo(0, 3);
  });

  it("evaluateSum returns perBandTarget for enabled bands", async () => {
    const result = await evaluateSum([flatBand("a"), flatBand("b")]);
    expect(result.perBandTarget).toHaveLength(2);
    expect(result.perBandTarget[0]).not.toBeNull();
    expect(result.perBandTarget[1]).not.toBeNull();
    expect(result.perBandTarget[0]!.mag).toHaveLength(result.freq.length);
    expect(result.perBandTarget[0]!.phase).toHaveLength(result.freq.length);
  });

  it("perBandTarget is null for disabled bands", async () => {
    const a = flatBand("a");
    const b = flatBand("b", { targetEnabled: false });
    const result = await evaluateSum([a, b]);
    expect(result.perBandTarget).toHaveLength(2);
    expect(result.perBandTarget[0]).not.toBeNull();
    expect(result.perBandTarget[1]).toBeNull();
  });
});

describe("evaluateSum (b140.3.1) — Σ Corrected", () => {
  it("two bands with measurement → coherent corrected sum +6 dB", async () => {
    // Both bands flat 0 dB measurement + phase=0 → coherent sum +6 dB.
    const result = await evaluateSum([flatBand("a"), flatBand("b")]);
    expect(result.sumCorrectedMag).not.toBeNull();
    expect(result.correctedCoherent).toBe(true);
    const mag = result.sumCorrectedMag!;
    const idx = mag.length / 2 | 0;
    expect(mag[idx]).toBeCloseTo(6.0206, 1);
  });

  it("polarity inversion in corrected → cancellation", async () => {
    const a = flatBand("a");
    const b = flatBand("b", { inverted: true });
    const result = await evaluateSum([a, b]);
    expect(result.correctedCoherent).toBe(true);
    const mag = result.sumCorrectedMag!;
    expect(mag[mag.length / 2 | 0]).toBeLessThan(-150);
  });

  it("missing measurement phase → power-sum fallback, correctedCoherent=false", async () => {
    const a = flatBand("a");
    const b = flatBand("b");
    // Strip phase from one band's measurement.
    (b.measurement as any).phase = null;
    const result = await evaluateSum([a, b]);
    expect(result.correctedCoherent).toBe(false);
    expect(result.sumCorrectedPhase).toBeNull();
    // Power sum of two 0 dB sources = 10·log10(2) ≈ 3.0103 dB
    const mag = result.sumCorrectedMag!;
    expect(mag[mag.length / 2 | 0]).toBeCloseTo(3.0103, 2);
  });

  it("no band has measurement → sumCorrectedMag=null", async () => {
    const a = flatBand("a");
    const b = flatBand("b");
    (a as any).measurement = null;
    (b as any).measurement = null;
    const result = await evaluateSum([a, b]);
    expect(result.sumCorrectedMag).toBeNull();
    expect(result.sumCorrectedPhase).toBeNull();
    expect(result.perBandCorrected[0]).toBeNull();
    expect(result.perBandCorrected[1]).toBeNull();
  });
});

describe("evaluateSum (b140.3.1.1) — per-band corrected normalize", () => {
  function bandWithMag(id: string, magDb: number, overrides: Partial<BandState> = {}): BandState {
    const b = flatBand(id, overrides);
    (b.measurement as any).magnitude = new Array(N).fill(magDb);
    return b;
  }

  it("corrected -3 dB, target 0 dB → after offset corrected = 0 dB", async () => {
    const result = await evaluateSum([bandWithMag("a", -3)]);
    expect(result.perBandCorrected[0]).not.toBeNull();
    const mag = result.perBandCorrected[0]!.mag;
    const idx = mag.length / 2 | 0;
    expect(mag[idx]).toBeCloseTo(0, 2);
  });

  it("offset uses per-band passband (HP·1.5 .. LP·0.7)", async () => {
    // Step measurement: +5 dB inside [300, 1400], 0 dB outside.
    // HP=200/LP=2000 → passband 300..1400 → offset = -5 dB.
    // After offset: 0 dB inside passband, -5 dB outside.
    // (Sign chosen so that the side regions are *deficit*, not excess —
    // otherwise the b140.3.1.4 global-shift kicks in. The test's purpose
    // is to verify that normalize uses passband-only, which is preserved
    // either way: inside-passband ends up at target.)
    const a = flatBand("a");
    a.target.high_pass = {
      filter_type: "Butterworth", order: 4, freq_hz: 200,
      shape: null, linear_phase: true, q: null,
    };
    a.target.low_pass = {
      filter_type: "Butterworth", order: 4, freq_hz: 2000,
      shape: null, linear_phase: true, q: null,
    };
    const f = a.measurement!.freq;
    const m = new Array(f.length);
    for (let i = 0; i < f.length; i++) m[i] = (f[i] >= 300 && f[i] <= 1400) ? 5 : 0;
    (a.measurement as any).magnitude = m;

    const result = await evaluateSum([a]);
    const mag = result.perBandCorrected[0]!.mag;
    // 1 kHz is inside passband — was +5 dB, now 0 dB.
    const i1k = result.freq.findIndex(v => v >= 1000);
    expect(mag[i1k]).toBeCloseTo(0, 1);
    // 100 Hz is outside passband — was 0 dB, now -5 dB (deficit, not clipped).
    const i100 = result.freq.findIndex(v => v >= 100);
    expect(mag[i100]).toBeCloseTo(-5, 1);
  });

  it("inverted passband (HP·1.5 ≥ LP·0.7) → fallback [200, 2000]", async () => {
    // HP=15000, LP=20: pbLow=22500, pbHigh=14 → fallback [200, 2000].
    const a = flatBand("a");
    a.target.high_pass = {
      filter_type: "Butterworth", order: 4, freq_hz: 15000,
      shape: null, linear_phase: true, q: null,
    };
    a.target.low_pass = {
      filter_type: "Butterworth", order: 4, freq_hz: 20,
      shape: null, linear_phase: true, q: null,
    };
    // Measurement: -5 dB in [200, 2000], 0 elsewhere → offset = +5 (fallback band).
    const f = a.measurement!.freq;
    const m = new Array(f.length);
    for (let i = 0; i < f.length; i++) m[i] = (f[i] >= 200 && f[i] <= 2000) ? -5 : 0;
    (a.measurement as any).magnitude = m;

    const result = await evaluateSum([a]);
    const mag = result.perBandCorrected[0]!.mag;
    const i1k = result.freq.findIndex(v => v >= 1000);
    expect(mag[i1k]).toBeCloseTo(0, 1);
  });

  it("targetEnabled=false → band left out of Σ Corrected (b141.103, audit 3.5)", async () => {
    // No target → no level to normalise to and no FIR: not summed at its raw
    // measurement level beside bands brought to their targets (Σ IR already
    // left it out).
    const a = bandWithMag("a", -3, { targetEnabled: false });
    const result = await evaluateSum([a]);
    expect(result.perBandCorrected[0]).toBeNull();
    expect(result.sumCorrectedMag).toBeNull();
  });

  it("offset < 0.01 dB → not applied", async () => {
    // measurement = 0 dB, target = 0 dB → offset ≈ 0 → no change.
    const result = await evaluateSum([flatBand("a")]);
    const mag = result.perBandCorrected[0]!.mag;
    const idx = mag.length / 2 | 0;
    expect(mag[idx]).toBeCloseTo(0, 4);
  });
});

describe("evaluateSum (b140.3.5) — Σ IR/Step via includeIr", () => {
  it("includeIr=false (default) → no ir field", async () => {
    const result = await evaluateSum([flatBand("a"), flatBand("b")]);
    expect(result.ir).toBeUndefined();
  });

  it("includeIr=true → ir.measurement / ir.target / ir.corrected populated", async () => {
    const result = await evaluateSum(
      [flatBand("a"), flatBand("b")],
      { includeIr: true },
    );
    expect(result.ir).toBeTruthy();
    expect(result.ir!.measurement).toBeTruthy();
    expect(result.ir!.target).toBeTruthy();
    expect(result.ir!.corrected).toBeTruthy();
    // Wide IR grid: 1024 main log bins + 32-bin noise-floor tail to Nyquist (b140.5).
    expect(result.ir!.target!.impulse.length).toBe(1024 + 32);
  });

  it("ir.target absent when no band has targetEnabled", async () => {
    const a = flatBand("a", { targetEnabled: false });
    const b = flatBand("b", { targetEnabled: false });
    const result = await evaluateSum([a, b], { includeIr: true });
    expect(result.ir).toBeTruthy();
    expect(result.ir!.target).toBeUndefined();
  });

  it("ir.measurement absent when no band has measurement", async () => {
    const a = flatBand("a");
    const b = flatBand("b");
    (a as any).measurement = null;
    (b as any).measurement = null;
    const result = await evaluateSum([a, b], { includeIr: true });
    expect(result.ir).toBeTruthy();
    expect(result.ir!.measurement).toBeUndefined();
  });

  it("polarity inversion → cancellation in Σ target IR (impulse ≈ 0)", async () => {
    // Two flat 0-dB target bands, one inverted → coherent sum cancels.
    // The mock impulse forwards magnitude → cancellation shows as -200 dB
    // magnitude bins, hence very small linear amplitudes in impulse.
    const a = flatBand("a");
    const b = flatBand("b", { inverted: true });
    const result = await evaluateSum([a, b], { includeIr: true });
    const imp = result.ir!.target!.impulse;
    // Mock impulse = magnitude (in dB). Cancellation → -200 (floor).
    expect(imp.every(v => v <= -150)).toBe(true);
  });
});

describe("evaluateSum (b141.97) — band level = passband mean only", () => {
  function bandWithFilters(): BandState {
    const b = flatBand("a");
    b.target.high_pass = {
      filter_type: "Butterworth", order: 4, freq_hz: 200,
      shape: null, linear_phase: true, q: null,
    };
    b.target.low_pass = {
      filter_type: "Butterworth", order: 4, freq_hz: 2000,
      shape: null, linear_phase: true, q: null,
    };
    return b;
  }

  function setMagInRange(b: BandState, fLo: number, fHi: number, value: number) {
    const f = b.measurement!.freq;
    const m = (b.measurement as any).magnitude as number[];
    for (let i = 0; i < f.length; i++) {
      if (f[i] >= fLo && f[i] <= fHi) m[i] = value;
    }
  }

  it("a wide bump stays as it is; the passband mean sits at target", async () => {
    const a = bandWithFilters();
    // 1-oct hump of +2 dB inside the passband (300–1400 Hz). The removed
    // b140.3.1.4 shift lowered the whole band until the hump sat at +0.1 dB.
    setMagInRange(a, 500, 1000, 2);
    const result = await evaluateSum([a]);
    const f = result.freq;
    const mag = result.perBandCorrected[0]!.mag;
    const tgt = result.perBandTarget?.[0]?.mag;
    let s = 0, n = 0;
    for (let i = 0; i < f.length; i++) {
      if (f[i] < 300 || f[i] > 1400) continue;
      s += mag[i] - (tgt ? tgt[i] : 0); n++;
    }
    expect(Math.abs(s / n)).toBeLessThan(0.05);
    const i700 = f.findIndex(v => v >= 700), i1200 = f.findIndex(v => v >= 1200);
    expect(mag[i700] - mag[i1200]).toBeGreaterThan(1.9);
  });
});

describe("evaluateSum (b140.3.1.5) — extension via target + Hilbert", () => {
  function partialFlatBand(id: string, fMin: number, fMax: number, n = 256, phaseDeg = 0): BandState {
    const b = flatBand(id);
    const freq: number[] = new Array(n);
    for (let i = 0; i < n; i++) freq[i] = fMin * Math.pow(fMax / fMin, i / (n - 1));
    (b.measurement as any).freq = freq;
    (b.measurement as any).magnitude = new Array(n).fill(0);
    (b.measurement as any).phase = new Array(n).fill(phaseDeg);
    return b;
  }

  it("magnitude extension follows target shape outside native (no -200 fence)", async () => {
    // Band a: full 20–20000 (anchors common grid).
    // Band b: native 1000–20000 only → bins below 1000 must be filled
    // via target (flat 0 dB in mock) + boundary offset (=0).
    const a = flatBand("a");
    const b = partialFlatBand("b", 1000, 20000);
    const result = await evaluateSum([a, b]);
    const mag = result.perBandCorrected[1]!.mag;
    const i100 = result.freq.findIndex(v => v >= 100);
    // Without extension this would be fenced to -200; with extension it
    // tracks the (flat) target.
    expect(mag[i100]).toBeGreaterThan(-50);
    expect(mag[i100]).toBeCloseTo(0, 1);
  });

  it("phase smoothly continues from native via Hilbert (no boundary jump)", async () => {
    // Measurement.phase = 10° everywhere on native. Mocked Hilbert returns
    // 0 → boundary offset = 10° → extended phase = 0 + 10 = 10° outside.
    // Across the boundary the phase is continuous (10° on both sides).
    const a = flatBand("a");
    const b = partialFlatBand("b", 1000, 20000, 256, 10);
    const result = await evaluateSum([a, b]);
    const phase = result.perBandCorrected[1]!.phase;
    const i100 = result.freq.findIndex(v => v >= 100);
    const i1k = result.freq.findIndex(v => v >= 1000);
    expect(phase[i1k]).toBeCloseTo(10, 1);
    expect(phase[i100]).toBeCloseTo(10, 1);
  });

  it("targetEnabled=false → only the bands with a target make Σ Corrected (b141.103)", async () => {
    const a = flatBand("a");
    const b = partialFlatBand("b", 1000, 20000);
    b.targetEnabled = false;
    const both = await evaluateSum([a, b]);
    const alone = await evaluateSum([flatBand("a")]);
    expect(both.perBandCorrected[1]).toBeNull();
    const i = both.freq.findIndex(v => v >= 2000);
    expect(both.sumCorrectedMag![i]).toBeCloseTo(alone.sumCorrectedMag![i], 6);
  });

  it("phase=null in measurement → power-sum fallback (no Hilbert call)", async () => {
    // Strip phase from a partial-range measurement → corrected phase null
    // → power-sum at Σ level, no extPhase.
    const a = flatBand("a");
    const b = partialFlatBand("b", 1000, 20000);
    (b.measurement as any).phase = null;
    const result = await evaluateSum([a, b]);
    expect(result.correctedCoherent).toBe(false);
    expect(result.sumCorrectedPhase).toBeNull();
  });
});

import { passbandRange } from "../band-evaluator/extension";
describe("passbandRange (b141.69)", () => {
  it("normal band", () => { expect(passbandRange(100, 2000)).toEqual([150, 1400]); });
  it("narrow band uses its crossover span", () => { expect(passbandRange(1000, 1400)).toEqual([1000, 1400]); });
  it("inverted pair keeps 200–2000", () => { expect(passbandRange(15000, 20)).toEqual([200, 2000]); });
  it("no filters", () => { expect(passbandRange(null, null)).toEqual([30, 14000]); });
});
