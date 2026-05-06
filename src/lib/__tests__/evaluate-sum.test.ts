// b140.3.0 — minimal evaluateSum: coherent sum of per-band Σ Target only.
//
// Old b140.2.x snapshots (avgRef, fence, extension, perBandResampled,
// sumIr, …) are intentionally gone — that pipeline accumulated 11
// fix iterations without parity. The new tests pin only what the
// minimal aggregator promises: Σ Target = coherent sum of per-band
// targets with polarity + alignment_delay phase rotation.

import { describe, it, expect, vi } from "vitest";
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

  it("alignment delay rotates Σ phase linearly with frequency", async () => {
    // 1 ms delay on a single band → phase = 360 · f · 0.001 deg
    const a = flatBand("a", { alignmentDelay: 0.001 });
    const result = await evaluateSum([a]);
    expect(result.sumTargetPhase).not.toBeNull();
    const f = result.freq;
    const ph = result.sumTargetPhase!;
    // Pick an early bin where the phase wrap stays linear (well below
    // ±180°). At 100 Hz, expected = 360·100·0.001 = 36°.
    const idx = f.findIndex((v) => v >= 100);
    expect(ph[idx]).toBeCloseTo(36, 0);
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
    // Step measurement: -5 dB inside [300, 1400], 0 dB outside.
    // HP=200/LP=2000 → passband 300..1400 → offset = 5 dB.
    // Result: 0 dB inside passband, +5 dB outside.
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
    for (let i = 0; i < f.length; i++) m[i] = (f[i] >= 300 && f[i] <= 1400) ? -5 : 0;
    (a.measurement as any).magnitude = m;

    const result = await evaluateSum([a]);
    const mag = result.perBandCorrected[0]!.mag;
    // 1 kHz is inside passband — was -5 dB, now 0 dB.
    const i1k = result.freq.findIndex(v => v >= 1000);
    expect(mag[i1k]).toBeCloseTo(0, 1);
    // 100 Hz is outside passband — was 0 dB, now +5 dB.
    const i100 = result.freq.findIndex(v => v >= 100);
    expect(mag[i100]).toBeCloseTo(5, 1);
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

  it("targetEnabled=false → no offset applied", async () => {
    const a = bandWithMag("a", -3, { targetEnabled: false });
    const result = await evaluateSum([a]);
    const mag = result.perBandCorrected[0]!.mag;
    // No target → no offset → corrected stays at measurement level (-3 dB).
    const idx = mag.length / 2 | 0;
    expect(mag[idx]).toBeCloseTo(-3, 2);
  });

  it("offset < 0.01 dB → not applied", async () => {
    // measurement = 0 dB, target = 0 dB → offset ≈ 0 → no change.
    const result = await evaluateSum([flatBand("a")]);
    const mag = result.perBandCorrected[0]!.mag;
    const idx = mag.length / 2 | 0;
    expect(mag[idx]).toBeCloseTo(0, 4);
  });
});

describe("evaluateSum (b140.3.1.2) — width-aware excess limiter", () => {
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

  it("wide excess (1 octave) clips to target + 1 dB", async () => {
    const a = bandWithFilters();
    // Large hump in [500, 1000] (1 oct) — large enough that even after
    // normalize the excess remains far above 1 dB.
    setMagInRange(a, 500, 1000, 20);
    const result = await evaluateSum([a]);
    const f = result.freq;
    const mag = result.perBandCorrected[0]!.mag;
    const i700 = f.findIndex(v => v >= 700);
    // Wide region → clipFactor=1 → corrected = target + 1 dB. target=0.
    expect(mag[i700]).toBeCloseTo(1, 1);
  });

  it("narrow peak (≈ 1/16 oct) preserved", async () => {
    const a = bandWithFilters();
    // Very narrow hump around 1 kHz — width ≪ 1/8 oct.
    setMagInRange(a, 990, 1010, 5);
    const result = await evaluateSum([a]);
    const f = result.freq;
    const mag = result.perBandCorrected[0]!.mag;
    const i1k = f.findIndex(v => v >= 999);
    // Narrow → clipFactor=0 → peak preserved (offset is tiny since narrow).
    expect(mag[i1k]).toBeGreaterThan(4);
  });

  it("medium width (≈ 1/4 oct) soft transition", async () => {
    const a = bandWithFilters();
    // 1/4 oct: 1000..1189 Hz. clipFactor = (0.25-0.125)/(0.5-0.125) = 0.333.
    // Excess +5 → newEx ≈ 5*0.667 + 1*0.333 ≈ 3.67 (offset≈0 because the
    // hump is small relative to passband).
    setMagInRange(a, 1000, 1189, 5);
    const result = await evaluateSum([a]);
    const f = result.freq;
    const mag = result.perBandCorrected[0]!.mag;
    const i1100 = f.findIndex(v => v >= 1090);
    // Should be clipped down from ~5 toward ~3.67 — strictly between them.
    expect(mag[i1100]).toBeGreaterThan(2);
    expect(mag[i1100]).toBeLessThan(4.5);
  });

  it("excess outside limiter zone (passband ± 1 oct) not clipped", async () => {
    const a = bandWithFilters();
    // HP=200, LP=2000 → passband [300,1400], zone [150, 2800].
    // Hump well outside zone: 30..50 Hz. Should NOT be clipped.
    setMagInRange(a, 30, 50, 10);
    const result = await evaluateSum([a]);
    const f = result.freq;
    const mag = result.perBandCorrected[0]!.mag;
    const i40 = f.findIndex(v => v >= 40);
    // Out-of-zone bins are untouched by the limiter (offset is ~0 because
    // the hump is outside the normalize passband too).
    expect(mag[i40]).toBeGreaterThan(8);
  });

  it("targetEnabled=false → limiter not applied", async () => {
    const a = bandWithFilters();
    a.targetEnabled = false;
    setMagInRange(a, 500, 1000, 20);
    const result = await evaluateSum([a]);
    const f = result.freq;
    const mag = result.perBandCorrected[0]!.mag;
    const i700 = f.findIndex(v => v >= 700);
    // No target → no normalize, no limiter → corrected = measurement = 20.
    expect(mag[i700]).toBeCloseTo(20, 1);
  });
});
