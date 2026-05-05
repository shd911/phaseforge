// b140.2.0 — unit tests for evaluateSum.
//
// Pre-integration cover. evaluateSum is currently dead code in the
// production tree (FrequencyPlot's renderSumMode runs an inline pipeline);
// these tests pin its observable behaviour so the b140.2.1 UI swap can
// land safely.

import { describe, it, expect, vi } from "vitest";
import type { BandState } from "../../stores/bands";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      const refLevel = args.target?.reference_level_db ?? 0;
      return { magnitude: freq.map(() => refLevel), phase: freq.map(() => 0) };
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
      // Flat magnitude → zero phase. Tests use only flat configs so this
      // path returns zeros and never actually contributes rotation.
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
    if (cmd === "compute_impulse" || cmd === "compute_corrected_impulse") {
      return { time: [0, 1], impulse: [0, 0], step: [0, 0] };
    }
    if (cmd === "compute_cross_section") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0), 0];
    }
    if (cmd === "interpolate_log") {
      // Real linear interpolation on log-spaced output grid.
      const srcFreq = args.freq as number[];
      const srcMag = args.magnitude as number[];
      const srcPhase = (args.phase as number[] | null | undefined) ?? null;
      const n = args.nPoints as number;
      const fMin = args.fMin as number;
      const fMax = args.fMax as number;
      const tgtFreq: number[] = new Array(n);
      for (let i = 0; i < n; i++) {
        tgtFreq[i] = fMin * Math.pow(fMax / fMin, i / (n - 1));
      }
      const interp = (data: number[]): number[] =>
        tgtFreq.map((f) => {
          if (f <= srcFreq[0]) return data[0];
          if (f >= srcFreq[srcFreq.length - 1]) return data[srcFreq.length - 1];
          let lo = 0, hi = srcFreq.length - 1;
          while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (srcFreq[mid] <= f) lo = mid;
            else hi = mid;
          }
          const t = (f - srcFreq[lo]) / (srcFreq[hi] - srcFreq[lo]);
          return data[lo] + t * (data[hi] - data[lo]);
        });
      const mag = interp(srcMag);
      const phase = srcPhase ? interp(srcPhase) : null;
      return [tgtFreq, mag, phase];
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateSum, resampleOntoGrid } from "../band-evaluator";

const N = 512;
function logGrid(fmin: number, fmax: number, n: number): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fmin * Math.pow(fmax / fmin, i / (n - 1));
  return out;
}

/** Flat measurement (mag = 0 dB, phase = 0°) on a 20 Hz – 20 kHz log grid. */
function flatBand(id: string): BandState {
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
  };
}

function nearest(freq: number[], target: number): number {
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < freq.length; i++) {
    const d = Math.abs(freq[i] - target);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  return bestIdx;
}

const round = (arr: number[], digits = 4) => {
  const k = Math.pow(10, digits);
  return arr.map((v) => Math.round(v * k) / k);
};

describe("evaluateSum — coherent magnitude (b140.2.0)", () => {
  it("two identical flat bands sum to +6 dB across the band", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    const result = await evaluateSum(bands);
    expect(result.sumTargetMag).not.toBeNull();
    expect(result.sumCorrectedMag).not.toBeNull();
    const idx1k = nearest(result.freq, 1000);
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(6.02, 1);
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(6.02, 1);
    // Phase: no delay, no inversion → 0°.
    expect(Math.abs(result.sumTargetPhase![idx1k])).toBeLessThan(0.01);
  });

  it("three identical flat bands sum to +9.54 dB", async () => {
    const bands = [flatBand("a"), flatBand("b"), flatBand("c")];
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // 20·log10(3) = 9.542
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(9.54, 1);
  });
});

describe("evaluateSum — polarity inversion", () => {
  it("two flat bands with one inverted cancel to far below 0 dB", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].inverted = true;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // 1 + (-1) = 0 → -∞ dB (clamped to -200 in coherentSum).
    expect(result.sumTargetMag![idx1k]).toBeLessThan(-60);
  });

  it("inversion of a single band shows up as 180° phase", async () => {
    const bands = [flatBand("a")];
    bands[0].inverted = true;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    expect(Math.abs(result.sumTargetPhase![idx1k])).toBeCloseTo(180, 0);
  });
});

describe("evaluateSum — alignment delay phase rotation", () => {
  it("0.5 ms delay on a single band rotates phase to ±180° at 1 kHz", async () => {
    const bands = [flatBand("a")];
    bands[0].alignmentDelay = 0.0005;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // Predicted phase from the actual nearest-bin freq, not 1000 exactly:
    // 360·f·delay (mod 360, wrapped to ±180).
    const expected = (360 * result.freq[idx1k] * 0.0005) % 360;
    const expectedWrapped = expected > 180 ? expected - 360 : expected;
    expect(result.sumTargetPhase![idx1k]).toBeCloseTo(expectedWrapped, 1);
  });

  it("0.5 ms delay between two bands cancels heavily at 1 kHz", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].alignmentDelay = 0.0005;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // Near-1k bin lands a fraction of a degree off 180° due to log grid
    // quantisation, so the sum sits around -35 dB instead of -∞. The point
    // is destructive interference — > 25 dB suppression vs +6 dB coherent.
    expect(result.sumTargetMag![idx1k]).toBeLessThan(-25);
  });

  it("0.25 ms delay at 1 kHz gives ~90° phase shift, ~+3 dB sum", async () => {
    // 360·1000·0.00025 = 90°. Two unit vectors at 0° and 90° sum to √2 → +3 dB.
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].alignmentDelay = 0.00025;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(3.01, 1);
  });
});

describe("evaluateSum — power-sum fallback (b140.2.0.5)", () => {
  // Mixed phase availability triggers the incoherent fallback: corrected
  // sum becomes 10·log10(Σ 10^(m/10)), polarity ignored, phase null.
  // Target sum is always coherent because targetPhase is reconstructed
  // analytically and never depends on measurement.phase.
  it("two flat bands without phase → power sum +3.01 dB, coherent=false, phase null", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[0].measurement!.phase = null;
    bands[1].measurement!.phase = null;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    expect(result.coherent).toBe(false);
    expect(result.sumCorrectedPhase).toBeNull();
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(3.01, 1);
  });

  it("mixed phase (one band w/o phase) → fallback fires, +3.01 dB, coherent=false", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].measurement!.phase = null;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    expect(result.coherent).toBe(false);
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(3.01, 1);
    // Target sum unaffected — both targets coherent regardless.
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(6.02, 1);
  });

  it("polarity ignored under power-sum fallback (no cancellation)", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[0].measurement!.phase = null;
    bands[1].measurement!.phase = null;
    bands[1].inverted = true; // would cancel under coherent sum
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    expect(result.coherent).toBe(false);
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(3.01, 1);
  });

  it("all bands with phase → coherent=true, +6.02 dB (no fallback)", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    expect(result.coherent).toBe(true);
    expect(result.sumCorrectedPhase).not.toBeNull();
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(6.02, 1);
  });

  it("band without targetEnabled → dropped from target sum", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].targetEnabled = false;
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // Only band[0] contributes a target.
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(0, 1);
    // Corrected still has both bands → coherent +6.02 dB.
    expect(result.coherent).toBe(true);
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(6.02, 1);
  });
});

describe("evaluateSum — perBandResampled (b140.2.1.5)", () => {
  function bandWithRange(id: string, fMin: number, fMax: number, levelDb = 0): BandState {
    const b = flatBand(id);
    b.measurement!.freq = logGrid(fMin, fMax, N);
    b.measurement!.magnitude = new Array(N).fill(levelDb);
    b.measurement!.phase = new Array(N).fill(0);
    return b;
  }

  it("perBandResampled arrays have length = common grid", async () => {
    const bands = [
      bandWithRange("a", 20, 22000),
      bandWithRange("b", 1000, 22000),
    ];
    const result = await evaluateSum(bands);
    expect(result.perBandResampled).toHaveLength(2);
    for (const r of result.perBandResampled) {
      if (r.measurementMag) expect(r.measurementMag.length).toBe(result.freq.length);
      if (r.measurementPhase) expect(r.measurementPhase.length).toBe(result.freq.length);
      if (r.targetMag) expect(r.targetMag.length).toBe(result.freq.length);
      if (r.targetPhase) expect(r.targetPhase.length).toBe(result.freq.length);
      if (r.correctedMag) expect(r.correctedMag.length).toBe(result.freq.length);
      if (r.correctedPhase) expect(r.correctedPhase.length).toBe(result.freq.length);
    }
  });

  it("supertweeter perBandResampled below native range follows extension (b140.2.1.7)", async () => {
    // flatBand has a flat target curve (no HP/LP). With globalRef = 0 the
    // analytical target on the common grid is 0 dB everywhere — extension
    // continues the curve at 0 dB rather than dropping to -200 dB silence.
    // The b140.2.1.4 silence-fence is now opt-in (default-options call).
    const bands = [
      bandWithRange("a", 20, 22000),
      bandWithRange("b", 1000, 22000), // supertweeter
    ];
    const result = await evaluateSum(bands);
    const idx100 = nearest(result.freq, 100);
    const supertweeter = result.perBandResampled[1];
    expect(supertweeter.measurementMag![idx100]).toBeCloseTo(0, 1);
    // In native range — same flat 0 dB.
    const idx5k = nearest(result.freq, 5000);
    expect(supertweeter.measurementMag![idx5k]).toBeCloseTo(0, 1);
  });

  it("first band keeps its full-range data (no spurious fencing)", async () => {
    const bands = [
      bandWithRange("a", 20, 22000),
      bandWithRange("b", 1000, 22000),
    ];
    const result = await evaluateSum(bands);
    const wide = result.perBandResampled[0];
    const idx100 = nearest(result.freq, 100);
    const idx5k = nearest(result.freq, 5000);
    expect(wide.measurementMag![idx100]).toBeCloseTo(0, 1);
    expect(wide.measurementMag![idx5k]).toBeCloseTo(0, 1);
  });
});

describe("resampleOntoGrid extension (b140.2.1.7)", () => {
  // Direct-call tests for the three out-of-range strategies of
  // resampleOntoGrid: target-shape extension, log-linear trend fallback,
  // and the -200 dB silence fence (default).
  it("target shape extension matches measurement at boundary", async () => {
    // Source: flat 0 dB measurement on [1k, 20k] (supertweeter).
    // Target: rolloff at low end (-40 dB at 100 Hz, 0 dB at 1k+).
    // Extension at 100 Hz should follow target with offset = 0 - 0 = 0 →
    // expect ≈ -40 dB.
    const targetFreq = [100, 1000, 10000];
    const result = await resampleOntoGrid(
      [1000, 20000], [0, 0], null, targetFreq,
      { extensionTargetMag: [-40, 0, 0] },
    );
    expect(result.mag).not.toBeNull();
    expect(result.mag![0]).toBeCloseTo(-40, 0);
    // In-range bins return measurement value (0 dB).
    expect(result.mag![1]).toBeCloseTo(0, 1);
    expect(result.mag![2]).toBeCloseTo(0, 1);
  });

  it("trend fallback extrapolates log-linear slope from tail", async () => {
    // Flat measurement on [200, 22000] → tail slope = 0 dB/oct → extension
    // stays at 0 dB.
    const result = await resampleOntoGrid(
      [200, 250, 300, 22000], [0, 0, 0, 0], null, [100, 200],
      { fallbackToTrend: true },
    );
    expect(result.mag![0]).toBeCloseTo(0, 0);
  });

  it("trend fallback follows a +12 dB/oct rising tail (HP-rolloff shape)", async () => {
    // Source rising at +12 dB / octave near fLo — typical HP filter
    // rolloff seen in the lowest measurement bins. Extending one octave
    // below fLo (100 Hz given fLo = 200) drops 12 dB.
    const srcFreq = [200, 210, 225, 235, 800];
    const srcMag = srcFreq.map(f => 12 * Math.log2(f / 200));
    const result = await resampleOntoGrid(
      srcFreq, srcMag, null, [100, 200],
      { fallbackToTrend: true },
    );
    expect(result.mag![0]).toBeCloseTo(-12, 0);
  });

  it("no options → -200 dB fence (b140.2.1.4 fallback)", async () => {
    const result = await resampleOntoGrid(
      [200, 22000], [0, 0], null, [100, 200],
    );
    expect(result.mag![0]).toBeLessThan(-150);
  });
});

describe("evaluateSum — globalRef + corrOffset (b140.2.1.3)", () => {
  // Multi-way fixture: band A's measurement averages 0 dB in the passband,
  // band B sits 30 dB lower. Pre-b140.2.1.3 the New pipeline used each
  // band's own autoRef → band B's target floated 30 dB below band A's, the
  // coherent sum was lopsided. After the fix globalRef = max passband-avg
  // = 0 dB, both targets are aligned, sum behaves as in renderSumMode.
  function flatBandAt(id: string, levelDb: number): BandState {
    const b = flatBand(id);
    b.measurement!.magnitude = new Array(N).fill(levelDb);
    return b;
  }

  it("Σ Target uses globalRef across bands with different SPL", async () => {
    const bands = [flatBandAt("a", 0), flatBandAt("b", -30)];
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // Both targets aligned to globalRef = 0 dB → coherent sum = +6.02 dB.
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(6.02, 1);
  });

  it("Σ Corrected pulls quiet band up to target via per-band corrOffset", async () => {
    const bands = [flatBandAt("a", 0), flatBandAt("b", -30)];
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // Per-band corrOffset: band A no shift (0 vs 0), band B +30 dB to
    // match its globalRef-aligned target. Σ Corrected = +6.02 dB.
    expect(result.sumCorrectedMag![idx1k]).toBeCloseTo(6.02, 1);
  });

  it("single-band project: globalRef collapses to that band's passband avg", async () => {
    const bands = [flatBandAt("a", -12.5)];
    const result = await evaluateSum(bands);
    const idx1k = nearest(result.freq, 1000);
    // Single band's target shifted to globalRef = -12.5 dB.
    expect(result.sumTargetMag![idx1k]).toBeCloseTo(-12.5, 1);
  });
});

describe("evaluateSum — Σ measurement (b140.2.1.1)", () => {
  it("two flat measurements with phase sum to +6.02 dB coherent", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    const result = await evaluateSum(bands);
    expect(result.sumMeasurementMag).not.toBeNull();
    expect(result.sumMeasurementPhase).not.toBeNull();
    const idx1k = nearest(result.freq, 1000);
    expect(result.coherentMeasurement).toBe(true);
    expect(result.sumMeasurementMag![idx1k]).toBeCloseTo(6.02, 1);
    expect(Math.abs(result.sumMeasurementPhase![idx1k])).toBeLessThan(0.01);
  });

  it("two flat measurements without phase fall back to power-sum +3.01 dB", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[0].measurement!.phase = null;
    bands[1].measurement!.phase = null;
    const result = await evaluateSum(bands);
    expect(result.coherentMeasurement).toBe(false);
    expect(result.sumMeasurementPhase).toBeNull();
    const idx1k = nearest(result.freq, 1000);
    expect(result.sumMeasurementMag![idx1k]).toBeCloseTo(3.01, 1);
  });

  it("polarity inversion under coherent measurement → cancellation", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].inverted = true;
    const result = await evaluateSum(bands);
    expect(result.coherentMeasurement).toBe(true);
    const idx1k = nearest(result.freq, 1000);
    expect(result.sumMeasurementMag![idx1k]).toBeLessThan(-60);
  });

  it("no measurements anywhere → sumMeasurement* null", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[0].measurement = null;
    bands[1].measurement = null;
    const result = await evaluateSum(bands);
    expect(result.sumMeasurementMag).toBeNull();
    expect(result.sumMeasurementPhase).toBeNull();
    // Vacuous coherent flag stays true (no incoherent path was taken).
    expect(result.coherentMeasurement).toBe(true);
  });
});

describe("evaluateSum — common grid construction", () => {
  it("returns a freq grid covering the union of band ranges", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    // Make band B narrower (50 Hz – 5 kHz).
    bands[1].measurement!.freq = logGrid(50, 5000, N);
    const result = await evaluateSum(bands);
    expect(result.freq[0]).toBeCloseTo(20, 1);
    expect(result.freq[result.freq.length - 1]).toBeCloseTo(20000, 0);
  });

  it("honours an explicit options.freq grid", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    const customFreq = logGrid(100, 10000, 256);
    const result = await evaluateSum(bands, { freq: customFreq });
    expect(result.freq).toBe(customFreq);
    expect(result.sumTargetMag!.length).toBe(256);
  });
});

describe("evaluateSum — snapshot baseline", () => {
  it("two-band identical sum — target mag/phase snapshot", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    const result = await evaluateSum(bands);
    expect(round(result.sumTargetMag!)).toMatchSnapshot("sumTargetMag-2band-identical");
    expect(round(result.sumTargetPhase!)).toMatchSnapshot("sumTargetPhase-2band-identical");
  });

  it("two-band with inversion — corrected mag/phase snapshot", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].inverted = true;
    const result = await evaluateSum(bands);
    expect(round(result.sumCorrectedMag!)).toMatchSnapshot("sumCorrectedMag-2band-inverted");
    expect(round(result.sumCorrectedPhase!)).toMatchSnapshot("sumCorrectedPhase-2band-inverted");
  });

  it("two-band with delay — phase snapshot", async () => {
    const bands = [flatBand("a"), flatBand("b")];
    bands[1].alignmentDelay = 0.0005;
    const result = await evaluateSum(bands);
    expect(round(result.sumTargetMag!)).toMatchSnapshot("sumTargetMag-2band-delay500us");
    expect(round(result.sumTargetPhase!)).toMatchSnapshot("sumTargetPhase-2band-delay500us");
  });
});
