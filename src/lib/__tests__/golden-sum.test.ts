/**
 * Phase 0 test (b140.10.2): golden SUM snapshots.
 *
 * Locks down `evaluateSum` output at sentinel frequencies for 6 canonical
 * multi-band scenarios. Tauri is mocked deterministically (same mock pattern
 * as evaluate-sum.test.ts) so snapshots stay stable across runs.
 *
 * Snapshots target the SUM aggregator behaviour explicitly:
 *   - sumTargetMag / sumTargetPhase                — coherent Σ target
 *   - sumCorrectedMag / sumCorrectedPhase          — coherent Σ corrected
 *   - correctedCoherent                            — fallback flag
 *   - perBandCorrected[i].{mag,phase}              — per-band post-normalize
 *
 * Values rounded to 6 decimals before snapshot (matches the existing
 * `golden-pipeline.test.ts` convention). Snapshots are committed under
 * `src/lib/__tests__/__snapshots__/golden-sum.test.ts.snap` — drift fails
 * the suite with a readable diff.
 *
 * Phase 5 (legacy SUM deletion) and any future SUM refactor must keep
 * these snapshots stable. Re-baseline by deleting the .snap file and
 * re-running vitest.
 */
import { describe, expect, it, vi } from "vitest";
import type { BandState } from "../../stores/bands";

// --- Deterministic Tauri mock ----------------------------------------------
// Identical contract to evaluate-sum.test.ts; kept inline so this file is
// self-contained and the snapshot output is reproducible from this file alone.

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
    if (cmd === "compute_impulse") {
      const mag = args.magnitude as number[];
      const phase = args.phase as number[];
      return {
        time: mag.map((_, i) => i / 48000),
        impulse: [...mag],
        step: [...phase],
      };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateSum } from "../band-evaluator";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const N = 512;

function logGrid(fmin: number, fmax: number, n: number): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = fmin * Math.pow(fmax / fmin, i / (n - 1));
  return out;
}

function flatBand(id: string, overrides: Partial<BandState> = {}): BandState {
  const freq = logGrid(20, 20000, N);
  return {
    id, name: id,
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

/** Sentinel frequencies — log-spread across audio band. */
const SENTINELS = [25, 100, 500, 1000, 5000, 10000, 18000] as const;

function pickSentinels(freq: number[], values: number[] | null): Record<string, number> | null {
  if (values === null) return null;
  const out: Record<string, number> = {};
  for (const f of SENTINELS) {
    const idx = freq.findIndex((v) => v >= f);
    if (idx < 0) continue;
    const raw = values[idx] ?? NaN;
    out[`${f}Hz`] = Math.round(raw * 1e6) / 1e6;
  }
  return out;
}

function snapshotShape(result: Awaited<ReturnType<typeof evaluateSum>>) {
  return {
    freqGridLen: result.freq.length,
    correctedCoherent: result.correctedCoherent,
    sumTargetMag: pickSentinels(result.freq, result.sumTargetMag),
    sumTargetPhase: pickSentinels(result.freq, result.sumTargetPhase),
    sumCorrectedMag: pickSentinels(result.freq, result.sumCorrectedMag),
    sumCorrectedPhase: pickSentinels(result.freq, result.sumCorrectedPhase),
    perBandCorrected: result.perBandCorrected.map((pb) =>
      pb === null ? null : {
        mag: pickSentinels(result.freq, pb.mag),
        phase: pickSentinels(result.freq, pb.phase),
      },
    ),
  };
}

function bw(order: number, freq: number, linear = true) {
  return {
    filter_type: "Butterworth" as const, order, freq_hz: freq,
    shape: null, linear_phase: linear, q: null, subsonic_protect: null,
  };
}

// ---------------------------------------------------------------------------
// Snapshot fixtures
// ---------------------------------------------------------------------------

describe("evaluateSum golden snapshots (b140.10.2 phase-0)", () => {
  it("single_flat_band — baseline sanity", async () => {
    const result = await evaluateSum([flatBand("a")]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("two_bands_coherent — identical flat bands sum to +6 dB", async () => {
    const result = await evaluateSum([flatBand("a"), flatBand("b")]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("two_bands_inverted — polarity cancellation", async () => {
    const result = await evaluateSum([
      flatBand("a"),
      flatBand("b", { inverted: true }),
    ]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("two_bands_alignment_delay — 1ms delay on band b rotates phase", async () => {
    const result = await evaluateSum([
      flatBand("a"),
      flatBand("b", { alignmentDelay: 0.001 }),
    ]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("two_bands_with_crossover — a=LP500, b=HP500", async () => {
    const a = flatBand("a");
    const b = flatBand("b");
    a.target.low_pass = bw(4, 500);
    b.target.high_pass = bw(4, 500);
    const result = await evaluateSum([a, b]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("partial_range_band — second band 1k-20k only, extension fills lows", async () => {
    const a = flatBand("a");
    const b = flatBand("b");
    const partialFreq = logGrid(1000, 20000, 256);
    (b.measurement as any).freq = partialFreq;
    (b.measurement as any).magnitude = new Array(256).fill(0);
    (b.measurement as any).phase = new Array(256).fill(10);
    const result = await evaluateSum([a, b]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("disabled_targets — both targets off → sumTargetMag=null", async () => {
    const result = await evaluateSum([
      flatBand("a", { targetEnabled: false }),
      flatBand("b", { targetEnabled: false }),
    ]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });

  it("power_sum_fallback — band b has null phase → correctedCoherent=false", async () => {
    const a = flatBand("a");
    const b = flatBand("b");
    (b.measurement as any).phase = null;
    const result = await evaluateSum([a, b]);
    expect(snapshotShape(result)).toMatchSnapshot();
  });
});
