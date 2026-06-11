// b141.7: BandEvalResult cache. evaluateBandFull / evaluateSum memoize on
// band content + request options; display-only re-renders must not re-run
// the DSP IPC pipeline. Tauri invoke is mocked (counting) — a cache hit is
// observable as "no new invoke calls".

import { describe, it, expect, vi, beforeEach } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import type { Mock } from "vitest";
import type { BandState } from "../../stores/bands";
import type { FilterConfig, PeqBand, TargetCurve } from "../types";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      const freq = args.freq as number[];
      return { magnitude: freq.map(() => 0), phase: freq.map(() => 0) };
    }
    if (cmd === "evaluate_target_standalone") {
      const f = mockLogGrid(5, 40000, 512);
      return [f, { magnitude: f.map(() => 0), phase: f.map(() => 0) }];
    }
    if (cmd === "compute_minimum_phase") return (args.freq as number[]).map(() => 0);
    if (cmd === "compute_peq_complex") {
      const n = (args.freq as number[]).length;
      // Depend on sampleRate so wrong cache keys would be visible in data too.
      return [new Array(n).fill(args.sampleRate / 48000), new Array(n).fill(0)];
    }
    if (cmd === "get_smoothed") return args.magnitude;
    if (cmd === "compute_cross_section") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0), 0];
    }
    if (cmd === "interpolate_log") return [args.freq, args.magnitude, args.phase ?? null];
    if (cmd === "compute_impulse") return { time: [0, 1], impulse: [1, 0], step: [1, 1] };
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { evaluateBandFull, evaluateSum } from "../band-evaluator";
import { clearBandEvalCache } from "../band-evaluator/cache";

function mockLogGrid(fmin: number, fmax: number, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(fmin * Math.pow(fmax / fmin, i / (n - 1)));
  return out;
}

function lr4hp(): FilterConfig {
  return {
    filter_type: "LinkwitzRiley", freq_hz: 100, order: 4,
    q: null, linear_phase: false, subsonic_protect: null,
  } as FilterConfig;
}

function makeBand(over?: Partial<BandState>): BandState {
  return {
    id: "cache-test", name: "cache-fixture",
    measurement: null, measurementFile: null,
    settings: { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false },
    target: {
      reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: lr4hp(), low_pass: null, low_shelf: null, high_shelf: null,
    } as TargetCurve,
    targetEnabled: true, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
    ...over,
  } as BandState;
}

function makeMeasurement() {
  const freq = mockLogGrid(20, 20000, 64);
  return {
    name: "m", source_path: null, sample_rate: 48000,
    freq, magnitude: freq.map(() => 80), phase: freq.map(() => 0),
    metadata: { date: null, mic: null, notes: null, smoothing: null },
  };
}

const peq = (over?: Partial<PeqBand>): PeqBand => ({
  freq_hz: 1000, gain_db: -3, q: 2, enabled: true, filter_type: "Peaking" as any, ...over,
});

const invokeCount = () => (invoke as Mock).mock.calls.length;

beforeEach(() => {
  clearBandEvalCache();
  (invoke as Mock).mockClear();
});

describe("evaluateBandFull cache (b141.7)", () => {
  it("identical repeat call → zero extra IPC", async () => {
    const band = makeBand();
    const r1 = await evaluateBandFull({ band, sampleRate: 48000 });
    const after1 = invokeCount();
    expect(after1).toBeGreaterThan(0);
    const r2 = await evaluateBandFull({ band, sampleRate: 48000 });
    expect(invokeCount()).toBe(after1);
    expect(r2.targetMag).toEqual(r1.targetMag);
  });

  it("returns a fresh clone — mutating a result does not poison the cache", async () => {
    const band = makeBand();
    const r1 = await evaluateBandFull({ band });
    const r2 = await evaluateBandFull({ band });
    expect(r2).not.toBe(r1);
    expect(r2.targetMag).not.toBe(r1.targetMag);
    r1.targetMag![0] = 999;
    const r3 = await evaluateBandFull({ band });
    expect(r3.targetMag![0]).not.toBe(999);
  });

  it("enabled PEQ change → recompute", async () => {
    const band = makeBand({ peqBands: [peq()] });
    await evaluateBandFull({ band });
    const after1 = invokeCount();
    const band2 = makeBand({ peqBands: [peq({ gain_db: -6 })] });
    await evaluateBandFull({ band: band2 });
    expect(invokeCount()).toBeGreaterThan(after1);
  });

  it("disabled PEQ edits do not invalidate", async () => {
    const band = makeBand({ peqBands: [peq({ enabled: false })] });
    await evaluateBandFull({ band });
    const after1 = invokeCount();
    const band2 = makeBand({ peqBands: [peq({ enabled: false, gain_db: 12 })] });
    await evaluateBandFull({ band: band2 });
    expect(invokeCount()).toBe(after1);
  });

  it("new measurement object (same content) → recompute (identity-keyed)", async () => {
    const band = makeBand({ measurement: makeMeasurement() as any });
    await evaluateBandFull({ band });
    const after1 = invokeCount();
    // Same band, same measurement object → hit.
    await evaluateBandFull({ band });
    expect(invokeCount()).toBe(after1);
    // Replaced measurement object → miss.
    const band2 = { ...band, measurement: makeMeasurement() as any };
    await evaluateBandFull({ band: band2 });
    expect(invokeCount()).toBeGreaterThan(after1);
  });

  it("smoothing change → recompute", async () => {
    const m = makeMeasurement() as any;
    const band = makeBand({ measurement: m });
    await evaluateBandFull({ band });
    const after1 = invokeCount();
    const band2 = makeBand({ measurement: m, settings: { ...band.settings, smoothing: "1/6" } as any });
    await evaluateBandFull({ band: band2 });
    expect(invokeCount()).toBeGreaterThan(after1);
  });

  it("different freq grid → recompute; same values → hit even for a new array", async () => {
    const band = makeBand();
    const grid = mockLogGrid(20, 20000, 128);
    await evaluateBandFull({ band, freq: grid });
    const after1 = invokeCount();
    // Equal-valued new array — key is content-hashed, not identity.
    await evaluateBandFull({ band, freq: [...grid] });
    expect(invokeCount()).toBe(after1);
    await evaluateBandFull({ band, freq: mockLogGrid(20, 20000, 256) });
    expect(invokeCount()).toBeGreaterThan(after1);
  });

  it("different sampleRate → recompute", async () => {
    const band = makeBand({ peqBands: [peq()] });
    await evaluateBandFull({ band, sampleRate: 48000 });
    const after1 = invokeCount();
    await evaluateBandFull({ band, sampleRate: 96000 });
    expect(invokeCount()).toBeGreaterThan(after1);
  });

  it("LRU eviction: oldest entry recomputes after cap overflow", async () => {
    const band = makeBand({ peqBands: [peq()] });
    await evaluateBandFull({ band, sampleRate: 1000 });
    // 32 more distinct keys → the sampleRate=1000 entry is evicted (cap 32).
    for (let i = 1; i <= 32; i++) {
      await evaluateBandFull({ band, sampleRate: 1000 + i });
    }
    const afterFill = invokeCount();
    await evaluateBandFull({ band, sampleRate: 1000 });
    expect(invokeCount()).toBeGreaterThan(afterFill);
  });
});

describe("evaluateSum cache (b141.7)", () => {
  it("identical repeat call → zero extra IPC", async () => {
    const bands = [makeBand(), makeBand({ id: "b2" })];
    const r1 = await evaluateSum(bands, { sampleRate: 48000 });
    const after1 = invokeCount();
    const r2 = await evaluateSum(bands, { sampleRate: 48000 });
    expect(invokeCount()).toBe(after1);
    expect(r2).not.toBe(r1);
    expect(r2.freq).toEqual(r1.freq);
  });

  // Note: inverted / alignmentDelay recomputes are pure JS at sum level
  // (sign flip / phase ramp; per-band IPC legitimately hits the band cache),
  // so the assertion is on differing OUTPUT, not on invoke count.
  it("band.inverted change → recompute (sum output differs)", async () => {
    const bands = [makeBand(), makeBand({ id: "b2" })];
    const r1 = await evaluateSum(bands, {});
    const bands2 = [makeBand(), makeBand({ id: "b2", inverted: true })];
    const r2 = await evaluateSum(bands2, {});
    expect(r2.sumTargetMag).not.toEqual(r1.sumTargetMag);
  });

  it("alignmentDelay change → recompute (sum output differs)", async () => {
    const bands = [makeBand(), makeBand({ id: "b2" })];
    const r1 = await evaluateSum(bands, {});
    const r2 = await evaluateSum(
      [makeBand(), makeBand({ id: "b2", alignmentDelay: 1.5 })], {},
    );
    expect(r2.sumTargetPhase).not.toEqual(r1.sumTargetPhase);
  });
});
