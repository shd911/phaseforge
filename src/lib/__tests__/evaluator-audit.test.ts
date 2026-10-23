/** b141.103 (audit 2026-10-23 3.8, 3.15, 3.23, 3.24): evaluator details. */
import { describe, it, expect, vi, beforeEach } from "vitest";

const calls: { cmd: string; args: any }[] = [];
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    calls.push({ cmd, args });
    if (cmd === "get_smoothed") return (args.magnitude as number[]).map((v) => v - 10);
    if (cmd === "compute_impulse") return { time: [0, 1], impulse: [0, 0], step: [0, 0], dt: 1, t0: 0 };
    if (cmd === "compute_peq_complex") {
      const n = (args.freq as number[]).length;
      return [new Array(n).fill(0), new Array(n).fill(0)];
    }
    if (cmd === "compute_target_hilbert_phase") return (args.freq as number[]).map(() => 0);
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { createBand } from "../../stores/bands";
import { evaluateBandFull, composeCorrected, sameGridValues } from "../band-evaluator/evaluate";
import { computeExtension } from "../band-evaluator/extension";

beforeEach(() => { calls.length = 0; });

function meas() {
  const freq = [20, 100, 1000, 10000, 20000];
  return {
    name: "m", source_path: null, sample_rate: 48000, freq,
    magnitude: [80, 81, 82, 83, 84], phase: [0, -10, -20, -30, -40],
    metadata: { date: null, mic: null, notes: null, smoothing: null },
  } as any;
}

describe("IR ignores the display smoothing (3.8)", () => {
  it("SPL is smoothed, the IR gets the raw measurement", async () => {
    const band = createBand(1);
    band.measurement = meas();
    band.settings = { smoothing: "1/3", delay_seconds: null, distance_meters: null, delay_removed: false,
      originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false };
    band.targetEnabled = false;
    const r = await evaluateBandFull({ band, includeIr: true });
    expect(r.measurementMag).toEqual([70, 71, 72, 73, 74]);
    const ir = calls.find((c) => c.cmd === "compute_impulse")!;
    expect(ir.args.magnitude).toEqual([80, 81, 82, 83, 84]);
  });
});

describe("corrected phase with the target off (3.15)", () => {
  it("adds no Gaussian Hilbert term", async () => {
    const band = createBand(1);
    band.targetEnabled = false;
    band.target.high_pass = { filter_type: "Gaussian", order: 4, freq_hz: 80, shape: 1, linear_phase: false, q: null };
    const freq = [20, 100, 1000];
    const c = await composeCorrected(freq, [0, 0, 0], [1, 2, 3], band, 48000);
    expect(c.phase).toEqual([1, 2, 3]);
    expect(calls.some((x) => x.cmd === "compute_target_hilbert_phase")).toBe(false);
  });
});

describe("sameGridValues (3.23)", () => {
  it("compares every bin, not length and endpoints", () => {
    expect(sameGridValues([20, 100, 20000], [20, 100, 20000])).toBe(true);
    expect(sameGridValues([20, 100, 20000], [20, 200, 20000])).toBe(false);
    expect(sameGridValues([20, 20000], [20, 100, 20000])).toBe(false);
  });
});

describe("computeExtension inside the native range (3.24)", () => {
  it("interpolates magnitude in log f", async () => {
    const f = Math.sqrt(100 * 1000);
    const r = await computeExtension([100, 1000], [0, 10], null, [100, f, 1000], [0, 0, 0]);
    expect(r.mag[1]).toBeCloseTo(5, 9);
  });
});
