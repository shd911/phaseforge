import { describe, it, expect } from "vitest";
import { estimateSealedF0, hp2Db, linkwitzDb, linkwitzTargetHz, measurementWithTransforms, linkwitzDefaults } from "../linkwitz";
import { peqRingCost } from "../peq-quality";
import type { PeqBand } from "../types";

// Node's fs through a variable (tsc has no node types here; Vite must not bundle it).
const FS_MODULE = "node:fs";
const fs: { existsSync(u: URL): boolean; readFileSync(u: URL, enc: "utf8"): string } =
  await import(/* @vite-ignore */ FS_MODULE);
const HERE = import.meta.url;

const logGrid = (n: number, a: number, b: number) => Array.from({ length: n }, (_, i) => a * Math.pow(b / a, i / (n - 1)));
const lt = (f0: number, q0: number, gain: number): PeqBand => ({ freq_hz: f0, gain_db: gain, q: q0, enabled: true, filter_type: "LinkwitzTransform" });

describe("Linkwitz transform (b141.107)", () => {
  it("box × transform = BU2 at fp", () => {
    const b = lt(25.3, 0.707, 6);
    const fp = linkwitzTargetHz(b);
    expect(fp).toBeCloseTo(25.3 / Math.pow(10, 6 / 40), 6);
    for (const f of logGrid(50, 5, 200)) {
      expect(hp2Db(f, 25.3, 0.707) + linkwitzDb(f, b)).toBeCloseTo(hp2Db(f, fp, Math.SQRT1_2), 6);
    }
  });

  it("estimates f0 at the rising −3 dB point, ignoring a room dip above it", () => {
    const freq = logGrid(400, 15, 300);
    const mag = freq.map((f) => 84 + hp2Db(f, 28, 0.707) + (f > 40 && f < 48 ? -4 : 0));
    const f0 = estimateSealedF0(freq, mag, 130)!;
    expect(f0).toBeGreaterThan(25);
    expect(f0).toBeLessThan(31);
  });

  // b141.129: the search stopped at 60 Hz, a higher corner fell back to 30 Hz.
  it("finds a corner above 60 Hz, gives up past 200 Hz", () => {
    const freq = logGrid(400, 15, 3000);
    const f0 = estimateSealedF0(freq, freq.map((f) => 84 + hp2Db(f, 80, 0.707)), 1000)!;
    expect(f0).toBeGreaterThan(70);
    expect(f0).toBeLessThan(90);
    expect(estimateSealedF0(freq, freq.map((f) => 84 + hp2Db(f, 500, 0.707)), 2000)).toBeNull();
  });

  it("the optimizer sees the model below f0 and the target alignment with the transform", () => {
    const freq = logGrid(300, 20, 300);
    // passband 84 dB; a sweep-edge cliff below 24 Hz instead of the box roll-off
    const mag = freq.map((f) => 84 + hp2Db(f, 25, 0.707) - (f < 24 ? 40 * Math.log2(24 / f) : 0));
    const t = lt(25, 0.707, 6);
    const seen = measurementWithTransforms(freq, mag, [t], 130);
    const fp = linkwitzTargetHz(t);
    freq.forEach((f, i) => { if (f < 25) expect(seen[i]).toBeCloseTo(84 + hp2Db(f, fp, Math.SQRT1_2), 1); });
  });

  it("rings like its target, shorter than an edge boost", () => {
    const r = peqRingCost(lt(25.3, 0.707, 6), 176400).ms60;
    const edge = peqRingCost({ freq_hz: 24, gain_db: 4.5, q: 1.92, filter_type: "Peaking" }, 176400).ms60;
    expect(r).toBeGreaterThan(50);
    expect(r).toBeLessThan(0.6 * edge);
  });

  it("defaults on the real wall 10x4 sub land at its −3 dB point", () => {
    const p = new URL("../../../test-fixtures/prering/sub_10x4.txt", HERE);
    if (!fs.existsSync(p)) return;
    const rows = fs.readFileSync(p, "utf8").split("\n").filter((l: string) => /^\d/.test(l))
      .map((l: string) => l.trim().split(/\s+/).map(Number));
    const d = linkwitzDefaults(rows.map((r: number[]) => r[0]), rows.map((r: number[]) => r[1]), 130);
    expect(d.freq_hz).toBeGreaterThan(23);
    expect(d.freq_hz).toBeLessThan(28);
    expect(d.gain_db).toBe(6);
  });
});
