import { describe, it, expect } from "vitest";
import { peqRingCost, peqRingVerdict, earRingMs, peqNetVerdict, netLengthening, type NetDecay } from "../peq-quality";

// b141.93: per-filter time cost shown in the PEQ table.
const pk = (freq_hz: number, gain_db: number, q: number) => ({ freq_hz, gain_db, q, filter_type: "Peaking" as const });

describe("peqRingCost", () => {
  it("time scales with Q / f: the wall project's 22 Hz boost", () => {
    const q4 = peqRingCost(pk(22, 3.1, 4), 176400);
    const q2 = peqRingCost(pk(22, 3.1, 2), 176400);
    expect(q4.ms60).toBeGreaterThan(380);
    expect(q4.ms60).toBeLessThan(460);
    expect(q4.ms60 / q2.ms60).toBeCloseTo(2, 1);
    expect(peqRingCost(pk(100, 3.1, 4), 176400).ms60).toBeLessThan(q4.ms60 / 4);
    expect(q4.ms100).toBeGreaterThan(q4.ms60);
  });
  it("does not depend on the export rate below Nyquist", () => {
    const a = peqRingCost(pk(22, 3.1, 4), 48000).ms60, b = peqRingCost(pk(22, 3.1, 4), 352800).ms60;
    expect(Math.abs(a - b) / b).toBeLessThan(0.01);
  });
  it("a shelf barely rings; a zero-gain filter costs nothing", () => {
    expect(peqRingCost({ freq_hz: 25, gain_db: 3, q: 0.7, filter_type: "LowShelf" }, 176400).ms60).toBeLessThan(80);
    expect(peqRingCost(pk(22, 0, 4), 176400)).toEqual({ ms60: 0, ms100: 0 });
  });
});

// b141.138: ring vs the ear's own ring (scratchpad ring_scale.py, 176.4 kHz).
describe("peqRingVerdict", () => {
  const on = (freq_hz: number, gain_db: number, q: number) => ({ ...pk(freq_hz, gain_db, q), enabled: true });
  const half = 371.5;
  it("the ear rings longer in periods at HF", () => {
    expect(earRingMs(30)).toBeCloseTo(38.6, 0);
    expect(earRingMs(1000)).toBeCloseTo(8.13, 1);
    expect(earRingMs(10000) * 10).toBeCloseTo(9.8, 0);
  });
  it("the same Q passes at 10 kHz, turns the number red at 1 kHz, the row at 100 Hz", () => {
    expect(peqRingVerdict(on(10000, 6, 2.5), 176400, half).level).toBe("");
    const k1 = peqRingVerdict(on(1000, 6, 3), 176400, half);
    expect(k1.earRatio).toBeCloseTo(1.15, 1);
    expect(k1.level).toBe("strict");
    expect(peqRingVerdict(on(100, 6, 3), 176400, half).level).toBe("tolerant");
  });
  it("wall band 1: 42.7 Hz +2.7 dB Q 1.1 is strict, 718 Hz +2.3 dB Q 0.5 is fine", () => {
    expect(peqRingVerdict(on(42.70, 2.68, 1.1), 176400, half).level).toBe("strict");
    expect(peqRingVerdict(on(718.4, 2.26, 0.5), 176400, half).level).toBe("");
  });
  it("a tail longer than half the file is tolerant whatever the ear says; disabled or 0 dB never flags", () => {
    expect(peqRingVerdict(on(1000, 6, 3), 176400, 5).level).toBe("tolerant");
    expect(peqRingVerdict({ ...on(30, 6, 5), enabled: false }, 176400, half).level).toBe("");
    expect(peqRingVerdict(on(30, 0, 5), 176400, half).level).toBe("");
  });
  it("a Linkwitz transform is judged at fp", () => {
    const lt = { freq_hz: 40, gain_db: 12, q: 0.7, filter_type: "LinkwitzTransform" as const, enabled: true };
    const v = peqRingVerdict(lt, 176400, half);
    expect(v.earRatio).toBeCloseTo(v.ms60 / earRingMs(40 / Math.pow(10, 12 / 40)), 6);
  });
});

describe("peqNetVerdict (b141.139)", () => {
  const pk = (freq_hz: number, gain_db: number, q: number, enabled = true) =>
    ({ freq_hz, gain_db, q, filter_type: "Peaking" as const, enabled });
  const nd = (ear: [number, number], without: [number, number], with_: [number, number]): NetDecay =>
    ({ freqHz: 50, earMs: ear, withoutMs: without, withMs: with_ });

  it("the worst level over the ear's decay at that level", () => {
    // wall PK 54.7 Hz −2.3 dB Q 2: −2.4 ms at −20 dB, +6.2 ms at −30 dB.
    expect(netLengthening(nd([20.7, 29.6], [57.0, 110.2], [54.6, 116.4]))).toBeCloseTo(0.21, 2);
  });

  it("two stages on the lengthening, not on the filter's own ring", () => {
    const f = pk(42.7, 2.7, 1.1); // alone ×1.5 the ear: strict under b141.138
    expect(peqRingVerdict(f, 176400, 371.5).level).toBe("strict");
    expect(peqNetVerdict(f, nd([25.6, 36], [71.4, 251.1], [58.9, 247]), 176400, 371.5).level).toBe("");
    expect(peqNetVerdict(f, nd([10, 10], [10, 10], [13, 12]), 176400, 371.5).level).toBe("strict");
    expect(peqNetVerdict(f, nd([10, 10], [10, 10], [21, 12]), 176400, 371.5).level).toBe("tolerant");
  });

  it("no phase: the filter-alone law; past the measurement's edge: half-file only", () => {
    const f = pk(100, 6, 3);
    expect(peqNetVerdict(f, undefined, 176400, 371.5).level).toBe(peqRingVerdict(f, 176400, 371.5).level);
    expect(peqNetVerdict(f, null, 176400, 371.5).level).toBe("");
    expect(peqNetVerdict(f, null, 176400, 1).level).toBe("tolerant");
    expect(peqNetVerdict(pk(100, 6, 3, false), nd([1, 1], [1, 1], [9, 9]), 176400, 371.5).level).toBe("");
  });
});

