import { describe, it, expect } from "vitest";
import { peqRingCost } from "../peq-quality";

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
