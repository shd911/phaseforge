// b141.101 (audit 2026-10-23 1.3): the tilt's minimum-phase term reaches the
// plot exactly where the FIR realises the tilt min-phase — a min-phase main.
import { describe, it, expect, vi, beforeEach } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { reconstructTargetPhase } from "../band-evaluator/evaluate";
import type { FilterConfig } from "../types";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) =>
    cmd === "compute_target_hilbert_phase" ? (args.freq as number[]).map(() => -14.5) : undefined),
}));

const lr4 = (fc: number, linear_phase: boolean): FilterConfig =>
  ({ filter_type: "LinkwitzRiley", order: 4, freq_hz: fc, shape: null, linear_phase, q: null } as FilterConfig);

describe("reconstructTargetPhase — tilt term", () => {
  beforeEach(() => vi.mocked(invoke).mockClear());
  const freq = [100, 1000, 10000];
  const base = [0, 0, 0];

  it("asks Rust for the tilt term on a min-phase main", async () => {
    const t = { high_pass: lr4(200, false), low_pass: lr4(5000, false), tilt_db_per_octave: -1, tilt_ref_freq: 1000 };
    expect(await reconstructTargetPhase(freq, base, t, 48000)).toEqual([-14.5, -14.5, -14.5]);
    expect(vi.mocked(invoke)).toHaveBeenCalledWith("compute_target_hilbert_phase",
      expect.objectContaining({ tiltDbPerOctave: -1, tiltRefFreq: 1000 }));
  });

  it("keeps the tilt zero-phase on a linear or mixed main, and without crossovers", async () => {
    for (const t of [
      { high_pass: lr4(200, true), low_pass: lr4(5000, true), tilt_db_per_octave: -1 },
      { high_pass: lr4(200, false), low_pass: lr4(5000, true), tilt_db_per_octave: -1 },
      { high_pass: null, low_pass: null, tilt_db_per_octave: -1 },
      { high_pass: lr4(200, false), low_pass: null, tilt_db_per_octave: 0 },
    ]) {
      expect(await reconstructTargetPhase(freq, base, t, 48000)).toEqual(base);
    }
    expect(vi.mocked(invoke)).not.toHaveBeenCalledWith("compute_target_hilbert_phase", expect.anything());
  });
});
