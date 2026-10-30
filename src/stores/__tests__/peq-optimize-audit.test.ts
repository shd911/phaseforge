/** b141.103 (audit 2026-10-23 3.4, 3.10, 3.18, 3.19): optimizer edge cases. */
import { describe, it, expect, vi, beforeEach } from "vitest";

const lmaCalls: any[] = [];
let onLma: (() => void) | null = null;

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      // A band with a low-pass has its target 100 dB down above 690 Hz.
      const ref = args.target.reference_level_db as number;
      const lp = args.target.low_pass;
      return {
        magnitude: (args.freq as number[]).map((f) => (lp && f > 690 ? ref - 100 : ref)),
        phase: (args.freq as number[]).map(() => 0),
      };
    }
    if (cmd === "auto_peq_lma") {
      lmaCalls.push(args);
      onLma?.();
      return { bands: [{ freq_hz: 2000, gain_db: -2, q: 1, enabled: true, filter_type: "Peaking" }], max_error_db: 0.5, iterations: 2 };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import {
  appState, addBand, setBandMeasurement, setBandPeqBands, setBandLowPass, resetAppState, createBand,
} from "../bands";
import {
  handleOptimizeAll, handleOptimizePeq, setPeqRangeMode, setPeqDirectLow, setPeqDirectHigh, setPeqFloor,
  setMaxBands, captureOptimizedTarget, peqStaleReason, peqRangeFor,
} from "../peq-optimize";
import { setBandPeqOptimizedTarget, exportSampleRate, setExportSampleRate } from "../bands";
import { undo, clearHistory } from "../history";

function meas(level = 80) {
  const n = 64;
  const freq: number[] = [];
  for (let i = 0; i < n; i++) freq.push(20 * Math.pow(20000 / 20, i / (n - 1)));
  return {
    name: "m", source_path: "/a/m.txt", sample_rate: 48000,
    freq, magnitude: new Array(n).fill(level), phase: new Array(n).fill(0),
    metadata: { date: null, mic: null, notes: null, smoothing: null },
  } as any;
}

beforeEach(() => {
  lmaCalls.length = 0;
  onLma = null;
  resetAppState({ bands: [createBand(1), createBand(2)], activeBandId: "__sum__",
    showPhase: true, showMag: true, showTarget: true, nextBandNum: 3 });
  clearHistory();
  setPeqRangeMode("auto");
  setPeqFloor(60);
  setMaxBands(20);
});

describe("«Оптимизировать все» with one empty range (3.4)", () => {
  it("applies the other bands and names the failing one", async () => {
    const [a, b] = appState.bands.map((x) => x.id);
    setBandMeasurement(a, meas());
    setBandMeasurement(b, meas());
    setBandLowPass(b, { filter_type: "LinkwitzRiley", order: 4, freq_hz: 500, shape: null, linear_phase: false, q: null });
    setPeqRangeMode("direct");
    setPeqDirectLow(1500);
    setPeqDirectHigh(3000);
    await handleOptimizeAll();
    expect(lmaCalls.length).toBe(1);
    expect(appState.bands[0].peqBands.length).toBe(1);
    expect(appState.bands[1].peqBands.length).toBe(0);
  });
});

// b141.129 (external audit 2026-10-29)
describe("a direct range wholly above the optimizer's cap", () => {
  it("is named instead of failing in Rust", async () => {
    const was = exportSampleRate();
    setExportSampleRate(44100);
    const id = appState.bands[0].id;
    setBandMeasurement(id, meas());
    setPeqRangeMode("direct");
    setPeqDirectLow(21000);
    setPeqDirectHigh(25000);
    await expect(peqRangeFor(appState.bands[0])).rejects.toThrow(/21000–25000 Hz пуст/);
    setExportSampleRate(was);
  });
});

describe("undo during a run (3.10)", () => {
  it("drops the result instead of landing it on the restored state", async () => {
    const a = appState.bands[0].id;
    setBandMeasurement(a, meas());
    setBandPeqBands(a, []);
    appState.activeBandId; // active band below
    resetActive(a);
    onLma = () => undo();
    await handleOptimizePeq();
    expect(lmaCalls.length).toBe(1);
    expect(appState.bands[0].peqBands.length).toBe(0);
  });
});

describe("disabled filters fill the budget (3.19)", () => {
  it("does not add maxBands + 1", async () => {
    const a = appState.bands[0].id;
    setBandMeasurement(a, meas());
    setMaxBands(2);
    setBandPeqBands(a, [
      { freq_hz: 100, gain_db: -1, q: 1, enabled: false, filter_type: "Peaking" },
      { freq_hz: 200, gain_db: -1, q: 1, enabled: false, filter_type: "Peaking" },
    ]);
    resetActive(a);
    await handleOptimizePeq();
    expect(lmaCalls.length).toBe(0);
    expect(appState.bands[0].peqBands.length).toBe(2);
  });
});

describe("new measurement (3.18)", () => {
  it("marks the PEQ stale and clears the old project file", () => {
    const a = appState.bands[0].id;
    setBandMeasurement(a, meas(80));
    setBandPeqBands(a, [{ freq_hz: 1000, gain_db: -2, q: 1, enabled: true, filter_type: "Peaking" }]);
    setBandPeqOptimizedTarget(a, captureOptimizedTarget(appState.bands[0]));
    expect(peqStaleReason(appState.bands[0])).toBeNull();
    // Same content in a new object (project reload) — still fresh.
    setBandMeasurement(a, meas(80));
    expect(peqStaleReason(appState.bands[0])).toBeNull();
    setBandMeasurement(a, meas(75));
    expect(peqStaleReason(appState.bands[0])).toBe("замер заменён");
    expect(appState.bands[0].measurementFile).toBeNull();
  });
});

import { setActiveBand } from "../bands";
function resetActive(id: string) { setActiveBand(id); }
