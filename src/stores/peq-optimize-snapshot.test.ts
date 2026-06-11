/**
 * b141.5 (audit MEDIUM): optimizeBand must snapshot store-proxy state
 * (peqBands, exclusionZones) BEFORE the first await. Reading the proxy
 * after an await mixes pre/post state when the user edits PEQ during a
 * long LMA run — frozen-band bake and exclusion zones silently pick up
 * the concurrent edit and the final setBandPeqBands loses it.
 * Project rule: "pre-read в plain objects ДО async loop".
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const lmaCalls: any[] = [];
const peqResponseCalls: any[] = [];
let onEvaluateTarget: (() => void) | null = null;
let onPeqResponse: (() => void) | null = null;

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "evaluate_target") {
      // Simulate a concurrent user edit happening while the IPC is in flight.
      onEvaluateTarget?.();
      const freq = args.freq as number[];
      return { magnitude: freq.map(() => 80), phase: freq.map(() => 0) };
    }
    if (cmd === "compute_peq_response") {
      peqResponseCalls.push(JSON.parse(JSON.stringify(args)));
      onPeqResponse?.();
      return (args.freq as number[]).map(() => 0);
    }
    if (cmd === "auto_peq_lma") {
      lmaCalls.push(JSON.parse(JSON.stringify(args)));
      return { bands: [], max_error_db: 0.1, iterations: 3 };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import {
  appState,
  addBand,
  setBandMeasurement,
  setBandPeqBands,
  updatePeqBand,
  addExclusionZone,
  removeExclusionZone,
} from "./bands";
import { handleOptimizePeq } from "./peq-optimize";
import type { PeqBand } from "../lib/types";

function flatMeasurement() {
  const n = 64;
  const freq: number[] = [];
  for (let i = 0; i < n; i++) freq.push(20 * Math.pow(20000 / 20, i / (n - 1)));
  return {
    name: "m", source_path: null, sample_rate: 48000,
    freq, magnitude: new Array(n).fill(80), phase: new Array(n).fill(0),
    metadata: { date: null, mic: null, notes: null, smoothing: null },
  };
}

// Factories, not consts: Solid's setState merges patches into the underlying
// object — sharing one literal across tests would leak mutations between them.
const frozenBand = (): PeqBand => ({ freq_hz: 100, gain_db: -4, q: 2, enabled: false, filter_type: "Peaking" });
const activePeq = (): PeqBand => ({ freq_hz: 1000, gain_db: -2, q: 1, enabled: true, filter_type: "Peaking" });

let bandId = "";
let bandIdx = 0;

beforeEach(() => {
  lmaCalls.length = 0;
  peqResponseCalls.length = 0;
  onEvaluateTarget = null;
  onPeqResponse = null;
  // addBand() makes the new band active — each test works on a fresh band.
  addBand();
  bandIdx = appState.bands.length - 1;
  bandId = appState.bands[bandIdx].id;
  setBandMeasurement(bandId, flatMeasurement() as any);
  setBandPeqBands(bandId, [frozenBand(), activePeq()]);
});

describe("optimizeBand snapshots store state before awaits (b141.5)", () => {
  it("frozen-band set is taken before the first await, not after", async () => {
    // While evaluate_target is in flight the user re-enables the frozen band.
    onEvaluateTarget = () => updatePeqBand(bandId, 0, { enabled: true });

    await handleOptimizePeq();

    // The bake must still see the pre-await frozen band (fc=100).
    expect(peqResponseCalls.length).toBe(1);
    expect(peqResponseCalls[0].bands.length).toBe(1);
    expect(peqResponseCalls[0].bands[0].freq_hz).toBe(100);
  });

  it("exclusion zones are taken before awaits, not after the bake IPC", async () => {
    addExclusionZone(bandId, { startHz: 200, endHz: 400 });
    // While the frozen-band bake IPC is in flight the user deletes the zone.
    onPeqResponse = () => {
      while (appState.bands[bandIdx].exclusionZones.length > 0) {
        removeExclusionZone(bandId, 0);
      }
    };

    await handleOptimizePeq();

    expect(lmaCalls.length).toBe(1);
    expect(lmaCalls[0].exclusionZones).not.toBeNull();
    expect(lmaCalls[0].exclusionZones.length).toBe(1);
    expect(lmaCalls[0].exclusionZones[0].startHz).toBe(200);
  });
});
