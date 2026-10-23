/** b141.103 (audit 2026-10-23 3.6, 3.9, 3.11, 3.20): history keeps band
 *  content consistent across add/remove/redo, smoothing is undoable, project
 *  reset drops IR/GD/Export snapshots and parked bands. */
import { describe, it, expect } from "vitest";
import {
  appState, addBand, removeBand, setBandMeasurement, resetAppState, createBand,
  addPlotSnapshot, plotSnapshots, _parkedCount,
} from "../bands";
import { undo, redo, clearHistory } from "../history";
import { applyRecommendation } from "../../lib/analysis-actions";

const meas = () => ({
  name: "m", source_path: null, sample_rate: 48000, freq: [20, 200, 2000, 20000],
  magnitude: [80, 81, 82, 83], phase: [0, -10, -20, -30],
  metadata: { date: null, mic: null, notes: null, smoothing: null },
} as any);

describe("undo/redo of «Add band» (3.6)", () => {
  it("redo brings the band back with its measurement", () => {
    clearHistory();
    addBand();
    const id = appState.bands[appState.bands.length - 1].id;
    setBandMeasurement(id, meas());
    undo();
    expect(appState.bands.find((b) => b.id === id)).toBeUndefined();
    redo();
    const back = appState.bands.find((b) => b.id === id);
    expect(back?.measurement?.magnitude).toEqual([80, 81, 82, 83]);
    expect(back?.settings).not.toBeNull();
  });
});

describe("undo of the «Сглаживание» recommendation (3.9)", () => {
  it("restores the previous smoothing in one step", () => {
    clearHistory();
    addBand();
    const id = appState.bands[appState.bands.length - 1].id;
    setBandMeasurement(id, meas());
    clearHistory();
    applyRecommendation(id, { action: { type: "ApplySmoothing", value: "1/3" } } as any, "f1");
    expect(appState.bands.find((b) => b.id === id)!.settings!.smoothing).toBe("1/3");
    undo();
    expect(appState.bands.find((b) => b.id === id)!.settings!.smoothing).toBe("off");
  });
});

describe("parked heavy parts (3.20)", () => {
  it("are dropped once no history entry can restore the band", () => {
    resetAppState({ bands: [createBand(1), createBand(2), createBand(3)], activeBandId: "__sum__",
      showPhase: true, showMag: true, showTarget: true, nextBandNum: 4 });
    clearHistory();
    removeBand(appState.bands[2].id);
    expect(_parkedCount()).toBe(1);
    clearHistory();
    removeBand(appState.bands[1].id);
    expect(_parkedCount()).toBe(1);
  });
});

describe("project reset (3.11)", () => {
  it("clears IR/GD/Export snapshots", () => {
    const id = appState.bands[0].id;
    addPlotSnapshot(id, { label: "s", color: "#fff", tab: "ir", timeMs: [0], impulse: [1] });
    expect(plotSnapshots(id, "ir").length).toBe(1);
    resetAppState({ bands: [createBand(1)], activeBandId: "__sum__",
      showPhase: true, showMag: true, showTarget: true, nextBandNum: 2 });
    expect(plotSnapshots(id, "ir").length).toBe(0);
  });
});
