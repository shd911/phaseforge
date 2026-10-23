/** b141.103 (audit 2026-10-23 3.1, 3.2, 3.13): undo restores band content
 *  only; global settings mark the project dirty and survive undo; a save
 *  keeps the dirty flag when an edit landed during the write. */
import { describe, it, expect, vi } from "vitest";

let onSave: (() => void) | null = null;
vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string) => {
    if (cmd === "save_project") { onSave?.(); return null; }
    return null;
  }),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ save: vi.fn(), open: vi.fn() }));
vi.mock("../../components/MeasurementAnalysisDialog", () => ({ openMeasurementAnalysis: vi.fn() }));

import {
  appState, addBand, setBandTilt, isDirty, setIsDirty, markDirty,
  exportSampleRate, setExportSampleRate, exportTaps, setExportTaps,
  exportHybridPhase, setExportHybridPhase, firMaxBoost, setFirMaxBoost,
} from "../bands";
import { peqBoostFloor, setPeqBoostFloor, peqSmoothFit, setPeqSmoothFit, peqFloor, setPeqFloor } from "../peq-optimize";
import { undo, redo, clearHistory } from "../history";
import { saveProject, setCurrentProjectPath, setProjectDir } from "../../lib/project-io";

describe("undo scope (3.1)", () => {
  it("slope → SR → undo: slope back, SR stays", async () => {
    clearHistory();
    addBand();
    const id = appState.bands[appState.bands.length - 1].id;
    await new Promise((r) => setTimeout(r, 300)); // past the coalesce window
    setBandTilt(id, -1.5);
    setExportSampleRate(96000);
    setExportTaps(262144);
    setPeqBoostFloor(12);
    setPeqSmoothFit(false);
    setExportHybridPhase(true);
    setFirMaxBoost(12);
    undo();
    const b = appState.bands.find((x) => x.id === id)!;
    expect(b.target.tilt_db_per_octave).toBe(0);
    expect(exportSampleRate()).toBe(96000);
    expect(exportTaps()).toBe(262144);
    expect(peqBoostFloor()).toBe(12);
    expect(peqSmoothFit()).toBe(false);
    expect(exportHybridPhase()).toBe(true);
    expect(firMaxBoost()).toBe(12);
    redo();
    expect(appState.bands.find((x) => x.id === id)!.target.tilt_db_per_octave).toBe(-1.5);
    expect(exportSampleRate()).toBe(96000);
  });
});

describe("settings mark the project dirty (3.2)", () => {
  it("each setter flags a change, an unchanged value does not", () => {
    setIsDirty(false);
    setExportSampleRate(exportSampleRate());
    expect(isDirty()).toBe(false);
    setPeqFloor(peqFloor() + 1);
    expect(isDirty()).toBe(true);
    setIsDirty(false);
    setExportTaps(exportTaps() === 65536 ? 131072 : 65536);
    expect(isDirty()).toBe(true);
  });
});

describe("save during an edit (3.13)", () => {
  it("keeps the dirty flag when a change landed while writing", async () => {
    setCurrentProjectPath("/tmp/x/p.pfproj");
    setProjectDir("/tmp/x");
    markDirty();
    onSave = () => markDirty();
    await saveProject();
    expect(isDirty()).toBe(true);
    onSave = null;
    await saveProject();
    expect(isDirty()).toBe(false);
  });
});
