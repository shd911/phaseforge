/**
 * b141.71 (audit stage 2 A7): the project schema is written twice — TS
 * interfaces (project-io.ts) and Rust serde structs (project.rs) — and serde
 * drops unknown fields without a word (the b141.18 delay-convention marker
 * was lost that way). This test builds a project with every DSP-relevant field
 * set and writes it to src-tauri/tests/fixtures/ts_project_contract.json; the
 * Rust test `project_contract` round-trips it through ProjectFile and fails
 * on any field that does not survive. Re-run cargo test after this changes.
 */
import { describe, it, expect } from "vitest";
import { writeFileSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  appState, addBand, setBandMeasurement, setBandPeqBands, setBandPeqOptimizedTarget,
  setBandTilt, setAlignmentDelay, addExclusionZone, setBandHighPass, setBandLowPass,
} from "../../stores/bands";
import { captureOptimizedTarget } from "../../stores/peq-optimize";
import { buildProjectData } from "../project-io";

describe("project schema contract (TS → Rust)", () => {
  it("writes a fully populated project for the Rust round-trip test", () => {
    addBand();
    const b = appState.bands[appState.bands.length - 1];
    const freq = [20, 200, 2000, 20000];
    setBandMeasurement(b.id, {
      name: "m", source_path: null, sample_rate: 48000, freq,
      magnitude: [80, 81, 82, 83], phase: [0, -10, -20, -30],
      metadata: { date: null, mic: null, notes: null, smoothing: null },
    } as any);
    setBandHighPass(b.id, { filter_type: "Gaussian", order: 4, freq_hz: 80, shape: 1.2, linear_phase: false, q: null, subsonic_protect: true } as any);
    setBandLowPass(b.id, { filter_type: "Custom", order: 2, freq_hz: 2500, shape: null, linear_phase: true, q: 0.6 } as any);
    setBandTilt(b.id, -0.5);
    setBandPeqBands(b.id, [
      { freq_hz: 40, gain_db: -3, q: 4, enabled: true, filter_type: "Peaking" },
      { freq_hz: 9000, gain_db: 1.5, q: 0.8, enabled: false, filter_type: "HighShelf" },
    ]);
    addExclusionZone(b.id, { startHz: 300, endHz: 500 });
    setAlignmentDelay(b.id, 0.00123);
    setBandPeqOptimizedTarget(b.id, captureOptimizedTarget(appState.bands.find((x) => x.id === b.id)!));

    const project = buildProjectData();
    const json = JSON.stringify(project, null, 1);
    const path = resolve(__dirname, "../../../src-tauri/tests/fixtures/ts_project_contract.json");
    let prev = "";
    try { prev = readFileSync(path, "utf8"); } catch { /* first run */ }
    if (prev !== json) writeFileSync(path, json);
    expect(project.bands.length).toBeGreaterThan(0);
  });
});
