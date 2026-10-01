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
// No Node typings in the repo (tsc treats src/ as browser code) — load fs
// through a non-literal specifier, as export-metrics.test.ts does.
const FS_MODULE = "node:fs";
const fs: { readFileSync(u: URL, enc: string): string; writeFileSync(u: URL, d: string): void } =
  await import(/* @vite-ignore */ FS_MODULE);
const HERE = import.meta.url;
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
    // Band ids are random per run — normalise so the fixture only changes
    // when the schema does.
    project.bands.forEach((pb: any, i: number) => {
      if (project.active_band_id === pb.id) project.active_band_id = `band${i}`;
      pb.id = `band${i}`;
    });
    const json = JSON.stringify(project, null, 1);
    const path = new URL("../../../src-tauri/tests/fixtures/ts_project_contract.json", HERE);
    let prev = "";
    try { prev = fs.readFileSync(path, "utf8"); } catch { /* first run */ }
    if (prev !== json) fs.writeFileSync(path, json);
    expect(project.bands.length).toBeGreaterThan(0);
  });
});
