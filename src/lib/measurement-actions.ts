import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { createSignal } from "solid-js";
import {
  activeBand, setBandMeasurement, renameBand, setBandMeasurementFile,
  setBandDelayInfo, markBandDelayRemoved, setBandMergeSource,
} from "../stores/bands";
import { setNeedAutoFit } from "../App";
import { copyMeasurementToProject, copyMergeFilesToProject, projectDir } from "./project-io";
import type { Measurement } from "./types";
import type { MergeSource } from "../stores/bands";

export const [showMergeDialog, setShowMergeDialog] = createSignal(false);

export async function handleImportMeasurement() {
  const b = activeBand();
  if (!b) return;
  try {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Measurement Files", extensions: ["txt", "frd"] }],
    });
    if (!selected) return;
    const filePath = Array.isArray(selected) ? selected[0] : selected;
    const measurement = await invoke<Measurement>("import_measurement", { path: filePath });
    setBandMeasurement(b.id, measurement);
    const bandNum = b.name.match(/\d+/)?.[0] ?? "1";
    const newName = `Band ${bandNum} · ${measurement.name}`;
    renameBand(b.id, newName);
    try {
      const fileName = await copyMeasurementToProject(filePath, newName);
      if (fileName) setBandMeasurementFile(b.id, fileName);
    } catch (e) { console.warn("Failed to copy measurement:", e); }
    setNeedAutoFit(true);
    if (measurement.phase) {
      try {
        const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
          "remove_measurement_delay",
          { freq: measurement.freq, magnitude: measurement.magnitude, phase: measurement.phase, sampleRate: measurement.sample_rate }
        );
        setBandDelayInfo(b.id, delay, distance);
        markBandDelayRemoved(b.id, newPhase);
      } catch (e) { console.error("Delay removal failed:", e); }
    }
  } catch (e) { console.error("Import failed:", e); }
}

export async function handleMergeComplete(measurement: Measurement, source: MergeSource) {
  const b = activeBand();
  if (!b) return;
  setBandMeasurement(b.id, measurement);
  setBandMergeSource(b.id, source);
  const bandNum = b.name.match(/\d+/)?.[0] ?? "1";
  const newName = `Band ${bandNum} · ${measurement.name}`;
  renameBand(b.id, newName);
  try {
    const files = await copyMergeFilesToProject(source.nfPath, source.ffPath, newName);
    if (files) {
      setBandMeasurementFile(b.id, files.nfFile);
      setBandMergeSource(b.id, {
        ...source,
        nfPath: `${projectDir()}/${files.nfFile}`,
        ffPath: `${projectDir()}/${files.ffFile}`,
      });
    }
  } catch (e) { console.warn("Failed to copy merge files to project folder:", e); }
  setNeedAutoFit(true);
  if (measurement.phase) {
    try {
      const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
        "remove_measurement_delay",
        { freq: measurement.freq, magnitude: measurement.magnitude, phase: measurement.phase, sampleRate: measurement.sample_rate }
      );
      setBandDelayInfo(b.id, delay, distance);
      markBandDelayRemoved(b.id, newPhase);
    } catch (e) { console.error("Delay removal failed:", e); }
  }
}
