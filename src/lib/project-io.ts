import { invoke } from "@tauri-apps/api/core";
import { save, open, ask } from "@tauri-apps/plugin-dialog";
import { createSignal } from "solid-js";
import {
  appState,
  resetAppState,
  createBand,
  setActiveTab,
  activeTab,
  setActiveBandSum,
  exportSampleRate,
  setExportSampleRate,
  exportTaps,
  setExportTaps,
  exportWindow,
  setExportWindow,
  isDirty,
  setIsDirty,
  setBandMeasurementFile,
} from "../stores/bands";
import type { FilterConfig } from "../lib/types";
import type { AppState, BandState, PerMeasurementSettings, FloorBounceConfig, MergeSource } from "../stores/bands";
import type { Measurement, WindowType } from "../lib/types";

// ---------------------------------------------------------------------------
// Signals: project path, project directory, project name
// ---------------------------------------------------------------------------

export const [currentProjectPath, setCurrentProjectPath] = createSignal<string | null>(null);
export const [projectDir, setProjectDir] = createSignal<string | null>(null);
export const [projectName, setProjectName] = createSignal<string | null>(null);

// ---------------------------------------------------------------------------
// Promise-based project name prompt (rendered by ProjectNameDialog)
// ---------------------------------------------------------------------------

export interface ProjectPromptResult {
  name: string;
  bandCount: number;
}

let _promptResolve: ((val: ProjectPromptResult | null) => void) | null = null;
export const [promptVisible, setPromptVisible] = createSignal(false);

export function showProjectNamePrompt(): Promise<ProjectPromptResult | null> {
  setPromptVisible(true);
  return new Promise((resolve) => { _promptResolve = resolve; });
}

export function resolvePrompt(value: ProjectPromptResult | null) {
  setPromptVisible(false);
  _promptResolve?.(value);
  _promptResolve = null;
}

// ---------------------------------------------------------------------------
// Utility: filename helpers
// ---------------------------------------------------------------------------

/** YYMMDD date string */
export function yymmdd(): string {
  const d = new Date();
  const yy = String(d.getFullYear() % 100).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yy}${mm}${dd}`;
}

/** Remove special characters, keep [a-zA-Z0-9-] */
export function sanitize(name: string): string {
  return name
    .replace(/[^a-zA-Z0-9\-\s]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    || "untitled";
}

/** Generate measurement filename: ProjectName-YYMMDD-BandName.ext */
export function measurementFileName(projName: string, bandName: string, ext: string): string {
  return `${sanitize(projName)}-${yymmdd()}-${sanitize(bandName)}.${ext}`;
}

/** Generate WAV filename: ProjectName-YYMMDD-BandName-SR-Taps.wav */
export function wavFileName(projName: string, bandName: string, sr: number, taps: number): string {
  return `${sanitize(projName)}-${yymmdd()}-${sanitize(bandName)}-${sr}-${taps}.wav`;
}

/** Extract project dir and name from a .pfproj path */
function deriveProjectInfo(pfprojPath: string): { dir: string; name: string } {
  const parts = pfprojPath.split("/");
  const filename = parts.pop() ?? "project.pfproj";
  const dir = parts.join("/");
  const name = filename.replace(/\.pfproj$/, "");
  return { dir, name };
}

/** Get file extension from a path */
function fileExt(path: string): string {
  const dot = path.lastIndexOf(".");
  return dot >= 0 ? path.substring(dot + 1) : "txt";
}

// ---------------------------------------------------------------------------
// Build project data: AppState (camelCase) → ProjectFile (snake_case)
// ---------------------------------------------------------------------------

interface ProjectFile {
  version: number;
  app_name: string;
  project_name?: string | null;
  bands: ProjectBand[];
  active_band_id: string;
  show_phase: boolean;
  show_mag: boolean;
  show_target: boolean;
  next_band_num: number;
  export_sample_rate: number;
  export_taps: number;
  export_window: string;
  active_tab: string;
}

interface ProjectBand {
  id: string;
  name: string;
  measurement: any | null;
  measurement_file?: string | null;
  settings: ProjectSettings | null;
  target: any; // TargetCurve already snake_case
  target_enabled: boolean;
  inverted: boolean;
  linked_to_next: boolean;
  peq_bands: any[]; // PeqBand already snake_case
}

interface ProjectSettings {
  smoothing: string;
  delay_seconds: number | null;
  distance_meters: number | null;
  delay_removed: boolean;
  original_phase: number[] | null;
  floor_bounce: ProjectFloorBounce | null;
  merge_source: ProjectMergeSource | null;
}

interface ProjectFloorBounce {
  enabled: boolean;
  speaker_height: number;
  mic_height: number;
  distance: number;
}

interface ProjectMergeSource {
  nf_path: string;
  ff_path: string;
  config: any; // MergeConfig already snake_case
}

function mapSettingsToProject(s: PerMeasurementSettings): ProjectSettings {
  let fb: ProjectFloorBounce | null = null;
  if (s.floorBounce) {
    fb = {
      enabled: s.floorBounce.enabled,
      speaker_height: s.floorBounce.speakerHeight,
      mic_height: s.floorBounce.micHeight,
      distance: s.floorBounce.distance,
    };
  }
  let ms: ProjectMergeSource | null = null;
  if (s.mergeSource) {
    ms = {
      nf_path: s.mergeSource.nfPath,
      ff_path: s.mergeSource.ffPath,
      config: s.mergeSource.config,
    };
  }
  return {
    smoothing: s.smoothing,
    delay_seconds: s.delay_seconds,
    distance_meters: s.distance_meters,
    delay_removed: s.delay_removed,
    original_phase: s.originalPhase,
    floor_bounce: fb,
    merge_source: ms,
  };
}

function mapBandToProject(b: BandState): ProjectBand {
  const isV2 = projectDir() !== null;
  return {
    id: b.id,
    name: b.name,
    measurement: isV2 ? null : b.measurement, // v2: don't embed data
    measurement_file: isV2 ? b.measurementFile : null,
    settings: b.settings ? mapSettingsToProject(b.settings) : null,
    target: b.target, // already snake_case
    target_enabled: b.targetEnabled,
    inverted: b.inverted,
    linked_to_next: b.linkedToNext,
    peq_bands: b.peqBands, // PeqBand fields already snake_case
  };
}

function buildProjectData(): ProjectFile {
  const isV2 = projectDir() !== null;
  return {
    version: isV2 ? 2 : 1,
    app_name: "PhaseForge",
    project_name: isV2 ? projectName() : null,
    bands: appState.bands.map(mapBandToProject),
    active_band_id: appState.activeBandId,
    show_phase: appState.showPhase,
    show_mag: appState.showMag,
    show_target: appState.showTarget,
    next_band_num: appState.nextBandNum,
    export_sample_rate: exportSampleRate(),
    export_taps: exportTaps(),
    export_window: exportWindow(),
    active_tab: activeTab(),
  };
}

// ---------------------------------------------------------------------------
// Restore state: ProjectFile (snake_case) → AppState (camelCase)
// ---------------------------------------------------------------------------

function mapSettingsFromProject(s: ProjectSettings): PerMeasurementSettings {
  let fb: FloorBounceConfig | null = null;
  if (s.floor_bounce) {
    fb = {
      enabled: s.floor_bounce.enabled,
      speakerHeight: s.floor_bounce.speaker_height,
      micHeight: s.floor_bounce.mic_height,
      distance: s.floor_bounce.distance,
    };
  }
  let ms: MergeSource | null = null;
  if (s.merge_source) {
    ms = {
      nfPath: s.merge_source.nf_path,
      ffPath: s.merge_source.ff_path,
      config: s.merge_source.config,
    };
  }
  return {
    smoothing: s.smoothing as any,
    delay_seconds: s.delay_seconds,
    distance_meters: s.distance_meters,
    delay_removed: s.delay_removed,
    originalPhase: s.original_phase,
    floorBounce: fb,
    mergeSource: ms,
  };
}

function mapBandFromProject(b: ProjectBand): BandState {
  return {
    id: b.id,
    name: b.name,
    measurement: b.measurement,
    measurementFile: b.measurement_file ?? null,
    settings: b.settings ? mapSettingsFromProject(b.settings) : null,
    target: b.target,
    targetEnabled: b.target_enabled,
    inverted: b.inverted,
    linkedToNext: b.linked_to_next,
    peqBands: b.peq_bands,
    firResult: null, // FIR not saved — recomputed
  };
}

/** Restore state from a loaded project. For v2, re-import measurements from files. */
async function restoreState(project: ProjectFile, projDir: string | null) {
  const bands = project.bands.map(mapBandFromProject);

  // v2: re-import measurements from files in project folder
  if (project.version >= 2 && projDir) {
    for (const band of bands) {
      if (band.measurementFile) {
        const filePath = `${projDir}/${band.measurementFile}`;
        try {
          const m = await invoke<Measurement>("import_measurement", { path: filePath });
          band.measurement = m;
          // Re-apply delay compensation if it was enabled when project was saved
          if (band.settings?.delay_removed && m.phase) {
            try {
              const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
                "remove_measurement_delay",
                { freq: m.freq, phase: m.phase }
              );
              band.settings.originalPhase = [...m.phase];
              band.measurement.phase = newPhase;
              band.settings.delay_seconds = delay;
              band.settings.distance_meters = distance;
            } catch (_) {
              // If delay removal fails, mark as not removed
              band.settings.delay_removed = false;
            }
          }
        } catch (e) {
          console.warn(`Failed to re-import measurement ${band.measurementFile}:`, e);
        }
      }
    }
  }

  const newState: AppState = {
    bands,
    activeBandId: project.active_band_id,
    showPhase: project.show_phase,
    showMag: project.show_mag,
    showTarget: project.show_target,
    nextBandNum: project.next_band_num,
  };
  resetAppState(newState);

  // Restore signals
  setActiveTab(project.active_tab as any);
  setExportSampleRate(project.export_sample_rate);
  setExportTaps(project.export_taps);
  setExportWindow(project.export_window as WindowType);
  setIsDirty(false);
}

// ---------------------------------------------------------------------------
// Confirm unsaved changes
// ---------------------------------------------------------------------------

async function confirmIfDirty(): Promise<boolean> {
  if (!isDirty()) return true;
  const proceed = await ask(
    "You have unsaved changes. Discard them?",
    { title: "Unsaved Changes", kind: "warning", okLabel: "Discard", cancelLabel: "Cancel" },
  );
  return proceed;
}

// ---------------------------------------------------------------------------
// Copy pending measurements (bands with measurement data but no measurementFile)
// ---------------------------------------------------------------------------

async function copyPendingMeasurements(): Promise<void> {
  const dir = projectDir();
  const pName = projectName();
  if (!dir || !pName) return;

  for (const band of appState.bands) {
    if (band.measurement && !band.measurementFile) {
      const sourcePath = band.measurement.source_path;
      if (!sourcePath) continue;
      const ext = fileExt(sourcePath);
      const fileName = measurementFileName(pName, band.name, ext);
      const destPath = `${dir}/${fileName}`;
      try {
        await invoke("copy_file_to_project", { sourcePath, destPath });
        setBandMeasurementFile(band.id, fileName);
      } catch (e) {
        console.warn(`Failed to copy measurement for ${band.name}:`, e);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const FILTERS = [{ name: "PhaseForge Project", extensions: ["pfproj"] }];

/** Create a new project with folder.
 *  1. Modal dialog for project name
 *  2. Folder picker for parent location
 *  3. Create subfolder → save .pfproj inside */
export async function newProject(): Promise<void> {
  if (!(await confirmIfDirty())) return;

  // 1. Ask project name + band count via HTML modal dialog
  const result = await showProjectNamePrompt();
  if (!result) return; // cancelled
  const trimmedName = result.name;
  const bandCount = result.bandCount;

  // 2. Pick parent folder via native directory dialog
  const parentDir = await open({
    title: `Select location for "${trimmedName}"`,
    directory: true,
    multiple: false,
  });
  if (!parentDir) return; // cancelled
  const parentPath = Array.isArray(parentDir) ? parentDir[0] : parentDir;

  // 3. Create project subfolder: parentPath/trimmedName/
  let folderPath: string;
  try {
    folderPath = await invoke<string>("create_project_folder", {
      parentDir: parentPath,
      projectName: trimmedName,
    });
  } catch (e) {
    // If folder already exists, reuse it
    const existsPath = `${parentPath}/${trimmedName}`;
    const exists = await invoke<boolean>("check_path_exists", { path: existsPath });
    if (exists) {
      folderPath = existsPath;
    } else {
      console.error("Cannot create project folder:", e);
      return;
    }
  }

  // 4. Reset state — create N bands with default crossover filters
  //    Split 20–20000 Hz into equal logarithmic intervals
  const F_MIN = 20;
  const F_MAX = 20000;
  const logMin = Math.log10(F_MIN);
  const logMax = Math.log10(F_MAX);
  // crossover frequencies: bandCount-1 points dividing the range
  const crossoverFreqs: number[] = [];
  for (let i = 1; i < bandCount; i++) {
    const logF = logMin + (logMax - logMin) * (i / bandCount);
    crossoverFreqs.push(Math.round(Math.pow(10, logF)));
  }

  const defaultFilter = (freq: number): FilterConfig => ({
    filter_type: "LinkwitzRiley",
    order: 4,
    freq_hz: freq,
    shape: null,
    linear_phase: true,
  });

  const bands: BandState[] = [];
  for (let i = 1; i <= bandCount; i++) {
    const b = createBand(i);
    // HP: все кроме первой полосы
    if (i > 1) {
      b.target.high_pass = defaultFilter(crossoverFreqs[i - 2]);
    }
    // LP: все кроме последней полосы
    if (i < bandCount) {
      b.target.low_pass = defaultFilter(crossoverFreqs[i - 1]);
    }
    b.targetEnabled = true;
    // Все полосы кроме последней — linked к следующей
    if (i < bandCount) b.linkedToNext = true;
    bands.push(b);
  }

  const newState: AppState = {
    bands,
    activeBandId: "__sum__",  // открываем SUM view
    showPhase: true,
    showMag: true,
    showTarget: true,
    nextBandNum: bandCount + 1,
  };
  resetAppState(newState);
  setActiveTab("target");
  setExportSampleRate(48000);
  setExportTaps(65536);
  setExportWindow("Blackman");

  // 5. Set project signals — .pfproj goes INSIDE the subfolder
  setProjectDir(folderPath);
  setProjectName(trimmedName);
  const pfprojPath = `${folderPath}/${trimmedName}.pfproj`;
  setCurrentProjectPath(pfprojPath);
  setIsDirty(false);

  // 6. Auto-save initial empty project
  try {
    const project = buildProjectData();
    await invoke("save_project", { path: pfprojPath, project });
    await invoke("add_recent_project", { path: pfprojPath }).catch(() => {});
  } catch (e) {
    console.warn("Auto-save of new project failed:", e);
  }
}

/** Save to current path, or show Save As dialog if no path yet. */
export async function saveProject(): Promise<void> {
  const existing = currentProjectPath();
  if (existing) {
    // Copy any pending measurements before saving
    await copyPendingMeasurements();
    await doSave(existing);
    return;
  }
  await saveProjectAs();
}

/** Always show Save dialog. */
export async function saveProjectAs(): Promise<void> {
  const defName = projectName() ? `${projectName()}.pfproj` : "project.pfproj";
  const defPath = projectDir() ? `${projectDir()}/${defName}` : defName;
  const path = await save({
    title: "Save Project",
    filters: FILTERS,
    defaultPath: defPath,
  });
  if (!path) return; // cancelled

  // If saving to a new location, update projectDir/projectName
  const info = deriveProjectInfo(path);
  setProjectDir(info.dir);
  setProjectName(info.name);

  // Copy pending measurements to new location
  await copyPendingMeasurements();
  await doSave(path);
}

async function doSave(path: string): Promise<void> {
  const project = buildProjectData();
  await invoke("save_project", { path, project });
  setCurrentProjectPath(path);
  setIsDirty(false);
  // Add to recent projects
  await invoke("add_recent_project", { path }).catch(() => {});
}

/** Show Open dialog, load project. */
export async function loadProject(): Promise<void> {
  if (!(await confirmIfDirty())) return;

  const path = await open({
    title: "Open Project",
    filters: FILTERS,
    multiple: false,
    directory: false,
  });
  if (!path) return; // cancelled

  await doLoad(path as string);
}

/** Load a project from a known path (used by Recent Projects menu). */
export async function loadProjectFromPath(path: string): Promise<void> {
  if (!(await confirmIfDirty())) return;
  await doLoad(path);
}

async function doLoad(path: string): Promise<void> {
  const project = await invoke<ProjectFile>("load_project", { path });

  // Determine project dir and name
  const info = deriveProjectInfo(path);

  if (project.version >= 2) {
    // v2: project folder mode
    setProjectDir(info.dir);
    setProjectName(project.project_name ?? info.name);
  } else {
    // v1: legacy embedded mode
    setProjectDir(null);
    setProjectName(null);
  }

  await restoreState(project, project.version >= 2 ? info.dir : null);
  setCurrentProjectPath(path);
  // Add to recent projects
  await invoke("add_recent_project", { path }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Copy measurement file to project folder (used by import handlers)
// ---------------------------------------------------------------------------

/** Copy a measurement file into the project folder and return the new filename.
 *  Returns null if no project folder is set. */
export async function copyMeasurementToProject(
  sourcePath: string,
  bandName: string,
): Promise<string | null> {
  const dir = projectDir();
  const pName = projectName();
  if (!dir || !pName) return null;

  const ext = fileExt(sourcePath);
  const fileName = measurementFileName(pName, bandName, ext);
  const destPath = `${dir}/${fileName}`;
  await invoke("copy_file_to_project", { sourcePath, destPath });
  return fileName;
}

/** Copy NF/FF merge source files to project folder. Returns new filenames. */
export async function copyMergeFilesToProject(
  nfPath: string,
  ffPath: string,
  bandName: string,
): Promise<{ nfFile: string; ffFile: string } | null> {
  const dir = projectDir();
  const pName = projectName();
  if (!dir || !pName) return null;

  const nfExt = fileExt(nfPath);
  const ffExt = fileExt(ffPath);
  const date = yymmdd();
  const safeBand = sanitize(bandName);
  const safeProjName = sanitize(pName);

  const nfFile = `${safeProjName}-${date}-${safeBand}-NF.${nfExt}`;
  const ffFile = `${safeProjName}-${date}-${safeBand}-FF.${ffExt}`;

  await invoke("copy_file_to_project", { sourcePath: nfPath, destPath: `${dir}/${nfFile}` });
  await invoke("copy_file_to_project", { sourcePath: ffPath, destPath: `${dir}/${ffFile}` });

  return { nfFile, ffFile };
}
