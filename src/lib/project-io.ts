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
  exportHybridPhase,
  setExportHybridPhase,
  isDirty,
  setIsDirty,
  setBandMeasurementFile,
  firIterations, setFirIterations,
  firFreqWeighting, setFirFreqWeighting,
  firNarrowbandLimit, setFirNarrowbandLimit,
  firNbSmoothingOct, setFirNbSmoothingOct,
  firNbMaxExcess, setFirNbMaxExcess,
  firMaxBoost, setFirMaxBoost,
  firNoiseFloor, setFirNoiseFloor,
} from "../stores/bands";
import type { FilterConfig } from "../lib/types";
import { tolerance, setTolerance, maxBands, setMaxBands } from "../stores/peq-optimize";
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
/** "new" = New Project (name + band count), "saveAs" = Save As (name only) */
export const [promptMode, setPromptMode] = createSignal<"new" | "saveAs">("new");

export function showProjectNamePrompt(): Promise<ProjectPromptResult | null> {
  setPromptMode("new");
  setPromptVisible(true);
  return new Promise((resolve) => { _promptResolve = resolve; });
}

/** Show prompt for Save As — only asks for project name, no band count. */
export function showSaveAsPrompt(): Promise<string | null> {
  setPromptMode("saveAs");
  setPromptVisible(true);
  return new Promise((resolve) => {
    _promptResolve = (val) => resolve(val?.name ?? null);
  });
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

/** Remove filesystem-unsafe characters, keep Unicode letters */
export function sanitize(name: string): string {
  return name
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .replace(/\.+$/g, "")
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
  export_hybrid_phase?: boolean;
  peq_tolerance?: number;
  peq_max_bands?: number;
  fir_iterations?: number;
  fir_freq_weighting?: boolean;
  fir_narrowband_limit?: boolean;
  fir_nb_smoothing_oct?: number;
  fir_nb_max_excess_db?: number;
  fir_max_boost_db?: number;
  fir_noise_floor_db?: number;
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
    export_hybrid_phase: exportHybridPhase(),
    peq_tolerance: tolerance(),
    peq_max_bands: maxBands(),
    fir_iterations: firIterations(),
    fir_freq_weighting: firFreqWeighting(),
    fir_narrowband_limit: firNarrowbandLimit(),
    fir_nb_smoothing_oct: firNbSmoothingOct(),
    fir_nb_max_excess_db: firNbMaxExcess(),
    fir_max_boost_db: firMaxBoost(),
    fir_noise_floor_db: firNoiseFloor(),
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
    crossNormDb: 0,
  };
}

/** Restore state from a loaded project. For v2, re-import measurements from files. */
async function restoreState(project: ProjectFile, projDir: string | null) {
  const bands = project.bands.map(mapBandFromProject);

  // v2: re-import measurements from files in project folder
  if (project.version >= 2 && projDir) {
    for (const band of bands) {
      if (band.measurementFile) {
        // Try the stored path first; if not found, try inbox/ prefix or root fallback
        let filePath = `${projDir}/${band.measurementFile}`;
        const exists = await invoke<boolean>("check_path_exists", { path: filePath }).catch(() => false);
        if (!exists) {
          // Old project without inbox/: file might be in root
          const baseName = band.measurementFile.split("/").pop() ?? band.measurementFile;
          const rootPath = `${projDir}/${baseName}`;
          const rootExists = await invoke<boolean>("check_path_exists", { path: rootPath }).catch(() => false);
          if (rootExists) {
            filePath = rootPath;
          }
          // Also try inbox/ in case measurementFile stored without prefix
          if (!rootExists) {
            const inboxPath = `${projDir}/inbox/${baseName}`;
            const inboxExists = await invoke<boolean>("check_path_exists", { path: inboxPath }).catch(() => false);
            if (inboxExists) filePath = inboxPath;
          }
        }
        try {
          const m = await invoke<Measurement>("import_measurement", { path: filePath });
          band.measurement = m;
          // Re-apply delay compensation if it was enabled when project was saved
          if (band.settings?.delay_removed && m.phase) {
            try {
              const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
                "remove_measurement_delay",
                { freq: m.freq, magnitude: m.magnitude, phase: m.phase }
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
    showPhase: true,   // always visible (b82.06)
    showMag: true,     // always visible (b82.06)
    showTarget: true,  // always visible (b82.06)
    nextBandNum: project.next_band_num,
  };
  resetAppState(newState);

  // Restore signals
  const savedTab = project.active_tab === "align" ? "target" : project.active_tab;
  setActiveTab(savedTab as any);
  setExportSampleRate(project.export_sample_rate);
  setExportTaps(project.export_taps);
  setExportWindow(project.export_window as WindowType);
  setExportHybridPhase(project.export_hybrid_phase ?? false);
  setTolerance(project.peq_tolerance ?? 1.0);
  setMaxBands(project.peq_max_bands ?? 20);
  // FIR optimization settings
  setFirIterations(project.fir_iterations ?? 3);
  setFirFreqWeighting(project.fir_freq_weighting ?? true);
  setFirNarrowbandLimit(project.fir_narrowband_limit ?? true);
  setFirNbSmoothingOct(project.fir_nb_smoothing_oct ?? 0.333);
  setFirNbMaxExcess(project.fir_nb_max_excess_db ?? 6.0);
  setFirMaxBoost(project.fir_max_boost_db ?? 24.0);
  setFirNoiseFloor(project.fir_noise_floor_db ?? -150.0);
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
  if (!dir) return;

  // Ensure inbox/ exists (backward compat with old projects)
  await invoke("ensure_dir", { path: `${dir}/inbox` }).catch(() => {});

  for (const band of appState.bands) {
    if (band.measurement && !band.measurementFile) {
      const sourcePath = band.measurement.source_path;
      if (!sourcePath) continue;
      // Use original filename (basename) — no renaming
      const fileName = sourcePath.split("/").pop() ?? sourcePath.split("\\").pop() ?? "measurement.txt";
      const destPath = `${dir}/inbox/${fileName}`;
      try {
        // Skip if already in project folder
        const srcNorm = sourcePath.replace(/\\/g, "/");
        const destNorm = destPath.replace(/\\/g, "/");
        if (srcNorm !== destNorm) {
          await invoke("copy_file_to_project", { sourcePath, destPath });
        }
        setBandMeasurementFile(band.id, `inbox/${fileName}`);
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

/** Save As: ask name → pick parent folder → create project folder tree → copy files → save. */
export async function saveProjectAs(): Promise<void> {
  // 1. Ask for new project name
  const newName = await showSaveAsPrompt();
  if (!newName) return; // cancelled

  // 2. Pick parent folder
  const parentDir = await open({
    title: `Select location for "${newName}"`,
    directory: true,
    multiple: false,
  });
  if (!parentDir) return; // cancelled
  const parentPath = Array.isArray(parentDir) ? parentDir[0] : parentDir;

  // 3. Create project folder: parentPath/newName/ + inbox/ + target/ + export/
  let newDir: string;
  try {
    newDir = await invoke<string>("create_project_folder", {
      parentDir: parentPath,
      projectName: newName,
    });
  } catch (_e) {
    // Folder may already exist — reuse it
    const existsPath = `${parentPath}/${newName}`;
    const exists = await invoke<boolean>("check_path_exists", { path: existsPath });
    if (exists) {
      newDir = existsPath;
      // Ensure subdirectories exist
      for (const sub of ["inbox", "target", "export"]) {
        await invoke("ensure_dir", { path: `${newDir}/${sub}` }).catch(() => {});
      }
    } else {
      console.error("Cannot create project folder:", _e);
      return;
    }
  }

  // 4. Copy files from old project folder (if exists)
  const oldDir = projectDir();
  console.log("[SaveAs] oldDir:", oldDir, "newDir:", newDir);
  if (oldDir && oldDir !== newDir) {
    for (const sub of ["inbox", "target", "export"]) {
      try {
        const copied = await invoke<number>("copy_dir_contents", {
          sourceDir: `${oldDir}/${sub}`,
          destDir: `${newDir}/${sub}`,
        });
        if (copied > 0) console.log(`[SaveAs] Copied ${copied} files: ${sub}/`);
      } catch (e) {
        console.warn(`[SaveAs] Failed to copy ${sub}/:`, e);
      }
    }
  } else if (!oldDir) {
    console.warn("[SaveAs] No previous projectDir — skipping file copy");
  }

  // 5. Update project signals
  setProjectDir(newDir);
  setProjectName(newName);

  // 6. Copy any pending measurements (bands with data but no measurementFile)
  await copyPendingMeasurements();

  // 7. Save .pfproj inside the new folder
  const pfprojPath = `${newDir}/${newName}.pfproj`;
  await doSave(pfprojPath);
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
  _bandName: string,
): Promise<string | null> {
  const dir = projectDir();
  if (!dir) return null;

  // Ensure inbox/ exists (backward compat)
  await invoke("ensure_dir", { path: `${dir}/inbox` }).catch(() => {});

  // Use original filename — no renaming, copy to inbox/
  const fileName = sourcePath.split("/").pop() ?? sourcePath.split("\\").pop() ?? "measurement.txt";
  const destPath = `${dir}/inbox/${fileName}`;
  // Skip if already in project folder
  const srcNorm = sourcePath.replace(/\\/g, "/");
  const destNorm = destPath.replace(/\\/g, "/");
  if (srcNorm !== destNorm) {
    await invoke("copy_file_to_project", { sourcePath, destPath });
  }
  return `inbox/${fileName}`;
}

/** Copy NF/FF merge source files to project folder. Returns new filenames. */
export async function copyMergeFilesToProject(
  nfPath: string,
  ffPath: string,
  _bandName: string,
): Promise<{ nfFile: string; ffFile: string } | null> {
  const dir = projectDir();
  if (!dir) return null;

  // Ensure inbox/ exists (backward compat)
  await invoke("ensure_dir", { path: `${dir}/inbox` }).catch(() => {});

  // Use original filenames — no renaming, copy to inbox/
  const nfBaseName = nfPath.split("/").pop() ?? nfPath.split("\\").pop() ?? "nf.txt";
  const ffBaseName = ffPath.split("/").pop() ?? ffPath.split("\\").pop() ?? "ff.txt";

  const nfFile = `inbox/${nfBaseName}`;
  const ffFile = `inbox/${ffBaseName}`;

  await invoke("copy_file_to_project", { sourcePath: nfPath, destPath: `${dir}/${nfFile}` });
  await invoke("copy_file_to_project", { sourcePath: ffPath, destPath: `${dir}/${ffFile}` });

  return { nfFile, ffFile };
}
