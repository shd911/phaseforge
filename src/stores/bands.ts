import { createStore, reconcile } from "solid-js/store";
import { createSignal } from "solid-js";
import type { Measurement, TargetCurve, MergeConfig, PeqBand, FirResult, WindowType } from "../lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SmoothingMode = "off" | "1/3" | "1/6" | "1/12" | "1/24" | "var";

export interface FloorBounceConfig {
  enabled: boolean;
  speakerHeight: number; // метры
  micHeight: number;     // метры
  distance: number;      // метры
}

export interface MergeSource {
  nfPath: string;
  ffPath: string;
  config: MergeConfig;
}

export interface PerMeasurementSettings {
  smoothing: SmoothingMode;
  delay_seconds: number | null;
  distance_meters: number | null;
  delay_removed: boolean;
  originalPhase: number[] | null; // оригинальная фаза до компенсации задержки
  floorBounce: FloorBounceConfig | null;
  mergeSource: MergeSource | null; // сохранённые параметры merge для интерактивного re-merge
}

export interface BandState {
  id: string;
  name: string;
  measurement: Measurement | null;
  measurementFile: string | null; // v2: relative filename in project folder
  settings: PerMeasurementSettings | null;
  target: TargetCurve;
  targetEnabled: boolean;
  inverted: boolean; // инвертирование полярности (фаза +180°)
  linkedToNext: boolean; // связь LP этой полосы ↔ HP следующей
  peqBands: PeqBand[]; // авто-подобранные PEQ полосы
  firResult: FirResult | null; // результат генерации FIR
  crossNormDb: number; // normalization estimate from cross-section peak (dB)
}

export interface AppState {
  bands: BandState[];
  activeBandId: string; // band id or "__sum__"
  showPhase: boolean;
  showMag: boolean; // отображение АЧХ замера
  showTarget: boolean; // отображение таргет-кривой
  nextBandNum: number;
}

export type PresetName = "flat" | "harman" | "bk" | "x-curve" | "custom";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SUM_ID = "__sum__";

function defaultTarget(): TargetCurve {
  return {
    reference_level_db: 0,
    tilt_db_per_octave: 0,
    tilt_ref_freq: 1000,
    high_pass: null,
    low_pass: null,
    low_shelf: null,
    high_shelf: null,
  };
}

function defaultSettings(): PerMeasurementSettings {
  return { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null };
}

export function createBand(num: number): BandState {
  return {
    id: crypto.randomUUID(),
    name: `Band ${num}`,
    measurement: null,
    measurementFile: null,
    settings: null,
    target: defaultTarget(),
    targetEnabled: false,
    inverted: false,
    linkedToNext: false,
    peqBands: [],
    firResult: null,
    crossNormDb: 0,
  };
}

// Флаг для предотвращения бесконечного цикла при пропагации linked freq
let _propagating = false;

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

const PRESETS: Record<PresetName, () => TargetCurve> = {
  flat: defaultTarget,
  harman: () => ({
    reference_level_db: 0,
    tilt_db_per_octave: -0.4,
    tilt_ref_freq: 1000,
    high_pass: null,
    low_pass: null,
    low_shelf: { freq_hz: 200, gain_db: 4, q: 0.7 },
    high_shelf: { freq_hz: 8000, gain_db: -2, q: 0.7 },
  }),
  bk: () => ({
    reference_level_db: 0,
    tilt_db_per_octave: -1.0,
    tilt_ref_freq: 1000,
    high_pass: null,
    low_pass: null,
    low_shelf: null,
    high_shelf: null,
  }),
  "x-curve": () => ({
    reference_level_db: 0,
    tilt_db_per_octave: 0,
    tilt_ref_freq: 1000,
    high_pass: null,
    low_pass: null,
    low_shelf: null,
    high_shelf: { freq_hz: 2000, gain_db: -6, q: 0.5 },
  }),
  custom: defaultTarget,
};

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const firstBand = createBand(1);

const [state, setState] = createStore<AppState>({
  bands: [firstBand],
  activeBandId: firstBand.id,
  showPhase: true,
  showMag: true,
  showTarget: true,
  nextBandNum: 2,
});

// ---------------------------------------------------------------------------
// Derived
// ---------------------------------------------------------------------------

function bandIndex(id: string): number {
  return state.bands.findIndex((b) => b.id === id);
}

export function resetAppState(newState: AppState) {
  setState(reconcile(newState));
}

export function activeBand(): BandState | null {
  if (state.activeBandId === SUM_ID) return null;
  return state.bands.find((b) => b.id === state.activeBandId) ?? null;
}

export function isSum(): boolean {
  return state.activeBandId === SUM_ID;
}

// ---------------------------------------------------------------------------
// Band management
// ---------------------------------------------------------------------------

export function addBand() {
  const num = state.nextBandNum;
  const band = createBand(num);
  setState("bands", (prev) => [...prev, band]);
  setState("nextBandNum", num + 1);
  setState("activeBandId", band.id);
  markDirty();
}

export function removeBand(id: string) {
  if (state.bands.length <= 1) return; // нельзя удалить последнюю полосу
  const idx = bandIndex(id);
  if (idx < 0) return;
  // Сбрасываем linkedToNext у предыдущей полосы (если была связана с удаляемой)
  if (idx > 0 && state.bands[idx - 1].linkedToNext) {
    setState("bands", idx - 1, "linkedToNext", false);
  }
  setState("bands", (prev) => prev.filter((b) => b.id !== id));
  // Если удалили активную полосу — переключаемся на первую
  if (state.activeBandId === id) {
    setState("activeBandId", state.bands[0]?.id ?? SUM_ID);
  }
  markDirty();
}

export function setActiveBand(id: string) {
  setState("activeBandId", id);
}

export function setActiveBandSum() {
  setState("activeBandId", SUM_ID);
}

export function renameBand(id: string, name: string) {
  const idx = bandIndex(id);
  if (idx < 0) return;
  setState("bands", idx, "name", name);
  markDirty();
}

/** Переместить полосу из позиции fromIdx в toIdx. Сбрасывает все linkedToNext. */
export function moveBand(fromIdx: number, toIdx: number) {
  if (fromIdx === toIdx) return;
  if (fromIdx < 0 || fromIdx >= state.bands.length) return;
  if (toIdx < 0 || toIdx >= state.bands.length) return;

  const bands = [...state.bands];
  const [moved] = bands.splice(fromIdx, 1);
  bands.splice(toIdx, 0, moved);

  // Сбрасываем все linked-связи — после перестановки порядок поменялся
  for (const b of bands) {
    b.linkedToNext = false;
  }

  setState("bands", bands);
  markDirty();
}

// ---------------------------------------------------------------------------
// Глобальные toggle-и отображения
// ---------------------------------------------------------------------------

export function togglePhase() {
  setState("showPhase", !state.showPhase);
  markDirty();
}

export function toggleMag() {
  setState("showMag", !state.showMag);
  markDirty();
}

export function toggleTarget() {
  setState("showTarget", !state.showTarget);
  markDirty();
}

// ---------------------------------------------------------------------------
// Per-band: measurement
// ---------------------------------------------------------------------------

export function setBandMeasurement(bandId: string, m: Measurement) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "measurement", m);
  setState("bands", idx, "settings", defaultSettings());
  markDirty();
}

/** Заменить measurement без сброса settings (для re-merge и подобных обновлений) */
export function replaceBandMeasurement(bandId: string, m: Measurement) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "measurement", m);
  markDirty();
}

export function clearBandMeasurement(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "measurement", null);
  setState("bands", idx, "measurementFile", null);
  setState("bands", idx, "settings", null);
  markDirty();
}

export function setBandMeasurementFile(bandId: string, filename: string | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "measurementFile", filename);
  markDirty();
}

export function setBandSmoothing(bandId: string, mode: SmoothingMode) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "smoothing", mode);
  markDirty();
}

export function setBandDelayInfo(bandId: string, delay: number, distance: number) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "delay_seconds", delay);
  setState("bands", idx, "settings", "distance_meters", distance);
  markDirty();
}

/** Сбросить сохранённую оригинальную фазу (чтобы следующий markBandDelayRemoved сохранил свежую) */
export function resetBandOriginalPhase(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "originalPhase", null);
}

// Применить компенсацию задержки (сохраняя оригинал для возможности отмены)
export function markBandDelayRemoved(bandId: string, newPhase: number[]) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings || !state.bands[idx].measurement) return;
  // Сохраняем оригинальную фазу перед заменой
  const origPhase = state.bands[idx].measurement!.phase;
  if (origPhase && !state.bands[idx].settings!.originalPhase) {
    setState("bands", idx, "settings", "originalPhase", [...origPhase]);
  }
  setState("bands", idx, "measurement", "phase", newPhase);
  setState("bands", idx, "settings", "delay_removed", true);
  markDirty();
}

// Восстановить оригинальную фазу (отключить компенсацию)
export function restoreBandDelay(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings || !state.bands[idx].measurement) return;
  const orig = state.bands[idx].settings!.originalPhase;
  if (orig) {
    setState("bands", idx, "measurement", "phase", [...orig]);
  }
  setState("bands", idx, "settings", "delay_removed", false);
  markDirty();
}

export function updateBandPhase(bandId: string, newPhase: number[]) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].measurement) return;
  setState("bands", idx, "measurement", "phase", newPhase);
  markDirty();
}

// ---------------------------------------------------------------------------
// Per-band: target / filters
// ---------------------------------------------------------------------------

export function toggleBandTarget(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "targetEnabled", !state.bands[idx].targetEnabled);
  markDirty();
}

// Автовключение таргета при включении любого фильтра
export function ensureTargetEnabled(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  if (!state.bands[idx].targetEnabled) {
    setState("bands", idx, "targetEnabled", true);
  }
}

// Инвертирование полярности полосы (фаза +180°)
export function toggleBandInverted(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "inverted", !state.bands[idx].inverted);
  markDirty();
}

// Связь LP текущей полосы ↔ HP следующей
export function toggleBandLinked(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0 || idx >= state.bands.length - 1) return; // нельзя link последнюю полосу
  const newVal = !state.bands[idx].linkedToNext;
  setState("bands", idx, "linkedToNext", newVal);
  // При включении связи — синхронизируем LP → HP следующей (freq + type + order + linear_phase)
  if (newVal) {
    const lp = state.bands[idx].target.low_pass;
    const nextIdx = idx + 1;
    if (lp && state.bands[nextIdx].target.high_pass) {
      const nextHp = state.bands[nextIdx].target.high_pass!;
      setState("bands", nextIdx, "target", "high_pass", {
        ...nextHp,
        freq_hz: lp.freq_hz,
        filter_type: lp.filter_type,
        order: lp.order,
        shape: lp.shape,
        linear_phase: lp.linear_phase,
      });
    }
  }
  markDirty();
}

// Проверяет, связана ли полоса с предыдущей (т.е. предыдущая имеет linkedToNext=true)
export function isBandLinkedFromPrev(bandId: string): boolean {
  const idx = bandIndex(bandId);
  if (idx <= 0) return false;
  return state.bands[idx - 1].linkedToNext;
}

export function setBandPreset(bandId: string, preset: PresetName) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "target", PRESETS[preset]());
  markDirty();
}

export function setBandReferenceLevel(bandId: string, db: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "target", "reference_level_db", db);
  markDirty();
}

export function setBandTilt(bandId: string, dbPerOctave: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "target", "tilt_db_per_octave", dbPerOctave);
  markDirty();
}

export function setBandHighPass(bandId: string, config: import("../lib/types").FilterConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  // Валидация: HP freq не может быть выше LP freq
  if (config) {
    const lpFreq = state.bands[idx].target.low_pass?.freq_hz;
    if (lpFreq != null && config.freq_hz > lpFreq) {
      config = { ...config, freq_hz: lpFreq };
    }
  }
  setState("bands", idx, "target", "high_pass", config);
  // Автовключение таргета при включении фильтра
  if (config && !state.bands[idx].targetEnabled) {
    setState("bands", idx, "targetEnabled", true);
  }
  // Пропагация linked: HP → LP предыдущей полосы (freq + type + order + linear_phase)
  if (!_propagating && config && idx > 0 && state.bands[idx - 1].linkedToNext) {
    const prevLp = state.bands[idx - 1].target.low_pass;
    if (prevLp) {
      _propagating = true;
      setState("bands", idx - 1, "target", "low_pass", {
        ...prevLp,
        freq_hz: config.freq_hz,
        filter_type: config.filter_type,
        order: config.order,
        shape: config.shape,
        linear_phase: config.linear_phase,
      });
      _propagating = false;
    }
  }
  markDirty();
}

export function setBandLowPass(bandId: string, config: import("../lib/types").FilterConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  // Валидация: LP freq не может быть ниже HP freq
  if (config) {
    const hpFreq = state.bands[idx].target.high_pass?.freq_hz;
    if (hpFreq != null && config.freq_hz < hpFreq) {
      config = { ...config, freq_hz: hpFreq };
    }
  }
  setState("bands", idx, "target", "low_pass", config);
  if (config && !state.bands[idx].targetEnabled) {
    setState("bands", idx, "targetEnabled", true);
  }
  // Пропагация linked: LP → HP следующей полосы (freq + type + order + linear_phase)
  if (!_propagating && config && state.bands[idx].linkedToNext && idx < state.bands.length - 1) {
    const nextHp = state.bands[idx + 1].target.high_pass;
    if (nextHp) {
      _propagating = true;
      setState("bands", idx + 1, "target", "high_pass", {
        ...nextHp,
        freq_hz: config.freq_hz,
        filter_type: config.filter_type,
        order: config.order,
        shape: config.shape,
        linear_phase: config.linear_phase,
      });
      _propagating = false;
    }
  }
  markDirty();
}

export function setBandLowShelf(bandId: string, config: import("../lib/types").ShelfConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "target", "low_shelf", config);
  if (config && !state.bands[idx].targetEnabled) {
    setState("bands", idx, "targetEnabled", true);
  }
  markDirty();
}

export function setBandHighShelf(bandId: string, config: import("../lib/types").ShelfConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "target", "high_shelf", config);
  if (config && !state.bands[idx].targetEnabled) {
    setState("bands", idx, "targetEnabled", true);
  }
  markDirty();
}

// ---------------------------------------------------------------------------
// Per-band: merge source (для интерактивного re-merge)
// ---------------------------------------------------------------------------

export function setBandMergeSource(bandId: string, source: MergeSource) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "mergeSource", source);
  markDirty();
}

export function updateBandSpliceFreq(bandId: string, freq: number) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings?.mergeSource) return;
  setState("bands", idx, "settings", "mergeSource", "config", "splice_freq", freq);
  markDirty();
}

// ---------------------------------------------------------------------------
// Per-band: floor bounce
// ---------------------------------------------------------------------------

export function setBandFloorBounce(bandId: string, config: FloorBounceConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "floorBounce", config);
  markDirty();
}

export function toggleBandFloorBounce(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  const fb = state.bands[idx].settings!.floorBounce;
  if (fb) {
    setState("bands", idx, "settings", "floorBounce", "enabled", !fb.enabled);
  } else {
    // Включаем с дефолтными значениями; distance берём из distance_meters (вычисленной при отмотке фазы)
    const measDist = state.bands[idx].settings!.distance_meters;
    setState("bands", idx, "settings", "floorBounce", {
      enabled: true,
      speakerHeight: 1.0,
      micHeight: 0.6,
      distance: measDist != null && measDist > 0 ? Math.round(measDist * 100) / 100 : 2.0,
    });
  }
  markDirty();
}

export function updateBandFloorBounceField(
  bandId: string,
  field: "speakerHeight" | "micHeight" | "distance",
  value: number,
) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings?.floorBounce) return;
  setState("bands", idx, "settings", "floorBounce", field, value);
  markDirty();
}

// ---------------------------------------------------------------------------
// Per-band: PEQ auto-align
// ---------------------------------------------------------------------------

export function setBandPeqBands(bandId: string, bands: PeqBand[]) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "peqBands", bands);
  markDirty();
}

export function removePeqBand(bandId: string, peqIdx: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "peqBands", (prev) => prev.filter((_, i) => i !== peqIdx));
  markDirty();
}

export function clearBandPeqBands(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "peqBands", []);
  markDirty();
}

export function addPeqBand(bandId: string, band: PeqBand) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  // Добавляем в начало — новый фильтр всегда первая строка таблицы
  setState("bands", idx, "peqBands", (prev) => [band, ...prev]);
  markDirty();
}

export function updatePeqBand(bandId: string, peqIdx: number, patch: Partial<PeqBand>) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  const peq = state.bands[idx].peqBands[peqIdx];
  if (!peq) return;
  const updated = { ...peq, ...patch };
  setState("bands", idx, "peqBands", peqIdx, updated);
  markDirty();
}

/** Sort PEQ band into its correct position by frequency. Returns new index. */
export function commitPeqBand(bandId: string, peqIdx: number): number {
  const idx = bandIndex(bandId);
  if (idx < 0) return peqIdx;
  const arr = [...state.bands[idx].peqBands];
  const band = arr[peqIdx];
  if (!band) return peqIdx;
  arr.sort((a, b) => a.freq_hz - b.freq_hz);
  setState("bands", idx, "peqBands", arr);
  markDirty();
  return arr.indexOf(band);
}

// ---------------------------------------------------------------------------
// Per-band: FIR result
// ---------------------------------------------------------------------------

export function setBandFirResult(bandId: string, result: FirResult | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "firResult", result);
  markDirty();
}

export function setBandCrossNormDb(bandId: string, val: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "crossNormDb", val);
}

// ---------------------------------------------------------------------------
// Active tab (shared between ControlPanel ↔ App.tsx bottom panel)
// ---------------------------------------------------------------------------

export type ActiveTab = "measurements" | "target" | "align" | "export";
export const [activeTab, setActiveTab] = createSignal<ActiveTab>("measurements");

// ---------------------------------------------------------------------------
// Selected PEQ band index (for highlighting on graphs)
// ---------------------------------------------------------------------------

export const [selectedPeqIdx, setSelectedPeqIdx] = createSignal<number | null>(null);

// ---------------------------------------------------------------------------
// One-shot command: FrequencyPlot shows only these categories, then resets to null
// ---------------------------------------------------------------------------

export const [plotShowOnly, setPlotShowOnly] =
  createSignal<("measurement" | "target" | "corrected")[] | null>(null);

// ---------------------------------------------------------------------------
// Shared X-scale (sync frequency axis between top FrequencyPlot and bottom PeqResponsePlot)
// ---------------------------------------------------------------------------

export interface XScale { min: number; max: number }
const [_xScale, _setXScale] = createSignal<XScale>({ min: 20, max: 20000 });
let _xScaleSuppressed = false; // prevent feedback loops

export const sharedXScale = _xScale;

/** Called by either plot when its X scale changes (zoom/pan). */
export function setSharedXScale(s: XScale) {
  if (_xScaleSuppressed) return;
  _setXScale(s);
}

/** Run `fn` without triggering shared scale updates (used inside setScale handlers). */
export function suppressXScaleSync(fn: () => void) {
  _xScaleSuppressed = true;
  fn();
  _xScaleSuppressed = false;
}

// ---------------------------------------------------------------------------
// Export FIR config signals (used by ExportTab + ExportPlot)
// ---------------------------------------------------------------------------

export const [exportSampleRate, setExportSampleRate] = createSignal(48000);
export const [exportTaps, setExportTaps] = createSignal(65536);
export const [exportWindow, setExportWindow] = createSignal<WindowType>("Blackman");

// ---------------------------------------------------------------------------
// Dirty state: true when project has unsaved changes
// ---------------------------------------------------------------------------

export const [isDirty, setIsDirty] = createSignal(false);
export function markDirty() { if (!isDirty()) setIsDirty(true); }

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export { state as appState, SUM_ID };
