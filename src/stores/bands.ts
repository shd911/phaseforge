import { createStore, reconcile } from "solid-js/store";
import { createSignal, batch } from "solid-js";
import type { Measurement, TargetCurve, MergeConfig, PeqBand, FirResult, WindowType, ExclusionZone } from "../lib/types";
import { MEASUREMENT_COLORS } from "../lib/types";

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
  exclusionZones: ExclusionZone[]; // частотные зоны исключённые из оптимизации
  firResult: FirResult | null; // результат генерации FIR
  crossNormDb: number; // normalization estimate from cross-section peak (dB)
  color: string; // user-assigned curve color
  alignmentDelay: number; // seconds, visual-only SUM phase alignment delay
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

/** Compute equal-octave crossover frequencies for n bands spanning fMin–fMax.
 *  Returns n-1 crossover points. E.g. 3 bands → 2 crossovers. */
function equalOctaveCrossovers(n: number, fMin = 20, fMax = 20000): number[] {
  if (n <= 1) return [];
  const logMin = Math.log2(fMin);
  const logMax = Math.log2(fMax);
  const step = (logMax - logMin) / n;
  const xovers: number[] = [];
  for (let i = 1; i < n; i++) {
    xovers.push(Math.round(Math.pow(2, logMin + step * i)));
  }
  return xovers;
}

const DEFAULT_FILTER = { filter_type: "LinkwitzRiley" as const, order: 4, shape: null, linear_phase: false, q: null };

/** Assign default HP/LP filters to all bands based on equal-octave split.
 *  Only assigns to bands that have NO measurement (pristine bands). */
function assignDefaultTargets(bands: BandState[]) {
  const n = bands.length;
  if (n <= 0) return;
  const xovers = equalOctaveCrossovers(n);
  for (let i = 0; i < n; i++) {
    // Skip bands that already have a measurement — user configured them
    if (bands[i].measurement) continue;
    const hp = i > 0 ? { ...DEFAULT_FILTER, freq_hz: xovers[i - 1] } : null;
    const lp = i < n - 1 ? { ...DEFAULT_FILTER, freq_hz: xovers[i] } : null;
    // Force fresh store nodes by setting null first (prevents shared-node bug)
    setState("bands", i, "target", "high_pass", null);
    setState("bands", i, "target", "high_pass", hp);
    setState("bands", i, "target", "low_pass", null);
    setState("bands", i, "target", "low_pass", lp);
    setState("bands", i, "targetEnabled", true);
    // Link to next band
    if (i < n - 1) {
      setState("bands", i, "linkedToNext", true);
    }
  }
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
    exclusionZones: [],
    firResult: null,
    crossNormDb: 0,
    color: MEASUREMENT_COLORS[(num - 1) % MEASUREMENT_COLORS.length],
    alignmentDelay: 0,
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
    low_shelf: null,
    high_shelf: null,
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
    high_shelf: null,
  }),
  custom: defaultTarget,
};

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const firstBand = createBand(1);
firstBand.targetEnabled = true; // show target by default

const [state, setState] = createStore<AppState>({
  bands: [firstBand],
  activeBandId: SUM_ID,
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
  // Deep clone filter configs to prevent shared references in SolidJS store
  for (const b of newState.bands) {
    if (b.target.high_pass) b.target.high_pass = { ...b.target.high_pass };
    if (b.target.low_pass) b.target.low_pass = { ...b.target.low_pass };
  }
  setState(reconcile(newState));
  clearAllExportSnapshots();
  clearAllFreqSnapshots();
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
  batch(() => {
    setState("bands", (prev) => [...prev, band]);
    setState("nextBandNum", num + 1);
    setState("activeBandId", band.id);
  });
  assignDefaultTargets(state.bands as BandState[]);
  markDirty();
}

export function removeBand(id: string) {
  if (state.bands.length <= 1) return; // нельзя удалить последнюю полосу
  const idx = bandIndex(id);
  if (idx < 0) return;
  batch(() => {
    // Сбрасываем linkedToNext у предыдущей полосы (если была связана с удаляемой)
    if (idx > 0 && state.bands[idx - 1].linkedToNext) {
      setState("bands", idx - 1, "linkedToNext", false);
    }
    setState("bands", (prev) => prev.filter((b) => b.id !== id));
    // Если удалили активную полосу — переключаемся на первую
    if (state.activeBandId === id) {
      setState("activeBandId", state.bands[0]?.id ?? SUM_ID);
    }
  });
  // Очищаем снэпшоты удалённой полосы
  setExportSnapshots(id, []);
  setFreqSnapshots(id, []);
  assignDefaultTargets(state.bands as BandState[]);
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

/** Переместить полосу из позиции fromIdx в toIdx. Перестраивает фильтры и linked-связи. */
export function moveBand(fromIdx: number, toIdx: number) {
  if (fromIdx === toIdx) return;
  if (fromIdx < 0 || fromIdx >= state.bands.length) return;
  if (toIdx < 0 || toIdx >= state.bands.length) return;

  const bands = [...state.bands].map(b => ({
    ...b,
    target: {
      ...b.target,
      high_pass: b.target.high_pass ? unwrapFilterConfig(b.target.high_pass) : null,
      low_pass: b.target.low_pass ? unwrapFilterConfig(b.target.low_pass) : null,
    },
  }));
  const [moved] = bands.splice(fromIdx, 1);
  bands.splice(toIdx, 0, moved);

  const n = bands.length;
  for (let i = 0; i < n; i++) {
    // First band (bass): remove HP, keep LP
    if (i === 0 && bands[i].target.high_pass) {
      bands[i].target = { ...bands[i].target, high_pass: null };
    }
    // Last band (tweeter): remove LP, keep HP
    if (i === n - 1 && bands[i].target.low_pass) {
      bands[i].target = { ...bands[i].target, low_pass: null };
    }

    // Rebuild linked-connections: auto-link where LP[i] freq ≈ HP[i+1] freq (±1%)
    bands[i].linkedToNext = false;
    if (i < n - 1) {
      const lp = bands[i].target.low_pass;
      const hp = bands[i + 1].target.high_pass;
      if (lp && hp && Math.abs(lp.freq_hz - hp.freq_hz) / lp.freq_hz < 0.01) {
        bands[i].linkedToNext = true;
      }
    }
  }

  setState("bands", bands);
  markDirty();
}

// ---------------------------------------------------------------------------
// Глобальные toggle-и отображения
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-band: measurement
// ---------------------------------------------------------------------------

export function setBandMeasurement(bandId: string, m: Measurement) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  batch(() => {
    setState("bands", idx, "measurement", m);
    setState("bands", idx, "settings", defaultSettings());
  });
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
  batch(() => {
    setState("bands", idx, "measurement", null);
    setState("bands", idx, "measurementFile", null);
    setState("bands", idx, "settings", null);
  });
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
  batch(() => {
    if (origPhase && !state.bands[idx].settings!.originalPhase) {
      setState("bands", idx, "settings", "originalPhase", [...origPhase]);
    }
    setState("bands", idx, "measurement", "phase", newPhase);
    setState("bands", idx, "settings", "delay_removed", true);
  });
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
      const nextHp = unwrapFilterConfig(state.bands[nextIdx].target.high_pass!);
      const lpPlain = unwrapFilterConfig(lp);
      setState("bands", nextIdx, "target", "high_pass", null);
      setState("bands", nextIdx, "target", "high_pass", {
        ...nextHp,
        freq_hz: lpPlain.freq_hz,
        filter_type: lpPlain.filter_type,
        order: lpPlain.order,
        shape: lpPlain.shape,
        linear_phase: lpPlain.linear_phase,
        q: lpPlain.q,
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

/** Unwrap a SolidJS store proxy FilterConfig into a plain object.
 *  Prevents cross-contamination when spreading proxy objects inside setState. */
function unwrapFilterConfig(f: import("../lib/types").FilterConfig): import("../lib/types").FilterConfig {
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
  };
}

export function setBandHighPass(bandId: string, config: import("../lib/types").FilterConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  // Deep clone to prevent shared reference with LP in SolidJS store
  if (config) config = unwrapFilterConfig(config);
  // Валидация: HP freq не может быть >= LP freq (enforce 5% minimum gap)
  if (config) {
    const lpFreq = state.bands[idx].target.low_pass?.freq_hz;
    if (lpFreq != null && config.freq_hz >= lpFreq) {
      config = { ...config, freq_hz: Math.round(lpFreq * 0.95) };
    }
  }
  batch(() => {
    // Force SolidJS to create a fresh store node by setting to null first.
    // Without this, SolidJS uses mergeStoreNode (in-place update) which
    // preserves shared internal nodes — if HP and LP ever point to the same
    // underlying object, changing one changes both (SolidJS store bug).
    // Setting null → object forces setProperty (fresh node) path.
    if (config && state.bands[idx].target.high_pass != null) {
      setState("bands", idx, "target", "high_pass", null);
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
        try {
          const plain = unwrapFilterConfig(prevLp);
          setState("bands", idx - 1, "target", "low_pass", null);
          setState("bands", idx - 1, "target", "low_pass", {
            ...plain,
            freq_hz: config.freq_hz,
            filter_type: config.filter_type,
            order: config.order,
            shape: config.shape,
            linear_phase: config.linear_phase,
            q: config.q,
          });
        } finally { _propagating = false; }
      }
    }
  });
  markDirty();
}

export function setBandLowPass(bandId: string, config: import("../lib/types").FilterConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  // Deep clone to prevent shared reference with HP in SolidJS store
  if (config) config = unwrapFilterConfig(config);
  // Валидация: LP freq не может быть <= HP freq (enforce 5% minimum gap)
  if (config) {
    const hpFreq = state.bands[idx].target.high_pass?.freq_hz;
    if (hpFreq != null && config.freq_hz <= hpFreq) {
      config = { ...config, freq_hz: Math.round(hpFreq * 1.05) };
    }
  }
  batch(() => {
    // Force fresh store node — see comment in setBandHighPass
    if (config && state.bands[idx].target.low_pass != null) {
      setState("bands", idx, "target", "low_pass", null);
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
        try {
          const plain = unwrapFilterConfig(nextHp);
          setState("bands", idx + 1, "target", "high_pass", null);
          setState("bands", idx + 1, "target", "high_pass", {
            ...plain,
            freq_hz: config.freq_hz,
            filter_type: config.filter_type,
            order: config.order,
            shape: config.shape,
            linear_phase: config.linear_phase,
            q: config.q,
          });
        } finally { _propagating = false; }
      }
    }
  });
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
  // Sort PEQ bands by frequency, return new index of the moved band
  const bands = [...state.bands[idx].peqBands];
  const movedBand = bands[peqIdx];
  bands.sort((a, b) => a.freq_hz - b.freq_hz);
  const newIdx = bands.indexOf(movedBand);
  setState("bands", idx, "peqBands", bands);
  markDirty();
  return newIdx;
}

// ---------------------------------------------------------------------------
// Per-band: Exclusion Zones
// ---------------------------------------------------------------------------

export function addExclusionZone(bandId: string, zone: ExclusionZone) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "exclusionZones", (prev) => [...prev, zone]);
  markDirty();
}

export function removeExclusionZone(bandId: string, zoneIdx: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "exclusionZones", (prev) => prev.filter((_, i) => i !== zoneIdx));
  markDirty();
}

export function updateExclusionZone(bandId: string, zoneIdx: number, patch: Partial<ExclusionZone>) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "exclusionZones", zoneIdx, (prev) => ({ ...prev, ...patch }));
  markDirty();
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
  // Guard: don't trigger store update if value unchanged (prevents render loop)
  if (Math.abs(state.bands[idx].crossNormDb - val) < 0.001) return;
  setState("bands", idx, "crossNormDb", val);
}

export function setBandColor(bandId: string, color: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "color", color);
  markDirty();
}

/** Set per-band alignment delay (seconds). Visual-only, used in SUM phase alignment. */
export function setAlignmentDelay(bandId: string, seconds: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "alignmentDelay", seconds);
  markDirty();
}

// ---------------------------------------------------------------------------
// Active tab (shared between ControlPanel ↔ App.tsx bottom panel)
// ---------------------------------------------------------------------------

export type ActiveTab = "measurements" | "target" | "peq" | "export";
export const [activeTab, setActiveTab] = createSignal<ActiveTab>("measurements");

export type PlotTab = "freq" | "ir" | "step" | "gd" | "export";
export const [plotTab, setPlotTab] = createSignal<PlotTab>("freq");

// ---------------------------------------------------------------------------
// Selected PEQ band index (for highlighting on graphs)
// ---------------------------------------------------------------------------

export const [selectedPeqIdx, setSelectedPeqIdx] = createSignal<number | null>(null);
export const [peqDragging, setPeqDragging] = createSignal(false);

// ---------------------------------------------------------------------------
// One-shot command: FrequencyPlot shows only these categories, then resets to null
// ---------------------------------------------------------------------------

export const [plotShowOnly, setPlotShowOnly] =
  createSignal<("measurement" | "target" | "corrected" | "peq")[] | null>(null);

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
export const [exportHybridPhase, setExportHybridPhase] = createSignal(false);

// FIR optimization settings
export const [firIterations, setFirIterations] = createSignal(3);
export const [firFreqWeighting, setFirFreqWeighting] = createSignal(true);
export const [firNarrowbandLimit, setFirNarrowbandLimit] = createSignal(true);
export const [firNbSmoothingOct, setFirNbSmoothingOct] = createSignal(0.333);
export const [firNbMaxExcess, setFirNbMaxExcess] = createSignal(6.0);
export const [firMaxBoost, setFirMaxBoost] = createSignal(24.0);
export const [firNoiseFloor, setFirNoiseFloor] = createSignal(-150.0);

// ---------------------------------------------------------------------------
// Snapshot overlays: frozen curves for visual A/B comparison.
// Per-band Maps — each band has its own set of snapshots.
// Stored at module level so they survive component unmount/remount.
// ---------------------------------------------------------------------------

export interface ExportSnapshot {
  label: string;
  freq: number[];
  mag: number[];
  phase: (number | null)[];
  color: string;
}

// --- Export Plot snapshots (per-band) ---
const [_exportSnapMap, _setExportSnapMap] = createSignal<Map<string, ExportSnapshot[]>>(new Map());

export function exportSnapshots(bandId: string): ExportSnapshot[] {
  return _exportSnapMap().get(bandId) ?? [];
}

export function setExportSnapshots(bandId: string, snaps: ExportSnapshot[]) {
  const m = new Map(_exportSnapMap());
  if (snaps.length === 0) m.delete(bandId);
  else m.set(bandId, snaps);
  _setExportSnapMap(m);
}

export function clearAllExportSnapshots() {
  _setExportSnapMap(new Map());
}

// --- FrequencyPlot snapshots (per-band, corrected curve) ---
export interface FreqSnapshot {
  label: string;
  freq: number[];
  mag: number[];
  phase: (number | null)[];
  color: string;
}

const [_freqSnapMap, _setFreqSnapMap] = createSignal<Map<string, FreqSnapshot[]>>(new Map());

export function freqSnapshots(bandId: string): FreqSnapshot[] {
  return _freqSnapMap().get(bandId) ?? [];
}

export function setFreqSnapshots(bandId: string, snaps: FreqSnapshot[]) {
  const m = new Map(_freqSnapMap());
  if (snaps.length === 0) m.delete(bandId);
  else m.set(bandId, snaps);
  _setFreqSnapMap(m);
}

export function clearAllFreqSnapshots() {
  _setFreqSnapMap(new Map());
}

// Generic plot snapshots for IR/Step, GD, Export tabs
export interface PlotSnapshot {
  label: string;
  color: string;
  tab: "ir" | "gd" | "export";
  // Time-domain data (IR/Step)
  timeMs?: number[];
  impulse?: number[];
  step?: number[];
  // Frequency-domain data (GD, Export)
  freq?: number[];
  gdMs?: number[];
  exportMag?: number[];
  exportPhase?: number[];
}

const [_plotSnapMap, _setPlotSnapMap] = createSignal<Map<string, PlotSnapshot[]>>(new Map());

export function plotSnapshots(bandId: string, tab: string): PlotSnapshot[] {
  return (_plotSnapMap().get(bandId) ?? []).filter(s => s.tab === tab);
}

export function addPlotSnapshot(bandId: string, snap: PlotSnapshot) {
  const m = new Map(_plotSnapMap());
  const existing = m.get(bandId) ?? [];
  m.set(bandId, [...existing, snap]);
  _setPlotSnapMap(m);
}

export function clearPlotSnapshots(bandId: string, tab: string) {
  const m = new Map(_plotSnapMap());
  const existing = m.get(bandId) ?? [];
  const filtered = existing.filter(s => s.tab !== tab);
  if (filtered.length === 0) m.delete(bandId);
  else m.set(bandId, filtered);
  _setPlotSnapMap(m);
}

export function clearAllPlotSnapshots() {
  _setPlotSnapMap(new Map());
}

// Export plot Y-scale: persists across ExportPlot unmount/remount.
// null = not yet set (use auto-range), {min,max} = user has zoomed/scrolled.
export const [exportYScale, setExportYScale] = createSignal<{ min: number; max: number } | null>(null);

// ---------------------------------------------------------------------------
// Dirty state: true when project has unsaved changes
// ---------------------------------------------------------------------------

export const [isDirty, setIsDirty] = createSignal(false);
export const [bandsVersion, setBandsVersion] = createSignal(0);
export function markDirty() {
  if (!isDirty()) setIsDirty(true);
  setBandsVersion(v => v + 1);
}

// ---------------------------------------------------------------------------
// Export metrics (set by UnifiedPlot export tab, read by ControlPanel)
// ---------------------------------------------------------------------------

export interface ExportMetrics {
  taps: number; sampleRate: number; window: string; phaseLabel: string;
  peqCount: number; normDb: number; causality: number;
  preRingMs: number; maxMagErr: number; gdRippleMs: number;
}
export const [exportMetrics, setExportMetrics] = createSignal<ExportMetrics | null>(null);

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export { state as appState, SUM_ID };
