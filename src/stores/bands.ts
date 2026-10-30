import { createStore, reconcile } from "solid-js/store";
import { createSignal, batch, untrack, createRoot, createEffect, on, type Accessor } from "solid-js";
import type { Measurement, TargetCurve, MergeConfig, PeqBand, FirResult, WindowType, ExclusionZone, AnalysisResult, PeqOptimizedTarget, FilterConfig } from "../lib/types";
import { MEASUREMENT_COLORS, cloneFilterConfig, withCrossoverOf, F_MIN_WORK, F_MAX_WORK } from "../lib/types";
import {
  pushHistory,
  beginInteraction,
  commitInteraction,
  registerHistoryHooks,
  historyBandIds,
  type HistoryEntry,
  type LightBand,
} from "./history";
// Re-exported for explicit drag wiring elsewhere.
export { pushHistory, beginInteraction, commitInteraction } from "./history";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SmoothingMode = "off" | "1/3" | "1/6" | "1/12" | "1/24" | "var";
export const SMOOTHING_MODES: readonly SmoothingMode[] = ["off", "1/3", "1/6", "1/12", "1/24", "var"];
export function isSmoothingMode(v: unknown): v is SmoothingMode {
  return typeof v === "string" && (SMOOTHING_MODES as readonly string[]).includes(v);
}

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
  analysis: AnalysisResult | null;
  analysisDismissed: boolean;
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
  /** Snapshot of target/exclusion at the time of the last successful
   *  optimization. null = optimize never ran (or PEQ cleared). Used for
   *  peqStale detection when target diverges. */
  peqOptimizedTarget: PeqOptimizedTarget | null;
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
  return { smoothing: "off", delay_seconds: null, distance_meters: null, delay_removed: false, originalPhase: null, floorBounce: null, mergeSource: null, analysis: null, analysisDismissed: false };
}

/** Compute equal-octave crossover frequencies for n bands spanning fMin–fMax.
 *  Returns n-1 crossover points. E.g. 3 bands → 2 crossovers. */
function equalOctaveCrossovers(n: number, fMin = F_MIN_WORK, fMax = F_MAX_WORK): number[] {
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

/** Band positions before an add/remove: id → index, and the band count. */
type PrevLayout = { n: number; idx: Map<string, number> };
function captureLayout(): PrevLayout {
  return { n: state.bands.length, idx: new Map(state.bands.map((b, i) => [b.id, i])) };
}

/** b141.103 (audit 3.16): a band without a measurement whose crossovers are
 *  still the equal-octave split of the previous layout (or that has none) —
 *  safe to re-split. One the user set (LP 3333 Hz) used to be overwritten
 *  by the new default on every add/remove. */
function isOnDefaultSplit(b: BandState, prev: PrevLayout | undefined): boolean {
  const hp = b.target.high_pass, lp = b.target.low_pass;
  if (!hp && !lp) return true;
  const i = prev?.idx.get(b.id);
  if (prev === undefined || i === undefined) return true; // a new band
  const x = equalOctaveCrossovers(prev.n);
  const expHp = i > 0 ? x[i - 1] : null;
  const expLp = i < prev.n - 1 ? x[i] : null;
  const near = (f: FilterConfig | null, e: number | null) =>
    e === null ? f === null : (f !== null && Math.abs(f.freq_hz - e) <= 1);
  return near(hp, expHp) && near(lp, expLp);
}

type Xo = FilterConfig | null;

/** b141.103 (audit 3.17): fill the internal edges a reorder (or a kept
 *  user band) left half-open — a band that was first/last sits in the middle
 *  without its HP/LP, and two full-range bands end up side by side. A
 *  half-present edge takes the existing side's filter when it fits between
 *  the two bands' outer crossovers; otherwise the edge goes to their
 *  geometric middle. Mutates `hp`/`lp`; returns the indices it touched. */
function repairEdges(hp: Xo[], lp: Xo[]): Set<number> {
  const touched = new Set<number>();
  for (let i = 0; i < hp.length - 1; i++) {
    const l = lp[i], h = hp[i + 1];
    if (l && h) continue;
    const lo = hp[i]?.freq_hz ?? F_MIN_WORK;
    const hi = lp[i + 1]?.freq_hz ?? F_MAX_WORK;
    const src = l ?? h;
    if (src && src.freq_hz > lo && src.freq_hz < hi) {
      if (!l) { lp[i] = cloneFilterConfig(src); touched.add(i); }
      else { hp[i + 1] = cloneFilterConfig(src); touched.add(i + 1); }
    } else if (lo < hi) {
      const f = Math.round(Math.sqrt(lo * hi));
      const base = src ? cloneFilterConfig(src) : DEFAULT_FILTER;
      lp[i] = { ...base, freq_hz: f };
      hp[i + 1] = { ...base, freq_hz: f };
      touched.add(i); touched.add(i + 1);
    }
  }
  return touched;
}

/** Assign default HP/LP filters based on equal-octave split — only to bands
 *  that have NO measurement and still sit on the default split (b141.103).
 *  An edge next to a band the user configured takes that band's frequency,
 *  so linked neighbours stay on one crossover; an unlinked pair keeps its own
 *  (b141.121). Links are the caller's: this function does not touch them. */
function assignDefaultTargets(bands: BandState[], prev?: PrevLayout) {
  const n = bands.length;
  if (n <= 0) return;
  const xovers = equalOctaveCrossovers(n);
  const pristine = bands.map((b) => !b.measurement && isOnDefaultSplit(b, prev));
  const hp: Xo[] = bands.map((b) => cloneFilterConfig(b.target.high_pass ?? null));
  const lp: Xo[] = bands.map((b) => cloneFilterConfig(b.target.low_pass ?? null));
  const edge = xovers.map((f, e) => {
    if (!bands[e].linkedToNext) return f;
    if (!pristine[e] && lp[e]) return lp[e]!.freq_hz;
    if (!pristine[e + 1] && hp[e + 1]) return hp[e + 1]!.freq_hz;
    return f;
  });
  for (let i = 0; i < n; i++) {
    if (!pristine[i]) continue;
    let hpF = i > 0 ? edge[i - 1] : null;
    let lpF = i < n - 1 ? edge[i] : null;
    if (hpF !== null && lpF !== null && hpF >= lpF) {
      // A neighbour's crossover would invert this band — keep the split.
      hpF = xovers[i - 1];
      lpF = xovers[i];
    }
    // Only the frequency moves: a type/order/phase the band already has stays.
    hp[i] = hpF !== null ? { ...(hp[i] ?? DEFAULT_FILTER), freq_hz: hpF } : null;
    lp[i] = lpF !== null ? { ...(lp[i] ?? DEFAULT_FILTER), freq_hz: lpF } : null;
  }
  const repaired = repairEdges(hp, lp);
  for (let i = 0; i < n; i++) {
    // Last band cannot be linked downstream; clear stale flag if any.
    if (i === n - 1 && bands[i].linkedToNext) {
      setState("bands", i, "linkedToNext", false);
    }
    // Bands with a measurement or crossovers the user set are left alone,
    // except for an edge the layout change left open.
    if (!pristine[i] && !repaired.has(i)) continue;
    // Force fresh store nodes by setting null first (prevents shared-node bug)
    setState("bands", i, "target", "high_pass", null);
    setState("bands", i, "target", "high_pass", hp[i]);
    setState("bands", i, "target", "low_pass", null);
    setState("bands", i, "target", "low_pass", lp[i]);
    setState("bands", i, "targetEnabled", true);
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
    peqOptimizedTarget: null,
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

/** b141.105 (audit 2026-10-23 §5.6): the preset a target matches, for the
 *  «Пресет» select — compared on the curve's shape (tilt and shelves), not on
 *  level or crossover filters. «custom» when nothing matches. */
export function presetOfTarget(t: TargetCurve | null | undefined): PresetName {
  if (!t) return "flat";
  for (const name of ["flat", "harman", "bk"] as const) {
    const p = PRESETS[name]();
    if (p.tilt_db_per_octave === t.tilt_db_per_octave
      && (p.tilt_db_per_octave === 0 || p.tilt_ref_freq === t.tilt_ref_freq)
      && JSON.stringify(p.low_shelf) === JSON.stringify(t.low_shelf ?? null)
      && JSON.stringify(p.high_shelf) === JSON.stringify(t.high_shelf ?? null)) return name;
  }
  return "custom";
}

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
  clearAllFreqSnapshots();
  // b141.103 (audit 3.11): IR/GD/Export snapshots belong to the old project's
  // bands too. (3.20) So do parked heavy parts — history is cleared after.
  clearAllPlotSnapshots();
  parkedHeavy.clear();
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
  pushHistory("Add band");
  const prev = captureLayout();
  const num = state.nextBandNum;
  const band = createBand(num);
  // b141.128 (external audit 2026-10-29): a band renamed to «Band 3» and the
  // third band added after it shared the name.
  band.name = uniqueBandName(band.name, state.bands);
  batch(() => {
    // b141.121 (external audit 2026-10-29): only the new pair gets a link. Every
    // pair used to be relinked here, so a pair the user had unlinked was
    // linked again and its neighbour's crossover rewritten.
    if (state.bands.length > 0) setState("bands", state.bands.length - 1, "linkedToNext", true);
    setState("bands", (prev) => [...prev, band]);
    setState("nextBandNum", num + 1);
    setState("activeBandId", band.id);
  });
  assignDefaultTargets(state.bands as BandState[], prev);
  markDirty();
}

// b141.66 (audit stage 2 B2): history entries are "light" — they carry no
// measurement/settings (MBs per band). Undo of a deletion therefore had no
// source for them and brought the band back empty. The heavy parts of a
// removed band are parked here by id and picked up by _applyBandsLight.
// Measurements are immutable (the eval cache keys them by identity), so
// holding the reference is enough.
const parkedHeavy = new Map<string, Pick<BandState,
  "measurement" | "measurementFile" | "settings" | "firResult" | "crossNormDb">>();

function parkHeavy(b: BandState) {
  parkedHeavy.set(b.id, {
    measurement: b.measurement, measurementFile: b.measurementFile,
    settings: b.settings ? JSON.parse(JSON.stringify(b.settings)) : null,
    firResult: b.firResult, crossNormDb: b.crossNormDb,
  });
}

/** b141.103 (audit 3.20): drop parked parts no history entry can bring back
 *  (the map only grew — every removed band's measurement stayed in memory). */
function pruneParked() {
  if (parkedHeavy.size === 0) return;
  const live = historyBandIds();
  for (const id of [...parkedHeavy.keys()]) {
    if (!live.has(id) && bandIndex(id) < 0) parkedHeavy.delete(id);
  }
}

/** Test hook: how many removed bands are parked. */
export function _parkedCount(): number { return parkedHeavy.size; }

export function removeBand(id: string) {
  if (state.bands.length <= 1) return; // нельзя удалить последнюю полосу
  const idx = bandIndex(id);
  if (idx < 0) return;
  pushHistory("Remove band");
  parkHeavy(state.bands[idx]);
  const prev = captureLayout();
  batch(() => {
    // b140.8.2: linkedToNext describes positional coupling — the two pairs
    // around the deleted band become one. b141.121: it stays linked only if
    // both were, so a pair the user unlinked does not come back linked. The
    // deleted band's own flag is false when it was last, which clears prev's.
    if (idx > 0 && state.bands[idx - 1].linkedToNext && !state.bands[idx].linkedToNext) {
      setState("bands", idx - 1, "linkedToNext", false);
    }
    setState("bands", (prev) => prev.filter((b) => b.id !== id));
    // Если удалили активную полосу — переключаемся на первую
    if (state.activeBandId === id) {
      setState("activeBandId", state.bands[0]?.id ?? SUM_ID);
    }
  });
  // Очищаем снэпшоты удалённой полосы
  setFreqSnapshots(id, []);
  assignDefaultTargets(state.bands as BandState[], prev);
  pruneParked();
  markDirty();
}

export function setActiveBand(id: string) {
  setState("activeBandId", id);
}

export function setActiveBandSum() {
  setState("activeBandId", SUM_ID);
}

/** b141.66 (audit stage 2 B4): band names are unique — the Σ table, legend
 *  visibility and IR exclusions look bands up by name, so a duplicate made
 *  the second band's DELAY field edit the first band. A taken name gets
 *  " (2)", " (3)", … */
export function uniqueBandName(name: string, others: readonly { id: string; name: string }[], selfId: string): string {
  const taken = new Set(others.filter((b) => b.id !== selfId).map((b) => b.name));
  if (!taken.has(name)) return name;
  for (let k = 2; ; k++) {
    const c = `${name} (${k})`;
    if (!taken.has(c)) return c;
  }
}

export function renameBand(id: string, name: string) {
  const idx = bandIndex(id);
  if (idx < 0) return;
  const unique = uniqueBandName(name, state.bands, id);
  if (unique === state.bands[idx].name) return;
  pushHistory("Rename band");
  setState("bands", idx, "name", unique);
  markDirty();
}

/** Переместить полосу из позиции fromIdx в toIdx. Перестраивает фильтры и linked-связи. */
export function moveBand(fromIdx: number, toIdx: number) {
  if (fromIdx === toIdx) return;
  if (fromIdx < 0 || fromIdx >= state.bands.length) return;
  if (toIdx < 0 || toIdx >= state.bands.length) return;
  pushHistory("Move band");

  const bands = [...state.bands].map(b => ({
    ...b,
    target: {
      ...b.target,
      high_pass: b.target.high_pass ? cloneFilterConfig(b.target.high_pass) : null,
      low_pass: b.target.low_pass ? cloneFilterConfig(b.target.low_pass) : null,
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
  }

  // b141.103 (audit 3.17): rebuild the internal edges the move broke.
  {
    const hp = bands.map((b) => b.target.high_pass);
    const lp = bands.map((b) => b.target.low_pass);
    for (const i of repairEdges(hp, lp)) {
      bands[i].target = { ...bands[i].target, high_pass: hp[i], low_pass: lp[i] };
    }
  }

  for (let i = 0; i < n; i++) {
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
    // b141.103 (audit 3.18): the project file belonged to the previous
    // measurement — cleared until the new one is copied in (a failed copy
    // used to leave the band pointing at the old file; on save the pending
    // copy now retries from source_path).
    setState("bands", idx, "measurementFile", null);
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
  if (state.bands[idx].settings!.smoothing === mode) return;
  // b141.103 (audit 3.9): smoothing is in the history entry — record it.
  pushHistory("Smoothing");
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


// ---------------------------------------------------------------------------
// Per-band: target / filters
// ---------------------------------------------------------------------------

export function toggleBandTarget(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Toggle target");
  setState("bands", idx, "targetEnabled", !state.bands[idx].targetEnabled);
  markDirty();
}


// Инвертирование полярности полосы (фаза +180°)
export function toggleBandInverted(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Invert polarity");
  setState("bands", idx, "inverted", !state.bands[idx].inverted);
  markDirty();
}

// Связь LP текущей полосы ↔ HP следующей
export function toggleBandLinked(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0 || idx >= state.bands.length - 1) return; // нельзя link последнюю полосу
  pushHistory("Toggle link");
  const newVal = !state.bands[idx].linkedToNext;
  setState("bands", idx, "linkedToNext", newVal);
  // При включении связи — синхронизируем LP → HP следующей (freq + type + order + linear_phase)
  if (newVal) {
    const lp = state.bands[idx].target.low_pass;
    const nextIdx = idx + 1;
    if (lp && state.bands[nextIdx].target.high_pass) {
      const nextHp = cloneFilterConfig(state.bands[nextIdx].target.high_pass!);
      const lpPlain = cloneFilterConfig(lp);
      setState("bands", nextIdx, "target", "high_pass", null);
      setState("bands", nextIdx, "target", "high_pass", withCrossoverOf(nextHp, lpPlain));
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
  pushHistory("Apply preset");
  // b141.121 (external audit 2026-10-29): a preset is a tilt and shelves
  // (`detectPreset` compares only those). Replacing the whole target wiped the
  // band's crossovers and level.
  const { tilt_db_per_octave, tilt_ref_freq, low_shelf, high_shelf } = PRESETS[preset]();
  setState("bands", idx, "target", { tilt_db_per_octave, tilt_ref_freq, low_shelf, high_shelf });
  markDirty();
}

export function setBandReferenceLevel(bandId: string, db: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Reference level");
  setState("bands", idx, "target", "reference_level_db", db);
  markDirty();
}

export function setBandTilt(bandId: string, dbPerOctave: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Target tilt");
  setState("bands", idx, "target", "tilt_db_per_octave", dbPerOctave);
  markDirty();
}

export function setBandHighPass(bandId: string, config: import("../lib/types").FilterConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Edit crossover");
  // Deep clone to prevent shared reference with LP in SolidJS store
  if (config) config = cloneFilterConfig(config);
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
          const plain = cloneFilterConfig(prevLp);
          setState("bands", idx - 1, "target", "low_pass", null);
          setState("bands", idx - 1, "target", "low_pass", withCrossoverOf(plain, config));
        } finally { _propagating = false; }
      }
    }
  });
  markDirty();
}

export function setBandLowPass(bandId: string, config: import("../lib/types").FilterConfig | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Edit crossover");
  // Deep clone to prevent shared reference with HP in SolidJS store
  if (config) config = cloneFilterConfig(config);
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
          const plain = cloneFilterConfig(nextHp);
          setState("bands", idx + 1, "target", "high_pass", null);
          setState("bands", idx + 1, "target", "high_pass", withCrossoverOf(plain, config));
        } finally { _propagating = false; }
      }
    }
  });
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
  pushHistory("Remove PEQ");
  setState("bands", idx, "peqBands", (prev) => prev.filter((_, i) => i !== peqIdx));
  markDirty();
}

export function clearBandPeqBands(bandId: string) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Clear PEQ");
  batch(() => {
    setState("bands", idx, "peqBands", []);
    setState("bands", idx, "peqOptimizedTarget", null);
  });
  markDirty();
}

export function setBandPeqOptimizedTarget(bandId: string, snapshot: PeqOptimizedTarget | null) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  setState("bands", idx, "peqOptimizedTarget", snapshot);
  markDirty();
}

export function addPeqBand(bandId: string, band: PeqBand) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Add PEQ");
  // Добавляем в начало — новый фильтр всегда первая строка таблицы
  setState("bands", idx, "peqBands", (prev) => [band, ...prev]);
  markDirty();
}

export function updatePeqBand(bandId: string, peqIdx: number, patch: Partial<PeqBand>) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  const peq = state.bands[idx].peqBands[peqIdx];
  if (!peq) return;
  pushHistory("PEQ adjust");
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
  pushHistory("Exclusion zone");
  setState("bands", idx, "exclusionZones", (prev) => [...prev, zone]);
  markDirty();
}

export function removeExclusionZone(bandId: string, zoneIdx: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Exclusion zone");
  setState("bands", idx, "exclusionZones", (prev) => prev.filter((_, i) => i !== zoneIdx));
  markDirty();
}

export function updateExclusionZone(bandId: string, zoneIdx: number, patch: Partial<ExclusionZone>) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Exclusion zone");
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

export function setBandAnalysis(bandId: string, analysis: AnalysisResult | null) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "analysis", analysis);
  setState("bands", idx, "settings", "analysisDismissed", false);
  markDirty();
}

export function setBandAnalysisDismissed(bandId: string, dismissed: boolean) {
  const idx = bandIndex(bandId);
  if (idx < 0 || !state.bands[idx].settings) return;
  setState("bands", idx, "settings", "analysisDismissed", dismissed);
  markDirty();
}

/** Set per-band alignment delay (seconds). Visual-only, used in SUM phase alignment. */
export function setAlignmentDelay(bandId: string, seconds: number) {
  const idx = bandIndex(bandId);
  if (idx < 0) return;
  pushHistory("Alignment delay");
  batch(() => {
    setState("bands", idx, "alignmentDelay", seconds);
    markDirty();
  });
}

// ---------------------------------------------------------------------------
// Active tab (shared between ControlPanel ↔ App.tsx bottom panel)
// ---------------------------------------------------------------------------

export type ActiveTab = "measurements" | "target" | "peq" | "export";
export const [activeTab, setActiveTab] = createSignal<ActiveTab>("measurements");

export type PlotTab = "freq" | "ir" | "gd" | "export";
export const [plotTab, setPlotTab] = createSignal<PlotTab>("freq");

/** Auto-FIT request after a measurement import (lived in App.tsx until
 *  2026-09-05 — components importing from the root component was an
 *  App ↔ FrequencyPlot/ControlPanel cycle). */
export const [needAutoFit, setNeedAutoFit] = createSignal(false);

// One-shot cleanup of the deleted SUM-mode toggle's localStorage key
// (the toggle itself was removed in b140.15). Best-effort — silent
// failure is fine if storage is unavailable.
try { localStorage.removeItem("phaseforge.sumMode"); } catch (_) { /* ignore */ }

// ---------------------------------------------------------------------------
// Selected PEQ band index (for highlighting on graphs)
// ---------------------------------------------------------------------------

export const [selectedPeqIdx, setSelectedPeqIdx] = createSignal<number | null>(null);
// b141.105 (audit 2026-10-23 §5.5): the selection is an index into the ACTIVE
// band's PEQ list — kept across a band switch it pointed at another band's
// filter, and the wheel then changed that filter's Q. Any change of the
// active band (tab, delete, project load, undo) clears it.
createRoot(() => createEffect(on(() => state.activeBandId, () => setSelectedPeqIdx(null), { defer: true })));
export const [peqDragging, setPeqDragging] = createSignal(false);

// ---------------------------------------------------------------------------
// One-shot command: FrequencyPlot shows only these categories, then resets to null
// ---------------------------------------------------------------------------

export const [plotShowOnly, setPlotShowOnly] =
  createSignal<("measurement" | "target" | "corrected" | "peq" | "snapshot" | "fir" | "corrfir")[] | null>(null);

// ---------------------------------------------------------------------------
// Shared X-scale (sync frequency axis between top FrequencyPlot and bottom PeqResponsePlot)
// ---------------------------------------------------------------------------

export interface XScale { min: number; max: number }
const [_xScale, _setXScale] = createSignal<XScale>({ min: F_MIN_WORK, max: F_MAX_WORK });
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

/** b141.103 (audit 2026-10-23 3.2): a project-wide setting saved in the
 *  .pfproj. Its setter marks the project dirty when the value actually
 *  changes — SR/taps/window/FIR/optimizer edits used to be lost on close
 *  without the «unsaved changes» prompt. Not part of undo history (3.1). */
export function settingSignal<T>(initial: T): [Accessor<T>, (v: T) => void] {
  const [get, set] = createSignal<T>(initial);
  const setter = (v: T) => {
    if (Object.is(untrack(get), v)) return;
    set(() => v);
    markSettingsDirty();
  };
  return [get, setter];
}

export const [exportSampleRate, setExportSampleRate] = settingSignal(48000);
export const [exportTaps, setExportTaps] = settingSignal(65536);
export const [exportWindow, setExportWindow] = settingSignal<WindowType>("Blackman");
export const [exportHybridPhase, setExportHybridPhase] = settingSignal(false);

// FIR optimization settings
export const [firIterations, setFirIterations] = settingSignal(3);
export const [firFreqWeighting, setFirFreqWeighting] = settingSignal(true);
export const [firNarrowbandLimit, setFirNarrowbandLimit] = settingSignal(true);
export const [firNbSmoothingOct, setFirNbSmoothingOct] = settingSignal(0.333);
export const [firNbMaxExcess, setFirNbMaxExcess] = settingSignal(6.0);
export const [firMaxBoost, setFirMaxBoost] = settingSignal(24.0);
export const [firNoiseFloor, setFirNoiseFloor] = settingSignal(-150.0);

// ---------------------------------------------------------------------------
// Snapshot overlays: frozen curves for visual A/B comparison.
// Per-band Maps — each band has its own set of snapshots.
// Stored at module level so they survive component unmount/remount.
// ---------------------------------------------------------------------------

// b141.5 (audit): the ExportSnapshot subsystem was write-only — superseded
// by PlotSnapshot with tab:"export". Removed.

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

export function clearAllPlotSnapshots() {
  _setPlotSnapMap(new Map());
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


// Export plot Y-scale: persists across ExportPlot unmount/remount.
// null = not yet set (use auto-range), {min,max} = user has zoomed/scrolled.
export const [exportYScale, setExportYScale] = createSignal<{ min: number; max: number } | null>(null);

// ---------------------------------------------------------------------------
// Dirty state: true when project has unsaved changes
// ---------------------------------------------------------------------------

export const [isDirty, setIsDirty] = createSignal(false);
export const [bandsVersion, setBandsVersion] = createSignal(0);
/** b141.103 (audit 3.13): bumped on EVERY change that makes the project
 *  dirty. A save clears the flag only if no change landed while it was
 *  writing (an edit during the await used to be marked as saved). */
let _dirtyGen = 0;
export function dirtyGeneration(): number { return _dirtyGen; }
/** A project-wide setting changed: dirty, but no band content to rebuild. */
export function markSettingsDirty() {
  _dirtyGen++;
  if (!untrack(isDirty)) setIsDirty(true);
}
export function markDirty() {
  markSettingsDirty();
  setBandsVersion(v => v + 1);
}

// ---------------------------------------------------------------------------
// History snapshot helpers (b132): band content only (b141.103, audit 3.1 —
// global export/FIR/optimizer settings are not undone).
// ---------------------------------------------------------------------------

export function _captureBandsLight(label: string): HistoryEntry {
  const lightBands: LightBand[] = state.bands.map((b) => ({
    id: b.id,
    name: b.name,
    peqBands: JSON.parse(JSON.stringify(b.peqBands)),
    target: JSON.parse(JSON.stringify(b.target)),
    targetEnabled: b.targetEnabled,
    inverted: b.inverted,
    linkedToNext: b.linkedToNext,
    alignmentDelay: b.alignmentDelay,
    color: b.color,
    exclusionZones: JSON.parse(JSON.stringify(b.exclusionZones)),
    peqOptimizedTarget: b.peqOptimizedTarget
      ? JSON.parse(JSON.stringify(b.peqOptimizedTarget))
      : null,
    smoothing: b.settings?.smoothing ?? null,
  }));
  return {
    bands: lightBands,
    activeBandId: state.activeBandId,
    nextBandNum: state.nextBandNum,
    label,
    ts: Date.now(),
  };
}

export function _applyBandsLight(entry: HistoryEntry): void {
  batch(() => {
    const currentById = new Map(state.bands.map((b) => [b.id, b]));
    // b141.103 (audit 3.6): a band this entry does not hold (undo of «Add
    // band») keeps its heavy parts parked — redo used to bring it back
    // without its measurement.
    const keep = new Set(entry.bands.map((lb) => lb.id));
    for (const b of state.bands) if (!keep.has(b.id)) parkHeavy(b);
    const newBands: BandState[] = entry.bands.map((lb) => {
      const cur = currentById.get(lb.id) ?? parkedHeavy.get(lb.id);
      const target = JSON.parse(JSON.stringify(lb.target)) as TargetCurve;
      const peqBands = JSON.parse(JSON.stringify(lb.peqBands));
      const exclusionZones = JSON.parse(JSON.stringify(lb.exclusionZones));
      const peqOptimizedTarget = lb.peqOptimizedTarget
        ? JSON.parse(JSON.stringify(lb.peqOptimizedTarget))
        : null;
      return {
        id: lb.id,
        name: lb.name,
        measurement: cur?.measurement ?? null,
        measurementFile: cur?.measurementFile ?? null,
        settings: cur?.settings ?? null,
        target,
        targetEnabled: lb.targetEnabled,
        inverted: lb.inverted,
        linkedToNext: lb.linkedToNext,
        peqBands,
        peqOptimizedTarget,
        exclusionZones,
        firResult: cur?.firResult ?? null,
        crossNormDb: cur?.crossNormDb ?? 0,
        color: lb.color,
        alignmentDelay: lb.alignmentDelay,
      };
    });
    // Build a plain AppState literal (no spread of store proxy).
    setState(reconcile({
      bands: newBands,
      activeBandId: entry.activeBandId,
      showPhase: state.showPhase,
      showMag: state.showMag,
      showTarget: state.showTarget,
      nextBandNum: entry.nextBandNum,
    }));
    entry.bands.forEach((lb, i) => {
      if (lb.smoothing && isSmoothingMode(lb.smoothing) && state.bands[i].settings
        && state.bands[i].settings!.smoothing !== lb.smoothing) {
        setState("bands", i, "settings", "smoothing", lb.smoothing);
      }
    });
    // b141.128 (external audit 2026-10-29): the restored list may differ, and an
    // index kept across it moved the highlight and the wheel to another filter.
    setSelectedPeqIdx(null);
    markDirty();
  });
  pruneParked();
}

registerHistoryHooks(_captureBandsLight, _applyBandsLight);

// ---------------------------------------------------------------------------
// Export metrics (set by UnifiedPlot export tab, read by ControlPanel)
// ---------------------------------------------------------------------------

export interface ExportMetrics {
  taps: number; sampleRate: number; window: string; phaseLabel: string;
  peqCount: number; normDb: number; causality: number;
  preRingMs: number; maxMagErr: number; gdRippleMs: number;
  /** b141.37: pre-ring threshold (dB re peak), raised above a loud floor. */
  preRingThresholdDb: number;
  /** Floor raised the threshold — preRingMs is a lower bound. */
  preRingLimited: boolean;
  /** Loudest pre-peak level outside the ring zone; null = no room to measure. */
  floorDb: number | null;
  ringZoneMs: number; ringZoneHz: number | null; ringZoneLinear: boolean;
  /** Passband the magnitude error and GD ripple are taken over. */
  passbandLoHz: number; passbandHiHz: number;
  /** b141.40: zero-phase ultrasonic low-pass corner in the FIR, null if none. */
  ultrasonicLpHz: number | null;
}
export const [exportMetrics, setExportMetrics] = createSignal<ExportMetrics | null>(null);

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export { state as appState, SUM_ID };
