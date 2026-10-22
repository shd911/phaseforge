// ---------------------------------------------------------------------------
// PEQ Auto-Fit shared store — extracted from ControlPanel.tsx (b82.06)
// ---------------------------------------------------------------------------
import { createSignal, batch } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import type { PeqBand, PeqConfig, PeqResult, FilterConfig, ExclusionZone, PeqOptimizedTarget, Measurement } from "../lib/types";
import { cloneFilterConfig, F_MAX_WORK, F_MAX_REF, fMaxForRate } from "../lib/types";
import {
  activeBand,
  appState,
  exportHybridPhase,
  exportSampleRate,
  exportTaps,
  setBandPeqBands,
  clearBandPeqBands,
  setBandPeqOptimizedTarget,
  setSelectedPeqIdx,
  _captureBandsLight,
  _applyBandsLight,
} from "./bands";
import type { BandState } from "./bands";
import { pushHistory, registerHistoryHooks, type HistoryEntry } from "./history";
import { passbandRange } from "../lib/band-evaluator/extension";
import { showToast } from "../lib/toast";

// --- Signals ---
export const [tolerance, setTolerance] = createSignal(1.0);
export const [maxBands, setMaxBands] = createSignal(20);
export const [gainRegularization, setGainRegularization] = createSignal(0.0);
export const [peqFloor, setPeqFloor] = createSignal(60); // dB below reference — don't optimize below this
// b141.87: «Без отражений» — fit the smoothed measurement (1/48 < 100 Hz,
// 1/3 above), not the reflection ripple of a long-window export.
export const [peqSmoothFit, setPeqSmoothFit] = createSignal(true);
// b141.92: «Подъём до» — PEQ may boost only where the target is within this
// many dB of its reference; deeper it may only cut (0 = no limit).
export const [peqBoostFloor, setPeqBoostFloor] = createSignal(30);
export type PeqRangeMode = "auto" | "direct";
export const [peqRangeMode, setPeqRangeMode] = createSignal<PeqRangeMode>("auto");
export const [peqDirectLow, setPeqDirectLow] = createSignal(20);
export const [peqDirectHigh, setPeqDirectHigh] = createSignal(F_MAX_WORK);
export const [computing, setComputing] = createSignal(false);
export const [peqError, setPeqError] = createSignal<string | null>(null);
// b141.67 (audit stage 2 B5): fit statistics are PER BAND. Global signals
// showed band A's error on band B and, after «Оптимизировать все», the
// maximum over all bands and the sum of all iterations on every band.
const [peqStats, setPeqStats] = createSignal<Record<string, { maxErr: number; iters: number }>>({});
export function maxErr(): number | null {
  const b = activeBand();
  return b ? peqStats()[b.id]?.maxErr ?? null : null;
}
export function iters(): number | null {
  const b = activeBand();
  return b ? peqStats()[b.id]?.iters ?? null : null;
}
function setBandStats(id: string, stats: { maxErr: number; iters: number } | null) {
  setPeqStats((prev) => {
    const next = { ...prev };
    if (stats) next[id] = stats; else delete next[id];
    return next;
  });
}

// --- Helpers ---
export function formatFreq(hz: number): string {
  if (hz >= 1000) return (hz / 1000).toFixed(1) + "k";
  return Math.round(hz).toString();
}


/** Narrow [lo, hi] to the span where the target is above `threshold` (dB).
 *  Leaves the range as is when the target never rises above it. */
export function trimRangeByFloor(
  freq: number[], targetMag: number[], lo: number, hi: number, threshold: number,
): [number, number] {
  let first = -1, last = -1;
  for (let i = 0; i < freq.length; i++) {
    if (targetMag[i] > threshold) { if (first < 0) first = i; last = i; }
  }
  if (first < 0) return [lo, hi];
  return [Math.max(lo, freq[first]), Math.min(hi, freq[last])];
}

interface RangeInputs {
  meas: Measurement;
  target: BandState["target"];
  mode: PeqRangeMode;
  directLow: number;
  directHigh: number;
  floorDb: number;
  boostFloorDb: number;
  sampleRate: number;
}

/** Plain copies of everything the PEQ range depends on — read before any await. */
function snapshotRangeInputs(b: BandState): RangeInputs {
  return {
    meas: JSON.parse(JSON.stringify(b.measurement!)),
    target: JSON.parse(JSON.stringify(b.target)),
    mode: peqRangeMode(), directLow: peqDirectLow(), directHigh: peqDirectHigh(),
    floorDb: peqFloor(), boostFloorDb: peqBoostFloor(), sampleRate: exportSampleRate(),
  };
}

/** The frequency range the optimizer works in — the one rule for the fit and
 *  for the «Hz» row (b141.91). Auto: 20 Hz – 0.95·Nyquist, trimmed by «Порог
 *  dB» only (the old fLow/8 cap and the ×8 shown beside the button disagreed
 *  with each other and with the fit). Direct: the user's range, then the floor. */
async function resolveRange(inp: RangeInputs) {
  const { meas } = inp;
  // adaptive passband for refOffset (the one shared rule)
  const [refLow, refHigh] = passbandRange(inp.target?.high_pass?.freq_hz, inp.target?.low_pass?.freq_hz);
  let refOffset = 0, count = 0;
  for (let i = 0; i < meas.freq.length; i++) {
    if (meas.freq[i] >= refLow && meas.freq[i] <= refHigh) {
      refOffset += meas.magnitude[i]; count++;
    }
  }
  refOffset = count > 0 ? refOffset / count : 0;
  const targetCurve = inp.target;
  targetCurve.reference_level_db += refOffset;
  const targetResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", {
    target: targetCurve, freq: meas.freq,
  });
  // Cap at Nyquist·0.95 of the export rate: a PEQ above Nyquist is not
  // realisable by the biquad it will be exported as.
  const fMax = fMaxForRate(inp.sampleRate);
  let range: [number, number] = inp.mode === "direct"
    ? [Math.min(inp.directLow, inp.directHigh), Math.min(Math.max(inp.directLow, inp.directHigh), fMax)]
    : [20, fMax];
  // b141.84: the floor trims BOTH modes — in direct mode it used to be
  // ignored, and a woofer LR2 @ 250 Hz got two of its six PEQ at 3.9/5.1 kHz,
  // where the target sits ~95 dB down.
  if (inp.floorDb > 0) {
    range = trimRangeByFloor(meas.freq, targetResp.magnitude, range[0], range[1],
      targetCurve.reference_level_db - inp.floorDb);
  }
  // b141.92: where boosts are allowed — the same trim at «Подъём до».
  const boostRange: [number, number] | null = inp.boostFloorDb > 0
    ? trimRangeByFloor(meas.freq, targetResp.magnitude, range[0], range[1],
      targetCurve.reference_level_db - inp.boostFloorDb)
    : null;
  return { targetCurve, targetResp, range, boostRange };
}

/** The optimizer's range for a band with a measurement (the «Hz» row). */
export async function peqRangeFor(b: BandState): Promise<[number, number]> {
  return (await resolveRange(snapshotRangeInputs(b))).range;
}

// --- Internal: optimize a specific band ---
// Disabled PEQ bands are kept in the table as they are; the optimizer re-fits
// only the remaining enabled slots and does not count the disabled ones.
async function optimizeBand(b: BandState): Promise<{ result: PeqResult; frozenBands: PeqBand[] }> {
  // b141.5 (audit): pre-read EVERY store-proxy field into plain objects
  // BEFORE the first await. Proxy reads after an await observe concurrent
  // user edits mid-run (mixed pre/post state, lost updates) — project rule
  // "pre-read в plain objects ДО async loop".
  const meas: Measurement = JSON.parse(JSON.stringify(b.measurement!));
  const peqBandsSnap: PeqBand[] = JSON.parse(JSON.stringify(b.peqBands ?? []));
  const exclusionZonesSnap: ExclusionZone[] = JSON.parse(JSON.stringify(b.exclusionZones ?? []));
  const fLow = b.target?.high_pass?.freq_hz ?? 20;
  const fHigh = b.target?.low_pass?.freq_hz ?? F_MAX_REF;
  // b141.67 (audit stage 2 B6): every setting is read BEFORE the first await —
  // a Standard/Hybrid switch during the run used to mix both modes in one fit.
  const isHybrid = exportHybridPhase();
  const smoothFit = peqSmoothFit(), sampleRate = exportSampleRate();
  // b141.94: no auto filter rings past half the file (the «мс» column's red).
  const maxRingMs = exportTaps() / 2 / sampleRate * 1000;
  const bandBudget = maxBands(), tol = tolerance(), gainReg = gainRegularization();
  const rangeIn = snapshotRangeInputs(b);
  const { targetResp, range: [peqLow, peqHigh], boostRange } = await resolveRange(rangeIn);

  // Disabled bands are kept in the table untouched, but they do NOT play:
  // the plot, the Σ and the FIR all skip them (evaluate.ts filters on
  // `enabled`). b141.46 (audit 2026-10-01 H2): they used to be baked into the
  // measurement as "frozen", so the optimizer fitted around a correction the
  // exported filter never contained — +6 dB left over where it reported 0.
  const frozenBands = peqBandsSnap.filter((p) => !p.enabled);
  const measMag = meas.magnitude;
  const activeBandBudget = Math.max(1, bandBudget - frozenBands.length);
  const config: PeqConfig = {
    max_bands: activeBandBudget,
    tolerance_db: tol,
    peak_bias: isHybrid ? 1.0 : 1.5,
    max_boost_db: isHybrid ? 60.0 : 6.0,
    max_cut_db: isHybrid ? 60.0 : 18.0,
    freq_range: [peqLow, peqHigh],
    boost_range: boostRange,
    max_ring_ms: maxRingMs,
    hybrid: isHybrid,
    gain_regularization: gainReg,
    // b141.5 (audit): optimize at the rate the biquads will actually run at.
    sample_rate: sampleRate,
  };
  const result = await invoke<PeqResult>("auto_peq_lma", {
    freq: meas.freq,
    measurementMag: measMag,
    targetMag: targetResp.magnitude,
    config,
    hpFreq: fLow,
    lpFreq: fHigh,
    exclusionZones: exclusionZonesSnap.length > 0 ? exclusionZonesSnap : null,
    smoothFit,
  });
  return { result, frozenBands };
}

/** Merge kept (disabled) bands with newly optimized bands, sorted by freq */
function mergeBands(frozen: PeqBand[], optimized: PeqBand[]): PeqBand[] {
  const all = [...frozen, ...optimized];
  all.sort((a, b) => a.freq_hz - b.freq_hz);
  return all;
}

// Snapshot of target/exclusion taken at successful optimization. Used by
// peqStale to detect divergence later.
export function captureOptimizedTarget(b: BandState): PeqOptimizedTarget {
  // b141.6 (audit): clone via the single source of truth, not an ad-hoc
  // spread — this snapshot is serialized into .pfproj (peq_optimized_target).
  return {
    high_pass: cloneFilterConfig(b.target.high_pass ?? null),
    low_pass: cloneFilterConfig(b.target.low_pass ?? null),
    exclusion_zones: JSON.parse(JSON.stringify(b.exclusionZones)),
    // b141.22 (audit): the rate the biquads were fitted at — see PeqOptimizedTarget.
    sample_rate: exportSampleRate(),
    shaping: currentShaping(b),
  };
}

function currentShaping(b: BandState): NonNullable<PeqOptimizedTarget["shaping"]> {
  return {
    tilt_db_per_octave: b.target.tilt_db_per_octave ?? 0,
    tilt_ref_freq: b.target.tilt_ref_freq ?? 1000,
    low_shelf: b.target.low_shelf ? JSON.parse(JSON.stringify(b.target.low_shelf)) : null,
    high_shelf: b.target.high_shelf ? JSON.parse(JSON.stringify(b.target.high_shelf)) : null,
  };
}

/** Why the PEQ is stale (for the banner), or null when it is not. */
export function peqStaleReason(b: BandState): string | null {
  if (!b.peqBands || b.peqBands.length === 0 || !b.peqOptimizedTarget) return null;
  const snap = b.peqOptimizedTarget;
  if (!filterEquals(b.target.high_pass, snap.high_pass) || !filterEquals(b.target.low_pass, snap.low_pass)) {
    return "кроссовер изменён";
  }
  if (snap.shaping && JSON.stringify(snap.shaping) !== JSON.stringify(currentShaping(b))) {
    return "наклон или полки цели изменены";
  }
  if (!exclusionZonesEquals(b.exclusionZones, snap.exclusion_zones)) return "зоны исключения изменены";
  // b141.22: a fit made at another export rate no longer describes the biquads
  // that will ship. Snapshots from before b141.22 carry no rate — "unknown".
  if (snap.sample_rate !== undefined && snap.sample_rate !== exportSampleRate()) {
    return "частота дискретизации экспорта изменена";
  }
  return null;
}

function filterEquals(a: FilterConfig | null, b: FilterConfig | null): boolean {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.filter_type === b.filter_type
    && a.order === b.order
    && a.freq_hz === b.freq_hz
    && a.shape === b.shape
    && a.q === b.q
    // b141.22 (audit): both change the target the PEQ was fitted against —
    // linear_phase flips the route and the phase the correction sees,
    // subsonic_protect adds an LF roll-off. Omitting them let the fit go
    // stale in silence.
    && a.linear_phase === b.linear_phase
    && (a.subsonic_protect ?? null) === (b.subsonic_protect ?? null);
}

function exclusionZonesEquals(a: ExclusionZone[], b: ExclusionZone[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].startHz !== b[i].startHz) return false;
    if (a[i].endHz !== b[i].endHz) return false;
  }
  return true;
}

/** True iff peqBands exist, an optimization snapshot exists, and target or
 *  exclusion zones diverge from the snapshot. Pure read — safe inside Solid
 *  reactive contexts. */
export function peqStale(b: BandState): boolean {
  return peqStaleReason(b) !== null;
}

// --- Main actions ---
export async function handleOptimizePeq() {
  const b = activeBand();
  if (!b || !b.measurement || computing()) return;
  pushHistory("Optimize PEQ");
  // Snapshot the target the optimizer is about to consume — concurrent edits
  // during the await must not poison the staleness check.
  const optimizedTarget = captureOptimizedTarget(b);
  setComputing(true);
  setPeqError(null);
  try {
    const { result, frozenBands } = await optimizeBand(b);
    setBandPeqBands(b.id, mergeBands(frozenBands, result.bands));
    setBandPeqOptimizedTarget(b.id, optimizedTarget);
    setBandStats(b.id, { maxErr: result.max_error_db, iters: result.iterations });
    setSelectedPeqIdx(null);
    showToast(
      `PEQ оптимизирован · фильтров: ${result.bands.length} · макс. ошибка ${result.max_error_db.toFixed(2)} dB`,
    );
  } catch (e) {
    setPeqError(String(e));
    showToast(`Ошибка оптимизации PEQ: ${String(e)}`, "warn");
  } finally {
    setComputing(false);
  }
}

/** Optimize PEQ for ALL bands that have a measurement */
export async function handleOptimizeAll() {
  const bands = appState.bands;
  const eligible = bands.filter((b) => b.measurement);
  if (eligible.length === 0 || computing()) return;
  pushHistory("Optimize all");
  setComputing(true);
  setPeqError(null);
  try {
    // 1. Compute ALL results first (no store writes during loop). Snapshot
    //    each band's target BEFORE its await so a concurrent target edit
    //    cannot retroactively make the post-optimize state look "fresh".
    // b141.67 (audit stage 2 P6): bands are independent — fit them in
    // parallel (async commands run on the multi-thread runtime).
    const results = await Promise.all(eligible.map(async (b) => {
      const target = captureOptimizedTarget(b);
      const { result, frozenBands } = await optimizeBand(b);
      return {
        id: b.id,
        peqBands: mergeBands(frozenBands, result.bands),
        maxErr: result.max_error_db,
        iters: result.iterations,
        target,
      };
    }));
    // 2. Apply all at once → single reactive update
    batch(() => {
      for (const r of results) {
        setBandPeqBands(r.id, r.peqBands);
        setBandPeqOptimizedTarget(r.id, r.target);
      }
      for (const r of results) setBandStats(r.id, { maxErr: r.maxErr, iters: r.iters });
      setSelectedPeqIdx(null);
    });
    showToast(
      `Оптимизировано бэндов: ${results.length} · макс. ошибка ${Math.max(...results.map(r => r.maxErr)).toFixed(2)} dB`,
    );
  } catch (e) {
    setPeqError(String(e));
    showToast(`Ошибка оптимизации: ${String(e)}`, "warn");
  } finally {
    setComputing(false);
  }
}

export function handleClearPeq() {
  const b = activeBand();
  if (b) { clearBandPeqBands(b.id); setBandStats(b.id, null); }
  setSelectedPeqIdx(null);
}

// ---------------------------------------------------------------------------
// History hook registration: combines bands' light snapshot with PEQ params.
// ---------------------------------------------------------------------------

registerHistoryHooks(
  (label: string): HistoryEntry => {
    const part = _captureBandsLight();
    return {
      ...part,
      peqParams: {
        tolerance: tolerance(),
        maxBands: maxBands(),
        gainRegularization: gainRegularization(),
        peqFloor: peqFloor(),
        peqSmoothFit: peqSmoothFit(),
        peqBoostFloor: peqBoostFloor(),
        peqRangeMode: peqRangeMode(),
        peqDirectLow: peqDirectLow(),
        peqDirectHigh: peqDirectHigh(),
      },
      label,
      ts: Date.now(),
    };
  },
  (entry: HistoryEntry) => {
    setTolerance(entry.peqParams.tolerance);
    setMaxBands(entry.peqParams.maxBands);
    setGainRegularization(entry.peqParams.gainRegularization);
    setPeqFloor(entry.peqParams.peqFloor);
    setPeqSmoothFit(entry.peqParams.peqSmoothFit ?? true);
    setPeqBoostFloor(entry.peqParams.peqBoostFloor ?? 30);
    setPeqRangeMode(entry.peqParams.peqRangeMode);
    setPeqDirectLow(entry.peqParams.peqDirectLow);
    setPeqDirectHigh(entry.peqParams.peqDirectHigh);
    _applyBandsLight(entry);
  },
);
