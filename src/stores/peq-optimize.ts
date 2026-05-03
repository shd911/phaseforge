// ---------------------------------------------------------------------------
// PEQ Auto-Fit shared store — extracted from ControlPanel.tsx (b82.06)
// ---------------------------------------------------------------------------
import { createSignal, batch } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import type { PeqBand, PeqConfig, PeqResult, FilterConfig, ExclusionZone, PeqOptimizedTarget } from "../lib/types";
import {
  activeBand,
  appState,
  exportHybridPhase,
  setBandPeqBands,
  clearBandPeqBands,
  setBandPeqOptimizedTarget,
  setSelectedPeqIdx,
  _captureBandsLight,
  _applyBandsLight,
} from "./bands";
import type { BandState } from "./bands";
import { pushHistory, registerHistoryHooks, type HistoryEntry } from "./history";

// --- Signals ---
export const [tolerance, setTolerance] = createSignal(1.0);
export const [maxBands, setMaxBands] = createSignal(20);
export const [gainRegularization, setGainRegularization] = createSignal(0.0);
export const [peqFloor, setPeqFloor] = createSignal(60); // dB below reference — don't optimize below this
export type PeqRangeMode = "auto" | "direct";
export const [peqRangeMode, setPeqRangeMode] = createSignal<PeqRangeMode>("auto");
export const [peqDirectLow, setPeqDirectLow] = createSignal(20);
export const [peqDirectHigh, setPeqDirectHigh] = createSignal(20000);
export const [computing, setComputing] = createSignal(false);
export const [peqError, setPeqError] = createSignal<string | null>(null);
export const [maxErr, setMaxErr] = createSignal<number | null>(null);
export const [iters, setIters] = createSignal<number | null>(null);

// --- Helpers ---
export function crossoverRange(): [number, number] {
  const b = activeBand();
  const t = b?.target;
  const fLow = t?.high_pass?.freq_hz ?? 20;
  const fHigh = t?.low_pass?.freq_hz ?? 20000;
  return [fLow, fHigh];
}

export function formatFreq(hz: number): string {
  if (hz >= 1000) return (hz / 1000).toFixed(1) + "k";
  return Math.round(hz).toString();
}

export function peqRange(): [number, number] {
  const [lo, hi] = crossoverRange();
  return [Math.max(20, lo / 8), Math.min(20000, hi * 8)];
}

// --- Internal: optimize a specific band ---
// If some PEQ bands are disabled, they are "frozen": their correction is baked
// into the measurement and the optimizer re-fits only the remaining enabled slots.
async function optimizeBand(b: BandState): Promise<{ result: PeqResult; frozenBands: PeqBand[] }> {
  const meas = b.measurement!;
  const fLow = b.target?.high_pass?.freq_hz ?? 20;
  const fHigh = b.target?.low_pass?.freq_hz ?? 20000;
  // adaptive passband for refOffset (matches FrequencyPlot autoRef)
  const pbLow = Math.max(20, fLow * 1.5);
  const pbHigh = Math.min(20000, fHigh * 0.7);
  const refLow = pbLow < pbHigh ? pbLow : 200;
  const refHigh = pbLow < pbHigh ? pbHigh : 2000;
  let refOffset = 0, count = 0;
  for (let i = 0; i < meas.freq.length; i++) {
    if (meas.freq[i] >= refLow && meas.freq[i] <= refHigh) {
      refOffset += meas.magnitude[i]; count++;
    }
  }
  refOffset = count > 0 ? refOffset / count : 0;
  const targetCurve = JSON.parse(JSON.stringify(b.target));
  targetCurve.reference_level_db += refOffset;
  const targetResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", {
    target: targetCurve, freq: meas.freq,
  });

  // Separate frozen (disabled) bands from active ones
  const frozenBands = (b.peqBands ?? []).filter((p) => !p.enabled);
  let measMag = meas.magnitude;

  // If there are frozen bands, bake their correction into measurement
  if (frozenBands.length > 0) {
    const frozenCorrection = await invoke<number[]>("compute_peq_response", {
      freq: meas.freq, bands: frozenBands.map((fb) => ({ ...fb, enabled: true })),
    });
    measMag = meas.magnitude.map((v, i) => v + frozenCorrection[i]);
  }

  const isHybrid = exportHybridPhase();
  let peqLow: number;
  let peqHigh: number;

  if (peqRangeMode() === "direct") {
    // Direct mode: user-specified range, ignore floor and crossover
    peqLow = peqDirectLow();
    peqHigh = peqDirectHigh();
  } else {
    // Auto mode: derive from crossover + floor
    peqLow = isHybrid ? 20 : Math.max(20, fLow / 8);
    peqHigh = 20000;

    // Trim PEQ range by target floor: don't optimize where target is below threshold
    const floorDb = peqFloor();
    if (floorDb > 0) {
      const refLevel = targetCurve.reference_level_db;
      const threshold = refLevel - floorDb;
      for (let i = 0; i < meas.freq.length; i++) {
        if (targetResp.magnitude[i] > threshold) {
          peqLow = Math.max(peqLow, meas.freq[i]);
          break;
        }
      }
      for (let i = meas.freq.length - 1; i >= 0; i--) {
        if (targetResp.magnitude[i] > threshold) {
          peqHigh = Math.min(peqHigh, meas.freq[i]);
          break;
        }
      }
    }
  }
  const activeBandBudget = Math.max(1, maxBands() - frozenBands.length);
  const config: PeqConfig = {
    max_bands: activeBandBudget,
    tolerance_db: tolerance(),
    peak_bias: isHybrid ? 1.0 : 1.5,
    max_boost_db: isHybrid ? 60.0 : 6.0,
    max_cut_db: isHybrid ? 60.0 : 18.0,
    freq_range: [peqLow, peqHigh],
    hybrid: isHybrid,
    gain_regularization: gainRegularization(),
  };
  const result = await invoke<PeqResult>("auto_peq_lma", {
    freq: meas.freq,
    measurementMag: measMag,
    targetMag: targetResp.magnitude,
    config,
    hpFreq: fLow,
    lpFreq: fHigh,
    exclusionZones: b.exclusionZones.length > 0 ? b.exclusionZones : null,
  });
  return { result, frozenBands };
}

/** Merge frozen (disabled) bands with newly optimized bands, sorted by freq */
function mergeBands(frozen: PeqBand[], optimized: PeqBand[]): PeqBand[] {
  const all = [...frozen, ...optimized];
  all.sort((a, b) => a.freq_hz - b.freq_hz);
  return all;
}

// Snapshot of target/exclusion taken at successful optimization. Used by
// peqStale to detect divergence later.
export function captureOptimizedTarget(b: BandState): PeqOptimizedTarget {
  return {
    high_pass: b.target.high_pass ? { ...b.target.high_pass } : null,
    low_pass: b.target.low_pass ? { ...b.target.low_pass } : null,
    exclusion_zones: JSON.parse(JSON.stringify(b.exclusionZones)),
  };
}

function filterEquals(a: FilterConfig | null, b: FilterConfig | null): boolean {
  if (a === null && b === null) return true;
  if (a === null || b === null) return false;
  return a.filter_type === b.filter_type
    && a.order === b.order
    && a.freq_hz === b.freq_hz
    && a.shape === b.shape
    && a.q === b.q;
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
  if (!b.peqBands || b.peqBands.length === 0) return false;
  if (!b.peqOptimizedTarget) return false;
  const snap = b.peqOptimizedTarget;
  if (!filterEquals(b.target.high_pass, snap.high_pass)) return true;
  if (!filterEquals(b.target.low_pass, snap.low_pass)) return true;
  if (!exclusionZonesEquals(b.exclusionZones, snap.exclusion_zones)) return true;
  return false;
}

// --- Main actions ---
export async function handleOptimizePeq() {
  const b = activeBand();
  if (!b || !b.measurement) return;
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
    setMaxErr(result.max_error_db);
    setIters(result.iterations);
    setSelectedPeqIdx(null);
  } catch (e) {
    setPeqError(String(e));
  } finally {
    setComputing(false);
  }
}

/** Optimize PEQ for ALL bands that have a measurement */
export async function handleOptimizeAll() {
  const bands = appState.bands;
  const eligible = bands.filter((b) => b.measurement);
  if (eligible.length === 0) return;
  pushHistory("Optimize all");
  setComputing(true);
  setPeqError(null);
  try {
    // 1. Compute ALL results first (no store writes during loop). Snapshot
    //    each band's target BEFORE its await so a concurrent target edit
    //    cannot retroactively make the post-optimize state look "fresh".
    const results: { id: string; peqBands: PeqBand[]; maxErr: number; iters: number; target: PeqOptimizedTarget }[] = [];
    for (const b of eligible) {
      const target = captureOptimizedTarget(b);
      const { result, frozenBands } = await optimizeBand(b);
      results.push({
        id: b.id,
        peqBands: mergeBands(frozenBands, result.bands),
        maxErr: result.max_error_db,
        iters: result.iterations,
        target,
      });
    }
    // 2. Apply all at once → single reactive update
    batch(() => {
      for (const r of results) {
        setBandPeqBands(r.id, r.peqBands);
        setBandPeqOptimizedTarget(r.id, r.target);
      }
      setMaxErr(Math.max(...results.map(r => r.maxErr)));
      setIters(results.reduce((sum, r) => sum + r.iters, 0));
      setSelectedPeqIdx(null);
    });
  } catch (e) {
    setPeqError(String(e));
  } finally {
    setComputing(false);
  }
}

export function handleClearPeq() {
  const b = activeBand();
  if (b) clearBandPeqBands(b.id);
  setMaxErr(null);
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
    setPeqRangeMode(entry.peqParams.peqRangeMode);
    setPeqDirectLow(entry.peqParams.peqDirectLow);
    setPeqDirectHigh(entry.peqParams.peqDirectHigh);
    _applyBandsLight(entry);
  },
);
