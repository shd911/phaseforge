// ---------------------------------------------------------------------------
// PEQ Auto-Fit shared store — extracted from ControlPanel.tsx (b82.06)
// ---------------------------------------------------------------------------
import { createSignal, batch } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import type { PeqBand, PeqConfig, PeqResult } from "../lib/types";
import {
  activeBand,
  appState,
  exportHybridPhase,
  setBandPeqBands,
  clearBandPeqBands,
  setSelectedPeqIdx,
} from "./bands";
import type { BandState } from "./bands";

// --- Signals ---
export const [tolerance, setTolerance] = createSignal(1.0);
export const [maxBands, setMaxBands] = createSignal(20);
export const [gainRegularization, setGainRegularization] = createSignal(0.0);
export const [peqFloor, setPeqFloor] = createSignal(60); // dB below reference — don't optimize below this
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
  let peqLow = isHybrid ? 20 : Math.max(20, fLow / 8);
  let peqHigh = 20000;

  // Trim PEQ range by target floor: don't optimize where target is below threshold
  const floorDb = peqFloor();
  if (floorDb > 0) {
    const refLevel = targetCurve.reference_level_db;
    const threshold = refLevel - floorDb;
    // Scan from low: find first freq where target > threshold
    for (let i = 0; i < meas.freq.length; i++) {
      if (targetResp.magnitude[i] > threshold) {
        peqLow = Math.max(peqLow, meas.freq[i]);
        break;
      }
    }
    // Scan from high: find last freq where target > threshold
    for (let i = meas.freq.length - 1; i >= 0; i--) {
      if (targetResp.magnitude[i] > threshold) {
        peqHigh = Math.min(peqHigh, meas.freq[i]);
        break;
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

// --- Main actions ---
export async function handleOptimizePeq() {
  const b = activeBand();
  if (!b || !b.measurement) return;
  setComputing(true);
  setPeqError(null);
  try {
    const { result, frozenBands } = await optimizeBand(b);
    setBandPeqBands(b.id, mergeBands(frozenBands, result.bands));
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
  setComputing(true);
  setPeqError(null);
  try {
    // 1. Compute ALL results first (no store writes during loop)
    const results: { id: string; peqBands: PeqBand[]; maxErr: number; iters: number }[] = [];
    for (const b of eligible) {
      const { result, frozenBands } = await optimizeBand(b);
      results.push({ id: b.id, peqBands: mergeBands(frozenBands, result.bands), maxErr: result.max_error_db, iters: result.iterations });
    }
    // 2. Apply all at once → single reactive update
    batch(() => {
      for (const r of results) setBandPeqBands(r.id, r.peqBands);
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
