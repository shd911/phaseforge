// ---------------------------------------------------------------------------
// PEQ Auto-Fit shared store — extracted from ControlPanel.tsx (b82.06)
// ---------------------------------------------------------------------------
import { createSignal } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import type { PeqConfig, PeqResult } from "../lib/types";
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
async function optimizeBand(b: BandState): Promise<PeqResult> {
  const meas = b.measurement!;
  const fLow = b.target?.high_pass?.freq_hz ?? 20;
  const fHigh = b.target?.low_pass?.freq_hz ?? 20000;
  let refOffset = 0, count = 0;
  for (let i = 0; i < meas.freq.length; i++) {
    if (meas.freq[i] >= 200 && meas.freq[i] <= 2000) {
      refOffset += meas.magnitude[i]; count++;
    }
  }
  refOffset = count > 0 ? refOffset / count : 0;
  const targetCurve = JSON.parse(JSON.stringify(b.target));
  targetCurve.reference_level_db += refOffset;
  const targetResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", {
    target: targetCurve, freq: meas.freq,
  });
  const isHybrid = exportHybridPhase();
  const peqLow = isHybrid ? 20 : Math.max(20, fLow / 8);
  const peqHigh = 20000;
  const config: PeqConfig = {
    max_bands: maxBands(),
    tolerance_db: tolerance(),
    peak_bias: isHybrid ? 1.0 : 1.5,
    max_boost_db: isHybrid ? 60.0 : 6.0,
    max_cut_db: isHybrid ? 60.0 : 18.0,
    freq_range: [peqLow, peqHigh],
    hybrid: isHybrid,
  };
  return invoke<PeqResult>("auto_peq_lma", {
    freq: meas.freq,
    measurementMag: meas.magnitude,
    targetMag: targetResp.magnitude,
    config,
    hpFreq: fLow,
    lpFreq: fHigh,
  });
}

// --- Main actions ---
export async function handleOptimizePeq() {
  const b = activeBand();
  if (!b || !b.measurement) return;
  setComputing(true);
  setPeqError(null);
  try {
    const result = await optimizeBand(b);
    setBandPeqBands(b.id, result.bands);
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
  let lastErr: number | null = null;
  let lastIters: number | null = null;
  try {
    for (const b of eligible) {
      const result = await optimizeBand(b);
      setBandPeqBands(b.id, result.bands);
      lastErr = result.max_error_db;
      lastIters = result.iterations;
    }
    setMaxErr(lastErr);
    setIters(lastIters);
    setSelectedPeqIdx(null);
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
