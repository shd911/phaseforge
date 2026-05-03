// Band evaluation helpers — extracted from FrequencyPlot.tsx
// Pure async functions with no component state dependencies.

import { invoke } from "@tauri-apps/api/core";
import type { Measurement, TargetResponse, FilterConfig } from "./types";
import type { BandState, SmoothingMode } from "../stores/bands";
import { smoothingConfig, isGaussianMinPhase, gaussianFilterMagDb, subsonicMagDb } from "./plot-helpers";

/** True when the HP carries an active subsonic_protect that must contribute
 *  min-phase even if the Gaussian itself is linear-phase. */
export function hasActiveSubsonicProtect(hp: FilterConfig | null | undefined): boolean {
  return !!hp
    && hp.filter_type === "Gaussian"
    && hp.subsonic_protect === true
    && hp.freq_hz > 40;
}

// ---------------------------------------------------------------------------
// Gaussian per-filter Hilbert: compute min-phase for each Gaussian filter
// individually, and ADD to existing phase (don't replace the whole phase)
// ---------------------------------------------------------------------------
export async function addGaussianMinPhase(
  freq: number[],
  phase: number[],
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
): Promise<number[]> {
  let result = phase;
  if (isGaussianMinPhase(hp)) {
    let hpMag = gaussianFilterMagDb(freq, hp!, false);
    if (hasActiveSubsonicProtect(hp)) {
      const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
      hpMag = hpMag.map((db, i) => db + subDb[i]);
    }
    const hpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: hpMag });
    result = result.map((v, i) => v + hpPh[i]);
  } else if (hasActiveSubsonicProtect(hp) && hp!.linear_phase === true) {
    // b138.4: linear-phase Gaussian still needs min-phase contribution from
    // the subsonic filter alone — Hilbert from subsonic-only magnitude.
    const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
    const subPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: subDb });
    result = result.map((v, i) => v + subPh[i]);
  }
  if (isGaussianMinPhase(lp)) {
    const lpMag = gaussianFilterMagDb(freq, lp!, true);
    const lpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: lpMag });
    result = result.map((v, i) => v + lpPh[i]);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Helper: apply smoothing to a measurement
// ---------------------------------------------------------------------------
export async function applySmoothing(m: Measurement, mode: SmoothingMode): Promise<Measurement> {
  if (mode === "off") return m;
  const config = smoothingConfig(mode);
  const smoothed = await invoke<number[]>("get_smoothed", {
    freq: m.freq, magnitude: m.magnitude, config,
  });
  return { ...m, magnitude: smoothed };
}

// ---------------------------------------------------------------------------
// Helper: evaluate a single band (target + measurement + Gaussian min-phase)
// ---------------------------------------------------------------------------
export async function evaluateBand(band: BandState): Promise<{
  measurement: Measurement | null;
  targetMag: number[] | null;
  targetPhase: number[] | null;
  freq: number[] | null;
}> {
  const targetCurve = JSON.parse(JSON.stringify(band.target));
  let measurement: Measurement | null = null;

  if (band.measurement) {
    const raw: Measurement = JSON.parse(JSON.stringify(band.measurement));
    const mode = band.settings?.smoothing ?? "off";
    measurement = await applySmoothing(raw, mode);
  }

  let targetMag: number[] | null = null;
  let targetPhase: number[] | null = null;
  let freq: number[] | null = measurement?.freq ?? null;

  if (band.targetEnabled) {
    if (measurement) {
      // Compute autoRef using the same adaptive passband as zoomCenter
      // so target and normalization are aligned in narrow-band configurations
      const hpFreq = band.target.high_pass?.freq_hz ?? 20;
      const lpFreq = band.target.low_pass?.freq_hz ?? 20000;
      const pbLow = Math.max(20, hpFreq * 1.5);
      const pbHigh = Math.min(20000, lpFreq * 0.7);
      const refLow = pbLow < pbHigh ? pbLow : 200;
      const refHigh = pbLow < pbHigh ? pbHigh : 2000;

      let sum = 0, n = 0;
      for (let i = 0; i < measurement.freq.length; i++) {
        if (measurement.freq[i] >= refLow && measurement.freq[i] <= refHigh) {
          sum += measurement.magnitude[i]; n++;
        }
      }
      const autoRef = n > 0 ? sum / n : 0;
      const curveWithRef = { ...targetCurve, reference_level_db: targetCurve.reference_level_db + autoRef };

      const response = await invoke<TargetResponse>("evaluate_target", {
        target: curveWithRef, freq: measurement.freq,
      });
      targetMag = response.magnitude;
      targetPhase = response.phase;
    } else {
      const [standaloneFreq, response] = await invoke<[number[], TargetResponse]>(
        "evaluate_target_standalone", { target: targetCurve }
      );
      freq = standaloneFreq;
      targetMag = response.magnitude;
      targetPhase = response.phase;
    }
  }

  // Gaussian min-phase: compute Hilbert per-filter (not blanket on full magnitude).
  // b138.4: also fire when HP is linear-phase Gaussian + subsonic — the
  // subsonic part stays min-phase even if the user asked for linear Gaussian.
  const needsLinearPhaseSubsonic = band.target.high_pass?.linear_phase === true
    && hasActiveSubsonicProtect(band.target.high_pass);
  if (targetPhase && freq && (
    isGaussianMinPhase(band.target.high_pass)
    || isGaussianMinPhase(band.target.low_pass)
    || needsLinearPhaseSubsonic
  )) {
    targetPhase = await addGaussianMinPhase(freq, targetPhase, band.target.high_pass, band.target.low_pass);
  }

  return { measurement, targetMag, targetPhase, freq };
}
