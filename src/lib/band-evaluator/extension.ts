/**
 * b140.14.2 — Phase 4 slice 3: target-extension + ref-level helpers.
 *
 * Three helpers extracted from band-evaluator.ts:
 *
 *   - appendNoiseFloorTail  — extend a (freq, mag, phase) trio up to
 *                              ~Nyquist with explicit noise-floor bins,
 *                              avoiding the Rust FFT pipeline's constant
 *                              boundary-clamp shift.
 *   - autoRefLevel          — average magnitude in the band passband
 *                              (HP·1.5 .. LP·0.7), fallback [200, 2000].
 *   - computeExtension      — fill bins outside the source range with
 *                              target shape (magnitude) + Hilbert
 *                              reconstruction (phase) for smooth
 *                              continuation. Calls compute_minimum_phase
 *                              for phase reconstruction.
 */
import { invoke } from "@tauri-apps/api/core";
import type { FilterConfig } from "../types";

/** b140.5: extend a (freq, mag, phase) trio up to ~Nyquist with explicit
 *  noise-floor magnitude bins (phase = 0). Without this, the Rust FFT
 *  pipeline's linear interpolation onto FFT bins above the log grid's
 *  fMax does a constant-clamp on the last value (e.g. target_mag at
 *  22.8 kHz ≈ -85 dB on a 48 kHz LP=2 kHz). That "shelf" between the
 *  log grid's fMax and Nyquist shows up as a rolloff shift in the
 *  generated FIR. The explicit silent tail replaces that shelf. */
export function appendNoiseFloorTail(
  freq: number[],
  mag: number[],
  phase: number[],
  sampleRate: number,
  noiseFloorDb = -150,
  extraBins = 32,
): { freq: number[]; mag: number[]; phase: number[] } {
  const nyquist = sampleRate / 2;
  const fHi = freq[freq.length - 1];
  if (fHi >= nyquist * 0.999) return { freq, mag, phase };
  const fEnd = nyquist * 0.999;
  const out = { freq: [...freq], mag: [...mag], phase: [...phase] };
  for (let i = 1; i <= extraBins; i++) {
    const t = i / extraBins;
    out.freq.push(fHi * Math.pow(fEnd / fHi, t));
    out.mag.push(noiseFloorDb);
    out.phase.push(0);
  }
  return out;
}

/** Average magnitude in the band passband (HP·1.5 .. LP·0.7),
 *  fallback [200, 2000] when the passband is inverted. */
export function autoRefLevel(
  freq: number[],
  magnitude: number[],
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
): number {
  const hpFreq = hp?.freq_hz ?? 20;
  const lpFreq = lp?.freq_hz ?? 20000;
  const pbLow = Math.max(20, hpFreq * 1.5);
  const pbHigh = Math.min(20000, lpFreq * 0.7);
  const refLow = pbLow < pbHigh ? pbLow : 200;
  const refHigh = pbLow < pbHigh ? pbHigh : 2000;
  let sum = 0, n = 0;
  for (let i = 0; i < freq.length; i++) {
    if (freq[i] >= refLow && freq[i] <= refHigh) { sum += magnitude[i]; n++; }
  }
  return n > 0 ? sum / n : 0;
}

/** b140.3.1.5: extension via target + Hilbert. For each bin of
 *  dstFreq outside the source range, fill magnitude from the target
 *  shape (with a boundary mag offset) and phase from Hilbert
 *  reconstruction (with separate low/high boundary phase offsets).
 *  Inside native, values are linearly interpolated from src; outside,
 *  they follow target+offset / Hilbert+offset for smooth continuation. */
export async function computeExtension(
  srcFreq: number[],
  srcMag: number[],
  srcPhase: number[] | null,
  dstFreq: number[],
  dstTargetMag: number[],
): Promise<{ mag: number[]; phase: number[] | null }> {
  const n = dstFreq.length;
  const fLo = srcFreq[0], fHi = srcFreq[srcFreq.length - 1];

  const nativeMag = new Array<number>(n);
  const nativePhase = srcPhase ? new Array<number>(n) : null;
  const inNative = new Array<boolean>(n);
  for (let k = 0; k < n; k++) {
    const f = dstFreq[k];
    if (f < fLo || f > fHi) {
      inNative[k] = false;
      nativeMag[k] = NaN;
      if (nativePhase) nativePhase[k] = NaN;
      continue;
    }
    inNative[k] = true;
    let lo = 0, hi = srcFreq.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (srcFreq[mid] <= f) lo = mid; else hi = mid;
    }
    const dt = srcFreq[hi] - srcFreq[lo];
    const frac = dt > 0 ? (f - srcFreq[lo]) / dt : 0;
    nativeMag[k] = srcMag[lo] + frac * (srcMag[hi] - srcMag[lo]);
    if (nativePhase && srcPhase) {
      nativePhase[k] = srcPhase[lo] + frac * (srcPhase[hi] - srcPhase[lo]);
    }
  }

  let idxLo = -1, idxHi = -1;
  for (let i = 0; i < n; i++) if (inNative[i]) { idxLo = i; break; }
  for (let i = n - 1; i >= 0; i--) if (inNative[i]) { idxHi = i; break; }
  if (idxLo < 0 || idxHi < 0) {
    return { mag: nativeMag.map(v => isFinite(v) ? v : -200), phase: nativePhase };
  }

  const magOffsetLo = nativeMag[idxLo] - dstTargetMag[idxLo];
  const magOffsetHi = nativeMag[idxHi] - dstTargetMag[idxHi];
  const extMag = new Array<number>(n);
  for (let k = 0; k < n; k++) {
    if (inNative[k]) extMag[k] = nativeMag[k];
    else if (k < idxLo) extMag[k] = dstTargetMag[k] + magOffsetLo;
    else extMag[k] = dstTargetMag[k] + magOffsetHi;
  }

  let extPhase: number[] | null = null;
  if (nativePhase) {
    const recon = await invoke<number[]>("compute_minimum_phase", {
      freq: dstFreq, magnitude: extMag,
    });
    const phOffsetLo = nativePhase[idxLo] - recon[idxLo];
    const phOffsetHi = nativePhase[idxHi] - recon[idxHi];
    extPhase = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      if (inNative[k]) extPhase[k] = nativePhase[k];
      else if (k < idxLo) extPhase[k] = recon[k] + phOffsetLo;
      else extPhase[k] = recon[k] + phOffsetHi;
    }
  }

  return { mag: extMag, phase: extPhase };
}
