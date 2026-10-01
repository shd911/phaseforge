/**
 * Auto-align delays for multi-band crossover systems.
 *
 * Algorithm: HF→LF sequential optimization.
 * The highest-frequency band (tweeter) is the reference (delay = 0).
 * Each subsequent lower band is adjusted to maximize coherent sum
 * amplitude at the crossover with its upper neighbour.
 *
 * If a lower band needs negative delay (should arrive earlier),
 * the absolute value is propagated as positive delay to all
 * higher bands, keeping all delays >= 0.
 */

import type { BandState } from "../stores/bands";
import { evaluateSum } from "./band-evaluator/sum";
import { alignmentPhaseDeg } from "./types";

/** Result of auto-align: map from bandId → delay in seconds */
export interface AlignResult {
  delays: Record<string, number>;
}

/** One band as the alignment sees it: its corrected curve on the common
 *  grid (exactly what Σ Corrected sums), its polarity, its crossover. */
export interface AlignBand {
  mag: number[];
  ph: number[];
  sign: 1 | -1;
  hpHz: number | null;
  lpHz: number | null;
}

/**
 * Compute optimal alignment delays for all bands.
 *
 * b141.47 (audit 2026-10-01 H7/H8): the band responses come from
 * `evaluateSum` — the same corrected curves the Σ draws (min-phase
 * Gaussian/subsonic, level match, extension) — and the polarity sign is
 * applied. The old private pipeline (measurement + PEQ + cross-section)
 * gave Gaussian a zero phase and ignored INV, so with an inverted tweeter
 * it picked 0.47 ms and dropped the Σ to −20.7 dB at the crossover.
 *
 * @param bands - bands with measurement phase; store bands are fine
 *                (evaluateSum snapshots them)
 * @param sampleRate - realization sample rate (= export sample rate)
 * @returns delays in seconds per band id
 */
export async function computeAutoAlign(bands: BandState[], sampleRate = 48000): Promise<AlignResult> {
  const validBands = bands.filter(
    b => b.measurement?.phase && b.measurement.phase.length > 0
  );

  if (validBands.length < 2) {
    const delays: Record<string, number> = {};
    for (const b of validBands) delays[b.id] = 0;
    return { delays };
  }

  // Sort bands HF→LF: highest HP frequency first (tweeter first)
  const sorted = [...validBands].sort((a, b) => {
    const aHP = a.target?.high_pass?.freq_hz ?? 0;
    const bHP = b.target?.high_pass?.freq_hz ?? 0;
    if (aHP !== bHP) return bHP - aHP;  // descending by HP freq
    return bands.indexOf(a) - bands.indexOf(b);
  });

  const sum = await evaluateSum(sorted, { sampleRate });
  const freq = sum.freq;
  const alignBands: AlignBand[] = sorted.map((b, i) => {
    const pc = sum.perBandCorrected[i];
    return {
      mag: pc ? pc.mag : freq.map(() => -200),
      ph: pc ? pc.phase : freq.map(() => 0),
      sign: b.inverted ? -1 : 1,
      hpHz: b.target?.high_pass?.freq_hz ?? null,
      lpHz: b.target?.low_pass?.freq_hz ?? null,
    };
  });

  const delays = alignDelays(freq, alignBands);
  const result: Record<string, number> = {};
  for (let i = 0; i < sorted.length; i++) {
    result[sorted[i].id] = Math.round(delays[i] * 1e6) / 1e6;
  }
  return { delays: result };
}

/**
 * Pure core: delays (s, all ≥ 0) for bands sorted HF→LF.
 * Each lower band is fitted to its upper neighbour at their crossover.
 */
export function alignDelays(freq: number[], bandData: AlignBand[]): number[] {
  // sorted[i] = higher freq, sorted[i+1] = lower freq
  const crossoverRegions: { optIdx: number; freqRange: [number, number] }[] = [];
  for (let i = 0; i < bandData.length - 1; i++) {
    const hpFreq = bandData[i].hpHz;
    const lpFreq = bandData[i + 1].lpHz;
    if (lpFreq && hpFreq) {
      const xoFreq = Math.sqrt(lpFreq * hpFreq); // geometric mean (log-scale center)
      crossoverRegions.push({ optIdx: i + 1, freqRange: [xoFreq / 1.4142, xoFreq * 1.4142] });
    }
  }

  // Initialize delays: HF band = 0 (reference)
  const delays = new Array(bandData.length).fill(0);

  // Optimize sequentially HF→LF
  for (const xo of crossoverRegions) {
    const bestDelay = optimizePairDelay(freq, bandData, delays, xo.optIdx, xo.freqRange);

    if (bestDelay >= 0) {
      // Positive delay: lower band needs more delay — just assign
      delays[xo.optIdx] = bestDelay;
    } else {
      // Negative delay: lower band should be EARLIER than upper bands
      // → propagate |bestDelay| to ALL already-processed bands (indices 0..optIdx-1)
      const shift = -bestDelay;
      for (let k = 0; k < xo.optIdx; k++) {
        delays[k] += shift;
      }
      delays[xo.optIdx] = 0;
    }
  }
  return delays;
}

/**
 * Optimize delay for band hiIdx to maximize coherent sum amplitude in the
 * crossover region (all other bands at their current delays).
 */
function optimizePairDelay(
  freq: number[],
  bandData: AlignBand[],
  delays: number[],
  hiIdx: number,
  freqRange: [number, number],
): number {
  const xoIndices: number[] = [];
  for (let j = 0; j < freq.length; j++) {
    if (freq[j] >= freqRange[0] && freq[j] <= freqRange[1]) xoIndices.push(j);
  }
  if (xoIndices.length === 0) return 0;

  // Cost function: negative mean amplitude in crossover region
  const cost = (delayHi: number): number => {
    let totalAmp = 0;
    for (const j of xoIndices) {
      let re = 0, im = 0;
      for (let b = 0; b < bandData.length; b++) {
        const d = b === hiIdx ? delayHi : delays[b];
        const amp = bandData[b].sign * Math.pow(10, bandData[b].mag[j] / 20);
        const phRad = (bandData[b].ph[j] + alignmentPhaseDeg(freq[j], d)) * Math.PI / 180;
        re += amp * Math.cos(phRad);
        im += amp * Math.sin(phRad);
      }
      totalAmp += Math.sqrt(re * re + im * im);
    }
    return -totalAmp / xoIndices.length;
  };

  // Adaptive sweep range: wider for low-frequency crossovers
  const xoCenterFreq = (freqRange[0] + freqRange[1]) / 2;
  const adaptiveMaxMs = xoCenterFreq < 200 ? 5.0 : xoCenterFreq < 500 ? 3.0 : 2.0;
  const scanRange = adaptiveMaxMs / 1000;

  const nSteps = 200;
  const step = (2 * scanRange) / nSteps;
  const ds: number[] = [], amps: number[] = [];
  for (let i = 0; i <= nSteps; i++) {
    const d = -scanRange + step * i;
    ds.push(d);
    amps.push(-cost(d));
  }
  let bestDelay = pickScanPeak(ds, amps);
  let bestCost = cost(bestDelay);

  // Golden-section refinement inside ±1 scan step of the best grid point.
  // (The old gradient step lr·grad was ~0.1 s and always rejected, so the
  // result never left the 20 µs scan grid.)
  const g = (Math.sqrt(5) - 1) / 2;
  let lo = Math.max(-scanRange, bestDelay - step);
  let hi = Math.min(scanRange, bestDelay + step);
  let x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
  let c1 = cost(x1), c2 = cost(x2);
  for (let iter = 0; iter < 40; iter++) {
    if (c1 < c2) { hi = x2; x2 = x1; c2 = c1; x1 = hi - g * (hi - lo); c1 = cost(x1); }
    else { lo = x1; x1 = x2; c1 = c2; x2 = lo + g * (hi - lo); c2 = cost(x2); }
  }
  const refined = (lo + hi) / 2;
  return cost(refined) <= bestCost ? refined : bestDelay;
}

/**
 * Pick the delay from a coherence scan. b141.60 (user report, VPV2 220 Hz
 * LR2): the coherence over the crossover octave has maxima one period of the
 * crossover apart; the global maximum sat ON the scan edge (−3.00 ms, still
 * rising past it) while an interior one (+1.35 ms) was 1 % lower — the edge
 * value put the upper bands 3 ms late and wrapped the Σ phase every 333 Hz.
 * An edge point is not a located optimum, so interior local maxima win;
 * among those within 0.5 % of the best, the smallest |delay| (the
 * period-consistent solution, not a skipped period). The scan edges are
 * used only when the scan has no interior maximum at all.
 */
export function pickScanPeak(ds: number[], amps: number[]): number {
  const n = ds.length;
  const interior: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (amps[i] >= amps[i - 1] && amps[i] >= amps[i + 1]) interior.push(i);
  }
  const pool = interior.length > 0 ? interior : [...Array(n).keys()];
  let best = pool[0];
  for (const i of pool) if (amps[i] > amps[best]) best = i;
  let pick = best;
  for (const i of pool) {
    if (amps[i] >= amps[best] * 0.995 && Math.abs(ds[i]) < Math.abs(ds[pick])) pick = i;
  }
  return ds[pick];
}
