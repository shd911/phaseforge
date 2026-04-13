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

import { invoke } from "@tauri-apps/api/core";
import type { BandState } from "../stores/bands";

/** Result of auto-align: map from bandId → delay in seconds */
export interface AlignResult {
  delays: Record<string, number>;
}

/** Linear interpolation on log-frequency scale. Returns NaN outside source range. */
function interpOnGrid(srcFreq: number[], srcData: number[], dstFreq: number[]): number[] {
  return dstFreq.map(f => {
    if (f < srcFreq[0] || f > srcFreq[srcFreq.length - 1]) return NaN;
    let lo = 0, hi = srcFreq.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (srcFreq[mid] <= f) lo = mid; else hi = mid;
    }
    const dt = srcFreq[hi] - srcFreq[lo];
    const frac = dt > 0 ? (f - srcFreq[lo]) / dt : 0;
    return srcData[lo] + frac * (srcData[hi] - srcData[lo]);
  });
}

/**
 * Compute optimal alignment delays for all bands.
 *
 * @param bands - array of BandState (must have measurement + phase + target with crossovers)
 * @returns delays in seconds per band id
 */
export async function computeAutoAlign(bands: BandState[]): Promise<AlignResult> {
  // Filter bands with measurement + phase data
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

  // Use the frequency grid with widest coverage (lowest start freq, most points)
  let refBand = sorted[0];
  for (const b of sorted) {
    if (b.measurement!.freq.length > refBand.measurement!.freq.length) {
      refBand = b;
    } else if (b.measurement!.freq.length === refBand.measurement!.freq.length
      && b.measurement!.freq[0] < refBand.measurement!.freq[0]) {
      refBand = b;
    }
  }
  const freq = [...refBand.measurement!.freq];
  const nPts = freq.length;
  console.log(`[auto-align] using freq grid from "${refBand.name}": ${nPts} points, ${freq[0].toFixed(1)}-${freq[freq.length-1].toFixed(1)} Hz`);

  // Get corrected magnitude + phase for each band (measurement + PEQ + crossover)
  const bandDataPromises = sorted.map(async (b) => {
    let mag = [...b.measurement!.magnitude];
    let ph = [...b.measurement!.phase!];

    // Apply PEQ
    const peqBands = b.peqBands?.filter(p => p.enabled) ?? [];
    if (peqBands.length > 0) {
      const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq: [...b.measurement!.freq],
        bands: JSON.parse(JSON.stringify(peqBands)),
      });
      mag = mag.map((v, i) => v + (pm[i] ?? 0));
      ph = ph.map((v, i) => v + (pp[i] ?? 0));
    }

    // Apply crossover filters
    if (b.targetEnabled && (b.target.high_pass || b.target.low_pass)) {
      const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
        freq: [...b.measurement!.freq],
        highPass: b.target.high_pass ? JSON.parse(JSON.stringify(b.target.high_pass)) : null,
        lowPass: b.target.low_pass ? JSON.parse(JSON.stringify(b.target.low_pass)) : null,
      });
      mag = mag.map((v, i) => v + (xm[i] ?? 0));
      ph = ph.map((v, i) => v + (xp[i] ?? 0));
    }

    return { id: b.id, name: b.name, mag, ph, freq: [...b.measurement!.freq] };
  });

  const bandData = await Promise.all(bandDataPromises);

  // Interpolate all bands onto common freq grid
  for (const bd of bandData) {
    if (bd.freq.length !== nPts || bd.freq[0] !== freq[0]) {
      const interpMag = interpOnGrid(bd.freq, bd.mag, freq);
      const interpPh = interpOnGrid(bd.freq, bd.ph, freq);
      // Replace with interpolated data; NaN → -200 dB / 0°
      bd.mag = interpMag.map(v => isNaN(v) ? -200 : v);
      bd.ph = interpPh.map(v => isNaN(v) ? 0 : v);
      bd.freq = [...freq];
      console.log(`[auto-align] interpolated "${bd.name}" onto common grid (${nPts} pts)`);
    } else {
      console.log(`[auto-align] "${bd.name}" already on common grid`);
    }
  }

  console.log("[auto-align] sorted bands (HF→LF):", sorted.map(b => ({
    name: b.name,
    hp: b.target?.high_pass?.freq_hz ?? "none",
    lp: b.target?.low_pass?.freq_hz ?? "none",
    targetEnabled: b.targetEnabled,
  })));

  // Find crossover regions between adjacent bands (HF→LF order)
  // sorted[i] = higher freq, sorted[i+1] = lower freq
  const crossoverRegions: { refIdx: number; optIdx: number; freqRange: [number, number] }[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const hpFreq = sorted[i].target?.high_pass?.freq_hz;
    const lpFreq = sorted[i + 1].target?.low_pass?.freq_hz;
    if (lpFreq && hpFreq) {
      const xoFreq = (lpFreq + hpFreq) / 2;
      const fLo = xoFreq / 1.4142;
      const fHi = xoFreq * 1.4142;
      crossoverRegions.push({ refIdx: i, optIdx: i + 1, freqRange: [fLo, fHi] });
      console.log(`[auto-align] XO pair: ${sorted[i].name} (HP=${hpFreq}) ↔ ${sorted[i+1].name} (LP=${lpFreq}), range=${fLo.toFixed(0)}-${fHi.toFixed(0)} Hz`);
    } else {
      console.log(`[auto-align] SKIP pair: ${sorted[i].name} (HP=${sorted[i].target?.high_pass?.freq_hz}) ↔ ${sorted[i+1].name} (LP=${sorted[i+1].target?.low_pass?.freq_hz})`);
    }
  }

  // Initialize delays: HF band = 0 (reference)
  const delays = new Array(sorted.length).fill(0);

  // Optimize sequentially HF→LF
  for (const xo of crossoverRegions) {
    const bestDelay = optimizePairDelay(
      freq, bandData, delays, xo.refIdx, xo.optIdx, xo.freqRange
    );

    if (bestDelay >= 0) {
      // Positive delay: lower band needs more delay — just assign
      delays[xo.optIdx] = bestDelay;
      console.log(`[auto-align] pair ${sorted[xo.refIdx].name}↔${sorted[xo.optIdx].name}: raw=${bestDelay.toFixed(6)}s, delays now=`, [...delays].map(d => (d*1000).toFixed(3)+"ms"));
    } else {
      // Negative delay: lower band should be EARLIER than upper bands
      // → propagate |bestDelay| to ALL already-processed bands (indices 0..optIdx-1)
      const shift = -bestDelay; // positive amount to add
      for (let k = 0; k < xo.optIdx; k++) {
        delays[k] += shift;
      }
      delays[xo.optIdx] = 0;
      console.log(`[auto-align] pair ${sorted[xo.refIdx].name}↔${sorted[xo.optIdx].name}: raw=${bestDelay.toFixed(6)}s, delays now=`, [...delays].map(d => (d*1000).toFixed(3)+"ms"));
    }
  }

  // Build result map (all delays guaranteed >= 0)
  const result: Record<string, number> = {};
  for (let i = 0; i < sorted.length; i++) {
    result[sorted[i].id] = Math.round(delays[i] * 1e6) / 1e6;
  }

  console.log("[auto-align] FINAL delays:", Object.entries(result).map(([id, d]) => {
    const b = sorted.find(s => s.id === id);
    return `${b?.name ?? id}: ${(d*1000).toFixed(3)}ms`;
  }));

  return { delays: result };
}

/**
 * Optimize delay for band hiIdx relative to loIdx to maximize
 * coherent sum amplitude in the crossover region.
 */
function optimizePairDelay(
  freq: number[],
  bandData: { id: string; mag: number[]; ph: number[] }[],
  delays: number[],
  loIdx: number,
  hiIdx: number,
  freqRange: [number, number],
): number {
  // Find frequency indices in the crossover region
  const xoIndices: number[] = [];
  for (let j = 0; j < freq.length; j++) {
    if (freq[j] >= freqRange[0] && freq[j] <= freqRange[1]) {
      xoIndices.push(j);
    }
  }

  if (xoIndices.length === 0) return 0;

  // Cost function: negative mean amplitude in crossover region
  const cost = (delayHi: number): number => {
    let totalAmp = 0;
    for (const j of xoIndices) {
      let re = 0, im = 0;
      // Sum all bands with their current delays
      for (let b = 0; b < bandData.length; b++) {
        const d = b === hiIdx ? delayHi : delays[b];
        const amp = Math.pow(10, bandData[b].mag[j] / 20);
        const phRad = (bandData[b].ph[j] + 360 * freq[j] * d) * Math.PI / 180;
        re += amp * Math.cos(phRad);
        im += amp * Math.sin(phRad);
      }
      totalAmp += Math.sqrt(re * re + im * im);
    }
    return -totalAmp / xoIndices.length;
  };

  // Phase scan: adaptive range based on crossover frequency
  const xoCenterFreq = (freqRange[0] + freqRange[1]) / 2;

  // Adaptive sweep range: wider for low-frequency crossovers
  const adaptiveMaxMs = xoCenterFreq < 200 ? 5.0 : xoCenterFreq < 500 ? 3.0 : 2.0;
  const scanRange = adaptiveMaxMs / 1000;

  // Scan over ±scanRange in fine steps
  const nSteps = 200;
  let bestDelay = 0;
  let bestCost = cost(0);

  for (let i = 0; i <= nSteps; i++) {
    const d = -scanRange + (2 * scanRange * i) / nSteps;
    const c = cost(d);
    if (c < bestCost) {
      bestCost = c;
      bestDelay = d;
    }
  }

  // Gradient descent refinement
  let lr = scanRange / nSteps / 2; // half step size
  let currentDelay = bestDelay;

  for (let iter = 0; iter < 50; iter++) {
    const eps = 1e-7;
    const grad = (cost(currentDelay + eps) - cost(currentDelay - eps)) / (2 * eps);
    const newDelay = currentDelay - lr * grad;
    const newCost = cost(newDelay);

    if (newCost < cost(currentDelay)) {
      currentDelay = newDelay;
    } else {
      lr *= 0.5; // shrink step
    }

    if (lr < 1e-9) break;
  }

  return currentDelay;
}

/** Estimate center frequency of a band based on its crossover filters */
function bandCenterFreq(b: BandState): number {
  const hp = b.target?.high_pass?.freq_hz ?? 20;
  const lp = b.target?.low_pass?.freq_hz ?? 20000;
  return Math.sqrt(hp * lp); // geometric mean
}
