/**
 * Auto-align delays for multi-band crossover systems.
 *
 * Algorithm: for each adjacent pair of bands with overlapping crossovers,
 * compute the delay offset that maximizes the coherent sum amplitude
 * in the crossover region. Uses gradient descent on the cost function
 * (negative sum amplitude at crossover frequencies).
 *
 * The first band is the reference (delay = 0); subsequent bands are
 * adjusted relative to it.
 */

import { invoke } from "@tauri-apps/api/core";
import type { BandState } from "../stores/bands";

/** Result of auto-align: map from bandId → delay in seconds */
export interface AlignResult {
  delays: Record<string, number>;
}

/**
 * Compute optimal alignment delays for all bands.
 *
 * @param bands - array of BandState (must have measurement + phase + target with crossovers)
 * @returns delays in seconds per band id
 */
export async function computeAutoAlign(bands: BandState[]): Promise<AlignResult> {
  // Filter bands with measurement + phase data, sorted by crossover frequency
  const validBands = bands.filter(
    b => b.measurement?.phase && b.measurement.phase.length > 0
  );

  if (validBands.length < 2) {
    // Nothing to align
    const delays: Record<string, number> = {};
    for (const b of validBands) delays[b.id] = 0;
    return { delays };
  }

  // Sort bands by HP frequency, with bandIndex fallback for stability
  const sorted = [...validBands].sort((a, b) => {
    const aHP = a.target?.high_pass?.freq_hz ?? 0;
    const bHP = b.target?.high_pass?.freq_hz ?? 0;
    if (aHP !== bHP) return aHP - bHP;
    return bands.indexOf(a) - bands.indexOf(b);
  });

  // Use the common frequency grid from first band
  const refBand = sorted[0];
  const freq = [...refBand.measurement!.freq];
  const nPts = freq.length;

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

    // Pad/trim to nPts
    while (mag.length < nPts) { mag.push(-200); ph.push(0); }

    return { id: b.id, name: b.name, mag, ph, freq: [...b.measurement!.freq] };
  });

  const bandData = await Promise.all(bandDataPromises);

  // Find crossover regions between adjacent bands
  const crossoverRegions: { loIdx: number; hiIdx: number; freqRange: [number, number] }[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const lpFreq = sorted[i].target?.low_pass?.freq_hz;
    const hpFreq = sorted[i + 1].target?.high_pass?.freq_hz;
    if (lpFreq && hpFreq) {
      const xoFreq = (lpFreq + hpFreq) / 2;
      // Crossover region: ±0.5 octave around crossover point
      const fLo = xoFreq / 1.4142; // ÷√2
      const fHi = xoFreq * 1.4142; // ×√2
      crossoverRegions.push({ loIdx: i, hiIdx: i + 1, freqRange: [fLo, fHi] });
    }
  }

  // Initialize delays: first band = 0, rest optimized
  const delays = new Array(sorted.length).fill(0);

  // Optimize each pair sequentially (each depends on the previous result)
  for (const xo of crossoverRegions) {
    const bestDelay = optimizePairDelay(
      freq, bandData, delays, xo.loIdx, xo.hiIdx, xo.freqRange
    );
    delays[xo.hiIdx] = bestDelay;
  }

  // Build result map
  const result: Record<string, number> = {};
  for (let i = 0; i < sorted.length; i++) {
    result[sorted[i].id] = delays[i];
  }

  // Normalize: shift all delays so minimum = 0 (only positive delays are physical)
  let minDelay = Infinity;
  for (const d of Object.values(result)) {
    if (d < minDelay) minDelay = d;
  }
  if (minDelay !== 0 && isFinite(minDelay)) {
    for (const k of Object.keys(result)) {
      result[k] = Math.round((result[k] - minDelay) * 1e6) / 1e6;
    }
  }

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
