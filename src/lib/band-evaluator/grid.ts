/**
 * b140.14 — Phase 4 slice 1: frequency-grid helpers extracted from
 * band-evaluator.ts.
 *
 * Three pure functions, no Tauri / store / SolidJS dependencies. Safe
 * to unit-test in isolation and import from anywhere.
 *
 *   - buildLogGrid       — N log-spaced points between fMin and fMax.
 *   - buildCommonGrid    — union grid covering all band measurements
 *                          (fallback 5–40000 Hz when no measurements).
 *   - resampleOnLogGrid  — linear interp in log-freq with constant clamp
 *                          at source boundaries.
 */
import type { BandState } from "../../stores/bands";

/** N log-spaced points between fMin and fMax (inclusive of both endpoints). */
export function buildLogGrid(n: number, fMin: number, fMax: number): number[] {
  const out = new Array(n);
  const lo = Math.log(fMin), hi = Math.log(fMax);
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(lo + (hi - lo) * i / (n - 1));
  }
  return out;
}

/** Common grid: union of band measurement ranges, 512 log-spaced points.
 *  Falls back to 5–40000 Hz when no band has a measurement. */
export function buildCommonGrid(bands: BandState[]): number[] {
  let fMin = Infinity, fMax = -Infinity;
  for (const b of bands) {
    if (!b.measurement || b.measurement.freq.length === 0) continue;
    const f = b.measurement.freq;
    if (f[0] < fMin) fMin = f[0];
    if (f[f.length - 1] > fMax) fMax = f[f.length - 1];
  }
  if (!isFinite(fMin) || !isFinite(fMax)) {
    fMin = 5; fMax = 40000;
  }
  return buildLogGrid(512, fMin, fMax);
}

/** Linear interpolation of `srcVals` (defined at `srcFreq`) onto
 *  `dstFreq`, in log-frequency space, with constant clamp at the source
 *  boundaries. Used to bring realized FIR magnitude/phase from the FIR
 *  generation grid (5..fMaxFir) onto the caller's display grid so plots
 *  on the same x-axis line up. */
export function resampleOnLogGrid(srcFreq: number[], srcVals: number[], dstFreq: number[]): number[] {
  const n = srcFreq.length;
  if (n === 0) return dstFreq.map(() => 0);
  if (n === 1) return dstFreq.map(() => srcVals[0]);
  const logSrc = srcFreq.map(f => Math.log(Math.max(f, 1e-12)));
  return dstFreq.map(f => {
    if (f <= srcFreq[0]) return srcVals[0];
    if (f >= srcFreq[n - 1]) return srcVals[n - 1];
    const lf = Math.log(f);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (logSrc[mid] <= lf) lo = mid; else hi = mid;
    }
    const dt = logSrc[hi] - logSrc[lo];
    const t = dt > 0 ? (lf - logSrc[lo]) / dt : 0;
    return srcVals[lo] + t * (srcVals[hi] - srcVals[lo]);
  });
}
