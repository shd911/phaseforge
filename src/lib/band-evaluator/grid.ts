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

/** b141.6 (audit): single home for the binary-search linear interpolation
 *  that previously existed as 5 near-identical copies (auto-align, three
 *  FrequencyPlot lambdas, sum.ts). Sites differ only in interpolation space
 *  and boundary policy — expressed here as options. */
export interface InterpOptions {
  /** Interpolate in log-frequency space (default: linear frequency). */
  logSpace?: boolean;
  /** Policy outside the source range: "clamp" to edge values (default),
   *  "nan", or a numeric fence value (e.g. -200 dB). */
  outside?: "clamp" | "nan" | number;
}

/** Interpolate `srcVals` (defined at ascending `srcFreq`) onto `dstFreq`.
 *  Null endpoints propagate as null so phase-wrap gaps survive resampling. */
export function interpOnGrid(
  srcFreq: number[],
  srcVals: readonly (number | null)[],
  dstFreq: number[],
  opts?: InterpOptions,
): (number | null)[] {
  const n = srcFreq.length;
  const outside = opts?.outside ?? "clamp";
  const edge = (i: number): number | null =>
    outside === "clamp" ? srcVals[i] : outside === "nan" ? NaN : outside;
  if (n === 0) return dstFreq.map(() => (outside === "clamp" || outside === "nan" ? NaN : outside));
  const xs = opts?.logSpace ? srcFreq.map(f => Math.log(Math.max(f, 1e-12))) : srcFreq;
  return dstFreq.map(f => {
    if (f < srcFreq[0]) return edge(0);
    if (f > srcFreq[n - 1]) return edge(n - 1);
    const x = opts?.logSpace ? Math.log(Math.max(f, 1e-12)) : f;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    const dt = xs[hi] - xs[lo];
    const t = dt > 0 ? (x - xs[lo]) / dt : 0;
    const a = srcVals[lo], b = srcVals[hi];
    if (a == null || b == null) return null;
    return a + t * (b - a);
  });
}

/** Linear interpolation of `srcVals` (defined at `srcFreq`) onto
 *  `dstFreq`, in log-frequency space, with constant clamp at the source
 *  boundaries. Used to bring realized FIR magnitude/phase from the FIR
 *  generation grid (5..fMaxFir) onto the caller's display grid so plots
 *  on the same x-axis line up. */
export function resampleOnLogGrid(srcFreq: number[], srcVals: number[], dstFreq: number[]): number[] {
  // Legacy edge cases predate interpOnGrid and are kept verbatim.
  if (srcFreq.length === 0) return dstFreq.map(() => 0);
  if (srcFreq.length === 1) return dstFreq.map(() => srcVals[0]);
  return interpOnGrid(srcFreq, srcVals, dstFreq, { logSpace: true, outside: "clamp" }) as number[];
}
