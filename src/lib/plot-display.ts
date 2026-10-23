// b141.104 (audit 2026-10-23, section 4): pure display rules of the plot
// (FrequencyPlot.tsx) — one home each, unit-tested, instead of inline copies
// in the render functions.

import type { PeqBand } from "./types";
import { alignmentPhaseDeg } from "./types";
import { unwrapDegrees } from "./plot-helpers";
import { bulkDelaySeconds, shortestPhaseDelta } from "./band-evaluator/grid";

// ---------------------------------------------------------------------------
// Polarity
// ---------------------------------------------------------------------------

/** Phase (degrees) of a band as it plays: an inverted band (NOR/INV) is
 *  +180°. The single rule for every phase curve of the band — target,
 *  corrected, FIR, corrected through the FIR, in band view and in Σ —
 *  the same flip the WAV (b141.100), Export and IR/Step apply. Nulls (wrap
 *  gaps) stay null; the caller wraps. The measurement is never flipped. */
export function polarityPhase<T extends number | null>(phase: readonly T[], inverted: boolean): T[] {
  if (!inverted) return phase as T[];
  return phase.map((v) => (v == null ? v : ((v as number) + 180) as T));
}

/** Time-domain counterpart of `polarityPhase`: impulse/step negated. */
export function polarityAmplitude(values: readonly number[], inverted: boolean): number[] {
  return inverted ? values.map((v) => -v) : (values as number[]);
}

// ---------------------------------------------------------------------------
// PEQ handle
// ---------------------------------------------------------------------------

/** Level of a PEQ filter's OWN response at its own frequency, relative to
 *  its 0 dB baseline — where its drag handle sits. RBJ peaking: the full
 *  gain; RBJ shelves: exactly half of it (A = 10^(g/40) at w0, for any Q). */
export function peqHandleDb(pb: Pick<PeqBand, "gain_db" | "filter_type">): number {
  return pb.filter_type === "Peaking" ? pb.gain_db : pb.gain_db / 2;
}

export const PEQ_DRAG_GAIN_LIMIT_DB = 24;

/** Gain after dragging the handle by `dyDb` (cursor Y now − at mousedown,
 *  in dB on the plot). The handle follows the cursor — a shelf's gain moves
 *  twice as fast because its handle sits at half the gain — and nothing
 *  jumps at the first move, wherever inside the hit radius it was grabbed. */
export function peqGainAfterDrag(
  startGainDb: number, dyDb: number, filterType: PeqBand["filter_type"],
): number {
  const k = filterType === "Peaking" ? 1 : 2;
  const g = startGainDb + k * dyDb;
  return Math.max(-PEQ_DRAG_GAIN_LIMIT_DB, Math.min(PEQ_DRAG_GAIN_LIMIT_DB, g));
}

// ---------------------------------------------------------------------------
// Y fit
// ---------------------------------------------------------------------------

export interface FitSeries {
  values: readonly (number | null)[];
  visible: boolean;
}

function visibleExtent(series: readonly FitSeries[], floor: number): { min: number; max: number } | null {
  let min = Infinity, max = -Infinity;
  for (const s of series) {
    if (!s.visible) continue;
    for (const v of s.values) {
      if (v == null || !isFinite(v) || v <= floor) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return isFinite(min) ? { min, max } : null;
}

/** SPL / Σ magnitude fit over the VISIBLE curves: 0 dB at 20 % from the top,
 *  bottom at the data minimum (5 dB steps, ≥ 20 dB below 0) — and the top
 *  raised to keep the highest visible point on screen (a +8 dB hump used to
 *  be cut at +5). Hidden curves (Σ: every band's target, default off) no
 *  longer stretch the range to −150 dB. */
export function fitMagRange(series: readonly FitSeries[]): { min: number; max: number } {
  const data = visibleExtent(series, -200);
  const ext = data ?? { min: -100, max: 100 };
  const lowStep = Math.floor(ext.min / 5) * 5;
  const range = Math.max(20, -lowStep);
  const min = -range;
  let max = range * 0.25;
  // b141.104: the old fit ignored the data maximum entirely.
  const peakTop = Math.ceil((ext.max + 1) / 5) * 5;
  if (data && peakTop > max) max = peakTop;
  return { min, max };
}

/** Export tab magnitude fit (Model / FIR): data span + 10 % (≥ 2 dB). */
export function fitExportMagRange(series: readonly FitSeries[]): { min: number; max: number } | null {
  const ext = visibleExtent(series, -190);
  if (!ext) return null;
  const pad = Math.max(2, (ext.max - ext.min) * 0.1);
  return { min: ext.min - pad, max: ext.max + pad };
}

// ---------------------------------------------------------------------------
// Group delay of a coherent sum
// ---------------------------------------------------------------------------

/** Phase slope (deg/Hz) of one wrapped phase curve, robust to a large bulk
 *  delay: the band's own delay (time of flight) is removed before
 *  unwrapping and added back to the slope, so a 3 ms band on a 512-point
 *  log grid no longer aliases above ~10 kHz. */
export function phaseSlopeDegPerHz(
  freq: readonly number[], phase: readonly number[], mag: readonly number[],
): number[] {
  const tau = bulkDelaySeconds(freq, phase, mag);
  const resid = phase.map((p, i) => shortestPhaseDelta((p ?? 0) + 360 * freq[i] * tau));
  const un = unwrapDegrees(resid);
  return centralDiff(freq, un).map((d) => d - 360 * tau);
}

function centralDiff(freq: readonly number[], y: readonly number[]): number[] {
  const n = freq.length;
  const out = new Array<number>(n).fill(0);
  if (n < 2) return out;
  for (let i = 0; i < n; i++) {
    const a = i === 0 ? 0 : i - 1;
    const b = i === n - 1 ? n - 1 : i + 1;
    const df = freq[b] - freq[a];
    out[i] = df > 0 ? (y[b] - y[a]) / df : 0;
  }
  return out;
}

export interface GdPart {
  /** dB on `freq`. */
  mag: readonly number[];
  /** Wrapped phase, degrees, WITHOUT polarity and alignment delay. */
  phase: readonly number[];
  sign: 1 | -1;
  /** Alignment delay, s (positive = later). */
  delay: number;
}

/** Group delay (ms) of Σ_k sign_k·|H_k|·e^{j(φ_k − 360·f·τ_k)} computed
 *  from the derivative of the complex sum, not by differentiating its
 *  wrapped phase: GD = −Im(Σ'·conj Σ) / (2π|Σ|²). Each band's phase slope
 *  is taken with its bulk delay removed and the alignment delay enters
 *  analytically, so nothing aliases however coarse the grid is at HF
 *  (b141.104, audit 4.7: on the 512-point Σ grid 3 ms of alignment delay
 *  read > 10 % wrong above 12.3 kHz). NaN where the sum cancels. */
export function sumGroupDelayMs(freq: readonly number[], parts: readonly (GdPart | null)[]): number[] {
  const n = freq.length;
  const sRe = new Float64Array(n), sIm = new Float64Array(n);
  const dRe = new Float64Array(n), dIm = new Float64Array(n);
  let peak = 0;
  for (const p of parts) {
    if (!p) continue;
    const phSlope = phaseSlopeDegPerHz(freq, p.phase, p.mag);
    const magSlope = centralDiff(freq, p.mag.map((m) => (isFinite(m) ? m : -200)));
    for (let j = 0; j < n; j++) {
      const m = p.mag[j];
      if (!isFinite(m) || m <= -200) continue;
      const a = p.sign * Math.pow(10, m / 20);
      const thDeg = (p.phase[j] ?? 0) + alignmentPhaseDeg(freq[j], p.delay);
      const th = (thDeg - 360 * Math.round(thDeg / 360)) * Math.PI / 180;
      const thSlope = phSlope[j] * Math.PI / 180 - 2 * Math.PI * p.delay; // rad/Hz
      const aSlope = a * (Math.LN10 / 20) * magSlope[j];
      const c = Math.cos(th), s = Math.sin(th);
      sRe[j] += a * c; sIm[j] += a * s;
      dRe[j] += aSlope * c - a * thSlope * s;
      dIm[j] += aSlope * s + a * thSlope * c;
      if (Math.abs(a) > peak) peak = Math.abs(a);
    }
  }
  const floor = (peak * 1e-6) ** 2;
  const out = new Array<number>(n);
  for (let j = 0; j < n; j++) {
    const den = sRe[j] * sRe[j] + sIm[j] * sIm[j];
    if (!(den > floor)) { out[j] = NaN; continue; }
    const dTh = (dIm[j] * sRe[j] - dRe[j] * sIm[j]) / den; // rad/Hz
    out[j] = -dTh / (2 * Math.PI) * 1000;
  }
  return out;
}
