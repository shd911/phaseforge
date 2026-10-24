// b141.107: Linkwitz transform of a sealed box — a THEORETICAL correction.
// The bottom of a sub measurement is often the sweep edge (REW export from
// 20.29 Hz: 35–97 dB/oct, phase turning the wrong way), so the box's f0/Q0
// cannot be fitted from it. The transform works on an ASSUMED model: a 12
// dB/oct high-pass at f0 with Q0 (Qtc), lifted to fp = f0 / 10^(gain/40) at
// Q 1/√2. Rust `PeqFilterType::LinkwitzTransform` is the same filter.

import type { PeqBand } from "./types";

export const LINKWITZ_TARGET_Q = Math.SQRT1_2;
/** Default lift: +6 dB → fp = f0/√2, half an octave lower. */
export const LINKWITZ_DEFAULT_LIFT_DB = 6;
export const LINKWITZ_DEFAULT_Q0 = 0.707;

export function isLinkwitz(b: Pick<PeqBand, "filter_type">): boolean {
  return b.filter_type === "LinkwitzTransform";
}

/** Target corner fp of a transform band. */
export function linkwitzTargetHz(b: Pick<PeqBand, "freq_hz" | "gain_db">): number {
  return b.freq_hz / Math.pow(10, b.gain_db / 40);
}

/** |HP2(f; fc, q)| in dB — the sealed-box model. */
export function hp2Db(f: number, fc: number, q: number): number {
  const x = f / fc;
  const re = 1 - x * x, im = x / q;
  return 20 * Math.log10((x * x) / Math.hypot(re, im));
}

/** Analog magnitude of the transform (dB): box poles out, BU2 at fp in. */
export function linkwitzDb(f: number, b: Pick<PeqBand, "freq_hz" | "gain_db" | "q">): number {
  return hp2Db(f, linkwitzTargetHz(b), LINKWITZ_TARGET_Q) - hp2Db(f, b.freq_hz, b.q);
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** Level L of the box model L + HP2(f0, Q0) fitted to the measurement above
 *  the corner: median of (measured − model) over [f0, min(4·f0, fHigh)]. */
function modelLevel(freq: number[], mag: number[], t: Pick<PeqBand, "freq_hz" | "q">, fHigh: number): number | null {
  const v = freq.flatMap((f, i) => (f >= t.freq_hz && f <= Math.min(4 * t.freq_hz, fHigh) ? [mag[i] - hp2Db(f, t.freq_hz, t.q)] : []));
  return v.length ? median(v) : null;
}

/** Assumed −3 dB corner of a sealed box: going up from the bottom of the
 *  measurement, the first frequency where the 1/6-octave level reaches 3 dB
 *  under the passband median (40 Hz .. fHigh) — room dips above the corner
 *  do not count. Null when the measurement starts already in the passband. */
export function estimateSealedF0(freq: number[], mag: number[], fHigh: number): number | null {
  const pb = freq.flatMap((f, i) => (f >= 40 && f <= Math.max(60, fHigh) ? [mag[i]] : []));
  if (pb.length < 4) return null;
  const level = median(pb);
  const k = Math.pow(2, 1 / 12);
  const smooth = (c: number) => {
    let p = 0, n = 0;
    for (let i = 0; i < freq.length; i++) {
      if (freq[i] >= c / k && freq[i] <= c * k) { p += Math.pow(10, mag[i] / 10); n++; }
    }
    return n ? 10 * Math.log10(p / n) : NaN;
  };
  let below = false;
  for (let i = 0; i < freq.length && freq[i] <= 60; i++) {
    const l = smooth(freq[i]);
    if (l < level - 3) { below = true; continue; }
    return below ? Math.round(freq[i] * 10) / 10 : null;
  }
  return null;
}

/** A new transform band for a measurement: f0 from the −3 dB estimate (else
 *  30 Hz), Q0 0.707, +6 dB lift. */
export function linkwitzDefaults(freq: number[] | null, mag: number[] | null, fHigh: number): PeqBand {
  const f0 = (freq && mag && estimateSealedF0(freq, mag, fHigh)) || 30;
  return { freq_hz: f0, gain_db: LINKWITZ_DEFAULT_LIFT_DB, q: LINKWITZ_DEFAULT_Q0, enabled: true, filter_type: "LinkwitzTransform" };
}

/** What the optimizer fits when transforms are present: below each box's f0
 *  the measurement is replaced by the assumed model (the defective bottom
 *  edge is not fitted), and the transforms are applied — the optimizer then
 *  places its filters around a measurement that already carries them. */
export function measurementWithTransforms(freq: number[], mag: number[], transforms: PeqBand[], fHigh: number): number[] {
  let out = mag.slice();
  for (const t of transforms) {
    const level = modelLevel(freq, out, t, fHigh);
    out = out.map((m, i) => {
      const f = freq[i];
      const base = level != null && f < t.freq_hz ? level + hp2Db(f, t.freq_hz, t.q) : m;
      return base + linkwitzDb(f, t);
    });
  }
  return out;
}
