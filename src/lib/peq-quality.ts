// Frontend mirror of Rust peq::q_envelope (b137). Used to flag PEQ bands
// whose Q exceeds the warning threshold. Disabled bands are never flagged.

import type { PeqBand } from "./types";

const F_LO = 200;
const F_HI = 2000;
const Q_LO = 8;
const Q_HI = 3;

export function qWarnAt(freqHz: number): number {
  if (freqHz <= F_LO) return Q_LO;
  if (freqHz >= F_HI) return Q_HI;
  const t = (Math.log2(freqHz) - Math.log2(F_LO))
          / (Math.log2(F_HI) - Math.log2(F_LO));
  return Q_LO - (Q_LO - Q_HI) * t;
}

export function highQIndices(bands: PeqBand[]): Set<number> {
  const s = new Set<number>();
  for (let i = 0; i < bands.length; i++) {
    const b = bands[i];
    if (!b.enabled) continue;
    if (b.q > qWarnAt(b.freq_hz)) s.add(i);
  }
  return s;
}

/** b141.93: the time cost of one PEQ filter — how long its resonance keeps
 *  ringing after the onset (ms) until it is 60 dB / 100 dB below the signal.
 *  A peaking/shelf filter is 1 + a resonant term of peak size |A² − 1| (the
 *  linear gain minus one) whose envelope decays at the pole radius of the
 *  biquad; time = (level budget + 20·log10|A² − 1|) / decay rate. A narrow LF
 *  boost is a long resonator: +3.1 dB at 22 Hz Q 4 rings ~0.5 s to −60 dB,
 *  Q 2 half that; the export keeps the band's delay at half the file only
 *  while the tail fits there. Same RBJ biquads as the Rust side. */
export interface RingCost { ms60: number; ms100: number }

export function peqRingCost(b: Pick<PeqBand, "freq_hz" | "gain_db" | "q" | "filter_type">, sampleRate: number): RingCost {
  const A = Math.pow(10, b.gain_db / 40);
  const c = Math.abs(A * A - 1);
  if (c < 1e-6) return { ms60: 0, ms100: 0 };
  const type = b.filter_type ?? "Peaking";
  const w0 = 2 * Math.PI * b.freq_hz / sampleRate;
  const cw = Math.cos(w0), al = Math.sin(w0) / (2 * b.q), s = 2 * Math.sqrt(A) * al;
  let a0: number, a1: number, a2: number;
  if (type === "LowShelf") { a0 = (A + 1) + (A - 1) * cw + s; a1 = -2 * ((A - 1) + (A + 1) * cw); a2 = (A + 1) + (A - 1) * cw - s; }
  else if (type === "HighShelf") { a0 = (A + 1) - (A - 1) * cw + s; a1 = 2 * ((A - 1) - (A + 1) * cw); a2 = (A + 1) - (A - 1) * cw - s; }
  else { a0 = 1 + al / A; a1 = -2 * cw; a2 = 1 - al / A; }
  a1 /= a0; a2 /= a0;
  // Slowest pole radius: complex pair → sqrt(a2); real pair → the larger root.
  const disc = a1 * a1 - 4 * a2;
  const r = disc < 0 ? Math.sqrt(a2) : Math.max(Math.abs((-a1 + Math.sqrt(disc)) / 2), Math.abs((-a1 - Math.sqrt(disc)) / 2));
  if (!(r > 0 && r < 1)) return { ms60: Infinity, ms100: Infinity };
  const dbPerS = -20 * Math.log10(r) * sampleRate;
  const start = 20 * Math.log10(c);
  const t = (budget: number) => Math.max(0, budget + start) / dbPerS * 1000;
  return { ms60: t(60), ms100: t(100) };
}
