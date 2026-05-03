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
