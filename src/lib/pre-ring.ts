// b141.11: pre-ringing control zone model.
//
// Pre-ringing is produced by LINEAR-PHASE crossovers, and it rings at the
// crossover's own cutoff frequency — the audibility window is a couple of
// periods of that frequency (backward masking is short and scales with the
// ringing period). The pre-b141.11 model sized the zone as 1.5 periods of
// the project's lowest HP regardless of phase mode, stretching a 20 Hz
// min-phase project to a 75 ms zone while the actual source (linear XO at
// 1800 Hz) needs ~1 ms.

import type { BandState } from "../stores/bands";

const PRE_RING_PERIODS = 2;

/** Zone extent in ms = PRE_RING_PERIODS / (lowest linear-phase crossover
 *  frequency across enabled bands). Returns null when the project has no
 *  linear-phase crossover — a min-phase system cannot pre-ring and the
 *  zone would be meaningless. */
export function preRingZoneMs(bands: BandState[]): number | null {
  let fMin = Infinity;
  for (const b of bands) {
    if (!b.targetEnabled) continue;
    for (const f of [b.target?.high_pass, b.target?.low_pass]) {
      if (f && f.linear_phase === true && f.freq_hz > 0) {
        fMin = Math.min(fMin, f.freq_hz);
      }
    }
  }
  return Number.isFinite(fMin) ? (PRE_RING_PERIODS / fMin) * 1000 : null;
}
