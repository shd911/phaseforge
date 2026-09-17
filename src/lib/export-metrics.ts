// b141.37: pre-ringing and pre-peak floor as two separate Export metrics.
//
// The old pre-ring figure was "first sample above −80 dB of peak → peak".
// On a short filter that threshold catches the floor, not the ringing:
// tails that do not fit into the tap count (LF PEQ boosts, the min-phase
// PEQ part of a hybrid band) wrap around and sit before the peak as a flat
// plateau. Measured on a real band (LR2 300 Hz lin + 20/30 Hz PEQ boosts,
// 352.8 kHz): 79 ms at 64K, 6.3 ms at 1024K — while the crossover's own
// ringing stayed the same. The plateau was −37 dB at 64K, −58 dB at 256K.
//
// Split:
//   - Ring zone: how far before the peak a linear-phase HP/LP can ring —
//     `ringPeriods` periods of its cutoff, the longest over the band's
//     linear-phase filters. Periods to −80 dB, linear-phase LP at 48 kHz:
//     LR4 3.6, BU8 5.4, LR8 6.4, Bessel8 1.2, Gaussian(2) 2.1; HP ones ring
//     shorter. The zone must stay tight: the plateau does not start at the
//     zone edge, and on that real band a 10-period zone (33 ms) still let
//     a −37 dB plateau at 24 ms pass as ringing.
//   - Floor: the loudest pre-peak sample OUTSIDE the ring zone. Nothing
//     there belongs to the design; it is truncation residue carried into
//     the WAV.
//   - Pre-ring: time from the first crossing of PRE_RING_DB to the peak.
//     When the floor comes within FLOOR_MARGIN_DB of that threshold, the
//     threshold is raised above the floor and the value is a lower bound.

import type { BandState } from "../stores/bands";
import { F_MIN_WORK, type FilterConfig } from "./types";
import { orderToSlope } from "./slope";

export const PRE_RING_DB = -60;
export const FLOOR_MARGIN_DB = 6;
/** Floor colour bands: at or below GOOD — clean, above BAD — too few taps. */
export const FLOOR_GOOD_DB = -80;
export const FLOOR_BAD_DB = -60;
/** Floor values are clamped here — below it the number carries no meaning. */
const FLOOR_MIN_DB = -150;

export interface RingZone {
  ms: number;
  /** Cutoff of the filter that sized the zone; null = no HP/LP, working-range fallback. */
  hz: number | null;
  /** True when that filter is linear-phase. */
  linear: boolean;
}

/** Periods of its cutoff a filter may ring for: 2 + slope/12, at least 4
 *  (covers the −80 dB points above with margin; Gaussian has no slope),
 *  stretched by Q above Butterworth for Custom. */
export function ringPeriods(f: FilterConfig): number {
  const base = Math.max(4, 2 + orderToSlope(f.filter_type, f.order) / 12);
  const q = f.filter_type === "Custom" && f.q != null ? f.q / Math.SQRT1_2 : 1;
  return base * Math.max(1, q);
}

/** Ring zone of one band: the longest ringPeriods/cutoff over its
 *  linear-phase HP/LP; without one, over its HP/LP of any phase (a
 *  min-phase band only needs room for its rise to the peak); without any
 *  filter, 4 periods of the bottom of the working range. */
export function ringZone(band: BandState): RingZone {
  const filters = band.targetEnabled
    ? [band.target?.high_pass, band.target?.low_pass].filter(
        (f): f is FilterConfig => !!f && f.freq_hz > 0)
    : [];
  const lin = filters.filter((f) => f.linear_phase === true);
  const pool = lin.length > 0 ? lin : filters;
  let best: RingZone = { ms: (4 / F_MIN_WORK) * 1000, hz: null, linear: false };
  let first = true;
  for (const f of pool) {
    const ms = (ringPeriods(f) / f.freq_hz) * 1000;
    if (first || ms > best.ms) best = { ms, hz: f.freq_hz, linear: lin.length > 0 };
    first = false;
  }
  return best;
}

export interface PreRingAnalysis {
  preRingMs: number;
  /** Threshold the pre-ring was measured at, dB re peak. */
  thresholdDb: number;
  /** Floor raised the threshold above PRE_RING_DB — preRingMs is a lower bound. */
  limited: boolean;
  /** Loudest pre-peak level outside the ring zone, dB re peak (clamped at
   *  −150). null = the filter is shorter than the ring zone before its peak,
   *  so there is no region to measure. */
  floorDb: number | null;
}

export function analyzePreRing(
  impulse: ArrayLike<number>, sampleRate: number, zoneMs: number,
): PreRingAnalysis {
  let peakIdx = 0, peakVal = 0;
  for (let i = 0; i < impulse.length; i++) {
    const a = Math.abs(impulse[i]);
    if (a > peakVal) { peakVal = a; peakIdx = i; }
  }
  if (peakVal === 0) {
    return { preRingMs: 0, thresholdDb: PRE_RING_DB, limited: false, floorDb: null };
  }

  const floorEnd = peakIdx - Math.round((zoneMs * sampleRate) / 1000);
  let floorDb: number | null = null;
  if (floorEnd > 0) {
    let m = 0;
    for (let i = 0; i < floorEnd; i++) m = Math.max(m, Math.abs(impulse[i]));
    floorDb = m > 0 ? Math.max(FLOOR_MIN_DB, 20 * Math.log10(m / peakVal)) : FLOOR_MIN_DB;
  }

  const thresholdDb = floorDb == null
    ? PRE_RING_DB
    : Math.max(PRE_RING_DB, floorDb + FLOOR_MARGIN_DB);
  const thr = peakVal * Math.pow(10, thresholdDb / 20);
  let first = peakIdx;
  for (let i = 0; i < peakIdx; i++) {
    if (Math.abs(impulse[i]) > thr) { first = i; break; }
  }
  return {
    preRingMs: ((peakIdx - first) * 1000) / sampleRate,
    thresholdDb,
    limited: thresholdDb > PRE_RING_DB,
    floorDb,
  };
}
