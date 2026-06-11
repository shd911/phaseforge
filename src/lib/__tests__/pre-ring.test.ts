// b141.11: pre-ringing control zone. Pre-ringing is produced by
// LINEAR-PHASE crossovers at their own cutoff frequencies. The old model
// sized the zone as 1.5 periods of the project's lowest HP regardless of
// phase mode — a 20 Hz min-phase HP stretched the zone to 75 ms while the
// actual pre-ringing source (linear-phase XO at 1800 Hz) rings at ~0.6 ms
// periods. New model: 2 periods of the LOWEST linear-phase crossover;
// no linear-phase crossovers → zone off (min-phase cannot pre-ring).

import { describe, it, expect } from "vitest";
import { preRingZoneMs } from "../pre-ring";
import type { BandState } from "../../stores/bands";
import type { FilterConfig } from "../types";

function filt(freq: number, linear: boolean): FilterConfig {
  return {
    filter_type: "LinkwitzRiley", freq_hz: freq, order: 2,
    q: null, linear_phase: linear, subsonic_protect: null,
  } as FilterConfig;
}

function band(hp: FilterConfig | null, lp: FilterConfig | null, targetEnabled = true): BandState {
  return {
    id: "x", name: "x", measurement: null, measurementFile: null,
    settings: { smoothing: "off" },
    target: {
      reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: hp, low_pass: lp, low_shelf: null, high_shelf: null,
    },
    targetEnabled, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  } as unknown as BandState;
}

describe("preRingZoneMs (b141.11)", () => {
  it("user 3WAY config: linear XO at 1800 → ~1.11 ms, not 75 ms from the 20 Hz HP", () => {
    const bands = [
      band(null, filt(200, false)),                 // woofer, min-phase LP
      band(filt(200, false), filt(1800, true)),     // mid, linear LP
      band(filt(1800, true), null),                 // tweeter, linear HP
    ];
    expect(preRingZoneMs(bands)).toBeCloseTo(2 / 1800 * 1000, 3);
  });

  it("all min-phase → null (min-phase systems cannot pre-ring)", () => {
    const bands = [band(null, filt(200, false)), band(filt(200, false), null)];
    expect(preRingZoneMs(bands)).toBeNull();
  });

  it("lowest linear-phase fc wins", () => {
    const bands = [band(null, filt(300, true)), band(filt(300, true), filt(2500, true))];
    expect(preRingZoneMs(bands)).toBeCloseTo(2 / 300 * 1000, 3);
  });

  it("targetEnabled=false bands are ignored", () => {
    const bands = [band(filt(1800, true), null, false), band(filt(200, false), null)];
    expect(preRingZoneMs(bands)).toBeNull();
  });
});
