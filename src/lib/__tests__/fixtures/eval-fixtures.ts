// Deterministic fixtures for the b139.x unified-evaluation refactor.
// No randomness, no clock — every value here is a pure function of the inputs.

import type { FilterConfig } from "../../types";

/** Synthetic measurement on a 512-point log freq grid 20 Hz – 20 kHz.
 *  Smooth shape: -3 dB / -12 dB·oct rolloff below 50 Hz, flat to 5 kHz,
 *  -6 dB·oct above. Phase = 0 everywhere. */
export function fixtureMeasurement(): { freq: number[]; magnitude: number[]; phase: number[] } {
  const n = 512;
  const freq: number[] = [];
  const magnitude: number[] = [];
  const phase: number[] = [];
  for (let i = 0; i < n; i++) {
    const f = 20 * Math.pow(20000 / 20, i / (n - 1));
    freq.push(f);
    let mag = 0;
    if (f < 50) mag = -3 - 12 * Math.log2(50 / f);
    else if (f > 5000) mag = -6 * Math.log2(f / 5000);
    magnitude.push(mag);
    phase.push(0);
  }
  return { freq, magnitude, phase };
}

export function fixtureGaussianHP(linearPhase: boolean, subsonicProtect: boolean | null): FilterConfig {
  return {
    filter_type: "Gaussian",
    order: 4,
    freq_hz: 632,
    shape: 1.0,
    linear_phase: linearPhase,
    q: null,
    subsonic_protect: subsonicProtect,
  };
}

export function fixtureLR4HP(): FilterConfig {
  return {
    filter_type: "LinkwitzRiley",
    order: 4,
    freq_hz: 80,
    shape: null,
    linear_phase: false,
    q: null,
    subsonic_protect: null,
  };
}

/** Six canonical configurations referenced in TZ-unified-evaluation.md. */
export const FIXTURE_CONFIGS = [
  { name: "gaussian_lin_subsonic_off", hp: fixtureGaussianHP(true,  false), label: "1. Gaussian linear, subsonic OFF" },
  { name: "gaussian_lin_subsonic_on",  hp: fixtureGaussianHP(true,  true),  label: "2. Gaussian linear, subsonic ON" },
  { name: "gaussian_min_subsonic_off", hp: fixtureGaussianHP(false, false), label: "3. Gaussian min-phase, subsonic OFF" },
  { name: "gaussian_min_subsonic_on",  hp: fixtureGaussianHP(false, true),  label: "4. Gaussian min-phase, subsonic ON" },
  { name: "lr4_baseline",              hp: fixtureLR4HP(),                  label: "5. LR4 baseline (non-Gaussian)" },
  { name: "no_hp_fullrange",           hp: null,                            label: "6. No HP (full-range)" },
] as const;
