// Band evaluation legacy helpers. After b139.4c the unified path lives in
// band-evaluator.ts; only the small filter predicate stayed because
// ControlPanel + BandEvaluator + plot-helpers all need it.

import type { FilterConfig } from "./types";

/** True when the HP carries an active subsonic_protect that must contribute
 *  min-phase even if the Gaussian itself is linear-phase. */
export function hasActiveSubsonicProtect(hp: FilterConfig | null | undefined): boolean {
  return !!hp
    && hp.filter_type === "Gaussian"
    && hp.subsonic_protect === true
    && hp.freq_hz > 40;
}
