import type { FilterConfig } from "./types";

/** Decision: which Rust FIR pipeline should evaluate this band?
 *
 *  - "iir"      → IIR-analytical cascade (`generate_model_fir_iir`).
 *                 Bit-exact phase via DigitalBiquad cascade.
 *                 Restricted to: min-phase main + no subsonic +
 *                 every active crossover is LR / Butterworth / Custom.
 *  - "cepstral" → FFT cepstral path (`generate_model_fir`).
 *                 Used for Gaussian, Bessel, linear-phase main,
 *                 composite + subsonic, custom measured targets, etc.
 *
 *  Pure function — no Tauri / store / DOM access. Safe to unit-test
 *  in isolation and to call from anywhere.
 *
 *  Phase 0 (b140.10): extracted from band-evaluator.ts inline logic
 *  (lines 413–422) so the routing decision is testable as a single
 *  source of truth before the planned Phase-2 unification refactor.
 */
export function pickFirRoute(
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
  linearMain: boolean,
  subsonicCutoffHz: number | null,
): "iir" | "cepstral" {
  if (linearMain) return "cepstral";
  if (subsonicCutoffHz !== null) return "cepstral";
  if (!isIirRealizable(hp)) return "cepstral";
  if (!isIirRealizable(lp)) return "cepstral";
  return "iir";
}

/** Single filter is realisable by the digital biquad cascade. */
function isIirRealizable(f: FilterConfig | null | undefined): boolean {
  if (!f) return true;
  return (
    f.filter_type === "LinkwitzRiley" ||
    f.filter_type === "Butterworth" ||
    f.filter_type === "Custom"
  );
}
