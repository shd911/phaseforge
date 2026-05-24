import { invoke } from "@tauri-apps/api/core";
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
 *  b140.15.5: this used to be a JS-side textual mirror of Rust
 *  `fir::route_for`. The audit flagged that both predicates could
 *  drift identically and the parity test would not catch it.
 *  Resolution: Rust is now the single source of truth — this function
 *  is an async wrapper around the `pick_fir_route` Tauri command.
 *  Adds one IPC round-trip per FIR generation (~ms; negligible vs
 *  the FFT cost that follows).
 */
export async function pickFirRoute(
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
  linearMain: boolean,
  subsonicCutoffHz: number | null,
): Promise<"iir" | "cepstral"> {
  const route = await invoke<string>("pick_fir_route", {
    hp: hp ?? null,
    lp: lp ?? null,
    linearMain,
    subsonicCutoffHz,
  });
  return route === "Iir" ? "iir" : "cepstral";
}
