/**
 * b140.14.1 — Phase 4 slice 2: Rust-side FIR pipeline dispatch.
 *
 * Routes a band-evaluator request to either `generate_model_fir_iir`
 * (IIR-analytical cascade) or `generate_model_fir` (FFT cepstral) via
 * the unified routing predicate (`pickFirRoute` in src/lib/fir-routing.ts,
 * mirrored Rust-side as `fir::route_for`).
 *
 * Extracted from the inline `useIirPath ? invoke(...) : invoke(...)`
 * block in band-evaluator.ts so the dispatch surface is testable in
 * isolation and the route-to-Tauri payload mapping has a single home.
 */
import { invoke } from "@tauri-apps/api/core";
import type { FilterConfig, PeqBand, ShelfConfig } from "../types";
import { pickFirRoute } from "../fir-routing";

/** Raw response shape returned by both `generate_model_fir_iir` and
 *  `generate_model_fir`. Kept identical to the legacy inline type so
 *  the band-evaluator unpacker doesn't need to change. */
export interface FirInvokeResult {
  impulse: number[];
  // b141.6: time_ms dropped from the IPC payload (pure linear ramp, ~MB of
  // JSON at 65k taps) — derived client-side from impulse.length + sample_rate.
  realized_mag: number[];
  realized_phase: number[];
  taps: number;
  sample_rate: number;
  norm_db: number;
  causality: number;
  /** b141.19: leading zeros applied to the shipped impulse = the band's
   *  latency. N/2 when the tail fits; less when it did not. */
  wav_delay_samples: number;
  /** b141.40: corner of the zero-phase ultrasonic low-pass Rust applied
   *  (fir/ultrasonic.rs); null below 88.2 kHz. */
  ultrasonic_lp_hz?: number | null;
}

/** What `dispatchFirInvoke` returns: the Rust payload plus the route that
 *  produced it. b141.36: the Export tab needs to know which pipeline ran —
 *  the IIR cascade ignores the window setting, and a dropdown that silently
 *  does nothing is worse than a disabled one. Reported by the dispatcher
 *  rather than re-derived in the UI, so there is still exactly one predicate. */
export interface FirDispatchResult extends FirInvokeResult {
  route: "iir" | "cepstral";
}

/** Structurally-typed subset of FirRequestConfig — keeping a local
 *  interface here avoids importing back from band-evaluator.ts, which
 *  would create a cycle. TypeScript structural typing means the caller
 *  passes its FirRequestConfig and TS accepts it. */
interface DispatchFirConfig {
  taps: number;
  sampleRate: number;
  window: string;
  maxBoostDb: number;
  noiseFloorDb: number;
  iterations: number;
  freqWeighting: boolean;
  narrowbandLimit: boolean;
  nbSmoothingOct: number;
  nbMaxExcessDb: number;
}

/** Build the snake_case Rust-side FirConfig payload from the camelCase
 *  TS FirRequestConfig + the linear-main / subsonic-cutoff overrides
 *  that the band-evaluator computes from the current band. */
function buildSharedFirConfig(
  cfg: DispatchFirConfig,
  linearMain: boolean,
  subsonicCutoffHz: number | null,
  mixedPhase: boolean,
) {
  return {
    taps: cfg.taps,
    sample_rate: cfg.sampleRate,
    max_boost_db: cfg.maxBoostDb,
    noise_floor_db: cfg.noiseFloorDb,
    window: cfg.window,
    // b141.2: MixedPhase tells the cepstral path to honour the per-filter
    // model phase verbatim (HP min + LP linear) instead of recomposing from a
    // single linear_phase_main flag. Composite for every other config.
    phase_mode: mixedPhase ? "MixedPhase" : "Composite",
    linear_phase_main: linearMain,
    subsonic_cutoff_hz: subsonicCutoffHz,
    iterations: cfg.iterations,
    freq_weighting: cfg.freqWeighting,
    narrowband_limit: cfg.narrowbandLimit,
    nb_smoothing_oct: cfg.nbSmoothingOct,
    nb_max_excess_db: cfg.nbMaxExcessDb,
  };
}

/** Dispatch a FIR-generation request to the appropriate Rust pipeline.
 *
 *  b140.7 routing intent: the IIR-analytical cascade for configurations
 *  whose phase can be expressed bit-exactly through digital biquads
 *  (LR / Butterworth / Custom HP+LP and PEQ peaking, no Gaussian, no
 *  subsonic protect, no linear-phase main). The FFT-cepstral path
 *  handles every other case. The routing predicate itself lives in
 *  fir-routing.ts and is mirrored Rust-side in fir/dispatch.rs.
 */
export async function dispatchFirInvoke(
  hp: FilterConfig | null,
  lp: FilterConfig | null,
  // b141.17 (audit): shelves and tilt are part of the target the plot draws.
  // Shelves ride the IIR cascade as RBJ sections; a non-zero tilt forces the
  // cepstral route (a constant log-slope has no rational realisation).
  lowShelf: ShelfConfig | null,
  highShelf: ShelfConfig | null,
  tiltDbPerOctave: number,
  enabledPeq: PeqBand[],
  linearMain: boolean,
  subsonicCutoffHz: number | null,
  firFreq: number[],
  firTargetMag: number[],
  firPeqMag: number[],
  firCombinedPhase: number[],
  cfg: DispatchFirConfig,
): Promise<FirDispatchResult> {
  const useIirPath =
    (await pickFirRoute(hp, lp, linearMain, subsonicCutoffHz, tiltDbPerOctave)) === "iir";
  // b141.2: a band whose HP and LP disagree on linear_phase (e.g. HP min +
  // LP linear) is not IIR-realisable and cannot be expressed by a single
  // linear_phase_main flag. Flag it so the cepstral path consumes the
  // per-filter model phase (firCombinedPhase) directly.
  const isLin = (f: FilterConfig | null) => !!f && f.linear_phase === true;
  const mixedPhase = !!hp && !!lp && isLin(hp) !== isLin(lp);
  const sharedFirConfig = buildSharedFirConfig(cfg, linearMain, subsonicCutoffHz, mixedPhase);

  if (useIirPath) {
    const out = await invoke<FirInvokeResult>("generate_model_fir_iir", {
      freq: firFreq,
      hp,
      lp,
      lowShelf,
      highShelf,
      peq: enabledPeq,
      config: sharedFirConfig,
    });
    return { ...out, route: "iir" };
  }
  const out = await invoke<FirInvokeResult>("generate_model_fir", {
    freq: firFreq,
    targetMag: firTargetMag,
    peqMag: firPeqMag,
    modelPhase: firCombinedPhase,
    config: sharedFirConfig,
  });
  return { ...out, route: "cepstral" };
}
