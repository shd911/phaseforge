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
import type { FilterConfig, PeqBand } from "../types";
import { pickFirRoute } from "../fir-routing";

/** Raw response shape returned by both `generate_model_fir_iir` and
 *  `generate_model_fir`. Kept identical to the legacy inline type so
 *  the band-evaluator unpacker doesn't need to change. */
export interface FirInvokeResult {
  impulse: number[];
  time_ms: number[];
  realized_mag: number[];
  realized_phase: number[];
  taps: number;
  sample_rate: number;
  norm_db: number;
  causality: number;
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
) {
  return {
    taps: cfg.taps,
    sample_rate: cfg.sampleRate,
    max_boost_db: cfg.maxBoostDb,
    noise_floor_db: cfg.noiseFloorDb,
    window: cfg.window,
    phase_mode: "Composite",
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
  enabledPeq: PeqBand[],
  linearMain: boolean,
  subsonicCutoffHz: number | null,
  firFreq: number[],
  firTargetMag: number[],
  firPeqMag: number[],
  firCombinedPhase: number[],
  cfg: DispatchFirConfig,
): Promise<FirInvokeResult> {
  const useIirPath = pickFirRoute(hp, lp, linearMain, subsonicCutoffHz) === "iir";
  const sharedFirConfig = buildSharedFirConfig(cfg, linearMain, subsonicCutoffHz);

  if (useIirPath) {
    return invoke<FirInvokeResult>("generate_model_fir_iir", {
      freq: firFreq,
      hp,
      lp,
      peq: enabledPeq,
      config: sharedFirConfig,
    });
  }
  return invoke<FirInvokeResult>("generate_model_fir", {
    freq: firFreq,
    targetMag: firTargetMag,
    peqMag: firPeqMag,
    modelPhase: firCombinedPhase,
    config: sharedFirConfig,
  });
}
