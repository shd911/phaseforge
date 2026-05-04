// Canonical BandEvaluator (b139.1) — single source of truth for band
// evaluation. The legacy inline pipelines in FrequencyPlot.tsx /
// fir-export.ts / band-evaluation.ts each implement a subset of this; later
// b139.x stages migrate them onto evaluateBandFull. This file alone must NOT
// change behaviour for any caller — it ships in parallel with the old code.

import { invoke } from "@tauri-apps/api/core";
import { createResource, type Resource } from "solid-js";
import type { FilterConfig, Measurement, PeqBand, TargetResponse } from "./types";
import type { BandState } from "../stores/bands";
import {
  isGaussianMinPhase,
  gaussianFilterMagDb,
  subsonicMagDb,
  smoothingConfig,
} from "./plot-helpers";
import { hasActiveSubsonicProtect } from "./band-evaluation";

export interface FirRequestConfig {
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

export interface BandEvalRequest {
  band: BandState;
  /** Override the freq grid. If omitted: measurement.freq when present,
   *  otherwise a 512-point log grid 5 Hz – 40 kHz via evaluate_target_standalone. */
  freq?: number[];
  /** Pass a config to also generate FIR coefficients via generate_model_fir.
   *  If omitted, fir output is undefined. */
  fir?: FirRequestConfig;
  /** Generate IR + step via compute_impulse / compute_corrected_impulse. */
  includeIr?: boolean;
}

export interface BandEvalResult {
  freq: number[];
  measurementMag: number[] | null;
  measurementPhase: number[] | null;

  /** Pure target response: HP × LP × shelves × tilt × subsonic. Phase is
   *  reconstructed for all four Gaussian × subsonic combinations. */
  targetMag: number[] | null;
  targetPhase: number[] | null;

  /** PEQ correction (always min-phase by physics — biquads). */
  peqMag: number[];
  peqPhase: number[];

  /** Target + PEQ — what the user actually hears. */
  combinedTargetMag: number[] | null;
  combinedTargetPhase: number[] | null;

  refLevel: number;

  fir?: {
    impulse: number[];
    timeMs: number[];
    realizedMag: number[];
    realizedPhase: number[];
    taps: number;
    sampleRate: number;
    normDb: number;
    causality: number;
  };
  ir?: { impulse: number[]; step: number[]; time: number[] };
}

// ---------------------------------------------------------------------------
// Phase reconstruction — the unified version that handles all four
// Gaussian × subsonic combinations. Mirrors band-evaluation.ts:addGaussianMinPhase
// but is callable on any (freq, basePhase) pair regardless of caller.
// ---------------------------------------------------------------------------
async function reconstructTargetPhase(
  freq: number[],
  basePhase: number[],
  hp: FilterConfig | null | undefined,
  lp: FilterConfig | null | undefined,
): Promise<number[]> {
  let phase = [...basePhase];

  if (isGaussianMinPhase(hp)) {
    let hpMag = gaussianFilterMagDb(freq, hp!, false);
    if (hasActiveSubsonicProtect(hp)) {
      const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
      hpMag = hpMag.map((db, i) => db + subDb[i]);
    }
    const hpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: hpMag });
    phase = phase.map((v, i) => v + hpPh[i]);
  } else if (hasActiveSubsonicProtect(hp) && hp!.linear_phase === true) {
    // Linear-phase Gaussian still ships a min-phase subsonic — Hilbert from
    // subsonic-only magnitude.
    const subDb = subsonicMagDb(freq, hp!.freq_hz / 8);
    const subPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: subDb });
    phase = phase.map((v, i) => v + subPh[i]);
  }

  if (isGaussianMinPhase(lp)) {
    const lpMag = gaussianFilterMagDb(freq, lp!, true);
    const lpPh = await invoke<number[]>("compute_minimum_phase", { freq, magnitude: lpMag });
    phase = phase.map((v, i) => v + lpPh[i]);
  }

  return phase;
}

async function applyMeasurementSmoothing(m: Measurement, mode: string | null | undefined): Promise<Measurement> {
  if (!mode || mode === "off") return m;
  const config = smoothingConfig(mode as any);
  const smoothed = await invoke<number[]>("get_smoothed", {
    freq: m.freq, magnitude: m.magnitude, config,
  });
  return { ...m, magnitude: smoothed };
}

function autoRefLevel(freq: number[], magnitude: number[], hp: FilterConfig | null | undefined, lp: FilterConfig | null | undefined): number {
  const hpFreq = hp?.freq_hz ?? 20;
  const lpFreq = lp?.freq_hz ?? 20000;
  const pbLow = Math.max(20, hpFreq * 1.5);
  const pbHigh = Math.min(20000, lpFreq * 0.7);
  const refLow = pbLow < pbHigh ? pbLow : 200;
  const refHigh = pbLow < pbHigh ? pbHigh : 2000;
  let sum = 0, n = 0;
  for (let i = 0; i < freq.length; i++) {
    if (freq[i] >= refLow && freq[i] <= refHigh) { sum += magnitude[i]; n++; }
  }
  return n > 0 ? sum / n : 0;
}

export async function evaluateBandFull(req: BandEvalRequest): Promise<BandEvalResult> {
  const { band } = req;

  // 1. Measurement (with smoothing).
  let measurement: Measurement | null = null;
  if (band.measurement) {
    const raw: Measurement = JSON.parse(JSON.stringify(band.measurement));
    measurement = await applyMeasurementSmoothing(raw, band.settings?.smoothing);
  }

  // 2 + 3. Pick freq grid + evaluate target. The standalone branch returns
  //        magnitude+phase along with the grid in a single Tauri round-trip
  //        — reuse it instead of calling evaluate_target a second time.
  const targetCurve = JSON.parse(JSON.stringify(band.target));
  const refLevel = measurement
    ? autoRefLevel(measurement.freq, measurement.magnitude, band.target.high_pass, band.target.low_pass)
    : (targetCurve.reference_level_db ?? 0);

  let freq: number[];
  let targetMag: number[] | null = null;
  let targetPhase: number[] | null = null;

  if (req.freq && req.freq.length > 0) {
    freq = req.freq;
    if (band.targetEnabled) {
      const response = await invoke<TargetResponse>("evaluate_target", { target: targetCurve, freq });
      targetMag = response.magnitude;
      targetPhase = await reconstructTargetPhase(freq, response.phase, band.target.high_pass, band.target.low_pass);
    }
  } else if (measurement) {
    freq = measurement.freq;
    if (band.targetEnabled) {
      const curveWithRef = { ...targetCurve, reference_level_db: targetCurve.reference_level_db + refLevel };
      const response = await invoke<TargetResponse>("evaluate_target", { target: curveWithRef, freq });
      targetMag = response.magnitude;
      targetPhase = await reconstructTargetPhase(freq, response.phase, band.target.high_pass, band.target.low_pass);
    }
  } else {
    // Standalone: one round-trip returns the grid + the response.
    const [standaloneFreq, response] = await invoke<[number[], TargetResponse]>(
      "evaluate_target_standalone",
      { target: targetCurve, nPoints: 512, fMin: 5, fMax: 40000 },
    );
    freq = standaloneFreq;
    if (band.targetEnabled) {
      targetMag = response.magnitude;
      targetPhase = await reconstructTargetPhase(freq, response.phase, band.target.high_pass, band.target.low_pass);
    }
  }

  // 4. PEQ contribution (mag + phase). PEQ phase is always min-phase
  //    physically (biquads); we ask Rust for the complex response so callers
  //    that previously used only peqMag pick up the correct phase too.
  const enabledPeq = (band.peqBands ?? []).filter((p: PeqBand) => p.enabled);
  let peqMag: number[] = new Array(freq.length).fill(0);
  let peqPhase: number[] = new Array(freq.length).fill(0);
  if (enabledPeq.length > 0) {
    const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
      freq, bands: enabledPeq,
    });
    peqMag = pm;
    peqPhase = pp;
  }

  // 5. Combined target + PEQ (what the listener experiences after correction).
  //    Only meaningful when a target curve exists — when targetEnabled=false
  //    the combined fields stay null (callers should fall back to peqMag/peqPhase
  //    directly if they want to render PEQ-only).
  let combinedTargetMag: number[] | null = null;
  let combinedTargetPhase: number[] | null = null;
  if (targetMag && targetPhase) {
    combinedTargetMag = targetMag.map((m, i) => m + peqMag[i]);
    combinedTargetPhase = targetPhase.map((p, i) => p + peqPhase[i]);
  }

  // 6. Optional FIR. b139.4a: send PhaseMode::Composite, which lets Rust
  //    honour the user's linear-phase choice for the main filter while
  //    keeping any subsonic-protect contribution min-phase. Composite
  //    degenerates to LinearPhase / MinimumPhase when no subsonic is on,
  //    so this single path replaces the old isLin / demotion logic.
  let fir: BandEvalResult["fir"];
  if (req.fir && targetMag) {
    const isUserLin = (f: FilterConfig | null | undefined) =>
      !f || f.linear_phase === true;
    const linearMain =
      isUserLin(band.target.high_pass) && isUserLin(band.target.low_pass);
    const hp = band.target.high_pass;
    const subsonicCutoff = hasActiveSubsonicProtect(hp) ? hp!.freq_hz / 8 : null;
    const cfg = req.fir;
    const result = await invoke<{
      impulse: number[]; time_ms: number[]; realized_mag: number[];
      realized_phase: number[]; taps: number; sample_rate: number;
      norm_db: number; causality: number;
    }>("generate_model_fir", {
      freq,
      targetMag,
      peqMag,
      modelPhase: combinedTargetPhase ?? new Array(freq.length).fill(0),
      config: {
        taps: cfg.taps,
        sample_rate: cfg.sampleRate,
        max_boost_db: cfg.maxBoostDb,
        noise_floor_db: cfg.noiseFloorDb,
        window: cfg.window,
        phase_mode: "Composite",
        linear_phase_main: linearMain,
        subsonic_cutoff_hz: subsonicCutoff,
        iterations: cfg.iterations,
        freq_weighting: cfg.freqWeighting,
        narrowband_limit: cfg.narrowbandLimit,
        nb_smoothing_oct: cfg.nbSmoothingOct,
        nb_max_excess_db: cfg.nbMaxExcessDb,
      },
    });
    fir = {
      impulse: result.impulse,
      timeMs: result.time_ms,
      realizedMag: result.realized_mag,
      realizedPhase: result.realized_phase,
      taps: result.taps,
      sampleRate: result.sample_rate,
      normDb: result.norm_db,
      causality: result.causality,
    };
  }

  // 7. Optional IR / step.
  let ir: BandEvalResult["ir"];
  if (req.includeIr && measurement) {
    if (combinedTargetMag && combinedTargetPhase) {
      const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
        "compute_corrected_impulse",
        {
          measFreq: measurement.freq,
          measMag: measurement.magnitude,
          measPhase: measurement.phase ?? new Array(measurement.freq.length).fill(0),
          realizedMag: combinedTargetMag,
          realizedPhase: combinedTargetPhase,
          firFreq: freq,
          sampleRate: measurement.sample_rate ?? 48000,
        },
      );
      ir = { impulse: r.impulse, step: r.step, time: r.time };
    } else {
      const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
        "compute_impulse",
        {
          freq: measurement.freq,
          magnitude: measurement.magnitude,
          phase: measurement.phase ?? new Array(measurement.freq.length).fill(0),
          sampleRate: measurement.sample_rate ?? null,
        },
      );
      ir = { impulse: r.impulse, step: r.step, time: r.time };
    }
  }

  return {
    freq,
    measurementMag: measurement ? measurement.magnitude : null,
    measurementPhase: measurement ? measurement.phase ?? null : null,
    targetMag,
    targetPhase,
    peqMag,
    peqPhase,
    combinedTargetMag,
    combinedTargetPhase,
    refLevel,
    fir,
    ir,
  };
}

export function createBandEvalResource(
  band: () => BandState,
  options?: {
    freq?: () => number[] | undefined;
    fir?: () => FirRequestConfig | undefined;
    includeIr?: () => boolean;
  },
): Resource<BandEvalResult> {
  const [resource] = createResource(
    () => ({
      band: band(),
      freq: options?.freq?.(),
      fir: options?.fir?.(),
      includeIr: options?.includeIr?.() ?? false,
    }),
    async (req) => evaluateBandFull(req),
  );
  return resource;
}
