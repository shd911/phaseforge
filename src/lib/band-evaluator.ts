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
  /** b140.2.1.3: when set, overrides the per-band autoRefLevel that
   *  shifts target_curve.reference_level_db. evaluateSum uses this to
   *  apply a project-wide globalRef so multi-way bands' targets line up
   *  on the same SPL baseline (parity with renderSumMode). */
  refLevelOverride?: number;
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

  /** b139.4c: HP/LP cross-section magnitudes/phases applied as a filter
   *  (compute_cross_section). Used for the corrected curve and sanity. */
  crossSectionMag: number[] | null;
  crossSectionPhase: number[] | null;

  /** b139.4c: corrected = measurement + PEQ + cross-section. Phase goes
   *  through reconstructTargetPhase so Gaussian/subsonic min-phase is
   *  applied uniformly with the target — fixes the SPL-tab bug where
   *  linear-Gaussian + subsonic produced corrected phase = 0. */
  correctedMag: number[] | null;
  correctedPhase: number[] | null;

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
  /** b139.4c: structured IR for the SPL/IR/Step views. Each sub-field is
   *  populated only when the underlying response exists; `time` is
   *  seconds (compute_impulse native). */
  ir?: {
    measurement?: { time: number[]; impulse: number[]; step: number[] };
    target?: { time: number[]; impulse: number[]; step: number[] };
    corrected?: { time: number[]; impulse: number[]; step: number[] };
  };
}

// ---------------------------------------------------------------------------
// Phase reconstruction — the unified version that handles all four
// Gaussian × subsonic combinations. Mirrors band-evaluation.ts:addGaussianMinPhase
// but is callable on any (freq, basePhase) pair regardless of caller.
// ---------------------------------------------------------------------------
export async function reconstructTargetPhase(
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
  // b140.2.1.3: refLevelOverride lets evaluateSum align every band's target
  // to a project-wide globalRef (max passband-avg across bands). When unset
  // we fall back to the per-band autoRefLevel as before.
  const refLevel = req.refLevelOverride !== undefined
    ? req.refLevelOverride
    : (measurement
        ? autoRefLevel(measurement.freq, measurement.magnitude, band.target.high_pass, band.target.low_pass)
        : (targetCurve.reference_level_db ?? 0));

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

  // 5b. b139.4c: cross-section (HP/LP filters as a standalone curve)
  //      — used to build the corrected response (measurement + PEQ + xs).
  let crossSectionMag: number[] | null = null;
  let crossSectionPhase: number[] | null = null;
  if (band.targetEnabled && (band.target.high_pass || band.target.low_pass)) {
    try {
      const [xm, xp] = await invoke<[number[], number[], number]>(
        "compute_cross_section",
        { freq, highPass: band.target.high_pass, lowPass: band.target.low_pass },
      );
      crossSectionMag = xm;
      crossSectionPhase = xp;
    } catch (_) {
      // No filters → leave null; corrected = meas + PEQ alone.
    }
  }

  // 5c. Corrected = measurement + PEQ + cross-section. Phase goes through
  //      the same Gaussian/subsonic reconstructTargetPhase as target so all
  //      four (linear × subsonic) combinations behave identically — fixes
  //      the SPL bug where corrected phase = 0 for linear-Gaussian + subsonic.
  let correctedMag: number[] | null = null;
  let correctedPhase: number[] | null = null;
  if (measurement) {
    correctedMag = measurement.magnitude.map((m, i) =>
      m + (peqMag[i] ?? 0) + (crossSectionMag?.[i] ?? 0)
    );
    if (measurement.phase) {
      let basePhase = measurement.phase.map((p, i) =>
        p + (peqPhase[i] ?? 0) + (crossSectionPhase?.[i] ?? 0)
      );
      basePhase = await reconstructTargetPhase(
        freq, basePhase, band.target.high_pass, band.target.low_pass,
      );
      correctedPhase = basePhase;
    }
  }

  // 6. Optional FIR. b139.4a: send PhaseMode::Composite, which lets Rust
  //    honour the user's linear-phase choice for the main filter while
  //    keeping any subsonic-protect contribution min-phase. Composite
  //    degenerates to LinearPhase / MinimumPhase when no subsonic is on,
  //    so this single path replaces the old isLin / demotion logic.
  //
  //    b139.5.3: FIR runs on its OWN log grid (5 Hz – min(40 kHz, Nyquist·0.95),
  //    512 points) — independent of the measurement grid that drives the
  //    SPL display. The display grid is fine for "what the listener
  //    perceives" but truncates the FIR's HP rolloff (no bins below 20 Hz)
  //    and its anti-aliasing headroom (no bins above 20 kHz when
  //    sr ≥ 88.2 kHz). This is the same grid the legacy
  //    generateBandImpulse used.
  let fir: BandEvalResult["fir"];
  if (req.fir && band.targetEnabled) {
    const isUserLin = (f: FilterConfig | null | undefined) =>
      !f || f.linear_phase === true;
    const linearMain =
      isUserLin(band.target.high_pass) && isUserLin(band.target.low_pass);
    const hp = band.target.high_pass;
    const subsonicCutoff = hasActiveSubsonicProtect(hp) ? hp!.freq_hz / 8 : null;
    const cfg = req.fir;

    // FIR-specific grid + target evaluation.
    const fMaxFir = Math.min(40000, cfg.sampleRate / 2 * 0.95);
    const [firFreq, firResp] = await invoke<[number[], TargetResponse]>(
      "evaluate_target_standalone",
      { target: targetCurve, nPoints: 512, fMin: 5, fMax: fMaxFir },
    );
    const firTargetMag = firResp.magnitude;
    const firTargetPhase = await reconstructTargetPhase(
      firFreq, firResp.phase, band.target.high_pass, band.target.low_pass,
    );

    // PEQ on the FIR grid (biquads are sample-rate independent — same bands).
    let firPeqMag: number[] = new Array(firFreq.length).fill(0);
    let firPeqPhase: number[] = new Array(firFreq.length).fill(0);
    if (enabledPeq.length > 0) {
      const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq: firFreq, bands: enabledPeq,
      });
      firPeqMag = pm;
      firPeqPhase = pp;
    }
    const firCombinedPhase = firTargetPhase.map((p, i) => p + firPeqPhase[i]);

    const result = await invoke<{
      impulse: number[]; time_ms: number[]; realized_mag: number[];
      realized_phase: number[]; taps: number; sample_rate: number;
      norm_db: number; causality: number;
    }>("generate_model_fir", {
      freq: firFreq,
      targetMag: firTargetMag,
      peqMag: firPeqMag,
      modelPhase: firCombinedPhase,
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

  // 7. Optional IR / step. b139.4c: structured by curve so callers can pick
  //     measurement / target / corrected independently.
  let ir: BandEvalResult["ir"];
  if (req.includeIr) {
    ir = {};
    const sr = measurement?.sample_rate ?? 48000;
    if (measurement) {
      try {
        const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
          "compute_impulse",
          {
            freq: measurement.freq,
            magnitude: measurement.magnitude,
            phase: measurement.phase ?? new Array(measurement.freq.length).fill(0),
            sampleRate: measurement.sample_rate ?? null,
          },
        );
        ir.measurement = { time: r.time, impulse: r.impulse, step: r.step };
      } catch (_) {}
    }
    if (targetMag && targetPhase) {
      try {
        const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
          "compute_impulse",
          { freq, magnitude: targetMag, phase: targetPhase, sampleRate: sr },
        );
        ir.target = { time: r.time, impulse: r.impulse, step: r.step };
      } catch (_) {}
    }
    if (correctedMag && correctedPhase) {
      try {
        const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
          "compute_impulse",
          { freq, magnitude: correctedMag, phase: correctedPhase, sampleRate: sr },
        );
        ir.corrected = { time: r.time, impulse: r.impulse, step: r.step };
      } catch (_) {}
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
    crossSectionMag,
    crossSectionPhase,
    correctedMag,
    correctedPhase,
    refLevel,
    fir,
    ir,
  };
}

// ---------------------------------------------------------------------------
// b139.4c: SUM aggregator. Evaluates each band via evaluateBandFull on a
// common log-spaced freq grid, then coherently sums target / corrected
// curves with polarity and alignment_delay phase rotation. The resulting
// sumMag/sumPhase/correctedSum* are what renderSumMode would draw on the
// SPL chart; sumIR is the IFFT of the corrected sum (optional).
// ---------------------------------------------------------------------------
export interface SumEvalResult {
  freq: number[];
  /** Per-band BandEvalResult resampled onto `freq`. Phase is unchanged
   *  (each band's evalRes is computed on its own grid, then results merged). */
  perBand: BandEvalResult[];
  /** Coherent sum of perBand[i].combinedTargetMag/Phase with polarity +
   *  alignment_delay phase rotation. Null when no band has a target. */
  sumTargetMag: number[] | null;
  sumTargetPhase: number[] | null;
  /** Sum of perBand[i].correctedMag/Phase. Coherent (re/im, polarity-aware)
   *  when every contributing band has a phase; otherwise b140.2.0.5 falls
   *  back to a power sum (10·log10(Σ 10^(m/10)), polarity ignored, phase
   *  null) for parity with the legacy renderSumMode. Null when no band has
   *  a measurement. */
  sumCorrectedMag: number[] | null;
  sumCorrectedPhase: number[] | null;
  /** b140.2.1.1: Σ Measurement (per-band measurementMag/Phase summed).
   *  Same coherent/incoherent semantics as the corrected sum. Null when
   *  no band has a measurement. */
  sumMeasurementMag: number[] | null;
  sumMeasurementPhase: number[] | null;
  /** b140.2.0.5: true → coherent corrected sum; false → power-sum fallback
   *  (sumCorrectedPhase will be null in that case). Always true when there
   *  is no corrected sum to compute. */
  coherent: boolean;
  /** b140.2.1.1: separate coherent flag for Σ Measurement. Same semantics
   *  as `coherent`. */
  coherentMeasurement: boolean;
  /** b140.2.1.5: per-band data resampled onto the common grid, with the
   *  out-of-range fence applied (b140.2.1.4). Lengths always equal
   *  `freq.length`, so UI plots can use a single x axis without per-band
   *  grid mismatches. Bins outside a band's native freq range carry
   *  −200 dB and 0° (i.e. effectively silent), so the curves only appear
   *  where the band actually has data. */
  perBandResampled: Array<{
    measurementMag: number[] | null;
    measurementPhase: number[] | null;
    targetMag: number[] | null;
    targetPhase: number[] | null;
    correctedMag: number[] | null;
    correctedPhase: number[] | null;
  }>;
  /** b140.2.2 Fix 4: project-wide reference SPL (max passband-avg
   *  across raw measurements). UI subtracts this from every magnitude
   *  curve so 0 dBr aligns with the loudest band. */
  globalRef: number;
  /** b140.2.2 Fix 1: average of enabled-target bands' refLevel. Σ target
   *  is shifted by this so its absolute SPL matches what the user expects
   *  on a dB SPL plot. */
  avgRef: number;
  /** Optional IR of the corrected sum (compute_impulse on summed mag/phase). */
  ir?: { time: number[]; impulse: number[]; step: number[] };
  /** b140.2.2 Fix 3: structured Σ IR (compute_impulse on each summed
   *  curve on the SUM IR grid — union of band ranges, ≥ 2048 pts, no
   *  20-20k extension). Populated only when SumEvalOptions.includeSumIr
   *  is true. */
  sumIr?: {
    measurement?: { time: number[]; impulse: number[]; step: number[] };
    target?: { time: number[]; impulse: number[]; step: number[] };
    corrected?: { time: number[]; impulse: number[]; step: number[] };
  };
}

export interface SumEvalOptions {
  freq?: number[];
  includeIr?: boolean;
  /** When true the per-band combined target is normalised (peak = 0 dB) before
   *  the coherent sum — matches how renderSumMode treats target IR aggregation. */
  normalizeTargetPerBand?: boolean;
  /** b140.2.2 Fix 3: when true, evaluateSum builds its own SUM IR grid
   *  (union of band ranges, ≥ 2048 pts, no 20-20k extension) and emits
   *  ir.measurement / ir.target / ir.corrected via compute_impulse. */
  includeSumIr?: boolean;
}

function buildLogGrid(n: number, fMin: number, fMax: number): number[] {
  const out = new Array(n);
  const lo = Math.log(fMin), hi = Math.log(fMax);
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(lo + (hi - lo) * i / (n - 1));
  }
  return out;
}

/** b140.2.1.7: log-linear slope (dB / octave) over a tail window of the
 *  source data. Used for trend extension when no target shape is supplied. */
function tailSlope(
  srcFreq: number[],
  srcMag: number[],
  fEnd: number,
  fromLow: boolean,
): number {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < srcFreq.length; i++) {
    const inWindow = fromLow ? srcFreq[i] <= fEnd : srcFreq[i] >= fEnd;
    if (inWindow) {
      xs.push(Math.log2(srcFreq[i]));
      ys.push(srcMag[i]);
    }
  }
  if (xs.length < 2) return 0;
  let mx = 0, my = 0;
  for (let i = 0; i < xs.length; i++) { mx += xs[i]; my += ys[i]; }
  mx /= xs.length; my /= ys.length;
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    num += dx * (ys[i] - my);
    den += dx * dx;
  }
  return den > 0 ? num / den : 0;
}

function nearestIdx(arr: number[], target: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - target);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

export interface ResampleExtensionOptions {
  /** Target curve magnitude on the targetFreq grid. When supplied, bins
   *  outside the source's native range are filled with this curve, offset
   *  so target ≡ measurement at the native boundary. Lets a band-limited
   *  measurement be physically extended through its design target (e.g.
   *  Gaussian HP rolloff for a tweeter measured only above 1 kHz). */
  extensionTargetMag?: number[];
  /** Fallback when extensionTargetMag is absent: linear extrapolation in
   *  log2(freq) space using the slope of the last ~1/4 octave of source
   *  data. Hits the right shape for full-range drivers without target. */
  fallbackToTrend?: boolean;
}

export async function resampleOntoGrid(
  srcFreq: number[],
  srcMag: number[] | null,
  srcPhase: number[] | null,
  targetFreq: number[],
  options?: ResampleExtensionOptions,
): Promise<{ mag: number[] | null; phase: number[] | null }> {
  if (srcMag === null) return { mag: null, phase: null };
  // interpolate_log handles both mag + phase in one call (phase optional).
  const [, mag, phase] = await invoke<[number[], number[], number[] | null]>(
    "interpolate_log",
    {
      freq: srcFreq, magnitude: srcMag, phase: srcPhase,
      nPoints: targetFreq.length,
      fMin: targetFreq[0], fMax: targetFreq[targetFreq.length - 1],
    },
  );

  // b140.2.1.4 / .7: Rust's interp_single clamps to boundary values when
  // the query freq is outside the source range. For SUM aggregation we
  // replace those with a physically motivated extension:
  //   • extensionTargetMag (offset so target meets meas at boundary), or
  //   • log-linear trend extrapolation from the tail, or
  //   • −200 dB silence fence (default — b140.2.1.4 behaviour).
  if (srcFreq.length === 0) return { mag, phase: phase ?? null };
  const fLo = srcFreq[0];
  const fHi = srcFreq[srcFreq.length - 1];
  // Math.exp(Math.log(x)) drifts ~4 ULPs in IEEE 754 — 1 ppb tolerance
  // keeps boundary bins inside while still catching genuine OOB queries.
  const tol = 1e-9;
  const lo = fLo * (1 - tol);
  const hi = fHi * (1 + tol);

  // Pre-compute target offsets so target shape meets measurement at the
  // native boundary. Without this the extension would jump by whatever
  // mismatch existed between target's analytical SPL and the measured one.
  let targetOffsetLo = 0, targetOffsetHi = 0;
  if (options?.extensionTargetMag) {
    const lo_idx = nearestIdx(targetFreq, fLo);
    const hi_idx = nearestIdx(targetFreq, fHi);
    targetOffsetLo = srcMag[0] - options.extensionTargetMag[lo_idx];
    targetOffsetHi = srcMag[srcMag.length - 1] - options.extensionTargetMag[hi_idx];
  }

  // Pre-compute tail slopes for trend extension. Window = ~1/4 octave at
  // each end (factor 2^0.25 ≈ 1.189).
  let lowSlope = 0, highSlope = 0;
  if (options?.fallbackToTrend) {
    lowSlope = tailSlope(srcFreq, srcMag, fLo * 1.189, true);
    highSlope = tailSlope(srcFreq, srcMag, fHi / 1.189, false);
  }

  const ext = mag.map((v, i) => {
    const f = targetFreq[i];
    if (f >= lo && f <= hi) return v;
    if (f < lo) {
      if (options?.extensionTargetMag) {
        return options.extensionTargetMag[i] + targetOffsetLo;
      }
      if (options?.fallbackToTrend) {
        return srcMag[0] + lowSlope * Math.log2(f / fLo);
      }
      return -200;
    }
    // f > hi
    if (options?.extensionTargetMag) {
      return options.extensionTargetMag[i] + targetOffsetHi;
    }
    if (options?.fallbackToTrend) {
      return srcMag[srcMag.length - 1] + highSlope * Math.log2(f / fHi);
    }
    return -200;
  });

  // Phase: out-of-range bins still get phase=0 — no physical meaning to
  // extrapolate phase outside measurement. Coherent sum sees the
  // extended magnitude with phase 0; that contributes a real-axis vector
  // which is the most neutral choice.
  const fencedPhase = phase
    ? phase.map((v, i) => {
        const f = targetFreq[i];
        return f < lo || f > hi ? 0 : v;
      })
    : null;
  return { mag: ext, phase: fencedPhase };
}

function coherentSum(
  freq: number[],
  bandsData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null>,
  refOffset?: number,
): { mag: number[]; phase: number[] } | null {
  const n = freq.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let any = false;
  for (const b of bandsData) {
    if (!b) continue;
    any = true;
    for (let j = 0; j < n; j++) {
      const amp = Math.pow(10, (b.mag[j] ?? -200) / 20) * b.sign;
      const phRad = ((b.phase[j] ?? 0) + 360 * freq[j] * b.delay) * Math.PI / 180;
      re[j] += amp * Math.cos(phRad);
      im[j] += amp * Math.sin(phRad);
    }
  }
  if (!any) return null;
  const off = refOffset ?? 0;
  const mag = new Array(n);
  const phase = new Array(n);
  for (let j = 0; j < n; j++) {
    const amplitude = Math.sqrt(re[j] * re[j] + im[j] * im[j]);
    mag[j] = amplitude > 0 ? 20 * Math.log10(amplitude) + off : -200;
    phase[j] = Math.atan2(im[j], re[j]) * 180 / Math.PI;
  }
  return { mag, phase };
}

/** b140.2.1.3: project-wide reference level — average passband (200..2000 Hz)
 *  of the LOUDEST band. Used by evaluateSum to align every band's target to
 *  the same SPL baseline; mirrors renderSumMode's `globalRef`. */
function passbandAvgDb(freq: number[], mag: number[], fLo: number, fHi: number): number | null {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < freq.length; i++) {
    if (freq[i] >= fLo && freq[i] <= fHi) {
      sum += mag[i];
      n++;
    }
  }
  return n > 0 ? sum / n : null;
}

export async function evaluateSum(
  bands: BandState[],
  options?: SumEvalOptions,
): Promise<SumEvalResult> {
  // b140.2.1.3 step A: project-wide reference level. max passband-avg over
  // raw measurements (200..2000 Hz) — multi-way fix so a tweeter's target
  // doesn't sit 30 dB below a woofer's. Falls back to 0 dB when no band
  // has a measurement (the standalone-target case).
  let globalRef: number | undefined;
  for (const b of bands) {
    if (!b.measurement) continue;
    const avg = passbandAvgDb(b.measurement.freq, b.measurement.magnitude, 200, 2000);
    if (avg === null) continue;
    if (globalRef === undefined || avg > globalRef) {
      globalRef = avg;
    }
  }

  // 1. Per-band evaluation. refLevelOverride aligns each band's target to
  //    globalRef (when computable) so the coherent sum sees consistent
  //    absolute SPL across the project.
  const perBand = await Promise.all(
    bands.map(b => evaluateBandFull({
      band: b,
      refLevelOverride: globalRef,
    })),
  );

  // 2. Build common log grid: union of band ranges, point count = max.
  let fMin = Infinity, fMax = -Infinity, nMax = 512;
  for (const r of perBand) {
    const f = r.freq;
    if (f.length === 0) continue;
    if (f[0] < fMin) fMin = f[0];
    if (f[f.length - 1] > fMax) fMax = f[f.length - 1];
    if (f.length > nMax) nMax = f.length;
  }
  if (!isFinite(fMin) || !isFinite(fMax)) {
    fMin = 20; fMax = 20000;
  }
  const freq = options?.freq ?? buildLogGrid(nMax, fMin, fMax);

  // 3. Resample target / corrected / measurement onto the common grid per band.
  const targetData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null> = [];
  const correctedData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null> = [];
  const measurementData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null> = [];
  // b140.2.0.5/.1.1: collect per-band magnitudes separately for the
  // power-sum fallback. The fallback is triggered when any band contributes
  // mag without phase — in that case ALL bands' mags participate, polarity
  // ignored. Mirrored for corrected and measurement (target is always
  // coherent because targetPhase is reconstructed analytically).
  const correctedMagsForFallback: number[][] = [];
  let anyCorrectedMagWithoutPhase = false;
  const measurementMagsForFallback: number[][] = [];
  let anyMeasurementMagWithoutPhase = false;
  // b140.2.1.5: per-band data on the common grid for UI plotting. Same
  // resampled arrays the aggregator sees, so curves never get stretched
  // across the wrong x range.
  const perBandResampled: SumEvalResult["perBandResampled"] = [];
  for (let i = 0; i < bands.length; i++) {
    const r = perBand[i];
    const sign: 1 | -1 = bands[i].inverted ? -1 : 1;
    const delay = bands[i].alignmentDelay ?? 0;
    // Resampled values for UI consumption — populated alongside the
    // aggregator-targeted resampled arrays so we never resample twice.
    const ui: SumEvalResult["perBandResampled"][number] = {
      measurementMag: null, measurementPhase: null,
      targetMag: null, targetPhase: null,
      correctedMag: null, correctedPhase: null,
    };

    // b140.2.1.7: target evaluated analytically on the COMMON grid (with
    // globalRef shift). This is the physically correct extension shape
    // for everything outside a band's native measurement range — gives
    // the supertweeter its HP rolloff below 1 kHz instead of a flat
    // boundary clamp or a hard −200 dB step.
    let extensionTargetMag: number[] | undefined;
    if (bands[i].targetEnabled) {
      const refShift = globalRef ?? 0;
      const tc = bands[i].target;
      const tcWithRef = {
        ...tc,
        reference_level_db: (tc.reference_level_db ?? 0) + refShift,
      };
      try {
        const resp = await invoke<TargetResponse>(
          "evaluate_target",
          { target: tcWithRef, freq },
        );
        extensionTargetMag = resp.magnitude;
      } catch (_) {
        extensionTargetMag = undefined;
      }
    }

    let resampledTargetMag: number[] | null = null;
    if (r.combinedTargetMag && r.combinedTargetPhase) {
      let tMag = r.combinedTargetMag;
      const tPhase = r.combinedTargetPhase;
      const resampled = await resampleOntoGrid(
        r.freq, tMag, tPhase, freq,
        // Target's own extension uses target shape (smooths its rolloff
        // through the analytical curve below/above the band's measurement
        // grid), with trend fallback if no target is available.
        { extensionTargetMag, fallbackToTrend: true },
      );
      if (resampled.mag && resampled.phase) {
        let mag = resampled.mag;
        if (options?.normalizeTargetPerBand) {
          let peak = -Infinity;
          for (const v of mag) if (v > peak) peak = v;
          mag = mag.map(v => v - peak);
        }
        resampledTargetMag = mag;
        ui.targetMag = mag;
        ui.targetPhase = resampled.phase;
        // b140.2.2 Fix 1: pass band's mag minus its refLevel to the
        // coherent sum, then add avgRef back globally. This matches
        // legacy renderSumMode and avoids double-counting refLevel
        // (combinedTargetMag already has refLevel baked in via
        // evaluate_target with reference_level_db += refLevel).
        const normMag = mag.map(v => v - r.refLevel);
        targetData.push({ mag: normMag, phase: resampled.phase, sign, delay });
      } else {
        targetData.push(null);
      }
    } else {
      targetData.push(null);
    }
    if (r.correctedMag) {
      const resampled = await resampleOntoGrid(
        r.freq, r.correctedMag, r.correctedPhase ?? null, freq,
        { extensionTargetMag, fallbackToTrend: true },
      );
      if (resampled.mag) {
        // b140.2.1.3 step B: per-band corrOffset. Legacy renderSumMode
        // shifts each band's corrected curve so its 200..2000 Hz
        // passband-average matches its target's. We do the same here so
        // the coherent / power-sum aggregation sees consistent SPL across
        // bands (parity with renderSumMode).
        let corrected = resampled.mag;
        if (resampledTargetMag) {
          let tAcc = 0, cAcc = 0, count = 0;
          for (let j = 0; j < freq.length; j++) {
            if (freq[j] < 200 || freq[j] > 2000) continue;
            const tv = resampledTargetMag[j];
            const cv = corrected[j];
            if (Number.isFinite(tv) && Number.isFinite(cv)) {
              tAcc += tv;
              cAcc += cv;
              count++;
            }
          }
          if (count > 0) {
            const offset = (tAcc - cAcc) / count;
            if (Math.abs(offset) > 0.01) {
              corrected = corrected.map(v => v + offset);
            }
          }
        }
        correctedMagsForFallback.push(corrected);
        ui.correctedMag = corrected;
        if (r.correctedPhase && resampled.phase) {
          ui.correctedPhase = resampled.phase;
          correctedData.push({ mag: corrected, phase: resampled.phase, sign, delay });
        } else {
          // Mag-only contributor — coherent sum cannot include it; mark to
          // trigger the power-sum fallback below.
          anyCorrectedMagWithoutPhase = true;
          correctedData.push(null);
        }
      } else {
        correctedData.push(null);
      }
    } else {
      correctedData.push(null);
    }
    if (r.measurementMag) {
      const resampled = await resampleOntoGrid(
        r.freq, r.measurementMag, r.measurementPhase ?? null, freq,
        { extensionTargetMag, fallbackToTrend: true },
      );
      if (resampled.mag) {
        measurementMagsForFallback.push(resampled.mag);
        ui.measurementMag = resampled.mag;
        if (r.measurementPhase && resampled.phase) {
          ui.measurementPhase = resampled.phase;
          measurementData.push({ mag: resampled.mag, phase: resampled.phase, sign, delay });
        } else {
          anyMeasurementMagWithoutPhase = true;
          measurementData.push(null);
        }
      } else {
        measurementData.push(null);
      }
    } else {
      measurementData.push(null);
    }
    perBandResampled.push(ui);
  }

  // 4. Sum. Power-sum fallback shared between corrected and measurement.
  const powerSum = (mags: number[][]): number[] => {
    const n = freq.length;
    const out = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (const m of mags) acc += Math.pow(10, (m[j] ?? -200) / 10);
      out[j] = acc > 0 ? 10 * Math.log10(acc) : -200;
    }
    return out;
  };

  // b140.2.2 Fix 1: avgRef = mean of enabled-target bands' refLevel.
  // Σ target adds avgRef back so its absolute SPL matches what each band's
  // target carries (legacy renderSumMode adds it post-coherent). Σ corrected
  // and Σ measurement do NOT — they live at their own measured SPL.
  let avgRef = 0;
  {
    let acc = 0, count = 0;
    for (let i = 0; i < bands.length; i++) {
      if (bands[i].targetEnabled && perBand[i].combinedTargetMag) {
        acc += perBand[i].refLevel;
        count++;
      }
    }
    if (count > 0) avgRef = acc / count;
  }

  const targetSum = coherentSum(freq, targetData, avgRef);

  let sumCorrectedMag: number[] | null;
  let sumCorrectedPhase: number[] | null;
  let coherent: boolean;
  if (correctedMagsForFallback.length === 0) {
    sumCorrectedMag = null;
    sumCorrectedPhase = null;
    coherent = true;
  } else if (anyCorrectedMagWithoutPhase) {
    sumCorrectedMag = powerSum(correctedMagsForFallback);
    sumCorrectedPhase = null;
    coherent = false;
    // b140.2.2 Fix 2: power-sum corrected loses absolute SPL relative to
    // target (per-band corrOffset doesn't apply — phase-less bands are
    // mag-only). Shift the entire curve so its mean matches target's in
    // the adaptive passband (target peak − 20 dB and above), where target
    // actually has signal. Mirrors legacy renderSumMode's offset block.
    if (sumCorrectedMag && targetSum?.mag) {
      let targetPeak = -Infinity;
      for (let j = 0; j < freq.length; j++) {
        const v = targetSum.mag[j];
        if (Number.isFinite(v) && v > targetPeak) targetPeak = v;
      }
      // Skip post-shift when the target sum is in a cancellation regime
      // (e.g. polarity-inverted bands) — pulling corrected down to a
      // degenerate target peak would destroy real signal. Real Σ Target
      // peaks are typically ≥ 0 dBr; a threshold of −50 dB cleanly
      // separates physical sums from cancellation residue.
      if (Number.isFinite(targetPeak) && targetPeak > -50) {
        const threshold = targetPeak - 20;
        let dSum = 0, dN = 0;
        for (let j = 0; j < freq.length; j++) {
          const t = targetSum.mag[j];
          const c = sumCorrectedMag[j];
          if (!Number.isFinite(t) || !Number.isFinite(c)) continue;
          if (t < threshold) continue;
          if (c < -150) continue;
          dSum += t - c;
          dN++;
        }
        if (dN > 0) {
          const off = dSum / dN;
          if (Math.abs(off) > 0.01) {
            sumCorrectedMag = sumCorrectedMag.map(v => v + off);
          }
        }
      }
    }
  } else {
    const cs = coherentSum(freq, correctedData);
    sumCorrectedMag = cs?.mag ?? null;
    sumCorrectedPhase = cs?.phase ?? null;
    coherent = true;
  }

  let sumMeasurementMag: number[] | null;
  let sumMeasurementPhase: number[] | null;
  let coherentMeasurement: boolean;
  if (measurementMagsForFallback.length === 0) {
    sumMeasurementMag = null;
    sumMeasurementPhase = null;
    coherentMeasurement = true;
  } else if (anyMeasurementMagWithoutPhase) {
    sumMeasurementMag = powerSum(measurementMagsForFallback);
    sumMeasurementPhase = null;
    coherentMeasurement = false;
  } else {
    const cs = coherentSum(freq, measurementData);
    sumMeasurementMag = cs?.mag ?? null;
    sumMeasurementPhase = cs?.phase ?? null;
    coherentMeasurement = true;
  }

  // 5. Optional IR for the corrected sum. Skipped when the power-sum
  //    fallback fires — there is no phase to feed into compute_impulse.
  let ir: SumEvalResult["ir"];
  if (options?.includeIr && coherent && sumCorrectedMag && sumCorrectedPhase) {
    try {
      const sr = bands.find(b => b.measurement)?.measurement?.sample_rate ?? 48000;
      const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
        "compute_impulse",
        { freq, magnitude: sumCorrectedMag, phase: sumCorrectedPhase, sampleRate: sr },
      );
      ir = { time: r.time, impulse: r.impulse, step: r.step };
    } catch (_) {}
  }

  // 6. b140.2.2 Fix 3: structured Σ IR. Separate grid from the SPL
  //    common grid: union of bands' native ranges, no 20-20k extension,
  //    ≥ 2048 points so the IFFT has enough resolution. Per-band: resample
  //    each measurement / target / corrected onto irFreq with target-shape
  //    extension + trend fallback, normalise to 0 dB peak, apply polarity
  //    + alignment_delay phase rotation, coherent-sum, then compute_impulse.
  let sumIr: SumEvalResult["sumIr"];
  if (options?.includeSumIr) {
    let irMin = Infinity, irMax = -Infinity, irPts = 2048;
    for (const r of perBand) {
      if (!r.measurementMag) continue;
      const f = r.freq;
      if (f.length === 0) continue;
      if (f[0] < irMin) irMin = f[0];
      if (f[f.length - 1] > irMax) irMax = f[f.length - 1];
      if (f.length > irPts) irPts = f.length;
    }
    if (Number.isFinite(irMin) && Number.isFinite(irMax)) {
      const irFreq = buildLogGrid(irPts, irMin, irMax);
      const sr = bands.find(b => b.measurement)?.measurement?.sample_rate ?? 48000;

      // Helper: per-band resample → 0 dB peak normalise → polarity + delay.
      const buildIrEntries = async (
        kind: "measurement" | "target" | "corrected",
      ): Promise<Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null>> => {
        const out: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null> = [];
        for (let i = 0; i < bands.length; i++) {
          const r = perBand[i];
          let srcMag: number[] | null = null;
          let srcPhase: number[] | null = null;
          if (kind === "measurement") {
            srcMag = r.measurementMag;
            srcPhase = r.measurementPhase ?? null;
          } else if (kind === "target") {
            if (!bands[i].targetEnabled) { out.push(null); continue; }
            srcMag = r.combinedTargetMag;
            srcPhase = r.combinedTargetPhase ?? null;
          } else {
            srcMag = r.correctedMag;
            srcPhase = r.correctedPhase ?? null;
          }
          if (!srcMag || !srcPhase) { out.push(null); continue; }
          // Build per-band target on irFreq for OOB extension.
          let extension: number[] | undefined;
          if (bands[i].targetEnabled) {
            const tc = bands[i].target;
            const tcWithRef = {
              ...tc,
              reference_level_db: (tc.reference_level_db ?? 0) + (globalRef ?? 0),
            };
            try {
              const resp = await invoke<TargetResponse>(
                "evaluate_target",
                { target: tcWithRef, freq: irFreq },
              );
              extension = resp.magnitude;
            } catch (_) {}
          }
          const resampled = await resampleOntoGrid(
            r.freq, srcMag, srcPhase, irFreq,
            { extensionTargetMag: extension, fallbackToTrend: true },
          );
          if (!resampled.mag || !resampled.phase) { out.push(null); continue; }
          // Normalise to 0 dB peak so each band contributes equally —
          // legacy SUM IR convention.
          let peak = -Infinity;
          for (const v of resampled.mag) if (v > peak) peak = v;
          const off = Number.isFinite(peak) ? -peak : 0;
          const normMag = resampled.mag.map(v => v + off);
          out.push({
            mag: normMag,
            phase: resampled.phase,
            sign: bands[i].inverted ? -1 : 1,
            delay: bands[i].alignmentDelay ?? 0,
          });
        }
        return out;
      };

      const computeImpulse = async (mag: number[], phase: number[]) => {
        try {
          const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
            "compute_impulse",
            { freq: irFreq, magnitude: mag, phase, sampleRate: sr },
          );
          return { time: r.time, impulse: r.impulse, step: r.step };
        } catch (_) {
          return undefined;
        }
      };

      const measEntries = await buildIrEntries("measurement");
      const tgtEntries = await buildIrEntries("target");
      const corrEntries = await buildIrEntries("corrected");
      const measSum = coherentSum(irFreq, measEntries);
      const tgtSum = coherentSum(irFreq, tgtEntries);
      const corrSum = coherentSum(irFreq, corrEntries);
      sumIr = {
        measurement: measSum ? await computeImpulse(measSum.mag, measSum.phase) : undefined,
        target: tgtSum ? await computeImpulse(tgtSum.mag, tgtSum.phase) : undefined,
        corrected: corrSum ? await computeImpulse(corrSum.mag, corrSum.phase) : undefined,
      };
    }
  }

  return {
    freq,
    perBand,
    sumTargetMag: targetSum?.mag ?? null,
    sumTargetPhase: targetSum?.phase ?? null,
    sumCorrectedMag,
    sumCorrectedPhase,
    sumMeasurementMag,
    sumMeasurementPhase,
    coherent,
    coherentMeasurement,
    perBandResampled,
    globalRef: globalRef ?? 0,
    avgRef,
    ir,
    sumIr,
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
