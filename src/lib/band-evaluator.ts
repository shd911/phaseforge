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
  /** b140.2.0.5: true → coherent corrected sum; false → power-sum fallback
   *  (sumCorrectedPhase will be null in that case). Always true when there
   *  is no corrected sum to compute. */
  coherent: boolean;
  /** Optional IR of the corrected sum (compute_impulse on summed mag/phase). */
  ir?: { time: number[]; impulse: number[]; step: number[] };
}

export interface SumEvalOptions {
  freq?: number[];
  includeIr?: boolean;
  /** When true the per-band combined target is normalised (peak = 0 dB) before
   *  the coherent sum — matches how renderSumMode treats target IR aggregation. */
  normalizeTargetPerBand?: boolean;
}

function buildLogGrid(n: number, fMin: number, fMax: number): number[] {
  const out = new Array(n);
  const lo = Math.log(fMin), hi = Math.log(fMax);
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(lo + (hi - lo) * i / (n - 1));
  }
  return out;
}

async function resampleOntoGrid(
  srcFreq: number[],
  srcMag: number[] | null,
  srcPhase: number[] | null,
  targetFreq: number[],
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
  return { mag, phase: phase ?? null };
}

function coherentSum(
  freq: number[],
  bandsData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null>,
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
  const mag = new Array(n);
  const phase = new Array(n);
  for (let j = 0; j < n; j++) {
    const amplitude = Math.sqrt(re[j] * re[j] + im[j] * im[j]);
    mag[j] = amplitude > 0 ? 20 * Math.log10(amplitude) : -200;
    phase[j] = Math.atan2(im[j], re[j]) * 180 / Math.PI;
  }
  return { mag, phase };
}

export async function evaluateSum(
  bands: BandState[],
  options?: SumEvalOptions,
): Promise<SumEvalResult> {
  // 1. Per-band evaluation on each band's native grid.
  const perBand = await Promise.all(bands.map(b => evaluateBandFull({ band: b })));

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

  // 3. Resample target + corrected onto the common grid for each band.
  const targetData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null> = [];
  const correctedData: Array<{ mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null> = [];
  // b140.2.0.5: collect per-band corrected magnitudes separately for the
  // power-sum fallback. The fallback is triggered when any band contributes
  // mag without phase — in that case ALL bands' mags participate, polarity
  // ignored.
  const correctedMagsForFallback: number[][] = [];
  let anyCorrectedMagWithoutPhase = false;
  for (let i = 0; i < bands.length; i++) {
    const r = perBand[i];
    const sign: 1 | -1 = bands[i].inverted ? -1 : 1;
    const delay = bands[i].alignmentDelay ?? 0;
    if (r.combinedTargetMag && r.combinedTargetPhase) {
      let tMag = r.combinedTargetMag;
      const tPhase = r.combinedTargetPhase;
      const resampled = await resampleOntoGrid(r.freq, tMag, tPhase, freq);
      if (resampled.mag && resampled.phase) {
        let mag = resampled.mag;
        if (options?.normalizeTargetPerBand) {
          let peak = -Infinity;
          for (const v of mag) if (v > peak) peak = v;
          mag = mag.map(v => v - peak);
        }
        targetData.push({ mag, phase: resampled.phase, sign, delay });
      } else {
        targetData.push(null);
      }
    } else {
      targetData.push(null);
    }
    if (r.correctedMag) {
      const resampled = await resampleOntoGrid(
        r.freq, r.correctedMag, r.correctedPhase ?? null, freq,
      );
      if (resampled.mag) {
        correctedMagsForFallback.push(resampled.mag);
        if (r.correctedPhase && resampled.phase) {
          correctedData.push({ mag: resampled.mag, phase: resampled.phase, sign, delay });
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
  }

  // 4. Sum.
  const targetSum = coherentSum(freq, targetData);
  let sumCorrectedMag: number[] | null;
  let sumCorrectedPhase: number[] | null;
  let coherent: boolean;
  if (correctedMagsForFallback.length === 0) {
    sumCorrectedMag = null;
    sumCorrectedPhase = null;
    coherent = true;
  } else if (anyCorrectedMagWithoutPhase) {
    // Power sum: amp²[j] = Σ 10^(m_i[j] / 10) → 10·log10. Polarity ignored,
    // phase undefined. Matches legacy renderSumMode mixed-phase behaviour.
    const n = freq.length;
    const mag = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (const m of correctedMagsForFallback) {
        acc += Math.pow(10, (m[j] ?? -200) / 10);
      }
      mag[j] = acc > 0 ? 10 * Math.log10(acc) : -200;
    }
    sumCorrectedMag = mag;
    sumCorrectedPhase = null;
    coherent = false;
  } else {
    const cs = coherentSum(freq, correctedData);
    sumCorrectedMag = cs?.mag ?? null;
    sumCorrectedPhase = cs?.phase ?? null;
    coherent = true;
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

  return {
    freq,
    perBand,
    sumTargetMag: targetSum?.mag ?? null,
    sumTargetPhase: targetSum?.phase ?? null,
    sumCorrectedMag,
    sumCorrectedPhase,
    coherent,
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
