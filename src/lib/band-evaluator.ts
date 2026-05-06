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
// b140.3.0: minimal SUM aggregator — Σ Target only.
//
// Coherent sum of per-band target curves on a common log grid. No
// measurement, no corrected, no IR, no globalRef/avgRef magic. Levels
// come straight from each band's `target.reference_level_db`. The previous
// b140.2.x pipeline tried to mimic the legacy renderSumMode and accumulated
// 11 fix iterations without parity; this restart drops legacy mimicry and
// exposes Σ Target as a clean coherent sum. Σ Measurement / Σ Corrected /
// SUM IR will be added back in later sub-promts (b140.3.1, b140.3.2, …).
// ---------------------------------------------------------------------------
export interface SumEvalResult {
  freq: number[];
  /** Coherent sum of per-band target curves (HP × LP × shelves × tilt
   *  × subsonic, with Gaussian/subsonic min-phase reconstruction) with
   *  polarity and alignment_delay phase rotation. Null when no band has
   *  targetEnabled=true. */
  sumTargetMag: number[] | null;
  sumTargetPhase: number[] | null;
  /** Per-band target on the common grid (no polarity / alignment_delay
   *  applied — those affect only the Σ aggregate). One entry per input
   *  band, parallel to `bands`; null for bands where targetEnabled=false. */
  perBandTarget: Array<{ mag: number[]; phase: number[] } | null>;
  /** b140.3.1: Σ Corrected = Σ (measurement + PEQ + cross-section).
   *  Coherent sum when every contributing band has measurement phase;
   *  power-sum fallback otherwise (correctedCoherent=false). Null when
   *  no band has a measurement. */
  sumCorrectedMag: number[] | null;
  sumCorrectedPhase: number[] | null;
  /** Per-band corrected on the common grid (resampled with -200 dB / 0°
   *  fence outside native range — no extension). Null for bands without
   *  measurement. */
  perBandCorrected: Array<{ mag: number[]; phase: number[] } | null>;
  /** True iff Σ Corrected was computed via coherentSum (all bands had
   *  phase). False = power-sum fallback (no phase, no polarity). */
  correctedCoherent: boolean;
}

export interface SumEvalOptions {
  /** Override the common freq grid. */
  freq?: number[];
}

function buildLogGrid(n: number, fMin: number, fMax: number): number[] {
  const out = new Array(n);
  const lo = Math.log(fMin), hi = Math.log(fMax);
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(lo + (hi - lo) * i / (n - 1));
  }
  return out;
}

/** Common grid: union of band measurement ranges, 512 log-spaced points.
 *  Falls back to 5–40000 Hz when no band has a measurement. */
function buildCommonGrid(bands: BandState[]): number[] {
  let fMin = Infinity, fMax = -Infinity;
  for (const b of bands) {
    if (!b.measurement || b.measurement.freq.length === 0) continue;
    const f = b.measurement.freq;
    if (f[0] < fMin) fMin = f[0];
    if (f[f.length - 1] > fMax) fMax = f[f.length - 1];
  }
  if (!isFinite(fMin) || !isFinite(fMax)) {
    fMin = 5; fMax = 40000;
  }
  return buildLogGrid(512, fMin, fMax);
}

/** b140.3.1.5: linear interp on log-freq from src grid onto dst grid.
 *  Outside native range:
 *    • without `extensionTargetMag` → -200 dB / 0° fence (b140.2.1.4 fallback);
 *    • with `extensionTargetMag` → magnitude follows target shape with a
 *      boundary offset (`mag[boundary] − target[boundary]`); phase is
 *      reconstructed via Hilbert (compute_minimum_phase) on the extended
 *      magnitude with separate offsets at the low and high boundaries so
 *      the curve continues smoothly. */
async function resampleOntoCommon(
  srcFreq: number[],
  srcMag: number[],
  srcPhase: number[] | null,
  dstFreq: number[],
  options?: { extensionTargetMag?: number[] },
): Promise<{ mag: number[]; phase: number[] | null } | null> {
  if (srcFreq.length < 2) return null;
  const n = dstFreq.length;
  const fLo = srcFreq[0], fHi = srcFreq[srcFreq.length - 1];

  const nativeMag = new Array<number>(n);
  const nativePhase = srcPhase ? new Array<number>(n) : null;
  const inNative = new Array<boolean>(n);
  for (let k = 0; k < n; k++) {
    const f = dstFreq[k];
    if (f < fLo || f > fHi) {
      inNative[k] = false;
      nativeMag[k] = -200;
      if (nativePhase) nativePhase[k] = 0;
      continue;
    }
    inNative[k] = true;
    let lo = 0, hi = srcFreq.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (srcFreq[mid] <= f) lo = mid; else hi = mid;
    }
    const dt = srcFreq[hi] - srcFreq[lo];
    const frac = dt > 0 ? (f - srcFreq[lo]) / dt : 0;
    nativeMag[k] = srcMag[lo] + frac * (srcMag[hi] - srcMag[lo]);
    if (nativePhase && srcPhase) {
      nativePhase[k] = srcPhase[lo] + frac * (srcPhase[hi] - srcPhase[lo]);
    }
  }

  const tgt = options?.extensionTargetMag;
  if (!tgt) return { mag: nativeMag, phase: nativePhase };

  let idxLo = -1, idxHi = -1;
  for (let i = 0; i < n; i++) if (inNative[i]) { idxLo = i; break; }
  for (let i = n - 1; i >= 0; i--) if (inNative[i]) { idxHi = i; break; }
  if (idxLo < 0 || idxHi < 0) return { mag: nativeMag, phase: nativePhase };

  const magOffsetLo = nativeMag[idxLo] - tgt[idxLo];
  const magOffsetHi = nativeMag[idxHi] - tgt[idxHi];
  const extMag = new Array<number>(n);
  for (let k = 0; k < n; k++) {
    if (inNative[k]) extMag[k] = nativeMag[k];
    else if (k < idxLo) extMag[k] = tgt[k] + magOffsetLo;
    else extMag[k] = tgt[k] + magOffsetHi;
  }

  let extPhase: number[] | null = null;
  if (nativePhase) {
    const recon = await invoke<number[]>("compute_minimum_phase", {
      freq: dstFreq,
      magnitude: extMag,
    });
    const phOffsetLo = nativePhase[idxLo] - recon[idxLo];
    const phOffsetHi = nativePhase[idxHi] - recon[idxHi];
    extPhase = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      if (inNative[k]) extPhase[k] = nativePhase[k];
      else if (k < idxLo) extPhase[k] = recon[k] + phOffsetLo;
      else extPhase[k] = recon[k] + phOffsetHi;
    }
  }

  return { mag: extMag, phase: extPhase };
}

/** b140.3.1.4: global-shift excess control. The passband ± 1 octave zone
 *  is the *control window*: if a wide excess (≥ 1/2 oct) is found there,
 *  shift the entire corrected curve down so that excess collapses to the
 *  threshold. Narrow peaks (≤ 1/8 oct — room modes / natural resonances)
 *  are ignored. Soft transition between 1/8..1/2 oct. Magnitude only —
 *  phase untouched. */
function applyGlobalShiftIfWideExcess(
  freq: number[],
  corrected: number[],
  target: number[],
  hpFreqHz: number | null,
  lpFreqHz: number | null,
): number[] {
  const EXCESS_THRESHOLD = 0.1;
  const NARROW_OCT = 1 / 8;
  const WIDE_OCT = 1 / 2;

  const pbLow = hpFreqHz ? hpFreqHz * 1.5 : 20;
  const pbHigh = lpFreqHz ? lpFreqHz * 0.7 : 20000;
  const zoneLow = Math.max(20, pbLow / 2);
  const zoneHigh = Math.min(20000, pbHigh * 2);

  let regionStart = -1;
  let regionMaxExcess = 0;
  let maxRequiredShift = 0;

  const finalize = (start: number, end: number, maxEx: number) => {
    const f0 = freq[start];
    const f1 = freq[end];
    const widthOct = f1 > 0 && f0 > 0 ? Math.log2(f1 / f0) : 0;
    let factor: number;
    if (widthOct <= NARROW_OCT) factor = 0;
    else if (widthOct >= WIDE_OCT) factor = 1;
    else factor = (widthOct - NARROW_OCT) / (WIDE_OCT - NARROW_OCT);
    if (factor === 0) return;
    const effectiveExcess = maxEx * factor;
    const required = effectiveExcess - EXCESS_THRESHOLD;
    if (required > maxRequiredShift) maxRequiredShift = required;
  };

  for (let j = 0; j < freq.length; j++) {
    const inZone = freq[j] >= zoneLow && freq[j] <= zoneHigh;
    const ex = inZone && isFinite(corrected[j]) && isFinite(target[j])
      ? corrected[j] - target[j]
      : 0;
    const isExcess = inZone && ex > EXCESS_THRESHOLD;

    if (isExcess) {
      if (regionStart < 0) {
        regionStart = j;
        regionMaxExcess = ex;
      } else if (ex > regionMaxExcess) {
        regionMaxExcess = ex;
      }
    } else if (regionStart >= 0) {
      finalize(regionStart, j - 1, regionMaxExcess);
      regionStart = -1;
      regionMaxExcess = 0;
    }
  }
  if (regionStart >= 0) finalize(regionStart, freq.length - 1, regionMaxExcess);

  if (maxRequiredShift > 0) {
    return corrected.map(v => isFinite(v) ? v - maxRequiredShift : v);
  }
  return corrected;
}

/** Power sum of magnitudes in dB (no phase). Returns -200 dB for empty bins. */
function powerSumDb(magsDb: number[][]): number[] {
  if (magsDb.length === 0) return [];
  const n = magsDb[0].length;
  const out = new Array<number>(n);
  for (let j = 0; j < n; j++) {
    let acc = 0;
    for (const m of magsDb) acc += Math.pow(10, (m[j] ?? -200) / 10);
    out[j] = acc > 0 ? 10 * Math.log10(acc) : -200;
  }
  return out;
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
  const freq = options?.freq ?? buildCommonGrid(bands);

  const perBandTargetData: Array<
    { mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null
  > = [];
  const perBandTarget: Array<{ mag: number[]; phase: number[] } | null> = [];

  for (const band of bands) {
    if (!band.targetEnabled) {
      perBandTargetData.push(null);
      perBandTarget.push(null);
      continue;
    }
    const target = JSON.parse(JSON.stringify(band.target));
    const response = await invoke<TargetResponse>("evaluate_target", {
      target, freq,
    });
    const phase = await reconstructTargetPhase(
      freq, response.phase, band.target.high_pass, band.target.low_pass,
    );
    perBandTargetData.push({
      mag: response.magnitude,
      phase,
      sign: band.inverted ? -1 : 1,
      delay: band.alignmentDelay ?? 0,
    });
    perBandTarget.push({ mag: response.magnitude, phase });
  }

  const sum = coherentSum(freq, perBandTargetData);

  // b140.3.1: per-band corrected via evaluateBandFull (gives correctedMag /
  // correctedPhase with proper Gaussian/subsonic phase reconstruction). We
  // do NOT pass req.freq — evaluateBandFull builds correctedMag from
  // measurement.magnitude on measurement.freq, so we resample afterwards.
  const perBandResults = await Promise.all(
    bands.map(b => evaluateBandFull({ band: b })),
  );

  const perBandCorrected: Array<{ mag: number[]; phase: number[] } | null> = [];
  const correctedDataForSum: Array<
    { mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null
  > = [];
  let anyMissingPhase = false;
  let anyCorrected = false;

  for (let i = 0; i < bands.length; i++) {
    const r = perBandResults[i];
    if (!r.correctedMag) {
      perBandCorrected.push(null);
      correctedDataForSum.push(null);
      continue;
    }
    anyCorrected = true;
    // b140.3.1.5: extend corrected outside native via target shape +
    // Hilbert phase. Falls back to fence -200 dB when no target curve.
    const pbTargetForExt = perBandTarget[i];
    const resampled = await resampleOntoCommon(
      r.freq, r.correctedMag, r.correctedPhase ?? null, freq,
      pbTargetForExt ? { extensionTargetMag: pbTargetForExt.mag } : undefined,
    );
    if (!resampled) {
      perBandCorrected.push(null);
      correctedDataForSum.push(null);
      continue;
    }

    // b140.3.1.1: per-band normalize corrected → target in this band's
    // passband (HP·1.5 .. LP·0.7). Compensates for measurement-level
    // miscalibration so Σ Corrected sits at Σ Target in the playing zone.
    // Magnitude only — phase is untouched.
    let correctedMag = resampled.mag;
    const pbTarget = perBandTarget[i];
    if (pbTarget) {
      const hp = bands[i].target.high_pass;
      const lp = bands[i].target.low_pass;
      const pbLow = hp ? Math.max(20, hp.freq_hz * 1.5) : 20;
      const pbHigh = lp ? Math.min(20000, lp.freq_hz * 0.7) : 20000;
      const eL = pbLow < pbHigh ? pbLow : 200;
      const eH = pbLow < pbHigh ? pbHigh : 2000;
      let dSum = 0, dN = 0;
      for (let j = 0; j < freq.length; j++) {
        if (freq[j] < eL || freq[j] > eH) continue;
        const t = pbTarget.mag[j];
        const c = correctedMag[j];
        if (!isFinite(t) || !isFinite(c) || c < -150) continue;
        dSum += t - c;
        dN++;
      }
      if (dN > 0) {
        const offset = dSum / dN;
        if (Math.abs(offset) > 0.01) {
          correctedMag = correctedMag.map(v => v + offset);
        }
      }

      // b140.3.1.4: global shift if a wide excess is detected in the
      // passband ± 1 octave control zone. Whole curve moves uniformly;
      // narrow resonances stay intact.
      correctedMag = applyGlobalShiftIfWideExcess(
        freq, correctedMag, pbTarget.mag,
        bands[i].target.high_pass?.freq_hz ?? null,
        bands[i].target.low_pass?.freq_hz ?? null,
      );
    }

    const phaseArr = resampled.phase ?? new Array<number>(freq.length).fill(0);
    perBandCorrected.push({ mag: correctedMag, phase: phaseArr });
    if (!resampled.phase) anyMissingPhase = true;
    correctedDataForSum.push({
      mag: correctedMag,
      phase: phaseArr,
      sign: bands[i].inverted ? -1 : 1,
      delay: bands[i].alignmentDelay ?? 0,
    });
  }

  let sumCorrectedMag: number[] | null = null;
  let sumCorrectedPhase: number[] | null = null;
  let correctedCoherent = true;
  if (anyCorrected) {
    if (!anyMissingPhase) {
      const cs = coherentSum(freq, correctedDataForSum);
      sumCorrectedMag = cs?.mag ?? null;
      sumCorrectedPhase = cs?.phase ?? null;
    } else {
      const mags = correctedDataForSum
        .filter((d): d is NonNullable<typeof d> => d !== null)
        .map(d => d.mag);
      sumCorrectedMag = mags.length > 0 ? powerSumDb(mags) : null;
      sumCorrectedPhase = null;
      correctedCoherent = false;
    }
  }

  return {
    freq,
    sumTargetMag: sum?.mag ?? null,
    sumTargetPhase: sum?.phase ?? null,
    perBandTarget,
    sumCorrectedMag,
    sumCorrectedPhase,
    perBandCorrected,
    correctedCoherent,
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
