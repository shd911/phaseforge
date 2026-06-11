/**
 * b140.14.4 — Phase 4 slice 5 (final): single-band evaluator.
 *
 * Houses `evaluateBandFull` — the canonical per-band evaluator that
 * builds magnitude / phase / cross-section / corrected curves, runs
 * optional FIR generation (via `dispatchFirInvoke`), and packages the
 * full result envelope. Also owns the supporting types:
 * FirRequestConfig, BandEvalRequest, BandEvalResult, plus the
 * `reconstructTargetPhase` and `applyMeasurementSmoothing` helpers
 * shared with `evaluateSum` in sum.ts.
 *
 * Behaviour byte-identical to the pre-b140.14.4 in-file definition —
 * locked down by golden_sum + filter-clone + routing-decision baselines.
 */
import { invoke } from "@tauri-apps/api/core";
import type { FilterConfig, Measurement, PeqBand, TargetResponse } from "../types";
import type { BandState } from "../../stores/bands";
import {
  isGaussianMinPhase,
  gaussianFilterMagDb,
  subsonicMagDb,
  smoothingConfig,
} from "../plot-helpers";
import { hasActiveSubsonicProtect } from "../types";
import { buildLogGrid, buildCommonGrid, resampleOnLogGrid } from "./grid";
import { dispatchFirInvoke } from "./route";
import { appendNoiseFloorTail, autoRefLevel, computeExtension } from "./extension";

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
  /** b141.5 (audit): sample rate the realized biquads run at (= export
   *  sample rate). Threaded into every compute_peq_complex call so the
   *  displayed/optimized PEQ matches the realized IIR/FIR response near
   *  Nyquist. Falls back to fir.sampleRate, then 48 kHz. */
  sampleRate?: number;
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

  /** b140.3.2: standard wide grid (20–20000 Hz, 512 log) used for extension.
   *  Populated when measurement+targetEnabled; null otherwise. */
  extendedFreq: number[] | null;
  /** Original measurement frequency bounds [fLo, fHi]. UI uses this to
   *  separate "real" data from synthesized extension visually. */
  nativeRange: [number, number] | null;
  /** Measurement extended onto extendedFreq via target shape + boundary
   *  offset (magnitude) and Hilbert + boundary offset (phase). Full
   *  coverage of extendedFreq — caller masks via nativeRange when needed. */
  extendedMeasurementMag: number[] | null;
  extendedMeasurementPhase: number[] | null;
  /** Same extension applied to corrected (measurement+PEQ+cross-section). */
  extendedCorrectedMag: number[] | null;
  extendedCorrectedPhase: number[] | null;

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
// Gaussian × subsonic combinations. b139.4c: replaces the legacy
// addGaussianMinPhase (since deleted from band-evaluation.ts) with one
// callable that works on any (freq, basePhase) pair regardless of caller.
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

// b140.14: `resampleOnLogGrid` moved to ./band-evaluator/grid.ts and
// re-imported above. Behaviour unchanged.

// b140.14.2: `appendNoiseFloorTail` and `autoRefLevel` moved to
// ./band-evaluator/extension.ts and re-imported above. Behaviour unchanged.

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
  // b141.5: biquads are NOT sample-rate independent (bilinear warp near
  // Nyquist) — every PEQ evaluation must run at the realization rate.
  const peqSampleRate = req.sampleRate ?? req.fir?.sampleRate ?? 48000;
  let peqMag: number[] = new Array(freq.length).fill(0);
  let peqPhase: number[] = new Array(freq.length).fill(0);
  if (enabledPeq.length > 0) {
    const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
      freq, bands: enabledPeq, sampleRate: peqSampleRate,
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
    } catch (e) {
      // No filters → leave null; corrected = meas + PEQ alone. Most
      // common cause: HP/LP both null. Logged so genuine compute
      // failures (e.g. malformed config) surface in console.
      console.warn("[evaluateBandFull] compute_cross_section failed:", e);
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
    // b141.2 (audit): AND-collapse is intentional. A single FIR cannot realise a
    // mixed-phase main (HP linear + LP min) — physical limitation, see CLAUDE.md.
    // So "both linear" → linear FIR, otherwise → min-phase IIR path. A user who
    // sets only one crossover to linear via the per-block toggle gets a min-phase
    // FIR (the displayed per-filter target may then differ from the realised FIR
    // for that filter). Left as-is by product decision — this combination is not
    // a supported configuration.
    const linearMain =
      isUserLin(band.target.high_pass) && isUserLin(band.target.low_pass);
    const hp = band.target.high_pass;
    const subsonicCutoff = hasActiveSubsonicProtect(hp) ? hp!.freq_hz / 8 : null;
    const cfg = req.fir;

    // FIR-specific grid + target evaluation.
    const fMaxFir = Math.min(40000, cfg.sampleRate / 2 * 0.95);
    const [firFreqRaw, firResp] = await invoke<[number[], TargetResponse]>(
      "evaluate_target_standalone",
      { target: targetCurve, nPoints: 512, fMin: 5, fMax: fMaxFir },
    );
    const firTargetPhaseRaw = await reconstructTargetPhase(
      firFreqRaw, firResp.phase, band.target.high_pass, band.target.low_pass,
    );

    // b140.5: extend log grid + target trio up to Nyquist with a noise-floor
    // tail to avoid Rust's constant boundary clamp on linear FFT bins above
    // fMaxFir (was shifting apparent rolloff by ~½ oct at sr=44.1/48 kHz).
    const firExt = appendNoiseFloorTail(
      firFreqRaw, firResp.magnitude, firTargetPhaseRaw,
      cfg.sampleRate, cfg.noiseFloorDb,
    );
    const firFreq = firExt.freq;
    const firTargetMag = firExt.mag;
    const firTargetPhase = firExt.phase;

    // PEQ on the FIR grid, at the FIR's sample rate (b141.5: biquads warp
    // near Nyquist — the baked-in PEQ curve must match the realized rate).
    let firPeqMag: number[] = new Array(firFreq.length).fill(0);
    let firPeqPhase: number[] = new Array(firFreq.length).fill(0);
    if (enabledPeq.length > 0) {
      const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq: firFreq, bands: enabledPeq, sampleRate: cfg.sampleRate,
      });
      firPeqMag = pm;
      firPeqPhase = pp;
    }
    const firCombinedPhase = firTargetPhase.map((p, i) => p + firPeqPhase[i]);

    // b140.14.1: routing predicate + Rust-side dispatch extracted to
    // ./band-evaluator/route.ts so the JS-side dispatch surface (decision +
    // Tauri payload mapping) has a single home. Behaviour unchanged.
    const result = await dispatchFirInvoke(
      band.target.high_pass,
      band.target.low_pass,
      enabledPeq,
      linearMain,
      subsonicCutoff,
      firFreq, firTargetMag, firPeqMag, firCombinedPhase,
      cfg,
    );
    // b140.6: realized_mag/phase come back on `firFreq` (5..fMaxFir, where
    // fMaxFir = min(40000, sr·0.95/2)). At sr=44.1/48 kHz this is < 40 kHz,
    // so its 512 points compress 0..fMaxFir while `freq` (the caller-side
    // grid that the SPL/Export plot uses) covers up to 40 kHz. Plotting
    // them positionally on a single x-axis shifted FIR by ~0.8 oct on
    // rolloff. Resample onto `freq` here so every consumer gets FIR on the
    // same grid as Model.
    const realizedMagOnFreq = resampleOnLogGrid(firFreq, result.realized_mag, freq);
    const realizedPhaseOnFreq = resampleOnLogGrid(firFreq, result.realized_phase, freq);
    fir = {
      impulse: result.impulse,
      // b141.6: ramp derived locally — was a ~MB linear array in the payload.
      timeMs: Array.from({ length: result.impulse.length }, (_, i) => i * 1000 / result.sample_rate),
      realizedMag: realizedMagOnFreq,
      realizedPhase: realizedPhaseOnFreq,
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
      } catch (e) {
        console.warn("[evaluateBandFull] measurement compute_impulse failed:", e);
      }
    }
    // b140.3.3 + b140.3.4: target and corrected IR on a wide standalone
    // grid (5 Hz – min(40 kHz, Nyquist·0.95)) instead of measurement.freq.
    // Target is a model — its impulse must reflect rolloff outside the
    // measurement range too, otherwise subsonic protect (active at HP/8 ≈
    // 5–80 Hz) and supersonic shaping vanish from the rendered IR.
    // Corrected reuses the same wide grid: measurement is extended via
    // target shape + Hilbert phase (computeExtension), then PEQ and
    // cross-section are recomputed on the wide grid before convolution.
    if (band.targetEnabled) {
      try {
        const irFMax = Math.min(40000, sr / 2 * 0.95);
        const irFreqRaw = buildLogGrid(512, 5, irFMax);
        const irTargetCurve = {
          ...JSON.parse(JSON.stringify(targetCurve)),
          reference_level_db: (targetCurve.reference_level_db ?? 0) + refLevel,
        };
        const irTargetResp = await invoke<TargetResponse>("evaluate_target", {
          target: irTargetCurve, freq: irFreqRaw,
        });
        const irTargetPhaseRaw = await reconstructTargetPhase(
          irFreqRaw, irTargetResp.phase, band.target.high_pass, band.target.low_pass,
        );
        // b140.5: extend up to Nyquist with noise-floor tail to neutralise the
        // Rust constant-clamp on linear FFT bins above irFMax.
        const irExt = appendNoiseFloorTail(
          irFreqRaw, irTargetResp.magnitude, irTargetPhaseRaw, sr,
        );
        const irFreq = irExt.freq;
        const irTargetMag = irExt.mag;
        const irTargetPhase = irExt.phase;
        const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
          "compute_impulse",
          { freq: irFreq, magnitude: irTargetMag, phase: irTargetPhase, sampleRate: sr },
        );
        ir.target = { time: r.time, impulse: r.impulse, step: r.step };

        // b140.3.4: corrected IR on the same wide grid.
        if (measurement) {
          const extMeas = await computeExtension(
            measurement.freq, measurement.magnitude,
            measurement.phase ?? null, irFreq, irTargetMag,
          );

          let irPeqMag: number[] = new Array(irFreq.length).fill(0);
          let irPeqPhase: number[] = new Array(irFreq.length).fill(0);
          if (enabledPeq.length > 0) {
            const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
              freq: irFreq, bands: enabledPeq, sampleRate: peqSampleRate,
            });
            irPeqMag = pm; irPeqPhase = pp;
          }

          let irXsMag: number[] = new Array(irFreq.length).fill(0);
          let irXsPhase: number[] = new Array(irFreq.length).fill(0);
          if (band.target.high_pass || band.target.low_pass) {
            try {
              const [xm, xp] = await invoke<[number[], number[], number]>(
                "compute_cross_section",
                { freq: irFreq, highPass: band.target.high_pass, lowPass: band.target.low_pass },
              );
              irXsMag = xm; irXsPhase = xp;
            } catch (e) {
              console.warn("[evaluateBandFull] IR compute_cross_section failed (leaving zeros):", e);
            }
          }

          const irCorrMag = extMeas.mag.map((m, i) => m + irPeqMag[i] + irXsMag[i]);
          const basePhase = (extMeas.phase ?? new Array<number>(irFreq.length).fill(0))
            .map((p, i) => p + irPeqPhase[i] + irXsPhase[i]);
          const irCorrPhase = await reconstructTargetPhase(
            irFreq, basePhase, band.target.high_pass, band.target.low_pass,
          );
          const cr = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
            "compute_impulse",
            { freq: irFreq, magnitude: irCorrMag, phase: irCorrPhase, sampleRate: sr },
          );
          ir.corrected = { time: cr.time, impulse: cr.impulse, step: cr.step };
        }
      } catch (e) {
        console.warn("[evaluateBandFull] target/corrected IR pipeline failed:", e);
      }
    }
  }

  // 8. b140.3.2: extension block. Single source of truth for "what does the
  //    measurement / corrected look like outside the native range?". When
  //    the band has measurement + targetEnabled we evaluate target on a
  //    standard wide grid and extend both measurement and corrected via
  //    target shape + boundary offset (magnitude) and Hilbert reconstruction
  //    + boundary offsets (phase). evaluateSum and the band view both read
  //    from these fields — no extension logic anywhere else.
  let extendedFreq: number[] | null = null;
  let nativeRange: [number, number] | null = null;
  let extendedMeasurementMag: number[] | null = null;
  let extendedMeasurementPhase: number[] | null = null;
  let extendedCorrectedMag: number[] | null = null;
  let extendedCorrectedPhase: number[] | null = null;

  if (measurement && band.targetEnabled) {
    extendedFreq = buildLogGrid(512, 20, 20000);
    nativeRange = [measurement.freq[0], measurement.freq[measurement.freq.length - 1]];

    // Target on extendedFreq for extension shape — mirrors the level
    // adjustment the measurement-grid target eval used above so the boundary
    // offset compensates for any constant level mismatch.
    const tgtCurveExt = { ...targetCurve, reference_level_db: targetCurve.reference_level_db + refLevel };
    const tgtRespExt = await invoke<TargetResponse>("evaluate_target", {
      target: tgtCurveExt, freq: extendedFreq,
    });
    const tgtMagExt = tgtRespExt.magnitude;

    const measExt = await computeExtension(
      measurement.freq, measurement.magnitude,
      measurement.phase ?? null, extendedFreq, tgtMagExt,
    );
    extendedMeasurementMag = measExt.mag;
    extendedMeasurementPhase = measExt.phase;

    if (correctedMag) {
      const corrExt = await computeExtension(
        measurement.freq, correctedMag, correctedPhase ?? null,
        extendedFreq, tgtMagExt,
      );
      extendedCorrectedMag = corrExt.mag;
      extendedCorrectedPhase = corrExt.phase;
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
    extendedFreq,
    nativeRange,
    extendedMeasurementMag,
    extendedMeasurementPhase,
    extendedCorrectedMag,
    extendedCorrectedPhase,
    fir,
    ir,
  };
}
