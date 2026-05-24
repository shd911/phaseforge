/**
 * b140.14.3 — Phase 4 slice 4: multi-band SUM aggregator.
 *
 * Carries `evaluateSum` and its four sum-only helpers
 * (`resampleOntoCommon`, `applyGlobalShiftIfWideExcess`, `powerSumDb`,
 * `coherentSum`) out of band-evaluator.ts. The SumEvalResult and
 * SumEvalOptions interfaces move with them.
 *
 * No cycles: this module imports `evaluateBandFull` and
 * `reconstructTargetPhase` directly from `./evaluate` (sibling). The
 * parent `band-evaluator.ts` re-exports `evaluateSum` for backward
 * compat but isn't imported back here.
 */
import { invoke } from "@tauri-apps/api/core";
import type { BandState } from "../../stores/bands";
import type { PeqBand, TargetResponse } from "../types";
import { buildCommonGrid, buildLogGrid } from "./grid";
import { appendNoiseFloorTail, computeExtension } from "./extension";
import { evaluateBandFull, reconstructTargetPhase } from "./evaluate";

// ---------------------------------------------------------------------------
// Public types
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
  /** b140.3.5: Σ IR/Step results (only set when options.includeIr=true).
   *  Each curve is a coherent sum on a wide IR grid (5 Hz – Nyquist·0.95)
   *  with polarity + alignment_delay applied per band, converted to time
   *  domain via compute_impulse. */
  ir?: {
    measurement?: { time: number[]; impulse: number[]; step: number[] };
    target?: { time: number[]; impulse: number[]; step: number[] };
    corrected?: { time: number[]; impulse: number[]; step: number[] };
  };
}

export interface SumEvalOptions {
  /** Override the common freq grid. */
  freq?: number[];
  /** b140.3.5: also compute Σ IR/Step (measurement, target, corrected) by
   *  coherent-summing per-band responses on a wide IR grid (5 Hz – Nyquist·0.95)
   *  in the frequency domain, then taking compute_impulse of the result. */
  includeIr?: boolean;
}

// ---------------------------------------------------------------------------
// Sum-only helpers
// ---------------------------------------------------------------------------

/** b140.3.2: linear interp on log-freq from src grid onto dst grid with a
 *  -200 dB / 0° fence outside the source range (no extension). Extension is
 *  now handled in evaluateBandFull via `computeExtension`. */
function resampleOntoCommon(
  srcFreq: number[],
  srcMag: number[],
  srcPhase: number[] | null,
  dstFreq: number[],
): { mag: number[]; phase: number[] | null } | null {
  if (srcFreq.length < 2) return null;
  const n = dstFreq.length;
  const fLo = srcFreq[0], fHi = srcFreq[srcFreq.length - 1];
  const mag = new Array<number>(n);
  const phase = srcPhase ? new Array<number>(n) : null;
  for (let k = 0; k < n; k++) {
    const f = dstFreq[k];
    if (f < fLo || f > fHi) {
      mag[k] = -200;
      if (phase) phase[k] = 0;
      continue;
    }
    let lo = 0, hi = srcFreq.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (srcFreq[mid] <= f) lo = mid; else hi = mid;
    }
    const dt = srcFreq[hi] - srcFreq[lo];
    const frac = dt > 0 ? (f - srcFreq[lo]) / dt : 0;
    mag[k] = srcMag[lo] + frac * (srcMag[hi] - srcMag[lo]);
    if (phase && srcPhase) {
      phase[k] = srcPhase[lo] + frac * (srcPhase[hi] - srcPhase[lo]);
    }
  }
  return { mag, phase };
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

/** Threshold below which we treat re/im as numerically zero and report
 *  the phase as NaN (signaling "phase is undefined here", typically a
 *  cancellation null). Without this, atan2 of two near-zero floats
 *  returns ulp-grid-quantized values that jump ±180° across bins —
 *  visually as jaggedness around crossover nulls.
 *
 *  Conservative threshold: 1e-15 ≈ 6 ulps for double-precision
 *  amplitudes near 1.0. Below it both re and im are pure rounding
 *  noise from accumulating cos/sin of large delay·freq products. */
const COHERENT_SUM_NULL_FLOOR = 1e-15;

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
      // b140.15.6: reduce delay·freq mod 1 cycle BEFORE trig to keep
      // the argument small. Math.cos/sin precision degrades with
      // argument magnitude — at f=20kHz × delay=50ms the raw argument
      // is 6283 rad and the absolute error in the result reaches ulp
      // size relative to that, which dominates the result at deep
      // crossover nulls where re,im → 0. Reducing modulo 2π preserves
      // the periodic value but keeps the trig arg in [-π, π].
      const phaseDeg = (b.phase[j] ?? 0) + 360 * freq[j] * b.delay;
      const phaseDegMod = phaseDeg - 360 * Math.round(phaseDeg / 360);
      const phRad = phaseDegMod * Math.PI / 180;
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
    // b140.15.6: report phase as 0 (not random atan2 ulp) at deep
    // cancellation nulls where re,im are numerical noise.
    phase[j] = amplitude > COHERENT_SUM_NULL_FLOOR
      ? Math.atan2(im[j], re[j]) * 180 / Math.PI
      : 0;
  }
  return { mag, phase };
}

// ---------------------------------------------------------------------------
// Public entry — evaluateSum
// ---------------------------------------------------------------------------
//
// Expected SUM characteristics (do not "fix" — these are LR / DSP properties):
//
//   1. ~0.1–0.5 dB smooth dip in sumCorrectedMag around each crossover
//      frequency in 3-band LR4 in-phase configurations. The middle band
//      has both HP+LP, and its LP contributes a small phase tail at the
//      lower crossover (and HP a tail at the upper one) that prevents
//      perfect coherent-sum complementarity. Verified bit-for-bit against
//      the analytical sum BW²(LP) + BW²(HP) × BW²(LP_outer): the dip IS
//      the expected mathematical residual, not an implementation bug.
//      Inverting alternate band polarity or adding all-pass phase
//      compensation removes it (DSP-design choice).
//
//   2. Phase wraps of sumCorrectedPhase at the crossover frequencies are
//      ±180° smooth single-bin transitions in the WRAPPED display (the
//      underlying complex value is continuous). The narrow ~120° spikes
//      reported in 2026-05-24 / b140.15.8–.9 audit were a scalar-phase-
//      summation bug in apply_filter; complex accumulator path fixed
//      them. If new narrow spikes ever reappear, run:
//         cargo test --test sum_3band_lr4_flat
//         npx vitest run src/lib/__tests__/sum-3band-lr4-spikes.test.ts

export async function evaluateSum(
  bands: BandState[],
  options?: SumEvalOptions,
): Promise<SumEvalResult> {
  const freq = options?.freq ?? buildCommonGrid(bands);

  // b140.15.4: per-band target eval runs in parallel (was serial for...of
  // with await inside — ~N × IPC latency stacked). Bands are independent so
  // Promise.all is safe; coherentSum below still requires the deterministic
  // band order, hence indexed Array.from instead of a plain map.
  //
  // Note: per-band target uses raw `band.target` (no refLevel shift) by
  // design — perBandTarget represents the user's *intended* SPL while
  // evaluateBandFull's internal target (with refLevel) is what perBandCorrected
  // normalizes onto. Audit Logic-#3 proposed unifying them; tests confirm
  // they MUST stay separate or per-band-corrected normalize stops working.
  const perBandTargetData: Array<
    { mag: number[]; phase: number[]; sign: 1 | -1; delay: number } | null
  > = new Array(bands.length).fill(null);
  const perBandTarget: Array<{ mag: number[]; phase: number[] } | null> =
    new Array(bands.length).fill(null);

  await Promise.all(bands.map(async (band, i) => {
    if (!band.targetEnabled) return;
    const target = JSON.parse(JSON.stringify(band.target));
    const response = await invoke<TargetResponse>("evaluate_target", {
      target, freq,
    });
    const phase = await reconstructTargetPhase(
      freq, response.phase, band.target.high_pass, band.target.low_pass,
    );
    perBandTargetData[i] = {
      mag: response.magnitude,
      phase,
      sign: band.inverted ? -1 : 1,
      delay: band.alignmentDelay ?? 0,
    };
    perBandTarget[i] = { mag: response.magnitude, phase };
  }));

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
    // b140.3.2: prefer the extended-onto-wide-grid corrected from
    // evaluateBandFull (single source of truth). Fall back to native
    // corrected (with -200 dB fence outside) when no extension exists.
    const sourceFreq = r.extendedFreq ?? r.freq;
    const sourceMag = r.extendedCorrectedMag ?? r.correctedMag;
    const sourcePhase = r.extendedCorrectedPhase ?? r.correctedPhase ?? null;
    const resampled = resampleOntoCommon(
      sourceFreq, sourceMag, sourcePhase, freq,
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

  // b140.3.5: Σ IR/Step. Coherent-sum per-band responses on a wide IR grid in
  // the frequency domain (alignment_delay → phase rotation, polarity → sign),
  // then compute_impulse for each category.
  let irOut: SumEvalResult["ir"];
  if (options?.includeIr) {
    const irSr = Math.max(48000, ...bands.map(b => b.measurement?.sample_rate ?? 48000));
    const irFMax = Math.min(40000, irSr / 2 * 0.95);
    const irFreqBase = buildLogGrid(1024, 5, irFMax);
    // b140.5: extend up to Nyquist with explicit silent tail. The dummy
    // arrays satisfy the helper signature; only freq is reused for SUM IR
    // since per-band target / measurement / corrected get re-evaluated on
    // the extended grid below.
    const irFreq = appendNoiseFloorTail(
      irFreqBase, new Array(irFreqBase.length).fill(0), new Array(irFreqBase.length).fill(0),
      irSr,
    ).freq;
    const N = irFreq.length;

    const tgtRe = new Float64Array(N), tgtIm = new Float64Array(N);
    const measRe = new Float64Array(N), measIm = new Float64Array(N);
    const corrRe = new Float64Array(N), corrIm = new Float64Array(N);
    let anyTgt = false, anyMeas = false, anyCorr = false;

    // b140.15.4: per-band IR build runs in parallel. Each band computes
    // its own (tgt|meas|corr) Re/Im partials, then the main thread reduces
    // them into the shared accumulator. Math is identical — accumulation
    // is associative on f64 within the precision we care about.
    interface IrPartial {
      tgt: { re: Float64Array; im: Float64Array } | null;
      meas: { re: Float64Array; im: Float64Array } | null;
      corr: { re: Float64Array; im: Float64Array } | null;
    }

    const partials: IrPartial[] = await Promise.all(bands.map(async (band): Promise<IrPartial> => {
      const sign: 1 | -1 = band.inverted ? -1 : 1;
      const delay = band.alignmentDelay ?? 0;
      const out: IrPartial = { tgt: null, meas: null, corr: null };

      // Target on irFreq (also reused as extension shape for measurement / corrected).
      let tgtMagOnIr: number[] | null = null;
      if (band.targetEnabled) {
        const target = JSON.parse(JSON.stringify(band.target));
        const resp = await invoke<TargetResponse>("evaluate_target", {
          target, freq: irFreq,
        });
        const tPhase = await reconstructTargetPhase(
          irFreq, resp.phase, band.target.high_pass, band.target.low_pass,
        );
        tgtMagOnIr = resp.magnitude;
        const re = new Float64Array(N), im = new Float64Array(N);
        for (let j = 0; j < N; j++) {
          const amp = Math.pow(10, (resp.magnitude[j] ?? -200) / 20) * sign;
          const phRad = ((tPhase[j] ?? 0) + 360 * irFreq[j] * delay) * Math.PI / 180;
          re[j] = amp * Math.cos(phRad);
          im[j] = amp * Math.sin(phRad);
        }
        out.tgt = { re, im };
      }

      if (!band.measurement) return out;

      // Measurement on irFreq: extension via target shape when target is on,
      // else fence outside native range.
      let extMeasMag: number[] | null = null;
      let extMeasPhase: number[] | null = null;
      if (tgtMagOnIr) {
        const ext = await computeExtension(
          band.measurement.freq, band.measurement.magnitude,
          band.measurement.phase ?? null, irFreq, tgtMagOnIr,
        );
        extMeasMag = ext.mag;
        extMeasPhase = ext.phase;
      } else {
        const r = resampleOntoCommon(
          band.measurement.freq, band.measurement.magnitude,
          band.measurement.phase ?? null, irFreq,
        );
        extMeasMag = r?.mag ?? null;
        extMeasPhase = r?.phase ?? null;
      }
      if (!extMeasMag) return out;
      const measPhaseArr = extMeasPhase ?? new Array<number>(N).fill(0);

      {
        const re = new Float64Array(N), im = new Float64Array(N);
        for (let j = 0; j < N; j++) {
          const amp = Math.pow(10, (extMeasMag[j] ?? -200) / 20) * sign;
          const phRad = ((measPhaseArr[j] ?? 0) + 360 * irFreq[j] * delay) * Math.PI / 180;
          re[j] = amp * Math.cos(phRad);
          im[j] = amp * Math.sin(phRad);
        }
        out.meas = { re, im };
      }

      // Corrected: measurement (extended) + PEQ + cross-section, on irFreq.
      if (band.targetEnabled) {
        const enabledPeq = (band.peqBands ?? []).filter((p: PeqBand) => p.enabled);
        let irPeqMag: number[] = new Array(N).fill(0);
        let irPeqPhase: number[] = new Array(N).fill(0);
        if (enabledPeq.length > 0) {
          const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
            freq: irFreq, bands: enabledPeq,
          });
          irPeqMag = pm; irPeqPhase = pp;
        }
        let irXsMag: number[] = new Array(N).fill(0);
        let irXsPhase: number[] = new Array(N).fill(0);
        if (band.target.high_pass || band.target.low_pass) {
          try {
            const [xm, xp] = await invoke<[number[], number[], number]>(
              "compute_cross_section",
              { freq: irFreq, highPass: band.target.high_pass, lowPass: band.target.low_pass },
            );
            irXsMag = xm; irXsPhase = xp;
          } catch (e) {
            console.warn("[evaluateSum] IR compute_cross_section failed (leaving zeros):", e);
          }
        }
        const corrMag = extMeasMag.map((m, j) => m + irPeqMag[j] + irXsMag[j]);
        const baseP = measPhaseArr.map((p, j) => p + irPeqPhase[j] + irXsPhase[j]);
        const corrPhase = await reconstructTargetPhase(
          irFreq, baseP, band.target.high_pass, band.target.low_pass,
        );
        const re = new Float64Array(N), im = new Float64Array(N);
        for (let j = 0; j < N; j++) {
          const amp = Math.pow(10, (corrMag[j] ?? -200) / 20) * sign;
          const phRad = ((corrPhase[j] ?? 0) + 360 * irFreq[j] * delay) * Math.PI / 180;
          re[j] = amp * Math.cos(phRad);
          im[j] = amp * Math.sin(phRad);
        }
        out.corr = { re, im };
      }

      return out;
    }));

    for (const p of partials) {
      if (p.tgt) {
        anyTgt = true;
        for (let j = 0; j < N; j++) { tgtRe[j] += p.tgt.re[j]; tgtIm[j] += p.tgt.im[j]; }
      }
      if (p.meas) {
        anyMeas = true;
        for (let j = 0; j < N; j++) { measRe[j] += p.meas.re[j]; measIm[j] += p.meas.im[j]; }
      }
      if (p.corr) {
        anyCorr = true;
        for (let j = 0; j < N; j++) { corrRe[j] += p.corr.re[j]; corrIm[j] += p.corr.im[j]; }
      }
    }

    irOut = {};
    const toIR = async (re: Float64Array, im: Float64Array) => {
      const mag = new Array<number>(N);
      const phase = new Array<number>(N);
      for (let j = 0; j < N; j++) {
        const amp = Math.sqrt(re[j] * re[j] + im[j] * im[j]);
        mag[j] = amp > 0 ? 20 * Math.log10(amp) : -200;
        phase[j] = Math.atan2(im[j], re[j]) * 180 / Math.PI;
      }
      try {
        const r = await invoke<{ time: number[]; impulse: number[]; step: number[] }>(
          "compute_impulse",
          { freq: irFreq, magnitude: mag, phase, sampleRate: irSr },
        );
        return { time: r.time, impulse: r.impulse, step: r.step };
      } catch (e) {
        console.warn("[evaluateSum] toIR compute_impulse failed:", e);
        return null;
      }
    };
    if (anyMeas) {
      const r = await toIR(measRe, measIm);
      if (r) irOut.measurement = r;
    }
    if (anyTgt) {
      const r = await toIR(tgtRe, tgtIm);
      if (r) irOut.target = r;
    }
    if (anyCorr) {
      const r = await toIR(corrRe, corrIm);
      if (r) irOut.corrected = r;
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
    ir: irOut,
  };
}
