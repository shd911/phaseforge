//! b141.70 (audit 2026-10-01 stage 2, A1/A2/A6): the band → FIR request,
//! assembled in ONE place.
//!
//! It used to live in the frontend (band-evaluator/evaluate.ts): FIR grid,
//! target evaluation, the Gaussian / subsonic Hilbert terms, the noise-floor
//! tail, the PEQ response, routing, the shared FirConfig and the dispatch —
//! while the release-readiness / golden tests drove a separate Rust harness
//! (`fir::pipeline`) that zeroed tilt and shelves and skipped the Hilbert
//! terms and the tail. "0 hard failures" then certified the harness, not the
//! app. Both now call [`generate_band_fir`], and the target travels whole
//! (`TargetCurve`) instead of field by field through 13 positional IPC
//! arguments — the route by which tilt (b141.17) and the shelves (H5) were
//! dropped before.
//!
//! The frontend's display phase uses the same Hilbert terms through
//! [`target_hilbert_phase`] (Tauri `compute_target_hilbert_phase`).

use serde::{Deserialize, Serialize};

use crate::dsp::{generate_log_freq_grid, minimum_phase_on_log_grid};
use crate::error::AppError;
use crate::peq::{apply_peq_complex, PeqBand};
use crate::target::{self, FilterConfig, FilterType, TargetCurve};

use super::cepstral::generate_model_fir_with_sections;
use super::dispatch::{route_for_target, Route};
use super::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use super::types::{FirConfig, FirModelResult, PhaseMode, WindowType};

/// Hilbert terms are reconstructed on one grid — the FIR's span — with the
/// FIR's absolute floor (b141.52), then interpolated onto the caller's grid.
const HILBERT_GRID_POINTS: usize = 1024;
const HILBERT_FLOOR_DB: f64 = -150.0;
const FIR_GRID_F_MIN: f64 = 5.0;
const NOISE_TAIL_POINTS: usize = 32;

/// The user-facing FIR settings (what the Export tab and the WAV export send).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandFirSettings {
    pub taps: usize,
    pub sample_rate: f64,
    pub window: WindowType,
    pub max_boost_db: f64,
    pub noise_floor_db: f64,
    pub iterations: usize,
    pub freq_weighting: bool,
    pub narrowband_limit: bool,
    pub nb_smoothing_oct: f64,
    pub nb_max_excess_db: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BandFirRoute { Iir, Cepstral }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandFirResult {
    #[serde(flatten)]
    pub fir: FirModelResult,
    pub route: BandFirRoute,
    /// Peak of target + PEQ (dB) the FIR was asked for (b141.53), relative
    /// to the target's own level (b141.101) — what «Макс. подъём» limits.
    pub peak_boost_db: f64,
    /// The FIR grid the realized curves are on (5 Hz – Nyquist, with tail).
    pub freq: Vec<f64>,
    /// b141.78: what the realized FIR does beyond the model it was asked for
    /// (target + PEQ, ultrasonic LP included), on `freq`: magnitude in dB
    /// (un-normalised) and wrapped phase in degrees. «Corrected» + this =
    /// the measurement through the exported file.
    pub dev_mag: Vec<f64>,
    pub dev_phase: Vec<f64>,
    /// b141.101: what the FIR could not realise as asked (user-facing,
    /// Russian) — shown with the export. Empty when everything fit.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// The filters that reach the file: at or above SECTION_MAX_FRACTION·sr none do.
fn realisable_peq(peq: &[PeqBand], sr: f64) -> Vec<PeqBand> {
    peq.iter().filter(|p| super::section_realisable(p.freq_hz, sr)).cloned().collect()
}

/// PEQ response for the plot, the filters the file drops left out (b141.127,
/// external audit 2026-10-29): 23.6 kHz +6 dB at 48 kHz drew 4.9 dB at 23.5 kHz
/// on Corr FIR beside an impulse without it.
pub fn peq_response_as_exported(freq: &[f64], peq: &[PeqBand], sr: f64) -> (Vec<f64>, Vec<f64>) {
    apply_peq_complex(freq, &realisable_peq(peq, sr), sr)
}

/// b141.101 (audit 2026-10-23 1.1/1.4): sections at or above
/// SECTION_MAX_FRACTION·sr. PEQ there is dropped on both routes (its RBJ
/// biquad is unstable on the IIR route and only an aliased mirror on the
/// cepstral one); crossovers and shelves there take the cepstral route,
/// which realises their in-band part — say so either way.
fn nyquist_warnings(target_curve: &TargetCurve, peq: &[PeqBand], sr: f64) -> Vec<String> {
    let lim = super::SECTION_MAX_FRACTION * sr;
    let ok = |f: f64| super::section_realisable(f, sr);
    let mut out = Vec::new();
    for (f, what) in [(target_curve.high_pass.as_ref(), "ФВЧ"), (target_curve.low_pass.as_ref(), "ФНЧ")] {
        if let Some(f) = f.filter(|f| !ok(f.freq_hz)) {
            out.push(format!(
                "{what} {:.0} Гц не ниже предела {:.0} Гц для частоты экспорта {:.0} Гц — \
                 в FIR только его спад ниже Найквиста.", f.freq_hz, lim, sr));
        }
    }
    for (s, what) in [(target_curve.low_shelf.as_ref(), "Нижняя полка"), (target_curve.high_shelf.as_ref(), "Верхняя полка")] {
        if let Some(s) = s.filter(|s| !ok(s.freq_hz)) {
            out.push(format!(
                "{what} {:.0} Гц не ниже предела {:.0} Гц для частоты экспорта {:.0} Гц — \
                 в FIR только её часть ниже Найквиста.", s.freq_hz, lim, sr));
        }
    }
    let dropped: Vec<String> = peq.iter().filter(|p| !ok(p.freq_hz))
        .map(|p| format!("{:.0} Гц", p.freq_hz)).collect();
    if !dropped.is_empty() {
        out.push(format!(
            "PEQ {} не ниже предела {:.0} Гц для частоты экспорта {:.0} Гц — не включено в FIR.",
            dropped.join(", "), lim, sr));
    }
    out
}

fn is_gaussian_min_phase(f: Option<&FilterConfig>) -> bool {
    matches!(f, Some(c) if matches!(c.filter_type, FilterType::Gaussian) && !c.linear_phase)
}

/// Mirrors `hasActiveSubsonicProtect` (lib/types.ts): Gaussian HP above
/// 40 Hz with the protect flag on.
pub fn subsonic_active(hp: Option<&FilterConfig>) -> bool {
    matches!(hp, Some(c) if matches!(c.filter_type, FilterType::Gaussian)
        && c.subsonic_protect == Some(true) && c.freq_hz > 40.0)
}

fn gaussian_db(freq: &[f64], f: &FilterConfig, is_lp: bool) -> Vec<f64> {
    let m = f.shape.unwrap_or(1.0);
    let ln2 = std::f64::consts::LN_2;
    freq.iter().map(|&x| {
        if x <= 0.0 { return if is_lp { 0.0 } else { -400.0 }; }
        let lp = (-ln2 * (x / f.freq_hz).powf(2.0 * m)).exp();
        let lin = if is_lp { lp } else { 1.0 - lp };
        if lin > 1e-20 { 20.0 * lin.log10() } else { -400.0 }
    }).collect()
}

fn subsonic_db(freq: &[f64], cutoff: f64) -> Vec<f64> {
    freq.iter().map(|&x| {
        if x <= 0.0 { return -400.0; }
        let lin = (1.0 / (1.0 + (cutoff / x).powi(16))).sqrt();
        if lin > 1e-20 { 20.0 * lin.log10() } else { -400.0 }
    }).collect()
}

/// Linear interpolation in log-frequency with edge clamping (the frontend's
/// `interpOnGrid(..., { logSpace: true, outside: "clamp" })`).
fn interp_log_clamp(src_f: &[f64], src_v: &[f64], dst_f: &[f64]) -> Vec<f64> {
    let n = src_f.len();
    let xs: Vec<f64> = src_f.iter().map(|f| f.max(1e-12).ln()).collect();
    dst_f.iter().map(|&f| {
        if f < src_f[0] { return src_v[0]; }
        if f > src_f[n - 1] { return src_v[n - 1]; }
        let x = f.max(1e-12).ln();
        let (mut lo, mut hi) = (0usize, n - 1);
        while hi - lo > 1 {
            let mid = (lo + hi) >> 1;
            if xs[mid] <= x { lo = mid; } else { hi = mid; }
        }
        let dt = xs[hi] - xs[lo];
        let t = if dt > 0.0 { (x - xs[lo]) / dt } else { 0.0 };
        src_v[lo] + t * (src_v[hi] - src_v[lo])
    }).collect()
}

/// True when the band's FIR main is minimum-phase: Composite with
/// `linear_phase_main = false` in [`band_fir_config`] — at least one
/// crossover and every crossover min-phase (an absent one counts as linear,
/// a min/linear pair is MixedPhase).
pub fn min_phase_main(hp: Option<&FilterConfig>, lp: Option<&FilterConfig>) -> bool {
    let present: Vec<&FilterConfig> = [hp, lp].into_iter().flatten().collect();
    !present.is_empty() && present.iter().all(|f| !f.linear_phase)
}

/// The minimum-phase terms the analytic target phase lacks (degrees on
/// `freq`): a min-phase Gaussian HP (with its subsonic), the subsonic of a
/// linear-phase Gaussian HP, a min-phase Gaussian LP, and (b141.101) the
/// tilt on a minimum-phase main. Mirrors the frontend's former
/// `reconstructTargetPhase` term by term.
///
/// b141.101 (audit 2026-10-23 1.3): the cepstral FIR realises a tilt on a
/// min-phase main through the Hilbert transform of its magnitude (−14.4° at
/// 1 kHz for −1 dB/oct), while `target::evaluate` gives a tilt zero phase —
/// the plot and Σ missed ~15° at a crossover when not every band had the
/// same tilt. On a linear or mixed main the FIR's tilt is zero-phase, and so
/// is this term.
pub fn target_hilbert_phase(
    freq: &[f64],
    hp: Option<&FilterConfig>,
    lp: Option<&FilterConfig>,
    tilt: Option<(f64, f64)>,
    sample_rate: f64,
) -> Result<Vec<f64>, String> {
    let mut terms: Vec<Vec<f64>> = Vec::new();
    let mut analytic: Option<Vec<f64>> = None;
    let g = || generate_log_freq_grid(HILBERT_GRID_POINTS, FIR_GRID_F_MIN, sample_rate / 2.0 * 0.95);
    let mut add = |mag: Vec<f64>, grid: &[f64]| -> Result<(), String> {
        let ph = minimum_phase_on_log_grid(grid, &mag, Some(sample_rate), Some(HILBERT_FLOOR_DB))?;
        terms.push(interp_log_clamp(grid, &ph, freq));
        Ok(())
    };
    if is_gaussian_min_phase(hp) {
        let grid = g();
        let h = hp.expect("checked");
        let mut mag = gaussian_db(&grid, h, false);
        if subsonic_active(hp) {
            for (m, s) in mag.iter_mut().zip(subsonic_db(&grid, h.freq_hz / 8.0)) { *m += s; }
        }
        add(mag, &grid)?;
    } else if subsonic_active(hp) && hp.map(|h| h.linear_phase).unwrap_or(false) {
        // b141.108: on a linear main the FIR realises the subsonic as the
        // true Butterworth-8 (Hilbert of its unclipped magnitude, helpers
        // `compose_target_phase`). Through the −150 dB-floored Hilbert the
        // plot lost its 8 zeros at DC: 4.2° at a 100 Hz crossover, 32° at
        // 13 Hz. The analytic phase is the exact term.
        let bw8 = FilterConfig { filter_type: FilterType::Butterworth, order: 8,
            freq_hz: hp.expect("checked").freq_hz / 8.0, shape: None, linear_phase: false, q: None,
            subsonic_protect: None, complementary: false };
        let only = crate::target::TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0, high_pass: Some(bw8), low_pass: None, low_shelf: None, high_shelf: None };
        analytic = Some(crate::target::evaluate(&only, freq).phase);
    }
    if is_gaussian_min_phase(lp) {
        let grid = g();
        add(gaussian_db(&grid, lp.expect("checked"), true), &grid)?;
    }
    if let Some((db_oct, f_ref)) = tilt {
        if db_oct != 0.0 && f_ref > 0.0 && min_phase_main(hp, lp) {
            let grid = g();
            add(grid.iter().map(|&f| db_oct * (f / f_ref).log2()).collect(), &grid)?;
        }
    }
    let mut out = vec![0.0; freq.len()];
    for t in terms.iter().chain(analytic.iter()) { for (o, v) in out.iter_mut().zip(t) { *o += v; } }
    Ok(out)
}

/// Points of the FIR grid per 5 Hz–40 kHz (the density is kept as the span
/// grows). b141.108: was 512 — the dB target between log points is linear,
/// and on a steep curved slope the chord sagged: −0.33 dB at 106 Hz on a
/// Gaussian M 10 low-pass at 100 Hz, at any tap count. 4096 cuts that 64×.
pub const FIR_GRID_POINTS: usize = 4096;

/// FIR grid 5 Hz – 0.95·Nyquist (b141.40).
pub fn fir_grid(sample_rate: f64) -> Vec<f64> {
    let f_max = sample_rate / 2.0 * 0.95;
    let n = FIR_GRID_POINTS as f64;
    let n_points = FIR_GRID_POINTS.max((n * (f_max / 5.0).ln() / (40000.0f64 / 5.0).ln()).ceil() as usize);
    generate_log_freq_grid(n_points, FIR_GRID_F_MIN, f_max)
}

/// The FirConfig a band's target implies: linear main when both crossovers
/// are linear-phase (an absent one counts as linear, b141.2), the subsonic
/// cutoff fc/8 of an active Gaussian protect (b138), MixedPhase when HP and
/// LP disagree on linear_phase.
pub fn band_fir_config(target_curve: &TargetCurve, s: &BandFirSettings) -> FirConfig {
    let hp = target_curve.high_pass.as_ref();
    let lp = target_curve.low_pass.as_ref();
    let is_lin = |f: Option<&FilterConfig>| f.map(|c| c.linear_phase).unwrap_or(false);
    let mixed = hp.is_some() && lp.is_some() && is_lin(hp) != is_lin(lp);
    FirConfig {
        taps: s.taps,
        sample_rate: s.sample_rate,
        max_boost_db: s.max_boost_db,
        noise_floor_db: s.noise_floor_db,
        window: s.window.clone(),
        phase_mode: if mixed { PhaseMode::MixedPhase } else { PhaseMode::Composite },
        iterations: s.iterations,
        freq_weighting: s.freq_weighting,
        narrowband_limit: s.narrowband_limit,
        nb_smoothing_oct: s.nb_smoothing_oct,
        nb_max_excess_db: s.nb_max_excess_db,
        linear_phase_main: crate::target::linear_main(hp, lp),
        subsonic_cutoff_hz: if subsonic_active(hp) { hp.map(|h| h.freq_hz / 8.0) } else { None },
    }
}

/// Generate the export FIR of one band.
pub fn generate_band_fir(
    target_curve: &TargetCurve,
    peq: &[PeqBand],
    s: &BandFirSettings,
    omit_impulse: bool,
) -> Result<BandFirResult, AppError> {
    generate_band_fir_routed(target_curve, peq, s, omit_impulse, None)
}

/// [`generate_band_fir`] with the route optionally forced — for tests that
/// compare the two realisations of the same request.
pub(crate) fn generate_band_fir_routed(
    target_curve: &TargetCurve,
    peq: &[PeqBand],
    s: &BandFirSettings,
    omit_impulse: bool,
    force_route: Option<Route>,
) -> Result<BandFirResult, AppError> {
    let hp = target_curve.high_pass.as_ref();
    let lp = target_curve.low_pass.as_ref();
    let sr = s.sample_rate;
    let all_enabled: Vec<PeqBand> = peq.iter().filter(|p| p.enabled).cloned().collect();
    let warnings = nyquist_warnings(target_curve, &all_enabled, sr);
    let enabled = realisable_peq(&all_enabled, sr);

    let config = band_fir_config(target_curve, s);
    let freq_raw = fir_grid(sr);
    // b141.101 (audit 2026-10-23 1.8): the FIR is built level-free. The
    // target's «Уровень» is a scalar the WAV normalisation removes anyway,
    // but inside the request it moved «Макс. подъём», the noise floor and
    // peak_boost_db with it: level 0/+10/+20 → realised PEQ boost
    // 5.73/5.73/3.73 dB at a 24 dB limit, a false «26 dB above the limit»
    // on the IIR route at +20, a −40 dB stopband for a −60 floor at −20.
    let level_db = target_curve.reference_level_db;
    let mut level_free = target_curve.clone();
    level_free.reference_level_db = 0.0;
    let resp = target::evaluate(&level_free, &freq_raw);
    let tilt = Some((target_curve.tilt_db_per_octave, target_curve.tilt_ref_freq));
    let hil = target_hilbert_phase(&freq_raw, hp, lp, tilt, sr).map_err(|e| AppError::Dsp { message: e })?;
    let mut freq = freq_raw.clone();
    let mut mag = resp.magnitude;
    let mut phase: Vec<f64> = resp.phase.iter().zip(&hil).map(|(p, h)| p + h).collect();

    // b140.5: explicit noise-floor tail up to Nyquist (the frontend's
    // appendNoiseFloorTail) — Rust's boundary clamp would otherwise hold the
    // last value flat across the remaining linear bins.
    // b141.101 (audit 2026-10-23 1.2): on a minimum-phase main the tail
    // continues the magnitude by its log-log slope instead. A −150 dB step at
    // 0.95·Nyquist is a deep notch to the Hilbert transform: an LR4 2 kHz HP
    // at 44.1 kHz lagged the IIR route by 7° at 2 kHz, 42° at 10 kHz and 91°
    // at 15 kHz — a phase the plot never showed (the plot's min-phase terms
    // also continue by slope, `minimum_phase_on_log_grid`).
    // b141.112: the linear main too. Its −150 dB step was a brick wall at
    // 0.95·Nyquist in every passband that reaches the top (a tweeter HP):
    // an LR4 2 kHz linear HP at 44.1/48 kHz pre-rang 7.3–8.0 ms to −60 dB
    // and 230–250 ms to −100 dB (−62 dB 10–100 ms ahead of the peak), 0.7–0.9
    // and 1.8 ms continued; 16–20 kHz unchanged, LP bands bit-identical
    // (their slope sits under the floor clip anyway).
    // b141.115 (Gemini audit 2026-10-29): the phase is held at its last value
    // — a 0° tail under a mixed main's verbatim model phase (HP min + LP lin)
    // was a phase step: 0.2–0.4° off at 5 kHz..0.95·Nyquist, 1–3 dB more
    // ring ahead of the peak; held, 0.01°. And the slope may not rise faster
    // than the target's own tilt: a shelf or boost still rising at the top
    // edge is not a reason to lift the ultrasonic band.
    let nyq = sr / 2.0;
    let f_hi = *freq.last().expect("non-empty grid");
    if f_hi < nyq * 0.999 {
        let f_end = nyq * 0.999;
        let n = freq.len();
        let (m_hi, slope) = if n >= 2 {
            let doct = (freq[n - 1] / freq[n - 2]).log2();
            (mag[n - 1], if doct > 0.0 { (mag[n - 1] - mag[n - 2]) / doct } else { 0.0 })
        } else {
            (mag[n - 1], 0.0)
        };
        let slope = slope.min(target_curve.tilt_db_per_octave.max(0.0));
        let ph_hi = *phase.last().expect("non-empty grid");
        for i in 1..=NOISE_TAIL_POINTS {
            let t = i as f64 / NOISE_TAIL_POINTS as f64;
            let f = f_hi * (f_end / f_hi).powf(t);
            freq.push(f);
            mag.push(m_hi + slope * (f / f_hi).log2());
            phase.push(ph_hi);
        }
    }

    // PEQ at the export rate (b141.5). The cepstral route gets a zero array
    // when there is no PEQ — exactly what the frontend sent.
    let (peq_mag, peq_phase) = if enabled.is_empty() {
        (vec![0.0; freq.len()], vec![0.0; freq.len()])
    } else {
        apply_peq_complex(&freq, &enabled, sr)
    };
    let combined: Vec<f64> = phase.iter().zip(&peq_phase).map(|(a, b)| a + b).collect();
    let peak_boost_db = mag.iter().zip(&peq_mag).map(|(a, b)| a + b).fold(f64::NEG_INFINITY, f64::max);

    let (mut fir, route) = match force_route.unwrap_or_else(|| route_for_target(target_curve, &config)) {
        Route::Iir => (generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp, lp,
            low_shelf: target_curve.low_shelf.as_ref(),
            high_shelf: target_curve.high_shelf.as_ref(),
            peq: &enabled, config: &config,
        })?, BandFirRoute::Iir),
        Route::Cepstral => (generate_model_fir_with_sections(
            &freq, &mag, &peq_mag, &combined, &config, hp, lp,
        )?, BandFirRoute::Cepstral),
    };
    // Deviation of the realized FIR from the request (same grid, no interp).
    let us = super::ultrasonic::ultrasonic_lp_for(sr);
    let wrap = |d: f64| ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
    let dev_mag: Vec<f64> = (0..freq.len()).map(|i| {
        let mut req = mag[i] + peq_mag[i];
        if let Some(c) = us { req += 20.0 * super::ultrasonic::ultrasonic_lp_gain(freq[i], c).max(1e-30).log10(); }
        fir.realized_mag[i] + fir.norm_db - req
    }).collect();
    let dev_phase: Vec<f64> = (0..freq.len()).map(|i| wrap(fir.realized_phase[i] - combined[i])).collect();
    // The contract of norm_db stays «realized_mag + norm_db = the requested
    // response», level included (the Export tab compares it with the target
    // it evaluates with the level). The IIR route never saw the level, so its
    // Export comparison and dev_mag were off by it before.
    fir.norm_db += level_db;
    if omit_impulse { fir.impulse = Vec::new(); }
    Ok(BandFirResult { fir, route, peak_boost_db, freq, dev_mag, dev_phase, warnings })
}

/// Test / harness entry: [`generate_band_fir`] driven by a `FirConfig` (its
/// user-facing fields; the phase mode, linear-main flag and subsonic cutoff
/// are derived from the target exactly as for the app), realized curves
/// resampled onto `freq`. b141.70: the release-readiness gate goes through
/// this, so it certifies the app's path, tilt and shelves included.
pub fn evaluate_on(
    target_curve: &TargetCurve,
    peq: &[PeqBand],
    config: &FirConfig,
    freq: &[f64],
) -> Result<FirModelResult, AppError> {
    let s = BandFirSettings {
        taps: config.taps, sample_rate: config.sample_rate, window: config.window.clone(),
        max_boost_db: config.max_boost_db, noise_floor_db: config.noise_floor_db,
        iterations: config.iterations, freq_weighting: config.freq_weighting,
        narrowband_limit: config.narrowband_limit, nb_smoothing_oct: config.nb_smoothing_oct,
        nb_max_excess_db: config.nb_max_excess_db,
    };
    let r = generate_band_fir(target_curve, peq, &s, false)?;
    let mut fir = r.fir;
    fir.realized_mag = interp_log_clamp(&r.freq, &fir.realized_mag, freq);
    fir.realized_phase = interp_log_clamp(&r.freq, &fir.realized_phase, freq);
    Ok(fir)
}

#[cfg(test)]
mod tests {
    //! Invariants that lived in the frontend's band-evaluator-fir.test.ts
    //! while the request was assembled there (b139.4a, b139.5.3, b139.3.1).
    use super::*;

    fn settings(sr: f64) -> BandFirSettings {
        BandFirSettings {
            taps: 8192, sample_rate: sr, window: WindowType::Blackman, max_boost_db: 24.0,
            noise_floor_db: -150.0, iterations: 0, freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
        }
    }
    fn curve(hp: Option<FilterConfig>, lp: Option<FilterConfig>) -> TargetCurve {
        TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None }
    }
    fn gauss(lin: bool, sub: bool) -> FilterConfig {
        FilterConfig { filter_type: FilterType::Gaussian, order: 4, freq_hz: 80.0, shape: Some(1.0),
            linear_phase: lin, q: None, subsonic_protect: Some(sub), complementary: false }
    }

    #[test]
    fn config_follows_the_target() {
        let s = settings(48_000.0);
        let c = band_fir_config(&curve(None, None), &s);
        assert!(c.linear_phase_main && c.subsonic_cutoff_hz.is_none() && c.phase_mode == PhaseMode::Composite);
        let c = band_fir_config(&curve(Some(gauss(true, true)), None), &s);
        assert!(c.linear_phase_main && c.subsonic_cutoff_hz == Some(10.0));
        let c = band_fir_config(&curve(Some(gauss(false, true)), None), &s);
        assert!(!c.linear_phase_main && c.subsonic_cutoff_hz == Some(10.0));
        let c = band_fir_config(&curve(Some(gauss(true, false)), None), &s);
        assert!(c.linear_phase_main && c.subsonic_cutoff_hz.is_none());
        let mut lp = gauss(false, false);
        lp.freq_hz = 2000.0;
        let c = band_fir_config(&curve(Some(gauss(true, false)), Some(lp)), &s);
        assert_eq!(c.phase_mode, PhaseMode::MixedPhase);
    }

    #[test]
    fn fir_grid_spans_5_hz_to_095_nyquist_and_keeps_density() {
        let g = fir_grid(48_000.0);
        assert_eq!(g.len(), FIR_GRID_POINTS);
        assert!((g[0] - 5.0).abs() < 1e-9 && (g[FIR_GRID_POINTS - 1] - 22_800.0).abs() < 1e-6);
        let g = fir_grid(176_400.0);
        assert!((g.last().unwrap() - 83_790.0).abs() < 1e-6);
        assert!(g.len() > FIR_GRID_POINTS);
        // Result grid ends at Nyquist (the tail continued by slope, b141.112).
        let r = generate_band_fir(&curve(None, None), &[], &settings(48_000.0), true).unwrap();
        assert!(*r.freq.last().unwrap() > 23_970.0 && r.freq.len() == FIR_GRID_POINTS + NOISE_TAIL_POINTS);
    }

    #[test]
    fn flat_target_is_flat_in_the_audio_band() {
        // No filters → both "linear" → linear-phase cepstral route; the FIR is
        // flat up to Nyquist since b141.112 (was a −150 dB wall at 0.95·Nyquist).
        let r = generate_band_fir(&curve(None, None), &[], &settings(48_000.0), false).unwrap();
        assert_eq!(r.route, BandFirRoute::Cepstral);
        for (f, m) in r.freq.iter().zip(&r.fir.realized_mag) {
            if *f >= 20.0 && *f <= 20_000.0 { assert!(m.abs() < 0.1, "{m:.3} dB at {f:.0} Hz"); }
        }
    }

    /// b141.101 (audit 2026-10-23 2.2): a noise floor above the boost limit
    /// panicked in f64::clamp on the cepstral min-phase route with a digital
    /// section (here: LR4 + tilt), hanging the IPC promise.
    #[test]
    fn noise_floor_above_boost_does_not_panic() {
        let lr4 = FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 2000.0,
            shape: None, linear_phase: false, q: None, subsonic_protect: None, complementary: false };
        let mut c = curve(Some(lr4), None);
        c.tilt_db_per_octave = -0.5;
        let mut s = settings(48_000.0);
        s.max_boost_db = 6.0;
        s.noise_floor_db = 10.0;
        let r = generate_band_fir(&c, &[], &s, false).unwrap();
        assert_eq!(r.route, BandFirRoute::Cepstral);
        assert!(r.fir.impulse.iter().all(|v| v.is_finite()));
    }

    /// b141.101 (audit 2026-10-23 1.1/2.1/1.4): 48 kHz export, PEQ 24.5 kHz
    /// and a high shelf at 26 kHz on an LR4 band that otherwise takes the
    /// IIR route — the poles of both RBJ sections sat outside the unit
    /// circle (8488 NaN samples). Now: finite impulse, warnings, no error;
    /// an LP above Nyquist no longer fails the export.
    #[test]
    fn sections_above_nyquist_give_a_finite_fir_with_warnings() {
        let lr4 = |fc: f64| FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: fc,
            shape: None, linear_phase: false, q: None, subsonic_protect: None, complementary: false };
        let s = settings(48_000.0);
        let pk = |f: f64| PeqBand { freq_hz: f, gain_db: 6.0, q: 2.0, enabled: true,
            filter_type: crate::peq::PeqFilterType::Peaking };
        let finite = |r: &BandFirResult| r.fir.impulse.iter().chain(&r.fir.realized_mag)
            .chain(&r.fir.realized_phase).all(|v| v.is_finite());

        // PEQ 24.5 kHz alone: still the IIR route, the PEQ is dropped.
        let c = curve(Some(lr4(2000.0)), None);
        let r = generate_band_fir(&c, &[pk(24_500.0), pk(5000.0)], &s, false).unwrap();
        assert_eq!(r.route, BandFirRoute::Iir);
        assert!(finite(&r) && r.fir.norm_db > -100.0, "norm_db {}", r.fir.norm_db);
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        assert!(r.warnings[0].contains("24500"));
        // The 5 kHz PEQ is still realised (+6 dB at its centre, relative).
        let at = |f: f64| { let i = r.freq.iter().position(|&x| x >= f).unwrap(); r.fir.realized_mag[i] };
        assert!((at(5000.0) - at(15_000.0) - 6.0).abs() < 0.5, "{} {}", at(5000.0), at(15_000.0));

        // High shelf 26 kHz: cepstral route (the analog shelf below Nyquist).
        let mut c = curve(Some(lr4(2000.0)), None);
        c.high_shelf = Some(crate::target::ShelfConfig { freq_hz: 26_000.0, gain_db: 6.0, q: 0.707 });
        let r = generate_band_fir(&c, &[pk(24_500.0)], &s, false).unwrap();
        assert_eq!(r.route, BandFirRoute::Cepstral);
        assert!(finite(&r));
        assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);

        // LP above Nyquist: no «fc out of range».
        let c = curve(Some(lr4(2000.0)), Some(lr4(30_000.0)));
        let r = generate_band_fir(&c, &[], &s, false).unwrap();
        assert_eq!(r.route, BandFirRoute::Cepstral);
        assert!(finite(&r) && r.warnings.len() == 1 && r.warnings[0].contains("ФНЧ"));

        // The IIR generator itself skips such sections instead of NaN.
        let cfg = band_fir_config(&curve(Some(lr4(2000.0)), None), &s);
        let freq = fir_grid(48_000.0);
        let hs = crate::target::ShelfConfig { freq_hz: 26_000.0, gain_db: 6.0, q: 0.707 };
        let r = generate_min_phase_fir_iir(&IirPathInput { freq: &freq, hp: Some(&lr4(2000.0)),
            lp: Some(&lr4(30_000.0)), low_shelf: None, high_shelf: Some(&hs), peq: &[pk(24_500.0)],
            config: &cfg }).unwrap();
        assert!(r.impulse.iter().all(|v| v.is_finite()));
    }

    /// b141.101 (audit 2026-10-23 1.2): the cepstral min-phase route must
    /// realise an LR4 HP with the phase of the IIR route. The −150 dB noise
    /// tail from 0.95·Nyquist was a step whose Hilbert transform lagged the
    /// phase: −7° @ 2 kHz, −42° @ 10 kHz, −89.5° @ 15 kHz at 44.1 kHz.
    #[test]
    fn cepstral_min_phase_matches_iir_up_to_nyquist() {
        let lr4 = FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 2000.0,
            shape: None, linear_phase: false, q: None, subsonic_protect: None, complementary: false };
        let c = curve(Some(lr4), None);
        let mut s = settings(44_100.0);
        s.taps = 65_536;
        let cep = generate_band_fir_routed(&c, &[], &s, true, Some(Route::Cepstral)).unwrap();
        let iir = generate_band_fir_routed(&c, &[], &s, true, Some(Route::Iir)).unwrap();
        let wrap = |d: f64| ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        let mut worst: f64 = 0.0;
        for f in [1000.0, 2000.0, 5000.0, 10_000.0, 15_000.0, 19_000.0] {
            let i = cep.freq.iter().position(|&x| x >= f).unwrap();
            let d = wrap(cep.fir.realized_phase[i] - iir.fir.realized_phase[i]);
            eprintln!("{f:>6} Hz: cepstral − IIR = {d:+.2}°");
            worst = worst.max(d.abs());
        }
        assert!(worst < 1.0, "worst phase gap {worst:.2}°");
    }

    /// b141.101 (audit 2026-10-23 1.3): the plot's tilt phase is the FIR's.
    /// −1 dB/oct on an LR4 band, cepstral route: what the tilt adds to the
    /// FIR's deviation from the request phase (analytic + Hilbert terms, the
    /// phase the plot draws) — dev(tilt) − dev(no tilt), which cancels the
    /// LR sections' bilinear deviation. It was the whole tilt phase on a
    /// min-phase main (−14.5° at 1 kHz); a linear main keeps a zero-phase
    /// tilt on both sides.
    #[test]
    fn tilt_phase_of_the_plot_is_the_fir_phase() {
        let lr4 = |fc: f64, lin: bool| FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4,
            freq_hz: fc, shape: None, linear_phase: lin, q: None, subsonic_protect: None, complementary: false };
        let mut s = settings(48_000.0);
        s.taps = 65_536;
        for lin in [false, true] {
            let mut c = curve(Some(lr4(200.0, lin)), Some(lr4(5000.0, lin)));
            let r0 = generate_band_fir_routed(&c, &[], &s, true, Some(Route::Cepstral)).unwrap();
            c.tilt_db_per_octave = -1.0;
            let r = generate_band_fir(&c, &[], &s, true).unwrap();
            assert_eq!(r.route, BandFirRoute::Cepstral);
            let terms = target_hilbert_phase(&r.freq, c.high_pass.as_ref(), c.low_pass.as_ref(),
                Some((-1.0, 1000.0)), 48_000.0).unwrap();
            for f in [100.0, 300.0, 1000.0, 3000.0, 8000.0] {
                let i = r.freq.iter().position(|&x| x >= f).unwrap();
                let d = r.dev_phase[i] - r0.dev_phase[i];
                eprintln!("lin={lin} {f:>6} Hz: tilt term {:+.2}°, its FIR deviation {d:+.2}°", terms[i]);
                if lin { assert!(terms[i].abs() < 1e-9); } else { assert!(terms[i] < -10.0, "{}", terms[i]); }
                assert!(d.abs() < 1.0, "tilt deviation {d:.2}° at {f} Hz");
            }
        }
    }

    /// b141.101 (audit 2026-10-23 1.8): the target's level does not change
    /// the FIR. 48k/64K, LR4 80/3k, PEQ +6 dB @ 200 Hz Q1: on the cepstral
    /// route (linear-phase) the realised boost was 5.73/5.73/3.73 dB at
    /// level 0/+10/+20 (24 dB limit) and 2.73 dB at +3 with a 6 dB limit; on
    /// the IIR route +20 read as a «26 dB» boost.
    #[test]
    fn target_level_does_not_shape_the_fir() {
        let lr4 = |fc: f64, lin: bool| FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4,
            freq_hz: fc, shape: None, linear_phase: lin, q: None, subsonic_protect: None, complementary: false };
        let peq = [PeqBand { freq_hz: 200.0, gain_db: 6.0, q: 1.0, enabled: true,
            filter_type: crate::peq::PeqFilterType::Peaking }];
        let mut s = settings(48_000.0);
        s.taps = 65_536;
        for (lin, limit) in [(true, 24.0), (true, 6.0), (false, 24.0)] {
            s.max_boost_db = limit;
            let run = |level: f64| {
                let mut c = curve(Some(lr4(80.0, lin)), Some(lr4(3000.0, lin)));
                c.reference_level_db = level;
                generate_band_fir(&c, &peq, &s, true).unwrap()
            };
            let r0 = run(0.0);
            let at = |r: &BandFirResult, f: f64| { let i = r.freq.iter().position(|&x| x >= f).unwrap(); r.fir.realized_mag[i] };
            let boost0 = at(&r0, 200.0) - at(&r0, 1000.0);
            eprintln!("lin={lin} limit={limit}: boost {boost0:.2} dB, peak_boost {:.2}", r0.peak_boost_db);
            assert!(boost0 > 5.0, "boost {boost0:.2} dB");
            for level in [3.0, 10.0, 20.0, -20.0] {
                let r = run(level);
                let worst = r.fir.realized_mag.iter().zip(&r0.fir.realized_mag)
                    .zip(&r.freq).filter(|(_, &f)| (20.0..=20_000.0).contains(&f))
                    .map(|((a, b), _)| (a - b).abs()).fold(0.0_f64, f64::max);
                assert!(worst < 1e-6, "level {level}: realised response moved by {worst:.4} dB");
                assert!((r.peak_boost_db - r0.peak_boost_db).abs() < 1e-9, "peak_boost {}", r.peak_boost_db);
                assert!((r.fir.norm_db - r0.fir.norm_db - level).abs() < 1e-6, "norm_db carries the level");
                let dev = r.dev_mag.iter().zip(&r0.dev_mag).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
                assert!(dev < 1e-6, "dev_mag moved by {dev}");
            }
        }
    }

    #[test]
    fn deviation_is_small_where_the_fir_realises_the_model() {
        let lr4 = |fc: f64| FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: fc,
            shape: None, linear_phase: false, q: None, subsonic_protect: None, complementary: false };
        let mut s = settings(96_000.0);
        s.taps = 65_536;
        let r = generate_band_fir(&curve(Some(lr4(200.0)), Some(lr4(3000.0))), &[], &s, true).unwrap();
        for (f, d) in r.freq.iter().zip(&r.dev_mag) {
            if *f >= 300.0 && *f <= 2000.0 { assert!(d.abs() < 0.05, "dev {d:.3} dB at {f:.0} Hz"); }
        }
    }

    #[test]
    fn plot_peq_drops_what_the_file_drops() {
        let peq = [PeqBand {
            freq_hz: 23_600.0, gain_db: 6.0, q: 1.0, enabled: true,
            filter_type: crate::peq::PeqFilterType::Peaking,
        }];
        let f = [20.0, 1000.0, 23_500.0];
        let (raw, _) = apply_peq_complex(&f, &peq, 48_000.0);
        assert!(raw[2] > 3.0, "the unstable biquad draws a boost: {}", raw[2]);
        let (mag, phase) = peq_response_as_exported(&f, &peq, 48_000.0);
        assert!(mag.iter().chain(&phase).all(|v| *v == 0.0), "{mag:?} {phase:?}");
        let (at_96k, _) = peq_response_as_exported(&f, &peq, 96_000.0);
        assert!(at_96k[2] > 3.0, "below the limit it stays: {}", at_96k[2]);
    }
}
