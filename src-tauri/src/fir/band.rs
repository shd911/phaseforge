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
use super::dispatch::{route_for, Route};
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
    /// Peak of target + PEQ (dB) the FIR was asked for (b141.53).
    pub peak_boost_db: f64,
    /// The FIR grid the realized curves are on (5 Hz – Nyquist, with tail).
    pub freq: Vec<f64>,
    /// b141.78: what the realized FIR does beyond the model it was asked for
    /// (target + PEQ, ultrasonic LP included), on `freq`: magnitude in dB
    /// (un-normalised) and wrapped phase in degrees. «Corrected» + this =
    /// the measurement through the exported file.
    pub dev_mag: Vec<f64>,
    pub dev_phase: Vec<f64>,
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

/// The minimum-phase terms the analytic target phase lacks (degrees on
/// `freq`): a min-phase Gaussian HP (with its subsonic), the subsonic of a
/// linear-phase Gaussian HP, a min-phase Gaussian LP. Mirrors the frontend's
/// former `reconstructTargetPhase` term by term.
pub fn target_hilbert_phase(
    freq: &[f64],
    hp: Option<&FilterConfig>,
    lp: Option<&FilterConfig>,
    sample_rate: f64,
) -> Result<Vec<f64>, String> {
    let mut terms: Vec<Vec<f64>> = Vec::new();
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
        let grid = g();
        add(subsonic_db(&grid, hp.expect("checked").freq_hz / 8.0), &grid)?;
    }
    if is_gaussian_min_phase(lp) {
        let grid = g();
        add(gaussian_db(&grid, lp.expect("checked"), true), &grid)?;
    }
    let mut out = vec![0.0; freq.len()];
    for t in &terms { for (o, v) in out.iter_mut().zip(t) { *o += v; } }
    Ok(out)
}

/// FIR grid 5 Hz – 0.95·Nyquist (b141.40); the 512-points-per-5 Hz–40 kHz
/// density is kept as the span grows.
pub fn fir_grid(sample_rate: f64) -> Vec<f64> {
    let f_max = sample_rate / 2.0 * 0.95;
    let n_points = 512usize.max((512.0 * (f_max / 5.0).ln() / (40000.0f64 / 5.0).ln()).ceil() as usize);
    generate_log_freq_grid(n_points, FIR_GRID_F_MIN, f_max)
}

/// The FirConfig a band's target implies: linear main when both crossovers
/// are linear-phase (an absent one counts as linear, b141.2), the subsonic
/// cutoff fc/8 of an active Gaussian protect (b138), MixedPhase when HP and
/// LP disagree on linear_phase.
pub fn band_fir_config(target_curve: &TargetCurve, s: &BandFirSettings) -> FirConfig {
    let hp = target_curve.high_pass.as_ref();
    let lp = target_curve.low_pass.as_ref();
    let lin = |f: Option<&FilterConfig>| f.map_or(true, |c| c.linear_phase);
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
        linear_phase_main: lin(hp) && lin(lp),
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
    let hp = target_curve.high_pass.as_ref();
    let lp = target_curve.low_pass.as_ref();
    let enabled: Vec<PeqBand> = peq.iter().filter(|p| p.enabled).cloned().collect();
    let sr = s.sample_rate;

    let config = band_fir_config(target_curve, s);
    let freq_raw = fir_grid(sr);
    let resp = target::evaluate(target_curve, &freq_raw);
    let hil = target_hilbert_phase(&freq_raw, hp, lp, sr).map_err(|e| AppError::Dsp { message: e })?;
    let mut freq = freq_raw.clone();
    let mut mag = resp.magnitude;
    let mut phase: Vec<f64> = resp.phase.iter().zip(&hil).map(|(p, h)| p + h).collect();

    // b140.5: explicit noise-floor tail up to Nyquist (the frontend's
    // appendNoiseFloorTail) — Rust's boundary clamp would otherwise hold the
    // last value flat across the remaining linear bins.
    let nyq = sr / 2.0;
    let f_hi = *freq.last().expect("non-empty grid");
    if f_hi < nyq * 0.999 {
        let f_end = nyq * 0.999;
        for i in 1..=NOISE_TAIL_POINTS {
            let t = i as f64 / NOISE_TAIL_POINTS as f64;
            freq.push(f_hi * (f_end / f_hi).powf(t));
            mag.push(s.noise_floor_db);
            phase.push(0.0);
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

    let (mut fir, route) = match route_for(hp, lp, target_curve.tilt_db_per_octave, &config) {
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
    if omit_impulse { fir.impulse = Vec::new(); }
    Ok(BandFirResult { fir, route, peak_boost_db, freq, dev_mag, dev_phase })
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
            linear_phase: lin, q: None, subsonic_protect: Some(sub) }
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
        assert_eq!(g.len(), 512);
        assert!((g[0] - 5.0).abs() < 1e-9 && (g[511] - 22_800.0).abs() < 1e-6);
        let g = fir_grid(176_400.0);
        assert!((g.last().unwrap() - 83_790.0).abs() < 1e-6);
        assert!(g.len() > 512);
        // Result grid ends at Nyquist (noise-floor tail).
        let r = generate_band_fir(&curve(None, None), &[], &settings(48_000.0), true).unwrap();
        assert!(*r.freq.last().unwrap() > 23_970.0 && r.freq.len() == 512 + NOISE_TAIL_POINTS);
    }

    #[test]
    fn flat_target_is_flat_in_the_audio_band() {
        // No filters → both "linear" → linear-phase cepstral route; the FIR is
        // flat to 0.95·Nyquist where the noise tail starts (an honest band
        // limit, so not a pure delta at 48 kHz).
        let r = generate_band_fir(&curve(None, None), &[], &settings(48_000.0), false).unwrap();
        assert_eq!(r.route, BandFirRoute::Cepstral);
        for (f, m) in r.freq.iter().zip(&r.fir.realized_mag) {
            if *f >= 20.0 && *f <= 20_000.0 { assert!(m.abs() < 0.1, "{m:.3} dB at {f:.0} Hz"); }
        }
    }

    #[test]
    fn deviation_is_small_where_the_fir_realises_the_model() {
        let lr4 = |fc: f64| FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: fc,
            shape: None, linear_phase: false, q: None, subsonic_protect: None };
        let mut s = settings(96_000.0);
        s.taps = 65_536;
        let r = generate_band_fir(&curve(Some(lr4(200.0)), Some(lr4(3000.0))), &[], &s, true).unwrap();
        for (f, d) in r.freq.iter().zip(&r.dev_mag) {
            if *f >= 300.0 && *f <= 2000.0 { assert!(d.abs() < 0.05, "dev {d:.3} dB at {f:.0} Hz"); }
        }
    }
}
