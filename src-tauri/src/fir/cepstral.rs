//! b140.13.2 — FFT-cepstral FIR pipeline extracted from `fir/mod.rs`.
//!
//! Houses `generate_model_fir`, the forward-going pipeline for FIR
//! synthesis from pre-evaluated target + PEQ magnitudes and a model
//! phase array. Used by the JS UI for every band configuration that
//! `pickFirRoute` (a.k.a. `fir::route_for`) classifies as `Cepstral`:
//! Gaussian / Bessel filters, linear-phase main, composite + subsonic,
//! and custom measured targets.
//!
//! The IIR-analytical sibling lives in `fir/iir_path.rs` (kept as-is
//! pending b140.13.3); the legacy measurement-based pipelines live in
//! `fir/legacy.rs`.
//!
//! Behaviour is byte-identical to the pre-b140.13.2 definition — locked
//! down by `golden_fir_snapshots` and `pipeline_contract` baselines
//! staying unchanged across this move.

use num_complex::Complex64;
use std::f64::consts::PI;

use tracing::info;

use crate::dsp::fft::FftEngine;
use crate::dsp::{interp_1d, interpolate_linear_grid, minimum_phase_from_magnitude};
use crate::error::AppError;

use super::helpers::{
    assemble_complex_spectrum, circular_shift_to_center, compose_target_phase,
    compute_causality, iterative_refine,
};
use super::windowing::{generate_half_window, generate_window};
use super::types::{FirConfig, FirModelResult, PhaseMode};

/// Generate FIR from a pure mathematical filter model (target curve without measurement).
///
/// Phase handling:
/// - **Target (HP/LP/shelf/tilt)**: phase_mode from config determines linear or min-phase.
///   LinearPhase → zero phase contribution. MinimumPhase → Hilbert from target magnitude.
/// - **PEQ**: ALWAYS minimum-phase (Hilbert from PEQ magnitude). PEQ bands are
///   inherently min-phase biquads — their phase must be in the FIR.
/// - Total phase = target_phase + PEQ_phase.
///   Hilbert is linear, so Hilbert(A+B) = Hilbert(A) + Hilbert(B).
///   For MinimumPhase target: total = Hilbert(target+PEQ) — identical to before.
///   For LinearPhase target: total = 0 + Hilbert(PEQ) — PEQ min-phase preserved.
///
/// # Arguments
/// - `freq` — log-spaced frequency axis
/// - `target_mag` — target-only magnitude in dB (HP/LP/shelf/tilt, no PEQ)
/// - `peq_mag` — PEQ-only magnitude in dB (may be empty → treated as 0 dB)
/// - `model_phase` — combined model phase in degrees (for display/zero-detection)
/// - `config` — FIR configuration (taps, sample_rate, window, phase_mode)
pub fn generate_model_fir(
    freq: &[f64],
    target_mag: &[f64],
    peq_mag: &[f64],
    model_phase: &[f64],
    config: &FirConfig,
) -> Result<FirModelResult, AppError> {
    let n = freq.len();
    if n < 2 || target_mag.len() != n || model_phase.len() != n {
        return Err(AppError::Config {
            message: "generate_model_fir: freq/mag/phase length mismatch".into(),
        });
    }
    let has_peq = !peq_mag.is_empty();
    if has_peq && peq_mag.len() != n {
        return Err(AppError::Config {
            message: "generate_model_fir: peq_mag length mismatch".into(),
        });
    }

    let n_fft = config.taps;
    let n_bins = n_fft / 2 + 1;

    // 1. Interpolate target mag (dB) to linear FFT grid
    let (lin_freq, lin_target_raw, _) = interpolate_linear_grid(
        freq, target_mag, None, n_bins, config.sample_rate,
    );

    // Clip target magnitude to prevent Hilbert instability from extreme HP/LP rolloff
    let lin_target: Vec<f64> = lin_target_raw.iter().map(|&v| {
        v.max(config.noise_floor_db).min(config.max_boost_db)
    }).collect();

    // 2. Interpolate PEQ mag (dB) to linear FFT grid (if present)
    let lin_peq: Vec<f64> = if has_peq {
        let (_, peq_raw, _) = interpolate_linear_grid(
            freq, peq_mag, None, n_bins, config.sample_rate,
        );
        // PEQ magnitude is typically small — no extreme clipping needed
        peq_raw.iter().map(|&v| v.max(-60.0).min(config.max_boost_db)).collect()
    } else {
        vec![0.0; n_bins]
    };

    // 3. Total magnitude = target + PEQ (in dB)
    let lin_mag: Vec<f64> = lin_target.iter().zip(lin_peq.iter()).map(|(&t, &p)| t + p).collect();

    // 4. Determine effective phase mode
    let max_phase_abs = model_phase.iter().map(|p| p.abs()).fold(0.0_f64, f64::max);
    let effective_linear = match config.phase_mode {
        PhaseMode::LinearPhase => true,
        // MixedPhase: linear only if no per-filter Gaussian info provided
        PhaseMode::MixedPhase => config.gaussian_min_phase_filters.is_empty(),
        PhaseMode::MinimumPhase | PhaseMode::HybridPhase => false,
        // b139.4a: Composite follows linear_phase_main for the impulse
        // structure (full window + center shift). The subsonic min-phase
        // contribution is carried by the assembled phase spectrum, not by
        // the windowing path, so half-window would corrupt the symmetric
        // base component. Even with subsonic on, the realized phase
        // analyzer subtracts the N/2 linear delay; subsonic min-phase
        // rotation survives in the residual.
        PhaseMode::Composite => config.linear_phase_main,
    };

    info!(
        "FIR: effective_linear={}, phase_mode={:?}, has_peq={}, max_phase_abs={:.2}°",
        effective_linear, config.phase_mode, has_peq, max_phase_abs
    );

    // 5. Phase for IFFT:
    //    LinearPhase: zero phase (symmetric FIR)
    //    MinimumPhase/HybridPhase: Hilbert from full magnitude (causal FIR)
    //    MixedPhase + non-zero model_phase: use frontend-provided per-filter phase
    //      (some filters lin-phase, some min-phase — frontend computes per-filter Hilbert)
    //    MixedPhase + zero model_phase: fallback to LinearPhase
    //    PEQ phase: ALWAYS Hilbert (min-phase biquads)
    let target_phase_rad = if config.phase_mode == PhaseMode::Composite {
        // b140.1: compose_target_phase now returns the full
        // main + peq + subsonic phase composed from three independent
        // Hilbert sources — matches what iterative_refine recomputes each
        // pass, so initial and iterated assemblies stay in lockstep.
        // Total magnitude (lin_mag = target + peq) and lin_peq are passed
        // separately so composite_phase_inner can subtract them out of
        // base_mag and Hilbert each piece on its own.
        compose_target_phase(
            &lin_mag,
            &lin_peq,
            n_fft,
            n_bins,
            config.sample_rate,
            config.linear_phase_main,
            config.subsonic_cutoff_hz,
            config.noise_floor_db,
        )
    } else if effective_linear {
        vec![0.0; n_bins]
    } else if config.phase_mode == PhaseMode::MixedPhase && !config.gaussian_min_phase_filters.is_empty() {
        // MixedPhase: compute per-filter Gaussian Hilbert on linear FFT grid
        let nyquist = config.sample_rate / 2.0;
        let mut phase_acc = vec![0.0_f64; n_bins];
        for gf in &config.gaussian_min_phase_filters {
            let mut filt_mag = vec![0.0_f64; n_bins];
            let ln2 = 2.0_f64.ln();
            for k in 0..n_bins {
                let f = nyquist * k as f64 / (n_bins - 1) as f64;
                let ratio = if gf.freq_hz > 0.0 { f / gf.freq_hz } else { 0.0 };
                let lp_lin = (-ln2 * ratio.powf(2.0 * gf.shape)).exp();
                let lin = if gf.is_lowpass { lp_lin } else { 1.0 - lp_lin };
                filt_mag[k] = if lin > 1e-20 { 20.0 * lin.log10() } else { -400.0 };
            }
            for v in filt_mag.iter_mut() {
                *v = v.max(config.noise_floor_db);
            }
            let filt_phase = minimum_phase_from_magnitude(&filt_mag, n_fft);
            for k in 0..n_bins {
                phase_acc[k] += filt_phase[k];
            }
        }
        phase_acc
    } else {
        minimum_phase_from_magnitude(&lin_target, n_fft)
    };

    // b140.1: Composite mode already incorporates the PEQ Hilbert via
    // compose_target_phase. Other phase modes still need it added here.
    let peq_phase_rad = if config.phase_mode == PhaseMode::Composite {
        vec![0.0; n_bins]
    } else if has_peq {
        minimum_phase_from_magnitude(&lin_peq, n_fft)
    } else {
        vec![0.0; n_bins]
    };

    let phase_rad: Vec<f64> = target_phase_rad.iter()
        .zip(peq_phase_rad.iter())
        .map(|(&t, &p)| t + p)
        .collect();

    // 6. Assemble complex spectrum and IFFT
    let mut spectrum = assemble_complex_spectrum(&lin_mag, &phase_rad, n_fft);

    let mut engine = FftEngine::new();
    engine.fft_inverse(&mut spectrum);

    let norm = 1.0 / n_fft as f64;
    let mut impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // 4. Phase-dependent reordering + windowing
    if effective_linear {
        // Symmetric impulse: shift peak to center, full window
        circular_shift_to_center(&mut impulse);
        let window = generate_window(n_fft, &config.window);
        for (i, w) in window.iter().enumerate() {
            impulse[i] *= w;
        }
    } else if config.phase_mode == PhaseMode::MixedPhase && !config.gaussian_min_phase_filters.is_empty() {
        // MixedPhase: impulse is neither fully causal nor symmetric.
        // Find the peak, shift it to center, apply full window.
        let peak_idx = impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        let shift = (n_fft / 2).wrapping_sub(peak_idx) % n_fft;
        impulse.rotate_right(shift);
        let window = generate_window(n_fft, &config.window);
        for (i, w) in window.iter().enumerate() {
            impulse[i] *= w;
        }
    } else {
        // Causal impulse (min-phase): half window
        let half_win = generate_half_window(n_fft, &config.window);
        for (i, w) in half_win.iter().enumerate() {
            impulse[i] *= w;
        }
    }

    // 5b. Iterative weighted refinement (compensates windowing distortion)
    if config.iterations > 0 {
        iterative_refine(
            &mut impulse,
            &lin_mag,         // target magnitude on linear grid
            &lin_peq,         // b140.1: PEQ magnitude (own Hilbert source)
            &phase_rad,       // phase on linear grid
            config,
            (20.0, 20000.0),  // model FIR: full range, no crossover
        );
    }

    // 6. FFT the windowed impulse back to get realized frequency response
    let mut realized_spectrum: Vec<Complex64> = impulse.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    engine.fft_forward(&mut realized_spectrum);

    // Extract magnitude (dB) and excess phase (degrees) for positive frequencies.
    //
    // For linear-phase: subtract N/2 linear delay to get excess phase ≈ 0°.
    // For minimum-phase: raw phase is the min-phase response (no delay to subtract).
    let mut realized_mag_lin: Vec<f64> = Vec::with_capacity(n_bins);
    let mut realized_phase_lin: Vec<f64> = Vec::with_capacity(n_bins);

    let delay_samples = if effective_linear { (n_fft / 2) as f64 } else { 0.0 };

    for i in 0..n_bins {
        let c = realized_spectrum[i];
        let amp = c.norm();
        let mag_db = if amp > 1e-20 { 20.0 * amp.log10() } else { -400.0 };

        // Raw phase in radians
        let raw_phase_rad = c.arg();

        // Subtract linear delay: phase_delay = -2π·k·delay/N where k=bin index
        let linear_delay_rad = -2.0 * PI * i as f64 * delay_samples / n_fft as f64;
        let excess_phase_rad = raw_phase_rad - linear_delay_rad;

        // Wrap to [-π, π]
        let excess_wrapped = ((excess_phase_rad + PI) % (2.0 * PI) + 2.0 * PI) % (2.0 * PI) - PI;
        let phase_deg = excess_wrapped * 180.0 / PI;

        realized_mag_lin.push(mag_db);
        realized_phase_lin.push(phase_deg);
    }

    // Unwrap phase for smooth interpolation:
    // After removing linear delay, excess phase should be smooth.
    // Unwrap: if jump > 180° between adjacent bins, add/subtract 360°.
    for i in 1..realized_phase_lin.len() {
        let diff = realized_phase_lin[i] - realized_phase_lin[i - 1];
        if diff > 180.0 {
            realized_phase_lin[i] -= 360.0 * ((diff + 180.0) / 360.0).floor();
        } else if diff < -180.0 {
            realized_phase_lin[i] += 360.0 * ((-diff + 180.0) / 360.0).floor();
        }
    }

    // 7. Interpolate realized back to original log-frequency grid
    let realized_mag = interp_1d(&lin_freq, &realized_mag_lin, freq);
    let realized_phase = interp_1d(&lin_freq, &realized_phase_lin, freq);

    // 8. Passband normalization: shift so peak of realized magnitude = 0 dB
    //    This works for any filter shape (narrow band, wide band, HP, LP, BP).
    let norm_db = realized_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let norm_db = if norm_db.is_finite() { norm_db } else { 0.0 };

    info!("generate_model_fir: realized_max={:.2} dB → normalizing by {:.2} dB", norm_db, norm_db);

    // Scale impulse so passband peak = 0 dB
    let norm_linear = 10.0_f64.powf(-norm_db / 20.0);
    for s in impulse.iter_mut() {
        *s *= norm_linear;
    }

    // Shift realized curves to match normalized output
    let realized_mag: Vec<f64> = realized_mag.iter().map(|&v| v - norm_db).collect();

    info!("generate_model_fir: norm_db={:.2} → passband normalized to 0 dB", norm_db);

    // 9. Time axis
    let dt_ms = 1000.0 / config.sample_rate;
    let time_ms: Vec<f64> = (0..n_fft).map(|i| i as f64 * dt_ms).collect();

    let causality = compute_causality(&impulse);
    info!("generate_model_fir: causality={:.4} ({}%)", causality, (causality * 100.0) as u32);

    Ok(FirModelResult {
        impulse,
        time_ms,
        realized_mag,
        realized_phase,
        taps: n_fft,
        sample_rate: config.sample_rate,
        norm_db,
        causality,
    })
}
