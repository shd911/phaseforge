//! b140.13.1 — legacy FIR-correction pipelines extracted from
//! `fir/mod.rs`. Two functions live here, both retained verbatim so
//! their callers (Tauri commands `generate_fir` / `generate_hybrid_fir`
//! in `crate::lib`) keep the same `crate::fir::generate_fir(...)`
//! resolution path via the re-export in `fir/mod.rs`.
//!
//! "Legacy" in the sense of architectural priority: the cepstral
//! `generate_model_fir` and the analytical `iir_path` are the two
//! supported pipelines going forward. These functions are kept because
//! the JS UI still calls them through dedicated Tauri commands for the
//! measurement-based correction flow; they are not deprecated.
//!
//! Behaviour is byte-identical to the pre-b140.13.1 definitions —
//! verified by `golden_fir_snapshots` and `pipeline_contract` baselines
//! staying unchanged across this move.

use num_complex::Complex64;
use std::f64::consts::PI;

use tracing::info;

use crate::dsp::fft::FftEngine;
use crate::dsp::{interp_1d, interpolate_linear_grid, minimum_phase_from_magnitude};
use crate::error::AppError;

use super::helpers::{
    assemble_complex_spectrum, build_effective_target, circular_shift_to_center,
    compute_causality, iterative_refine, limit_narrowband_boost,
};
use super::windowing::{generate_half_window, generate_window};
use super::types::{FirConfig, FirModelResult, FirResult, PhaseMode};

/// Generate FIR correction filter.
///
/// # Arguments
/// - `meas_freq` — measurement frequency axis (log-spaced)
/// - `meas_mag` — measurement magnitude in dB (potentially smoothed)
/// - `target_mag` — target magnitude in dB (evaluated at meas_freq)
/// - `peq_correction` — PEQ correction in dB (from `apply_peq`), or empty/zeros
/// - `config` — FIR configuration
/// - `crossover_range` — (f_low, f_high) from target HP/LP
pub fn generate_fir(
    meas_freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    peq_correction: &[f64],
    config: &FirConfig,
    crossover_range: (f64, f64),
) -> Result<FirResult, AppError> {
    let n = meas_freq.len();
    if n < 2 || meas_mag.len() != n || target_mag.len() != n {
        return Err(AppError::Config {
            message: "FIR: freq/mag/target length mismatch".into(),
        });
    }

    let n_fft = config.taps;
    let n_bins = n_fft / 2 + 1;

    // 1. Build effective target: within crossover use target, outside use smoothed measurement
    let effective_target = build_effective_target(
        meas_freq, meas_mag, target_mag, crossover_range,
    );

    // 2. Current magnitude = measurement + PEQ correction
    let current_mag: Vec<f64> = if peq_correction.len() == n {
        meas_mag.iter().zip(peq_correction.iter()).map(|(m, c)| m + c).collect()
    } else {
        meas_mag.to_vec()
    };

    // 3. Correction in dB on measurement grid
    let correction_log: Vec<f64> = current_mag.iter()
        .zip(effective_target.iter())
        .map(|(cur, tgt)| tgt - cur)
        .collect();

    // 4. Interpolate to linear grid (0..Nyquist, n_bins points)
    let (_lin_freq, lin_correction, _) = interpolate_linear_grid(
        meas_freq, &correction_log, None, n_bins, config.sample_rate,
    );

    // 5. Apply boost/cut limiting + narrowband boost protection
    let mut limited: Vec<f64> = lin_correction.iter().map(|&v| {
        v.max(config.noise_floor_db).min(config.max_boost_db)
    }).collect();
    let df = config.sample_rate / n_fft as f64;
    if config.narrowband_limit {
        limit_narrowband_boost(&mut limited, df, config.nb_smoothing_oct, config.nb_max_excess_db);
    }

    // 6. Compute minimum phase via Hilbert transform
    let phase_rad = match config.phase_mode {
        PhaseMode::MinimumPhase => minimum_phase_from_magnitude(&limited, n_fft),
        PhaseMode::LinearPhase => vec![0.0; n_bins], // zero phase → symmetric
        PhaseMode::MixedPhase => minimum_phase_from_magnitude(&limited, n_fft), // same as min for now
        PhaseMode::HybridPhase => minimum_phase_from_magnitude(&limited, n_fft), // fallback: same as min
        // Composite is not used by generate_fir (the legacy measurement path).
        // Fallback to MinimumPhase semantics so callers that wire it here
        // by mistake don't crash.
        PhaseMode::Composite => minimum_phase_from_magnitude(&limited, n_fft),
    };

    // 7. Assemble complex spectrum with conjugate symmetry
    let mut spectrum = assemble_complex_spectrum(&limited, &phase_rad, n_fft);

    // 8. IFFT
    let mut engine = FftEngine::new();
    engine.fft_inverse(&mut spectrum);

    let norm = 1.0 / n_fft as f64;
    let mut impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // 9. Phase-dependent reordering — only used by generate_fir (legacy
    //    measurement path), where Composite is not invoked. Honour the
    //    high-level mode here.
    match config.phase_mode {
        PhaseMode::MinimumPhase | PhaseMode::HybridPhase | PhaseMode::Composite => { /* causal */ }
        PhaseMode::LinearPhase => { circular_shift_to_center(&mut impulse); }
        PhaseMode::MixedPhase => { /* causal for now */ }
    }

    // 10. Apply window
    // For minimum phase: use half-window (1.0 at start, taper to 0 at end)
    // For linear phase: use full symmetric window (centered peak)
    match config.phase_mode {
        PhaseMode::MinimumPhase | PhaseMode::MixedPhase | PhaseMode::HybridPhase | PhaseMode::Composite => {
            let half_win = generate_half_window(n_fft, &config.window);
            for (i, w) in half_win.iter().enumerate() {
                impulse[i] *= w;
            }
        }
        PhaseMode::LinearPhase => {
            let window = generate_window(n_fft, &config.window);
            for (i, w) in window.iter().enumerate() {
                impulse[i] *= w;
            }
        }
    }

    // 11. Iterative weighted refinement (compensates windowing distortion)
    if config.iterations > 0 {
        // generate_fir is the legacy measurement-based path; it does not split
        // PEQ as a separate Hilbert source — pass an empty slice so the inner
        // loop's zero-PEQ guard short-circuits.
        let no_peq: Vec<f64> = vec![0.0; n_bins];
        iterative_refine(
            &mut impulse,
            &limited,         // target correction on linear grid
            &no_peq,          // b140.1: linear-grid PEQ magnitude (empty here)
            &phase_rad,       // phase on linear grid
            config,
            crossover_range,
        );
    }

    // 12. Passband normalization: FFT back to get realized spectrum, normalize peak to 0 dB
    let mut check: Vec<Complex64> = impulse.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    engine.fft_forward(&mut check);

    // Find peak magnitude across all positive frequencies (20 Hz .. Nyquist)
    let df = config.sample_rate / n_fft as f64;
    let mut max_db = f64::NEG_INFINITY;
    for k in 0..n_bins {
        let f = k as f64 * df;
        if f >= 20.0 {
            let amp = check[k].norm();
            let db = if amp > 1e-20 { 20.0 * amp.log10() } else { -200.0 };
            if db > max_db { max_db = db; }
        }
    }
    let norm_db = if max_db.is_finite() { max_db } else { 0.0 };

    let norm_linear = 10.0_f64.powf(-norm_db / 20.0);
    for s in impulse.iter_mut() {
        *s *= norm_linear;
    }

    info!("generate_fir: norm_db={:.2} → peak normalized to 0 dB", norm_db);

    // Build time axis in ms
    let dt_ms = 1000.0 / config.sample_rate;
    let time_ms: Vec<f64> = (0..n_fft).map(|i| i as f64 * dt_ms).collect();

    let causality = compute_causality(&impulse);
    info!("generate_fir: causality={:.4} ({}% post-peak energy)", causality, (causality * 100.0) as u32);

    Ok(FirResult {
        impulse,
        time_ms,
        taps: n_fft,
        sample_rate: config.sample_rate,
        norm_db,
        causality,
    })
}

/// Generate hybrid-phase FIR correction filter.
///
/// Strategy: decompose total correction into two components with different phase modes:
/// 1. **Correction** (measurement → flat): min-phase via Hilbert
///    → compensates both amplitude AND phase of the min-phase driver system
/// 2. **Filter** (flat → target shape): linear-phase (zero phase)
///    → only shapes amplitude, no phase distortion from crossover
///
/// Total magnitude = correction + filter = target - measurement (identical to generate_fir)
/// Total phase = Hilbert(correction) + 0 = Hilbert(correction)
///
/// Result: ideal crossover response with minimal phase deviation.
pub fn generate_hybrid_fir(
    meas_freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    peq_correction: &[f64],
    config: &FirConfig,
    crossover_range: (f64, f64),
) -> Result<FirModelResult, AppError> {
    let n = meas_freq.len();
    if n < 2 || meas_mag.len() != n || target_mag.len() != n {
        return Err(AppError::Config {
            message: "hybrid FIR: freq/mag/target length mismatch".into(),
        });
    }

    let n_fft = config.taps;
    let n_bins = n_fft / 2 + 1;

    // 1. Current magnitude = measurement + PEQ correction
    let current_mag: Vec<f64> = if peq_correction.len() == n {
        meas_mag.iter().zip(peq_correction.iter()).map(|(m, c)| m + c).collect()
    } else {
        meas_mag.to_vec()
    };

    // 2. Compute passband reference level (average in adaptive passband range)
    let (f_low, f_high) = crossover_range;
    let pb_lo = f_low.max(200.0);
    let pb_hi = f_high.min(2000.0).max(pb_lo + 50.0);
    let mut ref_sum = 0.0_f64;
    let mut ref_count = 0usize;
    for i in 0..n {
        if meas_freq[i] >= pb_lo && meas_freq[i] <= pb_hi {
            ref_sum += current_mag[i];
            ref_count += 1;
        }
    }
    let ref_level = if ref_count > 0 { ref_sum / ref_count as f64 } else { 0.0 };

    info!("hybrid_fir: ref_level={:.2} dB (passband {:.0}-{:.0} Hz, {} points)",
        ref_level, pb_lo, pb_hi, ref_count);

    // 3. Correction magnitude: flatten measurement to ref_level (min-phase component)
    let correction_log: Vec<f64> = current_mag.iter()
        .map(|&cur| (ref_level - cur).max(config.noise_floor_db).min(config.max_boost_db))
        .collect();

    // 4. Filter magnitude: target shape relative to ref_level (linear-phase component)
    let filter_log: Vec<f64> = target_mag.iter()
        .map(|&tgt| (tgt - ref_level).max(config.noise_floor_db).min(config.max_boost_db))
        .collect();

    // 5. Total magnitude = correction + filter (= target - current)
    let total_log: Vec<f64> = correction_log.iter()
        .zip(filter_log.iter())
        .map(|(&c, &f)| c + f)
        .collect();

    // 6. Interpolate all three to linear FFT grid
    let (_lin_freq, lin_correction, _) = interpolate_linear_grid(
        meas_freq, &correction_log, None, n_bins, config.sample_rate,
    );
    let (_lin_freq2, mut lin_total, _) = interpolate_linear_grid(
        meas_freq, &total_log, None, n_bins, config.sample_rate,
    );
    let df = config.sample_rate / n_fft as f64;
    if config.narrowband_limit {
        limit_narrowband_boost(&mut lin_total, df, config.nb_smoothing_oct, config.nb_max_excess_db);
    }

    // 7. Phase: Hilbert only from correction component (min-phase)
    // Filter component has zero phase (linear-phase)
    let correction_phase = minimum_phase_from_magnitude(&lin_correction, n_fft);
    // total_phase = correction_phase + 0 = correction_phase

    info!("hybrid_fir: correction range [{:.1}, {:.1}] dB, filter range [{:.1}, {:.1}] dB",
        correction_log.iter().cloned().fold(f64::INFINITY, f64::min),
        correction_log.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        filter_log.iter().cloned().fold(f64::INFINITY, f64::min),
        filter_log.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // 8. Assemble complex spectrum: total magnitude + correction-only phase
    let mut spectrum = assemble_complex_spectrum(&lin_total, &correction_phase, n_fft);

    // 9. IFFT
    let mut engine = FftEngine::new();
    engine.fft_inverse(&mut spectrum);

    let norm = 1.0 / n_fft as f64;
    let mut impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // 10. Half-window (causal): correction is min-phase → impulse is causal
    let half_win = generate_half_window(n_fft, &config.window);
    for (i, w) in half_win.iter().enumerate() {
        impulse[i] *= w;
    }

    // 10b. Iterative weighted refinement
    if config.iterations > 0 {
        // hybrid_fir does not split PEQ from the correction magnitude here;
        // pass an empty slice so Composite mode (if ever wired to this path)
        // short-circuits its PEQ Hilbert step.
        let no_peq: Vec<f64> = vec![0.0; n_bins];
        iterative_refine(
            &mut impulse,
            &lin_total,           // target correction on linear grid
            &no_peq,              // b140.1: linear-grid PEQ magnitude (empty here)
            &correction_phase,    // phase on linear grid
            config,
            crossover_range,
        );
    }

    // 11. Passband normalization: FFT → find peak → normalize to 0 dB
    let mut check: Vec<Complex64> = impulse.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    engine.fft_forward(&mut check);

    let df = config.sample_rate / n_fft as f64;
    let mut max_db = f64::NEG_INFINITY;
    for k in 0..n_bins {
        let f = k as f64 * df;
        if f >= 20.0 {
            let amp = check[k].norm();
            let db = if amp > 1e-20 { 20.0 * amp.log10() } else { -200.0 };
            if db > max_db { max_db = db; }
        }
    }
    let norm_db = if max_db.is_finite() { max_db } else { 0.0 };

    let norm_linear = 10.0_f64.powf(-norm_db / 20.0);
    for s in impulse.iter_mut() {
        *s *= norm_linear;
    }

    info!("generate_hybrid_fir: norm_db={:.2} → peak normalized to 0 dB", norm_db);

    // 12. Extract realized frequency response from normalized impulse
    let mut realized_spectrum: Vec<Complex64> = impulse.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    engine.fft_forward(&mut realized_spectrum);

    let lin_freq: Vec<f64> = (0..n_bins).map(|i| i as f64 * df).collect();

    let mut realized_mag_lin: Vec<f64> = Vec::with_capacity(n_bins);
    let mut realized_phase_lin: Vec<f64> = Vec::with_capacity(n_bins);

    for i in 0..n_bins {
        let c = realized_spectrum[i];
        let amp = c.norm();
        let mag_db = if amp > 1e-20 { 20.0 * amp.log10() } else { -400.0 };
        let phase_deg = c.arg() * 180.0 / PI;
        realized_mag_lin.push(mag_db);
        realized_phase_lin.push(phase_deg);
    }

    // Phase unwrap for smooth interpolation
    for i in 1..realized_phase_lin.len() {
        let diff = realized_phase_lin[i] - realized_phase_lin[i - 1];
        if diff > 180.0 {
            realized_phase_lin[i] -= 360.0 * ((diff + 180.0) / 360.0).floor();
        } else if diff < -180.0 {
            realized_phase_lin[i] += 360.0 * ((-diff + 180.0) / 360.0).floor();
        }
    }

    // Interpolate realized curves back to original log-frequency grid
    let realized_mag = interp_1d(&lin_freq, &realized_mag_lin, meas_freq);
    let realized_phase = interp_1d(&lin_freq, &realized_phase_lin, meas_freq);

    let causality = compute_causality(&impulse);
    info!("generate_hybrid_fir: causality={:.4} ({}%)", causality, (causality * 100.0) as u32);

    Ok(FirModelResult {
        impulse,
        realized_mag,
        realized_phase,
        taps: n_fft,
        sample_rate: config.sample_rate,
        norm_db,
        causality,
    })
}
