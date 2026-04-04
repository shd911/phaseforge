// FIR correction engine: internal computation helpers

use std::f64::consts::PI;

use num_complex::Complex64;
use tracing::info;

use crate::dsp::fft::FftEngine;
use crate::dsp::fractional_octave_smooth;
use crate::error::AppError;

use super::types::*;
use super::windowing::*;

// ---------------------------------------------------------------------------
// Internal: iterative weighted least-squares refinement
// ---------------------------------------------------------------------------

/// Frequency-dependent weight function for iterative optimization.
///
/// Higher weight = tighter correction in that frequency range.
/// - Crossover transition bands: 3.0 (critical for phase alignment)
/// - 200-4000 Hz (speech/music): 2.0 (max auditory sensitivity)
/// - Below 200 Hz: 1.0 (standard)
/// - Above 8000 Hz: 0.5 (psychoacoustic masking reduces requirements)
pub(crate) fn frequency_weight(freq_hz: f64, crossover_range: (f64, f64)) -> f64 {
    let (f_low, f_high) = crossover_range;

    // Crossover transition bands (±0.5 octave around HP and LP)
    let near_hp: f64 = if f_low > 10.0 {
        let ratio = (freq_hz / f_low).log2().abs();
        if ratio < 0.5 { 3.0 } else { 0.0 }
    } else { 0.0 };
    let near_lp: f64 = if f_high < 20000.0 {
        let ratio = (freq_hz / f_high).log2().abs();
        if ratio < 0.5 { 3.0 } else { 0.0 }
    } else { 0.0 };

    if near_hp > 0.0 || near_lp > 0.0 {
        return near_hp.max(near_lp);
    }

    // Frequency-based weight
    if freq_hz < 200.0 {
        1.0
    } else if freq_hz <= 4000.0 {
        2.0
    } else if freq_hz <= 8000.0 {
        1.5
    } else {
        0.5
    }
}

/// Iterative weighted refinement of FIR impulse response.
///
/// After initial frequency-sampling FIR generation, this function:
/// 1. FFTs the windowed impulse to get realized spectrum
/// 2. Computes weighted error vs desired correction
/// 3. Adds weighted error back to correction spectrum
/// 4. Regenerates impulse via IFFT + window
/// 5. Repeats for `iterations` passes
///
/// This compensates for windowing distortion and improves accuracy
/// especially in crossover and speech bands.
pub(crate) fn iterative_refine(
    impulse: &mut Vec<f64>,
    target_correction_db: &[f64],   // desired correction on linear grid (n_bins)
    phase_rad: &[f64],              // phase on linear grid (n_bins)
    config: &FirConfig,
    crossover_range: (f64, f64),
) {
    let iterations = config.iterations.min(10);
    if iterations == 0 {
        return;
    }

    let n_fft = config.taps;
    let n_bins = n_fft / 2 + 1;
    let df = config.sample_rate / n_fft as f64;

    // Pre-compute weights for each frequency bin
    let weights: Vec<f64> = (0..n_bins)
        .map(|k| {
            let f = k as f64 * df;
            if f < 10.0 { 0.0 } else if config.freq_weighting { frequency_weight(f, crossover_range) } else { 1.0 }
        })
        .collect();

    // Working copy of correction spectrum (will be refined)
    let mut refined_db: Vec<f64> = target_correction_db.to_vec();

    let mut engine = FftEngine::new();

    let is_linear_phase = matches!(config.phase_mode, PhaseMode::LinearPhase);

    for iter in 0..iterations {
        // 1. FFT current impulse → realized spectrum
        let mut spec: Vec<Complex64> = impulse.iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        engine.fft_forward(&mut spec);

        // 2. Compute realized magnitude in dB
        let mut max_err: f64 = 0.0;
        let mut sum_sq_err: f64 = 0.0;
        let mut err_count: usize = 0;

        for k in 0..n_bins {
            let f = k as f64 * df;
            if f < 10.0 || weights[k] == 0.0 { continue; }

            let realized_db = {
                let amp = spec[k].norm();
                if amp > 1e-20 { 20.0 * amp.log10() } else { -200.0 }
            };
            let desired_db = target_correction_db[k];
            let err = desired_db - realized_db;

            // Weighted error correction with damping factor (0.7) for stability
            let correction = err * weights[k] * 0.7;
            refined_db[k] = (refined_db[k] + correction)
                .max(config.noise_floor_db)
                .min(config.max_boost_db);

            let abs_err = err.abs();
            if abs_err > max_err { max_err = abs_err; }
            sum_sq_err += err * err;
            err_count += 1;
        }

        let rms_err = if err_count > 0 { (sum_sq_err / err_count as f64).sqrt() } else { 0.0 };
        info!("iterative_refine: iter={}, max_err={:.3} dB, rms_err={:.3} dB", iter + 1, max_err, rms_err);

        // Early exit if error is already very small
        if max_err < 0.05 {
            info!("iterative_refine: converged at iter {} (max_err < 0.05 dB)", iter + 1);
            break;
        }

        // 3. Rebuild impulse from refined correction
        let mut new_spectrum = assemble_complex_spectrum(&refined_db, phase_rad, n_fft);
        engine.fft_inverse(&mut new_spectrum);

        let norm = 1.0 / n_fft as f64;
        *impulse = new_spectrum.iter().map(|c| c.re * norm).collect();

        // 4. Phase-dependent reordering
        if is_linear_phase {
            circular_shift_to_center(impulse);
        }

        // 5. Apply window (must match initial windowing in generate_model_fir)
        match config.phase_mode {
            PhaseMode::MixedPhase if !config.gaussian_min_phase_filters.is_empty() => {
                // Peak-centered full window (same as generate_model_fir MixedPhase)
                let peak_idx = impulse.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let shift = (n_fft / 2).wrapping_sub(peak_idx) % n_fft;
                impulse.rotate_right(shift);
                let window = generate_window(n_fft, &config.window);
                for (i, w) in window.iter().enumerate() {
                    impulse[i] *= w;
                }
            }
            PhaseMode::MinimumPhase | PhaseMode::MixedPhase | PhaseMode::HybridPhase => {
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
    }
}

// ---------------------------------------------------------------------------
// Internal: narrowband boost limiting
// ---------------------------------------------------------------------------

/// Limit narrowband boost peaks in correction spectrum on a linear frequency grid.
///
/// Strategy: smooth the correction with a 1/3-octave window, then clamp any bin
/// that exceeds the smoothed value by more than `max_excess_db` down to that limit.
/// This prevents the filter from aggressively boosting narrow measurement dips
/// (caused by diffraction, interference, comb filtering) which are position-dependent
/// and harmful to amplify.
pub(crate) fn limit_narrowband_boost(correction_db: &mut [f64], df: f64, smoothing_oct: f64, max_excess_db: f64) {
    let n = correction_db.len();
    if n < 4 || df <= 0.0 { return; }

    // Prefix sum for O(n) windowed average instead of O(n × window_width)
    let mut prefix = vec![0.0_f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + correction_db[i];
    }

    let k_factor = 2.0_f64.powf(smoothing_oct / 2.0);
    let smoothed: Vec<f64> = (0..n).map(|k| {
        let f_center = k as f64 * df;
        if f_center < 20.0 { return correction_db[k]; }

        let f_lo = f_center / k_factor;
        let f_hi = f_center * k_factor;
        let k_lo = (f_lo / df).floor() as usize;
        let k_hi = ((f_hi / df).ceil() as usize).min(n - 1);

        if k_hi <= k_lo { return correction_db[k]; }

        let count = k_hi - k_lo + 1;
        (prefix[k_hi + 1] - prefix[k_lo]) / count as f64
    }).collect();

    // Clamp: where correction exceeds smoothed + max_excess, limit it
    let mut limited_count = 0usize;
    for k in 0..n {
        let limit = smoothed[k] + max_excess_db;
        if correction_db[k] > limit {
            correction_db[k] = limit;
            limited_count += 1;
        }
    }

    if limited_count > 0 {
        info!("limit_narrowband_boost: clamped {} bins (max_excess={:.1} dB)", limited_count, max_excess_db);
    }
}

// ---------------------------------------------------------------------------
// Internal: causality metric
// ---------------------------------------------------------------------------

/// Compute causality metric for an impulse response.
///
/// Returns ratio of post-peak energy to total energy (0.0-1.0).
/// 1.0 = perfectly causal (all energy after peak), lower = more pre-ringing.
///
/// For minimum-phase: typically >0.99
/// For linear-phase: ~0.50 (symmetric around center)
/// For hybrid-phase: between the two
pub(crate) fn compute_causality(impulse: &[f64]) -> f64 {
    if impulse.is_empty() { return 1.0; }

    // Find peak position
    let mut peak_idx = 0;
    let mut peak_val = 0.0_f64;
    for (i, &v) in impulse.iter().enumerate() {
        let abs = v.abs();
        if abs > peak_val {
            peak_val = abs;
            peak_idx = i;
        }
    }

    // Total energy
    let total: f64 = impulse.iter().map(|v| v * v).sum();
    if total < 1e-30 { return 1.0; }

    // Post-peak energy (including peak sample)
    let post: f64 = impulse[peak_idx..].iter().map(|v| v * v).sum();

    post / total
}

// ---------------------------------------------------------------------------
// Internal: build effective target
// ---------------------------------------------------------------------------

/// Build the effective target curve:
/// - Between HP and LP crossover frequencies: use target_mag as-is
/// - Below HP: 1/2-octave smoothed measurement (follow natural rolloff)
/// - Above LP: 1/2-octave smoothed measurement (follow natural rolloff)
/// - Smooth sigmoid blend ±0.5 octave around crossover frequencies
pub(crate) fn build_effective_target(
    freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    crossover_range: (f64, f64),
) -> Vec<f64> {
    let (f_low, f_high) = crossover_range;
    let n = freq.len();

    // Smooth measurement to 1/2 octave for out-of-band regions
    let smoothed_meas: Vec<f64> = (0..n)
        .map(|i| fractional_octave_smooth(freq, meas_mag, i, 0.5))
        .collect();

    // Blend width: ±0.5 octave in log-freq space
    let blend_octaves = 0.5;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let f = freq[i];

        // Sigmoid blend factor: 0 = use smoothed measurement, 1 = use target
        let low_blend = sigmoid_blend(f, f_low, blend_octaves);
        let high_blend = 1.0 - sigmoid_blend(f, f_high, blend_octaves);

        // Combined: within crossover → 1.0, outside → 0.0, transitions → sigmoid
        let blend = low_blend * high_blend;
        let val = smoothed_meas[i] * (1.0 - blend) + target_mag[i] * blend;
        result.push(val);
    }

    result
}

/// Sigmoid blend: returns 0.0 well below `center_freq`, 1.0 well above.
/// Transition width is `octaves` (centered on `center_freq`).
pub(crate) fn sigmoid_blend(freq: f64, center_freq: f64, octaves: f64) -> f64 {
    if freq <= 0.0 || center_freq <= 0.0 {
        return 0.0;
    }
    let log_ratio = (freq / center_freq).log2(); // octaves from center
    let steepness = 6.0 / octaves; // ~6 gives a nice sigmoid over the octave width
    let x = log_ratio * steepness;
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Internal: assemble complex spectrum
// ---------------------------------------------------------------------------

/// Build full N-point complex spectrum with conjugate symmetry.
pub(crate) fn assemble_complex_spectrum(
    correction_db: &[f64],
    phase_rad: &[f64],
    n_fft: usize,
) -> Vec<Complex64> {
    let n_bins = n_fft / 2 + 1;
    let mut spectrum = Vec::with_capacity(n_fft);

    // Positive frequencies (DC to Nyquist)
    for i in 0..n_bins {
        let amp = 10.0_f64.powf(correction_db[i.min(correction_db.len() - 1)] / 20.0);
        let ph = phase_rad[i.min(phase_rad.len() - 1)];
        spectrum.push(Complex64::new(amp * ph.cos(), amp * ph.sin()));
    }

    // Negative frequencies: conjugate mirror
    for i in 1..(n_fft - n_bins + 1) {
        let idx = n_bins - 1 - i;
        spectrum.push(spectrum[idx].conj());
    }

    spectrum
}

// ---------------------------------------------------------------------------
// Internal: circular shift for linear phase
// ---------------------------------------------------------------------------

pub(crate) fn circular_shift_to_center(impulse: &mut Vec<f64>) {
    let n = impulse.len();
    let half = n / 2;
    impulse.rotate_right(half);
}

// ---------------------------------------------------------------------------
// Internal: linear interpolation helper (for mapping between grids)
// ---------------------------------------------------------------------------

/// Linear interpolation: map y_data defined on x_data onto x_query.
/// Out-of-range values are clamped to boundary.
pub(crate) fn interp_1d_simple(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query.iter().map(|&xq| {
        if x_data.is_empty() { return 0.0; }
        if xq <= x_data[0] { return y_data[0]; }
        if xq >= x_data[x_data.len() - 1] { return y_data[y_data.len() - 1]; }
        let idx = match x_data.binary_search_by(|v| v.partial_cmp(&xq).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => return y_data[i],
            Err(i) => i,
        };
        let x0 = x_data[idx - 1];
        let x1 = x_data[idx];
        let y0 = y_data[idx - 1];
        let y1 = y_data[idx];
        let t = (xq - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }).collect()
}
