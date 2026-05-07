// FIR correction engine: internal computation helpers


use num_complex::Complex64;
use tracing::info;

use crate::dsp::fft::FftEngine;
use crate::dsp::fractional_octave_smooth;
use crate::dsp::minimum_phase_from_magnitude;

use super::types::*;
use super::windowing::*;

// Test-only capture for iterative_refine per-iteration errors. Production
// `info!` logging stays untouched; this lets cargo tests see the same
// numbers Kirill saw in the dev-server console.
//
// b140.3.8: thread_local instead of a shared Mutex<Vec<…>>. cargo test runs
// tests in parallel threads and any `generate_model_fir` call pushes here,
// so a global Mutex meant a test's `iter_stats_take()` saw rows from other
// concurrent tests — producing the spurious DIVERGENCE seen in
// `iterative_refine_converges_with_min_phase_subsonic`.
#[cfg(test)]
pub(crate) struct IterStats { pub iter: usize, pub max_err: f64, pub rms_err: f64 }
#[cfg(test)]
thread_local! {
    pub(crate) static ITER_STATS: std::cell::RefCell<Vec<IterStats>>
        = std::cell::RefCell::new(Vec::new());
}
#[cfg(test)]
pub(crate) fn iter_stats_reset() {
    ITER_STATS.with(|s| s.borrow_mut().clear());
}
#[cfg(test)]
pub(crate) fn iter_stats_take() -> Vec<IterStats> {
    ITER_STATS.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

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
    peq_mag_db: &[f64],             // b140.1: linear-grid PEQ magnitude (n_bins)
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

    // b139.4a: must mirror effective_linear in generate_model_fir so the
    // refinement loop reorders/windows the impulse the same way as the
    // initial assembly. Otherwise Composite + linear_main loses the
    // center-shift and gets half-windowed, destroying the symmetric base.
    let is_linear_phase = matches!(config.phase_mode, PhaseMode::LinearPhase)
        || (matches!(config.phase_mode, PhaseMode::Composite) && config.linear_phase_main);
    // b139.3.4: in MinimumPhase mode the (magnitude, phase) pair must stay
    // consistent across iterations — Hilbert(refined_db) recomputed each
    // pass. LinearPhase keeps zero phase; MixedPhase/HybridPhase carry a
    // composite phase from per-filter Hilbert that doesn't depend on the
    // running magnitude, so they reuse the entry-point phase.
    let recompute_min_phase = matches!(config.phase_mode, PhaseMode::MinimumPhase);
    let recompute_composite = matches!(config.phase_mode, PhaseMode::Composite);
    // For Composite mode, the subsonic part is fixed (depends only on
    // cutoff_hz, not on the running magnitude) — precompute it once.
    let subsonic_mag_fixed: Option<Vec<f64>> = if recompute_composite {
        Some(subsonic_mag_db_lin(n_bins, config.sample_rate, config.subsonic_cutoff_hz))
    } else {
        None
    };
    // Buffer reused for the per-iteration recompute when needed. Skip the
    // entry-point copy if we're going to overwrite on iter 0 anyway.
    let mut iter_phase: Vec<f64> = if recompute_min_phase || recompute_composite {
        Vec::with_capacity(n_bins)
    } else {
        phase_rad.to_vec()
    };

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
        #[cfg(test)]
        {
            ITER_STATS.with(|s| s.borrow_mut().push(IterStats { iter: iter + 1, max_err, rms_err }));
        }

        // Early exit if error is already very small
        if max_err < 0.05 {
            info!("iterative_refine: converged at iter {} (max_err < 0.05 dB)", iter + 1);
            break;
        }

        // 3. Rebuild impulse from refined correction. For MinimumPhase,
        //    recompute Hilbert from the refined magnitude so the (mag, phase)
        //    pair is internally consistent — without this the IFFT realises
        //    neither the refined magnitude nor the original phase, and the
        //    next FFT round-trip diverges (b139.3.3 repro: 0.151 → 12.091 dB).
        if recompute_min_phase {
            iter_phase = minimum_phase_from_magnitude(&refined_db, n_fft);
            debug_assert_eq!(iter_phase.len(), n_bins,
                "minimum_phase_from_magnitude must return n_bins entries");
        } else if recompute_composite {
            let sub = subsonic_mag_fixed.as_ref()
                .expect("subsonic_mag_fixed must exist in Composite mode");
            iter_phase = composite_phase_inner(
                &refined_db,
                sub,
                peq_mag_db,
                n_fft,
                config.linear_phase_main,
                config.noise_floor_db,
            );
            debug_assert_eq!(iter_phase.len(), n_bins);
        }
        let mut new_spectrum = assemble_complex_spectrum(&refined_db, &iter_phase, n_fft);
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
            PhaseMode::Composite => {
                // b139.4a: linear_main → centered impulse → full window.
                // min-phase main → asymmetric impulse → half window.
                if config.linear_phase_main {
                    let window = generate_window(n_fft, &config.window);
                    for (i, w) in window.iter().enumerate() {
                        impulse[i] *= w;
                    }
                } else {
                    let half_win = generate_half_window(n_fft, &config.window);
                    for (i, w) in half_win.iter().enumerate() {
                        impulse[i] *= w;
                    }
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
// b139.4a Composite phase: subsonic Butterworth-8 magnitude (linear FFT grid)
// ---------------------------------------------------------------------------

/// Compute Butterworth-8 HP magnitude in dB at the linear FFT grid for
/// the given subsonic cutoff. Mirrors target/mod.rs apply_filter exactly:
/// |H(f)|^2 = 1 / (1 + (fc/f)^16). Returns zeros if cutoff is None.
pub(crate) fn subsonic_mag_db_lin(
    n_bins: usize,
    sample_rate: f64,
    cutoff_hz: Option<f64>,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_bins];
    let Some(fc) = cutoff_hz else { return out; };
    let nyquist = sample_rate / 2.0;
    for k in 0..n_bins {
        let f = nyquist * k as f64 / (n_bins - 1) as f64;
        if f <= 0.0 {
            out[k] = -400.0;
            continue;
        }
        let ratio = (fc / f).powi(16);
        // -10·log10(1 + ratio) = 20·log10(sqrt(1/(1+ratio)))
        out[k] = -10.0 * (1.0 + ratio).log10();
    }
    out
}

/// Composite phase for one iteration / one assembly. b140.1 splits the
/// composed phase into THREE physically independent sources:
///   • main      = total − subsonic − peq  → zero (linear) or Hilbert (min)
///   • peq       = always Hilbert (biquads are min-phase by construction)
///   • subsonic  = always Hilbert (Butterworth-8 HP is min-phase)
/// The pre-fix version subtracted only subsonic, so PEQ phase was bundled
/// into base_mag — and `linear_phase_main = true` zeroed it. That made the
/// exported FIR drop the entire PEQ phase contribution under iterative_refine,
/// reproducible as the b140.1 e2e PEQ-rotation regression.
pub(crate) fn composite_phase_inner(
    total_mag_db: &[f64],
    subsonic_mag_db: &[f64],
    peq_mag_db: &[f64],
    n_fft: usize,
    linear_phase_main: bool,
    noise_floor_db: f64,
) -> Vec<f64> {
    debug_assert_eq!(total_mag_db.len(), subsonic_mag_db.len());
    debug_assert_eq!(total_mag_db.len(), peq_mag_db.len());
    let n = total_mag_db.len();

    // Main = total − subsonic − peq (clamped to noise floor).
    let base_mag: Vec<f64> = (0..n)
        .map(|k| (total_mag_db[k] - subsonic_mag_db[k] - peq_mag_db[k]).max(noise_floor_db))
        .collect();
    let base_phase = if linear_phase_main {
        vec![0.0_f64; n]
    } else {
        minimum_phase_from_magnitude(&base_mag, n_fft)
    };

    // PEQ contribution — Hilbert only when a real PEQ magnitude exists,
    // otherwise the Hilbert of zeros is itself zero (skip the FFT cost).
    let peq_phase = if peq_mag_db.iter().all(|&v| v.abs() < 1e-9) {
        vec![0.0_f64; n]
    } else {
        minimum_phase_from_magnitude(peq_mag_db, n_fft)
    };

    // Subsonic contribution — same skip-when-zero guard.
    let subsonic_phase = if subsonic_mag_db.iter().all(|&v| v.abs() < 1e-9) {
        vec![0.0_f64; n]
    } else {
        minimum_phase_from_magnitude(subsonic_mag_db, n_fft)
    };

    (0..n)
        .map(|k| base_phase[k] + peq_phase[k] + subsonic_phase[k])
        .collect()
}

/// Public composite-phase entry used by generate_model_fir's initial spectrum
/// assembly (before iterative_refine). Builds the subsonic mag internally.
/// b140.1: takes peq_mag_db so the PEQ phase contribution is reconstructed
/// as a third independent Hilbert source, not bundled into base_mag.
pub(crate) fn compose_target_phase(
    total_mag_db: &[f64],
    peq_mag_db: &[f64],
    n_fft: usize,
    n_bins: usize,
    sample_rate: f64,
    linear_phase_main: bool,
    cutoff_hz: Option<f64>,
    noise_floor_db: f64,
) -> Vec<f64> {
    debug_assert_eq!(total_mag_db.len(), n_bins);
    debug_assert_eq!(peq_mag_db.len(), n_bins);
    let subsonic = subsonic_mag_db_lin(n_bins, sample_rate, cutoff_hz);
    composite_phase_inner(
        total_mag_db,
        &subsonic,
        peq_mag_db,
        n_fft,
        linear_phase_main,
        noise_floor_db,
    )
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

// interp_1d_simple removed — use crate::dsp::interp_1d instead
