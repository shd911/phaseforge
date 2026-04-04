// Parametric EQ engine: Levenberg-Marquardt optimization
//
// Unified PEQ optimizer using LMA (Levenberg-Marquardt Algorithm) for
// simultaneous optimization of all band parameters [freq, gain, Q].
//
// Pipeline:
//   1. Compute target:
//      - Standard: 3-zone composite (service curve outside filters, flat in-band)
//      - Hybrid:   flat at avg measurement level for ALL frequencies
//   2. Compute weights:
//      - Standard: ERB-inspired frequency weights with null suppression
//      - Hybrid:   uniform weights (1.0) with null suppression only
//   3. Greedy initialization → starting band placement
//   4. LMA optimization → simultaneous refinement of all parameters
//   5. Band addition for large residuals → re-optimize
//   6. Post-processing: merge + prune + sort

mod types;
mod biquad;
mod lma;
mod greedy;

pub use types::*;
pub use biquad::{apply_peq, peq_band_response, apply_peq_complex};
pub(crate) use lma::{LmaSolver, compute_weights, compute_uniform_weights, apply_exclusion_zones, try_promote_to_shelves};
pub(crate) use greedy::{greedy_fit_adaptive, estimate_q_from_peak_width, mark_exclusion_zone, merge_nearby_bands, compute_max_error_in_range};

use tracing::info;

use crate::dsp::{variable_smoothing, SmoothingConfig};
use crate::error::AppError;

// ---------------------------------------------------------------------------
// Main unified optimizer
// ---------------------------------------------------------------------------

/// Unified PEQ optimizer using Levenberg-Marquardt Algorithm.
///
/// Handles in-band and above-LP correction in a single pass.
/// When `target_mag` is Some, builds a 3-zone composite target:
///   - Below HP: 1/1-octave smoothed measurement (service curve)
///   - HP..LP:   FLAT at max SPL of Target Curve (straight line, no tilt)
///   - Above LP: smoothed measurement with peaks clamped to that flat level
/// Cosine blends at HP and LP boundaries (1 octave each).
/// The tilt/shelves from Target Curve are handled later by FIR.
/// If `target_mag` is None, uses compute_unified_target() fallback.
pub fn auto_peq_lma(
    meas_mag: &[f64],
    target_mag: Option<&[f64]>,
    freq: &[f64],
    config: &PeqConfig,
    hp_freq: f64,
    lp_freq: f64,
    exclusion_zones: &[ExclusionZone],
) -> Result<PeqResult, AppError> {
    let n = freq.len();
    if n == 0 || meas_mag.len() != n {
        return Err(AppError::Dsp {
            message: format!(
                "auto_peq_lma: length mismatch: freq={}, meas={}",
                n, meas_mag.len()
            ),
        });
    }
    if config.freq_range.0 >= config.freq_range.1 {
        return Err(AppError::Config {
            message: format!(
                "auto_peq_lma: invalid freq_range: ({}, {})",
                config.freq_range.0, config.freq_range.1
            ),
        });
    }

    info!(
        "auto_peq_lma: {} points, hp={:.0}, lp={:.0}, range=({:.0}, {:.0}), max_bands={}, hybrid={}",
        n, hp_freq, lp_freq, config.freq_range.0, config.freq_range.1, config.max_bands, config.hybrid
    );

    // 1. Target construction
    let computed_target;
    let target = if config.hybrid {
        // --- Hybrid mode ---
        // Flat target at avg measurement level in HP..LP for ALL frequencies.
        // No 3-zone composite, no service curve, no blend zones.
        // PEQ corrects everything to a straight line; crossover applied later.
        let (sum, cnt) = freq.iter().zip(meas_mag.iter())
            .filter(|(&f, _)| f >= hp_freq && f <= lp_freq)
            .fold((0.0, 0usize), |(s, c), (_, &v)| (s + v, c + 1));
        let flat_level = if cnt > 0 { sum / cnt as f64 } else { 80.0 };

        info!(
            "auto_peq_lma [hybrid]: flat_level = {:.1} dB (avg meas in {:.0}..{:.0} Hz)",
            flat_level, hp_freq, lp_freq
        );

        computed_target = vec![flat_level; n];
        &computed_target
    } else {
        // --- Standard mode ---
        // Flat target at max SPL of Target Curve in HP..LP for ALL frequencies.
        // PEQ corrects measurement toward flat line everywhere (including below HP
        // and above LP) so the exported filter (target + PEQ) properly accounts
        // for the driver's natural rolloff outside the crossover passband.
        // The gain limits (6 dB boost / 18 dB cut) and ERB weights naturally
        // constrain how aggressively the PEQ corrects outside the passband.
        match target_mag {
            Some(t) => {
                if t.len() != n {
                    return Err(AppError::Dsp {
                        message: format!("auto_peq_lma: target length {} != freq length {}", t.len(), n),
                    });
                }

                // Flat level = max SPL of Target Curve between HP and LP
                let flat_level = freq.iter().zip(t.iter())
                    .filter(|(&f, _)| f >= hp_freq && f <= lp_freq)
                    .map(|(_, &v)| v)
                    .fold(f64::NEG_INFINITY, f64::max);
                let flat_level = if flat_level.is_finite() { flat_level } else { 80.0 };

                info!(
                    "auto_peq_lma: flat target level = {:.1} dB (max of Target Curve in {:.0}..{:.0} Hz)",
                    flat_level, hp_freq, lp_freq
                );

                computed_target = vec![flat_level; n];
                &computed_target
            }
            None => {
                // No target_mag provided — use flat at avg measurement level
                let (sum, cnt) = freq.iter().zip(meas_mag.iter())
                    .filter(|(&f, _)| f >= hp_freq && f <= lp_freq)
                    .fold((0.0, 0usize), |(s, c), (_, &v)| (s + v, c + 1));
                let flat_level = if cnt > 0 { sum / cnt as f64 } else { 80.0 };
                computed_target = vec![flat_level; n];
                &computed_target
            }
        }
    };

    // 2. Weights
    let mut weights = if config.hybrid {
        compute_uniform_weights(freq, meas_mag, hp_freq, lp_freq)
    } else {
        compute_weights(freq, meas_mag, hp_freq, lp_freq)
    };

    // 2b. Exclusion zones: zero out weights for excluded frequency ranges
    if !exclusion_zones.is_empty() {
        apply_exclusion_zones(&mut weights, freq, exclusion_zones);
        info!("auto_peq_lma: applied {} exclusion zone(s)", exclusion_zones.len());
    }

    // 3. Greedy initialization
    let raw_error: Vec<f64> = meas_mag.iter().zip(target.iter()).map(|(m, t)| m - t).collect();
    let smooth_config = if let Some(frac) = config.smoothing_fraction {
        SmoothingConfig {
            variable: false,
            fixed_fraction: Some(frac),
        }
    } else {
        SmoothingConfig {
            variable: true,
            fixed_fraction: None,
        }
    };
    let smoothed_error = variable_smoothing(freq, &raw_error, &smooth_config);

    let mut bands = greedy_fit_adaptive(freq, &smoothed_error, config, exclusion_zones);

    // Enforce Q constraints: Q <= 2.5 for bands above LP
    for b in &mut bands {
        if b.freq_hz > lp_freq && b.q > Q_MAX_ABOVE_LP {
            b.q = Q_MAX_ABOVE_LP;
        }
    }

    info!("auto_peq_lma: greedy init -> {} bands", bands.len());

    // 4. Merge nearby bands
    let merge_dist = config.min_band_distance_oct.unwrap_or(MIN_BAND_DISTANCE_OCT);
    merge_nearby_bands(&mut bands, merge_dist);

    // 5. LMA optimization
    let solver = LmaSolver::new(freq, meas_mag, target, &weights, lp_freq, config);
    let mut total_iterations = 0u32;

    if !bands.is_empty() {
        let (opt_bands, iters) = solver.optimize(&bands);
        bands = opt_bands;
        total_iterations += iters;
    }

    info!("auto_peq_lma: after first LMA -> {} bands, {} iters", bands.len(), total_iterations);

    // 6. Band addition loop: check for large residuals, add bands, re-optimize
    for add_round in 0..3 {
        if bands.len() >= config.max_bands {
            break;
        }

        // Find largest weighted residual
        let correction = apply_peq(freq, &bands, SAMPLE_RATE);
        let mut worst_idx = 0;
        let mut worst_val = 0.0_f64;
        let mut available = vec![true; n];

        // Apply user-defined exclusion zones
        for zone in exclusion_zones {
            for (i, &f) in freq.iter().enumerate() {
                if f >= zone.start_hz && f <= zone.end_hz {
                    available[i] = false;
                }
            }
        }
        // Mark zones around existing bands as unavailable
        for b in &bands {
            mark_exclusion_zone(freq, b.freq_hz, merge_dist, &mut available);
        }

        for i in 0..n {
            if freq[i] < config.freq_range.0 || freq[i] > config.freq_range.1 || !available[i] {
                continue;
            }
            let err = meas_mag[i] + correction[i] - target[i];
            let w_err = err.abs() * weights[i].sqrt();
            // Bias: peaks (positive error) are more important
            let biased = if err > 0.0 { w_err * config.peak_bias } else { w_err };
            if biased > worst_val {
                worst_val = biased;
                worst_idx = i;
            }
        }

        // Raw error at worst point
        let raw_err = meas_mag[worst_idx] + correction[worst_idx] - target[worst_idx];

        if raw_err.abs() < config.tolerance_db * LMA_ADD_BAND_FACTOR {
            info!("auto_peq_lma: add round {}: residual {:.1} dB < threshold, stopping", add_round, raw_err.abs());
            break;
        }

        // Add a new band at worst residual
        let new_fc = freq[worst_idx];
        let new_gain = (-raw_err).clamp(-config.max_cut_db, config.max_boost_db);
        let new_q = if new_fc > lp_freq {
            1.0_f64 // wide filter above LP
        } else {
            estimate_q_from_peak_width(freq, &{
                let c = apply_peq(freq, &bands, SAMPLE_RATE);
                meas_mag.iter().zip(target.iter()).zip(c.iter()).map(|((m, t), cr)| m + cr - t).collect::<Vec<_>>()
            }, worst_idx).min(Q_MAX)
        };

        bands.push(PeqBand {
            freq_hz: new_fc,
            gain_db: new_gain,
            q: new_q,
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        });

        info!(
            "auto_peq_lma: add round {}: added band at {:.0} Hz, gain={:.1} dB, Q={:.1}",
            add_round, new_fc, new_gain, new_q
        );

        // Re-optimize all bands together
        let (opt_bands, iters) = solver.optimize(&bands);
        bands = opt_bands;
        total_iterations += iters;
    }

    // 7. Post-processing
    // Merge nearby bands that LMA may have pushed together
    merge_nearby_bands(&mut bands, merge_dist * 0.8);

    // Remove weak bands
    bands.retain(|b| b.gain_db.abs() >= LMA_MIN_GAIN_DB);

    // Sort by frequency
    bands.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap_or(std::cmp::Ordering::Equal));

    // Final Q enforcement
    for b in &mut bands {
        if b.freq_hz > lp_freq {
            b.q = b.q.clamp(Q_MIN, Q_MAX_ABOVE_LP);
        } else {
            b.q = b.q.clamp(Q_MIN, Q_MAX);
        }
    }

    // 8. Shelf promotion: try converting edge bands to LowShelf / HighShelf
    try_promote_to_shelves(&mut bands, freq, meas_mag, target, config);

    // Compute final max error
    let correction = apply_peq(freq, &bands, SAMPLE_RATE);
    let max_err = compute_max_error_in_range(freq, meas_mag, target, &correction, config.freq_range);

    info!(
        "auto_peq_lma: result: {} bands, max_error={:.1} dB",
        bands.len(), max_err
    );
    for (i, b) in bands.iter().enumerate() {
        info!(
            "  band[{}]: freq={:.0} Hz, gain={:.1} dB, Q={:.2}",
            i, b.freq_hz, b.gain_db, b.q
        );
    }

    Ok(PeqResult {
        bands,
        max_error_db: max_err,
        iterations: total_iterations,
    })
}

// ---------------------------------------------------------------------------
// Public API (backward-compatible wrappers + direct functions)
// ---------------------------------------------------------------------------

/// Auto-fit PEQ bands to minimize error between measurement and target.
/// Backward-compatible wrapper: delegates to unified LMA optimizer.
pub fn auto_peq(
    meas_mag: &[f64],
    target_mag: &[f64],
    freq: &[f64],
    config: &PeqConfig,
) -> Result<PeqResult, AppError> {
    info!("auto_peq: delegating to auto_peq_lma");
    auto_peq_lma(
        meas_mag,
        Some(target_mag),
        freq,
        config,
        config.freq_range.0,
        config.freq_range.1,
        &[],
    )
}

/// Compute median magnitude (dB) in a frequency range [f_low, f_high].
/// Returns None if no data points fall within the range.
pub fn compute_median_in_range(freq: &[f64], mag: &[f64], f_low: f64, f_high: f64) -> Option<f64> {
    let mut values: Vec<f64> = freq
        .iter()
        .zip(mag)
        .filter(|(&f, _)| f >= f_low && f <= f_high)
        .map(|(_, &m)| m)
        .collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

// ---------------------------------------------------------------------------
// Public API: Above-LP Peak Reduction (Algorithm 2)
// ---------------------------------------------------------------------------

/// Auto-PEQ above LP frequency: cut peaks above LP down to median in-band level.
/// Backward-compatible wrapper: delegates to unified LMA optimizer with cuts-only config.
pub fn auto_peq_above_lp(
    meas_mag: &[f64],
    freq: &[f64],
    config: &PeqConfig,
    lp_freq: f64,
    hp_freq: f64,
) -> Result<PeqResult, AppError> {
    info!("auto_peq_above_lp: delegating to auto_peq_lma");

    let n = freq.len();
    if n == 0 || meas_mag.len() != n {
        return Err(AppError::Dsp {
            message: format!(
                "auto_peq_above_lp: length mismatch: freq={}, meas={}",
                n, meas_mag.len()
            ),
        });
    }

    // Compute flat target at median level above LP
    let ref_level = compute_median_in_range(freq, meas_mag, hp_freq, lp_freq).ok_or_else(
        || AppError::Dsp {
            message: "auto_peq_above_lp: no data in HP-LP range for median".into(),
        },
    )?;

    let f_max = freq.last().copied().unwrap_or(20000.0).min(20000.0);

    // Target: flat at ref_level above LP, smoothed measurement below
    let smooth_cfg = SmoothingConfig {
        variable: false,
        fixed_fraction: Some(1.0 / 6.0),
    };
    let smoothed = variable_smoothing(freq, meas_mag, &smooth_cfg);
    let target: Vec<f64> = (0..n)
        .map(|i| if freq[i] > lp_freq { ref_level } else { smoothed[i] })
        .collect();

    // Configure for above-LP: cuts only (max_boost_db = 0), range from HP up
    let above_config = PeqConfig {
        max_bands: config.max_bands,
        tolerance_db: config.tolerance_db,
        peak_bias: config.peak_bias,
        max_boost_db: 0.0, // Cuts only
        max_cut_db: config.max_cut_db,
        freq_range: (lp_freq * 0.85, f_max), // start slightly below LP for wide filter overlap
        smoothing_fraction: Some(1.0 / 3.0), // 1/3 octave smoothing for above-LP
        min_band_distance_oct: config.min_band_distance_oct,
        hybrid: false,
        gain_regularization: config.gain_regularization,
    };

    auto_peq_lma(meas_mag, Some(&target), freq, &above_config, hp_freq, lp_freq, &[])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::biquad::biquad_peaking_mag_db;
    use super::lma::cholesky_solve;

    fn make_log_freq(n: usize, f_min: f64, f_max: f64) -> Vec<f64> {
        let log_min = f_min.ln();
        let log_max = f_max.ln();
        (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect()
    }

    #[test]
    fn test_biquad_at_center_frequency() {
        let gain = 6.0;
        let fc = 1000.0;
        let q = 4.0;
        let sr = 48000.0;
        let result = biquad_peaking_mag_db(fc, fc, gain, q, sr);
        assert!(
            (result - gain).abs() < 0.01,
            "At center freq, gain should be {gain} dB, got {result}"
        );
    }

    #[test]
    fn test_biquad_at_center_frequency_negative_gain() {
        let gain = -8.0;
        let fc = 500.0;
        let q = 3.0;
        let sr = 48000.0;
        let result = biquad_peaking_mag_db(fc, fc, gain, q, sr);
        assert!(
            (result - gain).abs() < 0.01,
            "At center freq, gain should be {gain} dB, got {result}"
        );
    }

    #[test]
    fn test_biquad_far_from_center() {
        let result = biquad_peaking_mag_db(100.0, 10000.0, 10.0, 5.0, 48000.0);
        assert!(
            result.abs() < 0.5,
            "Far from center, gain should be ~0 dB, got {result}"
        );
    }

    #[test]
    fn test_biquad_zero_gain() {
        let result = biquad_peaking_mag_db(1000.0, 1000.0, 0.0, 4.0, 48000.0);
        assert!(
            result.abs() < 1e-10,
            "Zero gain should give 0 dB, got {result}"
        );
    }

    #[test]
    fn test_apply_peq_empty() {
        let freq = make_log_freq(100, 20.0, 20000.0);
        let result = apply_peq(&freq, &[], 48000.0);
        assert_eq!(result.len(), 100);
        for v in &result {
            assert!(v.abs() < 1e-10, "Empty PEQ should return all zeros");
        }
    }

    #[test]
    fn test_apply_peq_single_band() {
        let freq = make_log_freq(200, 20.0, 20000.0);
        let band = PeqBand {
            freq_hz: 1000.0,
            gain_db: -6.0,
            q: 4.0,
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        };
        let result = apply_peq(&freq, &[band], 48000.0);
        let idx = freq
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - 1000.0).abs())
                    .partial_cmp(&((**b - 1000.0).abs()))
                    .unwrap()
            })
            .unwrap()
            .0;
        assert!(
            (result[idx] - (-6.0)).abs() < 0.1,
            "Single band at 1kHz should give -6dB at 1kHz, got {}",
            result[idx]
        );
    }

    #[test]
    fn test_auto_peq_flat_error() {
        let freq = make_log_freq(200, 20.0, 20000.0);
        let mag = vec![80.0; 200];
        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq(&mag, &mag, &freq, &config).unwrap();
        assert!(
            result.bands.is_empty(),
            "Flat error should produce 0 bands, got {}",
            result.bands.len()
        );
    }

    #[test]
    fn test_auto_peq_single_peak() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let target = vec![80.0; 500];
        let mut meas = vec![80.0; 500];

        for (i, &f) in freq.iter().enumerate() {
            let log_ratio = (f / 1000.0).log2();
            meas[i] = 80.0 + 10.0 * (-log_ratio * log_ratio * 8.0).exp();
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq(&meas, &target, &freq, &config).unwrap();

        assert!(
            !result.bands.is_empty(),
            "Should produce at least 1 band for a 10dB peak"
        );

        // All bands should have unique frequencies (no duplicates)
        for i in 0..result.bands.len() {
            for j in (i + 1)..result.bands.len() {
                let oct_dist = (result.bands[i].freq_hz / result.bands[j].freq_hz)
                    .log2()
                    .abs();
                assert!(
                    oct_dist > 0.1,
                    "Bands should not overlap: #{} at {:.0} Hz and #{} at {:.0} Hz",
                    i, result.bands[i].freq_hz, j, result.bands[j].freq_hz
                );
            }
        }

        // The dominant band should be near 1 kHz with negative gain
        let dominant = result
            .bands
            .iter()
            .max_by(|a, b| {
                a.gain_db
                    .abs()
                    .partial_cmp(&b.gain_db.abs())
                    .unwrap()
            })
            .unwrap();
        assert!(
            dominant.freq_hz > 500.0 && dominant.freq_hz < 2000.0,
            "Dominant PEQ should be near 1kHz, got {} Hz",
            dominant.freq_hz
        );
        assert!(
            dominant.gain_db < 0.0,
            "Dominant PEQ should have negative gain (cut), got {} dB",
            dominant.gain_db
        );

        assert!(
            result.max_error_db < 5.0,
            "Max error should be significantly reduced, got {} dB",
            result.max_error_db
        );
    }

    #[test]
    fn test_auto_peq_length_mismatch() {
        let freq = vec![100.0, 200.0, 300.0];
        let meas = vec![80.0, 80.0];
        let target = vec![80.0, 80.0, 80.0];
        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (50.0, 500.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        assert!(auto_peq(&meas, &target, &freq, &config).is_err());
    }

    #[test]
    fn test_auto_peq_invalid_range() {
        let freq = vec![100.0, 200.0, 300.0];
        let mag = vec![80.0; 3];
        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (500.0, 50.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        assert!(auto_peq(&mag, &mag, &freq, &config).is_err());
    }

    #[test]
    fn test_pruning_removes_redundant() {
        let freq = make_log_freq(300, 20.0, 20000.0);
        let target = vec![80.0; 300];
        let mut meas = vec![80.0; 300];

        for (i, &f) in freq.iter().enumerate() {
            let log_ratio = (f / 1000.0).log2();
            meas[i] = 80.0 + 5.0 * (-log_ratio * log_ratio * 16.0).exp();
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 2.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq(&meas, &target, &freq, &config).unwrap();

        assert!(
            result.bands.len() <= 3,
            "Should keep few bands for moderate peak, got {}",
            result.bands.len()
        );
    }

    #[test]
    fn test_merge_nearby_bands() {
        let mut bands = vec![
            PeqBand { freq_hz: 100.0, gain_db: -3.0, q: 2.0, enabled: true, filter_type: PeqFilterType::Peaking },
            PeqBand { freq_hz: 105.0, gain_db: -2.0, q: 3.0, enabled: true, filter_type: PeqFilterType::Peaking },
            PeqBand { freq_hz: 1000.0, gain_db: 4.0, q: 5.0, enabled: true, filter_type: PeqFilterType::Peaking },
        ];
        merge_nearby_bands(&mut bands, MIN_BAND_DISTANCE_OCT);
        // 100 and 105 should merge (within 1/3 octave)
        // 1000 should stay separate
        assert_eq!(bands.len(), 2, "100+105 should merge, 1000 stays: got {} bands", bands.len());
        // Merged gain should be sum: -3 + -2 = -5
        assert!((bands[0].gain_db - (-5.0)).abs() < 0.1, "Merged gain should be -5, got {}", bands[0].gain_db);
    }

    #[test]
    fn test_no_duplicate_frequencies() {
        // Realistic scenario: complex error curve with multiple peaks
        let freq = make_log_freq(500, 20.0, 20000.0);
        let target = vec![80.0; 500];
        let mut meas = vec![80.0; 500];

        // Multiple peaks at different frequencies
        for (i, &f) in freq.iter().enumerate() {
            let p1 = 6.0 * (-(((f / 200.0).log2()) * 4.0).powi(2)).exp();
            let p2 = -4.0 * (-(((f / 800.0).log2()) * 3.0).powi(2)).exp();
            let p3 = 8.0 * (-(((f / 3000.0).log2()) * 5.0).powi(2)).exp();
            meas[i] = 80.0 + p1 + p2 + p3;
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq(&meas, &target, &freq, &config).unwrap();

        // Check no duplicate frequencies (all pairs should be > 1/4 octave apart)
        for i in 0..result.bands.len() {
            for j in (i + 1)..result.bands.len() {
                let oct_dist = (result.bands[i].freq_hz / result.bands[j].freq_hz)
                    .log2()
                    .abs();
                assert!(
                    oct_dist > 0.15,
                    "Bands too close: #{} at {:.0} Hz and #{} at {:.0} Hz (dist={:.2} oct)",
                    i, result.bands[i].freq_hz, j, result.bands[j].freq_hz, oct_dist
                );
            }
        }
    }

    #[test]
    fn test_estimate_q_from_peak_width() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        // Create a narrow peak error
        let error: Vec<f64> = freq
            .iter()
            .map(|&f| {
                let log_ratio = (f / 1000.0).log2();
                5.0 * (-log_ratio * log_ratio * 20.0).exp() // narrow peak
            })
            .collect();
        let peak_idx = freq
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - 1000.0).abs())
                    .partial_cmp(&((**b - 1000.0).abs()))
                    .unwrap()
            })
            .unwrap()
            .0;

        let q = estimate_q_from_peak_width(&freq, &error, peak_idx);
        assert!(
            q >= Q_MIN && q <= Q_MAX,
            "Q should be within bounds, got {}",
            q
        );
        // Narrow peak should produce higher Q
        assert!(
            q > 1.5,
            "Narrow peak should produce Q > 1.5, got {}",
            q
        );
    }

    // ------- Algorithm 2: Above-LP tests -------

    #[test]
    fn test_compute_median_simple() {
        let freq = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let mag = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_median_in_range(&freq, &mag, 100.0, 500.0);
        assert_eq!(result, Some(3.0));
    }

    #[test]
    fn test_compute_median_even() {
        let freq = vec![100.0, 200.0, 300.0, 400.0];
        let mag = vec![1.0, 2.0, 3.0, 4.0];
        let result = compute_median_in_range(&freq, &mag, 100.0, 400.0);
        assert_eq!(result, Some(2.5));
    }

    #[test]
    fn test_compute_median_empty_range() {
        let freq = vec![100.0, 200.0, 300.0];
        let mag = vec![1.0, 2.0, 3.0];
        let result = compute_median_in_range(&freq, &mag, 500.0, 1000.0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_auto_peq_above_lp_cuts_only() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let mut meas = vec![80.0; 500];

        // Add peaks above 3000 Hz
        for (i, &f) in freq.iter().enumerate() {
            if f > 3000.0 {
                let p = 8.0 * (-(((f / 5000.0).log2()) * 3.0).powi(2)).exp();
                meas[i] = 80.0 + p;
            }
        }

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0), // will be overridden
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq_above_lp(&meas, &freq, &config, 3000.0, 80.0).unwrap();

        // All bands should have negative gain (cuts only)
        for b in &result.bands {
            assert!(
                b.gain_db <= 0.0,
                "Above-LP bands should be cuts only, got {} dB at {} Hz",
                b.gain_db, b.freq_hz
            );
        }
        // Should have at least 1 band for the peak
        assert!(
            !result.bands.is_empty(),
            "Should produce at least 1 band for peak above LP"
        );
    }

    #[test]
    fn test_auto_peq_above_lp_flat_no_bands() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500]; // flat

        let config = PeqConfig {
            max_bands: 20,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq_above_lp(&meas, &freq, &config, 3000.0, 80.0).unwrap();

        assert!(
            result.bands.is_empty(),
            "Flat measurement should produce 0 bands above LP, got {}",
            result.bands.len()
        );
    }

    // ------- LMA-specific tests -------

    #[test]
    fn test_cholesky_solve_identity() {
        // A = I, b = [1, 2, 3] -> x = [1, 2, 3]
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve(&a, &b).unwrap();
        assert_eq!(x.len(), 3);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // Solution: x = [-1/8, 3/4] = [-0.125, 0.75]
        let a = vec![
            vec![4.0, 2.0],
            vec![2.0, 3.0],
        ];
        let b = vec![1.0, 2.0];
        let x = cholesky_solve(&a, &b).unwrap();
        assert_eq!(x.len(), 2);
        assert!((x[0] - (-0.125)).abs() < 1e-10, "x[0] = {}, expected -0.125", x[0]);
        assert!((x[1] - 0.75).abs() < 1e-10, "x[1] = {}, expected 0.75", x[1]);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite matrix
        let a = vec![
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let b = vec![1.0, 2.0];
        assert!(cholesky_solve(&a, &b).is_none());
    }

    #[test]
    fn test_weight_function_erb() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500];
        let weights = compute_weights(&freq, &meas, 80.0, 3000.0);
        assert_eq!(weights.len(), 500);

        // Check weights in different bands
        for (i, &f) in freq.iter().enumerate() {
            if f < 150.0 {
                assert!((weights[i] - 1.0).abs() < 0.01, "Low freq weight should be 1.0, got {} at {:.0} Hz", weights[i], f);
            } else if f > 800.0 && f < 3000.0 {
                // Below LP, max ear sensitivity -> 2.0
                assert!((weights[i] - 2.0).abs() < 0.01, "Mid freq weight should be 2.0, got {} at {:.0} Hz", weights[i], f);
            } else if f > 5000.0 {
                // Above LP: 0.5 x 0.6 = 0.3
                assert!((weights[i] - 0.3).abs() < 0.01, "HF above-LP weight should be 0.3, got {} at {:.0} Hz", weights[i], f);
            }
        }
    }

    #[test]
    fn test_weight_null_suppression() {
        let freq = make_log_freq(100, 20.0, 20000.0);
        let mut meas = vec![80.0; 100];
        // Create a deep null at point 50
        meas[50] = 60.0; // 20 dB below median

        let weights = compute_weights(&freq, &meas, 80.0, 3000.0);
        assert!(
            weights[50] < 0.01,
            "Deep null should have near-zero weight, got {}",
            weights[50]
        );
    }

    #[test]
    fn test_lma_single_peak() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let target = vec![80.0; 500];
        let mut meas = vec![80.0; 500];

        // Add a single 8 dB peak at 1 kHz
        for (i, &f) in freq.iter().enumerate() {
            let log_ratio = (f / 1000.0).log2();
            meas[i] = 80.0 + 8.0 * (-log_ratio * log_ratio * 10.0).exp();
        }

        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq_lma(&meas, Some(&target), &freq, &config, 80.0, 15000.0, &[]).unwrap();

        assert!(
            !result.bands.is_empty(),
            "Single peak should produce at least 1 band"
        );
        // Dominant band near 1 kHz with negative gain
        let dominant = result.bands.iter()
            .max_by(|a, b| a.gain_db.abs().partial_cmp(&b.gain_db.abs()).unwrap())
            .unwrap();
        assert!(
            dominant.freq_hz > 500.0 && dominant.freq_hz < 2000.0,
            "Dominant band should be near 1 kHz, got {:.0} Hz",
            dominant.freq_hz
        );
        assert!(
            dominant.gain_db < 0.0,
            "Dominant band should cut, got {:.1} dB",
            dominant.gain_db
        );
    }

    #[test]
    fn test_lma_flat_no_bands() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let meas = vec![80.0; 500];

        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (80.0, 15000.0),
            smoothing_fraction: None,
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };
        let result = auto_peq_lma(&meas, Some(&meas), &freq, &config, 80.0, 15000.0, &[]).unwrap();

        assert!(
            result.bands.is_empty(),
            "Flat meas = flat target should produce 0 bands, got {}",
            result.bands.len()
        );
    }

    #[test]
    fn test_lma_q_constraint_above_lp() {
        let freq = make_log_freq(500, 20.0, 20000.0);
        let mut meas = vec![80.0; 500];

        // Add narrow peaks above 3 kHz (LP)
        for (i, &f) in freq.iter().enumerate() {
            if f > 3000.0 {
                let p1 = 10.0 * (-(((f / 5000.0).log2()) * 6.0).powi(2)).exp();
                let p2 = 8.0 * (-(((f / 8000.0).log2()) * 6.0).powi(2)).exp();
                meas[i] = 80.0 + p1 + p2;
            }
        }

        let config = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 0.0, // cuts only
            max_cut_db: 18.0,
            freq_range: (2500.0, 20000.0),
            smoothing_fraction: Some(1.0 / 3.0),
            min_band_distance_oct: None,
            hybrid: false,
            gain_regularization: 0.0,
        };

        let target: Vec<f64> = freq.iter().map(|_| 80.0).collect();
        let result = auto_peq_lma(&meas, Some(&target), &freq, &config, 80.0, 3000.0, &[]).unwrap();

        // All bands above LP should have Q <= 2.5
        for b in &result.bands {
            if b.freq_hz > 3000.0 {
                assert!(
                    b.q <= Q_MAX_ABOVE_LP + 0.01,
                    "Band at {:.0} Hz above LP should have Q <= {}, got Q={:.2}",
                    b.freq_hz, Q_MAX_ABOVE_LP, b.q
                );
            }
        }
    }
}
