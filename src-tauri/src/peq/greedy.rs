// Greedy fitting + band management

use super::types::*;
use super::biquad::*;

// ---------------------------------------------------------------------------
// Internal: Greedy Fitting with Adaptive Q
// ---------------------------------------------------------------------------

/// Greedy fitting with adaptive Q estimated from error curve peak width,
/// and exclusion zones to prevent duplicate bands.
pub(crate) fn greedy_fit_adaptive(freq: &[f64], smoothed_error: &[f64], config: &PeqConfig, exclusion_zones: &[ExclusionZone]) -> Vec<PeqBand> {
    let n = freq.len();
    let mut bands: Vec<PeqBand> = Vec::new();
    let mut current_error = smoothed_error.to_vec();
    // Exclusion mask: true = available, false = excluded
    let mut available = vec![true; n];
    // Apply user-defined exclusion zones
    for zone in exclusion_zones {
        for (i, &f) in freq.iter().enumerate() {
            if f >= zone.start_hz && f <= zone.end_hz {
                available[i] = false;
            }
        }
    }
    let exclusion_oct = config.min_band_distance_oct.unwrap_or(MIN_BAND_DISTANCE_OCT);

    for _ in 0..config.max_bands {
        // Find the peak error within frequency range, excluding masked points
        let peak = find_peak_error_masked(freq, &current_error, config.peak_bias, config.freq_range, &available);
        let (peak_idx, peak_val) = match peak {
            Some(p) => p,
            None => break,
        };

        // Convergence check
        if peak_val.abs() < config.tolerance_db {
            break;
        }

        let peak_freq = freq[peak_idx];

        // Estimate Q from the width of the error peak (adaptive)
        let q = estimate_q_from_peak_width(freq, &current_error, peak_idx);

        // Gain = negative of error (to correct it), clamped to limits
        let mut gain = -peak_val;
        if gain > 0.0 {
            gain = gain.min(config.max_boost_db);
        } else {
            gain = gain.max(-config.max_cut_db);
        }

        let band = PeqBand {
            freq_hz: peak_freq,
            gain_db: gain,
            q,
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        };

        // Apply this band's response to the working error
        let response = peq_band_response(freq, &band, SAMPLE_RATE);
        for i in 0..n {
            current_error[i] += response[i];
        }

        // Mark exclusion zone around placed band
        mark_exclusion_zone(freq, peak_freq, exclusion_oct, &mut available);

        bands.push(band);
    }

    bands
}

/// Estimate Q from the half-power width of the error peak.
///
/// Measures how many octaves the error peak extends at half its amplitude,
/// then converts to Q: Q ~ fc / bandwidth_hz
pub(crate) fn estimate_q_from_peak_width(freq: &[f64], error: &[f64], peak_idx: usize) -> f64 {
    let peak_val = error[peak_idx].abs();
    let half_val = peak_val * 0.5; // -6 dB (half amplitude in dB)
    let peak_freq = freq[peak_idx];

    // Search left for half-amplitude crossing
    let mut left_freq = freq[0];
    for i in (0..peak_idx).rev() {
        if error[i].abs() <= half_val {
            // Interpolate between i and i+1
            let f0 = freq[i];
            let f1 = freq[i + 1];
            let e0 = error[i].abs();
            let e1 = error[i + 1].abs();
            let t = if (e1 - e0).abs() > 1e-10 {
                (half_val - e0) / (e1 - e0)
            } else {
                0.5
            };
            left_freq = f0 + t * (f1 - f0);
            break;
        }
    }

    // Search right for half-amplitude crossing
    let mut right_freq = freq[freq.len() - 1];
    for i in (peak_idx + 1)..freq.len() {
        if error[i].abs() <= half_val {
            let f0 = freq[i - 1];
            let f1 = freq[i];
            let e0 = error[i - 1].abs();
            let e1 = error[i].abs();
            let t = if (e0 - e1).abs() > 1e-10 {
                (e0 - half_val) / (e0 - e1)
            } else {
                0.5
            };
            right_freq = f0 + t * (f1 - f0);
            break;
        }
    }

    // Bandwidth in Hz
    let bw_hz = (right_freq - left_freq).max(1.0);

    // Q = fc / bw (clamped to frequency-dependent envelope; b137).
    let q = (peak_freq / bw_hz).clamp(Q_MIN, crate::peq::q_cap_at(peak_freq));
    q
}

/// Mark an exclusion zone around a frequency (+-octaves distance).
pub(crate) fn mark_exclusion_zone(freq: &[f64], center_freq: f64, octaves: f64, available: &mut [bool]) {
    let ratio = 2.0_f64.powf(octaves);
    let f_low = center_freq / ratio;
    let f_high = center_freq * ratio;
    for (i, &f) in freq.iter().enumerate() {
        if f >= f_low && f <= f_high {
            available[i] = false;
        }
    }
}

/// Find peak error with exclusion mask.
pub(crate) fn find_peak_error_masked(
    freq: &[f64],
    error: &[f64],
    peak_bias: f64,
    freq_range: (f64, f64),
    available: &[bool],
) -> Option<(usize, f64)> {
    let mut best_idx = 0;
    let mut best_weighted = 0.0_f64;
    let mut best_raw = 0.0_f64;
    let mut found = false;

    for (i, (&f, &e)) in freq.iter().zip(error).enumerate() {
        if f < freq_range.0 || f > freq_range.1 || !available[i] {
            continue;
        }
        let weighted = if e > 0.0 { e * peak_bias } else { e.abs() };
        if !found || weighted > best_weighted {
            best_weighted = weighted;
            best_raw = e;
            best_idx = i;
            found = true;
        }
    }

    if found && best_raw.abs() > 0.01 {
        Some((best_idx, best_raw))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Internal: Merge Nearby Bands
// ---------------------------------------------------------------------------

/// Merge PEQ bands that are within `merge_distance_oct` octaves of each other.
/// Combines them by weighted-average frequency and summed gain.
/// Keeps the wider Q (lower Q value).
pub(crate) fn merge_nearby_bands(bands: &mut Vec<PeqBand>, merge_distance_oct: f64) {
    if bands.len() < 2 {
        return;
    }

    // Sort by frequency for merging
    bands.sort_by(|a, b| a.freq_hz.partial_cmp(&b.freq_hz).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<PeqBand> = Vec::with_capacity(bands.len());
    let mut i = 0;

    while i < bands.len() {
        let mut group_freq_sum = bands[i].freq_hz * bands[i].gain_db.abs();
        let mut group_gain = bands[i].gain_db;
        let mut group_weight = bands[i].gain_db.abs();
        let mut group_q_min = bands[i].q; // keep widest Q (lowest value)
        let mut j = i + 1;

        while j < bands.len() {
            let octave_dist = (bands[j].freq_hz / bands[i].freq_hz).log2().abs();
            // Also check against the current group center
            let group_center = if group_weight > 0.0 {
                group_freq_sum / group_weight
            } else {
                bands[i].freq_hz
            };
            let dist_to_center = (bands[j].freq_hz / group_center).log2().abs();

            if octave_dist <= merge_distance_oct || dist_to_center <= merge_distance_oct {
                // Same-sign gains -> sum; opposite-sign -> sum (they partially cancel, that's OK)
                group_gain += bands[j].gain_db;
                group_freq_sum += bands[j].freq_hz * bands[j].gain_db.abs();
                group_weight += bands[j].gain_db.abs();
                if bands[j].q < group_q_min {
                    group_q_min = bands[j].q;
                }
                j += 1;
            } else {
                break;
            }
        }

        // Weighted average frequency
        let merged_freq = if group_weight > 0.0 {
            group_freq_sum / group_weight
        } else {
            bands[i].freq_hz
        };

        merged.push(PeqBand {
            freq_hz: merged_freq,
            gain_db: group_gain,
            q: group_q_min.clamp(Q_MIN, crate::peq::q_cap_at(merged_freq)),
            enabled: true,
            filter_type: PeqFilterType::Peaking,
        });

        i = j;
    }

    *bands = merged;
}

// ---------------------------------------------------------------------------
// Internal: Helpers
// ---------------------------------------------------------------------------

/// Compute the maximum absolute error in the frequency range after applying correction.
pub(crate) fn compute_max_error_in_range(
    freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    correction: &[f64],
    freq_range: (f64, f64),
) -> f64 {
    let mut max_err = 0.0_f64;
    for (i, &f) in freq.iter().enumerate() {
        if f < freq_range.0 || f > freq_range.1 {
            continue;
        }
        let corrected = meas_mag[i] + correction[i];
        let err = (corrected - target_mag[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}
