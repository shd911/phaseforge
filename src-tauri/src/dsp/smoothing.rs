use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingConfig {
    /// Use variable smoothing (frequency-dependent)
    pub variable: bool,
    /// Fixed octave fraction (used when variable=false)
    pub fixed_fraction: Option<f64>,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            variable: true,
            fixed_fraction: None,
        }
    }
}

/// Apply variable fractional-octave smoothing.
///
/// Smoothing fraction depends on frequency:
/// - < 100 Hz: 1/48 octave (detailed for room modes)
/// - 100–500 Hz: 1/12 octave (moderate)
/// - > 500 Hz: 1/3 octave (coarse for HF)
pub fn variable_smoothing(freq: &[f64], mag: &[f64], config: &SmoothingConfig) -> Vec<f64> {
    let n = freq.len();
    if n == 0 { return vec![]; }

    // Prefix sum for O(1) range queries
    let mut prefix = vec![0.0_f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + mag[i];
    }

    freq.iter()
        .enumerate()
        .map(|(i, &f)| {
            let fraction = if config.variable {
                if f < 100.0 {
                    1.0 / 48.0
                } else if f < 500.0 {
                    1.0 / 12.0
                } else {
                    1.0 / 3.0
                }
            } else {
                config.fixed_fraction.unwrap_or(1.0 / 6.0)
            };
            smooth_bin_prefix(freq, &prefix, i, fraction)
        })
        .collect()
}

/// Smooth a single frequency bin using fractional-octave window.
/// Uses binary search + prefix sum for O(log n) per bin instead of O(n).
pub fn fractional_octave_smooth(freq: &[f64], mag: &[f64], idx: usize, fraction: f64) -> f64 {
    let center = freq[idx];
    if center <= 0.0 {
        return mag[idx];
    }

    let k = 2.0_f64.powf(fraction / 2.0);
    let f_low = center / k;
    let f_high = center * k;

    let lo = freq.partition_point(|&f| f < f_low);
    let hi = freq.partition_point(|&f| f <= f_high);

    if hi <= lo {
        return mag[idx];
    }

    let mut sum = 0.0;
    for i in lo..hi {
        sum += mag[i];
    }
    sum / (hi - lo) as f64
}

/// Internal: smooth using pre-computed prefix sum (O(log n) per bin)
fn smooth_bin_prefix(freq: &[f64], prefix: &[f64], idx: usize, fraction: f64) -> f64 {
    let center = freq[idx];
    if center <= 0.0 {
        // prefix[idx+1] - prefix[idx] == mag[idx]
        return prefix[idx + 1] - prefix[idx];
    }

    let k = 2.0_f64.powf(fraction / 2.0);
    let f_low = center / k;
    let f_high = center * k;

    let lo = freq.partition_point(|&f| f < f_low);
    let hi = freq.partition_point(|&f| f <= f_high);

    if hi <= lo {
        return prefix[idx + 1] - prefix[idx];
    }

    (prefix[hi] - prefix[lo]) / (hi - lo) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothing_identity() {
        // Flat data should remain flat after smoothing
        let freq: Vec<f64> = (0..100).map(|i| 20.0 * (20000.0 / 20.0_f64).powf(i as f64 / 99.0)).collect();
        let mag = vec![80.0; 100];
        let config = SmoothingConfig::default();
        let smoothed = variable_smoothing(&freq, &mag, &config);
        for s in &smoothed {
            assert!((s - 80.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fractional_octave_smooth_matches_prefix() {
        let freq: Vec<f64> = (0..200).map(|i| 20.0 * (20000.0 / 20.0_f64).powf(i as f64 / 199.0)).collect();
        let mag: Vec<f64> = freq.iter().map(|f| 80.0 + (f / 1000.0).sin() * 10.0).collect();
        let fraction = 1.0 / 6.0;

        // Compare old-style per-bin with prefix-sum-based variable_smoothing
        let config = SmoothingConfig { variable: false, fixed_fraction: Some(fraction) };
        let smoothed_prefix = variable_smoothing(&freq, &mag, &config);

        for i in 0..freq.len() {
            let old_val = fractional_octave_smooth(&freq, &mag, i, fraction);
            assert!((smoothed_prefix[i] - old_val).abs() < 1e-10,
                "Mismatch at bin {}: prefix={} old={}", i, smoothed_prefix[i], old_val);
        }
    }
}
