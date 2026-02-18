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
            fractional_octave_smooth(freq, mag, i, fraction)
        })
        .collect()
}

/// Smooth a single frequency bin using fractional-octave window.
pub fn fractional_octave_smooth(freq: &[f64], mag: &[f64], idx: usize, fraction: f64) -> f64 {
    let center = freq[idx];
    if center <= 0.0 {
        return mag[idx];
    }

    let k = 2.0_f64.powf(fraction / 2.0);
    let f_low = center / k;
    let f_high = center * k;

    let mut sum = 0.0;
    let mut count = 0.0;

    for (i, &f) in freq.iter().enumerate() {
        if f >= f_low && f <= f_high {
            sum += mag[i];
            count += 1.0;
        }
    }

    if count > 0.0 {
        sum / count
    } else {
        mag[idx]
    }
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
}
