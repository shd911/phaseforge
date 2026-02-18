use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::error::AppError;

const SPEED_OF_SOUND: f64 = 343.0;
/// Each of the 4 edges contributes a +1.5 dB shelf (total = 6 dB).
const GAIN_PER_EDGE_DB: f64 = 1.5;

/// Baffle dimensions and driver position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaffleConfig {
    /// Baffle width in meters (horizontal)
    pub baffle_width_m: f64,
    /// Baffle height in meters (vertical)
    pub baffle_height_m: f64,
    /// Distance from left edge to driver center (m)
    pub driver_offset_x_m: f64,
    /// Distance from top edge to driver center (m)
    pub driver_offset_y_m: f64,
}

/// Result of baffle step computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaffleStepResult {
    /// Correction to apply to NF magnitude (dB). Negative at LF, ~0 at HF.
    pub correction_mag_db: Vec<f64>,
    /// Minimum-phase correction (radians).
    pub correction_phase_rad: Vec<f64>,
    /// Effective transition frequency (geometric mean of 4 edges).
    pub f3_hz: f64,
    /// Per-edge transition frequencies [left, right, top, bottom].
    pub edge_frequencies: [f64; 4],
}

/// Preview data for the UI (fixed 256-point grid).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaffleStepPreview {
    pub freq: Vec<f64>,
    pub correction_db: Vec<f64>,
    pub f3_hz: f64,
    pub edge_frequencies: [f64; 4],
}

const MIN_EDGE_DISTANCE: f64 = 0.001; // 1 mm minimum

fn validate_config(config: &BaffleConfig) -> Result<(), AppError> {
    if config.baffle_width_m <= 0.0 || config.baffle_height_m <= 0.0 {
        return Err(AppError::Config {
            message: "Baffle dimensions must be positive".into(),
        });
    }
    if config.driver_offset_x_m < MIN_EDGE_DISTANCE
        || config.driver_offset_x_m > config.baffle_width_m - MIN_EDGE_DISTANCE
    {
        return Err(AppError::Config {
            message: format!(
                "Driver X offset must be between {:.3} and {:.3} m",
                MIN_EDGE_DISTANCE,
                config.baffle_width_m - MIN_EDGE_DISTANCE
            ),
        });
    }
    if config.driver_offset_y_m < MIN_EDGE_DISTANCE
        || config.driver_offset_y_m > config.baffle_height_m - MIN_EDGE_DISTANCE
    {
        return Err(AppError::Config {
            message: format!(
                "Driver Y offset must be between {:.3} and {:.3} m",
                MIN_EDGE_DISTANCE,
                config.baffle_height_m - MIN_EDGE_DISTANCE
            ),
        });
    }
    Ok(())
}

/// Compute edge distances: [left, right, top, bottom]
fn edge_distances(config: &BaffleConfig) -> [f64; 4] {
    [
        config.driver_offset_x_m,                        // left
        config.baffle_width_m - config.driver_offset_x_m, // right
        config.driver_offset_y_m,                         // top
        config.baffle_height_m - config.driver_offset_y_m, // bottom
    ]
}

/// Transition frequency for a single edge at distance d.
fn edge_freq(d: f64) -> f64 {
    SPEED_OF_SOUND / (2.0 * PI * d)
}

/// Compute baffle step correction for a frequency array.
///
/// The correction brings NF level down at LF (where the real speaker loses
/// energy to full-space radiation) and leaves HF untouched.
///
/// Model: 4 independent first-order high-shelf filters, +1.5 dB each.
pub fn compute_baffle_step(
    freq: &[f64],
    config: &BaffleConfig,
) -> Result<BaffleStepResult, AppError> {
    validate_config(config)?;

    if freq.is_empty() {
        return Err(AppError::Dsp {
            message: "Frequency array must not be empty".into(),
        });
    }

    let distances = edge_distances(config);
    let edge_freqs: [f64; 4] = [
        edge_freq(distances[0]),
        edge_freq(distances[1]),
        edge_freq(distances[2]),
        edge_freq(distances[3]),
    ];

    // Effective f3 = geometric mean of 4 edge frequencies
    let f3 = (edge_freqs[0] * edge_freqs[1] * edge_freqs[2] * edge_freqs[3]).powf(0.25);

    // Amplitude parameter for single-edge shelf: A = 10^(G/40)
    let a = 10.0_f64.powf(GAIN_PER_EDGE_DB / 40.0);
    let a2 = a * a;
    let a_inv = 1.0 / a;
    let total_gain = 4.0 * GAIN_PER_EDGE_DB; // 6.0 dB

    let n = freq.len();
    let mut correction_mag = Vec::with_capacity(n);
    let mut correction_phase = Vec::with_capacity(n);

    for &f in freq {
        let mut mag_sum = 0.0;
        let mut phase_sum = 0.0;

        for &fc in &edge_freqs {
            let ratio = f / fc;

            // Magnitude of one edge shelf (dB): 10*log10(|H|^2)
            let num = 1.0 + a2 * ratio * ratio;
            let den = 1.0 + (ratio * a_inv) * (ratio * a_inv);
            mag_sum += 10.0 * (num / den).log10();

            // Phase of one edge shelf (radians)
            phase_sum += (a * ratio).atan() - (ratio * a_inv).atan();
        }

        // Correction = baffle_step - total_gain
        // baffle_step goes 0 → +6 dB; correction goes -6 → 0 dB
        correction_mag.push(mag_sum - total_gain);
        // Inverse phase for correction
        correction_phase.push(-phase_sum);
    }

    Ok(BaffleStepResult {
        correction_mag_db: correction_mag,
        correction_phase_rad: correction_phase,
        f3_hz: f3,
        edge_frequencies: edge_freqs,
    })
}

/// Apply baffle step correction to magnitude and phase arrays in-place.
/// Phase is in degrees.
pub fn apply_baffle_correction(
    magnitude: &mut [f64],
    phase_deg: &mut [f64],
    result: &BaffleStepResult,
) {
    for i in 0..magnitude.len().min(result.correction_mag_db.len()) {
        magnitude[i] += result.correction_mag_db[i];
        phase_deg[i] += result.correction_phase_rad[i].to_degrees();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> BaffleConfig {
        BaffleConfig {
            baffle_width_m: 0.25,
            baffle_height_m: 0.35,
            driver_offset_x_m: 0.125,
            driver_offset_y_m: 0.12,
        }
    }

    #[test]
    fn test_correction_at_dc() {
        let freq = vec![0.1]; // very low frequency
        let result = compute_baffle_step(&freq, &default_config()).unwrap();
        // At DC, correction should be close to -6 dB
        assert!(
            (result.correction_mag_db[0] - (-6.0)).abs() < 0.1,
            "DC correction should be ~-6 dB, got {:.2}",
            result.correction_mag_db[0]
        );
    }

    #[test]
    fn test_correction_at_hf() {
        let freq = vec![100000.0]; // very high frequency
        let result = compute_baffle_step(&freq, &default_config()).unwrap();
        // At HF, correction should be close to 0 dB
        assert!(
            result.correction_mag_db[0].abs() < 0.1,
            "HF correction should be ~0 dB, got {:.2}",
            result.correction_mag_db[0]
        );
    }

    #[test]
    fn test_correction_monotonic() {
        // Correction should increase from -6 dB towards 0 dB as frequency rises
        let freq: Vec<f64> = (0..100)
            .map(|i| 10.0 * (20000.0_f64 / 10.0).powf(i as f64 / 99.0))
            .collect();
        let result = compute_baffle_step(&freq, &default_config()).unwrap();
        for i in 1..result.correction_mag_db.len() {
            assert!(
                result.correction_mag_db[i] >= result.correction_mag_db[i - 1] - 0.01,
                "Correction should be monotonically increasing, but [{i}]={:.3} < [{prev}]={:.3}",
                result.correction_mag_db[i],
                result.correction_mag_db[i - 1],
                prev = i - 1
            );
        }
    }

    #[test]
    fn test_centered_driver_symmetric_edges() {
        let config = BaffleConfig {
            baffle_width_m: 0.30,
            baffle_height_m: 0.30,
            driver_offset_x_m: 0.15,
            driver_offset_y_m: 0.15,
        };
        let freq = vec![500.0];
        let result = compute_baffle_step(&freq, &config).unwrap();
        // All 4 edge frequencies should be equal for a centered square baffle
        let ef = result.edge_frequencies;
        assert!(
            (ef[0] - ef[1]).abs() < 0.01,
            "Left and right should match: {} vs {}",
            ef[0],
            ef[1]
        );
        assert!(
            (ef[2] - ef[3]).abs() < 0.01,
            "Top and bottom should match: {} vs {}",
            ef[2],
            ef[3]
        );
        assert!(
            (ef[0] - ef[2]).abs() < 0.01,
            "All edges should match for square centered: {} vs {}",
            ef[0],
            ef[2]
        );
    }

    #[test]
    fn test_f3_reasonable() {
        let config = default_config();
        let freq = vec![1000.0];
        let result = compute_baffle_step(&freq, &config).unwrap();
        // For a ~0.25m baffle, f3 should be roughly 115/0.25 ≈ 460 Hz (Murphy)
        // Our geometric mean model may differ, but should be in 200-800 Hz range
        assert!(
            result.f3_hz > 100.0 && result.f3_hz < 1500.0,
            "f3 should be reasonable, got {:.0} Hz",
            result.f3_hz
        );
    }

    #[test]
    fn test_validation_offset_out_of_range() {
        let config = BaffleConfig {
            baffle_width_m: 0.20,
            baffle_height_m: 0.30,
            driver_offset_x_m: 0.25, // > width
            driver_offset_y_m: 0.15,
        };
        assert!(compute_baffle_step(&[100.0], &config).is_err());
    }

    #[test]
    fn test_validation_negative_dimensions() {
        let config = BaffleConfig {
            baffle_width_m: -0.20,
            baffle_height_m: 0.30,
            driver_offset_x_m: 0.10,
            driver_offset_y_m: 0.15,
        };
        assert!(compute_baffle_step(&[100.0], &config).is_err());
    }

    #[test]
    fn test_apply_correction() {
        let config = default_config();
        let freq = vec![20.0, 1000.0, 20000.0];
        let result = compute_baffle_step(&freq, &config).unwrap();

        let mut mag = vec![80.0, 80.0, 80.0];
        let mut phase = vec![0.0, 0.0, 0.0];
        apply_baffle_correction(&mut mag, &mut phase, &result);

        // LF should be reduced (< 80), HF should stay near 80
        assert!(mag[0] < 78.0, "LF should be reduced, got {:.1}", mag[0]);
        assert!(mag[2] > 79.0, "HF should stay near 80, got {:.1}", mag[2]);
    }
}
