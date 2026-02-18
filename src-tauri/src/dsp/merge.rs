use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::io::{Measurement, MeasurementMetadata};
use crate::phase;

use super::baffle::BaffleConfig;
use super::interpolate_log_grid;

/// Configuration for NF+FF measurement merge (splice).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Splice center frequency in Hz (e.g. 200–500)
    pub splice_freq: f64,
    /// Blend zone width in octaves (e.g. 0.5, 1.0, 2.0, 3.0)
    pub blend_octaves: f64,
    /// Manual level offset override in dB. None = auto-compute from overlap region.
    pub level_offset_db: Option<f64>,
    /// Frequency range [lo, hi] Hz for auto level matching. None = [200, 600].
    pub match_range: Option<[f64; 2]>,
    /// Baffle step correction config. None = no correction.
    #[serde(default)]
    pub baffle: Option<BaffleConfig>,
}

/// Result of NF+FF merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// The merged measurement
    pub measurement: Measurement,
    /// Auto-computed level offset (dB) applied to NF
    pub auto_level_offset_db: f64,
    /// Auto-computed delay difference (seconds) between NF and FF
    pub delay_diff_seconds: f64,
}

/// Merge near-field and far-field measurements using frequency-domain splice
/// with cosine crossfade in complex domain.
///
/// Algorithm:
/// 1. Interpolate both onto common log-spaced grid (intersection of freq ranges)
/// 2. Level match: compute avg magnitude difference in overlap region, shift NF
/// 3. Delay align: compute group delay difference, correct NF phase
/// 4. Cosine crossfade in complex domain across blend zone
pub fn merge_nf_ff(
    nf: &Measurement,
    ff: &Measurement,
    config: &MergeConfig,
) -> Result<MergeResult, AppError> {
    // Validate: both must have phase data
    let nf_phase = nf.phase.as_ref().ok_or_else(|| AppError::Dsp {
        message: "Near-field measurement must have phase data".into(),
    })?;
    let ff_phase = ff.phase.as_ref().ok_or_else(|| AppError::Dsp {
        message: "Far-field measurement must have phase data".into(),
    })?;

    // Validate: non-empty
    if nf.freq.is_empty() || ff.freq.is_empty() {
        return Err(AppError::Dsp {
            message: "Measurements must not be empty".into(),
        });
    }

    // Common frequency range (intersection)
    let f_min = nf.freq[0].max(ff.freq[0]).max(1.0); // at least 1 Hz
    let f_max = nf.freq.last().unwrap().min(*ff.freq.last().unwrap());
    if f_min >= f_max {
        return Err(AppError::Dsp {
            message: format!(
                "No frequency overlap between NF ({:.0}–{:.0} Hz) and FF ({:.0}–{:.0} Hz)",
                nf.freq[0],
                nf.freq.last().unwrap(),
                ff.freq[0],
                ff.freq.last().unwrap()
            ),
        });
    }

    let n_points = 2048;

    // Interpolate both onto shared log grid
    let (grid_freq, nf_mag, nf_ph_opt) =
        interpolate_log_grid(&nf.freq, &nf.magnitude, Some(nf_phase), n_points, f_min, f_max);
    let nf_ph = nf_ph_opt.unwrap();

    let (_, ff_mag, ff_ph_opt) =
        interpolate_log_grid(&ff.freq, &ff.magnitude, Some(ff_phase), n_points, f_min, f_max);
    let ff_ph = ff_ph_opt.unwrap();

    // --- Baffle step correction (applied to NF before level matching) ---
    let (nf_mag, nf_ph) = if let Some(ref baffle_config) = config.baffle {
        let baffle_result = super::baffle::compute_baffle_step(&grid_freq, baffle_config)?;
        let mut corrected_mag = nf_mag;
        let mut corrected_ph = nf_ph;
        super::baffle::apply_baffle_correction(
            &mut corrected_mag,
            &mut corrected_ph,
            &baffle_result,
        );
        (corrected_mag, corrected_ph)
    } else {
        (nf_mag, nf_ph)
    };

    // --- Level matching ---
    let (match_lo, match_hi) = match config.match_range {
        Some([lo, hi]) => (lo, hi),
        None => (200.0, 600.0),
    };

    let mut diff_sum = 0.0;
    let mut diff_count = 0usize;
    for i in 0..n_points {
        if grid_freq[i] >= match_lo && grid_freq[i] <= match_hi {
            diff_sum += ff_mag[i] - nf_mag[i];
            diff_count += 1;
        }
    }
    let auto_offset = if diff_count > 0 {
        diff_sum / diff_count as f64
    } else {
        0.0
    };
    let level_offset = config.level_offset_db.unwrap_or(auto_offset);

    // Apply level shift to NF
    let nf_mag_shifted: Vec<f64> = nf_mag.iter().map(|&v| v + level_offset).collect();

    // --- Delay alignment ---
    let nf_delay = phase::compute_average_delay(&grid_freq, &nf_ph, match_lo, match_hi);
    let ff_delay = phase::compute_average_delay(&grid_freq, &ff_ph, match_lo, match_hi);
    let delay_diff = nf_delay - ff_delay;

    // Remove delay difference from NF phase
    let nf_ph_aligned = phase::remove_delay(&grid_freq, &nf_ph, delay_diff);

    // --- Blend zone boundaries ---
    let splice_f = config.splice_freq;
    let half_blend = config.blend_octaves / 2.0;
    let f_low = splice_f / 2.0_f64.powf(half_blend);
    let f_high = splice_f * 2.0_f64.powf(half_blend);

    // --- Crossfade in complex domain ---
    let mut merged_mag = Vec::with_capacity(n_points);
    let mut merged_phase = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let f = grid_freq[i];

        if f <= f_low {
            // 100% NF
            merged_mag.push(nf_mag_shifted[i]);
            merged_phase.push(nf_ph_aligned[i]);
        } else if f >= f_high {
            // 100% FF
            merged_mag.push(ff_mag[i]);
            merged_phase.push(ff_ph[i]);
        } else {
            // Blend zone: cosine crossfade in log-frequency domain
            let t = (f.ln() - f_low.ln()) / (f_high.ln() - f_low.ln());
            let weight_ff = 0.5 * (1.0 - (std::f64::consts::PI * t).cos());
            let weight_nf = 1.0 - weight_ff;

            // NF complex
            let nf_amp = 10.0_f64.powf(nf_mag_shifted[i] / 20.0);
            let nf_rad = nf_ph_aligned[i].to_radians();
            let nf_re = nf_amp * nf_rad.cos();
            let nf_im = nf_amp * nf_rad.sin();

            // FF complex
            let ff_amp = 10.0_f64.powf(ff_mag[i] / 20.0);
            let ff_rad = ff_ph[i].to_radians();
            let ff_re = ff_amp * ff_rad.cos();
            let ff_im = ff_amp * ff_rad.sin();

            // Weighted sum
            let re = weight_nf * nf_re + weight_ff * ff_re;
            let im = weight_nf * nf_im + weight_ff * ff_im;

            let amplitude = (re * re + im * im).sqrt();
            let mag_db = if amplitude > 0.0 {
                20.0 * amplitude.log10()
            } else {
                -200.0
            };
            let phase_deg = im.atan2(re).to_degrees();

            merged_mag.push(mag_db);
            merged_phase.push(phase_deg);
        }
    }

    // --- Build output Measurement ---
    let merged = Measurement {
        name: format!("{} + {} (merged)", nf.name, ff.name),
        source_path: None,
        sample_rate: ff.sample_rate.or(nf.sample_rate),
        freq: grid_freq,
        magnitude: merged_mag,
        phase: Some(merged_phase),
        metadata: MeasurementMetadata {
            date: None,
            mic: None,
            notes: Some(format!(
                "NF+FF merge: splice={:.0}Hz, blend={:.1}oct, offset={:.1}dB",
                splice_f, config.blend_octaves, level_offset
            )),
            smoothing: None,
        },
    };

    Ok(MergeResult {
        measurement: merged,
        auto_level_offset_db: auto_offset,
        delay_diff_seconds: delay_diff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a flat measurement at given dB level
    fn make_flat_measurement(
        name: &str,
        level_db: f64,
        phase_deg: f64,
        f_min: f64,
        f_max: f64,
        n: usize,
    ) -> Measurement {
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (f_min.ln() + t * (f_max.ln() - f_min.ln())).exp()
            })
            .collect();
        let magnitude = vec![level_db; n];
        let phase = vec![phase_deg; n];

        Measurement {
            name: name.to_string(),
            source_path: None,
            sample_rate: Some(48000.0),
            freq,
            magnitude,
            phase: Some(phase),
            metadata: MeasurementMetadata {
                date: None,
                mic: None,
                notes: None,
                smoothing: None,
            },
        }
    }

    #[test]
    fn test_merge_flat_same_level() {
        // Two flat measurements at same level → merged should be flat
        let nf = make_flat_measurement("NF", 80.0, 0.0, 20.0, 20000.0, 200);
        let ff = make_flat_measurement("FF", 80.0, 0.0, 20.0, 20000.0, 200);

        let config = MergeConfig {
            splice_freq: 300.0,
            blend_octaves: 1.0,
            level_offset_db: None,
            match_range: None,
            baffle: None,
        };

        let result = merge_nf_ff(&nf, &ff, &config).unwrap();
        let m = &result.measurement;

        assert_eq!(m.freq.len(), 2048);
        assert_eq!(m.magnitude.len(), 2048);
        assert!(m.phase.is_some());

        // Auto offset should be ~0 (same level)
        assert!(
            result.auto_level_offset_db.abs() < 0.1,
            "Auto offset should be ~0, got {}",
            result.auto_level_offset_db
        );

        // All magnitudes should be ~80 dB
        for (i, &mag) in m.magnitude.iter().enumerate() {
            assert!(
                (mag - 80.0).abs() < 0.5,
                "Magnitude at idx {} ({:.0} Hz) should be ~80 dB, got {:.1}",
                i,
                m.freq[i],
                mag
            );
        }
    }

    #[test]
    fn test_merge_level_offset() {
        // NF at 70 dB, FF at 80 dB → auto offset should be ~+10 dB
        let nf = make_flat_measurement("NF", 70.0, 0.0, 20.0, 20000.0, 200);
        let ff = make_flat_measurement("FF", 80.0, 0.0, 20.0, 20000.0, 200);

        let config = MergeConfig {
            splice_freq: 300.0,
            blend_octaves: 1.0,
            level_offset_db: None,
            match_range: None,
            baffle: None,
        };

        let result = merge_nf_ff(&nf, &ff, &config).unwrap();

        // Auto offset should be ~+10 dB
        assert!(
            (result.auto_level_offset_db - 10.0).abs() < 0.5,
            "Auto offset should be ~10, got {}",
            result.auto_level_offset_db
        );

        // After level correction, merged should be ~80 dB everywhere
        for &mag in &result.measurement.magnitude {
            assert!(
                (mag - 80.0).abs() < 1.0,
                "Merged magnitude should be ~80 dB, got {:.1}",
                mag
            );
        }
    }

    #[test]
    fn test_merge_blend_weights_at_boundaries() {
        // At f_low: weight should be 0 (100% NF)
        // At f_high: weight should be 1 (100% FF)
        let splice = 300.0_f64;
        let blend = 1.0_f64;
        let half = blend / 2.0;
        let f_low = splice / 2.0_f64.powf(half);
        let f_high = splice * 2.0_f64.powf(half);

        // At f_low
        let t0 = (f_low.ln() - f_low.ln()) / (f_high.ln() - f_low.ln());
        let w0 = 0.5 * (1.0 - (std::f64::consts::PI * t0).cos());
        assert!(w0.abs() < 1e-10, "Weight at f_low should be 0, got {}", w0);

        // At f_high
        let t1 = (f_high.ln() - f_low.ln()) / (f_high.ln() - f_low.ln());
        let w1 = 0.5 * (1.0 - (std::f64::consts::PI * t1).cos());
        assert!((w1 - 1.0).abs() < 1e-10, "Weight at f_high should be 1, got {}", w1);

        // At splice_freq (center): should be 0.5
        let tc = (splice.ln() - f_low.ln()) / (f_high.ln() - f_low.ln());
        let wc = 0.5 * (1.0 - (std::f64::consts::PI * tc).cos());
        assert!((wc - 0.5).abs() < 0.01, "Weight at center should be 0.5, got {}", wc);
    }

    #[test]
    fn test_merge_error_no_phase() {
        let mut nf = make_flat_measurement("NF", 80.0, 0.0, 20.0, 20000.0, 100);
        nf.phase = None; // remove phase
        let ff = make_flat_measurement("FF", 80.0, 0.0, 20.0, 20000.0, 100);

        let config = MergeConfig {
            splice_freq: 300.0,
            blend_octaves: 1.0,
            level_offset_db: None,
            match_range: None,
            baffle: None,
        };

        let result = merge_nf_ff(&nf, &ff, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_error_no_overlap() {
        // NF: 20-200 Hz, FF: 500-20000 Hz → no overlap
        let nf = make_flat_measurement("NF", 80.0, 0.0, 20.0, 200.0, 100);
        let ff = make_flat_measurement("FF", 80.0, 0.0, 500.0, 20000.0, 100);

        let config = MergeConfig {
            splice_freq: 300.0,
            blend_octaves: 1.0,
            level_offset_db: None,
            match_range: None,
            baffle: None,
        };

        let result = merge_nf_ff(&nf, &ff, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_name() {
        let nf = make_flat_measurement("NearField", 80.0, 0.0, 20.0, 20000.0, 100);
        let ff = make_flat_measurement("FarField", 80.0, 0.0, 20.0, 20000.0, 100);

        let config = MergeConfig {
            splice_freq: 300.0,
            blend_octaves: 1.0,
            level_offset_db: None,
            match_range: None,
            baffle: None,
        };

        let result = merge_nf_ff(&nf, &ff, &config).unwrap();
        assert_eq!(result.measurement.name, "NearField + FarField (merged)");
    }
}
