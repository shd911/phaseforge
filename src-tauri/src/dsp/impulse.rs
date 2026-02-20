use num_complex::Complex64;
use rustfft::FftPlanner;

use super::interpolation::interpolate_linear_grid;

/// Result of impulse response computation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImpulseResult {
    /// Time axis in seconds (can include negative values for pre-peak region)
    pub time: Vec<f64>,
    /// Impulse response amplitude (normalized: peak = 100%)
    pub impulse: Vec<f64>,
    /// Step response (cumulative sum of impulse, normalized)
    pub step: Vec<f64>,
}

/// Compute impulse and step response from frequency-domain measurement.
///
/// 1. Interpolate measurement onto linear frequency grid (0..Nyquist)
/// 2. Build complex spectrum H[k] = 10^(mag/20) * e^(j*phase_rad)
/// 3. Mirror for conjugate symmetry
/// 4. IFFT → time-domain impulse response
/// 5. Circular reorder: include pre-peak samples (negative time) from end of buffer
/// 6. Normalize peak to 100%, trim to 0.5% decay threshold
/// 7. Cumulative sum → step response
pub fn compute_impulse_response(
    freq: &[f64],
    magnitude: &[f64],
    phase: &[f64],
    sample_rate: f64,
) -> ImpulseResult {
    // Choose FFT size: next power of 2, at least 4096
    let fft_size = {
        let min_size = 4096usize;
        let desired = (freq.len() * 4).max(min_size);
        desired.next_power_of_two()
    };
    let n_bins = fft_size / 2 + 1; // positive freq bins (DC to Nyquist)

    // Interpolate measurement onto linear grid: 0 Hz to Nyquist
    let (_grid_freq, grid_mag, grid_phase_opt) =
        interpolate_linear_grid(freq, magnitude, Some(phase), n_bins, sample_rate);
    let grid_phase = grid_phase_opt.unwrap();

    // Build complex spectrum for positive frequencies
    let mut spectrum: Vec<Complex64> = Vec::with_capacity(fft_size);
    for i in 0..n_bins {
        let amp = 10.0_f64.powf(grid_mag[i] / 20.0);
        let ph_rad = grid_phase[i].to_radians();
        spectrum.push(Complex64::new(amp * ph_rad.cos(), amp * ph_rad.sin()));
    }

    // Mirror for negative frequencies (conjugate symmetry): bins n_bins..fft_size
    for i in 1..(fft_size - n_bins + 1) {
        let idx = n_bins - 1 - i;
        spectrum.push(spectrum[idx].conj());
    }

    // IFFT
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    ifft.process(&mut spectrum);

    // Normalize IFFT output (rustfft does not normalize)
    let norm = 1.0 / fft_size as f64;

    // Extract real part as impulse response
    let impulse_raw: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // Time step
    let dt = 1.0 / sample_rate;

    // Find peak (absolute maximum)
    let peak = impulse_raw.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    if peak <= 0.0 {
        return ImpulseResult {
            time: vec![0.0],
            impulse: vec![0.0],
            step: vec![0.0],
        };
    }

    // Find peak index in the raw IFFT buffer
    let peak_idx = impulse_raw
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // --- Determine pre-peak region (negative time) ---
    // The IFFT output is circular. Samples near the end of the buffer
    // represent the "pre-impulse" region (negative time).
    // We include these samples to show what happens before the peak.
    let threshold_raw = peak * 0.002; // 0.2% of peak (-54 dB) — catch subtle pre-ringing
    let max_pre = fft_size / 4; // max 25% of buffer as pre-peak
    let mut pre_peak_count = 0usize;

    for i in 0..max_pre {
        let idx = fft_size - 1 - i;
        if idx <= peak_idx + 1 {
            break; // don't overlap with forward data
        }
        if impulse_raw[idx].abs() > threshold_raw {
            pre_peak_count = i + 1;
        }
    }
    // Always include at least 5ms of pre-peak context (linear-phase filters need this)
    let min_pre = ((sample_rate * 0.005) as usize).min(max_pre);
    pre_peak_count = pre_peak_count.max(min_pre);

    // --- Determine post-peak trim point ---
    // Search in the first half of the buffer (forward direction from peak)
    let half = fft_size / 2;
    let impulse_norm_full: Vec<f64> = impulse_raw.iter().map(|v| (v / peak) * 100.0).collect();
    let threshold_norm = 0.5; // 0.5% of peak

    let mut trim_end = half; // default: up to half buffer
    for i in (peak_idx..half).rev() {
        if impulse_norm_full[i].abs() > threshold_norm {
            let padding = ((half - i) / 10).max(100).min(half - i);
            trim_end = (i + padding).min(half);
            break;
        }
    }
    let min_samples = (sample_rate * 0.001) as usize;
    trim_end = trim_end.max(min_samples);

    // --- Build output arrays ---
    // Layout: [pre-peak from end of buffer] + [0..trim_end from start of buffer]
    let total_len = pre_peak_count + trim_end;
    let pre_start = fft_size - pre_peak_count;

    let mut time = Vec::with_capacity(total_len);
    let mut impulse_out = Vec::with_capacity(total_len);
    let mut raw_reordered = Vec::with_capacity(total_len);

    // Pre-peak samples: buffer indices [pre_start..fft_size]
    // Their "true" time = (index - fft_size) * dt (negative values)
    for i in 0..pre_peak_count {
        let buf_idx = pre_start + i;
        let t = (buf_idx as f64 - fft_size as f64) * dt;
        time.push(t);
        impulse_out.push(impulse_norm_full[buf_idx]);
        raw_reordered.push(impulse_raw[buf_idx]);
    }

    // Forward samples: buffer indices [0..trim_end]
    // Their time = index * dt
    for i in 0..trim_end {
        let t = i as f64 * dt;
        time.push(t);
        impulse_out.push(impulse_norm_full[i]);
        raw_reordered.push(impulse_raw[i]);
    }

    // --- Step response: cumulative sum of reordered raw impulse ---
    let mut step = Vec::with_capacity(total_len);
    let mut acc = 0.0;
    for v in &raw_reordered {
        acc += v;
        step.push(acc);
    }

    // Normalize step to peak = 100%
    let step_peak = step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let step_norm: Vec<f64> = if step_peak > 0.0 {
        step.iter().map(|v| (v / step_peak) * 100.0).collect()
    } else {
        step
    };

    ImpulseResult {
        time,
        impulse: impulse_out,
        step: step_norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impulse_flat_spectrum() {
        // Flat magnitude, zero phase → should produce a delta-like impulse
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![0.0; n]; // 0 dB = unity
        let phase: Vec<f64> = vec![0.0; n]; // zero phase

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        assert_eq!(result.impulse.len(), result.time.len());
        assert_eq!(result.step.len(), result.time.len());

        // Find peak in the output
        let peak_idx = result
            .impulse
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Peak value should be 100 (percent)
        let peak_val = result.impulse[peak_idx];
        assert!((peak_val - 100.0).abs() < 1.0, "Peak should be ~100%, got {}", peak_val);

        // Time at peak should be near 0
        let peak_time = result.time[peak_idx];
        assert!(peak_time.abs() < 0.001, "Peak time should be near 0, got {}", peak_time);

        // Should have some negative time values (pre-peak region)
        assert!(result.time[0] < 0.0, "First time should be negative, got {}", result.time[0]);
    }

    #[test]
    fn test_impulse_trimmed() {
        // Verify trimming: result should be shorter than full FFT size
        let n = 50;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 400.0).collect();
        let mag: Vec<f64> = vec![0.0; n];
        let phase: Vec<f64> = vec![0.0; n];

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        // Trimmed length should be less than full FFT size (4096 at minimum)
        assert!(result.time.len() < 4096, "Should be trimmed, got len={}", result.time.len());
        assert_eq!(result.time.len(), result.impulse.len());
        assert_eq!(result.time.len(), result.step.len());
        // Time axis should start with negative value (pre-peak)
        assert!(result.time[0] < 0.0, "First time should be negative");
    }

    #[test]
    fn test_impulse_time_monotonic() {
        // Verify time axis is monotonically increasing
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![0.0; n];
        let phase: Vec<f64> = vec![0.0; n];

        let result = compute_impulse_response(&freq, &mag, &phase, 48000.0);

        for i in 1..result.time.len() {
            assert!(
                result.time[i] > result.time[i - 1],
                "Time should be monotonically increasing at index {}: {} <= {}",
                i, result.time[i], result.time[i - 1]
            );
        }
    }
}
