use num_complex::Complex64;
use super::fft::FftEngine;

/// Compute minimum phase from magnitude spectrum via Hilbert transform.
///
/// Input: `mag_db` — magnitude in dB for positive frequency bins (DC to Nyquist).
/// `n_fft` — FFT size (must be even, typically next power of 2 >= 2 * mag_db.len()).
///
/// Returns: minimum phase in **radians** for positive frequency bins (same length as input or n_bins).
pub fn minimum_phase_from_magnitude(mag_db: &[f64], n_fft: usize) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let ln10_over_20 = 10.0_f64.ln() / 20.0;

    // Build ln_magnitude as a real signal of length n_fft
    let mut ln_mag_signal: Vec<Complex64> = Vec::with_capacity(n_fft);

    for i in 0..n_bins {
        let ln_val = mag_db[i.min(mag_db.len() - 1)] * ln10_over_20;
        ln_mag_signal.push(Complex64::new(ln_val, 0.0));
    }
    // Mirror for negative frequencies
    for i in 1..(n_fft - n_bins + 1) {
        let idx = n_bins - 1 - i;
        ln_mag_signal.push(Complex64::new(ln_mag_signal[idx].re, 0.0));
    }

    // FFT
    let mut engine = FftEngine::new();
    engine.fft_forward(&mut ln_mag_signal);

    // Apply Hilbert window
    ln_mag_signal[0] *= Complex64::new(1.0, 0.0);
    for i in 1..n_fft / 2 {
        ln_mag_signal[i] *= Complex64::new(2.0, 0.0);
    }
    if n_fft > 1 {
        ln_mag_signal[n_fft / 2] *= Complex64::new(1.0, 0.0);
    }
    for i in (n_fft / 2 + 1)..n_fft {
        ln_mag_signal[i] = Complex64::new(0.0, 0.0);
    }

    // IFFT
    engine.fft_inverse(&mut ln_mag_signal);
    let norm = 1.0 / n_fft as f64;

    // Extract imaginary part = minimum phase (radians)
    (0..n_bins)
        .map(|i| -ln_mag_signal[i].im * norm)
        .collect()
}

/// Minimum phase (degrees) of a magnitude curve (dB) given on a log grid,
/// evaluated at the true Nyquist of `sample_rate` (see the Tauri command
/// `compute_minimum_phase` for the history of this function).
pub fn minimum_phase_on_log_grid(
    freq: &[f64],
    magnitude: &[f64],
    sample_rate: Option<f64>,
) -> Result<Vec<f64>, String> {
    let n = freq.len();
    if n < 2 { return Err("compute_minimum_phase: need at least 2 points".into()); }
    if magnitude.len() != n {
        return Err(format!("compute_minimum_phase: freq/magnitude length mismatch ({n} vs {})", magnitude.len()));
    }
    let f_last = *freq.last().expect("n >= 2");
    let nyquist = match sample_rate {
        Some(sr) if sr > 0.0 => sr / 2.0,
        _ => f_last,
    };
    let n_fft = ((n * 4).max(131_072)).next_power_of_two();
    let n_bins = n_fft / 2 + 1;

    // Clamp magnitude to a dynamic range the Hilbert kernel handles cleanly.
    let mag_peak = magnitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mag_floor = mag_peak - 120.0;
    let clamped: Vec<f64> = magnitude.iter().map(|&v| v.max(mag_floor)).collect();

    // Log-log slope at each grid edge (dB per octave), used to continue the
    // curve beyond the grid instead of holding it flat.
    let edge_slope = |i0: usize, i1: usize| -> f64 {
        let df = (freq[i1] / freq[i0]).log2();
        if df.abs() > 1e-9 { (clamped[i1] - clamped[i0]) / df } else { 0.0 }
    };
    let slope_lo = edge_slope(0, 1);
    let slope_hi = edge_slope(n - 2, n - 1);
    let f_first = freq[0].max(1e-3);

    // Resample magnitude from the log grid onto the linear FFT grid.
    let mut lin_mag = vec![clamped[0]; n_bins];
    for k in 0..n_bins {
        let f_lin = nyquist * k as f64 / (n_bins - 1) as f64;
        lin_mag[k] = if f_lin <= f_first {
            let oct = (f_lin.max(f_first / 4096.0) / f_first).log2(); // ≤ 0
            (clamped[0] + slope_lo * oct).max(mag_floor)
        } else if f_lin >= f_last {
            let oct = (f_lin / f_last).log2(); // ≥ 0
            (clamped[n - 1] + slope_hi * oct).max(mag_floor)
        } else {
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if freq[mid] <= f_lin { lo = mid; } else { hi = mid; }
            }
            let dt = freq[hi] - freq[lo];
            let frac = if dt > 0.0 { (f_lin - freq[lo]) / dt } else { 0.0 };
            clamped[lo] + frac * (clamped[hi] - clamped[lo])
        };
    }

    let min_ph_rad = minimum_phase_from_magnitude(&lin_mag, n_fft);

    // Resample phase from the linear grid back onto the log grid, in degrees.
    let mut phase_deg = Vec::with_capacity(n);
    for i in 0..n {
        let bin_f = (freq[i] / nyquist * (n_bins - 1) as f64).max(0.0);
        let lo = (bin_f as usize).min(n_bins - 2);
        let hi = lo + 1;
        let frac = (bin_f - lo as f64).clamp(0.0, 1.0);
        let ph_rad = min_ph_rad[lo] * (1.0 - frac) + min_ph_rad[hi] * frac;
        phase_deg.push(ph_rad.to_degrees());
    }
    Ok(phase_deg)
}

#[cfg(test)]
mod tests {
    /// Butterworth-2 HP fc=80 Hz has a closed-form min-phase response:
    /// φ = 180° − atan2(√2·w, 1 − w²), w = f/fc. Before the 2026-09-05 fix the
    /// grid-dependent implementation was 20–90° off at 20–40 Hz.
    #[test]
    fn min_phase_matches_analytic_bw2_hp_on_both_grids() {
        let fc = 80.0;
        let analytic = |f: f64| {
            let w = f / fc;
            180.0 - (2f64.sqrt() * w).atan2(1.0 - w * w).to_degrees()
        };
        let mag = |f: f64| {
            let w = f / fc;
            20.0 * (w * w / ((1.0 - w * w).powi(2) + 2.0 * w * w).sqrt()).log10()
        };
        for (f_min, f_max) in [(20.0_f64, 20000.0_f64), (5.0, 40000.0)] {
            let n = 512;
            let freq: Vec<f64> = (0..n)
                .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
                .collect();
            let mags: Vec<f64> = freq.iter().map(|&f| mag(f)).collect();
            let ph = super::minimum_phase_on_log_grid(&freq, &mags, Some(48000.0)).unwrap();
            for probe in [20.0, 40.0, 80.0, 160.0, 1000.0] {
                if probe < f_min { continue; }
                let i = freq.iter().position(|&f| f >= probe).unwrap();
                let want = analytic(freq[i]);
                let got = ph[i].rem_euclid(360.0);
                let want_m = want.rem_euclid(360.0);
                let d = ((got - want_m + 180.0).rem_euclid(360.0) - 180.0).abs();
                assert!(d < 3.0, "grid {f_min}-{f_max} @ {:.1} Hz: got {got:.2}°, want {want_m:.2}° (Δ {d:.2}°)", freq[i]);
            }
        }
    }
    use super::*;

    /// Sign anchor (DSP audit b141.2). A 1st-order lowpass is minimum-phase
    /// with analytical phase −atan(f/fc) — a LAG (negative). This pins the
    /// FFT forward-transform sign convention: the reconstructed min phase must
    /// come out NEGATIVE, matching −atan(r). A backend whose forward FFT uses
    /// the conjugate convention (e.g. a rustfft fallback diverging from vDSP)
    /// would flip the sign and fail here, instead of silently shipping a phase
    /// lead where a lag is required.
    #[test]
    fn min_phase_first_order_lp_is_a_lag() {
        let n_fft = 4096usize;
        let n_bins = n_fft / 2 + 1;
        let fc_bin = 64.0;
        // |H|² = 1/(1+r²) → mag_db = -10·log10(1+r²), r = bin/fc_bin.
        let mag_db: Vec<f64> = (0..n_bins)
            .map(|i| {
                let r = i as f64 / fc_bin;
                -10.0 * (1.0 + r * r).log10()
            })
            .collect();

        let phase = minimum_phase_from_magnitude(&mag_db, n_fft);

        // Sign anchor: every interior bin of a lowpass must be a lag (< 0).
        // A conjugate-convention forward FFT would flip all of these positive.
        for &i in &[64usize, 128, 256, 512, 1024] {
            assert!(phase[i] < 0.0, "bin {i}: phase {:.4} rad is not a lag", phase[i]);
        }
        // Magnitude match to −atan(r) on the well-reconstructed mid-band
        // (Hilbert discretisation error grows on the steep r≫1 tail).
        for &i in &[64usize, 128, 256] {
            let r = i as f64 / fc_bin;
            let expected = -(r.atan());
            assert!(
                (phase[i] - expected).abs() < 0.15,
                "bin {i}: phase {:.4} rad vs expected {:.4} rad", phase[i], expected,
            );
        }
        // DC phase ≈ 0.
        assert!(phase[0].abs() < 0.05, "DC phase {} should be ~0", phase[0]);
    }
}
