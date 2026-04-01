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
