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

#[cfg(test)]
mod tests {
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
