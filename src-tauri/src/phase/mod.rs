// Phase engine: group delay, distance computation, delay removal

/// Unwrap phase: remove discontinuities greater than 180 degrees.
///
/// Input phase is in degrees (wrapped ±180°). Output is continuous (unwrapped).
pub fn unwrap_phase(phase_deg: &[f64]) -> Vec<f64> {
    if phase_deg.is_empty() {
        return vec![];
    }
    let mut unwrapped = vec![phase_deg[0]];
    for i in 1..phase_deg.len() {
        let mut diff = phase_deg[i] - phase_deg[i - 1];
        while diff > 180.0 {
            diff -= 360.0;
        }
        while diff <= -180.0 {
            diff += 360.0;
        }
        unwrapped.push(unwrapped[i - 1] + diff);
    }
    unwrapped
}

/// Compute group delay at each frequency point.
///
/// Group delay τ(f) = -(1/360) · dφ/df  (seconds)
/// where φ is in degrees and f is in Hz.
///
/// Uses central finite differences for interior points,
/// forward/backward differences for endpoints.
pub fn compute_group_delay(freq: &[f64], phase_deg: &[f64]) -> Vec<f64> {
    let n = freq.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut gd = vec![0.0; n];

    // Forward difference for first point
    let df = freq[1] - freq[0];
    if df > 0.0 {
        gd[0] = -(phase_deg[1] - phase_deg[0]) / (360.0 * df);
    }

    // Central differences for interior points
    for i in 1..n - 1 {
        let df = freq[i + 1] - freq[i - 1];
        if df > 0.0 {
            gd[i] = -(phase_deg[i + 1] - phase_deg[i - 1]) / (360.0 * df);
        }
    }

    // Backward difference for last point
    let df = freq[n - 1] - freq[n - 2];
    if df > 0.0 {
        gd[n - 1] = -(phase_deg[n - 1] - phase_deg[n - 2]) / (360.0 * df);
    }

    gd
}

/// Estimate propagation delay via linear least-squares fit of unwrapped phase.
///
/// Fits φ(f) = slope × f + intercept over points in [f_low, f_high].
/// delay = -slope / 360 (seconds).
///
/// Linear fit is more robust than average group delay because it is not
/// biased by local group delay inflation from room modes or crossover resonances.
pub fn compute_average_delay(
    freq: &[f64],
    phase_deg: &[f64],
    f_low: f64,
    f_high: f64,
) -> f64 {
    let unwrapped = unwrap_phase(phase_deg);

    // Collect points in range
    let mut ff = Vec::new();
    let mut pp = Vec::new();
    for i in 0..freq.len() {
        if freq[i] >= f_low && freq[i] <= f_high {
            ff.push(freq[i]);
            pp.push(unwrapped[i]);
        }
    }
    // Fallback: use all points if none in range
    if ff.is_empty() {
        for i in 0..freq.len() {
            ff.push(freq[i]);
            pp.push(unwrapped[i]);
        }
    }
    if ff.len() < 2 {
        return 0.0;
    }

    // Linear least-squares: φ = slope × f + intercept
    // slope = (n Σ(f×φ) - Σf Σφ) / (n Σ(f²) - (Σf)²)
    let n = ff.len() as f64;
    let sum_f: f64 = ff.iter().sum();
    let sum_p: f64 = pp.iter().sum();
    let sum_ff: f64 = ff.iter().map(|f| f * f).sum();
    let sum_fp: f64 = ff.iter().zip(pp.iter()).map(|(f, p)| f * p).sum();
    let denom = n * sum_ff - sum_f * sum_f;
    if denom.abs() < 1e-30 {
        return 0.0;
    }
    let slope = (n * sum_fp - sum_f * sum_p) / denom;

    // delay = -slope / 360 (slope is dφ/df in deg/Hz)
    -slope / 360.0
}

/// Select delay estimation range based on measurement bandwidth.
///
/// For narrow-band data (sub: <1 decade), fit the lower half to avoid
/// port resonance. For mid-range, use central 50%. For wide-band, use
/// central 60%.
pub fn smart_delay_range(f_first: f64, f_last: f64) -> (f64, f64) {
    let log_lo = f_first.ln();
    let log_hi = f_last.ln();
    let decades = (log_hi - log_lo) / 10.0_f64.ln();

    if decades < 1.2 {
        // Narrow band (sub ~20-200 Hz): use lower 50% — below port resonance
        let f_lo = f_first;
        let f_hi = (log_lo + (log_hi - log_lo) * 0.5).exp();
        (f_lo, f_hi)
    } else if decades < 2.0 {
        // Mid band (~200-3000 Hz): central 50%
        let f_lo = (log_lo + (log_hi - log_lo) * 0.25).exp();
        let f_hi = (log_lo + (log_hi - log_lo) * 0.75).exp();
        (f_lo, f_hi)
    } else {
        // Wide band: central 60%
        let f_lo = (log_lo + (log_hi - log_lo) * 0.2).exp();
        let f_hi = (log_lo + (log_hi - log_lo) * 0.8).exp();
        (f_lo, f_hi)
    }
}

/// Check if delay removal resulted in overcorrection.
///
/// Returns average excess group delay (seconds) in the passband.
/// Positive value means delay was overestimated (phase slopes upward).
pub fn check_overcorrection(freq: &[f64], corrected_phase: &[f64]) -> f64 {
    if freq.len() < 4 { return 0.0; }
    let f_first = freq[0];
    let f_last = freq[freq.len() - 1];
    // Check group delay in central 50% of range
    let log_lo = f_first.ln();
    let log_hi = f_last.ln();
    let f_lo = (log_lo + (log_hi - log_lo) * 0.25).exp();
    let f_hi = (log_lo + (log_hi - log_lo) * 0.75).exp();

    let unwrapped = unwrap_phase(corrected_phase);
    let gd = compute_group_delay(freq, &unwrapped);

    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..freq.len() {
        if freq[i] >= f_lo && freq[i] <= f_hi {
            sum += gd[i];
            count += 1;
        }
    }
    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Estimate propagation delay via excess phase (measured − minimum phase).
///
/// 1. Interpolate measurement onto linear frequency grid
/// 2. Compute minimum phase from magnitude via Hilbert transform
/// 3. Excess phase = measured_phase − minimum_phase
/// 4. Fit linear slope to excess phase → delay
///
/// This is the most robust method: it separates filter/resonance phase
/// (captured by minimum phase) from propagation delay (excess phase).
pub fn compute_excess_phase_delay(
    freq: &[f64],
    magnitude: &[f64],
    phase_deg: &[f64],
) -> f64 {
    use num_complex::Complex64;
    use crate::dsp::fft::FftEngine;

    let n = freq.len();
    if n < 4 { return 0.0; }

    // FFT size for Hilbert transform
    let f_max = freq[n - 1];
    let fft_size = {
        let desired = (n * 4).max(4096);
        desired.next_power_of_two()
    };
    let n_bins = fft_size / 2 + 1;
    let df = f_max / (n_bins - 1) as f64;

    // Interpolate magnitude onto linear grid
    let mut grid_mag_db = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let f = i as f64 * df;
        grid_mag_db.push(interp_single_val(freq, magnitude, f));
    }

    // Compute minimum phase via Hilbert transform on ln(magnitude)
    let ln10_over_20 = 10.0_f64.ln() / 20.0;
    let mut ln_mag: Vec<Complex64> = Vec::with_capacity(fft_size);
    for i in 0..n_bins {
        ln_mag.push(Complex64::new(grid_mag_db[i] * ln10_over_20, 0.0));
    }
    for i in 1..(fft_size - n_bins + 1) {
        let idx = n_bins - 1 - i;
        ln_mag.push(Complex64::new(ln_mag[idx].re, 0.0));
    }

    let mut engine = FftEngine::new();
    engine.fft_forward(&mut ln_mag);

    // Hilbert window
    // DC stays ×1, positive freqs ×2, Nyquist ×1, negative freqs ×0
    for i in 1..fft_size / 2 {
        ln_mag[i] *= Complex64::new(2.0, 0.0);
    }
    for i in (fft_size / 2 + 1)..fft_size {
        ln_mag[i] = Complex64::new(0.0, 0.0);
    }

    engine.fft_inverse(&mut ln_mag);
    let norm = 1.0 / fft_size as f64;

    // Minimum phase (radians) at each bin
    let min_phase_rad: Vec<f64> = (0..n_bins)
        .map(|i| -ln_mag[i].im * norm)
        .collect();

    // Interpolate measured phase onto same linear grid and unwrap
    let mut grid_phase_deg = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let f = i as f64 * df;
        grid_phase_deg.push(interp_single_val(freq, phase_deg, f));
    }
    let grid_phase_unwrapped = unwrap_phase(&grid_phase_deg);

    // Excess phase = measured − minimum (both in degrees)
    let excess_deg: Vec<f64> = grid_phase_unwrapped.iter()
        .zip(min_phase_rad.iter())
        .map(|(&meas, &min_rad)| meas - min_rad.to_degrees())
        .collect();

    // Linear fit on excess phase in the useful range (skip DC and near-Nyquist)
    let fit_lo = (n_bins as f64 * 0.05) as usize; // skip lowest 5%
    let fit_hi = (n_bins as f64 * 0.85) as usize; // skip highest 15%
    if fit_hi <= fit_lo + 2 { return 0.0; }

    let mut sum_f = 0.0;
    let mut sum_p = 0.0;
    let mut sum_ff = 0.0;
    let mut sum_fp = 0.0;
    let cnt = (fit_hi - fit_lo) as f64;
    for i in fit_lo..fit_hi {
        let f = i as f64 * df;
        let p = excess_deg[i];
        sum_f += f;
        sum_p += p;
        sum_ff += f * f;
        sum_fp += f * p;
    }
    let denom = cnt * sum_ff - sum_f * sum_f;
    if denom.abs() < 1e-30 { return 0.0; }
    let slope = (cnt * sum_fp - sum_f * sum_p) / denom;

    // delay = -slope / 360
    -slope / 360.0
}

/// Simple linear interpolation of a single value at frequency f.
fn interp_single_val(freq: &[f64], data: &[f64], f: f64) -> f64 {
    if freq.is_empty() { return 0.0; }
    if f <= freq[0] { return data[0]; }
    let last = freq.len() - 1;
    if f >= freq[last] { return data[last]; }
    let mut lo = 0;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if freq[mid] <= f { lo = mid; } else { hi = mid; }
    }
    let t = (f - freq[lo]) / (freq[hi] - freq[lo]);
    data[lo] + t * (data[hi] - data[lo])
}

/// Convert delay in seconds to distance in meters.
/// Uses speed of sound at ~20°C: 343 m/s.
pub fn compute_distance(delay_seconds: f64) -> f64 {
    delay_seconds * 343.0
}

/// Remove propagation delay from phase data.
///
/// Subtracts the linear phase component corresponding to the given delay:
///   φ_new[i] = φ[i] + 360 · freq[i] · delay_seconds
///
/// (A positive delay means phase decreases with frequency, so we add to compensate.)
pub fn remove_delay(freq: &[f64], phase_deg: &[f64], delay_seconds: f64) -> Vec<f64> {
    freq.iter()
        .zip(phase_deg.iter())
        .map(|(&f, &p)| p + 360.0 * f * delay_seconds)
        .collect()
}

/// Estimate propagation delay via IR peak detection (IFFT method).
///
/// This matches REW's "estimated IR delay" method:
/// 1. Interpolate measurement onto linear frequency grid
/// 2. Build complex spectrum from magnitude + phase
/// 3. IFFT → impulse response
/// 4. Find peak → delay = peak_index × dt
pub fn compute_ir_delay(
    freq: &[f64],
    magnitude: &[f64],
    phase_deg: &[f64],
    sample_rate: f64,
) -> f64 {
    use num_complex::Complex64;
    use crate::dsp::fft::FftEngine;

    let n = freq.len();
    if n < 2 {
        return 0.0;
    }

    // FFT size: large enough for good time resolution
    let fft_size = {
        let desired = (n * 8).max(8192);
        desired.next_power_of_two()
    };
    let n_bins = fft_size / 2 + 1;
    let nyquist = sample_rate / 2.0;
    let df = nyquist / (n_bins - 1) as f64;

    // Interpolate onto linear grid
    let mut grid_mag = Vec::with_capacity(n_bins);
    let mut grid_phase = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let f = i as f64 * df;
        let (m, p) = interp_at(freq, magnitude, phase_deg, f);
        grid_mag.push(m);
        grid_phase.push(p);
    }

    // Build complex spectrum
    let mut spectrum: Vec<Complex64> = Vec::with_capacity(fft_size);
    for i in 0..n_bins {
        let amp = 10.0_f64.powf(grid_mag[i] / 20.0);
        let ph_rad = grid_phase[i].to_radians();
        spectrum.push(Complex64::new(amp * ph_rad.cos(), amp * ph_rad.sin()));
    }
    // Mirror for conjugate symmetry
    for i in 1..(fft_size - n_bins + 1) {
        let idx = n_bins - 1 - i;
        spectrum.push(spectrum[idx].conj());
    }

    // IFFT
    let mut engine = FftEngine::new();
    engine.fft_inverse(&mut spectrum);

    let norm = 1.0 / fft_size as f64;
    let impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // Find peak in first half (causal part)
    let half = fft_size / 2;
    let peak_idx = impulse[..half]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let dt = 1.0 / sample_rate;
    peak_idx as f64 * dt
}

/// Simple linear interpolation of magnitude and phase at frequency `f`.
fn interp_at(freq: &[f64], mag: &[f64], phase: &[f64], f: f64) -> (f64, f64) {
    if freq.is_empty() {
        return (0.0, 0.0);
    }
    if f <= freq[0] {
        return (mag[0], phase[0]);
    }
    let last = freq.len() - 1;
    if f >= freq[last] {
        return (mag[last], phase[last]);
    }
    // Binary search
    let mut lo = 0;
    let mut hi = last;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if freq[mid] <= f {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (f - freq[lo]) / (freq[hi] - freq[lo]);
    let m = mag[lo] + t * (mag[hi] - mag[lo]);
    let p = phase[lo] + t * (phase[hi] - phase[lo]);
    (m, p)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_delay_of_constant_phase_is_zero() {
        let freq = vec![100.0, 200.0, 500.0, 1000.0, 2000.0];
        let phase = vec![45.0, 45.0, 45.0, 45.0, 45.0]; // constant phase
        let gd = compute_group_delay(&freq, &phase);
        for &v in &gd {
            assert!(v.abs() < 1e-10, "Constant phase should have zero group delay, got {}", v);
        }
    }

    #[test]
    fn group_delay_of_linear_phase() {
        // Linear phase: φ(f) = -360 * τ * f
        // Group delay should be τ everywhere
        let tau = 0.005; // 5 ms
        let freq: Vec<f64> = (0..100).map(|i| 20.0 + i as f64 * 200.0).collect();
        let phase: Vec<f64> = freq.iter().map(|&f| -360.0 * tau * f).collect();
        let gd = compute_group_delay(&freq, &phase);
        // Interior points should be very close to τ
        for i in 1..gd.len() - 1 {
            assert!(
                (gd[i] - tau).abs() < 1e-8,
                "Group delay at f={} should be {}, got {}",
                freq[i], tau, gd[i]
            );
        }
    }

    #[test]
    fn average_delay_in_range() {
        let tau = 0.003; // 3 ms
        let freq: Vec<f64> = (0..200).map(|i| 20.0 + i as f64 * 100.0).collect();
        let phase: Vec<f64> = freq.iter().map(|&f| -360.0 * tau * f).collect();
        let avg = compute_average_delay(&freq, &phase, 1000.0, 4000.0);
        assert!(
            (avg - tau).abs() < 1e-6,
            "Average delay should be ~{}, got {}",
            tau, avg
        );
    }

    #[test]
    fn distance_from_delay() {
        let d = compute_distance(0.001); // 1 ms
        assert!((d - 0.343).abs() < 1e-6, "1ms delay = 0.343m, got {}", d);
    }

    #[test]
    fn remove_delay_roundtrip() {
        let tau = 0.005;
        let freq: Vec<f64> = (0..50).map(|i| 20.0 + i as f64 * 400.0).collect();
        let original_phase: Vec<f64> = freq.iter().map(|&f| -30.0 * (f / 1000.0).sin()).collect();

        // Add delay
        let delayed: Vec<f64> = freq.iter()
            .zip(original_phase.iter())
            .map(|(&f, &p)| p - 360.0 * f * tau)
            .collect();

        // Remove delay
        let restored = remove_delay(&freq, &delayed, tau);

        for i in 0..freq.len() {
            assert!(
                (restored[i] - original_phase[i]).abs() < 1e-8,
                "Phase at f={} should be restored: expected {}, got {}",
                freq[i], original_phase[i], restored[i]
            );
        }
    }

    #[test]
    fn unwrap_phase_removes_jumps() {
        // Wrapped: 170 → -170 (jump +200 → should be +20)
        let wrapped = vec![170.0, -170.0, -10.0, 170.0, -10.0];
        let unwrapped = unwrap_phase(&wrapped);
        // 170 → 190 → 350 → 530 → 710
        assert!((unwrapped[0] - 170.0).abs() < 1e-10);
        assert!((unwrapped[1] - 190.0).abs() < 1e-10);
        assert!((unwrapped[2] - 350.0).abs() < 1e-10);
        assert!((unwrapped[3] - 530.0).abs() < 1e-10);
        assert!((unwrapped[4] - 710.0).abs() < 1e-10);
    }

    #[test]
    fn unwrap_phase_with_delay() {
        // Simulate 2ms delay wrapped phase at log-spaced freqs
        let tau = 0.002;
        let freq: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 200.0).collect();
        let unwrapped_original: Vec<f64> = freq.iter().map(|&f| -360.0 * tau * f).collect();
        // Wrap to ±180
        let wrapped: Vec<f64> = unwrapped_original.iter().map(|&p| {
            (p + 180.0).rem_euclid(360.0) - 180.0
        }).collect();
        let restored = unwrap_phase(&wrapped);
        // Group delay from unwrapped should match tau
        let gd = compute_group_delay(&freq, &restored);
        for i in 1..gd.len() - 1 {
            assert!(
                (gd[i] - tau).abs() < 0.001,
                "Group delay at f={} should be ~{}, got {}",
                freq[i], tau, gd[i]
            );
        }
    }

    #[test]
    fn empty_input() {
        let gd = compute_group_delay(&[], &[]);
        assert!(gd.is_empty());

        let avg = compute_average_delay(&[], &[], 1000.0, 4000.0);
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn single_point() {
        let gd = compute_group_delay(&[1000.0], &[45.0]);
        assert_eq!(gd.len(), 1);
        assert_eq!(gd[0], 0.0);
    }
}
