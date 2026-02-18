// Phase engine: group delay, distance computation, delay removal

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

/// Compute average group delay over a frequency range [f_low, f_high].
///
/// Returns delay in seconds. Uses the 1–4 kHz range by default
/// where group delay is typically most stable in room measurements.
pub fn compute_average_delay(
    freq: &[f64],
    phase_deg: &[f64],
    f_low: f64,
    f_high: f64,
) -> f64 {
    let gd = compute_group_delay(freq, phase_deg);
    let mut sum = 0.0;
    let mut count = 0usize;

    for i in 0..freq.len() {
        if freq[i] >= f_low && freq[i] <= f_high {
            sum += gd[i];
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        // Fallback: average over all points
        let total: f64 = gd.iter().sum();
        if gd.is_empty() {
            0.0
        } else {
            total / gd.len() as f64
        }
    }
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
