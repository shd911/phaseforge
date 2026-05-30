/// Interpolate measurement onto a log-spaced frequency grid.
///
/// Used for display and PEQ fitting. Typically 1024 points from 10 Hz to 24 kHz.
/// Uses linear interpolation in log(freq) space.
pub fn interpolate_log_grid(
    freq: &[f64],
    mag: &[f64],
    phase: Option<&[f64]>,
    n_points: usize,
    f_min: f64,
    f_max: f64,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let log_min = f_min.ln();
    let log_max = f_max.ln();

    if n_points < 2 {
        let f = ((log_min + log_max) / 2.0).exp();
        let m = interp_1d(freq, mag, &[f]);
        let p = phase.map(|ph| interp_1d(freq, ph, &[f]));
        return (vec![f], m, p);
    }

    let grid_freq: Vec<f64> = (0..n_points)
        .map(|i| {
            let t = i as f64 / (n_points - 1) as f64;
            (log_min + t * (log_max - log_min)).exp()
        })
        .collect();

    let grid_mag = interp_1d(freq, mag, &grid_freq);
    let grid_phase = phase.map(|p| interp_1d(freq, p, &grid_freq));

    (grid_freq, grid_mag, grid_phase)
}

/// Interpolate measurement onto a linear frequency grid for FFT/FIR.
///
/// Grid: 0 Hz to Nyquist with `n_bins = tap_count / 2 + 1` points.
pub fn interpolate_linear_grid(
    freq: &[f64],
    mag: &[f64],
    phase: Option<&[f64]>,
    n_bins: usize,
    sample_rate: f64,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let nyquist = sample_rate / 2.0;
    let df = nyquist / (n_bins - 1) as f64;

    let grid_freq: Vec<f64> = (0..n_bins).map(|i| i as f64 * df).collect();

    let grid_mag = interp_1d(freq, mag, &grid_freq); // magnitude: clamp to boundary (flat extension)
    let grid_phase = phase.map(|p| interp_1d_phase_edges(freq, p, &grid_freq)); // phase: blend to 0 at DC

    (grid_freq, grid_mag, grid_phase)
}

/// Simple linear interpolation with clamping to boundary values.
pub fn interp_1d(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query
        .iter()
        .map(|&xq| interp_single(x_data, y_data, xq))
        .collect()
}

/// Phase interpolation with DC edge handling for FFT grids.
/// Below measurement range: linearly blend from 0 at DC to first data value.
/// This ensures DC phase = 0° for real signals.
/// Within and above range: standard interpolation with boundary clamping.
fn interp_1d_phase_edges(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query
        .iter()
        .map(|&xq| {
            if x_data.is_empty() {
                return 0.0;
            }
            if xq >= x_data[0] {
                return interp_single_phase(x_data, y_data, xq);
            }
            // Below measurement range: blend phase from 0 (at DC) to phase[0]
            if x_data[0] <= 0.0 {
                return y_data[0];
            }
            let t = xq / x_data[0]; // 0.0 at DC, 1.0 at first measurement freq
            y_data[0] * t
        })
        .collect()
}

/// Interpolate a single value. Clamps to boundary values for out-of-range queries.
fn interp_single(x_data: &[f64], y_data: &[f64], xq: f64) -> f64 {
    if x_data.is_empty() {
        return 0.0;
    }
    if xq <= x_data[0] {
        return y_data[0];
    }
    if xq >= x_data[x_data.len() - 1] {
        return y_data[y_data.len() - 1];
    }

    // Binary search for the interval
    let idx = match x_data.binary_search_by(|v| v.partial_cmp(&xq).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => return y_data[i],
        Err(i) => i,
    };

    // Linear interpolation between idx-1 and idx
    let x0 = x_data[idx - 1];
    let x1 = x_data[idx];
    let y0 = y_data[idx - 1];
    let y1 = y_data[idx];

    let t = (xq - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
}

/// Wrap-aware phase interpolation (degrees in, degrees out, range (-180, 180]).
///
/// Identical to `interp_single` whenever the two bracketing samples differ by
/// ≤ 180° (the dense-grid case), so existing golden snapshots are unchanged.
/// When the bracketing samples straddle the ±180° wrap (|Δ| > 180°), the
/// endpoint is unwrapped before the linear blend and the result re-wrapped —
/// taking the shortest arc instead of producing a phantom mid-value (e.g.
/// +179° and −179° interpolate to ±180°, not 0°).
fn interp_single_phase(x_data: &[f64], y_data: &[f64], xq: f64) -> f64 {
    if x_data.is_empty() {
        return 0.0;
    }
    if xq <= x_data[0] {
        return y_data[0];
    }
    if xq >= x_data[x_data.len() - 1] {
        return y_data[y_data.len() - 1];
    }
    let idx = match x_data.binary_search_by(|v| v.partial_cmp(&xq).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => return y_data[i],
        Err(i) => i,
    };
    let x0 = x_data[idx - 1];
    let x1 = x_data[idx];
    let y0 = y_data[idx - 1];
    let mut y1 = y_data[idx];

    // Unwrap a single ±360° jump so the blend follows the shortest arc.
    let d = y1 - y0;
    if d > 180.0 {
        y1 -= 360.0;
    } else if d < -180.0 {
        y1 += 360.0;
    }

    let t = (xq - x0) / (x1 - x0);
    let interp = y0 + t * (y1 - y0);
    // Re-wrap to (-180, 180]. No-op when no wrap was crossed.
    interp - 360.0 * ((interp + 180.0) / 360.0 - 1.0).ceil()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_interp_across_wrap_takes_shortest_arc() {
        // Sparse grid straddling the ±180° wrap: true continuous phase
        // runs 179° → 181° (≡ −179°). Midpoint must be 180°, not 0°.
        let x = vec![100.0, 300.0];
        let y = vec![179.0, -179.0];
        let out = interp_1d_phase_edges(&x, &y, &[200.0])[0];
        let w = out - 360.0 * (out / 360.0).round();
        assert!((w.abs() - 180.0).abs() < 1.0, "phantom phase: got {out}°");
    }

    #[test]
    fn phase_interp_no_wrap_matches_linear() {
        // |Δ| ≤ 180° → must be bit-identical to plain linear interp.
        let x = vec![100.0, 300.0];
        let y = vec![10.0, 50.0];
        let out = interp_1d_phase_edges(&x, &y, &[200.0])[0];
        assert!((out - 30.0).abs() < 1e-12, "got {out}");
    }

    #[test]
    fn test_interp_single_exact() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0];
        assert!((interp_single(&x, &y, 2.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_single_mid() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0];
        assert!((interp_single(&x, &y, 1.5) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_single_clamp() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![10.0, 20.0, 30.0];
        assert!((interp_single(&x, &y, 0.0) - 10.0).abs() < 1e-10);
        assert!((interp_single(&x, &y, 5.0) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_grid() {
        let freq = vec![20.0, 200.0, 2000.0, 20000.0];
        let mag = vec![60.0, 70.0, 80.0, 75.0];
        let (gf, gm, _) = interpolate_log_grid(&freq, &mag, None, 100, 20.0, 20000.0);
        assert_eq!(gf.len(), 100);
        assert_eq!(gm.len(), 100);
        assert!((gf[0] - 20.0).abs() < 1e-6);
        assert!((gf[99] - 20000.0).abs() < 0.1);
    }
}
