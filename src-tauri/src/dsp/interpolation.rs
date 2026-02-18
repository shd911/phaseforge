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

    let grid_mag = interp_1d_with_edges(freq, mag, &grid_freq);
    let grid_phase = phase.map(|p| interp_1d_with_edges(freq, p, &grid_freq));

    (grid_freq, grid_mag, grid_phase)
}

/// Simple linear interpolation with clamping to boundary values.
fn interp_1d(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query
        .iter()
        .map(|&xq| interp_single(x_data, y_data, xq))
        .collect()
}

/// Linear interpolation with special edge handling for FFT grids.
/// DC (0 Hz): extrapolate from first point, Phase = 0.
/// Beyond Nyquist: hold last value.
fn interp_1d_with_edges(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query
        .iter()
        .map(|&xq| interp_single(x_data, y_data, xq))
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
    let idx = match x_data.binary_search_by(|v| v.partial_cmp(&xq).unwrap()) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
