pub mod baffle;
pub mod impulse;
pub mod merge;
mod interpolation;
mod smoothing;

pub use interpolation::{interpolate_linear_grid, interpolate_log_grid};
pub use smoothing::{fractional_octave_smooth, variable_smoothing, SmoothingConfig};

/// Generate a logarithmically-spaced frequency grid.
pub fn generate_log_freq_grid(n: usize, f_min: f64, f_max: f64) -> Vec<f64> {
    if n < 2 {
        return vec![f_min];
    }
    let log_min = f_min.ln();
    let log_max = f_max.ln();
    (0..n)
        .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
        .collect()
}
