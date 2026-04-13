// FIR correction engine: window functions

use std::f64::consts::PI;

use super::types::WindowType;

// ---------------------------------------------------------------------------
// Public(crate) window generators
// ---------------------------------------------------------------------------

/// Half-window for minimum phase FIR: 1.0 at start, smooth taper to 0 at end.
/// Uses the right half of a symmetric window (so the peak is at sample 0).
pub(crate) fn generate_half_window(n: usize, wtype: &WindowType) -> Vec<f64> {
    // Generate a window of length 2*n, then take the right half (indices n..2n)
    // This gives: starts at peak (1.0), decays to 0
    let full = generate_window(2 * n, wtype);
    full[n..].to_vec()
}

pub(crate) fn generate_window(n: usize, wtype: &WindowType) -> Vec<f64> {
    match wtype {
        // Basic / classical
        WindowType::Rectangular => vec![1.0; n],
        WindowType::Bartlett => bartlett_window(n),
        WindowType::Hann => hann_window(n),
        WindowType::Hamming => hamming_window(n),
        WindowType::Blackman => blackman_window(n),
        // Blackman-Harris family
        WindowType::ExactBlackman => exact_blackman_window(n),
        WindowType::BlackmanHarris => blackman_harris_window(n),
        WindowType::Nuttall3 => cosine_sum_window(n, &[0.375, 0.5, 0.125]),
        WindowType::Nuttall4 => cosine_sum_window(n, &[0.3635819, 0.4891775, 0.1365995, 0.0106411]),
        WindowType::FlatTop => cosine_sum_window(n, &[0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]),
        // Parametric
        WindowType::Kaiser => kaiser_window(n, 10.0),
        WindowType::DolphChebyshev => dolph_chebyshev_window(n, 100.0),
        WindowType::Gaussian => gaussian_window(n, 2.5),
        WindowType::Tukey => tukey_window(n, 0.5),
        // Special
        WindowType::Lanczos => lanczos_window(n),
        WindowType::Poisson => poisson_window(n, 2.0),
        WindowType::HannPoisson => hann_poisson_window(n, 2.0),
        WindowType::Bohman => bohman_window(n),
        WindowType::Cauchy => cauchy_window(n, 3.0),
        WindowType::Riesz => riesz_window(n),
    }
}

// ---------------------------------------------------------------------------
// Generic cosine-sum window: w[i] = Σ (-1)^k · a_k · cos(2πki/(N-1))
// ---------------------------------------------------------------------------

fn cosine_sum_window(n: usize, coeffs: &[f64]) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    (0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / (n - 1) as f64;
        let mut val = 0.0;
        for (k, &a) in coeffs.iter().enumerate() {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            val += sign * a * (k as f64 * x).cos();
        }
        val
    }).collect()
}

// ---------------------------------------------------------------------------
// Basic / classical windows
// ---------------------------------------------------------------------------

fn bartlett_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        1.0 - (2.0 * i as f64 / (n - 1) as f64 - 1.0).abs()
    }).collect()
}

fn hann_window(n: usize) -> Vec<f64> {
    cosine_sum_window(n, &[0.5, 0.5])
}

fn hamming_window(n: usize) -> Vec<f64> {
    cosine_sum_window(n, &[0.54, 0.46])
}

fn blackman_window(n: usize) -> Vec<f64> {
    cosine_sum_window(n, &[0.42, 0.5, 0.08])
}

// ---------------------------------------------------------------------------
// Blackman-Harris family
// ---------------------------------------------------------------------------

fn exact_blackman_window(n: usize) -> Vec<f64> {
    cosine_sum_window(n, &[7938.0/18608.0, 9240.0/18608.0, 1430.0/18608.0])
}

fn blackman_harris_window(n: usize) -> Vec<f64> {
    cosine_sum_window(n, &[0.35875, 0.48829, 0.14128, 0.01168])
}

// ---------------------------------------------------------------------------
// Parametric windows
// ---------------------------------------------------------------------------

fn tukey_window(n: usize, alpha: f64) -> Vec<f64> {
    let alpha = alpha.clamp(0.0, 1.0);
    (0..n).map(|i| {
        let x = i as f64 / (n - 1) as f64;
        if x < alpha / 2.0 {
            0.5 * (1.0 + ((2.0 * PI * x / alpha) - PI).cos())
        } else if x > 1.0 - alpha / 2.0 {
            0.5 * (1.0 + ((2.0 * PI * (1.0 - x) / alpha) - PI).cos())
        } else {
            1.0
        }
    }).collect()
}

pub(crate) fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let denom = bessel_i0(beta);
    (0..n).map(|i| {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let arg = beta * (1.0 - x * x).max(0.0).sqrt();
        bessel_i0(arg) / denom
    }).collect()
}

fn gaussian_window(n: usize, sigma: f64) -> Vec<f64> {
    (0..n).map(|i| {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // [-1, 1]
        (-0.5 * (x * sigma).powi(2)).exp()
    }).collect()
}

/// Dolph-Chebyshev window: equiripple sidelobes at -atten_db.
/// Uses inverse DFT of Chebyshev polynomial on the unit circle.
fn dolph_chebyshev_window(n: usize, atten_db: f64) -> Vec<f64> {
    let nn = n as f64;
    let m = (nn - 1.0) / 2.0;
    let order = nn - 1.0;
    // r = sidelobe ratio (linear); x0 via inverse Chebyshev
    let r = 10.0_f64.powf(atten_db / 20.0);
    let x0 = (r.acosh() / order).cosh();

    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 1..n {
            let angle = PI * k as f64 / nn;
            let cheb_arg = x0 * angle.cos();
            let cheb_val = chebyshev_poly(order, cheb_arg);
            sum += cheb_val * (2.0 * PI * k as f64 * (i as f64 - m) / nn).cos();
        }
        w[i] = 1.0 / nn + 2.0 * sum / (nn * r);
    }

    // Normalize peak to 1.0
    let peak = w.iter().cloned().fold(0.0_f64, f64::max);
    if peak > 0.0 {
        for v in &mut w {
            *v /= peak;
        }
    }
    w
}

/// Chebyshev polynomial T_n(x) via the recursive identity
fn chebyshev_poly(order: f64, x: f64) -> f64 {
    if x.abs() <= 1.0 {
        (order * x.acos()).cos()
    } else if x > 1.0 {
        (order * x.acosh()).cosh()
    } else {
        // x < -1
        let sign = if order as i64 % 2 == 0 { 1.0 } else { -1.0 };
        sign * (order * (-x).acosh()).cosh()
    }
}

// ---------------------------------------------------------------------------
// Special windows
// ---------------------------------------------------------------------------

fn lanczos_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // [-1, 1]
        if x.abs() < 1e-12 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        }
    }).collect()
}

fn poisson_window(n: usize, alpha: f64) -> Vec<f64> {
    (0..n).map(|i| {
        let x = (2.0 * i as f64 / (n - 1) as f64 - 1.0).abs(); // |normalized| in [0,1]
        (-alpha * x).exp()
    }).collect()
}

fn hann_poisson_window(n: usize, alpha: f64) -> Vec<f64> {
    let h = hann_window(n);
    let p = poisson_window(n, alpha);
    h.iter().zip(p.iter()).map(|(&a, &b)| a * b).collect()
}

fn bohman_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let x = (2.0 * i as f64 / (n - 1) as f64 - 1.0).abs(); // |x| in [0,1]
        if x >= 1.0 {
            0.0
        } else {
            (1.0 - x) * (PI * x).cos() + (PI * x).sin() / PI
        }
    }).collect()
}

fn cauchy_window(n: usize, alpha: f64) -> Vec<f64> {
    (0..n).map(|i| {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // [-1, 1]
        1.0 / (1.0 + (alpha * x).powi(2))
    }).collect()
}

fn riesz_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // [-1, 1]
        1.0 - x * x
    }).collect()
}

// ---------------------------------------------------------------------------
// Bessel function for Kaiser window
// ---------------------------------------------------------------------------

/// Modified Bessel function of the first kind, order 0 (I₀).
/// Computed via series expansion (converges fast for typical beta values).
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..50 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-20 {
            break;
        }
    }
    sum
}
