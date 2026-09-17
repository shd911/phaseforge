//! b141.40: ultrasonic low-pass applied to every exported FIR at high rates.
//!
//! Before this, the two routes band-limited very differently above the
//! audio band:
//!   - cepstral: the target was evaluated only to 40 kHz and then forced to
//!     the noise floor within ~2 kHz — a brick wall. At 352.8 kHz a
//!     linear-phase tweeter (LR 24 dB/oct @ 2039 Hz) read 0 dB at 40 kHz and
//!     −79 dB at 41 kHz; the wall's 40 kHz ringing was what the pre-ring
//!     metric measured (2.24 ms at −60 dB vs 0.49 ms for the crossover).
//!   - IIR: no limit at all — the biquads pass everything to Nyquist.
//!
//! Measured on that tweeter (65536 taps, Blackman), comparing Butterworth
//! 4/8 and LR 24/48 at 30–50 kHz against the wall and no limit, over
//! 88.2–384 kHz. Criteria: loss at 20 kHz, ultrasonic noise bandwidth above
//! 24 kHz (∫|H|²df, white-noise load on tweeter/amp), worst-case output peak
//! (Σ|h|), pre-ring. Butterworth-8 at 30 kHz won:
//!   loss at 20 kHz 0.007 dB (BW4 0.17, LR48 0.33 at the same corner);
//!   ultrasonic bandwidth 6.6 kHz at every rate (wall 17, IIR today 161 at
//!   352.8 kHz); worst-case peak −4 dB vs the wall; pre-ring back to the
//!   crossover's own 0.49 ms.
//!
//! ZERO PHASE, on purpose. A min-phase BW8 @ 30 kHz delays by ~27 µs. It
//! would reach only the bands with content above 30 kHz (tweeters): the
//! cepstral route clips the others to the noise floor, where a Hilbert
//! transform sees no filter. 27 µs is 19° at a 2 kHz crossover. Zero phase
//! delays nothing, so no band moves and no alignment changes; a two-way LR
//! sum stays flat to 0.000 dB below 10 kHz (−0.007 dB at 20 kHz).
//!
//! Applied as a spectral multiply of the finished impulse: one step, both
//! routes, after the WAV delay is set. Only when the rate leaves room above
//! the corner (≥ 88.2 kHz); at 44.1/48 kHz nothing changes.

use num_complex::Complex64;

use crate::dsp::fft::FftEngine;
use super::types::FirModelResult;

/// Corner of the ultrasonic low-pass (−3 dB).
pub const ULTRASONIC_LP_HZ: f64 = 30_000.0;
/// Butterworth order (48 dB/oct).
pub const ULTRASONIC_LP_ORDER: i32 = 8;
/// Lowest export rate the low-pass applies at. 44.1/48 kHz exports are
/// already limited by their own Nyquist below the corner.
pub const ULTRASONIC_LP_MIN_SR: f64 = 88_200.0;

/// Corner in Hz when the export rate gets the low-pass, else None.
pub fn ultrasonic_lp_for(sample_rate: f64) -> Option<f64> {
    (sample_rate >= ULTRASONIC_LP_MIN_SR).then_some(ULTRASONIC_LP_HZ)
}

/// |H| of the Butterworth low-pass, linear.
pub fn ultrasonic_lp_gain(f: f64, corner: f64) -> f64 {
    (1.0 / (1.0 + (f.abs() / corner).powi(2 * ULTRASONIC_LP_ORDER))).sqrt()
}

/// Apply the zero-phase ultrasonic low-pass to a finished FIR in place:
/// the impulse (spectral multiply), `realized_mag` (on `freq`), and the
/// reported corner. `causality` is left as computed before the low-pass —
/// the symmetric 30 kHz main lobe straddles the peak and would read as
/// "pre-peak energy" in a metric meant for crossover pre-ringing.
pub fn apply_ultrasonic_lp(result: &mut FirModelResult, freq: &[f64]) {
    let Some(corner) = ultrasonic_lp_for(result.sample_rate) else { return };
    let n = result.impulse.len();
    if n < 2 { return; }
    let sr = result.sample_rate;

    let mut buf: Vec<Complex64> = result.impulse.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut engine = FftEngine::new();
    engine.fft_forward(&mut buf);
    for (k, c) in buf.iter_mut().enumerate() {
        let bin = if k <= n / 2 { k } else { n - k };
        *c *= ultrasonic_lp_gain(bin as f64 * sr / n as f64, corner);
    }
    engine.fft_inverse(&mut buf);
    let inv_n = 1.0 / n as f64;
    for (dst, c) in result.impulse.iter_mut().zip(buf.iter()) {
        *dst = c.re * inv_n;
    }

    if result.realized_mag.len() == freq.len() {
        for (m, &f) in result.realized_mag.iter_mut().zip(freq.iter()) {
            *m += 20.0 * ultrasonic_lp_gain(f, corner).log10();
        }
    }
    result.ultrasonic_lp_hz = Some(corner);
}
