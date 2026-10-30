//! b141.111: how long one crossover section (HP or LP) rings — the
//! crossover counterpart of the PEQ «мс» column (`peq::q_envelope::ring_ms`).
//!
//! A PEQ point rings by its deviation from flat; a crossover rings by its very
//! nature (longer the lower, steeper and higher-Q it is), so there is no pole
//! formula that covers every family (Gaussian has no poles at all). The ring
//! is measured on the section's own impulse instead: the response the FIR
//! realises, without the ultrasonic low-pass — a min-phase LR/BW/Custom/Bessel
//! section as the bilinear cascade pre-warped at its corner, a min-phase
//! Gaussian by cepstrum on this very grid, a linear section zero-phase plus
//! the plot's Hilbert terms (`target_hilbert_phase`, its subsonic) — at
//! the export rate, on a grid long enough for the tail to die, IFFT'd and
//! read from the peak to the last sample above −60 / −100 dB (after) and from
//! the first such sample to the peak (before; meaningful on linear phase,
//! where the impulse is symmetric). Same reading as export-metrics' pre-ring.
//!
//! A high-pass impulse is a click (δ) plus the decaying wave, and the click's
//! height relative to the wave grows with the sample rate (×8 from 48 to
//! 384 kHz) — a ring read against it would shrink with the rate. The ring is
//! read on the wave alone, 1 − HP: the deviation from flat, as for a PEQ.
use rustfft::{num_complex::Complex64, FftPlanner};
use serde::Serialize;
use std::f64::consts::PI;

use super::band::target_hilbert_phase;
use crate::target::{self, FilterConfig, FilterType, TargetCurve};

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SectionRing {
    /// Peak → last sample above −60 / −100 dB, ms.
    pub post_60_ms: f64,
    pub post_100_ms: f64,
    /// First sample above −60 / −100 dB → peak, ms (linear phase: pre-ring;
    /// minimum phase: only the rise time to the peak).
    pub pre_60_ms: f64,
    pub pre_100_ms: f64,
    /// The tail outlived the longest grid: the values are lower bounds.
    pub lower_bound: bool,
}

/// Envelope reading of an impulse (any scale): (pre60, pre100, post60,
/// post100) in samples around its absolute peak.
pub fn ring_samples(h: &[f64]) -> (usize, usize, usize, usize) {
    let (p, peak) = h.iter().enumerate().fold((0, 0.0f64), |(i, m), (k, v)| if v.abs() > m { (k, v.abs()) } else { (i, m) });
    let reach = |db: f64| {
        let thr = peak * 10f64.powf(-db / 20.0);
        let first = h.iter().position(|v| v.abs() > thr).unwrap_or(p);
        let last = h.iter().rposition(|v| v.abs() > thr).unwrap_or(p);
        (p - first.min(p), last.max(p) - p)
    };
    let ((b60, a60), (b100, a100)) = (reach(60.0), reach(100.0));
    (b60, b100, a60, a100)
}

/// Shortest grid (samples) and the longest one tried before giving a lower bound.
const N_MIN: usize = 1 << 15;
const N_MAX: usize = 1 << 22;

pub fn section_ring(cfg: &FilterConfig, is_low_pass: bool, sample_rate: f64) -> Result<SectionRing, String> {
    if !(cfg.freq_hz > 0.0) || !(sample_rate > 0.0) { return Err("section_ring: bad frequency or sample rate".into()); }
    let mut n = N_MIN.max((sample_rate * 0.5).ceil() as usize).next_power_of_two();
    loop {
        let mut h = impulse(cfg, is_low_pass, sample_rate, n)?;
        if !is_low_pass { h[n / 2] -= 1.0; }
        // A min-phase section is causal from its origin n/2: read it from
        // there (before it lies only the cepstrum's −300 dB floor noise).
        let (b60, b100, a60, a100) = ring_samples(if cfg.linear_phase { &h } else { &h[n / 2..] });
        // A tail closer than n/8 to either end may already wrap around, and
        // the far eighth(s) must sit 40 dB under the −100 dB threshold (the
        // cepstrum's time-aliasing floor falls ~17 dB per doubling: a Gaussian
        // M1 HP at 80 Hz read 144 ms to −100 dB at 32K, 98 ms at 64K with its
        // floor at −128 dB, 14.5–14.9 ms from 128K on).
        let peak = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let head: &[f64] = if cfg.linear_phase { &h[..n / 8] } else { &[] };
        let edge = head.iter().chain(&h[n - n / 8..]).fold(0.0f64, |m, v| m.max(v.abs()));
        let fits = b100 < n * 3 / 8 && a100 < n * 3 / 8 && edge <= peak * 1e-7;
        if fits || n >= N_MAX {
            let ms = |s: usize| s as f64 / sample_rate * 1000.0;
            return Ok(SectionRing { post_60_ms: ms(a60), post_100_ms: ms(a100), pre_60_ms: ms(b60),
                pre_100_ms: ms(b100), lower_bound: !fits });
        }
        n *= 2;
    }
}

/// The section's impulse on an n-point grid, its time origin at n/2.
fn impulse(cfg: &FilterConfig, is_low_pass: bool, sr: f64, n: usize) -> Result<Vec<f64>, String> {
    let half = n / 2;
    let freq: Vec<f64> = (0..=half).map(|k| k as f64 * sr / n as f64).collect();
    let one = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: (!is_low_pass).then(|| cfg.clone()), low_pass: is_low_pass.then(|| cfg.clone()),
        low_shelf: None, high_shelf: None };
    // A min-phase LR/BW/Custom/Bessel section is the bilinear cascade
    // pre-warped at its corner: the analog response at fc·tan(πf/sr)/tan(πfc/sr).
    let digital = !cfg.linear_phase && !matches!(cfg.filter_type, FilterType::Gaussian);
    let tw = (PI * cfg.freq_hz / sr).tan();
    let fe: Vec<f64> = if digital {
        freq.iter().map(|&f| if f >= sr / 2.0 { 1e12 } else { cfg.freq_hz * (PI * f / sr).tan() / tw }).collect()
    } else { freq.clone() };
    let e = target::evaluate(&one, &fe);
    // A min-phase Gaussian (and its subsonic) is the minimum phase of its
    // magnitude: taken by cepstrum on this very grid. The plot's Hilbert
    // grid starts at 5 Hz and holds its phase below — a phase step at the
    // lowest bins that read as a −100 dB tail over a second.
    let phase_deg: Vec<f64> = if !cfg.linear_phase && matches!(cfg.filter_type, FilterType::Gaussian) {
        min_phase_deg(&e.magnitude, n)
    } else {
        let (hp, lp) = (one.high_pass.as_ref(), one.low_pass.as_ref());
        let hil = target_hilbert_phase(&freq[1..], hp, lp, None, sr)?;
        (0..=half).map(|k| e.phase[k] + if k > 0 { hil[k - 1] } else { 0.0 }).collect()
    };
    let mut spec = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..=half {
        let ph = phase_deg[k];
        // (−1)^k puts the time origin at n/2.
        let v = Complex64::from_polar(10f64.powf(e.magnitude[k] / 20.0), ph.to_radians()) * if k % 2 == 0 { 1.0 } else { -1.0 };
        spec[k] = if k == 0 || k == half { Complex64::new(v.re, 0.0) } else { v };
        if k > 0 && k < half { spec[n - k] = v.conj(); }
    }
    FftPlanner::new().plan_fft_inverse(n).process(&mut spec);
    Ok(spec.iter().map(|c| c.re / n as f64).collect())
}

/// Minimum phase (degrees, bins 0..=n/2) of a magnitude in dB by the real
/// cepstrum on an n-point grid, the magnitude floored at −300 dB.
fn min_phase_deg(mag_db: &[f64], n: usize) -> Vec<f64> {
    let half = n / 2;
    let ln = |db: f64| (db.max(-300.0) / 20.0) * std::f64::consts::LN_10;
    let mut c: Vec<Complex64> = (0..n).map(|k| Complex64::new(ln(mag_db[if k <= half { k } else { n - k }]), 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_inverse(n).process(&mut c);
    for (k, v) in c.iter_mut().enumerate() {
        let w = if k == 0 || k == half { 1.0 } else if k < half { 2.0 } else { 0.0 };
        *v = Complex64::new(v.re / n as f64 * w, 0.0);
    }
    planner.plan_fft_forward(n).process(&mut c);
    c[..=half].iter().map(|v| v.im.to_degrees()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ft: FilterType, order: u8, fc: f64, lin: bool) -> FilterConfig {
        FilterConfig { filter_type: ft, order, freq_hz: fc, shape: None, linear_phase: lin, q: None,
            subsonic_protect: None, complementary: false }
    }

    /// A linear section is symmetric: as long before the peak as after.
    #[test]
    fn linear_section_rings_symmetrically() {
        let r = section_ring(&cfg(FilterType::Butterworth, 4, 100.0, true), false, 48_000.0).unwrap();
        assert!((r.pre_60_ms - r.post_60_ms).abs() <= 0.05, "{r:?}");
        assert!((r.pre_100_ms - r.post_100_ms).abs() <= 0.05, "{r:?}");
        assert!(!r.lower_bound);
    }

    /// Ring scales as 1/fc and grows with the slope.
    #[test]
    fn ring_scales_with_corner_and_slope() {
        let sr = 48_000.0;
        let lo = section_ring(&cfg(FilterType::Butterworth, 4, 100.0, false), false, sr).unwrap();
        let hi = section_ring(&cfg(FilterType::Butterworth, 4, 1000.0, false), false, sr).unwrap();
        let steep = section_ring(&cfg(FilterType::Butterworth, 8, 100.0, false), false, sr).unwrap();
        println!("BW4 HP 100 Hz {lo:?}\nBW4 HP 1 kHz {hi:?}\nBW8 HP 100 Hz {steep:?}");
        let ratio = lo.post_60_ms / hi.post_60_ms;
        assert!((ratio - 10.0).abs() < 1.5, "100 Hz / 1 kHz ring ratio {ratio:.2}");
        assert!(steep.post_60_ms > 1.5 * lo.post_60_ms);
    }
}
