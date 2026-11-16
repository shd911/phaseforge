//! b141.153: is a linear-phase section's pre-ring audible? (Algorithm from the
//! «5 полос» acoustics project, 16.11.2026.)
//!
//! The level of the first pre-lobe does not depend on the corner — only the
//! time scale does (∝ 1/fc) — so it cannot tell a 3 kHz LR from a 50 Hz one.
//! The corner enters through three things that do not scale with the filter:
//! the backward-masking window is fixed in ms, the hearing threshold (ISO 226)
//! rises steeply toward LF, and the ear's filter (ERB) is wide relative to f
//! at LF, narrow at HF.
//!
//! The impulse (peak = 1; an HP read on 1 − HP, as the ring) goes through a
//! bank of 4th-order gammatones, fc·2^−1.5 … fc·2^1 in ½-ERB steps. In each
//! channel the energy of the envelope that arrives earlier than τ before the
//! event is compared with the threshold tone's energy over 200 ms (temporal
//! integration):  X_k(τ) = 10·lg(E_k(τ) / 0.2 s) − ATH(f_k).
//! The margin over threshold at a peak level L (dB SPL at the listener) is
//! L + max_k X_k(τ) — the level is added in the UI.
use rustfft::{num_complex::Complex64, FftPlanner};
use serde::Serialize;

/// Masking windows (ms) the verdict is read at.
pub const TAUS_MS: [f64; 3] = [1.0, 2.0, 5.0];
/// Numerically nothing.
const NONE_DB: f64 = -200.0;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PreRingAudibility {
    /// max_k X_k at τ = 1, 2, 5 ms, dB re a 0 dB SPL peak (≥ −200).
    pub rel_db: [f64; 3],
    /// Channel that decides X at τ = 2 ms, Hz.
    pub channel_hz: f64,
}

/// ISO 226:2003 hearing threshold Tf, dB SPL.
const ISO226: [(f64, f64); 29] = [
    (20.0, 78.5), (25.0, 68.7), (31.5, 59.5), (40.0, 51.1), (50.0, 44.0), (63.0, 37.5), (80.0, 31.5),
    (100.0, 26.5), (125.0, 22.1), (160.0, 17.9), (200.0, 14.4), (250.0, 11.4), (315.0, 8.6), (400.0, 6.2),
    (500.0, 4.4), (630.0, 3.0), (800.0, 2.2), (1000.0, 2.4), (1250.0, 3.5), (1600.0, 1.7), (2000.0, -1.3),
    (2500.0, -4.2), (3150.0, -6.0), (4000.0, -5.4), (5000.0, -1.5), (6300.0, 6.0), (8000.0, 12.6),
    (10000.0, 13.9), (12500.0, 12.3),
];

/// Hearing threshold at f (log-f interpolation, held beyond the table).
pub fn hearing_threshold_db(f: f64) -> f64 {
    let (lo, hi) = (ISO226[0], ISO226[ISO226.len() - 1]);
    if f <= lo.0 { return lo.1; }
    if f >= hi.0 { return hi.1; }
    let i = ISO226.iter().position(|&(x, _)| x >= f).unwrap();
    let ((f0, t0), (f1, t1)) = (ISO226[i - 1], ISO226[i]);
    t0 + (t1 - t0) * (f / f0).ln() / (f1 / f0).ln()
}

/// Glasberg–Moore 1990.
fn erb(f: f64) -> f64 { 24.7 * (4.37 * f / 1000.0 + 1.0) }
fn cam(f: f64) -> f64 { 21.4 * (4.37 * f / 1000.0 + 1.0).log10() }
fn cam_inv(c: f64) -> f64 { (10f64.powf(c / 21.4) - 1.0) * 1000.0 / 4.37 }

/// Channel centres fc·2^−1.5 … fc·2^1, ½ ERB apart.
pub fn channels(fc: f64, nyquist: f64) -> Vec<f64> {
    let (c0, c1) = (cam(fc * 2f64.powf(-1.5)), cam((fc * 2.0).min(0.95 * nyquist)));
    let mut v = Vec::new();
    let mut c = c0;
    while c <= c1 + 1e-9 { v.push(cam_inv(c)); c += 0.5; }
    v
}

/// `h`: the section's impulse at `sr`, any scale (normalised to its peak
/// here), the event at its absolute peak.
pub fn pre_ring_audibility(h: &[f64], sr: f64, fc: f64) -> PreRingAudibility {
    let n = h.len();
    let (p, peak) = h.iter().enumerate().fold((0, 0.0f64), |(i, m), (k, v)| if v.abs() > m { (k, v.abs()) } else { (i, m) });
    let mut planner = FftPlanner::new();
    let mut spec: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v / peak, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut spec);
    let ifft = planner.plan_fft_inverse(n);
    let mut rel_db = [NONE_DB; 3];
    let mut best2 = (NONE_DB, fc);
    for fk in channels(fc, sr / 2.0) {
        let b = 1.019 * erb(fk);
        // Causal complex gammatone, unit gain at fk: 1/(1 + j(f − fk)/b)^4 on
        // EVERY bin, negative frequencies included (k > n/2 is f < 0). Zeroing
        // them (an analytic signal) is not causal: the LP's DC through the
        // channel's skirt spread a 1/t tail ahead of the event (LR 24 dB/oct
        // at 1 kHz read −8 instead of −37 dB at 1 ms). |y| is half the
        // amplitude of the tone the channel rings with.
        let mut y: Vec<Complex64> = (0..n).map(|k| {
            let f = if k <= n / 2 { k as f64 } else { k as f64 - n as f64 } * sr / n as f64;
            spec[k] * Complex64::new(1.0, (f - fk) / b).powi(4).inv()
        }).collect();
        ifft.process(&mut y);
        let ath = hearing_threshold_db(fk);
        for (i, tau) in TAUS_MS.iter().enumerate() {
            let end = p.saturating_sub((tau * 1e-3 * sr).round() as usize);
            // e²/2 = the tone's mean square; ∫ dt = Σ / sr.
            let e: f64 = y[..end].iter().map(|v| (2.0 * v.norm() / n as f64).powi(2) / 2.0).sum::<f64>() / sr;
            let x = if e > 0.0 { (10.0 * (e / 0.2).log10() - ath).max(NONE_DB) } else { NONE_DB };
            if x > rel_db[i] { rel_db[i] = x; }
            if i == 1 && x > best2.0 { best2 = (x, fk); }
        }
    }
    PreRingAudibility { rel_db, channel_hz: best2.1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hearing_threshold_table() {
        assert_eq!(hearing_threshold_db(31.5), 59.5);
        assert_eq!(hearing_threshold_db(1000.0), 2.4);
        assert_eq!(hearing_threshold_db(5.0), 78.5);
        let m = hearing_threshold_db((40.0f64 * 50.0).sqrt());
        assert!((m - (51.1 + 44.0) / 2.0).abs() < 1e-9);
    }

    #[test]
    fn channels_span_the_corner_in_half_erb() {
        let c = channels(1000.0, 24_000.0);
        assert!((c[0] - 1000.0 * 2f64.powf(-1.5)).abs() < 1e-6);
        assert!(*c.last().unwrap() <= 2000.0 + 1e-6);
        for w in c.windows(2) { assert!((cam(w[1]) - cam(w[0]) - 0.5).abs() < 1e-9); }
    }

    /// Calibration against the «5 полос» table (16.11.2026, L = 105 dB SPL),
    /// +6.0 dB: their envelope was |h ⊗ g| = A/2 (confirmed and corrected
    /// there the same day). Their LR4/LR8 are the textbook ones = PhaseForge
    /// LR order 2/4 (24/48 dB/oct).
    #[test]
    fn matches_the_reference_table() {
        use crate::fir::section_ring::section_ring;
        use crate::target::{FilterConfig, FilterType};
        let c = |ft: FilterType, order: u8, fc: f64, shape: Option<f64>| FilterConfig { filter_type: ft, order, freq_hz: fc,
            shape, linear_phase: true, q: None, subsonic_protect: None, complementary: false, nested: Vec::new() };
        // (name, config, [M1, M2, M5] at 105 dB; NaN = not given / < −60)
        let n = f64::NAN;
        let cases = [
            ("LR24 100", c(FilterType::LinkwitzRiley, 2, 100.0, None), [n, 8.5, -2.3]),
            ("LR24 300", c(FilterType::LinkwitzRiley, 2, 300.0, None), [-1.4, -12.2, n]),
            ("LR24 1000", c(FilterType::LinkwitzRiley, 2, 1000.0, None), [-37.0, n, n]),
            ("LR48 100", c(FilterType::LinkwitzRiley, 4, 100.0, None), [n, 21.3, 15.0]),
            ("LR48 300", c(FilterType::LinkwitzRiley, 4, 300.0, None), [16.9, 10.6, -8.2]),
            ("LR48 1000", c(FilterType::LinkwitzRiley, 4, 1000.0, None), [-6.4, -27.0, n]),
            ("Gauss M1.7 400", c(FilterType::Gaussian, 1, 400.0, Some(1.7)), [0.0, -20.0, n]),
        ];
        let mut fails = Vec::new();
        for (name, cfg, want) in cases {
            for lp in [true, false] {
                let r = section_ring(&cfg, lp, 48_000.0).unwrap();
                let a = r.audibility.unwrap();
                let got: Vec<f64> = a.rel_db.iter().map(|x| x + 105.0).collect();
                println!("{name} {}: M1 {:.1} M2 {:.1} M5 {:.1} @ {:.0} Hz (want {want:?})", if lp { "LP" } else { "HP" }, got[0], got[1], got[2], a.channel_hz);
                for i in 0..3 {
                    if want[i].is_finite() && (got[i] - want[i] - 6.0).abs() > 1.0 { fails.push(format!("{name} lp={lp} M{} {:.1} vs {}", TAUS_MS[i], got[i], want[i])); }
                }
            }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }
}

#[cfg(test)]
mod timing {
    #[test]
    #[ignore]
    fn time_worst_cases() {
        use crate::fir::section_ring::section_ring;
        use crate::target::{FilterConfig, FilterType};
        for (fc, sr, ord) in [(20.0, 384_000.0, 4u8), (3000.0, 384_000.0, 4), (100.0, 48_000.0, 4), (20.0, 48_000.0, 8)] {
            let c = FilterConfig { filter_type: FilterType::LinkwitzRiley, order: ord, freq_hz: fc, shape: None, linear_phase: true,
                q: None, subsonic_protect: None, complementary: false, nested: Vec::new() };
            let t = std::time::Instant::now();
            let r = section_ring(&c, true, sr).unwrap();
            println!("fc {fc} sr {sr} ord {ord}: {:?}, {:?}", t.elapsed(), r.audibility.map(|a| a.rel_db));
        }
    }
}
