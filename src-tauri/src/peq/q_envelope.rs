//! Frequency-dependent Q ceilings for PEQ optimization (b137).
//!
//! High Q on bass is acceptable (room modes); high Q on treble causes
//! audible ringing and usually over-fits measurement artefacts. Both
//! envelopes interpolate linearly in log2(f) between two anchor freqs.

use super::types::PeqFilterType;

const F_LO: f64 = 200.0;
const F_HI: f64 = 2000.0;

/// Hard ceiling enforced by the optimizer.
pub fn q_cap_at(freq_hz: f64) -> f64 {
    const Q_LO: f64 = 12.0;
    const Q_HI: f64 = 4.0;
    if freq_hz <= F_LO { return Q_LO; }
    if freq_hz >= F_HI { return Q_HI; }
    let t = (freq_hz.log2() - F_LO.log2()) / (F_HI.log2() - F_LO.log2());
    Q_LO - (Q_LO - Q_HI) * t
}

/// Soft warning threshold; UI flags bands above this. Always < q_cap_at.
pub fn q_warn_at(freq_hz: f64) -> f64 {
    const Q_LO: f64 = 8.0;
    const Q_HI: f64 = 3.0;
    if freq_hz <= F_LO { return Q_LO; }
    if freq_hz >= F_HI { return Q_HI; }
    let t = (freq_hz.log2() - F_LO.log2()) / (F_HI.log2() - F_LO.log2());
    Q_LO - (Q_LO - Q_HI) * t
}

/// b141.95: what the optimizer may do with Q at a given gain. Boosts stay
/// wide (Q ≤ 2): a narrow boost fills a dip that is usually position- or
/// reflection-made and rings for it (+1.9 dB Q8 at 91 Hz on a real woofer:
/// +43 ms of decay at −40 dB). Narrow filters only cut a peak of 3 dB and
/// more — a real resonance, where the cut also shortens the ringing (−1.1 dB
/// Q8 at a 67 Hz mode: −22 ms). Manual filters are not limited.
pub const Q_MAX_WIDE: f64 = 2.0;
pub const NARROW_CUT_MIN_DB: f64 = 3.0;

pub fn q_cap_for_gain(gain_db: f64) -> f64 {
    if gain_db > -NARROW_CUT_MIN_DB { Q_MAX_WIDE } else { f64::INFINITY }
}

/// b141.94: how long a PEQ filter rings (ms) until its resonance is
/// `budget_db` below the signal — the Rust twin of `peqRingCost`
/// (lib/peq-quality.ts, the «мс» column). The filter is 1 + a resonant term
/// of peak size |A² − 1|; its envelope decays at the slowest pole radius.
pub fn ring_ms(freq_hz: f64, gain_db: f64, q: f64, ftype: &PeqFilterType, sample_rate: f64, budget_db: f64) -> f64 {
    let a = 10f64.powf(gain_db / 40.0);
    let c = (a * a - 1.0).abs();
    if c < 1e-6 { return 0.0; }
    let w0 = 2.0 * std::f64::consts::PI * freq_hz / sample_rate;
    let (cw, al) = (w0.cos(), w0.sin() / (2.0 * q));
    let s = 2.0 * a.sqrt() * al;
    let (a0, a1, a2) = match ftype {
        PeqFilterType::LowShelf => ((a + 1.0) + (a - 1.0) * cw + s, -2.0 * ((a - 1.0) + (a + 1.0) * cw), (a + 1.0) + (a - 1.0) * cw - s),
        PeqFilterType::HighShelf => ((a + 1.0) - (a - 1.0) * cw + s, 2.0 * ((a - 1.0) - (a + 1.0) * cw), (a + 1.0) - (a - 1.0) * cw - s),
        PeqFilterType::Peaking => (1.0 + al / a, -2.0 * cw, 1.0 - al / a),
    };
    let (a1, a2) = (a1 / a0, a2 / a0);
    let disc = a1 * a1 - 4.0 * a2;
    let r = if disc < 0.0 { a2.sqrt() } else {
        let sq = disc.sqrt();
        ((-a1 + sq) / 2.0).abs().max(((-a1 - sq) / 2.0).abs())
    };
    if !(r > 0.0 && r < 1.0) { return f64::INFINITY; }
    let db_per_s = -20.0 * r.log10() * sample_rate;
    (budget_db + 20.0 * c.log10()).max(0.0) / db_per_s * 1000.0
}

/// The largest Q in `[q_min, q]` whose ring to −100 dB stays within
/// `limit_ms` — `Ok(q)` — or, when none does, `Err` with the Q in that range
/// that rings least (the optimizer's own filters only — manual ones are
/// shown red instead).
///
/// b141.101 (audit 2026-10-23 1.7): the walk stops at `q_min` (the
/// optimizer's Q_MIN 0.5). It used to run down to 0.1 and return the
/// shortest-ringing Q even when that broke the limit — a 20 Hz +48.9 dB
/// boost left at Q 0.099 rang 447 ms against a 371.5 ms limit, below the
/// optimizer's own minimum and the UI's.
pub fn q_cap_for_ring(freq_hz: f64, gain_db: f64, q: f64, ftype: &PeqFilterType, sample_rate: f64, limit_ms: f64, q_min: f64) -> Result<f64, f64> {
    let ring = |q: f64| ring_ms(freq_hz, gain_db, q, ftype, sample_rate, 100.0);
    if ring(q) <= limit_ms { return Ok(q); }
    // Ring time is not monotone in Q: below ~0.5 the poles turn real and one of
    // them slows down again. Walk Q down (2 % steps) to the first fit, then
    // bisect the last step; with no fit, report the shortest-ringing Q seen.
    let (mut prev, mut best, mut best_ring) = (q, q, ring(q));
    let mut cur = q;
    while cur > q_min {
        cur = (cur * 0.98).max(q_min);
        let r = ring(cur);
        if r <= limit_ms {
            let (mut lo, mut hi) = (cur, prev);
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                if ring(mid) <= limit_ms { lo = mid; } else { hi = mid; }
            }
            return Ok(lo);
        }
        if r < best_ring { best = cur; best_ring = r; }
        prev = cur;
    }
    Err(best)
}

/// (gain, Q) of a filter that rings within `limit_ms`: Q is lowered first
/// (not below `q_min`); when no Q fits, the gain is reduced at the Q that
/// rings least (b141.101, audit 2026-10-23 1.7 — the limit is a guarantee,
/// not a best effort). Ring time falls to zero with the gain, so a fit
/// always exists.
pub fn fit_ring(freq_hz: f64, gain_db: f64, q: f64, ftype: &PeqFilterType, sample_rate: f64, limit_ms: f64, q_min: f64) -> (f64, f64) {
    let q_min = q_min.min(q);
    let q_fit = match q_cap_for_ring(freq_hz, gain_db, q, ftype, sample_rate, limit_ms, q_min) {
        Ok(q) => return (gain_db, q),
        Err(best) => best,
    };
    let ring = |g: f64| ring_ms(freq_hz, g, q_fit, ftype, sample_rate, 100.0);
    let (mut prev, mut cur) = (gain_db, gain_db);
    for _ in 0..400 {
        cur *= 0.98;
        if ring(cur) <= limit_ms {
            let (mut lo, mut hi) = (cur, prev); // lo fits, hi does not
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                if ring(mid) <= limit_ms { lo = mid; } else { hi = mid; }
            }
            return (lo, q_fit);
        }
        prev = cur;
    }
    (0.0, q_fit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_plateau_low() {
        assert_eq!(q_cap_at(50.0), 12.0);
        assert_eq!(q_cap_at(200.0), 12.0);
    }

    #[test]
    fn cap_plateau_high() {
        assert_eq!(q_cap_at(2000.0), 4.0);
        assert_eq!(q_cap_at(20000.0), 4.0);
    }

    #[test]
    fn cap_midpoint() {
        // Geometric midpoint of 200..2000 → t = 0.5, expect 12 - 4 = 8
        let q = q_cap_at(632.456);
        assert!((q - 8.0).abs() < 0.01, "cap at 632 Hz = {}", q);
    }

    #[test]
    fn warn_lower_than_cap() {
        for &f in &[100.0, 500.0, 1000.0, 5000.0] {
            assert!(q_warn_at(f) < q_cap_at(f), "warn>=cap at {f}");
        }
    }

    #[test]
    fn warn_plateaus() {
        assert_eq!(q_warn_at(50.0), 8.0);
        assert_eq!(q_warn_at(200.0), 8.0);
        assert_eq!(q_warn_at(2000.0), 3.0);
        assert_eq!(q_warn_at(20000.0), 3.0);
    }

    #[test]
    fn ring_matches_the_ui_column() {
        // lib/__tests__/peq-ring-cost.test.ts: +3.1 dB @ 22 Hz Q4 ≈ 419 ms to −60 dB
        let pk = PeqFilterType::Peaking;
        let t60 = ring_ms(22.0, 3.1, 4.0, &pk, 176_400.0, 60.0);
        assert!((t60 - 419.3).abs() < 1.0, "{t60}");
        let cap = q_cap_for_ring(22.0, 3.1, 4.0, &pk, 176_400.0, 371.5, 0.5).unwrap();
        assert!(cap < 4.0 && ring_ms(22.0, 3.1, cap, &pk, 176_400.0, 100.0) <= 371.5 + 1e-6);
        assert!((ring_ms(22.0, 3.1, cap, &pk, 176_400.0, 100.0) - 371.5).abs() < 1.0);
        assert_eq!(q_cap_for_ring(1000.0, 3.0, 4.0, &pk, 48_000.0, 300.0, 0.5), Ok(4.0));
    }

    /// b141.101 (audit 2026-10-23 1.7): no Q in [0.5, q] fits → the gain
    /// gives way; Q never goes below the floor. 20 Hz +48.9 dB at 176.4 kHz
    /// ended at Q 0.099 / 447 ms against 371.5; 20 Hz +6 dB at 48 kHz at
    /// Q 0.355 / 90 ms against 42.7.
    #[test]
    fn ring_fit_lowers_the_gain_when_no_q_fits() {
        let pk = PeqFilterType::Peaking;
        for (f, g, q, sr, lim) in [(20.0, 48.9, 2.0, 176_400.0, 371.5), (20.0, 6.0, 2.0, 48_000.0, 42.7),
                                   (26.0, -29.5, 0.84, 176_400.0, 371.5)] {
            assert!(q_cap_for_ring(f, g, q, &pk, sr, lim, 0.5).is_err());
            let (g2, q2) = fit_ring(f, g, q, &pk, sr, lim, 0.5);
            let r = ring_ms(f, g2, q2, &pk, sr, 100.0);
            assert!(q2 >= 0.5 && r <= lim + 1e-6 && g2.abs() < g.abs() && g2 * g > 0.0,
                "{f} Hz {g:+} dB → {g2:+.2} dB Q{q2:.3} {r:.1} ms (limit {lim})");
            assert!(r > lim - 1.0, "gain cut further than needed: {r:.1} ms");
        }
        // A filter that fits keeps its gain and Q.
        assert_eq!(fit_ring(1000.0, 3.0, 4.0, &pk, 48_000.0, 300.0, 0.5), (3.0, 4.0));
    }

    #[test]
    fn warn_midpoint() {
        let q = q_warn_at(632.456);
        assert!((q - 5.5).abs() < 0.01, "warn at 632 Hz = {}", q);
    }
}
