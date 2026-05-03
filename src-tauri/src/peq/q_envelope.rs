//! Frequency-dependent Q ceilings for PEQ optimization (b137).
//!
//! High Q on bass is acceptable (room modes); high Q on treble causes
//! audible ringing and usually over-fits measurement artefacts. Both
//! envelopes interpolate linearly in log2(f) between two anchor freqs.

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
    fn warn_midpoint() {
        let q = q_warn_at(632.456);
        assert!((q - 5.5).abs() < 0.01, "warn at 632 Hz = {}", q);
    }
}
