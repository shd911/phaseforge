//! b141.59: Σ IR/Step with alignment delays of a few ms. The delay ramp
//! baked into the wrapped phase of the 1024-point Σ IR log grid aliased at
//! HF (> 180° per step near 20–40 kHz at 3 ms) and produced a 25 % pre-
//! response at 96 kHz. `compute_sum_impulse_response` applies the delay per
//! linear FFT bin; a band's own time of flight inside its phase is removed
//! before interpolation and restored per bin as well.

use phaseforge_lib::dsp::impulse::{compute_impulse_response, compute_sum_impulse_response, SpectrumPart};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};

/// The Σ IR grid: 1024 log points 5 Hz – min(40 k, 0.95·Nyquist) + 32-point tail.
fn grid(sr: f64) -> Vec<f64> {
    let fmax = (40000.0f64).min(sr / 2.0 * 0.95);
    let mut f: Vec<f64> = (0..1024).map(|i| 5.0 * (fmax / 5.0f64).powf(i as f64 / 1023.0)).collect();
    let end = sr / 2.0 * 0.999;
    for i in 1..=32 { f.push(fmax * (end / fmax).powf(i as f64 / 32.0)); }
    f
}

fn pre_peak(h: &[f64], sr: f64) -> f64 {
    let pk = h.iter().enumerate().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).unwrap().0;
    let (a, b) = (pk.saturating_sub((0.004 * sr) as usize), pk.saturating_sub((0.001 * sr) as usize));
    h[a..b].iter().fold(0.0, |m: f64, &v| m.max(v.abs()))
}

#[test]
fn delayed_band_has_no_extra_pre_response() {
    for sr in [48_000.0, 96_000.0] {
        let f = grid(sr);
        let t = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: Some(FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 2500.0,
                shape: None, linear_phase: true, q: None, subsonic_protect: None }),
            low_pass: None, low_shelf: None, high_shelf: None };
        let r = target::evaluate(&t, &f);
        let n0 = f.len() - 32;
        let mag: Vec<f64> = r.magnitude.iter().enumerate().map(|(i, &m)| if i >= n0 { -150.0 } else { m }).collect();
        let ph: Vec<f64> = (0..f.len()).map(|i| if i >= n0 { 0.0 } else { r.phase[i] }).collect();
        let wrap = |d: f64| ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        let tau = 0.00307;
        let reference = pre_peak(&compute_impulse_response(&f, &mag, &ph, sr).impulse, sr);
        let explicit = pre_peak(&compute_sum_impulse_response(&f,
            &[SpectrumPart { magnitude: &mag, phase: &ph, delay_s: tau, sign: 1.0 }], sr).impulse, sr);
        let baked: Vec<f64> = ph.iter().zip(&f).enumerate()
            .map(|(i, (&p, &q))| if i >= n0 { 0.0 } else { wrap(p - 360.0 * q * tau) }).collect();
        let in_phase = pre_peak(&compute_impulse_response(&f, &mag, &baked, sr).impulse, sr);
        println!("sr={sr}: undelayed {reference:.3} %, delay param {explicit:.3} %, delay in phase {in_phase:.3} %");
        assert!(explicit < reference + 0.2, "delay param: {explicit:.3} % vs {reference:.3} %");
        assert!(in_phase < reference + 0.2, "delay in phase: {in_phase:.3} % vs {reference:.3} %");
    }
}
