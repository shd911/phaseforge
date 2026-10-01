//! b141.51 (audit 2026-10-01 H3): the cepstral min-phase route must realise
//! the analytic phase of LR/BW sections. Composite took it from a Hilbert of
//! the magnitude clipped at the noise floor — −35° at an LR4 2 kHz corner and
//! a −2.46 dB dip in a two-way sum. The LR/BW sections now come from the
//! bilinear biquad cascade (exact digital phase), the Hilbert transform only
//! covers the residual.
//!
//! Production parameters: 48 kHz, 65536 taps, −150 dB floor, Blackman,
//! FIR grid 5 Hz – 0.95·Nyquist (512 points), as the frontend sends them.

use phaseforge_lib::dsp::fft::FftEngine;
use phaseforge_lib::fir::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use phaseforge_lib::fir::{generate_model_fir_with_sections, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use num_complex::Complex64;

const SR: f64 = 48_000.0;
const TAPS: usize = 65_536;

fn grid() -> Vec<f64> {
    let (a, b, n) = (5.0_f64, SR / 2.0 * 0.95, 512);
    (0..n).map(|i| a * (b / a).powf(i as f64 / (n - 1) as f64)).collect()
}

fn cfg(mode: PhaseMode) -> FirConfig {
    FirConfig {
        taps: TAPS, sample_rate: SR, max_boost_db: 24.0, noise_floor_db: -150.0,
        window: WindowType::Blackman, phase_mode: mode, iterations: 0,
        freq_weighting: false, narrowband_limit: false, nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0, linear_phase_main: false, subsonic_cutoff_hz: None,
    }
}

fn lr4(fc: f64) -> FilterConfig {
    FilterConfig {
        filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: fc, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn curve(hp: Option<FilterConfig>, lp: Option<FilterConfig>) -> TargetCurve {
    TargetCurve {
        reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None,
    }
}

/// Impulse → complex spectrum value at frequency f (direct DFT, no grid).
fn response_at(impulse: &[f64], f: f64) -> Complex64 {
    let w = -2.0 * std::f64::consts::PI * f / SR;
    impulse.iter().enumerate()
        .map(|(n, &x)| Complex64::from_polar(x, w * n as f64))
        .sum()
}

fn wav(t: &TargetCurve, mode: PhaseMode) -> (Vec<f64>, usize) {
    let f = grid();
    let r = target::evaluate(t, &f);
    let out = generate_model_fir_with_sections(
        &f, &r.magnitude, &[], &r.phase, &cfg(mode), t.high_pass.as_ref(), t.low_pass.as_ref(),
    ).unwrap();
    (out.impulse, out.wav_delay_samples)
}

fn wrap(d: f64) -> f64 { ((d + 180.0) % 360.0 + 360.0) % 360.0 - 180.0 }

#[test]
fn lr4_hp_phase_matches_the_iir_route() {
    // The IIR route is the reference digital system (REPhase-parity tested).
    let t = curve(Some(lr4(2000.0)), None);
    let (cep, dc) = wav(&t, PhaseMode::Composite);
    let f = grid();
    let iir = generate_min_phase_fir_iir(&IirPathInput {
        freq: &f, hp: t.high_pass.as_ref(), lp: None, low_shelf: None, high_shelf: None,
        peq: &[], config: &cfg(PhaseMode::Composite),
    }).unwrap();
    for &fq in &[1000.0, 2000.0, 2500.0, 5000.0, 10000.0] {
        let pc = response_at(&cep, fq).arg().to_degrees() + 360.0 * fq * dc as f64 / SR;
        let pi = response_at(&iir.impulse, fq).arg().to_degrees()
            + 360.0 * fq * iir.wav_delay_samples as f64 / SR;
        let err = wrap(pc - pi);
        let vs_analog = wrap(pc - target::evaluate(&t, &[fq]).phase[0]);
        println!("f={fq:>6}: cepstral−IIR {err:6.3}°, cepstral−analog {vs_analog:6.2}°");
        assert!(err.abs() < 0.5, "cepstral vs IIR {err:.3}° at {fq} Hz (Composite was −35° at fc)");
    }
}

#[test]
fn two_way_lr4_sum_is_flat_on_the_cepstral_route() {
    let (hp, dh) = wav(&curve(Some(lr4(2000.0)), None), PhaseMode::Composite);
    let (lp, dl) = wav(&curve(None, Some(lr4(2000.0))), PhaseMode::Composite);
    assert_eq!(dh, dl, "both bands must carry the same WAV delay");
    let sum: Vec<f64> = hp.iter().zip(&lp).map(|(a, b)| a + b).collect();
    for &f in &[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 4000.0, 8000.0] {
        let db = 20.0 * response_at(&sum, f).norm().log10();
        println!("Σ at {f:>6} Hz: {db:+.3} dB");
        assert!(db.abs() < 0.1, "two-way sum {db:+.3} dB at {f} Hz");
    }
    // FFT sanity: the summed impulse has no energy before the delay.
    let mut spec: Vec<Complex64> = sum.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    FftEngine::new().fft_forward(&mut spec);
    let pre: f64 = sum[..dh.saturating_sub(64)].iter().map(|v| v * v).sum();
    let tot: f64 = sum.iter().map(|v| v * v).sum();
    assert!(pre / tot < 1e-8, "pre-delay energy {:.2e}", pre / tot);
}

