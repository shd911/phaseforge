//! b141.14 — unified WAV peak convention acceptance.
//!
//! Every FIR route must ship its impulse peak near N/2 so that bands
//! exported from different pipelines stay time-aligned in a convolver:
//!
//!   - linear-phase cepstral   — symmetric FIR, peak at exactly N/2;
//!   - IIR-analytical          — adaptive shift (b141.8), peak ≈ N/2;
//!   - cepstral min-phase      — adaptive shift (this change), peak ≈ N/2.
//!
//! The adaptive shift is `min(N/2, N-1-last_significant)` — content
//! correctness wins over centering, so for pathological LF tails the peak
//! may land short of N/2. The fixtures here use production-scale taps
//! (16384 @ 48 kHz) where the tail decays below -100 dB well before N/2.

use phaseforge_lib::fir::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use phaseforge_lib::fir::{generate_model_fir, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::target::{
    self, FilterConfig as TargetFilterConfig, FilterType as TargetFilterType, TargetCurve,
};

const TAPS: usize = 16_384;
/// IIR/cepstral raw impulses peak a few samples after t=0 (cascade rise
/// time); after the N/2 shift the peak lands a hair past center.
const TOL: usize = 32;

fn log_freq_grid() -> Vec<f64> {
    let (f_min, f_max, n) = (5.0_f64, 40_000.0_f64, 512);
    (0..n)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn fir_config(linear_main: bool, subsonic: Option<f64>) -> FirConfig {
    FirConfig {
        taps: TAPS,
        sample_rate: 48_000.0,
        max_boost_db: 18.0,
        noise_floor_db: -60.0,
        window: WindowType::Hann,
        phase_mode: PhaseMode::Composite,
        iterations: 3,
        freq_weighting: true,
        narrowband_limit: true,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
        gaussian_min_phase_filters: vec![],
        linear_phase_main: linear_main,
        subsonic_cutoff_hz: subsonic,
    }
}

fn target_with(hp: Option<TargetFilterConfig>, lp: Option<TargetFilterConfig>) -> TargetCurve {
    TargetCurve {
        reference_level_db: 0.0,
        tilt_db_per_octave: 0.0,
        tilt_ref_freq: 1000.0,
        high_pass: hp,
        low_shelf: None,
        high_shelf: None,
        low_pass: lp,
    }
}

fn gaussian(freq: f64, subsonic: bool) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::Gaussian,
        order: 4,
        freq_hz: freq,
        shape: Some(1.0),
        linear_phase: false,
        q: None,
        subsonic_protect: Some(subsonic),
    }
}

fn bessel(freq: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::Bessel,
        order: 4,
        freq_hz: freq,
        shape: None,
        linear_phase: false,
        q: None,
        subsonic_protect: None,
    }
}

fn lr4(freq: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::LinkwitzRiley,
        order: 4,
        freq_hz: freq,
        shape: None,
        linear_phase: false,
        q: None,
        subsonic_protect: None,
    }
}

fn peak_index(impulse: &[f64]) -> usize {
    impulse
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn run_cepstral(t: &TargetCurve, cfg: &FirConfig) -> Vec<f64> {
    let freq = log_freq_grid();
    let resp = target::evaluate(t, &freq);
    generate_model_fir(&freq, &resp.magnitude, &[], &resp.phase, cfg)
        .expect("cepstral run")
        .impulse
}

fn assert_centered(impulse: &[f64], label: &str) {
    let half = TAPS / 2;
    let peak = peak_index(impulse);
    assert!(
        peak >= half && peak <= half + TOL,
        "{label}: peak at {peak}, expected within [{half}, {}]",
        half + TOL,
    );
    // The shift must pad exact zeros in front — no wrapped tail content.
    let lead_max = impulse[..half / 2]
        .iter()
        .fold(0.0_f64, |a, &v| a.max(v.abs()));
    let peak_abs = impulse[peak].abs();
    assert!(
        lead_max <= peak_abs * 1e-4,
        "{label}: leading quarter carries {lead_max:.3e} vs peak {peak_abs:.3e}",
    );
}

#[test]
fn cepstral_min_phase_gaussian_peak_centered() {
    let cfg = fir_config(false, None);
    let t = target_with(Some(gaussian(632.0, false)), None);
    assert_centered(&run_cepstral(&t, &cfg), "gaussian_hp_632");
}

#[test]
fn cepstral_min_phase_subsonic_peak_centered() {
    let cfg = fir_config(false, Some(632.0 / 8.0));
    let t = target_with(Some(gaussian(632.0, true)), None);
    assert_centered(&run_cepstral(&t, &cfg), "gaussian_hp_632_subsonic");
}

#[test]
fn cepstral_min_phase_bessel_peak_centered() {
    let cfg = fir_config(false, None);
    let t = target_with(None, Some(bessel(500.0)));
    assert_centered(&run_cepstral(&t, &cfg), "bessel_lp_500");
}

#[test]
fn cepstral_linear_phase_peak_centered() {
    let cfg = fir_config(true, None);
    let t = target_with(Some(lr4(80.0)), Some(lr4(2000.0)));
    assert_centered(&run_cepstral(&t, &cfg), "lr4_bandpass_linear");
}

#[test]
fn iir_path_peak_centered() {
    let cfg = fir_config(false, None);
    let freq = log_freq_grid();
    let hp = lr4(80.0);
    let r = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq,
        hp: Some(&hp),
        lp: None,
        peq: &[],
        config: &cfg,
    })
    .expect("iir run");
    assert_centered(&r.impulse, "iir_lr4_hp_80");
}
