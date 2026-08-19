//! b141.14 — unified WAV delay convention acceptance.
//!
//! Every FIR route pads its impulse with N/2 leading zeros so that bands
//! exported from different pipelines share a latency and stay aligned in a
//! convolver:
//!
//!   - linear-phase cepstral   — symmetric FIR, centred by construction;
//!   - IIR-analytical          — adaptive shift (b141.8);
//!   - cepstral min-phase      — adaptive shift (b141.14).
//!
//! b141.19 (audit) states the invariant precisely: what must match across
//! bands is the DELAY (leading zeros), not the peak position. A min-phase
//! band's peak sits past N/2 by its own rise time — an LF section really
//! does reach maximum later than an HF one, and forcing those peaks onto a
//! common index misaligns the bands. Measured on an LR4 500 Hz two-way:
//! equal padding sums to 0.00 dB across the crossover, peak-aligning digs
//! -1.15 dB at 400 Hz (`two_way_sum_stays_flat` pins this).
//!
//! The tail cap stands — content correctness wins over latency uniformity,
//! so a long LF tail on few taps gets less than N/2. That case is asserted
//! explicitly below rather than left out of the fixtures, and the applied
//! delay is reported as `wav_delay_samples` instead of promised away.

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
    fir_config_taps(linear_main, subsonic, TAPS)
}

fn fir_config_taps(linear_main: bool, subsonic: Option<f64>, taps: usize) -> FirConfig {
    FirConfig {
        taps,
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
    // The peak lands at N/2 plus the band's own rise time (zero for a
    // symmetric linear-phase FIR, a few samples for a min-phase cascade).
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
        low_shelf: None, high_shelf: None, peq: &[],
        config: &cfg,
    })
    .expect("iir run");
    assert_centered(&r.impulse, "iir_lr4_hp_80");
}


/// b141.19 (audit): the invariant the convention actually protects. Two
/// min-phase bands of one LR4 crossover, exported separately, must sum back
/// to the flat crossover response — which happens only if both carry the same
/// delay. Aiming each band's PEAK at N/2 instead (tried during this audit,
/// and what a peak-centred reading of the convention would demand) shifts the
/// low-pass forward by its rise time and digs -1.15 dB at 400 Hz.
#[test]
fn two_way_sum_stays_flat() {
    let cfg = fir_config(false, None);
    let freq = log_freq_grid();
    let (hp, lp) = (lr4(500.0), lr4(500.0));
    let woofer = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq, hp: None, lp: Some(&lp),
        low_shelf: None, high_shelf: None, peq: &[], config: &cfg,
    }).expect("woofer");
    let tweeter = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq, hp: Some(&hp), lp: None,
        low_shelf: None, high_shelf: None, peq: &[], config: &cfg,
    }).expect("tweeter");

    assert_eq!(woofer.wav_delay_samples, tweeter.wav_delay_samples,
        "bands of one crossover must share a delay");

    let sum: Vec<f64> = woofer.impulse.iter().zip(tweeter.impulse.iter())
        .map(|(a, b)| a + b).collect();
    for f in [100.0, 250.0, 400.0, 500.0, 630.0, 1000.0, 3000.0] {
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (n, &v) in sum.iter().enumerate() {
            let w = -2.0 * std::f64::consts::PI * f * n as f64 / 48_000.0;
            re += v * w.cos();
            im += v * w.sin();
        }
        let db = 20.0 * (re * re + im * im).sqrt().log10();
        assert!(db.abs() < 0.25, "two-way sum at {f} Hz: {db:.2} dB, expected flat");
    }
}

/// Every route must report the delay it applied — the number is the band's
/// latency, and the export layer warns from it.
#[test]
fn wav_delay_matches_the_leading_zeros() {
    let cfg = fir_config(false, None);
    let freq = log_freq_grid();
    let hp = lr4(80.0);
    let r = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq, hp: Some(&hp), lp: None,
        low_shelf: None, high_shelf: None, peq: &[], config: &cfg,
    }).expect("iir run");
    let lead = r.impulse.iter().position(|&v| v != 0.0).unwrap_or(0);
    assert_eq!(r.wav_delay_samples, lead, "iir delay vs leading zeros");
    assert_eq!(r.wav_delay_samples, TAPS / 2, "16384 taps @ LR4 80: tail fits, full N/2");

    let t = target_with(Some(gaussian(632.0, false)), None);
    let resp = target::evaluate(&t, &freq);
    let c = generate_model_fir(&freq, &resp.magnitude, &[], &resp.phase, &cfg).expect("cepstral");
    let lead = c.impulse.iter().position(|&v| v != 0.0).unwrap_or(0);
    assert_eq!(c.wav_delay_samples, lead, "cepstral delay vs leading zeros");
}

/// The documented exception, pinned rather than avoided: at 4096 taps an LF
/// high-pass still has tail above -100 dB at the last sample, so the shift
/// shrinks and the band ends up with less latency than a centred one.
/// Dropping that tail would corrupt the response, so the contract is "delay
/// as far as the tail allows, and report what was applied" — not "always N/2".
#[test]
fn short_taps_lf_tail_falls_short_of_center_and_says_so() {
    let taps = 4096_usize;
    let cfg = fir_config_taps(false, None, taps);
    let freq = log_freq_grid();
    let hp = lr4(20.0);
    let r = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq, hp: Some(&hp), lp: None,
        low_shelf: None, high_shelf: None, peq: &[], config: &cfg,
    }).expect("iir run");
    let lead = r.impulse.iter().position(|&v| v != 0.0).unwrap_or(0);
    assert_eq!(r.wav_delay_samples, lead, "reported delay must match the impulse");
    assert!(r.wav_delay_samples < taps / 2,
        "expected a tail-limited short delay, got {} at N/2 = {}",
        r.wav_delay_samples, taps / 2);
}
