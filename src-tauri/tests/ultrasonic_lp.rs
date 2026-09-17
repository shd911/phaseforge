//! b141.40 acceptance: the zero-phase ultrasonic low-pass (fir/ultrasonic.rs).
//!
//! Production shape: the FIR target runs to 0.95·Nyquist with a noise-floor
//! tail (evaluate.ts), a tweeter band LR 24 dB/oct @ 2039 Hz, 65536 taps,
//! Blackman, noise floor −150 dB. Numbers the design was chosen on, at
//! 352.8 kHz: 0.007 dB loss at 20 kHz, 6.6 kHz ultrasonic noise bandwidth
//! above 24 kHz, pre-ring 0.49 ms (the old 40 kHz wall read 2.24 ms).

use num_complex::Complex64;
use phaseforge_lib::dsp::fft::FftEngine;
use phaseforge_lib::fir::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use phaseforge_lib::fir::ultrasonic::ULTRASONIC_LP_HZ;
use phaseforge_lib::fir::{generate_model_fir, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::target::{
    self, FilterConfig as TargetFilterConfig, FilterType as TargetFilterType, TargetCurve,
};

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


const TAPS: usize = 65_536;

fn cfg(sr: f64, linear: bool) -> FirConfig {
    let mut c = fir_config_taps(linear, None, TAPS);
    c.sample_rate = sr;
    c.window = WindowType::Blackman;
    c.noise_floor_db = -150.0;
    c.max_boost_db = 24.0;
    c.phase_mode = PhaseMode::Composite;
    c
}

/// Mirrors evaluate.ts: log grid to 0.95·Nyquist + noise-floor tail.
fn fir_grid(sr: f64) -> Vec<f64> {
    let fmax = sr / 2.0 * 0.95;
    let n = 512.max((512.0 * (fmax / 5.0).ln() / (40_000.0f64 / 5.0).ln()).round() as usize);
    (0..n).map(|i| 5.0 * (fmax / 5.0f64).powf(i as f64 / (n - 1) as f64)).collect()
}

fn with_tail(freq: &[f64], mag: &[f64], sr: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut f, mut m, mut p) = (freq.to_vec(), mag.to_vec(), vec![0.0; freq.len()]);
    let fhi = *freq.last().unwrap();
    let fend = sr / 2.0 * 0.999;
    for i in 1..=32 {
        let t = i as f64 / 32.0;
        f.push(fhi * (fend / fhi).powf(t)); m.push(-150.0); p.push(0.0);
    }
    (f, m, p)
}

fn tweeter(linear: bool) -> TargetFilterConfig {
    let mut hp = lr4(2039.0);
    hp.order = 2;
    hp.linear_phase = linear;
    hp
}

fn spectrum(h: &[f64]) -> Vec<Complex64> {
    let mut b: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    FftEngine::new().fft_forward(&mut b);
    b
}

struct Resp { s: Vec<Complex64>, n: usize, sr: f64 }
impl Resp {
    fn new(h: &[f64], sr: f64) -> Self { Resp { s: spectrum(h), n: h.len(), sr } }
    fn db(&self, f: f64) -> f64 { 20.0 * self.s[(f * self.n as f64 / self.sr).round() as usize].norm().log10() }
    /// ∫|H/H(5k)|² df above 24 kHz, kHz.
    fn ultrasonic_bw_khz(&self) -> f64 {
        let g5 = 10f64.powf(self.db(5000.0) / 20.0);
        let df = self.sr / self.n as f64;
        (0..=self.n / 2).filter(|&k| k as f64 * df > 24_000.0)
            .map(|k| (self.s[k].norm() / g5).powi(2) * df).sum::<f64>() / 1000.0
    }
}

fn linear_tweeter_fir(sr: f64) -> phaseforge_lib::fir::FirModelResult {
    let grid = fir_grid(sr);
    let t = target_with(Some(tweeter(true)), None);
    let r = target::evaluate(&t, &grid);
    let (f, m, p) = with_tail(&grid, &r.magnitude, sr);
    generate_model_fir(&f, &m, &[], &p, &cfg(sr, true)).expect("cepstral")
}

fn pre_ring_ms(h: &[f64], sr: f64, db: f64) -> f64 {
    let p = peak_index(h);
    let th = h[p].abs() * 10f64.powf(db / 20.0);
    let first = (0..p).find(|&i| h[i].abs() > th).unwrap_or(p);
    (p - first) as f64 / sr * 1000.0
}

#[test]
fn applies_at_high_rates_only() {
    for sr in [88_200.0, 96_000.0, 192_000.0, 352_800.0] {
        assert_eq!(linear_tweeter_fir(sr).ultrasonic_lp_hz, Some(ULTRASONIC_LP_HZ), "sr {sr}");
    }
    for sr in [44_100.0, 48_000.0] {
        assert_eq!(linear_tweeter_fir(sr).ultrasonic_lp_hz, None, "sr {sr}: must stay untouched");
    }
}

#[test]
fn audio_band_untouched_ultrasound_limited() {
    for sr in [96_000.0, 192_000.0, 352_800.0] {
        let r = linear_tweeter_fir(sr);
        let resp = Resp::new(&r.impulse, sr);
        let tgt = target::evaluate(&target_with(Some(tweeter(true)), None), &[5000.0, 20_000.0]).magnitude;
        let loss20 = (resp.db(20_000.0) - resp.db(5000.0)) - (tgt[1] - tgt[0]);
        assert!(loss20.abs() < 0.03, "sr {sr}: loss at 20 kHz {loss20:.3} dB");
        // BW8 @ 30 kHz: −26.6 dB at 44 kHz, −48 dB at 60 kHz.
        let (fx, floor) = if sr / 2.0 > 60_000.0 { (60_000.0, -40.0) } else { (44_000.0, -20.0) };
        let ux = resp.db(fx) - resp.db(5000.0);
        assert!(ux < floor, "sr {sr}: ultrasound not limited at {fx} Hz ({ux:.1} dB)");
        let bw = resp.ultrasonic_bw_khz();
        assert!(bw < 8.0, "sr {sr}: ultrasonic noise bandwidth {bw:.1} kHz (wall was 17)");
    }
}

#[test]
fn wall_ringing_gone_from_pre_ring() {
    let r = linear_tweeter_fir(352_800.0);
    let ms = pre_ring_ms(&r.impulse, 352_800.0, -60.0);
    assert!(ms < 0.7, "pre-ring at −60 dB {ms:.2} ms (40 kHz wall read 2.24)");
}

#[test]
fn iir_route_limited_without_delay() {
    let sr = 192_000.0;
    let freq = fir_grid(sr);
    let c = cfg(sr, false);
    let hp = tweeter(false);
    let mut lp = lr4(2039.0);
    lp.order = 2;
    let tw = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq, hp: Some(&hp), lp: None, low_shelf: None, high_shelf: None, peq: &[], config: &c,
    }).expect("iir tweeter");
    let wf = generate_min_phase_fir_iir(&IirPathInput {
        freq: &freq, hp: None, lp: Some(&lp), low_shelf: None, high_shelf: None, peq: &[], config: &c,
    }).expect("iir woofer");
    assert_eq!(tw.ultrasonic_lp_hz, Some(ULTRASONIC_LP_HZ));
    // Zero phase: the band's latency and peak stay where the delay put them.
    assert_eq!(tw.wav_delay_samples, TAPS / 2);
    let p = peak_index(&tw.impulse);
    assert!(p >= TAPS / 2 - 1 && p <= TAPS / 2 + 32, "peak moved to {p}");
    assert!(Resp::new(&tw.impulse, sr).ultrasonic_bw_khz() < 8.0);
    // A two-way LR sum stays flat — both bands see the same zero-phase filter.
    let sum: Vec<f64> = tw.impulse.iter().zip(wf.impulse.iter()).map(|(a, b)| a + b).collect();
    let resp = Resp::new(&sum, sr);
    let g = resp.db(500.0);
    for f in [2039.0, 5000.0, 10_000.0] {
        let d = resp.db(f) - g;
        assert!(d.abs() < 0.02, "two-way sum at {f} Hz: {d:.3} dB");
    }
}
