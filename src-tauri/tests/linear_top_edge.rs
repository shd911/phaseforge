// b141.112: a linear-phase band whose passband reaches the top (a tweeter HP)
// is continued to Nyquist by slope, like the minimum-phase main. Before, the
// FIR request dropped to the −150 dB noise floor at 0.95·Nyquist: a brick
// wall in the passband, zero-phase, so it rang symmetrically — an LR4 2 kHz
// linear HP at 44.1/48 kHz read 7.3–8.0 ms of pre-ring to −60 dB and
// 230–250 ms to −100 dB, −62 dB 10–100 ms ahead of the peak.
//
// Production export settings: 65536 taps, Blackman (the app default) and
// Rectangular (no window to hide the tail), 3 refinement passes.
use phaseforge_lib::fir::band::{generate_band_fir, BandFirSettings};
use phaseforge_lib::fir::WindowType;
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};
use rustfft::{num_complex::Complex64, FftPlanner};

fn db(x: f64) -> f64 { 20.0 * x.max(1e-30).log10() }

fn lr4(fc: f64) -> FilterConfig {
    FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: fc, shape: None, linear_phase: true,
        q: None, subsonic_protect: None, complementary: false }
}

/// (impulse at passband level, peak index)
fn band(hp: Option<FilterConfig>, lp: Option<FilterConfig>, sr: f64, window: WindowType) -> (Vec<f64>, usize) {
    let t = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None };
    let s = BandFirSettings { taps: 65_536, sample_rate: sr, window, max_boost_db: 24.0, noise_floor_db: -150.0,
        iterations: 3, freq_weighting: true, narrowband_limit: true, nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 };
    let r = generate_band_fir(&t, &[], &s, false).unwrap();
    let g = 10f64.powf(r.fir.norm_db / 20.0);
    let h: Vec<f64> = r.fir.impulse.iter().map(|v| v * g).collect();
    let p = h.iter().enumerate().fold((0, 0.0f64), |(i, m), (k, v)| if v.abs() > m { (k, v.abs()) } else { (i, m) }).0;
    (h, p)
}

#[test]
fn linear_tweeter_hp_has_no_brick_wall_pre_ring() {
    for sr in [44_100.0, 48_000.0] {
        for window in [WindowType::Blackman, WindowType::Rectangular] {
            let (h, p) = band(Some(lr4(2000.0)), None, sr, window.clone());
            let pk = h[p].abs();
            let ms = |k: usize| k as f64 / sr * 1000.0;
            let pre = |d: f64| ms(p - h.iter().position(|v| v.abs() > pk * 10f64.powf(-d / 20.0)).unwrap());
            let far = db(h[..p - (0.010 * sr) as usize].iter().fold(0.0f64, |m, v| m.max(v.abs())) / pk);
            let mut sp: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            FftPlanner::new().plan_fft_forward(h.len()).process(&mut sp);
            let at = |f: f64| db(sp[(f / sr * h.len() as f64).round() as usize].norm());
            let (p60, p100) = (pre(60.0), pre(100.0));
            println!("{sr} {window:?}: pre −60 {p60:.2} ms, −100 {p100:.2} ms, >10 ms ahead {far:.0} dB, \
                20 kHz {:.4} dB, 0.99·Nyq {:.3} dB", at(20_000.0), at(0.99 * sr / 2.0));
            // Its own LR4 pre-ring is 0.7–0.9 / 1.8 ms; the wall read 7.3–8.0 / 230–250 ms.
            assert!(p60 < 1.5 && p100 < 3.0, "{sr} {window:?}: pre-ring {p60:.2} / {p100:.2} ms");
            assert!(far < -120.0, "{sr} {window:?}: {far:.0} dB more than 10 ms ahead of the peak");
            // The top of the audio band and the bins above 0.95·Nyquist stay in the passband.
            assert!(at(20_000.0).abs() < 0.01 && at(0.99 * sr / 2.0).abs() < 0.05, "{sr} {window:?}: top edge");
        }
    }
}

/// A linear LP band never reaches the top: its tail is under the floor
/// either way, so the change must leave it alone.
#[test]
fn linear_lp_band_keeps_its_stopband() {
    for sr in [44_100.0, 48_000.0] {
        let (h, p) = band(None, Some(lr4(2000.0)), sr, WindowType::Blackman);
        let mut sp: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        FftPlanner::new().plan_fft_forward(h.len()).process(&mut sp);
        let top = db(sp[(0.99 * sr / 2.0 / sr * h.len() as f64).round() as usize].norm());
        let pk = h[p].abs();
        let p60 = (p - h.iter().position(|v| v.abs() > pk * 1e-3).unwrap()) as f64 / sr * 1000.0;
        assert!(top < -140.0 && p60 < 1.5, "{sr}: 0.99·Nyq {top:.0} dB, pre −60 {p60:.2} ms");
    }
}

/// b141.115: the phase above 0.95·Nyquist is held, not dropped to 0°. Under
/// a mixed main (HP min + LP lin) the model phase is taken verbatim, and the
/// 0° tail was a step: 0.19–0.38° off at 5 kHz..0.95·Nyquist; held, 0.01°.
#[test]
fn mixed_main_tail_holds_the_phase() {
    for sr in [44_100.0, 48_000.0] {
        let mut hp = lr4(3000.0);
        hp.linear_phase = false;
        let t = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: Some(hp), low_pass: Some(lr4(20_000.0)), low_shelf: None, high_shelf: None };
        let s = BandFirSettings { taps: 65_536, sample_rate: sr, window: WindowType::Blackman, max_boost_db: 24.0,
            noise_floor_db: -150.0, iterations: 3, freq_weighting: true, narrowband_limit: true,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 };
        let r = generate_band_fir(&t, &[], &s, false).unwrap();
        let dp = (0..r.freq.len()).filter(|&i| r.freq[i] > 5_000.0 && r.freq[i] < 0.95 * sr / 2.0
            && r.fir.realized_mag[i] + r.fir.norm_db > -40.0).fold(0.0f64, |m, i| m.max(r.dev_phase[i].abs()));
        assert!(dp < 0.05, "{sr}: {dp:.3}° off at 5 kHz..0.95·Nyquist");
    }
}

/// b141.115: a target still rising at the top edge (a high shelf) does not
/// keep rising above 0.95·Nyquist — the continuation is at most the tilt.
#[test]
fn rising_top_edge_does_not_lift_the_ultrasonic_band() {
    use phaseforge_lib::target::ShelfConfig;
    let sr = 44_100.0;
    let hs = ShelfConfig { freq_hz: 18_000.0, gain_db: 6.0, q: 0.707 };
    let t = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(lr4(2000.0)), low_pass: None, low_shelf: None, high_shelf: Some(hs) };
    let s = BandFirSettings { taps: 65_536, sample_rate: sr, window: WindowType::Blackman, max_boost_db: 24.0,
        noise_floor_db: -150.0, iterations: 3, freq_weighting: true, narrowband_limit: true,
        nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 };
    let r = generate_band_fir(&t, &[], &s, false).unwrap();
    let g = 10f64.powf(r.fir.norm_db / 20.0);
    let mut sp: Vec<Complex64> = r.fir.impulse.iter().map(|&v| Complex64::new(v * g, 0.0)).collect();
    let n = sp.len();
    FftPlanner::new().plan_fft_forward(n).process(&mut sp);
    let at = |f: f64| db(sp[(f / sr * n as f64).round() as usize].norm());
    let (edge, top) = (at(0.95 * sr / 2.0), at(0.995 * sr / 2.0));
    println!("HS 18 kHz +6 dB: 0.95·Nyq {edge:.3} dB, 0.995·Nyq {top:.3} dB");
    assert!(top - edge < 0.05, "rose {:.3} dB above the edge", top - edge);
}
