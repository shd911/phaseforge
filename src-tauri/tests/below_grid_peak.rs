// b141.133: the file peaks at 0 dB down to DC, not only down to the FIR grid's
// first point (5 Hz), and a curve still rising toward DC at 5 Hz saturates.
// Continued by its slope to 12 octaves under the grid, a tilt of −1 dB/oct
// and a low shelf at 8 Hz +12 dB ended at +12 dB over full scale at DC.
use phaseforge_lib::fir::band::{generate_band_fir, BandFirSettings};
use phaseforge_lib::fir::WindowType;
use phaseforge_lib::peq::{PeqBand, PeqFilterType};
use phaseforge_lib::target::{FilterConfig, FilterType, ShelfConfig, TargetCurve};
use std::f64::consts::PI;

fn lp(linear: bool) -> Option<FilterConfig> {
    Some(FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 4, freq_hz: 80.0, shape: None,
        linear_phase: linear, q: None, subsonic_protect: None, complementary: false, nested: Vec::new() })
}

fn shelf(freq_hz: f64, gain_db: f64) -> Option<ShelfConfig> {
    Some(ShelfConfig { freq_hz, gain_db, q: 0.707 })
}

fn lt(f0: f64, gain: f64) -> Vec<PeqBand> {
    vec![PeqBand { freq_hz: f0, gain_db: gain, q: 0.707, enabled: true, filter_type: PeqFilterType::LinkwitzTransform }]
}

fn target(linear: bool, tilt: f64, low_shelf: Option<ShelfConfig>) -> TargetCurve {
    TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: tilt, tilt_ref_freq: 1000.0,
        high_pass: None, low_pass: lp(linear), low_shelf, high_shelf: None }
}

fn settings(sample_rate: f64, taps: usize) -> BandFirSettings {
    BandFirSettings { taps, sample_rate, window: WindowType::Blackman, max_boost_db: 24.0, noise_floor_db: -150.0,
        iterations: 3, freq_weighting: true, narrowband_limit: true, nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 }
}

/// The file's level at `f`, read from the impulse itself.
fn db_at(h: &[f64], f: f64, sr: f64) -> f64 {
    let (s, c) = (2.0 * PI * f / sr).sin_cos();
    let (mut re, mut im, mut cr, mut ci) = (0.0, 0.0, 1.0, 0.0);
    for &v in h {
        re += v * cr;
        im += v * ci;
        (cr, ci) = (cr * c - ci * s, cr * s + ci * c);
    }
    10.0 * (re * re + im * im).max(1e-40).log10()
}

const RATES: [(f64, usize); 2] = [(48_000.0, 65_536), (352_800.0, 262_144)];

#[test]
fn the_file_does_not_rise_above_0_db_below_the_grid() {
    let cases: Vec<(&str, TargetCurve, Vec<PeqBand>)> = vec![
        ("tilt −1 lin", target(true, -1.0, None), vec![]),
        ("tilt −1 min", target(false, -1.0, None), vec![]),
        ("LS 8 +12 lin", target(true, 0.0, shelf(8.0, 12.0)), vec![]),
        ("LS 8 +12 min", target(false, 0.0, shelf(8.0, 12.0)), vec![]),
        ("LS 20 +12 lin", target(true, 0.0, shelf(20.0, 12.0)), vec![]),
        ("LS 20 +12 min", target(false, 0.0, shelf(20.0, 12.0)), vec![]),
        ("LT 12 +12 lin", target(true, 0.0, None), lt(12.0, 12.0)),
        ("LT 12 +12 min", target(false, 0.0, None), lt(12.0, 12.0)),
    ];
    for (sr, taps) in RATES {
        for (name, t, peq) in &cases {
            let r = generate_band_fir(t, peq, &settings(sr, taps), false).unwrap();
            let x = &r.fir.impulse;
            let top = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0].iter()
                .map(|&f| db_at(x, f, sr)).fold(f64::NEG_INFINITY, f64::max);
            println!("{sr} {name:14} {:?} norm {:+.2}: max below 5 Hz {top:+.2} dB, DC {:+.2}, 60 Hz {:+.2}",
                r.route, r.fir.norm_db, db_at(x, 0.0, sr), db_at(x, 60.0, sr));
            assert!(top < 0.1, "{sr} {name}: {top:+.2} dB over full scale below the grid");
        }
    }
}

/// The lift the shelf asked for is what the file gets: not 10 dB more
/// (continued slope), and not lost either when the corner is above the grid.
#[test]
fn a_shelf_keeps_its_own_lift_at_dc() {
    for (sr, taps) in RATES {
        for linear in [true, false] {
            for (corner, gain) in [(30.0, 12.0), (20.0, 12.0), (8.0, 12.0)] {
                let plain = generate_band_fir(&target(linear, 0.0, None), &[], &settings(sr, taps), false).unwrap();
                let r = generate_band_fir(&target(linear, 0.0, shelf(corner, gain)), &[], &settings(sr, taps), false).unwrap();
                let lift = (db_at(&r.fir.impulse, 0.0, sr) - db_at(&r.fir.impulse, 60.0, sr))
                    - (db_at(&plain.fir.impulse, 0.0, sr) - db_at(&plain.fir.impulse, 60.0, sr));
                // 60 Hz still carries the shelf's own residual for high corners.
                let residual = if corner >= 30.0 { 0.6 } else { 0.2 };
                println!("{sr} lin={linear} LS {corner} +{gain}: lift DC−60 Hz {lift:.2} dB");
                // The saturation is an estimate from the slope at 5 Hz: exact
                // enough for a corner in the working range, +1 dB at 8 Hz.
                let tol = if corner >= 20.0 { 0.6 } else { 1.5 };
                assert!((lift + residual - gain).abs() < tol, "{sr} lin={linear} LS {corner}: lift {lift:.2} dB, asked {gain}");
            }
        }
    }
}

/// A falling curve still continues by its slope (b141.52): a Butterworth HP
/// reaches DC far below the passband, no plateau under it.
#[test]
fn a_falling_curve_still_continues_by_slope() {
    let hp = FilterConfig { filter_type: FilterType::Gaussian, order: 4, freq_hz: 40.0, shape: Some(1.0),
        linear_phase: true, q: None, subsonic_protect: None, complementary: false, nested: Vec::new() };
    let t = TargetCurve { high_pass: Some(hp), ..target(true, 0.0, None) };
    let (sr, taps) = RATES[0];
    let r = generate_band_fir(&t, &[], &settings(sr, taps), false).unwrap();
    let (dc, at_5) = (db_at(&r.fir.impulse, 0.0, sr), db_at(&r.fir.impulse, 5.0, sr));
    println!("HP: 5 Hz {at_5:+.1} dB, DC {dc:+.1} dB");
    assert!(dc < at_5 + 0.5, "DC {dc:+.1} dB above 5 Hz {at_5:+.1} dB");
}
