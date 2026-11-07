// b141.141: the iterative refinement corrects the window's magnitude error —
// it must never move the phase of what the file actually plays.
//
// Invariant: refined (3 passes) vs unrefined FIR of the same request, phase
// difference within TOL_DEG everywhere the band is above −140 dB. Matrix over
// the HP/LP phase combinations (both min, both linear, and the two mixed),
// LP families, PEQ placed in band, deep in the stopband and under the −150 dB
// noise floor, at 48 kHz/64K and 176.4 kHz/128K.
//
// Found by a 6 kHz −16 dB cut under a Gaussian 750 Hz LP's floor: the
// refinement clamped the −166 dB request up to the floor, the correction's own
// Hilbert was an inverted cut — 8° at 1 kHz, 18° at 2 kHz, 94° at the knee.
use phaseforge_lib::fir::band::{generate_band_fir, BandFirSettings};
use phaseforge_lib::fir::WindowType;
use phaseforge_lib::peq::PeqBand;
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};

/// Growth of the phase error vs the model allowed above / below 100 Hz. At
/// the LF edge the refinement restores what the window cut off the LF PEQ
/// tails and trades up to ~2.3° of phase for it (176.4 kHz/128K, 26 Hz);
/// above it the window's error is tiny and nothing may move.
const TOL_DEG: f64 = 0.5;
const TOL_LF_DEG: f64 = 3.0;
const LF_EDGE_HZ: f64 = 100.0;
const FLOOR_VIEW_DB: f64 = -140.0;

fn settings(taps: usize, sr: f64, iterations: usize) -> BandFirSettings {
    BandFirSettings {
        taps, sample_rate: sr, window: WindowType::Blackman, max_boost_db: 24.0,
        noise_floor_db: -150.0, iterations, freq_weighting: true, narrowband_limit: true,
        nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
    }
}

fn section(filter_type: FilterType, order: u8, freq_hz: f64, linear_phase: bool) -> FilterConfig {
    let shape = if matches!(filter_type, FilterType::Gaussian) { Some(1.0) } else { None };
    FilterConfig { filter_type, order, freq_hz, shape, linear_phase, q: None,
        subsonic_protect: None, complementary: false, nested: Vec::new() }
}

fn peq(json: &str) -> Vec<PeqBand> { serde_json::from_str(json).unwrap() }

fn wrap(d: f64) -> f64 { d - 360.0 * (d / 360.0).round() }

/// Above FLOOR_VIEW_DB: the growth of the phase error vs the model
/// (|dev| refined − |dev| unrefined), worst above and below LF_EDGE_HZ —
/// (freq, deg) each.
fn phase_growth(target: &TargetCurve, peq: &[PeqBand], taps: usize, sr: f64) -> ((f64, f64), (f64, f64)) {
    let r0 = generate_band_fir(target, peq, &settings(taps, sr, 0), true).unwrap();
    let r3 = generate_band_fir(target, peq, &settings(taps, sr, 3), true).unwrap();
    let (mut hi, mut lo) = ((0.0, 0.0), (0.0, 0.0));
    for k in 0..r3.freq.len() {
        if r3.freq[k] < 20.0 || r3.fir.realized_mag[k] < FLOOR_VIEW_DB
            || r0.fir.realized_mag[k] < FLOOR_VIEW_DB { continue; }
        let g = wrap(r3.dev_phase[k]).abs() - wrap(r0.dev_phase[k]).abs();
        let w = if r3.freq[k] >= LF_EDGE_HZ { &mut hi } else { &mut lo };
        if g > w.1 { *w = (r3.freq[k], g); }
    }
    (hi, lo)
}

fn check(case: &str, target: &TargetCurve, peq: &[PeqBand], taps: usize, sr: f64, fails: &mut Vec<String>) {
    let ((fh, h), (fl, l)) = phase_growth(target, peq, taps, sr);
    println!("{case}: worse {h:.2}° at {fh:.0} Hz; LF edge {l:.2}° at {fl:.0} Hz");
    if h > TOL_DEG { fails.push(format!("{case}: {h:.1}° at {fh:.0} Hz")); }
    if l > TOL_LF_DEG { fails.push(format!("{case}: {l:.1}° at {fl:.0} Hz (LF edge)")); }
}

#[test]
fn refinement_never_moves_the_audible_phase() {
    let peqs = [
        ("no PEQ", peq("[]")),
        ("in band", peq(r#"[
            {"freq_hz": 24.0, "gain_db": 7.7, "q": 1.5, "enabled": true},
            {"freq_hz": 300.0, "gain_db": -4.0, "q": 3.0, "enabled": true}]"#)),
        ("stopband + under floor", peq(r#"[
            {"freq_hz": 24.0, "gain_db": 7.7, "q": 1.5, "enabled": true},
            {"freq_hz": 1094.7, "gain_db": -1.45, "q": 1.5, "enabled": true},
            {"freq_hz": 6000.0, "gain_db": -16.0, "q": 1.5, "enabled": true},
            {"freq_hz": 9000.0, "gain_db": 6.0, "q": 2.0, "enabled": true},
            {"freq_hz": 14000.0, "gain_db": -12.0, "q": 4.0, "enabled": true}]"#)),
    ];
    let families = [
        (FilterType::Gaussian, 4), (FilterType::LinkwitzRiley, 4),
        (FilterType::Butterworth, 4), (FilterType::Bessel, 4),
    ];
    let mut fails = Vec::new();
    let mut n = 0;
    for &(taps, sr) in &[(65536usize, 48_000.0), (131072, 176_400.0)] {
        for (fam, order) in families.iter() {
            for &(hp_lin, lp_lin) in &[(false, false), (true, true), (false, true), (true, false)] {
                // Bessel is min-phase only.
                if matches!(fam, FilterType::Bessel) && lp_lin { continue; }
                let target = TargetCurve {
                    reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
                    high_pass: Some(section(FilterType::LinkwitzRiley, 2, 50.0, hp_lin)),
                    low_pass: Some(section(fam.clone(), *order, 750.0, lp_lin)),
                    low_shelf: None, high_shelf: None,
                };
                for (lab, p) in &peqs {
                    let case = format!("{sr} Hz/{taps}, HP {} + LP {:?} {}, {lab}",
                        if hp_lin { "lin" } else { "min" }, fam, if lp_lin { "lin" } else { "min" });
                    check(&case, &target, p, taps, sr, &mut fails);
                    n += 1;
                }
            }
        }
    }
    // The band that found it: wall band 2 with a Gaussian LP, at its export.
    let target = TargetCurve {
        reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(section(FilterType::LinkwitzRiley, 1, 50.0, false)),
        low_pass: Some(section(FilterType::Gaussian, 4, 750.0, true)),
        low_shelf: None, high_shelf: None,
    };
    let wall = peq(r#"[
        {"freq_hz": 24.0, "gain_db": 7.68, "q": 1.52, "enabled": true},
        {"freq_hz": 37.08, "gain_db": 3.31, "q": 1.82, "enabled": true},
        {"freq_hz": 1094.7, "gain_db": -1.45, "q": 1.5, "enabled": true},
        {"freq_hz": 6000.0, "gain_db": -16.0, "q": 1.5, "enabled": true}]"#);
    check("wall band 2, Gaussian 750 lin, 176.4k/128K", &target, &wall, 131072, 176_400.0, &mut fails);
    n += 1;
    assert!(fails.is_empty(), "refinement moved the phase above {FLOOR_VIEW_DB} dB in {}/{n} cases:\n{}",
        fails.len(), fails.join("\n"));
}
