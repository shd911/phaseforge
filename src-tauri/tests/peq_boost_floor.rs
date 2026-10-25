// b141.92: «Подъём до» — below it the PEQ may only cut. Real band 2-6.5m
// (LR2 250 / LR2 2000 linear, hybrid, range 30–21110, floor 70): the fit
// boosted +32.6 dB at 13.4 kHz and +15 dB at 53 Hz, where the target is
// −66 / −54 dB. Fixture in the gitignored test-fixtures/prering.
use phaseforge_lib::dsp::reflection_free_smoothing;
use phaseforge_lib::peq::{auto_peq_lma, PeqConfig};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

fn lr(order: u8, f: f64) -> FilterConfig {
    FilterConfig { filter_type: FilterType::LinkwitzRiley, order, freq_hz: f, shape: None,
        linear_phase: true, q: None, subsonic_protect: None, complementary: false }
}

/// First/last frequency where `t` is above `thr`, inside [lo, hi].
fn trim(freq: &[f64], t: &[f64], lo: f64, hi: f64, thr: f64) -> (f64, f64) {
    let a = t.iter().position(|&v| v > thr).map_or(lo, |i| freq[i].max(lo));
    let b = t.iter().rposition(|&v| v > thr).map_or(hi, |i| freq[i].min(hi));
    (a, b)
}

#[test]
fn no_boost_under_the_crossover_stopband() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/mid_2-6.5m.txt");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    let (hp, lp) = (250.0, 2000.0);
    let (s, c) = freq.iter().zip(&mag).filter(|(f, _)| **f >= hp && **f <= lp)
        .fold((0.0, 0), |(s, c), (_, m)| (s + m, c + 1));
    let tc = TargetCurve { reference_level_db: s / c as f64, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(lr(2, hp)), low_pass: Some(lr(2, lp)), low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, &freq).magnitude;
    let r = tc.reference_level_db;
    let range = trim(&freq, &t, 30.0, 21_110.0, r - 70.0);
    let boost = trim(&freq, &t, range.0, range.1, r - 30.0);
    let meas = reflection_free_smoothing(&freq, &mag);
    let run = |boost_range: Option<(f64, f64)>, lab: &str| {
        let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
            "max_bands": 20, "tolerance_db": 1.0, "peak_bias": 1.0, "max_boost_db": 60.0,
            "max_cut_db": 60.0, "freq_range": [range.0, range.1], "hybrid": true,
            "gain_regularization": 0.0, "sample_rate": 352800.0, "boost_range": boost_range })).unwrap();
        let res = auto_peq_lma(&meas, Some(&t), &freq, &cfg, hp, lp, &[]).unwrap();
        let bands: Vec<String> = res.bands.iter().map(|b| format!("{:.0}:{:+.1}/Q{:.2}", b.freq_hz, b.gain_db, b.q)).collect();
        println!("{lab}: {bands:?}");
        res.bands
    };
    println!("range {:.0}–{:.0} Hz, boosts allowed {:.0}–{:.0} Hz", range.0, range.1, boost.0, boost.1);
    let before = run(None, "no limit ");
    let after = run(Some(boost), "boost ≤30");
    assert!(before.iter().any(|b| b.gain_db > 6.0 && (b.freq_hz < boost.0 || b.freq_hz > boost.1)),
        "fixture no longer reproduces the stopband boost");
    for b in &after {
        if b.freq_hz < boost.0 || b.freq_hz > boost.1 {
            assert!(b.gain_db <= 1e-9, "boost {:+.1} dB at {:.0} Hz outside {:?}", b.gain_db, b.freq_hz, boost);
        }
    }
}

/// b141.101 (audit 2026-10-23 1.10): shelf promotion must honour «Подъём до».
/// It checked the boost at the band's centre only, and a low shelf keeps its
/// gain down to DC: mid90, LR4 300/3000, hybrid → LowShelf 196 Hz +7.1 dB =
/// +6.2 dB at 100 Hz (target −76 dB there) and so on to DC. A boosting shelf
/// may not lift the response outside the boost range by more than the
/// peaking band it replaces did.
#[test]
fn shelf_promotion_respects_the_boost_range() {
    use phaseforge_lib::peq::{peq_band_response, PeqBand, PeqFilterType};
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/mid90.txt");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    let (hp, lp) = (300.0, 3000.0);
    let (s, c) = freq.iter().zip(&mag).filter(|(f, _)| **f >= hp && **f <= lp)
        .fold((0.0, 0), |(s, c), (_, m)| (s + m, c + 1));
    let tc = TargetCurve { reference_level_db: s / c as f64, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(lr(4, hp)), low_pass: Some(lr(4, lp)), low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, &freq).magnitude;
    let r = tc.reference_level_db;
    let range = trim(&freq, &t, 20.0, 30_000.0, r - 80.0);
    let boost = trim(&freq, &t, range.0, range.1, r - 30.0);
    let meas = reflection_free_smoothing(&freq, &mag);
    let sr = 352_800.0;
    let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
        "max_bands": 20, "tolerance_db": 1.0, "peak_bias": 1.0, "max_boost_db": 60.0,
        "max_cut_db": 60.0, "freq_range": [range.0, range.1], "hybrid": true,
        "gain_regularization": 0.0, "sample_rate": sr, "boost_range": boost })).unwrap();
    let res = auto_peq_lma(&meas, Some(&t), &freq, &cfg, hp, lp, &[]).unwrap();
    let bands: Vec<String> = res.bands.iter()
        .map(|b| format!("{:?} {:.0}:{:+.1}/Q{:.2}", b.filter_type, b.freq_hz, b.gain_db, b.q)).collect();
    println!("range {:.0}–{:.0}, boosts {:.0}–{:.0}: {bands:?}", range.0, range.1, boost.0, boost.1);
    for b in res.bands.iter().filter(|b| b.filter_type != PeqFilterType::Peaking && b.gain_db > 0.0) {
        let shelf = peq_band_response(&freq, b, sr);
        let peak = peq_band_response(&freq, &PeqBand { filter_type: PeqFilterType::Peaking, q: b.q / 0.7, ..b.clone() }, sr);
        for (i, &f) in freq.iter().enumerate() {
            if f < boost.0 || f > boost.1 {
                assert!(shelf[i] <= peak[i].max(0.0) + 0.1,
                    "{:?} {:.0} Hz {:+.1} dB lifts {f:.0} Hz by {:+.1} dB outside the boost range",
                    b.filter_type, b.freq_hz, b.gain_db, shelf[i]);
            }
        }
    }
}
