// b141.94: the optimizer's filters must not ring past half the export file
// (where the WAV export shifts the band's delay). Real sub band of the
// «wall» project: LR2 130 Hz min-phase, hybrid, range 24–26000, floor 80,
// boost to 30, 176.4 kHz / 131072 taps → half file 371.5 ms. Fixture in the
// gitignored test-fixtures/prering.
use phaseforge_lib::dsp::reflection_free_smoothing;
use phaseforge_lib::peq::q_envelope::ring_ms;
use phaseforge_lib::peq::{auto_peq_lma, PeqConfig};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

fn trim(freq: &[f64], t: &[f64], lo: f64, hi: f64, thr: f64) -> (f64, f64) {
    let a = t.iter().position(|&v| v > thr).map_or(lo, |i| freq[i].max(lo));
    let b = t.iter().rposition(|&v| v > thr).map_or(hi, |i| freq[i].min(hi));
    (a, b)
}

#[test]
fn optimizer_filters_ring_within_half_the_file() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/sub_10x4.txt");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    let lp = 130.0;
    let (s, c) = freq.iter().zip(&mag).filter(|(f, _)| **f >= 20.0 && **f <= lp)
        .fold((0.0, 0), |(s, c), (_, m)| (s + m, c + 1));
    let tc = TargetCurve { reference_level_db: s / c as f64, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: None,
        low_pass: Some(FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 2, freq_hz: lp, shape: None,
            linear_phase: false, q: None, subsonic_protect: None }),
        low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, &freq).magnitude;
    let r = tc.reference_level_db;
    let range = trim(&freq, &t, 24.0, 26_000.0, r - 80.0);
    let boost = trim(&freq, &t, range.0, range.1, r - 30.0);
    let meas = reflection_free_smoothing(&freq, &mag);
    let (sr, half) = (176_400.0, 131_072.0 / 2.0 / 176_400.0 * 1000.0);
    let run = |limit: Option<f64>, lab: &str| {
        let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
            "max_bands": 20, "tolerance_db": 1.0, "peak_bias": 1.0, "max_boost_db": 60.0,
            "max_cut_db": 60.0, "freq_range": [range.0, range.1], "hybrid": true,
            "gain_regularization": 0.0, "sample_rate": sr, "boost_range": boost, "max_ring_ms": limit })).unwrap();
        let res = auto_peq_lma(&meas, Some(&t), &freq, &cfg, 20.0, lp, &[]).unwrap();
        let rings: Vec<f64> = res.bands.iter().map(|b| ring_ms(b.freq_hz, b.gain_db, b.q, &b.filter_type, sr, 100.0)).collect();
        let desc: Vec<String> = res.bands.iter().zip(&rings)
            .map(|(b, r)| format!("{:.0}:{:+.1}/Q{:.2}={:.0}ms", b.freq_hz, b.gain_db, b.q, r)).collect();
        println!("{lab}: max err {:.2} dB {desc:?}", res.max_error_db);
        rings.into_iter().fold(0.0, f64::max)
    };
    let before = run(None, "no limit");
    let after = run(Some(half), "limit   ");
    println!("half file {half:.1} ms; longest ring {before:.0} → {after:.0} ms");
    assert!(after <= half + 1.0, "an optimizer filter rings {after:.0} ms > {half:.0} ms");
}
