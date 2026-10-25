// b141.95: optimizer Q rules — boosts wide (Q ≤ 2), narrow filters only for
// cuts of 3 dB and more. Real woofer band «8» of «bd 3 - full» (LR2 300 Hz
// linear, hybrid, 20–30000 Hz, floor 60, boost to 30, 352.8 kHz / 262144):
// the fit put +1.9 dB Q8 at 91 Hz and −1.1 dB Q8 at 67 Hz. Fixture in the
// gitignored test-fixtures/prering.
use phaseforge_lib::dsp::reflection_free_smoothing;
use phaseforge_lib::peq::{auto_peq_lma, PeqConfig, PeqFilterType};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

fn trim(freq: &[f64], t: &[f64], lo: f64, hi: f64, thr: f64) -> (f64, f64) {
    let a = t.iter().position(|&v| v > thr).map_or(lo, |i| freq[i].max(lo));
    let b = t.iter().rposition(|&v| v > thr).map_or(hi, |i| freq[i].min(hi));
    (a, b)
}

#[test]
fn boosts_wide_narrow_only_for_real_cuts() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/woofer_8.txt");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    let lp = 300.0;
    let (s, c) = freq.iter().zip(&mag).filter(|(f, _)| **f >= 20.0 && **f <= lp)
        .fold((0.0, 0), |(s, c), (_, m)| (s + m, c + 1));
    let tc = TargetCurve { reference_level_db: s / c as f64, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: None,
        low_pass: Some(FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 2, freq_hz: lp, shape: None,
            linear_phase: true, q: None, subsonic_protect: None, complementary: false }),
        low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, &freq).magnitude;
    let r = tc.reference_level_db;
    let range = trim(&freq, &t, 20.0, 30_000.0, r - 60.0);
    let boost = trim(&freq, &t, range.0, range.1, r - 30.0);
    let meas = reflection_free_smoothing(&freq, &mag);
    let sr = 352_800.0;
    let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
        "max_bands": 20, "tolerance_db": 0.5, "peak_bias": 1.0, "max_boost_db": 60.0,
        "max_cut_db": 60.0, "freq_range": [range.0, range.1], "hybrid": true,
        "gain_regularization": 0.0, "sample_rate": sr, "boost_range": boost,
        "max_ring_ms": 262_144.0 / 2.0 / sr * 1000.0 })).unwrap();
    let res = auto_peq_lma(&meas, Some(&t), &freq, &cfg, 20.0, lp, &[]).unwrap();
    for b in &res.bands {
        println!("{:7.1} Hz {:+5.1} dB Q{:5.2} {:?}", b.freq_hz, b.gain_db, b.q, b.filter_type);
    }
    println!("max err {:.2} dB", res.max_error_db);
    for b in res.bands.iter().filter(|b| b.filter_type == PeqFilterType::Peaking) {
        assert!(b.gain_db <= -3.0 || b.q <= 2.0 + 1e-9,
            "narrow filter {:+.1} dB Q{:.2} at {:.0} Hz", b.gain_db, b.q, b.freq_hz);
    }
}
