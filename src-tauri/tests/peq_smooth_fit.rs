// b141.87: «Без отражений» — the PEQ fit on the smoothed measurement must not
// plant narrow filters in the reflection ripple of a long-window REW export.
// Real mid band (REW, 1/48 smoothing, 100 Hz–60 kHz) in the gitignored
// test-fixtures/prering; skipped when missing.
use phaseforge_lib::dsp::reflection_free_smoothing;
use phaseforge_lib::peq::{auto_peq_lma, PeqConfig};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

fn lr(order: u8, f: f64) -> FilterConfig {
    FilterConfig { filter_type: FilterType::LinkwitzRiley, order, freq_hz: f, shape: None,
        linear_phase: true, q: None, subsonic_protect: None }
}

#[test]
fn smooth_fit_drops_ripple_filters_on_real_mid_band() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/mid90.txt");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    // The project's band: LR2 300 / LR3 1800 linear, hybrid, tolerance 0.5.
    let (hp, lp) = (300.0, 1800.0);
    let (s, c) = freq.iter().zip(&mag).filter(|(f, _)| **f >= hp && **f <= lp)
        .fold((0.0, 0), |(s, c), (_, m)| (s + m, c + 1));
    let tc = TargetCurve { reference_level_db: s / c as f64, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(lr(2, hp)), low_pass: Some(lr(3, lp)), low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, &freq).magnitude;
    let thr = tc.reference_level_db - 60.0;
    let lo = freq[t.iter().position(|&v| v > thr).unwrap()].max(20.0);
    let hi = freq[t.iter().rposition(|&v| v > thr).unwrap()].min(30_000.0);
    let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
        "max_bands": 20, "tolerance_db": 0.5, "peak_bias": 1.0, "max_boost_db": 60.0,
        "max_cut_db": 60.0, "freq_range": [lo, hi], "hybrid": true, "gain_regularization": 0.0,
        "sample_rate": 352800.0 })).unwrap();
    let run = |m: &[f64], lab: &str| {
        let r = auto_peq_lma(m, Some(&t), &freq, &cfg, hp, lp, &[]).unwrap();
        let narrow: Vec<_> = r.bands.iter().filter(|b| b.q > 4.0 && b.freq_hz > 100.0)
            .map(|b| format!("{:.0} Hz Q{:.1} {:+.1} dB", b.freq_hz, b.q, b.gain_db)).collect();
        println!("{lab}: {} bands, max err {:.2} dB, narrow (Q>4): {narrow:?}", r.bands.len(), r.max_error_db);
        narrow.len()
    };
    let raw = run(&mag, "raw");
    let smooth = run(&reflection_free_smoothing(&freq, &mag), "smooth");
    // raw: 142/290/417/690 Hz at Q 5–7 (the project's own fit); smooth: one left
    assert!(raw >= 3 && smooth <= 1, "smooth fit still plants narrow filters: {smooth} (raw {raw})");
}
