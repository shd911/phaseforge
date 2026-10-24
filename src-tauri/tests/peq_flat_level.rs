// b141.106: the hybrid optimizer's flat line and «Без отражений».
//
// Real sub band of the «wall» project (10x4: LR1 130 Hz min-phase, hybrid,
// range 24–26000 trimmed by the 80 dB floor, boost to 30, 176.4 kHz /
// 131072 taps). The flat line was the MEAN of the measurement over HP..LP =
// 20..130 Hz — the driver's roll-off below the 24 Hz range start (−17 dB at
// 20 Hz) pulled it 1.5 dB under the median of the optimized range, so the
// whole 55–200 Hz looked too loud and got cut by four filters. «Без
// отражений» kept 1/48 octave below 100 Hz: ±1 dB room ripple (4 Hz period at
// 26–32 Hz, 13–15 Hz at 55–83 Hz) reached the fit and drew Q 2–2.5 filters
// that cannot follow it. Fixture in the gitignored test-fixtures/prering.
use phaseforge_lib::dsp::{reflection_free_smoothing, variable_smoothing, SmoothingConfig};
use phaseforge_lib::peq::q_envelope::ring_ms;
use phaseforge_lib::peq::{apply_peq, auto_peq_lma, PeqBand, PeqConfig};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

const SR: f64 = 176_400.0;
const LP: f64 = 130.0;

fn trim(freq: &[f64], t: &[f64], lo: f64, hi: f64, thr: f64) -> (f64, f64) {
    let a = t.iter().position(|&v| v > thr).map_or(lo, |i| freq[i].max(lo));
    let b = t.iter().rposition(|&v| v > thr).map_or(hi, |i| freq[i].min(hi));
    (a, b)
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn load(name: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering").join(name);
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return None; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    Some((freq, mag))
}

/// The production request for the band: what peq-optimize.ts sends.
fn optimize(freq: &[f64], mag: &[f64], range_lo: f64) -> (Vec<PeqBand>, (f64, f64)) {
    let tc = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: None,
        low_pass: Some(FilterConfig { filter_type: FilterType::LinkwitzRiley, order: 1, freq_hz: LP, shape: None,
            linear_phase: false, q: None, subsonic_protect: None }),
        low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, freq).magnitude;
    let range = trim(freq, &t, range_lo, 26_000.0, -80.0);
    let boost = trim(freq, &t, range.0, range.1, -30.0);
    let meas = reflection_free_smoothing(freq, mag);
    let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
        "max_bands": 20, "tolerance_db": 1.0, "peak_bias": 1.0, "max_boost_db": 60.0,
        "max_cut_db": 60.0, "freq_range": [range.0, range.1], "hybrid": true,
        "gain_regularization": 0.0, "sample_rate": SR, "boost_range": boost,
        "max_ring_ms": 131_072.0 / 2.0 / SR * 1000.0 })).unwrap();
    let res = auto_peq_lma(&meas, Some(&t), freq, &cfg, 20.0, LP, &[]).unwrap();
    (res.bands, range)
}

fn describe(bands: &[PeqBand]) -> Vec<String> {
    bands.iter().map(|b| format!("{:.0} Hz {:+.1} Q{:.2} ({:.0} ms)", b.freq_hz, b.gain_db, b.q,
        ring_ms(b.freq_hz, b.gain_db, b.q, &b.filter_type, SR, 60.0))).collect()
}

/// A flat driver with a steep roll-off below the range start: the flat line
/// sits on its passband, so nothing in the passband is cut (the old mean put
/// the line under it and drew a −1.1 dB cut at 502 Hz).
#[test]
fn roll_off_below_the_range_does_not_lower_the_flat_line() {
    let freq: Vec<f64> = (0..600).map(|i| 20.0 * 1000f64.powf(i as f64 / 599.0)).collect();
    // 84 dB flat, a 4th-order roll-off at 25 Hz (−17 dB at 20 Hz).
    let mag: Vec<f64> = freq.iter().map(|&f| 84.0 - 10.0 * (1.0 + (25.0 / f).powi(8)).log10()).collect();
    let (bands, range) = optimize(&freq, &mag, 24.0);
    println!("range {:.0}–{:.0}: {:?}", range.0, range.1, describe(&bands));
    for b in &bands {
        assert!(b.gain_db > -0.5, "a flat passband got a cut: {:.0} Hz {:+.1} dB", b.freq_hz, b.gain_db);
    }
}

/// 1/6 octave below 100 Hz keeps a real +6 dB Q 5 room mode — the narrow
/// cut the optimizer may still place (from 3 dB) sees it.
#[test]
fn reflection_free_keeps_room_modes() {
    let freq: Vec<f64> = (0..2000).map(|i| 20.0 * 1000f64.powf(i as f64 / 1999.0)).collect();
    // Peaking +6 dB Q 5 at 50 Hz (analog magnitude).
    let mode: Vec<f64> = freq.iter().map(|&f| {
        let (x, a) = (f / 50.0, 10f64.powf(6.0 / 40.0));
        let num = (1.0 - x * x).powi(2) + (x * a / 5.0).powi(2);
        let den = (1.0 - x * x).powi(2) + (x / (a * 5.0)).powi(2);
        80.0 + 10.0 * (num / den).log10()
    }).collect();
    let sm = reflection_free_smoothing(&freq, &mode);
    let peak = freq.iter().zip(&sm).filter(|(f, _)| **f >= 45.0 && **f <= 55.0)
        .map(|(_, &v)| v - 80.0).fold(f64::MIN, f64::max);
    println!("mode +6 dB Q5 → +{peak:.2} dB");
    assert!(peak > 4.5, "a real mode is smoothed away: +{peak:.2} dB");
}

/// The wall sub band. Before (b141.105): 24 +3.1/Q2, 30 −2.9/Q2, 43 +3.2/Q1.97,
/// 55 −3.1/Q2.47, 86 −1.9, 175 −1.7, 337 −3.6, 718 +1.3 — mean PEQ over
/// 30–130 Hz −1.32 dB, worst 1/3-octave deviation −2.26 dB. The corrected
/// passband must stay within 1.8 dB of its median, and the filters must not
/// shift the passband as a whole.
#[test]
fn wall_sub_band_fits_the_passband_not_the_room() {
    let Some((freq, mag)) = load("sub_10x4.txt") else { return };
    let (bands, range) = optimize(&freq, &mag, 24.0);
    println!("range {:.0}–{:.0} Hz, {} filters:", range.0, range.1, bands.len());
    for d in describe(&bands) { println!("  {d}"); }

    let mut pb: Vec<f64> = freq.iter().zip(&mag).filter(|(f, _)| **f >= range.0 && **f <= LP).map(|(_, &m)| m).collect();
    let level = median(&mut pb);
    let corr = apply_peq(&freq, &bands, SR);
    let out: Vec<f64> = mag.iter().zip(&corr).map(|(m, c)| m + c).collect();
    let third = variable_smoothing(&freq, &out, &SmoothingConfig { variable: false, fixed_fraction: Some(1.0 / 3.0) });
    let worst = freq.iter().enumerate().filter(|(_, f)| **f >= 26.0 && **f <= LP)
        .map(|(i, _)| third[i] - level).fold(0.0f64, |m, e| if e.abs() > m.abs() { e } else { m });
    // Mean PEQ over the passband: a filter set that shifts the whole band
    // down is fitting the wrong line.
    let pts: Vec<f64> = freq.iter().zip(&corr).filter(|(f, _)| **f >= 30.0 && **f <= LP).map(|(_, &c)| c).collect();
    let mean_peq = pts.iter().sum::<f64>() / pts.len() as f64;
    println!("passband median {level:.2} dB; worst 1/3-oct deviation {worst:+.2} dB; mean PEQ 30–130 Hz {mean_peq:+.2} dB");
    assert!(worst.abs() <= 1.8, "corrected passband off by {worst:+.2} dB");
    assert!(mean_peq.abs() <= 0.5, "PEQ shifts the whole passband by {mean_peq:+.2} dB");
}
