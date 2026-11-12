// b141.149: a break-up peak beyond the band's low-pass gets a narrow cut.
//
// The optimizer flattens the measurement to the passband line over its whole
// range, the stopband included (down to «Порог, дБ» under the reference), so
// a cone break-up above the crossover is its business. But above the low-pass
// it could only place WIDE filters (Q ≤ 2.5, penalised from Q 1.5), and a wide
// filter cannot take a narrow peak out: the fit settled on half the depth.
//
// Settings are the app's (peq-optimize.ts): Standard = bias 1.5, +6/−18 dB;
// range trimmed where the target is 60 dB down, boosts where it is 30 dB
// down, 48 kHz / 65536 taps (ring limit = half the file).
use phaseforge_lib::dsp::reflection_free_smoothing;
use phaseforge_lib::peq::{apply_peq, auto_peq_lma, PeqBand, PeqConfig};
use phaseforge_lib::target::{self, FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

const SR: f64 = 48_000.0;

fn flt(t: FilterType, order: u8, f: f64) -> FilterConfig {
    FilterConfig { filter_type: t, order, freq_hz: f, shape: None,
        linear_phase: true, q: None, subsonic_protect: None, complementary: false, nested: Vec::new() }
}

/// First/last frequency where `t` is above `thr`, inside [lo, hi].
fn trim(freq: &[f64], t: &[f64], lo: f64, hi: f64, thr: f64) -> (f64, f64) {
    let a = t.iter().position(|&v| v > thr).map_or(lo, |i| freq[i].max(lo));
    let b = t.iter().rposition(|&v| v > thr).map_or(hi, |i| freq[i].min(hi));
    (a, b)
}

struct Fit { bands: Vec<PeqBand>, peq: Vec<f64>, line: f64, range: (f64, f64) }

/// The app's optimizer call for a band with this low-pass and no high-pass.
fn fit(freq: &[f64], mag: &[f64], lp: &FilterConfig, smooth: bool) -> Fit {
    let (lo, hi) = (30.0, lp.freq_hz * 0.7);
    let (s, c) = freq.iter().zip(mag).filter(|(f, _)| **f >= lo && **f <= hi)
        .fold((0.0, 0), |(s, c), (_, m)| (s + m, c + 1));
    let line = s / c as f64;
    let tc = TargetCurve { reference_level_db: line, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: None, low_pass: Some(lp.clone()), low_shelf: None, high_shelf: None };
    let t = target::evaluate(&tc, freq).magnitude;
    let fmax = (0.95 * SR / 2.0).min(*freq.last().unwrap());
    let range = trim(freq, &t, 20.0, fmax, line - 60.0);
    let boost = trim(freq, &t, range.0, range.1, line - 30.0);
    let meas = if smooth { reflection_free_smoothing(freq, mag) } else { mag.to_vec() };
    let cfg: PeqConfig = serde_json::from_value(serde_json::json!({
        "max_bands": 20, "tolerance_db": 1.0, "peak_bias": 1.5, "max_boost_db": 6.0, "max_cut_db": 18.0,
        "freq_range": [range.0, range.1], "hybrid": false, "gain_regularization": 0.0, "sample_rate": SR,
        "boost_range": [boost.0, boost.1], "max_ring_ms": 65_536.0 / SR * 1000.0 / 2.0 })).unwrap();
    let res = auto_peq_lma(&meas, Some(&t), freq, &cfg, 20.0, lp.freq_hz, &[]).unwrap();
    let peq = apply_peq(freq, &res.bands, SR);
    Fit { bands: res.bands, peq, line, range }
}

/// Highest raw level over the passband line inside `win`, before / after the PEQ.
fn peak(freq: &[f64], mag: &[f64], f: &Fit, win: (f64, f64)) -> (f64, f64) {
    let idx = || freq.iter().enumerate().filter(|(_, &x)| x >= win.0 && x <= win.1 && x <= f.range.1).map(|(i, _)| i);
    (idx().map(|i| mag[i] - f.line).fold(f64::MIN, f64::max),
     idx().map(|i| mag[i] - f.line + f.peq[i]).fold(f64::MIN, f64::max))
}

fn show(b: &[PeqBand]) -> Vec<String> {
    b.iter().map(|b| format!("{:.0}:{:+.1}/Q{:.1}", b.freq_hz, b.gain_db, b.q)).collect()
}

/// A flat driver with a +15 dB break-up of Q 6 (half-height width f0/6) at
/// 4.5 kHz, low-pass 500 Hz 12 dB/oct — the target is −38 dB at the peak, well
/// inside the optimizer's range.
#[test]
fn narrow_breakup_beyond_the_low_pass_is_cut_out() {
    let freq: Vec<f64> = (0..).map(|i| 10.0 * 2f64.powf(i as f64 / 48.0)).take_while(|&f| f <= 24_000.0).collect();
    let (f0, g, q) = (4500.0, 15.0, 6.0);
    let mag: Vec<f64> = freq.iter().map(|&f| 85.0 + g / (1.0 + (q * (f / f0 - f0 / f)).powi(2))).collect();
    let lp = flt(FilterType::Butterworth, 2, 500.0);
    for smooth in [false, true] {
        let r = fit(&freq, &mag, &lp, smooth);
        let (before, after) = peak(&freq, &mag, &r, (f0 / 1.26, f0 * 1.26));
        let cut = r.bands.iter().filter(|b| (b.freq_hz / f0).log2().abs() < 0.2)
            .min_by(|a, b| a.gain_db.partial_cmp(&b.gain_db).unwrap()).expect("no filter on the peak");
        println!("smooth={smooth}: peak {before:+.1} → {after:+.1} dB, {:?}", show(&r.bands));
        // It was −8.8…−10.2 dB at Q 1.6, leaving +5.5…+6.4 dB of the peak.
        assert!(cut.q > 2.5, "smooth={smooth}: the cut is still wide, Q {:.2}", cut.q);
        assert!(after < MAX_LEFT[smooth as usize], "smooth={smooth}: {after:+.1} dB of the peak left");
        // Only the cut is narrow: boosts and shallow cuts stay wide above the low-pass too.
        for b in r.bands.iter().filter(|b| b.freq_hz > lp.freq_hz && b.gain_db > -3.0) {
            assert!(b.q <= 2.0 + 1e-9, "smooth={smooth}: {:+.1} dB at {:.0} Hz has Q {:.2}", b.gain_db, b.freq_hz, b.q);
        }
    }
}
/// Measured +2.0 dB raw, +2.5 dB with «Без отражений» (it was +6.4 / +5.5).
const MAX_LEFT: [f64; 2] = [3.0, 3.5];

/// Real sub (wall 10x4): the cone break-up at 3.8 kHz stands 20 dB over its
/// surroundings and +3.5 dB over the passband. With a 1 kHz 24 dB/oct low-pass
/// the PEQ used to RAISE it to +5.0…+5.5 dB (a wide +6 dB filling the dip in
/// front of it, a wide −6 dB cut on the peak itself).
#[test]
fn real_sub_breakup_is_not_raised() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/sub_10x4.txt");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let text = std::fs::read_to_string(&p).unwrap();
    let (mut freq, mut mag) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| l.starts_with(|c: char| c.is_ascii_digit())) {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        freq.push(v[0]); mag.push(v[1]);
    }
    let lp = flt(FilterType::LinkwitzRiley, 2, 1000.0);
    for smooth in [false, true] {
        let r = fit(&freq, &mag, &lp, smooth);
        let (before, after) = peak(&freq, &mag, &r, (3000.0, 4800.0));
        println!("smooth={smooth}: peak {before:+.1} → {after:+.1} dB, {:?}", show(&r.bands));
        // Measured +1.8 dB raw, +3.4 dB with «Без отражений» (1/3 octave blurs
        // the peak before the fit; the dip in front of it is still filled).
        let limit = if smooth { before } else { before - 1.0 };
        assert!(after <= limit, "smooth={smooth}: break-up {before:+.1} → {after:+.1} dB, expected ≤ {limit:+.1}");
    }
}
