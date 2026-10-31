// b141.134: nested crossovers — the sum of a tree of complementary pairs is
// exactly 1 on linear phase, and a band stops leaking past its neighbour.
// Bands from VPV2 (70/220/800/2500 Hz, 12 dB/oct, 1.5–2 octaves wide): without
// the tree the sum is off by 0.74 dB and band 2 in the middle of band 4 is at
// only −32.6 dB. Read from the files, not from the design.
use phaseforge_lib::fir::band::{generate_band_fir, BandFirResult, BandFirRoute, BandFirSettings};
use phaseforge_lib::fir::WindowType;
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};
use std::f64::consts::PI;

const XO: [f64; 4] = [70.0, 220.0, 800.0, 2500.0];

fn sec(ty: &FilterType, order: u8, shape: Option<f64>, fc: f64, nested: Vec<FilterConfig>) -> FilterConfig {
    FilterConfig { filter_type: ty.clone(), order, freq_hz: fc, shape, linear_phase: true, q: None, subsonic_protect: None, complementary: false, nested }
}

fn curve(hp: Option<FilterConfig>, lp: Option<FilterConfig>) -> TargetCurve {
    TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0, high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None }
}

/// With `tree`, from the centre 800 Hz (lib/nested-crossovers.ts pins the same
/// shape): L70·L220·L800, H70·L220·L800, H220·L800, H800·L2500, H800·H2500.
fn bands(ty: &FilterType, order: u8, shape: Option<f64>, tree: bool) -> Vec<TargetCurve> {
    let s = |i: usize, nested: &[usize]| sec(ty, order, shape, XO[i], if tree { nested.iter().map(|&n| sec(ty, order, shape, XO[n], vec![])).collect() } else { vec![] });
    vec![
        curve(None, Some(s(0, &[1, 2]))),
        curve(Some(s(0, &[])), Some(s(1, &[2]))),
        curve(Some(s(1, &[])), Some(s(2, &[]))),
        curve(Some(s(2, &[])), Some(s(3, &[]))),
        curve(Some(s(3, &[2])), None),
    ]
}

fn settings(sample_rate: f64, taps: usize) -> BandFirSettings {
    BandFirSettings { taps, sample_rate, window: WindowType::Blackman, max_boost_db: 24.0, noise_floor_db: -150.0,
        iterations: 3, freq_weighting: true, narrowband_limit: true, nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 }
}

/// The file's complex response at `f` (before its norm), by a direct DFT.
fn at(h: &[f64], f: f64, sr: f64) -> (f64, f64) {
    let (s, c) = (2.0 * PI * f / sr).sin_cos();
    let (mut re, mut im, mut cr, mut ci) = (0.0, 0.0, 1.0, 0.0);
    for &v in h {
        re += v * cr;
        im -= v * ci;
        (cr, ci) = (cr * c - ci * s, cr * s + ci * c);
    }
    (re, im)
}

fn gen(t: &[TargetCurve], sr: f64, taps: usize) -> Vec<BandFirResult> {
    t.iter().map(|b| generate_band_fir(b, &[], &settings(sr, taps), false).unwrap()).collect()
}

/// Level of the sum of the files at `f`, each with its norm undone.
fn sum_db(files: &[BandFirResult], f: f64, sr: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for b in files {
        let g = 10f64.powf(b.fir.norm_db / 20.0);
        let (r, i) = at(&b.fir.impulse, f, sr);
        re += r * g;
        im += i * g;
    }
    10.0 * (re * re + im * im).max(1e-40).log10()
}

fn band_db(b: &BandFirResult, f: f64, sr: f64) -> f64 {
    let (r, i) = at(&b.fir.impulse, f, sr);
    10.0 * (r * r + i * i).max(1e-40).log10() + b.fir.norm_db
}

const PROBES: [f64; 14] = [40.0, 70.0, 125.0, 220.0, 400.0, 600.0, 800.0, 1100.0, 1414.0, 2000.0, 2500.0, 4000.0, 8000.0, 15000.0];

fn families() -> Vec<(&'static str, FilterType, u8, Option<f64>)> {
    vec![("LR1", FilterType::LinkwitzRiley, 1, None), ("LR2", FilterType::LinkwitzRiley, 2, None), ("Gauss M1", FilterType::Gaussian, 1, Some(1.0))]
}

#[test]
fn the_sum_of_the_tree_is_flat_in_the_files() {
    for (sr, taps) in [(48_000.0, 65_536), (352_800.0, 262_144)] {
        for (name, ty, order, shape) in families() {
            let files = gen(&bands(&ty, order, shape, true), sr, taps);
            for b in &files { assert!(matches!(b.route, BandFirRoute::Cepstral), "{name}: a linear band on the IIR route"); }
            let worst = PROBES.iter().map(|&f| sum_db(&files, f, sr).abs()).fold(0.0, f64::max);
            println!("{sr} {name}: sum within ±{worst:.3} dB");
            assert!(worst < 0.05, "{sr} {name}: the sum is off by {worst:.3} dB");
        }
    }
}

/// The leak the tree exists for: band 2 in band 4, band 5 in band 3.
#[test]
fn a_band_stops_leaking_past_its_neighbour() {
    let (sr, taps) = (48_000.0, 65_536);
    let ty = FilterType::LinkwitzRiley;
    let plain = gen(&bands(&ty, 1, None, false), sr, taps);
    let tree = gen(&bands(&ty, 1, None, true), sr, taps);
    let (b2_mid4, b5_mid3) = ((800.0f64 * 2500.0).sqrt(), (220.0f64 * 800.0).sqrt());
    let row = |v: &[BandFirResult]| (band_db(&v[1], b2_mid4, sr), band_db(&v[4], b5_mid3, sr));
    let ((p2, p5), (t2, t5)) = (row(&plain), row(&tree));
    println!("band 2 @ middle of band 4: {p2:.1} → {t2:.1} dB; band 5 @ middle of band 3: {p5:.1} → {t5:.1} dB");
    assert!(t2 < -44.0 && t5 < -44.0, "tree: {t2:.1} / {t5:.1} dB");
    assert!(p2 > -34.0 && p5 > -34.0, "the plain crossover already suppressed: {p2:.1} / {p5:.1} dB");
    // A band's own crossover pays well under 1.5 dB at its passband centre…
    for (i, f) in [(1usize, (70.0f64 * 220.0).sqrt()), (2, (220.0f64 * 800.0).sqrt()), (3, (800.0f64 * 2500.0).sqrt())] {
        let (p, t) = (band_db(&plain[i], f, sr), band_db(&tree[i], f, sr));
        println!("band {} at its centre {f:.0} Hz: {p:.2} → {t:.2} dB", i + 1);
        assert!((p - t).abs() < 1.5, "band {}: {p:.2} → {t:.2} dB", i + 1);
    }
}

/// Time-domain cost of the tree: −60 dB before the peak, in ms.
fn pre_ring_ms(h: &[f64], sr: f64) -> f64 {
    let peak = h.iter().enumerate().fold((0, 0.0f64), |m, (i, v)| if v.abs() > m.1 { (i, v.abs()) } else { m });
    let first = h.iter().position(|v| v.abs() > peak.1 * 1e-3).unwrap();
    (peak.0 - first) as f64 / sr * 1000.0
}

#[test]
fn the_tree_costs_little_pre_ring() {
    let (sr, taps) = (48_000.0, 65_536);
    let ty = FilterType::LinkwitzRiley;
    for order in [1u8, 2] {
        let plain = gen(&bands(&ty, order, None, false), sr, taps);
        let tree = gen(&bands(&ty, order, None, true), sr, taps);
        for i in 0..5 {
            let (p, t) = (pre_ring_ms(&plain[i].fir.impulse, sr), pre_ring_ms(&tree[i].fir.impulse, sr));
            println!("LR{order} band {}: pre-ring to −60 dB {p:.1} → {t:.1} ms", i + 1);
            // Nested sections sit above a band's own LP and below its own HP,
            // so they cannot ring longer than that section's own lowest corner.
            assert!(t < p * 1.5 + 1.0, "LR{order} band {}: {p:.1} → {t:.1} ms", i + 1);
        }
    }
}

/// A nested list on a min-phase section is ignored: the file stays bit-identical.
#[test]
fn a_min_phase_section_ignores_nested() {
    let ty = FilterType::LinkwitzRiley;
    let min = |nested: Vec<FilterConfig>| FilterConfig { linear_phase: false, ..sec(&ty, 2, None, 300.0, nested) };
    let (sr, taps) = (48_000.0, 65_536);
    let a = generate_band_fir(&curve(None, Some(min(vec![]))), &[], &settings(sr, taps), false).unwrap();
    let b = generate_band_fir(&curve(None, Some(min(vec![sec(&ty, 2, None, 1000.0, vec![])]))), &[], &settings(sr, taps), false).unwrap();
    assert_eq!(a.fir.impulse, b.fir.impulse);
}

/// A mixed band: HP min-phase below a non-qualifying crossover, LP linear with
/// a nested LP — the file still has the nested section in it.
#[test]
fn a_mixed_band_keeps_its_nested_section() {
    let ty = FilterType::LinkwitzRiley;
    let (sr, taps) = (48_000.0, 65_536);
    let hp = FilterConfig { linear_phase: false, ..sec(&ty, 2, None, 100.0, vec![]) };
    let t = curve(Some(hp), Some(sec(&ty, 1, None, 500.0, vec![sec(&ty, 1, None, 2000.0, vec![])])));
    let plain = curve(t.high_pass.clone(), Some(sec(&ty, 1, None, 500.0, vec![])));
    let a = generate_band_fir(&t, &[], &settings(sr, taps), false).unwrap();
    let p = generate_band_fir(&plain, &[], &settings(sr, taps), false).unwrap();
    let want = -20.0 * (1.0 + (4000.0f64 / 2000.0).powi(2)).log10(); // LR1 = BW1² at 4 kHz
    let got = band_db(&a, 4000.0, sr) - band_db(&p, 4000.0, sr);
    println!("nested LP 2 kHz at 4 kHz in a mixed band: {got:.2} dB, asked {want:.2}");
    assert!((got - want).abs() < 0.2, "{got:.2} vs {want:.2} dB");
}

/// The real project this was asked for: 70 Hz LR4 min-phase stays out of the
/// tree, 220 LR2 / 800 LR2 / 2500 LR4 linear nest from 800 Hz.
#[test]
fn a_real_five_way_with_a_min_phase_bottom() {
    let (sr, taps) = (48_000.0, 65_536);
    let ty = FilterType::LinkwitzRiley;
    let min70 = || FilterConfig { linear_phase: false, ..sec(&ty, 4, None, 70.0, vec![]) };
    let chain = |tree: bool| {
        let n = |f: f64, order: u8| if tree { vec![sec(&ty, order, None, f, vec![])] } else { vec![] };
        vec![
            curve(None, Some(min70())),
            curve(Some(min70()), Some(sec(&ty, 2, None, 220.0, n(800.0, 2)))),
            curve(Some(sec(&ty, 2, None, 220.0, vec![])), Some(sec(&ty, 2, None, 800.0, vec![]))),
            curve(Some(sec(&ty, 2, None, 800.0, vec![])), Some(sec(&ty, 4, None, 2500.0, vec![]))),
            curve(Some(sec(&ty, 4, None, 2500.0, n(800.0, 2))), None),
        ]
    };
    let (plain, tree) = (gen(&chain(false), sr, taps), gen(&chain(true), sr, taps));
    // The min-phase HP 70 Hz has turned 96° by 220 Hz, where band 3 is zero
    // phase: the sum dips there with or without the tree. The tree must not
    // add to it, and above that it is flat.
    for &f in &PROBES {
        let (p, t) = (sum_db(&plain, f, sr), sum_db(&tree, f, sr));
        println!("{f} Hz: sum {p:+.3} → {t:+.3} dB");
        assert!((t - p).abs() < 0.25, "{f} Hz: {p:+.3} → {t:+.3} dB");
        if f >= 600.0 { assert!(t.abs() < 0.1, "{f} Hz: {t:+.3} dB"); }
    }
    println!("band 2 @ 1414 Hz {:.1} → {:.1} dB; band 5 @ 420 Hz {:.1} → {:.1} dB",
        band_db(&plain[1], 1414.0, sr), band_db(&tree[1], 1414.0, sr), band_db(&plain[4], 420.0, sr), band_db(&tree[4], 420.0, sr));
    assert!(band_db(&tree[1], 1414.0, sr) < band_db(&plain[1], 1414.0, sr) - 10.0);
}
