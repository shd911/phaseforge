// b141.85/86: the iterative refinement must not add pre-ring to a linear-main
// or mixed-phase FIR.
//
// Its correction restores mostly what the window cut off the causal tails of
// the min-phase PEQ; applied zero-phase (with the linear main) it is a
// symmetric copy — on a real woofer band (LR2 @ 250 Hz linear, +6 dB @ 23 Hz,
// 352.8 kHz / 256K) 25–35 Hz energy at −50 dB 10–200 ms ahead of the peak,
// vs −97 dB with no refinement at all. The real project lives in the
// gitignored test-fixtures/prering and is skipped when missing.
use phaseforge_lib::fir::band::{generate_band_fir, BandFirSettings};
use phaseforge_lib::fir::WindowType;
use phaseforge_lib::peq::PeqBand;
use phaseforge_lib::project::ProjectFile;
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

/// Max level (dB re peak) in [peak − b ms, peak − a ms].
fn pre_db(h: &[f64], sr: f64, a: f64, b: f64) -> f64 {
    let (pk, pv) = h.iter().enumerate()
        .fold((0, 0.0f64), |m, (i, v)| if v.abs() > m.1 { (i, v.abs()) } else { m });
    let s = pk.saturating_sub((b * sr / 1000.0) as usize);
    let e = pk.saturating_sub((a * sr / 1000.0) as usize);
    let m = h[s..e].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    20.0 * (m / pv).max(1e-30).log10()
}

fn settings(taps: usize, sr: f64, iterations: usize) -> BandFirSettings {
    BandFirSettings {
        taps, sample_rate: sr, window: WindowType::Blackman, max_boost_db: 24.0,
        noise_floor_db: -150.0, iterations, freq_weighting: true, narrowband_limit: true,
        nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
    }
}

/// Pre-ring of the refined FIR stays within 6 dB of the unrefined one, and
/// the refinement still pays off in the passband magnitude.
fn assert_refine_is_causal(target: &TargetCurve, peq: &[PeqBand], taps: usize, sr: f64, lab: &str) {
    let r0 = generate_band_fir(target, peq, &settings(taps, sr, 0), false).unwrap();
    let r3 = generate_band_fir(target, peq, &settings(taps, sr, 3), false).unwrap();
    let dev = |r: &phaseforge_lib::fir::band::BandFirResult| (0..r.freq.len())
        .filter(|&i| r.freq[i] >= 20.0 && r.freq[i] <= 2000.0)
        .fold((0.0f64, 0.0f64), |m, i| (m.0.max(r.dev_mag[i].abs()), m.1.max(r.dev_phase[i].abs())));
    let (p0, p3) = (pre_db(&r0.fir.impulse, sr, 10.0, 200.0), pre_db(&r3.fir.impulse, sr, 10.0, 200.0));
    let (d0, d3) = (dev(&r0), dev(&r3));
    println!("{lab}: pre-ring 10–200 ms {p0:.1} → {p3:.1} dB; dev mag {:.3} → {:.3} dB, phase {:.2} → {:.2}°",
        d0.0, d3.0, d0.1, d3.1);
    assert!(p3 <= p0 + 6.0, "{lab}: refinement added pre-ring: {p0:.1} → {p3:.1} dB");
    // Magnitude must improve; phase may move by under 1° (mixed phase: the
    // correction's own Hilbert at a sharp LF PEQ, 1.5 → 2.2° at 26 Hz).
    assert!(d3.0 <= d0.0 && d3.1 <= d0.1 + 1.0, "{lab}: refinement made the passband worse");
}

fn lp(order: u8, f: f64) -> FilterConfig {
    FilterConfig { filter_type: FilterType::LinkwitzRiley, order, freq_hz: f, shape: None,
        linear_phase: true, q: None, subsonic_protect: None, complementary: false }
}

#[test]
fn refinement_adds_no_pre_ring_synthetic() {
    let target = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: None, low_pass: Some(lp(2, 250.0)), low_shelf: None, high_shelf: None };
    let peq: Vec<PeqBand> = serde_json::from_str(r#"[
        {"freq_hz": 23.0, "gain_db": 6.0, "q": 2.7, "enabled": true},
        {"freq_hz": 37.0, "gain_db": 1.8, "q": 2.0, "enabled": true}]"#).unwrap();
    assert_refine_is_causal(&target, &peq, 65536, 48_000.0, "synthetic 48k/64K");
}

/// b141.86: mixed per-filter phase (HP min + LP linear) — the refinement
/// correction used to ride the fixed model phase zero-phase.
#[test]
fn refinement_adds_no_pre_ring_mixed_phase() {
    let hp = FilterConfig { linear_phase: false, ..lp(4, 15.0) };
    let target = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: Some(hp), low_pass: Some(lp(2, 250.0)), low_shelf: None, high_shelf: None };
    let peq: Vec<PeqBand> = serde_json::from_str(r#"[
        {"freq_hz": 23.0, "gain_db": 6.0, "q": 2.7, "enabled": true},
        {"freq_hz": 37.0, "gain_db": 1.8, "q": 2.0, "enabled": true}]"#).unwrap();
    assert_refine_is_causal(&target, &peq, 65536, 48_000.0, "mixed HP min + LP lin 48k/64K");
}

#[test]
fn refinement_adds_no_pre_ring_real_woofer() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-fixtures/prering/3way_bd25.pfproj");
    if !p.exists() { eprintln!("skip: {} missing", p.display()); return; }
    let proj: ProjectFile = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let band = &proj.bands[0];
    assert_refine_is_causal(&band.target, &band.peq_bands, proj.export_taps as usize,
        proj.export_sample_rate as f64, "real woofer 352.8k/256K");
}
