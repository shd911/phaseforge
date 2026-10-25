// b141.111: the crossover ring shown in the ФВЧ/ФНЧ blocks vs the ring
// measured on the exported WAV of that section.
//
// Each section goes through the app's own export path (`generate_band_fir` →
// `export_wav_f64`, 48 kHz / 256K taps, Rectangular so the window does not
// shape the tail), the file is read back from disk, its passband level is
// restored (`norm_db`), an HP's click is removed at the WAV delay (the ring
// is read on 1 − HP, as `section_ring` does) and the envelope is read with the same −60 / −100 dB rule.
// Pre-ring is compared on linear sections only.
use phaseforge_lib::fir::band::{generate_band_fir, BandFirSettings};
use phaseforge_lib::fir::section_ring::{ring_samples, section_ring};
use phaseforge_lib::fir::{export_wav_f64, WindowType};
use phaseforge_lib::target::{FilterConfig, FilterType, TargetCurve};
use std::path::PathBuf;

const SR: f64 = 48_000.0;
const TAPS: usize = 262_144;

fn cfg(ft: FilterType, order: u8, fc: f64, lin: bool, shape: Option<f64>, q: Option<f64>) -> FilterConfig {
    FilterConfig { filter_type: ft, order, freq_hz: fc, shape, linear_phase: lin, q, subsonic_protect: None, complementary: false }
}

fn read_wav_f64(p: &std::path::Path) -> Vec<f64> {
    let b = std::fs::read(p).unwrap();
    let mut i = 12usize;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        if &b[i..i + 4] == b"data" {
            return b[i + 8..i + 8 + len].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        }
        i += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

/// The section's exported WAV read back from disk at its passband level, and its delay.
fn exported(c: &FilterConfig, is_lp: bool, tag: &str) -> (Vec<f64>, usize) {
    let target = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: (!is_lp).then(|| c.clone()), low_pass: is_lp.then(|| c.clone()), low_shelf: None, high_shelf: None };
    let s = BandFirSettings { taps: TAPS, sample_rate: SR, window: WindowType::Rectangular, max_boost_db: 24.0,
        noise_floor_db: -150.0, iterations: 3, freq_weighting: true, narrowband_limit: true,
        nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0 };
    let r = generate_band_fir(&target, &[], &s, false).unwrap();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/section_ring");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{}_{tag}.wav", std::process::id()));
    export_wav_f64(&r.fir.impulse, SR, &path).unwrap();
    let mut h = read_wav_f64(&path);
    let _ = std::fs::remove_file(&path);
    let g = 10f64.powf(r.fir.norm_db / 20.0);
    for v in h.iter_mut() { *v *= g; }
    (h, r.fir.wav_delay_samples)
}

fn measured(c: &FilterConfig, is_lp: bool, tag: &str) -> (f64, f64, f64, f64) {
    let (mut h, d) = exported(c, is_lp, tag);
    if !is_lp { h[d] -= 1.0; }
    // The last 1/32 of the file is left out: a min-phase cepstral FIR wraps
    // its far tail there (−56 dB of the wave at 256K for a Gaussian M1 HP at
    // 80 Hz — the export's own shortfall, not the section's ring).
    let (b60, b100, a60, a100) = ring_samples(&h[..TAPS - TAPS / 32]);
    let ms = |n: usize| n as f64 / SR * 1000.0;
    (ms(b60), ms(b100), ms(a60), ms(a100))
}

#[test]
fn section_ring_matches_the_exported_impulse() {
    let cases = [
        ("BW4 ФВЧ 100 Гц min", cfg(FilterType::Butterworth, 4, 100.0, false, None, None), false),
        ("LR4 ФНЧ 2 кГц min", cfg(FilterType::LinkwitzRiley, 4, 2000.0, false, None, None), true),
        ("Bessel4 ФВЧ 500 Гц min", cfg(FilterType::Bessel, 4, 500.0, false, None, None), false),
        ("Q0.50·2 ФНЧ 300 Гц min", cfg(FilterType::Custom, 2, 300.0, false, None, Some(0.5)), true),
        ("Gauss M1 ФВЧ 80 Гц min", cfg(FilterType::Gaussian, 1, 80.0, false, Some(1.0), None), false),
        ("BW8 ФВЧ 40 Гц lin", cfg(FilterType::Butterworth, 8, 40.0, true, None, None), false),
        ("LR2 ФНЧ 500 Гц lin", cfg(FilterType::LinkwitzRiley, 2, 500.0, true, None, None), true),
        ("Gauss M2 ФНЧ 100 Гц lin", cfg(FilterType::Gaussian, 1, 100.0, true, Some(2.0), None), true),
    ];
    let mut fails = Vec::new();
    for (i, (name, c, is_lp)) in cases.iter().enumerate() {
        let calc = section_ring(c, *is_lp, SR).unwrap();
        let (b60, b100, a60, a100) = measured(c, *is_lp, &i.to_string());
        println!("{name}: расчёт после {:.2}/{:.2} мс, до {:.2}/{:.2}; WAV после {a60:.2}/{a100:.2}, до {b60:.2}/{b100:.2}",
            calc.post_60_ms, calc.post_100_ms, calc.pre_60_ms, calc.pre_100_ms);
        let close = |a: f64, b: f64| (a - b).abs() <= (0.05 * b).max(0.3);
        let mut pairs = vec![("после −60", calc.post_60_ms, a60), ("после −100", calc.post_100_ms, a100)];
        if c.linear_phase { pairs.extend([("до −60", calc.pre_60_ms, b60), ("до −100", calc.pre_100_ms, b100)]); }
        for (what, x, m) in pairs {
            if !close(x, m) { fails.push(format!("{name}: {what}: расчёт {x:.2} мс, WAV {m:.2} мс")); }
        }
        assert!(!calc.lower_bound, "{name}: tail did not fit the grid");
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// The ring does not depend on the export rate (the HP click is not the reference).
#[test]
fn section_ring_is_rate_independent() {
    for (c, is_lp) in [(cfg(FilterType::Butterworth, 4, 100.0, false, None, None), false),
        (cfg(FilterType::LinkwitzRiley, 4, 100.0, false, None, None), true)] {
        let (a, b) = (section_ring(&c, is_lp, 48_000.0).unwrap(), section_ring(&c, is_lp, 192_000.0).unwrap());
        println!("{:?} lp={is_lp}: 48k {:.2}/{:.2}, 192k {:.2}/{:.2} мс", c.filter_type, a.post_60_ms, a.post_100_ms, b.post_60_ms, b.post_100_ms);
        assert!((a.post_60_ms / b.post_60_ms - 1.0).abs() < 0.05 && (a.post_100_ms / b.post_100_ms - 1.0).abs() < 0.05);
    }
}
