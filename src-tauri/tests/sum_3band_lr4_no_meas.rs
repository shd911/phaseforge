//! b140.15.11 — exact reproduction of the user's flat.pfproj scenario.
//!
//! Loaded from test-fixtures/user_flat_3band.pfproj:
//!   - 3 bands, all measurement=None
//!   - B0: LP=LR4@200, no HP, ref=0, tilt=0, no shelves, no PEQ, delay=0
//!   - B1: HP=LR4@200 + LP=LR4@2000, same
//!   - B2: HP=LR4@2000, no LP, same
//!
//! Without measurement, sum.ts only computes sumTargetMag/Phase
//! (correctedMag = null for all bands). perBandTargetData is built
//! via evaluate_target Tauri → target::evaluate() → apply_filter_complex
//! (b140.15.9 fix), wrapped to (-180, 180].
//!
//! User screenshot 2026-05-25 shows a 1-bin spike at ~110-115 Hz on
//! the SUM target phase. This test mirrors the EXACT flow (no
//! measurement contribution, direct target::evaluate per band, coherent
//! sum). If it passes, the bug is in the rendering chain (sum.ts
//! resampleOntoCommon / chart rendering) — not in Rust math.

use phaseforge_lib::target::{
    self, FilterConfig as TargetFilterConfig, FilterType as TargetFilterType, TargetCurve,
};

fn lr4(freq_hz: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::LinkwitzRiley,
        order: 4, freq_hz, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn band_target(hp: Option<TargetFilterConfig>, lp: Option<TargetFilterConfig>) -> TargetCurve {
    TargetCurve {
        reference_level_db: 0.0,
        tilt_db_per_octave: 0.0,
        tilt_ref_freq: 1000.0,
        high_pass: hp,
        low_pass: lp,
        low_shelf: None,
        high_shelf: None,
    }
}

/// Common grid matching what buildCommonGrid produces when no band has a
/// measurement: fallback 5..40000 Hz, 512 log points.
fn common_grid() -> Vec<f64> {
    let n = 512;
    (0..n).map(|i| 5.0_f64 * (40000.0_f64 / 5.0).powf(i as f64 / (n - 1) as f64)).collect()
}

/// Coherent sum mirroring sum.ts coherentSum (b140.15.6 mod-360 trig path).
/// All bands have sign=1, delay=0, no alignmentDelay.
fn coherent_sum(freq: &[f64], bands: &[(Vec<f64>, Vec<f64>)]) -> (Vec<f64>, Vec<f64>) {
    let n = freq.len();
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for (mag, phase) in bands {
        for j in 0..n {
            let amp = 10.0_f64.powf(mag[j] / 20.0);
            let p_raw = phase[j];
            let p_mod = p_raw - 360.0 * (p_raw / 360.0).round();
            let p_rad = p_mod.to_radians();
            re[j] += amp * p_rad.cos();
            im[j] += amp * p_rad.sin();
        }
    }
    let mut sum_mag = vec![0.0_f64; n];
    let mut sum_phase = vec![0.0_f64; n];
    for j in 0..n {
        let a = (re[j] * re[j] + im[j] * im[j]).sqrt();
        sum_mag[j] = if a > 1e-15 { 20.0 * a.log10() } else { -300.0 };
        sum_phase[j] = if a > 1e-15 { im[j].atan2(re[j]).to_degrees() } else { 0.0 };
    }
    (sum_mag, sum_phase)
}

#[test]
fn sum_3band_lr4_no_meas_matches_user_flat_pfproj() {
    let freq = common_grid();

    let b0_target = band_target(None,                 Some(lr4(200.0)));
    let b1_target = band_target(Some(lr4(200.0)),     Some(lr4(2000.0)));
    let b2_target = band_target(Some(lr4(2000.0)),    None);

    // target::evaluate() = same Tauri "evaluate_target" command path
    let b0 = target::evaluate(&b0_target, &freq);
    let b1 = target::evaluate(&b1_target, &freq);
    let b2 = target::evaluate(&b2_target, &freq);

    // freq[i] = 5 * 8000^(i/511). i for 110 Hz: ln(22)/ln(8000) ≈ 0.344, × 511 ≈ 176
    eprintln!("\n=== PER-BAND TARGET (around 110-115 Hz, bins 170-185) ===");
    for i in 170..186 {
        eprintln!("  bin {} f={:.2}Hz | B0 mag={:.3} ph={:.2}° | B1 mag={:.3} ph={:.2}° | B2 mag={:.3} ph={:.2}°",
            i, freq[i],
            b0.magnitude[i], b0.phase[i],
            b1.magnitude[i], b1.phase[i],
            b2.magnitude[i], b2.phase[i],
        );
    }

    let bands = vec![
        (b0.magnitude.clone(), b0.phase.clone()),
        (b1.magnitude.clone(), b1.phase.clone()),
        (b2.magnitude.clone(), b2.phase.clone()),
    ];
    let (sum_mag, sum_phase) = coherent_sum(&freq, &bands);

    eprintln!("\n=== SUM TARGET (same bins) ===");
    for i in 170..186 {
        eprintln!("  bin {} f={:.2}Hz | sum_mag={:.4}dB sum_phase={:.2}°",
            i, freq[i], sum_mag[i], sum_phase[i]);
    }

    // Spike detector
    let mut phase_spikes = Vec::new();
    for j in 1..sum_phase.len() - 1 {
        let a = sum_phase[j - 1];
        let b = sum_phase[j];
        let c = sum_phase[j + 1];
        let wrap = |x: f64| x - 360.0 * (x / 360.0).round();
        if wrap(b - a).abs() > 30.0 && wrap(b - c).abs() > 30.0 && wrap(a - c).abs() < 10.0 {
            phase_spikes.push((j, a, b, c, freq[j]));
        }
    }

    eprintln!("\n=== SPIKES ===");
    eprintln!("phase spikes count: {}", phase_spikes.len());
    for (j, a, b, c, f) in phase_spikes.iter().take(10) {
        eprintln!("  bin {} f={:.2}Hz: prev={:.2} spike={:.2} next={:.2}", j, f, a, b, c);
    }

    // Scan SUM MAG around crossover frequencies to characterise the LR
    // ripple pattern that the user reports as "small negative spikes".
    eprintln!("\n=== SUM MAG near 200 Hz crossover (LR ripple region) ===");
    // freq[i] = 5 × 8000^(i/511). For 200 Hz: ln(40)/ln(8000) × 511 ≈ 211
    let mut max_dip_200 = 0.0_f64;
    let mut max_dip_200_bin = 0;
    for i in 200..230 {
        let dip = -sum_mag[i];
        if dip > max_dip_200 { max_dip_200 = dip; max_dip_200_bin = i; }
    }
    eprintln!("  max dip in bins 200..230 = {:.4} dB at bin {} (f={:.2}Hz)",
        max_dip_200, max_dip_200_bin, freq[max_dip_200_bin]);

    eprintln!("\n=== SUM MAG near 2k Hz crossover ===");
    // For 2000 Hz: ln(400)/ln(8000) × 511 ≈ 340
    let mut max_dip_2k = 0.0_f64;
    let mut max_dip_2k_bin = 0;
    for i in 320..360 {
        let dip = -sum_mag[i];
        if dip > max_dip_2k { max_dip_2k = dip; max_dip_2k_bin = i; }
    }
    eprintln!("  max dip in bins 320..360 = {:.4} dB at bin {} (f={:.2}Hz)",
        max_dip_2k, max_dip_2k_bin, freq[max_dip_2k_bin]);

    assert_eq!(phase_spikes.len(), 0,
        "sumTargetPhase has {} spike(s) — user-reported bug REPRODUCED",
        phase_spikes.len());
}
