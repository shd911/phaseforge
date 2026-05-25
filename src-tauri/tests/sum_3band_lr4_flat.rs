//! b140.15.10 regression / fixture test: 3-band LR4 SUM on flat-meas synthetic.
//!
//! Reproduces the user's reported bug ("узкие пики на 120° на стыке") with a
//! deterministic Rust-side simulation of the JS SUM pipeline. Captures the
//! full input/output as a JSON fixture so future iterations can diff against
//! it instead of round-tripping through the UI.
//!
//! Setup (per user description):
//!   - 3 bands, all min-phase LR4 (48 dB/oct)
//!   - B0: LP=200 only
//!   - B1: HP=200 + LP=2000
//!   - B2: HP=2000 only
//!   - flat measurement at 75 dB, ~0° phase per band (~constant delay 222 µs
//!     matching the diagnostic dump from production)
//!   - 512 log-spaced freq bins 20..20000 Hz
//!
//! What the test asserts (post-fix expectations):
//!   1. per-band correctedMag has no 1-bin spike (|Δ vs neighbours| < 1 dB
//!      where neighbours agree within 0.3 dB).
//!   2. per-band correctedPhase has no 1-bin spike (|Δ vs neighbours| < 30°
//!      mod 360° where neighbours agree).
//!   3. sumMag (coherent) has no 1-bin spike under the same rule.
//!   4. sumPhase has no 1-bin spike under the same rule.
//!
//! Dump (regardless of pass/fail): tests/fixtures/sum_3band_lr4_flat.json
//! with the full freq grid, per-band corrected mag/phase, sum mag/phase.
//! Inspect that file when iterating on a fix.

use phaseforge_lib::target::{
    self, FilterConfig as TargetFilterConfig, FilterType as TargetFilterType,
};

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Setup builders
// ---------------------------------------------------------------------------

fn lr4(freq_hz: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::LinkwitzRiley,
        order: 4, freq_hz, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn log_freq_grid() -> Vec<f64> {
    let (f_min, f_max, n) = (20.0_f64, 20_000.0_f64, 512);
    (0..n)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
        .collect()
}

/// Flat synthetic measurement matching what the user loaded:
///   - magnitude = 75.0 dB constant
///   - phase     = -360 · f · delay  (constant time delay ≈ 222 µs derived
///                                    from the diagnostic dump showing
///                                    ~-348° at ~111 Hz)
const FLAT_MEAS_MAG_DB: f64 = 75.0;
const FLAT_MEAS_DELAY_S: f64 = 222e-6;

fn flat_meas_phase(freq: &[f64]) -> Vec<f64> {
    freq.iter().map(|&f| -360.0 * f * FLAT_MEAS_DELAY_S).collect()
}

// ---------------------------------------------------------------------------
// Per-band corrected = measurement + cross-section
// (Mirrors evaluateBandFull's correctedMag/Phase for HP+LP-only bands.)
// ---------------------------------------------------------------------------

struct BandCorrected {
    mag: Vec<f64>,
    phase: Vec<f64>,
}

fn band_corrected(
    freq: &[f64],
    meas_mag: f64,
    meas_phase: &[f64],
    hp: Option<&TargetFilterConfig>,
    lp: Option<&TargetFilterConfig>,
) -> BandCorrected {
    let n = freq.len();
    // Mirror lib.rs compute_cross_section (b140.15.9 complex path).
    let mut xs_mag = vec![0.0_f64; n];
    let mut re_acc = vec![1.0_f64; n];
    let mut im_acc = vec![0.0_f64; n];
    if let Some(hp) = hp {
        target::apply_filter_complex(&mut xs_mag, &mut re_acc, &mut im_acc, freq, hp, false);
    }
    if let Some(lp) = lp {
        target::apply_filter_complex(&mut xs_mag, &mut re_acc, &mut im_acc, freq, lp, true);
    }
    let mut xs_phase = vec![0.0_f64; n];
    target::complex_acc_to_phase_deg(&re_acc, &im_acc, &mut xs_phase);

    // correctedMag = meas_mag + xs_mag (dB add)
    // correctedPhase = meas_phase + xs_phase (degrees)
    let mag: Vec<f64> = xs_mag.iter().map(|&x| meas_mag + x).collect();
    let phase: Vec<f64> = xs_phase.iter().zip(meas_phase.iter())
        .map(|(&x, &m)| m + x).collect();
    BandCorrected { mag, phase }
}

// ---------------------------------------------------------------------------
// Coherent sum (mirror of TS coherentSum, b140.15.6 mod-360 trig path)
// ---------------------------------------------------------------------------

fn coherent_sum(freq: &[f64], bands: &[BandCorrected]) -> (Vec<f64>, Vec<f64>) {
    let n = freq.len();
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for b in bands {
        for j in 0..n {
            let amp = 10.0_f64.powf(b.mag[j] / 20.0);
            // mod-360 reduction matches sum.ts coherentSum
            let p_raw = b.phase[j];
            let p_mod = p_raw - 360.0 * (p_raw / 360.0).round();
            let p_rad = p_mod.to_radians();
            re[j] += amp * p_rad.cos();
            im[j] += amp * p_rad.sin();
        }
    }
    let mut mag = vec![0.0_f64; n];
    let mut phase = vec![0.0_f64; n];
    for j in 0..n {
        let a = (re[j] * re[j] + im[j] * im[j]).sqrt();
        mag[j] = if a > 1e-15 { 20.0 * a.log10() } else { -300.0 };
        phase[j] = if a > 1e-15 { im[j].atan2(re[j]).to_degrees() } else { 0.0 };
    }
    (mag, phase)
}

// ---------------------------------------------------------------------------
// Spike detector — same rule as sum.ts diagnostic
// ---------------------------------------------------------------------------

fn find_spikes(values: &[f64], step_threshold: f64, neighbour_agreement: f64, mod_360: bool) -> Vec<(usize, f64, f64, f64)> {
    let mut spikes = Vec::new();
    for j in 1..values.len() - 1 {
        let a = values[j - 1];
        let b = values[j];
        let c = values[j + 1];
        let (ab, bc, ac) = if mod_360 {
            let wrap = |x: f64| x - 360.0 * (x / 360.0).round();
            (wrap(b - a).abs(), wrap(b - c).abs(), wrap(a - c).abs())
        } else {
            ((b - a).abs(), (b - c).abs(), (a - c).abs())
        };
        if ab > step_threshold && bc > step_threshold && ac < neighbour_agreement {
            spikes.push((j, a, b, c));
        }
    }
    spikes
}

// ---------------------------------------------------------------------------
// Fixture dump
// ---------------------------------------------------------------------------

fn dump_fixture(
    freq: &[f64],
    bands: &[BandCorrected],
    sum_mag: &[f64],
    sum_phase: &[f64],
    mag_spikes_per_band: &[Vec<(usize, f64, f64, f64)>],
    phase_spikes_per_band: &[Vec<(usize, f64, f64, f64)>],
    sum_mag_spikes: &[(usize, f64, f64, f64)],
    sum_phase_spikes: &[(usize, f64, f64, f64)],
) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sum_3band_lr4_flat.json");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"setup\": \"3-band LR4 min-phase 200/2000, flat meas 75dB, delay 222us\",\n"));
    json.push_str(&format!("  \"n_bins\": {},\n", freq.len()));
    json.push_str(&format!("  \"freq_hz\": [{:.4}, ..., {:.4}],\n", freq[0], freq[freq.len()-1]));

    for (i, b) in bands.iter().enumerate() {
        json.push_str(&format!("  \"band_{}\": {{ \"mag_db_min\": {:.3}, \"mag_db_max\": {:.3}, \"phase_min\": {:.2}, \"phase_max\": {:.2} }},\n",
            i,
            b.mag.iter().cloned().fold(f64::INFINITY, f64::min),
            b.mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            b.phase.iter().cloned().fold(f64::INFINITY, f64::min),
            b.phase.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ));
    }

    json.push_str(&format!("  \"sum_mag_db_min\": {:.3},\n", sum_mag.iter().cloned().fold(f64::INFINITY, f64::min)));
    json.push_str(&format!("  \"sum_mag_db_max\": {:.3},\n", sum_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max)));
    json.push_str(&format!("  \"sum_phase_min\": {:.2},\n", sum_phase.iter().cloned().fold(f64::INFINITY, f64::min)));
    json.push_str(&format!("  \"sum_phase_max\": {:.2},\n", sum_phase.iter().cloned().fold(f64::NEG_INFINITY, f64::max)));

    for (i, spikes) in mag_spikes_per_band.iter().enumerate() {
        json.push_str(&format!("  \"band_{}_mag_spikes\": [", i));
        for (k, &(b, a, v, c)) in spikes.iter().enumerate() {
            if k > 0 { json.push_str(", "); }
            json.push_str(&format!("{{\"bin\":{},\"f\":{:.2},\"prev\":{:.3},\"spike\":{:.3},\"next\":{:.3}}}",
                b, freq[b], a, v, c));
        }
        json.push_str("],\n");
    }
    for (i, spikes) in phase_spikes_per_band.iter().enumerate() {
        json.push_str(&format!("  \"band_{}_phase_spikes\": [", i));
        for (k, &(b, a, v, c)) in spikes.iter().enumerate() {
            if k > 0 { json.push_str(", "); }
            json.push_str(&format!("{{\"bin\":{},\"f\":{:.2},\"prev\":{:.2},\"spike\":{:.2},\"next\":{:.2}}}",
                b, freq[b], a, v, c));
        }
        json.push_str("],\n");
    }

    json.push_str(&format!("  \"sum_mag_spikes\": ["));
    for (k, &(b, a, v, c)) in sum_mag_spikes.iter().enumerate() {
        if k > 0 { json.push_str(", "); }
        json.push_str(&format!("{{\"bin\":{},\"f\":{:.2},\"prev\":{:.3},\"spike\":{:.3},\"next\":{:.3}}}",
            b, freq[b], a, v, c));
    }
    json.push_str("],\n");
    json.push_str(&format!("  \"sum_phase_spikes\": ["));
    for (k, &(b, a, v, c)) in sum_phase_spikes.iter().enumerate() {
        if k > 0 { json.push_str(", "); }
        json.push_str(&format!("{{\"bin\":{},\"f\":{:.2},\"prev\":{:.2},\"spike\":{:.2},\"next\":{:.2}}}",
            b, freq[b], a, v, c));
    }
    json.push_str("]\n}\n");
    std::fs::write(&path, json).expect("dump fixture");
    eprintln!("[fixture-dump] wrote {}", path.display());
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

/// b140.15.11: variant fixture — measurement.phase = 0 (no delay).
/// User screenshot 2026-05-25 showed a 1-bin spike at ~110-115 Hz on
/// SUM corrected/target phase. The default fixture (with -222µs delay)
/// passes. This variant verifies whether the spike is specific to
/// zero-phase synthetic input.
#[test]
fn sum_3band_lr4_zerophase_no_spikes() {
    let freq = log_freq_grid();
    let meas_phase: Vec<f64> = vec![0.0; freq.len()];

    let lp_200 = lr4(200.0);
    let hp_200 = lr4(200.0);
    let lp_2000 = lr4(2000.0);
    let hp_2000 = lr4(2000.0);

    let b0 = band_corrected(&freq, FLAT_MEAS_MAG_DB, &meas_phase, None, Some(&lp_200));
    let b1 = band_corrected(&freq, FLAT_MEAS_MAG_DB, &meas_phase, Some(&hp_200), Some(&lp_2000));
    let b2 = band_corrected(&freq, FLAT_MEAS_MAG_DB, &meas_phase, Some(&hp_2000), None);
    let bands = vec![b0, b1, b2];
    let (sum_mag, sum_phase) = coherent_sum(&freq, &bands);

    // Scan for 1-bin phase spikes near LR wrap points (~110-130 Hz, ~1100-1300 Hz)
    let phase_spikes = find_spikes(&sum_phase, 30.0, 10.0, true);
    let mag_spikes = find_spikes(&sum_mag, 1.0, 0.5, false);

    eprintln!("[zerophase variant] sum_mag range [{:.3}, {:.3}], sum_phase range [{:.2}, {:.2}]",
        sum_mag.iter().cloned().fold(f64::INFINITY, f64::min),
        sum_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        sum_phase.iter().cloned().fold(f64::INFINITY, f64::min),
        sum_phase.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    eprintln!("[zerophase variant] mag spikes: {}, phase spikes: {}",
        mag_spikes.len(), phase_spikes.len());
    for s in mag_spikes.iter().take(6) {
        eprintln!("  MAG spike bin {} @ {:.2}Hz: prev={:.3} spike={:.3} next={:.3} dB",
            s.0, freq[s.0], s.1, s.2, s.3);
    }
    for s in phase_spikes.iter().take(6) {
        eprintln!("  PHASE spike bin {} @ {:.2}Hz: prev={:.2} spike={:.2} next={:.2}°",
            s.0, freq[s.0], s.1, s.2, s.3);
    }
    assert_eq!(phase_spikes.len(), 0, "zerophase variant produced {} phase spikes", phase_spikes.len());
    assert_eq!(mag_spikes.len(), 0, "zerophase variant produced {} mag spikes", mag_spikes.len());
}

#[test]
fn sum_3band_lr4_flat_has_no_spikes() {
    let freq = log_freq_grid();
    let meas_phase = flat_meas_phase(&freq);

    // 3 bands
    let lp_200 = lr4(200.0);
    let hp_200 = lr4(200.0);
    let lp_2000 = lr4(2000.0);
    let hp_2000 = lr4(2000.0);

    let b0 = band_corrected(&freq, FLAT_MEAS_MAG_DB, &meas_phase, None, Some(&lp_200));
    let b1 = band_corrected(&freq, FLAT_MEAS_MAG_DB, &meas_phase, Some(&hp_200), Some(&lp_2000));
    let b2 = band_corrected(&freq, FLAT_MEAS_MAG_DB, &meas_phase, Some(&hp_2000), None);

    let bands = vec![b0, b1, b2];
    let (sum_mag, sum_phase) = coherent_sum(&freq, &bands);

    // Tightened thresholds (b140.15.10.1) — user's UI dips are 0.1–0.5
    // dB; the prior 1 dB threshold missed them.
    const MAG_STEP: f64 = 0.05;     // dB
    const MAG_NEIGHBOUR: f64 = 0.02;
    const PHASE_STEP: f64 = 30.0;   // degrees (mod 360)
    const PHASE_NEIGHBOUR: f64 = 10.0;

    let mag_spikes_per_band: Vec<_> = bands.iter()
        .map(|b| find_spikes(&b.mag, MAG_STEP, MAG_NEIGHBOUR, false))
        .collect();
    let phase_spikes_per_band: Vec<_> = bands.iter()
        .map(|b| find_spikes(&b.phase, PHASE_STEP, PHASE_NEIGHBOUR, true))
        .collect();
    let sum_mag_spikes = find_spikes(&sum_mag, MAG_STEP, MAG_NEIGHBOUR, false);
    let sum_phase_spikes = find_spikes(&sum_phase, PHASE_STEP, PHASE_NEIGHBOUR, true);

    dump_fixture(
        &freq, &bands, &sum_mag, &sum_phase,
        &mag_spikes_per_band, &phase_spikes_per_band,
        &sum_mag_spikes, &sum_phase_spikes,
    );

    // b140.15.10.1: scan for SMOOTH dips (not 1-bin spikes). LR property
    // says |LR_LP(fc) + LR_HP(fc)| = 1.0 exactly at every f. Any dip > 0.05
    // dB in sumMag means complex sum is losing magnitude somewhere.
    let baseline = FLAT_MEAS_MAG_DB; // 75 dB
    let mut dips: Vec<(usize, f64)> = Vec::new();
    for (i, &m) in sum_mag.iter().enumerate() {
        if (baseline - m) > 0.05 {
            dips.push((i, baseline - m));
        }
    }
    if !dips.is_empty() {
        eprintln!("[smooth-dips] {} bin(s) deviate >0.05 dB from baseline {} dB:", dips.len(), baseline);
        for &(i, d) in dips.iter().take(20) {
            eprintln!("  bin {} @ {:.2}Hz: sum_mag={:.4} dB (dip {:.3} dB)",
                i, freq[i], sum_mag[i], d);
        }
        let max_dip = dips.iter().map(|&(_, d)| d).fold(0.0_f64, f64::max);
        eprintln!("[smooth-dips] max dip = {:.4} dB", max_dip);
    } else {
        eprintln!("[smooth-dips] sum_mag is flat within 0.05 dB ✓");
    }

    let total_spikes = mag_spikes_per_band.iter().map(Vec::len).sum::<usize>()
        + phase_spikes_per_band.iter().map(Vec::len).sum::<usize>()
        + sum_mag_spikes.len() + sum_phase_spikes.len();

    if total_spikes > 0 {
        let mut msg = String::from("\n=== SPIKE REPORT ===\n");
        for (i, spikes) in mag_spikes_per_band.iter().enumerate() {
            if !spikes.is_empty() {
                msg.push_str(&format!("band {} mag: {} spikes\n", i, spikes.len()));
                for &(b, a, v, c) in spikes.iter().take(6) {
                    msg.push_str(&format!("  bin {} f={:.1}Hz prev={:.3} spike={:.3} next={:.3} dB\n",
                        b, freq[b], a, v, c));
                }
            }
        }
        for (i, spikes) in phase_spikes_per_band.iter().enumerate() {
            if !spikes.is_empty() {
                msg.push_str(&format!("band {} phase: {} spikes\n", i, spikes.len()));
                for &(b, a, v, c) in spikes.iter().take(6) {
                    msg.push_str(&format!("  bin {} f={:.1}Hz prev={:.2} spike={:.2} next={:.2} °\n",
                        b, freq[b], a, v, c));
                }
            }
        }
        if !sum_mag_spikes.is_empty() {
            msg.push_str(&format!("SUM mag: {} spikes\n", sum_mag_spikes.len()));
            for &(b, a, v, c) in sum_mag_spikes.iter().take(6) {
                msg.push_str(&format!("  bin {} f={:.1}Hz prev={:.3} spike={:.3} next={:.3} dB\n",
                    b, freq[b], a, v, c));
            }
        }
        if !sum_phase_spikes.is_empty() {
            msg.push_str(&format!("SUM phase: {} spikes\n", sum_phase_spikes.len()));
            for &(b, a, v, c) in sum_phase_spikes.iter().take(6) {
                msg.push_str(&format!("  bin {} f={:.1}Hz prev={:.2} spike={:.2} next={:.2} °\n",
                    b, freq[b], a, v, c));
            }
        }
        msg.push_str("\nFull dump: tests/fixtures/sum_3band_lr4_flat.json\n");
        panic!("{}", msg);
    }
}
