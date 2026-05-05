// b140.2.0 — Rust-side numerical reference for SUM coherent aggregation.
//
// evaluateSum (TS, src/lib/band-evaluator.ts) is currently dead code in
// production — the UI's renderSumMode runs an inline pipeline. This test
// pins the coherent-sum math used by both paths so the b140.2.1 swap of
// renderSumMode onto evaluateSum can be validated against a single-source
// numerical reference (this file).
//
// Verifies on synthetic flat-response fixtures:
//   • Two identical bands sum to +6.02 dB (20·log10(2)).
//   • Three identical bands sum to +9.54 dB (20·log10(3)).
//   • One inverted of two → cancellation (deep null).
//   • alignment_delay rotates phase: 0.5 ms · 1 kHz → 180°.
//   • Two bands with 0.5 ms delay difference → destructive sum at 1 kHz.

use std::f64::consts::PI;

#[derive(Clone)]
struct SumBand {
    /// magnitude in dB at each freq (same length as the common grid).
    mag_db: Vec<f64>,
    /// phase in degrees at each freq.
    phase_deg: Vec<f64>,
    /// +1 for normal polarity, −1 for inverted.
    sign: f64,
    /// alignment delay in seconds (positive = delayed band).
    delay_s: f64,
}

/// Coherent sum on a fixed freq grid. Mirrors evaluateSum's `coherentSum`
/// helper bit-for-bit — same dB→amplitude, phase rotation by 360·f·delay,
/// re/im accumulation, back to mag/phase.
fn coherent_sum(freq: &[f64], bands: &[SumBand]) -> (Vec<f64>, Vec<f64>) {
    let n = freq.len();
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for b in bands {
        for j in 0..n {
            let amp = 10f64.powf(b.mag_db[j] / 20.0) * b.sign;
            let ph_rad =
                (b.phase_deg[j] + 360.0 * freq[j] * b.delay_s) * PI / 180.0;
            re[j] += amp * ph_rad.cos();
            im[j] += amp * ph_rad.sin();
        }
    }
    let mut mag = vec![0.0_f64; n];
    let mut phase = vec![0.0_f64; n];
    for j in 0..n {
        let amplitude = (re[j] * re[j] + im[j] * im[j]).sqrt();
        mag[j] = if amplitude > 0.0 { 20.0 * amplitude.log10() } else { -200.0 };
        phase[j] = im[j].atan2(re[j]) * 180.0 / PI;
    }
    (mag, phase)
}

fn log_grid(fmin: f64, fmax: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| fmin * (fmax / fmin).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn flat_band(n: usize) -> SumBand {
    SumBand {
        mag_db: vec![0.0_f64; n],
        phase_deg: vec![0.0_f64; n],
        sign: 1.0,
        delay_s: 0.0,
    }
}

fn nearest_idx(freq: &[f64], target: f64) -> usize {
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, &f) in freq.iter().enumerate() {
        let d = (f - target).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

#[test]
fn sum_two_identical_bands_plus6db() {
    let freq = log_grid(20.0, 20000.0, 512);
    let bands = vec![flat_band(freq.len()), flat_band(freq.len())];
    let (mag, _phase) = coherent_sum(&freq, &bands);
    let idx = nearest_idx(&freq, 1000.0);
    assert!((mag[idx] - 6.0206).abs() < 0.01,
        "two identical bands should sum to +6.02 dB, got {:.4}", mag[idx]);
}

#[test]
fn sum_three_identical_bands_plus95db() {
    let freq = log_grid(20.0, 20000.0, 512);
    let bands = vec![flat_band(freq.len()); 3];
    let (mag, _phase) = coherent_sum(&freq, &bands);
    let idx = nearest_idx(&freq, 1000.0);
    // 20·log10(3) = 9.5424
    assert!((mag[idx] - 9.5424).abs() < 0.01,
        "three identical bands should sum to +9.54 dB, got {:.4}", mag[idx]);
}

#[test]
fn sum_inverted_polarity_cancels() {
    let freq = log_grid(20.0, 20000.0, 512);
    let mut bands = vec![flat_band(freq.len()), flat_band(freq.len())];
    bands[1].sign = -1.0;
    let (mag, _phase) = coherent_sum(&freq, &bands);
    let idx = nearest_idx(&freq, 1000.0);
    // Perfect cancellation across the band — coherent_sum clamps log10(0)
    // to −200 dB.
    assert!(mag[idx] < -100.0,
        "inverted polarity should cancel, got {:.4} dB", mag[idx]);
}

#[test]
fn sum_alignment_delay_rotates_phase_at_1khz() {
    let freq = log_grid(20.0, 20000.0, 512);
    let mut bands = vec![flat_band(freq.len())];
    bands[0].delay_s = 0.0005; // 0.5 ms
    let (_mag, phase) = coherent_sum(&freq, &bands);
    let idx = nearest_idx(&freq, 1000.0);
    // 360 · f · delay → at the actual bin freq (slightly off 1k).
    let expected_unwrapped = 360.0 * freq[idx] * 0.0005;
    let expected_wrapped = ((expected_unwrapped + 180.0).rem_euclid(360.0)) - 180.0;
    let actual = phase[idx];
    let dphase = (actual - expected_wrapped).abs().min((actual - expected_wrapped + 360.0).abs()).min((actual - expected_wrapped - 360.0).abs());
    assert!(dphase < 0.5,
        "0.5 ms delay phase at f={:.2} Hz: expected {:.2}°, got {:.2}°",
        freq[idx], expected_wrapped, actual);
}

#[test]
fn sum_delay_difference_cancels_two_bands() {
    let freq = log_grid(20.0, 20000.0, 512);
    let mut bands = vec![flat_band(freq.len()), flat_band(freq.len())];
    bands[1].delay_s = 0.0005;
    let (mag, _phase) = coherent_sum(&freq, &bands);
    let idx = nearest_idx(&freq, 1000.0);
    // 0.5 ms delay between two unit-amplitude bands → near-180° phase
    // difference at 1 kHz → strong cancellation. Grid quantisation puts
    // 1 k-ish at ~994/1009 Hz; cancellation is ≥ 25 dB.
    assert!(mag[idx] < -25.0,
        "two bands with 0.5 ms relative delay should cancel at 1 kHz, got {:.4} dB",
        mag[idx]);
}

#[test]
fn sum_partial_phase_offset_3db() {
    let freq = log_grid(20.0, 20000.0, 512);
    let mut bands = vec![flat_band(freq.len()), flat_band(freq.len())];
    // 0.25 ms delay → 90° at 1 kHz. Two unit vectors at 0° and 90° sum to √2.
    bands[1].delay_s = 0.00025;
    let (mag, _phase) = coherent_sum(&freq, &bands);
    let idx = nearest_idx(&freq, 1000.0);
    assert!((mag[idx] - 3.01).abs() < 0.05,
        "0.25 ms delay between two bands should sum to +3.01 dB at 1 kHz, got {:.4}",
        mag[idx]);
}
