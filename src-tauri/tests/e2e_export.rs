// b140.0 — E2E export pipeline harness.
//
// Mirrors the UI export path (evaluate_target_standalone → apply_peq →
// generate_model_fir on the FIR's standalone wide grid) and reports a PASS/FAIL
// row per acceptance-matrix configuration. The TZ instructs *not* to assert
// `failures == 0` — this run is the baseline that scopes b140.1.
//
// Acceptance criterion (per config): realised magnitude tracks the target on
// the 1 kHz – 10 kHz passband within 1 dB, after the FIR's automatic 0 dB-peak
// normalisation.
//
// A stable golden-hash (FNV-1a 64-bit) is printed for every PASS row so future
// regressions show up as hash mismatches.

mod common;

use common::{acceptance_configs, ExportConfig};
use num_complex::Complex;
use phaseforge_lib::fir::{generate_model_fir, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::target::FilterType;
use rustfft::FftPlanner;

const TAPS: usize = 65536;
const SR: f64 = 48000.0;
const N: usize = 512;

fn fir_grid() -> Vec<f64> {
    let f_max = (40000.0_f64).min(SR / 2.0 * 0.95);
    phaseforge_lib::dsp::generate_log_freq_grid(N, 5.0, f_max)
}

fn run_export_pipeline(cfg: &ExportConfig) -> Vec<f64> {
    let freq = fir_grid();
    let target_resp = phaseforge_lib::target::evaluate(&cfg.target, &freq);

    let (peq_mag, peq_phase) = if cfg.peq_bands.is_empty() {
        (vec![0.0; freq.len()], vec![0.0; freq.len()])
    } else {
        phaseforge_lib::peq::apply_peq_complex(&freq, &cfg.peq_bands, SR)
    };

    let model_phase: Vec<f64> = target_resp
        .phase
        .iter()
        .zip(peq_phase.iter())
        .map(|(t, p)| t + p)
        .collect();

    let subsonic_cutoff = cfg.target.high_pass.as_ref().and_then(|hp| {
        if matches!(hp.filter_type, FilterType::Gaussian)
            && hp.subsonic_protect == Some(true)
            && hp.freq_hz > 40.0
        {
            Some(hp.freq_hz / 8.0)
        } else {
            None
        }
    });

    let fir_cfg = FirConfig {
        taps: TAPS,
        sample_rate: SR,
        max_boost_db: 24.0,
        noise_floor_db: -150.0,
        window: WindowType::Blackman,
        phase_mode: PhaseMode::Composite,
        linear_phase_main: cfg.linear_phase_main,
        subsonic_cutoff_hz: subsonic_cutoff,
        iterations: 3,
        freq_weighting: true,
        narrowband_limit: true,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
        gaussian_min_phase_filters: vec![],
    };

    let result = generate_model_fir(
        &freq,
        &target_resp.magnitude,
        &peq_mag,
        &model_phase,
        &fir_cfg,
    )
    .expect("generate_model_fir failed");
    result.impulse
}

/// FFT the impulse and resample magnitude/phase onto the FIR log grid via
/// nearest-bin lookup. Cheap and good enough for the pass/fail check —
/// realised passband bins are densely covered at sr=48 kHz, taps=65536.
fn fft_realised(impulse: &[f64], freq: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n_fft = impulse.len();
    let mut buf: Vec<Complex<f64>> = impulse.iter().map(|&x| Complex::new(x, 0.0)).collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(n_fft)
        .process(&mut buf);

    let bin_hz = SR / n_fft as f64;
    let nyq_bin = n_fft / 2;
    let mut mag = Vec::with_capacity(freq.len());
    let mut phase = Vec::with_capacity(freq.len());
    for &f in freq {
        let bin = ((f / bin_hz).round() as usize).min(nyq_bin);
        let amp = buf[bin].norm();
        mag.push(if amp > 1e-20 { 20.0 * amp.log10() } else { -400.0 });
        phase.push(buf[bin].arg().to_degrees());
    }
    (mag, phase)
}

/// Expected magnitude — what the FIR should produce *before* its 0 dB-peak
/// normalisation. Phase reconstruction is not part of the acceptance check
/// here (that's b140.1's scope); we report realised passband phase variation
/// as informational only.
fn expected_mag(cfg: &ExportConfig, freq: &[f64]) -> Vec<f64> {
    let resp = phaseforge_lib::target::evaluate(&cfg.target, freq);
    let mut mag = resp.magnitude;
    if !cfg.peq_bands.is_empty() {
        let (pm, _) = phaseforge_lib::peq::apply_peq_complex(freq, &cfg.peq_bands, SR);
        for i in 0..mag.len() {
            mag[i] += pm[i];
        }
    }
    mag
}

/// FNV-1a 64-bit over the rounded impulse — same scheme as the cargo b139.0
/// golden test so the format is consistent across the project.
fn impulse_hash(impulse: &[f64]) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h: u64 = FNV_OFFSET;
    for &v in impulse {
        let r = (v * 1_000_000.0).round() as i64;
        for byte in r.to_le_bytes().iter() {
            h ^= *byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    format!("{:016x}", h)
}

struct E2ERow {
    name: String,
    mag_err_db: f64,
    phase_var_deg: f64,
    rolloff_db: f64,
    pass: bool,
    hash: String,
}

fn evaluate_config(cfg: &ExportConfig) -> E2ERow {
    let impulse = run_export_pipeline(cfg);
    let freq = fir_grid();
    let (real_mag, real_phase) = fft_realised(&impulse, &freq);
    let exp_mag = expected_mag(cfg, &freq);

    // 1 kHz – 10 kHz passband.
    let mut pb_idx: Vec<usize> = Vec::new();
    for (i, &f) in freq.iter().enumerate() {
        if f >= 1000.0 && f <= 10000.0 {
            pb_idx.push(i);
        }
    }
    // FIR auto-norms passband peak to 0 dB. Recover the same offset from
    // expected_mag so we compare apples to apples.
    let exp_peak = pb_idx
        .iter()
        .map(|&i| exp_mag[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let mag_err = pb_idx
        .iter()
        .map(|&i| (real_mag[i] - (exp_mag[i] - exp_peak)).abs())
        .fold(0.0_f64, f64::max);

    // Realised phase variation across the passband (informational).
    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;
    for &i in &pb_idx {
        let p = ((real_phase[i] + 180.0).rem_euclid(360.0)) - 180.0;
        if p < p_min {
            p_min = p;
        }
        if p > p_max {
            p_max = p;
        }
    }
    let phase_var = p_max - p_min;

    // HP rolloff sanity (any HP config): realised at 7 Hz must be ≥ 30 dB
    // below passband peak. Configs without HP get rolloff = 0.
    let realised_pb_peak = pb_idx
        .iter()
        .map(|&i| real_mag[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let rolloff = {
        let mut idx = 0usize;
        let mut best = f64::INFINITY;
        for (i, &f) in freq.iter().enumerate() {
            let d = (f - 7.0).abs();
            if d < best {
                best = d;
                idx = i;
            }
        }
        realised_pb_peak - real_mag[idx]
    };

    let pass = mag_err < 1.0;

    E2ERow {
        name: cfg.name.to_string(),
        mag_err_db: mag_err,
        phase_var_deg: phase_var,
        rolloff_db: rolloff,
        pass,
        hash: impulse_hash(&impulse),
    }
}

#[test]
fn e2e_acceptance_matrix() {
    let configs = acceptance_configs();
    let mut report = String::new();
    let mut failures = 0usize;

    report.push_str(&format!(
        "{:34} {:>10} {:>11} {:>10} {:>6} {}\n",
        "config", "mag_err", "phase_var", "rolloff", "verdict", "hash"
    ));
    report.push_str(&"-".repeat(96));
    report.push('\n');

    for cfg in &configs {
        let r = evaluate_config(cfg);
        if !r.pass {
            failures += 1;
        }
        report.push_str(&format!(
            "{:34} {:>7.3} dB {:>8.2}°  {:>7.2} dB  {:>6}  {}\n",
            r.name,
            r.mag_err_db,
            r.phase_var_deg,
            r.rolloff_db,
            if r.pass { "PASS" } else { "FAIL" },
            r.hash,
        ));
    }

    eprintln!("\n=== E2E Acceptance Matrix (b140.0 baseline) ===");
    eprintln!("{}", report);
    eprintln!(
        "Baseline: {} of {} configs FAIL on b140.0.",
        failures,
        configs.len()
    );
    eprintln!("b140.1 should bring this to 0/{}.\n", configs.len());

    // Per the TZ: do NOT assert failures == 0 here. The matrix is the
    // baseline that scopes b140.1.
}
