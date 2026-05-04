// b140.0 — E2E export pipeline harness.
//
// Mirrors the UI export path (evaluate_target_standalone → apply_peq →
// generate_model_fir on the FIR's standalone wide grid) and reports a PASS/FAIL
// row per acceptance-matrix configuration.
//
// Acceptance criteria (per config), evaluated on the 1 kHz – 10 kHz passband:
//   • mag_err   < 1.0 °  — realised magnitude tracks target+peq (after FIR
//                          0-dB normalisation).
//   • phase_err < 5.0 °  — realised phase tracks the assembled Composite phase
//                          (linear-main → 0 + min-phase subsonic + min-phase
//                          peq; min-main → min-phase target + min-phase peq).
//   • rolloff > 30 dB    — HP attenuation at 7 Hz vs passband peak.
//
// The FIR analyser inside generate_model_fir already subtracts the N/2 linear
// delay when effective_linear is true, so result.realized_phase is the
// "excess" phase comparable directly to the assembled phase. We use it
// instead of FFT-ing the impulse ourselves.
//
// Stable FNV-1a 64-bit hash of the rounded impulse is printed for each row
// so future regressions show up as hash mismatches.

mod common;

use common::{acceptance_configs, ExportConfig};
use phaseforge_lib::dsp;
use phaseforge_lib::fir::{generate_model_fir, FirConfig, FirModelResult, PhaseMode, WindowType};
use phaseforge_lib::target::FilterType;

const TAPS: usize = 65536;
const SR: f64 = 48000.0;
const N: usize = 512;

fn fir_grid() -> Vec<f64> {
    let f_max = (40000.0_f64).min(SR / 2.0 * 0.95);
    dsp::generate_log_freq_grid(N, 5.0, f_max)
}

fn subsonic_cutoff_for(cfg: &ExportConfig) -> Option<f64> {
    cfg.target.high_pass.as_ref().and_then(|hp| {
        if matches!(hp.filter_type, FilterType::Gaussian)
            && hp.subsonic_protect == Some(true)
            && hp.freq_hz > 40.0
        {
            Some(hp.freq_hz / 8.0)
        } else {
            None
        }
    })
}

fn run_export_pipeline(cfg: &ExportConfig) -> FirModelResult {
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

    let fir_cfg = FirConfig {
        taps: TAPS,
        sample_rate: SR,
        max_boost_db: 24.0,
        noise_floor_db: -150.0,
        window: WindowType::Blackman,
        phase_mode: PhaseMode::Composite,
        linear_phase_main: cfg.linear_phase_main,
        subsonic_cutoff_hz: subsonic_cutoff_for(cfg),
        iterations: 3,
        freq_weighting: true,
        narrowband_limit: true,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
        gaussian_min_phase_filters: vec![],
    };

    generate_model_fir(
        &freq,
        &target_resp.magnitude,
        &peq_mag,
        &model_phase,
        &fir_cfg,
    )
    .expect("generate_model_fir failed")
}

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

/// Subsonic Butterworth-8 magnitude in dB on a linear FFT grid. Mirrors
/// fir/helpers.rs:subsonic_mag_db_lin (same formula, can't import — pub(crate)).
fn subsonic_mag_lin(n_bins: usize, sample_rate: f64, cutoff_hz: Option<f64>) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_bins];
    let Some(fc) = cutoff_hz else { return out; };
    let nyquist = sample_rate / 2.0;
    for k in 0..n_bins {
        let f = nyquist * k as f64 / (n_bins - 1) as f64;
        if f <= 0.0 {
            out[k] = -400.0;
            continue;
        }
        let r = (fc / f).powi(16);
        out[k] = -10.0 * (1.0 + r).log10();
    }
    out
}

/// Resample a linear-grid array onto the log freq grid by nearest-bin lookup —
/// matches what generate_model_fir's analyser does internally for FFT bins.
fn linear_to_log(linear_phase: &[f64], log_freq: &[f64], sample_rate: f64) -> Vec<f64> {
    let n_bins = linear_phase.len();
    let n_fft = (n_bins - 1) * 2;
    let bin_hz = sample_rate / n_fft as f64;
    log_freq
        .iter()
        .map(|&f| {
            let bin = ((f / bin_hz).round() as usize).min(n_bins - 1);
            linear_phase[bin]
        })
        .collect()
}

/// Expected phase response per the Composite assembly that generate_model_fir
/// performs internally:
///   target_phase = (linear_main ? 0 : Hilbert(main = target − subsonic))
///                  + Hilbert(subsonic)
///   peq_phase    = Hilbert(peq)
///   total        = target_phase + peq_phase
/// Computed on the linear FFT grid (matching the FIR's own Hilbert), then
/// nearest-bin sampled at the log freq points.
fn expected_phase(cfg: &ExportConfig, freq: &[f64]) -> Vec<f64> {
    let n_bins = TAPS / 2 + 1;

    // Target magnitude (already includes subsonic when subsonic_protect=true).
    let (lin_freq, lin_target_mag, _) =
        dsp::interpolate_linear_grid(freq, &phaseforge_lib::target::evaluate(&cfg.target, freq).magnitude, None, n_bins, SR);
    let _ = lin_freq;

    // PEQ magnitude on the same linear grid (apply_peq_complex is per-bin
    // analytical so calling it directly on grid_freq is fine).
    let lin_grid_freq: Vec<f64> = (0..n_bins).map(|i| (SR / 2.0) * i as f64 / (n_bins - 1) as f64).collect();
    let lin_peq_mag: Vec<f64> = if cfg.peq_bands.is_empty() {
        vec![0.0; n_bins]
    } else {
        phaseforge_lib::peq::apply_peq_complex(&lin_grid_freq, &cfg.peq_bands, SR).0
    };

    // Subsonic magnitude (linear grid, BW8-HP analytical).
    let cutoff = subsonic_cutoff_for(cfg);
    let lin_subsonic_mag = subsonic_mag_lin(n_bins, SR, cutoff);

    // Main = target − subsonic (clamped to noise floor).
    let noise_floor_db = -150.0_f64;
    let lin_main_mag: Vec<f64> = (0..n_bins)
        .map(|k| (lin_target_mag[k] - lin_subsonic_mag[k]).max(noise_floor_db))
        .collect();

    // Hilberts.
    let main_phase = if cfg.linear_phase_main {
        vec![0.0; n_bins]
    } else {
        dsp::minimum_phase_from_magnitude(&lin_main_mag, TAPS)
    };
    let subsonic_phase = if cutoff.is_some() {
        dsp::minimum_phase_from_magnitude(&lin_subsonic_mag, TAPS)
    } else {
        vec![0.0; n_bins]
    };
    let peq_phase = if cfg.peq_bands.is_empty() {
        vec![0.0; n_bins]
    } else {
        dsp::minimum_phase_from_magnitude(&lin_peq_mag, TAPS)
    };

    // Composed phase on linear grid (radians) → degrees on log grid.
    let composed_rad: Vec<f64> = (0..n_bins)
        .map(|k| main_phase[k] + subsonic_phase[k] + peq_phase[k])
        .collect();
    let composed_deg: Vec<f64> = composed_rad.iter().map(|r| r.to_degrees()).collect();
    linear_to_log(&composed_deg, freq, SR)
}

fn wrap_deg(p: f64) -> f64 {
    ((p + 180.0).rem_euclid(360.0)) - 180.0
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
    phase_err_deg: f64,
    rolloff_db: f64,
    pass: bool,
    hash: String,
}

fn evaluate_config(cfg: &ExportConfig) -> E2ERow {
    let result = run_export_pipeline(cfg);
    let freq = fir_grid();

    // Realised: use the FIR's own analyser output (already excess-phase, ie
    // N/2 delay subtracted when effective_linear).
    let real_mag = &result.realized_mag;
    let real_phase = &result.realized_phase;
    let exp_mag = expected_mag(cfg, &freq);
    let exp_phase = expected_phase(cfg, &freq);

    let pb_idx: Vec<usize> = freq
        .iter()
        .enumerate()
        .filter(|(_, &f)| f >= 1000.0 && f <= 10000.0)
        .map(|(i, _)| i)
        .collect();

    // FIR auto-norms passband peak to 0 dB.
    let exp_peak = pb_idx
        .iter()
        .map(|&i| exp_mag[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let mag_err = pb_idx
        .iter()
        .map(|&i| (real_mag[i] - (exp_mag[i] - exp_peak)).abs())
        .fold(0.0_f64, f64::max);

    // Phase err: wrap the difference into [−180°, 180°] so absolute rotation
    // wrap-around does not mask sign-flipped agreement.
    let phase_err = pb_idx
        .iter()
        .map(|&i| wrap_deg(real_phase[i] - exp_phase[i]).abs())
        .fold(0.0_f64, f64::max);

    // HP rolloff sanity at 7 Hz vs passband peak.
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

    let pass = mag_err < 1.0 && phase_err < 5.0;

    E2ERow {
        name: cfg.name.to_string(),
        mag_err_db: mag_err,
        phase_err_deg: phase_err,
        rolloff_db: rolloff,
        pass,
        hash: impulse_hash(&result.impulse),
    }
}

#[test]
fn e2e_acceptance_matrix() {
    let configs = acceptance_configs();
    let mut report = String::new();
    let mut failures = 0usize;

    report.push_str(&format!(
        "{:34} {:>10} {:>10} {:>10} {:>7} {}\n",
        "config", "mag_err", "phase_err", "rolloff", "verdict", "hash"
    ));
    report.push_str(&"-".repeat(96));
    report.push('\n');

    for cfg in &configs {
        let r = evaluate_config(cfg);
        if !r.pass {
            failures += 1;
        }
        report.push_str(&format!(
            "{:34} {:>7.3} dB {:>7.2}°  {:>7.2} dB  {:>7}  {}\n",
            r.name,
            r.mag_err_db,
            r.phase_err_deg,
            r.rolloff_db,
            if r.pass { "PASS" } else { "FAIL" },
            r.hash,
        ));
    }

    eprintln!("\n=== E2E Acceptance Matrix (b140.0 baseline) ===");
    eprintln!("{}", report);
    eprintln!(
        "Phase + magnitude acceptance: {} of {} PASS, {} FAIL.",
        configs.len() - failures,
        configs.len(),
        failures,
    );

    // Per the TZ: do NOT assert failures == 0 here. The matrix is the
    // baseline that scopes b140.1.
}
