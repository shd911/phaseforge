pub mod dsp;
pub mod error;
pub mod export;
pub mod fir;
pub mod io;
pub mod peq;
pub mod phase;
pub mod project;
pub mod recent;
pub mod target;

use std::path::PathBuf;

use dsp::SmoothingConfig;
use dsp::impulse::ImpulseResult;
use dsp::baffle::{BaffleConfig, BaffleStepPreview};
use dsp::merge::{MergeConfig, MergeResult};
use io::Measurement;
use fir::{FirConfig, FirResult, FirModelResult};
use peq::{ExclusionZone, PeqBand, PeqConfig, PeqResult};
use target::{TargetCurve, TargetResponse};
use tracing::info;

#[tauri::command]
fn import_measurement(path: String) -> Result<Measurement, String> {
    info!("import_measurement: {}", path);
    let p = PathBuf::from(&path);
    io::import_measurement(&p).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_smoothed(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
    config: SmoothingConfig,
) -> Result<Vec<f64>, String> {
    info!("get_smoothed: {} points", freq.len());
    Ok(dsp::variable_smoothing(&freq, &magnitude, &config))
}

#[tauri::command]
fn interpolate_log(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
    phase: Option<Vec<f64>>,
    n_points: usize,
    f_min: f64,
    f_max: f64,
) -> Result<(Vec<f64>, Vec<f64>, Option<Vec<f64>>), String> {
    Ok(dsp::interpolate_log_grid(
        &freq,
        &magnitude,
        phase.as_deref(),
        n_points,
        f_min,
        f_max,
    ))
}

#[tauri::command]
fn evaluate_target(target: TargetCurve, freq: Vec<f64>) -> Result<TargetResponse, String> {
    info!("evaluate_target: {} points", freq.len());
    Ok(target::evaluate(&target, &freq))
}

#[tauri::command]
fn evaluate_target_standalone(
    target: TargetCurve,
    n_points: Option<usize>,
    f_min: Option<f64>,
    f_max: Option<f64>,
) -> Result<(Vec<f64>, TargetResponse), String> {
    let n = n_points.unwrap_or(512);
    let fmin = f_min.unwrap_or(20.0);
    let fmax = f_max.unwrap_or(20000.0);
    let freq = dsp::generate_log_freq_grid(n, fmin, fmax);
    let response = target::evaluate(&target, &freq);
    info!("evaluate_target_standalone: {} points", n);
    Ok((freq, response))
}

#[tauri::command]
fn compute_delay_info(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
    phase: Vec<f64>,
    sample_rate: Option<f64>,
) -> Result<(f64, f64), String> {
    let delay = estimate_delay(&freq, &magnitude, &phase, sample_rate);
    let distance = phase::compute_distance(delay);
    info!("compute_delay_info: delay={:.4}ms  dist={:.3}m", delay * 1000.0, distance);
    Ok((delay, distance))
}

#[tauri::command]
fn remove_measurement_delay(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
    phase: Vec<f64>,
    sample_rate: Option<f64>,
) -> Result<(Vec<f64>, f64, f64), String> {
    let delay = estimate_delay(&freq, &magnitude, &phase, sample_rate);
    let distance = phase::compute_distance(delay);
    let new_phase = phase::remove_delay(&freq, &phase, delay);
    // Check for overcorrection
    let excess_gd = phase::check_overcorrection(&freq, &new_phase);
    if excess_gd > 0.0002 {
        info!("remove_measurement_delay: WARNING overcorrection detected, excess GD={:.3}ms", excess_gd * 1000.0);
    }
    info!("remove_measurement_delay: removed {:.4}ms ({:.3}m)", delay * 1000.0, distance);
    Ok((new_phase, delay, distance))
}

/// Apply a user-specified delay (manual override).
#[tauri::command]
fn apply_manual_delay(
    freq: Vec<f64>,
    phase: Vec<f64>,
    delay_seconds: f64,
) -> Result<Vec<f64>, String> {
    let new_phase = phase::remove_delay(&freq, &phase, delay_seconds);
    let distance = phase::compute_distance(delay_seconds);
    info!("apply_manual_delay: {:.4}ms ({:.3}m)", delay_seconds * 1000.0, distance);
    Ok(new_phase)
}

/// Shared delay estimation logic
fn estimate_delay(freq: &[f64], magnitude: &[f64], phase: &[f64], sample_rate: Option<f64>) -> f64 {
    if let Some(sr) = sample_rate {
        let ir_delay = phase::compute_ir_delay(freq, magnitude, phase, sr);
        // Cross-validate with LS fit
        let f_first = freq.first().copied().unwrap_or(20.0);
        let f_last = freq.last().copied().unwrap_or(20000.0);
        let (f_lo, f_hi) = phase::smart_delay_range(f_first, f_last);
        let ls_delay = phase::compute_average_delay(freq, phase, f_lo, f_hi);
        // If IR and LS disagree by >50%, prefer LS (more robust for single drivers)
        let ratio = if ir_delay.abs() > 1e-6 { (ir_delay - ls_delay).abs() / ir_delay.abs() } else { 0.0 };
        if ratio > 0.5 {
            info!("estimate_delay: IR={:.4}ms vs LS={:.4}ms — disagree ({:.0}%), using LS",
                ir_delay * 1000.0, ls_delay * 1000.0, ratio * 100.0);
            ls_delay
        } else {
            info!("estimate_delay: IR={:.4}ms (LS={:.4}ms, agree)", ir_delay * 1000.0, ls_delay * 1000.0);
            ir_delay
        }
    } else {
        let f_first = freq.first().copied().unwrap_or(20.0);
        let f_last = freq.last().copied().unwrap_or(20000.0);
        let (f_lo, f_hi) = phase::smart_delay_range(f_first, f_last);
        let delay = phase::compute_average_delay(freq, phase, f_lo, f_hi);
        info!("estimate_delay: LS fit {:.4}ms (range {:.0}-{:.0} Hz)", delay * 1000.0, f_lo, f_hi);
        delay
    }
}

#[tauri::command]
fn compute_impulse(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
    phase: Vec<f64>,
    sample_rate: Option<f64>,
) -> Result<ImpulseResult, String> {
    let sr = sample_rate.unwrap_or(48000.0);
    info!("compute_impulse: {} points, sr={}", freq.len(), sr);
    Ok(dsp::impulse::compute_impulse_response(&freq, &magnitude, &phase, sr))
}

/// Compute minimum phase from magnitude spectrum via Hilbert transform.
/// Input: freq (log grid), magnitude (dB). Output: phase (degrees) on same grid.
#[tauri::command]
fn compute_minimum_phase(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
) -> Result<Vec<f64>, String> {
    let n = freq.len();
    if n == 0 { return Err("empty freq".into()); }
    let n_fft = ((n * 4).max(4096)).next_power_of_two();
    let n_bins = n_fft / 2 + 1;
    let nyquist = freq.last().copied().unwrap_or(24000.0);

    // Clamp magnitude to reasonable dynamic range to avoid Hilbert artifacts
    let mag_peak = magnitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mag_floor = mag_peak - 80.0; // -80 dB dynamic range
    let clamped: Vec<f64> = magnitude.iter().map(|&v| v.max(mag_floor)).collect();

    // Resample magnitude from log freq grid onto linear FFT grid
    let mut lin_mag = vec![clamped[0]; n_bins];
    for k in 0..n_bins {
        let f_lin = nyquist * k as f64 / (n_bins - 1) as f64;
        if f_lin <= freq[0] {
            lin_mag[k] = clamped[0];
        } else if f_lin >= *freq.last().expect("freq non-empty (checked at line 176)") {
            lin_mag[k] = *clamped.last().expect("clamped same len as freq");
        } else {
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if freq[mid] <= f_lin { lo = mid; } else { hi = mid; }
            }
            let dt = freq[hi] - freq[lo];
            let frac = if dt > 0.0 { (f_lin - freq[lo]) / dt } else { 0.0 };
            lin_mag[k] = clamped[lo] + frac * (clamped[hi] - clamped[lo]);
        }
    }

    let min_ph_rad = dsp::minimum_phase_from_magnitude(&lin_mag, n_fft);

    // Resample phase from linear grid back to log freq grid, convert to degrees
    let mut phase_deg = Vec::with_capacity(n);
    for i in 0..n {
        let bin_f = freq[i] / nyquist * (n_bins - 1) as f64;
        let lo = (bin_f as usize).min(n_bins - 2);
        let hi = lo + 1;
        let frac = bin_f - lo as f64;
        let ph_rad = min_ph_rad[lo] * (1.0 - frac) + min_ph_rad[hi] * frac;
        phase_deg.push(ph_rad.to_degrees());
    }
    Ok(phase_deg)
}

/// Compute corrected impulse: measurement convolved with FIR correction.
/// Adds realized_mag/phase to interpolated measurement, then IFFT.
#[tauri::command]
fn compute_corrected_impulse(
    meas_freq: Vec<f64>,
    meas_mag: Vec<f64>,
    meas_phase: Vec<f64>,
    realized_mag: Vec<f64>,
    realized_phase: Vec<f64>,
    fir_freq: Vec<f64>,
    sample_rate: f64,
) -> Result<ImpulseResult, String> {
    let n = fir_freq.len();
    if n == 0 || realized_mag.len() != n || realized_phase.len() != n {
        return Err("corrected_impulse: freq/mag/phase length mismatch".into());
    }
    // Interpolate measurement onto FIR freq grid
    // Outside measurement range: decay to -200 dB (for magnitude) or 0 (for phase)
    let interp_mag = |f: f64, src_f: &[f64], src_d: &[f64]| -> f64 {
        if src_f.is_empty() { return -200.0; }
        if f < src_f[0] { return -200.0; }
        if f > src_f[src_f.len() - 1] { return -200.0; }
        let idx = src_f.partition_point(|&x| x <= f);
        if idx == 0 { return src_d[0]; }
        if idx >= src_f.len() { return src_d[src_f.len() - 1]; }
        let t = (f - src_f[idx - 1]) / (src_f[idx] - src_f[idx - 1]);
        src_d[idx - 1] + t * (src_d[idx] - src_d[idx - 1])
    };
    let interp_phase = |f: f64, src_f: &[f64], src_d: &[f64]| -> f64 {
        if src_f.is_empty() { return 0.0; }
        if f < src_f[0] { return src_d[0]; }
        if f > src_f[src_f.len() - 1] { return src_d[src_f.len() - 1]; }
        let idx = src_f.partition_point(|&x| x <= f);
        if idx == 0 { return src_d[0]; }
        if idx >= src_f.len() { return src_d[src_f.len() - 1]; }
        let t = (f - src_f[idx - 1]) / (src_f[idx] - src_f[idx - 1]);
        src_d[idx - 1] + t * (src_d[idx] - src_d[idx - 1])
    };
    let corr_mag: Vec<f64> = fir_freq.iter().enumerate()
        .map(|(i, &f)| interp_mag(f, &meas_freq, &meas_mag) + realized_mag[i])
        .collect();
    let corr_phase: Vec<f64> = fir_freq.iter().enumerate()
        .map(|(i, &f)| interp_phase(f, &meas_freq, &meas_phase) + realized_phase[i])
        .collect();
    info!("compute_corrected_impulse: {} points, sr={}", n, sample_rate);
    Ok(dsp::impulse::compute_impulse_response(&fir_freq, &corr_mag, &corr_phase, sample_rate))
}

#[tauri::command]
fn merge_measurements(
    nf_path: String,
    ff_path: String,
    config: MergeConfig,
) -> Result<MergeResult, String> {
    info!("merge_measurements: NF={}, FF={}", nf_path, ff_path);
    let nf = io::import_measurement(&PathBuf::from(&nf_path)).map_err(|e| e.to_string())?;
    let ff = io::import_measurement(&PathBuf::from(&ff_path)).map_err(|e| e.to_string())?;
    dsp::merge::merge_nf_ff(&nf, &ff, &config).map_err(|e| e.to_string())
}

#[tauri::command]
fn preview_baffle_step(config: BaffleConfig) -> Result<BaffleStepPreview, String> {
    let freq = dsp::generate_log_freq_grid(256, 10.0, 24000.0);
    let result =
        dsp::baffle::compute_baffle_step(&freq, &config).map_err(|e| e.to_string())?;
    Ok(BaffleStepPreview {
        freq,
        correction_db: result.correction_mag_db,
        f3_hz: result.f3_hz,
        edge_frequencies: result.edge_frequencies,
    })
}

// ---------------------------------------------------------------------------
// Auto Align: PEQ commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn auto_peq(
    freq: Vec<f64>,
    measurement_mag: Vec<f64>,
    target_mag: Vec<f64>,
    config: PeqConfig,
) -> Result<PeqResult, String> {
    info!("auto_peq: {} points, range={:?}", freq.len(), config.freq_range);
    peq::auto_peq(&measurement_mag, &target_mag, &freq, &config).map_err(|e| e.to_string())
}

#[tauri::command]
fn auto_peq_above_lp(
    freq: Vec<f64>,
    measurement_mag: Vec<f64>,
    config: PeqConfig,
    lp_freq: f64,
    hp_freq: f64,
) -> Result<PeqResult, String> {
    info!(
        "auto_peq_above_lp: {} points, lp={}, hp={}",
        freq.len(), lp_freq, hp_freq
    );
    peq::auto_peq_above_lp(&measurement_mag, &freq, &config, lp_freq, hp_freq)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn auto_peq_lma(
    freq: Vec<f64>,
    measurement_mag: Vec<f64>,
    target_mag: Option<Vec<f64>>,
    config: PeqConfig,
    hp_freq: f64,
    lp_freq: f64,
    exclusion_zones: Option<Vec<ExclusionZone>>,
) -> Result<PeqResult, String> {
    let zones = exclusion_zones.unwrap_or_default();
    info!(
        "auto_peq_lma: {} points, hp={}, lp={}, range={:?}, exclusions={}",
        freq.len(), hp_freq, lp_freq, config.freq_range, zones.len()
    );
    peq::auto_peq_lma(
        &measurement_mag,
        target_mag.as_deref(),
        &freq,
        &config,
        hp_freq,
        lp_freq,
        &zones,
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn compute_peq_response(
    freq: Vec<f64>,
    bands: Vec<PeqBand>,
    sample_rate: Option<f64>,
) -> Result<Vec<f64>, String> {
    let sr = sample_rate.unwrap_or(48000.0);
    info!("compute_peq_response: {} points, {} bands, sr={}", freq.len(), bands.len(), sr);
    Ok(peq::apply_peq(&freq, &bands, sr))
}

#[tauri::command]
fn compute_peq_complex(
    freq: Vec<f64>,
    bands: Vec<PeqBand>,
    sample_rate: Option<f64>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let sr = sample_rate.unwrap_or(48000.0);
    info!("compute_peq_complex: {} points, {} bands, sr={}", freq.len(), bands.len(), sr);
    Ok(peq::apply_peq_complex(&freq, &bands, sr))
}

// ---------------------------------------------------------------------------
// Cross-section: apply user filters + min-phase makeup where corrected < target
// ---------------------------------------------------------------------------

/// Compute cross-section correction for the corrected curve.
///
/// 1. Apply HP/LP filters as configured (LR/BW/Bessel, lin-φ or min-φ).
/// 2. Compute `corrected = meas_mag + peq_correction + filter_response`.
/// 3. Where corrected < target outside the filter band → generate minimum-phase
///    makeup gain to bring corrected up to target.
/// 4. Return total (filter + makeup) magnitude and phase corrections.
#[tauri::command]
fn compute_cross_section(
    freq: Vec<f64>,
    high_pass: Option<target::FilterConfig>,
    low_pass: Option<target::FilterConfig>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let n = freq.len();
    if n == 0 {
        return Err("cross_section: empty freq".into());
    }

    // Step 1: compute user-specified filter response (as-is)
    let mut filt_mag = vec![0.0_f64; n];
    let mut filt_phase = vec![0.0_f64; n];
    if let Some(hp) = &high_pass {
        target::apply_filter_public(&mut filt_mag, &mut filt_phase, &freq, hp, false);
    }
    if let Some(lp) = &low_pass {
        target::apply_filter_public(&mut filt_mag, &mut filt_phase, &freq, lp, true);
    }

    // Return filter-only response (no makeup).
    // Makeup was a preview artifact — FIR export recomputes correction from scratch.
    // Showing meas + PEQ + filt (without makeup) gives honest preview of what filters do.

    // Normalization estimate: peak of filter correction
    let norm_db = filt_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let norm_db = if norm_db.is_finite() && norm_db > 0.0 { norm_db } else { 0.0 };

    info!("compute_cross_section: {} points, hp={}, lp={}, norm_db={:.1}",
        n, high_pass.is_some(), low_pass.is_some(), norm_db);

    Ok((filt_mag, filt_phase, norm_db))
}

// ---------------------------------------------------------------------------
// Auto Align: FIR commands
// ---------------------------------------------------------------------------

#[tauri::command]
fn generate_fir(
    freq: Vec<f64>,
    meas_mag: Vec<f64>,
    target_mag: Vec<f64>,
    peq_correction: Vec<f64>,
    config: FirConfig,
    crossover_range: (f64, f64),
) -> Result<FirResult, String> {
    info!(
        "generate_fir: {} points, taps={}, sr={}",
        freq.len(), config.taps, config.sample_rate
    );
    fir::generate_fir(&freq, &meas_mag, &target_mag, &peq_correction, &config, crossover_range)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn recommend_fir_taps(lowest_freq: f64, sample_rate: f64) -> Result<usize, String> {
    Ok(fir::recommend_taps(lowest_freq, sample_rate))
}

#[tauri::command]
fn generate_hybrid_fir(
    freq: Vec<f64>,
    meas_mag: Vec<f64>,
    target_mag: Vec<f64>,
    peq_correction: Vec<f64>,
    config: FirConfig,
    crossover_range: (f64, f64),
) -> Result<FirModelResult, String> {
    info!(
        "generate_hybrid_fir: {} points, taps={}, sr={}",
        freq.len(), config.taps, config.sample_rate
    );
    fir::generate_hybrid_fir(&freq, &meas_mag, &target_mag, &peq_correction, &config, crossover_range)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn generate_model_fir(
    freq: Vec<f64>,
    target_mag: Vec<f64>,
    peq_mag: Vec<f64>,
    model_phase: Vec<f64>,
    config: FirConfig,
) -> Result<FirModelResult, String> {
    info!(
        "generate_model_fir: {} points, taps={}, sr={}, phase_mode={:?}, peq_points={}",
        freq.len(), config.taps, config.sample_rate, config.phase_mode, peq_mag.len()
    );
    fir::generate_model_fir(&freq, &target_mag, &peq_mag, &model_phase, &config)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn export_fir_wav(impulse: Vec<f64>, sample_rate: f64, path: String) -> Result<(), String> {
    info!("export_fir_wav: {} samples, sr={}, path={}", impulse.len(), sample_rate, path);
    let p = PathBuf::from(&path);
    fir::export_wav_f64(&impulse, sample_rate, &p).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("PhaseForge v0.1.0-b105 starting...");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            import_measurement,
            get_smoothed,
            interpolate_log,
            evaluate_target,
            evaluate_target_standalone,
            compute_delay_info,
            remove_measurement_delay,
            apply_manual_delay,
            compute_impulse,
            compute_minimum_phase,
            compute_corrected_impulse,
            merge_measurements,
            preview_baffle_step,
            auto_peq,
            auto_peq_above_lp,
            auto_peq_lma,
            compute_peq_response,
            compute_peq_complex,
            compute_cross_section,
            generate_fir,
            generate_hybrid_fir,
            generate_model_fir,
            recommend_fir_taps,
            export_fir_wav,
            project::save_project,
            project::load_project,
            project::create_project_folder,
            project::copy_file_to_project,
            project::check_path_exists,
            project::ensure_dir,
            project::copy_dir_contents,
            export::export_target_txt,
            recent::load_recent_projects,
            recent::add_recent_project,
            recent::clear_recent_projects,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
