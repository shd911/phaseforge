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
use peq::{PeqBand, PeqConfig, PeqResult};
use target::{TargetCurve, TargetResponse};
use tracing::info;

#[tauri::command]
fn greet(name: String) -> Result<String, String> {
    info!("greet called with name: {}", name);
    Ok(format!("Hello from PhaseForge, {}!", name))
}

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
    phase: Vec<f64>,
    f_low: Option<f64>,
    f_high: Option<f64>,
) -> Result<(f64, f64), String> {
    let fl = f_low.unwrap_or(1000.0);
    let fh = f_high.unwrap_or(4000.0);
    let delay = phase::compute_average_delay(&freq, &phase, fl, fh);
    let distance = phase::compute_distance(delay);
    info!("compute_delay_info: delay={:.4}ms  dist={:.3}m", delay * 1000.0, distance);
    Ok((delay, distance))
}

#[tauri::command]
fn remove_measurement_delay(
    freq: Vec<f64>,
    phase: Vec<f64>,
    f_low: Option<f64>,
    f_high: Option<f64>,
) -> Result<(Vec<f64>, f64, f64), String> {
    let fl = f_low.unwrap_or(1000.0);
    let fh = f_high.unwrap_or(4000.0);
    let delay = phase::compute_average_delay(&freq, &phase, fl, fh);
    let distance = phase::compute_distance(delay);
    let new_phase = phase::remove_delay(&freq, &phase, delay);
    info!("remove_measurement_delay: removed {:.4}ms ({:.3}m)", delay * 1000.0, distance);
    Ok((new_phase, delay, distance))
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
) -> Result<PeqResult, String> {
    info!(
        "auto_peq_lma: {} points, hp={}, lp={}, range={:?}",
        freq.len(), hp_freq, lp_freq, config.freq_range
    );
    peq::auto_peq_lma(
        &measurement_mag,
        target_mag.as_deref(),
        &freq,
        &config,
        hp_freq,
        lp_freq,
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
    meas_mag: Vec<f64>,
    target_mag: Vec<f64>,
    peq_correction: Vec<f64>,
    high_pass: Option<target::FilterConfig>,
    low_pass: Option<target::FilterConfig>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let n = freq.len();
    if n == 0 || meas_mag.len() != n || target_mag.len() != n {
        return Err("cross_section: length mismatch".into());
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

    // Step 2: corrected = meas + PEQ + filter
    // Where corrected < target → apply makeup gain (zero phase here, min-phase in FIR export)
    let mut makeup_mag = vec![0.0_f64; n];
    let makeup_phase = vec![0.0_f64; n];

    let peq_empty = peq_correction.is_empty();
    for i in 0..n {
        let peq_db = if peq_empty { 0.0 } else { peq_correction[i] };
        let corrected = meas_mag[i] + peq_db + filt_mag[i];
        let deficit = target_mag[i] - corrected;
        if deficit > 0.0 {
            makeup_mag[i] = deficit;
        }
    }

    // Total correction = user filter + makeup
    let total_mag: Vec<f64> = (0..n).map(|i| filt_mag[i] + makeup_mag[i]).collect();
    let total_phase: Vec<f64> = (0..n).map(|i| filt_phase[i] + makeup_phase[i]).collect();

    // Normalization estimate: peak of total correction = how much FIR will scale down
    let norm_db = total_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let norm_db = if norm_db.is_finite() && norm_db > 0.0 { norm_db } else { 0.0 };

    info!("compute_cross_section: {} points, hp={}, lp={}, makeup_pts={}, norm_db={:.1}",
        n, high_pass.is_some(), low_pass.is_some(),
        makeup_mag.iter().filter(|&&v| v > 0.0).count(), norm_db);

    Ok((total_mag, total_phase, norm_db))
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

    info!("PhaseForge v0.1.0-b56 starting...");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            import_measurement,
            get_smoothed,
            interpolate_log,
            evaluate_target,
            evaluate_target_standalone,
            compute_delay_info,
            remove_measurement_delay,
            compute_impulse,
            merge_measurements,
            preview_baffle_step,
            auto_peq,
            auto_peq_above_lp,
            auto_peq_lma,
            compute_peq_response,
            compute_peq_complex,
            compute_cross_section,
            generate_fir,
            generate_model_fir,
            recommend_fir_taps,
            export_fir_wav,
            project::save_project,
            project::load_project,
            project::create_project_folder,
            project::copy_file_to_project,
            project::check_path_exists,
            recent::load_recent_projects,
            recent::add_recent_project,
            recent::clear_recent_projects,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
