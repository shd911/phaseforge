pub mod analysis;
pub mod dsp;
pub mod error;
pub mod export;
pub mod fir;
pub mod io;
pub mod peq;
pub mod phase;
pub mod project;
pub mod recent;
pub mod snapshots;
pub mod target;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use dsp::SmoothingConfig;
use dsp::impulse::ImpulseResult;
use dsp::baffle::{BaffleConfig, BaffleStepPreview};
use dsp::merge::{MergeConfig, MergeResult};
use io::Measurement;
use fir::{FirConfig, FirModelResult};
use peq::{ExclusionZone, PeqBand, PeqConfig, PeqResult};
use target::{TargetCurve, TargetResponse};
use tauri::{Emitter, Manager, WindowEvent};
use tracing::info;

pub struct AppCloseState {
    pub allow_close: AtomicBool,
}

#[tauri::command]
fn close_window_now(
    state: tauri::State<AppCloseState>,
    window: tauri::Window,
) -> Result<(), String> {
    state.allow_close.store(true, Ordering::SeqCst);
    match window.close() {
        Ok(()) => Ok(()),
        Err(e) => {
            // Reset so a subsequent CloseRequested still triggers the dialog.
            state.allow_close.store(false, Ordering::SeqCst);
            Err(e.to_string())
        }
    }
}

/// Validate an export destination. b141.6 (audit): delegates to the shared
/// write guard (traversal / null bytes / hidden dirs / launch persistence);
/// the old `contains("..")` heuristic also rejected legitimate "..." names.
fn validate_export_path(path: &str) -> Result<PathBuf, String> {
    project::validate_write_target(path)?;
    Ok(PathBuf::from(path))
}

#[tauri::command]
async fn import_measurement(path: String) -> Result<Measurement, String> {
    info!("import_measurement: {}", path);
    let p = PathBuf::from(&path);
    io::import_measurement(&p).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_smoothed(
    freq: Vec<f64>,
    magnitude: Vec<f64>,
    config: SmoothingConfig,
) -> Result<Vec<f64>, String> {
    info!("get_smoothed: {} points", freq.len());
    Ok(dsp::variable_smoothing(&freq, &magnitude, &config))
}


#[tauri::command]
async fn evaluate_target(target: TargetCurve, freq: Vec<f64>) -> Result<TargetResponse, String> {
    info!("evaluate_target: {} points", freq.len());
    Ok(target::evaluate(&target, &freq))
}

#[tauri::command]
async fn evaluate_target_standalone(
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
async fn compute_delay_info(
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
async fn remove_measurement_delay(
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
async fn apply_manual_delay(
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
async fn compute_impulse(
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
async fn compute_minimum_phase(
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

#[tauri::command]
async fn merge_measurements(
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
async fn preview_baffle_step(config: BaffleConfig) -> Result<BaffleStepPreview, String> {
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

// b141.2: removed dead Tauri commands `auto_peq`, `auto_peq_above_lp` (frontend
// uses only `auto_peq_lma`). The underlying peq::auto_peq / auto_peq_above_lp
// remain for the peq module's own tests.

#[tauri::command]
async fn auto_peq_lma(
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
async fn compute_peq_response(
    freq: Vec<f64>,
    bands: Vec<PeqBand>,
    sample_rate: Option<f64>,
) -> Result<Vec<f64>, String> {
    let sr = sample_rate.unwrap_or(48000.0);
    info!("compute_peq_response: {} points, {} bands, sr={}", freq.len(), bands.len(), sr);
    Ok(peq::apply_peq(&freq, &bands, sr))
}

#[tauri::command]
async fn compute_peq_complex(
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
async fn compute_cross_section(
    freq: Vec<f64>,
    high_pass: Option<target::FilterConfig>,
    low_pass: Option<target::FilterConfig>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    let n = freq.len();
    if n == 0 {
        return Err("cross_section: empty freq".into());
    }

    // b140.15.9: complex-accumulator path. Scalar phase summation produced
    // 1-bin spikes (~358° jump) where HP or LP wrapped at ±180° on a single
    // bin — visible as ~120° downward spikes in SUM corrected phase. The
    // complex form is wrap-invariant (cos/sin are periodic) so the final
    // atan2 sees the true product phase modulo 360° without ever crossing
    // a wrong wrap boundary.
    let mut filt_mag = vec![0.0_f64; n];
    let mut re_acc = vec![1.0_f64; n];
    let mut im_acc = vec![0.0_f64; n];
    if let Some(hp) = &high_pass {
        target::apply_filter_complex(&mut filt_mag, &mut re_acc, &mut im_acc, &freq, hp, false);
    }
    if let Some(lp) = &low_pass {
        target::apply_filter_complex(&mut filt_mag, &mut re_acc, &mut im_acc, &freq, lp, true);
    }
    let mut filt_phase = vec![0.0_f64; n];
    target::complex_acc_to_phase_deg(&re_acc, &im_acc, &mut filt_phase);

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

// b141.2: removed dead Tauri commands `generate_fir`, `generate_hybrid_fir`,
// `recommend_fir_taps` — the frontend never invoked them (production FIR routes
// through generate_model_fir / generate_model_fir_iir). The underlying
// fir::generate_fir / generate_hybrid_fir live on under cfg(test) as the
// golden-snapshot baseline; fir::recommend_taps remains for tests.

#[tauri::command]
async fn generate_model_fir(
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

/// b140.7: IIR-cascade min-phase FIR pipeline. Produces an analytical
/// time-domain impulse for non-Gaussian crossovers (LR / Butterworth /
/// Custom HP+LP and PEQ biquads), bypassing the cepstral path that
/// caused REW-visible phase mismatch on min-phase exports. TS routes
/// here only when the configuration is IIR-realisable; otherwise it
/// keeps calling the original `generate_model_fir` above.
#[tauri::command]
async fn generate_model_fir_iir(
    freq: Vec<f64>,
    hp: Option<target::FilterConfig>,
    lp: Option<target::FilterConfig>,
    peq: Vec<peq::PeqBand>,
    config: FirConfig,
) -> Result<FirModelResult, String> {
    info!(
        "generate_model_fir_iir: {} log points, taps={}, sr={}, hp={:?} lp={:?} peq_bands={}",
        freq.len(), config.taps, config.sample_rate,
        hp.as_ref().map(|h| (&h.filter_type, h.order, h.freq_hz)),
        lp.as_ref().map(|l| (&l.filter_type, l.order, l.freq_hz)),
        peq.iter().filter(|p| p.enabled).count(),
    );
    fir::iir_path::generate_min_phase_fir_iir(&fir::iir_path::IirPathInput {
        freq: &freq,
        hp: hp.as_ref(),
        lp: lp.as_ref(),
        peq: &peq,
        config: &config,
    }).map_err(|e| e.to_string())
}

/// b140.15.5: Tauri-exposed FIR routing predicate — single source of truth.
///
/// JS-side `pickFirRoute` (src/lib/fir-routing.ts) was a textual mirror of
/// `fir::route_for`. If both predicates drifted identically, the existing
/// `pipeline_contract` test could not catch it. With this command JS calls
/// into the Rust predicate directly; the JS-side mirror is now deleted.
///
/// Returns "Iir" or "Cepstral" as a plain string so the JS side doesn't
/// need to mirror the enum.
#[tauri::command]
fn pick_fir_route(
    hp: Option<target::FilterConfig>,
    lp: Option<target::FilterConfig>,
    linear_main: bool,
    subsonic_cutoff_hz: Option<f64>,
) -> String {
    use fir::FirConfig;
    // route_for() reads only linear_phase_main + subsonic_cutoff_hz from
    // FirConfig — build a minimal struct with the rest zeroed. Avoids
    // shipping the full FirConfig across the boundary just to check two
    // fields.
    let cfg = FirConfig {
        taps: 0, sample_rate: 0.0,
        max_boost_db: 0.0, noise_floor_db: 0.0,
        window: fir::WindowType::Hann, phase_mode: fir::PhaseMode::Composite,
        iterations: 0, freq_weighting: false,
        narrowband_limit: false, nb_smoothing_oct: 0.0, nb_max_excess_db: 0.0,
        gaussian_min_phase_filters: vec![],
        linear_phase_main: linear_main,
        subsonic_cutoff_hz,
    };
    match fir::route_for(hp.as_ref(), lp.as_ref(), &cfg) {
        fir::Route::Iir => "Iir".into(),
        fir::Route::Cepstral => "Cepstral".into(),
    }
}

#[tauri::command]
async fn export_fir_wav(impulse: Vec<f64>, sample_rate: f64, path: String) -> Result<(), String> {
    info!("export_fir_wav: {} samples, sr={}, path={}", impulse.len(), sample_rate, path);

    // Validate path to prevent path traversal attacks
    let p = validate_export_path(&path)?;

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

    info!("PhaseForge b141.6 starting...");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppCloseState { allow_close: AtomicBool::new(false) })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let state: tauri::State<AppCloseState> = window.state();
                if !state.allow_close.load(Ordering::SeqCst) {
                    api.prevent_close();
                    let _ = window.emit("request-close-confirm", ());
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            import_measurement,
            get_smoothed,
            evaluate_target,
            evaluate_target_standalone,
            compute_delay_info,
            remove_measurement_delay,
            apply_manual_delay,
            compute_impulse,
            compute_minimum_phase,
            merge_measurements,
            preview_baffle_step,
            auto_peq_lma,
            compute_peq_response,
            compute_peq_complex,
            compute_cross_section,
            generate_model_fir,
            generate_model_fir_iir,
            pick_fir_route,
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
            close_window_now,
            snapshots::create_snapshot,
            snapshots::list_snapshots,
            snapshots::load_snapshot,
            snapshots::delete_snapshot,
            snapshots::rebuild_snapshot_index,
            analysis::analyze_measurement,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
