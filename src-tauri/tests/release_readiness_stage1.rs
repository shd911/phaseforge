// b140.17 — Release-readiness Stage 1.
//
// Loads the real-world 4-way project (4way-10-6.5-74-25) from
// test-fixtures/user_4way/, runs target + PEQ + auto-routed FIR pipeline
// per band, then a coherent SUM. Asserts: no NaN/Inf anywhere; PEQ phase
// stays inside (-180, 180]; no 1-bin phase spikes; FIR impulse finite
// with non-zero peak. Dumps full diagnostic to
// target/release_report_stage1.json for human review.
//
// Skips cleanly when the fixture is missing (24MB, gitignored) so
// clean checkouts / CI still pass.

use phaseforge_lib::fir::pipeline::pick_pipeline;
use phaseforge_lib::fir::{route_for, FirConfig, PhaseMode, Route, WindowType};
use phaseforge_lib::io::import_measurement;
use phaseforge_lib::peq::apply_peq_complex;
use phaseforge_lib::project::ProjectFile;
use phaseforge_lib::target::{self, FilterConfig};

use std::path::PathBuf;

const FIXTURE_REL: &str = "../test-fixtures/user_4way";
const PROJECT_FILE: &str = "4way-10-6.5-74-25.pfproj";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_REL)
}

fn load_fixture() -> Option<ProjectFile> {
    let path = fixture_root().join(PROJECT_FILE);
    if !path.exists() {
        return None;
    }
    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}

fn log_freq_grid(n: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let l0 = fmin.ln();
    let l1 = fmax.ln();
    (0..n)
        .map(|i| (l0 + (l1 - l0) * (i as f64) / ((n - 1) as f64)).exp())
        .collect()
}

/// Unwrap then look for 1-bin deviations from the local linear trend.
/// Returns (bin, deviation_deg) for points where the residual exceeds
/// `threshold_deg`. This is the same shape of artifact we hunted in
/// b140.15.x (apply_filter scalar phase sum, apply_peq_complex bug).
fn count_phase_spikes(phase: &[f64], threshold_deg: f64) -> Vec<(usize, f64)> {
    let unwrap_step = |a: f64, b: f64| -> f64 {
        let mut d = b - a;
        while d > 180.0 {
            d -= 360.0;
        }
        while d < -180.0 {
            d += 360.0;
        }
        a + d
    };
    let n = phase.len();
    if n < 3 {
        return vec![];
    }
    let mut u = vec![0.0_f64; n];
    u[0] = phase[0];
    for i in 1..n {
        u[i] = unwrap_step(u[i - 1], phase[i]);
    }
    let mut out = Vec::new();
    for i in 1..n - 1 {
        let interp = 0.5 * (u[i - 1] + u[i + 1]);
        let dev = (u[i] - interp).abs();
        if dev > threshold_deg {
            out.push((i, dev));
        }
    }
    out
}

fn parse_window(s: &str) -> WindowType {
    match s {
        "Hann" => WindowType::Hann,
        "Hamming" => WindowType::Hamming,
        "Blackman" => WindowType::Blackman,
        "BlackmanHarris" => WindowType::BlackmanHarris,
        "Kaiser" => WindowType::Kaiser,
        "FlatTop" => WindowType::FlatTop,
        _ => WindowType::Blackman,
    }
}

fn route_label(
    hp: Option<&FilterConfig>,
    lp: Option<&FilterConfig>,
    cfg: &FirConfig,
) -> &'static str {
    match route_for(hp, lp, cfg) {
        Route::Iir => "Iir",
        Route::Cepstral => "Cepstral",
    }
}

#[test]
fn release_readiness_stage1_4way_real_project() {
    let Some(project) = load_fixture() else {
        eprintln!(
            "[stage1] fixture missing — skipping ({})",
            fixture_root().display()
        );
        return;
    };

    assert_eq!(project.bands.len(), 4, "expected 4-way project");

    let freq = log_freq_grid(512, 22.0, 22_000.0);
    let sample_rate = project.export_sample_rate as f64;
    let taps = project.export_taps as usize;
    let inbox = fixture_root();

    let mut sum_re = vec![0.0_f64; freq.len()];
    let mut sum_im = vec![0.0_f64; freq.len()];

    let mut report = serde_json::Map::new();
    report.insert(
        "project".into(),
        serde_json::json!({
            "version": format!("b140.17 stage1"),
            "bands": project.bands.len(),
            "sample_rate": sample_rate,
            "taps": taps,
            "window": project.export_window.clone(),
            "project_name": project.project_name.clone(),
        }),
    );
    let mut band_reports = Vec::new();

    for (bi, band) in project.bands.iter().enumerate() {
        // 1) Load measurement file.
        let meas_path = band
            .measurement_file
            .as_ref()
            .map(|f| inbox.join(f))
            .unwrap_or_else(|| panic!("B{bi} has no measurement_file"));
        let meas = import_measurement(&meas_path)
            .unwrap_or_else(|e| panic!("B{bi} measurement load failed: {e}"));
        assert!(
            !meas.freq.is_empty(),
            "B{bi} measurement has zero points"
        );

        // 2) Target evaluation on log grid.
        let tr = target::evaluate(&band.target, &freq);
        assert_eq!(tr.magnitude.len(), freq.len(), "B{bi} target mag len");
        assert_eq!(tr.phase.len(), freq.len(), "B{bi} target phase len");
        for (i, &m) in tr.magnitude.iter().enumerate() {
            assert!(m.is_finite(), "B{bi} target mag NaN/Inf at bin {i}");
        }
        for (i, &p) in tr.phase.iter().enumerate() {
            assert!(p.is_finite(), "B{bi} target phase NaN/Inf at bin {i}");
        }
        let target_phase_spikes = count_phase_spikes(&tr.phase, 90.0);
        assert!(
            target_phase_spikes.len() < 3,
            "B{bi} target phase has {} spikes (>90°): {:?}",
            target_phase_spikes.len(),
            &target_phase_spikes[..target_phase_spikes.len().min(5)]
        );

        // 3) PEQ application via complex accumulator (b140.16 fix).
        let (peq_mag, peq_phase) = if band.peq_bands.is_empty() {
            (vec![0.0; freq.len()], vec![0.0; freq.len()])
        } else {
            apply_peq_complex(&freq, &band.peq_bands, sample_rate)
        };
        for (i, &m) in peq_mag.iter().enumerate() {
            assert!(m.is_finite(), "B{bi} PEQ mag NaN/Inf at bin {i}");
            assert!(m.abs() < 100.0, "B{bi} PEQ mag insane {m} at bin {i}");
        }
        for (i, &p) in peq_phase.iter().enumerate() {
            assert!(p.is_finite(), "B{bi} PEQ phase NaN/Inf at bin {i}");
            assert!(
                p.abs() <= 180.5,
                "B{bi} PEQ phase wrap violation {p} at bin {i}"
            );
        }
        let peq_spikes = count_phase_spikes(&peq_phase, 60.0);
        assert!(
            peq_spikes.len() < 3,
            "B{bi} PEQ phase has {} 1-bin spikes (>60°): {:?}",
            peq_spikes.len(),
            &peq_spikes[..peq_spikes.len().min(5)]
        );

        // 4) FIR pipeline via auto-routing.
        let hp = band.target.high_pass.as_ref();
        let lp = band.target.low_pass.as_ref();
        let fir_config = FirConfig {
            taps,
            sample_rate,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: parse_window(&project.export_window),
            phase_mode: PhaseMode::Composite,
            iterations: 3,
            freq_weighting: true,
            narrowband_limit: true,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
            linear_phase_main: false,
            subsonic_cutoff_hz: None,
        };
        let pipeline = pick_pipeline(hp, lp, &fir_config);
        let fir_res = pipeline
            .evaluate(hp, lp, &band.peq_bands, &fir_config, &freq)
            .unwrap_or_else(|e| panic!("B{bi} FIR pipeline failed: {e}"));

        assert_eq!(fir_res.impulse.len(), taps, "B{bi} FIR taps mismatch");
        for (i, &s) in fir_res.impulse.iter().enumerate() {
            assert!(s.is_finite(), "B{bi} FIR sample NaN/Inf at {i}");
        }
        for (i, &m) in fir_res.realized_mag.iter().enumerate() {
            assert!(m.is_finite(), "B{bi} FIR realized mag NaN/Inf at bin {i}");
        }
        for (i, &p) in fir_res.realized_phase.iter().enumerate() {
            assert!(p.is_finite(), "B{bi} FIR realized phase NaN/Inf at bin {i}");
        }
        let fir_phase_spikes = count_phase_spikes(&fir_res.realized_phase, 90.0);
        assert!(
            fir_phase_spikes.len() < 5,
            "B{bi} FIR realized phase has {} 1-bin spikes (>90°)",
            fir_phase_spikes.len()
        );
        let peak = fir_res
            .impulse
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            peak > 0.0 && peak.is_finite() && peak < 100.0,
            "B{bi} FIR peak invalid: {peak}"
        );

        // 5) Accumulate band contribution into coherent SUM.
        let total_phase: Vec<f64> = tr
            .phase
            .iter()
            .zip(peq_phase.iter())
            .map(|(&t, &p)| t + p)
            .collect();
        let total_mag_db: Vec<f64> = tr
            .magnitude
            .iter()
            .zip(peq_mag.iter())
            .map(|(&t, &p)| t + p)
            .collect();
        for i in 0..freq.len() {
            let mag_lin = 10.0_f64.powf(total_mag_db[i] / 20.0);
            let prad = total_phase[i].to_radians();
            sum_re[i] += mag_lin * prad.cos();
            sum_im[i] += mag_lin * prad.sin();
        }

        band_reports.push(serde_json::json!({
            "index": bi,
            "id": band.id,
            "name": band.name,
            "measurement_file": band.measurement_file,
            "meas_points": meas.freq.len(),
            "peq_count": band.peq_bands.len(),
            "hp": hp.map(|c| format!("{:?}{} @ {:.0}Hz lin={}", c.filter_type, c.order, c.freq_hz, c.linear_phase)),
            "lp": lp.map(|c| format!("{:?}{} @ {:.0}Hz lin={}", c.filter_type, c.order, c.freq_hz, c.linear_phase)),
            "route": route_label(hp, lp, &fir_config),
            "fir_peak": peak,
            "fir_causality": fir_res.causality,
            "fir_norm_db": fir_res.norm_db,
            "target_phase_spikes": target_phase_spikes.len(),
            "peq_phase_spikes": peq_spikes.len(),
            "fir_phase_spikes": fir_phase_spikes.len(),
        }));
    }

    // SUM checks.
    let mut sum_mag_db = vec![0.0; freq.len()];
    let mut sum_phase = vec![0.0; freq.len()];
    for i in 0..freq.len() {
        let m = (sum_re[i] * sum_re[i] + sum_im[i] * sum_im[i])
            .sqrt()
            .max(1e-15);
        sum_mag_db[i] = 20.0 * m.log10();
        sum_phase[i] = sum_im[i].atan2(sum_re[i]).to_degrees();
        assert!(sum_mag_db[i].is_finite(), "SUM mag NaN/Inf at bin {i}");
        assert!(sum_phase[i].is_finite(), "SUM phase NaN/Inf at bin {i}");
    }
    let sum_spikes = count_phase_spikes(&sum_phase, 60.0);
    assert!(
        sum_spikes.len() < 3,
        "SUM phase has {} 1-bin spikes (>60°): {:?}",
        sum_spikes.len(),
        &sum_spikes[..sum_spikes.len().min(5)]
    );
    let sum_mag_min = sum_mag_db.iter().cloned().fold(f64::INFINITY, f64::min);
    let sum_mag_max = sum_mag_db
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        sum_mag_max - sum_mag_min < 80.0,
        "SUM dynamic range {} dB looks broken",
        sum_mag_max - sum_mag_min
    );

    report.insert("bands".into(), serde_json::Value::Array(band_reports));
    report.insert(
        "sum".into(),
        serde_json::json!({
            "phase_spikes": sum_spikes.len(),
            "mag_min_db": sum_mag_min,
            "mag_max_db": sum_mag_max,
            "mag_range_db": sum_mag_max - sum_mag_min,
        }),
    );

    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    let _ = std::fs::create_dir_all(&target_dir);
    let out = target_dir.join("release_report_stage1.json");
    let payload = serde_json::to_string_pretty(&serde_json::Value::Object(report)).unwrap();
    std::fs::write(&out, payload).expect("write report");
    eprintln!("[stage1] PASS — wrote {}", out.display());
}
