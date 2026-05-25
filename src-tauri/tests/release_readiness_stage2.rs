// b140.17.1 — Release-readiness Stage 2.
//
// FIR pipeline matrix on real 4-way bands. For each band we cross
// {PhaseMode} × {Window} × {linear_phase_main} × {subsonic on/off}
// and assert: pipeline returns Ok, impulse length matches taps, no
// NaN/Inf in impulse / realized_mag / realized_phase, peak > 0 and
// finite, causality in [0, 1].
//
// Routing fact-check: bands with at least one linear-phase filter
// MUST go through Cepstral (LR1 lin in B1, LR1/LR2 lin in B2/3);
// the pure-min-phase Iir route is exercised by B0 only.
//
// Skips when fixture missing.

use phaseforge_lib::fir::pipeline::pick_pipeline;
use phaseforge_lib::fir::{FirConfig, PhaseMode, WindowType};
use phaseforge_lib::project::ProjectFile;
use phaseforge_lib::target::FilterConfig;

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

fn all_phase_modes() -> Vec<(&'static str, PhaseMode)> {
    vec![
        ("MinimumPhase", PhaseMode::MinimumPhase),
        ("LinearPhase", PhaseMode::LinearPhase),
        ("MixedPhase", PhaseMode::MixedPhase),
        ("HybridPhase", PhaseMode::HybridPhase),
        ("Composite", PhaseMode::Composite),
    ]
}

fn all_windows() -> Vec<(&'static str, WindowType)> {
    vec![
        ("Hann", WindowType::Hann),
        ("Hamming", WindowType::Hamming),
        ("Blackman", WindowType::Blackman),
        ("BlackmanHarris", WindowType::BlackmanHarris),
        ("FlatTop", WindowType::FlatTop),
        ("Kaiser", WindowType::Kaiser),
        ("Nuttall4", WindowType::Nuttall4),
    ]
}

#[test]
fn release_readiness_stage2_pipeline_matrix() {
    let Some(project) = load_fixture() else {
        eprintln!(
            "[stage2] fixture missing — skipping ({})",
            fixture_root().display()
        );
        return;
    };

    let freq = log_freq_grid(256, 22.0, 22_000.0); // smaller for matrix speed
    let sample_rate = project.export_sample_rate as f64;
    let taps: usize = 16384; // reduce from 65536 — matrix has hundreds of runs

    let phase_modes = all_phase_modes();
    let windows = all_windows();

    let mut total = 0usize;
    let mut failures = Vec::<String>::new();
    let mut runs = Vec::<serde_json::Value>::new();

    for (bi, band) in project.bands.iter().enumerate() {
        let hp: Option<&FilterConfig> = band.target.high_pass.as_ref();
        let lp: Option<&FilterConfig> = band.target.low_pass.as_ref();

        // subsonic_cutoff candidates: None, and (if HP exists) fc/8
        let mut subsonic_opts: Vec<Option<f64>> = vec![None];
        if let Some(h) = hp {
            subsonic_opts.push(Some(h.freq_hz / 8.0));
        }

        for (pm_name, pm) in phase_modes.iter() {
            for (win_name, win) in windows.iter() {
                for &lin_main in &[false, true] {
                    for &subsonic in &subsonic_opts {
                        total += 1;
                        let cfg = FirConfig {
                            taps,
                            sample_rate,
                            max_boost_db: 18.0,
                            noise_floor_db: -60.0,
                            window: win.clone(),
                            phase_mode: pm.clone(),
                            iterations: 2,
                            freq_weighting: true,
                            narrowband_limit: true,
                            nb_smoothing_oct: 0.333,
                            nb_max_excess_db: 6.0,
                            gaussian_min_phase_filters: vec![],
                            linear_phase_main: lin_main,
                            subsonic_cutoff_hz: subsonic,
                        };
                        let pipeline = pick_pipeline(hp, lp, &cfg);
                        let label = format!(
                            "B{bi}/{pm_name}/{win_name}/lin={lin_main}/subsonic={:?}",
                            subsonic
                        );

                        match pipeline.evaluate(hp, lp, &band.peq_bands, &cfg, &freq) {
                            Ok(res) => {
                                let mut local_fail: Option<String> = None;
                                if res.impulse.len() != taps {
                                    local_fail = Some(format!(
                                        "{label}: taps {} != {taps}",
                                        res.impulse.len()
                                    ));
                                }
                                if local_fail.is_none()
                                    && res.impulse.iter().any(|s| !s.is_finite())
                                {
                                    local_fail = Some(format!("{label}: NaN/Inf in impulse"));
                                }
                                if local_fail.is_none()
                                    && res.realized_mag.iter().any(|m| !m.is_finite())
                                {
                                    local_fail =
                                        Some(format!("{label}: NaN/Inf in realized_mag"));
                                }
                                if local_fail.is_none()
                                    && res.realized_phase.iter().any(|p| !p.is_finite())
                                {
                                    local_fail = Some(format!(
                                        "{label}: NaN/Inf in realized_phase"
                                    ));
                                }
                                let peak = res
                                    .impulse
                                    .iter()
                                    .copied()
                                    .fold(0.0_f64, |a, b| a.max(b.abs()));
                                if local_fail.is_none() && !(peak > 0.0 && peak < 100.0) {
                                    local_fail = Some(format!(
                                        "{label}: peak {peak} out of range"
                                    ));
                                }
                                if local_fail.is_none()
                                    && !(res.causality >= 0.0 && res.causality <= 1.000_001)
                                {
                                    local_fail = Some(format!(
                                        "{label}: causality {} out of [0,1]",
                                        res.causality
                                    ));
                                }
                                if let Some(f) = local_fail {
                                    failures.push(f);
                                } else {
                                    runs.push(serde_json::json!({
                                        "label": label,
                                        "ok": true,
                                        "peak": peak,
                                        "causality": res.causality,
                                        "norm_db": res.norm_db,
                                    }));
                                }
                            }
                            Err(e) => {
                                failures.push(format!("{label}: pipeline err: {e}"));
                            }
                        }
                    }
                }
            }
        }
    }

    let report = serde_json::json!({
        "version": "b140.17.1 stage2",
        "total_runs": total,
        "failures": failures.len(),
        "failure_messages": &failures[..failures.len().min(20)],
        "sample_runs": &runs[..runs.len().min(10)],
    });
    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    let _ = std::fs::create_dir_all(&target_dir);
    let out = target_dir.join("release_report_stage2.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .expect("write report");
    eprintln!(
        "[stage2] {} runs, {} failures — wrote {}",
        total,
        failures.len(),
        out.display()
    );

    assert!(
        failures.is_empty(),
        "{} failures across {} runs; first: {}",
        failures.len(),
        total,
        failures.first().unwrap()
    );
}

/// Routing contract on real bands. `route_for` looks at filter TYPE
/// (LR/BW/Custom realisable, Gaussian/Bessel not), plus the FirConfig
/// flags `linear_phase_main` and `subsonic_cutoff_hz`. The per-filter
/// `linear_phase` flag is orthogonal to routing — verified here.
#[test]
fn release_readiness_stage2_routing_invariants() {
    use phaseforge_lib::fir::{route_for, Route};
    use phaseforge_lib::target::FilterType;

    let Some(project) = load_fixture() else {
        eprintln!("[stage2/routing] fixture missing — skipping");
        return;
    };
    let sample_rate = project.export_sample_rate as f64;

    let base = |lin_main: bool, subsonic: Option<f64>| FirConfig {
        taps: 8192,
        sample_rate,
        max_boost_db: 18.0,
        noise_floor_db: -60.0,
        window: WindowType::Blackman,
        phase_mode: PhaseMode::Composite,
        iterations: 1,
        freq_weighting: false,
        narrowband_limit: false,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
        gaussian_min_phase_filters: vec![],
        linear_phase_main: lin_main,
        subsonic_cutoff_hz: subsonic,
    };

    let realisable = |f: Option<&FilterConfig>| -> bool {
        f.map(|c| {
            matches!(
                c.filter_type,
                FilterType::LinkwitzRiley | FilterType::Butterworth | FilterType::Custom
            )
        })
        .unwrap_or(true)
    };

    for (bi, band) in project.bands.iter().enumerate() {
        let hp = band.target.high_pass.as_ref();
        let lp = band.target.low_pass.as_ref();
        let iir_ok = realisable(hp) && realisable(lp);

        // Default config — no linear main, no subsonic.
        let r = route_for(hp, lp, &base(false, None));
        if iir_ok {
            assert_eq!(r, Route::Iir, "B{bi}: realisable but routed {r:?}");
        } else {
            assert_eq!(r, Route::Cepstral, "B{bi}: not realisable but routed {r:?}");
        }

        // linear_phase_main forces Cepstral regardless.
        assert_eq!(
            route_for(hp, lp, &base(true, None)),
            Route::Cepstral,
            "B{bi}: linear_phase_main must force Cepstral"
        );
        // subsonic forces Cepstral regardless.
        assert_eq!(
            route_for(hp, lp, &base(false, Some(20.0))),
            Route::Cepstral,
            "B{bi}: subsonic_cutoff_hz must force Cepstral"
        );
    }
}
