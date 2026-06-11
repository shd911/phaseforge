// b140.17.3 — Release-readiness Stage 4.
//
// On the real 4-way project:
//   1. Per-band IR + Step from compute_impulse_response on
//      (target_mag + peq_mag, target_phase + peq_phase). No NaN; step
//      converges to a finite value.
//   2. Coherent-sum IR — same routine over the SUM (re, im) accumulator.
//   3. PEQ optimizers on raw measurements:
//        - auto_peq        (flat target, full passband)
//        - auto_peq_lma    (3-zone composite, hp..lp from band target)
//        - auto_peq_above_lp (cuts only, > lp)
//      Each must produce >= 0 bands with finite gains and a max-error
//      that is non-NaN.
//   4. Alignment delay: apply a phase ramp `phi += -2*pi*f*tau` to one
//      band's contribution and verify the SUM stays finite and the
//      change is measurable (not a no-op).

use phaseforge_lib::dsp::impulse::compute_impulse_response;
use phaseforge_lib::fir::pipeline::pick_pipeline;
use phaseforge_lib::fir::{FirConfig, PhaseMode, WindowType};
use phaseforge_lib::io::import_measurement;
use phaseforge_lib::peq::{
    apply_peq_complex, auto_peq, auto_peq_above_lp, auto_peq_lma, PeqConfig,
};
use phaseforge_lib::project::ProjectFile;
use phaseforge_lib::target;

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

#[test]
fn release_readiness_stage4_sum_ir_peq_alignment() {
    let Some(project) = load_fixture() else {
        eprintln!("[stage4] fixture missing — skipping");
        return;
    };
    let sample_rate = project.export_sample_rate as f64;
    let freq = log_freq_grid(512, 22.0, 22_000.0);
    let inbox = fixture_root();

    let mut sum_re = vec![0.0_f64; freq.len()];
    let mut sum_im = vec![0.0_f64; freq.len()];

    let mut report = serde_json::Map::new();
    report.insert(
        "version".into(),
        serde_json::Value::String("b140.17.3 stage4".into()),
    );
    let mut band_reports = Vec::<serde_json::Value>::new();

    for (bi, band) in project.bands.iter().enumerate() {
        // Load measurement (needed for PEQ optimizers).
        let meas_path = band
            .measurement_file
            .as_ref()
            .map(|f| inbox.join(f))
            .expect("measurement_file present");
        let meas = import_measurement(&meas_path)
            .unwrap_or_else(|e| panic!("B{bi} measurement load: {e}"));

        // Re-grid measurement magnitude onto our log grid via simple
        // linear-in-log interpolation (good enough for PEQ smoke).
        let meas_mag_grid = regrid_log(&meas.freq, &meas.magnitude, &freq);
        assert!(meas_mag_grid.iter().all(|v| v.is_finite()));

        // 1) Per-band IR/Step from target + PEQ.
        let tr = target::evaluate(&band.target, &freq);
        let (peq_mag, peq_phase) = if band.peq_bands.is_empty() {
            (vec![0.0; freq.len()], vec![0.0; freq.len()])
        } else {
            apply_peq_complex(&freq, &band.peq_bands, sample_rate)
        };
        let total_mag: Vec<f64> = tr
            .magnitude
            .iter()
            .zip(peq_mag.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let total_phase: Vec<f64> = tr
            .phase
            .iter()
            .zip(peq_phase.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let ir = compute_impulse_response(&freq, &total_mag, &total_phase, sample_rate);
        assert!(!ir.impulse.is_empty(), "B{bi} IR empty");
        assert!(
            ir.impulse.iter().all(|s| s.is_finite()),
            "B{bi} IR has NaN/Inf"
        );
        assert!(
            ir.step.iter().all(|s| s.is_finite()),
            "B{bi} step has NaN/Inf"
        );
        assert!(
            ir.raw_peak.is_finite() && ir.raw_peak > 0.0,
            "B{bi} IR raw_peak invalid: {}",
            ir.raw_peak
        );

        // 2) Accumulate into SUM contribution (used after the loop).
        for i in 0..freq.len() {
            let mag_lin = 10.0_f64.powf(total_mag[i] / 20.0);
            let prad = total_phase[i].to_radians();
            sum_re[i] += mag_lin * prad.cos();
            sum_im[i] += mag_lin * prad.sin();
        }

        // 3) PEQ optimizers on raw measurement.
        let hp_freq = band
            .target
            .high_pass
            .as_ref()
            .map(|c| c.freq_hz)
            .unwrap_or(20.0);
        let lp_freq = band
            .target
            .low_pass
            .as_ref()
            .map(|c| c.freq_hz)
            .unwrap_or(20_000.0);

        let cfg = PeqConfig {
            max_bands: 10,
            tolerance_db: 1.0,
            peak_bias: 1.5,
            max_boost_db: 6.0,
            max_cut_db: 18.0,
            freq_range: (hp_freq.max(20.0), lp_freq.min(20_000.0)),
            smoothing_fraction: Some(1.0 / 6.0),
            min_band_distance_oct: Some(0.333),
            hybrid: false,
            gain_regularization: 0.0,
            sample_rate: 48000.0,
        };

        // Target for auto_peq: flat at average meas in passband.
        let in_pass: Vec<f64> = freq
            .iter()
            .zip(meas_mag_grid.iter())
            .filter(|(&f, _)| f >= hp_freq && f <= lp_freq)
            .map(|(_, &m)| m)
            .collect();
        let avg = if in_pass.is_empty() {
            80.0
        } else {
            in_pass.iter().sum::<f64>() / in_pass.len() as f64
        };
        let flat_target = vec![avg; freq.len()];

        let r_auto = auto_peq(&meas_mag_grid, &flat_target, &freq, &cfg)
            .unwrap_or_else(|e| panic!("B{bi} auto_peq: {e}"));
        assert!(
            r_auto.max_error_db.is_finite(),
            "B{bi} auto_peq max_error NaN"
        );
        check_peq_result(bi, "auto_peq", &r_auto.bands);

        let r_lma = auto_peq_lma(
            &meas_mag_grid,
            Some(&flat_target),
            &freq,
            &cfg,
            hp_freq,
            lp_freq,
            &[],
        )
        .unwrap_or_else(|e| panic!("B{bi} auto_peq_lma: {e}"));
        assert!(r_lma.max_error_db.is_finite());
        check_peq_result(bi, "auto_peq_lma", &r_lma.bands);

        // above_lp: only meaningful when there is a LP corner.
        let r_above = if band.target.low_pass.is_some() {
            let r = auto_peq_above_lp(&meas_mag_grid, &freq, &cfg, lp_freq, hp_freq)
                .unwrap_or_else(|e| panic!("B{bi} auto_peq_above_lp: {e}"));
            assert!(r.max_error_db.is_finite());
            check_peq_result(bi, "auto_peq_above_lp", &r.bands);
            Some(r.bands.len())
        } else {
            None
        };

        // FIR routing smoke (already covered in stage 2, but include
        // the chosen pipeline label for completeness here).
        let cfg_fir = FirConfig {
            taps: 8192,
            sample_rate,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::Composite,
            iterations: 2,
            freq_weighting: true,
            narrowband_limit: true,
            nb_smoothing_oct: 0.333,
            nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
            linear_phase_main: false,
            subsonic_cutoff_hz: None,
        };
        let pipe = pick_pipeline(
            band.target.high_pass.as_ref(),
            band.target.low_pass.as_ref(),
            &cfg_fir,
        );
        let _ = pipe.evaluate(
            band.target.high_pass.as_ref(),
            band.target.low_pass.as_ref(),
            &band.peq_bands,
            &cfg_fir,
            &freq,
        );

        band_reports.push(serde_json::json!({
            "index": bi,
            "name": band.name,
            "ir_len": ir.impulse.len(),
            "ir_raw_peak": ir.raw_peak,
            "ir_step_raw_peak": ir.step_raw_peak,
            "peq_auto_bands": r_auto.bands.len(),
            "peq_auto_max_err_db": r_auto.max_error_db,
            "peq_lma_bands": r_lma.bands.len(),
            "peq_lma_max_err_db": r_lma.max_error_db,
            "peq_above_lp_bands": r_above,
        }));
    }

    // 2/cont) Coherent SUM IR.
    let n = freq.len();
    let mut sum_mag = vec![0.0_f64; n];
    let mut sum_phase = vec![0.0_f64; n];
    for i in 0..n {
        let m = (sum_re[i] * sum_re[i] + sum_im[i] * sum_im[i])
            .sqrt()
            .max(1e-15);
        sum_mag[i] = 20.0 * m.log10();
        sum_phase[i] = sum_im[i].atan2(sum_re[i]).to_degrees();
    }
    let sum_ir = compute_impulse_response(&freq, &sum_mag, &sum_phase, sample_rate);
    assert!(
        sum_ir.impulse.iter().all(|s| s.is_finite()),
        "SUM IR has NaN/Inf"
    );
    assert!(
        sum_ir.step.iter().all(|s| s.is_finite()),
        "SUM step has NaN/Inf"
    );
    assert!(
        sum_ir.raw_peak.is_finite() && sum_ir.raw_peak > 0.0,
        "SUM IR raw_peak invalid: {}",
        sum_ir.raw_peak
    );

    // 4) Alignment delay: apply 1 ms phase ramp to band 1's
    //    contribution and compare to baseline sum mag at 1 kHz.
    let baseline_mag_1k = mag_at(&freq, &sum_mag, 1000.0);
    let (sum_re_d, sum_im_d) = recompute_sum_with_delay(&project, &freq, sample_rate, 1, 1e-3);
    let mut sum_mag_d = vec![0.0_f64; n];
    let mut sum_phase_d = vec![0.0_f64; n];
    for i in 0..n {
        let m = (sum_re_d[i] * sum_re_d[i] + sum_im_d[i] * sum_im_d[i])
            .sqrt()
            .max(1e-15);
        sum_mag_d[i] = 20.0 * m.log10();
        sum_phase_d[i] = sum_im_d[i].atan2(sum_re_d[i]).to_degrees();
        assert!(sum_mag_d[i].is_finite() && sum_phase_d[i].is_finite());
    }
    let delayed_mag_1k = mag_at(&freq, &sum_mag_d, 1000.0);
    // Delay changes interference pattern → mag at 1 kHz must shift
    // (cannot be bit-identical to baseline). Threshold 0.001 dB.
    let delta = (baseline_mag_1k - delayed_mag_1k).abs();
    assert!(
        delta > 0.001,
        "Alignment delay had no effect at 1 kHz (Δ={delta:.6} dB)"
    );

    report.insert("bands".into(), serde_json::Value::Array(band_reports));
    report.insert(
        "sum".into(),
        serde_json::json!({
            "ir_len": sum_ir.impulse.len(),
            "ir_raw_peak": sum_ir.raw_peak,
            "ir_step_raw_peak": sum_ir.step_raw_peak,
            "baseline_mag_1k_db": baseline_mag_1k,
            "delayed_mag_1k_db": delayed_mag_1k,
            "alignment_delta_db": delta,
        }),
    );

    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    let _ = std::fs::create_dir_all(&target_dir);
    let out = target_dir.join("release_report_stage4.json");
    std::fs::write(&out, serde_json::to_string_pretty(&serde_json::Value::Object(report)).unwrap())
        .expect("write report");
    eprintln!("[stage4] PASS — wrote {}", out.display());
}

fn check_peq_result(bi: usize, label: &str, bands: &[phaseforge_lib::peq::PeqBand]) {
    for (i, b) in bands.iter().enumerate() {
        assert!(b.freq_hz.is_finite() && b.freq_hz > 0.0, "B{bi}/{label}#{i}: bad freq");
        assert!(b.gain_db.is_finite(), "B{bi}/{label}#{i}: bad gain");
        assert!(b.q.is_finite() && b.q > 0.0, "B{bi}/{label}#{i}: bad Q");
    }
}

fn mag_at(freq: &[f64], mag: &[f64], f0: f64) -> f64 {
    // Closest bin.
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, &f) in freq.iter().enumerate() {
        let d = (f - f0).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    mag[best]
}

fn regrid_log(src_freq: &[f64], src_mag: &[f64], dst_freq: &[f64]) -> Vec<f64> {
    // Simple linear-in-log interpolation; clamp outside source range.
    let mut out = vec![0.0_f64; dst_freq.len()];
    for (i, &f) in dst_freq.iter().enumerate() {
        if f <= src_freq[0] {
            out[i] = src_mag[0];
            continue;
        }
        if f >= *src_freq.last().unwrap() {
            out[i] = *src_mag.last().unwrap();
            continue;
        }
        // Binary search would be nicer but this is a test.
        let mut j = 0;
        while j + 1 < src_freq.len() && src_freq[j + 1] < f {
            j += 1;
        }
        let f0 = src_freq[j].ln();
        let f1 = src_freq[j + 1].ln();
        let t = (f.ln() - f0) / (f1 - f0);
        out[i] = src_mag[j] + t * (src_mag[j + 1] - src_mag[j]);
    }
    out
}

fn recompute_sum_with_delay(
    project: &ProjectFile,
    freq: &[f64],
    _sample_rate: f64,
    delayed_band_idx: usize,
    delay_s: f64,
) -> (Vec<f64>, Vec<f64>) {
    let sr = project.export_sample_rate as f64;
    let mut re = vec![0.0_f64; freq.len()];
    let mut im = vec![0.0_f64; freq.len()];
    for (bi, band) in project.bands.iter().enumerate() {
        let tr = target::evaluate(&band.target, freq);
        let (peq_mag, peq_phase) = if band.peq_bands.is_empty() {
            (vec![0.0; freq.len()], vec![0.0; freq.len()])
        } else {
            apply_peq_complex(freq, &band.peq_bands, sr)
        };
        let extra_phase_per_hz_deg = if bi == delayed_band_idx {
            -360.0 * delay_s
        } else {
            0.0
        };
        for i in 0..freq.len() {
            let mag_db = tr.magnitude[i] + peq_mag[i];
            let mut phase_deg = tr.phase[i] + peq_phase[i];
            phase_deg += extra_phase_per_hz_deg * freq[i];
            let mag_lin = 10.0_f64.powf(mag_db / 20.0);
            let prad = phase_deg.to_radians();
            re[i] += mag_lin * prad.cos();
            im[i] += mag_lin * prad.sin();
        }
    }
    (re, im)
}
