//! Phase 0 test (b140.10.1): golden FIR snapshots.
//!
//! Locks down bit-exact FIR output (impulse + realized magnitude + realized
//! phase) for 8 canonical band configurations covering BOTH pipelines:
//!
//!   - IIR-analytical (`generate_min_phase_fir_iir`) — 4 fixtures
//!   - FFT-cepstral   (`generate_model_fir`)         — 4 fixtures
//!
//! Each result is rounded to 6 decimal places (~1e-6 relative tolerance —
//! tighter than any DSP-meaningful drift, looser than f64 LSB noise from
//! parallel sums) and hashed with SHA-256. Hashes live in
//! `tests/fixtures/golden_fir.json`.
//!
//! Bootstrap workflow:
//!   1. First run: `golden_fir.json` doesn't exist → file is written from
//!      current output and the test FAILS with "baseline created — review
//!      and commit". This forces a human review of the captured values.
//!   2. Subsequent runs: existing JSON is compared. Any divergence prints
//!      which fixtures drifted and FAILS, leaving JSON untouched.
//!
//! To intentionally re-baseline (e.g. after a sanctioned DSP change):
//! delete `tests/fixtures/golden_fir.json` and re-run.
//!
//! This suite is the safety net for Phases 2-5 of the pipeline-unification
//! plan (FirPipeline trait, fir/mod.rs split, legacy deletion). Any of
//! those refactors must keep ALL 8 hashes stable.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

use phaseforge_lib::fir::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use phaseforge_lib::fir::{FirConfig, PhaseMode, WindowType, generate_model_fir};
use phaseforge_lib::peq::{PeqBand, PeqFilterType};
use phaseforge_lib::target::{
    self, FilterConfig as TargetFilterConfig, FilterType as TargetFilterType, TargetCurve,
};

// ---------------------------------------------------------------------------
// Hash + fixture I/O helpers
// ---------------------------------------------------------------------------

/// Round to 6 decimals, then hash the (impulse, realized_mag, realized_phase)
/// triple. Rounding absorbs f64 LSB noise from parallel sums; 6 decimals is
/// ~1e-6 — well below the magnitude of any DSP-meaningful change.
fn hash_fir_result(impulse: &[f64], realized_mag: &[f64], realized_phase: &[f64]) -> String {
    let mut h = Sha256::new();
    for arr in [impulse, realized_mag, realized_phase] {
        h.update(arr.len().to_le_bytes());
        for &v in arr {
            // Treat negative zero as positive to avoid signed-zero drift.
            let rounded = (v * 1e6).round() / 1e6;
            let cleaned = if rounded == 0.0 { 0.0 } else { rounded };
            h.update(cleaned.to_le_bytes());
        }
        // Domain separator between arrays so we don't accidentally
        // collide [a, b] + [c, d] with [a] + [b, c, d].
        h.update(b"||");
    }
    let bytes = h.finalize();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_fir.json")
}

fn load_baseline() -> Option<BTreeMap<String, String>> {
    let path = fixtures_path();
    let bytes = std::fs::read(&path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_baseline(map: &BTreeMap<String, String>) {
    let path = fixtures_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    let json = serde_json::to_string_pretty(map).expect("serialize baseline");
    std::fs::write(&path, json).expect("write baseline");
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Log-spaced grid 5..40000 Hz, 512 points — matches band-evaluator.ts
/// production grid for realized_mag/phase readout.
fn log_freq_grid() -> Vec<f64> {
    let (f_min, f_max, n) = (5.0_f64, 40_000.0_f64, 512);
    (0..n)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn fir_config(phase_mode: PhaseMode, linear_main: bool, subsonic: Option<f64>) -> FirConfig {
    FirConfig {
        taps: 16_384,             // keep test runtime under a few seconds
        sample_rate: 48_000.0,
        max_boost_db: 18.0,
        noise_floor_db: -60.0,
        window: WindowType::Hann,
        phase_mode,
        iterations: 3,
        freq_weighting: true,
        narrowband_limit: true,
        nb_smoothing_oct: 0.333,
        nb_max_excess_db: 6.0,
        gaussian_min_phase_filters: vec![],
        linear_phase_main: linear_main,
        subsonic_cutoff_hz: subsonic,
    }
}

fn lr4(freq: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::LinkwitzRiley,
        order: 4, freq_hz: freq, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn bw(order: u8, freq: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::Butterworth,
        order, freq_hz: freq, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn custom_q(freq: f64, q: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::Custom,
        order: 2, freq_hz: freq, shape: None,
        linear_phase: false, q: Some(q), subsonic_protect: None,
    }
}

fn gaussian(freq: f64, shape: f64, subsonic: bool) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::Gaussian,
        order: 4, freq_hz: freq, shape: Some(shape),
        linear_phase: false, q: None,
        subsonic_protect: Some(subsonic),
    }
}

fn bessel(freq: f64) -> TargetFilterConfig {
    TargetFilterConfig {
        filter_type: TargetFilterType::Bessel,
        order: 4, freq_hz: freq, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn peq_peaking(freq: f64, gain: f64, q: f64) -> PeqBand {
    PeqBand {
        freq_hz: freq, gain_db: gain, q,
        enabled: true, filter_type: PeqFilterType::Peaking,
    }
}

// ---------------------------------------------------------------------------
// Fixture runners — one entry per pipeline; both return (impulse, mag, phase)
// ---------------------------------------------------------------------------

struct FirOutput {
    impulse: Vec<f64>,
    realized_mag: Vec<f64>,
    realized_phase: Vec<f64>,
}

fn run_iir(
    hp: Option<&TargetFilterConfig>,
    lp: Option<&TargetFilterConfig>,
    peq: &[PeqBand],
    cfg: &FirConfig,
) -> FirOutput {
    let freq = log_freq_grid();
    let r = generate_min_phase_fir_iir(&IirPathInput { freq: &freq, hp, lp, peq, config: cfg })
        .expect("iir_path run");
    FirOutput { impulse: r.impulse, realized_mag: r.realized_mag, realized_phase: r.realized_phase }
}

fn run_cepstral(target: &TargetCurve, peq: &[PeqBand], cfg: &FirConfig) -> FirOutput {
    let freq = log_freq_grid();
    let resp = target::evaluate(target, &freq);
    // Cepstral path takes (target_mag, peq_mag, model_phase) separately.
    // For these golden fixtures we leave peq_mag empty (PEQ entered via the
    // dedicated IIR-path fixtures); model_phase comes from target evaluation
    // so the cepstral pipeline reconstructs the same phase the UI shows.
    let peq_mag: Vec<f64> = if peq.is_empty() {
        vec![]
    } else {
        let mut m = vec![0.0; freq.len()];
        for b in peq.iter().filter(|b| b.enabled) {
            // Trivial fallback — keep PEQ contribution out of cepstral
            // fixtures unless explicitly required. We pass empty above.
            let _ = b;
            m.fill(0.0);
        }
        m
    };
    let r = generate_model_fir(&freq, &resp.magnitude, &peq_mag, &resp.phase, cfg)
        .expect("model_fir run");
    FirOutput { impulse: r.impulse, realized_mag: r.realized_mag, realized_phase: r.realized_phase }
}

fn target_with(hp: Option<TargetFilterConfig>, lp: Option<TargetFilterConfig>) -> TargetCurve {
    TargetCurve {
        reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
        high_pass: hp, low_pass: lp, low_shelf: None, high_shelf: None,
    }
}

// ---------------------------------------------------------------------------
// Fixture catalog — order matters (file is committed; reordering = diff)
// ---------------------------------------------------------------------------

fn collect_hashes() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();

    // --- IIR-analytical pipeline (4 fixtures) -----------------------------
    {
        let cfg = fir_config(PhaseMode::Composite, false, None);
        let hp = lr4(80.0);
        let o = run_iir(Some(&hp), None, &[], &cfg);
        m.insert("iir_01_lr4_hp_80".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }
    {
        let cfg = fir_config(PhaseMode::Composite, false, None);
        let hp = lr4(100.0);
        let lp = lr4(2000.0);
        let peq = vec![peq_peaking(500.0, 3.0, 1.0)];
        let o = run_iir(Some(&hp), Some(&lp), &peq, &cfg);
        m.insert("iir_02_lr4_bandpass_with_peq".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }
    {
        let cfg = fir_config(PhaseMode::Composite, false, None);
        let hp = bw(2, 100.0);
        let o = run_iir(Some(&hp), None, &[], &cfg);
        m.insert("iir_03_bw2_hp_100".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }
    {
        let cfg = fir_config(PhaseMode::Composite, false, None);
        let hp = custom_q(150.0, 1.2);
        let lp = custom_q(3000.0, 0.707);
        let o = run_iir(Some(&hp), Some(&lp), &[], &cfg);
        m.insert("iir_04_custom_bandpass".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }

    // --- FFT-cepstral pipeline (4 fixtures) --------------------------------
    {
        let cfg = fir_config(PhaseMode::Composite, false, None);
        let t = target_with(Some(gaussian(632.0, 1.0, false)), None);
        let o = run_cepstral(&t, &[], &cfg);
        m.insert("cep_01_gaussian_hp_subsonic_off".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }
    {
        // subsonic ON → cepstral via UI routing (subsonic_cutoff_hz set)
        let cfg = fir_config(PhaseMode::Composite, false, Some(632.0 / 8.0));
        let t = target_with(Some(gaussian(632.0, 1.0, true)), None);
        let o = run_cepstral(&t, &[], &cfg);
        m.insert("cep_02_gaussian_hp_subsonic_on".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }
    {
        let cfg = fir_config(PhaseMode::Composite, false, None);
        let t = target_with(None, Some(bessel(500.0)));
        let o = run_cepstral(&t, &[], &cfg);
        m.insert("cep_03_bessel_lp_500".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }
    {
        // linear-phase main → cepstral via UI routing
        let cfg = fir_config(PhaseMode::Composite, true, None);
        let t = target_with(Some(lr4(80.0)), Some(lr4(2000.0)));
        let o = run_cepstral(&t, &[], &cfg);
        m.insert("cep_04_lr4_bandpass_linear_main".into(), hash_fir_result(&o.impulse, &o.realized_mag, &o.realized_phase));
    }

    m
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn golden_fir_snapshots_match_baseline() {
    let current = collect_hashes();
    match load_baseline() {
        None => {
            save_baseline(&current);
            panic!(
                "golden_fir baseline did not exist — wrote {} entries to {}. \
                 Review the file (each entry is a SHA-256 over a rounded FIR result), \
                 commit it, then re-run the test.",
                current.len(),
                fixtures_path().display(),
            );
        }
        Some(baseline) => {
            let mut diffs = Vec::new();
            for (k, v) in &current {
                match baseline.get(k) {
                    None => diffs.push(format!("  + NEW   {}: {}", k, v)),
                    Some(b) if b != v => diffs.push(format!(
                        "  ! DRIFT {}\n      expected: {}\n      got:      {}",
                        k, b, v
                    )),
                    _ => {}
                }
            }
            for k in baseline.keys() {
                if !current.contains_key(k) {
                    diffs.push(format!("  - GONE  {} (removed from fixture catalog)", k));
                }
            }
            assert!(
                diffs.is_empty(),
                "golden_fir snapshots drifted:\n{}\n\nIf the change is intentional, delete \
                 {} and re-run to re-baseline.",
                diffs.join("\n"),
                fixtures_path().display(),
            );
        }
    }
}
