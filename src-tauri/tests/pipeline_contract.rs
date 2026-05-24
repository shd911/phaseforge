//! Phase 0 test (b140.10.3): FirPipeline contract baseline.
//!
//! Locks down the unified-API contract that Phase 2 (`trait FirPipeline`)
//! must satisfy:
//!   - Single entry point taking (hp, lp, peq, fir_config, freq) → FirModelResult.
//!   - Routing decision derived from the same predicate that JS-side
//!     `pickFirRoute` uses (LR/BW/Custom + min-phase + no subsonic → IIR,
//!     everything else → cepstral).
//!   - For each route the produced FirModelResult hashes to a baseline
//!     committed in tests/fixtures/pipeline_contract.json.
//!
//! Today the test routes via local helpers below (`pick_route_rust` +
//! `evaluate_via_contract`) that directly call `generate_min_phase_fir_iir`
//! or `generate_model_fir`. After Phase 2 lands, only `evaluate_via_contract`
//! switches to `pick_pipeline(...).evaluate(...)`; baselines must remain
//! bit-identical (modulo the 6-decimal rounding in `hash_fir_result`).
//!
//! Bootstrap workflow mirrors golden_fir_snapshots.rs.
//!
//! Why this is separate from `golden_fir_snapshots.rs`:
//!   - golden_fir directly chooses the pipeline per fixture (value lock).
//!   - pipeline_contract uses the ROUTING PREDICATE to choose, so it
//!     additionally locks down that routing-decision parity across JS/Rust.
//!     Drift here = JS and Rust disagree about which path a given config
//!     should go to. Drift in golden_fir = DSP math changed.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;

use phaseforge_lib::fir::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use phaseforge_lib::fir::{FirConfig, FirModelResult, PhaseMode, WindowType, generate_model_fir};
use phaseforge_lib::peq::{PeqBand, PeqFilterType};
use phaseforge_lib::target::{
    self, FilterConfig as TargetFilterConfig, FilterType as TargetFilterType, TargetCurve,
};

// ---------------------------------------------------------------------------
// Routing predicate — mirrors src/lib/fir-routing.ts pickFirRoute exactly.
//
// Phase 2 will move this into the Rust source tree as the single source of
// truth and delete the JS copy (or vice versa). Until then, KEEPING THESE
// TWO IN SYNC is enforced by the per-fixture `expected_route` assertions
// below: if Rust and JS routing diverge, the JS routing-decision tests stay
// green but this test fails on the first fixture whose route changes.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Iir,
    Cepstral,
}

fn is_iir_realizable(f: Option<&TargetFilterConfig>) -> bool {
    match f {
        None => true,
        Some(c) => matches!(
            c.filter_type,
            TargetFilterType::LinkwitzRiley
                | TargetFilterType::Butterworth
                | TargetFilterType::Custom
        ),
    }
}

fn pick_route_rust(
    hp: Option<&TargetFilterConfig>,
    lp: Option<&TargetFilterConfig>,
    linear_main: bool,
    subsonic_cutoff_hz: Option<f64>,
) -> Route {
    if linear_main { return Route::Cepstral; }
    if subsonic_cutoff_hz.is_some() { return Route::Cepstral; }
    if !is_iir_realizable(hp) { return Route::Cepstral; }
    if !is_iir_realizable(lp) { return Route::Cepstral; }
    Route::Iir
}

// ---------------------------------------------------------------------------
// Contract evaluator — the SINGLE entry that Phase 2 will replace.
// ---------------------------------------------------------------------------

fn evaluate_via_contract(
    hp: Option<&TargetFilterConfig>,
    lp: Option<&TargetFilterConfig>,
    peq: &[PeqBand],
    cfg: &FirConfig,
    freq: &[f64],
) -> (Route, FirModelResult) {
    let route = pick_route_rust(hp, lp, cfg.linear_phase_main, cfg.subsonic_cutoff_hz);
    let result = match route {
        Route::Iir => {
            generate_min_phase_fir_iir(&IirPathInput {
                freq, hp, lp, peq, config: cfg,
            })
            .expect("iir_path run")
        }
        Route::Cepstral => {
            // Cepstral path consumes pre-evaluated arrays. The contract
            // wrapper evaluates the target internally so callers pass only
            // filter configs (the future trait signature).
            let target = TargetCurve {
                reference_level_db: 0.0,
                tilt_db_per_octave: 0.0,
                tilt_ref_freq: 1000.0,
                high_pass: hp.cloned(),
                low_pass: lp.cloned(),
                low_shelf: None, high_shelf: None,
            };
            let resp = target::evaluate(&target, freq);
            // PEQ is intentionally not folded into cepstral-path mag here.
            // The production FrontEnd evaluates PEQ separately and routes
            // its magnitude in via `peq_mag`. For contract bootstrap we
            // mirror "peq disabled at cepstral level" — IIR-eligible
            // configs are the ones that exercise PEQ inside this test.
            let peq_mag: Vec<f64> = vec![];
            generate_model_fir(freq, &resp.magnitude, &peq_mag, &resp.phase, cfg)
                .expect("model_fir run")
        }
    };
    (route, result)
}

// ---------------------------------------------------------------------------
// Hash + fixture I/O — identical pattern to golden_fir_snapshots.rs.
// ---------------------------------------------------------------------------

fn hash_fir_result(impulse: &[f64], realized_mag: &[f64], realized_phase: &[f64]) -> String {
    let mut h = Sha256::new();
    for arr in [impulse, realized_mag, realized_phase] {
        h.update(arr.len().to_le_bytes());
        for &v in arr {
            let rounded = (v * 1e6).round() / 1e6;
            let cleaned = if rounded == 0.0 { 0.0 } else { rounded };
            h.update(cleaned.to_le_bytes());
        }
        h.update(b"||");
    }
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pipeline_contract.json")
}

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
struct Entry {
    route: String,
    hash: String,
}

fn load_baseline() -> Option<BTreeMap<String, Entry>> {
    let bytes = std::fs::read(fixtures_path()).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_baseline(map: &BTreeMap<String, Entry>) {
    let path = fixtures_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    std::fs::write(&path, serde_json::to_string_pretty(map).unwrap()).unwrap();
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn log_freq_grid() -> Vec<f64> {
    let (f_min, f_max, n) = (5.0_f64, 40_000.0_f64, 512);
    (0..n)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn fir_config(linear_main: bool, subsonic: Option<f64>) -> FirConfig {
    FirConfig {
        taps: 16_384,
        sample_rate: 48_000.0,
        max_boost_db: 18.0,
        noise_floor_db: -60.0,
        window: WindowType::Hann,
        phase_mode: PhaseMode::Composite,
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
    PeqBand { freq_hz: freq, gain_db: gain, q, enabled: true, filter_type: PeqFilterType::Peaking }
}

// ---------------------------------------------------------------------------
// Fixture catalog — pairs (label, expected_route) with the input tuple.
// ---------------------------------------------------------------------------

fn collect_entries() -> BTreeMap<String, Entry> {
    let mut out = BTreeMap::new();
    let freq = log_freq_grid();

    let mut run = |label: &str,
                   expected_route: Route,
                   hp: Option<&TargetFilterConfig>,
                   lp: Option<&TargetFilterConfig>,
                   peq: &[PeqBand],
                   cfg: &FirConfig| {
        let (route, r) = evaluate_via_contract(hp, lp, peq, cfg, &freq);
        assert_eq!(
            route, expected_route,
            "{}: routing predicate disagrees with fixture expectation",
            label
        );
        out.insert(label.into(), Entry {
            route: format!("{:?}", route),
            hash: hash_fir_result(&r.impulse, &r.realized_mag, &r.realized_phase),
        });
    };

    // --- Route through the IIR analytical path -----------------------------
    let cfg_minphase = fir_config(false, None);
    let hp_lr4 = lr4(80.0);
    run("contract_01_lr4_hp_only", Route::Iir, Some(&hp_lr4), None, &[], &cfg_minphase);

    let hp_bp = lr4(100.0);
    let lp_bp = lr4(2000.0);
    let peq_v = vec![peq_peaking(500.0, 2.5, 1.2)];
    run("contract_02_lr4_bandpass_with_peq", Route::Iir,
        Some(&hp_bp), Some(&lp_bp), &peq_v, &cfg_minphase);

    let hp_bw = bw(2, 150.0);
    run("contract_03_bw2_hp_only", Route::Iir, Some(&hp_bw), None, &[], &cfg_minphase);

    // --- Route through the cepstral path -----------------------------------
    let gauss_hp = gaussian(632.0, 1.0, false);
    run("contract_04_gaussian_hp_route_cepstral", Route::Cepstral,
        Some(&gauss_hp), None, &[], &cfg_minphase);

    let bess_lp = bessel(500.0);
    run("contract_05_bessel_lp_route_cepstral", Route::Cepstral,
        None, Some(&bess_lp), &[], &cfg_minphase);

    let cfg_linear = fir_config(true, None);
    run("contract_06_linear_main_forces_cepstral", Route::Cepstral,
        Some(&hp_lr4), Some(&lp_bp), &[], &cfg_linear);

    let cfg_subsonic = fir_config(false, Some(80.0 / 8.0));
    run("contract_07_subsonic_forces_cepstral", Route::Cepstral,
        Some(&hp_lr4), None, &[], &cfg_subsonic);

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_contract_matches_baseline() {
    let current = collect_entries();
    match load_baseline() {
        None => {
            save_baseline(&current);
            panic!(
                "pipeline_contract baseline did not exist — wrote {} entries to {}. \
                 Review the file, commit it, then re-run.",
                current.len(),
                fixtures_path().display(),
            );
        }
        Some(baseline) => {
            let mut diffs = Vec::new();
            for (k, v) in &current {
                match baseline.get(k) {
                    None => diffs.push(format!("  + NEW   {}: route={} hash={}", k, v.route, v.hash)),
                    Some(b) if b != v => diffs.push(format!(
                        "  ! DRIFT {}\n      expected: route={} hash={}\n      got:      route={} hash={}",
                        k, b.route, b.hash, v.route, v.hash,
                    )),
                    _ => {}
                }
            }
            for k in baseline.keys() {
                if !current.contains_key(k) {
                    diffs.push(format!("  - GONE  {} (removed from catalog)", k));
                }
            }
            assert!(
                diffs.is_empty(),
                "pipeline_contract drifted:\n{}\n\nIf the change is intentional, delete \
                 {} and re-run to re-baseline.",
                diffs.join("\n"),
                fixtures_path().display(),
            );
        }
    }
}

#[test]
fn rust_routing_predicate_matches_js_decisions() {
    // Direct table-driven mirror of src/lib/__tests__/routing-decision.test.ts.
    // Any divergence here means the JS-side and Rust-side routing decisions
    // would disagree after Phase 2 unification.
    let lr = lr4(1000.0);
    let bw_ = bw(4, 1000.0);
    let cust = TargetFilterConfig {
        filter_type: TargetFilterType::Custom,
        order: 2, freq_hz: 1000.0, shape: None,
        linear_phase: false, q: Some(0.707), subsonic_protect: None,
    };
    let gauss = gaussian(1000.0, 1.0, false);
    let bess = bessel(1000.0);

    // (label, hp, lp, linear_main, subsonic, expected)
    let table: Vec<(&str, Option<&TargetFilterConfig>, Option<&TargetFilterConfig>, bool, Option<f64>, Route)> = vec![
        ("lr4 hp only, min-phase",                       Some(&lr),  None,        false, None,      Route::Iir),
        ("bw lp only, min-phase",                        None,       Some(&bw_),  false, None,      Route::Iir),
        ("custom hp + custom lp",                        Some(&cust),Some(&cust), false, None,      Route::Iir),
        ("lr hp + bw lp",                                Some(&lr),  Some(&bw_),  false, None,      Route::Iir),
        ("no hp no lp",                                  None,       None,        false, None,      Route::Iir),
        ("linearMain forces cepstral",                   Some(&lr),  Some(&lr),   true,  None,      Route::Cepstral),
        ("subsonic forces cepstral",                     Some(&lr),  None,        false, Some(50.0),Route::Cepstral),
        ("gaussian hp routes cepstral",                  Some(&gauss),None,       false, None,      Route::Cepstral),
        ("gaussian lp routes cepstral",                  None,       Some(&gauss),false, None,      Route::Cepstral),
        ("lr hp + gauss lp",                             Some(&lr),  Some(&gauss),false, None,      Route::Cepstral),
        ("bessel hp routes cepstral",                    Some(&bess),None,        false, None,      Route::Cepstral),
        ("bessel lp routes cepstral",                    None,       Some(&bess), false, None,      Route::Cepstral),
        ("all three disqualifiers",                      Some(&gauss),Some(&bess),true,  Some(50.0),Route::Cepstral),
    ];

    for (label, hp, lp, lin, sub, expected) in table {
        let got = pick_route_rust(hp, lp, lin, sub);
        assert_eq!(got, expected, "{}", label);
    }
}
