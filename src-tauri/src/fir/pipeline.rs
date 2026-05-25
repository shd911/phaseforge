//! b140.16.1 — Phase 2 full: unified `FirPipeline` trait + two
//! implementations + `pick_pipeline` factory.
//!
//! Previously (b140.12 "Phase 2 lite") only the routing predicate
//! `route_for` was unified in Rust; the two FIR-generation functions
//! kept distinct signatures and the dispatch happened in JS before
//! crossing the Tauri boundary.
//!
//! This module adds a single trait that both pipelines implement, with
//! a uniform `(hp, lp, peq, fir_config, freq) → FirModelResult`
//! signature. The cepstral implementation internally evaluates target
//! magnitude/phase and PEQ contribution so the trait surface is
//! transport-agnostic.
//!
//! Production routing (via the `pick_fir_route` Tauri command added
//! in b140.15.5) still happens JS-side for backwards compatibility.
//! This trait is consumed by the `pipeline_contract` integration test
//! to verify the unified path produces identical hashes to the direct
//! per-pipeline calls. When/if production gains a unified Tauri command
//! it can call `pick_pipeline(...).evaluate(...)` without any further
//! refactor.

use crate::error::AppError;
use crate::peq::{apply_peq_complex, PeqBand};
use crate::target::{self, FilterConfig, TargetCurve};

use super::cepstral::generate_model_fir;
use super::dispatch::{route_for, Route};
use super::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use super::types::{FirConfig, FirModelResult};

/// Unified pipeline interface. Each implementation accepts the same
/// `(hp, lp, peq, config, freq)` tuple and returns a `FirModelResult`.
pub trait FirPipeline {
    fn evaluate(
        &self,
        hp: Option<&FilterConfig>,
        lp: Option<&FilterConfig>,
        peq: &[PeqBand],
        config: &FirConfig,
        freq: &[f64],
    ) -> Result<FirModelResult, AppError>;
}

/// IIR-analytical cascade pipeline (uses `generate_min_phase_fir_iir`).
/// Restricted to min-phase main + no subsonic + every active crossover
/// is LR / Butterworth / Custom — selected by `route_for`.
pub struct IirAnalyticalPipeline;

impl FirPipeline for IirAnalyticalPipeline {
    fn evaluate(
        &self,
        hp: Option<&FilterConfig>,
        lp: Option<&FilterConfig>,
        peq: &[PeqBand],
        config: &FirConfig,
        freq: &[f64],
    ) -> Result<FirModelResult, AppError> {
        generate_min_phase_fir_iir(&IirPathInput {
            freq, hp, lp, peq, config,
        })
    }
}

/// FFT-cepstral pipeline (uses `generate_model_fir`). Evaluates the
/// target curve and PEQ contribution internally so the caller's
/// surface matches the IIR variant. PEQ phase is added via complex
/// multiplication (b140.16 fix in apply_peq_complex) so wrap noise
/// from multiple bands doesn't accumulate.
pub struct CepstralFftPipeline;

impl FirPipeline for CepstralFftPipeline {
    fn evaluate(
        &self,
        hp: Option<&FilterConfig>,
        lp: Option<&FilterConfig>,
        peq: &[PeqBand],
        config: &FirConfig,
        freq: &[f64],
    ) -> Result<FirModelResult, AppError> {
        let target_curve = TargetCurve {
            reference_level_db: 0.0,
            tilt_db_per_octave: 0.0,
            tilt_ref_freq: 1000.0,
            high_pass: hp.cloned(),
            low_pass: lp.cloned(),
            low_shelf: None,
            high_shelf: None,
        };
        let target_resp = target::evaluate(&target_curve, freq);

        // PEQ contribution: compute mag and phase separately (peq_mag is
        // fed as a dedicated array to generate_model_fir so it can apply
        // its own Hilbert; model_phase carries target + peq combined).
        let (peq_mag, peq_phase) = if peq.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            apply_peq_complex(freq, peq, config.sample_rate)
        };

        let combined_phase: Vec<f64> = if peq_phase.is_empty() {
            target_resp.phase
        } else {
            target_resp.phase.iter()
                .zip(peq_phase.iter())
                .map(|(&t, &p)| t + p)
                .collect()
        };

        generate_model_fir(
            freq,
            &target_resp.magnitude,
            &peq_mag,
            &combined_phase,
            config,
        )
    }
}

/// Pick the right pipeline for the given band configuration. Same
/// predicate as `route_for` — the dispatch happens here too for
/// callers that don't want to handle the enum themselves.
pub fn pick_pipeline(
    hp: Option<&FilterConfig>,
    lp: Option<&FilterConfig>,
    config: &FirConfig,
) -> Box<dyn FirPipeline> {
    match route_for(hp, lp, config) {
        Route::Iir => Box::new(IirAnalyticalPipeline),
        Route::Cepstral => Box::new(CepstralFftPipeline),
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fir::{PhaseMode, WindowType};
    use crate::target::FilterType;

    fn cfg(linear_main: bool, subsonic: Option<f64>) -> FirConfig {
        FirConfig {
            taps: 8192,
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

    fn lr4(freq: f64) -> FilterConfig {
        FilterConfig {
            filter_type: FilterType::LinkwitzRiley,
            order: 4, freq_hz: freq, shape: None,
            linear_phase: false, q: None, subsonic_protect: None,
        }
    }

    fn log_grid() -> Vec<f64> {
        let n = 512;
        (0..n).map(|i| 5.0_f64 * (40000.0_f64 / 5.0).powf(i as f64 / (n - 1) as f64)).collect()
    }

    #[test]
    fn pick_pipeline_routes_iir_for_lr_min_phase() {
        let cfg = cfg(false, None);
        let hp = lr4(80.0);
        let p = pick_pipeline(Some(&hp), None, &cfg);
        let freq = log_grid();
        let r = p.evaluate(Some(&hp), None, &[], &cfg, &freq).expect("evaluate");
        assert_eq!(r.impulse.len(), cfg.taps);
    }

    #[test]
    fn pick_pipeline_routes_cepstral_for_linear_main() {
        let cfg = cfg(true, None);
        let hp = lr4(80.0);
        let lp = lr4(2000.0);
        let p = pick_pipeline(Some(&hp), Some(&lp), &cfg);
        let freq = log_grid();
        let r = p.evaluate(Some(&hp), Some(&lp), &[], &cfg, &freq).expect("evaluate");
        assert_eq!(r.impulse.len(), cfg.taps);
    }

    #[test]
    fn iir_pipeline_matches_direct_call() {
        let cfg = cfg(false, None);
        let hp = lr4(100.0);
        let lp = lr4(2000.0);
        let freq = log_grid();
        let via_trait = IirAnalyticalPipeline.evaluate(Some(&hp), Some(&lp), &[], &cfg, &freq)
            .expect("trait");
        let direct = generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp: Some(&hp), lp: Some(&lp), peq: &[], config: &cfg,
        }).expect("direct");
        // Impulses must match bit-for-bit
        assert_eq!(via_trait.impulse.len(), direct.impulse.len());
        for i in 0..via_trait.impulse.len() {
            assert!((via_trait.impulse[i] - direct.impulse[i]).abs() < 1e-15,
                "iir trait differs from direct at sample {}: trait={} direct={}",
                i, via_trait.impulse[i], direct.impulse[i]);
        }
    }
}
