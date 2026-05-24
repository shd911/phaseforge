//! b140.12 — Phase 2 (lite): single Rust source of truth for the
//! FIR-pipeline routing predicate.
//!
//! Mirrors `src/lib/fir-routing.ts::pickFirRoute` exactly. The
//! `pipeline_contract` integration test asserts JS↔Rust parity on a
//! table of inputs — any drift between this function and the JS one
//! fails that test on the first disagreeing fixture.
//!
//! Why "lite": only the routing decision is unified here. Wrapping the
//! two pipelines behind a unified `trait FirPipeline` with internal
//! preprocessing (target evaluation, PEQ folding, phase composition) is
//! deferred until at least one production caller wants that surface.
//! Today the dispatch happens in JS before the Tauri boundary; each
//! Tauri command (`generate_model_fir`, `generate_model_fir_iir`) is
//! single-purpose. Centralising the routing predicate is the
//! foundation; the trait can land in a later promt without changing
//! any downstream behaviour.

use crate::target::{FilterConfig, FilterType};
use super::types::FirConfig;

/// Which FIR pipeline should be invoked for a given band configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `iir_path::generate_min_phase_fir_iir` — bit-exact phase via
    /// DigitalBiquad cascade. Restricted to min-phase main + no subsonic
    /// + every active crossover is LR / Butterworth / Custom.
    Iir,
    /// `generate_model_fir` (FFT cepstral). Used for Gaussian, Bessel,
    /// linear-phase main, composite + subsonic, and custom measured
    /// targets.
    Cepstral,
}

/// Pick the FIR pipeline for the given band configuration.
///
/// Pure function — no allocation, no global state, safe to call from
/// any thread. Reads `linear_phase_main` and `subsonic_cutoff_hz` from
/// the supplied `FirConfig`; HP/LP filter types come from the optional
/// crossover configs.
///
/// Contract (kept in lockstep with `src/lib/fir-routing.ts`):
///   - `linear_phase_main = true`            → Cepstral
///   - `subsonic_cutoff_hz = Some(_)`        → Cepstral
///   - any active filter is Gaussian/Bessel  → Cepstral
///   - everything else                        → Iir
pub fn route_for(
    hp: Option<&FilterConfig>,
    lp: Option<&FilterConfig>,
    fir_config: &FirConfig,
) -> Route {
    if fir_config.linear_phase_main { return Route::Cepstral; }
    if fir_config.subsonic_cutoff_hz.is_some() { return Route::Cepstral; }
    if !is_iir_realizable(hp) { return Route::Cepstral; }
    if !is_iir_realizable(lp) { return Route::Cepstral; }
    Route::Iir
}

/// Single-filter realisability check for the IIR cascade.
fn is_iir_realizable(f: Option<&FilterConfig>) -> bool {
    match f {
        None => true,
        Some(c) => matches!(
            c.filter_type,
            FilterType::LinkwitzRiley | FilterType::Butterworth | FilterType::Custom
        ),
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fir::{PhaseMode, WindowType};

    fn cfg(linear_main: bool, subsonic: Option<f64>) -> FirConfig {
        FirConfig {
            taps: 4096, sample_rate: 48_000.0,
            max_boost_db: 18.0, noise_floor_db: -60.0,
            window: WindowType::Hann, phase_mode: PhaseMode::Composite,
            iterations: 3, freq_weighting: true,
            narrowband_limit: true, nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
            linear_phase_main: linear_main,
            subsonic_cutoff_hz: subsonic,
        }
    }

    fn flt(ft: FilterType) -> FilterConfig {
        FilterConfig {
            filter_type: ft, order: 4, freq_hz: 1000.0, shape: Some(1.0),
            linear_phase: false, q: Some(0.707), subsonic_protect: None,
        }
    }

    #[test]
    fn iir_route_for_pure_lr_bw_custom() {
        let c = cfg(false, None);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, &c), Route::Iir);
        assert_eq!(route_for(None, Some(&flt(FilterType::Butterworth)), &c), Route::Iir);
        assert_eq!(route_for(Some(&flt(FilterType::Custom)), Some(&flt(FilterType::Custom)), &c), Route::Iir);
        assert_eq!(route_for(None, None, &c), Route::Iir);
    }

    #[test]
    fn cepstral_route_when_linear_main() {
        let c = cfg(true, None);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, &c), Route::Cepstral);
    }

    #[test]
    fn cepstral_route_when_subsonic_active() {
        let c = cfg(false, Some(80.0 / 8.0));
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, &c), Route::Cepstral);
    }

    #[test]
    fn cepstral_route_when_gaussian_or_bessel_present() {
        let c = cfg(false, None);
        assert_eq!(route_for(Some(&flt(FilterType::Gaussian)), None, &c), Route::Cepstral);
        assert_eq!(route_for(None, Some(&flt(FilterType::Bessel)), &c), Route::Cepstral);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), Some(&flt(FilterType::Gaussian)), &c), Route::Cepstral);
    }
}
