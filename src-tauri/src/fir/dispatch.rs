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
//! Since b141.70 the dispatch happens in Rust (`fir::band::generate_band_fir`,
//! via [`route_for_target`]); the per-route Tauri commands were removed in
//! b141.101.

use crate::target::{FilterConfig, FilterType, TargetCurve};
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
///   - `tilt_db_per_octave != 0`             → Cepstral
///   - any active filter is Gaussian         → Cepstral (Bessel is a cascade since b141.108)
///   - everything else                        → Iir
pub fn route_for(
    hp: Option<&FilterConfig>,
    lp: Option<&FilterConfig>,
    tilt_db_per_octave: f64,
    fir_config: &FirConfig,
) -> Route {
    if fir_config.linear_phase_main { return Route::Cepstral; }
    // b141.17 (audit): a target tilt is a constant log-slope — not a rational
    // transfer function, so the biquad cascade cannot carry it. Before this
    // guard the IIR route dropped the tilt silently: the plot drew the tilted
    // target while the exported WAV shipped flat (measured 9.8 dB apart at
    // 14.8 kHz on the stock "bk" preset, tilt = -1 dB/oct). Target shelves ARE
    // rational and stay on this route (iir_path.rs builds them into the
    // cascade); only the tilt forces the cepstral path.
    if tilt_db_per_octave != 0.0 { return Route::Cepstral; }
    if fir_config.subsonic_cutoff_hz.is_some() { return Route::Cepstral; }
    if !is_iir_realizable(hp) { return Route::Cepstral; }
    if !is_iir_realizable(lp) { return Route::Cepstral; }
    // b141.101 (audit 2026-10-23 1.4): a crossover at or above
    // SECTION_MAX_FRACTION·sr has no digital biquad (the IIR route failed the
    // whole export with «fc out of range»). The cepstral route realises what
    // the plot draws below Nyquist — the filter's in-band rolloff.
    let sr = fir_config.sample_rate;
    if [hp, lp].into_iter().flatten().any(|f| !super::section_realisable(f.freq_hz, sr)) {
        return Route::Cepstral;
    }
    Route::Iir
}

/// [`route_for`] for a whole target: a shelf at or above
/// SECTION_MAX_FRACTION·sr also leaves the IIR route (its RBJ biquad is
/// unstable there, b141.101 audit 2026-10-23 1.1); the cepstral route
/// realises the analog shelf the plot draws.
pub fn route_for_target(target: &TargetCurve, fir_config: &FirConfig) -> Route {
    let sr = fir_config.sample_rate;
    let shelf_out = [target.low_shelf.as_ref(), target.high_shelf.as_ref()].into_iter().flatten()
        .any(|s| !super::section_realisable(s.freq_hz, sr));
    if shelf_out { return Route::Cepstral; }
    route_for(target.high_pass.as_ref(), target.low_pass.as_ref(), target.tilt_db_per_octave, fir_config)
}

/// Single-filter realisability check for the IIR cascade.
///
/// b141.2 (audit C1): a linear-phase crossover is NOT IIR-realisable. The IIR
/// cascade builds every filter as a min-phase digital biquad — it cannot honour
/// a per-filter `linear_phase = true`. A mixed band (e.g. HP min + LP linear)
/// previously routed here and was silently collapsed to min+min, so the realised
/// FIR disagreed with the displayed model. Such configs now fall through to the
/// cepstral path, which consumes the per-filter model phase.
fn is_iir_realizable(f: Option<&FilterConfig>) -> bool {
    match f {
        None => true,
        Some(c) => !c.linear_phase && matches!(
            c.filter_type,
            FilterType::LinkwitzRiley | FilterType::Butterworth | FilterType::Custom | FilterType::Bessel
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
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, 0.0, &c), Route::Iir);
        assert_eq!(route_for(None, Some(&flt(FilterType::Butterworth)), 0.0, &c), Route::Iir);
        assert_eq!(route_for(Some(&flt(FilterType::Custom)), Some(&flt(FilterType::Custom)), 0.0, &c), Route::Iir);
        assert_eq!(route_for(None, None, 0.0, &c), Route::Iir);
    }

    #[test]
    fn cepstral_route_when_linear_main() {
        let c = cfg(true, None);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, 0.0, &c), Route::Cepstral);
    }

    #[test]
    fn cepstral_route_when_subsonic_active() {
        let c = cfg(false, Some(80.0 / 8.0));
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, 0.0, &c), Route::Cepstral);
    }

    #[test]
    fn cepstral_route_when_any_crossover_linear_phase() {
        // b141.2 (audit C1): a per-filter linear_phase crossover is not
        // IIR-realisable — mixed HP/LP phase must route to cepstral so the
        // model phase is honoured, not collapsed to min+min.
        let c = cfg(false, None);
        let mut lin_lp = flt(FilterType::LinkwitzRiley);
        lin_lp.linear_phase = true;
        let min_hp = flt(FilterType::LinkwitzRiley); // linear_phase = false
        assert_eq!(route_for(Some(&min_hp), Some(&lin_lp), 0.0, &c), Route::Cepstral);
        // symmetric: linear HP + min LP
        let mut lin_hp = flt(FilterType::LinkwitzRiley);
        lin_hp.linear_phase = true;
        let min_lp = flt(FilterType::Butterworth);
        assert_eq!(route_for(Some(&lin_hp), Some(&min_lp), 0.0, &c), Route::Cepstral);
        // pure min stays IIR
        assert_eq!(route_for(Some(&min_hp), Some(&min_lp), 0.0, &c), Route::Iir);
    }

    #[test]
    fn cepstral_route_when_gaussian_present_bessel_is_a_cascade() {
        let c = cfg(false, None);
        assert_eq!(route_for(Some(&flt(FilterType::Gaussian)), None, 0.0, &c), Route::Cepstral);
        // b141.108: min-phase Bessel is a bilinear cascade (IIR route).
        assert_eq!(route_for(None, Some(&flt(FilterType::Bessel)), 0.0, &c), Route::Iir);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), Some(&flt(FilterType::Gaussian)), 0.0, &c), Route::Cepstral);
    }

    /// b141.101: a crossover or shelf at/above 0.49·sr leaves the IIR route.
    #[test]
    fn section_above_the_limit_forces_cepstral() {
        let c = cfg(false, None); // 48 kHz
        let mut lp = flt(FilterType::LinkwitzRiley);
        lp.freq_hz = 30_000.0;
        assert_eq!(route_for(None, Some(&lp), 0.0, &c), Route::Cepstral);
        lp.freq_hz = 23_000.0;
        assert_eq!(route_for(None, Some(&lp), 0.0, &c), Route::Iir);
        let mut t = TargetCurve { reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: Some(flt(FilterType::LinkwitzRiley)), low_pass: None, low_shelf: None,
            high_shelf: Some(crate::target::ShelfConfig { freq_hz: 26_000.0, gain_db: 6.0, q: 0.707 }) };
        assert_eq!(route_for_target(&t, &c), Route::Cepstral);
        t.high_shelf.as_mut().unwrap().freq_hz = 10_000.0;
        assert_eq!(route_for_target(&t, &c), Route::Iir);
    }

    /// b141.17 (audit): a non-zero target tilt cannot be expressed by the
    /// biquad cascade, so it must leave the IIR route.
    #[test]
    fn tilt_forces_cepstral() {
        let c = cfg(false, None);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, -1.0, &c), Route::Cepstral);
        assert_eq!(route_for(None, None, 0.5, &c), Route::Cepstral);
        assert_eq!(route_for(Some(&flt(FilterType::LinkwitzRiley)), None, 0.0, &c), Route::Iir);
    }
}
