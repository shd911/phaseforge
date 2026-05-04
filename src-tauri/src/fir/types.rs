// FIR correction engine: data types and configuration structs

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Per-filter Gaussian info for MixedPhase mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianFilterInfo {
    pub freq_hz: f64,
    pub shape: f64,
    pub is_lowpass: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhaseMode {
    MinimumPhase,
    LinearPhase,
    MixedPhase,
    HybridPhase, // min-phase correction + linear-phase filter
    /// b139.4a: respect the user's linear-phase choice for the main filter
    /// while keeping any subsonic-protect contribution minimum-phase. The
    /// caller sets `subsonic_cutoff_hz = Some(fc/8)` and `linear_phase_main`
    /// per UI checkbox; Rust splits the magnitude (`base = total - subsonic`)
    /// and recombines two phases.
    Composite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    // Basic / classical
    Rectangular,
    Bartlett,
    Hann,
    Hamming,
    Blackman,
    // Blackman-Harris family
    ExactBlackman,
    BlackmanHarris,
    Nuttall3,
    Nuttall4,
    FlatTop,
    // Parametric
    Kaiser,
    DolphChebyshev,
    Gaussian,
    Tukey,
    // Special
    Lanczos,
    Poisson,
    HannPoisson,
    Bohman,
    Cauchy,
    Riesz,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirConfig {
    pub taps: usize,          // 4096..262144
    pub sample_rate: f64,     // e.g. 48000
    pub max_boost_db: f64,    // e.g. 18.0
    pub noise_floor_db: f64,  // e.g. -60.0
    pub window: WindowType,
    pub phase_mode: PhaseMode,
    #[serde(default = "default_iterations")]
    pub iterations: usize,                    // iterative WLS passes (0=off, 1-10)
    #[serde(default = "default_true")]
    pub freq_weighting: bool,                 // frequency-dependent WLS weights
    #[serde(default = "default_true")]
    pub narrowband_limit: bool,               // narrowband boost limiting
    #[serde(default = "default_nb_smoothing")]
    pub nb_smoothing_oct: f64,                // smoothing width in octaves (e.g. 1/3)
    #[serde(default = "default_nb_max_excess")]
    pub nb_max_excess_db: f64,                // max dB above smoothed curve
    #[serde(default)]
    pub gaussian_min_phase_filters: Vec<GaussianFilterInfo>,
    /// b139.4a Composite mode: user's linear-phase choice for the main filter.
    /// Ignored when phase_mode != Composite.
    #[serde(default)]
    pub linear_phase_main: bool,
    /// b139.4a Composite mode: subsonic Butterworth-8 corner (typically fc/8).
    /// Set to None when subsonic_protect is off; the Composite path then
    /// degenerates to {Linear,Min}Phase based on linear_phase_main.
    #[serde(default)]
    pub subsonic_cutoff_hz: Option<f64>,
}

pub(crate) fn default_iterations() -> usize { 3 }
pub(crate) fn default_true() -> bool { true }
pub(crate) fn default_nb_smoothing() -> f64 { 0.333 }
pub(crate) fn default_nb_max_excess() -> f64 { 6.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirResult {
    pub impulse: Vec<f64>,
    pub time_ms: Vec<f64>,
    pub taps: usize,
    pub sample_rate: f64,
    pub norm_db: f64,
    pub causality: f64,       // 0.0-1.0: ratio of post-peak energy to total (1.0 = perfectly causal)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirModelResult {
    pub impulse: Vec<f64>,
    pub time_ms: Vec<f64>,
    pub realized_mag: Vec<f64>,
    pub realized_phase: Vec<f64>,
    pub taps: usize,
    pub causality: f64,
    pub sample_rate: f64,
    pub norm_db: f64,
}
