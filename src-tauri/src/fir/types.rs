// FIR correction engine: data types and configuration structs

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
    pub taps: usize,          // 4096..1048576
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
    // b141.6 (audit): time_ms removed from the IPC payload — it was a pure
    // linear ramp (i * 1000 / sample_rate), ~0.5-1.3 MB of JSON per call at
    // 65k taps. The frontend derives it from taps + sample_rate.
    pub realized_mag: Vec<f64>,
    pub realized_phase: Vec<f64>,
    pub taps: usize,
    pub causality: f64,
    pub sample_rate: f64,
    pub norm_db: f64,
    /// b141.19 (audit): leading zeros in the shipped `impulse` — the band's
    /// latency in samples. Every route aims for N/2 so that bands share a
    /// latency and a crossover survives; the shift is capped so no tail
    /// content above -100 dB is dropped, so a long LF tail on few taps ends
    /// up with less. The caller needs that number to warn about the resulting
    /// desync instead of being promised it never happens.
    #[serde(default)]
    pub wav_delay_samples: usize,
    /// b141.40: corner of the zero-phase ultrasonic low-pass applied to this
    /// FIR (`fir::ultrasonic`), None when the export rate is below 88.2 kHz.
    #[serde(default)]
    pub ultrasonic_lp_hz: Option<f64>,
}
