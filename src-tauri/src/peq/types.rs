// Parametric EQ types and constants

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn default_peaking() -> PeqFilterType {
    PeqFilterType::Peaking
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExclusionZone {
    #[serde(alias = "startHz")]
    pub start_hz: f64,
    #[serde(alias = "endHz")]
    pub end_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PeqFilterType {
    Peaking,
    LowShelf,
    HighShelf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqBand {
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_peaking")]
    pub filter_type: PeqFilterType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqConfig {
    /// Maximum number of PEQ bands to generate (default: 20)
    pub max_bands: usize,
    /// Convergence tolerance in dB (default: 1.0, range: 0.5..3.0)
    pub tolerance_db: f64,
    /// Peak bias: weight positive errors more (cuts preferred, default: 1.5)
    pub peak_bias: f64,
    /// Maximum boost allowed in dB (default: 6.0)
    pub max_boost_db: f64,
    /// Maximum cut allowed in dB (default: 18.0)
    pub max_cut_db: f64,
    /// Frequency range for PEQ operation: (f_low, f_high) from HP/LP crossover
    pub freq_range: (f64, f64),
    /// Optional fixed smoothing fraction for error curve (e.g. 1/6 octave).
    /// If None, uses variable smoothing (default PEQ behavior).
    #[serde(default)]
    pub smoothing_fraction: Option<f64>,
    /// Minimum distance between bands in octaves (default: 1/3 octave).
    /// Smaller values allow denser band placement (useful for HF correction).
    #[serde(default)]
    pub min_band_distance_oct: Option<f64>,
    /// Hybrid mode: flat target at avg measurement level, uniform weights.
    /// When true, PEQ optimizes to a straight line across ALL frequencies
    /// (no 3-zone composite, no ERB weighting). Default: false.
    #[serde(default)]
    pub hybrid: bool,
    /// L2 gain regularization coefficient (0.0 = off, 0.0..1.0).
    /// Penalizes large filter amplitudes: cost += lambda x Sigma gain_i^2.
    /// Higher values produce more uniform, conservative corrections.
    #[serde(default)]
    pub gain_regularization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqResult {
    pub bands: Vec<PeqBand>,
    pub max_error_db: f64,
    pub iterations: u32,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub(crate) const SAMPLE_RATE: f64 = 48000.0;
/// Minimum distance between two PEQ bands in octaves
pub(crate) const MIN_BAND_DISTANCE_OCT: f64 = 0.333; // 1/3 octave
/// Q limits
pub(crate) const Q_MIN: f64 = 0.5;
pub(crate) const Q_MAX: f64 = 10.0;

// LMA-specific constants
/// Maximum Q above LP crossover (phase-safe wide filters)
pub(crate) const Q_MAX_ABOVE_LP: f64 = 2.5;
/// Maximum LMA iterations per optimization round
pub(crate) const LMA_MAX_ITER: usize = 50;
/// LMA damping factor upper bound (stop if stuck)
pub(crate) const LMA_LAMBDA_MAX: f64 = 1e6;
/// LMA convergence threshold (relative step size)
pub(crate) const LMA_CONVERGENCE: f64 = 1e-4;
/// Minimum band gain to keep (dB)
pub(crate) const LMA_MIN_GAIN_DB: f64 = 0.2;
/// Band addition residual multiplier for weighted threshold
pub(crate) const LMA_ADD_BAND_FACTOR: f64 = 1.5;
