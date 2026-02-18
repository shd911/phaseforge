mod parser;

pub use parser::{import_measurement, parse_frd, parse_rew_txt};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Single frequency response measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub source_path: Option<PathBuf>,
    pub sample_rate: Option<f64>,
    /// Hz, sorted ascending
    pub freq: Vec<f64>,
    /// dB SPL
    pub magnitude: Vec<f64>,
    /// degrees, unwrapped (None if source had no phase data)
    pub phase: Option<Vec<f64>>,
    pub metadata: MeasurementMetadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasurementMetadata {
    pub date: Option<String>,
    pub mic: Option<String>,
    pub notes: Option<String>,
    /// Octave fraction of smoothing applied at source
    pub smoothing: Option<f64>,
}
