// FIR correction engine: correction spectrum, minimum-phase via Hilbert, IFFT, windowing, WAV export

mod types;
mod windowing;
mod helpers;
mod wav;
mod legacy;
mod cepstral;
pub mod iir_path;
pub mod dispatch;
pub mod pipeline;

pub use types::*;
pub use dispatch::{Route, route_for};
pub use wav::{export_wav_f32, export_wav_f64};
pub use legacy::{generate_fir, generate_hybrid_fir};
pub use cepstral::generate_model_fir;
pub use pipeline::{FirPipeline, IirAnalyticalPipeline, CepstralFftPipeline, pick_pipeline};
// b140.13.2: production code in `mod.rs` no longer imports these — the
// last in-module pipeline (`generate_model_fir`) moved to
// `fir/cepstral.rs`. The submodule re-exports below stay so the in-file
// `#[cfg(test)] mod tests` can resolve helpers via `super::*` without
// each test having to spell out the path.
#[cfg(test)]
pub(crate) use windowing::*;
#[cfg(test)]
pub(crate) use helpers::*;
#[cfg(test)]
use crate::dsp::minimum_phase_from_magnitude;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Recommend tap count based on lowest frequency and sample rate.
///
/// Formula: next power of 2 ≥ 3 × sample_rate / lowest_freq, then clamp to standard set.
pub fn recommend_taps(lowest_freq: f64, sample_rate: f64) -> usize {
    let standard = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
    let desired = (3.0 * sample_rate / lowest_freq.max(10.0)) as usize;
    let pow2 = desired.next_power_of_two();
    // Find the smallest standard tap count >= pow2
    *standard.iter().find(|&&s| s >= pow2).unwrap_or(&262144)
}

// b140.13:   `export_wav_f32` / `export_wav_f64`  → `fir/wav.rs`
// b140.13.1: `generate_fir` / `generate_hybrid_fir` → `fir/legacy.rs`
// b140.13.2: `generate_model_fir`                  → `fir/cepstral.rs`
// b140.13.3: in-file integration tests              → `fir/tests.rs`
// All re-exported above; call sites keep the same `fir::*` paths.

#[cfg(test)]
mod tests;
