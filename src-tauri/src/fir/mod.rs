// FIR correction engine: correction spectrum, minimum-phase via Hilbert, IFFT, windowing, WAV export

mod types;
mod windowing;
mod helpers;
mod wav;
// b141.2: `legacy` (generate_fir) is no longer reachable
// from production — the frontend routes exclusively through generate_model_fir /
// generate_model_fir_iir, and the two dead Tauri commands were removed. The
// module is retained only as the baseline for the golden-snapshot tests in
// `tests.rs`, so it is gated behind cfg(test).
#[cfg(test)]
mod legacy;
mod cepstral;
pub mod iir_path;
pub mod dispatch;
pub mod pipeline;

pub use types::*;
pub use dispatch::{Route, route_for};
pub use wav::{export_wav_f32, export_wav_f64};
#[cfg(test)]
pub use legacy::generate_fir;
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

/// Tap counts the FFT backend accepts: powers of two in 32..=1048576.
/// The vDSP backend `assert!`s on non-powers-of-two, and a panic inside an
/// async Tauri command leaves the IPC promise pending forever, so every
/// entry point (project load, both FIR generators) checks this first.
///
/// The ceiling is 2^20 since b141.35 (was 2^18). It is a product limit, not
/// a backend one: `vDSP_create_fftsetupD` returns a working radix-2 setup up
/// to log2n=23, and `generate_half_window` asks for 2x the tap count, so the
/// largest transform a 1048576-tap export runs is 2^21 (~18 ms, measured).
pub const MAX_TAPS: usize = 1_048_576;
pub fn taps_valid(taps: usize) -> bool {
    taps.is_power_of_two() && (32..=MAX_TAPS).contains(&taps)
}

/// Recommend tap count based on lowest frequency and sample rate.
///
/// Formula: next power of 2 ≥ 3 × sample_rate / lowest_freq, then clamp to standard set.
pub fn recommend_taps(lowest_freq: f64, sample_rate: f64) -> usize {
    let standard = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
    let desired = (3.0 * sample_rate / lowest_freq.max(10.0)) as usize;
    let pow2 = desired.next_power_of_two();
    // Find the smallest standard tap count >= pow2
    *standard.iter().find(|&&s| s >= pow2).unwrap_or(&MAX_TAPS)
}

// b140.13:   `export_wav_f32` / `export_wav_f64`  → `fir/wav.rs`
// b140.13.1: `generate_fir` / `generate_hybrid_fir` → `fir/legacy.rs`
// b140.13.2: `generate_model_fir`                  → `fir/cepstral.rs`
// b140.13.3: in-file integration tests              → `fir/tests.rs`
// All re-exported above; call sites keep the same `fir::*` paths.

#[cfg(test)]
mod tests;
