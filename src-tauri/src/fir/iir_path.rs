//! b140.7: IIR-based min-phase FIR pipeline.
//!
//! For non-Gaussian Min-Phase user choice (linear_phase_main = false,
//! no subsonic protect, no Composite linear), we synthesise the FIR
//! impulse from an analytical IIR cascade rather than the cepstral
//! FFT pipeline. Because every section is built from analog filter
//! coefficients with all poles in the left-half plane and bilinear-
//! transformed to digital, the resulting impulse is **causal by
//! construction** — peak at sample 0, no constant group delay
//! artefact, and magnitude follows the analytical target exactly
//! (within bilinear pre-warping accuracy).
//!
//! Scope b140.7: LR / Butterworth / Custom HP+LP, optional PEQ
//! peaking sections, no Bessel (routes to FFT path), no Gaussian,
//! no subsonic protect, no Composite linear-phase. The FFT path
//! (`generate_model_fir`) is left untouched and serves every other
//! configuration.
//!
//! References used while writing this:
//! - Robert Bristow-Johnson, "Cookbook formulae for audio EQ biquad
//!   filter coefficients" (RBJ).
//! - Standard analog → digital bilinear transform with pre-warping.

use std::f64::consts::PI;

use tracing::info;

use crate::dsp::fft::FftEngine;
use crate::error::AppError;
use crate::peq::{PeqBand, PeqFilterType};
use crate::target::{FilterConfig, FilterType};
use num_complex::Complex64;

use super::types::*;

// ---------------------------------------------------------------------------
// Routing predicate — when can we use the IIR path?
// ---------------------------------------------------------------------------

/// b140.7: a filter is "IIR-realizable" via this module when it's a
/// rational analog form (Linkwitz-Riley, Butterworth, or Custom 2nd-order).
/// Bessel and Gaussian fall back to the FFT cepstral path.
pub fn iir_realizable(cfg: &FilterConfig) -> bool {
    matches!(cfg.filter_type,
        FilterType::LinkwitzRiley
        | FilterType::Butterworth
        | FilterType::Custom)
}

// ---------------------------------------------------------------------------
// Digital biquad (Direct Form I)
// ---------------------------------------------------------------------------

/// One second-order section in the standard biquad form
/// `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`.
/// Sections with `b2 = a2 = 0` represent first-order filters (LR-2 needs
/// these, Butterworth odd orders too).
#[derive(Clone, Debug)]
pub struct DigitalBiquad {
    pub b0: f64, pub b1: f64, pub b2: f64,
    pub a1: f64, pub a2: f64,
}

impl DigitalBiquad {
    /// One-sample step in Direct Form I.
    /// `state = [x_{n-1}, x_{n-2}, y_{n-1}, y_{n-2}]`.
    #[inline]
    pub fn process(&self, x: f64, state: &mut [f64; 4]) -> f64 {
        let y = self.b0 * x + self.b1 * state[0] + self.b2 * state[1]
              - self.a1 * state[2] - self.a2 * state[3];
        state[1] = state[0]; state[0] = x;
        state[3] = state[2]; state[2] = y;
        y
    }
}

// ---------------------------------------------------------------------------
// Analog → digital via bilinear transform
// ---------------------------------------------------------------------------

/// Pre-warped corner: bilinear transform compresses high frequencies, so
/// design the analog filter at this warped Fc to land exactly on `fc` in
/// the digital domain. `f_warped = (sr/π) · tan(π · fc / sr)`.
#[inline]
fn prewarp(fc: f64, sr: f64) -> f64 {
    (sr / PI) * (PI * fc / sr).tan()
}

/// Build a 2nd-order analog low-pass at warped corner ωc = 2π·fc with given
/// pole-Q: `H(s) = ωc² / (s² + (ωc/Q)·s + ωc²)` → bilinear → digital biquad.
fn lp_biquad_q(fc: f64, q: f64, sr: f64) -> DigitalBiquad {
    let fc_w = prewarp(fc, sr);
    let omega = 2.0 * PI * fc_w;
    let k = 2.0 * sr;
    let k2 = k * k;
    let omega2 = omega * omega;
    let alpha = omega / q;

    // Analog coefficients: numerator (b0,b1,b2) = (0,0,ωc²), denominator
    // (1, ωc/Q, ωc²). Bilinear (s → K·(z-1)/(z+1), K = 2·sr) and multiply
    // numerator/denominator by (z+1)² gives:
    //   B0 = ωc²,   B1 = 2·ωc²,   B2 = ωc²
    //   A0 = K² + α·K + ωc²
    //   A1 = -2·K² + 2·ωc²
    //   A2 = K² - α·K + ωc²
    let b0 = omega2;
    let b1 = 2.0 * omega2;
    let b2 = omega2;
    let a0 = k2 + alpha * k + omega2;
    let a1 = -2.0 * k2 + 2.0 * omega2;
    let a2 = k2 - alpha * k + omega2;

    DigitalBiquad {
        b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
        a1: a1 / a0, a2: a2 / a0,
    }
}

/// Build a 2nd-order analog high-pass: `H(s) = s² / (s² + (ωc/Q)·s + ωc²)`
/// → bilinear → digital biquad.
fn hp_biquad_q(fc: f64, q: f64, sr: f64) -> DigitalBiquad {
    let fc_w = prewarp(fc, sr);
    let omega = 2.0 * PI * fc_w;
    let k = 2.0 * sr;
    let k2 = k * k;
    let omega2 = omega * omega;
    let alpha = omega / q;

    // Analog numerator (1, 0, 0) → bilinear:
    //   B0 = K²,   B1 = -2·K²,   B2 = K²
    //   A0/A1/A2 same as LP.
    let b0 = k2;
    let b1 = -2.0 * k2;
    let b2 = k2;
    let a0 = k2 + alpha * k + omega2;
    let a1 = -2.0 * k2 + 2.0 * omega2;
    let a2 = k2 - alpha * k + omega2;

    DigitalBiquad {
        b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
        a1: a1 / a0, a2: a2 / a0,
    }
}

/// First-order analog low-pass `H(s) = ωc / (s + ωc)` → bilinear → digital.
/// Used for Butterworth/LR with odd primary order (LR2 = 2× first-order LP).
fn lp_first_order(fc: f64, sr: f64) -> DigitalBiquad {
    let fc_w = prewarp(fc, sr);
    let omega = 2.0 * PI * fc_w;
    let k = 2.0 * sr;
    // (z+1) in numerator, denominator K(z-1) + ωc(z+1) = (K+ωc)·z + (ωc-K)
    let a0 = k + omega;
    DigitalBiquad {
        b0: omega / a0, b1: omega / a0, b2: 0.0,
        a1: (omega - k) / a0, a2: 0.0,
    }
}

/// First-order analog high-pass `H(s) = s / (s + ωc)` → bilinear → digital.
fn hp_first_order(fc: f64, sr: f64) -> DigitalBiquad {
    let fc_w = prewarp(fc, sr);
    let omega = 2.0 * PI * fc_w;
    let k = 2.0 * sr;
    let a0 = k + omega;
    DigitalBiquad {
        b0: k / a0, b1: -k / a0, b2: 0.0,
        a1: (omega - k) / a0, a2: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Butterworth Q tables (per biquad section; cascade gives Butterworth-N)
// ---------------------------------------------------------------------------

/// For Butterworth order N, the cascade is `floor(N/2)` biquads (with the
/// listed Q values) plus one first-order section if N is odd.
/// Q_k = 1 / (2 · sin(π·(2k − 1) / (2N))) for k = 1..floor(N/2).
fn butterworth_qs(order: u8) -> Vec<f64> {
    let n = order as usize;
    let pairs = n / 2;
    (1..=pairs)
        .map(|k| 1.0 / (2.0 * (PI * (2.0 * k as f64 - 1.0) / (2.0 * n as f64)).sin()))
        .collect()
}

fn butterworth_has_first_order(order: u8) -> bool { order % 2 == 1 }

// ---------------------------------------------------------------------------
// Filter cascade builders (LR, BU, Custom)
// ---------------------------------------------------------------------------

/// Build the digital biquad cascade for a single FilterConfig (LP if
/// `is_lp`, HP otherwise). LR-N is realised as TWO Butterworth-(N/2)
/// cascades concatenated.
pub fn build_filter_cascade(
    cfg: &FilterConfig, is_lp: bool, sr: f64,
) -> Result<Vec<DigitalBiquad>, String> {
    let fc = cfg.freq_hz;
    if fc <= 0.0 || fc >= sr / 2.0 {
        return Err(format!("filter fc={} out of range for sr={}", fc, sr));
    }
    match cfg.filter_type {
        FilterType::Butterworth => Ok(build_butterworth_cascade(cfg.order, fc, is_lp, sr)),
        FilterType::LinkwitzRiley => {
            // PhaseForge convention (target/mod.rs:487-490): LR-N response =
            // BU-N magnitude × 2 (dB) and BU-N phase × 2. To match this in
            // the time-domain cascade we concatenate TWO Butterworth-N
            // cascades (so realised |H| and ∠H are squared / doubled-in-dB
            // respectively). This is the historical model the rest of the
            // codebase plots and REW compares against.
            let mut bs = build_butterworth_cascade(cfg.order, fc, is_lp, sr);
            bs.extend(build_butterworth_cascade(cfg.order, fc, is_lp, sr));
            Ok(bs)
        }
        FilterType::Custom => {
            // Single 2nd-order biquad with user-supplied Q (defaults to
            // Butterworth Q = 1/√2). Order field is informational here.
            let q = cfg.q.unwrap_or(std::f64::consts::FRAC_1_SQRT_2);
            Ok(vec![if is_lp { lp_biquad_q(fc, q, sr) } else { hp_biquad_q(fc, q, sr) }])
        }
        FilterType::Bessel | FilterType::Gaussian => {
            Err(format!("filter type {:?} is not IIR-realizable in b140.7", cfg.filter_type))
        }
    }
}

fn build_butterworth_cascade(order: u8, fc: f64, is_lp: bool, sr: f64) -> Vec<DigitalBiquad> {
    let mut out = Vec::with_capacity(order as usize / 2 + 1);
    if butterworth_has_first_order(order) {
        out.push(if is_lp { lp_first_order(fc, sr) } else { hp_first_order(fc, sr) });
    }
    for q in butterworth_qs(order) {
        out.push(if is_lp { lp_biquad_q(fc, q, sr) } else { hp_biquad_q(fc, q, sr) });
    }
    out
}

// ---------------------------------------------------------------------------
// PEQ peaking biquad (RBJ cookbook digital form, direct)
// ---------------------------------------------------------------------------

/// Build the digital biquad for a single PEQ band. Peaking / shelving forms
/// are taken from the RBJ Audio EQ Cookbook (already digital — no bilinear
/// step needed).
pub fn build_peq_biquad(band: &PeqBand, sr: f64) -> DigitalBiquad {
    let f0 = band.freq_hz.max(1.0);
    let q = band.q.max(1e-6);
    let a = 10f64.powf(band.gain_db / 40.0);
    let omega = 2.0 * PI * f0 / sr;
    let alpha = omega.sin() / (2.0 * q);
    let cos_w = omega.cos();
    match band.filter_type {
        PeqFilterType::Peaking => {
            let b0 = 1.0 + alpha * a;
            let b1 = -2.0 * cos_w;
            let b2 = 1.0 - alpha * a;
            let a0 = 1.0 + alpha / a;
            let a1 = -2.0 * cos_w;
            let a2 = 1.0 - alpha / a;
            DigitalBiquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
        }
        PeqFilterType::LowShelf => {
            let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
            let b0 =      a * ((a + 1.0) - (a - 1.0) * cos_w + two_sqrt_a_alpha);
            let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w);
            let b2 =      a * ((a + 1.0) - (a - 1.0) * cos_w - two_sqrt_a_alpha);
            let a0 =           (a + 1.0) + (a - 1.0) * cos_w + two_sqrt_a_alpha;
            let a1 = -2.0 *   ((a - 1.0) + (a + 1.0) * cos_w);
            let a2 =           (a + 1.0) + (a - 1.0) * cos_w - two_sqrt_a_alpha;
            DigitalBiquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
        }
        PeqFilterType::HighShelf => {
            let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
            let b0 =       a * ((a + 1.0) + (a - 1.0) * cos_w + two_sqrt_a_alpha);
            let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w);
            let b2 =       a * ((a + 1.0) + (a - 1.0) * cos_w - two_sqrt_a_alpha);
            let a0 =            (a + 1.0) - (a - 1.0) * cos_w + two_sqrt_a_alpha;
            let a1 =  2.0 *   ((a - 1.0) - (a + 1.0) * cos_w);
            let a2 =            (a + 1.0) - (a - 1.0) * cos_w - two_sqrt_a_alpha;
            DigitalBiquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
        }
    }
}

// ---------------------------------------------------------------------------
// Cascade impulse + windowing
// ---------------------------------------------------------------------------

/// Apply a unit impulse δ[n] through the cascade and return `n_taps`
/// samples. The impulse is causal by construction (analog poles in LHP →
/// bilinear-stable digital poles inside the unit circle), so impulse[0]
/// is the natural location of the peak.
pub fn cascade_impulse(biquads: &[DigitalBiquad], n_taps: usize) -> Vec<f64> {
    let mut impulse = vec![0.0_f64; n_taps];
    let mut states: Vec<[f64; 4]> = vec![[0.0; 4]; biquads.len()];
    for n in 0..n_taps {
        let mut x = if n == 0 { 1.0 } else { 0.0 };
        for (bq, state) in biquads.iter().zip(states.iter_mut()) {
            x = bq.process(x, state);
        }
        impulse[n] = x;
    }
    impulse
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

pub struct IirPathInput<'a> {
    pub freq: &'a [f64],          // log grid for realized_mag/phase output
    pub hp: Option<&'a FilterConfig>,
    pub lp: Option<&'a FilterConfig>,
    pub peq: &'a [PeqBand],
    pub config: &'a FirConfig,
}

/// b140.7: build min-phase FIR from an IIR cascade rather than cepstral
/// FFT. Returns the same `FirModelResult` shape as `generate_model_fir`
/// so the Tauri command can route transparently.
pub fn generate_min_phase_fir_iir(input: &IirPathInput) -> Result<FirModelResult, AppError> {
    let cfg = input.config;
    let sr = cfg.sample_rate;
    let n_fft = cfg.taps;
    if n_fft < 32 || sr <= 0.0 {
        return Err(AppError::Config {
            message: format!("invalid taps={} or sr={}", n_fft, sr),
        });
    }

    // 1. Build cascade: HP filter → LP filter → enabled PEQ biquads.
    let mut biquads: Vec<DigitalBiquad> = Vec::new();
    if let Some(hp) = input.hp {
        biquads.extend(build_filter_cascade(hp, false, sr)
            .map_err(|m| AppError::Config { message: m })?);
    }
    if let Some(lp) = input.lp {
        biquads.extend(build_filter_cascade(lp, true, sr)
            .map_err(|m| AppError::Config { message: m })?);
    }
    for band in input.peq.iter().filter(|p| p.enabled) {
        biquads.push(build_peq_biquad(band, sr));
    }

    info!(
        "[IIR PATH] taps={} sr={} sections={} (hp={:?} lp={:?} peq={})",
        n_fft, sr, biquads.len(),
        input.hp.map(|h| (h.filter_type.clone(), h.order, h.freq_hz)),
        input.lp.map(|l| (l.filter_type.clone(), l.order, l.freq_hz)),
        input.peq.iter().filter(|p| p.enabled).count(),
    );

    // 2. Apply impulse → raw causal IIR impulse response (truncated to n_fft).
    //    This is the analytical filter realisation in time domain. Its FFT
    //    equals the analytical frequency response by construction, so we
    //    use it directly for `realized_mag/phase` (no phase corrections).
    let raw_impulse = cascade_impulse(&biquads, n_fft);
    let n = raw_impulse.len();
    let raw_peak_idx = raw_impulse.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    // 3. Compute realised mag/phase from the RAW (un-centred) cascade
    //    impulse — its FFT phase is the analytical filter phase exactly,
    //    no shift correction needed. This is what the UI plot uses to
    //    overlay the model.
    let n_bins = n_fft / 2 + 1;
    let mut engine = FftEngine::new();
    let mut spec_raw: Vec<Complex64> = raw_impulse.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    engine.fft_forward(&mut spec_raw);
    let mut realized_mag_lin: Vec<f64> = Vec::with_capacity(n_bins);
    let mut realized_phase_lin: Vec<f64> = Vec::with_capacity(n_bins);
    for c in spec_raw.iter().take(n_bins) {
        let amp = c.norm();
        let mag_db = if amp > 1e-20 { 20.0 * amp.log10() } else { -400.0 };
        realized_mag_lin.push(mag_db);
        realized_phase_lin.push(c.arg() * 180.0 / PI);
    }
    // Unwrap phase across linear bins for smooth interpolation onto the log grid.
    for i in 1..realized_phase_lin.len() {
        let diff = realized_phase_lin[i] - realized_phase_lin[i - 1];
        if diff > 180.0 {
            realized_phase_lin[i] -= 360.0 * ((diff + 180.0) / 360.0).floor();
        } else if diff < -180.0 {
            realized_phase_lin[i] += 360.0 * ((-diff + 180.0) / 360.0).floor();
        }
    }
    let lin_freq: Vec<f64> = (0..n_bins).map(|k| sr * k as f64 / n_fft as f64).collect();
    let mut realized_mag = crate::dsp::interp_1d(&lin_freq, &realized_mag_lin, input.freq);
    let mut realized_phase = crate::dsp::interp_1d(&lin_freq, &realized_phase_lin, input.freq);
    // b140.7.7: at sr < 88 k the caller's log grid extends to 22.8 / 40 k
    // but Nyquist is sr/2 (< grid max). Bins above Nyquist get extrapolated
    // garbage from `interp_1d` — clamp to noise floor / 0.
    let nyquist = sr / 2.0;
    for (i, &f) in input.freq.iter().enumerate() {
        if f > nyquist {
            realized_mag[i] = cfg.noise_floor_db;
            realized_phase[i] = 0.0;
        }
    }

    // 4. b140.7.10: build the WAV impulse separately from the plot data.
    //    Pad with exactly N/2 leading zeros (REPhase convention) so REW's
    //    WAV-import path auto-locates the peak near the centre — REW
    //    mis-handled peak-at-start impulses on sr = 44.1/48 kHz (notch
    //    artefacts on otherwise correct content). The padding is applied
    //    only to the WAV output; UI plot keeps reading from the raw
    //    cascade (no phase correction needed → no parity / wrap artefacts
    //    near Nyquist, no over-subtraction of LP natural delay).
    let half = n / 2;
    let shift = half;
    let mut wav_impulse: Vec<f64> = if shift > 0 && shift < n {
        let mut out = vec![0.0_f64; n];
        let copy_len = n - shift;
        out[shift..shift + copy_len].copy_from_slice(&raw_impulse[..copy_len]);
        out
    } else {
        raw_impulse.clone()
    };
    info!(
        "[IIR PATH] WAV centered: raw_peak={} → wav_peak={} (shift=N/2={})",
        raw_peak_idx, raw_peak_idx + shift, shift,
    );

    // 5. Tail taper: fade the last ~5 % of WAV samples to avoid a hard
    //    truncation glitch.
    apply_tail_taper(&mut wav_impulse);

    // 6. Passband normalisation: scale impulse so the peak realised
    //    magnitude is 0 dB. Same convention as the FFT path. Applied to
    //    BOTH the WAV impulse (so the file peaks at 0 dB) and the
    //    realised_mag display (so the plot peaks at 0 dB).
    let norm_db = realized_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let norm_db = if norm_db.is_finite() { norm_db } else { 0.0 };
    let norm_linear = 10f64.powf(-norm_db / 20.0);
    for v in wav_impulse.iter_mut() { *v *= norm_linear; }
    let realized_mag: Vec<f64> = realized_mag.iter().map(|&v| v - norm_db).collect();

    let dt_ms = 1000.0 / sr;
    let time_ms: Vec<f64> = (0..n_fft).map(|i| i as f64 * dt_ms).collect();

    // Causality computed on the *centered* WAV impulse — that's what the
    // user sees in the UI status alongside the WAV file content.
    let causality = compute_causality(&wav_impulse);
    info!(
        "[IIR PATH] norm_db={:.2}, causality={:.4} ({}%), wav_peak_idx={}",
        norm_db, causality, (causality * 100.0) as u32,
        wav_impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0),
    );

    // Rename for the return statement: `impulse` is the WAV-bound centered
    // version; `realized_mag` / `realized_phase` came from raw FFT above.
    let impulse = wav_impulse;

    Ok(FirModelResult {
        impulse,
        time_ms,
        realized_mag,
        realized_phase,
        taps: n_fft,
        sample_rate: sr,
        norm_db,
        causality,
    })
}

/// Cosine-fade the last 5 % of samples to zero. Preserves the entire
/// rising edge + main decay of the IIR impulse, only smooths the very end
/// to avoid a step discontinuity from truncation.
fn apply_tail_taper(impulse: &mut [f64]) {
    let n = impulse.len();
    if n < 20 { return; }
    let fade_len = (n / 20).max(8); // 5 %, at least 8 samples
    let start = n - fade_len;
    for i in start..n {
        let t = (i - start) as f64 / fade_len as f64;
        let w = 0.5 * (1.0 + (PI * t).cos()); // raised-cosine 1 → 0
        impulse[i] *= w;
    }
}

/// Mirror of `super::compute_causality` — kept private so this module can
/// be lifted out without touching the parent.
fn compute_causality(impulse: &[f64]) -> f64 {
    if impulse.is_empty() { return 1.0; }
    let mut peak_idx = 0;
    let mut peak_val = 0.0_f64;
    for (i, &v) in impulse.iter().enumerate() {
        let a = v.abs();
        if a > peak_val { peak_val = a; peak_idx = i; }
    }
    let total: f64 = impulse.iter().map(|v| v * v).sum();
    if total < 1e-30 { return 1.0; }
    let post: f64 = impulse[peak_idx..].iter().map(|v| v * v).sum();
    post / total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lr4_filter(fc: f64) -> FilterConfig {
        FilterConfig {
            filter_type: FilterType::LinkwitzRiley,
            order: 4, freq_hz: fc, shape: None,
            linear_phase: false, q: None, subsonic_protect: None,
        }
    }

    fn cfg_min(taps: usize, sr: f64) -> FirConfig {
        FirConfig {
            taps, sample_rate: sr,
            max_boost_db: 24.0, noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::Composite,
            iterations: 0,
            freq_weighting: false, narrowband_limit: false,
            nb_smoothing_oct: 0.333, nb_max_excess_db: 6.0,
            gaussian_min_phase_filters: vec![],
            linear_phase_main: false,
            subsonic_cutoff_hz: None,
        }
    }

    fn log_grid(n: usize, fmin: f64, fmax: f64) -> Vec<f64> {
        (0..n).map(|i| fmin * (fmax / fmin).powf(i as f64 / (n - 1) as f64)).collect()
    }

    fn peak_idx(impulse: &[f64]) -> usize {
        impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap()
    }

    /// b140.7.6: the IIR impulse is centred at sample `n_fft/2` after
    /// `cascade_impulse` (REPhase / REW convention — REW mis-handles
    /// peak-at-start at sr=44.1/48k). Tests now expect peak near N/2
    /// rather than at 0; magnitude / phase tests still validate the
    /// underlying filter shape.
    #[test]
    fn iir_lr4_lp_200_centered_at_half_sr_48k() {
        let lp = lr4_filter(200.0);
        let cfg = cfg_min(65536, 48000.0);
        let freq = log_grid(512, 5.0, 22800.0);
        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp: None, lp: Some(&lp), peq: &[], config: &cfg,
        }).expect("LR4 LP=200 IIR should succeed");
        let p = peak_idx(&out.impulse);
        let half = cfg.taps / 2;
        // Centring puts the peak within ~600 samples of N/2 (LP rise time
        // shifts it slightly above N/2).
        assert!((half.saturating_sub(50)..=half + 600).contains(&p),
            "LR4 LP=200: peak idx={} not centred near N/2={}", p, half);
    }

    #[test]
    fn iir_lr4_hp_2000_centered_at_half_sr_48k() {
        let hp = lr4_filter(2000.0);
        let cfg = cfg_min(65536, 48000.0);
        let freq = log_grid(512, 5.0, 22800.0);
        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp: Some(&hp), lp: None, peq: &[], config: &cfg,
        }).expect("LR4 HP=2000 IIR should succeed");
        let p = peak_idx(&out.impulse);
        let half = cfg.taps / 2;
        // HP cascade has peak at sample 0..5 before centring → after
        // centring lands within ±5 of N/2.
        assert!((half.saturating_sub(5)..=half + 5).contains(&p),
            "LR4 HP=2000: peak idx={} not centred near N/2={}", p, half);
    }

    #[test]
    fn iir_bp_200_2000_centered_at_half_sr_48k() {
        let hp = lr4_filter(200.0);
        let lp = lr4_filter(2000.0);
        let cfg = cfg_min(65536, 48000.0);
        let freq = log_grid(512, 5.0, 22800.0);
        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp: Some(&hp), lp: Some(&lp), peq: &[], config: &cfg,
        }).expect("BP 200-2000 IIR should succeed");
        let p = peak_idx(&out.impulse);
        let half = cfg.taps / 2;
        assert!((half.saturating_sub(50)..=half + 100).contains(&p),
            "BP 200-2000: peak idx={} not centred near N/2={}", p, half);
    }

    #[test]
    fn iir_lr4_lp_200_realized_phase_matches_target() {
        // Critical regression check for the b140.6 → b140.7 fix: realised
        // phase from the IIR impulse must match `target::evaluate` analytical
        // phase in the passband, since the model curve plotted in the UI
        // and REW both use that analytical phase.
        use crate::target::{evaluate, TargetCurve};
        let lp = lr4_filter(200.0);
        let cfg = cfg_min(65536, 48000.0);
        let freq = log_grid(512, 5.0, 22800.0);
        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp: None, lp: Some(&lp), peq: &[], config: &cfg,
        }).expect("LR4 LP=200 IIR should succeed");
        let target = TargetCurve {
            reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: None, low_pass: Some(lp), low_shelf: None, high_shelf: None,
        };
        let ref_resp = evaluate(&target, &freq);
        let mut max_err = 0.0_f64;
        let mut n_pb = 0;
        for (i, &f) in freq.iter().enumerate() {
            if f < 20.0 || f > 100.0 { continue; }
            // Phase residual modulo 360.
            let mut d = out.realized_phase[i] - ref_resp.phase[i];
            while d > 180.0 { d -= 360.0; }
            while d < -180.0 { d += 360.0; }
            let err = d.abs();
            if err > max_err { max_err = err; }
            n_pb += 1;
        }
        assert!(n_pb > 0);
        assert!(max_err < 5.0,
            "LR4 LP=200 phase error in passband: max {:.2}° > 5° (model mismatch)", max_err);
    }

    #[test]
    fn iir_lr4_lp_200_realized_mag_matches_target_in_passband() {
        // Compare realised magnitude against the analytical evaluate() output
        // in the LR4 LP passband (≤ 0.5 × Fc → −0.5 dB-ish band).
        use crate::target::{evaluate, TargetCurve};
        let lp = lr4_filter(200.0);
        let cfg = cfg_min(65536, 48000.0);
        let freq = log_grid(512, 5.0, 22800.0);
        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &freq, hp: None, lp: Some(&lp), peq: &[], config: &cfg,
        }).expect("LR4 LP=200 IIR should succeed");
        let target = TargetCurve {
            reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: None, low_pass: Some(lp), low_shelf: None, high_shelf: None,
        };
        let ref_resp = evaluate(&target, &freq);
        let mut max_err = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut n_pb = 0;
        for (i, &f) in freq.iter().enumerate() {
            if f < 20.0 || f > 100.0 { continue; } // well inside the passband
            let err = (out.realized_mag[i] - ref_resp.magnitude[i]).abs();
            if err > max_err { max_err = err; }
            sum_sq += err * err;
            n_pb += 1;
        }
        let rms = (sum_sq / n_pb as f64).sqrt();
        assert!(max_err < 1.0, "LR4 LP=200 passband peak err {:.3} dB > 1 dB", max_err);
        assert!(rms < 0.5, "LR4 LP=200 passband RMS err {:.3} dB > 0.5 dB", rms);
    }

    #[test]
    fn iir_bilinear_unit_test_lr2_dc_gain() {
        // LR-2 LP at fc=1000, sr=48000. DC gain of cascade = 1 (0 dB). The
        // cascade impulse response sum equals DC gain.
        let lp = FilterConfig {
            filter_type: FilterType::LinkwitzRiley,
            order: 2, freq_hz: 1000.0, shape: None,
            linear_phase: false, q: None, subsonic_protect: None,
        };
        let bs = build_filter_cascade(&lp, true, 48000.0).unwrap();
        let imp = cascade_impulse(&bs, 16384);
        let dc: f64 = imp.iter().sum();
        assert!((dc - 1.0).abs() < 0.01, "LR2 LP DC gain = {:.4} (expected ≈ 1)", dc);
    }

    // -----------------------------------------------------------------
    // b140.7.11: HP IIR path acceptance suite. Mirrors user's exact REW
    // workflow — HP=2000 LR4, 65536 taps, Blackman, across all production
    // sample rates. Catches phase / impulse / WAV regressions before UI
    // verify cycles.
    // -----------------------------------------------------------------

    fn fft_mag_phase(samples: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = samples.len();
        let n_bins = n / 2 + 1;
        let mut spec: Vec<Complex64> = samples.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut engine = FftEngine::new();
        engine.fft_forward(&mut spec);
        let mut mag = Vec::with_capacity(n_bins);
        let mut phase = Vec::with_capacity(n_bins);
        for c in spec.iter().take(n_bins) {
            let amp = c.norm();
            mag.push(if amp > 1e-30 { 20.0 * amp.log10() } else { -400.0 });
            phase.push(c.arg() * 180.0 / PI);
        }
        (mag, phase)
    }

    /// Run HP=2000 LR4 through the IIR path at the given sample rate and
    /// validate the WAV impulse FFT against the analytical model evaluated
    /// at matching frequencies.
    fn run_hp2000_acceptance(sr: f64) {
        use crate::target::{evaluate, TargetCurve};

        let n_fft = 65_536_usize;
        let hp = lr4_filter(2000.0);
        let mut cfg = cfg_min(n_fft, sr);
        cfg.window = WindowType::Blackman;
        cfg.iterations = 0;

        // Caller-side log grid mirrors band-evaluator.ts: 5 .. min(40k, sr·0.95/2).
        let f_max = (40_000.0_f64).min(sr / 2.0 * 0.95);
        let log_freq = log_grid(512, 5.0, f_max);

        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &log_freq, hp: Some(&hp), lp: None, peq: &[], config: &cfg,
        }).unwrap_or_else(|e| panic!("sr={} IIR generation failed: {:?}", sr, e));

        // ===== WAV checks =====
        let wav = &out.impulse;
        let (wav_mag, _wav_phase) = fft_mag_phase(wav);
        // Re-normalise so passband peaks at 0 dB (matches `out.norm_db`).
        let wav_peak_mag = wav_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bin_for = |f: f64| ((f * n_fft as f64 / sr).round() as usize).min(wav_mag.len() - 1);

        // Magnitude reference from analytical model on the same linear bins.
        let lin_freq: Vec<f64> = (0..wav_mag.len()).map(|k| k as f64 * sr / n_fft as f64).collect();
        let target = TargetCurve {
            reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: Some(hp.clone()), low_pass: None, low_shelf: None, high_shelf: None,
        };
        let ref_resp = evaluate(&target, &lin_freq);

        // Passband checks (≥ 5 kHz): WAV magnitude (peak-normalised) within
        // 0.5 dB of analytical (peak should be ~0 dB there).
        for &f in &[5_000.0_f64, 10_000.0, 15_000.0] {
            if f >= sr / 2.0 { continue; }
            let bin = bin_for(f);
            let actual = wav_mag[bin] - wav_peak_mag;
            let expected = ref_resp.magnitude[bin] - 0.0; // analytical passband ≈ 0 dB
            let err = (actual - expected).abs();
            assert!(err < 0.5,
                "HP=2000 LR4 sr={}: passband mag at {} Hz: actual {:.2} dB, expected {:.2} dB (err {:.2})",
                sr, f, actual, expected, err);
        }
        // Corner check (2000 Hz): LR convention = -6 dB. WAV mag (peak-norm)
        // should land within 1 dB.
        {
            let bin = bin_for(2_000.0);
            let actual = wav_mag[bin] - wav_peak_mag;
            assert!((actual + 6.0).abs() < 1.0,
                "HP=2000 LR4 sr={}: corner mag at 2000 Hz: actual {:.2} dB, expected -6 dB",
                sr, actual);
        }

        // Peak position: WAV impulse should be centred near N/2 (REW happy).
        let peak_idx = wav.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap();
        let half = n_fft / 2;
        assert!(
            (peak_idx as i64 - half as i64).abs() < 50,
            "HP=2000 LR4 sr={}: WAV peak idx {} not within 50 of N/2={}",
            sr, peak_idx, half
        );
    }

    #[test]
    fn hp_lr4_2000_sr_44k1_wav_matches_analytical() {
        run_hp2000_acceptance(44_100.0);
    }

    #[test]
    fn hp_lr4_2000_sr_48k_wav_matches_analytical() {
        run_hp2000_acceptance(48_000.0);
    }

    #[test]
    fn hp_lr4_2000_sr_88k2_wav_matches_analytical() {
        run_hp2000_acceptance(88_200.0);
    }

    #[test]
    fn hp_lr4_2000_sr_176k4_wav_matches_analytical() {
        run_hp2000_acceptance(176_400.0);
    }

    /// b140.7.10 separates UI plot data (raw FFT, no centering) from WAV
    /// (centered). The plot must match the analytical filter phase to
    /// ≤ 5° in the audible passband for HP=2000 LR4 at sr=48k.
    #[test]
    fn hp_lr4_2000_sr_48k_ui_plot_phase_matches_model() {
        use crate::target::{evaluate, TargetCurve};

        let sr = 48_000.0_f64;
        let n_fft = 65_536_usize;
        let hp = lr4_filter(2000.0);
        let mut cfg = cfg_min(n_fft, sr);
        cfg.window = WindowType::Blackman;
        cfg.iterations = 0;

        let f_max = (40_000.0_f64).min(sr / 2.0 * 0.95);
        let log_freq = log_grid(512, 5.0, f_max);

        let out = generate_min_phase_fir_iir(&IirPathInput {
            freq: &log_freq, hp: Some(&hp), lp: None, peq: &[], config: &cfg,
        }).expect("IIR should succeed");

        let target = TargetCurve {
            reference_level_db: 0.0, tilt_db_per_octave: 0.0, tilt_ref_freq: 1000.0,
            high_pass: Some(hp.clone()), low_pass: None, low_shelf: None, high_shelf: None,
        };
        let ref_resp = evaluate(&target, &log_freq);

        let mut max_err = 0.0_f64;
        let mut probe_count = 0;
        for (i, &f) in log_freq.iter().enumerate() {
            // Skip deep stopband (phase ill-defined when |H| ≈ noise floor)
            // and Nyquist wraps. Audible passband 100 Hz .. 20 kHz.
            if f < 100.0 || f > 20_000.0 { continue; }
            let mut diff = out.realized_phase[i] - ref_resp.phase[i];
            while diff > 180.0 { diff -= 360.0; }
            while diff < -180.0 { diff += 360.0; }
            let err = diff.abs();
            if err > max_err { max_err = err; }
            probe_count += 1;
        }
        assert!(probe_count > 0);
        assert!(max_err < 5.0,
            "HP=2000 LR4 sr=48k UI plot phase: max err {:.2}° > 5° in 100 Hz–20 kHz band",
            max_err);
    }
}
