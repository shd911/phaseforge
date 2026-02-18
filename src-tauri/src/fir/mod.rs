// FIR correction engine: correction spectrum, minimum-phase via Hilbert, IFFT, windowing, WAV export

use num_complex::Complex64;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use tracing::info;

use crate::dsp::{fractional_octave_smooth, interpolate_linear_grid};
use crate::error::AppError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseMode {
    MinimumPhase,
    LinearPhase,
    MixedPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowType {
    Blackman,
    Kaiser,
    Tukey,
    Hann,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirConfig {
    pub taps: usize,          // 4096..262144
    pub sample_rate: f64,     // e.g. 48000
    pub max_boost_db: f64,    // e.g. 18.0
    pub noise_floor_db: f64,  // e.g. -60.0
    pub window: WindowType,
    pub phase_mode: PhaseMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirResult {
    pub impulse: Vec<f64>,
    pub time_ms: Vec<f64>,
    pub taps: usize,
    pub sample_rate: f64,
    pub norm_db: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirModelResult {
    pub impulse: Vec<f64>,
    pub time_ms: Vec<f64>,
    pub realized_mag: Vec<f64>,
    pub realized_phase: Vec<f64>,
    pub taps: usize,
    pub sample_rate: f64,
    pub norm_db: f64,
}

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

/// Generate FIR correction filter.
///
/// # Arguments
/// - `meas_freq` — measurement frequency axis (log-spaced)
/// - `meas_mag` — measurement magnitude in dB (potentially smoothed)
/// - `target_mag` — target magnitude in dB (evaluated at meas_freq)
/// - `peq_correction` — PEQ correction in dB (from `apply_peq`), or empty/zeros
/// - `config` — FIR configuration
/// - `crossover_range` — (f_low, f_high) from target HP/LP
pub fn generate_fir(
    meas_freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    peq_correction: &[f64],
    config: &FirConfig,
    crossover_range: (f64, f64),
) -> Result<FirResult, AppError> {
    let n = meas_freq.len();
    if n < 2 || meas_mag.len() != n || target_mag.len() != n {
        return Err(AppError::Config {
            message: "FIR: freq/mag/target length mismatch".into(),
        });
    }

    let n_fft = config.taps;
    let n_bins = n_fft / 2 + 1;

    // 1. Build effective target: within crossover use target, outside use smoothed measurement
    let effective_target = build_effective_target(
        meas_freq, meas_mag, target_mag, crossover_range,
    );

    // 2. Current magnitude = measurement + PEQ correction
    let current_mag: Vec<f64> = if peq_correction.len() == n {
        meas_mag.iter().zip(peq_correction.iter()).map(|(m, c)| m + c).collect()
    } else {
        meas_mag.to_vec()
    };

    // 3. Correction in dB on measurement grid
    let correction_log: Vec<f64> = current_mag.iter()
        .zip(effective_target.iter())
        .map(|(cur, tgt)| tgt - cur)
        .collect();

    // 4. Interpolate to linear grid (0..Nyquist, n_bins points)
    let (_lin_freq, lin_correction, _) = interpolate_linear_grid(
        meas_freq, &correction_log, None, n_bins, config.sample_rate,
    );

    // 5. Apply boost/cut limiting
    let limited: Vec<f64> = lin_correction.iter().map(|&v| {
        v.max(config.noise_floor_db).min(config.max_boost_db)
    }).collect();

    // 6. Compute minimum phase via Hilbert transform
    let phase_rad = match config.phase_mode {
        PhaseMode::MinimumPhase => minimum_phase_from_magnitude(&limited, n_fft),
        PhaseMode::LinearPhase => vec![0.0; n_bins], // zero phase → symmetric
        PhaseMode::MixedPhase => minimum_phase_from_magnitude(&limited, n_fft), // same as min for now
    };

    // 7. Assemble complex spectrum with conjugate symmetry
    let mut spectrum = assemble_complex_spectrum(&limited, &phase_rad, n_fft);

    // 8. IFFT
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut spectrum);

    let norm = 1.0 / n_fft as f64;
    let mut impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // 9. Phase-dependent reordering
    match config.phase_mode {
        PhaseMode::MinimumPhase => {
            // Minimum phase: impulse is causal, already correct
            // Just truncate to taps length (should already be n_fft)
        }
        PhaseMode::LinearPhase => {
            // Circular shift: move peak to center (N/2)
            circular_shift_to_center(&mut impulse);
        }
        PhaseMode::MixedPhase => {
            // Same as minimum phase for now
        }
    }

    // 10. Apply window
    // For minimum phase: use half-window (1.0 at start, taper to 0 at end)
    // For linear phase: use full symmetric window (centered peak)
    match config.phase_mode {
        PhaseMode::MinimumPhase | PhaseMode::MixedPhase => {
            let half_win = generate_half_window(n_fft, &config.window);
            for (i, w) in half_win.iter().enumerate() {
                impulse[i] *= w;
            }
        }
        PhaseMode::LinearPhase => {
            let window = generate_window(n_fft, &config.window);
            for (i, w) in window.iter().enumerate() {
                impulse[i] *= w;
            }
        }
    }

    // Passband normalization: FFT back to get realized spectrum, normalize peak to 0 dB
    let mut check: Vec<Complex64> = impulse.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    fft_fwd.process(&mut check);

    // Find peak magnitude across all positive frequencies (20 Hz .. Nyquist)
    let df = config.sample_rate / n_fft as f64;
    let mut max_db = f64::NEG_INFINITY;
    for k in 0..n_bins {
        let f = k as f64 * df;
        if f >= 20.0 {
            let amp = check[k].norm();
            let db = if amp > 1e-20 { 20.0 * amp.log10() } else { -200.0 };
            if db > max_db { max_db = db; }
        }
    }
    let norm_db = if max_db.is_finite() { max_db } else { 0.0 };

    let norm_linear = 10.0_f64.powf(-norm_db / 20.0);
    for s in impulse.iter_mut() {
        *s *= norm_linear;
    }

    info!("generate_fir: norm_db={:.2} → peak normalized to 0 dB", norm_db);

    // Build time axis in ms
    let dt_ms = 1000.0 / config.sample_rate;
    let time_ms: Vec<f64> = (0..n_fft).map(|i| i as f64 * dt_ms).collect();

    Ok(FirResult {
        impulse,
        time_ms,
        taps: n_fft,
        sample_rate: config.sample_rate,
        norm_db,
    })
}

/// Export impulse as WAV (32-bit float, mono).
///
/// Manual WAV writer — no external crate needed.
pub fn export_wav_f32(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
    use std::io::Write;

    let sr = sample_rate as u32;
    let num_samples = impulse.len() as u32;
    let bits_per_sample: u16 = 32;
    let num_channels: u16 = 1;
    let byte_rate = sr * (bits_per_sample as u32 / 8) * num_channels as u32;
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_samples * (bits_per_sample as u32 / 8);
    let riff_size = 36 + data_size; // 36 = 4 (WAVE) + 24 (fmt chunk) + 8 (data header)

    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk (format = 3 for IEEE float)
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&3u16.to_le_bytes());  // format: IEEE float
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sr.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in impulse {
        let f32_val = sample as f32;
        buf.extend_from_slice(&f32_val.to_le_bytes());
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;

    Ok(())
}

/// Export impulse response as a WAV file (64-bit IEEE float, mono).
pub fn export_wav_f64(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
    use std::io::Write;

    let sr = sample_rate as u32;
    let num_samples = impulse.len() as u32;
    let bits_per_sample: u16 = 64;
    let num_channels: u16 = 1;
    let byte_rate = sr * (bits_per_sample as u32 / 8) * num_channels as u32;
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_samples * (bits_per_sample as u32 / 8);
    let riff_size = 36 + data_size;

    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk (format = 3 for IEEE float, 64-bit)
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&3u16.to_le_bytes());  // format: IEEE float
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sr.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in impulse {
        buf.extend_from_slice(&sample.to_le_bytes()); // f64 directly
    }

    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Internal: build effective target
// ---------------------------------------------------------------------------

/// Build the effective target curve:
/// - Between HP and LP crossover frequencies: use target_mag as-is
/// - Below HP: 1/2-octave smoothed measurement (follow natural rolloff)
/// - Above LP: 1/2-octave smoothed measurement (follow natural rolloff)
/// - Smooth sigmoid blend ±0.5 octave around crossover frequencies
fn build_effective_target(
    freq: &[f64],
    meas_mag: &[f64],
    target_mag: &[f64],
    crossover_range: (f64, f64),
) -> Vec<f64> {
    let (f_low, f_high) = crossover_range;
    let n = freq.len();

    // Smooth measurement to 1/2 octave for out-of-band regions
    let smoothed_meas: Vec<f64> = (0..n)
        .map(|i| fractional_octave_smooth(freq, meas_mag, i, 0.5))
        .collect();

    // Blend width: ±0.5 octave in log-freq space
    let blend_octaves = 0.5;

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let f = freq[i];

        // Sigmoid blend factor: 0 = use smoothed measurement, 1 = use target
        let low_blend = sigmoid_blend(f, f_low, blend_octaves);
        let high_blend = 1.0 - sigmoid_blend(f, f_high, blend_octaves);

        // Combined: within crossover → 1.0, outside → 0.0, transitions → sigmoid
        let blend = low_blend * high_blend;
        let val = smoothed_meas[i] * (1.0 - blend) + target_mag[i] * blend;
        result.push(val);
    }

    result
}

/// Sigmoid blend: returns 0.0 well below `center_freq`, 1.0 well above.
/// Transition width is `octaves` (centered on `center_freq`).
fn sigmoid_blend(freq: f64, center_freq: f64, octaves: f64) -> f64 {
    if freq <= 0.0 || center_freq <= 0.0 {
        return 0.0;
    }
    let log_ratio = (freq / center_freq).log2(); // octaves from center
    let steepness = 6.0 / octaves; // ~6 gives a nice sigmoid over the octave width
    let x = log_ratio * steepness;
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Internal: minimum phase from magnitude via Hilbert transform
// ---------------------------------------------------------------------------

/// Compute minimum-phase from correction magnitude using Hilbert transform.
///
/// Algorithm:
/// 1. ln_mag[i] = ln(10^(correction_db[i]/20)) = correction_db[i] * ln(10) / 20
/// 2. Build full symmetric spectrum from ln_mag (like a real FFT input)
/// 3. FFT(ln_mag)
/// 4. Apply Hilbert window: ×2 for positive freqs, ×0 for negative freqs, ×1 for DC and Nyquist
/// 5. IFFT
/// 6. Extract imaginary part = minimum phase in radians
fn minimum_phase_from_magnitude(correction_db: &[f64], n_fft: usize) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let ln10_over_20 = 10.0_f64.ln() / 20.0;

    // Build ln_magnitude as a real signal of length n_fft
    // Positive frequencies: bins 0..n_bins
    // Negative frequencies: mirror (conjugate symmetry for real signal)
    let mut ln_mag_signal: Vec<Complex64> = Vec::with_capacity(n_fft);

    for i in 0..n_bins {
        let ln_val = correction_db[i.min(correction_db.len() - 1)] * ln10_over_20;
        ln_mag_signal.push(Complex64::new(ln_val, 0.0));
    }
    // Mirror for negative frequencies
    for i in 1..(n_fft - n_bins + 1) {
        let idx = n_bins - 1 - i;
        ln_mag_signal.push(Complex64::new(ln_mag_signal[idx].re, 0.0));
    }

    // FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut ln_mag_signal);

    // Apply Hilbert window: multiply positive freqs by 2, negative by 0
    // DC (bin 0) and Nyquist (bin n_fft/2) stay at 1
    ln_mag_signal[0] *= Complex64::new(1.0, 0.0); // DC: ×1
    for i in 1..n_fft / 2 {
        ln_mag_signal[i] *= Complex64::new(2.0, 0.0); // positive freqs: ×2
    }
    if n_fft > 1 {
        ln_mag_signal[n_fft / 2] *= Complex64::new(1.0, 0.0); // Nyquist: ×1
    }
    for i in (n_fft / 2 + 1)..n_fft {
        ln_mag_signal[i] = Complex64::new(0.0, 0.0); // negative freqs: ×0
    }

    // IFFT
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut ln_mag_signal);

    let norm = 1.0 / n_fft as f64;

    // Extract imaginary part = minimum phase (in radians)
    // We only need bins 0..n_bins (positive frequencies)
    (0..n_bins)
        .map(|i| -ln_mag_signal[i].im * norm) // negative sign: causal system convention
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: assemble complex spectrum
// ---------------------------------------------------------------------------

/// Build full N-point complex spectrum with conjugate symmetry.
fn assemble_complex_spectrum(
    correction_db: &[f64],
    phase_rad: &[f64],
    n_fft: usize,
) -> Vec<Complex64> {
    let n_bins = n_fft / 2 + 1;
    let mut spectrum = Vec::with_capacity(n_fft);

    // Positive frequencies (DC to Nyquist)
    for i in 0..n_bins {
        let amp = 10.0_f64.powf(correction_db[i.min(correction_db.len() - 1)] / 20.0);
        let ph = phase_rad[i.min(phase_rad.len() - 1)];
        spectrum.push(Complex64::new(amp * ph.cos(), amp * ph.sin()));
    }

    // Negative frequencies: conjugate mirror
    for i in 1..(n_fft - n_bins + 1) {
        let idx = n_bins - 1 - i;
        spectrum.push(spectrum[idx].conj());
    }

    spectrum
}

// ---------------------------------------------------------------------------
// Internal: windowing
// ---------------------------------------------------------------------------

/// Half-window for minimum phase FIR: 1.0 at start, smooth taper to 0 at end.
/// Uses the right half of a symmetric window (so the peak is at sample 0).
fn generate_half_window(n: usize, wtype: &WindowType) -> Vec<f64> {
    // Generate a window of length 2*n, then take the right half (indices n..2n)
    // This gives: starts at peak (1.0), decays to 0
    let full = generate_window(2 * n, wtype);
    full[n..].to_vec()
}

fn generate_window(n: usize, wtype: &WindowType) -> Vec<f64> {
    match wtype {
        WindowType::Blackman => blackman_window(n),
        WindowType::Kaiser => kaiser_window(n, 10.0),
        WindowType::Tukey => tukey_window(n, 0.5),
        WindowType::Hann => hann_window(n),
    }
}

fn blackman_window(n: usize) -> Vec<f64> {
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;
    (0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / (n - 1) as f64;
        a0 - a1 * x.cos() + a2 * (2.0 * x).cos()
    }).collect()
}

fn hann_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        let x = 2.0 * PI * i as f64 / (n - 1) as f64;
        0.5 * (1.0 - x.cos())
    }).collect()
}

fn tukey_window(n: usize, alpha: f64) -> Vec<f64> {
    let alpha = alpha.clamp(0.0, 1.0);
    (0..n).map(|i| {
        let x = i as f64 / (n - 1) as f64;
        if x < alpha / 2.0 {
            0.5 * (1.0 + ((2.0 * PI * x / alpha) - PI).cos())
        } else if x > 1.0 - alpha / 2.0 {
            0.5 * (1.0 + ((2.0 * PI * (1.0 - x) / alpha) - PI).cos())
        } else {
            1.0
        }
    }).collect()
}

fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let denom = bessel_i0(beta);
    (0..n).map(|i| {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let arg = beta * (1.0 - x * x).max(0.0).sqrt();
        bessel_i0(arg) / denom
    }).collect()
}

/// Modified Bessel function of the first kind, order 0 (I₀).
/// Computed via series expansion (converges fast for typical beta values).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..50 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-20 {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Internal: circular shift for linear phase
// ---------------------------------------------------------------------------

fn circular_shift_to_center(impulse: &mut Vec<f64>) {
    let n = impulse.len();
    let half = n / 2;
    impulse.rotate_right(half);
}

// ---------------------------------------------------------------------------
// Internal: linear interpolation helper (for mapping between grids)
// ---------------------------------------------------------------------------

/// Linear interpolation: map y_data defined on x_data onto x_query.
/// Out-of-range values are clamped to boundary.
fn interp_1d_simple(x_data: &[f64], y_data: &[f64], x_query: &[f64]) -> Vec<f64> {
    x_query.iter().map(|&xq| {
        if x_data.is_empty() { return 0.0; }
        if xq <= x_data[0] { return y_data[0]; }
        if xq >= x_data[x_data.len() - 1] { return y_data[y_data.len() - 1]; }
        let idx = match x_data.binary_search_by(|v| v.partial_cmp(&xq).unwrap()) {
            Ok(i) => return y_data[i],
            Err(i) => i,
        };
        let x0 = x_data[idx - 1];
        let x1 = x_data[idx];
        let y0 = y_data[idx - 1];
        let y1 = y_data[idx];
        let t = (xq - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }).collect()
}

// ---------------------------------------------------------------------------
// Public: generate FIR from pure model (no measurement)
// ---------------------------------------------------------------------------

/// Generate FIR from a pure mathematical filter model (target curve without measurement).
///
/// Phase handling:
/// - **Target (HP/LP/shelf/tilt)**: phase_mode from config determines linear or min-phase.
///   LinearPhase → zero phase contribution. MinimumPhase → Hilbert from target magnitude.
/// - **PEQ**: ALWAYS minimum-phase (Hilbert from PEQ magnitude). PEQ bands are
///   inherently min-phase biquads — their phase must be in the FIR.
/// - Total phase = target_phase + PEQ_phase.
///   Hilbert is linear, so Hilbert(A+B) = Hilbert(A) + Hilbert(B).
///   For MinimumPhase target: total = Hilbert(target+PEQ) — identical to before.
///   For LinearPhase target: total = 0 + Hilbert(PEQ) — PEQ min-phase preserved.
///
/// # Arguments
/// - `freq` — log-spaced frequency axis
/// - `target_mag` — target-only magnitude in dB (HP/LP/shelf/tilt, no PEQ)
/// - `peq_mag` — PEQ-only magnitude in dB (may be empty → treated as 0 dB)
/// - `model_phase` — combined model phase in degrees (for display/zero-detection)
/// - `config` — FIR configuration (taps, sample_rate, window, phase_mode)
pub fn generate_model_fir(
    freq: &[f64],
    target_mag: &[f64],
    peq_mag: &[f64],
    model_phase: &[f64],
    config: &FirConfig,
) -> Result<FirModelResult, AppError> {
    let n = freq.len();
    if n < 2 || target_mag.len() != n || model_phase.len() != n {
        return Err(AppError::Config {
            message: "generate_model_fir: freq/mag/phase length mismatch".into(),
        });
    }
    let has_peq = !peq_mag.is_empty();
    if has_peq && peq_mag.len() != n {
        return Err(AppError::Config {
            message: "generate_model_fir: peq_mag length mismatch".into(),
        });
    }

    let n_fft = config.taps;
    let n_bins = n_fft / 2 + 1;

    // 1. Interpolate target mag (dB) to linear FFT grid
    let (lin_freq, lin_target_raw, _) = interpolate_linear_grid(
        freq, target_mag, None, n_bins, config.sample_rate,
    );

    // Clip target magnitude to prevent Hilbert instability from extreme HP/LP rolloff
    let lin_target: Vec<f64> = lin_target_raw.iter().map(|&v| {
        v.max(config.noise_floor_db).min(config.max_boost_db)
    }).collect();

    // 2. Interpolate PEQ mag (dB) to linear FFT grid (if present)
    let lin_peq: Vec<f64> = if has_peq {
        let (_, peq_raw, _) = interpolate_linear_grid(
            freq, peq_mag, None, n_bins, config.sample_rate,
        );
        // PEQ magnitude is typically small — no extreme clipping needed
        peq_raw.iter().map(|&v| v.max(-60.0).min(config.max_boost_db)).collect()
    } else {
        vec![0.0; n_bins]
    };

    // 3. Total magnitude = target + PEQ (in dB)
    let lin_mag: Vec<f64> = lin_target.iter().zip(lin_peq.iter()).map(|(&t, &p)| t + p).collect();

    // 4. Determine effective phase mode
    let max_phase_abs = model_phase.iter().map(|p| p.abs()).fold(0.0_f64, f64::max);
    let is_zero_phase = max_phase_abs < 0.5;

    let effective_linear = match config.phase_mode {
        PhaseMode::LinearPhase => true,
        PhaseMode::MixedPhase => is_zero_phase,
        PhaseMode::MinimumPhase => false,
    };

    info!(
        "FIR: effective_linear={}, phase_mode={:?}, has_peq={}, max_phase_abs={:.2}°",
        effective_linear, config.phase_mode, has_peq, max_phase_abs
    );

    // 5. Phase for IFFT:
    //    Target phase: zero if linear, Hilbert if min-phase
    //    PEQ phase: ALWAYS Hilbert (min-phase biquads)
    //    Total phase = target_phase + PEQ_phase
    let target_phase_rad = if effective_linear {
        vec![0.0; n_bins]
    } else {
        minimum_phase_from_magnitude(&lin_target, n_fft)
    };

    let peq_phase_rad = if has_peq {
        minimum_phase_from_magnitude(&lin_peq, n_fft)
    } else {
        vec![0.0; n_bins]
    };

    let phase_rad: Vec<f64> = target_phase_rad.iter()
        .zip(peq_phase_rad.iter())
        .map(|(&t, &p)| t + p)
        .collect();

    // 6. Assemble complex spectrum and IFFT
    let mut spectrum = assemble_complex_spectrum(&lin_mag, &phase_rad, n_fft);

    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut spectrum);

    let norm = 1.0 / n_fft as f64;
    let mut impulse: Vec<f64> = spectrum.iter().map(|c| c.re * norm).collect();

    // 4. Phase-dependent reordering + windowing
    if effective_linear {
        // Symmetric impulse: shift peak to center, full window
        circular_shift_to_center(&mut impulse);
        let window = generate_window(n_fft, &config.window);
        for (i, w) in window.iter().enumerate() {
            impulse[i] *= w;
        }
    } else {
        // Causal impulse (min-phase or model-phase): half window
        let half_win = generate_half_window(n_fft, &config.window);
        for (i, w) in half_win.iter().enumerate() {
            impulse[i] *= w;
        }
    }

    // 6. FFT the windowed impulse back to get realized frequency response
    let mut realized_spectrum: Vec<Complex64> = impulse.iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut realized_spectrum);

    // Extract magnitude (dB) and excess phase (degrees) for positive frequencies.
    //
    // For linear-phase: subtract N/2 linear delay to get excess phase ≈ 0°.
    // For minimum-phase: raw phase is the min-phase response (no delay to subtract).
    let mut realized_mag_lin: Vec<f64> = Vec::with_capacity(n_bins);
    let mut realized_phase_lin: Vec<f64> = Vec::with_capacity(n_bins);

    let delay_samples = if effective_linear { (n_fft / 2) as f64 } else { 0.0 };

    for i in 0..n_bins {
        let c = realized_spectrum[i];
        let amp = c.norm();
        let mag_db = if amp > 1e-20 { 20.0 * amp.log10() } else { -400.0 };

        // Raw phase in radians
        let raw_phase_rad = c.arg();

        // Subtract linear delay: phase_delay = -2π·k·delay/N where k=bin index
        let linear_delay_rad = -2.0 * PI * i as f64 * delay_samples / n_fft as f64;
        let excess_phase_rad = raw_phase_rad - linear_delay_rad;

        // Wrap to [-π, π]
        let excess_wrapped = ((excess_phase_rad + PI) % (2.0 * PI) + 2.0 * PI) % (2.0 * PI) - PI;
        let phase_deg = excess_wrapped * 180.0 / PI;

        realized_mag_lin.push(mag_db);
        realized_phase_lin.push(phase_deg);
    }

    // Unwrap phase for smooth interpolation:
    // After removing linear delay, excess phase should be smooth.
    // Unwrap: if jump > 180° between adjacent bins, add/subtract 360°.
    for i in 1..realized_phase_lin.len() {
        let diff = realized_phase_lin[i] - realized_phase_lin[i - 1];
        if diff > 180.0 {
            realized_phase_lin[i] -= 360.0 * ((diff + 180.0) / 360.0).floor();
        } else if diff < -180.0 {
            realized_phase_lin[i] += 360.0 * ((-diff + 180.0) / 360.0).floor();
        }
    }

    // 7. Interpolate realized back to original log-frequency grid
    let realized_mag = interp_1d_simple(&lin_freq, &realized_mag_lin, freq);
    let realized_phase = interp_1d_simple(&lin_freq, &realized_phase_lin, freq);

    // 8. Passband normalization: shift so peak of realized magnitude = 0 dB
    //    This works for any filter shape (narrow band, wide band, HP, LP, BP).
    let norm_db = realized_mag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let norm_db = if norm_db.is_finite() { norm_db } else { 0.0 };

    info!("generate_model_fir: realized_max={:.2} dB → normalizing by {:.2} dB", norm_db, norm_db);

    // Scale impulse so passband peak = 0 dB
    let norm_linear = 10.0_f64.powf(-norm_db / 20.0);
    for s in impulse.iter_mut() {
        *s *= norm_linear;
    }

    // Shift realized curves to match normalized output
    let realized_mag: Vec<f64> = realized_mag.iter().map(|&v| v - norm_db).collect();

    info!("generate_model_fir: norm_db={:.2} → passband normalized to 0 dB", norm_db);

    // 9. Time axis
    let dt_ms = 1000.0 / config.sample_rate;
    let time_ms: Vec<f64> = (0..n_fft).map(|i| i as f64 * dt_ms).collect();

    Ok(FirModelResult {
        impulse,
        time_ms,
        realized_mag,
        realized_phase,
        taps: n_fft,
        sample_rate: config.sample_rate,
        norm_db,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_taps() {
        // For 20 Hz at 48000 Hz: 3 * 48000 / 20 = 7200 → next pow2 = 8192
        let taps = recommend_taps(20.0, 48000.0);
        assert_eq!(taps, 8192);

        // For 80 Hz at 48000 Hz: 3 * 48000 / 80 = 1800 → next pow2 = 2048 → clamp to 4096
        let taps = recommend_taps(80.0, 48000.0);
        assert_eq!(taps, 4096);
    }

    #[test]
    fn test_flat_correction_produces_dirac() {
        // 0 dB correction everywhere should produce a near-dirac impulse
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![80.0; n];
        let target: Vec<f64> = vec![80.0; n]; // same as measurement = 0 correction
        let peq: Vec<f64> = vec![0.0; n];

        let config = FirConfig {
            taps: 4096,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
        };

        let result = generate_fir(&freq, &mag, &target, &peq, &config, (20.0, 20000.0)).unwrap();
        assert_eq!(result.impulse.len(), 4096);
        assert_eq!(result.time_ms.len(), 4096);

        // Passband normalization should be near 0 dB (unity gain)
        assert!(result.norm_db.abs() < 3.0, "norm_db should be near 0 dB, got {}", result.norm_db);
    }

    #[test]
    fn test_boost_limiting() {
        // Large correction should be clamped
        let n = 100;
        let freq: Vec<f64> = (0..n).map(|i| 20.0 + i as f64 * 200.0).collect();
        let mag: Vec<f64> = vec![60.0; n];
        let target: Vec<f64> = vec![120.0; n]; // +60 dB correction!
        let peq: Vec<f64> = vec![0.0; n];

        let config = FirConfig {
            taps: 4096,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
        };

        let result = generate_fir(&freq, &mag, &target, &peq, &config, (20.0, 20000.0)).unwrap();
        // After passband normalization, norm_db reflects the passband level offset
        // With uniform +60dB correction clamped to +18dB, passband is 18dB → norm_db ≈ 18
        assert!(result.norm_db < 25.0, "norm_db should be limited, got {}", result.norm_db);
    }

    #[test]
    fn test_window_symmetry() {
        let n = 1024;
        let blackman = blackman_window(n);
        let hann = hann_window(n);
        let tukey = tukey_window(n, 0.5);

        for w in &[blackman, hann, tukey] {
            assert_eq!(w.len(), n);
            // Check symmetry
            for i in 0..n / 2 {
                let diff = (w[i] - w[n - 1 - i]).abs();
                assert!(diff < 1e-10, "Window not symmetric at i={}: {} vs {}", i, w[i], w[n - 1 - i]);
            }
            // End values should be small
            assert!(w[0] < 0.1, "Window start should be small, got {}", w[0]);
            // Center should be near 1.0
            assert!(w[n / 2] > 0.8, "Window center should be near 1.0, got {}", w[n / 2]);
        }
    }

    #[test]
    fn test_kaiser_window() {
        let n = 512;
        let w = kaiser_window(n, 10.0);
        assert_eq!(w.len(), n);
        // Symmetric
        for i in 0..n / 2 {
            assert!((w[i] - w[n - 1 - i]).abs() < 1e-10);
        }
        // Center should be 1.0
        assert!((w[n / 2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_blend() {
        // Well below center: should be ~0
        assert!(sigmoid_blend(10.0, 100.0, 0.5) < 0.01);
        // At center: should be ~0.5
        assert!((sigmoid_blend(100.0, 100.0, 0.5) - 0.5).abs() < 0.01);
        // Well above center: should be ~1
        assert!(sigmoid_blend(1000.0, 100.0, 0.5) > 0.99);
    }

    #[test]
    fn test_wav_export_f64() {
        let impulse = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let tmp = std::env::temp_dir().join("phaseforge_test_fir_f64.wav");
        export_wav_f64(&impulse, 48000.0, &tmp).unwrap();

        // Check file exists and has correct size (8 bytes per sample)
        let meta = std::fs::metadata(&tmp).unwrap();
        let expected_size = 44 + impulse.len() * 8; // header + data (64-bit)
        assert_eq!(meta.len(), expected_size as u64);

        // Read back and verify header
        let data = std::fs::read(&tmp).unwrap();
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");
        assert_eq!(&data[12..16], b"fmt ");
        // Format = 3 (IEEE float)
        assert_eq!(u16::from_le_bytes([data[20], data[21]]), 3);
        // Channels = 1
        assert_eq!(u16::from_le_bytes([data[22], data[23]]), 1);
        // Sample rate = 48000
        assert_eq!(u32::from_le_bytes([data[24], data[25], data[26], data[27]]), 48000);
        // Bits per sample = 64
        assert_eq!(u16::from_le_bytes([data[34], data[35]]), 64);

        // Verify first sample is 0.0 f64
        let sample0 = f64::from_le_bytes(data[44..52].try_into().unwrap());
        assert!((sample0 - 0.0).abs() < 1e-15);
        // Second sample is 0.5 f64
        let sample1 = f64::from_le_bytes(data[52..60].try_into().unwrap());
        assert!((sample1 - 0.5).abs() < 1e-15);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_minimum_phase_flat() {
        // Flat correction → phase should be ~0
        let correction = vec![0.0; 513]; // n_bins for n_fft=1024
        let phase = minimum_phase_from_magnitude(&correction, 1024);
        assert_eq!(phase.len(), 513);
        for &p in &phase {
            assert!(p.abs() < 0.01, "Phase should be ~0 for flat correction, got {}", p);
        }
    }

    #[test]
    fn test_bessel_i0() {
        // I₀(0) = 1
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        // I₀ is always ≥ 1 for x ≥ 0
        assert!(bessel_i0(5.0) > 1.0);
        assert!(bessel_i0(10.0) > bessel_i0(5.0));
    }

    #[test]
    fn test_generate_model_fir_flat() {
        // Flat 0 dB model → FIR realized should be near 0 dB everywhere
        let n = 256;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();
        let mag = vec![0.0; n];
        let phase = vec![0.0; n];

        let config = FirConfig {
            taps: 4096,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
        };

        let result = generate_model_fir(&freq, &mag, &[], &phase, &config).unwrap();
        assert_eq!(result.impulse.len(), 4096);
        assert_eq!(result.realized_mag.len(), n);
        assert_eq!(result.realized_phase.len(), n);

        // Realized magnitude should be near 0 dB in the midband (100-10000 Hz)
        for (i, &f) in freq.iter().enumerate() {
            if f >= 100.0 && f <= 10000.0 {
                assert!(
                    result.realized_mag[i].abs() < 3.0,
                    "Realized mag at {:.0} Hz should be near 0 dB, got {:.1}",
                    f, result.realized_mag[i]
                );
            }
        }
    }

    #[test]
    fn test_generate_model_fir_returns_valid() {
        // Low-pass model at -6dB/oct: verify structure
        let n = 128;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();
        let mag: Vec<f64> = freq.iter().map(|&f| {
            // Simple low-pass rolloff above 1kHz
            if f <= 1000.0 { 0.0 } else { -20.0 * (f / 1000.0).log10() }
        }).collect();
        let phase = vec![0.0; n];

        let config = FirConfig {
            taps: 8192,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Hann,
            phase_mode: PhaseMode::LinearPhase,
        };

        let result = generate_model_fir(&freq, &mag, &[], &phase, &config).unwrap();
        assert_eq!(result.taps, 8192);
        assert_eq!(result.sample_rate, 48000.0);
        assert_eq!(result.time_ms.len(), 8192);
        assert!(result.time_ms[0] < 0.001); // starts near 0
        assert!(result.time_ms[8191] > 170.0); // ~170ms for 8192 taps at 48k
    }

    #[test]
    fn test_interp_1d_simple() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let q = vec![0.5, 1.5, 2.5, 3.5, 5.0];
        let result = interp_1d_simple(&x, &y, &q);
        assert!((result[0] - 10.0).abs() < 0.01); // clamped
        assert!((result[1] - 15.0).abs() < 0.01);
        assert!((result[2] - 25.0).abs() < 0.01);
        assert!((result[3] - 35.0).abs() < 0.01);
        assert!((result[4] - 40.0).abs() < 0.01); // clamped
    }

    #[test]
    fn test_linear_phase_fir_symmetry() {
        // Linear phase FIR should produce impulse symmetric around N/2
        let n = 128;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();
        // Simple low-pass: flat to 1kHz, then -20dB/dec rolloff
        let mag: Vec<f64> = freq.iter().map(|&f| {
            if f <= 1000.0 { 0.0 } else { -20.0 * (f / 1000.0).log10() }
        }).collect();
        let phase = vec![0.0; n]; // zero phase model

        let config = FirConfig {
            taps: 8192,
            sample_rate: 48000.0,
            max_boost_db: 18.0,
            noise_floor_db: -60.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::LinearPhase,
        };

        let result = generate_model_fir(&freq, &mag, &[], &phase, &config).unwrap();
        let impulse = &result.impulse;
        let n_fft = impulse.len();
        let center = n_fft / 2;

        // Peak should be at or very near center
        let peak_idx = impulse.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i).unwrap();
        assert!(
            (peak_idx as i64 - center as i64).unsigned_abs() <= 1,
            "Peak at {} should be near center {}", peak_idx, center
        );

        // Check symmetry: impulse[center+k] ≈ impulse[center-k]
        let peak = impulse[center].abs();
        for k in 1..center.min(512) {
            let left = impulse[center - k];
            let right = impulse[center + k];
            let diff = (left - right).abs();
            let thresh = peak * 1e-6; // relative tolerance
            assert!(
                diff < thresh.max(1e-12),
                "Symmetry broken at k={}: left={}, right={}, diff={}",
                k, left, right, diff
            );
        }

        // Realized phase should be near zero (excess phase after delay removal)
        for (i, &f) in freq.iter().enumerate() {
            if f >= 100.0 && f <= 10000.0 {
                assert!(
                    result.realized_phase[i].abs() < 5.0,
                    "Excess phase at {:.0} Hz should be near 0°, got {:.1}°",
                    f, result.realized_phase[i]
                );
            }
        }
    }

    #[test]
    fn test_4band_all_filter_types() {
        use crate::target::{self, TargetCurve, FilterConfig, FilterType};

        // 1. Log frequency grid: 512 points, 20-20000 Hz
        let n = 512;
        let freq: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                (20.0_f64.ln() + t * (20000.0_f64.ln() - 20.0_f64.ln())).exp()
            })
            .collect();

        // 2. All filter type configurations: (label, FilterType, order, shape, linear_phase)
        let filter_types: Vec<(&str, FilterType, u8, Option<f64>, bool)> = vec![
            ("BW4", FilterType::Butterworth, 4, None, false),
            ("BS4", FilterType::Bessel, 4, None, false),
            ("LR4", FilterType::LinkwitzRiley, 4, None, false),
            ("GS2", FilterType::Gaussian, 4, Some(2.0), true),
        ];

        // 3. Band definitions: (name, hp_freq, lp_freq)
        let band_defs: Vec<(&str, Option<f64>, Option<f64>)> = vec![
            ("Sub", None, Some(80.0)),
            ("LowMid", Some(80.0), Some(500.0)),
            ("MidHigh", Some(500.0), Some(3500.0)),
            ("Tweeter", Some(3500.0), None),
        ];

        // Base FIR config (MinimumPhase for analog-type filters)
        let base_config = FirConfig {
            taps: 65536,
            sample_rate: 48000.0,
            max_boost_db: 24.0,
            noise_floor_db: -150.0,
            window: WindowType::Blackman,
            phase_mode: PhaseMode::MinimumPhase,
        };

        fn make_filter(
            ft: &FilterType, order: u8, freq_hz: f64,
            shape: Option<f64>, linear_phase: bool,
        ) -> FilterConfig {
            FilterConfig {
                filter_type: ft.clone(),
                order,
                freq_hz,
                shape,
                linear_phase,
            }
        }

        let tmp_dir = std::env::temp_dir().join("phaseforge_TEST");
        let _ = std::fs::create_dir_all(&tmp_dir);

        println!("\n=== 4-Band x 4-FilterType FIR Test ===\n");

        let mut total = 0u32;

        for (ft_name, ft, order, shape, linear) in &filter_types {
            // Gaussian with linear_phase → use LinearPhase FIR mode
            let fir_config = if *linear {
                FirConfig {
                    phase_mode: PhaseMode::LinearPhase,
                    ..base_config.clone()
                }
            } else {
                base_config.clone()
            };

            println!("--- {} (order={}, linear={}) ---", ft_name, order, linear);

            for (band_name, hp_freq, lp_freq) in &band_defs {
                let hp = hp_freq.map(|f| make_filter(ft, *order, f, *shape, *linear));
                let lp = lp_freq.map(|f| make_filter(ft, *order, f, *shape, *linear));

                let target_curve = TargetCurve {
                    reference_level_db: 0.0,
                    tilt_db_per_octave: 0.0,
                    tilt_ref_freq: 1000.0,
                    high_pass: hp,
                    low_pass: lp,
                    low_shelf: None,
                    high_shelf: None,
                };

                let response = target::evaluate(&target_curve, &freq);

                // Generate model FIR (no PEQ, no measurement correction)
                let result = generate_model_fir(
                    &freq,
                    &response.magnitude,
                    &[],
                    &response.phase,
                    &fir_config,
                )
                .unwrap_or_else(|e| panic!("{}/{}: generate_model_fir failed: {}", ft_name, band_name, e));

                // --- Assertions ---

                // Structure
                assert_eq!(result.taps, 65536, "{}/{}: wrong taps", ft_name, band_name);
                assert_eq!(result.impulse.len(), 65536, "{}/{}: wrong impulse len", ft_name, band_name);
                assert_eq!(result.realized_mag.len(), n, "{}/{}: wrong realized_mag len", ft_name, band_name);

                // norm_db finite and reasonable
                assert!(
                    result.norm_db.is_finite(),
                    "{}/{}: norm_db is not finite",
                    ft_name, band_name
                );
                assert!(
                    result.norm_db.abs() < 30.0,
                    "{}/{}: norm_db={:.1} too large",
                    ft_name, band_name, result.norm_db
                );

                // No NaN/Inf in impulse
                for (i, &v) in result.impulse.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "{}/{}: impulse[{}] is not finite ({})",
                        ft_name, band_name, i, v
                    );
                }

                // Passband realized_mag near 0 dB
                let pb_lo = hp_freq.unwrap_or(20.0) * 1.5;
                let pb_hi = lp_freq.unwrap_or(20000.0) / 1.5;
                if pb_lo < pb_hi {
                    for (i, &f) in freq.iter().enumerate() {
                        if f >= pb_lo && f <= pb_hi {
                            assert!(
                                result.realized_mag[i].abs() < 5.0,
                                "{}/{}: realized_mag at {:.0} Hz = {:.1} dB (expected ~0)",
                                ft_name, band_name, f, result.realized_mag[i]
                            );
                        }
                    }
                }

                // Export WAV
                let wav_path = tmp_dir.join(format!("TEST_{}_{}.wav", ft_name, band_name));
                export_wav_f64(&result.impulse, 48000.0, &wav_path).unwrap();
                assert!(wav_path.exists(), "{}/{}: WAV not created", ft_name, band_name);
                let meta = std::fs::metadata(&wav_path).unwrap();
                let expected_size = 44 + 65536 * 8; // header + 64-bit samples
                assert_eq!(
                    meta.len(),
                    expected_size as u64,
                    "{}/{}: WAV size mismatch",
                    ft_name, band_name
                );

                println!(
                    "  {} | norm_db={:+.2} dB | wav={:.0} KB",
                    band_name,
                    result.norm_db,
                    meta.len() as f64 / 1024.0
                );
                total += 1;
            }
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
        println!("\n=== All {} FIR filters generated and exported OK ===", total);
    }
}
