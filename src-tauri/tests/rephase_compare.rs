//! b140.7.12: REPhase reference comparison for HP min-phase IIR.
//!
//! Compares PhaseForge IIR-path output against REPhase reference WAVs
//! generated with: Min-Phase Filters mode, HP Linkwitz-Riley 48 dB/oct,
//! freq 2000 Hz, 65536 taps, centering=middle, hann window, Float64 mono.
//!
//! Reference files in `test-fixtures/rephase/{sr}.wav` (gitignored).
//! Tests skip cleanly when files are missing so CI / clean checkouts
//! still pass.

use std::path::PathBuf;

use phaseforge_lib::fir::iir_path::{generate_min_phase_fir_iir, IirPathInput};
use phaseforge_lib::fir::*;
use phaseforge_lib::target::{FilterConfig, FilterType};

use num_complex::Complex64;

// -----------------------------------------------------------------------------
// WAV parser — supports the formats REPhase / PhaseForge actually emit:
// IEEE float 64 / 32 mono, with possible LIST / JUNK chunks.
// -----------------------------------------------------------------------------

fn load_wav(path: &std::path::Path) -> Result<(u32, Vec<f64>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut fmt_code = 0u16;
    let mut channels = 0u16;
    let mut sr = 0u32;
    let mut bps = 0u16;
    let mut data_chunk: Option<&[u8]> = None;

    let mut i = 12;
    while i + 8 <= bytes.len() {
        let cid = &bytes[i..i + 4];
        let clen = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let body_start = i + 8;
        let body_end = body_start + clen;
        if body_end > bytes.len() {
            return Err(format!("chunk {:?} runs past EOF", String::from_utf8_lossy(cid)));
        }
        let body = &bytes[body_start..body_end];
        if cid == b"fmt " {
            fmt_code = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]);
            sr = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bps = u16::from_le_bytes([body[14], body[15]]);
        } else if cid == b"data" {
            data_chunk = Some(body);
        }
        i = body_end + (clen & 1); // pad to even
    }

    let data = data_chunk.ok_or("missing data chunk")?;
    if fmt_code != 3 {
        return Err(format!("unsupported fmt_code={} (need 3 = IEEE float)", fmt_code));
    }
    if channels != 1 {
        return Err(format!("unsupported channels={} (need 1)", channels));
    }
    let samples: Vec<f64> = match bps {
        64 => {
            let mut out = Vec::with_capacity(data.len() / 8);
            for chunk in data.chunks_exact(8) {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(chunk);
                out.push(f64::from_le_bytes(buf));
            }
            out
        }
        32 => {
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(chunk);
                out.push(f32::from_le_bytes(buf) as f64);
            }
            out
        }
        _ => return Err(format!("unsupported bps={}", bps)),
    };
    Ok((sr, samples))
}

// -----------------------------------------------------------------------------
// FFT helper — uses the same FftEngine as the production code.
// -----------------------------------------------------------------------------

fn fft_complex(samples: &[f64]) -> Vec<Complex64> {
    let mut spec: Vec<Complex64> = samples.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut engine = phaseforge_lib::dsp::fft::FftEngine::new();
    engine.fft_forward(&mut spec);
    spec
}

// -----------------------------------------------------------------------------
// Default-trait helpers (FirConfig has no Default impl).
// -----------------------------------------------------------------------------

fn lr_filter(order: u8, fc: f64) -> FilterConfig {
    FilterConfig {
        filter_type: FilterType::LinkwitzRiley,
        order, freq_hz: fc, shape: None,
        linear_phase: false, q: None, subsonic_protect: None,
    }
}

fn fir_cfg(taps: usize, sr: f64) -> FirConfig {
    FirConfig {
        taps, sample_rate: sr,
        max_boost_db: 6.0, noise_floor_db: -150.0,
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

// -----------------------------------------------------------------------------
// Comparison harness.
// -----------------------------------------------------------------------------

fn rephase_compare(sr: u32) {
    let path = PathBuf::from(format!(
        "{}/test-fixtures/rephase/{}.wav",
        env!("CARGO_MANIFEST_DIR").trim_end_matches("/src-tauri"),
        sr,
    ));
    if !path.exists() {
        // Try one level up (CARGO_MANIFEST_DIR includes /src-tauri).
        let alt = PathBuf::from(format!("../test-fixtures/rephase/{}.wav", sr));
        if !alt.exists() {
            eprintln!("[SKIP] sr={}: {} not found, skipping", sr, path.display());
            return;
        }
    }
    let resolved = if path.exists() {
        path
    } else {
        PathBuf::from(format!("../test-fixtures/rephase/{}.wav", sr))
    };

    let (rephase_sr, rephase_samples) = load_wav(&resolved)
        .unwrap_or_else(|e| panic!("load {}: {}", resolved.display(), e));
    assert_eq!(rephase_sr, sr, "REPhase WAV sr header mismatch");

    let n_fft = 65_536_usize;
    let hp = lr_filter(4, 2000.0); // PhaseForge LR4 = 48 dB/oct (REPhase LR8 equivalent)
    let cfg = fir_cfg(n_fft, sr as f64);
    let f_max = (40_000.0_f64).min(sr as f64 / 2.0 * 0.95);
    let lf = log_grid(512, 5.0, f_max);

    let pf_out = generate_min_phase_fir_iir(&IirPathInput {
        freq: &lf, hp: Some(&hp), lp: None, peq: &[], config: &cfg,
    }).expect("PhaseForge IIR generation");
    let pf_samples = &pf_out.impulse;

    assert_eq!(rephase_samples.len(), n_fft, "REPhase WAV length");
    assert_eq!(pf_samples.len(), n_fft, "PhaseForge impulse length");

    let r_spec = fft_complex(&rephase_samples);
    let p_spec = fft_complex(pf_samples);

    let key_freqs = [500.0_f64, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0];
    let mut max_mag = 0.0_f64;
    let mut max_phase = 0.0_f64;
    eprintln!("\n=== sr={} ===", sr);
    for &f in &key_freqs {
        if f >= sr as f64 / 2.0 - 100.0 { continue; }
        let bin = (f * n_fft as f64 / sr as f64).round() as usize;
        let r_mag = 20.0 * r_spec[bin].norm().max(1e-30).log10();
        let p_mag = 20.0 * p_spec[bin].norm().max(1e-30).log10();
        let r_phase = r_spec[bin].arg().to_degrees();
        let p_phase = p_spec[bin].arg().to_degrees();
        let mag_diff = (r_mag - p_mag).abs();
        let mut phase_diff = (r_phase - p_phase).abs();
        if phase_diff > 180.0 { phase_diff = 360.0 - phase_diff; }
        if mag_diff > max_mag { max_mag = mag_diff; }
        if phase_diff > max_phase { max_phase = phase_diff; }
        eprintln!(
            "f={:>6.0} Hz: REPhase mag={:+8.2} dB ph={:+7.1}° | PF mag={:+8.2} dB ph={:+7.1}° | Δmag={:.2} Δph={:.1}°",
            f, r_mag, r_phase, p_mag, p_phase, mag_diff, phase_diff,
        );
    }
    eprintln!("sr={} summary: max_mag_diff={:.2} dB, max_phase_diff={:.1}°", sr, max_mag, max_phase);

    assert!(max_mag < 1.0, "sr={}: max mag diff {:.2} dB > 1 dB", sr, max_mag);
    assert!(max_phase < 10.0, "sr={}: max phase diff {:.1}° > 10°", sr, max_phase);
}

#[test]
fn rephase_match_hp_lr8_2000_sr_44100() { rephase_compare(44_100); }

#[test]
fn rephase_match_hp_lr8_2000_sr_48000() { rephase_compare(48_000); }

#[test]
fn rephase_match_hp_lr8_2000_sr_88200() { rephase_compare(88_200); }

#[test]
fn rephase_match_hp_lr8_2000_sr_176400() { rephase_compare(176_400); }
