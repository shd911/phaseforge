//! b140.13 — WAV export utilities for FIR impulses.
//!
//! Two pure-I/O helpers extracted from `fir/mod.rs` to slim the
//! pipeline module and isolate the WAV writer (zero DSP, zero shared
//! state). Output format is RIFF / WAVE / IEEE-float mono, written
//! manually so no external WAV crate is pulled in.
//!
//! Re-exported as `crate::fir::{export_wav_f32, export_wav_f64}` —
//! all existing callers keep the same import path.
//!
//! b141.102 (audit 2026-10-23 2.8): both writers reject an empty impulse,
//! a NaN/Inf sample and a sample rate outside 8000..=768000 Hz (or not a
//! whole number) instead of writing a file a convolver loads as silence or
//! noise, and the file is written atomically (unique tmp + rename) so a
//! failed export never leaves a truncated WAV in place of the previous one.

use crate::error::AppError;

/// Same range the FIR config accepts.
const SR_MIN: f64 = 8000.0;
const SR_MAX: f64 = 768_000.0;

fn validate(impulse: &[f64], sample_rate: f64, bytes_per_sample: usize) -> Result<u32, AppError> {
    if impulse.is_empty() {
        return Err(AppError::Config { message: "WAV export: the impulse is empty".into() });
    }
    if let Some(i) = impulse.iter().position(|v| !v.is_finite()) {
        return Err(AppError::Dsp {
            message: format!("WAV export: sample {i} of {} is not finite ({})", impulse.len(), impulse[i]),
        });
    }
    if !(sample_rate.is_finite() && (SR_MIN..=SR_MAX).contains(&sample_rate) && sample_rate.fract() == 0.0) {
        return Err(AppError::Config {
            message: format!("WAV export: sample rate {sample_rate} Hz is outside {SR_MIN}..={SR_MAX} or not whole"),
        });
    }
    if impulse.len().saturating_mul(bytes_per_sample) > (u32::MAX as usize) - 64 {
        return Err(AppError::Config { message: "WAV export: impulse too long for a RIFF file".into() });
    }
    Ok(sample_rate as u32)
}

/// RIFF / WAVE / IEEE-float (format 3) mono, header + data, written atomically.
fn write_float_wav(
    impulse: &[f64],
    sample_rate: f64,
    path: &std::path::Path,
    bits_per_sample: u16,
    encode: impl Fn(f64, &mut Vec<u8>),
) -> Result<(), AppError> {
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let sr = validate(impulse, sample_rate, bytes_per_sample)?;
    let num_channels: u16 = 1;
    let byte_rate = sr * bytes_per_sample as u32 * num_channels as u32;
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = (impulse.len() * bytes_per_sample) as u32;
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
        encode(sample, &mut buf);
    }

    crate::project::write_atomic(path, &buf)?;
    Ok(())
}

/// Export an impulse response as a WAV file (32-bit IEEE float, mono).
///
/// Manual WAV writer — no external crate needed.
pub fn export_wav_f32(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
    write_float_wav(impulse, sample_rate, path, 32, |s, buf| {
        buf.extend_from_slice(&(s as f32).to_le_bytes())
    })
}

/// Export an impulse response as a WAV file (64-bit IEEE float, mono).
pub fn export_wav_f64(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
    write_float_wav(impulse, sample_rate, path, 64, |s, buf| buf.extend_from_slice(&s.to_le_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("pf_wav_{}_{name}.wav", std::process::id()))
    }

    #[test]
    fn rejects_bad_input_and_keeps_the_previous_file() {
        let path = tmp("reject");
        export_wav_f64(&[1.0, 0.5], 48000.0, &path).unwrap();
        let good = std::fs::read(&path).unwrap();
        assert_eq!(good.len(), 44 + 16);
        for (imp, sr) in [
            (vec![], 48000.0),
            (vec![1.0, f64::NAN], 48000.0),
            (vec![1.0, f64::INFINITY], 48000.0),
            (vec![1.0], 0.0),
            (vec![1.0], f64::NAN),
            (vec![1.0], 44100.5),
            (vec![1.0], 1_000_000.0),
        ] {
            assert!(export_wav_f64(&imp, sr, &path).is_err(), "f64 accepted {imp:?} @ {sr}");
            assert!(export_wav_f32(&imp, sr, &path).is_err(), "f32 accepted {imp:?} @ {sr}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), good, "a rejected export touched the file");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn header_matches_the_data() {
        let path = tmp("header");
        export_wav_f32(&[0.25; 10], 96000.0, &path).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 36 + 40);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 96000);
        assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 40);
        assert_eq!(f32::from_le_bytes(b[44..48].try_into().unwrap()), 0.25);
        std::fs::remove_file(&path).ok();
    }
}
