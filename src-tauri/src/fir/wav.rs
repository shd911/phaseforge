//! b140.13 — WAV export utilities for FIR impulses.
//!
//! Pure-I/O helper extracted from `fir/mod.rs` to slim the pipeline module and
//! isolate the WAV writer (zero DSP, zero shared state). Output format is
//! RIFF / WAVE / 64-bit IEEE-float mono, written manually so no external WAV
//! crate is pulled in. Re-exported as `crate::fir::export_wav_f64`.
//!
//! b141.124 (external audit 2026-10-29): the 32-bit writer is gone. The app
//! never called it, so its b141.118 cast check protected no exported file.
//!
//! b141.102 (audit 2026-10-23 2.8): the writer rejects an empty impulse,
//! a NaN/Inf sample and a sample rate outside 8000..=768000 Hz (or not a
//! whole number) instead of writing a file a convolver loads as silence or
//! noise, and the file is written atomically (unique tmp + rename) so a
//! failed export never leaves a truncated WAV in place of the previous one.

use crate::error::AppError;

/// Same range the FIR config accepts.
const SR_MIN: f64 = 8000.0;
const SR_MAX: f64 = 768_000.0;

const BITS_PER_SAMPLE: u16 = 64;
const BYTES_PER_SAMPLE: usize = (BITS_PER_SAMPLE / 8) as usize;

fn validate(impulse: &[f64], sample_rate: f64) -> Result<u32, AppError> {
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
    if impulse.len().saturating_mul(BYTES_PER_SAMPLE) > (u32::MAX as usize) - 64 {
        return Err(AppError::Config { message: "WAV export: impulse too long for a RIFF file".into() });
    }
    Ok(sample_rate as u32)
}

/// Export an impulse response as a WAV file: RIFF / WAVE / 64-bit IEEE float
/// (format 3) mono, header + data, written atomically.
pub fn export_wav_f64(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
    let sr = validate(impulse, sample_rate)?;
    let num_channels: u16 = 1;
    let byte_rate = sr * BYTES_PER_SAMPLE as u32 * num_channels as u32;
    let block_align = num_channels * BYTES_PER_SAMPLE as u16;
    let data_size = (impulse.len() * BYTES_PER_SAMPLE) as u32;
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
    buf.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &sample in impulse {
        buf.extend_from_slice(&sample.to_le_bytes());
    }

    crate::project::write_atomic(path, &buf)?;
    Ok(())
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
        }
        assert_eq!(std::fs::read(&path).unwrap(), good, "a rejected export touched the file");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn header_matches_the_data() {
        let path = tmp("header");
        export_wav_f64(&[0.25; 10], 96000.0, &path).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 36 + 80);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 96000);
        assert_eq!(u16::from_le_bytes(b[34..36].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 80);
        assert_eq!(f64::from_le_bytes(b[44..52].try_into().unwrap()), 0.25);
        std::fs::remove_file(&path).ok();
    }
}
