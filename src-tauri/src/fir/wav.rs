//! b140.13 — WAV export utilities for FIR impulses.
//!
//! Two pure-I/O helpers extracted from `fir/mod.rs` to slim the
//! pipeline module and isolate the WAV writer (zero DSP, zero shared
//! state). Output format is RIFF / WAVE / IEEE-float mono, written
//! manually so no external WAV crate is pulled in.
//!
//! Re-exported as `crate::fir::{export_wav_f32, export_wav_f64}` —
//! all existing callers keep the same import path.

use std::io::Write;

use crate::error::AppError;

/// Export an impulse response as a WAV file (32-bit IEEE float, mono).
///
/// Manual WAV writer — no external crate needed.
pub fn export_wav_f32(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
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

/// Export an impulse response as a WAV file (64-bit IEEE float, mono).
pub fn export_wav_f64(impulse: &[f64], sample_rate: f64, path: &std::path::Path) -> Result<(), AppError> {
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
