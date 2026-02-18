use std::path::Path;

use crate::error::AppError;

use super::{Measurement, MeasurementMetadata};

/// Import a measurement file, auto-detecting format by extension.
pub fn import_measurement(path: &Path) -> Result<Measurement, AppError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let content = std::fs::read_to_string(path).map_err(AppError::Io)?;

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string();

    let mut measurement = match ext.as_str() {
        "txt" => parse_rew_txt(&content)?,
        "frd" => parse_frd(&content)?,
        _ => {
            return Err(AppError::Parse {
                message: format!("Unsupported file format: .{ext}"),
            })
        }
    };

    measurement.name = name;
    measurement.source_path = Some(path.to_path_buf());

    Ok(measurement)
}

/// Parse REW .txt frequency response export.
///
/// Format:
/// ```text
/// * Freq(Hz)  SPL(dB)  Phase(degrees)
/// 20.000      65.3     -45.2
/// ```
///
/// Lines starting with `*` are comments/headers. Supports 2 columns (freq, mag)
/// or 3 columns (freq, mag, phase).
pub fn parse_rew_txt(content: &str) -> Result<Measurement, AppError> {
    let mut freq = Vec::new();
    let mut magnitude = Vec::new();
    let mut phase_raw = Vec::new();
    let mut has_phase = false;
    let mut first_data = true;

    for line in content.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comment lines
        if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();

        if parts.len() < 2 {
            continue;
        }

        let f: f64 = parts[0].parse().map_err(|_| AppError::Parse {
            message: format!("Invalid frequency value: '{}'", parts[0]),
        })?;

        let m: f64 = parts[1].parse().map_err(|_| AppError::Parse {
            message: format!("Invalid magnitude value: '{}'", parts[1]),
        })?;

        let p: Option<f64> = if parts.len() >= 3 {
            Some(parts[2].parse().map_err(|_| AppError::Parse {
                message: format!("Invalid phase value: '{}'", parts[2]),
            })?)
        } else {
            None
        };

        if first_data {
            has_phase = p.is_some();
            first_data = false;
        }

        freq.push(f);
        magnitude.push(m);
        if let Some(p_val) = p {
            phase_raw.push(p_val);
        }
    }

    if freq.is_empty() {
        return Err(AppError::Parse {
            message: "No data points found in file".to_string(),
        });
    }

    // Validate monotonically increasing frequencies
    validate_frequency_order(&freq)?;

    // Always unwrap phase — store continuous phase for DSP,
    // frontend wraps to ±180° for display when needed
    let phase = if has_phase && !phase_raw.is_empty() {
        Some(unwrap_phase(&phase_raw))
    } else {
        None
    };

    Ok(Measurement {
        name: String::new(),
        source_path: None,
        sample_rate: None,
        freq,
        magnitude,
        phase,
        metadata: MeasurementMetadata::default(),
    })
}

/// Parse .frd (Frequency Response Data) files.
///
/// Same columnar format as REW .txt but without header comments (sometimes).
/// Supports 2 columns (freq, mag) or 3 columns (freq, mag, phase).
pub fn parse_frd(content: &str) -> Result<Measurement, AppError> {
    // .frd has the same data format, possibly without comment headers.
    // We reuse the REW parser since it already handles both cases.
    parse_rew_txt(content)
}

/// Validate that frequencies are monotonically increasing.
fn validate_frequency_order(freq: &[f64]) -> Result<(), AppError> {
    for i in 1..freq.len() {
        if freq[i] <= freq[i - 1] {
            return Err(AppError::Parse {
                message: format!(
                    "Frequencies must be monotonically increasing, but freq[{}]={} <= freq[{}]={}",
                    i,
                    freq[i],
                    i - 1,
                    freq[i - 1]
                ),
            });
        }
    }
    Ok(())
}

/// Unwrap phase from ±180° wrapped to continuous.
fn unwrap_phase(wrapped: &[f64]) -> Vec<f64> {
    if wrapped.is_empty() {
        return vec![];
    }

    let mut unwrapped = vec![wrapped[0]];

    for i in 1..wrapped.len() {
        let mut diff = wrapped[i] - wrapped[i - 1];
        // Normalize to ±180°
        while diff > 180.0 {
            diff -= 360.0;
        }
        while diff < -180.0 {
            diff += 360.0;
        }
        unwrapped.push(unwrapped[i - 1] + diff);
    }

    unwrapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rew_txt_3col() {
        let data = "\
* Freq(Hz)\tSPL(dB)\tPhase(degrees)
20.000\t65.3\t-45.2
40.000\t70.1\t-30.0
100.000\t75.5\t10.5
";
        let m = parse_rew_txt(data).unwrap();
        assert_eq!(m.freq.len(), 3);
        assert_eq!(m.magnitude.len(), 3);
        assert!(m.phase.is_some());
        assert_eq!(m.phase.as_ref().unwrap().len(), 3);
        assert!((m.freq[0] - 20.0).abs() < 1e-6);
        assert!((m.magnitude[1] - 70.1).abs() < 1e-6);
    }

    #[test]
    fn test_parse_rew_txt_2col() {
        let data = "\
* Freq(Hz)\tSPL(dB)
20.000\t65.3
40.000\t70.1
";
        let m = parse_rew_txt(data).unwrap();
        assert_eq!(m.freq.len(), 2);
        assert!(m.phase.is_none());
    }

    #[test]
    fn test_parse_empty_file() {
        let data = "* comment only\n";
        let result = parse_rew_txt(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_non_monotonic() {
        let data = "100.0 65.0\n50.0 70.0\n";
        let result = parse_rew_txt(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_unwrap() {
        // Wrapped: 170, -170 (jump of 340°, should unwrap to 190)
        let wrapped = vec![170.0, -170.0, -10.0];
        let unwrapped = unwrap_phase(&wrapped);
        assert!((unwrapped[0] - 170.0).abs() < 1e-6);
        assert!((unwrapped[1] - 190.0).abs() < 1e-6);
        assert!((unwrapped[2] - 350.0).abs() < 1e-6);
    }
}
