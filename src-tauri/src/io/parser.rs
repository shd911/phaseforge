use std::path::Path;

use crate::error::AppError;
use crate::phase::unwrap_phase;

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

    // b139.5.1: normalise CR-only and CRLF endings to LF so str::lines()
    // splits the file regardless of source platform (classic Mac REW
    // exports use CR-only and were silently treated as one giant line).
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");

    for line in normalized.lines() {
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

        // b141.5 (audit): reject nan/inf — f64::parse accepts them and they
        // poison the DSP pipeline (10^(inf/20) → inf → NaN impulse).
        if !f.is_finite() || !m.is_finite() || p.is_some_and(|v| !v.is_finite()) {
            return Err(AppError::Parse {
                message: format!("Non-finite value in data line: '{trimmed}'"),
            });
        }

        if first_data {
            has_phase = p.is_some();
            first_data = false;
        } else if p.is_some() != has_phase {
            // b141.5 (audit): a truncated 3-column file (or a stray extra
            // column) used to yield phase.len() != freq.len(), accepted
            // silently → index panic later in interp. Reject up front.
            return Err(AppError::Parse {
                message: format!(
                    "Inconsistent column count: line '{trimmed}' {} a phase column while previous lines {}",
                    if p.is_some() { "has" } else { "is missing" },
                    if has_phase { "have one" } else { "do not" },
                ),
            });
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
    // b141.5 (audit): a single point breaks group-delay rendering and any
    // interpolation downstream — require at least two.
    if freq.len() < 2 {
        return Err(AppError::Parse {
            message: "File contains only one data point; at least two are required".to_string(),
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

// unwrap_phase removed — using crate::phase::unwrap_phase instead

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
    fn parser_handles_cr_only_line_endings() {
        // Synthetic REW measurement file with CR-only line endings (Mac classic).
        let content = "* Measurement data\r\
                       * Freq(Hz) SPL(dB) Phase(degrees)\r\
                       20.0 47.5 78.2\r\
                       100.0 50.0 -10.5\r\
                       1000.0 48.3 0.0\r";
        let result = parse_rew_txt(content);
        assert!(result.is_ok(), "CR-only parser failed: {:?}", result.err());
        let m = result.unwrap();
        assert_eq!(m.freq.len(), 3);
        assert!((m.freq[0] - 20.0).abs() < 1e-6);
        assert!((m.magnitude[0] - 47.5).abs() < 1e-6);
        assert!((m.freq[2] - 1000.0).abs() < 1e-6);
        assert!((m.magnitude[2] - 48.3).abs() < 1e-6);
    }

    #[test]
    fn parser_handles_lf_line_endings() {
        let content = "* Measurement data\n\
                       * Freq(Hz) SPL(dB) Phase(degrees)\n\
                       20.0 47.5 78.2\n\
                       100.0 50.0 -10.5\n";
        let result = parse_rew_txt(content);
        assert!(result.is_ok(), "LF parser failed: {:?}", result.err());
        let m = result.unwrap();
        assert_eq!(m.freq.len(), 2);
        assert!((m.magnitude[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn parser_handles_crlf_line_endings() {
        let content = "* Measurement data\r\n\
                       * Freq(Hz) SPL(dB) Phase(degrees)\r\n\
                       20.0 47.5 78.2\r\n\
                       100.0 50.0 -10.5\r\n";
        let result = parse_rew_txt(content);
        assert!(result.is_ok(), "CRLF parser failed: {:?}", result.err());
        let m = result.unwrap();
        assert_eq!(m.freq.len(), 2);
        assert!((m.freq[1] - 100.0).abs() < 1e-6);
    }

    // b141.5 (audit HIGH): a truncated/corrupt 3-column file whose tail rows
    // lose the phase column previously produced phase.len() < freq.len() —
    // accepted silently, then index-out-of-bounds panic deep in interp_at /
    // interp_single when the frontend fed freq+phase into DSP commands.
    #[test]
    fn test_inconsistent_phase_column_rejected() {
        let data = "\
20.0 65.3 -45.2
40.0 70.1 -30.0
100.0 75.5
";
        let result = parse_rew_txt(data);
        assert!(result.is_err(), "rows with missing phase column must be rejected");
    }

    #[test]
    fn test_phase_column_appearing_later_rejected() {
        let data = "\
20.0 65.3
40.0 70.1 -30.0
";
        let result = parse_rew_txt(data);
        assert!(result.is_err(), "rows gaining a phase column must be rejected");
    }

    // b141.5 (audit MEDIUM): f64::parse accepts nan/inf literals which then
    // poison the whole DSP pipeline (10^(inf/20) → inf → NaN impulse).
    #[test]
    fn test_non_finite_values_rejected() {
        for bad in ["20.0 nan\n40.0 70.0\n", "20.0 65.0\n40.0 inf\n",
                    "nan 65.0\n40.0 70.0\n", "20.0 65.0 -inf\n40.0 70.0 0.0\n"] {
            let result = parse_rew_txt(bad);
            assert!(result.is_err(), "non-finite value must be rejected: {bad:?}");
        }
    }

    // b141.5 (audit LOW): a single data point breaks GD rendering
    // ((ph[1]-ph[0])/(f[1]-f[0]) → undefined → NaN curve) and any interp.
    #[test]
    fn test_single_point_rejected() {
        let result = parse_rew_txt("1000.0 65.0\n");
        assert!(result.is_err(), "files with a single data point must be rejected");
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
