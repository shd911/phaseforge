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

    let bytes = std::fs::read(path).map_err(AppError::Io)?;
    let content = decode_text(&bytes);

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string();

    let mut measurement = match ext.as_str() {
        "txt" | "csv" => parse_rew_txt(&content)?,
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

/// Decode a measurement file's bytes. b141.102 (audit 2026-10-23 2.5):
/// `read_to_string` rejected every file that was not clean UTF-8 — a UTF-8
/// BOM survived as a junk first character, UTF-16 (Windows tools, Excel
/// «Unicode text») and Latin-1 headers (`°`, `µ`) failed outright.
/// UTF-8 (BOM stripped) → UTF-16 LE/BE (BOM, or NUL-byte pattern) → Latin-1.
pub(crate) fn decode_text(bytes: &[u8]) -> String {
    fn utf16(body: &[u8], le: bool) -> String {
        let units: Vec<u16> = body
            .chunks_exact(2)
            .map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) })
            .collect();
        String::from_utf16_lossy(&units)
    }
    if let Some(body) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(body).into_owned();
    }
    if let Some(body) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return utf16(body, true);
    }
    if let Some(body) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return utf16(body, false);
    }
    // BOM-less UTF-16: ASCII text has a NUL in every other byte.
    let head = &bytes[..bytes.len().min(512) & !1];
    if head.len() >= 8 {
        let half = head.len() / 2;
        let odd_nul = head.iter().skip(1).step_by(2).filter(|&&b| b == 0).count();
        let even_nul = head.iter().step_by(2).filter(|&&b| b == 0).count();
        if odd_nul * 10 >= half * 9 && even_nul * 10 < half {
            return utf16(bytes, true);
        }
        if even_nul * 10 >= half * 9 && odd_nul * 10 < half {
            return utf16(bytes, false);
        }
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        // Latin-1: every byte is the code point of the same value.
        Err(_) => bytes.iter().map(|&b| b as char).collect(),
    }
}

/// Parse one number; `decimal_comma` turns a single `,` into the point.
fn parse_number(field: &str, decimal_comma: bool) -> Option<f64> {
    let f = field.trim().trim_matches('"');
    if f.is_empty() {
        return None;
    }
    if decimal_comma && f.matches(',').count() == 1 && !f.contains('.') {
        f.replace(',', ".").parse().ok()
    } else {
        f.parse().ok()
    }
}

/// Split a data line into numeric fields. b141.102 (audit 2026-10-23 2.5):
/// whitespace / tab, comma (CSV) and semicolon separators; a decimal comma is
/// accepted where it cannot be a separator — with `;` or whitespace between
/// the fields («20,5;65,3» or «20,5 65,3»). A line of commas only that reads
/// as CSV («20.5,65.3» or «20,65,-45») is CSV. Trailing non-numeric columns
/// are ignored, as before. `None` = the first field is not a number (text).
fn split_fields(line: &str) -> Option<Vec<f64>> {
    // The leading numeric fields.
    let numeric = |parts: Vec<&str>, dc: bool| -> Vec<f64> {
        parts.into_iter()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map_while(|p| parse_number(p, dc))
            .collect()
    };
    let v = if line.contains(';') {
        numeric(line.split(';').collect(), true)
    } else if line.contains(',') {
        let csv = numeric(line.split(',').collect(), false);
        if csv.len() >= 2 { csv } else { numeric(line.split_whitespace().collect(), true) }
    } else {
        numeric(line.split_whitespace().collect(), false)
    };
    if v.is_empty() { None } else { Some(v) }
}

/// Comment / header lines: REW `*`, `#`, `;` (ARTA, LspCAD), `//`, `%`.
fn is_comment(line: &str) -> bool {
    line.starts_with('*') || line.starts_with('#') || line.starts_with(';')
        || line.starts_with("//") || line.starts_with('%')
}

/// Levels beyond this are not a measurement (10^(1e300/20) silences the
/// impulse; |phase| this large hangs the unwrap loop).
const MAX_ABS_LEVEL_DB: f64 = 500.0;
const MAX_ABS_PHASE_DEG: f64 = 1.0e7;

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
///
/// b141.102 (audit 2026-10-23 2.5, 2.9): also `#`, `;`, `//`, `%` comments,
/// text header lines before the data without a comment mark (VituixCAD,
/// Excel «Freq;SPL;Phase»), a leading BOM, CSV / semicolon / tab separators
/// and a decimal comma where unambiguous (see `split_fields`). A 0 Hz (DC)
/// row is skipped; a negative frequency, |level| > 500 dB or |phase| > 1e7°
/// rejects the file.
pub fn parse_rew_txt(content: &str) -> Result<Measurement, AppError> {
    let mut freq = Vec::new();
    let mut magnitude = Vec::new();
    let mut phase_raw = Vec::new();
    let mut has_phase = false;
    let mut first_data = true;

    // b139.5.1: normalise CR-only and CRLF endings to LF so str::lines()
    // splits the file regardless of source platform (classic Mac REW
    // exports use CR-only and were silently treated as one giant line).
    let normalized = content
        .trim_start_matches('\u{FEFF}')
        .replace("\r\n", "\n")
        .replace('\r', "\n");

    for line in normalized.lines() {
        let trimmed = line.trim().trim_start_matches('\u{FEFF}');

        // Skip empty lines and comment lines
        if trimmed.is_empty() || is_comment(trimmed) {
            continue;
        }

        let Some(parts) = split_fields(trimmed) else {
            // A text line. Before the data it is a header; inside the data it
            // is corruption and must not be skipped silently.
            if first_data {
                continue;
            }
            return Err(AppError::Parse {
                message: format!("Invalid data line: '{trimmed}'"),
            });
        };

        if parts.len() < 2 {
            continue;
        }

        let f = parts[0];
        let m = parts[1];
        let p: Option<f64> = parts.get(2).copied();

        // b141.5 (audit): reject nan/inf — f64::parse accepts them and they
        // poison the DSP pipeline (10^(inf/20) → inf → NaN impulse).
        if !f.is_finite() || !m.is_finite() || p.is_some_and(|v| !v.is_finite()) {
            return Err(AppError::Parse {
                message: format!("Non-finite value in data line: '{trimmed}'"),
            });
        }
        if f < 0.0 {
            return Err(AppError::Parse {
                message: format!("Negative frequency in data line: '{trimmed}'"),
            });
        }
        if m.abs() > MAX_ABS_LEVEL_DB || p.is_some_and(|v| v.abs() > MAX_ABS_PHASE_DEG) {
            return Err(AppError::Parse {
                message: format!("Level or phase out of range in data line: '{trimmed}'"),
            });
        }
        if f == 0.0 {
            // DC bin of an FFT export: no place on a log-frequency axis.
            continue;
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

    fn assert_three_rows(m: &Measurement, label: &str) {
        assert_eq!(m.freq, vec![20.5, 40.0, 100.0], "{label}: freq");
        assert_eq!(m.magnitude, vec![65.3, 70.1, 75.5], "{label}: mag");
        let ph = m.phase.as_ref().unwrap_or_else(|| panic!("{label}: phase lost"));
        assert!((ph[0] + 45.2).abs() < 1e-9 && (ph[2] - 10.5).abs() < 1e-9, "{label}: phase {ph:?}");
    }

    /// Audit 2026-10-23 2.5: separators, decimal comma, header and comment
    /// styles of ARTA / VituixCAD / Excel exports.
    #[test]
    fn foreign_text_layouts_parse() {
        let cases = [
            ("tab", "Freq(Hz)\tSPL(dB)\tPhase(deg)\n20.5\t65.3\t-45.2\n40\t70.1\t-30\n100\t75.5\t10.5\n"),
            ("csv", "Frequency,Magnitude,Phase\n20.5,65.3,-45.2\n40,70.1,-30\n100,75.5,10.5\n"),
            ("csv spaced", "20.5, 65.3, -45.2\n40, 70.1, -30\n100, 75.5, 10.5\n"),
            ("csv quoted", "\"Hz\",\"dB\",\"deg\"\n\"20.5\",\"65.3\",\"-45.2\"\n\"40\",\"70.1\",\"-30\"\n\"100\",\"75.5\",\"10.5\"\n"),
            ("semicolon + decimal comma", "Freq;SPL;Phase\n20,5;65,3;-45,2\n40;70,1;-30\n100;75,5;10,5\n"),
            ("semicolon + point", "20.5;65.3;-45.2\n40;70.1;-30\n100;75.5;10.5\n"),
            ("space + decimal comma", "20,5 65,3 -45,2\n40 70,1 -30\n100 75,5 10,5\n"),
            ("tab + decimal comma", "20,5\t65,3\t-45,2\n40\t70,1\t-30\n100\t75,5\t10,5\n"),
            ("; comments (ARTA)", "; ARTA frd\n; Freq  dB  deg\n20.5 65.3 -45.2\n40 70.1 -30\n100 75.5 10.5\n"),
            ("// and % comments", "// LspCAD\n% export\n20.5 65.3 -45.2\n40 70.1 -30\n100 75.5 10.5\n"),
            ("BOM", "\u{FEFF}* REW\n20.5 65.3 -45.2\n40 70.1 -30\n100 75.5 10.5\n"),
            ("DC row", "0 65.3 0\n20.5 65.3 -45.2\n40 70.1 -30\n100 75.5 10.5\n"),
            ("trailing text column", "20.5 65.3 -45.2 ok\n40 70.1 -30 ok\n100 75.5 10.5 ok\n"),
        ];
        for (label, text) in cases {
            let m = parse_rew_txt(text).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_three_rows(&m, label);
        }
    }

    #[test]
    fn text_encodings_decode() {
        let text = "* Freq(Hz) SPL(dB) Phase(°)\n20.5 65.3 -45.2\n40 70.1 -30\n100 75.5 10.5\n";
        let utf16 = |le: bool, bom: bool| -> Vec<u8> {
            let mut out = Vec::new();
            if bom { out.extend(if le { [0xFF, 0xFE] } else { [0xFE, 0xFF] }); }
            for u in text.encode_utf16() {
                out.extend(if le { u.to_le_bytes() } else { u.to_be_bytes() });
            }
            out
        };
        let mut utf8_bom = vec![0xEF, 0xBB, 0xBF];
        utf8_bom.extend(text.as_bytes());
        let latin1: Vec<u8> = text.chars().map(|c| c as u32 as u8).collect();
        for (label, bytes) in [
            ("utf8", text.as_bytes().to_vec()),
            ("utf8 bom", utf8_bom),
            ("latin1", latin1),
            ("utf16le bom", utf16(true, true)),
            ("utf16be bom", utf16(false, true)),
            ("utf16le", utf16(true, false)),
            ("utf16be", utf16(false, false)),
        ] {
            let decoded = decode_text(&bytes);
            assert!(decoded.contains("Phase(°)"), "{label}: {decoded:?}");
            let m = parse_rew_txt(&decoded).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_three_rows(&m, label);
        }
    }

    /// Audit 2026-10-23 2.5, 2.9: what must still be rejected.
    #[test]
    fn corrupt_or_absurd_data_rejected() {
        for (label, bad) in [
            ("negative freq", "-20 65\n40 70\n"),
            ("absurd level", "20 1e300\n40 70\n"),
            ("absurd phase", "20 65 1e300\n40 70 0\n"),
            ("text inside data", "20 65\ngarbage here\n40 70\n"),
        ] {
            assert!(parse_rew_txt(bad).is_err(), "{label} accepted");
        }
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
