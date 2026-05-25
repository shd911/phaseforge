// b140.17.2 — Release-readiness Stage 3.
//
// Export sweep. For band 0 of the real 4-way project, renders the FIR
// across the matrix of {sample_rate} × {taps} × {window} the user
// could plausibly select in the Export tab. Writes each to a temp
// WAV file via `export_wav_f32` / `export_wav_f64`, then parses the
// RIFF/WAVE header back and asserts:
//   - magic bytes RIFF / WAVE / fmt  / data
//   - sample-rate field round-trips
//   - bits-per-sample matches (32 / 64)
//   - data-chunk size matches num_samples * bytes-per-sample
//   - no clipping (|sample| < 1.0 after f32 cast for the f32 path)
//   - peak finite, > 0
//
// Skips when fixture missing.

use phaseforge_lib::fir::pipeline::pick_pipeline;
use phaseforge_lib::fir::{export_wav_f32, export_wav_f64, FirConfig, PhaseMode, WindowType};
use phaseforge_lib::project::ProjectFile;

use std::path::PathBuf;

const FIXTURE_REL: &str = "../test-fixtures/user_4way";
const PROJECT_FILE: &str = "4way-10-6.5-74-25.pfproj";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_REL)
}

fn load_fixture() -> Option<ProjectFile> {
    let path = fixture_root().join(PROJECT_FILE);
    if !path.exists() {
        return None;
    }
    let json = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&json).ok()
}

fn log_freq_grid(n: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let l0 = fmin.ln();
    let l1 = fmax.ln();
    (0..n)
        .map(|i| (l0 + (l1 - l0) * (i as f64) / ((n - 1) as f64)).exp())
        .collect()
}

fn windows() -> Vec<(&'static str, WindowType)> {
    vec![
        ("Blackman", WindowType::Blackman),
        ("Kaiser", WindowType::Kaiser),
        ("BlackmanHarris", WindowType::BlackmanHarris),
        ("FlatTop", WindowType::FlatTop),
    ]
}

fn sample_rates() -> Vec<u32> {
    vec![44_100, 48_000, 96_000, 176_400, 192_000]
}

fn taps_list() -> Vec<usize> {
    vec![8_192, 16_384, 65_536]
}

/// Parse a minimal RIFF/WAVE header. Returns (sample_rate, bits,
/// num_channels, data_size_bytes).
fn parse_wav_header(bytes: &[u8]) -> Option<(u32, u16, u16, u32)> {
    if bytes.len() < 44 {
        return None;
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" || &bytes[12..16] != b"fmt " {
        return None;
    }
    let num_channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    if &bytes[36..40] != b"data" {
        return None;
    }
    let data_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    Some((sr, bits, num_channels, data_size))
}

#[test]
fn release_readiness_stage3_export_sweep() {
    let Some(project) = load_fixture() else {
        eprintln!("[stage3] fixture missing — skipping");
        return;
    };
    let band = &project.bands[0];
    let hp = band.target.high_pass.as_ref();
    let lp = band.target.low_pass.as_ref();

    let freq = log_freq_grid(256, 22.0, 22_000.0);

    let tmp_dir = std::env::temp_dir().join("phaseforge_stage3");
    let _ = std::fs::create_dir_all(&tmp_dir);

    let mut total = 0usize;
    let mut failures = Vec::<String>::new();
    let mut runs = Vec::<serde_json::Value>::new();

    for &sr in sample_rates().iter() {
        for &taps in taps_list().iter() {
            for (win_name, win) in windows().iter() {
                for &bits in &[32u16, 64u16] {
                    total += 1;
                    let label = format!("sr={sr}/taps={taps}/{win_name}/bits={bits}");
                    let cfg = FirConfig {
                        taps,
                        sample_rate: sr as f64,
                        max_boost_db: 18.0,
                        noise_floor_db: -60.0,
                        window: win.clone(),
                        phase_mode: PhaseMode::Composite,
                        iterations: 2,
                        freq_weighting: true,
                        narrowband_limit: true,
                        nb_smoothing_oct: 0.333,
                        nb_max_excess_db: 6.0,
                        gaussian_min_phase_filters: vec![],
                        linear_phase_main: false,
                        subsonic_cutoff_hz: None,
                    };
                    let pipeline = pick_pipeline(hp, lp, &cfg);
                    let res = match pipeline.evaluate(hp, lp, &band.peq_bands, &cfg, &freq) {
                        Ok(r) => r,
                        Err(e) => {
                            failures.push(format!("{label}: pipeline err: {e}"));
                            continue;
                        }
                    };

                    let path = tmp_dir.join(format!(
                        "fir_{sr}_{taps}_{win_name}_{bits}.wav"
                    ));
                    let write_res = if bits == 32 {
                        export_wav_f32(&res.impulse, sr as f64, &path)
                    } else {
                        export_wav_f64(&res.impulse, sr as f64, &path)
                    };
                    if let Err(e) = write_res {
                        failures.push(format!("{label}: write err: {e}"));
                        continue;
                    }
                    let bytes = match std::fs::read(&path) {
                        Ok(b) => b,
                        Err(e) => {
                            failures.push(format!("{label}: readback err: {e}"));
                            continue;
                        }
                    };

                    let header = match parse_wav_header(&bytes) {
                        Some(h) => h,
                        None => {
                            failures.push(format!("{label}: bad WAV header"));
                            continue;
                        }
                    };
                    let (got_sr, got_bits, got_ch, got_data) = header;
                    let expected_data = (taps as u32) * (bits as u32 / 8);

                    let mut local_fail: Option<String> = None;
                    if got_sr != sr {
                        local_fail = Some(format!(
                            "{label}: SR roundtrip {got_sr} != {sr}"
                        ));
                    }
                    if local_fail.is_none() && got_bits != bits {
                        local_fail = Some(format!(
                            "{label}: bits {got_bits} != {bits}"
                        ));
                    }
                    if local_fail.is_none() && got_ch != 1 {
                        local_fail = Some(format!("{label}: ch {got_ch} != 1"));
                    }
                    if local_fail.is_none() && got_data != expected_data {
                        local_fail = Some(format!(
                            "{label}: data size {got_data} != {expected_data}"
                        ));
                    }
                    let peak = res
                        .impulse
                        .iter()
                        .copied()
                        .fold(0.0_f64, |a, b| a.max(b.abs()));
                    if local_fail.is_none() && !(peak.is_finite() && peak > 0.0) {
                        local_fail =
                            Some(format!("{label}: peak {peak} invalid"));
                    }
                    if local_fail.is_none() && bits == 32 && peak >= 1.0 {
                        local_fail = Some(format!(
                            "{label}: peak {peak} >= 1.0 — would clip f32"
                        ));
                    }

                    let _ = std::fs::remove_file(&path);

                    match local_fail {
                        Some(f) => failures.push(f),
                        None => runs.push(serde_json::json!({
                            "label": label,
                            "ok": true,
                            "wav_bytes": bytes.len(),
                            "peak": peak,
                        })),
                    }
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);

    let report = serde_json::json!({
        "version": "b140.17.2 stage3",
        "band": band.name,
        "total_runs": total,
        "failures": failures.len(),
        "failure_messages": &failures[..failures.len().min(20)],
        "sample_runs": &runs[..runs.len().min(10)],
    });
    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    let _ = std::fs::create_dir_all(&target_dir);
    let out = target_dir.join("release_report_stage3.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap())
        .expect("write report");
    eprintln!(
        "[stage3] {} runs, {} failures — wrote {}",
        total,
        failures.len(),
        out.display()
    );

    assert!(
        failures.is_empty(),
        "{} failures across {} runs; first: {}",
        failures.len(),
        total,
        failures.first().unwrap()
    );
}
