// Export: WAV (f32/f64), PEQ configs (APO, miniDSP, Roon, CamillaDSP), project files

use std::path::PathBuf;
use tracing::info;

/// Export target curve as REW-compatible TXT file.
///
/// Format:
/// ```text
/// * Freq(Hz)	SPL(dB)
/// 20.000	-3.5
/// 25.119	-2.1
/// ...
/// ```
#[tauri::command]
pub fn export_target_txt(freq: Vec<f64>, magnitude: Vec<f64>, path: String) -> Result<(), String> {
    if freq.len() != magnitude.len() {
        return Err("freq and magnitude length mismatch".into());
    }
    let p = PathBuf::from(&path);
    let mut out = String::with_capacity(freq.len() * 24);
    out.push_str("* Freq(Hz)\tSPL(dB)\n");
    for i in 0..freq.len() {
        out.push_str(&format!("{:.3}\t{:.3}\n", freq[i], magnitude[i]));
    }
    std::fs::write(&p, &out).map_err(|e| format!("Write error: {e}"))?;
    info!("export_target_txt: {} points → {}", freq.len(), path);
    Ok(())
}
