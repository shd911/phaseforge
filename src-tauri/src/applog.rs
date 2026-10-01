//! b141.63: persistent log file. Everything `tracing` emits (Rust) plus the
//! frontend's warnings/errors and tagged `[XX]` diagnostics (forwarded by
//! `log_frontend`) go to `phaseforge.log` — so a problem can be read back
//! from the file instead of being copied out of the devtools console.
//!
//! macOS: ~/Library/Logs/PhaseForge/, Windows: %LOCALAPPDATA%\PhaseForge\logs\,
//! elsewhere ~/.local/state/phaseforge/. One previous file is kept
//! (`phaseforge.1.log`) when the current one passes 5 MB at startup.

use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::sync::Mutex;

use tracing::{error, info, warn};

const MAX_BYTES: u64 = 5 * 1024 * 1024;

pub fn log_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    { std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Logs/PhaseForge")) }
    #[cfg(target_os = "windows")]
    { std::env::var_os("LOCALAPPDATA").map(|h| PathBuf::from(h).join("PhaseForge").join("logs")) }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    { std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state/phaseforge")) }
}

pub fn log_path() -> Option<PathBuf> {
    log_dir().map(|d| d.join("phaseforge.log"))
}

/// Open (and rotate) the log file. None when the directory cannot be made —
/// logging then stays on stdout only.
pub fn open_log_file() -> Option<Mutex<File>> {
    let dir = log_dir()?;
    fs::create_dir_all(&dir).ok()?;
    let path = dir.join("phaseforge.log");
    if fs::metadata(&path).map(|m| m.len() > MAX_BYTES).unwrap_or(false) {
        let _ = fs::rename(&path, dir.join("phaseforge.1.log"));
    }
    OpenOptions::new().create(true).append(true).open(&path).ok().map(Mutex::new)
}

/// Frontend → log file. `level`: "error" | "warn" | anything else = info.
#[tauri::command]
pub fn log_frontend(level: String, message: String) {
    let msg: String = message.chars().take(4000).collect();
    match level.as_str() {
        "error" => error!(target: "ui", "{msg}"),
        "warn" => warn!(target: "ui", "{msg}"),
        _ => info!(target: "ui", "{msg}"),
    }
}

/// Reveal the log file in Finder / Explorer.
#[tauri::command]
pub async fn reveal_log_file() -> Result<String, String> {
    let path = log_path().ok_or("no log directory")?;
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg("-R").arg(&path).spawn();
    #[cfg(target_os = "windows")]
    let r = std::process::Command::new("explorer").arg(format!("/select,{}", path.display())).spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let r = std::process::Command::new("xdg-open").arg(path.parent().unwrap_or(&path)).spawn();
    r.map_err(|e| e.to_string())?;
    Ok(path.display().to_string())
}
