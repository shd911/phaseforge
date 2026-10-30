//! b141.63: persistent log file. Everything `tracing` emits (Rust) plus the
//! frontend's warnings/errors and tagged `[XX]` diagnostics (forwarded by
//! `log_frontend`) go to `phaseforge.log` — so a problem can be read back
//! from the file instead of being copied out of the devtools console.
//!
//! macOS: ~/Library/Logs/PhaseForge/, Windows: %LOCALAPPDATA%\PhaseForge\logs\,
//! elsewhere ~/.local/state/phaseforge/. One previous file is kept
//! (`phaseforge.1.log`) when the current one passes 5 MB — at startup and,
//! since b141.102, during the session.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
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
pub fn open_log_file() -> Option<Mutex<RotatingLog>> {
    let dir = log_dir()?;
    fs::create_dir_all(&dir).ok()?;
    RotatingLog::open(dir.join("phaseforge.log"), MAX_BYTES).ok().map(Mutex::new)
}

/// b141.102 (audit 2026-10-23 2.13): the log file rotates DURING a session
/// too — it was checked only at startup, so a long session (a PEQ drag logs
/// on every step) grew it without bound. When a write would pass `max`
/// bytes the file becomes `<stem>.1.log` (replacing the older one) and a
/// fresh file is started: at most ~2 × `max` on disk.
pub struct RotatingLog {
    path: PathBuf,
    file: File,
    len: u64,
    max: u64,
}

impl RotatingLog {
    pub fn open(path: PathBuf, max: u64) -> std::io::Result<Self> {
        if fs::metadata(&path).map(|m| m.len() > max).unwrap_or(false) {
            let _ = fs::rename(&path, Self::previous(&path));
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self { path, file, len, max })
    }

    fn previous(path: &std::path::Path) -> PathBuf {
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        path.with_file_name(format!("{stem}.1.log"))
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        fs::rename(&self.path, Self::previous(&self.path))?;
        self.file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        self.len = 0;
        Ok(())
    }
}

impl Write for RotatingLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.len > 0 && self.len + buf.len() as u64 > self.max {
            // A failed rotation (the old file held open elsewhere) must not
            // stop logging: keep appending and retry after another `max`.
            if self.rotate().is_err() {
                self.len = 0;
            }
        }
        let n = self.file.write(buf)?;
        self.len += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_rotates_during_the_session() {
        let dir = std::env::temp_dir().join(format!("pf_log_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("phaseforge.log");
        let mut log = RotatingLog::open(path.clone(), 1000).unwrap();
        let line = [b'x'; 99];
        for _ in 0..50 {
            log.write_all(&line).unwrap();
            log.write_all(b"\n").unwrap();
        }
        log.flush().unwrap();
        let cur = fs::metadata(&path).unwrap().len();
        let prev = fs::metadata(dir.join("phaseforge.1.log")).unwrap().len();
        assert!(cur <= 1000 && prev <= 1000, "current {cur} B, previous {prev} B");
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn reveal_reports_a_failed_exit_code() {
        assert!(run_reveal(&mut std::process::Command::new("true")).is_ok());
        assert!(run_reveal(&mut std::process::Command::new("false")).is_err());
    }
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

/// b141.131 (external audit 2026-10-29): `open -R` on a missing file exits 1, and
/// the command reported success. Explorer exits 1 on success too: Windows checks
/// only that it started.
fn run_reveal(cmd: &mut std::process::Command) -> Result<(), String> {
    if cfg!(target_os = "windows") {
        cmd.spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
    let status = cmd.status().map_err(|e| e.to_string())?;
    if status.success() { Ok(()) } else { Err(format!("{:?}: {status}", cmd.get_program())) }
}

/// Reveal the log file in Finder / Explorer.
#[tauri::command]
pub async fn reveal_log_file() -> Result<String, String> {
    let path = log_path().ok_or("no log directory")?;
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = std::process::Command::new("open");
        c.arg("-R").arg(&path);
        c
    };
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = std::process::Command::new("explorer");
        c.arg(format!("/select,{}", path.display()));
        c
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let mut cmd = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(path.parent().unwrap_or(&path));
        c
    };
    run_reveal(&mut cmd)?;
    Ok(path.display().to_string())
}
