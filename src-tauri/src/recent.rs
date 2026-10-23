use serde::{Deserialize, Serialize};
use tracing::{info, warn};

const MAX_RECENT: usize = 8;
const CONFIG_FILE: &str = "recent.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RecentData {
    paths: Vec<String>,
}

fn config_path() -> Result<std::path::PathBuf, String> {
    let dir = dirs::config_dir()
        .ok_or("Cannot determine config directory")?
        .join("phaseforge");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Cannot create config dir: {e}"))?;
    Ok(dir.join(CONFIG_FILE))
}

/// Read the list. b141.102 (audit 2026-10-23 2.13): a file that does not
/// parse used to read as an empty list without a word, and the next
/// add/clear overwrote it. Now it is logged and kept as `recent.json.bad`.
fn read_recent(path: &std::path::Path) -> Result<RecentData, String> {
    if !path.exists() {
        return Ok(RecentData::default());
    }
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("Read recent error: {e}"))?;
    match serde_json::from_str::<RecentData>(&json) {
        Ok(d) => Ok(d),
        Err(e) => {
            let bad = path.with_extension("json.bad");
            let _ = std::fs::rename(path, &bad);
            warn!("recent projects list is corrupt ({e}); kept as {}, starting empty", bad.display());
            Ok(RecentData::default())
        }
    }
}

/// Atomic write (unique tmp + rename): a crash mid-write keeps the old list.
fn write_recent(path: &std::path::Path, data: &RecentData) -> Result<(), String> {
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| format!("Serialize error: {e}"))?;
    crate::project::write_atomic(path, json.as_bytes())
        .map_err(|e| format!("Write recent error: {e}"))
}

fn push_recent(data: &mut RecentData, path: String) {
    // Remove if already present, then prepend
    data.paths.retain(|p| p != &path);
    data.paths.insert(0, path);
    data.paths.truncate(MAX_RECENT);
}

#[tauri::command]
pub fn load_recent_projects() -> Result<Vec<String>, String> {
    let data = read_recent(&config_path()?)?;
    info!("load_recent_projects: {} entries", data.paths.len());
    Ok(data.paths)
}

#[tauri::command]
pub fn add_recent_project(path: String) -> Result<Vec<String>, String> {
    let config = config_path()?;
    let mut data = read_recent(&config)?;
    push_recent(&mut data, path);
    write_recent(&config, &data)?;
    info!("add_recent_project: now {} entries", data.paths.len());
    Ok(data.paths)
}

#[tauri::command]
pub fn clear_recent_projects() -> Result<(), String> {
    write_recent(&config_path()?, &RecentData::default())?;
    info!("clear_recent_projects: done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pf_recent_{label}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn corrupt_list_is_kept_aside_not_silently_lost() {
        let dir = scratch("corrupt");
        let path = dir.join(CONFIG_FILE);
        let broken = "{\"paths\": [\"/a.pfproj\",";
        std::fs::write(&path, broken).unwrap();
        let data = read_recent(&path).unwrap();
        assert!(data.paths.is_empty());
        assert!(!path.exists());
        assert_eq!(std::fs::read_to_string(dir.join("recent.json.bad")).unwrap(), broken);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn round_trip_dedups_and_caps() {
        let dir = scratch("roundtrip");
        let path = dir.join(CONFIG_FILE);
        let mut data = read_recent(&path).unwrap();
        for i in 0..10 {
            push_recent(&mut data, format!("/p{i}.pfproj"));
        }
        push_recent(&mut data, "/p5.pfproj".into());
        write_recent(&path, &data).unwrap();
        let back = read_recent(&path).unwrap();
        assert_eq!(back.paths.len(), MAX_RECENT);
        assert_eq!(back.paths[0], "/p5.pfproj");
        assert_eq!(back.paths.iter().filter(|p| *p == "/p5.pfproj").count(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
