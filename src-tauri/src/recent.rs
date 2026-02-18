use serde::{Deserialize, Serialize};
use tracing::info;

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

#[tauri::command]
pub fn load_recent_projects() -> Result<Vec<String>, String> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(vec![]);
    }
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("Read recent error: {e}"))?;
    let data: RecentData = serde_json::from_str(&json).unwrap_or_default();
    info!("load_recent_projects: {} entries", data.paths.len());
    Ok(data.paths)
}

#[tauri::command]
pub fn add_recent_project(path: String) -> Result<Vec<String>, String> {
    let config = config_path()?;
    let mut data = if config.exists() {
        let json = std::fs::read_to_string(&config).unwrap_or_default();
        serde_json::from_str::<RecentData>(&json).unwrap_or_default()
    } else {
        RecentData::default()
    };
    // Remove if already present, then prepend
    data.paths.retain(|p| p != &path);
    data.paths.insert(0, path);
    data.paths.truncate(MAX_RECENT);
    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| format!("Serialize error: {e}"))?;
    std::fs::write(&config, json)
        .map_err(|e| format!("Write recent error: {e}"))?;
    info!("add_recent_project: now {} entries", data.paths.len());
    Ok(data.paths)
}

#[tauri::command]
pub fn clear_recent_projects() -> Result<(), String> {
    let config = config_path()?;
    let data = RecentData { paths: vec![] };
    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| format!("Serialize error: {e}"))?;
    std::fs::write(&config, json)
        .map_err(|e| format!("Write recent error: {e}"))?;
    info!("clear_recent_projects: done");
    Ok(())
}
