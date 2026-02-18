use serde::{Deserialize, Serialize};
use tracing::info;

use crate::dsp::merge::MergeConfig;
use crate::io::Measurement;
use crate::peq::PeqBand;
use crate::target::TargetCurve;

// ---------------------------------------------------------------------------
// Project file data model (v1 + v2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    pub version: u32,
    pub app_name: String,

    /// v2: project name (used for file naming)
    #[serde(default)]
    pub project_name: Option<String>,

    pub bands: Vec<BandData>,
    pub active_band_id: String,
    pub show_phase: bool,
    pub show_mag: bool,
    pub show_target: bool,
    pub next_band_num: u32,

    // Export settings
    #[serde(default = "default_sample_rate")]
    pub export_sample_rate: u32,
    #[serde(default = "default_taps")]
    pub export_taps: u32,
    #[serde(default = "default_window")]
    pub export_window: String,

    // UI state
    #[serde(default = "default_tab")]
    pub active_tab: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandData {
    pub id: String,
    pub name: String,
    pub measurement: Option<Measurement>,
    /// v2: relative filename of measurement in project folder (measurement data loaded from file)
    #[serde(default)]
    pub measurement_file: Option<String>,
    pub settings: Option<SettingsData>,
    pub target: TargetCurve,
    pub target_enabled: bool,
    #[serde(default)]
    pub inverted: bool,
    #[serde(default)]
    pub linked_to_next: bool,
    #[serde(default)]
    pub peq_bands: Vec<PeqBand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsData {
    pub smoothing: String,
    pub delay_seconds: Option<f64>,
    pub distance_meters: Option<f64>,
    #[serde(default)]
    pub delay_removed: bool,
    pub original_phase: Option<Vec<f64>>,
    pub floor_bounce: Option<FloorBounceData>,
    pub merge_source: Option<MergeSourceData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorBounceData {
    pub enabled: bool,
    pub speaker_height: f64,
    pub mic_height: f64,
    pub distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSourceData {
    pub nf_path: String,
    pub ff_path: String,
    pub config: MergeConfig,
}

// ---------------------------------------------------------------------------
// Defaults for serde
// ---------------------------------------------------------------------------

fn default_sample_rate() -> u32 { 48000 }
fn default_taps() -> u32 { 65536 }
fn default_window() -> String { "Blackman".to_string() }
fn default_tab() -> String { "measurements".to_string() }

// ---------------------------------------------------------------------------
// IPC commands
// ---------------------------------------------------------------------------

const MAX_VERSION: u32 = 2;

#[tauri::command]
pub fn save_project(path: String, project: ProjectFile) -> Result<(), String> {
    info!("save_project: {}", path);
    let json = serde_json::to_string_pretty(&project)
        .map_err(|e| format!("Serialization error: {e}"))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Write error: {e}"))?;
    info!("save_project: wrote {} bytes", std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0));
    Ok(())
}

#[tauri::command]
pub fn load_project(path: String) -> Result<ProjectFile, String> {
    info!("load_project: {}", path);
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("Read error: {e}"))?;
    let project: ProjectFile = serde_json::from_str(&json)
        .map_err(|e| format!("Parse error: {e}"))?;
    if project.version > MAX_VERSION {
        return Err(format!(
            "Project version {} is newer than supported (max {}). Update PhaseForge.",
            project.version, MAX_VERSION
        ));
    }
    info!("load_project: {} bands loaded", project.bands.len());
    Ok(project)
}

// ---------------------------------------------------------------------------
// v2: Project folder management
// ---------------------------------------------------------------------------

/// Create a project folder: `parent_dir/project_name/`
#[tauri::command]
pub fn create_project_folder(parent_dir: String, project_name: String) -> Result<String, String> {
    let folder = std::path::PathBuf::from(&parent_dir).join(&project_name);
    if folder.exists() {
        return Err(format!("Folder already exists: {}", folder.display()));
    }
    std::fs::create_dir_all(&folder)
        .map_err(|e| format!("Cannot create folder: {e}"))?;
    let path = folder.to_string_lossy().to_string();
    info!("create_project_folder: {}", path);
    Ok(path)
}

/// Copy a file into the project folder.
#[tauri::command]
pub fn copy_file_to_project(source_path: String, dest_path: String) -> Result<(), String> {
    info!("copy_file_to_project: {} -> {}", source_path, dest_path);
    std::fs::copy(&source_path, &dest_path)
        .map_err(|e| format!("Copy error: {e}"))?;
    Ok(())
}

/// Check if a path exists on disk.
#[tauri::command]
pub fn check_path_exists(path: String) -> Result<bool, String> {
    Ok(std::path::Path::new(&path).exists())
}
