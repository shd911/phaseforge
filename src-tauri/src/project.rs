use serde::{Deserialize, Serialize};
use tracing::info;

use crate::analysis::AnalysisResult;
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

    // PEQ/FIR settings (saved by frontend, passed through for round-trip)
    #[serde(default)]
    pub export_hybrid_phase: Option<bool>,
    #[serde(default)]
    pub peq_tolerance: Option<f64>,
    #[serde(default)]
    pub peq_max_bands: Option<u32>,
    #[serde(default)]
    pub peq_gain_regularization: Option<f64>,
    #[serde(default)]
    pub peq_floor: Option<u32>,
    #[serde(default)]
    pub peq_range_mode: Option<String>,
    #[serde(default)]
    pub peq_direct_low: Option<f64>,
    #[serde(default)]
    pub peq_direct_high: Option<f64>,
    #[serde(default)]
    pub fir_iterations: Option<u32>,
    #[serde(default)]
    pub fir_freq_weighting: Option<bool>,
    #[serde(default)]
    pub fir_narrowband_limit: Option<bool>,
    #[serde(default)]
    pub fir_nb_smoothing_oct: Option<f64>,
    #[serde(default)]
    pub fir_nb_max_excess_db: Option<f64>,
    #[serde(default)]
    pub fir_max_boost_db: Option<f64>,
    #[serde(default)]
    pub fir_noise_floor_db: Option<f64>,

    // Snapshot metadata: present only inside snapshots/<id>.pfproj copies.
    // Lets rebuild_snapshot_index reconstruct the index from orphan files.
    // skip_serializing_if keeps the live <name>.pfproj free of these fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_app_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_ts: Option<String>,
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
    #[serde(default)]
    pub exclusion_zones: Vec<serde_json::Value>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub alignment_delay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peq_optimized_target: Option<PeqOptimizedTargetData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeqOptimizedTargetData {
    pub high_pass: Option<crate::target::FilterConfig>,
    pub low_pass: Option<crate::target::FilterConfig>,
    #[serde(default)]
    pub exclusion_zones: Vec<serde_json::Value>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analysis: Option<AnalysisResult>,
    #[serde(default)]
    pub analysis_dismissed: bool,
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
    // Security check: prevent path traversal
    let p = std::path::Path::new(&path);
    for component in p.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err("path traversal detected: '..' not allowed in save path".into());
        }
    }
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
    // Sanitize project_name: extract only the filename, reject path traversal attempts
    let safe_name = std::path::Path::new(&project_name)
        .file_name()
        .ok_or("invalid project name: cannot be empty or contain '..'")?
        .to_string_lossy()
        .to_string();

    // Additional check: ensure no ".." in the sanitized name
    if safe_name.contains("..") {
        return Err("invalid project name: contains '..'".into());
    }

    let folder = std::path::PathBuf::from(&parent_dir).join(&safe_name);
    if folder.exists() {
        return Err(format!("Folder already exists: {}", folder.display()));
    }
    std::fs::create_dir_all(&folder)
        .map_err(|e| format!("Cannot create folder: {e}"))?;
    // Create standard sub-directories
    for sub in &["inbox", "target", "export"] {
        std::fs::create_dir_all(folder.join(sub))
            .map_err(|e| format!("Cannot create {sub}/ folder: {e}"))?;
    }
    let path = folder.to_string_lossy().to_string();
    info!("create_project_folder: {}", path);
    Ok(path)
}

/// Resolve canonical paths for source and destination, then return true when
/// they point at the same on-disk file. Used to short-circuit copies that
/// would otherwise truncate the source (std::fs::copy opens dest first with
/// O_TRUNC, which empties the file before reading source). The destination
/// is allowed to not exist yet — in that case we canonicalise its parent
/// directory and join the file name back.
fn paths_resolve_to_same_file(source: &std::path::Path, dest: &std::path::Path) -> bool {
    let Ok(src_canon) = std::fs::canonicalize(source) else {
        return false;
    };
    let dst_canon_opt = if dest.exists() {
        std::fs::canonicalize(dest).ok()
    } else {
        dest.parent()
            .and_then(|p| std::fs::canonicalize(p).ok())
            .and_then(|cp| dest.file_name().map(|f| cp.join(f)))
    };
    matches!(dst_canon_opt, Some(d) if d == src_canon)
}

/// Copy a file into the project folder.
#[tauri::command]
pub fn copy_file_to_project(source_path: String, dest_path: String) -> Result<(), String> {
    // Reject paths containing ".." to prevent path traversal
    let dest = std::path::Path::new(&dest_path);
    for component in dest.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err("Invalid destination path: contains '..'".into());
        }
    }

    // b139.5.2: guard against self-copy. If a user drops a file directly into
    // <project>/inbox/ and then "imports" it via the picker pointing at the
    // same path, std::fs::copy would truncate dest before reading source and
    // leave the user with a 0-byte file. Skip the copy when paths resolve to
    // the same on-disk file.
    let source = std::path::Path::new(&source_path);
    if paths_resolve_to_same_file(source, dest) {
        info!(
            "copy_file_to_project: source == dest ({}), skipping self-copy",
            source_path
        );
        return Ok(());
    }

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

/// Ensure a directory exists (create if missing). For backward-compat with old projects.
#[tauri::command]
pub fn ensure_dir(path: String) -> Result<(), String> {
    std::fs::create_dir_all(&path).map_err(|e| format!("{e}"))?;
    Ok(())
}

/// Copy all top-level files from source_dir into dest_dir (non-recursive).
/// Skips subdirectories. Creates dest_dir if it doesn't exist.
#[tauri::command]
pub fn copy_dir_contents(source_dir: String, dest_dir: String) -> Result<u32, String> {
    let src = std::path::Path::new(&source_dir);
    let dst = std::path::Path::new(&dest_dir);
    if !src.is_dir() {
        return Ok(0); // source doesn't exist or isn't a dir — nothing to copy
    }
    std::fs::create_dir_all(dst).map_err(|e| format!("Cannot create {}: {e}", dst.display()))?;
    let mut count = 0u32;
    for entry in std::fs::read_dir(src).map_err(|e| format!("Read dir error: {e}"))? {
        let entry = entry.map_err(|e| format!("Dir entry error: {e}"))?;
        let ft = entry.file_type().map_err(|e| format!("File type error: {e}"))?;
        if ft.is_symlink() || !ft.is_file() {
            continue;
        }
        let dest_file = dst.join(entry.file_name());
        // b139.5.2: same self-copy guard — when source_dir == dest_dir the
        // entries resolve to themselves and std::fs::copy would zero them.
        if paths_resolve_to_same_file(&entry.path(), &dest_file) {
            count += 1;
            continue;
        }
        std::fs::copy(entry.path(), &dest_file)
            .map_err(|e| format!("Copy error {}: {e}", entry.path().display()))?;
        count += 1;
    }
    info!("copy_dir_contents: {} -> {} ({} files)", source_dir, dest_dir, count);
    Ok(count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_DIR_COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Allocate a unique scratch dir under the OS temp root. Caller is
    /// responsible for `remove_dir_all` cleanup.
    fn make_scratch_dir(label: &str) -> std::path::PathBuf {
        let n = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("phaseforge-{label}-{pid}-{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn copy_file_to_project_handles_self_copy() {
        let dir = make_scratch_dir("self-copy");
        let path = dir.join("test.txt");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, "important data").unwrap();
        }
        let original_size = std::fs::metadata(&path).unwrap().len();
        assert!(original_size > 0);

        let same = path.to_string_lossy().to_string();
        let result = copy_file_to_project(same.clone(), same.clone());
        assert!(result.is_ok(), "self-copy must not fail: {:?}", result.err());

        let after_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(after_size, original_size,
            "self-copy must not truncate: original={}, after={}", original_size, after_size);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn copy_file_to_project_normal_copy_works() {
        let dir = make_scratch_dir("normal-copy");
        let src = dir.join("src.txt");
        let dst = dir.join("dst.txt");
        {
            let mut f = std::fs::File::create(&src).unwrap();
            writeln!(f, "data").unwrap();
        }

        let result = copy_file_to_project(
            src.to_string_lossy().to_string(),
            dst.to_string_lossy().to_string(),
        );
        assert!(result.is_ok(), "normal copy failed: {:?}", result.err());
        assert!(dst.exists());
        assert_eq!(
            std::fs::metadata(&dst).unwrap().len(),
            std::fs::metadata(&src).unwrap().len(),
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn copy_dir_contents_handles_self_copy() {
        // When source_dir == dest_dir each top-level file would otherwise
        // truncate itself. The guard keeps file content intact.
        let dir = make_scratch_dir("dir-self-copy");
        let f1 = dir.join("a.txt");
        let f2 = dir.join("b.txt");
        {
            let mut a = std::fs::File::create(&f1).unwrap();
            writeln!(a, "alpha").unwrap();
            let mut b = std::fs::File::create(&f2).unwrap();
            writeln!(b, "beta beta beta").unwrap();
        }
        let s1 = std::fs::metadata(&f1).unwrap().len();
        let s2 = std::fs::metadata(&f2).unwrap().len();

        let same = dir.to_string_lossy().to_string();
        let result = copy_dir_contents(same.clone(), same.clone());
        assert!(result.is_ok(), "dir self-copy must not fail: {:?}", result.err());

        assert_eq!(std::fs::metadata(&f1).unwrap().len(), s1, "a.txt zeroed");
        assert_eq!(std::fs::metadata(&f2).unwrap().len(), s2, "b.txt zeroed");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
