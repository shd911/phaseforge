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
    /// b141.10 convention marker: `Some(true)` means the alignment delays in
    /// this file use "positive = plays later" (HQPlayer parity). Absent means a
    /// legacy file that the frontend migrates on load.
    ///
    /// b141.18 (audit): the frontend wrote this flag but `ProjectFile` had no
    /// field for it, so serde dropped it on every save. The migration then fired
    /// on every load and the delays flip-flopped between two states, never
    /// converging. Round-trip pinned by `delay_convention_flag_survives_round_trip`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay_positive_is_late: Option<bool>,
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
    pub peq_smooth_fit: Option<bool>,
    #[serde(default)]
    pub peq_boost_floor: Option<f64>,
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
    /// b141.22 (audit): export sample rate the PEQ was fitted at. Absent in
    /// projects written before b141.22 — `peqStale` reads that as "unknown"
    /// rather than stale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    /// b141.67: tilt + shelves at fit time (opaque to Rust; read by `peqStale`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shaping: Option<serde_json::Value>,
    /// b141.126: the target's level at fit time, absent for a hybrid fit, which
    /// does not depend on it (read by `peqStale`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_level_db: Option<f64>,
    /// b141.103: fingerprint of the measurement the PEQ was fitted to
    /// (opaque to Rust; read by `peqStale`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_key: Option<String>,
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

/// b141.6 (audit): shared guard for every command that WRITES to a path
/// received from the webview. The commands accept arbitrary absolute paths
/// (the user picks project folders via the OS dialog, including external
/// volumes), so we cannot whitelist a root — instead reject the classic
/// code-execution write primitives a compromised webview would aim for:
/// path traversal, null bytes, hidden files/dirs (~/.zshrc, ~/.ssh, ...)
/// and macOS launch-persistence directories.
pub(crate) fn validate_write_target(path: &str) -> Result<(), String> {
    if path.contains('\0') {
        return Err("invalid path: contains null bytes".into());
    }
    let p = std::path::Path::new(path);
    for component in p.components() {
        match component {
            std::path::Component::ParentDir => {
                return Err("path traversal detected: '..' not allowed".into());
            }
            std::path::Component::Normal(name) => {
                if name.to_string_lossy().starts_with('.') {
                    return Err(format!(
                        "invalid path: writing into hidden files/directories is not allowed ({})",
                        name.to_string_lossy()
                    ));
                }
            }
            _ => {}
        }
    }
    let lower = path.to_lowercase();
    for blocked in ["launchagents", "launchdaemons", "startupitems"] {
        if lower.contains(blocked) {
            return Err(format!("invalid path: writing into {blocked} is not allowed"));
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn save_project(path: String, project: ProjectFile) -> Result<(), String> {
    info!("save_project: {}", path);
    validate_write_target(&path)?;
    let json = serde_json::to_string_pretty(&project)
        .map_err(|e| format!("Serialization error: {e}"))?;
    write_atomic(std::path::Path::new(&path), json.as_bytes())
        .map_err(|e| format!("Write error: {e}"))?;
    info!("save_project: wrote {} bytes", json.len());
    Ok(())
}

/// Atomic file write: tmp in the same directory + fsync + rename, so a crash,
/// kill or ENOSPC mid-write never leaves a truncated `.pfproj` in place of
/// the previous good one (2026-09-05 audit).
///
/// b141.102 (audit 2026-10-23 2.4): the temporary name is unique per call
/// (pid + process-wide counter, created with `create_new`). A fixed
/// `<name>.tmp` let two concurrent saves to one path truncate / rename each
/// other's temp file — a false «write error» on one of them.
pub(crate) fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    let pid = std::process::id();
    let (tmp, mut f) = loop {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_file_name(format!("{file_name}.{pid}-{seq}.tmp"));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(f) => break (tmp, f),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let written = f.write_all(bytes).and_then(|_| f.sync_all());
    drop(f);
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Reject a project whose numeric payload would panic downstream DSP
/// (hand-edited or foreign `.pfproj`): FFT sizes must be powers of two,
/// measurement arrays must be equal-length and finite. Called by
/// `load_project` and `load_snapshot` (2026-09-05 audit).
pub(crate) fn validate_project(project: &ProjectFile) -> Result<(), String> {
    let taps = project.export_taps;
    if !crate::fir::taps_valid(taps as usize) {
        return Err(format!(
            "export_taps={taps} is invalid: must be a power of two between 32 and {}",
            crate::fir::MAX_TAPS
        ));
    }
    let sr = project.export_sample_rate;
    if !(8_000..=768_000).contains(&sr) {
        return Err(format!("export_sample_rate={sr} is out of range (8000..768000)"));
    }
    // b141.101 (audit 2026-10-23 2.2): a noise floor above the boost limit
    // is not a pair the FIR can honour (it used to panic in f64::clamp).
    for (name, v) in [("fir_max_boost_db", project.fir_max_boost_db), ("fir_noise_floor_db", project.fir_noise_floor_db)] {
        if let Some(v) = v {
            if !v.is_finite() { return Err(format!("{name}={v} is not a finite number")); }
        }
    }
    if let (Some(boost), Some(floor)) = (project.fir_max_boost_db, project.fir_noise_floor_db) {
        if floor > boost {
            return Err(format!("fir_noise_floor_db={floor} is above fir_max_boost_db={boost}"));
        }
    }
    for band in &project.bands {
        // b141.101 (audit 2026-10-23 2.17): PEQ bands are hand-editable in
        // the .pfproj — gain_db 1e300 on a min-phase band gave 32768 NaN
        // samples without an error. The bounds are generous (far outside
        // what the UI produces); they reject garbage, not unusual settings.
        for (i, p) in band.peq_bands.iter().enumerate() {
            let ok = p.freq_hz.is_finite() && p.gain_db.is_finite() && p.q.is_finite()
                && (1.0..=200_000.0).contains(&p.freq_hz)
                && p.gain_db.abs() <= 100.0
                && (0.01..=100.0).contains(&p.q);
            if !ok {
                return Err(format!(
                    "band '{}': PEQ #{} is out of range (freq {} Hz, gain {} dB, Q {}; \
                     expected 1..200000 Hz, |gain| <= 100 dB, Q 0.01..100)",
                    band.name, i + 1, p.freq_hz, p.gain_db, p.q
                ));
            }
        }
        // b141.46 (audit 2026-10-01 L2): order 0 silently fell through to
        // Bessel-8 (and to "no filter" for the others). The UI offers 1..=8.
        for f in [&band.target.high_pass, &band.target.low_pass].into_iter().flatten() {
            let ordered = !matches!(f.filter_type, crate::target::FilterType::Gaussian);
            if ordered && !(1..=8).contains(&f.order) {
                return Err(format!("band '{}': filter order {} is out of range (1..8)", band.name, f.order));
            }
        }
        if let Some(m) = &band.measurement {
            let n = m.freq.len();
            if n < 2 {
                return Err(format!("band '{}': measurement has fewer than 2 points", band.name));
            }
            if m.magnitude.len() != n {
                return Err(format!(
                    "band '{}': measurement freq/magnitude length mismatch ({} vs {})",
                    band.name, n, m.magnitude.len()
                ));
            }
            if let Some(ph) = &m.phase {
                if ph.len() != n {
                    return Err(format!(
                        "band '{}': measurement freq/phase length mismatch ({} vs {})",
                        band.name, n, ph.len()
                    ));
                }
            }
            let finite = m.freq.iter().chain(m.magnitude.iter())
                .chain(m.phase.iter().flatten())
                .all(|v| v.is_finite());
            if !finite {
                return Err(format!("band '{}': measurement contains NaN/Inf", band.name));
            }
            if m.freq.windows(2).any(|w| w[1] <= w[0]) {
                return Err(format!("band '{}': measurement freq is not strictly ascending", band.name));
            }
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn load_project(path: String) -> Result<ProjectFile, String> {
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
    validate_project(&project)?;
    info!("load_project: {} bands loaded", project.bands.len());
    Ok(project)
}

// ---------------------------------------------------------------------------
// v2: Project folder management
// ---------------------------------------------------------------------------

/// Create a project folder: `parent_dir/project_name/`
#[tauri::command]
pub async fn create_project_folder(parent_dir: String, project_name: String) -> Result<String, String> {
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
    validate_write_target(&folder.to_string_lossy())?;
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
pub async fn copy_file_to_project(source_path: String, dest_path: String) -> Result<(), String> {
    validate_write_target(&dest_path)?;
    let dest = std::path::Path::new(&dest_path);

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

/// b141.103 (audit 2026-10-23 3.12): copy a measurement into `dest_dir`
/// under a name no other file there holds. `copy_file_to_project` overwrote
/// `inbox/<name>` — `/L/wfr.txt` and `/R/wfr.txt` both landed on
/// `inbox/wfr.txt` and band 1 silently reopened with band 2's measurement.
/// A taken name gets `_2`, `_3`, … before the extension; a file that is the
/// source itself or has identical bytes is reused (copying the same source
/// again yields the same file). Returns the file name chosen.
#[tauri::command]
pub async fn copy_into_project_unique(source_path: String, dest_dir: String) -> Result<String, String> {
    let source = std::path::Path::new(&source_path);
    let base = source
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty() && *n != "..")
        .ok_or_else(|| format!("invalid source file name: {source_path}"))?;
    let (stem, ext) = match base.rfind('.') {
        Some(i) if i > 0 => (&base[..i], &base[i..]),
        _ => (base, ""),
    };
    let dir = std::path::Path::new(&dest_dir);
    for k in 1..=9999u32 {
        let name = if k == 1 { base.to_string() } else { format!("{stem}_{k}{ext}") };
        let dest = dir.join(&name);
        let dest_str = dest.to_string_lossy().to_string();
        validate_write_target(&dest_str)?;
        if !dest.exists() {
            info!("copy_into_project_unique: {} -> {}", source_path, dest_str);
            std::fs::copy(source, &dest).map_err(|e| format!("Copy error: {e}"))?;
            return Ok(name);
        }
        if paths_resolve_to_same_file(source, &dest) || files_identical(source, &dest) {
            return Ok(name);
        }
    }
    Err(format!("no free file name for {base} in {dest_dir}"))
}

fn files_identical(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(ma), Ok(mb)) if ma.len() == mb.len() => {}
        _ => return false,
    }
    matches!((std::fs::read(a), std::fs::read(b)), (Ok(x), Ok(y)) if x == y)
}

/// Check if a path exists on disk.
#[tauri::command]
pub fn check_path_exists(path: String) -> Result<bool, String> {
    Ok(std::path::Path::new(&path).exists())
}

/// Ensure a directory exists (create if missing). For backward-compat with old projects.
#[tauri::command]
pub fn ensure_dir(path: String) -> Result<(), String> {
    validate_write_target(&path)?;
    std::fs::create_dir_all(&path).map_err(|e| format!("{e}"))?;
    Ok(())
}

/// Copy all top-level files from source_dir into dest_dir (non-recursive).
/// Skips subdirectories. Creates dest_dir if it doesn't exist.
#[tauri::command]
pub async fn copy_dir_contents(source_dir: String, dest_dir: String) -> Result<u32, String> {
    validate_write_target(&dest_dir)?;
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
        // b141.131 (external audit 2026-10-29): a checked folder is not a checked
        // file — `.zshrc` in the source would land in the home folder. Hidden
        // files (.DS_Store) are not project content anyway.
        if validate_write_target(&dest_file.to_string_lossy()).is_err() {
            continue;
        }
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

    /// b141.6 (audit): the write guard must block the classic persistence /
    /// code-execution write primitives while keeping normal project paths
    /// (including external volumes) working.
    #[test]
    fn write_guard_blocks_dangerous_targets() {
        for bad in [
            "/Users/x/.zshrc",
            "/Users/x/.ssh/authorized_keys",
            "/Users/x/Library/LaunchAgents/evil.plist",
            "/Library/LaunchDaemons/evil.plist",
            "/Users/x/Projects/../../.zshrc",
            "/Users/x/Proj/inbox/m.txt\0.wav",
        ] {
            assert!(validate_write_target(bad).is_err(), "must reject: {bad}");
        }
        for good in [
            "/Users/x/Music/MyProj/MyProj.pfproj",
            "/Volumes/Ext/Speakers/proj/export/fir-48000-65536.wav",
            "/Users/x/Documents/Проект... финал/inbox/meas.txt",
        ] {
            assert!(validate_write_target(good).is_ok(), "must accept: {good}");
        }
    }

    #[test]
    fn hidden_names_do_not_get_past_the_guard() {
        let src = make_scratch_dir("guard-src");
        let dst = make_scratch_dir("guard-dst");
        std::fs::write(src.join(".zshrc"), "x").unwrap();
        std::fs::write(src.join("a.txt"), "x").unwrap();
        let n = tauri::async_runtime::block_on(copy_dir_contents(
            src.to_string_lossy().into(), dst.to_string_lossy().into())).unwrap();
        assert_eq!(n, 1);
        assert!(dst.join("a.txt").exists() && !dst.join(".zshrc").exists());
        let bad = dst.join(".hidden").to_string_lossy().to_string();
        assert!(tauri::async_runtime::block_on(create_project_folder(bad, "p".into())).is_err());
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&dst);
    }

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

    /// Audit 2026-10-23 2.4: concurrent atomic writes to ONE path all
    /// succeed (a shared fixed `.tmp` failed 40/40), the result is one whole
    /// payload and no temporary file is left behind.
    #[test]
    fn concurrent_atomic_writes_to_one_path_all_succeed() {
        let dir = make_scratch_dir("atomic-race");
        let path = dir.join("proj.pfproj");
        let handles: Vec<_> = (0..8).map(|t| {
            let path = path.clone();
            std::thread::spawn(move || {
                let payload = format!("{t}:").repeat(20_000);
                (0..20).map(|_| write_atomic(&path, payload.as_bytes()).is_ok()).all(|ok| ok)
            })
        }).collect();
        for h in handles {
            assert!(h.join().unwrap(), "a concurrent save reported a write error");
        }
        let body = std::fs::read_to_string(&path).unwrap();
        let first = body.split(':').next().unwrap().to_string();
        assert_eq!(body, format!("{first}:").repeat(20_000), "torn file");
        let leftovers: Vec<_> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temporary files left: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
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
        let result = tauri::async_runtime::block_on(copy_file_to_project(same.clone(), same.clone()));
        assert!(result.is_ok(), "self-copy must not fail: {:?}", result.err());

        let after_size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(after_size, original_size,
            "self-copy must not truncate: original={}, after={}", original_size, after_size);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn copy_into_project_unique_keeps_same_named_sources_apart() {
        let dir = make_scratch_dir("unique-copy");
        let (l, r, inbox) = (dir.join("L"), dir.join("R"), dir.join("inbox"));
        for d in [&l, &r, &inbox] { std::fs::create_dir_all(d).unwrap(); }
        std::fs::write(l.join("wfr.txt"), "left").unwrap();
        std::fs::write(r.join("wfr.txt"), "right").unwrap();
        let run = |src: &std::path::Path| tauri::async_runtime::block_on(copy_into_project_unique(
            src.to_string_lossy().to_string(), inbox.to_string_lossy().to_string(),
        )).unwrap();
        assert_eq!(run(&l.join("wfr.txt")), "wfr.txt");
        assert_eq!(run(&r.join("wfr.txt")), "wfr_2.txt");
        // The same source again → the same file, no third copy.
        assert_eq!(run(&l.join("wfr.txt")), "wfr.txt");
        assert_eq!(run(&r.join("wfr.txt")), "wfr_2.txt");
        // A file already inside inbox/ → itself.
        assert_eq!(run(&inbox.join("wfr_2.txt")), "wfr_2.txt");
        assert_eq!(std::fs::read_to_string(inbox.join("wfr.txt")).unwrap(), "left");
        assert_eq!(std::fs::read_to_string(inbox.join("wfr_2.txt")).unwrap(), "right");
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

        let result = tauri::async_runtime::block_on(copy_file_to_project(
            src.to_string_lossy().to_string(),
            dst.to_string_lossy().to_string(),
        ));
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
        let result = tauri::async_runtime::block_on(copy_dir_contents(same.clone(), same.clone()));
        assert!(result.is_ok(), "dir self-copy must not fail: {:?}", result.err());

        assert_eq!(std::fs::metadata(&f1).unwrap().len(), s1, "a.txt zeroed");
        assert_eq!(std::fs::metadata(&f2).unwrap().len(), s2, "b.txt zeroed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// b141.18 (audit, CRITICAL): the delay-convention marker written by the
    /// frontend had no field in `ProjectFile`, so `save_project` serialised it
    /// away. `restoreState` then saw an unmarked file, re-ran
    /// `migrateDelayConvention`, and the alignment delays flipped on every
    /// open — `[788, 60, 40, 0, 0] µs` became `[0, 728, 748, 788, 788]` and
    /// back again, without ever converging.
    /// A hand-edited `.pfproj` with a non-power-of-two tap count or a
    /// ragged measurement must be rejected at load, not panic inside the
    /// FFT later (2026-09-05 audit).
    #[test]
    fn validate_project_rejects_bad_taps_and_ragged_measurement() {
        let base = |extra: &str| format!(r#"{{
            "version": 2, "app_name": "PhaseForge", "bands": [{}],
            "active_band_id": "b1", "show_phase": true, "show_mag": true,
            "show_target": true, "next_band_num": 2, "export_taps": 1000
        }}"#, extra);
        let project: ProjectFile = serde_json::from_str(&base("")).expect("parse");
        let err = validate_project(&project).unwrap_err();
        assert!(err.contains("export_taps"), "{err}");

        let band = r#"{"id":"b1","name":"W","measurement":{"name":"m","source_path":null,
            "sample_rate":null,"freq":[20.0,30.0,40.0],"magnitude":[80.0,81.0],"phase":null,
            "metadata":{"date":null,"mic":null,"notes":null,"smoothing":null}},
            "settings":null,"target":{"reference_level_db":80.0,"tilt_db_per_octave":0.0,"tilt_ref_freq":1000.0,"high_pass":null,"low_pass":null,"low_shelf":null,"high_shelf":null},"target_enabled":true}"#;
        let src = base(band).replace("\"export_taps\": 1000", "\"export_taps\": 65536");
        let project: ProjectFile = serde_json::from_str(&src).expect("parse band");
        let err = validate_project(&project).unwrap_err();
        assert!(err.contains("length mismatch"), "{err}");
    }

    /// b141.46 (audit 2026-10-01 L2): filter order 0 must be rejected.
    #[test]
    fn validate_project_rejects_filter_order_zero() {
        let band = r#"{"id":"b1","name":"W","measurement":null,"settings":null,
            "target":{"reference_level_db":80.0,"tilt_db_per_octave":0.0,"tilt_ref_freq":1000.0,
            "high_pass":{"filter_type":"Bessel","order":0,"freq_hz":80.0,"shape":null},
            "low_pass":null,"low_shelf":null,"high_shelf":null},"target_enabled":true}"#;
        let src = format!(r#"{{"version": 2, "app_name": "PhaseForge", "bands": [{band}],
            "active_band_id": "b1", "show_phase": true, "show_mag": true,
            "show_target": true, "next_band_num": 2, "export_taps": 65536}}"#);
        let project: ProjectFile = serde_json::from_str(&src).expect("parse");
        let err = validate_project(&project).unwrap_err();
        assert!(err.contains("order"), "{err}");
        let ok: ProjectFile = serde_json::from_str(&src.replace("\"order\":0", "\"order\":4")).unwrap();
        assert!(validate_project(&ok).is_ok());
    }

    /// b141.101 (audit 2026-10-23 2.2, 2.17): garbage PEQ and an inverted
    /// floor/boost pair are rejected instead of reaching the DSP.
    #[test]
    fn validate_project_rejects_bad_peq_and_floor_above_boost() {
        let band = |peq: &str| format!(r#"{{"id":"b1","name":"W","measurement":null,"settings":null,
            "target":{{"reference_level_db":80.0,"tilt_db_per_octave":0.0,"tilt_ref_freq":1000.0,
            "high_pass":null,"low_pass":null,"low_shelf":null,"high_shelf":null}},
            "target_enabled":true,"peq_bands":[{peq}]}}"#);
        let proj = |band: String, extra: &str| format!(r#"{{"version": 2, "app_name": "PhaseForge",
            "bands": [{band}], "active_band_id": "b1", "show_phase": true, "show_mag": true,
            "show_target": true, "next_band_num": 2, "export_taps": 65536{extra}}}"#);
        let ok = r#"{"freq_hz":100.0,"gain_db":-3.0,"q":2.0}"#;
        let p: ProjectFile = serde_json::from_str(&proj(band(ok), "")).unwrap();
        assert!(validate_project(&p).is_ok());
        for bad in [r#"{"freq_hz":100.0,"gain_db":1e300,"q":2.0}"#,
                    r#"{"freq_hz":0.0,"gain_db":3.0,"q":2.0}"#,
                    r#"{"freq_hz":100.0,"gain_db":3.0,"q":0.0}"#] {
            let p: ProjectFile = serde_json::from_str(&proj(band(bad), "")).unwrap();
            let err = validate_project(&p).unwrap_err();
            assert!(err.contains("PEQ #1"), "{err}");
        }
        let extra = r#", "fir_max_boost_db": 6.0, "fir_noise_floor_db": 10.0"#;
        let p: ProjectFile = serde_json::from_str(&proj(band(ok), extra)).unwrap();
        assert!(validate_project(&p).unwrap_err().contains("fir_noise_floor_db"));
    }

    #[test]
    fn delay_convention_flag_survives_round_trip() {
        let src = r#"{
            "version": 2,
            "delay_positive_is_late": true,
            "app_name": "PhaseForge",
            "bands": [],
            "active_band_id": "b1",
            "show_phase": true,
            "show_mag": true,
            "show_target": true,
            "next_band_num": 2
        }"#;
        let project: ProjectFile = serde_json::from_str(src).expect("parse");
        assert_eq!(project.delay_positive_is_late, Some(true));

        let out = serde_json::to_string(&project).expect("serialise");
        assert!(out.contains("\"delay_positive_is_late\":true"),
            "flag must survive save — otherwise the migration re-fires on every load: {out}");

        // A legacy file (no marker) must still read as "needs migration".
        let legacy = src.replace("\"delay_positive_is_late\": true,", "");
        let legacy: ProjectFile = serde_json::from_str(&legacy).expect("parse legacy");
        assert_eq!(legacy.delay_positive_is_late, None);
        // ...and must not gain the key when written back untouched.
        let out = serde_json::to_string(&legacy).expect("serialise legacy");
        assert!(!out.contains("delay_positive_is_late"));
    }
}
