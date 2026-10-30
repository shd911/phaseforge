use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::project::ProjectFile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub id: String,
    pub ts: String,
    pub description: String,
    pub app_version: String,
    pub file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotIndex {
    pub version: u32,
    pub entries: Vec<SnapshotEntry>,
}

impl SnapshotIndex {
    fn empty() -> Self {
        Self { version: 1, entries: Vec::new() }
    }
}

const INDEX_CORRUPTED: &str = "INDEX_CORRUPTED";

fn validate_dir(project_dir: &str) -> Result<PathBuf, String> {
    if project_dir.is_empty() {
        return Err("project_dir is empty".into());
    }
    let p = PathBuf::from(project_dir);
    for c in p.components() {
        if matches!(c, std::path::Component::ParentDir) {
            return Err("project_dir must not contain '..'".into());
        }
    }
    if !p.is_dir() {
        return Err(format!("project_dir does not exist: {}", p.display()));
    }
    Ok(p)
}

fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 64 {
        return Err("invalid snapshot id length".into());
    }
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err("snapshot id contains invalid characters".into());
    }
    Ok(())
}

fn snapshots_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("snapshots")
}

fn index_path(project_dir: &Path) -> PathBuf {
    snapshots_dir(project_dir).join("index.json")
}

fn read_index(project_dir: &Path) -> Result<SnapshotIndex, String> {
    let path = index_path(project_dir);
    if !path.exists() {
        return Ok(SnapshotIndex::empty());
    }
    let txt = std::fs::read_to_string(&path)
        .map_err(|e| format!("read index: {e}"))?;
    serde_json::from_str(&txt).map_err(|_| INDEX_CORRUPTED.to_string())
}

fn write_index(project_dir: &Path, idx: &SnapshotIndex) -> Result<(), String> {
    let dir = snapshots_dir(project_dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create snapshots dir: {e}"))?;
    let json = serde_json::to_string_pretty(idx).map_err(|e| format!("serialize index: {e}"))?;
    // Atomic write: unique tmp + rename. Crash mid-write leaves the old index intact.
    crate::project::write_atomic(&index_path(project_dir), json.as_bytes())
        .map_err(|e| format!("write index: {e}"))
}

/// b141.102 (audit 2026-10-23 2.4): every read-modify-write of `index.json`
/// runs under this lock. Two concurrent create_snapshot calls each read the
/// index, pushed their entry and wrote it back — the later write dropped the
/// earlier entry (the .pfproj stayed on disk, invisible in the list).
static INDEX_LOCK: Mutex<()> = Mutex::new(());

fn index_lock() -> std::sync::MutexGuard<'static, ()> {
    INDEX_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn fresh_id() -> String {
    let now = Utc::now();
    let stamp = now.format("%Y%m%dT%H%M%S");
    // Low 16 bits of nanoseconds within the second — collision risk between
    // two clicks in the same second is ~1/65k, acceptable for manual snapshots.
    let nanos = now.timestamp_subsec_nanos();
    format!("{}_{:04x}", stamp, nanos & 0xFFFF)
}

#[tauri::command]
pub async fn create_snapshot(
    project_dir: String,
    description: String,
    app_version: String,
    project: ProjectFile,
) -> Result<SnapshotEntry, String> {
    let dir = validate_dir(&project_dir)?;
    if description.trim().is_empty() {
        return Err("description must not be empty".into());
    }
    if description.chars().count() > 1000 {
        return Err("description too long (>1000 chars)".into());
    }
    // b141.122 (external audit 2026-10-29): a corrupt index is refused before
    // anything is written. It used to be replaced with one holding only the
    // new version: every older one vanished from the list until a Rebuild.
    read_index(&dir)?;

    let now = Utc::now();
    let ts = now.to_rfc3339_opts(SecondsFormat::Secs, true);
    let snap_dir = snapshots_dir(&dir);
    std::fs::create_dir_all(&snap_dir).map_err(|e| format!("create snapshots dir: {e}"))?;

    // Pick an id that doesn't collide with an existing file. fresh_id has a
    // ~1/65k same-second collision rate; retry until create_new succeeds.
    let mut copy = project;
    copy.snapshot_description = Some(description.clone());
    copy.snapshot_app_version = Some(app_version.clone());
    copy.snapshot_ts = Some(ts.clone());

    let mut id = String::new();
    let mut snap_path = PathBuf::new();
    let mut file: Option<std::fs::File> = None;
    for attempt in 0..32 {
        let candidate = if attempt == 0 { fresh_id() } else { format!("{}_{:x}", fresh_id(), attempt) };
        validate_id(&candidate)?;
        let path = snap_dir.join(format!("{candidate}.pfproj"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(f) => {
                id = candidate;
                snap_path = path;
                file = Some(f);
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("create snapshot file: {e}")),
        }
    }
    if id.is_empty() {
        return Err("failed to allocate unique snapshot id".into());
    }
    // b141.102 (audit 2026-10-23 2.7): a failure after the file was
    // allocated (copying a measurement, writing JSON, the index) used to
    // leave an empty `<id>.pfproj` and a half-filled `<id>/` behind.
    let file = file.take().expect("file handle allocated above");
    match finish_snapshot(&dir, &id, copy, file, ts, description, app_version) {
        Ok(entry) => {
            info!("create_snapshot: {} ({})", entry.id, snap_path.display());
            Ok(entry)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&snap_path);
            let _ = std::fs::remove_dir_all(snapshots_dir(&dir).join(&id));
            Err(e)
        }
    }
}

fn finish_snapshot(
    dir: &Path,
    id: &str,
    mut copy: ProjectFile,
    mut file: std::fs::File,
    ts: String,
    description: String,
    app_version: String,
) -> Result<SnapshotEntry, String> {
    // 2026-09-05 audit: a snapshot referenced `inbox/<name>` by path only, so a
    // later re-import with the same file name silently rewrote every older
    // version's measurement. Copy referenced files into `snapshots/<id>/`
    // and point the snapshot at those copies.
    freeze_measurement_files(dir, id, &mut copy)?;
    copy.snapshot_id = Some(id.to_string());
    let json = serde_json::to_string_pretty(&copy).map_err(|e| format!("serialize snapshot: {e}"))?;
    file.write_all(json.as_bytes()).map_err(|e| format!("write snapshot: {e}"))?;
    file.sync_all().ok();
    drop(file);

    let entry = SnapshotEntry {
        id: id.to_string(),
        ts,
        description,
        app_version,
        file: format!("{id}.pfproj"),
    };

    let _guard = index_lock();
    let mut idx = read_index(dir)?;
    idx.entries.push(entry.clone());
    write_index(dir, &idx)?;
    Ok(entry)
}

/// Resolve a measurement reference the way `restoreState` does: absolute
/// path, then project-relative, then `inbox/<basename>`, then `<basename>`.
fn resolve_measurement_source(project_dir: &Path, reference: &str) -> Option<PathBuf> {
    if reference.is_empty() || reference.contains("..") {
        return None;
    }
    let base = Path::new(reference).file_name()?.to_string_lossy().into_owned();
    let candidates = [
        PathBuf::from(reference),
        project_dir.join(reference),
        project_dir.join("inbox").join(&base),
        project_dir.join(&base),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Copy every measurement / merge-source file the snapshot references into
/// `snapshots/<id>/` and rewrite the references to those copies
/// (project-relative). Missing sources are left untouched — the snapshot
/// then degrades exactly like the live project would.
fn freeze_measurement_files(project_dir: &Path, id: &str, project: &mut ProjectFile) -> Result<(), String> {
    let files_dir = snapshots_dir(project_dir).join(id);
    let rel_dir = format!("snapshots/{id}");
    // b141.102 (audit 2026-10-23 2.3): one name per SOURCE file, and a name
    // is never handed to two different sources. Before, a second file with a
    // taken basename got `<tag>_<base>` but was recorded under that new name
    // only — a third same-named file (the NF of another band) found
    // `<tag>_<base>` already on disk and was pointed at THAT band's copy: the
    // version restored with a foreign measurement and no error.
    let mut by_source: std::collections::HashMap<PathBuf, String> = std::collections::HashMap::new();
    let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut freeze = |reference: &mut String, tag: &str| -> Result<(), String> {
        let Some(src) = resolve_measurement_source(project_dir, reference) else { return Ok(()); };
        let key = std::fs::canonicalize(&src).unwrap_or_else(|_| src.clone());
        let name = match by_source.get(&key) {
            Some(name) => name.clone(),
            None => {
                let base = src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let name = unique_name(&base, tag, &taken);
                std::fs::create_dir_all(&files_dir).map_err(|e| format!("create snapshot files dir: {e}"))?;
                std::fs::copy(&src, files_dir.join(&name))
                    .map_err(|e| format!("copy {} into snapshot: {e}", src.display()))?;
                taken.insert(name.clone());
                by_source.insert(key, name.clone());
                name
            }
        };
        *reference = format!("{rel_dir}/{name}");
        Ok(())
    };
    for band in project.bands.iter_mut() {
        if let Some(mf) = band.measurement_file.as_mut() {
            freeze(mf, "meas")?;
        }
        if let Some(ms) = band.settings.as_mut().and_then(|s| s.merge_source.as_mut()) {
            freeze(&mut ms.nf_path, "nf")?;
            freeze(&mut ms.ff_path, "ff")?;
        }
    }
    Ok(())
}

/// `base` if free, else `<tag>_<base>`, else `<tag>_<n>_<base>` (n = 2, 3, …).
fn unique_name(base: &str, tag: &str, taken: &std::collections::HashSet<String>) -> String {
    if !taken.contains(base) {
        return base.to_string();
    }
    let tagged = format!("{tag}_{base}");
    if !taken.contains(&tagged) {
        return tagged;
    }
    (2..).map(|n| format!("{tag}_{n}_{base}")).find(|c| !taken.contains(c)).expect("unbounded range")
}

#[tauri::command]
pub async fn list_snapshots(project_dir: String) -> Result<Vec<SnapshotEntry>, String> {
    let dir = validate_dir(&project_dir)?;
    let idx = read_index(&dir)?;
    Ok(idx.entries)
}

#[tauri::command]
pub async fn load_snapshot(project_dir: String, id: String) -> Result<ProjectFile, String> {
    let dir = validate_dir(&project_dir)?;
    validate_id(&id)?;
    let path = snapshots_dir(&dir).join(format!("{id}.pfproj"));
    if !path.exists() {
        return Err(format!("snapshot not found: {id}"));
    }
    let json = std::fs::read_to_string(&path).map_err(|e| format!("read snapshot: {e}"))?;
    let project: ProjectFile = serde_json::from_str(&json)
        .map_err(|e| format!("parse snapshot: {e}"))?;
    crate::project::validate_project(&project)?;
    info!("load_snapshot: {} ({} bands)", id, project.bands.len());
    Ok(project)
}

#[tauri::command]
pub async fn delete_snapshot(project_dir: String, id: String) -> Result<(), String> {
    let dir = validate_dir(&project_dir)?;
    validate_id(&id)?;
    let snap_path = snapshots_dir(&dir).join(format!("{id}.pfproj"));
    if snap_path.exists() {
        std::fs::remove_file(&snap_path).map_err(|e| format!("delete snapshot: {e}"))?;
    }
    let files_dir = snapshots_dir(&dir).join(&id);
    if files_dir.is_dir() {
        std::fs::remove_dir_all(&files_dir).map_err(|e| format!("delete snapshot files: {e}"))?;
    }
    // Update index, surfacing corruption so the UI can offer Rebuild.
    let _guard = index_lock();
    match read_index(&dir) {
        Ok(mut idx) => {
            idx.entries.retain(|e| e.id != id);
            write_index(&dir, &idx)?;
        }
        Err(e) if e == INDEX_CORRUPTED => {
            return Err(INDEX_CORRUPTED.to_string());
        }
        Err(e) => return Err(e),
    }
    info!("delete_snapshot: {}", id);
    Ok(())
}

#[tauri::command]
pub async fn rebuild_snapshot_index(project_dir: String) -> Result<u32, String> {
    let dir = validate_dir(&project_dir)?;
    let snap_dir = snapshots_dir(&dir);
    let _guard = index_lock();
    if !snap_dir.is_dir() {
        write_index(&dir, &SnapshotIndex::empty())?;
        return Ok(0);
    }
    let mut entries: Vec<SnapshotEntry> = Vec::new();
    for de in std::fs::read_dir(&snap_dir).map_err(|e| format!("read snapshots dir: {e}"))? {
        let de = de.map_err(|e| format!("dir entry: {e}"))?;
        let path = de.path();
        if path.extension().and_then(|s| s.to_str()) != Some("pfproj") {
            continue;
        }
        let id_from_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if validate_id(&id_from_name).is_err() {
            continue;
        }
        let txt = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let proj: ProjectFile = match serde_json::from_str(&txt) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let id = proj.snapshot_id.clone().unwrap_or_else(|| id_from_name.clone());
        if validate_id(&id).is_err() {
            continue;
        }
        let ts = proj.snapshot_ts.clone().unwrap_or_default();
        let description = proj.snapshot_description.clone().unwrap_or_default();
        let app_version = proj.snapshot_app_version.clone().unwrap_or_default();
        let file = format!("{id_from_name}.pfproj");
        entries.push(SnapshotEntry { id, ts, description, app_version, file });
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    let count = entries.len() as u32;
    write_index(&dir, &SnapshotIndex { version: 1, entries })?;
    info!("rebuild_snapshot_index: {} entries", count);
    Ok(count)
}

#[cfg(test)]
mod freeze_tests {
    use super::*;

    fn temp_project() -> PathBuf {
        // A counter, not fresh_id(): tests run in parallel and the 16-bit
        // nanosecond suffix collided between two tests started together.
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("pf_snap_{}_{}_{n}", std::process::id(), fresh_id()));
        std::fs::create_dir_all(dir.join("inbox")).unwrap();
        dir
    }

    fn band(name: &str, file: Option<&str>, nf: Option<&str>) -> crate::project::BandData {
        let src = format!(r#"{{"id":"{name}","name":"{name}","measurement":null,
            "settings":null,"target":{{"reference_level_db":80.0,"tilt_db_per_octave":0.0,
            "tilt_ref_freq":1000.0,"high_pass":null,"low_pass":null,"low_shelf":null,"high_shelf":null}},
            "target_enabled":true}}"#);
        let mut b: crate::project::BandData = serde_json::from_str(&src).unwrap();
        b.measurement_file = file.map(String::from);
        if let Some(nf) = nf {
            let settings = format!(r#"{{"smoothing":"off","delay_seconds":null,"distance_meters":null,
                "original_phase":null,"floor_bounce":null,
                "merge_source":{{"nf_path":"{nf}","ff_path":"inbox/ff.txt",
                "config":{{"splice_freq":300.0,"blend_octaves":1.0,"level_offset_db":null,"match_range":null}}}}}}"#);
            b.settings = serde_json::from_str(&settings).ok();
        }
        b
    }

    /// Re-importing a file with the same name must not rewrite an older
    /// snapshot: the snapshot owns a copy under snapshots/<id>/.
    #[test]
    fn snapshot_owns_copies_of_referenced_measurements() {
        let dir = temp_project();
        std::fs::write(dir.join("inbox/Woofer.txt"), "v1").unwrap();
        std::fs::write(dir.join("inbox/nf.txt"), "nf").unwrap();
        std::fs::write(dir.join("inbox/ff.txt"), "ff").unwrap();
        let mut project: ProjectFile = serde_json::from_str(r#"{
            "version": 2, "app_name": "PhaseForge", "bands": [],
            "active_band_id": "b1", "show_phase": true, "show_mag": true,
            "show_target": true, "next_band_num": 2
        }"#).unwrap();
        project.bands.push(band("b1", Some("inbox/Woofer.txt"), None));
        project.bands.push(band("b2", Some("inbox/Woofer.txt"), Some("inbox/nf.txt")));
        project.bands.push(band("b3", None, None));

        freeze_measurement_files(&dir, "snap1", &mut project).unwrap();

        assert_eq!(project.bands[0].measurement_file.as_deref(), Some("snapshots/snap1/Woofer.txt"));
        assert_eq!(project.bands[1].measurement_file.as_deref(), Some("snapshots/snap1/Woofer.txt"));
        let ms = project.bands[1].settings.as_ref().unwrap().merge_source.as_ref().unwrap();
        assert_eq!(ms.nf_path, "snapshots/snap1/nf.txt");
        assert_eq!(ms.ff_path, "snapshots/snap1/ff.txt");
        assert!(project.bands[2].measurement_file.is_none());

        // Overwrite the live inbox file — the snapshot copy must keep v1.
        std::fs::write(dir.join("inbox/Woofer.txt"), "v2").unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("snapshots/snap1/Woofer.txt")).unwrap(), "v1");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn empty_project() -> ProjectFile {
        serde_json::from_str(r#"{
            "version": 2, "app_name": "PhaseForge", "bands": [],
            "active_band_id": "b1", "show_phase": true, "show_mag": true,
            "show_target": true, "next_band_num": 2
        }"#).unwrap()
    }

    /// Audit 2026-10-23 2.3: three bands whose NF files share one name but
    /// live in different folders — each restores to ITS OWN file. The third
    /// used to be pointed at the second band's copy.
    #[test]
    fn same_named_measurements_of_different_bands_stay_apart() {
        let dir = temp_project();
        let mut project = empty_project();
        for (i, sub) in ["a", "b", "c"].iter().enumerate() {
            std::fs::create_dir_all(dir.join(sub)).unwrap();
            std::fs::write(dir.join(sub).join("nf.txt"), format!("nf-{sub}")).unwrap();
            project.bands.push(band(&format!("b{i}"), Some(&format!("{sub}/nf.txt")), Some(&format!("{sub}/nf.txt"))));
        }
        std::fs::write(dir.join("inbox/ff.txt"), "ff").unwrap();

        freeze_measurement_files(&dir, "snap2", &mut project).unwrap();

        let read = |r: &str| std::fs::read_to_string(dir.join(r)).unwrap();
        let mut names = std::collections::HashSet::new();
        for (b, sub) in project.bands.iter().zip(["a", "b", "c"]) {
            let mf = b.measurement_file.as_deref().unwrap();
            let ms = b.settings.as_ref().unwrap().merge_source.as_ref().unwrap();
            assert_eq!(read(mf), format!("nf-{sub}"), "band {sub}: measurement");
            assert_eq!(read(&ms.nf_path), format!("nf-{sub}"), "band {sub}: NF");
            assert_eq!(mf, ms.nf_path, "one source → one copy");
            assert_eq!(read(&ms.ff_path), "ff");
            names.insert(mf.to_string());
        }
        assert_eq!(names.len(), 3, "three sources need three files: {names:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Audit 2026-10-23 2.4: concurrent snapshots must all reach the index.
    #[test]
    fn concurrent_snapshots_all_reach_the_index() {
        let dir = temp_project();
        let dir_s = dir.to_string_lossy().into_owned();
        let handles: Vec<_> = (0..8).map(|t| {
            let dir_s = dir_s.clone();
            std::thread::spawn(move || {
                tauri::async_runtime::block_on(create_snapshot(
                    dir_s, format!("v{t}"), "test".into(), empty_project(),
                )).unwrap()
            })
        }).collect();
        let ids: std::collections::HashSet<String> = handles.into_iter().map(|h| h.join().unwrap().id).collect();
        assert_eq!(ids.len(), 8);
        let listed = tauri::async_runtime::block_on(list_snapshots(dir_s)).unwrap();
        assert_eq!(listed.len(), 8, "index lost entries: {listed:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// b141.122: creating a version does not overwrite a corrupt index, and
    /// Rebuild still finds the versions on disk.
    #[test]
    fn corrupt_index_is_refused_and_left_alone() {
        let dir = temp_project();
        let dir_s = dir.to_string_lossy().into_owned();
        let create = |d: &str| tauri::async_runtime::block_on(create_snapshot(
            dir_s.clone(), d.into(), "test".into(), empty_project(),
        ));
        create("first").unwrap();
        let index = dir.join("snapshots/index.json");
        std::fs::write(&index, "{ not json").unwrap();
        let on_disk = || std::fs::read_dir(dir.join("snapshots")).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "pfproj")).count();
        let before = on_disk();
        assert_eq!(create("second").unwrap_err(), INDEX_CORRUPTED);
        assert_eq!(std::fs::read_to_string(&index).unwrap(), "{ not json");
        assert_eq!(on_disk(), before, "a refused version wrote a file");
        let rebuilt = tauri::async_runtime::block_on(rebuild_snapshot_index(dir_s.clone())).unwrap();
        assert_eq!(rebuilt as usize, before);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Audit 2026-10-23 2.7: a failed copy leaves neither the .pfproj nor the
    /// files folder behind.
    #[cfg(unix)]
    #[test]
    fn failed_snapshot_leaves_nothing_behind() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_project();
        let a = dir.join("inbox/a.txt");
        let b = dir.join("inbox/b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();
        std::fs::set_permissions(&b, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::read(&b).is_ok() {
            // Running as root: permissions do not stop the copy.
            std::fs::remove_dir_all(&dir).ok();
            return;
        }
        let mut project = empty_project();
        project.bands.push(band("b1", Some("inbox/a.txt"), None));
        project.bands.push(band("b2", Some("inbox/b.txt"), None));
        let r = tauri::async_runtime::block_on(create_snapshot(
            dir.to_string_lossy().into_owned(), "v".into(), "test".into(), project,
        ));
        assert!(r.is_err(), "{r:?}");
        let left: Vec<_> = std::fs::read_dir(dir.join("snapshots")).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.file_name()).collect();
        assert!(left.is_empty(), "left behind: {left:?}");
        std::fs::set_permissions(&b, std::fs::Permissions::from_mode(0o644)).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}
