import { invoke } from "@tauri-apps/api/core";
import { createSignal } from "solid-js";
import {
  projectDir,
  projectName,
  buildProjectData,
  restoreState,
  type ProjectFile,
} from "./project-io";
import { setIsDirty } from "../stores/bands";
import { clearHistory } from "../stores/history";
import { APP_VERSION } from "./version";

export interface SnapshotEntry {
  id: string;
  ts: string;
  description: string;
  app_version: string;
  file: string;
}

/** Last error from list_snapshots, exposed so the dialog can offer Rebuild. */
export const [snapshotsList, setSnapshotsList] = createSignal<SnapshotEntry[]>([]);
export const [snapshotsError, setSnapshotsError] = createSignal<string | null>(null);

export async function refreshSnapshots(): Promise<void> {
  const dir = projectDir();
  if (!dir) {
    setSnapshotsList([]);
    setSnapshotsError(null);
    return;
  }
  try {
    const list = await invoke<SnapshotEntry[]>("list_snapshots", { projectDir: dir });
    // Newest first. Sort by ts when available; fall back to id (lex-sortable).
    list.sort((a, b) => {
      const at = a.ts || a.id;
      const bt = b.ts || b.id;
      return at < bt ? 1 : at > bt ? -1 : 0;
    });
    setSnapshotsList(list);
    setSnapshotsError(null);
  } catch (e) {
    const msg = String(e);
    setSnapshotsError(msg);
    setSnapshotsList([]);
  }
}

export async function createSnapshotForCurrentProject(
  description: string,
): Promise<SnapshotEntry> {
  const dir = projectDir();
  if (!dir) throw new Error("Project must be saved before creating versions");
  const trimmed = description.trim();
  if (!trimmed) throw new Error("Description must not be empty");
  const project = buildProjectData();
  const entry = await invoke<SnapshotEntry>("create_snapshot", {
    projectDir: dir,
    description: trimmed,
    appVersion: APP_VERSION,
    project,
  });
  await refreshSnapshots();
  return entry;
}

export async function deleteSnapshotById(id: string): Promise<void> {
  const dir = projectDir();
  if (!dir) return;
  await invoke("delete_snapshot", { projectDir: dir, id });
  await refreshSnapshots();
}

export async function rebuildSnapshotIndex(): Promise<number> {
  const dir = projectDir();
  if (!dir) return 0;
  const count = await invoke<number>("rebuild_snapshot_index", { projectDir: dir });
  await refreshSnapshots();
  return count;
}

export async function restoreSnapshot(
  id: string,
  saveCurrentFirst: boolean,
  autoDescription: string,
): Promise<void> {
  const dir = projectDir();
  if (!dir) throw new Error("No project directory");
  if (saveCurrentFirst) {
    await createSnapshotForCurrentProject(autoDescription);
  }
  const project = await invoke<ProjectFile>("load_snapshot", { projectDir: dir, id });
  await restoreState(project, dir);
  // Live .pfproj on disk is unchanged — restored state diverges from it,
  // so flag dirty and clear session-undo (restore is not part of that stack).
  clearHistory();
  setIsDirty(true);
}
