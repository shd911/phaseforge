import { createSignal, Show, For } from "solid-js";
import Modal from "./Modal";
import { L, plural } from "../lib/i18n";
import { ask } from "@tauri-apps/plugin-dialog";
import {
  snapshotsList,
  snapshotsError,
  refreshSnapshots,
  createSnapshotForCurrentProject,
  restoreSnapshot,
  deleteSnapshotById,
  rebuildSnapshotIndex,
  type SnapshotEntry,
} from "../lib/snapshots";
import { isDirty } from "../stores/bands";

const [_open, setOpen] = createSignal(false);

/** Public API — call from FileMenu or anywhere else. */
export function openVersionsDialog(): void {
  setOpen(true);
  void refreshSnapshots();
}


type RestoreChoice = "yes" | "no" | "cancel";

function formatTs(ts: string): string {
  if (!ts) return "—";
  // ts is ISO 8601 UTC. Show local time in compact format.
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export default function VersionsDialog() {
  const [creating, setCreating] = createSignal(false);
  const [draft, setDraft] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [statusMsg, setStatusMsg] = createSignal<string | null>(null);
  const [pendingRestore, setPendingRestore] = createSignal<SnapshotEntry | null>(null);

  function close() {
    setOpen(false);
    setCreating(false);
    setDraft("");
    setStatusMsg(null);
    setPendingRestore(null);
  }

  async function handleCreate() {
    const desc = draft().trim();
    if (!desc) return;
    setBusy(true);
    setStatusMsg(null);
    try {
      await createSnapshotForCurrentProject(desc);
      setCreating(false);
      setDraft("");
    } catch (e) {
      // b141.122: the index may have gone bad since the dialog opened.
      if (`${e}`.includes("INDEX_CORRUPTED")) {
        setCreating(false);
        void refreshSnapshots();
      } else setStatusMsg(L(`Не удалось сохранить: ${e}`, `Could not save: ${e}`));
    } finally {
      setBusy(false);
    }
  }

  async function handleRebuild() {
    setBusy(true);
    setStatusMsg(null);
    try {
      const n = await rebuildSnapshotIndex();
      setStatusMsg(L(`Индекс перестроен: ${plural(n, ["найдена", "найдено", "найдено"], ["found", "found"])} ${n} ${plural(n, ["версия", "версии", "версий"], ["version", "versions"])}.`,
        `Index rebuilt: ${n} ${plural(n, ["версия", "версии", "версий"], ["version", "versions"])} found.`));
    } catch (e) {
      setStatusMsg(L(`Перестроить не удалось: ${e}`, `Rebuild failed: ${e}`));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(s: SnapshotEntry) {
    const ok = await ask(L(`Удалить версию «${s.description}»? Это действие необратимо.`, `Delete version “${s.description}”? This cannot be undone.`), {
      title: L("Удаление версии", "Delete version"), kind: "warning", okLabel: L("Удалить", "Delete"), cancelLabel: L("Отмена", "Cancel"),
    });
    if (!ok) return;
    setBusy(true);
    setStatusMsg(null);
    try {
      await deleteSnapshotById(s.id);
    } catch (e) {
      setStatusMsg(L(`Не удалось удалить: ${e}`, `Could not delete: ${e}`));
    } finally {
      setBusy(false);
    }
  }

  async function startRestore(s: SnapshotEntry) {
    if (isDirty()) {
      setPendingRestore(s);
      return;
    }
    await doRestore(s, false);
  }

  async function doRestore(s: SnapshotEntry, savePrevious: boolean) {
    setPendingRestore(null);
    setBusy(true);
    setStatusMsg(null);
    try {
      const now = new Date();
      const stamp = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
      const auto = L(`Авто-снимок · ${stamp} · перед загрузкой «${s.description}»`, `Auto-snapshot · ${stamp} · before loading “${s.description}”`);
      await restoreSnapshot(s.id, savePrevious, auto);
      close();
    } catch (e) {
      setStatusMsg(L(`Восстановление не удалось: ${e}`, `Restore failed: ${e}`));
    } finally {
      setBusy(false);
    }
  }

  function pendingResolve(choice: RestoreChoice) {
    const s = pendingRestore();
    if (!s) return;
    if (choice === "cancel") {
      setPendingRestore(null);
      return;
    }
    void doRestore(s, choice === "yes");
  }

  // ESC cancels the description draft first, then closes the dialog. The
  // restore question is its own modal on top and handles its own Escape.
  const onEscape = () => {
    if (creating()) { setCreating(false); setDraft(""); }
    else close();
  };

  return (
    <Show when={_open()}>
      <Modal overlayClass="pn-overlay" dialogClass="pn-dialog"
        dialogStyle={{ "min-width": "480px", "max-width": "640px" }} titleId="versions-dlg-title"
        onEscape={onEscape} onBackdrop={close}>
          <div class="pn-title" id="versions-dlg-title">{L("Версии проекта", "Project versions")}</div>

          <Show when={!creating()}>
            <div class="pn-buttons" style={{ "justify-content": "flex-start", "margin-bottom": "12px" }}>
              <button
                class="dlg-btn dlg-btn-primary"
                onClick={() => { setCreating(true); setDraft(""); setStatusMsg(null); }}
                disabled={busy() || snapshotsError() === "INDEX_CORRUPTED"}
                title={snapshotsError() === "INDEX_CORRUPTED"
                  ? L("Сначала перестройте индекс версий", "Rebuild the version index first")
                  : L("Создать версию текущего состояния с описанием", "Save the current state as a described version")}
              >{L("+ Сохранить версию", "+ Save version")}</button>
            </div>
          </Show>

          <Show when={creating()}>
            <div style={{ "margin-bottom": "12px" }}>
              <label class="pn-label">{L("Описание версии", "Version description")}</label>
              <textarea
                class="pn-input"
                style={{ "min-height": "60px", "width": "100%", "resize": "vertical" }}
                placeholder={L("Что изменилось?", "What changed?")}
                maxlength="1000"
                value={draft()}
                onInput={(e) => setDraft(e.currentTarget.value)}
                ref={(el) => requestAnimationFrame(() => el.focus())}
              />
              <div class="pn-buttons" style={{ "margin-top": "8px" }}>
                <button class="dlg-btn" onClick={() => { setCreating(false); setDraft(""); }} disabled={busy()}>{L("Отмена", "Cancel")}</button>
                <button
                  class="dlg-btn dlg-btn-primary"
                  onClick={handleCreate}
                  disabled={busy() || !draft().trim()}
                >{L("Сохранить", "Save")}</button>
              </div>
            </div>
          </Show>

          <Show when={snapshotsError() === "INDEX_CORRUPTED"}>
            <div style={{ padding: "12px", border: "1px solid #c66", "border-radius": "4px", "margin-bottom": "12px" }}>
              <div style={{ "margin-bottom": "8px" }}>{L("Индекс версий повреждён.", "The version index is corrupted.")}</div>
              <button class="dlg-btn" onClick={handleRebuild} disabled={busy()}>{L("Перестроить индекс", "Rebuild index")}</button>
            </div>
          </Show>

          <Show when={snapshotsError() && snapshotsError() !== "INDEX_CORRUPTED"}>
            <div style={{ padding: "8px", color: "#c66", "margin-bottom": "12px" }}>
              {L("Ошибка:", "Error:")} {snapshotsError()}
            </div>
          </Show>

          <Show when={!snapshotsError()}>
            <Show
              when={snapshotsList().length > 0}
              fallback={<div style={{ padding: "12px", color: "var(--text-secondary)" }}>{L("Версий ещё нет.", "No versions yet.")}</div>}
            >
              <div class="versions-list" style={{ "max-height": "320px", "overflow-y": "auto", border: "1px solid #444", "border-radius": "4px" }}>
                <For each={snapshotsList()}>
                  {(s) => (
                    <div style={{ padding: "8px 12px", "border-bottom": "1px solid #333", display: "flex", "align-items": "center", gap: "8px" }}>
                      <div style={{ flex: 1, "min-width": 0 }}>
                        <div style={{ "font-size": "12px", color: "var(--text-secondary)" }}>
                          {formatTs(s.ts)} · {s.app_version || "—"}
                        </div>
                        <div
                          style={{
                            "white-space": "nowrap",
                            "overflow": "hidden",
                            "text-overflow": "ellipsis",
                            "font-style": s.description ? "normal" : "italic",
                            "color": s.description ? undefined : "var(--text-secondary)",
                          }}
                          title={s.description || L("Описание не сохранилось при перестроении индекса", "Description lost when the index was rebuilt")}
                        >
                          {s.description || L("(описание утеряно)", "(description lost)")}
                        </div>
                      </div>
                      <button
                        class="dlg-btn"
                        onClick={() => startRestore(s)}
                        disabled={busy()}
                        title={L("Загрузить состояние этой версии в текущий проект", "Load this version's state into the current project")}
                      >{L("Восстановить", "Restore")}</button>
                      <button
                        class="dlg-btn"
                        onClick={() => handleDelete(s)}
                        disabled={busy()}
                        aria-label={L("Удалить версию", "Delete version")}
                        title={L("Удалить эту версию (файл будет удалён с диска)", "Delete this version (the file is removed from disk)")}
                      >×</button>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </Show>

          <Show when={statusMsg()}>
            <div style={{ "margin-top": "12px", color: "#aaa", "font-size": "12px" }}>{statusMsg()}</div>
          </Show>

          <div class="pn-buttons" style={{ "margin-top": "16px" }}>
            <button class="dlg-btn" onClick={close} disabled={busy()}>{L("Закрыть", "Close")}</button>
          </div>

          <Show when={pendingRestore()}>
            <Modal overlayClass="pn-overlay pn-overlay-nested" dialogClass="pn-dialog"
              dialogStyle={{ "max-width": "420px" }} titleId="versions-restore-title"
              onEscape={() => pendingResolve("cancel")}>
                <div class="pn-title" id="versions-restore-title">{L("Сохранить текущее состояние?", "Save the current state?")}</div>
                <div class="pn-label" style={{ "margin-bottom": "16px" }}>
                  {L(`Сохранить текущее состояние как версию перед восстановлением «${pendingRestore()!.description}»?`,
                    `Save the current state as a version before restoring “${pendingRestore()!.description}”?`)}
                </div>
                <div class="pn-buttons">
                  <button class="dlg-btn" onClick={() => pendingResolve("cancel")}>{L("Отмена", "Cancel")}</button>
                  <button class="dlg-btn" onClick={() => pendingResolve("no")}>{L("Нет", "No")}</button>
                  <button class="dlg-btn dlg-btn-primary" onClick={() => pendingResolve("yes")}>{L("Да", "Yes")}</button>
                </div>
            </Modal>
          </Show>
      </Modal>
    </Show>
  );
}
