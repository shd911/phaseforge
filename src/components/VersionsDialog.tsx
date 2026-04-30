import { createSignal, Show, For, onCleanup } from "solid-js";
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

/** Read-only signal so global shortcuts can suspend while the dialog is open. */
export const isVersionsDialogOpen = _open;

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
      setStatusMsg(`Не удалось сохранить: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleRebuild() {
    setBusy(true);
    setStatusMsg(null);
    try {
      const n = await rebuildSnapshotIndex();
      setStatusMsg(`Индекс перестроен: найдено ${n} версий.`);
    } catch (e) {
      setStatusMsg(`Перестроить не удалось: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(s: SnapshotEntry) {
    const ok = window.confirm(`Удалить версию «${s.description}»? Это действие необратимо.`);
    if (!ok) return;
    setBusy(true);
    setStatusMsg(null);
    try {
      await deleteSnapshotById(s.id);
    } catch (e) {
      setStatusMsg(`Не удалось удалить: ${e}`);
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
      const auto = `Авто-снимок · ${stamp} · перед загрузкой «${s.description}»`;
      await restoreSnapshot(s.id, savePrevious, auto);
      close();
    } catch (e) {
      setStatusMsg(`Восстановление не удалось: ${e}`);
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

  // ESC closes the dialog.
  const onKey = (e: KeyboardEvent) => {
    if (!_open()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      if (pendingRestore()) setPendingRestore(null);
      else if (creating()) { setCreating(false); setDraft(""); }
      else close();
    }
  };
  window.addEventListener("keydown", onKey);
  onCleanup(() => window.removeEventListener("keydown", onKey));

  return (
    <Show when={_open()}>
      <div class="pn-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
        <div class="pn-dialog" style={{ "min-width": "480px", "max-width": "640px" }}>
          <div class="pn-title">Версии проекта</div>

          <Show when={!creating()}>
            <div class="pn-buttons" style={{ "justify-content": "flex-start", "margin-bottom": "12px" }}>
              <button
                class="dlg-btn dlg-btn-primary"
                onClick={() => { setCreating(true); setDraft(""); setStatusMsg(null); }}
                disabled={busy()}
                title="Создать версию текущего состояния с описанием"
              >+ Сохранить версию</button>
            </div>
          </Show>

          <Show when={creating()}>
            <div style={{ "margin-bottom": "12px" }}>
              <label class="pn-label">Описание версии</label>
              <textarea
                class="pn-input"
                style={{ "min-height": "60px", "width": "100%", "resize": "vertical" }}
                placeholder="Что изменилось?"
                value={draft()}
                onInput={(e) => setDraft(e.currentTarget.value)}
                ref={(el) => requestAnimationFrame(() => el.focus())}
              />
              <div class="pn-buttons" style={{ "margin-top": "8px" }}>
                <button class="dlg-btn" onClick={() => { setCreating(false); setDraft(""); }} disabled={busy()}>Отмена</button>
                <button
                  class="dlg-btn dlg-btn-primary"
                  onClick={handleCreate}
                  disabled={busy() || !draft().trim()}
                >Сохранить</button>
              </div>
            </div>
          </Show>

          <Show when={snapshotsError() === "INDEX_CORRUPTED"}>
            <div style={{ padding: "12px", border: "1px solid #c66", "border-radius": "4px", "margin-bottom": "12px" }}>
              <div style={{ "margin-bottom": "8px" }}>Индекс версий повреждён.</div>
              <button class="dlg-btn" onClick={handleRebuild} disabled={busy()}>Перестроить индекс</button>
            </div>
          </Show>

          <Show when={snapshotsError() && snapshotsError() !== "INDEX_CORRUPTED"}>
            <div style={{ padding: "8px", color: "#c66", "margin-bottom": "12px" }}>
              Ошибка: {snapshotsError()}
            </div>
          </Show>

          <Show when={!snapshotsError()}>
            <Show
              when={snapshotsList().length > 0}
              fallback={<div style={{ padding: "12px", color: "#888" }}>Версий ещё нет.</div>}
            >
              <div class="versions-list" style={{ "max-height": "320px", "overflow-y": "auto", border: "1px solid #444", "border-radius": "4px" }}>
                <For each={snapshotsList()}>
                  {(s) => (
                    <div style={{ padding: "8px 12px", "border-bottom": "1px solid #333", display: "flex", "align-items": "center", gap: "8px" }}>
                      <div style={{ flex: 1, "min-width": 0 }}>
                        <div style={{ "font-size": "12px", color: "#888" }}>
                          {formatTs(s.ts)} · {s.app_version || "—"}
                        </div>
                        <div
                          style={{
                            "white-space": "nowrap",
                            "overflow": "hidden",
                            "text-overflow": "ellipsis",
                            "font-style": s.description ? "normal" : "italic",
                            "color": s.description ? undefined : "#888",
                          }}
                          title={s.description || "Описание не сохранилось при перестроении индекса"}
                        >
                          {s.description || "(описание утеряно)"}
                        </div>
                      </div>
                      <button
                        class="dlg-btn"
                        onClick={() => startRestore(s)}
                        disabled={busy()}
                        title="Загрузить состояние этой версии в текущий проект"
                      >Восстановить</button>
                      <button
                        class="dlg-btn"
                        onClick={() => handleDelete(s)}
                        disabled={busy()}
                        title="Удалить эту версию (файл будет удалён с диска)"
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
            <button class="dlg-btn" onClick={close} disabled={busy()}>Закрыть</button>
          </div>

          <Show when={pendingRestore()}>
            <div class="pn-overlay" style={{ background: "rgba(0,0,0,0.5)" }}>
              <div class="pn-dialog" style={{ "max-width": "420px" }}>
                <div class="pn-title">Сохранить текущее состояние?</div>
                <div class="pn-label" style={{ "margin-bottom": "16px" }}>
                  Сохранить текущее состояние как версию перед восстановлением «{pendingRestore()!.description}»?
                </div>
                <div class="pn-buttons">
                  <button class="dlg-btn" onClick={() => pendingResolve("cancel")}>Отмена</button>
                  <button class="dlg-btn" onClick={() => pendingResolve("no")}>Нет</button>
                  <button class="dlg-btn dlg-btn-primary" onClick={() => pendingResolve("yes")}>Да</button>
                </div>
              </div>
            </div>
          </Show>
        </div>
      </div>
    </Show>
  );
}
