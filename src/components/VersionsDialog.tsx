import { createSignal, Show, For } from "solid-js";
import Modal from "./Modal";
import { pluralRu } from "../lib/plural";
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
      setStatusMsg(`Индекс перестроен: ${pluralRu(n, "найдена", "найдено", "найдено")} ${n} ${pluralRu(n, "версия", "версии", "версий")}.`);
    } catch (e) {
      setStatusMsg(`Перестроить не удалось: ${e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(s: SnapshotEntry) {
    const ok = await ask(`Удалить версию «${s.description}»? Это действие необратимо.`, {
      title: "Удаление версии", kind: "warning", okLabel: "Удалить", cancelLabel: "Отмена",
    });
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
          <div class="pn-title" id="versions-dlg-title">Версии проекта</div>

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
                maxlength="1000"
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
              fallback={<div style={{ padding: "12px", color: "var(--text-secondary)" }}>Версий ещё нет.</div>}
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
                        aria-label="Удалить версию"
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
            <Modal overlayClass="pn-overlay pn-overlay-nested" dialogClass="pn-dialog"
              dialogStyle={{ "max-width": "420px" }} titleId="versions-restore-title"
              onEscape={() => pendingResolve("cancel")}>
                <div class="pn-title" id="versions-restore-title">Сохранить текущее состояние?</div>
                <div class="pn-label" style={{ "margin-bottom": "16px" }}>
                  Сохранить текущее состояние как версию перед восстановлением «{pendingRestore()!.description}»?
                </div>
                <div class="pn-buttons">
                  <button class="dlg-btn" onClick={() => pendingResolve("cancel")}>Отмена</button>
                  <button class="dlg-btn" onClick={() => pendingResolve("no")}>Нет</button>
                  <button class="dlg-btn dlg-btn-primary" onClick={() => pendingResolve("yes")}>Да</button>
                </div>
            </Modal>
          </Show>
      </Modal>
    </Show>
  );
}
