import "./App.css";
import { createSignal, createEffect, onCleanup, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ask } from "@tauri-apps/plugin-dialog";
import {
  appState,
  activeBand,
  isSum,
  isDirty,
  exportHybridPhase,
  setExportHybridPhase,
} from "./stores/bands";
import { handleOptimizePeq, handleOptimizeAll, computing } from "./stores/peq-optimize";
import {
  saveProject,
  saveProjectAs,
  loadProject,
  newProject,
  currentProjectPath,
  projectName,
  projectDir,
  confirmIfDirty,
  promptVisible,
  unsavedDialogVisible,
} from "./lib/project-io";
import { undo, redo, canUndo, canRedo, lastUndoLabel, lastRedoLabel } from "./stores/history";
import { peqDragging } from "./stores/bands";
import FileMenu from "./components/FileMenu";
import FrequencyPlot from "./components/FrequencyPlot";
import ControlPanel from "./components/ControlPanel";
import BandTabs from "./components/BandTabs";
import ProjectNameDialog from "./components/ProjectNameDialog";
import UnsavedChangesDialog from "./components/UnsavedChangesDialog";
import VersionsDialog, { openVersionsDialog, isVersionsDialogOpen } from "./components/VersionsDialog";
import MeasurementAnalysisDialog, { isAnalysisDialogOpen } from "./components/MeasurementAnalysisDialog";
import StalePeqExportDialog, { isStaleConfirmDialogOpen } from "./components/StalePeqExportDialog";
import HighQWarningPopup, { isHighQPopupOpen } from "./components/HighQWarningPopup";
import Toasts from "./components/Toasts";
import { showToast } from "./lib/toast";
import WelcomeDialog from "./components/WelcomeDialog";
import CrossoverDialog from "./components/CrossoverDialog";
import FirSettingsDialog from "./components/FirSettingsDialog";
import { openFirSettings } from "./components/FirSettingsDialog";
// PeqSidebar removed — PEQ controls now in ControlPanel PEQ tab
import { activeTab } from "./stores/bands";

// Глобальный сигнал для авто-FIT при импорте замера
export const [needAutoFit, setNeedAutoFit] = createSignal(false);

function App() {
  // Centralized global shortcuts (Cmd on macOS, Ctrl elsewhere). Suspended
  // when focus is in a text-editing surface or any modal dialog is open.
  const isEditableTarget = (t: EventTarget | null): boolean => {
    if (!(t instanceof HTMLElement)) return false;
    const tag = t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    return t.isContentEditable;
  };
  const isModalOpen = (): boolean =>
    promptVisible() || unsavedDialogVisible() || isVersionsDialogOpen()
    || isAnalysisDialogOpen() || isStaleConfirmDialogOpen() || isHighQPopupOpen();

  const handleKeys = (e: KeyboardEvent) => {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (isEditableTarget(e.target)) return;
    if (isModalOpen()) return;
    const k = e.key.toLowerCase();
    if (k === "z") {
      if (peqDragging()) return; // mid-drag — let the gesture finish first
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
      return;
    }
    if (e.shiftKey && k === "v") {
      e.preventDefault();
      if (projectDir() !== null) openVersionsDialog();
      else showToast("Сохраните проект через Save As, чтобы создавать версии", "info");
      return;
    }
    if (k === "s") {
      e.preventDefault();
      if (e.shiftKey) saveProjectAs();
      else saveProject();
    } else if (k === "o") {
      e.preventDefault();
      loadProject();
    } else if (k === "n") {
      e.preventDefault();
      newProject();
    }
  };
  window.addEventListener("keydown", handleKeys);
  onCleanup(() => window.removeEventListener("keydown", handleKeys));

  // Window close confirmation: backend prevents close, emits this event;
  // we delegate to confirmIfDirty (same flow as New/Open), then ack the close.
  const unlistenClose = listen("request-close-confirm", async () => {
    if (await confirmIfDirty()) {
      await invoke("close_window_now").catch((e) => console.error("close_window_now:", e));
    }
  });
  onCleanup(() => { unlistenClose.then((u) => u()).catch(() => {}); });

  createEffect(() => {
    const pName = projectName();
    const path = currentProjectPath();
    const dirty = isDirty();
    const name = pName ?? (path ? path.split("/").pop() : "Untitled");
    document.title = `PhaseForge — ${name}${dirty ? " •" : ""}`;
  });

  // Plot split removed — all modes now in FrequencyPlot tabs

  // Strategy switch re-optimizes ALL bands and overwrites manual PEQ edits.
  // Confirm first when any band already has PEQ so a stray click can't wipe
  // hand-tuning across the whole project.
  const switchStrategy = async (toHybrid: boolean) => {
    if (exportHybridPhase() === toHybrid) return;
    const hasPeq = appState.bands.some((b) => (b.peqBands?.length ?? 0) > 0);
    if (hasPeq) {
      const ok = await ask(
        "Переключение режима переоптимизирует все бэнды и перезапишет ручные правки PEQ. Продолжить?",
        { title: toHybrid ? "Перейти в режим Hybrid" : "Перейти в режим Standard", kind: "warning" },
      );
      if (!ok) return;
    }
    setExportHybridPhase(toHybrid);
    handleOptimizeAll();
  };

  // All plot modes now handled by FrequencyPlot tabs (freq/ir/step/gd/export)

  return (
    <div class="app">
      {/* Top Bar */}
      <div class="top-bar">
        <img src="/logo-icon.png" class="top-logo-icon" alt="" />
        <span class="top-logo">PhaseForge</span>
        <div class="top-sep" />
        <FileMenu />
        <button
          class="tb-btn"
          onClick={() => undo()}
          disabled={!canUndo()}
          title={canUndo() ? `Откатить: ${lastUndoLabel()}` : "Откатить (нет действий)"}
        >↶</button>
        <button
          class="tb-btn"
          onClick={() => redo()}
          disabled={!canRedo()}
          title={canRedo() ? `Повторить: ${lastRedoLabel()}` : "Повторить (нет действий)"}
        >↷</button>
        <span class="top-project-name" title={currentProjectPath() ?? "Untitled"}>
          {projectName() ?? (currentProjectPath()
            ? currentProjectPath()!.split("/").pop()
            : "Untitled")}
        </span>
        <Show when={isDirty()}>
          <span class="top-project-dirty">(изменён)</span>
        </Show>
        <div class="top-sep" />
        <div class="strategy-toggle">
          <button
            class={`strategy-btn ${!exportHybridPhase() ? "active" : ""}`}
            onClick={() => switchStrategy(false)}
            title="Стандартная коррекция (мин./лин. фаза). Переключение переоптимизирует все бэнды."
          >Standard</button>
          <button
            class={`strategy-btn ${exportHybridPhase() ? "active" : ""}`}
            onClick={() => switchStrategy(true)}
            title="Гибридная фаза экспорта. Переключение переоптимизирует все бэнды."
          >Hybrid</button>
        </div>
        <button
          class="tb-btn"
          onClick={openFirSettings}
          title="Настройки оптимизации и FIR"
        >Настройки</button>
        <button
          class="tb-btn primary"
          onClick={handleOptimizeAll}
          disabled={computing()}
          title="Оптимизировать PEQ для всех бэндов"
        >{computing() ? "..." : "Оптимизировать все"}</button>
        <div class="top-sep" />
      </div>

      {/* Band Tabs */}
      <BandTabs />

      {/* Main content row: plots+panel on left, PeqSidebar on right */}
      <div class="main-content-row">
        <div class="main-content-col">
          {/* Plot area — dual graph when not SUM */}
          <main class="plot-area">
            <div class="freq-plot-area" style={{ flex: "1" }}>
              <FrequencyPlot />
            </div>
          </main>

          {/* Bottom panel removed (b126) — controls moved to plot toolbar */}
        </div>

        {/* Right panel: Target + PEQ — editing surface (replaced by a hint in
            SUM, where the view is a read-only overview). */}
        <Show
          when={!isSum()}
          fallback={
            <div class="right-panel">
              <div class="sum-hint">
                <div class="sum-hint-title">Режим «Сумма»</div>
                <p>Это обзор суммарного отклика всех бэндов — только для просмотра.</p>
                <p>Чтобы редактировать цель и PEQ, выберите вкладку бэнда сверху.</p>
                <p>Двойной клик по линии раздела на графике — настройка кроссовера.</p>
              </div>
            </div>
          }
        >
          <div class="right-panel">
            <ControlPanel rightPanel={true} />
          </div>
        </Show>
      </div>

      {/* Status bar */}
      <footer class="status-bar">
        {appState.bands.length} бэнд{appState.bands.length === 1 ? "" : "а"}
        {isSum() ? " — режим Сумма" : ""}
      </footer>

      {/* Modal dialogs */}
      <WelcomeDialog />
      <ProjectNameDialog />
      <UnsavedChangesDialog />
      <VersionsDialog />
      <MeasurementAnalysisDialog />
      <StalePeqExportDialog />
      <HighQWarningPopup />
      <CrossoverDialog />
      <FirSettingsDialog />
      <Toasts />
    </div>
  );
}

export default App;
