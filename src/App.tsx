import "./App.css";
import { createSignal, createEffect, onCleanup, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
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
    promptVisible() || unsavedDialogVisible() || isVersionsDialogOpen();

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

  const infoText = () => {
    if (isSum()) return "";
    return "";
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
          <span class="top-project-dirty">(modified)</span>
        </Show>
        <div class="top-sep" />
        <div class="strategy-toggle">
          <button
            class={`strategy-btn ${!exportHybridPhase() ? "active" : ""}`}
            onClick={() => { if (exportHybridPhase()) { setExportHybridPhase(false); handleOptimizeAll(); } }}
          >Standard</button>
          <button
            class={`strategy-btn ${exportHybridPhase() ? "active" : ""}`}
            onClick={() => { if (!exportHybridPhase()) { setExportHybridPhase(true); handleOptimizeAll(); } }}
          >Hybrid</button>
        </div>
        <button
          class="tb-btn"
          onClick={openFirSettings}
          title="Optimization settings"
        >Settings</button>
        <button
          class="tb-btn primary"
          onClick={handleOptimizeAll}
          disabled={computing()}
          title="Optimize PEQ for all bands"
        >{computing() ? "..." : "Optimize All"}</button>
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

        {/* Right panel: Target + PEQ — always visible (hidden in SUM) */}
        <Show when={!isSum()}>
          <div class="right-panel">
            <ControlPanel rightPanel={true} />
          </div>
        </Show>
      </div>

      {/* Status bar */}
      <footer class="status-bar">
        {appState.bands.length} band{appState.bands.length > 1 ? "s" : ""}
        {isSum() ? " — SUM view" : ""}
      </footer>

      {/* Modal dialogs */}
      <WelcomeDialog />
      <ProjectNameDialog />
      <UnsavedChangesDialog />
      <VersionsDialog />
      <CrossoverDialog />
      <FirSettingsDialog />
      <Toasts />
    </div>
  );
}

export default App;
