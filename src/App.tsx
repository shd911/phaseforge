import "./App.css";
import { createSignal, createEffect, onCleanup, Show } from "solid-js";
import {
  appState,
  activeBand,
  isSum,
  isDirty,
  exportHybridPhase,
  setExportHybridPhase,
} from "./stores/bands";
import { handleOptimizePeq, handleOptimizeAll, computing } from "./stores/peq-optimize";
import { saveProject, saveProjectAs, loadProject, newProject, currentProjectPath, projectName } from "./lib/project-io";
import FileMenu from "./components/FileMenu";
import FrequencyPlot from "./components/FrequencyPlot";
import ControlPanel from "./components/ControlPanel";
import BandTabs from "./components/BandTabs";
import ProjectNameDialog from "./components/ProjectNameDialog";
import WelcomeDialog from "./components/WelcomeDialog";
import CrossoverDialog from "./components/CrossoverDialog";
import FirSettingsDialog from "./components/FirSettingsDialog";
import { openFirSettings } from "./components/FirSettingsDialog";
// PeqSidebar removed — PEQ controls now in ControlPanel PEQ tab
import { activeTab } from "./stores/bands";

// Глобальный сигнал для авто-FIT при импорте замера
export const [needAutoFit, setNeedAutoFit] = createSignal(false);

function App() {
  // Keyboard shortcuts: Cmd+N, Cmd+S, Cmd+Shift+S, Cmd+O
  const handleKeys = (e: KeyboardEvent) => {
    if (!(e.metaKey || e.ctrlKey)) return;
    if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      if (e.shiftKey) saveProjectAs();
      else saveProject();
    } else if (e.key === "o" || e.key === "O") {
      e.preventDefault();
      loadProject();
    } else if (e.key === "n" || e.key === "N") {
      e.preventDefault();
      newProject();
    }
  };
  window.addEventListener("keydown", handleKeys);
  onCleanup(() => window.removeEventListener("keydown", handleKeys));

  // Reactive window title: "PhaseForge — ProjectName *"
  createEffect(() => {
    const pName = projectName();
    const path = currentProjectPath();
    const dirty = isDirty();
    // Prefer project name, fallback to filename, fallback to "Untitled"
    const name = pName ?? (path ? path.split("/").pop() : "Untitled");
    document.title = `PhaseForge — ${name}${dirty ? " *" : ""}`;
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

          {/* Resize handle + Control panel — hidden on SUM */}
          <Show when={!isSum()}>
            <div class="ctrl-wrap">
              <ControlPanel />
            </div>
          </Show>
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
      <CrossoverDialog />
      <FirSettingsDialog />
    </div>
  );
}

export default App;
