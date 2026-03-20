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
import ImpulseResponsePlot from "./components/ImpulseResponsePlot";
import PeqResponsePlot from "./components/PeqResponsePlot";
import ExportPlot from "./components/ExportPlot";
import ExportImpulsePlot from "./components/ExportImpulsePlot";
import ControlPanel from "./components/ControlPanel";
import BandTabs from "./components/BandTabs";
import ProjectNameDialog from "./components/ProjectNameDialog";
import WelcomeDialog from "./components/WelcomeDialog";
import CrossoverDialog from "./components/CrossoverDialog";
import FirSettingsDialog from "./components/FirSettingsDialog";
import { openFirSettings } from "./components/FirSettingsDialog";
import PeqSidebar from "./components/PeqSidebar";
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

  const [impulseRatio, setImpulseRatio] = createSignal(0.35); // 35% для impulse plot

  // Resize для plot split (между freq и impulse)
  let plotDragging = false;
  let plotStartY = 0;
  let plotStartRatio = 0;
  let plotAreaHeight = 0;

  const onPlotResizeStart = (e: MouseEvent) => {
    const plotArea = (e.target as HTMLElement).parentElement;
    if (!plotArea) return;
    plotDragging = true;
    plotStartY = e.clientY;
    plotStartRatio = impulseRatio();
    plotAreaHeight = plotArea.getBoundingClientRect().height;

    const onMove = (ev: MouseEvent) => {
      if (!plotDragging || plotAreaHeight <= 0) return;
      const delta = ev.clientY - plotStartY;
      const ratioDelta = delta / plotAreaHeight;
      const newRatio = Math.max(0.15, Math.min(0.65, plotStartRatio - ratioDelta));
      setImpulseRatio(newRatio);
    };
    const onUp = () => {
      plotDragging = false;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const infoText = () => {
    const band = activeBand();
    if (!band) return "SUM — all bands";
    if (band.measurement) return `${band.measurement.name} · ${band.measurement.freq.length} pts`;
    return "no measurement";
  };

  // Показывать ли нижний plot: не SUM и есть активная полоса
  const showBottomPlot = () => true;
  // На вкладке target — показываем PEQ Response вместо Impulse (only in band mode)
  const showPeqPlot = () => activeTab() === "target" && !isSum();
  // На вкладке export — показываем ExportPlot + ExportImpulsePlot (only in band mode)
  const showExportPlot = () => activeTab() === "export" && !isSum();

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
        <span class="top-info">{infoText()}</span>
      </div>

      {/* Band Tabs */}
      <BandTabs />

      {/* Main content row: plots+panel on left, PeqSidebar on right */}
      <div class="main-content-row">
        <div class="main-content-col">
          {/* Plot area — dual graph when not SUM */}
          <main class="plot-area">
            <div
              class="freq-plot-area"
              style={{ flex: showBottomPlot() ? `${1 - impulseRatio()}` : "1", position: "relative" }}
            >
              {/* FrequencyPlot ALWAYS visible in normal flow.
                  ExportPlot overlays on top when active (absolute positioned). */}
              <div style={{ width: "100%", height: "100%" }}>
                <FrequencyPlot />
              </div>
              <Show when={showExportPlot()}>
                <div style={{ position: "absolute", inset: "0", "z-index": "10", background: "var(--bg-main, #1a1a2e)" }}>
                  <ExportPlot />
                </div>
              </Show>
            </div>
            <Show when={showBottomPlot()}>
              <div class="resize-handle-plots" onMouseDown={onPlotResizeStart} />
              <div class="impulse-plot-area" style={{ flex: `${impulseRatio()}` }}>
                <Show when={showExportPlot()}>
                  <ExportImpulsePlot />
                </Show>
                <Show when={showPeqPlot() && !showExportPlot()}>
                  <PeqResponsePlot />
                </Show>
                <Show when={!showPeqPlot() && !showExportPlot()}>
                  <ImpulseResponsePlot />
                </Show>
              </div>
            </Show>
          </main>

          {/* Resize handle + Control panel — hidden on SUM */}
          <Show when={!isSum()}>
            <div class="ctrl-wrap">
              <ControlPanel />
            </div>
          </Show>
        </div>

        {/* PEQ Sidebar — full height, only on align tab */}
        <Show when={showPeqPlot() && !isSum()}>
          <PeqSidebar />
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
