import { createEffect, createSignal, onCleanup, onMount, For, Show } from "solid-js";
import { createStore } from "solid-js/store";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { Measurement, TargetResponse, FilterType } from "../lib/types";
import { appState, activeBand, isSum, activeTab, plotTab, setPlotTab, sharedXScale, setSharedXScale, suppressXScaleSync, selectedPeqIdx, setSelectedPeqIdx, setBandLowPass, setBandCrossNormDb, plotShowOnly, setPlotShowOnly, addPeqBand, exportHybridPhase, freqSnapshots, setFreqSnapshots, peqDragging, bandsVersion, exportSampleRate, exportTaps, exportWindow, firIterations, firFreqWeighting, firNarrowbandLimit, firNbSmoothingOct, firNbMaxExcess, firMaxBoost, firNoiseFloor, setExportMetrics } from "../stores/bands";
import type { SmoothingMode, BandState, FreqSnapshot } from "../stores/bands";
import { needAutoFit, setNeedAutoFit } from "../App";
import { computeFloorBounce } from "../lib/floor-bounce";
import { openCrossoverDialog, type CrossoverDialogData } from "./CrossoverDialog";

import {
  SUM_TARGET_COLOR, SUM_TARGET_PHASE_COLOR, SUM_CORRECTED_COLOR, SUM_MEAS_COLOR,
  FREQ_SNAP_COLORS, bandColorFamily, smoothingConfig, wrapPhase, fmtFreq, computeGroupDelay,
} from "../lib/plot-helpers";

// ---------------------------------------------------------------------------
// Crossover point: band[i] LP ↔ band[i+1] HP
// ---------------------------------------------------------------------------
interface CrossoverPoint {
  bandIndex: number;   // index of band with LP
  bandId: string;
  bandName: string;
  nextBandName: string;
  freq: number;        // LP freq_hz
  filterType: FilterType;
  order: number;
  linearPhase: boolean;
  shape: number | null; // Gaussian M coefficient
  q: number | null;    // Custom Q factor
  dbLevel: number;     // dB at crossover freq (for Y position on chart)
}

function getCrossovers(): CrossoverPoint[] {
  const pts: CrossoverPoint[] = [];
  const bands = appState.bands;
  for (let i = 0; i < bands.length - 1; i++) {
    const b = bands[i];
    if (b.linkedToNext && b.target.low_pass) {
      pts.push({
        bandIndex: i,
        bandId: b.id,
        bandName: b.name,
        nextBandName: bands[i + 1].name,
        freq: b.target.low_pass.freq_hz,
        filterType: b.target.low_pass.filter_type,
        order: b.target.low_pass.order,
        linearPhase: b.target.low_pass.linear_phase,
        shape: b.target.low_pass.shape ?? null,
        q: b.target.low_pass.q ?? null,
        dbLevel: 0, // enriched later in renderSumMode
      });
    }
  }
  return pts;
}

function fmtCrossoverFreq(f: number): string {
  if (f >= 10000) return (f / 1000).toFixed(1) + "k";
  if (f >= 1000) return (f / 1000).toFixed(2).replace(/\.?0+$/, "") + "k";
  return Math.round(f).toString();
}

// Описание серии для кастомной легенды SUM
interface LegendEntry {
  label: string;
  color: string;
  dash: boolean;
  visible: boolean;
  seriesIdx: number;
  category: "measurement" | "target" | "corrected";
}

export default function FrequencyPlot() {
  let containerRef: HTMLDivElement | undefined;
  let chart: uPlot | undefined;
  let renderGen = 0; // generation counter to discard stale async renders

  let fitMagMin = -100;
  let fitMagMax = 100;
  // Mutable Y-scale state — updated by buttons, used by range function
  let curMagMin = -100;
  let curMagMax = 100;
  // Y-zoom anchor: passband reference level (adaptive to HP/LP filters, or 0 dB without measurements)
  let zoomCenter = 0;

  // Phase Y-scale state
  let curPhaseMin = -190;
  let curPhaseMax = 190;

  // Zoom history for undo (right-click)
  const zoomStack: { xMin: number; xMax: number; magMin: number; magMax: number; phaseMin: number; phaseMax: number }[] = [];
  function pushZoom() {
    if (!chart) return;
    const xs = chart.scales["x"];
    if (xs?.min != null && xs?.max != null) {
      zoomStack.push({ xMin: xs.min, xMax: xs.max, magMin: curMagMin, magMax: curMagMax, phaseMin: curPhaseMin, phaseMax: curPhaseMax });
      if (zoomStack.length > 20) zoomStack.shift();
    }
  }
  function popZoom() {
    if (!chart || zoomStack.length === 0) return;
    const prev = zoomStack.pop()!;
    curMagMin = prev.magMin;
    curMagMax = prev.magMax;
    curPhaseMin = prev.phaseMin;
    curPhaseMax = prev.phaseMax;
    chart.setScale("x", { min: prev.xMin, max: prev.xMax });
    chart.setScale("mag", { min: curMagMin, max: curMagMax });
    chart.setScale("phase", { min: curPhaseMin, max: curPhaseMax });
  }

  // Zoom box state (Ctrl+drag)
  let zoomBoxActive = false;
  let zoomBoxStartX = 0;
  let zoomBoxStartY = 0;
  let zoomBoxEl: HTMLDivElement | null = null;

  const [cursorFreq, setCursorFreq] = createSignal("—");
  const [cursorSPL, setCursorSPL] = createSignal("—");
  const [cursorPhase, setCursorPhase] = createSignal("—");
  // Per-curve values at cursor position: { label, color, value, unit }
  const [cursorValues, setCursorValues] = createSignal<{ label: string; color: string; value: string }[]>([]);

  // Snapshot system: keep last corrected curve for snapshot capture
  const lastCorrData: { freq: number[]; mag: number[]; phase: (number | null)[] } = { freq: [], mag: [], phase: [] };

  function takeFreqSnapshot() {
    const band = activeBand();
    if (!band || lastCorrData.freq.length === 0) return;
    const snaps = freqSnapshots(band.id);
    const idx = snaps.length;
    const color = FREQ_SNAP_COLORS[idx % FREQ_SNAP_COLORS.length];
    const label = `Snap ${idx + 1}`;
    setFreqSnapshots(band.id, [...snaps, {
      label,
      freq: [...lastCorrData.freq],
      mag: [...lastCorrData.mag],
      phase: [...lastCorrData.phase],
      color,
    }]);
  }

  function clearFreqSnapshots() {
    const band = activeBand();
    if (!band) return;
    setFreqSnapshots(band.id, []);
  }
  const [legendEntries, setLegendEntries] = createStore<LegendEntry[]>([]);
  const [showLegend, setShowLegend] = createSignal(false);

  // IR/Step tab options — matrix: [meas/target/corrected] × [ir/step]
  const [irDbMode, setIrDbMode] = createSignal(false);
  const [showMeasIr, setShowMeasIr] = createSignal(true);
  const [showMeasStep, setShowMeasStep] = createSignal(true);
  const [showTargetIr, setShowTargetIr] = createSignal(true);
  const [showTargetStep, setShowTargetStep] = createSignal(true);
  const [showCorrIr, setShowCorrIr] = createSignal(true);
  const [showCorrStep, setShowCorrStep] = createSignal(true);
  const [irShowMasking, setIrShowMasking] = createSignal(true);

  // Persistent visibility — two maps for different modes:
  // SUM mode: by label (each band has its own curves like "Band 1 tgt", "Band 2 tgt")
  // Band mode: by category key (labels change per band, categories don't)
  let sumVisMap = new Map<string, boolean>();
  let bandVisMap = new Map<string, boolean>();

  /** Key for band-mode visibility persistence — use label directly.
   *  Labels are stable across bands: "Measurement", "Target", "PEQ Corrected", etc. */
  function catKey(e: LegendEntry): string {
    return e.label;
  }

  // Crossover drag state
  const [hoveredXo, setHoveredXo] = createSignal<number | null>(null); // index in crossovers array
  const [draggingXo, setDraggingXo] = createSignal<number | null>(null);
  const [dragFreq, setDragFreq] = createSignal<number | null>(null); // visual override during drag
  let currentCrossovers: CrossoverPoint[] = []; // cached for mouse handlers

  function zoomY(factor: number) {
    if (!chart) return;
    const pTab = plotTab();
    const yKey = (pTab === "freq" || pTab === "export") ? "mag" : "y";
    const s = chart.scales[yKey];
    if (!s || s.min == null || s.max == null) return;
    pushZoom();
    const center = (pTab === "freq" || pTab === "export") ? zoomCenter : (s.min + s.max) / 2;
    const half = ((s.max - s.min) / 2) * factor;
    if (half * 2 < 0.01 || half * 2 > 500) { zoomStack.pop(); return; }
    const newMin = center - half;
    const newMax = center + half;
    if (pTab === "freq" || pTab === "export") { curMagMin = newMin; curMagMax = newMax; }
    chart.setScale(yKey, { min: newMin, max: newMax });
  }

  function scrollY(direction: number) {
    if (!chart) return;
    const pTab = plotTab();
    const yKey = (pTab === "freq" || pTab === "export") ? "mag" : "y";
    const s = chart.scales[yKey];
    if (!s || s.min == null || s.max == null) return;
    const range = s.max - s.min;
    const step = range * 0.2 * direction;
    const newMin = s.min + step;
    const newMax = s.max + step;
    if (pTab === "freq" || pTab === "export") { curMagMin = newMin; curMagMax = newMax; }
    chart.setScale(yKey, { min: newMin, max: newMax });
  }

  function zoomX(factor: number) {
    if (!chart) return;
    pushZoom();
    const s = chart.scales["x"];
    if (!s || s.min == null || s.max == null) { zoomStack.pop(); return; }
    const pTab = plotTab();
    if (pTab === "freq" || pTab === "gd" || pTab === "export") {
      // Log scale zoom
      if (s.min <= 0) { zoomStack.pop(); return; }
      const logMin = Math.log10(s.min);
      const logMax = Math.log10(s.max);
      const logCenter = (logMin + logMax) / 2;
      const logHalf = ((logMax - logMin) / 2) * factor;
      chart.setScale("x", { min: Math.max(1, Math.pow(10, logCenter - logHalf)), max: Math.min(100000, Math.pow(10, logCenter + logHalf)) });
    } else {
      // Linear scale zoom (IR/Step)
      const center = (s.min + s.max) / 2;
      const half = ((s.max - s.min) / 2) * factor;
      chart.setScale("x", { min: center - half, max: center + half });
    }
  }

  function scrollX(direction: number) {
    if (!chart) return;
    const s = chart.scales["x"];
    if (!s || s.min == null || s.max == null) return;
    const pTab = plotTab();
    if (pTab === "freq" || pTab === "gd" || pTab === "export") {
      if (s.min <= 0) return;
      const logMin = Math.log10(s.min);
      const logMax = Math.log10(s.max);
      const step = (logMax - logMin) * 0.15 * direction;
      chart.setScale("x", { min: Math.max(1, Math.pow(10, logMin + step)), max: Math.min(100000, Math.pow(10, logMax + step)) });
    } else {
      const range = s.max - s.min;
      const step = range * 0.15 * direction;
      chart.setScale("x", { min: s.min + step, max: s.max + step });
    }
  }

  function fitData() {
    if (!chart) return;
    const pTab = plotTab();
    pushZoom();
    if (pTab === "freq" || pTab === "export") {
      curMagMin = fitMagMin;
      curMagMax = fitMagMax;
      chart.setScale("mag", { min: curMagMin, max: curMagMax });
      curPhaseMin = -190;
      curPhaseMax = 190;
      chart.setScale("phase", { min: curPhaseMin, max: curPhaseMax });
      chart.setScale("x", { min: 20, max: 20000 });
    } else if (pTab === "gd") {
      chart.setScale("x", { min: 20, max: 20000 });
      // Y auto from data
      const s = chart.scales["y"];
      if (s) chart.setScale("y", { min: s.min, max: s.max });
    } else {
      // IR/Step — auto range from data
      const d = chart.data[0];
      if (d && d.length > 0) chart.setScale("x", { min: d[0], max: d[d.length - 1] });
      const s = chart.scales["y"];
      if (s) chart.setScale("y", { min: s.min, max: s.max });
    }
  }

  function zoomPhase(factor: number) {
    if (!chart) return;
    pushZoom();
    const center = (curPhaseMin + curPhaseMax) / 2;
    const half = ((curPhaseMax - curPhaseMin) / 2) * factor;
    if (half * 2 < 10 || half * 2 > 720) { zoomStack.pop(); return; }
    curPhaseMin = center - half;
    curPhaseMax = center + half;
    chart.setScale("phase", { min: curPhaseMin, max: curPhaseMax });
  }

  function scrollPhase(direction: number) {
    if (!chart) return;
    const range = curPhaseMax - curPhaseMin;
    const step = range * 0.2 * direction;
    curPhaseMin += step;
    curPhaseMax += step;
    chart.setScale("phase", { min: curPhaseMin, max: curPhaseMax });
  }

  // Zoom box handlers (Ctrl+drag)
  function handleZoomBoxDown(e: MouseEvent) {
    if (!e.ctrlKey && !e.metaKey) return;
    if (!chart || !containerRef) return;
    e.preventDefault();
    pushZoom();
    zoomBoxActive = true;
    const rect = containerRef.getBoundingClientRect();
    zoomBoxStartX = e.clientX - rect.left;
    zoomBoxStartY = e.clientY - rect.top;
    // Create overlay element
    if (!zoomBoxEl) {
      zoomBoxEl = document.createElement("div");
      zoomBoxEl.className = "zoom-box";
      containerRef.appendChild(zoomBoxEl);
    }
    zoomBoxEl.style.display = "block";
    zoomBoxEl.style.left = zoomBoxStartX + "px";
    zoomBoxEl.style.top = zoomBoxStartY + "px";
    zoomBoxEl.style.width = "0";
    zoomBoxEl.style.height = "0";
  }

  function handleZoomBoxMove(e: MouseEvent) {
    if (!zoomBoxActive || !zoomBoxEl || !containerRef) return;
    const rect = containerRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const left = Math.min(zoomBoxStartX, x);
    const top = Math.min(zoomBoxStartY, y);
    const w = Math.abs(x - zoomBoxStartX);
    const h = Math.abs(y - zoomBoxStartY);
    zoomBoxEl.style.left = left + "px";
    zoomBoxEl.style.top = top + "px";
    zoomBoxEl.style.width = w + "px";
    zoomBoxEl.style.height = h + "px";
  }

  function handleZoomBoxUp(e: MouseEvent) {
    if (!zoomBoxActive || !chart || !containerRef) return;
    zoomBoxActive = false;
    if (zoomBoxEl) zoomBoxEl.style.display = "none";

    const rect = containerRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const x1 = Math.min(zoomBoxStartX, x);
    const x2 = Math.max(zoomBoxStartX, x);
    const y1 = Math.min(zoomBoxStartY, y);
    const y2 = Math.max(zoomBoxStartY, y);

    // Min size threshold (10px)
    if (x2 - x1 < 10 || y2 - y1 < 10) { zoomStack.pop(); return; }

    // Convert pixel positions to data values via uPlot
    const fMin = chart.posToVal(x1, "x");
    const fMax = chart.posToVal(x2, "x");
    const magMax2 = chart.posToVal(y1, "mag");
    const magMin2 = chart.posToVal(y2, "mag");

    if (isFinite(fMin) && isFinite(fMax) && fMin > 0 && fMax > fMin) {
      chart.setScale("x", { min: fMin, max: fMax });
    }
    if (isFinite(magMin2) && isFinite(magMax2) && magMax2 > magMin2) {
      curMagMin = magMin2;
      curMagMax = magMax2;
      chart.setScale("mag", { min: curMagMin, max: curMagMax });
    }
  }

  // Right-click = undo zoom
  function handleContextMenu(e: MouseEvent) {
    if (!chart) return;
    e.preventDefault();
    popZoom();
  }

  // Переключение видимости серии через легенду
  function toggleLegendEntry(idx: number) {
    if (!chart) return;
    const entry = legendEntries[idx];
    const newVis = !entry.visible;
    setLegendEntries(idx, "visible", newVis);
    chart.setSeries(entry.seriesIdx, { show: newVis });
    if (isSum()) {
      sumVisMap.set(entry.label, newVis);
      // Also toggle paired phase series (mag ↔ phase)
      if (!entry.label.endsWith(" \u00B0")) {
        // This is a mag series — find paired phase
        const phaseSuffix = " \u00B0";
        for (let pi = 0; pi < legendEntries.length; pi++) {
          const pe = legendEntries[pi];
          if (pe.category === entry.category && pe.label === entry.label + phaseSuffix) {
            if (pe.visible !== newVis) {
              setLegendEntries(pi, "visible", newVis);
              chart.setSeries(pe.seriesIdx, { show: newVis });
              sumVisMap.set(pe.label, newVis);
            }
            break;
          }
        }
        // For Σ series: legend label may differ from series label
        // "Σ corrected" mag → "Σ corr °" phase, "Σ meas" → "Σ meas °", "Σ target" → "Σ target °"
        const sigmaPhaseMap: Record<string, string> = {
          "\u03A3 corrected": "\u03A3 corr \u00B0",
          "\u03A3 meas": "\u03A3 meas \u00B0",
          "\u03A3 target": "\u03A3 target \u00B0",
        };
        const sigmaPhLabel = sigmaPhaseMap[entry.label];
        if (sigmaPhLabel) {
          for (let pi = 0; pi < legendEntries.length; pi++) {
            if (legendEntries[pi].label === sigmaPhLabel) {
              if (legendEntries[pi].visible !== newVis) {
                setLegendEntries(pi, "visible", newVis);
                chart.setSeries(legendEntries[pi].seriesIdx, { show: newVis });
                sumVisMap.set(sigmaPhLabel, newVis);
              }
              break;
            }
          }
        }
      }
    } else {
      bandVisMap.set(catKey(entry), newVis);
    }
  }

  // Toggle all series for a given band column in SUM mode
  function toggleColumn(colName: string) {
    if (!chart) return;
    const matching: number[] = [];
    for (let i = 0; i < legendEntries.length; i++) {
      const e = legendEntries[i];
      if (colName === "\u03A3") {
        if (e.label.startsWith("\u03A3")) matching.push(i);
      } else {
        if (e.label === colName || e.label === colName + " \u00B0"
          || e.label === colName + " tgt"
          || e.label === colName + " corr+XO" || e.label === colName + " corr+XO \u00B0") matching.push(i);
      }
    }
    if (matching.length === 0) return;
    const allOn = matching.every(i => legendEntries[i].visible);
    const newVis = !allOn;
    for (const i of matching) {
      if (legendEntries[i].visible !== newVis) {
        setLegendEntries(i, "visible", newVis);
        chart.setSeries(legendEntries[i].seriesIdx, { show: newVis });
        if (isSum()) {
          sumVisMap.set(legendEntries[i].label, newVis);
        } else {
          bandVisMap.set(catKey(legendEntries[i]), newVis);
        }
      }
    }
  }

  // Find a legend entry for a specific [bandName, category] cell
  function findCellEntry(colName: string, cat: "measurement" | "target" | "corrected"): LegendEntry | undefined {
    for (let i = 0; i < legendEntries.length; i++) {
      const e = legendEntries[i];
      if (e.category !== cat) continue;
      if (colName === "\u03A3") {
        if (e.label.startsWith("\u03A3")) return e;
      } else {
        if (cat === "measurement" && e.label === colName) return e;
        if (cat === "target" && e.label === colName + " tgt") return e;
        if (cat === "corrected" && e.label === colName + " corr+XO") return e;
      }
    }
    return undefined;
  }

  // Переключение всей категории (targets / measurements / corrected)
  function toggleCategory(cat: "measurement" | "target" | "corrected") {
    if (!chart) return;
    const indices: number[] = [];
    for (let i = 0; i < legendEntries.length; i++) {
      if (legendEntries[i].category === cat) indices.push(i);
    }
    if (indices.length === 0) return;
    const allOn = indices.every(i => legendEntries[i].visible);
    const newVis = !allOn;
    for (const i of indices) {
      if (legendEntries[i].visible !== newVis) {
        setLegendEntries(i, "visible", newVis);
        chart.setSeries(legendEntries[i].seriesIdx, { show: newVis });
        if (isSum()) {
          sumVisMap.set(legendEntries[i].label, newVis);
        } else {
          bandVisMap.set(catKey(legendEntries[i]), newVis);
        }
      }
    }
  }

  // React to external "show only X" command
  createEffect(() => {
    const cats = plotShowOnly();
    if (!cats || !chart) return;
    const showSet = new Set(cats);
    const inSum = isSum();
    for (let i = 0; i < legendEntries.length; i++) {
      const show = showSet.has(legendEntries[i].category);
      if (legendEntries[i].visible !== show) {
        setLegendEntries(i, "visible", show);
        chart.setSeries(legendEntries[i].seriesIdx, { show });
      }
      // Persist so switching bands/SUM doesn't lose this state
      if (inSum) {
        sumVisMap.set(legendEntries[i].label, show);
      } else {
        bandVisMap.set(catKey(legendEntries[i]), show);
      }
    }
    setPlotShowOnly(null);
  });

  // ----------------------------------------------------------------
  // Универсальный рендерер чарта
  // ----------------------------------------------------------------
  interface RenderInput {
    freq: number[];
    uSeries: uPlot.Series[];
    uData: number[][];
    hasMeasurements: boolean;
    legend?: LegendEntry[];
    floorBounceNulls?: number[]; // частоты деструктивных интерференций floor bounce
    crossovers?: CrossoverPoint[]; // crossover points for SUM view
  }

  function renderChart(input: RenderInput) {
    if (!containerRef) return;

    // Авто-FIT: при needAutoFit не сохраняем старый масштаб
    const doAutoFit = needAutoFit();
    if (doAutoFit) setNeedAutoFit(false);

    let savedMagMin: number | null = null;
    let savedMagMax: number | null = null;
    let savedXMin: number | null = null;
    let savedXMax: number | null = null;

    if (chart && !doAutoFit) {
      const ms = chart.scales["mag"];
      if (ms?.min != null && ms?.max != null) { savedMagMin = ms.min; savedMagMax = ms.max; }
      const xs = chart.scales["x"];
      if (xs?.min != null && xs?.max != null) { savedXMin = xs.min; savedXMax = xs.max; }
    }
    if (chart) {
      // Remove event listeners before destroying
      if (chart.over) {
        chart.over.removeEventListener("mousemove", handleXoMouseMove);
        chart.over.removeEventListener("mousedown", handleXoMouseDown);
        chart.over.removeEventListener("dblclick", handleXoDblClick);
        chart.over.removeEventListener("dblclick", handlePeqDblClick);
        chart.over.removeEventListener("mousedown", handleZoomBoxDown);
        chart.over.removeEventListener("contextmenu", handleContextMenu);
      }
      window.removeEventListener("mousemove", handleZoomBoxMove);
      window.removeEventListener("mouseup", handleZoomBoxUp);
      try { chart.destroy(); } catch (_) {}
      chart = undefined;
    }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 200);

    if (input.uData.length !== input.uSeries.length) return;

    // Mag range — только из mag серий, исключаем значения < -200
    let magMin = Infinity;
    let magMax = -Infinity;
    for (let i = 1; i < input.uSeries.length; i++) {
      if ((input.uSeries[i] as any).scale !== "mag") continue;
      const d = input.uData[i];
      for (let j = 0; j < d.length; j++) {
        const v = d[j];
        if (v > -200 && v < magMin) magMin = v;
        if (v > -200 && v > magMax) magMax = v;
      }
    }
    if (!isFinite(magMin)) { magMin = -100; magMax = 100; }

    // Fit range: +10 dBr top, noise floor (data minimum) bottom
    fitMagMax = zoomCenter + 10;
    fitMagMin = Math.floor(magMin / 5) * 5; // round down to nearest 5 dB
    if (fitMagMax - fitMagMin < 20) fitMagMin = fitMagMax - 20;
    curMagMin = savedMagMin ?? fitMagMin;
    curMagMax = savedMagMax ?? fitMagMax;

    const yLabel = input.hasMeasurements ? "dBr" : "dB";

    const axes: uPlot.Axis[] = [
      {
        stroke: "#9b9ba6",
        grid: { stroke: "rgba(255,255,255,0.12)" },
        ticks: { stroke: "rgba(255,255,255,0.20)" },
        values: (_u: uPlot, vals: number[]) =>
          vals.map((v) => {
            if (v == null) return "";
            if (v >= 1000) return (v / 1000) + "k";
            return String(Math.round(v * 10) / 10);
          }),
      },
      {
        label: yLabel, scale: "mag", stroke: "#9b9ba6",
        grid: { stroke: "rgba(255,255,255,0.12)" },
        ticks: { stroke: "rgba(255,255,255,0.20)" },
        values: (_u: uPlot, vals: number[]) => vals.map((v) => (v == null ? "" : v.toFixed(0))),
        size: 50,
      },
      {
        label: "Phase (\u00B0)", scale: "phase", side: 1, stroke: "#9b9ba6",
        grid: { show: false },
        ticks: { stroke: "rgba(255,255,255,0.20)" },
      },
    ];

    // Restore previous legend visibility:
    // SUM mode → by label (each band has unique labels like "Band 1 tgt")
    // Band mode → by category key (labels change per band, categories don't)
    const inSum = isSum();
    const prevVisMap = new Map<string, boolean>();
    if (inSum && sumVisMap.size > 0 && input.legend) {
      for (const e of input.legend) {
        if (sumVisMap.has(e.label)) prevVisMap.set(e.label, sumVisMap.get(e.label)!);
      }
    } else if (!inSum && bandVisMap.size > 0 && input.legend) {
      for (const e of input.legend) {
        const key = catKey(e);
        if (bandVisMap.has(key)) prevVisMap.set(e.label, bandVisMap.get(key)!);
      }
    }

    let mergedLegend: LegendEntry[] | undefined;
    if (input.legend && input.legend.length > 0) {
      mergedLegend = input.legend.map((e) => {
        // If we have a saved visibility state, always use it (user's explicit choice)
        if (prevVisMap.has(e.label)) {
          return { ...e, visible: prevVisMap.get(e.label)! };
        }
        // Otherwise use the default from renderBandMode/renderSumMode
        return e;
      });
      // Apply show to series
      for (const entry of mergedLegend) {
        if (input.uSeries[entry.seriesIdx]) {
          (input.uSeries[entry.seriesIdx] as any).show = entry.visible;
        }
      }
    }

    const allSeries = input.uSeries;

    const opts: uPlot.Options = {
      width: w, height: h, series: allSeries,
      scales: {
        x: { distr: 3, log: 10, min: savedXMin ?? 20, max: savedXMax ?? 20000 },
        mag: { auto: false, range: () => [curMagMin, curMagMax] as uPlot.Range.MinMax },
        phase: { auto: false, range: [-190, 190] },
      },
      axes,
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
      hooks: {
        setScale: [
          (u: uPlot, key: string) => {
            if (key !== "x") return;
            const s = u.scales["x"];
            if (s?.min != null && s?.max != null && s.min > 0) {
              setSharedXScale({ min: s.min, max: s.max });
            }
          },
        ],
        setCursor: [
          (u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= u.data[0].length) {
              setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
              setCursorValues([]);
              return;
            }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? fmtFreq(f) : "—");

            // Collect values for all visible series
            const vals: { label: string; color: string; value: string }[] = [];
            let firstSPL: string | null = null;
            let firstPhase: string | null = null;
            const leg = mergedLegend ?? input.legend ?? [];

            for (let si = 1; si < u.series.length; si++) {
              if (!u.series[si].show) continue;
              const orig = allSeries[si] as any;
              if (!orig) continue;
              const v = u.data[si]?.[idx];
              if (v == null || !isFinite(v)) continue;
              const isMag = orig.scale === "mag";
              const isPhase = orig.scale === "phase";
              const unit = isMag ? " dBr" : isPhase ? "°" : "";
              const formatted = v.toFixed(1) + unit;
              const color = orig.stroke ?? "#ccc";
              const label = orig.label ?? `s${si}`;

              // Find short label from legend
              const le = leg.find((e: any) => e.seriesIdx === si);
              vals.push({ label: le?.label ?? label, color, value: formatted });

              if (isMag && !firstSPL) firstSPL = formatted;
              if (isPhase && !firstPhase) firstPhase = formatted;
            }

            setCursorSPL(firstSPL ?? "—");
            setCursorPhase(firstPhase ?? "—");
            setCursorValues(vals);
          },
        ],
        draw: [
          // Floor bounce overlay: рисуем вертикальные полосы на null-частотах
          (u: uPlot) => {
            const nullFreqs = input.floorBounceNulls;
            if (!nullFreqs || nullFreqs.length === 0) return;
            const ctx = u.ctx;
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotWidth = u.bbox.width;
            const plotHeight = u.bbox.height;

            ctx.save();
            ctx.fillStyle = "rgba(255, 165, 0, 0.12)";

            for (const f of nullFreqs) {
              const xPos = u.valToPos(f, "x", true);
              if (xPos < plotLeft || xPos > plotLeft + plotWidth) continue;
              // Ширина полосы: ~2% от plot width, min 2px
              const bandWidth = Math.max(2, plotWidth * 0.015);
              ctx.fillRect(xPos - bandWidth / 2, plotTop, bandWidth, plotHeight);
            }

            ctx.restore();
          },
          // Exclusion zones — semi-transparent gray bands
          (u: uPlot) => {
            const bd = activeBand();
            if (!bd?.exclusionZones?.length) return;
            const ctx = u.ctx;
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotHeight = u.bbox.height;
            const plotRight = plotLeft + u.bbox.width;

            ctx.save();

            ctx.fillStyle = "rgba(128, 128, 128, 0.15)";
            for (const zone of bd.exclusionZones) {
              const x1 = u.valToPos(zone.startHz, "x", true);
              const x2 = u.valToPos(zone.endHz, "x", true);
              const left = Math.max(x1, plotLeft);
              const right = Math.min(x2, plotRight);
              if (right > left) {
                ctx.fillRect(left, plotTop, right - left, plotHeight);
              }
            }

            ctx.restore();
          },
          // PEQ bands — vertical dashed lines for all enabled bands
          (u: uPlot) => {
            const bd = activeBand();
            if (!bd?.peqBands?.length) return;
            const selIdx = selectedPeqIdx();
            const ctx = u.ctx;
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotRight = plotLeft + u.bbox.width;
            const plotBottom = plotTop + u.bbox.height;
            const dpr = devicePixelRatio || 1;

            for (let i = 0; i < bd.peqBands.length; i++) {
              const pb = bd.peqBands[i];
              if (!pb.enabled) continue;
              const cx = u.valToPos(pb.freq_hz, "x", true);
              if (cx < plotLeft || cx > plotRight) continue;

              const isSel = i === selIdx;
              ctx.save();
              ctx.strokeStyle = isSel ? "#FF9F43" : "rgba(255,159,67,0.35)";
              ctx.lineWidth = isSel ? 2 : 1;
              ctx.setLineDash(isSel ? [6, 4] : [4, 4]);
              ctx.beginPath();
              ctx.moveTo(cx, plotTop);
              ctx.lineTo(cx, plotBottom);
              ctx.stroke();

              if (isSel) {
                ctx.setLineDash([]);
                ctx.fillStyle = "#FF9F43";
                ctx.font = `${Math.round(10 * dpr)}px sans-serif`;
                ctx.textAlign = "center";
                const label = pb.freq_hz >= 1000 ? (pb.freq_hz / 1000).toFixed(1) + "k" : Math.round(pb.freq_hz).toString();
                ctx.fillText(label, cx, plotTop - 4 * dpr);
              }
              ctx.restore();
            }
          },
          // Crossover markers (SUM mode)
          (u: uPlot) => {
            const xoList = input.crossovers;
            if (!xoList || xoList.length === 0) return;

            const ctx = u.ctx;
            const dpr = devicePixelRatio || 1;
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotRight = plotLeft + u.bbox.width;
            const plotBottom = plotTop + u.bbox.height;

            const hov = hoveredXo();
            const drg = draggingXo();

            ctx.save();
            for (let xi = 0; xi < xoList.length; xi++) {
              const xo = xoList[xi];
              // Use drag override freq if this is the one being dragged
              const f = (drg === xi && dragFreq() != null) ? dragFreq()! : xo.freq;
              const cx = u.valToPos(f, "x", true);
              if (cx < plotLeft || cx > plotRight) continue;

              const isHov = hov === xi;
              const isDrg = drg === xi;

              // Y position from dB level (crossover point on target curve)
              let cy = u.valToPos(xo.dbLevel, "mag", true);
              const radius = isDrg ? 10 * dpr : isHov ? 9 * dpr : 7 * dpr;
              // Clamp to plot area
              cy = Math.max(plotTop + radius + 2, Math.min(plotBottom - radius - 2, cy));

              // Vertical dashed line
              ctx.strokeStyle = isDrg ? "#FFC107" : isHov ? "#FFA726" : "#FF9800";
              ctx.lineWidth = isDrg ? 2 : 1.5;
              ctx.setLineDash([6, 4]);
              ctx.beginPath();
              ctx.moveTo(cx, plotTop);
              ctx.lineTo(cx, plotBottom);
              ctx.stroke();

              // Circle marker at dB level
              ctx.setLineDash([]);
              ctx.fillStyle = isDrg ? "#FFC107" : isHov ? "#FFA726" : "#FF9800";
              ctx.strokeStyle = "#1e1e24";
              ctx.lineWidth = 2 * dpr;
              ctx.beginPath();
              ctx.arc(cx, cy, radius, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();

              // Frequency label above marker
              ctx.fillStyle = isDrg ? "#FFC107" : isHov ? "#FFA726" : "#FF9800";
              ctx.font = `bold ${Math.round(11 * dpr)}px sans-serif`;
              ctx.textAlign = "center";
              ctx.fillText(fmtCrossoverFreq(f), cx, cy - radius - 6 * dpr);
            }
            ctx.restore();
          },
        ],
      },
    };

    try {
      chart = new uPlot(opts, input.uData as uPlot.AlignedData, containerRef);
    } catch (e) {
      console.error("uPlot error:", e);
    }

    // Update legend state (visibility was already applied to series before chart creation)
    if (mergedLegend && mergedLegend.length > 0) {
      setLegendEntries(mergedLegend);
      setShowLegend(true);
    } else {
      setLegendEntries([]);
      setShowLegend(false);
    }

    // Cache crossover points for mouse interaction
    currentCrossovers = input.crossovers ?? [];

    // Attach crossover mouse handlers to uPlot's overlay div (it captures all mouse events)
    if (chart && chart.over && currentCrossovers.length > 0) {
      const over = chart.over;
      over.addEventListener("mousemove", handleXoMouseMove);
      over.addEventListener("mousedown", handleXoMouseDown);
      over.addEventListener("dblclick", handleXoDblClick);
    }

    // PEQ double-click: add new band at cursor frequency (align tab only)
    if (chart && chart.over) {
      chart.over.addEventListener("dblclick", handlePeqDblClick);
    }

    // Zoom box (Ctrl+drag) + right-click undo zoom
    if (chart && chart.over) {
      chart.over.addEventListener("mousedown", handleZoomBoxDown);
      chart.over.addEventListener("contextmenu", handleContextMenu);
    }
    window.addEventListener("mousemove", handleZoomBoxMove);
    window.addEventListener("mouseup", handleZoomBoxUp);
  }

  onMount(() => {
    if (!containerRef) return;
    const observer = new ResizeObserver(() => {
      if (chart && containerRef) {
        const rect = containerRef.getBoundingClientRect();
        chart.setSize({ width: Math.max(rect.width, 400), height: Math.max(rect.height, 200) });
      }
    });
    observer.observe(containerRef);
    onCleanup(() => observer.disconnect());
  });

  // Sync X-scale from PeqResponsePlot (or any other plot that sets sharedXScale)
  createEffect(() => {
    const xs = sharedXScale();
    if (!chart) return;
    const cur = chart.scales["x"];
    if (cur?.min != null && cur?.max != null) {
      // Only update if significantly different to avoid loops
      if (Math.abs(cur.min - xs.min) < 0.01 && Math.abs(cur.max - xs.max) < 0.01) return;
    }
    suppressXScaleSync(() => {
      chart!.setScale("x", { min: xs.min, max: xs.max });
    });
  });

  // IR/Step visibility toggle: setSeries by fixed indices, preserve scales
  function irToggleVisibility() {
    if (!chart) return;
    try {
      // Save current scales
      const xs = chart.scales["x"];
      const ys = chart.scales["y"];
      const xMin = xs?.min; const xMax = xs?.max;
      const yMin = ys?.min; const yMax = ys?.max;
      // Toggle series
      if (chart.series[1]) chart.setSeries(1, { show: showMeasIr() });
      if (chart.series[2]) chart.setSeries(2, { show: showMeasStep() });
      if (chart.series[3]) chart.setSeries(3, { show: showTargetIr() });
      if (chart.series[4]) chart.setSeries(4, { show: showTargetStep() });
      if (chart.series[5]) chart.setSeries(5, { show: showCorrIr() });
      if (chart.series[6]) chart.setSeries(6, { show: showCorrStep() });
      // Restore scales
      if (xMin != null && xMax != null) chart.setScale("x", { min: xMin, max: xMax });
      if (yMin != null && yMax != null) chart.setScale("y", { min: yMin, max: yMax });
    } catch (_) {}
  }

  // IR dB/masking toggle: needs full rebuild (data changes)
  function irFullRedraw() {
    const pTab = plotTab();
    if (pTab === "ir" || pTab === "step") {
      renderTimeTab("ir", isSum(), activeBand());
    }
  }

  // Redraw when selected PEQ index changes (for vertical dashed line)
  createEffect(() => {
    const _sel = selectedPeqIdx(); // track
    if (chart) chart.redraw(false, false);
  });


  // ----------------------------------------------------------------
  // Helper: apply smoothing
  // ----------------------------------------------------------------
  async function applySmoothing(m: Measurement, mode: SmoothingMode): Promise<Measurement> {
    if (mode === "off") return m;
    const config = smoothingConfig(mode);
    const smoothed = await invoke<number[]>("get_smoothed", {
      freq: m.freq, magnitude: m.magnitude, config,
    });
    return { ...m, magnitude: smoothed };
  }

  // ----------------------------------------------------------------
  // Helper: evaluate a single band
  // ----------------------------------------------------------------
  async function evaluateBand(band: BandState, _showPhase: boolean): Promise<{
    measurement: Measurement | null;
    targetMag: number[] | null;
    targetPhase: number[] | null;
    freq: number[] | null;
  }> {
    const targetCurve = JSON.parse(JSON.stringify(band.target));
    let measurement: Measurement | null = null;

    if (band.measurement) {
      const raw: Measurement = JSON.parse(JSON.stringify(band.measurement));
      const mode = band.settings?.smoothing ?? "off";
      measurement = await applySmoothing(raw, mode);
    }

    let targetMag: number[] | null = null;
    let targetPhase: number[] | null = null;
    let freq: number[] | null = measurement?.freq ?? null;

    if (band.targetEnabled) {
      if (measurement) {
        // Compute autoRef using the same adaptive passband as zoomCenter
        // so target and normalization are aligned in narrow-band configurations
        const hpFreq = band.target.high_pass?.freq_hz ?? 20;
        const lpFreq = band.target.low_pass?.freq_hz ?? 20000;
        const pbLow = Math.max(20, hpFreq * 1.5);
        const pbHigh = Math.min(20000, lpFreq * 0.7);
        const refLow = pbLow < pbHigh ? pbLow : 200;
        const refHigh = pbLow < pbHigh ? pbHigh : 2000;

        let sum = 0, n = 0;
        for (let i = 0; i < measurement.freq.length; i++) {
          if (measurement.freq[i] >= refLow && measurement.freq[i] <= refHigh) {
            sum += measurement.magnitude[i]; n++;
          }
        }
        const autoRef = n > 0 ? sum / n : 0;
        const curveWithRef = { ...targetCurve, reference_level_db: targetCurve.reference_level_db + autoRef };

        const response = await invoke<TargetResponse>("evaluate_target", {
          target: curveWithRef, freq: measurement.freq,
        });
        targetMag = response.magnitude;
        targetPhase = response.phase;
      } else {
        const [standaloneFreq, response] = await invoke<[number[], TargetResponse]>(
          "evaluate_target_standalone", { target: targetCurve }
        );
        freq = standaloneFreq;
        targetMag = response.magnitude;
        targetPhase = response.phase;
      }
    }

    return { measurement, targetMag, targetPhase, freq };
  }

  // ----------------------------------------------------------------
  // Main reactive effect (debounced during PEQ drag)
  // ----------------------------------------------------------------
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  createEffect(() => {
    const showPhase = appState.showPhase;
    const showMag = appState.showMag;
    const showTarget = appState.showTarget;
    const sumMode = isSum();
    const band = activeBand();
    const _bv = bandsVersion();
    const _tab = activeTab();
    const _fsnaps = band ? freqSnapshots(band.id) : [];
    const dragging = peqDragging();
    const pTab = plotTab();

    // Immediately destroy chart on tab switch to prevent stale uPlot issues
    if (debounceTimer) clearTimeout(debounceTimer);
    try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
    ++renderGen; // invalidate any in-flight async renders

    if (pTab === "ir" || pTab === "step" || pTab === "gd") {
      renderTimeTab(pTab === "step" ? "ir" : pTab, sumMode, band);
      return;
    }

    if (pTab === "export") {
      renderExportTab(band);
      return;
    }

    // Freq tab (default)
    const doRender = () => {
      if (sumMode) {
        renderSumMode(showPhase, showMag, showTarget);
      } else if (band) {
        renderBandMode(band, showPhase, showMag, showTarget);
      } else {
        try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
        setShowLegend(false);
        setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
      }
    };

    if (debounceTimer) clearTimeout(debounceTimer);
    if (dragging) {
      debounceTimer = setTimeout(doRender, 150);
    } else {
      doRender();
    }
  });

  // ----------------------------------------------------------------
  // Export tab: Model vs FIR realization
  // ----------------------------------------------------------------
  async function renderExportTab(band: BandState | null) {
    const gen = ++renderGen;
    if (!band || !band.target) {
      try { try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; } } catch (_) { chart = undefined; }
      setShowLegend(false);
      setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
      return;
    }

    try {
      const sr = exportSampleRate();
      const taps = exportTaps();
      const win = exportWindow();
      const target = JSON.parse(JSON.stringify(band.target));
      const peqBands = band.peqBands?.filter((b: any) => b.enabled) ?? [];

      // Evaluate target
      const [freq, response] = await invoke<[number[], TargetResponse]>(
        "evaluate_target_standalone", { target, nPoints: 512, fMin: 5, fMax: 40000 },
      );
      if (gen !== renderGen) return;

      const targetMag = response.magnitude;
      let modelPhase = response.phase;

      // PEQ contribution
      let peqMagArr: number[] = [];
      if (peqBands.length > 0) {
        const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq, bands: peqBands, sampleRate: sr });
        peqMagArr = pm;
        modelPhase = modelPhase.map((v: number, i: number) => v + pp[i]);
      }
      if (gen !== renderGen) return;

      // Generate FIR
      const isLin = (f: any) => !f || f.linear_phase || f.filter_type === "Gaussian";
      const allLinear = isLin(target.high_pass) && isLin(target.low_pass);
      const firConfig = {
        taps, sample_rate: sr, max_boost_db: firMaxBoost(), noise_floor_db: firNoiseFloor(),
        window: win, phase_mode: allLinear ? "LinearPhase" : "MinimumPhase",
        iterations: firIterations(), freq_weighting: firFreqWeighting(),
        narrowband_limit: firNarrowbandLimit(), nb_smoothing_oct: firNbSmoothingOct(),
        nb_max_excess_db: firNbMaxExcess(),
      };
      const firResult = await invoke<{ realized_mag: number[]; realized_phase: number[]; impulse: number[]; time_ms: number[]; norm_db: number; causality: number; taps: number; sample_rate: number }>(
        "generate_model_fir", { freq, targetMag, peqMag: peqMagArr, modelPhase, config: firConfig },
      );
      if (gen !== renderGen) return;

      // Normalize model mag
      const normModelMag = targetMag.map((v: number, i: number) => (v + (peqMagArr[i] ?? 0)) - firResult.norm_db);

      // Set metrics
      setExportMetrics({
        taps: firResult.taps, sampleRate: firResult.sample_rate, window: win,
        phaseLabel: allLinear ? "Linear-Phase" : "Min-Phase",
        peqCount: peqBands.length, normDb: firResult.norm_db,
        causality: Math.round(firResult.causality * 100),
        preRingMs: 0, maxMagErr: 0, gdRippleMs: 0,
      });

      // Render chart
      try { try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; } } catch (_) { chart = undefined; }
      if (!containerRef) return;
      const rect = containerRef.getBoundingClientRect();

      const opts: uPlot.Options = {
        width: Math.max(rect.width, 400), height: Math.max(rect.height, 200),
        series: [
          {},
          { label: "Model dB", stroke: "#FF9F43", width: 2, scale: "mag" },
          { label: "FIR dB", stroke: "#38BDF8", width: 2, scale: "mag" },
        ],
        scales: {
          x: { min: 20, max: 20000, distr: 3 },
          mag: { auto: true },
        },
        axes: [
          { stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
            values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v >= 1000 ? (v/1000)+"k" : String(Math.round(v))) },
          { label: "dB", scale: "mag", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
            values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)), size: 50 },
        ],
        legend: { show: false },
        cursor: { drag: { x: false, y: false, setScale: false } },
        hooks: {
          setCursor: [(u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= u.data[0].length) { setCursorFreq("—"); setCursorSPL("—"); return; }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? (f >= 1000 ? (f/1000).toFixed(2)+" kHz" : Math.round(f)+" Hz") : "—");
            const m = u.data[1]?.[idx]; const r = u.data[2]?.[idx];
            setCursorSPL(
              (m != null ? `Model: ${(m as number).toFixed(1)}` : "") +
              (r != null ? ` FIR: ${(r as number).toFixed(1)}` : "") + " dB"
            );
          }],
        },
      };
      setShowLegend(false);
      try { chart = new uPlot(opts, [freq, normModelMag, firResult.realized_mag], containerRef); } catch (e) { console.error(e); }
    } catch (e) {
      console.error("Export tab render failed:", e);
    }
  }

  // ----------------------------------------------------------------
  // IR / Step / GD rendering (time-domain tabs)
  // ----------------------------------------------------------------
  async function renderTimeTab(mode: "ir" | "step" | "gd", sumMode: boolean, band: BandState | null) {
    const gen = ++renderGen;
    // Snapshot toggle state BEFORE any await
    const irCfg = {
      db: irDbMode(),
      measIr: showMeasIr(), measStep: showMeasStep(),
      targetIr: showTargetIr(), targetStep: showTargetStep(),
      corrIr: showCorrIr(), corrStep: showCorrStep(),
      masking: irShowMasking(),
    };

    // Collect bands with phase data
    const bands = sumMode
      ? appState.bands.filter(b => b.measurement?.phase)
      : (band?.measurement?.phase ? [band] : []);

    if (bands.length === 0) {
      try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
      setShowLegend(false);
      setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
      return;
    }

    try {
      // For GD: compute from phase directly (no IPC needed)
      if (mode === "gd") {
        const b = bands[0];
        const freq = [...b.measurement!.freq];
        const phase = [...b.measurement!.phase!];

        // If SUM mode, coherent sum first
        let gdFreq = freq;
        let gdPhase = phase;
        if (sumMode && bands.length > 1) {
          const n = freq.length;
          const sumRe = new Float64Array(n);
          const sumIm = new Float64Array(n);
          for (const sb of bands) {
            const sign = sb.inverted ? -1 : 1;
            for (let j = 0; j < n; j++) {
              const amp = Math.pow(10, (sb.measurement!.magnitude[j] ?? -200) / 20) * sign;
              const phRad = (sb.measurement!.phase![j] ?? 0) * Math.PI / 180;
              sumRe[j] += amp * Math.cos(phRad);
              sumIm[j] += amp * Math.sin(phRad);
            }
          }
          // Unwrap sum phase
          const sumPh: number[] = [];
          for (let j = 0; j < n; j++) {
            sumPh.push(Math.atan2(sumIm[j], sumRe[j]) * 180 / Math.PI);
          }
          const unwrapped: number[] = [sumPh[0]];
          for (let i = 1; i < n; i++) {
            let diff = sumPh[i] - sumPh[i - 1];
            while (diff > 180) diff -= 360;
            while (diff <= -180) diff += 360;
            unwrapped.push(unwrapped[i - 1] + diff);
          }
          gdPhase = unwrapped;
        }

        const { freqOut, gdMs } = computeGroupDelay(gdFreq, gdPhase);
        if (gen !== renderGen) return;
        renderGdChart(freqOut, gdMs);
        return;
      }

      // IR or Step: compute via IPC
      const b = bands[0];
      const freq = [...b.measurement!.freq];
      const magnitude = [...b.measurement!.magnitude];
      const phase = [...b.measurement!.phase!];
      const sr = b.measurement!.sample_rate ?? 48000;

      let resultFreq = freq;
      let resultMag = magnitude;
      let resultPhase = phase;

      // SUM mode: coherent sum in complex domain
      if (sumMode && bands.length > 1) {
        const n = freq.length;
        const sumRe = new Float64Array(n);
        const sumIm = new Float64Array(n);
        for (const sb of bands) {
          const sign = sb.inverted ? -1 : 1;
          for (let j = 0; j < n; j++) {
            const amp = Math.pow(10, (sb.measurement!.magnitude[j] ?? -200) / 20) * sign;
            const phRad = (sb.measurement!.phase![j] ?? 0) * Math.PI / 180;
            sumRe[j] += amp * Math.cos(phRad);
            sumIm[j] += amp * Math.sin(phRad);
          }
        }
        resultMag = [];
        resultPhase = [];
        for (let j = 0; j < n; j++) {
          const amplitude = Math.sqrt(sumRe[j] * sumRe[j] + sumIm[j] * sumIm[j]);
          resultMag.push(amplitude > 0 ? 20 * Math.log10(amplitude) : -200);
          resultPhase.push(Math.atan2(sumIm[j], sumRe[j]) * 180 / Math.PI);
        }
      }

      const result = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
        freq: resultFreq, magnitude: resultMag, phase: resultPhase, sampleRate: sr,
      });
      if (gen !== renderGen) return;

      // Target impulse (if target enabled)
      let targetResult: { time: number[]; impulse: number[]; step: number[] } | null = null;
      if (!sumMode && band && band.targetEnabled) {
        try {
          const targetCurve = JSON.parse(JSON.stringify(band.target));
          // Auto-ref: average measurement level in 200-2000 Hz
          let sum = 0, n = 0;
          for (let i = 0; i < freq.length; i++) {
            if (freq[i] >= 200 && freq[i] <= 2000) { sum += magnitude[i]; n++; }
          }
          targetCurve.reference_level_db += n > 0 ? sum / n : 0;
          const tResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: targetCurve, freq });
          if (gen !== renderGen) return;
          targetResult = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
            freq, magnitude: tResp.magnitude, phase: tResp.phase, sampleRate: sr,
          });
          if (gen !== renderGen) return;
        } catch (_) {}
      }

      // Corrected impulse (meas + PEQ + cross-section)
      let corrResult: { time: number[]; impulse: number[]; step: number[] } | null = null;
      if (!sumMode && band && band.targetEnabled && band.peqBands?.length > 0) {
        try {
          const targetCurve = JSON.parse(JSON.stringify(band.target));
          let sum2 = 0, n2 = 0;
          for (let i = 0; i < freq.length; i++) { if (freq[i] >= 200 && freq[i] <= 2000) { sum2 += magnitude[i]; n2++; } }
          targetCurve.reference_level_db += n2 > 0 ? sum2 / n2 : 0;
          const tResp2 = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: targetCurve, freq });
          if (gen !== renderGen) return;
          const peqBands = band.peqBands.filter((p: any) => p.enabled);
          let corrMag = [...magnitude];
          let corrPh = [...phase];
          if (peqBands.length > 0) {
            const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq, bands: peqBands });
            if (gen !== renderGen) return;
            corrMag = corrMag.map((v, i) => v + pm[i]);
            corrPh = corrPh.map((v, i) => v + pp[i]);
          }
          // Add cross-section (HP/LP filter response)
          if (band.target.high_pass || band.target.low_pass) {
            const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
              freq, highPass: band.target.high_pass, lowPass: band.target.low_pass,
            });
            if (gen !== renderGen) return;
            corrMag = corrMag.map((v, i) => v + xm[i]);
            corrPh = corrPh.map((v, i) => v + xp[i]);
          }
          corrResult = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
            freq, magnitude: corrMag, phase: corrPh, sampleRate: sr,
          });
          if (gen !== renderGen) return;
        } catch (_) {}
      }

      // HP freq for masking zone
      const hpFreq = band?.target?.high_pass?.freq_hz ?? 20;

      const timeMs = result.time.map(t => t * 1000);
      const targetTimeMs = targetResult?.time.map(t => t * 1000) ?? null;
      const corrTimeMs = corrResult?.time.map(t => t * 1000) ?? null;
      renderIrStepChart(timeMs, result.impulse, result.step, targetTimeMs, targetResult?.impulse ?? null, targetResult?.step ?? null, corrTimeMs, corrResult?.impulse ?? null, corrResult?.step ?? null, hpFreq, irCfg);
    } catch (e) {
      console.error("Time tab render failed:", e);
    }
  }

  function renderIrStepChart(
    timeMs: number[], impulse: number[], step: number[],
    targetTimeMs: number[] | null, targetImpulse: number[] | null, targetStep: number[] | null,
    corrTimeMs: number[] | null, corrImpulse: number[] | null, corrStep: number[] | null,
    hpFreq: number,
    irCfg: { db: boolean; measIr: boolean; measStep: boolean; targetIr: boolean; targetStep: boolean; corrIr: boolean; corrStep: boolean; masking: boolean },
  ) {
    try { try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; } } catch (_) { chart = undefined; }
    if (!containerRef) return;
    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 200);
    const isDb = irCfg.db;
    const toDb = (v: number) => { const a = Math.abs(v); return a > 1e-10 ? 20 * Math.log10(a) : -200; };

    // Normalize by impulse peak
    let irPeak = 0, peakIdx = 0;
    for (let i = 0; i < impulse.length; i++) { if (Math.abs(impulse[i]) > irPeak) { irPeak = Math.abs(impulse[i]); peakIdx = i; } }
    if (irPeak < 1e-20) irPeak = 1;
    const normIr = impulse.map(v => v / irPeak);
    const normSt = step.map(v => { let p = 0; for (const s of step) { if (Math.abs(s) > p) p = Math.abs(s); } return p > 1e-20 ? v / p : 0; });
    const peakTimeMs = timeMs[peakIdx];

    // Masking duration: 1.5 periods of HP freq
    const maskingMs = hpFreq > 0 ? (1.5 / hpFreq) * 1000 : 20;

    // Build series
    const uSeries: uPlot.Series[] = [{}];
    const uDataArr: number[][] = [timeMs];

    // Helper: align and resample another impulse onto timeMs grid
    const alignAndResample = (srcTime: number[], srcData: number[], srcPeak?: number) => {
      let sp = srcPeak ?? 0, spIdx = 0;
      for (let i = 0; i < srcData.length; i++) { if (Math.abs(srcData[i]) > sp) { sp = Math.abs(srcData[i]); spIdx = i; } }
      if (sp < 1e-20) sp = 1;
      const shift = peakTimeMs - srcTime[spIdx];
      return timeMs.map(t => {
        const st = t - shift;
        if (st <= srcTime[0] || st >= srcTime[srcTime.length - 1]) return isDb ? -200 : 0;
        let lo = 0, hi = srcTime.length - 1;
        while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (srcTime[mid] <= st) lo = mid; else hi = mid; }
        const frac = (st - srcTime[lo]) / (srcTime[hi] - srcTime[lo]);
        const v = (srcData[lo] + frac * (srcData[hi] - srcData[lo])) / sp;
        return isDb ? toDb(v) : v;
      });
    };

    const emptyData = timeMs.map(() => isDb ? -200 : 0);

    // Series 1: Measurement IR
    uSeries.push({ label: "Meas IR", stroke: "#4A9EFF", width: 1.5, scale: "y", show: irCfg.measIr });
    uDataArr.push(isDb ? normIr.map(toDb) : normIr);

    // Series 2: Measurement Step
    uSeries.push({ label: "Meas Step", stroke: "#22C55E", width: 1.5, scale: "y", show: irCfg.measStep });
    uDataArr.push(isDb ? normSt.map(toDb) : normSt);

    // Series 3: Target IR
    uSeries.push({ label: "Target IR", stroke: "#FFD700", width: 1.5, dash: [6, 3], scale: "y", show: irCfg.targetIr });
    uDataArr.push(targetTimeMs && targetImpulse ? alignAndResample(targetTimeMs, targetImpulse) : emptyData);

    // Series 4: Target Step
    uSeries.push({ label: "Target Step", stroke: "#B8960A", width: 1.5, dash: [6, 3], scale: "y", show: irCfg.targetStep });
    uDataArr.push(targetTimeMs && targetStep ? alignAndResample(targetTimeMs, targetStep) : emptyData);

    // Series 5: Corrected IR
    uSeries.push({ label: "Corr IR", stroke: "#F97316", width: 1.5, scale: "y", show: irCfg.corrIr });
    uDataArr.push(corrTimeMs && corrImpulse ? alignAndResample(corrTimeMs, corrImpulse) : emptyData);

    // Series 6: Corrected Step
    uSeries.push({ label: "Corr Step", stroke: "#D97706", width: 1.5, scale: "y", show: irCfg.corrStep });
    uDataArr.push(corrTimeMs && corrStep ? alignAndResample(corrTimeMs, corrStep) : emptyData);

    // Y range
    let yMin = Infinity, yMax = -Infinity;
    for (let s = 1; s < uDataArr.length; s++) {
      for (const v of uDataArr[s]) { if (v > -190 && v < yMin) yMin = v; if (v > -190 && v > yMax) yMax = v; }
    }
    if (!isFinite(yMin)) { yMin = isDb ? -80 : -1.1; yMax = isDb ? 0 : 1.1; }
    const pad = isDb ? 5 : Math.max(0.05, (yMax - yMin) * 0.05);

    setShowLegend(false);
    setCursorValues([]);

    const opts: uPlot.Options = {
      width: w, height: h,
      series: uSeries,
      scales: {
        // Center peak in view
        x: (() => {
          const range = timeMs[timeMs.length - 1] - timeMs[0];
          const halfView = Math.max(range * 0.6, 5);
          return { min: peakTimeMs - halfView, max: peakTimeMs + halfView };
        })(),
        y: { auto: false, range: [isDb ? Math.max(yMin - pad, -80) : yMin - pad, yMax + pad] as uPlot.Range.MinMax },
      },
      axes: [
        { label: "ms", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)) },
        { label: isDb ? "dBFS" : "%", scale: "y", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : isDb ? v.toFixed(0) : (v * 100).toFixed(0)), size: 50 },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
      hooks: {
        draw: irCfg.masking ? [(u: uPlot) => {
          const ctx = u.ctx;
          const plotLeft = u.bbox.left;
          const plotTop = u.bbox.top;
          const plotHeight = u.bbox.height;
          const peakX = u.valToPos(peakTimeMs, "x", true);
          const maskStartX = u.valToPos(peakTimeMs - maskingMs, "x", true);
          const clampedMaskStart = Math.max(maskStartX, plotLeft);

          ctx.save();
          if (peakX > clampedMaskStart) {
            // Green safe wedge
            const nSteps = 30;
            const maskWidthX = peakX - clampedMaskStart;
            if (isDb) {
              ctx.fillStyle = "rgba(34, 197, 94, 0.10)";
              ctx.beginPath();
              for (let s = 0; s <= nSteps; s++) {
                const t = s / nSteps;
                const db = -40 - 40 * t;
                if (s === 0) ctx.moveTo(peakX, u.valToPos(db, "y", true));
                else ctx.lineTo(peakX - t * maskWidthX, u.valToPos(db, "y", true));
              }
              const yBot = u.valToPos(-80, "y", true);
              ctx.lineTo(clampedMaskStart, yBot); ctx.lineTo(peakX, yBot);
              ctx.closePath(); ctx.fill();
              // Yellow caution
              ctx.fillStyle = "rgba(234, 179, 8, 0.08)";
              ctx.beginPath();
              for (let s = 0; s <= nSteps; s++) {
                const t = s / nSteps;
                const db = -26 - 40 * t;
                if (s === 0) ctx.moveTo(peakX, u.valToPos(db, "y", true));
                else ctx.lineTo(peakX - t * maskWidthX, u.valToPos(db, "y", true));
              }
              ctx.lineTo(clampedMaskStart, yBot); ctx.lineTo(peakX, yBot);
              ctx.closePath(); ctx.fill();
            } else {
              // Linear mode wedges
              ctx.fillStyle = "rgba(234, 179, 8, 0.08)";
              ctx.beginPath();
              for (let s = 0; s <= nSteps; s++) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(0.05 * Math.exp(-3 * t), "y", true)); }
              for (let s = nSteps; s >= 0; s--) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(-0.05 * Math.exp(-3 * t), "y", true)); }
              ctx.closePath(); ctx.fill();
              ctx.fillStyle = "rgba(34, 197, 94, 0.10)";
              ctx.beginPath();
              for (let s = 0; s <= nSteps; s++) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(0.01 * Math.exp(-4 * t), "y", true)); }
              for (let s = nSteps; s >= 0; s--) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(-0.01 * Math.exp(-4 * t), "y", true)); }
              ctx.closePath(); ctx.fill();
            }
          }
          // Red zone before masking
          if (maskStartX > plotLeft + 2) {
            ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
            ctx.fillRect(plotLeft, plotTop, maskStartX - plotLeft, plotHeight);
          }
          ctx.restore();
        }] : [],
        setCursor: [(u: uPlot) => {
          const idx = u.cursor.idx;
          if (idx == null || idx < 0 || idx >= u.data[0].length) { setCursorFreq("—"); setCursorSPL("—"); return; }
          setCursorFreq(u.data[0][idx]?.toFixed(2) + " ms");
          const ir = u.data[1]?.[idx];
          const st = u.data[2]?.[idx];
          if (isDb) {
            setCursorSPL((ir != null && (ir as number) > -190 ? `IR: ${(ir as number).toFixed(0)} dB` : "") + (st != null && (st as number) > -190 ? ` Step: ${(st as number).toFixed(0)} dB` : ""));
          } else {
            setCursorSPL((ir != null ? `IR: ${((ir as number) * 100).toFixed(1)}%` : "") + (st != null ? ` Step: ${((st as number) * 100).toFixed(1)}%` : ""));
          }
        }],
      },
    };
    try { chart = new uPlot(opts, uDataArr as uPlot.AlignedData, containerRef); } catch (e) { console.error(e); }
  }

  function renderGdChart(freq: number[], gdMs: number[]) {
    try { try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; } } catch (_) { chart = undefined; }
    if (!containerRef) return;
    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 200);

    let yMin = Infinity, yMax = -Infinity;
    for (const v of gdMs) { if (isFinite(v)) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; } }
    const pad = Math.max(0.5, (yMax - yMin) * 0.1);
    if (!isFinite(yMin)) { yMin = -5; yMax = 20; }

    setShowLegend(false);
    setCursorValues([]);

    const opts: uPlot.Options = {
      width: w, height: h,
      series: [{}, { label: "GD ms", stroke: "#F59E0B", width: 1.5, scale: "y" }],
      scales: {
        x: { min: 20, max: 20000, distr: 3 },
        y: { auto: false, range: [yMin - pad, yMax + pad] as uPlot.Range.MinMax },
      },
      axes: [
        { stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v >= 1000 ? (v/1000)+"k" : String(Math.round(v))) },
        { label: "ms", scale: "y", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)), size: 50 },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
      hooks: {
        setCursor: [(u: uPlot) => {
          const idx = u.cursor.idx;
          if (idx == null || idx < 0 || idx >= u.data[0].length) { setCursorFreq("—"); setCursorSPL("—"); return; }
          const f = u.data[0][idx];
          setCursorFreq(f != null ? (f >= 1000 ? (f/1000).toFixed(2) + " kHz" : Math.round(f) + " Hz") : "—");
          const gd = u.data[1]?.[idx];
          setCursorSPL(gd != null ? (gd as number).toFixed(2) + " ms" : "—");
        }],
      },
    };
    try { chart = new uPlot(opts, [freq, gdMs], containerRef); } catch (e) { console.error(e); }
  }

  // ----------------------------------------------------------------
  // Single band rendering
  // ----------------------------------------------------------------
  async function renderBandMode(band: BandState, showPhase: boolean, showMag: boolean, showTarget: boolean) {
    const gen = ++renderGen;
    zoomCenter = 0; // reset before async — will be recalculated from measurement
    try {
      const result = await evaluateBand(band, showPhase);
      if (gen !== renderGen) return; // stale render, discard

      if (!result.freq) {
        try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
        setShowLegend(false);
        setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
        return;
      }

      // Compute zoom anchor: avg magnitude in passband (adaptive to HP/LP), or 0 dB without measurement
      if (result.measurement) {
        // Determine passband from actual HP/LP filters
        const hpFreq = band.target.high_pass?.freq_hz ?? 20;
        const lpFreq = band.target.low_pass?.freq_hz ?? 20000;
        const pbLow = Math.max(20, hpFreq * 1.5);
        const pbHigh = Math.min(20000, lpFreq * 0.7);
        // Fallback to 200-2000 if no filters or range is empty
        const effLow = pbLow < pbHigh ? pbLow : 200;
        const effHigh = pbLow < pbHigh ? pbHigh : 2000;

        let s = 0, n = 0;
        for (let i = 0; i < result.measurement.freq.length; i++) {
          if (result.measurement.freq[i] >= effLow && result.measurement.freq[i] <= effHigh) {
            s += result.measurement.magnitude[i]; n++;
          }
        }
        zoomCenter = n > 0 ? s / n : 0;
      } else {
        zoomCenter = 0;
      }

      // Строим серии для одной полосы
      const uSeries: uPlot.Series[] = [{}];
      const uData: number[][] = [result.freq];
      const legend: LegendEntry[] = [];
      let sIdx = 1;

      const isTargetTab = activeTab() === "target";
      const measVisible = !isTargetTab; // hide raw measurement on target tab (PEQ curves shown instead)

      // Derive color family from band color
      const cf = bandColorFamily(band.color);

      if (result.measurement && showMag) {
        uSeries.push({ label: result.measurement.name + " dB", stroke: cf.meas, width: 1.5, scale: "mag" });
        uData.push(result.measurement.magnitude);
        legend.push({ label: "Measurement", color: cf.meas, dash: false, visible: measVisible, seriesIdx: sIdx, category: "measurement" });
        sIdx++;

        if (showPhase && result.measurement.phase) {
          uSeries.push({ label: result.measurement.name + " \u00B0", stroke: cf.measPhase, width: 1, dash: [6, 3], scale: "phase" });
          uData.push(wrapPhase(result.measurement.phase));
          legend.push({ label: "Meas \u00B0", color: cf.measPhase, dash: true, visible: measVisible, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        }
      }

      if (showTarget && result.targetMag && result.targetMag.length > 0) {
        uSeries.push({ label: "Target dB", stroke: cf.target, width: 1.5, dash: [8, 4], scale: "mag" });
        uData.push(result.targetMag);
        legend.push({ label: "Target", color: cf.target, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
        sIdx++;
      }

      if (showTarget && result.targetPhase && result.targetPhase.length > 0 && showPhase) {
        const phase = band.inverted
          ? result.targetPhase.map((v) => v + 180)
          : result.targetPhase;
        uSeries.push({ label: "Target \u00B0", stroke: cf.targetPhase, width: 1, dash: [4, 4], scale: "phase" });
        uData.push(wrapPhase(phase));
        legend.push({ label: "Target \u00B0", color: cf.targetPhase, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
        sIdx++;
      }

      // Corrected curve = measurement + PEQ + cross-section (filters + makeup)
      const hasPeq = band.peqBands && band.peqBands.length > 0;
      const hasFilters = band.targetEnabled && (band.target.high_pass || band.target.low_pass);
      if (result.measurement && (hasPeq || hasFilters)) {
        try {
          const isHybrid = exportHybridPhase();

          // PEQ correction
          let peqMag: number[] | null = null;
          let peqPhase: number[] | null = null;
          if (hasPeq) {
            const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
              freq: result.measurement.freq,
              bands: band.peqBands,
            });
            peqMag = pm;
            peqPhase = pp;
          }

          // Cross-section: filters + min-phase makeup where corrected < target
          let xsMag: number[] | null = null;
          let xsPhase: number[] | null = null;
          if (hasFilters && result.targetMag) {
            const [xm, xp, xNorm] = await invoke<[number[], number[], number]>("compute_cross_section", {
              freq: result.measurement.freq,
              highPass: band.target.high_pass,
              lowPass: band.target.low_pass,
            });
            xsMag = xm;
            xsPhase = xp;
          }

          // Full corrected = measurement + PEQ + cross-section
          // Hybrid: amber "Corrected + XO", Standard: green "Corrected"
          const fullCorrected = result.measurement.magnitude.map(
            (v: number, i: number) =>
              v + (peqMag ? peqMag[i] : 0) + (xsMag ? xsMag[i] : 0)
          );

          // Normalize corrected to target in passband (b82.06)
          if (result.targetMag) {
            const hpF = band.target.high_pass?.freq_hz ?? 20;
            const lpF = band.target.low_pass?.freq_hz ?? 20000;
            const pbL = Math.max(20, hpF * 1.5);
            const pbH = Math.min(20000, lpF * 0.7);
            const eL = pbL < pbH ? pbL : 200;
            const eH = pbL < pbH ? pbH : 2000;
            let dSum = 0, dN = 0;
            for (let k = 0; k < result.measurement.freq.length; k++) {
              if (result.measurement.freq[k] >= eL && result.measurement.freq[k] <= eH) {
                dSum += result.targetMag[k] - fullCorrected[k];
                dN++;
              }
            }
            const corrOffset = dN > 0 ? dSum / dN : 0;
            if (Math.abs(corrOffset) > 0.01) {
              for (let k = 0; k < fullCorrected.length; k++) fullCorrected[k] += corrOffset;
            }
          }

          if (showMag) {
            const corrLabel = isHybrid ? "Corrected + XO" : "Corrected";
            uSeries.push({
              label: corrLabel + " dB",
              stroke: cf.corrected,
              width: 2.5,
              scale: "mag",
            });
            uData.push(fullCorrected);
            legend.push({ label: corrLabel, color: cf.corrected, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          }

          // Corrected phase = measurement phase + PEQ phase + cross-section phase
          let fullCorrectedPhase: number[] | null = null;
          if (result.measurement.phase) {
            fullCorrectedPhase = result.measurement.phase.map(
              (v: number, i: number) =>
                v + (peqPhase ? peqPhase[i] : 0) + (xsPhase ? xsPhase[i] : 0)
            );
            if (showPhase) {
              const phaseLabel = isHybrid ? "Corrected + XO" : "Corrected";
              uSeries.push({
                label: phaseLabel + " \u00B0",
                stroke: cf.correctedPhase,
                width: 1.5,
                dash: [4, 4],
                scale: "phase",
              });
              uData.push(wrapPhase(fullCorrectedPhase));
              legend.push({ label: phaseLabel + " \u00B0", color: cf.correctedPhase, dash: true, visible: true, seriesIdx: sIdx, category: "corrected" });
              sIdx++;
            }
          }

          // Save corrected data for snapshot capture (raw, pre-normalization)
          lastCorrData.freq = result.freq!;
          lastCorrData.mag = [...fullCorrected];
          lastCorrData.phase = fullCorrectedPhase ? wrapPhase(fullCorrectedPhase) : [];
        } catch (e) {
          console.warn("Correction computation failed:", e);
        }
      } else {
        // No corrected curve — clear snapshot capture data
        lastCorrData.freq = [];
        lastCorrData.mag = [];
        lastCorrData.phase = [];
      }

      // Floor bounce
      let floorBounceNulls: number[] | undefined;
      const fb = band.settings?.floorBounce;
      if (fb && fb.enabled) {
        const fbResult = computeFloorBounce(fb.speakerHeight, fb.micHeight, fb.distance);
        floorBounceNulls = fbResult.nullFreqs;
      }

      // --- Snapshot overlays (per-band) ---
      const snaps = freqSnapshots(band.id);
      for (const snap of snaps) {
        const interpolateArr = (srcArr: (number | null)[]): (number | null)[] => {
          if (snap.freq.length === result.freq!.length && snap.freq[0] === result.freq![0]) {
            return srcArr;
          }
          return result.freq!.map(f => {
            let lo = 0, hi = snap.freq.length - 1;
            if (f <= snap.freq[0]) return srcArr[0];
            if (f >= snap.freq[hi]) return srcArr[hi];
            while (hi - lo > 1) {
              const mid = (lo + hi) >> 1;
              if (snap.freq[mid] <= f) lo = mid; else hi = mid;
            }
            const vLo = srcArr[lo], vHi = srcArr[hi];
            if (vLo == null || vHi == null) return null;
            const t = (f - snap.freq[lo]) / (snap.freq[hi] - snap.freq[lo]);
            return (vLo as number) + t * ((vHi as number) - (vLo as number));
          });
        };

        const snapMag = interpolateArr(snap.mag as (number | null)[]) as number[];
        const snapPhase = interpolateArr(snap.phase);

        // Magnitude series
        uSeries.push({
          label: `${snap.label} dB`,
          stroke: snap.color,
          width: 1.5,
          dash: [4, 3],
          scale: "mag",
        });
        uData.push(snapMag);
        legend.push({ label: snap.label, color: snap.color, dash: true, visible: true, seriesIdx: sIdx, category: "corrected" });
        sIdx++;

        // Phase series (same color, dotted)
        if (snapPhase.some(v => v != null)) {
          uSeries.push({
            label: `${snap.label} \u00B0`,
            stroke: snap.color,
            width: 1,
            dash: [2, 2],
            scale: "phase",
          });
          uData.push(snapPhase as number[]);
          legend.push({ label: `${snap.label} \u00B0`, color: snap.color, dash: true, visible: true, seriesIdx: sIdx, category: "corrected" });
          sIdx++;
        }
      }

      // Normalize all mag series to dBr (0 = passband level)
      if (zoomCenter !== 0) {
        for (let i = 0; i < uSeries.length; i++) {
          if ((uSeries[i] as any).scale === "mag") {
            uData[i] = uData[i].map((v: number) => v - zoomCenter);
          }
        }
      }
      zoomCenter = 0; // after normalization, center is 0 dBr

      if (gen !== renderGen) return;
      requestAnimationFrame(() => {
        if (!containerRef) return;
        renderChart({
          freq: result.freq!,
          uSeries,
          uData,
          hasMeasurements: !!result.measurement,
          legend,
          floorBounceNulls,
        });
      });
    } catch (e) {
      console.error("Band render failed:", e);
    }
  }

  // ----------------------------------------------------------------
  // SUM mode rendering
  // ----------------------------------------------------------------
  async function renderSumMode(showPhase: boolean, showMag: boolean, showTarget: boolean) {
    const gen = ++renderGen;
    zoomCenter = 0; // reset before async — will be recalculated from globalRef
    const bands: BandState[] = JSON.parse(JSON.stringify(appState.bands));

    try {
      const results = await Promise.all(bands.map((b) => evaluateBand(b, showPhase)));
      if (gen !== renderGen) return; // stale render, discard

      // Определяем общую частотную сетку (максимальное число точек, самый широкий диапазон)
      let commonFreq: number[] | null = null;
      let bestLen = 0;
      for (const r of results) {
        const f = r.measurement?.freq ?? r.freq;
        if (f && f.length > bestLen) { commonFreq = f; bestLen = f.length; }
      }
      if (!commonFreq) {
        for (const r of results) {
          if (r.freq) { commonFreq = r.freq; break; }
        }
      }
      if (!commonFreq || commonFreq.length === 0) {
        try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
        setShowLegend(false);
        setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
        return;
      }

      // Ensure SUM grid covers at least 20–20 kHz (extend via interpolation)
      const rawFMin = commonFreq[0];
      const rawFMax = commonFreq[commonFreq.length - 1];
      const fMin = Math.min(rawFMin, 20);
      const fMax = Math.max(rawFMax, 20000);
      const nPts = commonFreq.length;
      // If grid needs extending, all bands will be resampled via interpolate_log
      const needResample = fMin < rawFMin || fMax > rawFMax;

      // Build the common frequency grid (may be extended)
      let freq: number[];
      if (needResample) {
        // Generate a new log-spaced grid covering the full range
        const [newFreq] = await invoke<[number[], number[], number[] | null]>(
          "interpolate_log",
          { freq: commonFreq, magnitude: commonFreq.map(() => 0), phase: null, nPoints: nPts, fMin, fMax }
        );
        freq = newFreq;
      } else {
        freq = commonFreq;
      }

      // Ресемплируем каждый замер на общую сетку (если нужно)
      interface ResampledMeas {
        magnitude: number[];
        phase: number[] | null;
        name: string;
      }
      const resampled: (ResampledMeas | null)[] = [];
      for (const r of results) {
        if (!r.measurement) { resampled.push(null); continue; }
        const m = r.measurement;
        if (!needResample && m.freq.length === nPts && m.freq[0] === fMin && m.freq[m.freq.length - 1] === fMax) {
          // Уже на общей сетке
          resampled.push({ magnitude: m.magnitude, phase: m.phase ?? null, name: m.name });
        } else {
          // Интерполируем на общую сетку
          try {
            const [, intMag, intPhase] = await invoke<[number[], number[], number[] | null]>(
              "interpolate_log",
              { freq: m.freq, magnitude: m.magnitude, phase: m.phase, nPoints: nPts, fMin, fMax }
            );
            resampled.push({ magnitude: intMag, phase: intPhase, name: m.name });
          } catch (e) {
            console.warn("Interpolation failed for", m.name, e);
            resampled.push(null);
          }
        }
      }

      // Общий reference level: среднее самого громкого бэнда в 200–2000 Hz
      // (субвуфер и другие бэнды за пределами passband не сбивают reference)
      let globalRef = 0;
      {
        let bestAvg = -Infinity;
        for (const rm of resampled) {
          if (!rm) continue;
          let s = 0, n = 0;
          for (let k = 0; k < freq.length; k++) {
            if (freq[k] >= 200 && freq[k] <= 2000) { s += rm.magnitude[k]; n++; }
          }
          if (n > 0) {
            const avg = s / n;
            if (avg > bestAvg) bestAvg = avg;
          }
        }
        if (isFinite(bestAvg)) globalRef = bestAvg;
      }

      // Пересчитываем ВСЕ включённые таргеты на общую сетку
      const perBandTargetMags: (number[] | null)[] = [];
      const perBandTargetNorm: (number[] | null)[] = [];
      const perBandTargetNormPhase: (number[] | null)[] = [];
      const refLevels: number[] = [];
      for (let i = 0; i < bands.length; i++) {
        if (!bands[i].targetEnabled) {
          perBandTargetMags.push(null);
          perBandTargetNorm.push(null);
          perBandTargetNormPhase.push(null);
          continue;
        }
        const targetCurve = JSON.parse(JSON.stringify(bands[i].target));
        const totalRef = targetCurve.reference_level_db + globalRef;
        refLevels.push(totalRef);

        const curveWithRef = { ...targetCurve, reference_level_db: totalRef };
        const response = await invoke<TargetResponse>("evaluate_target", {
          target: curveWithRef, freq,
        });
        perBandTargetMags.push(response.magnitude);

        const curveNoRef = { ...targetCurve, reference_level_db: 0 };
        const responseNorm = await invoke<TargetResponse>("evaluate_target", {
          target: curveNoRef, freq,
        });
        perBandTargetNorm.push(responseNorm.magnitude);
        perBandTargetNormPhase.push(responseNorm.phase);
      }

      // Строим uPlot series + data + легенду
      const uSeries: uPlot.Series[] = [{}];
      const uData: number[][] = [freq];
      const legend: LegendEntry[] = [];
      let sIdx = 1;

      // --- Per-band замеры (ресемплированные) ---
      const measIndices: number[] = [];
      for (let i = 0; i < resampled.length; i++) {
        if (resampled[i]) measIndices.push(i);
      }
      if (showMag) {
        for (const i of measIndices) {
          const rm = resampled[i]!;
          const bcf = bandColorFamily(bands[i].color);
          uSeries.push({ label: bands[i].name + " dB", stroke: bcf.meas, width: 1, scale: "mag" });
          uData.push(rm.magnitude);
          legend.push({ label: bands[i].name, color: bcf.meas, dash: false, visible: false, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
          if (showPhase && rm.phase) {
            uSeries.push({ label: bands[i].name + " °", stroke: bcf.measPhase, width: 1, dash: [6, 3], scale: "phase" });
            uData.push(wrapPhase(rm.phase));
            legend.push({ label: bands[i].name + " °", color: bcf.measPhase, dash: true, visible: false, seriesIdx: sIdx, category: "measurement" });
            sIdx++;
          }
        }
      }

      // --- Per-band таргеты ---
      if (showTarget) {
        for (let i = 0; i < bands.length; i++) {
          const tMag = perBandTargetMags[i];
          if (!tMag) continue;
          const color = bandColorFamily(bands[i].color).target;
          uSeries.push({ label: bands[i].name + " tgt", stroke: color, width: 1, dash: [6, 4], scale: "mag" });
          uData.push(tMag);
          legend.push({ label: bands[i].name + " tgt", color, dash: true, visible: false, seriesIdx: sIdx, category: "target" });
          sIdx++;
        }
      }

      // --- Per-band corrected кривые (measurement + PEQ + filters) ---
      // Also store corrected phase (meas + PEQ + XO) for coherent Σ
      const perBandCorrected: (number[] | null)[] = [];
      const perBandCorrPhase: (number[] | null)[] = [];
      if (showMag) {
        for (let i = 0; i < bands.length; i++) {
          const rm = resampled[i];
          const tMag = perBandTargetMags[i];
          if (!rm) { perBandCorrected.push(null); perBandCorrPhase.push(null); continue; }

          const hasPeq = bands[i].peqBands && bands[i].peqBands.length > 0;
          const hasFilters = bands[i].targetEnabled && (bands[i].target.high_pass || bands[i].target.low_pass);

          if (!hasPeq && !hasFilters) { perBandCorrected.push(null); perBandCorrPhase.push(null); continue; }

          try {
            let peqMag: number[] | null = null;
            let peqPhase: number[] | null = null;
            if (hasPeq) {
              const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
                freq, bands: bands[i].peqBands,
              });
              peqMag = pm;
              peqPhase = pp;
            }

            let xsMag: number[] | null = null;
            let xsPhase: number[] | null = null;
            if (hasFilters && tMag) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq,
                highPass: bands[i].target.high_pass,
                lowPass: bands[i].target.low_pass,
              });
              xsMag = xm;
              xsPhase = xp;
            }

            const corrected = rm.magnitude.map(
              (v: number, j: number) => v + (peqMag ? peqMag[j] : 0) + (xsMag ? xsMag[j] : 0)
            );
            // Normalize per-band corrected to its target in passband (b82.07)
            if (tMag) {
              const hpF = bands[i].target.high_pass?.freq_hz ?? 20;
              const lpF = bands[i].target.low_pass?.freq_hz ?? 20000;
              const pbL = Math.max(20, hpF * 1.5);
              const pbH = Math.min(20000, lpF * 0.7);
              const eL = pbL < pbH ? pbL : 200;
              const eH = pbL < pbH ? pbH : 2000;
              let dS = 0, dN = 0;
              for (let j = 0; j < freq.length; j++) {
                if (freq[j] >= eL && freq[j] <= eH) {
                  dS += tMag[j] - corrected[j];
                  dN++;
                }
              }
              const off = dN > 0 ? dS / dN : 0;
              if (Math.abs(off) > 0.01) {
                for (let j = 0; j < corrected.length; j++) corrected[j] += off;
              }
            }
            perBandCorrected.push(corrected);

            // Corrected phase = measurement phase + PEQ phase + XO phase
            if (rm.phase) {
              const corrPhase = rm.phase.map(
                (v: number, j: number) => v + (peqPhase ? peqPhase[j] : 0) + (xsPhase ? xsPhase[j] : 0)
              );
              perBandCorrPhase.push(corrPhase);
            } else {
              perBandCorrPhase.push(null);
            }

            const color = bandColorFamily(bands[i].color).corrected;
            uSeries.push({ label: bands[i].name + " corr+XO", stroke: color, width: 2, scale: "mag" });
            uData.push(corrected);
            legend.push({ label: bands[i].name + " corr+XO", color, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          } catch (e) {
            console.warn("SUM corrected failed for band", bands[i].name, e);
            perBandCorrected.push(null);
            perBandCorrPhase.push(null);
          }
        }

        // --- Суммарный таргет (когерентное сложение — вычисляем ДО Σ corrected для нормализации) ---
        let sumTargetArr: number[] | null = null;
        {
          const enabledNorm: number[][] = [];
          const enabledPhase: number[][] = [];
          const enabledInverted: boolean[] = [];
          for (let i = 0; i < bands.length; i++) {
            if (perBandTargetNorm[i]) {
              enabledNorm.push(perBandTargetNorm[i]!);
              enabledPhase.push(perBandTargetNormPhase[i] ?? Array.from({ length: freq.length }, () => 0));
              enabledInverted.push(bands[i].inverted);
            }
          }
          if (enabledNorm.length > 0) {
            const avgRef = refLevels.length > 0
              ? refLevels.reduce((a, b) => a + b, 0) / refLevels.length
              : 0;
            const sumRe = new Float64Array(freq.length);
            const sumIm = new Float64Array(freq.length);
            for (let n = 0; n < enabledNorm.length; n++) {
              const mag = enabledNorm[n];
              const ph = enabledPhase[n];
              const sign = enabledInverted[n] ? -1 : 1;
              for (let j = 0; j < freq.length; j++) {
                const amp = Math.pow(10, mag[j] / 20) * sign;
                const phRad = ph[j] * Math.PI / 180;
                sumRe[j] += amp * Math.cos(phRad);
                sumIm[j] += amp * Math.sin(phRad);
              }
            }
            const st = new Array(freq.length);
            const stPhase = new Array(freq.length);
            for (let j = 0; j < freq.length; j++) {
              const re = sumRe[j];
              const im = sumIm[j];
              const amplitude = Math.sqrt(re * re + im * im);
              st[j] = amplitude > 0 ? 20 * Math.log10(amplitude) + avgRef : -200;
              stPhase[j] = Math.atan2(im, re) * 180 / Math.PI;
            }
            sumTargetArr = st;

            if (showTarget) {
              uSeries.push({ label: "\u03A3 tgt", stroke: SUM_TARGET_COLOR, width: 2.5, dash: [8, 4], scale: "mag" });
              uData.push(st);
              legend.push({ label: "\u03A3 target", color: SUM_TARGET_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
              sIdx++;
              if (showPhase) {
                uSeries.push({ label: "\u03A3 tgt \u00B0", stroke: SUM_TARGET_PHASE_COLOR, width: 1.5, dash: [4, 4], scale: "phase" });
                uData.push(wrapPhase(stPhase));
                legend.push({ label: "\u03A3 target \u00B0", color: SUM_TARGET_PHASE_COLOR, dash: true, visible: false, seriesIdx: sIdx, category: "target" });
                sIdx++;
              }
            }
          }
        }

        // Σ corrected (когерентное сложение с учётом полной фазы: замер + PEQ + фильтры)
        const corrIndices = perBandCorrected.map((c, i) => c ? i : -1).filter(i => i >= 0);
        if (corrIndices.length > 0) {
          const hasAllPhaseCorr = corrIndices.every(
            (ci) => perBandCorrPhase[ci] && perBandCorrPhase[ci]!.length === nPts
          );
          if (hasAllPhaseCorr) {
            const sumRe = new Float64Array(nPts);
            const sumIm = new Float64Array(nPts);
            for (const ci of corrIndices) {
              const corr = perBandCorrected[ci]!;
              const corrPh = perBandCorrPhase[ci]!;
              const sign = bands[ci].inverted ? -1 : 1;
              for (let j = 0; j < nPts; j++) {
                const amp = Math.pow(10, corr[j] / 20) * sign;
                const phRad = corrPh[j] * Math.PI / 180;
                sumRe[j] += amp * Math.cos(phRad);
                sumIm[j] += amp * Math.sin(phRad);
              }
            }
            const sumCorrDb = new Array(nPts);
            const sumCorrPhase = new Array(nPts);
            for (let j = 0; j < nPts; j++) {
              const re = sumRe[j];
              const im = sumIm[j];
              const amplitude = Math.sqrt(re * re + im * im);
              sumCorrDb[j] = amplitude > 0 ? 20 * Math.log10(amplitude) : -200;
              sumCorrPhase[j] = Math.atan2(im, re) * 180 / Math.PI;
            }
            // Normalize Σ corrected to Σ target in passband 200–2000 Hz (b82.07)
            if (sumTargetArr) {
              let dSum = 0, dN = 0;
              for (let j = 0; j < nPts; j++) {
                if (freq[j] >= 200 && freq[j] <= 2000) {
                  dSum += sumTargetArr[j] - sumCorrDb[j];
                  dN++;
                }
              }
              const corrOff = dN > 0 ? dSum / dN : 0;
              if (Math.abs(corrOff) > 0.01) {
                for (let j = 0; j < nPts; j++) sumCorrDb[j] += corrOff;
              }
            }
            uSeries.push({ label: "\u03A3 corr", stroke: SUM_CORRECTED_COLOR, width: 3, scale: "mag" });
            uData.push(sumCorrDb);
            legend.push({ label: "\u03A3 corrected", color: SUM_CORRECTED_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
            // Σ corrected phase
            if (showPhase) {
              uSeries.push({ label: "\u03A3 corr °", stroke: SUM_CORRECTED_COLOR, width: 1.5, dash: [6, 3], scale: "phase" });
              uData.push(wrapPhase(sumCorrPhase));
              legend.push({ label: "\u03A3 corr °", color: SUM_CORRECTED_COLOR, dash: true, visible: false, seriesIdx: sIdx, category: "corrected" });
              sIdx++;
            }
          } else {
            // Инкогерентное сложение (без фазы)
            const sumCorr = new Float64Array(nPts);
            for (const ci of corrIndices) {
              const corr = perBandCorrected[ci]!;
              const sign = bands[ci].inverted ? -1 : 1;
              for (let j = 0; j < nPts; j++) {
                sumCorr[j] += Math.pow(10, corr[j] / 20) * sign;
              }
            }
            const sumCorrDb = Array.from(sumCorr, (v: number) =>
              Math.abs(v) > 0 ? 20 * Math.log10(Math.abs(v)) : -200
            );
            // Normalize incoherent Σ corrected to Σ target in passband (b82.07)
            if (sumTargetArr) {
              let dSum = 0, dN = 0;
              for (let j = 0; j < nPts; j++) {
                if (freq[j] >= 200 && freq[j] <= 2000) {
                  dSum += sumTargetArr[j] - sumCorrDb[j];
                  dN++;
                }
              }
              const corrOff = dN > 0 ? dSum / dN : 0;
              if (Math.abs(corrOff) > 0.01) {
                for (let j = 0; j < nPts; j++) sumCorrDb[j] += corrOff;
              }
            }
            uSeries.push({ label: "\u03A3 corr", stroke: SUM_CORRECTED_COLOR, width: 3, scale: "mag" });
            uData.push(sumCorrDb);
            legend.push({ label: "\u03A3 corrected", color: SUM_CORRECTED_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          }
        }
      }

      // --- Суммарный замер (когерентное сложение с учётом фазы и инверсии) ---
      if (showMag && measIndices.length > 0) {
        const hasAllPhase = measIndices.every(
          (mi) => resampled[mi]!.phase && resampled[mi]!.phase!.length === nPts
        );
        if (hasAllPhase) {
          const sumRe = new Float64Array(nPts);
          const sumIm = new Float64Array(nPts);
          for (const mi of measIndices) {
            const rm = resampled[mi]!;
            const sign = bands[mi].inverted ? -1 : 1;
            for (let j = 0; j < nPts; j++) {
              const amp = Math.pow(10, rm.magnitude[j] / 20) * sign;
              const phRad = rm.phase![j] * Math.PI / 180;
              sumRe[j] += amp * Math.cos(phRad);
              sumIm[j] += amp * Math.sin(phRad);
            }
          }
          const sumMagDb = new Array(nPts);
          const sumPhase = new Array(nPts);
          for (let j = 0; j < nPts; j++) {
            const re = sumRe[j];
            const im = sumIm[j];
            const amplitude = Math.sqrt(re * re + im * im);
            sumMagDb[j] = amplitude > 0 ? 20 * Math.log10(amplitude) : -200;
            sumPhase[j] = Math.atan2(im, re) * 180 / Math.PI;
          }
          uSeries.push({ label: "\u03A3 dB", stroke: SUM_MEAS_COLOR, width: 2, scale: "mag" });
          uData.push(sumMagDb);
          legend.push({ label: "\u03A3 meas", color: SUM_MEAS_COLOR, dash: false, visible: false, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
          // Σ measurement phase
          if (showPhase) {
            uSeries.push({ label: "\u03A3 °", stroke: SUM_MEAS_COLOR, width: 1.5, dash: [6, 3], scale: "phase" });
            uData.push(wrapPhase(sumPhase));
            legend.push({ label: "\u03A3 meas °", color: SUM_MEAS_COLOR, dash: true, visible: false, seriesIdx: sIdx, category: "measurement" });
            sIdx++;
          }
        } else {
          const sumMag = new Float64Array(nPts);
          for (const mi of measIndices) {
            const rm = resampled[mi]!;
            const sign = bands[mi].inverted ? -1 : 1;
            for (let j = 0; j < nPts; j++) {
              sumMag[j] += Math.pow(10, rm.magnitude[j] / 20) * sign;
            }
          }
          const sumMagDb = Array.from(sumMag, (v: number) =>
            Math.abs(v) > 0 ? 20 * Math.log10(Math.abs(v)) : -200
          );
          uSeries.push({ label: "\u03A3 dB", stroke: SUM_MEAS_COLOR, width: 2, scale: "mag" });
          uData.push(sumMagDb);
          legend.push({ label: "\u03A3 meas", color: SUM_MEAS_COLOR, dash: false, visible: false, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        }
      }

      // Collect crossover points for interactive markers (enrich with dB level)
      const crossovers = getCrossovers();
      for (const xo of crossovers) {
        const tMag = perBandTargetMags[xo.bandIndex];
        if (tMag && freq.length > 0) {
          // Find closest freq index to crossover frequency (log-space)
          let bestIdx = 0;
          let bestDist = Infinity;
          const logXo = Math.log10(xo.freq);
          for (let j = 0; j < freq.length; j++) {
            const d = Math.abs(Math.log10(freq[j]) - logXo);
            if (d < bestDist) { bestDist = d; bestIdx = j; }
          }
          xo.dbLevel = tMag[bestIdx];
        }
      }

      // Zoom anchor = globalRef (already computed from all measurements)
      zoomCenter = globalRef;

      // Normalize all mag series to dBr (0 = passband level)
      if (zoomCenter !== 0) {
        for (let i = 0; i < uSeries.length; i++) {
          if ((uSeries[i] as any).scale === "mag") {
            uData[i] = uData[i].map((v: number) => v - zoomCenter);
          }
        }
        // Also normalize crossover dB levels
        for (const xo of crossovers) {
          if (xo.dbLevel != null) xo.dbLevel -= zoomCenter;
        }
      }
      zoomCenter = 0; // after normalization, center is 0 dBr

      if (gen !== renderGen) return; // stale after async work
      requestAnimationFrame(() =>
        renderChart({
          freq,
          uSeries,
          uData,
          hasMeasurements: measIndices.length > 0,
          legend,
          crossovers,
        })
      );
    } catch (e) {
      console.error("SUM render failed:", e);
    }
  }

  // ----------------------------------------------------------------
  // Crossover mouse handlers (SUM mode only)
  // ----------------------------------------------------------------
  function xoHitTest(clientX: number, _clientY: number): number | null {
    if (!chart || currentCrossovers.length === 0) return null;
    // Use chart.over for coordinate mapping (it receives the mouse events)
    const overEl = chart.over ?? containerRef;
    const rect = overEl?.getBoundingClientRect();
    if (!rect) return null;
    const mx = clientX - rect.left;

    const HIT_RADIUS_X = 16; // CSS pixels — click anywhere on the vertical line
    for (let i = 0; i < currentCrossovers.length; i++) {
      const xo = currentCrossovers[i];
      const drg = draggingXo();
      const f = (drg === i && dragFreq() != null) ? dragFreq()! : xo.freq;
      const xPos = chart.valToPos(f, "x", false); // CSS pixels
      if (Math.abs(mx - xPos) < HIT_RADIUS_X) return i;
    }
    return null;
  }

  function handleXoMouseMove(e: MouseEvent) {
    if (!isSum() || draggingXo() != null) return;
    const hit = xoHitTest(e.clientX, e.clientY);
    const prev = hoveredXo();
    setHoveredXo(hit);
    if (chart?.over) {
      chart.over.style.cursor = hit != null ? "ew-resize" : "";
    }
    // Redraw to show hover state change
    if (hit !== prev && chart) chart.redraw(false, false);
  }

  function handleXoMouseDown(e: MouseEvent) {
    if (!isSum() || e.button !== 0) return;
    const hit = xoHitTest(e.clientX, e.clientY);
    if (hit == null) return;

    e.preventDefault();
    e.stopPropagation();
    setDraggingXo(hit);
    setDragFreq(currentCrossovers[hit].freq);

    const onMove = (ev: MouseEvent) => {
      if (!chart || draggingXo() == null) return;
      const overEl = chart.over ?? containerRef;
      const rect = overEl?.getBoundingClientRect();
      if (!rect) return;
      const mx = ev.clientX - rect.left;
      let newFreq = chart.posToVal(mx, "x");
      if (!isFinite(newFreq) || newFreq < 20) newFreq = 20;
      if (newFreq > 20000) newFreq = 20000;

      // Clamp between adjacent crossovers
      const xi = draggingXo()!;
      const MIN_GAP = 1.1; // minimum ratio between adjacent crossovers
      if (xi > 0) {
        const prevFreq = currentCrossovers[xi - 1].freq;
        if (newFreq < prevFreq * MIN_GAP) newFreq = prevFreq * MIN_GAP;
      }
      if (xi < currentCrossovers.length - 1) {
        const nextFreq = currentCrossovers[xi + 1].freq;
        if (newFreq > nextFreq / MIN_GAP) newFreq = nextFreq / MIN_GAP;
      }

      newFreq = Math.round(newFreq);
      setDragFreq(newFreq);
      // Only redraw marker overlay — no store update, no IPC
      if (chart) chart.redraw(false, false);
    };

    const onUp = (_ev: MouseEvent) => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);

      const xi = draggingXo();
      const finalFreq = dragFreq();
      setDraggingXo(null);
      setDragFreq(null);

      // Apply final frequency to store (triggers full re-render once)
      if (xi != null && finalFreq != null) {
        const xo = currentCrossovers[xi];
        if (xo) {
          setBandLowPass(xo.bandId, {
            filter_type: xo.filterType,
            order: xo.order,
            freq_hz: finalFreq,
            shape: xo.shape,
            linear_phase: xo.linearPhase,
            q: xo.q,
          });
        }
      }
      if (chart?.over) chart.over.style.cursor = "";
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function handleXoDblClick(e: MouseEvent) {
    if (!isSum()) return;
    const hit = xoHitTest(e.clientX, e.clientY);
    if (hit == null) return;

    e.preventDefault();
    e.stopPropagation();

    const xo = currentCrossovers[hit];
    openCrossoverDialog({
      bandIndex: xo.bandIndex,
      bandId: xo.bandId,
      bandName: xo.bandName,
      nextBandName: xo.nextBandName,
      freq: xo.freq,
      filterType: xo.filterType,
      order: xo.order,
      linearPhase: xo.linearPhase,
      shape: xo.shape,
      q: xo.q,
    });
  }

  // Double-click on chart → add PEQ band at cursor frequency (align tab only)
  function handlePeqDblClick(e: MouseEvent) {
    if (activeTab() !== "target") return;
    // Don't interfere with crossover dblclick in SUM mode
    if (isSum()) return;
    const bd = activeBand();
    if (!bd || !chart) return;

    const overEl = chart.over ?? containerRef;
    const rect = overEl?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    let freq = chart.posToVal(mx, "x");
    if (!isFinite(freq) || freq < 20) freq = 20;
    if (freq > 20000) freq = 20000;

    addPeqBand(bd.id, { freq_hz: Math.round(freq), gain_db: 0, q: 4.32, enabled: true, filter_type: "Peaking" });
    setSelectedPeqIdx(0); // new band is added at index 0
  }

  onCleanup(() => { if (chart) chart.destroy(); });

  return (
    <div class="plot-wrapper">
      <div class="plot-tabs-strip">
        <button class={`plot-tab ${plotTab() === "freq" ? "active" : ""}`} onClick={() => setPlotTab("freq")}>АЧХ</button>
        <button class={`plot-tab ${plotTab() === "ir" || plotTab() === "step" ? "active" : ""}`} onClick={() => setPlotTab("ir")}>IR/Step</button>
        <button class={`plot-tab ${plotTab() === "gd" ? "active" : ""}`} onClick={() => setPlotTab("gd")}>GD</button>
        <button class={`plot-tab ${plotTab() === "export" ? "active" : ""}`} onClick={() => setPlotTab("export")}>Export</button>
      </div>
      <div class="cursor-readout">
        <div class="readout-curves-area">
          <span class="readout-item">
            <span class="readout-label">Freq:</span>
            <span class="readout-value">{cursorFreq()}</span>
          </span>
          {/* Per-curve values at cursor — colored by curve */}
          <For each={cursorValues()}>
            {(cv) => (
              <span class="readout-item readout-curve-val">
                <span class="readout-curve-dot" style={{ background: cv.color }} />
                <span class="readout-value" style={{ color: cv.color }}>{cv.value}</span>
              </span>
            )}
          </For>
          <Show when={cursorValues().length === 0}>
            <span class="readout-item">
              <span class="readout-label">SPL:</span>
              <span class="readout-value">{cursorSPL()}</span>
            </span>
            <span class="readout-item">
              <span class="readout-label">Phase:</span>
              <span class="readout-value">{cursorPhase()}</span>
            </span>
          </Show>
        </div>
        {/* Snapshot SNAP/CLR buttons (band mode only) */}
        <Show when={!isSum()}>
          <span class="readout-sep" />
          <button class="tb-btn" onClick={takeFreqSnapshot} title="Snapshot corrected curve for comparison">SNAP</button>
          {(() => {
            const b = activeBand();
            const snaps = b ? freqSnapshots(b.id) : [];
            return (
              <>
                {snaps.length > 0 && (
                  <button class="tb-btn" onClick={clearFreqSnapshots} title="Clear all snapshots">CLR</button>
                )}
                {snaps.length > 0 && (
                  <span style={{ "font-size": "9px", "color": "#8b8b96", "margin-left": "4px" }}>
                    {snaps.map(s => (
                      <span style={{ color: s.color, "margin-right": "6px" }}>{"\u2588"} {s.label}</span>
                    ))}
                  </span>
                )}
              </>
            );
          })()}
        </Show>
        {/* IR/Step tab toggles */}
        <Show when={plotTab() === "ir" || plotTab() === "step"}>
          <span class="readout-sep" />
          <button class={`tb-btn ${irDbMode() ? "active" : ""}`} onClick={() => { setIrDbMode(!irDbMode()); irFullRedraw(); }} style={{ "font-size": "9px", padding: "1px 4px" }}>{irDbMode() ? "dB" : "Lin"}</button>
        </Show>
      </div>
      {/* Unified visibility matrix — above plot, all modes */}
      <Show when={plotTab() === "ir" || plotTab() === "step"}>
        {/* IR/Step matrix */}
        <div class="sum-vis-table">
          <table>
            <thead><tr>
              <th class="sum-corner" />
              <th style={{ color: "#4A9EFF" }}>IR</th>
              <th style={{ color: "#22C55E" }}>Step</th>
            </tr></thead>
            <tbody>
              <tr>
                <td class="sum-row-header" onClick={() => { setShowMeasIr(!showMeasIr()); setShowMeasStep(!showMeasStep()); irToggleVisibility(); }}>MEAS</td>
                <td class="sum-cell"><input type="checkbox" checked={showMeasIr()} onChange={() => { setShowMeasIr(!showMeasIr()); irToggleVisibility(); }} /></td>
                <td class="sum-cell"><input type="checkbox" checked={showMeasStep()} onChange={() => { setShowMeasStep(!showMeasStep()); irToggleVisibility(); }} /></td>
              </tr>
              <tr>
                <td class="sum-row-header" onClick={() => { setShowTargetIr(!showTargetIr()); setShowTargetStep(!showTargetStep()); irToggleVisibility(); }}>TARGET</td>
                <td class="sum-cell"><input type="checkbox" checked={showTargetIr()} onChange={() => { setShowTargetIr(!showTargetIr()); irToggleVisibility(); }} /></td>
                <td class="sum-cell"><input type="checkbox" checked={showTargetStep()} onChange={() => { setShowTargetStep(!showTargetStep()); irToggleVisibility(); }} /></td>
              </tr>
              <tr>
                <td class="sum-row-header" onClick={() => { setShowCorrIr(!showCorrIr()); setShowCorrStep(!showCorrStep()); irToggleVisibility(); }}>CORR</td>
                <td class="sum-cell"><input type="checkbox" checked={showCorrIr()} onChange={() => { setShowCorrIr(!showCorrIr()); irToggleVisibility(); }} /></td>
                <td class="sum-cell"><input type="checkbox" checked={showCorrStep()} onChange={() => { setShowCorrStep(!showCorrStep()); irToggleVisibility(); }} /></td>
              </tr>
              <tr>
                <td class="sum-row-header">OPTIONS</td>
                <td class="sum-cell" colspan="2">
                  <label style={{ "font-size": "9px", cursor: "pointer", display: "flex", "align-items": "center", gap: "3px" }}>
                    <input type="checkbox" checked={irShowMasking()} onChange={() => { setIrShowMasking(!irShowMasking()); irFullRedraw(); }} />
                    Pre-ringing
                  </label>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Show>
      <Show when={isSum() && showLegend() && legendEntries.length > 0}>
        {/* SUM matrix */}
        <div class="sum-vis-table">
          {(() => {
            const bandNames = () => appState.bands.map(b => b.name);
            const cols = () => [...bandNames(), "\u03A3"];
            const categories: ("target" | "measurement" | "corrected")[] = ["target", "measurement", "corrected"];
            const catLabels: Record<string, string> = { target: "TARGETS", measurement: "MEAS", corrected: "CORR+XO" };
            const catColors: Record<string, string> = { target: "#AAB4C0", measurement: "#8898A8", corrected: "#B0C0D0" };
            return (
              <table>
                <thead><tr>
                  <th class="sum-corner" />
                  <For each={cols()}>{(col) => <th onClick={() => toggleColumn(col)} title={`Toggle all ${col}`}>{col}</th>}</For>
                </tr></thead>
                <tbody>
                  <For each={categories}>
                    {(cat) => {
                      const catEnts = () => legendEntries.filter(e => e.category === cat);
                      const allOn = () => { const ce = catEnts(); return ce.length > 0 && ce.every(e => e.visible); };
                      const anyOn = () => catEnts().some(e => e.visible);
                      return (
                        <tr>
                          <td class={`sum-row-header ${allOn() ? "row-on" : anyOn() ? "row-partial" : ""}`} onClick={() => toggleCategory(cat)} title={`Toggle all ${catLabels[cat]}`}>
                            <span class="sum-row-swatch" style={{ "border-color": catColors[cat] }} />{catLabels[cat]}
                          </td>
                          <For each={cols()}>
                            {(col) => {
                              const entry = () => findCellEntry(col, cat);
                              return (
                                <Show when={entry()} fallback={<td class="sum-cell-empty" />}>
                                  {(e) => {
                                    const idx = () => legendEntries.findIndex(le => le.seriesIdx === e().seriesIdx);
                                    return (
                                      <td class="sum-cell">
                                        <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}>
                                          <span class={`legend-swatch ${e().dash ? "legend-swatch-dash" : ""}`} style={{ "background-color": e().dash ? "transparent" : e().color, "border-color": e().color }} />
                                        </button>
                                      </td>
                                    );
                                  }}
                                </Show>
                              );
                            }}
                          </For>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            );
          })()}
        </div>
      </Show>
      <Show when={!isSum() && showLegend() && legendEntries.length > 0 && plotTab() === "freq"}>
        {/* Freq band matrix — checkboxes like IR/Step */}
        <div class="sum-vis-table">
          {(() => {
            const categories: ("target" | "measurement" | "corrected")[] = ["measurement", "target", "corrected"];
            const catLabels: Record<string, string> = { target: "TARGET", measurement: "MEAS", corrected: "CORR" };
            return (
              <table>
                <thead><tr><th class="sum-corner" /><th>dB</th><th>°</th></tr></thead>
                <tbody>
                  <For each={categories}>
                    {(cat) => {
                      const catEnts = () => legendEntries.filter(e => e.category === cat);
                      const magE = () => catEnts().find(e => !e.dash);
                      const phE = () => catEnts().find(e => e.dash);
                      return (
                        <tr>
                          <td class="sum-row-header" onClick={() => toggleCategory(cat)}>{catLabels[cat]}</td>
                          <td class="sum-cell">
                            <Show when={magE()}>{(e) => <input type="checkbox" checked={e().visible} onChange={() => toggleLegendEntry(legendEntries.indexOf(e()))} />}</Show>
                          </td>
                          <td class="sum-cell">
                            <Show when={phE()}>{(e) => <input type="checkbox" checked={e().visible} onChange={() => toggleLegendEntry(legendEntries.indexOf(e()))} />}</Show>
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            );
          })()}
        </div>
      </Show>
      <div class="plot-body">
        <div class="plot-center">
          <div ref={containerRef} class="frequency-plot" />
          <div class="axis-controls axis-controls-y axis-controls-y-left">
            <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In dB">+</button>
            <button class="axis-btn" onClick={() => scrollY(1)} title="Scroll Up dB">▲</button>
            <button class="axis-btn" onClick={() => scrollY(-1)} title="Scroll Down dB">▼</button>
            <button class="axis-btn" onClick={() => zoomY(1.6)} title="Zoom Out dB">−</button>
            <button class="axis-btn fit-btn" onClick={fitData} title="Fit data to view">FIT</button>
          </div>
          <Show when={plotTab() === "freq" || plotTab() === "export"}>
            <div class="axis-controls axis-controls-y axis-controls-y-right">
              <button class="axis-btn" onClick={() => zoomPhase(0.6)} title="Zoom In Phase">+</button>
              <button class="axis-btn" onClick={() => scrollPhase(1)} title="Scroll Up Phase">▲</button>
              <button class="axis-btn" onClick={() => scrollPhase(-1)} title="Scroll Down Phase">▼</button>
              <button class="axis-btn" onClick={() => zoomPhase(1.6)} title="Zoom Out Phase">−</button>
            </div>
          </Show>
          <div class="axis-controls axis-controls-x">
            <button class="axis-btn" onClick={() => zoomX(1.6)} title="Zoom Out Freq">−</button>
            <button class="axis-btn" onClick={() => scrollX(-1)} title="Scroll Left">◀</button>
            <button class="axis-btn" onClick={() => scrollX(1)} title="Scroll Right">▶</button>
            <button class="axis-btn" onClick={() => zoomX(0.6)} title="Zoom In Freq">+</button>
          </div>
        </div>
      </div>
      {/* Band legend — matrix table (not in SUM, SUM uses its own matrix above) */}
      <Show when={showLegend() && !isSum() && legendEntries.length > 0 && plotTab() === "freq"}>
        <div class="sum-vis-table">
          {(() => {
            const categories: ("target" | "measurement" | "corrected")[] = ["measurement", "target", "corrected"];
            const catLabels: Record<string, string> = { target: "TARGET", measurement: "MEAS", corrected: "CORR" };
            const colLabels = ["dB", "°"];
            return (
              <table>
                <thead>
                  <tr>
                    <th class="sum-corner" />
                    <For each={colLabels}>
                      {(col) => <th>{col}</th>}
                    </For>
                  </tr>
                </thead>
                <tbody>
                  <For each={categories}>
                    {(cat) => {
                      const catEnts = () => legendEntries.filter(e => e.category === cat);
                      const magEntry = () => catEnts().find(e => !e.dash);
                      const phEntry = () => catEnts().find(e => e.dash);
                      return (
                        <tr>
                          <td class="sum-row-header" onClick={() => toggleCategory(cat)}>
                            {catLabels[cat]}
                          </td>
                          <td class="sum-cell" onClick={() => { const e = magEntry(); if (e) toggleLegendEntry(legendEntries.indexOf(e)); }}>
                            <Show when={magEntry()}>
                              {(e) => (
                                <span class={`sum-cell-dot ${e().visible ? "on" : ""}`} style={{ background: e().visible ? e().color : "transparent", "border-color": e().color }} />
                              )}
                            </Show>
                          </td>
                          <td class="sum-cell" onClick={() => { const e = phEntry(); if (e) toggleLegendEntry(legendEntries.indexOf(e)); }}>
                            <Show when={phEntry()}>
                              {(e) => (
                                <span class={`sum-cell-dot ${e().visible ? "on" : ""}`} style={{ background: e().visible ? e().color : "transparent", "border-color": e().color }} />
                              )}
                            </Show>
                          </td>
                        </tr>
                      );
                    }}
                  </For>
                </tbody>
              </table>
            );
          })()}
        </div>
      </Show>
      {/* Old band-legend removed — all modes use matrix above */}
    </div>
  );
}
