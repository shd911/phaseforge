import { createEffect, createSignal, onCleanup, onMount, For, Show, untrack, batch } from "solid-js";
import { createStore } from "solid-js/store";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { Measurement, TargetResponse, FilterType, FilterConfig, PeqBand, WindowType } from "../lib/types";
import { appState, activeBand, isSum, activeTab, plotTab, setPlotTab, sharedXScale, setSharedXScale, suppressXScaleSync, selectedPeqIdx, setSelectedPeqIdx, setBandLowPass, setBandCrossNormDb, plotShowOnly, setPlotShowOnly, addPeqBand, exportHybridPhase, freqSnapshots, setFreqSnapshots, peqDragging, setPeqDragging, updatePeqBand, commitPeqBand, bandsVersion, exportSampleRate, setExportSampleRate, exportTaps, setExportTaps, exportWindow, setExportWindow, firIterations, firFreqWeighting, firNarrowbandLimit, firNbSmoothingOct, firNbMaxExcess, firMaxBoost, firNoiseFloor, exportMetrics, setExportMetrics, plotSnapshots, addPlotSnapshot, clearPlotSnapshots, setAlignmentDelay, setBandSmoothing } from "../stores/bands";
import type { SmoothingMode, BandState, FreqSnapshot } from "../stores/bands";
import { needAutoFit, setNeedAutoFit } from "../App";
import { computeFloorBounce } from "../lib/floor-bounce";
import { openCrossoverDialog, type CrossoverDialogData } from "./CrossoverDialog";
import { handleImportMeasurement, setShowMergeDialog, showMergeDialog } from "../lib/measurement-actions";
import MergeDialog from "./MergeDialog";
import { copyMeasurementToProject } from "../lib/project-io";
import { open } from "@tauri-apps/plugin-dialog";
import { exportBandWav } from "../lib/fir-export";

import {
  SUM_TARGET_COLOR, SUM_TARGET_PHASE_COLOR, SUM_CORRECTED_COLOR, SUM_MEAS_COLOR,
  FREQ_SNAP_COLORS, bandColorFamily, smoothingConfig, wrapPhase, fmtFreq, computeGroupDelay,
  isGaussianMinPhase, gaussianFilterMagDb, unwrapDegrees,
  PEQ_COLOR, PEQ_HOVER_COLOR, PEQ_DRAG_COLOR, PEQ_BASE_COLOR,
  FIR_COLOR, CORRECTED_COLOR, MEAS_DEFAULT_COLOR,
  STATUS_GOOD, STATUS_WARN, STATUS_BAD,
  DEFAULT_IR_COLORS, DEFAULT_GD_COLORS, DEFAULT_EXPORT_COLORS,
} from "../lib/plot-helpers";
import { addGaussianMinPhase, evaluateBand } from "../lib/band-evaluation";
import { computeAutoAlign } from "../lib/auto-align";

// Track which inputs have been explicitly clicked — wheel only fires when in set
const wheelEnabled = new WeakSet<Element>();

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
  category: "measurement" | "target" | "corrected" | "peq" | "snapshot";
}

// Per-band IR/Step data for SUM mode rendering
interface IrBandData {
  bandName: string;
  bandColor: string;
  timeMs: number[];
  impulse: number[];
  step: number[];
  rawPeak: number;       /* pre-normalization impulse peak for shared scaling */
  stepRawPeak: number;   /* pre-normalization step peak for shared scaling */
}

export default function FrequencyPlot() {
  let containerRef: HTMLDivElement | undefined;
  let chart: uPlot | undefined;
  let renderGen = 0; // generation counter to discard stale async renders

  // Map to store original stroke colors for series (for stroke=transparent toggle)
  // Key: seriesIdx, Value: original stroke color string
  const origStrokes = new Map<number, string>();

  // Safe series visibility toggle: uses stroke=transparent instead of setSeries
  // to avoid uPlot internal scale recalculation (which breaks zoom).
  // setSeries({ show }) is only safe on freq tab where scales use range functions.
  function safeToggleSeries(seriesIdx: number, visible: boolean, needsRedraw = true) {
    if (!chart || seriesIdx < 1 || seriesIdx >= chart.series.length) return;
    const s = chart.series[seriesIdx] as any;
    const pTab = plotTab();

    if (pTab === "freq" || pTab === "export") {
      // Freq/Export: scales use range functions → setSeries is safe
      chart.setSeries(seriesIdx, { show: visible });
    } else {
      // IR/Step/GD: stroke=transparent approach (no scale recalc)
      if (visible) {
        const orig = origStrokes.get(seriesIdx);
        if (orig) s.stroke = orig;
      } else {
        // Save original stroke before hiding
        const cur = typeof s.stroke === "string" ? s.stroke : null;
        if (cur && cur !== "transparent" && !origStrokes.has(seriesIdx)) {
          origStrokes.set(seriesIdx, cur);
        }
        s.stroke = "transparent";
      }
      if (needsRedraw) chart.redraw(false, false);
    }
  }

  // Cache original strokes from series opts (call with the uSeries array BEFORE chart creation)
  function cacheOrigStrokes(seriesOpts?: uPlot.Series[]) {
    origStrokes.clear();
    const src = seriesOpts ?? (chart ? chart.series : []);
    for (let si = 1; si < src.length; si++) {
      const s = src[si] as any;
      const stroke = typeof s.stroke === "string" ? s.stroke : null;
      if (stroke && stroke !== "transparent") origStrokes.set(si, stroke);
    }
  }

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

  // Persisted scales across chart rebuilds (saved before destroy in main effect)
  let persistedMagMin: number | null = null;
  let persistedMagMax: number | null = null;
  let persistedXMin: number | null = null;
  let persistedXMax: number | null = null;
  let userHasZoomed = false; // true once user zooms manually

  // PEQ dot markers — mutable, updated during drag
  let activePeqDots: { seriesIdx: number; dataIndices: Set<number> } | null = null;

  // IR/Step mutable Y scale (like curMagMin/Max for freq)
  let irCurYMin = -120;
  let irCurYMax = 120;
  let irCurXMin = -30;
  let irCurXMax = 30;

  // GD scale persistence across rebuilds (toggle redraw)
  let gdUserYMin: number | null = null;
  let gdUserYMax: number | null = null;

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
    const pTab = plotTab();
    if (pTab === "ir" || pTab === "step") {
      irCurXMin = prev.xMin;
      irCurXMax = prev.xMax;
    }
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

  // Snapshot system: keep last curves for snapshot capture, per category
  const lastCorrData: { freq: number[]; mag: number[]; phase: (number | null)[] } = { freq: [], mag: [], phase: [] };
  const lastMeasData: { freq: number[]; mag: number[]; phase: (number | null)[] } = { freq: [], mag: [], phase: [] };
  const lastTargetData: { freq: number[]; mag: number[]; phase: (number | null)[] } = { freq: [], mag: [], phase: [] };
  // IR/Step snapshot data — per-category LINEAR (pre-dB), peak-aligned
  const lastIrData: { timeMs: number[]; impulse: number[]; step: number[] } = { timeMs: [], impulse: [], step: [] };
  const lastIrMeasData: { timeMs: number[]; impulse: number[]; step: number[] } = { timeMs: [], impulse: [], step: [] };
  const lastIrTargetData: { timeMs: number[]; impulse: number[]; step: number[] } = { timeMs: [], impulse: [], step: [] };
  const lastIrCorrData: { timeMs: number[]; impulse: number[]; step: number[] } = { timeMs: [], impulse: [], step: [] };
  // GD snapshot data (corrected GD)
  const lastGdData: { freq: number[]; gdMs: number[] } = { freq: [], gdMs: [] };
  // Export snapshot data (FIR curve)
  const lastExportData: { freq: number[]; mag: number[]; phase: number[] } = { freq: [], mag: [], phase: [] };

  function takeFreqSnapshot() {
    const band = activeBand();
    if (!band || !chart) return;
    // Determine visible categories from chart series (source of truth)
    let corrMagVis = false, corrPhVis = false;
    let tgtMagVis = false, tgtPhVis = false;
    let measMagVis = false, measPhVis = false;
    for (let i = 1; i < chart.series.length; i++) {
      const s = chart.series[i] as any;
      if (!s.show) continue;
      const lbl = typeof s.label === "string" ? s.label : "";
      if (lbl.startsWith("Snap ")) continue;
      // Detect phase by scale or label suffix "°"
      const isPhase = s.scale === "phase" || lbl.includes("\u00B0");
      // Detect category: corrected/target by keyword, everything else = measurement
      if (lbl.includes("orrect") || lbl.includes("orr")) { if (isPhase) corrPhVis = true; else corrMagVis = true; }
      else if (lbl.includes("arget") || lbl.includes("tgt")) { if (isPhase) tgtPhVis = true; else tgtMagVis = true; }
      else { if (isPhase) measPhVis = true; else measMagVis = true; }
    }
    // Capture ALL visible categories (not just highest-priority)
    const categories = [
      { key: "Corr", magVis: corrMagVis, phVis: corrPhVis, data: lastCorrData },
      { key: "Tgt", magVis: tgtMagVis, phVis: tgtPhVis, data: lastTargetData },
      { key: "Meas", magVis: measMagVis, phVis: measPhVis, data: lastMeasData },
    ];
    const visCats = categories.filter(c => (c.magVis || c.phVis) && c.data.freq.length > 0);
    if (visCats.length === 0) return;
    const snaps = freqSnapshots(band.id);
    // Count snap groups for color cycling
    const snapNums = new Set(snaps.map(s => s.label.match(/Snap (\d+)/)?.[1]).filter(Boolean));
    const groupIdx = snapNums.size;
    const color = FREQ_SNAP_COLORS[groupIdx % FREQ_SNAP_COLORS.length];
    const label = `Snap ${groupIdx + 1}`;
    const needSuffix = visCats.length > 1;
    const newSnaps = [...snaps];
    for (const cat of visCats) {
      newSnaps.push({
        label: needSuffix ? `${label} ${cat.key}` : label,
        freq: [...cat.data.freq],
        mag: cat.magVis ? [...cat.data.mag] : [],
        phase: cat.phVis ? [...cat.data.phase] : [],
        color,
      });
    }
    setFreqSnapshots(band.id, newSnaps);
  }

  function clearFreqSnapshots() {
    const band = activeBand();
    if (!band) return;
    setFreqSnapshots(band.id, []);
  }

  function takeSnapshot() {
    const band = activeBand();
    if (!band) return;
    const tab = plotTab();
    if (tab === "freq") { takeFreqSnapshot(); return; }
    const existing = plotSnapshots(band.id, tab === "step" ? "ir" : tab);
    // Count snap groups (unique "Snap N" prefixes), not individual entries
    const snapNums = new Set(existing.map(s => s.label.match(/Snap (\d+)/)?.[1]).filter(Boolean));
    const groupIdx = snapNums.size;
    const color = FREQ_SNAP_COLORS[groupIdx % FREQ_SNAP_COLORS.length];
    const label = `Snap ${groupIdx + 1}`;
    if (tab === "ir" || tab === "step") {
      // Use bandVisMap as source of truth for visibility (not chart.series.show — avoids async race)
      const isVis = (lbl: string) => bandVisMap.get(lbl) !== false;
      // Capture ALL visible categories (not just highest-priority)
      const categories = [
        { key: "Corr", irLbl: "Corrected IR", stLbl: "Corrected Step", data: lastIrCorrData },
        { key: "Tgt", irLbl: "Target IR", stLbl: "Target Step", data: lastIrTargetData },
        { key: "Meas", irLbl: "Measurement IR", stLbl: "Measurement Step", data: lastIrMeasData },
      ];
      let captured = 0;
      for (const cat of categories) {
        const hasIr = isVis(cat.irLbl) && cat.data.impulse.length > 0;
        const hasSt = isVis(cat.stLbl) && cat.data.step.length > 0;
        if (!hasIr && !hasSt) continue;
        // Count visible categories to decide if suffix needed
        captured++;
      }
      if (captured === 0) return;
      const needSuffix = captured > 1;
      for (const cat of categories) {
        const hasIr = isVis(cat.irLbl) && cat.data.impulse.length > 0;
        const hasSt = isVis(cat.stLbl) && cat.data.step.length > 0;
        if (!hasIr && !hasSt) continue;
        const snapLabel = needSuffix ? `${label} ${cat.key}` : label;
        addPlotSnapshot(band.id, {
          label: snapLabel, color, tab: "ir", timeMs: [...cat.data.timeMs],
          impulse: hasIr ? [...cat.data.impulse] : undefined,
          step: hasSt ? [...cat.data.step] : undefined,
        });
      }
    } else if (tab === "gd" && lastGdData.freq.length > 0) {
      addPlotSnapshot(band.id, { label, color, tab: "gd", freq: [...lastGdData.freq], gdMs: [...lastGdData.gdMs] });
    } else if (tab === "export" && lastExportData.freq.length > 0) {
      const hasMag = showExpFir() && lastExportData.mag.length > 0;
      const hasPh = showExpFirPh() && lastExportData.phase.length > 0;
      if (!hasMag && !hasPh) return;
      addPlotSnapshot(band.id, {
        label, color, tab: "export", freq: [...lastExportData.freq],
        exportMag: hasMag ? [...lastExportData.mag] : undefined,
        exportPhase: hasPh ? [...lastExportData.phase] : undefined,
      });
    }
    // Trigger re-render to show snapshot on chart (preserve scales)
    irToggleRedraw();
  }

  function clearSnapshots() {
    const band = activeBand();
    if (!band) return;
    const tab = plotTab();
    if (tab === "freq") { clearFreqSnapshots(); return; }
    clearPlotSnapshots(band.id, tab === "step" ? "ir" : tab);
    // Trigger re-render to remove snapshot curves from chart (preserve scales)
    irToggleRedraw();
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

  // Force IR/Step re-render (incremented by legend band toggles)
  const [irRenderTrigger, setIrRenderTrigger] = createSignal(0);

  // IR/Step colors derived from band — updated on each render
  const defaultIrColors = DEFAULT_IR_COLORS;
  const [irColors, setIrColors] = createSignal(defaultIrColors);
  // GD colors and visibility
  const defaultGdColors = DEFAULT_GD_COLORS;
  const [gdColors, setGdColors] = createSignal(defaultGdColors);
  const [showGdMeas, setShowGdMeas] = createSignal(true);
  const [showGdTarget, setShowGdTarget] = createSignal(true);
  const [showGdCorr, setShowGdCorr] = createSignal(true);
  // Export colors derived from band
  const [exportColors, setExportColors] = createSignal(DEFAULT_EXPORT_COLORS);
  // Export loading indicator
  const [exportComputing, setExportComputing] = createSignal(false);
  // WAV export state
  const [exportingWav, setExportingWav] = createSignal(false);
  const [exportWavError, setExportWavError] = createSignal<string | null>(null);
  async function handleExportWav() {
    const b = activeBand();
    if (!b) return;
    setExportingWav(true);
    setExportWavError(null);
    try {
      await exportBandWav(b);
    } catch (e) {
      console.error("Export WAV failed:", e);
      setExportWavError(String(e));
    } finally {
      setExportingWav(false);
    }
  }
  // Export legend visibility
  const [showExpModel, setShowExpModel] = createSignal(true);
  const [showExpFir, setShowExpFir] = createSignal(true);
  const [showExpModelPh, setShowExpModelPh] = createSignal(true);
  const [showExpFirPh, setShowExpFirPh] = createSignal(true);

  // Persistent visibility — two maps for different modes:
  // SUM mode: by label (each band has its own curves like "Band 1 tgt", "Band 2 tgt")
  // Band mode: by category key (labels change per band, categories don't)
  let sumVisMap = new Map<string, boolean>();
  let bandVisMap = new Map<string, boolean>();
  // Reactive set of excluded band names for SUM IR/Step coherent sum
  const [irExcludedBands, setIrExcludedBands] = createSignal<Set<string>>(new Set());

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
    let newMin: number, newMax: number;
    if (pTab === "freq" || pTab === "export") {
      // Anchor 0 dB at 20% from top — zoom expands/contracts below
      const totalRange = (s.max - s.min) * factor;
      if (totalRange < 0.01 || totalRange > 500) { zoomStack.pop(); return; }
      newMax = 0 + totalRange * 0.2;
      newMin = 0 - totalRange * 0.8;
    } else {
      const center = (s.min + s.max) / 2;
      const half = ((s.max - s.min) / 2) * factor;
      if (half * 2 < 0.01 || half * 2 > 500) { zoomStack.pop(); return; }
      newMin = center - half;
      newMax = center + half;
    }
    if (pTab === "freq" || pTab === "export") { curMagMin = newMin; curMagMax = newMax; }
    else if (pTab === "ir" || pTab === "step") { irCurYMin = newMin; irCurYMax = newMax; }
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
    else if (pTab === "ir" || pTab === "step") { irCurYMin = newMin; irCurYMax = newMax; }
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
      irCurXMin = center - half;
      irCurXMax = center + half;
      chart.setScale("x", { min: irCurXMin, max: irCurXMax });
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
      irCurXMin = s.min + step;
      irCurXMax = s.max + step;
      chart.setScale("x", { min: irCurXMin, max: irCurXMax });
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
      let gdMin = Infinity, gdMax = -Infinity;
      for (let si = 1; si < chart.data.length; si++) {
        if (!(chart.series[si] as any)?.show) continue;
        const arr = chart.data[si];
        if (!arr) continue;
        for (const v of arr) { if (v != null && isFinite(v as number)) { if ((v as number) < gdMin) gdMin = v as number; if ((v as number) > gdMax) gdMax = v as number; } }
      }
      if (isFinite(gdMin) && isFinite(gdMax)) {
        const pad = Math.max(0.5, (gdMax - gdMin) * 0.1);
        chart.setScale("y", { min: gdMin - pad, max: gdMax + pad });
      }
    } else {
      // IR/Step — fit ±30ms around peak
      const d = chart.data[0];
      const ir = chart.data[1];
      if (d && d.length > 0 && ir) {
        let pkIdx = 0, pkVal = 0;
        for (let i = 0; i < ir.length; i++) { const v = ir[i]; if (v != null && Math.abs(v as number) > pkVal) { pkVal = Math.abs(v as number); pkIdx = i; } }
        const pkT = d[pkIdx] as number;
        irCurXMin = pkT - 30;
        irCurXMax = pkT + 30;
        chart.setScale("x", { min: irCurXMin, max: irCurXMax });
      }
      // Compute Y range from visible series data only
      let yMin = Infinity, yMax = -Infinity;
      for (let si = 1; si < chart.data.length; si++) {
        if (!(chart.series[si] as any)?.show) continue; // skip hidden
        const arr = chart.data[si];
        if (!arr) continue;
        for (const v of arr) {
          if (v != null && v > -190) {
            if (v < yMin) yMin = v;
            if (v > yMax) yMax = v;
          }
        }
      }
      if (isFinite(yMin) && isFinite(yMax)) {
        const pad = Math.max(0.05, (yMax - yMin) * 0.05);
        irCurYMin = yMin - pad;
        irCurYMax = yMax + pad;
        chart.setScale("y", { min: irCurYMin, max: irCurYMax });
      }
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
      const pTab = plotTab();
      if (pTab === "ir" || pTab === "step") {
        irCurXMin = fMin;
        irCurXMax = fMax;
      }
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
    const entry = legendEntries[idx];
    const newVis = !entry.visible;
    const pTab = plotTab();

    setLegendEntries(idx, "visible", newVis);
    const onFreq = pTab === "freq";
    // On freq tab: toggle chart series directly (safe — range functions protect scales)
    if (onFreq && chart) {
      chart.setSeries(entry.seriesIdx, { show: newVis });
    }
    // On IR/Step/GD: save visibility state, trigger rebuild (setSeries breaks scales)
    if (pTab === "ir" || pTab === "step") {
      if (isSum()) sumVisMap.set(entry.label, newVis);
      else bandVisMap.set(catKey(entry), newVis);
      irToggleRedrawAutoY();
      return;
    }
    if (pTab === "gd") {
      gdToggleRedraw();
      return;
    }
    if (isSum()) {
      sumVisMap.set(entry.label, newVis);
      // Also toggle paired phase series (mag ↔ phase)
      if (!entry.label.endsWith(" \u00B0")) {
        const phaseSuffix = " \u00B0";
        for (let pi = 0; pi < legendEntries.length; pi++) {
          const pe = legendEntries[pi];
          if (pe.category === entry.category && pe.label === entry.label + phaseSuffix) {
            if (pe.visible !== newVis) {
              setLegendEntries(pi, "visible", newVis);
              if (onFreq && chart) chart.setSeries(pe.seriesIdx, { show: newVis });
              sumVisMap.set(pe.label, newVis);
            }
            break;
          }
        }
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
                if (onFreq && chart) chart.setSeries(legendEntries[pi].seriesIdx, { show: newVis });
                sumVisMap.set(sigmaPhLabel, newVis);
              }
              break;
            }
          }
        }
      }
      // IR/Step/GD already returned above with full rebuild
    } else {
      bandVisMap.set(catKey(entry), newVis);
    }
  }

  // Toggle all series for a given band column in SUM mode
  function toggleColumn(colName: string) {
    const pTab = plotTab();
    const onIrStep = pTab === "ir" || pTab === "step";

    const matching: number[] = [];
    for (let i = 0; i < legendEntries.length; i++) {
      const e = legendEntries[i];
      if (colName === "\u03A3") {
        if (e.label.startsWith("\u03A3")) matching.push(i);
      } else if (onIrStep) {
        // IR/Step: match all entries for this band name (IR+Step for meas/tgt/corr)
        if (e.label === colName + " IR" || e.label === colName + " Step"
          || e.label === colName + " tgt IR" || e.label === colName + " tgt Step"
          || (e.label.startsWith(colName + " corr+XO") && e.label.endsWith(" IR")) || (e.label.startsWith(colName + " corr+XO") && e.label.endsWith(" Step"))) matching.push(i);
      } else {
        if (e.label === colName || e.label === colName + " \u00B0"
          || e.label === colName + " tgt"
          || e.label.startsWith(colName + " corr+XO") || e.label === colName + " corr+XO \u00B0") matching.push(i);
      }
    }
    if (matching.length === 0 && !(isSum() && onIrStep && colName !== "\u03A3")) return;
    const allOn = matching.length > 0 && matching.every(i => legendEntries[i].visible);
    const newVis = !allOn;
    const onFreq = pTab === "freq";
    for (const i of matching) {
      if (legendEntries[i].visible !== newVis) {
        setLegendEntries(i, "visible", newVis);
        if (onFreq && chart) chart.setSeries(legendEntries[i].seriesIdx, { show: newVis });
        if (isSum()) {
          sumVisMap.set(legendEntries[i].label, newVis);
        } else {
          bandVisMap.set(catKey(legendEntries[i]), newVis);
        }
      }
    }
    // Sync IR show signals when toggling Σ column on IR/Step
    if (isSum() && (pTab === "ir" || pTab === "step") && colName === "\u03A3") {
      setShowMeasIr(legendEntries.some(e => e.category === "measurement" && e.visible));
      setShowMeasStep(legendEntries.some(e => e.category === "measurement" && e.visible));
      setShowTargetIr(legendEntries.some(e => e.category === "target" && e.visible));
      setShowTargetStep(legendEntries.some(e => e.category === "target" && e.visible));
      setShowCorrIr(legendEntries.some(e => e.category === "corrected" && e.visible));
      setShowCorrStep(legendEntries.some(e => e.category === "corrected" && e.visible));
    }
    // On IR/Step: band column toggle also updates coherent sum inclusion
    if (isSum() && (pTab === "ir" || pTab === "step") && colName !== "\u03A3") {
      const cur = irExcludedBands().has(colName);
      setIrExcludedBands(prev => { const s = new Set(prev); if (cur) s.delete(colName); else s.add(colName); return s; });
      setIrRenderTrigger(v => v + 1);
    }
    if (isSum() && !onFreq && !(pTab === "ir" || pTab === "step")) {
      if (pTab === "gd") {
        setIrRenderTrigger(v => v + 1);
      }
    }
  }

  // Toggle all IR or all Step series at once (header click in legend table)
  function toggleAllIrOrStep(suffix: "IR" | "Step") {
    const matching: number[] = [];
    for (let i = 0; i < legendEntries.length; i++) {
      if (legendEntries[i].label.endsWith(" " + suffix)) matching.push(i);
    }
    if (matching.length === 0) return;

    // Collect unique categories from matching entries
    const categories = new Set<string>();
    for (const i of matching) {
      categories.add(catKey(legendEntries[i]));
    }

    // Determine which categories are enabled (at least one entry visible)
    const enabledCats = new Set<string>();
    for (const cat of categories) {
      for (let i = 0; i < legendEntries.length; i++) {
        if (catKey(legendEntries[i]) === cat && legendEntries[i].visible) {
          enabledCats.add(cat);
          break;
        }
      }
    }

    // Build list of entries from enabled categories only
    const scopedMatching: number[] = [];
    for (const i of matching) {
      if (enabledCats.has(catKey(legendEntries[i]))) {
        scopedMatching.push(i);
      }
    }
    if (scopedMatching.length === 0) return;

    const allOn = scopedMatching.every(i => legendEntries[i].visible);
    const newVis = !allOn;
    for (const i of scopedMatching) {
      if (legendEntries[i].visible !== newVis) {
        setLegendEntries(i, "visible", newVis);
        if (isSum()) sumVisMap.set(legendEntries[i].label, newVis);
        else bandVisMap.set(catKey(legendEntries[i]), newVis);
      }
    }
    irToggleRedrawAutoY();
  }

  // Find a legend entry for a specific [bandName, category] cell
  // On IR/Step tab: returns the IR entry (the Step entry is toggled via pairing in toggleLegendEntry)
  function findCellEntry(colName: string, cat: "measurement" | "target" | "corrected"): LegendEntry | undefined {
    const pTab = plotTab();
    const onIrStep = pTab === "ir" || pTab === "step";
    for (let i = 0; i < legendEntries.length; i++) {
      const e = legendEntries[i];
      if (e.category !== cat) continue;
      if (colName === "\u03A3") {
        if (onIrStep) {
          // Return the IR entry for this Σ category
          if (e.label.startsWith("\u03A3") && e.label.endsWith(" IR")) return e;
        } else {
          if (e.label.startsWith("\u03A3")) return e;
        }
      } else {
        // Band column — match by band name prefix
        if (onIrStep) {
          // IR/Step: match per-band IR entry
          if (cat === "measurement" && e.label === colName + " IR") return e;
          if (cat === "target" && e.label === colName + " tgt IR") return e;
          if (cat === "corrected" && e.label.startsWith(colName + " corr+XO") && e.label.endsWith(" IR")) return e;
        } else {
          if (cat === "measurement" && e.label === colName) return e;
          if (cat === "target" && e.label === colName + " tgt") return e;
          if (cat === "corrected" && e.label.startsWith(colName + " corr+XO") && !e.label.endsWith(" \u00B0")) return e;
        }
      }
    }
    return undefined;
  }

  // Find both IR and Step entries for a cell (IR/Step tab only)
  function findCellEntryPair(colName: string, cat: "measurement" | "target" | "corrected"): { ir: LegendEntry | undefined; step: LegendEntry | undefined } {
    let ir: LegendEntry | undefined, step: LegendEntry | undefined;
    for (let i = 0; i < legendEntries.length; i++) {
      const e = legendEntries[i];
      if (e.category !== cat) continue;

      const isIR = e.label.endsWith(" IR");
      const isStep = e.label.endsWith(" Step");
      let matches = false;
      if (colName === "\u03A3") {
        matches = e.label.startsWith("\u03A3");
      } else {
        if (cat === "measurement") matches = e.label.startsWith(colName + " ");
        else if (cat === "target") matches = e.label.startsWith(colName + " tgt ");
        else matches = e.label.startsWith(colName + " corr+XO ");
      }
      if (matches && isIR) ir = e;
      if (matches && isStep) step = e;
    }
    return { ir, step };
  }

  // Переключение всей категории (targets / measurements / corrected)
  function toggleCategory(cat: "measurement" | "target" | "corrected" | "peq") {
    const pTab = plotTab();

    const indices: number[] = [];
    for (let i = 0; i < legendEntries.length; i++) {
      if (legendEntries[i].category === cat) indices.push(i);
    }
    if (indices.length === 0) return;
    const allOn = indices.every(i => legendEntries[i].visible);
    const newVis = !allOn;
    const onFreq = pTab === "freq";
    for (const i of indices) {
      if (legendEntries[i].visible !== newVis) {
        setLegendEntries(i, "visible", newVis);
        // Freq: inline toggle (safe with range functions)
        if (onFreq && chart) chart.setSeries(legendEntries[i].seriesIdx, { show: newVis });
        if (isSum()) {
          sumVisMap.set(legendEntries[i].label, newVis);
        } else {
          bandVisMap.set(catKey(legendEntries[i]), newVis);
        }
      }
    }
    // IR/Step: save scales → full rebuild (setSeries breaks scales)
    if (pTab === "ir" || pTab === "step") {
      irSaveScales();
      setIrRenderTrigger(v => v + 1);
    } else if (pTab === "gd") {
      gdToggleRedraw();
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
        if (plotTab() === "freq" && chart) chart.setSeries(legendEntries[i].seriesIdx, { show });
      }
      // Persist so switching bands/SUM doesn't lose this state
      if (inSum) {
        sumVisMap.set(legendEntries[i].label, show);
      } else {
        bandVisMap.set(catKey(legendEntries[i]), show);
      }
    }
    // For non-freq tabs: trigger rebuild
    const pTab = plotTab();
    if (pTab === "ir" || pTab === "step") { irSaveScales(); setIrRenderTrigger(v => v + 1); }
    else if (pTab === "gd") gdToggleRedraw();
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
    // Fallback: use persisted scales from main effect (chart was already destroyed)
    if (!savedMagMin && !doAutoFit && persistedMagMin != null) {
      savedMagMin = persistedMagMin;
      savedMagMax = persistedMagMax;
    }
    if (!savedXMin && !doAutoFit && persistedXMin != null) {
      savedXMin = persistedXMin;
      savedXMax = persistedXMax;
    }
    // Discard non-positive X scales (from IR/GD linear chart) — log freq scale needs x > 0
    if (savedXMin != null && savedXMin <= 0) { savedXMin = null; savedXMax = null; }
    // Defer old chart cleanup — destroy AFTER new chart is created to avoid visual gap
    const oldChart = chart;
    if (oldChart) {
      if (oldChart.over) {
        oldChart.over.removeEventListener("mousemove", handleXoMouseMove);
        oldChart.over.removeEventListener("mousedown", handleXoMouseDown);
        oldChart.over.removeEventListener("dblclick", handleXoDblClick);
        oldChart.over.removeEventListener("dblclick", handlePeqDblClick);
        oldChart.over.removeEventListener("mousedown", handlePeqMouseDown);
        oldChart.over.removeEventListener("wheel", handlePeqWheel);
        oldChart.over.removeEventListener("mousedown", handleZoomBoxDown);
        oldChart.over.removeEventListener("contextmenu", handleContextMenu);
      }
      window.removeEventListener("mousemove", handleZoomBoxMove);
      window.removeEventListener("mouseup", handleZoomBoxUp);
    }
    chart = undefined;

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

    // Fit range: 0 dB at 20% from top, bottom from data minimum
    fitMagMin = Math.floor(magMin / 5) * 5; // round down to nearest 5 dB
    const fitRange = Math.max(20, 0 - fitMagMin); // range below 0
    fitMagMax = fitRange * 0.25; // 0 sits at 80% from bottom = 20% from top
    fitMagMin = 0 - fitRange;
    if (fitMagMax - fitMagMin < 20) { fitMagMax = 5; fitMagMin = -15; }
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
        values: (_u: uPlot, vals: number[]) => {
          if (!vals || vals.length < 2) return vals.map(v => v == null ? "" : v.toFixed(0));
          // Adaptive decimals: increase precision until no duplicate labels
          for (let dec = 0; dec <= 2; dec++) {
            const labels = vals.map(v => v == null ? "" : v.toFixed(dec));
            const nonEmpty = labels.filter(s => s !== "");
            if (new Set(nonEmpty).size === nonEmpty.length) return labels;
          }
          // Still duplicates at 2 decimals — hide duplicates, keep first occurrence
          const labels = vals.map(v => v == null ? "" : v.toFixed(2));
          const seen = new Set<string>();
          return labels.map(s => {
            if (s === "" || seen.has(s)) return "";
            seen.add(s);
            return s;
          });
        },
        size: 55,
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
        phase: { auto: false, range: () => [curPhaseMin, curPhaseMax] as uPlot.Range.MinMax },
      },
      axes,
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
      hooks: {
        setScale: [
          (u: uPlot, key: string) => {
            if (key !== "x") return;
            if (peqDragActive) return; // don't sync X during PEQ drag
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
              const le = leg.find((e: LegendEntry) => e.seriesIdx === si);
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
              ctx.strokeStyle = isSel ? PEQ_COLOR : "rgba(255,159,67,0.35)";
              ctx.lineWidth = isSel ? 2 : 1;
              ctx.setLineDash(isSel ? [6, 4] : [4, 4]);
              ctx.beginPath();
              ctx.moveTo(cx, plotTop);
              ctx.lineTo(cx, plotBottom);
              ctx.stroke();

              if (isSel) {
                ctx.setLineDash([]);
                ctx.fillStyle = PEQ_COLOR;
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
              ctx.strokeStyle = isDrg ? PEQ_DRAG_COLOR : isHov ? PEQ_HOVER_COLOR : PEQ_BASE_COLOR;
              ctx.lineWidth = isDrg ? 2 : 1.5;
              ctx.setLineDash([6, 4]);
              ctx.beginPath();
              ctx.moveTo(cx, plotTop);
              ctx.lineTo(cx, plotBottom);
              ctx.stroke();

              // Circle marker at dB level
              ctx.setLineDash([]);
              ctx.fillStyle = isDrg ? PEQ_DRAG_COLOR : isHov ? PEQ_HOVER_COLOR : PEQ_BASE_COLOR;
              ctx.strokeStyle = "#1e1e24";
              ctx.lineWidth = 2 * dpr;
              ctx.beginPath();
              ctx.arc(cx, cy, radius, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();

              // Frequency label above marker
              ctx.fillStyle = isDrg ? PEQ_DRAG_COLOR : isHov ? PEQ_HOVER_COLOR : PEQ_BASE_COLOR;
              ctx.font = `bold ${Math.round(11 * dpr)}px sans-serif`;
              ctx.textAlign = "center";
              ctx.fillText(fmtCrossoverFreq(f), cx, cy - radius - 6 * dpr);
            }
            ctx.restore();
          },
          // PEQ band dots on PEQ response curve (reads mutable activePeqDots)
          (u: uPlot) => {
            const dots = activePeqDots;
            if (!dots || dots.dataIndices.size === 0) return;
            const si = dots.seriesIdx;
            if (si >= u.series.length || !(u.series[si] as any).show) return;
            const ctx = u.ctx;
            const dpr = devicePixelRatio || 1;
            const xData = u.data[0];
            const yData = u.data[si];
            ctx.save();
            for (const idx of dots.dataIndices) {
              const xVal = xData[idx];
              const yVal = yData[idx];
              if (xVal == null || yVal == null) continue;
              const cx = u.valToPos(xVal, "x", true);
              const cy = u.valToPos(yVal, "mag", true);
              if (!isFinite(cx) || !isFinite(cy)) continue;
              const r = 4 * dpr;
              ctx.beginPath();
              ctx.arc(cx, cy, r, 0, 2 * Math.PI);
              ctx.fillStyle = PEQ_COLOR;
              ctx.fill();
              ctx.strokeStyle = "#fff";
              ctx.lineWidth = 1.5 * dpr;
              ctx.stroke();
            }
            ctx.restore();
          },
        ],
      },
    };

    try {
      cacheOrigStrokes(input.uSeries);
      chart = new uPlot(opts, input.uData as uPlot.AlignedData, containerRef);
      // Destroy old chart AFTER new one is in the DOM — no visual gap
      if (oldChart) { try { oldChart.destroy(); } catch (_) {} }
    } catch (e) {
      console.error("uPlot error:", e);
      if (oldChart) { try { oldChart.destroy(); } catch (_) {} }
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
    // PEQ drag: mousedown on PEQ dots to drag freq/gain
    if (chart && chart.over) {
      chart.over.addEventListener("dblclick", handlePeqDblClick);
      chart.over.addEventListener("mousedown", handlePeqMouseDown);
      chart.over.addEventListener("wheel", handlePeqWheel, { passive: false });
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
  // Skip for IR/Step — they use their own linear X range (irCurXMin/irCurXMax)
  createEffect(() => {
    const xs = sharedXScale();
    if (!chart) return;
    const pTab = untrack(() => plotTab());
    if (pTab === "ir" || pTab === "step") return;
    const cur = chart.scales["x"];
    if (cur?.min != null && cur?.max != null) {
      // Only update if significantly different to avoid loops
      if (Math.abs(cur.min - xs.min) < 0.01 && Math.abs(cur.max - xs.max) < 0.01) return;
    }
    suppressXScaleSync(() => {
      chart!.setScale("x", { min: xs.min, max: xs.max });
    });
  });

  // IR/Step: saved user zoom for restore after rebuild
  let irUserXScale: { min: number; max: number } | null = null;
  let irUserYScale: { min: number; max: number } | null = null;

  function irSaveScales() {
    if (chart && chart.scales["y"]) {
      const xs = chart.scales["x"];
      const ys = chart.scales["y"];
      if (xs?.min != null && xs?.max != null) irUserXScale = { min: xs.min, max: xs.max };
      if (ys?.min != null && ys?.max != null) irUserYScale = { min: ys.min, max: ys.max };
    }
  }

  function irRestoreScales() {
    if (chart && irUserXScale) {
      // Guard: discard stale X scales with unreasonable range (>5000ms = not IR data)
      const xRange = irUserXScale.max - irUserXScale.min;
      if (xRange > 0 && xRange < 5000) {
        irCurXMin = irUserXScale.min;
        irCurXMax = irUserXScale.max;
        try { chart.setScale("x", irUserXScale); } catch(_){}
      } else {
        irUserXScale = null;
      }
    }
    if (irUserYScale) {
      irCurYMin = irUserYScale.min;
      irCurYMax = irUserYScale.max;
      if (chart) try { chart.setScale("y", irUserYScale); } catch(_){}
    }
  }

  // IR/Step: save scales → rebuild, preserving Y zoom
  function irToggleRedraw() {
    irSaveScales();
    setIrRenderTrigger(v => v + 1);
  }

  // IR/Step: save X scale → rebuild with Y auto-fit (visibility changed)
  function irToggleRedrawAutoY() {
    irSaveScales();
    irUserYScale = null;
    setIrRenderTrigger(v => v + 1);
  }

  // Helper: sync persistent IR/Step show signals from category visibility in legendEntries
  // Called after row/category/Sigma toggles on IR/Step tab to keep signals in sync

  // applyIrShowStatesToChart removed — standard legendEntries flow handles visibility

  function gdToggleRedraw() {
    const pTab = plotTab();
    if (pTab === "gd") {
      // Save GD Y scale before rebuild
      if (chart && chart.scales["y"]) {
        const ys = chart.scales["y"];
        if (ys.min != null && ys.max != null) { gdUserYMin = ys.min; gdUserYMax = ys.max; }
      }
      renderTimeTab("gd", isSum(), activeBand());
    }
  }

  // Redraw when selected PEQ index changes (for vertical dashed line)
  createEffect(() => {
    const _sel = selectedPeqIdx(); // track
    if (chart) chart.redraw(false, false);
  });


  // Fast PEQ update during drag — recompute PEQ + corrected in-place, no chart rebuild
  let peqFastGen = 0;
  async function peqFastUpdate(band: BandState) {
    const gen = ++peqFastGen;
    if (!chart || !band.peqBands?.length) return;
    const peqSi = chart.series.findIndex(s => s.label === "PEQ dB");
    if (peqSi < 0) return;
    const freq = chart.data[0];
    if (!freq || freq.length === 0) return;

    const enabledBands = band.peqBands.filter((b: PeqBand) => b.enabled);
    if (enabledBands.length === 0) return;

    try {
      const [pm] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq: Array.from(freq), bands: enabledBands,
      });
      if (!chart || gen !== peqFastGen) return;

      const newData = [...chart.data];
      newData[peqSi] = pm;

      // Also update corrected = meas + peq + xs (xs stays constant during drag)
      const label = (s: uPlot.Series) => (typeof s.label === "string" ? s.label : "");
      const corrSi = chart.series.findIndex(s =>
        label(s).includes("Corrected") && (s as any).scale === "mag"
      );
      const measSi = chart.series.findIndex(s =>
        (s as any).scale === "mag" && s !== chart!.series[0] &&
        !label(s).startsWith("Target") && !label(s).startsWith("PEQ") &&
        !label(s).includes("Corrected") && !label(s).includes("°")
      );
      if (corrSi > 0 && measSi > 0) {
        const measData = chart.data[measSi];
        const oldCorr = chart.data[corrSi];
        const oldPeq = chart.data[peqSi];
        if (measData && oldCorr && oldPeq) {
          const corrData = new Array(freq.length);
          for (let i = 0; i < freq.length; i++) {
            const xs = (oldCorr[i] ?? 0) - (measData[i] ?? 0) - (oldPeq[i] ?? 0);
            corrData[i] = (measData[i] ?? 0) + pm[i] + xs;
          }
          newData[corrSi] = corrData;
        }
      }

      // Update dot positions for new PEQ frequencies
      const newDotIndices = new Set<number>();
      for (const pb of enabledBands) {
        let bestIdx = 0, bestDist = Infinity;
        for (let k = 0; k < freq.length; k++) {
          const d = Math.abs(freq[k] - pb.freq_hz);
          if (d < bestDist) { bestDist = d; bestIdx = k; }
        }
        newDotIndices.add(bestIdx);
      }
      activePeqDots = { seriesIdx: peqSi, dataIndices: newDotIndices };

      chart.setData(newData as uPlot.AlignedData, false);
      chart.redraw(false, false);
    } catch (_) {}
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
    const _irTrigger = irRenderTrigger(); // track: force IR re-render from legend band toggles

    if (debounceTimer) clearTimeout(debounceTimer);

    // During PEQ drag: skip full rebuild, use fast in-place update
    if (dragging && chart && pTab === "freq" && !sumMode && band) {
      peqFastUpdate(band);
      return;
    }

    // Detect current chart type by its contents (not by pTab, which is already the NEW tab)
    const hasLabel = (substr: string) => chart?.series.some(s => typeof s.label === "string" && s.label.includes(substr));
    const chartIsIr = chart && chart.scales["y"] && (hasLabel("IR") || hasLabel("Step"));
    const chartIsGd = chart && chart.scales["y"] && hasLabel("GD");
    const chartIsMag = chart && chart.scales["mag"];

    // Save scales based on what the current chart IS
    if (chartIsIr) {
      irSaveScales();
    }
    if (chartIsGd) {
      const ys = chart!.scales["y"];
      if (ys?.min != null && ys?.max != null) { gdUserYMin = ys.min; gdUserYMax = ys.max; }
    }
    if (chartIsMag) {
      const ms = chart!.scales["mag"];
      const xs = chart!.scales["x"];
      if (ms?.min != null && ms?.max != null) { persistedMagMin = ms.min; persistedMagMax = ms.max; }
      if (xs?.min != null && xs?.max != null) { persistedXMin = xs.min; persistedXMax = xs.max; }
    }
    // Clear IR saved scales only when switching AWAY from IR tab to non-IR tab
    if (!chartIsIr && (pTab !== "ir" && pTab !== "step")) {
      irUserXScale = null;
      irUserYScale = null;
    }
    // On freq tab: don't destroy here — renderChart will replace seamlessly (no flash gap)
    // On other tabs: destroy immediately
    if (pTab !== "freq") {
      try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
    }

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
      try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
      setShowLegend(false); setExportComputing(false);
      setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
      return;
    }

    setExportComputing(true);
    try {
      const sr = exportSampleRate();
      const taps = exportTaps();
      const win = exportWindow();
      const target = JSON.parse(JSON.stringify(band.target));
      const peqBands = band.peqBands?.filter((b: PeqBand) => b.enabled) ?? [];

      // Evaluate target
      const [freq, response] = await invoke<[number[], TargetResponse]>(
        "evaluate_target_standalone", { target, nPoints: 512, fMin: 5, fMax: 40000 },
      );
      if (gen !== renderGen) return;

      const targetMag = response.magnitude;
      let modelPhase = response.phase;

      // Gaussian min-phase: per-filter Hilbert (for display)
      if (isGaussianMinPhase(target.high_pass) || isGaussianMinPhase(target.low_pass)) {
        modelPhase = await addGaussianMinPhase(freq, modelPhase, target.high_pass, target.low_pass);
        if (gen !== renderGen) return;
      }

      // PEQ contribution
      let peqMagArr: number[] = [];
      if (peqBands.length > 0) {
        const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq, bands: peqBands, sampleRate: sr });
        peqMagArr = pm;
        modelPhase = modelPhase.map((v: number, i: number) => v + pp[i]);
      }
      if (gen !== renderGen) return;

      // Generate FIR
      const isLin = (f: FilterConfig | null | undefined) => !f || f.linear_phase;
      const allLinear = isLin(target.high_pass) && isLin(target.low_pass);
      // FIR phase mode:
      // LinearPhase: all filters lin → symmetric FIR
      // MixedPhase: ONLY when Gaussian filters have MIXED lin/min phase (one lin + one min)
      // MinimumPhase: all non-linear, or all Gaussian min-phase → blanket Hilbert
      // FIR: LinearPhase (all lin) or MinimumPhase (any min-phase)
      // MixedPhase per-filter Hilbert causes phase oscillations in FIR due to windowing
      // artifacts — this is a physical limitation of single-FIR design.
      // Per-filter Hilbert is for DISPLAY only (Model curve on SPL/GD/IR tabs).
      const firConfig = {
        taps, sample_rate: sr, max_boost_db: firMaxBoost(), noise_floor_db: firNoiseFloor(),
        window: win, phase_mode: allLinear ? "LinearPhase" : "MinimumPhase",
        iterations: firIterations(), freq_weighting: firFreqWeighting(),
        narrowband_limit: firNarrowbandLimit(), nb_smoothing_oct: firNbSmoothingOct(),
        nb_max_excess_db: firNbMaxExcess(),
      };
      const firResult = await invoke<{ realized_mag: number[]; realized_phase: number[]; impulse: number[]; time_ms: number[]; norm_db: number; causality: number; taps: number; sample_rate: number }>(
        "generate_model_fir", { freq, targetMag, peqMag: peqMagArr, modelPhase: new Array(freq.length).fill(0), config: firConfig },
      );
      if (gen !== renderGen) return;

      // Normalize model mag
      const normModelMag = targetMag.map((v: number, i: number) => (v + (peqMagArr[i] ?? 0)) - firResult.norm_db);

      // Compute export metrics from FIR result
      // Pre-ringing: time from start to peak of impulse (ms)
      let peakIdx = 0, peakVal = 0;
      for (let i = 0; i < firResult.impulse.length; i++) {
        if (Math.abs(firResult.impulse[i]) > peakVal) {
          peakVal = Math.abs(firResult.impulse[i]);
          peakIdx = i;
        }
      }
      const preRingMs = peakIdx > 0 ? firResult.time_ms[peakIdx] - firResult.time_ms[0] : 0;

      // Max magnitude error in passband (realized vs model)
      const modelMag = targetMag.map((v: number, i: number) => v + (peqMagArr[i] ?? 0));
      const hpF = band.target.high_pass?.freq_hz ?? 20;
      const lpF = band.target.low_pass?.freq_hz ?? 20000;
      const pbLo = Math.max(20, hpF * 1.2);
      const pbHi = Math.min(20000, lpF * 0.8);
      let maxErr = 0;
      for (let i = 0; i < freq.length; i++) {
        if (freq[i] >= pbLo && freq[i] <= pbHi) {
          const err = Math.abs(firResult.realized_mag[i] - (modelMag[i] - firResult.norm_db));
          if (err > maxErr) maxErr = err;
        }
      }

      // Group delay ripple from realized phase (in passband)
      const gdFromPhase = (ph: number[], f: number[]): number[] => {
        const gd: number[] = [];
        for (let i = 0; i < ph.length; i++) {
          let d: number;
          if (i === 0) d = (ph[1] - ph[0]) / (f[1] - f[0]);
          else if (i === ph.length - 1) d = (ph[i] - ph[i - 1]) / (f[i] - f[i - 1]);
          else d = (ph[i + 1] - ph[i - 1]) / (f[i + 1] - f[i - 1]);
          gd.push(-d / 360 * 1000); // degrees → ms
        }
        return gd;
      };
      const gd = gdFromPhase(firResult.realized_phase, freq);
      let gdMin = Infinity, gdMax = -Infinity;
      for (let i = 0; i < freq.length; i++) {
        if (freq[i] >= pbLo && freq[i] <= pbHi && isFinite(gd[i])) {
          if (gd[i] < gdMin) gdMin = gd[i];
          if (gd[i] > gdMax) gdMax = gd[i];
        }
      }
      const gdRipple = isFinite(gdMax - gdMin) ? gdMax - gdMin : 0;

      setExportMetrics({
        taps: firResult.taps, sampleRate: firResult.sample_rate, window: win,
        phaseLabel: allLinear ? "Linear-Phase" : "Min-Phase",
        peqCount: peqBands.length, normDb: firResult.norm_db,
        causality: Math.round(firResult.causality * 100),
        preRingMs: Math.round(preRingMs * 100) / 100,
        maxMagErr: Math.round(maxErr * 100) / 100,
        gdRippleMs: Math.round(gdRipple * 100) / 100,
      });

      // Derive colors from band
      const bandColor = band.color ?? PEQ_COLOR;
      const ecf = bandColorFamily(bandColor);
      const expClr = { model: ecf.target, fir: ecf.corrected, modelPhase: ecf.targetPhase, firPhase: ecf.correctedPhase };
      setExportColors(expClr);

      // Render chart
      try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
      if (!containerRef) return;
      const rect = containerRef.getBoundingClientRect();

      // Wrap phase for display
      const wrappedModelPhase = wrapPhase(modelPhase);
      const wrappedFirPhase = wrapPhase(firResult.realized_phase);

      const opts: uPlot.Options = {
        width: Math.max(rect.width, 400), height: Math.max(rect.height, 200),
        series: [
          {},
          { label: "Model dB", stroke: expClr.model, width: 2, scale: "mag" },
          { label: "FIR dB", stroke: expClr.fir, width: 2, scale: "mag" },
          { label: "Model °", stroke: expClr.modelPhase, width: 1, dash: [4, 4], scale: "phase" },
          { label: "FIR °", stroke: expClr.firPhase, width: 1, dash: [4, 4], scale: "phase" },
        ],
        scales: {
          x: { min: 20, max: 20000, distr: 3 },
          mag: { auto: false, range: () => [curMagMin, curMagMax] as uPlot.Range.MinMax },
          phase: { auto: false, range: [-180, 180] },
        },
        axes: [
          { stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
            values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v >= 1000 ? (v/1000)+"k" : String(Math.round(v))) },
          { label: "dB", scale: "mag", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
            values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)), size: 50 },
          { label: "Phase (°)", scale: "phase", side: 1, stroke: "#9b9ba6",
            grid: { show: false }, ticks: { stroke: "rgba(255,255,255,0.20)" } },
        ],
        legend: { show: false },
        cursor: { drag: { x: false, y: false, setScale: false } },
        hooks: {
          setCursor: [(u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= u.data[0].length) { setCursorFreq("—"); setCursorSPL("—"); return; }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? fmtFreq(f) : "—");
            const m = u.data[1]?.[idx]; const r = u.data[2]?.[idx];
            const mp = u.data[3]?.[idx]; const rp = u.data[4]?.[idx];
            const vals: { label: string; color: string; value: string }[] = [];
            if (m != null) vals.push({ label: "Model", color: expClr.model, value: (m as number).toFixed(1) + " dB" });
            if (r != null) vals.push({ label: "FIR", color: expClr.fir, value: (r as number).toFixed(1) + " dB" });
            if (mp != null) vals.push({ label: "Model°", color: expClr.modelPhase, value: (mp as number).toFixed(1) + "°" });
            if (rp != null) vals.push({ label: "FIR°", color: expClr.firPhase, value: (rp as number).toFixed(1) + "°" });
            setCursorValues(vals);
            setCursorSPL(vals.map(v => `${v.label}: ${v.value}`).join(" ") || "—");
          }],
        },
      };
      // Compute Y fit range from data
      let expMin = Infinity, expMax = -Infinity;
      for (const arr of [normModelMag, firResult.realized_mag]) {
        for (const v of arr) { if (v > -190 && isFinite(v)) { if (v < expMin) expMin = v; if (v > expMax) expMax = v; } }
      }
      if (isFinite(expMin)) {
        const expPad = Math.max(2, (expMax - expMin) * 0.1);
        // Only auto-fit if not already zoomed (persistedMagMin is from freq tab zoom)
        if (persistedMagMin == null) {
          curMagMin = expMin - expPad;
          curMagMax = expMax + expPad;
        }
      }
      if (!isSum()) setShowLegend(false);
      // Save FIR data for snapshot capture
      lastExportData.freq = [...freq]; lastExportData.mag = [...firResult.realized_mag]; lastExportData.phase = [...wrappedFirPhase];
      // Export snapshots
      const expData: number[][] = [freq, normModelMag, firResult.realized_mag, wrappedModelPhase, wrappedFirPhase];
      {
        const b = activeBand();
        const snaps = b ? plotSnapshots(b.id, "export") : [];
        const interpSnap = (srcFreq: number[], srcData: number[]) => freq.map(f => {
          if (f < srcFreq[0] || f > srcFreq[srcFreq.length - 1]) return NaN;
          let lo = 0, hi = srcFreq.length - 1;
          while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (srcFreq[mid] <= f) lo = mid; else hi = mid; }
          const dt = srcFreq[hi] - srcFreq[lo];
          const frac = dt > 0 ? (f - srcFreq[lo]) / dt : 0;
          return srcData[lo] + frac * (srcData[hi] - srcData[lo]);
        });
        for (const snap of snaps) {
          if (!snap.freq || (!snap.exportMag && !snap.exportPhase)) continue;
          if (snap.exportMag) {
            opts.series.push({ label: snap.label, stroke: snap.color, width: 1, dash: [4, 3], scale: "mag" });
            expData.push(interpSnap(snap.freq, snap.exportMag));
          }
          if (snap.exportPhase) {
            opts.series.push({ label: snap.label + " °", stroke: snap.color + "80", width: 1, dash: [4, 3], scale: "phase" });
            expData.push(interpSnap(snap.freq, snap.exportPhase));
          }
        }
      }
      try {
        chart = new uPlot(opts, expData as uPlot.AlignedData, containerRef);
        // Apply persisted visibility
        if (!showExpModel()) chart.setSeries(1, { show: false });
        if (!showExpFir()) chart.setSeries(2, { show: false });
        if (!showExpModelPh()) chart.setSeries(3, { show: false });
        if (!showExpFirPh()) chart.setSeries(4, { show: false });
      } catch (e) { console.error(e); }
      setExportComputing(false);
    } catch (e) {
      console.error("Export tab render failed:", e);
      setExportComputing(false);
    }
  }

  // ----------------------------------------------------------------
  // IR / Step / GD rendering (time-domain tabs)
  // ----------------------------------------------------------------
  async function renderTimeTab(mode: "ir" | "step" | "gd", sumMode: boolean, band: BandState | null) {
    const gen = ++renderGen;
    // Snapshot toggle state — untrack to prevent main effect re-trigger on toggle
    const irCfg = untrack(() => ({
      db: irDbMode(),
      masking: irShowMasking(),
    }));

    // Collect bands with phase data (phase must be non-empty array)
    const allWithPhase = sumMode
      ? appState.bands.filter(b => b.measurement?.phase && b.measurement.phase.length > 0)
      : (band?.measurement?.phase && band.measurement.phase.length > 0 ? [band] : []);
    // In SUM mode, filter by band visibility from legend matrix (for coherent sum only)
    const excluded = untrack(() => irExcludedBands());
    const bands = sumMode
      ? allWithPhase.filter(b => !excluded.has(b.name))
      : allWithPhase;
    // All bands for per-band curves (including excluded from sum)
    const allBands = allWithPhase;

    // Read alignment delays synchronously as plain array (store proxy safe)
    // untrack: prevent creating tracking dependency on .alignmentDelay
    // (effect is already triggered by bandsVersion signal from markDirty)
    const irDelayByName: Record<string, number> = {};
    untrack(() => {
      for (let i = 0; i < appState.bands.length; i++) {
        const d = appState.bands[i].alignmentDelay;
        const v = typeof d === "number" ? d : 0;
        irDelayByName[appState.bands[i].name] = v;
      }
    });

    if (allBands.length === 0) {
      try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
      setShowLegend(false);
      setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
      return;
    }

    try {
      // For GD: compute from phase directly (no IPC needed)
      if (mode === "gd") {
        if (bands.length === 0) return;
        const b = bands[0];
        const freq = [...b.measurement!.freq];
        const phase = [...b.measurement!.phase!];
        const magnitude = [...b.measurement!.magnitude];

        // Unwrap phase for GD computation (prevents spikes at ±180° boundaries)
        let gdFreq = freq;
        let gdPhase: number[];
        if (sumMode && bands.length > 1) {
          const n = freq.length;
          const sumRe = new Float64Array(n);
          const sumIm = new Float64Array(n);
          for (const sb of bands) {
            const sign = sb.inverted ? -1 : 1;
            const gdDelay = irDelayByName[sb.name] ?? 0;
            for (let j = 0; j < n; j++) {
              const amp = Math.pow(10, (sb.measurement!.magnitude[j] ?? -200) / 20) * sign;
              const phRad = ((sb.measurement!.phase![j] ?? 0) + 360 * freq[j] * gdDelay) * Math.PI / 180;
              sumRe[j] += amp * Math.cos(phRad);
              sumIm[j] += amp * Math.sin(phRad);
            }
          }
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
        } else {
          // Band mode: unwrap phase before GD computation
          const uw: number[] = [phase[0]];
          for (let i = 1; i < phase.length; i++) {
            let diff = phase[i] - phase[i - 1];
            while (diff > 180) diff -= 360;
            while (diff <= -180) diff += 360;
            uw.push(uw[i - 1] + diff);
          }
          gdPhase = uw;
        }

        // Measurement GD
        const measGd = computeGroupDelay(gdFreq, gdPhase);
        if (gen !== renderGen) return;

        // Target GD (band mode only)
        let targetGdMs: number[] | null = null;
        if (!sumMode && band && band.targetEnabled) {
          try {
            const targetCurve = JSON.parse(JSON.stringify(band.target));
            const tResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: targetCurve, freq });
            if (gen !== renderGen) return;
            let tPh = tResp.phase;
            if (isGaussianMinPhase(band.target.high_pass) || isGaussianMinPhase(band.target.low_pass)) {
              tPh = await addGaussianMinPhase(freq, tPh, band.target.high_pass, band.target.low_pass);
              if (gen !== renderGen) return;
            }
            const tgd = computeGroupDelay(freq, tPh);
            targetGdMs = tgd.gdMs;
          } catch (_) {}
        }

        // Corrected GD (band mode only, meas + PEQ + cross-section)
        let corrGdMs: number[] | null = null;
        if (!sumMode && band && band.targetEnabled) {
          try {
            let corrPh = [...phase];
            let corrMag = [...magnitude];
            const peqBands = band.peqBands?.filter((p: PeqBand) => p.enabled) ?? [];
            if (peqBands.length > 0) {
              const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq, bands: peqBands });
              if (gen !== renderGen) return;
              corrPh = corrPh.map((v, i) => v + pp[i]);
              corrMag = corrMag.map((v, i) => v + pm[i]);
            }
            if (band.target.high_pass || band.target.low_pass) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq, highPass: band.target.high_pass, lowPass: band.target.low_pass,
              });
              if (gen !== renderGen) return;
              corrPh = corrPh.map((v, i) => v + xp[i]);
              corrMag = corrMag.map((v, i) => v + xm[i]);
            }
            // Gaussian min-phase: add per-filter Hilbert-derived phase to corrected phase
            if (isGaussianMinPhase(band.target.high_pass) || isGaussianMinPhase(band.target.low_pass)) {
              corrPh = await addGaussianMinPhase(freq, corrPh, band.target.high_pass, band.target.low_pass);
              if (gen !== renderGen) return;
            }
            const cgd = computeGroupDelay(freq, corrPh);
            corrGdMs = cgd.gdMs;
          } catch (_) {}
        }

        // GD colors from band
        const bandColor = (!sumMode && band) ? band.color : DEFAULT_GD_COLORS.meas;
        const cf = bandColorFamily(bandColor);
        const stripAlpha = (c: string) => c.length === 9 ? c.slice(0, 7) : c;
        const gdClr = { meas: stripAlpha(cf.meas), target: cf.target, corr: cf.corrected };
        setGdColors(gdClr);

        // Snapshot visibility (untracked)
        const gdCfg = untrack(() => ({ meas: showGdMeas(), target: showGdTarget(), corr: showGdCorr() }));

        // Save corrected GD data for snapshot capture
        // Save GD data for snapshot: use corrected if available and visible, else measurement
        lastGdData.freq = [...measGd.freqOut];
        if (corrGdMs && gdCfg.corr) {
          lastGdData.gdMs = [...corrGdMs];
        } else if (gdCfg.meas) {
          lastGdData.gdMs = [...measGd.gdMs];
        } else if (targetGdMs && gdCfg.target) {
          lastGdData.gdMs = [...targetGdMs];
        } else {
          lastGdData.gdMs = [];
        }

        renderGdChart(measGd.freqOut, measGd.gdMs, targetGdMs, corrGdMs, gdClr, gdCfg);
        return;
      }

      // IR or Step: compute via IPC
      const b = bands[0];
      const freq = [...b.measurement!.freq];
      const sr = b.measurement!.sample_rate ?? 48000;

      if (sumMode) {
        // ============================================================
        // SUM MODE: per-band IR/Step + coherent sum IR/Step
        // ============================================================
        const n = freq.length;

        // --- Compute per-band measurement impulses in parallel (all bands, incl. excluded) ---
        // Normalize each band to 0 dB peak before IFFT — same as sum does
        // This ensures per-band IR shapes match what actually goes into the SUM
        const measPromises = allBands.map(async (sb) => {
          const sbFreq = [...sb.measurement!.freq];
          const sbMag = [...sb.measurement!.magnitude];
          const sbPh = [...sb.measurement!.phase!];
          const sbSr = sb.measurement!.sample_rate ?? 48000;
          const delay = irDelayByName[sb.name] ?? 0;

          // Normalize to 0 dB peak (same as sum normalization)
          let peakMag = -Infinity;
          for (let j = 0; j < sbMag.length; j++) {
            if ((sbMag[j] ?? -200) > peakMag) peakMag = sbMag[j] ?? -200;
          }
          const offset = -peakMag;
          const normMag = sbMag.map(v => (v ?? -200) + offset);

          let rotatedPhase = sbPh;
          if (delay !== 0) {
            rotatedPhase = sbPh.map((p, j) => p + 360 * sbFreq[j] * delay);
          }

          return invoke<{ time: number[]; impulse: number[]; step: number[]; raw_peak: number; step_raw_peak: number }>(
            "compute_impulse",
            { freq: sbFreq, magnitude: normMag, phase: rotatedPhase, sampleRate: sbSr }
          ).catch(() => null);
        });
        const measResults = await Promise.all(measPromises);
        if (gen !== renderGen) return;


        // Coherent sum measurement — resample all bands to a dense common grid first
        // (SPL tab already does this via interpolate_log; IR SUM needs the same)
        const nGrid = 2048;
        // Common freq range: union of all bands' measurement grids
        let sumFreqMin = Infinity, sumFreqMax = -Infinity;
        for (const sb of allBands) {
          const sf = sb.measurement!.freq;
          if (sf[0] < sumFreqMin) sumFreqMin = sf[0];
          if (sf[sf.length - 1] > sumFreqMax) sumFreqMax = sf[sf.length - 1];
        }

        // Build a log-spaced common grid using the widest-range band's point count
        let maxPts = 0;
        for (const sb of allBands) { maxPts = Math.max(maxPts, sb.measurement!.freq.length); }
        const gridPts = Math.max(maxPts, nGrid);

        // Resample each band onto the common grid
        const interpTasks = allBands.map(async (sb) => {
          const m = sb.measurement!;
          const [rFreq, rMag, rPhase] = await invoke<[number[], number[], number[] | null]>(
            "interpolate_log",
            { freq: m.freq, magnitude: m.magnitude, phase: m.phase, nPoints: gridPts, fMin: sumFreqMin, fMax: sumFreqMax }
          );
          return { freq: rFreq, mag: rMag, phase: rPhase ?? rMag.map(() => 0), delay: irDelayByName[sb.name] ?? 0, inverted: sb.inverted };
        });
        const resampledBands = await Promise.all(interpTasks);
        if (gen !== renderGen) return;
        const sumFreq = resampledBands[0].freq;
        const sumN = sumFreq.length;

        const sumRe = new Float64Array(sumN);
        const sumIm = new Float64Array(sumN);
        for (const bd of resampledBands) {
          let peakMag = -Infinity;
          for (let j = 0; j < bd.mag.length; j++) {
            if ((bd.mag[j] ?? -200) > peakMag) peakMag = bd.mag[j] ?? -200;
          }
          const offset = -peakMag;
          const sign = bd.inverted ? -1 : 1;
          for (let j = 0; j < sumN; j++) {
            const amp = Math.pow(10, ((bd.mag[j] ?? -200) + offset) / 20) * sign;
            const phRad = ((bd.phase[j] ?? 0) + 360 * sumFreq[j] * bd.delay) * Math.PI / 180;
            sumRe[j] += amp * Math.cos(phRad);
            sumIm[j] += amp * Math.sin(phRad);
          }
        }
        const sumMeasMag: number[] = [];
        const sumMeasPh: number[] = [];
        for (let j = 0; j < sumN; j++) {
          const amplitude = Math.sqrt(sumRe[j] * sumRe[j] + sumIm[j] * sumIm[j]);
          sumMeasMag.push(amplitude > 0 ? 20 * Math.log10(amplitude) : -200);
          sumMeasPh.push(Math.atan2(sumIm[j], sumRe[j]) * 180 / Math.PI);
        }
        const sumMeasResult = await invoke<{ time: number[]; impulse: number[]; step: number[]; raw_peak: number; step_raw_peak: number }>("compute_impulse", {
          freq: sumFreq, magnitude: sumMeasMag, phase: sumMeasPh, sampleRate: sr,
        });
        if (gen !== renderGen) return;
        const measSum = {
          timeMs: sumMeasResult.time.map(t => t * 1000),
          impulse: sumMeasResult.impulse,
          step: sumMeasResult.step,
        };

        // Build per-band measurement IrBandData — each normalized to 0dB peak (peak=100%)
        const measBands: IrBandData[] = [];
        for (let i = 0; i < allBands.length; i++) {
          const r = measResults[i];
          if (!r) continue;
          measBands.push({
            bandName: allBands[i].name,
            bandColor: allBands[i].color,
            timeMs: r.time.map(t => t * 1000),
            impulse: r.impulse,
            step: r.step,
            rawPeak: r.raw_peak || 0,
            stepRawPeak: r.step_raw_peak || 0,
          });
        }

        // --- Per-band targets ---
        const targetBands: IrBandData[] = [];
        // Common reference from SUM measurement level (200-2000 Hz)
        let sumRef = 0, nRef = 0;
        for (let i = 0; i < sumFreq.length; i++) {
          if (sumFreq[i] >= 200 && sumFreq[i] <= 2000) { sumRef += sumMeasMag[i]; nRef++; }
        }
        const commonRef = nRef > 0 ? sumRef / nRef : 0;

        // Per-band target: use auto-ref from THAT band's measurement
        const tgtBandPromises = allBands.map(async (sb) => {
          if (!sb.targetEnabled) return null;
          try {
            const tc = JSON.parse(JSON.stringify(sb.target));
            // Per-band auto-ref from this band's measurement
            let bRef = 0, bN = 0;
            for (let i = 0; i < sb.measurement!.freq.length; i++) {
              if (sb.measurement!.freq[i] >= 200 && sb.measurement!.freq[i] <= 2000) { bRef += sb.measurement!.magnitude[i]; bN++; }
            }
            tc.reference_level_db += bN > 0 ? bRef / bN : 0;
            const tResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: tc, freq });
            if (gen !== renderGen) return null;
            let tMag = tResp.magnitude;
            let tPh = tResp.phase;
            if (sb.target.high_pass || sb.target.low_pass) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq, highPass: sb.target.high_pass, lowPass: sb.target.low_pass,
              });
              if (gen !== renderGen) return null;
              tMag = tMag.map((v: number, i: number) => v + xm[i]);
              tPh = tPh.map((v: number, i: number) => v + xp[i]);
            }
            // Gaussian min-phase: add per-filter Hilbert-derived minimum phase
            if (isGaussianMinPhase(sb.target.high_pass) || isGaussianMinPhase(sb.target.low_pass)) {
              tPh = await addGaussianMinPhase(freq, tPh, sb.target.high_pass, sb.target.low_pass);
              if (gen !== renderGen) return null;
            }
            // Normalize to 0 dB peak before IFFT (same as sum normalization)
            let tPeakMag = -Infinity;
            for (let j = 0; j < tMag.length; j++) {
              if ((tMag[j] ?? -200) > tPeakMag) tPeakMag = tMag[j] ?? -200;
            }
            const tOffset = -tPeakMag;
            const normMag = tMag.map(v => (v ?? -200) + tOffset);

            // Apply delay as phase rotation BEFORE IFFT
            const delay = irDelayByName[sb.name] ?? 0;
            let adjPhase = tPh;
            if (delay !== 0) {
              adjPhase = tPh.map((p, j) => p + 360 * freq[j] * delay);
            }
            const r = await invoke<{ time: number[]; impulse: number[]; step: number[]; raw_peak: number; step_raw_peak: number }>("compute_impulse", {
              freq, magnitude: normMag, phase: adjPhase, sampleRate: sr,
            });
            if (gen !== renderGen) return null;
            return { bandName: sb.name, bandColor: sb.color, timeMs: r.time.map(t => t * 1000), impulse: r.impulse, step: r.step, rawPeak: r.raw_peak || 0, stepRawPeak: r.step_raw_peak || 0 } as IrBandData;
          } catch (_) { return null; }
        });
        const tgtBandResults = await Promise.all(tgtBandPromises);
        if (gen !== renderGen) return;
        for (const r of tgtBandResults) { if (r) targetBands.push(r); }

        // Coherent sum target (COMMON reference for all bands)
        // Resample target responses onto the same dense common grid
        let targetSum: { timeMs: number[]; impulse: number[]; step: number[] } | null = null;
        try {
          const tgtSumData = bands.map(sb => ({
            targetEnabled: sb.targetEnabled,
            target: JSON.parse(JSON.stringify(sb.target)),
            inverted: sb.inverted,
            delay: irDelayByName[sb.name] ?? 0,
          }));

          // Evaluate targets and resample onto sumFreq grid
          const tgtEvalTasks = tgtSumData.map(async (bd) => {
            if (!bd.targetEnabled) return null;
            const tc = { ...bd.target };
            tc.reference_level_db += commonRef;
            const tResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: tc, freq });
            if (gen !== renderGen) return null;
            let tMag = tResp.magnitude;
            let tPh = tResp.phase;
            if (bd.target.high_pass || bd.target.low_pass) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq, highPass: bd.target.high_pass, lowPass: bd.target.low_pass,
              });
              if (gen !== renderGen) return null;
              tMag = tMag.map((v: number, i: number) => v + xm[i]);
              tPh = tPh.map((v: number, i: number) => v + xp[i]);
            }
            // Resample onto dense common grid
            const [rFreq, rMag, rPhase] = await invoke<[number[], number[], number[] | null]>(
              "interpolate_log",
              { freq, magnitude: tMag, phase: tPh, nPoints: sumN, fMin: sumFreqMin, fMax: sumFreqMax }
            );
            if (gen !== renderGen) return null;
            return {
              mag: rMag,
              phase: rPhase ?? rMag.map(() => 0),
              delay: bd.delay,
              inverted: bd.inverted,
            };
          });
          const tgtResampled = await Promise.all(tgtEvalTasks);
          if (gen !== renderGen) return;

          const tgtRe = new Float64Array(sumN);
          const tgtIm = new Float64Array(sumN);
          let anyTarget = false;
          for (const bd of tgtResampled) {
            if (!bd) continue;
            anyTarget = true;
            // Normalize by peak magnitude so bands contribute equally
            let tPeakMag = -Infinity;
            for (let j = 0; j < bd.mag.length; j++) {
              if ((bd.mag[j] ?? -200) > tPeakMag) tPeakMag = bd.mag[j] ?? -200;
            }
            const tOffset = -tPeakMag;
            const sign = bd.inverted ? -1 : 1;
            for (let j = 0; j < sumN; j++) {
              const amp = Math.pow(10, ((bd.mag[j] ?? -200) + tOffset) / 20) * sign;
              const phRad = ((bd.phase[j] ?? 0) + 360 * sumFreq[j] * bd.delay) * Math.PI / 180;
              tgtRe[j] += amp * Math.cos(phRad);
              tgtIm[j] += amp * Math.sin(phRad);
            }
          }
          if (anyTarget) {
            const tgtMag: number[] = [];
            const tgtPh: number[] = [];
            for (let j = 0; j < sumN; j++) {
              const amp = Math.sqrt(tgtRe[j] * tgtRe[j] + tgtIm[j] * tgtIm[j]);
              tgtMag.push(amp > 0 ? 20 * Math.log10(amp) : -200);
              tgtPh.push(Math.atan2(tgtIm[j], tgtRe[j]) * 180 / Math.PI);
            }
            const r = await invoke<{ time: number[]; impulse: number[]; step: number[]; raw_peak: number; step_raw_peak: number }>("compute_impulse", {
              freq: sumFreq, magnitude: tgtMag, phase: tgtPh, sampleRate: sr,
            });
            if (gen !== renderGen) return;
            targetSum = { timeMs: r.time.map(t => t * 1000), impulse: r.impulse, step: r.step };
          }
        } catch (e) { console.error("TGT SUM ERR:", e); }

        // --- Per-band corrected (all bands) ---
        const corrBands: IrBandData[] = [];
        const corrBandPromises = allBands.map(async (sb) => {
          try {
            const sbFreq = [...sb.measurement!.freq];
            let cMag = [...sb.measurement!.magnitude];
            let cPh = [...sb.measurement!.phase!];
            const sbSr = sb.measurement!.sample_rate ?? 48000;
            const peqBands = sb.peqBands?.filter((p: PeqBand) => p.enabled) ?? [];
            if (peqBands.length > 0) {
              const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq: sbFreq, bands: peqBands });
              if (gen !== renderGen) return null;
              cMag = cMag.map((v, i) => v + (pm[i] ?? 0));
              cPh = cPh.map((v, i) => v + (pp[i] ?? 0));
            }
            if (sb.targetEnabled && (sb.target.high_pass || sb.target.low_pass)) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq: sbFreq, highPass: sb.target.high_pass, lowPass: sb.target.low_pass,
              });
              if (gen !== renderGen) return null;
              cMag = cMag.map((v, i) => v + (xm[i] ?? 0));
              cPh = cPh.map((v, i) => v + (xp[i] ?? 0));
            }
            // Gaussian min-phase: add per-filter Hilbert-derived phase to corrected phase
            if (sb.targetEnabled && (isGaussianMinPhase(sb.target.high_pass) || isGaussianMinPhase(sb.target.low_pass))) {
              cPh = await addGaussianMinPhase(sbFreq, cPh, sb.target.high_pass, sb.target.low_pass);
              if (gen !== renderGen) return null;
            }
            // Normalize to 0 dB peak before IFFT (same as sum normalization)
            let cPeakMag = -Infinity;
            for (let j = 0; j < cMag.length; j++) {
              if ((cMag[j] ?? -200) > cPeakMag) cPeakMag = cMag[j] ?? -200;
            }
            const cOffset = -cPeakMag;
            const normMag = cMag.map(v => (v ?? -200) + cOffset);

            // Apply delay as phase rotation BEFORE IFFT
            const delay = irDelayByName[sb.name] ?? 0;
            let adjPhase = cPh;
            if (delay !== 0) {
              adjPhase = cPh.map((p, j) => p + 360 * sbFreq[j] * delay);
            }
            const r = await invoke<{ time: number[]; impulse: number[]; step: number[]; raw_peak: number; step_raw_peak: number }>("compute_impulse", {
              freq: sbFreq, magnitude: normMag, phase: adjPhase, sampleRate: sbSr,
            });
            if (gen !== renderGen) return null;
            return { bandName: sb.name, bandColor: sb.color, timeMs: r.time.map(t => t * 1000), impulse: r.impulse, step: r.step, rawPeak: r.raw_peak || 0, stepRawPeak: r.step_raw_peak || 0 } as IrBandData;
          } catch (_) { return null; }
        });
        const corrBandResults = await Promise.all(corrBandPromises);
        if (gen !== renderGen) return;
        for (const r of corrBandResults) { if (r) corrBands.push(r); }

        // Coherent sum corrected — resample corrected responses onto dense common grid
        let corrSum: { timeMs: number[]; impulse: number[]; step: number[] } | null = null;
        try {
          const corrSumData = bands.map(sb => ({
            freq: [...sb.measurement!.freq],
            magnitude: [...sb.measurement!.magnitude],
            phase: [...sb.measurement!.phase!],
            targetEnabled: sb.targetEnabled,
            target: JSON.parse(JSON.stringify(sb.target)),
            peqBands: sb.peqBands ? JSON.parse(JSON.stringify(sb.peqBands.filter((p: PeqBand) => p.enabled))) : [],
            inverted: sb.inverted,
            delay: irDelayByName[sb.name] ?? 0,
          }));

          // Process each band's corrected response, then resample onto sumFreq
          const corrEvalTasks = corrSumData.map(async (bd) => {
            const sbFreq = bd.freq;
            let cMag = [...bd.magnitude];
            let cPh = [...bd.phase];
            if (bd.peqBands.length > 0) {
              const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq: sbFreq, bands: bd.peqBands });
              if (gen !== renderGen) return null;
              cMag = cMag.map((v, i) => v + (pm[i] ?? 0));
              cPh = cPh.map((v, i) => v + (pp[i] ?? 0));
            }
            if (bd.targetEnabled && (bd.target.high_pass || bd.target.low_pass)) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq: sbFreq, highPass: bd.target.high_pass, lowPass: bd.target.low_pass,
              });
              if (gen !== renderGen) return null;
              cMag = cMag.map((v, i) => v + (xm[i] ?? 0));
              cPh = cPh.map((v, i) => v + (xp[i] ?? 0));
            }
            // Gaussian min-phase
            if (bd.targetEnabled && (isGaussianMinPhase(bd.target.high_pass) || isGaussianMinPhase(bd.target.low_pass))) {
              cPh = await addGaussianMinPhase(sbFreq, cPh, bd.target.high_pass, bd.target.low_pass);
              if (gen !== renderGen) return null;
            }
            // Resample onto dense common grid
            const [rFreq, rMag, rPhase] = await invoke<[number[], number[], number[] | null]>(
              "interpolate_log",
              { freq: sbFreq, magnitude: cMag, phase: cPh, nPoints: sumN, fMin: sumFreqMin, fMax: sumFreqMax }
            );
            if (gen !== renderGen) return null;
            return {
              mag: rMag,
              phase: rPhase ?? rMag.map(() => 0),
              delay: bd.delay,
              inverted: bd.inverted,
            };
          });
          const corrResampled = await Promise.all(corrEvalTasks);
          if (gen !== renderGen) return;

          const corrRe = new Float64Array(sumN);
          const corrIm = new Float64Array(sumN);
          let anyCorrected = false;
          for (const bd of corrResampled) {
            if (!bd) continue;
            // Normalize by peak magnitude so bands contribute equally
            let peakMag = -Infinity;
            for (let j = 0; j < bd.mag.length; j++) {
              if ((bd.mag[j] ?? -200) > peakMag) peakMag = bd.mag[j] ?? -200;
            }
            const offset = -peakMag;
            const sign = bd.inverted ? -1 : 1;
            for (let j = 0; j < sumN; j++) {
              const amp = Math.pow(10, ((bd.mag[j] ?? -200) + offset) / 20) * sign;
              const phRad = ((bd.phase[j] ?? 0) + 360 * sumFreq[j] * bd.delay) * Math.PI / 180;
              corrRe[j] += amp * Math.cos(phRad);
              corrIm[j] += amp * Math.sin(phRad);
            }
            anyCorrected = true;
          }
          if (anyCorrected) {
            const cMagSum: number[] = [];
            const cPhSum: number[] = [];
            for (let j = 0; j < sumN; j++) {
              const amp = Math.sqrt(corrRe[j] * corrRe[j] + corrIm[j] * corrIm[j]);
              cMagSum.push(amp > 0 ? 20 * Math.log10(amp) : -200);
              cPhSum.push(Math.atan2(corrIm[j], corrRe[j]) * 180 / Math.PI);
            }
            const r = await invoke<{ time: number[]; impulse: number[]; step: number[]; raw_peak: number; step_raw_peak: number }>("compute_impulse", {
              freq: sumFreq, magnitude: cMagSum, phase: cPhSum, sampleRate: sr,
            });
            if (gen !== renderGen) return;
            corrSum = { timeMs: r.time.map(t => t * 1000), impulse: r.impulse, step: r.step };
          }
        } catch (e) { console.error("CORR SUM ERR:", e); }

        // HP freq for masking zone (use lowest band's HP or 20)
        const hpFreq = bands.reduce((min, sb) => {
          const hp = sb.target?.high_pass?.freq_hz;
          return hp && hp < min ? hp : min;
        }, 20);

        // Save IR data for snapshot capture
        // Priority: corrected → target → measurement
        {
          const src = corrSum ?? (corrBands.length > 0 ? corrBands[0] : null)
            ?? targetSum ?? (targetBands.length > 0 ? targetBands[0] : null)
            ?? measSum ?? (measBands.length > 0 ? measBands[0] : null);
          if (src) {
            let pkIdx = 0, pkV = 0;
            for (let i = 0; i < src.impulse.length; i++) { if (Math.abs(src.impulse[i]) > pkV) { pkV = Math.abs(src.impulse[i]); pkIdx = i; } }
            const pkT = src.timeMs[pkIdx] ?? 0;
            lastIrData.timeMs = src.timeMs.map(t => t - pkT);
            lastIrData.impulse = [...src.impulse]; lastIrData.step = [...src.step];
          }
        }

        renderIrStepChart(measBands, measSum, targetBands, targetSum, corrBands, corrSum, hpFreq, irCfg);

      } else {
        // ============================================================
        // BAND MODE: single band IR/Step (wrap into per-band format)
        // ============================================================
        const magnitude = [...b.measurement!.magnitude];
        const phase = [...b.measurement!.phase!];

        let result: { time: number[]; impulse: number[]; step: number[] };
        try {
          result = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
            freq, magnitude, phase, sampleRate: sr,
          });
        } catch (_) {
          return;
        }
        if (gen !== renderGen) return;

        const measBands: IrBandData[] = [{
          bandName: band!.name,
          bandColor: band!.color,
          timeMs: result.time.map(t => t * 1000),
          impulse: result.impulse,
          step: result.step,
        }];

        // Target impulse (single band)
        let targetResult: { time: number[]; impulse: number[]; step: number[] } | null = null;
        if (band && band.targetEnabled) {
          try {
            const targetCurve = JSON.parse(JSON.stringify(band.target));
            let sum = 0, n2 = 0;
            for (let i = 0; i < freq.length; i++) {
              if (freq[i] >= 200 && freq[i] <= 2000) { sum += magnitude[i]; n2++; }
            }
            targetCurve.reference_level_db += n2 > 0 ? sum / n2 : 0;
            const tResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: targetCurve, freq });
            if (gen !== renderGen) return;
            let tPh2 = tResp.phase;
            // Gaussian min-phase: add per-filter Hilbert-derived phase
            if (isGaussianMinPhase(band!.target.high_pass) || isGaussianMinPhase(band!.target.low_pass)) {
              tPh2 = await addGaussianMinPhase(freq, tPh2, band!.target.high_pass, band!.target.low_pass);
              if (gen !== renderGen) return;
            }
            targetResult = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
              freq, magnitude: tResp.magnitude, phase: tPh2, sampleRate: sr,
            });
            if (gen !== renderGen) return;
          } catch (_) {}
        }
        const targetBands: IrBandData[] = targetResult ? [{
          bandName: band!.name,
          bandColor: band!.color,
          timeMs: targetResult.time.map(t => t * 1000),
          impulse: targetResult.impulse,
          step: targetResult.step,
        }] : [];

        // Corrected impulse (single band: meas + PEQ + cross-section)
        let corrResult: { time: number[]; impulse: number[]; step: number[] } | null = null;
        if (band && band.targetEnabled) {
          try {
            const targetCurve = JSON.parse(JSON.stringify(band.target));
            let sum2 = 0, n3 = 0;
            for (let i = 0; i < freq.length; i++) { if (freq[i] >= 200 && freq[i] <= 2000) { sum2 += magnitude[i]; n3++; } }
            targetCurve.reference_level_db += n3 > 0 ? sum2 / n3 : 0;
            const tResp2 = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", { target: targetCurve, freq });
            if (gen !== renderGen) return;
            const peqBands = band.peqBands.filter((p: PeqBand) => p.enabled);
            let corrMag = [...magnitude];
            let corrPh = [...phase];
            if (peqBands.length > 0) {
              const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", { freq, bands: peqBands });
              if (gen !== renderGen) return;
              corrMag = corrMag.map((v, i) => v + pm[i]);
              corrPh = corrPh.map((v, i) => v + pp[i]);
            }
            if (band.target.high_pass || band.target.low_pass) {
              const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq, highPass: band.target.high_pass, lowPass: band.target.low_pass,
              });
              if (gen !== renderGen) return;
              corrMag = corrMag.map((v, i) => v + xm[i]);
              corrPh = corrPh.map((v, i) => v + xp[i]);
            }
            // Gaussian min-phase: add per-filter Hilbert-derived phase to corrected phase
            if (isGaussianMinPhase(band.target.high_pass) || isGaussianMinPhase(band.target.low_pass)) {
              corrPh = await addGaussianMinPhase(freq, corrPh, band.target.high_pass, band.target.low_pass);
              if (gen !== renderGen) return;
            }
            corrResult = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
              freq, magnitude: corrMag, phase: corrPh, sampleRate: sr,
            });
            if (gen !== renderGen) return;
          } catch (_) {}
        }
        const corrBands: IrBandData[] = corrResult ? [{
          bandName: band!.name,
          bandColor: band!.color,
          timeMs: corrResult.time.map(t => t * 1000),
          impulse: corrResult.impulse,
          step: corrResult.step,
        }] : [];

        // HP freq for masking zone
        const hpFreq = band?.target?.high_pass?.freq_hz ?? 20;

        // Save IR data for snapshot capture
        // Priority: corrected → target → measurement
        {
          const src = corrBands.length > 0 ? corrBands[0]
            : targetBands.length > 0 ? targetBands[0]
            : measBands[0];
          if (src) {
            let pkIdx = 0, pkV = 0;
            for (let i = 0; i < src.impulse.length; i++) { if (Math.abs(src.impulse[i]) > pkV) { pkV = Math.abs(src.impulse[i]); pkIdx = i; } }
            const pkT = src.timeMs[pkIdx] ?? 0;
            lastIrData.timeMs = src.timeMs.map(t => t - pkT);
            lastIrData.impulse = [...src.impulse]; lastIrData.step = [...src.step];
          }
        }

        renderIrStepChart(measBands, null, targetBands, null, corrBands, null, hpFreq, irCfg);
      }
    } catch (e) {
      console.error("Time tab render failed:", e);
    }
  }

  function renderIrStepChart(
    measBands: IrBandData[],
    measSum: { timeMs: number[]; impulse: number[]; step: number[] } | null,
    targetBands: IrBandData[],
    targetSum: { timeMs: number[]; impulse: number[]; step: number[] } | null,
    corrBands: IrBandData[],
    corrSum: { timeMs: number[]; impulse: number[]; step: number[] } | null,
    hpFreq: number,
    irCfg: { db: boolean; masking: boolean },
  ) {
    try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
    if (!containerRef) return;
    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 200);
    const isDb = irCfg.db;
    const inSum = isSum();
    const toDb = (v: number) => { const a = Math.abs(v); return a > 1e-8 ? 20 * Math.log10(a / 100) : -200; };
    const stripAlpha = (c: string) => c.length === 9 ? c.slice(0, 7) : c;

    // Reference time axis: first measurement band's raw timeMs (no peak alignment)
    const refBand = measBands[0];
    if (!refBand) return;
    const timeMs = [...refBand.timeMs];

    // Find peak time for view centering only (no shifting)
    let refPeakIdx = 0, refPeakVal = 0;
    for (let i = 0; i < refBand.impulse.length; i++) {
      if (Math.abs(refBand.impulse[i]) > refPeakVal) { refPeakVal = Math.abs(refBand.impulse[i]); refPeakIdx = i; }
    }
    const peakTimeMs = refBand.timeMs[refPeakIdx] ?? 0;
    const xViewMin = peakTimeMs - 30;
    const xViewMax = peakTimeMs + 30;
    // Set mutable X range (used by range function, zoomX, scrollX)
    // Guard: discard stale freq-range scales (e.g. 20–20000 from SPL tab)
    if (irUserXScale) {
      const xRange = irUserXScale.max - irUserXScale.min;
      if (xRange <= 0 || xRange >= 5000) irUserXScale = null;
    }
    if (!irUserXScale) {
      irCurXMin = xViewMin;
      irCurXMax = xViewMax;
    } else {
      irCurXMin = irUserXScale.min;
      irCurXMax = irUserXScale.max;
    }
    const maskingMs = hpFreq > 0 ? (1.5 / hpFreq) * 1000 : 20;

    // Resample srcData onto timeMs grid (linear interpolation)
    const resampleOnto = (srcTime: number[], srcData: number[]): number[] => {
      return timeMs.map(t => {
        if (t < srcTime[0] || t > srcTime[srcTime.length - 1]) return isDb ? -200 : 0;
        let lo = 0, hi2 = srcTime.length - 1;
        while (hi2 - lo > 1) { const mid = (lo + hi2) >> 1; if (srcTime[mid] <= t) lo = mid; else hi2 = mid; }
        const dt = srcTime[hi2] - srcTime[lo];
        const frac = dt > 0 ? (t - srcTime[lo]) / dt : 0;
        const v = srcData[lo] + frac * (srcData[hi2] - srcData[lo]);
        return isDb ? toDb(v) : v;
      });
    };

    const applyDb = (data: number[]): number[] => isDb ? data.map(toDb) : data;
    const emptyData = timeMs.map(() => NaN);

    // Build series + data + legend
    const uSeries: uPlot.Series[] = [{}];
    const uDataArr: number[][] = [timeMs];
    const irLegend: LegendEntry[] = [];
    let sIdx = 1;

    // SUM-specific colors — white/silver to avoid collision with band colors (VituixCAD convention: sum = white)
    const sumClr = {
      measIr: SUM_MEAS_COLOR, measStep: SUM_MEAS_COLOR,
      targetIr: SUM_TARGET_COLOR, targetStep: SUM_TARGET_COLOR,
      corrIr: SUM_CORRECTED_COLOR, corrStep: SUM_CORRECTED_COLOR,
    };

    // Helper: add IR+Step series for a band or sum
    const addIrStepPair = (
      label: string, irColor: string, stepColor: string,
      bandTimeMs: number[], impulse: number[], step: number[],
      lineWidth: number, category: "measurement" | "target" | "corrected",
      dash: boolean,
    ) => {
      // Resample onto reference grid if different time axis
      const sameGrid = bandTimeMs.length === timeMs.length
        && Math.abs((bandTimeMs[0] ?? 0) - (timeMs[0] ?? 0)) < 0.001
        && Math.abs((bandTimeMs[bandTimeMs.length - 1] ?? 0) - (timeMs[timeMs.length - 1] ?? 0)) < 0.001;
      const irData = sameGrid ? applyDb(impulse) : resampleOnto(bandTimeMs, impulse);
      const stData = sameGrid ? applyDb(step) : resampleOnto(bandTimeMs, step);

      uSeries.push({ label: label + " IR", stroke: irColor, width: lineWidth, scale: "y", show: true, dash: dash ? [6, 4] : undefined });
      uDataArr.push(irData);
      irLegend.push({ label: label + " IR", color: irColor, dash, visible: true, seriesIdx: sIdx, category });
      sIdx++;

      uSeries.push({ label: label + " Step", stroke: stepColor, width: lineWidth, scale: "y", show: true, dash: dash ? [6, 4] : undefined });
      uDataArr.push(stData);
      irLegend.push({ label: label + " Step", color: stepColor, dash, visible: true, seriesIdx: sIdx, category });
      sIdx++;
    };

    // --- Measurement per-band ---
    for (const rawBd of measBands) {
      if (inSum) {
        const cf = bandColorFamily(rawBd.bandColor);
        addIrStepPair(rawBd.bandName, stripAlpha(cf.meas), stripAlpha(cf.measPhase), rawBd.timeMs, rawBd.impulse, rawBd.step, 1.5, "measurement", false);
      } else {
        const cf = bandColorFamily(rawBd.bandColor);
        addIrStepPair("Measurement", stripAlpha(cf.meas), stripAlpha(cf.measPhase), rawBd.timeMs, rawBd.impulse, rawBd.step, 1.5, "measurement", false);
        lastIrMeasData.timeMs = [...rawBd.timeMs]; lastIrMeasData.impulse = [...rawBd.impulse]; lastIrMeasData.step = [...rawBd.step];
      }
    }
    // Measurement sum
    if (measSum) {
      addIrStepPair("\u03A3 meas", sumClr.measIr, sumClr.measStep, measSum.timeMs, measSum.impulse, measSum.step, 2, "measurement", false);
    }

    // --- Target per-band ---
    for (const rawBd of targetBands) {
      if (inSum) {
        const cf = bandColorFamily(rawBd.bandColor);
        addIrStepPair(rawBd.bandName + " tgt", cf.target, cf.targetPhase, rawBd.timeMs, rawBd.impulse, rawBd.step, 1.5, "target", true);
      } else {
        const cf = bandColorFamily(rawBd.bandColor);
        addIrStepPair("Target", cf.target, cf.targetPhase, rawBd.timeMs, rawBd.impulse, rawBd.step, 2, "target", true);
        lastIrTargetData.timeMs = [...rawBd.timeMs]; lastIrTargetData.impulse = [...rawBd.impulse]; lastIrTargetData.step = [...rawBd.step];
      }
    }
    if (!inSum && targetBands.length === 0) {
      lastIrTargetData.timeMs = []; lastIrTargetData.impulse = []; lastIrTargetData.step = [];
    }
    // Target sum
    if (targetSum) {
      addIrStepPair("\u03A3 target", sumClr.targetIr, sumClr.targetStep, targetSum.timeMs, targetSum.impulse, targetSum.step, 2, "target", true);
    }

    // --- Corrected per-band ---
    for (const rawBd of corrBands) {
      if (inSum) {
        const cf = bandColorFamily(rawBd.bandColor);
        const bdDelay = untrack(() => appState.bands.find(b => b.name === rawBd.bandName)?.alignmentDelay ?? 0);
        const dlSuffix = Math.abs(bdDelay) > 1e-6
          ? ` [${bdDelay >= 0 ? "+" : ""}${(bdDelay * 1000).toFixed(2)} ms]`
          : "";
        addIrStepPair(rawBd.bandName + " corr+XO" + dlSuffix, cf.corrected, cf.correctedPhase, rawBd.timeMs, rawBd.impulse, rawBd.step, 1.5, "corrected", false);
      } else {
        const cf = bandColorFamily(rawBd.bandColor);
        addIrStepPair("Corrected", cf.corrected, cf.correctedPhase, rawBd.timeMs, rawBd.impulse, rawBd.step, 2, "corrected", false);
        lastIrCorrData.timeMs = [...rawBd.timeMs]; lastIrCorrData.impulse = [...rawBd.impulse]; lastIrCorrData.step = [...rawBd.step];
      }
    }
    if (!inSum && corrBands.length === 0) {
      lastIrCorrData.timeMs = []; lastIrCorrData.impulse = []; lastIrCorrData.step = [];
    }
    // Corrected sum
    if (corrSum) {
      addIrStepPair("\u03A3 corrected", sumClr.corrIr, sumClr.corrStep, corrSum.timeMs, corrSum.impulse, corrSum.step, 2, "corrected", false);
    }

    // IR/Step snapshots — data stored in LINEAR units, apply dB conversion via same pipeline as live data
    if (!inSum) {
      const band = activeBand();
      const snaps = band ? plotSnapshots(band.id, "ir") : [];
      for (const snap of snaps) {
        if (!snap.timeMs) continue;
        const hasIr = snap.impulse && snap.impulse.length > 0;
        const hasSt = snap.step && snap.step.length > 0;
        if (!hasIr && !hasSt) continue;
        const sameSnGrid = snap.timeMs.length === timeMs.length
          && Math.abs((snap.timeMs[0] ?? 0) - (timeMs[0] ?? 0)) < 0.001;
        if (hasIr) {
          // Use same dB conversion pipeline as live data (resampleOnto applies toDb, applyDb for same grid)
          const irData = sameSnGrid ? applyDb(snap.impulse!) : resampleOnto(snap.timeMs, snap.impulse!);
          uSeries.push({ label: snap.label + " IR", stroke: snap.color, width: 1, scale: "y", show: true, dash: [4, 3] });
          uDataArr.push(irData);
          irLegend.push({ label: snap.label + " IR", color: snap.color, dash: true, visible: true, seriesIdx: sIdx, category: "snapshot" });
          sIdx++;
        }
        if (hasSt) {
          const stData = sameSnGrid ? applyDb(snap.step!) : resampleOnto(snap.timeMs, snap.step!);
          uSeries.push({ label: snap.label + " Step", stroke: snap.color + "80", width: 1, scale: "y", show: true, dash: [4, 3] });
          uDataArr.push(stData);
          irLegend.push({ label: snap.label + " Step", color: snap.color + "80", dash: true, visible: true, seriesIdx: sIdx, category: "snapshot" });
          sIdx++;
        }
      }
    }

    // Restore visibility persistence BEFORE Y-range so hidden series are excluded from auto-fit
    if (inSum) {
      for (const e of irLegend) {
        const saved = sumVisMap.get(e.label);
        if (saved !== undefined) e.visible = saved;
      }
    } else {
      for (const e of irLegend) {
        const saved = bandVisMap.get(catKey(e));
        if (saved !== undefined) e.visible = saved;
      }
    }
    for (const e of irLegend) {
      if (!e.visible && e.seriesIdx < uSeries.length) {
        (uSeries[e.seriesIdx] as any).show = false;
      }
    }

    // Y range — only from visible series and visible X window (±30ms around peak)
    let yMin = Infinity, yMax = -Infinity;
    const xData = uDataArr[0];
    for (let s = 1; s < uDataArr.length; s++) {
      // Skip hidden series
      if (s < uSeries.length && (uSeries[s] as any).show === false) continue;
      const arr = uDataArr[s];
      for (let j = 0; j < arr.length; j++) {
        // Only consider data within visible X window
        if (xData[j] < xViewMin || xData[j] > xViewMax) continue;
        const v = arr[j];
        if (v > -190 && v < yMin) yMin = v;
        if (v > -190 && v > yMax) yMax = v;
      }
    }
    if (!isFinite(yMin)) { yMin = isDb ? -80 : -1.1; yMax = isDb ? 0 : 1.1; }
    const pad = isDb ? 5 : Math.max(0.05, (yMax - yMin) * 0.05);
    // Set mutable Y range (will be used by range function and fitData)
    const fitYMin = isDb ? Math.max(yMin - pad, -80) : yMin - pad;
    const fitYMax = yMax + pad;
    // If saved scale from irSaveScales, use it; otherwise auto-fit
    if (irUserYScale) {
      irCurYMin = irUserYScale.min;
      irCurYMax = irUserYScale.max;
    } else {
      irCurYMin = fitYMin;
      irCurYMax = fitYMax;
    }

    if (!inSum) setShowLegend(false);
    setCursorValues([]);

    // Update irColors signal for band mode toolbar
    if (!inSum && measBands.length > 0) {
      const cf = bandColorFamily(measBands[0].bandColor);
      setIrColors({
        measIr: stripAlpha(cf.meas), measStep: stripAlpha(cf.measPhase),
        targetIr: cf.target, targetStep: cf.targetPhase,
        corrIr: cf.corrected, corrStep: cf.correctedPhase,
      });
    } else if (inSum) {
      setIrColors(sumClr);
    }

    const opts: uPlot.Options = {
      width: w, height: h,
      series: uSeries,
      scales: {
        x: { auto: false, range: () => [irCurXMin, irCurXMax] as uPlot.Range.MinMax },
        y: { auto: false, range: () => [irCurYMin, irCurYMax] as uPlot.Range.MinMax },
      },
      axes: [
        { label: "ms", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)) },
        { label: isDb ? "dBr" : "%", scale: "y", stroke: "#9b9ba6", grid: { stroke: "rgba(255,255,255,0.12)" }, ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : isDb ? v.toFixed(0) : v.toFixed(0) + "%"), size: 50 },
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
              ctx.fillStyle = "rgba(234, 179, 8, 0.08)";
              ctx.beginPath();
              for (let s = 0; s <= nSteps; s++) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(5.0 * Math.exp(-3 * t), "y", true)); }
              for (let s = nSteps; s >= 0; s--) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(-5.0 * Math.exp(-3 * t), "y", true)); }
              ctx.closePath(); ctx.fill();
              ctx.fillStyle = "rgba(34, 197, 94, 0.10)";
              ctx.beginPath();
              for (let s = 0; s <= nSteps; s++) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(1.0 * Math.exp(-4 * t), "y", true)); }
              for (let s = nSteps; s >= 0; s--) { const t = s / nSteps; ctx.lineTo(peakX - t * maskWidthX, u.valToPos(-1.0 * Math.exp(-4 * t), "y", true)); }
              ctx.closePath(); ctx.fill();
            }
          }
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
          // Show first visible IR/Step values
          const vals: string[] = [];
          for (let si = 1; si < u.series.length; si++) {
            if (!u.series[si].show) continue;
            const v = u.data[si]?.[idx];
            if (v == null || (v as number) <= -190) continue;
            const lbl = u.series[si].label ?? "";
            if (isDb) vals.push(`${lbl}: ${(v as number).toFixed(0)} dB`);
            else vals.push(`${lbl}: ${(v as number).toFixed(1)}%`);
            if (vals.length >= 4) break; // limit readout clutter
          }
          setCursorSPL(vals.length > 0 ? vals.join("  ") : "—");
        }],
      },
    };
    try {
      chart = new uPlot(opts, uDataArr as uPlot.AlignedData, containerRef);
      // Restore user zoom if saved (irCurYMin/Max already set from data or saved scale)
      if (irUserXScale) {
        irRestoreScales();
      }
      setLegendEntries(irLegend);
      setShowLegend(true);
    } catch (e) { console.error("IR chart error:", e); }
  }

  function renderGdChart(
    freq: number[], measGdMs: number[],
    targetGdMs: number[] | null, corrGdMs: number[] | null,
    clr: { meas: string; target: string; corr: string } = defaultGdColors,
    cfg: { meas: boolean; target: boolean; corr: boolean } = { meas: true, target: true, corr: true },
  ) {
    try { if (chart) { chart.destroy(); chart = undefined; } } catch (_) { chart = undefined; }
    if (!containerRef) return;
    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 200);

    const emptyData = freq.map(() => NaN);
    const uSeries: uPlot.Series[] = [{}];
    const uDataArr: number[][] = [freq];

    // Series 1: Measurement GD
    uSeries.push({ label: "Meas GD", stroke: clr.meas, width: 1.5, scale: "y", show: cfg.meas });
    uDataArr.push(measGdMs);

    // Series 2: Target GD
    uSeries.push({ label: "Target GD", stroke: clr.target, width: 1.5, dash: [6, 3], scale: "y", show: cfg.target });
    uDataArr.push(targetGdMs ?? emptyData);

    // Series 3: Corrected GD
    uSeries.push({ label: "Corr GD", stroke: clr.corr, width: 1.5, scale: "y", show: cfg.corr });
    uDataArr.push(corrGdMs ?? emptyData);

    // GD snapshots
    {
      const band = activeBand();
      const snaps = band ? plotSnapshots(band.id, "gd") : [];
      for (const snap of snaps) {
        if (snap.freq && snap.gdMs) {
          // Resample snapshot onto current freq grid
          const snapData = freq.map(f => {
            if (f < snap.freq![0] || f > snap.freq![snap.freq!.length - 1]) return NaN;
            let lo = 0, hi = snap.freq!.length - 1;
            while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (snap.freq![mid] <= f) lo = mid; else hi = mid; }
            const dt = snap.freq![hi] - snap.freq![lo];
            const frac = dt > 0 ? (f - snap.freq![lo]) / dt : 0;
            return snap.gdMs![lo] + frac * (snap.gdMs![hi] - snap.gdMs![lo]);
          });
          uSeries.push({ label: snap.label, stroke: snap.color, width: 1, dash: [4, 3], scale: "y", show: true });
          uDataArr.push(snapData);
        }
      }
    }

    let yMin = Infinity, yMax = -Infinity;
    for (let s = 1; s < uDataArr.length; s++) {
      for (const v of uDataArr[s]) { if (isFinite(v)) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; } }
    }
    const pad = Math.max(0.5, (yMax - yMin) * 0.1);
    if (!isFinite(yMin)) { yMin = -5; yMax = 20; }
    // Restore persisted GD Y scale if available (from toggle redraw)
    if (gdUserYMin != null && gdUserYMax != null) {
      yMin = gdUserYMin; yMax = gdUserYMax;
      gdUserYMin = null; gdUserYMax = null;
    } else {
      yMin -= pad; yMax += pad;
    }

    if (!isSum()) setShowLegend(false);
    setCursorValues([]);

    const opts: uPlot.Options = {
      width: w, height: h,
      series: uSeries,
      scales: {
        x: { min: 20, max: 20000, distr: 3 },
        y: { auto: false, range: [yMin, yMax] as uPlot.Range.MinMax },
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
          setCursorFreq(f != null ? fmtFreq(f) : "—");
          const vals: { label: string; color: string; value: string }[] = [];
          const m = u.data[1]?.[idx]; if (m != null && isFinite(m as number)) vals.push({ label: "Meas", color: clr.meas, value: (m as number).toFixed(2) + " ms" });
          const t = u.data[2]?.[idx]; if (t != null && isFinite(t as number)) vals.push({ label: "Target", color: clr.target, value: (t as number).toFixed(2) + " ms" });
          const c = u.data[3]?.[idx]; if (c != null && isFinite(c as number)) vals.push({ label: "Corr", color: clr.corr, value: (c as number).toFixed(2) + " ms" });
          setCursorValues(vals);
          setCursorSPL(vals.length > 0 ? vals.map(v => v.value).join(" / ") : "—");
        }],
      },
    };
    try {
      cacheOrigStrokes(uSeries);
      chart = new uPlot(opts, uDataArr as uPlot.AlignedData, containerRef);
    } catch (e) { console.error(e); }
  }

  // ----------------------------------------------------------------
  // Single band rendering
  // ----------------------------------------------------------------
  async function renderBandMode(band: BandState, showPhase: boolean, showMag: boolean, showTarget: boolean) {
    const gen = ++renderGen;
    zoomCenter = 0; // reset before async — will be recalculated from measurement
    try {
      const result = await evaluateBand(band);
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
        legend.push({ label: "Target", color: cf.target, dash: false, visible: true, seriesIdx: sIdx, category: "target" });
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
      let peqDotsInfo: { seriesIdx: number; dataIndices: Set<number> } | undefined;
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

            // PEQ-only response curve (0 dB baseline) with dots at PEQ band frequencies
            if (showMag) {
              // Shift PEQ response to sit at the zoomCenter baseline (will be normalized later)
              const peqOnly = pm.map((v: number) => v + zoomCenter);
              // Collect dot indices for PEQ band markers
              const enabledBands = band.peqBands!.filter((b: PeqBand) => b.enabled);
              const dotIndices = new Set<number>();
              for (const pb of enabledBands) {
                let bestIdx = 0, bestDist = Infinity;
                for (let k = 0; k < result.freq!.length; k++) {
                  const d = Math.abs(result.freq![k] - pb.freq_hz);
                  if (d < bestDist) { bestDist = d; bestIdx = k; }
                }
                dotIndices.add(bestIdx);
              }
              const peqSIdx = sIdx;
              uSeries.push({
                label: "PEQ dB", stroke: PEQ_COLOR, width: 1.5, scale: "mag",
                points: { show: false },
              });
              uData.push(peqOnly);
              legend.push({ label: "PEQ", color: PEQ_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "peq" });
              sIdx++;
              peqDotsInfo = { seriesIdx: peqSIdx, dataIndices: dotIndices };
              activePeqDots = peqDotsInfo;
            }
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
            // Gaussian min-phase: add per-filter Hilbert-derived phase to corrected phase
            if (isGaussianMinPhase(band.target.high_pass) || isGaussianMinPhase(band.target.low_pass)) {
              fullCorrectedPhase = await addGaussianMinPhase(result.freq!, fullCorrectedPhase, band.target.high_pass, band.target.low_pass);
              if (gen !== renderGen) return;
            }
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

          // Save per-category data for snapshot capture (raw, pre-normalization)
          lastCorrData.freq = result.freq!;
          lastCorrData.mag = [...fullCorrected];
          lastCorrData.phase = fullCorrectedPhase ? wrapPhase(fullCorrectedPhase) : [];
        } catch (e) {
          console.warn("Correction computation failed:", e);
        }
      } else {
        // No corrected curve — fallback to target or measurement for snapshot
        if (result.targetMag) {
          lastCorrData.freq = result.freq!;
          lastCorrData.mag = [...result.targetMag];
          lastCorrData.phase = result.targetPhase ? wrapPhase(result.targetPhase) : [];
        } else if (result.measurement) {
          lastCorrData.freq = [...result.measurement.freq];
          lastCorrData.mag = [...result.measurement.magnitude];
          lastCorrData.phase = result.measurement.phase ? wrapPhase(result.measurement.phase) : [];
        } else {
          lastCorrData.freq = [];
          lastCorrData.mag = [];
          lastCorrData.phase = [];
        }
      }

      // Save measurement and target data for snapshot capture
      if (result.measurement) {
        lastMeasData.freq = [...result.measurement.freq];
        lastMeasData.mag = [...result.measurement.magnitude];
        lastMeasData.phase = result.measurement.phase ? wrapPhase(result.measurement.phase) : [];
      }
      if (result.targetMag) {
        lastTargetData.freq = result.freq!;
        lastTargetData.mag = [...result.targetMag];
        lastTargetData.phase = result.targetPhase ? wrapPhase(result.targetPhase) : [];
      } else {
        lastTargetData.freq = []; lastTargetData.mag = []; lastTargetData.phase = [];
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

        const snapMag = snap.mag.length > 0 ? interpolateArr(snap.mag as (number | null)[]) as number[] : null;
        const snapPhase = snap.phase.length > 0 ? interpolateArr(snap.phase) : null;

        // Magnitude series (only if captured)
        if (snapMag && snapMag.length > 0) {
          uSeries.push({
            label: `${snap.label} dB`,
            stroke: snap.color,
            width: 1.5,
            dash: [4, 3],
            scale: "mag",
          });
          uData.push(snapMag);
          legend.push({ label: snap.label, color: snap.color, dash: true, visible: true, seriesIdx: sIdx, category: "snapshot" });
          sIdx++;
        }

        // Phase series (only if captured)
        if (snapPhase && snapPhase.some(v => v != null)) {
          uSeries.push({
            label: `${snap.label} \u00B0`,
            stroke: snap.color,
            width: 1,
            dash: [2, 2],
            scale: "phase",
          });
          uData.push(snapPhase as number[]);
          legend.push({ label: `${snap.label} \u00B0`, color: snap.color, dash: true, visible: true, seriesIdx: sIdx, category: "snapshot" });
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
        if (gen !== renderGen || !containerRef) return;
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
      const results = await Promise.all(bands.map((b) => evaluateBand(b)));
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
        if (gen !== renderGen) return;
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

      // Collect interpolation tasks for parallel execution
      const interpTasks: { idx: number; m: any }[] = [];
      for (let ri = 0; ri < results.length; ri++) {
        const r = results[ri];
        if (!r.measurement) { resampled.push(null); continue; }
        const m = r.measurement;
        if (!needResample && m.freq.length === nPts && m.freq[0] === fMin && m.freq[m.freq.length - 1] === fMax) {
          // Уже на общей сетке
          resampled.push({ magnitude: m.magnitude, phase: m.phase ?? null, name: m.name });
        } else {
          interpTasks.push({ idx: resampled.length, m });
          resampled.push(null); // placeholder
        }
      }

      // Run all interpolate_log calls in parallel
      if (interpTasks.length > 0) {
        const interpResults = await Promise.all(
          interpTasks.map(t =>
            invoke<[number[], number[], number[] | null]>(
              "interpolate_log",
              { freq: t.m.freq, magnitude: t.m.magnitude, phase: t.m.phase, nPoints: nPts, fMin, fMax }
            ).catch(e => {
              console.warn("Interpolation failed for", t.m.name, e);
              return null;
            })
          )
        );
        if (gen !== renderGen) return;
        for (let i = 0; i < interpTasks.length; i++) {
          const t = interpTasks[i];
          const result = interpResults[i];
          if (result) {
            const [, intMag, intPhase] = result;
            resampled[t.idx] = { magnitude: intMag, phase: intPhase, name: t.m.name };
          } else {
            resampled[t.idx] = null;
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
        if (gen !== renderGen) return;
        perBandTargetMags.push(response.magnitude);

        // Normalized response = response with ref_level=0, which is just response.magnitude - totalRef
        // (evaluate_target is linear in reference_level_db, so no second invoke needed)
        const normMag = response.magnitude.map(v => v - totalRef);
        perBandTargetNorm.push(normMag);
        perBandTargetNormPhase.push(response.phase);
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
            // Run PEQ and XO computations in parallel (independent operations)
            const peqPromise = hasPeq
              ? invoke<[number[], number[]]>("compute_peq_complex", {
                  freq, bands: bands[i].peqBands,
                })
              : Promise.resolve(null);

            const xsPromise = (hasFilters && tMag)
              ? invoke<[number[], number[], number]>("compute_cross_section", {
                  freq,
                  highPass: bands[i].target.high_pass,
                  lowPass: bands[i].target.low_pass,
                })
              : Promise.resolve(null);

            const [peqResult, xsResult] = await Promise.all([peqPromise, xsPromise]);
            if (gen !== renderGen) return;

            let peqMag: number[] | null = null;
            let peqPhase: number[] | null = null;
            if (peqResult) {
              peqMag = peqResult[0];
              peqPhase = peqResult[1];
            }

            let xsMag: number[] | null = null;
            let xsPhase: number[] | null = null;
            if (xsResult) {
              xsMag = xsResult[0];
              xsPhase = xsResult[1];
            }

            const corrected = rm.magnitude.map(
              (v: number, j: number) => v + (peqMag ? peqMag[j] : 0) + (xsMag ? xsMag[j] : 0)
            );
            // Per-band normalization: align corrected to target in passband
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
            // unwrapDegrees ensures wrapped PEQ/XO phase adds correctly to unwrapped meas phase
            if (rm.phase) {
              let corrPhase = unwrapDegrees(rm.phase.map(
                (v: number, j: number) => v + (peqPhase ? peqPhase[j] : 0) + (xsPhase ? xsPhase[j] : 0)
              ));
              // Gaussian min-phase: add per-filter Hilbert-derived phase to corrected phase
              if (isGaussianMinPhase(bands[i].target.high_pass) || isGaussianMinPhase(bands[i].target.low_pass)) {
                corrPhase = await addGaussianMinPhase(freq, corrPhase, bands[i].target.high_pass, bands[i].target.low_pass);
                if (gen !== renderGen) return;
              }
              perBandCorrPhase.push(corrPhase);
            } else {
              perBandCorrPhase.push(null);
            }

            const color = bandColorFamily(bands[i].color).corrected;
            const alignDelayCurr = bands[i].alignmentDelay ?? 0;
            const delaySuffix = Math.abs(alignDelayCurr) > 1e-6
              ? ` [${alignDelayCurr >= 0 ? "+" : ""}${(alignDelayCurr * 1000).toFixed(2)} ms]`
              : "";
            uSeries.push({ label: bands[i].name + " corr+XO", stroke: color, width: 2, scale: "mag" });
            uData.push(corrected);
            legend.push({ label: bands[i].name + " corr+XO" + delaySuffix, color, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;

            if (showPhase && perBandCorrPhase[i]) {
              const phColor = bandColorFamily(bands[i].color).corrected + "60";
              uSeries.push({ label: bands[i].name + " corr°", stroke: phColor, width: 1, dash: [4, 3], scale: "phase" });
              uData.push(wrapPhase(perBandCorrPhase[i]!));
              legend.push({ label: bands[i].name + " corr°", color: phColor, dash: true, visible: false, seriesIdx: sIdx, category: "corrected" });
              sIdx++;
            }
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
          const enabledBandIdx: number[] = [];
          for (let i = 0; i < bands.length; i++) {
            if (perBandTargetNorm[i]) {
              enabledNorm.push(perBandTargetNorm[i]!);
              enabledPhase.push(perBandTargetNormPhase[i] ?? Array.from({ length: freq.length }, () => 0));
              enabledInverted.push(bands[i].inverted);
              enabledBandIdx.push(i);
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
              const alignDelay = bands[enabledBandIdx[n]].alignmentDelay ?? 0;
              for (let j = 0; j < freq.length; j++) {
                const amp = Math.pow(10, mag[j] / 20) * sign;
                const phRad = (ph[j] + 360 * freq[j] * alignDelay) * Math.PI / 180;
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
                legend.push({ label: "\u03A3 target \u00B0", color: SUM_TARGET_PHASE_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
                sIdx++;
              }
            }
          }
        }

        // Σ corrected (когерентное сложение с учётом полной фазы: замер + PEQ + фильтры)
        const corrIndices = perBandCorrected.map((c, i) => c ? i : -1).filter(i => i >= 0);
        if (corrIndices.length > 0) {
          const hasAllPhaseCorr = corrIndices.every(
            (ci) => perBandCorrPhase[ci] && perBandCorrPhase[ci]!.length === freq.length
          );
          if (hasAllPhaseCorr) {
            // Coherent sum of all corrected bands
            const sumRe = new Float64Array(freq.length);
            const sumIm = new Float64Array(freq.length);
            for (const ci of corrIndices) {
              const corr = perBandCorrected[ci]!;
              const corrPh = perBandCorrPhase[ci]!;
              const sign = bands[ci].inverted ? -1 : 1;
              const alignDelay = bands[ci].alignmentDelay ?? 0;
              for (let j = 0; j < freq.length; j++) {
                const amp = Math.pow(10, corr[j] / 20) * sign;
                const phRad = (corrPh[j] + 360 * freq[j] * alignDelay) * Math.PI / 180;
                sumRe[j] += amp * Math.cos(phRad);
                sumIm[j] += amp * Math.sin(phRad);
              }
            }
            const sumCorrDb = new Array(freq.length);
            const sumCorrPhase = new Array(freq.length);
            for (let j = 0; j < freq.length; j++) {
              const amplitude = Math.sqrt(sumRe[j] * sumRe[j] + sumIm[j] * sumIm[j]);
              sumCorrDb[j] = amplitude > 0 ? 20 * Math.log10(amplitude) : -200;
              sumCorrPhase[j] = Math.atan2(sumIm[j], sumRe[j]) * 180 / Math.PI;
            }
            // No Σ offset — per-band normalization already aligns each band to its target.
            // Σ corrected is the pure coherent sum, showing real acoustic behavior (b101).
            uSeries.push({ label: "\u03A3 corr", stroke: SUM_CORRECTED_COLOR, width: 3, scale: "mag" });
            uData.push(sumCorrDb);
            legend.push({ label: "\u03A3 corrected", color: SUM_CORRECTED_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
            // Σ corrected phase
            if (showPhase) {
              uSeries.push({ label: "\u03A3 corr °", stroke: SUM_CORRECTED_COLOR, width: 1.5, dash: [6, 3], scale: "phase" });
              const sumCorrPhaseDisplay = new Array(freq.length);
              for (let j = 0; j < freq.length; j++) {
                let weightedDelay = 0;
                let totalAmplitude = 0;
                for (let i = 0; i < bands.length; i++) {
                  if (!perBandCorrected[i]) continue;
                  const amplitude = Math.pow(10, perBandCorrected[i]![j] / 20);
                  const alignDelay = bands[i].alignmentDelay ?? 0;
                  weightedDelay += amplitude * alignDelay;
                  totalAmplitude += amplitude;
                }
                const avgDelay = totalAmplitude > 1e-30 ? weightedDelay / totalAmplitude : 0;
                sumCorrPhaseDisplay[j] = sumCorrPhase[j] - 360 * freq[j] * avgDelay;
              }
              uData.push(wrapPhase(sumCorrPhaseDisplay));
              legend.push({ label: "\u03A3 corr °", color: SUM_CORRECTED_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "corrected" });
              sIdx++;
            }
          } else {
            // Инкогерентное сложение (power sum, без фазы)
            // Sum in power domain: Σ 10^(dB/10), then 10·log10(Σ)
            const sumPower = new Float64Array(nPts);
            for (const ci of corrIndices) {
              const corr = perBandCorrected[ci]!;
              for (let j = 0; j < nPts; j++) {
                sumPower[j] += Math.pow(10, corr[j] / 10);
              }
            }
            const sumCorrDb = Array.from(sumPower, (v: number) =>
              v > 1e-30 ? 10 * Math.log10(v) : -200
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
          (mi) => resampled[mi]!.phase && resampled[mi]!.phase!.length === freq.length
        );
        if (hasAllPhase) {
          const sumRe = new Float64Array(freq.length);
          const sumIm = new Float64Array(freq.length);
          for (const mi of measIndices) {
            const rm = resampled[mi]!;
            const sign = bands[mi].inverted ? -1 : 1;
            const alignDelay = bands[mi].alignmentDelay ?? 0;
            for (let j = 0; j < freq.length; j++) {
              const amp = Math.pow(10, rm.magnitude[j] / 20) * sign;
              const phRad = (rm.phase![j] + 360 * freq[j] * alignDelay) * Math.PI / 180;
              sumRe[j] += amp * Math.cos(phRad);
              sumIm[j] += amp * Math.sin(phRad);
            }
          }
          const sumMagDb = new Array(freq.length);
          const sumPhase = new Array(freq.length);
          for (let j = 0; j < freq.length; j++) {
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
            const sumPhaseDisplay = new Array(nPts);
            for (let j = 0; j < nPts; j++) {
              let weightedDelay = 0;
              let totalAmplitude = 0;
              for (const mi of measIndices) {
                const rm = resampled[mi]!;
                const amplitude = Math.pow(10, rm.magnitude[j] / 20);
                const alignDelay = bands[mi].alignmentDelay ?? 0;
                weightedDelay += amplitude * alignDelay;
                totalAmplitude += amplitude;
              }
              const avgDelay = totalAmplitude > 1e-30 ? weightedDelay / totalAmplitude : 0;
              sumPhaseDisplay[j] = sumPhase[j] - 360 * freq[j] * avgDelay;
            }
            uData.push(wrapPhase(sumPhaseDisplay));
            legend.push({ label: "\u03A3 meas °", color: SUM_MEAS_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "measurement" });
            sIdx++;
          }
        } else {
          // Incoherent sum: power domain (polarity irrelevant without phase)
          const sumPower = new Float64Array(nPts);
          for (const mi of measIndices) {
            const rm = resampled[mi]!;
            for (let j = 0; j < nPts; j++) {
              sumPower[j] += Math.pow(10, rm.magnitude[j] / 10);
            }
          }
          const sumMagDb = Array.from(sumPower, (v: number) =>
            v > 1e-30 ? 10 * Math.log10(v) : -200
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
      requestAnimationFrame(() => {
        if (gen !== renderGen) return;
        renderChart({
          freq,
          uSeries,
          uData,
          hasMeasurements: measIndices.length > 0,
          legend,
          crossovers,
        });
      });
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

  // ----------------------------------------------------------------
  // PEQ drag interaction: click to select, drag to move freq/gain
  // ----------------------------------------------------------------
  let peqDragIdx = -1; // index of PEQ band being dragged
  let peqDragBandId = "";
  let peqDragActive = false;
  let peqDragMoved = false; // true if mouse actually moved during drag
  let peqDragTimeout: ReturnType<typeof setTimeout> | null = null;

  function handlePeqMouseDown(e: MouseEvent) {
    if (isSum() || !chart) return;
    const bd = activeBand();
    if (!bd?.peqBands?.length) return;

    const overEl = chart.over ?? containerRef;
    const rect = overEl?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Find closest PEQ dot within hit radius
    const hitRadius = 12; // px
    let bestDist = Infinity;
    let bestIdx = -1;

    for (let i = 0; i < bd.peqBands.length; i++) {
      const pb = bd.peqBands[i];
      if (!pb.enabled) continue;
      const cx = chart.valToPos(pb.freq_hz, "x");
      // PEQ gain on the PEQ-only curve = gain_db (since PEQ curve is 0 dB baseline + peqMag)
      // But on chart the PEQ value is peqMag[at freq] which includes all bands
      // Simpler: use the PEQ series data if available
      const peqSeries = chart.series.findIndex(s => s.label === "PEQ dB");
      if (peqSeries < 0) continue;
      // Find data index closest to pb.freq_hz
      const xData = chart.data[0];
      let dataIdx = 0, dBest = Infinity;
      for (let k = 0; k < xData.length; k++) {
        const dd = Math.abs(xData[k] - pb.freq_hz);
        if (dd < dBest) { dBest = dd; dataIdx = k; }
      }
      const yVal = chart.data[peqSeries][dataIdx];
      if (yVal == null) continue;
      const cy = chart.valToPos(yVal, "mag");
      const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
      if (dist < hitRadius && dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0) {
      e.preventDefault();
      e.stopImmediatePropagation();
      peqDragIdx = bestIdx;
      peqDragBandId = bd.id;
      peqDragActive = true;
      peqDragMoved = false;
      setSelectedPeqIdx(bestIdx);
      window.addEventListener("mousemove", handlePeqDragMove);
      window.addEventListener("mouseup", handlePeqDragUp);
    }
  }

  function handlePeqDragMove(e: MouseEvent) {
    if (!peqDragActive || !chart || peqDragIdx < 0) return;
    if (!peqDragMoved) { peqDragMoved = true; setPeqDragging(true); }
    const overEl = chart.over ?? containerRef;
    const rect = overEl?.getBoundingClientRect();
    if (!rect) return;

    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let freq = chart.posToVal(mx, "x");
    if (!isFinite(freq)) return;
    freq = Math.max(20, Math.min(20000, freq));

    let gain = chart.posToVal(my, "mag");
    if (!isFinite(gain)) return;
    // PEQ curve is 0 dB baseline, so the Y value IS the gain
    gain = Math.max(-24, Math.min(24, gain));

    updatePeqBand(peqDragBandId, peqDragIdx, {
      freq_hz: Math.round(freq),
      gain_db: Math.round(gain * 10) / 10,
    });
  }

  function handlePeqDragUp() {
    window.removeEventListener("mousemove", handlePeqDragMove);
    window.removeEventListener("mouseup", handlePeqDragUp);
    if (peqDragMoved && peqDragActive && peqDragIdx >= 0) {
      // Save scales before commit triggers rebuild
      if (chart) {
        const ms = chart.scales["mag"];
        const xs = chart.scales["x"];
        if (ms?.min != null && ms?.max != null) { persistedMagMin = ms.min; persistedMagMax = ms.max; }
        if (xs?.min != null && xs?.max != null) { persistedXMin = xs.min; persistedXMax = xs.max; }
      }
      const newIdx = commitPeqBand(peqDragBandId, peqDragIdx);
      setSelectedPeqIdx(newIdx);
    }
    if (peqDragMoved) {
      if (peqDragTimeout) clearTimeout(peqDragTimeout);
      peqDragTimeout = setTimeout(() => setPeqDragging(false), 150);
    }
    // Click without move — no commit, no rebuild, just selection (already set in mousedown)
    peqDragActive = false;
    peqDragIdx = -1;
    peqDragMoved = false;
  }

  // Scroll wheel on chart → change Q of selected PEQ band
  let peqWheelTimeout: ReturnType<typeof setTimeout> | null = null;
  function handlePeqWheel(e: WheelEvent) {
    if (isSum() || !chart) return;
    const bd = activeBand();
    const selIdx = selectedPeqIdx();
    if (!bd?.peqBands?.length || selIdx == null || selIdx < 0 || selIdx >= bd.peqBands.length) return;
    const pb = bd.peqBands[selIdx];
    if (!pb.enabled) return;

    // First scroll: check proximity to selected dot. During active scroll: skip check.
    if (!peqDragging()) {
      const overEl = chart.over ?? containerRef;
      const rect = overEl?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const cx = chart.valToPos(pb.freq_hz, "x");
      // Only check X proximity (within 50px horizontal)
      if (Math.abs(mx - cx) > 50) return;
    }

    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    const newQ = Math.max(0.1, Math.min(30, pb.q + pb.q * delta));

    if (!peqDragging()) setPeqDragging(true);
    updatePeqBand(bd.id, selIdx!, { q: Math.round(newQ * 100) / 100 });
    peqFastUpdate(bd);
    if (peqWheelTimeout) clearTimeout(peqWheelTimeout);
    peqWheelTimeout = setTimeout(() => setPeqDragging(false), 400);
  }

  onCleanup(() => {
    window.removeEventListener("mousemove", handleZoomBoxMove);
    window.removeEventListener("mouseup", handleZoomBoxUp);
    window.removeEventListener("mousemove", handlePeqDragMove);
    window.removeEventListener("mouseup", handlePeqDragUp);
    try { if (chart) chart.destroy(); } catch (_) {}
    chart = undefined;
  });

  return (
    <div class="plot-wrapper">
      <div class="plot-tabs-strip">
        <button class={`plot-tab ${plotTab() === "freq" ? "active" : ""}`} onClick={() => setPlotTab("freq")}>SPL</button>
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
          <button class="tb-btn" onClick={takeSnapshot} title="Snapshot current curve for comparison">SNAP</button>
          {(() => {
            const b = activeBand();
            const tab = plotTab();
            const freqSnaps = b ? freqSnapshots(b.id) : [];
            const otherSnaps = b ? plotSnapshots(b.id, tab === "step" ? "ir" : tab) : [];
            const snaps = tab === "freq" ? freqSnaps : otherSnaps;
            return (
              <>
                {snaps.length > 0 && (
                  <button class="tb-btn" onClick={clearSnapshots} title="Clear all snapshots">CLR</button>
                )}
                {snaps.length > 0 && (
                  <span style={{ "font-size": "var(--fs-xs)", "color": "#8b8b96", "margin-left": "var(--space-xs)" }}>
                    {snaps.map(s => (
                      <span style={{ color: s.color, "margin-right": "var(--space-sm)" }}>{"\u2588"} {s.label}</span>
                    ))}
                  </span>
                )}
              </>
            );
          })()}
        </Show>
        {/* IR/Step dB/Lin toggle */}
        <Show when={plotTab() === "ir" || plotTab() === "step"}>
          <span class="readout-sep" />
          <button class={`tb-btn tb-btn-xs ${irDbMode() ? "active" : ""}`} onClick={() => { setIrDbMode(!irDbMode()); irToggleRedraw(); }}>{irDbMode() ? "dB" : "Lin"}</button>
        </Show>

        {/* Compact controls for active band (b126) — replaces bottom panel */}
        <Show when={!isSum() && activeBand()}>
          {(() => {
            const b = () => activeBand()!;
            const s = () => b().settings;
            const m = () => b().measurement;

            async function handleImport() {
              try {
                const selected = await open({
                  multiple: false,
                  filters: [{ name: "Measurement Files", extensions: ["txt", "frd"] }],
                });
                if (!selected) return;
                const filePath = Array.isArray(selected) ? selected[0] : selected;
                const measurement = await invoke<Measurement>("import_measurement", { path: filePath });
                setBandMeasurement(b().id, measurement);
                const bandNum = b().name.match(/\d+/)?.[0] ?? "1";
                const newName = `Band ${bandNum} · ${measurement.name}`;
                renameBand(b().id, newName);
                try {
                  const fileName = await copyMeasurementToProject(filePath, newName);
                  if (fileName) setBandMeasurementFile(b().id, fileName);
                } catch (e) { console.warn("Failed to copy measurement:", e); }
                setNeedAutoFit(true);
                if (measurement.phase) {
                  try {
                    const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
                      "remove_measurement_delay",
                      { freq: measurement.freq, magnitude: measurement.magnitude, phase: measurement.phase, sampleRate: measurement.sample_rate }
                    );
                    setBandDelayInfo(b().id, delay, distance);
                    markBandDelayRemoved(b().id, newPhase);
                  } catch (e) { console.error("Delay removal failed:", e); }
                }
              } catch (e) { console.error("Import failed:", e); }
            }

            async function handleToggleDelay() {
              if (!m()?.phase || !s()) return;
              if (s()?.delay_removed) {
                restoreBandDelay(b().id);
              } else {
                try {
                  const origPhase = s()?.originalPhase ?? m()?.phase!;
                  const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
                    "remove_measurement_delay",
                    { freq: m()!.freq, magnitude: m()!.magnitude, phase: origPhase, sampleRate: m()!.sample_rate }
                  );
                  setBandDelayInfo(b().id, delay, distance);
                  markBandDelayRemoved(b().id, newPhase);
                } catch (e) { console.error("Remove delay failed:", e); }
              }
            }

            return (
              <>
                {/* Import + Merge — SPL tab only */}
                <Show when={plotTab() === "freq"}>
                  <span class="readout-sep" />
                  <button class="tb-btn tb-btn-xs primary" onClick={handleImport} title="Import measurement file">Import</button>
                  <button class="tb-btn tb-btn-xs" onClick={() => setShowMergeDialog(true)} title="Merge NF+FF">Merge</button>
                </Show>

                {/* Smooth — all tabs except Export */}
                <Show when={plotTab() !== "export"}>
                  <span class="readout-sep" />
                  <span class="readout-label">Smooth</span>
                  <select
                    class="tb-select tb-select-xs"
                    value={s()?.smoothing ?? "off"}
                    onChange={(e) => setBandSmoothing(b().id, e.currentTarget.value as SmoothingMode)}
                    title="Smoothing"
                  >
                    <option value="off">Off</option>
                    <option value="1/3">1/3</option>
                    <option value="1/6">1/6</option>
                    <option value="1/12">1/12</option>
                    <option value="1/24">1/24</option>
                    <option value="var">Var</option>
                  </select>
                </Show>

                {/* Delay toggle — all tabs except Export, only if phase available */}
                <Show when={plotTab() !== "export" && m()?.phase}>
                  <span class="readout-sep" />
                  <span class="readout-label">Delay</span>
                  <label class="tb-inline-label" title="Compensate propagation delay">
                    <input
                      type="checkbox"
                      checked={s()?.delay_removed ?? false}
                      onChange={() => handleToggleDelay()}
                    />
                    <span>D:{s()?.delay_removed ? (Math.abs(s()!.delay_seconds ?? 0) * 1000).toFixed(2) : ((s()?.delay_seconds ?? 0) * 1000).toFixed(2)}ms</span>
                  </label>
                </Show>

              </>
            );
          })()}
        </Show>

      </div>

      {/* Merge dialog — wired to toolbar button (b126) */}
      <Show when={showMergeDialog()}>
        <MergeDialog
          onClose={() => setShowMergeDialog(false)}
          onMerge={(measurement, source) => {
            const b = activeBand();
            if (b) {
              setBandMeasurement(b.id, measurement);
              setBandMergeSource(b.id, source);
              const bandNum = b.name.match(/\d+/)?.[0] ?? "1";
              const sourceLabel = source === "nf" ? "NF" : "FF";
              renameBand(b.id, `Band ${bandNum} · ${measurement.name} ${sourceLabel}`);
              setNeedAutoFit(true);
            }
            setShowMergeDialog(false);
          }}
        />
      </Show>

      {/* Unified visibility matrix — above plot, all modes */}
      {/* SUM matrix is shared across all tabs — shown below via legendEntries */}
      <Show when={(plotTab() === "ir" || plotTab() === "step") && !isSum() && legendEntries.length > 0}>
        {/* Band IR/Step matrix — uses legendEntries from renderIrStepChart */}
        <div class="sum-vis-table">
          {(() => {
            const categories: ("measurement" | "target" | "corrected")[] = ["measurement", "target", "corrected"];
            const catLabels: Record<string, string> = { measurement: "MEAS", target: "TARGET", corrected: "CORR" };
            return (
              <table>
                <thead><tr>
                  <th class="sum-corner" />
                  <th style={{ color: irColors().measIr, cursor: "pointer" }} onClick={() => toggleAllIrOrStep("IR")}>IR</th>
                  <th style={{ color: irColors().measStep, cursor: "pointer" }} onClick={() => toggleAllIrOrStep("Step")}>Step</th>
                </tr></thead>
                <tbody>
                  <For each={categories}>
                    {(cat) => {
                      const irE = () => legendEntries.find(e => e.category === cat && e.label.endsWith(" IR"));
                      const stE = () => legendEntries.find(e => e.category === cat && e.label.endsWith(" Step"));
                      return (
                        <Show when={irE() || stE()}>
                          <tr>
                            <td class="sum-row-header" onClick={() => toggleCategory(cat)} style={{ opacity: (irE()?.visible || stE()?.visible) ? 1 : 0.4 }}>{catLabels[cat]}</td>
                            <td class="sum-cell">
                              <Show when={irE()}>{(e) => {
                                const idx = () => legendEntries.findIndex(le => le.seriesIdx === e().seriesIdx);
                                return (
                                  <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}>
                                    <span class={`legend-swatch ${e().dash ? "legend-swatch-dash" : ""}`} style={{ "background-color": e().dash ? "transparent" : e().color, "border-color": e().color }} />
                                  </button>
                                );
                              }}</Show>
                            </td>
                            <td class="sum-cell">
                              <Show when={stE()}>{(e) => {
                                const idx = () => legendEntries.findIndex(le => le.seriesIdx === e().seriesIdx);
                                return (
                                  <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}>
                                    <span class={`legend-swatch ${e().dash ? "legend-swatch-dash" : ""}`} style={{ "background-color": e().dash ? "transparent" : e().color, "border-color": e().color }} />
                                  </button>
                                );
                              }}</Show>
                            </td>
                          </tr>
                        </Show>
                      );
                    }}
                  </For>
                  <tr>
                    <td class="sum-row-header" onClick={() => { setIrShowMasking(!irShowMasking()); irToggleRedraw(); }}>ZONES</td>
                    <td class="sum-cell" colspan="2">
                      <button class={`legend-item ${irShowMasking() ? "" : "legend-off"}`} onClick={() => { setIrShowMasking(!irShowMasking()); irToggleRedraw(); }}>
                        <span class="legend-swatch" style={{ "background-color": irShowMasking() ? "rgba(34,197,94,0.5)" : "transparent", "border-color": "rgba(34,197,94,0.5)" }} />
                        <span class="legend-text" style={{ "font-size": "var(--fs-xs)" }}>Pre-ringing</span>
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            );
          })()}
        </div>
      </Show>
      <Show when={plotTab() === "gd"}>
        {/* GD legend — meas/target/corr */}
        <div class="sum-vis-table">
          <table>
            <thead><tr>
              <th class="sum-corner" />
              <th style={{ color: gdColors().meas }}>ms</th>
            </tr></thead>
            <tbody>
              {[
                { label: "MEAS", sig: showGdMeas, set: setShowGdMeas, color: () => gdColors().meas },
                { label: "TARGET", sig: showGdTarget, set: setShowGdTarget, color: () => gdColors().target },
                { label: "CORR", sig: showGdCorr, set: setShowGdCorr, color: () => gdColors().corr },
              ].map(row => (
                <tr>
                  <td class="sum-row-header" onClick={() => { row.set(!row.sig()); gdToggleRedraw(); }}>{row.label}</td>
                  <td class="sum-cell">
                    <button class={`legend-item ${row.sig() ? "" : "legend-off"}`} onClick={() => { row.set(!row.sig()); gdToggleRedraw(); }}>
                      <span class="legend-swatch" style={{ "background-color": row.sig() ? row.color() : "transparent", "border-color": row.color() }} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Show>
      <Show when={plotTab() === "export"}>
        {/* Export controls — SR / Taps / Window (always visible on Export tab) */}
        <div class="export-controls-row" style={{
          display: "flex", "align-items": "center", gap: "var(--space-md)",
          padding: "var(--space-xs) var(--space-lg)",
          "border-bottom": "1px solid var(--border)",
          "font-family": "var(--mono)", "font-size": "var(--fs-base)",
        }}>
          <span class="readout-label">SR</span>
          <select
            class="tb-select tb-select-xs"
            value={exportSampleRate()}
            onChange={(e) => setExportSampleRate(Number(e.currentTarget.value))}
            title="Sample rate"
          >
            {[44100, 48000, 88200, 96000, 176400, 192000].map((sr) => (
              <option value={sr}>{sr >= 1000 ? (sr / 1000) + "k" : sr}</option>
            ))}
          </select>
          <span class="readout-label">Taps</span>
          <select
            class="tb-select tb-select-xs"
            value={exportTaps()}
            onChange={(e) => setExportTaps(Number(e.currentTarget.value))}
            title="FIR filter length"
          >
            {[4096, 8192, 16384, 32768, 65536, 131072, 262144].map((t) => (
              <option value={t}>{t >= 1024 ? (t / 1024) + "K" : t}</option>
            ))}
          </select>
          <span class="readout-label">Win</span>
          <select
            class="tb-select tb-select-xs"
            value={exportWindow()}
            onChange={(e) => setExportWindow(e.currentTarget.value as WindowType)}
            title="Window function"
          >
            <optgroup label="Basic">
              <option value="Rectangular">Rectangular</option>
              <option value="Bartlett">Bartlett</option>
              <option value="Hann">Hann</option>
              <option value="Hamming">Hamming</option>
              <option value="Blackman">Blackman</option>
            </optgroup>
            <optgroup label="Blackman-Harris">
              <option value="ExactBlackman">Exact Blackman</option>
              <option value="BlackmanHarris">Blackman-Harris</option>
              <option value="Nuttall3">Nuttall 3-term</option>
              <option value="Nuttall4">Nuttall 4-term</option>
              <option value="FlatTop">Flat Top</option>
            </optgroup>
            <optgroup label="Parametric">
              <option value="Kaiser">Kaiser (β=10)</option>
              <option value="DolphChebyshev">Dolph-Chebyshev</option>
              <option value="Gaussian">Gaussian (σ=2.5)</option>
              <option value="Tukey">Tukey (α=0.5)</option>
            </optgroup>
            <optgroup label="Special">
              <option value="Lanczos">Lanczos</option>
              <option value="Poisson">Poisson</option>
              <option value="HannPoisson">Hann-Poisson</option>
              <option value="Bohman">Bohman</option>
              <option value="Cauchy">Cauchy</option>
              <option value="Riesz">Riesz</option>
            </optgroup>
          </select>
          <div style={{ "margin-left": "auto", display: "flex", "align-items": "center", gap: "var(--space-sm)" }}>
            <Show when={exportWavError()}>
              <span style={{ color: STATUS_BAD, "font-size": "var(--fs-sm)" }}>{exportWavError()}</span>
            </Show>
            <button
              class="tb-btn primary"
              disabled={exportingWav() || !activeBand()}
              onClick={handleExportWav}
              title="Export FIR as WAV"
            >
              {exportingWav() ? "Exporting..." : "Export WAV"}
            </button>
          </div>
        </div>

        {/* Export legend — Model + FIR (mag & phase) */}
        <div class="sum-vis-table">
          <table>
            <thead><tr>
              <td />
              <td class="sum-col-header" onClick={() => {
                const v = !(showExpModel() && showExpFir());
                setShowExpModel(v); setShowExpFir(v); safeToggleSeries(1, v); safeToggleSeries(2, v);
              }}>dB</td>
              <td class="sum-col-header" onClick={() => {
                const v = !(showExpModelPh() && showExpFirPh());
                setShowExpModelPh(v); setShowExpFirPh(v); safeToggleSeries(3, v); safeToggleSeries(4, v);
              }}>°</td>
            </tr></thead>
            <tbody>
              <tr>
                <td class="sum-row-header" onClick={() => {
                  const v = !(showExpModel() && showExpModelPh());
                  setShowExpModel(v); setShowExpModelPh(v); safeToggleSeries(1, v); safeToggleSeries(3, v);
                }}>MODEL</td>
                <td class="sum-cell">
                  <button class={`legend-item ${showExpModel() ? "" : "legend-off"}`}
                    onClick={() => { const v = !showExpModel(); setShowExpModel(v); safeToggleSeries(1, v); }}>
                    <span class="legend-swatch legend-swatch-dash" style={{ "border-color": exportColors().model }} />
                  </button>
                </td>
                <td class="sum-cell">
                  <button class={`legend-item ${showExpModelPh() ? "" : "legend-off"}`}
                    onClick={() => { const v = !showExpModelPh(); setShowExpModelPh(v); safeToggleSeries(3, v); }}>
                    <span class="legend-swatch legend-swatch-dash" style={{ "border-color": exportColors().modelPhase }} />
                  </button>
                </td>
              </tr>
              <tr>
                <td class="sum-row-header" onClick={() => {
                  const v = !(showExpFir() && showExpFirPh());
                  setShowExpFir(v); setShowExpFirPh(v); safeToggleSeries(2, v); safeToggleSeries(4, v);
                }}>FIR</td>
                <td class="sum-cell">
                  <button class={`legend-item ${showExpFir() ? "" : "legend-off"}`}
                    onClick={() => { const v = !showExpFir(); setShowExpFir(v); safeToggleSeries(2, v); }}>
                    <span class="legend-swatch" style={{ "background-color": exportColors().fir, "border-color": exportColors().fir }} />
                  </button>
                </td>
                <td class="sum-cell">
                  <button class={`legend-item ${showExpFirPh() ? "" : "legend-off"}`}
                    onClick={() => { const v = !showExpFirPh(); setShowExpFirPh(v); safeToggleSeries(4, v); }}>
                    <span class="legend-swatch legend-swatch-dash" style={{ "border-color": exportColors().firPhase }} />
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        {/* Export metrics bar */}
        <Show when={exportMetrics()}>
          {(m) => (
            <div class="export-metrics" style={{
              display: "flex", "flex-wrap": "wrap", gap: "6px 14px",
              padding: "3px 8px", "font-size": "var(--fs-sm)", color: "#b0b0bc",
              "border-top": "1px solid #2a2a35",
            }}>
              <span>{m().taps} taps</span>
              <span>{m().sampleRate / 1000}k</span>
              <span>{m().window}</span>
              <span>{m().phaseLabel}</span>
              <span style={{ color: m().causality >= 95 ? STATUS_GOOD : m().causality >= 80 ? STATUS_WARN : STATUS_BAD }}>
                Causal: {m().causality}%
              </span>
              <Show when={m().preRingMs > 0}>
                <span>Pre-ring: {m().preRingMs} ms</span>
              </Show>
              <span style={{ color: m().maxMagErr <= 0.5 ? STATUS_GOOD : m().maxMagErr <= 1.5 ? STATUS_WARN : STATUS_BAD }}>
                Mag err: {m().maxMagErr} dB
              </span>
              <span style={{ color: m().gdRippleMs <= 1 ? STATUS_GOOD : m().gdRippleMs <= 3 ? STATUS_WARN : STATUS_BAD }}>
                GD ripple: {m().gdRippleMs} ms
              </span>
              <Show when={m().peqCount > 0}>
                <span>PEQ: {m().peqCount}</span>
              </Show>
              <span>Norm: {m().normDb.toFixed(1)} dB</span>
            </div>
          )}
        </Show>
      </Show>
      <Show when={isSum() && legendEntries.length > 0 && plotTab() !== "gd" && plotTab() !== "export"}>
        {/* SUM matrix */}
        <div class="sum-vis-table">
          {(() => {
            const bandNames = () => appState.bands.map(b => b.name);
            const cols = () => [...bandNames(), "\u03A3"];
            const categories: ("target" | "measurement" | "corrected")[] = ["target", "measurement", "corrected"];
            const catLabels: Record<string, string> = { target: "TARGETS", measurement: "MEAS", corrected: "CORR+XO" };
            const catColors: Record<string, string> = { target: SUM_TARGET_COLOR, measurement: SUM_MEAS_COLOR, corrected: SUM_CORRECTED_COLOR };
            return (
              <table>
                <thead><tr>
                  <th class="sum-corner" />
                  <For each={cols()}>{(col) => {
                    const band = () => col === "\u03A3" ? null : appState.bands.find(b => b.name === col);
                    const hasPeq = () => {
                      const b = band();
                      return b ? (b.peqBands && b.peqBands.length > 0) : false;
                    };
                    const hasXo = () => {
                      const b = band();
                      return b ? (b.target.high_pass || b.target.low_pass) : false;
                    };
                    return (
                      <th onClick={() => toggleColumn(col)} title={`Toggle all ${col}`}>
                        {col}
                        <Show when={band() && hasXo()}>
                          <span
                            class="opt-dot"
                            style={{ background: hasPeq() ? "var(--status-good)" : "var(--status-bad)" }}
                            title={hasPeq() ? "PEQ optimized" : "Not optimized"}
                          />
                        </Show>
                      </th>
                    );
                  }}</For>
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
                              const isIrTab = () => plotTab() === "ir" || plotTab() === "step";
                              return (
                                <Show when={isIrTab()} fallback={(() => {
                                  // SPL: single swatch per cell
                                  const entry = () => findCellEntry(col, cat);
                                  const noPeq = () => {
                                    if (cat !== "corrected" || col === "\u03A3") return false;
                                    const b = appState.bands.find(bb => bb.name === col);
                                    return b ? (!b.peqBands || b.peqBands.length === 0) : false;
                                  };
                                  return (
                                    <Show when={entry()} fallback={<td class="sum-cell-empty" />}>
                                      {(e) => {
                                        const idx = () => legendEntries.findIndex(le => le.seriesIdx === e().seriesIdx);
                                        return (
                                          <td class="sum-cell">
                                            <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}>
                                              <span class={`legend-swatch ${e().dash ? "legend-swatch-dash" : ""}`} style={{ "background-color": e().dash ? "transparent" : e().color, "border-color": e().color }} />
                                            </button>
                                            <Show when={noPeq()}>
                                              <span title="No PEQ optimization" style={{ color: STATUS_BAD, "font-size": "var(--fs-xs)", "font-weight": "bold", "margin-left": "1px" }}>!</span>
                                            </Show>
                                          </td>
                                        );
                                      }}
                                    </Show>
                                  );
                                })()}>
                                  {/* IR/Step: two swatches per cell (IR + Step) */}
                                  {(() => {
                                    const pair = () => findCellEntryPair(col, cat);
                                    const irE = () => pair().ir;
                                    const stE = () => pair().step;
                                    const hasAny = () => irE() || stE();
                                    const noPeqIr = () => {
                                      if (cat !== "corrected" || col === "\u03A3") return false;
                                      const b = appState.bands.find(bb => bb.name === col);
                                      return b ? (!b.peqBands || b.peqBands.length === 0) : false;
                                    };
                                    return (
                                      <Show when={hasAny()} fallback={<td class="sum-cell-empty" />}>
                                        <td class="sum-cell" style={{ "white-space": "nowrap" }}>
                                          <Show when={irE()}>
                                            {(e) => {
                                              const idx = () => legendEntries.findIndex(le => le.seriesIdx === e().seriesIdx);
                                              return (
                                                <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }} style={{ padding: "1px var(--space-xxs)" }}>
                                                  <span class="legend-swatch" style={{ "background-color": e().visible ? e().color : "transparent", "border-color": e().color }} />
                                                </button>
                                              );
                                            }}
                                          </Show>
                                          <Show when={stE()}>
                                            {(e) => {
                                              const idx = () => legendEntries.findIndex(le => le.seriesIdx === e().seriesIdx);
                                              return (
                                                <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }} style={{ padding: "1px var(--space-xxs)" }}>
                                                  <span class="legend-swatch legend-swatch-dash" style={{ "border-color": e().color, "background-color": e().visible ? e().color : "transparent" }} />
                                                </button>
                                              );
                                            }}
                                          </Show>
                                          <Show when={noPeqIr()}>
                                            <span title="No PEQ optimization" style={{ color: STATUS_BAD, "font-size": "var(--fs-xs)", "font-weight": "bold", "margin-left": "1px" }}>!</span>
                                          </Show>
                                        </td>
                                      </Show>
                                    );
                                  })()}
                                </Show>
                              );
                            }}
                          </For>
                        </tr>
                      );
                    }}
                  </For>
                  <Show when={plotTab() === "ir" || plotTab() === "step"}>
                    <tr>
                      <td class="sum-row-header">VIEW</td>
                      <td colspan={cols().length} style={{ "text-align": "center" }}>
                        <button
                          class={`tb-btn tb-btn-sm ${legendEntries.some(e => e.label.endsWith(" IR") && e.visible) ? "active" : ""}`}
                          onClick={() => toggleAllIrOrStep("IR")}
                          style={{ "margin-right": "var(--space-xs)" }}
                        >IR</button>
                        <button
                          class={`tb-btn tb-btn-sm ${legendEntries.some(e => e.label.endsWith(" Step") && e.visible) ? "active" : ""}`}
                          onClick={() => toggleAllIrOrStep("Step")}
                        >Step</button>
                      </td>
                    </tr>
                  </Show>
                  <Show when={plotTab() === "freq" || plotTab() === "ir" || plotTab() === "step"}>
                    <tr>
                      <td class="sum-row-header">DELAY <span style={{ "font-size": "var(--fs-xs)", "font-weight": "normal", color: "var(--text-muted)" }}>ms</span></td>
                      <For each={bandNames()}>
                        {(bName) => {
                          const band = () => appState.bands.find(b => b.name === bName);
                          let dlTimer: ReturnType<typeof setTimeout> | null = null;
                          const commitDelay = (id: string, ms: number) => {
                            if (dlTimer) clearTimeout(dlTimer);
                            dlTimer = setTimeout(() => setAlignmentDelay(id, ms / 1000), 250);
                          };
                          return (
                            <td class="sum-cell">
                              <Show when={band()} fallback={<span />}>
                                {(b) => (
                                  <input
                                    type="number"
                                    class="delay-input"
                                    step="0.01"
                                    placeholder="0.00"
                                    value={((b().alignmentDelay ?? 0) * 1000).toFixed(2)}
                                    onChange={(e) => {
                                      const v = parseFloat(e.currentTarget.value);
                                      if (!isNaN(v)) commitDelay(b().id, v);
                                    }}
                                    onPointerDown={(e) => wheelEnabled.add(e.currentTarget)}
                                    onBlur={(e) => wheelEnabled.delete(e.currentTarget)}
                                    onWheel={(e) => {
                                      if (!wheelEnabled.has(e.currentTarget)) { e.preventDefault(); return; }
                                      e.preventDefault();
                                      const step = e.shiftKey ? 0.1 : 0.01;
                                      const cur = (b().alignmentDelay ?? 0) * 1000;
                                      const delta = e.deltaY < 0 ? step : -step;
                                      e.currentTarget.value = (cur + delta).toFixed(2);
                                      commitDelay(b().id, cur + delta);
                                    }}
                                  />
                                )}
                              </Show>
                            </td>
                          );
                        }}
                      </For>
                      {(() => {
                        const allPeqReady = () => {
                          const xoBands = appState.bands.filter(
                            b => b.measurement?.phase && b.measurement.phase.length > 0
                              && b.targetEnabled && (b.target.high_pass || b.target.low_pass)
                          );
                          return xoBands.length >= 2 && xoBands.every(b => b.peqBands && b.peqBands.length > 0);
                        };
                        return (
                          <td class="sum-cell-empty" style={{ "text-align": "center" }}>
                            <button
                              class="auto-align-btn"
                              title={allPeqReady()
                                ? "Auto-compute alignment delays (gradient descent)"
                                : "Requires PEQ optimization on all crossover bands"}
                              disabled={!allPeqReady()}
                              onClick={async () => {
                                if (!allPeqReady()) {
                                  console.warn("Auto-align requires PEQ optimization on all bands");
                                  return;
                                }
                                const bands = appState.bands.filter(
                                  b => b.measurement?.phase && b.measurement.phase.length > 0
                                );
                                if (bands.length < 2) return;
                                const plainBands = JSON.parse(JSON.stringify(bands));
                                const result = await computeAutoAlign(plainBands);
                                batch(() => {
                                  for (const [id, delay] of Object.entries(result.delays)) {
                                    setAlignmentDelay(id, delay);
                                  }
                                });
                              }}
                              class={`tb-btn tb-btn-sm`}
                              style={{
                                cursor: allPeqReady() ? "pointer" : "not-allowed",
                                opacity: allPeqReady() ? 1 : 0.4,
                              }}
                            >AUTO</button>
                          </td>
                        );
                      })()}
                    </tr>
                  </Show>
                </tbody>
              </table>
            );
          })()}
        </div>
      </Show>
      {/* IR-specific matrix removed — now using standard SUM matrix above */}
      <Show when={!isSum() && showLegend() && legendEntries.length > 0 && plotTab() === "freq"}>
        {/* Freq band matrix — checkboxes like IR/Step */}
        <div class="sum-vis-table">
          {(() => {
            const categories: ("target" | "measurement" | "corrected" | "peq")[] = ["measurement", "target", "peq", "corrected"];
            const catLabels: Record<string, string> = { target: "TARGET", measurement: "MEAS", corrected: "CORR", peq: "PEQ" };
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
                        <Show when={catEnts().length > 0}>
                          <tr>
                            <td class="sum-row-header" onClick={() => toggleCategory(cat)} style={{ opacity: (magE()?.visible || phE()?.visible) ? 1 : 0.4 }}>{catLabels[cat]}</td>
                            <td class="sum-cell">
                              <Show when={magE()}>{(e) => (
                                <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => toggleLegendEntry(legendEntries.indexOf(e()))}>
                                  <span class="legend-swatch" style={{ "background-color": e().visible ? e().color : "transparent", "border-color": e().color }} />
                                </button>
                              )}</Show>
                            </td>
                            <td class="sum-cell">
                              <Show when={phE()}>{(e) => (
                                <button class={`legend-item ${e().visible ? "" : "legend-off"}`} onClick={() => toggleLegendEntry(legendEntries.indexOf(e()))}>
                                  <span class={`legend-swatch legend-swatch-dash`} style={{ "background-color": "transparent", "border-color": e().color, opacity: e().visible ? 1 : 0.3 }} />
                                </button>
                              )}</Show>
                            </td>
                          </tr>
                        </Show>
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
          <Show when={exportComputing()}>
            <div class="plot-computing-overlay">
              <span class="plot-computing-text">Computing FIR...</span>
            </div>
          </Show>
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
      {/* Old band-legend removed — all modes use matrix above */}
    </div>
  );
}
