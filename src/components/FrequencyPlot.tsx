import { createEffect, createSignal, onCleanup, onMount, For, Show } from "solid-js";
import { createStore } from "solid-js/store";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import { MEASUREMENT_COLORS } from "../lib/types";
import type { Measurement, TargetResponse, FilterType } from "../lib/types";
import { appState, activeBand, isSum, activeTab, sharedXScale, setSharedXScale, suppressXScaleSync, selectedPeqIdx, setBandLowPass, setBandCrossNormDb, plotShowOnly, setPlotShowOnly } from "../stores/bands";
import type { SmoothingMode, BandState } from "../stores/bands";
import { needAutoFit, setNeedAutoFit } from "../App";
import { computeFloorBounce } from "../lib/floor-bounce";
import { openCrossoverDialog, type CrossoverDialogData } from "./CrossoverDialog";

const TARGET_COLOR = "#FFD700";
const TARGET_PHASE_COLOR = "#FFD700";
const CORRECTED_COLOR = "#22C55E"; // green — measurement + PEQ correction
const SMOOTHED_HALF_OCT_COLOR = "#FF6B6B"; // red — 1/2 oct smoothed measurement (PEQ intermediate target)

// Цвета per-band таргетов (светлее основных)
const TARGET_BAND_COLORS = [
  "#7CB3FF", // light blue
  "#FFB870", // light orange
  "#6EE89A", // light green
  "#C98DF7", // light purple
  "#F77070", // light red
  "#5CD6E8", // light cyan
  "#F7C35B", // light amber
  "#F77DBF", // light pink
];

// Цвета per-band corrected кривых (оттенки зелёного)
const CORRECTED_BAND_COLORS = [
  "#34D399", // emerald
  "#4ADE80", // green
  "#2DD4BF", // teal
  "#A3E635", // lime
  "#86EFAC", // light green
  "#5EEAD4", // light teal
  "#BEF264", // light lime
  "#6EE7B7", // mint
];

function smoothingConfig(mode: SmoothingMode): { variable: boolean; fixed_fraction: number | null } {
  if (mode === "var") return { variable: true, fixed_fraction: null };
  const fractions: Record<string, number> = { "1/3": 1/3, "1/6": 1/6, "1/12": 1/12, "1/24": 1/24 };
  return { variable: false, fixed_fraction: fractions[mode] ?? 1/6 };
}

function wrapPhase(phase: number[]): number[] {
  return phase.map((v) => {
    let w = v % 360;
    if (w > 180) w -= 360;
    if (w < -180) w += 360;
    return w;
  });
}

function fmtFreq(v: number): string {
  if (v >= 1000) return (v / 1000).toFixed(2) + " kHz";
  return v.toFixed(1) + " Hz";
}

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
  // Y-zoom anchor: passband reference level (avg 200-2000 Hz, or 0 dB without measurements)
  let zoomCenter = 0;

  const [cursorFreq, setCursorFreq] = createSignal("—");
  const [cursorSPL, setCursorSPL] = createSignal("—");
  const [cursorPhase, setCursorPhase] = createSignal("—");
  const [legendEntries, setLegendEntries] = createStore<LegendEntry[]>([]);
  const [showLegend, setShowLegend] = createSignal(false);

  // Crossover drag state
  const [hoveredXo, setHoveredXo] = createSignal<number | null>(null); // index in crossovers array
  const [draggingXo, setDraggingXo] = createSignal<number | null>(null);
  const [dragFreq, setDragFreq] = createSignal<number | null>(null); // visual override during drag
  let currentCrossovers: CrossoverPoint[] = []; // cached for mouse handlers

  function zoomY(factor: number) {
    if (!chart) return;
    const half = ((curMagMax - curMagMin) / 2) * factor;
    if (half * 2 < 2 || half * 2 > 400) return;
    curMagMin = zoomCenter - half;
    curMagMax = zoomCenter + half;
    chart.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  function scrollY(direction: number) {
    if (!chart) return;
    const range = curMagMax - curMagMin;
    const step = range * 0.2 * direction;
    curMagMin += step;
    curMagMax += step;
    chart.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  function zoomX(factor: number) {
    if (!chart) return;
    const s = chart.scales["x"];
    if (!s || s.min == null || s.max == null || s.min <= 0) return;
    const logMin = Math.log10(s.min);
    const logMax = Math.log10(s.max);
    const logCenter = (logMin + logMax) / 2;
    const logHalf = ((logMax - logMin) / 2) * factor;
    const newMin = Math.max(1, Math.pow(10, logCenter - logHalf));
    const newMax = Math.min(100000, Math.pow(10, logCenter + logHalf));
    chart.setScale("x", { min: newMin, max: newMax });
  }

  function scrollX(direction: number) {
    if (!chart) return;
    const s = chart.scales["x"];
    if (!s || s.min == null || s.max == null || s.min <= 0) return;
    const logMin = Math.log10(s.min);
    const logMax = Math.log10(s.max);
    const logRange = logMax - logMin;
    const step = logRange * 0.15 * direction;
    const newMin = Math.max(1, Math.pow(10, logMin + step));
    const newMax = Math.min(100000, Math.pow(10, logMax + step));
    chart.setScale("x", { min: newMin, max: newMax });
  }

  function fitData() {
    if (!chart) return;
    curMagMin = fitMagMin;
    curMagMax = fitMagMax;
    chart.setScale("mag", { min: curMagMin, max: curMagMax });
    chart.setScale("phase", { min: -190, max: 190 });
    chart.setScale("x", { min: 20, max: 20000 });
  }

  // Переключение видимости серии через легенду
  function toggleLegendEntry(idx: number) {
    if (!chart) return;
    const entry = legendEntries[idx];
    const newVis = !entry.visible;
    setLegendEntries(idx, "visible", newVis);
    chart.setSeries(entry.seriesIdx, { show: newVis });
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
      }
    }
  }

  // React to external "show only X" command (e.g. after PEQ optimize)
  createEffect(() => {
    const cats = plotShowOnly();
    if (!cats || !chart) return;
    const showSet = new Set(cats);
    for (let i = 0; i < legendEntries.length; i++) {
      const show = showSet.has(legendEntries[i].category);
      if (legendEntries[i].visible !== show) {
        setLegendEntries(i, "visible", show);
        chart.setSeries(legendEntries[i].seriesIdx, { show });
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
      // Remove crossover event listeners before destroying
      if (chart.over) {
        chart.over.removeEventListener("mousemove", handleXoMouseMove);
        chart.over.removeEventListener("mousedown", handleXoMouseDown);
        chart.over.removeEventListener("dblclick", handleXoDblClick);
      }
      chart.destroy();
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

    // Fixed default range: ±100 dB around passband level
    const halfFit = 100;
    fitMagMin = zoomCenter - halfFit;
    fitMagMax = zoomCenter + halfFit;
    curMagMin = savedMagMin ?? fitMagMin;
    curMagMax = savedMagMax ?? fitMagMax;

    const yLabel = input.hasMeasurements ? "dB SPL" : "dB";

    const axes: uPlot.Axis[] = [
      {
        stroke: "#8b8b96",
        grid: { stroke: "rgba(255,255,255,0.06)" },
        ticks: { stroke: "rgba(255,255,255,0.12)" },
        values: (_u: uPlot, vals: number[]) =>
          vals.map((v) => {
            if (v == null) return "";
            if (v >= 1000) return (v / 1000) + "k";
            return String(Math.round(v * 10) / 10);
          }),
      },
      {
        label: yLabel, scale: "mag", stroke: "#8b8b96",
        grid: { stroke: "rgba(255,255,255,0.06)" },
        ticks: { stroke: "rgba(255,255,255,0.12)" },
        values: (_u: uPlot, vals: number[]) => vals.map((v) => (v == null ? "" : v.toFixed(0))),
        size: 50,
      },
      {
        label: "Phase (\u00B0)", scale: "phase", side: 1, stroke: "#8b8b96",
        grid: { show: false },
        ticks: { stroke: "rgba(255,255,255,0.12)" },
      },
    ];

    // Restore previous legend visibility into series before chart creation
    const prevVisMap = new Map<string, boolean>();
    for (const e of legendEntries) prevVisMap.set(e.label, e.visible);

    let mergedLegend: LegendEntry[] | undefined;
    if (input.legend && input.legend.length > 0) {
      mergedLegend = input.legend.map((e) => ({
        ...e,
        visible: prevVisMap.has(e.label) ? prevVisMap.get(e.label)! : e.visible,
      }));
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
              return;
            }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? fmtFreq(f) : "—");
            // SPL: первая mag-серия
            let splVal: number | null | undefined;
            for (let si = 1; si < allSeries.length; si++) {
              if ((allSeries[si] as any).scale === "mag") {
                splVal = u.data[si]?.[idx];
                break;
              }
            }
            setCursorSPL(splVal != null ? splVal.toFixed(1) + " dB" : "—");
            // Phase: первая phase-серия
            let phVal: number | null | undefined;
            for (let si = 1; si < allSeries.length; si++) {
              if ((allSeries[si] as any).scale === "phase") {
                phVal = u.data[si]?.[idx];
                break;
              }
            }
            setCursorPhase(phVal != null ? phVal.toFixed(1) + "\u00B0" : "—");
          },
        ],
        draw: [
          // Floor bounce overlay: рисуем вертикальные полосы на null-частотах
          (u: uPlot) => {
            const nullFreqs = input.floorBounceNulls;
            if (!nullFreqs || nullFreqs.length === 0) return;
            const ctx = u.ctx;
            const plotLeft = u.bbox.left / devicePixelRatio;
            const plotTop = u.bbox.top / devicePixelRatio;
            const plotWidth = u.bbox.width / devicePixelRatio;
            const plotHeight = u.bbox.height / devicePixelRatio;

            ctx.save();
            ctx.fillStyle = "rgba(255, 165, 0, 0.12)";

            for (const f of nullFreqs) {
              const xPos = u.valToPos(f, "x", false);
              if (xPos < plotLeft || xPos > plotLeft + plotWidth) continue;
              // Ширина полосы: ~2% от plot width, min 2px
              const bandWidth = Math.max(2, plotWidth * 0.015);
              ctx.fillRect(xPos - bandWidth / 2, plotTop, bandWidth, plotHeight);
            }

            ctx.restore();
          },
          // Selected PEQ band — vertical dashed line
          (u: uPlot) => {
            const selIdx = selectedPeqIdx();
            if (selIdx == null) return;
            const bd = activeBand();
            if (!bd || !bd.peqBands || selIdx >= bd.peqBands.length) return;
            const freqHz = bd.peqBands[selIdx].freq_hz;

            const ctx = u.ctx;
            // valToPos with can=true returns canvas-pixel position including plot offset
            const cx = u.valToPos(freqHz, "x", true);
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotRight = plotLeft + u.bbox.width;
            const plotBottom = plotTop + u.bbox.height;

            if (cx < plotLeft || cx > plotRight) return;

            ctx.save();
            ctx.strokeStyle = "#FF9F43";
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(cx, plotTop);
            ctx.lineTo(cx, plotBottom);
            ctx.stroke();

            // Frequency label at top of line
            ctx.setLineDash([]);
            ctx.fillStyle = "#FF9F43";
            const dpr = devicePixelRatio || 1;
            ctx.font = `${Math.round(10 * dpr)}px sans-serif`;
            ctx.textAlign = "center";
            const label = freqHz >= 1000 ? (freqHz / 1000).toFixed(1) + "k" : Math.round(freqHz).toString();
            ctx.fillText(label, cx, plotTop - 4 * dpr);

            ctx.restore();
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
        let sum = 0, n = 0;
        for (let i = 0; i < measurement.freq.length; i++) {
          if (measurement.freq[i] >= 200 && measurement.freq[i] <= 2000) {
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
  // Main reactive effect
  // ----------------------------------------------------------------
  createEffect(() => {
    const showPhase = appState.showPhase;
    const showMag = appState.showMag;
    const showTarget = appState.showTarget;
    const sumMode = isSum();
    const band = activeBand();
    const bandsSnapshot = JSON.stringify(appState.bands);
    const _tab = activeTab(); // track tab changes (align tab shows extra curves)

    if (sumMode) {
      renderSumMode(showPhase, showMag, showTarget, bandsSnapshot);
    } else if (band) {
      renderBandMode(band, showPhase, showMag, showTarget);
    } else {
      if (chart) { chart.destroy(); chart = undefined; }
      setShowLegend(false);
      setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
    }
  });

  // ----------------------------------------------------------------
  // Single band rendering
  // ----------------------------------------------------------------
  async function renderBandMode(band: BandState, showPhase: boolean, showMag: boolean, showTarget: boolean) {
    const gen = ++renderGen;
    try {
      const result = await evaluateBand(band, showPhase);
      if (gen !== renderGen) return; // stale render, discard

      if (!result.freq) {
        if (chart) { chart.destroy(); chart = undefined; }
        setShowLegend(false);
        setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
        return;
      }

      // Compute zoom anchor: avg magnitude 200-2000 Hz, or 0 dB without measurement
      if (result.measurement) {
        let s = 0, n = 0;
        for (let i = 0; i < result.measurement.freq.length; i++) {
          if (result.measurement.freq[i] >= 200 && result.measurement.freq[i] <= 2000) {
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

      if (result.measurement && showMag) {
        const color = MEASUREMENT_COLORS[0];
        uSeries.push({ label: result.measurement.name + " dB", stroke: color, width: 2, scale: "mag" });
        uData.push(result.measurement.magnitude);
        legend.push({ label: "Measurement", color, dash: false, visible: true, seriesIdx: sIdx, category: "measurement" });
        sIdx++;

        if (showPhase && result.measurement.phase) {
          uSeries.push({ label: result.measurement.name + " \u00B0", stroke: color, width: 1, dash: [6, 3], scale: "phase" });
          uData.push(wrapPhase(result.measurement.phase));
          legend.push({ label: "Meas \u00B0", color, dash: true, visible: true, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        }
      }

      if (showTarget && result.targetMag && result.targetMag.length > 0) {
        uSeries.push({ label: "Target dB", stroke: TARGET_COLOR, width: 2, dash: [8, 4], scale: "mag" });
        uData.push(result.targetMag);
        legend.push({ label: "Target", color: TARGET_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
        sIdx++;
      }

      if (showTarget && result.targetPhase && result.targetPhase.length > 0 && showPhase) {
        // Если полоса инвертирована — сдвигаем фазу таргета на 180°
        const phase = band.inverted
          ? result.targetPhase.map((v) => v + 180)
          : result.targetPhase;
        uSeries.push({ label: "Target \u00B0", stroke: TARGET_PHASE_COLOR, width: 1.5, dash: [4, 4], scale: "phase" });
        uData.push(wrapPhase(phase));
        legend.push({ label: "Target \u00B0", color: TARGET_PHASE_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
        sIdx++;
      }

      // 1/1-octave smoothed measurement — intermediate target for PEQ (shown on align tab)
      const isAlignTab = activeTab() === "align";
      if (isAlignTab && result.measurement && showMag) {
        try {
          const smoothedHalf = await invoke<number[]>("get_smoothed", {
            freq: result.measurement.freq,
            magnitude: result.measurement.magnitude,
            config: { variable: false, fixed_fraction: 1.0 },
          });
          uSeries.push({
            label: "Meas 1/1 oct",
            stroke: SMOOTHED_HALF_OCT_COLOR,
            width: 2,
            scale: "mag",
          });
          uData.push(smoothedHalf);
          legend.push({ label: "Meas 1/1 oct", color: SMOOTHED_HALF_OCT_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        } catch (e) {
          console.warn("1/2 oct smoothing failed:", e);
        }
      }

      // Corrected curve = measurement + PEQ + cross-section (filters + makeup)
      const hasPeq = band.peqBands && band.peqBands.length > 0;
      const hasFilters = band.targetEnabled && (band.target.high_pass || band.target.low_pass);
      if (result.measurement && (hasPeq || hasFilters)) {
        try {
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
              measMag: result.measurement.magnitude,
              targetMag: result.targetMag,
              peqCorrection: peqMag ?? [],
              highPass: band.target.high_pass,
              lowPass: band.target.low_pass,
            });
            xsMag = xm;
            xsPhase = xp;
            setBandCrossNormDb(band.id, xNorm);
          }

          // Corrected magnitude = measurement + PEQ + cross-section (green, solid)
          if (showMag) {
            const corrected = result.measurement.magnitude.map(
              (v: number, i: number) =>
                v + (peqMag ? peqMag[i] : 0) + (xsMag ? xsMag[i] : 0)
            );
            uSeries.push({
              label: "Corrected dB",
              stroke: CORRECTED_COLOR,
              width: 2,
              scale: "mag",
            });
            uData.push(corrected);
            legend.push({ label: "Corrected", color: CORRECTED_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          }

          // Corrected phase = measurement phase + PEQ phase + cross-section phase (green, dashed)
          if (showPhase && result.measurement.phase) {
            const correctedPhase = result.measurement.phase.map(
              (v: number, i: number) =>
                v + (peqPhase ? peqPhase[i] : 0) + (xsPhase ? xsPhase[i] : 0)
            );
            uSeries.push({
              label: "Corrected \u00B0",
              stroke: CORRECTED_COLOR,
              width: 1.5,
              dash: [4, 4],
              scale: "phase",
            });
            uData.push(wrapPhase(correctedPhase));
            legend.push({ label: "Corrected \u00B0", color: CORRECTED_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          }
        } catch (e) {
          console.warn("Correction computation failed:", e);
        }
      }

      // Floor bounce
      let floorBounceNulls: number[] | undefined;
      const fb = band.settings?.floorBounce;
      if (fb && fb.enabled) {
        const fbResult = computeFloorBounce(fb.speakerHeight, fb.micHeight, fb.distance);
        floorBounceNulls = fbResult.nullFreqs;
      }

      if (gen !== renderGen) return; // stale after async work
      requestAnimationFrame(() =>
        renderChart({
          freq: result.freq!,
          uSeries,
          uData,
          hasMeasurements: !!result.measurement,
          legend,
          floorBounceNulls,
        })
      );
    } catch (e) {
      console.error("Band render failed:", e);
    }
  }

  // ----------------------------------------------------------------
  // SUM mode rendering
  // ----------------------------------------------------------------
  async function renderSumMode(showPhase: boolean, showMag: boolean, showTarget: boolean, _bandsJson: string) {
    const gen = ++renderGen;
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
        if (chart) { chart.destroy(); chart = undefined; }
        setShowLegend(false);
        setCursorFreq("—"); setCursorSPL("—"); setCursorPhase("—");
        return;
      }

      const freq = commonFreq;
      const fMin = freq[0];
      const fMax = freq[freq.length - 1];
      const nPts = freq.length;

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
        if (m.freq.length === nPts && m.freq[0] === fMin && m.freq[m.freq.length - 1] === fMax) {
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

      // Общий reference level из всех загруженных замеров (единый для всех полос)
      let globalRef = 0;
      {
        let totalS = 0, totalN = 0;
        for (const rm of resampled) {
          if (!rm) continue;
          for (let k = 0; k < freq.length; k++) {
            if (freq[k] >= 200 && freq[k] <= 2000) { totalS += rm.magnitude[k]; totalN++; }
          }
        }
        if (totalN > 0) globalRef = totalS / totalN;
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
          const color = MEASUREMENT_COLORS[i % MEASUREMENT_COLORS.length];
          uSeries.push({ label: bands[i].name + " dB", stroke: color, width: 1.5, scale: "mag" });
          uData.push(rm.magnitude);
          legend.push({ label: bands[i].name, color, dash: false, visible: true, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        }
      }

      // --- Per-band таргеты ---
      if (showTarget) {
        for (let i = 0; i < bands.length; i++) {
          const tMag = perBandTargetMags[i];
          if (!tMag) continue;
          const color = TARGET_BAND_COLORS[i % TARGET_BAND_COLORS.length];
          uSeries.push({ label: bands[i].name + " tgt", stroke: color, width: 1.5, dash: [6, 4], scale: "mag" });
          uData.push(tMag);
          legend.push({ label: bands[i].name + " tgt", color, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
          sIdx++;
        }
      }

      // --- Per-band corrected кривые (measurement + PEQ + filters) ---
      const perBandCorrected: (number[] | null)[] = [];
      if (showMag) {
        for (let i = 0; i < bands.length; i++) {
          const rm = resampled[i];
          const tMag = perBandTargetMags[i];
          if (!rm) { perBandCorrected.push(null); continue; }

          const hasPeq = bands[i].peqBands && bands[i].peqBands.length > 0;
          const hasFilters = bands[i].targetEnabled && (bands[i].target.high_pass || bands[i].target.low_pass);

          if (!hasPeq && !hasFilters) { perBandCorrected.push(null); continue; }

          try {
            let peqMag: number[] | null = null;
            if (hasPeq) {
              const [pm] = await invoke<[number[], number[]]>("compute_peq_complex", {
                freq, bands: bands[i].peqBands,
              });
              peqMag = pm;
            }

            let xsMag: number[] | null = null;
            if (hasFilters && tMag) {
              const [xm] = await invoke<[number[], number[], number]>("compute_cross_section", {
                freq, measMag: rm.magnitude, targetMag: tMag,
                peqCorrection: peqMag ?? [],
                highPass: bands[i].target.high_pass,
                lowPass: bands[i].target.low_pass,
              });
              xsMag = xm;
            }

            const corrected = rm.magnitude.map(
              (v: number, j: number) => v + (peqMag ? peqMag[j] : 0) + (xsMag ? xsMag[j] : 0)
            );
            perBandCorrected.push(corrected);

            const color = CORRECTED_BAND_COLORS[i % CORRECTED_BAND_COLORS.length];
            uSeries.push({ label: bands[i].name + " corr", stroke: color, width: 1.5, scale: "mag" });
            uData.push(corrected);
            legend.push({ label: bands[i].name + " corr", color, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          } catch (e) {
            console.warn("SUM corrected failed for band", bands[i].name, e);
            perBandCorrected.push(null);
          }
        }

        // Σ corrected (когерентное сложение корректированных кривых)
        const corrIndices = perBandCorrected.map((c, i) => c ? i : -1).filter(i => i >= 0);
        if (corrIndices.length > 0) {
          const hasAllPhaseCorr = corrIndices.every(
            (ci) => resampled[ci]!.phase && resampled[ci]!.phase!.length === nPts
          );
          if (hasAllPhaseCorr) {
            const sumRe = new Array(nPts).fill(0);
            const sumIm = new Array(nPts).fill(0);
            for (const ci of corrIndices) {
              const corr = perBandCorrected[ci]!;
              const rm = resampled[ci]!;
              const sign = bands[ci].inverted ? -1 : 1;
              // Фаза corrected ≈ фаза замера (PEQ/filter фазу не учитываем для простоты)
              for (let j = 0; j < nPts; j++) {
                const amp = Math.pow(10, corr[j] / 20) * sign;
                const phRad = rm.phase![j] * Math.PI / 180;
                sumRe[j] += amp * Math.cos(phRad);
                sumIm[j] += amp * Math.sin(phRad);
              }
            }
            const sumCorrDb = new Array(nPts);
            for (let j = 0; j < nPts; j++) {
              const amplitude = Math.sqrt(sumRe[j] * sumRe[j] + sumIm[j] * sumIm[j]);
              sumCorrDb[j] = amplitude > 0 ? 20 * Math.log10(amplitude) : -200;
            }
            uSeries.push({ label: "\u03A3 corr", stroke: CORRECTED_COLOR, width: 2.5, scale: "mag" });
            uData.push(sumCorrDb);
            legend.push({ label: "\u03A3 corrected", color: CORRECTED_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
            sIdx++;
          } else {
            // Инкогерентное сложение (без фазы)
            const sumCorr = new Array(nPts).fill(0);
            for (const ci of corrIndices) {
              const corr = perBandCorrected[ci]!;
              const sign = bands[ci].inverted ? -1 : 1;
              for (let j = 0; j < nPts; j++) {
                sumCorr[j] += Math.pow(10, corr[j] / 20) * sign;
              }
            }
            const sumCorrDb = sumCorr.map((v: number) =>
              Math.abs(v) > 0 ? 20 * Math.log10(Math.abs(v)) : -200
            );
            uSeries.push({ label: "\u03A3 corr", stroke: CORRECTED_COLOR, width: 2.5, scale: "mag" });
            uData.push(sumCorrDb);
            legend.push({ label: "\u03A3 corrected", color: CORRECTED_COLOR, dash: false, visible: true, seriesIdx: sIdx, category: "corrected" });
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
          const sumRe = new Array(nPts).fill(0);
          const sumIm = new Array(nPts).fill(0);
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
          uSeries.push({ label: "\u03A3 dB", stroke: "#FFFFFF", width: 2.5, scale: "mag" });
          uData.push(sumMagDb);
          legend.push({ label: "\u03A3 meas", color: "#FFFFFF", dash: false, visible: true, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        } else {
          const sumMag = new Array(nPts).fill(0);
          for (const mi of measIndices) {
            const rm = resampled[mi]!;
            const sign = bands[mi].inverted ? -1 : 1;
            for (let j = 0; j < nPts; j++) {
              sumMag[j] += Math.pow(10, rm.magnitude[j] / 20) * sign;
            }
          }
          const sumMagDb = sumMag.map((v: number) =>
            Math.abs(v) > 0 ? 20 * Math.log10(Math.abs(v)) : -200
          );
          uSeries.push({ label: "\u03A3 dB", stroke: "#FFFFFF", width: 2.5, scale: "mag" });
          uData.push(sumMagDb);
          legend.push({ label: "\u03A3 meas", color: "#FFFFFF", dash: false, visible: true, seriesIdx: sIdx, category: "measurement" });
          sIdx++;
        }
      }

      // --- Суммарный таргет (когерентное сложение с учётом фазы и инверсии) ---
      const enabledNorm: number[][] = [];
      const enabledPhase: number[][] = [];
      const enabledInverted: boolean[] = [];
      for (let i = 0; i < bands.length; i++) {
        if (perBandTargetNorm[i]) {
          enabledNorm.push(perBandTargetNorm[i]!);
          enabledPhase.push(perBandTargetNormPhase[i] ?? new Array(freq.length).fill(0));
          enabledInverted.push(bands[i].inverted);
        }
      }
      if (showTarget && enabledNorm.length > 0) {
        const avgRef = refLevels.length > 0
          ? refLevels.reduce((a, b) => a + b, 0) / refLevels.length
          : 0;

        // Когерентное (комплексное) суммирование нормализованных передаточных функций
        const sumRe = new Array(freq.length).fill(0);
        const sumIm = new Array(freq.length).fill(0);
        for (let n = 0; n < enabledNorm.length; n++) {
          const mag = enabledNorm[n];
          const ph = enabledPhase[n];
          const sign = enabledInverted[n] ? -1 : 1; // инверсия = умножение на -1
          for (let j = 0; j < freq.length; j++) {
            const amp = Math.pow(10, mag[j] / 20) * sign;
            const phRad = ph[j] * Math.PI / 180; // градусы → радианы
            sumRe[j] += amp * Math.cos(phRad);
            sumIm[j] += amp * Math.sin(phRad);
          }
        }
        const sumTarget = new Array(freq.length);
        const sumTargetPhase = new Array(freq.length);
        for (let j = 0; j < freq.length; j++) {
          const re = sumRe[j];
          const im = sumIm[j];
          const amplitude = Math.sqrt(re * re + im * im);
          sumTarget[j] = amplitude > 0 ? 20 * Math.log10(amplitude) + avgRef : -200;
          sumTargetPhase[j] = Math.atan2(im, re) * 180 / Math.PI; // радианы → градусы
        }

        uSeries.push({ label: "\u03A3 tgt", stroke: TARGET_COLOR, width: 2.5, dash: [8, 4], scale: "mag" });
        uData.push(sumTarget);
        legend.push({ label: "\u03A3 target", color: TARGET_COLOR, dash: true, visible: true, seriesIdx: sIdx, category: "target" });
        sIdx++;
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
    });
  }

  onCleanup(() => { if (chart) chart.destroy(); });

  return (
    <div class="plot-wrapper">
      <div class="cursor-readout">
        <span class="readout-item">
          <span class="readout-label">Freq:</span>
          <span class="readout-value">{cursorFreq()}</span>
        </span>
        <span class="readout-item">
          <span class="readout-label">SPL:</span>
          <span class="readout-value">{cursorSPL()}</span>
        </span>
        <span class="readout-item">
          <span class="readout-label">Phase:</span>
          <span class="readout-value">{cursorPhase()}</span>
        </span>
        {/* Легенда per-band — плоский список (не в SUM) */}
        <Show when={showLegend() && !isSum() && legendEntries.length > 0}>
          <span class="readout-sep" />
          <For each={legendEntries}>
            {(entry, idx) => (
              <button
                class={`legend-item ${entry.visible ? "" : "legend-off"}`}
                onClick={() => toggleLegendEntry(idx())}
              >
                <span
                  class={`legend-swatch ${entry.dash ? "legend-swatch-dash" : ""}`}
                  style={{ "background-color": entry.dash ? "transparent" : entry.color, "border-color": entry.color }}
                />
                <span class="legend-text">{entry.label}</span>
              </button>
            )}
          </For>
        </Show>
      </div>
      {/* SUM visibility table */}
      <Show when={isSum() && showLegend() && legendEntries.length > 0}>
        <div class="sum-vis-table">
          {/* Row: Targets */}
          {(() => {
            const catEntries = () => legendEntries.filter(e => e.category === "target");
            const allOn = () => { const ce = catEntries(); return ce.length > 0 && ce.every(e => e.visible); };
            const anyOn = () => catEntries().some(e => e.visible);
            return (
              <div class="sum-vis-row">
                <button
                  class={`sum-vis-all ${allOn() ? "on" : anyOn() ? "partial" : ""}`}
                  onClick={() => toggleCategory("target")}
                >
                  <span class="sum-vis-all-swatch" style={{ "border-color": TARGET_COLOR }} />
                  Targets
                </button>
                <div class="sum-vis-items">
                  <For each={catEntries()}>
                    {(entry) => {
                      const idx = () => legendEntries.findIndex(e => e.seriesIdx === entry.seriesIdx);
                      return (
                        <button
                          class={`legend-item ${entry.visible ? "" : "legend-off"}`}
                          onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}
                        >
                          <span
                            class={`legend-swatch ${entry.dash ? "legend-swatch-dash" : ""}`}
                            style={{ "background-color": entry.dash ? "transparent" : entry.color, "border-color": entry.color }}
                          />
                          <span class="legend-text">{entry.label}</span>
                        </button>
                      );
                    }}
                  </For>
                </div>
              </div>
            );
          })()}
          {/* Row: Measurements */}
          {(() => {
            const catEntries = () => legendEntries.filter(e => e.category === "measurement");
            const allOn = () => { const ce = catEntries(); return ce.length > 0 && ce.every(e => e.visible); };
            const anyOn = () => catEntries().some(e => e.visible);
            return (
              <div class="sum-vis-row">
                <button
                  class={`sum-vis-all ${allOn() ? "on" : anyOn() ? "partial" : ""}`}
                  onClick={() => toggleCategory("measurement")}
                >
                  <span class="sum-vis-all-swatch" style={{ "border-color": "#4A9EFF" }} />
                  Measurements
                </button>
                <div class="sum-vis-items">
                  <For each={catEntries()}>
                    {(entry) => {
                      const idx = () => legendEntries.findIndex(e => e.seriesIdx === entry.seriesIdx);
                      return (
                        <button
                          class={`legend-item ${entry.visible ? "" : "legend-off"}`}
                          onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}
                        >
                          <span
                            class={`legend-swatch ${entry.dash ? "legend-swatch-dash" : ""}`}
                            style={{ "background-color": entry.dash ? "transparent" : entry.color, "border-color": entry.color }}
                          />
                          <span class="legend-text">{entry.label}</span>
                        </button>
                      );
                    }}
                  </For>
                </div>
              </div>
            );
          })()}
          {/* Row: Corrected */}
          {(() => {
            const catEntries = () => legendEntries.filter(e => e.category === "corrected");
            const allOn = () => { const ce = catEntries(); return ce.length > 0 && ce.every(e => e.visible); };
            const anyOn = () => catEntries().some(e => e.visible);
            return (
              <div class={`sum-vis-row ${catEntries().length === 0 ? "sum-vis-row-disabled" : ""}`}>
                <button
                  class={`sum-vis-all ${allOn() ? "on" : anyOn() ? "partial" : ""}`}
                  onClick={() => toggleCategory("corrected")}
                  disabled={catEntries().length === 0}
                >
                  <span class="sum-vis-all-swatch" style={{ "border-color": CORRECTED_COLOR }} />
                  Corrected
                </button>
                <div class="sum-vis-items">
                  <For each={catEntries()}>
                    {(entry) => {
                      const idx = () => legendEntries.findIndex(e => e.seriesIdx === entry.seriesIdx);
                      return (
                        <button
                          class={`legend-item ${entry.visible ? "" : "legend-off"}`}
                          onClick={() => { const i = idx(); if (i >= 0) toggleLegendEntry(i); }}
                        >
                          <span
                            class={`legend-swatch ${entry.dash ? "legend-swatch-dash" : ""}`}
                            style={{ "background-color": entry.dash ? "transparent" : entry.color, "border-color": entry.color }}
                          />
                          <span class="legend-text">{entry.label}</span>
                        </button>
                      );
                    }}
                  </For>
                </div>
              </div>
            );
          })()}
        </div>
      </Show>
      <div class="plot-body">
        <div class="axis-controls axis-controls-y">
          <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In dB">+</button>
          <button class="axis-btn" onClick={() => scrollY(1)} title="Scroll Up dB">▲</button>
          <button class="axis-btn" onClick={() => scrollY(-1)} title="Scroll Down dB">▼</button>
          <button class="axis-btn" onClick={() => zoomY(1.6)} title="Zoom Out dB">−</button>
          <button class="axis-btn fit-btn" onClick={fitData} title="Fit data to view">FIT</button>
        </div>
        <div class="plot-center">
          <div ref={containerRef} class="frequency-plot" />
          <div class="axis-controls axis-controls-x">
            <button class="axis-btn" onClick={() => zoomX(0.6)} title="Zoom In Freq">+</button>
            <button class="axis-btn" onClick={() => scrollX(-1)} title="Scroll Left">◀</button>
            <button class="axis-btn" onClick={() => scrollX(1)} title="Scroll Right">▶</button>
            <button class="axis-btn" onClick={() => zoomX(1.6)} title="Zoom Out Freq">−</button>
          </div>
        </div>
      </div>
    </div>
  );
}
