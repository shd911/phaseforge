import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import { activeBand, appState, sharedXScale, setSharedXScale, suppressXScaleSync, selectedPeqIdx } from "../stores/bands";

const PEQ_MAG_COLOR = "#38BDF8";   // light blue — PEQ magnitude
const PEQ_PHASE_COLOR = "#F59E0B"; // amber — PEQ phase

export default function PeqResponsePlot() {
  let containerRef!: HTMLDivElement;
  const chartRef: { current: uPlot | undefined } = { current: undefined };

  const [cursorFreq, setCursorFreq] = createSignal("—");
  const [cursorMag, setCursorMag] = createSignal("—");
  const [cursorPhase, setCursorPhase] = createSignal("—");
  const [hasData, setHasData] = createSignal(false);

  // Axis zoom/pan helpers
  let dataXMin = 20;
  let dataXMax = 20000;

  // Mutable Y-scale state — updated by buttons, used by range function
  let curMagMin = -100;
  let curMagMax = 100;

  function getChart(): uPlot | undefined { return chartRef.current; }

  function zoomX(factor: number) {
    const c = getChart();
    if (!c) return;
    const s = c.scales["x"];
    if (!s || s.min == null || s.max == null) return;
    const logMin = Math.log10(s.min);
    const logMax = Math.log10(s.max);
    const center = (logMin + logMax) / 2;
    const half = ((logMax - logMin) / 2) * factor;
    c.setScale("x", { min: 10 ** (center - half), max: 10 ** (center + half) });
  }

  function scrollX(direction: number) {
    const c = getChart();
    if (!c) return;
    const s = c.scales["x"];
    if (!s || s.min == null || s.max == null) return;
    const logRange = Math.log10(s.max) - Math.log10(s.min);
    const step = logRange * 0.2 * direction;
    c.setScale("x", { min: 10 ** (Math.log10(s.min) + step), max: 10 ** (Math.log10(s.max) + step) });
  }

  function zoomY(factor: number) {
    const c = getChart();
    if (!c) return;
    const center = 0; // PEQ correction always centers on 0 dB
    const half = ((curMagMax - curMagMin) / 2) * factor;
    curMagMin = center - half;
    curMagMax = center + half;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  function fitData() {
    const c = getChart();
    if (!c) return;
    c.setScale("x", { min: 20, max: 20000 });
    curMagMin = -24;
    curMagMax = 12;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  function renderChart(freq: number[], mag: number[], phase: number[]) {
    if (!containerRef) return;

    // Save current Y scale
    let savedMagMin: number | null = null;
    let savedMagMax: number | null = null;
    if (chartRef.current) {
      const ys = chartRef.current.scales["mag"];
      if (ys?.min != null && ys?.max != null) { savedMagMin = ys.min; savedMagMax = ys.max; }
      chartRef.current.destroy();
      chartRef.current = undefined;
    }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 200);
    const h = Math.max(rect.height, 80);

    dataXMin = freq[0] ?? 20;
    dataXMax = freq[freq.length - 1] ?? 20000;

    const uData: uPlot.AlignedData = [freq, mag, phase];

    const series: uPlot.Series[] = [
      {},
      { label: "PEQ dB", stroke: PEQ_MAG_COLOR, width: 2, scale: "mag" },
      { label: "PEQ \u00B0", stroke: PEQ_PHASE_COLOR, width: 1.5, dash: [6, 3], scale: "phase" },
    ];

    // Auto Y-range from data
    let magMin = 0, magMax = 0;
    for (const v of mag) { if (v < magMin) magMin = v; if (v > magMax) magMax = v; }
    const pad = Math.max(3, (magMax - magMin) * 0.15);
    const yMin = savedMagMin ?? Math.min(-100, magMin - pad);
    const yMax = savedMagMax ?? Math.max(100, magMax + pad);
    curMagMin = yMin;
    curMagMax = yMax;

    // Use shared X-scale so both plots stay synced
    const xs = sharedXScale();

    const opts: uPlot.Options = {
      width: w,
      height: h,
      series,
      scales: {
        x: { min: xs.min, max: xs.max, distr: 3 }, // log — synced with FrequencyPlot
        mag: { auto: false, range: () => [curMagMin, curMagMax] as uPlot.Range.MinMax },
        phase: { auto: false, range: [-190, 190] as uPlot.Range.MinMax },
      },
      axes: [
        {
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) =>
              v == null ? "" : v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : v.toString()
            ),
        },
        {
          label: "dB",
          scale: "mag",
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(1))),
          size: 50,
        },
        {
          label: "\u00B0",
          scale: "phase",
          side: 1,
          stroke: PEQ_PHASE_COLOR,
          grid: { show: false },
          ticks: { stroke: "rgba(245,158,11,0.2)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(0) + "\u00B0")),
          size: 45,
        },
      ],
      legend: { show: false },
      cursor: {
        drag: { x: false, y: false, setScale: false },
      },
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
              setCursorFreq("—"); setCursorMag("—"); setCursorPhase("—");
              return;
            }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? (f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : Math.round(f) + " Hz") : "—");
            const m = u.data[1]?.[idx];
            setCursorMag(m != null ? (m > 0 ? "+" : "") + (m as number).toFixed(1) + " dB" : "—");
            const p = u.data[2]?.[idx];
            setCursorPhase(p != null ? (p as number).toFixed(1) + "\u00B0" : "—");
          },
        ],
        draw: [
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
        ],
      },
    };

    try {
      chartRef.current = new uPlot(opts, uData, containerRef);
    } catch (e) {
      console.error("PEQ Response uPlot error:", e);
    }
  }

  onMount(() => {
    const observer = new ResizeObserver(() => {
      if (chartRef.current && containerRef) {
        const rect = containerRef.getBoundingClientRect();
        chartRef.current.setSize({ width: Math.max(rect.width, 200), height: Math.max(rect.height, 80) });
      }
    });
    observer.observe(containerRef);
    onCleanup(() => observer.disconnect());
  });

  // Sync X-scale from FrequencyPlot
  createEffect(() => {
    const xs = sharedXScale();
    const c = chartRef.current;
    if (!c) return;
    const cur = c.scales["x"];
    if (cur?.min != null && cur?.max != null) {
      if (Math.abs(cur.min - xs.min) < 0.01 && Math.abs(cur.max - xs.max) < 0.01) return;
    }
    suppressXScaleSync(() => {
      c.setScale("x", { min: xs.min, max: xs.max });
    });
  });

  // Main reactive effect
  createEffect(() => {
    const band = activeBand();
    if (!band || !band.measurement || !band.peqBands || band.peqBands.length === 0) {
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = undefined; }
      setHasData(false);
      setCursorFreq("—"); setCursorMag("—"); setCursorPhase("—");
      return;
    }

    const freq = band.measurement.freq;
    const peqBands = [...band.peqBands]; // track reactivity

    computeAndRender(freq, peqBands);
  });

  // Redraw when selected PEQ index changes (for vertical dashed line)
  createEffect(() => {
    const _sel = selectedPeqIdx(); // track
    const c = chartRef.current;
    if (c) c.redraw(false, false);
  });

  async function computeAndRender(freq: number[], peqBands: any[]) {
    try {
      const [mag, phase] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq,
        bands: peqBands,
      });
      setHasData(true);
      requestAnimationFrame(() => renderChart(freq, mag, phase));
    } catch (e) {
      console.error("PEQ complex response failed:", e);
      setHasData(false);
    }
  }

  onCleanup(() => {
    if (chartRef.current) chartRef.current.destroy();
  });

  return (
    <div class="impulse-wrapper">
      <div class="impulse-toolbar">
        <span class="readout-item">
          <span class="readout-label">Freq:</span>
          <span class="readout-value">{cursorFreq()}</span>
        </span>
        <span class="readout-item">
          <span class="readout-label">Mag:</span>
          <span class="readout-value">{cursorMag()}</span>
        </span>
        <span class="readout-item">
          <span class="readout-label">Phase:</span>
          <span class="readout-value">{cursorPhase()}</span>
        </span>
        <span class="impulse-sep" />
        <span style={{ "font-size": "10px", "color": "#8b8b96" }}>PEQ Filter Response</span>
      </div>
      <div class="impulse-body">
        <div class="axis-controls axis-controls-y">
          <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In dB">+</button>
          <button class="axis-btn" onClick={() => zoomY(1.6)} title="Zoom Out dB">-</button>
          <button class="axis-btn fit-btn" onClick={fitData} title="Fit">FIT</button>
        </div>
        <div class="impulse-center">
          <div ref={containerRef} class="impulse-plot" />
          {!hasData() && (
            <div class="impulse-empty-overlay">No PEQ filters computed</div>
          )}
          <div class="axis-controls axis-controls-x">
            <button class="axis-btn" onClick={() => zoomX(0.6)} title="Zoom In Freq">+</button>
            <button class="axis-btn" onClick={() => scrollX(-1)} title="Scroll Left">&lsaquo;</button>
            <button class="axis-btn" onClick={() => scrollX(1)} title="Scroll Right">&rsaquo;</button>
            <button class="axis-btn" onClick={() => zoomX(1.6)} title="Zoom Out Freq">-</button>
          </div>
        </div>
      </div>
    </div>
  );
}
