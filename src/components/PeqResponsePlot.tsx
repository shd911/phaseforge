import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import { activeBand, appState, sharedXScale, setSharedXScale, suppressXScaleSync, selectedPeqIdx, setSelectedPeqIdx, addPeqBand, updatePeqBand, commitPeqBand, setPeqDragging } from "../stores/bands";
import type { PeqBand } from "../lib/types";

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

  function peqGainRange(): { top: number; bot: number } {
    const band = activeBand();
    const peqBands = (band?.peqBands ?? []).filter((b: PeqBand) => b.enabled);
    let gMin = 0, gMax = 0;
    for (const pb of peqBands) {
      if (pb.gain_db < gMin) gMin = pb.gain_db;
      if (pb.gain_db > gMax) gMax = pb.gain_db;
    }
    return { top: Math.max(gMax + 2, 3), bot: Math.min(gMin - 2, -3) };
  }

  function fitData() {
    const c = getChart();
    if (!c) return;
    c.setScale("x", { min: 20, max: 20000 });
    const { top, bot } = peqGainRange();
    curMagMin = bot;
    curMagMax = top;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  // --- Tooltip state ---
  const [tooltip, setTooltip] = createSignal<{ x: number; y: number; text: string } | null>(null);

  function handlePlotMouseMove(e: MouseEvent) {
    if (dragPeqIdx != null) return; // don't show tooltip during drag
    const c = getChart();
    const band = activeBand();
    if (!c || !band?.peqBands) { setTooltip(null); return; }
    const rect = containerRef.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (devicePixelRatio || 1);
    const cy = (e.clientY - rect.top) * (devicePixelRatio || 1);
    const hitIdx = findPeqBandAtPixel(cx, cy);
    if (hitIdx != null) {
      const pb = band.peqBands[hitIdx];
      const fLabel = pb.freq_hz >= 1000 ? (pb.freq_hz / 1000).toFixed(1) + "k" : Math.round(pb.freq_hz).toString();
      setTooltip({
        x: e.clientX - rect.left + 12,
        y: e.clientY - rect.top - 8,
        text: `${fLabel} Hz  ${pb.gain_db > 0 ? "+" : ""}${pb.gain_db.toFixed(1)} dB  Q${pb.q.toFixed(1)}`,
      });
    } else {
      setTooltip(null);
    }
  }

  // --- PEQ graph interaction: drag bands, double-click to add, wheel to adjust Q ---
  let dragPeqIdx: number | null = null;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragMoved = false;

  const PEQ_HIT_RADIUS = 12; // pixels

  function findPeqBandAtPixel(cx: number, cy: number): number | null {
    const c = getChart();
    const band = activeBand();
    if (!c || !band || !band.peqBands) return null;

    let closest: number | null = null;
    let closestDist = Infinity;

    for (let i = 0; i < band.peqBands.length; i++) {
      const pb = band.peqBands[i];
      if (!pb.enabled) continue;
      const px = c.valToPos(pb.freq_hz, "x", true);
      const py = c.valToPos(pb.gain_db, "mag", true);
      const dist = Math.hypot(cx - px, cy - py);
      if (dist < PEQ_HIT_RADIUS && dist < closestDist) {
        closest = i;
        closestDist = dist;
      }
    }
    return closest;
  }

  function handlePlotMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    const c = getChart();
    if (!c) return;
    const band = activeBand();
    if (!band) return;

    const rect = containerRef.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (devicePixelRatio || 1);
    const cy = (e.clientY - rect.top) * (devicePixelRatio || 1);

    const hitIdx = findPeqBandAtPixel(cx, cy);
    if (hitIdx != null) {
      dragPeqIdx = hitIdx;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      dragMoved = false;
      setSelectedPeqIdx(hitIdx);
      setTooltip(null);
      e.preventDefault();
      e.stopPropagation();

      const dpr = devicePixelRatio || 1;
      const plotLeftCSS = c.bbox.left / dpr;
      const plotTopCSS = c.bbox.top / dpr;

      const onMove = (ev: MouseEvent) => {
        if (dragPeqIdx == null) return;
        const dx = ev.clientX - dragStartX;
        const dy = ev.clientY - dragStartY;
        if (!dragMoved && Math.hypot(dx, dy) < 3) return;
        if (!dragMoved) setPeqDragging(true);
        dragMoved = true;

        const freq = c.posToVal(ev.clientX - rect.left - plotLeftCSS, "x");
        const gain = c.posToVal(ev.clientY - rect.top - plotTopCSS, "mag");
        if (freq != null && gain != null && freq > 0) {
          const clampedFreq = Math.max(20, Math.min(20000, freq));
          const clampedGain = Math.round(gain * 10) / 10;
          updatePeqBand(band.id, dragPeqIdx!, { freq_hz: Math.round(clampedFreq), gain_db: clampedGain });
        }
      };

      const onUp = () => {
        if (dragPeqIdx != null && dragMoved) {
          const newIdx = commitPeqBand(band.id, dragPeqIdx);
          setSelectedPeqIdx(newIdx);
        }
        dragPeqIdx = null;
        dragMoved = false;
        setPeqDragging(false);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    }
  }

  function handlePlotDblClick(e: MouseEvent) {
    const c = getChart();
    const band = activeBand();
    if (!c || !band || !c.over) return;

    const rect = c.over.getBoundingClientRect();
    const freq = c.posToVal(e.clientX - rect.left, "x");
    const gain = c.posToVal(e.clientY - rect.top, "mag");
    if (freq == null || gain == null || freq <= 0) return;

    const clampedFreq = Math.max(20, Math.min(20000, freq));
    const clampedGain = Math.round(gain * 10) / 10;
    const newBand: PeqBand = { freq_hz: Math.round(clampedFreq), gain_db: clampedGain, q: 2.0, enabled: true, filter_type: "Peaking" };
    addPeqBand(band.id, newBand);
    const newIdx = commitPeqBand(band.id, 0); // addPeqBand prepends, so idx=0
    setSelectedPeqIdx(newIdx);
  }

  function handlePlotWheel(e: WheelEvent) {
    const selIdx = selectedPeqIdx();
    const band = activeBand();
    if (selIdx == null || !band || !band.peqBands[selIdx]) return;

    // Only intercept when hovering near the selected band marker
    const c = getChart();
    if (!c) return;
    const rect = containerRef.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (devicePixelRatio || 1);
    const cy = (e.clientY - rect.top) * (devicePixelRatio || 1);
    const pb = band.peqBands[selIdx];
    const px = c.valToPos(pb.freq_hz, "x", true);
    const py = c.valToPos(pb.gain_db, "mag", true);
    if (Math.hypot(cx - px, cy - py) > 40) return; // only near the marker

    e.preventDefault();
    e.stopPropagation();
    const currentQ = pb.q;
    const step = e.shiftKey ? 0.5 : 0.1;
    const delta = e.deltaY > 0 ? -step : step;
    const newQ = Math.max(0.1, Math.min(20, Math.round((currentQ + delta) * 10) / 10));
    updatePeqBand(band.id, selIdx, { q: newQ });
  }

  function renderChart(freq: number[], mag: number[], phase: number[]) {
    if (!containerRef) return;

    const uData: uPlot.AlignedData = [freq, mag, phase];

    // If chart already exists, just update data — no destroy/recreate
    if (chartRef.current) {
      chartRef.current.setData(uData, false);
      chartRef.current.redraw(false, false);
      return;
    }

    // First time: create chart
    let savedMagMin: number | null = null;
    let savedMagMax: number | null = null;

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 200);
    const h = Math.max(rect.height, 80);

    dataXMin = freq[0] ?? 20;
    dataXMax = freq[freq.length - 1] ?? 20000;

    const series: uPlot.Series[] = [
      {},
      { label: "PEQ dB", stroke: PEQ_MAG_COLOR, width: 2, scale: "mag" },
      { label: "PEQ \u00B0", stroke: PEQ_PHASE_COLOR, width: 1.5, dash: [6, 3], scale: "phase" },
    ];

    // Auto Y-range from PEQ band gains (+2/-2 dB padding)
    const { top: autoTop, bot: autoBot } = peqGainRange();
    const yMin = savedMagMin ?? autoBot;
    const yMax = savedMagMax ?? autoTop;
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
          stroke: "#9b9ba6",
          grid: { stroke: "rgba(255,255,255,0.12)" },
          ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) =>
              v == null ? "" : v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : v.toString()
            ),
        },
        {
          label: "dB",
          scale: "mag",
          stroke: "#9b9ba6",
          grid: { stroke: "rgba(255,255,255,0.12)" },
          ticks: { stroke: "rgba(255,255,255,0.20)" },
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
          ticks: { stroke: "rgba(245,158,11,0.25)" },
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
          // Draw PEQ band markers (circles) + selected band highlight
          (u: uPlot) => {
            const bd = activeBand();
            if (!bd || !bd.peqBands || bd.peqBands.length === 0) return;
            const selIdx = selectedPeqIdx();
            const ctx = u.ctx;
            const dpr = devicePixelRatio || 1;
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotRight = plotLeft + u.bbox.width;
            const plotBottom = plotTop + u.bbox.height;

            ctx.save();

            for (let i = 0; i < bd.peqBands.length; i++) {
              const pb = bd.peqBands[i];
              if (!pb.enabled) continue;
              const fHz = pb.freq_hz;
              const gDb = pb.gain_db;
              const cx = u.valToPos(fHz, "x", true);
              const cy = u.valToPos(gDb, "mag", true);
              if (cx < plotLeft || cx > plotRight) continue;

              const isSel = i === selIdx;
              const r = (isSel ? 7 : 5) * dpr;

              // Circle
              ctx.beginPath();
              ctx.arc(cx, cy, r, 0, Math.PI * 2);
              ctx.fillStyle = isSel ? "#FF9F43" : "rgba(56, 189, 248, 0.8)";
              ctx.fill();
              ctx.strokeStyle = isSel ? "#fff" : "rgba(255,255,255,0.5)";
              ctx.lineWidth = isSel ? 2 : 1;
              ctx.setLineDash([]);
              ctx.stroke();

              // Selected: vertical dashed line + label
              if (isSel) {
                ctx.strokeStyle = "#FF9F43";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(cx, plotTop);
                ctx.lineTo(cx, plotBottom);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = "#FF9F43";
                ctx.font = `${Math.round(10 * dpr)}px sans-serif`;
                ctx.textAlign = "center";
                const fLabel = fHz >= 1000 ? (fHz / 1000).toFixed(1) + "k" : Math.round(fHz).toString();
                ctx.fillText(`${fLabel} | ${gDb > 0 ? "+" : ""}${gDb.toFixed(1)} dB | Q${pb.q.toFixed(1)}`, cx, plotTop - 4 * dpr);
              }
            }

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
          <div
            ref={containerRef}
            class="impulse-plot"
            onMouseDown={handlePlotMouseDown}
            onMouseMove={handlePlotMouseMove}
            onMouseLeave={() => setTooltip(null)}
            onDblClick={handlePlotDblClick}
            onWheel={handlePlotWheel}
            style={{ cursor: "crosshair", position: "relative" }}
          />
          {tooltip() && (
            <div style={{
              position: "absolute",
              left: tooltip()!.x + "px",
              top: tooltip()!.y + "px",
              background: "rgba(30,30,40,0.92)",
              color: "#e0e0e0",
              padding: "3px 8px",
              "border-radius": "4px",
              "font-size": "11px",
              "white-space": "nowrap",
              "pointer-events": "none",
              "z-index": "100",
              border: "1px solid rgba(255,255,255,0.15)",
            }}>{tooltip()!.text}</div>
          )}
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
