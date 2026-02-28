import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { TargetResponse, FirModelResult, FirConfig, PeqBand } from "../lib/types";
import {
  activeBand,
  sharedXScale,
  setSharedXScale,
  suppressXScaleSync,
  exportSampleRate,
  exportTaps,
  exportWindow,
  exportSnapshots,
  setExportSnapshots,
  exportYScale,
  setExportYScale,
} from "../stores/bands";
import type { ExportSnapshot } from "../stores/bands";

const MODEL_MAG_COLOR = "#FF9F43";     // orange — ideal model magnitude
const MODEL_PHASE_COLOR = "#FFCB80";   // light orange — model phase
const FIR_MAG_COLOR = "#38BDF8";       // light blue — FIR realized magnitude
const FIR_PHASE_COLOR = "#7DD3FC";     // lighter blue — FIR realized phase

// Snapshot overlay colors (muted, distinct from active curves)
const SNAP_COLORS = ["#808080", "#A855F7", "#EC4899", "#14B8A6"];

export default function ExportPlot() {
  let containerRef!: HTMLDivElement;
  const chartRef: { current: uPlot | undefined } = { current: undefined };

  const [cursorFreq, setCursorFreq] = createSignal("\u2014");
  const [cursorModelMag, setCursorModelMag] = createSignal("\u2014");
  const [cursorFirMag, setCursorFirMag] = createSignal("\u2014");
  const [hasData, setHasData] = createSignal(false);
  const [status, setStatus] = createSignal("");

  // Keep last rendered FIR data for snapshot capture
  const lastFirData: { freq: number[]; mag: number[]; phase: (number | null)[] } = { freq: [], mag: [], phase: [] };

  function takeSnapshot() {
    const band = activeBand();
    if (!band || lastFirData.freq.length === 0) return;
    const snaps = exportSnapshots(band.id);
    const idx = snaps.length;
    const color = SNAP_COLORS[idx % SNAP_COLORS.length];
    const label = `Snap ${idx + 1}`;
    setExportSnapshots(band.id, [...snaps, {
      label,
      freq: [...lastFirData.freq],
      mag: [...lastFirData.mag],
      phase: [...lastFirData.phase],
      color,
    }]);
    // Re-render chart to include new snapshot
    rerenderWithSnapshots();
  }

  function clearSnapshots() {
    const band = activeBand();
    if (!band) return;
    setExportSnapshots(band.id, []);
    rerenderWithSnapshots();
  }

  // Re-render chart with current data + updated snapshots
  function rerenderWithSnapshots() {
    const c = chartRef.current;
    if (!c || !c.data || c.data.length < 5) return;
    // Extract current base data from chart
    const freq = c.data[0] as number[];
    const modelMag = c.data[1] as number[];
    const modelPhase = c.data[2] as number[];
    const firMag = c.data[3] as number[];
    const firPhase = c.data[4] as number[];
    // Reconstruct unmasked phase from what we have (phase may be nulled)
    // For re-render we just pass through — renderChart will re-mask
    requestAnimationFrame(() =>
      renderChart(freq, modelMag, modelPhase as any, firMag, firPhase as any)
    );
  }

  // Track data range for anchored zoom
  let fitMagMin = -100;
  let fitMagMax = 100;

  // Mutable scale state — updated by setScale hook, used by range function
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

  function scrollY(direction: number) {
    const c = getChart();
    if (!c) return;
    const range = curMagMax - curMagMin;
    const step = range * 0.2 * direction;
    curMagMin += step;
    curMagMax += step;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
    setExportYScale({ min: curMagMin, max: curMagMax });
  }

  function zoomY(factor: number) {
    const c = getChart();
    if (!c) return;
    const center = (curMagMin + curMagMax) / 2;
    const half = ((curMagMax - curMagMin) / 2) * factor;
    curMagMin = center - half;
    curMagMax = center + half;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
    setExportYScale({ min: curMagMin, max: curMagMax });
  }

  function fitData() {
    const c = getChart();
    if (!c) return;
    c.setScale("x", { min: 5, max: 40000 });
    const pad = Math.max(3, (fitMagMax - fitMagMin) * 0.15);
    curMagMin = fitMagMin - pad;
    curMagMax = fitMagMax + pad;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
    setExportYScale({ min: curMagMin, max: curMagMax });
  }

  function renderChart(
    freq: number[],
    modelMag: number[],
    modelPhase: number[],
    firMag: number[],
    firPhase: number[],
  ) {
    if (!containerRef) return;

    // Save last FIR data for snapshot capture (phase saved after masking below)
    lastFirData.freq = freq;
    lastFirData.mag = firMag;

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

    // Mask phase where magnitude is too low (phase is meaningless noise there)
    const PHASE_MASK_THRESHOLD = -60; // dB below peak
    const peakModelMag = modelMag.reduce((a, b) => Math.max(a, b), -Infinity);
    const peakFirMag = firMag.reduce((a, b) => Math.max(a, b), -Infinity);

    const maskedModelPhase = modelPhase.map((p, i) =>
      modelMag[i] >= peakModelMag + PHASE_MASK_THRESHOLD ? p : null
    ) as (number | null)[];
    const maskedFirPhase = firPhase.map((p, i) =>
      firMag[i] >= peakFirMag + PHASE_MASK_THRESHOLD ? p : null
    ) as (number | null)[];

    // Save masked FIR phase for snapshot capture
    lastFirData.phase = maskedFirPhase;

    // Check if model phase is all zero (linear-phase filters) — hide model phase only
    const modelPhaseIsZero = modelPhase.every(p => p == null || Math.abs(p as number) < 0.5);

    // FIR phase is ALWAYS shown — it's the excess phase (deviation from ideal).
    const uData: uPlot.AlignedData = [
      freq,
      modelMag,
      modelPhaseIsZero ? modelPhase.map(() => null) as any : maskedModelPhase as any,
      firMag,
      maskedFirPhase as any,
    ];

    const series: uPlot.Series[] = [
      {},
      { label: "Model dB", stroke: MODEL_MAG_COLOR, width: 2, scale: "mag" },
      { label: "Model \u00B0", stroke: MODEL_PHASE_COLOR, width: 1.5, dash: [6, 3], scale: "phase", show: !modelPhaseIsZero },
      { label: "FIR dB", stroke: FIR_MAG_COLOR, width: 2, scale: "mag" },
      { label: "FIR \u00B0", stroke: FIR_PHASE_COLOR, width: 1.5, dash: [6, 3], scale: "phase" },
    ];

    // Add snapshot overlay series (magnitude + phase for each)
    const band = activeBand();
    const snaps = band ? exportSnapshots(band.id) : [];
    for (const snap of snaps) {
      // Interpolate snapshot data to current freq grid if needed
      const interpolateArr = (srcArr: (number | null)[]): (number | null)[] => {
        if (snap.freq.length === freq.length && snap.freq[0] === freq[0]) {
          return srcArr;
        }
        return freq.map(f => {
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
      series.push({
        label: `${snap.label} dB`,
        stroke: snap.color,
        width: 1.5,
        dash: [4, 3],
        scale: "mag",
      });
      (uData as (number | null)[][]).push(snapMag);

      // Phase series (same color, dotted)
      series.push({
        label: `${snap.label} °`,
        stroke: snap.color,
        width: 1,
        dash: [2, 2],
        scale: "phase",
      });
      (uData as (number | null)[][]).push(snapPhase);
    }

    // Auto Y-range from FIR data only (firMag is what actually gets exported)
    // Clamped to reasonable bounds so extreme model values (-700 dB) don't blow up the view
    let magMin = Infinity, magMax = -Infinity;
    for (const v of firMag) { if (v < magMin) magMin = v; if (v > magMax) magMax = v; }
    // Include snapshots in range
    for (const snap of snaps) {
      for (const v of snap.mag) { if (v < magMin) magMin = v; if (v > magMax) magMax = v; }
    }
    if (!isFinite(magMin)) magMin = -100;
    if (!isFinite(magMax)) magMax = 100;
    // Clamp auto-range to sensible bounds
    magMin = Math.max(magMin, -80);
    magMax = Math.min(magMax, 30);
    if (magMin >= magMax) { magMin = -80; magMax = 30; }
    fitMagMin = magMin;
    fitMagMax = magMax;
    const pad = Math.max(3, (magMax - magMin) * 0.15);

    // Y-scale priority: 1) saved from current chart, 2) persisted in store, 3) auto from data
    const storedY = exportYScale();
    const yMin = savedMagMin ?? storedY?.min ?? (magMin - pad);
    const yMax = savedMagMax ?? storedY?.max ?? (magMax + pad);
    curMagMin = yMin;
    curMagMax = yMax;

    const xs = sharedXScale();

    const opts: uPlot.Options = {
      width: w,
      height: h,
      series,
      scales: {
        x: { min: xs.min, max: xs.max, distr: 3 },
        mag: {
          auto: false,
          range: () => [curMagMin, curMagMax] as uPlot.Range.MinMax,
        },
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
          size: 56,
        },
        {
          label: "\u00B0",
          scale: "phase",
          side: 1,
          stroke: MODEL_PHASE_COLOR,
          grid: { show: false },
          ticks: { stroke: "rgba(255,203,128,0.2)" },
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
              setCursorFreq("\u2014"); setCursorModelMag("\u2014"); setCursorFirMag("\u2014");
              return;
            }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? (f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : Math.round(f) + " Hz") : "\u2014");
            const mm = u.data[1]?.[idx];
            setCursorModelMag(mm != null ? (mm > 0 ? "+" : "") + (mm as number).toFixed(1) + " dB" : "\u2014");
            const fm = u.data[3]?.[idx];
            setCursorFirMag(fm != null ? (fm > 0 ? "+" : "") + (fm as number).toFixed(1) + " dB" : "\u2014");
          },
        ],
      },
    };

    try {
      chartRef.current = new uPlot(opts, uData, containerRef);
    } catch (e) {
      console.error("ExportPlot uPlot error:", e);
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

  // Sync X-scale
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

  // Main effect: compute model + PEQ + FIR
  createEffect(() => {
    const band = activeBand();
    if (!band || !band.target) {
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = undefined; }
      setHasData(false);
      setStatus("No target configured");
      return;
    }

    // Track export FIR signals + per-band snapshots for reactivity
    const sr = exportSampleRate();
    const taps = exportTaps();
    const win = exportWindow();
    const _snaps = exportSnapshots(band.id); // track per-band snapshots

    const target = { ...band.target };
    const peqBands = band.peqBands?.filter((b: PeqBand) => b.enabled) ?? [];

    // Export is ALWAYS target + PEQ (model FIR), regardless of hybrid strategy.
    // Hybrid only affects PEQ optimization stage, not FIR export.
    computeModelFir(target, peqBands, sr, taps, win);
  });

  // --- Standard path: pure model FIR (target + PEQ, no measurement) ---
  async function computeModelFir(
    target: any,
    peqBands: PeqBand[],
    sampleRate: number,
    taps: number,
    window: string,
  ) {
    try {
      setStatus("Computing...");

      // 1. Evaluate pure target (HP/LP/shelf/tilt)
      const [freq, response] = await invoke<[number[], TargetResponse]>("evaluate_target_standalone", {
        target,
        nPoints: 512,
        fMin: 5,
        fMax: 40000,
      });

      const targetMag = response.magnitude;
      let modelMag = [...targetMag];
      let modelPhase = response.phase;

      // 2. Add PEQ contribution (if any bands exist)
      let peqMagArr: number[] = [];
      if (peqBands.length > 0) {
        const [peqMag, peqPhase] = await invoke<[number[], number[]]>("compute_peq_complex", {
          freq,
          bands: peqBands,
          sampleRate,
        });
        peqMagArr = peqMag;
        modelMag = modelMag.map((v, i) => v + peqMag[i]);
        modelPhase = modelPhase.map((v, i) => {
          let w = (v + peqPhase[i]) % 360;
          if (w > 180) w -= 360;
          if (w < -180) w += 360;
          return w;
        });
      }

      // 3. Generate FIR from combined model
      const isLin = (f: any) => !f || f.linear_phase || f.filter_type === "Gaussian";
      const allLinear = isLin(target.high_pass) && isLin(target.low_pass);

      const firConfig: FirConfig = {
        taps,
        sample_rate: sampleRate,
        max_boost_db: 24.0,
        noise_floor_db: -150.0,
        window: window as any,
        phase_mode: allLinear ? "LinearPhase" : "MinimumPhase",
      };

      const firResult = await invoke<FirModelResult>("generate_model_fir", {
        freq,
        targetMag,
        peqMag: peqMagArr,
        modelPhase,
        config: firConfig,
      });

      setHasData(true);
      const peqInfo = peqBands.length > 0 ? ` \u00B7 ${peqBands.length} PEQ` : "";
      const phaseLabel = allLinear ? "Linear-Phase" : "Min-Phase";
      const normLabel = firResult.norm_db !== 0
        ? ` \u00B7 Norm: ${firResult.norm_db > 0 ? "\u2212" : "+"}${Math.abs(firResult.norm_db).toFixed(1)} dB`
        : "";
      setStatus(`${taps} taps \u00B7 ${sampleRate / 1000}k \u00B7 ${window} \u00B7 ${phaseLabel}${peqInfo}${normLabel}`);
      const normModelMag = modelMag.map(v => v - firResult.norm_db);
      requestAnimationFrame(() =>
        renderChart(freq, normModelMag, modelPhase, firResult.realized_mag, firResult.realized_phase)
      );
    } catch (e) {
      console.error("ExportPlot compute failed:", e);
      setHasData(false);
      setStatus(`Error: ${e}`);
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
          <span class="readout-label" style={{ color: MODEL_MAG_COLOR }}>Model:</span>
          <span class="readout-value">{cursorModelMag()}</span>
        </span>
        <span class="readout-item">
          <span class="readout-label" style={{ color: FIR_MAG_COLOR }}>FIR:</span>
          <span class="readout-value">{cursorFirMag()}</span>
        </span>
        <span class="impulse-sep" />
        <button class="tb-btn" onClick={takeSnapshot} title="Save current FIR as overlay for comparison">SNAP</button>
        {(() => {
          const b = activeBand();
          const snaps = b ? exportSnapshots(b.id) : [];
          return (
            <>
              {snaps.length > 0 && (
                <button class="tb-btn" onClick={clearSnapshots} title="Clear all snapshots">CLR</button>
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
        <span class="impulse-sep" />
        <span style={{ "font-size": "10px", "color": "#8b8b96" }}>
          {status() || "Filter Model vs FIR Realization"}
        </span>
      </div>
      <div class="impulse-body">
        <div class="axis-controls axis-controls-y">
          <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In dB">+</button>
          <button class="axis-btn" onClick={() => scrollY(1)} title="Scroll Up">{"\u25B2"}</button>
          <button class="axis-btn" onClick={() => scrollY(-1)} title="Scroll Down">{"\u25BC"}</button>
          <button class="axis-btn" onClick={() => zoomY(1.6)} title="Zoom Out dB">-</button>
          <button class="axis-btn fit-btn" onClick={fitData} title="Fit">FIT</button>
        </div>
        <div class="impulse-center">
          <div ref={containerRef} class="impulse-plot" />
          {!hasData() && (
            <div class="impulse-empty-overlay">
              {status() || "Configure target filters to see model"}
            </div>
          )}
          <div class="axis-controls axis-controls-x">
            <button class="axis-btn" onClick={() => zoomX(0.6)} title="Zoom In Freq">+</button>
            <button class="axis-btn" onClick={() => scrollX(-1)} title="Scroll Left">{"\u25C0"}</button>
            <button class="axis-btn" onClick={() => scrollX(1)} title="Scroll Right">{"\u25B6"}</button>
            <button class="axis-btn" onClick={() => zoomX(1.6)} title="Zoom Out Freq">-</button>
          </div>
        </div>
      </div>
    </div>
  );
}
