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
} from "../stores/bands";

const MODEL_MAG_COLOR = "#FF9F43";     // orange — ideal model magnitude
const MODEL_PHASE_COLOR = "#FFCB80";   // light orange — model phase
const FIR_MAG_COLOR = "#38BDF8";       // light blue — FIR realized magnitude
const FIR_PHASE_COLOR = "#7DD3FC";     // lighter blue — FIR realized phase

export default function ExportPlot() {
  let containerRef!: HTMLDivElement;
  const chartRef: { current: uPlot | undefined } = { current: undefined };

  const [cursorFreq, setCursorFreq] = createSignal("—");
  const [cursorModelMag, setCursorModelMag] = createSignal("—");
  const [cursorFirMag, setCursorFirMag] = createSignal("—");
  const [hasData, setHasData] = createSignal(false);
  const [status, setStatus] = createSignal("");

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
  }

  function zoomY(factor: number) {
    const c = getChart();
    if (!c) return;
    const center = (curMagMin + curMagMax) / 2;
    const half = ((curMagMax - curMagMin) / 2) * factor;
    if (half * 2 < 2 || half * 2 > 400) return;
    curMagMin = center - half;
    curMagMax = center + half;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  function fitData() {
    const c = getChart();
    if (!c) return;
    c.setScale("x", { min: 5, max: 40000 });
    const pad = Math.max(3, (fitMagMax - fitMagMin) * 0.15);
    curMagMin = fitMagMin - pad;
    curMagMax = fitMagMax + pad;
    c.setScale("mag", { min: curMagMin, max: curMagMax });
  }

  function renderChart(
    freq: number[],
    modelMag: number[],
    modelPhase: number[],
    firMag: number[],
    firPhase: number[],
  ) {
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

    // Check if model phase is all zero (linear-phase filters) — hide model phase only
    const modelPhaseIsZero = modelPhase.every(p => Math.abs(p) < 0.5);

    // FIR phase is ALWAYS shown — it's the excess phase (deviation from ideal).
    // For linear-phase FIR, linear delay is already subtracted in Rust,
    // so what remains is the windowing/truncation error — useful to see.
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
      { label: "Model °", stroke: MODEL_PHASE_COLOR, width: 1.5, dash: [6, 3], scale: "phase", show: !modelPhaseIsZero },
      { label: "FIR dB", stroke: FIR_MAG_COLOR, width: 2, scale: "mag" },
      { label: "FIR °", stroke: FIR_PHASE_COLOR, width: 1.5, dash: [6, 3], scale: "phase" },
    ];

    // Auto Y-range from data — track for anchored zoom
    let magMin = Infinity, magMax = -Infinity;
    for (const v of modelMag) { if (v < magMin) magMin = v; if (v > magMax) magMax = v; }
    for (const v of firMag) { if (v < magMin) magMin = v; if (v > magMax) magMax = v; }
    if (!isFinite(magMin)) magMin = -100;
    if (!isFinite(magMax)) magMax = 100;
    fitMagMin = magMin;
    fitMagMax = magMax;
    const pad = Math.max(3, (magMax - magMin) * 0.15);
    const yMin = savedMagMin ?? (magMin - pad);
    const yMax = savedMagMax ?? (magMax + pad);
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
              setCursorFreq("—"); setCursorModelMag("—"); setCursorFirMag("—");
              return;
            }
            const f = u.data[0][idx];
            setCursorFreq(f != null ? (f >= 1000 ? (f / 1000).toFixed(2) + " kHz" : Math.round(f) + " Hz") : "—");
            const mm = u.data[1]?.[idx];
            setCursorModelMag(mm != null ? (mm > 0 ? "+" : "") + (mm as number).toFixed(1) + " dB" : "—");
            const fm = u.data[3]?.[idx];
            setCursorFirMag(fm != null ? (fm > 0 ? "+" : "") + (fm as number).toFixed(1) + " dB" : "—");
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

    // Track export FIR signals
    const sr = exportSampleRate();
    const taps = exportTaps();
    const win = exportWindow();

    const target = { ...band.target };
    const peqBands = band.peqBands?.filter((b: PeqBand) => b.enabled) ?? [];

    computeAndRender(target, peqBands, sr, taps, win);
  });

  async function computeAndRender(
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

      // 3. Generate FIR from combined model.
      //    Target phase: linear or min-phase per HP/LP flags.
      //    PEQ phase: always min-phase (Hilbert from PEQ magnitude).
      //    Gaussian filters are inherently linear-phase regardless of the flag.
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
      const peqInfo = peqBands.length > 0 ? ` · ${peqBands.length} PEQ` : "";
      const phaseLabel = allLinear ? "Linear-Phase" : "Min-Phase";
      const normLabel = firResult.norm_db !== 0
        ? ` · Norm: ${firResult.norm_db > 0 ? "−" : "+"}${Math.abs(firResult.norm_db).toFixed(1)} dB`
        : "";
      setStatus(`${taps} taps · ${sampleRate / 1000}k · ${window} · ${phaseLabel}${peqInfo}${normLabel}`);
      // Shift model curves down by norm_db to match normalized FIR
      const normModelMag = modelMag.map(v => v - firResult.norm_db);
      // Show realized FIR phase (excess phase after delay removal).
      // For linear-phase: ≈ 0°. For min-phase: actual min-phase response.
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
        <span style={{ "font-size": "10px", "color": "#8b8b96" }}>
          {status() || "Filter Model vs FIR Realization"}
        </span>
      </div>
      <div class="impulse-body">
        <div class="axis-controls axis-controls-y">
          <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In dB">+</button>
          <button class="axis-btn" onClick={() => scrollY(1)} title="Scroll Up">▲</button>
          <button class="axis-btn" onClick={() => scrollY(-1)} title="Scroll Down">▼</button>
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
            <button class="axis-btn" onClick={() => scrollX(-1)} title="Scroll Left">◀</button>
            <button class="axis-btn" onClick={() => scrollX(1)} title="Scroll Right">▶</button>
            <button class="axis-btn" onClick={() => zoomX(1.6)} title="Zoom Out Freq">-</button>
          </div>
        </div>
      </div>
    </div>
  );
}
