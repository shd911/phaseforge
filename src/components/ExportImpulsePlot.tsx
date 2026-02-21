import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { TargetResponse, FirModelResult, FirConfig, PeqBand } from "../lib/types";
import {
  activeBand,
  exportSampleRate,
  exportTaps,
  exportWindow,
} from "../stores/bands";

const FIR_IMPULSE_COLOR = "#38BDF8"; // light blue

export default function ExportImpulsePlot() {
  let containerRef!: HTMLDivElement;
  const chartRef: { current: uPlot | undefined } = { current: undefined };

  const [cursorTime, setCursorTime] = createSignal("—");
  const [cursorAmp, setCursorAmp] = createSignal("—");
  const [hasData, setHasData] = createSignal(false);
  const [info, setInfo] = createSignal("");

  // Track actual data range for fitData
  let dataXMin = -0.5;
  let dataXMax = 30;
  let dataYMin = -1.1;
  let dataYMax = 1.1;

  // Mutable Y scale state — updated by buttons, used by range function
  let curAmpMin = -1.1;
  let curAmpMax = 1.1;

  function getChart(): uPlot | undefined {
    return chartRef.current;
  }

  function zoomX(factor: number) {
    const c = getChart();
    if (!c) return;
    const s = c.scales["x"];
    if (!s || s.min == null || s.max == null) return;
    const center = (s.min + s.max) / 2;
    const half = ((s.max - s.min) / 2) * factor;
    c.setScale("x", { min: center - half, max: center + half });
  }

  function scrollX(direction: number) {
    const c = getChart();
    if (!c) return;
    const s = c.scales["x"];
    if (!s || s.min == null || s.max == null) return;
    const range = s.max - s.min;
    const step = range * 0.2 * direction;
    c.setScale("x", { min: s.min + step, max: s.max + step });
  }

  function zoomY(factor: number) {
    const c = getChart();
    if (!c) return;
    const center = (curAmpMin + curAmpMax) / 2;
    const half = ((curAmpMax - curAmpMin) / 2) * factor;
    curAmpMin = center - half;
    curAmpMax = center + half;
    c.setScale("amp", { min: curAmpMin, max: curAmpMax });
  }

  function scrollY(direction: number) {
    const c = getChart();
    if (!c) return;
    const range = curAmpMax - curAmpMin;
    const step = range * 0.2 * direction;
    curAmpMin += step;
    curAmpMax += step;
    c.setScale("amp", { min: curAmpMin, max: curAmpMax });
  }

  function fitData() {
    const c = getChart();
    if (!c) return;
    c.setScale("x", { min: dataXMin, max: dataXMax });
    curAmpMin = dataYMin;
    curAmpMax = dataYMax;
    c.setScale("amp", { min: curAmpMin, max: curAmpMax });
  }

  function renderChart(time: number[], impulse: number[], normDb: number, phaseLabel?: string, resetScales: boolean = false) {
    if (!containerRef) return;

    // Save current scales (only when not resetting due to config change)
    let savedXMin: number | null = null;
    let savedXMax: number | null = null;
    let savedYMin: number | null = null;
    let savedYMax: number | null = null;
    if (chartRef.current && !resetScales) {
      const xs = chartRef.current.scales["x"];
      if (xs?.min != null && xs?.max != null) {
        savedXMin = xs.min;
        savedXMax = xs.max;
      }
      const ys = chartRef.current.scales["amp"];
      if (ys?.min != null && ys?.max != null) {
        savedYMin = ys.min;
        savedYMax = ys.max;
      }
    }
    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = undefined;
    }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 200);
    const h = Math.max(rect.height, 80);

    // Find peak position and normalize
    let peak = 0;
    let peakIdx = 0;
    for (let i = 0; i < impulse.length; i++) {
      const a = Math.abs(impulse[i]);
      if (a > peak) { peak = a; peakIdx = i; }
    }
    if (peak < 1e-20) peak = 1;
    const norm = impulse.map((v) => v / peak);

    // Find significant range: first and last sample above -60dB
    const threshold = peak * 0.001; // -60dB
    let firstSig = peakIdx;
    let lastSig = peakIdx;
    for (let i = 0; i < impulse.length; i++) {
      if (Math.abs(impulse[i]) > threshold) {
        firstSig = i;
        break;
      }
    }
    for (let i = impulse.length - 1; i >= 0; i--) {
      if (Math.abs(impulse[i]) > threshold) {
        lastSig = i;
        break;
      }
    }

    // Show window: peak at ~25% from left edge, pre-ringing always visible
    const sigLen = lastSig - firstSig;
    const viewLen = Math.max(sigLen * 1.3, 256); // minimum 256 samples
    const prePeak = Math.ceil(viewLen * 0.25); // 25% before peak
    const startIdx = Math.max(0, peakIdx - prePeak);
    const endIdx = Math.min(impulse.length, startIdx + Math.ceil(viewLen));

    const trimTime = time.slice(startIdx, endIdx);
    const trimNorm = norm.slice(startIdx, endIdx);

    const uData: uPlot.AlignedData = [trimTime, trimNorm];

    // Compute natural fit ranges from trimmed data
    const fitXMin = trimTime[0] - 0.5;
    const fitXMax = trimTime[trimTime.length - 1] + 0.5;
    let normMin = 0, normMax = 0;
    for (const v of trimNorm) { if (v < normMin) normMin = v; if (v > normMax) normMax = v; }
    const fitYPad = Math.max(0.1, (normMax - normMin) * 0.1);
    const fitYMin = normMin - fitYPad;
    const fitYMax = normMax + fitYPad;

    // Store for fitData button
    dataXMin = fitXMin;
    dataXMax = fitXMax;
    dataYMin = fitYMin;
    dataYMax = fitYMax;

    // Check if saved scales are within trimmed data range (otherwise discard)
    const savedXValid = savedXMin != null && savedXMax != null &&
      savedXMax > fitXMin && savedXMin < fitXMax;
    const xMin = savedXValid ? savedXMin! : fitXMin;
    const xMax = savedXValid ? savedXMax! : fitXMax;
    const yMin = savedYMin ?? fitYMin;
    const yMax = savedYMax ?? fitYMax;
    curAmpMin = yMin;
    curAmpMax = yMax;

    const opts: uPlot.Options = {
      width: w,
      height: h,
      series: [
        {},
        {
          label: "Impulse",
          stroke: FIR_IMPULSE_COLOR,
          width: 1.5,
          scale: "amp",
        },
      ],
      scales: {
        x: { min: xMin, max: xMax },
        amp: {
          auto: false,
          range: () => [curAmpMin, curAmpMax] as uPlot.Range.MinMax,
        },
      },
      axes: [
        {
          label: "ms",
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(1))),
        },
        {
          label: "Amp",
          scale: "amp",
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(2))),
          size: 56,
        },
      ],
      legend: { show: false },
      cursor: {
        drag: { x: false, y: false, setScale: false },
      },
      hooks: {
        setCursor: [
          (u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= u.data[0].length) {
              setCursorTime("—");
              setCursorAmp("—");
              return;
            }
            const t = u.data[0][idx];
            setCursorTime(t != null ? t.toFixed(3) + " ms" : "—");
            const v = u.data[1]?.[idx];
            setCursorAmp(v != null ? (v as number).toFixed(4) : "—");
          },
        ],
      },
    };

    try {
      chartRef.current = new uPlot(opts, uData, containerRef);
    } catch (e) {
      console.error("ExportImpulsePlot uPlot error:", e);
    }

    const pLabel = phaseLabel ? ` · ${phaseLabel}` : "";
    const normLabel = normDb !== 0 ? ` · Norm: ${normDb > 0 ? "−" : "+"}${Math.abs(normDb).toFixed(1)} dB` : "";
    setInfo(`${impulse.length} samples${pLabel}${normLabel}`);
  }

  onMount(() => {
    const observer = new ResizeObserver(() => {
      if (chartRef.current && containerRef) {
        const rect = containerRef.getBoundingClientRect();
        chartRef.current.setSize({
          width: Math.max(rect.width, 200),
          height: Math.max(rect.height, 80),
        });
      }
    });
    observer.observe(containerRef);
    onCleanup(() => observer.disconnect());
  });

  // Track previous FIR params to detect config changes (taps/SR/window)
  let prevTaps = 0;
  let prevSR = 0;
  let prevWin = "";

  // Main effect: compute target + PEQ + FIR impulse
  createEffect(() => {
    const band = activeBand();
    if (!band || !band.target) {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = undefined;
      }
      setHasData(false);
      return;
    }

    const sr = exportSampleRate();
    const taps = exportTaps();
    const win = exportWindow();
    const target = { ...band.target };
    const peqBands = band.peqBands?.filter((b: PeqBand) => b.enabled) ?? [];

    // Reset scales when FIR config changes (taps, SR, window)
    const configChanged = taps !== prevTaps || sr !== prevSR || win !== prevWin;
    prevTaps = taps;
    prevSR = sr;
    prevWin = win;

    computeAndRender(target, peqBands, sr, taps, win, configChanged);
  });

  async function computeAndRender(
    target: any,
    peqBands: PeqBand[],
    sampleRate: number,
    taps: number,
    window: string,
    resetScales: boolean = false,
  ) {
    try {
      // 1. Evaluate pure target
      const [freq, response] = await invoke<[number[], TargetResponse]>(
        "evaluate_target_standalone",
        { target, nPoints: 512, fMin: 5, fMax: 40000 },
      );

      const targetMag = response.magnitude;
      let modelPhase = response.phase;

      // 2. Compute PEQ contribution separately (PEQ always min-phase)
      let peqMagArr: number[] = [];
      if (peqBands.length > 0) {
        const [peqMag, peqPhase] = await invoke<[number[], number[]]>("compute_peq_complex", {
          freq,
          bands: peqBands,
          sampleRate,
        });
        peqMagArr = peqMag;
        modelPhase = modelPhase.map((v, i) => v + peqPhase[i]);
      }

      // 3. Generate FIR: target phase by HP/LP flags, PEQ phase always min-phase
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
      const phaseLabel = allLinear ? "Linear-Phase" : "Min-Phase";
      requestAnimationFrame(() =>
        renderChart(firResult.time_ms, firResult.impulse, firResult.norm_db, phaseLabel, resetScales),
      );
    } catch (e) {
      console.error("ExportImpulsePlot compute failed:", e);
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
          <span class="readout-label">Time:</span>
          <span class="readout-value">{cursorTime()}</span>
        </span>
        <span class="readout-item">
          <span class="readout-label" style={{ color: FIR_IMPULSE_COLOR }}>
            Amp:
          </span>
          <span class="readout-value">{cursorAmp()}</span>
        </span>
        <span class="impulse-sep" />
        <span
          style={{
            "font-size": "10px",
            color: "#8b8b96",
          }}
        >
          {info() || "FIR Impulse Response"}
        </span>
      </div>
      <div class="impulse-body">
        <div class="axis-controls axis-controls-y">
          <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In Amp">+</button>
          <button class="axis-btn" onClick={() => scrollY(1)} title="Scroll Up">▲</button>
          <button class="axis-btn" onClick={() => scrollY(-1)} title="Scroll Down">▼</button>
          <button class="axis-btn" onClick={() => zoomY(1.6)} title="Zoom Out Amp">-</button>
          <button class="axis-btn fit-btn" onClick={fitData} title="Fit">FIT</button>
        </div>
        <div class="impulse-center">
          <div ref={containerRef} class="impulse-plot" />
          {!hasData() && (
            <div class="impulse-empty-overlay">No impulse data</div>
          )}
          <div class="axis-controls axis-controls-x">
            <button
              class="axis-btn"
              onClick={() => zoomX(0.6)}
              title="Zoom In Time"
            >
              +
            </button>
            <button
              class="axis-btn"
              onClick={() => scrollX(-1)}
              title="Scroll Left"
            >
              ◀
            </button>
            <button
              class="axis-btn"
              onClick={() => scrollX(1)}
              title="Scroll Right"
            >
              ▶
            </button>
            <button
              class="axis-btn"
              onClick={() => zoomX(1.6)}
              title="Zoom Out Time"
            >
              -
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
