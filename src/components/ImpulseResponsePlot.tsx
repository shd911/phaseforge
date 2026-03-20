import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { ImpulseResult, TargetResponse } from "../lib/types";
import { activeBand, appState, isSum, type BandState } from "../stores/bands";

type ViewMode = "impulse" | "step" | "gd";

const IR_COLOR = "#4A9EFF";
const STEP_COLOR = "#22C55E";
const GD_COLOR = "#F59E0B"; // amber
const TARGET_IR_COLOR = "#FFD700";
const TARGET_STEP_COLOR = "#FFD700";
const TARGET_GD_COLOR = "#FFD700";

/** Compute group delay from frequency and phase arrays.
 *  τ(f) = -(1/360) · dφ/df  (seconds → milliseconds) */
function computeGroupDelay(freq: number[], phaseDeg: number[]): { freqOut: number[]; gdMs: number[] } {
  const n = freq.length;
  if (n < 2) return { freqOut: [], gdMs: [] };
  const gd: number[] = new Array(n);
  // Forward diff for first point
  gd[0] = -(phaseDeg[1] - phaseDeg[0]) / (360 * (freq[1] - freq[0])) * 1000;
  // Central diff for interior
  for (let i = 1; i < n - 1; i++) {
    const df = freq[i + 1] - freq[i - 1];
    gd[i] = df > 0 ? -(phaseDeg[i + 1] - phaseDeg[i - 1]) / (360 * df) * 1000 : 0;
  }
  // Backward diff for last
  gd[n - 1] = -(phaseDeg[n - 1] - phaseDeg[n - 2]) / (360 * (freq[n - 1] - freq[n - 2])) * 1000;
  return { freqOut: freq, gdMs: gd };
}

export default function ImpulseResponsePlot() {
  let containerRef!: HTMLDivElement;

  const chartRef: { current: uPlot | undefined } = { current: undefined };

  // Данные для FIT и навигации
  let dataTimeMin = -1;   // мин. время данных (мс), может быть < 0
  let dataTimeMax = 50;   // макс. время данных (мс)
  let dataAmpMin = -110;
  let dataAmpMax = 110;
  let peakTimeMs = 0;     // время пика (мс) — якорь для зума

  // Mutable Y-scale state — updated by buttons, used by range function
  let curAmpMin = -110;
  let curAmpMax = 110;

  const [viewMode, setViewMode] = createSignal<ViewMode>("impulse");
  const [cursorTime, setCursorTime] = createSignal("—");
  const [cursorAmp, setCursorAmp] = createSignal("—");
  const [hasData, setHasData] = createSignal(false);

  function getChart(): uPlot | undefined {
    return chartRef.current;
  }

  // Вычислить xMin/xMax так, чтобы пик был на ~25% от левого края
  function computeXWithPeakAt25(visibleRange: number): { min: number; max: number } {
    const xMin = peakTimeMs - visibleRange * 0.25;
    const xMax = xMin + visibleRange;
    return { min: xMin, max: xMax };
  }

  function zoomX(factor: number) {
    const c = getChart();
    if (!c) return;
    const s = c.scales["x"];
    if (!s || s.min == null || s.max == null) return;
    const curRange = s.max - s.min;
    const newRange = curRange * factor;
    const { min, max } = computeXWithPeakAt25(newRange);
    c.setScale("x", { min, max });
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

  function fitData() {
    const c = getChart();
    if (!c) return;
    const totalRange = dataTimeMax - dataTimeMin;
    const { min, max } = computeXWithPeakAt25(totalRange);
    c.setScale("x", { min, max });
    curAmpMin = dataAmpMin;
    curAmpMax = dataAmpMax;
    c.setScale("amp", { min: curAmpMin, max: curAmpMax });
  }

  /** Интерполировать targetData на временную сетку masterTime.
   *  Оба массива имеют одинаковый dt, но разный offset и длину. */
  function alignToMasterTime(
    masterTime: number[], targetTime: number[], targetData: number[],
  ): (number | null)[] {
    if (targetTime.length === 0 || masterTime.length === 0) {
      return masterTime.map(() => null);
    }
    const dt = masterTime.length > 1 ? masterTime[1] - masterTime[0] : 1;
    const halfDt = dt * 0.6; // допуск для совпадения
    const tStart = targetTime[0];

    const result: (number | null)[] = new Array(masterTime.length);
    for (let i = 0; i < masterTime.length; i++) {
      const t = masterTime[i];
      // Индекс в target: (t - tStart) / dt
      const fi = (t - tStart) / dt;
      const idx = Math.round(fi);
      if (idx >= 0 && idx < targetTime.length && Math.abs(targetTime[idx] - t) < halfDt) {
        result[i] = targetData[idx];
      } else {
        result[i] = null;
      }
    }
    return result;
  }

  function renderChart(
    time: number[], data: number[], label: string, color: string,
    targetTime?: number[], targetData?: number[],
  ) {
    if (!containerRef) return;

    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = undefined;
    }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 200);
    const h = Math.max(rect.height, 80);

    // Конвертируем время в миллисекунды
    const timeMs = time.map((t) => t * 1000);

    // Найти пик (максимум абсолютного значения) — по measurement данным
    let peakIdx = 0;
    let peakAbs = 0;
    for (let i = 0; i < data.length; i++) {
      const abs = Math.abs(data[i]);
      if (abs > peakAbs) { peakAbs = abs; peakIdx = i; }
    }
    peakTimeMs = timeMs.length > 0 ? timeMs[peakIdx] : 0;

    // Подготовить target серию (если есть) — needed before Y range calc
    const hasTarget = targetTime && targetData && targetTime.length > 0;

    // Вычислить FIT-диапазоны из данных (including target if present)
    let ampMin = Infinity;
    let ampMax = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < ampMin) ampMin = data[i];
      if (data[i] > ampMax) ampMax = data[i];
    }
    if (hasTarget) {
      for (let i = 0; i < targetData!.length; i++) {
        if (targetData![i] < ampMin) ampMin = targetData![i];
        if (targetData![i] > ampMax) ampMax = targetData![i];
      }
    }
    const ampPad = Math.max(5, (ampMax - ampMin) * 0.05);
    dataTimeMin = timeMs.length > 0 ? timeMs[0] : -1;
    dataTimeMax = timeMs.length > 0 ? timeMs[timeMs.length - 1] : 50;
    dataAmpMin = isFinite(ampMin) ? ampMin - ampPad : -110;
    dataAmpMax = isFinite(ampMax) ? ampMax + ampPad : 110;
    let targetAligned: (number | null)[] | undefined;
    if (hasTarget) {
      const targetTimeMs = targetTime!.map((t) => t * 1000);
      targetAligned = alignToMasterTime(timeMs, targetTimeMs, targetData!);
    }

    // uPlot data
    const uData: uPlot.AlignedData = hasTarget
      ? [timeMs, data, targetAligned!]
      : [timeMs, data];

    // Series
    const targetColor = label === "Impulse" ? TARGET_IR_COLOR : TARGET_STEP_COLOR;
    const series: uPlot.Series[] = [
      {},
      { label, stroke: color, width: 1.5, scale: "amp" },
    ];
    if (hasTarget) {
      series.push({
        label: "Target " + label,
        stroke: targetColor,
        width: 1.5,
        scale: "amp",
        dash: [6, 4],
      });
    }

    // X: пик на 25% от левого края
    const totalRange = dataTimeMax - dataTimeMin;
    const fit = computeXWithPeakAt25(totalRange);
    const xMin = fit.min;
    const xMax = fit.max;
    const yMin = dataAmpMin;
    const yMax = dataAmpMax;
    curAmpMin = yMin;
    curAmpMax = yMax;

    const opts: uPlot.Options = {
      width: w,
      height: h,
      series,
      scales: {
        x: { min: xMin, max: xMax },
        amp: { auto: false, range: () => [curAmpMin, curAmpMax] as uPlot.Range.MinMax },
      },
      axes: [
        {
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(1) + " ms")),
        },
        {
          label: "%",
          scale: "amp",
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(0) + "%")),
          size: 50,
        },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
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
            const a = u.data[1]?.[idx];
            setCursorAmp(a != null ? a.toFixed(1) + "%" : "—");
          },
        ],
      },
    };

    try {
      chartRef.current = new uPlot(opts, uData, containerRef);
      // Auto-fit on new data
      fitData();
    } catch (e) {
      console.error("Impulse uPlot error:", e);
    }
  }

  function renderGD(freq: number[], gdMs: number[]) {
    if (!containerRef) return;
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = undefined; }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 200);
    const h = Math.max(rect.height, 80);

    // Y range from data
    let yMin = Infinity, yMax = -Infinity;
    for (const v of gdMs) {
      if (isFinite(v)) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    }
    const pad = Math.max(0.5, (yMax - yMin) * 0.1);
    yMin = isFinite(yMin) ? yMin - pad : -5;
    yMax = isFinite(yMax) ? yMax + pad : 20;
    curAmpMin = yMin; curAmpMax = yMax;
    dataAmpMin = yMin; dataAmpMax = yMax;

    const opts: uPlot.Options = {
      width: w, height: h,
      series: [
        {},
        { label: "GD ms", stroke: GD_COLOR, width: 1.5, scale: "amp" },
      ],
      scales: {
        x: { min: 20, max: 20000, distr: 3 },
        amp: { auto: false, range: () => [curAmpMin, curAmpMax] as uPlot.Range.MinMax },
      },
      axes: [
        {
          stroke: "#9b9ba6",
          grid: { stroke: "rgba(255,255,255,0.12)" },
          ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map(v => v == null ? "" : v >= 1000 ? (v/1000) + "k" : String(Math.round(v))),
        },
        {
          label: "ms", scale: "amp", stroke: "#9b9ba6",
          grid: { stroke: "rgba(255,255,255,0.12)" },
          ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)),
          size: 50,
        },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
      hooks: {
        setCursor: [
          (u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= u.data[0].length) {
              setCursorTime("—"); setCursorAmp("—"); return;
            }
            const f = u.data[0][idx];
            setCursorTime(f != null ? (f >= 1000 ? (f/1000).toFixed(2) + " kHz" : Math.round(f) + " Hz") : "—");
            const gd = u.data[1]?.[idx];
            setCursorAmp(gd != null ? (gd as number).toFixed(2) + " ms" : "—");
          },
        ],
      },
    };

    try {
      chartRef.current = new uPlot(opts, [freq, gdMs], containerRef);
      setHasData(true);
    } catch (e) {
      console.error("GD uPlot error:", e);
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

  // Главный реактивный эффект
  createEffect(() => {
    const sumMode = isSum();
    const band = activeBand();
    const mode = viewMode();

    if (sumMode) {
      // SUM mode: coherent sum of all bands with phase data
      const bands = appState.bands.filter(b => b.measurement?.phase);
      if (bands.length === 0) {
        if (chartRef.current) { chartRef.current.destroy(); chartRef.current = undefined; }
        setHasData(false);
        return;
      }
      computeSumAndRender(bands, mode);
      return;
    }

    if (!band || !band.measurement || !band.measurement.phase) {
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = undefined; }
      setHasData(false);
      setCursorTime("—");
      setCursorAmp("—");
      return;
    }

    const freq = [...band.measurement.freq];
    const magnitude = [...band.measurement.magnitude];
    const phase = [...band.measurement.phase];
    const sr = band.measurement.sample_rate ?? 48000;

    // Target: берём если включён
    const targetEnabled = band.targetEnabled;
    const targetCurve = targetEnabled ? JSON.parse(JSON.stringify(band.target)) : null;

    computeAndRender(freq, magnitude, phase, sr, mode, targetCurve);
  });

  async function computeAndRender(
    freq: number[], magnitude: number[], phase: number[],
    sampleRate: number, mode: ViewMode,
    targetCurve: any | null,
  ) {
    try {
      // Measurement IR
      const measResult = await invoke<ImpulseResult>("compute_impulse", {
        freq, magnitude, phase, sampleRate,
      });

      // Target IR (если есть таргет)
      let targetResult: ImpulseResult | null = null;
      if (targetCurve) {
        // Auto-reference: средний уровень measurement в 200–2000 Hz
        let sum = 0, n = 0;
        for (let i = 0; i < freq.length; i++) {
          if (freq[i] >= 200 && freq[i] <= 2000) { sum += magnitude[i]; n++; }
        }
        const autoRef = n > 0 ? sum / n : 0;
        const curveWithRef = {
          ...targetCurve,
          reference_level_db: targetCurve.reference_level_db + autoRef,
        };

        // Evaluate target → magnitude + phase
        const targetResp = await invoke<TargetResponse>("evaluate_target", {
          target: curveWithRef, freq,
        });

        // Compute target impulse/step
        targetResult = await invoke<ImpulseResult>("compute_impulse", {
          freq,
          magnitude: targetResp.magnitude,
          phase: targetResp.phase,
          sampleRate,
        });
      }

      setHasData(true);

      const doRender = () => {
        if (mode === "impulse") {
          renderChart(
            measResult.time, measResult.impulse, "Impulse", IR_COLOR,
            targetResult?.time, targetResult?.impulse,
          );
        } else if (mode === "step") {
          renderChart(
            measResult.time, measResult.step, "Step", STEP_COLOR,
            targetResult?.time, targetResult?.step,
          );
        } else {
          // Group delay — compute from unwrapped phase
          const { freqOut, gdMs } = computeGroupDelay(freq, phase);
          renderGD(freqOut, gdMs);
        }
      };

      requestAnimationFrame(doRender);
    } catch (e) {
      console.error("Impulse computation failed:", e);
      setHasData(false);
    }
  }

  async function computeSumAndRender(bands: BandState[], mode: ViewMode) {
    try {
      // Build common frequency grid from first band
      const refBand = bands[0];
      const freq = [...refBand.measurement!.freq];
      const n = freq.length;

      // Coherent sum in complex domain
      const sumRe = new Float64Array(n);
      const sumIm = new Float64Array(n);

      for (const b of bands) {
        const m = b.measurement!;
        // Resample if different frequency grid
        const bFreq = m.freq;
        const bMag = m.magnitude;
        const bPhase = m.phase!;
        const sign = b.inverted ? -1 : 1;

        for (let j = 0; j < n; j++) {
          // Simple: assume all bands share freq grid (from merge or same source)
          // If not, use nearest point
          let mag = bMag[j] ?? -200;
          let ph = bPhase[j] ?? 0;
          if (bFreq.length !== n) {
            // Find nearest frequency
            let best = 0;
            let bestDist = Math.abs(bFreq[0] - freq[j]);
            for (let k = 1; k < bFreq.length; k++) {
              const d = Math.abs(bFreq[k] - freq[j]);
              if (d < bestDist) { bestDist = d; best = k; }
              if (bFreq[k] > freq[j]) break;
            }
            mag = bMag[best];
            ph = bPhase[best] ?? 0;
          }
          const amp = Math.pow(10, mag / 20) * sign;
          const phRad = ph * Math.PI / 180;
          sumRe[j] += amp * Math.cos(phRad);
          sumIm[j] += amp * Math.sin(phRad);
        }
      }

      // Convert sum to magnitude + phase
      const sumMag = new Array(n);
      const sumPhase = new Array(n);
      for (let j = 0; j < n; j++) {
        const amp = Math.sqrt(sumRe[j] * sumRe[j] + sumIm[j] * sumIm[j]);
        sumMag[j] = amp > 0 ? 20 * Math.log10(amp) : -200;
        sumPhase[j] = Math.atan2(sumIm[j], sumRe[j]) * 180 / Math.PI;
      }

      const sr = refBand.measurement!.sample_rate ?? 48000;

      if (mode === "gd") {
        // Unwrap sum phase for GD
        const unwrapped: number[] = [sumPhase[0]];
        for (let i = 1; i < n; i++) {
          let diff = sumPhase[i] - sumPhase[i - 1];
          while (diff > 180) diff -= 360;
          while (diff <= -180) diff += 360;
          unwrapped.push(unwrapped[i - 1] + diff);
        }
        const { freqOut, gdMs } = computeGroupDelay(freq, unwrapped);
        requestAnimationFrame(() => renderGD(freqOut, gdMs));
      } else {
        // IR or Step from sum spectrum
        const result = await invoke<ImpulseResult>("compute_impulse", {
          freq, magnitude: sumMag, phase: sumPhase, sampleRate: sr,
        });
        setHasData(true);
        requestAnimationFrame(() => {
          if (mode === "impulse") {
            renderChart(result.time, result.impulse, "Σ Impulse", IR_COLOR);
          } else {
            renderChart(result.time, result.step, "Σ Step", STEP_COLOR);
          }
        });
      }
    } catch (e) {
      console.error("SUM impulse computation failed:", e);
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
          <span class="readout-label">Amp:</span>
          <span class="readout-value">{cursorAmp()}</span>
        </span>
        <span class="impulse-sep" />
        <button
          class={`impulse-mode-btn ${viewMode() === "impulse" ? "active" : ""}`}
          onClick={() => setViewMode("impulse")}
        >
          IR
        </button>
        <button
          class={`impulse-mode-btn ${viewMode() === "step" ? "active" : ""}`}
          onClick={() => setViewMode("step")}
        >
          Step
        </button>
        <button
          class={`impulse-mode-btn ${viewMode() === "gd" ? "active" : ""}`}
          onClick={() => setViewMode("gd")}
        >
          GD
        </button>
      </div>
      <div class="impulse-body">
        <div class="axis-controls axis-controls-y">
          <button class="axis-btn" onClick={() => zoomY(0.6)} title="Zoom In Amp">+</button>
          <button class="axis-btn" onClick={() => zoomY(1.6)} title="Zoom Out Amp">−</button>
          <button class="axis-btn fit-btn" onClick={fitData} title="Fit">FIT</button>
        </div>
        <div class="impulse-center">
          <div ref={containerRef} class="impulse-plot" />
          {!hasData() && (
            <div class="impulse-empty-overlay">No phase data — IR unavailable</div>
          )}
          <div class="axis-controls axis-controls-x">
            <button class="axis-btn" onClick={() => zoomX(0.6)} title="Zoom In Time">+</button>
            <button class="axis-btn" onClick={() => scrollX(-1)} title="Scroll Left">◀</button>
            <button class="axis-btn" onClick={() => scrollX(1)} title="Scroll Right">▶</button>
            <button class="axis-btn" onClick={() => zoomX(1.6)} title="Zoom Out Time">−</button>
          </div>
        </div>
      </div>
    </div>
  );
}
