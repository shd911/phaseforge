import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { ImpulseResult, TargetResponse } from "../lib/types";
import { activeBand } from "../stores/bands";

type ViewMode = "impulse" | "step";

const IR_COLOR = "#4A9EFF";
const STEP_COLOR = "#22C55E";
const TARGET_IR_COLOR = "#FFD700";   // Gold — как на freq plot
const TARGET_STEP_COLOR = "#FFD700";

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

    // Сохраняем только Y масштаб
    let savedYMin: number | null = null;
    let savedYMax: number | null = null;

    if (chartRef.current) {
      const ys = chartRef.current.scales["amp"];
      if (ys?.min != null && ys?.max != null) { savedYMin = ys.min; savedYMax = ys.max; }
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

    // Вычислить FIT-диапазоны из данных
    let ampMin = Infinity;
    let ampMax = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < ampMin) ampMin = data[i];
      if (data[i] > ampMax) ampMax = data[i];
    }
    const ampPad = Math.max(5, (ampMax - ampMin) * 0.05);
    dataTimeMin = timeMs.length > 0 ? timeMs[0] : -1;
    dataTimeMax = timeMs.length > 0 ? timeMs[timeMs.length - 1] : 50;
    dataAmpMin = isFinite(ampMin) ? ampMin - ampPad : -110;
    dataAmpMax = isFinite(ampMax) ? ampMax + ampPad : 110;

    // Подготовить target серию (если есть)
    const hasTarget = targetTime && targetData && targetTime.length > 0;
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
    const yMin = savedYMin ?? dataAmpMin;
    const yMax = savedYMax ?? dataAmpMax;
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
    } catch (e) {
      console.error("Impulse uPlot error:", e);
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
    const band = activeBand();
    const mode = viewMode();

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
        } else {
          renderChart(
            measResult.time, measResult.step, "Step", STEP_COLOR,
            targetResult?.time, targetResult?.step,
          );
        }
      };

      requestAnimationFrame(doRender);
    } catch (e) {
      console.error("Impulse computation failed:", e);
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
