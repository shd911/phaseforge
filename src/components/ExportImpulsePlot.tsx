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
  firIterations,
  firFreqWeighting,
  firNarrowbandLimit,
  firNbSmoothingOct,
  firNbMaxExcess,
  firMaxBoost,
  firNoiseFloor,
} from "../stores/bands";

const FIR_IMPULSE_COLOR = "#38BDF8"; // light blue
const CORRECTED_IMPULSE_COLOR = "#4ade80"; // green — measurement + FIR
const MASKING_ZONE_COLOR = "rgba(34, 197, 94, 0.08)";
const MASKING_BORDER_COLOR = "rgba(34, 197, 94, 0.25)";
const PRE_RING_ZONE_COLOR = "rgba(239, 68, 68, 0.08)";

export default function ExportImpulsePlot() {
  let containerRef!: HTMLDivElement;
  const chartRef: { current: uPlot | undefined } = { current: undefined };

  const [cursorTime, setCursorTime] = createSignal("—");
  const [cursorAmp, setCursorAmp] = createSignal("—");
  const [hasData, setHasData] = createSignal(false);
  const [info, setInfo] = createSignal("");
  const [showFir, setShowFir] = createSignal(true);
  const [showCorrected, setShowCorrected] = createSignal(true);
  const [showMasking, setShowMasking] = createSignal(true);

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

  // Masking duration in ms for pre-ringing based on HP frequency
  let maskingMs = 0;

  function renderChart(
    time: number[], impulse: number[],
    correctedTime: number[] | null, correctedImpulse: number[] | null,
    hpFreq: number,
    normDb: number, phaseLabel?: string, resetScales: boolean = false,
  ) {
    if (!containerRef) return;

    // Compute masking duration: ~1.5 periods of HP cutoff frequency
    maskingMs = hpFreq > 0 ? (1.5 / hpFreq) * 1000 : 20;

    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = undefined;
    }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 200);
    const h = Math.max(rect.height, 80);

    // Normalize FIR impulse
    let peak = 0, peakIdx = 0;
    for (let i = 0; i < impulse.length; i++) {
      const a = Math.abs(impulse[i]);
      if (a > peak) { peak = a; peakIdx = i; }
    }
    if (peak < 1e-20) peak = 1;
    const normFir = impulse.map(v => v / peak);
    const peakTimeMs = time[peakIdx];

    // Normalize corrected impulse (if present) — always compute, toggle via show flag
    let normCorr: number[] | null = null;
    if (correctedTime && correctedImpulse) {
      let cPeak = 0;
      for (const v of correctedImpulse) { const a = Math.abs(v); if (a > cPeak) cPeak = a; }
      if (cPeak < 1e-20) cPeak = 1;
      normCorr = correctedImpulse.map(v => v / cPeak);
    }

    // Find significant range
    const threshold = peak * 0.001;
    let firstSig = peakIdx, lastSig = peakIdx;
    for (let i = 0; i < impulse.length; i++) { if (Math.abs(impulse[i]) > threshold) { firstSig = i; break; } }
    for (let i = impulse.length - 1; i >= 0; i--) { if (Math.abs(impulse[i]) > threshold) { lastSig = i; break; } }

    const sigLen = lastSig - firstSig;
    const viewLen = Math.max(sigLen * 1.3, 256);
    const prePeak = Math.ceil(viewLen * 0.25);
    const startIdx = Math.max(0, peakIdx - prePeak);
    const endIdx = Math.min(impulse.length, startIdx + Math.ceil(viewLen));

    const trimTime = time.slice(startIdx, endIdx);
    const trimFir = normFir.slice(startIdx, endIdx);

    // Build series & data
    const uSeries: uPlot.Series[] = [{}];
    const uDataArr: (number | null)[][] = [trimTime];

    // FIR impulse
    uSeries.push({
      label: "FIR",
      stroke: FIR_IMPULSE_COLOR,
      width: 1.5,
      scale: "amp",
      show: showFir(),
    });
    uDataArr.push(trimFir);

    // Corrected impulse (meas+FIR) — align peaks, then resample onto FIR time grid
    if (normCorr && correctedTime) {
      // Find peak of corrected impulse
      let cPeakIdx = 0, cPeakVal = 0;
      for (let i = 0; i < normCorr.length; i++) {
        if (Math.abs(normCorr[i]) > cPeakVal) { cPeakVal = Math.abs(normCorr[i]); cPeakIdx = i; }
      }
      const cPeakTime = correctedTime[cPeakIdx];
      // Shift: align corrected peak to FIR peak
      const timeShift = peakTimeMs - cPeakTime;
      const shiftedTime = correctedTime.map(t => t + timeShift);

      // Resample shifted corrected onto FIR time grid
      const trimCorr = trimTime.map(t => {
        if (t <= shiftedTime[0]) return 0;
        if (t >= shiftedTime[shiftedTime.length - 1]) return 0;
        let lo = 0, hi = shiftedTime.length - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (shiftedTime[mid] <= t) lo = mid; else hi = mid;
        }
        const frac = (t - shiftedTime[lo]) / (shiftedTime[hi] - shiftedTime[lo]);
        return normCorr![lo] + frac * (normCorr![hi] - normCorr![lo]);
      });
      uSeries.push({
        label: "Corrected",
        stroke: CORRECTED_IMPULSE_COLOR,
        width: 1.5,
        scale: "amp",
        show: showCorrected(),
      });
      uDataArr.push(trimCorr);
    }

    const uData = uDataArr as uPlot.AlignedData;

    // Fit ranges
    let yMin = 0, yMax = 0;
    for (const arr of uDataArr.slice(1)) {
      for (const v of arr) { if (v != null) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; } }
    }
    const fitYPad = Math.max(0.1, (yMax - yMin) * 0.1);
    dataXMin = trimTime[0] - 0.5;
    dataXMax = trimTime[trimTime.length - 1] + 0.5;
    dataYMin = yMin - fitYPad;
    dataYMax = yMax + fitYPad;
    curAmpMin = resetScales ? dataYMin : curAmpMin || dataYMin;
    curAmpMax = resetScales ? dataYMax : curAmpMax || dataYMax;

    const opts: uPlot.Options = {
      width: w, height: h,
      series: uSeries,
      scales: {
        x: { min: dataXMin, max: dataXMax },
        amp: { auto: false, range: () => [curAmpMin, curAmpMax] as uPlot.Range.MinMax },
      },
      axes: [
        {
          label: "ms", stroke: "#9b9ba6",
          grid: { stroke: "rgba(255,255,255,0.12)" },
          ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(1)),
        },
        {
          label: "Amp", scale: "amp", stroke: "#9b9ba6",
          grid: { stroke: "rgba(255,255,255,0.12)" },
          ticks: { stroke: "rgba(255,255,255,0.20)" },
          values: (_u: uPlot, vals: number[]) => vals.map(v => v == null ? "" : v.toFixed(2)),
          size: 56,
        },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
      hooks: {
        draw: [
          // Pre-ringing masking zone overlay
          (u: uPlot) => {
            if (!showMasking()) return;
            const ctx = u.ctx;
            const plotLeft = u.bbox.left;
            const plotTop = u.bbox.top;
            const plotHeight = u.bbox.height;

            // Peak position
            const peakX = u.valToPos(peakTimeMs, "x", true);
            // Masking zone: peakTime - maskingMs .. peakTime
            const maskStartX = u.valToPos(peakTimeMs - maskingMs, "x", true);
            // Pre-ring danger zone: everything before masking zone
            const plotLeftEdge = plotLeft;

            ctx.save();
            // Green masking zone (safe pre-ringing)
            if (maskStartX > plotLeftEdge) {
              ctx.fillStyle = MASKING_ZONE_COLOR;
              ctx.fillRect(Math.max(maskStartX, plotLeftEdge), plotTop, peakX - Math.max(maskStartX, plotLeftEdge), plotHeight);
              // Border
              ctx.strokeStyle = MASKING_BORDER_COLOR;
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 4]);
              ctx.beginPath();
              ctx.moveTo(maskStartX, plotTop);
              ctx.lineTo(maskStartX, plotTop + plotHeight);
              ctx.stroke();
              ctx.setLineDash([]);
            }
            // Red danger zone (audible pre-ringing)
            if (maskStartX > plotLeftEdge + 2) {
              ctx.fillStyle = PRE_RING_ZONE_COLOR;
              ctx.fillRect(plotLeftEdge, plotTop, maskStartX - plotLeftEdge, plotHeight);
            }
            ctx.restore();
          },
        ],
        setCursor: [
          (u: uPlot) => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0 || idx >= u.data[0].length) {
              setCursorTime("—"); setCursorAmp("—"); return;
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
    const mLabel = ` · Mask: ${maskingMs.toFixed(1)}ms`;
    setInfo(`${impulse.length} taps${pLabel}${mLabel}`);
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

  // Redraw when visibility toggles change
  createEffect(() => {
    showFir(); showCorrected(); showMasking(); // track
    const c = chartRef.current;
    if (c) {
      // Toggle series visibility
      if (c.series[1]) c.setSeries(1, { show: showFir() });
      if (c.series[2]) c.setSeries(2, { show: showCorrected() });
      c.redraw(false, false);
    }
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
    // Track FIR optimization signals synchronously
    const firOpts = {
      maxBoost: firMaxBoost(),
      noiseFloor: firNoiseFloor(),
      iterations: firIterations(),
      freqWeighting: firFreqWeighting(),
      narrowbandLimit: firNarrowbandLimit(),
      nbSmoothingOct: firNbSmoothingOct(),
      nbMaxExcess: firNbMaxExcess(),
    };
    const target = { ...band.target };
    const peqBands = band.peqBands?.filter((b: PeqBand) => b.enabled) ?? [];
    // Snapshot measurement data synchronously (won't be tracked inside async)
    const measSnap = band.measurement?.phase ? {
      freq: [...band.measurement.freq],
      magnitude: [...band.measurement.magnitude],
      phase: [...band.measurement.phase],
    } : null;

    // Reset scales when FIR config changes (taps, SR, window)
    const configChanged = taps !== prevTaps || sr !== prevSR || win !== prevWin;
    prevTaps = taps;
    prevSR = sr;
    prevWin = win;

    computeAndRender(target, peqBands, sr, taps, win, configChanged, firOpts, measSnap);
  });

  async function computeAndRender(
    target: any,
    peqBands: PeqBand[],
    sampleRate: number,
    taps: number,
    window: string,
    resetScales: boolean = false,
    firOpts: { maxBoost: number; noiseFloor: number; iterations: number; freqWeighting: boolean; narrowbandLimit: boolean; nbSmoothingOct: number; nbMaxExcess: number } = { maxBoost: 24, noiseFloor: -150, iterations: 3, freqWeighting: true, narrowbandLimit: true, nbSmoothingOct: 0.333, nbMaxExcess: 6 },
    measSnap: { freq: number[]; magnitude: number[]; phase: number[] } | null = null,
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
        max_boost_db: firOpts.maxBoost,
        noise_floor_db: firOpts.noiseFloor,
        window: window as any,
        phase_mode: allLinear ? "LinearPhase" : "MinimumPhase",
        iterations: firOpts.iterations,
        freq_weighting: firOpts.freqWeighting,
        narrowband_limit: firOpts.narrowbandLimit,
        nb_smoothing_oct: firOpts.nbSmoothingOct,
        nb_max_excess_db: firOpts.nbMaxExcess,
      };

      const firResult = await invoke<FirModelResult>("generate_model_fir", {
        freq,
        targetMag,
        peqMag: peqMagArr,
        modelPhase,
        config: firConfig,
      });

      // Compute corrected impulse: measurement convolved with FIR
      // In frequency domain: corrected = meas_spectrum × FIR_spectrum
      // mag(dB) = meas_mag + realized_mag,  phase = meas_phase + realized_phase
      let corrTime: number[] | null = null;
      let corrImpulse: number[] | null = null;
      if (measSnap) {
        try {
          const mFreq = measSnap.freq;
          const mMag = measSnap.magnitude;
          const mPh = measSnap.phase;
          // Interpolate measurement onto FIR freq grid (512 pts)
          const interpAt = (srcF: number[], srcD: number[], f: number): number => {
            if (srcF.length === 0) return 0;
            if (f <= srcF[0]) return srcD[0];
            if (f >= srcF[srcF.length - 1]) return srcD[srcF.length - 1];
            let lo = 0, hi = srcF.length - 1;
            while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (srcF[mid] <= f) lo = mid; else hi = mid; }
            const t = (f - srcF[lo]) / (srcF[hi] - srcF[lo]);
            return srcD[lo] + t * (srcD[hi] - srcD[lo]);
          };
          // meas + FIR realized = corrected spectrum
          const corrMag = freq.map((f, i) => interpAt(mFreq, mMag, f) + firResult.realized_mag[i]);
          const corrPh = freq.map((f, i) => interpAt(mFreq, mPh, f) + firResult.realized_phase[i]);
          const corrResult = await invoke<{ time: number[]; impulse: number[]; step: number[] }>("compute_impulse", {
            freq, magnitude: corrMag, phase: corrPh, sampleRate,
          });
          corrTime = corrResult.time.map(t => t * 1000);
          corrImpulse = corrResult.impulse;
        } catch (e) { console.warn("Corrected impulse failed:", e); }
      }

      // HP frequency for masking zone
      const hpFreq = target.high_pass?.freq_hz ?? 20;

      setHasData(true);
      const phaseLabel = allLinear ? "Linear-Phase" : "Min-Phase";
      requestAnimationFrame(() =>
        renderChart(firResult.time_ms, firResult.impulse, corrTime, corrImpulse, hpFreq, firResult.norm_db, phaseLabel, resetScales),
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
        <button
          class={`tb-btn ${showFir() ? "active" : ""}`}
          onClick={() => setShowFir(!showFir())}
          style={{ color: FIR_IMPULSE_COLOR, "font-size": "9px", padding: "1px 4px" }}
        >FIR</button>
        <button
          class={`tb-btn ${showCorrected() ? "active" : ""}`}
          onClick={() => setShowCorrected(!showCorrected())}
          style={{ color: CORRECTED_IMPULSE_COLOR, "font-size": "9px", padding: "1px 4px" }}
        >Corr</button>
        <button
          class={`tb-btn ${showMasking() ? "active" : ""}`}
          onClick={() => setShowMasking(!showMasking())}
          style={{ "font-size": "9px", padding: "1px 4px" }}
        >Mask</button>
        <span class="impulse-sep" />
        <span style={{ "font-size": "10px", color: "#8b8b96" }}>
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
