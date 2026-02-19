import { createSignal, createEffect, Show, onCleanup } from "solid-js";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { invoke } from "@tauri-apps/api/core";
import type { ImpulseResult, TargetResponse } from "../lib/types";
import {
  activeBand,
  selectedPeqIdx,
  setSelectedPeqIdx,
  addPeqBand,
  updatePeqBand,
  commitPeqBand,
  removePeqBand,
} from "../stores/bands";

const CORRECTED_COLOR = "#22C55E";
const TARGET_COLOR = "#FFD700";

// ────────────────────────────────────────────────────────────────
// PEQ Sidebar — table + mini corrected impulse plot
// ────────────────────────────────────────────────────────────────

export default function PeqSidebar() {
  const band = () => activeBand();
  const peqBands = () => band()?.peqBands ?? [];

  // Pending band (newly added, not yet committed)
  const [pendingPeqIdx, setPendingPeqIdx] = createSignal<number | null>(null);

  // Reset pending when bands change externally (e.g. after optimize)
  createEffect(() => {
    peqBands(); // track
    setPendingPeqIdx(null);
  });

  function handleRemovePeq(idx: number) {
    const b = band();
    if (!b) return;
    if (selectedPeqIdx() === idx) setSelectedPeqIdx(null);
    else if (selectedPeqIdx() != null && selectedPeqIdx()! > idx)
      setSelectedPeqIdx(selectedPeqIdx()! - 1);
    removePeqBand(b.id, idx);
  }

  return (
    <div class="peq-sidebar">
      {/* Header */}
      <div class="peq-sidebar-header">
        <span class="fb-title" style={{ "font-size": "11px" }}>PEQ Bands</span>
        <button
          class="peq-add-btn"
          onClick={() => {
            const b = band();
            if (b) {
              if (pendingPeqIdx() != null) {
                commitPeqBand(b.id, pendingPeqIdx()!);
              }
              addPeqBand(b.id, { freq_hz: 1000, gain_db: 0, q: 2.0, enabled: true });
              setPendingPeqIdx(0);
              setSelectedPeqIdx(0);
            }
          }}
          title="Add PEQ band"
        >+</button>
      </div>

      {/* PEQ Table */}
      <Show when={peqBands().length > 0}>
        <div class="peq-sidebar-table-scroll">
          <table class="peq-table">
            <thead>
              <tr>
                <th></th>
                <th>#</th>
                <th>Freq</th>
                <th>Gain</th>
                <th>Q</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {peqBands().map((b, i) => {
                const isPending = pendingPeqIdx() === i;
                return (
                  <tr
                    class={`${selectedPeqIdx() === i ? "peq-row-selected" : ""} ${isPending ? "peq-row-pending" : ""} ${!b.enabled ? "peq-row-disabled" : ""}`}
                    onClick={() => setSelectedPeqIdx(selectedPeqIdx() === i ? null : i)}
                  >
                    <td>
                      <input
                        type="checkbox"
                        class="peq-toggle"
                        checked={b.enabled}
                        onChange={(e) => {
                          e.stopPropagation();
                          const bd = band();
                          if (bd) updatePeqBand(bd.id, i, { enabled: !b.enabled });
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </td>
                    <td>{i + 1}</td>
                    <td>
                      <input
                        class="peq-input"
                        type="number"
                        value={Math.round(b.freq_hz)}
                        min={20}
                        max={20000}
                        step={1}
                        onChange={(e) => {
                          const v = parseFloat(e.currentTarget.value);
                          if (!isNaN(v) && v >= 20 && v <= 20000) {
                            const bd = band();
                            if (bd) updatePeqBand(bd.id, i, { freq_hz: v });
                          }
                        }}
                      />
                    </td>
                    <td>
                      <input
                        class={`peq-input ${b.gain_db > 0 ? "peq-boost" : "peq-cut"}`}
                        type="number"
                        value={b.gain_db.toFixed(1)}
                        min={-18}
                        max={6}
                        step={0.1}
                        onChange={(e) => {
                          const v = parseFloat(e.currentTarget.value);
                          if (!isNaN(v) && v >= -18 && v <= 6) {
                            const bd = band();
                            if (bd) updatePeqBand(bd.id, i, { gain_db: v });
                          }
                        }}
                      />
                    </td>
                    <td>
                      <input
                        class="peq-input"
                        type="number"
                        value={b.q.toFixed(1)}
                        min={0.1}
                        max={20}
                        step={0.1}
                        onChange={(e) => {
                          const v = parseFloat(e.currentTarget.value);
                          if (!isNaN(v) && v >= 0.1 && v <= 20) {
                            const bd = band();
                            if (bd) updatePeqBand(bd.id, i, { q: v });
                          }
                        }}
                      />
                    </td>
                    <td>
                      {isPending ? (
                        <button class="peq-commit" onClick={(e) => {
                          e.stopPropagation();
                          const bd = band();
                          if (bd) {
                            const newIdx = commitPeqBand(bd.id, i);
                            setPendingPeqIdx(null);
                            setSelectedPeqIdx(newIdx);
                          }
                        }} title="Confirm filter">✓</button>
                      ) : (
                        <button class="peq-remove" onClick={() => handleRemovePeq(i)}>×</button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Show>

      {/* Mini Corrected Impulse Plot */}
      <CorrectedImpulseMiniPlot />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────
// Mini Corrected+Target Impulse Plot
// ────────────────────────────────────────────────────────────────

function CorrectedImpulseMiniPlot() {
  let containerRef!: HTMLDivElement;
  let chartRef: uPlot | undefined;
  let renderGen = 0;

  const [hasData, setHasData] = createSignal(false);

  // Determine target impulse main lobe width (-20 dB threshold)
  function targetLobeWidthMs(time: number[], impulse: number[]): number {
    let peakIdx = 0, peakAbs = 0;
    for (let i = 0; i < impulse.length; i++) {
      if (Math.abs(impulse[i]) > peakAbs) { peakAbs = Math.abs(impulse[i]); peakIdx = i; }
    }
    const thr = peakAbs * 0.1; // -20 dB
    let left = peakIdx, right = peakIdx;
    for (let i = peakIdx; i >= 0; i--) { if (Math.abs(impulse[i]) < thr) { left = i; break; } }
    for (let i = peakIdx; i < impulse.length; i++) { if (Math.abs(impulse[i]) < thr) { right = i; break; } }
    return (time[right] - time[left]) * 1000;
  }

  function peakTimeMs(time: number[], impulse: number[]): number {
    let peakIdx = 0, peakAbs = 0;
    for (let i = 0; i < impulse.length; i++) {
      if (Math.abs(impulse[i]) > peakAbs) { peakAbs = Math.abs(impulse[i]); peakIdx = i; }
    }
    return time[peakIdx] * 1000;
  }

  // Reactive: recompute when band/peq/target change
  createEffect(async () => {
    const b = activeBand();
    if (!b || !b.measurement || !b.measurement.phase) {
      setHasData(false);
      return;
    }

    const gen = ++renderGen;
    const freq = [...b.measurement.freq];
    const measMag = [...b.measurement.magnitude];
    const measPhase = [...b.measurement.phase];
    const sr = b.measurement.sample_rate ?? 48000;

    try {
      // 1. PEQ correction
      let peqMag: number[] | null = null;
      let peqPhase: number[] | null = null;
      const hasPeq = b.peqBands && b.peqBands.length > 0;
      if (hasPeq) {
        const [pm, pp] = await invoke<[number[], number[]]>("compute_peq_complex", {
          freq, bands: b.peqBands,
        });
        if (gen !== renderGen) return;
        peqMag = pm;
        peqPhase = pp;
      }

      // 2. Target evaluation
      let targetMag: number[] | null = null;
      let targetPhase: number[] | null = null;
      const hasTarget = b.targetEnabled;
      if (hasTarget) {
        let sum = 0, n = 0;
        for (let i = 0; i < freq.length; i++) {
          if (freq[i] >= 200 && freq[i] <= 2000) { sum += measMag[i]; n++; }
        }
        const autoRef = n > 0 ? sum / n : 0;
        const targetCurve = JSON.parse(JSON.stringify(b.target));
        targetCurve.reference_level_db += autoRef;

        const resp = await invoke<TargetResponse>("evaluate_target", {
          target: targetCurve, freq,
        });
        if (gen !== renderGen) return;
        targetMag = resp.magnitude;
        targetPhase = resp.phase;
      }

      // 3. Cross-section (filters + makeup)
      let xsMag: number[] | null = null;
      let xsPhase: number[] | null = null;
      const hasFilters = hasTarget && (b.target.high_pass || b.target.low_pass);
      if (hasFilters && targetMag) {
        const [xm, xp] = await invoke<[number[], number[], number]>("compute_cross_section", {
          freq, measMag, targetMag,
          peqCorrection: peqMag ?? [],
          highPass: b.target.high_pass,
          lowPass: b.target.low_pass,
        });
        if (gen !== renderGen) return;
        xsMag = xm;
        xsPhase = xp;
      }

      // 4. Corrected = measurement + PEQ + cross-section
      const corrMag = measMag.map((v, i) => v + (peqMag ? peqMag[i] : 0) + (xsMag ? xsMag[i] : 0));
      const corrPhase = measPhase.map((v, i) => v + (peqPhase ? peqPhase[i] : 0) + (xsPhase ? xsPhase[i] : 0));

      // 5. Compute corrected impulse
      const corrIR = await invoke<ImpulseResult>("compute_impulse", {
        freq, magnitude: corrMag, phase: corrPhase, sampleRate: sr,
      });
      if (gen !== renderGen) return;

      // 6. Compute target impulse (if target enabled)
      let targetIR: ImpulseResult | null = null;
      if (targetMag && targetPhase) {
        targetIR = await invoke<ImpulseResult>("compute_impulse", {
          freq, magnitude: targetMag, phase: targetPhase, sampleRate: sr,
        });
        if (gen !== renderGen) return;
      }

      setHasData(true);
      requestAnimationFrame(() => {
        if (gen !== renderGen) return;
        renderMiniChart(corrIR, targetIR);
      });
    } catch (e) {
      console.error("CorrectedImpulseMiniPlot error:", e);
      setHasData(false);
    }
  });

  /** Linearly interpolate srcVal(srcTime) onto dstTime grid */
  function interpOnGrid(srcTime: number[], srcVal: number[], dstTime: number[]): number[] {
    const out = new Array(dstTime.length);
    let j = 0;
    for (let i = 0; i < dstTime.length; i++) {
      const t = dstTime[i];
      while (j < srcTime.length - 2 && srcTime[j + 1] < t) j++;
      if (t <= srcTime[0]) { out[i] = srcVal[0]; continue; }
      if (t >= srcTime[srcTime.length - 1]) { out[i] = srcVal[srcTime.length - 1]; continue; }
      const t0 = srcTime[j], t1 = srcTime[j + 1];
      const frac = (t1 !== t0) ? (t - t0) / (t1 - t0) : 0;
      out[i] = srcVal[j] + frac * (srcVal[j + 1] - srcVal[j]);
    }
    return out;
  }

  function renderMiniChart(corrIR: ImpulseResult, targetIR: ImpulseResult | null) {
    if (!containerRef) return;

    if (chartRef) { chartRef.destroy(); chartRef = undefined; }

    const rect = containerRef.getBoundingClientRect();
    const w = Math.max(rect.width, 100);
    const h = Math.max(rect.height, 60);

    // Center corrected impulse: peak = 0 ms
    const corrPeakMs = peakTimeMs(corrIR.time, corrIR.impulse);
    const corrTime = corrIR.time.map(t => t * 1000 - corrPeakMs);

    // Determine X range from target lobe width + 5ms padding
    let halfSpan = 10; // default ±10ms if no target
    if (targetIR) {
      const lobeW = targetLobeWidthMs(targetIR.time, targetIR.impulse);
      halfSpan = Math.max(lobeW / 2 + 5, 5);
    }
    const xMin = -halfSpan;
    const xMax = halfSpan;

    // Build uPlot data
    const uData: number[][] = [corrTime, corrIR.impulse];
    const series: uPlot.Series[] = [
      {},
      { label: "Corrected", stroke: CORRECTED_COLOR, width: 1.5, scale: "amp" },
    ];

    if (targetIR) {
      // Center target impulse by its own peak, then interpolate onto corrTime grid
      const tgtPeakMs = peakTimeMs(targetIR.time, targetIR.impulse);
      const tgtTime = targetIR.time.map(t => t * 1000 - tgtPeakMs);
      const tgtInterp = interpOnGrid(tgtTime, targetIR.impulse, corrTime);
      uData.push(tgtInterp);
      series.push({
        label: "Target",
        stroke: TARGET_COLOR,
        width: 1.5,
        dash: [5, 3],
        scale: "amp",
      });
    }

    // Y range from data
    let ampMin = 0, ampMax = 0;
    for (const row of uData.slice(1)) {
      for (const v of row) {
        if (v < ampMin) ampMin = v;
        if (v > ampMax) ampMax = v;
      }
    }
    const ampPad = Math.max((ampMax - ampMin) * 0.1, 5);

    const opts: uPlot.Options = {
      width: w,
      height: h,
      pxAlign: 1,
      cursor: { show: false },
      legend: { show: false },
      series,
      axes: [
        {
          stroke: "#666",
          grid: { stroke: "rgba(255,255,255,0.06)", width: 1 },
          ticks: { stroke: "#444", width: 1, size: 4 },
          font: "9px sans-serif",
          values: (_u: uPlot, vals: number[]) => vals.map(v => v.toFixed(1)),
        },
        {
          scale: "amp",
          stroke: "#666",
          grid: { stroke: "rgba(255,255,255,0.06)", width: 1 },
          ticks: { stroke: "#444", width: 1, size: 4 },
          font: "9px sans-serif",
          size: 36,
        },
      ],
      scales: {
        x: { min: xMin, max: xMax },
        amp: { min: ampMin - ampPad, max: ampMax + ampPad },
      },
    };

    chartRef = new uPlot(opts, uData as uPlot.AlignedData, containerRef);
  }

  // Resize observer
  onCleanup(() => { if (chartRef) chartRef.destroy(); });

  let resizeObserver: ResizeObserver | undefined;
  const setupResize = (el: HTMLDivElement) => {
    containerRef = el;
    resizeObserver = new ResizeObserver(() => {
      if (chartRef && containerRef) {
        const r = containerRef.getBoundingClientRect();
        chartRef.setSize({ width: Math.max(r.width, 100), height: Math.max(r.height, 60) });
      }
    });
    resizeObserver.observe(el);
  };

  onCleanup(() => resizeObserver?.disconnect());

  return (
    <div class="peq-sidebar-impulse">
      <div class="peq-sidebar-impulse-label">Impulse: Corrected + Target</div>
      <div
        class="peq-sidebar-impulse-chart"
        ref={setupResize}
      />
      <Show when={!hasData()}>
        <div class="peq-sidebar-impulse-empty">No data</div>
      </Show>
    </div>
  );
}
