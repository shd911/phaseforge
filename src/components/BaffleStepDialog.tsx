import { createSignal, createEffect, on, onCleanup, Show } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import uPlot from "uplot";
import type { BaffleConfig, BaffleStepPreview } from "../lib/types";
import NumberInput from "./NumberInput";

interface BaffleStepDialogProps {
  config: BaffleConfig;
  onSave: (config: BaffleConfig) => void;
  onClose: () => void;
}

const PREVIEW_FREQS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];

function formatFreq(hz: number): string {
  if (hz >= 1000) return (hz / 1000).toFixed(hz >= 10000 ? 0 : 1) + " kHz";
  return hz.toFixed(0) + " Hz";
}

export default function BaffleStepDialog(props: BaffleStepDialogProps) {
  const [width, setWidth] = createSignal(props.config.baffle_width_m);
  const [height, setHeight] = createSignal(props.config.baffle_height_m);
  const [offsetX, setOffsetX] = createSignal(props.config.driver_offset_x_m);
  const [offsetY, setOffsetY] = createSignal(props.config.driver_offset_y_m);
  const [preview, setPreview] = createSignal<BaffleStepPreview | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  let plotRef!: HTMLDivElement;
  const chartRef: { current: uPlot | undefined } = { current: undefined };

  function renderChart(p: BaffleStepPreview) {
    if (!plotRef) return;

    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = undefined;
    }

    const rect = plotRef.getBoundingClientRect();
    const w = Math.max(rect.width, 100);
    const h = Math.max(rect.height, 80);

    const opts: uPlot.Options = {
      width: w,
      height: h,
      series: [
        {},
        {
          label: "dB",
          stroke: "#4A9EFF",
          width: 1.5,
          scale: "db",
        },
      ],
      scales: {
        x: { distr: 3 }, // log
        db: { min: -7, max: 1 },
      },
      axes: [
        {
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => {
              if (v == null) return "";
              if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k";
              return v.toFixed(0);
            }),
          size: 30,
        },
        {
          scale: "db",
          stroke: "#8b8b96",
          grid: { stroke: "rgba(255,255,255,0.06)" },
          ticks: { stroke: "rgba(255,255,255,0.12)" },
          values: (_u: uPlot, vals: number[]) =>
            vals.map((v) => (v == null ? "" : v.toFixed(0))),
          size: 32,
        },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
    };

    try {
      chartRef.current = new uPlot(opts, [p.freq, p.correction_db] as uPlot.AlignedData, plotRef);
    } catch (e) {
      console.error("Baffle uPlot error:", e);
    }
  }

  // Re-render chart when preview changes
  createEffect(
    on(
      () => preview(),
      (p) => {
        if (p) {
          requestAnimationFrame(() => renderChart(p));
        }
      }
    )
  );

  onCleanup(() => {
    if (chartRef.current) chartRef.current.destroy();
  });

  const isValid = () =>
    offsetX() > 0.001 &&
    offsetX() < width() - 0.001 &&
    offsetY() > 0.001 &&
    offsetY() < height() - 0.001 &&
    width() > 0 &&
    height() > 0;

  const currentConfig = (): BaffleConfig => ({
    baffle_width_m: width(),
    baffle_height_m: height(),
    driver_offset_x_m: offsetX(),
    driver_offset_y_m: offsetY(),
  });

  // Debounced preview fetch
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;

  createEffect(
    on(
      () => [width(), height(), offsetX(), offsetY()],
      () => {
        clearTimeout(debounceTimer);
        if (!isValid()) {
          setPreview(null);
          return;
        }
        debounceTimer = setTimeout(async () => {
          try {
            const result = await invoke<BaffleStepPreview>("preview_baffle_step", {
              config: currentConfig(),
            });
            setPreview(result);
            setError(null);
          } catch (e) {
            setError(String(e));
            setPreview(null);
          }
        }, 200);
      }
    )
  );

  function getCorrectionAtFreq(targetHz: number): string {
    const p = preview();
    if (!p) return "...";
    // Find nearest frequency in preview grid
    let bestIdx = 0;
    let bestDist = Math.abs(p.freq[0] - targetHz);
    for (let i = 1; i < p.freq.length; i++) {
      const d = Math.abs(p.freq[i] - targetHz);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    return p.correction_db[bestIdx].toFixed(1);
  }

  function handleSave() {
    if (isValid()) {
      props.onSave(currentConfig());
    }
  }

  return (
    <div class="baffle-overlay" onClick={props.onClose}>
      <div class="baffle-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div class="merge-header">
          <span class="merge-title">Baffle Step Correction</span>
          <button class="merge-close" onClick={props.onClose}>
            ×
          </button>
        </div>

        {/* Body */}
        <div class="merge-body">
          {/* Dimension inputs */}
          <div class="merge-param-row">
            <label class="merge-label">Baffle Width</label>
            <NumberInput
              value={width()}
              onChange={setWidth}
              min={0.05}
              max={2.0}
              step={0.01}
              unit="m"
            />
          </div>
          <div class="merge-param-row">
            <label class="merge-label">Baffle Height</label>
            <NumberInput
              value={height()}
              onChange={setHeight}
              min={0.05}
              max={2.0}
              step={0.01}
              unit="m"
            />
          </div>
          <div class="merge-param-row">
            <label class="merge-label">Driver X</label>
            <NumberInput
              value={offsetX()}
              onChange={setOffsetX}
              min={0.01}
              max={width() - 0.01}
              step={0.01}
              unit="m"
            />
          </div>
          <div class="merge-param-row">
            <label class="merge-label">Driver Y</label>
            <NumberInput
              value={offsetY()}
              onChange={setOffsetY}
              min={0.01}
              max={height() - 0.01}
              step={0.01}
              unit="m"
            />
          </div>

          {/* Chart — always mounted so plotRef is valid */}
          <div
            class="baffle-chart-container"
            style={{ display: preview() ? "block" : "none" }}
          >
            <div ref={plotRef} class="baffle-chart" />
          </div>

          {/* Preview */}
          <Show when={preview()}>
            {(p) => (
              <div class="baffle-preview">
                <div class="baffle-info">
                  <span class="baffle-info-label">f3: </span>
                  <span>{p().f3_hz.toFixed(0)} Hz</span>
                </div>
                <div class="baffle-info">
                  <span class="baffle-info-label">Edges: </span>
                  <span>
                    L {p().edge_frequencies[0].toFixed(0)}{" "}
                    R {p().edge_frequencies[1].toFixed(0)}{" "}
                    T {p().edge_frequencies[2].toFixed(0)}{" "}
                    B {p().edge_frequencies[3].toFixed(0)} Hz
                  </span>
                </div>
                <div class="baffle-preview-table">
                  {PREVIEW_FREQS.map((f) => (
                    <div class="baffle-preview-row">
                      <span class="freq">{formatFreq(f)}</span>
                      <span class="value">{getCorrectionAtFreq(f)} dB</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Show>

          <Show when={error()}>
            <div class="merge-error">{error()}</div>
          </Show>
        </div>

        {/* Footer */}
        <div class="merge-footer">
          <button class="tb-btn" onClick={props.onClose}>
            Cancel
          </button>
          <button
            class="tb-btn primary"
            disabled={!isValid()}
            onClick={handleSave}
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}
