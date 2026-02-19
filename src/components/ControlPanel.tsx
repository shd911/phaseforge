import { createSignal, createEffect, on, Show } from "solid-js";
import type { FilterType, Measurement, MergeConfig, MergeResult, PeqConfig, PeqResult, PeqBand, FirConfig, FirResult, WindowType, PhaseMode } from "../lib/types";
import NumberInput from "./NumberInput";
import { MEASUREMENT_COLORS } from "../lib/types";
import {
  activeBand,
  appState,
  setActiveBand,
  toggleBandTarget,
  toggleBandInverted,
  toggleBandLinked,
  isBandLinkedFromPrev,
  setBandPreset,
  setBandReferenceLevel,
  setBandTilt,
  setBandHighPass,
  setBandLowPass,
  setBandLowShelf,
  setBandHighShelf,
  setBandSmoothing,
  setBandMeasurement,
  replaceBandMeasurement,
  setBandDelayInfo,
  setBandMergeSource,
  updateBandSpliceFreq,
  resetBandOriginalPhase,
  markBandDelayRemoved,
  restoreBandDelay,
  clearBandMeasurement,
  renameBand,
  toggleBandFloorBounce,
  updateBandFloorBounceField,
  setBandPeqBands,
  clearBandPeqBands,
  setBandFirResult,
  setBandMeasurementFile,
  activeTab,
  setActiveTab,
  selectedPeqIdx,
  setSelectedPeqIdx,
  exportSampleRate,
  setExportSampleRate,
  exportTaps,
  setExportTaps,
  exportWindow,
  setExportWindow,
  setPlotShowOnly,
} from "../stores/bands";
import type { PresetName, SmoothingMode, MergeSource, BandState } from "../stores/bands";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { setNeedAutoFit } from "../App";
import { projectDir, projectName, copyMeasurementToProject, copyMergeFilesToProject, wavFileName, sanitize } from "../lib/project-io";
import MergeDialog from "./MergeDialog";

const FILTER_TYPES: FilterType[] = ["Butterworth", "Bessel", "LinkwitzRiley", "Gaussian"];

export default function ControlPanel() {
  const tab = activeTab;
  const setTab = setActiveTab;
  const band = () => activeBand();

  return (
    <div class="ctrl-panel">
      <div class="ctrl-tabs">
        <button
          class={`ctrl-tab ${tab() === "measurements" ? "active" : ""}`}
          onClick={() => setTab("measurements")}
        >
          Measurement
          <Show when={band()?.measurement}>
            <span class="ctrl-tab-badge">1</span>
          </Show>
        </button>
        <button
          class={`ctrl-tab ${tab() === "target" ? "active" : ""}`}
          onClick={() => setTab("target")}
        >Target</button>
        <button
          class={`ctrl-tab ${tab() === "align" ? "active" : ""}`}
          onClick={() => setTab("align")}
        >Auto Align</button>
        <button
          class={`ctrl-tab ${tab() === "export" ? "active" : ""}`}
          onClick={() => setTab("export")}
        >Export</button>
      </div>

      <div class="ctrl-body">
        <Show when={tab() === "measurements"}>
          <MeasurementsTab />
        </Show>
        <Show when={tab() === "target"}>
          <FiltersTab />
        </Show>
        <Show when={tab() === "align"}>
          <AutoAlignTab />
        </Show>
        <Show when={tab() === "export"}>
          <ExportTab />
        </Show>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Filters Tab
// ---------------------------------------------------------------------------

function FiltersTab() {
  const band = () => activeBand();
  const target = () => band()?.target;
  const enabled = () => band()?.targetEnabled ?? false;
  const inverted = () => band()?.inverted ?? false;
  const bandId = () => band()?.id;

  // Индекс текущей полосы и кол-во полос (для отображения Link)
  const bandIdx = () => {
    const id = bandId();
    return id ? appState.bands.findIndex((b) => b.id === id) : -1;
  };
  const isLastBand = () => bandIdx() >= appState.bands.length - 1;
  const linked = () => band()?.linkedToNext ?? false;
  // HP текущей полосы связан с LP предыдущей?
  const hpLinked = () => { const id = bandId(); return id ? isBandLinkedFromPrev(id) : false; };

  return (
    <Show when={band()} fallback={<p class="meas-empty">No active band.</p>}>
      <div class="filters-row">
        {/* General */}
        <div class="filter-block">
          <div class="fb-header">
            <span class="fb-title">General</span>
            <button
              class={`fb-toggle ${enabled() ? "on" : ""}`}
              onClick={() => { const id = bandId(); if (id) toggleBandTarget(id); }}
            >{enabled() ? "ON" : "OFF"}</button>
          </div>
          <div class="fb-row">
            <label class="fb-label">Preset</label>
            <select
              class="fb-select"
              onChange={(e) => {
                const id = bandId();
                if (id) setBandPreset(id, e.currentTarget.value as PresetName);
              }}
            >
              <option value="flat">Flat</option>
              <option value="harman">Harman</option>
              <option value="bk">B&K</option>
              <option value="x-curve">X-Curve</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div class="fb-row">
            <label class="fb-label">Level</label>
            <NumberInput
              value={target()?.reference_level_db ?? 0}
              onChange={(v) => { const id = bandId(); if (id) setBandReferenceLevel(id, v); }}
              min={-20} max={20} step={0.5} unit="dB"
            />
          </div>
          <div class="fb-row">
            <label class="fb-label">Tilt</label>
            <NumberInput
              value={target()?.tilt_db_per_octave ?? 0}
              onChange={(v) => { const id = bandId(); if (id) setBandTilt(id, v); }}
              min={-6} max={6} step={0.1} unit="dB/oct"
            />
          </div>
          <div class="fb-row">
            <label class="fb-label">Invert</label>
            <button
              class={`fb-toggle invert-toggle ${inverted() ? "on" : ""}`}
              onClick={() => { const id = bandId(); if (id) toggleBandInverted(id); }}
            >{inverted() ? "INV" : "NOR"}</button>
          </div>
        </div>

        {/* High-Pass (🔗 индикатор, если связан с LP предыдущей) */}
        <FilterBlock
          title="High-Pass"
          config={target()?.high_pass ?? null}
          linked={hpLinked()}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            if (target()?.high_pass) {
              setBandHighPass(id, null);
            } else {
              setBandHighPass(id, { filter_type: "Butterworth", order: 2, freq_hz: 80, shape: null, linear_phase: false });
            }
          }}
          onChange={(c) => { const id = bandId(); if (id) setBandHighPass(id, c); }}
        />

        {/* Low-Pass (🔗 кнопка, если не последняя полоса) */}
        <FilterBlock
          title="Low-Pass"
          config={target()?.low_pass ?? null}
          linked={linked()}
          canLink={!isLastBand()}
          onLinkToggle={() => { const id = bandId(); if (id) toggleBandLinked(id); }}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            if (target()?.low_pass) {
              setBandLowPass(id, null);
            } else {
              setBandLowPass(id, { filter_type: "Butterworth", order: 2, freq_hz: 15000, shape: null, linear_phase: false });
            }
          }}
          onChange={(c) => { const id = bandId(); if (id) setBandLowPass(id, c); }}
        />

        {/* Low Shelf */}
        <ShelfBlock
          title="Low Shelf"
          config={target()?.low_shelf ?? null}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            if (target()?.low_shelf) {
              setBandLowShelf(id, null);
            } else {
              setBandLowShelf(id, { freq_hz: 200, gain_db: 3, q: 0.7 });
            }
          }}
          onChange={(c) => { const id = bandId(); if (id) setBandLowShelf(id, c); }}
        />

        {/* High Shelf */}
        <ShelfBlock
          title="High Shelf"
          config={target()?.high_shelf ?? null}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            if (target()?.high_shelf) {
              setBandHighShelf(id, null);
            } else {
              setBandHighShelf(id, { freq_hz: 8000, gain_db: -2, q: 0.7 });
            }
          }}
          onChange={(c) => { const id = bandId(); if (id) setBandHighShelf(id, c); }}
        />
      </div>
    </Show>
  );
}

// ---------------------------------------------------------------------------
// FilterBlock (HP / LP)
// ---------------------------------------------------------------------------

interface FilterBlockProps {
  title: string;
  config: import("../lib/types").FilterConfig | null;
  linked?: boolean; // связь активна
  canLink?: boolean; // можно переключать связь (для LP — если не последняя полоса)
  onLinkToggle?: () => void; // callback переключения связи
  onToggle: () => void;
  onChange: (c: import("../lib/types").FilterConfig) => void;
}

function FilterBlock(props: FilterBlockProps) {
  const c = () => props.config;
  const isGaussian = () => c()?.filter_type === "Gaussian";

  return (
    <div class={`filter-block ${props.linked ? "fb-linked" : ""}`}>
      <div class="fb-header">
        <span class="fb-title">
          {props.title}
          {/* Кликабельная кнопка link (для LP) или read-only индикатор (для HP) */}
          <Show when={props.canLink && props.onLinkToggle}>
            <button
              class={`fb-link-btn ${props.linked ? "on" : ""}`}
              onClick={(e) => { e.stopPropagation(); props.onLinkToggle!(); }}
              title={props.linked ? "Unlink from next band" : "Link to next band"}
            >🔗</button>
          </Show>
          <Show when={!props.canLink && props.linked}>
            <span class="fb-link-indicator" title="Linked to adjacent band">🔗</span>
          </Show>
        </span>
        <button
          class={`fb-toggle ${c() ? "on" : ""}`}
          onClick={props.onToggle}
        >{c() ? "ON" : "OFF"}</button>
      </div>
      <Show when={c()}>
        <div class="fb-row">
          <label class="fb-label">Type</label>
          <select
            class="fb-select"
            value={c()!.filter_type}
            onChange={(e) => {
              const ft = e.currentTarget.value as FilterType;
              if (ft === "Gaussian") {
                // Gaussian filters are inherently linear-phase → force linear_phase=true
                props.onChange({ ...c()!, filter_type: ft, shape: c()!.shape ?? 1.0, linear_phase: true });
              } else {
                props.onChange({ ...c()!, filter_type: ft, shape: null });
              }
            }}
          >
            {FILTER_TYPES.map((t) => <option value={t}>{t}</option>)}
          </select>
          <label class="fb-checkbox" title={isGaussian() ? "Gaussian is always linear-phase" : "Linear phase (magnitude only, no phase rotation)"}>
            <input
              type="checkbox"
              checked={isGaussian() ? true : c()!.linear_phase}
              disabled={isGaussian()}
              onChange={(e) => props.onChange({ ...c()!, linear_phase: e.currentTarget.checked })}
            />
            <span class="fb-check-label">Lin-φ</span>
          </label>
        </div>
        <div class="fb-row">
          <label class="fb-label">Freq</label>
          <NumberInput
            value={c()!.freq_hz}
            onChange={(v) => props.onChange({ ...c()!, freq_hz: v })}
            min={10} max={20000} step={1} unit="Hz" freqMode
          />
        </div>
        <Show when={!isGaussian()}>
          <div class="fb-row">
            <label class="fb-label">Order</label>
            <NumberInput
              value={c()!.order}
              onChange={(v) => props.onChange({ ...c()!, order: v })}
              min={1} max={8} step={1} precision={0}
            />
          </div>
        </Show>
        <Show when={isGaussian()}>
          <div class="fb-row">
            <label class="fb-label">M</label>
            <NumberInput
              value={c()!.shape ?? 1.0}
              onChange={(v) => props.onChange({ ...c()!, shape: v })}
              min={0.5} max={10} step={0.1}
            />
          </div>
        </Show>
      </Show>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ShelfBlock
// ---------------------------------------------------------------------------

interface ShelfBlockProps {
  title: string;
  config: import("../lib/types").ShelfConfig | null;
  onToggle: () => void;
  onChange: (c: import("../lib/types").ShelfConfig) => void;
}

function ShelfBlock(props: ShelfBlockProps) {
  const c = () => props.config;

  return (
    <div class="filter-block">
      <div class="fb-header">
        <span class="fb-title">{props.title}</span>
        <button
          class={`fb-toggle ${c() ? "on" : ""}`}
          onClick={props.onToggle}
        >{c() ? "ON" : "OFF"}</button>
      </div>
      <Show when={c()}>
        <div class="fb-row">
          <label class="fb-label">Freq</label>
          <NumberInput
            value={c()!.freq_hz}
            onChange={(v) => props.onChange({ ...c()!, freq_hz: v })}
            min={10} max={20000} step={1} unit="Hz" freqMode
          />
        </div>
        <div class="fb-row">
          <label class="fb-label">Gain</label>
          <NumberInput
            value={c()!.gain_db}
            onChange={(v) => props.onChange({ ...c()!, gain_db: v })}
            min={-20} max={20} step={0.5} unit="dB"
          />
        </div>
        <div class="fb-row">
          <label class="fb-label">Q</label>
          <NumberInput
            value={c()!.q}
            onChange={(v) => props.onChange({ ...c()!, q: v })}
            min={0.1} max={10} step={0.1}
          />
        </div>
      </Show>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Measurements Tab — single measurement per band
// ---------------------------------------------------------------------------

function formatDistance(meters: number): string {
  if (meters < 0.01) return "< 1 cm";
  if (meters < 1) return (meters * 100).toFixed(1) + " cm";
  return meters.toFixed(2) + " m";
}

function MeasurementsTab() {
  const band = () => activeBand();
  const m = () => band()?.measurement;
  const s = () => band()?.settings;

  const [showMergeDialog, setShowMergeDialog] = createSignal(false);

  async function handleImport() {
    const b = band();
    if (!b) return;
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "Measurement Files", extensions: ["txt", "frd"] }],
      });
      if (!selected) return;
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      const measurement = await invoke<Measurement>("import_measurement", { path: filePath });
      setBandMeasurement(b.id, measurement);
      // Переименовать вкладку: "Band N · filename"
      const bandNum = b.name.match(/\d+/)?.[0] ?? "1";
      const newName = `Band ${bandNum} · ${measurement.name}`;
      renameBand(b.id, newName);
      // Copy measurement to project folder (if project folder exists)
      try {
        const fileName = await copyMeasurementToProject(filePath, newName);
        if (fileName) {
          setBandMeasurementFile(b.id, fileName);
        }
      } catch (e) {
        console.warn("Failed to copy measurement to project folder:", e);
      }
      setNeedAutoFit(true);
      // Авто-компенсация временной задержки при импорте
      if (measurement.phase) {
        try {
          const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
            "remove_measurement_delay",
            { freq: measurement.freq, phase: measurement.phase }
          );
          setBandDelayInfo(b.id, delay, distance);
          markBandDelayRemoved(b.id, newPhase);
        } catch (e) {
          console.error("Delay removal failed:", e);
        }
      }
    } catch (e) {
      console.error("Import failed:", e);
    }
  }

  async function handleMergeComplete(measurement: Measurement, source: MergeSource) {
    const b = band();
    if (!b) return;
    setBandMeasurement(b.id, measurement);
    setBandMergeSource(b.id, source);
    // Переименовать вкладку: "Band N · filename"
    const bandNum = b.name.match(/\d+/)?.[0] ?? "1";
    const newName = `Band ${bandNum} · ${measurement.name}`;
    renameBand(b.id, newName);
    // Copy NF/FF files to project folder (if project folder exists)
    try {
      const files = await copyMergeFilesToProject(source.nfPath, source.ffPath, newName);
      if (files) {
        // Also copy the merged measurement result
        const mFileName = await copyMeasurementToProject(
          measurement.source_path ?? source.nfPath,
          newName,
        );
        if (mFileName) {
          setBandMeasurementFile(b.id, mFileName);
        }
      }
    } catch (e) {
      console.warn("Failed to copy merge files to project folder:", e);
    }
    setNeedAutoFit(true);
    // Авто-компенсация временной задержки при merge
    if (measurement.phase) {
      try {
        const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
          "remove_measurement_delay",
          { freq: measurement.freq, phase: measurement.phase }
        );
        setBandDelayInfo(b.id, delay, distance);
        markBandDelayRemoved(b.id, newPhase);
      } catch (e) {
        console.error("Delay removal failed:", e);
      }
    }
  }

  async function handleToggleDelay() {
    const b = band();
    if (!b || !b.measurement?.phase || !b.settings) return;
    if (b.settings.delay_removed) {
      restoreBandDelay(b.id);
    } else {
      try {
        const origPhase = b.settings.originalPhase ?? b.measurement.phase;
        const [newPhase, _delay, _distance] = await invoke<[number[], number, number]>(
          "remove_measurement_delay",
          { freq: b.measurement.freq, phase: origPhase }
        );
        markBandDelayRemoved(b.id, newPhase);
      } catch (e) {
        console.error("Remove delay failed:", e);
      }
    }
  }

  function handleRemoveMeasurement() {
    const b = band();
    if (b) clearBandMeasurement(b.id);
  }

  return (
    <div class="measurements-tab">
      {/* Import buttons */}
      <div class="meas-actions">
        <button class="tb-btn primary" onClick={handleImport}>
          Import File
        </button>
        <button class="tb-btn" onClick={() => setShowMergeDialog(true)}>
          Merge NF+FF
        </button>
      </div>

      <Show
        when={m()}
        fallback={<p class="meas-empty">No measurement loaded.</p>}
      >
        <table class="meas-table">
          <thead>
            <tr>
              <th></th>
              <th>Name</th>
              <th>Pts</th>
              <Show when={s()?.mergeSource}>
                <th>Splice</th>
              </Show>
              <th>Smooth</th>
              <th>Dist</th>
              <th>Delay</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <span
                  class="color-dot"
                  style={{ "background-color": MEASUREMENT_COLORS[0] }}
                />
              </td>
              <td class="meas-name">{m()!.name}</td>
              <td class="meas-pts">{m()!.freq.length}</td>
              <Show when={s()?.mergeSource}>
                <td class="meas-splice-cell">
                  <SpliceSlider />
                </td>
              </Show>
              <td>
                <select
                  class="fb-select meas-smooth-select"
                  value={s()?.smoothing ?? "off"}
                  onChange={(e) => {
                    const b = band();
                    if (b) setBandSmoothing(b.id, e.currentTarget.value as SmoothingMode);
                  }}
                >
                  <option value="off">Off</option>
                  <option value="1/3">1/3</option>
                  <option value="1/6">1/6</option>
                  <option value="1/12">1/12</option>
                  <option value="1/24">1/24</option>
                  <option value="var">Var</option>
                </select>
              </td>
              <td class="meas-dist">
                {s()?.distance_meters != null
                  ? formatDistance(s()!.distance_meters!)
                  : "—"}
              </td>
              <td>
                <Show when={m()!.phase} fallback={<span class="meas-text-muted">—</span>}>
                  <label class="meas-delay-check" title="Compensate propagation delay">
                    <input
                      type="checkbox"
                      checked={s()?.delay_removed ?? false}
                      onChange={handleToggleDelay}
                    />
                  </label>
                </Show>
              </td>
              <td>
                <button
                  class="meas-remove"
                  onClick={handleRemoveMeasurement}
                >×</button>
              </td>
            </tr>
          </tbody>
        </table>

        {/* Floor Bounce — compact one-line layout */}
        <div class="floor-bounce-section">
          <div class="floor-bounce-inline">
            <span class="floor-bounce-title">Floor Bounce</span>
            <button
              class={`fb-toggle small ${s()?.floorBounce?.enabled ? "on" : ""}`}
              onClick={() => {
                const b = band();
                if (b) toggleBandFloorBounce(b.id);
              }}
            >{s()?.floorBounce?.enabled ? "ON" : "OFF"}</button>
            <Show when={s()?.floorBounce?.enabled}>
              <span class="fb-inline-sep" />
              <label class="fb-inline-label">Spk</label>
              <NumberInput
                value={s()!.floorBounce!.speakerHeight}
                onChange={(v) => {
                  const b = band();
                  if (b) updateBandFloorBounceField(b.id, "speakerHeight", v);
                }}
                min={0} max={5} step={0.01} unit="m"
              />
              <label class="fb-inline-label">Mic</label>
              <NumberInput
                value={s()!.floorBounce!.micHeight}
                onChange={(v) => {
                  const b = band();
                  if (b) updateBandFloorBounceField(b.id, "micHeight", v);
                }}
                min={0} max={5} step={0.01} unit="m"
              />
              <label class="fb-inline-label">Dist</label>
              <NumberInput
                value={s()!.floorBounce!.distance}
                onChange={(v) => {
                  const b = band();
                  if (b) updateBandFloorBounceField(b.id, "distance", v);
                }}
                min={0.1} max={20} step={0.1} unit="m"
              />
            </Show>
          </div>
        </div>
      </Show>

      {/* Merge dialog */}
      <Show when={showMergeDialog()}>
        <MergeDialog
          onClose={() => setShowMergeDialog(false)}
          onMerge={handleMergeComplete}
        />
      </Show>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Splice Frequency Slider — inline re-merge control
// ---------------------------------------------------------------------------

function SpliceSlider() {
  const b = () => activeBand();
  const src = () => b()?.settings?.mergeSource;

  // Log-scale slider: 50–1000 Hz
  const LOG_MIN = Math.log10(50);
  const LOG_MAX = Math.log10(1000);

  const sliderValue = () => {
    const f = src()?.config.splice_freq ?? 300;
    return ((Math.log10(f) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 1000;
  };

  const freqFromSlider = (v: number): number => {
    const logF = LOG_MIN + (v / 1000) * (LOG_MAX - LOG_MIN);
    return Math.round(Math.pow(10, logF));
  };

  const [remerging, setRemerging] = createSignal(false);
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;

  function handleSliderInput(v: number) {
    const freq = freqFromSlider(v);
    const band = b();
    if (!band) return;
    updateBandSpliceFreq(band.id, freq);

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => remerge(freq), 250);
  }

  async function remerge(freq: number) {
    const band = b();
    const s = src();
    if (!band || !s) return;

    // Запоминаем состояние delay compensation до re-merge
    const wasDelayRemoved = band.settings?.delay_removed ?? false;

    setRemerging(true);
    try {
      const config: MergeConfig = { ...s.config, splice_freq: freq };
      const result = await invoke<MergeResult>("merge_measurements", {
        nfPath: s.nfPath,
        ffPath: s.ffPath,
        config,
      });
      // Обновляем measurement без сброса settings
      replaceBandMeasurement(band.id, result.measurement);
      setBandMergeSource(band.id, { ...s, config });

      if (result.measurement.phase) {
        try {
          const [delay, distance] = await invoke<[number, number]>("compute_delay_info", {
            freq: result.measurement.freq,
            phase: result.measurement.phase,
          });
          setBandDelayInfo(band.id, delay, distance);

          // Если delay comp был включён — повторно применяем на новой фазе
          if (wasDelayRemoved) {
            // Сбросить originalPhase чтобы markBandDelayRemoved сохранила новую сырую фазу
            resetBandOriginalPhase(band.id);
            const [newPhase] = await invoke<[number[], number, number]>(
              "remove_measurement_delay",
              { freq: result.measurement.freq, phase: result.measurement.phase }
            );
            markBandDelayRemoved(band.id, newPhase);
          }
        } catch (_) {}
      }
    } catch (e) {
      console.error("Re-merge failed:", e);
    } finally {
      setRemerging(false);
    }
  }

  return (
    <div class="splice-inline">
      <input
        type="range"
        class="splice-range"
        min={0}
        max={1000}
        step={1}
        value={sliderValue()}
        onInput={(e) => handleSliderInput(parseInt(e.currentTarget.value))}
      />
      <span class="splice-value">{src()?.config.splice_freq ?? 300}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auto Align Tab — PEQ auto-fitting + FIR generation
// ---------------------------------------------------------------------------

function AutoAlignTab() {
  const band = () => activeBand();
  const m = () => band()?.measurement;
  const target = () => band()?.target;
  const peqBands = () => band()?.peqBands ?? [];
  const firResult = () => band()?.firResult ?? null;

  const [tolerance, setTolerance] = createSignal(1.0);
  const [maxBands, setMaxBands] = createSignal(20);
  const [computing, setComputing] = createSignal(false);
  const [peqError, setPeqError] = createSignal<string | null>(null);
  const [maxErr, setMaxErr] = createSignal<number | null>(null);
  const [iters, setIters] = createSignal<number | null>(null);
  // Index of uncommitted (newly added) PEQ band, null = all committed
  const [pendingPeqIdx, setPendingPeqIdx] = createSignal<number | null>(null);

  // FIR state
  const [firSampleRate, setFirSampleRate] = createSignal(48000);
  const [firTaps, setFirTaps] = createSignal(65536);
  const [firWindow, setFirWindow] = createSignal<WindowType>("Blackman");
  const [firPhaseMode, setFirPhaseMode] = createSignal<PhaseMode>("MinimumPhase");
  const [firGenerating, setFirGenerating] = createSignal(false);
  const [firError, setFirError] = createSignal<string | null>(null);
  const [recTaps, setRecTaps] = createSignal<number | null>(null);

  // Crossover range from target HP/LP
  const crossoverRange = (): [number, number] => {
    const t = target();
    const fLow = t?.high_pass?.freq_hz ?? 20;
    const fHigh = t?.low_pass?.freq_hz ?? 20000;
    return [fLow, fHigh];
  };

  async function handleOptimizePeq() {
    const b = band();
    if (!b || !b.measurement) return;

    setComputing(true);
    setPeqError(null);
    try {
      const meas = b.measurement;
      const [fLow, fHigh] = crossoverRange();

      // Evaluate the main Target Curve (same as FIR uses)
      // Auto-reference: mean measurement magnitude 200–2000 Hz
      let refOffset = 0, count = 0;
      for (let i = 0; i < meas.freq.length; i++) {
        if (meas.freq[i] >= 200 && meas.freq[i] <= 2000) {
          refOffset += meas.magnitude[i]; count++;
        }
      }
      refOffset = count > 0 ? refOffset / count : 0;

      const targetCurve = JSON.parse(JSON.stringify(b.target));
      targetCurve.reference_level_db += refOffset;

      const targetResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", {
        target: targetCurve, freq: meas.freq,
      });

      // Unified LMA optimizer: covers both in-band and above-LP in one pass.
      // Range: from 3 octaves below HP to 20 kHz.
      const peqLow = Math.max(20, fLow / 8);
      const peqHigh = 20000;
      const config: PeqConfig = {
        max_bands: maxBands(),
        tolerance_db: tolerance(),
        peak_bias: 1.5,
        max_boost_db: 6.0,
        max_cut_db: 18.0,
        freq_range: [peqLow, peqHigh],
      };

      const result = await invoke<PeqResult>("auto_peq_lma", {
        freq: meas.freq,
        measurementMag: meas.magnitude,
        targetMag: targetResp.magnitude,  // main Target Curve
        config,
        hpFreq: fLow,
        lpFreq: fHigh,
      });

      // Replace all bands (single unified result)
      setBandPeqBands(b.id, result.bands);
      setMaxErr(result.max_error_db);
      setIters(result.iterations);
      setSelectedPeqIdx(null);
      setPendingPeqIdx(null);
      // Show only corrected curve on FrequencyPlot after optimization
      setPlotShowOnly(["corrected"]);
    } catch (e) {
      setPeqError(String(e));
    } finally {
      setComputing(false);
    }
  }

  function handleClearPeq() {
    const b = band();
    if (b) clearBandPeqBands(b.id);
    setMaxErr(null);
    setSelectedPeqIdx(null);
    setPendingPeqIdx(null);
  }

  // FIR: update recommended taps
  async function updateRecTaps(sr?: number) {
    const sampleRate = sr ?? firSampleRate();
    const [fLow] = crossoverRange();
    try {
      const rec = await invoke<number>("recommend_fir_taps", {
        lowestFreq: fLow,
        sampleRate: sampleRate,
      });
      setRecTaps(rec);
    } catch (_) {
      setRecTaps(null);
    }
  }

  // FIR: generate
  async function handleGenerateFir() {
    const b = band();
    if (!b || !b.measurement) return;

    setFirGenerating(true);
    setFirError(null);
    try {
      const meas = b.measurement;
      let measMag = meas.magnitude;

      // Apply smoothing if set
      if (b.settings?.smoothing && b.settings.smoothing !== "off") {
        const fractionMap: Record<string, number> = {
          "1/3": 1/3, "1/6": 1/6, "1/12": 1/12, "1/24": 1/24, "var": 0,
        };
        const frac = fractionMap[b.settings.smoothing];
        const smoothConfig = frac === 0
          ? { variable: true, fixed_fraction: null }
          : { variable: false, fixed_fraction: frac };
        measMag = await invoke<number[]>("get_smoothed", {
          freq: meas.freq, magnitude: meas.magnitude, config: smoothConfig,
        });
      }

      // Auto-reference: mean measurement 200-2000 Hz
      let refOffset = 0, count = 0;
      for (let i = 0; i < meas.freq.length; i++) {
        if (meas.freq[i] >= 200 && meas.freq[i] <= 2000) {
          refOffset += measMag[i]; count++;
        }
      }
      refOffset = count > 0 ? refOffset / count : 0;

      const targetCurve = JSON.parse(JSON.stringify(b.target));
      targetCurve.reference_level_db += refOffset;

      const targetResp = await invoke<{ magnitude: number[]; phase: number[] }>("evaluate_target", {
        target: targetCurve, freq: meas.freq,
      });

      // PEQ correction (if any)
      let peqCorrection: number[] = [];
      if (b.peqBands.length > 0) {
        peqCorrection = await invoke<number[]>("compute_peq_response", {
          freq: meas.freq, bands: b.peqBands,
        });
      }

      const [fLow, fHigh] = crossoverRange();
      const firConfig: FirConfig = {
        taps: firTaps(),
        sample_rate: firSampleRate(),
        max_boost_db: 18.0,
        noise_floor_db: -60.0,
        window: firWindow(),
        phase_mode: firPhaseMode(),
      };

      const result = await invoke<FirResult>("generate_fir", {
        freq: meas.freq,
        measMag: measMag,
        targetMag: targetResp.magnitude,
        peqCorrection: peqCorrection,
        config: firConfig,
        crossoverRange: [fLow, fHigh],
      });

      setBandFirResult(b.id, result);
    } catch (e) {
      setFirError(String(e));
    } finally {
      setFirGenerating(false);
    }
  }

  // FIR: save WAV
  async function handleSaveWav() {
    const fir = firResult();
    const b = band();
    if (!fir) return;
    try {
      const pName = projectName();
      const dir = projectDir();
      let defPath: string;
      if (pName && b) {
        defPath = wavFileName(pName, b.name, fir.sample_rate, fir.taps);
        if (dir) defPath = `${dir}/${defPath}`;
      } else {
        defPath = `correction_${fir.taps}taps_${fir.sample_rate}Hz.wav`;
      }
      const filePath = await save({
        filters: [{ name: "WAV Audio", extensions: ["wav"] }],
        defaultPath: defPath,
      });
      if (!filePath) return;
      await invoke("export_fir_wav", {
        impulse: fir.impulse,
        sampleRate: fir.sample_rate,
        path: filePath,
      });
    } catch (e) {
      setFirError(String(e));
    }
  }

  function formatFreq(hz: number): string {
    if (hz >= 1000) return (hz / 1000).toFixed(1) + "k";
    return Math.round(hz).toString();
  }

  // PEQ effective range (crossover ±3 oct, clamped)
  const peqRange = (): [number, number] => {
    const [lo, hi] = crossoverRange();
    return [Math.max(20, lo / 8), Math.min(20000, hi * 8)];
  };

  return (
    <div class="align-tab">
      <Show
        when={m()}
        fallback={<p class="meas-empty">No measurement loaded. Import a measurement first.</p>}
      >
          <div class="align-sections">
            {/* PEQ Config Block */}
            <div class="align-block">
              <div class="fb-header">
                <span class="fb-title">PEQ Auto-Fit</span>
              </div>
              <div class="fb-row">
                <label class="fb-label">Tolerance</label>
                <NumberInput
                  value={tolerance()}
                  onChange={setTolerance}
                  min={0.5} max={3.0} step={0.1} unit="dB"
                />
              </div>
              <div class="fb-row">
                <label class="fb-label">Max bands</label>
                <NumberInput
                  value={maxBands()}
                  onChange={(v) => setMaxBands(Math.round(v))}
                  min={1} max={60} step={1} precision={0}
                />
              </div>
              <div class="fb-row">
                <label class="fb-label">Range</label>
                <span class="align-range-info">{formatFreq(peqRange()[0])}–{formatFreq(peqRange()[1])} Hz</span>
              </div>
              <div class="peq-buttons-row">
                <button
                  class="tb-btn primary"
                  onClick={handleOptimizePeq}
                  disabled={computing()}
                >
                  {computing() ? "Optimizing..." : "Optimize PEQ"}
                </button>
              </div>
              <Show when={peqBands().length > 0}>
                <button class="tb-btn" onClick={handleClearPeq}>Clear</button>
              </Show>
              <Show when={peqError()}>
                <div class="align-error">{peqError()}</div>
              </Show>
              <Show when={peqBands().length > 0}>
                <div class="align-status">
                  {peqBands().length} band{peqBands().length > 1 ? "s" : ""}
                  {maxErr() != null ? ` · max err: ${maxErr()!.toFixed(1)} dB` : ""}
                  {iters() != null ? ` · ${iters()} iter` : ""}
                </div>
              </Show>
            </div>

            {/* PEQ Bands Table moved to PeqSidebar (right of FrequencyPlot) */}

            {/* Cross Section — pure mathematical model, no FIR/taps/window here */}
            <div class="align-block fir-block">
              <div class="fb-header">
                <span class="fb-title">Cross Section</span>
              </div>
              <div class="align-status" style={{ "font-size": "10px", "color": "var(--text-secondary)" }}>
                Filters applied to corrected curve. Makeup below target generated automatically.
              </div>
              <Show when={band()?.crossNormDb}>
                <div class="align-status" style={{ "font-size": "10px", "margin-top": "2px" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Normalization: </span>
                  <span style={{ color: "#F87171", "font-weight": "600" }}>
                    −{band()!.crossNormDb.toFixed(1)} dB
                  </span>
                </div>
              </Show>
            </div>
          </div>
      </Show>
    </div>
  );
}

// ---------------------------------------------------------------------------
// FIR Mini Preview — canvas-based impulse preview
// ---------------------------------------------------------------------------

function FirMiniPreview(props: { result: FirResult }) {
  let canvasRef: HTMLCanvasElement | undefined;

  createEffect(() => {
    const fir = props.result;
    if (!canvasRef || !fir) return;

    const ctx = canvasRef.getContext("2d");
    if (!ctx) return;

    const w = canvasRef.width;
    const h = canvasRef.height;
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size for crisp rendering
    canvasRef.width = w * dpr;
    canvasRef.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#1a1a20";
    ctx.fillRect(0, 0, w, h);

    const impulse = fir.impulse;
    const n = impulse.length;
    if (n === 0) return;

    // Find peak for normalization
    const peak = Math.max(...impulse.map(Math.abs));
    if (peak === 0) return;

    // Draw impulse — downsample for display
    const displayN = Math.min(n, w * 2); // max 2 samples per pixel
    const step = n / displayN;

    ctx.strokeStyle = "#4A9EFF";
    ctx.lineWidth = 1;
    ctx.beginPath();

    const midY = h / 2;
    for (let i = 0; i < displayN; i++) {
      const idx = Math.floor(i * step);
      const x = (i / displayN) * w;
      const y = midY - (impulse[idx] / peak) * (h * 0.45);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Zero line
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();
  });

  return (
    <canvas
      ref={canvasRef}
      width={240}
      height={60}
      class="fir-preview-canvas"
    />
  );
}

// ---------------------------------------------------------------------------
// Export Tab — placeholder
// ---------------------------------------------------------------------------

function ExportTab() {
  const [exportError, setExportError] = createSignal<string | null>(null);
  const [exporting, setExporting] = createSignal(false);

  const sampleRates = [44100, 48000, 88200, 96000, 176400, 192000];
  const tapOptions = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
  const windowOptions: WindowType[] = ["Blackman", "Kaiser", "Tukey", "Hann"];

  // Determine phase mode from target filters (HP/LP linear_phase flags)
  const isFilterLinear = (f: import("../lib/types").FilterConfig | null | undefined) =>
    !f || f.linear_phase || f.filter_type === "Gaussian";

  const bandPhaseLabel = (b: BandState) => {
    if (!b.target) return "Linear";
    const lin = isFilterLinear(b.target.high_pass) && isFilterLinear(b.target.low_pass);
    return lin ? "Linear" : "Min-\u03C6";
  };

  const bandPhaseIsLinear = (b: BandState) => {
    if (!b.target) return true;
    return isFilterLinear(b.target.high_pass) && isFilterLinear(b.target.low_pass);
  };

  function formatFilterInfo(b: BandState): string {
    const parts: string[] = [];
    const hp = b.target?.high_pass;
    const lp = b.target?.low_pass;
    if (hp) {
      const f = hp.freq_hz >= 1000 ? (hp.freq_hz / 1000).toFixed(1) + "k" : Math.round(hp.freq_hz).toString();
      const t = hp.filter_type === "Gaussian" ? "Gauss" :
        hp.filter_type === "LinkwitzRiley" ? "LR" + (hp.order * 6) :
        hp.filter_type === "Butterworth" ? "BW" + (hp.order * 6) :
        hp.filter_type === "Bessel" ? "Bes" + (hp.order * 6) : hp.filter_type;
      parts.push(`HP ${f} ${t}`);
    }
    if (lp) {
      const f = lp.freq_hz >= 1000 ? (lp.freq_hz / 1000).toFixed(1) + "k" : Math.round(lp.freq_hz).toString();
      const t = lp.filter_type === "Gaussian" ? "Gauss" :
        lp.filter_type === "LinkwitzRiley" ? "LR" + (lp.order * 6) :
        lp.filter_type === "Butterworth" ? "BW" + (lp.order * 6) :
        lp.filter_type === "Bessel" ? "Bes" + (lp.order * 6) : lp.filter_type;
      parts.push(`LP ${f} ${t}`);
    }
    return parts.length > 0 ? parts.join(" · ") : "\u2014";
  }

  // sanitizeFilename kept for backward compat; new code uses sanitize() from project-io
  function sanitizeFilename(name: string): string {
    return name.replace(/[^a-zA-Z0-9_\-]/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "");
  }

  // Core: generate FIR impulse for a band, return impulse array
  async function generateBandImpulse(b: BandState): Promise<number[]> {
    const sr = exportSampleRate();
    const taps = exportTaps();
    const win = exportWindow();
    const peqBands = b.peqBands?.filter((p: any) => p.enabled) ?? [];

    // 1. Evaluate pure target (HP/LP/shelf/tilt)
    const [freq, response] = await invoke<[number[], { magnitude: number[]; phase: number[] }]>(
      "evaluate_target_standalone",
      { target: { ...b.target }, nPoints: 512, fMin: 5, fMax: 40000 }
    );

    const targetMag = response.magnitude;
    let modelPhase = response.phase;

    // 2. Compute PEQ contribution separately (PEQ always min-phase)
    let peqMagArr: number[] = [];
    if (peqBands.length > 0) {
      const [peqMag, peqPhase] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq,
        bands: peqBands,
        sampleRate: sr,
      });
      peqMagArr = peqMag;
      modelPhase = modelPhase.map((v: number, i: number) => v + peqPhase[i]);
    }

    // 3. Generate FIR
    const isLin = (f: any) => !f || f.linear_phase || f.filter_type === "Gaussian";

    const firResult = await invoke<{
      impulse: number[]; time_ms: number[]; realized_mag: number[];
      realized_phase: number[]; taps: number; sample_rate: number; norm_db: number;
    }>("generate_model_fir", {
      freq,
      targetMag,
      peqMag: peqMagArr,
      modelPhase,
      config: {
        taps,
        sample_rate: sr,
        max_boost_db: 24.0,
        noise_floor_db: -150.0,
        window: win,
        phase_mode: (isLin(b.target.high_pass) && isLin(b.target.low_pass)) ? "LinearPhase" : "MinimumPhase",
      },
    });

    return firResult.impulse;
  }

  function bandFileName(b: BandState): string {
    const pName = projectName();
    const sr = exportSampleRate();
    const taps = exportTaps();
    if (pName) {
      // v2: ProjectName-YYMMDD-BandName-SR-Taps.wav
      return wavFileName(pName, b.name, sr, taps);
    }
    // Fallback (no project folder): legacy naming
    const safeName = sanitizeFilename(b.name);
    return `phaseforge_${safeName}_${taps}taps_${sr}hz.wav`;
  }

  // Export active band with save dialog
  async function handleExport() {
    const b = activeBand();
    if (!b) return;
    setExporting(true);
    setExportError(null);
    try {
      const impulse = await generateBandImpulse(b);
      const sr = exportSampleRate();
      const fileName = bandFileName(b);
      const dir = projectDir();
      const defPath = dir ? `${dir}/${fileName}` : fileName;

      const path = await save({
        defaultPath: defPath,
        filters: [{ name: "WAV", extensions: ["wav"] }],
      });
      if (!path) { setExporting(false); return; }

      await invoke("export_fir_wav", { impulse, sampleRate: sr, path });
    } catch (e) {
      console.error("Export failed:", e);
      setExportError(String(e));
    } finally {
      setExporting(false);
    }
  }

  const band = () => activeBand();

  return (
    <div class="export-tab">
      {/* Settings row: SR / Taps / Win */}
      <div class="export-settings-row">
        <label class="fb-inline-label">SR</label>
        <select
          class="tb-select"
          value={exportSampleRate()}
          onChange={(e) => setExportSampleRate(Number(e.currentTarget.value))}
        >
          {sampleRates.map((sr) => (
            <option value={sr}>{sr >= 1000 ? (sr / 1000) + "k" : sr}</option>
          ))}
        </select>

        <label class="fb-inline-label">Taps</label>
        <select
          class="tb-select"
          value={exportTaps()}
          onChange={(e) => setExportTaps(Number(e.currentTarget.value))}
        >
          {tapOptions.map((t) => (
            <option value={t}>{t >= 1024 ? (t / 1024) + "K" : t}</option>
          ))}
        </select>

        <label class="fb-inline-label">Win</label>
        <select
          class="tb-select"
          value={exportWindow()}
          onChange={(e) => setExportWindow(e.currentTarget.value as WindowType)}
        >
          {windowOptions.map((w) => (
            <option value={w}>{w}</option>
          ))}
        </select>

        <div style={{ "margin-left": "auto", display: "flex", "align-items": "center", gap: "6px" }}>
          <Show when={band()}>
            {(b) => (
              <>
                <span class="export-phase-badge" style={{ color: bandPhaseIsLinear(b()) ? "#22C55E" : "#FF9F43" }}>
                  {bandPhaseLabel(b())}
                </span>
                <span style={{ "font-size": "10px", color: "var(--text-secondary)", "font-family": "var(--mono)" }}>
                  {formatFilterInfo(b())}
                </span>
                <Show when={b().peqBands.length > 0}>
                  <span style={{ "font-size": "10px", color: "var(--text-secondary)" }}>
                    PEQ: {b().peqBands.filter((p: any) => p.enabled).length}
                  </span>
                </Show>
                <button
                  class="tb-btn primary"
                  style={{ padding: "4px 10px" }}
                  disabled={exporting()}
                  onClick={handleExport}
                >
                  {exporting() ? "Exporting..." : "Export WAV"}
                </button>
              </>
            )}
          </Show>
        </div>
      </div>

      <Show when={exportError()}>
        <div class="align-status" style={{ color: "#EF4444", padding: "4px 8px" }}>
          {exportError()}
        </div>
      </Show>
    </div>
  );
}
