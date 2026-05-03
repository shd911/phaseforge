import { createSignal, createEffect, on, onCleanup, Show } from "solid-js";
import type { FilterType, FilterConfig, Measurement, MergeConfig, MergeResult, PeqBand, FirConfig, FirResult, WindowType, PhaseMode } from "../lib/types";
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
  setBandFirResult,
  setBandMeasurementFile,
  activeTab,
  setActiveTab,
  exportSampleRate,
  setExportSampleRate,
  exportTaps,
  setExportTaps,
  exportWindow,
  setExportWindow,
  exportHybridPhase,
  setBandColor,
  selectedPeqIdx,
  setSelectedPeqIdx,
  addPeqBand,
  updatePeqBand,
  commitPeqBand,
  removePeqBand,
  addExclusionZone,
  removeExclusionZone,
  updateExclusionZone,
} from "../stores/bands";
import type { PresetName, SmoothingMode, MergeSource, BandState } from "../stores/bands";
import {
  firMaxBoost, firNoiseFloor, firIterations,
  firFreqWeighting, firNarrowbandLimit, firNbSmoothingOct, firNbMaxExcess,
} from "../stores/bands";
import { isGaussianMinPhase, gaussianFilterMagDb, CORRECTED_COLOR, PEQ_COLOR, STATUS_BAD } from "../lib/plot-helpers";
import {
  tolerance, setTolerance,
  maxBands, setMaxBands,
  gainRegularization, setGainRegularization,
  peqFloor, setPeqFloor,
  peqRangeMode, setPeqRangeMode,
  peqDirectLow, setPeqDirectLow,
  peqDirectHigh, setPeqDirectHigh,
  computing, peqError, maxErr, iters,
  peqRange, formatFreq,
  handleOptimizePeq, handleClearPeq, peqStale,
} from "../stores/peq-optimize";
import { showStaleConfirmDialog } from "./StalePeqExportDialog";
import { qWarnAt } from "../lib/peq-quality";
import { openHighQPopup } from "./HighQWarningPopup";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { setNeedAutoFit } from "../App";
import { projectDir, projectName, copyMeasurementToProject, copyMergeFilesToProject, sanitize, yymmdd } from "../lib/project-io";
import { handleImportMeasurement, handleMergeComplete } from "../lib/measurement-actions";
import MergeDialog from "./MergeDialog";

const FILTER_TYPES: FilterType[] = ["Butterworth", "Bessel", "LinkwitzRiley", "Gaussian", "Custom"];

// Track which inputs have been explicitly clicked — wheel only fires when in set
const wheelEnabled = new WeakSet<Element>();

// Remember last filter config per band so toggle off→on restores settings
const lastHP = new Map<string, import("../lib/types").FilterConfig>();
const lastLP = new Map<string, import("../lib/types").FilterConfig>();

export default function ControlPanel(props: { rightPanel?: boolean }) {
  const tab = activeTab;
  const setTab = setActiveTab;
  const band = () => activeBand();

  if (props.rightPanel) {
    // Right panel: Target + PEQ — no tabs, always visible, scrollable
    return (
      <div class="right-panel-content">
        <div class="right-panel-section">
          <div class="right-panel-title">Target</div>
          <FiltersTab />
        </div>
        <div class="right-panel-section">
          <div class="right-panel-title">PEQ</div>
          <PeqTab />
        </div>
      </div>
    );
  }

  // Bottom panel removed (b126) — controls moved to plot toolbar in FrequencyPlot
  return null;
}

// ---------------------------------------------------------------------------
// Filters Tab
// ---------------------------------------------------------------------------

/** Deep-copy a FilterConfig from SolidJS store proxy to a plain object.
 *  This breaks the proxy reference so that spread/read in event handlers
 *  never accidentally subscribes to or cross-contaminates sibling paths. */
function unwrapFilter(f: import("../lib/types").FilterConfig | null | undefined): import("../lib/types").FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
  };
}

function FiltersTab() {
  const band = () => activeBand();
  const target = () => band()?.target;
  const enabled = () => band()?.targetEnabled ?? false;
  const inverted = () => band()?.inverted ?? false;
  const bandId = () => band()?.id;

  // PEQ Auto-Fit moved to peq-optimize.ts + PeqSidebar.tsx (b82.06)

  async function handleExportTarget() {
    const b = band();
    if (!b) return;
    try {
      const [freq, response] = await invoke<[number[], { magnitude: number[] }]>(
        "evaluate_target_standalone",
        { target: b.target, nPoints: 512, fMin: 10, fMax: 24000 }
      );
      const dir = projectDir();
      if (dir) await invoke("ensure_dir", { path: `${dir}/target` }).catch(() => {});
      const defName = `${sanitize(driverName(b))}_target.txt`;
      const defPath = dir ? `${dir}/target/${defName}` : defName;
      const filePath = await save({
        defaultPath: defPath,
        filters: [{ name: "REW TXT", extensions: ["txt"] }],
      });
      if (!filePath) return;
      await invoke("export_target_txt", { freq, magnitude: response.magnitude, path: filePath });
    } catch (e) {
      console.error("Export target failed:", e);
    }
  }

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
          isHighPass={true}
          config={unwrapFilter(target()?.high_pass)}
          linked={hpLinked()}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            const cur = unwrapFilter(target()?.high_pass);
            if (cur) {
              lastHP.set(id, cur);
              setBandHighPass(id, null);
            } else {
              setBandHighPass(id, lastHP.get(id) ?? { filter_type: "Butterworth", order: 2, freq_hz: 80, shape: null, linear_phase: false, q: null, subsonic_protect: null });
            }
          }}
          onChange={(c) => { const id = bandId(); if (id) setBandHighPass(id, c); }}
        />

        {/* Low-Pass (🔗 кнопка, если не последняя полоса) */}
        <FilterBlock
          title="Low-Pass"
          config={unwrapFilter(target()?.low_pass)}
          linked={linked()}
          canLink={!isLastBand()}
          onLinkToggle={() => { const id = bandId(); if (id) toggleBandLinked(id); }}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            const cur = unwrapFilter(target()?.low_pass);
            if (cur) {
              lastLP.set(id, cur);
              setBandLowPass(id, null);
            } else {
              setBandLowPass(id, lastLP.get(id) ?? { filter_type: "Butterworth", order: 2, freq_hz: 15000, shape: null, linear_phase: false, q: null });
            }
          }}
          onChange={(c) => { const id = bandId(); if (id) setBandLowPass(id, c); }}
        />


        {/* Export Target — rightmost in the row */}
        <button class="tb-btn tb-btn-sm" style={{ "align-self": "flex-start", "margin-left": "auto" }} onClick={handleExportTarget}>
          Export Target
        </button>
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
  isHighPass?: boolean; // b138: subsonic-protect UI for Gaussian HP only
}

function FilterBlock(props: FilterBlockProps) {
  const c = () => props.config;
  const isGaussian = () => c()?.filter_type === "Gaussian";
  const isCustom = () => c()?.filter_type === "Custom";

  /** Build a full FilterConfig from the current config, overriding specific fields.
   *  Reads each field explicitly from the (already unwrapped) plain config object
   *  to avoid any SolidJS proxy spread issues. */
  const withOverride = (overrides: Partial<import("../lib/types").FilterConfig>): import("../lib/types").FilterConfig => {
    const cur = c()!;
    return {
      filter_type: cur.filter_type,
      order: cur.order,
      freq_hz: cur.freq_hz,
      shape: cur.shape,
      linear_phase: cur.linear_phase,
      q: cur.q,
      subsonic_protect: cur.subsonic_protect ?? null,
      ...overrides,
    };
  };

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
            ><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={props.linked ? CORRECTED_COLOR : STATUS_BAD} stroke-width="2.5" stroke-linecap="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg></button>
          </Show>
          <Show when={!props.canLink && props.linked}>
            <span class="fb-link-indicator" title="Linked to adjacent band"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={CORRECTED_COLOR} stroke-width="2.5" stroke-linecap="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg></span>
          </Show>
        </span>
        <button
          class={`fb-toggle ${c() ? "on" : ""}`}
          onClick={props.onToggle}
        >{c() ? "ON" : "OFF"}</button>
      </div>
      <Show when={c()}>
        <div class="fb-grid">
          <div class="fb-row">
            <label class="fb-label">Type</label>
            <select
              class="fb-select"
              value={c()!.filter_type}
              onChange={(e) => {
                const ft = e.currentTarget.value as FilterType;
                if (ft === c()!.filter_type) return; // guard: SolidJS/WebKit programmatic set
                // b138: subsonic-protect auto-on for Gaussian HP, dropped otherwise.
                const subsonic = ft === "Gaussian" && props.isHighPass ? true : null;
                if (ft === "Gaussian") {
                  props.onChange(withOverride({ filter_type: ft, shape: c()!.shape ?? 1.0, q: null, subsonic_protect: subsonic }));
                } else if (ft === "Custom") {
                  props.onChange(withOverride({ filter_type: ft, shape: null, q: c()!.q ?? 0.707, subsonic_protect: null }));
                } else {
                  props.onChange(withOverride({ filter_type: ft, shape: null, q: null, subsonic_protect: null }));
                }
              }}
            >
              {FILTER_TYPES.map((t) => <option value={t}>{t}</option>)}
            </select>
          </div>
          <div class="fb-row">
            <label class="fb-label">Freq</label>
            <NumberInput
              value={c()!.freq_hz}
              onChange={(v) => props.onChange(withOverride({ freq_hz: v }))}
              min={10} max={20000} step={1} unit="Hz" freqMode
            />
          </div>
          <Show when={!isGaussian()}>
            <div class="fb-row">
              <label class="fb-label">Order</label>
              <NumberInput
                value={c()!.order}
                onChange={(v) => props.onChange(withOverride({ order: v }))}
                min={1} max={8} step={1} precision={0}
              />
            </div>
          </Show>
          <Show when={isGaussian()}>
            <div class="fb-row">
              <label class="fb-label">M</label>
              <NumberInput
                value={c()!.shape ?? 1.0}
                onChange={(v) => props.onChange(withOverride({ shape: v }))}
                min={0.5} max={10} step={0.1}
              />
            </div>
          </Show>
          <Show when={isCustom()}>
            <div class="fb-row">
              <label class="fb-label">Q</label>
              <NumberInput
                value={c()!.q ?? 0.707}
                onChange={(v) => props.onChange(withOverride({ q: v }))}
                min={0.1} max={5.0} step={0.01}
              />
            </div>
          </Show>
          <div class="fb-row">
            <label class="fb-label"></label>
            <span class="fb-checkbox" title="Linear phase (magnitude only, no phase rotation)"
              onClick={() => props.onChange(withOverride({ linear_phase: !c()!.linear_phase }))}>
              <span class={`fb-check-box ${c()!.linear_phase ? "checked" : ""}`} />
              <span class="fb-check-label">Lin-φ</span>
            </span>
          </div>
        </div>
        <Show when={isCustom()}>
          <div class="fb-hint">0.50=LR · 0.58=Bsl · 0.71=BW</div>
        </Show>
        <Show when={props.isHighPass && isGaussian()}>
          <div class="subsonic-protect-row">
            <label
              title="Минимально-фазовый Butterworth 48 дБ/окт на 3 октавы ниже HP. Защищает driver от излишнего excursion в инфразвуке."
              style={{ display: "flex", "align-items": "center", gap: "6px", cursor: c()!.freq_hz > 40 ? "pointer" : "not-allowed" }}
            >
              <input
                type="checkbox"
                checked={c()!.subsonic_protect === true}
                disabled={c()!.freq_hz <= 40}
                onChange={(e) => props.onChange(withOverride({ subsonic_protect: e.currentTarget.checked }))}
              />
              <span>Защитный subsonic фильтр</span>
            </label>
            <Show when={c()!.freq_hz <= 40}>
              <span class="hint" title="HP слишком низкий, защита не требуется">⊘</span>
            </Show>
            <Show when={c()!.subsonic_protect === false && c()!.freq_hz > 40}>
              <div class="warn">⚠ Защита отключена, риск excursion на инфразвуке</div>
            </Show>
          </div>
        </Show>
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

  // Both delegate to centralized handlers in measurement-actions.ts so the
  // post-import analysis runs uniformly. Keeping local thin wrappers makes
  // the JSX wiring (onClick, onMerge) unchanged.
  async function handleImport() {
    await handleImportMeasurement();
  }

  async function handleMergeCompleteLocal(measurement: Measurement, source: MergeSource) {
    await handleMergeComplete(measurement, source);
  }

  async function handleToggleDelay() {
    const b = band();
    if (!b || !b.measurement?.phase || !b.settings) return;
    if (b.settings.delay_removed) {
      restoreBandDelay(b.id);
    } else {
      try {
        const origPhase = b.settings.originalPhase ?? b.measurement.phase;
        const [newPhase, delay, distance] = await invoke<[number[], number, number]>(
          "remove_measurement_delay",
          { freq: b.measurement.freq, magnitude: b.measurement.magnitude, phase: origPhase, sampleRate: b.measurement.sample_rate }
        );
        setBandDelayInfo(b.id, delay, distance);
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
              <td class="color-dot-cell">
                <label class="color-dot-label">
                  <span
                    class="color-dot"
                    style={{ "background-color": band()?.color ?? MEASUREMENT_COLORS[0] }}
                  />
                  <input
                    type="color"
                    class="color-dot-input"
                    value={band()?.color ?? MEASUREMENT_COLORS[0]}
                    onInput={(e) => {
                      const b = band();
                      if (b) setBandColor(b.id, e.currentTarget.value);
                    }}
                  />
                </label>
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
                  <span style={{ display: "inline-flex", "align-items": "center", "min-width": "75px", visibility: s()?.delay_removed && s()?.delay_seconds != null ? "visible" : "hidden" }}>
                    <NumberInput
                      value={parseFloat(((s()!.delay_seconds ?? 0) * 1000).toFixed(2))}
                      onChange={async (ms) => {
                        const b = band();
                        if (!b || !b.measurement?.phase || !b.settings) return;
                        const delaySec = ms / 1000;
                        const origPhase = b.settings.originalPhase ?? b.measurement.phase;
                        try {
                          const newPhase = await invoke<number[]>("apply_manual_delay", {
                            freq: b.measurement.freq, phase: origPhase, delaySeconds: delaySec,
                          });
                          setBandDelayInfo(b.id, delaySec, delaySec * 343);
                          markBandDelayRemoved(b.id, newPhase);
                        } catch (e) { console.error("Manual delay failed:", e); }
                      }}
                      min={0} max={50} step={0.01} precision={2}
                    />
                    <span style={{ "font-size": "var(--fs-xs)", color: "#8b8b96", "margin-left": "var(--space-xxs)" }}>ms</span>
                  </span>
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
          onMerge={handleMergeCompleteLocal}
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
  onCleanup(() => clearTimeout(debounceTimer));

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
            magnitude: result.measurement.magnitude,
            phase: result.measurement.phase,
          });
          setBandDelayInfo(band.id, delay, distance);

          // Если delay comp был включён — повторно применяем на новой фазе
          if (wasDelayRemoved) {
            // Сбросить originalPhase чтобы markBandDelayRemoved сохранила новую сырую фазу
            resetBandOriginalPhase(band.id);
            const [newPhase] = await invoke<[number[], number, number]>(
              "remove_measurement_delay",
              { freq: result.measurement.freq, magnitude: result.measurement.magnitude, phase: result.measurement.phase, sampleRate: result.measurement.sample_rate }
            );
            markBandDelayRemoved(band.id, newPhase);
          }
        } catch (e) { console.warn("Delay recomputation after re-merge failed:", e); }
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
// Shared helpers
// ---------------------------------------------------------------------------

/** Extract driver/band label from measurement name, stripping REW "Band N · " prefix */
function driverName(b: BandState): string {
  let name = b.measurement?.name ?? b.name;
  const dotIdx = name.indexOf("·");
  if (dotIdx >= 0) name = name.substring(dotIdx + 1).trim();
  return name;
}

// ---------------------------------------------------------------------------
// PEQ Tab — migrated from PeqSidebar
// ---------------------------------------------------------------------------

function PeqTab() {
  const band = () => activeBand();
  const peqBands = () => band()?.peqBands ?? [];
  const [pendingPeqIdx, setPendingPeqIdx] = createSignal<number | null>(null);

  createEffect(() => { peqBands(); setPendingPeqIdx(null); });

  function handleRemovePeq(idx: number) {
    const b = band();
    if (!b) return;
    if (selectedPeqIdx() === idx) setSelectedPeqIdx(null);
    else if (selectedPeqIdx() != null && selectedPeqIdx()! > idx)
      setSelectedPeqIdx(selectedPeqIdx()! - 1);
    removePeqBand(b.id, idx);
  }

  const isStale = () => {
    const b = band();
    return b ? peqStale(b) : false;
  };

  return (
    <div
      class="peq-tab-content"
      classList={{ "peq-tab-stale": isStale() }}
    >
      <Show when={isStale()}>
        <div class="peq-stale-banner">
          <span>⚠ PEQ устарел: target изменён после последней оптимизации</span>
          <span style={{ display: "flex", gap: "var(--space-xs)" }}>
            <button
              class="tb-btn tb-btn-sm"
              onClick={handleOptimizePeq}
              disabled={computing() || !band()?.measurement}
            >Переоптимизировать</button>
            <button class="tb-btn tb-btn-sm" onClick={handleClearPeq}>
              Очистить
            </button>
          </span>
        </div>
      </Show>

      {/* Exclusion Zones — yellow */}
      <Show when={band()?.measurement}>
        <div class="peq-exclusion-section">
          <div class="peq-sidebar-header">
            <span class="fb-title" style={{ "font-size": "var(--fs-base)" }}>Exclude</span>
            <button class="peq-add-btn" onClick={() => { const b = band(); if (b) addExclusionZone(b.id, { startHz: 100, endHz: 200 }); }} title="Add exclusion zone">+</button>
          </div>
          <Show when={(band()?.exclusionZones?.length ?? 0) > 0}>
            <table class="peq-table peq-excl-table">
              <thead><tr><th>From</th><th>To</th><th></th></tr></thead>
              <tbody>
                {(band()?.exclusionZones ?? []).map((z, i) => (
                  <tr>
                    <td><input class="peq-input" type="number" value={Math.round(z.startHz)} min={20} max={20000} step={1} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v) && v >= 20) { const b = band(); if (b) updateExclusionZone(b.id, i, { startHz: v }); } }} /></td>
                    <td><input class="peq-input" type="number" value={Math.round(z.endHz)} min={20} max={20000} step={1} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v) && v >= 20) { const b = band(); if (b) updateExclusionZone(b.id, i, { endHz: v }); } }} /></td>
                    <td><button class="peq-remove" onClick={() => { const b = band(); if (b) removeExclusionZone(b.id, i); }}>×</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Show>
        </div>
      </Show>

      {/* Manual PEQ — blue */}
      <div class="peq-manual-section">
        <div class="peq-sidebar-header">
          <span class="fb-title" style={{ "font-size": "var(--fs-base)" }}>Manual</span>
          <button class="peq-add-btn" onClick={() => {
            const b = band();
            if (b) {
              if (pendingPeqIdx() != null) commitPeqBand(b.id, pendingPeqIdx()!);
              addPeqBand(b.id, { freq_hz: 1000, gain_db: 0, q: 2.0, enabled: true, filter_type: "Peaking" });
              setPendingPeqIdx(0); setSelectedPeqIdx(0);
            }
          }} title="Add manual PEQ band">+</button>
        </div>
        <Show when={pendingPeqIdx() != null && peqBands().length > 0}>
          {(() => {
            const pi = pendingPeqIdx()!;
            const pb = peqBands()[pi];
            if (!pb) return null;
            return (
              <table class="peq-table"><tbody>
                <tr class="peq-row-pending peq-row-selected" onClick={() => setSelectedPeqIdx(pi)}>
                  <td><input type="checkbox" class="peq-toggle" checked={pb.enabled} onChange={() => { const bd = band(); if (bd) updatePeqBand(bd.id, pi, { enabled: !pb.enabled }); }} /></td>
                  <td><select class="peq-type-select" value={pb.filter_type} onChange={(e) => { const bd = band(); if (bd) updatePeqBand(bd.id, pi, { filter_type: e.currentTarget.value as any }); }}><option value="Peaking">PK</option><option value="LowShelf">LS</option><option value="HighShelf">HS</option></select></td>
                  <td><input class="peq-input" type="number" value={Math.round(pb.freq_hz)} min={20} max={20000} step={1} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v) && v >= 20 && v <= 20000) { const bd = band(); if (bd) updatePeqBand(bd.id, pi, { freq_hz: v }); } }} /></td>
                  <td><input class={`peq-input ${pb.gain_db > 0 ? "peq-boost" : "peq-cut"}`} type="number" value={pb.gain_db.toFixed(1)} min={-60} max={60} step={0.1} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v)) { const bd = band(); if (bd) updatePeqBand(bd.id, pi, { gain_db: v }); } }} /></td>
                  <td><input class="peq-input" type="number" value={pb.q.toFixed(1)} min={0.1} max={20} step={0.1} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v) && v >= 0.1 && v <= 20) { const bd = band(); if (bd) updatePeqBand(bd.id, pi, { q: v }); } }} /></td>
                  <td><button class="peq-commit" onClick={() => { const bd = band(); if (bd) { const ni = commitPeqBand(bd.id, pi); setPendingPeqIdx(null); setSelectedPeqIdx(ni); } }}>✓</button></td>
                </tr>
              </tbody></table>
            );
          })()}
        </Show>
      </div>

      {/* Auto Optimizer — green */}
      <Show when={band()?.measurement}>
        <div class="peq-auto-section">
          <div class="peq-sidebar-header">
            <span class="fb-title" style={{ "font-size": "var(--fs-base)" }}>Auto Optimizer</span>
          </div>
          <div class="peq-grid">
            <div class="fb-row"><label class="fb-label">Tolerance</label><NumberInput value={tolerance()} onChange={setTolerance} min={0.5} max={3.0} step={0.1} unit="dB" /></div>
            <div class="fb-row"><label class="fb-label">Max bands</label><NumberInput value={maxBands()} onChange={(v: number) => setMaxBands(Math.round(v))} min={1} max={60} step={1} precision={0} /></div>
            <div class="fb-row"><label class="fb-label">Regularization</label><NumberInput value={gainRegularization()} onChange={setGainRegularization} min={0} max={1} step={0.0001} precision={4} /></div>
            <div class="fb-row">
              <label class="fb-label">Range</label>
              <select class="peq-range-select" value={peqRangeMode()} onChange={(e) => setPeqRangeMode(e.currentTarget.value as "auto" | "direct")}><option value="auto">Auto</option><option value="direct">Direct</option></select>
            </div>
            {peqRangeMode() === "auto" ? (
              <div class="fb-row"><label class="fb-label" title="Don't place PEQ where target is this many dB below reference level">Floor dB</label><NumberInput value={peqFloor()} onChange={setPeqFloor} min={0} max={120} step={1} precision={0} /></div>
            ) : (
              <div class="fb-row"><label class="fb-label">Hz</label><NumberInput value={peqDirectLow()} onChange={setPeqDirectLow} min={20} max={20000} step={10} precision={0} /><span style={{ margin: "0 var(--space-xxs)", color: "#8b8b96" }}>–</span><NumberInput value={peqDirectHigh()} onChange={setPeqDirectHigh} min={20} max={20000} step={10} precision={0} /></div>
            )}
          </div>
          <div class="peq-buttons-row">
            <span class="align-range-info">{formatFreq(peqRange()[0])}{"\u2013"}{formatFreq(peqRange()[1])} Hz</span>
            <button class="tb-btn primary" onClick={handleOptimizePeq} disabled={computing()}>{computing() ? "..." : "Optimize"}</button>
            <Show when={peqBands().length > 0}><button class="tb-btn" onClick={handleClearPeq}>Clear</button></Show>
          </div>
          <Show when={peqError()}><div class="align-error">{peqError()}</div></Show>
          <Show when={peqBands().length > 0}>
            <div class="align-status">
              {peqBands().length} band{peqBands().length > 1 ? "s" : ""}
              {maxErr() != null ? ` \u00B7 max: ${maxErr()!.toFixed(1)}dB` : ""}
              {iters() != null ? ` \u00B7 ${iters()}it` : ""}
            </div>
          </Show>
          <Show when={peqBands().length > 0}>
            <div class="peq-sidebar-table-scroll">
              <table class="peq-table">
                <thead><tr><th></th><th>Type</th><th>Freq</th><th>Gain</th><th>Q</th><th></th><th></th></tr></thead>
                <tbody>
                  {peqBands().map((b, i) => {
                    const isPending = pendingPeqIdx() === i;
                    const peqWheel = (e: WheelEvent, field: "freq_hz" | "gain_db" | "q") => {
                      if (!wheelEnabled.has(e.currentTarget as Element)) { e.preventDefault(); return; }
                      e.preventDefault(); e.stopPropagation();
                      const bd = band(); if (!bd) return;
                      const dir = e.deltaY < 0 ? 1 : -1;
                      if (field === "freq_hz") {
                        const step = Math.max(1, Math.round(b.freq_hz * 0.02));
                        const v = Math.max(20, Math.min(20000, b.freq_hz + dir * step));
                        updatePeqBand(bd.id, i, { freq_hz: v });
                      } else if (field === "gain_db") {
                        const v = Math.round((b.gain_db + dir * 0.1) * 10) / 10;
                        updatePeqBand(bd.id, i, { gain_db: v });
                      } else {
                        const v = Math.max(0.1, Math.min(20, Math.round((b.q + dir * 0.1) * 10) / 10));
                        updatePeqBand(bd.id, i, { q: v });
                      }
                    };
                    return (
                      <tr class={`${selectedPeqIdx() === i ? "peq-row-selected" : ""} ${isPending ? "peq-row-pending" : ""} ${!b.enabled ? "peq-row-disabled" : ""}`}
                        onClick={() => setSelectedPeqIdx(selectedPeqIdx() === i ? null : i)}>
                        <td><input type="checkbox" class="peq-toggle" checked={b.enabled} onChange={(e) => { e.stopPropagation(); const bd = band(); if (bd) updatePeqBand(bd.id, i, { enabled: !b.enabled }); }} onClick={(e) => e.stopPropagation()} /></td>
                        <td><select class="peq-type-select" value={b.filter_type ?? "Peaking"} onChange={(e) => { e.stopPropagation(); const bd = band(); if (bd) updatePeqBand(bd.id, i, { filter_type: e.currentTarget.value as any }); }} onClick={(e) => e.stopPropagation()}><option value="Peaking">PK</option><option value="LowShelf">LS</option><option value="HighShelf">HS</option></select></td>
                        <td><input class="peq-input" type="number" value={Math.round(b.freq_hz)} min={20} max={20000} step={1} onWheel={(e) => peqWheel(e, "freq_hz")} onPointerDown={(e) => wheelEnabled.add(e.currentTarget)} onBlur={(e) => wheelEnabled.delete(e.currentTarget)} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v) && v >= 20 && v <= 20000) { const bd = band(); if (bd) updatePeqBand(bd.id, i, { freq_hz: v }); } }} /></td>
                        <td><input class={`peq-input ${b.gain_db > 0 ? "peq-boost" : "peq-cut"}`} type="number" value={b.gain_db.toFixed(1)} min={exportHybridPhase() ? -60 : -18} max={exportHybridPhase() ? 60 : 6} step={0.1} onWheel={(e) => peqWheel(e, "gain_db")} onPointerDown={(e) => wheelEnabled.add(e.currentTarget)} onBlur={(e) => wheelEnabled.delete(e.currentTarget)} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v)) { const bd = band(); if (bd) updatePeqBand(bd.id, i, { gain_db: v }); } }} /></td>
                        <td><input class="peq-input" type="number" value={b.q.toFixed(1)} min={0.1} max={20} step={0.1} onWheel={(e) => peqWheel(e, "q")} onPointerDown={(e) => wheelEnabled.add(e.currentTarget)} onBlur={(e) => wheelEnabled.delete(e.currentTarget)} onChange={(e) => { const v = parseFloat(e.currentTarget.value); if (!isNaN(v) && v >= 0.1 && v <= 20) { const bd = band(); if (bd) updatePeqBand(bd.id, i, { q: v }); } }} /></td>
                        <td>{b.enabled && b.q > qWarnAt(b.freq_hz) ? (
                          <button class="peq-warn-icon" title="" aria-label="Высокая добротность"
                            onClick={(e) => { e.stopPropagation(); openHighQPopup(b, i); }}>⚠</button>
                        ) : null}</td>
                        <td>{isPending ? <button class="peq-commit" onClick={(e) => { e.stopPropagation(); const bd = band(); if (bd) { const ni = commitPeqBand(bd.id, i); setPendingPeqIdx(null); setSelectedPeqIdx(ni); } }}>✓</button> : <button class="peq-remove" onClick={() => handleRemovePeq(i)}>×</button>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Show>
        </div>
      </Show>
    </div>
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
  const windowGroups: { label: string; options: { value: WindowType; label: string }[] }[] = [
    { label: "Basic", options: [
      { value: "Rectangular", label: "Rectangular" },
      { value: "Bartlett", label: "Bartlett" },
      { value: "Hann", label: "Hann" },
      { value: "Hamming", label: "Hamming" },
      { value: "Blackman", label: "Blackman" },
    ]},
    { label: "Blackman-Harris", options: [
      { value: "ExactBlackman", label: "Exact Blackman" },
      { value: "BlackmanHarris", label: "Blackman-Harris" },
      { value: "Nuttall3", label: "Nuttall 3-term" },
      { value: "Nuttall4", label: "Nuttall 4-term" },
      { value: "FlatTop", label: "Flat Top" },
    ]},
    { label: "Parametric", options: [
      { value: "Kaiser", label: "Kaiser (\u03B2=10)" },
      { value: "DolphChebyshev", label: "Dolph-Chebyshev" },
      { value: "Gaussian", label: "Gaussian (\u03C3=2.5)" },
      { value: "Tukey", label: "Tukey (\u03B1=0.5)" },
    ]},
    { label: "Special", options: [
      { value: "Lanczos", label: "Lanczos" },
      { value: "Poisson", label: "Poisson" },
      { value: "HannPoisson", label: "Hann-Poisson" },
      { value: "Bohman", label: "Bohman" },
      { value: "Cauchy", label: "Cauchy" },
      { value: "Riesz", label: "Riesz" },
    ]},
  ];

  // Determine phase mode from target filters (HP/LP linear_phase flags)
  const isFilterLinear = (f: import("../lib/types").FilterConfig | null | undefined) =>
    !f || f.linear_phase;

  const bandPhaseLabel = (b: BandState) => {
    if (exportHybridPhase() && b.measurement) return "Hybrid-\u03C6";
    if (!b.target) return "Linear";
    const lin = isFilterLinear(b.target.high_pass) && isFilterLinear(b.target.low_pass);
    return lin ? "Linear" : "Min-\u03C6";
  };

  const bandPhaseIsLinear = (b: BandState) => {
    if (exportHybridPhase() && b.measurement) return false; // hybrid: not linear
    if (!b.target) return true;
    return isFilterLinear(b.target.high_pass) && isFilterLinear(b.target.low_pass);
  };

  const bandPhaseColor = (b: BandState) => {
    if (exportHybridPhase() && b.measurement) return "#60A5FA"; // blue for hybrid
    return bandPhaseIsLinear(b) ? CORRECTED_COLOR : PEQ_COLOR;
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


  // Core: generate FIR impulse for a band, return impulse array.
  // Export is ALWAYS target + PEQ (model FIR), regardless of hybrid strategy.
  // Hybrid only affects PEQ optimization stage, not FIR generation.
  async function generateBandImpulse(b: BandState): Promise<number[]> {
    const sr = exportSampleRate();
    const taps = exportTaps();
    const win = exportWindow();
    const peqBands = b.peqBands?.filter((p: PeqBand) => p.enabled) ?? [];

    // 1. Evaluate pure target (HP/LP/shelf/tilt)
    const [freq, response] = await invoke<[number[], { magnitude: number[]; phase: number[] }]>(
      "evaluate_target_standalone",
      { target: { ...b.target }, nPoints: 512, fMin: 5, fMax: 40000 }
    );

    const targetMag = response.magnitude;

    // 2. Compute PEQ contribution separately (PEQ always min-phase)
    let peqMagArr: number[] = [];
    if (peqBands.length > 0) {
      const [peqMag, peqPhase] = await invoke<[number[], number[]]>("compute_peq_complex", {
        freq,
        bands: peqBands,
        sampleRate: sr,
      });
      peqMagArr = peqMag;
    }

    // 3. Generate FIR
    const isLin = (f: FilterConfig | null | undefined) => !f || f.linear_phase;

    const firResult = await invoke<{
      impulse: number[]; time_ms: number[]; realized_mag: number[];
      realized_phase: number[]; taps: number; sample_rate: number; norm_db: number;
    }>("generate_model_fir", {
      freq,
      targetMag,
      peqMag: peqMagArr,
      modelPhase: new Array(freq.length).fill(0),
      config: {
        taps,
        sample_rate: sr,
        max_boost_db: firMaxBoost(),
        noise_floor_db: firNoiseFloor(),
        window: win,
        phase_mode: (isLin(b.target.high_pass) && isLin(b.target.low_pass)) ? "LinearPhase" : "MinimumPhase",
        iterations: firIterations(),
        freq_weighting: firFreqWeighting(),
        narrowband_limit: firNarrowbandLimit(),
        nb_smoothing_oct: firNbSmoothingOct(),
        nb_max_excess_db: firNbMaxExcess(),
      },
    });

    return firResult.impulse;
  }

  function bandFileName(b: BandState): string {
    const sr = exportSampleRate();
    const taps = exportTaps();
    const win = exportWindow();
    return `${sanitize(driverName(b))}_${sr}_${taps}_${win}.wav`;
  }

  // Export active band with save dialog
  async function handleExport() {
    const b = activeBand();
    if (!b) return;
    if (peqStale(b)) {
      const proceed = await showStaleConfirmDialog([b.name]);
      if (!proceed) return;
    }
    setExporting(true);
    setExportError(null);
    try {
      const impulse = await generateBandImpulse(b);
      const sr = exportSampleRate();
      const fileName = bandFileName(b);
      const dir = projectDir();
      if (dir) await invoke("ensure_dir", { path: `${dir}/export` }).catch(() => {});
      const defPath = dir ? `${dir}/export/${fileName}` : fileName;

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
          {windowGroups.map((g) => (
            <optgroup label={g.label}>
              {g.options.map((w) => (
                <option value={w.value}>{w.label}</option>
              ))}
            </optgroup>
          ))}
        </select>

        <div style={{ "margin-left": "auto", display: "flex", "align-items": "center", gap: "var(--space-sm)" }}>
          <Show when={band()}>
            {(b) => (
              <>
                <span class="export-phase-badge" style={{ color: bandPhaseColor(b()) }}>
                  {bandPhaseLabel(b())}
                </span>
                <span style={{ "font-size": "var(--fs-sm)", color: "var(--text-secondary)", "font-family": "var(--mono)" }}>
                  {formatFilterInfo(b())}
                </span>
                <Show when={b().peqBands.length > 0}>
                  <span style={{ "font-size": "var(--fs-sm)", color: "var(--text-secondary)" }}>
                    PEQ: {b().peqBands.filter((p: PeqBand) => p.enabled).length}
                  </span>
                </Show>
                <button
                  class="tb-btn primary"
                  style={{ padding: "var(--space-xs) var(--space-lg)" }}
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
        <div class="align-status" style={{ color: STATUS_BAD, padding: "var(--space-xs) var(--space-md)" }}>
          {exportError()}
        </div>
      </Show>
    </div>
  );
}
