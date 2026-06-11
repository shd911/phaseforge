import { createSignal, createEffect, on, onCleanup, Show } from "solid-js";
import type { FilterType, FilterConfig, Measurement, MergeConfig, MergeResult, PeqBand, FirConfig, FirResult, WindowType, PhaseMode } from "../lib/types";
import NumberInput from "./NumberInput";
import { MEASUREMENT_COLORS, cloneFilterConfig } from "../lib/types";
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
import { evaluateBandFull } from "../lib/band-evaluator";
import { orderToSlope, slopeToOrder, availableSlopes } from "../lib/slope";
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
import { showToast } from "../lib/toast";
import { hasActiveSubsonicProtect } from "../lib/types";
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
    <Show when={band()} fallback={<p class="meas-empty">Нет активного бэнда. Выберите вкладку бэнда выше.</p>}>
      <div class="filters-row">
        {/* General */}
        <div class="filter-block">
          <div class="fb-header">
            <span class="fb-title">Общие</span>
            <button
              class={`fb-toggle ${enabled() ? "on" : ""}`}
              onClick={() => { const id = bandId(); if (id) toggleBandTarget(id); }}
              title="Включить/выключить цель для этого бэнда"
            >{enabled() ? "ON" : "OFF"}</button>
          </div>
          <div class="fb-row">
            <label class="fb-label">Пресет</label>
            <select
              class="fb-select"
              onChange={(e) => {
                const id = bandId();
                if (id) setBandPreset(id, e.currentTarget.value as PresetName);
              }}
            >
              <option value="flat">Плоский</option>
              <option value="harman">Harman</option>
              <option value="bk">B&K</option>
              <option value="x-curve">X-Curve</option>
              <option value="custom">Свой</option>
            </select>
          </div>
          <div class="fb-row">
            <label class="fb-label">Уровень</label>
            <NumberInput
              value={target()?.reference_level_db ?? 0}
              onChange={(v) => { const id = bandId(); if (id) setBandReferenceLevel(id, v); }}
              min={-20} max={20} step={0.5} unit="dB"
            />
          </div>
          <div class="fb-row">
            <label class="fb-label">Наклон</label>
            <NumberInput
              value={target()?.tilt_db_per_octave ?? 0}
              onChange={(v) => { const id = bandId(); if (id) setBandTilt(id, v); }}
              min={-6} max={6} step={0.1} unit="dB/oct"
            />
          </div>
          <div class="fb-row">
            <label class="fb-label">Инверсия</label>
            <button
              class={`fb-toggle invert-toggle ${inverted() ? "on" : ""}`}
              onClick={() => { const id = bandId(); if (id) toggleBandInverted(id); }}
              title="Инверсия полярности (NOR — нормальная, INV — инвертированная)"
            >{inverted() ? "INV" : "NOR"}</button>
          </div>
        </div>

        {/* High-Pass (🔗 индикатор, если связан с LP предыдущей) */}
        <FilterBlock
          title="ФВЧ"
          isHighPass={true}
          config={cloneFilterConfig(target()?.high_pass)}
          linked={hpLinked()}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            const cur = cloneFilterConfig(target()?.high_pass);
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
          title="ФНЧ"
          config={cloneFilterConfig(target()?.low_pass)}
          linked={linked()}
          canLink={!isLastBand()}
          onLinkToggle={() => { const id = bandId(); if (id) toggleBandLinked(id); }}
          onToggle={() => {
            const id = bandId();
            if (!id) return;
            const cur = cloneFilterConfig(target()?.low_pass);
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
          Экспорт цели
        </button>
      </div>
    </Show>
  );
}

// b140.7.13/b140.8.2: Slope ↔ Order helpers extracted to lib/slope.ts for testing.

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

  /** Build a full FilterConfig from the current config, overriding specific
   *  fields. Cloning first via `cloneFilterConfig` breaks the SolidJS store
   *  proxy reference so the subsequent spread of `overrides` is safe.
   *  b140.15.3: null-guard added — `c()` can be null when the band has no
   *  filter on this slot; the legacy code crashed with TypeError on `c()!`,
   *  the post-b140.11 unaware code returned a malformed object. Now it
   *  throws an explicit Error that the caller (always an onClick / onChange
   *  handler inside a Show-when block) should never reach. */
  const withOverride = (overrides: Partial<import("../lib/types").FilterConfig>): import("../lib/types").FilterConfig => {
    const cur = c();
    if (!cur) throw new Error("withOverride called with null FilterConfig — caller must guard on c() first");
    return { ...cloneFilterConfig(cur), ...overrides };
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
              <label class="fb-label">Slope</label>
              <select
                class="fb-select"
                onChange={(e) => {
                  const slope = parseInt(e.currentTarget.value, 10);
                  const order = slopeToOrder(c()!.filter_type, slope);
                  if (order !== c()!.order) {
                    props.onChange(withOverride({ order }));
                  }
                }}
              >
                {availableSlopes(c()!.filter_type).map((s) => (
                  <option
                    value={String(s)}
                    selected={s === orderToSlope(c()!.filter_type, c()!.order)}
                  >{`${s} dB/oct`}</option>
                ))}
              </select>
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
              title="Минимально-фазовый Butterworth 48 дБ/окт на 3 октавы ниже ФВЧ. Защищает динамик от излишнего хода диффузора в инфразвуке."
              style={{ display: "flex", "align-items": "center", gap: "6px", cursor: c()!.freq_hz > 40 ? "pointer" : "not-allowed" }}
            >
              <input
                type="checkbox"
                checked={c()!.subsonic_protect === true}
                disabled={c()!.freq_hz <= 40}
                onChange={() => {
                  const newValue = !(c()!.subsonic_protect === true);
                  props.onChange(withOverride({ subsonic_protect: newValue }));
                }}
              />
              <span>Защитный инфразвуковой фильтр</span>
            </label>
            <Show when={c()!.freq_hz <= 40}>
              <span class="hint" title="HP слишком низкий, защита не требуется">⊘</span>
            </Show>
            <Show when={c()!.subsonic_protect === false && c()!.freq_hz > 40}>
              <div class="warn">⚠ Защита отключена, риск перегруза диффузора на инфразвуке</div>
            </Show>
          </div>
        </Show>
      </Show>
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
          <span>⚠ PEQ устарел: цель изменена после последней оптимизации</span>
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
            <span class="fb-title" style={{ "font-size": "var(--fs-base)" }}>Исключить</span>
            <button class="peq-add-btn" onClick={() => { const b = band(); if (b) addExclusionZone(b.id, { startHz: 100, endHz: 200 }); }} title="Добавить зону исключения">+</button>
          </div>
          <Show when={(band()?.exclusionZones?.length ?? 0) > 0}>
            <table class="peq-table peq-excl-table">
              <thead><tr><th>От</th><th>До</th><th></th></tr></thead>
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
          <span class="fb-title" style={{ "font-size": "var(--fs-base)" }}>Вручную</span>
          <button class="peq-add-btn" onClick={() => {
            const b = band();
            if (b) {
              if (pendingPeqIdx() != null) commitPeqBand(b.id, pendingPeqIdx()!);
              addPeqBand(b.id, { freq_hz: 1000, gain_db: 0, q: 2.0, enabled: true, filter_type: "Peaking" });
              setPendingPeqIdx(0); setSelectedPeqIdx(0);
            }
          }} title="Добавить PEQ-фильтр вручную">+</button>
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
            <span class="fb-title" style={{ "font-size": "var(--fs-base)" }}>Авто-оптимизатор</span>
          </div>
          <div class="peq-grid">
            <div class="fb-row"><label class="fb-label">Допуск</label><NumberInput value={tolerance()} onChange={setTolerance} min={0.5} max={3.0} step={0.1} unit="dB" /></div>
            <div class="fb-row"><label class="fb-label">Макс. фильтров</label><NumberInput value={maxBands()} onChange={(v: number) => setMaxBands(Math.round(v))} min={1} max={60} step={1} precision={0} /></div>
            <div class="fb-row"><label class="fb-label">Регуляризация</label><NumberInput value={gainRegularization()} onChange={setGainRegularization} min={0} max={1} step={0.0001} precision={4} /></div>
            <div class="fb-row">
              <label class="fb-label">Диапазон</label>
              <select class="peq-range-select" value={peqRangeMode()} onChange={(e) => setPeqRangeMode(e.currentTarget.value as "auto" | "direct")}><option value="auto">Авто</option><option value="direct">Заданный</option></select>
            </div>
            {peqRangeMode() === "auto" ? (
              <div class="fb-row"><label class="fb-label" title="Не ставить PEQ там, где цель на столько dB ниже опорного уровня">Порог dB</label><NumberInput value={peqFloor()} onChange={setPeqFloor} min={0} max={120} step={1} precision={0} /></div>
            ) : (
              <div class="fb-row"><label class="fb-label">Hz</label><NumberInput value={peqDirectLow()} onChange={setPeqDirectLow} min={20} max={20000} step={10} precision={0} /><span style={{ margin: "0 var(--space-xxs)", color: "#8b8b96" }}>–</span><NumberInput value={peqDirectHigh()} onChange={setPeqDirectHigh} min={20} max={20000} step={10} precision={0} /></div>
            )}
          </div>
          <div class="peq-buttons-row">
            <span class="align-range-info">{formatFreq(peqRange()[0])}{"\u2013"}{formatFreq(peqRange()[1])} Hz</span>
            <button class="tb-btn primary" onClick={handleOptimizePeq} disabled={computing()}>{computing() ? "..." : "Оптимизировать"}</button>
            <Show when={peqBands().length > 0}><button class="tb-btn" onClick={handleClearPeq}>Очистить</button></Show>
          </div>
          <Show when={peqError()}><div class="align-error">{peqError()}</div></Show>
          <Show when={peqBands().length > 0}>
            <div class="align-status">
              {peqBands().length} фильтр{peqBands().length === 1 ? "" : "ов"}
              {maxErr() != null ? ` \u00B7 max: ${maxErr()!.toFixed(1)}dB` : ""}
              {iters() != null ? ` \u00B7 ${iters()}it` : ""}
            </div>
          </Show>
          <Show when={peqBands().length > 0}>
            <div class="peq-sidebar-table-scroll">
              <table class="peq-table">
                <thead><tr><th></th><th>Тип</th><th>Гц</th><th>dB</th><th>Q</th><th></th><th></th></tr></thead>
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
