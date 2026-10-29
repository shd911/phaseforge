import { createSignal, createEffect, createResource, Show } from "solid-js";
import type { FilterType, FilterConfig, PeqBand, PeqFilterType } from "../lib/types";
import NumberInput from "./NumberInput";
import { cloneFilterConfig, complementPartnerOk, F_MIN_WORK, F_MAX_WORK } from "../lib/types";
import {
  activeBand,
  appState,
  toggleBandTarget,
  toggleBandInverted,
  toggleBandLinked,
  isBandLinkedFromPrev,
  setBandPreset,
  presetOfTarget,
  setBandReferenceLevel,
  setBandTilt,
  setBandHighPass,
  setBandLowPass,
  exportSampleRate,
  exportTaps,
  exportHybridPhase,
  selectedPeqIdx,
  setSelectedPeqIdx,
  addPeqBand,
  updatePeqBand,
  commitPeqBand,
  removePeqBand,
  addExclusionZone,
  removeExclusionZone,
  updateExclusionZone,
  peqDragging, setPeqDragging, beginInteraction, commitInteraction } from "../stores/bands";
import { attachFieldWheel } from "../lib/wheel-step";
import { driverName } from "../lib/fir-export";
import type { PresetName } from "../stores/bands";
import { CORRECTED_COLOR } from "../lib/plot-helpers";
import {
  orderToSlope, slopeToOrder, availableSlopes, clampNum,
  FILTER_TYPE_ORDER, FILTER_TYPE_LABEL, CUSTOM_Q_MIN, CUSTOM_Q_MAX, GAUSS_M_MIN, GAUSS_M_MAX, CUSTOM_Q_HINT,
} from "../lib/slope";
import {
  tolerance, setTolerance,
  maxBands, setMaxBands,
  gainRegularization, setGainRegularization,
  peqFloor, setPeqFloor,
  peqSmoothFit, setPeqSmoothFit,
  peqBoostFloor, setPeqBoostFloor,
  peqRangeMode, setPeqRangeMode,
  peqDirectLow, setPeqDirectLow,
  peqDirectHigh, setPeqDirectHigh,
  computing, peqError, maxErr, iters,
  peqRangeFor, formatFreq,
  handleOptimizePeq, handleClearPeq, peqStale, peqStaleReason,
} from "../stores/peq-optimize";
import { showToast } from "../lib/toast";
import { pluralRu } from "../lib/plural";
import { qWarnAt, peqRingCost } from "../lib/peq-quality";
import { isLinkwitz, linkwitzDefaults, linkwitzTargetHz } from "../lib/linkwitz";

const LT_TITLE = "Трансформация Линквица (ЗЯ): Гц — f0 ящика, Q — его Qtc, дБ — подъём НЧ. Теоретическая правка по предполагаемому спаду 12 дБ/окт; оптимизатор её сохраняет и подгоняет остальное вокруг неё.";
import { openHighQPopup } from "./HighQWarningPopup";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { projectDir, sanitize } from "../lib/project-io";
import { type SectionRing, ringWarning, ringLabel, ringTitle, ringTooLong as ringTooLongFor } from "../lib/section-ring";

/** b141.38: an out-of-range PEQ entry used to be dropped silently while the
 *  field kept showing it — the table and the model disagreed with no hint.
 *  Restore what the model holds and say what the range is. */
function rejectPeqInput(el: HTMLInputElement, shown: string, hint: string) {
  el.value = shown;
  showToast(hint, "warn");
}

/** b141.105 (audit 2026-10-23 §5.9): the PEQ limits, one set for the table,
 *  the «Вручную» row, typing and the wheel. */
const PEQ_Q_MIN = 0.1;
const PEQ_Q_MAX = 20;
const peqGainLimits = () => (exportHybridPhase() ? { lo: -60, hi: 60 } : { lo: -18, hi: 6 });
const fmtSigned = (v: number) => (v > 0 ? `+${v}` : `${v}`).replace("-", "−");
const peqGainHint = () => { const l = peqGainLimits(); return `Усиление PEQ: ${fmtSigned(l.lo)}…${fmtSigned(l.hi)} дБ`; };
const PEQ_FREQ_HINT = `Частота PEQ: 20–${F_MAX_WORK} Гц`;
const PEQ_Q_HINT = `Q PEQ: ${PEQ_Q_MIN}–${PEQ_Q_MAX}`;

/** Keyboard side of the custom span checkboxes (Lin-φ, «Без отражений»):
 *  they stay span+onClick (WebKit spurious events on a real checkbox), with
 *  role="checkbox" and Space/Enter (§5.15). */
function checkboxKeys(toggle: () => void) {
  return (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); }
  };
}

// Remember last filter config per band so toggle off→on restores settings
const lastHP = new Map<string, import("../lib/types").FilterConfig>();
const lastLP = new Map<string, import("../lib/types").FilterConfig>();


/** b141.67 (audit stage 2 B9): the optimizer places LF bands at 20.6 Hz —
 *  rounding them to 21 in the table hid the real value. */
function fmtPeqFreq(f: number): string {
  return f < 100 ? f.toFixed(1) : String(Math.round(f));
}

export default function ControlPanel(props: { rightPanel?: boolean }) {
  if (props.rightPanel) {
    // Right panel: Target + PEQ — no tabs, always visible, scrollable
    return (
      <div class="right-panel-content">
        <div class="right-panel-section">
          <div class="right-panel-title">Цель</div>
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
      // b141.105 (§5.19): the failure only reached the console.
      console.error("Export target failed:", e);
      showToast(`Не удалось экспортировать цель: ${e}`, "warn");
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
    <Show when={band()} fallback={<p class="meas-empty">Нет активной полосы. Выберите вкладку полосы выше.</p>}>
      <div class="filters-row">
        {/* General */}
        <div class="filter-block">
          <div class="fb-header">
            <span class="fb-title">Общие</span>
            <button
              class={`fb-toggle ${enabled() ? "on" : ""}`}
              onClick={() => { const id = bandId(); if (id) toggleBandTarget(id); }}
              aria-pressed={enabled()}
              title="Включить/выключить цель для этой полосы"
            >{enabled() ? "Вкл" : "Выкл"}</button>
          </div>
          <div class="fb-row">
            <label class="fb-label" for="cp-preset">Пресет</label>
            {/* b141.105 (§5.6): controlled — shows the preset the band's curve
                matches («Свой» otherwise). Uncontrolled, it always showed
                «Плоский», carried one band's choice to the next, and picking
                the option it showed did nothing. */}
            <select
              id="cp-preset"
              class="fb-select"
              value={presetOfTarget(target())}
              onChange={(e) => {
                const v = e.currentTarget.value as PresetName;
                const id = bandId();
                if (id && v !== "custom") setBandPreset(id, v);
                e.currentTarget.value = presetOfTarget(target());
              }}
            >
              <option value="flat">Плоский</option>
              <option value="harman">Harman</option>
              <option value="bk">B&K</option>
              <option value="custom" disabled>Свой</option>
            </select>
          </div>
          <div class="fb-row">
            <label class="fb-label">Уровень</label>
            <NumberInput
              value={target()?.reference_level_db ?? 0}
              onChange={(v) => { const id = bandId(); if (id) setBandReferenceLevel(id, v); }}
              min={-20} max={20} step={0.5} unit="дБ" label="Уровень цели"
            />
          </div>
          <div class="fb-row">
            <label class="fb-label">Наклон</label>
            <NumberInput
              value={target()?.tilt_db_per_octave ?? 0}
              onChange={(v) => { const id = bandId(); if (id) setBandTilt(id, v); }}
              min={-6} max={6} step={0.1} unit="дБ/окт" label="Наклон цели"
            />
          </div>
          <div class="fb-row">
            <label class="fb-label">Инверсия</label>
            <button
              class={`fb-toggle invert-toggle ${inverted() ? "on" : ""}`}
              onClick={() => { const id = bandId(); if (id) toggleBandInverted(id); }}
              aria-pressed={inverted()}
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
          partner={hpLinked() ? cloneFilterConfig(appState.bands[bandIdx() - 1]?.target.low_pass) : null}
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
          partner={linked() ? cloneFilterConfig(appState.bands[bandIdx() + 1]?.target.high_pass) : null}
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
        <button class="tb-btn tb-btn-sm" style={{ "align-self": "flex-start", "margin-left": "auto" }} onClick={handleExportTarget}
          title="Сохранить кривую цели в текстовый файл REW">
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
  isHighPass?: boolean; // b138: subsonic-protect UI for Gaussian HP; b141.111: ring of an HP
  partner?: import("../lib/types").FilterConfig | null; // b141.110: the linked filter on the other side
}


const COMPLEMENT = "Complement";
const COMPLEMENT_TITLE = "1 − H: этот фильтр — дополнение связанного «Произвольного Q» (та же частота, порядок и Q), "
  + "сумма двух полос ровная. Линейная фаза. Спад этой стороны пологий и несимметричный. "
  + "Доступно, когда у соседа «Произвольный Q» с Q не выше 0.707.";

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

  // b141.110: «1 − H» is a type of its own in the dropdown — the section is
  // the complement of the linked «Произвольный Q», not the filter it mirrors.
  const isComp = () => !!c()?.complementary;
  const canComp = () => complementPartnerOk(props.partner);

  // b141.111: how long this section rings (to −60 / −100 dB) at the export
  // rate — the crossover counterpart of the PEQ «мс» column.
  const [ring] = createResource(
    () => {
      const cur = c();
      if (!cur) return false;
      const config = cloneFilterConfig(cur);
      return { config, isLowPass: !props.isHighPass, sampleRate: exportSampleRate(), key: JSON.stringify(config) };
    },
    (src) => invoke<SectionRing>("compute_section_ring",
      { config: src.config, isLowPass: src.isLowPass, sampleRate: src.sampleRate }).catch(() => null),
  );
  const halfFileMs = () => exportTaps() / 2 / exportSampleRate() * 1000;
  const ringTooLong = () => { const r = ring(); return !!r && ringTooLongFor(r, !!c()?.linear_phase, halfFileMs()); };

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
              title={props.linked ? "Отвязать от следующей полосы" : "Связать со следующей полосой"}
            >{/* b141.116: unlinked = a broken chain, dimmed — not the same icon in red */}
              <Show when={props.linked} fallback={
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" stroke-width="2.5" stroke-linecap="round"><path d="m18.84 12.25 1.72-1.71a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="m5.17 11.75-1.71 1.71a5 5 0 0 0 7.07 7.07l1.71-1.71"/><path d="M8 2v3M2 8h3M16 19v3M19 16h3"/></svg>
              }>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={CORRECTED_COLOR} stroke-width="2.5" stroke-linecap="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
              </Show>
            </button>
          </Show>
          <Show when={!props.canLink && props.linked}>
            <span class="fb-link-indicator" title="Связана с соседней полосой"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={CORRECTED_COLOR} stroke-width="2.5" stroke-linecap="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg></span>
          </Show>
        </span>
        <button
          class={`fb-toggle ${c() ? "on" : ""}`}
          onClick={props.onToggle}
          aria-pressed={c() != null}
          aria-label={`${props.title}: ${c() ? "выключить" : "включить"}`}
        >{c() ? "Вкл" : "Выкл"}</button>
      </div>
      <Show when={c()}>
        <div class="fb-grid">
          <div class="fb-row">
            <label class="fb-label">Тип</label>
            <select
              class="fb-select"
              aria-label={`${props.title}: тип`}
              value={isComp() ? COMPLEMENT : c()!.filter_type}
              onChange={(e) => {
                const v = e.currentTarget.value;
                if (v === COMPLEMENT) {
                  const p = props.partner;
                  if (isComp() || !p || !canComp()) return;
                  props.onChange(withOverride({ filter_type: "Custom", order: p.order, q: p.q ?? 0.707, shape: null,
                    subsonic_protect: null, linear_phase: true, complementary: true }));
                  return;
                }
                const ft = v as FilterType;
                if (ft === c()!.filter_type && !isComp()) return; // guard: SolidJS/WebKit programmatic set
                // b138: subsonic-protect auto-on for Gaussian HP, dropped otherwise.
                const subsonic = ft === "Gaussian" && props.isHighPass ? true : null;
                if (ft === "Gaussian") {
                  props.onChange(withOverride({ filter_type: ft, shape: c()!.shape ?? 1.0, q: null, subsonic_protect: subsonic, complementary: false }));
                } else if (ft === "Custom") {
                  props.onChange(withOverride({ filter_type: ft, shape: null, q: c()!.q ?? 0.707, subsonic_protect: null, complementary: false }));
                } else {
                  props.onChange(withOverride({ filter_type: ft, shape: null, q: null, subsonic_protect: null, complementary: false }));
                }
              }}
            >
              {FILTER_TYPE_ORDER.map((t) => <option value={t}>{FILTER_TYPE_LABEL[t]}</option>)}
              {/* Always in the DOM (a <Show> inside <select> races the value binding). */}
              <option value={COMPLEMENT} hidden={!props.linked && !isComp()} disabled={!canComp() && !isComp()}
                title={COMPLEMENT_TITLE}>1 − H (вычитание)</option>
            </select>
          </div>
          <div class="fb-row">
            <label class="fb-label">Частота</label>
            <NumberInput
              value={c()!.freq_hz}
              onChange={(v) => props.onChange(withOverride({ freq_hz: v }))}
              min={10} max={F_MAX_WORK} step={1} unit="Гц" freqMode label={`${props.title}: частота`}
            />
          </div>
          <Show when={!isGaussian() && !isComp()}>
            <div class="fb-row">
              <label class="fb-label">Крутизна</label>
              <select
                class="fb-select"
                aria-label={`${props.title}: крутизна`}
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
                  >{`${s} дБ/окт`}</option>
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
                min={GAUSS_M_MIN} max={GAUSS_M_MAX} step={0.1} label={`${props.title}: M`}
              />
            </div>
          </Show>
          <Show when={isCustom() && !isComp()}>
            <div class="fb-row">
              <label class="fb-label">Q</label>
              <NumberInput
                value={c()!.q ?? 0.707}
                onChange={(v) => props.onChange(withOverride({ q: v }))}
                min={CUSTOM_Q_MIN} max={CUSTOM_Q_MAX} step={0.01} label={`${props.title}: Q`}
              />
            </div>
          </Show>
          <Show when={!isComp()}>
            <div class="fb-row">
              <label class="fb-label"></label>
              <span class="fb-checkbox" title="Линейная фаза (только АЧХ, без поворота фазы)"
                role="checkbox" tabIndex={0} aria-checked={c()!.linear_phase}
                aria-label={`${props.title}: линейная фаза`}
                onKeyDown={checkboxKeys(() => props.onChange(withOverride({ linear_phase: !c()!.linear_phase })))}
                onClick={() => props.onChange(withOverride({ linear_phase: !c()!.linear_phase }))}>
                <span class={`fb-check-box ${c()!.linear_phase ? "checked" : ""}`} />
                <span class="fb-check-label">Lin-φ</span>
              </span>
            </div>
          </Show>
          <Show when={isComp()}>
            <div class="fb-hint" title={COMPLEMENT_TITLE}>
              {`= 1 − соседний: Q ${(c()!.q ?? 0.707).toFixed(2)}, ${orderToSlope("Custom", c()!.order)} дБ/окт, линейная фаза`}
            </div>
          </Show>
        </div>
        <Show when={isCustom() && !isComp()}>
          <div class="fb-hint">{CUSTOM_Q_HINT}</div>
        </Show>
        <Show when={ring()}>
          {(r) => (
            <div class="fb-ring" title={ringTitle(r(), c()!.linear_phase, exportSampleRate(), halfFileMs())}>
              {ringLabel(r(), c()!.linear_phase)}
              <Show when={ringTooLong()}>
                <div class="warn">{ringWarning(halfFileMs())}</div>
              </Show>
            </div>
          )}
        </Show>
        <Show when={props.isHighPass && isGaussian()}>
          <div class="subsonic-protect-row">
            <label
              title={c()!.freq_hz > 40
                ? "Минимально-фазовый Butterworth 48 дБ/окт на 3 октавы ниже ФВЧ. Защищает динамик от излишнего хода диффузора в инфразвуке."
                : "Недоступно при ФВЧ ≤ 40 Гц: на 3 октавы ниже защита встала бы под 5 Гц и ничего бы не защищала. Поднимите ФВЧ выше 40 Гц."}
              style={{ display: "flex", "align-items": "center", gap: "6px", cursor: c()!.freq_hz > 40 ? "pointer" : "not-allowed", opacity: c()!.freq_hz > 40 ? 1 : 0.6 }}
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
              <span class="hint" title="ФВЧ слишком низко, защита не требуется">⊘</span>
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

// ---------------------------------------------------------------------------
// PEQ Tab — migrated from PeqSidebar
// ---------------------------------------------------------------------------

function PeqTab() {
  const band = () => activeBand();
  const peqBands = () => band()?.peqBands ?? [];
  // b141.91: the «Hz» row in «Авто» shows the optimizer's own range (the
  // floor-trimmed one), refetched when the target, floor or rate changes.
  const [autoPeqRange] = createResource(
    () => {
      const b = band();
      if (!b?.measurement || peqRangeMode() !== "auto") return false;
      return { b, key: `${JSON.stringify(b.target)}|${peqFloor()}|${exportSampleRate()}|${b.measurement.freq.length}` };
    },
    (src) => peqRangeFor(src.b).catch(() => null),
  );
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

  /** b141.105 (audit 2026-10-23 §5.8): «От»/«До» of an exclusion zone. A
   *  value below 20 Hz was dropped while the field kept showing it, and
   *  От ≥ До went through. Out-of-range input is clamped, a reversed zone
   *  rejected; the field always ends up showing the model. */
  function commitZone(el: HTMLInputElement, i: number, field: "startHz" | "endHz") {
    const b = band();
    const z = b?.exclusionZones?.[i];
    if (!b || !z) return;
    const shown = String(Math.round(z[field]));
    const v = parseFloat(el.value);
    if (!Number.isFinite(v)) { rejectPeqInput(el, shown, "Зона исключения: введите частоту в Гц"); return; }
    const c = Math.round(clampNum(v, F_MIN_WORK, F_MAX_WORK));
    const start = field === "startHz" ? c : z.startHz;
    const end = field === "endHz" ? c : z.endHz;
    if (start >= end) { rejectPeqInput(el, shown, "Зона исключения: «От» должно быть меньше «До»"); return; }
    el.value = String(c);
    if (c !== v) showToast(`Зона исключения: ${F_MIN_WORK}–${F_MAX_WORK} Гц`, "warn");
    if (c !== Math.round(z[field])) updateExclusionZone(b.id, i, { [field]: c });
  }

  /** b141.107: switching a row to «LT» fills it with a transform for this
   *  measurement (assumed box f0 at its −3 dB point, Q0 0.707, +6 dB) — a
   *  peaking filter's 1 kHz / 0 dB read as a box would be meaningless. */
  function setPeqType(i: number, type: PeqFilterType) {
    const bd = band(); if (!bd) return;
    const cur = bd.peqBands[i];
    if (type === "LinkwitzTransform" && cur && !isLinkwitz(cur)) {
      const m = bd.measurement;
      const d = linkwitzDefaults(m?.freq ?? null, m?.magnitude ?? null, bd.target.low_pass?.freq_hz ?? 200);
      updatePeqBand(bd.id, i, { ...d, enabled: cur.enabled });
    } else updatePeqBand(bd.id, i, { filter_type: type });
  }

  function addLinkwitz() {
    const bd = band(); if (!bd) return;
    if (pendingPeqIdx() != null) commitPeqBand(bd.id, pendingPeqIdx()!);
    const m = bd.measurement;
    addPeqBand(bd.id, linkwitzDefaults(m?.freq ?? null, m?.magnitude ?? null, bd.target.low_pass?.freq_hz ?? 200));
    setPendingPeqIdx(0); setSelectedPeqIdx(0);
  }

  /** One commit rule for a typed PEQ value — table rows and the «Вручную»
   *  row alike (§5.9): frequency and Q out of range are rejected, gain is
   *  clamped to the mode's limits; the field shows the model afterwards. */
  function commitPeqField(el: HTMLInputElement, i: number, field: "freq_hz" | "gain_db" | "q") {
    const bd = band();
    const cur: PeqBand | undefined = bd?.peqBands[i];
    if (!bd || !cur) return;
    const v = parseFloat(el.value);
    if (field === "freq_hz") {
      if (Number.isFinite(v) && v >= 20 && v <= F_MAX_WORK) updatePeqBand(bd.id, i, { freq_hz: v });
      else rejectPeqInput(el, fmtPeqFreq(cur.freq_hz), PEQ_FREQ_HINT);
    } else if (field === "gain_db") {
      if (!Number.isFinite(v)) { rejectPeqInput(el, cur.gain_db.toFixed(1), peqGainHint()); return; }
      const { lo, hi } = peqGainLimits();
      const g = clampNum(v, lo, hi);
      if (g !== v) rejectPeqInput(el, g.toFixed(1), peqGainHint());
      updatePeqBand(bd.id, i, { gain_db: g });
    } else {
      if (Number.isFinite(v) && v >= PEQ_Q_MIN && v <= PEQ_Q_MAX) updatePeqBand(bd.id, i, { q: v });
      else rejectPeqInput(el, cur.q.toFixed(2), PEQ_Q_HINT);
    }
  }

  // b141.43: shared wheel model — steps accumulate (trackpad), the field must
  // have been clicked, and a gesture runs as one PEQ interaction: fast chart
  // update, one undo entry, full rebuild when it ends. Same limits as typing.
  const peqWheel = (idx: () => number, field: "freq_hz" | "gain_db" | "q") => (el: HTMLInputElement) => attachFieldWheel(el, {
    onSteps: (steps, coarse) => {
      const i = idx();
      const bd = band(); const cur = bd?.peqBands[i]; if (!bd || !cur) return;
      if (!peqDragging()) { beginInteraction("PEQ wheel"); setPeqDragging(true); }
      const dir = Math.sign(steps); const n = Math.abs(steps); const mult = coarse ? 10 : 1;
      if (field === "freq_hz") {
        let f = cur.freq_hz;
        for (let k = 0; k < n; k++) f = clampNum(f + dir * Math.max(1, Math.round(f * 0.02)) * mult, 20, F_MAX_WORK);
        updatePeqBand(bd.id, i, { freq_hz: f });
      } else if (field === "gain_db") {
        const { lo, hi } = peqGainLimits();
        const g = clampNum(Math.round((cur.gain_db + steps * 0.1 * mult) * 10) / 10, lo, hi);
        updatePeqBand(bd.id, i, { gain_db: g });
      } else {
        const q = clampNum(Math.round((cur.q + steps * 0.1 * mult) * 10) / 10, PEQ_Q_MIN, PEQ_Q_MAX);
        updatePeqBand(bd.id, i, { q });
      }
    },
    onEnd: () => { if (peqDragging()) { setPeqDragging(false); commitInteraction(); } },
  });

  return (
    <div
      class="peq-tab-content"
      classList={{ "peq-tab-stale": isStale() }}
    >
      <Show when={isStale()}>
        <div class="peq-stale-banner">
          <span>⚠ PEQ устарел: {peqStaleReason(band()!) ?? "цель изменена"} после последней оптимизации</span>
          <span style={{ display: "flex", gap: "var(--space-xs)", "justify-content": "flex-end", "flex-wrap": "wrap" }}>
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
            <button class="peq-add-btn" onClick={() => { const b = band(); if (b) addExclusionZone(b.id, { startHz: 100, endHz: 200 }); }} title="Добавить зону исключения" aria-label="Добавить зону исключения">+</button>
          </div>
          <Show when={(band()?.exclusionZones?.length ?? 0) > 0}>
            <table class="peq-table peq-excl-table">
              <thead><tr><th>От, Гц</th><th>До, Гц</th><th></th></tr></thead>
              <tbody>
                {(band()?.exclusionZones ?? []).map((z, i) => (
                  <tr>
                    <td><input class="peq-input" type="number" value={Math.round(z.startHz)} min={F_MIN_WORK} max={F_MAX_WORK} step={1}
                      aria-label={`Зона ${i + 1}: от, Гц`} onChange={(e) => commitZone(e.currentTarget, i, "startHz")} /></td>
                    <td><input class="peq-input" type="number" value={Math.round(z.endHz)} min={F_MIN_WORK} max={F_MAX_WORK} step={1}
                      aria-label={`Зона ${i + 1}: до, Гц`} onChange={(e) => commitZone(e.currentTarget, i, "endHz")} /></td>
                    <td><button class="peq-remove" onClick={() => { const b = band(); if (b) removeExclusionZone(b.id, i); }} aria-label="Удалить зону исключения" title="Удалить зону исключения">×</button></td>
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
          }} title="Добавить PEQ-фильтр вручную" aria-label="Добавить PEQ-фильтр вручную">+</button>
          <button class="peq-add-btn" onClick={addLinkwitz} title={LT_TITLE} aria-label="Добавить трансформацию Линквица">LT</button>
        </div>
        <Show when={pendingPeqIdx() != null && peqBands().length > 0}>
          {(() => {
            const pi = pendingPeqIdx()!;
            const pb = () => peqBands()[pi];
            if (!pb()) return null;
            // b141.105 (§5.9): same limits as the table (gain ±60 was accepted
            // in Standard) and the same wheel.
            return (
              <table class="peq-table"><tbody>
                <tr class="peq-row-pending peq-row-selected" onClick={() => setSelectedPeqIdx(pi)}>
                  <td><input type="checkbox" class="peq-toggle" checked={pb()!.enabled} aria-label="Фильтр включён" onChange={() => { const bd = band(); if (bd) updatePeqBand(bd.id, pi, { enabled: !pb()!.enabled }); }} /></td>
                  <td><select class="peq-type-select" value={pb()!.filter_type} aria-label="Тип фильтра" onChange={(e) => setPeqType(pi, e.currentTarget.value as PeqFilterType)}><option value="Peaking">PK</option><option value="LowShelf">LS</option><option value="HighShelf">HS</option><option value="LinkwitzTransform" title={LT_TITLE}>LT</option></select></td>
                  <td><input class="peq-input" type="number" value={fmtPeqFreq(pb()!.freq_hz)} min={20} max={F_MAX_WORK} step={1} ref={peqWheel(() => pi, "freq_hz")} aria-label="Частота, Гц" onChange={(e) => commitPeqField(e.currentTarget, pi, "freq_hz")} /></td>
                  <td><input class={`peq-input ${pb()!.gain_db > 0 ? "peq-boost" : "peq-cut"}`} type="number" value={pb()!.gain_db.toFixed(1)} min={peqGainLimits().lo} max={peqGainLimits().hi} step={0.1} ref={peqWheel(() => pi, "gain_db")} aria-label="Усиление, дБ" onChange={(e) => commitPeqField(e.currentTarget, pi, "gain_db")} /></td>
                  <td><input class="peq-input" type="number" value={pb()!.q.toFixed(2)} min={PEQ_Q_MIN} max={PEQ_Q_MAX} step={0.1} ref={peqWheel(() => pi, "q")} aria-label="Добротность Q" onChange={(e) => commitPeqField(e.currentTarget, pi, "q")} /></td>
                  <td><button class="peq-commit" aria-label="Добавить фильтр" title="Добавить фильтр" onClick={() => { const bd = band(); if (bd) { const ni = commitPeqBand(bd.id, pi); setPendingPeqIdx(null); setSelectedPeqIdx(ni); } }}>✓</button></td>
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
            <div class="fb-row"><label class="fb-label">Допуск</label><NumberInput value={tolerance()} onChange={setTolerance} min={0.5} max={3.0} step={0.1} unit="дБ" label="Допуск" /></div>
            <div class="fb-row"><label class="fb-label">Макс. фильтров</label><NumberInput value={maxBands()} onChange={(v: number) => setMaxBands(Math.round(v))} min={1} max={60} step={1} precision={0} label="Макс. фильтров" /></div>
            {/* b141.105 (§5.19): step 0.001 — a 0.0001 step moved nothing visible per click */}
            <div class="fb-row"><label class="fb-label">Регуляризация</label><NumberInput value={gainRegularization()} onChange={setGainRegularization} min={0} max={1} step={0.001} precision={4} label="Регуляризация" /></div>
            <div class="fb-row">
              <label class="fb-label" for="cp-peq-range">Диапазон</label>
              <select id="cp-peq-range" class="peq-range-select" value={peqRangeMode()} onChange={(e) => setPeqRangeMode(e.currentTarget.value as "auto" | "direct")}><option value="auto">Авто</option><option value="direct">Заданный</option></select>
            </div>
            {/* b141.89: own row — beside the buttons it pushed them out of the 280 px panel */}
            <div class="fb-row">
              <label class="fb-label">Замер</label>
              <span class="fb-checkbox" title="Подгонять PEQ по сглаженному замеру (1/6 окт. ниже 100 Гц, 1/3 выше) — не гнаться за мелкими пульсациями от отражений"
                role="checkbox" tabIndex={0} aria-checked={peqSmoothFit()}
                onKeyDown={checkboxKeys(() => setPeqSmoothFit(!peqSmoothFit()))}
                onClick={() => setPeqSmoothFit(!peqSmoothFit())}>
                <span class={`fb-check-box ${peqSmoothFit() ? "checked" : ""}`} />
                <span class="fb-check-label">Без отражений</span>
              </span>
            </div>
            <div class="fb-row"><label class="fb-label" title="Не ставить PEQ там, где цель на столько дБ ниже опорного уровня (в обоих режимах диапазона)">Порог, дБ</label><NumberInput value={peqFloor()} onChange={setPeqFloor} min={0} max={120} step={1} precision={0} label="Порог, дБ" /></div>
            <div class="fb-row"><label class="fb-label" title="Подъём PEQ только там, где цель не ниже опорного уровня на столько дБ; глубже — только подавление (0 — без ограничения)">Подъём до, дБ</label><NumberInput value={peqBoostFloor()} onChange={setPeqBoostFloor} min={0} max={120} step={1} precision={0} label="Подъём до, дБ" /></div>
            {/* b141.90: one Hz row — the auto range as text (it sat beside the
                buttons), editable fields in «Заданный». */}
            {peqRangeMode() === "direct" ? (
              <div class="fb-row"><label class="fb-label">Гц</label><NumberInput value={peqDirectLow()} onChange={setPeqDirectLow} min={20} max={F_MAX_WORK} step={10} precision={0} label="Диапазон от, Гц" /><span class="range-dash">–</span><NumberInput value={peqDirectHigh()} onChange={setPeqDirectHigh} min={20} max={F_MAX_WORK} step={10} precision={0} label="Диапазон до, Гц" /></div>
            ) : (
              <div class="fb-row"><label class="fb-label">Гц</label><span class="align-range-info">{autoPeqRange() ? `${formatFreq(autoPeqRange()![0])}–${formatFreq(autoPeqRange()![1])}` : "—"}</span></div>
            )}
          </div>
          <div class="peq-buttons-row">
            {/* b141.105 (§5.19): both labels share one grid cell — the button
                keeps its width and says what it does while computing (it
                collapsed to «...»). */}
            <button class="tb-btn primary" onClick={handleOptimizePeq} disabled={computing()} aria-busy={computing()}>
              <span class="btn-label-stack">
                <span classList={{ "btn-label-off": computing() }}>Оптимизировать</span>
                <span classList={{ "btn-label-off": !computing() }}>Оптимизация…</span>
              </span>
            </button>
            <Show when={peqBands().length > 0}><button class="tb-btn" onClick={handleClearPeq}>Очистить</button></Show>
          </div>
          <Show when={peqError()}><div class="align-error">{peqError()}</div></Show>
          <Show when={peqBands().length > 0}>
            <div class="align-status">
              {peqBands().length} {pluralRu(peqBands().length, "фильтр", "фильтра", "фильтров")}
              {maxErr() != null ? ` · макс. ошибка ${maxErr()!.toFixed(1)} дБ` : ""}
              {iters() != null ? ` · ${iters()} ${pluralRu(iters()!, "итерация", "итерации", "итераций")}` : ""}
            </div>
          </Show>
          <Show when={peqBands().length > 0}>
            <div class="peq-sidebar-table-scroll">
              <table class="peq-table">
                <thead><tr><th></th><th>Тип</th><th>Гц</th><th>дБ</th><th>Q</th><th title="Время звона фильтра: за сколько мс его резонанс затухает до −60 дБ">мс</th><th></th><th></th></tr></thead>
                <tbody>
                  {peqBands().map((b, i) => {
                    const isPending = pendingPeqIdx() === i;
                    return (
                      <tr class={`${selectedPeqIdx() === i ? "peq-row-selected" : ""} ${isPending ? "peq-row-pending" : ""} ${!b.enabled ? "peq-row-disabled" : ""}`}
                        onClick={() => setSelectedPeqIdx(selectedPeqIdx() === i ? null : i)}>
                        <td><input type="checkbox" class="peq-toggle" checked={b.enabled} aria-label={`Фильтр ${i + 1} включён`} onChange={(e) => { e.stopPropagation(); const bd = band(); if (bd) updatePeqBand(bd.id, i, { enabled: !b.enabled }); }} onClick={(e) => e.stopPropagation()} /></td>
                        <td><select class="peq-type-select" value={b.filter_type ?? "Peaking"} aria-label={`Фильтр ${i + 1}: тип`} onChange={(e) => { e.stopPropagation(); setPeqType(i, e.currentTarget.value as PeqFilterType); }} onClick={(e) => e.stopPropagation()}><option value="Peaking">PK</option><option value="LowShelf">LS</option><option value="HighShelf">HS</option><option value="LinkwitzTransform" title={LT_TITLE}>LT</option></select></td>
                        <td><input class="peq-input" type="number" value={fmtPeqFreq(b.freq_hz)} min={20} max={F_MAX_WORK} step={1} ref={peqWheel(() => i, "freq_hz")} aria-label={`Фильтр ${i + 1}: частота, Гц`} onChange={(e) => commitPeqField(e.currentTarget, i, "freq_hz")} /></td>
                        <td><input class={`peq-input ${b.gain_db > 0 ? "peq-boost" : "peq-cut"}`} type="number" value={b.gain_db.toFixed(1)} min={peqGainLimits().lo} max={peqGainLimits().hi} step={0.1} ref={peqWheel(() => i, "gain_db")} aria-label={`Фильтр ${i + 1}: усиление, дБ`} onChange={(e) => commitPeqField(e.currentTarget, i, "gain_db")} /></td>
                        <td><input class="peq-input" type="number" value={b.q.toFixed(2)} min={PEQ_Q_MIN} max={PEQ_Q_MAX} step={0.1} ref={peqWheel(() => i, "q")} aria-label={`Фильтр ${i + 1}: Q`} onChange={(e) => commitPeqField(e.currentTarget, i, "q")} /></td>
                        {(() => {
                          // b141.93: time cost of this filter — its ringing to −60 dB.
                          // Red when the tail to −100 dB is longer than half the file:
                          // the export then shifts the band's delay.
                          const rc = () => peqRingCost(b, exportSampleRate());
                          const halfMs = () => exportTaps() / 2 / exportSampleRate() * 1000;
                          const cls = () => !b.enabled ? "" : rc().ms100 > halfMs() ? "peq-ring-bad" : rc().ms60 >= 150 ? "peq-ring-warn" : "";
                          return (
                            <td class={`peq-ring ${cls()}`}
                              title={`Звон до −60 дБ: ${Math.round(rc().ms60)} мс, до −100 дБ: ${Math.round(rc().ms100)} мс. Половина файла: ${Math.round(halfMs())} мс — если хвост длиннее, экспорт сдвинет задержку полосы.`}>
                              {Math.abs(b.gain_db) < 0.05 ? "" : Math.round(rc().ms60)}
                            </td>
                          );
                        })()}
                        <td>{isLinkwitz(b) ? (
                          <span class="peq-theory" title={`Теоретическая правка: трансформация Линквица по предполагаемому спаду ЗЯ 12 дБ/окт (f0 ${b.freq_hz.toFixed(1)} Гц, Q0 ${b.q.toFixed(2)}) → ${linkwitzTargetHz(b).toFixed(1)} Гц, Q 0,71. Нижний край замера не учитывается.`}
                            aria-label="Теоретическая правка">≈</span>
                        ) : b.enabled && b.q > qWarnAt(b.freq_hz) ? (
                          <button class="peq-warn-icon" title="" aria-label="Высокая добротность"
                            onClick={(e) => { e.stopPropagation(); openHighQPopup(b, i); }}>⚠</button>
                        ) : null}</td>
                        <td>{isPending ? <button class="peq-commit" aria-label="Добавить фильтр" title="Добавить фильтр" onClick={(e) => { e.stopPropagation(); const bd = band(); if (bd) { const ni = commitPeqBand(bd.id, i); setPendingPeqIdx(null); setSelectedPeqIdx(ni); } }}>✓</button> : <button class="peq-remove" onClick={() => handleRemovePeq(i)} aria-label="Удалить фильтр" title="Удалить фильтр">×</button>}</td>
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
