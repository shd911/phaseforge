import { createSignal, Show } from "solid-js";
import { F_MIN_WORK, F_MAX_WORK } from "../lib/types";
import type { FilterConfig, FilterType } from "../lib/types";
import { appState, setBandLowPass } from "../stores/bands";
import {
  availableSlopes, orderToSlope, slopeToOrder, clampNum,
  FILTER_TYPE_ORDER, FILTER_TYPE_LABEL, CUSTOM_Q_MIN, CUSTOM_Q_MAX, GAUSS_M_MIN, GAUSS_M_MAX, CUSTOM_Q_HINT,
} from "../lib/slope";
import NumField from "./NumField";
import Modal from "./Modal";

export interface CrossoverDialogData {
  bandIndex: number;   // index of band with LP filter
  bandId: string;
  bandName: string;
  nextBandName: string;
  freq: number;
  filterType: FilterType;
  order: number;
  linearPhase: boolean;
  shape: number | null;  // Gaussian M coefficient
  q: number | null;      // Custom Q factor
}

const [dialogData, setDialogData] = createSignal<CrossoverDialogData | null>(null);

export function openCrossoverDialog(data: CrossoverDialogData) {
  setDialogData(data);
}

export function closeCrossoverDialog() {
  setDialogData(null);
}

export default function CrossoverDialog() {
  const [freq, setFreq] = createSignal(0);
  const [filterType, setFilterType] = createSignal<FilterType>("LinkwitzRiley");
  const [order, setOrder] = createSignal(4);
  const [linearPhase, setLinearPhase] = createSignal(true);
  const [shape, setShape] = createSignal(1.0);
  const [customQ, setCustomQ] = createSignal(0.707);

  const isGaussian = () => filterType() === "Gaussian";
  const isCustom = () => filterType() === "Custom";

  // Sync local state when dialog opens
  function initFromData(data: CrossoverDialogData) {
    setFreq(data.freq);
    setFilterType(data.filterType);
    setOrder(data.order);
    setLinearPhase(data.linearPhase);
    setShape(data.shape ?? 1.0);
    setCustomQ(data.q ?? 0.707);
  }

  const freqValid = () => Number.isFinite(freq()) && freq() >= F_MIN_WORK && freq() <= F_MAX_WORK;

  function handleApply() {
    const data = dialogData();
    if (!data) return;
    // 2026-09-05 audit: Enter on an empty field applied freq_hz = 0 — the
    // model silently dropped the filter while FIR generation rejected it.
    if (!freqValid()) return;

    const config: FilterConfig = {
      filter_type: filterType(),
      order: order(),
      freq_hz: Math.round(freq()),
      // b141.105 (§5.7): M and Q are clamped like the filter block clamps them.
      shape: isGaussian() ? clampNum(shape(), GAUSS_M_MIN, GAUSS_M_MAX) : null,
      // b141.105 (§5.21): the checkbox is honoured for Gaussian too. Forcing
      // linear phase dates from b51, before Gaussian min-phase existed (b97):
      // «Применить» silently turned a min-phase Gaussian crossover — and the
      // linked HP of the next band — linear-phase.
      linear_phase: linearPhase(),
      q: isCustom() ? clampNum(customQ(), CUSTOM_Q_MIN, CUSTOM_Q_MAX) : null,
    };

    // Set LP on the band — propagation handles the HP on the next band
    setBandLowPass(data.bandId, config);
    closeCrossoverDialog();
  }

  function handleCancel() {
    closeCrossoverDialog();
  }

  // b141.38: `parseFloat(v) || 0` rewrote a cleared field to 0 on the first
  // Backspace. An unparsable frequency goes NaN so Apply stays disabled and
  // the hint explains why; M and Q keep their previous value (NumField).

  // 2026-09-05 audit: this dialog showed a raw "order" while the filter
  // block shows dB/oct for the same filter (LR order 2 = 24 dB/oct) and
  // silently snapped LR orders 1/3/5/6/7 to 4. Same slope.ts mapping now.

  return (
    <Show when={(() => {
      const d = dialogData();
      if (d) initFromData(d);
      return d;
    })()}>
      {(data) => (
        <Modal overlayClass="xo-overlay" dialogClass="xo-dialog" titleId="xo-dlg-title"
          onEnter={handleApply} onEscape={handleCancel}>
            <div class="xo-title" id="xo-dlg-title">
              Кроссовер: {data().bandName} / {data().nextBandName}
            </div>

            <div class="xo-row">
              <label class="xo-label" for="xo-freq">Частота</label>
              <NumField id="xo-freq" autofocus value={freq()} onChange={setFreq} invalidAsNaN
                min={F_MIN_WORK} max={F_MAX_WORK} step={1} />
              <span class="xo-unit">Гц</span>
            </div>
            <Show when={!freqValid()}>
              <div class="xo-hint" style={{ color: "var(--warn-amber-text)" }}>
                Частота: {F_MIN_WORK}–{F_MAX_WORK} Гц
              </div>
            </Show>

            <div class="xo-row">
              <label class="xo-label" for="xo-type">Тип</label>
              <select
                id="xo-type"
                class="xo-select"
                value={filterType()}
                onChange={(e) => {
                  const ft = e.currentTarget.value as FilterType;
                  const prevType = filterType();
                  setFilterType(ft);
                  if (ft === "Gaussian") {
                    setShape(shape() || 1.0);
                  } else if (ft === "Custom") {
                    setCustomQ(customQ() || 0.707);
                  } else {
                    // Keep the same dB/oct where the new type offers it,
                    // else the closest available slope.
                    const prevSlope = orderToSlope(prevType, order());
                    const avail = availableSlopes(ft);
                    const pick = avail.reduce((best, s) => Math.abs(s - prevSlope) < Math.abs(best - prevSlope) ? s : best, avail[0]);
                    setOrder(slopeToOrder(ft, pick));
                  }
                }}
              >
                {FILTER_TYPE_ORDER.map((t) => <option value={t}>{FILTER_TYPE_LABEL[t]}</option>)}
              </select>
            </div>

            {/* Order — for non-Gaussian types */}
            <Show when={!isGaussian()}>
              <div class="xo-row">
                <label class="xo-label" for="xo-slope">Крутизна</label>
                <select
                  id="xo-slope"
                  class="xo-select"
                  value={orderToSlope(filterType(), order())}
                  onChange={(e) => setOrder(slopeToOrder(filterType(), parseInt(e.currentTarget.value, 10)))}
                >
                  {availableSlopes(filterType()).map((s) => (
                    <option value={s}>{`${s} дБ/окт`}</option>
                  ))}
                </select>
              </div>
            </Show>

            {/* M (shape) — only for Gaussian */}
            <Show when={isGaussian()}>
              <div class="xo-row">
                <label class="xo-label" for="xo-m">M</label>
                <NumField id="xo-m" value={shape()} onChange={setShape}
                  min={GAUSS_M_MIN} max={GAUSS_M_MAX} step={0.1} />
              </div>
            </Show>

            {/* Q — only for Custom */}
            <Show when={isCustom()}>
              <div class="xo-row">
                <label class="xo-label" for="xo-q">Q</label>
                <NumField id="xo-q" value={customQ()} onChange={setCustomQ}
                  min={CUSTOM_Q_MIN} max={CUSTOM_Q_MAX} step={0.01} />
              </div>
              <div class="xo-hint">{CUSTOM_Q_HINT}</div>
            </Show>

            <div class="xo-checkbox">
              <input
                type="checkbox"
                id="xo-lin-phase"
                checked={linearPhase()}
                onChange={(e) => setLinearPhase(e.currentTarget.checked)}
              />
              <label class="xo-checkbox-label" for="xo-lin-phase">
                Линейная фаза
              </label>
            </div>

            <div class="xo-buttons">
              <button class="dlg-btn" onClick={handleCancel}>Отмена</button>
              <button class="dlg-btn dlg-btn-primary" onClick={handleApply} disabled={!freqValid()}>Применить</button>
            </div>
        </Modal>
      )}
    </Show>
  );
}
