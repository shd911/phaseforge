import { createSignal, Show } from "solid-js";
import { F_MIN_WORK, F_MAX_WORK } from "../lib/types";
import type { FilterConfig, FilterType } from "../lib/types";
import { appState, setBandLowPass } from "../stores/bands";
import { availableSlopes, orderToSlope, slopeToOrder } from "../lib/slope";
import { handleDialogKeys } from "../lib/dialog-keys";

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
      shape: isGaussian() ? shape() : null,
      linear_phase: isGaussian() ? true : linearPhase(),
      q: isCustom() ? customQ() : null,
    };

    // Set LP on the band — propagation handles the HP on the next band
    setBandLowPass(data.bandId, config);
    closeCrossoverDialog();
  }

  function handleCancel() {
    closeCrossoverDialog();
  }

  function handleKeyDown(e: KeyboardEvent) {
    handleDialogKeys(e, { onEnter: handleApply, onEscape: handleCancel });
  }

  // b141.38: `parseFloat(v) || 0` rewrote a cleared field to 0 on the first
  // Backspace. An unparsable draft leaves the value alone (freq goes NaN so
  // Apply stays disabled and the hint explains why).
  const finite = (v: string): number | null => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
  };

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
        <div class="xo-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) handleCancel(); }} onKeyDown={handleKeyDown}>
          <div class="xo-dialog">
            <div class="xo-title">
              Кроссовер: {data().bandName} / {data().nextBandName}
            </div>

            <div class="xo-row">
              <span class="xo-label">Частота</span>
              <input
                ref={(el) => requestAnimationFrame(() => { el.focus(); el.select(); })}
                class="xo-input"
                type="number"
                min="20"
                max={F_MAX_WORK}
                step="1"
                value={freq()}
                onInput={(e) => setFreq(finite(e.currentTarget.value) ?? NaN)}
                onKeyDown={handleKeyDown}
              />
              <span class="xo-unit">Hz</span>
            </div>
            <Show when={!freqValid()}>
              <div class="xo-hint" style={{ color: "var(--warn-amber-text)" }}>
                Частота: {F_MIN_WORK}–{F_MAX_WORK} Hz
              </div>
            </Show>

            <div class="xo-row">
              <span class="xo-label">Тип</span>
              <select
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
                <option value="LinkwitzRiley">Linkwitz-Riley</option>
                <option value="Butterworth">Butterworth</option>
                <option value="Bessel">Bessel</option>
                <option value="Gaussian">Gaussian</option>
                <option value="Custom">Произвольный Q</option>
              </select>
            </div>

            {/* Order — for non-Gaussian types */}
            <Show when={!isGaussian()}>
              <div class="xo-row">
                <span class="xo-label">Крутизна</span>
                <select
                  class="xo-select"
                  value={orderToSlope(filterType(), order())}
                  onChange={(e) => setOrder(slopeToOrder(filterType(), parseInt(e.currentTarget.value, 10)))}
                >
                  {availableSlopes(filterType()).map((s) => (
                    <option value={s}>{`${s} dB/oct`}</option>
                  ))}
                </select>
              </div>
            </Show>

            {/* M (shape) — only for Gaussian */}
            <Show when={isGaussian()}>
              <div class="xo-row">
                <span class="xo-label">M</span>
                <input
                  class="xo-input"
                  type="number"
                  min="0.5"
                  max="10"
                  step="0.1"
                  value={shape()}
                  onInput={(e) => { const n = finite(e.currentTarget.value); if (n != null) setShape(n); }}
                  onKeyDown={handleKeyDown}
                />
              </div>
            </Show>

            {/* Q — only for Custom */}
            <Show when={isCustom()}>
              <div class="xo-row">
                <span class="xo-label">Q</span>
                <input
                  class="xo-input"
                  type="number"
                  min="0.1"
                  max="20"
                  step="0.01"
                  value={customQ()}
                  onInput={(e) => { const n = finite(e.currentTarget.value); if (n != null) setCustomQ(n); }}
                  onKeyDown={handleKeyDown}
                />
              </div>
              <div class="xo-hint">
                0.500 = LR · 0.577 = Bessel · 0.707 = BW · &gt;0.707 = резонанс
              </div>
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
          </div>
        </div>
      )}
    </Show>
  );
}
