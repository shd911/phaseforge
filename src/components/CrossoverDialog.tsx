import { createSignal, Show } from "solid-js";
import type { FilterConfig, FilterType } from "../lib/types";
import { appState, setBandLowPass } from "../stores/bands";

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

  function handleApply() {
    const data = dialogData();
    if (!data) return;

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
    if (e.key === "Escape") {
      e.preventDefault();
      handleCancel();
    } else if (e.key === "Enter") {
      e.preventDefault();
      handleApply();
    }
  }

  // Available orders per filter type
  function availableOrders(): number[] {
    const ft = filterType();
    if (ft === "LinkwitzRiley") return [2, 4, 8];
    if (ft === "Custom") return [1, 2, 3, 4, 5, 6, 7, 8];
    return [1, 2, 3, 4, 5, 6, 7, 8]; // Butterworth, Bessel
  }

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
              Crossover: {data().bandName} / {data().nextBandName}
            </div>

            <div class="xo-row">
              <span class="xo-label">Frequency</span>
              <input
                ref={(el) => requestAnimationFrame(() => { el.focus(); el.select(); })}
                class="xo-input"
                type="number"
                min="20"
                max="20000"
                step="1"
                value={freq()}
                onInput={(e) => setFreq(parseFloat(e.currentTarget.value) || 0)}
                onKeyDown={handleKeyDown}
              />
              <span class="xo-unit">Hz</span>
            </div>

            <div class="xo-row">
              <span class="xo-label">Type</span>
              <select
                class="xo-select"
                value={filterType()}
                onChange={(e) => {
                  const ft = e.currentTarget.value as FilterType;
                  setFilterType(ft);
                  if (ft === "Gaussian") {
                    setShape(shape() || 1.0);
                  } else if (ft === "Custom") {
                    setCustomQ(customQ() || 0.707);
                  } else {
                    const avail = ft === "LinkwitzRiley" ? [2, 4, 8] :
                                  [1, 2, 3, 4, 5, 6, 7, 8];
                    if (!avail.includes(order())) {
                      setOrder(avail[1] ?? avail[0]);
                    }
                  }
                }}
              >
                <option value="LinkwitzRiley">Linkwitz-Riley</option>
                <option value="Butterworth">Butterworth</option>
                <option value="Bessel">Bessel</option>
                <option value="Gaussian">Gaussian</option>
                <option value="Custom">Custom Q</option>
              </select>
            </div>

            {/* Order — for non-Gaussian types */}
            <Show when={!isGaussian()}>
              <div class="xo-row">
                <span class="xo-label">Order</span>
                <select
                  class="xo-select"
                  value={order()}
                  onChange={(e) => setOrder(parseInt(e.currentTarget.value))}
                >
                  {availableOrders().map((o) => (
                    <option value={o}>{o}</option>
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
                  onInput={(e) => setShape(parseFloat(e.currentTarget.value) || 1.0)}
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
                  value={customQ().toFixed(3)}
                  onInput={(e) => setCustomQ(parseFloat(e.currentTarget.value) || 0.707)}
                  onKeyDown={handleKeyDown}
                />
              </div>
              <div class="xo-hint">
                0.500 = LR · 0.577 = Bessel · 0.707 = BW · &gt;0.707 = resonant
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
                Linear Phase
              </label>
            </div>

            <div class="xo-buttons">
              <button class="dlg-btn" onClick={handleCancel}>Cancel</button>
              <button class="dlg-btn dlg-btn-primary" onClick={handleApply}>Apply</button>
            </div>
          </div>
        </div>
      )}
    </Show>
  );
}
