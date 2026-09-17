import { createSignal, Show } from "solid-js";

import { handleDialogKeys } from "../lib/dialog-keys";

/** Parse a numeric draft: keep the previous value on garbage. (`parseFloat(v)
 *  || default` turned a typed 0 into the default — 2026-09-05 audit.) The
 *  field's min/max is applied on blur and Apply, not per keystroke: clamping
 *  while typing turned the "0" of "0.1" into 0.05 (b141.38). */
function numDraft(v: string, prev: number, integer = false): number {
  const n = integer ? parseInt(v, 10) : parseFloat(v);
  return Number.isFinite(n) ? n : prev;
}
const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));
import {
  firIterations, setFirIterations,
  firFreqWeighting, setFirFreqWeighting,
  firNarrowbandLimit, setFirNarrowbandLimit,
  firNbSmoothingOct, setFirNbSmoothingOct,
  firNbMaxExcess, setFirNbMaxExcess,
  firMaxBoost, setFirMaxBoost,
  firNoiseFloor, setFirNoiseFloor,
} from "../stores/bands";

const [visible, setVisible] = createSignal(false);

export function openFirSettings() { setVisible(true); }
export function closeFirSettings() { setVisible(false); }

export default function FirSettingsDialog() {
  // Local state synced on open
  const [iterations, setIterations] = createSignal(3);
  const [freqWeight, setFreqWeight] = createSignal(true);
  const [nbLimit, setNbLimit] = createSignal(true);
  const [nbSmoothing, setNbSmoothing] = createSignal(0.333);
  const [nbExcess, setNbExcess] = createSignal(6.0);
  const [maxBoost, setMaxBoost] = createSignal(24.0);
  const [noiseFloor, setNoiseFloor] = createSignal(-150.0);

  function initFromStore() {
    setIterations(firIterations());
    setFreqWeight(firFreqWeighting());
    setNbLimit(firNarrowbandLimit());
    setNbSmoothing(firNbSmoothingOct());
    setNbExcess(firNbMaxExcess());
    setMaxBoost(firMaxBoost());
    setNoiseFloor(firNoiseFloor());
  }

  function handleApply() {
    setFirIterations(clamp(iterations(), 0, 20));
    setFirFreqWeighting(freqWeight());
    setFirNarrowbandLimit(nbLimit());
    setFirNbSmoothingOct(clamp(nbSmoothing(), 0.05, 2));
    setFirNbMaxExcess(clamp(nbExcess(), 1, 24));
    setFirMaxBoost(clamp(maxBoost(), 0, 60));
    setFirNoiseFloor(clamp(noiseFloor(), -200, -40));
    closeFirSettings();
  }

  function handleCancel() {
    closeFirSettings();
  }

  function handleKeyDown(e: KeyboardEvent) {
    handleDialogKeys(e, { onEnter: handleApply, onEscape: handleCancel });
  }

  return (
    <Show when={(() => { const v = visible(); if (v) initFromStore(); return v; })()}>
      <div class="xo-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) handleCancel(); }} onKeyDown={handleKeyDown}>
        <div class="xo-dialog" style="min-width: 360px">
          <div class="xo-title">Настройки оптимизации</div>

          {/* Iterations */}
          <div class="xo-row">
            <span class="xo-label">Итерации (WLS)</span>
            <input
              class="xo-input"
              type="number" min="0" max="20" step="1"
              value={iterations()}
              onInput={(e) => setIterations(numDraft(e.currentTarget.value, iterations(), true))}
              onBlur={() => setIterations(clamp(iterations(), 0, 20))}
              onKeyDown={handleKeyDown}
              ref={(el) => requestAnimationFrame(() => { el.focus(); el.select(); })}
            />
          </div>
          <div class="xo-hint">Проходы итеративной коррекции ошибки. 0 = выкл, 3-5 оптимально</div>

          {/* Freq weighting */}
          <div class="xo-checkbox">
            <input
              type="checkbox" id="fir-freq-weight"
              checked={freqWeight()}
              onChange={(e) => setFreqWeight(e.currentTarget.checked)}
            />
            <label class="xo-checkbox-label" for="fir-freq-weight">
              Частотно-зависимое взвешивание
            </label>
          </div>
          <div class="xo-hint">Приоритет зонам кроссовера и речи (200-4k Hz)</div>

          {/* Narrowband limit */}
          <div class="xo-checkbox">
            <input
              type="checkbox" id="fir-nb-limit"
              checked={nbLimit()}
              onChange={(e) => setNbLimit(e.currentTarget.checked)}
            />
            <label class="xo-checkbox-label" for="fir-nb-limit">
              Ограничение узкополосного подъёма
            </label>
          </div>
          <div class="xo-hint">Срезает острые пики коррекции выше сглаженной кривой</div>

          {/* NB smoothing — only when NB limit is on */}
          <Show when={nbLimit()}>
            <div class="xo-row">
              <span class="xo-label">Сглаживание (узкопол.)</span>
              <input
                class="xo-input"
                type="number" min="0.05" max="2" step="0.01"
                value={nbSmoothing()}
                onInput={(e) => setNbSmoothing(numDraft(e.currentTarget.value, nbSmoothing()))}
                onBlur={() => setNbSmoothing(clamp(nbSmoothing(), 0.05, 2))}
                onKeyDown={handleKeyDown}
              />
              <span class="xo-unit">oct</span>
            </div>
            <div class="xo-hint">Ширина окна сглаживания. Меньше = точнее, больше = агрессивнее</div>

            <div class="xo-row">
              <span class="xo-label">Макс. превышение (узкопол.)</span>
              <input
                class="xo-input"
                type="number" min="1" max="24" step="0.5"
                value={nbExcess()}
                onInput={(e) => setNbExcess(numDraft(e.currentTarget.value, nbExcess()))}
                onBlur={() => setNbExcess(clamp(nbExcess(), 1, 24))}
                onKeyDown={handleKeyDown}
              />
              <span class="xo-unit">dB</span>
            </div>
            <div class="xo-hint">Макс. подъём над сглаженной кривой коррекции</div>
          </Show>

          {/* Max boost */}
          <div class="xo-row">
            <span class="xo-label">Макс. подъём</span>
            <input
              class="xo-input"
              type="number" min="0" max="60" step="1"
              value={maxBoost()}
              onInput={(e) => setMaxBoost(numDraft(e.currentTarget.value, maxBoost()))}
              onBlur={() => setMaxBoost(clamp(maxBoost(), 0, 60))}
              onKeyDown={handleKeyDown}
            />
            <span class="xo-unit">dB</span>
          </div>
          <div class="xo-hint">Глобальный лимит подъёма. Ниже = безопаснее для усилителя</div>

          {/* Noise floor */}
          <div class="xo-row">
            <span class="xo-label">Шумовой порог</span>
            <input
              class="xo-input"
              type="number" min="-200" max="-40" step="5"
              value={noiseFloor()}
              onInput={(e) => setNoiseFloor(numDraft(e.currentTarget.value, noiseFloor()))}
              onBlur={() => setNoiseFloor(clamp(noiseFloor(), -200, -40))}
              onKeyDown={handleKeyDown}
            />
            <span class="xo-unit">dB</span>
          </div>
          <div class="xo-hint">Коррекция ниже этого уровня игнорируется. Поднимите при шумном измерении</div>

          <div class="xo-buttons">
            <button class="dlg-btn" onClick={handleCancel}>Отмена</button>
            <button class="dlg-btn dlg-btn-primary" onClick={handleApply}>Применить</button>
          </div>
        </div>
      </div>
    </Show>
  );
}
