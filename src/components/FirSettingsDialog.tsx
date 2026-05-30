import { createSignal, Show } from "solid-js";
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
    setFirIterations(iterations());
    setFirFreqWeighting(freqWeight());
    setFirNarrowbandLimit(nbLimit());
    setFirNbSmoothingOct(nbSmoothing());
    setFirNbMaxExcess(nbExcess());
    setFirMaxBoost(maxBoost());
    setFirNoiseFloor(noiseFloor());
    closeFirSettings();
  }

  function handleCancel() {
    closeFirSettings();
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") { e.preventDefault(); handleCancel(); }
    else if (e.key === "Enter") { e.preventDefault(); handleApply(); }
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
              onInput={(e) => setIterations(parseInt(e.currentTarget.value) || 0)}
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
                onInput={(e) => setNbSmoothing(parseFloat(e.currentTarget.value) || 0.333)}
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
                onInput={(e) => setNbExcess(parseFloat(e.currentTarget.value) || 6.0)}
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
              onInput={(e) => setMaxBoost(parseFloat(e.currentTarget.value) || 24.0)}
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
              onInput={(e) => setNoiseFloor(parseFloat(e.currentTarget.value) || -150.0)}
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
