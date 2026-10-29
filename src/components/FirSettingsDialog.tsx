import { createSignal, Show } from "solid-js";
import Modal from "./Modal";
import NumField from "./NumField";
import { clampNum } from "../lib/slope";
import {
  firIterations, setFirIterations,
  firFreqWeighting, setFirFreqWeighting,
  firMaxBoost, setFirMaxBoost,
  firNoiseFloor, setFirNoiseFloor,
} from "../stores/bands";

// b141.105 (audit 2026-10-23 §1.6): «Ограничение узкополосного подъёма» and
// its two fields are gone from this window — narrowband_limit / nb_* are read
// only by the legacy path, the production FIR never looked at them. The
// fields stay in the store and the project file for compatibility.
//
// §5.26: fields are NumField — parse on input, clamp on blur/Apply, and the
// field always shows what Apply will save (an emptied field used to stay
// empty while the previous value was saved).

const ITER = { min: 0, max: 20 };
const BOOST = { min: 0, max: 60 };
const FLOOR = { min: -200, max: -40 };

const [visible, setVisible] = createSignal(false);

export function openFirSettings() { setVisible(true); }
export function closeFirSettings() { setVisible(false); }

export default function FirSettingsDialog() {
  // Local state synced on open
  const [iterations, setIterations] = createSignal(3);
  const [freqWeight, setFreqWeight] = createSignal(true);
  const [maxBoost, setMaxBoost] = createSignal(24.0);
  const [noiseFloor, setNoiseFloor] = createSignal(-150.0);

  function initFromStore() {
    setIterations(firIterations());
    setFreqWeight(firFreqWeighting());
    setMaxBoost(firMaxBoost());
    setNoiseFloor(firNoiseFloor());
  }

  function handleApply() {
    setFirIterations(Math.round(clampNum(iterations(), ITER.min, ITER.max)));
    setFirFreqWeighting(freqWeight());
    setFirMaxBoost(clampNum(maxBoost(), BOOST.min, BOOST.max));
    setFirNoiseFloor(clampNum(noiseFloor(), FLOOR.min, FLOOR.max));
    closeFirSettings();
  }

  function handleCancel() {
    closeFirSettings();
  }

  return (
    <Show when={(() => { const v = visible(); if (v) initFromStore(); return v; })()}>
      <Modal overlayClass="xo-overlay" dialogClass="xo-dialog" dialogStyle={{ "min-width": "360px" }}
        titleId="fir-dlg-title" onEnter={handleApply} onEscape={handleCancel}>
          <div class="xo-title" id="fir-dlg-title">Настройки оптимизации</div>

          {/* Iterations */}
          <div class="xo-row">
            <label class="xo-label" for="fir-iter">Итерации (WLS)</label>
            <NumField id="fir-iter" autofocus integer value={iterations()} onChange={setIterations}
              min={ITER.min} max={ITER.max} step={1} />
          </div>
          <div class="xo-hint">Проходы итеративной коррекции ошибки. 0 = выкл, 3–5 оптимально</div>

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
          <div class="xo-hint">Приоритет зонам кроссовера и речи (200 Гц – 4 кГц)</div>

          {/* Max boost */}
          <div class="xo-row">
            <label class="xo-label" for="fir-max-boost">Макс. подъём</label>
            <NumField id="fir-max-boost" value={maxBoost()} onChange={setMaxBoost}
              min={BOOST.min} max={BOOST.max} step={1} />
            <span class="xo-unit">дБ</span>
          </div>
          <div class="xo-hint">Ограничивает подъём цели и PEQ. Не действует на полосы, которые строятся аналитическим каскадом (мин-фаза без subsonic и гауссов) — экспорт об этом предупредит</div>

          {/* Noise floor */}
          <div class="xo-row">
            <label class="xo-label" for="fir-noise-floor">Шумовой порог</label>
            <NumField id="fir-noise-floor" value={noiseFloor()} onChange={setNoiseFloor}
              min={FLOOR.min} max={FLOOR.max} step={5} />
            <span class="xo-unit">дБ</span>
          </div>
          <div class="xo-hint">Коррекция ниже этого уровня игнорируется. Поднимите при шумном измерении</div>

          <div class="xo-buttons">
            <button class="dlg-btn" onClick={handleCancel}>Отмена</button>
            <button class="dlg-btn dlg-btn-primary" onClick={handleApply}>Применить</button>
          </div>
      </Modal>
    </Show>
  );
}
