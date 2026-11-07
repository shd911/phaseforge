import { createSignal, Show } from "solid-js";
import Modal from "./Modal";
import NumField from "./NumField";
import { clampNum } from "../lib/slope";
import { L } from "../lib/i18n";
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
          <div class="xo-title" id="fir-dlg-title">{L("Настройки оптимизации", "Optimization settings")}</div>

          {/* Iterations */}
          <div class="xo-row">
            <label class="xo-label" for="fir-iter">{L("Итерации (WLS)", "Iterations (WLS)")}</label>
            <NumField id="fir-iter" autofocus integer value={iterations()} onChange={setIterations}
              min={ITER.min} max={ITER.max} step={1} />
          </div>
          <div class="xo-hint">{L("Проходы итеративной коррекции ошибки. 0 = выкл, 3–5 оптимально", "Iterative error-correction passes. 0 = off, 3–5 is best")}</div>

          {/* Freq weighting */}
          <div class="xo-checkbox">
            <input
              type="checkbox" id="fir-freq-weight"
              checked={freqWeight()}
              onChange={(e) => setFreqWeight(e.currentTarget.checked)}
            />
            <label class="xo-checkbox-label" for="fir-freq-weight">
              {L("Частотно-зависимое взвешивание", "Frequency weighting")}
            </label>
          </div>
          <div class="xo-hint">{L("Приоритет зонам кроссовера и речи (200 Гц – 4 кГц)", "Favours crossover and speech regions (200 Hz – 4 kHz)")}</div>

          {/* Max boost */}
          <div class="xo-row">
            <label class="xo-label" for="fir-max-boost">{L("Макс. подъём", "Max boost")}</label>
            <NumField id="fir-max-boost" value={maxBoost()} onChange={setMaxBoost}
              min={BOOST.min} max={BOOST.max} step={1} />
            <span class="xo-unit">{L("дБ", "dB")}</span>
          </div>
          <div class="xo-hint">{L("Ограничивает подъём цели и PEQ. Не действует на полосы, которые строятся аналитическим каскадом (мин-фаза без subsonic и гауссов) — экспорт об этом предупредит", "Caps target and PEQ boost. Not applied to bands built by the analytical cascade (min phase, no subsonic or Gaussian); export warns about it")}</div>

          {/* Noise floor */}
          <div class="xo-row">
            <label class="xo-label" for="fir-noise-floor">{L("Шумовой порог", "Noise floor")}</label>
            <NumField id="fir-noise-floor" value={noiseFloor()} onChange={setNoiseFloor}
              min={FLOOR.min} max={FLOOR.max} step={5} />
            <span class="xo-unit">{L("дБ", "dB")}</span>
          </div>
          <div class="xo-hint">{L("Коррекция ниже этого уровня игнорируется. Поднимите при шумном измерении", "Correction below this level is ignored. Raise it for a noisy measurement")}</div>

          <div class="xo-buttons">
            <button class="dlg-btn" onClick={handleCancel}>{L("Отмена", "Cancel")}</button>
            <button class="dlg-btn dlg-btn-primary" onClick={handleApply}>{L("Применить", "Apply")}</button>
          </div>
      </Modal>
    </Show>
  );
}
