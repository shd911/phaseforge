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
  preRingPeakSpl, setPreRingPeakSpl,
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
const SPL = { min: 60, max: 130 };
// b141.148: the noise floor is also shown and typed as a word length —
// 1 bit = 20·log10(2) dB. Both fields edit the one dB value.
const DB_PER_BIT = 20 * Math.log10(2);
const round1 = (v: number) => Math.round(v * 10) / 10;
const dbToBits = (db: number) => round1(-db / DB_PER_BIT);
const bitsToDb = (bits: number) => clampNum(round1(-bits * DB_PER_BIT), FLOOR.min, FLOOR.max);
const BITS = { min: dbToBits(FLOOR.max), max: dbToBits(FLOOR.min) };

const [visible, setVisible] = createSignal(false);

export function openFirSettings() { setVisible(true); }
export function closeFirSettings() { setVisible(false); }

export default function FirSettingsDialog() {
  // Local state synced on open
  const [iterations, setIterations] = createSignal(3);
  const [freqWeight, setFreqWeight] = createSignal(true);
  const [maxBoost, setMaxBoost] = createSignal(24.0);
  const [noiseFloor, setNoiseFloor] = createSignal(-150.0);
  const [peakSpl, setPeakSpl] = createSignal(105);

  function initFromStore() {
    setIterations(firIterations());
    setFreqWeight(firFreqWeighting());
    setMaxBoost(firMaxBoost());
    setNoiseFloor(firNoiseFloor());
    setPeakSpl(preRingPeakSpl());
  }

  function handleApply() {
    setFirIterations(Math.round(clampNum(iterations(), ITER.min, ITER.max)));
    setFirFreqWeighting(freqWeight());
    setFirMaxBoost(clampNum(maxBoost(), BOOST.min, BOOST.max));
    setFirNoiseFloor(clampNum(noiseFloor(), FLOOR.min, FLOOR.max));
    setPreRingPeakSpl(clampNum(peakSpl(), SPL.min, SPL.max));
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
            <NumField id="fir-noise-floor-bits" label={L("Шумовой порог в битах", "Noise floor in bits")}
              value={dbToBits(noiseFloor())} onChange={(b) => setNoiseFloor(bitsToDb(b))}
              min={BITS.min} max={BITS.max} step={0.1} />
            <span class="xo-unit">{L("бит", "bit")}</span>
          </div>
          <div class="xo-hint">{L("Коррекция ниже этого уровня игнорируется. Поднимите при шумном измерении. 1 бит = 6,02 дБ: 16 бит ≈ −96 дБ, 24 бита ≈ −144,5 дБ", "Correction below this level is ignored. Raise it for a noisy measurement. 1 bit = 6.02 dB: 16 bit ≈ −96 dB, 24 bit ≈ −144.5 dB")}</div>

          {/* b141.153: pre-ring audibility level */}
          <div class="xo-row">
            <label class="xo-label" for="prering-peak-spl">{L("Пик у слушателя", "Peak at the listener")}</label>
            <NumField id="prering-peak-spl" value={peakSpl()} onChange={setPeakSpl}
              min={SPL.min} max={SPL.max} step={1} />
            <span class="xo-unit">{L("дБ SPL", "dB SPL")}</span>
          </div>
          <div class="xo-hint">{L("Пиковый уровень транзиента в месте прослушивания. По нему оценивается слышимость предзвона фазолинейных фильтров (строка «Предзвон» в блоках ФВЧ/ФНЧ): на 6 дБ громче — на 6 дБ ближе к порогу", "Transient peak level at the listening seat. The pre-ring audibility of linear-phase filters («Pre-ring» line in the HP/LP blocks) is judged at it: 6 dB louder — 6 dB closer to the threshold")}</div>

          <div class="xo-buttons">
            <button class="dlg-btn" onClick={handleCancel}>{L("Отмена", "Cancel")}</button>
            <button class="dlg-btn dlg-btn-primary" onClick={handleApply}>{L("Применить", "Apply")}</button>
          </div>
      </Modal>
    </Show>
  );
}
