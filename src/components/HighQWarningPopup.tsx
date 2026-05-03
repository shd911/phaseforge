import { createSignal, Show, onCleanup } from "solid-js";
import type { PeqBand } from "../lib/types";
import { qWarnAt } from "../lib/peq-quality";

interface HighQContext {
  band: PeqBand;
  index: number;
}

const [_state, _setState] = createSignal<HighQContext | null>(null);

export function openHighQPopup(band: PeqBand, index: number): void {
  _setState({ band, index });
}

export const isHighQPopupOpen = () => _state() !== null;

function close() { _setState(null); }

export default function HighQWarningPopup() {
  const onKey = (e: KeyboardEvent) => {
    if (!_state()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  };
  window.addEventListener("keydown", onKey);
  onCleanup(() => window.removeEventListener("keydown", onKey));

  return (
    <Show when={_state()}>
      {(ctx) => (
        <div class="pn-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
          <div class="pn-dialog" style={{ "min-width": "420px", "max-width": "520px" }}>
            <div class="pn-title">Высокая добротность</div>
            <div class="pn-label" style={{ "margin-bottom": "8px" }}>
              Полоса {ctx().index + 1}: {Math.round(ctx().band.freq_hz)} Гц
            </div>
            <div class="pn-label" style={{ "margin-bottom": "4px" }}>
              Q = {ctx().band.q.toFixed(2)}
            </div>
            <div class="pn-label" style={{ "margin-bottom": "16px", color: "#aaa", "font-size": "12px" }}>
              Порог предупреждения: Q ≤ {qWarnAt(ctx().band.freq_hz).toFixed(1)}
            </div>
            <div style={{ "margin-bottom": "16px", "font-size": "13px", "line-height": "1.5" }}>
              Узкая полоса с высокой добротностью даёт заметный звон на импульсе и
              часто корректирует артефакт замера, а не реальную особенность системы.
              Рекомендуется уменьшить Q вручную или применить notch с более широкой полосой.
            </div>
            <div class="pn-buttons">
              <button class="dlg-btn" onClick={close}>Закрыть</button>
            </div>
          </div>
        </div>
      )}
    </Show>
  );
}
