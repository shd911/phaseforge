import { createSignal, For, onCleanup } from "solid-js";
import {
  appState,
  addBand,
  setActiveBand,
  setActiveBandSum,
  moveBand,
  isSum,
} from "../stores/bands";

export default function BandTabs() {
  const [dragIdx, setDragIdx] = createSignal<number | null>(null);
  const [overIdx, setOverIdx] = createSignal<number | null>(null);

  let containerRef!: HTMLDivElement;
  let tabEls: HTMLButtonElement[] = [];
  let startX = 0;
  const DRAG_THRESHOLD = 5; // px до начала реального перетаскивания
  let didDrag = false;

  function getTabIndexAtX(clientX: number): number | null {
    for (let i = 0; i < tabEls.length; i++) {
      const el = tabEls[i];
      if (!el) continue;
      const rect = el.getBoundingClientRect();
      if (clientX >= rect.left && clientX <= rect.right) return i;
    }
    return null;
  }

  function handlePointerDown(e: PointerEvent, idx: number) {
    // Игнорируем клик по кнопке закрытия
    if ((e.target as HTMLElement).classList.contains("band-tab-close")) return;

    startX = e.clientX;
    didDrag = false;
    const startIdx = idx;

    const onMove = (ev: PointerEvent) => {
      if (!didDrag && Math.abs(ev.clientX - startX) < DRAG_THRESHOLD) return;

      if (!didDrag) {
        didDrag = true;
        setDragIdx(startIdx);
      }

      const over = getTabIndexAtX(ev.clientX);
      setOverIdx(over);
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);

      const from = dragIdx();
      const to = overIdx();
      if (from != null && to != null && from !== to) {
        moveBand(from, to);
      }

      // Если не было настоящего перетаскивания — это обычный клик
      if (!didDrag) {
        setActiveBand(appState.bands[startIdx].id);
      }

      setDragIdx(null);
      setOverIdx(null);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  return (
    <div class="band-tabs" ref={containerRef}>
      <For each={appState.bands}>
        {(band, idx) => (
          <button
            ref={(el) => { tabEls[idx()] = el; }}
            class={`band-tab ${appState.activeBandId === band.id ? "active" : ""} ${dragIdx() === idx() ? "dragging" : ""} ${overIdx() === idx() && dragIdx() !== null && dragIdx() !== idx() ? "drag-over" : ""}`}
            onPointerDown={(e) => handlePointerDown(e, idx())}
          >
            <span class="band-tab-label">{band.name}</span>
          </button>
        )}
      </For>
      <button class="band-tab band-tab-add" onClick={addBand} title="Add band">+</button>
      <button
        class={`band-tab band-tab-sum ${isSum() ? "active" : ""}`}
        onClick={setActiveBandSum}
      >SUM</button>
    </div>
  );
}
