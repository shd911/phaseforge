import { createSignal, For, Show } from "solid-js";
import { ask } from "@tauri-apps/plugin-dialog";
import {
  appState,
  addBand,
  removeBand,
  setActiveBand,
  setActiveBandSum,
  moveBand,
  renameBand,
  isSum,
} from "../stores/bands";
import { peqStale } from "../stores/peq-optimize";
import { projectName } from "../lib/project-io";

export default function BandTabs() {
  const [dragIdx, setDragIdx] = createSignal<number | null>(null);
  const [overIdx, setOverIdx] = createSignal<number | null>(null);
  const [editingId, setEditingId] = createSignal<string | null>(null);

  let containerRef!: HTMLDivElement;
  let startX = 0;
  const DRAG_THRESHOLD = 5; // px до начала реального перетаскивания
  let didDrag = false;

  // b141.105 (audit 2026-10-23 §5.22): tab elements used to be kept in an
  // array by index, written once per element. After deleting a middle band
  // the survivors kept their old slots (and the last slot a removed node),
  // so a drop landed one place off: A,B,C,D → delete B → drag A onto C gave
  // C,D,A. Hit-testing now reads the live DOM and maps it by band id.
  function getTabIndexAtX(clientX: number): number | null {
    const els = containerRef.querySelectorAll<HTMLElement>(".band-tab[data-band-id]");
    for (const el of Array.from(els)) {
      const rect = el.getBoundingClientRect();
      if (clientX >= rect.left && clientX <= rect.right) {
        const i = appState.bands.findIndex((b) => b.id === el.dataset.bandId);
        return i >= 0 ? i : null;
      }
    }
    return null;
  }

  async function confirmRemove(id: string, name: string) {
    const confirmed = await ask(`Удалить полосу «${name}»?`, { title: "Удаление полосы", kind: "warning" });
    if (confirmed) removeBand(id);
  }

  function handlePointerDown(e: PointerEvent, idx: number) {
    // Игнорируем клик по кнопке закрытия
    if ((e.target as HTMLElement).closest(".band-tab-close")) return;

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

    const detach = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onCancel);
    };
    // b141.105 (§5.20): a cancelled pointer (system gesture, lost capture)
    // left the listeners attached and the tab half-dragged.
    const onCancel = () => {
      detach();
      setDragIdx(null);
      setOverIdx(null);
    };

    const onUp = () => {
      detach();

      const from = dragIdx();
      const to = overIdx();
      if (from != null && to != null && from !== to) {
        moveBand(from, to);
      }

      // Если не было настоящего перетаскивания — это обычный клик
      if (!didDrag) {
        const b = appState.bands[startIdx];
        if (b) setActiveBand(b.id);
      }

      setDragIdx(null);
      setOverIdx(null);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onCancel);
  }

  // b141.105 (§5.20): a band tab is a role="tab" element, not a <button> —
  // the «×» inside is a real button now, and a button may not contain one.
  return (
    <div class="band-tabs" ref={containerRef} role="tablist" aria-label="Полосы">
      <button
        class={`band-tab band-tab-sum ${isSum() ? "active" : ""}`}
        role="tab"
        aria-selected={isSum()}
        onClick={setActiveBandSum}
        title="Сумма всех полос"
      >{projectName() ?? "Сумма"}</button>
      <For each={appState.bands}>
        {(band, idx) => (
          <div
            role="tab"
            tabIndex={0}
            aria-selected={appState.activeBandId === band.id}
            data-band-id={band.id}
            class={`band-tab ${appState.activeBandId === band.id ? "active" : ""} ${dragIdx() === idx() ? "dragging" : ""} ${overIdx() === idx() && dragIdx() !== null && dragIdx() !== idx() ? "drag-over" : ""}`}
            onPointerDown={(e) => handlePointerDown(e, idx())}
            // b141.38: selection lived only in the pointer path — a focused
            // tab ignored Enter/Space.
            onKeyDown={(e) => {
              if (editingId() === band.id) return;
              if (e.target !== e.currentTarget) return; // the «×» handles its own keys
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setActiveBand(band.id); }
              if (e.key === "F2") { e.preventDefault(); setEditingId(band.id); }
            }}
            onDblClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setActiveBand(band.id);
              setEditingId(band.id);
            }}
          >
            {editingId() === band.id ? (
              <input
                class="band-tab-edit"
                value={band.name}
                ref={(el) => { requestAnimationFrame(() => { el.focus(); el.select(); }); }}
                onBlur={(e) => {
                  const v = (e.target as HTMLInputElement).value.trim();
                  if (v && v !== band.name) renameBand(band.id, v);
                  setEditingId(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                  if (e.key === "Escape") { setEditingId(null); }
                }}
                onClick={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
              />
            ) : (
              <>
                <span class="band-tab-label">{band.name}</span>
                <Show when={peqStale(band)}>
                  <span class="band-tab-stale-dot" title="PEQ устарел">●</span>
                </Show>
              </>
            )}
            <Show when={appState.bands.length > 1}>
              <button
                type="button"
                class="band-tab-close"
                aria-label={`Удалить полосу ${band.name}`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  confirmRemove(band.id, band.name);
                }}
                title={`Удалить полосу ${band.name}`}
              >×</button>
            </Show>
          </div>
        )}
      </For>
      <button class="band-tab band-tab-add" onClick={addBand} title="Добавить полосу" aria-label="Добавить полосу">+</button>
    </div>
  );
}
