import { createSignal } from "solid-js";

export interface ToastEntry {
  id: number;
  text: string;
  kind: "info" | "warn";
}

const [_toasts, _setToasts] = createSignal<ToastEntry[]>([]);
export const toasts = _toasts;

let _nextId = 1;
const _timers = new Map<number, number>();

/** `durationMs = 0` keeps the toast until it is clicked (b141.74: the export
 *  note carries the number to type into the convolver — it must not vanish). */
export function showToast(text: string, kind: "info" | "warn" = "info", durationMs = 8000): void {
  const id = _nextId++;
  _setToasts([..._toasts(), { id, text, kind }]);
  if (durationMs <= 0) return;
  const handle = window.setTimeout(() => {
    _timers.delete(id);
    _setToasts(_toasts().filter((t) => t.id !== id));
  }, durationMs);
  _timers.set(id, handle);
}

export function dismissToast(id: number): void {
  const h = _timers.get(id);
  if (h !== undefined) {
    window.clearTimeout(h);
    _timers.delete(id);
  }
  _setToasts(_toasts().filter((t) => t.id !== id));
}
