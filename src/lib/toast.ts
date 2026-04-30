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

export function showToast(text: string, kind: "info" | "warn" = "info", durationMs = 8000): void {
  const id = _nextId++;
  _setToasts([..._toasts(), { id, text, kind }]);
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
