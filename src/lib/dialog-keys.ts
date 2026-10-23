// b141.38: Enter/Escape for modal dialogs, one rule for all of them.
//
// The handlers sat on the overlay and ran the primary action on ANY Enter.
// A focused button receives Enter first and would activate itself, but the
// bubbling handler preventDefault()-ed that and ran its own action instead:
// Enter on «Не сохранять» saved the project, Enter on «Отмена» exported.
// Enter on a focused button now belongs to that button.
import { createSignal } from "solid-js";

export interface DialogKeyActions {
  onEnter?: () => void;
  onEscape?: () => void;
}

export function handleDialogKeys(e: KeyboardEvent, actions: DialogKeyActions): void {
  if (e.key === "Escape" && actions.onEscape) {
    e.preventDefault();
    actions.onEscape();
    return;
  }
  if (e.key !== "Enter" || !actions.onEnter) return;
  const t = e.target as HTMLElement | null;
  if (t && (t.tagName === "BUTTON" || t.tagName === "TEXTAREA" || t.getAttribute?.("role") === "button")) return;
  e.preventDefault();
  actions.onEnter();
}

// ---------------------------------------------------------------------------
// b141.105 (audit 2026-10-23 §5.3/5.4/5.14): one modal mechanism.
//
// Every dialog kept its own overlay and listener: Escape and a click beside
// the nested Baffle window also closed «Слияние NF+FF» behind it, app
// shortcuts (⌘Z/⌘O/⌘N) ran under five of them, and none had a dialog role,
// a focus trap or focus return. The stack below answers «which dialog is on
// top» and «is any open»; components/Modal.tsx is the only overlay.
// ---------------------------------------------------------------------------

const [modalStack, setModalStack] = createSignal<readonly symbol[]>([]);

/** Register an open modal; returns its token. Call `closeModal` on unmount. */
export function openModal(): symbol {
  const t = Symbol("modal");
  setModalStack((s) => [...s, t]);
  return t;
}

export function closeModal(t: symbol): void {
  setModalStack((s) => s.filter((x) => x !== t));
}

/** True while any modal dialog is open — app shortcuts are suspended. */
export const anyModalOpen = (): boolean => modalStack().length > 0;

/** Only the topmost modal reacts to Escape, Enter, Tab and backdrop clicks. */
export function isTopModal(t: symbol): boolean {
  const s = modalStack();
  return s.length > 0 && s[s.length - 1] === t;
}

const FOCUSABLE =
  'button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), ' +
  'textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])';

export function focusableIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE));
}

/** Keep Tab / Shift+Tab inside `root`. Returns true when it moved focus. */
export function trapTab(e: KeyboardEvent, root: HTMLElement): boolean {
  if (e.key !== "Tab") return false;
  const list = focusableIn(root);
  if (list.length === 0) { e.preventDefault(); root.focus(); return true; }
  const first = list[0];
  const last = list[list.length - 1];
  const active = document.activeElement as HTMLElement | null;
  const inside = active != null && root.contains(active);
  if (e.shiftKey && (!inside || active === first || active === root)) {
    e.preventDefault(); last.focus(); return true;
  }
  if (!e.shiftKey && (!inside || active === last)) {
    e.preventDefault(); first.focus(); return true;
  }
  return false;
}
