// b141.38: Enter/Escape for modal dialogs, one rule for all of them.
//
// The handlers sat on the overlay and ran the primary action on ANY Enter.
// A focused button receives Enter first and would activate itself, but the
// bubbling handler preventDefault()-ed that and ran its own action instead:
// Enter on «Не сохранять» saved the project, Enter on «Отмена» exported.
// Enter on a focused button now belongs to that button.

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
  if (t && (t.tagName === "BUTTON" || t.getAttribute?.("role") === "button")) return;
  e.preventDefault();
  actions.onEnter();
}
