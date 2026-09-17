// b141.38: Enter on a focused dialog button must activate that button, not
// the dialog's primary action (Enter on «Не сохранять» used to save).
import { describe, it, expect, vi } from "vitest";
import { handleDialogKeys } from "../dialog-keys";

function press(key: string, target: HTMLElement) {
  const e = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
  Object.defineProperty(e, "target", { value: target });
  return e;
}

describe("handleDialogKeys", () => {
  it("Enter in a field runs the primary action", () => {
    const onEnter = vi.fn();
    const e = press("Enter", document.createElement("input"));
    handleDialogKeys(e, { onEnter });
    expect(onEnter).toHaveBeenCalledOnce();
    expect(e.defaultPrevented).toBe(true);
  });

  it("Enter on a focused button is left to the button", () => {
    const onEnter = vi.fn();
    const btn = document.createElement("button");
    const e = press("Enter", btn);
    handleDialogKeys(e, { onEnter });
    expect(onEnter).not.toHaveBeenCalled();
    expect(e.defaultPrevented).toBe(false);

    const span = document.createElement("span");
    span.setAttribute("role", "button");
    handleDialogKeys(press("Enter", span), { onEnter });
    expect(onEnter).not.toHaveBeenCalled();
  });

  it("Escape cancels from anywhere, including a focused button", () => {
    const onEscape = vi.fn();
    handleDialogKeys(press("Escape", document.createElement("button")), { onEscape });
    expect(onEscape).toHaveBeenCalledOnce();
  });
});
