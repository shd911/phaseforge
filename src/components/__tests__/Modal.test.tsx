// b141.105 (audit 2026-10-23 §5.3/5.4/5.14): one modal mechanism — the
// topmost dialog alone reacts to Escape / backdrop, app shortcuts see that a
// modal is open, dialogs carry their role and keep Tab inside.
import { describe, it, expect, vi, afterEach } from "vitest";
import { createSignal, Show } from "solid-js";
import { render } from "solid-js/web";
import Modal from "../Modal";
import { anyModalOpen } from "../../lib/dialog-keys";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

const key = (el: Element, k: string, init: KeyboardEventInit = {}) =>
  el.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...init }));
const down = (el: Element) => el.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));

/** Mirrors MergeDialog + BaffleStepDialog: the inner modal renders inside
 *  the outer one's children. */
function mountNested() {
  const [outer, setOuter] = createSignal(true);
  const [inner, setInner] = createSignal(true);
  const closeOuter = vi.fn(() => setOuter(false));
  const closeInner = vi.fn(() => setInner(false));
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => (
    <Show when={outer()}>
      <Modal overlayClass="merge-overlay" dialogClass="merge-dialog" titleId="t-outer" onEscape={closeOuter}>
        <span id="t-outer">Слияние</span>
        <button id="outer-btn">ok</button>
        <Show when={inner()}>
          <Modal overlayClass="baffle-overlay" dialogClass="baffle-dialog" titleId="t-inner" onEscape={closeInner}>
            <span id="t-inner">Baffle</span>
            <button id="inner-a">a</button>
            <button id="inner-b">b</button>
          </Modal>
        </Show>
      </Modal>
    </Show>
  ), host);
  return { closeOuter, closeInner, outer, inner };
}

describe("Modal", () => {
  it("Escape in the nested dialog closes only that dialog", () => {
    const m = mountNested();
    key(document.getElementById("inner-a")!, "Escape");
    expect(m.closeInner).toHaveBeenCalledOnce();
    expect(m.closeOuter).not.toHaveBeenCalled();
    expect(m.outer()).toBe(true);
    // Now the outer one is on top.
    key(document.getElementById("outer-btn")!, "Escape");
    expect(m.closeOuter).toHaveBeenCalledOnce();
  });

  it("a backdrop press closes only the top dialog", () => {
    const m = mountNested();
    down(document.querySelector(".baffle-overlay")!);
    expect(m.closeInner).toHaveBeenCalledOnce();
    expect(m.closeOuter).not.toHaveBeenCalled();
  });

  it("a press inside the dialog does not close it", () => {
    const m = mountNested();
    down(document.getElementById("inner-a")!);
    expect(m.closeInner).not.toHaveBeenCalled();
  });

  it("anyModalOpen follows the stack", () => {
    const m = mountNested();
    expect(anyModalOpen()).toBe(true);
    key(document.getElementById("inner-a")!, "Escape");
    key(document.getElementById("outer-btn")!, "Escape");
    expect(m.outer()).toBe(false);
    expect(anyModalOpen()).toBe(false);
  });

  it("dialogs carry role, aria-modal and their title", () => {
    mountNested();
    const dlg = document.querySelector(".baffle-dialog")!;
    expect(dlg.getAttribute("role")).toBe("dialog");
    expect(dlg.getAttribute("aria-modal")).toBe("true");
    expect(dlg.getAttribute("aria-labelledby")).toBe("t-inner");
  });

  it("Tab stays inside the top dialog", () => {
    mountNested();
    const b = document.getElementById("inner-b")!;
    b.focus();
    key(b, "Tab");
    expect(document.activeElement?.id).toBe("inner-a");
    key(document.getElementById("inner-a")!, "Tab", { shiftKey: true });
    expect(document.activeElement?.id).toBe("inner-b");
  });

  it("focus returns to the opener when the dialog closes", () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();
    const [open, setOpen] = createSignal(true);
    const host = document.createElement("div");
    document.body.appendChild(host);
    dispose = render(() => (
      <Show when={open()}>
        <Modal overlayClass="pn-overlay" dialogClass="pn-dialog" titleId="x" onEscape={() => setOpen(false)}>
          <button id="in">in</button>
        </Modal>
      </Show>
    ), host);
    document.getElementById("in")!.focus();
    key(document.getElementById("in")!, "Escape");
    expect(document.activeElement).toBe(opener);
  });
});
