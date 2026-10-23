// b141.105 (audit 2026-10-23 §5.22): after deleting a middle band, dragging a
// tab drops it where the pointer is (A,B,C,D → delete B → A onto C = C,A,D).
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render } from "solid-js/web";

vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => true), open: vi.fn(), save: vi.fn() }));

import BandTabs from "../BandTabs";
import { appState, addBand, removeBand, renameBand } from "../../stores/bands";

let dispose: (() => void) | undefined;
const origRect = HTMLElement.prototype.getBoundingClientRect;

beforeEach(() => {
  // Tabs laid out 100 px apart in DOM order.
  HTMLElement.prototype.getBoundingClientRect = function (this: HTMLElement) {
    const tabs = Array.from(document.querySelectorAll(".band-tab[data-band-id]"));
    const i = tabs.indexOf(this);
    const left = i < 0 ? -1000 : i * 100;
    return { left, right: left + 99, top: 0, bottom: 20, width: 99, height: 20, x: left, y: 0, toJSON() {} } as DOMRect;
  };
});
afterEach(() => {
  HTMLElement.prototype.getBoundingClientRect = origRect;
  dispose?.(); dispose = undefined;
  document.body.innerHTML = "";
});

const ptr = (target: EventTarget, type: string, clientX: number) =>
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX }));

describe("BandTabs drag after a delete", () => {
  it("drops the tab where the pointer is", () => {
    while (appState.bands.length < 4) addBand();
    while (appState.bands.length > 4) removeBand(appState.bands[appState.bands.length - 1].id);
    ["A", "B", "C", "D"].forEach((n, i) => renameBand(appState.bands[i].id, n));

    const host = document.createElement("div");
    document.body.appendChild(host);
    dispose = render(() => <BandTabs />, host);

    removeBand(appState.bands[1].id); // B
    expect(appState.bands.map((b) => b.name)).toEqual(["A", "C", "D"]);

    const tabA = document.querySelector(`.band-tab[data-band-id="${appState.bands[0].id}"]`)!;
    ptr(tabA, "pointerdown", 50);
    ptr(window, "pointermove", 150); // over C
    ptr(window, "pointerup", 150);
    expect(appState.bands.map((b) => b.name)).toEqual(["C", "A", "D"]);
  });

  it("the delete control is a real button outside any other button", () => {
    const close = document.querySelector(".band-tab-close");
    if (!close) {
      const host = document.createElement("div");
      document.body.appendChild(host);
      dispose = render(() => <BandTabs />, host);
    }
    const btn = document.querySelector(".band-tab-close")!;
    expect(btn.tagName).toBe("BUTTON");
    expect(btn.parentElement!.closest("button")).toBeNull();
  });
});
