// b141.105 (audit 2026-10-23 §1.6, §5.26): no dead narrowband controls; an
// emptied or out-of-range field shows what Apply saves.
import { describe, it, expect, vi, afterEach } from "vitest";
import { render } from "solid-js/web";

const store = vi.hoisted(() => ({ iter: 3, boost: 24, floor: -150, weight: true }));
vi.mock("../../stores/bands", () => ({
  firIterations: () => store.iter, setFirIterations: (v: number) => { store.iter = v; },
  firFreqWeighting: () => store.weight, setFirFreqWeighting: (v: boolean) => { store.weight = v; },
  firMaxBoost: () => store.boost, setFirMaxBoost: (v: number) => { store.boost = v; },
  firNoiseFloor: () => store.floor, setFirNoiseFloor: (v: number) => { store.floor = v; },
}));

import FirSettingsDialog, { openFirSettings, closeFirSettings } from "../FirSettingsDialog";

let dispose: (() => void) | undefined;
afterEach(() => { closeFirSettings(); dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

function mount() {
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => <FirSettingsDialog />, host);
  openFirSettings();
}
const type = (sel: string, v: string) => {
  const el = document.querySelector(sel) as HTMLInputElement;
  el.focus(); el.value = v;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.blur();
  return el;
};
const apply = () => (Array.from(document.querySelectorAll("button")).find((b) => b.textContent === "Применить") as HTMLButtonElement).click();

describe("FirSettingsDialog", () => {
  it("has no narrowband-limit controls", () => {
    mount();
    expect(document.body.textContent).not.toContain("узкопол");
    expect(document.querySelector("#fir-nb-limit")).toBeNull();
  });

  it("an emptied field shows and saves the previous value", () => {
    mount();
    const el = type("#fir-max-boost", "");
    expect(el.value).toBe("24");
    apply();
    expect(store.boost).toBe(24);
  });

  it("an out-of-range value is clamped in the field and in the store", () => {
    mount();
    const el = type("#fir-noise-floor", "-10");
    expect(el.value).toBe("-40");
    apply();
    expect(store.floor).toBe(-40);
  });
});
