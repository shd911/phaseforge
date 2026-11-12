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
afterEach(() => {
  closeFirSettings(); dispose?.(); dispose = undefined; document.body.innerHTML = "";
  Object.assign(store, { iter: 3, boost: 24, floor: -150, weight: true });
});

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

  // b141.148: the floor in bits is the same value, 1 bit = 6.02 dB.
  it("the noise floor reads in bits and a typed word length sets the dB value", () => {
    mount();
    const bits = document.querySelector("#fir-noise-floor-bits") as HTMLInputElement;
    const db = document.querySelector("#fir-noise-floor") as HTMLInputElement;
    expect(bits.value).toBe("24.9");
    type("#fir-noise-floor-bits", "24");
    expect(db.value).toBe("-144.5");
    type("#fir-noise-floor", "-96.3");
    expect(bits.value).toBe("16");
    apply();
    expect(store.floor).toBe(-96.3);
  });

  it("a bits field left untouched does not move the dB value", () => {
    mount();
    const bits = document.querySelector("#fir-noise-floor-bits") as HTMLInputElement;
    bits.focus(); bits.blur();
    apply();
    expect(store.floor).toBe(-150);
  });

  it("a word length outside the range is clamped to the dB limits", () => {
    mount();
    const bits = type("#fir-noise-floor-bits", "2");
    expect(bits.value).toBe("6.6");
    apply();
    expect(store.floor).toBe(-40);
  });
});
