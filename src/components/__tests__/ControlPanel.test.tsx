// b141.105 (audit 2026-10-23 §5.5, 5.6, 5.8, 5.9, 5.15): the right panel on
// the real store.
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render } from "solid-js/web";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => { throw new Error("no backend"); }) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => true), open: vi.fn(), save: vi.fn() }));

import ControlPanel from "../ControlPanel";
import {
  appState, addBand, setActiveBand, setBandTilt, setBandMeasurement, addExclusionZone,
  selectedPeqIdx, setSelectedPeqIdx, setExportHybridPhase, setBandHighPass,
} from "../../stores/bands";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

const meas = () => ({
  name: "m", source_path: null, sample_rate: 48000, freq: [20, 200, 2000, 20000],
  magnitude: [80, 81, 82, 83], phase: [0, -10, -20, -30],
  metadata: { date: null, mic: null, notes: null, smoothing: null },
}) as any;

let a = "", b = "";
beforeEach(() => {
  while (appState.bands.length < 2) addBand();
  a = appState.bands[0].id; b = appState.bands[1].id;
  setActiveBand(a);
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => <ControlPanel rightPanel />, host);
});

const commitInput = (el: HTMLInputElement, v: string) => {
  el.value = v;
  el.dispatchEvent(new Event("change", { bubbles: true }));
};

describe("ControlPanel", () => {
  it("«Пресет» shows the active band's curve and follows the band", () => {
    const sel = () => document.getElementById("cp-preset") as HTMLSelectElement;
    setBandTilt(a, -0.4);
    expect(sel().value).toBe("harman");
    setBandTilt(b, 0);
    setActiveBand(b);
    expect(sel().value).toBe("flat");
    setBandTilt(b, -0.7);
    expect(sel().value).toBe("custom");
  });

  it("the selected PEQ filter is cleared when the band changes", () => {
    setSelectedPeqIdx(2);
    setActiveBand(b);
    expect(selectedPeqIdx()).toBeNull();
  });

  it("exclusion zone: below 20 Hz is clamped and shown, От ≥ До is rejected", () => {
    setBandMeasurement(a, meas());
    addExclusionZone(a, { startHz: 100, endHz: 200 });
    const zi = appState.bands[0].exclusionZones.length - 1;
    const from = () => document.querySelector(`input[aria-label="Зона ${zi + 1}: от, Гц"]`) as HTMLInputElement;
    commitInput(from(), "5");
    expect(appState.bands[0].exclusionZones[zi].startHz).toBe(20);
    expect(from().value).toBe("20");
    commitInput(from(), "500");
    expect(appState.bands[0].exclusionZones[zi].startHz).toBe(20);
    expect(from().value).toBe("20");
  });

  it("«Вручную» row clamps gain to the Standard limits", () => {
    setExportHybridPhase(false);
    (document.querySelector('button[aria-label="Добавить PEQ-фильтр вручную"]') as HTMLButtonElement).click();
    const gain = document.querySelector(".peq-manual-section input[aria-label=\"Усиление, дБ\"]") as HTMLInputElement;
    commitInput(gain, "30");
    expect(appState.bands[0].peqBands[0].gain_db).toBe(6);
    expect(gain.value).toBe("6.0");
  });

  it("Lin-φ toggles from the keyboard", () => {
    setBandHighPass(a, { filter_type: "Butterworth", order: 2, freq_hz: 80, shape: null, linear_phase: false, q: null, subsonic_protect: null });
    const box = document.querySelector('[role="checkbox"][aria-label="ФВЧ: линейная фаза"]') as HTMLElement;
    expect(box.getAttribute("tabindex")).toBe("0");
    box.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
    expect(appState.bands[0].target.high_pass!.linear_phase).toBe(true);
  });
});
