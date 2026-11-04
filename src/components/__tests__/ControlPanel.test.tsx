// b141.105 (audit 2026-10-23 §5.5, 5.6, 5.8, 5.9, 5.15): the right panel on
// the real store.
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render } from "solid-js/web";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => { throw new Error("no backend"); }) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => true), open: vi.fn(), save: vi.fn() }));

import ControlPanel from "../ControlPanel";
import { invoke } from "@tauri-apps/api/core";
import {
  appState, addBand, setActiveBand, setBandTilt, setBandMeasurement, addExclusionZone,
  selectedPeqIdx, setSelectedPeqIdx, setExportHybridPhase, setBandHighPass, setBandLowPass,
  toggleBandLinked, setNestedCrossovers, setBandPeqBands, setExportSampleRate, setExportTaps,
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

  it("ring: longer than the ear reddens the number, twice as long the row (b141.138)", () => {
    setBandMeasurement(a, meas()); // the PEQ table shows with a measurement
    setExportSampleRate(176400); setExportTaps(131072);
    const pk = (freq_hz: number, gain_db: number, q: number) => ({ freq_hz, gain_db, q, filter_type: "Peaking" as const, enabled: true });
    setBandPeqBands(a, [pk(10000, 6, 2.5), pk(1000, 6, 3), pk(100, 6, 3)]);
    const rows = [...document.querySelectorAll(".peq-table:not(.peq-excl-table) tbody tr")];
    const cell = (r: Element) => r.querySelector("td.peq-ring")!;
    expect(cell(rows[0]).classList.contains("peq-ring-bad")).toBe(false);
    expect(cell(rows[0]).textContent).toBe("0.8");
    expect(cell(rows[1]).classList.contains("peq-ring-bad")).toBe(true);
    expect(rows[1].classList.contains("peq-row-ring")).toBe(false);
    expect(rows[2].classList.contains("peq-row-ring")).toBe(true);
  });

  it("ring judged on the result with the measurement: a cut that shortens stays plain (b141.139)", async () => {
    const nd = (ear: number, without: number, with_: number) =>
      ({ freqHz: 0, earMs: [ear, ear], withoutMs: [without, without], withMs: [with_, with_] });
    vi.mocked(invoke).mockImplementation(async (cmd: string) => {
      if (cmd !== "compute_peq_net_decay") throw new Error("no backend");
      return [nd(10, 10, 11), nd(10, 10, 15), null, nd(20, 60, 40), nd(20, 20, 45)];
    });
    try {
      setExportSampleRate(176400); setExportTaps(131072);
      const pk = (freq_hz: number, gain_db: number, q: number) => ({ freq_hz, gain_db, q, filter_type: "Peaking" as const, enabled: true });
      setBandMeasurement(a, meas());
      setBandPeqBands(a, [pk(1000, 3, 2), pk(500, 3, 2), pk(700, 3, 2), pk(40, -6, 4), pk(60, 4, 2)]);
      await new Promise((r) => setTimeout(r, 0));
      const rows = [...document.querySelectorAll(".peq-table:not(.peq-excl-table) tbody tr")];
      const red = (r: Element) => r.querySelector("td.peq-ring")!.classList.contains("peq-ring-bad");
      const redRow = (r: Element) => r.classList.contains("peq-row-ring");
      expect(rows.map(red)).toEqual([false, true, false, false, true]);
      expect(rows.map(redRow)).toEqual([false, false, false, false, true]);
      // 40 Hz −6 dB Q 4 rings ×7 the ear alone (a red row by b141.138), but it
      // shortens the room's decay.
      expect(rows[3].querySelector("td.peq-ring")!.getAttribute("title")).toContain("Укорачивает на 100 %");
      expect(rows[2].querySelector("td.peq-ring")!.getAttribute("title")).toContain("у края замера");
    } finally {
      vi.mocked(invoke).mockImplementation(async () => { throw new Error("no backend"); });
    }
  });

  it("Lin-φ toggles from the keyboard", () => {
    setBandHighPass(a, { filter_type: "Butterworth", order: 2, freq_hz: 80, shape: null, linear_phase: false, q: null, subsonic_protect: null });
    const box = document.querySelector('[role="checkbox"][aria-label="ФВЧ: линейная фаза"]') as HTMLElement;
    expect(box.getAttribute("tabindex")).toBe("0");
    box.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
    expect(appState.bands[0].target.high_pass!.linear_phase).toBe(true);
  });

  it("«Вложенные» left on with nothing to nest says it does nothing", () => {
    while (appState.bands.length < 3) addBand();
    setActiveBand(a); // addBand activates the new band
    const section = (filter_type: "LinkwitzRiley" | "Butterworth", order: number, freq_hz: number) =>
      ({ filter_type, order, freq_hz, shape: null, linear_phase: true, q: null, subsonic_protect: null });
    const crossovers = (filter_type: "LinkwitzRiley" | "Butterworth", order: number) => {
      const [x, y, z] = appState.bands.map((band) => band.id);
      setBandLowPass(x, section(filter_type, order, 220));
      setBandHighPass(y, section(filter_type, order, 220));
      setBandLowPass(y, section(filter_type, order, 800));
      setBandHighPass(z, section(filter_type, order, 800));
    };
    appState.bands.slice(0, 2).forEach((band) => { if (!band.linkedToNext) toggleBandLinked(band.id); });
    const warning = () => [...document.querySelectorAll(".fb-ring .warn")]
      .some((el) => el.textContent!.startsWith("Не действует."));

    crossovers("LinkwitzRiley", 1);
    setNestedCrossovers(true);
    expect(warning()).toBe(false);
    expect(document.body.textContent).toContain("+ ФНЧ 800 Гц");

    crossovers("Butterworth", 1); // 6 dB/oct: no such slope on Linkwitz-Riley
    expect(warning()).toBe(true);
    expect(document.body.textContent).not.toContain("+ ФНЧ 800 Гц");

    setNestedCrossovers(false);
    expect(warning()).toBe(false);
  });
});
