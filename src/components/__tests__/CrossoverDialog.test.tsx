// b141.105 (audit 2026-10-23 §5.7, §5.21): the Crossover window honours the
// Lin-φ checkbox for Gaussian and clamps M / Q to the filter block's limits.
import { describe, it, expect, vi, afterEach } from "vitest";
import { render } from "solid-js/web";

const setBandLowPass = vi.fn();
vi.mock("../../stores/bands", () => ({
  appState: { bands: [] },
  setBandLowPass: (...a: unknown[]) => setBandLowPass(...a),
}));

import CrossoverDialog, { openCrossoverDialog, closeCrossoverDialog } from "../CrossoverDialog";
import { CUSTOM_Q_MAX, GAUSS_M_MIN } from "../../lib/slope";

let dispose: (() => void) | undefined;
afterEach(() => {
  closeCrossoverDialog();
  dispose?.(); dispose = undefined;
  document.body.innerHTML = "";
  setBandLowPass.mockReset();
});

function mount() {
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => <CrossoverDialog />, host);
}

const base = {
  bandIndex: 0, bandId: "b1", bandName: "НЧ", nextBandName: "ВЧ",
  freq: 300, order: 4, shape: 1.5 as number | null, q: null as number | null,
};
const apply = () => (Array.from(document.querySelectorAll("button")).find((b) => b.textContent === "Применить") as HTMLButtonElement).click();
const type = (sel: string, v: string) => {
  const el = document.querySelector(sel) as HTMLInputElement;
  el.focus();
  el.value = v;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.blur();
  return el;
};

describe("CrossoverDialog", () => {
  it("a min-phase Gaussian stays min-phase", () => {
    mount();
    openCrossoverDialog({ ...base, filterType: "Gaussian", linearPhase: false });
    apply();
    expect(setBandLowPass).toHaveBeenCalledOnce();
    expect(setBandLowPass.mock.calls[0][1].linear_phase).toBe(false);
  });

  it("M is clamped and the field shows the clamped value", () => {
    mount();
    openCrossoverDialog({ ...base, filterType: "Gaussian", linearPhase: true });
    const el = type("#xo-m", "0");
    expect(el.value).toBe(String(GAUSS_M_MIN));
    apply();
    expect(setBandLowPass.mock.calls[0][1].shape).toBe(GAUSS_M_MIN);
  });

  it("Q is clamped to the filter block's range", () => {
    mount();
    openCrossoverDialog({ ...base, filterType: "Custom", order: 2, linearPhase: false, q: 0.707 });
    type("#xo-q", "50");
    apply();
    expect(setBandLowPass.mock.calls[0][1].q).toBe(CUSTOM_Q_MAX);
  });

  it("an emptied M field shows the kept value again", () => {
    mount();
    openCrossoverDialog({ ...base, filterType: "Gaussian", linearPhase: true });
    const el = type("#xo-m", "");
    expect(el.value).toBe("1.5");
  });
});
