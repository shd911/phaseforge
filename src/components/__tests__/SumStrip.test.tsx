// b141.146: the Σ strip under the plot — band columns and crossover cards on
// the real store.
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render } from "solid-js/web";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => { throw new Error("no backend"); }) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => true), open: vi.fn(), save: vi.fn() }));

import SumStrip, { type SumStripProps } from "../SumStrip";
import {
  appState, addBand, removeBand, setBandHighPass, setBandLowPass, toggleBandLinked, toggleBandInverted,
} from "../../stores/bands";
import type { FilterConfig } from "../../lib/types";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

const lr = (freq: number): FilterConfig =>
  ({ filter_type: "LinkwitzRiley", order: 2, freq_hz: freq, shape: null, linear_phase: false, q: null });

const props = (over: Partial<SumStripProps> = {}): SumStripProps => ({
  visEnabled: () => false,
  pairMode: () => false,
  cell: () => undefined,
  cellPair: () => ({ ir: undefined, step: undefined }),
  toggleCell: () => {},
  toggleColumn: () => {},
  toggleCategory: () => {},
  convDelays: () => ({}),
  xoX: () => ({}),
  hotXo: () => null,
  onHoverXo: () => {},
  ...over,
});

function mount(p: SumStripProps = props()) {
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => <SumStrip {...p} />, host);
}

/** Three bands, crossovers at 200 and 2000 Hz, both linked. */
function threeWay() {
  while (appState.bands.length > 3) removeBand(appState.bands[appState.bands.length - 1].id);
  while (appState.bands.length < 3) addBand();
  const [a, b, c] = appState.bands.map((x) => x.id);
  for (const id of [a, b, c]) { setBandHighPass(id, null); setBandLowPass(id, null); }
  for (const id of [a, b]) if (appState.bands.find((x) => x.id === id)!.linkedToNext) toggleBandLinked(id);
  setBandLowPass(a, lr(200)); setBandHighPass(b, lr(200));
  setBandLowPass(b, lr(2000)); setBandHighPass(c, lr(2000));
  toggleBandLinked(a); toggleBandLinked(b);
  return { a, b, c };
}

const merged = () => document.querySelectorAll("[data-xo-band]");
const freqOf = (card: Element) => (card.querySelector("input.num-field") as HTMLInputElement | null)?.value;

beforeEach(() => { threeWay(); });

describe("SumStrip", () => {
  it("a column per band and a crossover card between each pair", () => {
    mount();
    expect(document.querySelectorAll(".ss-band").length).toBe(3);
    expect(merged().length).toBe(2);
    expect(document.querySelectorAll(".ss-edge").length).toBe(2);
  });

  it("the cards follow the number of bands", () => {
    mount();
    addBand();
    expect(document.querySelectorAll(".ss-band").length).toBe(4);
    // the new band is not linked to its neighbour: its boundary is a split card
    expect(document.querySelectorAll(".ss-xo:not(.ss-edge)").length).toBe(3);
    removeBand(appState.bands[3].id);
    expect(document.querySelectorAll(".ss-band").length).toBe(3);
    expect(document.querySelectorAll(".ss-xo:not(.ss-edge)").length).toBe(2);
  });

  it("a linked crossover is one editor; unlinking splits it in two", () => {
    const { a } = threeWay();
    mount();
    expect(document.querySelectorAll(".ss-xo-split").length).toBe(0);
    toggleBandLinked(a);
    expect(merged().length).toBe(1);
    expect(document.querySelectorAll(".ss-xo-split .filter-block").length).toBe(2);
  });

  it("a card edits the same store the band tabs read — both sides of the link", () => {
    const { a, b } = threeWay();
    mount();
    const card = merged()[0];
    expect(card.getAttribute("data-xo-band")).toBe(a);
    const sel = card.querySelector("select.fb-select") as HTMLSelectElement;
    sel.value = "Butterworth";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    expect(appState.bands.find((x) => x.id === a)!.target.low_pass!.filter_type).toBe("Butterworth");
    expect(appState.bands.find((x) => x.id === b)!.target.high_pass!.filter_type).toBe("Butterworth");
  });

  it("a store change shows in the card", () => {
    const { a } = threeWay();
    mount();
    setBandLowPass(a, lr(250));
    expect(freqOf(merged()[0])).toContain("250");
  });

  it("polarity button writes the band's own flag", () => {
    const { b } = threeWay();
    mount();
    const col = document.querySelectorAll(".ss-band")[1];
    const btn = col.querySelector(".invert-toggle") as HTMLButtonElement;
    const before = appState.bands.find((x) => x.id === b)!.inverted;
    btn.click();
    expect(appState.bands.find((x) => x.id === b)!.inverted).toBe(!before);
    expect(btn.textContent).toBe(!before ? "INV" : "NOR");
    toggleBandInverted(b);
  });

  it("hovering a crossover card reports its band; the plot's hover lights the card", () => {
    const { a } = threeWay();
    const seen: (string | null)[] = [];
    mount(props({ hotXo: () => a, onHoverXo: (id) => seen.push(id) }));
    const card = merged()[0] as HTMLElement;
    expect(card.classList.contains("ss-xo-hot")).toBe(true);
    card.dispatchEvent(new MouseEvent("mouseenter"));
    card.dispatchEvent(new MouseEvent("mouseleave"));
    // Solid delegates mouseenter/leave as non-bubbling native listeners.
    expect(seen).toEqual([a, null]);
  });

  it("curve cells call back with the plot's legend entry", () => {
    const entry = { visible: true, color: "#fff", dash: false };
    const toggled: unknown[] = [];
    mount(props({
      visEnabled: () => true,
      cell: (col, cat) => (col === "Σ" && cat === "corrected" ? entry : undefined),
      toggleCell: (e) => toggled.push(e),
    }));
    const btns = document.querySelectorAll(".ss-sigma .legend-item");
    expect(btns.length).toBe(1);
    (btns[0] as HTMLButtonElement).click();
    expect(toggled).toEqual([entry]);
  });
});
