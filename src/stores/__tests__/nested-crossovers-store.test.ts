// b141.134: `nested` follows the crossovers and the flag, without marking the
// project dirty or entering undo history.
import { beforeEach, describe, expect, it } from "vitest";
import {
  appState, addBand, setBandHighPass, setBandLowPass, toggleBandLinked, setNestedCrossovers,
  nestedCrossovers, removeBand, setBandTilt, bandsVersion,
} from "../bands";
import { peqStaleReason, captureOptimizedTarget } from "../peq-optimize";
import { undo } from "../history";
import { ringZone } from "../../lib/export-metrics";
import { cloneFilterConfig, type FilterConfig } from "../../lib/types";

const lr = (freq_hz: number, over: Partial<FilterConfig> = {}): FilterConfig => ({
  filter_type: "LinkwitzRiley", order: 1, freq_hz, shape: null, linear_phase: true, q: null, ...over,
});

function fourBands(xo = [100, 500, 2000]) {
  for (const b of [...appState.bands]) removeBand(b.id);
  for (let i = 0; i <= xo.length; i++) addBand();
  const ids = appState.bands.map((b) => b.id);
  ids.forEach((id, i) => {
    setBandHighPass(id, i > 0 ? lr(xo[i - 1]) : null);
    setBandLowPass(id, i < xo.length ? lr(xo[i]) : null);
  });
  appState.bands.forEach((b, i) => { if (i < xo.length && !b.linkedToNext) toggleBandLinked(b.id); });
  ids.forEach((id, i) => {
    setBandHighPass(id, i > 0 ? lr(xo[i - 1]) : null);
    setBandLowPass(id, i < xo.length ? lr(xo[i]) : null);
  });
  return ids;
}

const lpNested = (i: number) => (appState.bands[i].target.low_pass?.nested ?? []).map((n) => n.freq_hz);
const hpNested = (i: number) => (appState.bands[i].target.high_pass?.nested ?? []).map((n) => n.freq_hz);

describe("nested crossovers in the store", () => {
  beforeEach(() => setNestedCrossovers(false));

  it("off leaves no key on the target", () => {
    fourBands();
    expect(nestedCrossovers()).toBe(false);
    expect("nested" in appState.bands[0].target.low_pass!).toBe(false);
  });

  it("the flag adds and removes the lists", () => {
    fourBands();
    setNestedCrossovers(true);
    expect(lpNested(0)).toEqual([500]);
    expect(hpNested(3)).toEqual([500]);
    expect(lpNested(1)).toEqual([]);
    setNestedCrossovers(false);
    expect(lpNested(0)).toEqual([]);
    expect(hpNested(3)).toEqual([]);
    expect("nested" in appState.bands[0].target.low_pass!).toBe(false);
  });

  it("moving a crossover moves it in the other bands", () => {
    const ids = fourBands();
    setNestedCrossovers(true);
    setBandLowPass(ids[1], lr(630));
    expect(lpNested(0)).toEqual([630]);
    expect(hpNested(3)).toEqual([630]);
  });

  it("a crossover leaving the conditions removes the tree it broke", () => {
    const ids = fourBands();
    setNestedCrossovers(true);
    setBandLowPass(ids[1], lr(500, { linear_phase: false }));
    expect(lpNested(0)).toEqual([]);
    expect(hpNested(3)).toEqual([]);
  });

  // b141.136: the plot redraws on bandsVersion only. The toggle changed the
  // lists and the file, the curves stayed until some other edit.
  it("the toggle redraws the plot when the lists change, and only then", () => {
    fourBands();
    let v = bandsVersion();
    setNestedCrossovers(true);
    expect(lpNested(0)).toEqual([500]);
    expect(bandsVersion()).toBe(v + 1);
    v = bandsVersion();
    setNestedCrossovers(false);
    expect(lpNested(0)).toEqual([]);
    expect(bandsVersion()).toBe(v + 1);

    setBandLowPass(appState.bands[1].id, lr(500, { linear_phase: false })); // no chain is left
    v = bandsVersion();
    setNestedCrossovers(true);
    expect(bandsVersion()).toBe(v);
  });

  it("the toggle makes a fitted PEQ stale", () => {
    fourBands();
    const b = appState.bands[0];
    const snap = captureOptimizedTarget(b);
    setNestedCrossovers(true);
    const stale = peqStaleReason({ ...appState.bands[0], peqBands: [{ freq_hz: 100, gain_db: 1, q: 1, enabled: true, filter_type: "Peaking" }], peqOptimizedTarget: snap } as any);
    expect(stale).not.toBeNull();
  });

  it("an undo to before the flag re-derives the lists", () => {
    const ids = fourBands();
    setBandTilt(ids[0], -0.5); // its history entry has no lists
    setNestedCrossovers(true);
    undo();
    expect(appState.bands[0].target.tilt_db_per_octave).toBe(0);
    expect(lpNested(0)).toEqual([500]);
    expect(hpNested(3)).toEqual([500]);
  });

  it("the ring zone counts a nested HP below the band's own", () => {
    fourBands();
    const before = ringZone(appState.bands[3] as any);
    expect(before.hz).toBe(2000);
    setNestedCrossovers(true);
    expect(ringZone(appState.bands[3] as any).hz).toBe(500);
  });

  it("cloneFilterConfig keeps the lists", () => {
    fourBands();
    setNestedCrossovers(true);
    const c = cloneFilterConfig(appState.bands[0].target.low_pass!);
    expect(c.nested?.map((n) => n.freq_hz)).toEqual([500]);
    expect(c.nested![0]).not.toBe(appState.bands[0].target.low_pass!.nested![0]);
  });
});
