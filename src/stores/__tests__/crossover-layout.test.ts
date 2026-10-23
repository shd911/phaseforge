/** b141.103 (audit 2026-10-23 3.16, 3.17): add/remove keep crossovers the
 *  user set; moving bands rebuilds the edges the move broke. */
import { describe, it, expect } from "vitest";
import { appState, addBand, removeBand, moveBand, resetAppState, createBand, setBandLowPass } from "../bands";
import type { BandState } from "../bands";

function fresh(n: number) {
  const bands: BandState[] = [];
  for (let i = 1; i <= n; i++) bands.push(createBand(i));
  resetAppState({ bands, activeBandId: "__sum__", showPhase: true, showMag: true, showTarget: true, nextBandNum: n + 1 });
}

describe("add/remove band (3.16)", () => {
  it("keeps a crossover the user set on a band without measurement", () => {
    fresh(1);
    addBand();
    const first = appState.bands[0].id;
    setBandLowPass(first, { filter_type: "LinkwitzRiley", order: 4, freq_hz: 3333, shape: null, linear_phase: false, q: null });
    expect(appState.bands[1].target.high_pass!.freq_hz).toBe(3333); // linked
    addBand();
    expect(appState.bands[0].target.low_pass!.freq_hz).toBe(3333);
    // The pristine neighbour follows the user's edge, the new band is split.
    expect(appState.bands[1].target.high_pass!.freq_hz).toBe(3333);
    expect(appState.bands[1].target.low_pass!.freq_hz).toBeGreaterThan(3333);
    removeBand(appState.bands[2].id);
    expect(appState.bands[0].target.low_pass!.freq_hz).toBe(3333);
  });

  it("still re-splits bands on the default split", () => {
    fresh(1);
    addBand();
    const x2 = appState.bands[0].target.low_pass!.freq_hz;
    addBand();
    expect(appState.bands[0].target.low_pass!.freq_hz).not.toBe(x2);
    expect(appState.bands[0].target.low_pass!.freq_hz).toBe(appState.bands[1].target.high_pass!.freq_hz);
  });
});

describe("moveBand (3.17)", () => {
  it("no two full-range bands after moving the bass to the end", () => {
    fresh(1);
    addBand();
    addBand();
    moveBand(0, 2);
    const [a, b, c] = appState.bands;
    expect(a.target.high_pass).toBeNull();
    expect(a.target.low_pass).not.toBeNull();
    expect(b.target.low_pass).not.toBeNull();
    expect(c.target.high_pass).not.toBeNull();
    expect(c.target.low_pass).toBeNull();
    expect(b.target.low_pass!.freq_hz).toBe(c.target.high_pass!.freq_hz);
    expect(b.target.high_pass!.freq_hz).toBeLessThan(b.target.low_pass!.freq_hz);
    expect(appState.bands[1].linkedToNext).toBe(true);
  });
});
