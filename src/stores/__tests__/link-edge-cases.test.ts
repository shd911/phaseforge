/**
 * b140.8.2: linkedToNext should be preserved across removeBand.
 *
 * Scenario: 5 bands, delete band 2 — bands 1 and 3 (positionally
 * adjacent after delete) must remain linked.
 */
import { describe, it, expect } from "vitest";
import { createRoot } from "solid-js";
import {
  appState,
  addBand,
  removeBand,
  resetAppState,
  setBandHighPass,
  setBandLowPass,
  toggleBandLinked,
} from "../bands";
import { complementPartnerOk, type FilterConfig } from "../../lib/types";
import type { AppState, BandState } from "../bands";

function freshFiveBands(): AppState {
  // Build a 5-band project state directly so each test starts clean.
  const bands: BandState[] = [];
  for (let i = 1; i <= 5; i++) {
    bands.push({
      id: `band-${i}`,
      name: `Band ${i}`,
      measurement: null,
      measurementFile: null,
      settings: null,
      target: {
        reference_level_db: 0,
        tilt_db_per_octave: 0,
        tilt_ref_freq: 1000,
        high_pass: null,
        low_pass: null,
        low_shelf: null,
        high_shelf: null,
      },
      targetEnabled: true,
      inverted: false,
      linkedToNext: i < 5,
      peqBands: [],
      peqOptimizedTarget: null,
      exclusionZones: [],
      firResult: null,
      crossNormDb: 0,
      color: "#ff0000",
      alignmentDelay: 0,
    });
  }
  return {
    bands,
    activeBandId: "band-1",
    showPhase: true,
    showMag: true,
    showTarget: true,
    nextBandNum: 6,
  } as AppState;
}

describe("linkedToNext across removeBand (b140.8.2)", () => {
  it("deleting middle band preserves prev band's link to new neighbour", () => {
    createRoot((dispose) => {
      resetAppState(freshFiveBands());
      // Initial: 5 bands, all linked (except last).
      expect(appState.bands.length).toBe(5);
      expect(appState.bands[0].linkedToNext).toBe(true);
      expect(appState.bands[1].linkedToNext).toBe(true);
      expect(appState.bands[2].linkedToNext).toBe(true);
      expect(appState.bands[3].linkedToNext).toBe(true);
      expect(appState.bands[4].linkedToNext).toBe(false);

      // Delete band 2 (idx=1).
      removeBand("band-2");

      // After: 4 bands. Band 1 must still be linked (was linked to deleted
      // band 2; should now be linked to its new neighbour, what was band 3).
      expect(appState.bands.length).toBe(4);
      expect(appState.bands[0].id).toBe("band-1");
      expect(appState.bands[1].id).toBe("band-3");
      expect(appState.bands[0].linkedToNext).toBe(true);
      expect(appState.bands[1].linkedToNext).toBe(true);
      expect(appState.bands[2].linkedToNext).toBe(true);
      expect(appState.bands[3].linkedToNext).toBe(false);
      dispose();
    });
  });

  it("deleting last band clears prev band's downstream link", () => {
    createRoot((dispose) => {
      resetAppState(freshFiveBands());
      removeBand("band-5");
      expect(appState.bands.length).toBe(4);
      // New last band (was band-4) has no downstream neighbour — link off.
      expect(appState.bands[3].id).toBe("band-4");
      expect(appState.bands[3].linkedToNext).toBe(false);
      // Earlier links untouched.
      expect(appState.bands[0].linkedToNext).toBe(true);
      expect(appState.bands[1].linkedToNext).toBe(true);
      expect(appState.bands[2].linkedToNext).toBe(true);
      dispose();
    });
  });

  it("deleting first band keeps remaining links intact", () => {
    createRoot((dispose) => {
      resetAppState(freshFiveBands());
      removeBand("band-1");
      expect(appState.bands.length).toBe(4);
      expect(appState.bands[0].id).toBe("band-2");
      expect(appState.bands[0].linkedToNext).toBe(true);
      expect(appState.bands[1].linkedToNext).toBe(true);
      expect(appState.bands[2].linkedToNext).toBe(true);
      expect(appState.bands[3].linkedToNext).toBe(false);
      dispose();
    });
  });
});

describe("«1 − H» on a link (b141.109–110)", () => {
  const custom = (complementary: boolean, q = 0.707): FilterConfig => ({
    filter_type: "Custom", order: 4, freq_hz: 500, shape: null, linear_phase: true, q,
    subsonic_protect: null, complementary,
  });

  it("only one side of a link subtracts", () => {
    createRoot((dispose) => {
      resetAppState(freshFiveBands());
      setBandLowPass("band-1", custom(false));
      setBandHighPass("band-2", custom(true));
      expect(appState.bands[1].target.high_pass!.complementary).toBe(true);
      expect(appState.bands[0].target.low_pass!.complementary).toBe(false);
      // Moving «1 − H» to the LP side clears the HP's.
      setBandLowPass("band-1", custom(true));
      expect(appState.bands[0].target.low_pass!.complementary).toBe(true);
      expect(appState.bands[1].target.high_pass!.complementary).toBe(false);
      dispose();
    });
  });

  it("editing the shared crossover keeps the «1 − H» side", () => {
    createRoot((dispose) => {
      resetAppState(freshFiveBands());
      setBandLowPass("band-1", custom(false));
      setBandHighPass("band-2", custom(true));
      setBandLowPass("band-1", { ...custom(false), freq_hz: 600, q: 0.5, order: 2 });
      const hp = appState.bands[1].target.high_pass!;
      expect([hp.freq_hz, hp.q, hp.order, hp.complementary]).toEqual([600, 0.5, 2, true]);
      dispose();
    });
  });

  it("falls back to the plain mirror when the partner stops being a linear Q ≤ 0.707", () => {
    for (const partner of [
      { ...custom(false), filter_type: "Butterworth" as const, q: null },
      custom(false, 1.0),
      { ...custom(false), linear_phase: false },
    ]) {
      createRoot((dispose) => {
        resetAppState(freshFiveBands());
        setBandLowPass("band-1", custom(false));
        setBandHighPass("band-2", custom(true));
        setBandLowPass("band-1", partner);
        expect(appState.bands[1].target.high_pass!.complementary).toBe(false);
        expect(appState.bands[1].target.high_pass!.filter_type).toBe(partner.filter_type);
        dispose();
      });
    }
  });

  it("is offered only against a «Произвольный Q» up to 0.707", () => {
    expect(complementPartnerOk(custom(false))).toBe(true);
    expect(complementPartnerOk(custom(false, 0.72))).toBe(false);
    expect(complementPartnerOk({ ...custom(false), filter_type: "Bessel" })).toBe(false);
    expect(complementPartnerOk(null)).toBe(false);
  });
});
