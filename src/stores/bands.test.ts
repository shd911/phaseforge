/**
 * E2E store test: verify that changing HP linear_phase does NOT affect LP linear_phase.
 *
 * Bug hypothesis: toggling linear_phase on HP filter also changes LP's linear_phase.
 * This test directly calls store functions — no UI involved.
 */
import { describe, it, expect } from "vitest";
import { createRoot } from "solid-js";
import { unwrap } from "solid-js/store";
import {
  appState,
  addBand,
  setBandHighPass,
  setBandLowPass,
} from "./bands";
import type { FilterConfig } from "../lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeGaussianFilter(freq: number, linearPhase: boolean): FilterConfig {
  return {
    filter_type: "Gaussian",
    order: 4,
    freq_hz: freq,
    shape: 1.0,
    linear_phase: linearPhase,
    q: null,
  };
}

function makeButterworthFilter(freq: number, linearPhase: boolean): FilterConfig {
  return {
    filter_type: "Butterworth",
    order: 4,
    freq_hz: freq,
    shape: null,
    linear_phase: linearPhase,
    q: null,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("HP/LP linear_phase isolation", () => {
  it("changing HP linear_phase does NOT change LP linear_phase (Gaussian)", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      // 1. Set both HP and LP to Gaussian, linear_phase=true
      setBandHighPass(bandId, makeGaussianFilter(80, true));
      setBandLowPass(bandId, makeGaussianFilter(2000, true));

      // Verify initial state
      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(true);
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);

      // 2. Change ONLY HP to linear_phase=false
      const hp = appState.bands[0].target.high_pass!;
      setBandHighPass(bandId, {
        filter_type: hp.filter_type,
        order: hp.order,
        freq_hz: hp.freq_hz,
        shape: hp.shape,
        linear_phase: false,    // <-- only this changes
        q: hp.q,
      });

      // 3. Verify HP changed
      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(false);

      // 4. KEY ASSERTION: LP must still be true
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("changing LP linear_phase does NOT change HP linear_phase (Gaussian)", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      // Set both to linear_phase=true
      setBandHighPass(bandId, makeGaussianFilter(80, true));
      setBandLowPass(bandId, makeGaussianFilter(2000, true));

      // Change ONLY LP to linear_phase=false
      const lp = appState.bands[0].target.low_pass!;
      setBandLowPass(bandId, {
        filter_type: lp.filter_type,
        order: lp.order,
        freq_hz: lp.freq_hz,
        shape: lp.shape,
        linear_phase: false,
        q: lp.q,
      });

      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(false);
      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("changing HP linear_phase does NOT change LP (Butterworth)", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      setBandHighPass(bandId, makeButterworthFilter(80, true));
      setBandLowPass(bandId, makeButterworthFilter(2000, true));

      const hp = appState.bands[0].target.high_pass!;
      setBandHighPass(bandId, {
        filter_type: hp.filter_type,
        order: hp.order,
        freq_hz: hp.freq_hz,
        shape: hp.shape,
        linear_phase: false,
        q: hp.q,
      });

      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(false);
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("withOverride-style pattern: spread proxy + override (simulates UI)", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      setBandHighPass(bandId, makeGaussianFilter(80, true));
      setBandLowPass(bandId, makeGaussianFilter(2000, true));

      // Simulate what FilterBlock.withOverride does:
      // reads current config via SolidJS proxy, spreads it, overrides linear_phase
      const cur = appState.bands[0].target.high_pass!;
      const patched: FilterConfig = {
        filter_type: cur.filter_type,
        order: cur.order,
        freq_hz: cur.freq_hz,
        shape: cur.shape,
        linear_phase: cur.linear_phase,  // reads from proxy
        q: cur.q,
        // now override:
        ...{ linear_phase: false },
      };
      setBandHighPass(bandId, patched);

      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(false);
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("object identity: HP and LP configs are NOT the same reference", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      const sharedConfig = makeGaussianFilter(500, true);
      // Deliberately pass the SAME object to both HP and LP
      setBandHighPass(bandId, sharedConfig);
      setBandLowPass(bandId, sharedConfig);

      // Now mutate HP via store
      const hp = appState.bands[0].target.high_pass!;
      setBandHighPass(bandId, {
        filter_type: hp.filter_type,
        order: hp.order,
        freq_hz: hp.freq_hz,
        shape: hp.shape,
        linear_phase: false,
        q: hp.q,
      });

      // LP must still be true — store should deep-copy
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("rapid toggles: HP true→false→true, LP stays true", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      setBandHighPass(bandId, makeGaussianFilter(80, true));
      setBandLowPass(bandId, makeGaussianFilter(2000, true));

      // Toggle HP three times
      for (const val of [false, true, false]) {
        const hp = appState.bands[0].target.high_pass!;
        setBandHighPass(bandId, {
          filter_type: hp.filter_type,
          order: hp.order,
          freq_hz: hp.freq_hz,
          shape: hp.shape,
          linear_phase: val,
          q: hp.q,
        });
      }

      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(false);
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);

      dispose();
    });
  });

  it("set-null-first fix: HP and LP internal nodes are never shared", () => {
    createRoot((dispose) => {
      const bandId = appState.bands[0].id;

      // Set both HP and LP identically
      setBandHighPass(bandId, makeGaussianFilter(500, true));
      setBandLowPass(bandId, makeGaussianFilter(500, true));

      // Verify internal nodes are separate (not shared reference)
      const raw = unwrap(appState);
      expect(raw.bands[0].target.high_pass).not.toBe(raw.bands[0].target.low_pass);

      // Change HP — LP must not change
      const hp = appState.bands[0].target.high_pass!;
      setBandHighPass(bandId, {
        filter_type: hp.filter_type,
        order: hp.order,
        freq_hz: hp.freq_hz,
        shape: hp.shape,
        linear_phase: false,
        q: hp.q,
      });

      expect(appState.bands[0].target.high_pass?.linear_phase).toBe(false);
      expect(appState.bands[0].target.low_pass?.linear_phase).toBe(true);
      dispose();
    });
  });

  it("after addBand + assignDefaultTargets: HP/LP nodes are separate", () => {
    createRoot((dispose) => {
      addBand(); // triggers assignDefaultTargets

      const raw = unwrap(appState);
      for (let i = 0; i < raw.bands.length; i++) {
        const t = raw.bands[i].target;
        if (t.high_pass && t.low_pass) {
          expect(t.high_pass).not.toBe(t.low_pass);
        }
      }

      dispose();
    });
  });
});
