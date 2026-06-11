// b141.8 (audit): WAV peak convention — IIR path centers the impulse at
// N/2, cepstral min-phase ships peak-at-0. A project that mixes the two
// routes for min-phase bands exports WAVs with N/2 relative latency: the
// crossover desynchronizes in any convolver. The export path must detect
// the mix and warn.

import { describe, it, expect, vi } from "vitest";
import type { BandState } from "../../stores/bands";
import type { FilterConfig } from "../types";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "pick_fir_route") {
      // Mirror Rust fir::route_for: IIR only for min-phase main, no
      // subsonic, all active crossovers realisable (LR/BW/Custom).
      const { hp, lp, linearMain, subsonicCutoffHz } = args;
      const realisable = (f: any) =>
        !f || f.filter_type === "LinkwitzRiley"
           || f.filter_type === "Butterworth"
           || f.filter_type === "Custom";
      if (linearMain) return "Cepstral";
      if (subsonicCutoffHz !== null) return "Cepstral";
      if (!realisable(hp) || !realisable(lp)) return "Cepstral";
      return "Iir";
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { mixedWavConventionWarning } from "../fir-export";

function filt(over?: Partial<FilterConfig>): FilterConfig {
  return {
    filter_type: "LinkwitzRiley", freq_hz: 100, order: 4,
    q: null, linear_phase: false, subsonic_protect: null,
    ...over,
  } as FilterConfig;
}

function band(name: string, hp: FilterConfig | null, lp: FilterConfig | null, targetEnabled = true): BandState {
  return {
    id: name, name,
    measurement: null, measurementFile: null,
    settings: { smoothing: "off" },
    target: {
      reference_level_db: 0, tilt_db_per_octave: 0, tilt_ref_freq: 1000,
      high_pass: hp, low_pass: lp, low_shelf: null, high_shelf: null,
    },
    targetEnabled, inverted: false, linkedToNext: false,
    peqBands: [], peqOptimizedTarget: null, exclusionZones: [],
    firResult: null, crossNormDb: 0, color: "#888", alignmentDelay: 0,
  } as unknown as BandState;
}

describe("mixedWavConventionWarning (b141.8)", () => {
  it("warns when min-phase IIR (centered) and Gaussian min-phase (peak-at-0) mix", async () => {
    const sub = band("Sub", filt({ freq_hz: 30 }), filt({ freq_hz: 120 }));
    const tweeter = band("Tweeter", filt({ filter_type: "Gaussian", freq_hz: 2500 }), null);
    const warn = await mixedWavConventionWarning([sub, tweeter]);
    expect(warn).toBeTruthy();
    expect(warn).toContain("Sub");
    expect(warn).toContain("Tweeter");
  });

  it("warns when subsonic-protect (cepstral) mixes with plain IIR bands", async () => {
    // subsonic_protect is a Gaussian-only feature, active for HP > 40 Hz
    const sub = band("Sub", filt({ filter_type: "Gaussian", freq_hz: 60, subsonic_protect: true }), filt({ freq_hz: 240 }));
    const mid = band("Mid", filt({ freq_hz: 120 }), filt({ freq_hz: 2500 }));
    const warn = await mixedWavConventionWarning([sub, mid]);
    expect(warn).toBeTruthy();
  });

  it("silent when all bands share the IIR (centered) convention", async () => {
    const a = band("A", filt({ freq_hz: 30 }), filt({ freq_hz: 120 }));
    const b = band("B", filt({ freq_hz: 120 }), filt({ freq_hz: 2500 }));
    expect(await mixedWavConventionWarning([a, b])).toBeNull();
  });

  it("silent when all bands are linear-phase (all centered)", async () => {
    const a = band("A", filt({ linear_phase: true }), null);
    const b = band("B", filt({ linear_phase: true, freq_hz: 500 }), null);
    expect(await mixedWavConventionWarning([a, b])).toBeNull();
  });

  it("ignores disabled-target bands and single-band projects", async () => {
    const a = band("A", filt({ freq_hz: 30 }), null);
    const gOff = band("G", filt({ filter_type: "Gaussian" }), null, false);
    expect(await mixedWavConventionWarning([a, gOff])).toBeNull();
    expect(await mixedWavConventionWarning([a])).toBeNull();
  });
});
