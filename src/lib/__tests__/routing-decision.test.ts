/**
 * Phase 0 test (b140.10): FIR routing decision table.
 *
 * Locks down the (hp, lp, linearMain, subsonicCutoff) → ("iir" | "cepstral")
 * decision before Phase 2 (FirPipeline trait) and Phase 4 (split of
 * band-evaluator.ts). Any refactor must preserve every cell in the table.
 *
 * The truth table is exhaustive over the relevant inputs:
 *   - filter_type ∈ {LR, BW, Custom, Gaussian, Bessel, null}
 *   - linearMain ∈ {false, true}
 *   - subsonicCutoffHz ∈ {null, 50.0}
 *
 * For each combination the test asserts the expected route. The expected
 * column is derived from the documented contract in fir-routing.ts —
 * NOT from re-reading the implementation.
 */
import { describe, expect, it } from "vitest";
import { pickFirRoute } from "../fir-routing";
import type { FilterConfig } from "../types";

function mk(filter_type: FilterConfig["filter_type"]): FilterConfig {
  return {
    filter_type,
    order: 4,
    freq_hz: 1000,
    shape: filter_type === "Gaussian" ? 1.0 : null,
    linear_phase: false,
    q: filter_type === "Custom" ? 0.707 : null,
    subsonic_protect: null,
  };
}

const LR = mk("LinkwitzRiley");
const BW = mk("Butterworth");
const CUST = mk("Custom");
const GAUSS = mk("Gaussian");
const BESS = mk("Bessel");

type Row = {
  desc: string;
  hp: FilterConfig | null;
  lp: FilterConfig | null;
  linearMain: boolean;
  subsonic: number | null;
  expected: "iir" | "cepstral";
};

const TABLE: Row[] = [
  // --- IIR-eligible: all paths min-phase + no subsonic + LR/BW/Custom only
  { desc: "LR4 HP only, min-phase",                hp: LR,   lp: null, linearMain: false, subsonic: null, expected: "iir" },
  { desc: "BW LP only, min-phase",                 hp: null, lp: BW,   linearMain: false, subsonic: null, expected: "iir" },
  { desc: "Custom HP + Custom LP, min-phase",      hp: CUST, lp: CUST, linearMain: false, subsonic: null, expected: "iir" },
  { desc: "LR HP + BW LP, min-phase",              hp: LR,   lp: BW,   linearMain: false, subsonic: null, expected: "iir" },
  { desc: "no HP no LP, min-phase",                hp: null, lp: null, linearMain: false, subsonic: null, expected: "iir" },

  // --- Cepstral: linear-phase main
  { desc: "LR HP + LR LP but linearMain=true",     hp: LR,   lp: LR,   linearMain: true,  subsonic: null, expected: "cepstral" },
  { desc: "no HP/LP but linearMain=true",          hp: null, lp: null, linearMain: true,  subsonic: null, expected: "cepstral" },

  // --- Cepstral: subsonic cutoff requested
  { desc: "LR HP min-phase + subsonic cutoff",     hp: LR,   lp: null, linearMain: false, subsonic: 50.0, expected: "cepstral" },
  { desc: "no HP min-phase + subsonic cutoff",     hp: null, lp: null, linearMain: false, subsonic: 50.0, expected: "cepstral" },

  // --- Cepstral: at least one Gaussian filter
  { desc: "Gaussian HP, min-phase, no subsonic",   hp: GAUSS, lp: null, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "Gaussian LP",                           hp: null,  lp: GAUSS, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "LR HP + Gaussian LP",                   hp: LR,    lp: GAUSS, linearMain: false, subsonic: null, expected: "cepstral" },

  // --- Cepstral: at least one Bessel filter
  { desc: "Bessel HP, min-phase, no subsonic",     hp: BESS, lp: null, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "Bessel LP",                             hp: null, lp: BESS, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "Bessel HP + BW LP",                     hp: BESS, lp: BW,   linearMain: false, subsonic: null, expected: "cepstral" },

  // --- Worst case: all three disqualifiers
  { desc: "Gaussian + linearMain + subsonic",      hp: GAUSS, lp: BESS, linearMain: true, subsonic: 50.0, expected: "cepstral" },
];

describe("pickFirRoute — decision table (b140.10 phase-0)", () => {
  for (const row of TABLE) {
    it(`${row.desc} → ${row.expected}`, () => {
      const got = pickFirRoute(row.hp, row.lp, row.linearMain, row.subsonic);
      expect(got).toBe(row.expected);
    });
  }

  describe("undefined HP/LP behave like null", () => {
    it("undefined HP, undefined LP, min-phase, no subsonic → iir", () => {
      expect(pickFirRoute(undefined, undefined, false, null)).toBe("iir");
    });
    it("undefined HP, Gaussian LP → cepstral", () => {
      expect(pickFirRoute(undefined, GAUSS, false, null)).toBe("cepstral");
    });
  });

  describe("priority of disqualifiers (any single one routes to cepstral)", () => {
    it("only linearMain set", () => {
      expect(pickFirRoute(LR, LR, true, null)).toBe("cepstral");
    });
    it("only subsonic set", () => {
      expect(pickFirRoute(LR, LR, false, 1.0)).toBe("cepstral");
    });
    it("only HP non-realisable", () => {
      expect(pickFirRoute(GAUSS, LR, false, null)).toBe("cepstral");
    });
    it("only LP non-realisable", () => {
      expect(pickFirRoute(LR, BESS, false, null)).toBe("cepstral");
    });
  });
});
