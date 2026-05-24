/**
 * Phase 0 test (b140.10): FIR routing decision table.
 *
 * b140.15.5: `pickFirRoute` is now an async wrapper around the Rust
 * `pick_fir_route` Tauri command. JS no longer carries its own
 * predicate. This test still documents the routing table for human
 * readers and verifies the JS wrapper marshals values correctly; the
 * actual decision logic lives in `src-tauri/src/fir/dispatch.rs` and
 * is exercised end-to-end by `tests/pipeline_contract.rs`.
 *
 * Tauri is mocked: we reproduce the Rust `route_for` predicate
 * locally inside the mock. This mock IS the duplicated definition
 * the audit flagged — but it's now isolated inside a test file
 * (not a production code path), and the cross-language drift surface
 * is the Rust-side `pipeline_contract.rs` baseline.
 */
import { describe, expect, it, vi } from "vitest";
import type { FilterConfig } from "../types";

// --- Mock Tauri: mirror of Rust route_for ----------------------------------

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd !== "pick_fir_route") throw new Error(`Unmocked command: ${cmd}`);
    const { hp, lp, linearMain, subsonicCutoffHz } = args;
    const realisable = (f: FilterConfig | null) =>
      !f || f.filter_type === "LinkwitzRiley"
         || f.filter_type === "Butterworth"
         || f.filter_type === "Custom";
    if (linearMain) return "Cepstral";
    if (subsonicCutoffHz !== null) return "Cepstral";
    if (!realisable(hp)) return "Cepstral";
    if (!realisable(lp)) return "Cepstral";
    return "Iir";
  }),
}));

import { pickFirRoute } from "../fir-routing";

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
  { desc: "LR4 HP only, min-phase",                hp: LR,   lp: null, linearMain: false, subsonic: null, expected: "iir" },
  { desc: "BW LP only, min-phase",                 hp: null, lp: BW,   linearMain: false, subsonic: null, expected: "iir" },
  { desc: "Custom HP + Custom LP, min-phase",      hp: CUST, lp: CUST, linearMain: false, subsonic: null, expected: "iir" },
  { desc: "LR HP + BW LP, min-phase",              hp: LR,   lp: BW,   linearMain: false, subsonic: null, expected: "iir" },
  { desc: "no HP no LP, min-phase",                hp: null, lp: null, linearMain: false, subsonic: null, expected: "iir" },

  { desc: "LR HP + LR LP but linearMain=true",     hp: LR,   lp: LR,   linearMain: true,  subsonic: null, expected: "cepstral" },
  { desc: "no HP/LP but linearMain=true",          hp: null, lp: null, linearMain: true,  subsonic: null, expected: "cepstral" },

  { desc: "LR HP min-phase + subsonic cutoff",     hp: LR,   lp: null, linearMain: false, subsonic: 50.0, expected: "cepstral" },
  { desc: "no HP min-phase + subsonic cutoff",     hp: null, lp: null, linearMain: false, subsonic: 50.0, expected: "cepstral" },

  { desc: "Gaussian HP, min-phase, no subsonic",   hp: GAUSS, lp: null, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "Gaussian LP",                           hp: null,  lp: GAUSS, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "LR HP + Gaussian LP",                   hp: LR,    lp: GAUSS, linearMain: false, subsonic: null, expected: "cepstral" },

  { desc: "Bessel HP, min-phase, no subsonic",     hp: BESS, lp: null, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "Bessel LP",                             hp: null, lp: BESS, linearMain: false, subsonic: null, expected: "cepstral" },
  { desc: "Bessel HP + BW LP",                     hp: BESS, lp: BW,   linearMain: false, subsonic: null, expected: "cepstral" },

  { desc: "Gaussian + linearMain + subsonic",      hp: GAUSS, lp: BESS, linearMain: true, subsonic: 50.0, expected: "cepstral" },
];

describe("pickFirRoute — decision table (b140.10 → b140.15.5)", () => {
  for (const row of TABLE) {
    it(`${row.desc} → ${row.expected}`, async () => {
      const got = await pickFirRoute(row.hp, row.lp, row.linearMain, row.subsonic);
      expect(got).toBe(row.expected);
    });
  }

  describe("undefined HP/LP behave like null", () => {
    it("undefined HP, undefined LP, min-phase, no subsonic → iir", async () => {
      expect(await pickFirRoute(undefined, undefined, false, null)).toBe("iir");
    });
    it("undefined HP, Gaussian LP → cepstral", async () => {
      expect(await pickFirRoute(undefined, GAUSS, false, null)).toBe("cepstral");
    });
  });

  describe("priority of disqualifiers (any single one routes to cepstral)", () => {
    it("only linearMain set", async () => {
      expect(await pickFirRoute(LR, LR, true, null)).toBe("cepstral");
    });
    it("only subsonic set", async () => {
      expect(await pickFirRoute(LR, LR, false, 1.0)).toBe("cepstral");
    });
    it("only HP non-realisable", async () => {
      expect(await pickFirRoute(GAUSS, LR, false, null)).toBe("cepstral");
    });
    it("only LP non-realisable", async () => {
      expect(await pickFirRoute(LR, BESS, false, null)).toBe("cepstral");
    });
  });
});
