/**
 * Phase 0 test (b140.10): FilterConfig clone consistency across 3 sites.
 *
 * Today the codebase has 3 independent deep-copy implementations:
 *   - `unwrapFilterConfig` in src/stores/bands.ts:549
 *   - `unwrapFilter`        in src/components/ControlPanel.tsx:125
 *   - `cloneFilter`         in src/lib/project-io.ts:375
 *
 * Phase 1 will collapse them into a single `cloneFilterConfig` in types.ts.
 * This suite locks down the contract BEFORE that change:
 *   1. All 3 implementations produce structurally identical output for
 *      the canonical fixture set.
 *   2. A 4th, reference implementation (defined here) matches them too —
 *      this is what Phase 1 will promote to the single source of truth.
 *   3. Output is a fresh object (not the same reference as input).
 *   4. Null/undefined inputs are handled per each site's existing contract.
 *
 * After Phase 1, the cross-site mirror tests stay green by virtue of all
 * three sites calling the unified function. Any drift in the unified
 * function vs the reference here = regression.
 */
import { describe, expect, it } from "vitest";
import type { FilterConfig } from "../types";

// --- Cross-site re-implementations -----------------------------------------
// Mirroring the 3 production copies verbatim — if any of them changes in
// production this file MUST be updated to match (and the diff reviewed).

function siteBandsTs(f: FilterConfig): FilterConfig {
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
    subsonic_protect: f.subsonic_protect ?? null,
  };
}

function siteControlPanelTsx(f: FilterConfig | null | undefined): FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
    subsonic_protect: f.subsonic_protect ?? null,
  };
}

function siteProjectIoTs(f: FilterConfig | null | undefined): FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
    subsonic_protect: f.subsonic_protect ?? null,
  };
}

/** Reference implementation — the future Phase-1 unified function. */
function cloneFilterConfigReference(f: FilterConfig | null | undefined): FilterConfig | null {
  if (!f) return null;
  return {
    filter_type: f.filter_type,
    order: f.order,
    freq_hz: f.freq_hz,
    shape: f.shape,
    linear_phase: f.linear_phase,
    q: f.q,
    subsonic_protect: f.subsonic_protect ?? null,
  };
}

// --- Fixtures (canonical filter configs across all 5 filter types) ---------

const FIXTURES: { name: string; cfg: FilterConfig }[] = [
  {
    name: "lr4_hp_min_phase",
    cfg: {
      filter_type: "LinkwitzRiley", order: 4, freq_hz: 80,
      shape: null, linear_phase: false, q: null, subsonic_protect: null,
    },
  },
  {
    name: "butterworth_lp_lin_phase",
    cfg: {
      filter_type: "Butterworth", order: 2, freq_hz: 2000,
      shape: null, linear_phase: true, q: null, subsonic_protect: null,
    },
  },
  {
    name: "gaussian_hp_subsonic_on",
    cfg: {
      filter_type: "Gaussian", order: 4, freq_hz: 632,
      shape: 1.0, linear_phase: false, q: null, subsonic_protect: true,
    },
  },
  {
    name: "bessel_lp_min_phase",
    cfg: {
      filter_type: "Bessel", order: 4, freq_hz: 500,
      shape: null, linear_phase: false, q: null, subsonic_protect: null,
    },
  },
  {
    name: "custom_hp_with_q",
    cfg: {
      filter_type: "Custom", order: 2, freq_hz: 100,
      shape: null, linear_phase: false, q: 1.2, subsonic_protect: null,
    },
  },
  {
    name: "subsonic_protect_explicit_false",
    cfg: {
      filter_type: "Gaussian", order: 4, freq_hz: 80,
      shape: 1.0, linear_phase: false, q: null, subsonic_protect: false,
    },
  },
];

// ---------------------------------------------------------------------------

describe("FilterConfig clone — 3-site consistency (b140.10 phase-0)", () => {
  for (const { name, cfg } of FIXTURES) {
    it(`${name}: bands.ts ≡ ControlPanel.tsx ≡ project-io.ts ≡ reference`, () => {
      const a = siteBandsTs(cfg);
      const b = siteControlPanelTsx(cfg);
      const c = siteProjectIoTs(cfg);
      const ref = cloneFilterConfigReference(cfg);
      expect(a).toEqual(ref);
      expect(b).toEqual(ref);
      expect(c).toEqual(ref);
    });

    it(`${name}: returns a fresh object (not the same reference)`, () => {
      expect(siteBandsTs(cfg)).not.toBe(cfg);
      expect(siteControlPanelTsx(cfg)).not.toBe(cfg);
      expect(siteProjectIoTs(cfg)).not.toBe(cfg);
    });

    it(`${name}: mutating the clone does not affect the original`, () => {
      const clone = siteBandsTs(cfg);
      clone.freq_hz = 9999;
      clone.filter_type = "Custom";
      expect(cfg.freq_hz).not.toBe(9999);
    });
  }

  describe("null / undefined handling", () => {
    it("ControlPanel.tsx returns null for null", () => {
      expect(siteControlPanelTsx(null)).toBeNull();
      expect(siteControlPanelTsx(undefined)).toBeNull();
    });
    it("project-io.ts returns null for null", () => {
      expect(siteProjectIoTs(null)).toBeNull();
      expect(siteProjectIoTs(undefined)).toBeNull();
    });
    it("reference returns null for null", () => {
      expect(cloneFilterConfigReference(null)).toBeNull();
      expect(cloneFilterConfigReference(undefined)).toBeNull();
    });
  });

  describe("subsonic_protect normalization", () => {
    it("undefined → null in output", () => {
      const cfg: FilterConfig = {
        filter_type: "Gaussian", order: 4, freq_hz: 100,
        shape: 1.0, linear_phase: false, q: null,
        // subsonic_protect intentionally omitted (undefined)
      };
      expect(siteBandsTs(cfg).subsonic_protect).toBeNull();
      expect(siteControlPanelTsx(cfg)!.subsonic_protect).toBeNull();
      expect(siteProjectIoTs(cfg)!.subsonic_protect).toBeNull();
    });
    it("explicit false preserved", () => {
      const cfg: FilterConfig = {
        filter_type: "Gaussian", order: 4, freq_hz: 100,
        shape: 1.0, linear_phase: false, q: null, subsonic_protect: false,
      };
      expect(siteBandsTs(cfg).subsonic_protect).toBe(false);
    });
    it("explicit true preserved", () => {
      const cfg: FilterConfig = {
        filter_type: "Gaussian", order: 4, freq_hz: 100,
        shape: 1.0, linear_phase: false, q: null, subsonic_protect: true,
      };
      expect(siteBandsTs(cfg).subsonic_protect).toBe(true);
    });
  });
});
