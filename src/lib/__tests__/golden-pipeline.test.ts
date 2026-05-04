// Golden snapshot reference for the b139.x unified-evaluation refactor.
// Captures the b138.4 magnitude-pipeline output for six canonical HP configs.
// Subsequent etaps must keep these snapshots stable bit-for-bit (modulo the
// 6-decimal rounding) — any drift means the refactor changed observable
// behavior and needs investigation.

import { describe, it, expect } from "vitest";
import { fixtureMeasurement, FIXTURE_CONFIGS } from "./fixtures/eval-fixtures";
import { gaussianFilterMagDb, subsonicMagDb } from "../plot-helpers";

function roundArr(arr: number[]): number[] {
  return arr.map((v) => Math.round(v * 1e6) / 1e6);
}

describe("golden pipeline (b139.0 reference)", () => {
  for (const cfg of FIXTURE_CONFIGS) {
    it(`${cfg.label} — magnitude snapshot`, () => {
      const meas = fixtureMeasurement();
      let mag = [...meas.magnitude];

      const hp = cfg.hp;
      if (hp && hp.filter_type === "Gaussian") {
        const hpMag = gaussianFilterMagDb(meas.freq, hp, false);
        mag = mag.map((m, i) => m + hpMag[i]);
        if (hp.subsonic_protect === true && hp.freq_hz > 40) {
          const subDb = subsonicMagDb(meas.freq, hp.freq_hz / 8);
          mag = mag.map((m, i) => m + subDb[i]);
        }
      }
      // Cases 5 ("lr4_baseline") and 6 ("no_hp_fullrange") intentionally
      // produce identical snapshots here — TS plot-helpers don't synthesize
      // non-Gaussian filters, so both fall through and capture only the
      // unchanged measurement magnitude. The Rust golden tests
      // (`evaluate_target_b139_golden_*`) are the actual reference for
      // those filter types; do NOT "fix" the duplicate snapshot.
      expect(roundArr(mag)).toMatchSnapshot();
    });
  }
});
