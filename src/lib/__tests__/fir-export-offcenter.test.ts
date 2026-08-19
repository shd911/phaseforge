// b141.16 (audit): every route pads the impulse with N/2 leading zeros so
// bands share a latency, but the shift is adaptive — when the (windowed) tail
// still carries content above -100 dB of peak, the shift shrinks and the band
// ships with less delay. Such a WAV is correct in isolation but plays early
// against the other bands in a convolver, so the export path must warn.
//
// b141.19: the check reads the APPLIED DELAY, not the peak index. A min-phase
// band's peak sits past its delay by the filter's own rise time — reading that
// as an offset both under-reports a real shortfall and hides one whenever the
// rise time happens to make up the difference.

import { describe, it, expect } from "vitest";
import { offCenterWavWarning } from "../fir-export";

const TAPS = 16_384;
const HALF = TAPS / 2;

describe("offCenterWavWarning (b141.16 → b141.19)", () => {
  it("silent when the band carries the full N/2 delay", () => {
    expect(offCenterWavWarning(HALF, TAPS, "Sub")).toBeNull();
  });

  it("silent within tolerance (64 samples ≈ 1.3 ms @ 48k)", () => {
    expect(offCenterWavWarning(HALF - 30, TAPS, "Sub")).toBeNull();
  });

  it("warns when the adaptive shift left the band short", () => {
    const warn = offCenterWavWarning(4096, TAPS, "Sub");
    expect(warn).toBeTruthy();
    expect(warn).toContain("Sub");
    expect(warn).toContain("4096");
  });

  it("reports the shortfall, not the delay, as the desync", () => {
    // 4096 of an expected 8192 → plays 4096 samples early.
    expect(offCenterWavWarning(4096, TAPS, "Sub")).toContain("4096 отсчётов раньше");
  });

  it("catches a tail-limited band whose rise time would have masked it", () => {
    // Pre-b141.19 this was measured from the peak: a min-phase cascade peaking
    // ~160 samples after its start would read 8160 — inside tolerance — while
    // the band actually shipped 100 samples of delay instead of 8192.
    expect(offCenterWavWarning(100, TAPS, "Woofer")).toBeTruthy();
  });

  it("silent on a degenerate impulse", () => {
    expect(offCenterWavWarning(0, 0, "Sub")).toBeNull();
  });
});
