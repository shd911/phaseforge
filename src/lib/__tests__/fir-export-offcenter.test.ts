// b141.16 (audit): the unified WAV peak convention centers every impulse at
// ~N/2, but the shift is adaptive — when the (windowed) tail still carries
// content above -100 dB of peak, the shift shrinks and the peak lands short
// of center. Such a WAV is correct in isolation but desynchronizes against
// other bands' centered WAVs in a convolver. The export path must detect the
// off-center peak and warn (replaces the broader b141.8 mixed-convention
// warning removed in b141.14).

import { describe, it, expect } from "vitest";
import { offCenterWavWarning } from "../fir-export";

function centeredImpulse(taps: number, peakAt: number): number[] {
  const imp = new Array(taps).fill(0);
  imp[peakAt] = 1.0;
  // small causal tail after the peak
  for (let i = peakAt + 1; i < Math.min(peakAt + 50, taps); i++) {
    imp[i] = Math.exp(-(i - peakAt) / 10);
  }
  return imp;
}

describe("offCenterWavWarning (b141.16)", () => {
  it("silent when the peak is at N/2", () => {
    expect(offCenterWavWarning(centeredImpulse(16384, 8192), "Sub")).toBeNull();
  });

  it("silent within tolerance of N/2 (raw cascade rise offset)", () => {
    expect(offCenterWavWarning(centeredImpulse(16384, 8192 + 30), "Sub")).toBeNull();
  });

  it("warns when the adaptive shift left the peak short of center", () => {
    const warn = offCenterWavWarning(centeredImpulse(16384, 4096), "Sub");
    expect(warn).toBeTruthy();
    expect(warn).toContain("Sub");
  });

  it("silent on empty impulse", () => {
    expect(offCenterWavWarning([], "Sub")).toBeNull();
  });
});
