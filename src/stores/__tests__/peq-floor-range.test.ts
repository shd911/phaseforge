import { describe, it, expect } from "vitest";
import { trimRangeByFloor } from "../peq-optimize";

// b141.84: «Порог dB» trims the PEQ range in direct mode too.
describe("trimRangeByFloor", () => {
  // LR2 low-pass @ 250 Hz, as on the real Purifi woofer band
  const freq = Array.from({ length: 400 }, (_, i) => 20 * Math.pow(2, i / 40));
  const mag = freq.map((f) => -20 * Math.log10(1 + (f / 250) ** 4));

  it("cuts a direct range down to where the target is above the floor", () => {
    const [lo, hi] = trimRangeByFloor(freq, mag, 30, 21110, -60);
    expect(lo).toBe(30);
    expect(hi).toBeGreaterThan(1300);
    expect(hi).toBeLessThan(1500); // −60 dB of LR2 @ 250 ≈ 1.41 kHz, far below 3.9 kHz
  });

  it("never widens the given range", () => {
    expect(trimRangeByFloor(freq, mag, 50, 400, -60)).toEqual([50, 400]);
  });

  it("leaves the range alone when the target never clears the floor", () => {
    expect(trimRangeByFloor(freq, mag, 30, 21110, 10)).toEqual([30, 21110]);
  });
});
