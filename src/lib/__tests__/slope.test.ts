import { describe, it, expect } from "vitest";
import { orderToSlope, slopeToOrder, availableSlopes } from "../slope";

describe("slope mapping (b140.8.2)", () => {
  it("LR order 4 → 48 dB/oct, 48 → order 4", () => {
    expect(orderToSlope("LinkwitzRiley", 4)).toBe(48);
    expect(slopeToOrder("LinkwitzRiley", 48)).toBe(4);
  });

  it("Bessel order 3 → 18 dB/oct, 18 → order 3", () => {
    expect(orderToSlope("Bessel", 3)).toBe(18);
    expect(slopeToOrder("Bessel", 18)).toBe(3);
  });

  it("Butterworth order 2 → 12 dB/oct, 12 → order 2", () => {
    expect(orderToSlope("Butterworth", 2)).toBe(12);
    expect(slopeToOrder("Butterworth", 12)).toBe(2);
  });

  it("Custom roundtrip identical to Butterworth/Bessel", () => {
    for (let order = 1; order <= 8; order++) {
      expect(orderToSlope("Custom", order)).toBe(order * 6);
      expect(slopeToOrder("Custom", order * 6)).toBe(order);
    }
  });

  it("roundtrip preserves order across all available slopes", () => {
    for (const type of ["LinkwitzRiley", "Butterworth", "Bessel", "Custom"]) {
      for (const slope of availableSlopes(type)) {
        const order = slopeToOrder(type, slope);
        expect(orderToSlope(type, order)).toBe(slope);
      }
    }
  });

  it("availableSlopes is type-specific", () => {
    expect(availableSlopes("LinkwitzRiley")).not.toContain(18);
    expect(availableSlopes("LinkwitzRiley")).toContain(48);
    expect(availableSlopes("Butterworth")).toContain(18);
    expect(availableSlopes("Bessel")).toContain(18);
    expect(availableSlopes("Custom")).toContain(18);
  });

  it("unknown filter type returns empty slope list and slope 0", () => {
    expect(availableSlopes("Gaussian")).toEqual([]);
    expect(orderToSlope("Gaussian", 4)).toBe(0);
  });

  it("Bessel order=3 dropdown rendering uses slope=18", () => {
    // Reproduces the user-reported scenario: Bessel order=3, dropdown
    // should display 18 dB/oct, which must be in availableSlopes.
    const slope = orderToSlope("Bessel", 3);
    expect(slope).toBe(18);
    expect(availableSlopes("Bessel")).toContain(slope);
  });
});
