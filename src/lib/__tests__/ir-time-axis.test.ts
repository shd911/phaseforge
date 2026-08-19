// b141.23 (audit): compute_impulse used to serialise its time axis in full —
// a pure ramp worth 2.7 MB at 65536 taps and 11.3 MB at the 262144 the
// window-growth loop can reach, a third of the payload, ~15 calls per IR
// render. It now returns dt + pre_peak_count and the axis is rebuilt here.

import { describe, it, expect } from "vitest";
import { buildIrTimeAxis, irTime } from "../band-evaluator/grid";

describe("IR time axis (b141.23)", () => {
  it("places t=0 at the pre-peak boundary", () => {
    const dt = 1 / 48_000;
    const axis = buildIrTimeAxis(dt, 3, 8);
    expect(axis[3]).toBeCloseTo(0, 15);
    expect(axis[0]).toBeCloseTo(-3 * dt, 15);
    expect(axis[7]).toBeCloseTo(4 * dt, 15);
  });

  it("is strictly increasing with a constant step", () => {
    const axis = buildIrTimeAxis(1 / 44_100, 128, 512);
    for (let i = 1; i < axis.length; i++) {
      expect(axis[i]).toBeGreaterThan(axis[i - 1]);
      expect(axis[i] - axis[i - 1]).toBeCloseTo(1 / 44_100, 15);
    }
  });

  it("irTime sizes the axis from the impulse it belongs to", () => {
    const r = { dt: 1 / 96_000, pre_peak_count: 2, impulse: [0, 0, 1, 0.5], step: [0, 0, 1, 1] };
    expect(irTime(r)).toHaveLength(4);
    expect(irTime(r)[2]).toBeCloseTo(0, 15);
  });
});
