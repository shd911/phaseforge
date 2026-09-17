// b141.43: one wheel model for NumberInput, the PEQ table, the Σ delay and
// PEQ Q on the chart. Trackpads send many small pixel deltas per gesture —
// one step per event made values run away.
import { describe, it, expect, vi } from "vitest";
import {
  wheelSteps, newWheelAccumulator, throttleTrailing, WHEEL_PX_PER_STEP, WHEEL_GESTURE_IDLE_MS,
} from "../wheel-step";

describe("wheelSteps", () => {
  it("a notched mouse wheel gives one step per notch", () => {
    const st = newWheelAccumulator();
    expect(wheelSteps({ deltaY: -4, deltaMode: 0, wheelDeltaY: 120 }, st, 0)).toBe(1);
    expect(wheelSteps({ deltaY: 4, deltaMode: 0, wheelDeltaY: -120 }, st, 10)).toBe(-1);
    expect(wheelSteps({ deltaY: -8, deltaMode: 0, wheelDeltaY: 240 }, st, 20)).toBe(2);
    expect(wheelSteps({ deltaY: 3, deltaMode: 1 }, st, 30)).toBe(-1);
  });

  // Recorded on the user's mouse (macOS WebKit): slow clicks are exactly
  // 4.000244 px each, spaced 120–300 ms — within one gesture.
  it("slow macOS mouse clicks give one step each", () => {
    const st = newWheelAccumulator();
    const slow = [[72015, 4.000244140625], [72196, 4.000244140625], [72496, 4.000244140625]];
    expect(slow.map(([t, d]) => wheelSteps({ deltaY: d, deltaMode: 0, wheelDeltaY: -12 }, st, t))).toEqual([-1, -1, -1]);
  });

  it("an accelerated macOS spin still scales with distance", () => {
    const st = newWheelAccumulator();
    const fast: [number, number, number][] = [
      [74707, 4.000244140625, -12], [74722, 51.61865234375, -154],
      [74746, 215.02685546875, -645], [74772, 309.6551513671875, -928],
    ];
    const total = fast.reduce((s, [t, d, w]) => s + wheelSteps({ deltaY: d, deltaMode: 0, wheelDeltaY: w }, st, t), 0);
    expect(total).toBe(-15); // 1 notch + (51.6+215+309.7)/40 = 14.4 → 14
  });

  it("trackpad pixel deltas accumulate into steps instead of one per event", () => {
    const st = newWheelAccumulator();
    let steps = 0;
    // a 200 px swipe in 40 events of 5 px
    for (let k = 0; k < 40; k++) steps += wheelSteps({ deltaY: -5, deltaMode: 0, wheelDeltaY: 15 }, st, k * 8);
    expect(steps).toBe(Math.trunc(200 / WHEEL_PX_PER_STEP));
  });

  it("a pause or a direction change drops the leftover", () => {
    const st = newWheelAccumulator();
    expect(wheelSteps({ deltaY: -30, deltaMode: 0 }, st, 0)).toBe(0);
    expect(wheelSteps({ deltaY: -30, deltaMode: 0 }, st, WHEEL_GESTURE_IDLE_MS + 1)).toBe(0);
    expect(wheelSteps({ deltaY: 30, deltaMode: 0 }, st, WHEEL_GESTURE_IDLE_MS + 10)).toBe(0);
    expect(wheelSteps({ deltaY: 30, deltaMode: 0 }, st, WHEEL_GESTURE_IDLE_MS + 20)).toBe(-1);
  });
});

describe("throttleTrailing", () => {
  it("passes the first value, throttles the rest, delivers the last", () => {
    vi.useFakeTimers();
    const now = vi.spyOn(performance, "now");
    let t = 1000; now.mockImplementation(() => t);
    const seen: number[] = [];
    const f = throttleTrailing((v: number) => seen.push(v), 120);
    f(1); t += 10; f(2); t += 10; f(3);
    expect(seen).toEqual([1]);
    t += 200; vi.advanceTimersByTime(200);
    expect(seen).toEqual([1, 3]);
    now.mockRestore(); vi.useRealTimers();
  });
});
