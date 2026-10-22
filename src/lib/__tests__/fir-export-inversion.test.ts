import { describe, it, expect } from "vitest";
import { wavImpulse } from "../fir-export";

// b141.100: an inverted band's WAV carries its polarity (the Export tab draws
// the file; the convolver needs no flip).
describe("export inversion", () => {
  const h = [0, 0.5, -1, 0.25];
  it("negates an inverted band's impulse, leaves a normal one as is", () => {
    expect(wavImpulse(h, true)).toEqual([-0, -0.5, 1, -0.25]);
    expect(wavImpulse(h, false)).toBe(h);
  });
});
