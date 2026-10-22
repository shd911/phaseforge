import { describe, it, expect } from "vitest";
import { wavImpulse, hostSettingsNote } from "../fir-export";

// b141.98: «INV в WAV» — polarity written into the file instead of the convolver.
describe("export inversion", () => {
  const h = [0, 0.5, -1, 0.25];
  it("negates only an inverted band, only when baking is on", () => {
    expect(wavImpulse(h, true, true)).toEqual([-0, -0.5, 1, -0.25]);
    expect(wavImpulse(h, true, false)).toBe(h);
    expect(wavImpulse(h, false, true)).toBe(h);
  });
  it("the convolver note drops the inversion once it is in the WAV", () => {
    expect(hostSettingsNote(true, 0, "74")).toContain("инверсию полярности");
    // the export passes `inverted && !bake`
    expect(hostSettingsNote(false, 0, "74")).toBeNull();
    expect(hostSettingsNote(false, 0.0005, "74")).not.toContain("инверс");
  });
});
