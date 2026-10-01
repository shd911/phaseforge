// b141.50 (audit 2026-10-01 H6): INV and alignment delay are host settings,
// not baked into the WAV — the export must say so.
import { describe, it, expect } from "vitest";
import { hostSettingsNote } from "../fir-export";

describe("hostSettingsNote", () => {
  it("silent for a plain band", () => {
    expect(hostSettingsNote(false, 0, "Mid")).toBeNull();
  });
  it("names the inversion", () => {
    expect(hostSettingsNote(true, 0, "Tw")).toMatch(/инверсия полярности/);
  });
  it("names the delay in ms", () => {
    const s = hostSettingsNote(false, 0.000473, "Woofer")!;
    expect(s).toMatch(/задержка 0\.473 мс/);
    expect(s).toMatch(/«Woofer»/);
  });
  it("names both", () => {
    expect(hostSettingsNote(true, 0.001, "Tw")).toMatch(/инверсия полярности и задержка 1\.000 мс/);
  });
});
