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

import { boostLimitNote } from "../fir-export";

describe("boostLimitNote (b141.53)", () => {
  it("silent on the cepstral route (it clips)", () => {
    expect(boostLimitNote("cepstral", 30, 24, "Sub")).toBeNull();
  });
  it("silent within the limit", () => {
    expect(boostLimitNote("iir", 23.9, 24, "Sub")).toBeNull();
  });
  it("warns on the IIR route above the limit", () => {
    expect(boostLimitNote("iir", 30, 24, "Sub")).toMatch(/30\.0 dB выше лимита 24\.0 dB/);
  });
});

import { convolverDelaySeconds } from "../fir-export";

describe("convolverDelaySeconds (b141.61)", () => {
  it("full N/2 delay → just the alignment", () => {
    expect(convolverDelaySeconds(0.00082, 131072, 262144, 352800)).toBeCloseTo(0.00082, 9);
  });
  it("VPV2 rip: WAV delay 40 803 of 131 072 → +255.9 ms", () => {
    expect(convolverDelaySeconds(0, 40803, 262144, 352800) * 1000).toBeCloseTo(255.86, 1);
  });
});
