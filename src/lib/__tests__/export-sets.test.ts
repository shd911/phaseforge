// b141.35: the Export dropdowns are the only place a user picks sample rate
// and tap count, and `loadProject` silently falls back to 48000/65536 for any
// value not in these sets. Two things can drift apart unnoticed: the tap set
// versus the Rust ceiling (`fir::MAX_TAPS`, which gates the vDSP FFT), and the
// rate set versus the range `validate_project` accepts. Both are pinned here.
import { describe, it, expect } from "vitest";
import { STANDARD_SAMPLE_RATES, STANDARD_TAPS, fMaxForRate, F_MAX_WORK } from "../types";

/** Mirrors `fir::MAX_TAPS` in src-tauri/src/fir/mod.rs. */
const RUST_MAX_TAPS = 1_048_576;
/** Mirrors the range check in `validate_project` (src-tauri/src/project.rs). */
const RUST_SR_RANGE = [8_000, 768_000] as const;

describe("export dropdown sets", () => {
  it("offers the DSD-derived and 8x PCM rates", () => {
    expect(STANDARD_SAMPLE_RATES).toEqual([44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]);
  });

  it("offers tap counts up to the Rust ceiling", () => {
    expect(STANDARD_TAPS).toEqual([4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]);
    expect(STANDARD_TAPS[STANDARD_TAPS.length - 1]).toBe(RUST_MAX_TAPS);
  });

  it("every tap count passes the Rust taps_valid gate", () => {
    for (const t of STANDARD_TAPS) {
      expect(Number.isInteger(Math.log2(t)), `${t} must be a power of two`).toBe(true);
      expect(t).toBeGreaterThanOrEqual(32);
      expect(t).toBeLessThanOrEqual(RUST_MAX_TAPS);
    }
  });

  it("every sample rate is inside the range validate_project accepts", () => {
    for (const sr of STANDARD_SAMPLE_RATES) {
      expect(sr).toBeGreaterThanOrEqual(RUST_SR_RANGE[0]);
      expect(sr).toBeLessThanOrEqual(RUST_SR_RANGE[1]);
    }
  });

  it("the new rates put the whole 30 kHz working range under Nyquist", () => {
    // Below 88.2 kHz the grid is capped by Nyquist·0.95; at and above it the
    // full working range fits, so the new rates must not shrink it.
    for (const sr of [352800, 384000]) {
      expect(fMaxForRate(sr)).toBe(F_MAX_WORK);
    }
  });

  it("the dropdown labels stay short", () => {
    const srLabels = STANDARD_SAMPLE_RATES.map((sr) => (sr >= 1000 ? sr / 1000 + "k" : String(sr)));
    expect(srLabels).toContain("352.8k");
    expect(srLabels).toContain("384k");
    const tapLabels = STANDARD_TAPS.map((t) => (t >= 1024 ? t / 1024 + "K" : String(t)));
    expect(tapLabels).toContain("512K");
    expect(tapLabels).toContain("1024K");
  });
});
