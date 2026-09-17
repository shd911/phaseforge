// b141.37: pre-ring and pre-peak floor are separate Export metrics — see
// export-metrics.ts for the measurement that motivated the split.
import { describe, it, expect } from "vitest";
import {
  analyzePreRing, ringZone, ringPeriods, PRE_RING_DB, FLOOR_GOOD_DB, FLOOR_BAD_DB,
} from "../export-metrics";
import type { BandState } from "../../stores/bands";
import type { FilterConfig } from "../types";

const SR = 48_000;

/** Symmetric linear-phase-like impulse: peak at N/2, envelope falling by
 *  `dbPerMs` on both sides, plus an optional flat plateau before the peak
 *  from `plateau.fromMs` onward. */
function synth(n: number, dbPerMs: number, plateau?: { db: number; fromMs: number }): Float64Array {
  const h = new Float64Array(n);
  const c = n / 2;
  for (let i = 0; i < n; i++) {
    const ms = (Math.abs(i - c) * 1000) / SR;
    h[i] = Math.pow(10, (-dbPerMs * ms) / 20) * (i % 2 === 0 ? 1 : -1);
    if (plateau && i < c && ms >= plateau.fromMs) {
      h[i] += Math.pow(10, plateau.db / 20) * (i % 3 === 0 ? 1 : -1);
    }
  }
  return h;
}

function filt(freq: number, linear: boolean, order = 2, type = "LinkwitzRiley", q: number | null = null): FilterConfig {
  return {
    filter_type: type, freq_hz: freq, order,
    q, linear_phase: linear, subsonic_protect: null,
  } as FilterConfig;
}

function band(hp: FilterConfig | null, lp: FilterConfig | null, targetEnabled = true): BandState {
  return { targetEnabled, target: { high_pass: hp, low_pass: lp } } as unknown as BandState;
}

describe("ringZone", () => {
  it("covers the measured −80 dB ringing of the steep slopes", () => {
    // Periods to −80 dB, linear-phase LP (export-metrics.ts header).
    expect(ringPeriods(filt(100, true, 8))).toBeGreaterThan(6.4);            // LR8
    expect(ringPeriods(filt(100, true, 8, "Butterworth"))).toBeGreaterThan(5.4);
    expect(ringPeriods(filt(100, true, 4))).toBeGreaterThan(3.6);            // LR4
    expect(ringPeriods(filt(100, true, 4, "Gaussian"))).toBeGreaterThan(2.1);
    expect(ringPeriods(filt(100, true, 2, "Custom", 2))).toBeGreaterThan(ringPeriods(filt(100, true, 2, "Custom", 0.707)));
  });

  it("takes the longest linear-phase filter and ignores min-phase ones", () => {
    const z = ringZone(band(filt(20, false), filt(300, true)));
    expect(z).toMatchObject({ hz: 300, linear: true });
    expect(z.ms).toBeCloseTo((4 / 300) * 1000, 6);
    // LR8 at 2 kHz (10 periods = 5 ms) vs LR2 at 300 Hz (4 periods = 13.3 ms).
    expect(ringZone(band(filt(300, true), filt(2000, true, 8))).hz).toBe(300);
  });

  it("falls back to any filter, then to the working range", () => {
    expect(ringZone(band(filt(80, false), filt(2000, false)))).toMatchObject({ hz: 80, linear: false });
    expect(ringZone(band(null, null))).toMatchObject({ hz: null, ms: 200 });
    expect(ringZone(band(filt(300, true), null, false)).hz).toBeNull();
  });
});

describe("analyzePreRing", () => {
  // 3 dB/ms → the envelope crosses −60 dB at 20 ms before the peak.
  const RATE = 3;
  const TRUE_MS = -PRE_RING_DB / RATE;

  it("measures the ringing at −60 dB when the floor is clean", () => {
    const r = analyzePreRing(synth(1 << 16, RATE), SR, 33);
    expect(r.preRingMs).toBeCloseTo(TRUE_MS, 1);
    expect(r.limited).toBe(false);
    expect(r.floorDb!).toBeLessThanOrEqual(FLOOR_GOOD_DB);
  });

  it("does not count a quiet floor as ringing, but reports it", () => {
    const r = analyzePreRing(synth(1 << 16, RATE, { db: -75, fromMs: 25 }), SR, 33);
    expect(r.preRingMs).toBeCloseTo(TRUE_MS, 1);
    expect(r.floorDb!).toBeGreaterThan(-76);
    expect(r.floorDb!).toBeLessThan(-73);
  });

  it("raises the threshold over a loud floor and flags the value as a lower bound", () => {
    const r = analyzePreRing(synth(1 << 16, RATE, { db: -40, fromMs: 25 }), SR, 33);
    expect(r.floorDb!).toBeGreaterThan(FLOOR_BAD_DB);
    expect(r.limited).toBe(true);
    expect(r.thresholdDb).toBeGreaterThan(r.floorDb!);
    expect(r.preRingMs).toBeLessThan(TRUE_MS);
  });

  it("has no floor to report when the filter is shorter than the ring zone", () => {
    const n = 2048; // 21 ms before the peak at 48 kHz
    const r = analyzePreRing(synth(n, RATE), SR, 33);
    expect(r.floorDb).toBeNull();
    expect(r.limited).toBe(false);
  });

  it("min-phase leading zeros are neither floor nor ringing", () => {
    const h = new Float64Array(1 << 14);
    for (let i = h.length / 2, k = 0; i < h.length; i++, k++) h[i] = Math.exp(-k / 50) * (k === 0 ? 1 : 0.5);
    const r = analyzePreRing(h, SR, 125);
    expect(r.preRingMs).toBe(0);
    expect(r.floorDb).toBe(-150);
  });
});

// Real band (user project "bd 3 - full", Band 2 · 8x-NF): LR2 300 Hz
// linear-phase LP + PEQ boosts at 20/30 Hz, 352.8 kHz, Blackman. Impulses
// dumped from generate_model_fir; test-fixtures/ is gitignored, so the
// case skips on a clean checkout. The old −80 dB metric read 79 → 131 →
// 112 → 6.3 → 6.3 ms over these tap counts; the true −60 dB ringing is
// 5.45 ms.
// The repo has no Node typings (tsc covers src/ as browser code) — load fs
// through a non-literal specifier so it stays untyped.
const FS_MODULE = "node:fs";
const fs: { existsSync(u: URL): boolean; readFileSync(u: URL): Uint8Array } =
  await import(/* @vite-ignore */ FS_MODULE);
const HERE = import.meta.url; // a variable: Vite rewrites `new URL(literal, import.meta.url)` as an asset
const fixture = (t: number) => new URL(`../../../test-fixtures/prering/bd3_8xnf_352k8_${t}.f32`, HERE);
const TAPS = [65536, 131072, 262144, 524288, 1048576];
const haveFixture = TAPS.every((t) => fs.existsSync(fixture(t)));

describe.skipIf(!haveFixture)("analyzePreRing on a real band across tap counts", () => {
  const zone = ringZone(band(null, filt(300, true)));
  const run = (t: number) => {
    const b = fs.readFileSync(fixture(t));
    const h = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    return analyzePreRing(h, 352_800, zone.ms);
  };

  it("long filters agree on the ringing and show a clean floor", () => {
    const long = [run(524288), run(1048576)];
    for (const r of long) {
      expect(r.limited).toBe(false);
      expect(r.floorDb!).toBeLessThanOrEqual(FLOOR_GOOD_DB);
    }
    expect(Math.abs(long[0].preRingMs - long[1].preRingMs)).toBeLessThan(0.1);
  });

  it("short filters show the floor instead of inflating the pre-ring", () => {
    const ref = run(1048576).preRingMs;
    for (const t of [65536, 131072, 262144]) {
      const r = run(t);
      expect(r.limited, `${t}`).toBe(true);
      expect(r.preRingMs, `${t}`).toBeLessThanOrEqual(ref);
    }
    expect(run(65536).floorDb!).toBeGreaterThan(FLOOR_BAD_DB);
  });

  it("floor falls as taps grow until the tails fit", () => {
    const floors = [65536, 131072, 262144, 524288].map((t) => run(t).floorDb!);
    for (let i = 1; i < floors.length; i++) expect(floors[i]).toBeLessThan(floors[i - 1]);
  });
});
