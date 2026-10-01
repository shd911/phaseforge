/**
 * b141.47 (audit 2026-10-01 H7): auto-align must align the polarity that
 * plays. Analytic LR responses on a log grid; no backend.
 */
import { describe, it, expect } from "vitest";
import { alignDelays, type AlignBand } from "../auto-align";

const freq = Array.from({ length: 400 }, (_, i) => 20 * Math.pow(1000, i / 399));

/** Butterworth-1 LP/HP squared (LR2, 12 dB/oct) as dB/deg. */
function lr2(f: number, fc: number, hp: boolean): { db: number; deg: number } {
  const w = f / fc;
  // BU1 LP = 1/(1+jw); HP = jw/(1+jw)
  const magLp = 1 / Math.sqrt(1 + w * w);
  const phLp = -Math.atan(w);
  const mag = hp ? w * magLp : magLp;
  const ph = hp ? Math.PI / 2 + phLp : phLp;
  return { db: 40 * Math.log10(mag), deg: (2 * ph * 180) / Math.PI };
}

function band(hp: boolean, fc: number, sign: 1 | -1, extraDelayS = 0): AlignBand {
  const mag: number[] = [];
  const ph: number[] = [];
  for (const f of freq) {
    const r = lr2(f, fc, hp);
    mag.push(r.db);
    ph.push(r.deg - 360 * f * extraDelayS);
  }
  return { mag, ph, sign, hpHz: hp ? fc : null, lpHz: hp ? null : fc };
}

function sumDbAt(bands: AlignBand[], delays: number[], f0: number): number {
  const j = freq.findIndex((f) => f >= f0);
  let re = 0, im = 0;
  bands.forEach((b, k) => {
    const a = b.sign * Math.pow(10, b.mag[j] / 20);
    const p = ((b.ph[j] - 360 * freq[j] * delays[k]) * Math.PI) / 180;
    re += a * Math.cos(p);
    im += a * Math.sin(p);
  });
  return 20 * Math.log10(Math.hypot(re, im));
}

describe("alignDelays", () => {
  it("LR2 with the inverted tweeter is already aligned — keeps ~0 delay", () => {
    const bands = [band(true, 1000, -1), band(false, 1000, 1)];
    const d = alignDelays(freq, bands);
    expect(Math.abs(d[0] - d[1])).toBeLessThan(20e-6);
    expect(sumDbAt(bands, d, 1000)).toBeGreaterThan(-0.5);
  });

  it("recovers a 0.3 ms woofer lead to within 5 µs", () => {
    // Woofer arrives 0.3 ms early → it must be delayed by 0.3 ms.
    const tw = band(true, 1000, -1, 0.0003);
    const wf = band(false, 1000, 1, 0);
    const d = alignDelays(freq, [tw, wf]);
    expect(Math.abs(d[1] - d[0] - 0.0003)).toBeLessThan(5e-6);
  });
});

import { pickScanPeak } from "../auto-align";

describe("pickScanPeak (b141.60)", () => {
  // The scan recorded on VPV2 (156–311 Hz pair), ms → coherence.
  const ms = [-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0];
  const amp = [0.928, 0.912, 0.863, 0.784, 0.679, 0.559, 0.441, 0.361, 0.394, 0.491, 0.606, 0.718, 0.811, 0.878, 0.914, 0.916, 0.887, 0.832, 0.758, 0.680, 0.611];
  it("an edge maximum loses to the interior one", () => {
    expect(pickScanPeak(ms, amp)).toBeCloseTo(1.5, 6);
  });
  it("ties go to the smaller |delay|", () => {
    // Two maxima one "period" apart, the farther one 0.2 % higher → nearer wins.
    expect(pickScanPeak([-3, -2.5, -2, 0, 0.5, 1, 2], [0.5, 0.902, 0.5, 0.4, 0.9, 0.5, 0.4])).toBe(0.5);
    // …but not when the farther one is clearly better.
    expect(pickScanPeak([-3, -2.5, -2, 0, 0.5, 1, 2], [0.5, 0.95, 0.5, 0.4, 0.9, 0.5, 0.4])).toBe(-2.5);
  });
  it("monotonic scan falls back to the edge", () => {
    expect(pickScanPeak([-1, 0, 1], [0.1, 0.2, 0.3])).toBe(1);
  });
});
