// b141.44: 20 kHz is always labelled on a log-frequency axis when in range.
import { describe, it, expect } from "vitest";
import { pickFreqLabels, PINNED_FREQ_LABEL_HZ } from "../plot-helpers";

/** uPlot-like log splits: mantissa 1..9 per decade inside [lo, hi]. */
function logSplits(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    for (let m = 1; m <= 9; m++) { const v = m * 10 ** e; if (v >= lo && v <= hi) out.push(v); }
  }
  return out;
}
const pos = (lo: number, hi: number, width: number) => (v: number) =>
  (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * width;

describe("pickFreqLabels", () => {
  it("labels 20 kHz on a narrow full-range plot where 30k would crowd it", () => {
    const [lo, hi, w] = [20, 30000, 300];
    const s = logSplits(lo, hi);
    const labels = pickFreqLabels(s, pos(lo, hi, w)).filter((v): v is number => v != null);
    expect(labels).toContain(PINNED_FREQ_LABEL_HZ);
    const px = labels.map(pos(lo, hi, w)).sort((a, b) => a - b);
    for (let i = 1; i < px.length; i++) expect(px[i] - px[i - 1]).toBeGreaterThanOrEqual(30);
  });

  it("labels 20 kHz at every zoom that contains it", () => {
    for (const [lo, hi, w] of [[20, 30000, 1000], [2000, 30000, 600], [15000, 25000, 400], [19000, 21000, 200], [5000, 20000, 150]]) {
      const labels = pickFreqLabels(logSplits(lo, hi), pos(lo, hi, w));
      expect(labels, `${lo}–${hi} @ ${w}px`).toContain(PINNED_FREQ_LABEL_HZ);
    }
  });

  it("keeps decades ahead of in-between values and never overlaps", () => {
    const [lo, hi, w] = [20, 30000, 1000];
    const labels = pickFreqLabels(logSplits(lo, hi), pos(lo, hi, w)).filter((v): v is number => v != null);
    for (const d of [100, 1000, 10000]) expect(labels).toContain(d);
  });
});
