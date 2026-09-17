// b141.44–45: frequency X axes — ticks on round frequencies only, 20 kHz
// always labelled when visible, labels never touch.
import { describe, it, expect } from "vitest";
import { pickFreqLabels, niceFreqSplits, fmtFreqTick, PINNED_FREQ_LABEL_HZ } from "../plot-helpers";

const pos = (lo: number, hi: number, width: number) => (v: number) =>
  (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * width;

describe("niceFreqSplits", () => {
  // The user's screenshot: after a scroll the scale started at 284.8 Hz and
  // uPlot put ticks at 284.8, 385, 1.085k, 3.255k, 21.7k, 97.65k.
  it("stays on round frequencies when the scale minimum is not round", () => {
    const s = niceFreqSplits(284.8, 97650);
    expect(s[0]).toBe(300);
    expect(s).toContain(20000);
    for (const v of s) {
      const m = v / 10 ** Math.floor(Math.log10(v));
      expect(Math.abs(m - Math.round(m)), `${v}`).toBeLessThan(1e-9);
    }
  });

  it("falls back to a round linear step on a deep zoom", () => {
    expect(niceFreqSplits(19500, 20500)).toEqual([19500, 19600, 19700, 19800, 19900, 20000, 20100, 20200, 20300, 20400, 20500]);
  });
});

describe("fmtFreqTick", () => {
  it("formats without float noise", () => {
    expect([20, 300, 1000, 1500, 20000, 19600, 5].map(fmtFreqTick)).toEqual(["20", "300", "1k", "1.5k", "20k", "19.6k", "5"]);
  });
});

describe("pickFreqLabels", () => {
  const widthOf = (v: number) => fmtFreqTick(v).length * 7;
  const noTouch = (labels: number[], p: (v: number) => number) => {
    const xs = labels.map((v) => ({ px: p(v), half: widthOf(v) / 2 })).sort((a, b) => a.px - b.px);
    for (let i = 1; i < xs.length; i++) expect(xs[i].px - xs[i - 1].px).toBeGreaterThanOrEqual(xs[i].half + xs[i - 1].half + 8);
  };

  it("labels 20 kHz at every zoom that contains it, without touching labels", () => {
    for (const [lo, hi, w] of [[20, 30000, 300], [20, 30000, 1800], [284.8, 97650, 1700], [2000, 30000, 600], [15000, 25000, 400], [19000, 21000, 300]]) {
      const p = pos(lo, hi, w);
      const labels = pickFreqLabels(niceFreqSplits(lo, hi), p).filter((v): v is number => v != null);
      expect(labels, `${lo}–${hi} @ ${w}px`).toContain(PINNED_FREQ_LABEL_HZ);
      noTouch(labels, p);
    }
  });

  it("keeps decades ahead of in-between values", () => {
    const p = pos(20, 30000, 1000);
    const labels = pickFreqLabels(niceFreqSplits(20, 30000), p);
    for (const d of [100, 1000, 10000]) expect(labels).toContain(d);
  });
});
