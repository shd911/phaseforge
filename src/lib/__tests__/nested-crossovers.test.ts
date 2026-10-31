import { describe, expect, it } from "vitest";
import { hasNestableRun, nestedCrossoverSections } from "../nested-crossovers";
import type { FilterConfig } from "../types";

const sec = (freq_hz: number, over: Partial<FilterConfig> = {}): FilterConfig => ({
  filter_type: "LinkwitzRiley", order: 1, freq_hz, shape: null, linear_phase: true, q: null, ...over,
});

function chain(xo: number[], over: (i: number) => Partial<FilterConfig> = () => ({})) {
  return Array.from({ length: xo.length + 1 }, (_, i) => ({
    linkedToNext: i < xo.length,
    target: {
      high_pass: i > 0 ? sec(xo[i - 1], over(i - 1)) : null,
      low_pass: i < xo.length ? sec(xo[i], over(i)) : null,
    },
  }));
}

const freqs = (v: FilterConfig[]) => v.map((f) => f.freq_hz);

/** Sum of the tree at `f` with LP = 1/(1+(f/fc)²), HP = 1 − LP: LR1 on zero phase. */
function sumAt(bands: ReturnType<typeof chain>, f: number): number {
  const lp = (fc: number) => 1 / (1 + (f / fc) ** 2);
  const nested = nestedCrossoverSections(bands, true);
  let sum = 0;
  bands.forEach((b, i) => {
    let v = 1;
    if (b.target.high_pass) v *= 1 - lp(b.target.high_pass.freq_hz);
    if (b.target.low_pass) v *= lp(b.target.low_pass.freq_hz);
    for (const h of nested[i].hp) v *= 1 - lp(h.freq_hz);
    for (const l of nested[i].lp) v *= lp(l.freq_hz);
    sum += v;
  });
  return sum;
}

describe("nested crossovers", () => {
  it("branches from the centre", () => {
    const s = nestedCrossoverSections(chain([70, 220, 800, 2500]), true);
    expect(s.map((b) => freqs(b.lp))).toEqual([[220, 800], [800], [], [], []]);
    expect(s.map((b) => freqs(b.hp))).toEqual([[], [], [], [], [800]]);
  });

  it("sums to exactly 1 for any number of bands", () => {
    for (const xo of [[100, 1000], [70, 220, 800], [70, 220, 800, 2500], [50, 100, 200, 400, 800, 1600, 3200]]) {
      for (const f of [20, 80, 300, 1000, 5000, 20000]) {
        expect(sumAt(chain(xo), f)).toBeCloseTo(1, 12);
      }
    }
  });

  it("is off by default and on two-way systems", () => {
    expect(nestedCrossoverSections(chain([70, 220, 800]), false).every((b) => !b.lp.length && !b.hp.length)).toBe(true);
    expect(hasNestableRun(chain([500]))).toBe(false);
    expect(nestedCrossoverSections(chain([500]), true).every((b) => !b.lp.length && !b.hp.length)).toBe(true);
    expect(hasNestableRun(chain([70, 220]))).toBe(true);
  });

  it("needs a linear LR or Gaussian pair of the same shape", () => {
    for (const bad of [
      { linear_phase: false }, { filter_type: "Butterworth" as const }, { filter_type: "Bessel" as const },
      { filter_type: "Custom" as const, q: 0.5 }, { complementary: true },
    ]) {
      expect(hasNestableRun(chain([70, 220], () => bad))).toBe(false);
    }
    const gaussian = chain([70, 220], () => ({ filter_type: "Gaussian" as const, shape: 1 }));
    expect(hasNestableRun(gaussian)).toBe(true);
    const mismatch = chain([70, 220]);
    mismatch[1].target.high_pass!.freq_hz = 75;
    expect(hasNestableRun(mismatch)).toBe(false);
  });

  it("a broken link splits the chain into separate trees", () => {
    const bands = chain([70, 220, 800, 2500, 8000]);
    bands[2].linkedToNext = false; // 800 Hz is not part of any tree
    const s = nestedCrossoverSections(bands, true);
    expect(s.map((b) => freqs(b.lp))).toEqual([[220], [], [], [8000], [], []]);
    expect(s.map((b) => freqs(b.hp))).toEqual([[], [], [], [], [], []]);
  });

  it("a nested copy carries neither its own list nor the subsonic", () => {
    const bands = chain([70, 220, 800], (i) => ({ subsonic_protect: i === 1 ? true : null }));
    bands[0].target.low_pass!.nested = [sec(5000)];
    const s = nestedCrossoverSections(bands, true);
    for (const n of [...s.flatMap((b) => b.lp), ...s.flatMap((b) => b.hp)]) {
      expect(n.nested).toBeUndefined();
      expect(n.subsonic_protect).toBeNull();
    }
  });
});
