import { describe, it, expect } from "vitest";
import { ringLabel, ringSymmetric, ringTooLong, type SectionRing } from "../section-ring";

// Measured by Rust section_ring at 48 kHz (b141.115 audit probe).
const gaussLinSub: SectionRing = { pre60Ms: 8.2, pre100Ms: 169.9, post60Ms: 53.0, post100Ms: 399.6, lowerBound: false };
const bw8Lin: SectionRing = { pre60Ms: 202.1, pre100Ms: 380.1, post60Ms: 202.1, post100Ms: 380.1, lowerBound: false };

describe("section ring line (b141.115)", () => {
  it("a symmetric linear section keeps one pair «до и после пика»", () => {
    expect(ringSymmetric(bw8Lin)).toBe(true);
    expect(ringLabel(bw8Lin, true)).toBe("Звон −60/−100 дБ: до и после пика 202 / 380 мс");
  });

  it("linear Gaussian + subsonic shows before and after apart", () => {
    expect(ringSymmetric(gaussLinSub)).toBe(false);
    expect(ringLabel(gaussLinSub, true)).toBe("Звон −60/−100 дБ: до пика 8.2 / 170, после 53 / 400 мс");
  });

  it("min-phase: one pair, the tail after the peak", () => {
    const r: SectionRing = { pre60Ms: 0.4, pre100Ms: 0.9, post60Ms: 12, post100Ms: 30, lowerBound: true };
    expect(ringLabel(r, false)).toBe("Звон −60/−100 дБ: ≥ 12 / 30 мс");
  });

  it("min-phase warning counts the rise to the peak, linear each side alone", () => {
    const r: SectionRing = { pre60Ms: 1, pre100Ms: 3, post60Ms: 20, post100Ms: 40, lowerBound: false };
    expect(ringTooLong(r, false, 42)).toBe(true);   // 3 + 40 > 42
    expect(ringTooLong(r, false, 43.5)).toBe(false);
    expect(ringTooLong(r, true, 42)).toBe(false);   // max(3, 40) fits each half
    expect(ringTooLong(gaussLinSub, true, 341)).toBe(true); // 64K @ 48 kHz: 341 ms
  });
});
