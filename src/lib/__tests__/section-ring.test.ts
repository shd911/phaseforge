import { describe, it, expect } from "vitest";
import { ringLabel, ringSymmetric, ringTooLong, preRingMargin, preRingLabel, type SectionRing } from "../section-ring";

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
    expect(ringLabel(gaussLinSub, true)).toBe("Звон −60/−100 дБ: до пика 8.2 / 169.9, после 53 / 400 мс");
  });

  it("min-phase: one pair, the tail after the peak", () => {
    const r: SectionRing = { pre60Ms: 0.4, pre100Ms: 0.9, post60Ms: 12, post100Ms: 30, lowerBound: true };
    expect(ringLabel(r, false)).toBe("Звон −60/−100 дБ: ≥ 12 / 30 мс");
  });

  it("a pair shares one precision across 10 ms", () => {
    const r: SectionRing = { pre60Ms: 0, pre100Ms: 0, post60Ms: 9.8, post100Ms: 12.04, lowerBound: false };
    expect(ringLabel(r, false)).toBe("Звон −60/−100 дБ: 9.8 / 12.0 мс");
  });

  it("min-phase warning counts the rise to the peak, linear each side alone", () => {
    const r: SectionRing = { pre60Ms: 1, pre100Ms: 3, post60Ms: 20, post100Ms: 40, lowerBound: false };
    expect(ringTooLong(r, false, 42)).toBe(true);   // 3 + 40 > 42
    expect(ringTooLong(r, false, 43.5)).toBe(false);
    expect(ringTooLong(r, true, 42)).toBe(false);   // max(3, 40) fits each half
    expect(ringTooLong(gaussLinSub, true, 341)).toBe(true); // 64K @ 48 kHz: 341 ms
  });
});

describe("pre-ring audibility (b141.153)", () => {
  // relDb = M − L for LR 48 dB/oct linear at 300 Hz (Rust prering_audibility).
  const lr48: SectionRing = { ...bw8Lin, preLobeDb: -17.43, audibility: { relDb: [-82.3, -88.6, -107.4], channelHz: 345 } };
  it("adds the listening level and colours by the 1 ms and 5 ms windows", () => {
    expect(preRingMargin(lr48, 105)!.verdict).toBe("maybe");   // M1 +22.7, M5 −2.4
    expect(preRingMargin(lr48, 110)!.verdict).toBe("loud");    // M5 +2.6
    expect(preRingMargin(lr48, 76)!.verdict).toBe("ok");       // M1 −6.3
    expect(preRingLabel(preRingMargin(lr48, 105)!)).toBe("Предзвон на грани: +16.4 дБ к порогу (2 мс)");
  });
  it("flags a verdict decided below 300 Hz as unvalidated", () => {
    const lf: SectionRing = { ...lr48, audibility: { relDb: [-75, -78, -85], channelHz: 136 } };
    const m = preRingMargin(lf, 105)!;
    expect(m.verdict).toBe("loud");
    expect(m.unvalidated).toBe(true);
    expect(preRingLabel(m)).toBe("Предзвон слышен: +27.0 дБ к порогу (2 мс) *");
  });
  it("min phase has no verdict, a numerically clean pre-ring prints < −60", () => {
    expect(preRingMargin({ ...bw8Lin, audibility: null }, 105)).toBeNull();
    const clean = preRingMargin({ ...bw8Lin, audibility: { relDb: [-200, -200, -200], channelHz: 3000 } }, 105)!;
    expect(clean.verdict).toBe("ok");
    expect(preRingLabel(clean)).toBe("Предзвон не слышен: < −60 дБ к порогу (2 мс)");
  });
});
