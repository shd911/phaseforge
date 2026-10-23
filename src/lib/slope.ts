// b140.7.13: Slope ↔ Order conversion. UI shows actual slope in dB/oct;
// the storage / wire format keeps `filter.order` (1..8). LR is treated
// per the PhaseForge convention `(BU-N)² = 2N effective order = 12N dB/oct`
// — matches `target/mod.rs::filter_lp_response::LinkwitzRiley` which
// doubles both magnitude (dB) and phase relative to BU-N.

const LR_SLOPES = [12, 24, 36, 48, 60, 72, 84, 96];
const STD_SLOPES = [6, 12, 18, 24, 30, 36, 42, 48];

export function orderToSlope(filterType: string, order: number): number {
  switch (filterType) {
    case "LinkwitzRiley": return order * 12;
    case "Butterworth":
    case "Bessel":
    case "Custom": return order * 6;
    default: return 0;
  }
}

export function slopeToOrder(filterType: string, slope: number): number {
  switch (filterType) {
    case "LinkwitzRiley": return Math.round(slope / 12);
    case "Butterworth":
    case "Bessel":
    case "Custom": return Math.round(slope / 6);
    default: return 1;
  }
}

// b141.105 (audit 2026-10-23 §5.7): the filter block and the Crossover window
// showed the same filter with different type names, Q limits (0.1–5 vs
// 0.1–20) and hints, and the window did not clamp M or Q at all.
export const FILTER_TYPE_ORDER = ["LinkwitzRiley", "Butterworth", "Bessel", "Gaussian", "Custom"] as const;
export const FILTER_TYPE_LABEL: Record<string, string> = {
  LinkwitzRiley: "Linkwitz-Riley",
  Butterworth: "Butterworth",
  Bessel: "Bessel",
  Gaussian: "Gaussian",
  Custom: "Произвольный Q",
};
/** Q of a «Произвольный Q» section. */
export const CUSTOM_Q_MIN = 0.1;
export const CUSTOM_Q_MAX = 5;
/** Gaussian M coefficient. */
export const GAUSS_M_MIN = 0.5;
export const GAUSS_M_MAX = 10;
export const CUSTOM_Q_HINT = "0.5 = LR · 0.577 = Bessel · 0.707 = BW · > 0.707 — резонанс";

export const clampNum = (v: number, min: number, max: number): number => Math.min(max, Math.max(min, v));

export function availableSlopes(filterType: string): number[] {
  switch (filterType) {
    case "LinkwitzRiley": return LR_SLOPES;
    case "Butterworth":
    case "Bessel":
    case "Custom": return STD_SLOPES;
    default: return [];
  }
}
