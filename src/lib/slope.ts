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

export function availableSlopes(filterType: string): number[] {
  switch (filterType) {
    case "LinkwitzRiley": return LR_SLOPES;
    case "Butterworth":
    case "Bessel":
    case "Custom": return STD_SLOPES;
    default: return [];
  }
}
