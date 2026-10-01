/** b141.70: test stand-in for the Rust `compute_target_hilbert_phase` — the
 *  former TS `reconstructTargetPhase` term selection, with the test file's own
 *  min-phase mock on the FIR-span grid. Keeps the b139 snapshots meaningful. */
import type { FilterConfig } from "../../types";
import { hasActiveSubsonicProtect } from "../../types";
import { gaussianFilterMagDb, isGaussianMinPhase, subsonicMagDb } from "../../plot-helpers";
import { buildLogGrid, interpOnGrid } from "../../band-evaluator/grid";

export function hilbertTermsMock(
  args: { freq: number[]; highPass: FilterConfig | null; lowPass: FilterConfig | null; sampleRate: number },
  minPhase: (freq: number[], magnitude: number[]) => number[],
): number[] {
  const { freq, highPass: hp, lowPass: lp, sampleRate } = args;
  const g = buildLogGrid(1024, 5, (sampleRate / 2) * 0.95);
  const terms: number[][] = [];
  const add = (mag: number[]) =>
    terms.push(interpOnGrid(g, minPhase(g, mag), freq, { logSpace: true, outside: "clamp" }) as number[]);
  if (isGaussianMinPhase(hp)) {
    let m = gaussianFilterMagDb(g, hp!, false);
    if (hasActiveSubsonicProtect(hp)) {
      const s = subsonicMagDb(g, hp!.freq_hz / 8);
      m = m.map((v, i) => v + s[i]);
    }
    add(m);
  } else if (hasActiveSubsonicProtect(hp) && hp!.linear_phase === true) {
    add(subsonicMagDb(g, hp!.freq_hz / 8));
  }
  if (isGaussianMinPhase(lp)) add(gaussianFilterMagDb(g, lp!, true));
  return freq.map((_, i) => terms.reduce((a, t) => a + t[i], 0));
}
