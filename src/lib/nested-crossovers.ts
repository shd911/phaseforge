/**
 * b141.134: nested crossovers. Any binary tree of pairs that sum to 1 sums to
 * exactly 1 itself, so a band may also carry the LPs/HPs of other crossovers
 * and stop leaking past its neighbour. The tree branches from the centre, the
 * way a two-way split nests: 70/220/800/2500 Hz gives L70·L220·L800,
 * H70·L220·L800, H220·L800, H800·L2500, H800·H2500.
 *
 * Only a linear-phase LR or Gaussian pair qualifies: other families do not sum
 * to 1, a min-phase section would turn the phase of the others. «1 − H» bands
 * are HP + LP − 1 as a whole, not a product a section could join.
 *
 * Rust knows nothing of the tree: a section just multiplies in `nested`.
 */
import { cloneFilterConfig, type FilterConfig } from "./types";

interface BandCrossovers {
  linkedToNext: boolean;
  target: { high_pass: FilterConfig | null; low_pass: FilterConfig | null };
}

export interface NestedSections {
  hp: FilterConfig[];
  lp: FilterConfig[];
}

function crossoverNestable(band: BandCrossovers, next: BandCrossovers | undefined): boolean {
  const [lp, hp] = [band.target.low_pass, next?.target.high_pass];
  if (!band.linkedToNext || !lp || !hp) return false;
  if (!lp.linear_phase || !hp.linear_phase || lp.complementary || hp.complementary) return false;
  if (lp.filter_type !== "LinkwitzRiley" && lp.filter_type !== "Gaussian") return false;
  return lp.filter_type === hp.filter_type && lp.order === hp.order && lp.freq_hz === hp.freq_hz
    && (lp.shape ?? null) === (hp.shape ?? null);
}

/** Runs of nestable crossovers `[first, last]` (crossover i is between bands i
 *  and i + 1). A lone crossover has nothing to nest with, so it is not a run. */
function nestableRuns(bands: readonly BandCrossovers[]): [number, number][] {
  const runs: [number, number][] = [];
  let start = -1;
  for (let i = 0; i <= bands.length - 1; i++) {
    const ok = i < bands.length - 1 && crossoverNestable(bands[i], bands[i + 1]);
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) {
      if (i - start >= 2) runs.push([start, i - 1]);
      start = -1;
    }
  }
  return runs;
}

export function hasNestableRun(bands: readonly BandCrossovers[]): boolean {
  return nestableRuns(bands).length > 0;
}

/** A nested copy: the section's own response only. */
function nestedCopy(f: FilterConfig): FilterConfig {
  const { nested: _nested, ...own } = cloneFilterConfig(f);
  return { ...own, subsonic_protect: null };
}

/** The sections each band carries on top of its own HP and LP, highest LP
 *  and lowest HP last. Empty lists when `enabled` is off. */
export function nestedCrossoverSections(bands: readonly BandCrossovers[], enabled: boolean): NestedSections[] {
  const out: NestedSections[] = bands.map(() => ({ hp: [], lp: [] }));
  if (!enabled) return out;
  // Crossovers lo..hi−1 split bands lo..hi.
  const split = (lo: number, hi: number) => {
    if (lo >= hi) return;
    const root = Math.floor((lo + hi) / 2);
    const lp = bands[root].target.low_pass!;
    const hp = bands[root + 1].target.high_pass!;
    for (let k = lo; k < root; k++) out[k].lp.push(nestedCopy(lp));
    for (let k = root + 2; k <= hi; k++) out[k].hp.push(nestedCopy(hp));
    split(lo, root);
    split(root + 1, hi);
  };
  for (const [first, last] of nestableRuns(bands)) split(first, last + 1);
  for (const s of out) {
    s.lp.sort((a, b) => a.freq_hz - b.freq_hz);
    s.hp.sort((a, b) => a.freq_hz - b.freq_hz);
  }
  return out;
}
