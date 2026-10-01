/** b141.68 (audit stage 2 P2): cache returns clones, but the FIR impulse is shared and frozen. */
import { describe, it, expect } from "vitest";
import { memoEval } from "../band-evaluator/cache";

describe("memoEval impulse sharing", () => {
  it("envelope is a fresh clone, impulse is the same frozen array", async () => {
    const big = Array.from({ length: 8192 }, (_, i) => i);
    const compute = async () => ({ freq: [1, 2, 3], fir: { impulse: big, taps: 8192 } });
    const a = await memoEval("k-shared", compute);
    const b = await memoEval("k-shared", compute);
    expect(a).not.toBe(b);
    a.freq[0] = 99;
    expect(b.freq[0]).toBe(1);
    expect(a.fir.impulse).toBe(b.fir.impulse);
    expect(Object.isFrozen(a.fir.impulse)).toBe(true);
  });
});
