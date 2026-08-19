// b141.19 (audit): the WAV export called evaluateBandFull without a grid, so
// the evaluator fell back to the band's measurement grid while the Export tab
// previewed on its own 5–40 kHz one. Below the measurement's first bin the
// Rust-side interpolation clamps to the boundary value, so the target
// roll-off became a plateau: a Gaussian HP 80 min-phase band measured
// -27.5 dB of DC gain in the exported impulse against -51.3 dB in the
// previewed one — 52 dB apart relative to peak. The user shipped a file they
// had never seen.

import { describe, it, expect, vi, beforeEach } from "vitest";

// vi.mock is hoisted above imports, so the spy has to be created inside it.
vi.mock("../band-evaluator", () => ({
  evaluateBandFull: vi.fn(async () => ({
    fir: { impulse: [0, 1, 0, 0], wavDelaySamples: 2 },
  })),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ save: vi.fn(async () => null) }));

import { evaluateBandFull } from "../band-evaluator";
import { buildFirGrid } from "../band-evaluator/grid";
import { exportBandWav } from "../fir-export";
import type { BandState } from "../../stores/bands";

describe("FIR export grid (b141.19)", () => {
  const evalMock = vi.mocked(evaluateBandFull);
  beforeEach(() => evalMock.mockClear());

  it("buildFirGrid spans 5 Hz–40 kHz in 512 log-spaced points", () => {
    const g = buildFirGrid();
    expect(g).toHaveLength(512);
    expect(g[0]).toBeCloseTo(5, 6);
    expect(g[511]).toBeCloseTo(40_000, 6);
    const r1 = g[1] / g[0], r2 = g[400] / g[399];
    expect(r1).toBeCloseTo(r2, 9);
  });

  it("the WAV export evaluates on that grid, not the measurement's", async () => {
    const band = { name: "Woofer", measurement: { freq: [20, 100, 20_000] } } as unknown as BandState;
    await exportBandWav(band);
    expect(evalMock).toHaveBeenCalledTimes(1);
    const req = evalMock.mock.calls[0][0] as unknown as { freq?: number[] };
    expect(req.freq).toEqual(buildFirGrid());
  });
});
