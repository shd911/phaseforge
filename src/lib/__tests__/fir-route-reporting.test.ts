/**
 * b141.36: the Export tab disables the Win dropdown when the band is built by
 * the analytical IIR cascade, because that path never reads `window` — it runs
 * a delta through the biquads and applies its own fixed tail taper.
 *
 * The UI must not re-derive that condition (a second predicate would drift
 * from `pick_fir_route`), so `dispatchFirInvoke` reports the route it took.
 * The invariant worth pinning is exactly that: the reported route matches the
 * Tauri command that actually ran.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { FilterConfig } from "../types";

const calls: string[] = [];
let routeAnswer = "Iir";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    calls.push(cmd);
    if (cmd === "pick_fir_route") return routeAnswer;
    if (cmd === "generate_model_fir_iir" || cmd === "generate_model_fir") {
      const taps = args.config.taps as number;
      const n = (args.freq as number[]).length;
      return {
        impulse: new Array(taps).fill(0),
        realized_mag: new Array(n).fill(0),
        realized_phase: new Array(n).fill(0),
        taps,
        sample_rate: args.config.sample_rate,
        norm_db: 0,
        causality: 1,
        wav_delay_samples: taps / 2,
      };
    }
    throw new Error(`Unmocked command: ${cmd}`);
  }),
}));

import { dispatchFirInvoke } from "../band-evaluator/route";

const LR4: FilterConfig = {
  filter_type: "LinkwitzRiley", order: 4, freq_hz: 200,
  shape: null, linear_phase: false, q: null, subsonic_protect: null,
};

const cfg = {
  taps: 4096, sampleRate: 48000, window: "Blackman",
  maxBoostDb: 18, noiseFloorDb: -60, iterations: 0, freqWeighting: false,
  narrowbandLimit: false, nbSmoothingOct: 0.333, nbMaxExcessDb: 6,
};

async function run() {
  const freq = [20, 200, 2000, 20000];
  const flat = [0, 0, 0, 0];
  return dispatchFirInvoke(
    null, LR4, null, null, 0, [], false, null,
    freq, flat, [], flat, cfg,
  );
}

describe("dispatchFirInvoke route reporting", () => {
  beforeEach(() => { calls.length = 0; });

  it("reports 'iir' exactly when the IIR command ran", async () => {
    routeAnswer = "Iir";
    const out = await run();
    expect(out.route).toBe("iir");
    expect(calls).toContain("generate_model_fir_iir");
    expect(calls).not.toContain("generate_model_fir");
  });

  it("reports 'cepstral' exactly when the cepstral command ran", async () => {
    routeAnswer = "Cepstral";
    const out = await run();
    expect(out.route).toBe("cepstral");
    expect(calls).toContain("generate_model_fir");
    expect(calls).not.toContain("generate_model_fir_iir");
  });

  it("still carries the Rust payload through untouched", async () => {
    routeAnswer = "Iir";
    const out = await run();
    expect(out.taps).toBe(4096);
    expect(out.sample_rate).toBe(48000);
    expect(out.wav_delay_samples).toBe(2048);
    expect(out.impulse).toHaveLength(4096);
  });

  it("passes the window through to Rust on both routes — the IIR path is the side that ignores it", async () => {
    // The dropdown being disabled is a UI affordance, not a payload change:
    // the config still carries `window`, so nothing downstream has to change
    // if the cascade ever grows a use for it.
    for (const answer of ["Iir", "Cepstral"]) {
      routeAnswer = answer;
      calls.length = 0;
      const { invoke } = await import("@tauri-apps/api/core");
      await run();
      const mocked = invoke as unknown as { mock: { calls: any[][] } };
      const gen = mocked.mock.calls.filter(([c]) => c !== "pick_fir_route").pop();
      expect(gen?.[1].config.window).toBe("Blackman");
    }
  });
});
