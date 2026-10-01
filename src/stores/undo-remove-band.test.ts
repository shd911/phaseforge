/** b141.66 (audit stage 2 B2): undo of a band deletion restores its measurement. */
import { describe, it, expect } from "vitest";
import { appState, addBand, removeBand, setBandMeasurement } from "./bands";
import { undo } from "./history";
import "./peq-optimize"; // registers the history hooks

describe("undo remove band", () => {
  it("brings the measurement back", () => {
    addBand();
    addBand();
    const victim = appState.bands[appState.bands.length - 1];
    const freq = [20, 200, 2000, 20000];
    setBandMeasurement(victim.id, {
      name: "m", source_path: null, sample_rate: 48000, freq,
      magnitude: [80, 81, 82, 83], phase: [0, -10, -20, -30],
      metadata: { date: null, mic: null, notes: null, smoothing: null },
    } as any);
    removeBand(victim.id);
    expect(appState.bands.find((b) => b.id === victim.id)).toBeUndefined();
    undo();
    const back = appState.bands.find((b) => b.id === victim.id);
    expect(back).toBeDefined();
    expect(back!.measurement?.magnitude).toEqual([80, 81, 82, 83]);
    expect(back!.settings).not.toBeNull();
  });
});

import { renameBand, uniqueBandName } from "./bands";

describe("unique band names (b141.66)", () => {
  it("rename to a taken name gets a suffix", () => {
    addBand(); addBand();
    const [a, b] = appState.bands.slice(-2);
    renameBand(a.id, "Woofer");
    renameBand(b.id, "Woofer");
    expect(appState.bands.find((x) => x.id === b.id)!.name).toBe("Woofer (2)");
  });
  it("helper", () => {
    expect(uniqueBandName("X", [{ id: "1", name: "X" }, { id: "2", name: "X (2)" }], "3")).toBe("X (3)");
    expect(uniqueBandName("X", [{ id: "1", name: "X" }], "1")).toBe("X");
  });
});
