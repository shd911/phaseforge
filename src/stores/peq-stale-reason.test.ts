/** b141.67 (audit stage 2 B7): tilt/shelf edits make the PEQ stale, with a reason. */
import { describe, it, expect } from "vitest";
import { appState, addBand, setBandPeqBands, setBandPeqOptimizedTarget, setBandTilt } from "./bands";
import { captureOptimizedTarget, peqStaleReason } from "./peq-optimize";

describe("peqStaleReason", () => {
  it("tilt change → stale, named", () => {
    addBand();
    const b = appState.bands[appState.bands.length - 1];
    setBandPeqBands(b.id, [{ freq_hz: 100, gain_db: -3, q: 2, enabled: true, filter_type: "Peaking" }]);
    setBandPeqOptimizedTarget(b.id, captureOptimizedTarget(b));
    expect(peqStaleReason(appState.bands.find((x) => x.id === b.id)!)).toBeNull();
    setBandTilt(b.id, -1);
    expect(peqStaleReason(appState.bands.find((x) => x.id === b.id)!)).toMatch(/наклон/);
  });
});
