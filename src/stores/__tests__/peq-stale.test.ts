/**
 * b141.22 (audit HIGH): peqStale must notice everything the fit depended on.
 *
 * Two gaps: the optimizer has honoured the export sample rate since b141.5
 * (bilinear warping moves a band several dB near Nyquist), but the snapshot
 * did not record it — optimize at 48 kHz, switch export to 96 kHz, and the
 * shipped biquads no longer match the fit with nothing to say so, across a
 * reload since export_sample_rate persists. And filterEquals compared only
 * type/order/freq/shape/q, so toggling linear phase or subsonic protect —
 * both of which move the target the PEQ was fitted against — read as "no
 * change".
 */
import { describe, it, expect, beforeEach, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => undefined) }));

import {
  appState, addBand, setExportSampleRate, setBandPeqBands,
  setBandHighPass, setBandPeqOptimizedTarget,
} from "../bands";
import { captureOptimizedTarget, peqStale } from "../peq-optimize";
import type { FilterConfig } from "../../lib/types";

const HP: FilterConfig = {
  filter_type: "LinkwitzRiley", order: 4, freq_hz: 80,
  shape: null, linear_phase: false, q: null, subsonic_protect: null,
};

function seedBand() {
  addBand();                       // addBand() returns void; it activates the new band
  const id = appState.activeBandId;
  setBandHighPass(id, { ...HP });
  setBandPeqBands(id, [
    { freq_hz: 100, gain_db: -3, q: 2, enabled: true, filter_type: "Peaking" },
  ]);
  const band = () => appState.bands.find(b => b.id === id)!;
  return { id, band };
}

describe("peqStale (b141.22)", () => {
  beforeEach(() => setExportSampleRate(48_000));

  it("fresh right after capture", () => {
    const { id, band } = seedBand();
    setBandPeqOptimizedTarget(id, captureOptimizedTarget(band()));
    expect(peqStale(band())).toBe(false);
  });

  it("stale when the export sample rate moved away from the fitted one", () => {
    const { id, band } = seedBand();
    setBandPeqOptimizedTarget(id, captureOptimizedTarget(band()));
    setExportSampleRate(96_000);
    expect(peqStale(band())).toBe(true);
  });

  it("a pre-b141.22 snapshot without a rate is not stale on that ground", () => {
    const { id, band } = seedBand();
    const snap = captureOptimizedTarget(band());
    delete (snap as { sample_rate?: number }).sample_rate;
    setBandPeqOptimizedTarget(id, snap);
    setExportSampleRate(96_000);
    expect(peqStale(band())).toBe(false);
  });

  it("stale when the crossover switches to linear phase", () => {
    const { id, band } = seedBand();
    setBandPeqOptimizedTarget(id, captureOptimizedTarget(band()));
    setBandHighPass(id, { ...HP, linear_phase: true });
    expect(peqStale(band())).toBe(true);
  });

  it("stale when subsonic protect is toggled", () => {
    const { id, band } = seedBand();
    setBandPeqOptimizedTarget(id, captureOptimizedTarget(band()));
    setBandHighPass(id, { ...HP, subsonic_protect: true });
    expect(peqStale(band())).toBe(true);
  });
});
