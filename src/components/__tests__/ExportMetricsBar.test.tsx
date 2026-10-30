// b141.123 (external audit 2026-10-29): the energy-after-peak share carries no
// traffic light, a correct min-phase LR4 band reads 42–72 %. b141.119: nor does
// the GD span.
import { describe, it, expect, afterEach } from "vitest";
import { render } from "solid-js/web";
import ExportMetricsBar from "../ExportMetricsBar";
import type { ExportMetrics } from "../../stores/bands";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

const metrics = (causality: number): ExportMetrics => ({
  taps: 65536, sampleRate: 48000, window: "Blackman", phaseLabel: "Linear-Phase",
  peqCount: 0, normDb: 0, causality,
  preRingMs: 0, maxMagErr: 0.1, gdRippleMs: 12,
  preRingThresholdDb: -60, preRingLimited: false, floorDb: -120,
  ringZoneMs: 10, ringZoneHz: 100, ringZoneLinear: true,
  passbandLoHz: 120, passbandHiHz: 16000, ultrasonicLpHz: null,
});

function label(m: ExportMetrics, text: string): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  dispose = render(() => <ExportMetricsBar m={m} />, host);
  const el = [...host.querySelectorAll<HTMLElement>("[role=button]")].find((e) => e.textContent!.startsWith(text));
  expect(el, text).toBeTruthy();
  return el!;
}

describe("ExportMetricsBar", () => {
  it.each([42, 51, 72, 100])("%i %% after the peak is neutral", (c) => {
    expect(label(metrics(c), "После пика").style.color).toBe("");
  });
  it("the GD span has no traffic light", () => {
    expect(label(metrics(50), "Размах ГЗ").style.color).toBe("");
  });
});
