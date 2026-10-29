// b141.119 (UI audit 2026-10-29 §3, §4): the energy-after-peak share is
// graded only where ~100 % is expected; the GD span carries no traffic light.
import { describe, it, expect, afterEach } from "vitest";
import { render } from "solid-js/web";
import ExportMetricsBar from "../ExportMetricsBar";
import type { ExportMetrics } from "../../stores/bands";

let dispose: (() => void) | undefined;
afterEach(() => { dispose?.(); dispose = undefined; document.body.innerHTML = ""; });

const metrics = (expectCausal: boolean): ExportMetrics => ({
  taps: 65536, sampleRate: 48000, window: "Blackman", phaseLabel: "Linear-Phase",
  peqCount: 0, normDb: 0, causality: 50, expectCausal,
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
  it("50 % after the peak is neutral on linear phase", () => {
    expect(label(metrics(false), "После пика").style.color).toBe("");
  });
  it("50 % after the peak is graded on min phase", () => {
    expect(label(metrics(true), "После пика").style.color).not.toBe("");
  });
  it("the GD span has no traffic light", () => {
    expect(label(metrics(true), "Размах ГЗ").style.color).toBe("");
  });
});
