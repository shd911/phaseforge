/** b141.103 (audit 2026-10-23 3.3): the WAV header rate, taps and file name
 *  come from the settings read before the FIR generation's await. */
import { describe, it, expect, vi } from "vitest";

const wavCalls: any[] = [];
let duringFir: (() => void) | null = null;
let firReq: any = null;

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (cmd: string, args: any) => {
    if (cmd === "export_fir_wav") { wavCalls.push(args); return null; }
    return null;
  }),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({
  save: vi.fn(async (o: any) => `/out/${o.defaultPath}`),
  open: vi.fn(),
}));
vi.mock("../../components/MeasurementAnalysisDialog", () => ({ openMeasurementAnalysis: vi.fn() }));
vi.mock("../band-evaluator", () => ({
  evaluateBandFull: vi.fn(async (req: any) => {
    firReq = req;
    duringFir?.();
    return { fir: { impulse: new Array(req.fir.taps).fill(0), wavDelaySamples: req.fir.taps / 2, route: "cepstral", peakBoostDb: 0, warnings: [] } };
  }),
}));

import { appState, setExportSampleRate, setExportTaps, setExportWindow } from "../../stores/bands";
import { exportBandWav } from "../fir-export";

describe("WAV export settings snapshot (3.3)", () => {
  it("an SR/taps/window switch during generation does not reach the file", async () => {
    setExportSampleRate(48000);
    setExportTaps(65536);
    setExportWindow("Blackman");
    duringFir = () => { setExportSampleRate(384000); setExportTaps(131072); setExportWindow("Hann"); };
    const ok = await exportBandWav(appState.bands[0]);
    expect(ok).toBe(true);
    expect(firReq.fir.sampleRate).toBe(48000);
    expect(wavCalls[0].sampleRate).toBe(48000);
    expect(wavCalls[0].path).toMatch(/_48000_65536_Blackman\.wav$/);
  });
});
